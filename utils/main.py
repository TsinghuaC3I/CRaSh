#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import math
import os
import re
from copy import deepcopy
import logging
import random
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import torch
from tqdm import tqdm, trange
from tqdm.auto import tqdm as auto_tqdm
import srsly
import pandas as pd
from datasets import load_from_disk, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, set_seed
from transformers import OPTForCausalLM, GPT2LMHeadModel, BloomForCausalLM, LlamaForCausalLM

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from src.data_utils import get_raw_datasets, process_text2text_datasets, get_tokenized_datasets, get_lm_datasets
from src.task_utils import task_dict
from src.args_utils import parse_args
from src.model_utils import get_layers, get_student_by_choose, set_layers, load_adapter, load_student, add_prologue

from similarity_metrics import CudaCKA, layer_similarity
from visualize_utils import visualize_heatmap, visualize_plot

try:
    from tuned_lens import TunedLens
    from tuned_lens.plotting import PredictionTrajectory
except:
    pass
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_student(model, student, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[:
                                                                l] + student + model.model.decoder.layers[
                                                                    r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            student + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            student + model.transformer.h[r:]
    else:
        raise NotImplementedError


def load_dataloader(model, accelerator, args):
    # Get layer representations of the model on the validation set
    # Visualize the similarity matrix
    if "llama" in args.model_name_or_path:
        args.use_slow_tokenizer = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if isinstance(model, LlamaForCausalLM):
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        # tokenizer.padding_side = "left"  # Allow batched inference
    if args.dataset_name in task_dict:
        raw_datasets = get_raw_datasets(args)
        lm_datasets = process_text2text_datasets(raw_datasets, args, tokenizer,
                                                 accelerator)
    else:
        if args.train_tokenized_dataset and args.val_tokenized_dataset:
            # tokenized_datasets = load_from_disk(args.train_tokenized_dataset)
            train_dataset = load_from_disk(args.train_tokenized_dataset)
            val_dataset = load_from_disk(args.val_tokenized_dataset)
            # switch validation set from the pile to wikitext
            # if 'validation' in val_dataset:
            #     tokenized_datasets["validation"] = val_dataset['validation']
            # else:
            #     tokenized_datasets["validation"] = val_dataset['train']
            tokenized_datasets = DatasetDict({
                "train": train_dataset,
                "validation": val_dataset
            })
        else:
            raw_datasets = get_raw_datasets(args)

            tokenized_datasets = get_tokenized_datasets(
                raw_datasets, args, accelerator, tokenizer)

        lm_datasets = get_lm_datasets(tokenized_datasets, args, accelerator,
                                      tokenizer)

    val_dataset = lm_datasets["validation"]
    if args.validation_num_samples is not None:
        # check if we have enough samples for the validation set
        if args.validation_num_samples > len(val_dataset):
            args.validation_num_samples = len(val_dataset)
        val_dataset = val_dataset.select(range(args.validation_num_samples))

    # train_dataloader = DataLoader(train_dataset,
    #                               shuffle=True,
    #                               collate_fn=collator,
    #                               batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(val_dataset,
                                 collate_fn=default_data_collator,
                                 batch_size=args.per_device_eval_batch_size)

    for index in random.sample(range(len(val_dataset)), 3):
        print(f"Sample {index} of the training set: {val_dataset[index]}.")
    return eval_dataloader


def pool_hidden_states(pooling, hidden_states, batch):
    # hidden_states = [hs.to("cpu") for hs in hidden_states]
    # (num_layer, batch_size, seq_len, hidden_size)
    # special case for opt/opt-350m, which word embedding size is 512 but hidden size is 1024
    if hidden_states[-1].shape[-1] != hidden_states[-2].shape[-1]:
        hidden_states = hidden_states[:-1]
    if pooling == "mean":
        res = []
        for hs in hidden_states:
            mask_expanded = batch["attention_mask"].unsqueeze(-1).expand(
                hs.size()).float()
            sum_hs = torch.sum(hs * mask_expanded, 1)

            # smoothing, avoid being divided by zero
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            res.append(sum_hs / sum_mask)
        return res
        # return [hs.mean(dim=1).float() for hs in hidden_states]
    elif pooling == "mean_all":
        return [hs.mean(dim=1).float() for hs in hidden_states]
    elif pooling == "max":
        return [hs.max(dim=1).values.float() for hs in hidden_states]
    elif pooling == "min":
        return [hs.min(dim=1).values.float() for hs in hidden_states]
    elif pooling == "last":
        # we use the last token as the representation accordding to the mask
        attention_mask = batch["attention_mask"]
        last_index = attention_mask.sum(dim=1) - 1
        # last_index = last_index.unsqueeze(1)[:, :, None].expand(
        last_index = last_index.unsqueeze(1).unsqueeze(2).expand(
            -1, -1, hidden_states[0].shape[-1])
        # print("last_index", last_index.shape)
        last_size = None
        res = []
        for lid, hs in enumerate(hidden_states):
            # print(lid, hs.shape)
            # special case for opt/opt-350m, which word embedding size is 512 but hidden size is 1024
            if last_size is None:
                last_size = hs.shape[-1]
            elif last_size != hs.shape[-1]:
                continue
            res.append(torch.gather(hs.float(), 1, last_index).squeeze())
        return res
    elif pooling == "max_nonpad":
        attention_mask = batch["attention_mask"]
        pooled_representation = []
        for hs in hidden_states:
            # Set masked positions to a very small value
            hs_masked = hs.masked_fill(
                attention_mask.unsqueeze(-1) == 0, float('-inf'))
            # Apply max pooling along the sequence dimension
            max_pooling = torch.max(hs_masked, dim=1).values.float()
            pooled_representation.append(max_pooling)
        return pooled_representation
    elif pooling == "mean_weighted":
        # refer to https://github.com/Muennighoff/sgpt/blob/main/other/sgpt_utils.ipynb
        # last_hidden_state = hidden_states[-1]  # Get hidden states of the last layer
        all_embeddings = []
        for layer_hidden_state in hidden_states:
            # Get weights of shape [bs, seq_len, hid_dim]
            weights = (torch.arange(start=1,
                                    end=layer_hidden_state.shape[1] +
                                    1).unsqueeze(0).unsqueeze(-1).expand(
                                        layer_hidden_state.size()).float().to(
                                            layer_hidden_state.device))

            # Get attn mask of shape [bs, seq_len, hid_dim]
            input_mask_expanded = (
                batch["attention_mask"].unsqueeze(-1).expand(
                    layer_hidden_state.size()).float())

            # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
            sum_embeddings = torch.sum(layer_hidden_state *
                                       input_mask_expanded * weights,
                                       dim=1)
            sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

            embeddings = sum_embeddings / sum_mask
            all_embeddings.append(embeddings)
        return all_embeddings
    else:
        # without pooling
        return hidden_states


def data_driven_similarity(model, accelerator, args):
    eval_dataloader = load_dataloader(model, accelerator, args)

    another_model = None
    if args.another_model_name_or_path:
        print("load another model", args.another_model_name_or_path)
        another_model = AutoModelForCausalLM.from_pretrained(
            args.another_model_name_or_path, torch_dtype=torch.float16)
        another_model.to("cuda")

        if args.load_interp_adapter:
            args.student_l_pad = args.interp_student_l_pad
            args.student_r_pad = args.interp_student_r_pad
            interp_adapter_state_dict = torch.load(args.load_interp_adapter,
                                                   map_location='cpu')
            another_model = load_adapter(another_model,
                                         interp_adapter_state_dict, args)

        if args.load_interp_student:
            args.student_l_pad = args.interp_student_l_pad
            args.student_r_pad = args.interp_student_r_pad
            interp_student_state_dict = torch.load(args.load_interp_student,
                                                   map_location='cpu')
            another_model = load_student(another_model,
                                         interp_student_state_dict, args)

    model.eval()
    losses = []
    hidden_states = []
    another_hidden_states = []
    cnt = 0
    if args.split_layer_by_layer:
        # set prologue function to store embedding layer outputs for each layer
        layers = get_layers(model)
        add_prologue(layers[0], None)
        set_layers(model, torch.nn.ModuleList([layers[0]]))
        print("reset memory", model.get_memory_footprint())
        model.to("cuda")

    for batch in tqdm(eval_dataloader, desc="forward"):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)

            if args.split_layer_by_layer:
                layer_kwargs = layers[0].input_kwargs
                layer_args = layers[0].input_args
                layer_args = tuple(list(layer_args[1:]))

                # embedding hidden states
                tmp_hidden_states = outputs.hidden_states[0]
                output_hidden_states = [tmp_hidden_states.to("cpu")]
                for layer in tqdm(layers):
                    layer.to("cuda")
                    layer_outputs = layer(tmp_hidden_states, *layer_args,
                                          **layer_kwargs)
                    if isinstance(layer_outputs, tuple):
                        tmp_hidden_states = layer_outputs[0]
                        output_hidden_states.append(
                            tmp_hidden_states.to("cpu"))

                    layer.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                # prepare for model
                layers[0] = layers[0].to("cuda")
                batch = {k: v.to("cpu") for k, v in batch.items()}

            if another_model is not None:
                another_outputs = another_model(**batch,
                                                output_hidden_states=True)
        tmp = torch.stack(
            pool_hidden_states(
                args.pooling, output_hidden_states if args.split_layer_by_layer
                else outputs.hidden_states, batch))
        if args.swap_to_cpu:
            tmp = tmp.to("cpu")
        hidden_states.append(tmp)
        if another_model is not None:
            tmp = torch.stack(
                pool_hidden_states(args.pooling, another_outputs.hidden_states,
                                   batch))
            if args.swap_to_cpu:
                tmp = tmp.to("cpu")
            another_hidden_states.append(tmp)
        # loss = outputs.loss
        # losses.append(
        #     accelerator.gather_for_metrics(
        #         loss.repeat(args.per_device_eval_batch_size)).cpu())

        cnt += 1
        if args.cka_minibatch > 0 and cnt >= args.cka_minibatch:
            break

        gc.collect()
        torch.cuda.empty_cache()

    hidden_states = torch.cat(hidden_states, 1)
    if another_model is not None:
        another_hidden_states = torch.cat(another_hidden_states, 1)

    if args.swap_to_cpu:
        model.to("cpu")
        hidden_states = hidden_states.to("cuda")
        if another_model is not None:
            another_hidden_states = another_hidden_states.to("cuda")

    # losses = torch.cat(losses).flatten()
    # # filter out nan
    # losses = losses[~torch.isnan(losses)]
    # try:
    #     eval_loss = torch.mean(losses)
    #     perplexity = math.exp(eval_loss)
    # except OverflowError:
    #     perplexity = float("inf")

    # print(eval_loss, perplexity)
    cuda_cka = CudaCKA("cuda")
    if another_model is None:
        print(
            "another_model is None, so we use hidden_states as another_hidden_states and compare with itself"
        )
        another_hidden_states = hidden_states
    else:
        print("another_model is not None, so we compare hidden_states with it")
    # print(len(another_hidden_states), len(hidden_states))
    logger.info(
        f"xlabel = {args.xlabel}, ylabel = {args.ylabel}, another_hidden_states = {len(another_hidden_states)}, hidden_states = {len(hidden_states)}"
    )
    # print(another_hidden_states[40])
    layer2layer_cka = {"layer": list(range(len(another_hidden_states)))}
    for i, hs in enumerate(another_hidden_states):
        for j, hsp in enumerate(hidden_states):
            cka_score = cuda_cka.linear_CKA(hs, hsp).item()
            # print(i, j, cka_score)
            if str(j) not in layer2layer_cka:
                layer2layer_cka[str(j)] = []
            layer2layer_cka[str(j)].append(cka_score)

    prepare_data_for_heatmap(layer2layer_cka)


def weight_similarity(model, args):
    pattern = r"layers.(\d+).fc2.weight"
    linear_weights = []
    layers = []
    for name, param in model.named_parameters():
        match = re.search(pattern, name)
        if match:
            layer = int(match.group(1))
            layers.append(layer)
            linear_weights.append(param)

    layer2layer_l2 = {"layer": layers}
    for i, w in enumerate(linear_weights):
        for j, wp in enumerate(linear_weights):
            l2_score = torch.norm(w - wp)
            if str(j) not in layer2layer_l2:
                layer2layer_l2[str(j)] = []
            layer2layer_l2[str(j)].append(l2_score.item())

    prepare_data_for_heatmap(layer2layer_l2)


def layer_var(model, args):
    pattern = r"layers.(\d+).fc2.weight"
    linear_variance = []
    layers = []
    for name, param in model.named_parameters():
        match = re.search(pattern, name)
        if match:
            layer = int(match.group(1))
            layers.append(layer)
            linear_variance.append(torch.var(param).cpu().item())
    df = pd.DataFrame(data={"variance": linear_variance},
                      columns=["variance"],
                      index=layers)

    visualize_plot(df,
                   fname=args.additional_note,
                   xlabel=args.xlabel,
                   ylabel=args.ylabel,
                   title=args.title)


def prepare_data_for_heatmap(layer2layer):
    columns = list(layer2layer.keys())
    columns.remove("layer")
    l2_df = pd.DataFrame(data=layer2layer,
                         columns=columns,
                         index=layer2layer["layer"])

    visualize_heatmap(l2_df,
                      fname=args.additional_note,
                      xlabel=args.xlabel,
                      ylabel=args.ylabel,
                      title=args.title,
                      fig_dir=args.fig_dir)


def get_forward_outputs(args,
                        model,
                        layers,
                        eval_dataloader,
                        forward_type=None):
    model.eval()
    hidden_states = []
    cnt = 0
    # set prologue function to store embedding layer outputs for each layer
    if args.visual_forward_type == "single":
        add_prologue(layers[0], None)

    for batch in tqdm(eval_dataloader, desc="forward"):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True).hidden_states

            # here we only use the embedding layer outputs as the inputs of the next each layer
            if forward_type == "single":
                layer_kwargs = layers[0].input_kwargs
                layer_args = layers[0].input_args
                layer_args = tuple(list(layer_args[1:]))

                emb_outputs = outputs[0]
                new_outputs = [emb_outputs]
                for layer in layers:
                    layer_outputs = layer(emb_outputs, *layer_args,
                                          **layer_kwargs)
                    if isinstance(layer_outputs, tuple):
                        new_outputs.append(layer_outputs[0])
                outputs = new_outputs

        # batch = {k: v.to("cpu") for k, v in batch.items()}
        tmp = torch.stack(pool_hidden_states(args.pooling, outputs, batch))
        if args.swap_to_cpu:
            tmp = tmp.to("cpu")
        hidden_states.append(tmp)

        # loss = outputs.loss
        # losses.append(
        #     accelerator.gather_for_metrics(
        #         loss.repeat(args.per_device_eval_batch_size)).cpu())

        cnt += 1
        if args.cka_minibatch > 0 and cnt >= args.cka_minibatch:
            break

        gc.collect()
        torch.cuda.empty_cache()

    hidden_states = torch.cat(hidden_states, 1)

    if args.swap_to_cpu:
        model.to("cpu")
        hidden_states = hidden_states.to("cuda")
    return hidden_states


def layer_iter_drop(model, args):
    eval_dataloader = load_dataloader(model, accelerator, args)
    # compute cka on adjacent layer
    layers = get_layers(model)

    # set orig index for remove and review
    for idx, layer in enumerate(layers):
        layer.orig_index = idx

    while len(layers) > args.num_keep_layers:
        hidden_states = get_forward_outputs(args, model, layers,
                                            eval_dataloader)
        cuda_cka = CudaCKA("cuda")
        hidden_states = hidden_states[1:]
        assert len(hidden_states) == len(layers)
        layer2cka = []
        for i in range(len(hidden_states) - 1):
            cka_score = cuda_cka.linear_CKA(hidden_states[i],
                                            hidden_states[i + 1])
            layer2cka.append(cka_score.item())
        # delete rightmost layer
        rmv_idx = torch.argmax(torch.tensor(layer2cka)).item() + 1

        print(layer2cka)
        print("remove layer %d" % layers[rmv_idx].orig_index)
        del layers[rmv_idx]
        set_layers(model, layers)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("dataset", args.dataset_name)
    print("keep_layer_index", [l.orig_index for l in layers])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # layer2layer_cka = {"layer": list(range(len(hidden_states)))}
    # for i, hs in enumerate(hidden_states):
    #     for j, hsp in enumerate(hidden_states):
    #         cka_score = cuda_cka.linear_CKA(hs, hsp).item()
    #         if str(j) not in layer2layer_cka:
    #             layer2layer_cka[str(j)] = []
    #         layer2layer_cka[str(j)].append(cka_score)


def layer_iter_cluster(model, args):
    eval_dataloader = load_dataloader(model, accelerator, args)
    # compute cka on adjacent layer
    layers = get_layers(model)

    hidden_states = get_forward_outputs(args,
                                        model,
                                        layers,
                                        eval_dataloader,
                                        forward_type=args.visual_forward_type)
    hidden_states = hidden_states[1:]

    cuda_cka = CudaCKA("cuda")

    num_layers = len(layers)
    assert len(
        hidden_states) == num_layers, "hidden_states: %d, layers: %d" % (
            len(hidden_states), num_layers)

    cluster_process = []
    if args.visual_forward_type != "single":
        clusters = [[i] for i in range(num_layers)]
        # linkage_matrix = []
        while len(clusters) > args.num_layer_cluster:
            max_distance = -np.inf
            merge_indices = None

            # find the closest pair clusters to merge
            for i in range(len(clusters) - 1):
                dist = cuda_cka.linear_CKA(
                    hidden_states[clusters[i][-1]],
                    hidden_states[clusters[i + 1][0]]).item()
                if dist > max_distance:
                    max_distance = dist
                    merge_indices = (i, i + 1)

            print("merge loss", max_distance, clusters[merge_indices[0]],
                  clusters[merge_indices[1]])
            cluster_process.append({
                "distance": max_distance,
                "left_cluster": clusters[merge_indices[0]],
                "right_cluster": clusters[merge_indices[1]]
            })
            clusters[merge_indices[0]].extend(clusters[merge_indices[1]])
            del clusters[merge_indices[1]]
    elif args.visual_forward_type == "single":
        ######### hierarchical clustering via cka metric ##########
        similarity_matrix = np.array([[0.0 for _ in range(num_layers)]
                                      for _ in range(num_layers)])
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                cka_score = cuda_cka.linear_CKA(hidden_states[i],
                                                hidden_states[j])
                similarity_matrix[i][j] = cka_score.item()
                similarity_matrix[j][i] = cka_score.item()

        linkage_matrix = linkage(similarity_matrix,
                                 method='single')  # single, complete, average

        cluster_labels = fcluster(linkage_matrix,
                                  args.num_layer_cluster,
                                  criterion='maxclust')

        clusters = [[] for _ in range(args.num_layer_cluster)]
        for layer_index, cluster_label in enumerate(cluster_labels):
            clusters[cluster_label - 1].append(layer_index)

    # # visualize clustering result
    # labels = np.arange(num_layers)
    # cluster_index_map = {i: i for i in range(num_layers)}
    # for i, cluster in enumerate(clusters):
    #     for layer_index in cluster:
    #         cluster_index_map[layer_index] = num_layers + i

    # labels = np.array([cluster_index_map[i] for i in labels])

        dendrogram(linkage_matrix)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.title('Hierarchical Clustering Dendrogram')
        plt.savefig(
            f"../figs_empir/layer_cluster_cka_pair_single_forward_{args.dataset_name}_{args.num_layer_cluster}.pdf",
            format="pdf")

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("dataset", args.dataset_name, "num_layer_cluster",
          args.num_layer_cluster)
    print(clusters)

    cluster_centers = []
    for cluster in clusters:
        if len(cluster) == 1:
            cluster_centers.append(cluster[0])
        else:
            cluster_centers.append(cluster[len(cluster) // 2])

    metadata = {
        "dataset": args.dataset_name,
        "num_layer_cluster": args.num_layer_cluster,
        "clusters": clusters,
        "cluster_centers": cluster_centers,
        "cluster_process": cluster_process
    }
    srsly.write_json(
        path=
        f"./metadata/layer_cluster/{args.dataset_name}_{args.num_layer_cluster}_adjacent_layer_stacking_{args.additional_note}.json",
        data=metadata)


def pair_layer_replace(model, args):
    """Compare the functional similarity between pair of layers
    Goal: similarity matrix (L, L)
        where L is the number of layers, (i,j) is performance compared to original model while replacing layer i with layer j
    """
    eval_dataloader = load_dataloader(model, accelerator, args)
    # compute cka on adjacent layer
    layers = get_layers(model)

    hidden_states = get_forward_outputs(args, model, layers, eval_dataloader)
    hidden_states = hidden_states[1:]

    cuda_cka = CudaCKA("cuda")

    num_layers = len(layers)
    layer2layer_cka = {"layer": list(range(num_layers))}
    progress_bar = auto_tqdm(range(num_layers * num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            tmp_layers = deepcopy(layers)
            tmp_layers[i] = layers[j]
            set_layers(model, tmp_layers)
            tmp_hidden_states = get_forward_outputs(args, model, tmp_layers,
                                                    eval_dataloader)

            cka_score = cuda_cka.linear_CKA(hidden_states[-1],
                                            tmp_hidden_states[-1]).item()
            if str(j) not in layer2layer_cka:
                layer2layer_cka[str(j)] = []
            layer2layer_cka[str(j)].append(cka_score)
            progress_bar.update(1)
            gc.collect()
            torch.cuda.empty_cache()

    prepare_data_for_heatmap(layer2layer_cka)


def logits_len(model, args):
    """We get intermediate outputs using logits-len method
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if isinstance(model, LlamaForCausalLM):
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    prompt = "We’ve created GPT-4, the latest milestone in OpenAI’s effort in scaling up deep learning."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    print(tokenizer.tokenize(prompt))
    layers = get_layers(model)
    mode = "full"
    is_softmax = False
    if mode == "oft":
        # oft
        ids = [0, 1, 2, 5, 7, 10, 13, 16, 18, 21, 22, 23]
    elif mode == "crash":
        # crash
        ids = [0, 1, 2, 6, 10, 14, 17, 18, 19, 20, 22, 23]

    if mode in ["oft", "crash"]:
        new_layers = torch.nn.ModuleList([layers[i] for i in ids])
        set_layers(model, new_layers)

    model.to("cuda")
    model.eval()

    layer2token = []
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True).hidden_states

        for layer, hidden_state in enumerate(outputs):
            # (N, L, H) -> (N, L, V)
            # print(hidden_state.shape)
            with torch.no_grad():
                output_logits = model.lm_head(hidden_state)
                output_token_ids = torch.argmax(output_logits, dim=-1)
            logits = torch.max(torch.softmax(output_logits, dim=-1)
                               if is_softmax else output_logits,
                               dim=-1).values.detach().cpu().tolist()[0]
            output_tokens = [
                tokenizer.convert_ids_to_tokens(output_token_id)
                for output_token_id in output_token_ids
            ]
            layer2token.append((layer, output_tokens, logits))
            print(layer)
            for output_token in output_tokens:
                print(output_token)
            print()

    srsly.write_json(
        f"./metadata/logits_len_{mode}{'_sfx' if is_softmax else '_wosfx'}.json",
        {"metadata": layer2token})


def evaluate_layer_importance(model, args):
    eval_dataloader = load_dataloader(model, accelerator, args)
    # compute cka on adjacent layer
    layers = get_layers(model)

    orig_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16)
    orig_layers = get_layers(orig_model)

    def eval_model(model):
        model.eval()
        losses = []
        for _, batch in enumerate(eval_dataloader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)).cpu())
        losses = torch.cat(losses).flatten()
        # filter out nan
        losses = losses[~torch.isnan(losses)]
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        return eval_loss, perplexity

    model.to("cuda")
    orig_layers.to("cuda")
    layer2loss = {"delete": [], "rewind": []}

    set_layers(model, layers)
    loss, ppl = eval_model(model)
    layer2loss["finetuned"] = (float(loss.detach().cpu()), ppl)

    set_layers(model, orig_layers)
    loss, ppl = eval_model(model)
    layer2loss["original"] = (float(loss.detach().cpu()), ppl)

    for i in trange(len(layers)):
        # evaluate by deleting
        set_layers(model, layers[:i] + layers[i + 1:])
        loss, ppl = eval_model(model)
        loss = float(loss.detach().cpu())
        layer2loss["delete"].append((i, loss, ppl))

        # evaluate by rewinding
        set_layers(model, layers[:i] + orig_layers[i:i + 1] + layers[i + 1:])
        loss, ppl = eval_model(model)
        loss = float(loss.detach().cpu())
        layer2loss["rewind"].append((i, loss, ppl))

    srsly.write_json(
        "./metadata/layer_importance-1.3b/%s.json" % args.dataset_name,
        layer2loss)


def datasets_cluster(model, args):
    tasks = args.datasets_to_cluster.split(",")
    task2dataloader = {}
    for task in tasks:
        args.dataset_name = task
        eval_dataloader = load_dataloader(model, accelerator, args)
        task2dataloader[task] = eval_dataloader

    model.eval()

    encoded_samples = None
    dataset_labels = []
    encoded_samples_for_cka = []
    cka_index = []
    for task, eval_dataloader in task2dataloader.items():
        print("processing %s" % task)
        hidden_states = []
        cnt = 0
        for batch in tqdm(eval_dataloader, desc="forward"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)

            tmp = torch.stack(
                pool_hidden_states(args.pooling, outputs.hidden_states, batch))

            if args.swap_to_cpu:
                tmp = tmp.to("cpu")

            hidden_states.append(tmp)

            cnt += 1
            if args.cka_minibatch > 0 and cnt >= args.cka_minibatch:
                break

            gc.collect()
            torch.cuda.empty_cache()

        hidden_states = torch.cat(hidden_states, 1)[-1].detach()

        encoded_samples_for_cka.append(hidden_states)
        cka_index.append(task)

        # get the last layer
        last_hidden_states = deepcopy(hidden_states).cpu().numpy()
        if encoded_samples is None:
            encoded_samples = last_hidden_states
        else:
            encoded_samples = np.concatenate(
                (encoded_samples, last_hidden_states), axis=0)
        dataset_labels.extend([task] * len(last_hidden_states))

    ### CKA
    cuda_cka = CudaCKA("cuda")
    # print(another_hidden_states[40])
    layer2layer_cka = {"layer": cka_index}
    for i, hs in enumerate(encoded_samples_for_cka):
        for j, hsp in enumerate(encoded_samples_for_cka):
            cka_score = cuda_cka.linear_CKA(hs, hsp).item()
            print(cka_index[i], cka_index[j], cka_score)
            key = cka_index[j]
            if key not in layer2layer_cka:
                layer2layer_cka[key] = []
            layer2layer_cka[key].append(cka_score)

    prepare_data_for_heatmap(layer2layer_cka)

    plt.clf()
    ### t-SNE
    print(encoded_samples.shape)
    assert len(encoded_samples) == len(dataset_labels)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(encoded_samples)

    df = pd.DataFrame(tsne_embeddings, columns=['Dimension 1', 'Dimension 2'])

    df['Dataset'] = dataset_labels

    # 使用seaborn进行可视化
    sns.scatterplot(data=df,
                    x='Dimension 1',
                    y='Dimension 2',
                    hue='Dataset',
                    palette='Set1')

    if not os.path.exists(f"{args.fig_dir}/metadata"):
        os.makedirs(f"{args.fig_dir}/metadata")
    df.to_csv(
        f"{args.fig_dir}/metadata/{args.additional_note}_scatterplot.csv")

    plt.savefig(f'{args.fig_dir}/{args.additional_note}_scatterplot.pdf',
                format='pdf')


def show_datasets(model, args):
    tasks = args.datasets_to_cluster.split(",")
    if "llama" in args.model_name_or_path:
        args.use_slow_tokenizer = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    if isinstance(model, LlamaForCausalLM):
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        # tokenizer.padding_side = "left"  # Allow batched inference

    for task in tasks:
        args.dataset_name = task
        cur_task = task_dict[args.dataset_name]

        print("\n\n##################################")
        print("processing %s" % task)
        print("##################################")

        def tokenize_function(examples):
            context = cur_task.get_context(examples)
            target = cur_task.get_target(examples)
            for ctx, tgt in zip(context, target):
                print(ctx)
                print(tgt)
                print()
                print()

        if args.dataset_name in task_dict:
            raw_datasets = get_raw_datasets(args)
            val_dataset = raw_datasets["validation"]
            val_dataset.select(range(args.validation_num_samples)).map(
                tokenize_function,
                batched=True,
            )


def tuned_lens_similarity(model, accelerator, args):
    # Get layer representations of the model on the validation set
    # Visualize the similarity matrix
    if "llama" in args.model_name_or_path:
        args.use_slow_tokenizer = True

    print("load tuned lens")
    directory_path = "/root/kyzhang/studio/transfer_llm/logs/tuned_lens/opt-1.3b"
    tuned_lens = TunedLens.from_model_and_pretrained(model, directory_path)
    tuned_lens = tuned_lens.half()
    tuned_lens.to(model.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if isinstance(model, LlamaForCausalLM):
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        # tokenizer.padding_side = "left"  # Allow batched inference
    tasks = args.datasets_to_cluster.split(",")

    for task in tasks:
        args.dataset_name = task
        if args.dataset_name not in task_dict:
            pass
        else:
            cur_task = task_dict[args.dataset_name]

        print("\n\n##################################")
        print("processing %s" % task)
        print("##################################")

        def tokenize_function(examples):
            context = cur_task.get_context(examples)
            target = cur_task.get_target(examples)

            context = tokenizer(context,
                                padding=False,
                                truncation=True,
                                max_length=args.max_length)

            target = tokenizer(target,
                               padding=False,
                               truncation=True,
                               max_length=args.max_length)

            # if target is starting with special token, remove it
            if len(target['input_ids'][0]) > 0 and target['input_ids'][0][
                    0] in tokenizer.all_special_ids:
                target['input_ids'] = [i[1:] for i in target['input_ids']]
                target['attention_mask'] = [
                    a[1:] for a in target['attention_mask']
                ]

            inputs = [
                ctx_inp + tgt_inp for ctx_inp, tgt_inp in zip(
                    context['input_ids'], target['input_ids'])
            ]
            outputs = [
                ctx_inp[1:] + tgt_inp + [tokenizer.eos_token_id] for ctx_inp,
                tgt_inp in zip(context['input_ids'], target['input_ids'])
            ]

            # note the start and end position of the target
            start_pos = [len(ctx_inp) for ctx_inp in context['input_ids']]
            end_pos = [len(inp) for inp in inputs]
            return {
                "input_ids":
                inputs,
                "targets":
                outputs,
                "start_pos":
                start_pos,
                "end_pos":
                end_pos,
                "attention_mask": [
                    ctx_mask + tgt_mask for ctx_mask, tgt_mask in zip(
                        context['attention_mask'], target['attention_mask'])
                ]
            }

        def tokenize_function_for_wikitext(examples):
            inputs = tokenizer(examples['text'],
                               padding="max_length",
                               truncation=True,
                               max_length=args.max_length)
            target = [
                input_ids[1:] + [tokenizer.eos_token_id]
                for input_ids in inputs['input_ids']
            ]
            start_pos = [0] * len(target)
            end_pos = [sum(mask) for mask in inputs['attention_mask']]
            return {
                "input_ids": inputs["input_ids"],
                "targets": target,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "attention_mask": inputs['attention_mask']
            }

        if args.dataset_name in task_dict:
            raw_datasets = get_raw_datasets(args)
            val_dataset = raw_datasets["validation"]
            val_dataset = val_dataset.select(
                range(min(len(val_dataset), args.validation_num_samples))).map(
                    tokenize_function,
                    batched=True,
                )
        elif args.dataset_name == "wikitext":
            val_dataset = load_dataset(
                "json",
                data_files={
                    "train":
                    f"/root/kyzhang/studio/transfer_llm/data/wikitext-103-v1-clean/train_{args.wikitext_range}.json"
                })["train"]
            val_dataset = val_dataset.select(
                range(min(len(val_dataset), args.validation_num_samples))).map(
                    tokenize_function_for_wikitext,
                    batched=True,
                )
        else:
            if args.train_tokenized_dataset and args.val_tokenized_dataset:
                # tokenized_datasets = load_from_disk(args.train_tokenized_dataset)
                train_dataset = load_from_disk(args.train_tokenized_dataset)
                val_dataset = load_from_disk(args.val_tokenized_dataset)
                # switch validation set from the pile to wikitext
                # if 'validation' in val_dataset:
                #     tokenized_datasets["validation"] = val_dataset['validation']
                # else:
                #     tokenized_datasets["validation"] = val_dataset['train']
                tokenized_datasets = DatasetDict({
                    "train": train_dataset,
                    "validation": val_dataset
                })
            else:
                raw_datasets = get_raw_datasets(args)

                tokenized_datasets = get_tokenized_datasets(
                    raw_datasets, args, accelerator, tokenizer)

            lm_datasets = get_lm_datasets(tokenized_datasets, args,
                                          accelerator, tokenizer)

            val_dataset = lm_datasets["validation"]
            if args.validation_num_samples is not None:
                # check if we have enough samples for the validation set
                if args.validation_num_samples > len(val_dataset):
                    args.validation_num_samples = len(val_dataset)
                val_dataset = val_dataset.select(
                    range(args.validation_num_samples))

        eval_dataloader = DataLoader(
            val_dataset,
            collate_fn=default_data_collator,
            batch_size=args.per_device_eval_batch_size)

        similarity = []
        for batch in tqdm(eval_dataloader, desc="Out"):
            # batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(input_ids=batch["input_ids"].to("cuda"),
                                attention_mask=batch["attention_mask"].to("cuda"),
                                output_hidden_states=True)

            # num_layers, batch_size, seq_len, hidden_size
            traj_log_probs = []
            progress_bar1 = auto_tqdm(range(len(outputs.hidden_states) - 1))
            for i, h in enumerate(outputs.hidden_states[:-1]):
                # h is of shape (batch_size, seq_len, hidden_size)
                logits = tuned_lens.forward(h, i)
                traj_log_probs.append(
                    logits.log_softmax(-1).squeeze().detach().cpu().numpy())
                progress_bar1.update(1)

            traj_log_probs = np.array(traj_log_probs)
            print(traj_log_probs.shape)
            progress_bar2 = auto_tqdm(range(len(batch)))
            for i, (start,
                    end) in enumerate(zip(batch["start_pos"],
                                          batch["end_pos"])):
                tmp_predictition_traj = [
                    traj_log_probs[j, i, start:end, :]
                    for j in range(len(traj_log_probs))
                ]
                logits = torch.tensor(np.array(tmp_predictition_traj)).to(
                    model.device)
                tmp_similarity = layer_similarity(logits,
                                                  stype=args.similarity_type)
                similarity.append(tmp_similarity)
                progress_bar2.update(1)

        # similarity = []
        # for sample in tqdm(val_dataset, desc=args.dataset_name):
        #     try:
        #         if args.dataset_name in task_dict or args.dataset_name == "wikitext":
        #             tmp_predictition_traj = PredictionTrajectory.from_lens_and_model(
        #                 tuned_lens,
        #                 model,
        #                 start_pos=sample["start_pos"],
        #                 end_pos=sample["end_pos"],
        #                 tokenizer=tokenizer,
        #                 input_ids=sample["input_ids"],
        #                 targets=sample["targets"],
        #             )

        #         else:
        #             tmp_predictition_traj = PredictionTrajectory.from_lens_and_model(
        #                 tuned_lens,
        #                 model,
        #                 start_pos=0,
        #                 end_pos=len(sample["input_ids"]),
        #                 tokenizer=tokenizer,
        #                 input_ids=sample["input_ids"],
        #                 targets=sample["labels"][1:] +
        #                 [tokenizer.eos_token_id],
        #             )

        #         logits = torch.tensor(tmp_predictition_traj.log_probs).to(
        #             model.device)
        #         tmp_similarity = layer_similarity(logits,
        #                                           stype=args.similarity_type)
        #         similarity.append(tmp_similarity)
        #     except:
        #         print("error")
        #         print(sample)

        similarity = sum(similarity) / len(similarity)
        print(similarity)
        with open(
                f'/root/kyzhang/studio/transfer_llm/test/metadata/layer_similarity/{args.dataset_name}-{args.similarity_type}-{args.additional_note}.npy',
                'wb') as f:
            np.save(f, similarity.cpu().numpy())

if __name__ == "__main__":
    args = parse_args()
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    accelerator = Accelerator()

    device_count = torch.cuda.device_count()
    if device_count > 1:
        max_memory_mapping = {
            idx: "48GB"
            for idx in range(torch.cuda.device_count())
        }
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            max_memory=max_memory_mapping,
            torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     torch_dtype=torch.float16)

    # uniform choose student layer for w/o kd
    if args.num_student_layers is not None:
        full_uniform = False
        layers = get_layers(model)
        if full_uniform:
            layers = get_student_by_choose(layers, args)
            set_layers(model, layers)
        else:
            # fix student layers: 2:8:2
            l, r = args.student_l_pad, len(layers) - args.student_r_pad
            print("fix student layers: %d:%d:%d" %
                  (l, args.num_student_layers, r))
            student = deepcopy(layers[l:r])
            layers = get_student_by_choose(student, args)
            to_student(model, layers, args)

    # load adapter layer for student-ft and plug-in
    if args.load_adapter:
        print("load_adapter from %s" % args.load_adapter)
        adapter_state_dict = torch.load(args.load_adapter, map_location='cpu')
        model = load_adapter(model, adapter_state_dict, args)

    # load student layer from kd for student-kd/ft
    if args.load_student:
        print("load_student from %s" % args.load_student)
        student_state_dict = torch.load(args.load_student, map_location='cpu')
        model = load_student(model, student_state_dict, args)

    if device_count == 1 and not args.split_layer_by_layer:
        model = model.to("cuda")

    if args.visual_type == "layer_cka":
        data_driven_similarity(model, accelerator, args)
    elif args.visual_type == "tuned_lens_similarity":
        tuned_lens_similarity(model, accelerator, args)
    elif args.visual_type == "weight_l2":
        weight_similarity(model, args)
    elif args.visual_type == "layer_var":
        layer_var(model, args)
    elif args.visual_type == "layer_iter_drop":
        layer_iter_drop(model, args)
    elif args.visual_type == "layer_iter_cluster":
        layer_iter_cluster(model, args)
    elif args.visual_type == "pair_layer_replace":
        pair_layer_replace(model, args)
    elif args.visual_type == "logits_len":
        logits_len(model, args)
    elif args.visual_type == "evaluate_layer_importance":
        evaluate_layer_importance(model, args)
    elif args.visual_type == "datasets_cluster":
        datasets_cluster(model, args)
    elif args.visual_type == "show_datasets":
        show_datasets(model, args)
    else:
        raise NotImplementedError(args.visual_type)
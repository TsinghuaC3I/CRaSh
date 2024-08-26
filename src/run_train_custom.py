#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import gc
import os
import random
import sys
import math
import json
import logging
import srsly
import datasets
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_from_disk, DatasetDict
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM,
                          AutoTokenizer, default_data_collator, get_scheduler,
                          LlamaForCausalLM)

from src.args_utils_custom import parse_args
from src.task_utils import task_dict
from src.data_utils import get_raw_datasets, get_tokenized_datasets, get_lm_datasets, process_text2text_datasets
from src.model_utils import (MLP, add_epilogue, add_prologue,
                             uniform_choose_layers, magnitude_prune, quantize,
                             setup_teacher_student, save_state_dict,
                             to_student, to_teacher,
                             compute_and_print_parameters, get_layers,
                             set_layers)

from src.peft_utils import (use_lora, use_bitfit, use_adapter)
from src.linear_decompose import find_linear_and_replace_with_svd_module

logger = get_logger(__name__)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    # also log to a file in output_dir
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        ] if accelerator.is_main_process else [])
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # TODO: fix
    if "llama" in args.model_name_or_path:
        args.use_slow_tokenizer = True

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            torch_dtype=torch.float16)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if isinstance(model, LlamaForCausalLM):
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        # tokenizer.padding_side = "left"  # Allow batched inference

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.dataset_name in task_dict:  # special case for e2e_nlg dataset
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
            # tokenized_datasets["train"].save_to_disk("data/wikitext-2-raw-v1/train")
            # tokenized_datasets["validation"].save_to_disk(
            #     "data/wikitext-2-raw-v1/validation")

        lm_datasets = get_lm_datasets(tokenized_datasets, args, accelerator,
                                      tokenizer)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.train_num_samples is not None:
        # check if we have enough samples for the training set
        if args.train_num_samples > len(train_dataset):
            args.train_num_samples = len(train_dataset)
        train_dataset = train_dataset.select(range(args.train_num_samples))

    if args.validation_num_samples is not None:
        # check if we have enough samples for the validation set
        if args.validation_num_samples > len(eval_dataset):
            args.validation_num_samples = len(eval_dataset)
        eval_dataset = eval_dataset.select(range(args.validation_num_samples))

    collator = default_data_collator
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=collator,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=collator,
                                 batch_size=args.per_device_eval_batch_size)

    ######## Setup trainable modules
    for param in model.parameters():
        param.requires_grad = False

    if args.train_lm_head:
        for param in model.lm_head.parameters():
            param.data = param.data.float()
            param.requires_grad = True

    train_layer_indexs = [int(x) for x in args.train_layers.split(",")]
    if args.student_layers == "-":
        student_layer_indexs = []
    else:
        student_layer_indexs = [int(x) for x in args.student_layers.split(",")]
    assert len(set(train_layer_indexs)) == len(
        train_layer_indexs
    ), "train_layer_indexs should not have duplicate layers"
    assert len(set(student_layer_indexs)) == len(
        student_layer_indexs
    ), "student_layer_indexs should not have duplicate layers"
    assert len(
        set(train_layer_indexs) & set(student_layer_indexs)
    ) == 0, "train_layer_indexs and student_layer_indexs should not overlap"

    layers = get_layers(model)
    trainable_layers = torch.nn.ModuleList()
    for train_layer_index in train_layer_indexs:
        trainable_layers.append(layers[train_layer_index])
        logger.info(f"Add trainable layer: {train_layer_index}")
        for param in layers[train_layer_index].parameters():
            param.data = param.data.float()
            param.requires_grad = True

    model.trainable_module = trainable_layers

    if args.student_strategy == "choose":
        total_layer_indexs = sorted(train_layer_indexs + student_layer_indexs)
        logger.info(f"Get all student layers: {total_layer_indexs}")
        layers = torch.nn.ModuleList([layers[i] for i in total_layer_indexs])

    elif args.student_strategy == "lowrank_svd":
        # prepare args for lowrank svd
        assert args.lowrank_svd_topk is not None
        topk = float(args.lowrank_svd_topk)
        if topk.is_integer():
            topk = int(topk)
        low_rank = args.lowrank_svd_strategy == "lowrank"
        sampling = args.lowrank_svd_strategy == "sampling"

        # do lowrank svd for layers not in train_layer_indexs
        for index, layer in tqdm(enumerate(layers), total=len(layers)):
            if index not in train_layer_indexs:
                logger.info(f"Decompose layer {index}")
                layer.to(torch.float32)
                find_linear_and_replace_with_svd_module(layer,
                                                        topk=topk,
                                                        low_rank=low_rank,
                                                        sampling=sampling)
                layer.half()
                for param in layer.parameters():
                    param.requires_grad = False
    elif args.student_strategy == "repeat":
        # we repeat trainable layers to get student layers
        # e.g. trainable layers = [22, 23], student layers = []
        new_layers = torch.nn.ModuleList()
        for idx in range(len(layers)):
            # print(idx)
            if idx in train_layer_indexs or idx in student_layer_indexs:
                new_layers.append(layers[idx])
                logger.info(f"Add student layer: {idx}")
            else:
                # copy
                left, right = 0, 0
                while idx + left not in train_layer_indexs + student_layer_indexs and idx + left >= 0:
                    left -= 1
                while idx + right not in train_layer_indexs + student_layer_indexs and idx + right < len(
                        layers) - 1:
                    right += 1

                # TODO: case while idx + right == len(layers) - 1
                assert left != 0 or right != 0, "left and right should not be 0"
                if left == 0:
                    copy_idx = idx + right
                elif right == 0:
                    copy_idx = idx + left
                else:
                    copy_idx = idx + left if abs(left) < right else idx + right

                print("copy_idx", copy_idx, "while idx", idx, "left", left,
                      "right", right)

                # TODO: maybe copy idx not in student and trainble layer indexs
                # we don't want to copy train_layers
                if args.only_repeat_non_trainable_layers and copy_idx in train_layer_indexs:
                    print("reset copy idx which is in train layer indexs")
                    if idx + left not in train_layer_indexs:
                        copy_idx = idx + left
                    elif idx + right not in train_layer_indexs:
                        copy_idx = idx + right
                    else:
                        print(copy_idx, "in", train_layer_indexs)
                        copy_idx = copy_idx

                new_layers.append(
                    deepcopy(layers[copy_idx]) if copy_idx in
                    train_layer_indexs else layers[copy_idx])

                logger.info(f"Copy student layer: {copy_idx}")

            if idx not in train_layer_indexs:
                for param in new_layers[-1].parameters():
                    param.requires_grad = False
        layers = new_layers
    set_layers(model, layers)

    if args.use_lora:
        use_lora(model.trainable_module, args.lora_rank, args.lora_alpha)

    if args.use_adapter:
        use_adapter(model.trainable_module, args.adapter_size)

    if args.use_bitfit:
        use_bitfit(model.trainable_module)

    starting_epoch = 0
    resume_step = -1

    trainable_params, all_param, trainable = compute_and_print_parameters(
        model)
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {trainable}"
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"Trainable parameter: {name} with shape {param.shape} and dtype {param.dtype}"
            )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps *
        args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps *
        args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config[
        "lr_scheduler_type"].value
    accelerator.init_trackers("offsite_tuning", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def eval_epoch():
        model.eval()
        losses = []
        for _, batch in enumerate(eval_dataloader):
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

    # if not args.no_teacher:
    #     to_teacher(model.module, args)
    #     _, teacher_zero_shot_perplexity = eval_epoch()
    #     logger.info(
    #         f"Teacher zero shot perplexity: {teacher_zero_shot_perplexity}")
    # else:
    #     teacher_zero_shot_perplexity = 0

    # to_student(model.module, args)

    # for name, param in model.named_parameters():
    #     logger.info(
    #         f"Parameter: {name} with shape {param.shape}, dtype {param.dtype}, and requires_grad {param.requires_grad}")

    _, student_zero_shot_perplexity = eval_epoch()
    logger.info(
        f"Student zero shot perplexity: {student_zero_shot_perplexity}")
    best_perplexity = float("inf")
    early_stop = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    completed_steps = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_lm_loss = 0
        interval_lm_loss = 0
        best_lm_loss = float("inf")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                lm_loss = outputs.loss

                loss = lm_loss
                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - LM loss: {lm_loss:.4f}"
                )

                total_lm_loss += lm_loss.item()
                interval_lm_loss += lm_loss.item()
                best_lm_loss = min(best_lm_loss, lm_loss.item())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # end accumulate gradients

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            else:
                continue

            if completed_steps % args.eval_steps == 0:
                plug_eval_loss, plug_ppl = 0, 0
                eval_loss, perplexity = eval_epoch()

                lm_loss = interval_lm_loss / args.eval_steps
                interval_lm_loss = 0
                interval_kd_loss = 0

                logger.info(
                    f"epoch {epoch} step {completed_steps}: student_ppl: {perplexity:.4f} plug_ppl: {plug_ppl:.4f} lm_loss: {lm_loss:.4f}"
                )

                log_info = {
                    "student_ppl": perplexity,
                    "student_eval_loss": eval_loss,
                    "plug_ppl": plug_ppl,
                    "plug_eval_loss": plug_eval_loss,
                    "ppl_gap": perplexity - plug_ppl,
                    "train_lm_loss": lm_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                }
                accelerator.log(
                    log_info,
                    step=completed_steps,
                )
                is_best = perplexity < best_perplexity
                best_perplexity = min(best_perplexity, perplexity)

                if args.save_for_eval and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    state_dict = unwrapped_model.trainable_module.state_dict()
                    save_state_dict(state_dict, args.output_dir,
                                    f"adapter_step{completed_steps}.pt")
                    srsly.write_json(
                        os.path.join(args.output_dir,
                                     f"metadata_step{completed_steps}.json"),
                        {
                            "step": completed_steps,
                            "ppl": perplexity,
                            "loss": eval_loss.item(),
                        })

                    gc.collect()
                    torch.cuda.empty_cache()

                if not args.no_save_model and is_best and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    state_dict = unwrapped_model.trainable_module.state_dict()
                    save_state_dict(state_dict, args.output_dir, "adapter.pt")
                    srsly.write_json(
                        os.path.join(args.output_dir, "metadata.json"),
                        args.__dict__)

                    gc.collect()
                    torch.cuda.empty_cache()

                if is_best and accelerator.is_main_process:
                    with open(
                            os.path.join(args.output_dir, "all_results.json"),
                            "w+") as f:
                        json.dump(
                            {
                                "best_perplexity": best_perplexity,
                                "plug_perplexity": plug_ppl,
                                "student_zero_shot_perplexity":
                                student_zero_shot_perplexity,
                                "train_lm_loss": lm_loss,
                                "epoch": epoch,
                                "step": completed_steps,
                                "trainable_params": trainable_params,
                                "all_param": all_param,
                                "trainable": trainable
                            },
                            f,
                            indent=2)

                # early stop
                # if accelerator.is_main_process:
                #     if is_best:
                #         early_stop = 0
                #     else:
                #         early_stop += 1

                #     if early_stop >= 10:
                #         logger.info("Early stop")
                #         break

    accelerator.end_training()


if __name__ == "__main__":
    main()

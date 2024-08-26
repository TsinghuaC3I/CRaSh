import logging
import os
import json
from copy import deepcopy
import math
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from transformers import GPT2LMHeadModel, OPTForCausalLM, BloomForCausalLM, LlamaForCausalLM
import srsly
from accelerate.logging import get_logger
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.task_utils import task_dict
from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks
from datasets import load_from_disk, DatasetDict
from src.data_utils import get_raw_datasets, get_tokenized_datasets, get_lm_datasets, process_text2text_datasets

from src.args_utils import parse_args
from src.model_utils import load_adapter, load_student, get_layers, set_layers, uniform_choose_layers
from src.task_utils import LM_EVAL_TASK_NAME_MAPPING
from visualize_utils import visual_for_loss_surface, visualize_plot

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


class LMEvalAdaptor(BaseLM):

    def __init__(self, model, tokenizer, batch_size=1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        else:
            return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            out = self.model(inps)[0]
            return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(context,
                                   max_length=max_length,
                                   eos_token_id=eos_token_id,
                                   do_sample=False)


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
    elif isinstance(model, LlamaForCausalLM):
        r = len(model.model.layers) - args.student_r_pad
        model.model.layers = model.model.layers[:l] + \
            student + model.model.layers[r:]
    else:
        raise NotImplementedError


def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs)

    print(">>>>>>>>>>>>>>>>> create finetuned model <<<<<<<<<<<<<<<<<<")
    print("loading model from %s" % args.model_name_or_path)
    # for target offsite-tuning model
    init_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16)

    model = deepcopy(init_model)

    if args.load_adapter:
        print("loading adapter from %s" % args.load_adapter)
        adapter_state_dict = torch.load(args.load_adapter, map_location='cpu')
        model = load_adapter(model, adapter_state_dict, args)
    if args.load_student:
        print("loading student from %s" % args.load_student)
        student_state_dict = torch.load(args.load_student, map_location='cpu')
        model = load_student(model, student_state_dict, args)
    finetuned_model = deepcopy(model)

    print(">>>>>>>>>>>>>>>>>>> create assist model <<<<<<<<<<<<<<<<<<")
    assist_model = deepcopy(init_model)
    print("Load train and student layer indexs from:", args.assist_adapter)
    metadata = srsly.read_json(
        args.assist_adapter.replace("adapter.pt", "metadata.json"))
    train_layer_indexs = [int(x) for x in metadata["train_layers"].split(",")]
    if args.assist_adapter:
        layers = get_layers(assist_model)
        print("Get train_layers:", train_layer_indexs)
        # assign adapter to teacher
        adapter_layers = torch.nn.ModuleList(
            [layers[index] for index in train_layer_indexs])

        print("Load adapter from:", args.assist_adapter)
        adapter_state_dict = torch.load(args.assist_adapter,
                                        map_location='cpu')
        adapter_layers.load_state_dict(adapter_state_dict)
        for index, layer in enumerate(train_layer_indexs):
            print("Assign", index, "to", layer)
            layers[layer] = adapter_layers[index]

        set_layers(assist_model, layers)
        print("End loading")

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

    if isinstance(model, LlamaForCausalLM):
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        # tokenizer.padding_side = "left"  # Allow batched inference

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

    # train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    def eval_model(model):
        collator = default_data_collator
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size)

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
    finetuned_model.to("cuda")
    assist_model.to("cuda")
    # init_model.to("cuda")

    data = {"X": [], "loss": []}
    start, end = 0, 1
    points = 25
    progress_bar = tqdm(range(points))
    for x in np.linspace(start, end, points):
        # \theta_1 + \alpha * (\theta_2 - \theta_1)
        for (p, p1, p2) in zip(model.parameters(),
                               finetuned_model.parameters(),
                               assist_model.parameters()):
            p.data = p1.data + x * (p2.data - p1.data)

        data["X"].append(x)
        loss, _ = eval_model(model)
        z = float(loss.detach().cpu())
        data["loss"].append(z)
        progress_bar.update(1)
        progress_bar.set_description(f"X: {x} - loss: {z}")

    # print(data)
    df = pd.DataFrame(data, columns=["loss"], index=data["X"])
    visualize_plot(df,
                   fname=args.additional_note,
                   xlabel="coefficient",
                   ylabel="loss",
                   title=args.title)

    with open(f"./figures/{args.additional_note}_coef.json", "w") as f:
        f.write(json.dumps(data))
        f.write("\n")


if __name__ == '__main__':
    main()

import os
import json
from copy import deepcopy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import GPT2LMHeadModel, OPTForCausalLM, BloomForCausalLM, LlamaForCausalLM

from accelerate.logging import get_logger
from accelerate import Accelerator

from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks

from src.args_utils import parse_args
from src.model_utils import load_adapter, load_student, get_layers, set_layers, get_student_by_choose
from src.peft_utils import use_lora
from src.task_utils import LM_EVAL_TASK_NAME_MAPPING

logger = get_logger(__name__)


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
        **accelerator_log_kwargs)

    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 torch_dtype=torch.float16)

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

    if args.use_lora:
        layers = get_layers(model)
        l, r = args.student_l_pad, len(layers) - args.student_r_pad
        layers = get_layers(model)
        lora_layers = layers[:l] + layers[r:]
        use_lora(lora_layers, args.lora_rank, args.lora_alpha)

    if args.load_adapter:
        adapter_state_dict = torch.load(args.load_adapter, map_location='cpu')
        model = load_adapter(model, adapter_state_dict, args)

    if args.load_student:
        student_state_dict = torch.load(args.load_student, map_location='cpu')
        model = load_student(model, student_state_dict, args)

    model = model.to("cuda").half()

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

    lm_eval_model = LMEvalAdaptor(model, tokenizer)

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")

    task_names = [LM_EVAL_TASK_NAME_MAPPING.get(t, t) for t in task_names]

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=128,
        no_cache=True,
    )

    print(evaluator.make_table(results))

    if args.output_dir is not None:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        del results["config"]["model"]
        with open(args.output_dir, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()

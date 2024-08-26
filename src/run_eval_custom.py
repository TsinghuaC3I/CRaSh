import os
import json
from copy import deepcopy
import srsly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import GPT2LMHeadModel, OPTForCausalLM, BloomForCausalLM, LlamaForCausalLM

from accelerate.logging import get_logger

from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks

from src.args_utils_custom import parse_args
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

def repeat_layers():
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

def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 torch_dtype=torch.float16)

    if args.train_layers is not None and args.student_layers is not None:
        train_layer_indexs = [int(x) for x in args.train_layers.split(",")]
        student_layer_indexs = [int(x) for x in args.student_layers.split(",")]

    else:
        print("Load train and student layer indexs from:", args.load_adapter)
        metadata = srsly.read_json(
            args.load_adapter.replace("adapter.pt", "metadata.json"))
        train_layer_indexs = [
            int(x) for x in metadata["train_layers"].split(",")
        ]
        student_layer_indexs = [
            int(x) for x in metadata["student_layers"].split(",")
        ]

    if args.load_adapter and not args.zero_shot_eval:
        layers = get_layers(model)
        # if zero_shot_eval, then we don't need to load adapter
        # for plugin adapter or student?
        # for transfering adapter from student to teacher
        print("Get train_layers:", train_layer_indexs)
        # assign adapter to teacher
        adapter_layers = torch.nn.ModuleList(
            [layers[index] for index in train_layer_indexs])

        if args.use_lora:
            print("Use lora")
            use_lora(adapter_layers, metadata["lora_rank"],
                     metadata["lora_alpha"])

        print("Load adapter from:", args.load_adapter)
        adapter_state_dict = torch.load(args.load_adapter, map_location='cpu')
        adapter_layers.load_state_dict(adapter_state_dict)
        for index, layer in enumerate(train_layer_indexs):
            print("Assign", index, "to", layer)
            layers[layer] = adapter_layers[index]

        set_layers(model, layers)
        print("End loading")
    # else:
    # for zero-shot student

    if args.load_student:
        # delete layer not in train_layer_indexs + student_layer_indexs
        print("Get student_layers:", student_layer_indexs + train_layer_indexs)
        layers = get_layers(model)
        adapter_student_layers = torch.nn.ModuleList()
        for index, layer in enumerate(layers):
            if index in student_layer_indexs:
                print("Add student layer:", index)
                adapter_student_layers.append(layer)
            elif index in train_layer_indexs:
                print("Add trainable layer:", index)
                adapter_student_layers.append(layer)

        set_layers(model, adapter_student_layers)

    # if args.use_lora:
    #     layers = get_layers(model)
    #     l, r = args.student_l_pad, len(layers) - args.student_r_pad
    #     layers = get_layers(model)
    #     lora_layers = layers[:l] + layers[r:]
    #     use_lora(lora_layers, args.lora_rank, args.lora_alpha)

    model = model.to("cuda").half()

    if "llama" in args.model_name_or_path:
        args.use_slow_tokenizer = True

    print("Init tokenizer")
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

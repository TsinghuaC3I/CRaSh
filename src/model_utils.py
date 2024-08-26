import gc
import os
from copy import deepcopy
from scipy.stats import norm
import torch
from torch import nn
from accelerate.logging import get_logger
from transformers import (
    MODEL_MAPPING,
    OPTForCausalLM,
    GPT2LMHeadModel,
    BloomForCausalLM,
    LlamaForCausalLM,
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

logger = get_logger(__name__)


class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers=1,
                 activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


def add_prologue(module, prologue):
    module.old_forward = module.forward
    module.prologue = prologue

    def new_forward(self):

        def lambda_forward(*args, **kwargs):
            self.input_args = args
            self.input_kwargs = kwargs
            if self.prologue is not None:
                x = self.prologue(args[0])
            else:
                x = args[0]
            args = (x, ) + args[1:]
            return self.old_forward(*args, **kwargs)

        return lambda_forward

    module.forward = new_forward(module)
    return module

def add_epilogue(module, epilogue):
    module.old_forward = module.forward
    module.epilogue = epilogue

    def new_forward(self):

        def lambda_forward(*args, **kwargs):
            output = self.old_forward(*args, **kwargs)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output

            if self.epilogue is not None:
                x = self.epilogue(x)

            if isinstance(output, tuple):
                output = (x, ) + output[1:]
            else:
                output = x

            self.cached_output = x
            return output

        return lambda_forward

    module.forward = new_forward(module)
    return module


# def uniform_choose_layers(layers: nn.ModuleList, num_student_layers=None):
#     if num_student_layers is None:
#         num_student_layers = len(layers)

#     student = nn.ModuleList()
#     stride = (len(layers) - 1) / (num_student_layers - 1)

#     for i in range(num_student_layers):
#         idx = round(i * stride)
#         logger.info(f"Adding layer {idx} to student")
#         student.append(layers[idx])

#     return student


def uniform_choose_layers(layers: nn.ModuleList,
                          num_student_layers=None,
                          uniform_deviation=0,
                          uniform_percent=1 / 4.):
    # print(uniform_deviation, uniform_percent)
    if num_student_layers is None:
        num_student_layers = len(layers)

    student = nn.ModuleList()
    stride = (len(layers) - 1) / (num_student_layers - 1)

    if uniform_deviation == 0 or uniform_deviation is None:
        idxs = [round(i * stride) for i in range(num_student_layers)]
    else:
        left = round(num_student_layers * uniform_percent)
        right = num_student_layers - left
        print(left, right)
        if uniform_deviation > 0:
            left, right = right, left
        assert left <= len(layers) * 1 / 2. and right <= len(
            layers) * 1 / 2., f"{left} {len(layers) * 1 / 2. } {right}"
        left_stride = (len(layers) * 1 / 2. - 1) / (left - 1)
        right_stride = (len(layers) * 1 / 2. - 1) / (right - 1)
        idxs = [round(i * left_stride) for i in range(left)]
        start = round(len(layers) * 1 / 2. - 1)
        if start <= idxs[-1]:
            start = idxs[-1] + 1
        idxs += [round(i * right_stride) + start for i in range(right)]
    print(",".join([str(i) for i in idxs]))
    for idx in idxs:
        logger.info(f"Adding layer {idx} to student")
        student.append(layers[idx])

    return student


def normal_choose_layers(layers: nn.ModuleList,
                         num_student_layers=None,
                         variance=3.0,
                         stratety="normal"):
    if num_student_layers is None:
        num_student_layers = len(layers)

    student = nn.ModuleList()

    if stratety == "normal":
        logits = torch.tensor([
            norm.pdf(i, loc=len(layers) // 2, scale=variance)
            for i in range(len(layers))
        ])
    elif stratety == "left_normal":
        logits = torch.tensor(
            [norm.pdf(i, 0, scale=variance) for i in range(len(layers))])
    elif stratety == "right_normal":
        logits = torch.tensor([
            norm.pdf(i, loc=len(layers), scale=variance)
            for i in range(len(layers))
        ])
    elif stratety == "inverse_normal":
        logits = torch.tensor([
            norm.pdf(i, loc=len(layers) // 2, scale=variance)
            for i in range(len(layers))
        ])
        middle = len(layers) // 2
        logits = logits[-middle:] + logits[:middle]
    else:
        raise ValueError(f"Unknown strategy {stratety}")

    indices = torch.multinomial(logits, num_student_layers, replacement=False)

    logger.info(f"Strategy: {stratety}")
    for idx in indices.sort().values:
        logger.info(f"Adding layer {idx} to student")
        student.append(layers[idx])
    return student


@torch.no_grad()
def magnitude_prune(model, ratio):
    for param in model.parameters():
        if param.dim() == 1:
            continue
        num_prune = int(param.numel() * ratio)
        threshold = param.abs().view(-1).kthvalue(num_prune).values.item()
        mask = (param.abs() >= threshold).to(param.dtype)
        param.mul_(mask)


@torch.no_grad()
def quantize(model, bits):
    for param in model.parameters():
        if param.dim() == 1:
            continue
        min, max = param.min(), param.max()
        zp = (max + min) / 2
        scale = (max - min) / (2**bits - 1)
        param.sub_(zp).div_(scale).round_().mul_(scale).add_(zp)


def get_layers(model):
    if isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, GPT2LMHeadModel):
        layers = model.transformer.h
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    else:
        raise NotImplementedError
    return layers


def set_layers(model, layers):
    if isinstance(model, OPTForCausalLM):
        model.model.decoder.layers = layers
    elif isinstance(model, GPT2LMHeadModel):
        model.transformer.h = layers
    elif isinstance(model, BloomForCausalLM):
        model.transformer.h = layers
    elif isinstance(model, LlamaForCausalLM):
        model.model.layers = layers
    else:
        raise NotImplementedError


# def set_layers(model, layers):
#     if isinstance(model, OPTForCausalLM):
#         model.model.decoder.layers = layers
#     elif isinstance(model, GPT2LMHeadModel):
#         model.transformer.h = layers
#     elif isinstance(model, BloomForCausalLM):
#         model.transformer.h = layers
#     elif isinstance(model, LlamaForCausalLM):
#         layers = model.model.layers
#     else:
#         raise NotImplementedError


def get_student_by_choose(student, args):
    if args.student_layer_selection_strategy == 'uniform':
        student = uniform_choose_layers(
            student,
            args.num_student_layers,
            uniform_deviation=args.uniform_deviation,
            uniform_percent=args.uniform_percent)
    elif "normal" in args.student_layer_selection_strategy:
        student = normal_choose_layers(
            student,
            num_student_layers=args.num_student_layers,
            variance=args.normal_variance,
            stratety=args.student_layer_selection_strategy)
    elif args.student_layer_selection_strategy == "fixed":
        layers = nn.ModuleList()
        for student_idx in args.fixed_student_index.split(","):
            layers.append(student[int(student_idx)])
        student = layers
    elif args.student_layer_selection_strategy == "all":
        # student = student
        pass
    else:
        raise NotImplementedError
    return student


def setup_teacher_student(model, args, accelerator):
    for param in model.parameters():
        param.requires_grad = False

    layers = get_layers(model)

    l, r = args.student_l_pad, len(layers) - args.student_r_pad
    if args.load_student:
        student_state_dict = torch.load(os.path.join(args.load_student,
                                                     'student.pt'),
                                        map_location='cpu')
        student_layers_len = len(
            set([k.split('.')[0] for k in student_state_dict.keys()]))
        logger.info(
            f"Loading student module from {args.load_student} with {student_layers_len} layers."
        )
        student = deepcopy(layers[:student_layers_len])
        student.load_state_dict(student_state_dict)
    else:
        student = deepcopy(layers[l:r])

    student = get_student_by_choose(student, args)

    student = student.to(accelerator.device)

    if args.magnitude_pruning_ratio > 0:
        logger.info(
            f"Pruning student module with magnitude ratio {args.magnitude_pruning_ratio}"
        )
        magnitude_prune(student, args.magnitude_pruning_ratio)

    if args.weight_quantization_bits is not None:
        logger.info(
            f"Quantizing student module with {args.weight_quantization_bits} bits"
        )
        quantize(student, args.weight_quantization_bits)

    if args.train_module == 'student':
        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True
            
    elif args.train_module == 'adapter':
        for param in student.parameters():
            param.requires_grad = False
        if not args.freeze_bottom:
            for param in layers[:l].parameters():
                param.data = param.data.float()
                param.requires_grad = True
        for param in layers[r:].parameters():
            param.data = param.data.float()
            param.requires_grad = True
            
    elif args.train_module == 'all':
        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in layers[:l].parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in layers[r:].parameters():
            param.data = param.data.float()
            param.requires_grad = True
    else:
        raise NotImplementedError

    # layers of student, teacher and adapter, where only parameters in adapter are updated
    model.student = student

    model.teacher = layers[l:r].half()

    # if args.student_layer_selection_strategy == "fixed":
    #     adapter = nn.ModuleList()
    #     for adapter_idx in args.fixed_adapter_index.split(","):
    #         adapter.append(layers[int(adapter_idx)])
    #     model.adapter = adapter
    # else:
    model.adapter = layers[:l] + layers[r:]

    for param in model.teacher.parameters():
        param.requires_grad = False

    # save input and output hidden states of student (which are used to compute kd loss compared with teacher)
    add_prologue(model.student[0], None)
    add_epilogue(model.student[-1], None)

    # input and output
    model.student_l = model.student[0]
    model.student_r = model.student[-1]

    num_student_layers = len(model.student)
    logger.info(f"Number of student layers: {num_student_layers}")

    # assign trainalbe module
    if args.train_module == 'student':
        model.trainable_module = model.student
        
    elif args.train_module == 'adapter':
        model.trainable_module = model.adapter
        
    elif args.train_module == 'all':
        model.trainable_module = model.student + model.adapter
    else:
        raise NotImplementedError

    gc.collect()
    torch.cuda.empty_cache()
    return model


def to_teacher(model, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[:
                                                                l] + model.teacher + model.model.decoder.layers[
                                                                    r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, LlamaForCausalLM):
        r = len(model.model.layers) - args.student_r_pad
        model.model.layers = model.model.layers[:l] + \
            model.teacher + model.model.layers[r:]
    else:
        raise NotImplementedError


def to_student(model, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[:
                                                                l] + model.student + model.model.decoder.layers[
                                                                    r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.student + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.student + model.transformer.h[r:]
    elif isinstance(model, LlamaForCausalLM):
        r = len(model.model.layers) - args.student_r_pad
        model.model.layers = model.model.layers[:l] + \
            model.student + model.model.layers[r:]
    else:
        raise NotImplementedError


def assign_adapter_weights(adapter_layers, adapter_state_dict, args):
    if args.interpolate_adapter:
        logger.info(
            "assign adapter weights by interpolation with coefficient = %f" %
            args.interpolate_coef)
        for key, value in adapter_layers.state_dict().items():
            assert key in adapter_state_dict
            adapter_state_dict[key] = (
                1 - args.interpolate_coef
            ) * value + args.interpolate_coef * adapter_state_dict[key]

    adapter_layers.load_state_dict(adapter_state_dict)


def load_adapter(model, adapter_state_dict, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        adapter_layers = model.model.decoder.layers[:
                                                    l] + model.model.decoder.layers[
                                                        r:]
        # adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        adapter_layers = model.transformer.h[:l] + model.transformer.h[r:]
        # adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        adapter_layers = model.transformer.h[:l] + model.transformer.h[r:]
        # adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, LlamaForCausalLM):
        r = len(model.model.layers) - args.student_r_pad
        adapter_layers = model.model.layers[:l] + model.model.layers[r:]
        # adapter_layers.load_state_dict(adapter_state_dict)
    else:
        raise NotImplementedError
    assign_adapter_weights(adapter_layers, adapter_state_dict, args)
    return model


def load_student(model, student_state_dict, args):
    l = args.student_l_pad

    student_layers_len = len(
        set([k.split('.')[0] for k in student_state_dict.keys()]))
    logger.info(
        f"Loading student module from with {student_layers_len} layers.")
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        student_layers = model.model.decoder.layers[l:l + student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.model.decoder.layers = model.model.decoder.layers[:l] + \
            student_layers + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l + student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
            student_layers + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l + student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
            student_layers + model.transformer.h[r:]
    elif isinstance(model, LlamaForCausalLM):
        r = len(model.model.layers) - args.student_r_pad
        student_layers = model.model.layers[l:l + student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.model.layers = model.model.layers[:l] + \
            student_layers + model.model.layers[r:]
    else:
        raise NotImplementedError
    return model


def save_state_dict(state_dict, output_dir, filename):
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch.float16).cpu()
    torch.save(state_dict, os.path.join(output_dir, filename))


def compute_and_print_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    trainable = 100 * trainable_params / all_param
    return trainable_params, all_param, trainable

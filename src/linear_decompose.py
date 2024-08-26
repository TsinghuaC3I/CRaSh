#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-04-20 00:19:07
# @Author  : KaiyanZhang (zhang-ky22@mails.tsinghua.edu.cn)
# @Link    : https://github.com/iseesaw
import torch
from tqdm import tqdm
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class SVDLinearModule(torch.nn.Module):

    def __init__(self,
                 weight=None,
                 bias=None,
                 topk=None,
                 sampling=False,
                 low_rank=False):
        super(SVDLinearModule, self).__init__()
        in_hidden_size = weight.shape[0]
        out_hidden_size = weight.shape[1]
        self.in_hidden_size = in_hidden_size
        self.out_hidden_size = out_hidden_size
        self.rank = min(self.in_hidden_size, self.out_hidden_size)
        self.topk = topk
        self.low_rank = low_rank
        self.sampling = sampling
        if self.topk is not None:
            assert self.topk <= self.rank, "topk should be smaller than rank"
            self.rank = self.topk if isinstance(self.topk, int) else int(
                self.rank * self.topk)

        self.U = torch.nn.Parameter(
            torch.Tensor(self.in_hidden_size, self.rank))
        self.S = torch.nn.Parameter(torch.Tensor(self.rank))
        self.weight = torch.nn.Parameter(
            torch.Tensor(self.out_hidden_size, self.rank))

        self.reset_parameters(weight)
        if bias is not None:
            self.bias = bias
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self, weight):
        # (In, Out) -> (In, k), (k), (k, Out)
        if self.low_rank:
            self.U.data, self.S.data, self.weight.data = torch.svd_lowrank(
                weight, self.rank)
            return
        u, s, v = torch.svd(weight)
        # u, s, v = torch.svd_lowrank(weight, self.rank)
        if self.topk is not None:
            if self.sampling:
                indices = torch.multinomial(s, self.rank)
                idx = indices.sort().values
            else:
                topk = torch.topk(s, self.rank)
                idx = topk.indices.sort().values

            self.U.data = u[:, idx]
            self.S.data = s[idx]
            self.weight.data = v[:, idx]
        else:
            self.U.data = u
            self.S.data = s
            self.weight.data = v

    def reset_parameters_via_topk(self, topk):
        self.U = torch.nn.Parameter(self.U.data[:, topk])
        self.S = torch.nn.Parameter(self.S.data[topk])
        self.weight = torch.nn.Parameter(self.weight.data[:, topk])

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.linear(
            x, self.U @ torch.diag(self.S) @ self.weight.T, self.bias)

    def extra_repr(self):
        return 'in_hidden_size={}, out_hidden_size={}, bias={}'.format(
            self.in_hidden_size, self.out_hidden_size, self.bias is not None)


def get_target_module(model, key):
    parent_module = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent_module, target_name, target


def find_linear_and_replace_with_svd_module(model,
                                            disabled_tqdm=True,
                                            **kwargs):
    keys = [key for key, _ in model.named_modules()]
    target_modules = get_target_modules_from_model_type(model)
    for key in keys if disabled_tqdm else tqdm(keys):
        target_module_found = any(
            [key.endswith(target_module) for target_module in target_modules])

        if target_module_found:
            parent_module, target_name, target = get_target_module(model, key)
            if isinstance(target, torch.nn.Linear):
                # print("apply SVD to ", key, target, kwargs)
                new_module = SVDLinearModule(target.weight,
                                             bias=target.bias,
                                             **kwargs)
                setattr(parent_module, target_name, new_module)


def get_target_modules_from_model_type(layer):
    if isinstance(layer, OPTDecoderLayer):
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    elif isinstance(layer, LlamaDecoderLayer):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj"]
    else:
        raise NotImplementedError

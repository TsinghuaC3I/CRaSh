#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-04-22 12:58:26
# @Author  : KaiyanZhang (zhang-ky22@mails.tsinghua.edu.cn)
# @Link    : https://github.com/iseesaw
import torch


# origin knowledge distillation refer to offsite-tuning
def get_kd_loss(model):
    # only compute hidden states in student and teacher transformer layers
    # Get kwargs and args from prologue
    kwargs = model.student_l.input_kwargs
    args = model.student_l.input_args
    
    # use float16 to speed knowledge distillation
    output_teacher = args[0].to(torch.float16)
    args = list(args[1:])
    for i, arg in enumerate(args):
        if torch.is_tensor(arg) and arg.dtype == torch.float32:
            args[i] = arg.to(torch.float16)
    args = tuple(args)

    for k, v in kwargs.items():
        if torch.is_tensor(v) and v.dtype == torch.float32:
            kwargs[k] = v.to(torch.float16)

    with torch.no_grad():
        model.teacher.eval()
        for teacher_layer in model.teacher:
            output_teacher = teacher_layer(output_teacher, *args, **kwargs)
            if isinstance(output_teacher, tuple):
                output_teacher = output_teacher[0]

    # get cached_output from epilogue
    output_student = model.student_r.cached_output.float()
    output_teacher = output_teacher.float()

    # compute kl loss
    std = output_teacher.pow(2).mean().sqrt()
    kd_loss = (output_teacher - output_student).div(std).pow(2).mean()
    return kd_loss


def get_layer_align_kd_loss(model):
    pass


def get_attention_align_kd_loss(model):
    pass
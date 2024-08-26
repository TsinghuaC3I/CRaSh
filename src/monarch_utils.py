#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-05-18 11:06:18
# @Author  : KaiyanZhang (zhang-ky22@mails.tsinghua.edu.cn)
# @Link    : https://github.com/iseesaw
import math
import torch
from einops import rearrange
import numpy as np

def low_rank_project(M, rank):
    """Supports batches of matrices as well.
    """
    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt

def factors(n):
    return [(i, n // i) for i in range(1, math.floor(math.sqrt(n)) + 1) if n % i == 0]


def blockdiag_butterfly_project_einsum_rank(M, nblocks1, nblocks2, rank):
    """
    Arguments:
        M: (m, n)
    Outputs:
        w1_bfly: (nblocks1, r * nblocks2, i)
        w2_bfly: (nblocks2, l, nblocks1 * r)
    """
    m, n = M.shape
    k, j = nblocks1, nblocks2
    M_permuted_batched = rearrange(M, '(l j) (k i) -> k j l i', k=nblocks1, j=nblocks2)
    U, Vt = low_rank_project(M_permuted_batched, rank=rank)
    w1_bfly = rearrange(Vt, 'k j r i -> k (r j) i')
    w2_bfly = rearrange(U, 'k j l r -> j l (k r)')
    return w1_bfly, w2_bfly

def blockdiag_butterfly_project(M, sizes=None):
    """Only works for square matrices for now
    """
    m, n = M.shape
    if m != n:
        raise NotImplementedError('Only support square matrices')
    if sizes is None:
        # Find the factors that are closest to sqrt(n)
        sizes = factors(n)[-1]
        # Larger factor first is probably more efficient, idk
        sizes = (sizes[1], sizes[0])
    assert n == sizes[0] * sizes[1]
    M_permuted_batched = rearrange(M, '(p k) (r s) -> k r p s', k=sizes[1], r=sizes[0])
    U, Vt = low_rank_project(M_permuted_batched, rank=1)
    w1_bfly = rearrange(Vt, 'k r 1 s -> r k s')
    w2_bfly = rearrange(U, 'k r s 1 -> k s r')
    return w1_bfly, w2_bfly


class BlockdiagButterflyMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly

blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply


if __name__ == "__main__":
    state_dict = torch.load("/root/pubmodels/transformers/bert/bert-base-uncased/pytorch_model.bin", map_location="cpu")
    for key, value in state_dict.items():
        if len(value.shape) == 2 and value.shape[0] == value.shape[1]:
            # print(key, value.shape)
            w1_bfly, w2_bfly = blockdiag_butterfly_project_einsum_rank(value, 4, 4, 4)
            print(w1_bfly.shape, w2_bfly.shape)
            
            w1_bfly, w2_bfly = blockdiag_butterfly_project(value)
            print(w1_bfly.shape, w2_bfly.shape)
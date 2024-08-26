#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-05-19 16:11:40
# @Author  : KaiyanZhang (zhang-ky22@mails.tsinghua.edu.cn)
# @Link    : https://github.com/iseesaw

import torch
import tensorly as tl
from tensorly.decomposition import tucker



def test_tucker():
    state_dict = torch.load("/root/pubmodels/transformers/bert/bert-base-uncased/pytorch_model.bin", map_location="cpu")
    # Set tensorly backend to PyTorch
    tl.set_backend('pytorch')
    for key, value in state_dict.items():
        if len(value.shape) == 2 and value.shape[0] == value.shape[1]:
            # Define rank for Tucker decomposition
            rank = (6, 6)
            # Apply Tucker decomposition
            core, factors = tucker(value, rank=rank)

            # Reconstruct the matrix
            reconstructed_matrix = tl.tucker_to_tensor((core, factors))

            # Compute the reconstruction error
            reconstruction_error = torch.norm(value - reconstructed_matrix)

            print(f"Reconstruction error: {reconstruction_error.item()}")

            U, S, V = torch.svd_lowrank(value, q=rank[0])
            # print(U.shape, S.shape, V.shape, value.shape)
            svd_error = torch.norm(U @ torch.diag(S) @ V.T - value)
            print("svd lowrank", svd_error)
            
if __name__ == "__main__":
    pass
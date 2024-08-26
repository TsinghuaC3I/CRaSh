#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-05-19 16:11:40
# @Author  : KaiyanZhang (zhang-ky22@mails.tsinghua.edu.cn)
# @Link    : https://github.com/iseesaw

import torch
import tensorly as tl
from tensorly.decomposition import tucker

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorly.decomposition import partial_tucker

# Define a custom TuckerLinear layer
class TuckerLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(TuckerLinear, self).__init__()

        # Create the Tucker decomposition parameters
        self.core = nn.Parameter(torch.Tensor())
        self.last_projection = nn.Parameter(torch.Tensor())
        self.first_projection = nn.Parameter(torch.Tensor())

        # Initialize the parameters
        self.initialize_parameters(in_features, out_features, rank)

    def initialize_parameters(self, in_features, out_features, rank):
        # Perform Tucker decomposition for the weight tensor
        weights_shape = (out_features, in_features)
        core_shape, [last_dim, first_dim] = partial_tucker(
            weights_shape, modes=[0, 1], rank=rank
        )

        # Load the weights from an existing transformer model checkpoint
        # For example, if you have a pretrained model checkpoint loaded in `state_dict`
        pretrained_weights = state_dict['your_layer_name.weight']

        # Reshape the pretrained weights to match the Tucker decomposition
        pretrained_weights = pretrained_weights.view(weights_shape)

        # Perform Tucker decomposition on the pretrained weights
        core, [last_projection, first_projection] = partial_tucker(
            pretrained_weights, modes=[0, 1], rank=rank
        )

        # Set the parameters with the decomposed tensors
        self.core = nn.Parameter(torch.Tensor(core))
        self.last_projection = nn.Parameter(torch.Tensor(last_projection))
        self.first_projection = nn.Parameter(torch.Tensor(first_projection))

    def forward(self, input):
        # Apply Tucker decomposition to the input
        tucker_input = torch.matmul(torch.matmul(input, self.first_projection), self.core)

        # Apply linear transformation with the last projection matrix
        output = torch.matmul(tucker_input, self.last_projection)

        return output

if __name__ == "__main__":
    # Example usage
    # input_dim = 512
    # output_dim = 256
    # rank = 32

    # model = TuckerLinear(input_dim, output_dim, rank)
    # input_tensor = torch.randn(10, input_dim)  # Example input tensor
    # output_tensor = model(input_tensor)
    # print(output_tensor.size())

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
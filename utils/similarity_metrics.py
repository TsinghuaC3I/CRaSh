#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import numpy as np
from scipy.linalg import sqrtm
from torch.nn.functional import kl_div
from torch.nn import CosineSimilarity
from scipy.stats import wasserstein_distance


# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/CKA.py
# https://github.com/jayroxis/CKA-similarity
class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(
            self.centering(self.rbf(X, sigma)) *
            self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


def kl_divergence(p, q):
    return (p * (p / q).log()).sum()


def js_divergence(p, q):
    p = p + 1e-10
    q = q + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def hellinger_distance(p, q):
    return torch.sqrt(torch.sum(
        (torch.sqrt(p) - torch.sqrt(q))**2)) / np.sqrt(2)


def layer_similarity(layers_output, stype="js_div"):
    N = layers_output.shape[0]
    similarity_matrix = torch.zeros(
        (N, N)).to(layers_output.device)  # ensure on the same device
    if stype == "cosine_similarity":
        cosine_similarity = CosineSimilarity(dim=2)

    for i in range(N):
        for j in range(i + 1, N):
            # Wasserstein distance requires a different approach because torch doesn't have built-in support for it
            # Here we use a simple approximation of Wasserstein distance - the L2 distance, also known as Euclidean distance
            if stype == "wasserstein_distance":
                avg_dist = torch.mean(
                    torch.stack([
                        torch.dist(layers_output[i, k, :], layers_output[j,
                                                                         k, :])
                        for k in range(layers_output.shape[1])
                    ]))
                # avg_dist = np.mean([
                #     wasserstein_distance(layers_output[i, k, :].cpu().numpy(),
                #                          layers_output[j, k, :].cpu().numpy())
                #     for k in range(layers_output.shape[1])
                # ])
            elif stype == "hellinger_distance":
                avg_dist = torch.mean(
                    torch.stack([
                        hellinger_distance(layers_output[i, k, :],
                                           layers_output[j, k, :])
                        for k in range(layers_output.shape[1])
                    ]))
            elif stype == "kl_div":
                avg_dist = torch.mean(
                    torch.stack([
                        kl_divergence(layers_output[i, k, :],
                                      layers_output[j, k, :])
                        for k in range(layers_output.shape[1])
                    ]))
            elif stype == "js_div":
                avg_dist = torch.mean(
                    torch.stack([
                        js_divergence(layers_output[i, k, :],
                                      layers_output[j, k, :])
                        for k in range(layers_output.shape[1])
                    ]))
            elif stype == "cosine_similarity":
                avg_dist = torch.mean(
                    cosine_similarity(layers_output[i], layers_output[j]))
            else:
                raise NotImplementedError

            similarity_matrix[i, j] = avg_dist
            similarity_matrix[j, i] = avg_dist

    return similarity_matrix

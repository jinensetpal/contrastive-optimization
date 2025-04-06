#!/usr/bin/env python3

import torch.nn.functional as F
from torch import nn
import torch


# code copied from: https://github.com/jiachenzhu/DyT/blob/main/dynamic_tanh.py
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


# code modified from: https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
class ModifiedBN2d(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ModifiedBN2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        torch._dynamo.config.force_parameter_static_shapes = False
        self.update_proxy_stats = False

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")

    @torch.compile(dynamic=True, backend='cudagraphs')
    @staticmethod
    def functional(input, running_mean, running_var, eps, weight, bias, affine):
        input = (input - running_mean[None, :, None, None]) / (torch.sqrt(running_var[None, :, None, None] + eps))
        if affine: input = input * weight[None, :, None, None] + bias[None, :, None, None]

        return input

    @torch.compiler.disable
    def forward(self, input):
        self._check_input_dim(input)

        if self.update_proxy_stats:
            self.running_mean = input.mean([0, 2, 3])
            self.running_var = input.var([0, 2, 3], unbiased=False)

        if torch.is_grad_enabled(): return ModifiedBN2d.functional(input, self.running_mean, self.running_var, self.eps, self.weight, self.bias, self.affine)
        else:
            if self.affine: F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)
            return F.batch_norm(input, self.running_mean, self.running_var, training=False, eps=self.eps)

#!/usr/bin/env python3

import torch.nn as nn
import torch


class LazyDyT(nn.modules.lazy.LazyModuleMixin, nn.Module):
    def __init__(self, init_alpha=.5, inplace=None):
        super().__init__()

        self.init_alpha = init_alpha
        self.alpha = nn.UninitializedParameter()
        self.gamma = nn.UninitializedParameter()
        self.beta = nn.UninitializedParameter()

    def initialize_parameters(self, input):
        if self.has_uninitialized_params():
            with torch.no_grad():
                channels = input.size(1)
                self.alpha = nn.Parameter(torch.full((channels,), float(self.init_alpha)))
                self.gamma = nn.Parameter(torch.ones(channels))
                self.beta = nn.Parameter(torch.zeros(channels))

    @torch.compile(dynamic=True)
    def forward(self, x):
        return (torch.tanh(x.transpose(1, 3) * self.alpha) * self.gamma + self.beta).transpose(1, 3)


class EEU(nn.Module):
    def __init__(self, inplace=None):
        super().__init__()

    @torch.compile(dynamic=True)
    def forward(self, x):
        return (x < 0).to(torch.float) * (torch.exp(x)-1) + (x > 0).to(torch.float) * (-torch.exp(-x))+1


class ExtendedSigmoid(nn.Module):
    def __init__(self, inplace=None):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    @torch.compile(dynamic=True)
    def forward(self, x):
        return 2 * self.sigmoid(x) - 1

#!/usr/bin/env python3

import torch.nn.functional as F
from src import const
from torch import nn
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        return torch.exp(-cc[heatmap < .1]).mean() + torch.exp(cc[heatmap >= .1].pow(2)).mean()

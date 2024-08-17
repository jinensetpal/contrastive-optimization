#!/usr/bin/env python3

import torch.nn.functional as F
from src import const
from torch import nn
import torch


class SoftmaxKLDLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_pred, y, debug=False):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        cc_log_probs = F.softmax((cc * const.LAMBDAS[0]).flatten(start_dim=2), dim=2).reshape(*cc.shape).log()
        heatmap_probs = F.softmax(((y[0] + 5) * const.LAMBDAS[1]).flatten(start_dim=1), dim=1).reshape(cc.shape[0], *cc.shape[-2:]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        heatmap_log_probs = heatmap_probs.log()
        heatmap_log_probs[heatmap != 0] == 0

        return const.LAMBDAS[2] * (heatmap_probs * (heatmap_log_probs - cc_log_probs)).sum() / cc.size(0) + const.LAMBDAS[3] * cc[heatmap == 0].pow(2).mean() - const.LAMBDAS[4] * cc.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        heatmap[heatmap < .5] = .5
        heatmap[heatmap >= 1] = 1.  # glitching torch, my theory -> some values are 1.0000 and extra 0's are counted as extending the [0,1] range

        return F.binary_cross_entropy(F.sigmoid(cc), heatmap) + const.LAMBDAS[3] * cc[heatmap == .5].pow(2).mean() - const.LAMBDAS[4] * cc.mean()


class PieceWiseLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        return torch.exp(-cc[heatmap > .3]).mean() + torch.exp(cc[heatmap <= .3].abs()).mean()


class KLDLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        cc -= cc.min(dim=2).values.min(dim=2).values.broadcast_to((*cc.shape[-2:], *cc.shape[:2])).permute(2, 3, 0, 1)
        cc += 5E-3  # smoothing
        cc /= cc.sum(dim=2).sum(dim=2).broadcast_to((*cc.shape[-2:], *cc.shape[:2])).permute(2, 3, 0, 1)

        y[0] += 5E-3  # smoothing
        y[0] = torch.div(y[0].T, y[0].sum(dim=1).sum(dim=1)).T  # double-transpose to exploit torch's broadcasting rules

        return self.kld(cc.log(), y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3))


class NonPositiveLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        return (cc[heatmap <= .5]).pow(2).mean() - torch.nan_to_num(cc[(heatmap > .5) & (cc < 0)]).mean()

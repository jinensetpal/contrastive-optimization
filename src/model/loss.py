#!/usr/bin/env python3

from torch import nn
import torch


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
        self.cse = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        return (cc[heatmap <= .5]).pow(2).mean() - torch.nan_to_num(cc[(heatmap > .5) & (cc < 0)]).mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn
        self.cse = nn.CrossEntropyLoss()
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        cc = self.softmax_2((cc * 1E1).flatten(start_dim=2)).reshape(*cc.shape)
        heatmap = self.softmax_1((y[0] * 5E1).flatten(start_dim=1)).reshape(cc.shape[0], *cc.shape[-2:]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        return self.kld(cc.log(), heatmap) + (cc[heatmap <= .2]).abs().mean()

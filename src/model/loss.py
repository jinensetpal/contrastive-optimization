#!/usr/bin/env python3

import torch.nn.functional as F
from src import const
from torch import nn
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn, debug=False):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn
        self.debug = debug

    def kld(self, cc, heatmap, reverse=False):
        heatmap[heatmap > .5] = 1

        cc_log_probs = F.sigmoid(cc)
        heatmap_probs = F.softmax((heatmap * const.LAMBDAS[1]).flatten(start_dim=1), dim=1).reshape(cc.shape[0], *cc.shape[-2:]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        heatmap[heatmap <= .5] = .5
        heatmap_log_probs = heatmap.log()
        # heatmap_log_probs[heatmap > .5] = 0

        # heatmap_probs[heatmap == 1] = .8 / (heatmap == 1).sum()
        # heatmap_probs[heatmap != 1] = .2 / (heatmap != 1).sum()

        return heatmap_probs * (heatmap_log_probs - cc_log_probs)

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        kld = self.kld(cc.abs(), heatmap.clone())
        kld_prime = self.kld((-cc).abs(), -heatmap.clone() + 1, reverse=True)
        foreground = cc[heatmap != 0].mean()

        self.prev = ((kld.sum() / cc.size(0)).item(), (kld_prime.sum() / cc.size(0)).item(), foreground.item())
        if self.debug: return const.LAMBDAS[2] * kld.sum() / cc.size(0) + const.LAMBDAS[3] * kld_prime.sum() / cc.size(0) - const.LAMBDAS[4] * foreground, kld, kld_prime
        return const.LAMBDAS[2] * kld.sum() / cc.size(0) + const.LAMBDAS[3] * kld_prime.sum() / cc.size(0)


class DoubleKLDLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn, debug=False):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn
        self.debug = debug

    def kld(self, cc, heatmap, reverse=False):
        heatmap[heatmap > .5] = 1

        cc_log_probs = F.sigmoid(cc)
        heatmap_probs = F.softmax((heatmap * const.LAMBDAS[1]).flatten(start_dim=1), dim=1).reshape(cc.shape[0], *cc.shape[-2:]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        heatmap[heatmap <= .5] = .5
        heatmap_log_probs = heatmap.log()

        return heatmap_probs * (heatmap_log_probs - cc_log_probs)

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        kld = self.kld(cc.abs(), heatmap.clone())
        kld_prime = self.kld((-cc).abs(), -heatmap + 1, reverse=True)

        self.prev = ((kld.sum() / cc.size(0)).item(), (kld_prime.sum() / cc.size(0)).item())
        if self.debug: return const.LAMBDAS[2] * kld.sum() / cc.size(0) + const.LAMBDAS[3] * kld_prime.sum() / cc.size(0), kld, kld_prime
        return const.LAMBDAS[2] * kld.sum() / cc.size(0) + const.LAMBDAS[3] * kld_prime.sum() / cc.size(0)


class KLDPenaltyLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn, debug=False):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn
        self.debug = debug

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        cc_log_probs = F.softmax((cc * const.LAMBDAS[0]).flatten(start_dim=2), dim=2).reshape(*cc.shape).log()
        heatmap_probs = F.softmax(((y[0]) * const.LAMBDAS[1]).flatten(start_dim=1), dim=1).reshape(cc.shape[0], *cc.shape[-2:]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        heatmap_log_probs = heatmap_probs.log()
        heatmap_log_probs[heatmap != 0] = 0

        # rids antialiasing
        # heatmap_probs[heatmap == 0] = 0.2 / (heatmap == 0).sum()
        # heatmap_probs[heatmap != 0] = 0.8 / (heatmap != 0).sum()
        # heatmap_probs[heatmap == 0] *= 1E20

        kld = heatmap_probs * (heatmap_log_probs - cc_log_probs)
        background = cc[heatmap == 0].pow(2).mean()
        foreground = cc[heatmap != 0].mean()

        self.prev = ((kld.sum() / cc.size(0)).item(), background.item(), foreground.item())
        if self.debug: return const.LAMBDAS[2] * kld.sum() / cc.size(0) + const.LAMBDAS[3] * background - const.LAMBDAS[4] * foreground, kld
        return const.LAMBDAS[2] * kld.sum() / cc.size(0) + const.LAMBDAS[3] * background - const.LAMBDAS[4] * foreground


class BCELoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn, debug=False):  # `debug` does nothing
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        heatmap[heatmap < .5] = .5
        heatmap[heatmap >= 1] = 1.  # glitching torch, my theory -> some values are 1.0000 and extra 0's are counted as extending the [0,1] range

        loss_weights = torch.empty(cc.shape, device=const.DEVICE)
        loss_weights[heatmap < .9] = .5 / (heatmap < .9).sum()
        loss_weights[heatmap >= .9] = .5 / (heatmap >= .9).sum()

        bce = F.binary_cross_entropy(F.sigmoid(cc), heatmap, reduction='none')
        return (loss_weights * bce).mean(), bce


class PieceWiseLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        foreground = torch.exp(-cc[heatmap == 1]).mean()
        background = torch.exp(cc[heatmap != 1].abs()).mean()

        self.prev = (foreground.item(), background.item())
        return torch.log(foreground + background)


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

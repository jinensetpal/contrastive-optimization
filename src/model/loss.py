#!/usr/bin/env python3

import torch.nn.functional as F
from src import const
from torch import nn
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, get_contrastive_cams_fn, debug=False, is_label_mask=False, multilabel=False, divergence=False):
        super().__init__()

        self.ce = nn.BCEWithLogitsLoss(reduction='none') if multilabel else nn.CrossEntropyLoss(label_smoothing=const.LABEL_SMOOTHING)
        self.get_contrastive_cams = get_contrastive_cams_fn
        self.is_label_mask = is_label_mask
        self.multilabel = multilabel
        self.divergence = divergence

    def forward(self, y_pred, y):
        if self.multilabel:
            labels = ((torch.arange(const.N_CLASSES) + 1) * torch.ones(*const.CAM_SIZE, const.N_CLASSES)).T[None,].repeat(y[0].size(0), 1, 1, 1).to(const.DEVICE)
            fg_mask = (labels == y[0].repeat(1, const.N_CLASSES, 1).view(y[0].size(0), -1, *y[0].shape[1:])).to(torch.int)
            ablation = (fg_mask * y_pred[1] - (1 - fg_mask) * y_pred[1].abs()).sum(dim=[2, 3]) * y[1] + y_pred[1].sum(dim=[2, 3]) * (1 - y[1])
        elif self.is_label_mask:
            cc = self.get_contrastive_cams(y[1], y_pred[1]).to(const.DEVICE)

            labels = y[1].argmax(1)
            fg_mask = torch.cat([(c == y).to(torch.int)[None,] for c, y in zip(y[0], labels)]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3).to(const.DEVICE)

            mixup_mask = 1 - fg_mask[:, 0] - (y[0] == -1).to(torch.int)
            mixed_idx = (mixup_mask * (y[0] + 1)).to(torch.int).flatten(start_dim=1).max(dim=1).values

            mixup_sparse_mask = torch.zeros(cc.shape, device=const.DEVICE, dtype=torch.int)
            mixup_sparse_mask.view(-1, *const.CAM_SIZE)[mixed_idx[mixed_idx.nonzero().flatten()] - 1 + mixed_idx.nonzero().flatten() * const.N_CLASSES] = mixup_mask[mixed_idx.nonzero().flatten()]

            ablation = (-cc * fg_mask + cc.abs() * (1 - fg_mask) + (-cc.abs() + cc) * mixup_sparse_mask).sum(dim=[2, 3])
        else:
            cc = self.get_contrastive_cams(y[1], y_pred[1]).to(const.DEVICE)

            fg_mask = y[0].repeat(1, const.N_CLASSES, 1).view(y[0].size(0), -1, *y[0].shape[1:]).to(torch.int).to(const.DEVICE)
            ablation = (-cc * fg_mask + cc.abs() * (1 - fg_mask)).sum(dim=[2, 3])

        ace = self.ce(ablation, y[1])
        if self.multilabel: ace = (ace[y[1] == 0].mean() + ace[y[1] == 1].mean()) / 2

        if self.divergence:
            cam_log_probs = ((y_pred[1] if self.multilabel else cc) * const.LAMBDAS[0]).view(*y_pred[1].shape[:2], -1).softmax(dim=2).clamp(min=1E-6).view(*y_pred[1].shape).log()
            fg_mask_probs = (fg_mask * const.LAMBDAS[1]).view(*y_pred[1].shape[:2], -1).to(torch.float).softmax(dim=2).view(y_pred[1].shape)
            fg_mask_log_probs = fg_mask_probs.log()
            fg_mask_log_probs[fg_mask != 0] = 0

            kld = fg_mask_probs * (fg_mask_log_probs - cam_log_probs) * fg_mask
        else: kld = torch.tensor(0)

        self.prev = (ace.item(), (kld.sum() / y[0].size(0)).item())
        return const.LAMBDAS[2] * ace + const.LAMBDAS[3] * kld.sum() / y[0].size(0)


class ACEKLDLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn, debug=False):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=const.LABEL_SMOOTHING)
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        fg_mask = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        ablation = (-cc * fg_mask + cc.abs() * (1 - fg_mask)).sum(dim=[2, 3])

        cc_log_probs = F.softmax((cc * const.LAMBDAS[0]).flatten(start_dim=2), dim=2).clamp(min=1E-6).reshape(*cc.shape).log()
        fg_mask_probs = F.softmax(((y[0]) * const.LAMBDAS[1]).flatten(start_dim=1), dim=1).reshape(cc.shape[0], *cc.shape[-2:]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        fg_mask_log_probs = fg_mask_probs.log()
        fg_mask_log_probs[fg_mask != 0] = 0

        kld = fg_mask_probs * (fg_mask_log_probs - cc_log_probs) * fg_mask
        ace = self.ce(ablation, y[1])

        self.prev = (ace.item(), (kld.sum() / cc.size(0)).item())
        return const.LAMBDAS[2] * ace + const.LAMBDAS[3] * kld.sum() / cc.size(0)


class DoubleKLDLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn, debug=False):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn
        self.debug = debug

    def kld(self, cc, fg_mask):
        fg_mask[fg_mask > .5] = 1

        cc_log_probs = F.sigmoid(cc)
        fg_mask_probs = F.softmax((fg_mask * const.LAMBDAS[1]).flatten(start_dim=1), dim=1).reshape(cc.shape[0], *cc.shape[-2:]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        fg_mask[fg_mask <= .5] = .5
        fg_mask_log_probs = fg_mask.log()

        return fg_mask_probs * (fg_mask_log_probs - cc_log_probs)

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        fg_mask = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        kld = self.kld(cc.abs(), fg_mask.clone())
        kld_prime = self.kld((-cc).abs(), -fg_mask + 1)

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
        fg_mask = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        cc_log_probs = F.softmax((cc * const.LAMBDAS[0]).flatten(start_dim=2), dim=2).reshape(*cc.shape).log()
        fg_mask_probs = F.softmax(((y[0]) * const.LAMBDAS[1]).flatten(start_dim=1), dim=1).reshape(cc.shape[0], *cc.shape[-2:]).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        fg_mask_log_probs = fg_mask_probs.log()
        fg_mask_log_probs[fg_mask != 0] = 0

        # rids antialiasing
        # fg_mask_probs[fg_mask == 0] = 0.2 / (fg_mask == 0).sum()
        # fg_mask_probs[fg_mask != 0] = 0.8 / (fg_mask != 0).sum()
        # fg_mask_probs[fg_mask == 0] *= 1E20

        kld = fg_mask_probs * (fg_mask_log_probs - cc_log_probs)
        background = cc[fg_mask == 0].pow(2).mean()
        foreground = cc[fg_mask != 0].mean()

        self.prev = ((kld.sum() / cc.size(0)).item(), background.item(), foreground.item())
        if self.debug: return const.LAMBDAS[2] * kld.sum() / cc.size(0) + const.LAMBDAS[3] * background - const.LAMBDAS[4] * foreground, kld
        return const.LAMBDAS[2] * kld.sum() / cc.size(0) + const.LAMBDAS[3] * background - const.LAMBDAS[4] * foreground


class BCELoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn, debug=False):  # `debug` does nothing
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        fg_mask = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
        fg_mask[fg_mask < .5] = .5
        fg_mask[fg_mask >= 1] = 1.  # glitching torch, my theory -> some values are 1.0000 and extra 0's are counted as extending the [0,1] range

        loss_weights = torch.empty(cc.shape, device=const.DEVICE)
        loss_weights[fg_mask < .9] = .5 / (fg_mask < .9).sum()
        loss_weights[fg_mask >= .9] = .5 / (fg_mask >= .9).sum()

        bce = F.binary_cross_entropy(F.sigmoid(cc), fg_mask, reduction='none')
        return (loss_weights * bce).mean(), bce


class PieceWiseLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        fg_mask = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        foreground = torch.exp(-cc[fg_mask == 1]).mean()
        background = torch.exp(cc[fg_mask != 1].abs()).mean()

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
        fg_mask = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        return (cc[fg_mask <= .5]).pow(2).mean() - torch.nan_to_num(cc[(fg_mask > .5) & (cc < 0)]).mean()

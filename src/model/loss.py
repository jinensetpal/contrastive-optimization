#!/usr/bin/env python3

from geomloss import SamplesLoss
import torch.nn.functional as F
from src import const
from torch import nn
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, get_contrastive_cams_fn, debug=False, is_label_mask=False,
                 multilabel=False, divergence=None, pos_weight=None, pos_only=False):
        super().__init__()

        self.ce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none' if pos_weight is None else 'mean') if multilabel else nn.CrossEntropyLoss(label_smoothing=const.LABEL_SMOOTHING)
        self.get_contrastive_cams = get_contrastive_cams_fn
        self.is_label_mask = is_label_mask
        self.multilabel = multilabel
        self.divergence = divergence
        self.pos_only = pos_only

        self.sinkhorn = SamplesLoss('sinkhorn', p=const.SINKHORN_COST_POW, blur=const.SINKHORN_BLUR)

    # code adapted from: https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis/blob/main/texture_optimization_slicing.py
    def sliced_wasserstein(self, cc, fg_mask, y, n_directions=None, spatial_scale_factor=10):
        n_cam_pixels = const.CAM_SIZE[0] * const.CAM_SIZE[1]
        if n_directions is None: n_directions = cc.size(1)

        if self.multilabel:
            target_mask = fg_mask.to(torch.float)
            target_mask.view(-1, n_cam_pixels)[(y[1].flatten() - 1).nonzero()] = -1 / n_cam_pixels
            target_mask = const.LAMBDAS[0] * (target_mask.T / (y[1] * target_mask.sum(2).sum(2) + 1 - y[1]).T).T.view(*target_mask.shape[:2], n_cam_pixels)

            cc = cc.view(*cc.shape[:2], n_cam_pixels)

            if self.pos_only:
                target_mask = target_mask[y[1].to(torch.bool)].unsqueeze(1)
                cc = cc[y[1].to(torch.bool)].unsqueeze(1)

                spatial_maps = fg_mask[y[1].to(torch.bool)].unsqueeze(1).flatten(2) * spatial_scale_factor
                target_mask = torch.hstack((target_mask, spatial_maps))
                cc = torch.hstack((cc, spatial_maps))
            else:
                spatial_maps = fg_mask.view(-1, *const.CAM_SIZE).unsqueeze(1).flatten(2) * spatial_scale_factor
                target_mask = torch.hstack((target_mask.view(-1, n_cam_pixels).unsqueeze(1), spatial_maps))
                cc = torch.hstack((cc.view(-1, n_cam_pixels).unsqueeze(1), spatial_maps))
        else:
            spatial_maps = fg_mask[:, 0].flatten(1).unsqueeze(1).clone()
            target_mask = (const.LAMBDAS[0] * spatial_maps / spatial_maps.sum(2, keepdim=True)).repeat(1, const.N_CLASSES - 1, 1)
            spatial_maps *= spatial_scale_factor

            cc = cc[(1 - y[1]).to(torch.bool)].view(-1, const.N_CLASSES - 1, n_cam_pixels)
            cc = torch.hstack((cc, spatial_maps))
            target_mask = torch.hstack((target_mask, spatial_maps))

        directions = torch.randn(n_directions, cc.size(1) - 1, device=const.DEVICE)
        directions /= directions.pow(2).sum(1, keepdim=True).sqrt()
        directions = torch.hstack((directions, torch.ones(n_directions, 1, device=const.DEVICE)))

        sorted_mask_projections = torch.einsum('bdn,md->bmn', target_mask, directions).sort(2)[0]
        distance = sorted_mask_projections - torch.einsum('bdn,md->bmn', cc, directions).sort(2)[0]

        if self.multilabel and not self.pos_only:
            distance[target_mask[:, 1].to(torch.bool).unsqueeze(1).repeat(1, n_directions, 1) & (sorted_mask_projections != 0) & (distance < 0)] = 0
            distance[(1 - target_mask[:, 1]).to(torch.bool).unsqueeze(1).repeat(1, n_directions, 1) & (distance > 0)] = 0
        else:   # (pos_only & multilabel) | contrastive
            distance[(sorted_mask_projections != 0) & (distance < 0)] = 0
        return distance.pow(2).mean()

    def wasserstein(self, cc, fg_mask, y, y_pred):
        fg_mask = fg_mask.to(torch.float)
        if not self.pos_only: fg_mask[(y[1].flatten() - 1).nonzero()] = -1 / (const.CAM_SIZE[0] * const.CAM_SIZE[1])
        fg_mask = const.LAMBDAS[0] * (fg_mask.T / (y[1].flatten() * fg_mask.sum(1).sum(1) + 1 - y[1].flatten())).T

        distance = self.sinkhorn(cc, fg_mask)
        distance = (distance[y[1].flatten() == 0].mean() + distance[y[1].flatten() == 1].mean()).mean()
        return distance + const.LAMBDAS[1] * (y_pred[0][y[1] == 0].pow(2).mean() + y_pred[0][y[1] == 0].pow(2).mean()).mean()  # term added for regularization; sinkhorn underpenalizes activation map being off in scale but this explodes entropy

    @staticmethod
    def kld(cc, fg_mask):
        fg_mask_probs = (fg_mask * const.LAMBDAS[1]).view(*cc.shape[:-2], -1).to(torch.float).softmax(dim=-1).view(cc.shape)
        cam_log_probs = (cc * const.LAMBDAS[0]).view(*cc.shape[:-2], -1).softmax(dim=-1).clamp(min=1E-6).view(cc.shape).log()
        fg_mask_log_probs = fg_mask_probs.log()
        fg_mask_log_probs[fg_mask != 0] = 0

        divergence = fg_mask_probs * (fg_mask_log_probs - cam_log_probs)
        return divergence.sum() / divergence.size(0)

    def forward(self, y_pred, y):
        if self.multilabel:
            labels = ((torch.arange(const.N_CLASSES) + 1) * torch.ones(*const.CAM_SIZE, const.N_CLASSES)).T[None,].repeat(y[0].size(0), 1, 1, 1).to(const.DEVICE)
            fg_mask = (labels == y[0].repeat(1, const.N_CLASSES, 1).view(y[0].size(0), -1, *y[0].shape[1:])).to(torch.int)
            fg_mask[y[1].to(torch.bool) & (fg_mask.sum(2).sum(2) == 0)] = 1
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

            y[0][y[0].sum(1).sum(1) == 0] = 1
            fg_mask = y[0].repeat(1, const.N_CLASSES, 1).view(y[0].size(0), -1, *y[0].shape[1:]).to(torch.int).to(const.DEVICE)
            ablation = (-cc * fg_mask + cc.abs() * (1 - fg_mask)).sum(dim=[2, 3])

        ace = self.ce(ablation, y[1])
        if self.multilabel and self.ce.pos_weight is None: ace = (ace[y[1] == 0].mean() + ace[y[1] == 1].mean()) / 2

        if self.divergence:
            if self.divergence == 'sliced_wasserstein': divergence = self.sliced_wasserstein(y_pred[1] if self.multilabel else cc, fg_mask, y, n_directions=const.SWD_N_DIRECTIONS)
            else:
                if self.multilabel:
                    if self.pos_only:
                        target_idx = y[1].flatten().nonzero()
                        cc = y_pred[1].view(-1, *y_pred[1].shape[-2:])[target_idx][:, 0]
                        fg_mask = fg_mask.view(-1, *fg_mask.shape[-2:])[target_idx][:, 0]
                    else:
                        cc = y_pred[1].view(-1, *const.CAM_SIZE).clone()
                        fg_mask = fg_mask.view(-1, *const.CAM_SIZE).clone()

                if self.divergence == 'wasserstein': divergence = self.wasserstein(cc, fg_mask, y, y_pred)
                elif self.divergence == 'kld': divergence = self.kld(cc, fg_mask)
        else: divergence = torch.tensor(0)

        self.prev = (ace.item(), divergence.item())
        return const.LAMBDAS[2] * ace + const.LAMBDAS[3] * divergence


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

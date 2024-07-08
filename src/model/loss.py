#!/usr/bin/env python3

from torch.nn import Module
from src import const


class ContrastiveLoss(Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        # if const.QUANTILE_CLIP_CAMS: cc[(cc < cc.view(-1, 56**2).quantile(.1, dim=1) & (cc > cc.quantile(.1))] = 0

        return (cc - (y[0] * const.SCALE_HEATMAPS).repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)).pow(3).mean()

#!/usr/bin/env python3

from torch.nn import Module
from src import const


class ConstrastiveLoss(Module):
    def __init__(self, get_contrastive_cam_fn):
        self.get_contrastive_cam = get_contrastive_cam_fn

    def forward(self, y_pred, y):
        y_pred, cam = y_pred

        cc = self.get_contrastive_cam_fn(y, y_pred[:, 1])
        if const.QUANTILE_CLIP_CAMS: cc[(cc < cc.quantile(.9)) & (cc > cc.quantile(.1))] = 0

        return (cc - (target_cam * 1E3)).pow(2).mean()

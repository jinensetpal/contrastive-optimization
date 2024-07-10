#!/usr/bin/env python3

from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, get_contrastive_cam_fn):
        super().__init__()
        self.get_contrastive_cam = get_contrastive_cam_fn
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_pred, y):
        cc = self.get_contrastive_cam(y[1], y_pred[1])
        cc /= cc.sum()
        y[0] /= y[0].sum()

        # if const.QUANTILE_CLIP_CAMS: cc[(cc < cc.view(-1, 56**2).quantile(.1, dim=1) & (cc > cc.quantile(.1))] = 0
        return self.kld(cc, y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3))

#!/usr/bin/env python3
from matplotlib.colors import Normalize

from ..data.oxford_iiit_pet import Dataset as oxford_iiit_pet
import matplotlib.animation as animation
from src.data.sbd import Dataset as sbd
from .loss import ContrastiveLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .arch import Model
from src import const
import random
import torch
import sys


if __name__ == '__main__':
    multilabel = len(sys.argv) == 2
    data = sbd(mode='segmentation') if multilabel else oxford_iiit_pet('train')
    model = Model(const.IMAGE_SHAPE, multilabel=multilabel, logits_only=True)
    optim = torch.optim.Adam(model.parameters(), lr=5E-4)
    criterion = ContrastiveLoss(model.get_contrastive_cams, multilabel=multilabel, divergence=const.DIVERGENCE, pos_only=const.POS_ONLY, pos_weight=data.reweight if multilabel else None)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=data.reweight)

    idx = random.randint(0, len(data) - 1)
    print(idx)
    X = data[idx][0].unsqueeze(0)
    y = [x.unsqueeze(0) for x in data[idx][1]]

    if multilabel: plt.imshow(y[0][0].cpu())

    frames = []
    for i in range(50):
        optim.zero_grad()

        model.train()
        y_pred = model(X)

        loss = criterion(y_pred, y)
        loss.backward()

        if not multilabel:
            cc = model.get_contrastive_cams(y[1], y_pred[1])
            frames.append([plt.imshow(F.interpolate(y[0][None], const.IMAGE_SIZE, mode='bilinear')[0][0].cpu(), cmap='jet', alpha=0.5, animated=True),
                           plt.imshow(F.interpolate(cc[:, y[1].argmin(1)], const.IMAGE_SIZE, mode='bilinear')[0, 0].detach().cpu(), cmap='jet', alpha=0.5, animated=True)])
            print(loss.item(), F.cross_entropy(y_pred[0], y[1]).detach().item())
        else: print(loss.item(), F.binary_cross_entropy_with_logits(y_pred[0], y[1], pos_weight=data.reweight).detach().item())

        optim.step()

    print(y_pred[0].detach(), y[1])

    fig = plt.figure(facecolor='white')
    if multilabel:
        for idx, cam in enumerate(y_pred[1][0]):
            fig.add_subplot(4, 5, idx+1)
            plt.imshow(cam.detach().cpu(), norm=Normalize(vmin=y_pred[1][0].min(), vmax=y_pred[1][0].max()))
    else:
        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
        ani.save(const.DATA_DIR / 'evals' / 'overfit_test.mp4')

        fig.add_subplot(2, 1, 1)
        plt.imshow(cc[0, y[1].argmin(1)][0].detach().cpu())
        fig.add_subplot(2, 1, 2)
        plt.imshow(y[0][0].detach().cpu())

    plt.tight_layout()
    plt.show()

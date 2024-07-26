#!/usr/bin/env python3

from src.data.oxford_iiit_pet import Dataset
import matplotlib.animation as animation
from .loss import ContrastiveLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .arch import Model
from src import const
import random
import torch


if __name__ == '__main__':
    data = Dataset('train')
    model = Model(const.IMAGE_SHAPE)
    optim = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
    loss_fn = ContrastiveLoss(model.get_contrastive_cams)

    idx = random.randint(0, len(data) - 1)
    X = data[idx][0].unsqueeze(0)
    y = [x.unsqueeze(0) for x in data[idx][1]]

    frames = []
    fig = plt.figure()
    for i in range(20):
        optim.zero_grad()

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()

        cc = model.get_contrastive_cams(y[1], y_pred[1])
        frames.append([plt.imshow(F.interpolate(y[0][None], const.IMAGE_SIZE, mode='bilinear')[0][0], cmap='jet', alpha=0.5, animated=True),
                       plt.imshow(F.interpolate(cc, const.IMAGE_SIZE, mode='bilinear')[0][0].detach(), cmap='jet', alpha=0.5, animated=True)])
        print(loss.item())

        optim.step()

    print(cc.mean().item(), y_pred[0].detach(), y[1])

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    ani.save(const.DATA_DIR / 'evals' / 'overfit_test.mp4')

    fig = plt.figure(figsize=(1, 2),
                     facecolor='white')
    fig.add_subplot(2, 1, 1)
    plt.imshow(cc[0, 0].detach())
    fig.add_subplot(2, 1, 2)
    plt.imshow(y[0][0].detach())

    plt.tight_layout()
    plt.show()

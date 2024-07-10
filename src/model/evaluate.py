#!/usr/bin/env python3

from ..data.oxford_iiit_pet import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .arch import Model
from src import const
import random
import torch
import sys


def visualize(model, gen):
    fig = plt.figure(figsize=(14, 14),
                     facecolor='white')

    for idx, sample in enumerate(random.sample(range(len(gen)), 16)):
        X, (heatmap, y) = gen[sample]
        y_pred, cam = model(X.unsqueeze(0).to(const.DEVICE))
        cam = model.get_contrastive_cams(y.unsqueeze(0).to(const.DEVICE), cam)[0, 0].detach()
        fig.add_subplot(4, 4, idx + 1)
        plt.xlabel(f'Pred: {str(y_pred.argmax().item())}, Actual: {str(y[1].argmax().item())}')
        plt.imshow(X.permute(1, 2, 0).detach().numpy(), alpha=0.5)
        plt.imshow(F.interpolate(cam[None, None, ...], const.IMAGE_SIZE, mode='bilinear')[0][0].numpy(), cmap='jet', alpha=0.5)

    plt.tight_layout()
    plt.show()
    fig.savefig(const.DATA_DIR / 'evals' / f'{model.name}.png')


if __name__ == '__main__':
    name = sys.argv[1]

    model = Model(input_shape=const.IMAGE_SHAPE)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    visualize(model, Dataset('test'))

#!/usr/bin/env python3

from ..data.oxford_iiit_pet import Dataset as oxford_iiit_pets
from ..data.pet_image import Dataset as pet_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .arch import Model
from src import const
import random
import torch
import sys


def kernel_diff(model):
    plt.plot((model.linear.weight[0] - model.linear.weight[1]).sort(descending=True).values.pow(2).detach())
    plt.savefig(const.DATA_DIR / 'evals' / f'kerneldiff_{model.name}.png')
    plt.show()


def visualize(model, gen):
    fig = plt.figure(figsize=(14, 14),
                     facecolor='white')

    for idx, sample in enumerate(random.sample(range(len(gen)), 16)):
        X, (heatmap, y) = gen[sample]
        y_pred, cam = model(X.unsqueeze(0))
        cam = model.get_contrastive_cams(y.unsqueeze(0), cam).detach()
        fig.add_subplot(4, 4, idx + 1)
        plt.xlabel(f'Pred: {str(y_pred[0].argmax().item())}, Actual: {str(y.argmax().item())}')
        plt.imshow(X.permute(1, 2, 0).cpu().detach(), alpha=0.5)
        plt.imshow(F.interpolate(cam, const.IMAGE_SIZE, mode='bilinear')[0, 0].cpu(), cmap='jet', alpha=0.5)

    plt.tight_layout()
    fig.savefig(const.DATA_DIR / 'evals' / f'{model.name}_cam.png')
    plt.show()


def accuracy(model, gen, samples=1000):
    score = 0

    for sample in random.sample(range(len(gen)), samples):
        X, y = gen[sample]
        y_pred, cam = model(X.unsqueeze(0))
        score += (y.argmax(dim=0) == y_pred.argmax(dim=1)).sum()

    return score / samples


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive=name != 'default')
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    if sys.argv[2] == 'visualize': visualize(model, oxford_iiit_pets('test'))
    else: print(accuracy(model, pet_image()))

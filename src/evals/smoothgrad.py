#!/usr/bin/env python3

from src.data.soodimagenet import get_generators
import torchvision.transforms.v2 as T
from src.model.arch import Model
import matplotlib.pyplot as plt
from src import const
import random
import torch
import sys


def gradient_influence(batch, model, idx=0, n_random_samples=50):
    batch[0] = batch[0].to(const.DEVICE)

    X = T.functional.gaussian_noise_image(batch[0][idx][None,].repeat(n_random_samples, 1, 1, 1), clip=False)
    X.requires_grad = True
    y_pred, cam = model(X)

    cam.sum().backward(inputs=X)
    influence = X.grad.abs().mean(0).mean(0).cpu()

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(influence)
    fig.add_subplot(1, 2, 2)
    plt.imshow(batch[0][idx].permute(1,2,0).detach().cpu())
    plt.show()

    return influence


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    train, val, _ = get_generators('test')

    gradient_influence(next(iter(train)), model)

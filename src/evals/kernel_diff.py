#!/usr/bin/env python3

import matplotlib.pyplot as plt
from ..model.arch import Model
from src import const
import random
import torch
import sys


def kernel_diff(model):
    plt.plot((model.linear.weight[0] - model.linear.weight[1]).sort(descending=True).values.pow(2).detach().cpu())
    plt.savefig(const.DATA_DIR / 'evals' / f'kerneldiff_{model.name.replace("/", "_")}.png')
    plt.show()

    class_sorted = model.linear.weight.T.sort(1, descending=True)
    plt.imshow(class_sorted.values[(class_sorted.values[:, 0] - class_sorted.values[:, 1]).sort(descending=True).indices].detach().cpu())
    plt.show()


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.DOWNSTREAM_MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    kernel_diff(model)

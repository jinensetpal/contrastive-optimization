#!/usr/bin/env python3


import matplotlib.pyplot as plt
from ..model.arch import Model
from src import const
import random
import torch
import sys


def kernel_diff(model):
    plt.plot((model.linear.weight[0] - model.linear.weight[1]).sort(descending=True).values.pow(2).detach().cpu())
    plt.savefig(const.DATA_DIR / 'evals' / f'kerneldiff_{model.name}.png')
    plt.show()


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    kernel_diff(model)

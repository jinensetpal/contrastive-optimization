#!/usr/bin/env python3

from ..data.pet_image import Dataset
from ..model.arch import Model
from src import const
import random
import torch
import sys


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

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / name / 'best.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    print(accuracy(model, Dataset()))

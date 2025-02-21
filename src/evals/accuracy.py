#!/usr/bin/env python3

from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
from torch.utils.data import DataLoader
from ..model.arch import Model
from src import const
import random
import torch
import sys


def accuracy(model, metric, gen):
    with torch.no_grad():
        for X, (heatmap, y) in gen: metric.update(model(X)[0].detach().flatten(), y.flatten())
    return metric.compute()


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    if sys.argv[2] == 'imagenet': from ..data.imagenet import Dataset
    elif sys.argv[2] == 'oxford': from ..data.oxford_iiit_pet import Dataset
    else: from ..data.pet_image import Dataset

    model = Model(is_contrastive='default' not in name, modified_bn=True)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    torch.multiprocessing.set_start_method('spawn', force=True)
    print(accuracy(model, BinaryAccuracy() if (sys.argv[2] != 'imagenet' and const.BINARY_CLS) else MulticlassAccuracy(),
                   DataLoader(Dataset('valid'), batch_size=const.BATCH_SIZE, num_workers=const.N_WORKERS, shuffle=True)))

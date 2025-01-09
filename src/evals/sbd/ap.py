#!/usr/bin/env python3

from torcheval.metrics import BinaryAUPRC
from torch.utils.data import DataLoader
from src.data.sbd import Dataset
from src.model.arch import Model
from src import const
import random
import torch
import sys


def average_precision(model, gen):
    metric = BinaryAUPRC()
    for X, (heatmap, y) in gen:
        metric.update(model(X.to(const.DEVICE))[0].detach().flatten(), y.to(const.DEVICE).flatten())
    return metric.compute()


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    torch.multiprocessing.set_start_method('spawn', force=True)
    print(average_precision(model, DataLoader(Dataset(mode='segmentation', image_set=sys.argv[2]), batch_size=const.BATCH_SIZE, num_workers=const.N_WORKERS, shuffle=False)))

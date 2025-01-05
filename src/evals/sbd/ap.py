#!/usr/bin/env python3

from torcheval.metrics.functional import binary_auprc
from torch.utils.data import DataLoader
from src.data.sbd import Dataset
from src.model.arch import Model
from src import const
import random
import torch
import sys


def average_precision(model, gen):
    preds = torch.empty(0, device=const.DEVICE)
    targets = torch.empty(0, device=const.DEVICE)
    for X, (heatmap, y) in gen:
        preds = torch.cat([preds, model(X.to(const.DEVICE))[0].detach()])
        targets = torch.cat([targets, y.to(const.DEVICE)])

    return binary_auprc(preds.flatten(), targets.flatten())


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    torch.multiprocessing.set_start_method('spawn', force=True)
    print(average_precision(model, DataLoader(Dataset(mode='segmentation', image_set=sys.argv[2]), batch_size=const.BATCH_SIZE, num_workers=4, shuffle=False)))

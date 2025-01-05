#!/usr/bin/env python3

from torch.utils.data import DataLoader
from src.data.sbd import Dataset
from src.model.arch import Model
from src import const
import random
import torch
import sys


def iou(model, gen):
    tp = torch.tensor(0, device=const.DEVICE)
    fp_fn = torch.tensor(0, device=const.DEVICE)

    for X, (heatmap, y) in gen:
        pred, cams = model(X)
        pred_map = model.get_semantic_map(cams)

        tp += (pred_map == heatmap).sum()
        fp_fn += (pred_map != heatmap).sum()

    return tp / (tp + fp_fn)


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name

    torch.multiprocessing.set_start_method('spawn', force=True)
    print(iou(model, DataLoader(Dataset(mode='segmentation', image_set=sys.argv[2]), batch_size=const.BATCH_SIZE, num_workers=const.N_WORKERS, shuffle=False)))

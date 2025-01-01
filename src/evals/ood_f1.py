#!/usr/bin/env python3

from torcheval.metrics.functional import multiclass_f1_score
from ..data.soodimagenet import get_generators
from ..model.arch import Model
from src import const
import random
import torch
import sys


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    for test, test_loader in zip(('easy', 'hard'), get_generators('test')):
        preds = torch.empty(0, device=const.DEVICE)
        targets = torch.empty(0, device=const.DEVICE)
        for X, (heatmap, y) in test_loader:
            preds = torch.cat([preds, model(X.to(const.DEVICE))[0].argmax(1)])
            targets = torch.cat([targets, y.to(const.DEVICE).argmax(1)])

            del X, heatmap, y

        print(test, multiclass_f1_score(preds, targets, num_classes=const.N_CLASSES).item())

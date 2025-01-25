#!/usr/bin/env python3

from src.data.soodimagenet import get_generators
from torcheval.metrics import MulticlassF1Score
from src.model.arch import Model
from src import const
import torch
import sys


if __name__ == '__main__':
    name = sys.argv[1]

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name, logits_only=True)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    for test, test_loader in zip(('easy', 'hard'), get_generators('test')):
        metric = MulticlassF1Score(num_classes=const.N_CLASSES)
        for X, (heatmap, y) in test_loader:
            y_pred = model(X.to(const.DEVICE))[0].detach()
            metric.update(y_pred, y.to(const.DEVICE).argmax(1))

        print(test, metric.compute().item())

#!/usr/bin/env python3

from ..data.oxford_iiit_pet import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ..model.arch import Model
from src import const
import random
import torch
import sys


def average_prediction_noise(model, gen, misclassified_only=False):
    pixelwise_influence = torch.zeros(const.CAM_SIZE, device=const.DEVICE)
    avgs = torch.tensor([0., 0., 0.], device=const.DEVICE)
    n_miscls = torch.tensor([0.], device=const.DEVICE)

    for X, y in gen:
        y_pred = model(X)
        cc = model.get_contrastive_cams(y[1], y_pred[1]).detach().abs()
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        if misclassified_only:
            mask = y[1].argmax(1) != y_pred[0].argmax(1)
            cc = cc[mask]
            heatmap = heatmap[mask]
            n_miscls += mask.sum()

        pixelwise_influence += cc.sum(dim=0).sum(dim=0)
        avgs[0] += cc.sum()
        avgs[1] += cc[heatmap == 1].sum()
        avgs[2] += cc[heatmap != 1].sum()

    avgs /= n_miscls if misclassified_only else len(gen.dataset)
    print('\n'.join([f'{feature} Average Influence: {metric}' for (feature, metric) in zip(('Net', 'Foreground', 'Background'), avgs.tolist())]))

    plt.imshow(pixelwise_influence.detach().cpu())
    plt.savefig(const.DATA_DIR / 'evals' / f'{model.name}_influence.png')
    plt.show()


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    average_prediction_noise(model, DataLoader(Dataset('valid'), batch_size=10, shuffle=False), misclassified_only=len(sys.argv) == 3)

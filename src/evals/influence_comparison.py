#!/usr/bin/env python3

from ..data.oxford_iiit_pet import Dataset
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ..model.arch import Model
from src import const
import torchvision
import random
import torch
import sys


def evaluate_influence(model, gen, misclassified_only=False):
    (const.DATA_DIR / 'evals' / 'influence_plots' / gen.dataset.split / model.name).mkdir(exist_ok=True, parents=True)

    const.CAM_SIZE = const.IMAGE_SIZE
    for batch_idx, (X, y) in enumerate(gen):
        y_pred = model(X)
        cc = model.get_contrastive_cams(y[1], y_pred[1]).detach().abs()
        heatmap = y[0].repeat((cc.shape[1] - 1, 1, 1, 1)).permute(1, 0, 2, 3)

        if misclassified_only:
            mask = y[1].argmax(1) != y_pred[0].argmax(1)
            X = X[mask]
            cc = cc[mask]
            heatmap = heatmap[mask]

        for idx, (X_i, cc_i, heatmap_i) in enumerate(zip(X, cc, heatmap)):
            fig = plt.figure()
            cc_i = cc_i.sum(dim=0)

            fig.add_subplot(2, 2, 1)
            plt.imshow(X_i.cpu().permute(1, 2, 0))
            fig.add_subplot(2, 2, 2)
            plt.imshow((X_i * heatmap_i).cpu().permute(1, 2, 0))

            heatmap_i = torchvision.transforms.functional.resize(heatmap_i, (14, 14), antialias=False).squeeze(0)
            norm = Normalize(vmin=cc_i.min(), vmax=cc_i.max())
            foreground = cc_i * heatmap_i
            background = cc_i * (1 - heatmap_i)

            fig.add_subplot(2, 2, 3)
            plt.imshow(foreground.cpu(), norm=norm)
            fig.add_subplot(2, 2, 4)
            plt.imshow(background.cpu(), norm=norm)

            fig.suptitle(f'Net Influence: {cc_i.sum()}\nForeground Influence: {foreground.sum()}\nBackground Influence: {background.sum()}')
            plt.tight_layout()
            fig.savefig(const.DATA_DIR / 'evals' / 'influence_plots' / gen.dataset.split / model.name / f'{batch_idx * gen.batch_size + idx}.png')
            plt.close()


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    evaluate_influence(model, DataLoader(Dataset(sys.argv[2]), batch_size=10, shuffle=False), misclassified_only=len(sys.argv) == 4)

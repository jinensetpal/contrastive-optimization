#!/usr/bin/env python3

from ..data.oxford_iiit_pet import Dataset
from matplotlib.colors import Normalize
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ..model.arch import Model
from src import const
import random
import torch
import sys


def visualize(model, gen, norm=None):
    fig = plt.figure(figsize=(14, 14),
                     facecolor='white')

    for idx, sample in enumerate(random.sample(range(len(gen)), 16)):
        print(sample)
        X, (heatmap, y) = gen[sample]
        y_pred, cam = model(X.unsqueeze(0))
        cam = model.get_contrastive_cams(y.unsqueeze(0), cam).detach()[0, y.argmin(0)].abs()
        fig.add_subplot(4, 4, idx + 1)
        plt.xlabel(f'Pred: {str(y_pred[0].argmax().item())}, Actual: {str(y.argmax().item())}')
        plt.imshow(X.permute(1, 2, 0).cpu().detach(), alpha=0.5)
        plt.imshow(F.interpolate(cam[None, None, :], const.IMAGE_SIZE, mode='bilinear')[0, 0].cpu(), cmap='jet', alpha=0.5, norm=norm)

    plt.tight_layout()
    fig.savefig(const.DATA_DIR / 'evals' / f'{model.name}_cam.png')
    plt.show()


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    if len(sys.argv) == 3 and sys.argv[2] == 'normed': visualize(model, Dataset('train'), Normalize(vmin=-.5, vmax=.5))
    else: visualize(model, Dataset('train'))

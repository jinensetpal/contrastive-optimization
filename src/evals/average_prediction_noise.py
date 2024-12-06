#!/usr/bin/env python3

from ..data.oxford_iiit_pet import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ..model.arch import Model
from src import const
import random
import torch
import sys


def average_prediction_noise(model, gen, failures_only=False):
    pixelwise_influence = torch.zeros(const.CAM_SIZE, device=const.DEVICE)
    avgs = torch.zeros(4, device=const.DEVICE)
    n_miscls = torch.zeros(1, device=const.DEVICE)

    for X, y in gen:
        y_pred = model(X)
        conf = y_pred[0].max(1).values.detach()
        cc = model.get_contrastive_cams(y[1], y_pred[1]).detach().abs()
        heatmap = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)

        if failures_only:
            mask = (y[1].argmax(1) != y_pred[0].argmax(1)) | (y_pred[0].max(dim=1).values < const.CONFIDENCE_THRESHOLD)
            cc = cc[mask]
            conf = conf[mask]
            heatmap = heatmap[mask]
            n_miscls += mask.sum()

        pixelwise_influence += cc.sum(dim=0).sum(dim=0)
        avgs[0] += cc.sum()
        avgs[1] += cc[heatmap == 1].sum()
        avgs[2] += cc[heatmap != 1].sum()
        avgs[3] += conf.sum()

    tot = n_miscls.item() if failures_only else len(gen.dataset)
    avgs /= tot
    print(model.name)
    print('\n'.join([f'{feature} Average Influence: {metric}' for (feature, metric) in zip(('Net', 'Foreground', 'Background'), avgs.tolist()[:-1])]))
    print(f'#images: {tot}\nAverage Confidence: {avgs[-1].item()}')

    plt.imshow(pixelwise_influence.detach().cpu())
    plt.savefig(const.DATA_DIR / 'evals' / f'{model.name.replace("/", "_")}_{gen.dataset.split}_influence.png')
    plt.show()


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    average_prediction_noise(model, DataLoader(Dataset(sys.argv[2]), batch_size=const.BATCH_SIZE, shuffle=False), failures_only=len(sys.argv) == 4)

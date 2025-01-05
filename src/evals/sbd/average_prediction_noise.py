#!/usr/bin/env python3

from torch.utils.data import DataLoader
from src.model.arch import Model
from src.data.sbd import Dataset
from src import const
import random
import torch
import sys


def average_prediction_noise(model, gen, failures_only=False):
    avgs = torch.zeros(4, device=const.DEVICE)
    n_miscls = torch.zeros(1, device=const.DEVICE)

    for X, y in gen:
        y_pred, cams = model(X)
        conf = (y_pred - .5).abs() + .5

        labels = ((torch.arange(const.N_CLASSES) + 1) * torch.ones(*const.CAM_SIZE, const.N_CLASSES)).T[None,].repeat(y[0].size(0), 1, 1, 1).to(const.DEVICE)
        fg_mask = (labels == y[0].repeat(1, const.N_CLASSES, 1).view(y[0].size(0), -1, *y[0].shape[1:])).to(torch.int)

        if failures_only:
            mask = (y[1] != (y_pred > .5).to(torch.int)) | (conf < const.CONFIDENCE_THRESHOLD)
            cams = cams[mask]
            conf = conf[mask]
            fg_mask = fg_mask[mask]
            n_miscls += mask.sum()

        avgs[0] += cams.sum().detach()
        avgs[1] += (fg_mask * cams.abs()).detach().sum()
        avgs[2] += ((1-fg_mask) * cams.abs()).detach().sum()
        avgs[3] += conf.detach().sum()

    tot = n_miscls.item() if failures_only else len(gen.dataset) * const.N_CLASSES
    avgs /= tot
    print(model.name)
    print('\n'.join([f'{feature} Average Influence: {metric}' for (feature, metric) in zip(('Net', 'Foreground', 'Background'), avgs.tolist()[:-1])]))
    print(f'#images: {tot}\nAverage Confidence: {avgs[-1].item()}')


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    torch.multiprocessing.set_start_method('spawn', force=True)
    average_prediction_noise(model, DataLoader(Dataset(mode='segmentation', image_set=sys.argv[2]), batch_size=const.BATCH_SIZE,
                                               num_workers=const.N_WORKERS, shuffle=False), failures_only=len(sys.argv) == 4)

#!/usr/bin/env python3

from torcheval.metrics import BinaryAUPRC
from torch.utils.data import DataLoader
from src.data.sbd import Dataset
from src.model.arch import Model
from src import const
import random
import torch
import sys


# debug imports
from src.model.loss import ContrastiveLoss
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from IPython import embed


def visualize(cams, heatmap, y, y_pred, dataidx=0):
    cams = cams.detach().cpu()
    heatmap = heatmap.detach().cpu()
    plt.imshow(heatmap[dataidx])

    fig = plt.figure(facecolor='white')
    for idx, cam in enumerate(cams[dataidx]):
        fig.add_subplot(4, 5, idx+1)
        plt.imshow(cam.detach().cpu(), norm=Normalize(vmin=cams[0].min(), vmax=cams[0].max()))

    print(y[dataidx].nonzero(), (y_pred[dataidx] > 0).nonzero())
    plt.show()


def average_precision(model, gen, debug=False):
    metric = BinaryAUPRC()
    for X, (heatmap, y) in gen:
        metric.update(model(X)[0].detach().flatten(), y.flatten())
        if debug:
            criterion = ContrastiveLoss(model.get_contrastive_cams, is_label_mask=const.USE_CUTMIX, multilabel=True, divergence=const.DIVERGENCE)  # noqa: F841
            y_pred = model(X)  # noqa: F841
            embed()
    return metric.compute()


if __name__ == '__main__':
    random.seed(const.SEED)
    name = sys.argv[1]

    model = Model(is_contrastive='default' not in name, multilabel=True, xl_backbone=False, logits_only=True)
    model.load_state_dict(torch.load(const.DOWNSTREAM_MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name

    model.disable_batchnorms()
    model.eval()

    torch.multiprocessing.set_start_method('spawn', force=True)
    print(average_precision(model, DataLoader(Dataset(mode='segmentation', image_set=sys.argv[2]), batch_size=const.BATCH_SIZE, num_workers=const.N_WORKERS, shuffle=True), debug=len(sys.argv) == 4))

#!/usr/bin/env python3

from torchvision.transforms.functional import resize, pil_to_tensor
from src.utils import DataLoader
from src import const
import torchvision
import torch


class Dataset(torchvision.datasets.SBDataset):
    def __init__(self, *args, device=const.DEVICE, **kwargs):
        super().__init__(const.DATA_DIR / 'sbd', *args, **kwargs)
        self.device = device
        self.reweight = torch.tensor([17.5546, 20.1393, 13.2824, 22.1553, 14.6790, 24.7515, 8.7566, 9.6625, 9.3382, 37.2793, 20.4596, 8.2069, 23.1420, 21.1302, 1.8195, 20.6234, 32.8566, 21.7828, 20.4055, 18.7628], device=const.DEVICE)  # computed over train to establish a .5 prior of detection for each class

    def compute_reweight_tensor(self):
        n_pos = torch.zeros(const.N_CLASSES, device=const.DEVICE)
        for datapoint in self: n_pos += datapoint[1][1]
        return (len(self) - n_pos) / n_pos

    def __getitem__(self, idx):
        X, heatmap = super().__getitem__(idx)
        y = torch.zeros(const.N_CLASSES)

        heatmap = pil_to_tensor(heatmap)
        y[heatmap.flatten().bincount()[1:].nonzero()] = 1
        heatmap = resize(heatmap, const.CAM_SIZE, interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT)[0]

        return pil_to_tensor(resize(X, const.IMAGE_SIZE)).to(self.device) / 255, (heatmap.to(self.device), y.to(self.device))


def get_generators():
    torch.multiprocessing.set_start_method('spawn', force=True)
    const.SPLITS[1] = 'val'

    dataloaders = *[DataLoader(Dataset(mode='segmentation', image_set=split, device='cpu', download=not (const.DATA_DIR / 'sbd').exists()),
                               num_workers=const.N_WORKERS, pin_memory=True, batch_size=const.BATCH_SIZE, shuffle=True) for split in const.SPLITS[:2]], None
    const.SPLITS[1] = 'valid'
    return dataloaders


if __name__ == '__main__':
    print(Dataset(mode='segmentation', image_set='train')[0])

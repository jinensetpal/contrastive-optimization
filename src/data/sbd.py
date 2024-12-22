#!/usr/bin/env python3

from torchvision.transforms.functional import resize, pil_to_tensor
from src import const
import torchvision
import torch


class Dataset(torchvision.datasets.SBDataset):
    def __init__(self, *args, device=const.DEVICE, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __getitem__(self, idx):
        X, heatmap = super().__getitem__(idx)
        y = torch.zeros(const.N_CLASSES)

        heatmap = pil_to_tensor(heatmap)
        y[heatmap.flatten().bincount()[1:].nonzero()] = 1
        heatmap = resize(heatmap, const.CAM_SIZE, antialias=False, interpolation=torchvision.transforms.InterpolationMode.NEAREST)[0]

        return pil_to_tensor(resize(X, const.CAM_SIZE, antialias=False)).to(self.device) / 255, (heatmap.to(self.device), y.to(self.device))


def get_generators():
    torch.multiprocessing.set_start_method('spawn', force=True)
    const.SPLITS[1] = 'val'

    dataloaders = *[torch.utils.data.DataLoader(Dataset(const.DATA_DIR / 'sbd', mode='segmentation', image_set=split, device='cpu', download=True),
                                                num_workers=2, pin_memory=True, batch_size=const.BATCH_SIZE, shuffle=True) for split in const.SPLITS[:2]], None
    const.SPLITS[1] = 'valid'
    return dataloaders


if __name__ == '__main__':
    print(Dataset(const.DATA_DIR / 'sbd', mode='segmentation')[0])

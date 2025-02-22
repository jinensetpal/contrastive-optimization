#!/usr/bin/env python3

from torch.utils.data import DistributedSampler
from torchvision.ops import masks_to_boxes
import torchvision.transforms.v2 as T
from src.utils import DataLoader
from src import const
import pandas as pd
import torchvision
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split=None, bbox=False, device=const.DEVICE):
        self.images_dir = const.DATA_DIR / 'oxford-iiit-pet' / 'images'
        self.annotations_dir = const.DATA_DIR / 'oxford-iiit-pet' / 'annotations'
        self.y_col = 'SPECIES' if const.BINARY_CLS else 'CLASS-ID'
        self.device = device
        self.bbox = bbox

        df = pd.read_csv(self.annotations_dir / 'list.txt')
        if split:
            self.df = df[df['split'] == split].reset_index()
            self.split = split
        else:
            self.df = df
            self.split = 'all'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = torchvision.transforms.functional.resize(torchvision.io.read_image((self.images_dir / f'{self.df["Image"].iloc[idx]}.jpg').as_posix()), const.IMAGE_SIZE, antialias=True)[:3]
        X = X / 255  # normalization

        heatmap = torchvision.io.read_image((self.annotations_dir / 'trimaps' / f'{self.df["Image"].iloc[idx]}.png').as_posix())
        heatmap = heatmap.to(torch.float)
        heatmap[heatmap == 3] = 0  # set boundary points to background; saliency map demonstrates influence beyond downsampled target region
        heatmap[heatmap == 2] = 0  # set background to 0 (for optimization objective)
        heatmap = T.functional.resize(heatmap, const.CAM_SIZE, antialias=False, interpolation=T.InterpolationMode.NEAREST_EXACT).squeeze(0)

        if self.bbox and heatmap.min() != heatmap.max():
            bbox = masks_to_boxes(heatmap[None, ])[0].to(torch.int).tolist()
            heatmap[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = 1.

        y = torch.zeros((const.N_CLASSES))
        y[self.df[self.y_col][idx] - 1] = 1

        return X.to(self.device), (heatmap.to(self.device), y.to(self.device))


def get_generators():
    datasets = [Dataset(split=split, device='cpu', bbox=const.BBOX_MAP) for split in const.SPLITS]
    return [DataLoader(dataset, num_workers=const.N_WORKERS, sampler=DistributedSampler(dataset) if const.DDP else None,
                       pin_memory=True, batch_size=const.BATCH_SIZE, shuffle=None if const.DDP else True) for dataset in datasets]


if __name__ == '__main__':
    print(Dataset()[0])

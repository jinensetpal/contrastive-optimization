#!/usr/bin/env python3

from torchvision.ops import masks_to_boxes
from src import const
import pandas as pd
import torchvision
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split=None, bbox=False):
        self.images_dir = const.DATA_DIR / 'oxford-iiit-pet' / 'images'
        self.annotations_dir = const.DATA_DIR / 'oxford-iiit-pet' / 'annotations'
        self.y_col = 'SPECIES' if const.BINARY_CLS else 'CLASS-ID'
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
        heatmap[heatmap == 3] = 1  # set boundary points to target; downsampling requires greater margin of approximation
        heatmap[heatmap == 2] = 0  # set background to 0 (for optimization objective)
        heatmap = torchvision.transforms.functional.resize(heatmap, const.CAM_SIZE, antialias=False).squeeze(0)

        if self.bbox:
            bbox = masks_to_boxes(heatmap[None, ])[0].to(torch.int).tolist()
            heatmap[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = 1.

        y = torch.zeros((const.N_CLASSES))
        y[self.df[self.y_col][idx] - 1] = 1

        return X.to(const.DEVICE), (heatmap.to(const.DEVICE), y.to(const.DEVICE))


def get_generators():
    return [torch.utils.data.DataLoader(Dataset(split=split, bbox=const.BBOX_MAP), batch_size=const.BATCH_SIZE if split == 'train' else const.EVAL_BATCH_SIZE, shuffle=True) for split in const.SPLITS]


if __name__ == '__main__':
    print(Dataset()[0])

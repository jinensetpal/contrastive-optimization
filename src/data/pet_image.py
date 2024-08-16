#!/usr/bin/env python3

from src import const
from glob import glob
import torchvision
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = glob((const.DATA_DIR / 'pet-images' / '**' / '*.jpg').as_posix(), recursive=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = torchvision.transforms.functional.resize(torchvision.io.read_image(self.images[idx]), const.IMAGE_SIZE, antialias=True)[:3]
        X = X / 255  # normalization

        y = torch.zeros((2))
        y[int('cat' in self.images[idx])] = 1

        return X.to(const.DEVICE), y.to(const.DEVICE)


def get_generator():
    return torch.utils.data.DataLoader(Dataset(), batch_size=const.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    print(Dataset()[0])

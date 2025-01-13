#!/usr/bin/env python3

from torchvision.transforms.functional import resize
from glob import glob
from src import const
import torchvision
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, device=const.DEVICE):

        self.device = device
        self.images = glob((const.DATA_DIR / 'VOC2012Test' / '**' / '*').as_posix(), recursive=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.images[idx]).to(self.device)
        return resize(img, const.IMAGE_SIZE, antialias=True) / 255, list(img.shape[1:])


def get_generator():
    torch.multiprocessing.set_start_method('spawn', force=True)
    return torch.utils.data.DataLoader(Dataset(device='cpu'), num_workers=const.N_WORKERS, pin_memory=True, batch_size=const.BATCH_SIZE, shuffle=False)


if __name__ == '__main__':
    print(Dataset()[0])

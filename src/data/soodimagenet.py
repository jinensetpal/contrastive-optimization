#!/usr/bin/env python3

from torchvision.transforms.functional import resize
from utils.SOODImageNetDataset import SOODImageNetS
from torch.utils.data import DistributedSampler
from src.utils import DataLoader
from src import const
import torchvision
import random
import torch


class Dataset(SOODImageNetS):
    def __init__(self, mode, device=const.DEVICE):
        dataset_path = const.DATA_DIR / 'soodimagenet'

        if mode == 'train_val': lists_file = dataset_path / 'lists' / 'segmentation' / 'train_iid.txt'
        elif mode == 'test_easy': lists_file = dataset_path / 'lists' / 'segmentation' / 'test_easy_ood.txt'
        else: lists_file = dataset_path / 'lists' / 'segmentation' / 'test_hard_ood.txt'

        with open(lists_file, 'r') as f: images = f.read().splitlines()

        super().__init__(images,
                         mask_base_path=dataset_path / 'masks',
                         base_path=dataset_path / 'images',
                         mode='train_val' if 'train' in mode else 'test',
                         resize=const.IMAGE_SIZE)
        self.device = device

    def __getitem__(self, idx):
        X, heatmap, label_idx, class_name, synset = super().__getitem__(idx)
        heatmap[heatmap.nonzero()] = 1
        heatmap = resize(heatmap[None,], const.CAM_SIZE, interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT)[0]

        y = torch.zeros(const.N_CLASSES)
        y[int(label_idx) - 1] = 1

        return X.to(self.device), (heatmap.to(self.device), y.to(self.device))


def get_generators(split):
    random.seed(const.SEED)

    if split == 'train':
        dataset = Dataset('train_val', device='cpu')
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        datasets = torch.utils.data.random_split(dataset, [train_size, val_size])
    else: datasets = Dataset('test_easy', device='cpu'), Dataset('test_hard', device='cpu')

    return *[DataLoader(dataset, num_workers=const.N_WORKERS, pin_memory=True, sampler=DistributedSampler(dataset) if const.DDP else None,
                        batch_size=const.BATCH_SIZE, shuffle=None if const.DDP else split == 'train') for dataset in datasets], None


if __name__ == '__main__':
    print(Dataset(mode='train_val')[0])

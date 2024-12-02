#!/usr/bin/env python3

from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import resize
import xml.etree.ElementTree as ET
from src import const, utils
import pandas as pd
import torchvision
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split=None, bbox=True):
        df = pd.read_csv(const.DATA_DIR / 'imagenet' / 'index.csv')
        if bbox: df = df[df['bbox_exists']].reset_index()
        if split:
            self.df = df[df['split'] == split].reset_index()
            self.split = split
        else:
            self.df = df
            self.split = 'all'

        if self.split == 'train': self.transforms = utils.ClassificationPresetTrain(crop_size=const.TRAIN_CROP_SIZE,
                                                                                    auto_augment_policy=const.AUTO_AUGMENT_POLICY,
                                                                                    random_erase_prob=const.RANDOM_ERASE_PROB)
        else: self.transforms = utils.ClassificationPresetEval(crop_size=const.VAL_CROP_SIZE,
                                                               resize_size=const.VAL_RESIZE_SIZE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = torchvision.io.read_image((const.DATA_DIR / 'imagenet' / self.df['path'].iloc[idx]).as_posix())
        if X.size(0) == 1: X = X.repeat(3, 1, 1)

        heatmap = torch.zeros(X.shape[1:])
        X = resize(X, const.IMAGE_SIZE, antialias=True)[:3]

        if const.BBOX_MAP:
            bbox = ET.parse(const.DATA_DIR / 'imagenet' / self.df['bbox'].iloc[idx]).getroot()
            for obj in bbox.findall('object'):
                box = [int(x.text) for x in obj.find('bndbox')[:]]
                heatmap[box[1]:box[3], box[0]:box[2]] = 1.

            heatmap = resize(heatmap[None, None], const.IMAGE_SIZE, antialias=True)
        X, heatmap = self.transforms([X[None,], heatmap])

        heatmap = resize(heatmap.mean(dim=1), const.CAM_SIZE, antialias=False).squeeze(0)

        if const.USE_CUTMIX:
            heatmap[heatmap <= 0] = -1
            heatmap[heatmap > 0] = int(self.df['label_idx'][idx])   # set to class labels for cutmix
        else:
            heatmap[heatmap < 0] = 0
            heatmap[heatmap > 0] = 1

        y = torch.zeros(const.N_CLASSES)
        y[int(self.df['label_idx'][idx])] = 1

        return X[0], (heatmap, y)


def collate_fn(batch):
    return utils.CutMix(alpha=const.CUTMIX_ALPHA, num_classes=const.N_CLASSES, labels_getter=lambda x: x[1][1])(*default_collate(batch)) if const.USE_CUTMIX else default_collate(batch)


def get_generators():
    torch.multiprocessing.set_start_method('spawn', force=True)
    const.SPLITS[1] = 'val'

    datasets = [Dataset(split=split, bbox=const.BBOX_MAP) for split in const.SPLITS[:2]]
    samplers = [utils.RASampler(dataset, shuffle=True, repetitions=const.AUGMENT_REPITIONS) for dataset in datasets]
    dataloaders = *[torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, sampler=sampler, num_workers=2,
                                                batch_size=const.BATCH_SIZE if split == 'train' else const.EVAL_BATCH_SIZE)
                    for dataset, sampler, split in zip(datasets, samplers, const.SPLITS[:2])], None
    const.SPLITS[1] = 'valid'
    return dataloaders


if __name__ == '__main__':
    print(Dataset()[0])

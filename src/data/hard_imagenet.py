#!/usr/bin/env python3

# code adapted from: https://github.com/mmoayeri/HardImageNet/blob/main/datasets/hard_imagenet.py, https://github.com/mmoayeri/HardImageNet/blob/main/augmentations.py

import torchvision.transforms.v2 as T
from src.utils import DataLoader
from src import const
from PIL import Image
import pickle
import random
import torch
import glob
import os

_IMAGENET_ROOT = const.DATA_DIR / 'imagenet' / 'images'
_MASK_ROOT = const.DATA_DIR / 'hardimagenet'

with open(_MASK_ROOT/'meta/idx_to_wnid.pkl', 'rb') as f:
    idx_to_wnid = pickle.load(f)
wnid_to_idx = dict({v: k for k, v in idx_to_wnid.items()})
with open(_MASK_ROOT/'meta/hard_imagenet_idx.pkl', 'rb') as f:
    inet_idx = pickle.load(f)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split='val', ft=False, balanced_subset=True, device=const.DEVICE, eval_mode=False):
        '''
        Returns original ImageNet index when ft is False, otherwise returns label between 0 and 14
        '''
        self.split = split
        self.device = device
        self.balanced_subset = balanced_subset
        self.collect_mask_paths()
        # self.mask_paths = glob.glob(_MASK_ROOT + split+'/*')
        self.num_classes = 15
        self.eval_mode = eval_mode
        self.ft = ft

    def standard_resize_center_crop(self, img, mask, resize_shape=const.IMAGE_SIZE):
        t = T.Compose([T.Resize(resize_shape), T.CenterCrop(224), T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        img, mask = [t(x) for x in [img, mask]]
        if img.shape[0] == 1:
            img = torch.cat([img, img, img], axis=0)
        if mask.shape[0] == 1:
            mask = torch.cat([mask, mask, mask], axis=0)
        return img, mask

    def map_wnid_to_label(self, wnid):
        ind = wnid_to_idx[wnid]
        if self.ft:
            ind = inet_idx.index(ind)
        return ind

    def collect_mask_paths(self):
        if self.balanced_subset and self.split == 'train':
            # hard coded for now
            self.subset_size = 100

            with open(_MASK_ROOT / 'meta' / 'paths_by_rank.pkl', 'rb') as f:
                ranked_paths = pickle.load(f)
            paths = []
            for c in ranked_paths:
                cls_paths = ranked_paths[c]
                paths += cls_paths[:self.subset_size] + cls_paths[(-1*self.subset_size):]
            self.mask_paths = [(_MASK_ROOT/'train'/'_'.join(p.split('/')[-2:])).as_posix() for p in paths]
            for p in self.mask_paths:
                if not os.path.exists(p):
                    self.mask_paths.remove(p)
        else:
            self.mask_paths = glob.glob((_MASK_ROOT / self.split / '*').as_posix())

    def __getitem__(self, ind):
        mask_path = self.mask_paths[ind]
        mask_path_suffix = mask_path.split('/')[-1]
        wnid = mask_path_suffix.split('_')[0]
        fname = mask_path_suffix[len(wnid)+1:]  # if self.split == 'val' else mask_path_suffix

        img_path = os.path.join(_IMAGENET_ROOT, self.split, wnid, fname)
        img, mask = [Image.open(p) for p in [img_path, mask_path]]

        img, mask = self.standard_resize_center_crop(img, mask)

        if img.shape[0] > 3:  # weird bug
            img, mask = [x[:3] for x in [img, mask]]

        class_ind = self.map_wnid_to_label(wnid)
        mask[mask > 0] = 1

        if not self.eval_mode:
            mask = T.functional.resize(mask[0][None,], const.CAM_SIZE, interpolation=T.InterpolationMode.NEAREST)[0]
            y = torch.zeros(const.N_CLASSES, device=self.device)
            y[class_ind] = 1

            return img.to(self.device), (mask.to(self.device), y.to(self.device))
        else: return img.to(self.device), mask.to(self.device), class_ind

    def __len__(self):
        return len(self.mask_paths)


def get_generators():
    random.seed(const.SEED)
    const.SPLITS[1] = 'val'

    dataloaders = *[DataLoader(Dataset(split=split, ft=True, device='cpu'), shuffle=True,
                               num_workers=const.N_WORKERS, pin_memory=True, batch_size=const.BATCH_SIZE) for split in const.SPLITS[:2]], None

    const.SPLITS[1] = 'valid'
    return dataloaders


if __name__ == '__main__':
    print(Dataset('train')[0])

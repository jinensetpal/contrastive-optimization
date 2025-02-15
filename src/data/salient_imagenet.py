#!/usr/bin/env python3

from .hard_imagenet import Dataset as hardimagenet
from src.utils import DataLoader, trim_mask
import torchvision.transforms.v2 as T
from PIL import Image
from src import const
import numpy as np
import random
import torch
import json


# code adapted from: https://github.com/singlasahil14/salient_imagenet/blob/main/inspection_utils.py
class Dataset(torch.utils.data.Dataset):
    def __init__(self, trim_masks=False, mask_threshold=const.SAL_INET_MASK_THRESHOLD, resize_size=256, crop_size=const.IMAGE_SIZE, device=const.DEVICE):
        self.transform = T.Compose([
            T.Resize(resize_size),
            T.CenterCrop(crop_size),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

        self.dataset_path = const.DATA_DIR / 'salientimagenet'
        self.mask_threshold = mask_threshold
        self.trim_masks = trim_masks
        self.device = device

        self.core_features_dict = json.load(open(self.dataset_path / 'core_features_dict.json', 'r'))
        self.wordnet_dict = json.load(open(self.dataset_path / 'wordnet_dict.json', 'r'))

        images_class_dict = json.load(open(self.dataset_path / 'image_class_dict.json', 'r'))
        self.image_names = list(images_class_dict.keys())
        self.image_labels = list(images_class_dict.values())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        curr_image_path = const.DATA_DIR / 'imagenet' / 'images' / 'train' / self.wordnet_dict[str(self.image_labels[index])] / (image_name + '.JPEG')

        image = Image.open(curr_image_path).convert("RGB")
        image_tensor = self.transform(image)

        all_mask = np.zeros(image_tensor.shape[1:])
        for feature_index in self.core_features_dict[str(self.image_labels[index])]:
            curr_mask_path = self.dataset_path / self.wordnet_dict[str(self.image_labels[index])] / ('feature_' + str(feature_index)) / (image_name + '.JPEG')
            if not curr_mask_path.exists(): continue

            mask = np.asarray(Image.open(curr_mask_path))
            mask = (mask/255.)

            all_mask = np.maximum(all_mask, mask)

        all_mask = np.uint8(all_mask * 255)
        all_mask = Image.fromarray(all_mask)
        mask_tensor = self.transform(all_mask)

        mask_tensor = trim_mask(mask_tensor[0], const.CAM_SIZE, reduce_factor=const.HARD_INET_TRIM_FACTOR, center_bias=const.HARD_INET_CENTER_BIAS) if self.trim_masks else T.functional.resize(mask_tensor, const.CAM_SIZE, interpolation=T.InterpolationMode.NEAREST_EXACT)[0]

        if self.mask_threshold is not None: mask_tensor = (mask_tensor > self.mask_threshold).to(torch.float32)

        y = torch.zeros(const.N_CLASSES, device=self.device)
        y[self.image_labels[index]] = 1

        return image_tensor.to(self.device), (mask_tensor.to(self.device), y.to(self.device))


# complete salient-imagenet, supplemented 15 classes containing spurious-only features from hard imagenet
def get_generators():
    random.seed(const.SEED)
    const.SPLITS[1] = 'val'

    dataset = Dataset(device='cpu')
    train_len = int(len(dataset)*.95)
    salient_subsets = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    dataloaders = *[DataLoader(torch.utils.data.ConcatDataset([hardimagenet(split=split, ft=False, balanced_subset=False, trim_masks=const.HARD_INET_TRIM_MASKS, device='cpu'), salient_subset]),
                               shuffle=True, num_workers=const.N_WORKERS, pin_memory=True, batch_size=const.BATCH_SIZE) for split, salient_subset in zip(const.SPLITS[:2], salient_subsets)], None

    const.SPLITS[1] = 'valid'
    return dataloaders


if __name__ == '__main__':
    data = Dataset()
    print(len(data))

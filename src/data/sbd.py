#!/usr/bin/env python3

from torchvision.transforms.functional import resize, pil_to_tensor
from src import const
import torchvision
import torch


class Dataset(torchvision.datasets.SBDataset):
    def __init__(self, *args, device=const.DEVICE, **kwargs):
        super().__init__(const.DATA_DIR / 'sbd', *args, **kwargs)
        self.device = device
        self.reweight = torch.tensor([9.2773, 10.5697, 7.1412, 11.5777, 7.8395, 12.8758, 4.8783, 5.3312, 5.1691, 19.1396, 10.7298, 4.6035, 12.0710, 11.0651, 1.4098, 10.8117, 16.9283, 11.3914, 10.7028, 9.8814], device=const.DEVICE)  # computed over train to establish a .5 prior of detection for each class

    def __getitem__(self, idx):
        X, heatmap = super().__getitem__(idx)
        y = torch.zeros(const.N_CLASSES)

        heatmap = pil_to_tensor(heatmap)
        y[heatmap.flatten().bincount()[1:].nonzero()] = 1
        heatmap = resize(heatmap, const.CAM_SIZE, interpolation=torchvision.transforms.InterpolationMode.NEAREST)[0]

        return pil_to_tensor(resize(X, const.IMAGE_SIZE)).to(self.device) / 255, (heatmap.to(self.device), y.to(self.device))


def get_generators():
    torch.multiprocessing.set_start_method('spawn', force=True)
    const.SPLITS[1] = 'val'

    dataloaders = *[torch.utils.data.DataLoader(Dataset(mode='segmentation', image_set=split, device='cpu', download=not (const.DATA_DIR / 'sbd').exists()),
                                                num_workers=const.N_WORKERS, pin_memory=True, batch_size=const.BATCH_SIZE, shuffle=True) for split in const.SPLITS[:2]], None
    const.SPLITS[1] = 'valid'
    return dataloaders


if __name__ == '__main__':
    print(Dataset(const.DATA_DIR / 'sbd', mode='segmentation')[0])

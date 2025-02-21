#!/usr/bin/env python3

from src.data.oxford_iiit_pet import Dataset
from torch.utils.data import DataLoader
from ..model.arch import Model
from src import const
import torch
import sys


if __name__ == '__main__':
    name = sys.argv[1]

    model = Model(is_contrastive='default' not in name, modified_bn=True)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.train()

    torch.multiprocessing.set_start_method('spawn', force=True)
    model.update_tracked_statistics(DataLoader(Dataset('train'), batch_size=const.BATCH_SIZE, num_workers=const.N_WORKERS, shuffle=True))
    torch.save(model.state_dict(), const.MODELS_DIR / f'{name}_updated_tracked_statistics.pt')

#!/usr/bin/env python3

from src.data.pascal_voc import get_generator
from torchvision.io import write_png
from src.model.arch import Model
from src import const
import random
import torch
import sys


CLASSES = 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'


def segmentation(model, gen):
    path = const.DATA_DIR / 'evals' / 'VOC2012Test' / 'Segmentation'
    (path).mkdir(exist_ok=True, parents=True)

    images = gen.dataset.images
    for batch_idx, X in enumerate(gen):
        for idx, semantic_map in enumerate(model.get_semantic_map(model(X.to(const.DEVICE))[1].detach())):
            write_png(semantic_map[None,].to('cpu'), path / f'{images[batch_idx * X.size(0) + idx].split("/")[-1][:-3]}png')


def classification(model, gen):
    path = const.DATA_DIR / 'evals' / 'OC2012Test' / 'Main'
    (path).mkdir(exist_ok=True, parents=True)

    logits = torch.empty(0, device=const.DEVICE)
    for X in gen: logits = torch.cat([logits, model(X.to(const.DEVICE))[0].detach()])

    for classname, class_logit in zip(CLASSES, logits.T):
        with open(path / f'comp1_cls_test_{classname}.txt', 'w') as file:
            for image, image_logit in zip(gen.dataset.images, class_logit):
                file.write(' '.join([image.split('/')[-1][:-4], str(image_logit.item())]) + '\n')


if __name__ == '__main__':
    name = sys.argv[1]
    random.seed(const.SEED)

    model = Model(input_shape=const.IMAGE_SHAPE, is_contrastive='default' not in name)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name

    segmentation(model, get_generator())
    classification(model, get_generator())

#!/usr/bin/env python3

from torchvision.transforms.functional import resize, InterpolationMode
from src.data.pascal_voc import get_generator
from torchvision.io import write_png
from src.model.arch import Model
from src import const
import torch
import sys


CLASSES = 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'


# function adapted heavily from: https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = torch.zeros((N, 3), dtype=torch.float32 if normalized else torch.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = torch.tensor([r, g, b])

    return cmap / 255 if normalized else cmap


def segmentation(model, gen):
    path = const.DATA_DIR / 'evals' / 'VOC2012Test' / 'Segmentation' / 'comp5_test_cls'
    (path).mkdir(exist_ok=True, parents=True)

    cmap = color_map()
    images = gen.dataset.images
    for batch_idx, (X, X_sizes) in enumerate(gen):
        for idx, semantic_map in enumerate(model.get_semantic_map(model(X.to(const.DEVICE))[1].detach())):
            write_png(resize(cmap[[semantic_map.cpu().flatten().tolist()]].unflatten(dim=0, sizes=const.CAM_SIZE).T, (X_sizes[0][idx], X_sizes[1][idx]), interpolation=InterpolationMode.NEAREST_EXACT), path / f'{images[batch_idx * X.size(0) + idx].split("/")[-1][:-3]}png')


def classification(model, gen):
    path = const.DATA_DIR / 'evals' / 'VOC2012Test' / 'Main'
    (path).mkdir(exist_ok=True, parents=True)

    model.probabilities = torch.nn.Identity()

    logits = torch.empty(0, device=const.DEVICE)
    for (X, X_size) in gen: logits = torch.cat([logits, model(X.to(const.DEVICE))[0].detach()])

    for classname, class_logits in zip(CLASSES, logits.T):
        with open(path / f'comp1_cls_test_{classname}.txt', 'w') as file:
            for image, image_logit in zip(gen.dataset.images, class_logits):
                file.write(' '.join([image.split('/')[-1][:-4], str(int(image_logit.item() * 1E6) / 1E6)]) + '\n')


if __name__ == '__main__':
    name = sys.argv[1]

    model = Model(is_contrastive='default' not in name, multilabel=True, xl_backbone=False)
    model.load_state_dict(torch.load(const.DOWNSTREAM_MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    segmentation(model, get_generator())
    classification(model, get_generator())

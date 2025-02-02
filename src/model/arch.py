#!/usr/bin/env python3

from src import const
from torch import nn
import torchvision
import torch


class Model(nn.Module):
    def __init__(self, randomized_flatten=const.RANDOMIZED_FLATTEN, multilabel=False, logits_only=False, disable_bn=const.DISABLE_BN,
                 hardinet_eval=False, xl_backbone=const.XL_BACKBONE, device=const.DEVICE, is_contrastive=True, downsampling_level=1):
        super().__init__()

        self.segmentation_threshold = const.SEGMENTATION_THRESHOLD
        self.randomized_flatten = const.RANDOMIZED_FLATTEN
        self.is_contrastive = is_contrastive
        self.hardinet_eval = hardinet_eval
        self.disable_bn = disable_bn
        self.device = device

        if xl_backbone: self.backbone = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2 if const.PRETRAINED_BACKBONE else None)
        else: self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if const.PRETRAINED_BACKBONE else None)

        if downsampling_level >= 1:
            self.backbone.layer4[0].conv2.stride = (1, 1)
            self.backbone.layer4[0].downsample[0].stride = (1, 1)
        if downsampling_level >= 2:
            self.backbone.conv1.stride = (1, 1)
            self.backbone.maxpool.stride = 1

            self.backbone.layer2[0].conv2.stride = (1, 1)
            self.backbone.layer2[0].downsample[0].stride = (1, 1)

            self.backbone.layer3[0].conv2.stride = (1, 1)
            self.backbone.layer3[0].downsample[0].stride = (1, 1)

        if is_contrastive:
            self.backbone.layer4[-1].bn3 = nn.Identity()
            self.backbone.layer4[-1].relu = nn.Identity()
        self.backbone.layer4[-1].conv3.register_forward_hook(self._hook)

        self.linear = nn.Linear(2048, const.N_CLASSES, bias=not is_contrastive)
        self.probabilities = nn.Identity() if logits_only else nn.Softmax(dim=1) if not multilabel else nn.Sigmoid()

        if const.DATASET == 'imagenet' and const.PRETRAINED_BACKBONE:
            self.linear.weight = self.backbone.fc.weight
            if not is_contrastive: self.linear.bias = self.backbone.fc.bias
        self.backbone.fc = nn.Identity()

        self.to(self.device)
        self.disable_batchnorms()

    def disable_batchnorms(self):
        for x in self.modules():
            if x._get_name() == 'BatchNorm2d': x.eval()

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

        if self.disable_bn: self.disable_batchnorms()
        return self

    def _hook(self, model, i, o):
        def assign(grad):
            self.feature_grad = grad
        self.feature_rect = o
        o.register_hook(assign)

    def forward(self, x):
        x = self.backbone(x)
        if self.training and self.randomized_flatten: x = x[:, torch.randperm(x.shape[1])]
        logits = self.linear(x)

        if self.hardinet_eval: return logits
        return logits if self.training else self.probabilities(logits), self._bp_free_hi_res_cams()

    def get_semantic_map(self, cams):
        segmentation_map = cams.max(1)
        return (segmentation_map.values > self.segmentation_threshold).to(torch.uint8) * (segmentation_map.indices + 1).to(torch.uint8)

    def get_contrastive_cams(self, y, cams):
        return torch.index_select(cams.view(-1, *cams.shape[2:]), 0, y.argmax(1) + (torch.arange(cams.size(0), device=const.DEVICE) * cams.size(1))).repeat(1, cams.size(1), 1).view(*cams.shape) - cams

    def _bp_free_hi_res_cams(self):  # required to obtain gradients on self.linear.weight
        return (self.linear.weight @ self.feature_rect.flatten(2)).unflatten(2, self.feature_rect.shape[2:]) / self.feature_rect.shape[-1]**2

    def _hi_res_cams(self, logits):  # inefficient but more general; not restricted to single dense layer
        cams = torch.zeros(*logits.shape, *self.feature_rect.shape[2:], device=self.device)
        for img_idx in range(logits.shape[0]):
            for class_idx in range(logits.shape[1]):
                logits[img_idx, class_idx].backward(retain_graph=True, inputs=self.backbone.layer4[-1].conv3.weight)
                cams[img_idx, class_idx] = (self.feature_rect * self.feature_grad).sum(dim=1)[img_idx]

        self.feature_grad = None
        self.feature_rect = None

        return cams


if __name__ == '__main__':
    model = Model()
    model.eval()

    x = torch.rand((3, *const.IMAGE_SHAPE)).to(const.DEVICE)
    y, cam = model(x)
    print(cam.shape)

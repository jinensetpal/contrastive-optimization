#!/usr/bin/env python3

from src import const
from torch import nn
import torchvision
import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape, is_contrastive=True):
        super().__init__()

        self.is_contrastive = is_contrastive
        self.backbone = torchvision.models.resnet50(weights=None)
        self.backbone.layer4[0].conv2 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.layer4[0].downsample[0] = torch.nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

        if is_contrastive:
            self.backbone.layer4[-1].bn3 = nn.Identity()
            self.backbone.layer4[-1].relu = nn.Identity()
        self.backbone.layer4[-1].conv3.register_forward_hook(self._hook)
        self.backbone.fc = nn.Identity()

        self.linear = nn.LazyLinear(const.N_CLASSES, bias=not is_contrastive)
        self.softmax = nn.Softmax(dim=1)  # ~equivalent to sigmoid since classes = 2; relevant for CAMs

        self.to(const.DEVICE)
        self(torch.randn(1, *input_shape).to(const.DEVICE))  # initialization

    def _hook(self, model, i, o):
        def assign(grad):
            self.feature_grad = grad
        self.feature_rect = o
        o.register_hook(assign)

    def forward(self, x):
        x = self.backbone(x)
        logits = self.linear(x)

        return self.softmax(logits), self._compute_hi_res_cam(logits)

    @staticmethod
    def get_contrastive_cams(y, cams):
        contrastive_cams = torch.empty((y.shape[0], cams.shape[-3], *cams.shape[-2:]), device=const.DEVICE)
        for idx, (cam, y_idx) in enumerate(zip(cams, y.argmax(dim=1))): contrastive_cams[idx] = cam[y_idx] - cam

        return contrastive_cams

    def _compute_hi_res_cam(self, logits):
        cams = torch.zeros(*logits.shape, *self.feature_rect.shape[2:], device=const.DEVICE)
        for img_idx in range(logits.shape[0]):
            for class_idx in range(logits.shape[1]):
                logits[img_idx, class_idx].backward(retain_graph=True, inputs=self.backbone.layer4[-1].conv3.weight)
                cams[img_idx, class_idx] = (self.feature_rect * self.feature_grad).sum(dim=1)[img_idx]

        self.feature_grad = None
        self.feature_rect = None

        return cams


if __name__ == '__main__':
    model = Model(input_shape=const.IMAGE_SHAPE)
    model.eval()

    x = torch.rand((3, *const.IMAGE_SHAPE)).to(const.DEVICE)
    y, cam = model(x)
    print(cam.shape)

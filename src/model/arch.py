#!/usr/bin/env python3

from src import const
from torch import nn
import torchvision
import torch


class Model(nn.Module):
    def __init__(self, input_shape, is_contrastive=True, no_downsampling=False):
        super().__init__()

        self.is_contrastive = is_contrastive
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if const.PRETRAINED_BACKBONE else None)

        if no_downsampling:
            self.backbone.conv1.stride = (1, 1)
            self.backbone.maxpool.stride = 1

            self.backbone.layer2[0].conv2.stride = (1, 1)
            self.backbone.layer2[0].downsample[0].stride = (1, 1)

            self.backbone.layer3[0].conv2.stride = (1, 1)
            self.backbone.layer3[0].downsample[0].stride = (1, 1)

        # one ablated downsampling is required for (224, 224) input
        self.backbone.layer4[0].conv2.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)

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

        if self.training: return logits, self._bp_free_hi_res_cams(logits)
        else: return self.softmax(logits), self._hi_res_cams(logits)

    @staticmethod
    def get_contrastive_cams(y, cams):
        contrastive_cams = torch.empty((y.shape[0], cams.shape[-3], *cams.shape[-2:]), device=const.DEVICE)
        for idx, (cam, y_idx) in enumerate(zip(cams, y.argmax(dim=1))): contrastive_cams[idx] = cam[y_idx] - cam

        return contrastive_cams

    def _bp_free_hi_res_cams(self, logits):  # required to obtain gradients on self.linear.weight
        cams = torch.zeros(*logits.shape, *self.feature_rect.shape[2:], device=const.DEVICE)
        for img_idx in range(logits.shape[0]):
            for class_idx, weight in enumerate(self.linear.weight):
                cams[img_idx, class_idx] = (weight[None, None].repeat(14, 14, 1).permute(2, 0, 1) * self.feature_rect[img_idx]).sum(dim=0)
        cams /= 14**2

        self.feature_rect = None

        return cams

    def _hi_res_cams(self, logits):
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

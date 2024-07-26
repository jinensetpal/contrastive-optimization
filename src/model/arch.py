#!/usr/bin/env python3

from src import const
from torch import nn
import torchvision
import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.backbone = torchvision.models.resnet50(weights=None)
        self.backbone.layer4[0].conv2 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.layer4[0].downsample[0] = torch.nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.backbone.fc = nn.Identity()
        self.backbone.layer4[-1].conv3.register_forward_hook(self._hook)

        self.feature_grad = None
        self.feature_rect = None

        self.linear = nn.Linear(2048, const.N_CLASSES)
        self.softmax = nn.Softmax(dim=1)  # ~equivalent to sigmoid since classes = 2; relevant for CAMs

        self.to(const.DEVICE)
        self(torch.randn(1, *input_shape).to(const.DEVICE))  # initialization
        const.CAM_SIZE = tuple(self.feature_rect.shape[2:])

    def _hook(self, model, i, o):
        def assign(grad):
            self.feature_grad = grad
        self.feature_rect = o
        o.register_hook(assign)

    def forward(self, x):
        self.pre_logit = self.backbone(x)
        x = self.pre_logit.pow(3)  # scale activations to ignore lower, noisy values
        logits = x @ self.linear.weight.pow(1 if self.training else 3).T
        return self.softmax(logits * (1 if self.training else 1E3)), self._compute_hi_res_cam(logits)

    @staticmethod
    def get_contrastive_cams(y, cams):
        contrastive_cams = torch.empty((y.shape[0], cams.shape[-3] - 1, *cams.shape[-2:]), device=const.DEVICE)
        for idx, (cam, y_idx) in enumerate(zip(cams, y.argmax(dim=1))):
            cam = cam[y_idx] - cam  # objective: maximize this difference, constraining contrast to segmentation target region only
            contrastive_cams[idx] = torch.cat([cam[:y_idx], cam[y_idx+1:]])
        return contrastive_cams

    def _compute_hi_res_cam(self, logits):
        cams = torch.zeros(*logits.shape, *self.feature_rect.shape[2:])
        for img_idx in range(logits.shape[0]):
            for class_idx in range(logits.shape[1]):
                logits[img_idx, class_idx].backward(retain_graph=True, inputs=self.feature_rect)
                cams[img_idx, class_idx] = (self.feature_rect * self.feature_grad).sum(1)[img_idx]
        return cams


if __name__ == '__main__':
    model = Model(input_shape=const.IMAGE_SHAPE)
    model.eval()

    x = torch.rand((3, *const.IMAGE_SHAPE)).to(const.DEVICE)
    y, cam = model(x)
    print(cam.shape)

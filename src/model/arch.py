#!/usr/bin/env python3

from src import const
from torch import nn
import torchvision
import torch


# code modified from: https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
class ModifiedBN2d(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ModifiedBN2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            with torch.no_grad():
                mean = input.mean([0, 2, 3])
                var = input.var([0, 2, 3], unbiased=False)
                n = input.numel() / input.size(1)

                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class Model(nn.Module):
    def __init__(self, randomized_flatten=const.RANDOMIZED_FLATTEN, multilabel=False, logits_only=False,
                 disable_bn=const.DISABLE_BN, modified_bn=const.MODIFY_BN, register_backward_hook=False, hardinet_eval=False, xl_backbone=const.XL_BACKBONE,
                 device=const.DEVICE, is_contrastive=True, segmentation_threshold=const.SEGMENTATION_THRESHOLD, upsampling_level=1):
        super().__init__()

        self.segmentation_threshold = segmentation_threshold
        self.register_backward_hook = register_backward_hook
        self.randomized_flatten = randomized_flatten
        self.is_contrastive = is_contrastive
        self.hardinet_eval = hardinet_eval
        self.modified_bn = modified_bn
        self.disable_bn = disable_bn
        self.device = device

        self._orig_bn = nn.BatchNorm2d
        if modified_bn: nn.BatchNorm2d = ModifiedBN2d

        if xl_backbone: self.backbone = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2 if const.PRETRAINED_BACKBONE else None)
        else: self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if const.PRETRAINED_BACKBONE else None)

        nn.BatchNorm2d = self._orig_bn

        if upsampling_level >= 1 or upsampling_level <= -5:
            self.backbone.layer4[0].conv2.stride = (1, 1)
            self.backbone.layer4[0].downsample[0].stride = (1, 1)
        if upsampling_level >= 2 or upsampling_level <= -4:
            self.backbone.layer3[0].conv2.stride = (1, 1)
            self.backbone.layer3[0].downsample[0].stride = (1, 1)
        if upsampling_level >= 3 or upsampling_level <= -3:
            self.backbone.layer2[0].conv2.stride = (1, 1)
            self.backbone.layer2[0].downsample[0].stride = (1, 1)
        if upsampling_level >= 4 or upsampling_level <= -2:
            self.backbone.conv1.stride = (1, 1)
        if upsampling_level >= 5 or upsampling_level <= -1:
            self.backbone.maxpool.stride = 1

        if is_contrastive:
            self.backbone.layer4[-1].bn3 = nn.Identity()
            self.backbone.layer4[-1].relu = nn.Identity()
        self.backbone.layer4[-1].conv3.register_forward_hook(self._hook)

        self.linear = nn.Linear(2048, const.N_CLASSES, bias=not is_contrastive)
        self.probabilities = nn.Identity() if logits_only else nn.Softmax(dim=1) if not multilabel else nn.Sigmoid()

        if const.DATASET in ['imagenet', 'salientimagenet'] and const.PRETRAINED_BACKBONE:
            self.linear.weight = self.backbone.fc.weight
            if not is_contrastive: self.linear.bias = self.backbone.fc.bias
        self.backbone.fc = nn.Identity()

        if disable_bn: self.disable_batchnorms()
        if modified_bn:
            for x in self.modules():
                if x._get_name() == 'ModifiedBN2d':
                    x.reset_parameters()
                    x.reset_running_stats()

        self.to(self.device)

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
        if self.register_backward_hook: o.register_hook(assign)

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

    def update_tracked_statistics(self, gen):
        for x in self.modules():
            if x._get_name() in ['ModifiedBN2d', 'BatchNorm2d']:
                x.momentum = None  # obtain cumulative statistics
                x.reset_running_stats()

        with torch.no_grad():
            for X, y in gen: self(X.to(const.DEVICE))

        for x in self.modules():
            if x._get_name() in ['ModifiedBN2d', 'BatchNorm2d']: x.momentum = 0.1  # original momentum value; hardcoded for this network
        return self.state_dict()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = Model(disable_bn=False, modified_bn=True, upsampling_level=const.UPSAMPLING_LEVEL)
    print(model)

    x = torch.rand(1, *const.IMAGE_SHAPE, device=const.DEVICE, requires_grad=True)
    y, cam = model(x)
    print(cam.shape)

    for i in range(cam.size(1)):
        cam[0][i][cam.size(2) // 2, cam.size(3) // 2].backward(inputs=x, retain_graph=True)

    plt.imshow(x.grad[0].abs().mean(0).detach().cpu())
    plt.show()

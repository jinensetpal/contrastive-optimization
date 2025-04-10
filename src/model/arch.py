#!/usr/bin/env python3

from .resnet import Bottleneck
from .activations import *
from src import const
from .norms import *
from torch import nn
import torchvision
import torch


class Model(nn.Module):
    def __init__(self, randomized_flatten=const.RANDOMIZED_FLATTEN, multilabel=False, logits_only=False, backbone_acts=const.ACTIVATIONS,
                 disable_bn=const.DISABLE_BN, modified_bn=None, register_backward_hook=False, hardinet_eval=False,
                 xl_backbone=const.XL_BACKBONE, load_pretrained_weights=const.PRETRAINED_BACKBONE, n_classes=const.N_CLASSES,
                 device=const.DEVICE, is_contrastive=True, segmentation_threshold=const.SEGMENTATION_THRESHOLD, upsampling_level=1):
        super().__init__()

        self.load_pretrained_weights = load_pretrained_weights
        self.segmentation_threshold = segmentation_threshold
        self.register_backward_hook = register_backward_hook
        self.randomized_flatten = randomized_flatten
        self.is_contrastive = is_contrastive
        self.hardinet_eval = hardinet_eval
        self.modified_bn = modified_bn
        self.disable_bn = disable_bn
        self.device = device

        self._orig_bneck = torchvision.models.resnet.Bottleneck
        self._orig_bn = nn.BatchNorm2d
        self._orig_relu = nn.ReLU

        if modified_bn == 'Causal': nn.BatchNorm2d = ModifiedBN2d
        elif modified_bn == 'DyT': nn.BatchNorm2d = DynamicTanh

        if backbone_acts == 'ELU': nn.ReLU = nn.ELU
        elif backbone_acts == 'EEU': nn.ReLU = EEU
        elif backbone_acts == 'ExtendedSigmoid': nn.ReLU = ExtendedSigmoid
        elif backbone_acts == 'DyT':
            nn.ReLU = LazyDyT
            torchvision.models.resnet.Bottleneck = Bottleneck

        load_from_torchvision = load_pretrained_weights and backbone_acts != 'DyT' and modified_bn != 'DyT'
        if xl_backbone: self.backbone = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2 if load_from_torchvision else None)
        else: self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if load_from_torchvision else None)

        if backbone_acts != 'ReLU': nn.ReLU = self._orig_relu
        if backbone_acts == 'DyT': torchvision.models.resnet.Bottleneck = self._orig_bneck
        if modified_bn: nn.BatchNorm2d = self._orig_bn

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

        self.linear = nn.Linear(2048, n_classes, bias=not is_contrastive)
        self.probabilities = nn.Identity() if logits_only else nn.Softmax(dim=1) if not multilabel else nn.Sigmoid()

        if const.DATASET in ['imagenet', 'salientimagenet'] and load_pretrained_weights:
            self.linear.weight = self.backbone.fc.weight
            if not is_contrastive: self.linear.bias = self.backbone.fc.bias
        self.backbone.fc = nn.Identity()

        if load_pretrained_weights and not load_from_torchvision:
            if const.ACTIVATIONS == 'ELU' and const.MODIFY_BN == 'DyT': pretrained_model = 'resnet50_upsampled_dytbn_elu.pt'
            elif const.ACTIVATIONS == 'ReLU' and const.MODIFY_BN == 'Causal': pretrained_model = 'resnet50_upsampled_causalbn.pt'
            else: raise AttributeError('Pre-Trained match for configuration not found.')

            state_dict = torch.load(const.PRETRAINED_MODELS_DIR / pretrained_model, map_location=torch.device('cpu'), weights_only=True)

            if n_classes != 1000: del state_dict['linear.weight']
            self.load_state_dict(state_dict, strict=n_classes == 1000)

        if disable_bn: self.disable_batchnorms()
        if modified_bn or backbone_acts == 'DyT':
            for x in self.modules():
                if x._get_name() == 'LazyDyT':
                    x.device = self.device
                elif x._get_name() == 'ModifiedBN2d':
                    x.reset_parameters()
                    x.reset_running_stats()

                    if not const.AFFINE_BN:
                        self.weight = None
                        self.bias = None
                        self.affine = False

        self.backbone = torch.compile(self.backbone, backend='cudagraphs')

        self.to(self.device)
        if not hardinet_eval: self.initialize_and_verify()

    def disable_batchnorms(self):
        for x in self.modules():
            if x._get_name() == 'BatchNorm2d':
                x.eval()
                x.reset_parameters()
                x.reset_running_stats()
                x.eps = 0

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

        if self.disable_bn: self.disable_batchnorms()
        return self

    def initialize_and_verify(self):
        with torch.no_grad():
            x = torch.randn(100, *const.IMAGE_SHAPE, device=self.device)
            if self.modified_bn == 'Causal' and not self.load_pretrained_weights: self.overwrite_tracked_statistics(((x, None),))

            logits, cam = self(x)
            cam_logits = cam.view(*cam.shape[:2], -1).sum(2)

            if not self.is_contrastive: cam_logits -= self.linear.bias
            print('Approx. cam logit err bound:', (logits - cam_logits).abs().max().item())

            if self.is_contrastive: assert torch.allclose(logits, cam_logits, atol=1E-5 if torch.get_float32_matmul_precision() == 'highest' else 1E-2)

    def _hook(self, model, i, o):
        def assign(grad):
            self.feature_grad = grad
        self.feature_rect = o
        if self.register_backward_hook: o.register_hook(assign)

    @torch.compiler.disable
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
        return torch.index_select(cams.view(-1, *cams.shape[2:]), 0, y.argmax(1) + (torch.arange(cams.size(0), device=self.device) * cams.size(1))).repeat(1, cams.size(1), 1).view(*cams.shape) - cams

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

    def overwrite_tracked_statistics(self, gen):
        for module in self.modules():
            if module._get_name() in ['ModifiedBN2d', 'BatchNorm2d']:
                module.momentum = None  # obtain cumulative statistics
                module.reset_running_stats()
                module.update_proxy_stats = True

        for X, y in gen: self(X.to(self.device))

        for module in self.modules():
            if module._get_name() in ['ModifiedBN2d', 'BatchNorm2d']:
                module.momentum = 0.1  # original momentum value; hardcoded for this network
                module.update_proxy_stats = False
        return self.state_dict()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = Model(is_contrastive=True, multilabel=False, xl_backbone=False, upsampling_level=const.UPSAMPLING_LEVEL, load_pretrained_weights=False,
                  n_classes=1000, logits_only=True, disable_bn=False, modified_bn='DyT', backbone_acts='ELU', device=const.DEVICE)
    print(model)

    x = torch.rand(1, *const.IMAGE_SHAPE, device=const.DEVICE, requires_grad=True)
    y, cam = model(x)
    print(cam.shape)

    for i in range(cam.size(1)):
        cam[0][i][cam.size(2) // 2, cam.size(3) // 2].backward(inputs=x, retain_graph=True)

    plt.imshow(x.grad[0].abs().mean(0).detach().cpu())
    plt.show()

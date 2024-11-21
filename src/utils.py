#!/usr/bin/env python3

# Utils for imagenet training recipe. Code taken with minor modifications from:
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py
# https://github.com/pytorch/vision/blob/main/references/classification/presets.py
# https://github.com/pytorch/vision/blob/main/references/classification/sampler.py

import math
import torch
import torch.distributed as dist
import torchvision.transforms.v2 as T
from torchvision.transforms.functional import InterpolationMode


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy='ta_wide',
        ra_magnitude=None,
        augmix_severity=None,
        random_erase_prob=0.0,
        use_v2=False,
    ):
        random_transforms = []
        deterministic_transforms = []

        random_transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True)) # random
        if hflip_prob > 0:
            random_transforms.append(T.RandomHorizontalFlip(hflip_prob)) # random
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                random_transforms.append(T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                random_transforms.append(T.TrivialAugmentWide(interpolation=interpolation)) # random
            elif auto_augment_policy == "augmix":
                random_transforms.append(T.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                random_transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if random_erase_prob > 0: # random
            random_transforms.append(T.RandomErasing(p=random_erase_prob))

        deterministic_transforms.extend(
            [
                T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float), # deterministic
                T.Normalize(mean=mean, std=std), # deterministic
            ]
        )

        if use_v2:
            deterministic_transforms.append(T.ToPureTensor()) # deterministic

        self.random_transforms = T.Compose(random_transforms)
        self.deterministic_transforms = T.Compose(deterministic_transforms)

    def __call__(self, imgs):
        transformed_imgs = []
        state = torch.get_rng_state()
        for idx, img in enumerate(imgs):
            torch.set_rng_state(state)
            img = self.random_transforms(img)
            if idx == 1: img = self.deterministic_transforms(img)
            transformed_imgs.append(img)
        return transformed_imgs


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms += [
            T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, imgs):
        transformed_imgs = []
        state = torch.get_rng_state()
        for img in imgs:
            torch.set_rng_state(state)
            transformed_imgs.append(self.transforms(img))
        return transformed_imgs


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

#!/usr/bin/env python3

# utils for imagenet training recipe. Code taken with modifications from:
# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
# https://stackoverflow.com/a/2838309/10671309
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py
# https://github.com/pytorch/vision/blob/main/references/classification/presets.py
# https://github.com/pytorch/vision/blob/main/references/classification/sampler.py
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_augment.py

from torchvision.transforms.v2._utils import is_pure_tensor, query_size
from torchvision.transforms.functional import InterpolationMode, resize
from torchvision.transforms.v2.functional._meta import get_size
from torchvision import transforms as tv_tensors
from torchvision.ops import masks_to_boxes
import torchvision.transforms.v2 as T
import torch.distributed as dist
import socket
import torch
import math


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def trim_mask(mask, cam_size, reduce_factor=4, center_bias=1):
    small_mask = resize(mask[None,], [x // reduce_factor for x in mask.shape], interpolation=T.InterpolationMode.NEAREST_EXACT)
    if small_mask.min() == small_mask.max(): return resize(mask[None,], cam_size, interpolation=T.InterpolationMode.NEAREST_EXACT)[0]
    x_1, y_1, x_2, y_2 = masks_to_boxes(small_mask)[0]

    r1 = (x_1 + center_bias) / (small_mask.size(1) - x_2 + center_bias - 1)
    t1 = r1 + 1
    r2 = (y_1 + center_bias) / (small_mask.size(2) - y_2 + center_bias - 1)
    t2 = r2 + 1

    target_pad = cam_size[0] - small_mask.size(1)
    return T.functional.pad(small_mask, (int((target_pad * r1 / t1 + 1E-4).round()),
                                         int((target_pad * r2 / t2 + 1E-4).round()),
                                         int((target_pad * (1 - r1 / t1) - 1E-4).round()),
                                         int((target_pad * (1 - r2 / t2) - 1E-4).round())))[0]


def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


class CutMix(T.CutMix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from src import const

        self.cam_size = const.CAM_SIZE

    def _needs_transform_list(self, flat_inputs):
        return [True for _ in range(len(flat_inputs))]

    def _get_params(self, flat_inputs):
        lam = float(self._dist.sample(()))  # type: ignore[arg-type]

        H, W = query_size(flat_inputs[0])

        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        ret = dict(box=box, lam_adjusted=lam_adjusted)
        return ret

    def _transform(self, inpt, params):
        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=params["lam_adjusted"])
        elif is_pure_tensor(inpt) or isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
            if self.cam_size == inpt.shape[1:]: params['box'] = [int(x / 16) for x in params['box']]

            x1, y1, x2, y2 = params["box"]
            rolled = inpt.roll(1, 0)
            output = inpt.clone()
            output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]
            return output
        else:
            return inpt


class TrivialAugmentWide(T.TrivialAugmentWide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.heatmap_turn = False

    def forward(self, *inputs):
        flat_inputs_with_spec, image_or_video = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)  # type: ignore[arg-type]

        transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)
        if self.heatmap_turn and transform_id in ['Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize', 'Solarize', 'AutoContrast']:  # won't affect position therefore invariant to heatmap
            self.heatmap_turn = False
            return image_or_video  # do nothing

        magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
        if magnitudes is not None:
            magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1
        else:
            magnitude = 0.0

        image_or_video = self._apply_image_or_video_transform(
            image_or_video, transform_id, magnitude, interpolation=self.interpolation, fill=self._fill
        )
        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)


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

        random_transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))  # random
        if hflip_prob > 0:
            random_transforms.append(T.RandomHorizontalFlip(hflip_prob))  # random
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                random_transforms.append(T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                random_transforms.append(TrivialAugmentWide(interpolation=interpolation))  # random
            elif auto_augment_policy == "augmix":
                random_transforms.append(T.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                random_transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if random_erase_prob > 0:  # random
            random_transforms.append(T.RandomErasing(p=random_erase_prob))

        deterministic_transforms.extend(
            [
                T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),  # deterministic
                T.Normalize(mean=mean, std=std),  # deterministic
            ]
        )

        if use_v2:
            deterministic_transforms.append(T.ToPureTensor())  # deterministic

        self.random_transforms = T.Compose(random_transforms)
        self.deterministic_transforms = T.Compose(deterministic_transforms)

    def __call__(self, imgs):
        transformed_imgs = []
        state = torch.get_rng_state()
        for idx, img in enumerate(imgs):
            torch.set_rng_state(state)

            self.random_transforms.transforms[2].heatmap_turn = idx == 1
            img = self.random_transforms(img)

            if idx == 0: img = self.deterministic_transforms(img)
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

#!/usr/bin/env python3

from src.utils import get_open_port
from pathlib import Path
import torch
import os

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# evals
CONFIDENCE_THRESHOLD = .85
SEGMENTATION_THRESHOLD = 1E-2

# dataset
SEED = 1024
N_WORKERS = 3
N_CHANNELS = 3
BBOX_MAP = False
BINARY_CLS = True
DATASET = 'imagenet'
CAM_SIZE = (14, 14)
VAL_CROP_SIZE = 224
VAL_RESIZE_SIZE = 232
TRAIN_CROP_SIZE = 224
IMAGE_SIZE = (224, 224)
SPLITS = ['train', 'valid', 'test']
IMAGE_SHAPE = (N_CHANNELS,) + IMAGE_SIZE

SAL_INET_MASK_THRESHOLD = .75
HARD_INET_BALANCED_SUBSET = False
HARD_INET_TRIM_MASKS = False
HARD_INET_TRIM_FACTOR = 12
HARD_INET_CENTER_BIAS = 2

if DATASET in ['imagenet', 'salientimagenet']: N_CLASSES = 1000
elif DATASET == 'hardimagenet': N_CLASSES = 15
elif DATASET == 'soodimagenet': N_CLASSES = 56
elif DATASET == 'sbd': N_CLASSES = 20
else: N_CLASSES = 2 if BINARY_CLS else 37

# model
PRETRAINED_BACKBONE = False
RANDOMIZED_FLATTEN = False
UPSAMPLING_LEVEL = 1  # -ve changes direction not magnitude
XL_BACKBONE = False
DISABLE_BN = False
AFFINE_BN = False
MODIFY_BN = None
ACTIVATIONS = 'ReLU'

# training
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TRAIN_CUTOFF = 12600  # max training time in seconds
LAMBDAS = [10, 0, 1, 1E3]  # means different things for different approaches
POS_ONLY = False and DATASET == 'sbd'  # (multilabel only) restrict divergence loss to just positive classes
GRAD_ACCUMULATION_STEPS = 1
PORT = get_open_port()
CHECKPOINTING = True
WEIGHT_DECAY = 1E-4
SELECT_BEST = True
FINETUNING = False
USE_ZERO = False
OPTIMIZER = 'SGD'
BATCH_SIZE = 768
MOMENTUM = .9
EPOCHS = 50
DDP = os.getenv('WORLD_SIZE') is not None

# divergence
DIVERGENCE = 'sliced_wasserstein' if LAMBDAS[-1] != 0 else None  # edit first string only to set divergence
SINKHORN_COST_POW = 2
SINKHORN_BLUR = .1

# ema
EMA = True
EMA_STEPS = 32
EMA_DECAY = .99998

# learning rate
LR = 1E-4
LR_WARMUP_EPOCHS = 5
LR_WARMUP_DECAY = .01

# augmentation
AUGMENT = False
USE_CUTMIX = False
CUTMIX_ALPHA = 1.
LABEL_SMOOTHING = .1
AUGMENT_REPITIONS = 4
RANDOM_ERASE_PROB = .1
AUTO_AUGMENT_POLICY = 'ta_wide'

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/contrastive-optimization.mlflow'
LOG_BATCHWISE = True
LOG_REMOTE = False

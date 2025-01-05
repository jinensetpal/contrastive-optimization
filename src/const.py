#!/usr/bin/env python3

from src.utils import get_open_port
from pathlib import Path
import torch
import os

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# training
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TRAIN_CUTOFF = 12600  # max training time in seconds
LAMBDAS = [1E1, 5E1, 1, 1]
GRAD_ACCUMULATION_STEPS = 1
PRETRAINED_BACKBONE = False
RANDOMIZED_FLATTEN = False
PORT = get_open_port()
CHECKPOINTING = True
WEIGHT_DECAY = 1E-4
SELECT_BEST = True
FINETUNING = False
USE_ZERO = False
OPTIMIZER = 'SGD'
BATCH_SIZE = 440
MOMENTUM = .9
EPOCHS = 150
DDP = os.getenv('WORLD_SIZE') is not None

# ema
EMA = True
EMA_STEPS = 32
EMA_DECAY = .99998

# learning rate
LR = 1E-3
LR_WARMUP_EPOCHS = 0
LR_WARMUP_DECAY = .01

# evals
CONFIDENCE_THRESHOLD = .85
SEGMENTATION_THRESHOLD = 1E-2

# dataset
SEED = 1024
N_WORKERS = 2
N_CHANNELS = 3
BBOX_MAP = False
BINARY_CLS = True
DATASET = 'sbd'
CAM_SIZE = (14, 14)
VAL_CROP_SIZE = 224
VAL_RESIZE_SIZE = 232
TRAIN_CROP_SIZE = 224
IMAGE_SIZE = (224, 224)
SPLITS = ['train', 'valid', 'test']
IMAGE_SHAPE = (N_CHANNELS,) + IMAGE_SIZE

if DATASET == 'imagenet': N_CLASSES = 1000
elif DATASET == 'soodimagenet': N_CLASSES = 56
elif DATASET == 'sbd': N_CLASSES = 20
else: N_CLASSES = 2 if BINARY_CLS else 37

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

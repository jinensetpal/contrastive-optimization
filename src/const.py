#!/usr/bin/env python3

from pathlib import Path
import torch
import uuid

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
TMP_FILESTORE = BASE_DIR / 'filestore' / uuid.uuid1().hex

# training
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TRAIN_CUTOFF = 12600  # max training time in seconds
LAMBDAS = [1E1, 5E1, 1, 0]
GRAD_ACCUMULATION_STEPS = 2
PRETRAINED_BACKBONE = False
RANDOMIZED_FLATTEN = False
CHECKPOINTING = True
WEIGHT_DECAY = 2E-5
SELECT_BEST = True
FINETUNING = False
OPTIMIZER = 'SGD'
BATCH_SIZE = 128
MOMENTUM = .9
EPOCHS = 600
DDP = True

# ema
EMA = True
EMA_STEPS = 32
EMA_DECAY = .99998

# learning rate
LR = .5
LR_WARMUP_EPOCHS = 5
LR_WARMUP_DECAY = .01

# evals
EVAL_BATCH_SIZE = 10
CONFIDENCE_THRESHOLD = .85

# dataset
SEED = 1024
BBOX_MAP = True
USE_CUTMIX = True
CUTMIX_ALPHA = 1.
BINARY_CLS = True
N_CHANNELS = 3
CAM_SIZE = (14, 14)
DATASET = 'imagenet'
VAL_CROP_SIZE = 224
LABEL_SMOOTHING = .1
VAL_RESIZE_SIZE = 232
TRAIN_CROP_SIZE = 224
AUGMENT_REPITIONS = 4
RANDOM_ERASE_PROB = .1
IMAGE_SIZE = (224, 224)
AUTO_AUGMENT_POLICY = 'ta_wide'
SPLITS = ['train', 'valid', 'test']
IMAGE_SHAPE = (N_CHANNELS,) + IMAGE_SIZE

if DATASET == 'imagenet': N_CLASSES = 1000
else: N_CLASSES = 2 if BINARY_CLS else 37

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/contrastive-optimization.mlflow'
LOG_REMOTE = False

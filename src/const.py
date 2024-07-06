#!/usr/bin/env python3

from pathlib import Path
import torch

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
QUANTILE_CLIP_CAMS = True
LEARNING_RATE = 1E-3
SELECT_BEST = True
BATCH_SIZE = 64
MOMENTUM = 0.9
EPOCHS = 20
MODEL_NAME = 'contrastiveloss' if LOSS_WEIGHTS[1] else 'default'

# dataset
IMAGE_SIZE = (224, 224)
N_CHANNELS = 3
N_CLASSES = 2
IMAGE_SHAPE = (N_CHANNELS,) + IMAGE_SIZE

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/contrastive-optimization.mlflow'
LOG_REMOTE = False

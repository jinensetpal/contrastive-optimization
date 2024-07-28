#!/usr/bin/env python3

from pathlib import Path
import torch

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ACTIVATION_CLIPPING = False
LEARNING_RATE = 1E-3
SELECT_BEST = True
BATCH_SIZE = 16
EPOCHS = 20

# dataset
SEED = 1024
SPLITS = ['train', 'valid', 'test']
IMAGE_SIZE = (224, 224)
N_CHANNELS = 3
BINARY_CLS = True
N_CLASSES = 2 if BINARY_CLS else 37
IMAGE_SHAPE = (N_CHANNELS,) + IMAGE_SIZE

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/contrastive-optimization.mlflow'
LOG_REMOTE = False

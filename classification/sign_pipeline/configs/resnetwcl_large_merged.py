import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize, Compose, RandomAffine
import imgaug.augmenters as iaa

from pipeline.schedulers.learning_rate.reduce_on_plateau import SchedulerWrapperLossOnPlateau
from .base import ConfigSignBase, PredictConfigSignBase, NUM_CLASSES, TRAIN_DATASET_PATH_MERGED

MODEL_SAVE_PATH = "models/sign_resnetwcl_large_merged"
BATCH_SIZE = 64

SEED = 85
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)


def get_model():
    model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


class Config(ConfigSignBase):
    def __init__(self):
        model = get_model()
        train_transforms = Compose([
            RandomAffine(degrees=20, scale=(0.8, 1.1)),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Invert(0.05, per_channel=True),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        val_transforms = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        optimizer = optim.Adam(
            model.parameters(),
            lr=3e-4)

        scheduler = SchedulerWrapperLossOnPlateau(optimizer, patience=3, min_lr=1e-6)

        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataset_path=TRAIN_DATASET_PATH_MERGED,
            model_save_path=MODEL_SAVE_PATH,
            epoch_count=100,
            batch_size=BATCH_SIZE,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            mixup_alpha=0.7,
            scheduler=scheduler)


class PredictConfig(PredictConfigSignBase):
    def __init__(self):
        super().__init__(model=get_model(), model_save_path=MODEL_SAVE_PATH, batch_size=BATCH_SIZE)

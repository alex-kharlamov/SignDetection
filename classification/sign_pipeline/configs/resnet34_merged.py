import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34
from torchvision.transforms import ToTensor, Normalize, Compose, RandomAffine

from .base import ConfigSignBase, PredictConfigSignBase, NUM_CLASSES, TRAIN_DATASET_PATH_MERGED

MODEL_SAVE_PATH = "models/sign_resnet34_merged"
BATCH_SIZE = 128

SEED = 85
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)


def get_model():
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


class Config(ConfigSignBase):
    def __init__(self):
        model = get_model()
        train_transforms = Compose([
            RandomAffine(degrees=20, scale=(0.8, 1.1)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        ])

        val_transforms = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        ])

        super().__init__(
            model=model,
            train_dataset_path=TRAIN_DATASET_PATH_MERGED,
            model_save_path=MODEL_SAVE_PATH,
            epoch_count=100,
            batch_size=BATCH_SIZE,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            mixup_alpha=0.5)


class PredictConfig(PredictConfigSignBase):
    def __init__(self):
        super().__init__(model=get_model(), model_save_path=MODEL_SAVE_PATH, batch_size=BATCH_SIZE)

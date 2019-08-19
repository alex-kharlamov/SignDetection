import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34
from torchvision.transforms import ToTensor, Normalize, Compose, RandomAffine

from .base import ConfigSignBase, PredictConfigSignBase
from sign_pipeline.associated import AVAILABLE_CLASSES

MODEL_SAVE_PATH = "models/sign_resnet34"
BATCH_SIZE = 128

SEED = 85
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)


def get_model():
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(AVAILABLE_CLASSES) + 2)
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
            model_save_path=MODEL_SAVE_PATH,
            epoch_count=100,
            batch_size=BATCH_SIZE,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            mixup_alpha=0.0)


class PredictConfig(PredictConfigSignBase):
    def __init__(self):
        super().__init__(
            model=get_model(), model_save_path=MODEL_SAVE_PATH, batch_size=BATCH_SIZE)

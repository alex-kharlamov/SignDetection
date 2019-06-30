import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34
from torchvision.transforms import ToTensor

from .base import ConfigSignBase, PredictConfigSignBase, NUM_CLASSES

MODEL_SAVE_PATH = "models/sign_resnet34"
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
        transforms = ToTensor()
        super().__init__(
            model=model,
            model_save_path=MODEL_SAVE_PATH,
            epoch_count=30,
            batch_size=BATCH_SIZE,
            transforms=transforms)


class PredictConfig(PredictConfigSignBase):
    def __init__(self):
        super().__init__(model=get_model(), model_save_path=MODEL_SAVE_PATH, batch_size=BATCH_SIZE)

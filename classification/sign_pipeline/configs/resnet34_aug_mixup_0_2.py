import random

import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose, RandomAffine

from sign_pipeline.associated import AVAILABLE_CLASSES
from .base import ConfigSignBase, PredictConfigSignBase

from torchvision.models import resnet34

MODEL_SAVE_PATH = "models/sign_resnet34_0_2"
BATCH_SIZE = 128

SEED = 85
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)


class ImgAugTransforms:
    def __init__(self):
        self._seq = iaa.Sequential([
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Invert(0.05, per_channel=True),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
        ])

    def __call__(self, image):
        image = np.array(image)
        image = self._seq.augment_image(image)
        return Image.fromarray(image)


def get_model():
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(AVAILABLE_CLASSES) + 2)
    return model


class Config(ConfigSignBase):
    def __init__(self):
        model = get_model()
        train_transforms = Compose([
            RandomAffine(degrees=20, scale=(0.8, 1.1)),
            ImgAugTransforms(),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        val_transforms = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        super().__init__(
            model=model,
            model_save_path=MODEL_SAVE_PATH,
            epoch_count=100,
            batch_size=BATCH_SIZE,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            mixup_alpha=0.2)


class PredictConfig(PredictConfigSignBase):
    def __init__(self):
        transforms = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        super().__init__(
            model=get_model(), model_save_path=MODEL_SAVE_PATH, batch_size=BATCH_SIZE, transforms=transforms)

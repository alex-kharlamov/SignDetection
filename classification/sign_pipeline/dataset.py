import pickle

import torch.utils.data as data
from torchvision.transforms import Compose, Resize, RandomCrop
import torch
from sign_pipeline.associated import AVAILABLE_CLASSES


class SignDataset(data.Dataset):
    def __init__(self, path, load_size, crop_size, use_mixup=False):
        with open(path, "rb") as fin:
            self._data = pickle.load(fin)

        self._transforms = Compose([
            Resize(load_size),
            RandomCrop(crop_size),
        ])

        self._use_mixup = use_mixup

    def get_image(self, item):
        image = self._data[item]["cropped_image"]
        return self._transforms(image)

    def get_class(self, item):
        associated_label = self._data[item]["associated_label"]
        temporary = float(self._data[item]["temporary"])

        if self._use_mixup:
            result = torch.zeros(len(AVAILABLE_CLASSES) + 1, dtype=torch.float32)
            result[associated_label] = 1
        else:
            result = associated_label

        return [result, temporary]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]


class SignImagesDataset(SignDataset):
    def __getitem__(self, item):
        return self.get_image(item)


class SignTargetsDataset(SignDataset):
    def __getitem__(self, item):
        return self.get_class(item)

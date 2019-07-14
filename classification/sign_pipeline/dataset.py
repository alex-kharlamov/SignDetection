import pickle

import torch.utils.data as data
from torchvision.transforms import Compose, Resize, RandomCrop


class SignDataset(data.Dataset):
    def __init__(self, path, load_size, crop_size):
        with open(path, "rb") as fin:
            self._data = pickle.load(fin)

        self._transforms = Compose([
            Resize(load_size),
            RandomCrop(crop_size),
        ])

    def get_image(self, item):
        image = self._data[item]["cropped_image"]
        return self._transforms(image)

    def get_class(self, item):
        associated_label = self._data[item]["associated_label"]
        temporary = float(self._data[item]["temporary"])
        return [associated_label, temporary]

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

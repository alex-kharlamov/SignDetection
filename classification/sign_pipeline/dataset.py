import pickle

import torch.utils.data as data
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor


class SignDataset(data.Dataset):
    def __init__(self, path, labels_mapping_path, load_size=256, crop_size=224):
        with open(path, "rb") as fin:
            self._data = pickle.load(fin)

        with open(labels_mapping_path, "rb") as fin:
            self._labels_mapping = pickle.load(fin)

        self._transforms = Compose([
            Resize(load_size),
            RandomCrop(crop_size),
        ])

    def get_image(self, item):
        image = self._data[item]["cropped_image"]
        return self._transforms(image)

    def get_class(self, item):
        return self._labels_mapping[self._data[item]["label"]]

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

    def get_class_count(self):
        return len(self._labels_mapping)

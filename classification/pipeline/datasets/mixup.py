import torch.utils.data as data
import random
import numpy as np


def _mixup(elem1, elem2, coeff):
    if isinstance(elem1, (tuple, list)):
        result = []
        for inner_elem1, inner_elem2 in zip(elem1, elem2):
            result.append(_mixup(inner_elem1, inner_elem2, coeff))

        if isinstance(elem1, tuple):
            result = tuple(result)

        return result
    else:
        return elem1 * coeff + elem2 * (1 - coeff)


class MixUpDatasetWrapper(data.Dataset):
    def __init__(self, dataset, alpha=1):
        super().__init__()
        self._dataset = dataset
        self._alpha = alpha

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        first = self._dataset[item]
        second = random.choice(self._dataset)

        coeff = np.random.beta(self._alpha, self._alpha)

        return _mixup(first, second, coeff)

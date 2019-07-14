import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score

from pipeline.metrics.base import MetricsCalculatorBase


class SignLoss(nn.Module):
    def __init__(self, binary_weight=1.0):
        super().__init__()
        self._multi = nn.CrossEntropyLoss()
        self._binary = nn.BCEWithLogitsLoss()
        self._binary_weight = binary_weight

    def forward(self, y_pred, y_true):
        multi_pred = y_pred[:, :-1]
        binary_pred = y_pred[:, -1:]

        import pdb
        pdb.set_trace()
        multi_true = y_true[0]
        binary_true = y_true[1]

        loss = self._multi(multi_pred, multi_true) + self._binary_weight * self._binary(binary_pred, binary_true)
        return loss


class SignMetricsCalculator(MetricsCalculatorBase):
    def __init__(self, border=0):
        super().__init__()
        self.zero_cache()
        self._border = border

    def zero_cache(self):
        self._predictions = []
        self._true_labels = []

    def add(self, y_predicted, y_true):
        self._predictions.append(y_predicted.cpu().data.numpy())
        self._true_labels.append(y_true.cpu().data.numpy())

    def calculate(self):
        y_pred = np.concatenate(self._predictions)
        y_true = np.concatenate(self._true_labels)

        y_pred_multi = np.argmax(y_pred[:, :-1], -1)
        y_pred_binary = (y_pred[:, -1] >= self._border).astype("int")

        y_true_multi = y_true[0]
        y_true_binary = y_true[1]

        return {"accuracy_multi": accuracy_score(y_true_multi, y_pred_multi),
                "accuracy_binary": accuracy_score(y_true_binary, y_pred_binary)}

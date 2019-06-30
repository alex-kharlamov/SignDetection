import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

from cifar_pipeline.dataset import CIFARImagesDataset, CIFARTargetsDataset
from pipeline.config_base import ConfigBase
from pipeline.datasets.base import DatasetWithPostprocessingFunc, DatasetComposer, OneHotTargetsDataset
from pipeline.datasets.mixup import MixUpDatasetWrapper
from pipeline.losses.vector_cross_entropy import VectorCrossEntropy
from pipeline.metrics.accuracy import MetricsCalculatorAccuracy
from pipeline.schedulers.learning_rate.reduce_on_plateau import SchedulerWrapperLossOnPlateau
from pipeline.trainers.classification import TrainerClassification

TRAIN_DATASET_PATH = "~/.pipeline/cifar/train"
TEST_DATASET_PATH = "~/.pipeline/cifar/test"


def get_dataset(path, transforms, train, use_mixup):
    images_dataset = DatasetWithPostprocessingFunc(
        CIFARImagesDataset(path=path, train=train, download=True),
        transforms)

    targets_dataset = CIFARTargetsDataset(path=path, train=train)
    if use_mixup:
        targets_dataset = OneHotTargetsDataset(targets_dataset, 10)

    return DatasetComposer([images_dataset, targets_dataset])


class ConfigCIFARBase(ConfigBase):
    def __init__(self, model, model_save_path, num_workers=8, batch_size=128, transforms=None,
                 epoch_count=200, print_frequency=10, mixup_alpha=0):
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4)

        scheduler = SchedulerWrapperLossOnPlateau(optimizer)
        loss = nn.CrossEntropyLoss()
        metrics_calculator = MetricsCalculatorAccuracy()
        trainer_cls = TrainerClassification

        if transforms is None:
            transforms = ToTensor()

        train_dataset = get_dataset(path=TRAIN_DATASET_PATH, transforms=transforms, train=True,
                                    use_mixup=mixup_alpha > 0)
        val_dataset = get_dataset(path=TEST_DATASET_PATH, transforms=transforms, train=False,
                                  use_mixup=mixup_alpha > 0)

        if mixup_alpha > 0:
            train_dataset = MixUpDatasetWrapper(train_dataset, alpha=mixup_alpha)
            loss = VectorCrossEntropy()

        super().__init__(
            model=model,
            model_save_path=model_save_path,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            metrics_calculator=metrics_calculator,
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            trainer_cls=trainer_cls,
            print_frequency=print_frequency,
            epoch_count=epoch_count,
            device="cpu")

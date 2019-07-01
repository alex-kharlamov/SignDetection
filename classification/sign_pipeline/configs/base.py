import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

from pipeline.config_base import ConfigBase, PredictConfigBase
from pipeline.datasets.base import DatasetWithPostprocessingFunc, DatasetComposer, OneHotTargetsDataset
from pipeline.datasets.mixup import MixUpDatasetWrapper
from pipeline.losses.vector_cross_entropy import VectorCrossEntropy
from pipeline.metrics.accuracy import MetricsCalculatorAccuracy
from pipeline.predictors.classification import PredictorClassification
from pipeline.schedulers.learning_rate.reduce_on_plateau import SchedulerWrapperLossOnPlateau
from pipeline.trainers.classification import TrainerClassification
from sign_pipeline.dataset import SignImagesDataset, SignTargetsDataset

TRAIN_DATASET_PATH = "/group-volume/orc_srr/multimodal/iceblood/classification/first_part_skolkovo"
TRAIN_DATASET_PATH_VMK = "/group-volume/orc_srr/multimodal/iceblood/classification/full_russia_vmk"
TRAIN_DATASET_PATH_MERGED = "/group-volume/orc_srr/multimodal/iceblood/classification/merged"

TEST_DATASET_PATH = "/group-volume/orc_srr/multimodal/iceblood/classification/final_skolkovo"

LABELS_MAPPING_PATH = "/group-volume/orc_srr/multimodal/iceblood/classification/labels_mapping"
TRAIN_LOAD_SIZE = 128 + 6
TRAIN_CROP_SIZE = 128

TEST_LOAD_SIZE = 128
TEST_CROP_SIZE = 128

NUM_CLASSES = 198

MAX_EPOCH_LENGTH = 10000


def get_dataset(path, transforms, train, use_mixup):
    load_size = TRAIN_LOAD_SIZE if train else TEST_LOAD_SIZE
    crop_size = TRAIN_CROP_SIZE if train else TEST_CROP_SIZE

    images_dataset = DatasetWithPostprocessingFunc(
        SignImagesDataset(path=path, labels_mapping_path=LABELS_MAPPING_PATH,
                          load_size=load_size, crop_size=crop_size),
        transforms)

    targets_dataset = SignTargetsDataset(path=path, labels_mapping_path=LABELS_MAPPING_PATH,
                                         load_size=load_size, crop_size=crop_size)
    if use_mixup:
        targets_dataset = OneHotTargetsDataset(targets_dataset, targets_dataset.get_class_count())

    return DatasetComposer([images_dataset, targets_dataset])


class ConfigSignBase(ConfigBase):
    def __init__(
            self,
            model,
            model_save_path,
            train_dataset_path=TRAIN_DATASET_PATH,
            num_workers=8,
            batch_size=128,
            train_transforms=None,
            val_transforms=None,
            epoch_count=200,
            print_frequency=10,
            mixup_alpha=0,
            optimizer=None,
            device="cuda:0",
    ):
        if optimizer is None:
            optimizer = optim.Adam(
                model.parameters(),
                lr=1e-4)

        scheduler = SchedulerWrapperLossOnPlateau(optimizer, patience=2)
        loss = nn.CrossEntropyLoss()
        metrics_calculator = MetricsCalculatorAccuracy()
        trainer_cls = TrainerClassification

        train_dataset = get_dataset(path=train_dataset_path, transforms=train_transforms, train=True,
                                    use_mixup=mixup_alpha > 0)
        val_dataset = get_dataset(path=TEST_DATASET_PATH, transforms=val_transforms, train=False,
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
            max_epoch_length=MAX_EPOCH_LENGTH,
            device=device)


class PredictConfigSignBase(PredictConfigBase):
    def __init__(self, model, model_save_path, num_workers=4, batch_size=128):
        predictor_cls = PredictorClassification

        images_dataset = DatasetWithPostprocessingFunc(
            SignImagesDataset(path=TEST_DATASET_PATH, labels_mapping_path=LABELS_MAPPING_PATH,
                              load_size=TEST_LOAD_SIZE, crop_size=TEST_CROP_SIZE),
            ToTensor())

        dataset = DatasetComposer([images_dataset, list(range(len(images_dataset)))])

        super().__init__(
            model=model,
            model_save_path=model_save_path,
            dataset=dataset,
            predictor_cls=predictor_cls,
            num_workers=num_workers,
            batch_size=batch_size)

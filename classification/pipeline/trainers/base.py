import time
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ..core import PipelineError
from ..logger import LOGGER
from ..metrics.base import MetricsCalculatorBase
from pipeline.schedulers.base import SchedulerWrapperMetricsMeanBase, SchedulerWrapperBase
from ..storage.state import StateStorageBase
from ..utils import move_to_device, save_model, load_model

import os


class TrainerBase:
    def __init__(
            self,
            model: nn.Module,
            train_data_loader: Iterable,
            val_data_loader: Iterable,
            epoch_count: int,
            optimizer: Optimizer,
            scheduler: SchedulerWrapperBase,
            loss: nn.Module,
            metrics_calculator: MetricsCalculatorBase,
            print_frequency: None or int,
            device: str,
            model_save_path: str,
            state_storage: StateStorageBase) -> None:

        self.model = model.to(device)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.epoch_count = epoch_count
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.metrics_calculator = metrics_calculator
        self.print_frequency = print_frequency
        self.device = device
        self.model_save_path = model_save_path
        self.state_storage = state_storage

    def train_step(self, input_data: torch.Tensor, target: torch.Tensor):
        input_data = move_to_device(input_data, device=self.device)
        target = move_to_device(target, device=self.device)

        model_output = self.model(input_data)

        self.optimizer.zero_grad()
        loss = self.loss(model_output, target)

        loss.backward()

        self.optimizer.step(closure=None)

        return loss.cpu().data.numpy()

    def predict_step(self, input_data: torch.Tensor):
        input_data = move_to_device(input_data, device=self.device)
        model_output = self.model(input_data)
        return model_output

    def log_train_step(self, epoch_id: int, step_id: int, epoch_time: float, loss: float, mean_loss: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{} s] Epoch {}. Train step {}. Loss {}. Mean loss {}".format(
                epoch_time, epoch_id, step_id, loss, mean_loss))
            return True

        return False

    def log_validation_step(self, epoch_id: int, step_id: int, epoch_time: float, loss: float, mean_loss: float):
        if self.print_frequency is None or step_id % self.print_frequency == 0:
            LOGGER.info("[{} s] Epoch {}. Validation step {}. Loss {}. Mean loss {}".format(
                epoch_time, epoch_id, step_id, loss, mean_loss))

            return True

        return False

    def log_train_epoch(self, epoch_id: int, epoch_time: float, mean_loss: float):
        LOGGER.info("Training Epoch {} has completed. Time: {}. Mean loss: {}".format(
            epoch_id, epoch_time, mean_loss))
        return True

    def log_validation_epoch(self, epoch_id: int, epoch_time: float, mean_loss: float, metrics: dict):
        LOGGER.info("Validation Epoch {} has completed. Time: {}. Mean loss: {}. Metrics: {}".format(
            epoch_id, epoch_time, mean_loss, str(metrics)))
        return True

    def run_train_epoch(self, epoch_id: int):
        self.model.train()

        start_time = time.time()
        mean_loss = 0
        step_count = 0

        for step_id, (input_data, target) in enumerate(self.train_data_loader):
            loss = self.train_step(input_data, target)
            epoch_time = time.time() - start_time

            mean_loss += loss
            step_count += 1

            self.log_train_step(epoch_id, step_id, epoch_time, loss, mean_loss / step_count)

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)

        self.log_train_epoch(epoch_id, epoch_time, mean_loss)

        return epoch_time, mean_loss

    def run_validation_epoch(self, epoch_id: int):
        self.model.eval()

        self.metrics_calculator.zero_cache()
        mean_loss = 0
        step_count = 0
        start_time = time.time()

        with torch.no_grad():
            for step_id, (input_data, target) in enumerate(self.val_data_loader):
                target = move_to_device(target, device=self.device)
                model_output = self.predict_step(input_data)

                loss = self.loss(model_output, target)
                mean_loss += loss
                step_count += 1
                epoch_time = time.time() - start_time

                self.metrics_calculator.add(model_output, target)
                self.log_validation_step(epoch_id, step_id, epoch_time, loss, mean_loss / step_count)

        epoch_time = time.time() - start_time
        mean_loss /= max(step_count, 1)
        metrics = self.metrics_calculator.calculate()

        self.log_validation_epoch(epoch_id, epoch_time, mean_loss, metrics)

        return epoch_time, mean_loss, metrics

    def load_optimizer_state(self):
        if not self.state_storage.has_key("learning_rates"):
            return

        learning_rates = self.state_storage.get_value("learning_rates")

        for learning_rate, param_group in zip(learning_rates, self.optimizer.param_groups):
            param_group["lr"] = learning_rate

    def save_optimizer_state(self):
        learning_rates = []
        for param_group in self.optimizer.param_groups:
            learning_rates.append(float(param_group['lr']))

        self.state_storage.set_value("learning_rates", learning_rates)

    def save_last_model(self, epoch_id):
        os.makedirs(self.model_save_path, exist_ok=True)
        model_path = os.path.join(self.model_save_path, "epoch_{}".format(epoch_id))
        save_model(self.model, model_path)
        LOGGER.info("Model was saved in {}".format(model_path))

    def load_last_model(self, epoch_id):
        last_model_path = os.path.join(self.model_save_path, "epoch_{}".format(epoch_id))
        load_model(self.model, last_model_path)

    def run(self):
        start_epoch_id = 0

        if self.state_storage.has_key("start_epoch_id"):
            start_epoch_id = self.state_storage.get_value("start_epoch_id")
            try:
                self.load_last_model(start_epoch_id - 1)
            except:
                LOGGER.exception("Exception occurs during loading a model. Starting to train a model from scratch...")
        else:
            LOGGER.info("Model not found in {}. Starting to train a model from scratch...".format(self.model_save_path))

        self.load_optimizer_state()

        epoch_id = start_epoch_id
        while self.epoch_count is None or epoch_id < self.epoch_count:
            _, mean_train_loss = self.run_train_epoch(epoch_id)

            if self.val_data_loader is None:
                if isinstance(self.scheduler, SchedulerWrapperMetricsMeanBase):
                    raise PipelineError("You can't use a scheduler based on metrics without validation data")
                self.scheduler.step(mean_train_loss, {}, epoch_id)
                continue

            _, mean_validation_loss, validation_metrics = self.run_validation_epoch(epoch_id)
            self.scheduler.step(mean_validation_loss, validation_metrics, epoch_id)

            self.state_storage.set_value("start_epoch_id", epoch_id + 1)
            self.save_optimizer_state()
            self.save_last_model(epoch_id)

            epoch_id += 1

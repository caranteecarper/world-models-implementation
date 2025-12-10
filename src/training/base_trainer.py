import glob
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from torch import Tensor
from tqdm.notebook import tqdm

from src.metrics.wandb import WandbTrainingLogger, DummyWandbLogger


class BaseTrainer(ABC):
    def __init__(self,
                 model: torch.nn.Module,
                 weights_folder: str,
                 train_epoch_length: int,
                 num_epochs: int,
                 batch_size: int,
                 load_checkpoint: bool = False,
                 device: Optional[torch.device] = "cpu",
                 test_epoch_length: Optional[int] = 0,
                 wandb_setup: Optional[dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self.model = model
        self.weights_folder = weights_folder
        self.train_epoch_length = train_epoch_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.load_checkpoint = load_checkpoint
        self.device = device
        self.test_epoch_length = test_epoch_length
        self.model.to(self.device)
        self.start_epoch = 1
        self.total_trained_batches = 0
        if self.load_checkpoint:
            self.start_epoch, self.total_trained_batches = self.__load_checkpoint(self.model)
        self.wandb_setup = wandb_setup
        self.wandb_logger = self.__initialize_wandb(self.wandb_setup)
        self._logger.debug(f"Base Trainer initialized with: {self.__dict__}")

    def __load_checkpoint(self, model: torch.nn.Module) -> tuple[int, int]:
        checkpoints = glob.glob(os.path.join(self.weights_folder, "epoch_*.pth"))
        if not checkpoints:
            self._logger.info("No existing checkpoints found. Starting from scratch.")
            return 1, 0
        epochs = []
        for path in checkpoints:
            match = re.search(r'epoch_(\d+).pth', path)
            if match:
                epochs.append(int(match.group(1)))
        if not epochs:
            return 1, 0
        latest_epoch = max(epochs)
        checkpoint_path = os.path.join(self.weights_folder, f"epoch_{latest_epoch}.pth")
        self._logger.info(f"Resuming training from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        current_global_step = latest_epoch * self.train_epoch_length
        return latest_epoch + 1, current_global_step

    def __initialize_wandb(self, wandb_setup: Optional[dict[str, Any]]) -> WandbTrainingLogger:
        if wandb_setup is None:
            return DummyWandbLogger()
        api_key = self.__get_mandatory_argument(wandb_setup, "api_key")
        project = self.__get_mandatory_argument(wandb_setup, "project")
        run_name = self.__get_mandatory_argument(wandb_setup, "run_name")
        config = self.__get_mandatory_argument(wandb_setup, "config")
        resume = self.start_epoch > 1
        return WandbTrainingLogger(api_key, project, run_name, self.model, config, resume)

    def __get_mandatory_argument(self, kwargs: dict[str, Any], argument_name: str) -> Any:
        if argument_name not in kwargs:
            raise ValueError(f"Mandatory parameter {argument_name} is missing")
        argument_value = kwargs.get(argument_name)
        if argument_value is None:
            raise ValueError(f"Mandatory parameter {argument_name} is None")
        return argument_value

    def train(self) -> torch.nn.Module:
        self.best_epoch = -1
        self.best_epoch_loss = float('inf')
        epochs_loop = tqdm(total=self.num_epochs, desc="Epoch", unit="epoch")
        epochs_loop.n = self.start_epoch - 1
        self.train_loop = tqdm(total=self.train_epoch_length, desc=f"Train Epoch {self.start_epoch}", unit="batch")
        if self.test_epoch_length > 0:
            self.test_loop = tqdm(total=self.test_epoch_length, desc=f"Test Epoch {self.start_epoch}", unit="batch")
        try:
            for epoch in range(self.start_epoch, self.num_epochs + 1):
                epochs_loop.update(1)
                epochs_loop.set_description(f"Epoch {epoch}")
                self.train_loop.reset()
                if self.test_epoch_length > 0:
                    self.test_loop.reset()
                self.train_epoch(epoch)
                torch.save(self.model.state_dict(), os.path.join(self.weights_folder, f"epoch_{epoch}.pth"))
                if self.test_epoch_length > 0:
                    should_early_stop = self.test_epoch(epoch)
                    if should_early_stop:
                        break
        except Exception as e:
            self._logger.error(f"Interrupted training. An error occurred: {str(e)}")
        if self.best_epoch != -1:
            self._logger.info(f"Best epoch: {self.best_epoch}, Loss: {self.best_epoch_loss}")
            self.model.load_state_dict(torch.load(os.path.join(self.weights_folder, f"epoch_{self.best_epoch}.pth")))
        torch.save(self.model.state_dict(), os.path.join(self.weights_folder, f"model.pth"))
        self.wandb_logger.finish()
        return self.model
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> None:
        """
        Runs one epoch of training for the model.
        """
        self._logger.error("Method train_epoch was called with no implementation")
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def test_epoch(self, epoch: int) -> bool:
        """
        Runs testing for one epoch of training for the model.
        """
        self._logger.error("Method test_epoch was called with no implementation")
        raise NotImplementedError("This method should be implemented by subclasses")
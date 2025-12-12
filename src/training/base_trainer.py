import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
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
            self.__load_checkpoint(self.model)
        self.wandb_setup = wandb_setup
        self.wandb_logger = self.__initialize_wandb(self.wandb_setup)
        self._logger.debug(f"Base Trainer initialized with: {self.__dict__}")

    def __load_checkpoint(self, model: torch.nn.Module) -> None:
        self.epoch_already_trained = False
        metadata = self._get_metadata()
        self._logger.debug(f"Loaded Training Metadata: {metadata}")
        self.best_epoch = metadata.get("best_epoch", -1)
        self.best_epoch_loss = metadata.get("best_epoch_loss", float('inf'))
        self.total_trained_batches = metadata.get("total_trained_batches", 0)
        if self.total_trained_batches == 0:
            self._logger.info("No existing checkpoints found. Starting from scratch.")
            self.start_epoch = 1
            return
        epoch_metadata = metadata.get("epochs", {})
        trained_epochs = [int(epoch) for epoch in epoch_metadata.keys()]
        latest_epoch = max(trained_epochs)
        latest_epoch_metadata = epoch_metadata.get(str(latest_epoch), {})
        latest_epoch_path = os.path.join(self.weights_folder, latest_epoch_metadata.get("path"))
        self._logger.info(f"Resuming training from: {latest_epoch_path}")
        model.load_state_dict(torch.load(latest_epoch_path, map_location=self.device))
        if latest_epoch_metadata.get("loss") is None:
            self.epoch_already_trained = True
            self.start_epoch = latest_epoch
            return
        self.start_epoch = latest_epoch + 1
    
    def _get_metadata(self) -> dict:
        metadata_path = os.path.join(self.weights_folder, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)
                return metadata
        else:
            return {
                "best_epoch": -1,
                "best_epoch_loss": float('inf'),
                "total_trained_batches": 0,
                "epochs": {}
            }
        
    def _update_metadata_epoch(self, epoch: int, update: dict[str, Any]) -> None:
        metadata = self._get_metadata()
        epochs_metadata = metadata.get("epochs", {})
        current_epoch_metadata = epochs_metadata.get(str(epoch), {})
        current_epoch_metadata.update(update)
        epochs_metadata[str(epoch)] = current_epoch_metadata
        metadata["epochs"] = epochs_metadata
        metadata["total_trained_batches"] = self.total_trained_batches
        self._save_metadata(metadata)

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        metadatafile_path = os.path.join(self.weights_folder, "metadata.json")
        with open(metadatafile_path, "w") as metadata_file:
            json.dump(metadata, metadata_file)

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
                if not self.epoch_already_trained:
                    self.train_epoch(epoch)
                    self._save_model(epoch)
                if self.test_epoch_length > 0:
                    should_early_stop = self.test_epoch(epoch)
                    if should_early_stop:
                        break
                self.epoch_already_trained = False
        except KeyboardInterrupt:
            self._logger.info("Training interrupted manually by user. Stopping gracefully...")
        except Exception as e:
            self._logger.error(f"Interrupted training. An error occurred: {str(e)}")
        self.train_loop.close()
        if self.test_epoch_length > 0:
            self.test_loop.close()
        self._load_best_epoch()
        torch.save(self.model.state_dict(), os.path.join(self.weights_folder, f"model.pth"))
        self.wandb_logger.finish()
        return self.model
    
    def _save_model(self, epoch: int) -> None:
        epoch_file_name = f"epoch_{epoch}.pth"
        torch.save(self.model.state_dict(), os.path.join(self.weights_folder, epoch_file_name))
        self._update_metadata_epoch(epoch, {"path": epoch_file_name})

    def _evaluate_best_epoch(self, epoch: int, epoch_loss: float) -> None:
        self._update_metadata_epoch(epoch, {"loss": epoch_loss})
        if epoch_loss < self.best_epoch_loss:
            self.best_epoch = epoch
            self.best_epoch_loss = epoch_loss
            metadata = self._get_metadata()
            metadata["best_epoch"] = self.best_epoch
            metadata["best_epoch_loss"] = self.best_epoch_loss
            self._save_metadata(metadata)

    def _load_best_epoch(self) -> None:
        if self.best_epoch != -1:
            self._logger.info(f"Best epoch: {self.best_epoch}, Loss: {self.best_epoch_loss}")
            self.model.load_state_dict(torch.load(os.path.join(self.weights_folder, f"epoch_{self.best_epoch}.pth")))
    
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
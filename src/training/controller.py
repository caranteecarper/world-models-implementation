import logging
from typing import Any, Optional

import torch
import numpy as np
from cma import CMAEvolutionStrategy

from src.models.simulation_worldmodel import SimulationWorldModel
from src.training.base_trainer import BaseTrainer


class ControllerEvolutionaryTrainer(BaseTrainer):
    def __init__(self,
                 model: torch.nn.Module,
                 weights_folder: str,
                 num_generations: int,
                 population_size: int,
                 simulation_world_model: SimulationWorldModel,
                 steps_per_rollout: int,
                 rollouts_per_solution: int,
                 sigma_init: Optional[float] = 0.1,
                 load_checkpoint: bool = False,
                 device: Optional[torch.device] = "cpu",
                 wandb_setup: Optional[dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(model=model,
                         weights_folder=weights_folder,
                         train_epoch_length=population_size,
                         num_epochs=num_generations,
                         batch_size=rollouts_per_solution,
                         load_checkpoint=load_checkpoint,
                         device=device,
                         wandb_setup=wandb_setup,
                         logger=logger)
        self.simulation_world_model = simulation_world_model
        self.steps_per_rollout = steps_per_rollout
        self.rollouts_per_solution = rollouts_per_solution
        initial_params = self.get_flat_parameters(self.model)
        self.evolution_strategy = CMAEvolutionStrategy(initial_params, sigma_init, {'popsize': population_size})

    def get_flat_parameters(self, model: torch.nn.Module) -> np.ndarray:
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_parameters(self, model: torch.nn.Module, flat_params: np.ndarray, device: torch.device):
        offset = 0
        for param in model.parameters():
            param_shape = param.data.shape
            param_size = np.prod(param_shape)
            param_data = flat_params[offset : offset + param_size]
            param.data = torch.tensor(param_data, device=device, dtype=torch.float32).view(param_shape)
            offset += param_size

    def rollout(self) -> float:
        self.simulation_world_model.reset()
        actions = torch.zeros(self.rollouts_per_solution, 3).to(self.device)
        observations, rewards = self.simulation_world_model.predict_next_state(action=actions)
        hidden_states = self.simulation_world_model.hidden
        cumulative_reward = 0.0
        with torch.no_grad():
            for _ in range(self.steps_per_rollout):
                actions = self.model(observations, hidden_states[0].squeeze(0))
                observations, rewards = self.simulation_world_model.predict_next_state(action=actions)
                hidden_states = self.simulation_world_model.hidden
                cumulative_reward = cumulative_reward + rewards
        loss = -torch.mean(cumulative_reward)
        return loss.item()
    
    def train_epoch(self, epoch: int) -> None:
        self.model.eval()
        self.train_loop.set_description(f"Train Epoch {epoch}")
        solutions = self.evolution_strategy.ask()
        losses = []
        for solution in solutions:
                self.train_loop.update(1)
                self.total_trained_batches += 1
                self.wandb_logger.set_step(self.total_trained_batches)
                self.set_flat_parameters(self.model, solution, self.device)
                loss = self.rollout()
                losses.append(loss)
                self.train_loop.set_description(f"Train Epoch {epoch} Loss: {loss:.4f}")
                self.wandb_logger.log({
                    "train/loss": loss,
                    "epoch": epoch
                })
        best_solution = solutions[np.argmin(losses)]
        self.set_flat_parameters(self.model, best_solution, self.device)
        self.evolution_strategy.tell(solutions, losses)
        avg_reward = -np.mean(losses)
        max_reward = -np.min(losses)
        self._logger.info(f"Epoch {epoch} Training Average Reward: {avg_reward:.4f}")
        self._logger.info(f"Epoch {epoch} Training Max Reward: {max_reward:.4f}")
        self.wandb_logger.log({
            "train/avg_reward": avg_reward,
            "train/max_reward": max_reward,
        })

    def test_epoch(self, epoch: int) -> bool:
        return False
from __future__ import annotations

import json
from logging import Logger
from typing import Optional, Union

import torch
import numpy as np

from src.models.vae import ConvVAE
from src.models.worldmodel import MdnRnn
from src.utils.logging import get_logger

class SimulationWorldModel():  
    def __init__(self,
                 worldmodel_path: str,
                 settings_path: str,
                 vae_path: Optional[str] = None,
                 device: Optional[torch.device] = "cpu",
                 batch_size: int = 1,
                 starting_observation: Optional[torch.Tensor] = None,
                 starting_observation_representation: Optional[torch.Tensor] = None,
                 starting_reward: Optional[float] = 0.0,
                 starting_hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                 logger: Optional[Logger] = None):
        self.logger = logger or get_logger()
        self.get_model_settings(settings_path)
        self.device = device
        self.logger.debug(f"WorldModel device: {self.device}")
        self.batch_size = batch_size
        if vae_path is not None:
            self.vae = ConvVAE(image_channels=self.image_channels,
                               h_dim=self.vae_hidden_dim,
                               z_dim=self.representation_dim,
                               device=self.device,
                               weights_path=vae_path)
            self.vae.freeze_weights().eval()
        else:
            self.vae = None
        self.worldmodel = MdnRnn(input_size=self.rnn_input_dim,
                                 hidden_size=self.rnn_hidden_dim,
                                 output_size=self.rnn_output_dim,
                                 num_gaussians=self.rnn_num_gaussians,
                                 min_sigma=self.rnn_min_sigma,
                                 max_sigma=self.rnn_max_sigma,
                                 device=self.device,
                                 weights_path=worldmodel_path)
        self.worldmodel.freeze_weights().eval()
        self.reset(starting_observation, starting_observation_representation, starting_reward, starting_hidden_state)

    def get_model_settings(self, settings_path: str):
        with open(settings_path, "r") as settings_file:
            settings = json.load(settings_file)
            self.image_channels = settings["vae"]["model"]["image_channels"]
            self.logger.debug(f"image_channels: {self.image_channels}")
            self.vae_hidden_dim = settings["vae"]["model"]["hidden_dim"]
            self.logger.debug(f"vae_hidden_dim: {self.vae_hidden_dim}")
            self.representation_dim = settings["vae"]["model"]["representation_dim"]
            self.logger.debug(f"representation_dim: {self.representation_dim}")
            self.rnn_hidden_dim = settings["world_model"]["model"]["hidden_dim"]
            self.logger.debug(f"rnn_hidden_dim: {self.rnn_hidden_dim}")
            self.rnn_num_gaussians = settings["world_model"]["model"]["num_gaussians"]
            self.logger.debug(f"rnn_num_gaussians: {self.rnn_num_gaussians}")
            self.rnn_min_sigma = settings["world_model"]["model"]["min_sigma"]
            self.logger.debug(f"rnn_min_sigma: {self.rnn_min_sigma}")
            self.rnn_max_sigma = settings["world_model"]["model"]["max_sigma"]
            self.logger.debug(f"rnn_max_sigma: {self.rnn_max_sigma}")
            self.rnn_input_state_dim = settings["world_model"]["model"]["input_state_dim"]
            self.logger.debug(f"rnn_input_state_dim: {self.rnn_input_state_dim}")
            self.rnn_output_state_dim = settings["world_model"]["model"]["output_state_dim"]
            self.logger.debug(f"rnn_output_state_dim: {self.rnn_output_state_dim}")
            self.rnn_input_dim = self.representation_dim + self.rnn_input_state_dim
            self.logger.debug(f"rnn_input_dim: {self.rnn_input_dim}")
            self.rnn_output_dim = self.representation_dim + self.rnn_output_state_dim
            self.logger.debug(f"rnn_output_dim: {self.rnn_output_dim}")

    def reset(self,
              starting_observation: Optional[torch.Tensor] = None,
              starting_observation_representation: Optional[torch.Tensor] = None,
              starting_reward: Optional[float] = 0.0,
              starting_hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None):
        if starting_observation is not None and starting_observation_representation is not None:
            raise ValueError("Cannot specify both starting_observation and starting_observation_representation")
        if starting_observation is not None and self.vae is None:
            raise ValueError("VAE is not initialized")
        if starting_observation is not None:
            with torch.no_grad():
                self.current_observation_representation, _, _ = self.vae.encode(starting_observation.to(self.device).unsqueeze(0))
        if starting_observation_representation is not None:
            self.current_observation_representation = starting_observation_representation.to(self.device)
        else:
            self.current_observation_representation = torch.randn(self.batch_size, self.representation_dim).to(self.device)
        self.current_reward = torch.full(
            (self.batch_size, 1), 
            starting_reward, 
            device=self.device
        )
        if starting_hidden_state is not None:
            self.hidden = starting_hidden_state
        else:
            self.h0 = torch.zeros(1, self.batch_size, self.rnn_hidden_dim).to(self.device)
            self.c0 = torch.zeros(1, self.batch_size, self.rnn_hidden_dim).to(self.device)
            self.hidden = (self.h0, self.c0)
        return self

    def predict_next_state(self,
                           action: Union[np.ndarray, torch.Tensor],
                           current_reward: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if current_reward is not None:
                if isinstance(current_reward, (float, int)):
                    self.current_reward = torch.full((self.batch_size, 1), current_reward, device=self.device)
                else:
                    self.current_reward = torch.tensor(current_reward, dtype=torch.float32, device=self.device).view(self.batch_size, 1)
            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32).to(self.device)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            rnn_input = torch.cat([self.current_observation_representation, action, self.current_reward], dim=1).unsqueeze(1)
            pi, _, mu, self.hidden = self.worldmodel(rnn_input, self.hidden)
            k = torch.argmax(pi, dim=-1)
            batch_indices = torch.arange(self.batch_size, device=self.device)
            gaussian_indices = k.squeeze(-1)
            next_data = mu[batch_indices, 0, gaussian_indices, :].unsqueeze(1)
            self.current_observation_representation = next_data[:, :, :self.representation_dim].squeeze(1)
            self.current_reward = next_data[:, :, self.representation_dim:].squeeze(1)
            self.current_observation_representation = torch.nan_to_num(
                self.current_observation_representation,
                nan=0.0,
                posinf=30.0,
                neginf=-30.0
            ).clamp(-30.0, 30.0)
            self.current_reward = torch.nan_to_num(
                self.current_reward,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0
            ).clamp(-10.0, 10.0)
            return self.current_observation_representation, self.current_reward
        
    def predict_next_frame(self,
                           action: Union[np.ndarray, torch.Tensor],
                           current_reward: Optional[float] = None) -> tuple[np.ndarray, torch.Tensor]:
        if self.vae is None:
            raise ValueError("VAE is not initialized")
        observation_representation, reward = self.predict_next_state(action, current_reward)
        with torch.no_grad():
            reconstructed_image = self.vae.decode(observation_representation)
        image_numpy = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_numpy = np.clip(image_numpy, 0, 1) * 255
        image_numpy = image_numpy.astype(np.uint8)
        return image_numpy, reward

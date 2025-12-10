from logging import Logger, getLogger
from typing import Optional, Union

import torch
import numpy as np

from src.models.vae import ConvVAE
from src.models.worldmodel import MdnRnn

class SimulationWorldModel():  
    def __init__(self,
                 worldmodel_path: str,
                 vae_path: Optional[str] = None,
                 device: Optional[torch.device] = "cpu",
                 batch_size: int = 1,
                 starting_observation: Optional[torch.Tensor] = None,
                 starting_observation_representation: Optional[torch.Tensor] = None,
                 starting_reward: Optional[float] = 0.0,
                 starting_hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                 image_channels: int = 3,
                 vae_h_dim: int = 1024,
                 vae_z_dim: int = 32,
                 rnn_hidden_dim: int = 256,
                 num_gaussians: int = 5,
                 action_dim: int = 3,
                 reward_dim: int = 1,
                 logger: Optional[Logger] = None):
        self.logger = logger or getLogger(__name__)
        self.device = device
        logger.debug(f"WorldModel device: {self.device}")
        self.is_training = False
        self.batch_size = batch_size
        self.vae_z_dim = vae_z_dim
        if vae_path is not None:
            logger.debug(f"Creating VAE with: image_channels={image_channels} vae_h_dim={vae_h_dim} vae_z_dim={vae_z_dim}")
            self.vae = ConvVAE(image_channels, vae_h_dim, vae_z_dim, device=self.device, weights_path=vae_path)
            self.vae.freeze_weights().eval()
        else:
            self.vae = None
        rnn_input_dim = vae_z_dim + action_dim + reward_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        rnn_output_dim = vae_z_dim + reward_dim
        logger.debug(f"Creating RNN with: rnn_input_dim={rnn_input_dim} rnn_hidden_dim={rnn_hidden_dim} rnn_output_dim={rnn_output_dim} num_gaussians={num_gaussians}")
        self.worldmodel = MdnRnn(rnn_input_dim, rnn_hidden_dim, rnn_output_dim, num_gaussians, device=self.device, weights_path=worldmodel_path)
        self.worldmodel.freeze_weights().eval()
        self.reset(starting_observation, starting_observation_representation, starting_reward, starting_hidden_state)        

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
            self.current_observation_representation = torch.randn(self.batch_size, self.vae_z_dim).to(self.device)
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
    
    def train(self):
        self.is_training = True
        self.worldmodel.train()
        if self.vae is not None:
            self.vae.train()

    def eval(self):
        self.is_training = False
        self.worldmodel.eval()
        if self.vae is not None:
            self.vae.eval()

    def predict_next_state(self,
                           action: Union[np.ndarray, torch.Tensor],
                           current_reward: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(self.is_training):
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
            self.current_observation_representation = next_data[:, :, :self.vae_z_dim].squeeze(1)
            self.current_reward = next_data[:, :, self.vae_z_dim:].squeeze(1)
            return self.current_observation_representation, self.current_reward
        
    def predict_next_frame(self,
                           action: Union[np.ndarray, torch.Tensor],
                           current_reward: Optional[float] = None) -> tuple[np.ndarray, torch.Tensor]:
        if self.vae is None:
            raise ValueError("VAE is not initialized")
        observation_representation, reward = self.predict_next_state(action, current_reward)
        with torch.set_grad_enabled(self.is_training):
            reconstructed_image = self.vae.decode(observation_representation)
        image_numpy = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_numpy = np.clip(image_numpy, 0, 1) * 255
        image_numpy = image_numpy.astype(np.uint8)
        return image_numpy, reward
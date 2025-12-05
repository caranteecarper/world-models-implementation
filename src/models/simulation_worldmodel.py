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
        self.vae_z_dim = vae_z_dim
        if vae_path is not None:
            logger.debug(f"Creating VAE with: image_channels={image_channels} vae_h_dim={vae_h_dim} vae_z_dim={vae_z_dim}")
            self.vae = ConvVAE(image_channels, vae_h_dim, vae_z_dim, device=self.device, weights_path=vae_path)
            self.vae.freeze_weights().eval()
        else:
            self.vae = None
        rnn_input_dim = vae_z_dim + action_dim + reward_dim
        rnn_output_dim = vae_z_dim + reward_dim
        logger.debug(f"Creating RNN with: rnn_input_dim={rnn_input_dim} rnn_hidden_dim={rnn_hidden_dim} rnn_output_dim={rnn_output_dim} num_gaussians={num_gaussians}")
        self.worldmodel = MdnRnn(rnn_input_dim, rnn_hidden_dim, rnn_output_dim, num_gaussians, device=self.device, weights_path=worldmodel_path)
        self.worldmodel.freeze_weights().eval()
        if starting_observation_representation is not None:
            self.current_observation_representation = starting_observation_representation.to(self.device)
        else:
            self.current_observation_representation = torch.randn(1, vae_z_dim).to(self.device)
        self.current_reward = torch.tensor(starting_reward).to(self.device).unsqueeze(0).unsqueeze(0)
        if starting_hidden_state is not None:
            self.hidden = starting_hidden_state
        else:
            self.h0 = torch.zeros(1, 1, rnn_hidden_dim).to(self.device)
            self.c0 = torch.zeros(1, 1, rnn_hidden_dim).to(self.device)
            self.hidden = (self.h0, self.c0)

    def reset(self):
        self.current_observation_representation = torch.randn(1, self.vae_z_dim).to(self.device)
        self.current_reward = torch.tensor(0.0).to(self.device)
        self.hidden = (self.h0, self.c0)

    def predict_next_state(self, action: Union[np.ndarray, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)
            rnn_input = torch.cat([self.current_observation_representation, action, self.current_reward], dim=1).unsqueeze(1) 
            pi, _, mu, self.hidden = self.worldmodel(rnn_input, self.hidden)
            k = torch.argmax(pi, dim=-1).item()
            next_data = mu[:, :, k, :] 
            self.current_observation_representation = next_data[:, :, :self.vae_z_dim].squeeze(1)
            self.current_reward = next_data[:, :, self.vae_z_dim:].squeeze(1)
            return self.current_observation_representation, self.current_reward
        
    def predict_next_frame(self, action: Union[np.ndarray, torch.Tensor]) -> tuple[np.ndarray, torch.Tensor]:
        if self.vae is None:
            raise ValueError("VAE is not initialized")
        observation_representation, reward = self.predict_next_state(action)
        with torch.no_grad():
            reconstructed_image = self.vae.decode(observation_representation)
        image_numpy = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_numpy = np.clip(image_numpy, 0, 1) * 255
        image_numpy = image_numpy.astype(np.uint8)
        return image_numpy, reward
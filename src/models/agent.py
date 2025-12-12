import json
from logging import Logger
from typing import Optional

import torch
import numpy as np

from src.models.vae import ConvVAE
from src.models.worldmodel import MdnRnn
from src.models.controller import Controller
from src.utils.logging import get_logger

class Agent():
    def __init__(self,
                 vae_path: str,
                 worldmodel_path: str,
                 controller_path: str,
                 device: torch.device,
                 logger: Optional[Logger] = None):
        self.logger = logger or get_logger()
        self.device = device
        logger.debug(f"WorldModel device: {self.device}")
        self.vae = ConvVAE(image_channels=self.image_channels,
                           h_dim=self.vae_hidden_dim,
                           z_dim=self.representation_dim,
                           device=self.device,
                           weights_path=vae_path)
        self.vae.freeze_weights().eval()
        self.worldmodel = MdnRnn(input_size=self.rnn_input_dim,
                                 hidden_size=self.rnn_hidden_dim,
                                 output_size=self.rnn_output_dim,
                                 num_gaussians=self.rnn_num_gaussians,
                                 min_sigma=self.rnn_min_sigma,
                                 max_sigma=self.rnn_max_sigma,
                                 device=self.device,
                                 weights_path=worldmodel_path)
        self.worldmodel.freeze_weights().eval()
        self.controller = Controller(observation_dim=self.representation_dim,
                                     hidden_dim=self.rnn_hidden_dim,
                                     action_dim=self.action_dim,
                                     device=self.device,
                                     weights_path=controller_path)
        self.reset()

    def get_model_settings(self, settings_path: str):
        with open(settings_path, "r") as settings_file:
            settings = json.load(settings_file)
            self.observation_crop_dim = settings["data_ingestion"]["observation_crop_dim"]
            self.logger.debug(f"observation_crop_dim: {self.observation_crop_dim}")
            self.observation_dim = settings["vae"]["model"]["observation_dim"]
            self.logger.debug(f"observation_dim: {self.observation_dim}")
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
            self.action_dim = settings["controller"]["model"]["action_dim"]
            self.logger.debug(f"action_dim: {self.action_dim}")


    def reset(self):
        self.h0 = torch.zeros(1, 1, self.rnn_hidden_dim).to(self.device)
        self.c0 = torch.zeros(1, 1, self.rnn_hidden_dim).to(self.device)
        self.hidden = (self.h0, self.c0)

    def __rescale_observation_to_tensor(self, observation: np.ndarray):
        tensor = torch.from_numpy(observation).permute(2, 0, 1).float() / 255.0
        cropped_tensor = tensor[:, :self.observation_crop_dim, :].to(self.device)
        resized_tensor = torch.nn.functional.interpolate(
            cropped_tensor.unsqueeze(0),
            size=(self.observation_dim, self.observation_dim),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        return resized_tensor

    def step(self, observation, reward, previous_action):
        with torch.no_grad():
            observation_tensor = self.__rescale_observation_to_tensor(observation).unsqueeze(0).to(self.device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(self.device)
            action_tensor = torch.tensor(previous_action, dtype=torch.float32).unsqueeze(0).to(self.device)
            observation_representation, _, _ = self.vae.encode(observation_tensor)
            rnn_input = torch.cat([observation_representation, action_tensor, reward_tensor], dim=1)
            _, _, _, self.hidden = self.worldmodel(rnn_input.unsqueeze(0), self.hidden)
            actions = self.controller(observation_representation, self.hidden[0].squeeze(0))
            next_action = actions.cpu().numpy()[0]
        return next_action
    
    def to(self, device):
        self.device = device
        self.vae.to(device)
        self.worldmodel.to(device)
        self.controller.to(device)
        return self
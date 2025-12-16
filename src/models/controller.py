from typing import Optional

import torch

class Controller(torch.nn.Module):
    def __init__(self,
                 observation_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 device: Optional[torch.device] = "cpu",
                 weights_path: Optional[str] = None):
        super(Controller, self).__init__()
        self.device = device
        dim_with_brake_and_accel_together = action_dim - 1
        self.fc = torch.nn.Linear(observation_dim + hidden_dim, dim_with_brake_and_accel_together).to(self.device)
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location=self.device))

    def forward(self, observation, hidden_state):
        x = torch.cat([observation, hidden_state], dim=1)
        x = self.fc(x)
        steering = torch.tanh(x[:, 0:1])
        acceleration = torch.tanh(x[:, 1:2])
        gas = torch.nn.functional.relu(acceleration)
        brake = torch.nn.functional.relu(-1*acceleration)
        return torch.cat([steering, gas, brake], dim=1)
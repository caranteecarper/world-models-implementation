import numpy as np

import gymnasium as gym

from src.models.agent import Agent
from src.utils.torch import get_device
from src.utils.logging import get_logger

VAE_PATH = "./weights/vae/model.pth"
RNN_PATH = "./weights/worldmodel/model.pth"
CONTROLLER_PATH = "./weights/controller/model.pth"

IMAGE_CHANNELS = 3
VAE_HIDDEN_DIMENSION = 1024
OBSERVATION_REPRESENTATION_DIMESNION = 32
RNN_HIDDEN_DIMENSION = 256
ACTION_DIMENSION = 3
REWARD_DIMENSION = 1
RNN_OUTPUT_DIMENSION = OBSERVATION_REPRESENTATION_DIMESNION + REWARD_DIMENSION
OBSERVATION_Y_CROP = 83
OBSERVATION_DIMENSION = 64

LOG_LEVEL = "DEBUG"


logger = get_logger()
device = get_device()

agent = Agent(
    vae_path=VAE_PATH,
    rnn_path=RNN_PATH,
    controller_path=CONTROLLER_PATH,
    device=device,
    image_channels=IMAGE_CHANNELS,
    vae_h_dim=VAE_HIDDEN_DIMENSION,
    vae_z_dim=OBSERVATION_REPRESENTATION_DIMESNION,
    rnn_hidden_dim=RNN_HIDDEN_DIMENSION,
    action_dim=ACTION_DIMENSION,
    reward_dim=REWARD_DIMENSION,
    observation_y_crop=OBSERVATION_Y_CROP,
    observation_dim=OBSERVATION_DIMENSION,
    logger=logger
)

env = gym.make("CarRacing-v3",
                render_mode="human",
                lap_complete_percent=0.95,
                domain_randomize=False,
                continuous=True,
                max_episode_steps=-1)
observation, _ = env.reset()
action = np.array([0.0, 0.0, 0.0])
negative_reward_streak = 0
reward = 0
while True:
    action = agent.step(observation, reward, action)
    observation, reward, terminated, truncated, info = env.step(action)
    if reward < 0:
        negative_reward_streak += 1
    else:
        negative_reward_streak = 0
    if negative_reward_streak > 300:
        logger.debug("Aborting run: Car is stuck off-road.")
        break
    if terminated or truncated:
        break
env.close()
import os

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from src.models.agent import Agent
from src.utils.torch import get_device
from src.utils.logging import get_logger


LOG_LEVEL = "INFO"
logger = get_logger(LOG_LEVEL)

project_folder = "./"
settings_path = os.path.join(project_folder, "settings.json")
vae_path = os.path.join(project_folder, "weights/vae/model.pth")
worldmodel_path = os.path.join(project_folder, "weights/worldmodel/model.pth")
controller_path = os.path.join(project_folder, "weights/controller/model.pth")

device = get_device()

agent = Agent(
    settings_path=settings_path,
    vae_path=vae_path,
    worldmodel_path=worldmodel_path,
    controller_path=controller_path,
    device=device,
    logger=logger)

env = gym.make("CarRacing-v3",
                render_mode="rgb_array",
                lap_complete_percent=0.95,
                domain_randomize=False,
                continuous=True,
                max_episode_steps=-1)
env = RecordVideo(env, video_folder="./", name_prefix="test_final_worldmodel",episode_trigger=lambda x: True)
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
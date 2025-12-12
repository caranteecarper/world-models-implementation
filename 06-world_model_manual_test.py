import json
import os

import numpy as np
import pygame

from src.models.simulation_worldmodel import SimulationWorldModel
from src.utils.logging import get_logger
from src.utils.torch import get_device


LOG_LEVEL = "INFO"
logger = get_logger(LOG_LEVEL)

project_folder = "./"
settings_path = os.path.join(project_folder, "settings.json")
vae_path = os.path.join(project_folder, "weights/vae/model.pth")
worldmodel_path = os.path.join(project_folder, "weights/worldmodel/model.pth")

with open(settings_path, "r") as settings_file:
    settings = json.load(settings_file)
    OBSERVATION_ORIGINAL_DIM = settings["data_ingestion"]["observation_original_dim"]
    OBSERVATION_CROP_DIM = settings["data_ingestion"]["observation_crop_dim"]
    DISPLAY_SCALE = 6
    DISPLAY_SIZE = (OBSERVATION_ORIGINAL_DIM * DISPLAY_SCALE, OBSERVATION_CROP_DIM * DISPLAY_SCALE)
    CAPTION = "World Model Dream"
    logger.debug(f"OBSERVATION_ORIGINAL_DIM: {OBSERVATION_ORIGINAL_DIM}")
    logger.debug(f"OBSERVATION_CROP_DIM: {OBSERVATION_CROP_DIM}")
    logger.debug(f"DISPLAY_SCALE: {DISPLAY_SCALE}")
    logger.debug(f"DISPLAY_SIZE: {DISPLAY_SIZE}")
    logger.debug(f"CAPTION: {CAPTION}")

DEVICE = get_device(logger)

logger.info("--- World Model Dream Control ---")
logger.info("Left/Right Arrows: Steer")
logger.info("Up Arrow: Accelerate")
logger.info("Down Arrow: Brake")
logger.info("R: Reset Dream (New Random Z)")
logger.info("Q: Quit")
logger.info("---------------------------------")

pygame.init()
screen = pygame.display.set_mode(DISPLAY_SIZE)
pygame.display.set_caption(CAPTION)
clock = pygame.time.Clock()

simulation_worldmodel = SimulationWorldModel(worldmodel_path=worldmodel_path,
                                             settings_path=settings_path,
                                             vae_path=vae_path,
                                             device=DEVICE,
                                             logger=logger)

action = np.array([0.0, 0.0, 0.0])
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_r:
                logger.info("Resetting dream...")
                simulation_worldmodel.reset()
                action[:] = 0.0

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action[0] = -1.0
    elif keys[pygame.K_RIGHT]:
        action[0] = +1.0
    else:
        action[0] = 0.0
    if keys[pygame.K_UP]:
        action[1] = 1.0
    else:
        action[1] = 0.0
    if keys[pygame.K_DOWN]:
        action[2] = 0.8
    else:
        action[2] = 0.0
    next_frame, next_reward = simulation_worldmodel.predict_next_frame(action)
    surface = pygame.surfarray.make_surface(np.transpose(next_frame, (1, 0, 2)))
    surface = pygame.transform.scale(surface, DISPLAY_SIZE)
    screen.fill((0,0,0))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
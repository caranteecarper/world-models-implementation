import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame

from src.models.simulation_worldmodel import SimulationWorldModel
from src.utils.logging import get_logger


VAE_PATH = "./weights/vae/model.pth"
RNN_PATH = "./weights/worldmodel/model.pth"

NATIVE_SIZE = 64
CROP_SIZE = (96, 83)
DISPLAY_SCALE = 6
DISPLAY_SIZE = (CROP_SIZE[0] * DISPLAY_SCALE, CROP_SIZE[1] * DISPLAY_SCALE)


logger = get_logger(level="INFO")

DEVICE = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else DEVICE)
logger.info(f"Using device: {DEVICE}")

pygame.init()
screen = pygame.display.set_mode(DISPLAY_SIZE)
pygame.display.set_caption("World Model Dream")
clock = pygame.time.Clock()

simulation_worldmodel = SimulationWorldModel(worldmodel_path=RNN_PATH, vae_path=VAE_PATH, device=DEVICE, logger=logger)

action = np.array([0.0, 0.0, 0.0])

print("--- World Model Dream Control ---")
print("Left/Right Arrows: Steer")
print("Up Arrow: Accelerate")
print("Down Arrow: Brake")
print("R: Reset Dream (New Random Z)")
print("Q: Quit")
print("---------------------------------")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_r:
                print("Resetting dream...")
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
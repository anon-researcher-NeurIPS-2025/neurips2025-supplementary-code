import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sn

from src.environment.apple_grid import AppleGridMDP
from src.agents.ppo_agent import PPOAgent

# Configuration
TILE_SIZE = 40
FPS = 10
np.random.seed(48)
random.seed(48)

title_height = 50
color_map = {
    '0': (87, 174, 209),
    '1': (194, 38, 121),
    '#': (0, 0, 255),
    '*': (117, 199, 249),
}

# Drawing function
def draw_grid(surface, grid, offset_x=0):
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            screen_x = offset_x + x * TILE_SIZE
            screen_y = y * TILE_SIZE + title_height
            if cell == '1':
                surface.blit(apple_image, (screen_x, screen_y))
            elif cell == '#':
                surface.blit(agent_1_image, (screen_x, screen_y))
            elif cell == '*':
                surface.blit(agent_2_image, (screen_x, screen_y))
            else:
                color = color_map.get(cell, (255, 0, 255))
                pygame.draw.rect(surface, color, (screen_x, screen_y, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(surface, (250, 250, 250), (screen_x, screen_y, TILE_SIZE, TILE_SIZE), 1)

# Load environments and agents
env_test = AppleGridMDP(rate_regen=0.06)
env_random = AppleGridMDP(rate_regen=0.06)

grid_size = env_test.grid_size[0] * env_test.grid_size[1]
agent1 = PPOAgent(4 + grid_size, 4)
agent2 = PPOAgent(4 + grid_size, 4)

# -----------------------------------------------
# Load pretrained PPO agents
# -----------------------------------------------
# The agents used here are loaded from the 'models/' folder.
# If you're looking for the best-performing agents (based on different reward settings),
# check the 'models/best' folder.
# Additional variants can be found in:
# - 'models/baseline/ppo : agents trained with tradicional rewards apple consumption
# - 'models/hybrid'     : agents trained with hybrid rewards
# - 'models/resilience' : agents trained under each resilience parametrization
# - 'models/example'    : dummy/testing agents not used in the paper

agent1.model.load_state_dict(torch.load("models/best/agent1.pth"))
agent2.model.load_state_dict(torch.load("models/best/agent2.pth"))

print("Models loaded successfully.")

# Initialize pygame
pygame.init()
font = pygame.font.SysFont(None, 36)
title_font = pygame.font.SysFont("Verdana", 20, bold=True)
score_font = pygame.font.SysFont("Verdana", 18)

rows, cols = env_test.grid.shape
screen_width = (cols * TILE_SIZE) * 2
screen_height = rows * TILE_SIZE + title_height
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Load images
ASSETS = "visualization/assets"
apple_image = pygame.image.load(os.path.join(ASSETS, "apple.png")).convert_alpha()
apple_image = pygame.transform.scale(apple_image, (TILE_SIZE, TILE_SIZE))
agent_1_image = pygame.image.load(os.path.join(ASSETS, "agent1.png")).convert_alpha()
agent_1_image = pygame.transform.scale(agent_1_image, (TILE_SIZE, TILE_SIZE))
agent_2_image = pygame.image.load(os.path.join(ASSETS, "agent2.png")).convert_alpha()
agent_2_image = pygame.transform.scale(agent_2_image, (TILE_SIZE, TILE_SIZE))

# Initialize game state
state_test = env_test.get_state()
state_random = env_random.get_state()

score_test = [0, 0]
score_random = [0, 0]
board_agent1 = np.zeros((8, 8))
board_agent2 = np.zeros((8, 8))
board_agent1r = np.zeros((8, 8))
board_agent2r = np.zeros((8, 8))

# Main loop
for iteration in range(500):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    if iteration in [140, 240]:
        disrupcion_activada = True
        disrupcion_timer = 150
        env_test.trigger_disruption()
        env_random.trigger_disruption()
    else:
        disrupcion_activada = False

    game_over_test = np.sum(env_test.grid == 1) == 0
    game_over_random = np.sum(env_random.grid == 1) == 0

    if not game_over_test:
        state_tensor = torch.tensor(state_test, dtype=torch.float32).unsqueeze(0)
        action1, _ = agent1.select_action(state_tensor)
        action2, _ = agent2.select_action(state_tensor)
        state_test, rewards_test = env_test.step([action1, action2])
        for i, reward in enumerate(rewards_test):
            if reward > 0:
                score_test[i] += reward

    if not game_over_random:
        actions_random = [random.randint(0, 3), random.randint(0, 3)]
        state_random, rewards_random = env_random.step(actions_random)
        for i, reward in enumerate(rewards_random):
            if reward > 0:
                score_random[i] += reward

    screen.fill((0, 0, 0))

    # Titles and scores
    title1 = title_font.render("Our hybrid strategy", True, (255, 255, 255))
    title2 = title_font.render("Random Policy", True, (255, 255, 255))
    screen.blit(title1, ((cols * TILE_SIZE) // 2 - title1.get_width() // 2, 5))
    screen.blit(title2, ((cols * TILE_SIZE) * 3 // 2 - title2.get_width() // 2, 5))

    score1 = score_font.render(f"Score: {score_test[0]} | {score_test[1]} | Step {iteration}", True, (200, 200, 200))
    score2 = score_font.render(f"Score: {score_random[0]} | {score_random[1]} | Step {iteration}", True, (200, 200, 200))
    screen.blit(score1, ((cols * TILE_SIZE) // 2 - score1.get_width() // 2, 25))
    screen.blit(score2, ((cols * TILE_SIZE) * 3 // 2 - score2.get_width() // 2, 25))

    # Render both environments
    display_grid_test = env_test.grid.astype(str).copy()
    for i, (x, y) in enumerate(env_test.agent_positions):
        display_grid_test[x, y] = '#' if i == 0 else '*'
    draw_grid(screen, display_grid_test, offset_x=0)

    display_grid_random = env_random.grid.astype(str).copy()
    for i, (x, y) in enumerate(env_random.agent_positions):
        display_grid_random[x, y] = '#' if i == 0 else '*'
    draw_grid(screen, display_grid_random, offset_x=cols * TILE_SIZE)

    # Update board stats
    a11, a12 = env_test.agent_positions[0]
    a21, a22 = env_test.agent_positions[1]
    board_agent1[a11, a12] += 1
    board_agent2[a21, a22] += 1

    a11r, a12r = env_random.agent_positions[0]
    a21r, a22r = env_random.agent_positions[1]
    board_agent1r[a11r, a12r] += 1
    board_agent2r[a21r, a22r] += 1

    # Divider lines
    border_x = cols * TILE_SIZE
    for offset in [0, border_x, 2 * border_x]:
        pygame.draw.line(screen, (0, 0, 0), (offset, 0), (offset, screen_height), 5)

    if disrupcion_activada:
        msg = font.render("DISRUPTION TRIGGERED!", True, (255, 0, 0))
        screen.blit(msg, ((screen_width - msg.get_width()) // 2, screen_height - 40))
        disrupcion_timer -= 1
        if disrupcion_timer <= 0:
            disrupcion_activada = False

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()

# Show heatmaps
for i, board in enumerate([board_agent1, board_agent2, board_agent1r, board_agent2r], 1):
    plt.figure(i)
    sn.heatmap(board / 1e3, annot=True)
    plt.title(f"Heatmap {i}")
    plt.show()
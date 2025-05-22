import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
import torch
import seaborn as sn
import matplotlib.pyplot as plt

from src.environment.apple_grid import AppleGridMDP
from src.agents.ppo_agent import PPOAgent
from src.utils.save_and_load import save_data

# Set seeds
np.random.seed(48)
random.seed(48)

# --- Load environment and agents --- #
env_test = AppleGridMDP(rate_regen=0.05)
env_random = AppleGridMDP(rate_regen=0.05)

grid_size = env_test.grid_size[0] * env_test.grid_size[1]
output_dim = 4
agent1 = PPOAgent(4 + grid_size, output_dim)
agent2 = PPOAgent(4 + grid_size, output_dim)

# Load trained models
agent1.model.load_state_dict(torch.load("models/baseline/ppo/agent1.pth"))
agent2.model.load_state_dict(torch.load("models/baseline/ppo/agent2.pth"))
print("Models loaded successfully.")

# --- Initialize tracking boards --- #
board_agent1_total = np.zeros((8, 8))
board_agent2_total = np.zeros((8, 8))
board_agent1_total_r = np.zeros((8, 8))
board_agent2_total_r = np.zeros((8, 8))

# --- Main loop --- #
for episode in range(500):
    board_agent1 = np.zeros((8, 8))
    board_agent2 = np.zeros((8, 8))
    board_agent1r = np.zeros((8, 8))
    board_agent2r = np.zeros((8, 8))

    env_test.reset()
    env_random.reset()

    # Random initial positions
    pos1 = (random.randint(0, 7), random.randint(0, 7))
    while True:
        pos2 = (random.randint(0, 7), random.randint(0, 7))
        if pos2 != pos1:
            break

    env_test.initial_agent_positions = [pos1, pos2]
    env_random.initial_agent_positions = [pos1, pos2]

    state_test = env_test.get_state()
    state_random = env_random.get_state()

    for iteration in range(1000):
        if iteration == 500:
            env_random.trigger_disruption()
            env_test.trigger_disruption()

        # PPO agents
        state_tensor = torch.tensor(state_test, dtype=torch.float32).unsqueeze(0)
        action1, _ = agent1.select_action(state_tensor)
        action2, _ = agent2.select_action(state_tensor)
        state_test, rewards_test = env_test.step([action1, action2])

        # Random agents
        actions_random = [random.randint(0, 3), random.randint(0, 3)]
        state_random, rewards_random = env_random.step(actions_random)

        if rewards_test[0] >= 0 and rewards_test[1] >= 0:
            x1, y1 = env_test.agent_positions[0]
            x2, y2 = env_test.agent_positions[1]
            board_agent1[x1, y1] += 1
            board_agent2[x2, y2] += 1

        if rewards_random[0] >= 0 or rewards_random[1] >= 0:
            x1r, y1r = env_random.agent_positions[0]
            x2r, y2r = env_random.agent_positions[1]
            board_agent1r[x1r, y1r] += 1
            board_agent2r[x2r, y2r] += 1

    board_agent1_total += board_agent1 / 1e3
    board_agent2_total += board_agent2 / 1e3
    board_agent1_total_r += board_agent1r / 1e3
    board_agent2_total_r += board_agent2r / 1e3

# --- Save results --- #
save_data(board_agent1_total, 'results/trajectories_maps/resilience_agent1.pkl')
save_data(board_agent2_total, 'results/trajectories_maps/resilience_agent2.pkl')
save_data(board_agent1_total_r, 'results/trajectories_maps/random_agent1.pkl')
save_data(board_agent2_total_r, 'results/trajectories_maps/random_agent2.pkl')

# --- Plot heatmaps --- #
plt.figure(1)
sn.heatmap(board_agent1_total / 500, annot=True)
plt.title("Agent 1 - PPO")
plt.show()

plt.figure(2)
sn.heatmap(board_agent2_total / 500, annot=True)
plt.title("Agent 2 - PPO")
plt.show()

plt.figure(3)
sn.heatmap(board_agent1_total_r / 500, annot=True)
plt.title("Agent 1 - Random")
plt.show()

plt.figure(4)
sn.heatmap(board_agent2_total_r / 500, annot=True)
plt.title("Agent 2 - Random")
plt.show()
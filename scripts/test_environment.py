import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.apple_grid import AppleGridMDP
import numpy as np
import random

# Test environment with 2 agents, single action
print("=== AppleGrid with 2 agents ===")
env = AppleGridMDP()
state, reward = env.step([0, 1])  # agent 0 moves up, agent 1 moves down
env.render()
print("Rewards:", reward)

# Test loop with random actions
env = AppleGridMDP()
env.reset()
env.render()
for _ in range(20):
    actions = [random.randint(0, 3), random.randint(0, 3)]  # Agents move randomly
    state, rewards = env.step(actions)
    print("Rewards:", rewards)
    env.render()
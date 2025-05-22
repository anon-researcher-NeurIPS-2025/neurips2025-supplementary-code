import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import torch
from src.environment.apple_grid import AppleGridMDP
from src.agents.ppo_agent import PPOAgent
from src.utils.save_and_load import load_data
# -----------------------------------------------
# Load learned reward weights from an IRL model
# IMPORTANT: You must first train an IRL linear model and obtain (w, b) using get_weights().
# You can either:
# - Import and call get_weights() from a model you just trained, OR
# - Load them from a saved file (recommended for reproducibility) 
#   path data/learning/linear chose the model you want to train 
# -----------------------------------------------

import os

weights_path = 'data/learning/resilience/linear/data_PPL.pkl'

if os.path.exists(weights_path):
    data = load_data(weights_path)
    try:
        learned_weights = data['weights']
        learned_bias = data['bias']
        print("[INFO] Loaded learned reward weights from saved file.")
    except:
        learned_weights = data['weights']
        learned_bias = 0
        print("[INFO] Loaded learned reward weights from saved file.")

else:
    # Fallback (for debugging/testing)
    learned_weights = np.random.randn(68)  # Dummy weights for example
    learned_bias = 0.0
    print("[WARNING] Using dummy reward weights. Train an IRL model and save the weights.")


# --- Environment and PPO Agent Setup ---
env = AppleGridMDP()
grid_size = env.grid_size[0] * env.grid_size[1]  # Should be 8x8 = 64
state_dim = 4 + grid_size
output_dim = 4  # Number of actions

agent1 = PPOAgent(state_dim, output_dim)
agent2 = PPOAgent(state_dim, output_dim)

num_episodes = 500
agent1_cum_reward = []
agent2_cum_reward = []

for episode in range(num_episodes):
    env.reset()
    state = np.array(env.get_state()).flatten()

    agent1_rewards = []
    agent2_rewards = []

    for _ in range(800):  # Max steps per episode
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action1, prob1 = agent1.select_action(state_tensor)
        action2, prob2 = agent2.select_action(state_tensor)
        actions = [action1, action2]

        next_state, rewards = env.step(actions)
        next_state = np.array(next_state).flatten()

        # --------------------------------------------------------------------
        # Option 1: Use inferred reward from learned resilience function only
        # --------------------------------------------------------------------
        # This approach evaluates the agent solely based on the reward 
        # learned via handcrafted IRL (e.g., MPL_K1, PPL_Mk, etc.),
        # completely replacing the environment reward.
        
        inferred_reward1 = np.dot(learned_weights, state) + learned_bias
        inferred_reward2 = np.dot(learned_weights, state) + learned_bias

        agent1.store_transition((state, action1, inferred_reward1, next_state, prob1))
        agent2.store_transition((state, action2, inferred_reward2, next_state, prob2))

        # -------------------------------------------------------------------
        # Option 2: Use hybrid reward (IRL reward + environment reward)
        # -------------------------------------------------------------------
        # This approach combines the IRL reward with the original reward 
        # provided by the environment. It supports training agents that
        # balance learned preferences with the environment's objectives.
        # Note: rewards[0] and rewards[1] are the env-provided rewards
        # for agent 1 and agent 2 respectively related with apple consumption.
        # hybrid_reward1 = inferred_reward1 + rewards[0]
        # hybrid_reward2 = inferred_reward2 + rewards[1]
        # agent1.store_transition((state, action1, hybrid_reward1, next_state, prob1))
        # agent2.store_transition((state, action2, hybrid_reward2, next_state, prob2))

        state = next_state
        agent1_rewards.append(rewards[0])
        agent2_rewards.append(rewards[1])

        if rewards[0] < 0:
            agent1_rewards.pop()
            break
        if rewards[1] < 0:
            agent2_rewards.pop()
            break

    agent1_cum_reward.append(np.sum(agent1_rewards))
    agent2_cum_reward.append(np.sum(agent2_rewards))

    agent1.train()
    agent2.train()

    print(f"[Episode {episode}] Agent 1 Return: {agent1_cum_reward[-1]} | Agent 2 Return: {agent2_cum_reward[-1]}")


# -----------------------------------------------
# Save trained PPO agent models
# -----------------------------------------------
# Trained agents from previous runs are organized in the 'models/' directory:
# - baseline/: standard PPO agents without reward shaping
# - best/: best-performing agents under our IRL-based reward models
# - hybrid/: agents trained under different reward combinations (hybrid settings)
# - resilience/: agents trained per configuration for resilience evaluation
# - example/: dummy or quick-test agents (not used in paper results)
#
# Agents used to produce the results reported in the paper are stored
# in the appropriate folders for reproducibility.
torch.save(agent1.model.state_dict(), "models/example/agent_1_.pth")
torch.save(agent2.model.state_dict(), "models/example/agent_2_.pth")
print("Agents successfully saved")
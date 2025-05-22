import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.utils.save_and_load import load_data

# Note: You must load the IRL model you want to train. It can be from a neural net model
# Available training approaches include Margin Preference Learning (MPL) or Probabilistic Preference Learning (PPL).
# Examples: 
# from src.irl.nn.margin_preference_learning import MPL_R1 
# from src.irl.nn.margin_preference_learning import MPL_K1

from src.irl.nn.probabilistic_preference_learning import PPL_K

# Load ranked trajectories (generated from random agents)
random_trajectories = load_data('data/random_trajectories.pkl')

# Define state dimensionality: 4 positions + 8x8 grid = 4 + 64
state_dim = 4 + 64

# Initialize the IRL model
irl_model = PPL_K(state_dim)

# Train the model
irl_model.train(random_trajectories, num_epochs=200)

# To save 
irl_model.save_model('data/learning/nn/dummy_irl_model_nn.pth')
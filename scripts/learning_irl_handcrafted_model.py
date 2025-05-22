import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.utils.save_and_load import load_data
from src.irl.handcrafted.hand_crafted_features import phi # to return handcrafted features
from src.utils.save_and_load import load_data, save_data

import inspect
# Note: You must load the IRL model you want to train. It can be from a handcrafted feature model
# Available training approaches include Margin Preference Learning (MPL) or Probabilistic Preference Learning (PPL).
# Examples: 
# from src.irl.handcrafted.margin_preference_learning import MPL_R1 
# from src.irl.handcrafted.margin_preference_learning import MPL_K1
from src.irl.handcrafted.probabilistic_preference_learning import PPL_M


# Load ranked trajectories (generated from random agents)
random_trajectories = load_data('data/random_trajectories.pkl')

# Define handcrafted feature dimensionality
state_dim = 6  # phi returns 6 handcrafted features

# Initialize the IRL model
irl_model = PPL_M(state_dim)

train_sig = inspect.signature(irl_model.train)

if "random_train" in train_sig.parameters:
    print('Training in random sampling')
    # Precompute handcrafted features for each state in the trajectories
    trajectories_caracteristics = {}
    for i, trajectory in enumerate(random_trajectories):
        for j, state in enumerate(trajectory[0]):
            trajectories_caracteristics[(i, j)] = phi(state)
    
    # Train the model
    irl_model.train(random_trajectories, trajectories_caracteristics, num_epochs=200)
    
else:
    print('Training in not random sampling')
    # Sort the trajectories by resilience score in descending order
    ranked_trajectories = sorted(random_trajectories, key=lambda x: x[1], reverse=True)

    # Precompute handcrafted features for each state in the trajectories
    trajectories_caracteristics = {}
    for i, trajectory in enumerate(ranked_trajectories):
        for j, state in enumerate(trajectory[0]):
            trajectories_caracteristics[(i, j)] = phi(state)


    # Train the model
    irl_model.train(ranked_trajectories, trajectories_caracteristics, num_epochs=200)

# Retrieve and display learned weights and bias
weights, bias = irl_model.get_weights()
print("Learned weights:", weights)
print("Learned bias:", bias)

# To save weights and bias
data_learning = {
    'weights': weights,
    'bias': bias
}
save_data(data_learning, filename="data/learning/handcrafted/data_linear_learning_.pkl")
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

class PPL_R:
    def __init__(self, state_dim, lr=0.001):
        """
        Initializes the linear reward model: R(s) = w^T * phi(s) + b
        """
        self.w = nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes the estimated reward R(s) = w^T * phi(s) + b for a given state.
        """
        phi = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, phi) + self.b

    def train(self, ranked_trajectories, num_epochs=100):
        """
        Trains weights 
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                if res_i < res_j:  # Ensure traj2 is better than traj1
                    R_traj1 = sum(self.reward(state) for state in traj1)
                    R_traj2 = sum(self.reward(state) for state in traj2)

                    # Bradley-Terry loss with numerical stability
                    max_r = torch.max(R_traj1, R_traj2)
                    loss = - (R_traj2 - (max_r + torch.log(torch.exp(R_traj1 - max_r) + torch.exp(R_traj2 - max_r))))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()


class PPL_K:
    def __init__(self, state_dim, lr=0.001):
        """
        Initializes a linear reward model: R(s) = w^T * phi(s) + b
        """
        self.w = nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)

        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes the reward R(s) = w^T * phi(s) + b
        """
        phi = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, phi) + self.b

    def train(self, trajectories, num_epochs=100):
        """
        Trains the reward model using loss over sorted (ascending) trajectories.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0
            ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=False)

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                if res_i < res_j:
                    R_traj1 = sum(self.reward(state) for state in traj1)
                    R_traj2 = sum(self.reward(state) for state in traj2)

                    max_r = torch.max(R_traj1, R_traj2)
                    loss = - (R_traj2 - (max_r + torch.log(torch.exp(R_traj1 - max_r) + torch.exp(R_traj2 - max_r))))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()
    
class PPL_M:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes the reward model with weights and bias: R(s) = w^T * phi(s) + b
        """
        self.w = nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)

        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes the reward R(s) for a given state.
        """
        phi = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, phi) + self.b

    def train(self, trajectories, num_epochs=100):
        """
        Trains the reward model using a probabilistic preference loss over ranked trajectories.
        This variant mixes comparisons between nearby and distant pairs.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0
            ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=False)

            for _ in range(len(ranked_trajectories) - 1):
                if random.random() < 0.5:
                    i = random.randint(0, len(ranked_trajectories) - 2)
                    traj1, res1, _, _ = ranked_trajectories[i]
                    traj2, res2, _, _ = ranked_trajectories[i + 1]
                else:
                    i, j = random.sample(range(len(ranked_trajectories)), 2)
                    if ranked_trajectories[i][1] > ranked_trajectories[j][1]:
                        traj1, res1, _, _ = ranked_trajectories[i]
                        traj2, res2, _, _ = ranked_trajectories[j]
                    else:
                        traj1, res1, _, _ = ranked_trajectories[j]
                        traj2, res2, _, _ = ranked_trajectories[i]

                R_traj1 = sum(self.reward(state) for state in traj1)
                R_traj2 = sum(self.reward(state) for state in traj2)

                max_r = torch.max(R_traj1, R_traj2)
                loss = - (R_traj2 - (max_r + torch.log(torch.exp(R_traj1 - max_r) + torch.exp(R_traj2 - max_r))))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()
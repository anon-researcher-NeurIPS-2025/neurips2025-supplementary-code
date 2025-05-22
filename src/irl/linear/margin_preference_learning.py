import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

class MPL_R1:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a linear reward model: R(s) = w^T * phi(s) + b
        """
        self.w = torch.nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes reward R(s) for a given state vector.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, state_tensor) + self.b

    def train(self, ranked_trajectories, num_epochs=100, margin=1.0):
        """
        Trains weights using ranked trajectories and a margin-based loss.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(s, dtype=torch.float32) for s in traj1)
                phi_2 = sum(torch.tensor(s, dtype=torch.float32) for s in traj2)

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                if res_i > res_j:
                    loss = torch.clamp(margin + Rj - Ri, min=0)
                    total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()


class MPL_Rk:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a linear reward model: R(s) = w^T * phi(s) + b
        """
        self.w = torch.nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes reward R(s) for a given state vector.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, state_tensor) + self.b

    def train(self, ranked_trajectories, num_epochs=100):
        """
        Trains weights using ranked trajectories and a margin-based loss
        where the margin depends on the difference in resilience scores.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(s, dtype=torch.float32) for s in traj1)
                phi_2 = sum(torch.tensor(s, dtype=torch.float32) for s in traj2)

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                margin = torch.abs(torch.tensor(res_i - res_j, dtype=torch.float32))

                if res_i > res_j:
                    loss = torch.clamp(margin + Rj - Ri, min=0)
                    total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()

class MPL_K1:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a linear reward model with bias: R(s) = w^T * phi(s) + b
        """
        self.w = nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes reward R(s) for a given state vector.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, state_tensor) + self.b

    def train(self, trajectories, num_epochs=100):
        """
        Trains weights using ranked trajectories with a constant margin of 1.
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(s, dtype=torch.float32) for s in traj1)
                phi_2 = sum(torch.tensor(s, dtype=torch.float32) for s in traj2)

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                margin = 1.0

                if res_i > res_j:
                    loss = torch.clamp(margin + Rj - Ri, min=0)
                    total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()
    
class MPL_Kk:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a linear reward model with bias: R(s) = w^T * phi(s) + b
        """
        self.w = nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes reward R(s) for a given state vector.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, state_tensor) + self.b

    def train(self, trajectories, num_epochs=100):
        """
        Trains the model using ranked trajectories. Margin is proportional to the absolute
        difference in resilience scores between consecutive pairs.
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(s, dtype=torch.float32) for s in traj1)
                phi_2 = sum(torch.tensor(s, dtype=torch.float32) for s in traj2)

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                margin = np.abs(res_i - res_j)

                if res_i > res_j:
                    loss = torch.clamp(margin + Rj - Ri, min=0)
                    total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()
    
class MPL_M1:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a linear reward model with bias: R(s) = w^T * phi(s) + b
        """
        self.w = nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes the estimated reward R(s) = w^T * phi(s) + b.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, state_tensor) + self.b

    def train(self, trajectories, num_epochs=100):
        """
        Trains the weights using ranked trajectories.
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for _ in range(len(ranked_trajectories) - 1):
                if random.random() < 0.5:
                    i = random.randint(0, len(ranked_trajectories) - 2)
                    traj1, res_i, _, _ = ranked_trajectories[i]
                    traj2, res_j, _, _ = ranked_trajectories[i + 1]
                else:
                    i, j = random.sample(range(len(ranked_trajectories)), 2)
                    traj1, res_i, _, _ = ranked_trajectories[i]
                    traj2, res_j, _, _ = ranked_trajectories[j]
                    if res_i < res_j:
                        traj1, res_i, traj2, res_j = traj2, res_j, traj1, res_i

                phi_1 = sum(torch.tensor(state, dtype=torch.float32) for state in traj1)
                phi_2 = sum(torch.tensor(state, dtype=torch.float32) for state in traj2)

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                margin = 1.0
                if res_i > res_j:
                    loss = torch.clamp(margin + Rj - Ri, min=0)
                    total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()
    
class MPL_Mk:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes the model with weights w and bias b to estimate R(s) = w^T * phi(s) + b.
        """
        self.w = nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes the estimated reward R(s) = w^T * phi(s) + b.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, state_tensor) + self.b

    def train(self, trajectories, num_epochs=100):
        """
        Trains the model using ranked trajectories. Margin is proportional to 
        the absolute difference in resilience scores.
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for _ in range(len(ranked_trajectories) - 1):
                if random.random() < 0.5:
                    i = random.randint(0, len(ranked_trajectories) - 2)
                    traj1, res_i, _, _ = ranked_trajectories[i]
                    traj2, res_j, _, _ = ranked_trajectories[i + 1]
                else:
                    i, j = random.sample(range(len(ranked_trajectories)), 2)
                    traj1, res_i, _, _ = ranked_trajectories[i]
                    traj2, res_j, _, _ = ranked_trajectories[j]
                    if res_i < res_j:
                        traj1, res_i, traj2, res_j = traj2, res_j, traj1, res_i

                phi_1 = sum(torch.tensor(state, dtype=torch.float32) for state in traj1)
                phi_2 = sum(torch.tensor(state, dtype=torch.float32) for state in traj2)

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                margin = np.abs(res_i - res_j)
                if res_i > res_j:
                    loss = torch.clamp(margin + Rj - Ri, min=0)
                    total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()
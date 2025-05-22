import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random 

class PPL_R:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes the reward model: R(s) = w^T * phi(s) + b
        """
        self.w = nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)

        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def phi(self, state_vector):
        """
        Extract handcrafted features from the state vector.
        Format: [x1, x2, y1, y2, m0, m1, ..., m63] of length 67
        """
        x1, x2, y1, y2 = map(int, state_vector[:4])
        apple_grid = np.array(state_vector[4:]).reshape(8, 8)
        apples = np.argwhere(apple_grid == 1)

        pos_a1 = np.array([x1, x2])
        pos_a2 = np.array([y1, y2])

        phi_1 = len(apples)
        dists_a1 = [np.linalg.norm(pos_a1 - p) for p in apples] if apples.size > 0 else [0]
        dists_a2 = [np.linalg.norm(pos_a2 - p) for p in apples] if apples.size > 0 else [0]

        phi_2 = min(dists_a1)
        phi_3 = min(dists_a2)
        phi_4 = abs(phi_2 - phi_3)

        def count_apples_near(pos):
            x, y = pos
            return sum(apple_grid[nx, ny]
                       for dx in [-1, 0, 1]
                       for dy in [-1, 0, 1]
                       if 0 <= (nx := x + dx) < 8 and 0 <= (ny := y + dy) < 8)

        phi_5 = count_apples_near(pos_a1)
        phi_6 = count_apples_near(pos_a2)

        return torch.tensor([phi_1, phi_2, phi_3, phi_4, phi_5, phi_6], dtype=torch.float32)

    def reward(self, state_vector):
        features = self.phi(state_vector)
        return torch.dot(self.w, features) + self.b

    def train(self, ranked_trajectories, trajectory_features, num_epochs=100, random_train = True):
        """
        Trains the reward model using a probabilistic preference-based loss (Bradley-Terry model).
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(trajectory_features[(i, j)], dtype=torch.float32) for j in range(len(traj1)))
                phi_2 = sum(torch.tensor(trajectory_features[(i + 1, j)], dtype=torch.float32) for j in range(len(traj2)))

                if res_i < res_j:
                    R1 = torch.dot(self.w, phi_1) + self.b
                    R2 = torch.dot(self.w, phi_2) + self.b

                    # Bradley-Terry loss (numerically stable)
                    max_r = torch.max(R1, R2)
                    loss = - (R2 - (max_r + torch.log(torch.exp(R1 - max_r) + torch.exp(R2 - max_r))))

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
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a linear reward model with bias: R(s) = w^T * phi(s) + b
        """
        with torch.no_grad():
            self.w = torch.empty(state_dim, requires_grad=True, dtype=torch.float32).uniform_(-0.5, 0.5)
            self.b = torch.zeros(1, requires_grad=True, dtype=torch.float32)

        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes reward R(s) = w^T * phi(s)
        """
        phi = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, phi) + self.b

    def train(self, trajectories, trajectories_caracteristics, num_epochs=100):
        """
        Trains the reward weights using ranked trajectories and a probabilistic loss.
        """
        for epoch in range(num_epochs):
            total_loss = 0

            for i in range(len(trajectories) - 1):
                traj1, res_i, _, _ = trajectories[i]
                traj2, res_j, _, _ = trajectories[i + 1]

                if res_i < res_j:
                    phi_1 = sum(torch.tensor(trajectories_caracteristics[(i, j)], dtype=torch.float32) for j, _ in enumerate(traj1))
                    phi_2 = sum(torch.tensor(trajectories_caracteristics[(i + 1, j)], dtype=torch.float32) for j, _ in enumerate(traj2))

                    R_traj1 = torch.dot(self.w, phi_1) + self.b
                    R_traj2 = torch.dot(self.w, phi_2) + self.b

                    # Bradley-Terry loss (with numerical stabilization)
                    max_r = torch.max(R_traj1, R_traj2)
                    loss = - (R_traj2 - (max_r + torch.log(torch.exp(R_traj1 - max_r) + torch.exp(R_traj2 - max_r))))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

            if epoch % 2 == 0:
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
        Initializes a linear reward model with bias: R(s) = w^T * phi(s) + b
        """
        with torch.no_grad():
            self.w = torch.empty(state_dim, requires_grad=True, dtype=torch.float32).uniform_(-0.5, 0.5)
            self.b = torch.zeros(1, requires_grad=True, dtype=torch.float32)

        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def reward(self, state):
        """
        Computes reward R(s) = w^T * phi(s)
        """
        phi = torch.tensor(state, dtype=torch.float32)
        return torch.dot(self.w, phi) + self.b

    def train(self, trajectories, trajectories_caracteristics, num_epochs=100):
        """
        Trains the model using mixed sampling strategy and probabilistic loss.
        """
        for epoch in range(num_epochs):
            total_loss = 0

            for _ in range(len(trajectories) - 1):
                if random.random() < 0.5:
                    i = random.randint(0, len(trajectories) - 2)
                    traj1, res1, _, _ = trajectories[i]
                    traj2, res2, _, _ = trajectories[i + 1]
                    phi_1 = sum(torch.tensor(trajectories_caracteristics[(i,jj)], dtype=torch.float32) for jj, state in enumerate(traj1))
                    phi_2 = sum(torch.tensor(trajectories_caracteristics[(i+1,jj)], dtype=torch.float32) for jj, state in enumerate(traj2))

                else:
                    i, j = random.sample(range(len(trajectories)), 2)
                    if trajectories[i][1] > trajectories[j][1]:
                        traj1, res1, _, _ = trajectories[i]
                        traj2, res2, _, _ = trajectories[j]
                        phi_1 = sum(torch.tensor(trajectories_caracteristics[(i,jj)], dtype=torch.float32) for jj, state in enumerate(traj1))
                        phi_2 = sum(torch.tensor(trajectories_caracteristics[(j,jj)], dtype=torch.float32) for jj, state in enumerate(traj2))
                    else:
                        traj1, res1, _, _ = trajectories[j]
                        traj2, res2, _, _ = trajectories[i]
                        phi_1 = sum(torch.tensor(trajectories_caracteristics[(j,jj)], dtype=torch.float32) for jj, state in enumerate(traj1))
                        phi_2 = sum(torch.tensor(trajectories_caracteristics[(i,jj)], dtype=torch.float32) for jj, state in enumerate(traj2))

                R_traj1 = torch.dot(self.w, phi_1) + self.b
                R_traj2 = torch.dot(self.w, phi_2) + self.b

                max_r = torch.max(R_traj1, R_traj2)
                loss = - (R_traj2 - (max_r + torch.log(torch.exp(R_traj1 - max_r) + torch.exp(R_traj2 - max_r))))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        print("Training complete.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()
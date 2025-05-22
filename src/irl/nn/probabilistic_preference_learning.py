import torch
import torch.nn as nn
import torch.optim as optim
import random

class PPL_R(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, lr=0.001):
        """
        Initializes a neural network to estimate R(s) = NN(phi(s)).
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def reward(self, state):
        """
        Computes the reward R(s) using the neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.model(state_tensor).squeeze()

    def train(self, ranked_trajectories, num_epochs=100):
        """
        Trains the model using ranked trajectories and T-REX loss (softmax-based pairwise comparison).
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                # Ensure traj2 is the better trajectory
                if res_i < res_j:
                    R_traj1 = sum(self.reward(s) for s in traj1)
                    R_traj2 = sum(self.reward(s) for s in traj2)

                    # T-REX loss using numerically stable softmax difference
                    max_r = torch.max(R_traj1, R_traj2)
                    loss = - (R_traj2 - (max_r + torch.log(
                        torch.exp(R_traj1 - max_r) + torch.exp(R_traj2 - max_r)
                    )))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        print("Training completed.")

    def get_model(self):
        """
        Returns the neural network model.
        """
        return self.model

    def save_model(self, path='trex_reward_model.pth'):
        """
        Saves the model weights to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='trex_reward_model.pth'):
        """
        Loads the model weights from a file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class PPL_K(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, lr=0.001):
        """
        Initializes a neural network to estimate R(s) = NN(phi(s)).
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def reward(self, state):
        """
        Computes the reward R(s) using the neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.model(state_tensor).squeeze()

    def train(self, trajectories, num_epochs=100):
        """
        Trains the model using T-REX loss with ranked trajectories.
        Trajectories are sorted in ascending order of quality (lower to higher).
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=False)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                R_traj1 = sum(self.reward(s) for s in traj1)
                R_traj2 = sum(self.reward(s) for s in traj2)

                # Ensure traj2 is preferred over traj1
                if res_i < res_j:
                    max_r = torch.max(R_traj1, R_traj2)
                    loss = - (R_traj2 - (max_r + torch.log(
                        torch.exp(R_traj1 - max_r) + torch.exp(R_traj2 - max_r)
                    )))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        print("Training completed.")

    def get_model(self):
        """
        Returns the trained model.
        """
        return self.model

    def save_model(self, path='trex_reward_model.pth'):
        """
        Saves the model parameters to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='trex_reward_model.pth'):
        """
        Loads model parameters from a file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class PPL_M:
    def __init__(self, state_dim, hidden_dim=32, lr=0.001):
        """
        Initializes a neural network to estimate the reward function: R(s) = NN(phi(s)).
        """
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def reward(self, state):
        """
        Computes the scalar reward R(s) using the trained neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.model(state_tensor).item()

    def train(self, trajectories, num_epochs=100):
        """
        Trains the neural network using pairwise comparisons from ranked trajectories.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0
            ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=False)

            for _ in range(len(ranked_trajectories) - 1):
                # Randomly sample pairs of trajectories (nearby or distant)
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

                # Compute total reward for each trajectory
                R_traj1 = sum(self.model(torch.tensor(s, dtype=torch.float32).unsqueeze(0)) for s in traj1)
                R_traj2 = sum(self.model(torch.tensor(s, dtype=torch.float32).unsqueeze(0)) for s in traj2)

                # Apply Bradley-Terry loss (numerically stable version)
                max_r = torch.max(R_traj1, R_traj2)
                loss = - (R_traj2 - (max_r + torch.log(torch.exp(R_traj1 - max_r) + torch.exp(R_traj2 - max_r))))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

        print("Training completed.")

    def save_model(self, path='reward_model_trex_nn.pth'):
        """
        Saves the model to disk.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='reward_model_trex_nn.pth'):
        """
        Loads the model from disk.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
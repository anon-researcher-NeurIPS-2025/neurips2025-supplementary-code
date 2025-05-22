import torch
import torch.nn as nn
import torch.optim as optim
import random 

class MPL_R1:
    def __init__(self, state_dim, hidden_dim=32):
        """
        Initializes a neural network to estimate R(s).
        """
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MarginRankingLoss(margin=1.0)

    def reward(self, state):
        """
        Computes the reward R(s) using the neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # batch size = 1
        return self.model(state_tensor).squeeze()

    def train(self, ranked_trajectories, num_epochs=100):
        """
        Trains the neural network using ranked trajectories.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(state, dtype=torch.float32) for state in traj1)
                phi_2 = sum(torch.tensor(state, dtype=torch.float32) for state in traj2)

                R1 = self.model(phi_1.unsqueeze(0))  # [1, 1]
                R2 = self.model(phi_2.unsqueeze(0))  # [1, 1]

                target = torch.tensor([1.0]) if res_i > res_j else torch.tensor([-1.0])
                target = target.view(-1, 1)  # Ensure shape [1, 1]

                loss = self.loss_fn(R1, R2, target)
                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Neural network training completed.")

    def get_weights(self):
        """
        Returns the neural network parameters.
        """
        return self.model.state_dict()

    def save_model(self, path='reward_model.pth'):
        """
        Saves the model to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='reward_model.pth'):
        """
        Loads model weights from a file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


class MPL_Rk:
    def __init__(self, state_dim, hidden_dim=32):
        """
        Initializes a neural network to estimate R(s).
        """
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def reward(self, state):
        """
        Computes the reward R(s) using the neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.model(state_tensor).squeeze()

    def train(self, ranked_trajectories, num_epochs=100):
        """
        Trains the neural network using ranked trajectories with a dynamic margin.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(state, dtype=torch.float32) for state in traj1)
                phi_2 = sum(torch.tensor(state, dtype=torch.float32) for state in traj2)

                R1 = self.model(phi_1.unsqueeze(0))  # Shape: [1, 1]
                R2 = self.model(phi_2.unsqueeze(0))  # Shape: [1, 1]

                margin = abs(res_i - res_j)
                if res_i > res_j:
                    loss = torch.clamp(margin + R2 - R1, min=0)
                else:
                    loss = torch.clamp(margin + R1 - R2, min=0)

                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Neural network training completed.")

    def get_weights(self):
        """
        Returns the neural network parameters.
        """
        return self.model.state_dict()

    def save_model(self, path='reward_model.pth'):
        """
        Saves the model weights to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='reward_model.pth'):
        """
        Loads model weights from a file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class MPL_K1:
    def __init__(self, state_dim, hidden_dim=32):
        """
        Initializes a neural network to estimate R(s).
        """
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MarginRankingLoss(margin=1.0)

    def reward(self, state):
        """
        Computes the reward R(s) using the neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: [1, state_dim]
        return self.model(state_tensor).squeeze()

    def train(self, trajectories, num_epochs=100):
        """
        Trains the neural network using ranked trajectories.
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(state, dtype=torch.float32) for state in traj1)
                phi_2 = sum(torch.tensor(state, dtype=torch.float32) for state in traj2)

                R1 = self.model(phi_1.unsqueeze(0))  # shape: [1, 1]
                R2 = self.model(phi_2.unsqueeze(0))  # shape: [1, 1]

                target = torch.tensor([1.0]) if res_i > res_j else torch.tensor([-1.0])
                target = target.view(-1, 1)  # shape: [1, 1]

                loss = self.loss_fn(R1, R2, target)
                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Neural network training completed.")

    def get_weights(self):
        """
        Returns the parameters of the neural network.
        """
        return self.model.state_dict()

    def save_model(self, path='reward_model.pth'):
        """
        Saves the model parameters to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='reward_model.pth'):
        """
        Loads the model parameters from a file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class MPL_Kk:
    def __init__(self, state_dim, hidden_dim=32):
        """
        Initializes a neural network to estimate R(s).
        """
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def reward(self, state):
        """
        Computes the reward R(s) using the neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: [1, state_dim]
        return self.model(state_tensor).squeeze()

    def train(self, trajectories, num_epochs=100):
        """
        Trains the neural network using ranked trajectories with a dynamic margin.
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(state, dtype=torch.float32) for state in traj1)
                phi_2 = sum(torch.tensor(state, dtype=torch.float32) for state in traj2)

                R1 = self.model(phi_1.unsqueeze(0))  # shape: [1, 1]
                R2 = self.model(phi_2.unsqueeze(0))  # shape: [1, 1]

                margin = abs(res_i - res_j)

                if res_i > res_j:
                    loss = torch.clamp(margin + R2 - R1, min=0)
                else:
                    loss = torch.clamp(margin + R1 - R2, min=0)

                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Neural network training completed.")

    def get_weights(self):
        """
        Returns the neural network parameters.
        """
        return self.model.state_dict()

    def save_model(self, path='reward_model.pth'):
        """
        Saves the model parameters to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='reward_model.pth'):
        """
        Loads the model parameters from a file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


class MPL_M1:
    def __init__(self, state_dim, hidden_dim=32):
        """
        Initializes a neural network to estimate R(s).
        """
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MarginRankingLoss(margin=1.0)

    def reward(self, state):
        """
        Computes the reward R(s) using the neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: [1, state_dim]
        return self.model(state_tensor).squeeze()

    def train(self, trajectories, num_epochs=100):
        """
        Trains the neural network using ranked trajectories.
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for _ in range(len(ranked_trajectories) - 1):
                if random.random() < 0.5:  # 50% chance to compare neighboring trajectories
                    i = random.randint(0, len(ranked_trajectories) - 2)
                    traj1, res_i, _, _ = ranked_trajectories[i]
                    traj2, res_j, _, _ = ranked_trajectories[i + 1]
                else:  # 50% chance to compare distant trajectories
                    i, j = random.sample(range(len(ranked_trajectories)), 2)
                    if ranked_trajectories[i][1] > ranked_trajectories[j][1]:
                        traj1, res_i, _, _ = ranked_trajectories[i]
                        traj2, res_j, _, _ = ranked_trajectories[j]
                    else:
                        traj1, res_i, _, _ = ranked_trajectories[j]
                        traj2, res_j, _, _ = ranked_trajectories[i]

                phi_1 = sum(torch.tensor(state, dtype=torch.float32) for state in traj1)
                phi_2 = sum(torch.tensor(state, dtype=torch.float32) for state in traj2)

                R1 = self.model(phi_1.unsqueeze(0))  # shape: [1, 1]
                R2 = self.model(phi_2.unsqueeze(0))  # shape: [1, 1]

                target = torch.tensor([1.0]) if res_i > res_j else torch.tensor([-1.0])
                target = target.view(-1, 1)  # ensure shape [1, 1]

                loss = self.loss_fn(R1, R2, target)
                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

    def get_weights(self):
        """
        Returns the parameters of the neural network.
        """
        return self.model.state_dict()

    def save_model(self, path='reward_model.pth'):
        """
        Saves the model parameters to a file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='reward_model.pth'):
        """
        Loads the model parameters from a file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class MPL_Mk:
    def __init__(self, state_dim, hidden_dim=32):
        """
        Initializes a neural network to estimate R(s).
        """
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def reward(self, state):
        """
        Computes the reward R(s) using the neural network.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: [1, state_dim]
        return self.model(state_tensor).squeeze()

    def train(self, trajectories, num_epochs=100):
        """
        Trains the neural network using ranked trajectories and dynamic margins.
        """
        ranked_trajectories = sorted(trajectories, key=lambda x: x[1], reverse=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for _ in range(len(ranked_trajectories) - 1):
                # Mixed comparison: near or far trajectory pairs
                if random.random() < 0.5:
                    i = random.randint(0, len(ranked_trajectories) - 2)
                    traj1, res_i, _, _ = ranked_trajectories[i]
                    traj2, res_j, _, _ = ranked_trajectories[i + 1]
                else:
                    i, j = random.sample(range(len(ranked_trajectories)), 2)
                    if ranked_trajectories[i][1] > ranked_trajectories[j][1]:
                        traj1, res_i, _, _ = ranked_trajectories[i]
                        traj2, res_j, _, _ = ranked_trajectories[j]
                    else:
                        traj1, res_i, _, _ = ranked_trajectories[j]
                        traj2, res_j, _, _ = ranked_trajectories[i]

                # Aggregate state features
                phi_1 = sum(torch.tensor(state, dtype=torch.float32) for state in traj1)
                phi_2 = sum(torch.tensor(state, dtype=torch.float32) for state in traj2)

                # Compute predicted rewards
                R1 = self.model(phi_1.unsqueeze(0))  # shape: [1, 1]
                R2 = self.model(phi_2.unsqueeze(0))  # shape: [1, 1]

                # Dynamic margin based on resilience difference
                margin = abs(res_i - res_j)
                if res_i > res_j:
                    loss = torch.clamp(margin + R2 - R1, min=0)
                else:
                    loss = torch.clamp(margin + R1 - R2, min=0)

                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Neural network training complete.")

    def get_weights(self):
        """
        Returns the learned network parameters.
        """
        return self.model.state_dict()

    def save_model(self, path='reward_model.pth'):
        """
        Saves the model parameters to the specified file.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='reward_model.pth'):
        """
        Loads the model parameters from the specified file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
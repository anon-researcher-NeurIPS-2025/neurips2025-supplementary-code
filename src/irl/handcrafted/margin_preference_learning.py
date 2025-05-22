import torch
import torch.optim as optim
import numpy as np
import random 

class MPL_R1:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a preference-based IRL model with handcrafted features.
        The reward is modeled as: R(s) = w^T * phi(s) + b
        """
        self.w = torch.nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)

        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def phi(self, state_vector):
        """
        Extracts handcrafted features from a given state vector.
        State format: [x1, x2, y1, y2, m0, ..., m63] where m0–m63 are the flattened 8x8 apple grid.
        """
        x1, x2, y1, y2 = map(int, state_vector[:4])
        apple_grid = np.array(state_vector[4:]).reshape(8, 8)
        apples = np.argwhere(apple_grid == 1)

        pos_a1 = np.array([x1, x2])
        pos_a2 = np.array([y1, y2])

        f1 = len(apples)

        dists_a1 = [np.linalg.norm(pos_a1 - np.array(p)) for p in apples] if apples.size > 0 else [0]
        dists_a2 = [np.linalg.norm(pos_a2 - np.array(p)) for p in apples] if apples.size > 0 else [0]

        f2 = min(dists_a1)
        f3 = min(dists_a2)
        f4 = abs(f2 - f3)

        def count_apples_near(pos):
            x, y = pos
            return sum(
                apple_grid[nx, ny]
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if 0 <= (nx := x + dx) < 8 and 0 <= (ny := y + dy) < 8
            )

        f5 = count_apples_near(pos_a1)
        f6 = count_apples_near(pos_a2)

        return torch.tensor([f1, f2, f3, f4, f5, f6], dtype=torch.float32)

    def reward(self, state_vector):
        """
        Computes the estimated reward R(s) = w^T * phi(s) + b
        """
        features = self.phi(state_vector)
        return torch.dot(self.w, features) + self.b

    def train(self, ranked_trajectories, trajectories_features, num_epochs=100, random_train = True):
        """
        Trains the reward model using ranked trajectories and precomputed features.
        
        Args:
            ranked_trajectories: list of tuples (trajectory, resilience_score, r1, r2)
            trajectories_features: dict[(i,j)] with handcrafted features for state j of trajectory i
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(trajectories_features[(i, j)], dtype=torch.float32) for j in range(len(traj1)))
                phi_2 = sum(torch.tensor(trajectories_features[(i + 1, j)], dtype=torch.float32) for j in range(len(traj2)))

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                margin = 1

                if res_i > res_j:
                    loss = torch.clamp(margin + Rj - Ri, min=0)
                    total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

        print("Finished optimizing w and b.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()


class MPL_Rk:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a preference-based IRL model with handcrafted features.
        The reward is modeled as: R(s) = w^T * phi(s) + b
        """
        self.w = torch.nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)

        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def phi(self, state_vector):
        """
        Extracts handcrafted features from a given state vector.
        State format: [x1, x2, y1, y2, m0, ..., m63] where m0–m63 are the flattened 8x8 apple grid.
        """
        x1, x2, y1, y2 = map(int, state_vector[:4])
        apple_grid = np.array(state_vector[4:]).reshape(8, 8)
        apples = np.argwhere(apple_grid == 1)

        pos_a1 = np.array([x1, x2])
        pos_a2 = np.array([y1, y2])

        f1 = len(apples)

        dists_a1 = [np.linalg.norm(pos_a1 - np.array(p)) for p in apples] if apples.size > 0 else [0]
        dists_a2 = [np.linalg.norm(pos_a2 - np.array(p)) for p in apples] if apples.size > 0 else [0]

        f2 = min(dists_a1)
        f3 = min(dists_a2)
        f4 = abs(f2 - f3)

        def count_apples_near(pos):
            x, y = pos
            return sum(
                apple_grid[nx, ny]
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if 0 <= (nx := x + dx) < 8 and 0 <= (ny := y + dy) < 8
            )

        f5 = count_apples_near(pos_a1)
        f6 = count_apples_near(pos_a2)

        return torch.tensor([f1, f2, f3, f4, f5, f6], dtype=torch.float32)

    def reward(self, state_vector):
        """
        Computes the estimated reward R(s) = w^T * phi(s) + b
        """
        features = self.phi(state_vector)
        return torch.dot(self.w, features) + self.b

    def train(self, ranked_trajectories, trajectories_features, num_epochs=100, random_train = True):
        """
        Trains the reward model using ranked trajectories and precomputed features.
        
        Args:
            ranked_trajectories: list of tuples (trajectory, resilience_score, r1, r2)
            trajectories_features: dict[(i,j)] with handcrafted features for state j of trajectory i
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(trajectories_features[(i, j)], dtype=torch.float32) for j in range(len(traj1)))
                phi_2 = sum(torch.tensor(trajectories_features[(i + 1, j)], dtype=torch.float32) for j in range(len(traj2)))

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

        print("Finished optimizing w and b.")

    def get_weights(self):
        """
        Returns the learned weights and bias.
        """
        return self.w.detach().numpy(), self.b.item()


class MPL_K1:
    def __init__(self, state_dim, lr=0.01):
        """
        Initializes a linear reward model with handcrafted features and bias:
        R(s) = w^T * phi(s) + b
        """
        self.w = torch.nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def phi(self, state_vector):
        """
        Extracts handcrafted features from a state vector.
        Expected state_vector format: [x1, x2, y1, y2, m0, ..., m63] (length 67)
        """
        x1, x2, y1, y2 = map(int, state_vector[:4])
        apple_grid = np.array(state_vector[4:]).reshape(8, 8)
        apples = np.argwhere(apple_grid == 1)

        pos_a1 = np.array([x1, x2])
        pos_a2 = np.array([y1, y2])

        phi_1 = len(apples)
        dists_a1 = [np.linalg.norm(pos_a1 - np.array(p)) for p in apples] if apples.size > 0 else [0]
        dists_a2 = [np.linalg.norm(pos_a2 - np.array(p)) for p in apples] if apples.size > 0 else [0]

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
        """
        Computes the estimated reward: R(s) = w^T * phi(s) + b
        """
        features = self.phi(state_vector)
        return torch.dot(self.w, features) + self.b

    def train(self, ranked_trajectories, trajectory_features, num_epochs=100):
        """
        Trains weights w and bias b using ranked trajectories and precomputed handcrafted features.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(trajectory_features[(i, j)], dtype=torch.float32) for j in range(len(traj1)))
                phi_2 = sum(torch.tensor(trajectory_features[(i + 1, j)], dtype=torch.float32) for j in range(len(traj2)))

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
        Initializes a linear reward model with handcrafted features and bias:
        R(s) = w^T * phi(s) + b
        """
        self.w = torch.nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def phi(self, state_vector):
        """
        Extracts handcrafted features from a state vector.
        Expected format: [x1, x2, y1, y2, m0, ..., m63] (length 67)
        """
        x1, x2, y1, y2 = map(int, state_vector[:4])
        apple_grid = np.array(state_vector[4:]).reshape(8, 8)
        apples = np.argwhere(apple_grid == 1)

        pos_a1 = np.array([x1, x2])
        pos_a2 = np.array([y1, y2])

        phi_1 = len(apples)
        dists_a1 = [np.linalg.norm(pos_a1 - p) for p in apples] if len(apples) > 0 else [0]
        dists_a2 = [np.linalg.norm(pos_a2 - p) for p in apples] if len(apples) > 0 else [0]

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
        """
        Computes the estimated reward R(s) = w^T * phi(s) + b
        """
        features = self.phi(state_vector)
        return torch.dot(self.w, features) + self.b

    def train(self, ranked_trajectories, trajectory_features, num_epochs=100):
        """
        Trains the reward model using preference-based learning with adaptive margin.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for i in range(len(ranked_trajectories) - 1):
                traj1, res_i, _, _ = ranked_trajectories[i]
                traj2, res_j, _, _ = ranked_trajectories[i + 1]

                phi_1 = sum(torch.tensor(trajectory_features[(i, j)], dtype=torch.float32) for j in range(len(traj1)))
                phi_2 = sum(torch.tensor(trajectory_features[(i + 1, j)], dtype=torch.float32) for j in range(len(traj2)))

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                margin = abs(res_i - res_j)

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
        Initializes a handcrafted reward model with bias: R(s) = w^T * phi(s) + b
        """
        self.w = torch.nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def phi(self, state_vector):
        """
        Extracts handcrafted features from the state vector.
        Format: [x1, x2, y1, y2, m0, m1, ..., m63] -> total size 67
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

    def train(self, ranked_trajectories, trajectory_features, num_epochs=100):
        """
        Trains the reward function using margin-based preference learning with mixed pair selection.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for _ in range(len(ranked_trajectories) - 1):
                # Mixed sampling: local or global comparisons
                if random.random() < 0.5:
                    i = random.randint(0, len(ranked_trajectories) - 2)
                    traj1, res_i, _, _ = ranked_trajectories[i]
                    traj2, res_j, _, _ = ranked_trajectories[i + 1]
                    phi_1 = sum(torch.tensor(trajectory_features[(i, j)], dtype=torch.float32) for j in range(len(traj1)))
                    phi_2 = sum(torch.tensor(trajectory_features[(i + 1, j)], dtype=torch.float32) for j in range(len(traj2)))
                else:
                    i, j = random.sample(range(len(ranked_trajectories)), 2)
                    if ranked_trajectories[i][1] > ranked_trajectories[j][1]:
                        traj1, res_i, _, _ = ranked_trajectories[i]
                        traj2, res_j, _, _ = ranked_trajectories[j]
                        phi_1 = sum(torch.tensor(trajectory_features[(i, k)], dtype=torch.float32) for k in range(len(traj1)))
                        phi_2 = sum(torch.tensor(trajectory_features[(j, k)], dtype=torch.float32) for k in range(len(traj2)))
                    else:
                        traj1, res_i, _, _ = ranked_trajectories[j]
                        traj2, res_j, _, _ = ranked_trajectories[i]
                        phi_1 = sum(torch.tensor(trajectory_features[(j, k)], dtype=torch.float32) for k in range(len(traj1)))
                        phi_2 = sum(torch.tensor(trajectory_features[(i, k)], dtype=torch.float32) for k in range(len(traj2)))

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
        Initializes a handcrafted reward model with bias: R(s) = w^T * phi(s) + b
        """
        self.w = torch.nn.Parameter(torch.empty(state_dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        with torch.no_grad():
            self.w.uniform_(-0.5, 0.5)
        self.optimizer = optim.Adam([self.w, self.b], lr=lr)

    def phi(self, state_vector):
        """
        Extracts handcrafted features from the state vector.
        Format: [x1, x2, y1, y2, m0, m1, ..., m63] -> total size 67
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

    def train(self, ranked_trajectories, trajectory_features, num_epochs=100):
        """
        Trains the model using margin preference learning with dynamic margin and mixed sampling.
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for _ in range(len(ranked_trajectories) - 1):
                if random.random() < 0.5:
                    i = random.randint(0, len(ranked_trajectories) - 2)
                    traj1, res_i, _, _ = ranked_trajectories[i]
                    traj2, res_j, _, _ = ranked_trajectories[i + 1]
                    phi_1 = sum(torch.tensor(trajectory_features[(i, j)], dtype=torch.float32) for j in range(len(traj1)))
                    phi_2 = sum(torch.tensor(trajectory_features[(i + 1, j)], dtype=torch.float32) for j in range(len(traj2)))
                else:
                    i, j = random.sample(range(len(ranked_trajectories)), 2)
                    if ranked_trajectories[i][1] > ranked_trajectories[j][1]:
                        traj1, res_i, _, _ = ranked_trajectories[i]
                        traj2, res_j, _, _ = ranked_trajectories[j]
                        phi_1 = sum(torch.tensor(trajectory_features[(i, k)], dtype=torch.float32) for k in range(len(traj1)))
                        phi_2 = sum(torch.tensor(trajectory_features[(j, k)], dtype=torch.float32) for k in range(len(traj2)))
                    else:
                        traj1, res_i, _, _ = ranked_trajectories[j]
                        traj2, res_j, _, _ = ranked_trajectories[i]
                        phi_1 = sum(torch.tensor(trajectory_features[(j, k)], dtype=torch.float32) for k in range(len(traj1)))
                        phi_2 = sum(torch.tensor(trajectory_features[(i, k)], dtype=torch.float32) for k in range(len(traj2)))

                Ri = torch.dot(self.w, phi_1) + self.b
                Rj = torch.dot(self.w, phi_2) + self.b

                margin = abs(res_i - res_j)

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


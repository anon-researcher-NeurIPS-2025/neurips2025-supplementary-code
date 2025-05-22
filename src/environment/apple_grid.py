import numpy as np
import random

class AppleGridMDP:
    def __init__(self, grid_size=(8, 8), regen_threshold=16, rate_regen=0.05):
        self.grid_size = grid_size
        self.regen_threshold = regen_threshold
        self.rate_regen = rate_regen

        self.apple_positions = [
            (2, 3), (2, 4),
            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
            (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
            (5, 4), (5, 3)
        ]

        self.initial_grid = np.zeros(grid_size, dtype=int)
        for x, y in self.apple_positions:
            self.initial_grid[x, y] = 1

        self.initial_agent_positions = [(1, 1), (grid_size[0]-2, grid_size[1]-2)]
        self.reset()

    def get_state(self):
        agent_state = np.array(self.agent_positions).flatten()
        grid_state = np.array(self.grid).flatten()
        return np.concatenate([agent_state, grid_state])

    def move_agent(self, agent_idx, action):
        x, y = self.agent_positions[agent_idx]
        new_x, new_y = x, y

        if action == 0 and x > 0: new_x -= 1
        elif action == 1 and x < self.grid_size[0] - 1: new_x += 1
        elif action == 2 and y > 0: new_y -= 1
        elif action == 3 and y < self.grid_size[1] - 1: new_y += 1

        if (new_x, new_y) not in self.agent_positions:
            self.agent_positions[agent_idx] = (new_x, new_y)

    def step(self, actions):
        rewards = [0] * len(actions)

        for i, action in enumerate(actions):
            self.move_agent(i, action)
            x, y = self.agent_positions[i]
            if self.grid[x, y] == 1:
                remaining = sum(self.grid[x, y] for x, y in self.apple_positions)
                rewards[i] = -1000 if remaining == 1 else 1
                self.grid[x, y] = 0

        current_apples = sum(self.grid[x, y] for x, y in self.apple_positions)
        missing_apples = self.regen_threshold - current_apples
        regen_prob = self.rate_regen * (current_apples / self.regen_threshold) if self.regen_threshold > 0 else 0

        if missing_apples > 0:
            agent_positions_set = set(self.agent_positions)
            for _ in range(missing_apples):
                empty = [(x, y) for x, y in self.apple_positions if self.grid[x, y] == 0 and (x, y) not in agent_positions_set]
                if empty and random.random() < regen_prob:
                    new_apple = random.choice(empty)
                    self.grid[new_apple] = 1

        return self.get_state(), rewards

    def render(self):
        grid_display = self.grid.astype(str)
        for i, (x, y) in enumerate(self.agent_positions):
            grid_display[x, y] = '#' if i == 0 else '*'
        print("\n".join(" ".join(row) for row in grid_display))
        print()

    def reset(self):
        self.grid = self.initial_grid.copy()
        self.agent_positions = self.initial_agent_positions.copy()

    def trigger_disruption(self, magnitude=0.4):
        current = [(x, y) for x, y in self.apple_positions if self.grid[x, y] == 1]
        if len(current) > 1:
            to_remove = random.sample(current, int(np.ceil(len(current) * magnitude)))
            for x, y in to_remove:
                self.grid[x, y] = 0

    def set_state(self, state):
        self.agent_positions = [(state[0], state[1]), (state[2], state[3])]
        grid_state = np.array(state[4:]).reshape(self.grid_size)
        if grid_state.shape != self.grid.shape:
            raise ValueError("Grid size mismatch.")
        self.grid = grid_state.copy()


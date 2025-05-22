import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)


class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=1e-7, gamma=0.99, eps_clip=0.01):
        self.model = ActorCritic(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.memory = deque(maxlen=1000)

    def select_action(self, state):
        """Selects an action using the policy"""
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        probs, _ = self.model(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs.squeeze(0)[action].item()

    def store_transition(self, transition):
        """Stores a transition tuple: (state, action, reward, next_state, old_prob)"""
        self.memory.append(transition)

    def train(self):
        """Trains the PPO agent from stored memory"""
        if len(self.memory) < 200:
            return

        transitions = list(self.memory)
        states, actions, rewards, next_states, old_probs = zip(*transitions)

        states = torch.tensor(states, dtype=torch.float32).view(len(states), -1)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)

        new_probs, values = self.model(states)
        new_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze()
        advantages = rewards - values.squeeze()

        ratio = new_probs / old_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean() + 0.5 * (advantages**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.clear()

import numpy as np
import random
import torch

from src.environment.apple_grid import AppleGridMDP
from src.metrics.hunger_and_equality import calculate_hunger, calculate_equality
from src.metrics.resilience_metrics import ResilienceMetrics

class TrajectoryCollector:
    def __init__(self, env):
        self.env = env

    def collect_ranked_trajectories_random(self, num_episodes=100, disruption_step=500,
                                           range_episode=1000, random_state=0):
        """
        Collects trajectories using random agents and evaluates them with a resilience score.

        Returns:
            ranked_trajectories: list of tuples (trajectory, resilience_score, agent1_total, agent2_total)
        """
        np.random.seed(random_state)
        random.seed(random_state)

        ranked_trajectories = []
        n = 2  # number of agents

        for episode in range(num_episodes):
            self.env.reset()
            env_baseline = AppleGridMDP()
            env_baseline.reset()

            state = self.env.get_state()
            state_baseline = env_baseline.get_state()

            r1, r2 = np.zeros(range_episode), np.zeros(range_episode)
            r1_base, r2_base = np.zeros(range_episode), np.zeros(range_episode)
            apples, apples_base = [], []
            trajectory = []

            for i in range(range_episode):
                if i == disruption_step:
                    self.env.trigger_disruption()

                if i < disruption_step:
                    env_baseline.set_state(tuple(state))
                    state_baseline = state
                else:
                    actions_bl = [random.randint(0, 3), random.randint(0, 3)]
                    state_baseline, rewards_bl = env_baseline.step(actions_bl)

                actions = [random.randint(0, 3), random.randint(0, 3)]
                state, rewards = self.env.step(actions)

                r1[i], r2[i] = max(rewards[0], 0), max(rewards[1], 0)

                if i < disruption_step:
                    r1_base[i], r2_base[i] = r1[i], r2[i]
                else:
                    r1_base[i] = max(rewards_bl[0], 0)
                    r2_base[i] = max(rewards_bl[1], 0)

                apples.append(np.sum(self.env.grid))
                apples_base.append(np.sum(env_baseline.grid))
                trajectory.append(state)

            hunger = calculate_hunger(range_episode, n, [r1, r2])
            hunger_base = calculate_hunger(range_episode, n, [r1_base, r2_base])
            gini = calculate_equality(range_episode, n, [r1, r2])
            gini_base = calculate_equality(range_episode, n, [r1_base, r2_base])

            Pset = {0: [apples, gini, hunger, np.cumsum(r1), np.cumsum(r2)]}
            Rset = {0: [apples_base, gini_base, hunger_base, np.cumsum(r1_base), np.cumsum(r2_base)]}
            disturbances = {0: [disruption_step]}

            rm = ResilienceMetrics(K=5, numberScenarios=1, assemblyIndicatorFuction='harmonic')
            resilience_score, _ = rm.fit(disturbances, Pset, Rset)

            ranked_trajectories.append((trajectory, resilience_score, r1.sum(), r2.sum()))

        return ranked_trajectories

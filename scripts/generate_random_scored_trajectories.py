import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import matplotlib.pyplot as plt
import numpy as np
from src.trajectories.collect_ranked_trajs import TrajectoryCollector
from src.environment.apple_grid import AppleGridMDP
from src.utils.save_and_load import save_data

env = AppleGridMDP()
collector = TrajectoryCollector(env)

ranked_example = collector.collect_ranked_trajectories_random(num_episodes=50, 
                                                              range_episode=1000, 
                                                              disruption_step=500)

# Extract metrics
resilience_scores = [res[1] for res in ranked_example]
agent1_consumption = [res[2] for res in ranked_example]
agent2_consumption = [res[3] for res in ranked_example]

# --- Boxplot: Cooperative Resilience --- #
fig, ax = plt.subplots(figsize=(9, 5))
box = ax.boxplot(resilience_scores, labels=['Ranked Trajectories Examples'],
                 patch_artist=False, showmeans=True, showfliers=True,
                 capprops=dict(color='black'),
                 whiskerprops=dict(color='black'),
                 flierprops=dict(marker='.', color='blue', alpha=1),
                 medianprops=dict(color='blue', linewidth=3))

# Markers
mean_r = np.mean(resilience_scores)
median_r = np.median(resilience_scores)
ax.plot(1, mean_r, 'g^', label='Mean', markersize=8)
ax.plot(1, median_r, 'bs', label='Median', markersize=6)

# Style
ax.set_ylabel('Cooperative Resilience', fontsize=24)
ax.grid(linestyle='--', alpha=0.7)
ax.legend(loc='upper right', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(0.9)
plt.show()

# --- Boxplot: Agent Consumption --- #
plt.figure(figsize=(9, 5))
box = plt.boxplot([agent1_consumption, agent2_consumption], labels=['Agent 1', 'Agent 2'],
                  patch_artist=False, showmeans=True, showfliers=False,
                  capprops=dict(color='black'),
                  whiskerprops=dict(color='black'),
                  flierprops=dict(marker='.', color='blue', alpha=1),
                  medianprops=dict(color='blue', linewidth=3))

# Markers for agent 1
mean_a1 = np.mean(agent1_consumption)
median_a1 = np.median(agent1_consumption)
plt.plot(1, mean_a1, 'g^', label='Mean', markersize=8)
plt.plot(1, median_a1, 'bs', label='Median', markersize=6)

# Markers for agent 2
mean_a2 = np.mean(agent2_consumption)
median_a2 = np.median(agent2_consumption)
plt.plot(2, mean_a2, 'g^', markersize=8)
plt.plot(2, median_a2, 'bs', markersize=6)

# Style
plt.ylabel('Agent Consumption', fontsize=24)
plt.grid(linestyle='--', alpha=0.7)
plt.legend(fontsize=20)
plt.tight_layout()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

save_data(ranked_example, "data/trajectories_example.pkl")

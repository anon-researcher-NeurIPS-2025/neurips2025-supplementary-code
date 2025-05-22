import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.utils.save_and_load import load_data

# Load heatmap data
agent1_map = load_data("results/trajectories_maps/resilience_agent1.pkl")
agent2_map = load_data("results/trajectories_maps/resilience_agent2.pkl")

# Normalize
norm_agent1 = agent1_map / 500
norm_agent2 = agent2_map / 500

# Configure plot appearance
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Create layout
fig = plt.figure(figsize=(6, 4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 0.05])

# Heatmap axes
ax = fig.add_subplot(gs[0, 0])
sns.heatmap(norm_agent1, ax=ax, cmap="BuGn", cbar=False, alpha=0.7, linewidths=0.5, linecolor='white')
sns.heatmap(norm_agent2, ax=ax, cmap="RdPu", cbar=False, alpha=0.5, linewidths=0.5, linecolor='white')

# Colorbar for Agent 1
cax1 = fig.add_subplot(gs[0, 1])
sm1 = ScalarMappable(cmap="BuGn", norm=Normalize(vmin=0, vmax=1))
sm1.set_array([])
fig.colorbar(sm1, cax=cax1, label='Agent 1 Position Frequency')

# Colorbar for Agent 2
cax2 = fig.add_subplot(gs[0, 2])
sm2 = ScalarMappable(cmap="RdPu", norm=Normalize(vmin=0, vmax=1))
sm2.set_array([])
fig.colorbar(sm2, cax=cax2, label='Agent 2 Position Frequency')

# Highlight apple zones
apple_zones = [
    patches.Rectangle((1, 3), 6, 2, linewidth=1, facecolor='red', alpha=0.3),
    patches.Rectangle((3, 2), 2, 1, linewidth=1, facecolor='red', alpha=0.3),
    patches.Rectangle((3, 5), 2, 1, linewidth=1, facecolor='red', alpha=0.3)
]
for zone in apple_zones:
    ax.add_patch(zone)

ax.text(4, 3.8, "Apple Positions", color='red', ha='center', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])

# Final layout
plt.tight_layout()
plt.savefig("figures/resilience_overlay.png", format="png", bbox_inches='tight')
plt.show()

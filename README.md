# Learning Rewards Functions for Cooperative Resilience in Multi-Agent Systems
## Supplementary Code

This repository accompanies the paper **"Learning Rewards Functions for Cooperative Resilience in Multi-Agent Systems" (NeurIPS 2025)**, which investigates how reward function design impacts **cooperative resilience** in Multi-Agent Reinforcement Learning (MARL).

In dynamic and failure-prone environments, agents must not only optimize individual objectives but also ensure the **collective system remains functional under disruptions**. We define cooperative resilience as the ability of agents to **anticipate, resist, recover, and adapt** in the presence of external shocks. This repository provides tools and experiments to study and improve this emergent property through IRL-guided reward learning.

We introduce a novel **reward learning framework** that learns reward functions from **ranked trajectories**—evaluated via a cooperative resilience score. Agents are then trained in **social dilemma environments** using:

* **(i)** Traditional individual reward functions
* **(ii)** Inferred rewards aligned with cooperative resilience
* **(iii)** Hybrid rewards combining both

The reward inference is performed using two preference-based IRL algorithms across three types of parameterizations:

* **Handcrafted features**
* **Linear reward models**
* **Neural networks**

Our results show that **resilience-guided rewards** lead to improved robustness and coordination, helping agents avoid catastrophic outcomes (e.g., resource depletion), without sacrificing individual performance. 

---

## 📁 Repository Structure

* `src/`: Core source code (agents, enviroment, IRL models, metrics, trajectories, utils)
* `data/`: Sample trajectories and inferred reward models (learning)
* `models/`: Trained PPO agents grouped by strategy (baseline, hybrid, resilience) also include the best (best)
* `scripts/`: Training and evaluation scripts
* `visualization/`: Tools for plotting heatmaps, trajectory maps, and feature visualizations
* `results/`: Saved results from experimental runs
* `resilience/`: Metrics to evaluate cooperative resilience

---

## 🚀 Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/anon-researcher-NeurIPS-2025/neurips2025-supplementary-code.git
```

2. **Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📈 Generating Trajectories & Resilience Scores

You can generate your own agent-environment trajectories using:

```bash
python scripts/generate_random_scored_trajectories.py
```

Our experiments rely on **precomputed resilience-scored trajectories**, located in: https://drive.google.com/file/d/1Y4hkSGUbrzo-NVPo5w8Tk32KYBCAVhuZ/view?usp=sharing 

The metric used to **evaluate and rank trajectories** by cooperative resilience is implemented in:

```bash
resilience/resilience_metrics.py
```

This metric combines fairness, sustainability, and disruption recovery into a unified score.

---


## ⚙️ Training IRL Models

Run the scripts to train IRL models with handcrafted, linear, or neural net parameterizations:

```bash
python scripts/learning_irl_handcrafted_model.py
python scripts/learning_irl_linear_model.py
python scripts/learning_irl_nn_model.py
```

This will output the learned weights or models into the `data/learning/` folder.

---

## 🧪 Training PPO Agents with Inferred Rewards

Use the following scripts to train PPO agents using the inferred rewards:

```bash
python scripts/train_ppo_with_irl_handcrafted_reward.py
python scripts/train_ppo_with_irl_linear_reward.py
python scripts/train_ppo_with_irl_nn_reward.py
```

Agents are saved in the `models/` folder.

---

## 📊 Visualizations

Generate heatmaps from trained agents:

```bash
python visualization/plot_heatmaps.py
```

Alternatively, visualize agent behavior through an animation that compares a trained policy against a random baseline:

```bash
python visualization/visualization.py
```

You can also watch a demo video here: https://www.youtube.com/watch?v=S9UqFlKAgwE.

---

## 📁 Reproducibility Notes

Pre-trained agents are provided in `models/`:

* `baseline/`: Traditional PPO agents
* `resilience/`: Agents trained with inferred rewards
* `hybrid/`: Agents trained with hybrid strategies
* `best/`: Agents selected for highest cooperative resilience
* `example/`: Dummy agents for code testing

---

## 📩 Contact

For questions, feedback, or potential collaborations, please reach out to: **[Contact information will be provided upon de-anonymization]**


---


"""
train_ppo.py

Trains a reinforcement learning agent to optimize the MBTA transit network
using the custom Gymnasium environment MBTAEnv.

Loads the base MBTA graph, initializes the environment, and trains a
MaskablePPO agent with invalid-action masking enabled so the agent avoids
illegal edge additions/removals.

After training, the model is saved for later evaluation or visualization.

Run (creates mbta_graph.pkl, maskable_mbta_ppo.zip):
    pip install gymnasium stable-baselines3 sb3-contrib networkx matplotlib numpy
    python train_ppo.py
"""

import pickle
from mbta_env import MBTAEnv
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

# CHANGE 
TOTAL_TIMESTEPS = 10240
# DONT CHANGE
MAX_STEPS = 50

class TrainingCallback(BaseCallback):
    def __init__(self, print_freq=100):
        super().__init__()
        self.print_freq = print_freq
        self.episode_rewards = []
        self._current_reward = 0.0  
        

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        if self.n_calls % self.print_freq == 0:
            info = self.locals.get("infos", [{}])[0]
            print(
                f"  step {self.n_calls:6d} | "
                f"mean_tt={info.get('mean_travel_time_min', 0):.2f} min | "
                f"edges={info.get('n_edges', 0)} | "
                f"improvement={info.get('improvement_pct', 0):+.1f}% | "
                f"period={info.get('current_period', '?')}"
            )
        return True


# load base MBTA graph
with open("mbta_data/mbta_graph.pkl", "rb") as f:
    G = pickle.load(f)

# initialize environment
# change render=True to watch it train
env = MBTAEnv(G, max_steps=MAX_STEPS, render=False, budget=5000.0)

model = MaskablePPO("MlpPolicy", env, verbose=1)
callback = TrainingCallback(print_freq=100)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# save trained model
model.save("maskable_mbta_ppo")
env.close()

print("Training complete. Model saved to maskable_mbta_ppo.zip")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(callback.episode_rewards, color="#1D9E75", linewidth=1.0, alpha=0.4, label="per episode")

# rolling average over 10 episodes
window = 10
rolling = np.convolve(callback.episode_rewards, np.ones(window)/window, mode='valid')

plt.plot(range(window-1, len(callback.episode_rewards)), rolling, color="#1D9E75", linewidth=2.0, label="10-ep average")

plt.title("PPO — reward per episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ppo_training_curve.png", dpi=150)
plt.show()
print("Chart saved to ppo_training_curve.png")
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
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.mbta_env import MBTAEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

# CHANGE 
TOTAL_TIMESTEPS = 3
# DONT CHANGE
MAX_STEPS = 50

class TrainingCallback(BaseCallback):
    def __init__(self, print_freq=100):
        super().__init__()
        self.print_freq = print_freq
        self.episode_rewards = []
        self._current_reward = 0.0  
        self.episode_mean_tts = []

        

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
            info = self.locals.get("infos", [{}])[0]
            self.episode_mean_tts.append(info.get("mean_travel_time_min", 0))
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
with open(os.path.join(PROJECT_ROOT, "outputs", "mbta_graph.pkl"), "rb") as f:
    G = pickle.load(f)

# initialize environment
# change render_mode="human" to watch it train
env = MBTAEnv(G, max_steps=MAX_STEPS, render_mode=None, budget=5000.0)

model = MaskablePPO("MlpPolicy", env, verbose=1)
callback = TrainingCallback(print_freq=100)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# run tag encodes key hyperparameters for easy identification
RUN_TAG = f"ppo_ts{TOTAL_TIMESTEPS}_steps{MAX_STEPS}"

# save trained model
model_path = os.path.join(PROJECT_ROOT, "outputs", "models", f"{RUN_TAG}")
model.save(model_path)
print(f"Training complete. Model saved to {model_path}.zip")

# compute baseline from the environment for the plot
baseline_obs, baseline_info = env.reset()
baseline_tt = baseline_info["mean_travel_time_min"]
env.close()

import matplotlib.pyplot as plt
plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
window = 10

# --- mean travel time plot ---
plt.figure(figsize=(10, 5))
plt.plot(callback.episode_mean_tts, color="#1D9E75", linewidth=1.0, alpha=0.4, label="per episode")
rolling_tt = np.convolve(callback.episode_mean_tts, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(callback.episode_mean_tts)), rolling_tt, color="#1D9E75", linewidth=2.0, label="10-ep average")
plt.axhline(y=baseline_tt, color='red', linestyle='--', linewidth=1.5, label=f"baseline ({baseline_tt:.2f} min)")
plt.title(f"PPO — mean travel time per episode ({RUN_TAG})")
plt.xlabel("Episode")
plt.ylabel("Mean travel time (min)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
tt_path = os.path.join(plots_dir, f"{RUN_TAG}_mean_tt.png")
plt.savefig(tt_path, dpi=150)
plt.show()
print(f"Chart saved to {tt_path}")

# --- reward plot ---
plt.figure(figsize=(10, 5))
plt.plot(callback.episode_rewards, color="#1D9E75", linewidth=1.0, alpha=0.4, label="per episode")
rolling_rw = np.convolve(callback.episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(callback.episode_rewards)), rolling_rw, color="#1D9E75", linewidth=2.0, label="10-ep average")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
plt.title(f"PPO — reward per episode ({RUN_TAG})")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
rw_path = os.path.join(plots_dir, f"{RUN_TAG}_reward.png")
plt.savefig(rw_path, dpi=150)
plt.show()
print(f"Chart saved to {rw_path}")
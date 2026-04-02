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

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from mbta_env import MBTAEnv

# load base MBTA graph
with open("mbta_data/mbta_graph.pkl", "rb") as f:
    G = pickle.load(f)

# initialize environment
# change render=True to watch it train
env = MBTAEnv(G, render=False)

# train MaskablePPO agent with action masking
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5)

# save trained model
model.save("maskable_mbta_ppo")
env.close()
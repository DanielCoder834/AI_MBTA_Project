"""
evaluate.py

Loads a trained MaskablePPO model and runs it inside the MBTAEnv environment
to visualize how the agent modifies the transit network.

Rendering shows the evolving graph and tracks improvements in mean commuter
travel time as the agent applies actions step-by-step.
"""

import pickle

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from mbta_env import MBTAEnv

# load saved MBTA network graph used during training
with open("mbta_data/mbta_graph.pkl", "rb") as f:
    G = pickle.load(f)

# initialize environment with rendering enabled to visualize network changes
env = MBTAEnv(G, render=True)

# load trained MaskablePPO agent
model = MaskablePPO.load("maskable_mbta_ppo")

# reset environment to its initial state
obs, info = env.reset()

done = False

# run one evaluation episode using trained policy
while not done:
    # retrieve current invalid action mask from environment
    masks = get_action_masks(env)
    # select an action using the trained model (deterministic policy)
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    # apply action and update env state
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    # stop when episode ends (naturally or by step limit)
    done = terminated or truncated

env.close()
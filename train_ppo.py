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

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback
MAX_STEPS = 50

class TrainingCallback(BaseCallback):
    def __init__(self, print_freq=10):
        super().__init__()
        self.print_freq = print_freq

    def _on_step(self) -> bool:
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
'''
with open("mbta_data/mbta_graph.pkl", "rb") as f:
    G = pickle.load(f)
'''

# Create a mock 10-node MBTA graph for testing
# Nodes and edges mirror the structure built by network.py:
#   node attrs : name, lat, lon, lines
#   edge attrs : line, travel_time_min
import networkx as nx
G = nx.Graph()

MOCK_NODES = {
    "place-pktrm": {"name": "Park Street",        "lat": 42.3564, "lon": -71.0632, "lines": ["green", "red"]},
    "place-dwnxg": {"name": "Downtown Crossing",  "lat": 42.3553, "lon": -71.0600, "lines": ["orange", "red"]},
    "place-state": {"name": "State",               "lat": 42.3588, "lon": -71.0577, "lines": ["blue", "orange"]},
    "place-north": {"name": "North Station",       "lat": 42.3657, "lon": -71.0618, "lines": ["green", "orange"]},
    "place-haecl": {"name": "Haymarket",           "lat": 42.3627, "lon": -71.0586, "lines": ["green", "orange"]},
    "place-sstat": {"name": "South Station",       "lat": 42.3523, "lon": -71.0551, "lines": ["red"]},
    "place-brdwy": {"name": "Broadway",            "lat": 42.3426, "lon": -71.0566, "lines": ["red"]},
    "place-bbsta": {"name": "Back Bay",            "lat": 42.3470, "lon": -71.0780, "lines": ["orange"]},
    "place-chncl": {"name": "Chinatown",           "lat": 42.3526, "lon": -71.0626, "lines": ["orange"]},
    "place-tumnl": {"name": "Tufts Medical Center","lat": 42.3488, "lon": -71.0640, "lines": ["orange"]},
}

MOCK_EDGES = [
    # (src, dst, travel_time_min, line)  — matches t_edges.txt format
    ("place-pktrm", "place-dwnxg", 1, "red"),
    ("place-dwnxg", "place-sstat", 2, "red"),
    ("place-sstat", "place-brdwy", 2, "red"),
    ("place-haecl", "place-north", 1, "orange"),
    ("place-state", "place-haecl", 1, "orange"),
    ("place-dwnxg", "place-state", 1, "orange"),
    ("place-chncl", "place-dwnxg", 2, "orange"),
    ("place-tumnl", "place-chncl", 1, "orange"),
    ("place-bbsta", "place-tumnl", 2, "orange"),
]

for node_id, attrs in MOCK_NODES.items():
    G.add_node(node_id, **attrs)

for src, dst, travel_time, line in MOCK_EDGES:
    G.add_edge(src, dst, line=line, travel_time_min=travel_time)



# initialize environment
# change render=True to watch it train
env = MBTAEnv(G, max_steps=MAX_STEPS, render=False)

# train MaskablePPO agent with action masking
model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=5,
    learning_rate=3e-4,
)
model.learn(total_timesteps=10000, log_interval=1, callback=TrainingCallback(print_freq=100))

# save trained model
model.save("maskable_mbta_ppo")
env.close()

print("Training complete. Model saved to maskable_mbta_ppo.zip")

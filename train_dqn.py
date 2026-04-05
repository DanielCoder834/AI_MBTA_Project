"""
train_dqn.py

Trains DQN agent on the MBTAEnv reinforcement learning environment.

The agent learns how to modify the MBTA transit graph by adding/removing
edges to minimize average commuter travel time.

Run (creates dqn_mbta.pt with trained Q-network weights):
python train_dqn.py
"""

import pickle
import numpy as np

from mbta_env import MBTAEnv
from dqn_agent import DQNAgent

NUM_EPISODES = 200
MAX_STEPS = 50

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



# create environment
env = MBTAEnv(G, max_steps=MAX_STEPS, render=False)

# get observation/action sizes
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# initialize agent
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.98,
    buffer_capacity=5000,
    batch_size=64,
    target_update_freq=50,
)

episode_rewards = []
total_steps = 0

# training for loop
for episode in range(NUM_EPISODES):
    state, info = env.reset()
    done = False
    total_reward = 0.0
    losses = []

    while not done:
        total_steps += 1
        # get valid action mask from env
        valid_mask = env.action_masks()
        # choose action
        action = agent.select_action(state, valid_mask=valid_mask)
        # apply action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # store transition
        agent.store_transition(state, action, reward, next_state, done)
        # track losses
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
        
        state = next_state
        total_reward += reward
    
    # reduce exploration overt time
    agent.decay_epsilon()
    episode_rewards.append(total_reward)

    mean_loss = np.mean(losses) if losses else 0.0

    print(
        f"Episode {episode+1:3d} | "
        f"Total steps: {total_steps:6d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Epsilon: {agent.epsilon:.3f} | "
        f"Mean TT: {info['mean_travel_time_min']:.2f} | "
        f"Loss: {mean_loss:.4f}"
    )

# save trained model
agent.save("dqn_mbta.pt")

env.close()
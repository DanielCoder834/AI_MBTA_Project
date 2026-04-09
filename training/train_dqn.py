"""
train_dqn.py

Trains DQN agent on the MBTAEnv reinforcement learning environment.

The agent learns how to modify the MBTA transit graph by adding/removing
edges to minimize average commuter travel time.

Run (creates dqn_mbta.pt with trained Q-network weights):
python train_dqn.py
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.mbta_env import MBTAEnv
from agents.dqn_agent import DQNAgent

# CHANGE
VERSION        = 2     # v1: budget=5000, add_cost=w*2, no remove refund
                       # v2: budget=1000, add_cost=w*5, remove refund=tt*2.5,
                       #     freq actions modify travel_time (±0.5min), speed_up cost=1.5, slow_down refund=0.75
NUM_EPISODES   = 100
EPSILON_DECAY  = 0.995

# DONT CHANGE
MAX_STEPS = 50

# load base MBTA graph
with open(os.path.join(PROJECT_ROOT, "outputs", "mbta_graph.pkl"), "rb") as f:
    G = pickle.load(f)

# create environment
env = MBTAEnv(G, max_steps=MAX_STEPS, render=False, budget=1000.0)

# get observation/action sizes
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# initialize agent
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=EPSILON_DECAY,
    buffer_capacity=5000,
    batch_size=64,
    target_update_freq=200,
)

episode_rewards = []
episode_mean_tts = []
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
    episode_mean_tts.append(info['mean_travel_time_min'])


    mean_loss = np.mean(losses) if losses else 0.0

    print(
        f"Episode {episode+1:3d} | "
        f"Total steps: {total_steps:6d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Epsilon: {agent.epsilon:.3f} | "
        f"Mean TT: {info['mean_travel_time_min']:.2f} | "
        f"Loss: {mean_loss:.4f}"
    )

# run tag encodes key hyperparameters for easy identification
RUN_TAG = f"dqn_v{VERSION}_ep{NUM_EPISODES}_lr{agent.optimizer.param_groups[0]['lr']}_ed{EPSILON_DECAY}_buf{agent.replay_buffer.buffer.maxlen}_tgt{agent.target_update_freq}"

# save trained model
model_path = os.path.join(PROJECT_ROOT, "outputs", "models", f"{RUN_TAG}.pt")
agent.save(model_path)
print(f"Model saved to {model_path}")

# compute baseline from the environment for the plot
baseline_obs, baseline_info = env.reset()
baseline_tt = baseline_info["mean_travel_time_min"]
env.close()

plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
window = 10

# --- mean travel time plot ---
plt.figure(figsize=(10, 5))
plt.plot(episode_mean_tts, color="#378ADD", linewidth=1.0, alpha=0.4, label="per episode")
rolling_tt = np.convolve(episode_mean_tts, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(episode_mean_tts)), rolling_tt, color="#378ADD", linewidth=2.0, label="10-ep average")
plt.axhline(y=baseline_tt, color='red', linestyle='--', linewidth=1.5, label=f"baseline ({baseline_tt:.2f} min)")
plt.title(f"DQN — mean travel time per episode ({RUN_TAG})")
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
plt.plot(episode_rewards, color="#378ADD", linewidth=1.0, alpha=0.4, label="per episode")
rolling_rw = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(episode_rewards)), rolling_rw, color="#378ADD", linewidth=2.0, label="10-ep average")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
plt.title(f"DQN — reward per episode ({RUN_TAG})")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
rw_path = os.path.join(plots_dir, f"{RUN_TAG}_reward.png")
plt.savefig(rw_path, dpi=150)
plt.show()
print(f"Chart saved to {rw_path}")
"""
dqn_agent.py

Implements a Deep Q-Network (DQN) agent for MBTA reinforcement learning environment.

This agent learns a Q-function mapping states to action values using:

- epsilon-greedy exploration
- experience replay buffer
- target network stabilization
- Bellman update rule
"""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """
    Stores transitions (state, action, reward, next_state, done)
    for experience replay during DQN training
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add one transition to the replay buffer"""
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int):
        """Sample a random minibatch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural network approximating Q(s,a)
    Takes environment state as input and outputs Q-values for all actions
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        """Forward pass through Q-network"""
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network agent
    Uses epsilon-greedy exploration, replay buffer, target network, Bellman updates
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        device=None,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # choose GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # main Q-network
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)

        # target network (stabilizes training)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # experience replay memory
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.train_steps = 0

    def select_action(self, state, valid_mask=None):
        """
        Select action using epsilon-greedy strategy
        With probability epsilon, choose random valid action
        Otherwise, choose action with highest Q-value
        """

        if random.random() < self.epsilon:
            if valid_mask is not None:
                valid_actions = np.where(valid_mask)[0]
                if len(valid_actions) > 0:
                    return int(np.random.choice(valid_actions))

            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        if valid_mask is not None:
            q_values = np.where(valid_mask, q_values, -1e9)

        return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done):
        """Save transition to replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one gradient update step using replay buffer minibatch
        """

        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # predicted Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # periodically update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Gradually reduce exploration over training"""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save trained model"""
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        """Load trained model"""
        self.q_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
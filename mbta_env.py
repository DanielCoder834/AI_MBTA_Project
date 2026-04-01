"""
mbta_env.py  —  Gymnasium environment for MBTA network optimisation

Actions available to the agent:
  0  ADD_EDGE    — connect any two stations with a new edge
  1  REMOVE_EDGE — remove an existing edge

Reward: negative mean travel time for all commuters (start-destination pairs)
- (maximising reward ≡ minimising average commuter travel time).
- Disconnected pairs are penalised with a large constant.

Observation:
A 1D array of 5 normalized scalar features describing current MBTA network state:
  1: normalized mean travel time [0, 1]
     - mean commuter travel time divided by DISCONNECT_PENALTY
  2: normalized edge count [0, 1]
     - current number of edges divided by N^2
     - how dense the network currently is
  3: normalized improvement [-1, 1]
     - percent improvement relative to baseline network
     - (baseline_mean − current_mean) / baseline_mean
  4: reachability ratio [0, 1]
     - fraction of commuter origin–destination pairs that remain connected
  5: normalized mean node degree [0, 1]
     - average node degree divided by number of stations N
     - overall network connectivity level

Example observation:
[ normalized_mean_travel_time,
  normalized_edge_density,
  normalized_improvement,
  reachability_ratio,
  normalized_mean_degree ]
"""

import copy  
import pickle    
import time
from typing import Any 
from matplotlib.pylab import norm
import networkx as nx  
from commuter_model import CommuterPopulation
import numpy as np     
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt


# CONSTANTS

DISCONNECT_PENALTY = 500.0
MAX_STEPS = 500


class MBTAEnv(gym.Env):
    """
    Gymnasium environment for optimising the MBTA rapid-transit network.

    base_graph : nx.Graph
        The starting MBTA network.
    max_steps : int
        Maximum steps per episode.
    disconnect_penalty : float
        Travel-time penalty charged per unreachable start-destination station pair.
    render : bool
        Whether to render the network graph after each action (slows down training).
    """

    # rendering modes we support
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        base_graph: nx.Graph,
        max_steps: int = MAX_STEPS,
        disconnect_penalty: float = DISCONNECT_PENALTY,
        render: bool = False,
    ):
        super().__init__()

        self._base_graph = base_graph
        self.nodes: list[str] = sorted(self._base_graph.nodes())
        self.N: int = len(self.nodes)

        # station name to its integer index.
        self._node_idx: dict[str, int] = {n: i for i, n in enumerate(self.nodes)}

        self.max_steps = max_steps
        self.disconnect_penalty = disconnect_penalty
        self._render_enabled = render

        # working copy of the graph the agent will modify
        self._G = None

        # variables to track reward and improvement calculations
        self._prev_mean_tt = None
        self._baseline_mean = None
        self._step_count = 0

        # [action_type, node_u, node_v]
        self.action_space = spaces.MultiDiscrete([2, self.N, self.N])

        # example observation:
        # [ normalized_mean_travel_time,
        #   normalized_edge_density,
        #   normalized_improvement,
        #   reachability_ratio,
        #   normalized_mean_degree ]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # create the commuter population with the given graph and number of commuters
        self.commuters = CommuterPopulation(base_graph)

    # resets env to initial state at start of new episode
    # MBTA network to original base graph, resets reward variables, recomputes baseline travel time
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        # restore graph
        self._G = copy.deepcopy(self._base_graph)
        self.commuters.update_graph(self._G)
        self._step_count = 0

        # build cached node positions once
        self._pos = {
            n: (d["lon"], d["lat"])
            for n, d in self._G.nodes(data=True)
            if d.get("lon") is not None and d.get("lat") is not None
        }

        self._baseline_mean = self._mean_travel_time()
        self._prev_mean_tt = self._baseline_mean
        reachability = self._reachability()
        obs = self._observation(self._baseline_mean, reachability)
        info = self._info(self._baseline_mean)
        return obs, info
    
    # determine whether adding an edge between two stations is allowed
    def _is_valid_add(self, u: str, v: str) -> bool:
        return u != v and not self._G.has_edge(u, v)

    # determine whether removing an edge between two stations is allowed
    def _is_valid_remove(self, u: str, v: str) -> bool:
        if not self._G.has_edge(u, v):
            return False

        # do not allow removing bridge edges
        bridges = set(map(frozenset, nx.bridges(self._G)))
        return frozenset((u, v)) not in bridges

    # determine whether a proposed action is valid for current network state
    def _is_valid_action(self, action_type: int, u: str, v: str) -> bool:
        if action_type == 0:
            return self._is_valid_add(u, v)
        if action_type == 1:
            return self._is_valid_remove(u, v)
        return False

    def action_masks(self) -> list[np.ndarray]:
        # action mask for MultiDiscrete([2, N, N]) to describe which actions are valid

        # for SB3 MaskablePPO + MultiDiscrete, return one boolean array per action dimension:
        #    - mask[0]: valid action_type values
        #    - mask[1]: valid u indices
        #    - mask[2]: valid v indices

        # validity depends on (action_type, u, v), so safest mask:
        #   - allows both action types only if each has at least one valid (u, v)
        #   - allows all u, v indices, then reject invalid full triples in step() as a fallback
        add_exists = False
        remove_exists = False

        for u in self.nodes:
            for v in self.nodes:
                if self._is_valid_add(u, v):
                    add_exists = True
                if self._is_valid_remove(u, v):
                    remove_exists = True
                if add_exists and remove_exists:
                    break
            if add_exists and remove_exists:
                break

        action_type_mask = np.array([add_exists, remove_exists], dtype=bool)

        # multiDiscrete masks are per-dimension, not full joint masks
        # so keep node dimensions open and do exact validity check in step()
        u_mask = np.ones(self.N, dtype=bool)
        v_mask = np.ones(self.N, dtype=bool)

        return np.concatenate([action_type_mask, u_mask, v_mask])

    # apply graph action chosen by the agent 
    def _apply_action(self, action_type: int, u: str, v: str) -> bool:
        # add edge
        if action_type == 0:
            if self._is_valid_add(u, v):
                w = self.commuters.edge_weight_from_distance(u, v)
                self._G.add_edge(u, v, travel_time_min=w, line="new")
                return True
            return False

        # remove edge
        if action_type == 1:
            if self._is_valid_remove(u, v):
                self._G.remove_edge(u, v)
                return True
            return False

        return False
    
    # apply one action and advance the environment by one timestep
    # returns obs, reward, terminated, truncated, info
    def step(self, action: np.ndarray):
        action_type, u_idx, v_idx = map(int, action)
        # convert integer indices back to station name strings
        u = self.nodes[u_idx]
        v = self.nodes[v_idx]

        # mutate graph for chosen action
        valid = self._apply_action(action_type, u, v)
        self.commuters.update_graph(self._G)
        self._step_count += 1

        # recompute mean travel time
        mean_tt = self._mean_travel_time()

        reachability = self._reachability()

        # reward = change in travel time compared to the previous step
        reward = self._prev_mean_tt - mean_tt
        self._prev_mean_tt = mean_tt

        # fallback guard for action_masks(): MultiDiscrete masking is only per-dimension,
        # so still penalize impossible joint triples (action_type, u, v) if sampled somehow
        if not valid:
            reward -= 1.0

        # no "winning" state, agent just runs until max_steps
        terminated = False
        truncated = self._step_count >= self.max_steps

        obs = self._observation(mean_tt, reachability)
        info = self._info(mean_tt)

        return obs, reward, terminated, truncated, info

    def render(self):
        if not self._render_enabled:
            return
        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(12, 8))

        self._ax.clear()

        nx.draw_networkx_edges(self._G, self._pos, ax=self._ax)
        nx.draw_networkx_nodes(self._G, self._pos, node_size=20, ax=self._ax)

        self._ax.set_title(f"Mean TT: {self._mean_travel_time():.2f}")
        self._ax.axis("off")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def _mean_travel_time(self) -> float:
        return self.commuters.get_mean_commute_time()
    
    def _reachability(self) -> float:
        reachable = 0
        for c in self.commuters.commuters:
            if nx.has_path(self._G, c.home_station, c.work_station):
                reachable += 1
        return reachable / max(len(self.commuters.commuters), 1)
    
    def _observation(self, mean_tt: float, reachability: float) -> np.ndarray:
        norm_tt = float(np.clip(mean_tt / DISCONNECT_PENALTY, 0.0, 1.0))

        n_edges = self._G.number_of_edges()
        norm_edges = float(n_edges / max(self.N ** 2, 1))

        improvement = (
            (self._baseline_mean - mean_tt) / self._baseline_mean
            if self._baseline_mean and self._baseline_mean > 0
            else 0.0
        )
        norm_improvement = float(np.clip(improvement, -1.0, 1.0))

        degrees = [d for _, d in self._G.degree()]
        mean_degree = float(np.mean(degrees)) / self.N if degrees else 0.0

        return np.array(
            [norm_tt, norm_edges, norm_improvement, reachability, mean_degree],
            dtype=np.float32,
        )

    def _info(self, mean_tt: float | None = None) -> dict:
        # dictionary to get info about the current state of the environment.
        # fields:
        #   - mean_travel_time_min : average commute time in minutes
        #   - improvement_pct      : % improvement vs the baseline (start of episode)
        #   - n_edges              : current number of connections in the network
        #   - step                 : how many steps have been taken this episode
        if mean_tt is None:
            mean_tt = self._mean_travel_time()

        improvement = (
            (self._baseline_mean - mean_tt) / self._baseline_mean * 100
            if self._baseline_mean and self._baseline_mean > 0
            else 0.0
        )

        return {
            "mean_travel_time_min": mean_tt,
            "improvement_pct":      improvement,
            "n_edges":              self._G.number_of_edges(),
            "step":                 self._step_count,
        }

    def close(self):
        if hasattr(self, "_fig"):
            plt.close(self._fig)


# test to make sure it runs without errors and the observation/action spaces look correct.
if __name__ == "__main__":
    # Load the MBTA graph
    with open("mbta_data/mbta_graph.pkl", "rb") as f:
        G = pickle.load(f)

    env = MBTAEnv(G, max_steps=50, render=True)

    print("Running gymnasium env_checker …")
    check_env(env, warn=True)
    print("check_env passed.\n")

    obs, info = env.reset(seed=42)
    print(f"Baseline mean travel time : {info['mean_travel_time_min']:.2f} min")
    print(f"Observation shape         : {obs.shape}")
    print(f"Action space              : {env.action_space}\n")

    # run 50 completely random actions and track total reward
    total_reward = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0 and env._render_enabled:   # render every 5 steps
            env.render()
            time.sleep(0.1)

        # print progress update every 10 steps
        if (step + 1) % 10 == 0:
            print(
                f"  step {step+1:3d} | "
                f"mean_tt={info['mean_travel_time_min']:.2f} min | "
                f"edges={info['n_edges']:3d} | "
                f"improvement={info['improvement_pct']:+.1f}%"
            )

        # shouldn't happen here since terminated is always False and we're within max_steps, but just in case break for loop
        if terminated or truncated:
            break

    print(f"\nTotal reward over episode : {total_reward:.2f}")
"""
mbta_env.py  —  Gymnasium environment for MBTA network optimisation
"""

import copy  
import pickle    
import time
from typing import Any 

import gymnasium as gym
import networkx as nx  
import numpy as np     
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import math

# CONSTANTS
DISCONNECT_PENALTY = 500.0
MAX_STEPS = 500
EDGE_COST = 0.05 
DEFAULT_EDGE_WEIGHT = 10.0

class MBTAEnv(gym.Env):
    """
    Gymnasium environment for optimising the MBTA rapid-transit network.

    base_graph : nx.Graph
        The starting MBTA network.
    max_steps : int
        Maximum steps per episode.
    render : bool
        Whether to render the network graph after each action (slows down training).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        base_graph: nx.Graph,
        max_steps: int = MAX_STEPS,
        render: bool = False,
    ):
        super().__init__()

        self._base_graph = base_graph
        self.nodes: list[str] = sorted(self._base_graph.nodes())
        self.N: int = len(self.nodes)
        self.max_steps = max_steps
        
        self.all_edges = [
            (u, v) for i, u in enumerate(self.nodes) for v in self.nodes[i+1:]
        ]
        self.action_space = spaces.MultiDiscrete([2, self.N, self.N])
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # copy of graph the agent will modify
        self._G = None

        self._prev_mean_tt = None
        self._baseline_mean = None
        self._step_count = 0



        self._render_enabled = render

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Resets the environment to an initial state and returns an initial observation and info.
        """
        super().reset(seed=seed)
        self._G = copy.deepcopy(self._base_graph)
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
    
    def _is_valid_add(self, u: str, v: str) -> bool:
        """ only allow adding edges between distinct stations that aren't already directly connected"""
        return u != v and not self._G.has_edge(u, v)

    def _is_valid_remove(self, u: str, v: str) -> bool:
        """ only allow removing edges that exist and aren't bridges (whose removal would disconnect the graph)"""
        if not self._G.has_edge(u, v):
            return False
        bridges = set(map(frozenset, nx.bridges(self._G)))
        return frozenset((u, v)) not in bridges

    def _is_valid_action(self, action_type: int, u: str, v: str) -> bool:
        """ check if the proposed action is valid for the current graph state """
        if action_type == 0:
            return self._is_valid_add(u, v)
        if action_type == 1:
            return self._is_valid_remove(u, v)
        return False

    def action_masks(self) -> list[np.ndarray]:
        """
        Returns a list of boolean arrays indicating which actions are valid.
        Always returns a mask of all True for right now.
        """
        return np.ones(2 + 2 * self.N, dtype=bool)
    
    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance in kilometers between two points on the Earth."""
        r = 6371.0

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c
    
    def _edge_weight_from_distance(self, u: str, v: str) -> float:
        """ estimate travel time in minutes for a new edge based on the haversine distance between stations """
        try:
            lat1, lon1 = self._G.nodes[u]["lat"], self._G.nodes[u]["lon"]
            lat2, lon2 = self._G.nodes[v]["lat"], self._G.nodes[v]["lon"]
        except KeyError:
            return DEFAULT_EDGE_WEIGHT
        km = self._haversine(lat1, lon1, lat2, lon2)
        return float(max(1.0, round((km / 30.0) * 60.0, 1)))

    def _apply_action(self, action_type: int, u: str, v: str) -> bool:
        """ applies the proposed action to the graph if valid, returns whether the action was valid """
        # add edge
        if action_type == 0:
            if self._is_valid_add(u, v):
                w = self._edge_weight_from_distance(u, v)
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
    
    def step(self, action: np.ndarray):
        """ applies the given action to the environment and returns the new observation, reward, done flags, and info. """
        action_type, u_idx, v_idx = map(int, action)

        u = self.nodes[u_idx]
        v = self.nodes[v_idx]

        # mutate graph for chosen action
        valid = self._apply_action(action_type, u, v)
        self._step_count += 1

        mean_tt = self._mean_travel_time()
        reachability = self._reachability()
        reward = self._prev_mean_tt - mean_tt
        self._prev_mean_tt = mean_tt

        # penalize for more edges 
        reward -= EDGE_COST * self._G.number_of_edges()
       
        # apply penalty for invalid actions - fallback for action masking
        if not valid:
            reward -= 1.0

        # no "winning" state, agent just runs until max_steps
        terminated = False
        truncated = self._step_count >= self.max_steps

        obs = self._observation(mean_tt, reachability)
        info = self._info(mean_tt)

        return obs, reward, terminated, truncated, info

    def render(self):
        """Renders the current state of the environment."""
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
        """ calculates the average travel time in minutes across all pairs of stations, applying a penalty for unreachable pairs. """
        total, count = 0.0, 0

        lengths = dict(
            nx.all_pairs_dijkstra_path_length(self._G, weight="travel_time_min")
        )

        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if i >= j:
                    continue
                count += 1
                dist = lengths.get(u, {}).get(v, None)
                total += dist if dist is not None else DISCONNECT_PENALTY

        return total / count if count > 0 else 0.0
    
    def _reachability(self) -> float:
        """ calculates the fraction of station pairs that are reachable from each other (i.e. have a path in the graph) """
        reachable_pairs = 0
        total_pairs = 0

        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if i >= j:
                    continue
                total_pairs += 1
                if nx.has_path(self._G, u, v):
                    reachable_pairs += 1

        return reachable_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _observation(self, mean_tt: float, reachability: float) -> np.ndarray:
        """ constructs the observation vector based on the current graph state and baseline mean travel time. """
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
        """ constructs the info dictionary with human-readable metrics about the current environment state. """
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


if __name__ == "__main__":
    """Loads the graph, runs the gymnasium env_checker."""

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

        if terminated or truncated:
            break

    print(f"\nTotal reward over episode : {total_reward:.2f}")
"""
mbta_env.py  —  Gymnasium environment for MBTA network optimisation
"""

import copy
import math
import os
import pickle
import time
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env


# CONSTANTS
DISCONNECT_PENALTY = 500.0
MAX_STEPS = 50
MIN_TRAVEL_TIME = 0.5     # minutes — fastest an edge can go
MAX_TRAVEL_TIME = 12.0    # minutes — slowest an edge can go
FREQ_STEP = 0.5           # minutes — each frequency action adjusts travel_time_min by this

class MBTAEnv(gym.Env):
    """
    Gymnasium environment for optimising the MBTA rapid-transit network.

    base_graph : nx.Graph
        The starting MBTA network.
    max_steps : int
        Maximum steps per episode.
    disconnect_penalty : float
        Travel-time penalty charged per unreachable start-destination station pair.
    render_mode : str | None
        Pass "human" to render the network graph after each action (slows down training).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}
    DOWNTOWN = {
        "place-pktrm",   # Park Street
        "place-dwnxg",   # Downtown Crossing
        "place-sstat",   # South Station
        "place-north",   # North Station
        "place-bbsta",   # Back Bay
    }

    SUBURBS = {
        "place-brntn",   # Braintree
        "place-asmnl",   # Alewife
        "place-wondl",   # Wonderland
        "place-forhl",   # Forest Hills
        "place-ogmnl",   # Oak Grove
        "place-bomnl",   # Bowdoin
    }
    TIME_PERIODS = {
        "am_rush":   {"hours": (7, 9),   "weight_downtown": 3.0, "weight_suburb": 0.5, "weight_other": 1.0},
        "pm_rush":   {"hours": (17, 19), "weight_downtown": 0.5, "weight_suburb": 3.0, "weight_other": 1.0},
        "midday":    {"hours": (9, 17),  "weight_downtown": 1.0, "weight_suburb": 1.0, "weight_other": 1.0},
        "evening":   {"hours": (19, 22), "weight_downtown": 0.8, "weight_suburb": 1.2, "weight_other": 0.8},
        "overnight": {"hours": (22, 7),  "weight_downtown": 0.2, "weight_suburb": 0.2, "weight_other": 0.2},
    }

    def __init__(
        self,
        base_graph: nx.Graph,
        max_steps: int = MAX_STEPS,
        disconnect_penalty: float = DISCONNECT_PENALTY,
        render_mode: str | None = None,
        budget: float = 500.0,
    ):
        super().__init__()

        self.render_mode = render_mode
        self._base_graph = base_graph
        self.nodes: list[str] = sorted(self._base_graph.nodes())
        self.N: int = len(self.nodes)

        self.max_steps = max_steps
        self.disconnect_penalty = disconnect_penalty
        self._render_enabled = render_mode == "human"

        self._G = None
        self._prev_mean_tt = None
        self._baseline_mean = None
        self._step_count = 0
        self._current_period = "midday"
        self._hour = 7

        # action_type: 0=ADD_EDGE, 1=REMOVE_EDGE, 2=SPEED_UP_EDGE, 3=SLOW_DOWN_EDGE
        self._num_pairs = self.N * (self.N - 1)  # ordered pairs, no self-loops
        self.num_actions = 4 * self._num_pairs
        self.action_space = spaces.Discrete(self.num_actions)

        self._cached_mask = None
        self._graph_changed = True
        self.budget = budget
        self._remaining_budget = budget

        self.observation_space = spaces.Box(
            low=np.array( [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def encode_action(self, action_type: int, u_idx: int, v_idx: int) -> int:
        pair_idx = u_idx * (self.N - 1) + (v_idx if v_idx < u_idx else v_idx - 1)
        return action_type * self._num_pairs + pair_idx

    def decode_action(self, action: int) -> tuple[int, int, int]:
        action = int(action)
        action_type = action // self._num_pairs
        pair_idx = action % self._num_pairs
        u_idx = pair_idx // (self.N - 1)
        temp = pair_idx % (self.N - 1)
        v_idx = temp if temp < u_idx else temp + 1
        return action_type, u_idx, v_idx

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._G = copy.deepcopy(self._base_graph)
        self._step_count = 0
        self._cached_mask = None
        self._graph_changed = True
        self._remaining_budget = self.budget
        self._hour = 7
        self._current_period = self._get_current_period()
        # build cached node positions once
        self._pos = {
            n: (d["lon"], d["lat"])
            for n, d in self._G.nodes(data=True)
            if d.get("lon") is not None and d.get("lat") is not None
        }
        
        # check edges
        for u, v, data in self._G.edges(data=True):
            if "travel_time_min" not in data:
                raise ValueError(f"Missing travel_time_min on edge {u}-{v} in base graph")

        lengths = self._dijkstra_lengths()
        self._baseline_mean = self._mean_travel_time(lengths)
        self._prev_mean_tt = self._baseline_mean
        reachability = self._reachability(lengths)
        per_line = self._per_line_stats()

        obs = self._observation(self._baseline_mean, reachability, per_line)
        info = self._info(self._baseline_mean)

        return obs, info
    
    def _is_valid_add(self, u: str, v: str) -> bool:
        """ only allow adding edges between distinct stations that aren't already directly connected"""
        return u != v and not self._G.has_edge(u, v)

    def _is_valid_remove(self, u: str, v: str) -> bool:
        """ only allow removing edges that exist and aren't bridges (whose removal would disconnect the graph)"""
        if not self._G.has_edge(u, v):
            return False
        bridges = getattr(self, "_bridges", None)
        if bridges is None or self._graph_changed:
            bridges = set(map(frozenset, nx.bridges(self._G)))
            self._bridges = bridges
        return frozenset((u, v)) not in bridges

    def _is_valid_action(self, action_type: int, u: str, v: str) -> bool:
        """ check if the proposed action is valid for the current graph state """
        if action_type == 0:
            if not self._is_valid_add(u, v):
                return False
            w = self._edge_weight_from_distance(u, v)
            return self._remaining_budget >= w * 5.0
        if action_type == 1:
            return self._is_valid_remove(u, v)
        if action_type == 2:  # speed up
            if not self._G.has_edge(u, v):
                return False
            tt = self._G[u][v].get("travel_time_min", 1)
            return tt > MIN_TRAVEL_TIME + 1e-9 and self._remaining_budget >= FREQ_STEP * 3.0
        if action_type == 3:  # slow down
            if not self._G.has_edge(u, v):
                return False
            tt = self._G[u][v].get("travel_time_min", 1)
            return tt < MAX_TRAVEL_TIME - 1e-9
        return False

    def action_masks(self) -> np.ndarray:
        if not self._graph_changed and self._cached_mask is not None:
            return self._cached_mask
        mask = np.zeros(self.num_actions, dtype=bool)

        # precompute bridges once instead of per-action
        self._bridges = set(map(frozenset, nx.bridges(self._G)))

        for action_type in range(4):
            for u_idx, u in enumerate(self.nodes):
                for v_idx, v in enumerate(self.nodes):
                    if u_idx == v_idx:
                        continue
                    a = self.encode_action(action_type, u_idx, v_idx)
                    mask[a] = self._is_valid_action(action_type, u, v)
        self._cached_mask = mask
        self._graph_changed = False
        return mask
    
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
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _edge_weight_from_distance(self, u: str, v: str) -> float:
        """ estimate travel time in minutes for a new edge based on the haversine distance between stations """
        try:
            lat1, lon1 = self._G.nodes[u]["lat"], self._G.nodes[u]["lon"]
            lat2, lon2 = self._G.nodes[v]["lat"], self._G.nodes[v]["lon"]
        except KeyError:
            raise ValueError(f"Missing coordinates for {u} or {v}")
        km = self._haversine(lat1, lon1, lat2, lon2)
        return float(max(1.0, round((km / 30.0) * 60.0, 1)))

    def _apply_action(self, action_type: int, u: str, v: str) -> bool:
        if action_type == 0:  # add edge
            if not self._is_valid_add(u, v):
                return False
            w = self._edge_weight_from_distance(u, v)
            cost = w * 5.0
            if self._remaining_budget < cost:
                return False
            self._G.add_edge(u, v, travel_time_min=w, line="new")
            self._remaining_budget -= cost
            self._graph_changed = True
            return True

        if action_type == 1:  # remove edge
            if not self._is_valid_remove(u, v):
                return False
            edge_data = self._G.get_edge_data(u, v)
            refund = edge_data.get("travel_time_min", 0) * 2.5
            self._G.remove_edge(u, v)
            self._remaining_budget = min(self._remaining_budget + refund, self.budget)
            self._graph_changed = True
            return True

        if action_type == 2:  # speed up edge
            if not self._G.has_edge(u, v):
                return False
            old_tt = self._G[u][v].get("travel_time_min", 1)
            if old_tt <= MIN_TRAVEL_TIME + 1e-9:
                return False
            cost = FREQ_STEP * 3.0
            if self._remaining_budget < cost:
                return False
            self._G[u][v]["travel_time_min"] = max(MIN_TRAVEL_TIME, old_tt - FREQ_STEP)
            self._remaining_budget -= cost
            self._graph_changed = True
            return True

        if action_type == 3:  # slow down edge
            if not self._G.has_edge(u, v):
                return False
            old_tt = self._G[u][v].get("travel_time_min", 1)
            if old_tt >= MAX_TRAVEL_TIME - 1e-9:
                return False
            self._G[u][v]["travel_time_min"] = min(MAX_TRAVEL_TIME, old_tt + FREQ_STEP)
            self._remaining_budget = min(self._remaining_budget + FREQ_STEP * 1.5, self.budget)
            self._graph_changed = True
            return True

        return False
    
    def step(self, action: np.ndarray):
        """ applies the given action to the environment and returns the new observation, reward, done flags, and info. """
        action_type, u_idx, v_idx = self.decode_action(action)

        u = self.nodes[u_idx]
        v = self.nodes[v_idx]

        # mutate graph for chosen action
        valid = self._apply_action(action_type, u, v)
        self._step_count += 1

        self._hour = (self._hour + 0.5) % 24
        self._current_period = self._get_current_period()

        lengths = self._dijkstra_lengths()
        mean_tt = self._mean_travel_time(lengths)
        reachability = self._reachability(lengths)
        per_line = self._per_line_stats()
        reward = (self._prev_mean_tt - mean_tt) * 20
        self._prev_mean_tt = mean_tt

        # apply penalty for invalid actions - fallback for action masking
        if not valid:
            reward -= 10.0

        # no "winning" state, agent just runs until max_steps
        terminated = False
        truncated = self._step_count >= self.max_steps

        obs = self._observation(mean_tt, reachability, per_line)
        info = self._info(mean_tt)
        info["last_action"] = (action_type, u, v)
        info["action_valid"] = valid

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

        self._ax.set_title(f"Mean TT: {self._mean_travel_time(self._dijkstra_lengths()):.2f}")
        self._ax.axis("off")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if hasattr(self, "_fig"):
            plt.close(self._fig)

    def _dijkstra_lengths(self) -> dict:
        """Compute all-pairs shortest path lengths once (expensive operation)."""
        return dict(
            nx.all_pairs_dijkstra_path_length(self._G, weight="travel_time_min")
        )

    def _per_line_stats(self) -> dict[str, float]:
        """Compute mean edge travel time per transit line."""
        line_totals = {"red": 0.0, "orange": 0.0, "blue": 0.0, "green": 0.0, "new": 0.0}
        line_counts = {"red": 0.0, "orange": 0.0, "blue": 0.0, "green": 0.0, "new": 0.0}
        for u, v, data in self._G.edges(data=True):
            line = data.get("line", "other")
            if line in line_totals:
                line_totals[line] += data.get("travel_time_min", 0)
                line_counts[line] += 1.0
        return {
            line: line_totals[line] / line_counts[line]
            if line_counts[line] > 0 else 0.0
            for line in line_totals
        }

    def _mean_travel_time(self, lengths: dict) -> float:
        """ calculates the average travel time in minutes across all pairs of stations.
            Routes by ride time only; wait time is added once (first edge of the path). """
        period = self.TIME_PERIODS[self._current_period]
        downtown = self.DOWNTOWN
        total, count = 0.0, 0.0

        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if i >= j:
                    continue

                u_downtown = u in downtown
                v_downtown = v in downtown
                u_suburb   = u in self.SUBURBS
                v_suburb   = v in self.SUBURBS

                is_suburb_downtown = (u_suburb and v_downtown) or (u_downtown and v_suburb)

                if self._current_period == "am_rush":
                    if is_suburb_downtown:
                        w = period["weight_downtown"]
                    else:
                        w = period["weight_other"]
                elif self._current_period == "pm_rush":
                    if is_suburb_downtown:
                        w = period["weight_suburb"]
                    else:
                        w = period["weight_other"]
                else:
                    w = period["weight_other"]

                dist = lengths.get(u, {}).get(v, None)
                travel_time = dist if dist is not None else self.disconnect_penalty
                total += w * travel_time
                count += w

        return total / count if count > 0 else 0.0

    def _reachability(self, lengths: dict) -> float:
        """ calculates the fraction of station pairs reachable from each other using Dijkstra's algorithm """
        reachable = 0
        total = 0
        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if i >= j:
                    continue
                total += 1
                if v in lengths.get(u, {}):
                    reachable += 1
        return reachable / total if total > 0 else 0.0
    
    def _observation(self, mean_tt: float, reachability: float, per_line: dict[str, float]) -> np.ndarray:
        """ constructs the observation vector based on the current graph state and baseline mean travel time. """
        norm_tt = float(np.clip(mean_tt / DISCONNECT_PENALTY, 0.0, 1.0))

        n_edges = self._G.number_of_edges()
        max_edges = max(self.N * (self.N - 1) // 2, 1)  # undirected graph
        norm_edges = float(n_edges / max_edges)

        improvement = (
            (self._baseline_mean - mean_tt) / self._baseline_mean
            if self._baseline_mean and self._baseline_mean > 0
            else 0.0
        )
        norm_improvement = float(np.clip(improvement, -1.0, 1.0))

        degrees = [d for _, d in self._G.degree()]
        mean_degree = float(np.mean(degrees)) / self.N if degrees else 0.0

        # per-line normalized mean travel times
        norm_red    = float(np.clip(per_line.get("red",    0) / MAX_TRAVEL_TIME, 0.0, 1.0))
        norm_orange = float(np.clip(per_line.get("orange", 0) / MAX_TRAVEL_TIME, 0.0, 1.0))
        norm_blue   = float(np.clip(per_line.get("blue",   0) / MAX_TRAVEL_TIME, 0.0, 1.0))
        norm_green  = float(np.clip(per_line.get("green",  0) / MAX_TRAVEL_TIME, 0.0, 1.0))
        norm_budget = float(np.clip(self._remaining_budget / self.budget, 0.0, 1.0))

        return np.array(
            [norm_tt, norm_edges, norm_improvement, reachability, mean_degree,
             norm_red, norm_orange, norm_blue, norm_green, norm_budget],
            dtype=np.float32,
        )

    def _info(self, mean_tt: float) -> dict:
        """ constructs the info dictionary with human-readable metrics about the current environment state. """

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
            "current_period":       self._current_period,
            "remaining_budget":     self._remaining_budget,
        }

    def _get_current_period(self) -> str:
        for name, config in self.TIME_PERIODS.items():
            start, end = config["hours"]
            if start < end:
                if start <= self._hour < end:
                    return name
            else:  # overnight wraps midnight
                if self._hour >= start or self._hour < end:
                    return name
        return "midday"
    
if __name__ == "__main__":
    """Loads the graph, runs the gymnasium env_checker."""
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    with open(os.path.join(_PROJECT_ROOT, "outputs", "mbta_graph.pkl"), "rb") as f:
        G = pickle.load(f)

    env = MBTAEnv(G, max_steps=50, render_mode="human")

    print("Running gymnasium env_checker …")
    check_env(env, warn=True)
    print("check_env passed.\n")

    obs, info = env.reset()
    print(f"Baseline mean travel time : {info['mean_travel_time_min']} min")
    print(f"Observation shape         : {obs.shape}")
    print(f"Action space              : {env.action_space}\n")
    print(f"Per-line mean TT          : {env._per_line_stats()}")
    print(f"Observation values: {obs}")
    print(f"budget={info['remaining_budget']}")

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

"""
mbta_env.py  —  Gymnasium environment for MBTA network optimisation

Actions available to the agent:
  0  ADD_EDGE    — connect any two stations with a new edge
  1  REMOVE_EDGE — remove an existing edge
  2  REROUTE     — move one endpoint of an existing edge to a different station

Reward: negative mean travel time for all commuters (start-destination pairs)
- (maximising reward ≡ minimising average commuter travel time).
- Disconnected pairs are penalised with a large constant.

Observation:
- N×N adjacency matrix of minimum travel times (0 = no edge)
- Scalar: current mean travel time (normalised)

How to use:
    import pickle, gymnasium
    from mbta_env import MBTAEnv

    with open("mbta_data/mbta_graph.pkl", "rb") as f:
        G = pickle.load(f)

    env = MBTAEnv(G)
    obs, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

import copy  
import pickle    
from typing import Any 
from matplotlib.pylab import norm
import networkx as nx  
from commuter_model import CommuterPopulation
import numpy as np     
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env


# CONSTANTS

# If two stations can't reach each other at all, we add this penalty to the total travel time so the agent learns to avoid disconnecting the network.
DISCONNECT_PENALTY = 120.0



# When the agent adds a brand new edge it gets assigned a random travel time. 
# TODO: we could make better by basing it on the distance between the two stations.
DEFAULT_EDGE_WEIGHT = 3
MAX_NEW_WEIGHT = 30

# How many steps the agent gets per episode before the game ends.
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
    """

    # what rendering modes we support, currently none.
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_graph: nx.Graph,
        max_steps: int = MAX_STEPS,
        disconnect_penalty: float = DISCONNECT_PENALTY,
        number_of_commuters: int = 20
    ):
        super().__init__()

        self._base_graph = base_graph
        self.nodes: list[str] = sorted(self._base_graph.nodes())
        self.N: int = len(self.nodes)

        # station name to its integer index.
        self._node_idx: dict[str, int] = {n: i for i, n in enumerate(self.nodes)}

        self.max_steps = max_steps
        self.disconnect_penalty = disconnect_penalty

        # working copy of the graph the agent will modify.
        self._G: nx.Graph = None
        self._prev_mean_tt = None

        # action space 
        # - each action is tuple of 4 ints:
        # - [action_type, node_u, node_v, aux_node]
        #
        # action_type = {0, 1, 2}  — which operation to perform (add/remove/reroute)
        # node_u, node_v           — indices of the two stations involved in the action
        # aux_node                 — only used for reroute actions - specifies the new station to connect to
        self.action_space = spaces.MultiDiscrete([3, self.N, self.N, self.N])

        # observation space 
        # The observation is a 1D array with two parts:
        #
        #   Part 1: N×N matrix (flattened to a 1D array of length N*N)
        #     - Each cell [i][j] = direct edge weight between station i and j
        #     - 0.0 means there is no direct connection
        #
        #   Part 2: A single float — the current mean travel time, normalised
        #     to the range [0, 1] by dividing by DISCONNECT_PENALTY
        #
        # Total length = N*N + 1
        # Ex. [ edge_0_0, edge_0_1, ..., edge_N-1_N-1, normalized_mean_travel_time ]
        high = np.full(self.N * self.N + 1, np.float32(DISCONNECT_PENALTY))
        self.observation_space = spaces.Box(
            low=np.zeros_like(high), high=high, dtype=np.float32
        )

        self._step_count: int = 0
        self._baseline_mean: float = None  # set in reset()

        # create the commuter population with the given graph and number of commuters
        self.commuters = CommuterPopulation(base_graph, number_of_commuters)
        self.num_commuters = number_of_commuters

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a fresh episode.
        Restores the graph to its original state and returns the first observation.

        Returns: (observation, info_dict)
        """
        super().reset(seed=seed)

        self._G = copy.deepcopy(self._base_graph)
        self.commuters = CommuterPopulation(self._G, self.num_commuters)
        self._step_count = 0
        self._baseline_mean = self._mean_travel_time()
        self._prev_mean_tt = self._baseline_mean
       

        obs  = self._observation(self._baseline_mean)
        info = self._info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply one action and advance the environment by one timestep.

          1. Apply the action to the graph (add/remove/reroute an edge)
          2. Compute the new mean travel time
          3. Compute the reward (negative mean travel time)
          4. Return the new observation, reward, and info

        Returns: (obs, reward, terminated, truncated, info)
        """
        action_type, u_idx, v_idx, aux_idx = (
            int(action[0]), int(action[1]), int(action[2]), int(action[3])
        )

        # convert integer indices back to station name strings.
        u   = self.nodes[u_idx]
        v   = self.nodes[v_idx]
        aux = self.nodes[aux_idx]

        # mutate the graph for chosen action.
        self._apply_action(action_type, u, v, aux)
        self.commuters.update_graph(self._G)  
        self._step_count += 1

        # recompute the mean travel time.
        mean_tt = self._mean_travel_time()

        # reward = change in travel time compared to the previous step.
        reward = self._prev_mean_tt - mean_tt
        self._prev_mean_tt = mean_tt

        # no "winning" state, agent just runs until max_steps.
        terminated = False
        truncated  = self._step_count >= self.max_steps

        obs  = self._observation(mean_tt)
        info = self._info(mean_tt=mean_tt)
        return obs, reward, terminated, truncated, info

    def render(self):
        # TODO: implement visualization.
        pass

    def _apply_action(self, action_type: int, u: str, v: str, aux: str):
        """
        Modify the graph based on the chosen action.

        Invalid actions are silently ignored:
          - Adding an edge that already exists 
          - Removing an edge that doesn't exist 
          - Rerouting an edge to where it already connects
        """

        if action_type == 0:
            # ADD_EDGE
            if u != v and not self._G.has_edge(u, v):
                w = self.np_random.integers(1, MAX_NEW_WEIGHT + 1)
                self._G.add_edge(u, v, travel_time_min=int(w), line="new")

        elif action_type == 1:
            # REMOVE_EDGE
            if self._G.has_edge(u, v):
                self._G.remove_edge(u, v)

        elif action_type == 2:
            # REROUTE 
            # Move one endpoint of the edge u–v so it now connects u–aux instead.
            if u != aux and self._G.has_edge(u, v) and not self._G.has_edge(u, aux):
                # Copy all the edge attributes (e.g. travel_time_min, line name)
                # from the old edge so the rerouted edge keeps the same metadata.
                # TODO: change to updating the new travel time + make define a new line name for rerouted edges.
                data = dict(self._G[u][v])
                self._G.remove_edge(u, v)
                self._G.add_edge(u, aux, **data)

    def _mean_travel_time(self) -> float:
        return self.commuters.get_mean_commute_time()

    def _observation(self, mean_tt: float) -> np.ndarray:
        """
        The observation has two parts:

        Part 1 — N×N weight matrix (flattened):
          mat[i][j] = travel time of the DIRECT edge between station i and j
          mat[i][j] = 0 if there is no direct connection

        Part 2 — Normalised mean travel time (single float):
          mean_travel_time / DISCONNECT_PENALTY

        Total shape: (N*N + 1,)
        """
        mat = np.zeros((self.N, self.N), dtype=np.float32)

        # fill in edge weights from the current graph
        for u, v, d in self._G.edges(data=True):
            i, j = self._node_idx[u], self._node_idx[v]
            # get the travel time or use DEFAULT_EDGE_WEIGHT if missing
            # TODO: make sure none are missing and remove the default fallback
            w = float(d.get("travel_time_min", DEFAULT_EDGE_WEIGHT))
            mat[i, j] = w
            mat[j, i] = w

        norm = np.float32(mean_tt / self.disconnect_penalty)
        return np.append(mat.flatten(), norm)

    def _info(self, mean_tt: float | None = None) -> dict:
        """
        Dictionary to get info about the current state of the environment.
        Fields:
          mean_travel_time_min : average commute time in minutes
          improvement_pct      : % improvement vs the baseline (start of episode)
          n_edges              : current number of connections in the network
          step                 : how many steps have been taken this episode
        """
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


# test to make sure it runs without errors and the observation/action spaces look correct.
if __name__ == "__main__":
    # Load the MBTA graph
    with open("mbta_data/mbta_graph.pkl", "rb") as f:
        G = pickle.load(f)

    env = MBTAEnv(G, max_steps=50)

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
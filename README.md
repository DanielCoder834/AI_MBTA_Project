# AI_MBTA_Project

network.py:
- networkx representatiom of MBTA using data from Google

mbta_env:
- reinforcement learning enviorment for the MBTA

    Actions available to the agent:
    0  ADD_EDGE    — connect any two stations with a new edge
    1  REMOVE_EDGE — remove an existing edge

    Reward: negative mean travel time for all start-destination pairs
    - (maximising reward ≡ minimising average travel time).
    - Disconnected pairs are penalised with a large constant.

    Observation:
    A 1D array of 5 normalized scalar features describing current MBTA network state:
    1: normalized mean travel time [0, 1]
        - mean travel time divided by DISCONNECT_PENALTY
    2: normalized edge count [0, 1]
        - current number of edges divided by N^2
        - how dense the network currently is
    3: normalized improvement [-1, 1]
        - percent improvement relative to baseline network
        - (baseline_mean − current_mean) / baseline_mean
    4: reachability ratio [0, 1]
        - fraction of origin–destination pairs that remain connected
    5: normalized mean node degree [0, 1]
        - average node degree divided by number of stations N
        - overall network connectivity level

    Example observation:
    [ normalized_mean_travel_time,
    normalized_edge_density,
    normalized_improvement,
    reachability_ratio,
    normalized_mean_degree ]
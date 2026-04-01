from typing import List, Tuple
import networkx as nx
import math
import numpy as np

DEFAULT_EDGE_WEIGHT = 3
DISCONNECT_PENALTY = 500.0

# Commuter for RL MBTA environment
# represents a commuter(commuter) commuting from home to work(start to destination)
class Commuter:
    # initializes the commuter
    def __init__(self, home_station: str, work_station: str, commuter_id: int):
        self.home_station = home_station
        self.work_station = work_station
        self.commuter_id = commuter_id

    # Represents the start and end of the route where the commuter wants to reach(home and work)
    def get_start_destination_pair(self) -> Tuple[str, str]:
        return (self.home_station, self.work_station)


# Manages multiple commuters at once, each has a start location (home) and work destination
# Initialized commuters, pulls all stations and calls generate_commuters
# Random number generator is used to assign home and work pairs
class CommuterPopulation:
    """Manages a population of commuters, each with a home and work station."""
    def __init__(self, mbta_graph: nx.Graph, random_seed: int = 42):
        self.graph = mbta_graph
        self.commuters: List[Commuter] = []
        # Gets all stations from the graph (stations are nodes)
        self.station_list = list(mbta_graph.nodes())
        self.rng = np.random.default_rng(random_seed)
        # creates commuters
        self.generate_commuters()

    # Generates commuters by finding all home work pairs
    def generate_commuters(self):
        commuter_id = 0
        pairs = []

        for home in self.station_list:
            for work in self.station_list:
                if home != work:
                    pairs.append((home, work))

        self.rng.shuffle(pairs)

        for home, work in pairs:
            self.commuters.append(Commuter(home, work, commuter_id))
            commuter_id += 1

    # updates graph for commute time calculations
    def update_graph(self, new_graph: nx.Graph):
        # replaces current graph with new one
        self.graph = new_graph

    # computes distance between two locations using their latitude and longitude
    # uses a* with haversine heruistic to find shortest path for commuters
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        # Earth radius in km
        r = 6371.0
        # Haversine formula
        # degrees to radians
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        # square of half the chord length 
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        # Computes the angular distance 
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c
    
    
    # Heuristic formula for A*
    def heuristic(self, u, v):
        lon1, lat1 = self.graph.nodes[u]["lon"], self.graph.nodes[u]["lat"]
        lon2, lat2 = self.graph.nodes[v]["lon"], self.graph.nodes[v]["lat"]

        distance_km = self.haversine(lat1, lon1, lat2, lon2)

        speed_kmh = 55  # TODO: find average speed of MBTA trains in km/h

        # Coverts travel time to minutes
        return (distance_km / speed_kmh) * 60

    # calculates the mean commute time of all commuters based on graph
    def get_mean_commute_time(self):
        total = 0.0
        n = len(self.commuters)

        # loops through each commuter and calculates the travel time of shortest path
        for c in self.commuters:
            try:
                # uses A* to compute travel time between home and work
                dist = nx.astar_path_length(
                    self.graph,
                    c.home_station,
                    c.work_station,
                    heuristic=self.heuristic,
                    weight="travel_time_min"
                )
                total += dist
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                total += DISCONNECT_PENALTY

        return total / n
    
    # computes travel time based on distance
    def edge_weight_from_distance(self, u: str, v: str) -> float:
        try:
            lat1, lon1 = self.graph.nodes[u]["lat"], self.graph.nodes[u]["lon"]
            lat2, lon2 = self.graph.nodes[v]["lat"], self.graph.nodes[v]["lon"]
        except KeyError:
            return DEFAULT_EDGE_WEIGHT
        km = self.haversine(lat1, lon1, lat2, lon2)
        return float(max(1.0, round((km / 30.0) * 60.0, 1)))
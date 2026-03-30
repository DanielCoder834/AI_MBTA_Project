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


    def generate_commuters(self):
        commuter_id = 0
        for home in self.station_list:
            for work in self.station_list:
                if home != work:
                    self.commuters.append(Commuter(home, work, commuter_id))
                    commuter_id += 1

    def update_graph(self, new_graph: nx.Graph):
        """Updates the graph used for commute time calculations."""
        self.graph = new_graph

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
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
    
    def heuristic(self, u, v):
        lon1, lat1 = self.graph.nodes[u]["lon"], self.graph.nodes[u]["lat"]
        lon2, lat2 = self.graph.nodes[v]["lon"], self.graph.nodes[v]["lat"]

        distance_km = self.haversine(lat1, lon1, lat2, lon2)

        speed_kmh = 55  # TODO: find average speed of MBTA trains in km/h

        return (distance_km / speed_kmh) * 60

    def get_mean_commute_time(self):
        """Calculates the mean commute time for all commuters based on the current graph."""
        total = 0.0
        n = len(self.commuters)

        for c in self.commuters:
            try:
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
    def edge_weight_from_distance(self, u: str, v: str) -> float:
        try:
            lat1, lon1 = self.graph.nodes[u]["lat"], self.graph.nodes[u]["lon"]
            lat2, lon2 = self.graph.nodes[v]["lat"], self.graph.nodes[v]["lon"]
        except KeyError:
            return DEFAULT_EDGE_WEIGHT
        km = self.haversine(lat1, lon1, lat2, lon2)
        return float(max(1.0, round((km / 30.0) * 60.0, 1)))
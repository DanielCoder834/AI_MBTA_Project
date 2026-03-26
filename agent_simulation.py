import random
from typing import Dict, List, Tuple
import networkx as nx

# Commuter for RL MBTA environment
# represents a commuter(commuter) commuting from home to work(start to destination)
class Commuter:
    # initializes the commuter
    def __init__(self, home_station: str, work_station: str, commuter_id: int):
        self.home_station = home_station
        self.work_station = work_station
        self.commuter_id = commuter_id
        # commuter always starts at home
        self.current_station = home_station 

    # resets commuter's location to back home
    def reset_location(self):
        self.current_station = self.home_station

    # checks if the commuter is at work 
    def is_at_work(self) -> bool:
        return self.current_station == self.work_station

    # Represents the start and end of the route where the commuter wants to reach(home and work)
    def get_start_destination_pair(self) -> Tuple[str, str]:
        return (self.home_station, self.work_station)


# Manages multiple commuters at once, each has a start location (home) and work destination
class CommuterPopulation:

    # Initializes the population
    # mbta_graph: transit network
    # num_commuters: # of commuters
    def initialization(self, mbta_graph: nx.DiGraph, num_commuters: int = 100):
        self.graph = mbta_graph
        self.num_commuters = num_commuters
        self.commuters: List[Commuter] = []
        # Gets all stations from the graph(stations are nodes)
        self.station_list = list(mbta_graph.nodes())

        # creates commuters
        self.generate_commuters()

    # function that creates commuters
    def generate_commuters(self):
        for i in range(self.num_commuters):
            # chooses a random home location
            home = random.choice(self.station_list)
            # chooses a random work destination location
            work = random.choice(self.station_list)

            # Picks a different location if work location is same as home
            while work == home:
                work = random.choice(self.station_list)

            # single commuter is created
            commuter = Commuter(home, work, i)
            
            # single commuter is added to list
            self.commuters.append(commuter)

    # resents commuters locations to home
    def reset_location(self):
        for commuter in self.commuters:
            commuter.reset_location()

    # gives how many agenets are at different stops
    def get_commuter_count_at_locations(self):
        location_counts = {station: 0 for station in self.station_list}
        for commuter in self.commuters:
            location_counts[commuter.current_station] += 1
        return location_counts

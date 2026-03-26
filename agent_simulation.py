import random
from typing import Dict, List, Tuple
import networkx as nx

# Agent for RL MBTA environment
# represents a person(agent) commuting from home to work(start to destination)
class Agent:
    # initializes the agent
    def __init__(self, home_station: str, work_station: str, agent_id: int):
        self.home_station = home_station
        self.work_station = work_station
        self.agent_id = agent_id
        # agent always starts at home
        self.current_station = home_station 

    # resets agent's location to back home
    def reset_location(self):
        self.current_station = self.home_station

    # checks if the agent is at work 
    def is_at_work(self):
        return self.current_station == self.work_station

    # Represents the start and end of the route where the agent wants to reach(home and work)
    # Returns a tuple
    def get_start_destination_pair(self):
        return (self.home_station, self.work_station)


# Manages multiple agents at once, each has a start location (home) and work destination
class AgentPopulation:

    # Initializes the population
    # mbta_graph: transit network
    # num_agents: # of commuters
    def initialization(self, mbta_graph: nx.DiGraph, num_agents: int = 100):
        self.graph = mbta_graph
        self.num_agents = num_agents
        self.agents: List[Agent] = []
        # Gets all stations from the graph(stations are nodes)
        self.station_list = list(mbta_graph.nodes())

        # creates agents
        self.generate_agents()

    # function that creates agents
    def generate_agents(self):
        for i in range(self.num_agents):
            # chooses a random home location
            home = random.choice(self.station_list)
            # chooses a random work destination location
            work = random.choice(self.station_list)

            # Picks a different location if work location is same as home
            while work == home:
                work = random.choice(self.station_list)

            # single agent is created
            agent = Agent(home, work, i)
            
            # single agent is added to list
            self.agents.append(agent)

    # resents agents locations to home
    def reset_location(self):
        for agent in self.agents:
            agent.reset_location()

    # gives how many agenets are at different stops
    def get_agent_count_at_locations(self):
        location_counts = {station: 0 for station in self.station_list}
        for agent in self.agents:
            location_counts[agent.current_station] += 1
        return location_counts

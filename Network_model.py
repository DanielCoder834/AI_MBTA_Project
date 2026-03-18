"""
Network_model.py: Build a Boston MBTA rail network graph for reinforcement learning.

Nodes: parent stations (not every platform)
Edges: direct train movements between consecutive stations on real MBTA route patterns

Data source: MBTA V3 API

Setup: 
pip install requests networkx
export MBTA_API_KEY="fa68e5761f124bf19e38b137c6c78f2e"
"""

import os
import time
import math
import requests
import networkx as nx


MBTA_BASE = "https://api-v3.mbta.com"


class MBTANetwork:
    def __init__(self, api_key=None, request_delay=0.6, max_retries=6):
        self.api_key = api_key or os.getenv("MBTA_API_KEY")
        self.request_delay = request_delay
        self.max_retries = max_retries

        self.graph = nx.DiGraph()
        self.routes = {}
        self.cache = {}

        # raw stop_id -> canonical station id
        self.stop_to_station = {}

        # canonical station id -> info
        self.station_info = {}

    def get_json(self, url):
        if url in self.cache:
            return self.cache[url]

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        last_error = None

        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, headers=headers, timeout=30)

                if resp.status_code == 429:
                    wait = min(2 ** attempt, 20)
                    print(f"Rate limited. Waiting {wait}s: {url}")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if "errors" in data:
                    raise RuntimeError(f"API returned errors: {data['errors']}")

                self.cache[url] = data
                time.sleep(self.request_delay)
                return data

            except Exception as e:
                last_error = e
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(f"Failed after retries for {url}: {last_error}")

    def load_routes(self):
        url = f"{MBTA_BASE}/routes?filter[type]=0,1"
        data = self.get_json(url)

        for route in data["data"]:
            route_id = route["id"]
            long_name = route["attributes"]["long_name"]
            self.routes[route_id] = long_name

        print(f"Loaded {len(self.routes)} routes")

    def load_stations(self):
        url = f"{MBTA_BASE}/stops?filter[route_type]=0,1"
        data = self.get_json(url)

        for stop in data["data"]:
            stop_id = stop["id"]
            attrs = stop["attributes"]

            parent_station = attrs.get("parent_station")
            location_type = attrs.get("location_type")

            # If a stop has a parent station, collapse to that, otherwise keep itself
            station_id = parent_station if parent_station else stop_id
            self.stop_to_station[stop_id] = station_id

            # Only add canonical station once
            if station_id not in self.station_info:
                self.station_info[station_id] = {
                    "name": attrs["name"],
                    "lat": attrs["latitude"],
                    "lon": attrs["longitude"],
                    "location_type": location_type,
                }

                self.graph.add_node(
                    station_id,
                    name=attrs["name"],
                    lat=attrs["latitude"],
                    lon=attrs["longitude"],
                    location_type=location_type,
                )

        print(f"Loaded {self.graph.number_of_nodes()} stations")

    def canonical_station_id_from_stop_obj(self, stop_obj):
        stop_id = stop_obj["id"]
        attrs = stop_obj["attributes"]
        parent_station = attrs.get("parent_station")
        return parent_station if parent_station else stop_id

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

    def estimate_travel_time(self, station_a, station_b, speed_kmh=28):
        lat1 = self.graph.nodes[station_a]["lat"]
        lon1 = self.graph.nodes[station_a]["lon"]
        lat2 = self.graph.nodes[station_b]["lat"]
        lon2 = self.graph.nodes[station_b]["lon"]

        distance_km = self.haversine(lat1, lon1, lat2, lon2)
        minutes = (distance_km / speed_kmh) * 60
        return max(2.0, round(minutes, 2))

    def load_connections(self):
        seen_edges = set()

        for route_id, route_name in self.routes.items():
            print(f"Processing route: {route_name}")

            try:
                rp_url = f"{MBTA_BASE}/route_patterns?filter[route]={route_id}"
                rp_data = self.get_json(rp_url)
            except Exception as e:
                print(f"Skipping route {route_id}: {e}")
                continue

            for pattern in rp_data["data"]:
                pattern_id = pattern["id"]

                try:
                    trip_url = (
                        f"{MBTA_BASE}/trips"
                        f"?filter[route_pattern]={pattern_id}"
                        f"&include=stops"
                        f"&page[limit]=1"
                    )
                    trip_data = self.get_json(trip_url)
                except Exception as e:
                    print(f"Skipping pattern {pattern_id}: {e}")
                    continue

                included = trip_data.get("included", [])
                stop_objs = [obj for obj in included if obj["type"] == "stop"]

                if not stop_objs:
                    continue

                station_sequence = []
                for stop in stop_objs:
                    sid = self.canonical_station_id_from_stop_obj(stop)

                    if sid not in self.graph.nodes:
                        continue

                    # remove consecutive duplicates
                    if not station_sequence or station_sequence[-1] != sid:
                        station_sequence.append(sid)

                if len(station_sequence) < 2:
                    continue

                for i in range(len(station_sequence) - 1):
                    s1 = station_sequence[i]
                    s2 = station_sequence[i + 1]

                    if s1 == s2:
                        continue

                    edge_key = (s1, s2, route_id)
                    if edge_key in seen_edges:
                        continue

                    seen_edges.add(edge_key)

                    travel_time = self.estimate_travel_time(s1, s2)

                    self.graph.add_edge(
                        s1,
                        s2,
                        weight=travel_time,
                        travel_time=travel_time,
                        route_id=route_id,
                        route_name=route_name,
                        route_pattern=pattern_id,
                    )

        print(f"Built {self.graph.number_of_edges()} directed edges")

    def remove_orphan_nodes(self):
        orphan_nodes = [
            n for n in self.graph.nodes
            if self.graph.in_degree(n) == 0 and self.graph.out_degree(n) == 0
        ]
        self.graph.remove_nodes_from(orphan_nodes)
        print(f"Removed {len(orphan_nodes)} orphan nodes")

    def build(self):
        print("Building MBTA network...")
        self.load_routes()
        self.load_stations()
        self.load_connections()
        self.remove_orphan_nodes()
        print("Done.")
        return self.graph

    def get_neighbors(self, station_id):
        return list(self.graph.successors(station_id))

    def get_station_name(self, station_id):
        return self.graph.nodes[station_id]["name"]

    def get_edge_weight(self, a, b):
        return self.graph[a][b]["weight"]

    def summary(self):
        print("\n=== NETWORK SUMMARY ===")
        print("Stations:", self.graph.number_of_nodes())
        print("Edges:", self.graph.number_of_edges())

    def print_sample_neighbors(self, station_id, limit=10):
        print(f"\nStation: {self.get_station_name(station_id)} ({station_id})")
        print("Neighbors:")
        for neighbor in self.get_neighbors(station_id)[:limit]:
            edge = self.graph[station_id][neighbor]
            print(
                f" -> {self.get_station_name(neighbor)} "
                f"({neighbor}) | {edge['route_name']} | {edge['travel_time']} min"
            )

    def print_duplicate_station_names(self, limit=20):
        name_to_ids = {}
        for node_id, attrs in self.graph.nodes(data=True):
            name = attrs["name"]
            name_to_ids.setdefault(name, []).append(node_id)

        dupes = {name: ids for name, ids in name_to_ids.items() if len(ids) > 1}

        print(f"\nDuplicate station names: {len(dupes)}")
        count = 0
        for name, ids in dupes.items():
            print(name, ids)
            count += 1
            if count >= limit:
                break


if __name__ == "__main__":
    network = MBTANetwork()
    G = network.build()
    network.summary()

    sample_station = next(iter(G.nodes))
    network.print_sample_neighbors(sample_station)

    # Debug duplicate names
    network.print_duplicate_station_names()
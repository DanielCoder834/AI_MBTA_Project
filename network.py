"""
MBTA NetworkX graph.
"""

import csv
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import pickle

EDGES_FILE = "./mbta_data/t_edges.txt"
STOPS_FILE = "./mbta_data/stops.txt"
OUTPUT_PNG = "./mbta_data/mbta_map.png"

# store edge data: (source_station_id, destination_station_id, travel_time_minutes, line_color)
edges_raw = []
station_ids = set()

with open(EDGES_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        src, dst, time, color = line.split(",")
        edges_raw.append((src, dst, int(time), color))
        station_ids.update([src, dst])

# store station info: stop_id -> {name, lat, lon}
station_info = {}
with open(STOPS_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["stop_id"] in station_ids:
            if not row["stop_lat"] or not row["stop_lon"]:
                print(f"Warning: Missing coordinates for {row['stop_id']} ({row['stop_name']})")
                continue
            station_info[row["stop_id"]] = {
                "name": row["stop_name"],
                "lat":  float(row["stop_lat"]),
                "lon":  float(row["stop_lon"]),
            }

# some green line stops missing coordinates add manually 
MANUAL_COORDS = {
    "place-buwst": (-71.1228, 42.3503),
    "place-stplb": (-71.1192, 42.3501),
    "place-plsgr": (-71.1160, 42.3499),
}

# create the graph and add nodes with (name, lat, lon)
G = nx.MultiGraph()
for id in station_ids:
    # get station info from the stops file or use manual coordinates
    info = station_info.get(id, {})
    lon, lat = MANUAL_COORDS.get(id, (info.get("lon"), info.get("lat")))
    G.add_node(id, name=info.get("name", id), lat=lat, lon=lon)

# track which lines intersect for transfers
node_lines: dict[str, set] = defaultdict(set)
for src, dst, t, color in edges_raw:
    node_lines[src].add(color)
    node_lines[dst].add(color)
nx.set_node_attributes(G, {n: sorted(v) for n, v in node_lines.items()}, "lines")

for src, dst, travel_time, color in edges_raw:
    G.add_edge(src, dst, line=color, travel_time_min=travel_time)

# save graph
with open("./mbta_data/mbta_graph.pkl", "wb") as f:
    pickle.dump(G, f)

# node positions for plotting
pos = {
    n: (d["lon"], d["lat"])
    for n, d in G.nodes(data=True)
    if d.get("lon") and d.get("lat")
}

LINE_COLORS = {
    "red": "#DA291C",
    "orange": "#ED8B00",
    "blue": "#003DA5",
    "green": "#00843D",
}

# draw graph
fig, ax = plt.subplots(figsize=(15, 10))

# draw edges
for lc in ["green", "red", "orange", "blue"]:
    elist = [(u, v) for u, v, d in G.edges(data=True) if d["line"] == lc]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=elist,
        edge_color=LINE_COLORS[lc],
        width=3.0,
        alpha=0.92,
        ax=ax,
    )

# draw connections vs regular stations
connection = [n for n in pos if len(G.nodes[n].get("lines", [])) > 1]
regular     = [n for n in pos if len(G.nodes[n].get("lines", [])) == 1]

# regular = coloured by line
def single_color(n):
    ls = G.nodes[n].get("lines", [])
    return LINE_COLORS[ls[0]] if ls else "#888"

nx.draw_networkx_nodes(G, pos, nodelist=regular,
                       node_color=[single_color(n) for n in regular],
                       node_size=30, ax=ax,
                       linewidths=0.5, edgecolors="#cccccc")

# connection = black
nx.draw_networkx_nodes(G, pos, nodelist=connection,
                       node_color="black",
                       node_size=40, ax=ax,
                       linewidths=0.5)

# label connection + end stops of lines
degree_one  = {n for n in G.nodes() if G.degree(n) == 1 and n in pos}
label_nodes = set(connection) | degree_one
labels = {n: G.nodes[n]["name"] for n in label_nodes}

nx.draw_networkx_labels(
    G, pos,
    labels=labels,
    font_size=6.2,
    font_color="#0f0101",
    font_weight="bold",
    ax=ax,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.15",  ec="none", alpha=0.65),
)

# legend 
legend_items = [
    Line2D([0], [0], color=LINE_COLORS[c], linewidth=3, label=f"{c.capitalize()} Line")
    for c in ["red", "orange", "blue", "green"]
] + [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
           markeredgecolor="#aaa", markersize=9, label="connection"),
]
ax.legend(handles=legend_items, loc="upper left",
          fontsize=11, edgecolor="#444",
          labelcolor="black", framealpha=0.95,
          handlelength=1.8, handletextpad=0.8)


ax.set_title("MBTA Rapid Transit Network", fontsize=22,
             color="black", fontweight="bold", pad=18)
ax.axis("off")
plt.tight_layout()
plt.show()
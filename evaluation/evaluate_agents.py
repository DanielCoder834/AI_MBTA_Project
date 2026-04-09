"""
evaluate_agents.py

Evaluates the implemented DQN agent on the MBTA environment.

For each agent:
- loads the saved model
- runs one evaluation episode
- prints summary metrics
- optionally renders the network changes
- prints a comparison table

Run:
    python evaluate_agents.py
"""

import os
import sys
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.mbta_env import MBTAEnv
from agents.dqn_agent import DQNAgent

# from sb3_contrib import MaskablePPO
# from sb3_contrib.common.maskable.utils import get_action_masks


# CHANGE THESE TO EVALUATE DIFFERENT RUNS
DQN_VERSION        = 2     # v1: budget=5000, add_cost=w*2, no remove refund
                           # v2: budget=1000, add_cost=w*5, remove refund=tt*2.5,
                           #     freq actions modify travel_time (±0.5min), speed_up cost=1.5, slow_down refund=0.75
DQN_EPISODES       = 100
DQN_LR             = 0.0001
DQN_EPSILON_DECAY  = 0.995
DQN_BUFFER         = 5000
DQN_TARGET_UPDATE  = 200

# PPO_TIMESTEPS = 30720


# DONT CHANGE
GRAPH_PATH = os.path.join(PROJECT_ROOT, "outputs", "mbta_graph.pkl")
DQN_RUN_TAG = f"dqn_v{DQN_VERSION}_ep{DQN_EPISODES}_lr{DQN_LR}_ed{DQN_EPSILON_DECAY}_buf{DQN_BUFFER}_tgt{DQN_TARGET_UPDATE}"
DQN_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", f"{DQN_RUN_TAG}.pt")
# PPO_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", f"maskable_mbta_ppo_{PPO_TIMESTEPS}.zip")
MAX_STEPS = 50
RENDER = True


def evaluate_dqn(graph, render=False):
    """Evaluate implemented DQN agent for one episode."""
    # initialize env
    env = MBTAEnv(graph, max_steps=MAX_STEPS, render=render, budget=5000.0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # load trained agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(DQN_MODEL_PATH)
     # fully greedy at evaluation time, disable exploration
    agent.epsilon = 0.0

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    # run evaluation episode
    while not done:
        # mask invalid actions
        valid_mask = env.action_masks()
        # select best valid action
        action = agent.select_action(obs, valid_mask=valid_mask)
        # apply action
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward

        if render:
            env.render()

        done = terminated or truncated
    
    # capture the final modified graph before closing
    final_graph = env._G.copy()

    # performance metrics
    summary = {
        "agent": "DQN",
        "final_mean_tt": info["mean_travel_time_min"],
        "improvement_pct": info["improvement_pct"],
        "n_edges": info["n_edges"],
        "steps": info["step"],
        "total_reward": total_reward,
    }

    env.close()
    return summary, final_graph


# def evaluate_ppo(graph, render=False):
#     """Evaluate imported MaskablePPO agent for one episode."""
#     # initialize env
#     env = MBTAEnv(graph, max_steps=MAX_STEPS, render=render,  budget=500.0)
#     # load trained PPO policy
#     model = MaskablePPO.load(PPO_MODEL_PATH)
#
#     obs, info = env.reset()
#     done = False
#     total_reward = 0.0
#
#     while not done:
#         # retrieve valid action mask from environment
#         masks = get_action_masks(env)
#         # select deterministic action from trained policy
#         action, _ = model.predict(obs, action_masks=masks, deterministic=True)
#
#         obs, reward, terminated, truncated, info = env.step(action)
#         total_reward += reward
#
#         if render:
#             env.render()
#
#         done = terminated or truncated
#
#     # performance metrics
#     summary = {
#         "agent": "MaskablePPO",
#         "final_mean_tt": info["mean_travel_time_min"],
#         "improvement_pct": info["improvement_pct"],
#         "n_edges": info["n_edges"],
#         "steps": info["step"],
#         "total_reward": total_reward,
#     }
#
#     env.close()
#     return summary


LINE_COLORS = {
    "red": "#DA291C",
    "orange": "#ED8B00",
    "blue": "#003DA5",
    "green": "#00843D",
    "new": "#9B59B6",
}


def save_final_graph(baseline_graph, final_graph, agent_name, run_tag):
    """Save the final modified graph as a pickle and a visualization PNG.

    Compares against baseline_graph to highlight:
    - removed edges (dotted red)
    - new edges added by agent (dashed purple)
    - sped-up edges / lower travel time (thicker, green glow)
    - slowed-down edges / higher travel time (thinner, red glow)
    """
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    graphs_dir = os.path.join(outputs_dir, "graphs")

    # save pickle
    pkl_path = os.path.join(graphs_dir, f"{run_tag}_final_graph.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(final_graph, f)
    print(f"\nFinal graph saved to {pkl_path}")

    # build node positions from baseline (has all nodes including removed-edge endpoints)
    pos = {
        n: (d["lon"], d["lat"])
        for n, d in baseline_graph.nodes(data=True)
        if d.get("lon") is not None and d.get("lat") is not None
    }

    # diff edges against baseline
    baseline_edges = set(baseline_graph.edges())
    final_edges = set(final_graph.edges())
    removed_edges = baseline_edges - final_edges
    new_edges = [(u, v) for u, v, d in final_graph.edges(data=True) if d.get("line") == "new"]

    # categorise surviving original edges by travel-time change
    sped_up = []     # travel_time decreased (faster service)
    slowed_down = [] # travel_time increased (slower service)
    freq_unchanged = []  # no change
    edge_tt_labels = {}

    for u, v, d in final_graph.edges(data=True):
        if d.get("line") == "new":
            continue
        # compare against baseline travel time
        if baseline_graph.has_edge(u, v):
            base_tt = baseline_graph[u][v].get("travel_time_min", 0)
        else:
            continue
        final_tt = d.get("travel_time_min", 0)
        delta = final_tt - base_tt
        if abs(delta) < 1e-9:
            freq_unchanged.append((u, v))
        elif delta < 0:
            sped_up.append((u, v))
            edge_tt_labels[(u, v)] = f"{delta:+.1f}m"
        else:
            slowed_down.append((u, v))
            edge_tt_labels[(u, v)] = f"{delta:+.1f}m"

    fig, ax = plt.subplots(figsize=(15, 10))

    # 1) removed edges — dotted red, drawn on baseline positions
    if removed_edges:
        removed_list = list(removed_edges)
        nx.draw_networkx_edges(
            baseline_graph, pos, edgelist=removed_list,
            edge_color="#CC0000", width=2.5, alpha=0.6,
            style="dotted", ax=ax,
        )

    # 2) unchanged original edges — normal style per line colour
    for lc in ["green", "red", "orange", "blue"]:
        elist = [(u, v) for u, v in freq_unchanged
                 if final_graph[u][v].get("line") == lc]
        if elist:
            nx.draw_networkx_edges(
                final_graph, pos, edgelist=elist,
                edge_color=LINE_COLORS[lc], width=3.0, alpha=0.92, ax=ax,
            )

    # 3) sped-up edges — thicker with green glow
    for lc in ["green", "red", "orange", "blue"]:
        elist = [(u, v) for u, v in sped_up
                 if final_graph[u][v].get("line") == lc]
        if elist:
            # green glow behind
            nx.draw_networkx_edges(
                final_graph, pos, edgelist=elist,
                edge_color="#2ECC71", width=6.0, alpha=0.35, ax=ax,
            )
            # line colour on top
            nx.draw_networkx_edges(
                final_graph, pos, edgelist=elist,
                edge_color=LINE_COLORS[lc], width=4.0, alpha=0.92, ax=ax,
            )

    # 4) slowed-down edges — thinner with red glow
    for lc in ["green", "red", "orange", "blue"]:
        elist = [(u, v) for u, v in slowed_down
                 if final_graph[u][v].get("line") == lc]
        if elist:
            # red glow behind
            nx.draw_networkx_edges(
                final_graph, pos, edgelist=elist,
                edge_color="#E74C3C", width=5.0, alpha=0.35, ax=ax,
            )
            # line colour on top, thinner
            nx.draw_networkx_edges(
                final_graph, pos, edgelist=elist,
                edge_color=LINE_COLORS[lc], width=1.5, alpha=0.92, ax=ax,
            )

    # 5) new agent-added edges — dashed purple
    if new_edges:
        nx.draw_networkx_edges(
            final_graph, pos, edgelist=new_edges,
            edge_color=LINE_COLORS["new"], width=3.0, alpha=0.92,
            style="dashed", ax=ax,
        )

    # 6) travel-time delta labels on changed edges
    if edge_tt_labels:
        nx.draw_networkx_edge_labels(
            final_graph, pos, edge_labels=edge_tt_labels,
            font_size=5.5, font_color="#333333", ax=ax,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )

    # draw connection vs regular stations
    connection = [n for n in pos if len(final_graph.nodes[n].get("lines", [])) > 1]
    regular = [n for n in pos if len(final_graph.nodes[n].get("lines", [])) <= 1]

    def single_color(n):
        ls = final_graph.nodes[n].get("lines", [])
        return LINE_COLORS.get(ls[0], "#888") if ls else "#888"

    nx.draw_networkx_nodes(final_graph, pos, nodelist=regular,
                           node_color=[single_color(n) for n in regular],
                           node_size=30, ax=ax, linewidths=0.5, edgecolors="#cccccc")
    nx.draw_networkx_nodes(final_graph, pos, nodelist=connection,
                           node_color="black", node_size=40, ax=ax, linewidths=0.5)

    # labels for connections and end stops
    degree_one = {n for n in final_graph.nodes() if final_graph.degree(n) == 1 and n in pos}
    label_nodes = set(connection) | degree_one
    labels = {n: final_graph.nodes[n].get("name", n) for n in label_nodes}

    nx.draw_networkx_labels(
        final_graph, pos, labels=labels, font_size=6.2, font_color="#0f0101",
        font_weight="bold", ax=ax, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.15", ec="none", alpha=0.65),
    )

    # legend
    legend_items = [
        Line2D([0], [0], color=LINE_COLORS[c], linewidth=3, label=f"{c.capitalize()} Line")
        for c in ["red", "orange", "blue", "green"]
    ] + [
        Line2D([0], [0], color=LINE_COLORS["new"], linewidth=3, linestyle="dashed",
               label="New edge (agent)"),
        Line2D([0], [0], color="#CC0000", linewidth=2.5, linestyle="dotted",
               label="Removed edge"),
        Line2D([0], [0], color="#2ECC71", linewidth=6, alpha=0.5,
               label="Sped up (lower travel time)"),
        Line2D([0], [0], color="#E74C3C", linewidth=5, alpha=0.5,
               label="Slowed down (higher travel time)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
               markeredgecolor="#aaa", markersize=9, label="Connection"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9,
              edgecolor="#444", labelcolor="black", framealpha=0.95,
              handlelength=1.8, handletextpad=0.8)

    ax.set_title(f"MBTA Network After {agent_name} Optimization ({run_tag})",
                 fontsize=18, color="black", fontweight="bold", pad=18)
    ax.axis("off")
    plt.tight_layout()

    png_path = os.path.join(graphs_dir, f"{run_tag}_final_graph.png")
    plt.savefig(png_path, dpi=150)
    plt.show()
    print(f"Final graph visualization saved to {png_path}")

    # print change summary
    print(f"\n--- Graph Changes ({agent_name}) ---")
    print(f"  Edges removed:          {len(removed_edges)}")
    print(f"  New edges added:        {len(new_edges)}")
    print(f"  Sped up:                {len(sped_up)} edges")
    print(f"  Slowed down:            {len(slowed_down)} edges")
    print(f"  Unchanged:              {len(freq_unchanged)} edges")


def print_summary(summary):
    """Print one agent's evaluation summary."""
    print(f"\n=== {summary['agent']} Evaluation ===")
    print(f"Final mean travel time: {summary['final_mean_tt']:.2f} min")
    print(f"Improvement from baseline: {summary['improvement_pct']:+.2f}%")
    print(f"Final number of edges: {summary['n_edges']}")
    print(f"Episode steps: {summary['steps']}")
    print(f"Total reward: {summary['total_reward']:.2f}")


def print_comparison(results):
    """Print side by side comparison of all evaluated agents."""
    print("\n" + "=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(
        f"{'Agent':<15}"
        f"{'Mean TT (min)':<18}"
        f"{'Improve %':<14}"
        f"{'Edges':<10}"
        f"{'Steps':<8}"
        f"{'Reward':<10}"
    )
    print("-" * 72)

    for result in results:
        print(
            f"{result['agent']:<15}"
            f"{result['final_mean_tt']:<18.2f}"
            f"{result['improvement_pct']:<14.2f}"
            f"{result['n_edges']:<10}"
            f"{result['steps']:<8}"
            f"{result['total_reward']:<10.2f}"
        )

    print("=" * 72)

    # determine best performing agents
    best_tt = min(results, key=lambda r: r["final_mean_tt"])
    best_reward = max(results, key=lambda r: r["total_reward"])

    print(f"Lowest final mean travel time: {best_tt['agent']}")
    print(f"Highest total reward: {best_reward['agent']}")


class Tee:
    """Write to both a file and the original stdout."""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    """
    Main evaluation entry point
    Loads MBTA graph and evaluates each available trained agent
    """

    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(
            f"Could not find graph file: {GRAPH_PATH}\n"
            "Run python network.py first."
        )

    # save all text output to a log file
    log_path = os.path.join(PROJECT_ROOT, "outputs", "logs", f"{DQN_RUN_TAG}_eval.txt")
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(log_file, original_stdout)

    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    results = []

    if os.path.exists(DQN_MODEL_PATH):
        dqn_result, dqn_final_graph = evaluate_dqn(graph, render=RENDER)
        print_summary(dqn_result)
        results.append(dqn_result)
        save_final_graph(graph, dqn_final_graph, "DQN", DQN_RUN_TAG)
    else:
        print(f"\nSkipping DQN: model file not found at {DQN_MODEL_PATH}")

    # if os.path.exists(PPO_MODEL_PATH):
    #     ppo_result = evaluate_ppo(graph, render=RENDER)
    #     print_summary(ppo_result)
    #     results.append(ppo_result)
    # else:
    #     print(f"\nSkipping PPO: model file not found at {PPO_MODEL_PATH} "
    #           f"or sb3_contrib is not installed.")

    print_comparison(results)

    # restore stdout and close log
    sys.stdout = original_stdout
    log_file.close()
    print(f"\nEvaluation log saved to {log_path}")


if __name__ == "__main__":
    main()
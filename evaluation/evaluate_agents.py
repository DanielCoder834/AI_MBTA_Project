"""
evaluate_agents.py

Evaluates and compares the implemented DQN agent and the
imported MaskablePPO agent on the MBTA environment.

For each agent:
- loads the saved model
- runs one evaluation episode
- prints summary metrics
- optionally renders the network changes
- prints a comparison table

Run (creates mbta_graph.pkl, maskable_mbta_ppo.zip):
    pip install stable-baselines3 sb3-contrib
    python evaluate_agents.py
"""

import os
import sys
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.mbta_env import MBTAEnv
from agents.dqn_agent import DQNAgent

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


# CHANGE THESE TO EVALUATE DIFFERENT RUNS
DQN_EPISODES = 615
PPO_TIMESTEPS = 30720


# DONT CHANGE
GRAPH_PATH = os.path.join(PROJECT_ROOT, "outputs", "mbta_graph.pkl")
DQN_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", f"dqn_mbta_{DQN_EPISODES}.pt")
PPO_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", f"maskable_mbta_ppo_{PPO_TIMESTEPS}.zip")
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
    return summary


def evaluate_ppo(graph, render=False):
    """Evaluate imported MaskablePPO agent for one episode."""
    # initialize env
    env = MBTAEnv(graph, max_steps=MAX_STEPS, render=render,  budget=5000.0)
    # load trained PPO policy
    model = MaskablePPO.load(PPO_MODEL_PATH)

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # retrieve valid action mask from environment
        masks = get_action_masks(env)
        # select deterministic action from trained policy
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render:
            env.render()

        done = terminated or truncated

    # performance metrics
    summary = {
        "agent": "MaskablePPO",
        "final_mean_tt": info["mean_travel_time_min"],
        "improvement_pct": info["improvement_pct"],
        "n_edges": info["n_edges"],
        "steps": info["step"],
        "total_reward": total_reward,
    }

    env.close()
    return summary


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

    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    results = []

    if os.path.exists(DQN_MODEL_PATH):
        dqn_result = evaluate_dqn(graph, render=RENDER)
        print_summary(dqn_result)
        results.append(dqn_result)
    else:
        print(f"\nSkipping DQN: model file not found at {DQN_MODEL_PATH}")

    if os.path.exists(PPO_MODEL_PATH):
        ppo_result = evaluate_ppo(graph, render=RENDER)
        print_summary(ppo_result)
        results.append(ppo_result)
    else:
        print(f"\nSkipping PPO: model file not found at {PPO_MODEL_PATH} "
              f"or sb3_contrib is not installed.")

    print_comparison(results)
    
if __name__ == "__main__":
    main()
# Presentation Plan (~2 minutes)

---

## Slide 1: Motivation & Problem (~1 min)

**Title:** "Optimizing Boston's Transit Network with Reinforcement Learning"

**Visual:** MBTA network map (generated from `network.py`) — colored by line, labeled transfer stations

**Talking points (keep to 3-4):**
- MBTA serves 1.3M+ daily riders across 4 rapid-transit lines — transit planners must decide where to add routes, adjust service frequency, and allocate limited budgets
- These decisions are interdependent: improving one corridor shifts travel patterns across the whole network — hard to reason about manually
- We frame this as a reinforcement learning problem: an agent interacts with a graph representation of the MBTA network and learns which modifications reduce average commuter travel time

**Transition to problem setup (same slide or quick sub-slide):**

**Visual:** Simple diagram showing the RL loop — Agent observes network state, takes an action (add route / remove route / adjust wait time), environment returns new state + reward

**Talking points:**
- The network is a weighted graph — stations are nodes, routes are edges, weights are ride times from real MBTA data
- The agent can: add a new connection, remove an underused one, or increase/decrease service frequency on a route (modeled as wait time)
- Reward = reduction in average commuter travel time after each action
- Key constraints: fixed budget, network must stay connected, rush-hour travel is weighted more heavily

---

## Slide 2: Methods (~1 min)

**Title:** "Two Approaches: Value-Based vs Policy-Based RL"

**Visual:** Side-by-side comparison table (not code — conceptual)

| | DQN (Value-Based) | PPO (Policy-Based) |
|---|---|---|
| **Core idea** | Learns "how good is each action?" then picks the best | Learns "what should I do?" directly as a probability distribution |
| **Why this approach** | Well-suited for discrete action spaces; can evaluate all actions at once | Stable training via constrained policy updates; handles large action spaces gracefully |
| **Exploration** | Starts random, gradually becomes greedy (exploration vs exploitation tradeoff) | Built into the policy — naturally explores less-certain actions |
| **Action masking** | Invalid actions get lowest value | Invalid actions get zero probability |

**Talking points:**
- We compare two foundational RL approaches to understand which is better suited for network optimization
- **DQN**: learns a value function Q(state, action) — "if I'm in this network state and take this action, how much will travel time improve?" Picks the highest-value valid action. Trained with experience replay (learns from shuffled past experiences to break correlations)
- **PPO**: learns a policy directly — outputs a probability distribution over actions. Updates are clipped to prevent the policy from changing too drastically between steps, which stabilizes training
- Both agents see the same state and action space — the comparison isolates the effect of the learning algorithm

**Assumptions & Limitations (mention briefly):**
- We assume travel times are static (no real-time congestion modeling)
- Wait time is uniform across a line — in reality it varies by station and time of day
- The action space scales as O(N^2) with stations — becomes expensive for larger networks
- Currently uses random commuter pairs — real ridership data would make the optimization more meaningful

---

## Suggested Visuals Summary

| Slide | Visual | Purpose |
|-------|--------|---------|
| 1 | MBTA network map (colored by line) | Ground the audience — this is a real system |
| 1 | RL loop diagram (state -> action -> reward -> state) | Explain the problem framing quickly |
| 2 | DQN vs PPO comparison table | Convey method differences without math |
| 2 | (Optional) One-sentence limitation callouts | Shows awareness of scope |

---

## Things to Avoid
- No code, class names, or hyperparameters on slides
- No math formulas unless someone asks during Q&A
- Don't explain replay buffers or target networks in detail — just say "learns from past experience" and "stabilizes training"
- Don't list all 10 observation features — say "the agent sees a compact summary of network health"

## Tips for Delivery
- Slide 1: spend more time on the "why" than the "what" — the audience should understand why this problem is hard before you explain the solution
- Slide 2: the comparison table does the heavy lifting — talk through it row by row rather than reading bullet points
- Keep transitions smooth: "So the problem is hard because decisions are interdependent — that's exactly what RL is designed for. Here's how we approached it."

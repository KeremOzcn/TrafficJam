"""
Monte Carlo Tree Search (MCTS) for Traffic Signal Control.

At each decision point the agent builds a shallow search tree:
  - Root node  = current simulation state snapshot
  - Children   = result of applying action 0 (HOLD) or 1 (SWITCH)
  - Rollout    = simulate forward with a random policy for ROLLOUT_DEPTH steps
  - Backprop   = average wait-time improvement (lower is better)
  - Selection  = UCB1 (Upper Confidence Bound) balances exploration/exploitation

After N_SIMULATIONS rollouts the action with the best average score is chosen.

UCB1 formula:
    UCB1(node) = exploit + C * sqrt(ln(parent_visits) / node_visits)
    where exploit = mean reward, C = exploration constant
"""

import copy
import math
import random

from TrafficSimulator import Simulation
from TrafficSimulator.Setups.two_way_intersection import (
    two_way_intersection_setup, VEHICLE_RATE, PATHS, INTERSECTIONS_DICT
)

# ── MCTS hyper-parameters ─────────────────────────────────────────────────────
N_SIMULATIONS  = 12    # rollouts per decision point
ROLLOUT_DEPTH  = 3     # actions simulated in each rollout
UCB_C          = 1.41  # exploration constant (≈ sqrt(2))
ACTION_SPACE   = [0, 1]
MAX_GEN        = 50


# ── Simulation snapshot helper ────────────────────────────────────────────────

def _clone_sim(sim: Simulation) -> Simulation:
    """
    Create a lightweight deep-copy of the simulation for lookahead.
    Mirrors the pattern used in alt_state.Gstate.
    """
    clone = Simulation(sim.max_gen)
    clone.t                    = sim.t
    clone.n_vehicles_generated = sim.n_vehicles_generated
    clone.n_vehicles_on_map    = sim.n_vehicles_on_map
    clone._inbound_roads       = sim.inbound_roads
    clone._outbound_roads      = sim.outbound_roads
    clone._non_empty_roads     = copy.deepcopy(sim.non_empty_roads)
    clone.collision_detected   = sim.collision_detected
    clone._waiting_times_sum   = sim._waiting_times_sum
    clone.roads                = copy.deepcopy(sim.roads)
    clone.add_generator(VEHICLE_RATE, PATHS)
    clone.traffic_signals      = [clone.roads[0].traffic_signal]
    clone.add_intersections(INTERSECTIONS_DICT)
    return clone


# ── MCTS Node ─────────────────────────────────────────────────────────────────

class MCTSNode:
    """
    Represents one node in the MCTS tree.

    Attributes:
        action   : the action taken to reach this node (None for root)
        parent   : parent MCTSNode (None for root)
        children : list of child MCTSNodes
        visits   : number of times this node was visited
        value    : cumulative reward (higher = better throughput)
    """

    def __init__(self, action=None, parent=None):
        self.action   = action
        self.parent   = parent
        self.children = []
        self.visits   = 0
        self.value    = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(ACTION_SPACE)

    def best_child(self, c: float = UCB_C) -> 'MCTSNode':
        """Select child with highest UCB1 score."""
        def ucb1(node):
            exploit = node.value / (node.visits + 1e-9)
            explore = c * math.sqrt(math.log(self.visits + 1) /
                                    (node.visits + 1e-9))
            return exploit + explore
        return max(self.children, key=ucb1)

    def expand(self) -> 'MCTSNode':
        """Add one untried child node."""
        tried = {child.action for child in self.children}
        untried = [a for a in ACTION_SPACE if a not in tried]
        action = random.choice(untried)
        child = MCTSNode(action=action, parent=self)
        self.children.append(child)
        return child

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value  += reward


# ── MCTS algorithm ────────────────────────────────────────────────────────────

def _rollout(sim_snapshot: Simulation, depth: int) -> float:
    """
    Random-policy rollout from a simulation snapshot.
    Returns throughput reward: vehicles cleared from inbound roads.
    """
    sim = _clone_sim(sim_snapshot)
    before = sim.n_vehicles_on_map
    for _ in range(depth):
        if sim.completed:
            break
        sim.run(random.choice(ACTION_SPACE))
    after = sim.n_vehicles_on_map
    # Reward = vehicles that left the map (higher = better)
    reward = before - after
    # Penalty for collision
    if sim.collision_detected:
        reward -= 20
    return float(reward)


def mcts_select_action(sim: Simulation) -> int:
    """
    Run N_SIMULATIONS MCTS rollouts from the current state and return
    the best action (0=HOLD, 1=SWITCH).

    Algorithm (standard MCTS loop):
        1. Selection   — traverse tree by UCB1 until a leaf
        2. Expansion   — add one untried child
        3. Simulation  — random rollout for ROLLOUT_DEPTH steps
        4. Backprop    — propagate reward up to root
    """
    root = MCTSNode()

    for _ in range(N_SIMULATIONS):
        node = root
        sim_copy = _clone_sim(sim)

        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            sim_copy.run(node.action)
            if sim_copy.completed:
                break

        # 2. Expansion
        if not sim_copy.completed and not node.is_fully_expanded():
            node = node.expand()
            sim_copy.run(node.action)

        # 3. Rollout
        reward = _rollout(sim_copy, ROLLOUT_DEPTH)

        # 4. Backpropagation
        while node is not None:
            node.update(reward)
            node = node.parent

    # Choose action of child with most visits (most robust)
    if not root.children:
        return 0
    best = max(root.children, key=lambda n: n.visits)
    return best.action


# ── Episode runner ─────────────────────────────────────────────────────────────

def mcts_sim_run(render: bool = False, episode: int = 0,
                 total_episodes: int = 0, collisions: int = 0) -> float:
    """
    Run one full episode with MCTS signal control.

    Returns:
        average wait time (float), or -1 if a collision occurred
    """
    sim = two_way_intersection_setup(MAX_GEN)
    sim.dashboard_info.update({
        'method': 'mcts',
        'episode': episode,
        'total_episodes': total_episodes,
        'collisions': collisions,
    })
    if render:
        sim.init_gui()

    while not (sim.gui_closed or sim.completed):
        action = mcts_select_action(sim)
        sim.run(action)

    if sim.gui_closed:
        exit()
    if sim.collision_detected:
        return -1
    return sim.current_average_wait_time


def mcts_cycle(n_episodes: int, render: bool = False) -> None:
    """Run MCTS for n_episodes and print results."""
    print(f"\n -- Running MCTS agent for {n_episodes} episodes --")
    print(f"    (N_SIMULATIONS={N_SIMULATIONS}, ROLLOUT_DEPTH={ROLLOUT_DEPTH}, "
          f"UCB_C={UCB_C})\n")

    total_wait, collisions = 0.0, 0
    for ep in range(1, n_episodes + 1):
        result = mcts_sim_run(render=render, episode=ep,
                              total_episodes=n_episodes,
                              collisions=collisions)
        if result == -1:
            print(f"Episode {ep} - Collisions detected")
            collisions += 1
        else:
            total_wait += result
            print(f"Episode {ep} - Wait time: {result:.2f}")

    n_completed = n_episodes - collisions
    print(f"\n -- Results after {n_episodes} episodes: --")
    if n_completed:
        print(f"Average wait time per completed episode: "
              f"{total_wait / n_completed:.2f}")
    print(f"Average collisions per episode: {collisions / n_episodes:.2f}")

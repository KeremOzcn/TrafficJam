"""
Comparison runner — executes every method for n_episodes and produces:
  1. A formatted results table printed to the terminal
  2. A bar-chart PNG saved to results/comparison_<timestamp>.png
"""

import os
import time
import statistics

from ReinforcementLearning import Environment, QLearningAgent
from ReinforcementLearning.q_learning_utils import get_q_values
from DefaultCycles.default_cycles_utils import fixed_cycle_action, longest_queue_action
from Logic.logic_utils import logic_action
from MathModel.math_utils import math_action
from MathModel.network_utils import network_math_action
from MathModel.grid_network_utils import grid_math_action
from Search.search import sim_run as search_sim_run
from Search.mcts import mcts_sim_run
from TrafficSimulator.Setups.two_way_intersection import two_way_intersection_setup
from ReinforcementLearning.network_environment import NetworkEnvironment
from ReinforcementLearning.grid_environment import GridEnvironment

# Q-Learning hyperparameters (must match training)
ALPHA, EPSILON, DISCOUNT = 0.1, 0.0, 0.6   # epsilon=0 for evaluation
Q_TABLE_PATH = "ReinforcementLearning/Traffic_q_values_10000.txt"


def _run_env_method(method_name, action_fn, n_episodes):
    """Run an Environment-based method and return (wait_times, collision_count)."""
    env = Environment()
    wait_times = []
    collisions = 0
    for ep in range(1, n_episodes + 1):
        state = env.reset(render=False)
        env.sim.dashboard_info.update({
            'method': method_name,
            'episode': ep,
            'total_episodes': n_episodes,
            'collisions': collisions,
        })
        collision_detected = 0
        done = False
        while not done:
            action = action_fn(env.sim, state)
            state, _, done, truncated = env.step(action)
            if truncated:
                break
            collision_detected += env.sim.collision_detected
        if collision_detected:
            collisions += 1
        else:
            wait_times.append(env.sim.current_average_wait_time)
    return wait_times, collisions


def compare_all(n_episodes: int) -> None:
    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # Load Q-table once
    q_agent = QLearningAgent(ALPHA, EPSILON, DISCOUNT, [0, 1])
    q_agent.q_values = eval(get_q_values(Q_TABLE_PATH))
    q_agent.epsilon = 0.0  # greedy evaluation

    methods = [
        ('fc',        'Fixed Cycle',         lambda: _run_env_method('fc', fixed_cycle_action, n_episodes)),
        ('lqf',       'Longest Queue First', lambda: _run_env_method('lqf', longest_queue_action, n_episodes)),
        ('logic',     'Propositional Logic', lambda: _run_env_method('logic', logic_action, n_episodes)),
        ('math',      'Math Model',          lambda: _run_env_method('math', math_action, n_episodes)),
        ('qlearning', 'Q-Learning',          lambda: _run_ql(q_agent, n_episodes)),
        ('search',    'Genetic Algorithm',   lambda: _run_search(n_episodes)),
        ('mcts',      'MCTS',                lambda: _run_mcts(n_episodes)),
        ('network',   'Network Math (2-int)',lambda: _run_network(n_episodes)),
        ('grid',      'Grid Math (2x2)',     lambda: _run_grid(n_episodes)),
    ]

    results = {}
    for key, label, runner in methods:
        print(f'  Running {label:<25} ...', end=' ', flush=True)
        t0 = time.time()
        wait_times, collisions = runner()
        elapsed = time.time() - t0
        results[key] = {
            'label':      label,
            'wait_times': wait_times,
            'collisions': collisions,
            'elapsed':    elapsed,
        }
        avg = statistics.mean(wait_times) if wait_times else float('inf')
        print(f'avg={avg:.2f}s  col={collisions}  ({elapsed:.0f}s)')

    _print_table(results, n_episodes)
    _save_chart(results, n_episodes, timestamp)


# ── Sub-runners ───────────────────────────────────────────────────────────────

def _run_ql(agent, n_episodes):
    env = Environment()
    wait_times = []
    collisions = 0
    for ep in range(1, n_episodes + 1):
        state = env.reset(render=False)
        env.sim.dashboard_info.update({
            'method': 'qlearning', 'episode': ep,
            'total_episodes': n_episodes, 'collisions': collisions,
        })
        collision_detected = 0
        done = False
        while not done:
            action = agent.get_action(state)
            state, _, done, truncated = env.step(action)
            if truncated:
                break
            collision_detected += env.sim.collision_detected
        if collision_detected:
            collisions += 1
        else:
            wait_times.append(env.sim.current_average_wait_time)
    return wait_times, collisions


def _run_search(n_episodes):
    wait_times = []
    collisions = 0
    for ep in range(1, n_episodes + 1):
        result = search_sim_run(render=False, episode=ep,
                                total_episodes=n_episodes,
                                collisions=collisions)
        if result == -1:
            collisions += 1
        else:
            wait_times.append(result)
    return wait_times, collisions


def _run_mcts(n_episodes):
    wait_times = []
    collisions = 0
    for ep in range(1, n_episodes + 1):
        result = mcts_sim_run(render=False, episode=ep,
                              total_episodes=n_episodes,
                              collisions=collisions)
        if result == -1:
            collisions += 1
        else:
            wait_times.append(result)
    return wait_times, collisions


def _run_network(n_episodes):
    env = NetworkEnvironment()
    wait_times = []
    collisions = 0
    for ep in range(1, n_episodes + 1):
        state = env.reset(render=False)
        env.sim.dashboard_info.update({
            'method': 'network', 'episode': ep,
            'total_episodes': n_episodes, 'collisions': collisions,
        })
        collision_detected = 0
        done = False
        while not done:
            action = network_math_action(env.sim, state)
            state, _, done, truncated = env.step(action)
            if truncated:
                break
            collision_detected += env.sim.collision_detected
        if collision_detected:
            collisions += 1
        else:
            wait_times.append(env.sim.current_average_wait_time)
    return wait_times, collisions


def _run_grid(n_episodes):
    env = GridEnvironment()
    wait_times = []
    collisions = 0
    for ep in range(1, n_episodes + 1):
        state = env.reset(render=False)
        env.sim.dashboard_info.update({
            'method': 'grid', 'episode': ep,
            'total_episodes': n_episodes, 'collisions': collisions,
        })
        collision_detected = 0
        done = False
        while not done:
            action = grid_math_action(env.sim, state)
            state, _, done, truncated = env.step(action)
            if truncated:
                break
            collision_detected += env.sim.collision_detected
        if collision_detected:
            collisions += 1
        else:
            wait_times.append(env.sim.current_average_wait_time)
    return wait_times, collisions


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_table(results: dict, n_episodes: int) -> None:
    print()
    print('=' * 74)
    print(f'  COMPARISON RESULTS  ({n_episodes} episodes each)')
    print('=' * 74)
    header = f"{'Method':<25} {'Avg Wait':>9} {'Best':>7} {'Worst':>7} {'Std':>7} {'Collisions':>11}"
    print(header)
    print('-' * 74)

    rows = []
    for key, data in results.items():
        wt = data['wait_times']
        col = data['collisions']
        if wt:
            avg   = statistics.mean(wt)
            best  = min(wt)
            worst = max(wt)
            std   = statistics.stdev(wt) if len(wt) > 1 else 0.0
        else:
            avg = best = worst = std = float('inf')
        rows.append((avg, data['label'], avg, best, worst, std, col))

    rows.sort(key=lambda r: r[0])  # sort by avg wait time
    for rank, (_, label, avg, best, worst, std, col) in enumerate(rows, 1):
        star = ' *' if rank == 1 else '  '
        print(f"  {label:<23}{star} {avg:>8.2f}s {best:>7.2f} {worst:>7.2f} {std:>7.2f} {col:>10}")

    print('=' * 74)
    print("  * Best performing method")
    print()


def _save_chart(results: dict, n_episodes: int, timestamp: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        labels  = [d['label'] for d in results.values()]
        avgs    = [statistics.mean(d['wait_times']) if d['wait_times'] else 0
                   for d in results.values()]
        stds    = [statistics.stdev(d['wait_times']) if len(d['wait_times']) > 1 else 0
                   for d in results.values()]
        colors  = ['#4e9af1', '#f18f4e', '#4ef18f', '#f14e6e',
                   '#b44ef1', '#f1e44e', '#4ef1e4', '#e44e4e', '#4ef1a1']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'AI Traffic Controller — Method Comparison ({n_episodes} episodes)',
                     fontsize=14, fontweight='bold')

        # ── Bar chart: average wait time ──────────────────────────────────
        ax = axes[0]
        x = np.arange(len(labels))
        bars = ax.bar(x, avgs, color=colors[:len(labels)],
                      yerr=stds, capsize=5, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Average Wait Time (s)')
        ax.set_title('Average Wait Time per Method')
        ax.set_ylim(0, max(avgs) * 1.3 if avgs else 5)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=8)
        best_idx = int(np.argmin(avgs))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

        # ── Episode history line chart ────────────────────────────────────
        ax2 = axes[1]
        for (key, data), color in zip(results.items(), colors):
            wt = data['wait_times']
            if wt:
                ax2.plot(range(1, len(wt) + 1), wt,
                         marker='o', markersize=3, label=data['label'],
                         color=color, linewidth=1.5)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Wait Time (s)')
        ax2.set_title('Wait Time per Episode')
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f'results/comparison_{timestamp}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Chart saved: {path}')
    except ImportError:
        print('  (matplotlib not installed — skipping chart)')

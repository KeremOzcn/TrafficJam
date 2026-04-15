from ReinforcementLearning import Environment
from MathModel.probability_model import (
    congestion_score, mm1_expected_queue, mm1_expected_wait,
    LAMBDA, MU, print_summary as print_prob_summary
)
from MathModel.linear_algebra import (
    linear_algebra_action, total_congestion, print_state_analysis
)

MIN_CYCLE_TIME = 10  # Minimum seconds between signal switches


def math_action(sim, state: tuple) -> bool:
    """
    Decide signal action by combining probability scores (Poisson/M/M/1)
    with linear algebra (weighted congestion vector + norm-based dominant
    direction detection).

    Steps:
        1. Enforce minimum cycle time guard.
        2. Compute probability-based congestion scores for each direction.
        3. Use those scores as weights for the linear algebra decision.
        4. Return SWITCH or HOLD.

    Args:
        sim:   current Simulation object
        state: (signal_state, q1, q2, junction_busy)

    Returns:
        True  → switch signal
        False → hold current signal
    """
    traffic_signal = sim.traffic_signals[0]
    time_elapsed = sim.t - traffic_signal.prev_update_time >= MIN_CYCLE_TIME
    if not time_elapsed:
        return False

    _, q1, q2, _ = state
    prob_weights = congestion_score(q1, q2, LAMBDA, MU)

    switch = linear_algebra_action(state, prob_weights)
    if switch:
        traffic_signal.prev_update_time = sim.t
    return switch


def math_cycle(n_episodes: int, render: bool = False) -> None:
    """
    Run the Math-based agent for n_episodes and report results.

    Prints:
        - Probability model summary (M/M/1 queuing theory)
        - Per-episode wait time or collision count
        - Final averages

    Args:
        n_episodes: number of evaluation episodes
        render:     open pygame window if True
    """
    print_prob_summary(LAMBDA, MU)

    print(f"\n -- Running Math agent for {n_episodes} episodes --")
    print("    (Poisson arrivals + M/M/1 queuing + L2-norm weighted decisions)\n")

    environment = Environment()
    total_wait_time = 0
    total_collisions = 0

    # Track per-episode congestion norms for the report
    episode_l2_norms = []

    for episode in range(1, n_episodes + 1):
        state = environment.reset(render)
        collision_detected = 0
        done = False
        step_norms = []

        environment.sim.dashboard_info.update({
            'method': 'math',
            'episode': episode,
            'total_episodes': n_episodes,
            'collisions': total_collisions,
        })

        while not done:
            action = math_action(environment.sim, state)
            state, reward, done, truncated = environment.step(action)
            if truncated:
                exit()
            collision_detected += environment.sim.collision_detected
            norms = total_congestion(state)
            step_norms.append(norms['l2'])

        avg_norm = sum(step_norms) / len(step_norms) if step_norms else 0.0
        episode_l2_norms.append(avg_norm)

        if collision_detected:
            print(f"Episode {episode} - Collisions: {int(collision_detected)}")
            total_collisions += 1
            environment.sim.dashboard_info['collisions'] = total_collisions
        else:
            wait_time = environment.sim.current_average_wait_time
            total_wait_time += wait_time
            print(f"Episode {episode} - Wait time: {wait_time:.2f}  "
                  f"(avg L2 congestion norm: {avg_norm:.2f})")
            if environment.sim._gui:
                environment.sim._gui.record_episode(wait_time)

    n_completed = n_episodes - total_collisions
    overall_avg_norm = (sum(episode_l2_norms) / len(episode_l2_norms)
                        if episode_l2_norms else 0.0)

    print(f"\n -- Results after {n_episodes} episodes: --")
    print(f"Average wait time per completed episode: "
          f"{total_wait_time / n_completed:.2f}")
    print(f"Average collisions per episode:          "
          f"{total_collisions / n_episodes:.2f}")
    print(f"Average L2 congestion norm:              "
          f"{overall_avg_norm:.4f}")
    print(f"\nM/M/1 theoretical E[W] = {mm1_expected_wait(LAMBDA, MU):.2f} sec  "
          f"| E[N] = {mm1_expected_queue(LAMBDA, MU):.2f} vehicles")

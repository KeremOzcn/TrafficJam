from ReinforcementLearning import Environment
from Logic.inference_engine import infer, truth_table

MIN_CYCLE_TIME = 5  # Minimum seconds between signal switches


def logic_action(sim, state: tuple) -> bool:
    """
    Decide whether to switch the traffic signal using propositional logic
    inference (Modus Ponens).

    A minimum cycle time guard prevents rapid oscillation: the inference
    engine is only consulted after MIN_CYCLE_TIME seconds have elapsed
    since the last switch.

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

    switch = infer(state)
    if switch:
        traffic_signal.prev_update_time = sim.t
    return switch


def logic_cycle(n_episodes: int, render: bool = False) -> None:
    """
    Run the Logic-based agent for n_episodes and report results.

    Args:
        n_episodes: number of evaluation episodes
        render:     open pygame window if True
    """
    print(f"\n -- Running Logic agent for {n_episodes} episodes --")
    print("\nPropositional Logic Truth Table:")
    print(truth_table())
    print()

    environment = Environment()
    total_wait_time = 0
    total_collisions = 0

    for episode in range(1, n_episodes + 1):
        state = environment.reset(render)
        collision_detected = 0
        done = False

        environment.sim.dashboard_info.update({
            'method': 'logic',
            'episode': episode,
            'total_episodes': n_episodes,
            'collisions': total_collisions,
        })

        while not done:
            action = logic_action(environment.sim, state)
            state, reward, done, truncated = environment.step(action)
            if truncated:
                exit()
            collision_detected += environment.sim.collision_detected

        if collision_detected:
            print(f"Episode {episode} - Collisions: {int(collision_detected)}")
            total_collisions += 1
            environment.sim.dashboard_info['collisions'] = total_collisions
        else:
            wait_time = environment.sim.current_average_wait_time
            total_wait_time += wait_time
            print(f"Episode {episode} - Wait time: {wait_time:.2f}")
            if environment.sim._gui:
                environment.sim._gui.record_episode(wait_time)

    n_completed = n_episodes - total_collisions
    print(f"\n -- Results after {n_episodes} episodes: --")
    print(f"Average wait time per completed episode: {total_wait_time / n_completed:.2f}")
    print(f"Average collisions per episode: {total_collisions / n_episodes:.2f}")

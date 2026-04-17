"""
Network Math Agent — runner for the two-intersection corridor.

Decision logic per intersection:
  1. Enforce a minimum cycle time guard (no rapid oscillation).
  2. Compute probability-based congestion scores using EFFECTIVE arrival
     rates from the Jackson Network model (not just raw external rates).
  3. Use those scores as weights for the linear-algebra dominant-direction
     decision (same as the single-intersection math model).
  4. Additionally, check the propagation prediction: if the upstream
     intersection is releasing a surge of through-vehicles, the downstream
     intersection pre-emptively switches to green (green-wave coordination).

This is an extension of math_utils.py to the multi-intersection case.
"""

from MathModel.network_model import (
    network_congestion_scores,
    predict_downstream_surge,
    print_network_summary,
    VEHICLE_RATE,
    MU,
)
from MathModel.linear_algebra import linear_algebra_action
from ReinforcementLearning.network_environment import NetworkEnvironment

MIN_CYCLE_TIME = 10   # seconds between signal switches


def network_math_action(sim, state: tuple) -> int:
    """
    Decide actions for BOTH intersections and return an encoded action.

    Uses Jackson-Network effective arrival rates for congestion scoring
    and the propagation prediction for downstream green-wave preparation.

    Args:
        sim:   Simulation object (access to traffic_signals and sim.t)
        state: (sig_L, q1_L, q2_L, busy_L, sig_R, q1_R, q2_R, busy_R)

    Returns:
        0 hold both | 1 switch left | 2 switch right | 3 switch both
    """
    sig_L, q1_L, q2_L, busy_L = state[0], state[1], state[2], state[3]
    sig_R, q1_R, q2_R, busy_R = state[4], state[5], state[6], state[7]

    sig_obj_L = sim.traffic_signals[0]
    sig_obj_R = sim.traffic_signals[1]

    # Per-intersection congestion weights from Jackson model
    (w_L, w_R) = network_congestion_scores(state)

    switch_L = False
    switch_R = False

    # ── Left intersection decision ────────────────────────────────────────
    if not busy_L:
        elapsed_L = sim.t - sig_obj_L.prev_update_time >= MIN_CYCLE_TIME
        if elapsed_L:
            switch_L = linear_algebra_action(
                (sig_L, q1_L, q2_L, busy_L), w_L)
            if switch_L:
                sig_obj_L.prev_update_time = sim.t

    # ── Right intersection decision (with propagation prediction) ─────────
    if not busy_R:
        elapsed_R = sim.t - sig_obj_R.prev_update_time >= MIN_CYCLE_TIME
        if elapsed_R:
            # Standard math decision using effective-rate weights
            switch_R_base = linear_algebra_action(
                (sig_R, q1_R, q2_R, busy_R), w_R)

            # Green-wave override: is a vehicle surge arriving from the left?
            surge = predict_downstream_surge(
                upstream_q_ew=q1_L,
                upstream_ew_green=bool(sig_L),
            )
            if surge['recommend_green'] and not sig_R:
                # Downstream E-W is currently red but a surge is coming →
                # switch to green to avoid vehicles piling up on the connector
                switch_R = True
            else:
                switch_R = switch_R_base

            if switch_R:
                sig_obj_R.prev_update_time = sim.t

    # Encode action
    if switch_L and switch_R:
        return 3
    if switch_L:
        return 1
    if switch_R:
        return 2
    return 0


def network_math_cycle(n_episodes: int, render: bool = False) -> None:
    """
    Run the Network Math agent for n_episodes and report results.

    Prints:
      - Jackson Network model summary (effective rates, E[N], E[W])
      - Per-episode wait time or collision
      - Final averages including per-intersection breakdown
    """
    print_network_summary(VEHICLE_RATE, MU)
    print(f"\n -- Running Network Math agent for {n_episodes} episodes --")
    print("    (Jackson Network + L2-norm weighted decisions + green-wave)\n")

    env = NetworkEnvironment()
    total_wait  = 0.0
    collisions  = 0

    for ep in range(1, n_episodes + 1):
        state = env.reset(render)
        done  = False
        collision = False

        env.sim.dashboard_info.update({
            'method':         'network',
            'episode':        ep,
            'total_episodes': n_episodes,
            'collisions':     collisions,
        })

        while not done:
            action = network_math_action(env.sim, state)
            state, _, done, truncated = env.step(action)
            if truncated:
                exit()
            collision = env.sim.collision_detected

        if collision:
            print(f"Episode {ep} - Collision detected")
            collisions += 1
            env.sim.dashboard_info['collisions'] = collisions
        else:
            wt = env.sim.current_average_wait_time
            total_wait += wt
            print(f"Episode {ep} - Wait time: {wt:.2f}")
            if env.sim._gui:
                env.sim._gui.record_episode(wt)

    n_completed = n_episodes - collisions
    print(f"\n -- Results after {n_episodes} episodes: --")
    if n_completed:
        print(f"Average wait time per completed episode: "
              f"{total_wait / n_completed:.2f}")
    print(f"Average collisions per episode: {collisions / n_episodes:.2f}")

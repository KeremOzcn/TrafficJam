"""
Grid Network Agent -- runner for the 2x2 intersection grid.

Decision logic per intersection (decentralized, runs independently):
  1. Minimum cycle-time guard (no rapid oscillation).
  2. Compute congestion scores using EFFECTIVE arrival rates from the
     4-node Jackson Network model.
  3. Use scores as weights for the linear-algebra dominant-direction decision.
  4. Green-wave override: if either upstream neighbour is releasing a surge
     of vehicles heading here, pre-emptively switch to the correct phase.

Green-wave topology:
  TL receives surges from  TR (westbound via C_TR_TL)
                       and BL (northbound via C_BL_TL)
  TR receives surges from  TL (eastbound  via C_TL_TR)
                       and BR (northbound via C_BR_TR)
  BL receives surges from  TL (southbound via C_TL_BL)
                       and BR (westbound  via C_BR_BL)
  BR receives surges from  BL (eastbound  via C_BL_BR)
                       and TR (southbound via C_TR_BR)
"""

from MathModel.grid_network_model import (
    grid_congestion_scores,
    predict_surge,
    print_grid_summary,
    VEHICLE_RATE,
    MU,
)
from MathModel.linear_algebra import linear_algebra_action
from ReinforcementLearning.grid_environment import GridEnvironment, encode_action

MIN_CYCLE_TIME = 10   # seconds between signal switches


def _elapsed(sim, sig_idx: int) -> bool:
    return sim.t - sim.traffic_signals[sig_idx].prev_update_time >= MIN_CYCLE_TIME


def _green_wave_override(sig_cur: bool,
                         ew_surge: bool, ns_surge: bool) -> bool:
    """
    Should we force a switch to accommodate an incoming surge?

    sig_cur = True  -> E-W currently green
    ew_surge        -> surge arriving on E-W connector (needs E-W green)
    ns_surge        -> surge arriving on N-S connector (needs N-S green)

    Returns True if we should switch now to prepare the right phase.
    """
    if ew_surge and not sig_cur:
        return True   # E-W surge but N-S is green -> switch to E-W
    if ns_surge and sig_cur:
        return True   # N-S surge but E-W is green -> switch to N-S
    return False


def grid_math_action(sim, state: tuple) -> int:
    """
    Decide actions for all 4 intersections, return encoded action (0-15).

    State layout:
      indices 0-3   : TL (sig, q1, q2, busy)
      indices 4-7   : TR
      indices 8-11  : BL
      indices 12-15 : BR
    """
    sig_TL, q1_TL, q2_TL, busy_TL = state[0:4]
    sig_TR, q1_TR, q2_TR, busy_TR = state[4:8]
    sig_BL, q1_BL, q2_BL, busy_BL = state[8:12]
    sig_BR, q1_BR, q2_BR, busy_BR = state[12:16]

    sigs = sim.traffic_signals   # [TL, TR, BL, BR]

    # Jackson-weighted congestion scores for all 4 nodes
    w_TL, w_TR, w_BL, w_BR = grid_congestion_scores(state)

    sw = [False, False, False, False]   # [TL, TR, BL, BR]

    # ── TL ────────────────────────────────────────────────────────────────────
    if not busy_TL and _elapsed(sim, 0):
        base = linear_algebra_action((sig_TL, q1_TL, q2_TL, busy_TL), w_TL)
        # Surges: from TR going west (E-W) and from BL going north (N-S)
        ew = predict_surge(q1_TR, bool(sig_TR))['recommend_green']
        ns = predict_surge(q2_BL, not bool(sig_BL))['recommend_green']
        sw[0] = _green_wave_override(sig_TL, ew, ns) or base
        if sw[0]:
            sigs[0].prev_update_time = sim.t

    # ── TR ────────────────────────────────────────────────────────────────────
    if not busy_TR and _elapsed(sim, 1):
        base = linear_algebra_action((sig_TR, q1_TR, q2_TR, busy_TR), w_TR)
        # Surges: from TL going east (E-W) and from BR going north (N-S)
        ew = predict_surge(q1_TL, bool(sig_TL))['recommend_green']
        ns = predict_surge(q2_BR, not bool(sig_BR))['recommend_green']
        sw[1] = _green_wave_override(sig_TR, ew, ns) or base
        if sw[1]:
            sigs[1].prev_update_time = sim.t

    # ── BL ────────────────────────────────────────────────────────────────────
    if not busy_BL and _elapsed(sim, 2):
        base = linear_algebra_action((sig_BL, q1_BL, q2_BL, busy_BL), w_BL)
        # Surges: from BR going west (E-W) and from TL going south (N-S)
        ew = predict_surge(q1_BR, bool(sig_BR))['recommend_green']
        ns = predict_surge(q2_TL, bool(sig_TL))['recommend_green']
        sw[2] = _green_wave_override(sig_BL, ew, ns) or base
        if sw[2]:
            sigs[2].prev_update_time = sim.t

    # ── BR ────────────────────────────────────────────────────────────────────
    if not busy_BR and _elapsed(sim, 3):
        base = linear_algebra_action((sig_BR, q1_BR, q2_BR, busy_BR), w_BR)
        # Surges: from BL going east (E-W) and from TR going south (N-S)
        ew = predict_surge(q1_BL, bool(sig_BL))['recommend_green']
        ns = predict_surge(q2_TR, bool(sig_TR))['recommend_green']
        sw[3] = _green_wave_override(sig_BR, ew, ns) or base
        if sw[3]:
            sigs[3].prev_update_time = sim.t

    return encode_action(*sw)


def grid_math_cycle(n_episodes: int, render: bool = False) -> None:
    """
    Run the Grid Network Math agent for n_episodes and report results.

    Prints:
      - 4-node Jackson Network model summary
      - Per-episode wait time or collision
      - Final averages
    """
    print_grid_summary(VEHICLE_RATE, MU)
    print(f"\n -- Running Grid Network Math agent for {n_episodes} episodes --")
    print("    (4-node Jackson Network + LA weighted decisions + green-wave)\n")

    env = GridEnvironment()
    total_wait = 0.0
    collisions = 0

    for ep in range(1, n_episodes + 1):
        state = env.reset(render)
        done  = False
        collision = False

        env.sim.dashboard_info.update({
            'method':         'grid',
            'episode':        ep,
            'total_episodes': n_episodes,
            'collisions':     collisions,
        })

        while not done:
            action = grid_math_action(env.sim, state)
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

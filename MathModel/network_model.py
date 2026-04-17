"""
Network Flow Model for Multi-Intersection Traffic Control.

Models the two-intersection corridor as a Jackson Network:
  - Each intersection is a node with external arrival rate λᵢ
  - Vehicles routed between intersections with probability pᵢⱼ
  - Effective arrival rate: Λᵢ = λᵢ + Σⱼ pⱼᵢ · Λⱼ
  - Matrix form: (I − Pᵀ) Λ = λ   →   solved with numpy.linalg.solve

Why Jackson Network?
  Standard M/M/1 (used in probability_model.py) only models a single
  isolated queue. When intersections are connected, vehicles leaving one
  join the queue of the next — the effective arrival rate Λᵢ is HIGHER
  than the raw external rate λᵢ.  Ignoring this leads to systematic
  underestimation of congestion at downstream intersections.

Path weight breakdown (from two_intersection.py PATHS):
  Total weight = 14
  Paths entering LEFT  (W/N-S_L): weights 3+2+2 = 7  → p(L→R) = 3/7
  Paths entering RIGHT (E/N-S_R): weights 3+2+2 = 7  → p(R→L) = 3/7

Propagation prediction:
  Vehicles released by the left intersection arrive at the right after
  ~connector_length / avg_speed seconds. Knowing this surge in advance
  lets the right signal pre-emptively turn green (green-wave coordination).
"""

import numpy as np
from MathModel.probability_model import mm1_expected_queue, mm1_expected_wait

# ── Network topology constants ────────────────────────────────────────────
VEHICLE_RATE  = 30          # vehicles / minute (matches two_intersection.py)
TOTAL_WEIGHT  = 14          # sum of all PATHS weights
MU            = 1.5         # service rate per intersection (vehicles/sec)

# Fraction of total traffic entering each intersection from external sources
EXT_FRAC_L = 7 / TOTAL_WEIGHT   # 0.5
EXT_FRAC_R = 7 / TOTAL_WEIGHT   # 0.5

# Routing probabilities: of vehicles that entered node X from external,
# what fraction exits toward the other intersection?
P_LR = 3 / 7   # left → right  (only the W→E through-path crosses both)
P_RL = 3 / 7   # right → left  (only the E→W through-path crosses both)

CONNECTOR_LENGTH  = 96.0   # simulation units between intersections
AVG_CONNECTOR_SPD = 10.0   # vehicles travel slower on the connector (units/s)


# ── Core Jackson Network solver ───────────────────────────────────────────

def external_arrival_rates(vehicle_rate: float = VEHICLE_RATE) -> np.ndarray:
    """
    External arrival rates λ = [λ_L, λ_R] in vehicles/second.

    Only counts vehicles that ENTER the network from outside
    (not recycled between intersections).
    """
    rate_per_sec = vehicle_rate / 60.0
    return np.array([EXT_FRAC_L * rate_per_sec,
                     EXT_FRAC_R * rate_per_sec])


def routing_matrix() -> np.ndarray:
    """
    2×2 routing matrix  P  where P[i,j] = P(vehicle leaving i goes to j).

        P = [[ 0,    P_LR ],
             [ P_RL,  0   ]]
    """
    return np.array([[0.0,  P_LR],
                     [P_RL, 0.0 ]])


def effective_arrival_rates(vehicle_rate: float = VEHICLE_RATE) -> np.ndarray:
    """
    Solve  (I − Pᵀ) Λ = λ  for effective arrival rates Λ = [Λ_L, Λ_R].

    Λᵢ accounts for both external arrivals AND vehicles flowing in from
    the other intersection. These are the rates to plug into M/M/1 formulas.

    Returns:
        numpy array [Λ_L, Λ_R]  (vehicles / second)
    """
    lam = external_arrival_rates(vehicle_rate)
    P   = routing_matrix()
    A   = np.eye(2) - P.T          # (I − Pᵀ)
    return np.linalg.solve(A, lam)  # Λ = A⁻¹ λ


def network_mm1_metrics(vehicle_rate: float = VEHICLE_RATE,
                        mu: float = MU) -> dict:
    """
    Per-intersection M/M/1 queue metrics using effective arrival rates.

    Returns dict keyed by 'L' and 'R', each with:
      lambda_ext  — external arrival rate (veh/s)
      lambda_eff  — effective arrival rate after routing (veh/s)
      rho         — traffic intensity  ρ = Λ/μ
      E_N         — expected queue length  E[N] = ρ/(1−ρ)
      E_W         — expected wait time     E[W] = 1/(μ−Λ)
    """
    lam_ext = external_arrival_rates(vehicle_rate)
    lam_eff = effective_arrival_rates(vehicle_rate)
    metrics = {}
    for node, le, lx in zip(['L', 'R'], lam_eff, lam_ext):
        rho = le / mu
        metrics[node] = {
            'lambda_ext': lx,
            'lambda_eff': le,
            'rho':        rho,
            'E_N':        mm1_expected_queue(le, mu),
            'E_W':        mm1_expected_wait(le, mu),
        }
    return metrics


# ── Propagation prediction ────────────────────────────────────────────────

def propagation_delay(connector_length: float = CONNECTOR_LENGTH,
                      avg_speed: float = AVG_CONNECTOR_SPD) -> float:
    """
    Estimated travel time (seconds) from one intersection to the other.

    When the upstream signal turns green, vehicles released onto the
    connector road will reach the downstream intersection after this delay.
    """
    return connector_length / avg_speed


def predict_downstream_surge(upstream_q_ew: int,
                              upstream_ew_green: bool,
                              mu: float = MU) -> dict:
    """
    Predict the vehicle surge arriving at the DOWNSTREAM intersection
    after the upstream releases vehicles on the through-path (E-W).

    Algorithm:
      If upstream has E-W green, vehicles clear at rate μ.
      Fraction P_LR of them are heading to the downstream intersection.
      They arrive after propagation_delay seconds.

    Args:
        upstream_q_ew:    queue length on the upstream E-W inbound road
        upstream_ew_green: True if upstream currently has E-W green
        mu:               service rate (vehicles/second)

    Returns:
        vehicles_to_arrive  — estimated vehicles heading downstream
        arrival_in_seconds  — when they'll arrive
        recommend_green     — should downstream prepare green?
    """
    delay = propagation_delay()
    if upstream_ew_green and upstream_q_ew > 0:
        vehicles_to_arrive = max(0, int(upstream_q_ew * P_LR))
        return {
            'vehicles_to_arrive': vehicles_to_arrive,
            'arrival_in_seconds': delay,
            'recommend_green':    vehicles_to_arrive >= 2,
        }
    return {
        'vehicles_to_arrive': 0,
        'arrival_in_seconds': delay,
        'recommend_green':    False,
    }


# ── Live congestion score (feeds into linear_algebra decision) ────────────

def network_congestion_scores(state: tuple) -> tuple:
    """
    Compute probability-weighted congestion scores for both intersections,
    using their EFFECTIVE arrival rates (not just external).

    Args:
        state: (sig_L, q1_L, q2_L, busy_L, sig_R, q1_R, q2_R, busy_R)

    Returns:
        ((w1_L, w2_L), (w1_R, w2_R))  — weight pairs for LA decision
    """
    _, q1_L, q2_L, _, _, q1_R, q2_R, _ = state
    lam_eff = effective_arrival_rates()
    metrics  = network_mm1_metrics()

    def score(q, node):
        e_n = metrics[node]['E_N']
        if e_n == 0 or e_n == float('inf'):
            return float(q)
        return q / e_n

    return (
        (score(q1_L, 'L'), score(q2_L, 'L')),
        (score(q1_R, 'R'), score(q2_R, 'R')),
    )


# ── Summary printer ───────────────────────────────────────────────────────

def print_network_summary(vehicle_rate: float = VEHICLE_RATE,
                          mu: float = MU) -> None:
    """Print a formatted Jackson Network model summary."""
    lam_ext = external_arrival_rates(vehicle_rate)
    lam_eff = effective_arrival_rates(vehicle_rate)
    metrics  = network_mm1_metrics(vehicle_rate, mu)

    print("\n" + "=" * 62)
    print("  NETWORK FLOW MODEL  (Jackson Network - 2 Intersections)")
    print("=" * 62)
    print(f"  Vehicle rate   : {vehicle_rate} veh/min")
    print(f"  Service rate mu: {mu} veh/sec per intersection")
    print()
    print(f"  Routing  L->R : P_LR = {P_LR:.4f}")
    print(f"           R->L : P_RL = {P_RL:.4f}")
    print()
    print("  Jackson system  (I - P^T) L = lam  ->  L = (I-P^T)^-1 lam")
    print()
    print(f"  {'Node':<8} {'lam_ext':>10} {'L_eff':>10} {'rho':>8} {'E[N]':>8} {'E[W]':>8}")
    print("  " + "-" * 56)
    for node in ['L', 'R']:
        m    = metrics[node]
        name = 'Left ' if node == 'L' else 'Right'
        print(f"  {name:<8} {m['lambda_ext']:>10.4f} {m['lambda_eff']:>10.4f}"
              f" {m['rho']:>8.4f} {m['E_N']:>8.4f} {m['E_W']:>8.4f}")
    print()
    print(f"  Propagation delay between intersections : "
          f"{propagation_delay():.1f} s")
    print("=" * 62)

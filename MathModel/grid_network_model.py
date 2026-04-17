"""
Jackson Network Model for 2x2 Grid Intersection.

Four nodes: TL, TR, BL, BR arranged as:
    [TL] - [TR]
     |       |
    [BL] - [BR]

Each node has 2 external-entry routes and receives vehicles from its
2 adjacent neighbours via connector roads.

Routing matrix P (4x4):
         TL    TR    BL    BR
    TL [  0,  1/4,  1/4,   0  ]
    TR [1/4,    0,    0,  1/4  ]
    BL [1/4,    0,    0,  1/4  ]
    BR [  0,  1/4,  1/4,   0  ]

Row sums = 1/2, so half of vehicles served at each node exit the network.

External fraction: 6 routes start at each node out of 24 total weight
  -> EXT_FRAC = 6/24 = 0.25 per node

With VEHICLE_RATE=30 veh/min and mu=1.5 veh/s the Jackson system gives:
  lambda_eff = 0.25 veh/s per node, rho ~ 0.167 (lightly loaded)
"""

import numpy as np
from MathModel.probability_model import mm1_expected_queue, mm1_expected_wait

VEHICLE_RATE     = 30     # veh / minute  (matches grid_intersection.py)
TOTAL_WEIGHT     = 24     # sum of PATHS weights
MU               = 1.5    # service rate per intersection (veh/s)
CONNECTOR_LENGTH = 96.0   # simulation units between adjacent intersections
AVG_CONNECTOR_SPD = 10.0  # vehicles travel slower on connectors

EXT_FRAC = 6 / TOTAL_WEIGHT   # 0.25 -- equal for all 4 nodes

# Routing probabilities: each node forwards 1/4 of its served vehicles
# to each of its 2 neighbours (and 1/2 exit the network)
P_NEIGHBOUR = 3 / 12   # = 1/4

NODES = ['TL', 'TR', 'BL', 'BR']   # index order throughout this module


# ── Core Jackson Network solver ───────────────────────────────────────────────

def external_arrival_rates(vehicle_rate: float = VEHICLE_RATE) -> np.ndarray:
    """
    External arrival rates lambda = [lam_TL, lam_TR, lam_BL, lam_BR] veh/s.
    All equal by symmetry.
    """
    rate_per_sec = vehicle_rate / 60.0
    lam = EXT_FRAC * rate_per_sec
    return np.full(4, lam)


def routing_matrix() -> np.ndarray:
    """
    4x4 routing matrix P where P[i,j] = P(vehicle leaving i goes to j).

        TL TR BL BR
    TL [0  q  q  0 ]
    TR [q  0  0  q ]    where q = P_NEIGHBOUR = 1/4
    BL [q  0  0  q ]
    BR [0  q  q  0 ]
    """
    q = P_NEIGHBOUR
    return np.array([
        [0, q, q, 0],
        [q, 0, 0, q],
        [q, 0, 0, q],
        [0, q, q, 0],
    ])


def effective_arrival_rates(vehicle_rate: float = VEHICLE_RATE) -> np.ndarray:
    """
    Solve (I - P^T) Lambda = lambda for effective arrival rates.

    Returns:
        numpy array [Lambda_TL, Lambda_TR, Lambda_BL, Lambda_BR]  veh/s
    """
    lam = external_arrival_rates(vehicle_rate)
    P   = routing_matrix()
    A   = np.eye(4) - P.T
    return np.linalg.solve(A, lam)


def grid_mm1_metrics(vehicle_rate: float = VEHICLE_RATE,
                     mu: float = MU) -> dict:
    """
    Per-node M/M/1 metrics using effective arrival rates.

    Returns dict keyed by node name, each with:
      lambda_ext, lambda_eff, rho, E_N, E_W
    """
    lam_ext = external_arrival_rates(vehicle_rate)
    lam_eff = effective_arrival_rates(vehicle_rate)
    metrics = {}
    for i, node in enumerate(NODES):
        le = lam_eff[i]
        rho = le / mu
        metrics[node] = {
            'lambda_ext': lam_ext[i],
            'lambda_eff': le,
            'rho':        rho,
            'E_N':        mm1_expected_queue(le, mu),
            'E_W':        mm1_expected_wait(le, mu),
        }
    return metrics


# ── Propagation prediction ────────────────────────────────────────────────────

def propagation_delay(connector_length: float = CONNECTOR_LENGTH,
                      avg_speed: float = AVG_CONNECTOR_SPD) -> float:
    """Travel time (seconds) along one connector road."""
    return connector_length / avg_speed


def predict_surge(upstream_q: int, upstream_green: bool) -> dict:
    """
    Generic upstream-surge prediction for one connector direction.

    Args:
        upstream_q:     queue length on the upstream road group
        upstream_green: True if upstream has green for that direction

    Returns:
        vehicles_to_arrive, arrival_in_seconds, recommend_green
    """
    delay = propagation_delay()
    if upstream_green and upstream_q > 0:
        n = max(0, int(upstream_q * P_NEIGHBOUR))
        return {
            'vehicles_to_arrive': n,
            'arrival_in_seconds': delay,
            'recommend_green':    n >= 2,
        }
    return {
        'vehicles_to_arrive': 0,
        'arrival_in_seconds': delay,
        'recommend_green':    False,
    }


# ── Live congestion scores (feeds linear_algebra decision) ────────────────────

def grid_congestion_scores(state: tuple) -> tuple:
    """
    Probability-weighted congestion scores for all 4 nodes using effective
    arrival rates from the Jackson Network.

    Args:
        state: 16-tuple
          (sig_TL, q1_TL, q2_TL, busy_TL,
           sig_TR, q1_TR, q2_TR, busy_TR,
           sig_BL, q1_BL, q2_BL, busy_BL,
           sig_BR, q1_BR, q2_BR, busy_BR)

    Returns:
        tuple of 4 weight-pairs: ((w1_TL,w2_TL), ..., (w1_BR,w2_BR))
    """
    metrics = grid_mm1_metrics()

    def score(q, node):
        e_n = metrics[node]['E_N']
        if e_n == 0 or e_n == float('inf'):
            return float(q)
        return q / e_n

    results = []
    for i, node in enumerate(NODES):
        base = i * 4
        q1 = state[base + 1]
        q2 = state[base + 2]
        results.append((score(q1, node), score(q2, node)))
    return tuple(results)


# ── Summary printer ───────────────────────────────────────────────────────────

def print_grid_summary(vehicle_rate: float = VEHICLE_RATE,
                       mu: float = MU) -> None:
    """Print a formatted 4-node Jackson Network summary."""
    metrics = grid_mm1_metrics(vehicle_rate, mu)

    print("\n" + "=" * 66)
    print("  GRID NETWORK MODEL  (Jackson Network - 2x2 Grid, 4 Nodes)")
    print("=" * 66)
    print(f"  Vehicle rate   : {vehicle_rate} veh/min")
    print(f"  Service rate mu: {mu} veh/sec per intersection")
    print()
    print("  Routing topology (each node forwards 1/4 to each neighbour):")
    print("    TL <-> TR  (horizontal top)")
    print("    BL <-> BR  (horizontal bottom)")
    print("    TL <-> BL  (vertical left)")
    print("    TR <-> BR  (vertical right)")
    print()
    print("  Jackson system  (I - P^T) Lambda = lam  ->  Lambda = (I-P^T)^-1 lam")
    print()
    print(f"  {'Node':<8} {'lam_ext':>10} {'Lambda':>10} {'rho':>8} {'E[N]':>8} {'E[W]':>8}")
    print("  " + "-" * 60)
    for node in NODES:
        m = metrics[node]
        print(f"  {node:<8} {m['lambda_ext']:>10.4f} {m['lambda_eff']:>10.4f}"
              f" {m['rho']:>8.4f} {m['E_N']:>8.4f} {m['E_W']:>8.4f}")
    print()
    print(f"  Propagation delay between intersections: {propagation_delay():.1f} s")
    print("=" * 66)

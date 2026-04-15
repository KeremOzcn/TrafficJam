"""
Linear Algebra Model for Traffic State Representation and Decision Making.

The traffic state is represented as vectors and matrices. Linear algebra
operations (norms, dot products, matrix operations) are used to measure
congestion and decide signal switching.

Key Concepts:
-------------
State Vector:
    v = [q1, q2, signal, junction]  ∈ R^4
    where q1, q2 are queue lengths and signal/junction are binary flags.

Congestion Vector:
    c = [q1, q2]  ∈ R^2
    Represents the queue load in each direction.

L1 Norm  (Manhattan):  ||c||_1 = |q1| + |q2|
L2 Norm  (Euclidean):  ||c||_2 = sqrt(q1^2 + q2^2)
L∞ Norm  (Chebyshev):  ||c||_∞ = max(|q1|, |q2|)

Congestion Matrix:
    C = [[q1,  0 ],   (diagonal matrix — independent direction loads)
         [ 0,  q2]]

    Frobenius norm: ||C||_F = sqrt(q1^2 + q2^2)  (same as L2 of c)

Decision via Dominant Direction:
    The direction with the higher queue component dominates.
    If the dominant direction has a red light → SWITCH.
    This is the LQF heuristic expressed in linear algebra terms.

Weighted Decision:
    Combine probability congestion scores (from probability_model) as
    weights: w = [score_1, score_2]
    Weighted direction = argmax(w)
"""

import math
import numpy as np


# ── Vector operations ────────────────────────────────────────────────────────

def state_vector(state: tuple) -> np.ndarray:
    """
    Convert environment state tuple to a numpy state vector.

    Args:
        state: (signal_state, q1, q2, junction_busy)

    Returns:
        v = [signal_state, q1, q2, junction_busy]  as float64 array
    """
    signal_state, q1, q2, junction_busy = state
    return np.array([float(signal_state), float(q1), float(q2),
                     float(junction_busy)], dtype=np.float64)


def congestion_vector(state: tuple) -> np.ndarray:
    """
    Extract the congestion sub-vector [q1, q2] from state.

    Args:
        state: (signal_state, q1, q2, junction_busy)

    Returns:
        c = [q1, q2]  as float64 array
    """
    _, q1, q2, _ = state
    return np.array([float(q1), float(q2)], dtype=np.float64)


# ── Norms ────────────────────────────────────────────────────────────────────

def l1_norm(v: np.ndarray) -> float:
    """L1 (Manhattan) norm: ||v||_1 = Σ|v_i|"""
    return float(np.sum(np.abs(v)))


def l2_norm(v: np.ndarray) -> float:
    """L2 (Euclidean) norm: ||v||_2 = sqrt(Σ v_i^2)"""
    return float(np.linalg.norm(v))


def linf_norm(v: np.ndarray) -> float:
    """L∞ (Chebyshev) norm: ||v||_∞ = max|v_i|"""
    return float(np.max(np.abs(v)))


def total_congestion(state: tuple) -> dict:
    """
    Compute all three norms of the congestion vector for a given state.

    Args:
        state: (signal_state, q1, q2, junction_busy)

    Returns:
        dict with keys 'l1', 'l2', 'linf', 'vector'
    """
    c = congestion_vector(state)
    return {
        'vector': c,
        'l1':     l1_norm(c),
        'l2':     l2_norm(c),
        'linf':   linf_norm(c),
    }


# ── Congestion Matrix ────────────────────────────────────────────────────────

def congestion_matrix(state: tuple) -> np.ndarray:
    """
    Build a 2×2 diagonal congestion matrix from state queues.

    C = [[q1,  0],
         [ 0, q2]]

    Eigenvalues of C are q1 and q2 directly (diagonal matrix).
    Frobenius norm ||C||_F = sqrt(q1^2 + q2^2) equals L2 norm of c.

    Args:
        state: (signal_state, q1, q2, junction_busy)

    Returns:
        2×2 numpy array
    """
    _, q1, q2, _ = state
    return np.diag([float(q1), float(q2)])


def frobenius_norm(M: np.ndarray) -> float:
    """Frobenius norm: ||M||_F = sqrt(Σ_ij M_ij^2)"""
    return float(np.linalg.norm(M, 'fro'))


def dominant_direction(state: tuple) -> int:
    """
    Return the index of the more congested direction using L∞ norm
    (i.e., simply which queue is longer).

    Returns:
        0 → direction 1 (West/East) dominates
        1 → direction 2 (South/North) dominates
       -1 → tie (equal queues)
    """
    c = congestion_vector(state)
    if c[0] > c[1]:
        return 0
    elif c[1] > c[0]:
        return 1
    return -1


# ── Decision function ────────────────────────────────────────────────────────

def linear_algebra_action(state: tuple,
                           prob_weights: tuple = (1.0, 1.0)) -> bool:
    """
    Decide signal action using linear algebra.

    Algorithm:
        1. Build weighted congestion vector w = [w1*q1, w2*q2]
           where w1, w2 are probability-based congestion scores.
        2. Find dominant direction via argmax of w.
        3. If dominant direction currently has red → SWITCH.
        4. If tie or junction busy → HOLD.

    Args:
        state:        (signal_state, q1, q2, junction_busy)
        prob_weights: (w1, w2) probability congestion scores from
                      probability_model.congestion_score()

    Returns:
        True  → SWITCH
        False → HOLD
    """
    signal_state, q1, q2, junction_busy = state

    if junction_busy:
        return False  # Never switch while intersection is occupied

    w1, w2 = prob_weights
    weighted = np.array([w1 * q1, w2 * q2], dtype=np.float64)

    dom = int(np.argmax(weighted))

    # signal_state True  → dir1 (index 0) has green
    # signal_state False → dir2 (index 1) has green
    dir1_has_green = bool(signal_state)

    if weighted[0] == weighted[1]:
        return False  # Tie → hold

    if dom == 0 and not dir1_has_green:
        return True   # Dir1 dominates but has red → switch
    if dom == 1 and dir1_has_green:
        return True   # Dir2 dominates but has red → switch

    return False  # Dominant direction already has green → hold


# ── Report helpers ───────────────────────────────────────────────────────────

def print_state_analysis(state: tuple) -> None:
    """Print a formatted linear algebra analysis of the current state."""
    signal_state, q1, q2, junction_busy = state
    c = congestion_vector(state)
    C = congestion_matrix(state)
    eigenvalues = np.linalg.eigvals(C)

    print(f"\n  State vector     v = {state_vector(state)}")
    print(f"  Congestion vec   c = [{q1:.0f}, {q2:.0f}]")
    print(f"  L1 norm          ||c||_1 = {l1_norm(c):.2f}")
    print(f"  L2 norm          ||c||_2 = {l2_norm(c):.4f}")
    print(f"  L∞ norm          ||c||_∞ = {linf_norm(c):.2f}")
    print(f"  Congestion matrix C =\n{C}")
    print(f"  Eigenvalues      λ = {eigenvalues}")
    print(f"  Frobenius norm   ||C||_F = {frobenius_norm(C):.4f}")
    dom = dominant_direction(state)
    dom_str = {0: "Direction 1 (W/E)", 1: "Direction 2 (S/N)", -1: "Tie"}[dom]
    print(f"  Dominant dir     = {dom_str}")

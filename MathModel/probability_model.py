"""
Probability Model for Traffic Arrival and Queue Analysis.

Models vehicle arrivals as a Poisson process and uses M/M/1 queuing
theory to compute expected queue lengths and wait times.

Key Concepts:
-------------
Poisson Process:
    Vehicles arrive randomly at rate λ (vehicles/second).
    The probability of exactly k arrivals in t seconds:
        P(N=k) = e^(-λt) * (λt)^k / k!
    Expected arrivals in t seconds:
        E[N] = λ * t

M/M/1 Queue (single-server model):
    Arrival rate:  λ  (vehicles/second)
    Service rate:  μ  (vehicles/second — how fast intersection clears)
    Traffic intensity (utilization): ρ = λ / μ   (must be < 1 for stability)

    Expected queue length:  E[N] = ρ / (1 - ρ)
    Expected wait time:     E[W] = 1 / (μ - λ)
    Probability of n vehicles in system: P(N=n) = (1 - ρ) * ρ^n
"""

import math

# Simulation constants (from two_way_intersection.py and vehicle.py)
VEHICLE_RATE_PER_MINUTE = 35          # vehicles/minute per direction
LAMBDA = VEHICLE_RATE_PER_MINUTE / 60 # arrivals per second (≈ 0.583)

# Average service rate: intersection clears ~1.5 vehicles/second when green
MU = 1.5


def poisson_pmf(k: int, lam: float, t: float = 1.0) -> float:
    """
    Probability of exactly k vehicle arrivals in t seconds.

    P(N = k) = e^(-λt) * (λt)^k / k!

    Args:
        k:   number of arrivals
        lam: arrival rate (vehicles/second)
        t:   time window (seconds)

    Returns:
        Probability value in [0, 1]
    """
    rate = lam * t
    return math.exp(-rate) * (rate ** k) / math.factorial(k)


def expected_arrivals(t: float, lam: float = LAMBDA) -> float:
    """
    Expected number of vehicle arrivals in t seconds.

    E[N] = λ * t

    Args:
        t:   time window (seconds)
        lam: arrival rate (vehicles/second)

    Returns:
        Expected number of arrivals (float)
    """
    return lam * t


def mm1_expected_queue(lam: float = LAMBDA, mu: float = MU) -> float:
    """
    Expected number of vehicles waiting in an M/M/1 queue.

    E[N] = ρ / (1 - ρ)   where ρ = λ / μ

    Args:
        lam: arrival rate (vehicles/second)
        mu:  service rate (vehicles/second)

    Returns:
        Expected queue length, or inf if system is unstable (ρ >= 1)
    """
    rho = lam / mu
    if rho >= 1.0:
        return float('inf')
    return rho / (1.0 - rho)


def mm1_expected_wait(lam: float = LAMBDA, mu: float = MU) -> float:
    """
    Expected wait time for a vehicle in an M/M/1 queue (Little's Law).

    E[W] = 1 / (μ - λ)

    Args:
        lam: arrival rate (vehicles/second)
        mu:  service rate (vehicles/second)

    Returns:
        Expected wait time in seconds, or inf if system is unstable
    """
    if lam >= mu:
        return float('inf')
    return 1.0 / (mu - lam)


def queue_overflow_probability(queue_len: int, lam: float = LAMBDA,
                                mu: float = MU) -> float:
    """
    Probability that the queue length exceeds queue_len in the M/M/1 model.

    P(N > n) = ρ^(n+1)   where ρ = λ / μ

    Args:
        queue_len: threshold queue length
        lam:       arrival rate
        mu:        service rate

    Returns:
        Probability in [0, 1]
    """
    rho = lam / mu
    if rho >= 1.0:
        return 1.0
    return rho ** (queue_len + 1)


def congestion_score(q1: int, q2: int, lam: float = LAMBDA,
                     mu: float = MU) -> tuple:
    """
    Compute a probabilistic congestion score for each direction.

    Score = observed_queue / E[N_mm1]  (ratio of actual to expected queue)
    A score > 1 means the queue is longer than statistically expected.

    Args:
        q1:  vehicles in direction 1
        q2:  vehicles in direction 2
        lam: arrival rate
        mu:  service rate

    Returns:
        (score_1, score_2): congestion scores for each direction
    """
    expected = mm1_expected_queue(lam, mu)
    if expected == 0 or expected == float('inf'):
        return (float(q1), float(q2))
    return (q1 / expected, q2 / expected)


def print_summary(lam: float = LAMBDA, mu: float = MU) -> None:
    """Print a formatted probability model summary for the report."""
    rho = lam / mu
    print("\n" + "=" * 55)
    print("  PROBABILITY MODEL SUMMARY (M/M/1 Queuing Theory)")
    print("=" * 55)
    print(f"  Arrival rate      lam = {lam:.4f} vehicles/sec")
    print(f"  Service rate       mu = {mu:.4f} vehicles/sec")
    print(f"  Traffic intensity rho = lam/mu = {rho:.4f}")
    print(f"  Expected queue   E[N] = rho/(1-rho) = {mm1_expected_queue(lam, mu):.4f} vehicles")
    print(f"  Expected wait    E[W] = 1/(mu-lam)  = {mm1_expected_wait(lam, mu):.4f} sec")
    print()
    print("  Poisson PMF  P(k arrivals in 1 sec):")
    for k in range(6):
        p = poisson_pmf(k, lam)
        bar = "#" * int(p * 80)
        print(f"    k={k}: {p:.4f}  {bar}")
    print("=" * 55)

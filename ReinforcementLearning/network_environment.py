"""
Multi-Intersection RL Environment.

Wraps the two-intersection simulation as a standard gym-style environment.

State (8-tuple):
    (sig_L, q1_L, q2_L, busy_L, sig_R, q1_R, q2_R, busy_R)
    where each (sig, q1, q2, busy) is the per-intersection state:
      sig  — True if E-W direction currently has green
      q1   — vehicles in signal group-0 roads (E-W inbound)
      q2   — vehicles in signal group-1 roads (N-S inbound)
      busy — any vehicle on the intersection's through roads

Action space: {0, 1, 2, 3}
    0 → hold both signals
    1 → switch LEFT signal only
    2 → switch RIGHT signal only
    3 → switch BOTH signals

Reward:
    Decrease in total inbound queue length since the previous step
    (positive = vehicles cleared, negative = congestion increased).
"""

from typing import Optional, Tuple

from TrafficSimulator import Simulation
from TrafficSimulator.Setups.two_intersection import (
    two_intersection_setup,
    LEFT_THROUGH_ROADS,
    RIGHT_THROUGH_ROADS,
)

# Action → (switch_left, switch_right)
ACTION_MAP = {
    0: (False, False),
    1: (True,  False),
    2: (False, True),
    3: (True,  True),
}

MAX_GEN = 80   # more vehicles than single-intersection to fill both nodes


class NetworkEnvironment:
    """Gym-style environment for the two-intersection corridor."""

    def __init__(self):
        self.action_space = list(ACTION_MAP.keys())
        self.sim: Optional[Simulation] = None
        self._prev_inbound: int = 0

    # ── Core interface ────────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[Tuple, float, bool, bool]:
        """
        Apply action and advance simulation one step.

        Returns:
            new_state, reward, terminated, truncated
        """
        switch_l, switch_r = ACTION_MAP[action]
        self.sim.run_with_actions([switch_l, switch_r])

        new_state = self.get_state()
        reward    = self.get_reward(new_state)

        self._prev_inbound = (new_state[1] + new_state[2] +
                              new_state[5] + new_state[6])

        terminated = self.sim.completed
        truncated  = self.sim.gui_closed
        return new_state, reward, terminated, truncated

    def reset(self, render: bool = False) -> Tuple:
        self.sim = two_intersection_setup(MAX_GEN)
        if render:
            self.sim.init_gui()
        self._prev_inbound = 0
        return self.get_state()

    # ── State / reward ────────────────────────────────────────────────────

    def get_state(self) -> Tuple:
        """
        Build the 8-tuple state from the current simulation.

        For each of the two traffic signals:
          sig   = current_cycle[0]  (True = E-W green)
          q1    = total vehicles in group-0 (E-W) inbound roads
          q2    = total vehicles in group-1 (N-S) inbound roads
          busy  = any vehicle on the intersection's through roads
        """
        state = []
        through_sets = [LEFT_THROUGH_ROADS, RIGHT_THROUGH_ROADS]

        for sig_idx, sig in enumerate(self.sim.traffic_signals):
            sig_state = bool(sig.current_cycle[0])
            q1 = sum(len(r.vehicles) for r in sig.roads[0])
            q2 = sum(len(r.vehicles) for r in sig.roads[1])
            busy = any(
                len(self.sim.roads[i].vehicles) > 0
                for i in through_sets[sig_idx]
                if i in self.sim.non_empty_roads
            )
            state.extend([sig_state, q1, q2, busy])

        return tuple(state)

    def get_reward(self, state: Tuple) -> float:
        """
        Reward = reduction in total inbound queue relative to previous step.
        """
        total_now = state[1] + state[2] + state[5] + state[6]
        return float(self._prev_inbound - total_now)

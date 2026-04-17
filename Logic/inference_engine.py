"""
Propositional Logic Inference Engine using Modus Ponens.

State tuple: (signal_state, q1, q2, junction_busy)

Atomic Propositions:
    A  : signal_state is True   — Direction 1 (West/East) currently has green
    B  : q1 > QUEUE_THRESHOLD   — Direction 1 queue is heavy
    C  : q2 > QUEUE_THRESHOLD   — Direction 2 queue is heavy
    E  : q1 > q2                — Direction 1 queue is strictly longer than Dir2

Note on D (non_empty_junction):
    D is monitored and reported but NOT used in switching rules.
    The minimum cycle time guard in logic_utils.py already ensures
    the signal is never switched while a phase transition is unsafe.

Inference Rules (Modus Ponens: IF antecedent is TRUE THEN fire consequent):
    R1: ~A & B & ~C        -> SWITCH  (Dir1 red but congested, Dir2 clear)
    R2:  A & C & ~B        -> SWITCH  (Dir2 congested, Dir1 clear)
    R3: ~A & B & C & E     -> SWITCH  (Both heavy, Dir1 longer — favour Dir1)
    R4:  A & B & C & ~E    -> SWITCH  (Both heavy, Dir2 longer — favour Dir2)
    R5: ~B & ~C            -> HOLD    (Both empty, no urgency)
    R6: DEFAULT            -> HOLD

Rules are evaluated in priority order (R1 first).
The first matching rule fires (classic Modus Ponens chain).
"""

QUEUE_THRESHOLD = 3  # vehicles — queues above this are considered "heavy"

# Each rule: (label, antecedent_fn(A,B,C,E)->bool, action)
# action: True = SWITCH, False = HOLD
RULES = [
    (
        "R1 [Congest-1]  : ~A & B & ~C      -> SWITCH",
        lambda A, B, C, E: (not A) and B and (not C),
        True,
    ),
    (
        "R2 [Congest-2]  :  A & C & ~B      -> SWITCH",
        lambda A, B, C, E: A and C and (not B),
        True,
    ),
    (
        "R3 [Balanced-1] : ~A & B & C & E   -> SWITCH",
        lambda A, B, C, E: (not A) and B and C and E,
        True,
    ),
    (
        "R4 [Balanced-2] :  A & B & C & ~E  -> SWITCH",
        lambda A, B, C, E: A and B and C and (not E),
        True,
    ),
    (
        "R5 [Idle]       : ~B & ~C          -> HOLD",
        lambda A, B, C, E: (not B) and (not C),
        False,
    ),
    (
        "R6 [Default]    : TRUE             -> HOLD",
        lambda A, B, C, E: True,
        False,
    ),
]


def extract_propositions(state: tuple) -> tuple:
    """
    Map raw environment state to atomic propositions.

    Args:
        state: (signal_state, q1, q2, junction_busy)

    Returns:
        (A, B, C, D, E) — D is returned for monitoring only
    """
    signal_state, q1, q2, junction_busy = state
    A = bool(signal_state)
    B = q1 > QUEUE_THRESHOLD
    C = q2 > QUEUE_THRESHOLD
    D = bool(junction_busy)   # monitored, not used in rules
    E = q1 > q2
    return A, B, C, D, E


def infer(state: tuple, verbose: bool = False) -> bool:
    """
    Run Modus Ponens over all rules and return the action decided by the
    first rule whose antecedent evaluates to True.

    Modus Ponens: IF (P -> Q) AND P is TRUE, THEN conclude Q.

    Args:
        state:   (signal_state, q1, q2, junction_busy)
        verbose: print which rule fired

    Returns:
        True  -> SWITCH signal
        False -> HOLD current signal
    """
    A, B, C, D, E = extract_propositions(state)

    for label, antecedent, action in RULES:
        if antecedent(A, B, C, E):           # Modus Ponens
            if verbose:
                print(f"  [Logic] {label}  props=(A={A},B={B},C={C},D={D},E={E})")
            return action

    return False  # unreachable — R6 always fires


def truth_table() -> str:
    """
    Generate the full truth table for all propositions and rule firings.

    Returns:
        Formatted string of the truth table (32 rows: 2^5 combos of A,B,C,D,E
        but D is display-only so we show 16 meaningful rows).
    """
    n_rules = len(RULES)
    rule_headers = "  ".join(f"{'R'+str(i+1):^5}" for i in range(n_rules))
    header = (
        f"{'A':^5} {'B':^5} {'C':^5} {'E':^5} | "
        f"{rule_headers} | "
        f"{'Fired':^8} {'Action':^8}"
    )
    separator = "-" * len(header)
    rows = [header, separator]

    for A in [False, True]:
        for B in [False, True]:
            for C in [False, True]:
                for E in [False, True]:
                    rule_vals = [ant(A, B, C, E) for _, ant, _ in RULES]
                    fired_idx = next(i for i, v in enumerate(rule_vals) if v)
                    fired_label = f"R{fired_idx + 1}"
                    action = RULES[fired_idx][2]
                    action_str = "SWITCH" if action else "HOLD"
                    rule_str = "  ".join(
                        f"{'T' if v else 'F':^5}" for v in rule_vals)
                    rows.append(
                        f"{'T' if A else 'F':^5} {'T' if B else 'F':^5} "
                        f"{'T' if C else 'F':^5} {'T' if E else 'F':^5} | "
                        f"{rule_str} | "
                        f"{fired_label:^8} {action_str:^8}"
                    )

    return "\n".join(rows)

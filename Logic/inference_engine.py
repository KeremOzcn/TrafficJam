"""
Propositional Logic Inference Engine using Modus Ponens.

State tuple: (signal_state, q1, q2, junction_busy)

Atomic Propositions:
    A  : signal_state is True   — Direction 1 (West/East) currently has green
    B  : q1 > QUEUE_THRESHOLD   — Direction 1 queue is heavy
    C  : q2 > QUEUE_THRESHOLD   — Direction 2 queue is heavy
    D  : non_empty_junction     — Vehicles are currently inside the intersection

Inference Rules (Modus Ponens: IF antecedent is TRUE THEN fire consequent):
    R1: D                        → HOLD   (safety: junction occupied, never switch mid-crossing)
    R2: ¬A ∧ B ∧ ¬C ∧ ¬D       → SWITCH (dir1 congested & has red, dir2 is light)
    R3:  A ∧ C ∧ ¬B ∧ ¬D       → SWITCH (dir2 congested & has red, dir1 is light)
    R4:  B ∧ C ∧ ¬D             → HOLD   (both directions heavy, keep current green)
    R5: ¬B ∧ ¬C ∧ ¬D           → HOLD   (both directions empty, no need to switch)
    R6: DEFAULT                  → HOLD

Rules are evaluated in priority order (R1 first). The first matching rule fires.
"""

QUEUE_THRESHOLD = 3  # vehicles — queues above this are considered "heavy"

# Each rule is: (label, antecedent_fn, action)
# antecedent_fn(A, B, C, D) -> bool
# action: True = SWITCH, False = HOLD
RULES = [
    (
        "R1 [Safety]    : D → HOLD",
        lambda A, B, C, D: D,
        False,  # HOLD
    ),
    (
        "R2 [Congest-1] : ¬A ∧ B ∧ ¬C ∧ ¬D → SWITCH",
        lambda A, B, C, D: (not A) and B and (not C) and (not D),
        True,   # SWITCH
    ),
    (
        "R3 [Congest-2] :  A ∧ C ∧ ¬B ∧ ¬D → SWITCH",
        lambda A, B, C, D: A and C and (not B) and (not D),
        True,   # SWITCH
    ),
    (
        "R4 [Balanced]  :  B ∧ C ∧ ¬D → HOLD",
        lambda A, B, C, D: B and C and (not D),
        False,  # HOLD
    ),
    (
        "R5 [Idle]      : ¬B ∧ ¬C ∧ ¬D → HOLD",
        lambda A, B, C, D: (not B) and (not C) and (not D),
        False,  # HOLD
    ),
    (
        "R6 [Default]   : TRUE → HOLD",
        lambda A, B, C, D: True,
        False,  # HOLD
    ),
]


def extract_propositions(state: tuple) -> tuple:
    """
    Map raw environment state to atomic propositions.

    Args:
        state: (signal_state, q1, q2, junction_busy)

    Returns:
        (A, B, C, D) — boolean proposition values
    """
    signal_state, q1, q2, junction_busy = state
    A = bool(signal_state)
    B = q1 > QUEUE_THRESHOLD
    C = q2 > QUEUE_THRESHOLD
    D = bool(junction_busy)
    return A, B, C, D


def infer(state: tuple, verbose: bool = False) -> bool:
    """
    Run Modus Ponens over all rules and return the action decided by the
    first rule whose antecedent evaluates to True.

    Modus Ponens: IF (P → Q) AND P is TRUE, THEN conclude Q.

    Args:
        state:   (signal_state, q1, q2, junction_busy)
        verbose: print which rule fired

    Returns:
        True  → SWITCH signal
        False → HOLD current signal
    """
    A, B, C, D = extract_propositions(state)

    for label, antecedent, action in RULES:
        if antecedent(A, B, C, D):           # Modus Ponens: antecedent is True
            if verbose:
                print(f"  [Logic] Fired: {label}  |  props=({A},{B},{C},{D})")
            return action

    # Unreachable (R6 is always True) but kept for safety
    return False


def truth_table() -> str:
    """
    Generate the full truth table for all propositions and rule firings.
    Useful for the project report.

    Returns:
        Formatted string of the truth table.
    """
    header = (
        f"{'A':^5} {'B':^5} {'C':^5} {'D':^5} | "
        f"{'R1':^5} {'R2':^5} {'R3':^5} {'R4':^5} {'R5':^5} {'R6':^5} | "
        f"{'Fired':^10} {'Action':^8}"
    )
    separator = "-" * len(header)
    rows = [header, separator]

    for A in [False, True]:
        for B in [False, True]:
            for C in [False, True]:
                for D in [False, True]:
                    rule_vals = [antecedent(A, B, C, D) for _, antecedent, _ in RULES]
                    fired_idx = next(i for i, v in enumerate(rule_vals) if v)
                    fired_label = f"R{fired_idx + 1}"
                    action = RULES[fired_idx][2]
                    action_str = "SWITCH" if action else "HOLD"
                    rule_str = "  ".join(f"{'T' if v else 'F':^5}" for v in rule_vals)
                    rows.append(
                        f"{'T' if A else 'F':^5} {'T' if B else 'F':^5} "
                        f"{'T' if C else 'F':^5} {'T' if D else 'F':^5} | "
                        f"{rule_str} | "
                        f"{fired_label:^10} {action_str:^8}"
                    )

    return "\n".join(rows)

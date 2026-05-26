"""A Cox-Ross-Rubinstein binomial tree pricer.

Black-Scholes has a closed form but only prices European options. The binomial
tree is slower but more flexible: it prices American options too, by checking at
every node whether early exercise beats holding on. As the number of steps grows,
the European binomial price converges to the Black-Scholes price, which is a
useful sanity check that both implementations agree.
"""

from __future__ import annotations

import math


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    steps: int = 500,
    american: bool = False,
) -> float:
    """Price a European or American option on a CRR binomial tree."""
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if T <= 0 or sigma <= 0:
        intrinsic = (S - K) if option_type == "call" else (K - S)
        return max(intrinsic, 0.0)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "Risk-neutral probability fell outside [0, 1]; increase steps or "
            "check the inputs."
        )

    # Option values at expiry across the terminal nodes.
    values = []
    for i in range(steps + 1):
        spot = S * (u ** (steps - i)) * (d ** i)
        payoff = (spot - K) if option_type == "call" else (K - spot)
        values.append(max(payoff, 0.0))

    # Step backwards through the tree, allowing early exercise if American.
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            hold = disc * (p * values[i] + (1.0 - p) * values[i + 1])
            if american:
                spot = S * (u ** (step - i)) * (d ** i)
                exercise = (spot - K) if option_type == "call" else (K - spot)
                values[i] = max(hold, exercise, 0.0)
            else:
                values[i] = hold
    return values[0]

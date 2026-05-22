"""An independent Monte Carlo price, used to check the closed form.

If the analytic formula and a completely separate simulation agree, that is
strong evidence the implementation is right. This is not how you would price a
vanilla European option in practice (the formula is exact and instant), but it
is exactly how you gain confidence in the formula, and how you would price the
exotic payoffs that have no closed form.

The terminal price is simulated under the risk-neutral measure:

    S_T = S * exp((r - q - 0.5 * sigma**2) * T + sigma * sqrt(T) * Z)

with Z standard normal. The option value is the discounted average payoff.
"""

from __future__ import annotations

import math

import numpy as np


def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 200_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Return (price estimate, standard error) for a European option.

    The standard error lets you see how tight the estimate is, and it shrinks
    with the square root of the number of paths.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * math.sqrt(T) * z
    terminal = S * np.exp(drift + diffusion)

    if option_type == "call":
        payoffs = np.maximum(terminal - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal, 0.0)

    discounted = math.exp(-r * T) * payoffs
    price_estimate = float(discounted.mean())
    standard_error = float(discounted.std(ddof=1) / math.sqrt(n_paths))
    return price_estimate, standard_error

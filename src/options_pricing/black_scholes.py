"""The Black-Scholes-Merton pricing formula for European options.

The model assumes the underlying follows geometric Brownian motion with
constant volatility, that trading is continuous and frictionless, and that the
option can only be exercised at expiry. Those assumptions are exactly what the
README pushes back on, but within them the price has a clean closed form.

A continuous dividend yield ``q`` is supported (set it to 0 for a
non-dividend-paying stock). The standard-normal CDF comes from the standard
library, so there is no SciPy dependency.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()   # standard normal, mean 0, variance 1


def _norm_cdf(x: float) -> float:
    return _N.cdf(x)


def d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float]:
    """The d1 and d2 terms that appear throughout Black-Scholes.

    Raises if S, K, T, or sigma are not strictly positive, since the formula is
    undefined there (those cases are handled as intrinsic value in ``price``).
    """
    if S <= 0 or K <= 0:
        raise ValueError("Spot and strike must be positive.")
    if T <= 0 or sigma <= 0:
        raise ValueError("Use price() for the T<=0 or sigma<=0 edge cases.")
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes price of a European call or put.

    ``option_type`` is "call" or "put". At or past expiry (T <= 0), or with
    zero volatility, the price collapses to discounted intrinsic value.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # Edge cases where the closed form is undefined: fall back to the value the
    # option must have by no-arbitrage.
    if T <= 0 or sigma <= 0:
        forward = S * math.exp(-q * T) - K * math.exp(-r * T)
        if option_type == "call":
            return max(forward, 0.0) if T > 0 else max(S - K, 0.0)
        return max(-forward, 0.0) if T > 0 else max(K - S, 0.0)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_div = math.exp(-q * T)
    disc_rate = math.exp(-r * T)

    if option_type == "call":
        return S * disc_div * _norm_cdf(d1) - K * disc_rate * _norm_cdf(d2)
    return K * disc_rate * _norm_cdf(-d2) - S * disc_div * _norm_cdf(-d1)

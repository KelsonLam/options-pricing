"""The Greeks: how the option price moves as the inputs move.

Each is the analytic Black-Scholes partial derivative, which is both exact and
far cheaper than bumping an input and re-pricing. Conventions:

    delta   per 1.00 change in the underlying
    gamma   change in delta per 1.00 change in the underlying
    vega    per 1.00 (that is, 100 percentage points) change in volatility
    theta   per year of time decay
    rho     per 1.00 change in the interest rate

Vega and theta are often quoted per 1% vol and per calendar day. The helpers
return the raw per-unit figures; the README explains the rescaling.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from .black_scholes import d1_d2

_N = NormalDist()


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def delta(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_div = math.exp(-q * T)
    if option_type.lower() == "call":
        return disc_div * _N.cdf(d1)
    return disc_div * (_N.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0) -> float:
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0) -> float:
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_div = math.exp(-q * T)
    disc_rate = math.exp(-r * T)
    term1 = -S * disc_div * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    if option_type.lower() == "call":
        return (term1
                - r * K * disc_rate * _N.cdf(d2)
                + q * S * disc_div * _N.cdf(d1))
    return (term1
            + r * K * disc_rate * _N.cdf(-d2)
            - q * S * disc_div * _N.cdf(-d1))


def rho(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_rate = math.exp(-r * T)
    if option_type.lower() == "call":
        return K * T * disc_rate * _N.cdf(d2)
    return -K * T * disc_rate * _N.cdf(-d2)


def all_greeks(S, K, T, r, sigma, option_type="call", q=0.0) -> dict[str, float]:
    """Every Greek in one dictionary."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }

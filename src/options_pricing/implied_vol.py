"""Back out the volatility implied by a market price.

Pricing takes volatility and returns a price. Implied volatility runs that
backwards: given the price the market is actually paying, what volatility does
Black-Scholes need to reproduce it? It is the number traders quote, because it
is the one input the formula cannot observe directly.

The solver tries Newton-Raphson (fast, using vega as the derivative) and falls
back to bisection if Newton wanders out of bounds, which keeps it robust even
for deep in- or out-of-the-money options where vega is tiny.
"""

from __future__ import annotations

from .black_scholes import price
from .greeks import vega


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """Volatility that makes the Black-Scholes price equal ``market_price``.

    Raises ValueError if the price is below intrinsic value (no volatility can
    explain it) or the search fails to converge.
    """
    intrinsic = price(S, K, T, r, 1e-9, option_type, q)
    if market_price < intrinsic - tol:
        raise ValueError(
            "Market price is below intrinsic value, so no implied "
            "volatility exists."
        )

    low, high = 1e-6, 5.0
    sigma = 0.2   # a sensible starting guess

    for _ in range(max_iter):
        model_price = price(S, K, T, r, sigma, option_type, q)
        diff = model_price - market_price
        if abs(diff) < tol:
            return sigma

        v = vega(S, K, T, r, sigma, q)
        if v > 1e-8:
            step = diff / v
            new_sigma = sigma - step
            if low < new_sigma < high:
                sigma = new_sigma
                continue

        # Newton stepped out of bounds or vega vanished: tighten a bracket.
        if diff > 0:
            high = sigma
        else:
            low = sigma
        sigma = 0.5 * (low + high)

    raise ValueError("Implied volatility did not converge.")

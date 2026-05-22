"""Charts that make the model tangible.

Two views: how the option's value bends above its intrinsic payoff (the "time
value" curve), and how the Greeks change as the underlying moves. Matplotlib
only; each function returns the Figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import black_scholes, greeks


def plot_price_vs_spot(
    K, T, r, sigma, option_type="call", q=0.0,
    spot_range=None, title=None,
):
    """Option value against spot, with the intrinsic payoff for comparison."""
    if spot_range is None:
        spot_range = np.linspace(0.5 * K, 1.5 * K, 200)
    values = [black_scholes.price(s, K, T, r, sigma, option_type, q) for s in spot_range]
    if option_type.lower() == "call":
        intrinsic = np.maximum(spot_range - K, 0.0)
    else:
        intrinsic = np.maximum(K - spot_range, 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spot_range, values, label="Option value", color="tab:blue", linewidth=2)
    ax.plot(spot_range, intrinsic, label="Intrinsic value at expiry",
            color="tab:gray", linestyle="--")
    ax.axvline(K, color="tab:red", alpha=0.4, linestyle=":", label="Strike")
    ax.set_title(title or f"{option_type.title()} value vs spot")
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Option value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_greeks_vs_spot(
    K, T, r, sigma, option_type="call", q=0.0, spot_range=None,
):
    """Delta, gamma, vega, and theta across a range of spot prices."""
    if spot_range is None:
        spot_range = np.linspace(0.5 * K, 1.5 * K, 200)

    delta = [greeks.delta(s, K, T, r, sigma, option_type, q) for s in spot_range]
    gamma = [greeks.gamma(s, K, T, r, sigma, q) for s in spot_range]
    vega = [greeks.vega(s, K, T, r, sigma, q) for s in spot_range]
    theta = [greeks.theta(s, K, T, r, sigma, option_type, q) for s in spot_range]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], delta, "Delta"),
        (axes[0, 1], gamma, "Gamma"),
        (axes[1, 0], vega, "Vega (per 1.00 vol)"),
        (axes[1, 1], theta, "Theta (per year)"),
    ]
    for ax, series, name in panels:
        ax.plot(spot_range, series, color="tab:blue")
        ax.axvline(K, color="tab:red", alpha=0.4, linestyle=":")
        ax.set_title(name)
        ax.set_xlabel("Spot price")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{option_type.title()} Greeks vs spot", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path

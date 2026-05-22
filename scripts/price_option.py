"""Command line entry point: price an option and show its Greeks.

Examples
--------
Price a call (spot 100, strike 100, 1 year, 5% rate, 20% vol)::

    python scripts/price_option.py --spot 100 --strike 100 --t 1 --r 0.05 --sigma 0.20

Price a put, cross-check against Monte Carlo, and save the charts::

    python scripts/price_option.py --type put --mc --save-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from options_pricing import black_scholes, greeks, plotting
from options_pricing.montecarlo import monte_carlo_price


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Price a European option with Black-Scholes.")
    p.add_argument("--spot", type=float, default=100.0)
    p.add_argument("--strike", type=float, default=100.0)
    p.add_argument("--t", type=float, default=1.0, help="Years to expiry.")
    p.add_argument("--r", type=float, default=0.05, help="Risk-free rate.")
    p.add_argument("--sigma", type=float, default=0.20, help="Volatility.")
    p.add_argument("--q", type=float, default=0.0, help="Dividend yield.")
    p.add_argument("--type", choices=["call", "put"], default="call")
    p.add_argument("--mc", action="store_true", help="Cross-check with Monte Carlo.")
    p.add_argument("--save-plots", action="store_true", help="Write charts to results/.")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    bs = black_scholes.price(a.spot, a.strike, a.t, a.r, a.sigma, a.type, a.q)
    g = greeks.all_greeks(a.spot, a.strike, a.t, a.r, a.sigma, a.type, a.q)

    print(f"\nBlack-Scholes {a.type} price: {bs:,.4f}")
    print("-" * 34)
    for name, value in g.items():
        print(f"{name:<7} {value:>12,.4f}")

    if a.mc:
        mc, se = monte_carlo_price(a.spot, a.strike, a.t, a.r, a.sigma, a.type, a.q, seed=0)
        print(f"\nMonte Carlo price: {mc:,.4f}  (standard error {se:.4f})")
        print(f"Difference from formula: {abs(mc - bs):,.4f}")

    if a.save_plots:
        f1 = plotting.plot_price_vs_spot(a.strike, a.t, a.r, a.sigma, a.type, a.q)
        f2 = plotting.plot_greeks_vs_spot(a.strike, a.t, a.r, a.sigma, a.type, a.q)
        out1 = plotting.save_figure(f1, "results/price_vs_spot.png")
        out2 = plotting.save_figure(f2, "results/greeks_vs_spot.png")
        print(f"\nSaved charts to {out1} and {out2}")


if __name__ == "__main__":
    main()

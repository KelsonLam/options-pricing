"""Tests for the pricing formula, the Greeks, implied vol, and the MC check.

These lean on relationships that must hold regardless of the inputs (put-call
parity, sign constraints on the Greeks) plus a couple of textbook values.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from options_pricing import black_scholes as bs
from options_pricing import greeks
from options_pricing.implied_vol import implied_volatility
from options_pricing.montecarlo import monte_carlo_price

# A standard reference case.
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_atm_call_matches_known_value():
    # The textbook value for this at-the-money call is about 10.4506.
    call = bs.price(S, K, T, R, SIGMA, "call")
    assert call == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity():
    call = bs.price(S, K, T, R, SIGMA, "call")
    put = bs.price(S, K, T, R, SIGMA, "put")
    # C - P should equal S - K * exp(-rT) with no dividends.
    lhs = call - put
    rhs = S - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_call_price_increases_with_volatility():
    low = bs.price(S, K, T, R, 0.10, "call")
    high = bs.price(S, K, T, R, 0.40, "call")
    assert high > low


def test_intrinsic_value_at_expiry():
    assert bs.price(120, 100, 0.0, R, SIGMA, "call") == pytest.approx(20.0)
    assert bs.price(80, 100, 0.0, R, SIGMA, "put") == pytest.approx(20.0)
    assert bs.price(90, 100, 0.0, R, SIGMA, "call") == pytest.approx(0.0)


def test_delta_bounds():
    call_delta = greeks.delta(S, K, T, R, SIGMA, "call")
    put_delta = greeks.delta(S, K, T, R, SIGMA, "put")
    assert 0.0 < call_delta < 1.0
    assert -1.0 < put_delta < 0.0
    # delta_call - delta_put = 1 with no dividends
    assert call_delta - put_delta == pytest.approx(1.0, abs=1e-9)


def test_gamma_and_vega_positive():
    assert greeks.gamma(S, K, T, R, SIGMA) > 0
    assert greeks.vega(S, K, T, R, SIGMA) > 0


def test_implied_vol_round_trip():
    # Price with a known vol, then recover it from the price.
    target = 0.27
    market = bs.price(S, K, T, R, target, "call")
    recovered = implied_volatility(market, S, K, T, R, "call")
    assert recovered == pytest.approx(target, abs=1e-5)


def test_monte_carlo_agrees_with_formula():
    formula = bs.price(S, K, T, R, SIGMA, "call")
    mc, se = monte_carlo_price(S, K, T, R, SIGMA, "call", n_paths=400_000, seed=1)
    # Within a few standard errors of the closed form.
    assert abs(mc - formula) < 4 * se

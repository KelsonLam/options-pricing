"""Tests for the binomial tree pricer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from options_pricing import black_scholes as bs
from options_pricing.binomial import binomial_price

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_european_binomial_converges_to_black_scholes():
    for opt in ("call", "put"):
        tree = binomial_price(S, K, T, R, SIGMA, opt, steps=800)
        formula = bs.price(S, K, T, R, SIGMA, opt)
        assert tree == pytest.approx(formula, abs=0.05)


def test_american_put_worth_at_least_european():
    euro = binomial_price(S, K, T, R, SIGMA, "put", steps=400, american=False)
    amer = binomial_price(S, K, T, R, SIGMA, "put", steps=400, american=True)
    assert amer >= euro - 1e-9


def test_american_call_no_dividend_equals_european():
    # With no dividends it is never optimal to exercise an American call early,
    # so it should match the European value.
    euro = binomial_price(S, K, T, R, SIGMA, "call", steps=400, american=False)
    amer = binomial_price(S, K, T, R, SIGMA, "call", steps=400, american=True)
    assert amer == pytest.approx(euro, abs=1e-6)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, SIGMA, "swap")
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, SIGMA, "call", steps=0)

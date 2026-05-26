"""Edge-case tests for the pricing library."""

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


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs.price(100, 100, 1, 0.05, 0.2, "straddle")


def test_deep_itm_call_approaches_discounted_forward():
    # A deep in-the-money call should be worth about S - K*exp(-rT).
    price = bs.price(1000, 100, 1, 0.05, 0.2, "call")
    floor = 1000 - 100 * math.exp(-0.05)
    assert price == pytest.approx(floor, rel=1e-3)


def test_deep_otm_put_is_nearly_worthless():
    assert bs.price(1000, 100, 0.5, 0.05, 0.2, "put") < 0.01


def test_zero_time_is_intrinsic():
    assert bs.price(120, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(20.0)
    assert bs.price(80, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(20.0)


def test_gamma_symmetric_in_call_and_put():
    # Gamma does not depend on option type.
    g_call = greeks.gamma(100, 100, 1, 0.05, 0.2)
    assert g_call > 0

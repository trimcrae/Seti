"""Analytic synthetic witness, not an empirical sky false-positive rate."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from seti.parallax4 import deepvet as DV


def fixed_bls(monkeypatch, *, period=1.0, duration=0.1, transit_time=0.0):
    """Condition on a fixed independent ZTF ephemeris to test Gaia phasing."""
    class FixedBLS:
        def __init__(self, *args, **kwargs):
            pass

        def power(self, *args, **kwargs):
            n = 128
            return SimpleNamespace(
                power=np.r_[100.0, np.zeros(n - 1)],
                period=np.full(n, period), transit_time=np.full(n, transit_time),
                duration=np.full(n, duration), depth=np.full(n, 0.2),
            )

    monkeypatch.setattr("astropy.timeseries.BoxLeastSquares", FixedBLS)
    return pd.DataFrame({
        "mjd": DV.GAIA_T0_MJD + np.linspace(0, 40, 80),
        "mag": 15.0, "magerr": 0.01, "filtercode": "zr", "oid": "123456789",
    })


def test_baseline_alias_bound_counterexample(monkeypatch):
    z = fixed_bls(monkeypatch)
    result = DV.ztf_eclipse_test(z, list(range(6)))
    old_bound = 3 * 0.32 ** 6
    half_period_chance = 0.52 ** 6
    assert result["phase_p_chance"] == pytest.approx(old_bound)
    assert old_bound < 0.01 < half_period_chance
    assert result["ztf_class"] == "ZTF_ECLIPSING_PHASED"
    print(f"BASELINE_WITNESS old={old_bound:.12f} half_period={half_period_chance:.12f}")

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


def test_alias_specific_bound_prevents_understated_veto(monkeypatch):
    z = fixed_bls(monkeypatch)
    result = DV.ztf_eclipse_test(z, list(range(6)))
    old_bound = 3 * 0.32 ** 6
    half_period_chance = 0.52 ** 6
    corrected_bound = 0.32 ** 6 + 0.22 ** 6 + 0.52 ** 6
    assert old_bound < 0.01 < half_period_chance <= corrected_bound
    assert result["phase_p_chance"] == pytest.approx(0.020957731392)
    assert result["ztf_class"] == "ZTF_ECLIPSING_PHASE_AMBIGUOUS"
    assert result["phase_p_chance_method"] == "UNION_BOUND_FIXED_ZTF_EPHEMERIS"
    assert result["phase_p_chance_assumptions"] == "INDEPENDENT_UNIFORM_GAIA_EPISODE_PHASES"
    assert result["phase_p_chance_is_empirical"] is False
    trials = result["phase_window_trials"]
    assert [x["period_d"] for x in trials] == [1.0, 2.0, 0.5]
    assert [x["accepted_phase_fraction"] for x in trials] == pytest.approx([0.32, 0.22, 0.52])
    assert [x["all_dips_probability"] for x in trials] == pytest.approx(
        [0.32 ** 6, 0.22 ** 6, 0.52 ** 6])
    assert all(x["n_dips"] == 6 for x in trials)


@pytest.mark.parametrize(("duration", "tol"), [
    (0.01, 0.0), (0.1, 0.03), (0.24, 0.0), (0.25, 0.0),
    (0.5, 0.0), (0.9, 0.03), (0.1, 0.25),
])
def test_each_fraction_matches_actual_circular_acceptance(duration, tol):
    # Deterministic quadrature of the production acceptance sets, not a
    # Monte Carlo sky false-positive estimate. Midpoints avoid edge counting.
    _, trials = DV._ztf_phase_chance_bound(1.0, duration, tol, 6)
    grid_size = 100000
    phases = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    for trial in trials:
        half = duration / trial["period_d"] / 2 + tol
        accepted = ((np.abs(phases) <= half)
                    | (np.abs(np.abs(phases) - 0.5) <= half))
        assert trial["accepted_phase_fraction"] == pytest.approx(
            accepted.mean(), abs=2 / grid_size)
        assert 0 <= trial["accepted_phase_fraction"] <= 1


def test_overlap_saturates_each_trial_before_exponentiation():
    bound, trials = DV._ztf_phase_chance_bound(1.0, 0.5, 0.0, 20)
    assert [x["accepted_phase_fraction"] for x in trials] == [1.0, 0.5, 1.0]
    assert trials[1]["all_dips_probability"] == 0.5 ** 20
    assert bound == 1.0


@pytest.mark.parametrize("n_dips", [0, 1, 2, 6, 10, 30])
def test_union_bound_matches_sum_without_alias_independence(n_dips):
    bound, trials = DV._ztf_phase_chance_bound(1.0, 0.1, 0.03, n_dips)
    assert bound == pytest.approx(min(
        1.0, 0.32 ** n_dips + 0.22 ** n_dips + 0.52 ** n_dips))
    assert bound >= max(x["all_dips_probability"] for x in trials)


@pytest.mark.parametrize(("period", "duration", "tol", "n_dips"), [
    (0, 0.1, 0.03, 6), (-1, 0.1, 0.03, 6), (np.inf, 0.1, 0.03, 6),
    (np.nan, 0.1, 0.03, 6), (1, 0, 0.03, 6), (1, -0.1, 0.03, 6),
    (1, np.nan, 0.03, 6), (1, 0.1, -0.03, 6), (1, 0.1, np.inf, 6),
    (1, 0.1, np.nan, 6), (1, 0.1, 0.03, -1), (1, 0.1, 0.03, 1.5),
    (1, 0.1, 0.03, True),
])
def test_invalid_bound_geometry_refuses(period, duration, tol, n_dips):
    with pytest.raises(ValueError):
        DV._ztf_phase_chance_bound(period, duration, tol, n_dips)


def test_no_gaia_episodes_does_not_promote_veto(monkeypatch):
    result = DV.ztf_eclipse_test(fixed_bls(monkeypatch), [])
    assert result["ztf_class"] == "ZTF_PERIODIC_NOT_PHASED"
    assert result["phase_p_chance"] == 1.0
    assert all(x["n_dips"] == 0 for x in result["phase_window_trials"])


def test_many_matching_episodes_still_pass_conditional_bound(monkeypatch):
    result = DV.ztf_eclipse_test(fixed_bls(monkeypatch), list(range(10)))
    assert result["ztf_class"] == "ZTF_ECLIPSING_PHASED"
    assert result["phase_p_chance"] < 0.01
    assert result["gaia_dip_phases"] == [0.0] * 10


def test_small_bound_without_matching_phase_does_not_promote(monkeypatch):
    result = DV.ztf_eclipse_test(fixed_bls(monkeypatch), [k + 0.17 for k in range(10)])
    assert result["phase_p_chance"] < 0.01
    assert result["ztf_class"] == "ZTF_PERIODIC_NOT_PHASED"


def test_alias_acceptance_and_bound_ignore_epoch_order(monkeypatch):
    z = fixed_bls(monkeypatch)
    dips = [k + 0.25 for k in range(10)]
    forward = DV.ztf_eclipse_test(z, dips)
    backward = DV.ztf_eclipse_test(z, dips[::-1])
    assert forward["ztf_class"] == backward["ztf_class"] == "ZTF_ECLIPSING_PHASED"
    assert forward["phase_window_trials"] == backward["phase_window_trials"]
    assert forward["ztf_period_d"] == backward["ztf_period_d"] == 0.5

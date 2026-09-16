"""Offline test suite for GROWTH **stage 2** --- the measured TESS depth (S57).

No network anywhere (``conftest.py`` raises on any socket); every fetch in
:mod:`seti.growth.stage2` takes an injectable callable and every test injects
one.  What this suite is the gate for:

* **the epoch conversion**, BKJD -> BTJD, against a hand-worked value.  This is
  the one arithmetic that silently destroys a fold if it is wrong: a 2167-day
  error puts the transit anywhere in phase and the depth comes out zero.
* **the injected depth is recovered** on a synthetic light curve --- the
  load-bearing test.  Everything else in this module is bookkeeping around a
  number, and if that number is not the injected one the module measures
  nothing.  Recovery is required to within the fit's OWN quoted error, so a
  fitter that hides bias by inflating its errors does not pass either: the
  error is separately required to be small compared with the depth.
* **a smeared FFI cadence is flagged**: a 30-minute integration on a 2-hour
  transit is a different measurement from a 2-minute one, and the record has to
  say so rather than quietly reporting a shallow depth as a depth.
* **an odd-even difference is detected** on a synthetic eclipsing binary --- the
  classic EB signature, and the only handle stage 2 has on the mechanism that
  actually can make a TESS transit DEEPER (a signal originating on a different
  star inside the pixel).
* **a failed fetch is ``QUERY_FAILED``** and never a depth, and
  ``QUERY_RETURNED_ZERO_ROWS`` stays a different fact.
* **the three-way verdict** returns the right branch for each of: the measured
  depth agrees with Kepler (the TOI value was wrong), agrees with the TOI (the
  depth really changed), agrees with neither.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.growth.stage2 import (
    BKJD_MINUS_BTJD,
    BKJD_OFFSET,
    BTJD_OFFSET,
    MATCH_BOTH,
    MATCH_KEPLER,
    MATCH_NEITHER,
    MATCH_TOI,
    MATCH_UNMEASURED,
    RUN_CONFIRMED,
    RUN_NO_DATA,
    RUN_REFUTED,
    RUN_UNRESOLVED,
    UNMEASURED_NO_EPHEMERIS,
    UNMEASURED_NO_REFERENCE,
    UNMEASURED_QUERY_FAILED,
    UNMEASURED_ZERO_ROWS,
    CompareParams,
    Deadline,
    FitParams,
    MastParams,
    binned_fold,
    bkjd_to_bjd,
    bkjd_to_btjd,
    btjd_to_bkjd,
    combine_transit_depths,
    compare_three_depths,
    core_half_width,
    fetch_koi_ephemerides,
    fetch_lightcurves,
    fetch_toi_depths,
    fit_transits,
    fold,
    load_shortlist,
    mast_probe,
    measure_one,
    measure_target,
    propagate_epoch,
    run_verdict,
    stage2_assess,
    stage2_measure,
    stage2_probe,
    stage2_run,
    synth_lightcurve,
    trapezoid_transit,
)
from seti.growth.stage2 import (
    main as stage2_main,
)

# A short-period planet like the stage-1 candidate: P = 2.05 d, T14 = 2.0 h.
P = 2.0523499
T14_H = 2.0
T14_D = T14_H / 24.0
T0_BKJD = 170.0


# ---------------------------------------------------------------------------
# 1. The epoch conversion (hand-worked)
# ---------------------------------------------------------------------------
def test_bkjd_to_btjd_against_a_hand_worked_value():
    """BKJD 170.0 is BJD 2455003.0 is BTJD -1997.0.  By hand, then by code."""
    # by hand
    assert bkjd_to_bjd(170.0) == pytest.approx(2455003.0, abs=1e-9)
    assert 2455003.0 - BTJD_OFFSET == pytest.approx(-1997.0, abs=1e-9)
    # by code
    assert bkjd_to_btjd(170.0) == pytest.approx(-1997.0, abs=1e-9)
    # the constant itself: BJD - 2454833 -> BJD - 2457000 is a shift of -2167 d
    assert BKJD_MINUS_BTJD == pytest.approx(-2167.0, abs=1e-12)
    assert BKJD_OFFSET - BTJD_OFFSET == pytest.approx(BKJD_MINUS_BTJD, abs=1e-12)
    # and the inverse round-trips, on scalars and on arrays
    assert btjd_to_bkjd(bkjd_to_btjd(1234.5)) == pytest.approx(1234.5, abs=1e-9)
    arr = np.array([0.0, 170.0, 1500.25])
    assert np.allclose(bkjd_to_btjd(arr), arr - 2167.0)
    assert np.allclose(btjd_to_bkjd(bkjd_to_btjd(arr)), arr)


def test_a_wrong_epoch_conversion_would_destroy_the_fold():
    """The guard the hand value exists for: forget the shift and the depth goes."""
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.01,
                           exptime_s=120.0, n_transits=10, noise_ppm=200.0)
    right = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=T14_D)
    wrong = measure_target([sec], period_days=P, t0_btjd=T0_BKJD, duration_days=T14_D)
    assert right["depth_ppm"] == pytest.approx(10000.0, rel=0.05)
    # the unconverted epoch is 2167 d away: nothing at the fitted phase is a transit
    assert (not np.isfinite(wrong["depth_ppm"])) or abs(wrong["depth_ppm"]) < 1000.0


def test_epoch_propagation_counts_periods_and_accumulates_the_period_error():
    """A KOI epoch carried to the TESS window, with its accumulated uncertainty."""
    t0 = bkjd_to_btjd(T0_BKJD)              # -1997.0 BTJD
    prop = propagate_epoch(t0, P, 1500.0, t0_err_days=1e-4, period_err_days=1e-7)
    n_expected = int(round((1500.0 - t0) / P))
    assert prop["n_epochs"] == n_expected
    assert abs(prop["t0_btjd"] - 1500.0) <= 0.5 * P + 1e-9
    sig = math.sqrt(1e-4**2 + (n_expected * 1e-7) ** 2)
    assert prop["sigma_days"] == pytest.approx(sig, rel=1e-9)
    assert prop["sigma_minutes"] == pytest.approx(sig * 1440.0, rel=1e-9)
    # ~1700 epochs at 1e-7 d is seconds, not a fraction of a 2-hour transit
    assert prop["sigma_minutes"] < 0.1 * T14_H * 60.0
    # a missing ephemeris does not fabricate one
    bad = propagate_epoch(float("nan"), P, 1500.0)
    assert not np.isfinite(bad["t0_btjd"])


def test_fold_returns_integer_epochs_and_signed_offsets():
    t0 = -1997.0
    t = np.array([t0 - 0.01, t0 + 0.01, t0 + P - 0.02, t0 + 5 * P + 0.03])
    ep, dt = fold(t, P, t0)
    assert list(ep) == [0, 0, 1, 5]
    assert np.allclose(dt, [-0.01, 0.01, -0.02, 0.03])


# ---------------------------------------------------------------------------
# 2. THE LOAD-BEARING TEST: a known injected depth is recovered
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depth_ppm", [3000.0, 10000.0, 34476.0])
def test_injected_depth_is_recovered_within_its_own_quoted_error(depth_ppm):
    """The measurement is what it claims to be, at three depths.

    Recovery is required within 3x the fit's OWN quoted error, AND the quoted
    error is required to be a small fraction of the depth --- otherwise a
    fitter could "pass" by quoting an error large enough to cover any bias.
    """
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D,
                           depth=depth_ppm * 1e-6, exptime_s=120.0, n_transits=14,
                           noise_ppm=400.0, seed=3)
    m = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert m["n_transits"] == 14
    assert np.isfinite(m["depth_ppm"]) and np.isfinite(m["depth_err_ppm"])
    assert m["depth_err_ppm"] > 0
    # the error is informative, not a blanket
    assert m["depth_err_ppm"] < 0.15 * depth_ppm
    # and the depth is right, to within it
    assert abs(m["depth_ppm"] - depth_ppm) < 3.0 * m["depth_err_ppm"]
    # the bootstrap was used (14 transits >= min_transits_for_bootstrap) and said so
    assert m["depth_err_method"] == "bootstrap"
    assert np.isfinite(m["depth_err_analytic_ppm"])
    assert np.isfinite(m["depth_err_bootstrap_ppm"])
    # the two error estimates agree to a factor of a few: neither is nonsense
    assert 0.25 < m["depth_err_bootstrap_ppm"] / m["depth_err_analytic_ppm"] < 4.0


def test_the_quoted_error_is_calibrated_over_many_noise_realisations():
    """The strongest form of the load-bearing test: the PULL distribution.

    One realisation can be lucky.  Over 30 independent noise draws the pull
    ``(fitted - injected)/quoted_error`` must have mean ~0 (the fitter is
    unbiased) and spread ~1 (the quoted error is the right size).  A fitter
    that passed the single-realisation test by quoting a generous error would
    show a pull spread well below 1 here, and a biased one a non-zero mean.
    """
    t0 = bkjd_to_btjd(T0_BKJD)
    params = FitParams(bootstrap_draws=600)
    pulls = []
    for seed in range(30):
        sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.010,
                               exptime_s=120.0, n_transits=14, noise_ppm=400.0, seed=seed)
        m = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=T14_D,
                           params=params)
        pulls.append((m["depth_ppm"] - 10000.0) / m["depth_err_ppm"])
    pulls = np.array(pulls)
    assert abs(pulls.mean()) < 0.5, f"biased fitter: mean pull {pulls.mean():+.3f}"
    assert 0.6 < pulls.std(ddof=1) < 1.6, f"mis-sized error: pull sd {pulls.std(ddof=1):.3f}"


def test_injected_depth_is_recovered_across_several_sectors_and_per_sector():
    """The combined depth AND each sector's depth recover the injection."""
    t0 = bkjd_to_btjd(T0_BKJD)
    secs = []
    for i, s in enumerate((14, 15, 41)):
        secs.append(synth_lightcurve(period_days=P, t0_btjd=t0 + i * 400 * P,
                                     duration_days=T14_D, depth=0.012, exptime_s=120.0,
                                     n_transits=9, noise_ppm=400.0, sector=s, seed=20 + i))
    m = measure_target(secs, period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert m["n_sectors"] == 3 and m["n_sectors_measured"] == 3
    assert m["sector_list"] == "14,15,41"
    assert abs(m["depth_ppm"] - 12000.0) < 3.0 * m["depth_err_ppm"]
    for _, r in m["sectors"].iterrows():
        assert abs(r["depth_ppm"] - 12000.0) < 4.0 * r["depth_err_ppm"]
    # consistent sectors must NOT raise the sector-scatter flag
    assert "sector_scatter" not in str(m["flags"])


def test_a_sector_that_disagrees_raises_sector_scatter_not_growth():
    """A depth that differs between sectors is a systematic, and is named one."""
    t0 = bkjd_to_btjd(T0_BKJD)
    a = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.010,
                         exptime_s=120.0, n_transits=10, noise_ppm=200.0, sector=14, seed=5)
    b = synth_lightcurve(period_days=P, t0_btjd=t0 + 400 * P, duration_days=T14_D, depth=0.020,
                         exptime_s=120.0, n_transits=10, noise_ppm=200.0, sector=15, seed=6)
    m = measure_target([a, b], period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert "sector_scatter" in str(m["flags"])
    assert m["sector_scatter_chi2_per_dof"] > 3.0


def test_the_baseline_is_fitted_with_the_transit_masked():
    """A sloping baseline does not bleed into the depth."""
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.01,
                           exptime_s=120.0, n_transits=12, noise_ppm=200.0, seed=9)
    slope = 0.02          # 2 % per day of trend, far larger than the 1 % transit
    sec["flux"] = sec["flux"] * (1.0 + slope * (sec["time"] - sec["time"][0]))
    m = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert abs(m["depth_ppm"] - 10000.0) < 3.0 * m["depth_err_ppm"]


def test_per_transit_rows_say_why_a_transit_was_skipped():
    """A transit with no usable baseline is skipped and names the reason."""
    t0 = 0.0
    # one transit's window, truncated so only the leading side has baseline
    t = np.linspace(-0.15, 0.02, 200)
    f = trapezoid_transit(t, T14_D, 0.01)
    rows = fit_transits(t, f, None, period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert len(rows) == 1
    assert rows[0]["used"] is False
    assert rows[0]["reject_reason"] == "baseline_on_one_side_only"
    assert not np.isfinite(rows[0]["depth"])
    # nothing usable -> no depth, and the combiner says so rather than guessing
    c = combine_transit_depths(rows)
    assert c["n_transits"] == 0
    assert not np.isfinite(c["depth_ppm"])
    assert c["depth_err_method"] == "none"


# ---------------------------------------------------------------------------
# 3. Smearing
# ---------------------------------------------------------------------------
def test_a_30_minute_ffi_cadence_on_a_2_hour_transit_is_flagged_as_smeared():
    """30 min > 0.2 * 120 min, so the record says `smeared` --- per sector and per target."""
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.01,
                           exptime_s=1800.0, n_transits=30, noise_ppm=300.0,
                           sector=20, author="QLP", seed=4)
    m = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert m["smeared"] is True
    assert "smeared" in str(m["flags"])
    assert bool(m["sectors"].iloc[0]["smeared"]) is True
    assert m["sectors"].iloc[0]["exptime_over_duration"] == pytest.approx(0.25, rel=1e-6)
    assert m["sectors"].iloc[0]["author"] == "QLP"


def test_a_2_minute_cadence_on_the_same_transit_is_not_flagged():
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.01,
                           exptime_s=120.0, n_transits=12, noise_ppm=300.0, seed=4)
    m = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert m["smeared"] is False
    assert "smeared" not in str(m["flags"])


def test_core_window_shrinks_by_half_the_exposure_and_says_when_none_survives():
    """The flat core is the part no exposure straddles; when there is none, say so."""
    params = FitParams()
    half, no_flat = core_half_width(T14_D, 120.0 / 86400.0, params)
    assert no_flat is False
    assert half == pytest.approx(0.5 * 0.7 * T14_D - 60.0 / 86400.0, rel=1e-9)
    # a 30-minute cadence on a 20-minute transit leaves no flat bottom at all
    half2, no_flat2 = core_half_width(20.0 / 1440.0, 1800.0 / 86400.0, params)
    assert no_flat2 is True
    assert half2 > 0                      # the unshrunk window is used instead
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=20.0 / 1440.0, depth=0.01,
                           exptime_s=1800.0, n_transits=40, noise_ppm=200.0, seed=8)
    m = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=20.0 / 1440.0)
    assert m["depth_is_lower_bound"] is True
    assert "depth_is_lower_bound" in str(m["flags"])


# ---------------------------------------------------------------------------
# 4. Odd-even (the eclipsing-binary handle)
# ---------------------------------------------------------------------------
def test_an_odd_even_difference_is_detected_on_a_synthetic_eclipsing_binary():
    """Alternating 10,000 / 6,000 ppm eclipses: the classic EB signature."""
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.010,
                           odd_depth=0.006, exptime_s=120.0, n_transits=20, noise_ppm=300.0,
                           seed=12)
    m = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert m["n_transits_odd"] == 10 and m["n_transits_even"] == 10
    assert m["depth_even_ppm"] == pytest.approx(10000.0, rel=0.1)
    assert m["depth_odd_ppm"] == pytest.approx(6000.0, rel=0.15)
    assert m["odd_even_diff_ppm"] == pytest.approx(-4000.0, abs=800.0)
    assert m["odd_even_sigma"] > 3.0
    assert "odd_even_significant" in str(m["flags"])


def test_a_clean_planet_does_not_raise_the_odd_even_flag():
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.010,
                           exptime_s=120.0, n_transits=20, noise_ppm=300.0, seed=13)
    m = measure_target([sec], period_days=P, t0_btjd=t0, duration_days=T14_D)
    assert m["odd_even_sigma"] < 3.0
    assert "odd_even_significant" not in str(m["flags"])


def test_the_binned_fold_is_written_and_has_a_transit_in_it():
    t0 = bkjd_to_btjd(T0_BKJD)
    sec = synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D, depth=0.01,
                           exptime_s=120.0, n_transits=8, noise_ppm=200.0, seed=14)
    fdf = binned_fold(sec["time"], sec["flux"] / np.median(sec["flux"]), period_days=P,
                      t0_btjd=t0, duration_days=T14_D)
    assert len(fdf) > 20
    centre = fdf.iloc[(fdf["dt_hours"].abs()).argmin()]
    assert centre["flux"] == pytest.approx(0.99, abs=0.002)
    assert fdf["flux"].max() == pytest.approx(1.0, abs=0.005)


# ---------------------------------------------------------------------------
# 5. A failed fetch is QUERY_FAILED and NEVER a depth
# ---------------------------------------------------------------------------
def test_a_failed_lightcurve_fetch_is_query_failed_and_never_a_depth():
    def boom(_tic, **_kw):
        raise RuntimeError("MAST said 503")

    secs, status, _route = fetch_lightcurves(268924036, lc_fn=boom,
                                             params=MastParams(retries=2, retry_pause_s=0.0))
    assert status == UNMEASURED_QUERY_FAILED == "QUERY_FAILED"
    assert secs == []


def test_an_empty_lightcurve_answer_is_zero_rows_not_a_failure():
    """Two different facts: the service errored, versus it answered with nothing."""
    secs, status, _route = fetch_lightcurves(1, lc_fn=lambda _t, **_k: [],
                                             params=MastParams())
    assert status == UNMEASURED_ZERO_ROWS == "QUERY_RETURNED_ZERO_ROWS"
    assert status != UNMEASURED_QUERY_FAILED
    assert secs == []


def test_an_unfetchable_target_is_unmeasured_and_agrees_with_nothing():
    """The rule that must never be broken: no light curve, no verdict about depth."""
    def boom(_tic, **_kw):
        raise OSError("connection reset")

    koi = {"koi_period": P, "koi_time0bk": T0_BKJD, "koi_duration": T14_H,
           "koi_depth": 14280.6, "koi_depth_err1": 17.9, "koi_depth_err2": -17.9,
           "koi_impact": 0.437, "koi_steff": 5800.0, "koi_slogg": 4.5}
    toi = {"pl_trandep": 34476.0, "pl_trandeperr1": 2349.1, "pl_trandeperr2": -2349.1}
    rec, sec, fdf = measure_one({"kepoi_name": "K00897.01", "tic_id": 268924036}, koi, toi,
                                lc_fn=boom, mast=MastParams(retries=1, retry_pause_s=0.0),
                                fit=FitParams(),
                                compare=CompareParams())
    assert rec["verdict"] == MATCH_UNMEASURED
    assert rec["unmeasured_reason"] == UNMEASURED_QUERY_FAILED
    assert rec["lc_status"] == "QUERY_FAILED"
    assert rec["agrees_with_kepler"] is False and rec["agrees_with_toi"] is False
    assert not np.isfinite(rec["depth_measured_ppm"])
    assert len(sec) == 0 and len(fdf) == 0


def test_a_missing_ephemeris_is_unmeasured_and_the_light_curve_is_not_even_fetched():
    calls = []

    def lc(_tic, **_kw):
        calls.append(1)
        return []

    rec, _s, _f = measure_one({"kepoi_name": "K99999.01", "tic_id": 1}, {"koi_period": P},
                              {"pl_trandep": 1000.0}, lc_fn=lc, mast=MastParams(),
                              fit=FitParams(), compare=CompareParams())
    assert rec["verdict"] == MATCH_UNMEASURED
    assert rec["unmeasured_reason"] == UNMEASURED_NO_EPHEMERIS
    assert rec["lc_status"] == "not_attempted"
    assert calls == []


def test_an_exhausted_budget_stops_the_fetch_and_is_recorded_not_hidden():
    spent = Deadline(budget_s=0.0)
    calls = []

    def lc(_tic, **_kw):
        calls.append(1)
        return []

    secs, status, _r = fetch_lightcurves(1, lc_fn=lc, params=MastParams(), deadline=spent)
    assert status == UNMEASURED_QUERY_FAILED
    assert secs == [] and calls == []


def test_a_failed_ephemeris_query_is_query_failed_and_an_empty_one_is_zero_rows():
    def boom(_adql):
        raise RuntimeError("archive 500")

    df, st = fetch_koi_ephemerides(["K00897.01"], query_fn=boom)
    assert st == UNMEASURED_QUERY_FAILED and not len(df)
    df, st = fetch_koi_ephemerides(["K00897.01"], query_fn=lambda _a: pd.DataFrame())
    assert st == UNMEASURED_ZERO_ROWS and not len(df)
    df, st = fetch_toi_depths([268924036], query_fn=boom)
    assert st == UNMEASURED_QUERY_FAILED
    df, st = fetch_toi_depths([268924036],
                              query_fn=lambda _a: pd.DataFrame({"toi": [4490.01],
                                                                "tid": [268924036]}))
    assert st == "OK" and len(df) == 1


def test_the_ephemeris_query_asks_for_the_epoch_column():
    """``koi_time0bk`` is not in the stage-1 column list; stage 2 must pull it."""
    seen = {}

    def q(adql):
        seen["adql"] = adql
        return pd.DataFrame({"kepoi_name": ["K00897.01"], "koi_time0bk": [170.0]})

    fetch_koi_ephemerides(["K00897.01"], query_fn=q)
    assert "koi_time0bk" in seen["adql"]
    assert "koi_period" in seen["adql"]
    assert "K00897.01" in seen["adql"]


# ---------------------------------------------------------------------------
# 6. THE THREE-WAY VERDICT
# ---------------------------------------------------------------------------
KEPLER_PPM, KEPLER_ERR = 14280.6, 17.9
TOI_PPM, TOI_ERR = 34476.0, 2349.1


def test_three_way_verdict_measured_depth_matches_kepler_kills_the_candidate():
    """Case 1: the measurement lands on the Kepler value -> the TOI number was wrong."""
    out = compare_three_depths(14300.0, 200.0, KEPLER_PPM, KEPLER_ERR, TOI_PPM, TOI_ERR,
                               ld_band_ratio=1.0)
    assert out["verdict"] == MATCH_KEPLER
    assert out["agrees_with_kepler"] is True and out["agrees_with_toi"] is False
    assert abs(out["z_vs_kepler"]) < 3.0 and abs(out["z_vs_toi"]) >= 3.0
    assert out["references_separated"] is True


def test_three_way_verdict_measured_depth_matches_toi_keeps_the_candidate_alive():
    """Case 2: the measurement lands on the deep value -> the depth really changed."""
    out = compare_three_depths(34000.0, 800.0, KEPLER_PPM, KEPLER_ERR, TOI_PPM, TOI_ERR,
                               ld_band_ratio=1.0)
    assert out["verdict"] == MATCH_TOI
    assert out["agrees_with_toi"] is True and out["agrees_with_kepler"] is False
    assert out["z_vs_kepler"] > 3.0


def test_three_way_verdict_measured_depth_matches_neither_says_so_and_does_not_pick():
    """Case 3: outside both.  Reported, not resolved."""
    out = compare_three_depths(22000.0, 300.0, KEPLER_PPM, KEPLER_ERR, TOI_PPM, TOI_ERR,
                               ld_band_ratio=1.0)
    assert out["verdict"] == MATCH_NEITHER
    assert out["agrees_with_kepler"] is False and out["agrees_with_toi"] is False
    assert np.isfinite(out["z_vs_kepler"]) and np.isfinite(out["z_vs_toi"])


def test_a_measurement_too_imprecise_to_separate_the_references_is_not_a_pick():
    """The fourth branch: 'both' is an admission, never a choice between them."""
    out = compare_three_depths(20000.0, 40000.0, KEPLER_PPM, KEPLER_ERR, TOI_PPM, TOI_ERR,
                               ld_band_ratio=1.0)
    assert out["verdict"] == MATCH_BOTH
    assert out["agrees_with_kepler"] and out["agrees_with_toi"]


def test_the_kepler_depth_is_carried_into_the_tess_band_before_comparison():
    """The two catalogues are different bandpasses; the ratio is applied, not ignored."""
    out = compare_three_depths(14280.6 * 0.97, 50.0, KEPLER_PPM, KEPLER_ERR, TOI_PPM, TOI_ERR,
                               ld_band_ratio=0.97)
    assert out["depth_kepler_in_tess_band_ppm"] == pytest.approx(KEPLER_PPM * 0.97, rel=1e-9)
    assert out["verdict"] == MATCH_KEPLER
    assert abs(out["z_vs_kepler"]) < 0.5
    # switched off, the same measurement is 3 % away from the reference
    off = compare_three_depths(14280.6 * 0.97, 50.0, KEPLER_PPM, KEPLER_ERR, TOI_PPM, TOI_ERR,
                               ld_band_ratio=0.97,
                               params=CompareParams(apply_band_ratio=False))
    assert off["depth_kepler_in_tess_band_ppm"] == pytest.approx(KEPLER_PPM, rel=1e-9)
    assert abs(off["z_vs_kepler"]) > abs(out["z_vs_kepler"])


def test_an_unmeasured_target_never_reaches_a_match_branch():
    out = compare_three_depths(float("nan"), float("nan"), KEPLER_PPM, KEPLER_ERR, TOI_PPM,
                               TOI_ERR, unmeasured_reason=UNMEASURED_QUERY_FAILED)
    assert out["verdict"] == MATCH_UNMEASURED
    assert out["unmeasured_reason"] == UNMEASURED_QUERY_FAILED
    assert out["agrees_with_kepler"] is False and out["agrees_with_toi"] is False


def test_a_measured_depth_with_no_reference_is_not_called_a_mismatch():
    """Nothing to compare with is not "matches neither" --- that would assert a
    disagreement with values that do not exist.  The MEASUREMENT stands; the
    COMPARISON is what is unmeasured."""
    out = compare_three_depths(14300.0, 200.0, float("nan"), float("nan"), float("nan"),
                               float("nan"))
    assert out["verdict"] == MATCH_UNMEASURED
    assert out["unmeasured_reason"] == UNMEASURED_NO_REFERENCE
    assert out["n_references_usable"] == 0
    assert out["depth_measured_ppm"] == pytest.approx(14300.0)
    # one usable reference is enough to ask the question
    one = compare_three_depths(14300.0, 200.0, KEPLER_PPM, KEPLER_ERR, float("nan"),
                               float("nan"), ld_band_ratio=1.0)
    assert one["n_references_usable"] == 1
    assert one["verdict"] == MATCH_KEPLER


def test_the_run_verdict_follows_the_target_verdicts():
    assert run_verdict(pd.DataFrame())[0] == RUN_NO_DATA
    assert run_verdict(pd.DataFrame({"verdict": [MATCH_UNMEASURED],
                                     "unmeasured_reason": ["QUERY_FAILED"]}))[0] == RUN_NO_DATA
    assert run_verdict(pd.DataFrame({"verdict": [MATCH_KEPLER]}))[0] == RUN_REFUTED
    assert run_verdict(pd.DataFrame({"verdict": [MATCH_KEPLER, MATCH_TOI]}))[0] == RUN_CONFIRMED
    assert run_verdict(pd.DataFrame({"verdict": [MATCH_KEPLER,
                                                 MATCH_NEITHER]}))[0] == RUN_UNRESOLVED
    assert run_verdict(pd.DataFrame({"verdict": [MATCH_BOTH]}))[0] == RUN_UNRESOLVED


# ---------------------------------------------------------------------------
# 7. End to end, all three branches, through the real stage code
# ---------------------------------------------------------------------------
KOI_ROW = {"kepoi_name": "K00897.01", "kepid": 7849854, "kepler_name": "Kepler-718 b",
           "koi_period": P, "koi_period_err1": 1.0e-7, "koi_period_err2": -1.0e-7,
           "koi_time0bk": T0_BKJD, "koi_time0bk_err1": 1.0e-4, "koi_time0bk_err2": -1.0e-4,
           "koi_duration": T14_H, "koi_duration_err1": 0.02, "koi_duration_err2": -0.02,
           "koi_depth": KEPLER_PPM, "koi_depth_err1": KEPLER_ERR, "koi_depth_err2": -KEPLER_ERR,
           "koi_ror": 0.11123, "koi_impact": 0.437, "koi_dor": 7.743, "koi_steff": 5800.0,
           "koi_slogg": 4.5, "ra": 285.0, "dec": 43.0}
TOI_ROW = {"toi": 4490.01, "tid": 268924036, "tfopwg_disp": "KP", "pl_orbper": 2.0523417,
           "pl_trandep": TOI_PPM, "pl_trandeperr1": TOI_ERR, "pl_trandeperr2": -TOI_ERR,
           "pl_trandurh": 2.043, "pl_trandurherr1": 0.05, "pl_trandurherr2": -0.05}


def _conf(tmp_path: Path, **stage2) -> dict:
    s2 = {"shortlist_from_candidates_csv": False,
          "shortlist": [{"kepoi_name": "K00897.01", "kepler_name": "Kepler-718 b",
                         "kepid": 7849854, "tic_id": 268924036, "toi": 4490.01}],
          "mast": {"budget_s": 60.0, "per_target_budget_s": 60.0, "retries": 1},
          "compare": {"n_agree": 3.0, "sigma_sys_ln": 0.05}}
    s2.update(stage2)
    return {"stage2": s2, "archive": {}, "limb_darkening": None}


def _query_fn(adql):
    if "from cumulative" in adql:
        return pd.DataFrame([KOI_ROW])
    if "from toi" in adql:
        return pd.DataFrame([TOI_ROW])
    return pd.DataFrame()


def _lc_fn_at_depth(depth_ppm, *, noise_ppm=250.0, n_transits=16, seed=21):
    t0 = bkjd_to_btjd(T0_BKJD) + 900 * P     # a TESS-era epoch, propagated back by the code

    def lc(_tic, **_kw):
        return [synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D,
                                 depth=depth_ppm * 1e-6, exptime_s=120.0,
                                 n_transits=n_transits, noise_ppm=noise_ppm, sector=41,
                                 seed=seed)]
    return lc


@pytest.mark.parametrize(
    ("injected_ppm", "expect_target", "expect_run"),
    [(KEPLER_PPM, MATCH_KEPLER, RUN_REFUTED),
     (TOI_PPM, MATCH_TOI, RUN_CONFIRMED),
     (22000.0, MATCH_NEITHER, RUN_UNRESOLVED)],
)
def test_end_to_end_each_of_the_three_outcomes(tmp_path, injected_ppm, expect_target,
                                               expect_run):
    """An injected light curve at each reference depth drives the right branch."""
    out = tmp_path / "stage2"
    conf = _conf(tmp_path)
    rep = stage2_measure(conf, out, query_fn=_query_fn, lc_fn=_lc_fn_at_depth(injected_ppm))
    assert rep["koi_ephemeris_status"] == "OK" and rep["toi_status"] == "OK"
    meas = pd.read_csv(out / "measurements.csv")
    assert len(meas) == 1
    row = meas.iloc[0]
    assert row["lc_status"] == "OK"
    assert row["verdict"] == expect_target
    assert abs(row["depth_measured_ppm"] - injected_ppm) < 4.0 * row["depth_measured_err_ppm"]
    summary = stage2_assess(conf, out)
    assert summary["verdict"] == expect_run
    assert summary["target_verdicts"][expect_target] == 1
    assert summary["n_measured"] == 1
    assert (out / "sectors.csv").exists()
    assert (out / "folds" / "K00897.01.csv").exists()
    # the summary is machine-readable and keeps the three references apart
    s = json.loads((out / "summary.json").read_text())
    t = s["targets"][0]
    assert t["depth_kepler_ppm"] == pytest.approx(KEPLER_PPM, rel=1e-6)
    assert t["depth_toi_ppm"] == pytest.approx(TOI_PPM, rel=1e-6)
    assert np.isfinite(t["depth_measured_ppm"])
    # the CSV round-trip may re-type a one-element list as a number; the FACT
    # under test is that the sector and the product are on the record.
    assert str(t["sector_list"]) == "41" and str(t["authors"]) == "SPOC"


def test_end_to_end_unreachable_mast_is_no_data_reached_and_not_a_refutation(tmp_path):
    """The failure this channel must never disguise: no data is not a null."""
    out = tmp_path / "stage2"
    conf = _conf(tmp_path)

    def boom(_tic, **_kw):
        raise RuntimeError("MAST unreachable")

    stage2_measure(conf, out, query_fn=_query_fn, lc_fn=boom)
    summary = stage2_assess(conf, out)
    assert summary["verdict"] == RUN_NO_DATA
    assert summary["n_measured"] == 0
    assert summary["target_verdicts"][MATCH_UNMEASURED] == 1
    assert summary["unmeasured_reasons"].get("QUERY_FAILED") == 1
    assert summary["verdict"] != RUN_REFUTED


def test_end_to_end_a_failed_ephemeris_pull_measures_nothing(tmp_path):
    out = tmp_path / "stage2"
    conf = _conf(tmp_path)

    def boom_q(_adql):
        raise RuntimeError("exoarchive 500")

    calls = []
    stage2_measure(conf, out, query_fn=boom_q,
                   lc_fn=lambda _t, **_k: calls.append(1) or [])
    summary = stage2_assess(conf, out)
    assert summary["verdict"] == RUN_NO_DATA
    assert summary["unmeasured_reasons"].get(UNMEASURED_NO_EPHEMERIS) == 1
    assert calls == []


def test_end_to_end_smeared_ffi_photometry_reaches_the_summary_as_a_flag(tmp_path):
    out = tmp_path / "stage2"
    conf = _conf(tmp_path)
    t0 = bkjd_to_btjd(T0_BKJD) + 900 * P

    def lc(_tic, **_kw):
        return [synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D,
                                 depth=TOI_PPM * 1e-6, exptime_s=1800.0, n_transits=40,
                                 noise_ppm=300.0, sector=55, author="QLP", seed=31)]

    stage2_measure(conf, out, query_fn=_query_fn, lc_fn=lc)
    summary = stage2_assess(conf, out)
    assert summary["flags"].get("smeared") == 1
    t = summary["targets"][0]
    assert bool(t["smeared"]) is True
    assert t["authors"] == "QLP"
    assert str(t["exptimes_s"]) == "1800"


def test_end_to_end_an_eclipsing_binary_is_measured_deep_AND_flagged_odd_even(tmp_path):
    """The mechanism that really can deepen a TESS transit leaves a mark here."""
    out = tmp_path / "stage2"
    conf = _conf(tmp_path)
    t0 = bkjd_to_btjd(T0_BKJD) + 900 * P

    def lc(_tic, **_kw):
        return [synth_lightcurve(period_days=P, t0_btjd=t0, duration_days=T14_D,
                                 depth=TOI_PPM * 1e-6, odd_depth=TOI_PPM * 0.6e-6,
                                 exptime_s=120.0, n_transits=24, noise_ppm=300.0,
                                 sector=41, seed=41)]

    stage2_measure(conf, out, query_fn=_query_fn, lc_fn=lc)
    summary = stage2_assess(conf, out)
    assert summary["flags"].get("odd_even_significant") == 1
    t = summary["targets"][0]
    assert abs(t["odd_even_sigma"]) > 3.0
    # the depth verdict is still reported; the flag is what makes it disbelievable
    assert t["verdict"] in ("MEASURED_DEPTH_MATCHES_TOI", "MEASURED_DEPTH_MATCHES_NEITHER")


def test_the_summary_carries_the_fields_the_workflow_guard_reads(tmp_path):
    """``growth_stage2.yml`` fails the run if an UNMEASURED target claims an
    agreement or gives no reason --- so those fields have to be IN the summary,
    or the guard is a no-op that always passes."""
    out = tmp_path / "stage2"
    conf = _conf(tmp_path)

    def boom(_tic, **_kw):
        raise RuntimeError("MAST unreachable")

    stage2_measure(conf, out, query_fn=_query_fn, lc_fn=boom)
    summary = stage2_assess(conf, out)
    t = summary["targets"][0]
    assert t["verdict"] == MATCH_UNMEASURED
    for field in ("unmeasured_reason", "agrees_with_kepler", "agrees_with_toi"):
        assert field in t, f"the workflow guard reads {field!r} and it is not in the summary"
    assert bool(t["agrees_with_kepler"]) is False
    assert bool(t["agrees_with_toi"]) is False
    assert t["unmeasured_reason"]


def test_the_summary_names_the_check_it_did_not_perform(tmp_path):
    """Stage 2 cannot exclude a signal from a different star in the pixel; it says so."""
    out = tmp_path / "stage2"
    conf = _conf(tmp_path)
    stage2_measure(conf, out, query_fn=_query_fn, lc_fn=_lc_fn_at_depth(KEPLER_PPM))
    summary = stage2_assess(conf, out)
    joined = " ".join(summary["checks_not_performed"]).lower()
    assert "centroid" in joined
    assert "different star" in joined or "pixel" in joined
    assert "not written up" in summary["note"]


# ---------------------------------------------------------------------------
# 8. Shortlist, config, probe, CLI wiring
# ---------------------------------------------------------------------------
def test_the_shortlist_comes_from_config_and_from_stage_1_candidates(tmp_path):
    csv = tmp_path / "candidates.csv"
    pd.DataFrame([{"kepoi_name": "K00897.01", "kepler_name": "Kepler-718 b", "kepid": 7849854,
                   "tic_id": 268924036.0, "toi": 4490.01},
                  {"kepoi_name": "K01234.01", "kepler_name": "", "kepid": 1, "tic_id": 2.0,
                   "toi": 9.01}]).to_csv(csv, index=False)
    conf = {"stage2": {"shortlist_from_candidates_csv": True,
                       "shortlist": [{"kepoi_name": "K00897.01", "tic_id": 268924036,
                                      "note": "from config"}]}}
    sl = load_shortlist(conf, candidates_csv=csv)
    assert list(sl["kepoi_name"]) == ["K00897.01", "K01234.01"]
    # the config row wins the duplicate, and says where it came from
    assert sl.iloc[0]["shortlist_source"] == "config"
    assert sl.iloc[1]["shortlist_source"] == "candidates_csv"
    off = load_shortlist({"stage2": {"shortlist_from_candidates_csv": False,
                                     "shortlist": []}}, candidates_csv=csv)
    assert not len(off)


def test_the_repository_config_carries_the_stage1_candidate_and_the_budgets():
    """`config/growth.yaml` is the source of record for the shortlist and the ceilings."""
    from seti.growth.run import load_growth_config

    conf = load_growth_config()
    s2 = conf.get("stage2") or {}
    names = [str(e.get("kepoi_name")) for e in (s2.get("shortlist") or [])]
    assert "K00897.01" in names
    m = MastParams.from_config(conf)
    assert m.budget_s > 0 and m.per_target_budget_s > 0 and m.target_timeout_s > 0
    assert "SPOC" in m.authors
    f = FitParams.from_config(conf)
    assert 0 < f.core_fraction < 1 and f.smear_fraction > 0
    c = CompareParams.from_config(conf)
    assert c.n_agree >= 3.0 and c.sigma_sys_ln > 0


def test_the_probe_records_the_mast_route_without_touching_the_network(tmp_path):
    out = tmp_path / "stage2"
    rep = stage2_probe(_conf(tmp_path), out)
    assert (out / "probe.json").exists()
    assert set(rep["mast"]) >= {"lightkurve", "astroquery_mast", "astropy_io_fits",
                                "route_preferred"}
    assert isinstance(rep["mast"]["lightkurve"]["importable"], bool)
    # imports only: reachability of the SERVICE is a measure-stage fact
    assert "does MAST answer" not in json.dumps(rep)
    assert "QUERY_FAILED" in rep["mast"]["note"]
    probe = mast_probe()
    assert probe["route_preferred"] in (None, "lightkurve", "astroquery_mast_fits")


def test_stage2_run_all_wires_probe_measure_and_assess(tmp_path):
    """The orchestration, end to end, with both services injected."""
    out = tmp_path / "stage2"
    rep = stage2_run("all", out_dir=out, conf=_conf(tmp_path), query_fn=_query_fn,
                     lc_fn=_lc_fn_at_depth(KEPLER_PPM))
    assert rep["verdict"] == RUN_REFUTED
    for name in ("probe.json", "acquire.json", "acquisition_log.json", "measurements.csv",
                 "sectors.csv", "summary.json"):
        assert (out / name).exists(), name
    # every file the workflow commits back is one this run actually produced
    assert (out / "folds").is_dir()


def test_stage2_run_rejects_an_unknown_stage(tmp_path):
    with pytest.raises(SystemExit):
        stage2_run("nonsense", out_dir=tmp_path, conf=_conf(tmp_path))


def test_the_cli_exposes_growth_stage2():
    from seti import cli

    assert callable(getattr(cli, "_cmd_growth_stage2", None))
    with pytest.raises(SystemExit) as exc:
        stage2_main(["--help"])
    assert exc.value.code == 0

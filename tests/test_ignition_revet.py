"""IGNITION re-vet: a brightness-dependent survey drift must not survive as a riser."""

import numpy as np
import pandas as pd

from seti.ignition.ensemble import apply_ensemble, ensemble_offsets
from seti.ignition.revet import _quick_slopes, drift_by_magnitude, stratified_offsets


def _fake_epochs(n_faint=300, n_bright=60, seed=1):
    """Faint stars drift +6 mmag/yr fainter; bright ones do not drift at all."""
    rng = np.random.default_rng(seed)
    t = 2014.2 + 0.5 * np.arange(21)
    rows = []
    for i in range(n_faint + n_bright):
        bright = i >= n_faint
        m0 = 9.6 if bright else 11.5
        drift = 0.0 if bright else 0.006
        for band in ("W1", "W2"):
            mag = m0 + drift * (t - t[0]) + rng.normal(0, 0.004, t.size)
            rows += [{"source_id": str(i), "band": band, "t_yr": tt, "mag": mm, "err": 0.004}
                     for tt, mm in zip(t, mag, strict=False)]
    return pd.DataFrame(rows)


def test_global_ensemble_makes_bright_stars_rise_and_stratified_does_not():
    ep = _fake_epochs()
    off = ensemble_offsets(ep, bin_yr=0.25, min_stars=8)
    glob = apply_ensemble(ep, off, bin_yr=0.25)
    qs = _quick_slopes(glob, 0.005)
    bright = [str(i) for i in range(300, 360)]
    # the median (faint) drift subtracted from non-drifting bright stars: a "rise"
    assert (qs.loc[bright, ("sigma", "W2")] > 5).mean() > 0.9
    beta = pd.Series(80.0, index=ep["source_id"].unique())
    corr, rep = stratified_offsets(ep, beta, min_stars=20)
    qs2 = _quick_slopes(corr, 0.005)
    assert (qs2.loc[bright, ("sigma", "W2")].abs() < 5).all()
    assert rep["frac_epochs_uncorrected"] == 0.0


def test_drift_by_magnitude_counts():
    stars = pd.DataFrame({"w1_median": [9.6, 9.7, 11.5], "w1_slope_raw_mag_yr": [0, 0, 0.006],
                          "w2_slope_raw_mag_yr": [0, 0, 0.006],
                          "w1_slope_mag_yr": [-0.006, -0.006, 0],
                          "w2_slope_mag_yr": [-0.006, -0.006, 0], "w1_slope_sigma": [6, 6, 0],
                          "w2_slope_sigma": [6, 6, 0],
                          "screen_verdict": ["IGNITION_CANDIDATE", "NOT_RISING", "NOT_RISING"]})
    rows = {r["mag_lo"]: r for r in drift_by_magnitude(stars)}
    assert rows[9.5]["n"] == 2 and rows[9.5]["n_corr_two_band_rising_5sig"] == 2
    assert rows[9.5]["n_candidates"] == 1

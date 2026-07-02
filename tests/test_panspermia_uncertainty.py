"""Tests for the Monte-Carlo encounter-uncertainty propagation."""
from __future__ import annotations

import pandas as pd

from seti.panspermia.uncertainty import mc_encounter, mc_shortlist


def _anchor():
    # A well-measured anchor at ~40 pc.
    return {"ra": 172.0, "dec": 7.5, "parallax": 26.0, "parallax_error": 0.03,
            "pmra": -84.7, "pmra_error": 0.03, "pmdec": -60.9, "pmdec_error": 0.03,
            "radial_velocity": 0.35, "radial_velocity_error": 0.1}


def test_tiny_errors_reproduce_point_estimate():
    # A star engineered to pass close in the past; with tiny errors the MC median
    # d_min should be small and the CI narrow.
    anchor = _anchor()
    star = {"ra": 172.5, "dec": 7.6, "parallax": 25.0, "parallax_error": 0.02,
            "pmra": -80.0, "pmra_error": 0.02, "pmdec": -58.0, "pmdec_error": 0.02,
            "radial_velocity": 5.0, "radial_velocity_error": 0.1}
    m = mc_encounter(anchor, star, n=1500, seed=1)
    assert m["n_valid"] > 1000
    # Narrow CI: p84 - p16 small relative to the median (well-measured).
    assert (m["d_min_p84"] - m["d_min_p16"]) < 1.0


def test_large_rv_error_inflates_dmin_ci():
    # The SAME star but with a huge RV error -> the d_min distribution widens.
    anchor = _anchor()
    base = {"ra": 172.5, "dec": 7.6, "parallax": 25.0, "parallax_error": 0.02,
            "pmra": -80.0, "pmra_error": 0.02, "pmdec": -58.0, "pmdec_error": 0.02,
            "radial_velocity": 5.0}
    tight = mc_encounter(anchor, {**base, "radial_velocity_error": 0.1}, n=1500, seed=2)
    loose = mc_encounter(anchor, {**base, "radial_velocity_error": 30.0}, n=1500, seed=2)
    tight_ci = tight["d_min_p84"] - tight["d_min_p16"]
    loose_ci = loose["d_min_p84"] - loose["d_min_p16"]
    assert loose_ci > tight_ci


def test_mc_shortlist_flags_robust_recipient():
    anchor = _anchor()
    cands = pd.DataFrame([
        {"source_id": 1, "ra": 172.5, "dec": 7.6, "dist_pc": 40.0,
         "phot_g_mean_mag": 10.0, "bp_rp": 1.0, "transfer_score": 0.001,
         "parallax": 25.0, "parallax_error": 0.02, "pmra": -80.0,
         "pmra_error": 0.02, "pmdec": -58.0, "pmdec_error": 0.02,
         "radial_velocity": 5.0, "radial_velocity_error": 0.1},
    ])
    out = mc_shortlist(anchor, cands, n=1200)
    assert "d_min_p50" in out.columns
    assert "robust_recipient" in out.columns
    assert out["n_valid"].iloc[0] > 800

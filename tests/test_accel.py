"""Offline tests for the astrometric-acceleration (dark-companion) search."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seti.accel.analyze import (
    analyze_accelerations,
    implied_companion_mass,
    physical_acceleration,
    rank_candidates,
)


def test_physical_acceleration_scales_with_distance():
    # Same angular acceleration, farther star -> larger physical acceleration.
    a_near = physical_acceleration(np.array([1.0]), np.array([10.0]))  # 100 pc
    a_far = physical_acceleration(np.array([1.0]), np.array([2.0]))    # 500 pc
    assert a_far[0] > a_near[0] > 0


def test_implied_mass_positive_and_monotonic():
    a = physical_acceleration(np.array([5.0]), np.array([5.0]))
    m = implied_companion_mass(a, separation_au=3.0)
    assert m[0] > 0
    # larger separation for the same acceleration implies a larger mass
    assert implied_companion_mass(a, 5.0)[0] > implied_companion_mass(a, 2.0)[0]


def test_rank_selects_massive_dark_nearby_high_significance():
    df = pd.DataFrame({
        "source_id": [1, 2, 3],
        # 1: strong accel, nearby, main-sequence primary -> dark massive companion
        # 2: low significance -> rejected
        # 3: far away -> rejected
        "accel_ra": [8.0, 0.2, 8.0],
        "accel_dec": [8.0, 0.2, 8.0],
        "accel_ra_error": [0.1, 0.1, 0.1],
        "accel_dec_error": [0.1, 0.1, 0.1],
        "parallax": [20.0, 20.0, 1.0],           # 50 pc, 50 pc, 1000 pc
        "parallax_over_error": [100, 100, 5],
        "phot_g_mean_mag": [10.0, 10.0, 15.0],
        "bp_rp": [0.6, 0.6, 0.6],
    })
    ranked = rank_candidates(df, sig_min=20.0, max_dist_pc=500.0)
    assert list(ranked["source_id"]) == [1]        # only the strong nearby one
    assert ranked.iloc[0]["rank_score"] > 0
    assert ranked.iloc[0]["accel_significance"] >= 20.0


def test_analyze_adds_expected_columns():
    df = pd.DataFrame({
        "source_id": [1], "accel_ra": [5.0], "accel_dec": [5.0],
        "accel_ra_error": [0.1], "accel_dec_error": [0.1], "parallax": [10.0],
        "parallax_over_error": [100], "phot_g_mean_mag": [11.0], "bp_rp": [0.7],
    })
    out = analyze_accelerations(df)
    for c in ("accel_significance", "dist_pc", "accel_m_s2",
              "implied_companion_msun", "dark_companion"):
        assert c in out.columns
    assert np.isfinite(out["accel_significance"].iloc[0])


def test_accel_run_offline(tmp_path):
    from seti.accel.run import accel_run
    from seti.config import load_config
    df = pd.DataFrame({
        "source_id": [1, 2], "accel_ra": [9.0, 0.1], "accel_dec": [9.0, 0.1],
        "accel_ra_error": [0.1, 0.1], "accel_dec_error": [0.1, 0.1],
        "parallax": [25.0, 25.0], "parallax_over_error": [100, 100],
        "phot_g_mean_mag": [9.5, 9.5], "bp_rp": [0.5, 0.5],
    })
    cfg = load_config()
    cfg.root = tmp_path
    summary = accel_run(cfg, table=df)
    assert summary["n_searched"] == 2
    assert summary["n_candidates"] >= 1
    assert (tmp_path / "results" / "accel" / "summary.json").exists()


def test_companion_mass_inverts_mass_function():
    from seti.accel.orbit import companion_mass
    # f = M2^3/(M1+M2)^2; for M1=1, M2=10 -> f=1000/121=8.264
    import numpy as np
    m2 = companion_mass(np.array([1000.0 / 121.0]), np.array([1.0]))
    assert abs(m2[0] - 10.0) < 0.2


def test_rank_dark_companions_flags_bh_like_system():
    import numpy as np
    import pandas as pd

    from seti.accel.orbit import rank_dark_companions
    # Construct a BH-like astrometric orbit: solar-mass G star (M_G~4.8 at 100pc,
    # G~9.8), period 1 yr, photocentre a0 giving a massive dark companion.
    # a0_au^3 / P_yr^2 = f; want f ~ 8 (=> M2~10 for M1~1). a0_au = f^(1/3) for P=1yr.
    a0_au = (8.0) ** (1 / 3)                     # ~2.0 AU
    plx = 10.0                                   # 100 pc
    a0_mas = a0_au * plx                         # a0_au = a0_mas/plx
    # Thiele-Innes with A=a0_mas, others 0 -> a0 recovered = a0_mas
    df = pd.DataFrame({
        "source_id": [42], "a_thiele_innes": [a0_mas], "b_thiele_innes": [0.0],
        "f_thiele_innes": [0.0], "g_thiele_innes": [0.0], "period": [365.25],
        "parallax": [plx], "phot_g_mean_mag": [9.8],
    })
    ranked = rank_dark_companions(df, m2_min=3.0)
    assert len(ranked) == 1
    assert ranked.iloc[0]["m2_msun"] > 3.0
    assert bool(ranked.iloc[0]["compact_object_candidate"]) is True

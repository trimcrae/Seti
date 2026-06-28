"""Offline tests for the multi-modal anomaly indicator suite.

These exercise the energy-balance, UV-deficit, variability and kinematic axes and
the combiner on small synthetic frames -- no network, no real catalogues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seti.config import load_config
from seti.indicators.combine import combine_indicators
from seti.indicators.energy_balance import energy_balance, uv_deficit
from seti.indicators.other_axes import (
    ir_excess,
    ir_variability,
    kinematic,
    optical_variability,
)
from seti.indicators.run import indicator_summary, run_multimodal


def _frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "source_id": np.arange(n).astype("int64"),
        "ra": rng.uniform(0, 360, n),
        "dec": rng.uniform(-30, 30, n),
        "teff": rng.uniform(5000, 20000, n),
        "parallax": rng.uniform(10, 100, n),
        "pmra": rng.normal(0, 50, n),
        "pmdec": rng.normal(0, 50, n),
        "chi_W2": rng.normal(2, 3, n),
        "has_excess": rng.random(n) < 0.2,
        "tau": 10 ** rng.uniform(-3, -1, n),
        "t_dust_k": rng.uniform(200, 1500, n),
        "sed_scale": 10 ** rng.uniform(-22, -20, n),
        "NUVmag": rng.uniform(16, 22, n),
        "e_NUVmag": rng.uniform(0.05, 0.3, n),
        "phot_variable_flag": rng.choice(["NOT_AVAILABLE", "VARIABLE"], n, p=[0.9, 0.1]),
    })


def test_each_axis_returns_aligned_result():
    df = _frame()
    cfg = load_config()
    for fn in (ir_excess, uv_deficit, energy_balance, optical_variability,
               ir_variability, kinematic):
        res = fn(df, cfg.thresholds)
        assert len(res.score) == len(df)
        assert len(res.flag) == len(df)
        assert len(res.available) == len(df)
        # Scores are in [0, 1] where finite.
        s = res.score.to_numpy(dtype=float)
        finite = np.isfinite(s)
        assert np.all((s[finite] >= 0.0) & (s[finite] <= 1.0))
        # A flagged object must also be available.
        flag = res.flag.fillna(False).to_numpy(dtype=bool)
        avail = res.available.fillna(False).to_numpy(dtype=bool)
        assert np.all(avail[flag])


def test_combine_attaches_multimodal_columns():
    df = _frame()
    cfg = load_config()
    comb = run_multimodal(df, cfg.thresholds, min_axes=2)
    assert "n_axes" in comb
    assert "multimodal_candidate" in comb
    assert "multimodal_score" in comb
    # n_axes equals the count of per-axis flags.
    flag_cols = [c for c in comb.columns if c.startswith("flag_")]
    recomputed = comb[flag_cols].sum(axis=1).to_numpy()
    assert np.array_equal(recomputed, comb["n_axes"].to_numpy())
    # Candidate iff at least min_axes flags.
    assert np.array_equal(comb["multimodal_candidate"].to_numpy(),
                          comb["n_axes"].to_numpy() >= 2)


def test_detail_columns_propagate():
    df = _frame()
    cfg = load_config()
    comb = run_multimodal(df, cfg.thresholds, min_axes=2)
    # Energy-balance and kinematic details should surface as columns.
    assert "nuv_deficit_frac" in comb
    assert "uv_ir_ratio" in comb
    assert "vtan_km_s" in comb


def test_unavailable_axis_not_counted():
    # A frame with no UV/variability data: those axes must be unavailable, not
    # silently flagged.
    df = _frame().drop(columns=["NUVmag", "e_NUVmag", "phot_variable_flag"])
    cfg = load_config()
    res = uv_deficit(df, cfg.thresholds)
    assert not res.available.any()
    assert not res.flag.any()


def test_energy_balance_flags_matched_absorption_reemission():
    # Construct one object where the NUV deficit fraction equals tau (perfect
    # balance) and one where there is excess but no deficit.
    cfg = load_config()
    from seti.photometry import band_freq_hz, mag_to_flux_jy, planck_bnu
    teff = 12000.0
    scale = 1e-21
    pred_nuv = scale * np.pi * planck_bnu(teff, band_freq_hz("NUV")) * 1e26
    # Object A: observe half the predicted NUV -> deficit fraction 0.5; tau=0.5.
    obs_a = 0.5 * pred_nuv
    nuv_a = -2.5 * np.log10(obs_a / 3631.0)
    # Object B: observe full predicted NUV -> no deficit; tau=0.5.
    nuv_b = -2.5 * np.log10(pred_nuv / 3631.0)
    df = pd.DataFrame({
        "source_id": [1, 2],
        "teff": [teff, teff],
        "sed_scale": [scale, scale],
        "tau": [0.5, 0.5],
        "NUVmag": [nuv_a, nuv_b],
        "e_NUVmag": [0.05, 0.05],
    })
    res = energy_balance(df, cfg.thresholds)
    assert bool(res.flag.iloc[0]) is True
    assert bool(res.flag.iloc[1]) is False
    # sanity: round-trip the flux conversion used above
    assert np.isclose(mag_to_flux_jy(nuv_b, "NUV"), pred_nuv, rtol=1e-6)


def test_summary_counts_are_consistent():
    df = _frame()
    cfg = load_config()
    comb = run_multimodal(df, cfg.thresholds, min_axes=2)
    summ = indicator_summary(comb)
    assert summ["n_objects"] == len(df)
    assert summ["n_multimodal"] == int(comb["multimodal_candidate"].sum())


def test_combine_handles_empty_results():
    df = _frame(10)
    comb = combine_indicators(df, [], weights={}, min_axes=2)
    assert (comb["n_axes"] == 0).all()
    assert not comb["multimodal_candidate"].any()

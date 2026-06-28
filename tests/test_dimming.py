"""Offline tests for the anomalous-dimming (Boyajian-star) dip statistics."""

from __future__ import annotations

import numpy as np

from seti.dimming.dips import detect_dips


def _flat(n=200, base=12.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1000, n))
    m = base + rng.normal(0, noise, n)
    return t, m, np.full(n, noise)


def test_flat_lightcurve_no_dips():
    t, m, e = _flat()
    s = detect_dips(t, m, e)
    assert s is not None
    assert s.n_dips == 0
    assert s.max_depth < 0.05
    assert s.score < 0.3


def test_boyajian_like_aperiodic_dimming_scores_high():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1500, 400))
    m = 12.0 + rng.normal(0, 0.01, t.size)
    # A handful of deep, irregular (aperiodic) dips of 10-22% (0.11-0.27 mag).
    for tc, depth_mag in [(220, 0.16), (540, 0.27), (910, 0.12), (1290, 0.20)]:
        m += depth_mag * np.exp(-0.5 * ((t - tc) / 12.0) ** 2)
    e = np.full(t.size, 0.01)
    s = detect_dips(t, m, e)
    assert s is not None
    assert s.max_depth > 0.1
    assert s.n_dips >= 3
    assert s.asymmetry > 1.5           # dimming-only, not symmetric
    assert s.score > 0.5


def test_smooth_sinusoid_is_symmetric_low_score():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 1000, 300))
    m = 12.0 + 0.05 * np.sin(2 * np.pi * t / 30.0) + rng.normal(0, 0.005, t.size)
    s = detect_dips(t, m, magerr=np.full(t.size, 0.005))
    # A symmetric oscillation is not dimming-dominated.
    assert s.asymmetry < 1.6
    assert s.score < 0.6


def test_eclipsing_binary_is_periodic():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 1000, 600))
    period = 3.2
    phase = (t % period) / period
    m = 12.0 + rng.normal(0, 0.01, t.size)
    m += 0.3 * (np.abs(phase - 0.5) < 0.05)   # periodic box eclipses
    s = detect_dips(t, m, magerr=np.full(t.size, 0.01))
    assert s is not None
    # Strictly periodic eclipses produce strong Lomb-Scargle power.
    assert s.period_power > 0.1


def test_too_few_epochs_returns_none():
    t, m, e = _flat(n=10)
    assert detect_dips(t, m, e) is None


def test_hr_class_separates_giants_dwarfs_main_sequence():
    from seti.dimming.context import absolute_g, hr_class

    # A nearby F/G main-sequence star: BP-RP~0.7, M_G~4.6.
    # parallax 20 mas (50 pc) => G - 5log10(50/10)... use absolute_g to set G.
    plx = 20.0  # mas, d=50 pc
    g_ms = 4.6 - 5.0 * np.log10(plx / 100.0)        # apparent G giving M_G=4.6
    assert abs(absolute_g(g_ms, plx) - 4.6) < 1e-6
    assert hr_class(g_ms, 0.7, plx, parallax_over_error=50.0) == "main_sequence"

    # A red giant at the same colour but hugely over-luminous (M_G ~ 0).
    g_giant = 0.0 - 5.0 * np.log10(plx / 100.0)
    assert hr_class(g_giant, 0.7, plx, parallax_over_error=50.0) == "giant"

    # A white dwarf: blue and very faint (M_G ~ 12).
    g_wd = 12.0 - 5.0 * np.log10(plx / 100.0)
    assert hr_class(g_wd, 0.2, plx, parallax_over_error=50.0) == "white_dwarf"

    # No reliable parallax => never assert a class.
    assert hr_class(g_ms, 0.7, plx, parallax_over_error=1.0) == "unknown"


def test_dimming_run_end_to_end(tmp_path):
    from seti.config import load_config
    from seti.dimming.run import dimming_run

    lightcurves = []
    # Five flat (non-dipping) stars.
    for i in range(5):
        t, m, e = _flat(n=200, seed=10 + i)
        lightcurves.append({"source_id": 1000 + i, "ra": 270.0 + 0.01 * i,
                            "dec": 30.0, "mjd": t, "mag": m, "magerr": e})
    # One Boyajian-like deep aperiodic dipper.
    rng = np.random.default_rng(99)
    t = np.sort(rng.uniform(0, 1500, 400))
    m = 12.0 + rng.normal(0, 0.01, t.size)
    for tc, depth in [(220, 0.16), (540, 0.27), (910, 0.12), (1290, 0.20)]:
        m += depth * np.exp(-0.5 * ((t - tc) / 12.0) ** 2)
    lightcurves.append({"source_id": 2002, "ra": 271.0, "dec": 31.0,
                        "mjd": t, "mag": m, "magerr": np.full(t.size, 0.01),
                        # main-sequence F star context: blue-ish, good parallax
                        "phot_g_mean_mag": 12.0, "bp_rp": 0.6, "parallax": 5.0,
                        "parallax_over_error": 40.0})

    cfg = load_config()
    cfg.root = tmp_path
    summary = dimming_run(cfg, lightcurves=lightcurves)

    assert summary["n_searched"] == 6
    assert summary["n_candidates"] >= 1
    # The injected dipper is the top candidate.
    assert summary["top_candidates"][0]["source_id"] == 2002
    # ...and it is flagged as resisting the mundane (main-sequence, aperiodic).
    assert summary["top_candidates"][0]["hr_class"] == "main_sequence"
    assert summary["top_candidates"][0]["resists_mundane"] is True
    assert summary["n_resists_mundane"] >= 1
    assert "occurrence_limit" in summary
    assert (tmp_path / "results" / "dimming" / "summary.json").exists()
    assert (tmp_path / "results" / "dimming" / "top_dippers.json").exists()

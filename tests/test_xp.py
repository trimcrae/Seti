"""Offline tests for the Gaia XP spectral-anomaly technosignature detector.

We synthesise a population of normal stellar XP-like spectra (smooth Planck
continua across the XP range with varying temperature + photometric noise), fit
the colour-conditional median-spectrum locus, then verify that ordinary stars
score low while two technosignature-like anomalies score high: a sharp band
deficit (band-limited reprocessing) and a narrow artificial feature.
"""

from __future__ import annotations

import numpy as np

from seti.xp.anomaly import anomaly_score, fit_locus, normalize_spectrum

# XP covers ~336-1020 nm; use a representative sampled grid.
WAVE = np.linspace(336.0, 1020.0, 120) * 1e-9  # metres


def _planck(wave_m: np.ndarray, teff: float) -> np.ndarray:
    h, c, kB = 6.626e-34, 3.0e8, 1.381e-23
    x = h * c / (wave_m * kB * teff)
    return 1.0 / (wave_m**5 * (np.expm1(np.clip(x, 1e-6, 700))))


def _teff_to_bprp(teff: float) -> float:
    # Monotonic proxy: hot -> blue (small BP-RP), cool -> red.
    return float(np.clip(9000.0 / teff - 0.4, -0.3, 4.0))


def _normal_pop(n=400, seed=0):
    rng = np.random.default_rng(seed)
    specs, cols = [], []
    for _ in range(n):
        teff = rng.uniform(3500, 9000)
        f = _planck(WAVE, teff)
        f = f / np.median(f)
        f = f * (1.0 + rng.normal(0, 0.01, f.size))   # photometric noise
        specs.append(normalize_spectrum(f))
        cols.append(_teff_to_bprp(teff))
    return np.array(specs), np.array(cols)


def test_normal_stars_score_low():
    pop, cols = _normal_pop(seed=1)
    locus = fit_locus(pop, cols)
    held, hcols = _normal_pop(n=20, seed=99)
    sigmas = [anomaly_score(held[i], hcols[i], locus)["global_sigma"]
              for i in range(held.shape[0])]
    assert np.median(sigmas) < 5.0


def test_dyson_deficit_flagged():
    # Anomaly is included in the training population (catalogue-scale reality);
    # the robust colour-median locus must still flag it.
    pop, cols = _normal_pop(seed=2)
    f = _planck(WAVE, 6000.0)
    f = f / np.median(f)
    f[(WAVE > 500e-9) & (WAVE < 600e-9)] *= 0.3      # sharp non-stellar band notch
    s = normalize_spectrum(f)
    col = _teff_to_bprp(6000.0)
    locus = fit_locus(np.vstack([pop, s]), np.append(cols, col))
    score = anomaly_score(s, col, locus)
    pop_sigma = np.array([anomaly_score(pop[i], cols[i], locus)["global_sigma"]
                          for i in range(pop.shape[0])])
    assert score["global_sigma"] > np.nanquantile(pop_sigma, 0.99) + 5.0


def test_artificial_narrow_feature_flagged():
    pop, cols = _normal_pop(seed=3)
    f = _planck(WAVE, 5500.0)
    f = f / np.median(f)
    j = 60
    f[j] *= 3.0                                      # narrow artificial emission
    s = normalize_spectrum(f)
    col = _teff_to_bprp(5500.0)
    locus = fit_locus(np.vstack([pop, s]), np.append(cols, col))
    score = anomaly_score(s, col, locus)
    assert score["feature_resid"] > 6.0
    assert abs(score["feature_index"] - j) <= 2


def test_xp_run_end_to_end(tmp_path):
    import pandas as pd

    from seti.config import load_config
    from seti.xp.run import classify_xp_anomaly, xp_run

    pop, cols = _normal_pop(n=300, seed=5)
    f = _planck(WAVE, 6000.0)
    f = f / np.median(f)
    f[(WAVE > 500e-9) & (WAVE < 600e-9)] *= 0.3      # sharp non-stellar notch
    anom = normalize_spectrum(f)
    flux = np.vstack([pop, anom])
    colors = np.append(cols, _teff_to_bprp(6000.0))
    meta = pd.DataFrame({
        "source_id": np.arange(flux.shape[0]), "ra": 180.0, "dec": 30.0,
        "phot_g_mean_mag": 15.0, "bp_rp": colors, "parallax": 5.0,
        "parallax_over_error": 50.0, "teff_gspphot": 6000.0,
        "classprob_dsc_combmod_quasar": 0.0, "classprob_dsc_combmod_galaxy": 0.0,
        "classprob_dsc_combmod_star": 1.0, "non_single_star": 0,
        "phot_variable_flag": "NOT_AVAILABLE"})
    cfg = load_config()
    cfg.root = tmp_path
    summary = xp_run(cfg, chunk={"wave": WAVE, "flux": flux, "meta": meta},
                     global_sigma_min=8.0)
    assert summary["n_searched"] == flux.shape[0]
    assert summary["n_clean_anomalies"] >= 1
    assert summary["top_clean"][0]["source_id"] == flux.shape[0] - 1   # planted one
    assert classify_xp_anomaly({"classprob_dsc_combmod_quasar": 0.9}) == "quasar"


def test_fit_locus_shapes():
    pop, cols = _normal_pop(n=200, seed=4)
    locus = fit_locus(pop, cols, n_bins=10)
    assert locus.medians.shape[1] == WAVE.size
    assert locus.global_mad > 0

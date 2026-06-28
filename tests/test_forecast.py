"""Tests for the WD population model and the projected-sensitivity forecast."""

import numpy as np

from seti.population import generate_population, mass_radius_rsun
from seti.stats.sensitivity import (
    forecast_sensitivity,
    headline_limit,
    minimum_detectable_tau,
)


def test_mass_radius_monotonic_decreasing():
    m = np.array([0.4, 0.6, 0.8, 1.0])
    r = mass_radius_rsun(m)
    # More massive white dwarfs are smaller.
    assert np.all(np.diff(r) < 0)
    # Radii are of order 0.01 Rsun.
    assert np.all((r > 0.005) & (r < 0.02))


def test_population_detection_fraction_reasonable(cfg):
    pop = generate_population(cfg, seed=3)
    frac = pop["detected"].mean()
    # Within 100 pc a substantial but not total fraction of WDs are WISE-detected.
    assert 0.4 < frac < 0.9
    det = pop[pop["detected"]]
    # Detected white dwarfs are brighter than the CatWISE2020 W2 5-sigma limit.
    assert det["W2mag"].median() < cfg.population["wise_depth"]["catwise2020"]["W2_5sigma"]


def test_population_nearby_always_detected(cfg):
    pop = generate_population(cfg, seed=3)
    near = pop[pop["dist_pc"] < 25]
    assert near["detected"].mean() > 0.95  # very nearby WDs essentially all detected


def test_forecast_limit_and_completeness(cfg):
    fc = forecast_sensitivity(cfg, t_grid=[300, 800], tau_grid=[0.01, 0.1])
    # Completeness rises with tau at fixed temperature.
    for t in (300, 800):
        sub = fc[fc.t_dust_k == t].sort_values("tau")
        assert sub["recovered_fraction"].is_monotonic_increasing
    # Occurrence-rate upper limit falls as recovery (hence N_eff) rises.
    assert fc["f_upper_95"].min() < 1e-3
    # N_eff never exceeds the detected sample scaled to the real catalogue.
    assert (fc["n_eff"] <= fc["n_detected_real"] + 1e-6).all()


def test_headline_limit_well_defined(cfg):
    fc = forecast_sensitivity(cfg, t_grid=[300, 800], tau_grid=[0.03, 0.3])
    h = headline_limit(fc)
    assert 0 < h["f_upper_95"] < 1e-2
    assert h["n_eff"] > 0


def test_minimum_detectable_tau_increases_for_cool_dust(cfg):
    fc = forecast_sensitivity(cfg, t_grid=[300, 800], tau_grid=[0.003, 0.01, 0.03, 0.1, 0.3])
    mdt = minimum_detectable_tau(fc).set_index("t_dust_k")["tau_min_detectable"]
    # Cool dust (300 K) requires at least as large a covering fraction to detect
    # as warm dust (800 K), where the search is most sensitive.
    if np.isfinite(mdt.get(300, np.nan)) and np.isfinite(mdt.get(800, np.nan)):
        assert mdt[300] >= mdt[800]

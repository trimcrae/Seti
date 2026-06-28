"""Tests for SED prediction and infrared-excess recovery."""

import numpy as np

from seti.sed.excess import characterise_dust, compute_excess, fit_dust, select_excess
from seti.sed.predict import predict_photosphere


def test_clean_sources_have_no_excess(cfg, sample):
    clean = sample[sample.label == "clean"]
    ex = compute_excess(predict_photosphere(clean), cfg.thresholds)
    # Clean photospheres should scatter around zero excess significance.
    assert abs(np.nanmedian(ex["chi_W1"])) < 1.0
    assert select_excess(ex, cfg.thresholds).mean() < 0.02  # ~no false positives


def test_known_disks_and_anomalies_recovered(cfg, sample):
    ex = compute_excess(predict_photosphere(sample), cfg.thresholds)
    sel = select_excess(ex, cfg.thresholds)
    ex = ex.assign(sel=sel)
    # Most injected debris disks and anomalies should be selected as excess.
    for label in ("known_disk", "anomaly"):
        frac = ex.loc[ex.label == label, "sel"].mean()
        assert frac > 0.5, f"{label} recovery too low: {frac}"


def test_fit_dust_recovers_injected_parameters(cfg, sample):
    ex = compute_excess(predict_photosphere(sample), cfg.thresholds)
    disks = ex[ex.label == "known_disk"]
    disks = characterise_dust(disks)
    good = disks.dropna(subset=["t_dust_k", "tau"])
    # Injected debris disks were T_dust in [1000,1500] K -> recovered near there.
    assert good["t_dust_k"].between(800, 1800).mean() > 0.8
    # Injected tau in [0.005, 0.05] -> recovered within a factor of a few.
    assert good["tau"].between(1e-3, 2e-1).mean() > 0.8


def test_fit_dust_handles_no_excess():
    import pandas as pd

    row = pd.Series({"W1_excess_jy": -1e-9, "W2_excess_jy": 1e-9,
                     "sed_scale": 1e-20, "teff": 10000.0})
    t, tau = fit_dust(row)
    assert np.isnan(t) and np.isnan(tau)

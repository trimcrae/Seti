"""Offline tests for the CRADLE lead vet (no network)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from seti.cradle import leadvet as lv


def _exposures(amp_w2=0.0, n_visits=18, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_visits):
        t0 = 56650 + 182.6 * v
        for k in range(12):
            w2 = 11.18 + amp_w2 * math.sin(2 * math.pi * v / 7.0)
            rows.append({"mjd": t0 + 0.1 * k, "w1mpro": 11.19 + rng.normal(0, 0.02),
                         "w1sigmpro": 0.02, "w2mpro": w2 + rng.normal(0, 0.03), "w2sigmpro": 0.03,
                         "qual_frame": 10, "qi_fact": 1.0, "saa_sep": 50, "moon_masked": "00",
                         "cc_flags": "0000"})
    return pd.DataFrame(rows)


def test_visit_binning_and_quiet_star():
    v = lv.bin_visits(lv._clean_neowise(_exposures()))
    assert len(v) == 18
    s = lv.lc_stats(v)
    assert s["w2"]["chi2_red"] < 3 and abs(s["w2"]["slope_sigma"]) < 4


def test_injected_w2_variability_recovered_and_colour_tracks_w2():
    s = lv.lc_stats(lv.bin_visits(lv._clean_neowise(_exposures(amp_w2=0.15))))
    assert s["w2"]["chi2_red"] > 20 and s["w2"]["range"] > 0.25
    assert s["w1w2"]["chi2_red"] > 10
    assert s["dcolour_dW2"] < -0.5          # W1-W2 reddens as W2 brightens


def test_bad_frames_are_dropped():
    d = _exposures()
    d.loc[:5, "qual_frame"] = 0
    d.loc[6:8, "moon_masked"] = "11"
    assert len(lv._clean_neowise(d)) == len(d) - 9


def test_blackbody_fit_recovers_temperature():
    lam = np.array([11.56, 22.09])
    f = 3e-17 * lv.bnu(lam, 262.0)
    r = lv.fit_bb(lam, f, 0.05 * f)
    assert r["t_k"] == pytest.approx(262.0, rel=0.03)


def test_li_measure_removes_fe_blend_and_gives_error():
    w = np.arange(6690.0, 6730.0, 0.25)
    li = 0.10 * np.exp(-0.5 * ((w - 6709.66) / 0.6) ** 2)
    f = 1.0 - li
    iv = np.full_like(w, 1e4)
    m = lv.li_measure(w, f, iv, 5700.0)
    ew_true = 0.10 * 0.6 * math.sqrt(2 * math.pi)
    assert m["ew_blend_A"] == pytest.approx(ew_true, rel=0.1)
    assert m["ew_li_A"] == pytest.approx(m["ew_blend_A"] - m["ew_fe_correction_A"], rel=1e-9)
    assert 0.005 < m["ew_fe_correction_A"] < 0.015
    assert m["ew_li_err_A"] > 0


def test_gyro_and_bv():
    bv = lv.teff_to_bv(5770)
    assert bv == pytest.approx(0.65, abs=0.01)
    # the Sun: P ~ 26 d at B-V 0.65 -> ~4.6 Gyr within the calibration's scatter
    assert 3000 < lv.gyro_age_myr(26.0, bv) < 6500


def test_bensby_thin_vs_thick():
    assert lv.bensby_odds(0, -10, 0)["TD_over_D"] < 0.1
    assert lv.bensby_odds(60, -90, 60)["TD_over_D"] > 10


def test_orbit_of_the_sun_is_nearly_circular_and_flat():
    o = lv.orbit((-8.122, 0.0, 0.0208), (12.9, 245.6, 7.78), t_gyr=0.5, dt_myr=0.5)
    assert o["ecc"] < 0.15 and o["z_max_kpc"] < 0.2 and 6.5 < o["r_peri_kpc"] < 8.5


def test_core_ratio_detects_emission_filling():
    w = np.arange(3880.0, 4020.0, 0.5)
    model = 1.0 - 0.8 * np.exp(-0.5 * ((w - 3934.78) / 1.0) ** 2)
    data = model + 0.3 * np.exp(-0.5 * ((w - 3934.78) / 0.5) ** 2)
    assert lv.core_ratio(w, data, model, 3934.78, 1.0, [(3891, 3911), (3991, 4011)]) > 1.2
    assert lv.core_ratio(w, model, model, 3934.78, 1.0, [(3891, 3911), (3991, 4011)]) == pytest.approx(1.0)

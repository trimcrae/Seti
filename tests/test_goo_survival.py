"""Offline tests for seti.goo.survival: the interstellar functional-survival scan."""
from __future__ import annotations

import math

import numpy as np
import pytest

from seti.goo import survival as SV
from seti.goo.constants import KM, PC, YR
from seti.goo.run import load_cfg


def test_landing_rate_is_the_product_of_carriers_targets_speed_and_cross_section():
    v = 26 * KM
    rate = SV.landing_rate_per_yr(1.0, 1.0, 1.0, v)
    expect = (1.0 / PC**3) * v * SV.sigma_strict_m2(v) * YR
    assert rate == pytest.approx(expect)
    # linear in every factor; the volume the cloud occupies never enters
    assert SV.landing_rate_per_yr(10.0, 0.1, 0.1, v) == pytest.approx(0.1 * rate)


def test_r0_of_tau_has_the_right_limits():
    rate0, W = 3.0, 5e9
    assert SV.R0_of_tau(rate0, float("inf"), W) == pytest.approx(rate0 * W)
    assert SV.R0_of_tau(rate0, 1e12, W) == pytest.approx(rate0 * W, rel=1e-2)
    assert SV.R0_of_tau(rate0, 1e3, W) == pytest.approx(rate0 * 1e3)
    taus = np.logspace(3, 11, 50)
    vals = [SV.R0_of_tau(rate0, t, W) for t in taus]
    assert all(b >= a for a, b in zip(vals[:-1], vals[1:], strict=True))


def test_reach_is_speed_times_survival_time():
    assert SV.reach_pc(1 * KM, 1e6) == pytest.approx(1 * KM * 1e6 * YR / PC)
    assert SV.reach_pc(18 * KM, 1e6) == pytest.approx(18.4, rel=1e-2)


def test_planets_within_reach_goes_sphere_then_slab_then_caps_at_the_annulus():
    eta, n = 0.1, 0.1
    small = SV.planets_within(1.0, eta, n, 300.0, 800.0, 1e9)
    assert small == pytest.approx(eta * n * 4 / 3 * math.pi)
    slab = SV.planets_within(1000.0, eta, n, 300.0, 800.0, 1e9)
    assert slab == pytest.approx(eta * n * math.pi * 1000.0**2 * 600.0)
    assert SV.planets_within(1e6, eta, n, 300.0, 800.0, 1e9) == 1e9


def test_cosmic_ray_hits_scale_with_area():
    assert SV.cosmic_ray_hits_per_yr(5.0) == pytest.approx(100 * SV.cosmic_ray_hits_per_yr(0.5))
    assert 0.1 < SV.cosmic_ray_hits_per_yr(0.5) < 1.0


def test_tau_grid_ends_with_infinity():
    g = SV.tau_grid(1e3, 1e10, 8)
    assert len(g) == 9
    assert math.isinf(g[-1])
    assert g[0] == pytest.approx(1e3)


def test_scan_sub_micron_devices_front_needs_survival_above_the_hop_time():
    cfg = load_cfg()
    taus = np.array([1e4, 1e5, 1e6, 1e8, float("inf")])
    car = cfg["survival"]["carriers"][0]
    s = SV.scan(car, cfg, taus)
    assert 1e10 < s["rate0_per_yr"] < 1e12
    assert 1e5 < s["t_hop_yr"] < 2e5
    rows = {r["tau_yr"]: r for r in s["rows"]}
    assert rows[1e4]["R0"] > 1 and not rows[1e4]["front"]      # landings yes, chain no
    assert not rows[1e5]["front"]
    assert rows[1e6]["front"] and 5 < rows[1e6]["v_front_kms"] < 18
    assert 1e9 < rows[1e6]["t_galaxy_yr"] < 3e9
    assert rows[1e8]["reach_pc"] == pytest.approx(SV.reach_pc(18 * KM, 1e8))
    assert rows[float("inf")]["t_galaxy_yr"] == pytest.approx(rows[1e6]["t_galaxy_yr"])


def test_scan_rock_borne_devices_never_form_a_front():
    cfg = load_cfg()
    taus = np.array([1e6, 1e9, float("inf")])
    rock = SV.scan(cfg["survival"]["carriers"][3], cfg, taus)
    assert all(r["R0"] < 1 for r in rock["rows"])
    assert not any(r["front"] for r in rock["rows"])
    iso = SV.scan(cfg["survival"]["carriers"][2], cfg, taus)
    rows = {r["tau_yr"]: r for r in iso["rows"]}
    assert rows[1e9]["R0"] < 1
    assert 1 < rows[float("inf")]["R0"] < 10
    assert iso["t_first_yr"] > 1e9

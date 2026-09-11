"""Contact-graph tests: cross-sections, the ISO sanity number, channel dispatch and
the threshold logic."""
from __future__ import annotations

import math

from seti.goo import contact as CT
from seti.goo.constants import AU, KM, M_EARTH, M_SUN, R_EARTH, G
from seti.goo.run import load_cfg


def test_strict_cross_section_is_focused_earth_disc():
    v = 26 * KM
    ve = math.sqrt(2 * G * M_EARTH / R_EARTH)
    assert abs(CT.sigma_strict_m2(v) / (math.pi * R_EARTH**2) - (1 + (ve / v) ** 2)) < 1e-9


def test_loose_cross_section_at_100_au_is_barely_focused():
    v = 26 * KM
    r = 100 * AU
    ve = math.sqrt(2 * G * M_SUN / r)
    f = CT.sigma_loose_m2(v, 100.0) / (math.pi * r**2)
    assert abs(f - (1 + (ve / v) ** 2)) < 1e-9
    assert 1.0 < f < 1.2


def test_iso_impacts_on_earth_are_tens_over_its_history():
    cfg = load_cfg()
    n = CT.iso_impacts_on_earth_over_age(cfg)
    # n_ISO = 0.2 AU^-3 of >100 m bodies at 26 km/s onto a focused Earth: tens of hits in 4.5 Gyr
    assert 5 < n < 200


def test_iso_single_source_is_near_the_percolation_threshold():
    cfg = load_cfg()
    rows = {r.name: r for r in CT.run_contact(cfg, 5e9)}
    iso = rows["interstellar objects (>100 m)"]
    assert 0.3 < iso.R0_strict < 30
    assert iso.R0_loose > 1e10
    assert iso.t_epidemic_loose_yr < 2e9


def test_impact_ejecta_do_not_percolate_in_the_field():
    cfg = load_cfg()
    rows = {r.name: r for r in CT.run_contact(cfg, 5e9)}
    ej = rows["impact ejecta from a planet (rocks > 1 m)"]
    assert ej.R0_strict < 1e-2
    assert ej.t_epidemic_strict_yr == float("inf")


def test_free_floating_planets_are_not_carriers():
    cfg = load_cfg()
    rows = {r.name: r for r in CT.run_contact(cfg, 5e9)}
    ffp = rows["free-floating planets"]
    assert ffp.R0_loose < 10
    assert 1.0 / ffp.rate_loose_natural_per_system_yr > 1e9


def test_dust_touches_every_planet_continuously():
    cfg = load_cfg()
    rows = {r.name: r for r in CT.run_contact(cfg, 5e9)}
    dust = rows["interstellar dust (ISM grains, 0.1-1 um)"]
    assert dust.rate_strict_natural_per_planet_yr > 1e15


def test_encounters_connect_the_annulus_in_under_a_gigayear():
    cfg = load_cfg()
    rows = {r.name: r for r in CT.run_contact(cfg, 5e9)}
    enc = rows["stellar encounters (within the Oort cloud, 2e4 AU)"]
    assert 1e6 < 1.0 / enc.rate_loose_natural_per_system_yr < 1e8
    assert enc.t_epidemic_loose_yr < 1e9
    assert enc.rate_strict_natural_per_planet_yr == 0.0


def test_functional_survival_scales_r0_by_the_mixing_delay():
    cfg = load_cfg()
    rows = {r.name: r for r in CT.run_contact(cfg, 5e9)}
    gr = rows["radiation-pressure grains from a converted belt"]
    t_mix = cfg["contact"]["t_mix_annulus_yr"]
    for tf, val in gr.R0_strict_functional.items():
        assert abs(val - gr.R0_strict * math.exp(-t_mix / float(tf))) <= 1e-6 * max(1.0, gr.R0_strict)


def test_every_channel_dispatches_to_a_result():
    cfg = load_cfg()
    rows = CT.run_contact(cfg, 5e9)
    assert len(rows) == len(cfg["contact"]["channels"])
    assert {r.kind for r in rows} >= {"emitted carriers", "measured flux", "measured density", "local encounters", "local events", "birth cluster (first 100 Myr only)"}

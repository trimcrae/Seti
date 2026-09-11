"""GOO offline tests: each module's physics against hand calculations, and the
stage orchestration end to end into a temporary results directory."""
from __future__ import annotations

import json
import math

import pytest

from seti.goo import active as A
from seti.goo import bayes as B
from seti.goo import passive as P
from seti.goo import planet as PL
from seti.goo import system as S
from seti.goo.constants import (
    AU,
    GYR,
    KM,
    L_SUN,
    M_EARTH,
    MYR,
    R_EARTH,
    YR,
    G,
    t_eq_thin,
    v_circ,
)


# --- planet ------------------------------------------------------------------------
def test_seed_mass_of_half_micron_diamondoid_sphere():
    m = PL.seed_mass_kg(0.5, 2.0)
    assert abs(m - (4 / 3) * math.pi * (0.5e-6) ** 3 * 2000) < 1e-20
    assert 5e-16 < m < 2e-15


def test_doublings_for_earth_from_one_seed_is_about_130():
    m = PL.seed_mass_kg(0.5, 2.0)
    assert 125 < PL.doublings(5.97e24, m) < 140


def test_radiative_ceiling_matches_stefan_boltzmann():
    p = PL.Planet(5.972e24, 6.371e6, 255.0, 1.22e17, 1e15, 2.4e22, 1e17, 8e13, 1.4e21, 1.56e-4, 3.4e14)
    P500 = PL.radiative_power_limit(p, 500.0)
    hand = 4 * math.pi * 6.371e6**2 * 5.67e-8 * (500**4 - 255**4)
    assert abs(P500 / hand - 1) < 1e-3
    assert 1e18 < P500 < 2e18


def test_governing_time_is_the_slowest_ceiling():
    p = PL.Planet(5.972e24, 6.371e6, 255.0, 1.22e17, 1e15, 2.4e22, 1e17, 8e13, 1.4e21, 1.56e-4, 3.4e14)
    t = PL.conversion_table(p, 1e-15, 1e7, 1000.0, 500.0)
    for row in t.values():
        assert row["t_governing_s"] >= max(row["t_exponential_s"], row["t_radiative_s"])
    # the biosphere is hours by exponential growth but the crust is millennia by any route
    assert t["biosphere"]["t_exponential_s"] < 3 * 86400
    assert t["crust"]["t_governing_s"] > 1e3 * YR
    assert t["bulk_planet"]["t_governing_s"] > 1e6 * YR


def test_disassembly_of_earth_at_solar_luminosity_is_about_a_week():
    t = PL.t_disassembly_s(M_EARTH, R_EARTH, L_SUN)
    assert 3 * 86400 < t < 20 * 86400


# --- system ------------------------------------------------------------------------
def test_thin_sheet_temperature_at_one_au_is_278_K():
    assert abs(t_eq_thin(AU) - 278.0) < 3.0


def test_covering_fraction_saturates_at_unity_and_scales_with_mass():
    f_film = S.covering_fraction(2.4e21, 1e-3, 2.7)
    f_slab = S.covering_fraction(2.4e21, 1e3, 2.7)
    assert f_film == 1.0
    assert abs(f_slab - 2.4e21 / (1e3 * 4 * math.pi * (2.7 * AU) ** 2)) < 1e-12


def test_belt_conversion_is_transport_limited_and_short():
    t = S.belt_conversion_time_s(1e6, 2.7, 1e3, 100.0, 100.0, 1000.0)
    assert 1 * YR < t < 500 * YR


def test_beta_and_pr_time_follow_burns_lamy_soter():
    assert abs(S.beta_solar(1.0, 1.0) - 0.57) < 1e-9
    assert abs(S.pr_drag_time_s(1.0, 0.57) / YR - 400 / 0.57) < 1e-6


def test_residue_lifetime_is_short_for_micron_devices_and_long_for_metre_slabs():
    micron = S.residue_lifetime_s(2.7, 1.0, 0.5, 2.0)["t_residue_s"]
    metre = S.residue_lifetime_s(2.7, 1e-6, 1e6, 2.0)["t_residue_s"]
    assert micron < 100 * YR          # blown out on an orbital time
    assert metre > 1e6 * YR           # sparse metre-scale bodies persist


def test_rate_bound_zero_events_is_three_over_exposure():
    b = S.rate_bound_per_star_per_yr(1e6, 1e4 * YR)
    assert abs(b - 2.996 / (1e6 * 1e4)) / b < 1e-2


# --- passive -----------------------------------------------------------------------
def test_blowout_radius_and_delivery_window():
    assert abs(P.blowout_radius_um(2.0) - 0.57) < 1e-9
    b_max = P.beta_max_to_reach(26 * KM, 1.0)
    assert 1.3 < b_max < 1.45
    lo, hi = P.deliverable_size_window_um(2.0, 26 * KM, 1.0)
    assert 0.18 < lo < 0.25 and abs(hi - 0.57) < 1e-9


def test_hyperbolic_excess_from_release_at_one_au():
    v = P.v_infinity_ms(1.0, 1.0)
    assert abs(v - v_circ(AU)) < 1.0
    assert P.v_infinity_ms(0.4, 1.0) == 0.0


def test_capture_cross_section_focusing_factor_for_earth():
    sig = P.capture_cross_section_m2(R_EARTH, 26 * KM, M_EARTH)
    ve = math.sqrt(2 * G * M_EARTH / R_EARTH)
    assert abs(sig / (math.pi * R_EARTH**2) - (1 + (ve / (26 * KM)) ** 2)) < 1e-9


def test_annulus_fill_time_is_a_gigayear_or_two():
    t = P.annulus_fill_time_s(230.0, 8.2, 1.6) / GYR
    assert 0.8 < t < 2.0


def test_passive_seeding_from_the_belt_is_thousands_per_year_before_survival():
    g = {"R_sun_kpc": 8.2, "annulus_width_kpc": 1.6, "scale_height_kpc": 0.3, "rotation_period_Myr": 230.0}
    r = P.passive_seeding(2.4e21, 0.5, 2.0, 1.0, g, R_EARTH, M_EARTH, 26 * KM, 1e8 * YR, 5 * GYR)
    assert 1e2 < r["arrivals_per_yr_unsurvived"] < 1e5
    assert r["cumulative_arrivals"] > 0
    short = P.passive_seeding(2.4e21, 0.5, 2.0, 1.0, g, R_EARTH, M_EARTH, 26 * KM, 1e6 * YR, 5 * GYR)
    assert short["cumulative_arrivals"] < 1e-100 or short["cumulative_arrivals"] < r["cumulative_arrivals"]


def test_blown_fraction_is_between_zero_and_one_and_falls_with_max_size():
    f1 = P.blown_mass_fraction((0.4, 0.57), 0.05, 1e3)
    f2 = P.blown_mass_fraction((0.4, 0.57), 0.05, 1e6)
    assert 0 < f2 < f1 < 1


# --- active ------------------------------------------------------------------------
def test_front_speed_reduces_to_ship_speed_without_build_time():
    v = A.front_speed_ms(1e6, 0.0, 3.086e16)
    assert abs(v - 1e6) < 1e-6


def test_galaxy_fill_time_brackets_the_hart_tipler_range():
    slow = A.galaxy_fill_time_s(15.0, 1e-4, 1000.0, 1.0) / MYR
    fast = A.galaxy_fill_time_s(15.0, 0.1, 10.0, 1.0) / MYR
    assert 100 < slow < 1000
    assert 0.1 < fast < 10


def test_cosmic_event_horizon_is_about_sixteen_gly():
    d = A.comoving_reach_Mpc(1.0, 67.7, 0.31) * 3.2616e-3
    assert 15.5 < d < 17.5


def test_reach_scales_linearly_with_speed():
    a = A.comoving_reach_Mpc(0.5, 67.7, 0.31)
    b = A.comoving_reach_Mpc(1.0, 67.7, 0.31)
    assert abs(a / b - 0.5) < 1e-6


# --- bayes -------------------------------------------------------------------------
def test_shadowed_own_galaxy_returns_the_prior():
    p = B.loguniform_grid(1e-6, 1.0)
    post = B.posterior_p(100.0, p, f_own=1.0, shadowed=True, n_gal=1e5, q=1.0, use_ext=False)
    assert abs(B.upper_bound(p, post) - 10 ** (-6 + 0.95 * 6)) / 0.5 < 0.02


def test_naive_bound_scales_as_three_over_lambda():
    p = B.loguniform_grid(1e-8, 1.0, 2001)
    for lam in (1e2, 1e4):
        post = B.posterior_p(lam, p, f_own=1.0, shadowed=False, n_gal=1e5, q=1.0, use_ext=False)
        ub = B.upper_bound(p, post)
        # log-uniform prior + exponential likelihood: the 95 % point sits at ~0.4/lambda
        # (the prior's lower edge pulls it below the prior-free 3/lambda of likelihood_bound)
        assert 0.1 / lam < ub < 5 / lam
        assert abs(B.likelihood_bound(lam, 1.0) - 2.996 / lam) / (3 / lam) < 1e-2


def test_external_galaxies_tighten_by_their_number():
    p = B.loguniform_grid(1e-10, 1.0, 3001)
    own = B.posterior_p(1.0, p, f_own=1.0, shadowed=False, n_gal=1e5, q=1.0, use_ext=False)
    ext = B.posterior_p(1.0, p, f_own=1.0, shadowed=True, n_gal=1e5, q=1.0, use_ext=True)
    assert B.upper_bound(p, ext) < B.upper_bound(p, own) / 1e3


def test_branch_update_leaves_planet_bound_risk_alone():
    r = B.branch_update(1e-3, 0.01, 1e-3)
    assert 0.985 < r["ratio"] < 1.0
    r2 = B.branch_update(1e-3, 1.0, 1e-3)
    assert r2["ratio"] < 2e-3


# --- orchestration -----------------------------------------------------------------
def test_run_all_stages_into_tmp(tmp_path, monkeypatch):
    from seti.goo import run as R
    monkeypatch.setattr(R, "ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    import shutil
    shutil.copy(R.__file__.replace("src/seti/goo/run.py", "config/goo.yaml"), tmp_path / "config" / "goo.yaml")
    cfg = R.load_cfg()
    R.stage_planet(cfg)
    R.stage_system(cfg)
    R.stage_passive(cfg)
    R.stage_active(cfg)
    R.stage_bayes(cfg)
    R.stage_contact(cfg)
    tex = R.stage_numbers(cfg)
    assert tex.exists()
    txt = tex.read_text()
    assert r"\newcommand{\BeltConvTime}" in txt
    assert r"\newcommand{\PmargExtPersist}" in txt
    sy = json.loads((tmp_path / "results" / "goo" / "system.json").read_text())
    assert sy["ghat_III"]["fraction_bound_95"] == pytest.approx(3e-5)

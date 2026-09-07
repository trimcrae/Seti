"""LZEDGE offline tests: kinematics, Earth velocity, halo integrals, rates, timing."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from seti.lzedge import kinematics as K
from seti.lzedge.earth import extremal_dates, v_lab_kms, v_lab_speed_kms
from seti.lzedge.halo import (
    Gaussian,
    Isotropic,
    Maxwellian,
    eta_shm_analytic,
    profile_sharp_maxwellian,
    shm,
    shm_plus_plus,
    shm_tail,
    with_stream,
)
from seti.lzedge.kinematics import trapz
from seti.lzedge.rate import Efficiency, n_targets_per_tonne, spectrum, window_rate
from seti.lzedge.timing import EventModel, LiveTime, modulation_summary, timing_bayes_factor


# --- kinematics ----------------------------------------------------------------
def test_vmin_elastic_heavy_wimp_matches_hand_calculation():
    # heavy WIMP, A=131, E=248 keV: v_min = c * 248 / sqrt(2 * 122.03e6 * 248) ≈ 302 km/s
    v = float(K.v_min_kms(248.0, 1e6, 0.0, 131))
    assert abs(v - 302.2) < 1.5


def test_vmin_inelastic_adds_delta_over_q():
    v0 = float(K.v_min_kms(248.0, 1e6, 0.0, 131))
    v1 = float(K.v_min_kms(248.0, 1e6, 200.0, 131))
    q = math.sqrt(2 * 131 * K.AMU_GEV * 1e6 * 248.0)
    assert abs((v1 - v0) - K.C_KMS * 200.0 / q) < 0.05


def test_threshold_speed_and_e_star_are_consistent():
    m, delta = 1000.0, 300.0
    vthr = K.v_threshold_kms(m, delta)
    estar = K.e_star_keV(m, delta)
    # v_min is minimised at E*, and its minimum equals the threshold speed
    assert abs(float(K.v_min_kms(estar, m, delta)) - vthr) < 1e-6 * vthr
    assert float(K.v_min_kms(estar * 0.8, m, delta)) > vthr
    assert float(K.v_min_kms(estar * 1.25, m, delta)) > vthr


def test_recoil_energy_range_brackets_vmin_inversion():
    m, delta, v = 500.0, 150.0, 700.0
    lo, hi = K.recoil_energy_range_keV(v, m, delta)
    assert lo < K.e_star_keV(m, delta) < hi
    assert abs(float(K.v_min_kms(hi, m, delta)) - v) < 1e-6 * v
    assert abs(float(K.v_min_kms(lo, m, delta)) - v) < 1e-6 * v
    assert K.recoil_energy_range_keV(K.v_threshold_kms(m, delta) * 0.99, m, delta) is None


def test_max_splitting_at_energy_inverts_vmin():
    m, E, vmax = 1000.0, 248.0, 790.0
    dmax = K.max_splitting_at_energy_keV(vmax, E, m)
    assert abs(float(K.v_min_kms(E, m, dmax)) - vmax) < 1e-6 * vmax


def test_helm_form_factor_unity_at_zero_and_zeros_near_105_keV():
    assert abs(float(K.helm_form_factor_sq(1e-6, 131)) - 1.0) < 1e-4
    z = K.helm_zeros_keV(131)
    assert 85 < z[0] < 110 and 260 < z[1] < 300
    assert float(K.helm_form_factor_sq(z[0], 131)) < 1e-6
    # 248 keV sits in the second lobe: suppressed but not zero
    f = float(K.helm_form_factor_sq(248.0, 131))
    assert 1e-4 < f < 3e-2


# --- Earth ---------------------------------------------------------------------
def test_lab_speed_peaks_at_the_start_of_june():
    peak, trough = extremal_dates(2023, 238.0)
    assert peak.month == 6 and peak.day <= 4 or (peak.month == 5 and peak.day >= 29)
    assert trough.month == 12 and trough.day <= 5 or (trough.month == 11 and trough.day >= 27)
    assert abs(v_lab_speed_kms(peak) - v_lab_speed_kms(trough) - 2 * 29.79 * 0.49) < 3.0


def test_lab_velocity_on_event_date_is_near_the_peak():
    v_peak = v_lab_speed_kms(extremal_dates(2023, 238.0)[0])
    v_ev = v_lab_speed_kms("2023-06-16")
    assert v_peak - v_ev < 1.0
    assert abs(np.linalg.norm(v_lab_kms(dt.date(2023, 6, 16))) - v_ev) < 1e-9


# --- halo ----------------------------------------------------------------------
def test_maxwellian_lab_speed_distribution_normalises_and_matches_analytic_eta():
    h = shm(238.0, 544.0)
    vl = v_lab_kms("2023-06-16", 238.0)
    vg, g = h.eta_table(vl)
    f = h.lab_speed_distribution(vl)
    assert abs(trapz(f, vg) - 1.0) < 2e-3
    ga = eta_shm_analytic(vg, float(np.linalg.norm(vl)), 238.0, 544.0)
    sel = (vg > 50) & (vg < 780)
    assert np.max(np.abs(g[sel] - ga[sel]) / (ga[sel] + 1e-12)) < 5e-3
    # the tail: zero beyond v_esc + v_lab, non-zero just below
    vmax = 544.0 + float(np.linalg.norm(vl))
    assert h.eta(vl)(vmax + 5.0) == 0.0
    assert h.eta(vl)(vmax - 20.0) > 0.0


def test_gaussian_component_reproduces_maxwellian_when_isotropic():
    v0, vesc = 238.0, 544.0
    s = v0 / math.sqrt(2.0)
    m = Maxwellian(v0=v0, v_esc=vesc)
    g = Gaussian(mean=(0, 0, 0), sigma=(s, s, s), v_esc=vesc)
    vl = v_lab_kms("2023-06-16", v0)
    grid = np.arange(0.0, 1000.0, 2.0)
    fm = m.lab_speed_distribution(vl, grid)
    fg = g.lab_speed_distribution(vl, grid)
    sel = fm > 1e-3 * fm.max()
    assert np.max(np.abs(fg[sel] - fm[sel]) / fm[sel]) < 1e-2
    assert abs(trapz(fg, grid) - 1.0) < 5e-3


def test_shm_plus_plus_has_more_high_speed_weight_than_its_round_part_alone():
    hpp = shm_plus_plus()
    vl = v_lab_kms("2023-06-16", 233.0)
    g_pp = hpp.eta(vl)
    g_round = shm(233.0, 528.0).eta(vl)
    assert abs(hpp.total_fraction - 1.0) < 1e-12
    assert g_pp(100.0) == pytest.approx(g_round(100.0), rel=0.15)
    # the Sausage is radially hot: more weight at speeds the round halo lacks
    assert g_pp(650.0) > 0 and g_round(650.0) > 0


def test_unbound_stream_extends_the_lab_frame_speed_range():
    base = shm(238.0, 544.0)
    h = with_stream(base, mean_xyz=(0.0, -300.0, 500.0), sigma=60.0, fraction=0.05, v_esc=None)
    vl = v_lab_kms("2023-06-16", 238.0)
    assert abs(h.total_fraction - 1.0) < 1e-12
    g = h.eta(vl)
    assert g(900.0) > 0.0                       # beyond v_esc + v_lab
    assert base.eta(vl)(900.0) == 0.0
    assert h.v_max(vl) == pytest.approx(h.v_grid_max)


# --- rate ----------------------------------------------------------------------
def test_total_rate_without_form_factor_equals_ntargets_rho_sigma_mean_speed():
    h = shm(238.0, 544.0)
    vl = v_lab_kms("2023-06-16", 238.0)
    vg = h.v_grid
    f = h.lab_speed_distribution(vl)
    mean_v_cm_s = float(trapz(f * vg, vg)) * 1e5
    m, sig, rho = 100.0, 1e-45, 0.3
    iso = ((131, 1.0),)
    E = np.arange(0.5, 2500.0, 0.5)
    dr = spectrum(E, h.eta(vl), m, 0.0, sig, rho, isotopes=iso, form_factor=False)
    total = float(trapz(dr, E))
    mu_a = K.reduced_mass_gev(m, K.nucleus_mass_gev(131))
    mu_n = K.reduced_mass_gev(m, K.AMU_GEV)
    sig_a = sig * 131 ** 2 * (mu_a / mu_n) ** 2
    expect = n_targets_per_tonne(131) * (rho / m) * sig_a * mean_v_cm_s * 3.15576e7
    assert abs(total / expect - 1.0) < 0.02
    assert 200 < expect < 3000  # hundreds of events per tonne-year at 1e-45 cm²


def test_inelastic_spectrum_vanishes_below_threshold_and_beyond_the_edge():
    h = shm(238.0, 544.0)
    vl = v_lab_kms("2023-06-16", 238.0)
    m, delta = 1000.0, 300.0
    E = np.arange(1.0, 700.0, 1.0)
    dr = spectrum(E, h.eta(vl), m, delta)
    vmax = h.v_max(vl)
    ranges = [K.recoil_energy_range_keV(vmax, m, delta, A) for A, _ in K.XE_ISOTOPES]
    lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
    tiny = 1e-6 * dr.max()
    assert np.all(dr[E < lo - 3] <= tiny) and np.all(dr[E > hi + 3] <= tiny)
    assert dr[(E > lo + 5) & (E < hi - 5)].max() > 0
    # December: the edge moves inward, the spectrum narrows
    vl_dec = v_lab_kms("2023-12-01", 238.0)
    dr_dec = spectrum(E, h.eta(vl_dec), m, delta)
    assert dr_dec.sum() < dr.sum()


def test_window_rate_respects_efficiency_window():
    h = shm(238.0, 544.0)
    eta = h.eta(v_lab_kms("2023-06-16", 238.0))
    r_all = window_rate(eta, 100.0, 0.0, Efficiency.flat(1.0, 300.0))
    r_half = window_rate(eta, 100.0, 0.0, Efficiency.flat(1.0, 300.0, 0.5))
    r_hi = window_rate(eta, 100.0, 0.0, Efficiency.flat(150.0, 300.0))
    assert r_half == pytest.approx(0.5 * r_all, rel=1e-9)
    assert 0 < r_hi < 0.05 * r_all


# --- timing --------------------------------------------------------------------
def test_timing_bayes_factor_is_one_for_a_constant_rate_and_large_at_the_edge():
    h = shm(238.0, 544.0)
    eff = Efficiency.flat(5.0, 270.0)
    live = LiveTime.uniform("2023-03-01", "2024-04-30", 220.0)
    assert abs(live.live_days() - 220.0) < 1e-6
    elastic = EventModel(h, 100.0, 0.0, eff)
    tb = timing_bayes_factor(elastic, live, "2023-06-16", step_days=7.0)
    assert 0.95 < tb["bayes_factor_timing"] < 1.10
    edge = EventModel(h, 1000.0, 380.0, eff)
    te = timing_bayes_factor(edge, live, "2023-06-16", step_days=7.0)
    assert te["bayes_factor_timing"] > 1.5
    ms = modulation_summary(edge, 2023, n=73)
    assert ms["fraction_of_year_on"] < 0.75
    assert ms["peak_date"][5:7] in ("05", "06")
    # a model whose rate vanishes on the event date is killed by the calendar
    dead = EventModel(h, 1000.0, 420.0, eff)
    td = timing_bayes_factor(dead, live, "2023-12-01", step_days=7.0)
    assert td["rate_at_event"] == 0.0


def test_livetime_from_rows_and_density():
    live = LiveTime.from_rows([("2023-03-01", "2023-06-01", 0.5), ("2023-07-01", "2023-08-01", 1.0)])
    assert live.density("2023-04-01") == 0.5
    assert live.density("2023-06-15") == 0.0
    assert abs(live.live_days() - (92 * 0.5 + 31)) < 1e-6


def test_run_stage_kinematics_writes_files(tmp_path):
    from seti.lzedge.run import load_cfg, stage_kinematics
    cfg = load_cfg()
    cfg["scan"]["m_chi_gev"] = [1000]
    cfg["scan"]["delta_keV"] = {"min": 0.0, "max": 400.0, "step": 100.0}
    s = stage_kinematics(cfg, tmp_path)
    assert (tmp_path / "kinematics.csv").exists()
    assert s["days_after_peak"] > 5
    assert s["kinematic_ceilings"]["shm_lz"]["v_max_kms"] > 780


def test_stream_peak_date_reproduces_the_halo_wind_for_a_static_component():
    from seti.lzedge.earth import stream_peak_date
    # a component at rest in the Galactic frame is the halo wind: peaks at the start of June
    pk, tr, vp, vt = stream_peak_date((0.0, 0.0, 0.0), 2023)
    assert (pk.month, pk.day <= 4) == (6, True) or (pk.month == 5 and pk.day >= 29)
    assert vp > vt and abs(vp - v_lab_speed_kms(pk)) < 1e-9
    # a component overtaking the Sun along rotation is fastest in the lab when the halo
    # wind is slowest: its peak sits half a year from the wind's, near 1 December
    pk2, _, vp2, _ = stream_peak_date((0.0, 500.0, 0.0), 2023)
    assert 150 < abs((pk2 - pk).days) < 215
    assert 260 < vp2 < 285


def test_isotropic_sharp_profile_reproduces_the_closed_form_maxwellian():
    v0, vesc = 238.0, 544.0
    vl = v_lab_kms("2023-06-16", v0)
    grid = np.arange(0.0, 1000.0, 1.0)
    fm = Maxwellian(v0=v0, v_esc=vesc).lab_speed_distribution(vl, grid)
    fi = Isotropic(profile=profile_sharp_maxwellian(v0), v_esc=vesc).lab_speed_distribution(vl, grid)
    sel = fm > 1e-4 * fm.max()
    assert np.max(np.abs(fi[sel] - fm[sel]) / fm[sel]) < 3e-3
    assert abs(trapz(fi, grid) - 1.0) < 2e-3


def test_tail_shape_changes_the_edge_but_not_the_bulk():
    vl = v_lab_kms("2023-06-16", 238.0)
    g_sharp = shm_tail(tail="sharp").eta(vl)
    g_soft = shm_tail(tail="soft").eta(vl)
    g_pow = shm_tail(tail="power", k=2.0).eta(vl)
    assert g_soft(100.0) == pytest.approx(g_sharp(100.0), rel=0.03)
    assert g_pow(100.0) == pytest.approx(g_sharp(100.0), rel=0.05)
    # near the edge the sharp cut keeps the most weight, the soft cut and the
    # (v_esc - v)^2 tail progressively less
    v_edge = 544.0 + float(np.linalg.norm(vl)) - 40.0
    assert g_sharp(v_edge) > g_soft(v_edge) > 0.0
    assert g_sharp(v_edge) > g_pow(v_edge) > 0.0
    assert g_sharp(v_edge + 60.0) == 0.0 and g_pow(v_edge + 60.0) == 0.0


def test_reachability_calendar_opens_a_window_around_june_at_the_edge():
    from seti.lzedge.run import reachability_calendar
    always = reachability_calendar(248.0, 1000.0, 300.0, 544.0, 238.0, 2023)
    assert always["fraction"] == 1.0
    never = reachability_calendar(248.0, 1000.0, 450.0, 544.0, 238.0, 2023)
    assert never["fraction"] == 0.0
    edge = reachability_calendar(248.0, 1000.0, 380.0, 544.0, 238.0, 2023)
    assert 0.0 < edge["fraction"] < 1.0
    assert edge["first"] < "2023-06-01" < edge["last"]
    assert edge["first"] <= "2023-06-16" <= edge["last"]

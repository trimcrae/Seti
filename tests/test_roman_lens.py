"""Offline test suite for S40 --- the opaque lens (``seti.roman.lens``).

No network anywhere.  The suite checks the point-lens occultation algebra
against hand-computed values, the finite-source integrator against a brute
force source-centred grid and the analytic flattened peak, recovers an injected
occulting event with a symmetric kink pair and the right rho_L, and returns
clean nulls on the confounders the design names: a pure Paczynski event, a
finite-source event, a blended event, a one-sided (caustic-like) anomaly, a
constant star, and a curve too short to fit.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from seti.roman import lens as L
from seti.roman.schema import LightCurve, json_safe, load_roman_config

T0, TE = 60000.0, 20.0
TIER_ORDER = {"NOT_LENSING": 0, "LENSING_NO_OCCULTATION": 1,
              "OCCULTING_PREFERRED_PENDING_VET": 2, "OPAQUE_LENS_CANDIDATE": 3}


@pytest.fixture(scope="module")
def conf():
    c = load_roman_config()
    assert c.get("lens"), "config/roman.yaml must carry a lens: block"
    return c


# --------------------------------------------------------------------------------------
# Point-source algebra
# --------------------------------------------------------------------------------------

def test_image_positions_and_magnifications():
    u = np.array([0.0, 0.5, 1.0, 3.0])
    tp, tm = L.image_positions(u)
    assert np.allclose(tp - np.abs(tm), u)               # theta_+ - |theta_-| = u
    assert np.allclose(tp * np.abs(tm), 1.0)             # images straddle the ring
    ap, am = L.magnifications(u)
    assert np.allclose(ap - am, 1.0)                     # A_+ - A_- = 1 at every u
    assert np.isclose(ap[2] + am[2], 3.0 / np.sqrt(5.0))  # A(u=1) = 1.3416
    assert np.isfinite(ap[0]) and ap[0] > 1e5            # u -> 0 floored, not inf
    assert np.allclose(L.u_from_magnification(L.paczynski_magnification(u[1:])), u[1:])


def test_wing_occultation_rho_half():
    rho_l = 0.5
    assert np.isclose(L.u_crit(rho_l), 1.5)
    assert L.regime(rho_l) == "wing_occultation"
    _, am = L.magnifications(1.5)
    assert np.isclose(L.step_depth(rho_l), am)
    assert abs(L.step_depth(rho_l) - 0.0667) < 1e-3
    u_in = np.linspace(0.01, 1.499, 200)
    assert np.allclose(L.occulted_magnification_point(u_in, rho_l), L.paczynski_magnification(u_in))
    u_out = np.linspace(1.501, 5.0, 200)
    ap, am = L.magnifications(u_out)
    assert np.allclose(L.occulted_magnification_point(u_out, rho_l), ap)
    # the step at u_c is exactly A_-(u_c)
    just_in = L.occulted_magnification_point(1.5 - 1e-9, rho_l)
    just_out = L.occulted_magnification_point(1.5 + 1e-9, rho_l)
    assert abs((just_in - just_out) - L.step_depth(rho_l)) < 1e-6
    # the step at u_c is the ONLY discontinuity: inside it the occulted curve tracks Paczynski
    both = np.linspace(0.3, 2.5, 2000)
    extra = np.diff(L.occulted_magnification_point(both, rho_l)) - np.diff(L.paczynski_magnification(both))
    assert np.sum(np.abs(extra) > 1e-3) == 1


def test_central_hole_rho_1p2():
    rho_l = 1.2
    u_h = rho_l - 1.0 / rho_l
    assert np.isclose(L.u_crit(rho_l), u_h)
    assert L.regime(rho_l) == "central_hole"
    u_hole = np.linspace(0.0, u_h - 1e-6, 50)
    assert np.all(L.occulted_magnification_point(u_hole, rho_l) == 0.0)   # zero source flux
    u_wing = np.linspace(u_h + 1e-6, 4.0, 50)
    ap, _ = L.magnifications(u_wing)
    assert np.allclose(L.occulted_magnification_point(u_wing, rho_l), ap)  # minor image gone
    assert np.isclose(L.step_depth(rho_l), L.magnifications(u_h)[0])


def test_table_of_the_design_document():
    # docs/roman.md section 2.1: rho_L -> (u_c, A_-(u_c)).  The table quotes 0.7 % for
    # rho_L = 0.3; the exact A_-(3.033) is 0.82 %, hence the looser tolerance on that row.
    for rho_l, uc, depth, tol in ((0.3, 3.03, 0.007, 0.2), (0.5, 1.50, 0.067, 0.02),
                                  (0.7, 0.73, 0.31, 0.03), (0.9, 0.21, 1.9, 0.03)):
        assert abs(L.u_crit(rho_l) - uc) < 0.01
        assert abs(L.step_depth(rho_l) / depth - 1.0) < tol


# --------------------------------------------------------------------------------------
# Finite source
# --------------------------------------------------------------------------------------

def test_finite_source_reduces_to_point_source():
    u = np.linspace(0.05, 3.0, 40)
    a_pt = L.paczynski_magnification(u)
    a_fs = L.paczynski_finite(u, 0.001)
    assert np.max(np.abs(a_fs / a_pt - 1.0)) < 1e-3
    o_pt = L.occulted_magnification_point(u, 0.6)
    o_fs = L.occulted_magnification_finite(u, 0.6, 0.001)
    assert np.max(np.abs(o_fs / o_pt - 1.0)) < 1e-3


def test_finite_source_peak_flattening():
    for rho in (0.1, 0.3, 0.5, 1.0):
        a0 = L.paczynski_finite(np.array([0.0]), rho)[0]
        assert abs(a0 / np.sqrt(1.0 + 4.0 / rho**2) - 1.0) < 0.03


def _brute_disc(u, rho, rho_l=None, n=300):
    r = (np.arange(n) + 0.5) / n * rho
    ph = (np.arange(2 * n) + 0.5) / (2 * n) * 2 * np.pi
    rr, pp = np.meshgrid(r, ph, indexing="ij")
    up = np.sqrt(u * u + rr * rr + 2 * u * rr * np.cos(pp))
    a = L.occulted_magnification_point(up, rho_l) if rho_l else L.paczynski_magnification(up)
    return float(np.sum(a * rr) / np.sum(rr))


def test_disc_integrator_matches_brute_force():
    for rho in (0.3, 0.05):
        for u in (0.1, 0.5, 1.0, 1.05, 1.2, 2.0):
            a = L.paczynski_finite(np.array([u]), rho)[0]
            assert abs(a / _brute_disc(u, rho) - 1.0) < 3e-3
            o = L.occulted_magnification_finite(np.array([u]), 0.6, rho)[0]
            assert abs(o / _brute_disc(u, rho, 0.6) - 1.0) < 3e-3


def test_finite_source_rounds_the_step():
    rho_l = 0.6
    uc = L.u_crit(rho_l)
    u = np.linspace(uc - 0.3, uc + 0.3, 601)
    sharp = L.occulted_magnification_point(u, rho_l)
    soft = L.occulted_magnification_finite(u, rho_l, 0.1)
    assert np.max(np.abs(np.diff(sharp))) > 0.1              # a jump
    assert np.max(np.abs(np.diff(soft))) < 0.01              # rounded over ~rho_*
    # away from the edge the finite-source curve differs from the point source by O(rho_*^2)
    assert np.allclose(soft[0], sharp[0], rtol=1e-2) and np.allclose(soft[-1], sharp[-1], rtol=1e-2)


def test_trajectory_and_model_flux():
    t = np.array([T0 - TE, T0, T0 + TE])
    assert np.allclose(L.trajectory(t, T0, TE, 0.3), [np.hypot(0.3, 1), 0.3, np.hypot(0.3, 1)])
    p = {"t0": T0, "tE": TE, "u0": 0.3, "fs": 2.0, "fb": 0.5}
    f = L.model_flux(t, p, "paczynski")
    assert np.isclose(f[1], 2.0 * L.paczynski_magnification(0.3) + 0.5)
    with pytest.raises(ValueError):
        L.model_flux(t, p, "no_such_model")


# --------------------------------------------------------------------------------------
# Injection and rejection through the screen
# --------------------------------------------------------------------------------------

def test_injection_recovers_occulting_event(conf):
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=1.0, fb=0.1, rho_l=0.6, seed=11)
    assert lc.n > 8000 and abs(np.median(np.diff(lc.mjd)) - 12.0 / 1440.0) < 1e-9
    t0 = time.time()
    res = L.screen_lightcurve(lc, conf, theta_star_uas=1.0)
    elapsed = time.time() - t0
    fit = res["fit"]
    assert fit["dchi2_paczynski_vs_flat"] > conf["lens"]["paczynski_delta_chi2_min"]
    assert fit["dchi2_occult_vs_paczynski"] > conf["lens"]["occult_delta_chi2_min"]
    assert fit["dbic_occult_vs_finite"] > conf["lens"]["occult_delta_bic_min"]
    assert fit["dbic_occult_vs_paczynski"] > conf["lens"]["occult_delta_bic_min"]
    assert res["regime"] == "wing_occultation"
    assert abs(res["rho_l"] / 0.6 - 1.0) < 0.15
    assert fit["kink_status"] == "both_sides"
    assert fit["kink_asymmetry"] < conf["lens"]["kink_asymmetry_max"]
    assert abs(fit["u_c_rise"] / L.u_crit(0.6) - 1.0) < 0.1
    assert abs(fit["u_c_fall"] / L.u_crit(0.6) - 1.0) < 0.1
    assert abs(fit["step_depth_measured"] / fit["step_depth_predicted"] - 1.0) < 0.15
    assert "kink_asymmetric" not in res["rejections"]
    assert not any(r.startswith("occult_not") for r in res["rejections"])
    assert res["tier"] in (conf["lens"]["tiers"]["interest"], conf["lens"]["tiers"]["candidate"])
    assert "colour_test_not_run" in res["pending"]
    assert res["density"]["blend_moving_test_needed"] is True
    assert res["density"]["theta_e_kind"] == "lower_bound_from_a_max"
    json.dumps(json_safe(res))          # every stage writes it
    assert elapsed < 30.0


def test_pure_paczynski_is_lensing_without_occultation(conf):
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=1.0, fb=0.1, seed=2)
    res = L.screen_lightcurve(lc, conf)
    assert res["tier"] == "LENSING_NO_OCCULTATION"
    assert res["fit"]["dchi2_paczynski_vs_flat"] > conf["lens"]["paczynski_delta_chi2_min"]
    assert any(r.startswith("occult_") for r in res["rejections"])


def test_finite_source_event_is_not_occulting(conf):
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=1.0, fb=0.1, rho_star=0.5, seed=3)
    res = L.screen_lightcurve(lc, conf)
    assert res["tier"] in ("LENSING_NO_OCCULTATION", "NOT_LENSING")
    assert "occult_not_preferred_vs_finite" in res["rejections"]
    assert abs(res["fit"]["models"]["finite"]["params"]["rho_star"] / 0.5 - 1.0) < 0.1


def test_blended_event_is_not_occulting(conf):
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.3, fs=0.3, fb=0.7, seed=4)
    res = L.screen_lightcurve(lc, conf)
    assert res["tier"] == "LENSING_NO_OCCULTATION"
    p = res["fit"]["models"]["paczynski"]["params"]
    assert abs(p["fb"] - 0.7) < 0.1 and abs(p["fs"] - 0.3) < 0.1


def test_one_sided_anomaly_fails_the_symmetry_gate(conf):
    def bump(t):                      # a caustic-like feature on the falling side only
        return 0.3 * np.exp(-0.5 * ((t - (T0 + 0.8 * TE)) / (0.05 * TE)) ** 2)

    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=1.0, fb=0.1, seed=5, anomaly=bump)
    res = L.screen_lightcurve(lc, conf, theta_star_uas=1.0)
    # A one-sided bump is time-asymmetric about t0, so the (earlier) time-symmetry
    # gate may reject it before the kink filter does; either rejection is the point.
    assert {"kink_asymmetric", "time_asymmetric"} & set(res["rejections"])
    assert TIER_ORDER[res["tier"]] <= TIER_ORDER["OCCULTING_PREFERRED_PENDING_VET"]


def test_constant_star_is_not_lensing(conf):
    lc = L.synthesise_event(t0=T0, tE=TE, u0=50.0, fs=1.0, fb=0.1, seed=6)   # no event in window
    res = L.screen_lightcurve(lc, conf)
    assert res["tier"] == "NOT_LENSING"
    assert res["rejections"] and res["fit"] is None


def test_too_few_epochs_degrades_honestly(conf):
    t = T0 + np.linspace(-10, 10, 20)
    lc = LightCurve("short", 0.0, 0.0, "F146", t, np.ones(20), np.full(20, 0.01))
    res = L.screen_lightcurve(lc, conf)
    assert res["tier"] == "NOT_LENSING"
    assert res["rejections"] == ["too_few_epochs"]
    assert res["fit"] is None and res["density"] is None and res["colour"] is None


def test_dq_rejected_epochs_are_dropped(conf):
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.2, seed=7)
    lc.dq[:] = 2                                     # SATURATED everywhere
    res = L.screen_lightcurve(lc, conf)
    assert res["n_used"] == 0 and res["rejections"] == ["too_few_epochs"]


# --------------------------------------------------------------------------------------
# Achromaticity
# --------------------------------------------------------------------------------------

def test_colour_consistency_zscore():
    assert L.colour_consistency(0.1, 0.01, 0.1, 0.01) == 0.0
    assert abs(L.colour_consistency(0.15, 0.03, 0.10, 0.04) - 1.0) < 1e-9
    assert L.colour_consistency(1.0, 0.0, 0.5, 0.0) == float("inf")


def test_colour_band_achromatic_and_chromatic(conf):
    prim = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=1.0, fb=0.1, rho_l=0.6, seed=21)
    grey = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=0.6, fb=0.3, rho_l=0.6, seed=22,
                              cadence_min=720.0, noise_frac=0.01, band="F087")
    res = L.screen_lightcurve(prim, conf, colour_lc=grey, theta_star_uas=1.0)
    assert res["colour"] is not None
    assert abs(res["colour"]["z"]) <= conf["lens"]["colour_deficit_sigma_max"]
    assert "colour_chromatic" not in res["rejections"]
    assert "colour_test_not_run" not in res["pending"]
    # the same event with NO deficit in the colour band is chromatic
    chrom = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=0.6, fb=0.3, seed=23,
                               cadence_min=720.0, noise_frac=0.01, band="F087")
    res2 = L.screen_lightcurve(prim, conf, colour_lc=chrom, theta_star_uas=1.0)
    assert "colour_chromatic" in res2["rejections"]
    assert TIER_ORDER[res2["tier"]] <= TIER_ORDER["OCCULTING_PREFERRED_PENDING_VET"]


# --------------------------------------------------------------------------------------
# Physical scale
# --------------------------------------------------------------------------------------

def test_theta_e_lower_bound_numeric():
    # A_max = sqrt(5): sqrt(A^2-1)/2 = 1, so theta_E >= theta_* exactly (1 uas -> 1e-3 mas)
    assert np.isclose(L.theta_e_lower_bound(np.sqrt(5.0), 1.0), 1e-3)
    assert np.isclose(L.theta_e_lower_bound(np.sqrt(5.0), 5.0), 5e-3)
    assert L.theta_e_lower_bound(1.0, 3.0) == 0.0
    # consistency with the finite-source peak: the bound saturates at rho_* = theta_*/theta_E
    a = L.paczynski_finite(np.array([0.0]), 0.5)[0]
    assert np.isclose(L.theta_e_lower_bound(a, 1.0), 1e-3 / 0.5, rtol=1e-3)


def test_lens_scale_units():
    s = L.lens_scale(0.6, 0.5, 0.2, 8.0)
    assert np.isclose(s["r_au"], 0.6 * 0.5 * 0.2)              # 1 mas x 1 kpc = 1 AU
    assert np.isclose(s["pi_rel_mas"], 5.0 - 0.125)
    assert np.isclose(s["mass_msun"], 0.25 / (8.144 * 4.875))
    vol = 4 / 3 * np.pi * (s["r_km"] * 1e5) ** 3
    assert np.isclose(s["density_g_cc"], s["mass_msun"] * L.MSUN_G / vol)


def test_density_bound_wide_einstein_radius_is_unnatural():
    d = L.density_bound(0.6, 0.5, 8.0, grid=200)
    far = d["curve"]["d_l_kpc"] > 0.2
    assert np.max(d["curve"]["density_g_cc"][far]) < 0.01
    assert d["density_max_g_cc"] < 0.01
    win = d["natural_distance_window_kpc"]
    assert win is None or win[1] <= 0.2
    assert 0.0 < d["natural_distance_max_kpc"] <= 0.2
    assert d["blend_moving_test_needed"] is True
    assert d["far_branch_window_kpc"] is None
    assert len(d["curve"]["d_l_kpc"]) == 200 and d["curve"]["d_l_kpc"][-1] < 8.0
    json.dumps(json_safe(d))


def test_density_bound_small_einstein_radius_is_natural():
    d = L.density_bound(0.05, 0.0003, 8.0, grid=200)
    assert d["natural_distance_window_kpc"] is not None
    assert d["natural_distance_max_kpc"] > 0.2
    assert d["blend_moving_test_needed"] is False


def test_density_bound_pinned_by_parallax():
    d = L.density_bound(0.6, 0.5, 8.0, pi_e=0.2)
    pin = d["pinned"]
    assert np.isclose(1.0 / pin["d_l_kpc"] - 1.0 / 8.0, 0.2 * 0.5)
    assert pin["density_g_cc"] < 0.01 and pin["natural"] is False


# --------------------------------------------------------------------------------------
# Kink filter in isolation and the central-hole regime
# --------------------------------------------------------------------------------------

def test_kink_filter_on_noiseless_truth(conf):
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=1.0, fb=0.1, rho_l=0.6, noise_frac=1e-3, seed=8)
    truth = dict(lc.meta["truth"])
    kk = L.kink_test(lc.mjd, lc.flux, lc.flux_err, truth, conf)
    assert kk["status"] == "both_sides"
    assert abs(kk["u_c_rise"] / L.u_crit(0.6) - 1.0) < 0.02
    assert abs(kk["u_c_fall"] / L.u_crit(0.6) - 1.0) < 0.02
    assert abs(kk["d_measured"] - 1.0) < 0.05
    # and finds nothing on a pure Paczynski curve with the same geometry
    lc2 = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=1.0, fb=0.1, noise_frac=1e-3, seed=9)
    kk2 = L.kink_test(lc2.mjd, lc2.flux, lc2.flux_err, truth, conf)
    assert kk2["status"] == "not_detected"


def test_central_hole_event_through_the_screen(conf):
    rho_l = 1.3
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.1, fs=1.0, fb=0.1, rho_l=rho_l, seed=15)
    res = L.screen_lightcurve(lc, conf, theta_star_uas=1.0)
    fit = res["fit"]
    assert res["regime"] == "central_hole"
    assert abs(res["rho_l"] / rho_l - 1.0) < 0.15
    assert fit["dbic_occult_vs_box"] > conf["lens"]["occult_delta_bic_min"]   # the wings are there
    assert fit["kink_status"] == "both_sides" and fit["kink_asymmetry"] < 0.05
    assert not any(r.startswith("occult_not") for r in res["rejections"])
    assert "hole_without_wings" not in res["rejections"]
    # a hard-edged truth pins rho_star at its floor: theta_E must NOT be claimed from it
    assert res["density"]["theta_e_kind"] == "lower_bound_from_a_max"
    json.dumps(json_safe(res))


def test_total_eclipse_is_a_hole_without_wings(conf):
    t = np.arange(T0 - 36.0, T0 + 36.0, 12.0 / 1440.0)
    model = np.where(np.abs(t - T0) < 5.0, 0.1, 1.1)
    rng = np.random.default_rng(16)
    lc = LightCurve("eclipse", 0.0, 0.0, "F146", t, model + rng.normal(0, 0.011, t.size),
                    np.full(t.size, 0.011))
    res = L.screen_lightcurve(lc, conf)
    assert res["tier"] in ("NOT_LENSING", "LENSING_NO_OCCULTATION")
    if res["tier"] == "LENSING_NO_OCCULTATION":
        assert "hole_without_wings" in res["rejections"] or "rho_l_out_of_range" in res["rejections"]


def test_finite_source_occulter_measures_theta_e(conf):
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.2, fs=1.0, fb=0.1, rho_l=0.6, rho_star=0.05, seed=12)
    res = L.screen_lightcurve(lc, conf, theta_star_uas=1.0)
    fit = res["fit"]
    assert fit["best_occult_kind"] == "occult_finite" and fit["rho_star_resolved"]
    assert abs(fit["models"]["occult_finite"]["params"]["rho_star"] / 0.05 - 1.0) < 0.2
    assert abs(res["rho_l"] / 0.6 - 1.0) < 0.15
    assert res["density"]["theta_e_kind"] == "from_finite_source"
    assert abs(res["density"]["theta_e_mas"] / 0.02 - 1.0) < 0.2       # 1 uas / 0.05
    assert res["density"]["blend_moving_test_needed"] is True
    assert not any(r.startswith("occult_not") for r in res["rejections"])


def test_central_hole_kink_geometry(conf):
    rho_l = 1.3
    lc = L.synthesise_event(t0=T0, tE=TE, u0=0.1, fs=1.0, fb=0.1, rho_l=rho_l, noise_frac=1e-3, seed=10)
    truth = dict(lc.meta["truth"])
    assert np.min(lc.flux) < 0.15                       # the hole: blend only
    kk = L.kink_test(lc.mjd, lc.flux, lc.flux_err, truth, conf)
    assert kk["regime"] == "central_hole" and kk["status"] == "both_sides"
    assert abs(kk["u_c_measured"] / L.u_crit(rho_l) - 1.0) < 0.05
    assert kk["kink_asymmetry"] < 0.05


def test_a_supernova_shaped_transient_is_not_a_lens():
    """Run 34349717932 put 1,262 of 2,688 simulated SN Ia curves in the occulting
    tier: the (symmetric) occulting model out-fits Paczynski on an asymmetric
    transient for the wrong reason.  A fast-rise / slow-decline curve must now
    fail the time-symmetry gate and come out NOT_LENSING."""
    conf = load_roman_config()
    rng = np.random.default_rng(3)
    t = 60000.0 + np.arange(0, 120, 12.0 / 1440.0 * 60)      # one epoch per hour for 120 d
    dt = t - 60040.0
    model = 1.0 + 3.0 * np.where(dt < 0, np.exp(dt / 4.0), np.exp(-dt / 25.0))   # rise 4 d, decline 25 d
    f = model * (1 + 0.01 * rng.standard_normal(t.size))
    lc = LightCurve("syn_sn", 10.0, -44.0, "F146", t, f, 0.01 * model, flux_zp_ab=27.7)
    rec = L.screen_lightcurve(lc, conf)
    assert rec["tier"] == "NOT_LENSING"
    assert "time_asymmetric" in rec["rejections"]
    assert rec["time_symmetry"]["status"] == "ok" and rec["time_symmetry"]["chi2_red"] > 3.0


def test_the_time_symmetry_gate_passes_a_real_occulting_event():
    conf = load_roman_config()
    lc = L.synthesise_event(rho_l=0.6, u0=0.2, tE=20.0, seed=11, star_id="syn_occ_sym")
    rec = L.screen_lightcurve(lc, conf)
    assert "time_asymmetric" not in rec["rejections"]
    assert rec["time_symmetry"]["status"] == "ok" and rec["time_symmetry"]["chi2_red"] < 3.0

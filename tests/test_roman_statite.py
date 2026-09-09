"""Offline suite for S43 --- the statite (``seti.roman.statite``).

No network anywhere (``conftest.py`` enforces it).  Every synthetic source goes
through the same functions the screen will run on CGI products: the injected
signature (a source that does not move) is recovered as ``FIXED_PREFERRED``, the
two confounders (a Keplerian companion, a background star with the foreground
parallax and proper motion) come back as themselves, an arc too short to test
Kepler is ``UNRESOLVED`` rather than a candidate, and an empty input is
``NO_DATA_REACHED``.  The one physical degeneracy --- an edge-on e ~ 0.95 orbit
seen at its projected turning point --- is asserted to be *reported*, with the
baseline that would exclude it, not hidden.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from seti.roman.schema import load_roman_config
from seti.roman.statite import (
    E_MAX,
    E_MAX_EXTREME,
    PLANET_TEMPLATE,
    CGISource,
    arc_testability,
    assess_sources,
    astrometric_verdict,
    fit_fixed,
    fit_kepler,
    fit_linear,
    kepler_period_yr,
    kepler_xy,
    lambert_phase,
    parallax_factors,
    phase_function_test,
    reflectance_test,
    screen_source,
    solve_kepler,
    synthesise_source,
)

BANDS = ("B1", "B2", "B3", "B4")


@pytest.fixture(scope="module")
def conf():
    c = load_roman_config()
    assert "statite" in c, "config/roman.yaml must carry the statite block"
    return c


def _grey(c0=1e-8, rel_err=0.02):
    return {b: (c0, rel_err * c0) for b in BANDS}


def _jupiter(c0=1e-8, rel_err=0.02):
    return {b: (c0 * PLANET_TEMPLATE[b], rel_err * c0) for b in BANDS}


def _specular_phase(rng, n=10, alpha0=60.0, width=4.0, c0=1e-8, err=2e-10):
    al = np.linspace(20.0, 110.0, n)
    c = c0 * np.exp(-0.5 * ((al - alpha0) / width) ** 2) + rng.normal(0.0, err, n)
    return al, c, np.full(n, err)


def _lambert_phase(rng, n=10, c0=1e-8, err=2e-10):
    al = np.linspace(20.0, 110.0, n)
    c = c0 * lambert_phase(np.radians(al)) + rng.normal(0.0, err, n)
    return al, c, np.full(n, err)


# --------------------------------------------------------------------------------------
# Shared, expensive screens (module scope keeps the suite well under the budget)
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixed_candidate_screen(conf):
    rng = np.random.default_rng(11)
    al, c, e = _specular_phase(rng)
    src = synthesise_source("fixed", 5, 1.5, 300.0, 1.0, 10.0, 2.0, rng, source_id="fx_cand",
                            contrast_by_band=_grey(), phase_angle_deg=al, contrast_per_epoch=c,
                            contrast_err_per_epoch=e, polarization=(0.45, 0.05))
    return screen_source(src, conf)


@pytest.fixture(scope="module")
def fixed_interest_screen(conf):
    rng = np.random.default_rng(12)
    src = synthesise_source("fixed", 5, 1.5, 300.0, 1.0, 10.0, 2.0, rng, source_id="fx_int",
                            polarization=(0.1, 0.03))
    return screen_source(src, conf)


@pytest.fixture(scope="module")
def kepler_screen(conf):
    rng = np.random.default_rng(13)
    src = synthesise_source("kepler", 5, 1.5, 300.0, 1.0, 10.0, 2.0, rng, source_id="kep",
                            inc_deg=30.0, contrast_by_band=_jupiter())
    return screen_source(src, conf)


@pytest.fixture(scope="module")
def background_screen(conf):
    rng = np.random.default_rng(14)
    src = synthesise_source("background", 5, 1.5, 300.0, 1.0, 10.0, 2.0, rng, source_id="bg",
                            pm_masyr=(50.0, 0.0))
    return screen_source(src, conf)


@pytest.fixture(scope="module")
def untestable_screen(conf):
    rng = np.random.default_rng(15)
    src = synthesise_source("fixed", 2, 30.0 / 365.25, 300.0, 1.0, 10.0, 2.0, rng, source_id="short",
                            contrast_by_band=_grey())
    return screen_source(src, conf)


# --------------------------------------------------------------------------------------
# Kepler solver and projection
# --------------------------------------------------------------------------------------

def test_kepler_equation_is_solved_to_machine_precision():
    m = np.linspace(0.0, 2 * np.pi, 97)
    for e in (0.0, 0.3, 0.9, 0.95):
        ecc = solve_kepler(m, e)
        assert np.allclose(ecc - e * np.sin(ecc), np.mod(m, 2 * np.pi), atol=1e-9)


def test_circular_faceon_orbit_has_the_third_law_period_and_separation():
    # a = 3 AU, M = 1 Msun, d = 10 pc: 300 mas, P = sqrt(27) = 5.196 yr,
    # 360/5.196 = 69.28 deg of arc in one year.
    assert kepler_period_yr(3.0, 1.0) == pytest.approx(5.196, abs=1e-3)
    t = np.array([0.0, 1.0])
    dx, dy = kepler_xy(t, (3.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1.0, 10.0)
    sep = np.hypot(dx, dy)
    assert np.allclose(sep, 300.0, atol=1e-9)
    pa = np.degrees(np.arctan2(dx, dy))
    swept = (pa[1] - pa[0]) % 360.0
    assert swept == pytest.approx(69.28, abs=0.05)
    arc_mas = np.radians(swept) * 300.0
    assert arc_mas == pytest.approx(np.radians(69.28) * 300.0, rel=1e-3)
    # The arc length is what the testability check predicts for span/P.
    assert arc_mas == pytest.approx(2 * np.pi * (1.0 / 5.196) * 300.0, rel=1e-3)


def test_kepler_xy_accepts_dict_params_and_scales_with_distance():
    t = np.linspace(0.0, 2.0, 7)
    p = {"a_au": 3.0, "e": 0.4, "inc_rad": 0.5, "Omega_rad": 1.0, "omega_rad": 0.3, "t_peri_yr": 0.2}
    dx1, dy1 = kepler_xy(t, p, 1.0, 10.0)
    dx2, dy2 = kepler_xy(t, (3.0, 0.4, 0.5, 1.0, 0.3, 0.2), 1.0, 20.0)
    assert np.allclose(dx1, 2 * dx2) and np.allclose(dy1, 2 * dy2)
    # Eccentric orbit: periapsis/apoapsis separation ratio (1-e)/(1+e) in 3-D.
    r = 3.0 * (1 - 0.4 * np.cos(solve_kepler(2 * np.pi * (t - 0.2) / kepler_period_yr(3.0, 1.0), 0.4)))
    assert r.min() >= 3.0 * 0.6 - 1e-9 and r.max() <= 3.0 * 1.4 + 1e-9


# --------------------------------------------------------------------------------------
# Input structure and testability
# --------------------------------------------------------------------------------------

def test_source_validation_and_properties():
    src = CGISource("s", "star", 1.0, 10.0, [61000.0, 61365.25], [300.0, 300.0], [0.0, 0.0], 2.0)
    assert src.n == 2 and src.span_yr == pytest.approx(1.0) and src.separation_mas == 300.0
    assert np.allclose(src.err_mas, 2.0)
    with pytest.raises(ValueError):
        CGISource("s", "star", 1.0, 10.0, [1.0, 2.0], [1.0], [1.0, 2.0], 2.0)
    with pytest.raises(ValueError):
        CGISource("s", "star", 1.0, 10.0, [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0, 3.0])
    d = src.as_dict()
    json.dumps(d)


def test_arc_testability_short_arc_is_untestable(conf):
    rng = np.random.default_rng(1)
    short = synthesise_source("fixed", 2, 30.0 / 365.25, 300.0, 1.0, 10.0, 2.0, rng)
    t = arc_testability(short, conf)
    assert not t["testable"]
    assert any(r.startswith("arc_too_short") for r in t["reasons"])
    assert any(r.startswith("too_few_epochs") for r in t["reasons"])
    assert t["period_faceon_circular_yr"] == pytest.approx(5.196, abs=1e-2)
    long = synthesise_source("fixed", 5, 1.5, 300.0, 1.0, 10.0, 2.0, rng)
    t2 = arc_testability(long, conf)
    assert t2["testable"] and t2["span_fraction_of_period"] > conf["statite"]["min_span_fraction_of_period"]
    assert t2["expected_motion_mas"] > 3 * t2["median_err_mas"]


def test_arc_testability_motion_below_error_is_untestable(conf):
    # A wide face-on orbit at 4 arcsec around a 0.5 Msun star at 20 pc: P ~ 1000 yr,
    # 0.2 yr of baseline -> sub-mas expected motion at 3 mas errors.
    rng = np.random.default_rng(2)
    src = synthesise_source("fixed", 4, 0.2, 4000.0, 0.5, 20.0, 3.0, rng)
    t = arc_testability(src, conf)
    assert not t["testable"] and any(r.startswith("motion_below_error") for r in t["reasons"])


# --------------------------------------------------------------------------------------
# The three fits
# --------------------------------------------------------------------------------------

def test_fit_fixed_recovers_the_position():
    rng = np.random.default_rng(3)
    src = synthesise_source("fixed", 6, 1.0, 250.0, 1.0, 10.0, 2.0, rng, pa_deg=30.0)
    f = fit_fixed(src)
    assert f["n_params"] == 2 and f["chi2"] < 30
    assert f["params"]["dx_mas"] == pytest.approx(250 * np.sin(np.radians(30)), abs=3)
    assert f["params"]["dy_mas"] == pytest.approx(250 * np.cos(np.radians(30)), abs=3)


def test_fit_linear_imposes_the_parallax_and_recovers_the_proper_motion():
    rng = np.random.default_rng(4)
    src = synthesise_source("background", 6, 1.5, 300.0, 1.0, 10.0, 2.0, rng, pm_masyr=(50.0, -20.0))
    f = fit_linear(src)
    assert f["n_params"] == 4 and f["params"]["parallax_imposed"] and not f["nests_fixed"]
    assert f["params"]["parallax_mas"] == pytest.approx(100.0)
    assert f["params"]["pm_x_masyr"] == pytest.approx(50.0, abs=5.0)
    assert f["params"]["pm_y_masyr"] == pytest.approx(-20.0, abs=5.0)
    assert f["chi2"] < 40


def test_fit_linear_without_geometry_or_span_says_so():
    rng = np.random.default_rng(5)
    src = synthesise_source("fixed", 6, 1.5, 300.0, 1.0, 10.0, 2.0, rng)
    src.meta.pop("ecl_lat_deg")
    src.meta.pop("ecl_lon_deg")
    f = fit_linear(src)
    assert f["nests_fixed"] and not f["params"]["parallax_imposed"]
    assert any(n.startswith("parallax_term_skipped") for n in f["notes"])
    short = synthesise_source("fixed", 4, 0.3, 300.0, 1.0, 10.0, 2.0, rng)
    f2 = fit_linear(short)
    assert f2["nests_fixed"] and any("0.3" in n for n in f2["notes"])


def test_fit_linear_free_parallax_phase_when_longitude_unknown():
    rng = np.random.default_rng(6)
    src = synthesise_source("background", 6, 1.5, 300.0, 1.0, 10.0, 2.0, rng, pm_masyr=(50.0, 0.0))
    src.meta.pop("ecl_lon_deg")
    f = fit_linear(src)
    assert f["n_params"] == 5 and f["params"]["parallax_imposed"]
    assert f["params"]["parallax_phase_deg"] is not None
    assert f["chi2"] < 40


def test_parallax_factors_precedence_and_absence():
    mjd = 61000.0 + np.linspace(0, 400, 5)
    src = CGISource("s", "star", 1.0, 10.0, mjd, np.zeros(5), np.zeros(5), 2.0,
                    meta={"parallax_factor_x": np.ones(5), "parallax_factor_y": np.zeros(5),
                          "ecl_lat_deg": 30.0})
    fx, fy, how = parallax_factors(src)
    assert how == "exact_factors_from_meta" and np.allclose(fx, 1.0)
    src.meta = {}
    assert parallax_factors(src) is None
    src.meta = {"ecl_lat_deg": 30.0}
    fx, fy, how = parallax_factors(src)
    assert how.endswith("free_phase") and fx.shape == (2, 5)


def test_fit_kepler_recovers_an_injected_orbit():
    rng = np.random.default_rng(7)
    src = synthesise_source("kepler", 8, 2.0, 300.0, 1.0, 10.0, 2.0, rng, inc_deg=30.0, e=0.3)
    k = fit_kepler(src)
    assert k["kepler_model"] == "full" and k["n_params"] == 6 and k["e_max"] == E_MAX
    assert k["chi2"] < 40
    assert k["params"]["a_au"] == pytest.approx(3.0, abs=0.15)
    assert k["params"]["e"] == pytest.approx(0.3, abs=0.1)
    assert k["params"]["inc_deg"] == pytest.approx(30.0, abs=8.0)
    assert k["params"]["period_yr"] == pytest.approx(kepler_period_yr(k["params"]["a_au"], 1.0))


def test_fit_kepler_falls_back_to_circular_faceon_for_three_epochs():
    rng = np.random.default_rng(8)
    src = synthesise_source("kepler", 3, 1.5, 300.0, 1.0, 10.0, 2.0, rng)
    k = fit_kepler(src)
    assert k["kepler_model"] == "circular_faceon" and k["n_params"] == 2
    assert k["params"]["a_au"] == pytest.approx(3.0, abs=0.2)
    assert k["chi2"] < 25


# --------------------------------------------------------------------------------------
# Astrometric verdicts
# --------------------------------------------------------------------------------------

def test_fixed_source_is_fixed_preferred_with_the_extreme_orbit_reported(fixed_interest_screen, conf):
    a = fixed_interest_screen["astrometry"]
    assert a["verdict"] == "FIXED_PREFERRED"
    assert a["background_excluded_by"] == "imposed_parallax"
    thr = conf["statite"]["fixed_vs_kepler_delta_chi2_min"]
    assert a["delta_chi2"]["kepler_minus_fixed"] >= thr
    assert a["delta_chi2"]["linear_minus_fixed"] >= conf["statite"]["fixed_vs_linear_delta_chi2_min"]
    assert a["fits"]["fixed"]["chi2"] < 30
    # The one degeneracy is reported, not hidden: an e ~ E_MAX_EXTREME edge-on orbit at
    # its projected turning point, with the baseline that would exclude it.
    ext = a["extreme_orbit"]
    assert ext is not None and ext["e_max_extreme"] == E_MAX_EXTREME
    assert "kepler_extreme" in a["fits"]
    if not ext["excluded"]:
        assert ext["span_needed_yr_estimate"] > 1.5
        assert ext["r_mid_au"] > 3.0
        assert any(n.startswith("extreme_orbit_mimic_not_excluded") for n in a["notes"])
        assert "extreme_orbit_mimic_not_excluded" in fixed_interest_screen["cautions"]


def test_keplerian_companion_is_keplerian(kepler_screen):
    a = kepler_screen["astrometry"]
    assert a["verdict"] == "KEPLERIAN" and a["best_model"] == "kepler"
    p = a["fits"]["kepler"]["params"]
    assert p["a_au"] == pytest.approx(3.0, abs=0.3)
    assert p["inc_deg"] == pytest.approx(30.0, abs=10.0)
    assert a["fits"]["kepler"]["chi2"] < 40
    assert a["extreme_orbit"] is None


def test_background_star_is_background(background_screen):
    a = background_screen["astrometry"]
    assert a["verdict"] == "BACKGROUND" and a["best_model"] == "linear"
    assert a["fits"]["linear"]["params"]["pm_x_masyr"] == pytest.approx(50.0, abs=5.0)
    assert a["fits"]["linear"]["chi2"] < 40


def test_two_epochs_a_month_apart_are_unresolved(untestable_screen):
    a = untestable_screen["astrometry"]
    assert a["verdict"] == "UNRESOLVED" and not a["testability"]["testable"]
    assert a["notes"] and a["notes"][0].startswith("arc_untestable")
    assert a["fits"] == {}


def test_without_parallax_the_background_is_not_excluded_unless_the_star_moves(conf):
    rng = np.random.default_rng(16)
    src = synthesise_source("fixed", 5, 1.5, 300.0, 1.0, 10.0, 2.0, rng)
    src.meta.pop("ecl_lat_deg")
    src.meta.pop("ecl_lon_deg")
    v = astrometric_verdict(src, conf)
    assert v["verdict"] == "UNRESOLVED"
    assert "background_not_excluded" in v["notes"]
    assert v["background_excluded_by"] is None
    # The star's own proper motion (200, -100) mas/yr: a background star would have to
    # drift at minus that; the fixed source does not, so the background is excluded.
    src.meta.update(pm_x_masyr=200.0, pm_y_masyr=-100.0)
    v2 = astrometric_verdict(src, conf)
    assert v2["verdict"] == "FIXED_PREFERRED"
    assert v2["background_excluded_by"] == "proper_motion_vs_star"
    assert v2["background_pm_check"]["excluded"]


# --------------------------------------------------------------------------------------
# Reflectance and phase function
# --------------------------------------------------------------------------------------

def test_reflectance_grey_vs_planet(conf):
    g = reflectance_test(_grey(), conf)
    assert g["verdict"] == "GREY_PREFERRED"
    assert g["delta_chi2"] >= conf["statite"]["grey_vs_planet_delta_chi2_min"]
    assert g["chi2_grey"] < 1e-6 and sorted(g["bands_used"]) == list(BANDS)
    p = reflectance_test(_jupiter(), conf)
    assert p["verdict"] == "PLANET_LIKE" and p["chi2_planet"] < 1e-6
    assert p["params"]["ratio_B3_B1"] == pytest.approx(0.45)
    one = reflectance_test({"B1": (1e-8, 2e-10)}, conf)
    assert one["verdict"] == "UNRESOLVED" and any(n.startswith("insufficient_bands") for n in one["notes"])
    assert one["chi2_grey"] is None and one["delta_chi2"] is None
    none = reflectance_test(None, conf)
    assert none["verdict"] == "UNRESOLVED" and any("insufficient_bands" in n for n in none["notes"])


def test_reflectance_noisy_and_unknown_bands(conf):
    rng = np.random.default_rng(17)
    noisy = {b: (1e-8 * (1 + rng.normal(0, 0.3)), 3e-9) for b in BANDS}
    r = reflectance_test(noisy, conf)
    assert r["verdict"] == "UNRESOLVED"
    odd = dict(_grey())
    odd["F146"] = (1e-8, 2e-10)
    r2 = reflectance_test(odd, conf)
    assert r2["verdict"] == "GREY_PREFERRED" and any("band_not_in_template" in n for n in r2["notes"])


def test_lambert_phase_function_limits():
    assert lambert_phase(0.0) == pytest.approx(1.0)
    assert lambert_phase(np.pi) == pytest.approx(0.0, abs=1e-12)
    assert lambert_phase(np.pi / 2) == pytest.approx(1 / np.pi)


def test_phase_function_specular_vs_lambertian(conf):
    rng = np.random.default_rng(18)
    al, c, e = _specular_phase(rng)
    s = phase_function_test(al, c, e, conf)
    assert s["verdict"] == "SPECULAR_PREFERRED"
    assert s["delta_chi2"] >= conf["statite"]["specular_vs_lambert_delta_chi2_min"]
    assert s["params"]["specular_alpha0_deg"] == pytest.approx(60.0, abs=2.0)
    assert 0.5 <= s["params"]["specular_width_deg"] <= 10.0
    al, c, e = _lambert_phase(rng)
    lam = phase_function_test(al, c, e, conf)
    assert lam["verdict"] == "LAMBERTIAN" and lam["chi2_lambert"] < 30
    three = phase_function_test(al[:3], c[:3], e[:3], conf)
    assert three["verdict"] == "UNRESOLVED"
    assert any(n.startswith("insufficient_phase_coverage") for n in three["notes"])
    narrow = phase_function_test(np.linspace(40, 60, 6), c[:6], e[:6], conf)
    assert narrow["verdict"] == "UNRESOLVED" and narrow["delta_chi2"] is None
    missing = phase_function_test(None, None, None, conf)
    assert missing["verdict"] == "UNRESOLVED"


# --------------------------------------------------------------------------------------
# Screening, tiers, assessment
# --------------------------------------------------------------------------------------

def test_screen_candidate_tier_needs_fixed_plus_a_physical_test(fixed_candidate_screen, conf):
    r = fixed_candidate_screen
    assert r["astrometry"]["verdict"] == "FIXED_PREFERRED"
    assert r["reflectance"]["verdict"] == "GREY_PREFERRED"
    assert r["phase_function"]["verdict"] == "SPECULAR_PREFERRED"
    assert set(r["supporting_tests"]) == {"grey_reflectance", "specular_phase"}
    assert r["tier"] == conf["statite"]["tiers"]["candidate"]
    assert r["rejections"] == []
    assert r["polarization"]["high_polarization"] is True
    assert any("does not discriminate" in n for n in r["polarization"]["notes"])
    assert r["simulated"] is True
    json.dumps(r)


def test_screen_interest_tier_for_fixed_alone(fixed_interest_screen, conf):
    r = fixed_interest_screen
    assert r["tier"] == conf["statite"]["tiers"]["interest"]
    assert r["reflectance"]["verdict"] == "NOT_RUN" and r["phase_function"]["verdict"] == "NOT_RUN"
    assert r["polarization"]["high_polarization"] is False
    assert r["supporting_tests"] == [] and r["rejections"] == []


def test_screen_rejects_keplerian_and_background(kepler_screen, background_screen):
    assert kepler_screen["tier"] == "NO_ANOMALY"
    assert kepler_screen["rejections"] == ["keplerian_orbit_fits"]
    assert "planet_like_reflectance" in kepler_screen["cautions"]
    assert background_screen["tier"] == "NO_ANOMALY"
    assert background_screen["rejections"] == ["background_star_fits"]


def test_screen_untestable_arc_is_unresolved_even_with_grey_colour(untestable_screen):
    r = untestable_screen
    assert r["reflectance"]["verdict"] == "GREY_PREFERRED"
    assert r["tier"] == "UNRESOLVED" and r["rejections"] == ["arc_untestable"]


def test_assess_empty_is_no_data_reached(conf):
    s = assess_sources([], conf)
    assert s["verdict"] == "NO_DATA_REACHED" and s["n_sources"] == 0
    assert s["simulated_inputs"] is None
    json.dumps(s)


def test_assess_counts_tiers_and_never_calls_a_simulation_a_candidate(
        fixed_candidate_screen, fixed_interest_screen, kepler_screen, background_screen,
        untestable_screen, conf):
    recs = [fixed_candidate_screen, fixed_interest_screen, kepler_screen, background_screen,
            untestable_screen]
    s = assess_sources(recs, conf)
    tiers = conf["statite"]["tiers"]
    assert s["counts_by_tier"][tiers["candidate"]] == 1
    assert s["counts_by_tier"][tiers["interest"]] == 1
    assert s["counts_by_tier"]["NO_ANOMALY"] == 2 and s["counts_by_tier"]["UNRESOLVED"] == 1
    assert s["funnel"]["counts"]["sources_screened"] == 5
    assert s["funnel"]["counts"]["pending_vet"] == 2
    assert s["funnel"]["rejections"]["keplerian_orbit_fits"] == 1
    assert s["simulated_inputs"] is True
    assert s["verdict"] == "NO_CANDIDATE"          # all synthetic: not a sky result
    assert any("simulated_inputs_only" in n for n in s["funnel"]["notes"])
    # The same records flagged as real data would be pending vet.
    real = [dict(r, simulated=False) for r in recs]
    s2 = assess_sources(real, conf)
    assert s2["verdict"] == "STATITE_CANDIDATES_PENDING_VET" and s2["simulated_inputs"] is False
    assert sorted(s2["pending_vet_ids"][tiers["candidate"]]) == ["fx_cand"]
    nulls = [dict(kepler_screen, simulated=False), dict(background_screen, simulated=False)]
    assert assess_sources(nulls, conf)["verdict"] == "NO_CANDIDATE"


def test_synthesise_rejects_unknown_kind():
    with pytest.raises(ValueError):
        synthesise_source("hovering", 5, 1.0, 300.0, 1.0, 10.0, 2.0)

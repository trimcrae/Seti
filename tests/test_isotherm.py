"""Offline tests for the ISOTHERM shape detector (no network).

Every spectrum is synthesised here.  Imports are restricted to the pure-function
modules (``sed_model``, ``shape_stats``) plus the offline parts of ``run``; no
test touches ``acquire``'s network paths.

The four injections the channel lives or dies by:

* a true ``beta = 0`` isothermal emitter IS flagged (S5);
* a realistic ``beta ~ 1`` dust disk with a radial gradient is NOT flagged;
* a natural two-belt system (warm belt + cold belt, both with radial extent) is
  NOT flagged — "two components" is common and natural, not anomalous;
* a 3-shell geometric cascade IS recovered (S6).

Plus honest degradation on missing/empty/low-SNR data, and a case for every
rejection rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from seti.isotherm.run import analyse_one, cascade_sensitivity
from seti.isotherm.sed_model import (
    bin_to_resolution,
    bolometric,
    chi2_one_component,
    fit_discrete,
    fit_gradient,
    gradient_sed,
    mbb,
    planck_nu,
    select_n_components,
)
from seti.isotherm.shape_stats import (
    DEFAULT_THRESHOLDS,
    analyse_spectrum,
    classify,
    compute_shape_stats,
    estimate_beta,
    extragalactic_interloper,
    feature_equivalent_width,
    geometric_progression_test,
    measure_features,
    natural_floor_dt_over_t,
    order_step,
    screen_spectrum,
    temperature_width,
    wavelength_leverage,
)

IRS_LAM = np.geomspace(5.2, 38.0, 180)


def _noisy(model, snr, seed):
    rng = np.random.default_rng(seed)
    m = np.asarray(model, float)
    m = m / m.max()
    e = m / float(snr)
    return m + rng.normal(0.0, e), e


def _silicate(lam, strength_10=0.45, strength_18=0.25):
    return (1.0
            + strength_10 * np.exp(-0.5 * ((lam - 9.8) / 1.4) ** 2)
            + strength_18 * np.exp(-0.5 * ((lam - 18.0) / 2.2) ** 2))


# ---------------------------------------------------------------------------
# Physics primitives
# ---------------------------------------------------------------------------

def test_planck_peak_matches_wien_displacement():
    """F_nu peak obeys lam_peak * T = 5099 micron K."""
    for t in (100.0, 300.0, 900.0):
        grid = np.geomspace(0.5, 400.0, 4000)
        peak = grid[int(np.argmax(planck_nu(grid, t)))]
        assert peak * t == pytest.approx(5099.44, rel=0.02)


def test_planck_underflows_not_overflows_on_wien_side():
    out = planck_nu(np.array([0.1, 1.0, 5.0]), 10.0)
    assert np.all(np.isfinite(out)) and np.all(out >= 0)


def test_bolometric_of_planck_scales_as_t4():
    grid = np.geomspace(0.3, 5000.0, 3000)
    a = bolometric(mbb(grid, 200.0, 0.0), grid)
    b = bolometric(mbb(grid, 400.0, 0.0), grid)
    assert b / a == pytest.approx(16.0, rel=0.02)


def test_gradient_degenerates_to_blackbody_when_isothermal():
    g = gradient_sed(IRS_LAM, 250.0, 250.0, 1.0, 0.0)
    b = mbb(IRS_LAM, 250.0, 0.0)
    assert np.allclose(g / g.max(), b / b.max(), rtol=1e-6)


def test_gradient_is_broader_than_a_single_blackbody():
    """A radial gradient is NECESSARILY a broad superposition — the core claim."""
    g = gradient_sed(IRS_LAM, 400.0, 100.0, 1.0, 0.0)
    b = mbb(IRS_LAM, 200.0, 0.0)

    def logwidth(y):
        y = y / y.max()
        above = IRS_LAM[y > 0.5]
        return np.log10(above.max() / above.min())

    assert logwidth(g) > logwidth(b)


def test_bin_to_resolution_reduces_count_and_shrinks_errors():
    flux = mbb(IRS_LAM, 250.0, 0.0)
    err = flux / 50.0
    lam_b, f_b, e_b, n_b = bin_to_resolution(IRS_LAM, flux, err, resolution=40.0)
    assert 0 < lam_b.size < IRS_LAM.size
    assert np.all(n_b >= 1)
    assert np.median(e_b / f_b) < np.median(err / flux)


def test_chi2_one_component_matches_full_fit():
    f, e = _noisy(mbb(IRS_LAM, 300.0, 0.0), 100, 1)
    chi2_fast, t_fast = chi2_one_component(IRS_LAM, f, e, 0.0)
    full = fit_discrete(IRS_LAM, f, e, 1, beta=0.0)
    assert t_fast == pytest.approx(full.temps_k[0], rel=0.05)
    assert chi2_fast == pytest.approx(full.chi2, rel=0.10)


# ---------------------------------------------------------------------------
# Statistic 1: beta
# ---------------------------------------------------------------------------

def test_beta_recovered_for_true_planck_emitter():
    f, e = _noisy(mbb(IRS_LAM, 250.0, 0.0), 120, 2)
    b = estimate_beta(IRS_LAM, f, e)
    assert abs(b["beta"]) < 0.2
    assert b["planck_consistent"]
    assert b["leverage_beta_constrained"]
    assert b["t_k"] == pytest.approx(250.0, rel=0.05)


def test_beta_recovered_for_dusty_emitter():
    """beta ~ 1 grains must be measured as dust-like, not as a Planck function."""
    f, e = _noisy(mbb(IRS_LAM, 250.0, 1.0), 120, 3)
    b = estimate_beta(IRS_LAM, f, e)
    assert b["beta"] == pytest.approx(1.0, abs=0.3)
    assert not b["planck_consistent"]


def test_beta_of_a_gradient_disk_is_read_off_the_gradient_model():
    """A single-component beta is BIASED to 0 for a broad source.

    This is the trap the channel had to be built around: fitting one modified
    blackbody to a beta=1 radial gradient returns beta = 0.00, a false "true
    Planck function".  The gradient model recovers the truth, and classify()
    must use it.
    """
    f, e = _noisy(gradient_sed(IRS_LAM, 400.0, 90.0, 1.0, 1.0), 80, 4)
    single = estimate_beta(IRS_LAM, f, e)
    grad = fit_gradient(IRS_LAM, f, e)
    assert abs(single["beta"]) < 0.3            # the bias is real
    assert grad.beta == pytest.approx(1.0, abs=0.35)   # the truth is recoverable


def test_wavelength_leverage_gates_beta_honestly():
    """Outside the in-band-peak window, beta and T are degenerate — say so."""
    assert wavelength_leverage(IRS_LAM, 250.0)["beta_constrained"]
    assert not wavelength_leverage(IRS_LAM, 4000.0)["beta_constrained"]
    assert not wavelength_leverage(IRS_LAM, 25.0)["beta_constrained"]


# ---------------------------------------------------------------------------
# Statistic 2: temperature-distribution width
# ---------------------------------------------------------------------------

def test_natural_floor_follows_from_the_radial_temperature_law():
    """T ~ r^(-1/2) gives dT/T = 0.5 dr/r — the floor is derived, not asserted."""
    assert natural_floor_dt_over_t(0.20) == pytest.approx(0.10)
    assert natural_floor_dt_over_t(0.05) == pytest.approx(0.025)


@pytest.mark.parametrize("seed", [5, 23, 42])
def test_width_is_bounded_small_for_an_isothermal_emitter(seed):
    """Must hold for EVERY noise realisation, not a lucky seed.

    The limit scales as ~SNR^(-1/2): 0.196 at SNR 60, 0.130 at 120, 0.080 at
    300, 0.043 at 1000.  SNR 300 is the measured requirement for this test to
    reach the 0.10 natural floor (docs/isotherm.md sec. 7).
    """
    f, e = _noisy(mbb(IRS_LAM, 250.0, 0.0), 300, seed)
    w = temperature_width(IRS_LAM, f, e)
    assert w["success"]
    assert w["dt_over_t_upper95"] < w["floor_nominal"]


def test_width_limit_tightens_with_snr():
    """Below SNR ~300 the S5 test cannot reach the floor — and must say so."""
    f_lo, e_lo = _noisy(mbb(IRS_LAM, 250.0, 0.0), 60, 5)
    f_hi, e_hi = _noisy(mbb(IRS_LAM, 250.0, 0.0), 1000, 5)
    lo = temperature_width(IRS_LAM, f_lo, e_lo)
    hi = temperature_width(IRS_LAM, f_hi, e_hi)
    assert hi["dt_over_t_upper95"] < lo["dt_over_t_upper95"]
    assert not lo["narrower_than_nominal_floor"]


def test_width_is_large_for_a_radial_gradient():
    f, e = _noisy(gradient_sed(IRS_LAM, 400.0, 90.0, 1.0, 1.0), 80, 6)
    w = temperature_width(IRS_LAM, f, e)
    assert w["dt_over_t"] > w["floor_nominal"]
    assert not w["narrower_than_nominal_floor"]


# ---------------------------------------------------------------------------
# Statistic 3: multiplicity and geometric progression
# ---------------------------------------------------------------------------

def test_geometric_progression_detected_and_rejected():
    assert geometric_progression_test([600.0, 300.0, 150.0])["is_geometric"]
    assert geometric_progression_test([600.0, 300.0, 150.0])["mean_ratio"] == \
        pytest.approx(2.0, rel=0.01)
    assert not geometric_progression_test([600.0, 500.0, 120.0])["is_geometric"]
    assert not geometric_progression_test([600.0, 300.0])["is_geometric"]  # n<3


def test_component_count_must_earn_its_bic():
    """A one-component source must not be decomposed into many."""
    f, e = _noisy(mbb(IRS_LAM, 250.0, 0.0), 120, 7)
    best, ladder = select_n_components(IRS_LAM, f, e, n_max=4)
    assert best.n_components == 1
    assert len(ladder) == 4


# ---------------------------------------------------------------------------
# Statistic 4: spectral features
# ---------------------------------------------------------------------------

def test_no_false_feature_on_a_featureless_planck():
    """The continuum must not manufacture a silicate band out of Planck curvature.

    A LINEAR log-log continuum yields a false EW of +1.67 micron here, as large
    as a real silicate feature; the cubic continuum is what makes the veto safe.
    """
    for t in (150.0, 250.0, 400.0):
        f = mbb(IRS_LAM, t, 0.0)
        f = f / f.max()
        e = f / 100.0
        feats = measure_features(IRS_LAM, f, e)
        for name in ("silicate_10", "silicate_18"):
            m = feats[name]
            assert m["covered"]
            assert abs(m["ew_um"]) < DEFAULT_THRESHOLDS["feature_veto_abs_ew_um"], \
                f"false {name} EW={m['ew_um']} at T={t}"


def test_real_silicate_feature_is_detected():
    base = mbb(IRS_LAM, 250.0, 0.0)
    base = base / base.max()
    f, e = _noisy(base * _silicate(IRS_LAM), 80, 8)
    feats = measure_features(IRS_LAM, f, e)
    assert feats["silicate_10"]["ew_um"] > 0.3
    assert abs(feats["silicate_10"]["significance"]) > 4.0
    assert feats["silicate_18"]["ew_um"] > 0.1


def test_feature_not_measured_when_anchors_run_off_the_spectrum():
    """No extrapolation, ever: a truncated anchor window must return not-covered."""
    short = np.geomspace(5.2, 13.0, 60)
    f = mbb(short, 250.0, 0.0)
    m = feature_equivalent_width(short, f, f / 100.0, (15.5, 21.5),
                                 (14.0, 15.0), (24.0, 26.5))
    assert not m["covered"]
    assert np.isnan(m["ew_um"])


def test_equivalent_width_has_a_continuum_systematic_floor():
    f = mbb(IRS_LAM, 250.0, 0.0)
    m = feature_equivalent_width(IRS_LAM, f, f / 10000.0, (8.2, 12.3),
                                 (5.6, 7.4), (13.0, 14.2))
    assert m["ew_err_sys_um"] > 0
    assert m["ew_err_um"] >= m["ew_err_sys_um"]


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------

def test_extragalactic_test_does_not_fire_on_a_pure_blackbody():
    """Background-galaxy confusion killed every Hephaistos candidate — but the
    test must not flag a featureless continuum at every redshift offered."""
    f, e = _noisy(mbb(IRS_LAM, 250.0, 0.0), 120, 9)
    assert not extragalactic_interloper(IRS_LAM, f, e)["extragalactic_flag"]


def test_extragalactic_test_finds_a_redshifted_pah_silicate_system():
    z = 0.4
    base = mbb(IRS_LAM, 300.0, 0.0)
    base = base / base.max()
    lam_rest = IRS_LAM / (1 + z)
    galaxy = base * (1
                     + 0.55 * np.exp(-0.5 * ((lam_rest - 7.7) / 0.5) ** 2)
                     + 0.45 * np.exp(-0.5 * ((lam_rest - 11.3) / 0.3) ** 2)
                     + 0.50 * np.exp(-0.5 * ((lam_rest - 9.8) / 1.3) ** 2))
    f, e = _noisy(galaxy, 100, 10)
    out = extragalactic_interloper(IRS_LAM, f, e)
    assert out["extragalactic_flag"]
    assert out["z"] == pytest.approx(z, abs=0.12)


def test_order_step_detects_a_module_stitching_jump():
    f = mbb(IRS_LAM, 250.0, 0.0)
    assert abs(order_step(IRS_LAM, f)["step_frac"]) < 0.05
    jumped = f * np.where(IRS_LAM >= 14.2, 1.6, 1.0)
    assert order_step(IRS_LAM, jumped)["step_frac"] > 0.3


def test_companion_photosphere_is_vetoed():
    """A fitted component hotter than grains survive is a companion, not dust."""
    stats = {"usable": True, "features": {}, "extragalactic": {},
             "order_step": {"covered": False},
             "fit_best": {"n_components": 1, "temps_k": [2400.0],
                          "reduced_chi2": 1.0},
             "beta": {}, "width": {}, "geometric": {}}
    v = classify(stats)
    assert v.verdict == "REJECTED_NATURAL"
    assert "component_hotter_than_grain_survival" in v.vetoes


# ---------------------------------------------------------------------------
# THE FOUR REQUIRED INJECTIONS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [11, 23, 42])
def test_injected_isothermal_beta0_emitter_is_flagged_s5(seed):
    f, e = _noisy(mbb(IRS_LAM, 250.0, 0.0), 300, seed)
    v = analyse_spectrum(IRS_LAM, f, e)
    assert v.verdict == "S5_ISOTHERMAL_REVIEW", (v.verdict, v.flags, v.vetoes)
    assert "beta_planck_consistent" in v.flags
    assert v.vetoes == []


def test_injected_realistic_dust_disk_is_not_flagged():
    """beta ~ 1 grains on a radial gradient: the dominant natural population."""
    f, e = _noisy(gradient_sed(IRS_LAM, 400.0, 90.0, 1.0, 1.0), 80, 12)
    v = analyse_spectrum(IRS_LAM, f, e)
    assert not v.verdict.startswith(("S5", "S6")), (v.verdict, v.flags)
    assert "gradient_model_preferred" in v.flags or "beta_dustlike" in v.flags


def test_injected_natural_two_belt_system_is_not_flagged():
    """Warm belt + cold belt, both with radial extent.

    Two-temperature debris disks are common and entirely natural (asteroid belt
    + Kuiper belt), so "two components" must NOT be anomalous on its own.
    """
    warm = gradient_sed(IRS_LAM, 350.0, 190.0, 1.0, 1.0)
    cold = gradient_sed(IRS_LAM, 95.0, 55.0, 1.0, 1.0)
    model = warm / warm.max() + 0.9 * cold / cold.max()
    f, e = _noisy(model, 80, 13)
    v = analyse_spectrum(IRS_LAM, f, e)
    assert not v.verdict.startswith(("S5", "S6")), (v.verdict, v.flags)


def test_injected_three_shell_geometric_cascade_is_recovered_s6():
    """600 / 300 / 150 K: ratio 2, all three Wien peaks inside 5-38 micron."""
    model = np.zeros_like(IRS_LAM)
    for k in range(3):
        c = mbb(IRS_LAM, 600.0 / 2.0**k, 0.0)
        model += c / c.max()
    f, e = _noisy(model, 500, 14)
    v = analyse_spectrum(IRS_LAM, f, e)
    assert v.verdict == "S6_MATRIOSHKA_CASCADE_REVIEW", (v.verdict, v.flags,
                                                         v.vetoes)
    temps = sorted(v.stats["fit_best"]["temps_k"], reverse=True)
    assert len(temps) == 3
    for got, want in zip(temps, [600.0, 300.0, 150.0], strict=True):
        assert got == pytest.approx(want, rel=0.12)
    assert v.stats["geometric"]["mean_ratio"] == pytest.approx(2.0, rel=0.10)


def test_silicate_bearing_disk_is_rejected_even_though_it_is_multicomponent():
    base = gradient_sed(IRS_LAM, 300.0, 120.0, 1.0, 1.0)
    base = base / base.max()
    f, e = _noisy(base * _silicate(IRS_LAM), 80, 15)
    v = analyse_spectrum(IRS_LAM, f, e)
    assert v.verdict == "REJECTED_NATURAL"
    assert any(x.startswith("feature_silicate") for x in v.vetoes)


# ---------------------------------------------------------------------------
# Stage-1 screen
# ---------------------------------------------------------------------------

def test_screen_passes_an_isothermal_emitter_and_a_cascade():
    """The screen may only REJECT. It must never drop a true signal."""
    f, e = _noisy(mbb(IRS_LAM, 250.0, 0.0), 120, 16)
    assert screen_spectrum(IRS_LAM, f, e)["pass"]

    model = np.zeros_like(IRS_LAM)
    for k in range(3):
        c = mbb(IRS_LAM, 600.0 / 2.0**k, 0.0)
        model += c / c.max()
    f, e = _noisy(model, 500, 17)
    assert screen_spectrum(IRS_LAM, f, e)["pass"]


def test_screen_rejects_dust_features_and_dusty_beta():
    base = mbb(IRS_LAM, 250.0, 0.0)
    base = base / base.max()
    f, e = _noisy(base * _silicate(IRS_LAM), 80, 18)
    s = screen_spectrum(IRS_LAM, f, e)
    assert not s["pass"] and s["reason"].startswith("dust_features")

    f, e = _noisy(mbb(IRS_LAM, 250.0, 1.5), 120, 19)
    s = screen_spectrum(IRS_LAM, f, e)
    assert not s["pass"] and s["reason"] == "beta_dustlike"


# ---------------------------------------------------------------------------
# Honest degradation
# ---------------------------------------------------------------------------

def test_empty_spectrum_degrades_honestly():
    res = analyse_one(np.array([]), np.array([]), np.array([]))
    assert res["verdict"] == "INSUFFICIENT_DATA"
    assert "empty_spectrum" in res["vetoes"]


def test_too_few_points_degrades_honestly():
    lam = np.geomspace(5.2, 38.0, 6)
    f = mbb(lam, 250.0, 0.0)
    stats = compute_shape_stats(lam, f, f / 50.0)
    assert not stats["usable"]
    assert stats["reason"] == "insufficient_points"
    assert classify(stats).verdict == "INSUFFICIENT_DATA"


def test_low_snr_degrades_honestly_rather_than_emitting_a_candidate():
    f, e = _noisy(mbb(IRS_LAM, 250.0, 0.0), 1.5, 20)
    v = analyse_spectrum(IRS_LAM, f, e)
    assert v.verdict == "INSUFFICIENT_DATA"
    assert "low_snr" in v.vetoes


def test_all_nan_errors_do_not_raise():
    f = mbb(IRS_LAM, 250.0, 0.0)
    stats = compute_shape_stats(IRS_LAM, f, np.full_like(f, np.nan))
    assert not stats["usable"]


def test_analyse_one_reports_screened_out_rather_than_a_verdict():
    base = mbb(IRS_LAM, 250.0, 0.0)
    base = base / base.max()
    f, e = _noisy(base * _silicate(IRS_LAM), 80, 21)
    res = analyse_one(IRS_LAM, f, e)
    assert res["verdict"] == "screened_out"
    assert res["screen"]["reason"].startswith("dust_features")


# ---------------------------------------------------------------------------
# Sensitivity calibration
# ---------------------------------------------------------------------------

def test_cascade_sensitivity_map_is_monotonic_in_temperature_ratio():
    """Widely spaced shells must be easier to separate than closely spaced ones.

    This is the completeness map that makes a non-detection interpretable: it
    states where a detection would have been possible at all.
    """
    df = cascade_sensitivity(snrs=(150,), ratios=(1.5, 3.0), n_shells=3)
    easy = df[df["temperature_ratio"] == 3.0].iloc[0]
    hard = df[df["temperature_ratio"] == 1.5].iloc[0]
    assert easy["delta_bic_discrete_minus_gradient"] < \
        hard["delta_bic_discrete_minus_gradient"]
    assert bool(easy["recovered"])


def test_isotherm_thresholds_load_from_config(cfg):
    from seti.isotherm.run import channel_thresholds

    th = channel_thresholds(cfg)
    assert th["beta_planck_max"] == pytest.approx(0.35)
    assert th["delta_bic_component"] >= 10.0
    assert th["feature_veto_abs_ew_um"] > 0

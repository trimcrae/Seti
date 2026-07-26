"""Offline tests for EMBER (mid-infrared waste heat that switched off).

The contract this suite enforces, per ``docs/channel-brief.md`` section 5:

* an excess that switches off is **recovered**;
* a constant-excess source is **not** flagged;
* a variable AGB-like confounder is **not** flagged;
* the bandpass transformation is unit-tested against physics that is known
  independently of the implementation;
* an empty archive response degrades to an explicit verdict rather than a
  candidate;
* every rejection rule has a case that trips it.

No network is used anywhere.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.ember import bands as B
from seti.ember import crossepoch as X
from seti.ember import vet as V
from seti.ember.run import ember_run, stage_audit

RNG = np.random.default_rng(20260726)


# ==========================================================================
# 1. Bandpass / transfer physics
# ==========================================================================
def test_vega_spectrum_reproduces_published_zero_points():
    """The Vega stand-in must return each band's published zero point.

    Ks is exact by construction (it sets the normalisation). Every other band
    is a genuine prediction, and the residual is the combined error of the
    blackbody Vega approximation and the response model. W3 -- the primary
    band of the channel -- must agree to better than 3%.
    """
    q_ks = B.quoted_flux_ratio(B.BANDS["Ks"], B.vega_fnu_jy)
    assert q_ks == pytest.approx(B.BANDS["Ks"].zp_jy, rel=1e-6)

    q_w3 = B.quoted_flux_ratio(B.BANDS["W3"], B.vega_fnu_jy)
    assert q_w3 == pytest.approx(B.BANDS["W3"].zp_jy, rel=0.03)

    for key in ("W1", "W2", "W4", "H", "J"):
        q = B.quoted_flux_ratio(B.BANDS[key], B.vega_fnu_jy)
        assert q == pytest.approx(B.BANDS[key].zp_jy, rel=0.10), key


def _spread(early: str, late: str) -> float:
    v = [B.transfer(B.BANDS[early], B.BANDS[late], t)
         for t in (150.0, 300.0, 1500.0)]
    return max(v) / min(v)


def test_transfer_is_not_a_null_transformation_for_9_to_12_micron():
    """AKARI 9 um -> WISE W3 is strongly temperature-dependent; IRAS 25 -> W4 is not.

    This is the single most important quantitative claim of the channel's
    design. Treating the 9-to-12 micron step as a null transformation would
    manufacture apparent fades of order unity purely from the dust temperature.

    The thresholds here are those of the **real SVO response curves** committed
    in ``src/seti/data_assets/rsr/``.  With the documented trapezoid fallback
    the numbers are different and, for I12->W3, misleadingly benign (1.20
    against a true 1.71) -- which is why the audit records ``rsr_source`` per
    band and why this test asserts the ordering rather than one magic number.
    """
    spread_s9 = _spread("S9W", "W3")
    spread_i12 = _spread("I12", "W3")
    spread_i25 = _spread("I25", "W4")

    assert spread_s9 > 3.0, f"9->12 um should swing a lot, got {spread_s9:.2f}"
    # I25 -> W4 is the genuinely near-null pair: 25 um and 22 um are close in
    # wavelength AND both sit far enough down the Rayleigh-Jeans side that the
    # ratio barely moves over 150-1500 K.
    assert spread_i25 < 1.3, f"IRAS25->W4 should be near-null, got {spread_i25:.2f}"
    # I12 -> W3 is well conditioned but NOT null; the ordering is what the
    # channel's design rests on.
    assert spread_i25 < spread_i12 < 2.5
    assert spread_i12 < spread_s9


def test_transfer_tends_to_rayleigh_jeans_ratio_for_hot_sources():
    """At high temperature the excess is Rayleigh-Jeans and the transfer saturates."""
    hot = [B.transfer(B.BANDS["I12"], B.BANDS["W3"], t) for t in (3000.0, 20000.0)]
    assert hot[0] == pytest.approx(hot[1], rel=0.05)


def test_transfer_round_trip_is_consistent():
    """Transferring a flux there and back must return it."""
    for t in (200.0, 600.0):
        fwd = B.transfer(B.BANDS["S9W"], B.BANDS["W3"], t)
        back = B.transfer(B.BANDS["W3"], B.BANDS["S9W"], t)
        assert fwd * back == pytest.approx(1.0, rel=1e-9)


def test_planck_has_no_overflow_in_the_wien_tail():
    v = B.planck_fnu(np.array([0.1, 1.0, 10.0, 200.0]), 50.0)
    assert np.all(np.isfinite(v)) and np.all(v > 0)


def test_pair_audit_flags_the_iras_beam_and_the_saturation_window():
    """The audit must independently rediscover the two structural systematics."""
    a = B.audit_pair("I12", "W3")
    # IRAS's beam covers orders of magnitude more sky than W3's.
    assert a.beam_area_ratio > 100
    assert "beam-summed" in a.verdict
    # WISE W3 saturates around 1 Jy, close to the IRAS completeness limit, so
    # the usable window is genuinely narrow.
    assert a.usable_flux_lo_jy == pytest.approx(0.4, rel=1e-6)
    assert 0.5 < a.usable_flux_hi_jy < 5.0

    # I25 -> W4 has far more headroom before saturation.
    a25 = B.audit_pair("I25", "W4")
    assert a25.usable_flux_hi_jy > a.usable_flux_hi_jy


def test_neowise_is_absent_from_the_epoch_ladder():
    """NEOWISE flies W1/W2 only and cannot measure 100-300 K dust at any epoch."""
    surveys = [s for s, _y, _b in B.EPOCH_LADDER]
    assert surveys == ["IRAS", "AKARI", "WISE"]
    assert "NEOWISE" not in surveys
    for _s, _y, bs in B.EPOCH_LADDER:
        assert all(BANDS_LAM(b) > 8.0 for b in bs)


def BANDS_LAM(key: str) -> float:
    return B.BANDS[key].lam_ref_um


# ==========================================================================
# 2. Photospheric locus
# ==========================================================================
def _synthetic_locus_sample(n=4000, excess_frac=0.05, seed=1):
    """Stars on a colour-dependent photospheric locus, a few with real excess."""
    rng = np.random.default_rng(seed)
    colour = rng.uniform(0.4, 2.5, n)
    true_ks_minus_band = 0.02 + 0.10 * colour  # a plausible reddening-like locus
    ks_mag = rng.uniform(4.0, 9.0, n)
    band_mag = ks_mag - true_ks_minus_band + rng.normal(0, 0.03, n)
    has_excess = rng.random(n) < excess_frac
    band_mag[has_excess] -= rng.uniform(0.5, 3.0, has_excess.sum())
    return colour, ks_mag, band_mag, has_excess


def test_locus_recovers_the_photospheric_ridge_despite_an_excess_tail():
    colour, ks, band, _ = _synthetic_locus_sample()
    locus = X.fit_photosphere_locus(colour, ks, band, band="W3")
    assert not locus.degraded
    pred, sca = locus.predict(np.array([0.5, 1.5, 2.4]))
    truth = 0.02 + 0.10 * np.array([0.5, 1.5, 2.4])
    assert np.allclose(pred, truth, atol=0.03), f"{pred} vs {truth}"
    # The one-sided excess population must not inflate the fitted scatter.
    assert np.median(sca) < 0.08


def test_locus_survives_heavy_one_sided_contamination():
    """Half the sample carrying a large excess must not drag the ridge upward.

    An infrared-selected catalogue is enriched in exactly the sources this
    channel hunts, so the calibration population can be majority-contaminated.
    A median-based fit breaks down at 50%; the low-quantile reconstruction must
    not.
    """
    for frac in (0.3, 0.5, 0.7):
        colour, ks, band, _ = _synthetic_locus_sample(n=6000, excess_frac=frac,
                                                      seed=9)
        locus = X.fit_photosphere_locus(colour, ks, band, band="W3")
        pred, _ = locus.predict(np.array([0.6, 1.5, 2.3]))
        truth = 0.02 + 0.10 * np.array([0.6, 1.5, 2.3])
        assert np.allclose(pred, truth, atol=0.05), f"frac={frac}: {pred} vs {truth}"


def test_photosphere_prediction_uses_the_right_sign_convention():
    """F_band / F_Ks must equal 10**(+0.4 * (Ks - band)), not its inverse.

    Regression test. With the sign inverted the predicted photosphere is wrong
    by 10**(0.8*C), and because the colour term C differs between bands the
    error differs per epoch -- which fabricates fades across the whole
    catalogue rather than cancelling in the difference.
    """
    ks_jy = 0.05
    ratio_true = 0.06          # the mid-IR band is fainter than Ks in Jy
    colour_term = 2.5 * np.log10(ratio_true)   # = Ks - band, strongly negative
    locus = X.PhotosphereLocus(
        band="W3", colour_name="bp_rp", knots=np.array([0.0, 3.0]),
        values=np.array([colour_term, colour_term]),
        scatter=np.array([0.01, 0.01]), n_per_bin=np.array([500, 500]),
        n_calib=1000)
    # A bare photosphere: observed flux exactly equals the prediction.
    m = X.measure_excess("W3", ks_jy * ratio_true, 1e-4, ks_jy, 1e-5, 1.0, locus)
    assert m.f_phot_jy == pytest.approx(ks_jy * ratio_true, rel=1e-6)
    assert abs(m.f_exc_jy) < 1e-9
    assert abs(m.chi) < 0.1


def test_locus_degrades_honestly_on_an_empty_sample():
    locus = X.fit_photosphere_locus(np.array([]), np.array([]), np.array([]),
                                    band="W3")
    assert locus.degraded
    pred, sca = locus.predict(np.array([1.0]))
    assert np.isnan(pred[0]) and sca[0] > 0


# ==========================================================================
# 3. Eddington / Malmquist bias
# ==========================================================================
def test_eddington_deboost_only_ever_reduces_a_flux():
    f = np.array([0.5, 1.0, 5.0])
    s = np.array([0.1, 0.1, 0.1])
    out = X.eddington_deboost(f, s, count_slope=2.5)
    assert np.all(out <= f)
    # The correction shrinks as S/N rises.
    assert (f - out)[0] > (f - out)[2]


def test_count_slope_recovers_a_known_power_law():
    rng = np.random.default_rng(3)
    # dN/dS ~ S^-2.5 => sample from a Pareto with alpha = gamma - 1 = 1.5
    s = 0.4 * (1.0 + rng.pareto(1.5, 200_000))
    gamma = X.measure_count_slope(s, 0.5, 10.0)
    assert 2.0 < gamma < 3.0, gamma


def test_eddington_bias_alone_can_manufacture_a_fade():
    """Selection bias fabricates a fade with no astrophysical change whatsoever.

    Take sources of *identical* true flux and catalogue them only when a noisy
    measurement clears a threshold. The catalogued fluxes are then biased high,
    a deeper later survey measures the truth, and the pair reads as a fade --
    with nothing having changed. The bias is strictly one-directional, and it
    collapses once the true flux sits comfortably above the threshold, which is
    exactly what the funnel's S/N floor buys.
    """
    rng = np.random.default_rng(11)
    n = 200_000
    threshold, sigma = 0.4, 0.08

    def catalogued_bias(true_flux):
        meas = true_flux + rng.normal(0, sigma, n)
        det = meas > threshold
        return meas[det].mean() - true_flux, det.mean()

    bias_at_limit, frac_at_limit = catalogued_bias(threshold)
    bias_high_snr, frac_high = catalogued_bias(threshold + 8 * sigma)

    # At the limit only upward excursions are catalogued: a large positive bias.
    assert bias_at_limit > 0.5 * sigma, bias_at_limit
    assert 0.4 < frac_at_limit < 0.6
    # Well above the limit essentially everything is catalogued and the bias dies.
    assert frac_high > 0.99
    assert abs(bias_high_snr) < 0.05 * sigma
    assert bias_at_limit > 10 * abs(bias_high_snr)


def test_eddington_deboost_has_the_right_sign_and_scaling():
    """The correction must go as sigma^2 / F and never increase a flux."""
    f = np.array([1.0, 1.0, 2.0])
    s = np.array([0.05, 0.10, 0.10])
    out = X.eddington_deboost(f, s, count_slope=2.5)
    shift = f - out
    assert np.all(shift >= 0)
    # Quadrupling sigma^2 quadruples the shift at fixed flux.
    assert shift[1] == pytest.approx(4 * shift[0], rel=1e-6)
    # Doubling the flux halves the shift at fixed sigma.
    assert shift[2] == pytest.approx(shift[1] / 2, rel=1e-6)


def test_eddington_deboost_is_capped_for_hopeless_sources():
    """Below the expansion's validity the correction is clipped, not extrapolated."""
    out = X.eddington_deboost(np.array([0.01]), np.array([1.0]), count_slope=2.5)
    assert out[0] == pytest.approx(0.005)


# ==========================================================================
# 4. Excess and cessation — injection / recovery
# ==========================================================================
def _flat_locus(band: str, value: float = 0.0) -> X.PhotosphereLocus:
    return X.PhotosphereLocus(band=band, colour_name="bp_rp",
                              knots=np.array([0.0, 3.0]),
                              values=np.array([value, value]),
                              scatter=np.array([0.02, 0.02]),
                              n_per_bin=np.array([500, 500]), n_calib=1000)


def _source(f_exc_early_jy: float, f_exc_late_jy: float, ks_jy: float = 0.05,
            t_dust: float = 400.0, early: str = "I12", late: str = "W3",
            noise_frac: float = 0.02):
    """Build an early/late ExcessMeasurement pair with a specified excess.

    Fluxes are kept well below the WISE saturation onset (W3 saturates near
    0.96 Jy); otherwise the saturation guard fires first and the statistic is
    never exercised, which is itself the behaviour tested in
    ``test_saturated_late_band_refuses_to_claim_a_fade``.
    """
    phot_e = ks_jy * 10 ** (-0.4 * 0.0)
    phot_l = ks_jy * 10 ** (-0.4 * 0.0)
    f_e = phot_e + f_exc_early_jy
    f_l = phot_l + f_exc_late_jy
    m_e = X.measure_excess(early, f_e, noise_frac * f_e, ks_jy, 0.005 * ks_jy,
                           1.0, _flat_locus(early), count_slope=None,
                           phot_sys_frac=0.01)
    m_l = X.measure_excess(late, f_l, noise_frac * f_l, ks_jy, 0.005 * ks_jy,
                           1.0, _flat_locus(late), count_slope=None,
                           phot_sys_frac=0.01)
    return m_e, m_l


def test_injected_switch_off_is_recovered():
    """An excess present early and gone late must be flagged as a cessation."""
    t = 400.0
    exc_early = 0.5  # 10x the photosphere: a large, IRAS-scale excess
    m_e, m_l = _source(exc_early, 0.0, t_dust=t)
    res = X.cessation(m_e, m_l, t_dust_k=t, t_dust_lo=t, t_dust_hi=t,
                      t_dust_source="fitted")
    assert res.verdict == "cessation", res
    assert res.z > 5
    assert res.f_cess > 0.9


def test_constant_excess_is_not_flagged():
    """An unchanged excess, correctly transported, must come back 'constant'."""
    t = 400.0
    exc_early = 0.5
    exc_late = exc_early * B.transfer(B.BANDS["I12"], B.BANDS["W3"], t)
    m_e, m_l = _source(exc_early, exc_late, t_dust=t)
    res = X.cessation(m_e, m_l, t_dust_k=t, t_dust_lo=t, t_dust_hi=t,
                      t_dust_source="fitted")
    assert res.verdict == "constant", res
    assert abs(res.z) < 3
    assert abs(res.f_cess) < 0.15


def test_constant_excess_stays_unflagged_across_the_dust_temperature_prior():
    """With T_dust unknown, a constant excess must not be flagged at ANY temperature.

    This is the test that a naive implementation fails: transporting a 9-micron
    excess to 12 micron with the wrong temperature produces an apparent factor
    of a few, which the statistic would read as a cessation. Marginalising over
    the prior must widen the error enough to stay honest.
    """
    for t_true in (200.0, 400.0, 1000.0):
        exc_early = 0.5
        exc_late = exc_early * B.transfer(B.BANDS["S9W"], B.BANDS["W3"], t_true)
        m_e, m_l = _source(exc_early, exc_late, t_dust=t_true,
                           early="S9W", late="W3")
        res = X.cessation(m_e, m_l, t_dust_k=X.T_DUST_DEFAULT_K,
                          t_dust_lo=X.T_DUST_PRIOR_K[0],
                          t_dust_hi=X.T_DUST_PRIOR_K[1], t_dust_source="prior")
        assert res.verdict != "cessation", (t_true, res.z, res.f_cess)


def test_fitting_the_dust_temperature_from_two_bands_beats_the_prior():
    """Two same-epoch bands pin T_dust and shrink the transfer uncertainty."""
    t_true = 300.0
    exc9 = 0.5
    exc18 = exc9 * B.transfer(B.BANDS["S9W"], B.BANDS["L18W"], t_true)
    m9, _ = _source(exc9, 0.0, early="S9W", late="W3")
    m18, _ = _source(exc18, 0.0, early="L18W", late="W4")
    t_fit, lo, hi, src = X.fit_dust_temperature([m9, m18])
    assert src == "fitted"
    assert t_fit == pytest.approx(t_true, rel=0.15), t_fit
    # The delta-chi2 <= 1 interval must be far narrower than the prior. With
    # noiseless input it collapses to the grid spacing, so compare the *ratio*.
    assert (hi / lo) < 1.5
    prior_ratio = X.T_DUST_PRIOR_K[1] / X.T_DUST_PRIOR_K[0]
    assert (hi / lo) < prior_ratio


def test_dust_temperature_falls_back_to_the_prior_with_one_band():
    m9, _ = _source(0.5, 0.0, early="S9W")
    t, lo, hi, src = X.fit_dust_temperature([m9])
    assert src == "prior"
    assert (lo, hi) == X.T_DUST_PRIOR_K


def test_rising_excess_is_labelled_rise_not_cessation():
    t = 400.0
    exc_early = 0.1
    exc_late = 5.0 * exc_early * B.transfer(B.BANDS["I12"], B.BANDS["W3"], t)
    m_e, m_l = _source(exc_early, exc_late, t_dust=t)
    res = X.cessation(m_e, m_l, t_dust_k=t, t_dust_lo=t, t_dust_hi=t,
                      t_dust_source="fitted")
    assert res.verdict == "rise"
    assert res.z < 0


def test_no_early_excess_is_not_a_cessation():
    m_e, m_l = _source(0.0, 0.0)
    res = X.cessation(m_e, m_l, t_dust_k=400.0, t_dust_lo=400.0, t_dust_hi=400.0,
                      t_dust_source="fitted")
    assert res.verdict == "no_early_excess"


def test_saturated_late_band_refuses_to_claim_a_fade():
    """A saturated WISE measurement under-reports flux and fakes exactly this signal."""
    m_e, m_l = _source(0.5, 0.0)
    m_l.saturated = True
    m_l.f_obs_jy = B.BANDS["W3"].sat_jy + 1.0
    res = X.cessation(m_e, m_l, t_dust_k=400.0, t_dust_lo=400.0, t_dust_hi=400.0,
                      t_dust_source="fitted")
    assert res.verdict == "late_saturated"


def test_monte_carlo_agrees_with_the_analytic_path_when_temperature_is_pinned():
    t = 400.0
    m_e, m_l = _source(0.5, 0.0, t_dust=t)
    ana = X.cessation(m_e, m_l, t_dust_k=t, t_dust_lo=t, t_dust_hi=t,
                      t_dust_source="fitted")
    mc = X.cessation_mc(m_e, m_l, t_dust_lo=t * 0.999, t_dust_hi=t * 1.001,
                        n_draws=4000, seed=5)
    assert mc["delta_median_jy"] == pytest.approx(ana.delta_jy, rel=0.1)
    assert mc["p_delta_gt_0"] > 0.99


# ==========================================================================
# 5. Confounder: a variable AGB-like source must not flag
# ==========================================================================
def test_variable_agb_like_confounder_is_rejected():
    """A large mid-IR drop on a pulsating dusty giant must be killed by the funnel.

    The source genuinely faded, so the *statistic* fires; the funnel is what has
    to stop it. This is the correct division of labour: the detector measures,
    the funnel interprets.
    """
    t = 400.0
    m_e, m_l = _source(0.6, 0.04, t_dust=t)
    res = X.cessation(m_e, m_l, t_dust_k=t, t_dust_lo=t, t_dust_hi=t,
                      t_dust_source="fitted")
    assert res.verdict == "cessation"  # the statistic fires...

    agb = {
        "source_id": "agb-1",
        "ladder_verdict": "monotone_decline",
        "f_cess": res.f_cess,
        "beam_explained": False, "n_neighbours": 0,
        "iras_100um_bkg_mjysr": 2.0,
        "late_saturated": False, "early_saturated": False,
        "iras_qual": 3, "early_snr": 30.0,
        "cirr2": 1, "cirr3": 1,
        "sep_arcsec": 0.4, "n_optical_in_beam": 1,
        "parallax_over_error": 40.0, "pm_over_error": 60.0,
        "solar_system_assoc": False,
        "w1_w2": 0.3,
        "iras_var": 98,                      # <- IRAS says it is variable
        "gaia_variable": True,
        "optical_amp_mag": 0.9,
        "neowise_w1_rms_mag": 0.30, "neowise_n_epochs": 40,
        "spectral_type": "M7III",
        "abs_g": -1.5, "bp_rp": 2.8,         # luminous and red: on the AGB
        "h_ks_excess": 0.02, "v_tan_kms": 55.0,
    }
    out = V.vet_source(agb)
    assert not out.passed
    assert out.stage_failed == "variability", out.reasons


def test_extreme_debris_disk_analogue_is_rejected_on_youth():
    """A young star with a collisional dust reservoir is the known natural analogue."""
    edd = _clean_candidate_row()
    edd.update({"source_id": "edd-1", "v_tan_kms": 8.0,
                "young_moving_group": "beta Pic", "h_ks_excess": 0.25})
    out = V.vet_source(edd)
    assert not out.passed
    assert out.stage_failed == "not_young"


def test_persistently_variable_late_epoch_is_rejected():
    """Every natural confounder keeps varying; the signature is a step-and-stay."""
    row = _clean_candidate_row()
    row.update({"source_id": "wobbler", "neowise_w1_rms_mag": 0.25})
    out = V.vet_source(row)
    assert not out.passed
    assert out.stage_failed == "late_epoch_flat"


# ==========================================================================
# 6. Beam-sum consistency (the dominant IRAS contaminant)
# ==========================================================================
def test_beam_sum_explains_a_blended_iras_source():
    """Several WISE sources summing to the IRAS flux means resolution, not a fade."""
    ratio = B.audit_pair("I12", "W3").transfer_300k
    neighbours = [0.30, 0.25, 0.20, 0.15]  # W3 fluxes, Jy
    iras = sum(neighbours) / ratio
    out = X.beam_sum_consistency(iras, 0.05 * iras, neighbours, ratio)
    assert out["beam_explained"]
    assert out["n_neighbours"] == 4


def test_beam_sum_leaves_a_real_fade_unexplained():
    ratio = B.audit_pair("I12", "W3").transfer_300k
    neighbours = [0.02]
    iras = 3.0
    out = X.beam_sum_consistency(iras, 0.05 * iras, neighbours, ratio)
    assert not out["beam_explained"]
    assert out["z_unexplained"] > 3


# ==========================================================================
# 7. Three-epoch adjudication
# ==========================================================================
def _res(verdict: str) -> X.CessationResult:
    return X.CessationResult(
        early_band="I12", late_band="W3", baseline_yr=26.9, t_dust_k=400.0,
        t_dust_source="fitted", exc_early_jy=1.0, exc_early_err_jy=0.05,
        exc_late_pred_jy=1.0, exc_late_pred_err_jy=0.05, exc_late_obs_jy=0.0,
        exc_late_obs_err_jy=0.05, delta_jy=1.0, delta_err_jy=0.07, z=14.0,
        f_cess=1.0, verdict=verdict, f_cess_err=0.1)


def test_ladder_separates_the_two_fade_epochs():
    assert X.adjudicate_ladder(_res("cessation"), _res("cessation"),
                               _res("cessation"))[0] == "monotone_decline"
    # IRAS high, AKARI high, WISE low -- the TYC 8241 morphology.
    assert X.adjudicate_ladder(_res("cessation"), _res("constant"),
                               _res("cessation"))[0] == "fade_2007_2010"
    # IRAS high, AKARI low, WISE low.
    assert X.adjudicate_ladder(_res("cessation"), _res("cessation"),
                               _res("constant"))[0] == "fade_1983_2006"


def test_ladder_rejects_incoherent_and_unadjudicable_cases():
    assert X.adjudicate_ladder(_res("cessation"), _res("cessation"),
                               _res("rise"))[0] == "incoherent"
    v, notes = X.adjudicate_ladder(_res("cessation"), None, None)
    assert v == "no_mid_epoch"
    assert any("IRAS blend" in n for n in notes)


def test_ladder_reports_insufficient_ir_with_nothing_to_go_on():
    assert X.adjudicate_ladder(None, None, None)[0] == "insufficient_ir"


# ==========================================================================
# 8. Empirical null calibration
# ==========================================================================
def test_null_threshold_comes_from_the_rising_tail():
    z = RNG.normal(0, 1, 200_000)
    null = X.calibrate_null(z)
    assert not null["degraded"]
    assert 2.5 < null["threshold"] < 5.0
    # A symmetric sample must show no excess of faders over risers.
    assert abs(null["asymmetry_excess"]) < 0.02 * len(z)


def test_null_detects_a_one_sided_injected_population():
    z = np.concatenate([RNG.normal(0, 1, 200_000), RNG.normal(9, 1, 300)])
    null = X.calibrate_null(z)
    assert null["asymmetry_excess"] > 100


def test_null_degrades_honestly_when_starved():
    null = X.calibrate_null(np.array([1.0, 2.0, 3.0]))
    assert null["degraded"]
    assert null["threshold"] == 5.0


# ==========================================================================
# 9. The contamination funnel — one case per rule
# ==========================================================================
def _clean_candidate_row() -> dict:
    """A source that passes every rule; individual tests break one field at a time."""
    return {
        "source_id": "clean-1",
        "ladder_verdict": "monotone_decline",
        "f_cess": 0.95,
        "beam_explained": False, "n_neighbours": 1,
        "iras_100um_bkg_mjysr": 1.5,
        "late_saturated": False, "early_saturated": False,
        "iras_qual": 3, "early_snr": 25.0,
        "cirr2": 1, "cirr3": 0,
        "sep_arcsec": 0.5, "n_optical_in_beam": 1,
        "parallax_over_error": 35.0, "pm_over_error": 40.0,
        "solar_system_assoc": False,
        "w1_w2": 0.05,
        "iras_var": 3, "gaia_variable": False, "optical_amp_mag": 0.01,
        "neowise_w1_rms_mag": 0.02, "neowise_w1_slope_mag_yr": 0.0005,
        "neowise_w1_slope_sigma": 0.001, "neowise_n_epochs": 42,
        "spectral_type": "K2V",
        "abs_g": 6.2, "bp_rp": 1.1,
        "h_ks_excess": 0.01, "v_tan_kms": 65.0,
    }


def test_clean_candidate_passes_the_whole_funnel():
    out = V.vet_source(_clean_candidate_row())
    assert out.passed, out.reasons
    assert out.checks_run == out.checks_possible
    assert out.coverage_str.endswith(f"of_{len(V.RULES)}_observed_channels")


@pytest.mark.parametrize(
    ("field", "value", "expected_stage"),
    [
        ("ladder_verdict", "incoherent", "ladder_coherent"),
        ("ladder_verdict", "no_mid_epoch", "ladder_coherent"),
        ("f_cess", 0.2, "fade_amplitude"),
        ("f_cess", 0.10, "fade_amplitude"),          # inside 3x the 4% floor
        ("beam_explained", True, "beam_blending"),
        ("iras_100um_bkg_mjysr", 25.0, "far_ir_background"),
        ("late_saturated", True, "saturation"),
        ("early_saturated", True, "saturation"),
        ("iras_qual", 1, "early_quality"),
        ("early_snr", 3.0, "early_quality"),
        ("cirr3", 9, "cirrus"),
        ("sep_arcsec", 8.0, "association"),
        ("n_optical_in_beam", 3, "association"),
        ("parallax_over_error", 1.0, "stellar_astrometry"),
        ("pm_over_error", 0.5, "stellar_astrometry"),
        ("solar_system_assoc", True, "solar_system"),
        ("w1_w2", -0.4, "blend"),
        ("iras_var", 95, "variability"),
        ("gaia_variable", True, "variability"),
        ("optical_amp_mag", 0.5, "variability"),
        ("neowise_w1_rms_mag", 0.4, "late_epoch_flat"),
        ("spectral_type", "RCB", "not_eruptive"),
        ("spectral_type", "C-N5", "not_eruptive"),
        ("abs_g", -2.0, "not_evolved"),
        ("h_ks_excess", 0.5, "not_young"),
        ("v_tan_kms", 4.0, "not_young"),
        ("young_moving_group", "TW Hya", "not_young"),
        ("xray_active", True, "not_young"),
    ],
)
def test_every_rejection_rule_has_a_case_that_trips_it(field, value, expected_stage):
    row = _clean_candidate_row()
    if field == "abs_g":
        row["bp_rp"] = 2.0  # luminous AND red is what defines the giant branch
    row[field] = value
    out = V.vet_source(row)
    assert not out.passed, f"{field}={value} should have been rejected"
    assert out.stage_failed == expected_stage, out.reasons


def test_untestable_checks_do_not_count_as_passes():
    """A source that survives only because nothing could be tested is not a candidate."""
    row = {"source_id": "sparse", "ladder_verdict": "monotone_decline", "f_cess": 0.9}
    out = V.vet_source(row, require_all=True)
    assert not out.passed
    assert out.stage_failed == "coverage"
    assert out.checks_run < out.checks_possible
    assert any("could be evaluated" in r for r in out.reasons)


def test_vet_all_reports_funnel_counts():
    rows = [_clean_candidate_row(),
            {**_clean_candidate_row(), "source_id": "b", "beam_explained": True},
            {**_clean_candidate_row(), "source_id": "c", "iras_var": 99}]
    out = V.vet_all(rows)
    assert out["n_in"] == 3
    assert out["n_survivors"] == 1
    assert out["killed_by_stage"]["beam_blending"] == 1
    assert out["killed_by_stage"]["variability"] == 1


# ==========================================================================
# 10. End-to-end: injection, confounders, and honest degradation
# ==========================================================================
def _end_to_end_table(n_bg: int = 900, seed: int = 4) -> tuple[pd.DataFrame, list[str]]:
    """A synthetic catalogue: constant-excess background plus injected switch-offs.

    Every source is built from a real photospheric colour locus plus, for half
    of them, a genuine blackbody excess at 400 K transported into each band with
    the module's own transfer function. Six sources keep their early excess and
    lose it entirely by the WISE epoch.

    Fluxes are drawn faint enough that WISE W3 never saturates -- otherwise the
    saturation guard fires before the statistic is ever exercised, which is the
    behaviour tested separately.
    """
    rng = np.random.default_rng(seed)
    all_bands = ("I12", "I25", "S9W", "L18W", "W3", "W4")
    t_dust = 400.0
    # Excess transfer relative to the I12 band, for a 400 K blackbody.
    tr = {b: B.transfer(B.BANDS["I12"], B.BANDS[b], t_dust) for b in all_bands}
    # Photospheric transfer relative to Ks, as a linear colour locus per band.
    loc = {b: 0.02 + 0.05 * i for i, b in enumerate(all_bands)}

    rows, injected = [], []
    for i in range(n_bg):
        colour = rng.uniform(0.5, 2.0)
        ks = rng.uniform(0.01, 0.06)          # Jy: keeps W3 well below 0.96 Jy
        switch_off = i < 6
        # Injected sources always carry a large early excess; the background
        # carries one half the time.
        exc12 = ks * rng.uniform(4.0, 9.0) if (switch_off or rng.random() < 0.5) else 0.0

        row = {"source_id": f"src{i:05d}", "bp_rp": colour,
               "ks_jy": ks, "ks_err_jy": 0.005 * ks}
        for b in all_bands:
            phot = ks * 10 ** (-0.4 * loc[b] * colour)
            exc = exc12 * tr[b]
            if switch_off and b in ("W3", "W4"):
                exc = 0.0                      # the excess switched off by 2010
            f = phot + exc
            row[f"f_{b}_jy"] = f * (1 + rng.normal(0, 0.02))
            row[f"e_{b}_jy"] = 0.02 * f
        if switch_off:
            injected.append(row["source_id"])
        rows.append(row)
    return pd.DataFrame(rows), injected


def test_end_to_end_recovers_injected_switch_offs(tmp_path):
    from seti.config import load_config

    table, injected = _end_to_end_table()
    cfg = load_config()
    cfg.root = tmp_path
    summary = ember_run(cfg, stage="all", table=table)

    assert summary["verdict"] in ("CANDIDATES", "NO_SURVIVOR")
    assert summary["counts"]["acquired"] == len(table)
    shortlist = json.loads((tmp_path / "results" / "ember" / "shortlist.json").read_text())
    found = {s["source_id"] for s in shortlist}
    hit = found & set(injected)
    assert len(hit) >= 5, f"recovered only {sorted(hit)} of {injected}"

    # The constant-excess background must not flood the shortlist. This is the
    # test that catches a photosphere-prediction bias: any per-band systematic
    # in the predicted photosphere shows up here as wholesale false fades.
    n_false = len(found - set(injected))
    assert n_false <= 0.01 * len(table), f"{n_false} background false positives"

    # And the injected population must be enormously enriched over background.
    rate_inj = len(hit) / len(injected)
    rate_bg = n_false / (len(table) - len(injected))
    assert rate_inj > 20 * max(rate_bg, 1e-4), (rate_inj, rate_bg)


def test_end_to_end_writes_the_audit_and_report(tmp_path):
    from seti.config import load_config

    table, _ = _end_to_end_table(n_bg=300)
    cfg = load_config()
    cfg.root = tmp_path
    ember_run(cfg, stage="all", table=table)
    out = tmp_path / "results" / "ember"
    assert (out / "pair_audit.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "REPORT.md").exists()
    report = (out / "REPORT.md").read_text()
    assert "Honest sensitivity statement" in report
    assert "not a result" in report


def test_empty_archive_degrades_honestly(tmp_path):
    """An unreached archive must produce NO_DATA_REACHED, never a candidate."""
    from seti.config import load_config

    cfg = load_config()
    cfg.root = tmp_path
    summary = ember_run(cfg, stage="all", table=pd.DataFrame())
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["counts"]["survivors"] == 0
    assert "not a null result" in summary["note"]
    # The systematics audit still runs, because it needs no data at all.
    assert (tmp_path / "results" / "ember" / "pair_audit.json").exists()


def test_audit_stage_runs_with_no_data_at_all(tmp_path):
    payload = stage_audit(tmp_path)
    assert payload["usable_pairs"]
    assert all(k in payload["rsr_source"] for k in ("I12", "S9W", "W3"))
    # With no cached SVO curves the fallback must be declared, not hidden.
    assert set(payload["rsr_source"].values()) <= {"svo", "trapezoid"}


def test_summary_records_which_response_model_was_used(tmp_path):
    from seti.config import load_config

    cfg = load_config()
    cfg.root = tmp_path
    summary = ember_run(cfg, stage="audit", table=None)
    assert summary["verdict"] == "AUDIT_ONLY"
    assert "rsr_source" in summary["pair_audit"]


# ==========================================================================
# 11. Acquisition: the failure modes that cost run 30203763934
# ==========================================================================
#
# That run reported `acquired: 0` and `archive_reachable: false`. Both were
# false. AKARI returned 871,331 rows and was checkpointed to parquet; IRAS was
# lost because its RA column did not resolve and the RuntimeError was swallowed
# by a bare `except`; Gaia was lost to HTTP 500 on a 20,000-row upload. Three
# different causes, one indistinguishable zero. Every test below pins one of
# them.
from seti.ember import acquire as A  # noqa: E402


def test_ucd_resolution_beats_an_unknown_column_name():
    """A column nobody has aliased is still found, because its UCD says so."""
    schema = [
        {"column_name": "SomeUnaliasedRA", "ucd": "pos.eq.ra;meta.main",
         "unit": "deg", "description": "Right ascension"},
        {"column_name": "SomeUnaliasedDE", "ucd": "pos.eq.dec;meta.main",
         "unit": "deg", "description": "Declination"},
        {"column_name": "Fnu_12", "ucd": "phot.flux.density", "unit": "Jy",
         "description": "12 micron flux"},
    ]
    res = A.resolve_positions(schema, [r["column_name"] for r in schema],
                              A._IRAS_ALIASES)
    assert res["ra"] == "SomeUnaliasedRA"
    assert res["dec"] == "SomeUnaliasedDE"
    assert res["route"] == "ucd"
    assert res["frame"] == "icrs"


def test_b1950_columns_resolve_and_are_flagged_for_precession():
    """The exact IRAS regression: B1950 column names, no J2000 alias in sight."""
    names = ["IRAS", "RA1950", "DE1950", "Fnu_12", "Fnu_25", "Fnu_100"]
    res = A.resolve_positions(None, names, A._IRAS_ALIASES)
    assert res["ra"] == "RA1950" and res["dec"] == "DE1950"
    assert res["frame"] == "b1950", "a B1950 column used as J2000 is 0.5 deg wrong"
    assert res["route"] == "alias"


def test_unresolvable_positions_are_reported_not_silently_empty():
    res = A.resolve_positions(None, ["col_a", "col_b"], A._IRAS_ALIASES)
    assert res["ra"] is None and res["dec"] is None
    assert res["route"] == "unresolved"


def test_precession_moves_a_b1950_position_by_about_half_a_degree():
    """B1950 -> J2000 is a real, large shift; treating it as a no-op loses matches."""
    ra_j, dec_j = A.precess_b1950_to_j2000([0.0, 180.0], [0.0, 30.0])
    sep = A.angular_sep_arcsec(ra_j, dec_j, [0.0, 180.0], [0.0, 30.0])
    assert np.all(sep > 900.0), sep       # > 15 arcmin everywhere
    assert np.all(sep < 3600.0), sep      # but of order half a degree, not degrees
    # A known anchor: B1950 (0,0) precesses to J2000 (0.640, 0.279) deg.
    ra0, dec0 = A.precess_b1950_to_j2000([0.0], [0.0])
    assert ra0[0] == pytest.approx(0.6404, abs=0.01)
    assert dec0[0] == pytest.approx(0.2784, abs=0.01)


def test_status_separates_a_failed_query_from_an_empty_one():
    assert A._classify(pd.DataFrame({"a": [1]}), None) == A.STATUS_OK
    assert A._classify(pd.DataFrame(), None) == A.STATUS_ZERO_ROWS
    assert A._classify(None, RuntimeError("boom")) == A.STATUS_QUERY_FAILED
    assert A.STATUS_ZERO_ROWS != A.STATUS_QUERY_FAILED


def test_build_working_table_records_which_archive_failed(tmp_path):
    """AKARI succeeds, IRAS raises, Gaia raises -- and all three are legible."""
    akari = pd.DataFrame({"ra": [10.0, 11.0], "dec": [1.0, 2.0],
                          "s09": [0.5, 0.6], "s18": [0.4, 0.5]})

    def _akari(st):
        st.query = "SELECT ... FROM II/297/irc"
        return akari

    def _iras(st):
        raise RuntimeError("could not resolve RA/Dec columns")

    def _gaia(st):
        raise RuntimeError("HTTP 500 from the ESA archive")

    df, status = A.build_working_table(
        0.0, 60.0, tmp_path,
        fetchers={"akari": _akari, "iras_psc": _iras, "iras_fsc": _iras,
                  "gaia": _gaia})

    assert df.empty
    # The archive that WORKED is recorded as having worked.
    assert status["archive_reachable"] is True
    assert status["counts"]["akari"] == 2
    assert status["fetches"]["akari"]["status"] == A.STATUS_OK
    # The archives that FAILED are recorded as failures, not as empty sky.
    for name in ("iras_psc", "iras_fsc", "gaia"):
        assert status["fetches"][name]["status"] == A.STATUS_QUERY_FAILED
        assert status["fetches"][name]["error"]
    assert status["stopped_at"] == "gaia"
    # The expensive infrared table survives a downstream archive outage.
    assert list(tmp_path.glob("ir_anchor_*.parquet")), \
        "871k catalogued rows must not be discarded because Gaia 500'd"


def test_build_working_table_distinguishes_a_genuinely_empty_sky(tmp_path):
    """Queries that work and return nothing are ZERO_ROWS, never QUERY_FAILED."""
    df, status = A.build_working_table(
        0.0, 60.0, tmp_path,
        fetchers={"akari": lambda st: pd.DataFrame(),
                  "iras_psc": lambda st: pd.DataFrame(),
                  "iras_fsc": lambda st: pd.DataFrame()})
    assert df.empty
    assert status["archive_reachable"] is False
    assert status["stopped_at"] == "early_epoch_catalogues"
    for name in ("akari", "iras_psc", "iras_fsc"):
        assert status["fetches"][name]["status"] == A.STATUS_ZERO_ROWS
        assert not status["fetches"][name]["error"]


def _write_shard_status(root, shard, fetches, stopped_at=None):
    cache = root / "results" / "ember" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    payload = {"source": "archives", "shard": shard, "fetches": fetches,
               "counts": {}, "errors": []}
    if stopped_at:
        payload["stopped_at"] = stopped_at
    (cache / f"acquire_status_{shard:03d}.json").write_text(json.dumps(payload))


def test_analyse_reports_a_failed_query_as_a_query_failure(tmp_path):
    """The headline regression: 871k AKARI rows must not read as 'no archive'."""
    from seti.config import load_config

    _write_shard_status(tmp_path, 0, {
        "akari": {"label": "akari", "status": "OK", "n_rows": 871331,
                  "query": "SELECT TOP 4000000 ... FROM \"II/297/irc\"",
                  "error": None, "detail": {}, "strategy": ""},
        "gaia": {"label": "gaia", "status": "QUERY_FAILED", "n_rows": 0,
                 "query": "SELECT ... FROM tap_upload.targets",
                 "error": "HTTPError('Error 500')", "detail": {}, "strategy": ""},
    }, stopped_at="gaia")

    cfg = load_config()
    cfg.root = tmp_path
    summary = ember_run(cfg, stage="analyse")

    assert summary["verdict"] == "ACQUISITION_QUERY_FAILED"
    acq = summary["acquisition"]
    assert acq["archive_reachable"] is True
    assert acq["archives_that_returned_rows"] == ["akari"]
    assert acq["per_archive"]["akari"]["n_rows"] == 871331
    assert summary["acquisition_failure"]["query_failed"] == ["gaia"]
    assert summary["acquisition_failure"]["stopped_at"] == ["gaia"]
    # The literal query text is carried so a reader can re-issue it.
    assert "II/297/irc" in acq["per_archive"]["akari"]["query"]


def test_analyse_reports_a_genuinely_empty_result_as_zero_rows(tmp_path):
    from seti.config import load_config

    _write_shard_status(tmp_path, 0, {
        "akari": {"label": "akari", "status": "QUERY_RETURNED_ZERO_ROWS",
                  "n_rows": 0, "query": "SELECT ...", "error": None,
                  "detail": {}, "strategy": ""},
    }, stopped_at="early_epoch_catalogues")

    cfg = load_config()
    cfg.root = tmp_path
    summary = ember_run(cfg, stage="analyse")
    assert summary["verdict"] == "ARCHIVES_RETURNED_ZERO_ROWS"
    assert summary["acquisition_failure"]["returned_zero_rows"] == ["akari"]
    assert summary["acquisition_failure"]["query_failed"] == []
    assert "verify them" in summary["note"]


def test_analyse_with_no_acquisition_record_at_all_says_no_data_reached(tmp_path):
    from seti.config import load_config

    cfg = load_config()
    cfg.root = tmp_path
    summary = ember_run(cfg, stage="analyse")
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert "not a null result" in summary["note"]


def test_allwise_uses_bulk_xmatch_above_the_cone_ceiling(monkeypatch):
    """10^5 per-object cone queries is not a strategy; the ceiling is enforced."""
    n = A.ALLWISE_CONE_MAX_ROWS + 10
    rows = pd.DataFrame({"match_id": [f"m{i}" for i in range(n)],
                         "ra": np.linspace(10, 20, n), "dec": np.zeros(n),
                         "pmra": np.zeros(n), "pmdec": np.zeros(n)})
    calls = {"xmatch": 0, "cone": 0}

    def _fake_xmatch(positions, cat2, radius_arcsec=6.0, **kw):
        calls["xmatch"] += 1
        return pd.DataFrame({"match_id": positions["match_id"].to_numpy(),
                             "AllWISE": ["J000" for _ in range(len(positions))],
                             "W1mag": np.full(len(positions), 8.0),
                             "W3mag": np.full(len(positions), 5.0),
                             "angDist": np.full(len(positions), 0.5)})

    def _fake_cone(*a, **kw):
        calls["cone"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(A, "xmatch_cds", _fake_xmatch)
    monkeypatch.setattr(A, "fetch_allwise_cone", _fake_cone)

    st = A.FetchStatus(label="allwise")
    out = A._allwise_for_rows(rows, None, status=st)
    assert calls["xmatch"] == 1 and calls["cone"] == 0
    assert st.strategy == "cds_xmatch_vizier_II328"
    assert st.status == A.STATUS_OK
    # VizieR column names are mapped onto the canonical ones the science uses.
    assert "w1mpro" in out.columns and "w3mpro" in out.columns
    assert st.detail["xmatch_column_mapping"]["w1mpro"] == "W1mag"


def test_allwise_falls_back_to_cones_when_xmatch_fails(monkeypatch):
    n = A.ALLWISE_CONE_MAX_ROWS + 5
    rows = pd.DataFrame({"match_id": [f"m{i}" for i in range(n)],
                         "ra": np.linspace(10, 20, n), "dec": np.zeros(n),
                         "pmra": np.zeros(n), "pmdec": np.zeros(n)})

    def _boom(*a, **kw):
        raise RuntimeError("xmatch down")

    monkeypatch.setattr(A, "xmatch_cds", _boom)
    monkeypatch.setattr(A, "fetch_allwise_cone", lambda *a, **kw: pd.DataFrame())
    st = A.FetchStatus(label="allwise")
    A._allwise_for_rows(rows, None, status=st)
    assert st.strategy == "irsa_cone_per_object"
    assert "xmatch_error" in st.detail


#: The columns CDS X-Match actually returns for ``vizier:I/355/gaiadr3``,
#: measured by the probe in run 30209647320. Using the real names is the whole
#: point: the bug these tests pin was invisible against invented ones.
_XM_GAIA_COLS = ("angDist", "match_id", "xm_query_ra", "xm_query_dec",
                 "DR3Name", "RAdeg", "DEdeg", "Source", "Plx", "e_Plx",
                 "pmRA", "e_pmRA", "pmDE", "e_pmDE", "RUWE", "Gmag",
                 "BPmag", "RPmag", "BP-RP", "Teff")


def _fake_xmatch_gaia(n=2, gaia_offset_deg=0.001):
    """An X-Match response shaped like the real one, Gaia offset from the IR."""
    ir_ra = np.linspace(10.0, 11.0, n)
    d = {"angDist": np.full(n, 1.2), "match_id": [f"m{i}" for i in range(n)],
         "xm_query_ra": ir_ra, "xm_query_dec": np.zeros(n),
         "DR3Name": [f"Gaia DR3 {i}" for i in range(n)],
         "RAdeg": ir_ra + gaia_offset_deg, "DEdeg": np.full(n, gaia_offset_deg),
         "Source": np.arange(n), "Plx": np.full(n, 5.0),
         "e_Plx": np.full(n, 0.02), "pmRA": np.full(n, 12.0),
         "e_pmRA": np.full(n, 0.03), "pmDE": np.full(n, -7.0),
         "e_pmDE": np.full(n, 0.03), "RUWE": np.full(n, 1.0),
         "Gmag": np.full(n, 10.0), "BPmag": np.full(n, 10.5),
         "RPmag": np.full(n, 9.6), "BP-RP": np.full(n, 0.9),
         "Teff": np.full(n, 5500.0)}
    return pd.DataFrame(d)[list(_XM_GAIA_COLS)]


def test_gaia_uses_cds_xmatch_first(monkeypatch):
    """Measured, probe run 30209647320: the ESA upload 500s even at 200 rows.

    It is therefore not a size limit a smaller chunk can duck, and leading with
    it costs 80 s an attempt to learn nothing. X-Match went first from then on.
    """
    positions = pd.DataFrame({"match_id": ["m0", "m1"], "ra": [10.0, 11.0],
                              "dec": [0.0, 0.0]})
    uploads: list[int] = []
    monkeypatch.setattr(A, "_gaia_upload",
                        lambda *a, **kw: uploads.append(1) or pd.DataFrame())
    monkeypatch.setattr(A, "xmatch_cds", lambda *a, **kw: _fake_xmatch_gaia())

    st = A.FetchStatus(label="gaia")
    out = A.fetch_gaia_for_positions(positions, status=st)
    assert st.strategy == "cds_xmatch_vizier_I355"
    assert st.status == A.STATUS_OK
    assert not uploads, "the ESA archive must not be tried when X-Match works"
    assert {"source_id", "ra", "dec", "parallax", "pmra", "pmdec", "ruwe",
            "bp_rp", "phot_g_mean_mag"} <= set(out.columns)


def test_gaia_xmatch_returns_the_gaia_position_not_the_infrared_one(monkeypatch):
    """The bug the probe caught before it could ship.

    An X-Match response carries the *uploaded* ``ra``/``dec`` next to the
    catalogue's ``RAdeg``/``DEdeg``. If the alias for ``ra`` resolves to the
    uploaded column, every Gaia position becomes the infrared position it was
    queried with, ``sep_arcsec`` collapses to ~0 for every source, and the
    astrometric association test -- the veto that rejects background galaxies --
    silently passes everything.
    """
    positions = pd.DataFrame({"match_id": ["m0", "m1"], "ra": [10.0, 11.0],
                              "dec": [0.0, 0.0]})
    monkeypatch.setattr(A, "_gaia_upload", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(A, "xmatch_cds",
                        lambda *a, **kw: _fake_xmatch_gaia(gaia_offset_deg=0.001))
    st = A.FetchStatus(label="gaia")
    out = A.fetch_gaia_for_positions(positions, status=st)

    assert st.detail["xmatch_column_mapping"]["ra"] == "RAdeg"
    assert st.detail["xmatch_column_mapping"]["dec"] == "DEdeg"
    # The Gaia position is genuinely displaced from the queried position.
    sep = A.angular_sep_arcsec(out["ra"], out["dec"],
                               out["xm_query_ra"], out["xm_query_dec"])
    assert np.all(sep > 3.0), f"Gaia position collapsed onto the IR one: {sep}"


def test_gaia_refuses_an_xmatch_result_with_no_usable_position(monkeypatch):
    """No position is a failure, never a silent substitution."""
    positions = pd.DataFrame({"match_id": ["a"], "ra": [1.0], "dec": [0.0]})
    monkeypatch.setattr(A, "xmatch_cds", lambda *a, **kw: pd.DataFrame(
        {"match_id": ["a"], "Source": [1], "Plx": [5.0]}))
    monkeypatch.setattr(A, "_gaia_upload",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            RuntimeError("HTTP 500")))
    st = A.FetchStatus(label="gaia")
    out = A.fetch_gaia_for_positions(positions, status=st)
    assert out.empty
    assert st.status == A.STATUS_QUERY_FAILED
    assert any("no usable Gaia position" in str(a.get("error"))
               for a in st.detail["attempts"])


def test_gaia_ladder_shrinks_the_upload_when_xmatch_is_unavailable(monkeypatch):
    """With X-Match down, the ESA ladder still shrinks before giving up."""
    positions = pd.DataFrame({"match_id": [f"m{i}" for i in range(30)],
                              "ra": np.linspace(0, 1, 30),
                              "dec": np.zeros(30)})
    tried: list[int] = []

    def _upload(pos, q, chunk, out_dir, retries):
        tried.append(chunk)
        if chunk > 2_000:
            raise RuntimeError("HTTP 500")
        return pd.DataFrame({"match_id": pos["match_id"],
                             "source_id": range(len(pos))})

    monkeypatch.setattr(A, "_gaia_upload", _upload)
    monkeypatch.setattr(A, "xmatch_cds",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            RuntimeError("xmatch down")))
    st = A.FetchStatus(label="gaia")
    out = A.fetch_gaia_for_positions(positions, status=st)
    assert tried == [5_000, 2_000]
    assert st.strategy == "esa_upload_2000"
    assert st.status == A.STATUS_OK and len(out) == 30


def test_xmatch_moves_the_uploaded_position_out_of_the_way(monkeypatch):
    """`xmatch_cds` renames the uploaded ra/dec so they cannot shadow cat2's."""
    monkeypatch.setattr(A, "_retry", lambda fn, **kw: fn())

    class _FakeXMatch:
        @staticmethod
        def query(cat1=None, cat2=None, max_distance=None, colRA1=None,
                  colDec1=None):
            n = len(cat1)

            class _R:
                @staticmethod
                def to_pandas():
                    return pd.DataFrame({
                        "angDist": np.full(n, 0.4),
                        "match_id": list(cat1["match_id"]),
                        "ra": list(cat1["ra"]), "dec": list(cat1["dec"]),
                        "RAJ2000": np.array(cat1["ra"]) + 1e-4,
                        "DEJ2000": np.array(cat1["dec"]) + 1e-4,
                        "W3mag": np.full(n, 5.0)})
            return _R()

    import astroquery.xmatch as _x
    monkeypatch.setattr(_x, "XMatch", _FakeXMatch)
    pos = pd.DataFrame({"match_id": ["a", "b"], "ra": [10.0, 11.0],
                        "dec": [0.0, 1.0]})
    out = A.xmatch_cds(pos, "vizier:II/328/allwise", 3.0)
    assert "xm_query_ra" in out.columns and "xm_query_dec" in out.columns
    assert "ra" not in out.columns and "dec" not in out.columns
    # And the catalogue position now maps cleanly onto the canonical name.
    named, mapping = A._rename_by_alias(out, A._ALLWISE_VIZIER_ALIASES)
    assert mapping["ra"] == "RAJ2000"
    assert not named.columns.duplicated().any()


def test_gaia_total_failure_is_query_failed_not_an_empty_sky(monkeypatch):
    positions = pd.DataFrame({"match_id": ["a"], "ra": [1.0], "dec": [0.0]})

    def _boom(*a, **kw):
        raise RuntimeError("everything is down")

    monkeypatch.setattr(A, "_gaia_upload", _boom)
    monkeypatch.setattr(A, "xmatch_cds", _boom)
    st = A.FetchStatus(label="gaia")
    out = A.fetch_gaia_for_positions(positions, status=st)
    assert out.empty
    assert st.status == A.STATUS_QUERY_FAILED
    assert st.error and "every Gaia strategy failed" in st.error


def test_alias_rename_does_not_collide_with_the_uploaded_columns():
    """An X-Match result carries BOTH the uploaded ra/dec and the catalogue's.

    Renaming ``RAJ2000`` onto ``ra`` when ``ra`` is already present yields two
    columns of the same name, after which ``df["ra"]`` is a DataFrame and every
    downstream numeric operation is quietly wrong.
    """
    raw = pd.DataFrame({"match_id": ["a"], "ra": [10.0], "dec": [1.0],
                        "AllWISE": ["J0001"], "RAJ2000": [10.0001],
                        "DEJ2000": [1.0001], "W3mag": [5.0]})
    out, mapping = A._rename_by_alias(raw, A._ALLWISE_VIZIER_ALIASES)
    assert not out.columns.duplicated().any(), list(out.columns)
    # The uploaded position keeps the canonical name; the catalogue's is kept
    # too, under a distinct one -- neither is silently dropped.
    assert out["ra"].tolist() == [10.0]
    assert out["ra_cat"].tolist() == [10.0001]
    assert out["w3mpro"].tolist() == [5.0]
    assert mapping["ra"] == "RAJ2000"


def test_alias_rename_is_a_noop_when_names_already_canonical():
    raw = pd.DataFrame({"match_id": ["a"], "w1mpro": [8.0], "w2mpro": [7.0]})
    out, mapping = A._rename_by_alias(raw, A._ALLWISE_VIZIER_ALIASES)
    assert list(out.columns) == list(raw.columns)
    assert mapping["w1mpro"] == "w1mpro"


def test_acquire_status_lands_inside_the_uploaded_cache_directory(tmp_path):
    """The status must travel in the artifact, or the diagnosis is unreadable.

    The workflow uploads ``results/ember/cache/`` and the analyse job
    re-downloads exactly that. Writing the acquisition status beside
    ``summary.json`` instead is how run 30203763934 ended up with no record at
    all of *why* its shards were empty.
    """
    from seti.ember.run import load_acquire_status, stage_acquire

    out = tmp_path / "results" / "ember"
    df, status = stage_acquire(
        out, n_ra_chunks=1, shard=2, n_shards=6,
        fetchers={"akari": lambda st: pd.DataFrame(),
                  "iras_psc": lambda st: pd.DataFrame(),
                  "iras_fsc": lambda st: pd.DataFrame()})
    assert df.empty
    path = out / "cache" / "acquire_status_002.json"
    assert path.exists(), "status must be inside the artifact directory"
    saved = json.loads(path.read_text())
    assert saved["shard"] == 2
    assert saved["fetches"]["akari"]["status"] == "QUERY_RETURNED_ZERO_ROWS"
    assert load_acquire_status(out)["shard_002"]["shard"] == 2

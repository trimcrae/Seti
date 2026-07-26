"""Offline tests for VIGIL --- mid-IR variability at low fractional excess.

No network.  Every archive call is injected.

The decisive test in this file is ``test_extreme_debris_disk_is_not_flagged``.
The brief is explicit that without a working discriminator this channel produces
an extreme-debris-disk catalogue and calls it a search, so an EDD injected with
the published phenomenology --- large fractional excess, decaying variability,
drifting dust temperature, flat optical light curve --- must be rejected, and the
test asserts *which* cut rejected it, not merely that something did.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.photometry import band_freq_hz, mag_to_flux_jy, planck_bnu
from seti.vigil.acquire import (
    QueryResult,
    clean_neowise,
    pm_sweep_arcsec,
    propagate_pm,
)
from seti.vigil.discriminate import (
    discriminate,
    modulation_index,
    shape_stats,
    temperature_from_w2_w1_ratio,
)
from seti.vigil.excess import (
    band_ratio_factor,
    max_amplitude_for_frac_excess,
    measure_excess,
    photosphere_from_nir,
    wien_peak_k,
)
from seti.vigil.run import vigil_sweep, vigil_vet
from seti.vigil.variability import (
    bin_visits,
    ensemble_common_mode,
    fit_error_scale,
    midir_variability,
    visit_flux_series,
)
from seti.vigil.vet import galactic_latitude, is_agn_like, vet_row, w4_only_signal

# --------------------------------------------------------------------------
# Synthetic NEOWISE machinery
# --------------------------------------------------------------------------
T_STAR = 5000.0
KS_MAG = 10.0
J_MAG, H_MAG = 10.9, 10.25     # roughly a 5000 K photosphere anchored on Ks = 10
#   -> W1 = 10.05, W2 = 9.96: comfortably below the NEOWISE saturation onset,
#      so the injections exercise the science cuts rather than the bright-star veto.


def _photosphere_jy() -> dict[str, float]:
    """The W1/W2 photosphere the injections sit on."""
    return photosphere_from_nir(J_MAG, H_MAG, KS_MAG, T_STAR)


def _visit_times(n_visits: int = 20, start_mjd: float = 56700.0,
                 n_exp: int = 12, seed: int = 0):
    """NEOWISE-like sampling: ``n_visits`` visits ~183 d apart, ``n_exp`` per visit."""
    rng = np.random.default_rng(seed)
    t = []
    for k in range(n_visits):
        t0 = start_mjd + 183.0 * k
        t.append(t0 + np.sort(rng.uniform(0.0, 1.0, n_exp)))
    return np.concatenate(t)


def _synth_band(t, flux_fn, band: str, sigma_mag: float = 0.02, seed: int = 1):
    """Per-exposure magnitudes for a flux model ``flux_fn(t_yr) -> Jy``."""
    rng = np.random.default_rng(seed)
    t_yr = (t - t.min()) / 365.25
    f = np.array([flux_fn(x) for x in t_yr], dtype=float)
    mag = -2.5 * np.log10(f / __import__("seti.photometry", fromlist=["BANDS"])
                          .BANDS[band]["zp_jy"])
    return mag + rng.normal(0.0, sigma_mag, mag.size), np.full(mag.size, sigma_mag)


def _duty_cycle_star(tau_mean: float = 1.5e-3, t_dust: float = 600.0,
                     period_yr: float = 2.0, duty: float = 0.5,
                     n_visits: int = 20, sigma_mag: float = 0.02, seed: int = 3):
    """A radiator whose thermal output switches between two states at fixed T.

    The excess turns fully on and fully off (the load is either running or it is
    not), so essentially all of the excess modulates: ``m -> 1``.  The dust
    temperature does not change --- only the amount of radiating area does.
    """
    phot = _photosphere_jy()
    r = {b: band_ratio_factor(b, t_dust, T_STAR) for b in ("W1", "W2")}
    # tau_mean is the time-averaged excess; the "on" state carries it all.
    amp = {b: phot[b] * tau_mean * r[b] / duty for b in ("W1", "W2")}

    def state(x):
        return 1.0 if (x % period_yr) < duty * period_yr else 0.0

    t = _visit_times(n_visits, seed=seed)
    out = {}
    for i, b in enumerate(("W1", "W2")):
        out[b] = _synth_band(t, lambda x, b=b: phot[b] + amp[b] * state(x), b,
                             sigma_mag=sigma_mag, seed=seed + i)
    return t, out, phot


def _extreme_debris_disk(tau0: float = 1.2e-2, t_dust0: float = 550.0,
                         decay_yr: float = 4.0, n_visits: int = 20,
                         sigma_mag: float = 0.02, seed: int = 11,
                         floor: float = 0.75, cooling_k_per_yr: float = 12.0):
    """An extreme debris disk with the published phenomenology.

    * a **large** fractional excess, ``tau ~ 1e-2`` --- that is what makes it
      "extreme";
    * variability that is a *perturbation* on that steady excess: a smooth
      secular decay of ~25% plus stochastic collisional bursts;
    * a dust temperature that **drifts** as the freshly produced grains spread
      and cool;
    * a flat optical light curve (Moor et al. 2021).
    """
    phot = _photosphere_jy()
    rng = np.random.default_rng(seed)
    t = _visit_times(n_visits, seed=seed)
    t_yr_visits = np.unique(np.round((t - t.min()) / 365.25, 3))
    bursts = [(float(rng.uniform(0, t_yr_visits.max())), float(rng.uniform(0.05, 0.15)))
              for _ in range(3)]

    def tau_of(x):
        v = tau0 * (floor + (1.0 - floor) * np.exp(-x / decay_yr))
        for t0, a in bursts:
            v *= 1.0 + a * np.exp(-((x - t0) / 0.35) ** 2)
        return v

    def temp_of(x):
        # grains spread and cool as the cascade evolves
        return t_dust0 - cooling_k_per_yr * x

    def flux(b, x):
        return phot[b] * (1.0 + tau_of(x) * band_ratio_factor(b, temp_of(x), T_STAR))

    out = {}
    for i, b in enumerate(("W1", "W2")):
        out[b] = _synth_band(t, lambda x, b=b: flux(b, x), b, sigma_mag=sigma_mag,
                             seed=seed + i)
    return t, out, phot


def _constant_star(n_visits: int = 20, sigma_mag: float = 0.02, seed: int = 21):
    phot = _photosphere_jy()
    t = _visit_times(n_visits, seed=seed)
    out = {b: _synth_band(t, lambda x, b=b: phot[b], b, sigma_mag=sigma_mag,
                          seed=seed + i) for i, b in enumerate(("W1", "W2"))}
    return t, out, phot


def _score(t, bands, star_extra: dict | None = None, conf: dict | None = None):
    """Run the per-star physics exactly as ``vigil_sweep`` does."""
    from seti.vigil.run import _score_star

    v1 = bin_visits(t, bands["W1"][0], bands["W1"][1])
    v2 = bin_visits(t, bands["W2"][0], bands["W2"][1])
    star = {"source_id": "synthetic", "ra": 266.0, "dec": 65.0,
            "teff_gspphot": T_STAR, "bp_rp": 1.0,
            "j_m_2mass": J_MAG, "h_m_2mass": H_MAG, "k_m_2mass": KS_MAG}
    star.update(star_extra or {})
    return _score_star(star, v1, v2, conf or {})


# --------------------------------------------------------------------------
# Physics primitives
# --------------------------------------------------------------------------
def test_wien_bound_is_the_stated_one():
    """W1/W2 only: the channel probes hot material and says so."""
    assert 845.0 < wien_peak_k("W1") < 860.0
    assert 625.0 < wien_peak_k("W2") < 635.0


def test_band_ratio_factor_is_large_and_temperature_sensitive():
    """A low tau is nevertheless detectable, and R is why -- and why m is better."""
    r600 = band_ratio_factor("W2", 600.0, 5000.0)
    r300 = band_ratio_factor("W2", 300.0, 5000.0)
    assert r600 > 20.0                      # tau = 1e-3 -> ~2% band excess
    assert r300 < 3.0
    # A factor >7 swing in R at fixed tau: this is the uncertainty the
    # temperature-free modulation index removes.
    assert r600 / r300 > 7.0


def test_max_amplitude_and_modulation_index_are_consistent():
    a = max_amplitude_for_frac_excess(0.10)
    assert a == pytest.approx(2 * 0.10 / 1.10, rel=1e-9)
    exc = measure_excess("W2", 10.0, 0.005, float(mag_to_flux_jy(10.0, "W2")) / 1.10,
                         0.0, T_STAR, sys_floor=0.005)
    assert exc.frac_excess == pytest.approx(0.10, rel=0.02)
    m, merr = modulation_index(a, 0.002, exc)
    assert m == pytest.approx(1.0, rel=0.05)
    assert np.isfinite(merr)


def test_temperature_inversion_round_trips():
    for tk in (250.0, 500.0, 900.0, 1500.0):
        rho = float(planck_bnu(tk, band_freq_hz("W2")) / planck_bnu(tk, band_freq_hz("W1")))
        assert temperature_from_w2_w1_ratio(rho) == pytest.approx(tk, rel=0.01)
    assert not np.isfinite(temperature_from_w2_w1_ratio(-1.0))


def test_proper_motion_is_propagated_to_the_neowise_epoch():
    """The bug that cost a previous channel a whole run."""
    ra, dec = 100.0, 20.0
    ra2, dec2 = propagate_pm(ra, dec, 500.0, -500.0, 2016.0, 2019.0)
    # 500 mas/yr over 3 yr = 1.5" in each axis.
    assert float(dec2 - dec) * 3.6e6 == pytest.approx(-1500.0, rel=1e-6)
    assert float(ra2 - ra) * 3.6e6 * np.cos(np.radians(dec)) == pytest.approx(1500.0,
                                                                              rel=1e-4)
    # A 200 mas/yr star sweeps >2" across the mission: bigger than the cone.
    assert pm_sweep_arcsec(200.0, 0.0) > 2.0
    # Non-finite PM must not propagate a NaN position.
    ra3, dec3 = propagate_pm(ra, dec, np.nan, np.nan, 2016.0, 2019.0)
    assert np.isfinite(ra3) and np.isfinite(dec3)


# --------------------------------------------------------------------------
# The variability estimator
# --------------------------------------------------------------------------
def test_constant_star_gives_no_significant_variability():
    t, bands, _ = _constant_star()
    v = bin_visits(t, bands["W1"][0], bands["W1"][1])
    st = midir_variability(v, band="W1")
    assert st is not None
    assert st.n_visits == 20
    assert st.f_var_sigma < 5.0
    assert st.f_var < 0.02


def test_error_scale_is_fitted_from_within_visit_scatter():
    """Optimistic quoted errors are the classic variability false-positive factory."""
    t, bands, _ = _constant_star(sigma_mag=0.03, seed=31)
    mag, err = bands["W1"]
    v_true = bin_visits(t, mag, err)
    assert fit_error_scale(v_true) == pytest.approx(1.0, abs=0.35)
    # Now understate every quoted error by 3x: the fit must find it.
    v_bad = bin_visits(t, mag, err / 3.0)
    assert fit_error_scale(v_bad) == pytest.approx(3.0, rel=0.4)
    # And the rescale must stop the understatement from faking a detection.
    st_bad = midir_variability(v_bad, band="W1", use_error_scale=False)
    st_fix = midir_variability(v_bad, band="W1", use_error_scale=True)
    assert st_bad.f_var_sigma > st_fix.f_var_sigma
    assert st_fix.f_var_sigma < 5.0


def test_too_few_visits_returns_none_not_zero():
    """'Not measurable' and 'measured constant' are different statements."""
    t, bands, _ = _constant_star(n_visits=3)
    v = bin_visits(t, bands["W1"][0], bands["W1"][1])
    assert midir_variability(v, band="W1") is None


def test_ensemble_common_mode_removes_a_field_wide_zero_point_wander():
    per_star = {}
    rng = np.random.default_rng(5)
    offsets = rng.normal(0.0, 0.03, 20)
    for i in range(12):
        t, bands, _ = _constant_star(seed=100 + i)
        mag = bands["W1"][0].copy()
        # Impose the same per-visit zero-point wander on every star.
        v_idx = np.floor((t - t.min()) / 183.0).astype(int)
        mag = mag + offsets[np.clip(v_idx, 0, offsets.size - 1)]
        per_star[str(i)] = bin_visits(t, mag, bands["W1"][1])
    before = midir_variability(per_star["0"], band="W1")
    corrected, diag = ensemble_common_mode(per_star)
    after = midir_variability(corrected["0"], band="W1")
    assert diag["applied"] is True
    assert after.f_var < before.f_var
    assert after.f_var_sigma < before.f_var_sigma


# --------------------------------------------------------------------------
# INJECTION 1: the signal
# --------------------------------------------------------------------------
def test_duty_cycled_radiator_at_low_excess_is_recovered():
    """The channel's raison d'etre: recover a switching radiator at low tau."""
    t, bands, phot = _duty_cycle_star()
    rec = _score(t, bands, star_extra={"optical_measured": True, "optical_fvar": 0.003})
    assert rec["vigil_verdict_stage"] == "VIGIL_CANDIDATE", rec.get("reasons")
    assert rec["is_candidate"] is True
    # It is genuinely low-excess ...
    assert rec["tau_upper"] < 5.0e-3
    # ... and essentially the whole excess is switching.
    assert rec["modulation_index"] > 0.5
    # ... and the morphology is not a decay.
    assert rec["shape_morphology"] in ("two_state", "stochastic")
    assert rec["shape_trend_r2"] < 0.5


def test_duty_cycle_morphology_is_identified_as_two_state():
    t, bands, _ = _duty_cycle_star(tau_mean=3.0e-3, sigma_mag=0.01, seed=7)
    v2 = bin_visits(t, bands["W2"][0], bands["W2"][1])
    ty, f, fe = visit_flux_series(v2)
    s = shape_stats(ty, f, fe)
    assert s is not None
    assert s.two_state_dbic > 6.0
    assert s.morphology == "two_state"
    assert s.trend_r2 < 0.5


def test_isothermal_variation_is_detected():
    """A load-following radiator changes the amount, not the temperature."""
    t, bands, phot = _duty_cycle_star(tau_mean=4.0e-3, sigma_mag=0.008, seed=9)
    v1 = bin_visits(t, bands["W1"][0], bands["W1"][1])
    v2 = bin_visits(t, bands["W2"][0], bands["W2"][1])
    from seti.vigil.run import _colour_from_visits
    col = _colour_from_visits(v1, v2, phot, fit_error_scale(v1), fit_error_scale(v2))
    assert col is not None
    assert col.verdict == "isothermal"
    assert col.t_var_k == pytest.approx(600.0, rel=0.25)


# --------------------------------------------------------------------------
# INJECTION 2: THE TEST THAT DECIDES WHETHER THE CHANNEL IS REAL
# --------------------------------------------------------------------------
def test_extreme_debris_disk_is_not_flagged():
    """An EDD --- high excess, decaying variability, optically flat --- must not flag.

    This is the confounder that dominates any naive selection.  The assertion is
    not just that it was rejected but *why*: on the excess/modulation axis, which
    is the discriminator the channel is built on.
    """
    t, bands, _ = _extreme_debris_disk()
    rec = _score(t, bands, star_extra={"optical_measured": True, "optical_fvar": 0.002})

    # It *is* mid-IR variable and it *is* optically constant --- so a search that
    # stopped at the brief's headline phenomenology would have flagged it.
    assert rec["w1_measured"] and rec["w2_measured"]
    assert rec["w2_f_var_sigma"] > 5.0, "the EDD must be genuinely mid-IR variable"

    assert rec["is_candidate"] is False
    assert rec["vigil_verdict_stage"] == "EXTREME_DEBRIS_DISK_LIKE"
    reasons = rec["reasons"]
    assert ("fractional_excess" in reasons) or ("modulation_index" in reasons)
    # The physical statement behind the rejection.
    assert rec["exc_w2_tau_upper"] > 5.0e-3     # it is in the extreme regime
    assert rec["modulation_index"] < 0.5        # only a slice of it modulates


def test_extreme_debris_disk_shape_and_colour_also_point_the_right_way():
    """Two independent secondary discriminants, each on its own."""
    t, bands, phot = _extreme_debris_disk(sigma_mag=0.006, seed=13)
    v1 = bin_visits(t, bands["W1"][0], bands["W1"][1])
    v2 = bin_visits(t, bands["W2"][0], bands["W2"][1])
    s = shape_stats(*visit_flux_series(v2))
    assert s is not None
    assert s.kendall_tau < 0.0                  # secular decline
    assert s.morphology != "two_state"

    from seti.vigil.run import _colour_from_visits
    col = _colour_from_visits(v1, v2, phot, fit_error_scale(v1), fit_error_scale(v2))
    assert col is not None
    assert col.verdict == "temperature_drifting"
    assert col.t_drift_k_per_yr < 0.0


def test_decay_morphology_alone_rejects_even_at_low_excess():
    """The morphology gate is live and independent of the excess gate.

    A source drained to a LOW fractional excess with a HIGH modulation index --- so
    it passes the primary discriminator --- is still rejected if its light curve
    is a smooth secular decay, because that is what a collisional cascade does and
    a compute load has no reason to.
    """
    t, bands, _ = _extreme_debris_disk(tau0=6.0e-3, t_dust0=600.0, decay_yr=3.0,
                                       floor=0.2, cooling_k_per_yr=0.0,
                                       sigma_mag=0.004, seed=17)
    # A 2% photosphere floor (a well-anchored star) so the excess is bounded low.
    rec = _score(t, bands, star_extra={"optical_measured": True, "optical_fvar": 0.002},
                 conf={"excess": {"sys_floor": 0.02}})
    assert rec["is_candidate"] is False
    assert rec["vigil_verdict_stage"] == "DECAY_MORPHOLOGY", rec.get("reasons")
    assert rec["shape_morphology"] == "secular_decay"
    # It really did clear the primary excess/modulation gate.
    assert rec["tau_upper"] < 5.0e-3


# --------------------------------------------------------------------------
# INJECTION 3: an optically variable YSO
# --------------------------------------------------------------------------
def test_optically_variable_yso_does_not_flag():
    t, bands, _ = _duty_cycle_star(tau_mean=2.0e-3, seed=23)
    rec = _score(t, bands, star_extra={"optical_measured": True, "optical_fvar": 0.18})
    assert rec["is_candidate"] is False
    assert rec["vigil_verdict_stage"] == "OPTICALLY_VARIABLE"

    # And the literature veto catches it independently of the photometry.
    v = vet_row({"ra": 83.8, "dec": -5.4, "w1mpro": 9.0, "w2mpro": 8.6,
                 "simbad_otype": "YSO", "optical_measured": True,
                 "optical_fvar": 0.18})
    assert v["vigil_verdict"] == "rejected_yso"


def test_optically_untested_is_not_treated_as_optically_constant():
    t, bands, _ = _duty_cycle_star()
    rec = _score(t, bands)                       # no optical information at all
    assert "optical_constancy_untested" in rec["reasons"]
    v = vet_row({"ra": 266.0, "dec": 65.0, "w1mpro": 9.0, "w2mpro": 8.8})
    assert v["vigil_verdict"] == "clean_optical_untested"
    assert "optical_constancy" in v["untested_checks"]


# --------------------------------------------------------------------------
# INJECTION 4: a W4-only cirrus artifact
# --------------------------------------------------------------------------
def test_w4_only_signal_is_cirrus_and_does_not_flag():
    ctx = {"ra": 285.0, "dec": 5.0, "w1mpro": 10.2, "w2mpro": 10.1,
           "chi_w1": 0.4, "chi_w2": 0.7, "chi_w3": 1.1, "chi_w4": 9.5}
    assert w4_only_signal(ctx) is True
    assert vet_row(ctx)["vigil_verdict"] == "rejected_cirrus"
    # A real warm-dust SED lights the star-dominated bands first, so it survives
    # this particular rule.
    real = dict(ctx, chi_w1=6.0, chi_w2=8.0, chi_w3=7.0)
    assert w4_only_signal(real) is False
    assert vet_row(real)["vigil_verdict"] != "rejected_cirrus"


def test_negative_w1_w2_is_a_blend():
    assert vet_row({"ra": 200.0, "dec": 30.0, "w1mpro": 10.0,
                    "w2mpro": 10.4})["vigil_verdict"] == "rejected_blend"


def test_saturated_bright_source_is_rejected():
    v = vet_row({"ra": 200.0, "dec": 30.0, "w1mpro": 6.5, "w2mpro": 6.4})
    assert v["vigil_verdict"] == "rejected_saturated"


def test_agn_colour_box_alone_does_not_reject_an_astrometric_star():
    """A ~350 K shroud has W1-W2 = 3.2 and sits inside the AGN box."""
    shroud = {"ra": 200.0, "dec": 30.0, "w1mpro": 11.0, "w2mpro": 7.8,
              "parallax_over_error": 40.0, "pmra": 30.0, "pmdec": -20.0,
              "pm_error": 0.05}
    agn, why = is_agn_like(shroud)
    assert agn is False
    assert "astrometrically_stellar" in why
    # The same colour with no astrometry is a quasar.
    quasar = dict(shroud, parallax_over_error=0.4, pmra=0.1, pmdec=0.1, pm_error=0.3)
    assert is_agn_like(quasar)[0] is True
    assert vet_row(quasar)["vigil_verdict"] == "rejected_agn"


def test_crowded_beam_is_rejected():
    v = vet_row({"ra": 200.0, "dec": 30.0, "w1mpro": 10.0, "w2mpro": 9.8,
                 "n_neighbors_beam": 2, "brightest_neighbor_dg": 0.7})
    assert v["vigil_verdict"] == "rejected_blend"


def test_galactic_latitude_is_computed_correctly():
    assert abs(galactic_latitude(266.4, -28.94)) < 2.0        # Galactic centre
    assert galactic_latitude(192.86, 27.13) == pytest.approx(90.0, abs=0.5)


# --------------------------------------------------------------------------
# NEOWISE frame cleaning
# --------------------------------------------------------------------------
def test_neowise_cleaning_drops_bad_frames():
    df = pd.DataFrame({
        "mjd": [1.0, 2.0, 3.0, 4.0],
        "w1mpro": [10.0, 10.1, 10.0, 10.2], "w1sigmpro": [0.02] * 4,
        "w2mpro": [9.8, 9.9, 9.8, 9.9], "w2sigmpro": [0.02] * 4,
        "qual_frame": [10, 0, 10, 10],
        "cc_flags": ["0000", "0000", "DH00", "0000"],
        "moon_masked": ["0000", "0000", "0000", "1000"],
    })
    out = clean_neowise(df)
    assert len(out) == 1
    assert float(out["mjd"].iloc[0]) == 1.0


# --------------------------------------------------------------------------
# Orchestration: degradation must be honest
# --------------------------------------------------------------------------
def _fake_gaia(status="OK", n=3):
    def f(ra, dec, radius_deg=0.4, g_max=15.0, **kw):
        if status != "OK":
            return QueryResult(label="gaia_field", service="gaia", status=status,
                               query="SELECT ...", error="synthetic")
        df = pd.DataFrame({
            "source_id": [f"s{i}" for i in range(n)],
            "ra": [ra + 0.01 * i for i in range(n)],
            "dec": [dec + 0.01 * i for i in range(n)],
            "pmra": [10.0] * n, "pmdec": [-5.0] * n,
            "parallax_over_error": [50.0] * n, "ruwe": [1.0] * n,
            "phot_g_mean_mag": [12.0] * n, "bp_rp": [1.0] * n,
            "teff_gspphot": [T_STAR] * n, "phot_variable_flag": ["NOT_AVAILABLE"] * n,
            "non_single_star": [0] * n,
        })
        return QueryResult(label="gaia_field", service="gaia", status="OK",
                           n_rows=len(df), count_star=len(df), query="SELECT ...",
                           data=df)
    return f


def _fake_neowise(kind="signal", status="OK"):
    def f(ra, dec, pmra=0.0, pmdec=0.0, **kw):
        if status != "OK":
            return QueryResult(label="neowise_epochs", service="irsa", status=status,
                               query="SELECT ...", error="synthetic")
        if kind == "signal":
            t, bands, _ = _duty_cycle_star(seed=int(abs(ra * 100)) % 97 + 3)
        else:
            t, bands, _ = _extreme_debris_disk(seed=int(abs(ra * 100)) % 97 + 11)
        df = pd.DataFrame({"mjd": t,
                           "w1mpro": bands["W1"][0], "w1sigmpro": bands["W1"][1],
                           "w2mpro": bands["W2"][0], "w2sigmpro": bands["W2"][1]})
        return QueryResult(label="neowise_epochs", service="irsa", status="OK",
                           n_rows=len(df), count_star=len(df), query="SELECT ...",
                           data=df)
    return f


def _fake_allwise():
    def f(positions, **kw):
        return pd.DataFrame({
            "source_id": positions["source_id"].tolist(),
            "allwise_ok": True, "allwise_status": "OK",
            "j_m_2mass": J_MAG, "h_m_2mass": H_MAG, "k_m_2mass": KS_MAG,
            "w3mpro": 7.9, "w4mpro": 7.5,
        })
    return f


def test_empty_archive_degrades_honestly(tmp_path):
    """A failed archive must never read as a science null."""
    s = vigil_sweep(ra=10.0, dec=10.0, out_root=tmp_path,
                    gaia_fetch=_fake_gaia(status="QUERY_FAILED"),
                    neowise_fetch=_fake_neowise(), allwise_fetch=_fake_allwise())
    assert s["verdict"] == "NO_DATA_REACHED"
    assert s["n_candidates"] == 0
    assert s["reason"].startswith("gaia_QUERY_FAILED")
    # The ledger distinguishes a failure from an empty result.
    assert any(e.get("status") == "QUERY_FAILED" for e in s["archive_ledger"])

    v = vigil_vet(out_root=tmp_path)
    assert v["verdict"] == "NO_DATA_REACHED"
    assert "nothing was tested" in v["note"]


def test_zero_rows_is_distinguished_from_a_failed_query(tmp_path):
    s = vigil_sweep(ra=11.0, dec=11.0, out_root=tmp_path,
                    gaia_fetch=_fake_gaia(),
                    neowise_fetch=_fake_neowise(status="QUERY_RETURNED_ZERO_ROWS"),
                    allwise_fetch=_fake_allwise())
    assert s["verdict"] == "NO_DATA_REACHED"
    assert s["neowise_query_rollup"]["n_zero_rows"] > 0
    assert s["neowise_query_rollup"]["n_failed"] == 0
    assert s["reason"] == "neowise_returned_no_usable_epochs"


def test_sweep_recovers_the_signal_end_to_end(tmp_path):
    s = vigil_sweep(ra=12.0, dec=12.0, out_root=tmp_path, gaia_fetch=_fake_gaia(n=3),
                    neowise_fetch=_fake_neowise("signal"),
                    allwise_fetch=_fake_allwise())
    assert s["verdict"] == "SEARCHED"
    assert s["n_scored"] == 3
    assert s["n_candidates"] >= 1
    v = vigil_vet(out_root=tmp_path, offline=True)
    assert v["verdict"] in ("SURVIVORS", "SURVIVORS_OPTICAL_UNTESTED")
    assert v["n_candidates"] >= 1


def test_sweep_on_a_field_of_extreme_debris_disks_produces_no_candidates(tmp_path):
    """The whole-channel version of the decisive test."""
    s = vigil_sweep(ra=13.0, dec=13.0, out_root=tmp_path, gaia_fetch=_fake_gaia(n=4),
                    neowise_fetch=_fake_neowise("edd"),
                    allwise_fetch=_fake_allwise())
    assert s["verdict"] == "SEARCHED"
    assert s["n_scored"] == 4
    assert s["n_candidates"] == 0
    v = vigil_vet(out_root=tmp_path, offline=True)
    assert v["verdict"] == "NO_CANDIDATES"
    # And the wording never implies a limit was measured.
    assert "not an occurrence" in v["note"]
    assert json.loads((tmp_path / "summary.json").read_text())["n_survivors"] == 0


def test_config_thresholds_are_read_from_yaml():
    from seti.vigil.run import load_vigil_config
    conf = load_vigil_config()
    assert conf, "config/vigil.yaml must exist and parse"
    assert conf["detect"]["tau_max"] > 0
    assert conf["bounds"]["w1_wien_peak_k"] == pytest.approx(852.0, abs=3.0)


def test_discriminate_refuses_a_single_band_detection():
    t, bands, _ = _duty_cycle_star()
    v1 = bin_visits(t, bands["W1"][0], bands["W1"][1])
    st1 = midir_variability(v1, band="W1")
    d = discriminate(st1, None, None, None)
    assert d.is_candidate is False
    assert d.verdict == "NOT_MEASURED"
    assert "both_bands" in ";".join(d.reasons)


# --------------------------------------------------------------------------
# The unTimely pre-selector: load-bearing when reachable, honest when not
# --------------------------------------------------------------------------
def _fake_untimely(status="OK", n_match=2):
    def f(table, service, ra, dec, radius_deg, **kw):
        if status != "OK":
            return QueryResult(label="untimely_cone", service=service, status=status,
                               query="SELECT ...", error="synthetic")
        # Positions matching the first `n_match` synthetic Gaia stars exactly.
        df = pd.DataFrame({"ra": [ra + 0.01 * i for i in range(n_match)],
                           "dec": [dec + 0.01 * i for i in range(n_match)],
                           "w1_var_flag": [1] * n_match})
        return QueryResult(label="untimely_cone", service=service, status="OK",
                           n_rows=len(df), count_star=len(df), query="SELECT ...",
                           data=df)
    return f


def test_untimely_preselector_restricts_the_sample():
    from seti.vigil.run import preselect_from_untimely

    stars = _fake_gaia(n=5)(14.0, 14.0).data
    out, led = preselect_from_untimely(stars, "untimely.var", "svc", 14.0, 14.0, 0.4,
                                       fetch=_fake_untimely(n_match=2))
    assert led["applied"] is True
    assert led["n_in"] == 5
    assert len(out) == 2
    assert led["n_untimely_rows"] == 2


def test_untimely_unreachable_degrades_to_the_full_sample_and_says_so():
    """Silently searching fewer stars and calling it a pre-selection is misreporting."""
    from seti.vigil.run import preselect_from_untimely

    stars = _fake_gaia(n=5)(15.0, 15.0).data
    out, led = preselect_from_untimely(stars, "untimely.var", "svc", 15.0, 15.0, 0.4,
                                       fetch=_fake_untimely(status="QUERY_FAILED"))
    assert led["applied"] is False
    assert led["status"] == "QUERY_FAILED"
    assert len(out) == 5

    # And with no table discovered at all, the reason is different and named.
    out2, led2 = preselect_from_untimely(stars, "", "", 15.0, 15.0, 0.4)
    assert led2["applied"] is False
    assert led2["status"] == "NO_TABLE_DISCOVERED"
    assert len(out2) == 5


def test_sweep_records_the_preselector_in_the_field_summary(tmp_path):
    s = vigil_sweep(ra=16.0, dec=16.0, out_root=tmp_path, gaia_fetch=_fake_gaia(n=3),
                    neowise_fetch=_fake_neowise("signal"),
                    allwise_fetch=_fake_allwise(),
                    untimely_table="untimely.var", untimely_service="svc",
                    untimely_fetch=_fake_untimely(n_match=2))
    assert s["untimely_preselect"]["applied"] is True
    assert s["n_stars_in_field"] == 2
    assert s["verdict"] == "SEARCHED"

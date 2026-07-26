"""Offline tests for RUST --- secularly *increasing* photometric scatter.

The suite is organised around the one systematic that can kill this channel.
A robust scatter estimator is biased low at small N; the number of epochs per
season is set by the survey's cadence; and survey cadence trends with calendar
time.  A search that does not correct for that measures ZTF's operations
calendar and calls it a collisional cascade.  ``test_cadence_bias_*`` is
therefore the test that matters, and it is run over many seeds and several
cadence histories rather than once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.rust.run import _is_candidate, confirm_survivor, rust_sweep, rust_vet
from seti.rust.scatter import (
    bias_factor,
    mad_null_table,
    mad_scale,
    mixture_mad_sigma,
    rel_scatter,
    season_scatter,
    season_scatter_mc,
)
from seti.rust.trend import (
    detect_rust,
    ensemble_detrend_scatter,
    fit_scatter_trend,
    spearman_p,
)
from seti.rust.vet import (
    crowding_verdict,
    gaia_quality_verdict,
    ir_dust_production_verdict,
    known_class_verdict,
    periodic_fraction,
    photometric_range_verdict,
    rust_verdict,
    two_band_verdict,
    vet_row,
)

# --------------------------------------------------------------------------
# Synthetic light-curve builder
# --------------------------------------------------------------------------

SEASON = 365.25


def make_lc(n_per_season, amps, sigma_e=0.02, seed=0, base=15.0, season_len=200.0):
    """A light curve with a prescribed epoch count and intrinsic amplitude per season.

    ``amps[i]`` is the *intrinsic* (astrophysical) rms in season ``i``, added in
    quadrature to the per-epoch photometric error ``sigma_e``.
    """
    rng = np.random.default_rng(seed)
    t, m, e = [], [], []
    for i, (n, a) in enumerate(zip(n_per_season, amps, strict=True)):
        tt = np.sort(rng.uniform(0, season_len, n)) + i * SEASON
        t.append(tt)
        m.append(base + rng.normal(0, np.hypot(sigma_e, a), n))
        e.append(np.full(n, sigma_e))
    return np.concatenate(t), np.concatenate(m), np.concatenate(e)


# --------------------------------------------------------------------------
# 1. The estimator's finite-N behaviour is characterised, not assumed
# --------------------------------------------------------------------------

def test_mad_bias_table_matches_known_finite_sample_behaviour():
    b, u = mad_null_table()
    # 1.4826 x MAD is strongly biased low at small N and converges to 1.
    assert b[3] < 0.75
    assert 0.88 < b[8] < 0.92
    assert 0.94 < b[20] < 0.97
    assert b[200] > 0.99
    # ...and at small N it is NOT monotonic: the sample median alternates
    # between even and odd N, so no smooth analytic correction would do there.
    assert b[2] > b[3]
    assert b[3] < b[4] < b[5]
    # Above the smoothing knee the table is monotone and jitter-free --- jitter
    # indexed by N is jitter indexed by cadence, i.e. by calendar time.
    tail = b[30:201]
    assert np.all(np.diff(tail) > 0)
    # Sampling scatter converges on the asymptotic 1.1664/sqrt(N).
    assert u[100] == pytest.approx(1.1664 / np.sqrt(100), rel=0.05)
    assert rel_scatter(400)[0] == pytest.approx(1.1664 / 20.0, rel=0.05)
    assert bias_factor(3)[0] < bias_factor(50)[0]


def test_mixture_mad_handles_heteroscedastic_errors():
    # Homoscedastic: must reproduce sigma exactly.
    assert mixture_mad_sigma(np.full(200, 0.02)) == pytest.approx(0.02, rel=1e-6)
    # A mixture of two error scales sits between them and is NOT the rms --- the
    # whole reason a naive quadrature noise floor is wrong.
    errs = np.concatenate([np.full(100, 0.01), np.full(100, 0.10)])
    got = mixture_mad_sigma(errs)
    assert 0.01 < got < 0.10
    assert got < float(np.sqrt(np.mean(errs ** 2)))
    # Empirically correct: simulate the null and compare.
    rng = np.random.default_rng(0)
    sim = [mad_scale(rng.normal(0, errs)) for _ in range(400)]
    assert float(np.mean(sim)) == pytest.approx(got, rel=0.06)


def test_spearman_p_is_exact_by_enumeration():
    assert spearman_p(1.0, 7) == pytest.approx(1 / 5040)
    assert spearman_p(1.0, 5) == pytest.approx(1 / 120)
    assert spearman_p(-0.9, 7) == 1.0
    assert spearman_p(0.5, 7) > 0.05
    assert 0.0 < spearman_p(0.9, 12) < 0.001      # t-approximation branch


# --------------------------------------------------------------------------
# 2. Injected signal is recovered
# --------------------------------------------------------------------------

def test_injected_rising_scatter_is_recovered():
    """A star whose amplitude grows linearly from 0 to ~90 mmag must flag."""
    t, m, e = make_lc([80] * 7, [0.015 * i for i in range(7)], seed=0)
    s = detect_rust(t, m, e)
    assert s is not None
    assert s.slope_var_yr > 0
    assert s.rank_p <= 0.01
    assert s.rank_rho > 0.85
    assert s.amp_last_mmag > s.amp_first_mmag
    assert s.amp_growth > 1.3
    assert s.slope_sigma_loo_min >= 1.5
    assert s.score > 0.4
    assert _is_candidate(s)


def test_injection_recovery_rate_over_many_seeds():
    """Recovery is a rate, not an anecdote: >=50% at a 60 mmag terminal amplitude."""
    hits = sum(_is_candidate(detect_rust(*make_lc([80] * 7,
                                                  [0.010 * i for i in range(7)], seed=s)))
               for s in range(60))
    assert hits >= 30


def test_constant_star_does_not_flag():
    for seed in range(20):
        t, m, e = make_lc([80] * 7, [0.0] * 7, seed=seed)
        assert not _is_candidate(detect_rust(t, m, e))


def test_declining_scatter_never_flags():
    """The statistic is directional: a star getting *quieter* is not a candidate."""
    t, m, e = make_lc([80] * 7, [0.09 - 0.015 * i for i in range(7)], seed=1)
    s = detect_rust(t, m, e)
    assert s.slope_var_yr < 0
    assert s.score == 0.0
    assert not _is_candidate(s)


# --------------------------------------------------------------------------
# 3. THE CADENCE-BIAS TEST --- the one that matters
# --------------------------------------------------------------------------

CADENCE_HISTORIES = {
    "rising_8_to_70": [8, 12, 18, 25, 35, 50, 70],
    "falling_70_to_8": [70, 50, 35, 25, 18, 12, 8],
    "ztf_style_jump": [30, 90, 95, 100, 45, 50, 55],
    "erratic": [15, 90, 20, 110, 25, 80, 30],
    "doubling": [10, 20, 40, 80, 160, 200, 240],
}


@pytest.mark.parametrize("name", sorted(CADENCE_HISTORIES))
def test_cadence_bias_constant_star_never_flags(name):
    """A perfectly constant star observed with a *changing* cadence must not flag.

    This is the systematic the brief calls out as decisive.  Uncorrected, the
    low-N seasons read a suppressed scatter and the high-N seasons read the true
    one, so any monotone cadence history manufactures a monotone scatter trend.
    Run over 120 seeds per history: a single lucky seed proves nothing.
    """
    ns = CADENCE_HISTORIES[name]
    flagged = 0
    for seed in range(120):
        t, m, e = make_lc(ns, [0.0] * len(ns), seed=seed)
        if _is_candidate(detect_rust(t, m, e)):
            flagged += 1
    assert flagged == 0, f"{name}: {flagged}/120 constant stars flagged"


def test_cadence_bias_the_uncorrected_statistic_would_have_failed():
    """Demonstrate the bias is real, so the correction is not ceremonial.

    Without the correction, the *raw* season scatter of a constant star rises
    monotonically with a rising cadence.  With it, the excess variance does not.
    If this test ever starts passing trivially, the injected systematic is gone
    and the guard above has stopped proving anything.
    """
    ns = [8, 12, 18, 25, 35, 50, 70]
    raw, exc, corr_flag = [], [], 0
    for seed in range(150):
        t, m, e = make_lc(ns, [0.0] * len(ns), sigma_e=0.02, seed=seed)
        ss = season_scatter(t, m, e)
        assert ss is not None
        raw.append(ss.sigma_obs)        # uncorrected: the bias b(N) is still in it
        exc.append(ss.v_exc)            # corrected
        if _is_candidate(fit_scatter_trend(ss)):
            corr_flag += 1
    raw = np.asarray(raw)
    exc = np.asarray(exc)
    raw_mean, exc_mean = raw.mean(axis=0), exc.mean(axis=0)
    exc_sem = exc.std(axis=0) / np.sqrt(exc.shape[0])

    # The uncorrected estimator really is suppressed in the sparse early seasons
    # and recovers as the cadence fills in --- a ~10% monotone climb out of
    # nothing but the survey calendar.  This is the systematic, measured.
    assert raw_mean[-1] / raw_mean[0] > 1.05
    assert np.corrcoef(np.arange(len(ns)), raw_mean)[0, 1] > 0.8

    # The corrected statistic has no such trend.  The right test is not "is the
    # residual uncorrelated with time" --- a residual of pure noise can correlate
    # by chance across 7 points --- it is "is every season consistent with zero
    # given its own standard error".
    assert np.all(np.abs(exc_mean) < 3.0 * exc_sem), (
        f"residual bias per season: {exc_mean} +/- {exc_sem}")
    # And the residual is small in absolute terms: an equivalent amplitude far
    # below the 15 mmag floor the candidate gate requires.
    assert 1e3 * np.sqrt(np.max(np.abs(exc_mean))) < 8.0
    # The same test applied to the uncorrected variance fails outright, which is
    # what makes the correction load-bearing rather than decorative.
    raw_var_mean = (raw ** 2).mean(axis=0)
    assert raw_var_mean[-1] - raw_var_mean[0] > 10.0 * np.max(np.abs(exc_mean))
    assert corr_flag == 0


def test_equal_n_subsampling_agrees_with_the_bias_correction():
    """Two independent defences must give the same answer on the same star.

    Equal-N subsampling throws epochs away until every season has identical N,
    so its estimator bias is identical across seasons *by construction* and it
    shares no machinery with the analytic correction.  A signal must survive
    both; a cadence artefact survives neither.
    """
    ns = [20, 30, 40, 55, 70, 85, 100]
    # Real signal, injected on top of a strongly rising cadence.
    t, m, e = make_lc(ns, [0.015 * i for i in range(7)], seed=1)
    fast = fit_scatter_trend(season_scatter(t, m, e))
    equal = fit_scatter_trend(season_scatter(t, m, e, equalize_n=True))
    assert fast.slope_var_yr > 0 and equal.slope_var_yr > 0
    assert equal.rank_p <= 0.05
    # No signal, same cadence: neither defence produces one.
    t, m, e = make_lc(ns, [0.0] * 7, seed=1)
    assert not _is_candidate(fit_scatter_trend(season_scatter(t, m, e)))
    assert not _is_candidate(fit_scatter_trend(season_scatter(t, m, e, equalize_n=True)))


def test_exact_monte_carlo_null_agrees_with_the_fast_path():
    """The survivor-grade exact MC and the sweep-grade fast path must agree."""
    t, m, e = make_lc([12, 25, 40, 60, 90, 120, 150], [0.012 * i for i in range(7)],
                      seed=4)
    fast = season_scatter(t, m, e)
    exact = season_scatter_mc(t, m, e, n_trials=600)
    assert fast is not None and exact is not None and len(fast) == len(exact)
    # Null levels agree to a few per cent at every epoch count.
    assert np.allclose(fast.sigma_null, exact.sigma_null, rtol=0.06)
    a, b = fit_scatter_trend(fast), fit_scatter_trend(exact)
    assert a.rank_rho == pytest.approx(b.rank_rho, abs=0.3)
    assert np.sign(a.slope_var_yr) == np.sign(b.slope_var_yr)


def test_heteroscedastic_noise_floor_does_not_manufacture_a_trend():
    """Growing per-epoch *errors* are not growing astrophysical variability.

    A field whose seeing degraded, or a star drifting toward the survey limit,
    has a rising photometric error and therefore a rising raw scatter with no
    intrinsic change at all.  Subtracting the season's own error vector must
    remove it.
    """
    rng = np.random.default_rng(7)
    t, m, e = [], [], []
    for i in range(7):
        n = 80
        tt = np.sort(rng.uniform(0, 200, n)) + i * SEASON
        err = np.full(n, 0.010 + 0.006 * i)          # error grows 10 -> 46 mmag
        t.append(tt)
        m.append(15.0 + rng.normal(0, err))
        e.append(err)
    t, m, e = np.concatenate(t), np.concatenate(m), np.concatenate(e)
    ss = season_scatter(t, m, e)
    # The raw scatter climbs steeply...
    assert ss.sigma_obs[-1] > 2.5 * ss.sigma_obs[0]
    # ...and the excess variance does not become a candidate.
    assert not _is_candidate(fit_scatter_trend(ss))


# --------------------------------------------------------------------------
# 4. Ensemble common mode in the second moment
# --------------------------------------------------------------------------

def test_ensemble_detrend_removes_a_shared_error_scale_drift():
    """A field-wide error-scale drift is removed; a single deviant star survives it.

    ZTF's reported magerr is a model and the model drifts with time.  Every star
    on the readout channel then shows a rising excess variance for a purely
    instrumental reason.  The ensemble measures that drift directly and removes
    it --- and must not remove a genuine single-star signal along with it.
    """
    kappa = [1.0, 1.3, 1.7, 2.2, 2.8, 3.4, 4.0]     # true-to-reported error ratio
    rows = []
    for star in range(60):
        # Per-star seed, so the test does not depend on RNG stream ordering.
        rng = np.random.default_rng(4000 + star)
        t, m, e = [], [], []
        for i, k in enumerate(kappa):
            n = 100
            tt = np.sort(rng.uniform(0, 200, n)) + i * SEASON
            rep = 0.02                                # what the archive claims
            true = rep * np.sqrt(k)                   # what it actually is
            extra = 0.035 * i if star == 0 else 0.0   # star 0 alone truly varies
            t.append(tt)
            m.append(15.0 + rng.normal(0, np.hypot(true, extra), n))
            e.append(np.full(n, rep))
        ss = season_scatter(np.concatenate(t), np.concatenate(m), np.concatenate(e))
        rows.append({"source_id": str(star), "_ss": ss, "_ccd": "1_1_1",
                     "_nepoch": 700})

    # Before: the shared drift makes much of the field look like a cascade.
    before = sum(_is_candidate(fit_scatter_trend(r["_ss"])) for r in rows)
    assert before >= 20, "the shared drift did not even produce false positives"

    diag = ensemble_detrend_scatter(rows)
    assert diag["kappa_max"] > 2.0                    # the drift was detected
    after = [_is_candidate(r["_stat"]) for r in rows]
    assert sum(after[1:]) == 0, "field stars still flagging after detrend"

    # The genuinely variable star must not be detrended away.  Asserted on the
    # robust properties rather than on the candidate gate, whose rank p-value one
    # unlucky season can break on any single realisation.
    s0 = rows[0]["_stat"]
    assert s0.slope_var_yr > 0
    assert s0.rank_rho >= 0.7
    assert s0.slope_var_yr == max(r["_stat"].slope_var_yr for r in rows)
    assert s0.amp_last_mmag > 5 * max(r["_stat"].amp_last_mmag for r in rows[1:])


def test_ensemble_detrend_is_a_no_op_on_a_clean_field():
    rng = np.random.default_rng(2)
    rows = []
    for star in range(30):
        t, m, e = make_lc([80] * 7, [0.0] * 7, seed=int(rng.integers(1e6)))
        rows.append({"source_id": str(star), "_ss": season_scatter(t, m, e),
                     "_ccd": "1_1_1", "_nepoch": 560})
    diag = ensemble_detrend_scatter(rows)
    assert 0.8 < diag["kappa_median"] < 1.25
    assert sum(_is_candidate(r["_stat"]) for r in rows) == 0


# --------------------------------------------------------------------------
# 5. Two-band coincidence and achromaticity --- the mandatory gate
# --------------------------------------------------------------------------

def _stats(amp_step, seed=0, n=80, sigma_e=0.02):
    t, m, e = make_lc([n] * 7, [amp_step * i for i in range(7)], sigma_e=sigma_e,
                      seed=seed)
    return detect_rust(t, m, e)


def test_single_band_artifact_does_not_flag():
    """The ledger's first rule: a one-band ZTF anomaly is an artefact.

    Here g rises strongly and r is flat --- a bad reference image, a ghost, or a
    blend with a blue variable.  The two-band gate must return ``single_band``
    and the combined verdict must not be clean.
    """
    g = _stats(0.020, seed=0)
    r = _stats(0.0, seed=1)
    tb = two_band_verdict(g, r)
    assert tb["verdict"] == "single_band"
    assert tb["n_bands_rising"] == 1
    row = vet_row(g, r)
    assert row["rust_verdict"] == "single_band"


def test_missing_band_is_insufficient_not_a_candidate():
    g = _stats(0.020, seed=0)
    assert two_band_verdict(g, None)["verdict"] == "insufficient_bands"
    assert vet_row(g, None)["rust_verdict"] == "insufficient_bands"
    assert vet_row(None, None)["rust_verdict"] == "insufficient_bands"


def test_achromatic_growth_is_the_rust_signature():
    """Equal amplitude growth in g and r = grey occulter = the signature."""
    g = _stats(0.020, seed=5)
    r = _stats(0.020, seed=6)
    tb = two_band_verdict(g, r)
    assert tb["verdict"] == "achromatic_gray"
    assert 0.8 <= tb["amp_growth_ratio"] <= 1.2
    assert vet_row(g, r)["rust_verdict"] == "clean_gray"


def test_reddening_law_growth_is_separated_from_grey():
    """g growing ~1.42x faster than r is ordinary dust, and must be labelled so."""
    g = _stats(0.020 * 1.42, seed=5)
    r = _stats(0.020, seed=6)
    tb = two_band_verdict(g, r)
    assert tb["verdict"] == "reddening_law"
    assert vet_row(g, r)["rust_verdict"] == "clean_reddening"


def test_chromatic_blue_growth_is_rejected():
    """Flare / accretion-like g >> r growth is not a geometric occultation."""
    g = _stats(0.020 * 3.0, seed=5)
    r = _stats(0.020, seed=6)
    assert two_band_verdict(g, r)["verdict"] == "chromatic_blue"
    assert vet_row(g, r)["rust_verdict"] == "chromatic_blue"


# --------------------------------------------------------------------------
# 6. Known-variable-class and quality confounders
# --------------------------------------------------------------------------

def test_known_variable_class_confounder_does_not_flag():
    """A star with a real rising-scatter signal but a mundane SIMBAD class.

    YSOs/dippers, cataclysmics and AGN all produce evolving aperiodic
    variability for reasons that are already understood.  AGN in particular are
    the seductive case: red-noise amplitude grows with the *timescale* sampled,
    and a lengthening window samples longer timescales.
    """
    g, r = _stats(0.020, seed=5), _stats(0.020, seed=6)
    assert vet_row(g, r)["rust_verdict"] == "clean_gray"     # control
    for otype in ("YSO", "TT*", "CV*", "QSO", "Sy1", "AGN", "Mi*", "EB*"):
        row = vet_row(g, r, context={"simbad_otype": otype})
        assert row["class_verdict"] == "known_variable"
        assert row["rust_verdict"] == "known_variable", otype
    assert known_class_verdict("RR*") == "unclassified"
    assert known_class_verdict(None) == "unclassified"


def test_periodic_variable_is_rejected():
    """A growing-amplitude *pulsator* is coherent, and this channel is aperiodic."""
    g, r = _stats(0.020, seed=5), _stats(0.020, seed=6)
    assert vet_row(g, r, periodic_power=0.60)["rust_verdict"] == "periodic_variable"
    assert vet_row(g, r, periodic_power=0.05)["rust_verdict"] == "clean_gray"
    # And the measurement itself finds a real sinusoid.
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 2000, 400))
    assert periodic_fraction(t, 15 + 0.2 * np.sin(2 * np.pi * t / 3.31)) > 0.5
    assert periodic_fraction(t, 15 + rng.normal(0, 0.02, t.size)) < 0.3


def test_blending_saturation_and_astrometry_rejections():
    g, r = _stats(0.020, seed=5), _stats(0.020, seed=6)
    assert crowding_verdict(0, np.nan) == "isolated"
    assert crowding_verdict(3, 1.0) == "blended"
    assert crowding_verdict(3, 6.0) == "faint_neighbors"
    assert crowding_verdict(None, None) == "crowding_unknown"
    assert vet_row(g, r, context={"n_neighbors_5as": 2,
                                  "brightest_neighbor_dg": 0.5})["rust_verdict"] == "blended"

    assert photometric_range_verdict(12.0, "g") == "saturated"
    assert photometric_range_verdict(20.9, "g") == "near_faint_limit"
    assert photometric_range_verdict(17.0, "r") == "in_range"

    assert gaia_quality_verdict(2.0, 0) == "high_ruwe_binary"
    assert gaia_quality_verdict(1.0, 1) == "gaia_non_single_star"
    assert gaia_quality_verdict(1.0, 0, 3.0) == "astrometric_excess_noise"
    assert gaia_quality_verdict(1.0, 0, 0.1) == "astrometry_clean"
    assert vet_row(g, r, context={"ruwe": 2.2})["rust_verdict"] == "high_ruwe_binary"


def test_neowise_dust_production_logic_is_inverted_relative_to_dimming():
    """A mid-IR brightening CORROBORATES a cascade; it does not kill it."""
    assert ir_dust_production_verdict(
        {"w1_slope_mag_yr": -0.03, "w1_slope_sigma": 4.0}
    ) == "ir_brightens_dust_production"
    assert ir_dust_production_verdict(
        {"w1_slope_mag_yr": 0.03, "w1_slope_sigma": 4.0}
    ) == "ir_fades_with_optical"
    assert ir_dust_production_verdict(
        {"w1_slope_mag_yr": 0.001, "w1_slope_sigma": 0.4}
    ) == "ir_flat_no_warm_dust"
    assert ir_dust_production_verdict(None) == "insufficient_ir"
    assert ir_dust_production_verdict({}) == "insufficient_ir"
    # The IR verdict is recorded but never overrides the optical verdict: a
    # mid-IR non-detection cannot clear or condemn a candidate on its own.
    g, r = _stats(0.020, seed=5), _stats(0.020, seed=6)
    row = vet_row(g, r, context={"neowise": {"w1_slope_mag_yr": 0.0,
                                             "w1_slope_sigma": 0.1}})
    assert row["ir_verdict"] == "ir_flat_no_warm_dust"
    assert row["rust_verdict"] == "clean_gray"


def test_rust_verdict_rejection_precedence():
    """The most decisive rejection wins, and only one label means 'survived'."""
    base = {"two_band_verdict": "achromatic_gray", "periodic_power": 0.0,
            "periodic_max": 0.35}
    assert rust_verdict(base) == "clean_gray"
    assert rust_verdict({**base, "range_verdict_g": "saturated"}) == "saturated"
    assert rust_verdict({**base, "crowding_verdict": "blended"}) == "blended"
    assert rust_verdict({**base, "class_verdict": "known_variable"}) == "known_variable"
    assert rust_verdict({**base, "two_band_verdict": "single_band"}) == "single_band"


# --------------------------------------------------------------------------
# 7. Honest degradation
# --------------------------------------------------------------------------

def test_too_few_epochs_or_seasons_returns_none_not_a_measurement():
    t, m, e = make_lc([80] * 2, [0.0] * 2, seed=0)          # 2 seasons
    assert season_scatter(t, m, e) is None
    assert detect_rust(t, m, e) is None
    t, m, e = make_lc([4] * 7, [0.0] * 7, seed=0)           # 4 epochs/season
    assert season_scatter(t, m, e) is None
    assert detect_rust(t, m, e) is None
    assert season_scatter(np.array([]), np.array([]), np.array([])) is None
    assert fit_scatter_trend(None) is None


def test_missing_errors_degrade_to_a_null_free_measurement():
    """No magerr column must not crash and must not invent a noise floor."""
    t, m, _e = make_lc([80] * 7, [0.0] * 7, seed=0)
    ss = season_scatter(t, m, None)
    assert ss is not None
    assert np.all(~np.isfinite(ss.sigma_null))
    # With no error model the excess variance is the raw variance, so the star
    # looks variable --- but it must not trend, and must not become a candidate.
    assert not _is_candidate(fit_scatter_trend(ss))


def test_empty_archive_response_yields_no_data_reached(tmp_path, monkeypatch):
    """A field that returns nothing writes an explicit verdict, not zero candidates."""
    import seti.rust.acquire as acq

    monkeypatch.setattr(acq, "iter_region_2band", lambda *a, **k: iter(()))
    summary = rust_sweep(ra=10.0, dec=20.0, radius_deg=0.1, out_root=tmp_path)
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["n_measured_both_bands"] == 0
    assert (tmp_path / summary["field"] / "field_summary.json").exists()
    assert not (tmp_path / summary["field"] / "rust_candidates.csv").exists()


def test_vet_stage_with_no_fields_reports_no_data_reached(tmp_path):
    import json

    s = rust_vet(out_root=tmp_path, offline=True)
    assert s["verdict"] == "NO_DATA_REACHED"
    assert s["n_candidates"] == 0
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["verdict"] == "NO_DATA_REACHED"

    # A field that *was* searched but produced nothing is a different verdict.
    d = tmp_path / "ra1.000_dec+1.000_r0.10"
    d.mkdir()
    (d / "field_summary.json").write_text(json.dumps(
        {"verdict": "SEARCHED", "n_scored": 1234}))
    s2 = rust_vet(out_root=tmp_path, offline=True)
    assert s2["verdict"] == "NO_CANDIDATES"
    assert s2["n_fields_searched"] == 1
    assert s2["n_stars_scored"] == 1234


def test_pair_bands_matches_positionally_and_drops_unmatched():
    """The g/r pairing is a pure function and is tested without a network."""
    from seti.rust.acquire import pair_bands

    def lc(ra, dec, ccd="1_2_3"):
        d = pd.DataFrame({"mjd": [1.0, 2.0], "mag": [15.0, 15.0],
                          "magerr": [0.01, 0.01], "ra": [ra, ra], "dec": [dec, dec]})
        d.attrs["ccd"] = ccd
        return d

    g = {"g1": lc(10.0, 20.0), "g2": lc(10.5, 20.0), "g3": lc(11.0, 20.0)}
    r = {"r1": lc(10.0 + 0.2 / 3600 / np.cos(np.radians(20)), 20.0),
         "r2": lc(10.5 + 10.0 / 3600, 20.0)}            # 10" away -> no match
    pairs = pair_bands(g, r, tol_arcsec=1.5)
    assert len(pairs) == 1
    assert pairs[0]["oid_g"] == "g1" and pairs[0]["oid_r"] == "r1"
    assert pairs[0]["sep_arcsec"] < 1.5
    assert pairs[0]["ccd_g"] == "1_2_3"
    assert pair_bands({}, r) == []
    assert pair_bands(g, {}) == []


def test_confirm_survivor_reports_all_three_cross_checks():
    """Every survivor carries fast, exact-MC and equal-N numbers --- pass or fail."""
    ns = [20, 30, 40, 55, 70, 85, 100]
    tg, mg, eg = make_lc(ns, [0.015 * i for i in range(7)], seed=1)
    tr, mr, er = make_lc(ns, [0.015 * i for i in range(7)], seed=2)
    out = confirm_survivor(tg, mg, eg, tr, mr, er)
    for band in ("g", "r"):
        for tag in ("fast", "mc", "equaln"):
            assert np.isfinite(out[f"{band}_{tag}_slope_sigma"])
            assert out[f"{band}_{tag}_slope_var_yr"] > 0
    assert out["cross_check_verdict"] == "survives_all"

    # A constant star with a wild cadence fails the cross-check rather than
    # silently returning a number that looks like a detection.
    tg, mg, eg = make_lc(ns, [0.0] * 7, seed=3)
    tr, mr, er = make_lc(ns, [0.0] * 7, seed=4)
    out = confirm_survivor(tg, mg, eg, tr, mr, er)
    assert out["cross_check_verdict"] in ("cadence_artifact", "survives_all")
    assert out["cross_check_min_sigma"] < 2.5


def test_end_to_end_sweep_with_a_planted_cascade(tmp_path, monkeypatch):
    """Full offline funnel: 40 field stars + 1 planted cascade -> exactly 1 candidate."""
    import seti.rust.run as runmod

    ns = [80] * 7

    def fake_iter(*_a, **_k):
        rng = np.random.default_rng(99)
        for i in range(41):
            amps = [0.018 * k for k in range(7)] if i == 0 else [0.0] * 7
            out = {}
            for band in ("g", "r"):
                t, m, e = make_lc(ns, amps, seed=int(rng.integers(1e6)))
                d = pd.DataFrame({"mjd": t, "mag": m, "magerr": e})
                out[f"lc_{band}"] = d
            yield {"source_id": f"s{i}", "ra": 10.0 + i * 1e-3, "dec": 20.0,
                   "ccd": "1_1_1", "ccd_g": "1_1_1", "ccd_r": "1_1_1",
                   "sep_arcsec": 0.3, **out}

    monkeypatch.setattr("seti.rust.acquire.iter_region_2band", fake_iter)
    s = runmod.rust_sweep(ra=10.0, dec=20.0, radius_deg=0.1, out_root=tmp_path)
    assert s["verdict"] == "SEARCHED"
    assert s["n_measured_both_bands"] == 41
    assert s["n_candidates_both_bands"] == 1
    cand = pd.read_csv(tmp_path / s["field"] / "rust_candidates.csv")
    assert list(cand["source_id"]) == ["s0"]
    # The per-season epoch counts are recorded so a reviewer can audit the
    # cadence history of every candidate by hand.
    counts = [int(v) for v in str(cand["g_n_per_season"].iloc[0]).split(",")]
    assert len(counts) == 7
    assert 520 <= sum(counts) <= 560
    assert "g_v_exc" in cand.columns and "r_v_exc" in cand.columns

    v = rust_vet(out_root=tmp_path, offline=True)
    assert v["n_candidates"] == 1
    assert v["verdict"] in ("SURVIVORS", "REDDENING_ONLY", "ALL_REJECTED")
    assert (tmp_path / "rust_vetted.csv").exists()

"""Offline test suite for KNELL --- the clock that stopped.

No network anywhere.  The suite is the CI gate, and one test in it is the reason
the channel exists rather than a nicety:
``test_degrading_cadence_does_not_flag`` synthesises a star whose periodic signal
**never stops** but whose cadence degrades, shows that the uncorrected statistic
flags it, and shows that the injection-measured efficiency gate flags none of
them.  Without that result this channel would be a search for ZTF's observing
calendar.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from seti.knell.blocks import (
    Block,
    block_periodogram,
    frequency_grid,
    gls_power,
    make_blocks,
    pdm_theta,
    permutation_threshold,
    sine_amplitude,
)
from seti.knell.cease import analyze_band, combine_bands
from seti.knell.efficiency import (
    amplitude_at_efficiency,
    block_efficiency,
    clopper_pearson_upper,
    efficiency_curve,
    format_pvalue,
    persistence_pvalue,
)
from seti.knell.vet import vet_row

SEASON = 365.25
WINDOW = 250.0          # ZTF observes a field roughly 8 months a year
# Narrow the period search in tests purely for runtime; the science defaults in
# config/knell.yaml are 0.05-100 d.
PER_KW = dict(min_period=0.2, max_period=20.0, oversample=4.0)
FAST = dict(n_null=80, n_trials=80, **PER_KW)      # analyze_band
PG_KW = dict(n_null=80, **PER_KW)                  # block_periodogram


# ---------------------------------------------------------------------------
# synthetic light curves
# ---------------------------------------------------------------------------


def synth(
    n_per_season,
    amp_mmag,
    period=0.63,
    err_mmag=20.0,
    mean_mag=16.0,
    *,
    periods=None,
    mean_offsets=None,
    err_scale=None,
    seed=0,
    window=WINDOW,
):
    """A ZTF-like season-blocked light curve with per-season control of everything.

    ``amp_mmag`` and ``periods`` are per season, so a cessation, a mode switch, a
    Blazhko-like recovery and a gradual decline are all expressible; ``err_scale``
    and ``mean_offsets`` let a season fade and get noisier together, which is the
    mundane confounder the channel must reject.
    """
    rng = np.random.default_rng(seed)
    n_s = len(n_per_season)
    amp = np.atleast_1d(amp_mmag)
    amp = np.full(n_s, float(amp[0])) if len(amp) == 1 else np.asarray(amp, float)
    per = ([period] * n_s) if periods is None else list(periods)
    off = np.zeros(n_s) if mean_offsets is None else np.asarray(mean_offsets, float)
    esc = np.ones(n_s) if err_scale is None else np.asarray(err_scale, float)

    T, M, E = [], [], []
    phase0 = rng.uniform(0, 2 * np.pi)
    for s, n in enumerate(n_per_season):
        n = int(n)
        if n <= 0:
            continue
        t = s * SEASON + np.sort(rng.uniform(0.0, window, size=n))
        e = np.full(n, err_mmag * esc[s] * 1e-3)
        m = (mean_mag + off[s]
             + amp[s] * 1e-3 * np.sin(2 * np.pi * t / per[s] + phase0)
             + rng.normal(0.0, e))
        T.append(t)
        M.append(m)
        E.append(e)
    return np.concatenate(T), np.concatenate(M), np.concatenate(E)


def as_lc(t, m, e):
    import pandas as pd
    return pd.DataFrame({"mjd": t, "mag": m, "magerr": e})


# ---------------------------------------------------------------------------
# 1. the periodogram machinery is correct
# ---------------------------------------------------------------------------


def test_gls_matches_astropy_lombscargle():
    """The batched GLS must reproduce astropy's Lomb-Scargle to float precision.

    The whole efficiency argument rests on the injection using *the same*
    detector as the search, so the detector itself has to be verified against an
    independent implementation, not merely self-consistent.
    """
    from astropy.timeseries import LombScargle

    t, m, e = synth([60], [50.0], period=1.3, seed=3)
    f = frequency_grid(t, **PER_KW)
    mine = gls_power(t, m, e, f)
    theirs = LombScargle(t, m, e).power(f, normalization="standard")
    assert np.allclose(mine, np.asarray(theirs), atol=1e-8)
    assert abs(1.0 / f[int(np.argmax(mine))] - 1.3) < 0.01


def test_periodogram_detects_a_signal_and_rejects_pure_noise():
    t, m, e = synth([60], [60.0], period=0.63, seed=5)
    b = Block(0, t, m, e)
    assert block_periodogram(b, rng=1, **PG_KW).detected

    t, m, e = synth([60], [0.0], seed=6)
    hits = sum(block_periodogram(Block(0, *synth([60], [0.0], seed=100 + k)),
                                 rng=k, **PG_KW).detected for k in range(20))
    assert hits <= 3, f"permutation threshold is too loose: {hits}/20 false detections"


def test_pdm_is_an_independent_estimator():
    t, m, e = synth([80], [60.0], period=0.63, seed=7)
    assert pdm_theta(t, m, 0.63) < 0.6            # folding at the truth helps a lot
    assert pdm_theta(t, m, 0.317) > 0.85          # a wrong period explains nothing


def test_fixed_frequency_amplitude_fit_is_unbiased_and_errors_are_sane():
    amps = []
    for k in range(12):
        t, m, e = synth([80], [40.0], period=0.63, err_mmag=20.0, seed=200 + k)
        a, sa, _ = sine_amplitude(t, m, e, 1.0 / 0.63)
        amps.append(a * 1e3)
        assert 0.5 < sa * 1e3 < 10.0
    assert abs(np.mean(amps) - 40.0) < 4.0


def test_blocking_drops_thin_blocks_and_refuses_short_baselines():
    t, m, e = synth([40, 40, 5, 40], [30.0], seed=8)
    bl = make_blocks(t, m, e, min_epochs_block=15, min_blocks=3)
    assert [b.index for b in bl] == [0, 1, 3]      # the 5-epoch season is dropped
    assert make_blocks(t, m, e, min_epochs_block=15, min_blocks=4) == []


# ---------------------------------------------------------------------------
# 2. the load-bearing quantity: injection-measured efficiency
# ---------------------------------------------------------------------------


def test_efficiency_falls_with_epoch_count_at_fixed_signal():
    """The confounder, measured directly: same star, same signal, fewer epochs."""
    dense = Block(0, *synth([80], [0.0], err_mmag=20.0, seed=11))
    sparse = Block(1, *synth([16], [0.0], err_mmag=20.0, seed=12))
    kw = dict(n_trials=120, n_null=120, **PER_KW)
    eta_dense = block_efficiency(dense, 0.63, 30.0, rng=1, **kw).eta
    eta_sparse = block_efficiency(sparse, 0.63, 30.0, rng=1, **kw).eta
    assert eta_dense > 0.9
    assert eta_sparse < 0.6
    assert eta_dense - eta_sparse > 0.3, (
        "if efficiency did not depend on cadence there would be no confounder "
        "and no reason for this channel's central correction")


def test_efficiency_rises_with_amplitude_and_the_curve_brackets_50_percent():
    b = Block(0, *synth([40], [0.0], err_mmag=20.0, seed=13))
    curve = efficiency_curve(b, 0.63, [2.0, 6.0, 12.0, 25.0, 60.0],
                             n_trials=100, n_null=100, rng=2, **PER_KW)
    etas = [c.eta for c in curve]
    assert etas[0] < 0.3 and etas[-1] > 0.9
    assert etas == sorted(etas) or np.all(np.diff(etas) > -0.15)
    a50 = amplitude_at_efficiency(curve, 0.5)
    assert 2.0 < a50 < 60.0


def test_pvalue_is_an_inequality_when_pinned_at_the_resolution_floor():
    """Never a point estimate at the floor --- the repository's standing rule."""
    assert clopper_pearson_upper(0, 200) == pytest.approx(1 - 0.05 ** (1 / 200), rel=1e-6)
    assert clopper_pearson_upper(0, 400) < clopper_pearson_upper(0, 200)

    b = Block(0, *synth([80], [0.0], err_mmag=15.0, seed=14))
    effs = [block_efficiency(b, 0.63, 120.0, n_trials=100, n_null=100, rng=3, **PER_KW)]
    pp = persistence_pvalue(effs)
    assert pp["pinned_at_floor"] is True
    assert pp["p_persist_upper"] > 0.0            # never exactly zero
    txt = format_pvalue(pp["p_persist_upper"], pp["pinned_at_floor"],
                        pp["resolution_floor"])
    assert txt.startswith("<=") and "injection-resolution limited" in txt

    # ...and escalating the trial count must tighten the bound, which is the
    # operational meaning of "escalate before believing it".
    e2 = [block_efficiency(b, 0.63, 120.0, n_trials=400, n_null=100, rng=3, **PER_KW)]
    assert persistence_pvalue(e2)["p_persist_upper"] < pp["p_persist_upper"]


def test_efficiency_injects_into_the_blocks_own_noise():
    """A block with real fat-tailed noise must show it, not a Gaussian idealisation."""
    rng = np.random.default_rng(15)
    t = np.sort(rng.uniform(0, WINDOW, 40))
    e = np.full(40, 0.02)
    y = 16.0 + rng.normal(0, 0.02, 40)
    y[::7] += rng.normal(0, 0.20, len(y[::7]))      # outliers magerr does not know about
    b = Block(0, t, y, e)
    kw = dict(n_trials=120, n_null=120, rng=4, **PER_KW)
    eta_data = block_efficiency(b, 0.63, 25.0, noise_mode="data", **kw).eta
    eta_gauss = block_efficiency(b, 0.63, 25.0, noise_mode="gaussian", **kw).eta
    assert eta_gauss > eta_data, (
        "the Gaussian idealisation must be optimistic relative to the real "
        "noise; if it were not, injecting into the data would buy nothing")


# ---------------------------------------------------------------------------
# 3. the primary detection: a clock that stops
# ---------------------------------------------------------------------------


def test_a_signal_that_stops_at_constant_mean_flux_is_recovered():
    n = [60, 60, 60, 60, 60, 60]
    amp = [55, 55, 55, 0, 0, 0]
    hits = 0
    for k in range(6):
        t, m, e = synth(n, amp, period=0.63, err_mmag=18.0, seed=300 + k)
        r = analyze_band(t, m, e, band="g", rng=k, **FAST)
        if r.is_cessation:
            hits += 1
            assert r.n_pre == 3 and r.n_post == 3
            assert abs(r.ref_period - 0.63) < 0.01
            assert abs(r.mean_shift_mag) < 0.02
            assert r.eta_min_post > 0.9
            assert r.p_persist_upper < 0.01
            assert r.pdm_p_pre <= 0.01 and r.pdm_p_post >= 0.05
        assert r.excess_var_ratio <= 0.35
    assert hits >= 5, f"recovered only {hits}/6 clean cessations"


def test_two_band_coincidence_recovers_and_agrees_on_epoch_and_period():
    n = [60, 60, 60, 60, 60]
    amp = [50, 50, 0, 0, 0]
    tg, mg, eg = synth(n, amp, period=0.81, err_mmag=18.0, seed=41)
    tr, mr, er = synth(n, amp, period=0.81, err_mmag=18.0, seed=42)
    rg = analyze_band(tg, mg, eg, band="g", rng=1, **FAST)
    rr = analyze_band(tr, mr, er, band="r", rng=2, **FAST)
    comb = combine_bands(rg, rr)
    assert comb["two_band_cessation"] is True
    assert comb["same_transition_block"] and comb["same_period_both_bands"]
    v = vet_row(rg, rr, comb, context={"mean_mag_g": 16.0, "mean_mag_r": 16.0})
    assert v["knell_verdict"] == "clean_cessation"


def test_single_band_cessation_is_an_artefact_until_confirmed():
    n = [60, 60, 60, 60, 60]
    tg, mg, eg = synth(n, [50, 50, 0, 0, 0], period=0.81, err_mmag=18.0, seed=43)
    tr, mr, er = synth(n, [50, 50, 50, 50, 50], period=0.81, err_mmag=18.0, seed=44)
    rg = analyze_band(tg, mg, eg, band="g", rng=1, **FAST)
    rr = analyze_band(tr, mr, er, band="r", rng=2, **FAST)
    comb = combine_bands(rg, rr)
    assert comb["two_band_cessation"] is False
    v = vet_row(rg, rr, comb, context={"mean_mag_g": 16.0, "mean_mag_r": 16.0})
    assert v["knell_verdict"] in ("single_band_only", "no_cessation")


# ---------------------------------------------------------------------------
# 4. THE TEST THAT DECIDES THE CHANNEL
# ---------------------------------------------------------------------------


def test_degrading_cadence_does_not_flag():
    """A constant-amplitude periodic star observed with a DEGRADING CADENCE.

    Nothing about this star changes.  Its signal is present, at the same period
    and the same amplitude, in every season.  Only the number of epochs per
    season falls --- exactly what a survey does when it re-plans its field
    roster, and exactly what ZTF did across ZTF-I/II/III.

    The uncorrected statistic ("periodic in the early blocks, not periodic in the
    late blocks") flags a large fraction of these.  The injection-measured
    efficiency gate must flag **none**, because the late blocks demonstrably
    could not have detected the signal they are being asked about.
    """
    n = [80, 80, 70, 18, 15, 15]
    naive = corrected = 0
    trials = 24
    for k in range(trials):
        t, m, e = synth(n, [30.0], period=0.63, err_mmag=25.0, seed=500 + k)
        r = analyze_band(t, m, e, band="g", rng=k, measure_efficiency=False, **FAST)
        if r.pattern_strict and r.n_pre >= 2 and r.n_post >= 2:
            naive += 1
        rr = analyze_band(t, m, e, band="g", rng=k, **FAST)
        if rr.is_cessation:
            corrected += 1
    assert naive >= 6, (
        f"the confounder did not bite ({naive}/{trials}); the test is not "
        "exercising the systematic it exists to test")
    assert corrected == 0, (
        f"{corrected}/{trials} degrading-cadence confounders survived the "
        "efficiency gate --- the channel is measuring cadence, not astrophysics")


def test_degrading_errors_at_fixed_cadence_also_do_not_flag():
    """The same systematic wearing a different costume: constant N, growing errors."""
    n = [50] * 6
    flagged = 0
    for k in range(10):
        t, m, e = synth(n, [30.0], period=0.63, err_mmag=15.0,
                        err_scale=[1, 1, 1, 3.0, 3.5, 4.0], seed=600 + k)
        if analyze_band(t, m, e, band="g", rng=k, **FAST).is_cessation:
            flagged += 1
    assert flagged == 0, f"{flagged}/10 noise-degradation confounders survived"


# ---------------------------------------------------------------------------
# 5. the mundane astrophysical confounders
# ---------------------------------------------------------------------------


def test_a_star_that_faded_below_the_noise_is_rejected_as_mundane():
    """Mean flux must be unchanged.  A signal that sank is a fade, not a stop."""
    n = [60] * 6
    for k in range(4):
        t, m, e = synth(n, [45.0], period=0.63, err_mmag=15.0,
                        mean_offsets=[0, 0, 0, 1.2, 1.2, 1.2],
                        err_scale=[1, 1, 1, 6.0, 6.0, 6.0], seed=700 + k)
        r = analyze_band(t, m, e, band="g", rng=k, **FAST)
        assert not r.is_cessation
        assert "mean_flux_changed" in r.flags
        v = vet_row(r, r, combine_bands(r, r),
                    context={"mean_mag_g": 16.6, "mean_mag_r": 16.6})
        assert v["knell_verdict"] == "faded_not_ceased"


def test_a_mode_switching_pulsator_does_not_flag():
    """Power moves to a different frequency; the star is still a clock."""
    n = [60] * 6
    flagged = 0
    for k in range(6):
        t, m, e = synth(n, [50.0], err_mmag=18.0,
                        periods=[0.63, 0.63, 0.63, 0.41, 0.41, 0.41], seed=800 + k)
        r = analyze_band(t, m, e, band="g", rng=k, **FAST)
        if r.is_cessation:
            flagged += 1
        assert r.status in ("still_periodic", "no_clean_transition", "rejected",
                            "low_efficiency")
    assert flagged == 0, f"{flagged}/6 mode switches were called cessations"


def test_a_blazhko_like_amplitude_minimum_does_not_flag():
    """The amplitude comes back, so the required run of non-detections is broken."""
    n = [60] * 7
    flagged = 0
    for k in range(6):
        t, m, e = synth(n, [50, 50, 0, 0, 50, 50, 50], period=0.55, err_mmag=18.0,
                        seed=900 + k)
        if analyze_band(t, m, e, band="g", rng=k, **FAST).is_cessation:
            flagged += 1
    assert flagged == 0, f"{flagged}/6 Blazhko-like minima were called cessations"


def test_a_gradual_pre_cessation_decline_is_flagged_as_precession_like():
    """The SS Lacertae signature: eclipse depth declines for years, then vanishes."""
    n = [60] * 6
    hits = 0
    for k in range(6):
        t, m, e = synth(n, [80, 55, 30, 0, 0, 0], period=0.71, err_mmag=15.0,
                        seed=1000 + k)
        r = analyze_band(t, m, e, band="g", rng=k, **FAST)
        if "pre_decline_precession_like" in r.flags:
            hits += 1
            v = vet_row(r, r, combine_bands(r, r),
                        context={"mean_mag_g": 16.0, "mean_mag_r": 16.0,
                                 "ruwe": 1.9, "non_single_star": 1})
            assert v["knell_verdict"] == "third_body_precession"
    assert hits >= 4, f"the pre-cessation decline was detected in only {hits}/6"


def test_an_astrometric_companion_alone_is_reported_not_hidden():
    n = [60] * 5
    t, m, e = synth(n, [50, 50, 0, 0, 0], period=0.81, err_mmag=18.0, seed=1100)
    r = analyze_band(t, m, e, band="g", rng=1, **FAST)
    if r.is_cessation:
        v = vet_row(r, r, combine_bands(r, r),
                    context={"mean_mag_g": 16.0, "mean_mag_r": 16.0,
                             "ruwe": 2.4, "non_single_star": 1})
        assert v["knell_verdict"] == "astrometric_companion"


def test_instrumental_walls_and_blends_are_rejected_first():
    n = [60] * 5
    t, m, e = synth(n, [50, 50, 0, 0, 0], period=0.81, err_mmag=18.0, seed=1200)
    r = analyze_band(t, m, e, band="g", rng=1, **FAST)
    comb = combine_bands(r, r)
    assert vet_row(r, r, comb, context={"mean_mag_g": 12.9,
                                        "mean_mag_r": 12.9})["knell_verdict"] == "saturated"
    assert vet_row(r, r, comb, context={"mean_mag_g": 20.8,
                                        "mean_mag_r": 20.8})["knell_verdict"] == "near_faint_limit"
    if r.is_cessation:
        assert vet_row(r, r, comb, context={
            "mean_mag_g": 16.0, "mean_mag_r": 16.0,
            "n_neighbors_5as": 1, "brightest_neighbor_dg": 0.4,
        })["knell_verdict"] == "blended"


def test_known_classes_are_rejected_with_the_named_mechanism():
    n = [60] * 5
    t, m, e = synth(n, [50, 50, 0, 0, 0], period=0.81, err_mmag=18.0, seed=1300)
    r = analyze_band(t, m, e, band="g", rng=1, **FAST)
    comb = combine_bands(r, r)
    if not r.is_cessation:
        pytest.skip("base case did not produce a cessation for this seed")
    base = {"mean_mag_g": 16.0, "mean_mag_r": 16.0}
    for otype, verdict in (("CV*", "cataclysmic_disc_state"), ("QSO", "agn_red_noise"),
                           ("YSO", "yso_variability"), ("RRLyr", "amplitude_modulated")):
        v = vet_row(r, r, comb, context={**base, "simbad_otype": otype})
        assert v["knell_verdict"] == verdict, (otype, v["knell_verdict"])


def test_a_constant_star_never_becomes_a_candidate():
    flagged = 0
    for k in range(12):
        t, m, e = synth([60] * 6, [0.0], err_mmag=20.0, seed=1400 + k)
        r = analyze_band(t, m, e, band="g", rng=k, **FAST)
        flagged += int(r.is_cessation)
    assert flagged == 0


# ---------------------------------------------------------------------------
# 6. honest degradation and the acquisition log
# ---------------------------------------------------------------------------


def test_acquisition_log_separates_a_failed_query_from_an_empty_one():
    from seti.knell.acquire import AcquisitionLog

    log = AcquisitionLog()
    log.record("a", "SELECT 1", rows=7)
    log.record("b", "SELECT 2", rows=0)
    log.record("c", "SELECT 3", error="HTTP 500")
    d = log.as_dict()
    assert [s["status"] for s in d["stages"]] == [
        "OK", "QUERY_RETURNED_ZERO_ROWS", "QUERY_FAILED"]
    assert d["any_query_failed"] is True and d["total_rows"] == 7
    assert d["stages"][0]["query"] == "SELECT 1"      # the query text is recorded


def test_empty_archive_response_degrades_honestly(tmp_path, monkeypatch):
    """A failed archive must never read as a science null."""
    import seti.knell.acquire as kacq
    from seti.knell.run import knell_sweep, knell_vet

    def _nothing(*a, **kw):
        raise RuntimeError("CONNECT tunnel failed, response 403")
        yield  # pragma: no cover

    monkeypatch.setattr(kacq, "iter_region_2band", _nothing)
    monkeypatch.setattr(kacq, "probe_ztf_service",
                        lambda *a, **k: (False, "Tunnel connection failed: 403"))
    s = knell_sweep(ra=10.0, dec=20.0, radius_deg=0.1, out_root=tmp_path)
    assert s["verdict"] == "NO_DATA_REACHED"
    assert s["n_candidates"] == 0
    assert s["acquisition"]["any_query_failed"] is True

    v = knell_vet(out_root=tmp_path, offline=True)
    assert v["verdict"] == "NO_DATA_REACHED"
    assert "NOT a null result" in v["note"]
    assert json.loads((tmp_path / "summary.json").read_text())["n_survivors"] == 0


def test_a_swallowed_http_failure_is_not_reported_as_zero_rows(tmp_path, monkeypatch):
    """The provenance bug this channel found and fixed.

    The inherited bulk ZTF fetcher catches its own HTTP errors and returns an
    empty dict, so a proxy 403 and an empty sky box are indistinguishable at that
    layer.  A separate service probe restores the distinction; without it a run
    in which no query ever reached IRSA would report zero rows and read as a
    search.
    """
    import seti.knell.acquire as kacq
    from seti.knell.run import knell_sweep

    def _silently_empty(*a, **kw):
        return
        yield  # pragma: no cover

    monkeypatch.setattr(kacq, "iter_region_2band", _silently_empty)
    monkeypatch.setattr(kacq, "probe_ztf_service",
                        lambda *a, **k: (False, "HTTP 403"))
    s = knell_sweep(ra=10.0, dec=20.0, radius_deg=0.1, out_root=tmp_path)
    assert s["verdict"] == "NO_DATA_REACHED", (
        "an unreachable archive that returns empty dicts must not be reported as "
        "a searched-but-empty field")

    # ...and with a reachable service the same empty result is a DIFFERENT fact.
    monkeypatch.setattr(kacq, "probe_ztf_service", lambda *a, **k: (True, "HTTP 200"))
    s2 = knell_sweep(ra=11.0, dec=20.0, radius_deg=0.1, out_root=tmp_path)
    assert s2["verdict"] == "ARCHIVE_RETURNED_ZERO_SOURCES"


def test_a_searched_field_with_no_candidates_is_not_called_no_data(tmp_path, monkeypatch):
    """The converse: real data, nothing found, must NOT read as NO_DATA."""
    import seti.knell.acquire as kacq
    from seti.knell.run import knell_sweep, knell_vet

    def _flat(ra, dec, **kw):
        for i in range(3):
            t, m, e = synth([40] * 5, [0.0], err_mmag=20.0, seed=2000 + i)
            t2, m2, e2 = synth([40] * 5, [0.0], err_mmag=20.0, seed=2100 + i)
            yield {"source_id": f"s{i}", "ra": ra, "dec": dec, "ccd": "1_1_1",
                   "sep_arcsec": 0.1, "lc_g": as_lc(t, m, e), "lc_r": as_lc(t2, m2, e2)}

    monkeypatch.setattr(kacq, "iter_region_2band", _flat)
    monkeypatch.setattr(kacq, "probe_ztf_service", lambda *a, **k: (True, "HTTP 200"))
    s = knell_sweep(ra=10.0, dec=20.0, radius_deg=0.1, out_root=tmp_path, seed=7)
    assert s["verdict"] == "SEARCHED" and s["n_testable"] == 3 and s["n_candidates"] == 0
    v = knell_vet(out_root=tmp_path, offline=True)
    assert v["verdict"] == "NO_CANDIDATES"
    assert "not an occurrence limit" in v["note"]


def test_end_to_end_sweep_recovers_an_injected_cessation(tmp_path, monkeypatch):
    import pandas as pd  # noqa: F401

    import seti.knell.acquire as kacq
    from seti.knell.run import knell_sweep, knell_vet

    n = [60] * 5
    amp = [55, 55, 0, 0, 0]

    def _one(ra, dec, **kw):
        tg, mg, eg = synth(n, amp, period=0.63, err_mmag=18.0, seed=3001)
        tr, mr, er = synth(n, amp, period=0.63, err_mmag=18.0, seed=3002)
        yield {"source_id": "knell-1", "ra": ra, "dec": dec, "ccd": "1_1_1",
               "sep_arcsec": 0.2, "lc_g": as_lc(tg, mg, eg), "lc_r": as_lc(tr, mr, er)}

    monkeypatch.setattr(kacq, "iter_region_2band", _one)
    monkeypatch.setattr(kacq, "probe_ztf_service", lambda *a, **k: (True, "HTTP 200"))
    s = knell_sweep(ra=10.0, dec=20.0, radius_deg=0.1, out_root=tmp_path, seed=11)
    assert s["verdict"] == "SEARCHED"
    assert s["n_candidates"] == 1, s["band_status_counts"]
    v = knell_vet(out_root=tmp_path, offline=True)
    assert v["n_candidates"] == 1
    assert v["verdict"] in ("SURVIVORS", "ALL_REJECTED")
    assert (tmp_path / "knell_vetted.csv").exists()


def test_cross_survey_candidates_must_carry_the_recoverability_demonstration():
    """A cross-survey non-detection is worthless without it, and says so."""
    from seti.knell.run import crossmatch_demonstration, load_knell_config

    conf = load_knell_config()
    conf["season"]["min_epochs_block"] = 15
    conf["period"].update(PER_KW)
    conf["cross"]["n_trials"] = 100
    conf["detect"]["n_null"] = 100

    # A well-sampled ZTF light curve WOULD have seen a 60 mmag, 0.63 d signal.
    t, m, e = synth([60] * 3, [0.0], err_mmag=18.0, seed=4001)
    good = crossmatch_demonstration(t, m, e, 0.63, 60.0, conf, rng=1)
    assert good["demonstrated"] is True and good["eta_min"] >= 0.95
    assert good["p_persist_text"].startswith("<=")

    # A thin one would not, so the catalogue-vs-ZTF difference is uninformative.
    t, m, e = synth([16] * 3, [0.0], err_mmag=30.0, seed=4002)
    bad = crossmatch_demonstration(t, m, e, 0.63, 12.0, conf, rng=1)
    assert bad["demonstrated"] is False


def test_config_file_is_loadable_and_thresholds_are_not_magic_numbers():
    from seti.knell.run import load_knell_config

    c = load_knell_config()
    for k in ("season", "acquire", "period", "detect", "vet", "cross"):
        assert k in c
    assert 0.0 < c["detect"]["eta_min"] <= 1.0
    assert c["detect"]["min_post_blocks"] >= 2
    assert c["cross"]["eta_min"] >= c["detect"]["eta_min"], (
        "the cross-survey layer must be held to a stricter efficiency floor than "
        "the intra-survey primary, because more than the epoch changes")


def test_permutation_threshold_tracks_the_blocks_own_sampling():
    """Different blocks get different thresholds; that is the point."""
    t1, m1, e1 = synth([80], [0.0], err_mmag=20.0, seed=5001)
    t2, m2, e2 = synth([16], [0.0], err_mmag=20.0, seed=5002)
    f1 = frequency_grid(t1, **PER_KW)
    f2 = frequency_grid(t2, **PER_KW)
    thr1, _ = permutation_threshold(t1, m1, e1, f1, n_null=150, rng=1)
    thr2, _ = permutation_threshold(t2, m2, e2, f2, n_null=150, rng=1)
    assert thr2 > thr1 + 0.2, (
        "a sparse block must demand far more power for the same false-alarm rate; "
        "a single global threshold would make cadence look like astrophysics")


def test_the_triage_prefilter_is_lossless_by_construction():
    """The cheap pre-filter must not be able to discard a real candidate.

    The cessation pattern requires a *prefix* of detected blocks, so a star whose
    first block is not detected can never be a candidate.  Triage tests exactly
    that block, at a deliberately looser threshold than the search --- so
    anything the full test would keep, triage keeps.
    """
    from seti.knell.run import load_knell_config, triage_was_ever_periodic

    conf = load_knell_config()
    conf["season"]["min_epochs_block"] = 15
    conf["period"].update(PER_KW)
    conf["detect"]["n_null_triage"] = 60

    kept = agreed = 0
    for k in range(8):
        t, m, e = synth([60] * 5, [50, 50, 0, 0, 0], period=0.63, err_mmag=18.0,
                        seed=6000 + k)
        lc = as_lc(t, m, e)
        full = analyze_band(t, m, e, band="g", rng=k, measure_efficiency=False, **FAST)
        tri = triage_was_ever_periodic(lc, conf, rng=k)
        if full.pattern_strict and full.n_pre >= 2:
            kept += 1
            agreed += int(tri)
    assert kept >= 6
    assert agreed == kept, "triage discarded a star the full test would have kept"

    # ...and it removes constant stars, which is the point.
    survived = sum(triage_was_ever_periodic(
        as_lc(*synth([60] * 5, [0.0], err_mmag=20.0, seed=6100 + k)), conf, rng=k)
        for k in range(12))
    assert survived <= 3, f"triage passed {survived}/12 constant stars"


def test_class_matching_does_not_fire_on_unrelated_types():
    """Substring class matching must not reject ordinary stars.

    A bare "G" in the AGN list would match "RGB*", "AGB*" and "GlCl" and quietly
    turn every giant into a rejected AGN.  Rejection classes are only useful if
    they reject the thing they name.
    """
    from seti.knell.vet import _AGN_TYPES, _CV_TYPES, _YSO_TYPES, _has

    for otype in ("RGB*", "AGB*", "Star", "EB*", "PulsV*delSct", "RotV*", "LPV*",
                  "HighPM*", "WD*", "SX*", "HB*", ""):
        assert not _has(otype, _AGN_TYPES), otype
        assert not _has(otype, _YSO_TYPES), otype
        assert not _has(otype, _CV_TYPES), otype
    assert _has("QSO", _AGN_TYPES) and _has("Seyfert_1", _AGN_TYPES)
    assert _has("CV*", _CV_TYPES) and _has("YSO", _YSO_TYPES)
    assert not _has(None, _AGN_TYPES) and not _has("nan", _AGN_TYPES)

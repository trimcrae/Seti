"""CENOTAPH offline tests. No network, no archives, no fixtures on disk.

The suite is organised around the four things that can destroy this channel:

1. it fails to recover a real grey deficit;
2. it flags an ordinary reddening column as grey;
3. it flags a metal-poor subdwarf as grey;
4. it emits a candidate when an archive gave it nothing.

Every one of those has a test that trips it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seti.cenotaph import budget, run, synth, vet
from seti.cenotaph.extinction import (
    BANDS,
    EXCESS_BANDS,
    FIT_BANDS,
    a_over_av,
    covering_fraction_from_grey,
    grey_from_covering_fraction,
)
from seti.cenotaph.greyfit import (
    fit_grey_reddening,
    grey_significance_floor,
    minimum_detectable_f,
)
from seti.cenotaph.twins import TwinConfig, absolute_magnitude, twin_statistics

FIT = list(FIT_BANDS)


# ------------------------------------------------------------------ physics --
def test_covering_fraction_round_trip():
    for f in (0.01, 0.1, 0.2, 0.5, 0.9):
        assert grey_from_covering_fraction(f) == pytest.approx(
            -2.5 * math.log10(1 - f))
        assert covering_fraction_from_grey(grey_from_covering_fraction(f)) == \
            pytest.approx(f)
    # The values quoted in the channel documentation must be the real ones.
    assert grey_from_covering_fraction(0.1) == pytest.approx(0.1144, abs=1e-3)
    assert grey_from_covering_fraction(0.5) == pytest.approx(0.7526, abs=1e-3)


def test_extinction_lever_arm_is_what_makes_the_channel_work():
    """Grey and reddening must be near-orthogonal over the fitted bands."""
    r = [a_over_av(b) for b in FIT]
    assert max(r) / min(r) > 30, "band set has lost its chromatic lever arm"
    # W3/W4 are excluded from the fit by construction (leg 2 lives there).
    assert not set(FIT) & {e.name for e in EXCESS_BANDS}
    assert {b.name for b in BANDS} == set(FIT)


def test_wise_wien_ceilings_close_the_mid_ir_route():
    """The structural argument: deeper WISE data means *warmer* sensitivity."""
    ceil = budget.wise_temperature_ceilings()
    assert ceil["w4"] == pytest.approx(131.2, rel=0.02)
    assert ceil["w3"] == pytest.approx(250.7, rel=0.02)
    # The catalogues that got deeper after 2010 (CatWISE2020, unWISE) are
    # W1/W2 only, and those bands peak at 630-865 K -- hotter, not colder.
    assert ceil["w2"] > 600 and ceil["w1"] > 800
    assert ceil["w4"] > 100, "W4 cannot reach the sub-100 K regime at all"


def test_equilibrium_temperature_and_radius_round_trip():
    for t in (30.0, 50.0, 100.0, 300.0):
        r = budget.radius_for_temperature(1.0, t)
        assert budget.equilibrium_temperature(1.0, r) == pytest.approx(t, rel=1e-9)
    # 50 K around a solar-luminosity star is a ~60 AU structure.
    assert budget.radius_for_temperature(1.0, 50.0) == pytest.approx(62.0, rel=0.05)
    assert budget.wien_peak_um(50.0) == pytest.approx(58.0, rel=0.02)
    assert budget.wien_peak_um(100.0) == pytest.approx(29.0, rel=0.02)


def test_blackbody_normalisation_integrates_to_the_bolometric_flux():
    """π B_ν/(σT⁴) must integrate to 1 over frequency, or every flux is wrong."""
    t = 50.0
    nu = np.geomspace(1e9, 1e14, 20000)
    lam_um = budget.C_LIGHT / nu * 1e6
    shape = budget.blackbody_shape_per_hz(lam_um, t)
    assert np.trapezoid(shape, nu) == pytest.approx(1.0, rel=2e-3)


def test_cold_dyson_far_ir_flux_is_detectable_and_horizon_scales_correctly():
    """The quantitative basis for leg 3 being decisive rather than decorative."""
    f_jy = budget.predicted_flux_jy(0.1, 1.0, 100.0, 50.0, 100.0)
    assert 3.0 < f_jy < 12.0, f"expected a few Jy at 100 pc, got {f_jy}"
    h = budget.detection_horizon_pc(0.1, 1.0, 50.0, "akari90")
    assert h > 150.0
    # Inverse-square: halving f moves the horizon by sqrt(2).
    h2 = budget.detection_horizon_pc(0.05, 1.0, 50.0, "akari90")
    assert h / h2 == pytest.approx(math.sqrt(2.0), rel=1e-6)


def test_closure_ratio_separates_isotropic_occulter_from_edge_on_disk():
    """The single most diagnostic number in the channel."""
    f_dim, lsun, dpc, t = 0.20, 1.0, 120.0, 50.0
    # An isotropic occulter re-radiates the full intercepted power.
    flux = budget.predicted_flux_jy(f_dim, lsun, dpc, t, 90.0)
    good = budget.close_budget(f_dim, 0.02, lsun, dpc, t,
                               far_ir_fluxes_jy={"akari90": flux})
    assert good.verdict == "closes"
    assert good.closure_ratio == pytest.approx(1.0, rel=1e-6)

    # An edge-on disk blocks the sightline but intercepts ~2% of the solid angle.
    disk = budget.close_budget(f_dim, 0.02, lsun, dpc, t,
                               far_ir_fluxes_jy={"akari90": flux * 0.02})
    assert disk.verdict == "anisotropic_or_nonthermal"

    # Cirrus or a background galaxy in the beam over-closes the budget.
    over = budget.close_budget(f_dim, 0.02, lsun, dpc, t,
                               far_ir_fluxes_jy={"akari90": flux * 30.0})
    assert over.verdict == "over_closure"


def test_mid_ir_nondetection_excludes_an_interval_not_a_half_line():
    """A fixed-band flux is not monotonic in T; reporting 'T < Tmax' would lie."""
    ex = budget.temperature_exclusion(0.2, 1.0, 100.0, {"w4": 0.012, "w3": 0.0018})
    assert ex["excluded_lo"] is not None
    assert ex["excluded_hi"] > ex["excluded_lo"]
    # The cold side survives: that is precisely the CENOTAPH regime.
    assert ex["t_max_cold"] < ex["excluded_lo"]


def test_material_cost_of_going_cold_matches_the_published_estimate():
    """Blain (2024): a Jupiter mass at 1 mm thickness reaches ~81 AU / ~33 K."""
    r = 81.3
    m = budget.material_mass_kg(1.0, r, areal_density_kg_m2=3.0)  # 1 mm at 3 g/cc
    assert 0.5 < m / 1.898e27 < 5.0, "mass budget disagrees with Blain (2024)"
    assert budget.equilibrium_temperature(1.0, r) == pytest.approx(43.7, rel=0.1)


# ------------------------------------------------------------------ greyfit --
def _residuals(bands, grey, av, noise=0.0, rng=None):
    r = np.array([a_over_av(b) for b in bands])
    y = grey + av * r
    if noise and rng is not None:
        y = y + rng.normal(0.0, noise, len(bands))
    return y


def test_injected_grey_deficit_is_recovered():
    rng = np.random.default_rng(0)
    y = _residuals(FIT, grey=0.30, av=0.0, noise=0.01, rng=rng)
    fit = fit_grey_reddening(FIT, y, np.full(len(FIT), 0.01))
    assert fit.verdict == "ok"
    assert fit.grey_mag == pytest.approx(0.30, abs=0.03)
    assert fit.significance > 5.0
    assert fit.covering_fraction == pytest.approx(0.242, abs=0.03)


def test_injected_reddening_is_NOT_flagged_as_grey():
    """The channel's central discriminant. A dust column must give g ≈ 0."""
    rng = np.random.default_rng(1)
    y = _residuals(FIT, grey=0.0, av=0.60, noise=0.01, rng=rng)
    fit = fit_grey_reddening(FIT, y, np.full(len(FIT), 0.01))
    assert fit.av == pytest.approx(0.60, abs=0.05)
    assert abs(fit.grey_mag) < 0.05
    assert abs(fit.significance) < 3.0, "a pure reddening column was flagged grey"


def test_grey_and_reddening_are_separated_when_both_are_present():
    rng = np.random.default_rng(2)
    y = _residuals(FIT, grey=0.25, av=0.40, noise=0.01, rng=rng)
    fit = fit_grey_reddening(FIT, y, np.full(len(FIT), 0.01))
    assert fit.grey_mag == pytest.approx(0.25, abs=0.04)
    assert fit.av == pytest.approx(0.40, abs=0.06)
    assert fit.significance > 4.0


def test_distance_error_is_degenerate_with_grey_and_inflates_its_error():
    """A parallax error is *exactly* a grey offset; the fit must admit that."""
    y = _residuals(FIT, grey=0.30, av=0.25)
    sig = np.full(len(FIT), 0.01)
    tight = fit_grey_reddening(FIT, y, sig, dist_modulus_sigma=0.0)
    loose = fit_grey_reddening(FIT, y, sig, dist_modulus_sigma=0.15)
    assert loose.grey_err > 5 * tight.grey_err
    # ...but A_V is immune, because the degeneracy lies along the grey axis.
    assert loose.av_err == pytest.approx(tight.av_err, rel=0.25)
    assert loose.significance < 3.0 < tight.significance


def test_degenerate_band_set_is_refused_not_reported():
    """Ks+W1+W2 alone cannot separate grey from reddening. Say so, don't fit."""
    bands = ["ks", "w1", "w2"]
    y = _residuals(bands, grey=0.3, av=0.0)
    fit = fit_grey_reddening(bands, y, np.full(3, 0.01))
    assert fit.verdict == "degenerate_band_set"
    assert fit.lever_arm < 0.3


def test_too_few_bands_degrades_honestly():
    fit = fit_grey_reddening(["g", "ks"], [0.2, 0.2], [0.01, 0.01])
    assert fit.verdict == "too_few_bands"
    assert not math.isfinite(fit.grey_err) or fit.grey_err == float("inf")


def test_non_reddening_sed_shape_is_rejected():
    """A hot companion or a real excess is not grey + R_V = 3.1 reddening."""
    y = _residuals(FIT, grey=0.30, av=0.0)
    y[FIT.index("nuv")] -= 1.5   # UV excess from a white-dwarf companion
    y[FIT.index("fuv")] -= 2.0
    fit = fit_grey_reddening(FIT, y, np.full(len(FIT), 0.02))
    assert fit.verdict == "bad_sed_fit"


def test_negative_av_is_pinned_rather_than_stealing_flux_from_grey():
    rng = np.random.default_rng(5)
    y = _residuals(FIT, grey=0.30, av=-0.20, noise=0.005, rng=rng)
    fit = fit_grey_reddening(FIT, y, np.full(len(FIT), 0.005))
    assert fit.av_at_boundary
    assert fit.av == pytest.approx(0.0, abs=1e-9)


def test_delta_chi2_matches_the_significance_for_a_linear_model():
    rng = np.random.default_rng(6)
    y = _residuals(FIT, grey=0.22, av=0.30, noise=0.01, rng=rng)
    fit = fit_grey_reddening(FIT, y, np.full(len(FIT), 0.01))
    assert fit.delta_chi2_grey == pytest.approx(fit.significance**2, rel=0.02)


def test_sensitivity_floor_is_computed_not_asserted():
    """At poe = 20 the distance term alone is 0.109 mag -- state it honestly."""
    assert grey_significance_floor(0.0, 20.0, phot_err_mag=0.0) == \
        pytest.approx(0.1086, abs=1e-3)
    f20 = minimum_detectable_f(0.06, 20.0)
    f100 = minimum_detectable_f(0.06, 100.0)
    assert 0.2 < f20 < 0.45
    assert f100 < f20
    # The improvement over Zackrisson et al.'s f_cov > 0.75 floor is the claim.
    assert f20 < 0.75 and f100 < 0.30


# -------------------------------------------------------------------- twins --
def test_absolute_magnitude_and_distance_modulus():
    assert absolute_magnitude([10.0], [10.0])[0] == pytest.approx(5.0)
    assert np.isnan(absolute_magnitude([10.0], [-1.0])[0])


def test_metal_poor_subdwarf_does_NOT_flag():
    """The #1 astrophysical false positive, killed by construction.

    A subdwarf is genuinely ~0.5 mag underluminous relative to solar-metallicity
    dwarfs of the same Teff -- larger than the injected signal. Because [M/H] is
    a matching axis its twins are subdwarfs too, so the deficit must cancel.
    """
    df = synth.make_population(n=1500, seed=3)
    df = synth.add_subdwarfs(df, n=900, seed=4, teff_range=(5000.0, 5800.0))
    tw = twin_statistics(df, TwinConfig(n_twins=40, n_twins_min=15), verbose=False)
    is_sub = (df["label"] == "subdwarf").to_numpy()
    sub = tw[is_sub & (tw["twin_verdict"] == "ok")]
    assert len(sub) > 100, "subdwarfs did not get twin sets; test is vacuous"

    # They are genuinely, hugely underluminous relative to the solar-metallicity
    # population -- ~0.5 mag, four times the f = 0.1 signal. Confirm that first,
    # or the test proves nothing.
    raw_deficit = float(np.median(df.loc[is_sub, "m_ks_true"])
                        - np.median(df.loc[~is_sub, "m_ks_true"]))
    assert raw_deficit > 0.35, "the synthetic subdwarfs are not actually subluminous"

    # ...and the twin estimator returns them to zero, because [M/H] is a
    # matching axis so their twins are subdwarfs too.
    assert abs(float(np.median(sub["dm_twin"]))) < 0.10
    assert float((sub["z_twin"] > 3.0).mean()) < 0.05, \
        "subdwarfs are leaking into the underluminous tail"


def test_twins_recover_an_injected_grey_deficit():
    df = synth.make_population(n=2500, seed=8)
    rng = np.random.default_rng(9)
    idx = rng.choice(df.index, size=25, replace=False)
    df = synth.inject_grey(df, idx, grey_mag=0.50)
    tw = twin_statistics(df, TwinConfig(n_twins=40, n_twins_min=15), verbose=False)
    hit = tw.loc[idx]
    hit = hit[hit["twin_verdict"] == "ok"]
    assert len(hit) > 10
    assert float(np.median(hit["dm_twin"])) == pytest.approx(0.50, abs=0.10)
    assert float((hit["z_twin"] > 3.0).mean()) > 0.6


def test_twin_scatter_is_close_to_the_injected_intrinsic_scatter():
    df = synth.make_population(n=2000, seed=12, intrinsic_scatter=0.06)
    tw = twin_statistics(df, TwinConfig(n_twins=40, n_twins_min=15), verbose=False)
    ok = tw[tw["twin_verdict"] == "ok"]
    assert float(np.nanmedian(ok["twin_scatter"])) == pytest.approx(0.06, abs=0.03)


def test_stars_without_a_twin_set_are_reported_not_guessed():
    df = synth.make_population(n=40, seed=13)
    tw = twin_statistics(df, TwinConfig(n_twins=50, n_twins_min=35), verbose=False)
    assert (tw["twin_verdict"] != "ok").all()
    assert tw["dm_twin"].isna().all()


# ---------------------------------------------------------------- vetting ----
def test_binary_and_blend_screens_trip():
    n = 6
    df = pd.DataFrame({
        "ruwe": [1.0, 1.5, 1.0, 1.0, 1.0, 1.0],
        "parallax_over_error": [50.0] * n,
        "ipd_frac_multi_peak": [0, 0, 10, 0, 0, 0],
        "astrometric_excess_noise_sig": [0.5, 0.5, 0.5, 9.0, 0.5, 0.5],
        "non_single_star": [0, 0, 0, 0, 1, 0],
        "bp_rp": [0.8] * n, "phot_bp_rp_excess_factor": [1.21] * n,
        "g_mag": [13.0] * n,
        "phot_g_mean_flux_over_error": [3000.0] * n,
        "phot_g_n_obs": [200.0] * n,
        "b_gal": [45.0] * n, "pmra": [1.0] * n, "pmdec": [1.0] * n,
        "parallax": [5.0] * n, "n_beam_neighbours": [0] * n,
        "dm_nuv": [0.0, 0.0, 0.0, 0.0, 0.0, -1.2], "dm_nuv_err": [0.1] * n,
        "dm_fuv": [0.0] * n, "dm_fuv_err": [0.1] * n,
    })
    v = vet.vet_table(df)
    assert list(v["pass_binarity"]) == [True, False, False, False, False, False]
    # The hot-companion (UV excess) case is the dangerous one: it biases Teff up
    # and makes a star look underluminous. It must be caught.
    assert bool(v["uv_excess_hot_companion"].iloc[5])


def test_variable_star_caught_in_a_low_state_is_rejected():
    df = pd.DataFrame({"phot_g_mean_flux_over_error": [3000.0, 200.0],
                       "phot_g_n_obs": [200.0, 200.0]})
    v = vet.vet_table(df)
    assert bool(v["pass_constant"].iloc[0])
    assert not bool(v["pass_constant"].iloc[1])


def test_background_galaxy_confusion_budget_is_large_for_far_ir_beams():
    """The universal killer: every Hephaistos candidate died of this."""
    p_wise = vet.background_galaxy_probability(6.0)
    p_akari = vet.background_galaxy_probability(25.0)
    assert p_akari / p_wise == pytest.approx((25.0 / 6.0) ** 2, rel=0.02)
    # With 10^6 targets, chance far-IR associations run to thousands.
    n_false = vet.expected_false_matches(
        1_000_000, vet.FAR_IR_SOURCE_DENSITY_PER_SQDEG["akari"], 25.0)
    assert n_false > 1000


def test_missing_columns_are_reported_rather_than_passed():
    v = vet.vet_table(pd.DataFrame({"ruwe": [1.0]}))
    assert "parallax_over_error" in v.attrs["vet_missing"]
    assert not bool(v["pass_parallax"].iloc[0]), "an absent test must not pass"


# ------------------------------------------------------------- end to end ----
def test_full_funnel_recovers_injection_and_rejects_confounders(tmp_path):
    """Grey in, reddening out, subdwarfs out -- through the real orchestrator."""
    out = tmp_path / "cenotaph"
    out.mkdir()
    df = synth.make_population(n=2200, seed=21)
    rng = np.random.default_rng(22)
    idx_g = rng.choice(df.index, size=20, replace=False)
    df = synth.inject_grey(df, idx_g, grey_mag=0.60)
    idx_r = rng.choice(df.index.difference(idx_g), size=60, replace=False)
    df = synth.inject_reddening(df, idx_r, av=0.70)
    df = synth.add_subdwarfs(df, n=250, seed=23)
    df.to_parquet(out / "sample.parquet", index=False)

    tw = run.stage_twins(df, out, TwinConfig(n_twins=40, n_twins_min=15))
    fits = run.stage_grey(tw, out, z_min=3.0)
    assert len(fits) > 500

    got = set(fits.loc[fits["grey_sigma"] > 3.0, "source_id"])
    labels = df.set_index("source_id")["label"]
    injected = set(df.loc[idx_g, "source_id"])
    assert len(got & injected) >= 10, "injected grey occulters were not recovered"

    flagged = labels.reindex(sorted(got)).value_counts().to_dict()
    assert flagged.get("reddened", 0) == 0, "a reddening column was flagged grey"
    assert flagged.get("subdwarf", 0) == 0, "a subdwarf was flagged grey"

    farir = run.stage_farir(tw, fits, out, synthetic=True)
    summary = run.stage_reduce(tw, fits, farir, out, z_min=3.0)
    assert summary["funnel"]["n_leg1_grey_significant"] >= 10
    assert summary["sensitivity"]["min_detectable_f_poe50"] < 0.75
    assert (out / "summary.json").exists()
    json.loads((out / "summary.json").read_text())


def test_empty_archive_degrades_to_NO_DATA_REACHED(tmp_path):
    """An archive that returns nothing must produce a verdict, not a candidate."""
    out = tmp_path / "cenotaph"
    out.mkdir()
    pd.DataFrame().to_parquet(out / "sample.parquet", index=False)
    summary = run.cenotaph_run(out_dir=out, stage="reduce")
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert json.loads((out / "summary.json").read_text())["verdict"] == \
        "NO_DATA_REACHED"


# ------------------------------------------------- acquisition honesty guard --
# Run 30203250183 reported `NO_DATA_REACHED, n_sample: 0` after successfully
# pulling 703,555 rows. These tests exist so that cannot happen again silently:
# a failed query, an empty-but-valid query, a truncated query and a partial
# pull must each carry their own name.


def _fake_rows(n, start=0):
    return pd.DataFrame({"source_id": np.arange(start, start + n, dtype=np.int64),
                         "parallax": np.full(n, 2.1)})


def test_run_gaia_query_separates_failure_from_an_empty_result(monkeypatch):
    """The whole point: an exception and an empty table are different facts."""
    from seti.cenotaph import acquire

    acquire.reset_transport_state()
    monkeypatch.setattr(acquire, "GAIA_TRANSPORTS",
                        (("boom", lambda q, m: (_ for _ in ()).throw(
                            RuntimeError("connection refused")), False),))
    df, rec = acquire.run_gaia_query("SELECT 1", retries_per_transport=1,
                                     base_sleep=0.0)
    assert rec["status"] == acquire.QUERY_FAILED
    assert df.empty
    assert "connection refused" in rec["error"]
    assert rec["query"] == "SELECT 1"

    acquire.reset_transport_state()
    monkeypatch.setattr(acquire, "GAIA_TRANSPORTS",
                        (("empty", lambda q, m: pd.DataFrame(), False),))
    df, rec = acquire.run_gaia_query("SELECT 1", retries_per_transport=1,
                                     base_sleep=0.0)
    assert rec["status"] == acquire.QUERY_ZERO
    assert rec["n_rows"] == 0
    assert rec["error"] is None


def test_truncated_result_is_flagged_against_its_own_count_star(monkeypatch):
    """8193 rows where COUNT(*) says 200000 is a row cap, not astrophysics."""
    from seti.cenotaph import acquire

    acquire.reset_transport_state()
    monkeypatch.setattr(acquire, "GAIA_TRANSPORTS",
                        (("capped", lambda q, m: _fake_rows(8193), True),))
    df, rec = acquire.run_gaia_query("SELECT x", expect_rows=200000,
                                     retries_per_transport=1, base_sleep=0.0)
    assert rec["status"] == acquire.QUERY_TRUNCATED
    assert rec["n_rows"] == 8193
    assert rec["expected_rows"] == 200000


def test_a_working_transport_is_preferred_over_a_truncating_one(monkeypatch):
    """A cap on one transport must not become the answer while another works."""
    from seti.cenotaph import acquire

    acquire.reset_transport_state()
    monkeypatch.setattr(acquire, "GAIA_TRANSPORTS", (
        ("capped", lambda q, m: _fake_rows(8193), True),
        ("full", lambda q, m: _fake_rows(50000), False),
    ))
    df, rec = acquire.run_gaia_query("SELECT x", expect_rows=50000,
                                     retries_per_transport=1, base_sleep=0.0)
    assert rec["status"] == acquire.QUERY_OK
    assert rec["transport"] == "full"
    assert len(df) == 50000


def test_dead_async_transport_is_disabled_after_the_esa_500(monkeypatch):
    """The observed ESA 500 never recovers; retrying it cost 75 min of a run."""
    from seti.cenotaph import acquire

    acquire.reset_transport_state()
    calls = {"n": 0}

    def _esa_500(q, m):
        calls["n"] += 1
        raise RuntimeError(
            "Error 500:\nCannot find result 'result' for job 'abc-O'. "
            "Path does not exists: /gaia_netapp/tap-server/storage/O/anonymous/x")

    monkeypatch.setattr(acquire, "GAIA_TRANSPORTS", (
        ("astroquery_async", _esa_500, False),
        ("astroquery_sync", lambda q, m: _fake_rows(10), True),
    ))
    for _ in range(3):
        df, rec = acquire.run_gaia_query("SELECT x", retries_per_transport=3,
                                         base_sleep=0.0)
        assert rec["transport"] == "astroquery_sync"
    # One attempt on the first query, then never again in this process.
    assert calls["n"] == 1


def test_one_dead_shell_does_not_discard_the_shells_that_worked(monkeypatch):
    """The actual failure of run 30203250183, as a regression test."""
    from seti.cenotaph import acquire

    acquire.reset_transport_state()
    state = {"n": 0}

    def _chunk(plx_lo, plx_hi, w, top, rlo, rhi, depth, max_depth, ledger, **kw):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("Error 408: Job timeout/aborted.")
        ledger.append({"chunk": f"shell[{plx_lo:g},{plx_hi:g})",
                       "status": acquire.QUERY_OK, "n_rows": 100,
                       "kind": "shell_total",
                       "expected_rows": 100, "depth": 0})
        return _fake_rows(100, start=state["n"] * 1000)

    monkeypatch.setattr(acquire, "_fetch_shell_chunk", _chunk)
    ledger = []
    df = acquire.fetch_gspspec_sample(plx_min_mas=5.0, checkpoint_dir=None,
                                      ledger_out=ledger)
    assert len(df) > 0, "a dead shell must not take the good shells with it"
    acq = acquire.summarise_acquisition(ledger)
    assert acq["acquisition_verdict"] == "PARTIAL_SAMPLE"
    assert acq["n_chunks_failed"] == 1
    assert any("408" in (f.get("error") or "") for f in acq["failures"])


def test_shell_is_presplit_from_its_count_not_halved_from_a_doomed_query():
    """Pre-splitting is a time fix as well as a correctness one.

    The probe measured COUNT(*) = 199,572 for the [2, 2.5) mas shell while the
    sync endpoint cut the response at 16,385 rows on a 60 s execution limit.
    Recursively halving from 200k would spend a doomed attempt at every level;
    planning the split from the count issues only queries that can finish.
    """
    from seti.cenotaph.acquire import (
        _GAIA_RANDOM_INDEX_MAX,
        TARGET_CHUNK_ROWS,
        plan_random_index_slices,
    )

    sl = plan_random_index_slices(199_572, target=15_000)
    assert len(sl) == 14, "199572/15000 rounds up to 14 slices"
    assert sl[0][0] == 0
    # The last slice must stay open-ended: random_index's upper bound is an
    # assumption, and an assumption must not be able to drop the catalogue tail.
    assert sl[-1][1] is None
    # Contiguous, no gaps — a gap is silently lost stars.
    for (_a1, b1), (a2, _) in zip(sl, sl[1:], strict=False):
        assert b1 == a2
    assert sl[-1][0] < _GAIA_RANDOM_INDEX_MAX

    # A shell that already fits is issued as one unsliced query.
    assert plan_random_index_slices(500, target=TARGET_CHUNK_ROWS) == [(None, None)]


def test_shell_totals_are_not_double_counted_by_the_slice_entries():
    """Completeness is the number that says whether the sample is whole."""
    from seti.cenotaph.acquire import QUERY_OK, summarise_acquisition

    ledger = [
        {"kind": "slice", "status": QUERY_OK, "n_rows": 15000,
         "expected_rows": 15000},
        {"kind": "slice", "status": QUERY_OK, "n_rows": 5000,
         "expected_rows": 5000},
        {"kind": "shell_total", "status": QUERY_OK, "n_rows": 20000,
         "expected_rows": 20000},
    ]
    acq = summarise_acquisition(ledger)
    assert acq["n_rows_returned"] == 20000, "slices must not be added to the total"
    assert acq["completeness"] == 1.0


def test_acquisition_verdicts_are_distinct_not_merged():
    from seti.cenotaph.acquire import (
        QUERY_FAILED,
        QUERY_OK,
        QUERY_TRUNCATED,
        QUERY_ZERO,
        summarise_acquisition,
    )

    assert summarise_acquisition(
        [{"status": QUERY_FAILED, "n_rows": 0}] * 3
    )["acquisition_verdict"] == "NO_DATA_REACHED"
    # Reached the archive, valid ADQL, cuts matched nothing. NOT the same fact.
    assert summarise_acquisition(
        [{"status": QUERY_ZERO, "n_rows": 0, "expected_rows": 0}] * 3
    )["acquisition_verdict"] == "QUERY_RETURNED_ZERO_ROWS"
    assert summarise_acquisition(
        [{"status": QUERY_OK, "n_rows": 10, "expected_rows": 10}]
    )["acquisition_verdict"] == "COMPLETE"
    assert summarise_acquisition(
        [{"status": QUERY_OK, "n_rows": 10, "expected_rows": 10},
         {"status": QUERY_TRUNCATED, "n_rows": 5, "expected_rows": 90}]
    )["acquisition_verdict"] == "PARTIAL_SAMPLE"


def test_summary_names_zero_rows_differently_from_an_unreached_archive(tmp_path):
    """summary.json must never call a valid empty result 'NO_DATA_REACHED'."""
    out = tmp_path / "cenotaph"
    out.mkdir()
    pd.DataFrame().to_parquet(out / "sample.parquet", index=False)
    (out / "sample_meta.json").write_text(json.dumps({
        "mode": "archive", "n": 0, "verdict": "QUERY_RETURNED_ZERO_ROWS",
        "acquisition": {"acquisition_verdict": "QUERY_RETURNED_ZERO_ROWS",
                        "n_chunks": 4, "n_chunks_zero_rows": 4,
                        "n_chunks_failed": 0, "n_rows_returned": 0,
                        "chunks": [{"chunk": "shell[2,2.5)", "status":
                                    "QUERY_RETURNED_ZERO_ROWS", "n_rows": 0,
                                    "query": "SELECT ... WHERE ..."}]},
    }))
    summary = run.cenotaph_run(out_dir=out, stage="reduce")
    assert summary["verdict"] == "QUERY_RETURNED_ZERO_ROWS"
    written = json.loads((out / "summary.json").read_text())
    assert written["verdict"] == "QUERY_RETURNED_ZERO_ROWS"
    # The query text and the per-stage row counts must be in the report.
    assert written["acquisition"]["chunks"][0]["query"]
    assert written["acquisition"]["n_chunks"] == 4


def test_shell_predicate_is_shared_by_count_and_select():
    """If the ruler and the query disagree, the truncation guard is worthless."""
    from seti.cenotaph.acquire import _GSPSPEC_FROM, _GSPSPEC_SELECT, _shell_where

    w = {"poe_min": 20.0, "ruwe_max": 1.4, "logg_min": 3.8,
         "teff_lo": 4000.0, "teff_hi": 7000.0}
    where = _shell_where(2.0, 2.5, w)
    count_q = f"SELECT COUNT(*) AS n {_GSPSPEC_FROM}{where}"
    select_q = f"SELECT TOP 10{_GSPSPEC_SELECT}{_GSPSPEC_FROM}{where}"
    assert where in count_q and where in select_q
    # Units and cut senses, spelled out because a flipped sign returns zero
    # rows without erroring: dwarfs are HIGH log g, parallax is in mas.
    assert "logg_gspspec > 3.8" in where
    assert "teff_gspspec BETWEEN 4000.0 AND 7000.0" in where
    assert "g.parallax >= 2.0 AND g.parallax < 2.5" in where
    assert "ruwe < 1.4" in where
    # random_index slicing must be absent unless a sub-chunk asked for it.
    assert "random_index" not in where
    assert "random_index >= 0" in _shell_where(2.0, 2.5, w, 0, 100)


def test_far_ir_non_detection_beyond_the_horizon_makes_no_claim():
    """Honest degradation: outside the horizon a non-detection means nothing."""
    far = budget.close_budget(0.1, 0.01, 1.0, 5000.0, 50.0,
                              far_ir_fluxes_jy={"akari90": float("nan")})
    assert far.verdict == "far_ir_undecidable"
    assert not far.decidable
    near = budget.close_budget(0.3, 0.01, 1.0, 60.0, 50.0,
                               far_ir_fluxes_jy={"akari90": float("nan")})
    assert near.decidable
    assert near.verdict == "anisotropic_or_nonthermal"


def test_synthetic_end_to_end_run_writes_a_summary(tmp_path):
    summary = run.cenotaph_run(out_dir=tmp_path / "c", synthetic=True,
                               n_synth=1200, seed=31)
    assert summary["verdict"] in {"no_candidates",
                                  "leg1_leg2_survivors_far_ir_pending",
                                  "candidates_with_energy_closure"}
    assert summary["funnel"]["n_sample"] > 1000
    assert summary["wise_wien_ceilings_k"]["w4"] < 140

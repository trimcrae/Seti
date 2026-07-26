"""Offline tests for TIDEMARK --- the spatial-structure test on anomaly populations.

The CI gate for this channel is unusual in that the *null* tests matter more than
the recovery test.  A spatial rate statistic that fires on an injected front but
also fires on the Galactic density gradient, the magnitude limit, or the survey
footprint is worse than useless --- it manufactures technosignature detections
out of survey design.  So every synthetic parent here carries the real
confounders switched on (exponential disk, distance-dependent detectability,
extinction in the plane, radial metallicity gradient, age--velocity dispersion),
and the null tests assert that the detectors stay quiet.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.tidemark import ingest
from seti.tidemark.agerate import age_proxies, age_rate_test
from seti.tidemark.edge import edge_scan_1d, edge_scan_cap, edge_scan_shell3d, step_score
from seti.tidemark.gradient import gradient_test, poisson_glm, rate_profile
from seti.tidemark.inject import (
    inject_age_dependence,
    inject_bubble,
    inject_gradient,
    inject_none,
    inject_selection_artifact,
    synthetic_parent,
)
from seti.tidemark.nulls import MatchedNull, empirical_p

STRICT = ["phot_g_mean_mag", "dist_pc", "bp_rp", "ebv", "log_local_density", "n_obs"]
PERMISSIVE = ["phot_g_mean_mag", "bp_rp", "ebv", "log_local_density"]


def _parent(n=45000, seed=7, footprint="all"):
    return synthetic_parent(n=n, seed=seed, footprint=footprint)


@pytest.fixture(scope="module")
def parent():
    return _parent()


# --- the matched null itself -------------------------------------------------
def test_weights_sum_to_the_anomaly_count(parent):
    """The selection weight is a redistribution, never an invention: the
    expected counts must sum to exactly the number of anomalies observed."""
    m = inject_none(parent, base_rate=0.02, seed=11)
    null = MatchedNull(parent, m, STRICT, seed=1)
    assert np.isclose(null.weights.sum(), m.sum())
    # And it partitions correctly over any region.
    hi = parent["R_gal_kpc"].to_numpy(float) > np.nanmedian(parent["R_gal_kpc"])
    assert np.isclose(null.expected(hi) + null.expected(~hi), m.sum())


def test_null_refuses_to_match_on_the_tested_coordinate(parent):
    """Stratifying on the coordinate under test would cancel the signal by
    construction.  That must be an error, not a silent null result."""
    m = inject_none(parent, base_rate=0.02, seed=11)
    with pytest.raises(ValueError, match="under test"):
        MatchedNull(parent, m, ["phot_g_mean_mag", "R_gal_kpc"], seed=1)
    with pytest.raises(KeyError):
        MatchedNull(parent, m, ["no_such_column"], seed=1)


def test_strata_meet_the_minimum_pool_size(parent):
    """Thin strata are collapsed onto coarser stratifications until they can
    actually be resampled from."""
    m = inject_none(parent, base_rate=0.02, seed=11)
    null = MatchedNull(parent, m, STRICT, min_pool=25, seed=1)
    d = null.diagnostics()
    assert d.min_pool >= 25
    assert d.frac_anom_in_thin_strata == 0.0
    # Covariate balance: the matched null must reproduce the anomaly set's
    # detectability distribution.
    for col, bal in d.balance.items():
        assert abs(bal["std_diff"]) < 0.25, f"{col} is not balanced: {bal}"


def test_draws_reproduce_the_covariate_distribution(parent):
    m = inject_selection_artifact(parent, base_rate=0.02, seed=14, strength=2.5)
    null = MatchedNull(parent, m, STRICT, seed=1)
    g = parent["phot_g_mean_mag"].to_numpy(float)
    obs = float(np.mean(g[m]))
    drawn = np.array([float(np.mean(g[idx])) for idx in null.draws(50)])
    assert abs(obs - drawn.mean()) < 3.0 * (drawn.std() + 1e-6)


def test_empirical_p_never_returns_zero():
    assert empirical_p(10.0, np.zeros(100), tail="greater") == pytest.approx(1 / 101)
    assert np.isnan(empirical_p(np.nan, np.zeros(10)))


def test_poisson_glm_recovers_a_known_rate_slope():
    x = np.linspace(0.0, 10.0, 40)
    expected = np.full_like(x, 200.0)
    rng = np.random.default_rng(3)
    counts = rng.poisson(expected * np.exp(-1.2 + 0.15 * x)).astype(float)
    theta, cov, conv = poisson_glm(counts, expected, np.stack([np.ones_like(x), x], 1))
    assert conv
    assert theta[1] == pytest.approx(0.15, abs=0.02)
    assert np.sqrt(cov[1, 1]) < 0.02


def test_step_score_is_signed_and_zero_for_a_flat_rate():
    assert step_score(50.0, 50.0, 50.0, 50.0) == pytest.approx(0.0, abs=1e-9)
    assert step_score(90.0, 50.0, 20.0, 50.0) > 3.0        # inner over-dense
    assert step_score(20.0, 50.0, 90.0, 50.0) < -3.0       # outer over-dense
    # A window with no expectation carries no information and must not score.
    assert step_score(5.0, 0.1, 50.0, 50.0, min_expected=1.0) == 0.0


# --- REQUIRED: clean null on a pure Galactic density gradient ---------------
def test_clean_null_on_pure_galactic_density_gradient(parent):
    """The stellar density falls exponentially with Galactocentric radius and
    with |z|, and the magnitude limit removes distant stars.  With the anomaly
    *rate* constant, every statistic must stay quiet."""
    m = inject_none(parent, base_rate=0.02, seed=11)
    null = MatchedNull(parent, m, STRICT, seed=1)
    for coord in ("R_gal_kpc", "abs_z_gal_kpc"):
        g = gradient_test(parent[coord].to_numpy(float), null, name=coord,
                          n_bins=10, n_null=250)
        assert g["headline_p"] > 0.05, f"{coord} gradient fired on a pure density gradient"
        e = edge_scan_1d(parent[coord].to_numpy(float), null, name=coord,
                         n_bins=20, n_null=200)
        assert e["p_value"] > 0.05, f"{coord} edge fired on a pure density gradient"
    d = gradient_test(parent["l_deg"].to_numpy(float), null, name="l_deg",
                      n_bins=12, n_null=250, periodic=True)
    assert d["dipole"]["p_value"] > 0.05


def test_clean_null_on_a_hard_detectability_artifact(parent):
    """Anomaly probability driven *only* by apparent magnitude and extinction.
    Both track distance, so the raw sky map has a strong apparent gradient with
    no intrinsic structure whatsoever."""
    m = inject_selection_artifact(parent, base_rate=0.02, seed=14, strength=2.5)
    null = MatchedNull(parent, m, STRICT, seed=1)
    for coord in ("R_gal_kpc", "abs_z_gal_kpc"):
        g = gradient_test(parent[coord].to_numpy(float), null, name=coord,
                          n_bins=10, n_null=250)
        assert g["headline_p"] > 0.05, f"{coord}: selection artifact leaked through"


def test_uncorrected_rate_would_have_been_fooled(parent):
    """The correction is load-bearing: without it the same artifact *does* show a
    spurious gradient.  If this test ever fails, the confounder has stopped being
    a confounder and the null tests above have become vacuous."""
    m = inject_selection_artifact(parent, base_rate=0.02, seed=14, strength=3.5)
    x = parent["R_gal_kpc"].to_numpy(float)
    edges = np.quantile(x, np.linspace(0, 1, 9))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 7)
    raw_rate = np.array([m[b == i].mean() for i in range(8)])
    assert raw_rate.max() / max(raw_rate.min(), 1e-9) > 1.3, (
        "the synthetic confounder is too weak to make the null tests meaningful")


# --- REQUIRED: clean null on a footprint-shaped selection artifact ----------
def test_clean_null_on_a_footprint_shaped_selection(parent):
    """A high-Galactic-latitude footprint (the classic extragalactic survey
    shape) carves a hole out of the plane.  With a constant rate, the |z| and
    radial statistics and both edge scans must stay quiet."""
    p = _parent(n=40000, seed=9, footprint="high_lat")
    m = inject_selection_artifact(p, base_rate=0.025, seed=14, strength=2.5)
    null = MatchedNull(p, m, STRICT, seed=1)
    for coord in ("R_gal_kpc", "abs_z_gal_kpc"):
        g = gradient_test(p[coord].to_numpy(float), null, name=coord, n_bins=10,
                          n_null=250)
        assert g["headline_p"] > 0.05, f"footprint faked a {coord} gradient"
        e = edge_scan_1d(p[coord].to_numpy(float), null, name=coord, n_bins=20,
                         n_null=200)
        assert e["p_value"] > 0.05, f"footprint faked a {coord} edge"
    cap = edge_scan_cap(p["l_deg"].to_numpy(float), p["b_deg"].to_numpy(float),
                        null, n_directions=48, n_null=150)
    assert cap["p_value"] > 0.05, "footprint boundary was reported as a sky edge"
    xyz = p[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
    sh = edge_scan_shell3d(xyz, null, n_per_axis=3, n_bins=18, n_null=150,
                           smooth_coords={"R_gal_kpc": p["R_gal_kpc"].to_numpy(float)})
    assert sh["p_value"] > 0.05, "footprint faked a 3D shell edge"


def test_stripe_footprint_also_returns_a_clean_null():
    p = _parent(n=35000, seed=21, footprint="stripe")
    m = inject_selection_artifact(p, base_rate=0.03, seed=5, strength=2.0)
    null = MatchedNull(p, m, STRICT, seed=1)
    g = gradient_test(p["R_gal_kpc"].to_numpy(float), null, name="R_gal_kpc",
                      n_bins=8, n_null=250)
    assert g["headline_p"] > 0.05


# --- REQUIRED: recover an injected sharp-edged bubble -----------------------
def test_recovers_an_injected_sharp_edged_bubble(parent):
    """The headline capability: a colonised volume with a boundary, injected on
    top of the full confounder stack, must be found *and localised*."""
    true_centre = np.array([600.0, -400.0, 0.0])
    m = inject_bubble(parent, centre_pc=tuple(true_centre), radius_pc=900.0,
                      contrast=5.0, base_rate=0.02, seed=12)
    null = MatchedNull(parent, m, STRICT, seed=1)
    xyz = parent[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
    sh = edge_scan_shell3d(xyz, null, n_per_axis=4, n_bins=20, n_null=300,
                           smooth_coords={"R_gal_kpc": parent["R_gal_kpc"].to_numpy(float)})
    assert sh["significant"], f"missed an injected bubble: {sh['p_value']}"
    assert sh["max_abs_score"] > 4.0
    best = sh["best_shell"]
    assert best["rho_inside"] > best["rho_outside"], "step recovered with the wrong sign"
    off = float(np.linalg.norm(np.array(best["centre"]) - true_centre))
    assert off < 600.0, f"bubble centre mislocated by {off:.0f} pc"
    assert 500.0 < best["radius"] < 1400.0, f"bubble radius {best['radius']:.0f} pc"


def test_bubble_is_not_recovered_when_it_is_not_there(parent):
    """Same detector, same parent, no bubble: the 3D scan's enormous trials
    factor must be paid for, not exploited."""
    m = inject_none(parent, base_rate=0.02, seed=33)
    null = MatchedNull(parent, m, STRICT, seed=1)
    xyz = parent[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
    sh = edge_scan_shell3d(xyz, null, n_per_axis=4, n_bins=20, n_null=250,
                           smooth_coords={"R_gal_kpc": parent["R_gal_kpc"].to_numpy(float)})
    assert sh["p_value"] > 0.05, "3D shell scan fired on a structureless field"


# --- gradient: recovery, direction, and the smooth/edge separation ----------
def test_recovers_an_injected_radial_gradient_with_the_right_sign(parent):
    m = inject_gradient(parent, coord="R_gal_kpc", slope_ln_per_unit=0.6,
                        base_rate=0.02, seed=13)
    null = MatchedNull(parent, m, STRICT, seed=1)
    g = gradient_test(parent["R_gal_kpc"].to_numpy(float), null, name="R_gal_kpc",
                      n_bins=10, n_null=300)
    assert g["headline_p"] < 0.01
    assert g["slope_ln_per_unit"] > 0
    assert g["direction"] == "increasing_outward"
    # Conservative by construction: matching on covariates that correlate with
    # position absorbs part of the signal.  Recovery must be partial but real.
    assert 0.15 < g["slope_ln_per_unit"] < 0.6


def test_inward_gradient_is_reported_as_inward(parent):
    """The two headline predictions differ only in sign, so the sign must be
    right --- an outward-biased estimator would 'confirm' Cirkovic & Bradbury
    from a Wright et al. universe."""
    m = inject_gradient(parent, coord="R_gal_kpc", slope_ln_per_unit=-0.6,
                        base_rate=0.02, seed=17)
    null = MatchedNull(parent, m, STRICT, seed=1)
    g = gradient_test(parent["R_gal_kpc"].to_numpy(float), null, name="R_gal_kpc",
                      n_bins=10, n_null=300)
    assert g["headline_p"] < 0.01
    assert g["direction"] == "decreasing_outward"


def test_a_smooth_gradient_is_not_reported_as_an_edge(parent):
    """The separation the edge test exists to make: a strong monotone trend must
    not be scored as a sharp step.

    Asserted as a *false-positive rate over realisations*, not on one seed --- a
    single draw of a steep gradient will occasionally contain a step-like
    fluctuation, and the meaningful guarantee is that this happens no more often
    than the nominal level.  (Measured over 20 realisations: 5% at smooth_order 3,
    versus 15% at order 2 --- which is why the default is 3.)
    """
    x = parent["R_gal_kpc"].to_numpy(float)
    fired_edge, detected_gradient = 0, 0
    seeds = (13, 23, 33, 43, 53, 63, 73, 83)
    for sd in seeds:
        m = inject_gradient(parent, coord="R_gal_kpc", slope_ln_per_unit=0.8,
                            base_rate=0.02, seed=sd)
        null = MatchedNull(parent, m, STRICT, seed=1)
        g = gradient_test(x, null, name="R_gal_kpc", n_bins=10, n_null=200)
        detected_gradient += int(g["headline_p"] < 0.05)
        e = edge_scan_1d(x, null, name="R_gal_kpc", n_bins=20, n_null=200)
        fired_edge += int(e["p_value"] < 0.05)
    assert detected_gradient == len(seeds), "the gradient itself should always be detected"
    assert fired_edge <= 2, (
        f"edge detector fired on {fired_edge}/{len(seeds)} pure smooth gradients")


def test_rate_profile_is_flat_under_the_null(parent):
    m = inject_none(parent, base_rate=0.02, seed=11)
    null = MatchedNull(parent, m, STRICT, seed=1)
    prof = rate_profile(parent["R_gal_kpc"].to_numpy(float), null, n_bins=8)
    rho = np.array(prof["rho"], float)
    assert np.isfinite(rho).all()
    assert abs(np.nanmean(rho) - 1.0) < 0.1
    assert np.isclose(np.nansum(prof["n_expected"]), m.sum(), rtol=1e-6)


def test_longitude_dipole_recovers_a_directional_excess(parent):
    """A bubble offset from the Sun makes a longitude dipole; its phase should
    point at the offset."""
    m = inject_bubble(parent, centre_pc=(1200.0, 0.0, 0.0), radius_pc=1100.0,
                      contrast=5.0, base_rate=0.02, seed=19)
    null = MatchedNull(parent, m, STRICT, seed=1)
    d = gradient_test(parent["l_deg"].to_numpy(float), null, name="l_deg",
                      n_bins=12, n_null=300, periodic=True)
    assert d["dipole"]["p_value"] < 0.05
    phase = d["dipole"]["phase_deg"]
    assert min(phase, 360.0 - phase) < 60.0, f"dipole points at l={phase:.0f}, expected ~0"


# --- REQUIRED: recover an injected age-dependent rate -----------------------
def test_recovers_an_injected_age_dependent_rate(parent):
    p = age_proxies(parent)
    m = inject_age_dependence(parent, base_rate=0.02, slope_per_gyr=0.30, seed=15,
                              detect_strength=0.0)
    res = age_rate_test(p, m, covariates=PERMISSIVE, n_bins=8, n_null=200)
    assert res["verdict"] == "OK"
    assert res["shape"] in ("rising", "saturating"), res["shape"]
    assert res["shape_p_value"] < 0.05
    prof = res["variants"]["metallicity_matched"]["profile"]["rho"]
    assert np.nanmean(prof[-2:]) > np.nanmean(prof[:2]), "rate should rise with age"


def test_age_test_is_quiet_when_rate_does_not_depend_on_age(parent):
    p = age_proxies(parent)
    m = inject_none(parent, base_rate=0.02, seed=11)
    res = age_rate_test(p, m, covariates=PERMISSIVE, n_bins=8, n_null=200)
    assert res["shape"] == "flat"
    assert res["shape_p_value"] > 0.05


def test_age_test_says_so_when_there_is_no_age_proxy(parent):
    bare = parent.drop(columns=[c for c in ("pmra", "pmdec", "radial_velocity",
                                            "true_age_gyr") if c in parent.columns])
    p = age_proxies(bare)
    m = inject_none(parent, base_rate=0.02, seed=11)
    res = age_rate_test(p, m, covariates=PERMISSIVE)
    assert res["verdict"] == "NO_AGE_PROXY"
    assert res["insufficient"]


# --- REQUIRED: honest degradation -------------------------------------------
def test_empty_anomaly_catalogue_degrades_honestly(parent):
    cat = ingest.from_frames("empty_axis", parent, mask=np.zeros(len(parent), bool))
    assert cat.verdict == ingest.EMPTY_ANOMALY_SET
    assert not cat.usable


def test_missing_parent_sample_is_refused_not_faked(parent):
    """A bare candidate list has no denominator.  Inventing one would be
    fabricating data, so the channel must decline."""
    cat = ingest.from_frames("candidates_only", parent.head(200),
                             mask=np.ones(200, bool))
    assert cat.verdict == ingest.NO_PARENT_SAMPLE
    assert "candidate list" in " ".join(cat.notes)


def test_no_data_reached_on_an_empty_frame():
    cat = ingest.from_frames("nothing", pd.DataFrame(columns=["ra", "dec", "parallax"]),
                             mask=np.zeros(0, bool))
    assert cat.verdict == ingest.NO_DATA_REACHED


def test_too_few_anomalies_is_reported_not_tested(parent):
    mask = np.zeros(len(parent), bool)
    mask[:5] = True
    cat = ingest.from_frames("thin_axis", parent, mask=mask)
    assert cat.verdict == ingest.INSUFFICIENT_ANOMALIES
    from seti.tidemark.run import analyse_catalogue
    res = analyse_catalogue(cat, quick=True)
    assert res["tested"] is False
    assert res["verdict"] == ingest.INSUFFICIENT_ANOMALIES
    assert "p_values" not in res


def test_catalogue_without_positions_is_refused(parent):
    """Channels like `derelict` (solar-system bodies) and `midden` (named stars)
    carry no sky position; a spatial test must decline rather than guess."""
    bare = parent[["phot_g_mean_mag", "bp_rp"]].copy()
    bare["source_id"] = np.arange(len(bare))
    mask = np.zeros(len(bare), bool)
    mask[:400] = True
    cat = ingest.AnomalyCatalogue(name="no_sky", parent=bare, anomaly_mask=mask)
    cat.validate()
    assert cat.verdict == ingest.NO_POSITIONS


def test_missing_parent_file_produces_a_verdict_not_a_crash(tmp_path):
    spec = {"parent": "results/nonexistent/parent.csv",
            "candidates": "results/nonexistent/cands.csv"}
    cat = ingest.load_channel(tmp_path, "ghost", spec)
    assert cat.verdict == ingest.NO_PARENT_SAMPLE
    assert "publishes candidates but not the population" in " ".join(cat.notes)


# --- the ingest interface ---------------------------------------------------
def test_load_channel_reads_a_sharded_layout(tmp_path, parent):
    """Channels shard per field (``f*``/``fp*`` directories).  The adapter must
    glob-and-concatenate, dedupe, and build the mask by id join."""
    p = parent.head(3000).copy()
    for i, chunk in enumerate([p.iloc[a:b] for a, b in ((0, 1000), (1000, 2000), (2000, 3000))]):
        d = tmp_path / "results" / "demo" / f"f{i}"
        d.mkdir(parents=True)
        chunk.to_csv(d / "parent.csv", index=False)
        chunk.head(40).to_csv(d / "candidates.csv", index=False)
    cat = ingest.load_channel(tmp_path, "demo", {
        "parent": "results/demo/f*/parent.csv",
        "candidates": "results/demo/f*/candidates.csv",
        "id_col": "source_id"})
    assert cat.verdict == ingest.OK
    assert cat.n_parent == 3000
    assert cat.n_anomaly == 120
    assert "R_gal_kpc" in cat.parent.columns and "l_deg" in cat.parent.columns


def test_load_channel_supports_a_score_threshold(tmp_path, parent):
    p = parent.head(4000).copy()
    rng = np.random.default_rng(2)
    p["ir_excess_z"] = rng.normal(0, 1, len(p))
    d = tmp_path / "results" / "scored"
    d.mkdir(parents=True)
    p.to_csv(d / "parent.csv", index=False)
    cat = ingest.load_channel(tmp_path, "scored", {
        "parent": "results/scored/parent.csv", "score_col": "ir_excess_z",
        "score_min": 1.5})
    assert cat.verdict == ingest.OK
    assert cat.n_anomaly == int((p["ir_excess_z"] >= 1.5).sum())
    assert cat.score is not None


def test_union_matches_on_channel_coverage(parent):
    """A star searched by more channels has more chances to be flagged, so the
    union test must match on coverage or it just maps which sky each channel
    happened to cover."""
    a = parent.head(6000).copy()
    b = parent.tail(6000).copy()
    ca = ingest.from_frames("a", a, mask=inject_none(a, base_rate=0.03, seed=1))
    cb = ingest.from_frames("b", b, mask=inject_none(b, base_rate=0.03, seed=2))
    u = ingest.union_catalogue([ca, cb])
    assert u.verdict in (ingest.OK, ingest.INSUFFICIENT_ANOMALIES)
    assert "n_channels_searched" in u.parent.columns
    assert "n_channels_searched" in u.covariates
    assert u.n_parent <= len(a) + len(b)


def test_galactic_frame_puts_the_sun_at_the_right_radius():
    from seti.galactic.orbits import R0_KPC
    df = pd.DataFrame({"ra": [0.0, 90.0, 180.0], "dec": [0.0, 30.0, -45.0],
                       "parallax": [100.0, 100.0, 100.0]})
    out = ingest.add_galactic_frame(df)
    assert np.allclose(out["dist_pc"], 10.0)
    assert np.allclose(out["R_gal_kpc"], R0_KPC, atol=0.02)
    assert set(["l_deg", "b_deg", "z_gal_kpc", "abs_z_gal_kpc"]) <= set(out.columns)


# --- end to end -------------------------------------------------------------
def test_end_to_end_run_writes_a_summary(tmp_path, parent):
    from seti.config import load_config
    from seti.tidemark.run import tidemark_run
    cfg = load_config()
    cfg.root = tmp_path
    m = inject_bubble(parent, centre_pc=(600.0, -400.0, 0.0), radius_pc=900.0,
                      contrast=5.0, base_rate=0.02, seed=12)
    cat = ingest.from_frames("synthetic_bubble", parent, mask=m)
    out = tidemark_run(cfg, catalogues=[cat], quick=True, out_dir=tmp_path / "res",
                       do_calibrate=False)
    assert out["verdict"] in ("DETECTION", "CLEAN_NULL")
    assert out["n_tested"] == 1
    assert "predictions_discriminated" in out
    written = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert written["catalogue_verdicts"]["synthetic_bubble"] == ingest.OK
    per = json.loads((tmp_path / "res" / "synthetic_bubble" / "summary.json").read_text())
    assert per["p_values"]["edge:shell_3d"] <= 0.05
    weights = pd.read_csv(tmp_path / "res" / "synthetic_bubble" / "selection_weights.csv")
    assert "selection_weight" in weights.columns and len(weights) > 0


def test_run_reports_no_testable_catalogue_when_nothing_has_a_parent(tmp_path):
    from seti.config import load_config
    from seti.tidemark.run import tidemark_run
    cfg = load_config()
    cfg.root = tmp_path
    cat = ingest.from_frames("nothing", pd.DataFrame(columns=["ra", "dec", "parallax"]),
                             mask=np.zeros(0, bool))
    out = tidemark_run(cfg, catalogues=[cat], quick=True, out_dir=tmp_path / "res")
    assert out["verdict"] == "NO_TESTABLE_CATALOGUE"
    assert out["n_tested"] == 0


def test_gradient_transfer_is_measured_and_positive(parent):
    """The run must quantify how much of a real gradient it would have absorbed;
    an uncalibrated amplitude is an unstated systematic."""
    from seti.tidemark.run import gradient_transfer
    m = inject_none(parent, base_rate=0.02, seed=11)
    cat = ingest.from_frames("calib", parent, mask=m)
    tr = gradient_transfer(cat, slopes=(0.0, 0.6), quick=True)
    assert tr["performed"]
    assert tr["transfer_coefficient"] is not None
    assert 0.1 < tr["transfer_coefficient"] < 1.2


# --- acquisition wiring (offline: geometry only) ----------------------------
def test_cone_grid_covers_both_hemispheres_and_all_longitudes():
    from seti.tidemark.acquire import cone_grid, galactic_to_icrs
    g = cone_grid("sparse")
    assert len(g) >= 20
    assert (g["b_centre"] > 0).any() and (g["b_centre"] < 0).any()
    assert g["l_centre"].max() - g["l_centre"].min() > 300
    ra, dec = galactic_to_icrs(0.0, 0.0)          # Galactic centre
    assert float(ra) == pytest.approx(266.405, abs=0.01)
    assert float(dec) == pytest.approx(-28.936, abs=0.01)


def test_excess_axis_fits_one_global_locus_not_one_per_cone():
    """The single most important line in ``acquire``: fitting the excess locus
    per cone would normalise every field to its own median and delete exactly the
    field-to-field rate differences TIDEMARK exists to measure.  A cone-wide
    offset in W1-W2 must survive into ``ir_excess_z``; fitted per cone it would
    vanish identically.
    """
    from seti.tidemark.acquire import excess_axis
    rng = np.random.default_rng(4)
    n = 5000
    cone = (rng.uniform(0, 1, n) < 0.2).astype(int)      # 20% of sky is offset
    df = pd.DataFrame({
        "source_id": np.arange(n), "cone": cone,
        "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-40, 40, n),
        "parallax": rng.uniform(2, 8, n), "bp_rp": rng.uniform(0.5, 2.5, n),
        "w1mpro": np.zeros(n),
        "w2mpro": -np.where(cone == 1, 0.35, 0.0) + rng.normal(0, 0.05, n),
    })
    out = excess_axis(df)
    z0 = float(np.nanmedian(out.loc[out["cone"] == 0, "ir_excess_z"]))
    z1 = float(np.nanmedian(out.loc[out["cone"] == 1, "ir_excess_z"]))
    assert z1 - z0 > 3.0, "the cone-to-cone offset was normalised away"
    # The counterfactual: the same statistic fitted per cone erases it entirely.
    per_cone = pd.concat([excess_axis(g) for _, g in df.groupby("cone")])
    d0 = float(np.nanmedian(per_cone.loc[per_cone["cone"] == 0, "ir_excess_z"]))
    d1 = float(np.nanmedian(per_cone.loc[per_cone["cone"] == 1, "ir_excess_z"]))
    assert abs(d1 - d0) < 0.2, "the per-cone counterfactual should be flat"

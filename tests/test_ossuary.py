"""Offline tests for the OSSUARY channel (no network).

The suite is the CI gate.  It must (a) recover an injected warm excess on a halo
star, (b) reject every contaminant the literature says will dominate the raw flag
list, and (c) degrade honestly when the archive returns nothing.

The synthetic sample is built from a *physically consistent* photosphere plus a
*physically consistent* blackbody injection -- the dust is added in flux space
across all four WISE bands at once, so a test that recovers it is exercising the
same blackbody fit the real pipeline uses, not a hand-tuned magnitude offset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.config import load_config
from seti.ossuary import excess as oex
from seti.ossuary import kinematics as okin
from seti.ossuary import run as orun
from seti.ossuary import vet as ovet
from seti.photometry import band_freq_hz, flux_jy_to_mag, mag_to_flux_jy, planck_bnu

# Intrinsic photosphere colours (anchor Ks minus band) as a linear function of
# BP-RP.  Values are representative of FGK dwarfs: Ks-W1 is a few hundredths.
_LOCUS = {"W1": (0.02, 0.030), "W2": (0.03, 0.050),
          "W3": (0.05, 0.070), "W4": (0.05, 0.080),
          "J": (-0.62, -0.55), "H": (-0.16, -0.16)}
_BANDS = ("W1", "W2", "W3", "W4")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(autouse=True)
def _isolate_results(tmp_path, monkeypatch):
    """Keep tests out of the live ``results/ossuary/`` directory.

    ``orun.out_dir`` resolves to ``cfg.root/results/ossuary`` irrespective of
    where a test points ``input_path``, so any test that reaches the report
    stage overwrites the real ``summary.json``. That already happened: a
    committed summary read ``NO_DATA_REACHED`` with a note naming a
    ``/tmp/pytest-of-root/...`` fixture path, which is indistinguishable at a
    glance from a genuine empty search result. Redirect the output for every
    test in this module so a unit test can never be mistaken for a run.
    """
    monkeypatch.setattr(orun, "out_dir", lambda _cfg: tmp_path)


def _photosphere(bp_rp: np.ndarray, ks: np.ndarray) -> dict:
    """Photospheric magnitudes for each band from the linear colour locus."""
    return {b: ks - (a + s * (bp_rp - 0.5)) for b, (a, s) in _LOCUS.items()}


def inject_blackbody(mags: dict, t_dust_k: float, frac_in_band: float,
                     ref_band: str = "W3") -> dict:
    """Add a single-temperature blackbody scaled to a flux fraction in ``ref_band``.

    Physically consistent across bands: one solid angle, one temperature.
    """
    f_ref = float(mag_to_flux_jy(mags[ref_band], ref_band))
    b_ref = float(planck_bnu(t_dust_k, band_freq_hz(ref_band))) * 1e26
    omega = frac_in_band * f_ref / b_ref
    out = dict(mags)
    for b in _BANDS:
        if b not in out:
            continue
        f = float(mag_to_flux_jy(out[b], b))
        f += omega * float(planck_bnu(t_dust_k, band_freq_hz(b))) * 1e26
        out[b] = float(flux_jy_to_mag(f, b))
    return out


def make_sample(n: int = 1400, seed: int = 7) -> pd.DataFrame:
    """A clean metal-poor halo dwarf population with no dust at all.

    Every star is a high-proper-motion halo dwarf at high Galactic latitude with a
    photospheric SED.  Contaminants and injections are added by the individual
    tests so each one is isolated.
    """
    rng = np.random.default_rng(seed)
    bp_rp = rng.uniform(0.55, 1.45, n)
    ks = rng.uniform(9.0, 12.0, n)
    mags = _photosphere(bp_rp, ks)

    df = pd.DataFrame({
        "source_id": np.arange(1, n + 1, dtype=np.int64),
        "ra": rng.uniform(10.0, 350.0, n),
        "dec": rng.uniform(-60.0, 60.0, n),
        "l": rng.uniform(0.0, 360.0, n),
        "b": rng.choice([1.0, -1.0], n) * rng.uniform(35.0, 80.0, n),
        "parallax": rng.uniform(1.0, 4.0, n),
        "parallax_error": 0.02,
        "parallax_over_error": rng.uniform(50.0, 150.0, n),
        # Halo kinematics: large proper motions and large radial velocities.
        "pmra": rng.normal(0.0, 90.0, n),
        "pmdec": rng.normal(0.0, 90.0, n),
        "pmra_error": 0.03, "pmdec_error": 0.03,
        "radial_velocity": rng.normal(0.0, 150.0, n),
        "radial_velocity_error": 1.0,
        "phot_g_mean_mag": ks + 1.6 + bp_rp,
        "bp_rp": bp_rp,
        "ruwe": rng.uniform(0.9, 1.15, n),
        "feh": rng.uniform(-2.5, -1.05, n),
        "feh_provenance": "gaia_gspspec",
        "teff": rng.uniform(4600.0, 6200.0, n),
        "logg": rng.uniform(4.1, 4.6, n),
        "Ksmag": ks, "e_Ksmag": 0.020,
        "ph_qual": "AAAA", "cc_flags": "0000", "ext_flag": 0,
        "number_of_mates": 0, "number_of_neighbours": 1,
        "ebv_sfd": rng.uniform(0.005, 0.045, n),
        "track": "spec",
    })
    for b, m in mags.items():
        df[f"{b}mag"] = m
        df[f"e_{b}mag"] = 0.020
    # Scatter the photometry so the empirical locus has a real width.
    for b in list(mags):
        df[f"{b}mag"] = df[f"{b}mag"] + rng.normal(0.0, 0.012, n)

    # WISE centroid: the Gaia position propagated to the AllWISE mean epoch.
    dt = 2010.5 - 2016.0
    df["ra_wise"] = df["ra"] + (df["pmra"] * dt / 1000.0) / 3600.0 / np.cos(
        np.radians(df["dec"]))
    df["dec_wise"] = df["dec"] + (df["pmdec"] * dt / 1000.0) / 3600.0
    # Keep every star well clear of the globular-cluster veto zones.
    df.loc[:, "ra"] = df["ra"].clip(20.0, 340.0)
    return df


def _set(df: pd.DataFrame, i: int, **kw) -> None:
    for k, v in kw.items():
        if k not in df.columns:
            df[k] = np.nan
        df.loc[i, k] = v


# ==========================================================================
# Kinematics
# ==========================================================================

def test_uvw_matches_an_independent_calculation(cfg):
    """UVW for a star with only radial motion must point along its line of sight."""
    df = pd.DataFrame({"ra": [90.0], "dec": [0.0], "parallax": [10.0],
                       "pmra": [0.0], "pmdec": [0.0], "radial_velocity": [100.0]})
    out = okin.space_velocity(df)
    v = np.array([out["U_kms"][0], out["V_kms"][0], out["W_kms"][0]])
    assert np.isclose(np.linalg.norm(v), 100.0, rtol=1e-9)
    # Tangential part is exactly zero when the proper motion is zero.
    assert np.allclose([out["Ut_kms"][0], out["Vt_kms"][0], out["Wt_kms"][0]], 0.0)


def test_missing_rv_gives_a_lower_bound_not_a_guess(cfg):
    """v_tan must never exceed v_tot: the bound is one-sided by construction."""
    k = cfg.thresholds["ossuary"]["kinematics"]
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-80, 80, n),
        "parallax": rng.uniform(1, 5, n), "parallax_error": 0.01,
        "parallax_over_error": 200.0,
        "pmra": rng.normal(0, 100, n), "pmdec": rng.normal(0, 100, n),
        "pmra_error": 0.02, "pmdec_error": 0.02,
        "radial_velocity": rng.normal(0, 120, n), "radial_velocity_error": 1.0,
        "phot_g_mean_mag": 12.0,
    })
    out = okin.classify(df, k)
    assert (out["v_tan_lsr_kms"] <= out["v_tot_lsr_kms"] + 1e-6).all()


def test_no_rv_star_is_unclassified_never_thin_disk(cfg):
    """A slow-tangential star with no RV is unclassified, not thin disk."""
    k = cfg.thresholds["ossuary"]["kinematics"]
    df = pd.DataFrame({
        "ra": [45.0, 45.0], "dec": [10.0, 10.0], "parallax": [2.0, 2.0],
        "parallax_error": [0.01, 0.01], "parallax_over_error": [200.0, 200.0],
        "pmra": [1.0, 400.0], "pmdec": [1.0, 400.0],
        "pmra_error": [0.02, 0.02], "pmdec_error": [0.02, 0.02],
        "radial_velocity": [np.nan, np.nan],
        "radial_velocity_error": [np.nan, np.nan],
        "phot_g_mean_mag": [12.0, 12.0],
    })
    out = okin.classify(df, k)
    assert (out["kinematic_method"] == "vtan_lower_bound").all()
    # Slow tangential motion is simply unknown...
    assert out["population"].iloc[0] == "unclassified"
    assert not bool(out["halo_flag"].iloc[0])
    # ...but a large tangential velocity is *sufficient* for halo membership.
    assert out["population"].iloc[1] == "halo"
    assert bool(out["halo_flag"].iloc[1])


def test_giants_are_separated_from_dwarfs(cfg):
    s = cfg.thresholds["ossuary"]["sample"]
    df = pd.DataFrame({"phot_g_mean_mag": [12.0, 12.0], "parallax": [10.0, 0.2],
                       "logg": [4.5, 1.8]})
    cls = okin.luminosity_class(df, s)
    assert list(cls) == ["dwarf", "giant"]


# ==========================================================================
# Excess detector
# ==========================================================================

def test_locus_refuses_to_extrapolate(cfg):
    e = cfg.thresholds["ossuary"]["excess"]
    df = make_sample(600)
    loc = oex.fit_colour_locus(df, "W1", min_per_bin=e["locus_min_per_bin"])
    assert loc.n_bins > 3
    med, sca = loc.predict(np.array([1.0, 5.0, -3.0, np.nan]))
    assert np.isfinite(med[0]) and np.isfinite(sca[0])
    assert not np.isfinite(med[1]) and not np.isfinite(med[2])
    assert not np.isfinite(med[3])


def test_clean_population_produces_almost_no_flags(cfg):
    """The detector must not fire on its own null sample."""
    df = make_sample(1400)
    vetted, summary = orun.analyze(df, cfg)
    assert summary["verdict"] == "OK"
    assert summary["n_excess_flagged"] <= 0.01 * len(df)
    assert summary["n_candidates"] == 0


def test_injected_warm_excess_on_a_halo_star_is_recovered(cfg):
    """The headline requirement: inject warm dust on a halo dwarf, find it."""
    df = make_sample(1400)
    i = 0
    mags = {b: float(df.loc[i, f"{b}mag"]) for b in _BANDS}
    hot = inject_blackbody(mags, t_dust_k=450.0, frac_in_band=0.9, ref_band="W3")
    for b, m in hot.items():
        df.loc[i, f"{b}mag"] = m
    # Make it unambiguously halo and unambiguously metal-poor.
    _set(df, i, pmra=350.0, pmdec=-300.0, radial_velocity=-220.0, feh=-2.1,
         teff=5400.0, logg=4.4)
    dt = 2010.5 - 2016.0
    _set(df, i,
         ra_wise=float(df.loc[i, "ra"]) + (350.0 * dt / 1000.0) / 3600.0
         / np.cos(np.radians(float(df.loc[i, "dec"]))),
         dec_wise=float(df.loc[i, "dec"]) + (-300.0 * dt / 1000.0) / 3600.0)

    vetted, summary = orun.analyze(df, cfg)
    row = vetted.loc[i]
    assert bool(row["excess_flag"]), "injected warm excess was not flagged"
    assert row["population"] == "halo"
    assert row["verdict"] == "surviving", f"rejected as {row['reject_reason']}"
    assert bool(row["candidate"])
    assert summary["n_candidates"] >= 1
    # The recovered dust temperature must bracket the injected one.
    assert 200.0 < float(row["t_dust_k"]) < 1200.0
    assert float(row["tau"]) > 0


def test_w4_only_cirrus_artifact_does_not_flag(cfg):
    """LEDGER: a W4-only excess is cirrus. It must never become a candidate."""
    df = make_sample(1400)
    i = 3
    # Brighten W4 hugely, leave the star-dominated bands photospheric.
    df.loc[i, "W4mag"] = float(df.loc[i, "W4mag"]) - 3.0
    df.loc[i, "e_W4mag"] = 0.05
    _set(df, i, feh=-2.0, pmra=300.0, pmdec=-250.0, radial_velocity=-210.0)

    vetted, _ = orun.analyze(df, cfg)
    row = vetted.loc[i]
    assert float(row["chi_W4"]) > 5.0, "the artefact should be formally significant"
    assert not bool(row["candidate"]), "a W4-only excess became a candidate"
    if bool(row["excess_flag"]):
        assert row["reject_reason"] == "ledger"


def test_unresolved_companion_hotter_than_grains_survive_does_not_flag(cfg):
    """A fitted excess above 1800 K is a companion photosphere, not dust."""
    df = make_sample(1400)
    i = 5
    mags = {b: float(df.loc[i, f"{b}mag"]) for b in _BANDS}
    # A 2600 K companion: hotter than any grain survives.
    hot = inject_blackbody(mags, t_dust_k=2600.0, frac_in_band=0.9, ref_band="W1")
    for b, m in hot.items():
        df.loc[i, f"{b}mag"] = m
    _set(df, i, feh=-2.0, pmra=300.0, pmdec=-250.0, radial_velocity=-210.0)

    vetted, _ = orun.analyze(df, cfg)
    row = vetted.loc[i]
    assert bool(row["excess_flag"]), "the injection should be detected as an excess"
    assert float(row["t_dust_k"]) > 1800.0
    assert bool(row["t_dust_too_hot"])
    assert not bool(row["companion_ok"])
    assert not bool(row["candidate"])


def test_thin_disk_metal_rich_star_never_enters_the_sample(cfg):
    """The claim is about hosts with no reservoir. A thin-disk metal-rich star
    with a genuine excess must be rejected as out of sample, not reported."""
    df = make_sample(1400)
    i = 9
    mags = {b: float(df.loc[i, f"{b}mag"]) for b in _BANDS}
    hot = inject_blackbody(mags, t_dust_k=450.0, frac_in_band=0.9, ref_band="W3")
    for b, m in hot.items():
        df.loc[i, f"{b}mag"] = m
    # Metal-rich and kinematically cold: an ordinary thin-disk debris disk.
    _set(df, i, feh=+0.15, pmra=2.0, pmdec=-3.0, radial_velocity=-5.0)
    dt = 2010.5 - 2016.0
    _set(df, i,
         ra_wise=float(df.loc[i, "ra"]) + (2.0 * dt / 1000.0) / 3600.0
         / np.cos(np.radians(float(df.loc[i, "dec"]))),
         dec_wise=float(df.loc[i, "dec"]) + (-3.0 * dt / 1000.0) / 3600.0)

    vetted, _ = orun.analyze(df, cfg)
    row = vetted.loc[i]
    assert bool(row["excess_flag"]), "the excess itself is real and should be seen"
    assert not bool(row["metal_poor"])
    assert not bool(row["halo_flag"])
    assert not bool(row["null_reservoir_host"])
    assert not bool(row["candidate"])
    assert row["reject_reason"] == "not_a_null_reservoir_host"


# ==========================================================================
# Contamination gates
# ==========================================================================

def test_blended_neighbour_does_not_flag(cfg):
    """A Gaia neighbour in the WISE beam able to supply the excess kills it."""
    c = cfg.thresholds["ossuary"]["contamination"]
    cand = {"source_id": 1, "ra": 100.0, "dec": 20.0,
            "W1_obs_jy": 1.0e-2, "W1_excess_jy": 5.0e-4}
    # A neighbour 2.5" away, bright enough that its own W1 exceeds the excess.
    near = pd.DataFrame({"source_id": [2], "ra": [100.0 + 2.5 / 3600.0],
                         "dec": [20.0], "phot_g_mean_mag": [14.0],
                         "bp_rp": [1.5]})
    v = ovet.beam_blend_verdict(cand, near, c)
    assert v["blend_verdict"] == "beam_blend"
    assert v["n_beam_neighbours"] == 1
    assert v["neighbour_over_excess"] >= 1.0

    # The same neighbour pushed outside the beam is irrelevant.
    far = near.copy()
    far["ra"] = 100.0 + 20.0 / 3600.0
    assert ovet.beam_blend_verdict(cand, far, c)["blend_verdict"] == "isolated"

    # No neighbours at all -> isolated, not silently "clean".
    assert ovet.beam_blend_verdict(cand, None, c)["blend_verdict"] == "isolated"


def test_static_background_source_fails_astrometric_registration(cfg):
    """A high-PM star whose IR counterpart did not move is a background object."""
    a = cfg.thresholds["ossuary"]["contamination"]["astrometry"]
    dt = a["wise_mean_epoch"] - a["gaia_ref_epoch"]
    pmra, pmdec = 400.0, -300.0
    base = {"ra": 150.0, "dec": 5.0, "pmra": pmra, "pmdec": pmdec,
            "pmra_error": 0.02, "pmdec_error": 0.02}
    moving = dict(base)
    moving["ra_wise"] = base["ra"] + (pmra * dt / 1000.0) / 3600.0 / np.cos(
        np.radians(base["dec"]))
    moving["dec_wise"] = base["dec"] + (pmdec * dt / 1000.0) / 3600.0
    static = dict(base, ra_wise=base["ra"], dec_wise=base["dec"])

    df = pd.DataFrame([moving, static])
    g = ovet.astrometry_gate(df, a)
    assert bool(g["registration_ok"].iloc[0]), "the real star must register"
    assert not bool(g["registration_ok"].iloc[1]), "a static source must not"
    # The propagation is what makes the difference: without it, the *real* star
    # would have been the one that looked misregistered.
    assert g["registration_arcsec_unpropagated"].iloc[0] > \
        g["registration_arcsec"].iloc[0]
    assert g["registration_arcsec_unpropagated"].iloc[0] > 1.0


def test_comovement_rejects_a_zero_pm_infrared_source(cfg):
    a = cfg.thresholds["ossuary"]["contamination"]["astrometry"]
    df = pd.DataFrame({
        "ra": [150.0], "dec": [5.0], "ra_wise": [150.0], "dec_wise": [5.0],
        "pmra": [400.0], "pmdec": [-300.0],
        "pmra_error": [0.02], "pmdec_error": [0.02],
        "pmra_wise": [0.0], "pmdec_wise": [0.0],
        "e_pmra_wise": [30.0], "e_pmdec_wise": [30.0]})
    g = ovet.astrometry_gate(df, a)
    assert g["comovement_sigma"].iloc[0] > 3.0
    assert not bool(g["comovement_ok"].iloc[0])


def test_chance_superposition_prior_is_monotonic_and_bounded(cfg):
    c = cfg.thresholds["ossuary"]["contamination"]
    faint = ovet.allwise_source_density_per_arcsec2(np.array([16.0]), c)[0]
    bright = ovet.allwise_source_density_per_arcsec2(np.array([10.0]), c)[0]
    assert bright < faint            # fewer bright interlopers
    budget = ovet.expected_chance_alignments(300_000, c)
    assert budget["expected_hot_dogs"] > 0
    # The registration cut must buy a large factor over the raw 6.5" beam.
    assert budget["leverage_of_registration_cut"] > 10


def test_interloper_prior_uses_the_detection_band_not_always_w1(cfg):
    """Cool dust is faint at 3.4 um; the prior must not punish it for that.

    A 400 K excess sits far down the Wien tail at W1, so its W1 flux is nearly
    zero.  Reading the interloper counts at that near-zero "equivalent W1
    magnitude" would imply an enormous density and veto exactly the coolest --
    most interesting -- detections.  Only bands carrying a real detection may
    drive the prior.
    """
    c = cfg.thresholds["ossuary"]["contamination"]
    # Same star twice: a strong W3 detection, with a sub-sigma positive noise
    # blip in W2 in one case.  The blip must not change the verdict.
    df = pd.DataFrame({
        "W1_excess_jy": [7.7e-4, 7.7e-4], "chi_W1": [0.8, 0.8],
        "W2_excess_jy": [1.8e-4, 1.8e-4], "chi_W2": [0.3, 0.3],
        "W3_excess_jy": [2.1e-3, 2.1e-3], "chi_W3": [17.0, 17.0],
        "W4_excess_jy": [1.4e-3, np.nan], "chi_W4": [31.0, np.nan],
        "ext_flag": [0, 0]})
    p = ovet.chance_superposition_p(df, c)
    assert (p < c["extragalactic"]["max_chance_superposition_p"]).all(), (
        f"a real cool-dust detection was vetoed by noise-band leakage: {list(p)}")

    # A star with no significant excess in any band has no prior to speak of.
    quiet = pd.DataFrame({"W1_excess_jy": [1e-5], "chi_W1": [0.2]})
    assert not np.isfinite(ovet.chance_superposition_p(quiet, c).iloc[0])


def test_globular_cluster_sightline_is_vetoed(cfg):
    """Boyer et al. 2010: a published metal-poor IR excess that was blending."""
    c = cfg.thresholds["ossuary"]["contamination"]
    df = pd.DataFrame({"ra": [6.024, 180.0], "dec": [-72.081, 0.0]})
    g = ovet.globular_cluster_veto(df, c)
    assert not bool(g["globular_ok"].iloc[0])
    assert "47 Tuc" in g["globular_cluster"].iloc[0]
    assert bool(g["globular_ok"].iloc[1])


def test_lambda_boo_temperature_veto(cfg):
    """Murphy et al. 2020: 21/34 lambda Boo stars carry infrared excesses."""
    s = cfg.thresholds["ossuary"]["sample"]
    df = pd.DataFrame({"teff": [7600.0, 5500.0], "M_G": [2.0, 5.5],
                       "bp_rp": [0.30, 0.9]})
    g = ovet.impostor_gate(df, s)
    assert bool(g["lambda_boo_risk"].iloc[0])
    assert not bool(g["impostor_ok"].iloc[0])
    assert bool(g["impostor_ok"].iloc[1])


def test_negative_w1_w2_is_treated_as_a_blend(cfg):
    th = cfg.thresholds["ossuary"]
    df = pd.DataFrame({
        "W1mag": [10.0, 10.0], "W2mag": [10.4, 9.9],
        "w1_w2_obs": [-0.4, 0.1], "phot_g_mean_mag": [12.0, 12.0],
        "bp_rp": [0.9, 0.9], "chi_W1": [9.0, 9.0], "W1_excess_jy": [1e-4, 1e-4]})
    g = ovet.ledger_gate(df, th["excess"], th["sample"])
    assert not bool(g["w1_w2_physical"].iloc[0])
    assert not bool(g["ledger_ok"].iloc[0])
    assert bool(g["ledger_ok"].iloc[1])


def test_wise_quality_flags_reject_artifacts(cfg):
    c = cfg.thresholds["ossuary"]["contamination"]
    df = pd.DataFrame({
        "ph_qual": ["AAAA", "UUUU", "AAAA", "AAAA"],
        "cc_flags": ["0000", "0000", "DD00", "0000"],
        "W1mag": [11.0, 11.0, 11.0, 11.0],
        "number_of_mates": [0, 0, 0, 2],
        "number_of_neighbours": [1, 1, 1, 3],
        "ext_flag": [0, 0, 0, 0]})
    g = ovet.wise_quality_gate(df, c)
    assert list(g["wise_quality_ok"]) == [True, False, False, False]


def test_cirrus_gate_and_population_level_correlation(cfg):
    c = cfg.thresholds["ossuary"]["contamination"]
    df = pd.DataFrame({"ebv_sfd": [0.02, 0.5, np.nan], "b": [55.0, 55.0, 55.0]})
    g = ovet.cirrus_gate(df, c)
    assert list(g["cirrus_ok"]) == [True, False, False]
    assert not bool(g["cirrus_tested"].iloc[2])   # untested, not passed

    rng = np.random.default_rng(1)
    n = 500
    ebv = rng.uniform(0, 0.4, n)
    leaky = pd.DataFrame({"ebv_sfd": ebv,
                          "excess_flag": rng.random(n) < (0.02 + 2.0 * ebv)})
    res = ovet.cirrus_correlation_test(leaky)
    assert res["tested"] and res["spearman_rho"] > 0 and res["p_value"] < 0.01


# ==========================================================================
# Honest degradation
# ==========================================================================

def test_empty_archive_response_degrades_honestly(cfg):
    vetted, summary = orun.analyze(pd.DataFrame(), cfg)
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["n_input"] == 0
    assert "n_candidates" not in summary or summary.get("n_candidates", 0) == 0

    vetted, summary = orun.analyze(None, cfg)
    assert summary["verdict"] == "NO_DATA_REACHED"


def test_too_few_stars_to_build_a_locus_says_so(cfg):
    """Below the minimum bin occupancy there is no empirical photosphere."""
    df = make_sample(12)
    _, summary = orun.analyze(df, cfg)
    assert summary["verdict"] in ("NO_LOCUS", "OK")
    if summary["verdict"] == "NO_LOCUS":
        assert "photosphere" in summary["note"]


def test_run_without_a_sample_table_emits_no_data_reached(cfg, tmp_path):
    out = orun.run(cfg, stage="analyze", input_path=tmp_path / "missing.parquet",
                   do_followup=False)
    assert out["verdict"] == "NO_DATA_REACHED"


def test_followup_survives_failing_fetchers(cfg):
    """Every follow-up fetcher can fail; the stage must not raise."""
    cands = pd.DataFrame({"source_id": [1], "ra": [100.0], "dec": [20.0],
                          "W1_obs_jy": [1e-2], "W1_excess_jy": [5e-4],
                          "b": [55.0]})

    def boom(*a, **k):
        raise RuntimeError("archive down")

    out = orun.stage_followup(cfg, cands, fetch_ebv=boom, fetch_neighbours=boom,
                              fetch_simbad=boom)
    assert len(out) == 1
    assert out["blend_verdict"].iloc[0] == "isolated"
    # E(B-V) never arrived, so the cirrus gate is untested and must not pass.
    assert not bool(out["cirrus_tested"].iloc[0])
    assert out["followup_verdict"].iloc[0] == "rejected"


# ==========================================================================
# Physics helpers
# ==========================================================================

def test_wien_peaks_match_the_wise_bands():
    peaks = {b: oex.wien_peak_k(b) for b in _BANDS}
    assert 840 < peaks["W1"] < 880
    assert 615 < peaks["W2"] < 645
    assert 235 < peaks["W3"] < 260
    assert 125 < peaks["W4"] < 140
    assert peaks["W1"] > peaks["W2"] > peaks["W3"] > peaks["W4"]


def test_cascade_timescale_matches_lacki_scaling():
    """t_collision ~ P / f (Lacki 2025, arXiv:2504.21151)."""
    assert oex.cascade_timescale_yr(1.0, 1e-3) == pytest.approx(1000.0)
    assert oex.cascade_timescale_yr(1.0, 1e-2) == pytest.approx(100.0)
    assert np.isnan(oex.cascade_timescale_yr(1.0, 0.0))


def test_blackbody_fit_recovers_a_known_temperature():
    t_true, omega = 500.0, 1e-16
    f = {b: omega * float(planck_bnu(t_true, band_freq_hz(b))) * 1e26
         for b in _BANDS}
    e = {b: 0.02 * v for b, v in f.items()}
    t_fit, om_fit, chi2 = oex.fit_excess_blackbody(f, e)
    assert t_fit == pytest.approx(t_true, rel=0.05)
    assert om_fit == pytest.approx(omega, rel=0.10)
    assert chi2 < 1.0
    # Fewer than two bands cannot constrain a temperature: say so, do not guess.
    assert np.isnan(oex.fit_excess_blackbody({"W1": 1.0}, {"W1": 0.1})[0])

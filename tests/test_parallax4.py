"""PARALLAX4 offline suite (no network; see tests/conftest.py).

Real data used offline: the ESA Gaia DR4 epoch-astrometry prerelease
(12 sources, 2026-06-26, sha256-pinned) and one real Gaia DR3 DataLink
EPOCH_PHOTOMETRY product (Gaia-4).  Everything else is DR4-shaped synthetic
data from ``seti.parallax4.simulate`` written in the real column spellings.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.parallax4 import acquire as acq
from seti.parallax4 import controls as C
from seti.parallax4 import epochs as E
from seti.parallax4 import greydip as G
from seti.parallax4 import photocentre as P
from seti.parallax4 import run as R
from seti.parallax4 import schema as S
from seti.parallax4 import simulate as SIM
from seti.parallax4 import watchlist as W

FIX = Path(__file__).parent / "fixtures" / "parallax4"
PRE = FIX / acq.PRERELEASE_ZIP
GAIA4 = 1457486023639239296
GAIA4_VOT = FIX / f"dr3_epoch_photometry_{GAIA4}.vot"


@pytest.fixture(scope="module")
def conf():
    return R.load_config()


@pytest.fixture(scope="module")
def gcfg(conf):
    return G.GreyConfig.from_dict(conf.get("grey"))


@pytest.fixture(scope="module")
def prerelease_ccd():
    return E.read_prerelease_zip(PRE)


@pytest.fixture(scope="module")
def gaia4_phot():
    from astropy.table import Table

    return E.photometry_from_long(Table.read(GAIA4_VOT, format="votable"), source_id=GAIA4)


# ---------------------------------------------------------------------------
# schema discovery
# ---------------------------------------------------------------------------
def test_roles_resolve_across_naming_conventions():
    r = S.resolve(["sourceId", "transitId", "scanPosAngle", "centroidPosAl", "centroidPosErrorAl",
                   "parallaxFactorAl", "obsTimeTcb", "brand_new_dr4_column"],
                  S.ASTRO_ROLES, S.ASTRO_REQUIRED)
    assert not r.missing
    assert r.get("scan_angle") == "scanPosAngle"
    assert "brand_new_dr4_column" in r.unused          # extra columns never break parsing


def test_missing_role_is_named_not_guessed():
    r = S.resolve(["source_id", "transit_id", "centroid_pos_al"], S.ASTRO_ROLES, S.ASTRO_REQUIRED)
    assert "scan_angle" in r.missing and "parallax_factor_al" in r.missing
    with pytest.raises(E.SchemaError) as ei:
        E.astrometry_ccd_frame(pd.DataFrame({"source_id": [1], "transit_id": [2],
                                             "centroid_pos_al": [0.1]}))
    assert "scan_angle" in ei.value.resolution.missing


@pytest.mark.parametrize("cell,expect", [
    ("(1.5,2.5,null)", [1.5, 2.5, np.nan]),
    ("[1.5, 2.5, NaN]", [1.5, 2.5, np.nan]),
    ("{1,2,3}", [1, 2, 3]),
    ('"[true, false, true]"', [1, 0, 1]),
    ("1.0 2.0 3.0", [1, 2, 3]),
    ("", []),
    ("[]", []),
])
def test_array_cells_parse_in_every_spelling(cell, expect):
    vals, lens = S.parse_array_column([cell])
    assert lens[0] == len(expect)
    np.testing.assert_allclose(vals, np.array(expect, float), equal_nan=True)


def test_flags_are_cast_unsigned_before_bit_tests():
    f = S.as_uint16_flags(np.array([-32208, 560], dtype=np.int16))
    assert f.dtype == np.uint16 and int(f[0]) == 33328 and bool(f[0] & 0x8000)


# ---------------------------------------------------------------------------
# readers on REAL data
# ---------------------------------------------------------------------------
def test_prerelease_fixture_is_the_esa_file():
    assert hashlib.sha256(PRE.read_bytes()).hexdigest() == acq.PRERELEASE_SHA256


def test_prerelease_reader_shape(prerelease_ccd):
    c = prerelease_ccd
    assert c["source_id"].nunique() == 12
    assert len(c) == 10080                    # 1008 transits x 10 CCDs
    assert int(c["used"].sum()) == 7467       # used_by_agis_al, as measured
    assert c.attrs["release"] == "Gaia DR4_RC3"
    # NaN centroid is never used
    assert not (c["used"] & ~np.isfinite(c["x_al"])).any()


def test_prerelease_parallaxes_recovered(prerelease_ccd):
    """A2: the direction convention (sin theta, cos theta) and the time system
    are right, or the published parallaxes would not come back."""
    rep = C.prerelease_control(prerelease_ccd)
    assert rep["n_sources"] == 12
    assert rep["n_parallax_ok"] == 12, rep["parallax_recovery"]
    bh = next(r for r in rep["parallax_recovery"] if r["source_id"] == 3937211745905473024)
    assert abs(bh["parallax"] - 25.6) < 0.1


def test_dr3_photometry_joins_dr4_astrometry_on_transit_id(prerelease_ccd, gaia4_phot):
    """The DR3 transit_id IS the DR4 transit_id: 63 of Gaia-4's 65 DR3
    photometric transits are in the prerelease; barycentric times agree."""
    ids3 = set(gaia4_phot["transit_id"])
    ids4 = set(prerelease_ccd.loc[prerelease_ccd["source_id"] == GAIA4, "transit_id"])
    assert len(ids3) == 65 and len(ids3 & ids4) == 63
    tr = E.collapse_transits(prerelease_ccd)
    j = E.join_photometry_astrometry(gaia4_phot, tr)
    assert len(j) >= 40
    assert np.max(np.abs(j["t"] - j["t_astro"])) * 86400 < 5.0


def test_joint_fit_on_real_gaia4(prerelease_ccd, gaia4_phot):
    """The first per-transit photometry x astrometry fit on real Gaia data:
    Gaia-4 varies by ~0.5% and its photocentre does not follow (no blend)."""
    rep = C.prerelease_control(prerelease_ccd, {GAIA4: gaia4_phot})
    j = rep["joint"][0]
    assert j["n_joint_usable"] >= 40
    assert j["verdict"] not in ("BLEND_NEIGHBOUR", "BLEND_UNRESOLVED")
    assert j["p_D"] > 1e-4
    assert rep["join_ok"] is True


def test_bulk_wide_csv_round_trip(tmp_path):
    long = pd.concat([SIM.simulate_grey_lightcurve("grey_multi", seed=s, source_id=100 + s)
                      for s in range(3)], ignore_index=True)
    ref = E.photometry_from_long(long)
    for style in ("paren", "bracket", "brace"):
        p = tmp_path / f"EpochPhotometry_000000-000001_{style}.csv.gz"
        with gzip.open(p, "wt") as fh:
            fh.write(SIM.to_wide_csv_text(long, style=style))
        got = E.read_bulk_csv(p)
        assert got.attrs["mismatched_array_rows"] == 0
        assert len(got) == len(ref)
        np.testing.assert_allclose(got["f_g"].to_numpy(), ref["f_g"].to_numpy(), rtol=1e-12)
        np.testing.assert_array_equal(got["transit_id"].to_numpy(), ref["transit_id"].to_numpy())
        assert set(got["source_id"]) == {100, 101, 102}


def test_wide_reader_refuses_to_misalign_arrays():
    df = pd.DataFrame({"source_id": [1], "g_transit_time": ["(1,2,3)"], "g_transit_flux": ["(5,5,5)"],
                       "g_transit_flux_error": ["(1,1,1)"], "bp_flux": ["(5,5)"],
                       "bp_flux_error": ["(1,1,1)"], "rp_flux": ["(5,5,5)"], "rp_flux_error": ["(1,1,1)"]})
    out = E.photometry_from_wide(df)
    assert out.attrs["mismatched_array_rows"] == 1
    assert out["f_bp"].isna().all()          # the short column is blanked, not shifted


# ---------------------------------------------------------------------------
# the photometric detector
# ---------------------------------------------------------------------------
def _run(kind, gcfg, seed=0, **kw):
    lc = E.photometry_from_long(SIM.simulate_grey_lightcurve(kind, seed=seed, source_id=9, **kw))
    return G.detect_source(lc, gcfg)


@pytest.mark.parametrize("seed", range(4))
def test_injected_grey_multi_transit_dip_is_tier_A(gcfg, seed):
    ev, summ = _run("grey_multi", gcfg, seed=seed)
    assert len(ev) == 1
    e = ev[0]
    assert e["grey_class"] == "GREY" and e["coherence"] == "MULTI" and G.tier(e) == "A"
    assert abs(e["delta_g"] + 0.2) < 0.02


@pytest.mark.parametrize("seed", range(4))
def test_dust_dip_is_chromatic_not_grey(gcfg, seed):
    """The dominant astrophysical confounder of a grey dip: dust (BP/RP 1.7)."""
    ev, _ = _run("dust_multi", gcfg, seed=seed)
    assert len(ev) == 1 and ev[0]["grey_class"] == "CHROMATIC_BLUE" and G.tier(ev[0]) == "C"
    assert abs(ev[0]["ratio_bp_rp"] - 1.7) < 0.15


def test_flare_is_chromatic(gcfg):
    ev, _ = _run("flare", gcfg)
    assert ev and ev[0]["kind"] == "BRIGHTENING" and ev[0]["grey_class"].startswith("CHROMATIC")


@pytest.mark.parametrize("seed", range(3))
def test_eclipsing_binary_is_caught_by_the_period_test(gcfg, seed):
    ev, summ = _run("eb", gcfg, seed=seed)
    assert summ["period_class"] == "PERIODIC"
    assert not any(G.tier(e) == "A" for e in ev)


def test_single_band_anomaly_is_not_promoted(gcfg):
    ev, _ = _run("g_only", gcfg)
    assert ev and ev[0]["grey_class"] == "G_ONLY" and G.tier(ev[0]) == "C"


def test_fov_partner_contradiction(gcfg):
    ev, _ = _run("partner_contradicted", gcfg)
    assert ev and ev[0]["coherence"] == "SINGLE_CONTRADICTED" and G.tier(ev[0]) == "C"


def test_isolated_single_transit_is_tier_B(gcfg):
    ev, _ = _run("grey_single", gcfg)
    assert ev and ev[0]["coherence"] == "SINGLE_UNCHECKED" and G.tier(ev[0]) == "B"


def test_quiet_and_ordinary_variable_give_nothing(gcfg):
    for kind in ("quiet", "variable"):
        for seed in range(3):
            ev, _ = _run(kind, gcfg, seed=seed)
            assert not ev, (kind, seed, ev)


def test_too_few_transits_is_reported(gcfg):
    lc = E.photometry_from_long(SIM.simulate_grey_lightcurve("quiet", seed=1, n_visits=3))
    ev, summ = G.detect_source(lc, gcfg)
    assert summ["status"] == "too_few_transits" and not ev


def test_grey_test_refuses_an_unmeasured_ratio(gcfg):
    t = G.grey_test(-0.1, 0.005, -0.1, 0.05, -0.1, 0.05, gcfg)
    assert t["grey_class"] == "GREY_UNCONSTRAINED"
    t = G.grey_test(-0.1, 0.003, -0.1, 0.004, -0.1, 0.004, gcfg)
    assert t["grey_class"] == "GREY"


def test_period_check_needs_enough_episodes(gcfg):
    r = G.period_check(np.linspace(0, 1000, 100), np.zeros(100, bool), np.array([10.0, 20.0]), gcfg)
    assert r["period_class"] == "PERIOD_UNTESTABLE"


def test_injection_into_real_gaia4_lightcurve(gaia4_phot, gcfg):
    """P1 on real data: a 20% grey two-transit dip in Gaia-4's own DR3 light
    curve is recovered as tier A; the same dip with dust colours is not grey."""
    rng = np.random.default_rng(3)
    lc2, meta = G.inject_episode(gaia4_phot, 0.2, ratio_bp_rp=1.0, n_transits=2, rng=rng)
    ev, _ = G.detect_source(lc2, gcfg)
    hit = G.recovered(ev, meta)
    assert hit is not None and G.tier(hit) == "A"
    lc3, meta3 = G.inject_episode(gaia4_phot, 0.2, ratio_bp_rp=1.7, n_transits=2,
                                  rng=np.random.default_rng(3))
    hit3 = G.recovered(G.detect_source(lc3, gcfg)[0], meta3)
    assert hit3 is not None and hit3["grey_class"] != "GREY"


# ---------------------------------------------------------------------------
# the photocentre test
# ---------------------------------------------------------------------------
def _scene(scene, seed=0, drop_neighbours=False, **kw):
    sim = SIM.simulate_source(scene, seed=seed, **kw)
    tr = E.collapse_transits(E.astrometry_ccd_frame(sim["astro"]))
    ph = E.photometry_from_long(sim["phot"], source_id=int(sim["astro"]["source_id"].iloc[0]))
    j = P.prepare_joint(E.join_photometry_astrometry(ph, tr))
    nb = None if drop_neighbours else sim["neighbours"]
    return P.fit_photocentre(j, target_flux_g=sim["target_flux_g"], neighbours=nb), sim


@pytest.mark.parametrize("seed", range(3))
def test_blended_eclipsing_binary_shows_the_photocentre_shift(seed):
    """Positive control: a blended EB's photocentre moves with its flux, by
    rho (1 - beta) along the neighbour's position angle."""
    r, sim = _scene("blended_eb", seed=seed)
    assert r["verdict"] == "BLEND_NEIGHBOUR"
    assert abs(r["D_mas"] - sim["truth"]["D_true_mas"]) / sim["truth"]["D_true_mas"] < 0.03
    assert abs(r["D_pa_deg"] - 60.0) < 2.0


def test_blend_without_a_catalogued_neighbour_is_still_a_blend():
    r, _ = _scene("blended_eb", drop_neighbours=True)
    assert r["verdict"] == "BLEND_UNRESOLVED"


@pytest.mark.parametrize("scene", ["eb_on_target", "grey_dip"])
def test_variation_on_the_target_gives_no_shift(scene):
    """Negative control: a single EB (and a grey dip ON the target) must not move."""
    for seed in range(3):
        r, _ = _scene(scene, seed=seed)
        assert r["verdict"] == "ON_TARGET", (scene, seed, r["reasons"])
        assert r["ul95_mas"] < 20.0


def test_scan_angle_neighbour_is_rejected_as_systematic():
    r, _ = _scene("window_neighbour", rho_mas=700.0, flux_ratio=0.5)
    assert r["verdict"] == "SCAN_ANGLE_FLUX"


def test_scan_angle_flux_without_neighbour_table():
    r, _ = _scene("window_neighbour", drop_neighbours=True, rho_mas=700.0, flux_ratio=0.5)
    assert r["verdict"] in ("SCAN_ANGLE_FLUX", "SECTOR_INCONSISTENT")


def test_detector_frame_coupling_is_rejected():
    r, _ = _scene("cti")
    assert r["verdict"] == "DETECTOR_FRAME"


def test_constant_star_has_nothing_to_test():
    r, _ = _scene("single")
    assert r["verdict"] == "NO_FLUX_VARIATION"


def test_too_few_transits_is_insufficient():
    r, _ = _scene("blended_eb", n_visits=1)
    assert r["verdict"] == "INSUFFICIENT"


def test_neighbour_on_the_wrong_side_does_not_match():
    sim = SIM.simulate_source("blended_eb", seed=1)
    tr = E.collapse_transits(E.astrometry_ccd_frame(sim["astro"]))
    ph = E.photometry_from_long(sim["phot"], source_id=int(sim["astro"]["source_id"].iloc[0]))
    j = P.prepare_joint(E.join_photometry_astrometry(ph, tr))
    nb = sim["neighbours"].copy()
    nb["dra_mas"], nb["ddec_mas"] = nb["ddec_mas"], -nb["dra_mas"]      # rotate 90 deg
    r = P.fit_photocentre(j, target_flux_g=sim["target_flux_g"], neighbours=nb)
    assert r["verdict"] == "BLEND_UNRESOLVED"


def test_simulated_scene_control_passes(conf):
    rep = C.simulated_scene_control(n_seeds=2, cfg=P.PhotocentreConfig.from_dict(conf.get("photocentre")))
    assert rep["gate"] == "PASS", [r for r in rep["rows"] if not r["ok"]]


# ---------------------------------------------------------------------------
# controls logic
# ---------------------------------------------------------------------------
def _trials(rec_rate, dust_grey_rate, n=100):
    rows = []
    for i in range(n):
        rows.append({"kind": "grey", "snr": 30.0, "pre_existing_event": False,
                     "recovered": True, "tier": "A" if i < rec_rate * n else "C", "grey_class": "GREY"})
        rows.append({"kind": "dust", "snr": 30.0, "pre_existing_event": False, "recovered": True,
                     "tier": "C", "grey_class": "GREY" if i < dust_grey_rate * n else "CHROMATIC_BLUE"})
    return rows


def test_photometric_gate_pass_and_fail():
    assert C.photometric_gate(_trials(0.95, 0.0))["gate"] == "PASS"
    assert C.photometric_gate(_trials(0.5, 0.0))["gate"] == "FAIL"
    assert C.photometric_gate(_trials(0.95, 0.3))["gate"] == "FAIL"
    assert C.photometric_gate(_trials(0.95, 0.0), reader_mismatched_rows=3)["gate"] == "FAIL"
    assert C.photometric_gate([])["gate"] == "FAIL"


def _varstro_pop(n, rng, excess_blend, excess_iso, ecl):
    g = rng.uniform(13, 17.5, n)
    ipd = np.where(rng.random(n) < 0.3, rng.uniform(10, 40, n), 0.0)
    amp = rng.uniform(0.05, 0.3, n) if ecl else rng.uniform(0.001, 0.01, n)
    base = 1.0 + 0.05 * rng.standard_normal(n) + np.where(ipd > 0, 0.3, 0.0)
    if ecl:
        base = base + np.where(ipd > 0, excess_blend * amp / 0.3, excess_iso * amp / 0.3)
    return pd.DataFrame({"phot_g_mean_mag": g, "ipd_frac_multi_peak": ipd, "ruwe": base,
                         "astrometric_excess_noise": base - 0.9, "phot_g_n_obs": 400,
                         "phot_g_mean_flux_over_error": np.sqrt(400) / amp})


def test_varstrometry_control_pass_and_fail():
    rng = np.random.default_rng(1)
    q = _varstro_pop(4000, rng, 0, 0, False)
    good = _varstro_pop(3000, rng, 1.0, 0.02, True)
    assert C.varstrometry_control(good, q)["gate"] == "PASS"
    flat = _varstro_pop(3000, rng, 0.0, 0.0, True)
    assert C.varstrometry_control(flat, q)["gate"] == "FAIL"
    assert C.varstrometry_control(good.head(10), q)["gate"] == "DEGRADED"


def test_eclipsing_binary_control_counts():
    ev = pd.DataFrame({"source_id": [1, 1, 2], "kind": "DIP", "grey_class": "GREY",
                       "period_class": ["PERIODIC", "PERIODIC", "APERIODIC"],
                       "n_dip_episodes": [5, 5, 3], "tier": ["C", "C", "A"]})
    vari = pd.DataFrame({"source_id": [1, 2], "in_vari_eclipsing_binary": [True, True]})
    r = C.eclipsing_binary_control(ev, vari)
    assert r["n_ecl_with_grey_dip"] == 2 and r["frac_testable_caught_periodic"] == 0.5
    assert r["n_reaching_tier_A"] == 1


# ---------------------------------------------------------------------------
# honest degradation / inert DR4
# ---------------------------------------------------------------------------
def _dead_http(*a, **k):
    raise RuntimeError("no egress")


def _dead_tap(*a, **k):
    raise RuntimeError("no egress")


def test_dr4_probe_not_released():
    tap = lambda q, **k: pd.DataFrame({"schema_name": ["gaiadr3", "gaiadr2", "gaiafpr"]})  # noqa: E731
    http = lambda url, **k: (404, None)  # noqa: E731
    assert acq.probe_dr4(tap=tap, http=http)["verdict"] == "DR4_NOT_RELEASED"


def test_dr4_probe_available():
    def tap(q, **k):
        if "tap_schema.tables" in q:
            return pd.DataFrame({"schema_name": ["gaiadr4"], "table_name": ["gaiadr4.epoch_astrometry"]})
        return pd.DataFrame({"schema_name": ["gaiadr3", "gaiadr4"]})

    rep = acq.probe_dr4(tap=tap, http=lambda url, **k: (200, None))
    assert rep["verdict"] == "DR4_AVAILABLE"
    assert rep["checks"]["tap_schemas"]["dr4_tables"] == ["gaiadr4.epoch_astrometry"]


def test_dr4_probe_unreachable():
    assert acq.probe_dr4(tap=_dead_tap, http=_dead_http)["verdict"] == "ARCHIVE_UNREACHABLE"


def test_dr4_stage_is_inert_before_release(tmp_path, conf):
    tap = lambda q, **k: pd.DataFrame({"schema_name": ["gaiadr3"]})  # noqa: E731
    rep = R.stage_dr4(conf, tmp_path, tap=tap, http=lambda url, **k: (404, None))
    assert rep["verdict"] == "DR4_NOT_RELEASED"
    assert json.loads((tmp_path / "dr4_status.json").read_text())["verdict"] == "DR4_NOT_RELEASED"
    assert not (tmp_path / "dr4_photocentre.csv").exists()


def test_probe_with_no_archive_says_so(tmp_path, conf):
    rep = R.stage_probe(conf, tmp_path, http=_dead_http, tap=_dead_tap)
    assert rep["verdict"] == "NO_DATA_REACHED"
    assert rep["dr4"]["verdict"] == "ARCHIVE_UNREACHABLE"


def test_controls_with_no_archive_fail_the_photometric_gate(tmp_path, conf):
    c = dict(conf)
    c["controls"] = {**conf["controls"], "pilot_files": 1}
    rep = R.stage_controls(c, tmp_path, http=_dead_http, tap=_dead_tap, prerelease_zip=PRE)
    assert rep["gate_photometric"] == "FAIL"
    assert rep["A1_simulated_scenes"]["gate"] == "PASS"
    assert rep["A2_prerelease"]["n_parallax_ok"] == 12     # the fixture still validates the reader
    assert rep["A3_varstrometry"]["gate"] == "DEGRADED"
    # and the sweep refuses to run on failed controls
    sw = R.stage_sweep(c, tmp_path, shard=0, n_shards=2, http=_dead_http)
    assert sw["verdict"] == "REFUSED_CONTROLS_NOT_PASSED"
    assert R.stage_reduce(c, tmp_path, n_shards_expected=2)["verdict"] == "REFUSED_CONTROLS_NOT_PASSED"
    assert R.stage_summary(c, tmp_path)["verdict"] == "CONTROLS_NOT_PASSED"


def test_reduce_and_summary_with_nothing(tmp_path, conf):
    assert R.stage_reduce(conf, tmp_path)["verdict"] == "NO_SHARD_OUTPUTS"
    assert R.stage_summary(conf, tmp_path)["verdict"] == "NO_CONTROLS_RUN"


# ---------------------------------------------------------------------------
# end to end, offline: sweep -> reduce -> vet on synthetic CDN files
# ---------------------------------------------------------------------------
def _write_cdn_files(tmp_path: Path, n_files=2, per_file=12):
    names = []
    sid = 5_000_000_000_000_000
    kinds = ["quiet", "grey_multi", "dust_multi", "eb", "variable", "g_only"]
    for f in range(n_files):
        frames = []
        for k in range(per_file):
            sid += 1
            frames.append(SIM.simulate_grey_lightcurve(kinds[k % len(kinds)], seed=1000 * f + k,
                                                       source_id=sid))
        name = f"EpochPhotometry_{f:06d}-{f:06d}.csv.gz"
        with gzip.open(tmp_path / name, "wt") as fh:
            fh.write(SIM.to_wide_csv_text(pd.concat(frames, ignore_index=True)))
        names.append(name)
    return names


def test_offline_sweep_reduce_vet(tmp_path, conf):
    data = tmp_path / "cdn"
    data.mkdir()
    names = _write_cdn_files(data)
    out = tmp_path / "out"
    out.mkdir()
    (out / "controls.json").write_text(json.dumps({"gate_photometric": "PASS"}))
    c = dict(conf)
    c["sweep"] = {**conf["sweep"], "inject_every": 5}
    for i in range(2):
        rep = R.stage_sweep(c, out, shard=i, n_shards=2, files_override=names, local_dir=data)
        assert rep["verdict"] == "SHARD_COMPLETE"
    red = R.stage_reduce(c, out, n_shards_expected=2)
    assert red["n_files_done"] == 2 and red["counts"]["n_searched"] == 24
    assert red["n_tier_A_after_epoch"] >= 4          # the grey_multi stars (4 of 24)
    ab = pd.read_csv(out / "events_AB.csv")
    tierA = ab[ab["tier"] == "A"]
    # only grey_multi sources reach tier A: the dust, EB, G-only and variable ones never do
    assert set((tierA["source_id"] - 5_000_000_000_000_001) % 6) == {1}

    # vet with stubbed Gaia: one ECL, one duplicated, one whose flux matches a neighbour
    ids = sorted(tierA["source_id"].unique())

    def tap(q, **k):
        if "nss_vim_fl" in q:
            return pd.DataFrame({"source_id": [ids[3]]})
        if "vari_classifier_result" in q:
            return pd.DataFrame({"source_id": [ids[0]], "best_class_name": ["ECL"],
                                 "best_class_score": [0.9]})
        if "vari_eclipsing_binary" in q:
            return pd.DataFrame(columns=["source_id", "frequency", "global_ranking"])
        if "CONTAINS" in q:
            ra = float(q.split("CIRCLE('ICRS', ")[1].split(",")[0])
            sid = int(round((ra - 10.0) * 1000))
            if sid == ids[2] - ids[0]:
                row = tierA[tierA["source_id"] == ids[2]].iloc[0]
                f_nb = float(row["med_flux_g"]) * (1 + float(row["delta_g"]))
                return pd.DataFrame({"source_id": [1, 2], "ra": [ra, ra], "dec": [0, 0.001],
                                     "phot_g_mean_mag": [15, 15.2], "phot_g_mean_flux": [1.0, f_nb],
                                     "sep_arcsec": [0.0, 3.6]})
            return pd.DataFrame({"source_id": [1], "ra": [ra], "dec": [0.0], "phot_g_mean_mag": [15.0],
                                 "phot_g_mean_flux": [1.0], "sep_arcsec": [0.0]})
        # gaia_source rows
        return pd.DataFrame({"source_id": ids, "ra": [10.0 + (s - ids[0]) / 1000 for s in ids],
                             "dec": 0.0, "phot_g_mean_mag": 15.0, "bp_rp": 1.0, "ruwe": 1.0,
                             "ipd_frac_multi_peak": 0.0,
                             "duplicated_source": [False, True] + [False] * (len(ids) - 2)})

    vet = R.stage_vet(c, out, tap=tap)
    v = pd.read_csv(out / "vetted.csv")
    kr = dict(zip(v["source_id"], v["kill_reasons"].fillna(""), strict=False))
    assert "GAIA_CLASS_ECL" in kr[ids[0]]
    assert "DUPLICATED_SOURCE" in kr[ids[1]]
    assert "FLUX_MATCHES_NEIGHBOUR" in kr[ids[2]]
    assert "GAIA_VIM_BLEND" in kr[ids[3]]
    assert vet["classes"].get("KILLED") == 4
    summ = R.stage_summary(c, out)
    assert summ["self_consistency"]["ok"]


def test_epoch_cluster_veto():
    h_tr = np.full(100, 1000.0)
    h_ev = np.full(100, 1.0)
    h_ev[37] = 60.0
    bad = R.epoch_clusters(h_tr, h_ev)
    assert bad[37] and bad.sum() == 1


def test_sweep_only_uploads_its_own_files(tmp_path, conf):
    """Shard hygiene: shard i writes only *_s{i}of{n}* files."""
    data = tmp_path / "cdn"
    data.mkdir()
    names = _write_cdn_files(data, n_files=2, per_file=3)
    out = tmp_path / "o"
    out.mkdir()
    (out / "controls.json").write_text(json.dumps({"gate_photometric": "PASS"}))
    R.stage_sweep(conf, out, shard=1, n_shards=3, files_override=names, local_dir=data)
    written = {p.name for p in out.iterdir()} - {"controls.json"}
    assert written and all("_s1of3" in n for n in written), written


# ---------------------------------------------------------------------------
# watchlist
# ---------------------------------------------------------------------------
def test_watchlist_collects_every_channel(tmp_path):
    (tmp_path / "results" / "alpha").mkdir(parents=True)
    (tmp_path / "results" / "beta").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    pd.DataFrame({"source_id": [6225457312033584384, 123], "cradle_class": ["CANDIDATE", "X"]}).to_csv(
        tmp_path / "results" / "alpha" / "candidates.csv", index=False)
    (tmp_path / "results" / "beta" / "shortlist.json").write_text(json.dumps(
        {"rows": [{"gaia_source_id": 4497414466452138496, "verdict": "interest"}]}))
    pd.DataFrame({"source_id": [5]}).to_csv(tmp_path / "results" / "beta" / "parent_s0of4.csv",
                                            index=False)
    (tmp_path / "STATUS.md").write_text("## Channel X\nthe star Gaia DR3 2053563953175632768 and "
                                        "GALAH 170203001601307 and run 35741356662\n")
    df, st = W.build_watchlist(tmp_path)
    u = W.unique_watchlist(df)
    ids = set(u["source_id"])
    assert {6225457312033584384, 4497414466452138496, 2053563953175632768} <= ids
    assert 123 not in ids and 5 not in ids and 170203001601307 not in ids
    assert int(u.loc[u["source_id"] == 2053563953175632768, "priority"].iloc[0]) == 1


def test_dr4_control_gate():
    ctrl = pd.DataFrame({"source_id": range(40), "role": ["VIM_POSITIVE"] * 20 + ["SINGLE_EB_NEGATIVE"] * 20})
    good = pd.DataFrame({"source_id": range(40),
                         "verdict": ["BLEND_UNRESOLVED"] * 18 + ["AMBIGUOUS"] * 2 + ["ON_TARGET"] * 20})
    assert C.dr4_control_gate(good, ctrl)["gate"] == "PASS"
    bad = good.copy()
    bad.loc[20:30, "verdict"] = "BLEND_NEIGHBOUR"
    assert C.dr4_control_gate(bad, ctrl)["gate"] == "FAIL"
    few = good.copy()
    few.loc[:15, "verdict"] = "INSUFFICIENT"
    assert C.dr4_control_gate(few, ctrl)["gate"] == "INSUFFICIENT_CONTROLS"
    assert C.dr4_control_gate(good, pd.DataFrame())["gate"] == "NO_CONTROLS"


def test_dr4_controls_list_is_built_from_gaia_vims_and_clean_ebs(tmp_path):
    def tap(q, **k):
        if "nss_vim_fl" in q:
            return pd.DataFrame({"SOURCE_ID": [1, 2], "VIM_D_RA": [1.0, 2.0], "VIM_D_DEC": [0.5, 0.1]})
        return pd.DataFrame({"source_id": [3, 4, 5], "ra": [1.0, 2, 3], "dec": [0.0, 0, 0],
                             "phot_g_mean_mag": [10.0, 11, 12], "ruwe": [1.0, 1.0, 1.0],
                             "duplicated_source": [False, True, False]})

    rep = R.build_dr4_controls(tmp_path, tap=tap)
    d = pd.read_csv(tmp_path / "dr4_controls.csv")
    assert rep["n_vim"] == 2 and rep["n_single_eb"] == 2          # the duplicated one is dropped
    assert set(d.loc[d["role"] == "VIM_POSITIVE", "source_id"]) == {1, 2}
    assert "vim_d_ra" in d.columns
    assert R.build_dr4_controls(tmp_path / "x", tap=_dead_tap)["status"] == "UNREACHABLE"


def test_dr4_probe_is_not_fooled_by_an_empty_cdn_directory():
    """Run 36007202395 (2026-09-24): the CDN `gdr4/` directory answered 200
    ten weeks before release and the probe called DR4 available.  A 200 with
    no epoch products in the listing is not DR4."""
    tap = lambda q, **k: pd.DataFrame({"schema_name": ["gaiadr3", "gaiafpr"]})  # noqa: E731
    body = b'<a href="../">../</a><a href="README.txt">README.txt</a>'
    rep = acq.probe_dr4(tap=tap, http=lambda url, **k: (200, body))
    assert rep["verdict"] == "DR4_NOT_RELEASED"
    body2 = b'<a href="Astrometry/">Astrometry/</a><a href="epoch_photometry/">epoch_photometry/</a>'
    assert acq.probe_dr4(tap=tap, http=lambda url, **k: (200, body2))["verdict"] == "DR4_AVAILABLE"

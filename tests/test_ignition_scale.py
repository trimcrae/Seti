"""Offline tests for IGNITION's acquire fix and all-sky scale-up.

No network.  What run 35039105536 taught is asserted here: the upload ladder
records the rung that answered and falls through the others; a chunk no rung
answered goes to per-star cones instead of being lost; upload chunks shrink
toward the ecliptic poles; the tiling covers the sky exactly once; the sweep
stage checkpoints tile by tile and resumes; the assess stage reads only this
run's shards and reports the sky fraction covered; and a survey-wide
zero-point drift is removed by the ensemble correction while a genuine riser
among constant neighbours survives it.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from seti.ignition.acquire import (
    EpochStore,
    acquire_stars,
    ecliptic_latitude_deg,
    expected_rows_per_star,
    fetch_neowise_upload,
    group_by_star,
    transport_order,
    upload_chunks,
)
from seti.ignition.ensemble import apply_ensemble, ensemble_offsets, summarise_offsets
from seti.ignition.run import (
    ignition_run,
    load_ignition_config,
    screen_epochs,
    stage_assess,
    stage_sweep,
)
from seti.ignition.tiles import (
    box_area_deg2,
    cover_radius_deg,
    galactic_latitude_deg,
    owns,
    sky_tiles,
    tile_grid,
    tiles_for_shard,
)
from seti.vigil.acquire import QueryResult
from test_ignition import (
    _asu_dead,
    _cone_factory,
    _frames,
    _gaia_rows,
    _irsa_dead,
)


# --------------------------------------------------------------------------
# The upload ladder
# --------------------------------------------------------------------------
def _rows_at(stars: pd.DataFrame, seed0: int = 0, ramp: float = 0.0) -> pd.DataFrame:
    frames = []
    for i, (_, s) in enumerate(stars.iterrows()):
        d = _frames(seed=seed0 + i, ramp=ramp)
        d["ra"] = float(s["ra"]) + d["ra"] - 266.0
        d["dec"] = float(s["dec"]) + d["dec"] - 65.0
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def test_upload_ladder_falls_through_and_records_the_rung():
    stars = _gaia_rows(3)
    calls: list[str] = []

    def dead(q, tbl, timeout_s, **_k):
        calls.append("pyvo_sync")
        raise RuntimeError("Query Error: QUERY must be set")

    def alive(q, tbl, timeout_s, **_k):
        calls.append("http_sync")
        assert "TAP_UPLOAD.pos" in q and len(tbl) == 3
        return _rows_at(stars)

    r = fetch_neowise_upload(stars, transports={"pyvo_sync": dead, "http_sync": alive})
    assert r.status == "OK" and r.label == "neowise_upload[http_sync]_3"
    assert calls == ["pyvo_sync", "http_sync"]
    assert "QUERY must be set" in r.error           # the failed rung's error is kept
    r2 = fetch_neowise_upload(stars, transports={"pyvo_sync": dead})
    assert r2.status == "QUERY_FAILED" and "pyvo_sync" in r2.error
    assert r2.label == "neowise_upload[none]_3"
    # The probe's answer goes first on the next call.
    assert transport_order("gator")[0] == "gator" and transport_order(None)[0] == "pyvo_sync"
    assert set(transport_order("http_async")) == set(transport_order(None))


def test_uploaded_table_carries_no_unicodechar_column():
    """Run 35653615329's probe: every rung refused with the SAME server error.

    ``INTERNAL_SERVER_ERROR: Unimplemented data type: unicodeChar`` from IRSA's
    own TAP on all four rungs is not four transport failures, it is one column
    type: ``Table.from_pandas`` on ``source_id.astype(str)`` gives a numpy
    ``<U19`` column that astropy serialises as VOTable ``unicodeChar``.  The
    uploaded ``sid`` must be a ``long`` (or, failing that, ASCII ``char``), and
    the serialised VOTable must not contain the word at all.
    """
    from astropy.table import Table

    from seti.ignition.acquire import _ascii_string_columns, _upload_table, _votable_bytes

    stars = _gaia_rows(4)
    tbl, _rad = _upload_table(stars, 2.5)
    assert tbl["sid"].dtype.kind == "i"                      # a long, not a string
    assert list(tbl["sid"]) == [int(s) for s in stars["source_id"]]
    xml = _votable_bytes(tbl).decode("utf-8", "replace")
    assert "unicodeChar" not in xml
    assert 'datatype="long"' in xml

    # A non-numeric id still goes up, as ASCII `char` -- never as unicodeChar.
    odd = stars.copy()
    odd["source_id"] = [f"NAME-{i}" for i in range(len(odd))]
    tbl2, _ = _upload_table(odd, 2.5)
    assert tbl2["sid"].dtype.kind == "S"
    assert "unicodeChar" not in _votable_bytes(tbl2).decode("utf-8", "replace")

    # And the belt-and-braces guard converts any unicode column a caller adds.
    t3 = Table({"lbl": np.array(["a", "bb"], dtype="U8"), "x": np.array([1.0, 2.0])})
    assert _ascii_string_columns(t3)["lbl"].dtype.kind == "S"
    assert "unicodeChar" not in _votable_bytes(t3).decode("utf-8", "replace")


def test_uws_job_url_is_found_when_there_is_no_location_header():
    """IRSA answered the async submission ``200`` with no ``Location``."""
    from seti.ignition.acquire import IRSA_TAP, _uws_job_url

    class R:
        def __init__(self, headers, text, url=""):
            self.headers, self.text, self.url = headers, text, url

    assert _uws_job_url(R({"Location": f"{IRSA_TAP}/async/abc123/"}, "")) == \
        f"{IRSA_TAP}/async/abc123"
    body = '<uws:job xmlns:uws="x"><uws:jobId>j42</uws:jobId></uws:job>'
    assert _uws_job_url(R({}, body)) == f"{IRSA_TAP}/async/j42"
    href = f'<a xlink:href="{IRSA_TAP}/async/j43">x</a>'
    assert _uws_job_url(R({}, href)) == f"{IRSA_TAP}/async/j43"
    assert _uws_job_url(R({}, "no job here", url=f"{IRSA_TAP}/async?QUERY=x")) is None


def test_pyvo_async_is_on_the_ladder_after_the_rung_the_server_parsed():
    from seti.ignition.acquire import _TRANSPORT_FNS, UPLOAD_TRANSPORTS

    assert "pyvo_async" in UPLOAD_TRANSPORTS
    assert set(UPLOAD_TRANSPORTS) == set(_TRANSPORT_FNS)
    order = list(UPLOAD_TRANSPORTS)
    # The two rungs whose requests IRSA actually parsed come before the two
    # that never reached the upload table at all.
    assert order.index("pyvo_sync") < order.index("gator")
    assert order.index("pyvo_async") < order.index("gator")


def test_upload_chunk_falls_back_to_cones_when_no_rung_answers(tmp_path):
    stars = _gaia_rows(5)
    cone_calls = {"n": 0}
    real = _cone_factory("constant")

    def cone(*a, **k):
        cone_calls["n"] += 1
        return real(*a, **k)

    def upload_dead(sub, radius_arcsec=2.5, **_k):
        return QueryResult(label="neowise_upload[none]_5", service="irsa",
                           status="QUERY_FAILED", error="every rung refused")

    store = EpochStore.open(tmp_path, "s0of1")
    roll = acquire_stars(stars, store, {"upload_chunk": 5, "cone_workers": 2},
                         route="upload", upload_fn=upload_dead, cone_fn=cone)
    assert roll["n_ok"] == 5 and roll["n_failed"] == 0
    assert roll["n_fallback_cone"] == 5 and cone_calls["n"] == 5
    # And with the fallback switched off the chunk is counted as failed, not lost silently.
    store2 = EpochStore.open(tmp_path, "s1of1")
    roll2 = acquire_stars(stars, store2, {"upload_chunk": 5, "upload_fallback_cone": False},
                          route="upload", upload_fn=upload_dead, cone_fn=cone)
    assert roll2["n_failed"] == 5 and roll2["n_ok"] == 0


def test_truncated_upload_chunk_is_split_not_kept():
    stars = _gaia_rows(12)
    sizes: list[int] = []

    def upload(sub, radius_arcsec=2.5, transport=None, timeout_s=0.0, max_rows=None):
        sizes.append(len(sub))
        d = _rows_at(sub)
        # A full-size chunk "hits the cap"; halves do not.
        trunc = len(sub) == 12
        return QueryResult(label=f"neowise_upload[http_sync]_{len(sub)}", service="irsa",
                           status="OK", n_rows=len(d), truncated=trunc, data=d)

    import tempfile
    from pathlib import Path

    store = EpochStore.open(Path(tempfile.mkdtemp()), "s0of1")
    roll = acquire_stars(stars, store, {"upload_chunk": 12, "upload_target_rows": 10**9},
                         route="upload", upload_fn=upload)
    assert sizes == [12, 6, 6]
    assert roll["n_ok"] == 12
    assert any(e.get("label") == "upload_truncated_split" for e in store.ledger)


def test_upload_chunks_shrink_toward_the_ecliptic_poles():
    # The NEP (RA 270, Dec +66.56) has ~4,000 exposures per source; the ecliptic ~260.
    assert ecliptic_latitude_deg(270.0, 66.56) == pytest.approx(90.0, abs=0.1)
    assert ecliptic_latitude_deg(0.0, 0.0) == pytest.approx(0.0, abs=0.1)
    assert expected_rows_per_star(270.0, 66.5) > 10 * expected_rows_per_star(0.0, 0.0)
    n = 400
    nep = pd.DataFrame({"source_id": [str(i) for i in range(n)],
                        "ra": 268.0 + 0.001 * np.arange(n), "dec": 65.0 + 0.001 * np.arange(n)})
    eq = nep.assign(ra=10.0 + 0.001 * np.arange(n), dec=0.001 * np.arange(n))
    c_nep = upload_chunks(nep, chunk_max=200, target_rows=300_000)
    c_eq = upload_chunks(eq, chunk_max=200, target_rows=300_000)
    assert max(len(c) for c in c_nep) < 100 < max(len(c) for c in c_eq) <= 200
    assert sum(len(c) for c in c_nep) == n and sum(len(c) for c in c_eq) == n
    assert set(pd.concat(c_nep)["source_id"]) == set(nep["source_id"])


def test_group_by_star_is_exact_at_high_declination():
    """A flat (ra cos dec, dec) tree with one cos dec is 10 % off in RA across a
    4-degree tile at dec 60; the unit-vector match is not."""
    stars = pd.DataFrame({"source_id": ["a", "b"], "ra": [100.0, 103.5], "dec": [58.5, 61.5],
                          "pmra": [0.0, 0.0], "pmdec": [0.0, 0.0]})
    rows = _rows_at(stars)
    # Push star b's rows 2.0" east in true angle: still inside a 2.5" cone.
    mask = rows["dec"] > 60.0
    rows.loc[mask, "ra"] += 2.0 / 3600.0 / np.cos(np.radians(61.5))
    g = group_by_star(rows, stars, tol_arcsec=2.5)
    assert set(g) == {"a", "b"}
    assert len(g["b"]) == int(mask.sum())
    # And 3.5" east is outside it.
    rows.loc[mask, "ra"] += 1.5 / 3600.0 / np.cos(np.radians(61.5))
    g2 = group_by_star(rows, stars, tol_arcsec=2.5)
    assert set(g2) == {"a"}


def test_concurrent_cones_fetch_every_star_once(tmp_path):
    stars = _gaia_rows(9)
    seen: list[float] = []
    real = _cone_factory("constant")

    def cone(ra, dec, pmra=0.0, pmdec=0.0, radius_arcsec=2.5):
        seen.append(ra)
        return real(ra, dec, pmra, pmdec, radius_arcsec=radius_arcsec)

    store = EpochStore.open(tmp_path, "s0of1")
    roll = acquire_stars(stars, store, {"cone_workers": 3, "checkpoint_every": 4},
                         route="cone", cone_fn=cone)
    assert roll["n_ok"] == 9 and len(store.done) == 9
    assert sorted(seen) == sorted(stars["ra"].tolist())
    ep = pd.read_csv(tmp_path / "epochs_s0of1.csv", dtype={"source_id": str})
    assert ep["source_id"].nunique() == 9


# --------------------------------------------------------------------------
# The tiling
# --------------------------------------------------------------------------
def test_tile_grid_covers_the_sky_exactly_once():
    g = tile_grid({"dec_step_deg": 6.0})
    assert g["area_deg2"].sum() == pytest.approx(41252.96, rel=1e-3)
    rng = np.random.default_rng(1)
    ra = rng.uniform(0, 360, 2000)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 2000)))
    counts = np.zeros(2000, int)
    for _, t in g.iterrows():
        counts += owns(t, ra, dec).astype(int)
    assert (counts == 1).all()
    # The covering cone really covers the box: every corner is inside it.
    for _, t in g.head(40).iterrows():
        rc, dc, rad = cover_radius_deg(t["ra0"], t["ra1"], t["dec0"], t["dec1"])
        for ra_c, dec_c in ((t["ra0"], t["dec0"]), (t["ra1"], t["dec1"]),
                            (t["ra0"], t["dec1"]), (t["ra1"], t["dec0"])):
            from seti.ignition.tiles import angular_sep_deg
            assert float(angular_sep_deg(rc, dc, ra_c, dec_c)) <= rad
    assert g["order"].nunique() == len(g)      # a permutation, not a sweep from one pole
    assert box_area_deg2(0.0, 360.0, -90.0, 90.0) == pytest.approx(41252.96, rel=1e-3)


def test_plane_tiles_are_skipped_and_shards_partition_the_rest():
    g = tile_grid({"dec_step_deg": 4.0, "abs_b_min_deg": 15.0})
    sky = sky_tiles({"dec_step_deg": 4.0, "abs_b_min_deg": 15.0})
    assert 0 < len(sky) < len(g)
    # A tile wholly inside the plane: the Galactic centre's box.
    gc_ra, gc_dec = 266.4, -28.9
    assert float(galactic_latitude_deg(gc_ra, gc_dec)) == pytest.approx(0.0, abs=0.2)
    gc = g[[bool(owns(t, gc_ra, gc_dec)) for _, t in g.iterrows()]]
    assert len(gc) == 1 and bool(gc["in_plane"].iloc[0])
    assert gc["tile"].iloc[0] not in set(sky["tile"])
    # The NEP box is swept (|b| ~ 30).
    nep = g[[bool(owns(t, 270.0, 66.5)) for _, t in g.iterrows()]]
    assert nep["tile"].iloc[0] in set(sky["tile"])
    parts = [tiles_for_shard(sky, i, 5) for i in range(5)]
    assert sum(len(p) for p in parts) == len(sky)
    assert len(set.union(*[set(p["tile"]) for p in parts])) == len(sky)
    # Roughly a quarter of the sky is |b| < 15; the swept area is the rest.
    assert 0.68 < sky["area_deg2"].sum() / 41253.0 < 0.80


# --------------------------------------------------------------------------
# The ensemble zero-point correction
# --------------------------------------------------------------------------
def _epoch_table(n_stars=30, drift=0.0, riser=None, seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n_stars):
        # NEOWISE's real cadence: ~183 d apart with jitter (exactly two samples a
        # year would make the scan-band sinusoid degenerate on every star).
        t = 2014.0 + 0.5 * np.arange(20) + r.normal(0.0, 0.02, 20)
        for band, m0 in (("W1", 10.0 + 0.1 * i), ("W2", 9.95 + 0.1 * i)):
            mag = m0 + drift * (t - t[0]) / 10.0 + r.normal(0, 0.008, t.size)
            if riser is not None and i == riser:
                mag = mag - 0.3 * (t - t[0]) / 10.0
            for k in range(t.size):
                rows.append({"source_id": str(i), "band": band, "epoch": k, "t_yr": t[k],
                             "mag": mag[k], "err": 0.008, "n_exp": 12})
    return pd.DataFrame(rows)


def test_survey_drift_is_removed_and_a_real_riser_survives():
    conf = load_ignition_config()
    conf["sensitivity"]["max_stars"] = 3
    # A +0.06 mag/decade common FADING drift on 30 constant stars, one true riser.
    ep = _epoch_table(drift=0.06, riser=7)
    off = ensemble_offsets(ep, bin_yr=0.25, min_stars=8)
    assert set(off) == {"W1", "W2"}
    s = summarise_offsets(off)
    assert s["W1"]["first_to_last_mag"] == pytest.approx(0.06 * 9.5 / 10.0, abs=0.008)
    corrected = apply_ensemble(ep, off)
    assert corrected["ensemble_applied"].all()
    df, rep = screen_epochs(ep, conf)
    assert rep["ensemble"]["applied"] and rep["ensemble"]["drift"]["W1"]["n_applied"] >= 20
    by = dict(zip(df["source_id"], df["screen_verdict"], strict=False))
    assert by["7"] == "IGNITION_CANDIDATE"
    assert sum(v == "FADING" for v in by.values()) == 0
    assert rep["ensemble"]["raw_two_band_fading_5sigma"] >= 25   # what the raw slope said
    assert (df.loc[df["source_id"] != "7", "w1_slope_raw_sigma"] < -5).mean() > 0.8
    # With the correction off, the same table is a field of FADING stars.
    conf_raw = dict(conf)
    conf_raw["screen"] = {"ensemble_correct": False}
    df_raw, rep_raw = screen_epochs(ep, conf_raw)
    assert rep_raw["screen_counts"].get("FADING", 0) >= 25
    assert not rep_raw["ensemble"]["applied"]


def test_ensemble_needs_enough_stars_and_leaves_thin_bins_alone():
    ep = _epoch_table(n_stars=5, drift=0.04)
    off = ensemble_offsets(ep, bin_yr=0.25, min_stars=8)
    assert not any(off["W1"]["applied"])
    corrected = apply_ensemble(ep, off)
    assert not corrected["ensemble_applied"].any()
    assert np.allclose(corrected["mag"], corrected["mag_raw"])


# --------------------------------------------------------------------------
# The sweep stage and its aggregate
# --------------------------------------------------------------------------
_CIRCLE = re.compile(r"CIRCLE\('ICRS',\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)")


class _TileGaia:
    """Answers the parent query with rows around the cone it was asked for."""

    def __init__(self, n=5, fail_tiles=()):
        self.n, self.fail_tiles, self.queries = n, set(fail_tiles), []

    def __call__(self, adql):
        self.queries.append(adql)
        m = _CIRCLE.search(adql)
        ra, dec, _rad = (float(x) for x in m.groups())
        key = (round(ra, 3), round(dec, 3))
        if key in self.fail_tiles:
            raise RuntimeError("synthetic outage on this tile")
        if adql.startswith("SELECT COUNT"):
            return pd.DataFrame({"n": [self.n]})
        d = _gaia_rows(self.n)
        d["ra"] = ra + 0.01 * np.arange(self.n)
        d["dec"] = dec + 0.01 * np.arange(self.n)
        d["source_id"] = [f"{int(ra * 100)}_{int(abs(dec) * 100)}_{i}" for i in range(self.n)]
        # One row outside the box on purpose: the ownership cut must drop it.
        d.loc[self.n - 1, "ra"] = (ra + 30.0) % 360.0
        return d


def _sweep_conf():
    conf = load_ignition_config()
    conf["tiles"] = {"dec_step_deg": 30.0, "abs_b_min_deg": 15.0, "seed": 1}
    conf["sensitivity"]["max_stars"] = 3
    conf["sweep"] = {"time_budget_s": 600.0, "sample_timeout_s": 60.0, "max_tiles": 0}
    return conf


def test_sweep_stage_checkpoints_per_tile_and_resumes(tmp_path):
    conf = _sweep_conf()
    gaia = _TileGaia(n=5)
    rep = stage_sweep(conf, tmp_path, shard=0, n_shards=4, route="cone", query_fn=gaia,
                      cone_fn=_cone_factory("constant"), asu_fetch_fn=_asu_dead,
                      irsa_fetch_fn=_irsa_dead, max_tiles=3)
    assert rep["tiles_done"] == 3 and rep["tiles_failed"] == 0
    assert rep["n_parent_done_tiles"] == 12          # 4 owned of 5 returned, per tile
    assert rep["n_stars_ok"] == 12 and not rep["stopped_on_budget"]
    assert (tmp_path / "parent_s0of4.parquet").exists()
    assert (tmp_path / "sweep_s0of4.json").exists()
    assert len(gaia.queries) == 3                    # no COUNT(*) per tile: uncapped rows count
    assert all(not q.startswith("SELECT COUNT") for q in gaia.queries)
    # Resume: nothing is re-queried.
    rep2 = stage_sweep(conf, tmp_path, shard=0, n_shards=4, route="cone", query_fn=gaia,
                       cone_fn=_cone_factory("constant"), asu_fetch_fn=_asu_dead,
                       irsa_fetch_fn=_irsa_dead, max_tiles=3)
    assert rep2["tiles_done"] == 3 and rep2["tiles_new_this_run"] == 0
    assert len(gaia.queries) == 3
    # Every parent star carries its tile and the tile owns it.
    parent = pd.read_parquet(tmp_path / "parent_s0of4.parquet")
    assert parent["tile"].nunique() == 3 and len(parent) == 12


def test_sweep_stops_on_its_wall_clock_and_says_so(tmp_path):
    conf = _sweep_conf()
    rep = stage_sweep(conf, tmp_path, shard=1, n_shards=4, route="cone", query_fn=_TileGaia(),
                      cone_fn=_cone_factory("constant"), asu_fetch_fn=_asu_dead,
                      irsa_fetch_fn=_irsa_dead, time_budget_s=1e-6, max_tiles=3)
    assert rep["stopped_on_budget"] and rep["tiles_done"] == 0


def test_sweep_records_a_failed_tile_and_the_assess_reports_coverage(tmp_path):
    conf = _sweep_conf()
    from seti.ignition.tiles import tile_field

    sky = sky_tiles(conf["tiles"])
    mine = tiles_for_shard(sky, 0, 2)
    bad = tile_field(mine.iloc[1])
    gaia = _TileGaia(n=5, fail_tiles={(round(bad["ra"], 3), round(bad["dec"], 3))})
    ramp, flat = _cone_factory("ramp"), _cone_factory("constant")

    def cone(ra, dec, pmra=0.0, pmdec=0.0, radius_arcsec=2.5):
        # One riser per tile (the star at the tile centre, i = 0); the rest are
        # constant --- if every star rose alike, the ensemble correction would
        # rightly call that the survey and remove it.
        riser = abs(ra - round(ra * 2.0) / 2.0) < 1e-6
        return (ramp if riser else flat)(ra, dec, pmra, pmdec, radius_arcsec=radius_arcsec)

    for shard in (0, 1):
        ignition_run("sweep,screen", out_dir=tmp_path, shard=shard, n_shards=2, route="cone",
                     conf=conf, query_fn=gaia, cone_fn=cone,
                     asu_fetch_fn=_asu_dead, irsa_fetch_fn=_irsa_dead, max_tiles=3)
    # A stale fields-mode sample.json and a stale 8-shard screen file in the
    # checkout must not leak into a tiles-mode verdict.
    (tmp_path / "sample.json").write_text(json.dumps({"status": "OK", "n_shards_planned": 8,
                                                      "parent_count": 846, "mode": "fields"}))
    (tmp_path / "screen_s3of8.json").write_text(json.dumps({"tag": "s3of8", "n_shards": 8,
                                                            "screen_counts": {"FADING": 99}}))
    s = stage_assess(conf, tmp_path, n_shards_expected=2)
    cov = s["coverage"]
    assert cov["mode"] == "tiles" and cov["tiles_done"] == 5 and cov["tiles_failed"] == 1
    assert cov["shards_reporting"] == ["s0of2", "s1of2"]
    assert 0 < cov["sky_fraction_done"] < 1
    assert s["denominators"]["sample_mode"] == "tiles"
    assert s["denominators"]["parent_count_archive"] == cov["n_parent_done_tiles"] == 20
    assert s["shards"]["expected"] == 2 and s["shards"]["found"] == 2
    assert "tiles_failed:1" in s["degraded"]
    assert s["veto_counters"]["screen"].get("FADING", 0) == 0     # the stale s3of8 was not read
    assert s["stage_counts"]["screened"] == 20
    assert s["verdict"].endswith("IGNITION_CANDIDATES")
    assert s["stage_counts"]["rise_candidates"] == 5          # one riser per finished tile
    assert s["ensemble"]["per_shard_drift"]


def test_probe_records_the_upload_rung_and_acquire_starts_from_it(tmp_path):
    conf = load_ignition_config()
    conf["sample"]["fields"] = [{"ra": 266.0, "dec": 65.0, "radius_deg": 1.0}]
    conf["sensitivity"]["max_stars"] = 3
    from test_ignition import _FakeGaia

    seen: list[str | None] = []

    def upload(sub, radius_arcsec=2.5, transport=None, timeout_s=0.0, max_rows=None,
               ladder=True):
        seen.append(transport)
        d = _rows_at(sub)
        return QueryResult(label=f"neowise_upload[gator]_{len(sub)}", service="irsa",
                           status="OK", n_rows=len(d), data=d)

    rep = ignition_run("probe", out_dir=tmp_path, conf=conf, query_fn=_FakeGaia(n=4),
                       cone_fn=_cone_factory(), upload_fn=upload, asu_fetch_fn=_asu_dead,
                       irsa_fetch_fn=_irsa_dead)
    assert rep["neowise_route_recommended"] == "upload"
    assert rep["neowise_upload_transport"] == "gator"
    rep = ignition_run("sample,acquire", out_dir=tmp_path, conf=conf, query_fn=_FakeGaia(n=4),
                       cone_fn=_cone_factory(), upload_fn=upload, asu_fetch_fn=_asu_dead,
                       irsa_fetch_fn=_irsa_dead)
    assert rep["route"] == "upload" and rep["upload_transport_preferred"] == "gator"
    assert seen[-1] == "gator" and rep["n_ok"] == 4

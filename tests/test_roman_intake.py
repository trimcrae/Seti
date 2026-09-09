"""Offline tests for the ROMAN intake (archive discovery + product readers).

No network (tests/conftest.py enforces it): every archive call goes through a
fake session that answers from canned text, and every reader runs on
synthetic tables / arrays.  The battery covers honest degradation of the
probe (nothing reachable, simulations only, mission data present), product
classification and sharding, the table readers' role/zero-point/unit logic,
the dq reject mask, ramp timing, and cutout geometry.
"""

from __future__ import annotations

import builtins
import copy
import json
import sys
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from seti.roman import archive as A
from seti.roman import products as P
from seti.roman.schema import DQCutout, LightCurve, Ramp, Spectrum, load_roman_config

CONF = load_roman_config()


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------

class FakeResp:
    def __init__(self, status=200, text="", headers=None, content=None):
        self.status_code = status
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.headers = headers or {"content-type": "text/html"}
        self.elapsed = 0.01

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def close(self):
        pass


class FakeSession:
    """Routes by URL substring; anything unrouted raises like a dead network."""

    def __init__(self, routes=None):
        self.routes = list(routes or [])
        self.calls = []

    def _answer(self, url):
        self.calls.append(url)
        for needle, resp in self.routes:
            if needle in url:
                return resp(url) if callable(resp) else resp
        raise ConnectionError(f"no route to {url}")

    def get(self, url, timeout=None, **kw):
        return self._answer(url)

    def head(self, url, timeout=None, **kw):
        return self._answer(url)


S3_NS = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'


def s3_xml(prefixes=(), contents=(), truncated=False, token=None):
    body = [f"<ListBucketResult {S3_NS}><Name>b</Name>"]
    for p in prefixes:
        body.append(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>")
    for k, s in contents:
        body.append(f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>")
    body.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
    if token:
        body.append(f"<NextContinuationToken>{token}</NextContinuationToken>")
    body.append("</ListBucketResult>")
    return "".join(body)


def tap_json(rows, cols):
    return json.dumps({"metadata": [{"name": c} for c in cols], "data": rows})


# --------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------

def test_probe_with_dead_network_reports_no_archive_reached():
    sess = FakeSession()
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    assert rec["data_state"] == "NO_ARCHIVE_REACHED"
    assert rec["n_endpoints_reached"] == 0
    assert rec["endpoints"] and all(e["status"] == "failed" and e["error"]
                                    for e in rec["endpoints"].values())
    assert rec["data_state_evidence"]
    assert "asdf" in rec["packages"] and "importable" in rec["packages"]["asdf"]
    json.dumps(rec)   # JSON-safe record


def test_probe_with_only_a_simulation_bucket_is_simulations_only():
    listing = s3_xml(prefixes=["openuniverse2024/roman/preview/"],
                     contents=[("openuniverse2024/roman/README.txt", 1200)])
    sess = FakeSession([("nasa-irsa-simulations.s3.amazonaws.com", FakeResp(200, listing,
                                                                             {"content-type": "application/xml"}))])
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    assert rec["data_state"] == "SIMULATIONS_ONLY"
    assert any("simulation bucket" in e for e in rec["data_state_evidence"])
    ep = rec["endpoints"]["s3:openuniverse2024"]
    assert ep["status"] == "ok" and ep["prefixes"] == ["openuniverse2024/roman/preview/"]
    assert ep["contents"][0]["size"] == 1200
    assert rec["n_endpoints_reached"] == 1


def test_probe_with_a_roman_irsa_table_is_mission_data_present():
    tap = tap_json([["roman_wfi_gbtds_lightcurves", "GBTDS per-star light curves"]],
                   ["table_name", "description"])
    sess = FakeSession([("irsa.ipac.caltech.edu/TAP/sync", FakeResp(200, tap, {"content-type": "application/json"})),
                        ("irsa.ipac.caltech.edu/SIA", FakeResp(200, "<VOTABLE/>"))])
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    assert rec["data_state"] == "MISSION_DATA_PRESENT"
    assert any("roman_wfi_gbtds_lightcurves" in e for e in rec["data_state_evidence"])
    tables = rec["endpoints"]["irsa_tap"]["tables"]
    assert tables[0]["table_name"] == "roman_wfi_gbtds_lightcurves" and not tables[0]["simulated"]


def test_probe_answering_archives_with_no_roman_is_not_yet_public():
    empty_tap = tap_json([], ["table_name", "description"])
    err_xml = ('<?xml version="1.0"?><Error><Code>NoSuchBucket</Code>'
               '<Message>The specified bucket does not exist</Message></Error>')
    sess = FakeSession([("irsa.ipac.caltech.edu/TAP/sync", FakeResp(200, empty_tap)),
                        ("mast.stsci.edu", FakeResp(200, tap_json([[0]], ["n"]))),
                        ("nasa-irsa-roman.s3", FakeResp(404, err_xml)),
                        ("roman-docs.stsci.edu", FakeResp(200, "<html><a href='x.html'>x</a></html>"))])
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    assert rec["data_state"] == "NOT_YET_PUBLIC"
    assert rec["endpoints"]["s3:roman_mission"]["exists"] is False
    assert rec["endpoints"]["mast:Roman"]["count"] == 0
    assert rec["n_endpoints_reached"] >= 4


def conf_with(**archive_overrides):
    """A deep copy of the config with keys of the ``archive`` block replaced."""
    c = copy.deepcopy(CONF)
    c["archive"].update(archive_overrides)
    return c


def test_simulation_page_data_links_are_collected_and_capped():
    # the 2018 data-challenge page answered 404 on 2026-09-09 and left the config;
    # a gbtds_like page entry is still the evidence path this test exercises
    conf = conf_with(simulations=[{"name": "microlensing_data_challenge", "kind": "gbtds_like",
                                   "url": "https://roman.ipac.caltech.edu/sims/Microlensing_Data_Challenge.html"}])
    html = "".join(f"<a href='lc_{i:05d}.csv'>lc</a>" for i in range(300)) + \
        "<a href='index.html'>idx</a><a href='truth.parquet'>t</a>"
    sess = FakeSession([("Microlensing_Data_Challenge", FakeResp(200, html))])
    rec = A.probe(conf, session=sess, timeout_s=1.0)
    ep = rec["endpoints"]["sim:microlensing_data_challenge"]
    assert ep["status"] == "ok" and len(ep["data_links"]) == A.MAX_SIM_LINKS
    assert all(link.endswith(".csv") for link in ep["data_links"])
    assert ep["data_links"][0].startswith("https://roman.ipac.caltech.edu/sims/")
    assert ep["crawl"] is None                       # only ``kind: index`` pages are crawled
    assert rec["data_state"] == "SIMULATIONS_ONLY"


def test_probe_step_that_raises_is_recorded_not_propagated(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("parser exploded")
    monkeypatch.setattr(A, "_probe_irsa_tap", boom)
    rec = A.probe(CONF, session=FakeSession(), timeout_s=1.0)
    assert "probe step raised" in rec["endpoints"]["irsa_tap"]["error"]
    assert rec["data_state"] == "NO_ARCHIVE_REACHED"


def test_tap_json_parser_handles_the_dialects():
    assert A.parse_tap_json(tap_json([["a", "b"]], ["table_name", "description"])) == \
        [{"table_name": "a", "description": "b"}]
    assert A.parse_tap_json(json.dumps({"fields": [{"name": "n"}], "data": [[3]]})) == [{"n": 3}]
    assert A.parse_tap_json(json.dumps([{"n": 1}])) == [{"n": 1}]
    assert A.parse_tap_json("table_name,description\nx,y\n") == [{"table_name": "x", "description": "y"}]
    assert A.parse_tap_json("<html>nope</html>") == []


# --------------------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------------------

SYNTHETIC_KEYS = [
    # key, expected (level, kind, survey, simulated)
    ("openuniverse2024/roman/preview/RomanTDS/images/simple_model/F184/r0000101001001001001_0001_wfi01_uncal.asdf",
     ("L1", "uncal", "HLTDS", True)),
    ("gbtds/field03/F146/r0002301001001001001_0012_wfi05_cal.asdf", ("L2", "cal", "GBTDS", False)),
    ("hlwas/coadd/F158/r0009_coadd_wfi_F158_i2d.asdf", ("L3", "coadd", "HLWAS", False)),
    ("gbtds/lightcurves/F146/lc_000123456.parquet", ("L4", "lightcurve", "GBTDS", False)),
    ("bulge/field01/star_9988.lc.csv", ("L4", "lightcurve", "GBTDS", False)),
    ("hlwas/grism/x1d/r0007_wfi03_x1d.fits", ("L4", "spectrum_1d", "HLWAS", False)),
    ("hltds/prism/r0004_wfi11_cal.asdf", ("L2", "spectrum_2d", "HLTDS", False)),
    ("hlwas/catalog/cat_F129_tile0042.parquet", ("catalog", "catalog", "HLWAS", False)),
    ("cgi/target01/cgi_b3_target01_cal.fits", ("L2", "cgi", "", False)),
    ("openuniverse2024/roman/preview/RomanWAS/truth/index.txt", ("unknown", "unknown", "HLWAS", True)),
    ("sims/microlensing_challenge/lightcurve_00042.txt", ("L4", "lightcurve", "GBTDS", True)),
    ("gbtds/field02/F087/r0002_wfi02_cal.asdf", ("L2", "cal", "GBTDS", False)),
]


def test_classification_of_synthetic_keys():
    for key, (level, kind, survey, sim) in SYNTHETIC_KEYS:
        p = A.classify_product(f"s3://nasa-irsa-roman/{key}", "irsa_s3")
        assert (p.level, p.kind, p.survey, p.simulated) == (level, kind, survey, sim), key
    assert A.classify_product("s3://b/gbtds/F146/lc_1.parquet").band == "F146"
    assert A.classify_product("s3://b/hlwas/g150/x.fits").band == "G150"
    assert A.classify_product("s3://b/cgi/x_cal.fits").instrument == "CGI"
    # a simulation bucket makes everything under it simulated, whatever the path says
    assert A.classify_product("s3://x/gbtds/lc_1.parquet", simulated_hint=True).simulated is True


def test_inventory_walks_the_bucket_with_pagination_and_stamps_simulated():
    page1 = s3_xml(contents=[("openuniverse2024/roman/RomanTDS/F184/a_uncal.asdf", 10),
                             ("openuniverse2024/roman/RomanTDS/F184/a_cal.asdf", 20)],
                   truncated=True, token="tok1")
    page2 = s3_xml(contents=[("openuniverse2024/roman/RomanTDS/lightcurves/lc_1.parquet", 5),
                             ("openuniverse2024/roman/RomanWAS/x1d/s_x1d.fits", 7)])

    def s3(url):
        return FakeResp(200, page2 if "continuation-token=tok1" in url else page1)

    sess = FakeSession([("nasa-irsa-simulations.s3.amazonaws.com", s3)])
    probe = A.probe(CONF, session=sess, timeout_s=1.0)
    inv = A.inventory(CONF, probe, session=sess, max_listing=100, n_shards=3)
    assert inv["verdict"] == "INVENTORIED"
    assert inv["n_products"] == 4 and inv["n_simulated"] == 4 and inv["n_mission"] == 0
    assert inv["counts_by_kind"] == {"uncal": 1, "cal": 1, "lightcurve": 1, "spectrum_1d": 1}
    assert inv["counts_by_level"] == {"L1": 1, "L2": 1, "L4": 2}
    assert inv["listings"]["s3:openuniverse2024"]["tree"]["n_requests"] == 2
    assert inv["listings"]["s3:openuniverse2024"]["tree"]["n_dirs"] == 1
    assert inv["counts_by_format"] == {"none": 4}
    assert all(p["simulated"] for p in inv["products"])
    assert {p["survey"] for p in inv["products"]} == {"HLTDS", "HLWAS"}
    assert len(inv["shards"]) == 3 and sum(len(s) for s in inv["shards"]) == 4
    json.dumps(inv)


def test_inventory_on_a_dead_probe_is_no_data_reached():
    probe = A.probe(CONF, session=FakeSession(), timeout_s=1.0)
    inv = A.inventory(CONF, probe, session=FakeSession())
    assert inv["verdict"] == "NO_DATA_REACHED" and inv["products"] == [] and inv["shards"] == [[]]


def test_shard_planning_round_robin_by_size():
    prods = [{"uri": f"u{i}", "kind": "cal" if i % 2 else "uncal", "size_bytes": (10 - i) * 100}
             for i in range(10)]
    shards = A.plan_shards(prods, 3)
    assert [len(s) for s in shards] == [4, 3, 3]
    assert shards[0] == ["u0", "u3", "u6", "u9"]   # largest first, dealt round-robin
    only_cal = A.plan_shards(prods, 2, kinds=["cal"])
    assert sorted(sum(only_cal, [])) == ["u1", "u3", "u5", "u7", "u9"]
    assert A.plan_shards([], 4) == [[]]
    assert len(A.plan_shards(prods[:2], 8)) == 2   # never more shards than products


def test_s3_uri_translation_and_cache_path(tmp_path):
    assert A.to_https("s3://nasa-irsa-roman/a b/x_cal.asdf") == \
        "https://nasa-irsa-roman.s3.amazonaws.com/a%20b/x_cal.asdf"
    assert A.to_https("https://x/y") == "https://x/y"
    assert A.cache_path_for("s3://bkt/p/q.asdf", tmp_path) == tmp_path / "bkt.s3.amazonaws.com" / "p" / "q.asdf"


def test_fetch_to_cache_writes_provenance_and_respects_the_cap(tmp_path):
    payload = b"x" * 5000
    sess = FakeSession([("bkt.s3.amazonaws.com/p/q.asdf", FakeResp(200, "", {"content-type": "application/octet-stream"}, payload))])
    out = A.fetch_to_cache("s3://bkt/p/q.asdf", tmp_path, session=sess, pause=0.0)
    assert out is not None and out.read_bytes() == payload
    prov = json.loads((out.parent / (out.name + ".provenance.json")).read_text())
    assert prov["bytes"] == 5000 and prov["uri"] == "s3://bkt/p/q.asdf" and len(prov["sha256_first_mb"]) == 64
    n_calls = len(sess.calls)
    assert A.fetch_to_cache("s3://bkt/p/q.asdf", tmp_path, session=sess, pause=0.0) == out
    assert len(sess.calls) == n_calls            # cached, not refetched
    assert A.fetch_to_cache("s3://bkt/p/big.asdf", tmp_path, session=FakeSession(
        [("big.asdf", FakeResp(200, "", None, b"y" * 5000))]), max_bytes=1000, pause=0.0) is None
    assert not (tmp_path / "bkt.s3.amazonaws.com" / "p" / "big.asdf").exists()
    assert A.fetch_to_cache("https://nowhere/x.fits", tmp_path, session=FakeSession(), retries=2, pause=0.0) is None


# --------------------------------------------------------------------------------------
# S3 tree listing and the OpenUniverse 2024 layout (shape verified on the bucket, 2026-09-09)
# --------------------------------------------------------------------------------------

OU = "openuniverse2024/roman/"
SNANA_DIR = OU + "full/ROMAN+LSST_LARGE_SNIa-normal/"
TDS = OU + "full/RomanTDS/"
CATS = OU + "full/roman_rubin_cats_v1.1.2_faint/"

# prefix -> list of pages; each page = (prefixes, contents, truncated, next_token)
FAKE_TREE = {
    OU: [([OU + "full/", OU + "preview/"], [], False, None)],
    OU + "full/": [([SNANA_DIR, TDS, CATS], [], False, None)],
    SNANA_DIR: [
        ([], [(SNANA_DIR + "ROMAN+LSST_LARGE_SNIa-normal.DUMP", 76591943),
              (SNANA_DIR + "ROMAN+LSST_LARGE_SNIa-normal.LIST", 1140),
              (SNANA_DIR + "ROMAN+LSST_LARGE_SNIa-normal.README", 4777),
              (SNANA_DIR + "ROMAN+LSST_NONIaMODEL0-0001_HEAD.FITS.gz", 1952346),
              (SNANA_DIR + "ROMAN+LSST_NONIaMODEL0-0001_PHOT.FITS.gz", 150112285)], True, "tokA"),
        ([], [(SNANA_DIR + "ROMAN+LSST_NONIaMODEL0-0002_HEAD.FITS.gz", 1952912),
              (SNANA_DIR + "ROMAN+LSST_NONIaMODEL0-0002_PHOT.FITS.gz", 150000000)], True, "tokB"),
        ([], [(SNANA_DIR + "ROMAN+LSST_NONIaMODEL0-0003_HEAD.FITS.gz", 1950000)], False, None),
    ],
    TDS: [([TDS + "images/", TDS + "truth/"],
           [(TDS + "Roman_TDS_obseq_11_6_23.fits", 2531520),
            (TDS + "Roman_TDS_obseq_11_6_23_radec.fits", 16758720),
            (TDS + "tds.yaml", 6080), (TDS + "TDS.pdf", 900000)], False, None)],
    TDS + "images/": [([TDS + "images/simple_model/", TDS + "images/truth/"], [], False, None)],
    TDS + "images/simple_model/": [([TDS + "images/simple_model/F184/",
                                     TDS + "images/simple_model/Z087/"], [], False, None)],
    TDS + "images/simple_model/F184/": [([TDS + "images/simple_model/F184/10307/",
                                          TDS + "images/simple_model/F184/10312/",
                                          TDS + "images/simple_model/F184/10317/"], [], False, None)],
    TDS + "images/simple_model/F184/10307/": [
        ([], [(TDS + "images/simple_model/F184/10307/Roman_TDS_simple_model_F184_10307_17.fits.gz",
               40940489)], False, None)],
    TDS + "images/simple_model/Z087/": [([TDS + "images/simple_model/Z087/1062/"], [], False, None)],
    TDS + "images/simple_model/Z087/1062/": [
        ([], [(TDS + "images/simple_model/Z087/1062/Roman_TDS_simple_model_Z087_1062_3.fits.gz",
               40000000)], False, None)],
    TDS + "images/truth/": [([TDS + "images/truth/F184/"], [], False, None)],
    TDS + "images/truth/F184/": [([TDS + "images/truth/F184/10307/"], [], False, None)],
    TDS + "images/truth/F184/10307/": [
        ([], [(TDS + "images/truth/F184/10307/Roman_TDS_truth_F184_10307_17.fits.gz", 36671958)],
         False, None)],
    TDS + "truth/": [([TDS + "truth/F184/"], [], False, None)],
    TDS + "truth/F184/": [([TDS + "truth/F184/10307/"], [], False, None)],
    TDS + "truth/F184/10307/": [
        ([], [(TDS + "truth/F184/10307/Roman_TDS_index_F184_10307_17.txt", 3603459)], False, None)],
    CATS: [([], [(CATS + "galaxy_10307.parquet", 373399168),
                 (CATS + "galaxy_flux_10307.parquet", 249800659),
                 (CATS + "galaxy_sed_10307.hdf5", 14522511591),
                 (CATS + "pointsource_10307.parquet", 2083170),
                 (CATS + "pointsource_flux_10307.parquet", 1719738),
                 (CATS + "snana_10307.parquet", 5000000),
                 (CATS + "snana_10307.hdf5", 17086895832)], False, None)],
    OU + "preview/": [([OU + "preview/RomanWAS/"], [], False, None)],
    OU + "preview/RomanWAS/": [([OU + "preview/RomanWAS/images/"],
                                [(OU + "preview/RomanWAS/Roman_WAS_obseq_11_1_23.fits", 754560)],
                                False, None)],
    OU + "preview/RomanWAS/images/": [([OU + "preview/RomanWAS/images/coadds/"], [], False, None)],
    OU + "preview/RomanWAS/images/coadds/": [([OU + "preview/RomanWAS/images/coadds/F184/"], [], False, None)],
    OU + "preview/RomanWAS/images/coadds/F184/": [([OU + "preview/RomanWAS/images/coadds/F184/Row12/"], [],
                                                   False, None)],
    OU + "preview/RomanWAS/images/coadds/F184/Row12/": [
        ([], [(OU + "preview/RomanWAS/images/coadds/F184/Row12/prod_F_12_07_map.fits.gz", 450926654)],
         False, None)],
}
# a directory of 2000+ numbered pointings, served as prefix-only pages (as S3 does)
BIG = OU + "full/RomanWAS/truth/H158/"
FAKE_TREE[BIG] = [([f"{BIG}{i}/" for i in range(1000)], [], True, "pgA"),
                  ([f"{BIG}{i}/" for i in range(1000, 2000)], [], True, "pgB"),
                  ([f"{BIG}{i}/" for i in range(2000, 2583)], [], False, None)]
FAKE_TREE[BIG + "0/"] = [([], [(BIG + "0/Roman_WAS_index_H158_0_1.txt", 100)], False, None)]
FAKE_TREE[BIG + "1/"] = [([], [(BIG + "1/Roman_WAS_index_H158_1_1.txt", 101)], False, None)]
FAKE_TREE[OU + "full/RomanWAS/"] = [([OU + "full/RomanWAS/truth/"], [], False, None)]
FAKE_TREE[OU + "full/RomanWAS/truth/"] = [([BIG], [(OU + "full/RomanWAS/truth/tmp.txt", 5)], False, None)]
FAKE_TREE[OU + "full/"] = [([SNANA_DIR, TDS, OU + "full/RomanWAS/", CATS], [], False, None)]
# directories the walk with max_numeric_subdirs_per_dir=1 never lists
NOT_WALKED = {BIG + "1/"}
LISTED = set(FAKE_TREE) - NOT_WALKED


def fake_s3_tree(url):
    """Answers ListObjectsV2 like S3 does: the query string is form-decoded, so a
    raw ``+`` in the prefix becomes a space and matches nothing (empty listing)."""
    q = parse_qs(urlparse(url).query)
    prefix = (q.get("prefix") or [""])[0]
    token = (q.get("continuation-token") or [None])[0]
    assert q.get("delimiter") == ["/"], url
    pages = FAKE_TREE.get(prefix)
    if pages is None:
        return FakeResp(200, s3_xml())
    idx = 0
    if token:
        idx = next(i + 1 for i, pg in enumerate(pages) if pg[3] == token)
    prefixes, contents, truncated, nxt = pages[idx]
    return FakeResp(200, s3_xml(prefixes=prefixes, contents=contents, truncated=truncated, token=nxt))


NO_BUCKET_XML = ('<?xml version="1.0"?><Error><Code>NoSuchBucket</Code>'
                 '<Message>The specified bucket does not exist</Message></Error>')


def test_list_s3_tree_walks_a_deep_tree_records_truncation_and_encodes_plus():
    sess = FakeSession([("nasa-irsa-simulations.s3.amazonaws.com", fake_s3_tree)])
    tree = A.list_s3_tree("nasa-irsa-simulations", OU, sess, max_depth=6, max_keys_per_dir=6,
                          max_dirs=400, max_numeric_subdirs_per_dir=1, timeout_s=1.0)
    assert tree["exists"] is True and tree["error"] is None and tree["http_status"] == 200
    by_path = {d["path"]: d for d in tree["dirs"]}
    assert set(by_path) == LISTED                           # every directory of the fake was listed
    assert tree["n_dirs_listed"] == len(LISTED) and tree["max_depth_reached"] == 6
    # numbered siblings are instances: one of three pointings descended, all three recorded
    band = by_path[TDS + "images/simple_model/F184/"]
    assert (band["n_subdirs"], band["n_subdirs_numeric"], band["subdirs_descended"]) == (3, 3, 1)
    assert TDS + "images/simple_model/F184/10312/" not in by_path
    # a 2583-pointing directory: one prefix-only page is enough, and says so
    big = by_path[BIG]
    assert big["n_requests"] == 1 and big["n_subdirs"] == 1000 and big["subdirs_truncated"] is True
    assert big["truncated"] is True and big["subdirs_descended"] == 1 and BIG + "0/" in by_path
    assert by_path[OU + "full/RomanWAS/truth/"]["n_keys_listed"] == 1        # named dir + a loose file
    root = by_path[OU]
    assert root["depth"] == 0 and root["n_subdirs"] == 2 and root["n_keys_listed"] == 0
    snana = by_path[SNANA_DIR]
    assert snana["n_requests"] == 2                          # page 2 reached the 6-key cap
    assert snana["n_keys_listed"] == 6 and snana["truncated"] is True
    assert snana["total_bytes_listed"] == 76591943 + 1140 + 4777 + 1952346 + 150112285 + 1952912
    assert snana["keys"][3]["key"].endswith("NONIaMODEL0-0001_HEAD.FITS.gz")
    assert snana["keys"][3]["size"] == 1952346
    tds = by_path[TDS]
    assert tds["n_keys_listed"] == 4 and tds["subdirs"] == [TDS + "images/", TDS + "truth/"]
    assert tds["truncated"] is False
    assert by_path[TDS + "images/simple_model/F184/10307/"]["depth"] == 6
    assert tree["n_keys_listed"] == sum(d["n_keys_listed"] for d in tree["dirs"])
    cats = by_path[CATS]                                     # 7 keys, sampled at 6: truncated too
    assert cats["n_keys_listed"] == 6 and cats["truncated"] is True and cats["n_requests"] == 1
    assert tree["n_dirs_truncated"] == 3 and not tree["dirs_capped"] and not tree["depth_capped"]
    # the ``+`` directory travelled as %2B (a raw + would have listed as empty)
    plus_calls = [u for u in sess.calls if "ROMAN%2BLSST_LARGE_SNIa-normal" in u]
    assert len(plus_calls) == 2 and not any("ROMAN+LSST" in u for u in sess.calls)
    assert "continuation-token=tokA" in plus_calls[1]
    assert all("prefix=openuniverse2024/roman/" in u for u in sess.calls)   # '/' kept readable
    json.dumps(tree)


def test_list_s3_tree_honours_depth_dir_and_subdir_caps():
    sess = FakeSession([("nasa-irsa-simulations.s3.amazonaws.com", fake_s3_tree)])
    shallow = A.list_s3_tree("nasa-irsa-simulations", OU, sess, max_depth=2, timeout_s=1.0)
    assert shallow["max_depth_reached"] == 2 and shallow["depth_capped"] is True
    assert all(d["depth"] <= 2 for d in shallow["dirs"])
    assert {d["path"] for d in shallow["dirs"] if d["depth"] == 2} == {SNANA_DIR, TDS, CATS,
                                                                        OU + "full/RomanWAS/",
                                                                        OU + "preview/RomanWAS/"}
    few = A.list_s3_tree("nasa-irsa-simulations", OU, sess, max_dirs=3, timeout_s=1.0)
    # a larger numeric cap lists the empty pointings too (the fake answers an empty listing)
    more = A.list_s3_tree("nasa-irsa-simulations", OU, sess, max_numeric_subdirs_per_dir=3,
                          timeout_s=1.0)
    paths = {d["path"] for d in more["dirs"]}
    assert {TDS + "images/simple_model/F184/10312/", TDS + "images/simple_model/F184/10317/",
            BIG + "1/", BIG + "2/"} <= paths
    assert more["n_dirs_listed"] == len(LISTED) + 4
    assert few["n_dirs_listed"] == 3 and few["dirs_capped"] is True
    one_child = A.list_s3_tree("nasa-irsa-simulations", OU, sess, max_subdirs_per_dir=1,
                               timeout_s=1.0)
    root = one_child["dirs"][0]
    assert root["n_subdirs"] == 2 and root["subdirs_descended"] == 1
    assert OU + "preview/" not in {d["path"] for d in one_child["dirs"]}


def test_list_s3_tree_on_a_missing_or_forbidden_bucket_records_and_stops():
    sess = FakeSession([("nasa-irsa-roman.s3", FakeResp(404, NO_BUCKET_XML))])
    tree = A.list_s3_tree("nasa-irsa-roman", "", sess, timeout_s=1.0)
    assert tree["exists"] is False and "NoSuchBucket" in tree["error"] and tree["http_status"] == 404
    assert tree["n_dirs_listed"] == 0 and tree["n_requests"] == 1 and len(sess.calls) == 1
    denied = '<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>'
    sess = FakeSession([("private.s3", FakeResp(403, denied))])
    tree = A.list_s3_tree("private", "x/", sess, timeout_s=1.0)
    assert tree["exists"] is None and "AccessDenied" in tree["error"] and tree["n_dirs_listed"] == 0
    dead = A.list_s3_tree("nowhere", "", FakeSession(), timeout_s=1.0)
    assert dead["error"] and dead["exists"] is None and dead["http_status"] is None
    json.dumps(tree)


def test_classification_of_openuniverse_patterns():
    b = "s3://nasa-irsa-simulations/"
    head = A.classify_product(b + SNANA_DIR + "ROMAN+LSST_NONIaMODEL0-0001_HEAD.FITS.gz", "irsa_s3", True)
    assert (head.level, head.kind, head.survey, head.simulated) == ("L4", "lightcurve", "HLTDS", True)
    assert head.meta["format"] == "snana_head"
    assert head.meta["phot_sibling"] == b + SNANA_DIR + "ROMAN+LSST_NONIaMODEL0-0001_PHOT.FITS.gz"
    phot = A.classify_product(head.meta["phot_sibling"], "irsa_s3", True)
    assert phot.kind == "lightcurve_phot" and phot.meta["format"] == "snana_phot"
    assert phot.meta["head_sibling"] == head.uri
    img = A.classify_product(b + TDS + "images/simple_model/F184/10307/Roman_TDS_simple_model_F184_10307_17.fits.gz",
                             "irsa_s3", True)
    assert (img.level, img.kind, img.survey, img.band) == ("L2", "cal", "HLTDS", "F184")
    assert img.meta["format"] == "openuniverse_image"
    assert img.meta["truth_index"] == b + TDS + "truth/F184/10307/Roman_TDS_index_F184_10307_17.txt"
    assert img.meta["truth_image"] == b + TDS + "images/truth/F184/10307/Roman_TDS_truth_F184_10307_17.fits.gz"
    assert (img.meta["pointing"], img.meta["sca"], img.meta["band_token"]) == (10307, 17, "F184")
    z = A.classify_product(b + TDS + "images/simple_model/Z087/1062/Roman_TDS_simple_model_Z087_1062_3.fits.gz",
                           "irsa_s3", True)
    assert z.band == "F087" and z.meta["band_token"] == "Z087"
    assert z.meta["truth_index"].endswith("truth/Z087/1062/Roman_TDS_index_Z087_1062_3.txt")
    was = A.classify_product(b + OU + "full/RomanWAS/images/simple_model/H158/5/Roman_WAS_simple_model_H158_5_2.fits.gz",
                             "irsa_s3", True)
    assert (was.survey, was.band, was.meta["openuniverse_survey"]) == ("HLWAS", "F158", "RomanWAS")
    assert was.meta["truth_index"] == b + OU + "full/RomanWAS/truth/H158/5/Roman_WAS_index_H158_5_2.txt"
    truth = A.classify_product(b + TDS + "images/truth/F184/10307/Roman_TDS_truth_F184_10307_17.fits.gz",
                               "irsa_s3", True)
    assert (truth.level, truth.kind, truth.meta["format"]) == ("L2", "truth_image", "openuniverse_truth_image")
    assert truth.meta["image"] == img.uri and truth.meta["truth_index"] == img.meta["truth_index"]
    index = A.classify_product(img.meta["truth_index"], "irsa_s3", True)
    assert (index.level, index.kind, index.meta["format"], index.band) == ("catalog", "catalog", "truth_index", "F184")
    assert index.meta["image"] == img.uri
    for name, fmt, variant in [("pointsource_10307.parquet", "pointsource", "main"),
                               ("pointsource_flux_10307.parquet", "pointsource", "flux"),
                               ("galaxy_10307.parquet", "galaxy", "main"),
                               ("galaxy_sed_10307.hdf5", "galaxy", "sed"),
                               ("snana_10307.parquet", "snana_truth", "main"),
                               ("snana_10307.hdf5", "snana_truth", "main")]:
        c = A.classify_product(b + CATS + name, "irsa_s3", True)
        assert (c.level, c.kind, c.meta["format"], c.meta["variant"], c.meta["healpix"]) == \
            ("catalog", "catalog", fmt, variant, 10307), name
    co = A.classify_product(b + OU + "full/RomanWAS/images/coadd/H158/prod_F_00_03_map.fits.gz", "irsa_s3", True)
    assert (co.level, co.kind, co.band, co.survey) == ("L3", "coadd", "F158", "HLWAS")
    assert co.meta == {"format": "openuniverse_coadd", "tile_row": 0, "tile_col": 3,
                       "openuniverse_survey": "RomanWAS"}
    co2 = A.classify_product(b + OU + "preview/RomanWAS/images/coadds/F184/Row12/prod_F_12_07_map.fits.gz")
    assert (co2.kind, co2.band, co2.meta["tile_row"], co2.meta["tile_col"], co2.simulated) == \
        ("coadd", "F184", 12, 7, True)
    obseq = A.classify_product(b + TDS + "Roman_TDS_obseq_11_6_23_radec.fits", "irsa_s3", True)
    assert (obseq.kind, obseq.level, obseq.meta["variant"], obseq.survey) == ("obseq", "sim", "radec", "HLTDS")
    assert A.classify_product(b + TDS + "Roman_TDS_obseq_11_6_23.fits", "irsa_s3", True).meta["variant"] == "pointings"
    for name in ("ROMAN+LSST_LARGE_SNIa-normal.DUMP", "ROMAN+LSST_LARGE_SNIa-normal.LIST",
                 "ROMAN+LSST_LARGE_SNIa-normal.README"):
        d = A.classify_product(b + SNANA_DIR + name, "irsa_s3", True)
        assert (d.kind, d.level, d.survey) == ("doc", "sim", "HLTDS"), name
    for name in ("tds.yaml", "TDS.pdf", "fig.png"):
        assert A.classify_product(b + TDS + name, "irsa_s3", True).kind == "doc", name
    assert A.classify_product("s3://x/notes.pdf").level == "unknown"          # a doc nobody called simulated
    # the band map is a config key: a custom map wins, and an unmapped token stays None
    assert A.classify_product(b + TDS + "images/simple_model/Q999/1/Roman_TDS_simple_model_Q999_1_1.fits.gz",
                              band_map={"Q999": "F213"}).band == "F213"
    assert A.band_from_path("a/H158/b", band_map={}) is None and A.band_from_path("a/H158/b") == "F158"
    assert A.band_from_path("x/F146/y", band_map={}) == "F146"                # WFI names need no map
    # a mission-style key inside the same bucket keeps the documented classification
    u = A.classify_product(b + TDS + "images/simple_model/F184/r0000101001001001001_0001_wfi01_uncal.asdf", "irsa_s3", True)
    assert (u.level, u.kind, u.band) == ("L1", "uncal", "F184")


def test_inventory_from_the_tree_counts_formats_and_records_the_missing_mission_bucket():
    sess = FakeSession([("nasa-irsa-simulations.s3.amazonaws.com", fake_s3_tree),
                        ("nasa-irsa-roman.s3", FakeResp(404, NO_BUCKET_XML))])
    conf = conf_with(s3_tree={"max_depth": 6, "max_keys_per_dir": 7, "max_dirs": 400,
                              "max_subdirs_per_dir": 12, "max_numeric_subdirs_per_dir": 1,
                              "max_pages_per_dir": 5})
    probe = A.probe(conf, session=sess, timeout_s=1.0)
    assert probe["data_state"] == "SIMULATIONS_ONLY"
    inv = A.inventory(conf, probe, session=sess, n_shards=4)
    assert inv["verdict"] == "INVENTORIED" and inv["data_state"] == "SIMULATIONS_ONLY"
    assert inv["n_simulated"] == inv["n_products"] and inv["n_mission"] == 0
    assert all(p["origin"] == "irsa_s3" and p["simulated"] for p in inv["products"])
    assert inv["counts_by_format"] == {
        "doc": 5, "snana_head": 2, "snana_phot": 2, "obseq": 3, "openuniverse_image": 2,
        "openuniverse_truth_image": 1, "truth_index": 2, "galaxy": 3, "pointsource": 2,
        "snana_truth": 2, "openuniverse_coadd": 1, "none": 1}
    assert inv["counts_by_kind"]["lightcurve"] == 2 and inv["counts_by_kind"]["cal"] == 2
    assert inv["counts_by_kind"]["lightcurve_phot"] == 2 and inv["counts_by_kind"]["doc"] == 5
    assert inv["counts_by_kind"]["unknown"] == 1                 # full/RomanWAS/truth/tmp.txt, honestly
    assert inv["counts_by_level"] == {"L4": 4, "L2": 3, "L3": 1, "catalog": 9, "sim": 8, "unknown": 1}
    by_uri = {p["uri"]: p for p in inv["products"]}
    head = by_uri["s3://nasa-irsa-simulations/" + SNANA_DIR + "ROMAN+LSST_NONIaMODEL0-0001_HEAD.FITS.gz"]
    assert head.get("meta", {})["phot_sibling"] in by_uri
    assert head["meta"]["bucket"] == "nasa-irsa-simulations" and head["meta"]["endpoint"] == "s3:openuniverse2024"
    img = next(p for p in inv["products"] if p["meta"].get("format") == "openuniverse_image" and p["band"] == "F184")
    assert img["meta"]["truth_index"] in by_uri and by_uri[img["meta"]["truth_index"]]["meta"]["format"] == "truth_index"
    assert {p["band"] for p in inv["products"] if p["meta"].get("format") == "openuniverse_image"} == {"F184", "F087"}
    # the tree summary, and the truncated SNANA directory in it
    rows = {r["path"]: r for r in inv["tree"]}
    assert set(rows) == LISTED and all(r["endpoint"] == "s3:openuniverse2024" for r in rows.values())
    assert rows[BIG]["subdirs_truncated"] is True and rows[BIG]["n_subdirs_numeric"] == 1000
    assert rows[SNANA_DIR]["truncated"] is True and rows[SNANA_DIR]["n_keys_listed"] == 7
    assert rows[CATS]["truncated"] is False and rows[CATS]["n_keys_listed"] == 7
    assert rows[OU]["n_subdirs"] == 2 and rows[TDS]["n_subdirs"] == 2 and "keys" not in rows[TDS]
    sim = inv["listings"]["s3:openuniverse2024"]
    assert sim["tree"]["n_dirs"] == len(LISTED) and sim["tree"]["n_dirs_truncated"] == 2
    assert sim["tree"]["limits"]["max_numeric_subdirs_per_dir"] == 1
    assert sim["n_keys"] == inv["n_products"] and "dirs" not in sim["tree"]
    # the mission bucket does not exist: a recorded status, no products, no walk
    mission = inv["listings"]["s3:roman_mission"]
    assert mission["status"] == "http_error" and mission["exists"] is False and mission["tree"] is None
    assert mission["n_keys"] == 0 and "NoSuchBucket" in mission["error"]
    assert not any(p["meta"].get("endpoint") == "s3:roman_mission" for p in inv["products"])
    assert sum(len(s) for s in inv["shards"]) == inv["n_products"] and len(inv["shards"]) == 4
    assert inv["total_bytes"] == sum(p["size_bytes"] for p in inv["products"])
    json.dumps(inv)
    # max_listing caps the products, and says so
    small = A.inventory(conf, probe, session=sess, max_listing=5)
    assert small["n_products"] == 5 and small["listings"]["s3:openuniverse2024"]["n_keys_capped"] is True


# --------------------------------------------------------------------------------------
# MAST CAOM: TOP 1 first, COUNT second, the whole error kept
# --------------------------------------------------------------------------------------

MAST_400 = ('<?xml version="1.0" encoding="UTF-8"?>\n<VOTABLE version="1.4" '
            'xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n<RESOURCE type="results">\n'
            '<INFO name="QUERY_STATUS" value="ERROR">Column obs_collection is not valid for '
            'table dbo.CaomObservation</INFO>\n<INFO name="ERROR_DETAIL" value="">line 1</INFO>\n'
            '</RESOURCE>\n</VOTABLE>\n')


def _mast_router(top1=None, count=None):
    def route(url):
        if "SELECT+TOP+1" in url:
            return top1(url) if callable(top1) else top1
        assert "COUNT" in url, url
        return count(url) if callable(count) else count
    return route


def test_mast_400_records_the_votable_info_text_and_the_error_body():
    sess = FakeSession([("mast.stsci.edu", _mast_router(top1=FakeResp(400, MAST_400),
                                                        count=FakeResp(400, MAST_400)))])
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    ep = rec["endpoints"]["mast:Roman"]
    assert ep["status"] == "http_error" and ep["http_status"] == 400 and ep["count"] is None
    assert ep["present"] is None and ep["query_form"] is None
    assert [a["form"] for a in ep["attempts"]] == ["top1", "count"]
    assert ep["attempts"][0]["url"].startswith(CONF["archive"]["mast"]["tap"].rstrip("/") + "/sync?")
    assert "SELECT+TOP+1+obs_collection" in ep["attempts"][0]["url"]
    assert "COUNT" in ep["attempts"][1]["url"]
    info = ep["attempts"][0]["votable_info"]
    assert info[0] == ("QUERY_STATUS=ERROR: Column obs_collection is not valid for table "
                       "dbo.CaomObservation")
    assert info[1] == "ERROR_DETAIL=: line 1"
    assert ep["votable_info"] == info + info and ep["error"] == info[0]
    assert ep["error_body"].startswith("<?xml") and len(ep["error_body"]) <= A.MAX_ERROR_BODY
    assert ep["attempts"][0]["error_body"] == MAST_400
    assert rec["data_state"] == "NOT_YET_PUBLIC" and rec["n_endpoints_reached"] == 3
    # a body longer than the cap is cut at the cap, its INFO still extracted
    long_body = MAST_400.replace("line 1", "x" * 5000)
    sess = FakeSession([("mast.stsci.edu", FakeResp(400, long_body))])
    ep = A.probe(CONF, session=sess, timeout_s=1.0)["endpoints"]["mast:Roman"]
    assert len(ep["error_body"]) == A.MAX_ERROR_BODY and ep["votable_info"][0].startswith("QUERY_STATUS")
    # a truncated / unparseable document still yields the INFO text by regex
    assert A.votable_info(MAST_400[:-30])[0].startswith("QUERY_STATUS=ERROR: Column obs_collection")
    assert A.votable_info("<html>nope</html>") == []


def test_mast_top1_row_then_count_and_the_count_fallback():
    top1 = FakeResp(200, tap_json([["Roman"]], ["obs_collection"]))
    count = FakeResp(200, tap_json([[42]], ["n"]))
    sess = FakeSession([("mast.stsci.edu", _mast_router(top1=top1, count=count))])
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    ep = rec["endpoints"]["mast:Roman"]
    assert ep["status"] == "ok" and ep["present"] is True and ep["count"] == 42
    assert ep["query_form"] == "top1" and [a["form"] for a in ep["attempts"]] == ["top1", "count"]
    assert rec["data_state"] == "MISSION_DATA_PRESENT"
    assert any("MAST collection Roman: 42 observations" in e for e in rec["data_state_evidence"])
    # TOP 1 present but the count request fails: present stands, count unknown, evidence says so
    sess = FakeSession([("mast.stsci.edu", _mast_router(top1=top1, count=FakeResp(500, "boom")))])
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    ep = rec["endpoints"]["mast:Roman"]
    assert ep["status"] == "ok" and ep["present"] is True and ep["count"] is None
    assert rec["data_state"] == "MISSION_DATA_PRESENT"
    assert any("count unknown" in e for e in rec["data_state_evidence"])
    # TOP 1 empty: absent, count 0, no second request per collection
    sess = FakeSession([("mast.stsci.edu", _mast_router(top1=FakeResp(200, tap_json([], ["obs_collection"]))))])
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    ep = rec["endpoints"]["mast:Roman"]
    assert (ep["status"], ep["present"], ep["count"], len(ep["attempts"])) == ("ok", False, 0, 1)
    assert rec["data_state"] == "NOT_YET_PUBLIC"
    # TOP 1 refused, COUNT answers: the fallback form carries the record
    sess = FakeSession([("mast.stsci.edu", _mast_router(top1=FakeResp(400, MAST_400),
                                                        count=FakeResp(200, tap_json([[7]], ["n"]))))])
    ep = A.probe(CONF, session=sess, timeout_s=1.0)["endpoints"]["mast:RomanSim"]
    assert (ep["status"], ep["query_form"], ep["count"], ep["present"]) == ("ok", "count", 7, True)
    assert ep["attempts"][0]["votable_info"][0].startswith("QUERY_STATUS=ERROR")
    assert ep["error"] is None and ep["simulated"] is True     # a collection named RomanSim is a sim
    assert not A._is_simulated_name("hltds/prism/x_cal.asdf") and A._is_simulated_name("wfisim/x.fits")


# --------------------------------------------------------------------------------------
# index-page crawl
# --------------------------------------------------------------------------------------

INDEX_HTML = """<html><head><title>Roman at Example</title></head><body>
<a href="sims/">Simulations</a>
<a href="/data/challenge.html">Microlensing <b>data</b> challenge</a>
<a href="/contact.html">Contact us</a>
<a href="https://example.org/roman/">home</a>
<a href="#top">top</a><a href="mailto:x@example.org">mail</a>
<a href='sims/'>Simulations (again)</a>
</body></html>"""
SIMS_HTML = "<html><head><title>Simulation index</title></head><a href='lc_0001.csv'>lc</a><a href='about.html'>a</a></html>"
CHALLENGE_HTML = "<title>Data Challenge</title><a href='truth.parquet'>t</a><a href='lc_0001.csv'>dup</a>"


def _crawl_conf(**kw):
    return conf_with(simulations=[{"name": "idx", "url": "https://example.org/roman/", "kind": "index"}],
                     **kw)


def test_index_page_crawl_follows_matching_links_one_level_and_ignores_the_rest():
    sess = FakeSession([("example.org/roman/sims/", FakeResp(200, SIMS_HTML)),
                        ("example.org/data/challenge.html", FakeResp(200, CHALLENGE_HTML)),
                        ("example.org/contact.html", FakeResp(200, "<title>Contact</title>")),
                        ("example.org/roman/", FakeResp(200, INDEX_HTML))])
    rec = A.probe(_crawl_conf(), session=sess, timeout_s=1.0)
    ep = rec["endpoints"]["sim:idx"]
    assert ep["status"] == "ok" and ep["title"] == "Roman at Example" and ep["data_links"] == []
    crawl = ep["crawl"]
    assert crawl["n_links_total"] == 4 and crawl["n_links_matched"] == 2 and crawl["n_pages_fetched"] == 2
    assert crawl["capped"] is False
    # microlens/challenge outrank a bare "sim" match, so the challenge page is fetched first
    assert crawl["links_matched"] == ["https://example.org/data/challenge.html", "https://example.org/roman/sims/"]
    pages = {p["url"]: p for p in crawl["pages"]}
    assert pages["https://example.org/roman/sims/"]["title"] == "Simulation index"
    assert pages["https://example.org/roman/sims/"]["anchor_text"] == "Simulations"
    assert pages["https://example.org/roman/sims/"]["data_links"] == ["https://example.org/roman/sims/lc_0001.csv"]
    assert pages["https://example.org/roman/sims/"]["http_status"] == 200
    assert pages["https://example.org/roman/sims/"]["head_status"] == 200
    ch = pages["https://example.org/data/challenge.html"]
    assert ch["title"] == "Data Challenge" and ch["anchor_text"] == "Microlensing data challenge"
    assert ch["data_links"] == ["https://example.org/data/truth.parquet", "https://example.org/data/lc_0001.csv"]
    assert set(crawl["data_links"]) == {"https://example.org/roman/sims/lc_0001.csv",
                                        "https://example.org/data/truth.parquet",
                                        "https://example.org/data/lc_0001.csv"}
    assert len(crawl["data_links"]) == 3                            # de-duplicated across pages
    assert not any("contact" in u for u in sess.calls)          # the unrelated link was never fetched
    assert sum(1 for u in sess.calls if u == "https://example.org/roman/") == 2   # HEAD + GET, no self-crawl
    assert rec["data_state"] == "SIMULATIONS_ONLY"
    assert any("sim:idx: index page and its crawl expose 3 data link(s)" in e
               for e in rec["data_state_evidence"])
    inv = A.inventory(_crawl_conf(), rec, session=sess)
    crawled = [p for p in inv["products"] if p["meta"].get("crawled_from")]
    assert len(crawled) == 3 and all(p["origin"] == "ipac_sim_page" and p["simulated"] for p in crawled)
    assert {p["kind"] for p in crawled} == {"lightcurve", "unknown"}
    json.dumps(rec)


def test_index_page_crawl_budget_priority_and_failures():
    # priority: a link naming a simulation is fetched before one matching only ``roman``
    html = ('<title>t</title><a href="/roman/news.html">Roman news</a>'
            '<a href="/other/sim_data.html">simulated data</a><a href="/roman/x.fits">x</a>')
    def dead(url):
        raise ConnectionError(f"no route to {url}")
    sess = FakeSession([("news.html", dead),
                        ("example.org/roman/", FakeResp(200, html)),
                        ("sim_data.html", FakeResp(200, "<title>Sim</title>")),
                        ("x.fits", FakeResp(200, "", {"content-type": "application/fits"}))])
    rec = A.probe(_crawl_conf(crawl_max_pages=1), session=sess, timeout_s=1.0)
    crawl = rec["endpoints"]["sim:idx"]["crawl"]
    assert crawl["n_links_matched"] == 3 and crawl["capped"] is True and crawl["n_pages_fetched"] == 1
    assert crawl["pages"][0]["url"] == "https://example.org/other/sim_data.html"
    assert not any("news" in u for u in sess.calls)
    # a data-file link is HEADed and recorded, never downloaded; a dead link is recorded with its error
    rec = A.probe(_crawl_conf(crawl_max_pages=3), session=sess, timeout_s=1.0)
    pages = {p["url"]: p for p in rec["endpoints"]["sim:idx"]["crawl"]["pages"]}
    fits = pages["https://example.org/roman/x.fits"]
    assert fits["is_data_file"] and fits["head_status"] == 200 and fits["data_links"] == [fits["url"]]
    assert sum(1 for u in sess.calls if u.endswith("x.fits")) == 1
    news = pages["https://example.org/roman/news.html"]
    assert news["http_status"] is None and "no route" in news["error"] and news["title"] is None
    assert rec["endpoints"]["sim:idx"]["crawl"]["data_links"] == ["https://example.org/roman/x.fits"]
    # a page that is not html-ish: the crawl is only for ``kind: index``
    plain = conf_with(simulations=[{"name": "p", "url": "https://example.org/roman/", "kind": "page"}])
    assert A.probe(plain, session=sess, timeout_s=1.0)["endpoints"]["sim:p"]["crawl"] is None
    assert A.extract_anchors("<a href='a.html'>A</a><a href='javascript:void(0)'>j</a>", "https://h/x/") == \
        [("https://h/x/a.html", "A")]
    assert A.page_title("<TITLE> Two\n words </TITLE>") == "Two words" and A.page_title("x") is None


def test_fetch_to_cache_encodes_plus_and_mirrors_the_decoded_key(tmp_path):
    key = "openuniverse2024/roman/full/ROMAN+LSST_LARGE_SNIa-normal/ROMAN+LSST_NONIaMODEL0-0001_HEAD.FITS.gz"
    payload = b"\x1f\x8b" + b"z" * 100
    sess = FakeSession([("ROMAN%2BLSST_LARGE_SNIa-normal/ROMAN%2BLSST_NONIaMODEL0-0001_HEAD.FITS.gz",
                         FakeResp(200, "", {"content-type": "application/octet-stream"}, payload))])
    url = A.to_https("s3://nasa-irsa-simulations/" + key)
    assert url == ("https://nasa-irsa-simulations.s3.amazonaws.com/openuniverse2024/roman/full/"
                   "ROMAN%2BLSST_LARGE_SNIa-normal/ROMAN%2BLSST_NONIaMODEL0-0001_HEAD.FITS.gz")
    out = A.fetch_to_cache("s3://nasa-irsa-simulations/" + key, tmp_path, session=sess, pause=0.0)
    assert out is not None and out.read_bytes() == payload
    assert sess.calls == [url] and "+" not in url
    assert out == tmp_path / "nasa-irsa-simulations.s3.amazonaws.com" / key       # decoded on disk
    prov = json.loads((out.parent / (out.name + ".provenance.json")).read_text())
    assert prov["url"] == url and prov["uri"] == "s3://nasa-irsa-simulations/" + key
    # the listing URL builder makes the same promise
    lst = A.s3_listing_url("nasa-irsa-simulations", "openuniverse2024/roman/full/ROMAN+LSST_LARGE_SNIa-normal/")
    assert "prefix=openuniverse2024/roman/full/ROMAN%2BLSST_LARGE_SNIa-normal/" in lst and "+" not in lst
    assert "delimiter=%2F" in lst and lst.startswith("https://nasa-irsa-simulations.s3.amazonaws.com/?list-type=2&")


# --------------------------------------------------------------------------------------
# dq flags
# --------------------------------------------------------------------------------------

def test_dq_flag_table_and_reject_mask():
    table, source = P.dq_flags(CONF)
    assert source in ("roman_datamodels.dqflags", "config/roman.yaml")
    assert table["JUMP_DET"] == 4 and table["SATURATED"] == 2
    mask = P.lightcurve_dq_mask(CONF, table)
    assert mask == (1 | 2 | 1024 | 2048)
    assert mask & table["JUMP_DET"] == 0        # a jump is a signal, never a rejection
    assert P.lightcurve_dq_mask({"lightcurve_dq_reject": ["NOT_A_FLAG", "HOT"]}, table) == 2048


def test_dq_flags_falls_back_to_config_when_roman_datamodels_is_absent(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("roman_datamodels"):
            raise ImportError("no roman_datamodels here")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    table, source = P.dq_flags({"dq_flags": {"JUMP_DET": 4, "CUSTOM": 1 << 20}})
    assert source == "config/roman.yaml" and table["CUSTOM"] == 1 << 20 and table["DEAD"] == 1024


# --------------------------------------------------------------------------------------
# light-curve reader
# --------------------------------------------------------------------------------------

def test_lightcurve_reader_mag_only_table_converts_to_flux():
    df = pd.DataFrame({"MJD": [60001.0, 60000.0, 60002.0], "mag": [20.0, 20.0, 21.0],
                       "magerr": [0.01, 0.01, 0.02]})
    lcs = P.read_lightcurve_table(df, conf=CONF, band="F146", survey="GBTDS", star_id="s1")
    assert isinstance(lcs, list) and len(lcs) == 1
    lc = lcs[0]
    assert isinstance(lc, LightCurve)
    assert lc.flux_unit == "mag_derived" and lc.flux_zp_ab == 25.0
    assert lc.meta["zp_source"] == "mag_conversion_zp25"
    assert np.allclose(lc.mjd, [60000.0, 60001.0, 60002.0])     # sorted by time
    assert np.allclose(lc.flux[:2], 10 ** (-0.4 * (20.0 - 25.0)))
    assert np.isclose(lc.flux[2] / lc.flux[0], 10 ** -0.4)
    assert np.allclose(lc.flux_err / lc.flux, np.array([0.01, 0.01, 0.02]) * np.log(10) / 2.5)
    assert lc.band == "F146" and lc.survey == "GBTDS" and lc.star_id == "s1" and lc.dq is None
    t = lc.to_mag()
    assert t is not None and np.allclose(t[1], [20.0, 20.0, 21.0])


def test_lightcurve_reader_flux_table_takes_zero_point_from_meta_then_config():
    df = pd.DataFrame({"star_id": ["a", "a", "b", "b"], "band": ["F146", "F146", "F146", "F087"],
                       "mjd": [1.0, 2.0, 1.0, 1.5], "flux": [100.0, 101.0, 50.0, 60.0],
                       "flux_err": [1.0, 1.0, 1.0, 1.0], "dq": [0, 2, 0, np.nan],
                       "ra": [10.0, 10.0, 11.0, 11.0], "dec": [-1.0, -1.0, -2.0, -2.0]})
    df.attrs = {"zp": 27.7}
    lcs = P.read_lightcurve_table(df, conf=CONF, survey="GBTDS")
    assert {(lc.star_id, lc.band) for lc in lcs} == {("a", "F146"), ("b", "F146"), ("b", "F087")}
    a = next(lc for lc in lcs if lc.star_id == "a")
    assert a.flux_zp_ab == 27.7 and a.meta["zp_source"] == "table_meta" and a.flux_unit == "e-/s"
    assert a.ra == 10.0 and a.dec == -1.0 and a.dq.tolist() == [0, 2]
    mask = P.lightcurve_dq_mask(CONF, P.dq_flags(CONF)[0])
    assert a.good(mask).tolist() == [True, False]          # SATURATED epoch rejected
    b087 = next(lc for lc in lcs if lc.band == "F087")
    assert b087.dq.tolist() == [0]                           # NaN dq -> clean, not rejected
    # no zp in the metadata -> the config filter table, per band
    df2 = df.copy()
    df2.attrs = {}
    lcs2 = P.read_lightcurve_table(df2, conf=CONF)
    zps = {lc.band: (lc.flux_zp_ab, lc.meta["zp_source"]) for lc in lcs2}
    assert zps["F146"] == (27.7, "config/roman.yaml") and zps["F087"] == (26.5, "config/roman.yaml")
    # unknown unit and no meta -> relative flux, honestly marked
    df3 = df.copy()
    df3.attrs = {"flux_unit": "normalised"}
    lcs3 = P.read_lightcurve_table(df3, conf=CONF)
    assert all(lc.flux_zp_ab is None and lc.meta["zp_source"] == "none" for lc in lcs3)


def test_lightcurve_reader_time_columns_and_colmap():
    df = pd.DataFrame({"BJD_TDB": [2460000.5, 2460001.5], "f": [1.0, 2.0], "ferr": [0.1, 0.1]})
    lc = P.read_lightcurve_table(df, conf=CONF, band="F146")[0]
    assert lc.time_system == "BJD_TDB" and np.allclose(lc.mjd, [60000.0, 60001.0])
    df2 = pd.DataFrame({"epoch_days": [1.0, 2.0], "counts_per_s": [1.0, 2.0], "sig": [0.1, 0.1]})
    lc2 = P.read_lightcurve_table(df2, colmap={"time": "epoch_days", "flux": "counts_per_s",
                                               "flux_err": "sig"}, band="F087")[0]
    assert lc2.n == 2 and lc2.time_system == "unknown"


def test_lightcurve_reader_degrades_on_missing_roles():
    ru = P.read_lightcurve_table(pd.DataFrame({"mjd": [1.0], "flux": [1.0]}), conf=CONF)
    assert isinstance(ru, P.ReaderUnavailable) and "flux_err" in ru.needs
    ru = P.read_lightcurve_table(pd.DataFrame({"flux": [1.0], "flux_err": [1.0]}), conf=CONF)
    assert isinstance(ru, P.ReaderUnavailable) and "time" in ru.needs
    ru = P.read_lightcurve_table(pd.DataFrame({"mjd": [1.0], "mag": [1.0]}), conf=CONF)
    assert isinstance(ru, P.ReaderUnavailable) and "mag_err" in ru.needs
    ru = P.read_lightcurve_table("/nonexistent/file.xyz", conf=CONF)
    assert isinstance(ru, P.ReaderUnavailable)
    assert "reader_unavailable" in ru.as_dict()


def test_lightcurve_reader_from_parquet_and_csv_files(tmp_path):
    df = pd.DataFrame({"mjd": [1.0, 2.0, 3.0], "flux": [1.0, 1.1, 0.9], "flux_err": [0.1] * 3})
    pq = tmp_path / "lc_1.parquet"
    df.to_parquet(pq)
    lcs = P.read_lightcurve_table(pq, conf=CONF, band="F146")
    assert isinstance(lcs, list) and lcs[0].n == 3 and lcs[0].flux_zp_ab == 27.7
    csv = tmp_path / "lc_2.csv"
    csv.write_text("# zp = 26.0\n# band = F087\nmjd,flux,flux_err\n1,1.0,0.1\n2,1.2,0.1\n")
    lcs = P.read_lightcurve_table(csv, conf=CONF)
    assert lcs[0].band == "F087" and lcs[0].flux_zp_ab == 26.0 and lcs[0].meta["zp_source"] == "table_meta"


# --------------------------------------------------------------------------------------
# spectrum reader
# --------------------------------------------------------------------------------------

def test_spectrum_reader_converts_angstrom_and_keeps_micron():
    wl_a = np.linspace(10000.0, 19300.0, 50)
    df = pd.DataFrame({"wave": wl_a, "flux": np.ones(50), "flux_err": np.full(50, 0.1),
                       "contam": np.zeros(50)})
    df.attrs = {"class_star": 0.98}
    sp = P.read_spectrum_table(df, conf=CONF, mode="G150", source_id="src1")
    assert isinstance(sp, list) and len(sp) == 1 and isinstance(sp[0], Spectrum)
    s = sp[0]
    assert np.allclose(s.wavelength_um, wl_a / 1e4) and s.meta["wavelength_unit_in"] == "angstrom"
    assert isinstance(s.resolving_power, np.ndarray)
    assert np.allclose(s.resolving_power, 461.0 * s.wavelength_um)
    assert s.meta["r_source"] == "config/roman.yaml" and s.is_point_source is True
    assert s.contam is not None and s.dq is None and s.mode == "G150"
    wl_um = np.linspace(1.0, 1.9, 40)
    df2 = pd.DataFrame({"wavelength": wl_um, "flux": np.ones(40), "flux_err": np.full(40, 0.1)})
    s2 = P.read_spectrum_table(df2, conf=CONF, mode="G150")[0]
    assert np.allclose(s2.wavelength_um, wl_um) and s2.meta["wavelength_unit_in"] == "um"
    assert s2.is_point_source is None and s2.contam is None
    df3 = pd.DataFrame({"lambda": np.linspace(800.0, 1800.0, 10), "flux": np.ones(10),
                        "err": np.ones(10)})
    s3 = P.read_spectrum_table(df3, conf=CONF)[0]
    assert s3.meta["wavelength_unit_in"] == "nm" and s3.mode == "unknown" and s3.resolving_power is None


def test_spectrum_reader_prism_resolving_power_is_interpolated():
    wl = np.array([0.75, 1.275, 1.80])
    df = pd.DataFrame({"wl": wl, "flux": [1, 1, 1], "fluxerr": [0.1, 0.1, 0.1], "dq": [0, 4, 0]})
    df.attrs = {"morphology": "extended", "R": None}
    s = P.read_spectrum_table(df, conf=CONF, mode="P127")[0]
    assert np.allclose(s.resolving_power, [80.0, 130.0, 180.0])
    assert s.is_point_source is False and s.dq.tolist() == [0, 4, 0]
    assert np.isclose(s.r_at(1.0125), 105.0)
    df.attrs = {"R": 100.0}
    s = P.read_spectrum_table(df, conf=CONF, mode="P127")[0]
    assert s.resolving_power == 100.0 and s.meta["r_source"] == "table_meta"


def test_spectrum_reader_groups_by_source_and_degrades():
    df = pd.DataFrame({"source_id": ["x", "x", "y"], "wave_um": [1.0, 1.5, 1.2],
                       "flux": [1, 2, 3], "flux_err": [0.1, 0.1, 0.1]})
    sp = P.read_spectrum_table(df, conf=CONF, mode="G150")
    assert [s.source_id for s in sp] == ["x", "y"] and sp[0].n == 2
    ru = P.read_spectrum_table(pd.DataFrame({"flux": [1.0], "flux_err": [0.1]}), conf=CONF)
    assert isinstance(ru, P.ReaderUnavailable) and "wavelength" in ru.needs


def test_wavelength_unit_inference():
    assert P.wavelength_to_um(np.array([12000.0]))[1] == "angstrom"
    assert P.wavelength_to_um(np.array([1200.0]))[1] == "nm"
    assert P.wavelength_to_um(np.array([1.2]))[1] == "um"
    assert P.wavelength_to_um(np.array([1.2]), "nm")[1] == "nm"
    assert P.wavelength_to_um(np.array([50.0]))[1] == "unknown"


# --------------------------------------------------------------------------------------
# ASDF readers
# --------------------------------------------------------------------------------------

def test_read_asdf_arrays_reports_missing_asdf(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "asdf" or name.startswith("asdf."):
            raise ImportError("no asdf in this sandbox")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "asdf", raising=False)
    ru = P.read_asdf_arrays("/tmp/does_not_matter_cal.asdf", keys=("dq",))
    assert isinstance(ru, P.ReaderUnavailable) and ru.needs == ["asdf"]
    ru2 = P.read_ramp("/tmp/x_uncal.asdf", box=(0, 4, 0, 4), conf=CONF)
    assert isinstance(ru2, P.ReaderUnavailable) and "asdf" in ru2.needs


def test_read_asdf_arrays_on_a_real_file(tmp_path):
    asdf = pytest.importorskip("asdf")
    dq = (np.arange(16, dtype=np.uint32).reshape(4, 4) * 4)
    tree = {"roman": {"meta": {"exposure": {"read_pattern": [[1], [2, 3]], "frame_time": 3.04},
                               "instrument": {"optical_element": "F146", "detector": "WFI01"}},
                      "dq": dq, "data": np.ones((2, 4, 4), dtype=np.float32)}}
    path = tmp_path / "r0001_wfi01_cal.asdf"
    asdf.AsdfFile(tree).write_to(str(path))
    out = P.read_asdf_arrays(path, keys=("dq", "nope"))
    assert isinstance(out, dict) and out["missing"] == ["nope"]
    assert np.array_equal(out["dq"], dq)
    assert out["meta"]["instrument.optical_element"] == "F146"
    assert out["meta"]["exposure.frame_time"] == 3.04
    ramp = P.read_ramp(path, box=(1, 3, 0, 2), conf=CONF)
    assert isinstance(ramp, Ramp) and ramp.resultants.shape == (2, 2, 2)
    assert np.allclose(ramp.times_s, [3.04, 2.5 * 3.04]) and ramp.x0 == 1 and ramp.y0 == 0


def test_read_ramp_from_memory_dict_mid_times():
    cube = np.arange(6 * 3 * 3, dtype=float).reshape(6, 3, 3)
    ru = P.read_ramp({"data": cube, "read_pattern": [[1], [2, 3], [4, 5, 6]], "frame_time": 3.04})
    assert isinstance(ru, P.ReaderUnavailable)         # 3 groups vs 6 resultants: reported
    cube = np.arange(3 * 3 * 3, dtype=float).reshape(3, 3, 3)
    ramp = P.read_ramp({"data": cube, "read_pattern": [[1], [2, 3], [4, 5, 6]], "frame_time": 3.04},
                       box=(0, 2, 1, 3), conf=CONF)
    assert isinstance(ramp, Ramp)
    assert np.allclose(ramp.times_s, [1.0 * 3.04, 2.5 * 3.04, 5.0 * 3.04])
    assert ramp.resultants.shape == (3, 2, 2) and ramp.x0 == 0 and ramp.y0 == 1
    assert np.allclose(ramp.resultants[0], cube[0, 1:3, 0:2])
    assert ramp.gain_e_per_dn == CONF["instruments"]["WFI"]["gain_e_per_dn"]
    assert ramp.meta["frame_time_source"] == "file"
    # frame time from the config when the dict has none; none anywhere -> reported
    r2 = P.read_ramp({"data": cube, "read_pattern": [[1], [2], [3]]}, conf=CONF)
    assert isinstance(r2, Ramp) and np.allclose(r2.times_s, np.array([1, 2, 3]) * 3.04)
    assert r2.meta["frame_time_source"] == "config/roman.yaml"
    assert isinstance(P.read_ramp({"data": cube, "read_pattern": [[1], [2], [3]]}), P.ReaderUnavailable)
    assert isinstance(P.read_ramp({"data": cube}, conf=CONF), P.ReaderUnavailable)


# --------------------------------------------------------------------------------------
# cutouts and pixel coordinates
# --------------------------------------------------------------------------------------

def test_dq_cutouts_geometry_and_grouping():
    dq = np.zeros((200, 300), dtype=np.uint32)
    dq[100, 150] = 4
    dq[3, 297] = 2
    stars = [{"star_id": "centre", "x": 150.0, "y": 100.0},
             {"star_id": "near", "x": 158.0, "y": 104.0},        # inside the first box
             {"star_id": "corner", "x": 297.0, "y": 3.0},         # clipped to the detector edge
             {"star_id": "edge_of_box", "x": 165.0, "y": 100.0},  # within box but inside margin
             {"star_id": "off", "x": 400.0, "y": 10.0}]
    cuts = P.dq_cutouts_from_image(dq, stars, "img1", box=32, margin=4, band="F146", mjd=60000.0)
    assert all(isinstance(c, DQCutout) for c in cuts)
    assert [c.meta["n_stars"] for c in cuts] == [2, 1, 1]
    c0 = cuts[0]
    assert (c0.x0, c0.y0) == (150 - 16, 100 - 16) and c0.shape == (32, 32)
    assert c0.dq[16, 16] == 4 and c0.stars[0]["x"] == 16.0 and c0.stars[0]["y"] == 16.0
    assert c0.stars[1]["star_id"] == "near" and (c0.stars[1]["x"], c0.stars[1]["y"]) == (24.0, 20.0)
    assert c0.stars[0]["x_det"] == 150.0 and c0.band == "F146" and c0.mjd == 60000.0
    corner = cuts[1]
    assert (corner.x0, corner.y0) == (300 - 32, 0) and corner.dq[3, 297 - 268] == 2
    assert corner.stars[0]["x"] == 297.0 - 268 and corner.stars[0]["y"] == 3.0
    assert cuts[2].stars[0]["star_id"] == "edge_of_box" and cuts[2].meta["n_off_image"] == 1
    # no grouping -> one cutout per on-detector star
    assert len(P.dq_cutouts_from_image(dq, stars, "img1", box=32, group=False)) == 4
    # a detector smaller than the box yields the whole image
    small = P.dq_cutouts_from_image(np.zeros((10, 10)), [{"star_id": "s", "x": 5, "y": 5}], "i", box=32)
    assert small[0].shape == (10, 10) and (small[0].x0, small[0].y0) == (0, 0)


def test_stars_to_pixels_accepts_callable_and_wcs_like():
    def linear(ra, dec):
        return (np.asarray(ra) - 10.0) * 100.0, (np.asarray(dec) + 5.0) * 100.0
    x, y = P.stars_to_pixels(linear, [10.5, 11.0], [-4.0, -3.0])
    assert np.allclose(x, [50.0, 100.0]) and np.allclose(y, [100.0, 200.0])

    class W:
        def world_to_pixel_values(self, ra, dec):
            return linear(ra, dec)
    x2, y2 = P.stars_to_pixels(W(), 10.5, -4.0)
    assert np.isclose(x2, 50.0) and np.isclose(y2, 100.0)
    with pytest.raises(TypeError):
        P.stars_to_pixels(object(), 1.0, 2.0)


def test_flatten_meta_is_json_safe():
    m = P.flatten_meta({"a": {"b": np.float32(1.5), "c": np.zeros((3, 3))}, "d": [1, 2], "e": None})
    assert m["a.b"] == 1.5 and m["a.c"].startswith("<array") and m["d"] == [1, 2] and m["e"] is None
    json.dumps(m)

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
import json
import sys

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


def test_simulation_page_data_links_are_collected_and_capped():
    html = "".join(f"<a href='lc_{i:05d}.csv'>lc</a>" for i in range(300)) + \
        "<a href='index.html'>idx</a><a href='truth.parquet'>t</a>"
    sess = FakeSession([("Microlensing_Data_Challenge", FakeResp(200, html))])
    rec = A.probe(CONF, session=sess, timeout_s=1.0)
    ep = rec["endpoints"]["sim:microlensing_data_challenge"]
    assert ep["status"] == "ok" and len(ep["data_links"]) == A.MAX_SIM_LINKS
    assert all(link.endswith(".csv") for link in ep["data_links"])
    assert ep["data_links"][0].startswith("https://roman.ipac.caltech.edu/sims/")
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
    assert inv["listings"]["s3:openuniverse2024"]["walk"]["n_requests"] == 2
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

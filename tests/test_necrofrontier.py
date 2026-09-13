"""Offline tests for the NECROFRONTIER scripts (data-source probe and prior-art sweep).

The network is never touched (``tests/conftest.py`` raises on any socket); only
the pure functions are exercised: endpoint-table integrity, the per-endpoint
verdict, the per-signature readiness reduction, and the decoy-aware abstract
scan of the literature sweep.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load("necrofrontier_probe")


@pytest.fixture(scope="module")
def fetch(tmp_path_factory):
    import os
    # The sweep script creates its output directory at import time; point it
    # at a temporary directory so a test run never writes under results/.
    os.environ["NECROFRONTIER_OUT"] = str(tmp_path_factory.mktemp("necrofrontier_lit"))
    return _load("necrofrontier_fetch")


# ---------------------------------------------------------------- probe ----
def test_endpoint_table_is_well_formed(probe):
    names = list(probe.REST) + list(probe.TAP)
    assert len(names) == len(set(names)), "endpoint names must be unique"
    for name, spec in probe.REST.items():
        assert spec["url"].startswith(("http://", "https://")), name
        assert spec["expect"] and all(isinstance(p, str) for p in spec["expect"]), name
        assert spec["signatures"] and all(re.fullmatch(r"S\d+", s) for s in spec["signatures"]), name
        assert spec["why"], name
        for p in spec["expect"]:
            re.compile(p)
    for name, spec in probe.TAP.items():
        assert spec["service"].startswith("https://"), name
        assert "SELECT" in spec["adql"].upper(), name
        assert spec["signatures"], name


def test_every_new_signature_has_at_least_one_endpoint(probe):
    covered = set()
    for spec in list(probe.REST.values()) + list(probe.TAP.values()):
        covered.update(spec["signatures"])
    for sig in ("S46", "S47", "S48", "S50", "S51", "S52", "S54", "S55", "S56",
                "S57", "S58", "S59", "S60", "S61"):
        assert sig in covered, f"{sig} has no data-source probe"


def test_classify_reached_with_product(probe):
    rec = {"status": 200, "body": "The Presolar Grain Database: Silicon Carbide download .xlsx"}
    assert probe.classify(rec, [r"(?i)silicon\s+carbide", r"(?i)nothing"]) == "REACHED_WITH_PRODUCT"
    assert rec["expect_hits"] == [r"(?i)silicon\s+carbide"]


def test_classify_reached_without_product_is_a_finding(probe):
    rec = {"status": 200, "body": "<html>Welcome to some unrelated page</html>"}
    assert probe.classify(rec, [r"(?i)spherex"]) == "REACHED_NO_PRODUCT"


def test_classify_not_reached_on_error_and_4xx(probe):
    assert probe.classify({"status": None, "body": None}, [r"x"]) == "NOT_REACHED"
    assert probe.classify({"status": 404, "body": "spherex"}, [r"spherex"]) == "NOT_REACHED"
    assert probe.classify({"status": 500, "body": "spherex"}, [r"spherex"]) == "NOT_REACHED"


def test_classify_skipped_is_not_probed(probe):
    assert probe.classify({"skipped": True}, [r"x"]) == "NOT_PROBED"


def test_summarise_readiness_map(probe):
    records = {
        "a": {"verdict": "REACHED_WITH_PRODUCT", "signatures": ["S46"]},
        "b": {"verdict": "REACHED_NO_PRODUCT", "signatures": ["S47", "S46"]},
        "c": {"verdict": "NOT_REACHED", "signatures": ["S48"]},
    }
    s = probe.summarise(records)
    assert s["verdict"] == "PARTIAL_REACH"
    assert s["n_endpoints"] == 3
    assert s["per_signature"]["S46"]["readiness"] == "PARTIAL"
    assert s["per_signature"]["S47"]["readiness"] == "BLOCKED"
    assert s["per_signature"]["S48"]["readiness"] == "BLOCKED"
    records["b"]["verdict"] = "REACHED_WITH_PRODUCT"
    assert probe.summarise(records)["per_signature"]["S46"]["readiness"] == "READY"


def test_summarise_degrades_honestly_when_nothing_answers(probe):
    records = {"a": {"verdict": "NOT_REACHED", "signatures": ["S46"]}}
    assert probe.summarise(records)["verdict"] == "NO_DATA_SOURCE_REACHED"
    assert probe.summarise({})["verdict"] == "NOTHING_PROBED"


def test_render_md_lists_every_endpoint(probe):
    records = {"a": {"verdict": "REACHED_WITH_PRODUCT", "signatures": ["S46"], "status": 200,
                     "bytes": 10, "expect_hits": ["x"]}}
    md = probe.render_md(probe.summarise(records), records)
    assert "`a`" in md and "S46" in md and "REACHED_WITH_PRODUCT" in md


# ---------------------------------------------------------------- fetch ----
def test_fetch_groups_are_well_formed(fetch):
    assert fetch.GROUPS, "no query groups"
    for g, spec in fetch.GROUPS.items():
        assert re.fullmatch(r"g\d+_[a-z_]+", g), g
        for key in ("question", "target", "decoys", "boosters", "queries", "by_id",
                    "by_title", "interpretation"):
            assert key in spec, (g, key)
        assert spec["target"], g
        for aid, (arxiv_id, frag, conf) in spec["by_id"].items():
            assert re.fullmatch(r"(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})", arxiv_id), (g, aid, arxiv_id)
            assert frag and conf in ("high", "medium", "low"), (g, aid)
        assert all(isinstance(q, str) and q for q in spec["queries"].values()), g


def test_scan_text_fires_target_and_tags_decoys(fetch):
    g = "g1_isotope_purity"
    r = fetch.scan_text(g, "Isotopically pure silicon as a technosignature",
                        "We propose that isotope separation leaves grains with delta29Si "
                        "near -999 permil without any supernova partner signature.")
    assert r is not None and r["target_matches"]
    r2 = fetch.scan_text(g, "Presolar SiC X grains as a technosignature test case",
                         "Large 28Si excesses with 44Ti and 26Al from core-collapse supernova "
                         "nucleosynthesis; delta29Si down to -700 permil.")
    assert r2 is not None and "supernova_grain" in r2["decoys"]
    # No target phrase at all: the natural-background paper is not a hit.
    assert fetch.scan_text(g, "Presolar SiC X grains from supernovae",
                           "Large 28Si excesses with 44Ti from core-collapse supernovae.") is None


def test_scan_text_returns_none_without_target(fetch):
    assert fetch.scan_text("g1_isotope_purity", "Galaxy clusters at z=2",
                           "We measure the Sunyaev-Zeldovich effect.") is None


def test_scan_over_synthetic_atom_files(fetch, tmp_path):
    atom = ("<feed><entry><id>http://arxiv.org/abs/2401.00001v1</id>"
            "<title>A star-pair beam spillover search</title>"
            "<summary>We select star pairs such that Earth lies on the extension of the "
            "line between them and intercept interstellar communication beams.</summary>"
            "</entry><entry><id>http://arxiv.org/abs/2401.00002v1</id>"
            "<title>Unrelated</title><summary>Nothing here.</summary></entry></feed>")
    (tmp_path / "arxiv_q_g10_relay_geometry__x.atom").write_text(atom)
    out = fetch.scan(tmp_path)
    assert out["n_abstracts_scanned"] == 2
    assert out["per_group_counts"]["g10_relay_geometry"]["hits"] >= 1
    assert (tmp_path / "concept_scan.json").exists()

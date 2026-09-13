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


def test_arxiv_to_ads_query_translation(fetch):
    assert fetch.arxiv_to_ads_query('abs:"technosignature" AND abs:isotop') == 'abs:"technosignature" abs:isotop'
    assert fetch.arxiv_to_ads_query('ti:"Do A-type stars flare?"') == 'title:"Do A-type stars flare?"'
    assert fetch.arxiv_to_ads_query('all:DASCH AND abs:(fading OR dimming)') == 'DASCH abs:(fading OR dimming)'


def test_ads_to_atom_round_trips_through_entries(fetch):
    docs = [{"bibcode": "2025ApJ...979..137C", "title": ["Deuterium & fusion"],
             "abstract": "D/H <depleted>", "identifier": ["2024arXiv241118595C", "arXiv:2411.18595"]},
            {"bibcode": "1980Icar...42..149W", "title": ["Nuclear waste spectrum"], "abstract": "x"}]
    text = fetch.ads_to_atom(docs)
    ents = list(fetch._entries(text))
    assert ents[0][0] == "http://arxiv.org/abs/2411.18595"
    assert ents[0][1] == "Deuterium & fusion" and ents[0][2] == "D/H <depleted>"
    assert ents[1][0] == "bibcode:1980Icar...42..149W"


def test_ads_fetch_without_token_degrades_honestly(fetch, tmp_path):
    fetch.ADS_TOKEN = ""
    before = len(fetch.STATUS)
    assert fetch.ads_fetch("abs:x", tmp_path / "x.atom", 5) is False
    assert fetch.STATUS[before]["ok"] is False
    assert "no ADS_TOKEN" in fetch.STATUS[before]["attempts"][0]["error"]


def test_parse_arxiv_abs_page(fetch):
    page = ('<html><head><title>[2411.18595] Deuterium as a technosignature &amp; more</title>'
            '<meta name="citation_abstract" content="We show D/H &lt; ISM." /></head></html>')
    atom = fetch.parse_arxiv_abs_page("2411.18595", page)
    ents = list(fetch._entries(atom))
    assert ents == [("http://arxiv.org/abs/2411.18595", "Deuterium as a technosignature & more", "We show D/H < ISM.")]
    assert "<entry>" not in fetch.parse_arxiv_abs_page("x", "<html></html>")


def test_openalex_translation_and_atom(fetch):
    assert fetch.arxiv_to_openalex_query('abs:"technosignature" AND abs:(isotope OR isotopic)') == '"technosignature" isotope isotopic'
    works = [{"id": "https://openalex.org/W1", "display_name": "A paper",
              "abstract_inverted_index": {"Second": [1], "First": [0]},
              "locations": [{"landing_page_url": "https://arxiv.org/abs/2411.18595v2"}]},
             {"id": "https://openalex.org/W2", "display_name": "No arXiv", "abstract_inverted_index": None}]
    ents = list(fetch._entries(fetch.openalex_to_atom(works)))
    assert ents[0] == ("http://arxiv.org/abs/2411.18595", "A paper", "First Second")
    assert ents[1][0] == "https://openalex.org/W2" and ents[1][2] == ""

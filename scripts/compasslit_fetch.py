#!/usr/bin/env python3
"""COMPASS literature novelty check (docs/next-question.md).

Question to clear: has anyone searched for SPATIALLY COHERENT PATCHES of
aligned binary orbital poles (as opposed to testing global isotropy), at any
scale — and specifically on Gaia DR3 non-single-star orbits — under any
framing (technosignature or natural)?

Runs on the GitHub runner (sandbox blocks these hosts); commits verbatim
abstracts + citation lists under results/litcheck/compass_*.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/litcheck")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-litcheck/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 4, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name,
                           "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:44s} {len(data):8d}B")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i + 1}) {url} -> {e}")
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}")
    return False


ARXIV = "http://export.arxiv.org/api/query?"

# --- 1. Known prior-art anchors, by arXiv ID (verbatim abstracts) -----------
# IDs from memory; the atom response itself is the verification.
IDS = [
    "1411.4919",   # Agati+ 2015 — orbital poles of nearby binaries, isotropy test
    "2206.05595",  # Gaia DR3 non-single-star catalogue paper
    "2111.01789",  # Hwang+ — wide-binary eccentricities from Gaia (method kin)
    "1706.01013",  # Tokovinin — orbit alignment in hierarchical multiples
    "2007.02885",  # Justesen & Albrecht — spin-orbit alignment in binaries
]
get(ARXIV + urllib.parse.urlencode(
    {"id_list": ",".join(IDS), "max_results": len(IDS)}),
    OUT / "compass_arxiv_ids.atom")
time.sleep(3)

# --- 2. arXiv full-text searches (any framing) ------------------------------
QUERIES = {
    "poles_aniso": 'all:"orbital poles" AND all:binaries AND all:anisotropy',
    "poles_align": 'all:"orbital pole" AND all:alignment AND all:binary',
    "nss_incl": 'all:"non-single star" AND all:"Gaia DR3" AND all:inclination',
    "spin_orbit_wide": 'all:"spin-orbit alignment" AND all:"wide binaries"',
    "mutual_incl_field": 'all:"mutual inclination" AND all:"field binaries"',
    "techno_geo": 'all:technosignature AND (all:alignment OR all:geometry)',
    "seti_orbit": 'abs:SETI AND all:"orbital elements" AND all:Gaia',
    "aligned_patches": 'all:aligned AND all:"orbital planes" AND all:coherent',
    "angmom_align": 'all:"angular momentum" AND all:alignment AND all:"wide binaries" AND all:Gaia',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": 40, "sortBy": "relevance"}),
        OUT / f"compass_arxiv_q_{name}.atom")
    time.sleep(3)

# --- 3. Semantic Scholar keyword sweeps + citation lists --------------------
S2 = "https://api.semanticscholar.org/graph/v1/paper/search?"
for name, q in {
    "s2_poles": "binary star orbital poles anisotropy alignment",
    "s2_nss_poles": "Gaia DR3 non-single stars orbit orientation isotropy",
    "s2_techno_align": "technosignature orbital alignment artificial",
}.items():
    get(S2 + urllib.parse.urlencode(
        {"query": q, "limit": 40,
         "fields": "title,abstract,year,externalIds,citationCount"}),
        OUT / f"compass_{name}.json")
    time.sleep(4)

# Citations INTO Agati 2015 — anyone building on the pole-isotropy question.
get("https://api.semanticscholar.org/graph/v1/paper/arXiv:1411.4919/"
    "citations?fields=title,abstract,year,externalIds&limit=200",
    OUT / "compass_s2_agati_citations.json")
time.sleep(4)

# OpenAlex fallback for the same sweep.
get("https://api.openalex.org/works?" + urllib.parse.urlencode(
    {"search": "orbital poles binaries anisotropy alignment",
     "per-page": "50"}),
    OUT / "compass_openalex_poles.json")

(OUT / "compass_summary.json").write_text(json.dumps(
    {"fetched": STATUS, "n_ok": sum(1 for s in STATUS if s.get("ok")),
     "n_fail": sum(1 for s in STATUS if not s.get("ok"))}, indent=2))
print(f"done: {sum(1 for s in STATUS if s.get('ok'))} ok / "
      f"{sum(1 for s in STATUS if not s.get('ok'))} fail")

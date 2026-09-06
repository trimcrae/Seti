#!/usr/bin/env python3
"""LANTERN prior-art sweep: fetch the primary sources on the GitHub runner.

The sandbox blocks arxiv.org / OpenAlex (``CONNECT tunnel failed, response
403``); the Actions runner has egress.  This script establishes, from the
record rather than from memory, whether anyone has ever EXECUTED a search for a
narrow (laser) emission line whose flux vanishes at secondary eclipse, across
the JWST exoplanet time-series archive -- as opposed to proposing it.

Three questions, kept separate because they are different:

1. **What was proposed.**  Kipping & Teachey 2016 (arXiv:1603.08928, "A
   cloaking device for transiting planets") proposed lasers used *during
   transit* to alter/cloak the transit signature and, as a corollary, that a
   laser could be used to broadcast; that is a proposal about the transit
   light curve, not a search of eclipse-phased spectra.
2. **What was executed.**  Optical-SETI line searches on archival spectra
   (e.g. Tellis & Marcy 2015/2017 on Keck/HIRES; Zuckerman et al. on Dyson
   candidates; Marcy 2021/2022 LRIS surveys) are single-epoch, phase-agnostic
   scans of stars.  The question is whether any search has ever used the
   planet's OCCULTATION as the discriminant.
3. **The astrophysical background.**  Planet-origin emission that vanishes at
   eclipse exists and is broad: the day-side thermal spectrum (JWST eclipse
   spectroscopy), and any resolved molecular emission.  The channel's claim
   depends on the NARROW (unresolved) requirement, so the sweep also pulls the
   eclipse-spectroscopy and high-resolution day-side emission literature that
   defines what a non-artificial vanishing feature looks like.

Outputs under ``results/lanternlit/`` (verbatim; nothing is paraphrased):
  arxiv_id_<name>.atom   arXiv API metadata for a specific paper
  arxiv_q_<name>.atom    arXiv API keyword search (discovery)
  txt_<name>.txt         extracted plain text (PDF -> pdftotext) for the anchors
  oa_<name>.json / citedby_<name>.json   OpenAlex records / citation trees
  concept_scan.json      keyword scan over every fetched abstract, with decoys tagged
  summary.json           fetch status for every URL
Modelled on scripts/rustlit_fetch.py.  Never fabricates a literature result: a
fetch that fails is recorded as failed, and a scan with no hits says so.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/lanternlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-lanternlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = 3.0

ARXIV_API = "http://export.arxiv.org/api/query"
OA_API = "https://api.openalex.org/works"


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = PAUSE) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "dest": dest.name, "ok": True, "bytes": len(data)})
            print(f"  ok  {len(data):>9,}B  {dest.name}")
            time.sleep(pause)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  try {i + 1}/{tries} failed: {exc!r}")
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "dest": dest.name, "ok": False})
    return False


# --------------------------------------------------------------------------
# 1. Named papers: the proposal, the executed searches, the background
# --------------------------------------------------------------------------
PAPERS = {
    # name: (arXiv id, title used for the OpenAlex title-search fallback)
    "kipping_teachey_cloaking": ("1603.08928", "A cloaking device for transiting planets"),
    "tellis_marcy_2015": ("1504.03369",
                          "A Search for Optical Laser Emission Using Keck HIRES"),
    "tellis_marcy_2017": ("1704.02535",
                          "A Search for Laser Emission with Megawatt Thresholds from 5600 "
                          "FGKM Stars"),
    "marcy_2021_lris": ("2103.09915",
                        "A search for optical laser emission from Proxima Centauri"),
    "schwartz_townes_1961": ("", "Interstellar and Interplanetary Communication by Optical Masers"),
    "wright_2018_glossary": ("1809.06857", "Exoplanets and SETI"),
    "lingam_loeb_laser": ("1610.00593",
                          "Fast Radio Bursts from Extragalactic Light Sails"),
    "sheikh_2020_nine_axes": ("2004.12184",
                              "Nine Axes of Merit for Technosignature Searches"),
    "hippke_laser_beacons": ("1706.03795",
                             "Interstellar communication. II. Application to the solar "
                             "gravitational lens mission"),
    "jwst_eclipse_spectroscopy_review": ("2306.04643", "JWST secondary eclipse spectroscopy"),
}

# --------------------------------------------------------------------------
# 2. Keyword sweeps.  Target concept in several phrasings, plus decoys.
# --------------------------------------------------------------------------
QUERIES = {
    # --- the target concept ---
    "laser_technosig_transit": 'all:"laser" AND all:"technosignature" AND all:"transit"',
    "secondary_eclipse_technosig": 'all:"secondary eclipse" AND all:"technosignature"',
    "eclipse_seti": 'all:"eclipse" AND all:"SETI" AND all:"exoplanet"',
    "kipping_teachey_cloaking": 'all:"cloaking" AND all:"transit" AND all:"laser"',
    "optical_seti_jwst": 'all:"optical SETI" AND all:"JWST"',
    "laser_seti_infrared": 'all:"laser" AND all:"SETI" AND all:"infrared"',
    "narrow_emission_line_eclipse": 'all:"narrow emission line" AND all:"exoplanet" AND all:"eclipse"',
    "artificial_spectral_line_exoplanet": 'all:"artificial" AND all:"spectral line" AND all:"exoplanet"',
    "monochromatic_exoplanet_emission": 'all:"monochromatic" AND all:"exoplanet" AND all:"emission"',
    "phase_resolved_technosignature": 'all:"phase-resolved" AND all:"technosignature"',
    "technosignature_jwst_archive": 'all:"technosignature" AND all:"JWST"',
    "laser_line_search_spectra": 'all:"laser" AND all:"search" AND all:"spectra" AND all:"emission line" AND cat:astro-ph.EP',
    # --- the executed optical-SETI line searches ---
    "optical_seti_archival_spectra": 'all:"optical SETI" AND all:"archival"',
    "laser_emission_search_stars": 'all:"laser emission" AND all:"search" AND all:"stars"',
    # --- the astrophysical background (broad vanishing emission) ---
    "jwst_eclipse_spectroscopy_dayside": 'all:"JWST" AND all:"secondary eclipse" AND all:"emission spectrum"',
    "highres_dayside_emission": 'all:"high-resolution" AND all:"dayside" AND all:"emission" AND all:"exoplanet"',
    "exoplanet_line_emission_eclipse": 'all:"line emission" AND all:"exoplanet" AND all:"eclipse"',
    # --- decoys ---
    "decoy_stellar_flare_lines": 'all:"flare" AND all:"emission line" AND all:"M dwarf" AND all:"JWST"',
    "decoy_planet_auroral_emission": 'all:"auroral" AND all:"emission" AND all:"exoplanet" AND all:"eclipse"',
    "decoy_chromospheric_eclipse": 'all:"chromospheric" AND all:"secondary eclipse"',
}

MAX_RESULTS = 60


def arxiv_id(name: str, aid: str) -> None:
    url = f"{ARXIV_API}?search_query=&id_list={aid}&start=0&max_results=1"
    get(url, OUT / f"arxiv_id_{name}.atom")


def arxiv_query(name: str, q: str) -> None:
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&start=0&max_results={MAX_RESULTS}"
           f"&sortBy=relevance&sortOrder=descending")
    get(url, OUT / f"arxiv_q_{name}.atom")


def fulltext(name: str, aid: str) -> None:
    pdf = OUT / f"pdf_{name}.pdf"
    if get(f"https://arxiv.org/pdf/{aid}", pdf):
        try:
            subprocess.run(["pdftotext", str(pdf), str(OUT / f"txt_{name}.txt")],
                           check=False, timeout=180)
            pdf.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  pdftotext failed for {name}: {exc!r}")


def openalex(name: str, title: str, aid: str) -> None:
    """Work record plus the works citing it.  The arXiv DOI alone is not enough
    (it often resolves to a stub with cited_by_count 0), so a title search is
    run as well and the record with the larger citation count is kept, with
    the choice logged in ``oa_pick_<name>.json``."""
    cands: list[dict] = []
    if aid and get(f"{OA_API}/https://doi.org/10.48550/arXiv.{aid}", OUT / f"oa_doi_{name}.json"):
        try:
            cands.append(json.loads((OUT / f"oa_doi_{name}.json").read_text()))
        except Exception:  # noqa: BLE001
            pass
    q = urllib.parse.quote(title)
    if get(f"{OA_API}?filter=title.search:{q}&per-page=10&sort=cited_by_count:desc",
           OUT / f"oa_search_{name}.json"):
        try:
            res = json.loads((OUT / f"oa_search_{name}.json").read_text()).get("results", [])
            cands.extend(res[:5])
        except Exception:  # noqa: BLE001
            pass
    cands = [c for c in cands if isinstance(c, dict) and c.get("id")]
    if not cands:
        (OUT / f"oa_pick_{name}.json").write_text(json.dumps(
            {"resolved": False, "note": "no OpenAlex record found"}, indent=2))
        return
    best = max(cands, key=lambda c: int(c.get("cited_by_count") or 0))
    (OUT / f"oa_{name}.json").write_text(json.dumps(best, indent=2))
    (OUT / f"oa_pick_{name}.json").write_text(json.dumps({
        "resolved": True, "chosen_id": best.get("id"),
        "chosen_cited_by_count": int(best.get("cited_by_count") or 0),
        "candidates": [{"id": c.get("id"), "title": c.get("display_name"),
                        "cited_by_count": int(c.get("cited_by_count") or 0),
                        "type": c.get("type")} for c in cands],
        "warning": ("cited_by_count == 0 here means the citation-tree leg is "
                    "UNINFORMATIVE for this paper, not that nobody cited it"),
    }, indent=2))
    wid = str(best.get("id", "")).rsplit("/", 1)[-1]
    if wid and int(best.get("cited_by_count") or 0) > 0:
        get(f"{OA_API}?filter=cites:{wid}&per-page=200", OUT / f"citedby_{name}.json")


# --------------------------------------------------------------------------
# 3. Decoy-aware concept scan
# --------------------------------------------------------------------------
# A genuine hit: an abstract that pairs a laser / artificial / narrow-line
# signal with the planet's eclipse (occultation) or orbital phase as the test.
TARGET = re.compile(
    r"(laser|artificial|technosignature|beacon|monochromatic|SETI)\W{0,80}"
    r"(eclipse|occultation|orbital phase|phase-resolved|behind the star|secondary)"
    r"|(eclipse|occultation|orbital phase|phase-resolved|secondary)\W{0,80}"
    r"(laser|artificial|technosignature|beacon|monochromatic|SETI)", re.I | re.S)

DECOYS = {
    "transit_cloaking_proposal": re.compile(r"cloak", re.I),
    "thermal_dayside_emission": re.compile(r"(thermal|brightness temperature|dayside|day-side)"
                                           r"\W{0,40}(emission|spectrum)", re.I),
    "stellar_or_flare_line": re.compile(r"(flare|chromospher|corona)", re.I),
    "radio_not_optical": re.compile(r"\b(radio|GHz|MHz)\b", re.I),
    "eclipsing_binary_not_planet": re.compile(r"eclipsing binar", re.I),
}


def scan() -> dict:
    hits, n_abs = [], 0
    for p in sorted(OUT.glob("arxiv_*.atom")):
        text = p.read_text(errors="ignore")
        for m in re.finditer(r"<entry>(.*?)</entry>", text, re.S):
            e = m.group(1)
            n_abs += 1
            aid = (re.search(r"<id>(.*?)</id>", e, re.S) or [None, ""])[1].strip()
            title = " ".join((re.search(r"<title>(.*?)</title>", e, re.S)
                              or [None, ""])[1].split())
            summ = " ".join((re.search(r"<summary>(.*?)</summary>", e, re.S)
                             or [None, ""])[1].split())
            blob = f"{title} {summ}"
            if not TARGET.search(blob):
                continue
            tags = [k for k, rx in DECOYS.items() if rx.search(blob)]
            hits.append({"arxiv": aid, "title": title, "decoys": tags,
                         "source_query": p.name, "snippet": blob[:600]})
    clean = [h for h in hits if not h["decoys"]]
    out = {
        "n_abstracts_scanned": n_abs,
        "n_target_regex_hits": len(hits),
        "n_after_decoy_removal": len(clean),
        "decoy_free_hits": clean,
        "all_hits": hits,
        "interpretation": (
            "A decoy-free hit is an abstract that ties a laser / artificial / "
            "narrow-line signal to the planet's ECLIPSE or orbital phase as the "
            "test. transit_cloaking_proposal tags the Kipping & Teachey proposal "
            "(a transit-light-curve idea, not an eclipse-phased line search); "
            "thermal_dayside_emission tags the broad astrophysical background; "
            "stellar_or_flare_line, radio_not_optical and "
            "eclipsing_binary_not_planet are off-target. An empty decoy-free list "
            "is evidence the eclipse-gated narrow-line search is unexecuted, NOT "
            "proof; docs/lantern.md states the position as 'to be verified'."),
    }
    (OUT / "concept_scan.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    print("== named papers ==")
    for name, (aid, title) in PAPERS.items():
        print(f"-- {name} ({aid or 'no arXiv id'})")
        if aid:
            arxiv_id(name, aid)
            if name in ("kipping_teachey_cloaking", "tellis_marcy_2017", "marcy_2021_lris"):
                fulltext(name, aid)
        openalex(name, title, aid)

    print("== keyword sweeps ==")
    for name, q in QUERIES.items():
        print(f"-- {name}")
        arxiv_query(name, q)

    print("== decoy-aware concept scan ==")
    res = scan()
    print(json.dumps({k: v for k, v in res.items() if k != "all_hits"}, indent=2)[:4000])

    (OUT / "summary.json").write_text(json.dumps(
        {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "n_urls": len(STATUS),
         "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"]),
         "status": STATUS}, indent=2))
    print(f"\n{sum(1 for s in STATUS if s['ok'])}/{len(STATUS)} fetches ok -> {OUT}")


if __name__ == "__main__":
    main()

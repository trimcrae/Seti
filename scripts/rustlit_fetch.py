#!/usr/bin/env python3
"""RUST novelty check: fetch the primary sources on the GitHub runner.

The sandbox blocks arxiv.org / ADS / OpenAlex (``CONNECT tunnel failed, response
403``) and the session WebSearch budget is spent; the Actions runner has egress.
This script establishes, from the record rather than from memory, whether anyone
has ever built a catalogue of stars whose **aperiodic variability amplitude is
secularly increasing with calendar time**.

Three things have to be established, and they are different questions:

1. **The mechanism exists and proposes no observable.**  Lacki 2025 "Ground to
   Dust" (arXiv:2504.21151, ApJ 985, 191) --- fetch the abstract and full text
   and confirm it contains a collisional-cascade *timescale* and no photometric
   search.
2. **The honest counterweight.**  McInnes 2026 (arXiv:2603.00203) on passively
   stable ring-supported stellar engines and dense-cloud Dyson bubbles, and
   Wright 2020 (arXiv:2006.16734, SerAJ 200, 1) on the instability of monolithic
   spheres.  The channel must state which architectures it is and is not
   sensitive to, and that statement has to be sourced.
3. **The statistic is unoccupied.**  The nearest published machinery is Petz &
   Kochanek 2025, "Life in the Slow Lane" (arXiv:2501.14058) --- 9,361,613
   ASAS-SN sources selected on brightness change > 0.03 mag/yr.  That is a
   **mean-flux slope**.  The question is whether anyone has ever run the
   equivalent on the **second moment**, so this script also pulls the citation
   trees and runs keyword sweeps designed to separate a genuine hit from the
   four known decoys: amplitude rising with *timescale* (red noise), with
   *evolutionary stage*, with *wavelength*, or in a single named star (Polaris).

Outputs under ``results/rustlit/``:
  arxiv_id_<name>.atom   arXiv API metadata for a specific paper
  arxiv_q_<name>.atom    arXiv API keyword search (discovery)
  txt_<name>.txt         extracted plain text (PDF -> pdftotext)
  oa_<name>.json         OpenAlex work record
  citedby_<name>.json    OpenAlex "works citing this" listing
  concept_scan.json      decoy-aware keyword scan over every fetched abstract
  summary.json           fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/rustlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-rustlit/1.0 (mailto:trimcrae@gmail.com)"}
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
# 1. Named papers: the mechanism, the counterweight, the nearest machinery
# --------------------------------------------------------------------------
PAPERS = {
    # name: (arXiv id, title used for the OpenAlex title-search fallback)
    "lacki_ground_to_dust": ("2504.21151",
                             "Ground to Dust: Collisional Cascades and the Fate "
                             "of Kardashev II Megaswarms"),
    "mcinnes2026_stable": ("2603.00203",
                           "Stellar engines and Dyson bubbles can be stable"),
    "wright2020_spheres": ("2006.16734", "Dyson Spheres"),
    "petz_kochanek_slow_lane": ("2501.14058",
                                "Life in the Slow Lane: A Search for Long Term "
                                "Variability in ASAS-SN"),
    "kochanek_asassn_slow": ("2011.02502",
                             "The ASAS-SN catalog of variable stars"),
    "hephaistos2": ("2405.08657",
                    "Project Hephaistos II: Dyson sphere candidates from Gaia "
                    "DR3, 2MASS, and WISE"),
}

# --------------------------------------------------------------------------
# 2. Keyword sweeps.  Written to catch the target concept AND the decoys, so a
#    null is interpretable rather than merely empty.
# --------------------------------------------------------------------------
QUERIES = {
    # --- the target concept, several phrasings ---
    "amp_increase_time": 'all:"variability amplitude" AND all:"increasing with time"',
    "amp_increase_secular": 'all:"amplitude" AND all:"secular increase" AND cat:astro-ph.SR',
    "scatter_increase": 'all:"increasing scatter" AND all:"light curve"',
    "rms_increase_time": 'all:"RMS variability" AND all:"increase" AND all:"epoch"',
    "growing_variability": 'all:"growing variability" OR all:"variability is increasing"',
    "changing_variability": 'all:"changing variability" AND (cat:astro-ph.SR OR cat:astro-ph.EP)',
    "variability_evolution_survey": 'all:"variability" AND all:"evolution" AND all:"ZTF"',
    "second_moment_survey": 'all:"excess variance" AND all:"survey" AND cat:astro-ph.SR',
    # --- the mechanism and its citation neighbourhood ---
    "megastructure_cascade": 'all:"collisional cascade" AND all:"megastructure"',
    "dyson_swarm_debris": 'all:"Dyson swarm" AND all:"debris"',
    "station_keeping_swarm": 'all:"station-keeping" AND all:"swarm" AND cat:astro-ph.EP',
    "kessler_technosignature": 'all:"Kessler" AND all:"technosignature"',
    # --- architecture stability, for the sensitivity statement ---
    "dyson_stability": 'all:"Dyson sphere" AND all:"stability"',
    "stellar_engine_ring": 'all:"stellar engine" AND all:"ring"',
    # --- the decoys, fetched deliberately so the scan can distinguish them ---
    "decoy_polaris_amplitude": 'all:"Polaris" AND all:"amplitude" AND all:"increase"',
    "decoy_amp_vs_timescale": 'all:"amplitude" AND all:"increases with timescale"',
    "decoy_yso_evolutionary": 'all:"variability" AND all:"evolutionary stage" AND all:"YSO"',
    "decoy_agn_red_noise": 'all:"red noise" AND all:"amplitude" AND all:"timescale" AND cat:astro-ph.GA',
    "decoy_amp_vs_wavelength": 'all:"amplitude" AND all:"shorter wavelength" AND all:"increases"',
    # --- confounder physics the vetting has to know about ---
    "yso_dipper_longterm": 'all:"dipper" AND all:"long-term" AND all:"variability"',
    "cv_amplitude_evolution": 'all:"cataclysmic variable" AND all:"amplitude" AND all:"evolution"',
    "ztf_variability_catalogue": 'all:"ZTF" AND all:"variability catalog"',
    "atlas_variable_stars": 'all:"ATLAS" AND all:"variable star catalog"',
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
    """Work record plus the list of works citing it --- the 'did anyone run it' test.

    **The arXiv DOI is not enough.**  ``10.48550/arXiv.<id>`` resolves to the
    *preprint stub*, which OpenAlex frequently reports with
    ``cited_by_count = 0`` even for heavily-cited papers (the 2026-07-26 run
    returned 0 for Wright 2020, which is certainly wrong).  A citation tree
    fetched that way is not evidence of anything.  So we additionally search by
    title and keep whichever record has the **larger** citation count, recording
    in ``oa_pick_<name>.json`` which route won and what each returned --- an
    uninformative leg must be visibly uninformative, not silently empty.
    """
    cands: list[dict] = []
    if get(f"{OA_API}/https://doi.org/10.48550/arXiv.{aid}", OUT / f"oa_doi_{name}.json"):
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
        get(f"{OA_API}?filter=cites:{wid}&per-page=200",
            OUT / f"citedby_{name}.json")


# --------------------------------------------------------------------------
# 3. Decoy-aware concept scan
# --------------------------------------------------------------------------
TARGET = re.compile(
    r"(amplitude|scatter|rms|dispersion|variance|variability)\W{0,40}"
    r"(increas|grow|ris|larger|strengthen)\w*\W{0,40}"
    r"(with|over|as a function of)?\W{0,20}"
    r"(time|epoch|year|calendar|baseline|decade|season|secular)", re.I)

DECOYS = {
    "with_timescale": re.compile(r"increas\w*\W{0,30}with\W{0,20}(the\W)?(time-?scale|lag|"
                                 r"frequency)", re.I),
    "with_evolutionary_stage": re.compile(r"evolutionary\W(stage|class|phase)", re.I),
    "with_wavelength": re.compile(r"(shorter|bluer|longer|redder)\W{0,20}wavelength", re.I),
    "single_named_star": re.compile(r"\bPolaris\b|\bRR Lyr\b|\bBetelgeuse\b", re.I),
    "period_not_amplitude": re.compile(r"period\W{0,20}(increas|change|chang)", re.I),
}


def scan() -> dict:
    """Scan every fetched abstract for the target concept, tagging the decoys."""
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
                         "source_query": p.name,
                         "snippet": blob[:600]})
    clean = [h for h in hits if not h["decoys"]]
    out = {
        "n_abstracts_scanned": n_abs,
        "n_target_regex_hits": len(hits),
        "n_after_decoy_removal": len(clean),
        "decoy_free_hits": clean,
        "all_hits": hits,
        "interpretation": (
            "A decoy-free hit is a paper whose abstract claims a variability "
            "amplitude/scatter increasing with CALENDAR TIME. Hits tagged "
            "with_timescale are red-noise statements, with_evolutionary_stage are "
            "population statements, with_wavelength are colour statements, and "
            "single_named_star are individual objects, not catalogues."),
    }
    (OUT / "concept_scan.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    print("== named papers ==")
    for name, (aid, title) in PAPERS.items():
        print(f"-- {name} ({aid})")
        arxiv_id(name, aid)
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
        {"n_urls": len(STATUS),
         "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"]),
         "status": STATUS}, indent=2))
    print(f"\n{sum(1 for s in STATUS if s['ok'])}/{len(STATUS)} fetches ok "
          f"-> {OUT}")


if __name__ == "__main__":
    main()

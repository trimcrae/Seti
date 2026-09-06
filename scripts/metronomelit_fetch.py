#!/usr/bin/env python3
"""METRONOME prior-art sweep: fetch the record on the GitHub runner.

The sandbox blocks arxiv.org (``CONNECT tunnel failed, response 403``); the
Actions runner has egress.  This script establishes, from the record rather
than from memory, whether anyone has ever searched the public flare catalogues
for **strictly periodic flare timing** --- a clock in the *event times* of
catalogued flares --- as opposed to the four things that sound like it and are
not:

1. **QPPs** --- quasi-periodic pulsations *within* one flare (seconds to
   minutes; MHD oscillations of the flaring loop).  The dominant decoy.
2. **Rotational-phase dependence of flare occurrence** --- whether flares
   cluster at a rotational phase (Hawley et al. 2014; Doyle et al. 2018/2019;
   Roettenbacher & Vida 2018).  A *natural* quasi-periodicity with large
   jitter; this channel's ``rotation_alias`` veto exists because of it.
3. **Periodic bursts from compact objects** --- FRB / magnetar periodicity.
   Different objects, different physics, but the *statistics* (H-test,
   window-resampled nulls) are shared and worth reading.
4. **Flare waiting-time distributions** --- Poisson vs power-law waiting times
   (solar and stellar).  The natural null this channel resamples.

Nothing here is asserted from memory: named papers are fetched by *title
search* where the arXiv id is not certain, and where an id is given the fetched
title is compared against the expected one and any mismatch is recorded in
``id_title_check.json``.  Verbatim abstracts are saved; nothing is paraphrased.

Outputs under ``results/metronomelit/``:
  arxiv_q_<name>.atom     arXiv API keyword / title search (verbatim)
  arxiv_id_<name>.atom    arXiv API metadata for a specific id (verbatim)
  concept_scan.json       decoy-aware scan over every fetched abstract
  id_title_check.json     did each asserted id resolve to the expected title?
  summary.json            fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/metronomelit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-metronomelit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = 3.0
ARXIV_API = "http://export.arxiv.org/api/query"
MAX_RESULTS = 60


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
# 1. Named papers.  Ids are asserted ONLY where confidence is high, and even
#    then the fetched title is checked against the expected title.  Everything
#    else is a title search, which cannot be wrong about the id.
# --------------------------------------------------------------------------
PAPERS_BY_ID = {
    # name: (arXiv id, expected title fragment)
    "davenport2016_kepler_flares": ("1607.03494", "Kepler Catalog of Stellar Flares"),
    "gunther2020_tess_flares": ("1901.00443", "Stellar Flares from the First TESS Data Release"),
}

PAPERS_BY_TITLE = {
    "yang_liu2019_kepler_flare_catalog": "The Flare Catalog and the Flare Activity in the Kepler Mission",
    "pietras2022_tess_flares": "Statistical analysis of stellar flares from the first three years of TESS observations",
    "tu2020_tess_superflares": "Superflares on solar-type stars from the first year observation of TESS",
    "hawley2014_gj1243": "Kepler Flares I. Active and Inactive M dwarfs",
    "doyle2018_rotational_phase": "Investigating the rotational phase of stellar flares on M dwarfs using K2 short cadence data",
    "roettenbacher_vida2018": "The connection between starspots and flares on Kepler stars",
    "dejager1989_htest": "A powerful test for weak periodic signals with unknown light curve shape",
    "dejager_busching2010": "The H-test probability distribution revisited",
    "besag_clifford1991": "Sequential Monte Carlo p-values",
    "wheatland2000_waiting_times": "The origin of the solar flare waiting-time distribution",
    "chime_frb_periodicity": "Periodic activity from a fast radio burst source",
    "sheikh2020_nine_axes": "The Nine Axes of Merit for Technosignature Searches",
}

# --------------------------------------------------------------------------
# 2. Keyword sweeps.  The brief's phrasings plus decoys fetched deliberately
#    so a null is interpretable rather than merely empty.
# --------------------------------------------------------------------------
QUERIES = {
    # --- the target concept ---
    "periodic_flares": 'all:"periodic flares"',
    "flare_periodicity_technosignature": 'all:"flare" AND all:"periodicity" AND all:"technosignature"',
    "quasi_periodic_flare_timing": 'all:"quasi-periodic" AND all:"flare timing"',
    "flare_waiting_time_distribution": 'all:"flare" AND all:"waiting time distribution"',
    "technosignature_timing": 'all:"technosignature" AND all:"timing"',
    "periodic_flare_timing_stars": 'all:"periodic" AND all:"flare" AND all:"timing" AND cat:astro-ph.SR',
    "flare_recurrence_period": 'all:"flare" AND all:"recurrence" AND all:"period" AND cat:astro-ph.SR',
    "flare_catalog_periodicity": 'all:"flare catalog" AND all:"periodic"',
    "clock_beacon_pulsed_optical": 'all:"beacon" AND all:"pulsed" AND all:"optical" AND all:"SETI"',
    "artificial_periodic_signal_stars": 'all:"artificial" AND all:"periodic" AND all:"technosignature" AND all:"stars"',
    "h_test_optical_events": 'all:"H-test" AND all:"periodicity" AND all:"events"',
    "rayleigh_test_flares": 'all:"Rayleigh test" AND all:"flares"',
    # --- decoys, fetched on purpose ---
    "decoy_qpp": 'all:"quasi-periodic pulsations" AND all:"stellar flares"',
    "decoy_rotational_phase_flares": 'all:"flares" AND all:"rotational phase" AND cat:astro-ph.SR',
    "decoy_frb_periodicity": 'all:"periodic" AND all:"fast radio burst" AND all:"activity"',
    "decoy_magnetar_burst_periodicity": 'all:"magnetar" AND all:"burst" AND all:"periodicity"',
    "decoy_solar_flare_waiting_times": 'all:"solar flare" AND all:"waiting times" AND all:"Poisson"',
    "decoy_flare_rate_activity_cycle": 'all:"flare rate" AND all:"activity cycle" AND all:"Kepler"',
    # --- the substrate catalogues and the systematics they carry ---
    "kepler_flare_catalog": 'all:"Kepler" AND all:"flare catalog"',
    "tess_flare_catalog": 'all:"TESS" AND all:"flare catalog"',
    "kepler_momentum_dump_systematics": 'all:"Kepler" AND all:"momentum dump"',
    "tess_momentum_dump": 'all:"TESS" AND all:"momentum dump"',
    "argabrightening": 'all:"Argabrightening"',
    "rr_lyrae_flare_misclassification": 'all:"flare" AND all:"false positive" AND all:"RR Lyrae"',
}


def arxiv_query(name: str, q: str) -> None:
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&start=0&max_results={MAX_RESULTS}&sortBy=relevance&sortOrder=descending")
    get(url, OUT / f"arxiv_q_{name}.atom")


def arxiv_title(name: str, title: str) -> None:
    q = 'ti:"' + title.replace('"', "") + '"'
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&start=0&max_results=5&sortBy=relevance&sortOrder=descending")
    get(url, OUT / f"arxiv_q_title_{name}.atom")


def arxiv_id(name: str, aid: str) -> None:
    get(f"{ARXIV_API}?search_query=&id_list={aid}&start=0&max_results=1",
        OUT / f"arxiv_id_{name}.atom")


def _entries(text: str):
    for m in re.finditer(r"<entry>(.*?)</entry>", text, re.S):
        e = m.group(1)
        aid = (re.search(r"<id>(.*?)</id>", e, re.S) or [None, ""])[1].strip()
        title = " ".join((re.search(r"<title>(.*?)</title>", e, re.S) or [None, ""])[1].split())
        summ = " ".join((re.search(r"<summary>(.*?)</summary>", e, re.S) or [None, ""])[1].split())
        yield aid, title, summ


def id_title_check() -> dict:
    """Did each asserted arXiv id resolve to the expected title?  Recorded, not assumed."""
    out = {}
    for name, (aid, frag) in PAPERS_BY_ID.items():
        p = OUT / f"arxiv_id_{name}.atom"
        if not p.exists():
            out[name] = {"id": aid, "fetched": False}
            continue
        ents = list(_entries(p.read_text(errors="ignore")))
        title = ents[0][1] if ents else ""
        out[name] = {"id": aid, "fetched": True, "title_fetched": title,
                     "expected_fragment": frag,
                     "match": frag.lower() in title.lower()}
    (OUT / "id_title_check.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# 3. Decoy-aware concept scan
# --------------------------------------------------------------------------
TARGET = re.compile(
    r"(strict(ly)?|precise(ly)?|regular(ly)?|periodic(ally)?|clock|coherent)\W{0,30}"
    r"(flare|burst|brightening|event)s?\W{0,40}(timing|times|occurrence|recurrence|interval)",
    re.I)
TARGET2 = re.compile(r"(flare|brightening)s?\W{0,20}(occur|recur)\w*\W{0,30}(period|periodic|regular)",
                     re.I)

DECOYS = {
    "qpp_within_flare": re.compile(r"quasi-?periodic pulsation|QPP", re.I),
    "rotational_phase": re.compile(r"rotational? phase|phase of rotation|spot", re.I),
    "compact_object": re.compile(r"fast radio burst|FRB|magnetar|pulsar|neutron star", re.I),
    "solar": re.compile(r"\bsolar\b|\bSun\b", re.I),
    "waiting_time_only": re.compile(r"waiting[- ]time", re.I),
}


def scan() -> dict:
    hits, n_abs = [], 0
    for p in sorted(OUT.glob("arxiv_*.atom")):
        for aid, title, summ in _entries(p.read_text(errors="ignore")):
            n_abs += 1
            blob = f"{title} {summ}"
            if not (TARGET.search(blob) or TARGET2.search(blob)):
                continue
            tags = [k for k, rx in DECOYS.items() if rx.search(blob)]
            hits.append({"arxiv": aid, "title": title, "decoys": tags,
                         "source_query": p.name, "snippet": blob[:600]})
    seen, dedup = set(), []
    for h in hits:
        if h["arxiv"] in seen:
            continue
        seen.add(h["arxiv"])
        dedup.append(h)
    clean = [h for h in dedup if not h["decoys"]]
    out = {
        "n_abstracts_scanned": n_abs,
        "n_target_regex_hits": len(dedup),
        "n_after_decoy_removal": len(clean),
        "decoy_free_hits": clean,
        "all_hits": dedup,
        "interpretation": (
            "A decoy-free hit is an abstract that speaks of periodic / regular / "
            "clock-like OCCURRENCE TIMES of stellar flares or brightenings.  Hits tagged "
            "qpp_within_flare are oscillations inside one flare; rotational_phase are the "
            "natural quasi-periodicity this channel vetoes; compact_object are FRB/magnetar "
            "work (shared statistics, different objects); solar and waiting_time_only are "
            "the natural null.  The novelty position in docs/metronome.md is to be read "
            "against decoy_free_hits, and it is not established until this file exists."),
    }
    (OUT / "concept_scan.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    print("== named papers by id (title-checked) ==")
    for name, (aid, _) in PAPERS_BY_ID.items():
        print(f"-- {name} ({aid})")
        arxiv_id(name, aid)
    print("== named papers by title ==")
    for name, title in PAPERS_BY_TITLE.items():
        print(f"-- {name}")
        arxiv_title(name, title)
    print("== keyword sweeps ==")
    for name, q in QUERIES.items():
        print(f"-- {name}")
        arxiv_query(name, q)
    print("== id/title check ==")
    print(json.dumps(id_title_check(), indent=2))
    print("== decoy-aware concept scan ==")
    res = scan()
    print(json.dumps({k: v for k, v in res.items() if k != "all_hits"}, indent=2)[:4000])
    (OUT / "summary.json").write_text(json.dumps(
        {"n_urls": len(STATUS), "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"]), "status": STATUS}, indent=2))
    print(f"\n{sum(1 for s in STATUS if s['ok'])}/{len(STATUS)} fetches ok -> {OUT}")


if __name__ == "__main__":
    main()

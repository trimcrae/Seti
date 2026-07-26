#!/usr/bin/env python3
"""Literature survey fetcher: disappearing / failed supernovae & vanishing stars.

The sandbox egress policy blocks arxiv.org / Semantic Scholar / OpenAlex, so this
runs on the GitHub Actions runner (CLAUDE.md acquisition pattern) and commits
verbatim abstracts + full text back to the branch.

Outputs under results/vanishlit/:
  arxiv_ids_<n>.atom      - arXiv API metadata (title/authors/abstract) by ID
  arxiv_q_<name>.atom     - arXiv API search results per query
  html_<id>.html          - arXiv full-text HTML for argument-critical papers
  s2_<name>.json          - Semantic Scholar records / citation lists
  summary.json            - fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/vanishlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-vanishlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:44s} {len(data):8d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


ARXIV = "http://export.arxiv.org/api/query?"

# --- 1. Papers by arXiv ID (verbatim abstracts). Some IDs are best guesses;
#        the search queries in section 2 are the authoritative resolver. -------
IDS = [
    # LBT "Survey About Nothing" core sequence
    "0802.0456",   # Kochanek+ 2008 A Survey About Nothing (verify)
    "1411.1761",   # Gerke, Kochanek & Stanek 2015 - first candidates
    "1609.01283",  # Adams+ 2017 - confirmation of a disappearing star (N6946-BH1)
    "1610.02402",  # Adams+ 2017 - constraints from 7 yr of data (verify)
    "2104.03318",  # Neustadt+ 2021 - new candidate + failed SN fraction, 11 yr
    "2007.15658",  # Basinger+ 2021 - N6946-BH1, still no star (verify)
    "2310.01514",  # Kochanek+ 2024 - mid-IR counterpart to N6946-BH1
    "2309.16121",  # Beasor+ 2024 - JWST luminous IR source at N6946-BH1
    # 2026-era follow-ups surfaced by search
    "2604.05019",  # neighboring stars of N6946-BH1 / observational chars of failed SNe
    "2601.14497",  # the failed failed-supernova scenario of M31-2014-DS1
    # Vanishing star claims outside LBT
    "2003.02242",  # Allan+ 2020 - PHL 293B / Kinman dwarf disappearance
    # Theory / IR signature of failed SNe
    "1210.6353",   # Lovegrove & Woosley 2013 very low energy transients (verify)
    "1302.0735",   # Piro 2013 taking the "un" out of unnovae (verify)
    "1704.06797",  # Adams+ theory (verify)
]
CH = 8
for i in range(0, len(IDS), CH):
    chunk = IDS[i:i + CH]
    get(ARXIV + urllib.parse.urlencode({"id_list": ",".join(chunk),
                                        "max_results": len(chunk)}),
        OUT / f"arxiv_ids_{i//CH}.atom")
    time.sleep(3.2)

# --- 2. arXiv API searches (Atom) --------------------------------------------
QUERIES = {
    # 1. LBT survey history
    "failed_sn_lbt": 'all:"failed supernova" AND all:"Large Binocular Telescope"',
    "failed_sne_all": 'ti:"failed supernova" OR ti:"failed supernovae"',
    "survey_about_nothing": 'all:"Survey About Nothing"',
    "disappearing_star": 'ti:"disappearing star" OR ti:"disappearing stars"',
    "vanishing_star": 'ti:"vanishing star" OR ti:"vanishing stars" OR ti:"vanished"',
    "unnova": 'all:"unnova" OR all:"unnovae"',
    # 2. N6946-BH1
    "n6946bh1": 'all:"N6946-BH1"',
    "n6946_bh1_alt": 'abs:"NGC 6946" AND abs:"failed supernova"',
    # 3. Other vanishing claims
    "phl293b": 'all:"PHL 293B" OR all:"Kinman dwarf"',
    "lbv_disappear": 'abs:"luminous blue variable" AND (abs:disappear OR abs:vanish)',
    "sn_impostor_fade": 'abs:"supernova impostor" AND (abs:faded OR abs:fading OR abs:disappear)',
    "m31_2014_ds1": 'all:"M31-2014-DS1"',
    "magellanic_vanish": 'abs:"Magellanic" AND (abs:vanishing OR abs:disappearance OR abs:"disappearing")',
    # 4. dust vs vanished methodology
    "rsg_dust_obscured_collapse": 'abs:"red supergiant" AND abs:dust AND (abs:"black hole" OR abs:collapse) AND abs:infrared',
    "spitzer_failed_sn": 'abs:Spitzer AND abs:"failed supernova"',
    "jwst_failed_sn": 'abs:JWST AND (abs:"failed supernova" OR abs:"disappearing star")',
    "wise_massive_star_fade": 'abs:WISE AND abs:"massive star" AND (abs:fading OR abs:disappear)',
    # 5. wide-field / catalogue-scale vanishing searches
    "vasco": 'all:"VASCO" AND (all:vanishing OR all:"appearing sources")',
    "vanishing_appearing": 'all:"Vanishing and Appearing Sources"',
    "vanished_survey_compare": 'abs:vanished AND (abs:"Pan-STARRS" OR abs:"USNO" OR abs:"POSS" OR abs:"Sloan")',
    "gaia_disappear": 'abs:Gaia AND (abs:"disappearing" OR abs:"vanishing") AND abs:star',
    "ztf_disappear": 'abs:"ZTF" AND (abs:disappear OR abs:vanish)',
    # 6. theory: IR appearance / duration of dust signature
    "failed_sn_theory": 'abs:"failed supernova" AND (abs:transient OR abs:"shock breakout" OR abs:"weak transient")',
    "bh_formation_transient": 'abs:"black hole formation" AND abs:transient AND abs:"massive star"',
    "dust_echo_duration": 'abs:"dust" AND abs:"fallback" AND abs:"black hole" AND abs:"supergiant"',
    "rsg_problem": 'abs:"red supergiant problem"',
    # 7. vanished-and-returned / recurrent obscuration
    "gaia17bpp": 'all:"Gaia17bpp"',
    "rcb_decline": 'abs:"R Coronae Borealis" AND abs:decline AND abs:dust',
    "epsilon_aurigae": 'all:"epsilon Aurigae" OR all:"eps Aurigae"',
    "long_period_eclipse_dusty": 'abs:"long-duration" AND abs:eclipse AND abs:dust AND abs:disk',
    "dipper_deep_dimming": 'abs:"deep dimming" OR abs:"dimming event" AND abs:"dust"',
    "tycho_dimming": 'abs:"long duration dimming" OR abs:"multi-year dimming"',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "50", "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)

# also recency-sorted sweeps for the 2023-2026 window
for name, q in {
    "recent_failed_sn": 'all:"failed supernova" OR all:"failed supernovae"',
    "recent_disappear": 'abs:"disappearance" AND abs:"massive star"',
    "recent_vanishing": 'abs:vanishing AND (abs:star OR abs:stars)',
}.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "80", "sortBy": "submittedDate",
         "sortOrder": "descending"}),
        OUT / f"arxiv_recent_{name}.atom")
    time.sleep(3.2)

# --- 3. Full-text HTML of argument-critical papers ---------------------------
for aid in ["2310.01514", "2309.16121", "2104.03318", "2604.05019", "2601.14497"]:
    if not get(f"https://arxiv.org/html/{aid}v1", OUT / f"html_{aid}.html", tries=2):
        get(f"https://arxiv.org/html/{aid}", OUT / f"html_{aid}.html", tries=2)
    time.sleep(3)

# --- 4. Semantic Scholar: citation lists to catch follow-ups ------------------
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2FIELDS = "title,year,abstract,externalIds,venue,authors"
for name, ident in {
    "beasor2024": "arXiv:2309.16121",
    "kochanek2024midir": "arXiv:2310.01514",
    "neustadt2021": "arXiv:2104.03318",
    "adams2017": "arXiv:1609.01283",
    "allan2020": "arXiv:2003.02242",
}.items():
    get(S2 + urllib.parse.quote(ident) + f"?fields={S2FIELDS},citationCount",
        OUT / f"s2_{name}.json", pause=6.0)
    time.sleep(3)
    get(S2 + urllib.parse.quote(ident) + f"/citations?fields={S2FIELDS}&limit=300",
        OUT / f"s2_{name}_citations.json", pause=6.0)
    time.sleep(3)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")

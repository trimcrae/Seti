#!/usr/bin/env python3
"""Dysonian-SETI literature dossier: fetch primary sources on the GitHub runner.

The sandbox egress policy blocks arxiv.org / ADS / IOP / Semantic Scholar, so this
script runs on the Actions runner (CLAUDE.md acquisition pattern) and commits
verbatim metadata + full text back to the branch.

Targets: the Penn State G-HAT / "Ghat" (G with circumflex) infrared survey series
(Wright, Griffith, Sigurdsson, Povich, Mullan, 2014-2016), Richard Carrigan's
IRAS Dyson-sphere searches (2009 and precursors), and later re-analyses.

Outputs under results/dysonlit/:
  arxiv_ids.atom        - arXiv API metadata (title/authors/abstract/DOI/journal-ref)
  arxiv_q_<name>.atom   - arXiv API keyword searches (to catch IDs we do not know)
  txt_<id>.txt          - extracted plain text of the arXiv full text (HTML or PDF)
  html_<id>.html        - raw arXiv/ar5iv full-text HTML where available
  oa_<name>.json        - OpenAlex records (DOI/bibcode-ish metadata, citations)
  summary.json          - fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/dysonlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-dysonlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 4, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:34s} {len(data):9d}B  {url}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


ARXIV_API = "http://export.arxiv.org/api/query?"

# ---------------------------------------------------------------- 1. known IDs
IDS = [
    # --- G-HAT / Ghat series (Penn State) ---
    "1408.1133",   # Ghat I  : Background and Justification (Wright+ 2014 ApJ 792,26)
    "1408.1134",   # Ghat II : Framework, Strategy, and First Result (Wright+ 2014 ApJ 792,27)
    "1504.03418",  # Ghat III: Reddest Extended Sources in WISE (Griffith+ 2015 ApJS 217,25)
    "1510.04606",  # Ghat IV? : Galactic-scale waste heat in WISE resolved sources (verify)
    "1607.07500",  # possible Ghat V / follow-up (verify)
    # --- Carrigan ---
    "0811.2376",   # IRAS-based whole-sky upper limit on Dyson Spheres (ApJ 698,2075)
    "1001.5455",   # Starry Messages / interstellar archaeology (verify)
    "1201.2680",   # possible Carrigan Fermi-paradox paper (verify)
    # --- re-analyses / responses ---
    "1508.02624",  # Garrett 2015 mid-IR radio correlation applied to Ghat sample
    "1508.02406",  # Zackrisson+ 2015 Tully-Fisher probe of Dysonian astroengineering
    "1604.07844",  # Lacki 2016 "Type III Societies (Apparently) Do Not Exist" (verify)
    "2601.07297",  # WISE/CatWISE constraints on Dysonian waste heat in nearby galaxies
    "2201.11123",  # Suazo+ Project Hephaistos I
    "2405.02927",  # Suazo+ Project Hephaistos II
    "1804.08351",  # Zackrisson+ SETI with Gaia
]
get(ARXIV_API + urllib.parse.urlencode({"id_list": ",".join(IDS),
                                        "max_results": len(IDS)}),
    OUT / "arxiv_ids.atom")
time.sleep(3)

# ------------------------------------------------- 2. keyword searches for IDs
QUERIES = {
    "ghat_series": 'all:"large energy supplies"',
    "ghat_waste": 'abs:"waste heat" AND (abs:Kardashev OR abs:"extraterrestrial")',
    "kardashev_III": 'all:"Kardashev Type III" OR all:"Type III civilizations"',
    "carrigan": 'au:Carrigan_R AND (all:Dyson OR all:SETI OR all:archaeology)',
    "dyson_sphere_search": 'ti:"Dyson sphere" OR ti:"Dyson spheres"',
    "iras_dyson": 'all:IRAS AND all:Dyson',
    "wise_technosig_gal": 'abs:WISE AND (abs:technosignature OR abs:"extraterrestrial")',
    "interstellar_archaeology": 'all:"interstellar archaeology" OR all:"interstellar archeology"',
    "ghat_milkyway": 'abs:"Kardashev Type II" AND abs:"Milky Way"',
    "griffith_wise": 'au:Griffith_R AND abs:WISE',
}
for name, q in QUERIES.items():
    get(ARXIV_API + urllib.parse.urlencode(
        {"search_query": q, "max_results": 60,
         "sortBy": "submittedDate", "sortOrder": "descending"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.5)

# ------------------------------------------------------------ 3. full texts
# arXiv HTML (post-2023 papers) is best; for older ones use the PDF and pdftotext.
FULLTEXT = ["1408.1133", "1408.1134", "1504.03418", "1510.04606",
            "0811.2376", "1001.5455", "1508.02624", "1604.07844", "2601.07297"]

for aid in FULLTEXT:
    pdf = OUT / f"pdf_{aid}.pdf"
    if get(f"https://arxiv.org/pdf/{aid}", pdf):
        txt = OUT / f"txt_{aid}.txt"
        try:
            subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)],
                           check=True, timeout=300)
            print(f"     -> text {txt.name} {txt.stat().st_size}B", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"     pdftotext failed for {aid}: {e}", flush=True)
        pdf.unlink(missing_ok=True)   # keep the repo small; text is what we need
    time.sleep(3)
    # ar5iv HTML as a second rendering (tables/equations often survive better)
    get(f"https://ar5iv.labs.arxiv.org/html/{aid}", OUT / f"ar5iv_{aid}.html")
    time.sleep(3)

# ------------------------------------------------------- 4. OpenAlex metadata
OA = "https://api.openalex.org/works/doi:"
DOIS = {
    "ghat1": "10.1088/0004-637X/792/1/26",
    "ghat2": "10.1088/0004-637X/792/1/27",
    "ghat3": "10.1088/0067-0049/217/2/25",
    "carrigan2009": "10.1088/0004-637X/698/2/2075",
    "garrett2015": "10.1051/0004-6361/201526687",
}
for name, doi in DOIS.items():
    get(OA + urllib.parse.urlencode({"": doi})[1:] + "?mailto=trimcrae@gmail.com",
        OUT / f"oa_{name}.json")
    time.sleep(2)

# OpenAlex title search to find Ghat IV/V and any Carrigan items we missed
for name, title in {
    "oa_ghat_search": "infrared search for extraterrestrial civilizations with large energy supplies",
    "oa_carrigan_search": "Dyson sphere IRAS upper limit",
    "oa_starry": "Starry Messages interstellar archaeology",
}.items():
    get("https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": title, "per-page": 50, "mailto": "trimcrae@gmail.com"}),
        OUT / f"{name}.json")
    time.sleep(2)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n=== {ok}/{len(STATUS)} fetches OK ===", flush=True)

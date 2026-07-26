#!/usr/bin/env python3
"""Project Hephaistos literature dossier: fetch primary sources on the GitHub runner.

The sandbox egress policy blocks arxiv.org / ADS / IOP / Semantic Scholar / OUP, so
this script runs on the Actions runner (CLAUDE.md acquisition pattern) and commits
verbatim metadata + full text back to the branch.

Targets: the Uppsala "Project Hephaistos" Dyson-sphere survey series (Suazo,
Zackrisson et al., 2022-2026) and every follow-up / refutation paper that
re-analyses its candidates (Ren/Garrett/Siemion radio contamination, the EVN
candidate-G imaging, the archival background-contaminant diagnostics, and the
JWST paper).

Outputs under results/hephlit/:
  arxiv_ids.atom        - arXiv API metadata (title/authors/abstract/DOI/journal-ref)
  arxiv_q_<name>.atom   - arXiv API keyword searches (to find installments we lack IDs for)
  txt_<id>.txt          - extracted plain text of the arXiv full text (PDF -> pdftotext)
  html_<id>.html        - raw arXiv full-text HTML where available
  ar5iv_<id>.html       - ar5iv rendering (tables/equations often survive better)
  oa_<name>.json        - OpenAlex records (DOI metadata, references, citation counts)
  cited_by_<name>.json  - OpenAlex "cites this work" listings (to catch refutations)
  summary.json          - fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/hephlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-hephlit/1.0 (mailto:trimcrae@gmail.com)"}
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
    "2201.11123",  # Hephaistos I   : upper limits on partial Dyson spheres (MNRAS 512, 2988)
    "2405.02927",  # Hephaistos II  : DS candidates from Gaia DR3+2MASS+WISE (MNRAS 531, 695)
    "2607.09460",  # Hephaistos IV  : JWST observations of two DS candidates
    "2405.14921",  # Ren, Garrett & Siemion: background contamination (RNAAS 8, 145)
    "2501.05152",  # high-res radio imaging of candidate G (MNRAS Letters 538, L56)
    "2607.03619",  # archival diagnostics for background contaminants (IAUS 404)
    "2204.01959",  # Zackrisson+ : Dyson spheres at white dwarfs (MNRAS 514, 227) - series context
    "2409.11447",  # Blain: "Did WISE detect Dyson Spheres/Structures around Gaia-2MASS-selected stars?"
]

get(ARXIV_API + urllib.parse.urlencode(
    {"id_list": ",".join(IDS), "max_results": len(IDS)}), OUT / "arxiv_ids.atom")
time.sleep(3.5)

# ------------------------------------------------------- 2. keyword searches
# The decisive one is all:"Project Hephaistos" -- it must return every installment,
# including whichever paper is numbered III (we do not know its ID or its title).
QUERIES = {
    "hephaistos_all": 'all:"Project Hephaistos"',
    "hephaistos_abs": 'abs:"Hephaistos"',
    "suazo_all": 'au:Suazo_M',
    "zackrisson_dyson": 'au:Zackrisson_E AND abs:"Dyson"',
    "korn_dyson": 'au:Korn_A AND abs:"Dyson"',
    "dyson_sphere_candidates": 'abs:"Dyson sphere" AND abs:"candidates"',
    "dyson_contamination": 'abs:"Dyson" AND abs:"contamination"',
    "hot_dog_dyson": 'abs:"dust-obscured" AND abs:"Dyson"',
    "mahto": 'au:Mahto_P',
    "wright_dyson_2025": 'au:Wright_J AND abs:"Dyson"',
}
for name, q in QUERIES.items():
    get(ARXIV_API + urllib.parse.urlencode(
        {"search_query": q, "max_results": 100,
         "sortBy": "submittedDate", "sortOrder": "descending"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.5)

# ------------------------------------------------------------ 3. full texts
FULLTEXT = ["2201.11123", "2405.02927", "2607.09460",
            "2405.14921", "2501.05152", "2607.03619", "2409.11447"]

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
    # native arXiv HTML (2023+) keeps table structure best of all
    get(f"https://arxiv.org/html/{aid}v1", OUT / f"html_{aid}.html")
    time.sleep(3)
    get(f"https://ar5iv.labs.arxiv.org/html/{aid}", OUT / f"ar5iv_{aid}.html")
    time.sleep(3)

# ------------------------------------------------- 4. OpenAlex metadata + cites
OA = "https://api.openalex.org/works/doi:"
DOIS = {
    "heph1": "10.1093/mnras/stac280",
    "heph2": "10.1093/mnras/stae1186",
    "ren2024": "10.3847/2515-5172/ad5017",
    "wd_dyson": "10.1093/mnras/stac1284",
}
for name, doi in DOIS.items():
    get(OA + doi + "?mailto=trimcrae@gmail.com", OUT / f"oa_{name}.json")
    time.sleep(2)

# Everything that cites Hephaistos I and II -- this is how we catch any refutation
# or later installment we do not already know about. Paginate to be safe.
CITED = {
    "heph1": "10.1093/mnras/stac280",
    "heph2": "10.1093/mnras/stae1186",
}
for name, doi in CITED.items():
    for page in (1, 2, 3):
        get("https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": f"cites:doi:{doi}", "per-page": 200, "page": page,
             "mailto": "trimcrae@gmail.com"}),
            OUT / f"cited_by_{name}_p{page}.json")
        time.sleep(2)

# OpenAlex title/keyword search: catches non-arXiv items and the III installment
for name, title in {
    "oa_hephaistos": "Project Hephaistos Dyson sphere",
    "oa_dyson_candidates": "Dyson sphere candidates infrared excess M dwarfs",
    "oa_hephaistos_iii": "Project Hephaistos III",
}.items():
    get("https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": title, "per-page": 50, "mailto": "trimcrae@gmail.com"}),
        OUT / f"{name}.json")
    time.sleep(2)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n=== {ok}/{len(STATUS)} fetches OK ===", flush=True)

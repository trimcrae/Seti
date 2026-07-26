#!/usr/bin/env python3
"""CHEMLIT pass 2: close the prior-art gaps left by pass 1.

Pass 1 established that the Whitmire & Wright (1980) citation tree is dominated
by reviews and essays, but two things still need primary-source resolution:

  1. The full OpenAlex cited-by list for W2075782720 (the pass-1 `cites:` filter
     query failed), so "nobody ever executed it" rests on the complete tree
     rather than the Semantic Scholar subset.
  2. AJ 144, 181 (2012), doi:10.1088/0004-6256/144/6/181 -- "Searching for
     extraterrestrial intelligence signals in astronomical spectra, including
     existing data" -- the one entry in the tree whose title suggests an
     actually-executed archival spectral search. Its abstract decides whether
     the chemical-technosignature verdict is PROPOSED-BUT-NEVER-SEARCHED.

Also pulls OpenAlex recall for the search phrasings that arXiv misses (much of
this literature is journal-only: Icarus, IJAsB, Acta Astronautica, JBIS).
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/chemlit2")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Seti-chemlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:46s} {len(data):9d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {dest.name} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {dest.name}  {url}", flush=True)
    return False


OA = "https://api.openalex.org/works"
MAIL = "trimcrae@gmail.com"

# --- 1. Complete Whitmire & Wright cited-by tree, cursor-paged ---------------
cursor = "*"
page = 0
while cursor and page < 12:
    url = OA + "?" + urllib.parse.urlencode({
        "filter": "cites:W2075782720", "per-page": "200",
        "cursor": cursor, "mailto": MAIL})
    dest = OUT / f"oa_ww1980_citedby_p{page}.json"
    if not get(url, dest):
        break
    try:
        d = json.loads(dest.read_text())
    except Exception:
        break
    cursor = (d.get("meta") or {}).get("next_cursor")
    n = len(d.get("results", []))
    print(f"     page {page}: {n} works; next_cursor={'yes' if cursor else 'no'}", flush=True)
    page += 1
    if n == 0:
        break
    time.sleep(1.5)

# --- 2. The one candidate executed archival spectral search ------------------
for name, doi in {
    "aj2012_spectra_seti": "10.1088/0004-6256/144/6/181",
    "ww1980": "10.1016/0019-1035(80)90253-5",
    "stevens2016": "10.1017/S1473550415000397",
    "carrigan2012": "10.1016/J.ACTAASTRO.2011.12.002",
}.items():
    get(OA + "/doi:" + doi + "?mailto=" + MAIL, OUT / f"oa_doi_{name}.json")
    time.sleep(1.2)
    get("https://api.crossref.org/works/" + urllib.parse.quote(doi) +
        "?mailto=" + MAIL, OUT / f"crossref_doi_{name}.json")
    time.sleep(1.2)

# --- 3. OpenAlex recall for journal-only phrasings arXiv misses --------------
SEARCHES = {
    "tc_main_sequence": "technetium main sequence star detection",
    "tc_upper_limit": "technetium upper limit stellar abundance dwarf",
    "pm_star_survey": "promethium stellar spectrum identification survey",
    "fission_product_star": "fission products stellar photosphere abundance pattern",
    "nuclear_waste_star": "nuclear waste disposal star civilization spectrum",
    "artificial_abundance": "artificial abundance anomaly star intelligent civilization",
    "single_element_outlier": "single element abundance outlier survey stars anomalous",
    "sparse_chemical_anomaly": "star normal abundances except one element enhanced",
    "techno_spectroscopic_survey": "technosignature search spectroscopic survey abundances stars",
    "engulfment_max_mass": "planet engulfment maximum rocky mass accreted convective envelope",
    "cool_dwarf_peculiar": "chemically peculiar cool dwarf single element anomaly",
}
for name, q in SEARCHES.items():
    get(OA + "?" + urllib.parse.urlencode(
        {"search": q, "per-page": "50", "mailto": MAIL}),
        OUT / f"oa_s_{name}.json")
    time.sleep(1.5)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
print(f"\n{sum(1 for s in STATUS if s.get('ok'))}/{len(STATUS)} fetches succeeded", flush=True)

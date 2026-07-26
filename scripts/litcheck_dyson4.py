#!/usr/bin/env python3
"""Dyson/IR-excess literature survey, pass 4: targeted gap-filling.

Pass 3 used `filter=title_and_abstract.search:A AND title_and_abstract.search:B`,
which OpenAlex does not parse as a conjunction -- every two-term query returned 0.
This pass (a) redoes those with the `search=` parameter, which OpenAlex does treat
as a multi-term relevance search, (b) resolves by DOI and by title the specific
IR-excess/debris-disk catalogue papers that define the "normal excess" baseline,
and (c) pulls arXiv abstracts for the IDs discovered in pass 3.

Outputs under results/litcheck_dyson/gap4/.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/litcheck_dyson/gap4")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-litcheck-dyson4/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def fetch(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:40s} {len(data):8d}B", flush=True)
            try:
                return json.loads(data.decode("utf-8", errors="replace"))
            except Exception:
                return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return None


OA = "https://api.openalex.org/works"
SELECT = ("id,doi,title,publication_year,authorships,primary_location,"
          "cited_by_count,type,ids,abstract_inverted_index")

# --- 1. Redo the two-term queries with the `search` parameter ---------------
OA_SEARCH = {
    "technosignature_ir": "technosignature infrared excess waste heat",
    "waste_heat_civ": "waste heat extraterrestrial civilization infrared search",
    "irexcess_wise": "infrared excess WISE main-sequence stars survey",
    "debrisdisk_wise": "debris disk candidates WISE photometry excess",
    "irexcess_census": "census infrared excess stars catalog WISE",
    "allwise_excess": "AllWISE infrared excess catalog stars",
    "extraterrestrial_ir": "search extraterrestrial civilizations infrared waste heat",
    "dyson_wise_gaia": "Dyson sphere candidates Gaia WISE infrared excess search",
    "partial_dyson": "partial Dyson sphere covering fraction search stars",
    "ml_dyson": "machine learning Dyson sphere candidate prioritization",
    "anomaly_wise": "anomaly detection infrared survey WISE unusual sources",
    "mdwarf_wise_excess": "M dwarf infrared excess WISE circumstellar dust",
    "solar_analog_excess": "solar analog solar twin infrared excess debris disk",
    "kdwarf_excess": "K dwarf infrared excess debris disk WISE",
    "unwise_coadd": "unWISE coadd NEOWISE deep infrared catalog",
    "catwise_catalog": "CatWISE2020 catalog W1 W2 sources",
}
for slug, q in OA_SEARCH.items():
    cursor, page = "*", 0
    while cursor and page < 4:
        d = fetch(OA + "?" + urllib.parse.urlencode(
            {"search": q, "per-page": "200", "cursor": cursor,
             "select": SELECT, "mailto": "trimcrae@gmail.com"}),
            OUT / f"oas_{slug}_p{page}.json")
        if not isinstance(d, dict):
            break
        cursor = (d.get("meta") or {}).get("next_cursor")
        n = len(d.get("results") or [])
        print(f"     {slug} p{page}: {n} (total {(d.get('meta') or {}).get('count')})",
              flush=True)
        page += 1
        if n == 0:
            break
        time.sleep(1.2)

# --- 2. Resolve the baseline IR-excess / debris-disk catalogues by DOI ------
DOIS = {
    "cotten_song2016": "10.3847/0067-0049/225/1/15",
    "nguyen2018_ml_wise": "10.1016/j.ascom.2018.02.004",
    "ml_dyson_prioritize2026": "10.1016/j.eswa.2026.131232",
    "liu2014_akari": "10.1088/0004-6256/148/1/3",
    "zuckerman2022_wd": "10.1093/mnras/stac1113",
    "hsiao2021_bh": "10.1093/mnras/stab1832",
    "garrett2015_ghat_radio": "10.1051/0004-6361/201526687",
    "osmanov2015_pulsar": "10.1017/s1473550415000257",
    "osmanov2017_rings": "10.1017/s1473550417000155",
    "osmanov2018_beyondIR": "10.1017/s1473550418000174",
    "huang_tao_zhang2026": "10.3847/1538-3881/ae31e9",
    "ren2024_rnaas": "10.3847/2515-5172/ad5017",
    "ren2025_mnrasl": "10.1093/mnrasl/slaf006",
    "spitzer_legacy_chen2020": "10.1038/s41550-020-1067-6",
    "carrigan2009": "10.1088/0004-637x/698/2/2075",
    "wright2020_dysonspheres": "10.2298/saj2000001w",
    "lacki2019_sunscreen": "10.1088/1538-3873/aaf3df",
    "baghram2025_pbh": "10.3847/1538-4357/ad9b10",
}
for name, doi in DOIS.items():
    fetch(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?"
          + urllib.parse.urlencode({"mailto": "trimcrae@gmail.com"}),
          OUT / f"oa_doi_{name}.json")
    time.sleep(1.0)
    fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi),
          OUT / f"cr_doi_{name}.json")
    time.sleep(1.0)

# --- 3. Crossref bibliographic lookups for catalogues without a known DOI ---
CR_TITLES = {
    "patel2014": "A sensitive identification of warm debris disks in the solar "
                 "neighborhood through precise calibration of saturated WISE photometry",
    "patel2017": "A comprehensive statistical assessment of star-planet interaction "
                 "debris disks WISE excesses main sequence",
    "kennedy2013_exozodi": "The bright end of the exo-Zodi luminosity function",
    "kennedy2012_wise": "Warm exozodi from cool exo-Kuiper belts WISE 12 micron excess",
    "silverberg2016_dd": "Disk Detective discovery of a circumstellar disk citizen science",
    "silverberg2018_peterpan": "Peter Pan disks long-lived accretion disks around young M stars",
    "kuchner2016_dd": "Disk Detective discovery of new circumstellar disk candidates WISE",
    "marton2016_wise": "An all-sky support vector machine selection of WISE YSO candidates",
    "marton2019_gaia": "Identification of young stellar object candidates in the Gaia DR2 "
                       "x AllWISE catalogue with machine learning methods",
    "mcdonald2012": "Fundamental parameters and infrared excesses of Hipparcos stars",
    "mcdonald2017": "Fundamental parameters and infrared excesses of Tycho-Gaia stars",
    "theissen2014": "Warm dust around cool stars WISE M dwarfs infrared excess",
    "dennihy_wd": "Five new post main sequence debris disks with gaseous emission WISE",
    "wu2013_wise_excess": "Infrared excess of main sequence stars WISE LAMOST",
    "avenhaus2012": "WISE detection of debris disks around nearby stars",
    "morales2012": "Common warm dust temperatures around main-sequence stars WISE Spitzer",
    "gaia_wise_irexcess2024": "Gaia DR3 WISE cross-match infrared excess catalogue "
                              "main-sequence stars",
}
for name, q in CR_TITLES.items():
    fetch("https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": "20",
         "select": "DOI,title,author,issued,container-title,abstract,"
                   "is-referenced-by-count",
         "mailto": "trimcrae@gmail.com"}),
        OUT / f"crq_{name}.json")
    time.sleep(1.2)

# --- 4. arXiv abstracts for IDs discovered in pass 3 ------------------------
ARXIV = "http://export.arxiv.org/api/query?"
IDS = [
    "2601.07297",  # Huang, Tao & Zhang 2026 - WISE/CatWISE Dysonian waste heat, nearby galaxies
    "2409.11447",  # Blain 2024 - Did WISE detect Dyson Spheres/Structures?
    "2405.14921",  # Ren, Garrett & Siemion 2024 - background contamination (RNAAS)
    "2501.05152",  # Ren+ 2025 - high-resolution radio imaging of candidate G
    "2607.03619",  # Ren+ 2026 - archival diagnostics
    "1503.04376",  # Semiz & Ogur 2015 - Dyson spheres around white dwarfs
    "2412.02671",  # Baghram 2024 - Dyson-sphere-like structures around PBHs
    "2512.07924",  # Baghram 2025 - microlensing signatures
    "2604.21886",  # Curtis+ 2026 - Dyson Minds workshop
    "2109.11443",  # Smith 2021 - viability of a Dyson swarm
    "2209.05348",  # Page 2022 - perfectly reflecting Dyson sphere
    "2303.08013",  # Loeb 2023 - interstellar objects from broken Dyson spheres
    "1503.01509",  # Lacki 2015 - SETI at Planck energy
    "2607.09460",  # Zackrisson+ 2026 - Hephaistos IV (seed only)
    "1408.1133", "1408.1134", "1504.03418", "1510.04336",  # G-hat I-IV (seed only)
]
fetch(ARXIV + urllib.parse.urlencode(
    {"id_list": ",".join(IDS), "max_results": str(len(IDS))}),
    OUT / "arxiv_ids_gap.atom")
time.sleep(3.2)

# --- 5. arXiv searches for the baseline catalogues and for ML/anomaly work --
AQ = {
    "patel_wise": 'ti:"WISE" AND abs:"debris disks" AND abs:"solar neighborhood"',
    "cotten_song": 'ti:"infrared excess" AND abs:census',
    "kennedy_wyatt": 'au:Kennedy AND au:Wyatt AND abs:excess',
    "disk_detective": 'all:"Disk Detective"',
    "marton_wise": 'au:Marton AND abs:WISE',
    "mcdonald_excess": 'au:McDonald AND abs:"infrared excess"',
    "theissen_west": 'au:Theissen AND au:West',
    "osmanov_all": 'au:Osmanov AND (abs:Dyson OR abs:SETI OR abs:extraterrestrial)',
    "ml_dyson": 'abs:"machine learning" AND abs:Dyson',
    "anomaly_wise": 'abs:"anomaly detection" AND (abs:WISE OR abs:infrared)',
    "wise_excess_catalog": 'abs:WISE AND abs:"infrared excess" AND abs:catalog',
    "zhang_techno": 'au:"Tong-Jie Zhang" AND (abs:SETI OR abs:technosignature)',
    "waste_heat_galaxies": 'abs:"waste heat" AND abs:galaxies',
}
for slug, q in AQ.items():
    fetch(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "80", "sortBy": "relevance"}),
        OUT / f"arxiv_{slug}.atom")
    time.sleep(3.2)

(OUT.parent / "summary4.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")

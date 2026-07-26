#!/usr/bin/env python3
"""Dyson/IR-excess literature survey, pass 5: abstracts for the resolved DOIs.

Pass 4 resolved the baseline IR-excess / debris-disk catalogue papers to DOIs via
Crossref but Crossref rarely carries abstracts. OpenAlex does (as an inverted
index), and a single GET per DOI is cheap, so resolve every DOI identified so far
and de-invert the abstract. Abstracts carry most of the sample sizes, colour cuts
and dust-temperature ranges this survey needs.

Outputs under results/litcheck_dyson/abs5/.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/litcheck_dyson/abs5")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-litcheck-dyson5/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def fetch(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0):
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


# Every DOI resolved so far that matters to the survey.
DOIS = {
    # --- the non-SETI WISE IR-excess / debris-disk baseline ---
    "patel2014": "10.1088/0067-0049/212/1/10",
    "patel2014_erratum1": "10.1088/0067-0049/214/1/14",
    "patel2014_erratum2": "10.1088/0067-0049/220/1/21",
    "cotten_song2016": "10.3847/0067-0049/225/1/15",
    "kennedy_wyatt2013": "10.1093/mnras/stt900",
    "kennedy_piette2015": "10.1093/mnras/stv453",
    "huang_kennedy2025_exozodi10pc": "10.1051/0004-6361/202554746",
    "theissen_west2014": "10.1088/0004-637x/794/2/146",
    "morales2011": "10.1088/2041-8205/730/2/l29",
    "mizuki2024_warmdebris": "10.3847/1538-3881/ad3df8",
    "ishihara2017_akari": "10.1051/0004-6361/201526215",
    "liu2014_akari": "10.1088/0004-6256/148/1/3",
    "mcdonald2012_hipparcos": "10.1111/j.1365-2966.2012.21873.x",
    "mcdonald2017_tychogaia": "10.1093/mnras/stx1433",
    "marton2016_svm_wise": "10.1093/mnras/stw398",
    "marton2019_gaia_allwise": "10.1093/mnras/stz1301",
    "kuchner2016_diskdetective": "10.3847/0004-637x/830/2/84",
    "silverberg2016_mdwarf": "10.3847/2041-8205/830/2/l28",
    "silverberg2018_followup": "10.3847/1538-4357/aae3e3",
    "silverberg2020_peterpan": "10.3847/1538-4357/ab68e6",
    "schutte2020_bd": "10.3847/1538-3881/abaccd",
    "higashio2022_vr": "10.3847/1538-4357/ac649f",
    "nguyen2018_ml_wise": "10.1016/j.ascom.2018.02.004",
    "dennihy2020_wd": "10.3847/1538-4357/abc339",
    "madurga2024_wd_catwise": "10.1051/0004-6361/202347368",
    "marocco2021_catwise2020": "10.3847/1538-4365/abd805",
    "chen2020_spitzer_legacy": "10.1038/s41550-020-1067-6",
    # --- the searches themselves ---
    "contardo_hogg2024": "10.3847/1538-3881/ad6b90",
    "mignone2026_ml_dyson": "10.1016/j.eswa.2026.131232",
    "huang_tao_zhang2026": "10.3847/1538-3881/ae31e9",
    "garrett2015_ghat_midir_radio": "10.1051/0004-6361/201526687",
    "griffith2015_ghat3": "10.1088/0067-0049/217/2/25",
    "zackrisson2015_tullyfisher": "10.1088/0004-637x/810/1/23",
    "zackrisson2018_gaia": "10.3847/1538-4357/aac386",
    "lintott2016_construction_time": "10.5281/zenodo.44755",
    "cirkovic2016_stellified": "10.1017/s1473550415000257",
    "wright2016_dysonian_shortcut": "10.2298/saj2000001w",
    # --- exotic hosts (brief) ---
    "osmanov2016_rings": "10.1017/s1473550416000045",
    "osmanov2019_variability": "10.1017/s1473550419000260",
    "osmanov2022_optical": "10.2298/saj210922003o",
    "amiri2026_hr": "10.3390/universe12040113",
    "mcinnes2025_stable": "10.1093/mnras/staf028",
    "lacki2025_groundtodust": "10.3847/1538-4357/adccc5",
    "huston_wright2022": "10.3847/1538-4357/ac3421",
    "wright2023_thermo": "10.3847/1538-4357/acf44f",
    "curtis2026_dysonminds": "10.1088/1538-3873/ae5a02",
    "ivanov2020_classification": "10.1051/0004-6361/202037597",
}
for name, doi in DOIS.items():
    fetch(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?"
          + urllib.parse.urlencode({"mailto": "trimcrae@gmail.com"}),
          OUT / f"oa_{name}.json")
    time.sleep(0.8)

# Crossref abstracts as a second source (some publishers deposit them).
for name, doi in DOIS.items():
    fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi),
          OUT / f"cr_{name}.json")
    time.sleep(0.8)

# A few more arXiv abstracts whose IDs are now known or worth probing.
ARXIV = "http://export.arxiv.org/api/query?"
fetch(ARXIV + urllib.parse.urlencode({
    "search_query": 'ti:"data-driven search" AND abs:"mid-infrared"',
    "max_results": "20", "sortBy": "relevance"}), OUT / "arxiv_contardo.atom")
time.sleep(3.2)
for slug, q in {
    "exozodi_10pc": 'abs:"exozodi" AND abs:WISE',
    "warm_debris_freq": 'abs:"warm debris" AND abs:WISE',
    "wise_saturated": 'abs:WISE AND abs:saturated AND abs:photometry',
    "lamost_debris": 'abs:LAMOST AND (abs:"debris disk" OR abs:"debris disc")',
    "lamost_gaia_fgk_disk": 'abs:LAMOST AND abs:Gaia AND abs:disk AND abs:infrared',
    "dyson_solar_analog": 'abs:Dyson AND (abs:"solar analog" OR abs:"solar-type")',
    "mdwarf_dyson": 'abs:Dyson AND abs:"M dwarf"',
}.items():
    fetch(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "60", "sortBy": "relevance"}),
        OUT / f"arxiv_{slug}.atom")
    time.sleep(3.2)

(OUT.parent / "summary5.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")

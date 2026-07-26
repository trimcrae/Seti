#!/usr/bin/env python3
"""Dyson-sphere / waste-heat IR-excess literature survey: fetch on the runner.

The sandbox egress policy blocks arxiv.org / Semantic Scholar / OpenAlex, so this
script runs on the GitHub Actions runner (per CLAUDE.md acquisition pattern) and
commits verbatim abstracts + citation lists back to the branch.

Scope: the "long tail" of WISE/AllWISE/unWISE/CatWISE/NEOWISE-based Dyson sphere,
waste-heat and IR-excess technosignature searches 2013-2026, EXCLUDING the Project
Hephaistos (Suazo/Zackrisson) and G-hat (Wright/Griffith) series -- though those
two ARE used as citation-mining seeds, since their citing papers are exactly the
long tail we want.

Outputs under results/litcheck_dyson/:
  arxiv_q_<name>.atom   - arXiv API search results per query
  arxiv_ids.atom        - arXiv metadata for specific known IDs
  s2_<name>.json        - Semantic Scholar paper record / title search
  s2_<name>_cit.json    - Semantic Scholar citation lists (long-tail discovery)
  oa_<name>.json        - OpenAlex fallback search / citation lists
  summary.json          - fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/litcheck_dyson")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-litcheck-dyson/1.0 (mailto:trimcrae@gmail.com)"}
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

# --- 1. arXiv searches -------------------------------------------------------
QUERIES = {
    # core Dysonian
    "dyson_sphere": 'abs:"Dyson sphere"',
    "dyson_spheres": 'abs:"Dyson spheres"',
    "dyson_swarm": 'abs:"Dyson swarm" OR abs:"Dyson swarms"',
    "dysonian": 'all:"Dysonian"',
    "megastructure": 'abs:"megastructure" OR abs:"megastructures"',
    "kardashev_ir": 'abs:Kardashev AND (abs:infrared OR abs:"waste heat")',
    "waste_heat": 'abs:"waste heat" AND (abs:SETI OR abs:technosignature OR abs:civilization OR abs:extraterrestrial)',
    # IR-excess technosignature framing
    "techno_ir": 'abs:technosignature AND abs:infrared',
    "irexcess_seti": 'abs:"infrared excess" AND (abs:SETI OR abs:technosignature OR abs:extraterrestrial OR abs:"alien")',
    "midir_techno": 'abs:"mid-infrared" AND (abs:technosignature OR abs:SETI)',
    # WISE-based catalogs that define "normal" excess
    "wise_debris": 'abs:WISE AND abs:"debris disk"',
    "wise_debris2": 'abs:WISE AND abs:"debris disks"',
    "wise_irexcess": 'abs:WISE AND abs:"infrared excess"',
    "allwise_excess": 'abs:AllWISE AND abs:excess',
    "unwise": 'abs:unWISE OR abs:CatWISE OR abs:NEOWISE',
    "unwise_excess": '(abs:unWISE OR abs:CatWISE) AND (abs:excess OR abs:infrared)',
    "disk_detective": 'all:"Disk Detective"',
    "census_irexcess": 'abs:"infrared excess" AND abs:census',
    "gaia_wise_excess": 'abs:Gaia AND abs:WISE AND abs:excess',
    "sed_excess_gaia": 'abs:Gaia AND abs:"spectral energy distribution" AND abs:excess',
    # stellar-population-specific
    "mdwarf_excess": 'abs:"M dwarf" AND abs:"infrared excess"',
    "mdwarfs_excess": 'abs:"M dwarfs" AND abs:"infrared excess"',
    "solar_analog_excess": 'abs:"solar analog" AND abs:"infrared excess"',
    "kdwarf_excess": 'abs:"K dwarf" AND abs:"infrared excess"',
    "solar_twin_dyson": '(abs:"solar analogs" OR abs:"solar twins") AND (abs:technosignature OR abs:"Dyson")',
    # ML / anomaly detection
    "anomaly_techno": 'abs:"anomaly detection" AND (abs:technosignature OR abs:SETI)',
    "ml_techno": 'abs:"machine learning" AND abs:technosignature',
    "outlier_survey": 'abs:outlier AND abs:detection AND abs:WISE',
    "ml_irexcess": 'abs:"machine learning" AND abs:"infrared excess"',
    # reviews / theory
    "seti_review": 'abs:SETI AND abs:review',
    "techno_review": 'abs:technosignature AND (abs:review OR abs:overview OR abs:"state of the art")',
    "osmanov_dyson": 'all:Osmanov AND all:Dyson',
    "dyson_exotic": 'abs:"Dyson" AND (abs:pulsar OR abs:"white dwarf" OR abs:"neutron star" OR abs:"red dwarf")',
    # attenuation / non-excess variants (for the excess-vs-dimming contrast)
    "dyson_transit": 'abs:"Dyson" AND (abs:transit OR abs:dimming OR abs:occultation)',
    "boyajian_like": 'abs:"KIC 8462852" OR abs:"Boyajian"',
    # Chinese-group specific
    "lamost_excess": 'abs:LAMOST AND abs:"infrared excess"',
    "lamost_wise": 'abs:LAMOST AND abs:WISE',
    "china_techno": 'abs:technosignature AND (abs:FAST OR abs:LAMOST OR abs:China)',
    # generic wide nets
    "extraterrestrial_survey": 'abs:extraterrestrial AND abs:survey AND abs:infrared',
    "civilization_search": 'abs:"extraterrestrial civilizations" AND abs:search',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "100", "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)  # arXiv API politeness

# also a date-sorted sweep of the two core phrases, to catch very recent work
for name, q in {
    "dyson_sphere_recent": 'abs:"Dyson sphere"',
    "techno_ir_recent": 'abs:technosignature AND abs:infrared',
    "dysonian_recent": 'all:"Dysonian"',
}.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "100",
         "sortBy": "submittedDate", "sortOrder": "descending"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)

# --- 2. Specific known arXiv IDs (verbatim abstracts) ------------------------
IDS = [
    "1409.5104",   # Patel, Metchev & Heinze 2014 - sensitive identification of WISE debris disks (verify)
    "1606.01755",  # Cotten & Song 2016 - comprehensive census of nearby IR excess stars (verify)
    "1211.7095",   # Kennedy & Wyatt 2013 - bright end of exozodi luminosity function (verify)
    "1510.05610",  # Silverberg+ 2016 - Disk Detective (verify)
    "1809.03621",  # Silverberg+ 2018 - Disk Detective M dwarf disks (verify)
    "2012.07830",  # Silverberg+ 2020 - Disk Detective (verify)
    "1705.08163",  # McDonald+ 2017 - astrophysical params & IR excesses of Gaia sources (verify)
    "1602.05229",  # Marton+ 2016 - WISE YSO candidate catalog (verify)
    "1902.09153",  # Marton+ 2019 - Gaia DR2 + WISE ML classification (verify)
    "1403.7141",   # Theissen & West 2014 - dusty M dwarfs WISE (verify)
    "2201.11123",  # Suazo+ 2022 Hephaistos I (seed only)
    "2405.02927",  # Suazo+ 2024 Hephaistos II (seed only)
    "1408.1133",   # Wright+ 2014 G-hat I (seed only)
    "1504.03418",  # Griffith+ 2015 G-hat III WISE catalog (seed only)
    "1804.08351",  # Zackrisson+ 2018 SETI with Gaia
    "2107.07512",  # Wright 2021 SETI review
    "2501.05152",  # 2025 radio imaging of Hephaistos candidate G
    "2607.03619",  # 2026 archival diagnostics of Hephaistos contaminants
    "2602.23270",  # 2026 Dyson spheres on HR diagram
    "2605.21093",  # 2026 Search for Technosignatures review
]
get(ARXIV + urllib.parse.urlencode({"id_list": ",".join(IDS), "max_results": str(len(IDS))}),
    OUT / "arxiv_ids.atom")
time.sleep(3.2)

# --- 3. Semantic Scholar title resolution ------------------------------------
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2FIELDS = "title,year,abstract,externalIds,venue,authors,citationCount"

TITLE_SEARCHES = {
    "patel2014": "A sensitive identification of warm debris disks in the solar neighborhood through precise calibration of saturated WISE photometry",
    "patel2017": "A comprehensive study of debris disks WISE excess main sequence stars Patel Metchev",
    "cotten2016": "A comprehensive census of nearby infrared excess stars",
    "kennedy2013": "The bright end of the exo-Zodi luminosity function",
    "kennedy2012": "Do two-temperature debris discs have multiple belts",
    "silverberg2016": "Disk Detective discovery of a circumstellar disk",
    "silverberg2018": "Peter Pan disks long-lived accretion disks around young M stars",
    "disk_detective_2020": "Disk Detective citizen scientist classifications WISE disk candidates",
    "mcdonald2017": "Astrophysical parameters and infrared excesses of Gaia stars",
    "theissen2014": "Dusty M dwarfs WISE infrared excess",
    "wu_debris": "WISE debris disk candidates LAMOST",
    "carrigan2009": "IRAS-based whole-sky upper limit on Dyson spheres",
    "dysonian_review": "Dysonian approach to SETI a fruitful middle ground",
    "osmanov_pulsar": "Are the dyson spheres around pulsars detectable",
    "osmanov_wd": "Dyson spheres around white dwarfs",
    "osmanov_neutron": "On the search for Dyson spheres around neutron stars",
    "zhang_dyson": "Dyson sphere search Gaia WISE candidates China",
    "tang_dyson": "search for Dyson spheres infrared excess catalog",
    "ml_dyson": "machine learning anomaly detection technosignature infrared WISE",
    "lacki_review": "Lacki SETI waste heat infrared limits galaxies",
    "kuhn_berdyugina": "Global warming as a detectable thermodynamic marker of Earth-like extrasolar civilizations",
    "villarroel": "Vanishing and appearing sources during a century of observations",
}
for name, q in TITLE_SEARCHES.items():
    get(S2 + "search?" + urllib.parse.urlencode(
        {"query": q, "fields": S2FIELDS, "limit": "8"}),
        OUT / f"s2_{name}.json", pause=6.0)
    time.sleep(3.0)

# --- 4. Semantic Scholar citation lists (the long-tail discovery engine) -----
CITE_SEEDS = {
    "hephaistos2": "arXiv:2405.02927",
    "hephaistos1": "arXiv:2201.11123",
    "ghat1": "arXiv:1408.1133",
    "ghat3": "arXiv:1504.03418",
    "zackrisson2018": "arXiv:1804.08351",
    "carrigan2009": "DOI:10.1088/0004-637X/698/2/2075",
}
for name, ident in CITE_SEEDS.items():
    get(S2 + urllib.parse.quote(ident) + f"?fields={S2FIELDS}",
        OUT / f"s2_seed_{name}.json", pause=6.0)
    time.sleep(3.0)
    get(S2 + urllib.parse.quote(ident) + f"/citations?fields={S2FIELDS}&limit=1000",
        OUT / f"s2_{name}_cit.json", pause=6.0)
    time.sleep(3.0)

# --- 5. OpenAlex fallback (generous rate limits, good citation graph) --------
OA = "https://api.openalex.org/works"
OA_SEARCHES = {
    "dyson_sphere": "Dyson sphere search infrared excess candidates",
    "waste_heat": "waste heat technosignature infrared survey civilizations",
    "wise_debris": "WISE debris disk catalog infrared excess main sequence stars",
    "irexcess_census": "census of infrared excess stars nearby",
    "dyson_ml": "machine learning search Dyson spheres Gaia WISE",
    "dysonian_review": "Dysonian SETI review megastructures",
    "partial_dyson": "partial Dyson sphere solar neighborhood search",
}
for name, q in OA_SEARCHES.items():
    get(OA + "?" + urllib.parse.urlencode(
        {"search": q, "per-page": "50", "mailto": "trimcrae@gmail.com"}),
        OUT / f"oa_{name}.json")
    time.sleep(1.5)

# OpenAlex citing-works lists for the seeds (by DOI)
OA_CITED_BY = {
    "hephaistos2": "10.1093/mnras/stae1186",
    "carrigan2009": "10.1088/0004-637X/698/2/2075",
    "ghat3": "10.1088/0067-0049/217/2/25",
}
for name, doi in OA_CITED_BY.items():
    get(OA + "?" + urllib.parse.urlencode(
        {"filter": f"cites:doi:{doi}", "per-page": "200",
         "mailto": "trimcrae@gmail.com"}),
        OUT / f"oa_citedby_{name}.json")
    time.sleep(1.5)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")

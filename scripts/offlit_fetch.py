#!/usr/bin/env python3
"""Literature survey fetcher: sources that DISAPPEAR / signatures that TURN OFF.

Prior-art reconnaissance for a proposed "IR excess present in IRAS, absent in
WISE" technosignature search. Four strands:

  (A) VASCO / vanishing stars (Villarroel, Solano et al.) and the 2026 critiques
  (B) Infrared excess that disappears - real astrophysical precedent
      (TYC 8241 2652 1, extreme debris disks, dusty white dwarfs) plus the
      IRAS/WISE catalogue-systematics literature
  (C) Technosignatures that ceased (Wow!, BLC1, intermittency, dead civs)
  (D) Carrigan IRAS Dyson-sphere search, G-HAT, Project Hephaistos - and
      whether ANY of them did an epoch-to-epoch (IRAS vs WISE) comparison

The sandbox egress policy blocks arxiv.org / ADS / Semantic Scholar / IOP
(CONNECT 403), so this runs on the GitHub Actions runner per the CLAUDE.md
acquisition pattern and commits verbatim metadata back to the branch.

Outputs under results/offlit/:
  arxiv_ids_<n>.atom    - arXiv API metadata (title/authors/abstract/DOI) by ID
  arxiv_q_<name>.atom   - arXiv API search results per query
  html_<id>.html        - arXiv full-text HTML for argument-critical papers
  s2_<name>.json        - Semantic Scholar records + citation lists
  summary.json          - fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/offlit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-offlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:46s} {len(data):8d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


ARXIV = "http://export.arxiv.org/api/query?"

# --- 1. Papers by arXiv ID (verbatim abstracts + DOI + journal ref) -----------
# Some IDs are best guesses; the section-2 searches are the authoritative
# resolver and will expose any ID that is wrong.
IDS = [
    # (A) VASCO / vanishing stars
    "1606.08992",  # Villarroel, Imaz & Bergstedt 2016 AJ 152,76 Our Sky now and then
    "1911.05068",  # Villarroel+ 2020 AJ 159,8 VASCO I (USNO vs Pan-STARRS)
    "2009.10813",  # Villarroel+ Launching the VASCO citizen science project
    "2106.11780",  # Villarroel+ 2021 SciRep nine transients 12 April 1950
    "2310.14895",  # Villarroel Astronomical Anomalies / quest for ET life
    "2507.15896",  # On the Image Profiles of Transients in the Palomar Sky Survey
    "2601.21946",  # Watters+ 2026 Critical Evaluation of POSS1-E technosignature claims
    "2602.15171",  # Villarroel+ 2026 Response to Watters+
    "2604.04810",  # Independent Recovery of Vanishing Sources on POSS-I
    "2604.00056",  # Independent Replication of Nuclear Test-Transient Correlations
    "2604.04950",  # Geomagnetic storm suppression of plate transient detections
    "2604.06234",  # Storm-Driven Suppression / dusty plasma mechanism
    "2605.01190",  # Statistically Significant Linear Alignments POSS-I
    # (B) IR excess that disappears
    "1205.1040",   # Meng+ 2012 ApJL 751,L17 Variability of IR excess of extreme debris disks
    "1611.01371",  # Ertel? / TYC 8241 2652 1 disappearing disk: no smoking gun yet (A&A 2017)
    "1903.10627",  # Su+ 2019 Extreme Debris Disk Variability
    "1802.04313",  # Hughes, Duchene & Matthews 2018 ARA&A Debris Disks
    # (C) technosignature that ceased
    "2111.06350",  # Sheikh+ 2021 NatAstron analysis of blc1 (verify)
    "2111.06351",  # Smith+ 2021 NatAstron Proxima signal of interest (verify)
    "2408.08513",  # Mendez+ 2024 Arecibo Wow! astrophysical explanation (verify)
    "2011.06090",  # Caballero 2022 approximation to source of WOW! signal (verify)
    "1809.07252",  # Wright, Kanodia & Lubar 2018 AJ 156,260 cosmic haystack (verify)
    "2301.07165",  # Grimaldi 2023 rate of technosignatures from 60 yr nondetection
    "2605.21093",  # The Search for Technosignatures: a Review of Possibilities (2026)
    # (D) Dyson spheres
    "0811.2376",   # Carrigan 2009 ApJ 698,2075 IRAS whole-sky upper limit Dyson spheres
    "1001.5455",   # Carrigan 2010 Starry Messages / interstellar archaeology (verify)
    "1504.03418",  # Griffith+ 2015 ApJS 217,25 G-HAT III reddest extended WISE sources
    "1804.08351",  # Zackrisson+ 2018 SETI with Gaia / nearly complete Dyson spheres
    "2405.02927",  # Suazo+ 2024 MNRAS 531,695 Project Hephaistos II
    "2405.14921",  # Ren, Garrett & Siemion 2024 RNAAS background contamination
    "2501.05152",  # High-res imaging of radio source assoc. w/ Hephaistos candidate G
    "2607.03619",  # Archival Diagnostics for Hephaistos background contaminants (2026)
    "1604.07844",  # Lacki 2016 Type III Societies (Apparently) Do Not Exist (verify)
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
    # ---- (A) VASCO / vanishing sources -------------------------------------
    "vasco_all": 'all:"VASCO" AND (all:vanishing OR all:"appearing sources")',
    "vanishing_appearing": 'all:"Vanishing and Appearing Sources"',
    "villarroel_all": 'au:"Villarroel, B"',
    "solano_vanishing": 'au:"Solano" AND (abs:vanishing OR abs:"POSS")',
    "poss_transients": 'abs:"POSS-I" OR abs:"Palomar Observatory Sky Survey" AND abs:transient',
    "plate_transients_glint": 'abs:glint AND (abs:plate OR abs:"photographic")',
    "missing_star": 'ti:"missing star" OR abs:"vanished star"',
    "usno_panstarrs_missing": 'abs:"USNO" AND abs:"Pan-STARRS" AND (abs:missing OR abs:vanishing)',
    # ---- (B) IR excess that disappears -------------------------------------
    "disappearing_disk": 'abs:"disappearance" AND abs:disk AND abs:dust',
    "tyc8241": 'all:"TYC 8241 2652 1"',
    "rapid_disappearance_disk": 'ti:"rapid disappearance" OR abs:"disappearing disk"',
    "extreme_debris_disks": 'abs:"extreme debris disk" OR abs:"extreme debris disks"',
    "debris_disk_variability": 'abs:"debris disk" AND abs:variability AND abs:infrared',
    "ngc2547_id8": 'all:"NGC 2547-ID8" OR all:"ID8"',
    "v488per": 'all:"V488 Per"',
    "ir_excess_variable": 'abs:"infrared excess" AND (abs:variable OR abs:variability OR abs:fading)',
    "ir_excess_disappear": 'abs:"infrared excess" AND (abs:disappear OR abs:disappeared OR abs:vanished OR abs:"turned off")',
    "dusty_wd_variability": 'abs:"white dwarf" AND abs:"infrared excess" AND abs:variab',
    "wd_disk_disappear": 'abs:"white dwarf" AND abs:disk AND (abs:disappear OR abs:brightening) AND abs:infrared',
    # ---- (B2) IRAS / WISE catalogue systematics ----------------------------
    "iras_wise_crossmatch": 'abs:IRAS AND abs:WISE AND (abs:"cross-match" OR abs:crossmatch OR abs:counterpart)',
    "iras_psc_fsc_combined": 'all:"IRAS PSC/FSC" OR abs:"Faint Source Catalogue" AND abs:IRAS',
    "iras_reliability": 'abs:IRAS AND (abs:reliability OR abs:spurious OR abs:"false detections")',
    "iras_cirrus": 'abs:IRAS AND abs:cirrus AND (abs:confusion OR abs:contamination)',
    "vega_like_spurious": 'abs:"Vega-like" OR abs:"Vega-type" AND (abs:spurious OR abs:cirrus OR abs:contamination)',
    "iras_excess_reexamined": 'abs:IRAS AND abs:excess AND (abs:Spitzer OR abs:WISE) AND abs:re-examin',
    "rhee_debris_iras_hipparcos": 'abs:"dusty debris" AND abs:IRAS AND abs:Hipparcos',
    "wise_excess_contamination": 'abs:WISE AND abs:"infrared excess" AND (abs:contamination OR abs:spurious OR abs:"background galaxies")',
    "wise_saturation": 'abs:WISE AND abs:saturation AND abs:photometry',
    "w4_calibration": 'abs:"W4" AND abs:WISE AND abs:calibration',
    "akari_iras_wise": 'abs:AKARI AND (abs:IRAS OR abs:WISE) AND abs:catalog',
    "iras_no_counterpart": 'abs:IRAS AND abs:"no counterpart"',
    "mid_ir_variability_survey": 'abs:"mid-infrared" AND abs:variability AND (abs:WISE OR abs:NEOWISE) AND abs:survey',
    "neowise_variability": 'abs:NEOWISE AND abs:variability AND abs:"light curve"',
    # ---- (C) technosignature that ceased -----------------------------------
    "wow_signal": 'all:"Wow! signal" OR all:"Wow signal"',
    "wow_repetition": 'abs:"Wow" AND abs:signal AND (abs:repetition OR abs:search OR abs:follow-up)',
    "blc1": 'all:"BLC1" OR all:"blc1" OR abs:"Breakthrough Listen signal of interest"',
    "hd164595": 'all:"HD 164595"',
    "shgb02": 'all:"SHGb02+14a"',
    "intermittent_seti": 'abs:intermittent AND (abs:SETI OR abs:technosignature OR abs:"extraterrestrial")',
    "transient_seti": 'abs:transient AND abs:SETI AND (abs:strategy OR abs:search)',
    "technosignature_longevity": 'abs:technosignature AND (abs:longevity OR abs:lifetime OR abs:"L")',
    "dead_civilizations": 'abs:"extinct" AND (abs:civilization OR abs:civilisation) AND abs:technosignature',
    "interstellar_archaeology": 'all:"interstellar archaeology" OR abs:"archaeology" AND abs:extraterrestrial',
    "civilization_collapse_signature": 'abs:"collapse" AND abs:civilization AND (abs:detect OR abs:signature)',
    "seti_nondetection_rate": 'abs:SETI AND abs:nondetection AND abs:rate',
    # ---- (C2) astrophysical disappearing-source precedent ------------------
    "vanishing_radio_sources": 'abs:radio AND (abs:vanishing OR abs:disappearing) AND abs:sources',
    "radio_transient_epoch_compare": 'abs:"radio transient" AND (abs:VLASS OR abs:FIRST OR abs:NVSS) AND abs:comparison',
    "disappearing_maser": 'abs:maser AND (abs:disappear OR abs:vanish OR abs:"turned off")',
    "oh_megamaser_variability": 'abs:"OH megamaser" AND abs:variability',
    "changing_look_agn": 'abs:"changing-look" AND abs:AGN AND abs:disappear',
    "changing_look_ir": 'abs:"changing-look" AND abs:AGN AND abs:infrared AND abs:echo',
    # ---- (D) Dyson spheres / waste heat ------------------------------------
    "dyson_sphere_search": 'abs:"Dyson sphere" OR abs:"Dyson spheres"',
    "dyson_iras": 'abs:"Dyson sphere" AND abs:IRAS',
    "dyson_wise": 'abs:"Dyson sphere" AND abs:WISE',
    "dyson_variability": 'abs:"Dyson sphere" AND (abs:variability OR abs:transient OR abs:"turned off" OR abs:disappear)',
    "waste_heat_seti": 'abs:"waste heat" AND (abs:SETI OR abs:technosignature OR abs:civilization)',
    "ghat": 'all:"G-HAT" OR abs:"Glimpsing Heat from Alien Technologies"',
    "hephaistos": 'all:"Project Hephaistos"',
    "kardashev_iii_search": 'abs:"Kardashev" AND abs:search AND abs:infrared',
    "megastructure_search": 'abs:megastructure AND (abs:search OR abs:transit OR abs:infrared)',
    # ---- (E) does anyone do multi-epoch archival technosignature work? -----
    "archival_technosignature": 'abs:archival AND abs:technosignature',
    "multi_epoch_technosignature": 'abs:"multi-epoch" AND (abs:technosignature OR abs:SETI)',
    "boyajian_star": 'all:"KIC 8462852" OR all:"Boyajian"',
    "seti_infrared_variability": 'abs:SETI AND abs:infrared AND abs:variability',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "40", "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)

# recency-sorted sweeps for the 2024-2026 window (catch very new work)
for name, q in {
    "recent_dyson": 'abs:"Dyson sphere" OR abs:"Dyson spheres"',
    "recent_vasco": 'all:"VASCO" OR abs:"vanishing sources"',
    "recent_ir_excess": 'abs:"infrared excess" AND abs:excess',
    "recent_technosig": 'abs:technosignature',
    "recent_iras": 'abs:IRAS AND abs:catalog',
}.items():
    get(ARXIV + urllib.parse.urlencode(
        {"search_query": q, "max_results": "80", "sortBy": "submittedDate",
         "sortOrder": "descending"}),
        OUT / f"arxiv_recent_{name}.atom")
    time.sleep(3.2)

# --- 3. Full-text HTML of argument-critical papers ---------------------------
for aid in ["0811.2376", "2405.02927", "1504.03418", "2601.21946", "2602.15171",
            "1911.05068", "2607.03619", "2405.14921", "1611.01371", "1903.10627"]:
    if not get(f"https://arxiv.org/html/{aid}v1", OUT / f"html_{aid}.html", tries=2):
        get(f"https://arxiv.org/html/{aid}", OUT / f"html_{aid}.html", tries=2)
    time.sleep(3)

# --- 4. Semantic Scholar: citation lists to catch follow-ups -----------------
# Key question: did ANYONE citing Carrigan 2009 / Griffith 2015 / Suazo 2024
# do an epoch-to-epoch IRAS-vs-WISE comparison?
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2FIELDS = "title,year,abstract,externalIds,venue,authors"
for name, ident in {
    "carrigan2009": "arXiv:0811.2376",
    "griffith2015": "arXiv:1504.03418",
    "suazo2024": "arXiv:2405.02927",
    "villarroel2016": "arXiv:1606.08992",
    "vasco1": "arXiv:1911.05068",
    "melis2012": "DOI:10.1038/nature11210",
    "meng2012": "arXiv:1205.1040",
    "su2019": "arXiv:1903.10627",
}.items():
    get(S2 + urllib.parse.quote(ident) + f"?fields={S2FIELDS},citationCount",
        OUT / f"s2_{name}.json", pause=6.0)
    time.sleep(3)
    get(S2 + urllib.parse.quote(ident) + f"/citations?fields={S2FIELDS}&limit=500",
        OUT / f"s2_{name}_citations.json", pause=6.0)
    time.sleep(3)

# --- 5. OpenAlex fallback for non-arXiv journal papers -----------------------
OA = "https://api.openalex.org/works?"
for name, q in {
    "melis_disappearing_disk": "rapid disappearance warm dusty circumstellar disk",
    "meng_planetary_collisions": "planetary collisions outside the solar system extreme debris disks",
    "solano_vanishing_poss": "discovering vanishing objects POSS I red images virtual observatory",
    "abrahamyan_iras_combined": "IRAS PSC FSC combined catalogue",
    "timofeev_dyson_iras": "search of the IRAS database for evidence of Dyson spheres",
    "jugaku_nishimura_dyson": "search for Dyson spheres around late-type stars",
    "slysh_iras_dyson": "Slysh search for IR sources Dyson sphere IRAS",
    "gray_marvel_wow": "VLA search for the Ohio State Wow signal",
    "gray_ellingsen_wow": "search for periodic emissions at the Wow locale",
    "harp_ata_wow": "ATA search repetition Wow signal",
    "villarroel_nuclear_uap": "transients Palomar Observatory Sky Survey nuclear testing unidentified anomalous phenomena",
    "aligned_multiple_transients": "aligned multiple-transient events first Palomar sky survey",
}.items():
    get(OA + urllib.parse.urlencode({"search": q, "per-page": "15",
                                     "mailto": "trimcrae@gmail.com"}),
        OUT / f"oa_{name}.json", pause=4.0)
    time.sleep(2)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")

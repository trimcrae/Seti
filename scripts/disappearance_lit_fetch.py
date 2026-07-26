#!/usr/bin/env python3
"""Literature survey fetcher: time-domain DISAPPEARANCE astronomy.

Complements scripts/vanishing_lit_fetch.py (failed SNe / vanishing massive
stars). This one covers the five domains that fetcher does not:

  A. Infrared excesses that VANISHED (TYC 8241 2652 1, extreme debris disks)
     and systematic IRAS(1983)/AKARI(2006) vs WISE(2010)/NEOWISE(2024)
     cross-epoch searches.                                        [prior-art S1]
  B. Catalogue-scale WISE/NEOWISE mid-IR variability surveys, and the
     specific question of optically-CONSTANT / mid-IR-VARIABLE stars.
  C. Radio sources that disappeared: NVSS/FIRST vs VLASS, and whether anyone
     cross-matched faded radio sources to STELLAR (Gaia) counterparts. [S31]
  D. Periodic signals that CEASED (variables, pulsators, eclipsing binaries,
     transits) and cross-epoch variability-catalogue comparisons.     [S32]
  E. Longest-baseline photometric-change searches: DASCH / Harvard plates,
     APPLAUSE, POSS/USNO/GSC vs Gaia/Pan-STARRS/ZTF; and stars whose
     photometric SCATTER is secularly increasing.                     [S9]
  F. VASCO vanished sources vs infrared catalogues (enshrouded not
     destroyed).                                                      [S33]

The sandbox egress policy blocks arxiv.org / ADS / publisher hosts, so this
runs on the GitHub Actions runner (CLAUDE.md acquisition pattern) and commits
verbatim abstracts + full text back to the branch.

Outputs under results/disaplit/:
  arxiv_ids_<n>.atom    - arXiv API metadata (title/authors/abstract) by ID
  arxiv_q_<name>.atom   - arXiv API search results per query
  html_<id>.html        - arXiv full-text HTML for argument-critical papers
  s2_<name>.json        - Semantic Scholar records / citation lists
  summary.json          - fetch status for every URL
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/disaplit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-disaplit/1.0 (mailto:trimcrae@gmail.com)"}
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

# --- 1. Papers by arXiv ID (verbatim abstracts). IDs marked (verify) are best
#        guesses; the section-2 searches are the authoritative resolver. ------
IDS = [
    # A. vanished / variable infrared excesses
    "1207.0697",   # Melis+ 2012 Nature - rapid disappearance of a warm dusty disk (verify)
    "1612.06144",  # Osten? / TYC 8241 2652 1 "no smoking gun yet" A&A 2017 (verify)
    "1205.0034",   # Meng+ 2012 variability of the IR excess of extreme debris disks (verify)
    "1408.1636",   # Meng+ 2014 Science large impacts around a solar analogue (verify)
    "1503.05610",  # Meng+ 2015 planetary collisions outside the solar system (verify)
    "1903.10627",  # Su+ 2019 extreme debris disk variability (confirmed by search)
    "2103.00568",  # Moor+ 2021 new sample of warm extreme debris disks from ALLWISE
    "1903.03041",  # BD+20 307 SOFIA evolution of warm dust
    "2001.04475",  # "A Word to the WISE: confusion is unavoidable" (verify)
    # B. NEOWISE / WISE variability at catalogue scale
    "2511.22071",  # Catalogue of mid-IR variable sources from unTimely (confirmed)
    "2209.14990",  # unTimely catalog, Meisner+ 2023 (verify)
    "2107.10751",  # Park+ 2021 YSO mid-IR variability over 6 yr with NEOWISE
    "2503.13971",  # Illuminating Youth: decades of mid-IR variability of YSOs
    "2012.10195",  # MIRONG Jiang+ 2021 mid-IR outbursts in nearby galaxies (verify)
    "1805.06920",  # Stern+ 2018 mid-IR selected changing-look / WISE AGN var (verify)
    "2012.13084",  # CatWISE2020 Marocco+ 2021 (verify)
    # C. radio disappearance
    "astro-ph/0512197",  # Gal-Yam+ 2006 radio transient survey NVSS vs FIRST (verify)
    "1105.6047",   # Thyagarajan+ 2011 variable and transient sources in FIRST (verify)
    "1607.04907",  # Mooley+ 2016 CNSS Stripe 82 (verify)
    "2007.02947",  # Nyland+ 2020 VLASS quasars radio-quiet to radio-loud (verify)
    "2410.06210",  # inverted-spectrum VLASS/VCSS transients (confirmed by search)
    "2101.05238",  # Callingham+ 2021 LOFAR radio star census (verify)
    "2012.14993",  # Pritchard+ 2021 RACS circular polarization radio stars (verify)
    # D. periodicity that ceased
    "2501.14058",  # Life in the Slow Lane: long-term variability in ASAS-SN (confirmed)
    "2303.02523",  # GCVS x Gaia DR3 cross-match (verify)
    # E. long baseline plates
    "2501.12977",  # DASCH: bringing 100+ years of photographic data (confirmed)
    "1509.01606",  # Schaefer 2016 KIC 8462852 century-long dimming (verify)
    "1601.07314",  # Hippke & Angerhausen rebuttal (verify)
    "1605.09623",  # Lund+ 2016 / Hippke+ DASCH systematics (verify)
    # F. VASCO
    "1911.05068",  # Villarroel+ 2020 VASCO I USNO missing objects (verify)
    "2009.10813",  # Villarroel+ 2021 nine simultaneous POSS-I transients (verify)
    "2204.00429",  # Villarroel+ 2022 VASCO citizen science (verify)
]
CH = 8
for i in range(0, len(IDS), CH):
    chunk = IDS[i:i + CH]
    get(ARXIV + urllib.parse.urlencode({"id_list": ",".join(chunk),
                                        "max_results": len(chunk)}),
        OUT / f"arxiv_ids_{i//CH}.atom")
    time.sleep(3.2)

# --- 2. arXiv API searches (Atom). These are the authoritative resolver. -----
QUERIES = {
    # ---------- A. infrared excesses that vanished / turned on ----------
    "disk_disappear": 'abs:"debris disk" AND (abs:disappearance OR abs:disappearing OR abs:vanish)',
    "tyc8241": 'all:"TYC 8241"',
    "edd_variability": 'abs:"extreme debris disk" AND abs:variability',
    "ir_excess_variable": 'abs:"infrared excess" AND (abs:variable OR abs:variability) AND abs:star',
    "iras_wise_compare": 'abs:IRAS AND abs:WISE AND (abs:comparison OR abs:crossmatch OR abs:"cross-match")',
    "iras_not_detected": 'abs:IRAS AND abs:"not detected" AND abs:WISE',
    "akari_wise": 'abs:AKARI AND abs:WISE AND abs:"point source catalogue"',
    "ir_excess_vanished": 'abs:"infrared excess" AND (abs:vanished OR abs:disappeared OR abs:"turned off")',
    "iras_faint_source_reliability": 'abs:"IRAS Faint Source" AND (abs:reliability OR abs:spurious OR abs:confusion)',
    "disk_dispersal_caught": 'abs:"disk dispersal" AND abs:timescale AND abs:transition',
    "dyson_variability": 'abs:"Dyson sphere" AND (abs:variability OR abs:variable OR abs:transit)',
    "hephaistos": 'all:"Project Hephaistos"',
    # ---------- B. NEOWISE / mid-IR variability at scale ----------
    "neowise_variability": 'abs:NEOWISE AND (abs:variability OR abs:"variable sources")',
    "midir_variability_survey": 'ti:"mid-infrared" AND ti:variability AND (ti:survey OR ti:catalog OR ti:catalogue)',
    "wise_variable_catalog": 'ti:WISE AND (ti:"variable" OR ti:"variability") AND ti:catalog',
    "optically_quiet_ir_variable": 'abs:"mid-infrared" AND abs:variable AND abs:"no optical" ',
    "optical_ir_discrepant": 'abs:"optical" AND abs:"mid-infrared" AND abs:variability AND abs:"in contrast"',
    "secular_ir_trend": 'abs:"mid-infrared" AND (abs:"secular" OR abs:"monotonic" OR abs:"long-term trend") AND abs:fading',
    "neowise_photometric_offset": 'abs:NEOWISE AND abs:AllWISE AND (abs:calibration OR abs:offset OR abs:photometric)',
    "untimely": 'all:"unTimely"',
    "unwise_timeresolved": 'abs:unWISE AND abs:"time-resolved" AND abs:coadds',
    "neowise_ml_lightcurves": 'abs:NEOWISE AND abs:"light curves" AND (abs:"machine learning" OR abs:classification)',
    # ---------- C. radio sources that disappeared ----------
    "vanishing_radio": 'abs:radio AND (abs:"disappearing sources" OR abs:"vanishing sources" OR abs:"vanished")',
    "nvss_vlass_transient": 'abs:NVSS AND abs:VLASS AND (abs:transient OR abs:variable)',
    "first_nvss_transient": 'abs:FIRST AND abs:NVSS AND abs:transient AND abs:survey',
    "radio_transient_archival": 'ti:"radio transient" AND (ti:survey OR ti:search)',
    "dying_radio_galaxy": 'abs:"remnant radio galaxy" OR abs:"dying radio galaxy"',
    "radio_faded_agn": 'abs:radio AND abs:AGN AND (abs:"switched off" OR abs:"faded" OR abs:"turned off")',
    "radio_stars_gaia": 'abs:"radio stars" AND abs:Gaia',
    "vlass_stellar": 'abs:VLASS AND (abs:stellar OR abs:star) AND abs:counterpart',
    "vast_askap_transient": 'abs:ASKAP AND abs:VAST AND abs:variable AND abs:radio',
    "lotss_transient": 'abs:LOFAR AND abs:transient AND abs:survey',
    "resolution_bias_nvss_first": 'abs:NVSS AND abs:FIRST AND (abs:"resolved out" OR abs:"resolution" ) AND abs:flux',
    # ---------- D. periodic signals that ceased ----------
    "ceased_pulsation": 'abs:pulsation AND (abs:ceased OR abs:"stopped" OR abs:"disappeared")',
    "amplitude_decline_cepheid": 'abs:Cepheid AND abs:amplitude AND (abs:decline OR abs:decreasing)',
    "polaris_amplitude": 'all:"Polaris" AND abs:amplitude',
    "mode_switching_rr_lyrae": 'abs:"RR Lyrae" AND (abs:"mode switching" OR abs:"mode change")',
    "disappearing_pulsations_wd": 'abs:"white dwarf" AND abs:pulsation AND (abs:amplitude AND abs:variability)',
    "eclipses_ceased": 'abs:eclipsing AND abs:binary AND (abs:"eclipses ceased" OR abs:"no longer eclipsing" OR abs:"disappearance of eclipses")',
    "ss_lac": 'all:"SS Lacertae" OR all:"SS Lac"',
    "precessing_eb": 'abs:"eclipsing binary" AND abs:precession AND abs:inclination AND abs:"third body"',
    "vanishing_transits": 'abs:transit AND (abs:"disappeared" OR abs:"vanishing" OR abs:"ceased")',
    "kic12557548": 'all:"KIC 12557548" OR all:"Kepler-1520"',
    "stopped_varying": 'abs:"stopped varying" OR abs:"cessation of variability" OR abs:"no longer variable"',
    "variability_crossepoch": 'abs:"variable stars" AND abs:"cross-match" AND (abs:ZTF OR abs:"ASAS-SN") AND abs:catalog',
    "gcvs_modern": 'abs:GCVS AND (abs:Gaia OR abs:ZTF OR abs:"ASAS-SN")',
    "spurious_variables": 'abs:"variable star" AND (abs:spurious OR abs:"misclassified") AND abs:catalog',
    # ---------- E. longest-baseline photometric change ----------
    "dasch": 'all:"DASCH"',
    "harvard_plates": 'abs:"Harvard" AND abs:"photographic plates" AND abs:"light curves"',
    "applause": 'all:"APPLAUSE" AND abs:plates',
    "plate_archive_variability": 'abs:"plate archive" AND (abs:variability OR abs:photometry) AND abs:century',
    "century_photometry": 'abs:"century" AND abs:photometric AND (abs:variability OR abs:"long-term")',
    "usno_gaia_photometry": 'abs:"USNO-B" AND (abs:Gaia OR abs:"Pan-STARRS") AND abs:photometr',
    "poss_digitization": 'abs:POSS AND (abs:digitiz OR abs:scan) AND abs:photometr',
    "photographic_photometry_systematics": 'abs:photographic AND abs:photometry AND (abs:systematics OR abs:nonlinearity OR abs:"color term")',
    "secular_brightness_change": 'abs:"secular" AND abs:"brightness" AND abs:change AND abs:stars',
    "increasing_variability": 'abs:"increasing" AND abs:"variability amplitude" AND abs:stars',
    "emerging_variability": 'abs:"newly variable" OR abs:"emerging variability" OR abs:"onset of variability"',
    "photometric_scatter_trend": 'abs:"photometric scatter" AND (abs:"increase" OR abs:"trend") AND abs:survey',
    # ---------- F. VASCO and its infrared follow-up ----------
    "vasco_all": 'all:"VASCO" AND all:"vanishing"',
    "vasco_ir": 'abs:vanishing AND abs:sources AND (abs:infrared OR abs:WISE OR abs:2MASS)',
    "vasco_critique": 'abs:"POSS" AND abs:transient AND (abs:"plate defect" OR abs:contamination OR abs:artefact OR abs:artifact)',
    "plate_defects": 'abs:"photographic plate" AND (abs:defects OR abs:artefacts) AND abs:transient',
    "glint_satellite_plate": 'abs:glint AND (abs:plate OR abs:"geosynchronous" OR abs:satellite) AND abs:transient',
    "technosignature_transient": 'abs:technosignature AND (abs:transient OR abs:"optical flash")',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode({"search_query": q,
                                        "start": 0, "max_results": 40,
                                        "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)

# --- 3. Full-text HTML for argument-critical papers --------------------------
HTML_IDS = [
    "2511.22071",  # unTimely mid-IR variable catalogue - selection criteria
    "2501.12977",  # DASCH DR7 - coverage, depth, product description
    "2501.14058",  # ASAS-SN slow-variability search - what they selected on
    "2103.00568",  # warm extreme debris disks from ALLWISE
    "2503.13971",  # decades of mid-IR variability of YSOs
    "2410.06210",  # VLASS/VCSS transients - NVSS non-detection methodology
]
for aid in HTML_IDS:
    get(f"https://arxiv.org/html/{aid}v1", OUT / f"html_{aid.replace('/', '_')}.html")
    time.sleep(2.0)

# --- 4. Semantic Scholar: citation graph for the pivotal papers --------------
S2 = "https://api.semanticscholar.org/graph/v1/paper/"
S2_FIELDS = "title,year,externalIds,abstract,citationCount,venue"
S2_TARGETS = {
    # arXiv IDs of works whose CITING papers reveal whether the follow-up
    # searches I care about were ever run.
    "melis2012_tyc8241": "arXiv:1207.0697",
    "untimely_varcat": "arXiv:2511.22071",
    "dasch_dr7": "arXiv:2501.12977",
    "vasco_i": "arXiv:1911.05068",
    "su2019_edd": "arXiv:1903.10627",
}
for name, pid in S2_TARGETS.items():
    get(f"{S2}{urllib.parse.quote(pid)}?fields={S2_FIELDS}",
        OUT / f"s2_{name}.json")
    time.sleep(3.5)
    get(f"{S2}{urllib.parse.quote(pid)}/citations?fields={S2_FIELDS}&limit=200",
        OUT / f"s2_{name}_citations.json")
    time.sleep(3.5)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n=== disaplit fetch done: {ok}/{len(STATUS)} OK ===")

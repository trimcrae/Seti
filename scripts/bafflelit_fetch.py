#!/usr/bin/env python3
"""BAFFLE prior-art sweep: fetch the record on the GitHub runner.

The sandbox blocks arxiv.org (``CONNECT tunnel failed, response 403``); the
Actions runner has egress.  This script establishes, from the record rather
than from memory, whether anyone has already looked for the BAFFLE signature.

THE CHANNEL'S CLAIM.  Under the zoo hypothesis a warden civilisation would
shield the solar system's technosignatures from other stars with
band-selective screens ("baffles") placed on the Sun->observer lines in the
outer solar system (10^2--10^4 AU, >= 1 AU in radius).  By reciprocity, from
Earth such a screen makes the target star (and its few-arcminute surroundings)
appear DEFICIENT in the mid-infrared (W1/W2/W3 fainter than the photosphere
predicts, with normal optical / near-IR), or deficient in the radio (a hole in
deep radio source counts around a nearby star), and the deficit carries an
annual-parallax signature because the screen is in the solar system.

Five novelty questions, each its own query group with its own decoys:

  g1_mir_deficit     Has anyone searched catalogues for stars whose WISE /
                     Spitzer / AKARI flux is significantly BELOW the
                     photospheric prediction (the inverse of an IR-excess
                     search)?  Decoys: IR-EXCESS searches (Dyson / debris
                     disks), T-dwarf methane W1 suppression, CO-band W2
                     depression in giants, WISE saturation / photometric-bias
                     papers, "missing WISE counterpart" cross-match
                     completeness papers.
  g2_zoo_tests       Observational tests of the zoo / planetarium hypothesis
                     (Ball 1973; Baxter 2001; Forgan 2011 / 2017; Crawford &
                     Schulze-Makuch 2024; anything proposing an observable).
                     Decoys: philosophical Fermi-paradox reviews with no
                     proposed observable.
  g3_occulters       Large artificial screens / occulters in the outer solar
                     system and their parallax signature; Planet Nine and
                     distant-body searches in WISE / Planck / AKARI / IRAS by
                     parallax and motion; the cloaking literature (Kipping &
                     Teachey 2016).  Decoys: engineering of OUR OWN starshades.
  g4_radio_voids     Any search for an anomalous absence of background radio
                     sources near stars; LoTSS DR2 completeness near bright
                     sources; "radio shadow"; the ARCADE-2 excess as a possible
                     foreground.  Decoys: radio emission FROM stars, stellar
                     radio SETI.
  g5_concealment     Anisotropic / directional technosignature concealment
                     (Lacki "Sunscreen"; Wright et al. Ĝ concealment remarks;
                     Bradbury / Cirkovic Dysonian-SETI concealment arguments).
                     Decoys: generic Fermi-paradox reviews.

Nothing here is asserted from memory: named papers are fetched by *title
search* where the arXiv id is not certain, and where an id is given the fetched
title is compared against the expected one and any mismatch is recorded in
``id_title_check.json`` --- never silently trusted.  Verbatim abstracts are
saved; nothing is paraphrased.  Every fetch's HTTP status is recorded.

Outputs under ``results/bafflelit/``:
  arxiv_q_<group>__<name>.atom        arXiv API keyword search (verbatim)
  arxiv_q_title_<group>__<name>.atom  arXiv API title search (verbatim)
  arxiv_id_<group>__<name>.atom       arXiv API metadata for one id (verbatim)
  concept_scan.json                   decoy-aware scan, per group, over every
                                      fetched abstract
  id_title_check.json                 did each asserted id resolve to the
                                      expected title?  and did each title
                                      search return the expected title?
  summary.json                        HTTP status of every URL, entry counts,
                                      the non-arXiv references the brief names
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path(os.environ.get("BAFFLELIT_OUT", "results/bafflelit"))
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-bafflelit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = float(os.environ.get("BAFFLELIT_PAUSE", "3.0"))
TRIES = int(os.environ.get("BAFFLELIT_TRIES", "3"))
ARXIV_API = "http://export.arxiv.org/api/query"
MAX_RESULTS = 60


def _write_summary() -> None:
    """Written after EVERY fetch so a soft-deadline kill still leaves a record."""
    (OUT / "summary.json").write_text(json.dumps(
        {"n_urls": len(STATUS), "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"]),
         "non_arxiv_references": NON_ARXIV_REFERENCES,
         "groups": {g: spec["question"] for g, spec in GROUPS.items()},
         "status": STATUS}, indent=2))


def get(url: str, dest: pathlib.Path, tries: int = TRIES, pause: float = PAUSE) -> bool:
    """Fetch one URL to ``dest`` verbatim; the HTTP status of every attempt is recorded."""
    attempts: list[dict] = []
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
                http_status = int(getattr(r, "status", 0) or 0)
            dest.write_bytes(data)
            n_ent = len(re.findall(r"<entry>", data.decode("utf-8", "ignore")))
            attempts.append({"http_status": http_status, "bytes": len(data)})
            STATUS.append({"url": url, "dest": dest.name, "ok": True, "http_status": http_status,
                           "bytes": len(data), "n_entries": n_ent, "attempts": attempts})
            print(f"  ok  HTTP {http_status}  {len(data):>9,}B  {n_ent:>3} entries  {dest.name}")
            _write_summary()
            time.sleep(pause)
            return True
        except urllib.error.HTTPError as exc:
            attempts.append({"http_status": int(exc.code), "error": repr(exc)})
            print(f"  try {i + 1}/{tries} HTTP {exc.code}: {exc!r}")
            time.sleep(pause * (i + 1))
        except Exception as exc:  # noqa: BLE001
            attempts.append({"http_status": None, "error": repr(exc)})
            print(f"  try {i + 1}/{tries} failed: {exc!r}")
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "dest": dest.name, "ok": False,
                   "http_status": attempts[-1]["http_status"] if attempts else None,
                   "attempts": attempts})
    _write_summary()
    return False


# --------------------------------------------------------------------------
# References the brief names that PREDATE arXiv or were never posted there.
# They cannot be title-verified through the arXiv API; they are recorded here
# as bibliographic assertions FROM THE BRIEF, flagged as such, and a title
# search is still run so that any arXiv paper quoting the title is caught.
# --------------------------------------------------------------------------
NON_ARXIV_REFERENCES = [
    {"key": "ball1973_zoo", "citation": "Ball, J. A. 1973, Icarus 19, 347, 'The zoo hypothesis'",
     "verification": "not possible via the arXiv API (pre-arXiv journal paper); asserted by the brief"},
    {"key": "baxter2001_planetarium",
     "citation": "Baxter, S. 2001, JBIS 54, 210, 'The planetarium hypothesis: a resolution of the Fermi paradox'",
     "verification": "not possible via the arXiv API (JBIS, not on arXiv); asserted by the brief"},
    {"key": "shimwell2022_lotss_dr2",
     "citation": "Shimwell et al. 2022, A&A 659, A1, 'The LOFAR Two-metre Sky Survey. V. Second data release'",
     "verification": "journal citation asserted by the brief; the arXiv posting is title-checked below (g4)"},
    {"key": "fixsen2011_arcade2",
     "citation": "Fixsen et al. 2011, ApJ 734, 5, 'ARCADE 2 Measurement of the Absolute Sky Brightness at 3-90 GHz'",
     "verification": "journal citation asserted by the brief; the arXiv posting is title-checked below (g4)"},
]


# --------------------------------------------------------------------------
# The five query groups.  For each: named papers by id (title-checked), named
# papers by title (cannot be wrong about the id), keyword sweeps (target AND
# decoys, fetched on purpose so a null is interpretable), and the regexes of
# the decoy-aware scan.  ``confidence`` on an id is the author's prior; the
# fetched title decides, and a mismatch is recorded in id_title_check.json.
# --------------------------------------------------------------------------
GROUPS: dict[str, dict] = {
    # ------------------------------------------------------------------
    "g1_mir_deficit": {
        "question": (
            "Has anyone searched catalogues for stars whose WISE/Spitzer/AKARI flux is "
            "significantly BELOW the photospheric prediction (the inverse of an IR-excess "
            "search)?  A genuine hit speaks of a mid-IR DEFICIT / sub-photospheric flux, "
            "not of an excess."),
        "by_id": {},
        "by_title": {
            "wright2014_ghat_ii_wise_framework":
                "Infrared Search for Extraterrestrial Civilizations with Large Energy Supplies. II. Framework, Strategy, and First Result",
            "griffith2015_ghat_iii":
                "Infrared Search for Extraterrestrial Civilizations with Large Energy Supplies. III. The Reddest Extended Sources in WISE",
            "suazo2022_hephaistos_i":
                "Project Hephaistos I. Upper limits on partial Dyson spheres in the Milky Way",
            "suazo2024_hephaistos_ii":
                "Project Hephaistos II. Dyson sphere candidates from Gaia DR3, 2MASS, and WISE",
            "kirkpatrick2011_wise_brown_dwarfs":
                "The First Hundred Brown Dwarfs Discovered by the Wide-field Infrared Survey Explorer (WISE)",
            "patel2014_wise_debris_disks":
                "A Sensitive Identification of Warm Debris Disks in the Solar Neighborhood through Precise Calibration of Saturated WISE Photometry",
        },
        "queries": {
            # --- the target concept ---
            "infrared_deficit_stars_wise": 'all:"infrared deficit" AND all:stars AND all:WISE',
            "infrared_deficit": 'all:"infrared deficit"',
            "sub_photospheric_mid_infrared": 'all:"sub-photospheric" AND all:infrared',
            "fainter_than_photosphere_wise": 'all:"fainter than" AND all:photosphere AND all:WISE',
            "fainter_than_photosphere_infrared": 'all:"fainter than the photosphere" AND all:infrared',
            "missing_allwise_counterparts_bright_stars": 'all:"missing" AND all:AllWISE AND all:counterparts AND all:"bright stars"',
            "negative_infrared_excess": 'all:"negative infrared excess"',
            "negative_excess_wise": 'all:"negative excess" AND all:WISE',
            "infrared_deficient_stars": 'all:"infrared deficient" AND all:stars',
            "mid_infrared_deficit_photosphere": 'all:"mid-infrared" AND all:deficit AND all:photosphere',
            "flux_below_photospheric_prediction": 'all:"below the photospheric" AND all:infrared',
            "underluminous_mid_infrared": 'all:underluminous AND all:"mid-infrared" AND all:stars',
            "technosignature_infrared_deficit": 'all:technosignature AND all:infrared AND all:deficit',
            # --- decoys, fetched on purpose ---
            "decoy_infrared_excess_dyson": 'all:"infrared excess" AND all:"Dyson sphere"',
            "decoy_infrared_excess_debris_wise": 'all:"infrared excess" AND all:"debris disk" AND all:WISE',
            "decoy_t_dwarf_methane_w1": 'all:"T dwarf" AND all:methane AND all:W1 AND all:WISE',
            "decoy_co_band_w2_giants": 'all:"CO" AND all:"4.6" AND all:giants AND all:WISE AND all:absorption',
            "decoy_wise_saturation_bias": 'all:WISE AND all:saturation AND all:photometry AND all:bias',
            "decoy_missing_wise_counterparts_completeness": 'all:WISE AND all:counterparts AND all:completeness AND all:"cross-match"',
        },
        "target": [
            r"(infrared|IR|mid-?infrared|WISE|W[123]|Spitzer|AKARI|IRAC|MIPS).{0,40}(flux |emission )?defici(t|ency|ent)",
            r"defici(t|ency|ent).{0,40}(in the |of |at )?(infrared|IR|mid-?infrared|W[123]|WISE|Spitzer|AKARI)",
            r"fainter than (the |its |their )?(predicted |expected )?photospher",
            r"sub-?photospheric",
            r"negative (infrared |IR |mid-?infrared )?excess",
            r"below (the |its )?(predicted |expected )?photospher(e|ic)",
            r"(missing|lack of|absent|absence of) (WISE|AllWISE|mid-?infrared|W[123]) (counterpart|detection|emission|flux)",
            r"(too )?faint(er)? in (the )?(mid-?infrared|WISE|W[123])",
            r"(under-?luminous|underluminous).{0,40}(infrared|mid-?infrared|WISE)",
        ],
        "decoys": {
            "ir_excess": r"(infrared|IR|mid-?infrared|WISE|W[1234]|Spitzer|AKARI).{0,20}excess|excess.{0,20}(infrared|IR|emission)|debris dis[ck]|Dyson (sphere|swarm|shell)|waste heat|circumstellar dust",
            "t_dwarf_methane": r"\bT ?dwarf|methane|CH4|\bY ?dwarf|brown dwarf",
            "giant_co_band": r"\bCO\b.{0,40}(band|absorption|fundamental)|4\.[67] ?(micron|μm|um).{0,40}absorption|red giant|AGB|carbon star",
            "wise_saturation_bias": r"saturat(ed|ion)|photometric bias|zero-?point|calibration (error|offset|bias)",
            "crossmatch_completeness": r"cross-?match|completeness|counterpart(s)? (identification|rate)|astrometric (offset|error)|source confusion|blend(ed|ing)",
        },
        "boosters": {
            "stellar_catalogue_search": r"catalog(ue)?|survey|sample of \d|all-?sky|Gaia",
            "seti_framing": r"technosignature|SETI|extraterrestrial|artificial",
        },
        "interpretation": (
            "A decoy-free hit is an abstract that speaks of a stellar mid-IR DEFICIT / "
            "sub-photospheric flux / negative excess.  ir_excess tags the thousands of "
            "excess searches the target regex can brush against (a deficit paper may also "
            "say 'excess', so an ir_excess tag is not a veto -- read the snippet); "
            "t_dwarf_methane and giant_co_band are natural band suppressions; "
            "wise_saturation_bias and crossmatch_completeness are the instrumental "
            "reasons a star can look faint in WISE.  If no decoy-free hit describes a "
            "catalogue search for deficit stars, g1 is novel on this record."),
    },
    # ------------------------------------------------------------------
    "g2_zoo_tests": {
        "question": (
            "Has anyone proposed or performed an OBSERVATIONAL test of the zoo / "
            "planetarium hypothesis (as opposed to discussing it philosophically)?"),
        "by_id": {
            # name: (arXiv id, expected title fragment, prior confidence in the id)
            "forgan2011_spatiotemporal_zoo": ("1105.2497", "Spatio-temporal constraints on the zoo hypothesis", "medium"),
            "forgan2017_galactic_club_cliques": ("1608.08770", "Galactic Club or Galactic Cliques", "medium"),
            "crawford2024_zoo_or_nothing": ("2401.01532", "zoo hypothesis or nothing", "low"),
        },
        "by_title": {
            "ball1973_zoo_hypothesis": "The zoo hypothesis",
            "baxter2001_planetarium_hypothesis": "The planetarium hypothesis",
            "forgan2011_spatiotemporal_zoo": "Spatio-temporal constraints on the zoo hypothesis, and the breakdown of total hegemony",
            "forgan2017_galactic_club_cliques": "The Galactic Club or Galactic Cliques? Exploring the limits of interstellar hegemony and the Zoo Hypothesis",
            "crawford2024_zoo_or_nothing": "Is the apparent absence of extraterrestrial technological civilizations down to the zoo hypothesis or nothing?",
            "hair_hedman2013_percolation": "Spatial dispersion of interstellar civilizations: a probabilistic site percolation model in three dimensions",
            "cirkovic2018_great_silence": "The Great Silence: Science and Philosophy of Fermi's Paradox",
        },
        "queries": {
            # --- the target concept ---
            "zoo_hypothesis": 'all:"zoo hypothesis"',
            "planetarium_hypothesis": 'all:"planetarium hypothesis"',
            "zoo_hypothesis_observational_test": 'all:"zoo hypothesis" AND all:observational AND all:test',
            "zoo_hypothesis_prediction": 'all:"zoo hypothesis" AND all:prediction',
            "interdict_hypothesis": 'all:"interdict" AND all:extraterrestrial',
            "galactic_club_quarantine": 'all:"galactic club" OR all:"quarantine" AND all:extraterrestrial',
            "interstellar_hegemony": 'all:"hegemony" AND all:interstellar',
            "zoo_hypothesis_technosignature": 'all:"zoo hypothesis" AND all:technosignature',
            # --- decoys, fetched on purpose ---
            "decoy_fermi_paradox_review": 'all:"Fermi paradox" AND all:review',
            "decoy_fermi_paradox_solutions": 'all:"Fermi paradox" AND all:solutions',
            "decoy_great_filter": 'all:"Great Filter"',
        },
        "target": [
            r"zoo hypothesis",
            r"planetarium hypothesis",
            r"interdict(ion)? (hypothesis|scenario)",
            r"galactic (club|clique)s?",
            r"quarantine",
            r"(total |interstellar )?hegemony",
            r"(deliberately|intentionally) (hidden|concealed|isolated|avoid)",
        ],
        "decoys": {
            "fermi_review_no_observable": r"Fermi('s)? paradox|Great Silence|Great Filter|Drake equation",
            "sociological_only": r"sociolog|philosoph|ethic|policy|game[- ]theor|cultural",
        },
        "boosters": {
            "proposes_observable": r"observ(able|ational|ationally) (test|signature|consequence|prediction|constraint)|testable|falsif|predict(s|ion)|signature|detect(able|ion)|search(ed|es|ing)? for",
            "solar_system_screen": r"solar system|Oort|Kuiper|occult|screen|shield|cloak|baffle",
        },
        "interpretation": (
            "A decoy-free hit is an abstract that names the zoo / planetarium / interdict "
            "hypothesis without being a generic Fermi-paradox review.  Decoys here are "
            "weak (most zoo papers also say 'Fermi paradox'), so the boosters matter: a "
            "hit with proposes_observable AND solar_system_screen is the prior art to "
            "read.  Ball 1973 and Baxter 2001 predate arXiv; their title searches can "
            "only catch arXiv papers that quote the title."),
    },
    # ------------------------------------------------------------------
    "g3_occulters": {
        "question": (
            "Has anyone searched for large artificial screens / occulters in the outer "
            "solar system, or for their annual-parallax signature, in WISE / Planck / "
            "AKARI / IRAS?  (Planet-Nine parallax searches are the closest natural-body "
            "analogue; cloaking papers are the closest artificial one.)"),
        "by_id": {
            "kipping_teachey2016_cloaking": ("1603.08928", "cloaking device for transiting planets", "high"),
            "meisner2017_p9_wise_coadd": ("1611.00015", "Planet Nine", "medium"),
            "meisner2018_p9_3pi_wise": ("1712.04950", "Planet Nine", "medium"),
        },
        "by_title": {
            "kipping_teachey2016_cloaking": "A cloaking device for transiting planets",
            "meisner2017_p9_wise_coadd": "Searching for Planet Nine with Coadded WISE and NEOWISE-Reactivation Images",
            "meisner2018_p9_3pi_wise": "Search for Planet Nine at 3.4 microns with WISE",
            "arnold2005_artificial_transits": "Transit lightcurve signatures of artificial objects",
            "wright2020_dyson_spheres_review": "Dyson Spheres",
            "batygin_brown2016_p9": "Evidence for a Distant Giant Planet in the Solar System",
            "cowan2016_p9_thermal": "Cosmologists in Search of Planet Nine: the Case for CMB Experiments",
        },
        "queries": {
            # --- the target concept ---
            "planet_nine_wise_parallax": 'all:"Planet Nine" AND all:WISE AND all:parallax',
            "planet_nine_planck": 'all:"Planet Nine" AND all:Planck',
            "planet_nine_akari": 'all:"Planet Nine" AND all:AKARI',
            "planck_cold_sources_parallax": 'all:Planck AND all:"cold sources" AND all:parallax',
            "planck_moving_source_solar_system": 'all:Planck AND all:"moving" AND all:"solar system" AND all:planet',
            "iras_moving_source_planet_x": 'all:IRAS AND all:"Planet X"',
            "akari_moving_object_far_infrared": 'all:AKARI AND all:"moving" AND all:"solar system" AND all:"far-infrared"',
            "distant_solar_system_body_parallax_infrared": 'all:"solar system" AND all:parallax AND all:"infrared" AND all:distant AND all:search',
            "cloaking_technosignature": 'all:cloaking AND all:technosignature',
            "cloaking_extraterrestrial": 'all:cloaking AND all:extraterrestrial',
            "artificial_occulter_star": 'all:"artificial" AND all:occulter AND all:star',
            "dyson_swarm_shield": 'all:"Dyson" AND all:shield AND all:SETI',
            "megastructure_shadow_outer_solar_system": 'all:megastructure AND all:"solar system" AND all:"outer"',
            "artificial_object_oort_cloud": 'all:artificial AND all:"Oort cloud" AND all:SETI',
            "technosignature_solar_system_search": 'all:technosignature AND all:"solar system" AND all:search AND all:artifact',
            # --- decoys, fetched on purpose ---
            "decoy_starshade_engineering": 'all:starshade AND all:"formation flying"',
            "decoy_starshade_occulter_design": 'all:"external occulter" AND all:starshade AND all:design',
            "decoy_kbo_wise_parallax": 'all:"Kuiper belt" AND all:WISE AND all:parallax',
        },
        "target": [
            r"cloak(ing|ed)?",
            r"artificial (occult\w*|screen|shield|shade|megastructure|object)",
            r"occult\w*.{0,60}(artificial|megastructure|technosignature|extraterrestrial)",
            r"Planet (Nine|9|X)\b.{0,120}(WISE|Planck|AKARI|IRAS|parallax|proper motion)",
            r"(WISE|Planck|AKARI|IRAS).{0,120}Planet (Nine|9|X)\b",
            r"parallax.{0,100}(WISE|Planck|AKARI|IRAS|solar system|distant)",
            r"moving (source|object)s?.{0,80}(IRAS|AKARI|WISE|Planck|infrared)",
            r"Dyson (swarm|sphere|shell)s?.{0,20}(shield|screen|shadow|partial)",
            r"(Oort cloud|outer solar system|Kuiper belt).{0,80}(artificial|technosignature|megastructure|probe|artifact)",
        ],
        "decoys": {
            "own_starshade_engineering": r"starshade|external occulter|coronagraph|formation[- ]flying|HabEx|LUVOIR|WFIRST|Roman Space Telescope|petal",
            "natural_body_search": r"Kuiper belt object|KBO|trans-?Neptunian|TNO|asteroid|comet|dwarf planet|Sedna|scattered disk",
            "exoplanet_transit_only": r"transit(ing|s)?.{0,20}(planet|exoplanet|light ?curve)",
        },
        "boosters": {
            "parallax_or_motion": r"parallax|proper motion|moving|motion",
            "mid_ir_or_radio_data": r"WISE|NEOWISE|AKARI|IRAS|Planck|Spitzer|radio|LOFAR",
            "seti_framing": r"technosignature|SETI|extraterrestrial|artificial|megastructure",
        },
        "interpretation": (
            "A decoy-free hit is an abstract about an artificial occulter / screen / cloak, "
            "or a parallax / motion search for distant bodies in the mid-IR or mm data.  "
            "natural_body_search tags Planet-Nine / KBO searches: not prior art for an "
            "ARTIFICIAL screen, but the methodological baseline this channel reuses "
            "(their sensitivity to a >= 1 AU screen is what the search must beat).  "
            "own_starshade_engineering is the scale reference only.  A hit with "
            "seti_framing AND parallax_or_motion AND mid_ir_or_radio_data is the "
            "prior art to read."),
    },
    # ------------------------------------------------------------------
    "g4_radio_voids": {
        "question": (
            "Has anyone searched for an anomalous ABSENCE of background radio sources "
            "around (nearby) stars -- a hole in deep radio source counts -- and what is "
            "the known completeness of deep surveys near bright stars?"),
        "by_id": {
            "fixsen2011_arcade2": ("0901.0555", "ARCADE 2", "high"),
            "shimwell2022_lotss_dr2": ("2202.11733", "LOFAR Two-metre Sky Survey", "high"),
        },
        "by_title": {
            "fixsen2011_arcade2": "ARCADE 2 Measurement of the Absolute Sky Brightness at 3-90 GHz",
            "shimwell2022_lotss_dr2": "The LOFAR Two-metre Sky Survey V. Second data release",
            "seiffert2011_arcade2_interpretation": "Interpretation of the ARCADE 2 Absolute Sky Brightness Measurement",
            "hardcastle2023_lotss_dr2_optical": "The LOFAR Two-Metre Sky Survey VI. Optical identifications for the second data release",
            "condon2012_source_counts": "Resolving the Radio Source Background: Deeper Understanding Through Confusion",
        },
        "queries": {
            # --- the target concept ---
            "radio_shadow": 'all:"radio shadow"',
            "radio_void_star": 'all:"radio" AND all:void AND all:"nearby star"',
            "absence_radio_sources_near_stars": 'all:absence AND all:"radio sources" AND all:"nearby stars"',
            "deficit_background_sources_around_stars": 'all:deficit AND all:"background sources" AND all:stars AND all:radio',
            "source_counts_around_stars_radio": 'all:"source counts" AND all:radio AND all:"around stars"',
            "lotss_completeness_bright_sources": 'all:LoTSS AND all:completeness AND all:"bright sources"',
            "lotss_dr2_dynamic_range": 'all:LoTSS AND all:"dynamic range" AND all:artefacts',
            "radio_occultation_background_source_star": 'all:"radio" AND all:occultation AND all:"background source" AND all:star',
            "arcade_excess_foreground": 'all:ARCADE AND all:excess AND all:foreground',
            "radio_background_excess_origin": 'all:"radio background" AND all:excess AND all:origin',
            "technosignature_radio_absence": 'all:technosignature AND all:radio AND all:absence',
            # --- decoys, fetched on purpose ---
            "decoy_stellar_radio_emission_lotss": 'all:LoTSS AND all:"stellar radio emission"',
            "decoy_radio_emission_from_m_dwarfs": 'all:"radio emission" AND all:"M dwarfs" AND all:coherent',
            "decoy_radio_seti_nearby_stars": 'all:SETI AND all:radio AND all:"nearby stars" AND all:"Breakthrough Listen"',
        },
        "target": [
            r"radio (shadow|void|hole|deficit|gap)",
            r"(absence|deficit|lack|dearth|under-?density) of (background |extragalactic )?(radio )?sources",
            r"(fewer|deficit of|underdensity of) (background |radio )?sources (near|around|towards|toward|behind) (bright |nearby )?stars?",
            r"source[- ]count(s)?.{0,30}(deficit|suppression|depletion|hole|incompleteness)",
            r"(incomplete(ness)?|completeness).{0,80}(near|around|close to) (bright|nearby) (star|source)s?",
            r"dynamic[- ]range.{0,80}(bright (star|source)s?|incompleteness|missing sources)",
            r"ARCADE",
            r"occult\w*.{0,60}(background|extragalactic) (radio )?source",
        ],
        "decoys": {
            "radio_emission_from_stars": r"stellar radio emission|radio (emission|bursts?|flare|activity) (from|of) (the |a |an )?(star|M ?dwarf|dwarf|ultracool|brown dwarf)|electron[- ]cyclotron|maser|star[- ]planet interaction|radio[- ]loud (star|dwarf)|coronal",
            "stellar_radio_seti": r"SETI|technosignature|Breakthrough Listen|narrow-?band|artificial (signal|transmitter)",
            "cosmological_background_only": r"cosmic microwave background|CMB|reionization|21 ?cm|cosmolog",
        },
        "boosters": {
            "deep_survey_data": r"LoTSS|LOFAR|VLASS|FIRST|NVSS|MeerKAT|ASKAP|EMU|RACS|MIGHTEE|JVLA|VLA",
            "counts_or_completeness": r"source counts|completeness|detection fraction|number density",
        },
        "interpretation": (
            "A decoy-free hit is an abstract about a shortage of background radio sources "
            "near stars or a survey's incompleteness near bright sources.  "
            "radio_emission_from_stars and stellar_radio_seti are emission-side work on "
            "the same objects (the opposite sign); cosmological_background_only tags the "
            "ARCADE-2 excess literature that is about the isotropic background rather "
            "than a foreground hole.  ARCADE hits are the excess-as-foreground thread "
            "to read; LoTSS hits with counts_or_completeness are the completeness "
            "baseline the radio-hole test must be calibrated against."),
    },
    # ------------------------------------------------------------------
    "g5_concealment": {
        "question": (
            "Has anyone argued for, or searched for, ANISOTROPIC / DIRECTIONAL "
            "concealment of technosignatures -- shielding one's own or another "
            "civilisation's emission from particular observers?"),
        "by_id": {
            "wright2014_ghat_i": ("1408.1133", "Infrared Search for Extraterrestrial Civilizations", "high"),
            "cirkovic_bradbury2006_galactic_gradients": ("astro-ph/0506110", "Galactic gradients", "medium"),
        },
        "by_title": {
            "lacki2019_sunscreen": "Sunscreen: Photometric Signatures of Galaxies Partially Cloaked in Dyson Spheres",
            "wright2014_ghat_i": "Infrared Search for Extraterrestrial Civilizations with Large Energy Supplies. I. Background and Justifications",
            "cirkovic_bradbury2006_galactic_gradients": "Galactic Gradients, Postbiological Evolution and the Apparent Failure of SETI",
            "bradbury2011_dysonian_seti": "Dysonian Approach to SETI: A Fruitful Middle Ground?",
            "wright2018_prior_indigenous": "Prior Indigenous Technological Species",
            "lacki2016_log_seti": "Type III Societies (Apparently) Do Not Exist",
        },
        "queries": {
            # --- the target concept ---
            "concealment_technosignature": 'all:concealment AND all:technosignature',
            "conceal_extraterrestrial_civilization": 'all:conceal AND all:"extraterrestrial" AND all:civilization',
            "directional_shielding_seti": 'all:directional AND all:shielding AND all:SETI',
            "anisotropic_emission_technosignature": 'all:anisotropic AND all:technosignature',
            "hide_from_observers_seti": 'all:hide AND all:SETI AND all:civilizations',
            "sunscreen_dyson": 'all:sunscreen AND all:Dyson',
            "partial_dyson_sphere_directional": 'all:"partial Dyson" AND all:directional',
            "waste_heat_beamed_away": 'all:"waste heat" AND all:beamed AND all:SETI',
            "dysonian_seti_concealment": 'all:"Dysonian" AND all:SETI',
            "stealth_civilization_seti": 'all:stealth AND all:SETI AND all:civilization',
            "camouflage_technosignature": 'all:camouflage AND all:extraterrestrial',
            # --- decoys, fetched on purpose ---
            "decoy_fermi_paradox_review": 'all:"Fermi paradox" AND all:review AND all:civilizations',
            "decoy_fermi_paradox_explanations": 'all:"Fermi paradox" AND all:explanation AND all:silence',
        },
        "target": [
            r"conceal\w*",
            r"cloak\w*",
            r"hid(e|es|ing|den) (from|themselves|their (presence|emission|signature)s?)",
            r"shield\w*.{0,40}(from|against) (observ|detect|view)",
            r"sunscreen",
            r"(directional|anisotropic|beamed|collimated|one-?sided).{0,40}(emission|waste heat|radiation|technosignature|shield|screen)",
            r"(deliberate|intentional|active)(ly)? (avoid|suppress|hid|mask|obscur)\w*",
            r"camouflage|stealth",
        ],
        "decoys": {
            "fermi_review_generic": r"Fermi('s)? paradox|Great Silence|Great Filter|Drake equation",
            "non_seti_cloaking": r"metamaterial|invisibility|optical cloak|acoustic|plasmonic|transformation optics",
        },
        "boosters": {
            "seti_framing": r"technosignature|SETI|extraterrestrial|Dyson|megastructure|civili[sz]ation",
            "proposes_observable": r"observ(able|ational) (test|signature|consequence)|photometric signature|detect(able|ion)|search(ed|es|ing)? for|signature",
            "directional": r"directional|anisotrop|beam|line of sight|toward(s)? (the )?observer|from particular|from specific",
        },
        "interpretation": (
            "A decoy-free hit is an abstract about concealing or directionally shielding "
            "technosignatures.  non_seti_cloaking removes the metamaterial-cloak "
            "literature that 'cloak' always drags in; fermi_review_generic tags reviews "
            "that mention concealment as one of many resolutions (not a veto -- read the "
            "snippet).  A hit with seti_framing AND directional AND proposes_observable "
            "is the prior art to read.  The channel's specific claim -- screens on the "
            "Sun->observer lines in OUR outer solar system, seen from Earth by "
            "reciprocity as a deficit with annual parallax -- is what none of the "
            "phrasings above is expected to return; the record decides."),
    },
}


def _compile(spec: dict) -> dict:
    return {
        "target": [re.compile(p, re.I) for p in spec["target"]],
        "decoys": {k: re.compile(p, re.I) for k, p in spec["decoys"].items()},
        "boosters": {k: re.compile(p, re.I) for k, p in spec.get("boosters", {}).items()},
    }


COMPILED = {g: _compile(spec) for g, spec in GROUPS.items()}


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------
def arxiv_query(group: str, name: str, q: str) -> None:
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&start=0&max_results={MAX_RESULTS}&sortBy=relevance&sortOrder=descending")
    get(url, OUT / f"arxiv_q_{group}__{name}.atom")


def arxiv_title(group: str, name: str, title: str) -> None:
    q = 'ti:"' + title.replace('"', "") + '"'
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&start=0&max_results=5&sortBy=relevance&sortOrder=descending")
    get(url, OUT / f"arxiv_q_title_{group}__{name}.atom")


def arxiv_id(group: str, name: str, aid: str) -> None:
    get(f"{ARXIV_API}?search_query=&id_list={aid}&start=0&max_results=1",
        OUT / f"arxiv_id_{group}__{name}.atom")


def _entries(text: str):
    for m in re.finditer(r"<entry>(.*?)</entry>", text, re.S):
        e = m.group(1)
        aid = (re.search(r"<id>(.*?)</id>", e, re.S) or [None, ""])[1].strip()
        title = " ".join((re.search(r"<title>(.*?)</title>", e, re.S) or [None, ""])[1].split())
        summ = " ".join((re.search(r"<summary>(.*?)</summary>", e, re.S) or [None, ""])[1].split())
        yield aid, title, summ


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def id_title_check() -> dict:
    """Did each asserted id resolve to the expected title?  Did each title search
    return the expected title?  Recorded, not assumed; a mismatch is a finding."""
    out: dict = {"by_id": {}, "by_title": {}, "n_id_mismatch": 0, "n_title_search_miss": 0}
    for group, spec in GROUPS.items():
        for name, (aid, frag, conf) in spec["by_id"].items():
            key = f"{group}__{name}"
            p = OUT / f"arxiv_id_{key}.atom"
            rec = {"group": group, "id": aid, "expected_fragment": frag, "prior_confidence": conf}
            if not p.exists():
                rec.update({"fetched": False, "match": None})
            else:
                ents = list(_entries(p.read_text(errors="ignore")))
                title = ents[0][1] if ents else ""
                rec.update({"fetched": True, "n_entries": len(ents), "title_fetched": title,
                            "id_fetched": ents[0][0] if ents else "",
                            "match": _norm(frag) in _norm(title)})
                if not rec["match"]:
                    out["n_id_mismatch"] += 1
            out["by_id"][key] = rec
        for name, title in spec["by_title"].items():
            key = f"{group}__{name}"
            p = OUT / f"arxiv_q_title_{key}.atom"
            rec = {"group": group, "expected_title": title}
            if not p.exists():
                rec.update({"fetched": False, "found": None})
            else:
                ents = list(_entries(p.read_text(errors="ignore")))
                want = _norm(title)
                found = [(a, t) for a, t, _ in ents if want in _norm(t) or _norm(t) in want]
                rec.update({"fetched": True, "n_entries": len(ents),
                            "titles_returned": [t for _, t, _ in ents],
                            "found": bool(found),
                            "matched_ids": [a for a, _ in found]})
                if not found:
                    out["n_title_search_miss"] += 1
            out["by_title"][key] = rec
    out["note"] = (
        "by_id.match False means the asserted arXiv id does NOT carry the expected title: "
        "the id is wrong and must not be cited.  by_title.found False means no arXiv entry "
        "has that title (pre-arXiv papers such as Ball 1973 / Baxter 2001 are expected "
        "here); matched_ids are the ids the record itself supplies for the title.")
    (OUT / "id_title_check.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# Decoy-aware concept scan, per group, over EVERY fetched abstract (a paper
# fetched for one group can be prior art for another).
# --------------------------------------------------------------------------
def _atom_files(out_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(out_dir.glob("arxiv_*.atom"))


def _file_group(name: str) -> str | None:
    m = re.match(r"arxiv_(?:q_title|q|id)_(g\d_[a-z_]+?)__", name)
    return m.group(1) if m else None


def scan_text(group: str, title: str, summary: str) -> dict | None:
    """Pure function: scan one abstract for one group.  None if no target fired."""
    rx = COMPILED[group]
    blob = f"{title} {summary}"
    matches = []
    for r in rx["target"]:
        m = r.search(blob)
        if m:
            matches.append(m.group(0)[:80])
    if not matches:
        return None
    decoys = [k for k, r in rx["decoys"].items() if r.search(blob)]
    boosters = [k for k, r in rx["boosters"].items() if r.search(blob)]
    return {"target_matches": matches, "decoys": decoys, "boosters": boosters}


def scan(out_dir: pathlib.Path | None = None) -> dict:
    out_dir = out_dir or OUT
    files = _atom_files(out_dir)
    records: list[tuple[str, str, str, str]] = []  # (file, aid, title, summ)
    for p in files:
        for aid, title, summ in _entries(p.read_text(errors="ignore")):
            records.append((p.name, aid, title, summ))
    groups_out: dict = {}
    for group, spec in GROUPS.items():
        hits: dict[str, dict] = {}
        for fname, aid, title, summ in records:
            r = scan_text(group, title, summ)
            if r is None:
                continue
            key = aid or f"{fname}:{title[:40]}"
            if key in hits:
                if fname not in hits[key]["source_files"]:
                    hits[key]["source_files"].append(fname)
                continue
            hits[key] = {
                "arxiv": aid, "title": title,
                "target_matches": r["target_matches"], "decoys": r["decoys"],
                "boosters": r["boosters"], "decoy_free": not r["decoys"],
                "source_files": [fname],
                "fetched_for_this_group": _file_group(fname) == group,
                "abstract_verbatim": summ,
            }
        allh = list(hits.values())
        clean = [h for h in allh if h["decoy_free"]]
        strong = [h for h in allh if len(h["boosters"]) >= 2]
        groups_out[group] = {
            "question": spec["question"],
            "n_target_regex_hits": len(allh),
            "n_after_decoy_removal": len(clean),
            "n_with_two_or_more_boosters": len(strong),
            "decoy_free_hits": [{k: v for k, v in h.items() if k != "abstract_verbatim"} for h in clean],
            "all_hits": allh,
            "interpretation": spec["interpretation"],
        }
    out = {
        "n_atom_files": len(files),
        "n_abstracts_scanned": len(records),
        "n_unique_arxiv_ids": len({a for _, a, _, _ in records if a}),
        "per_group_counts": {g: {"hits": v["n_target_regex_hits"], "decoy_free": v["n_after_decoy_removal"],
                                 "two_plus_boosters": v["n_with_two_or_more_boosters"]}
                             for g, v in groups_out.items()},
        "groups": groups_out,
        "reading_guide": (
            "Every group's regexes were run over EVERY fetched abstract.  For each hit the "
            "record shows which target phrase fired (target_matches), which decoy concepts "
            "co-occur (decoys), which supporting concepts co-occur (boosters), and the "
            "verbatim abstract, so a human can see at a glance whether a hit is genuine "
            "prior art.  Decoy tags are flags, not vetoes.  The novelty position of the "
            "BAFFLE channel is to be read against each group's decoy_free_hits and "
            "booster-rich hits; it is not established until this file exists and has "
            "been read."),
    }
    (out_dir / "concept_scan.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    try:
        for group, spec in GROUPS.items():
            print(f"==== {group} ====")
            print("== named papers by id (title-checked) ==")
            for name, (aid, _, conf) in spec["by_id"].items():
                print(f"-- {name} ({aid}, prior confidence {conf})")
                arxiv_id(group, name, aid)
            print("== named papers by title ==")
            for name, title in spec["by_title"].items():
                print(f"-- {name}")
                arxiv_title(group, name, title)
            print("== keyword sweeps ==")
            for name, q in spec["queries"].items():
                print(f"-- {name}")
                arxiv_query(group, name, q)
    finally:
        # Whatever happened above, the scan and the checks run over what exists.
        print("== id/title check ==")
        chk = id_title_check()
        print(json.dumps({k: v for k, v in chk.items() if k in ("n_id_mismatch", "n_title_search_miss")}))
        for key, rec in chk["by_id"].items():
            print(f"  {key}: {rec.get('match')}  {rec.get('title_fetched', '')[:80]!r}")
        print("== decoy-aware concept scan ==")
        res = scan()
        print(json.dumps({"n_abstracts_scanned": res["n_abstracts_scanned"],
                          "per_group_counts": res["per_group_counts"]}, indent=2))
        _write_summary()
        print(f"\n{sum(1 for s in STATUS if s['ok'])}/{len(STATUS)} fetches ok -> {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ROMAN prior-art sweep: fetch the record on the GitHub runner.

The sandbox blocks arxiv.org and OpenAlex (``CONNECT tunnel failed, response
403``); the Actions runner has egress.  This script establishes, from the
record rather than from memory, whether anyone has already asked the four
questions the ROMAN channels ask (docs/roman.md §2, §4) -- and whether any
technosignature search has been proposed for Roman / WFIRST at all.

HONESTY RULES (the lit-sweep house pattern; scripts/bafflelit_fetch.py):

* every arXiv / OpenAlex response is saved VERBATIM; nothing is paraphrased;
* an arXiv id is NEVER asserted from memory without a title check --
  ``id_title_check.json`` records, per asserted id, whether the fetched title
  carries the expected fragment, and per title search whether the record
  returned that title at all.  A mismatch is a finding, not an error;
* every HTTP status of every attempt is recorded in ``fetch_log.json``;
* the decoy-aware concept scan runs every group's regexes over EVERY fetched
  abstract (a paper fetched for one group can be prior art for another), tags
  decoys and boosters, and quotes a verbatim ``evidence`` snippet per hit;
* the script runs to completion when every fetch fails -- each failure is
  recorded and the skeleton files are still written -- and exits 0;
* ``ROMANLIT_OFFLINE=1`` skips every network call (recorded as
  ``skipped: offline``) and still writes every output file, so the script is
  testable in the sandbox.

Five novelty questions, each its own query group with its own decoys:

  g1_opaque_lens   Microlensing by an OCCULTING / opaque lens -- the S40
                   signature is a symmetric pair of downward steps at
                   +/- u_c where the minor image falls inside the occulter.
                   Agol 2002 ("Occultation and microlensing") is the natural
                   reference; lensing BY Dyson spheres / megastructures and
                   finite-LENS-size / black-hole-shadow microlensing are the
                   nearest neighbours.  Decoys: planetary microlensing
                   (caustics), finite-SOURCE effects, transit surveys.
  g2_nir_laser     Near-infrared laser SETI: 1064 nm / 1550 nm technosignature
                   proposals, NIRSPEC laser-line searches, Euclid / Roman
                   grism SETI, "laser line" slitless spectroscopy.  Decoys:
                   adaptive-optics laser guide stars, Ti:sapphire
                   instrumentation, terrestrial lidar.
  g3_ramp_flash    Up-the-ramp jump detection, cosmic-ray rejection in
                   non-destructive reads, H4RG / H2RG snowballs, pulsed
                   nanosecond optical SETI (Howard 2004, Wright 2001, Maire
                   2019), wide-field sub-second transients.  Decoys: GRB
                   afterglows, FRB optical counterparts, satellite glints.
  g4_statite       Statites / solar-sail station-keeping (Forward 1993),
                   non-Keplerian astrometry of direct-imaging point sources,
                   artificial satellites of exoplanets / the Clarke exobelt
                   (Socas-Navarro 2018), Roman coronagraph technosignatures.
                   Decoys: hovering asteroid probes, our own solar-sail
                   missions.
  g5_roman_seti    Any technosignature / SETI proposal for the Roman Space
                   Telescope / WFIRST (GBTDS, HLTDS, HLWAS, CGI).  Decoys:
                   Roman exoplanet-demographics papers with no SETI content.

Outputs under ``results/romanlit/`` (see the README.md written there):
  arxiv_id_<group>__<name>.atom       arXiv API metadata for an asserted id
  arxiv_q_title_<group>__<name>.atom  arXiv title search for a named paper
  arxiv_q_<group>__<name>.atom        arXiv keyword sweep (target or decoy)
  oa_q_<group>__<name>.json           OpenAlex keyword search (when reachable)
  txt_<group>__<name>.txt             pdftotext full text of a verified anchor
  concept_scan.json                   the decoy-aware scan, per group
  id_title_check.json                 id/title verification record
  fetch_log.json                      every URL, every attempt, every status
  summary.json                        counts, groups, non-arXiv references
  README.md                           what every file is
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path(os.environ.get("ROMANLIT_OUT", "results/romanlit"))
OUT.mkdir(parents=True, exist_ok=True)

OFFLINE = os.environ.get("ROMANLIT_OFFLINE", "") not in ("", "0", "false", "no")
UA = {"User-Agent": "Seti-romanlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = float(os.environ.get("ROMANLIT_PAUSE", "3.0"))
TRIES = int(os.environ.get("ROMANLIT_TRIES", "3"))
ARXIV_API = "http://export.arxiv.org/api/query"
OA_API = "https://api.openalex.org/works"
MAX_RESULTS = 60
OA_PER_PAGE = 25
# OpenAlex is tried once at the start; if it does not answer, every OpenAlex
# fetch is recorded as skipped rather than retried (it is optional).
OA_REACHABLE: bool | None = None


def _write_fetch_log() -> None:
    """Written after EVERY fetch so a soft-deadline kill still leaves a record."""
    (OUT / "fetch_log.json").write_text(json.dumps(
        {"offline": OFFLINE, "n_urls": len(STATUS),
         "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"] and not s.get("skipped")),
         "n_skipped": sum(1 for s in STATUS if s.get("skipped")),
         "status": STATUS}, indent=2))


def get(url: str, dest: pathlib.Path, tries: int = TRIES, pause: float = PAUSE,
        skip_reason: str | None = None) -> bool:
    """Fetch one URL to ``dest`` verbatim; the HTTP status of every attempt is recorded."""
    if OFFLINE or skip_reason:
        reason = skip_reason or "offline"
        STATUS.append({"url": url, "dest": dest.name, "ok": False, "http_status": None,
                       "skipped": reason, "attempts": []})
        print(f"  skip ({reason})  {dest.name}")
        _write_fetch_log()
        return False
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
            _write_fetch_log()
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
    _write_fetch_log()
    return False


# --------------------------------------------------------------------------
# References the brief names that PREDATE arXiv or were never posted there.
# They cannot be title-verified through the arXiv API; they are recorded as
# bibliographic assertions FROM THE BRIEF, flagged as such, and a title search
# is still run so that any arXiv paper quoting the title is caught.
# --------------------------------------------------------------------------
NON_ARXIV_REFERENCES = [
    {"key": "forward1993_statite",
     "citation": "Forward, R. L. 1993, J. Spacecraft and Rockets 30, 217, "
                 "'Statite: a spacecraft that does not orbit'",
     "verification": "not possible via the arXiv API (engineering journal, pre-arXiv); "
                     "asserted by the brief; title-searched in g4"},
    {"key": "wright2001_optical_seti_detector",
     "citation": "Wright, S. A., et al. 2001, Proc. SPIE 4273, 173, "
                 "'Improved optical SETI detector'",
     "verification": "SPIE proceedings, not expected on arXiv; asserted by the brief; "
                     "title-searched in g3"},
    {"key": "agol2002_occultation_microlensing",
     "citation": "Agol, E. 2002, ApJ 579, 430, 'Occultation and microlensing'",
     "verification": "journal citation asserted by the brief; the arXiv posting is "
                     "id- and title-checked in g1"},
    {"key": "howard2004_nanosecond_pulses",
     "citation": "Howard, A. W., et al. 2004, ApJ 613, 1270, "
                 "'Search for nanosecond optical pulses from nearby solar-type stars'",
     "verification": "journal citation asserted by the brief; the arXiv posting is "
                     "id- and title-checked in g3"},
    {"key": "maire2019_nir_transients",
     "citation": "Maire, J., et al. 2019, AJ 158, 203, "
                 "'Search for nanosecond near-infrared transients around 1280 celestial objects'",
     "verification": "journal citation asserted by the brief; title-checked in g3 "
                     "(no id asserted)"},
    {"key": "socas_navarro2018_clarke_exobelt",
     "citation": "Socas-Navarro, H. 2018, ApJ 855, 110, "
                 "'Possible photometric signatures of moderately advanced civilizations: "
                 "the Clarke exobelt'",
     "verification": "journal citation asserted by the brief; the arXiv posting is "
                     "id- and title-checked in g4"},
]


# --------------------------------------------------------------------------
# The five query groups.  For each: named papers by id (title-checked), named
# papers by title (cannot be wrong about the id), arXiv keyword sweeps
# (target AND decoys, fetched on purpose so a null is interpretable), OpenAlex
# keyword searches, and the regexes of the decoy-aware scan.  ``confidence``
# on an id is the author's prior; the fetched title decides, and a mismatch
# is recorded in id_title_check.json.
# --------------------------------------------------------------------------
GROUPS: dict[str, dict] = {
    # ------------------------------------------------------------------
    "g1_opaque_lens": {
        "question": (
            "Has anyone searched microlensing light curves for a lens that is OPAQUE over "
            "a fraction of its Einstein radius -- an occulter that removes the minor image "
            "in the wings (symmetric steps at +/- u_c) -- or asked what density an observed "
            "lens's occulting radius implies?  Lensing BY Dyson spheres / megastructures "
            "and finite-lens-size / black-hole-shadow microlensing are the neighbours."),
        "by_id": {
            "agol2002_occultation_microlensing":
                ("astro-ph/0207228", "Occultation and microlensing", "medium"),
        },
        "by_title": {
            "agol2002_occultation_microlensing": "Occultation and microlensing",
            "bromley1996_finite_lens": "Finite size lensing effects",
            "wright2020_dyson_spheres_review": "Dyson Spheres",
            "penny2019_wfirst_microlensing_predictions":
                "Predictions of the WFIRST Microlensing Survey. I. Bound Planet Detection Rates",
            "gaudi2012_microlensing_review": "Microlensing Surveys for Exoplanets",
            "lacki2016_type_iii": "Type III Societies (Apparently) Do Not Exist",
        },
        "queries": {
            # --- the target concept ---
            "occultation_and_microlensing": 'all:occultation AND all:microlensing AND all:lens',
            "occulting_lens_microlensing": 'all:"occulting" AND all:microlensing',
            "opaque_lens_microlensing": 'all:opaque AND all:lens AND all:microlensing',
            "finite_lens_size_microlensing": 'all:"finite lens" AND all:microlensing',
            "lens_radius_einstein_radius_occult": 'all:"Einstein radius" AND all:occult AND all:lens',
            "eclipsing_microlensing_lens": 'all:microlensing AND all:eclipse AND all:"lens" AND all:"image"',
            "black_hole_shadow_microlensing": 'all:"black hole" AND all:shadow AND all:microlensing',
            "dyson_sphere_microlensing": 'all:"Dyson sphere" AND all:microlensing',
            "megastructure_gravitational_lens": 'all:megastructure AND all:"gravitational lens"',
            "dyson_sphere_lensing_signature": 'all:Dyson AND all:lensing AND all:technosignature',
            "self_lensing_opaque_companion": 'all:"self-lensing" AND all:opaque',
            "microlensing_technosignature": 'all:microlensing AND all:technosignature',
            "microlensing_seti": 'all:microlensing AND all:SETI',
            "artificial_lens_seti": 'all:artificial AND all:lens AND all:SETI',
            "extended_lens_microlensing_light_curve": 'all:"extended lens" AND all:microlensing',
            "microlensing_image_blocked": 'all:microlensing AND all:"image" AND all:"blocked"',
            # --- decoys, fetched on purpose ---
            "decoy_planetary_microlensing_caustic": 'all:planetary AND all:microlensing AND all:caustic',
            "decoy_finite_source_microlensing": 'all:"finite source" AND all:microlensing',
            "decoy_transit_survey_wfirst": 'all:transit AND all:survey AND all:WFIRST',
            "decoy_binary_lens_anomaly": 'all:"binary lens" AND all:anomaly AND all:microlensing',
        },
        "openalex": {
            "occultation_microlensing": "occultation and microlensing opaque lens",
            "dyson_sphere_microlensing": "Dyson sphere gravitational microlensing technosignature",
            "finite_lens_size": "finite lens size microlensing occulting lens image",
        },
        "target": [
            r"occult\w*.{0,80}(microlens|lens(ed|ing)? image|gravitational lens)",
            r"(microlens|gravitational lens)\w*.{0,80}occult\w*",
            r"opaque (lens|body|object|screen|disk|sphere)",
            r"(lens|deflector).{0,40}(opaque|blocks?|obscur\w*|eclips\w*).{0,40}image",
            r"finite[- ](size|radius|extent) (of the )?lens",
            r"(lens|deflector) (radius|size).{0,40}Einstein radius",
            r"(image|images).{0,40}(blocked|hidden|obscured|eclipsed|occulted) by the lens",
            r"(black hole )?shadow.{0,60}microlens",
            r"Dyson (sphere|swarm|shell)s?.{0,80}(microlens|gravitational lens|lensing)",
            r"(megastructure|artificial).{0,60}(microlens|gravitational lens|lensing)",
            r"(microlens|lensing).{0,60}(megastructure|technosignature|SETI|extraterrestrial)",
        ],
        "decoys": {
            "planetary_caustic": r"caustic|planetary (microlensing|companion|anomaly|perturbation)|mass ratio q\b|planet[- ]host",
            "finite_source": r"finite[- ]source|source (radius|size).{0,20}(effect|crossing)|limb[- ]darken",
            "transit_survey": r"transit(ing|s)? (survey|planet|method|photometry)|transit light ?curve",
            "cosmological_strong_lensing": r"galaxy[- ]cluster|strong lensing|weak lensing|Einstein ring|quasar lens|time[- ]delay cosmograph",
        },
        "boosters": {
            "microlensing_data": r"OGLE|MOA|KMTNet|WFIRST|Roman|Galactic bulge|light ?curve",
            "seti_framing": r"technosignature|SETI|extraterrestrial|artificial|megastructure|Dyson",
            "density_or_compactness": r"density|compact(ness)?|physical (radius|size)|Schwarzschild",
        },
        "interpretation": (
            "A decoy-free hit is an abstract about a lens that OCCULTS its own images, an "
            "opaque / finite-size lens, or lensing by a megastructure.  planetary_caustic "
            "and finite_source tag the ordinary anomaly literature the target regexes brush "
            "against; cosmological_strong_lensing removes galaxy-scale lensing.  Agol 2002 "
            "is expected here as the natural-body treatment; the question is whether any "
            "hit SEARCHES a survey for the occulting signature or infers a lens density "
            "from it.  A hit with seti_framing AND microlensing_data is the prior art to "
            "read.  Empty decoy-free lists are evidence, not proof -- docs/roman.md §4 "
            "states the position as 'to be verified'."),
    },
    # ------------------------------------------------------------------
    "g2_nir_laser": {
        "question": (
            "Has anyone searched, or proposed to search, for laser lines in the "
            "NEAR-INFRARED (1.0-1.9 um; 1064 nm Nd:YAG, 1550 nm Er-fibre) on stellar "
            "point sources -- and in particular with wide-field SLITLESS spectroscopy "
            "(Euclid NISP, Roman G150/P127)?"),
        "by_id": {
            "tellis_marcy2017_megawatt": ("1704.02535", "Laser Emission", "high"),
            "tellis_marcy2015_keck": ("1504.03369", "Optical Laser Emission", "medium"),
        },
        "by_title": {
            "tellis_marcy2017_megawatt":
                "A Search for Laser Emission with Megawatt Thresholds from 5600 FGKM Stars",
            "tellis_marcy2015_keck": "A Search for Optical Laser Emission Using Keck HIRES",
            "marcy2021_lris_proxima": "A search for optical laser emission from Proxima Centauri",
            "schwartz_townes1961_optical_masers":
                "Interstellar and Interplanetary Communication by Optical Masers",
            "maire2019_nir_transients":
                "Search for Nanosecond Near-infrared Transients around 1280 Celestial Objects",
            "wright2014_nirospe_infrared_seti": "Near-infrared SETI",
            "zuckerman2023_laser_dyson": "Laser Communication with Dyson Sphere Candidates",
            "euclid2024_nisp_grism_overview": "Euclid. I. Overview of the Euclid mission",
        },
        "queries": {
            # --- the target concept ---
            "near_infrared_laser_seti": 'all:"near-infrared" AND all:laser AND all:SETI',
            "infrared_laser_technosignature": 'all:infrared AND all:laser AND all:technosignature',
            "laser_1064_nm_seti": 'all:1064 AND all:laser AND all:SETI',
            "laser_1550_nm_seti": 'all:1550 AND all:laser AND all:extraterrestrial',
            "nd_yag_seti": 'all:"Nd:YAG" AND all:SETI',
            "laser_line_nirspec": 'all:laser AND all:NIRSPEC AND all:search',
            "laser_line_slitless_spectroscopy": 'all:"laser" AND all:slitless AND all:spectroscopy',
            "grism_seti": 'all:grism AND all:SETI',
            "grism_technosignature": 'all:grism AND all:technosignature',
            "euclid_seti": 'all:Euclid AND all:SETI',
            "euclid_technosignature": 'all:Euclid AND all:technosignature',
            "roman_grism_laser": 'all:Roman AND all:grism AND all:laser',
            "wfirst_seti_laser": 'all:WFIRST AND all:laser AND all:SETI',
            "infrared_optical_seti_pulsed_continuous": 'all:"infrared" AND all:"optical SETI"',
            "unresolved_emission_line_star_artificial": 'all:"unresolved emission line" AND all:star AND all:artificial',
            "monochromatic_emission_stars_survey": 'all:monochromatic AND all:emission AND all:stars AND all:survey AND all:artificial',
            "laser_emission_search_stars": 'all:"laser emission" AND all:search AND all:stars',
            # --- decoys, fetched on purpose ---
            "decoy_laser_guide_star_ao": 'all:"laser guide star" AND all:"adaptive optics"',
            "decoy_ti_sapphire_instrument": 'all:"Ti:sapphire" AND all:laser AND all:instrument',
            "decoy_lidar_atmospheric": 'all:lidar AND all:atmospheric AND all:laser',
            "decoy_laser_frequency_comb_calibration": 'all:"laser frequency comb" AND all:calibration AND all:spectrograph',
        },
        "openalex": {
            "nir_laser_seti": "near-infrared laser SETI technosignature 1064 nm",
            "slitless_laser_line": "slitless spectroscopy laser line technosignature grism",
            "euclid_roman_seti": "Euclid Roman grism SETI technosignature",
        },
        "target": [
            r"(near[- ]?infrared|NIR|infrared|IR).{0,60}laser",
            r"laser.{0,60}(near[- ]?infrared|NIR|infrared)",
            r"\b10[0-9]{2} ?nm\b.{0,80}(laser|SETI|technosignature)",
            r"\b15[0-9]{2} ?nm\b.{0,80}(laser|SETI|technosignature)",
            r"1\.0[0-9]{1,2} ?(um|μm|micron).{0,60}laser|laser.{0,60}1\.0[0-9]{1,2} ?(um|μm|micron)",
            r"1\.5[0-9]{1,2} ?(um|μm|micron).{0,60}laser|laser.{0,60}1\.5[0-9]{1,2} ?(um|μm|micron)",
            r"Nd:?YAG|(Er|Yb)[- ]?(doped )?fib(er|re)",
            r"(slitless|grism|prism).{0,80}(laser|SETI|technosignature|artificial (line|emission))",
            r"(laser|SETI|technosignature).{0,80}(slitless|grism|NISP|G150|P127)",
            r"(Euclid|Roman|WFIRST|NIRSPEC|NIRSpec).{0,80}(laser|SETI|technosignature)",
            r"(laser|SETI|technosignature).{0,80}(Euclid|Roman Space Telescope|WFIRST|NIRSPEC|NIRSpec)",
            r"(optical|infrared) SETI",
            r"laser (line|emission|signal|beacon|communication)s?.{0,80}(star|stellar|extraterrestrial|civili[sz]ation)",
        ],
        "decoys": {
            "laser_guide_star": r"laser guide ?star|LGS\b|sodium (laser|layer)|adaptive optics|wavefront",
            "ti_sapphire_instrument": r"Ti:?sapphire|femtosecond|mode-?locked|frequency comb|oscillator|amplifier|laser (system|source|cavity) (design|development)",
            "terrestrial_lidar": r"lidar|LIDAR|ranging|remote sensing|atmospheric (aerosol|boundary)|satellite laser",
            "laboratory_or_engineering": r"laboratory|fabricat|photonic (chip|integrated)|waveguide|free-?space optical communication (link|terminal)",
        },
        "boosters": {
            "seti_framing": r"technosignature|SETI|extraterrestrial|artificial|civili[sz]ation|beacon",
            "nir_band": r"near[- ]?infrared|NIR|1\.0[0-9]|1\.[3-6][0-9]|10[0-9]{2} ?nm|15[0-9]{2} ?nm|H[- ]band|J[- ]band|Y[- ]band",
            "wide_field_or_survey": r"wide[- ]field|survey|slitless|grism|prism|thousands of|10\^[5-9]|all[- ]sky",
        },
        "interpretation": (
            "A decoy-free hit is an abstract about a laser search / proposal in the "
            "near-infrared or in slitless data.  laser_guide_star, ti_sapphire_instrument "
            "and terrestrial_lidar tag our own lasers (the target regexes cannot avoid them); "
            "laboratory_or_engineering tags photonics.  Executed NIR laser searches on "
            "small targeted samples (Maire 2019, NIROSETI) are EXPECTED and are not the "
            "S41 claim; a hit with seti_framing AND nir_band AND wide_field_or_survey is "
            "the prior art to read.  Tellis & Marcy stop near 0.98 um; the record decides "
            "whether any wide-field NIR slitless search has been proposed."),
    },
    # ------------------------------------------------------------------
    "g3_ramp_flash": {
        "question": (
            "Has anyone used the up-the-ramp jump (cosmic-ray) flags of a non-destructive-"
            "read detector as a detector of ASTROPHYSICAL sub-exposure flashes -- or "
            "asked the pulsed-optical-SETI question of a wide-field NIR survey?  The "
            "engineering neighbours are jump detection, snowballs and cosmic-ray "
            "rejection in H4RG/H2RG ramps; the science neighbours are nanosecond pulsed "
            "SETI and wide-field sub-second transients."),
        "by_id": {
            "howard2004_nanosecond_pulses":
                ("astro-ph/0311347", "nanosecond optical pulses", "medium"),
        },
        "by_title": {
            "howard2004_nanosecond_pulses":
                "Search for Nanosecond Optical Pulses from Nearby Solar-Type Stars",
            "wright2001_improved_optical_seti_detector": "Improved optical SETI detector",
            "maire2019_nir_transients":
                "Search for Nanosecond Near-infrared Transients around 1280 Celestial Objects",
            "hanna2009_veritas_optical_seti": "OSETI with STACEE",
            "anderson2011_jump_detection_nondestructive":
                "Optimal Cosmic-Ray Detection for Nondestructive Read Ramps",
            "regan2019_snowballs_h2rg": "Snowballs in the JWST NIRCam and NIRSpec detectors",
            "fixsen2000_cosmic_ray_rejection_up_the_ramp":
                "Cosmic-Ray Rejection and Readout Efficiency for Large-Area Arrays",
            "richards2021_deep_wide_fast_transients": "The Deep, Wide, Fast",
        },
        "queries": {
            # --- the target concept ---
            "up_the_ramp_jump_detection": 'all:"up-the-ramp" AND all:jump AND all:detection',
            "cosmic_ray_rejection_non_destructive_reads": 'all:"cosmic ray" AND all:"non-destructive" AND all:reads',
            "snowballs_h2rg": 'all:snowball AND all:H2RG',
            "snowballs_h4rg": 'all:snowball AND all:H4RG',
            "jump_detection_astrophysical_transient_ramp": 'all:"jump" AND all:ramp AND all:transient AND all:detector',
            "cosmic_ray_flags_transients": 'all:"cosmic ray" AND all:flags AND all:transients AND all:infrared',
            "ramp_fitting_roman_wfi": 'all:"ramp fitting" AND all:Roman AND all:WFI',
            "sub_exposure_transient_infrared_detector": 'all:"within an exposure" AND all:transient AND all:infrared',
            "optical_seti_pulsed_nanosecond": 'all:"optical SETI" AND all:nanosecond',
            "pulsed_laser_seti_flash": 'all:pulsed AND all:laser AND all:SETI AND all:flash',
            "nanosecond_pulses_stars_search": 'all:nanosecond AND all:pulses AND all:stars AND all:search',
            "optical_seti_wide_field": 'all:"optical SETI" AND all:"wide field"',
            "panoseti": 'all:PANOSETI',
            "sub_second_transients_wide_field": 'all:"sub-second" AND all:transients AND all:"wide-field"',
            "millisecond_optical_transients_survey": 'all:millisecond AND all:optical AND all:transients AND all:survey',
            "fast_optical_transients_stars_flash": 'all:"fast optical transients" AND all:stars AND all:flash',
            "technosignature_flash_star_infrared": 'all:technosignature AND all:flash AND all:infrared',
            "roman_transients_short_timescale": 'all:Roman AND all:transients AND all:"short timescale"',
            # --- decoys, fetched on purpose ---
            "decoy_grb_afterglow_optical": 'all:"gamma-ray burst" AND all:afterglow AND all:optical AND all:early',
            "decoy_frb_optical_counterpart": 'all:"fast radio burst" AND all:optical AND all:counterpart',
            "decoy_satellite_glints_transients": 'all:satellite AND all:glints AND all:transients',
        },
        "openalex": {
            "ramp_jump_transient": "up-the-ramp jump detection astrophysical transient non-destructive reads",
            "pulsed_optical_seti": "pulsed optical SETI nanosecond flash search stars",
            "snowballs_h4rg": "snowballs H4RG H2RG cosmic ray infrared detector",
        },
        "target": [
            r"up[- ]the[- ]ramp",
            r"non[- ]?destructive(ly)? read",
            r"jump (detection|step|flag)s?",
            r"snowball",
            r"cosmic[- ]ray (rejection|detection|hit|flag)s?.{0,80}(ramp|read|infrared|HgCdTe|H[24]RG)",
            r"(ramp|HgCdTe|H[24]RG|NIRCam|NIRSpec|WFI).{0,80}cosmic[- ]ray",
            r"(nanosecond|sub-?second|millisecond|short[- ]duration|brief) (optical |infrared |near-?infrared )?(pulse|flash|transient)s?",
            r"pulsed (laser|optical|beacon)",
            r"optical SETI|OSETI|PANOSETI|NIROSETI",
            r"(flash|pulse)\w*.{0,80}(technosignature|SETI|extraterrestrial|artificial|beacon)",
            r"(technosignature|SETI|extraterrestrial|beacon).{0,80}(flash|pulse)\w*",
        ],
        "decoys": {
            "grb_afterglow": r"gamma[- ]ray burst|GRB\b|afterglow|kilonova|supernova shock breakout",
            "frb_optical_counterpart": r"fast radio burst|FRB\b|magnetar|pulsar",
            "satellite_glint": r"satellite (glint|flare|streak|trail)|space debris|Starlink|geosynchronous|artificial satellite glint",
            "detector_engineering_only": r"dark current|persistence|read ?noise|linearity|quantum efficiency|interpixel|reference pixel|conversion gain",
        },
        "boosters": {
            "ramp_or_detector": r"ramp|non[- ]?destructive|HgCdTe|H[24]RG|resultant|MULTIACCUM|multi-?accum|group",
            "seti_framing": r"technosignature|SETI|extraterrestrial|artificial|beacon|civili[sz]ation",
            "wide_field_or_survey": r"wide[- ]field|survey|10\^[5-9]|thousands of|all[- ]sky|Roman|WFIRST|Euclid|Rubin|LSST|ZTF",
        },
        "interpretation": (
            "The scan runs two lineages at once.  Engineering hits (up-the-ramp, jump "
            "detection, snowballs) with detector_engineering_only are the baseline the S42 "
            "morphology cuts are calibrated against, not prior art; the S42 claim is a hit "
            "with ramp_or_detector AND (seti_framing OR an astrophysical flash).  Science "
            "hits (nanosecond pulsed SETI: Howard 2004, Wright 2001, Maire 2019, PANOSETI) "
            "are the targeted-photometer lineage and are EXPECTED; a hit with seti_framing "
            "AND wide_field_or_survey AND ramp_or_detector is the prior art to read.  "
            "grb_afterglow, frb_optical_counterpart and satellite_glint tag the natural / "
            "artificial-but-ours fast-transient literature."),
    },
    # ------------------------------------------------------------------
    "g4_statite": {
        "question": (
            "Has anyone proposed NON-KEPLERIAN ASTROMETRY of a direct-imaging point "
            "source -- a reflector that does not move on an orbit -- as a technosignature "
            "test, or searched coronagraphic images for statites / station-kept "
            "structures / artificial satellites?  Forward 1993 (statite), Socas-Navarro "
            "2018 (Clarke exobelt) and Roman CGI technosignature proposals are the "
            "neighbours."),
        "by_id": {
            "socas_navarro2018_clarke_exobelt":
                ("1802.07723", "Clarke exobelt", "medium"),
        },
        "by_title": {
            "forward1993_statite": "Statite: a spacecraft that does not orbit",
            "socas_navarro2018_clarke_exobelt":
                "Possible photometric signatures of moderately advanced civilizations: the Clarke exobelt",
            "mcinnes_solar_sail_non_keplerian": "Solar sail non-Keplerian orbits",
            "korpela2015_clarke_belt_transits": "Modeling indications of technology in planetary transit light curves",
            "arnold2005_artificial_transits": "Transit lightcurve signatures of artificial objects",
            "wright2016_ghat_iv_transits": "The G Search for Extraterrestrial Civilizations with Large Energy Supplies. IV. The Signatures and Information Content of Transiting Megastructures",
            "kasdin2020_roman_cgi": "The Nancy Grace Roman Space Telescope Coronagraph Instrument",
            "bailey2023_roman_cgi_science": "Roman Coronagraph Instrument science",
        },
        "queries": {
            # --- the target concept ---
            "statite": 'all:statite',
            "solar_sail_station_keeping_star": 'all:"solar sail" AND all:"station-keeping" AND all:star',
            "non_keplerian_orbit_solar_sail": 'all:"non-Keplerian" AND all:"solar sail"',
            "non_keplerian_astrometry_direct_imaging": 'all:"non-Keplerian" AND all:astrometry AND all:"direct imaging"',
            "non_keplerian_motion_technosignature": 'all:"non-Keplerian" AND all:technosignature',
            "artificial_satellite_exoplanet": 'all:"artificial satellite" AND all:exoplanet',
            "clarke_exobelt": 'all:"Clarke exobelt"',
            "clarke_belt_technosignature": 'all:"Clarke" AND all:belt AND all:technosignature',
            "megastructure_direct_imaging": 'all:megastructure AND all:"direct imaging"',
            "technosignature_coronagraph": 'all:technosignature AND all:coronagraph',
            "roman_coronagraph_technosignature": 'all:Roman AND all:coronagraph AND all:technosignature',
            "coronagraph_seti_artificial": 'all:coronagraph AND all:SETI AND all:artificial',
            "reflected_light_technosignature_direct_imaging": 'all:"reflected light" AND all:technosignature AND all:imaging',
            "hovering_structure_radiation_pressure_star": 'all:hovering AND all:"radiation pressure" AND all:star AND all:sail',
            "stationary_point_source_near_star_artificial": 'all:stationary AND all:"point source" AND all:star AND all:artificial',
            "light_sail_technosignature": 'all:"light sail" AND all:technosignature',
            "specular_reflection_technosignature": 'all:specular AND all:technosignature',
            # --- decoys, fetched on purpose ---
            "decoy_asteroid_hovering_probe": 'all:asteroid AND all:hovering AND all:spacecraft',
            "decoy_solar_sail_mission_our_own": 'all:"solar sail" AND all:mission AND all:trajectory AND all:spacecraft',
            "decoy_lightsail_ikaros": 'all:IKAROS AND all:"solar sail"',
        },
        "openalex": {
            "statite_technosignature": "statite non-Keplerian technosignature direct imaging",
            "clarke_exobelt": "Clarke exobelt artificial satellites exoplanet",
            "roman_coronagraph_technosignature": "Roman coronagraph technosignature megastructure direct imaging",
        },
        "target": [
            r"statite",
            r"non[- ]Keplerian.{0,80}(astrometr|orbit|motion|trajector)",
            r"(does|do) not (orbit|obey Kepler)",
            r"station[- ]?keep\w*.{0,80}(star|stellar|sail|sun)",
            r"(hover|hovering|levitat\w*|suspended).{0,80}(radiation pressure|solar sail|light sail|star)",
            r"(radiation pressure|solar sail|light sail).{0,80}(hover|hovering|levitat\w*|balanc\w*|station)",
            r"artificial satellite\w*.{0,80}(exoplanet|planet|extrasolar)",
            r"Clarke (exo)?belt",
            r"(megastructure|technosignature|artificial|SETI|extraterrestrial).{0,80}(coronagraph|direct imag|reflected light|high[- ]contrast)",
            r"(coronagraph|direct imag|high[- ]contrast).{0,80}(megastructure|technosignature|artificial (object|structure|satellite)|SETI)",
            r"(specular|mirror[- ]like|grey|gray) reflect\w*.{0,80}(artificial|technosignature|megastructure|sail)",
        ],
        "decoys": {
            "asteroid_probe_hovering": r"asteroid|comet|small body|Hayabusa|OSIRIS|Rosetta|hovering (above|over) (the )?(surface|asteroid)",
            "own_sail_mission": r"IKAROS|LightSail|NEA Scout|Solar Cruiser|Breakthrough Starshot|mission (design|analysis|concept)|trajectory (design|optimi[sz]ation)|Lagrange|halo orbit|sun-?synchronous",
            "transit_photometry_only": r"transit (light ?curve|photometry|depth|signature)s?",
            "planet_or_disc_astrophysics_only": r"debris dis[ck]|protoplanetary|orbital fit|orbit (retrieval|fitting)|Keplerian (fit|orbit)s? of",
        },
        "boosters": {
            "seti_framing": r"technosignature|SETI|extraterrestrial|artificial|megastructure|civili[sz]ation",
            "imaging_data": r"coronagraph|direct imag|high[- ]contrast|CGI\b|Roman|WFIRST|GPI|SPHERE|JWST",
            "astrometry_or_motion": r"astrometr|proper motion|orbital motion|Keplerian|position",
        },
        "interpretation": (
            "A decoy-free hit is an abstract about a statite / station-kept / hovering "
            "reflector, non-Keplerian astrometry as a test, artificial satellites of "
            "exoplanets, or technosignatures in coronagraphic images.  own_sail_mission and "
            "asteroid_probe_hovering tag OUR spacecraft engineering (Forward 1993 belongs "
            "here and is the concept source, not prior art for a search); "
            "transit_photometry_only tags Socas-Navarro / Korpela / Arnold, which use "
            "transits, not imaging astrometry.  A hit with seti_framing AND imaging_data "
            "AND astrometry_or_motion is the prior art to read."),
    },
    # ------------------------------------------------------------------
    "g5_roman_seti": {
        "question": (
            "Has ANY technosignature / SETI search or proposal been made for the Nancy "
            "Grace Roman Space Telescope / WFIRST -- its GBTDS, HLTDS, HLWAS, grism / prism "
            "or CGI products -- of any kind?"),
        "by_id": {},
        "by_title": {
            "spergel2015_wfirst_afta_report":
                "Wide-Field InfrarRed Survey Telescope-Astrophysics Focused Telescope Assets WFIRST-AFTA 2015 Report",
            "akeson2019_wfirst_100_hubbles":
                "The Wide Field Infrared Survey Telescope: 100 Hubbles for the 2020s",
            "wright2022_seti_roadmap_technosignatures":
                "The Case for Technosignatures: Why Should We Bother?",
            "sheikh2020_nine_axes": "Nine Axes of Merit for Technosignature Searches",
            "lacki2020_technosignature_surveys_breakthrough":
                "One of Everything: The Breakthrough Listen Exotica Catalog",
            "socas_navarro2021_technosignature_review":
                "Concepts for future missions to search for technosignatures",
            "penny2019_wfirst_microlensing_predictions":
                "Predictions of the WFIRST Microlensing Survey. I. Bound Planet Detection Rates",
            "johnson2020_wfirst_free_floating":
                "Predictions of the Nancy Grace Roman Space Telescope Galactic Exoplanet Survey. II. Free-floating Planet Detection Rates",
        },
        "queries": {
            # --- the target concept ---
            "roman_space_telescope_technosignature": 'all:"Roman Space Telescope" AND all:technosignature',
            "roman_space_telescope_seti": 'all:"Roman Space Telescope" AND all:SETI',
            "roman_technosignature": 'all:Roman AND all:technosignature',
            "wfirst_technosignature": 'all:WFIRST AND all:technosignature',
            "wfirst_seti": 'all:WFIRST AND all:SETI',
            "wfirst_extraterrestrial": 'all:WFIRST AND all:extraterrestrial AND all:civilization',
            "roman_extraterrestrial_intelligence": 'all:"Roman" AND all:"extraterrestrial intelligence"',
            "gbtds_technosignature": 'all:"Galactic Bulge Time Domain" AND all:technosignature',
            "bulge_time_domain_seti": 'all:"bulge" AND all:"time domain" AND all:SETI',
            "hltds_technosignature": 'all:"High Latitude Time Domain" AND all:technosignature',
            "hlwas_technosignature": 'all:"High Latitude Wide Area" AND all:technosignature',
            "roman_grism_technosignature": 'all:Roman AND all:grism AND all:technosignature',
            "roman_coronagraph_seti": 'all:Roman AND all:coronagraph AND all:SETI',
            "roman_megastructure": 'all:"Roman Space Telescope" AND all:megastructure',
            "roman_dyson": 'all:"Roman Space Telescope" AND all:Dyson',
            "wfirst_megastructure": 'all:WFIRST AND all:megastructure',
            "roman_artificial_transit": 'all:Roman AND all:artificial AND all:transit AND all:technosignature',
            "roman_microlensing_seti": 'all:Roman AND all:microlensing AND all:SETI',
            "technosignature_space_telescope_survey_2020s":
                'all:technosignature AND all:"space telescope" AND all:survey AND all:infrared',
            # --- decoys, fetched on purpose ---
            "decoy_roman_exoplanet_demographics": 'all:"Roman Space Telescope" AND all:exoplanet AND all:demographics',
            "decoy_wfirst_microlensing_yield": 'all:WFIRST AND all:microlensing AND all:yield',
            "decoy_roman_supernova_cosmology": 'all:"Roman Space Telescope" AND all:supernova AND all:cosmology',
        },
        "openalex": {
            "roman_technosignature": "Nancy Grace Roman Space Telescope technosignature SETI",
            "wfirst_seti": "WFIRST SETI extraterrestrial intelligence search",
            "roman_megastructure_dyson": "Roman Space Telescope megastructure Dyson sphere search",
        },
        "target": [
            r"(Roman Space Telescope|Nancy Grace Roman|\bRoman\b|WFIRST|Wide[- ]Field Infrared Survey Telescope).{0,200}(technosignature|SETI|extraterrestrial (intelligence|civili[sz]ation)|megastructure|Dyson|artificial (object|structure|signal|transit))",
            r"(technosignature|SETI|extraterrestrial (intelligence|civili[sz]ation)|megastructure|Dyson|artificial (object|structure|signal|transit)).{0,200}(Roman Space Telescope|Nancy Grace Roman|\bRoman\b|WFIRST|Wide[- ]Field Infrared Survey Telescope)",
            r"(GBTDS|Galactic Bulge Time Domain|HLTDS|High Latitude Time Domain|HLWAS|High Latitude Wide Area|Coronagraph Instrument|\bCGI\b).{0,200}(technosignature|SETI|extraterrestrial|megastructure|Dyson|artificial)",
        ],
        "decoys": {
            "roman_demographics_no_seti": r"(planet|exoplanet) (demographics|occurrence|yield|detection rate|population)s?|bound planet|free[- ]floating planet|mass function",
            "roman_cosmology": r"dark energy|weak lensing|baryon acoustic|supernova cosmology|galaxy (clustering|survey)|cosmological",
            "roman_not_the_telescope": r"Roman (Empire|period|era|numeral|law|Catholic|coin|villa|amphora|road|army|Britain|Republic)|ancient Rome|Romanesque|Romania|Romanov",
            "wide_field_generic_not_roman": r"Euclid|Rubin|LSST|PLATO|TESS|Kepler|Gaia|JWST|Hubble",
        },
        "boosters": {
            "seti_framing": r"technosignature|SETI|extraterrestrial|artificial|megastructure|Dyson|civili[sz]ation",
            "roman_survey_named": r"GBTDS|Galactic Bulge Time Domain|HLTDS|High Latitude Time Domain|HLWAS|High Latitude Wide Area|Coronagraph Instrument|\bCGI\b|Wide Field Instrument|\bWFI\b|grism|prism",
            "proposes_search": r"propos\w*|search(es|ed|ing)? (for|of)|we (search|scan|examine|analy[sz]e)|survey(s|ed|ing)? for|could detect|sensitiv",
        },
        "interpretation": (
            "This group is the sanity check on the whole channel: a decoy-free hit is any "
            "abstract that pairs Roman / WFIRST with a technosignature concept.  "
            "roman_not_the_telescope removes the classics literature 'Roman' drags in; "
            "roman_demographics_no_seti and roman_cosmology tag the mission's own science "
            "papers when the target regex fires on an incidental word; "
            "wide_field_generic_not_roman flags abstracts that name several facilities "
            "(Roman may be one of a list).  A hit with seti_framing AND roman_survey_named "
            "AND proposes_search is the prior art to read.  Roman SETI white papers that "
            "propose transit / Dyson / microlensing-anomaly searches are EXPECTED; the S40-"
            "S43 claims are the specific signatures, judged in g1-g4."),
    },
}

# Anchors whose full text is pulled ONLY after the id/title check passed.
FULLTEXT_ANCHORS = {
    "g1_opaque_lens": ["agol2002_occultation_microlensing"],
    "g3_ramp_flash": ["howard2004_nanosecond_pulses"],
    "g4_statite": ["socas_navarro2018_clarke_exobelt"],
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


def _openalex_probe() -> bool:
    """One cheap request decides whether OpenAlex is reachable from this runner."""
    global OA_REACHABLE
    if OA_REACHABLE is not None:
        return OA_REACHABLE
    ok = get(f"{OA_API}?search=technosignature&per-page=1", OUT / "oa_probe.json",
             tries=1, pause=1.0)
    OA_REACHABLE = ok
    return ok


def openalex_query(group: str, name: str, q: str) -> None:
    dest = OUT / f"oa_q_{group}__{name}.json"
    url = (f"{OA_API}?search={urllib.parse.quote(q)}&per-page={OA_PER_PAGE}"
           f"&select=id,doi,title,publication_year,cited_by_count,abstract_inverted_index,ids")
    if not _openalex_probe():
        get(url, dest, skip_reason="openalex unreachable")
        return
    get(url, dest, pause=1.0)


def fulltext(group: str, name: str, aid: str) -> None:
    pdf = OUT / f"pdf_{group}__{name}.pdf"
    if get(f"https://arxiv.org/pdf/{aid}", pdf):
        try:
            subprocess.run(["pdftotext", str(pdf), str(OUT / f"txt_{group}__{name}.txt")],
                           check=False, timeout=180)
        except Exception as exc:  # noqa: BLE001
            print(f"  pdftotext failed for {name}: {exc!r}")
        finally:
            pdf.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Parsers (arXiv atom, OpenAlex JSON) -> (source, id, title, abstract)
# --------------------------------------------------------------------------
def _entries(text: str):
    for m in re.finditer(r"<entry>(.*?)</entry>", text, re.S):
        e = m.group(1)
        aid = (re.search(r"<id>(.*?)</id>", e, re.S) or [None, ""])[1].strip()
        title = " ".join((re.search(r"<title>(.*?)</title>", e, re.S) or [None, ""])[1].split())
        summ = " ".join((re.search(r"<summary>(.*?)</summary>", e, re.S) or [None, ""])[1].split())
        yield aid, title, summ


def _oa_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index; this puts the words back in order.
    A reconstruction, not a paraphrase: every token is the record's own."""
    if not inv:
        return ""
    pos: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            pos.append((i, word))
    return " ".join(w for _, w in sorted(pos))


def _oa_entries(text: str):
    try:
        d = json.loads(text)
    except Exception:  # noqa: BLE001
        return
    for w in d.get("results", []) or []:
        wid = w.get("doi") or w.get("id") or ""
        title = " ".join((w.get("title") or "").split())
        yield wid, title, _oa_abstract(w.get("abstract_inverted_index"))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def id_title_check() -> dict:
    """Did each asserted id resolve to the expected title?  Did each title search
    return the expected title?  Recorded, not assumed; a mismatch is a finding."""
    out: dict = {"offline": OFFLINE, "by_id": {}, "by_title": {},
                 "n_id_mismatch": 0, "n_title_search_miss": 0, "n_unfetched": 0}
    for group, spec in GROUPS.items():
        for name, (aid, frag, conf) in spec["by_id"].items():
            key = f"{group}__{name}"
            p = OUT / f"arxiv_id_{key}.atom"
            rec = {"group": group, "id": aid, "expected_fragment": frag, "prior_confidence": conf}
            if not p.exists():
                rec.update({"fetched": False, "match": None})
                out["n_unfetched"] += 1
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
                out["n_unfetched"] += 1
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
        "the id is wrong and must not be cited.  by_id.match null means the id was never "
        "fetched (offline or failed) and is UNVERIFIED -- do not cite it either.  "
        "by_title.found False means no arXiv entry has that title (pre-arXiv papers such as "
        "Forward 1993 and SPIE proceedings such as Wright 2001 are expected here); "
        "matched_ids are the ids the record itself supplies for the title.")
    (OUT / "id_title_check.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# Decoy-aware concept scan, per group, over EVERY fetched abstract (a paper
# fetched for one group can be prior art for another).
# --------------------------------------------------------------------------
def _atom_files(out_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(out_dir.glob("arxiv_*.atom"))


def _oa_files(out_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(out_dir.glob("oa_q_*.json"))


def _file_group(name: str) -> str | None:
    m = re.match(r"(?:arxiv_(?:q_title|q|id)|oa_q)_(g\d_[a-z_]+?)__", name)
    return m.group(1) if m else None


def _evidence(blob: str, m: re.Match, width: int = 120) -> str:
    """A verbatim window of the abstract around the first target match."""
    lo, hi = max(0, m.start() - width), min(len(blob), m.end() + width)
    return ("..." if lo > 0 else "") + blob[lo:hi] + ("..." if hi < len(blob) else "")


def scan_text(group: str, title: str, summary: str) -> dict | None:
    """Pure function: scan one abstract for one group.  None if no target fired."""
    rx = COMPILED[group]
    blob = f"{title} {summary}"
    matches: list[str] = []
    first: re.Match | None = None
    for r in rx["target"]:
        m = r.search(blob)
        if m:
            matches.append(m.group(0)[:80])
            if first is None or m.start() < first.start():
                first = m
    if not matches or first is None:
        return None
    decoys = [k for k, r in rx["decoys"].items() if r.search(blob)]
    boosters = [k for k, r in rx["boosters"].items() if r.search(blob)]
    return {"target_matches": matches, "decoys": decoys, "boosters": boosters,
            "evidence": _evidence(blob, first)}


def _rank_key(h: dict) -> tuple:
    # Closest neighbour first: decoy-free, then most boosters, then most target phrases.
    return (not h["decoy_free"], -len(h["boosters"]), -len(h["target_matches"]))


def scan(out_dir: pathlib.Path | None = None) -> dict:
    out_dir = out_dir or OUT
    atoms = _atom_files(out_dir)
    oas = _oa_files(out_dir)
    records: list[tuple[str, str, str, str, str]] = []  # (file, source, id, title, summ)
    for p in atoms:
        for aid, title, summ in _entries(p.read_text(errors="ignore")):
            records.append((p.name, "arxiv", aid, title, summ))
    for p in oas:
        for wid, title, summ in _oa_entries(p.read_text(errors="ignore")):
            records.append((p.name, "openalex", wid, title, summ))
    groups_out: dict = {}
    for group, spec in GROUPS.items():
        hits: dict[str, dict] = {}
        for fname, source, rid, title, summ in records:
            r = scan_text(group, title, summ)
            if r is None:
                continue
            key = _norm(title)[:120] or rid or f"{fname}:{title[:40]}"
            if key in hits:
                h = hits[key]
                if fname not in h["source_files"]:
                    h["source_files"].append(fname)
                if rid and rid not in h["ids"]:
                    h["ids"].append(rid)
                h["fetched_for_this_group"] = h["fetched_for_this_group"] or _file_group(fname) == group
                # The same paper can arrive with a stub abstract (an id lookup)
                # and a full one (a sweep); the scan fields follow the fuller record.
                if len(summ) > len(h["abstract_verbatim"]):
                    h.update({"target_matches": r["target_matches"], "decoys": r["decoys"],
                              "boosters": r["boosters"], "decoy_free": not r["decoys"],
                              "evidence": r["evidence"], "abstract_verbatim": summ,
                              "source": source})
                continue
            hits[key] = {
                "ids": [rid] if rid else [], "source": source, "title": title,
                "target_matches": r["target_matches"], "decoys": r["decoys"],
                "boosters": r["boosters"], "decoy_free": not r["decoys"],
                "evidence": r["evidence"],
                "source_files": [fname],
                "fetched_for_this_group": _file_group(fname) == group,
                "abstract_verbatim": summ,
            }
        allh = sorted(hits.values(), key=_rank_key)
        clean = [h for h in allh if h["decoy_free"]]
        strong = [h for h in allh if len(h["boosters"]) >= 2]
        neighbours = [{k: h[k] for k in ("ids", "source", "title", "decoy_free", "decoys",
                                          "boosters", "target_matches", "evidence")}
                      for h in allh[:15]]
        groups_out[group] = {
            "question": spec["question"],
            "n_target_regex_hits": len(allh),
            "n_after_decoy_removal": len(clean),
            "n_with_two_or_more_boosters": len(strong),
            "closest_neighbours": neighbours,
            "decoy_free_hits": [{k: v for k, v in h.items() if k != "abstract_verbatim"} for h in clean],
            "all_hits": allh,
            "interpretation": spec["interpretation"],
        }
    out = {
        "offline": OFFLINE,
        "n_atom_files": len(atoms),
        "n_openalex_files": len(oas),
        "n_abstracts_scanned": len(records),
        "n_unique_ids": len({r[2] for r in records if r[2]}),
        "per_group_counts": {g: {"hits": v["n_target_regex_hits"], "decoy_free": v["n_after_decoy_removal"],
                                 "two_plus_boosters": v["n_with_two_or_more_boosters"]}
                             for g, v in groups_out.items()},
        "groups": groups_out,
        "reading_guide": (
            "Every group's regexes were run over EVERY fetched abstract (arXiv atom and "
            "OpenAlex JSON).  For each hit the record shows which target phrase fired "
            "(target_matches), which decoy concepts co-occur (decoys), which supporting "
            "concepts co-occur (boosters), a VERBATIM evidence window around the first "
            "match, and the verbatim abstract, so a human can see at a glance whether a hit "
            "is genuine prior art.  closest_neighbours ranks the hits decoy-free first, then "
            "by booster count.  Decoy tags are flags, not vetoes.  The novelty position of "
            "the ROMAN channels (docs/roman.md §4) is to be read against each group's "
            "decoy_free_hits and booster-rich hits; it is not established until this file "
            "exists with n_abstracts_scanned > 0 and has been read.  When "
            "n_abstracts_scanned is 0 the scan says NOTHING about novelty."),
    }
    (out_dir / "concept_scan.json").write_text(json.dumps(out, indent=2))
    return out


def _write_summary() -> None:
    (OUT / "summary.json").write_text(json.dumps(
        {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "offline": OFFLINE,
         "n_urls": len(STATUS), "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"] and not s.get("skipped")),
         "n_skipped": sum(1 for s in STATUS if s.get("skipped")),
         "openalex_reachable": OA_REACHABLE,
         "non_arxiv_references": NON_ARXIV_REFERENCES,
         "groups": {g: spec["question"] for g, spec in GROUPS.items()},
         "n_queries_per_group": {g: {"by_id": len(s["by_id"]), "by_title": len(s["by_title"]),
                                     "arxiv_queries": len(s["queries"]),
                                     "openalex_queries": len(s.get("openalex", {}))}
                                 for g, s in GROUPS.items()},
         "status": STATUS}, indent=2))


def _write_readme() -> None:
    lines = [
        "# results/romanlit — ROMAN prior-art sweep",
        "",
        "Written by `scripts/romanlit_fetch.py` (workflow `roman-lit.yml`).  Every file is",
        "either a VERBATIM copy of what an archive returned or a machine record of what",
        "was asked and what came back.  Nothing is paraphrased; nothing is asserted from",
        "memory.  The novelty position in `docs/roman.md` §4 is to be read against",
        "`concept_scan.json`, and only when `n_abstracts_scanned > 0`.",
        "",
        "## Files",
        "",
        "| File | What it is |",
        "|---|---|",
        "| `README.md` | this file |",
        "| `summary.json` | run metadata: offline flag, counts of URLs ok / failed / skipped, whether OpenAlex answered, the five questions, the non-arXiv references the brief names (flagged as assertions), and the full status list |",
        "| `fetch_log.json` | every URL requested with every attempt's HTTP status (or the exception), bytes, entry count; `skipped` records a call that was never made (offline, or OpenAlex unreachable). Rewritten after every fetch so a killed run still leaves a record |",
        "| `id_title_check.json` | for every arXiv id the script asserted: did the fetched entry carry the expected title fragment (`match`)? `null` = unverified. For every named title: did the title search return it (`found`), and which ids does the record itself give (`matched_ids`)? |",
        "| `concept_scan.json` | the decoy-aware scan: every group's `target` regexes over every fetched abstract; per hit the phrase that fired, decoy and booster tags, a verbatim `evidence` window, and the verbatim abstract; per group `closest_neighbours` (decoy-free first, then booster-rich), `decoy_free_hits`, `all_hits`, and the reading `interpretation` |",
        "| `arxiv_id_<group>__<name>.atom` | arXiv API response for one asserted id (verbatim Atom) |",
        "| `arxiv_q_title_<group>__<name>.atom` | arXiv API title search for one named paper (verbatim Atom) |",
        "| `arxiv_q_<group>__<name>.atom` | arXiv API keyword sweep; `decoy_*` names are searches run ON PURPOSE for the confounding literature so a null is interpretable (verbatim Atom) |",
        "| `oa_probe.json` | the single OpenAlex reachability probe (verbatim JSON) |",
        "| `oa_q_<group>__<name>.json` | OpenAlex `works?search=` response (verbatim JSON; abstracts are stored inverted and are re-ordered, not rewritten, for the scan) |",
        "| `txt_<group>__<name>.txt` | `pdftotext` full text of an anchor paper, pulled ONLY after its id/title check passed |",
        "",
        "## Groups",
        "",
    ]
    for g, spec in GROUPS.items():
        lines.append(f"* **{g}** — {spec['question']}")
    lines += [
        "",
        "## How to read a null",
        "",
        "A group with `n_target_regex_hits: 0` after a run with `n_abstracts_scanned > 0`",
        "means the phrasings tried returned no abstract that matches the concept; it is",
        "evidence of novelty on THIS record, not proof.  A run with `offline: true` or",
        "`n_abstracts_scanned: 0` says nothing about novelty at all.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    print(f"romanlit: offline={OFFLINE} out={OUT}")
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
            print("== arXiv keyword sweeps ==")
            for name, q in spec["queries"].items():
                print(f"-- {name}")
                arxiv_query(group, name, q)
            print("== OpenAlex keyword searches ==")
            for name, q in spec.get("openalex", {}).items():
                print(f"-- {name}")
                openalex_query(group, name, q)
    except Exception as exc:  # noqa: BLE001
        # Recorded, never raised: the scan and the checks still run over what exists.
        print(f"romanlit: fetch loop aborted: {exc!r}")
        STATUS.append({"url": None, "dest": None, "ok": False, "http_status": None,
                       "attempts": [], "error": f"fetch loop aborted: {exc!r}"})
    finally:
        print("== id/title check ==")
        chk = id_title_check()
        print(json.dumps({k: v for k, v in chk.items()
                          if k in ("n_id_mismatch", "n_title_search_miss", "n_unfetched")}))
        for key, rec in chk["by_id"].items():
            print(f"  {key}: {rec.get('match')}  {rec.get('title_fetched', '')[:80]!r}")
        # Full texts only for anchors whose asserted id was VERIFIED by title.
        print("== full text of verified anchors ==")
        for group, names in FULLTEXT_ANCHORS.items():
            for name in names:
                rec = chk["by_id"].get(f"{group}__{name}", {})
                aid = GROUPS[group]["by_id"].get(name, ("", "", ""))[0]
                if rec.get("match") is True and aid:
                    print(f"-- {group}__{name} ({aid})")
                    try:
                        fulltext(group, name, aid)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  fulltext failed: {exc!r}")
                else:
                    print(f"-- {group}__{name}: id not verified ({rec.get('match')}); no full text pulled")
        print("== decoy-aware concept scan ==")
        res = scan()
        print(json.dumps({"n_abstracts_scanned": res["n_abstracts_scanned"],
                          "per_group_counts": res["per_group_counts"]}, indent=2))
        _write_fetch_log()
        _write_summary()
        _write_readme()
        n_ok = sum(1 for s in STATUS if s["ok"])
        print(f"\n{n_ok}/{len(STATUS)} fetches ok -> {OUT}")


if __name__ == "__main__":
    main()

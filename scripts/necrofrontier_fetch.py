#!/usr/bin/env python3
"""NECROFRONTIER prior-art sweep: fetch the record on the GitHub runner.

The sandbox blocks arxiv.org (``CONNECT tunnel failed, response 403``); the
Actions runner has egress.  This script establishes, from the record rather
than from memory, whether anyone has already looked for each of the new
necrosignatures proposed in ``docs/necrofrontier.md`` (S46 onward) --- the
observables of a NON-BIOLOGICAL successor that ended its makers, and the data
sources this repository has never touched.

Fourteen novelty questions, each its own query group with its own decoys
(a decoy is the natural-science literature that shares the vocabulary but
answers a different question):

  g1_isotope_purity     Isotopic purity (28Si / 12C / 6Li-7Li) as a
                        technosignature; the Presolar Grain Database mined for a
                        grain that no nucleosynthetic source can make.
                        Decoys: supernova X-grain papers (28Si-rich WITH 44Ti),
                        isotope-separation engineering, Galactic chemical evolution.
  g2_hot_swarm          Hot (~1500 K) engineered swarms read against the
                        hot-exozodi interferometric excess population.
                        Decoys: nano-grain trapping / exozodi dust modelling.
  g3_spherex_line       Single-channel stellar excess in SPHEREx / Euclid
                        slitless spectra (the industrial NIR line at all-sky scale).
                        Decoys: ices, galaxy emission lines, quasar selection.
  g4_century_fade       Century-baseline (DASCH / APPLAUSE) permanent fade,
                        cessation, or rising irregularity.
                        Decoys: KIC 8462852 plate debates, Menzel-gap artefacts,
                        LPV / K-giant century variability.
  g5_wd_residue         Polluted-white-dwarf abundance vectors no natural rock
                        process makes.  Decoys: chondritic / core-mantle
                        geochemistry, Li-rich cool DZ primordial interpretations.
  g6_shattered_cradle   Extreme debris disks around mature (>1 Gyr) stars read as
                        a destroyed or disassembled planet; slag vs collision
                        glass.  Decoys: young-star giant-impact EDDs.
  g7_artificial_molecule  Polar fluorocarbons / NF3 / SO2F2 etc. in ISM line
                        surveys and U-line lists.  Decoys: exoplanet CFC
                        detectability (JWST), interstellar CH3Cl / HF chemistry.
  g8_lunar_crypt        Artifacts in lunar permanently shadowed regions from
                        Diviner thermal, Mini-RF radar, ShadowCam.  Decoys:
                        water-ice cold traps, rock-abundance retrievals,
                        optical-NAC machine-learning anomaly papers.
  g9_terrestrial_record Prior technology in Earth's geological record (Silurian
                        hypothesis) and a fission / reactor-nuclide test at
                        extinction boundaries.  Decoys: Anthropocene markers,
                        Oklo geology, impact / volcanic boundary papers.
  g10_relay_geometry    Star-pair beam-spillover interception; eavesdropping on
                        interstellar links by geometry.  Decoys: Earth-transit-
                        zone, SETI ellipsoid, gravitational-lens relay theory.
  g11_growing_transit   Secular transit-depth growth across Kepler -> TESS
                        (construction), long-period tailed transits (mining).
                        Decoys: dilution / TDV / precession papers.
  g12_ignition          Monotonic mid-IR rise on an optically flat mature star
                        (the takeover transient).  Decoys: YSO bursts, AGN
                        turn-on, dust-forming novae, R CrB.
  g13_spot_ceiling      Superflares above the starspot magnetic-energy ceiling
                        as engineered stellar events.  Decoys: superflare
                        occurrence statistics, A-star flare contamination.
  g14_postbio_hosts     Technosignatures at hosts where biology is impossible
                        (pulsar planets, WD / NS Dyson rings, brown dwarfs,
                        free-floating planets) and the techno + anti-bio
                        conjunction.  Decoys: natural NS/WD debris disks, FFP
                        microlensing demographics.
  g15_insitu_grains     In-situ interstellar-grain composition (Cassini CDA,
                        Ulysses, Stardust) as a sample of sub-micron devices.
                        Decoys: interstellar dust flux / dynamics papers.

Nothing here is asserted from memory: named papers are fetched by *title
search* where the arXiv id is not certain, and where an id is given the fetched
title is compared against the expected one and any mismatch is recorded in
``id_title_check.json`` --- never silently trusted.  Verbatim abstracts are
saved; nothing is paraphrased.  Every fetch's HTTP status is recorded.

Outputs under ``results/necrofrontier_lit/``:
  arxiv_q_<group>__<name>.atom        arXiv API keyword search (verbatim)
  arxiv_q_title_<group>__<name>.atom  arXiv API title search (verbatim)
  arxiv_id_<group>__<name>.atom       arXiv API metadata for one id (verbatim)
  concept_scan.json                   decoy-aware scan, per group, over every
                                      fetched abstract
  id_title_check.json                 did each asserted id resolve to the
                                      expected title?  and did each title
                                      search return the expected title?
  summary.json                        HTTP status of every URL, entry counts
"""
from __future__ import annotations

import html
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path(os.environ.get("NECROFRONTIER_OUT", "results/necrofrontier_lit"))
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-necrofrontier/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
PAUSE = float(os.environ.get("NECROFRONTIER_PAUSE", "6.0"))
TRIES = int(os.environ.get("NECROFRONTIER_TRIES", "3"))
ARXIV_API = "http://export.arxiv.org/api/query"
ADS_API = "https://api.adsabs.harvard.edu/v1/search/query"
OPENALEX_API = "https://api.openalex.org/works"
ADS_TOKEN = os.environ.get("ADS_TOKEN") or os.environ.get("ADS_API_TOKEN") or ""
MAX_RESULTS = 60
# Optional comma-separated subset of groups to fetch (run 3 reached its
# deadline after g12; run 4 fetches g13-g15 only).
ONLY_GROUPS = {g for g in os.environ.get("NECROFRONTIER_GROUPS", "").split(",") if g}
# The first run (2026-09-13, run 34754534065) got HTTP 429 from the arXiv API
# on 64 of 65 requests at a 3 s pace and delivered nothing.  Three defences:
# a slower base pace; a Retry-After-aware exponential backoff (30 s, 90 s,
# 270 s); and, when arXiv still refuses, the same query against the ADS API
# with the repository's ADS_TOKEN secret, written in the same Atom shape so
# the id/title check and the concept scan run unchanged.  Every file records
# which service produced it.
BACKOFF = [20.0, 45.0, 45.0]
# Once the arXiv API has refused a request three times, the rest of the run
# goes ADS-first: the second run (34756xxx) spent its whole 75-minute budget
# in backoffs on 31 consecutive 429s and never reached the fallback.
ARXIV_THROTTLED = False
ARXIV_FAILS_BEFORE_SWITCH = 1


def _write_summary() -> None:
    """Written after EVERY fetch so a soft-deadline kill still leaves a record."""
    (OUT / "summary.json").write_text(json.dumps(
        {"n_urls": len(STATUS), "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"]),
         "groups": {g: spec["question"] for g, spec in GROUPS.items()},
         "status": STATUS}, indent=2))


def get(url: str, dest: pathlib.Path, tries: int = TRIES, pause: float = PAUSE) -> bool:
    """Fetch one URL to ``dest`` verbatim; the HTTP status of every attempt is recorded.

    When the arXiv API is throttling this run (``ARXIV_THROTTLED``), arXiv URLs
    are not attempted at all: the record says ``skipped: arxiv throttled`` and the
    caller goes straight to ADS."""
    global ARXIV_THROTTLED
    attempts: list[dict] = []
    if ARXIV_THROTTLED and "export.arxiv.org" in url:
        STATUS.append({"url": url, "dest": dest.name, "ok": False,
                       "attempts": [{"try": 0, "http": None, "error": "skipped: arxiv throttled this run"}]})
        _write_summary()
        return False
    throttled_here = 0
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
                code = getattr(r, "status", 200)
            dest.write_bytes(data)
            attempts.append({"try": i + 1, "http": code, "bytes": len(data)})
            STATUS.append({"url": url, "dest": dest.name, "ok": True, "bytes": len(data),
                           "attempts": attempts})
            _write_summary()
            print(f"  ok  {len(data):>9,}B  {dest.name}")
            time.sleep(pause)
            return True
        except urllib.error.HTTPError as exc:
            attempts.append({"try": i + 1, "http": exc.code, "error": str(exc)[:200]})
            print(f"  try {i + 1}/{tries} HTTP {exc.code}: {dest.name}")
            if exc.code in (429, 503):
                throttled_here += 1
                ra = exc.headers.get("Retry-After") if exc.headers else None
                wait = BACKOFF[min(i, len(BACKOFF) - 1)]
                if ra and str(ra).strip().isdigit():
                    wait = max(wait, float(ra))
                print(f"  throttled; sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
        except Exception as exc:  # noqa: BLE001
            attempts.append({"try": i + 1, "http": None, "error": repr(exc)[:200]})
            print(f"  try {i + 1}/{tries} failed: {exc!r}")
        time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "dest": dest.name, "ok": False, "attempts": attempts})
    _write_summary()
    if throttled_here >= tries and "export.arxiv.org" in url:
        _note_throttled()
    return False


def _note_throttled() -> None:
    global ARXIV_THROTTLED
    if not ARXIV_THROTTLED:
        ARXIV_THROTTLED = True
        print("  !! arXiv API is throttling this run: switching to ADS-first for the remainder")


# --------------------------------------------------------------------------
# Query groups.  by_id: name -> (arxiv id, title fragment expected, prior
# confidence in the id).  by_title: name -> exact title to search.  queries:
# name -> arXiv API search string.  target/decoys/boosters: regexes over
# title+abstract for the concept scan.
# --------------------------------------------------------------------------
GROUPS: dict[str, dict] = {
    "g1_isotope_purity": {
        "question": ("Has anyone proposed or searched for ISOTOPIC PURITY (28Si, 12C, 6Li/7Li, "
                     "mono-isotopic metals) as a technosignature, in presolar grains, "
                     "in-situ dust, or stellar photospheres?"),
        "target": [r"isotop\w*\s+(pur\w+|separat\w+|enrich\w+)\s+.*(technosignature|artificial|alien|extraterrestrial)",
                   r"(technosignature|artificial|alien|extraterrestrial).*isotop\w*\s+(pur\w+|separat\w+|enrich\w+|anomal\w+|ratio)",
                   r"presolar.*(technosignature|artificial|alien)",
                   r"(deuterium|D/H).*(technosignature|fusion.*civili)",
                   r"nuclear waste.*(star|stellar|photospher)"],
        "decoys": {"supernova_grain": r"supernova|core[- ]collapse|X grain|44Ti|26Al",
                   "engineering": r"centrifug|laser isotope|enrichment plant|quantum comput",
                   "gce": r"galactic chemical evolution|chemical evolution of the (galaxy|milky way)"},
        "boosters": {"technosig": r"technosignature", "purity": r"isotopically pure|mono-?isotopic|-99\d",
                     "grain_db": r"presolar grain database|20,?230"},
        "queries": {
            "isotope_technosignature": 'abs:"technosignature" AND abs:isotop',
            "isotope_pure_alien": 'abs:"isotopically pure" OR abs:"isotope separation" AND abs:(alien OR extraterrestrial OR civilization)',
            "presolar_db": 'abs:"presolar grain database"',
            "x_grain_extreme": 'abs:"presolar" AND abs:"28Si" AND abs:supernova',
            "deuterium_fusion_techno": 'abs:deuterium AND abs:(technosignature OR "fusion" AND civilization)',
            "nuclear_waste_star": 'abs:"nuclear waste" AND abs:(star OR stellar)',
            "isotope_stellar_anomaly": 'abs:"isotope ratio" AND abs:(dwarf OR "main sequence") AND abs:(magnesium OR lithium OR titanium)',
        },
        "by_id": {
            "catling2025_deuterium": ("2411.18595", "deuterium", "high"),
            "ellery2025_probe_technosig": ("2510.00082", "self-replicating", "medium"),
        },
        "by_title": {
            "pgd_sic": "The Presolar Grain Database. I. Silicon Carbide",
            "ww1980": "Nuclear waste spectrum as evidence of technological extraterrestrial civilizations",
            # run 3: the asserted ids for these two resolved to unrelated papers
            "lin2010_qingzhen": "Isotopic analysis of nanoSIMS-identified presolar SiC and Si3N4 grains from the Qingzhen EH3 chondrite",
        },
        "interpretation": ("A decoy-free hit that pairs isotope purity or isotope separation with a "
                           "technosignature reading is prior art for S46.  Supernova-grain papers are "
                           "the natural-background literature, not prior art.  Catling+2025 (D/H "
                           "depletion by fusion) is the nearest precedent and is expected to fire."),
    },
    "g2_hot_swarm": {
        "question": ("Has anyone read the ~1 % near-infrared interferometric excess ('hot exozodi', "
                     "~1500 K) population as a hot Dyson swarm, or tested those SEDs for a Planck "
                     "grey body against a nano-grain emissivity?"),
        "target": [r"(hot|near-?infrared|K-?band|H-?band).*(dyson|swarm|megastructure|technosignature)",
                   r"(dyson|swarm|megastructure).*(sublimation radius|1[0-9]{3}\s*K|hot dust|exozodi)",
                   r"exozodi\w*.*(artificial|technosignature|dyson)"],
        "decoys": {"nanograin": r"nano-?grain|magnetic trapping|charged grain|sublimation",
                   "comet": r"comet\w* (supply|deliver)|exocomet",
                   "companion": r"closure phase|faint companion|binary"},
        "boosters": {"technosig": r"technosignature|dyson", "interferom": r"interferomet|PIONIER|FLUOR|MATISSE|CHARA|VLTI",
                     "blackbody": r"black-?body|planck"},
        "queries": {
            "hot_exozodi_review": 'abs:"hot exozodiacal" OR abs:"hot exozodi"',
            "dyson_near_ir": 'abs:dyson AND abs:("near-infrared" OR "near infrared" OR "K band")',
            "dyson_hot": 'abs:dyson AND abs:("1000 K" OR "1500 K" OR "2000 K" OR "hot")',
            "swarm_sublimation": 'abs:(swarm OR megastructure) AND abs:"sublimation"',
            "exozodi_technosig": 'abs:exozodi AND abs:(technosignature OR artificial OR dyson)',
            "interferometric_excess_origin": 'abs:"near-infrared excess" AND abs:interferomet AND abs:(origin OR nature)',
        },
        "by_id": {
            "absil2013_fluor": ("1307.2488", "First statistics", "high"),
            "ertel2014_pionier": ("1409.6143", "PIONIER", "medium"),
            "ertel2025_review": ("2504.00295", "hot exozodiacal", "medium"),
            "kirchschlager2017": ("1701.07271", "hot exozodiacal", "medium"),
            "kirchschlager2020_kappa_tuc": ("2009.02334", "L band", "medium"),
            "stuber2026_kappa_tuc_companion": ("2512.03209", "Tuc", "medium"),
            "hephaistos1": ("2201.11123", "Hephaistos", "high"),
            "dyson_hr_2026": ("2602.23270", "Dyson", "low"),
        },
        "by_title": {
            "ertel2016_variability": "A near-infrared interferometric survey of debris-disc stars. V. PIONIER search for variability",
            "marshall2016_polarimetry": "Polarization measurements of hot dust stars and the local interstellar medium",
            "rieke2016_trapping": "Magnetic Grain Trapping and the Hot Excesses Around Early-type Stars",
        },
        "interpretation": ("Expected: the exozodi literature fires 'hot dust' targets with nanograin decoys; "
                           "a decoy-free hit that names a swarm or technosignature for this population is "
                           "prior art for S47.  Hephaistos I is fetched to quote its 100-1000 K restriction."),
    },
    "g3_spherex_line": {
        "question": ("Has anyone searched SPHEREx (102-channel all-sky) or Euclid NISP slitless spectra of "
                     "STARS for a single-channel excess / unresolved emission line (a laser or "
                     "industrial line), or proposed it?"),
        "target": [r"spherex.*(technosignature|laser|seti|narrow.?line|monochromatic|single.?channel)",
                   r"(technosignature|laser|seti).*spherex",
                   r"euclid.*(technosignature|laser|seti)", r"(technosignature|laser|seti).*euclid",
                   r"(nisp|slitless|grism).*(laser|technosignature|artificial line)"],
        "decoys": {"ices": r"\bices?\b|H2O ice|CO2 ice",
                   "galaxy_lines": r"H-?alpha emitter|emission-line galax|Lyman|redshift survey",
                   "quasar": r"quasar|AGN|QSO"},
        "boosters": {"technosig": r"technosignature|seti|laser", "stellar": r"\bstars?\b|stellar",
                     "single_channel": r"single (channel|band)|unresolved (line|feature)"},
        "queries": {
            "spherex_seti": 'abs:SPHEREx AND abs:(technosignature OR SETI OR laser)',
            "euclid_seti": 'abs:Euclid AND abs:(technosignature OR SETI OR laser)',
            "nir_laser_search": 'abs:laser AND abs:(SETI OR technosignature) AND abs:("near-infrared" OR "near infrared" OR "1.06" OR "1.55")',
            "spherex_stellar_spectra": 'abs:SPHEREx AND abs:(stars OR stellar) AND abs:(spectra OR spectrophotometry)',
            "slitless_stellar_lines": 'abs:slitless AND abs:(stellar OR star) AND abs:"emission line"',
            "spherex_anomaly": 'abs:SPHEREx AND abs:(anomal OR outlier OR unusual)',
        },
        "by_id": {
            "spherex_pipeline_2025": ("2511.15823", "SPHEREx", "medium"),
            "euclid_q1_overview": ("2503.15302", "Euclid", "medium"),
        },
        "by_title": {
            "spherex_mission": "Cosmology with the SPHEREX All-Sky Spectral Survey",
            "vides2019_wfirst_laser": "Sensitivity of the WFIRST Coronagraph to Extrasolar Laser Beacons",
        },
        "interpretation": ("Any decoy-free hit pairing SPHEREx/Euclid with a laser or technosignature "
                           "search on stars is prior art for S48/S49.  Vides+2019 and NIROSETI are the "
                           "targeted-NIR precedents (not all-sky)."),
    },
    "g4_century_fade": {
        "question": ("Has anyone run a blind DASCH DR7 / APPLAUSE search for stars that faded "
                     "permanently, whose periodicity ceased, or whose scatter rose over a century, "
                     "with the Menzel gap and per-plate limits modelled?"),
        "target": [r"dasch.*(fad\w+|dimm\w+|vanish\w+|disappear\w+|ceas\w+|secular)",
                   r"(fad\w+|dimm\w+|vanish\w+|disappear\w+|ceas\w+|secular).*dasch",
                   r"(photographic plate|plate archive).*(vanish|disappear|fade|century)",
                   r"applause.*(fad|vanish|variab)"],
        "decoys": {"tabby": r"KIC ?8462852|Boyajian|Tabby",
                   "menzel_artefact": r"Menzel gap|emulsion|plate (limit|sensitivity)",
                   "lpv": r"long-?period variable|Mira|K giant|red giant"},
        "boosters": {"blind": r"blind|all-?sky|catalog|population", "dr7": r"DR7|data release 7",
                     "ceased": r"ceas|stopped|permanent"},
        "queries": {
            "dasch_fade": 'all:DASCH AND abs:(fading OR dimming OR vanish OR disappear)',
            "dasch_dr7": 'all:DASCH AND abs:"DR7"',
            "dasch_variability_blind": 'all:DASCH AND abs:(variability OR variable) AND abs:(catalog OR survey)',
            "applause_plates": 'abs:APPLAUSE AND abs:plate',
            "plate_vanish": 'abs:"photographic plates" AND abs:(vanish OR disappear OR "no longer")',
            "century_cessation": 'abs:(pulsation OR periodicity) AND abs:(ceased OR stopped OR cessation) AND abs:(century OR plates)',
        },
        "by_id": {
            "dasch_dr7_paper": ("2501.12977", "DASCH", "high"),
            "schaefer2016_kic8462852": ("1601.03256", "KIC 8462852", "medium"),
            "lund2016_dasch": ("1605.02760", "F-star Brightness", "high"),
            "applause_2024": ("2404.17355", "APPLAUSE", "medium"),
            "plate_sensitivity_2026": ("2604.16470", "plate", "low"),
        },
        "by_title": {
            "dasch_dr7_title": "DASCH: Bringing 100+ Years of Photographic Data into the 21st Century and Beyond",
            "hippke2016_plates": "A statistical analysis of the accuracy of the digitized magnitudes of photometric plates on the time scale of decades",
        },
        "interpretation": ("Expected hits are the KIC 8462852 plate debate (decoy 'tabby') and the "
                           "DR7 description.  A decoy-free blind fade / cessation search on DR7 is prior "
                           "art for S50; none is expected."),
    },
    "g5_wd_residue": {
        "question": ("Beyond Huang, Tao & Zhang 2026 (siderophile template), has anyone tested polluted "
                     "white dwarf abundance vectors for compositions no natural rock process can make "
                     "(process-orthogonal element pairs, alloy patterns, Be without Li/B), or published "
                     "a calibrated misfit list against the full natural family?"),
        "target": [r"(polluted|metal-?polluted|DZ|DAZ|DBZ).*(white dwarf).*(technosignature|artificial|refined|engineered|alloy)",
                   r"(technosignature|artificial|refined|engineered).*(white dwarf)",
                   r"white dwarf.*(no (known )?meteorite|cannot be (reproduced|explained)|unexplain\w+|defies)",
                   r"(beryllium|lithium|potassium).*white dwarf.*(anomal|excess|enhanc)"],
        "decoys": {"chondritic": r"chondrit|bulk earth|core[- ]mantle|differentiat",
                   "primordial_li": r"big bang|primordial|galactic lithium",
                   "spallation": r"spallation|cosmic ray"},
        "boosters": {"technosig": r"technosignature", "misfit": r"misfit|goodness of fit|posterior predictive|unexplain",
                     "trace": r"scandium|vanadium|strontium|copper|beryllium"},
        "queries": {
            "wd_technosig": 'abs:"white dwarf" AND abs:(technosignature OR artificial OR "refined material")',
            "wd_no_meteorite": 'abs:"white dwarf" AND abs:(pollut OR accret) AND abs:("no known" OR "cannot be explained" OR unusual OR anomalous OR "defies")',
            "wd_beryllium": 'abs:"white dwarf" AND abs:beryllium',
            "wd_lithium_dz": 'abs:"white dwarf" AND abs:lithium AND abs:(pollut OR DZ)',
            "wd_bayesian_pollution": 'abs:"white dwarf" AND abs:pollut AND abs:(bayesian OR "model selection" OR evidence)',
            "wd_pewdd": 'abs:"white dwarf" AND abs:(database OR compilation) AND abs:(pollut OR planetary)',
        },
        "by_id": {
            "huang2026_wd_technosig": ("2605.29811", "Technosignatures", "high"),
            "pewdd_2024": ("2409.16046", "White Dwarf", "medium"),
            "kaiser2024_lithium": ("2412.01878", "lithium", "medium"),
            "klein2021_beryllium": ("2102.01834", "beryllium", "medium"),
            "xu2013_pg1225": ("1302.4799", "Beyond-Primitive", "high"),
            "putirka_xu2021": ("2111.03124", "exoplanet", "medium"),
            "buchan2022_pressure": ("2111.08779", "white dwarf", "medium"),
            "farihi2026_pure_metal": ("2601.16253", "white dwarf", "low"),
        },
        "by_title": {
            "kawka_vennes_nltt19868": "Deep secrets of the polluted white dwarf NLTT 19868",
        },
        "interpretation": ("Huang+2026 will fire and is known prior art (dense siderophile template on "
                           "PEWDD).  S51's claim is the process-orthogonal pair test and the calibrated "
                           "misfit list; a decoy-free hit with the 'misfit' booster would occupy it."),
    },
    "g6_shattered_cradle": {
        "question": ("Has anyone read extreme debris disks around MATURE (>1 Gyr) stars, or warm dust at "
                     "habitable-zone temperature above the collisional steady-state maximum, as the "
                     "residue of a destroyed or disassembled planet, with a mineralogy (slag vs "
                     "collision glass) discriminant?"),
        "target": [r"(debris|dust).*(destroyed|disassembl\w+|dismantl\w+) planet",
                   r"(planet|planetary).*(disassembl\w+|dismantl\w+).*(debris|dust|disk|disc)",
                   r"(debris disk|debris disc|warm dust|extreme debris).*(technosignature|artificial|megastructure|dyson)",
                   r"(technosignature|megastructure|dyson).*(collisional cascade|debris disk|dust veil|ground to dust)",
                   r"(refined|engineered|slag|smelt\w+).*(dust|debris|grain)"],
        "decoys": {"young_impact": r"giant impact|terrestrial planet formation|\b(10|20|30|50|80|100|120|150) ?Myr\b",
                   "hephaistos_contam": r"Hot DOG|background galax|dusty galaxy",
                   "protoplanetary": r"protoplanetary|T Tauri|Herbig"},
        "boosters": {"mature": r"\bGyr\b|old star|mature|main-?sequence", "mineralogy": r"silica|forsterite|enstatite|FeS|mineralog",
                     "technosig": r"technosignature|dyson|megastructure"},
        "queries": {
            "edd_old": 'abs:"extreme debris" AND abs:(Gyr OR old OR mature)',
            "edd_review": 'abs:"extreme debris disk" OR abs:"extreme debris disc"',
            "planet_destroyed_debris": 'abs:(destroyed OR disassembled OR dismantled) AND abs:planet AND abs:(debris OR dust)',
            "dyson_debris_confusion": 'abs:dyson AND abs:(debris OR "collisional cascade")',
            "warm_dust_old_star_origin": 'abs:"warm dust" AND abs:(Gyr OR old) AND abs:(origin OR transient OR collision)',
            "silica_dust_debris": 'abs:silica AND abs:debris AND abs:(disk OR disc)',
        },
        "by_id": {
            "moor2021_edd": ("2103.00568", "extreme debris", "high"),
            "su2026_edd_review": ("2607.06684", "Extreme Debris", "medium"),
            "kennedy_wyatt2013": ("1305.6607", "exo-Zodi luminosity function", "high"),
            "stevens2016_selfdestruct": ("1507.08530", "self-destructive", "high"),
            "lacki2025_ground_to_dust": ("2504.21151", "Dust", "high"),
            "lacki2026_dust_to_dust": ("2606.08373", "Dust", "medium"),
            "weinberger2011_bd20307": ("1010.6218", "BD+20 307", "medium"),
            "hd15407a_jwst_2026": ("2607.21948", "Fading", "low"),
        },
        "by_title": {
            "wyatt2007_transient": "Transience of hot dust around sun-like stars",
            "melis2012_disappearing": "Rapid disappearance of a warm, dusty circumstellar disk",
            "lisse2009_hd172555": "Abundant Circumstellar Silica Dust and SiO Gas Created by a Giant Hypervelocity Collision in the ~12 Myr HD172555 System",
        },
        "interpretation": ("Stevens+2016 and Lacki 2025/2026 name the channel and will fire; both "
                           "decline a search.  A decoy-free hit with 'mature' and 'mineralogy' boosters "
                           "and a technosignature framing is prior art for S52/S53."),
    },
    "g7_artificial_molecule": {
        "question": ("Has anyone searched interstellar or circumstellar line surveys (or their "
                     "unidentified-line lists) for industrial molecules --- polar fluorocarbons, "
                     "HFCs, CFCs, NF3, SO2F2, COF2, CF3CN --- as a technosignature?"),
        "target": [r"(interstellar|circumstellar|molecular cloud|line survey).*(technosignature|artificial|industrial|anthropogenic|pollut\w+)",
                   r"(technosignature|artificial|industrial).*(interstellar medium|molecular cloud|line survey|rotational)",
                   r"(fluoroform|CHF3|CH2F2|NF3|SO2F2|COF2|CF3CN|chlorofluorocarbon|CFC|freon).*(interstellar|circumstellar|molecular cloud|search)",
                   r"unidentified line.*(artificial|technosignature)"],
        "decoys": {"exoplanet_cfc": r"exoplanet|TRAPPIST|transmission spectr|JWST|atmospher",
                   "ch3cl": r"CH3Cl|methyl chloride|organohalogen",
                   "hf_chem": r"\bHF\b|CF\+|fluorine chemistry|hydrogen fluoride"},
        "boosters": {"technosig": r"technosignature|artificial|industrial", "uline": r"unidentified (line|feature)|U-?line",
                     "polyfluoro": r"CHF3|CH2F2|NF3|SO2F2|COF2|CF3|fluorocarbon"},
        "queries": {
            "ism_technosig": 'abs:(interstellar OR "molecular cloud") AND abs:(technosignature OR "artificial molecule" OR "industrial pollution")',
            "fluorocarbon_ism": 'abs:(fluorocarbon OR fluoroform OR CHF3 OR NF3 OR chlorofluorocarbon) AND abs:(interstellar OR circumstellar OR "molecular cloud")',
            "uline_lists": 'abs:"unidentified lines" AND abs:(survey OR "line survey")',
            "organohalogen_ism": 'abs:(organohalogen OR "methyl chloride" OR CH3Cl) AND abs:interstellar',
            "fluorine_chemistry_ism": 'abs:fluorine AND abs:interstellar AND abs:chemistry',
            "cfc_technosig_exoplanet": 'abs:(chlorofluorocarbon OR CFC OR "SF6" OR "NF3") AND abs:technosignature',
        },
        "by_id": {
            "lin2014_cfc_wd": ("1406.3025", "Detecting Industrial Pollution", "medium"),
            "haqq_misra2022_cfc": ("2202.05858", "Detectability", "medium"),
            "seager2023_nf3_sf6": ("2308.13667", "NF3", "medium"),
            "schwieterman2024_pfc": ("2405.11149", "Artificial Greenhouse", "medium"),
            "vidal2026_review": ("2605.21093", "Technosignatures", "high"),
            "crockett2014_orion_hifi": ("1405.2351", "Orion", "medium"),
            "he2008_irc10216": ("0802.1963", "IRC", "medium"),
            "gotham_inventory_2025": ("2509.06256", "TMC-1", "low"),
        },
        "by_title": {
            "fayolle2017_ch3cl": "Protostellar and cometary detections of organohalogens",
            "acharyya_herbst2017": "Gas-Grain Fluorine- and Chlorine Chemistry in the Interstellar Medium",
        },
        "interpretation": ("All expected hits are exoplanet-atmosphere CFC papers (decoy 'exoplanet_cfc') "
                           "or natural organohalogen chemistry.  A decoy-free hit pairing an ISM line "
                           "survey with an artificial-molecule search is prior art for S54."),
    },
    "g8_lunar_crypt": {
        "question": ("Has anyone searched Diviner thermal maps, Mini-RF radar, or ShadowCam imagery of "
                     "lunar permanently shadowed regions for artifacts (anisothermal hot components, "
                     "point-like high-CPR reflectors)?"),
        "target": [r"(lunar|moon).*(artifact|artefact|technosignature|alien|extraterrestrial (probe|artifact))",
                   r"(artifact|artefact|technosignature).*(lunar|moon)",
                   r"(diviner|mini-?rf|shadowcam|permanently shadowed).*(anomal|artifact|technosignature)",
                   r"search for extraterrestrial artifacts|SETA\b"],
        "decoys": {"ice": r"water ice|volatile|cold trap|ice deposit",
                   "rock_abundance": r"rock abundance|thermal inertia|boulder",
                   "nac_ml": r"LROC|narrow angle camera|NAC\b|convolutional|autoencoder|variational"},
        "boosters": {"technosig": r"technosignature|artifact|artefact|lurker", "psr": r"permanently shadowed|PSR",
                     "thermal_radar": r"diviner|radar|circular polari"},
        "queries": {
            "lunar_artifacts": 'abs:(lunar OR moon) AND abs:(artifact OR artefact OR technosignature) AND abs:(search OR alien OR extraterrestrial)',
            "lunar_ml_anomaly": 'abs:(lunar OR moon) AND abs:anomaly AND abs:("machine learning" OR unsupervised OR autoencoder)',
            "diviner_psr": 'abs:Diviner AND abs:("permanently shadowed" OR "cold trap")',
            "minirf_cpr": 'abs:("Mini-RF" OR "circular polarization ratio") AND abs:(lunar OR moon)',
            "shadowcam": 'abs:ShadowCam',
            "lurkers": 'abs:lurker AND abs:(artifact OR probe OR alien)',
            "solar_system_technosig": 'abs:technosignature AND abs:"solar system" AND abs:(artifact OR probe OR lunar)',
        },
        "by_id": {
            "benford2019_lurkers": ("1903.09582", "Lurkers", "high"),
            "lesnikowski2020_lunar_anomaly": ("2001.04634", "Lunar", "medium"),
            "lunar_ml_anomalies_2026": ("2608.09350", "Lunar", "low"),
            "solar_system_technosig_review_2026": ("2606.13797", "Solar System", "low"),
            "lunar_regolith_micron_2026": ("2606.24028", "regolith", "low"),
        },
        "by_title": {
            "davies_wagner2013": "Searching for alien artifacts on the moon",
        },
        "interpretation": ("Optical-NAC machine-learning anomaly papers (decoy 'nac_ml') are expected and "
                           "are not prior art for a THERMAL / RADAR PSR search.  A decoy-free hit with "
                           "'psr' and 'thermal_radar' boosters occupies S55."),
    },
    "g9_terrestrial_record": {
        "question": ("Beyond Schmidt & Frank 2018, has anyone tested Earth's own sedimentary record for a "
                     "prior technological event --- a fission-product or reactor-nuclide pattern at an "
                     "extinction boundary, or refined particulates in pre-industrial strata?"),
        "target": [r"silurian hypothesis|prior (industrial|technological) civili[sz]ation|pre-?human civili",
                   r"(geolog\w+|sediment\w+|stratigraph\w+) record.*(civili[sz]ation|technolog\w+ species|technosignature)",
                   r"(fission|reactor|236U|uranium-236|129I).*(extinction|boundary|sediment|crust)",
                   r"(anthropogenic|industrial).*(spherule|particle).*(pre-?industrial|cretaceous|older)"],
        "decoys": {"anthropocene": r"anthropocene|GSSP|1950|great acceleration",
                   "oklo": r"Oklo|natural (fission )?reactor",
                   "impact_volcanic": r"Chicxulub|iridium|Deccan|Siberian Traps|large igneous"},
        "boosters": {"technosig": r"technosignature|civili[sz]ation", "nuclide": r"236U|129I|244Pu|60Fe|fission product",
                     "database": r"database|compilation|EarthChem|GEOROC|SGP"},
        "queries": {
            "silurian": 'abs:"Silurian hypothesis" OR abs:"prior industrial civilization"',
            "prior_species": 'abs:"technological species" AND abs:(prior OR indigenous OR earlier)',
            "fission_boundary": 'abs:(fission OR "236U" OR "129I") AND abs:(extinction OR boundary OR sediment)',
            "crust_radionuclide": 'abs:("ferromanganese crust" OR "deep-sea sediment") AND abs:("244Pu" OR "60Fe" OR "236U" OR "129I")',
            "extinction_unexplained": 'abs:(Devonian OR Ordovician OR Hangenberg OR Kellwasser) AND abs:extinction AND abs:cause',
            "sedimentary_geochem_db": 'abs:("Sedimentary Geochemistry and Paleoenvironments" OR SGP) AND abs:database',
        },
        "by_id": {
            "schmidt_frank2018": ("1804.03748", "Silurian", "high"),
            "wright2018_prior_species": ("1704.07263", "Prior Indigenous", "high"),
            "fields2020_devonian_sn": ("2007.01887", "Devonian", "medium"),
        },
        "by_title": {
            "wallner2016_60fe": "Recent near-Earth supernovae probed by global deposition of interstellar radioactive 60Fe",
            "rose2015_scp": "Spheroidal Carbonaceous Fly Ash Particles Provide a Globally Synchronous Stratigraphic Marker for the Anthropocene",
            "wallner2021_science": "60Fe and 244Pu deposited on Earth constrain the r-process yields of recent nearby supernovae",
        },
        "interpretation": ("Schmidt & Frank 2018 and Wright 2018 are the framing papers and will fire; "
                           "neither proposes the fission vector or a database test.  A decoy-free hit "
                           "with the 'nuclide' and 'database' boosters is prior art for S56."),
    },
    "g10_relay_geometry": {
        "question": ("Has anyone selected star PAIRS at catalogue scale such that Earth lies inside the "
                     "far-field spillover cone of an A->B beam, and re-cut archival SETI data on that "
                     "geometry, or is the concept confined to single targets (SGL antipode, anti-solar "
                     "point, planet-planet occultation)?"),
        "target": [r"(spillover|spill-?over|leakage|eavesdrop\w+|intercept\w+).*(interstellar|communication|beam|transmi)",
                   r"(star|stellar) pairs?.*(align\w+|line of sight|collinear|conjunction).*(seti|technosignature|communication)",
                   r"(align\w+|antipod\w+|extension of the line|anti-?solar point).*(seti|technosignature|transmitter)",
                   r"(relay|network|node).*(interstellar communication|seti|technosignature)"],
        "decoys": {"etz": r"earth transit zone|ETZ|transit of earth",
                   "ellipsoid": r"SETI ellipsoid|synchroni[sz]",
                   "lens_theory": r"gravitational lens\w*.*(relay|node|focal)"},
        "boosters": {"pair": r"pairs? of stars|star pairs?|two stars|between (two|the) stars", "geometry": r"geometr|cone|divergence|beam width",
                     "archival": r"archiv|re-?analy|open data|Breakthrough Listen|MeerKAT"},
        "queries": {
            "spillover_seti": 'abs:(spillover OR eavesdrop OR intercept) AND abs:(SETI OR technosignature OR "interstellar communication")',
            "aligned_stars_seti": 'abs:(aligned OR alignment OR collinear OR conjunction) AND abs:(SETI OR technosignature) AND abs:stars',
            "antisolar_sgl": 'abs:("solar gravitational lens" OR "anti-solar" OR antipode) AND abs:(SETI OR technosignature)',
            "occultation_spillover": 'abs:occultation AND abs:(SETI OR technosignature OR "planet-planet")',
            "relay_network": 'abs:(relay OR network) AND abs:"interstellar communication"',
            "bl_archival_recut": 'abs:"Breakthrough Listen" AND abs:(archival OR "open data" OR reanalysis)',
        },
        "by_id": {
            "sheikh2020_etz": ("2002.06162", "Earth Transit Zone", "high"),
            "tusay2024_trappist_ppo": ("2409.08313", "TRAPPIST-1", "medium"),
            "davenport2022_ellipsoid": ("2206.04092", "SETI Ellipsoid", "medium"),
            "hippke2020_network_i": ("2009.01866", "communication network", "low"),
            "forgan2019_transit_network": ("1707.03730", "Communications Network", "high"),
            "kerins2021_mutual": ("2010.04089", "Mutual detectability", "low"),
        },
        "by_title": {
            "tusay2022_sgl_alphacen": "A Search for Radio Technosignatures at the Solar Gravitational Lens Targeting Alpha Centauri",
            "kerby_wright2021": "Stellar Gravitational Lens Engineering for Interstellar Communication and Artifact SETI",
        },
        "interpretation": ("The single-target precedents (Tusay 2022/2024, Hort 2024) will fire; a "
                           "decoy-free hit with the 'pair' and 'geometry' boosters at catalogue scale "
                           "is prior art for S60."),
    },
    "g11_growing_transit": {
        "question": ("Has anyone searched for SECULAR transit-depth (Rp/R*) growth across Kepler -> K2 -> "
                     "TESS as construction around a planet, or for long-period (>30 d) transits with "
                     "asymmetric, variable, chromatic tails (mining) --- or a systematic Kepler-vs-TESS "
                     "depth-consistency catalogue at all?"),
        "target": [r"(transit depth|radius ratio|Rp/R\*?).*(secular|increas\w+|grow\w+|drift|time-?variab)",
                   r"(transit depth variation|TDV).*(kepler|tess|search|survey)",
                   r"(construction|megastructure|artificial).*(transit depth|transit)",
                   r"(kepler|K2).*(tess).*(depth|radius ratio).*(compar|consisten)",
                   r"(tail|disintegrat\w+|evaporat\w+).*(long-?period|beyond|cold|distant).*(transit|planet)"],
        "decoys": {"dilution": r"dilution|contaminat|crowding|blend",
                   "precession": r"precess|oblate|nodal|impact parameter",
                   "usp": r"ultra-?short|Kepler-1520|KIC 12557548|K2-22|catastrophically evaporating"},
        "boosters": {"technosig": r"technosignature|megastructure|construction|artificial", "cross_mission": r"kepler.*tess|tess.*kepler",
                     "secular": r"secular|decade|long-?term|years"},
        "queries": {
            "tdv_search": 'abs:"transit depth variation" OR abs:"transit depth variations"',
            "kepler_tess_depth": 'abs:Kepler AND abs:TESS AND abs:("transit depth" OR "radius ratio")',
            "depth_increase_secular": 'abs:"transit depth" AND abs:(secular OR increasing OR "time-variable" OR drift)',
            "megastructure_transit": 'abs:(megastructure OR "artificial transit" OR technosignature) AND abs:transit',
            "long_period_tail": 'abs:(disintegrating OR "dust tail" OR "comet-like") AND abs:transit AND abs:(period OR "long-period")',
            "asymmetric_transit_survey": 'abs:asymmetric AND abs:transit AND abs:(search OR survey OR catalog)',
        },
        "by_id": {
            "wang_espinoza2024_tdv": ("2311.02154", "Transit Depth", "medium"),
            "zuckerman2023_bl_transits": ("2312.07903", "Kepler", "medium"),
            "han2025_tess_radii": ("2506.19985", "TESS", "medium"),
            "judkovsky2020_koi120": ("2010.13051", "KOI", "medium"),
            "wright2016_ghat4": ("1510.04606", "Transiting Megastructures", "medium"),
            "haqq_misra2022_missions": ("2206.00030", "technosignatures", "medium"),
            "hon2025_bd05": ("2501.05431", "disintegrating", "medium"),
        },
        "by_title": {
        },
        "interpretation": ("Wang & Espinoza 2024 (within-TESS TDV) and Zuckerman 2023 (Kepler single-transit "
                           "anomalies) are the nearest executed searches and will fire.  A decoy-free "
                           "cross-mission secular-growth search is prior art for S57; none is expected."),
    },
    "g12_ignition": {
        "question": ("Has anyone selected field (non-YSO, non-AGN) stars whose W1/W2 rose monotonically "
                     "over the NEOWISE decade with a flat optical light curve --- an infrared excess "
                     "being BORN around a mature star?"),
        "target": [r"(mid-?infrared|W1|W2|NEOWISE|WISE).*(brighten\w+|rising|monotonic|increas\w+).*(star|stellar|main.?sequence)",
                   r"(infrared excess).*(appear\w+|emerg\w+|born|onset|new).*(star)",
                   r"(neowise|unTimely|WISE).*(variab\w+).*(catalog|catalogue|census)",
                   r"(optically (constant|flat|quiescent)).*(infrared)"],
        "decoys": {"yso": r"young stellar|YSO|protostar|FUor|EXor|Class (I|II)",
                   "agn": r"AGN|quasar|changing-?look|active galactic|tidal disruption|TDE",
                   "evolved": r"Mira|AGB|carbon star|R Coronae|RCB|dust-?forming nova|nova"},
        "boosters": {"monotonic": r"monotonic|secular|linear (rise|trend)", "optical_flat": r"optical\w* (constant|flat|quiescent|unchanged)",
                     "field_star": r"main-?sequence|field star|dwarf"},
        "queries": {
            "neowise_variables_catalog": 'abs:(NEOWISE OR unTimely) AND abs:variab AND abs:(catalog OR catalogue)',
            "mir_brightening_stars": 'abs:("mid-infrared" OR "mid infrared") AND abs:(brightening OR rising) AND abs:(star OR stellar)',
            "ir_excess_appearance": 'abs:"infrared excess" AND abs:(appear OR emerg OR onset OR new) AND abs:(star OR disk)',
            "iras_akari_wise_brightening": 'abs:IRAS AND abs:AKARI AND abs:WISE AND abs:(brighten OR variab)',
            "optically_flat_ir_variable": 'abs:infrared AND abs:variab AND abs:("optically constant" OR "no optical" OR "optical light curve")',
            "agn_turn_on_wise": 'abs:WISE AND abs:("turn-on" OR "turning on") AND abs:monotonic',
        },
        "by_id": {
            "kang2025_untimely_variables": ("2511.22071", "unTimely", "high"),
            "onozato_ita2015": ("1501.05721", "brighten", "low"),
            "yso_decade_2025": ("2503.13971", "Young", "low"),
            "ngc6447_turn_on": ("2602.21502", "NGC6447", "high"),
            "contardo_hogg2024": ("2403.18941", "infrared excess", "medium"),
        },
        "by_title": {
            "park2021_neowise_yso": "Quantifying Variability of Young Stellar Objects in the Mid-infrared over 6 Years with the Near-Earth Object Wide-field Infrared Survey Explorer",
        },
        "interpretation": ("The unTimely catalogue, the YSO decade papers and the AGN turn-on cases will "
                           "fire with their decoys.  A decoy-free hit with 'monotonic', 'optical_flat' "
                           "and 'field_star' boosters is prior art for S61."),
    },
    "g13_spot_ceiling": {
        "question": ("Has anyone mined the superflare catalogues for flares whose energy EXCEEDS the "
                     "magnetic energy the star's measured spot coverage can store, as an anomaly "
                     "population --- or read any flare as engineered?"),
        "target": [r"(superflare|flare).*(exceed\w+|above|larger than|cannot be (stored|explained)).*(spot|magnetic energy)",
                   r"(spot|starspot).*(energy).*(upper limit|ceiling|bound).*(flare)",
                   r"(flare|superflare).*(artificial|technosignature|engineered|weapon)",
                   r"(artificial|technosignature).*(flare|stellar activity)"],
        "decoys": {"occurrence": r"occurrence (rate|frequency)|once per|per century|frequency distribution",
                   "a_star": r"A-?type|A star|delta Scuti",
                   "shield": r"shield|mitigat|magnetosphere"},
        "boosters": {"technosig": r"technosignature|artificial|engineered", "energy_bound": r"upper limit|magnetic energy|B\^2|spot area|starspot coverage",
                     "sunlike": r"sun-?like|solar-?type|slowly rotating"},
        "queries": {
            "superflare_spot_energy": 'abs:superflare AND abs:(starspot OR "spot area" OR "magnetic energy") AND abs:("upper limit" OR maximum)',
            "flare_technosig": 'abs:(flare OR superflare) AND abs:(technosignature OR artificial OR SETI)',
            "superflare_slow_rotators": 'abs:superflare AND abs:("slowly rotating" OR "sun-like" OR "solar-type") AND abs:Kepler',
            "superflare_tess_catalog": 'abs:superflare AND abs:TESS AND abs:(catalog OR catalogue OR survey)',
            "stellar_engine_search": 'abs:("stellar engine" OR "Shkadov" OR "star lifting") AND abs:(search OR constraint OR detect)',
            "activity_residual": 'abs:(chromospheric OR "Ca II") AND abs:activity AND abs:(anomalous OR "Maunder minimum") AND abs:rotation',
        },
        "by_id": {
            "notsu2019_superflares": ("1904.00142", "superflare", "medium"),
            "okamoto2021_superflares": ("2011.02117", "superflare", "medium"),
            "vasilyev2024_science": ("2412.12265", "superflare", "medium"),
            "tu2020_tess": ("1912.11572", "superflare", "medium"),
            "forgan2013_shkadov": ("1306.1672", "Stellar Engines", "high"),
            "lingam_loeb2020_engines": ("2009.08874", "stellar engine", "low"),
            "lingam_loeb2017_shield": ("1709.05348", "flare", "medium"),
            "boro_saikia2018_hk": ("1803.11123", "chromospheric", "medium"),
        },
        "by_title": {
            "pedersen2017_a_stars": "Do A-type stars flare?",
        },
        "interpretation": ("Notsu / Okamoto / Vasilyev will fire as the natural literature with the "
                           "'occurrence' decoy.  A decoy-free hit with 'technosig' and 'energy_bound' "
                           "boosters is prior art for S59."),
    },
    "g14_postbio_hosts": {
        "question": ("Has anyone executed a technosignature search on hosts where biology is impossible "
                     "(pulsar planets, white-dwarf / neutron-star Dyson rings in Gaia x WISE, brown "
                     "dwarfs, free-floating planets over-luminous for their age), or tested the "
                     "conjunction of a technosignature with an anti-biosignature on one planet?"),
        "target": [r"(pulsar|neutron star|white dwarf|brown dwarf|free-?floating|rogue planet).*(technosignature|dyson|artificial|seti)",
                   r"(technosignature|dyson|seti).*(pulsar|neutron star|white dwarf|brown dwarf|free-?floating|rogue planet)",
                   r"(anti-?biosignature|dead (world|planet|biosphere)|airless).*(technosignature|seti)",
                   r"(technosignature|seti).*(anti-?biosignature|biosignature).*(conjunction|joint|both|together|correlat)",
                   r"post-?biological.*(observ|signature|search)"],
        "decoys": {"natural_disk": r"debris disk around (a |the )?(pulsar|neutron star|white dwarf)|fallback disk|magnetar disk",
                   "ffp_demographics": r"microlensing|free-floating planet (mass function|abundance|occurrence)",
                   "wd_planet_search": r"transit\w* (search|survey) (around|of) white dwarf"},
        "boosters": {"executed": r"we (search|observe|analy|survey)|limits? on|no (signal|detection)", "ring_temp": r"[3-7]00 K|Dyson ring",
                     "conjunction": r"anti-?biosignature|dead"},
        "queries": {
            "pulsar_dyson": 'abs:pulsar AND abs:(Dyson OR technosignature OR "artificial")',
            "wd_ns_dyson_search": 'abs:("white dwarf" OR "neutron star") AND abs:(Dyson OR technosignature) AND abs:(search OR WISE OR infrared)',
            "brown_dwarf_technosig": 'abs:("brown dwarf" OR "Y dwarf") AND abs:(technosignature OR SETI)',
            "ffp_technosig": 'abs:("free-floating" OR "rogue planet") AND abs:(technosignature OR civilization OR artificial)',
            "antibiosignature_techno": 'abs:(antibiosignature OR "anti-biosignature") AND abs:technosignature',
            "postbiological_seti": 'abs:(postbiological OR "post-biological") AND abs:(SETI OR technosignature OR search)',
            "trappist_seti": 'abs:TRAPPIST-1 AND abs:(SETI OR technosignature)',
        },
        "by_id": {
            "osmanov2016_pulsar_rings": ("1505.05131", "pulsar", "medium"),
            "osmanov2018_pulsar_rings": ("1705.04142", "pulsar", "medium"),
            "kayali2025_dyson_ring_lc": ("2412.17086", "Dyson", "low"),
            "cirkovic_bradbury2006": ("astro-ph/0506110", "Galactic Gradients", "high"),
            "romanovskaya2022_hitchhikers": ("2202.03364", "Hitchhikers", "medium"),
            "fast_trappist_2025": ("2509.06310", "TRAPPIST-1", "low"),
            "garrett2024_ai_filter": ("2405.00042", "artificial intelligence", "high"),
            "opatrny2017_bh_dyson": ("1601.02897", "black sun", "high"),
        },
        "by_title": {
            "hsiao2021_bh_dyson": "A Dyson sphere around a black hole",
            "wogan_catling2020_antibio": "When is Chemical Disequilibrium in Earth-like Planetary Atmospheres a Biosignature versus an Anti-biosignature",
        },
        "interpretation": ("The theory papers (Osmanov, Ćirković & Bradbury, Opatrný, Hsiao) will fire; "
                           "an EXECUTED search on these hosts (booster 'executed') or a techno + anti-bio "
                           "conjunction test is prior art for S62/S63."),
    },
    # ------------------------------------------------------------------
    # Added 2026-09-16 to TRACK DOWN one object, not to establish novelty.
    # GROWTH run 35038510064 flagged Kepler-718 b (KOI-897.01 / TIC 268924036 /
    # TOI 4490.01) and stage-2 run 35041932130 measured its TESS-era depth at
    # 30,614 ppm from the light curve against a Kepler-era CATALOGUE depth of
    # 13,884 ppm in the same band.  Before any pixel is fetched, the cheapest
    # decisive question is whether somebody has already written this down: a
    # known blend, a known bad koi_depth, or a documented Kepler-to-TESS depth
    # offset would end the investigation for nothing.
    # ------------------------------------------------------------------
    "g16_transit_depth_offset": {
        "question": ("Is Kepler-718 b / KOI-897.01 recorded anywhere as a blend, a false "
                     "positive, or a revised radius?  Is a KEPLER-ERA to TESS-ERA transit "
                     "DEPTH OFFSET documented as a systematic, and are time-varying transit "
                     "depths (TDV) reported for any confirmed planet?"),
        "target": [r"(Kepler-?718|KOI-?897|KIC ?7849854|TOI-?4490|TIC ?268924036)",
                   r"transit depth.*(variabilit|variation|chang\w+|drift|evolv|secular)",
                   r"(TESS|transit).*(radi(i|us)).*(systematic|offset|underestimat|overestimat|inflat).*(Kepler|ground)",
                   r"(blend|contaminat\w+|nearby eclipsing binary|NEB).*(false positive).*(Kepler|TESS)",
                   r"(depth).*(Kepler).*(TESS).*(compar|discrepan|disagree)"],
        "decoys": {"ttv": r"transit timing variation|TTV|O-C diagram",
                   "starspot": r"spot[- ]crossing|starspot anomal|rotational modulation",
                   "trappist": r"TRAPPIST-1|WASP-|HAT-P-",
                   "atmosphere": r"transmission spectr|atmospher\w+ (escape|retriev)"},
        "boosters": {"object": r"Kepler-?718|KOI-?897|7849854|4490",
                     "depth": r"transit depth|depth ratio|radius ratio|Rp/R",
                     "revision": r"revis\w+|re-?analys|catalog\w+ error|updated"},
        "queries": {
            "kepler718": 'all:"Kepler-718"',
            "koi897": 'all:"KOI-897"',
            "toi4490": 'all:"TOI-4490"',
            "depth_variability": 'abs:"transit depth" AND abs:(variability OR variation OR "long-term")',
            "kepler_tess_radius_offset": 'abs:(TESS AND Kepler) AND abs:("planet radii" OR "radius ratio") AND abs:(systematic OR offset OR discrepancy)',
            "neb_false_positive": 'abs:("nearby eclipsing binary" OR "background eclipsing binary") AND abs:(TESS OR Kepler) AND abs:"false positive"',
            "koi_depth_revision": 'abs:("Kepler Objects of Interest" OR "KOI catalog") AND abs:(revis OR reanalysis OR "uniform fit")',
            "disintegrating_dusty": 'abs:(disintegrating OR "dusty tail" OR evaporating) AND abs:planet AND abs:"transit depth"',
        },
        "by_id": {},
        "by_title": {
            # The expected-deficit reference config/growth.yaml already cites; the
            # sweep checks it exists and says what it actually claims.
            "han2025_tess_radii": "TESS planet radii are systematically underestimated",
        },
        "interpretation": ("A hit naming this object as a blend or false positive ENDS the "
                           "investigation.  A hit documenting a Kepler-to-TESS depth offset as a "
                           "systematic reframes it as a population effect rather than an object.  "
                           "A decoy-free hit reporting genuinely time-varying transit depths in a "
                           "confirmed planet is the nearest published analogue and must be read "
                           "before any claim is made.  Finding NOTHING is not evidence the object "
                           "is interesting --- it is the absence of a cheap exit."),
    },
    "g15_insitu_grains": {
        "question": ("Has anyone read the in-situ composition of interstellar grains (Cassini CDA, "
                     "Ulysses, Stardust ISPE) for refined / non-condensation-sequence outliers, or "
                     "proposed interstellar dust as a sample of sub-micron self-replicating devices?"),
        "target": [r"(interstellar (dust|grain)s?).*(technosignature|artificial|alien|probe|device|replicat)",
                   r"(technosignature|artificial|probe|replicat\w+).*(interstellar (dust|grain)s?|micron-?scale|nano-?scale)",
                   r"(cosmic dust analy[sz]er|CDA|Ulysses|Stardust).*(interstellar).*(composition|outlier|anomal)",
                   r"(technograin|techno-?grain)"],
        "decoys": {"flux_dynamics": r"flux|direction|radiation pressure|Lorentz|heliosphere|dynamics",
                   "meteor": r"meteor|bolide|fireball|CNEOS",
                   "spherule": r"spherule|coal|BeLaU"},
        "boosters": {"technosig": r"technosignature|artificial|replicat", "composition": r"composition|mass spectr|time-of-flight|Mg-rich|silicate",
                     "insitu": r"in situ|in-situ|Cassini|Ulysses|Stardust|IMAP|IDEX|SUDA"},
        "queries": {
            "isd_composition_cassini": 'abs:"interstellar dust" AND abs:(Cassini OR "Cosmic Dust Analyzer") AND abs:composition',
            "isd_technosig": 'abs:"interstellar dust" AND abs:(technosignature OR artificial OR probe OR "self-replicating")',
            "stardust_interstellar": 'abs:Stardust AND abs:interstellar AND abs:(particle OR grain)',
            "ulysses_isd": 'abs:Ulysses AND abs:"interstellar dust"',
            "technograin": 'abs:(technograin OR "techno-grain" OR "micron-scale technosignature")',
            "imap_idex_dust": 'abs:(IMAP OR IDEX OR "Interstellar Mapping") AND abs:dust',
        },
        "by_id": {
            "lacki2026_dust_to_dust": ("2606.08373", "Dust", "medium"),
            "kruger2015_ulysses_16yr": ("1510.06180", "Ulysses", "low"),
            "lunar_regolith_micron_2026": ("2606.24028", "regolith", "low"),
        },
        "by_title": {
            "altobelli2016_cda": "Flux and composition of interstellar dust at Saturn from Cassini's Cosmic Dust Analyzer",
            "westphal2014_ispe": "Evidence for interstellar origin of seven dust particles collected by the Stardust spacecraft",
        },
        "interpretation": ("Lacki 2026 (technograins dispersed to the ISM) is the nearest framing and "
                           "will fire.  A decoy-free hit reading the in-situ COMPOSITION record for "
                           "artificial outliers is prior art for S58; none is expected."),
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
# Keyless fallbacks.  The repository's ADS_TOKEN secret turned out to be unset
# (run 2: every ADS attempt recorded "no ADS_TOKEN"), so two routes earlier
# sweeps used without a key come first: the arXiv ABSTRACT PAGE for an id
# (lzlit: "no API, no 429") and OpenAlex for keyword / title searches
# (necrolit, zacklit, cenotaph_recon).  ADS remains last, if a token appears.
# --------------------------------------------------------------------------
def parse_arxiv_abs_page(aid: str, page: str) -> str:
    """Pure function: the arXiv abstract page -> one Atom entry (or an empty feed)."""
    m = re.search(r"<title>(.*?)</title>", page, re.S)
    title = " ".join(html.unescape(m.group(1)).split()) if m else ""
    title = re.sub(r"^\[[^\]]+\]\s*", "", title)
    ab = re.search(r'name="citation_abstract"\s+content="(.*?)"', page, re.S)
    summ = " ".join(html.unescape(ab.group(1)).split()) if ab else ""
    head = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><source>arxiv-abs-page</source>'
    if not title:
        return head + "</feed>"
    return (head + f"<entry><id>http://arxiv.org/abs/{aid}</id><title>{_xml_escape(title)}</title>"
            f"<summary>{_xml_escape(summ)}</summary></entry></feed>")


def arxiv_abs_fetch(aid: str, dest: pathlib.Path) -> bool:
    tmp = dest.with_suffix(".abs.html")
    ok = get(f"https://arxiv.org/abs/{aid}", tmp, tries=2, pause=3.0)
    if not ok:
        return False
    dest.write_text(parse_arxiv_abs_page(aid, tmp.read_text(errors="ignore")))
    tmp.unlink(missing_ok=True)
    return "<entry>" in dest.read_text(errors="ignore")


def arxiv_to_openalex_query(q: str) -> str:
    """Pure function: arXiv-API syntax -> OpenAlex plain-text search terms.

    OpenAlex's ``search`` parameter has no field prefixes and no boolean
    operators, so fields are dropped, quotes kept as phrases, AND/OR removed.
    The translation is lossy and is recorded verbatim in summary.json."""
    out = re.sub(r"\b(abs|ti|all|au):", "", q)
    out = re.sub(r"\b(AND|OR|NOT)\b", " ", out)
    out = out.replace("(", " ").replace(")", " ")
    return " ".join(out.split())


def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def openalex_to_atom(works: list[dict]) -> str:
    """Pure function: OpenAlex works -> the Atom shape ``_entries`` reads."""
    parts = ['<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">', "<source>openalex</source>"]
    for w in works:
        ident = ""
        urls = []
        for loc in (w.get("locations") or []) + [w.get("primary_location") or {}]:
            urls.append(str((loc or {}).get("landing_page_url") or ""))
            urls.append(str((loc or {}).get("pdf_url") or ""))
        for u in urls:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})", u)
            if m:
                ident = f"http://arxiv.org/abs/{m.group(1)}"
                break
        ident = ident or str(w.get("id") or "")
        parts.append(f"<entry><id>{_xml_escape(ident)}</id><title>{_xml_escape(w.get('display_name') or '')}</title>"
                     f"<summary>{_xml_escape(_openalex_abstract(w.get('abstract_inverted_index')))}</summary></entry>")
    parts.append("</feed>")
    return "".join(parts)


def openalex_fetch(q: str, dest: pathlib.Path, rows: int) -> bool:
    url = (f"{OPENALEX_API}?search={urllib.parse.quote(q)}&per-page={min(rows, 50)}"
           "&select=id,display_name,abstract_inverted_index,primary_location,locations"
           "&mailto=trimcrae@gmail.com")
    tmp = dest.with_suffix(".openalex.json")
    if not get(url, tmp, tries=2, pause=1.5):
        return False
    try:
        works = json.loads(tmp.read_text(errors="ignore")).get("results") or []
    except Exception:  # noqa: BLE001
        return False
    dest.write_text(openalex_to_atom(works))
    tmp.unlink(missing_ok=True)
    STATUS[-1]["source"] = "openalex"
    STATUS[-1]["n_docs"] = len(works)
    _write_summary()
    return True


# --------------------------------------------------------------------------
# ADS fallback: the same question asked of a second service.
# --------------------------------------------------------------------------
def arxiv_to_ads_query(q: str) -> str:
    """Translate an arXiv-API search string into ADS syntax (pure function).

    arXiv: ``abs:foo AND abs:(bar OR baz)``, ``ti:"..."``, ``all:X``.
    ADS:   fields joined by implicit AND, ``abs:``/``title:`` phrases quoted,
           ``all:`` dropped (unfielded term).  OR and parentheses pass through.
    """
    out = q.replace(" AND ", " ")
    out = re.sub(r"\bti:", "title:", out)
    out = re.sub(r"\ball:", "", out)
    return out.strip()


def _xml_escape(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def ads_to_atom(docs: list[dict]) -> str:
    """Render ADS documents in the minimal Atom shape ``_entries`` reads.

    The ``<id>`` carries the arXiv id when ADS knows one (``identifier``
    list), else the bibcode, so the concept scan keys hits the same way."""
    parts = ['<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">',
             "<source>ads</source>"]
    for d in docs:
        ident = ""
        for x in d.get("identifier") or []:
            m = re.match(r"arXiv:(.+)", x)
            if m:
                ident = f"http://arxiv.org/abs/{m.group(1)}"
                break
        ident = ident or f"bibcode:{d.get('bibcode', '')}"
        title = " ".join(d.get("title") or [])
        parts.append(f"<entry><id>{_xml_escape(ident)}</id><title>{_xml_escape(title)}</title>"
                     f"<summary>{_xml_escape(d.get('abstract') or '')}</summary></entry>")
    parts.append("</feed>")
    return "".join(parts)


def ads_fetch(q: str, dest: pathlib.Path, rows: int) -> bool:
    """Query ADS with the token; write Atom-shaped output to ``dest``."""
    if not ADS_TOKEN:
        STATUS.append({"url": "ads:" + q, "dest": dest.name, "ok": False,
                       "attempts": [{"try": 0, "http": None, "error": "no ADS_TOKEN"}]})
        _write_summary()
        return False
    url = (f"{ADS_API}?q={urllib.parse.quote(q)}&rows={rows}"
           "&fl=bibcode,title,abstract,identifier&sort=score+desc")
    attempts: list[dict] = []
    for i in range(TRIES):
        try:
            req = urllib.request.Request(url, headers={**UA, "Authorization": f"Bearer {ADS_TOKEN}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8", "ignore"))
            docs = (data.get("response") or {}).get("docs") or []
            dest.write_text(ads_to_atom(docs))
            attempts.append({"try": i + 1, "http": 200, "n_docs": len(docs)})
            STATUS.append({"url": url.split("&fl=")[0], "dest": dest.name, "ok": True,
                           "bytes": dest.stat().st_size, "source": "ads", "attempts": attempts})
            _write_summary()
            print(f"  ok  ADS {len(docs):>3} docs  {dest.name}")
            time.sleep(1.0)
            return True
        except urllib.error.HTTPError as exc:
            attempts.append({"try": i + 1, "http": exc.code, "error": str(exc)[:200]})
            print(f"  ADS try {i + 1}/{TRIES} HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001
            attempts.append({"try": i + 1, "http": None, "error": repr(exc)[:200]})
            print(f"  ADS try {i + 1}/{TRIES} failed: {exc!r}")
        time.sleep(5.0 * (i + 1))
    STATUS.append({"url": url.split("&fl=")[0], "dest": dest.name, "ok": False, "source": "ads",
                   "attempts": attempts})
    _write_summary()
    return False


# --------------------------------------------------------------------------
# Fetchers: arXiv first, ADS when arXiv refuses.
# --------------------------------------------------------------------------
def arxiv_query(group: str, name: str, q: str) -> None:
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&start=0&max_results={MAX_RESULTS}&sortBy=relevance&sortOrder=descending")
    dest = OUT / f"arxiv_q_{group}__{name}.atom"
    if not get(url, dest):
        if not openalex_fetch(arxiv_to_openalex_query(q), dest, MAX_RESULTS):
            ads_fetch(arxiv_to_ads_query(q), dest, MAX_RESULTS)


def arxiv_title(group: str, name: str, title: str) -> None:
    q = 'ti:"' + title.replace('"', "") + '"'
    url = (f"{ARXIV_API}?search_query={urllib.parse.quote(q)}"
           f"&start=0&max_results=5&sortBy=relevance&sortOrder=descending")
    dest = OUT / f"arxiv_q_title_{group}__{name}.atom"
    if not get(url, dest):
        if not openalex_fetch(title, dest, 5):
            ads_fetch(arxiv_to_ads_query(q), dest, 5)


def arxiv_ids(group: str, named: dict[str, tuple]) -> None:
    """One id_list request per group (not one per paper), then split the feed
    into the per-paper files the id/title check reads; ADS per id on refusal."""
    if not named:
        return
    ids = [aid for aid, _, _ in named.values()]
    url = (f"{ARXIV_API}?search_query=&id_list={','.join(ids)}&start=0"
           f"&max_results={len(ids)}")
    combined = OUT / f"arxiv_idlist_{group}.atom"
    if get(url, combined):
        text = combined.read_text(errors="ignore")
        by_id = {}
        for m in re.finditer(r"<entry>.*?</entry>", text, re.S):
            e = m.group(0)
            mid = re.search(r"<id>http://arxiv.org/abs/([^<v]+)", e)
            if mid:
                by_id[mid.group(1)] = e
        for name, (aid, _, _) in named.items():
            e = by_id.get(aid)
            (OUT / f"arxiv_id_{group}__{name}.atom").write_text(
                '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
                + (e or "") + "</feed>")
        return
    for name, (aid, _, _) in named.items():
        dest = OUT / f"arxiv_id_{group}__{name}.atom"
        if not arxiv_abs_fetch(aid, dest):
            ads_fetch(f"identifier:arXiv:{aid}", dest, 1)


def _entries(text: str):
    for m in re.finditer(r"<entry>(.*?)</entry>", text, re.S):
        e = m.group(1)
        aid = (re.search(r"<id>(.*?)</id>", e, re.S) or [None, ""])[1].strip()
        title = " ".join((re.search(r"<title>(.*?)</title>", e, re.S) or [None, ""])[1].split())
        summ = " ".join((re.search(r"<summary>(.*?)</summary>", e, re.S) or [None, ""])[1].split())
        yield aid, html.unescape(title), html.unescape(summ)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def id_title_check(out_dir: pathlib.Path | None = None) -> dict:
    """Did each asserted id resolve to the expected title?  Did each title search
    return the expected title?  Recorded, not assumed; a mismatch is a finding."""
    out_dir = out_dir or OUT
    out: dict = {"by_id": {}, "by_title": {}, "n_id_mismatch": 0, "n_title_search_miss": 0}
    for group, spec in GROUPS.items():
        for name, (aid, frag, conf) in spec["by_id"].items():
            key = f"{group}__{name}"
            p = out_dir / f"arxiv_id_{key}.atom"
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
            p = out_dir / f"arxiv_q_title_{key}.atom"
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
        "the id is wrong and must not be cited (several ids in this file are marked "
        "prior_confidence 'low' for exactly this reason).  by_title.found False means no "
        "arXiv entry has that title (pre-arXiv or journal-only papers are expected here); "
        "matched_ids are the ids the record itself supplies for the title.")
    (out_dir / "id_title_check.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# Decoy-aware concept scan, per group, over EVERY fetched abstract (a paper
# fetched for one group can be prior art for another).
# --------------------------------------------------------------------------
def _atom_files(out_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in out_dir.glob("arxiv_*.atom") if not p.name.startswith("arxiv_idlist_"))


def _file_group(name: str) -> str | None:
    m = re.match(r"arxiv_(?:q_title|q|id)_(g\d+_[a-z_]+?)__", name)
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
            "prior art.  Decoy tags are flags, not vetoes.  The novelty position of each "
            "signature in docs/necrofrontier.md is to be read against its group's "
            "decoy_free_hits and booster-rich hits; it is not established until this file "
            "exists and has been read."),
    }
    (out_dir / "concept_scan.json").write_text(json.dumps(out, indent=2))
    return out


def _finish() -> None:
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


def main(argv: list[str] | None = None) -> None:
    import signal
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if "--scan-only" in argv:
        # Re-derive the check and the scan from whatever files exist (a run
        # killed at its deadline leaves the fetched files but no scan).  The
        # fetch record of that run is kept: run 3 overwrote summary.json with
        # this process's empty STATUS and reported "0 / 0 fetches".
        prev = OUT / "summary.json"
        if prev.exists():
            try:
                STATUS.extend(json.loads(prev.read_text()).get("status") or [])
            except Exception:  # noqa: BLE001
                pass
        _finish()
        return

    def _term(signum, frame):  # noqa: ARG001
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, _term)
    try:
        for group, spec in GROUPS.items():
            if ONLY_GROUPS and group not in ONLY_GROUPS:
                continue
            print(f"==== {group} ====")
            print("== named papers by id (title-checked; one id_list call per group) ==")
            for name, (aid, _, conf) in spec["by_id"].items():
                print(f"-- {name} ({aid}, prior confidence {conf})")
            arxiv_ids(group, spec["by_id"])
            print("== named papers by title ==")
            for name, title in spec["by_title"].items():
                print(f"-- {name}")
                arxiv_title(group, name, title)
            print("== keyword sweeps ==")
            for name, q in spec["queries"].items():
                print(f"-- {name}")
                arxiv_query(group, name, q)
    finally:
        _finish()


if __name__ == "__main__":
    main()

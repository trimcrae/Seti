#!/usr/bin/env python3
"""SEXTANTLIT — prior-art / novelty adjudication for SEXTANT.

SEXTANT's claim under test: *screen Gaia's per-observation asteroid astrometry*
(`gaiafpr.sso_observation`, 46.26 M observations of 156,823 objects;
`gaiadr3.sso_observation`, 23.34 M of 158,152) *for a POPULATION of objects whose
milliarcsecond ephemeris residuals — computed against INDEPENDENT (JPL/MPC)
orbits and projected on the Gaia scan direction — carry an anomalous,
non-gravitational, non-binary signature, read as a technosignature.*

Five questions, in order of how badly each could sink the channel (mirrored in
the query-name prefixes below):

  Q1  Has anyone searched Gaia SSO astrometric residuals for ANOMALOUS /
      non-gravitational / unexplained signatures, as opposed to using them for
      orbit improvement, binarity, masses or reference-frame work?
  Q2  Liberato & Tanga (A&A 688, A50 = arXiv:2406.07195, "Binary asteroid
      candidates in Gaia DR3 astrometry") and its FPR successor
      (arXiv:2605.22702, "Follow the wobble"): exactly what residual statistic,
      over what data, and how much of SEXTANT's method does it occupy?
  Q3  Has ANY technosignature paper used asteroid astrometry at all?
  Q4  Who else computed O-C residuals of Gaia asteroid observations against
      INDEPENDENT (JPL/MPC/ephemeris) orbits, and what is in the tails?
      Published outlier lists are prior art AND our contamination catalogue.
  Q5  Non-gravitational acceleration in the main belt generally: dark comets,
      Yarkovsky detection catalogues (Del Vigna and successors), and their
      sensitivity versus mas astrometry over the 2014-2020 Gaia baseline.

The dev sandbox has NO egress at all (every host, including export.arxiv.org,
api.openalex.org, ui.adsabs.harvard.edu, www.aanda.org and vizier.cds.unistra.fr,
returns "CONNECT tunnel failed, response 403"; WebFetch is blocked for the same
hosts).  The runner has egress -- CLAUDE.md acquisition pattern.  Everything is
written under results/sextantlit/ and committed back to the branch.

Outputs
  queries.json          every query issued: URL, HTTP status, totalResults, ids
  ax_q_<name>.atom      raw arXiv Atom per query
  hits.tsv              flat table: question, query, arXiv id, date, title
  zero_hits.json        the queries that returned nothing -- the novelty evidence
  ids_verified.json     id -> REAL title/authors/date (nothing cited unverified)
  oa_*.json             OpenAlex resolutions and citing sets
  txt_<name>.txt        full texts of the papers whose content decides a verdict
  term_occupancy.tsv    term x document matrix, counted over those full texts
  vizier_*.txt          VizieR ReadMe / metadata for J/A+A/688/A50 and friends
  summary.json          counts, zero-hit statements, failures
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/sextantlit")
OUT.mkdir(parents=True, exist_ok=True)
MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-sextantlit/1.0 (mailto:{MAIL})"}

ARXIV = "https://export.arxiv.org/api/query?"
OPENALEX = "https://api.openalex.org/"
VIZIER = "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/"

STATUS: list[dict] = []
T0 = time.time()
DEADLINE = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0


def out_of_time(margin: float = 60.0) -> bool:
    return (time.time() - T0) > (DEADLINE - margin)


def get(url: str, dest: pathlib.Path | None = None, tries: int = 3,
        pause: float = 3.0) -> bytes | None:
    """GET with retries; record status. Returns bytes or None."""
    last = ""
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=90) as r:
                data = r.read()
                code = r.status
            if dest is not None:
                dest.write_bytes(data)
            STATUS.append({"url": url, "dest": dest.name if dest else None,
                           "http": code, "bytes": len(data), "ok": True})
            print(f"OK   {code} {len(data):9d}B  {url[:130]}", flush=True)
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        print(f"RETRY({i+1}) {last}  {url[:130]}", flush=True)
        time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "dest": dest.name if dest else None,
                   "http": None, "ok": False, "error": last})
    print(f"FAIL {last}  {url[:130]}", flush=True)
    return None


# --------------------------------------------------------------------------
# arXiv Atom parsing
# --------------------------------------------------------------------------
def total_results(blob: bytes) -> int | None:
    m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<",
                  blob.decode("utf-8", "replace"))
    return int(m.group(1)) if m else None


def entries(blob: bytes) -> list[dict]:
    txt = blob.decode("utf-8", "replace")
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", txt, re.S):
        def f(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
            return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip()) if m else ""
        auth = [html.unescape(re.sub(r"\s+", " ", a).strip())
                for a in re.findall(r"<author>\s*<name>(.*?)</name>", e, re.S)]
        cat = re.findall(r'<category term="([^"]+)"', e)
        out.append({"id": f("id").replace("http://arxiv.org/abs/", "")
                                 .replace("https://arxiv.org/abs/", ""),
                    "title": f("title"), "published": f("published")[:10],
                    "updated": f("updated")[:10],
                    "authors": auth[:8], "cats": cat[:6],
                    "summary": f("summary")})
    return out


# --------------------------------------------------------------------------
# THE QUERY BATTERY.  (question, name, search_query)
# Every query is recorded with its totalResults; zero-result queries are the
# evidence for any "unoccupied" claim, so they are deliberately phrased to be
# the *easiest* way anyone would have written the paper we fear exists.
# --------------------------------------------------------------------------
Q: list[tuple[str, str, str]] = [
    # ================= Q1  anomalous signatures IN Gaia SSO residuals
    ("Q1", "gaia_sso_residual_anomalous",
     'abs:"Gaia" AND abs:"asteroid" AND abs:"residuals" AND abs:"anomalous"'),
    ("Q1", "gaia_astrometry_nongrav_asteroid",
     'abs:"Gaia" AND abs:"astrometry" AND abs:"non-gravitational" AND abs:"asteroid"'),
    ("Q1", "gaia_asteroid_unexplained",
     'abs:"Gaia" AND abs:"asteroid" AND abs:"unexplained"'),
    ("Q1", "gaia_sso_outlier_detection",
     'abs:"Gaia" AND abs:"solar system objects" AND abs:"outlier"'),
    ("Q1", "gaia_epoch_astrometry_asteroid_anomaly",
     'abs:"epoch astrometry" AND abs:"asteroid" AND abs:"anomaly"'),
    ("Q1", "astrometric_residual_anomaly_minor_planet",
     'abs:"astrometric residuals" AND abs:"minor planet"'),
    ("Q1", "astrometric_residuals_anomalous",
     'abs:"astrometric residuals" AND abs:"anomalous"'),
    ("Q1", "ephemeris_residual_phrase", 'all:"ephemeris residual"'),
    ("Q1", "ephemeris_residuals_phrase", 'all:"ephemeris residuals"'),
    ("Q1", "ephemeris_residual_asteroid",
     'all:"ephemeris residual" AND all:"asteroid"'),
    ("Q1", "gaia_sso_observation_table",
     'all:"sso_observation"'),
    ("Q1", "gaia_fpr_sso_residuals",
     'abs:"Focused Product Release" AND abs:"asteroid" AND abs:"residuals"'),
    ("Q1", "anomalous_acceleration_population_asteroids",
     'abs:"anomalous acceleration" AND abs:"asteroids"'),
    ("Q1", "unmodelled_acceleration_asteroid",
     'abs:"unmodeled acceleration" OR abs:"unmodelled acceleration"'),
    ("Q1", "gaia_asteroid_machine_learning_anomaly",
     'abs:"Gaia" AND abs:"asteroids" AND abs:"anomaly detection"'),
    ("Q1", "along_scan_residual_asteroid",
     'abs:"along-scan" AND abs:"asteroid"'),
    ("Q1", "gaia_sso_astrometry_search_signature",
     'abs:"Gaia" AND abs:"asteroid astrometry" AND abs:"signature"'),
    ("Q1", "residual_population_screen_small_bodies",
     'abs:"residuals" AND abs:"population" AND abs:"small bodies" AND abs:"survey"'),

    # ================= Q2  Liberato & Tanga and the binary-astrometry line
    ("Q2", "liberato_binary_gaia", 'all:Liberato AND all:"binary" AND all:"Gaia"'),
    ("Q2", "binary_asteroid_candidates_gaia_dr3",
     'ti:"Binary asteroid candidates in Gaia DR3 astrometry"'),
    ("Q2", "follow_the_wobble", 'ti:"Follow the wobble"'),
    ("Q2", "tanga_gaia_asteroid", 'all:Tanga AND all:"Gaia" AND all:"asteroid"'),
    ("Q2", "astrometric_detection_binary_asteroids",
     'ti:"Astrometric detection of binary asteroids"'),
    ("Q2", "photocentre_offset_binary_asteroid",
     'abs:"photocenter" AND abs:"binary asteroid"'),
    ("Q2", "photocentre_wobble_astrometry",
     'abs:"wobble" AND abs:"astrometry" AND abs:"asteroid"'),
    ("Q2", "binary_asteroid_gaia_occultation_confirm",
     'abs:"binary asteroid" AND abs:"occultation" AND abs:"Gaia"'),
    ("Q2", "gaia_satellite_asteroid_astrometric_signature",
     'abs:"satellite" AND abs:"asteroid" AND abs:"astrometric signature"'),
    ("Q2", "periodogram_residuals_asteroid",
     'abs:"periodogram" AND abs:"residuals" AND abs:"asteroid"'),
    ("Q2", "false_discovery_rate_asteroid_detection",
     'abs:"false discovery rate" AND abs:"asteroid"'),
    ("Q2", "gaia_binary_asteroid_candidates_catalogue",
     'abs:"binary asteroid candidates" AND abs:"Gaia"'),
    ("Q2", "arecibo_4337_gaia_binary", 'all:"Arecibo" AND all:"binary" AND all:"Gaia"'),

    # ================= Q3  technosignatures using asteroid astrometry
    ("Q3", "technosignature_asteroid_astrometry",
     'abs:"technosignature" AND abs:"astrometry"'),
    ("Q3", "technosignature_asteroid",
     'abs:"technosignature" AND abs:"asteroid"'),
    ("Q3", "technosignature_minor_planet",
     'abs:"technosignature" AND abs:"minor planet"'),
    ("Q3", "technosignature_gaia", 'abs:"technosignature" AND abs:"Gaia"'),
    ("Q3", "seti_asteroid_astrometry", 'abs:"SETI" AND abs:"asteroid astrometry"'),
    ("Q3", "artificial_object_asteroid_astrometry",
     'abs:"artificial" AND abs:"asteroid" AND abs:"astrometry"'),
    ("Q3", "nongrav_technosignature", 'abs:"non-gravitational" AND abs:"technosignature"'),
    ("Q3", "solar_system_technosignatures_lazio", 'ti:"Solar System Technosignatures"'),
    ("Q3", "technosignatures_in_solar_system", 'ti:"Technosignatures in the Solar System"'),
    ("Q3", "anomalous_asteroid_accelerations_lazio",
     'ti:"Anomalous Asteroid Accelerations"'),
    ("Q3", "lazio_mahabal", 'all:Lazio AND all:Mahabal'),
    ("Q3", "dark_comet_technosignature", 'abs:"dark comet" AND abs:"technosignature"'),
    ("Q3", "phobos1_1998ky26", 'all:"1998 KY26" OR ti:"Phobos 1"'),
    ("Q3", "artificial_object_solar_system", 'abs:"artificial object" AND abs:"solar system"'),
    ("Q3", "self_replicating_solar_system_technosignature",
     'abs:"self-replicating" AND abs:"solar system" AND abs:"technosignature"'),
    ("Q3", "lurkers_coorbital_seti", 'all:"lurkers" AND all:"SETI"'),
    ("Q3", "ellery_selfreplicating", 'all:Ellery AND all:"self-replicating"'),
    ("Q3", "technosignature_orbital_dynamics",
     'abs:"technosignature" AND abs:"orbital" AND abs:"dynamics"'),
    ("Q3", "maneuver_detection_technosignature", 'abs:"maneuver" AND abs:"technosignature"'),
    ("Q3", "probe_population_solar_system_search",
     'abs:"probes" AND abs:"solar system" AND abs:"search" AND abs:"population"'),

    # ================= Q4  O-C against INDEPENDENT orbits; outlier lists
    ("Q4", "gaia_asteroid_orbit_assessment_jpl",
     'abs:"Gaia" AND abs:"asteroid" AND abs:"orbital solution" AND abs:"assessment"'),
    ("Q4", "gaia_fpr_orbit_determination_statistical",
     'ti:"Asteroid Orbit Determination Using Gaia FPR"'),
    ("Q4", "gaia_crf3_planetary_ephemerides_asteroids",
     'abs:"Gaia-CRF3" AND abs:"planetary ephemerides"'),
    ("Q4", "gaia_dr2_inpop_asteroids",
     'abs:"INPOP" AND abs:"asteroid" AND abs:"Gaia"'),
    ("Q4", "asteroid_observations_reference_frame_rotation",
     'abs:"asteroid" AND abs:"reference frame" AND abs:"rotation" AND abs:"ephemerides"'),
    ("Q4", "center_of_light_offset_phase_gaia",
     'abs:"center of light" AND abs:"asteroid"'),
    ("Q4", "reweighting_asteroid_orbit_determination_gaia",
     'abs:"weighting" AND abs:"asteroid" AND abs:"orbit determination" AND abs:"Gaia"'),
    ("Q4", "gaia_asteroid_mass_determination",
     'abs:"Gaia" AND abs:"asteroid" AND abs:"mass determination"'),
    ("Q4", "gaia_dr3_dynamical_masses_asteroids",
     'abs:"dynamical masses" AND abs:"asteroids" AND abs:"Gaia"'),
    ("Q4", "outlier_rejection_asteroid_astrometry",
     'abs:"outlier" AND abs:"rejection" AND abs:"asteroid astrometry"'),
    ("Q4", "star_catalog_bias_asteroid_astrometry",
     'abs:"star catalog" AND abs:"bias" AND abs:"asteroid astrometry"'),
    ("Q4", "debiasing_asteroid_astrometry", 'abs:"debiasing" AND abs:"astrometry"'),
    ("Q4", "astrometric_error_model_minor_planet",
     'abs:"astrometric" AND abs:"error model" AND abs:"minor planet"'),
    ("Q4", "gaia_occultation_validation_orbits",
     'abs:"occultation" AND abs:"Gaia" AND abs:"orbit" AND abs:"validation"'),
    ("Q4", "gaia_asteroid_ephemeris_prediction_accuracy",
     'abs:"Gaia" AND abs:"asteroid" AND abs:"ephemeris" AND abs:"accuracy"'),
    ("Q4", "residual_tails_non_gaussian_astrometry",
     'abs:"non-Gaussian" AND abs:"residuals" AND abs:"astrometry" AND abs:"asteroid"'),

    # ================= Q5  non-grav acceleration in the belt; sensitivity
    ("Q5", "yarkovsky_gaia_dr3_fpr",
     'abs:"Yarkovsky" AND abs:"Gaia"'),
    ("Q5", "yarkovsky_main_belt_detection",
     'abs:"Yarkovsky" AND abs:"main belt" AND abs:"detection"'),
    ("Q5", "del_vigna_yarkovsky_nea",
     'ti:"Detecting the Yarkovsky effect among near-Earth asteroids"'),
    ("Q5", "yarkovsky_catalogue_updated",
     'abs:"Yarkovsky" AND abs:"catalog" AND abs:"near-Earth asteroids"'),
    ("Q5", "yarkovsky_spurious_detections",
     'abs:"Yarkovsky" AND abs:"spurious"'),
    ("Q5", "dark_comets", 'abs:"dark comet"'),
    ("Q5", "dark_comet_populations_seligman",
     'abs:"dark comets" AND abs:"populations"'),
    ("Q5", "nongrav_acceleration_small_asteroids",
     'abs:"nongravitational accelerations" AND abs:"asteroids"'),
    ("Q5", "1998_sh2_activity", 'all:"1998 SH2"'),
    ("Q5", "2006_vc_nongrav", 'all:"2006 VC" AND all:"non-gravitational"'),
    ("Q5", "srp_area_to_mass_asteroid",
     'abs:"solar radiation pressure" AND abs:"area-to-mass" AND abs:"asteroid"'),
    ("Q5", "yarkovsky_main_belt_families_drift",
     'abs:"Yarkovsky" AND abs:"asteroid family" AND abs:"drift"'),
    ("Q5", "nongrav_detection_sensitivity_astrometry",
     'abs:"non-gravitational" AND abs:"detection" AND abs:"astrometry" AND abs:"sensitivity"'),
    ("Q5", "main_belt_comet_activity_astrometry",
     'abs:"main-belt comet" AND abs:"non-gravitational"'),
    ("Q5", "a2_transverse_acceleration_fit",
     'abs:"transverse acceleration" AND abs:"orbit fit"'),

    # ================= Q0  the SEXTANT phrase space itself (novelty probes)
    ("Q0", "population_artificial_residual_screen",
     'abs:"population" AND abs:"artificial" AND abs:"residuals"'),
    ("Q0", "milliarcsecond_astrometry_technosignature",
     'abs:"milliarcsecond" AND abs:"technosignature"'),
    ("Q0", "gaia_asteroid_technosignature_population",
     'abs:"Gaia" AND abs:"asteroid" AND abs:"technosignature"'),
    ("Q0", "artificial_population_astrometric_screen",
     'abs:"artificial" AND abs:"astrometric" AND abs:"population" AND abs:"screen"'),
    ("Q0", "momentum_ceiling_asteroid", 'all:"momentum" AND all:"ceiling" AND all:"asteroid"'),
    ("Q0", "thrust_signature_minor_planet",
     'abs:"thrust" AND abs:"minor planet"'),
    ("Q0", "station_keeping_signature_detection",
     'abs:"station-keeping" AND abs:"detection" AND abs:"astrometry"'),
]

MAX = 60


def run_queries() -> tuple[list[dict], dict]:
    rows, zero = [], {}
    for question, name, sq in Q:
        if out_of_time():
            print("### deadline: stopping query battery", flush=True)
            break
        url = (ARXIV + urllib.parse.urlencode(
            {"search_query": sq, "start": 0, "max_results": MAX,
             "sortBy": "relevance"}))
        blob = get(url, OUT / f"ax_q_{name}.atom")
        if blob is None:
            zero[name] = {"question": question, "query": sq, "status": "QUERY_FAILED"}
            continue
        tot = total_results(blob)
        es = entries(blob)
        if not es:
            zero[name] = {"question": question, "query": sq,
                          "status": "ZERO_RESULTS", "totalResults": tot}
        for e in es:
            rows.append({"question": question, "query_name": name, "query": sq,
                         "id": e["id"], "published": e["published"],
                         "title": e["title"], "authors": "; ".join(e["authors"]),
                         "cats": ",".join(e["cats"]), "summary": e["summary"]})
        STATUS[-1].update({"query_name": name, "question": question,
                           "search_query": sq, "totalResults": tot,
                           "n_entries": len(es)})
    return rows, zero


# --------------------------------------------------------------------------
# IDs harvested in-sandbox via WebSearch.  NONE of them may be cited until the
# runner has resolved them to a real title/author list here.
# --------------------------------------------------------------------------
VERIFY_IDS = [
    "2406.07195",   # suspected Liberato+ Binary asteroid candidates in Gaia DR3 astrometry
    "2605.22702",   # suspected Liberato+ Follow the wobble (Gaia FPR)
    "2606.13353",   # suspected binary asteroid candidates via stellar occultations
    "2411.09750",   # suspected Dziadura+ Yarkovsky with Gaia DR3 and FPR
    "2310.14699",   # suspected Gaia FPR: Asteroid orbital solution, properties & assessment
    "2203.01586",   # suspected Gaia-DR2 asteroid observations and INPOP
    "2211.04498",   # suspected Astrometric detection of binary asteroids (MNRAS)
    "1805.05947",   # suspected Del Vigna+ Yarkovsky among NEAs from astrometric data
    "2212.08115",   # suspected Seligman+ Dark comets? unexpectedly large nongrav accels
    "2310.02733",   # suspected seasonally varying outgassing / dark comet accelerations
    "2407.01839",   # suspected dynamical origins of the dark comets
    "2606.01288",   # suspected Hibberd, Crowl, Gomez de Olea, Loeb: 1998 KY26 = Phobos 1?
    "2606.13797",   # suspected Lazio, Solar System Technosignatures
    "2508.16825",   # suspected Davenport+, Technosignature Searches of Interstellar Objects
    "2510.00082",   # suspected Ellery, self-replicating probe technosignatures
    "2605.21093",   # suspected The Search for Technosignatures: a Review of Possibilities
    "2604.08820",   # suspected re-weighting scheme, asteroid OD with Gaia
    "2405.20176",   # suspected SNAPS asteroid population outlier detection
    "1804.09379",   # suspected Gaia DR2 observations of solar system objects
    "2206.05561",   # suspected Gaia DR3 the Solar System survey
    "2608.16060",   # suspected Gaia DR3 limits on stellar engine technosignatures
    "2508.00056",   # suspected identifying anomalous asteroids via predictive modeling
]

# Papers whose CONTENT decides a verdict.  Fetched in full and term-counted.
FULLTEXT_TARGETS = {
    "liberato_dr3_binaries": "2406.07195",     # Q2 -- the nearest neighbour
    "follow_the_wobble_fpr": "2605.22702",     # Q2 -- the FPR successor
    "dziadura_yarkovsky_gaia": "2411.09750",   # Q1/Q5 -- non-grav fit ON Gaia data
    "gaia_fpr_orbits": "2310.14699",           # Q4 -- Gaia orbits vs independent orbits
    "delvigna_yarkovsky": "1805.05947",        # Q5 -- the spurious-detection list
    "dark_comets_seligman": "2212.08115",      # Q5
    "lazio_solar_system_technosig": "2606.13797",  # Q3 -- the competitor review
    "ky26_phobos1": "2606.01288",              # Q3 -- artificial-object claim on a dark comet
    "gaia_dr2_inpop": "2203.01586",            # Q4
    "astrometric_binary_asteroids": "2211.04498",  # Q2 -- method precursor
    "ellery_selfrep": "2510.00082",            # Q3 -- LOOM's established baseline
    "snaps_outliers": "2405.20176",            # Q1 -- population outlier detection, no residuals
}

# Term occupancy is counted over exactly the FULLTEXT_TARGETS corpus above.
# Any "unoccupied" claim in docs/sextant-priorart.md must name this corpus.
TERMS = [
    "technosignature", "technosignatures", "extraterrestrial", "SETI",
    "artificial", "probe", "self-replicating", "von Neumann",
    "ephemeris residual", "astrometric residual", "post-fit residual",
    "non-gravitational", "nongravitational", "Yarkovsky", "solar radiation pressure",
    "area-to-mass", "thrust", "manoeuvre", "maneuver",
    "along-scan", "across-scan", "scan direction", "position_angle_scan",
    "sso_observation", "Focused Product Release", "milliarcsecond",
    "JPL", "Horizons", "SBDB", "Minor Planet Center", "OrbFit", "AstDyS",
    "binary", "satellite", "photocenter", "photocentre", "wobble",
    "population", "family", "clustering", "outlier", "anomalous", "anomaly",
    "secular", "trend", "drift", "linear systematics", "dark comet",
    "false discovery rate", "LSST", "Rubin",
]

OA_TITLES = [
    ("liberato_dr3", "Binary asteroid candidates in Gaia DR3 astrometry"),
    ("follow_wobble", "Follow the wobble: Statistical methods to detect astrometric "
                      "binary asteroids in Gaia FPR"),
    ("dziadura_yark", "Assessing the detection of the Yarkovsky effect using Gaia DR3 "
                      "and FPR catalogues"),
    ("fuentes_munoz_fpr", "Asteroid Orbit Determination Using Gaia FPR: Statistical Analysis"),
    ("gaia_crf3_ephem", "Comparison of the Gaia-CRF3 and planetary ephemerides via "
                        "asteroid observations"),
    ("delvigna", "Detecting the Yarkovsky effect among near-Earth asteroids from "
                 "astrometric data"),
    ("farnocchia_sh2", "Non-gravitational acceleration indicative of cometary activity "
                       "of near-Earth object"),
    ("gaia_fpr_orbits", "Gaia Focused Product Release: Asteroid orbital solution. "
                        "Properties and assessment"),
    ("gaia_dr3_sso_survey", "Gaia Data Release 3: The Solar System survey"),
]

# Citing sets: who has built on the nearest neighbours, and did any of them
# turn the residual into an anomaly/technosignature search?
OA_CITING = [
    ("citing_liberato_dr3", "10.1051/0004-6361/202349122"),   # Liberato+ 2024 A&A 688 A50
    ("citing_delvigna", "10.1051/0004-6361/201833153"),       # Del Vigna+ 2018 A&A 617 A61
    ("citing_gaia_fpr_orbits", "10.1051/0004-6361/202347270"),  # Gaia FPR orbits
]

# VizieR catalogues: confirm the real catalogue title, table structure and the
# candidate counts we intend to use as a contamination veto.
VIZIER_CATS = [
    ("J_A+A_688_A50", "J/A+A/688/A50"),      # Liberato+ 2024 binary candidates
    ("J_A+A_617_A61", "J/A+A/617/A61"),      # Del Vigna+ 2018 Yarkovsky
    ("J_A+A_680_A37", "J/A+A/680/A37"),      # Gaia FPR asteroid orbits
]


def verify_ids(ids: list[str]) -> dict:
    ver = {}
    for i in range(0, len(ids), 8):
        if out_of_time():
            break
        chunk = ids[i:i + 8]
        url = ARXIV + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)})
        blob = get(url, OUT / f"ax_verify_{i//8}.atom")
        if blob is None:
            continue
        for e in entries(blob):
            base = e["id"].split("v")[0]
            ver[base] = e
    for want in ids:
        if want not in ver:
            ver[want] = {"error": "NOT_RESOLVED"}
    return ver


def openalex_title(name: str, title: str) -> None:
    url = (OPENALEX + "works?" + urllib.parse.urlencode(
        {"search": title, "per-page": 5, "mailto": MAIL,
         "select": "id,doi,title,publication_year,cited_by_count,"
                   "primary_location,authorships"}))
    get(url, OUT / f"oa_{name}.json", pause=1.0)


def openalex_citing(name: str, doi: str) -> None:
    w = get(OPENALEX + f"works/doi:{doi}?mailto={MAIL}",
            OUT / f"oa_work_{name}.json", pause=1.0)
    if w is None:
        return
    try:
        wid = json.loads(w)["id"].rsplit("/", 1)[-1]
    except Exception:  # noqa: BLE001
        return
    for page in (1, 2, 3):
        url = OPENALEX + "works?" + urllib.parse.urlencode(
            {"filter": f"cites:{wid}", "per-page": 100, "page": page,
             "mailto": MAIL,
             "select": "id,doi,title,publication_year,cited_by_count,"
                       "abstract_inverted_index"})
        b = get(url, OUT / f"oa_{name}_p{page}.json", pause=1.0)
        if b is None:
            break
        try:
            if len(json.loads(b).get("results", [])) < 100:
                break
        except Exception:  # noqa: BLE001
            break


def strip_html(b: bytes) -> str:
    t = b.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", t))
    return re.sub(r"\n\s*\n\s*\n+", "\n\n",
                  re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", t))).strip()


def fulltext(name: str, aid: str) -> None:
    for u in (f"https://arxiv.org/html/{aid}",
              f"https://www.arxiv.org/html/{aid}",
              f"https://ar5iv.labs.arxiv.org/html/{aid}",
              f"https://arxiv.org/abs/{aid}"):
        b = get(u, OUT / f"html_{name}.html", tries=2, pause=2.0)
        if b and len(b) > 20000:
            (OUT / f"txt_{name}.txt").write_text(strip_html(b)[:600000])
            return
    if (OUT / f"html_{name}.html").exists():
        (OUT / f"txt_{name}.txt").write_text(
            strip_html((OUT / f"html_{name}.html").read_bytes())[:600000])


def term_occupancy() -> dict:
    """Count TERMS over every full text actually retrieved.  Case-insensitive,
    whole-phrase.  This is the corpus any 'term is unoccupied' claim rests on."""
    docs = sorted(OUT.glob("txt_*.txt"))
    matrix: dict[str, dict[str, int]] = {}
    sizes: dict[str, int] = {}
    for d in docs:
        text = d.read_text(errors="replace").lower()
        sizes[d.stem[4:]] = len(text)
        for t in TERMS:
            matrix.setdefault(t, {})[d.stem[4:]] = text.count(t.lower())
    names = [d.stem[4:] for d in docs]
    with (OUT / "term_occupancy.tsv").open("w") as fh:
        fh.write("term\t" + "\t".join(names) + "\ttotal\n")
        for t in TERMS:
            row = [matrix.get(t, {}).get(n, 0) for n in names]
            fh.write(t + "\t" + "\t".join(str(v) for v in row) +
                     f"\t{sum(row)}\n")
        fh.write("__chars__\t" + "\t".join(str(sizes[n]) for n in names) +
                 f"\t{sum(sizes.values())}\n")
    return {"docs": names, "chars": sizes,
            "zero_everywhere": [t for t in TERMS
                                if sum(matrix.get(t, {}).values()) == 0]}


def vizier() -> None:
    for name, cat in VIZIER_CATS:
        if out_of_time():
            break
        get(VIZIER + urllib.parse.quote(cat), OUT / f"vizier_{name}.txt",
            tries=2, pause=2.0)


def main() -> None:
    rows, zero = run_queries()

    with (OUT / "hits.tsv").open("w") as fh:
        fh.write("question\tquery_name\tarxiv_id\tpublished\ttitle\tauthors\tcats\n")
        for r in rows:
            fh.write("\t".join([r["question"], r["query_name"], r["id"],
                                r["published"], r["title"].replace("\t", " "),
                                r["authors"].replace("\t", " "), r["cats"]]) + "\n")
    (OUT / "hits_full.json").write_text(json.dumps(rows, indent=1))
    (OUT / "zero_hits.json").write_text(json.dumps(zero, indent=1))

    ver = verify_ids(VERIFY_IDS)
    (OUT / "ids_verified.json").write_text(json.dumps(ver, indent=1))

    for name, aid in FULLTEXT_TARGETS.items():
        if out_of_time():
            break
        fulltext(name, aid)
    occ = term_occupancy()
    (OUT / "term_occupancy.json").write_text(json.dumps(occ, indent=1))

    for name, title in OA_TITLES:
        if out_of_time():
            break
        openalex_title(name, title)
    for name, doi in OA_CITING:
        if out_of_time():
            break
        openalex_citing(name, doi)

    vizier()

    (OUT / "queries.json").write_text(json.dumps(STATUS, indent=1))
    summary = {
        "n_queries_planned": len(Q),
        "n_queries_issued": sum(1 for s in STATUS if s.get("query_name")),
        "n_query_failed": sum(1 for s in STATUS
                              if s.get("query_name") and not s["ok"]),
        "n_hits_rows": len(rows),
        "n_unique_ids": len({r["id"] for r in rows}),
        "per_question_totalresults": {
            s["query_name"]: {"question": s.get("question"),
                              "query": s.get("search_query"),
                              "totalResults": s.get("totalResults")}
            for s in STATUS if s.get("query_name")},
        "zero_or_failed": zero,
        "n_ids_verified": sum(1 for v in ver.values() if "error" not in v),
        "ids_not_resolved": [k for k, v in ver.items() if "error" in v],
        "fulltexts_retrieved": occ["docs"],
        "terms_zero_across_corpus": occ["zero_everywhere"],
        "elapsed_s": round(time.time() - T0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "per_question_totalresults"}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VNPROBELIT — prior-art / novelty adjudication for a *population* search for
alien self-replicating (von Neumann) probes among solar-system objects, using
Rubin/LSST SSO alerts (ephemeris residuals + non-gravitational parameters).

Eight angles (see docs; mirrored in the query names below):
  A1 artificial objects / probes / lurkers in the solar system
  A2 non-gravitational acceleration as a technosignature (solar-system, population)
  A3 self-replicating / von Neumann / Bracewell probe detection strategies
  A4 artificial asteroid families; SFD / albedo homogeneity as artificiality test
  A5 anomalous albedo / phase curve / colour / rotation-attitude technosignatures
  A6 Lagrange points / resonances / co-orbitals / quasi-satellites as SETI targets
  A7 Rubin/LSST SSO alerts or ephemeris residuals for technosignature work
  A8 non-SETI baseline: Yarkovsky detections, A1/A2/A3 fitting, ephemeris residuals

The dev sandbox blocks arxiv.org / api.openalex.org / ADS; the runner has egress
(CLAUDE.md acquisition pattern). Everything is written under results/vnprobelit/
and committed back to the branch.

Outputs
  queries.json        every query issued: URL, HTTP status, n_results, ids
  ax_q_<name>.atom    raw arXiv Atom per query
  hits.tsv            flat table: angle, query, arxiv id, date, title
  ids_verified.json   id -> REAL title/date/authors (nothing is cited unverified)
  oa_<name>.json      OpenAlex resolutions / citing sets
  txt_<name>.txt      full text of the papers whose content decides a verdict
  summary.json        counts, zero-hit statements, failures
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

OUT = pathlib.Path("results/vnprobelit")
OUT.mkdir(parents=True, exist_ok=True)
MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-vnprobelit/1.0 (mailto:{MAIL})"}

ARXIV = "https://export.arxiv.org/api/query?"
OPENALEX = "https://api.openalex.org/"

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
    m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", blob.decode("utf-8", "replace"))
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
# THE QUERY BATTERY.  (angle, name, search_query)
# --------------------------------------------------------------------------
Q: list[tuple[str, str, str]] = [
    # ---------------- A1  artificial objects / probes in the solar system
    ("A1", "lurkers_benford", 'all:"lurkers" AND all:"SETI"'),
    ("A1", "lurker_coorbital", 'all:"lurker" AND all:"co-orbital"'),
    ("A1", "seta_artifacts", 'abs:"extraterrestrial artifacts"'),
    ("A1", "seta_acronym", 'all:"SETA" AND all:"artifacts"'),
    ("A1", "probe_solar_system_search", 'abs:"probe" AND abs:"solar system" AND abs:"extraterrestrial"'),
    ("A1", "alien_artefact_search", 'abs:"alien" AND abs:"artefact"'),
    ("A1", "alien_artifact_search", 'abs:"alien artifact"'),
    ("A1", "artificial_object_solar_system", 'abs:"artificial object" AND abs:"solar system"'),
    ("A1", "dormant_probe", 'all:"dormant" AND all:"probe" AND all:"extraterrestrial"'),
    ("A1", "galileo_project", 'all:"Galileo Project" AND all:"Loeb"'),
    ("A1", "villarroel_vanishing", 'all:Villarroel AND all:"vanishing"'),
    ("A1", "villarroel_glint", 'all:Villarroel AND all:"glint"'),
    ("A1", "geosync_high_albedo", 'all:"geosynchronous" AND all:"albedo" AND all:"search"'),
    ("A1", "earth_trojan_seti", 'all:"Earth Trojan" AND all:"SETI"'),
    ("A1", "quasisatellite_seti", 'all:"quasi-satellite" AND all:"SETI"'),
    ("A1", "oumuamua_artificial", 'abs:"Oumuamua" AND abs:"artificial"'),
    ("A1", "interstellar_object_technosig", 'abs:"interstellar object" AND abs:"technosignature"'),
    ("A1", "3iatlas", 'abs:"3I/ATLAS"'),
    ("A1", "2i_borisov_technosig", 'abs:"Borisov" AND abs:"technosignature"'),
    ("A1", "sbdb_artificial_designation", 'all:"artificial" AND all:"minor planet" AND all:"designation"'),

    # ---------------- A2  non-gravitational acceleration as a technosignature
    ("A2", "nongrav_technosignature", 'abs:"non-gravitational" AND abs:"technosignature"'),
    ("A2", "nongrav_artificial", 'abs:"non-gravitational acceleration" AND abs:"artificial"'),
    ("A2", "lightsail_technosignature", 'abs:"light sail" AND abs:"technosignature"'),
    ("A2", "lightsail_solar_system", 'abs:"lightsail" AND abs:"solar system"'),
    ("A2", "solar_sail_detection", 'abs:"solar sail" AND abs:"detect" AND abs:"asteroid"'),
    ("A2", "dark_comets", 'abs:"dark comet"'),
    ("A2", "amr_small_body", 'all:"area-to-mass ratio" AND all:"small body"'),
    ("A2", "hamr_debris", 'all:"high area-to-mass ratio" AND all:"debris"'),
    ("A2", "srp_asteroid_astrometry", 'abs:"solar radiation pressure" AND abs:"asteroid" AND abs:"astrometry"'),
    ("A2", "venera_2005vl1", 'all:"2005 VL1"'),
    ("A2", "propulsion_signature_asteroid", 'abs:"propulsion" AND abs:"asteroid" AND abs:"signature"'),
    ("A2", "maneuver_detection_technosignature", 'abs:"maneuver" AND abs:"technosignature"'),
    ("A2", "thrust_extraterrestrial_spacecraft", 'abs:"thrust" AND abs:"extraterrestrial" AND abs:"spacecraft"'),

    # ---------------- A3  self-replicating / von Neumann / Bracewell probes
    ("A3", "von_neumann_probe", 'all:"von Neumann probe"'),
    ("A3", "von_neumann_probes", 'all:"von Neumann probes"'),
    ("A3", "self_replicating_probes", 'abs:"self-replicating" AND abs:"probes"'),
    ("A3", "self_replicating_spacecraft", 'all:"self-replicating spacecraft"'),
    ("A3", "self_reproducing_probe", 'all:"self-reproducing" AND all:"probe"'),
    ("A3", "bracewell_probe", 'all:"Bracewell probe"'),
    ("A3", "interstellar_probe_fermi", 'abs:"interstellar probes" AND abs:"Fermi paradox"'),
    ("A3", "gertz_probes", 'all:Gertz AND all:"probes"'),
    ("A3", "freitas_interstellar_probe", 'all:Freitas AND all:"probe"'),
    ("A3", "probe_galaxy_exploration", 'abs:"probes" AND abs:"galaxy" AND abs:"exploration" AND abs:"settlement"'),
    ("A3", "replicator_technosignature", 'abs:"replicator" AND abs:"technosignature"'),
    ("A3", "self_replication_detect_signature", 'abs:"self-replicating" AND abs:"detect"'),
    ("A3", "berserker_probes", 'all:"berserker"'),

    # ---------------- A4  artificial asteroid families / SFD / albedo homogeneity
    ("A4", "asteroid_family_artificial", 'abs:"asteroid family" AND abs:"artificial"'),
    ("A4", "size_frequency_artificial", 'abs:"size-frequency distribution" AND abs:"artificial"'),
    ("A4", "sfd_collisional_family", 'abs:"size frequency distribution" AND abs:"asteroid family"'),
    ("A4", "family_albedo_homogeneity", 'abs:"asteroid family" AND abs:"albedo" AND abs:"homogeneous"'),
    ("A4", "manufactured_object_population", 'abs:"manufactured" AND abs:"population" AND abs:"asteroid"'),
    ("A4", "artificial_population_statistical_seti", 'abs:"population" AND abs:"artificial" AND abs:"technosignature"'),
    ("A4", "asteroid_family_identification_methods", 'abs:"asteroid families" AND abs:"identification" AND abs:"hierarchical clustering"'),
    ("A4", "young_asteroid_family_clusters", 'abs:"young asteroid" AND abs:"cluster" AND abs:"backward integration"'),

    # ---------------- A5  albedo / phase curve / colour / rotation as technosignature
    ("A5", "albedo_technosignature", 'abs:"albedo" AND abs:"technosignature"'),
    ("A5", "phase_curve_artificial", 'abs:"phase curve" AND abs:"artificial"'),
    ("A5", "specular_reflection_artificial_object", 'abs:"specular" AND abs:"artificial" AND abs:"reflection"'),
    ("A5", "engineered_surface_reflectance", 'abs:"engineered" AND abs:"surface" AND abs:"reflectance" AND abs:"technosignature"'),
    ("A5", "polarimetry_artificial_object", 'abs:"polarimetry" AND abs:"artificial" AND abs:"space debris"'),
    ("A5", "nonprincipal_axis_rotation", 'abs:"non-principal axis rotation" AND abs:"asteroid"'),
    ("A5", "tumbling_asteroid_population", 'abs:"tumbling" AND abs:"asteroid" AND abs:"rotation"'),
    ("A5", "attitude_control_technosignature", 'abs:"attitude" AND abs:"technosignature"'),
    ("A5", "rotation_technosignature_asteroid", 'abs:"rotation" AND abs:"technosignature" AND abs:"asteroid"'),
    ("A5", "colour_outlier_minor_planet", 'abs:"colors" AND abs:"outlier" AND abs:"minor planets"'),
    ("A5", "spectral_anomaly_asteroid_search", 'abs:"spectral" AND abs:"anomalous" AND abs:"asteroid" AND abs:"survey"'),
    ("A5", "retroreflector_seti", 'abs:"retroreflector" AND abs:"SETI"'),

    # ---------------- A6  Lagrange points / resonances / stable niches
    ("A6", "lagrange_seti_search", 'abs:"Lagrange" AND abs:"SETI"'),
    ("A6", "l4_l5_search_objects", 'abs:"L4" AND abs:"L5" AND abs:"search" AND abs:"Earth"'),
    ("A6", "kordylewski", 'all:"Kordylewski"'),
    ("A6", "earth_trojan_search", 'abs:"Earth Trojan" AND abs:"survey"'),
    ("A6", "earth_coorbital_population", 'abs:"co-orbital" AND abs:"Earth" AND abs:"population"'),
    ("A6", "quasisatellite_earth", 'abs:"quasi-satellite" AND abs:"Earth"'),
    ("A6", "minimoon_population", 'abs:"minimoon"'),
    ("A6", "resonance_seti_target", 'abs:"mean motion resonance" AND abs:"technosignature"'),
    ("A6", "stable_niche_artifact", 'abs:"stable" AND abs:"orbit" AND abs:"artifact" AND abs:"extraterrestrial"'),

    # ---------------- A7  Rubin/LSST SSO alerts & ephemeris residuals for SETI
    ("A7", "lsst_technosignature", 'abs:"LSST" AND abs:"technosignature"'),
    ("A7", "rubin_technosignature", 'abs:"Rubin" AND abs:"technosignature"'),
    ("A7", "lsst_solar_system_alerts", 'abs:"LSST" AND abs:"solar system" AND abs:"alert"'),
    ("A7", "rubin_sso_alert_stream", 'abs:"Rubin" AND abs:"solar system objects" AND abs:"alerts"'),
    ("A7", "ephemeris_residual_anomaly_search", 'abs:"ephemeris" AND abs:"residual" AND abs:"anomaly"'),
    ("A7", "lsst_sssc_roadmap", 'ti:"Solar System Science" AND all:"LSST" AND all:"roadmap"'),
    ("A7", "tvs_roadmap", 'ti:"Transients and Variable Stars" AND all:"Rubin"'),
    ("A7", "lsst_yarkovsky", 'abs:"LSST" AND abs:"Yarkovsky"'),
    ("A7", "rubin_anomaly_detection_solar_system", 'abs:"Rubin" AND abs:"anomaly detection" AND abs:"solar system"'),
    ("A7", "lsst_serendipity_weird", 'abs:"LSST" AND abs:"unexpected" AND abs:"discoveries"'),
    ("A7", "mpc_alert_nongrav_lsst", 'abs:"Minor Planet Center" AND abs:"LSST"'),

    # ---------------- A8  non-SETI baseline literature
    ("A8", "yarkovsky_detection_catalog", 'abs:"Yarkovsky" AND abs:"detection" AND abs:"asteroids"'),
    ("A8", "yarkovsky_semimajor_drift", 'abs:"Yarkovsky" AND abs:"semimajor axis drift"'),
    ("A8", "yarkovsky_farnocchia", 'all:Farnocchia AND all:Yarkovsky'),
    ("A8", "yarkovsky_greenberg", 'all:Greenberg AND all:Yarkovsky AND all:asteroids'),
    ("A8", "yarkovsky_vokrouhlicky", 'all:Vokrouhlicky AND all:Yarkovsky'),
    ("A8", "a2_nongrav_comet_fitting", 'abs:"non-gravitational" AND abs:"comet" AND abs:"A2"'),
    ("A8", "marsden_nongrav_model", 'all:"Marsden" AND all:"non-gravitational" AND all:"model"'),
    ("A8", "star_catalog_astrometric_bias", 'abs:"star catalog" AND abs:"bias" AND abs:"asteroid astrometry"'),
    ("A8", "debiasing_asteroid_astrometry", 'abs:"debiasing" AND abs:"astrometry"'),
    ("A8", "astrometric_error_model_mpc", 'abs:"astrometric" AND abs:"error model" AND abs:"minor planet"'),
    ("A8", "binary_asteroid_photocenter_offset", 'abs:"binary asteroid" AND abs:"photocenter"'),
    ("A8", "unmodeled_perturbation_asteroid_orbit", 'abs:"perturbations" AND abs:"asteroid" AND abs:"orbit determination" AND abs:"mass"'),
    ("A8", "bennu_nongrav_anomaly", 'abs:"Bennu" AND abs:"non-gravitational"'),
    ("A8", "phaethon_nongrav", 'abs:"Phaethon" AND abs:"non-gravitational"'),
    ("A8", "activity_asteroid_unexplained_accel", 'abs:"active asteroid" AND abs:"non-gravitational acceleration"'),
]

MAX = 60


def run_queries() -> tuple[list[dict], dict]:
    rows, zero = [], {}
    for angle, name, sq in Q:
        if out_of_time():
            print("### deadline: stopping query battery", flush=True)
            break
        url = (ARXIV + urllib.parse.urlencode(
            {"search_query": sq, "start": 0, "max_results": MAX,
             "sortBy": "relevance"}))
        blob = get(url, OUT / f"ax_q_{name}.atom")
        if blob is None:
            zero[name] = {"angle": angle, "query": sq, "status": "QUERY_FAILED"}
            continue
        tot = total_results(blob)
        es = entries(blob)
        if not es:
            zero[name] = {"angle": angle, "query": sq,
                          "status": "ZERO_RESULTS", "totalResults": tot}
        for e in es:
            rows.append({"angle": angle, "query_name": name, "query": sq,
                         "id": e["id"], "published": e["published"],
                         "title": e["title"], "authors": "; ".join(e["authors"]),
                         "cats": ",".join(e["cats"]), "summary": e["summary"]})
        STATUS[-1].update({"query_name": name, "angle": angle,
                           "search_query": sq, "totalResults": tot,
                           "n_entries": len(es)})
    return rows, zero


# --------------------------------------------------------------------------
# IDs harvested from web search that MUST be verified before citation
# --------------------------------------------------------------------------
VERIFY_IDS = [
    "1903.09582",   # suspected Benford, Looking for Lurkers
    "2011.02495",   # Corbett orbital foregrounds (control, already verified)
    "2208.04499",   # suspected Rubin TVS roadmap
    "2212.08115",   # suspected Seligman dark comets
    "1810.11490",   # suspected Bialy & Loeb radiation pressure
    "2503.03552",   # Loeb & Cloete Venera 2 / 2005 VL1
    "1903.05839",   # suspected Lacki specular glint
    "2401.08763",   # suspected weird & wonderful solar system LSST
    "2606.13797",   # suspected Solar System Technosignatures
    "2103.01536",   # suspected concepts for future technosignature missions
    "1802.01783",   # suspected LSST SSSC roadmap
    "2009.07653",   # candidate: SSSC / LSST solar system
    "2506.14744",   # Gallay negative-flux ZTF
    "2602.12955",   # AHA anomaly hunter for alerts
    "2209.11685",   # technosignature science in planetary decadal
    "1911.05055",   # candidate: technosignature NASA workshop report
]

FULLTEXT_TARGETS = {
    # name: arXiv id.  These decide verdicts and must be read, not guessed.
    "benford_lurkers": "1903.09582",
    "rubin_tvs_roadmap": "2208.04499",
    "weird_wonderful_lsst": "2401.08763",
}

OA_TITLES = [
    ("benford_lurkers_aj", "Looking for Lurkers: Co-orbiters as SETI Observables"),
    ("sssc_roadmap", "The Large Synoptic Survey Telescope Solar System Science Roadmap"),
    ("freitas_search_artifacts", "The search for extraterrestrial artifacts"),
    ("freitas_selfrep_probes", "A self-reproducing interstellar probe"),
    ("tipler_no_eti", "Extraterrestrial intelligent beings do not exist"),
    ("sagan_newman_solipsist", "The solipsist approach to extraterrestrial intelligence"),
    ("valdes_freitas_l4l5", "A search for objects near the Earth-Moon Lagrange points"),
    ("freitas_valdes_seta_survey", "A search for natural or artificial objects located "
                                   "at the Earth-Moon libration points"),
    ("greenberg_yarkovsky", "Asteroid Yarkovsky Effects via Statistical Analysis of "
                            "Minor Planet Center Data"),
    ("farnocchia_yarkovsky", "Near Earth Asteroids with measurable Yarkovsky effect"),
    ("nesvorny_families", "Identification and Dynamical Properties of Asteroid Families"),
    ("rubin_alert_content", "Vera C. Rubin Observatory Alert Content"),
]

OA_CITING = [
    # name, OpenAlex "cites:" filter target resolved from a DOI/arXiv id
    ("citing_benford_lurkers", "10.3847/1538-3881/ab3e35"),
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
    w = get(OPENALEX + f"works/doi:{doi}?mailto={MAIL}", OUT / f"oa_work_{name}.json",
            pause=1.0)
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
             "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index"})
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
    for u in (f"https://arxiv.org/abs/{aid}",
              f"https://www.arxiv.org/html/{aid}",
              f"https://ar5iv.labs.arxiv.org/html/{aid}"):
        b = get(u, OUT / f"html_{name}.html", tries=2, pause=2.0)
        if b and len(b) > 20000:
            (OUT / f"txt_{name}.txt").write_text(strip_html(b)[:400000])
            return
    if (OUT / f"html_{name}.html").exists():
        (OUT / f"txt_{name}.txt").write_text(
            strip_html((OUT / f"html_{name}.html").read_bytes())[:400000])


def main() -> None:
    rows, zero = run_queries()

    with (OUT / "hits.tsv").open("w") as fh:
        fh.write("angle\tquery_name\tarxiv_id\tpublished\ttitle\tauthors\tcats\n")
        for r in rows:
            fh.write("\t".join([r["angle"], r["query_name"], r["id"],
                                r["published"], r["title"].replace("\t", " "),
                                r["authors"].replace("\t", " "), r["cats"]]) + "\n")
    (OUT / "hits_full.json").write_text(json.dumps(rows, indent=1))

    ver = verify_ids(VERIFY_IDS)
    (OUT / "ids_verified.json").write_text(json.dumps(ver, indent=1))

    for name, title in OA_TITLES:
        if out_of_time():
            break
        openalex_title(name, title)
    for name, doi in OA_CITING:
        if out_of_time():
            break
        openalex_citing(name, doi)
    for name, aid in FULLTEXT_TARGETS.items():
        if out_of_time():
            break
        fulltext(name, aid)

    (OUT / "queries.json").write_text(json.dumps(STATUS, indent=1))
    summary = {
        "n_queries_planned": len(Q),
        "n_queries_issued": sum(1 for s in STATUS if s.get("query_name")),
        "n_query_failed": sum(1 for s in STATUS
                              if s.get("query_name") and not s["ok"]),
        "n_hits_rows": len(rows),
        "n_unique_ids": len({r["id"] for r in rows}),
        "zero_or_failed": zero,
        "elapsed_s": round(time.time() - T0, 1),
        "n_ids_verified": sum(1 for v in ver.values() if "error" not in v),
        "ids_not_resolved": [k for k, v in ver.items() if "error" in v],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

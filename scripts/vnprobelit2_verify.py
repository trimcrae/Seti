#!/usr/bin/env python3
"""VNPROBELIT pass 2 — verification + decisive full text + citing sets.

Pass 1 (scripts/vnprobelit_fetch.py) ran the broad arXiv query battery.
This pass:
  1. resolves EVERY arXiv id harvested from web search to its REAL title
     (a guessed id resolves to an unrelated paper often enough that citing
     an unverified id is unacceptable);
  2. resolves titles that may be journal-only through arXiv ti: search and
     OpenAlex (Freitas/Valdes/Tipler/Sagan-Newman/Gertz/Ellery/Papagiannis,
     and Lazio & Mahabal "On Anomalous Asteroid Accelerations", which
     arXiv:2606.13797 cites as *submitted* to Acta Astronautica);
  3. pulls FULL TEXT for the papers whose content decides a verdict:
       * Ellery 2510.00082 -- the only paper titled for self-replicating-probe
         technosignatures in the solar system.  Theory only, or a search?
       * SNAPS 2405.20176 / 2604.27420 -- an SSO alert broker that already does
         population outlier detection.  Does it touch technosignatures, or
         ephemeris residuals / non-gravitational parameters?
       * Levine et al. 2410.06874 -- strong non-grav accelerations and NEO
         misidentification: the systematics baseline for a residual search.
       * Benford 1903.09582 -- lurkers: proposal or executed observations?
       * SSSC roadmap / LSST solar-system yield -- what is CLAIMED as planned.
  4. pulls OpenAlex CITING SETS for the proposal papers, so "proposed but never
     executed" is a counted statement rather than an impression;
  5. runs a second, gap-filling arXiv query battery.

Runs on the GitHub Actions runner; the dev sandbox blocks arxiv.org.
Outputs under results/vnprobelit2/.
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

OUT = pathlib.Path("results/vnprobelit2")
OUT.mkdir(parents=True, exist_ok=True)
MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-vnprobelit2/1.0 (mailto:{MAIL})"}
ARXIV = "https://export.arxiv.org/api/query?"
OPENALEX = "https://api.openalex.org/"
STATUS: list[dict] = []
T0 = time.time()
DEADLINE = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0


def out_of_time(margin: float = 90.0) -> bool:
    return (time.time() - T0) > (DEADLINE - margin)


def get(url: str, dest: pathlib.Path | None = None, tries: int = 3,
        pause: float = 3.0) -> bytes | None:
    last = ""
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=90) as r:
                data, code = r.read(), r.status
            if dest is not None:
                dest.write_bytes(data)
            STATUS.append({"url": url, "dest": dest.name if dest else None,
                           "http": code, "bytes": len(data), "ok": True})
            print(f"OK   {code} {len(data):9d}B  {url[:135]}", flush=True)
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        print(f"RETRY({i+1}) {last}  {url[:135]}", flush=True)
        time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "dest": dest.name if dest else None,
                   "http": None, "ok": False, "error": last})
    print(f"FAIL {last}  {url[:135]}", flush=True)
    return None


def entries(blob: bytes) -> list[dict]:
    txt = blob.decode("utf-8", "replace")
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", txt, re.S):
        def f(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
            return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip()) if m else ""
        auth = [html.unescape(re.sub(r"\s+", " ", a).strip())
                for a in re.findall(r"<author>\s*<name>(.*?)</name>", e, re.S)]
        out.append({"id": f("id").replace("http://arxiv.org/abs/", "")
                                 .replace("https://arxiv.org/abs/", ""),
                    "title": f("title"), "published": f("published")[:10],
                    "authors": auth[:8], "summary": f("summary"),
                    "cats": re.findall(r'<category term="([^"]+)"', e)[:5]})
    return out


def total_results(blob: bytes) -> int | None:
    m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<",
                  blob.decode("utf-8", "replace"))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# 1. every id harvested from web search, to be resolved to its real title
# --------------------------------------------------------------------------
IDS = [
    # von Neumann / self-replicating
    "2510.00082",  # suspected Ellery, Technosignatures of Self-Replicating Probes
    "1909.05078",  # suspected interstellar von Neumann micro self-reproducing probes
    # solar-system / ISO technosignature framing
    "2606.13797", "2508.16825", "2605.21093", "2107.07512", "2410.08253",
    "1903.09582",  # Benford lurkers
    "2209.11685", "1812.08681", "2603.17741", "2310.14895",
    "2507.17790",  # suspected rebalancing astronomical research priorities
    # non-grav acceleration baseline / dark comets
    "2410.06874",  # suspected strong nongrav accel + NEO misidentification
    "2212.08115", "2503.03552", "1810.11490", "1612.06920",
    "2208.12364",  # suspected chasing tails active asteroids
    # Yarkovsky baseline
    "1708.05513", "2206.05561",
    # Rubin / LSST solar system
    "2405.20176", "2604.27420",  # SNAPS
    "2009.07653", "2506.02140", "2310.17575", "1901.08549", "1906.11346",
    "1812.01149", "1812.00937", "2208.02781", "2401.08763", "2606.00574",
    "2208.04499", "2507.13409",
    # Lagrange / Trojans / co-orbitals
    "2606.31751",  # suspected WFST L4 Earth Trojan search
    "2302.11086",  # suspected MEGASIM Earth Trojans
    "2110.14743",  # suspected lunar Trojan survey
    # albedo / specular / colour / phase curve
    "2306.07859", "1903.05839", "2110.15217", "2204.06091", "2509.22301",
    # clustering / anomaly detection
    "2603.13177",  # suspected clustering orbital synthetic data
    "2410.18875", "2507.06217", "2602.12955",
]

# --------------------------------------------------------------------------
# 2. titles that may be journal-only: arXiv ti: search + OpenAlex
# --------------------------------------------------------------------------
TITLE_SEARCH = [
    ("lazio_mahabal_anomalous_accel", "On Anomalous Asteroid Accelerations"),
    ("ellery_selfrep_imminent", "Self-replicating probes are imminent implications for SETI"),
    ("gertz_et_probes_here", "ET Probes: Looking Here as Well as There"),
    ("matloff_artificial_kbo", "A Proposed Infrared Search for Artificial Kuiper Belt Objects"),
    ("benford_how_many_probes", "How Many Alien Probes Could Have Come from Stars Passing by Earth"),
    ("freitas_valdes_1985_seta", "The Search for Extraterrestrial Artifacts SETA"),
    ("valdes_freitas_1983_l4l5", "A search for objects near the Earth-Moon Lagrange points"),
    ("papagiannis_asteroid_belt", "Are we alone or could they be in the asteroid belt"),
    ("tipler_1980", "Extraterrestrial intelligent beings do not exist"),
    ("sagan_newman_1983", "The Solipsist Approach to Extraterrestrial Intelligence"),
    ("greenberg_yarkovsky_247", "Yarkovsky Drift Detections for 247 Near-Earth Asteroids"),
    ("farnocchia_yarkovsky_neas", "Near Earth Asteroids with measurable Yarkovsky effect"),
    ("fenucci_neocc_yarkovsky", "An automated procedure for the detection of the Yarkovsky effect"),
    ("fenucci_gaia_yarkovsky", "The Yarkovsky effect and bulk density of near-Earth asteroids from Gaia DR3"),
    ("nugent_yarkovsky", "Detection of semimajor axis drifts in 54 near-Earth asteroids"),
    ("delvigna_yarkovsky", "Detecting the Yarkovsky effect among near-Earth asteroids from astrometric data"),
    ("eggl_yarkovsky_mainbelt", "Detectability of the Yarkovsky Effect in the Main Belt"),
    ("chesley_bennu_yarkovsky", "Orbit and bulk density of the OSIRIS-REx target Asteroid 101955 Bennu"),
    ("nesvorny_family_catalog", "Identification and Dynamical Properties of Asteroid Families"),
    ("milani_family_classification", "Asteroid families classification: exploiting very large datasets"),
    ("novakovic_young_families", "Debiased population of very young asteroid families"),
    ("eggl_debiasing", "Debiasing the minor planet astrometric catalog"),
    ("farnocchia_star_catalog_debias", "Star catalog position and proper motion corrections in asteroid astrometry"),
    ("veres_error_model", "Statistical analysis of astrometric errors for the most productive asteroid surveys"),
    ("trilling_snaps1", "The Solar System Notification Alert Processing System (SNAPS): Design, Architecture, and First Data Release"),
    ("rubin_sssc_roadmap", "Solar System Science with the Large Synoptic Survey Telescope"),
    ("kurlander_lsst_yield", "Predictions of the LSST Solar System Yield"),
    ("sorcha_simulator", "Sorcha: A Solar System Survey Simulator for the Legacy Survey of Space and Time"),
    ("rogers_colours_lsst", "The weird and the wonderful in our Solar System"),
    ("haqq_kopparapu_seta", "On the likelihood of non-terrestrial artifacts in the Solar System"),
    ("lazio_datadriven_2023", "Data-Driven Approaches to Searches for the Technosignatures of Advanced Civilizations"),
]

# --------------------------------------------------------------------------
# 3. full text of the decisive papers
# --------------------------------------------------------------------------
FULLTEXT = {
    "ellery_selfrep_probes": "2510.00082",
    "snaps_outliers": "2405.20176",
    "snaps_public": "2604.27420",
    "levine_misidentification": "2410.06874",
    "benford_lurkers": "1903.09582",
    "vonneumann_micro_probes": "1909.05078",
    "lsst_solar_system_impact": "2009.07653",
    "wfst_earth_trojan": "2606.31751",
    "megasim_earth_trojans": "2302.11086",
    "clustering_orbital_data": "2603.13177",
    "chasing_tails_active": "2208.12364",
    "dark_comets_seligman": "2212.08115",
    "rubin_astrobio_prospects": "2606.00574",
}

# --------------------------------------------------------------------------
# 4. citing sets: "proposed but never executed" must be a counted claim
# --------------------------------------------------------------------------
CITING_DOI = [
    ("benford_lurkers", "10.3847/1538-3881/ab3e35"),
    ("ellery_selfrep_imminent", "10.1017/S1473550422000234"),
    ("snaps_outliers", "10.3847/1538-3881/ad4da3"),
    ("dark_comets_seligman", "10.3847/PSJ/acb697"),
    ("lacki_specular", "10.1088/1538-3873/ab1304"),
]
CITING_ARXIV = [
    ("lazio_solar_system_technosig", "2606.13797"),
    ("davenport_iso_technosig", "2508.16825"),
    ("ellery_selfrep_probes", "2510.00082"),
]

# --------------------------------------------------------------------------
# 5. gap-filling arXiv queries
# --------------------------------------------------------------------------
Q2: list[tuple[str, str, str]] = [
    ("A2", "residual_population_nongrav", 'abs:"non-gravitational" AND abs:"population" AND abs:"survey" AND abs:"asteroids"'),
    ("A2", "astrometric_residual_technosig", 'abs:"astrometric residuals" AND abs:"anomalous"'),
    ("A2", "nongrav_misidentification", 'abs:"nongravitational" AND abs:"misidentification"'),
    ("A2", "propulsion_detectable_solar_system", 'abs:"spacecraft" AND abs:"detect" AND abs:"minor planet" AND abs:"orbit"'),
    ("A3", "ellery_self_replication", 'all:Ellery AND all:"self-replicating"'),
    ("A3", "vonneumann_solar_system_technosig", 'abs:"self-replicating" AND abs:"solar system" AND abs:"technosignature"'),
    ("A3", "replication_signature_population", 'abs:"replication" AND abs:"population" AND abs:"extraterrestrial"'),
    ("A4", "family_outlier_artificial_test", 'abs:"asteroid" AND abs:"outlier" AND abs:"artificial"'),
    ("A4", "sfd_slope_artificial_debris", 'abs:"size distribution" AND abs:"debris" AND abs:"fragmentation" AND abs:"power law"'),
    ("A4", "orbital_clustering_seti", 'abs:"clustering" AND abs:"orbits" AND abs:"SETI"'),
    ("A4", "cluster_orbital_elements_anomaly", 'abs:"orbital elements" AND abs:"clustering" AND abs:"anomaly"'),
    ("A5", "albedo_homogeneity_family", 'abs:"albedo" AND abs:"asteroid family" AND abs:"members"'),
    ("A5", "phase_curve_lsst_asteroids", 'abs:"phase curve" AND abs:"LSST" AND abs:"asteroids"'),
    ("A5", "hg12_phase_curve_population", 'abs:"H, G12" OR abs:"G12" AND abs:"phase curve"'),
    ("A5", "starshade_phase_curve_technosig", 'abs:"starshade" AND abs:"technosignature"'),
    ("A6", "coorbital_technosignature_search", 'abs:"co-orbital" AND abs:"technosignature"'),
    ("A6", "horseshoe_orbit_seti", 'abs:"horseshoe orbit" AND abs:"SETI"'),
    ("A6", "lagrange_point_survey_artefact", 'abs:"libration" AND abs:"search" AND abs:"artificial"'),
    ("A7", "rubin_sso_alert_broker", 'abs:"solar system" AND abs:"alert broker"'),
    ("A7", "lsst_alert_solar_system_anomaly", 'abs:"LSST" AND abs:"alerts" AND abs:"outlier"'),
    ("A7", "snaps_alert_processing", 'all:"Solar System Notification Alert Processing System"'),
    ("A7", "ephemeris_offset_alert", 'abs:"ephemeris" AND abs:"offset" AND abs:"alert"'),
    ("A7", "rubin_first_year_solar_system", 'abs:"Rubin" AND abs:"first year" AND abs:"solar system"'),
    ("A8", "yarkovsky_automated_catalogue", 'abs:"Yarkovsky" AND abs:"automated" AND abs:"catalogue"'),
    ("A8", "yarkovsky_main_belt_detect", 'abs:"Yarkovsky" AND abs:"main belt" AND abs:"detectability"'),
    ("A8", "astrometry_quality_gaia_asteroid", 'abs:"Gaia" AND abs:"asteroid astrometry" AND abs:"accuracy"'),
    ("A8", "orbit_determination_nongrav_estimate", 'abs:"orbit determination" AND abs:"non-gravitational parameters"'),
    ("A8", "mpc_orbit_fitting_pipeline", 'abs:"Minor Planet Center" AND abs:"orbit" AND abs:"pipeline"'),
    ("A8", "satellite_designated_asteroid", 'abs:"2020 SO" OR abs:"rocket booster" AND abs:"near-Earth object"'),
]


def strip_html(b: bytes) -> str:
    t = b.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", t))
    return re.sub(r"\n\s*\n\s*\n+", "\n\n",
                  re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", t))).strip()


def fulltext(name: str, aid: str) -> None:
    for u in (f"https://arxiv.org/html/{aid}",
              f"https://arxiv.org/html/{aid}v1",
              f"https://ar5iv.labs.arxiv.org/html/{aid}",
              f"https://arxiv.org/abs/{aid}"):
        b = get(u, OUT / f"html_{name}.html", tries=2, pause=2.0)
        if b and len(b) > 25000:
            (OUT / f"txt_{name}.txt").write_text(strip_html(b)[:500000])
            return
    if (OUT / f"html_{name}.html").exists():
        (OUT / f"txt_{name}.txt").write_text(
            strip_html((OUT / f"html_{name}.html").read_bytes())[:500000])


def main() -> None:
    # ---- 1. id verification
    ver: dict = {}
    for i in range(0, len(IDS), 8):
        if out_of_time():
            break
        chunk = IDS[i:i + 8]
        b = get(ARXIV + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}),
            OUT / f"ax_verify_{i//8}.atom")
        if b:
            for e in entries(b):
                ver[e["id"].split("v")[0]] = e
    for want in IDS:
        ver.setdefault(want, {"error": "NOT_RESOLVED"})
    (OUT / "ids_verified.json").write_text(json.dumps(ver, indent=1))

    # ---- 2. title searches
    tsearch: dict = {}
    for name, title in TITLE_SEARCH:
        if out_of_time():
            break
        q = 'ti:"%s"' % title.replace('"', "")
        b = get(ARXIV + urllib.parse.urlencode(
            {"search_query": q, "max_results": 5}), OUT / f"ax_ti_{name}.atom")
        es = entries(b) if b else []
        if not es:  # relaxed: all: instead of ti:
            b2 = get(ARXIV + urllib.parse.urlencode(
                {"search_query": 'all:"%s"' % title.replace('"', ""),
                 "max_results": 5}), OUT / f"ax_all_{name}.atom")
            es = entries(b2) if b2 else []
        tsearch[name] = {"query_title": title,
                         "arxiv_hits": [{"id": e["id"], "title": e["title"],
                                         "published": e["published"]}
                                        for e in es]}
        oa = get(OPENALEX + "works?" + urllib.parse.urlencode(
            {"search": title, "per-page": 3, "mailto": MAIL,
             "select": "id,doi,title,publication_year,cited_by_count,type"}),
            OUT / f"oa_ti_{name}.json", tries=2, pause=1.0)
        if oa:
            try:
                tsearch[name]["openalex"] = [
                    {k: w.get(k) for k in ("doi", "title", "publication_year",
                                           "cited_by_count", "type")}
                    for w in json.loads(oa).get("results", [])]
            except Exception:  # noqa: BLE001
                pass
    (OUT / "title_searches.json").write_text(json.dumps(tsearch, indent=1))

    # ---- 3. full text
    for name, aid in FULLTEXT.items():
        if out_of_time():
            break
        fulltext(name, aid)

    # ---- 4. citing sets
    def citing(name: str, wid: str) -> None:
        for page in (1, 2, 3):
            b = get(OPENALEX + "works?" + urllib.parse.urlencode(
                {"filter": f"cites:{wid}", "per-page": 100, "page": page,
                 "mailto": MAIL,
                 "select": "id,doi,title,publication_year,cited_by_count,"
                           "abstract_inverted_index,type"}),
                OUT / f"cites_{name}_p{page}.json", tries=2, pause=1.0)
            if not b:
                break
            try:
                if len(json.loads(b).get("results", [])) < 100:
                    break
            except Exception:  # noqa: BLE001
                break

    for name, doi in CITING_DOI:
        if out_of_time():
            break
        w = get(OPENALEX + f"works/doi:{doi}?mailto={MAIL}",
                OUT / f"oa_work_{name}.json", tries=2, pause=1.0)
        if w:
            try:
                citing(name, json.loads(w)["id"].rsplit("/", 1)[-1])
            except Exception:  # noqa: BLE001
                pass
    for name, aid in CITING_ARXIV:
        if out_of_time():
            break
        w = get(OPENALEX + "works?" + urllib.parse.urlencode(
            {"filter": f"locations.landing_page_url.search:arxiv.org/abs/{aid}",
             "mailto": MAIL, "per-page": 3}),
            OUT / f"oa_arxivwork_{name}.json", tries=2, pause=1.0)
        if not w:
            continue
        try:
            res = json.loads(w).get("results", [])
            if res:
                citing(name, res[0]["id"].rsplit("/", 1)[-1])
        except Exception:  # noqa: BLE001
            pass

    # ---- 5. gap-filling queries
    rows, zero = [], {}
    for angle, name, sq in Q2:
        if out_of_time():
            break
        b = get(ARXIV + urllib.parse.urlencode(
            {"search_query": sq, "start": 0, "max_results": 40,
             "sortBy": "relevance"}), OUT / f"ax_q2_{name}.atom")
        if b is None:
            zero[name] = {"angle": angle, "query": sq, "status": "QUERY_FAILED"}
            continue
        es = entries(b)
        if not es:
            zero[name] = {"angle": angle, "query": sq, "status": "ZERO_RESULTS",
                          "totalResults": total_results(b)}
        for e in es:
            rows.append({"angle": angle, "query_name": name, "query": sq,
                         "id": e["id"], "published": e["published"],
                         "title": e["title"],
                         "authors": "; ".join(e["authors"])})
        STATUS[-1].update({"query_name": name, "angle": angle,
                           "search_query": sq,
                           "totalResults": total_results(b),
                           "n_entries": len(es)})
    with (OUT / "hits2.tsv").open("w") as fh:
        fh.write("angle\tquery_name\tarxiv_id\tpublished\ttitle\tauthors\n")
        for r in rows:
            fh.write("\t".join([r["angle"], r["query_name"], r["id"],
                                r["published"], r["title"].replace("\t", " "),
                                r["authors"].replace("\t", " ")]) + "\n")

    (OUT / "queries.json").write_text(json.dumps(STATUS, indent=1))
    (OUT / "summary.json").write_text(json.dumps({
        "n_ids_requested": len(IDS),
        "n_ids_resolved": sum(1 for v in ver.values() if "error" not in v),
        "ids_not_resolved": [k for k, v in ver.items() if "error" in v],
        "n_title_searches": len(tsearch),
        "titles_with_no_arxiv_hit": [k for k, v in tsearch.items()
                                     if not v.get("arxiv_hits")],
        "n_fulltext_written": len(list(OUT.glob("txt_*.txt"))),
        "n_q2_rows": len(rows),
        "q2_zero_or_failed": zero,
        "elapsed_s": round(time.time() - T0, 1),
    }, indent=1))
    print(json.dumps(json.loads((OUT / "summary.json").read_text()), indent=1))


if __name__ == "__main__":
    main()

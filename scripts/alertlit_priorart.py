#!/usr/bin/env python3
"""Prior-art / novelty adjudication: technosignature searches on TRANSIENT ALERT STREAMS.

Question: which technosignature searches that use a real-time transient alert
stream (Rubin/LSST, ZTF, ATLAS, Pan-STARRS, Gaia alerts) have already been
PERFORMED on data, which have merely been PROPOSED in white papers/reviews, and
which specific ideas are genuinely UNOCCUPIED?

Eight angles are adjudicated (A1..A8):
  A1 specular glint / achromatic single-epoch flash from a flat reflector
  A2 cross-night RECURRENCE of achromatic transients at a fixed sky position
  A3 negative-difference-flux (DIA "fainter than template") alert screening for
     artificial occulters / megastructure transits
  A4 optical laser-pulse searches inside IMAGING surveys (not dedicated pulse
     detectors like PANOSETI)
  A5 non-gravitational acceleration / anomalous astrometry of solar-system and
     interstellar objects in LSST-era discovery streams
  A6 transit-timing / transit-shape anomaly technosignatures in survey photometry
  A7 "LSST technosignature" / "Rubin SETI" reviews and white papers -- the full
     list of what the community has already claimed as planned
  A8 colour-space outliers / ML anomaly detection in alert streams framed as
     technosignature search (SNAD, Malanchev, Pruzhinskaya, Villar, Nir, ...)

Three independent recall instruments, because any one of them misses papers:
  * arXiv API   -- title+abstract, but misses journal-only papers
  * OpenAlex    -- abstracts for essentially the whole literature, with citations
  * Crossref    -- bibliographic tie-breaker on named papers
Plus ar5iv/arxiv full text for the key review papers, so their *planned-search*
lists can be enumerated verbatim rather than guessed.

Runs on the GitHub Actions runner; the dev sandbox blocks all these hosts.

Outputs under results/alertlit/:
  ax_<angle>_<name>.atom   arXiv API Atom response per query
  oa_<angle>_<name>.json   OpenAlex work list per query
  ids_verified.json        arXiv id -> real title, for every id we intend to cite
  ar5iv_<slug>.html/.txt   full text of key reviews
  digest.txt               flattened title/year/venue per query, all instruments
  counts.json              per-query hit counts (the evidence for emptiness)
  summary.json             fetch status for every URL
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/alertlit")
OUT.mkdir(parents=True, exist_ok=True)

MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-alertlit/1.0 (mailto:{MAIL})"}
STATUS: list[dict] = []
COUNTS: dict[str, dict] = {}
DIGEST: list[str] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bytes | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:56s} {len(data):9d}B", flush=True)
            return data
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return None


# --------------------------------------------------------------------------
# arXiv API
# --------------------------------------------------------------------------
ARXIV = "http://export.arxiv.org/api/query?"


def _atom_entries(blob: bytes) -> list[dict]:
    txt = blob.decode("utf-8", "replace")
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", txt, re.S):
        def f(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
            return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip()) if m else ""
        aid = f("id").replace("http://arxiv.org/abs/", "")
        out.append({"id": aid, "title": f("title"), "published": f("published")[:10],
                    "summary": f("summary")[:600]})
    return out


def _atom_total(blob: bytes) -> int:
    m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", blob.decode("utf-8", "replace"))
    return int(m.group(1)) if m else -1


def arxiv_query(angle: str, name: str, search: str, n: int = 100) -> list[dict]:
    url = ARXIV + urllib.parse.urlencode({
        "search_query": search, "start": 0, "max_results": n,
        "sortBy": "submittedDate", "sortOrder": "descending"})
    blob = get(url, OUT / f"ax_{angle}_{name}.atom")
    time.sleep(3.2)  # arXiv asks for >=3s between calls
    if blob is None:
        COUNTS.setdefault(angle, {})[f"arxiv:{name}"] = {"query": search, "total": None,
                                                         "error": True}
        return []
    ents = _atom_entries(blob)
    tot = _atom_total(blob)
    COUNTS.setdefault(angle, {})[f"arxiv:{name}"] = {"query": search, "total": tot,
                                                     "returned": len(ents)}
    DIGEST.append(f"\n### [{angle}] arXiv  {name}\n  query: {search}\n  totalResults={tot}")
    for x in ents:
        DIGEST.append(f"    {x['published']}  {x['id']:20s} {x['title']}")
    return ents


def arxiv_ids(ids: list[str]) -> dict[str, dict]:
    """Verify real titles for every arXiv id we might cite."""
    verified: dict[str, dict] = {}
    for k in range(0, len(ids), 25):
        chunk = ids[k:k + 25]
        url = ARXIV + urllib.parse.urlencode({"id_list": ",".join(chunk),
                                              "max_results": len(chunk)})
        blob = get(url, OUT / f"ax_verify_{k//25}.atom")
        time.sleep(3.2)
        if blob:
            for x in _atom_entries(blob):
                verified[x["id"].split("v")[0]] = x
    (OUT / "ids_verified.json").write_text(json.dumps(verified, indent=1))
    DIGEST.append("\n### arXiv id verification (id -> real title)")
    for aid in ids:
        v = verified.get(aid)
        DIGEST.append(f"    {aid:20s} -> {v['title'] if v else '*** NOT FOUND ***'}")
    return verified


# --------------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------------
OA = "https://api.openalex.org/works?"


def oa_query(angle: str, name: str, phrase: str, per: int = 50) -> list[dict]:
    url = OA + urllib.parse.urlencode({
        "search": phrase, "per-page": str(per), "sort": "relevance_score:desc",
        "mailto": MAIL})
    blob = get(url, OUT / f"oa_{angle}_{name}.json")
    time.sleep(1.3)
    if blob is None:
        COUNTS.setdefault(angle, {})[f"oa:{name}"] = {"query": phrase, "total": None,
                                                      "error": True}
        return []
    try:
        d = json.loads(blob)
    except Exception:  # noqa: BLE001
        return []
    res = d.get("results", []) or []
    tot = (d.get("meta") or {}).get("count")
    COUNTS.setdefault(angle, {})[f"oa:{name}"] = {"query": phrase, "total": tot,
                                                  "returned": len(res)}
    DIGEST.append(f"\n### [{angle}] OpenAlex  {name}\n  query: {phrase}\n  count={tot}")
    for w in res[:40]:
        venue = (((w.get("primary_location") or {}).get("source") or {}) or {}).get(
            "display_name") or ""
        DIGEST.append(f"    {w.get('publication_year')}  cit={w.get('cited_by_count'):<5} "
                      f"{(w.get('title') or '')[:130]}  [{venue[:45]}]")
    return res


# ==========================================================================
# query batteries
# ==========================================================================
# arXiv: abstract+title field searches. Recall over precision.
AX: dict[str, dict[str, str]] = {
    "A1_glint": {
        "glint_seti": 'abs:"glint" AND (abs:"SETI" OR abs:"technosignature" OR abs:"extraterrestrial")',
        "specular_reflection_seti": 'abs:"specular" AND (abs:"technosignature" OR abs:"SETI" OR abs:"artificial")',
        "mirror_reflector_seti": '(abs:"reflector" OR abs:"mirror") AND (abs:"technosignature" OR abs:"SETI")',
        "solar_panel_technosig": 'abs:"solar panel" AND (abs:"technosignature" OR abs:"exoplanet" OR abs:"SETI")',
        "achromatic_transient": 'abs:"achromatic" AND (abs:"transient" OR abs:"flare" OR abs:"variability")',
        "glint_survey": 'abs:"glint" AND (abs:"survey" OR abs:"transient" OR abs:"satellite")',
        "satellite_glint_survey": 'abs:"satellite glint" OR abs:"satellite glints"',
        "fast_optical_flash_survey": 'abs:"optical flash" AND (abs:"survey" OR abs:"search")',
        "artificial_illumination": 'abs:"artificial" AND abs:"reflected" AND abs:"starlight"',
        "geosynchronous_glint": 'abs:"geosynchronous" OR abs:"geostationary" AND abs:"transient"',
    },
    "A2_recurrence": {
        "repeating_transient_seti": '(abs:"repeating" OR abs:"recurrent") AND (abs:"technosignature" OR abs:"SETI")',
        "optical_beacon": 'abs:"beacon" AND (abs:"optical" OR abs:"laser" OR abs:"SETI")',
        "simultaneous_transients_plates": 'abs:"simultaneous transients" OR abs:"aligned transients"',
        "vasco": 'abs:"vanishing" AND abs:"appearing" AND abs:"sources"',
        "periodic_optical_signal_seti": 'abs:"periodic" AND abs:"signal" AND abs:"technosignature"',
        "repeating_flash_star": 'abs:"repeated" AND abs:"brightening" AND abs:"star"',
        "coincident_alerts": 'abs:"coincident" AND abs:"alerts" AND abs:"survey"',
    },
    "A3_negflux": {
        "negative_flux_alerts": 'abs:"negative" AND abs:"difference" AND abs:"flux" AND abs:"alert"',
        "dipper_alert_stream": 'abs:"dipper" AND (abs:"ZTF" OR abs:"LSST" OR abs:"alert")',
        "occulter_artificial": 'abs:"occulter" AND (abs:"artificial" OR abs:"technosignature")',
        "megastructure_transit": 'abs:"megastructure" AND abs:"transit"',
        "dyson_transit": 'abs:"Dyson" AND (abs:"transit" OR abs:"occultation")',
        "boyajian_analogue_search": 'abs:"Boyajian" AND abs:"search"',
        "deep_dimming_survey": 'abs:"dimming" AND abs:"survey" AND (abs:"ZTF" OR abs:"LSST" OR abs:"ASAS-SN")',
        "occultation_survey_kbo": 'abs:"occultation" AND abs:"survey" AND abs:"high cadence"',
        "difference_imaging_negative": 'abs:"difference imaging" AND abs:"negative"',
        "achromatic_dimming": 'abs:"grey" AND abs:"dimming" OR abs:"achromatic dimming"',
    },
    "A4_pulse": {
        "optical_pulse_seti": 'abs:"optical" AND abs:"pulse" AND (abs:"SETI" OR abs:"technosignature")',
        "nanosecond_pulse": 'abs:"nanosecond" AND (abs:"pulse" OR abs:"laser") AND abs:"search"',
        "laser_emission_search": 'abs:"laser" AND abs:"emission" AND abs:"search" AND abs:"stars"',
        "panoseti": 'abs:"PANOSETI" OR abs:"pulsed all-sky near-infrared"',
        "optical_seti_survey": 'abs:"optical SETI"',
        "millisecond_optical_transient": 'abs:"millisecond" AND abs:"optical" AND abs:"transient" AND abs:"survey"',
        "single_epoch_streak": 'abs:"streak" AND abs:"survey" AND abs:"detection"',
        "laser_ztf_lsst": 'abs:"laser" AND (abs:"ZTF" OR abs:"LSST" OR abs:"Pan-STARRS")',
    },
    "A5_accel": {
        "nongrav_accel_technosig": 'abs:"non-gravitational" AND (abs:"technosignature" OR abs:"artificial" OR abs:"SETI")',
        "oumuamua_artificial": 'abs:"Oumuamua" AND (abs:"artificial" OR abs:"lightsail" OR abs:"technosignature")',
        "interstellar_object_lsst": 'abs:"interstellar object" AND (abs:"LSST" OR abs:"Rubin")',
        "iso_seti": 'abs:"interstellar object" AND (abs:"technosignature" OR abs:"SETI" OR abs:"probe")',
        "artificial_probe_search": 'abs:"artificial" AND abs:"probe" AND abs:"solar system" AND abs:"search"',
        "lurker_search": 'abs:"lurker" OR (abs:"dormant" AND abs:"probe")',
        "anomalous_astrometry_asteroid": 'abs:"anomalous" AND abs:"astrometry" AND (abs:"asteroid" OR abs:"comet")',
        "dark_comet_accel": 'abs:"dark comet" OR (abs:"nongravitational acceleration" AND abs:"asteroid")',
        "geosync_artefact_search": 'abs:"search" AND abs:"extraterrestrial" AND abs:"artifacts" AND abs:"solar system"',
    },
    "A6_transit": {
        "artificial_transit": 'abs:"artificial" AND abs:"transit" AND (abs:"signature" OR abs:"technosignature")',
        "transit_lightcurve_artificial": 'abs:"transit" AND abs:"signatures" AND abs:"artificial objects"',
        "cloaking_transit": 'abs:"cloaking" AND abs:"transit"',
        "transit_timing_technosig": 'abs:"transit" AND abs:"timing" AND abs:"technosignature"',
        "nonkeplerian_dip": 'abs:"non-Keplerian" AND (abs:"transit" OR abs:"dip")',
        "planetary_transit_zone": 'abs:"transit zone" AND (abs:"SETI" OR abs:"technosignature" OR abs:"Earth")',
        "shape_anomaly_transit": 'abs:"transit" AND abs:"shape" AND abs:"anomal"',
        "swarm_transit": 'abs:"swarm" AND abs:"transit" AND (abs:"artificial" OR abs:"megastructure")',
    },
    "A7_reviews": {
        "technosig_lsst": 'abs:"technosignature" AND abs:"LSST"',
        "technosig_rubin": 'abs:"technosignature" AND abs:"Rubin"',
        "technosig_ztf": 'abs:"technosignature" AND abs:"ZTF"',
        "technosig_alert_broker": 'abs:"technosignature" AND abs:"broker"',
        "technosig_transient": 'abs:"technosignature" AND abs:"transient"',
        "technosig_timedomain": 'abs:"technosignature" AND abs:"time-domain"',
        "seti_lsst": 'abs:"SETI" AND abs:"LSST"',
        "seti_survey_alerts": 'abs:"SETI" AND abs:"alert"',
        "technosig_survey_photometry": 'abs:"technosignature" AND abs:"photometry"',
        "technosig_all_recent": 'abs:"technosignatures"',
        "seti_ellipsoid": 'abs:"SETI Ellipsoid"',
        "exotica_catalog": 'abs:"Exotica"',
        "technosig_roadmap": 'abs:"technosignature" AND (abs:"roadmap" OR abs:"white paper" OR abs:"strategy")',
        "technosig_gaia": 'abs:"technosignature" AND abs:"Gaia"',
        "technosig_atlas_panstarrs": 'abs:"technosignature" AND (abs:"ATLAS" OR abs:"Pan-STARRS")',
    },
    "A8_anomaly": {
        "anomaly_detection_ztf": 'abs:"anomaly detection" AND (abs:"ZTF" OR abs:"Zwicky")',
        "anomaly_detection_lsst": 'abs:"anomaly detection" AND (abs:"LSST" OR abs:"Rubin")',
        "anomaly_technosig": 'abs:"anomaly" AND abs:"technosignature"',
        "snad": 'abs:"SNAD"',
        "active_anomaly_detection": 'abs:"active anomaly detection" OR abs:"active learning" AND abs:"transient"',
        "outlier_colour_space": 'abs:"outlier" AND abs:"colour" AND abs:"survey"',
        "real_time_anomaly_transient": 'abs:"real-time" AND abs:"anomaly" AND abs:"transients"',
        "unknown_unknowns_survey": 'abs:"unknown unknowns" AND abs:"survey"',
        "novelty_detection_astronomy": 'abs:"novelty detection" AND abs:"astronomical"',
        "color_technosignature": 'abs:"color" AND abs:"technosignature"',
    },
}

# OpenAlex: natural-language phrases (title+abstract full-corpus recall)
OAQ: dict[str, dict[str, str]] = {
    "A1_glint": {
        "specular_glint_technosig": "specular glint technosignature artificial reflector",
        "mirror_glint_exoplanet": "specular reflection glint exoplanet ocean detection",
        "achromatic_flash_search": "achromatic optical flash transient search survey",
        "satellite_glint_foreground": "satellite glints orbital foreground short duration optical transients",
        "solar_panel_reflection_seti": "solar panel reflection signature extraterrestrial detection",
        "flat_mirror_seti": "flat mirror artificial reflection SETI optical",
    },
    "A2_recurrence": {
        "repeating_optical_transient_seti": "repeating optical transient same position SETI beacon",
        "recurrence_alert_stream_seti": "recurrent brightening quiescent star technosignature search",
        "vasco_glint_geostationary": "vanishing appearing sources glints geostationary orbit transients",
        "optical_beacon_search": "optical beacon interstellar signalling search survey",
    },
    "A3_negflux": {
        "negative_difference_flux_alert": "negative difference flux alert stream difference image analysis",
        "artificial_occulter_search": "artificial occulter megastructure occultation search survey",
        "dyson_sphere_transit_search": "Dyson sphere partial transit occultation light curve search",
        "grey_achromatic_dimming_seti": "achromatic grey dimming star technosignature dust",
        "dipper_search_ztf_lsst": "dipper stars deep dimming search ZTF survey light curves",
        "high_cadence_occultation": "high cadence occultation survey serendipitous small bodies",
    },
    "A4_pulse": {
        "optical_pulse_imaging_survey": "nanosecond optical pulse search imaging survey extraterrestrial",
        "laser_line_survey_spectra": "laser emission line search stellar spectra SETI",
        "fast_optical_transient_subsecond": "sub-second optical transient search wide field survey",
        "single_image_artifact_pulse": "single epoch detection optical transient laser pulse candidates",
    },
    "A5_accel": {
        "nongrav_accel_seti": "non-gravitational acceleration artificial object interstellar technosignature",
        "iso_lsst_discovery": "interstellar objects LSST Rubin discovery rate detection",
        "lurker_probe_solar_system": "search extraterrestrial artifacts probes solar system lurkers",
        "oumuamua_lightsail": "Oumuamua non-gravitational acceleration lightsail artificial origin",
        "astrometric_anomaly_technosig": "astrometric anomaly technosignature artificial trajectory",
    },
    "A6_transit": {
        "artificial_transit_signature": "transit light curve signatures artificial objects planetary",
        "transit_cloaking_signalling": "cloaking planet transit laser signalling",
        "transit_timing_anomaly_technosig": "transit timing anomaly technosignature artificial",
        "megastructure_lightcurve_shape": "megastructure light curve shape asymmetric transit search",
    },
    "A7_reviews": {
        "technosig_alert_brokers": "technosignature searches real-time alert brokers",
        "lsst_technosignature_review": "LSST Rubin technosignature search opportunities review",
        "technosig_strategy_review": "technosignature search strategy review future surveys",
        "seti_time_domain": "SETI time domain astronomy optical surveys opportunities",
        "rubin_transient_roadmap": "Rubin Observatory LSST transients variable stars roadmap",
        "nine_axes_merit": "axes of merit technosignature search",
    },
    "A8_anomaly": {
        "anomaly_detection_transient_survey": "anomaly detection transient surveys machine learning astronomy",
        "snad_ztf_anomalies": "SNAD anomaly detection Zwicky Transient Facility data release",
        "anomaly_seti_ml": "machine learning anomaly detection SETI technosignature candidate",
        "live_anomaly_extragalactic": "deep learning live anomaly detection extragalactic transients",
        "color_space_outlier_alerts": "color space outliers alert stream classification transients",
    },
}

# arXiv ids we may want to cite -- EVERY one gets its real title verified.
VERIFY_IDS = [
    "2506.14744",  # suspected: Technosignature Searches with Real-time Alert Brokers
    "2208.04499",  # suspected: Rubin LSST Transients and Variable Stars Roadmap
    "2208.02781",  # suspected: From Data to Software to Science with Rubin LSST
    "1812.08681",  # suspected: NASA technosignatures workshop report
    "2010.15577",  # suspected: Breakthrough Listen Exotica Catalog
    "2107.07512",  # suspected: Nine axes of merit for technosignature searches
    "2001.03071",  # suspected: Corbett et al orbital foregrounds / Evryscope
    "0810.1043",   # Arnold-type transit artificial objects?
    "1603.08928",  # Kipping & Teachey cloaking?
    "2211.10748",  # SETI Ellipsoid?
    "2306.11386",  # SETI Ellipsoid TESS?
    "1809.09107",  # Villarroel VASCO?
    "2102.03293",  # ?
    "2012.12742",  # Malanchev SNAD ZTF anomalies?
    "1905.11516",  # Pruzhinskaya anomaly Open Supernova Catalog?
    "2103.12102",  # Villar live anomaly detection?
    "1810.11441",  # ?
    "1701.06592",  # ?
]

# Key documents to pull full text for (enumerate their planned-search lists).
FULLTEXT = {
    "alert_brokers": "2506.14744",
    "rubin_tvs_roadmap": "2208.04499",
    "exotica": "2010.15577",
    "nine_axes": "2107.07512",
}


def strip_html(b: bytes) -> str:
    t = b.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", html.unescape(t))).strip()


def main() -> None:
    print("=== arXiv id verification ===", flush=True)
    arxiv_ids(VERIFY_IDS)

    print("=== arXiv query batteries ===", flush=True)
    for angle, qs in AX.items():
        for name, q in qs.items():
            arxiv_query(angle, name, q)

    print("=== OpenAlex query batteries ===", flush=True)
    for angle, qs in OAQ.items():
        for name, q in qs.items():
            oa_query(angle, name, q)

    print("=== full text of key reviews ===", flush=True)
    for slug, aid in FULLTEXT.items():
        for url in (f"https://arxiv.org/html/{aid}",
                    f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            b = get(url, OUT / f"ar5iv_{slug}.html", tries=2)
            if b and len(b) > 20000:
                (OUT / f"txt_{slug}.txt").write_text(strip_html(b))
                break
        else:
            b = get(f"https://arxiv.org/abs/{aid}", OUT / f"abs_{slug}.html", tries=2)
            if b:
                (OUT / f"txt_{slug}.txt").write_text(strip_html(b))

    (OUT / "counts.json").write_text(json.dumps(COUNTS, indent=1))
    (OUT / "summary.json").write_text(json.dumps(STATUS, indent=1))
    (OUT / "digest.txt").write_text("\n".join(DIGEST))
    ok = sum(1 for s in STATUS if s.get("ok"))
    print(f"\ndone: {ok}/{len(STATUS)} fetches ok", flush=True)


if __name__ == "__main__":
    main()

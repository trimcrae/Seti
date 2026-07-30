#!/usr/bin/env python3
"""VNPROBELIT pass 3 — full text of the papers that decide the verdicts.

Pass 1 ran 101 arXiv queries; pass 2 verified ids, resolved journal-only
titles and pulled citing sets.  Pass 1 surfaced papers that had not been
harvested by web search and whose CONTENT decides whether an angle is
occupied.  This pass reads them:

  2510.17907  A Cost-Effective Search for Extraterrestrial Probes in the
              Solar System            -- executed search, or a proposal?
  2011.12446  Strategies for the Detection of ET Probes Within Our Own Solar
              System (Gertz)          -- what does it actually propose to do?
  1609.04635  ET Probes: Looking Here as Well as There (Gertz)
  1802.01783  LSST Solar System Science Roadmap  -- is any artificiality /
              technosignature / anomalous-acceleration science CLAIMED?
  2007.00693  On a spectral pattern of the Von-Neumann probes
  2110.00406  Can China's FAST telescope detect extraterrestrial von-Neumann
              probes?
  2209.11262  The Inferred Abundance of Interstellar Objects of Technological
              Origin
  2302.01239  SNAPS design/architecture/first data release
  2305.01123  Enabling discovery of solar system objects in large alert data
              streams (Fink SSO module)
  2604.00206  Predictions of the LSST Solar System (non-)Yield
  2606.01288  Is the Dark Comet 1998 KY26 the Spacecraft Phobos 1?
  1111.1212   On the likelihood of non-terrestrial artifacts in the Solar System
  2606.24028  Micron-Scale Technosignatures ... lunar regolith
  2504.01184  Asteroid masses from mutual encounters observed in the LSST
  1612.06920  Non-Gravitational Acceleration of the Active Asteroids
  2412.07603  Two Distinct Populations of Dark Comets
  2310.02733  Seasonally Varying Outgassing as an Explanation for Dark Comet
              Accelerations

Then it greps each text for the discriminating vocabulary and writes a
machine-checkable occupancy matrix (results/vnprobelit3/occupancy.json) so the
verdicts rest on counted term hits in real text rather than recollection.
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

OUT = pathlib.Path("results/vnprobelit3")
OUT.mkdir(parents=True, exist_ok=True)
MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-vnprobelit3/1.0 (mailto:{MAIL})"}
ARXIV = "https://export.arxiv.org/api/query?"
OPENALEX = "https://api.openalex.org/"
STATUS: list[dict] = []
T0 = time.time()
DEADLINE = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0


def out_of_time(margin: float = 90.0) -> bool:
    return (time.time() - T0) > (DEADLINE - margin)


def get(url: str, dest: pathlib.Path | None = None, tries: int = 3,
        pause: float = 2.5) -> bytes | None:
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


FULLTEXT = {
    "cost_effective_probe_search": "2510.17907",
    "gertz_strategies_detect_probes": "2011.12446",
    "gertz_et_probes_here": "1609.04635",
    "sssc_roadmap": "1802.01783",
    "vn_spectral_pattern": "2007.00693",
    "fast_vn_probes": "2110.00406",
    "inferred_abundance_tech_isos": "2209.11262",
    "snaps1_design": "2302.01239",
    "fink_sso_module": "2305.01123",
    "lsst_non_yield": "2604.00206",
    "phobos1_dark_comet": "2606.01288",
    "haqq_misra_likelihood": "1111.1212",
    "micron_scale_technosig": "2606.24028",
    "lsst_asteroid_masses_encounters": "2504.01184",
    "hui_jewitt_active_asteroid_accel": "1612.06920",
    "two_populations_dark_comets": "2412.07603",
    "seasonal_outgassing_dark_comets": "2310.02733",
    "greenberg_yarkovsky_247": "1708.05513",
    "fenucci_neocc_automated": "2311.10175",
    "yarkovsky_main_belt_detectability": "2310.00055",
    "eggl_star_catalog_debias_ii": "1909.04558",
    "vokrouhlicky_yarkovsky_yorp_review": "1502.01249",
    "loeb_turner_alpha_tno": "2605.17098",
    "ellery_selfrep_probes": "2510.00082",
}

# the discriminating vocabulary: does a paper occupy the axis, or only name it?
TERMS = {
    "technosignature": r"technosignature",
    "self_replicating": r"self[- ]replicat|von neumann|self[- ]reproduc",
    "population_family": r"asteroid famil|dynamical famil|hierarchical clustering|family membership",
    "sfd": r"size[- ]frequency distribution|size distribution|power[- ]law slope|cumulative slope",
    "nongrav": r"non[- ]?gravitational",
    "a1a2a3": r"\bA1\b|\bA2\b|\bA3\b|Marsden",
    "yarkovsky": r"yarkovsky",
    "srp": r"solar radiation pressure|radiation pressure|area[- ]to[- ]mass",
    "eph_residual": r"ephemeris residual|observed minus predicted|astrometric residual|O-C residual|ephOffset",
    "alert_stream": r"alert stream|alert broker|alert packet|real[- ]time alert",
    "rubin_lsst": r"\bLSST\b|Rubin Observatory",
    "coorbital": r"co[- ]?orbital|quasi[- ]satellite|Trojan|libration|Lagrang",
    "albedo": r"albedo|phase curve|G12",
    "rotation": r"tumbl|non[- ]principal axis|attitude control|spin state",
    "outlier_detection": r"outlier|anomaly detection|novelty detection",
    "executed_data": r"we (?:analyse|analyze|search|observ|present observ)|our observations|we report the detection|"
                     r"we surveyed|data (?:release|set) of|we obtained \d",
    "proposal_only": r"we propose|is proposed|future (?:work|surveys)|could be searched|we suggest",
}


def strip_html(b: bytes) -> str:
    t = b.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", t))
    return re.sub(r"\n\s*\n\s*\n+", "\n\n",
                  re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", t))).strip()


def fulltext(name: str, aid: str) -> str | None:
    for u in (f"https://arxiv.org/html/{aid}",
              f"https://arxiv.org/html/{aid}v1",
              f"https://ar5iv.labs.arxiv.org/html/{aid}",
              f"https://arxiv.org/abs/{aid}"):
        b = get(u, OUT / f"html_{name}.html", tries=2, pause=2.0)
        if b and len(b) > 25000:
            txt = strip_html(b)[:600000]
            (OUT / f"txt_{name}.txt").write_text(txt)
            return txt
    if (OUT / f"html_{name}.html").exists():
        txt = strip_html((OUT / f"html_{name}.html").read_bytes())[:600000]
        (OUT / f"txt_{name}.txt").write_text(txt)
        return txt
    return None


def main() -> None:
    occ: dict = {}
    for name, aid in FULLTEXT.items():
        if out_of_time():
            print("### deadline", flush=True)
            break
        txt = fulltext(name, aid)
        if txt is None:
            occ[name] = {"arxiv": aid, "status": "NO_TEXT"}
            continue
        low = txt.lower()
        occ[name] = {"arxiv": aid, "chars": len(txt),
                     "status": "FULL" if len(txt) > 25000 else "ABS_ONLY",
                     "terms": {k: len(re.findall(v, low, re.I))
                               for k, v in TERMS.items()}}
    (OUT / "occupancy.json").write_text(json.dumps(occ, indent=1))

    # id verification for everything cited out of pass 1 that pass 2 did not cover
    ids = ["2510.17907", "2011.12446", "1609.04635", "1802.01783", "2007.00693",
           "2110.00406", "2209.11262", "2302.01239", "2305.01123", "2604.00206",
           "2606.01288", "1111.1212", "2606.24028", "2504.01184", "1612.06920",
           "2412.07603", "2310.02733", "1708.05513", "2311.10175", "2310.00055",
           "1909.04558", "1502.01249", "1204.5990", "1212.4812", "1805.05947",
           "2411.09750", "1402.5573", "1811.10953", "2407.01839", "2503.07972",
           "2503.09137", "2503.09668", "2506.09478", "2001.08229", "2102.09059",
           "1111.1127", "1910.07471", "1910.07466", "1608.01518", "2604.13296",
           "2605.27683", "2209.14244", "1903.00770", "2005.12303", "1307.1648",
           "1605.02169", "1111.6131", "1904.04914", "1808.07024", "2604.20896",
           "2111.05334", "2202.03364", "1704.07263", "2603.13177", "2602.19656",
           "2601.12972", "2403.20179", "2506.02779", "2009.04489", "1705.10903"]
    ver: dict = {}
    for i in range(0, len(ids), 10):
        if out_of_time():
            break
        chunk = ids[i:i + 10]
        b = get(ARXIV + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}),
            OUT / f"ax_verify_{i//10}.atom")
        if not b:
            continue
        for e in re.findall(r"<entry>(.*?)</entry>", b.decode("utf-8", "replace"), re.S):
            def f(tag: str, e=e) -> str:
                m = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
                return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip()) if m else ""
            aid = f("id").replace("http://arxiv.org/abs/", "").split("v")[0]
            ver[aid] = {"title": f("title"), "published": f("published")[:10],
                        "authors": [html.unescape(re.sub(r"\s+", " ", a).strip())
                                    for a in re.findall(
                                        r"<author>\s*<name>(.*?)</name>", e, re.S)][:6]}
    for want in ids:
        ver.setdefault(want, {"error": "NOT_RESOLVED"})
    (OUT / "ids_verified.json").write_text(json.dumps(ver, indent=1))

    # citing sets for the two papers that most threaten novelty
    for name, aid in (("ellery_selfrep_probes", "2510.00082"),
                      ("cost_effective_probe_search", "2510.17907"),
                      ("gertz_strategies", "2011.12446")):
        if out_of_time():
            break
        w = get(OPENALEX + "works?" + urllib.parse.urlencode(
            {"filter": f"locations.landing_page_url.search:arxiv.org/abs/{aid}",
             "mailto": MAIL, "per-page": 3}), OUT / f"oa_work_{name}.json",
            tries=2, pause=1.0)
        if not w:
            continue
        try:
            res = json.loads(w).get("results", [])
            if not res:
                continue
            wid = res[0]["id"].rsplit("/", 1)[-1]
        except Exception:  # noqa: BLE001
            continue
        get(OPENALEX + "works?" + urllib.parse.urlencode(
            {"filter": f"cites:{wid}", "per-page": 100, "mailto": MAIL,
             "select": "id,doi,title,publication_year,type"}),
            OUT / f"cites_{name}.json", tries=2, pause=1.0)

    (OUT / "queries.json").write_text(json.dumps(STATUS, indent=1))
    (OUT / "summary.json").write_text(json.dumps({
        "n_fulltext_targets": len(FULLTEXT),
        "n_full": sum(1 for v in occ.values() if v.get("status") == "FULL"),
        "n_abs_only": sum(1 for v in occ.values() if v.get("status") == "ABS_ONLY"),
        "n_no_text": sum(1 for v in occ.values() if v.get("status") == "NO_TEXT"),
        "n_ids_requested": len(ids),
        "n_ids_resolved": sum(1 for v in ver.values() if "error" not in v),
        "ids_not_resolved": [k for k, v in ver.items() if "error" in v],
        "elapsed_s": round(time.time() - T0, 1),
    }, indent=1))
    print(json.dumps(json.loads((OUT / "summary.json").read_text()), indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VNPROBELIT pass 4 — FEASIBILITY of the two Rubin SSO-alert observables.

The novelty question is settled by literature (passes 1-3).  The *soundness*
question is settled by numbers, and there are only two that matter:

  (a) mpc_orbits.{yarkovsky,srp,a1,a2,a3}: how many solar-system objects have a
      fitted non-gravitational parameter AT ALL?  If the answer is hundreds, a
      "population" search on that column is impossible by construction, no
      matter how many alerts arrive.  Measured here against JPL's SBDB query
      API (the MPC and JPL fit the same Marsden decomposition; JPL's API is the
      only one that supports a catalogue-wide constrained count).
  (b) ssSource.ephOffset*: this is delivered for EVERY detection of EVERY known
      object, so it is the observable that actually scales.  Its usefulness is
      set by the size of the *systematic* floor (astrometric + star-catalogue +
      timing + unmodelled-perturbation), which is what the residual would have
      to exceed.  Recorded here from the Rubin documents that define the
      column, plus the survey/astrometric-accuracy numbers.

Also captures the Rubin documents that state what is CLAIMED as planned for
solar-system alerts, so "nobody claims this" is a quotation rather than a
belief: DMTN-087, the DPDD, the SDM schema browser, and the SSSC pages.

Runs on the GitHub Actions runner (the sandbox blocks these hosts).
Outputs under results/vnprobelit4/.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/vnprobelit4")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Seti-vnprobelit4/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
T0 = time.time()
DEADLINE = float(sys.argv[1]) if len(sys.argv) > 1 else 2400.0


def out_of_time(margin: float = 60.0) -> bool:
    return (time.time() - T0) > (DEADLINE - margin)


def get(url: str, dest: pathlib.Path | None = None, tries: int = 3,
        pause: float = 2.0) -> bytes | None:
    last = ""
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=180) as r:
                data, code = r.read(), r.status
            if dest is not None:
                dest.write_bytes(data)
            STATUS.append({"url": url, "dest": dest.name if dest else None,
                           "http": code, "bytes": len(data), "ok": True})
            print(f"OK   {code} {len(data):9d}B  {url[:150]}", flush=True)
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            try:
                last += " :: " + e.read()[:300].decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        print(f"RETRY({i+1}) {last}  {url[:150]}", flush=True)
        time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "dest": dest.name if dest else None,
                   "http": None, "ok": False, "error": last})
    print(f"FAIL {last}  {url[:150]}", flush=True)
    return None


SBDB = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?"


def sbdb(name: str, params: dict) -> dict | None:
    b = get(SBDB + urllib.parse.urlencode(params), OUT / f"sbdb_{name}.json",
            pause=3.0)
    if b is None:
        return None
    try:
        d = json.loads(b)
    except Exception as e:  # noqa: BLE001
        print(f"  parse fail {name}: {e}", flush=True)
        return None
    n = len(d.get("data") or [])
    print(f"  -> {name}: {n} rows", flush=True)
    return d


# constrained counts: how many objects have each non-grav parameter fitted?
# sb-cdata uses JPL's constraint JSON; "DF" = defined (non-null).
def cdata(field: str) -> str:
    return json.dumps({"AND": [f"{field}|DF"]})


COUNTS = [
    ("ast_A1", {"fields": "full_name,A1,A1_sigma,A2,A3,H,diameter,albedo,class",
                "sb-kind": "a", "sb-cdata": cdata("A1"), "limit": 2000}),
    ("ast_A2", {"fields": "full_name,A2,A2_sigma,A1,A3,H,diameter,albedo,class",
                "sb-kind": "a", "sb-cdata": cdata("A2"), "limit": 2000}),
    ("ast_A3", {"fields": "full_name,A3,A3_sigma,A1,A2,H,class",
                "sb-kind": "a", "sb-cdata": cdata("A3"), "limit": 2000}),
    ("com_A1", {"fields": "full_name,A1,A1_sigma,A2,A3,class",
                "sb-kind": "c", "sb-cdata": cdata("A1"), "limit": 2000}),
    ("com_A2", {"fields": "full_name,A2,A2_sigma,A1,A3,class",
                "sb-kind": "c", "sb-cdata": cdata("A2"), "limit": 2000}),
    # denominators
    ("ast_all_count", {"fields": "full_name", "sb-kind": "a", "limit": 1}),
    ("neo_all_count", {"fields": "full_name", "sb-kind": "a",
                       "sb-class": "APO,ATE,AMO,IEO", "limit": 1}),
    # objects with albedo, for the albedo-homogeneity axis
    ("ast_albedo", {"fields": "full_name,albedo,diameter,H,class",
                    "sb-kind": "a", "sb-cdata": cdata("albedo"), "limit": 2}),
]

# Rubin / MPC documents that define the columns and state what is planned
DOCS = {
    "dmtn087": "https://dmtn-087.lsst.io/",
    "dpdd_ss": "https://dp0-3.lsst.io/data-products-dp0-3/index.html",
    "ssp_pipeline": "https://dp0-3.lsst.io/data-products-dp0-3/solar-system-processing-pipeline.html",
    "sdm_apdb": "https://sdm-schemas.lsst.io/apdb.html",
    "sdm_dp03": "https://sdm-schemas.lsst.io/dp03.html",
    "sssc_home": "https://solarsystem.science.lsst.org/",
    "rubin_alerts_news": "https://rubinobservatory.org/news/rubin-first-look/alerts",
    "mpc_orb_format": "https://minorplanetcenter.net/mpcops/documentation/mpcorb-format/",
    "mpc_api_docs": "https://minorplanetcenter.net/mpcops/documentation/",
    "rtn011": "https://rtn-011.lsst.io/",
    "dmtn337": "https://dmtn-337.lsst.io/",
}


def main() -> None:
    stats: dict = {}
    for name, params in COUNTS:
        if out_of_time():
            break
        d = sbdb(name, params)
        if d is None:
            stats[name] = {"status": "QUERY_FAILED"}
            continue
        rows = d.get("data") or []
        fields = d.get("fields") or []
        stats[name] = {"status": "OK", "n_rows": len(rows),
                       "total_reported": d.get("count"),
                       "fields": fields}
        # significance distribution where a sigma column is present
        for val_f, sig_f in (("A1", "A1_sigma"), ("A2", "A2_sigma"),
                             ("A3", "A3_sigma")):
            if val_f in fields and sig_f in fields:
                iv, isg = fields.index(val_f), fields.index(sig_f)
                sig = []
                for r in rows:
                    try:
                        v, s = float(r[iv]), float(r[isg])
                        if s > 0:
                            sig.append(abs(v) / s)
                    except (TypeError, ValueError):
                        continue
                sig.sort()
                if sig:
                    stats[name][f"{val_f}_snr"] = {
                        "n": len(sig),
                        "n_gt3": sum(1 for x in sig if x > 3),
                        "n_gt5": sum(1 for x in sig if x > 5),
                        "median": round(sig[len(sig) // 2], 3),
                        "max": round(sig[-1], 3)}
    (OUT / "nongrav_census.json").write_text(json.dumps(stats, indent=1))

    docs: dict = {}
    for name, url in DOCS.items():
        if out_of_time():
            break
        b = get(url, OUT / f"doc_{name}.html", tries=2, pause=1.5)
        if b is None:
            docs[name] = {"url": url, "status": "FETCH_FAILED"}
            continue
        t = re.sub(r"(?s)<[^>]+>", " ", b.decode("utf-8", "replace"))
        t = re.sub(r"\s+", " ", t)
        (OUT / f"doc_{name}.txt").write_text(t[:400000])
        low = t.lower()
        docs[name] = {"url": url, "status": "OK", "chars": len(t),
                      "terms": {k: len(re.findall(k, low))
                                for k in ("technosignature", "ephoffset",
                                          "yarkovsky", "non-gravitational",
                                          "nongravitational", "artificial",
                                          "anomal", "outlier", "mpc_orbits",
                                          "sssource", "residual")}}
    (OUT / "docs.json").write_text(json.dumps(docs, indent=1))

    (OUT / "queries.json").write_text(json.dumps(STATUS, indent=1))
    (OUT / "summary.json").write_text(json.dumps({
        "nongrav_census": {k: {kk: vv for kk, vv in v.items() if kk != "fields"}
                           for k, v in stats.items()},
        "docs": docs,
        "elapsed_s": round(time.time() - T0, 1),
    }, indent=1))
    print(json.dumps(json.loads((OUT / "summary.json").read_text()), indent=1)[:6000])


if __name__ == "__main__":
    main()

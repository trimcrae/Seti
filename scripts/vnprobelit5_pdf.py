#!/usr/bin/env python3
"""VNPROBELIT pass 5 — PDF text for the papers arXiv serves no HTML for.

Passes 1-4 left five papers with abstract-only capture, and three of them are
the ones that most directly threaten the novelty of a von Neumann-probe
POPULATION search among minor planets:

  2510.00082  Ellery, Technosignatures of Self-Replicating Probes in the Solar
              System.  Abstract says asteroid processing is hard to discern and
              pivots to lunar isotopes -- but does the body anywhere consider
              orbital dynamics, non-gravitational acceleration, families,
              size-frequency distributions, or Rubin?
  2011.12446  Gertz, Strategies for the Detection of ET Probes Within Our Own
              Solar System.  Which strategies, and are any dynamical?
  1609.04635  Gertz, ET Probes: Looking Here as Well as There.
  1903.09582  Benford, Looking for Lurkers.  Proposal or executed observation?
  2405.20176  SNAPS asteroid population outlier detection (full feature list:
              is any dynamical/residual feature used, and is artificiality
              mentioned?)

pypdf text extraction on the runner (the sandbox blocks arxiv.org).
Outputs under results/vnprobelit5/.
"""
from __future__ import annotations

import io
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

OUT = pathlib.Path("results/vnprobelit5")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Seti-vnprobelit5/1.0 (mailto:trimcrae@gmail.com)"}
T0 = time.time()

TARGETS = {
    "ellery_selfrep_probes": "2510.00082",
    "gertz_strategies_detect_probes": "2011.12446",
    "gertz_et_probes_here": "1609.04635",
    "benford_lurkers": "1903.09582",
    "snaps_outliers": "2405.20176",
    "gertz_oumuamua_scout_probes": "1904.04914",
    "gertz_nodes_landbases": "1808.07024",
    "vn_lotka_volterra": "2209.14244",
    "haqq_misra_likelihood": "1111.1212",
    "inferred_abundance_tech_isos": "2209.11262",
}

TERMS = {
    "technosignature": r"technosignature",
    "self_replicating": r"self[- ]replicat|von neumann|self[- ]reproduc",
    "population_family": r"asteroid famil|dynamical famil|family membership|"
                         r"hierarchical clustering|cluster of objects",
    "sfd": r"size[- ]frequency|size distribution|power[- ]law",
    "nongrav": r"non[- ]?gravitational",
    "yarkovsky": r"yarkovsky",
    "srp": r"radiation pressure|area[- ]to[- ]mass",
    "eph_residual": r"ephemeris residual|observed minus predicted|astrometric residual|"
                    r"astrometric anomal|orbit(?:al)? residual",
    "orbit_anomaly": r"anomalous orbit|orbital anomal|trajectory anomal|maneuver|manoeuvre",
    "alert_stream": r"alert stream|alert broker|alert packet",
    "rubin_lsst": r"\bLSST\b|Rubin Observatory|Vera C\. Rubin",
    "coorbital": r"co[- ]?orbital|quasi[- ]satellite|Trojan|libration|Lagrang",
    "albedo": r"albedo|phase curve|reflectance",
    "rotation": r"tumbl|non[- ]principal axis|attitude control|spin state",
    "outlier": r"outlier|anomaly detection",
    "radio_optical_em": r"radio|radar|laser|infrared|spectrum|spectra",
    "moon_lunar": r"\blunar\b|\bMoon\b",
}


def get(url: str, dest: pathlib.Path, tries: int = 3) -> bytes | None:
    last = ""
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=180) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"OK   {len(data):9d}B {url}", flush=True)
            time.sleep(2.5)
            return data
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        print(f"RETRY({i+1}) {last} {url}", flush=True)
        time.sleep(3 * (i + 1))
    print(f"FAIL {last} {url}", flush=True)
    return None


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypdf"],
                   check=False)
    from pypdf import PdfReader  # noqa: PLC0415

    occ: dict = {}
    for name, aid in TARGETS.items():
        pdf = get(f"https://arxiv.org/pdf/{aid}", OUT / f"pdf_{name}.pdf")
        if pdf is None:
            occ[name] = {"arxiv": aid, "status": "NO_PDF"}
            continue
        try:
            reader = PdfReader(io.BytesIO(pdf))
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:  # noqa: BLE001
            occ[name] = {"arxiv": aid, "status": f"EXTRACT_FAIL:{e!r}"}
            continue
        txt = re.sub(r"[ \t]+", " ", txt)
        (OUT / f"txt_{name}.txt").write_text(txt[:900000])
        low = txt.lower()
        occ[name] = {"arxiv": aid, "status": "OK", "pages": len(reader.pages),
                     "chars": len(txt),
                     "terms": {k: len(re.findall(v, low, re.I))
                               for k, v in TERMS.items()}}
        print(f"  {name}: {len(reader.pages)}p {len(txt)}ch "
              f"{occ[name]['terms']}", flush=True)

    (OUT / "occupancy_pdf.json").write_text(json.dumps(occ, indent=1))
    (OUT / "summary.json").write_text(json.dumps({
        "n_targets": len(TARGETS),
        "n_ok": sum(1 for v in occ.values() if v.get("status") == "OK"),
        "failures": {k: v["status"] for k, v in occ.items()
                     if v.get("status") != "OK"},
        "elapsed_s": round(time.time() - T0, 1),
    }, indent=1))
    print(json.dumps(json.loads((OUT / "summary.json").read_text()), indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CHEMLIT/NIST: primary atomic data for the chemical-technosignature feasibility table.

Pull every NIST ASD catalogued line for the radionuclide species (Tc, Pm, Th, U)
and the comparison species, over the full optical->H-band range AND inside each
public spectroscopic survey's band, so that "survey X cannot see element Y" is a
statement backed by primary atomic data rather than recollection.

The first attempt used an output-column set that ASD rejects ("Invalid Column
Setting"); this script uses the parameter set already proven in CI by
src/seti/midden/lines.py, and tries progressively richer column sets, keeping
the richest one that the service actually accepts.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/chemlit_nist")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Seti-chemlit/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
NIST = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?"

# Exactly the parameter set that src/seti/midden/lines.py uses successfully.
BASE = {
    "unit": 0, "format": 1, "line_out": 0, "en_unit": 0, "output": 0,
    "bibrefs": 1, "page_size": 15, "show_obs_wl": 1, "show_calc_wl": 1,
    "order_out": 0, "show_av": 2, "tsb_value": 0, "A_out": 0,
    "allowed_out": 1, "forbid_out": 1, "conf_out": "on", "term_out": "on",
    "enrg_out": "on", "J_out": "on",
}
# Richer variants, tried first; fall back to BASE on "Invalid Column Setting".
VARIANTS = [
    {**BASE, "A_out": 1, "intens_out": "on"},   # + transition probabilities + rel. intensity
    {**BASE, "intens_out": "on"},               # + relative intensity only
    BASE,
]

SPECIES = [
    "Tc I", "Tc II", "Pm I", "Pm II", "Th II", "Th I", "U II", "U I",
    "Pu I", "Np I", "Nb I", "Mo I", "Ru I", "Zr I",
    "Fe I", "Ba II", "Eu II", "Nd II", "Ce II", "La II", "Sr II",
    "P I", "K I", "Li I",
]

BANDS = {
    "full":     (3000, 18000),
    "blue":     (3500, 5000),   # Tc/Pm/Th/U resonance region
    "galah_b":  (4713, 4903),
    "galah_v":  (5648, 5873),
    "galah_r":  (6478, 6737),
    "galah_i":  (7585, 7887),
    "rvs":      (8460, 8700),   # Gaia RVS / RAVE Ca-triplet region
    "apogee":   (15100, 17000),  # APOGEE H band
    "lamost_lrs": (3700, 9000),  # LAMOST low-res / SDSS-BOSS optical
}


def fetch(url: str, dest: pathlib.Path) -> str | None:
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                txt = r.read().decode("utf-8", "replace")
            dest.write_text(txt)
            return txt
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {dest.name}: {e}", flush=True)
            time.sleep(3 * (i + 1))
    return None


def pull(species: str, band: str, lo: float, hi: float) -> None:
    tag = species.replace(" ", "")
    dest = OUT / f"nist_{tag}_{band}.txt"
    for vi, var in enumerate(VARIANTS):
        params = {"spectra": species, "low_w": f"{lo:.2f}", "upp_w": f"{hi:.2f}", **var}
        txt = fetch(NIST + urllib.parse.urlencode(params), dest)
        if txt is None:
            continue
        if "Invalid Column Setting" in txt or "Input Error" in txt:
            time.sleep(1.0)
            continue
        empty = ("No lines are available" in txt) or ("no lines" in txt.lower())
        STATUS.append({"species": species, "band": band, "lo": lo, "hi": hi,
                       "variant": vi, "bytes": len(txt), "empty_result": empty,
                       "ok": True, "file": dest.name})
        print(f"OK   {dest.name:34s} v{vi} {len(txt):8d}B "
              f"{'EMPTY(no lines in band)' if empty else ''}", flush=True)
        return
    STATUS.append({"species": species, "band": band, "ok": False, "file": dest.name})
    print(f"FAIL {dest.name}", flush=True)


for sp in SPECIES:
    pull(sp, "full", *BANDS["full"])
    time.sleep(1.5)

# Per-survey-band pulls: the decisive "can this survey see it at all" test.
for sp in ("Tc I", "Tc II", "Pm I", "Pm II", "Th II", "U II", "P I", "K I",
           "Ba II", "Eu II", "Nd II", "Ce II", "La II", "Sr II"):
    for band, (lo, hi) in BANDS.items():
        if band == "full":
            continue
        pull(sp, band, lo, hi)
        time.sleep(1.2)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} NIST pulls succeeded", flush=True)

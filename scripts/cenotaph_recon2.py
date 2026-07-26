#!/usr/bin/env python3
"""CENOTAPH recon, pass 2: close the gaps left by pass 1.

Pass 1 (scripts/cenotaph_recon.py) nailed AKARI FIS/IRC and IRAS PSC/FSC exactly
and confirmed AllWISE on IRSA. It left four gaps:

  G1  Planck PCCS2 = VizieR J/A+A/594/A26 -- but we still need the PER-FREQUENCY
      sub-table names (353/545/857 GHz) and their exact flux columns.
  G2  Herschel HPPSC / SPIRE SPSC -- absent from VizieR TAP. Confirm whether they
      exist anywhere reachable, and get their sky coverage.
  G3  AKARI FIS/IRC release notes (JAXA) -- both PDF URLs 404'd in pass 1; the
      per-band 5-sigma sensitivities and the confusion discussion live there.
  G4  AllWISE row count + the VizieR mirror's DIFFERENT column naming, plus the
      Vega->Jy zero-magnitude flux densities.

Outputs under results/cenotaph2/.
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/cenotaph2")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Seti-cenotaph2/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
VIZ_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def get(url: str, name: str, tries: int = 3) -> bool:
    dest = OUT / name
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": name, "bytes": len(data), "ok": True})
            print(f"OK   {name:46s} {len(data):9d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {name} -> {e}", flush=True)
            time.sleep(3 * (i + 1))
    STATUS.append({"url": url, "file": name, "ok": False})
    print(f"FAIL {name}  <- {url}", flush=True)
    return False


def viz_tap(query: str, name: str) -> None:
    q = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query})
    get(f"{VIZ_TAP}?{q}", name)
    time.sleep(2)


# --- G1: Planck PCCS2 per-frequency tables -----------------------------------
get("https://cdsarc.cds.unistra.fr/ftp/J/A+A/594/A26/ReadMe", "readme_pccs2.txt")
time.sleep(1)
get("https://cdsarc.cds.unistra.fr/ftp/VIII/91/ReadMe", "readme_pccs1.txt")
time.sleep(1)
for pat in ("J/A+A/594/A26%", "VIII/91%"):
    viz_tap("SELECT table_name, description FROM TAP_SCHEMA.tables "
            f"WHERE table_name LIKE '{pat}'",
            f"tables_{pat.split('/')[-1].strip('%')}.csv")
# the 857/545/353 GHz tables: probe the likely names directly
for ghz in ("857", "545", "353", "217", "143", "100"):
    tid = f"J/A+A/594/A26/pccs2{ghz}"
    get("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source="
        f"{urllib.parse.quote(tid)}&-out.max=3&-out.all", f"sample_pccs2_{ghz}.tsv")
    time.sleep(2)

# --- G2: Herschel point source catalogues ------------------------------------
for nm, u in {
    "esa_herschel_psc": "https://www.cosmos.esa.int/web/herschel/point-source-catalogues",
    "esa_hppsc": "https://www.cosmos.esa.int/web/herschel/pacs-point-source-catalogue",
    "esa_spsc": "https://www.cosmos.esa.int/web/herschel/spire-point-source-catalogue",
    "irsa_tables_herschel": "https://irsa.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(
        {"QUERY": "SELECT table_name, description FROM TAP_SCHEMA.tables "
                  "WHERE table_name LIKE '%erschel%' OR description LIKE '%erschel%'",
         "FORMAT": "csv"}),
}.items():
    get(u, f"{nm}." + ("csv" if "TAP" in u else "html"))
    time.sleep(2)
viz_tap("SELECT table_name, description FROM TAP_SCHEMA.tables "
        "WHERE description LIKE '%PACS%' OR description LIKE '%SPIRE%'",
        "tables_pacs_spire.csv")

# --- G3: AKARI release notes (try every known mirror) ------------------------
for nm, u in {
    "akari_fis_rn_a": "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/AKARI-FIS_BSC_V1_RN.pdf",
    "akari_fis_rn_b": "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/AKARI-FIS_BSC_V1_RN.pdf",
    "akari_psc_index": "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/",
    "akari_irc_rn_a": "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/AKARI-IRC_PSC_V1_RN.pdf",
    "darts_akari": "https://darts.isas.jaxa.jp/astro/akari/catalogue/",
}.items():
    get(u, f"{nm}." + ("pdf" if u.endswith(".pdf") else "html"))
    time.sleep(2)

# --- G4: AllWISE row count + VizieR mirror naming ----------------------------
viz_tap('SELECT COUNT(*) AS n FROM "II/328/allwise"', "count_allwise.csv")
get("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source="
    + urllib.parse.quote("II/328/allwise") + "&-out.max=3&-out.all",
    "sample_allwise_vizier.tsv")
time.sleep(2)
get("https://cdsarc.cds.unistra.fr/ftp/II/328/ReadMe", "readme_allwise.txt")
time.sleep(1)
get("https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html",
    "wise_zeropoints.html")
time.sleep(2)
get("https://irsa.ipac.caltech.edu/TAP/sync?" + urllib.parse.urlencode(
    {"QUERY": "SELECT COUNT(*) AS n FROM allwise_p3as_psd", "FORMAT": "csv"}),
    "irsa_count_allwise.csv")

(OUT / "status.json").write_text(json.dumps(STATUS, indent=1))
ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n=== {ok}/{len(STATUS)} OK ===", flush=True)
for s in STATUS:
    if not s.get("ok"):
        print("  MISSING", s["file"], flush=True)

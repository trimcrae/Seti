#!/usr/bin/env python3
"""Catalogue-metadata reconnaissance for far-IR all-sky catalogues.

Pure metadata: fetches VizieR ReadMe files, TAP_SCHEMA table/column listings,
and exact row counts for the AKARI and IRAS all-sky catalogues, then writes
everything under results/catrecon/. No science, no big downloads.
"""
import json
import os
import time
import urllib.parse

import requests

OUT = os.path.join("results", "catrecon")
os.makedirs(OUT, exist_ok=True)

TAP_MIRRORS = [
    "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
    "https://tapvizier.u-strasbg.fr/TAPVizieR/tap",
]

S = requests.Session()
S.headers["User-Agent"] = "seti-catrecon/1.0 (metadata reconnaissance)"


def log(*a):
    print(*a, flush=True)


def get(url, timeout=180, tries=3):
    last = None
    for i in range(tries):
        try:
            r = S.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            last = "HTTP %d" % r.status_code
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(3 * (i + 1))
    log("  FAIL %s -> %s" % (url, last))
    return None


def tap(query, fmt="csv", timeout=300):
    """Run a synchronous ADQL query against VizieR TAP, trying both mirrors."""
    q = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": fmt, "QUERY": query}
    )
    for base in TAP_MIRRORS:
        url = base + "/sync?" + q
        try:
            r = S.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            log("  TAP %s -> HTTP %d" % (base, r.status_code))
            log("  body head: %s" % r.text[:400].replace("\n", " "))
        except Exception as e:  # noqa: BLE001
            log("  TAP %s -> %r" % (base, e))
    return None


def write(name, text):
    if text is None:
        return
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)
    log("  wrote %s (%d bytes)" % (p, len(text)))


# ---------------------------------------------------------------- ReadMe files
# Byte-by-byte column definitions: the single most authoritative artifact.
READMES = {
    "II_298": ["II/298"],   # AKARI/FIS Bright Source Catalogue
    "II_297": ["II/297"],   # AKARI/IRC Point Source Catalogue
    "II_125": ["II/125"],   # IRAS PSC
    "II_156A": ["II/156A"],  # IRAS FSC
    "II_156": ["II/156"],   # IRAS FSC (older designation, if present)
    "II_275": ["II/275"],   # IRAS Small Scale Structure Catalog (candidate)
    "II_126": ["II/126"],   # IRAS Small Scale Structure Catalog (candidate)
    "II_124": ["II/124"],   # IRAS Serendipitous Survey Catalog (candidate)
}
READ_BASES = [
    "https://cdsarc.cds.unistra.fr/ftp/{cat}/ReadMe",
    "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/{cat}",
    "https://cdsarc.u-strasbg.fr/ftp/{cat}/ReadMe",
]

log("=== ReadMe files ===")
for key, cats in READMES.items():
    for cat in cats:
        got = None
        for base in READ_BASES:
            got = get(base.format(cat=cat))
            if got and len(got) > 200:
                break
        if got:
            write("readme_%s.txt" % key, got)
        else:
            log("  NO ReadMe for %s" % cat)

# ------------------------------------------------------------ TAP_SCHEMA.tables
log("=== TAP_SCHEMA.tables (targeted) ===")
tbl_q = (
    "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE "
    "table_name LIKE 'II/298%' OR table_name LIKE 'II/297%' OR "
    "table_name LIKE 'II/125%' OR table_name LIKE 'II/156%' OR "
    "table_name LIKE 'II/275%' OR table_name LIKE 'II/126%' OR "
    "table_name LIKE 'II/124%'"
)
write("tables_targeted.csv", tap(tbl_q))

log("=== TAP_SCHEMA.tables (all AKARI) ===")
write(
    "tables_akari.csv",
    tap(
        "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE "
        "description LIKE '%AKARI%' OR description LIKE '%Akari%' OR "
        "table_name LIKE '%akari%'"
    ),
)

log("=== TAP_SCHEMA.tables (all IRAS) ===")
write(
    "tables_iras.csv",
    tap(
        "SELECT table_name, description FROM TAP_SCHEMA.tables WHERE "
        "description LIKE '%IRAS%'"
    ),
)

# ----------------------------------------------------------- TAP_SCHEMA.columns
CAND_TABLES = [
    "II/298/fis",
    "II/297/irc",
    "II/125/main",
    "II/156A/main",
    "II/156/main",
]

log("=== TAP_SCHEMA.columns per candidate table ===")
for t in CAND_TABLES:
    safe = t.replace("/", "_")
    q = (
        "SELECT column_name, datatype, unit, ucd, description FROM "
        "TAP_SCHEMA.columns WHERE table_name = '%s'" % t
    )
    write("columns_%s.csv" % safe, tap(q))

# Broad sweep: every column of every table under these catalogues, in case the
# table suffix differs from the guess above.
log("=== TAP_SCHEMA.columns (prefix sweep) ===")
for pref in ["II/298", "II/297", "II/125", "II/156A", "II/156"]:
    safe = pref.replace("/", "_")
    q = (
        "SELECT table_name, column_name, datatype, unit, ucd, description FROM "
        "TAP_SCHEMA.columns WHERE table_name LIKE '%s%%'" % pref
    )
    write("colsweep_%s.csv" % safe, tap(q))

# ------------------------------------------------------------------ row counts
log("=== row counts ===")
counts = {}
for t in CAND_TABLES:
    txt = tap('SELECT COUNT(*) AS n FROM "%s"' % t)
    counts[t] = (txt or "").strip().splitlines()[-1:] or ["FAILED"]
    log("  %-16s %s" % (t, counts[t]))
write("rowcounts.json", json.dumps(counts, indent=2))

# ------------------------------------------- quality-flag population histograms
# Confirms flag value semantics empirically (how many rows carry each value).
log("=== flag histograms (empirical, confirms flag domains) ===")
HIST = [
    ("II/298/fis", ["FQUAL65", "FQUAL90", "FQUAL140", "FQUAL160"]),
    ("II/297/irc", ["q_S09", "q_S18", "Fqual09", "Fqual18"]),
    ("II/125/main", ["q_Fnu_12", "q_Fnu_25", "q_Fnu_60", "q_Fnu_100"]),
    ("II/156A/main", ["q_Fnu_12", "q_Fnu_25", "q_Fnu_60", "q_Fnu_100"]),
]
hist = {}
for t, cols in HIST:
    for c in cols:
        txt = tap(
            'SELECT "%s" AS v, COUNT(*) AS n FROM "%s" GROUP BY "%s" ORDER BY 1'
            % (c, t, c)
        )
        key = "%s.%s" % (t, c)
        if txt and "error" not in txt.lower()[:300]:
            hist[key] = txt.strip()[:4000]
            log("  %s OK" % key)
        else:
            hist[key] = "FAILED/absent: " + (txt or "")[:300].replace("\n", " ")
            log("  %s absent-or-error" % key)
write("flag_histograms.json", json.dumps(hist, indent=2))

# ------------------------------------------------------- sample rows (3 per cat)
log("=== sample rows ===")
for t in CAND_TABLES:
    safe = t.replace("/", "_")
    write("sample_%s.csv" % safe, tap('SELECT TOP 3 * FROM "%s"' % t))

# --------------------------------------------------- astroquery catalogue search
log("=== astroquery Vizier.find_catalogs ===")
try:
    from astroquery.vizier import Vizier

    for word in ["AKARI", "IRAS"]:
        try:
            cats = Vizier.find_catalogs(word, max_catalogs=500)
            lines = ["%s\t%s" % (k, v.description) for k, v in cats.items()]
            write("find_catalogs_%s.txt" % word, "\n".join(sorted(lines)))
        except Exception as e:  # noqa: BLE001
            log("  find_catalogs(%s) failed: %r" % (word, e))
except Exception as e:  # noqa: BLE001
    log("  astroquery unavailable: %r" % e)

log("=== DONE ===")
for f in sorted(os.listdir(OUT)):
    log("  %s  %d bytes" % (f, os.path.getsize(os.path.join(OUT, f))))

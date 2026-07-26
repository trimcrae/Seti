#!/usr/bin/env python3
"""Catalogue-metadata recon, pass 2.

Pass 1 established that VizieR's TAP_SCHEMA stores table_name WITH embedded
single quotes (e.g. the literal value is 'II/298/fis'), which made the pass-1
column queries return nothing. This pass fixes the quoting, and uses the real
column names that pass 1 recovered from SELECT * headers.
"""
import json
import os
import urllib.parse

import requests

OUT = os.path.join("results", "catrecon")
os.makedirs(OUT, exist_ok=True)

TAP_MIRRORS = [
    "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
    "https://tapvizier.u-strasbg.fr/TAPVizieR/tap",
]
S = requests.Session()
S.headers["User-Agent"] = "seti-catrecon/2.0 (metadata reconnaissance)"


def log(*a):
    print(*a, flush=True)


def tap(query, fmt="csv", timeout=300):
    q = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": fmt, "QUERY": query}
    )
    for base in TAP_MIRRORS:
        try:
            r = S.get(base + "/sync?" + q, timeout=timeout)
            if r.status_code == 200:
                return r.text
            log("  TAP %s HTTP %d :: %s" % (base, r.status_code,
                                            r.text[:250].replace("\n", " ")))
        except Exception as e:  # noqa: BLE001
            log("  TAP %s %r" % (base, e))
    return None


def write(name, text):
    if text is None:
        return
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)
    log("  wrote %s (%d bytes)" % (p, len(text)))


def adql_lit(value):
    """ADQL string literal for a python string (doubling internal quotes)."""
    return "'" + value.replace("'", "''") + "'"


TABLES = ["II/298/fis", "II/297/irc", "II/125/main", "II/156A/main"]

# ------------------------------------------------ TAP_SCHEMA.columns (fixed)
log("=== TAP_SCHEMA.columns with corrected quoting ===")
for t in TABLES:
    safe = t.replace("/", "_")
    stored = "'" + t + "'"          # the value actually stored by VizieR
    for tag, where in [
        ("q", "table_name = " + adql_lit(stored)),
        ("like", "table_name LIKE " + adql_lit("%" + t + "%")),
    ]:
        txt = tap(
            "SELECT table_name, column_name, datatype, arraysize, unit, ucd, "
            "description FROM TAP_SCHEMA.columns WHERE " + where
        )
        if txt and len(txt.strip().splitlines()) > 1:
            write("cols2_%s.csv" % safe, txt)
            log("  %s via %s: %d rows" % (t, tag,
                                          len(txt.strip().splitlines()) - 1))
            break
        log("  %s via %s: empty" % (t, tag))

# ------------------------------------------------------- flag distributions
log("=== flag value histograms (real column names) ===")
HIST = {
    "II/298/fis": ["q_S65", "q_S90", "q_S140", "q_S160",
                   "f_S65", "f_S90", "f_S140", "f_S160",
                   "M65", "M90", "M140", "M160"],
    "II/297/irc": ["q_S09", "q_S18", "f09", "f18", "X09", "X18"],
    "II/125/main": ["q_Fnu_12", "q_Fnu_25", "q_Fnu_60", "q_Fnu_100",
                    "Cirr1", "Cirr2", "Cirr3", "Confuse", "Disc",
                    "HSDFlag", "IDType", "NHcon"],
    "II/156A/main": ["q_Fnu12", "q_Fnu25", "q_Fnu60", "q_Fnu100",
                     "Cir1", "Conf", "Type"],
}
hist = {}
for t, cols in HIST.items():
    for c in cols:
        txt = tap('SELECT "%s" AS v, COUNT(*) AS n FROM "%s" '
                  'GROUP BY "%s" ORDER BY 1' % (c, t, c))
        key = "%s.%s" % (t, c)
        ok = txt and len(txt.strip().splitlines()) > 1 and \
            "error" not in txt[:300].lower()
        hist[key] = txt.strip()[:3000] if ok else \
            "FAILED: " + (txt or "")[:250].replace("\n", " ")
        log("  %-24s %s" % (key, "OK" if ok else "FAIL"))
write("flag_hist2.json", json.dumps(hist, indent=2))

# ------------------------------------- empirical faint-end / sensitivity floor
log("=== empirical flux floor for high-quality sources ===")
FLOOR = [
    ("II/298/fis", "S65", "q_S65"), ("II/298/fis", "S90", "q_S90"),
    ("II/298/fis", "S140", "q_S140"), ("II/298/fis", "S160", "q_S160"),
    ("II/297/irc", "S09", "q_S09"), ("II/297/irc", "S18", "q_S18"),
    ("II/125/main", "Fnu_12", "q_Fnu_12"),
    ("II/125/main", "Fnu_25", "q_Fnu_25"),
    ("II/125/main", "Fnu_60", "q_Fnu_60"),
    ("II/125/main", "Fnu_100", "q_Fnu_100"),
    ("II/156A/main", "Fnu12", "q_Fnu12"),
    ("II/156A/main", "Fnu25", "q_Fnu25"),
    ("II/156A/main", "Fnu60", "q_Fnu60"),
    ("II/156A/main", "Fnu100", "q_Fnu100"),
]
floor = {}
for t, f, qf in FLOOR:
    txt = tap('SELECT COUNT(*) AS n, MIN("%s") AS fmin, MAX("%s") AS fmax, '
              'AVG("%s") AS favg FROM "%s" WHERE "%s" = 3 AND "%s" > 0'
              % (f, f, f, t, qf, f))
    key = "%s.%s(q=3)" % (t, f)
    floor[key] = (txt or "FAILED").strip()[:600]
    log("  %-28s %s" % (key, floor[key].replace("\n", " | ")[:120]))
write("flux_floor.json", json.dumps(floor, indent=2))

# ------------------------------------------------ catalogue discovery sweeps
log("=== catalogue discovery ===")
DISC = {
    "disc_sss.csv": "description LIKE '%Small Scale Structure%'",
    "disc_serendip.csv": "description LIKE '%Serendipitous%'",
    "disc_reject.csv": "description LIKE '%Reject%'",
    "disc_faintsource.csv": "description LIKE '%Faint Source%'",
    "disc_akari_v2.csv": ("description LIKE '%AKARI%' AND "
                          "(description LIKE '%2.0%' OR "
                          "description LIKE '%Version 2%' OR "
                          "description LIKE '%version 2%')"),
}
for name, where in DISC.items():
    write(name, tap("SELECT table_name, description FROM TAP_SCHEMA.tables "
                    "WHERE " + where))

# all tables whose table_name is under the IRAS/AKARI catalogue numbers
write("disc_prefixes.csv",
      tap("SELECT table_name, description FROM TAP_SCHEMA.tables WHERE "
          "table_name LIKE '%II/125/%' OR table_name LIKE '%II/126/%' OR "
          "table_name LIKE '%II/156A/%' OR table_name LIKE '%II/275/%' OR "
          "table_name LIKE '%II/297/%' OR table_name LIKE '%II/298/%' OR "
          "table_name LIKE '%II/327/%'"))

# row counts for the extra catalogues
log("=== extra row counts ===")
EXTRA = ["II/275/fsr", "II/126/main", "II/125/assoc", "II/156A/assoc",
         "II/327/catalog", "II/126/sssc", "II/126/catalog"]
extra = {}
for t in EXTRA:
    txt = tap('SELECT COUNT(*) AS n FROM "%s"' % t)
    val = (txt or "FAILED").strip().splitlines()[-1]
    extra[t] = val
    log("  %-20s %s" % (t, val))
write("extra_rowcounts.json", json.dumps(extra, indent=2))

# ---------------------------------------------------- AKARI FIS release note
log("=== AKARI release notes ===")
RN = [
    ("fis_rn.pdf",
     "http://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/"
     "AKARI-FIS_BSC_V1_RN.pdf"),
    ("irc_rn.pdf",
     "http://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/"
     "AKARI-IRC_PSC_V1_RN.pdf"),
    ("psc_public.html",
     "http://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/"),
]
notes = {}
for name, url in RN:
    try:
        r = S.get(url, timeout=120)
        notes[name] = "HTTP %d, %d bytes" % (r.status_code, len(r.content))
        if r.status_code == 200:
            with open(os.path.join(OUT, name), "wb") as f:
                f.write(r.content)
    except Exception as e:  # noqa: BLE001
        notes[name] = repr(e)
    log("  %-18s %s" % (name, notes[name]))
write("release_notes_status.json", json.dumps(notes, indent=2))

# ---------------------------------------------------- extra ReadMe fetches
log("=== extra ReadMes ===")
for cat in ["II/275", "II/126", "II/327", "II/122B"]:
    for base in ["https://cdsarc.cds.unistra.fr/ftp/{c}/ReadMe",
                 "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/{c}"]:
        try:
            r = S.get(base.format(c=cat), timeout=120)
            if r.status_code == 200 and len(r.text) > 200:
                write("readme2_%s.txt" % cat.replace("/", "_"), r.text)
                break
        except Exception as e:  # noqa: BLE001
            log("  %s %r" % (cat, e))

log("=== DONE ===")
for f in sorted(os.listdir(OUT)):
    log("  %s  %d bytes" % (f, os.path.getsize(os.path.join(OUT, f))))

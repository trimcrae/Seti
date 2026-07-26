#!/usr/bin/env python3
"""Catalogue-metadata recon, pass 4: query-shape canaries.

`SELECT *` on II/125/main and II/156A/main returned no J2000/ICRS column,
while VII/73, II/126 and II/275 did expose _RA_icrs/_DE_icrs. That difference
decides whether an IRAS cross-match must precess B1950 by hand, so verify it
directly. Also canary the exact ADQL shapes a production cross-match job would
use (cone CONTAINS, JOIN), so a real job never fails on syntax.
"""
import json
import os
import urllib.parse

import requests

OUT = os.path.join("results", "catrecon")
os.makedirs(OUT, exist_ok=True)
BASE = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
S = requests.Session()
S.headers["User-Agent"] = "seti-catrecon/4.0 (metadata reconnaissance)"


def log(*a):
    print(*a, flush=True)


def tap(query, timeout=300):
    p = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
         "QUERY": query}
    try:
        r = S.get(BASE + "/sync?" + urllib.parse.urlencode(p), timeout=timeout)
        return r.status_code, r.text
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


def probe(label, query):
    code, txt = tap(query)
    head = (txt or "")[:400].replace("\n", " | ")
    ok = code == 200 and "ERROR" not in (txt or "")[:200].upper() \
        and "<" not in (txt or "")[:3]
    log("  [%s] %-46s HTTP=%s" % ("OK " if ok else "ERR", label, code))
    log("        %s" % head[:300])
    return {"query": query, "http": code, "ok": bool(ok), "head": head[:600]}


res = {}

log("=== J2000 / ICRS column availability ===")
for t in ["II/125/main", "II/156A/main", "II/275/fsr", "II/126/sources",
          "VII/73/irassss", "II/298/fis", "II/297/irc"]:
    res["icrs::" + t] = probe(
        "_RA_icrs on " + t,
        'SELECT TOP 1 "_RA_icrs", "_DE_icrs" FROM "%s"' % t)

log("=== RAJ2000 availability (AKARI) ===")
for t in ["II/298/fis", "II/297/irc"]:
    res["raj2000::" + t] = probe(
        "RAJ2000 on " + t,
        'SELECT TOP 1 "RAJ2000", "DEJ2000" FROM "%s"' % t)

log("=== B1950 columns on IRAS PSC/FSC ===")
for t, ra, de in [("II/125/main", "RA1950", "DE1950"),
                  ("II/156A/main", "RA1950", "DE1950"),
                  ("II/275/fsr", "RAB1950", "DEB1950")]:
    res["b1950::" + t] = probe(
        "%s/%s on %s" % (ra, de, t),
        'SELECT TOP 1 "%s", "%s" FROM "%s"' % (ra, de, t))

log("=== cone-search canary (ADQL CONTAINS) ===")
res["cone_fis"] = probe(
    "cone on II/298/fis via RAJ2000",
    'SELECT COUNT(*) AS n FROM "II/298/fis" WHERE '
    "CONTAINS(POINT('ICRS', \"RAJ2000\", \"DEJ2000\"), "
    "CIRCLE('ICRS', 45.0, 0.0, 1.0)) = 1")
res["cone_psc_1950"] = probe(
    "cone on II/125/main via RA1950 (B1950 values!)",
    'SELECT COUNT(*) AS n FROM "II/125/main" WHERE '
    "CONTAINS(POINT('ICRS', \"RA1950\", \"DE1950\"), "
    "CIRCLE('ICRS', 45.0, 0.0, 1.0)) = 1")

log("=== JOIN canary (FIS x IRC) ===")
res["join_fis_irc"] = probe(
    "FIS x IRC positional join, 1 deg patch",
    'SELECT COUNT(*) AS n FROM "II/298/fis" AS f '
    'JOIN "II/297/irc" AS i ON 1=CONTAINS('
    "POINT('ICRS', f.\"RAJ2000\", f.\"DEJ2000\"), "
    "CIRCLE('ICRS', i.\"RAJ2000\", i.\"DEJ2000\", 0.0055)) "
    'WHERE f."RAJ2000" BETWEEN 45.0 AND 46.0 '
    'AND f."DEJ2000" BETWEEN 0.0 AND 1.0')

log("=== quality-cut canary (the actual science filter shape) ===")
res["qualcut"] = probe(
    "FIS 4-band high-quality count",
    'SELECT COUNT(*) AS n FROM "II/298/fis" WHERE '
    '"q_S65"=3 AND "q_S90"=3 AND "q_S140"=3 AND "q_S160"=3')
res["qualcut_90_160"] = probe(
    "FIS q_S90=3 AND q_S160=3",
    'SELECT COUNT(*) AS n FROM "II/298/fis" WHERE '
    '"q_S90"=3 AND "q_S160"=3')
res["fsc_qual"] = probe(
    "FSC 12+25+60 high quality",
    'SELECT COUNT(*) AS n FROM "II/156A/main" WHERE '
    '"q_Fnu12"=3 AND "q_Fnu25"=3 AND "q_Fnu60"=3')

log("=== unquoted-identifier canary (does VizieR need quotes?) ===")
res["unquoted"] = probe(
    "unquoted columns on II/298/fis",
    'SELECT TOP 1 objID, q_S90, S90 FROM "II/298/fis"')

with open(os.path.join(OUT, "query_canaries.json"), "w") as f:
    json.dump(res, f, indent=2)
log("wrote query_canaries.json")
log("=== DONE ===")

#!/usr/bin/env python3
"""Catalogue-metadata recon, pass 3.

Fills the three gaps left by passes 1-2:
  (a) AKARI FIS/IRC official per-band detection limits (the JAXA release-note
      URLs used in pass 2 all 404'd -- try the current site layout + DARTS),
  (b) row counts / table names for the secondary IRAS catalogues,
  (c) the sync-query row cap (MAXREC) actually enforced by VizieR TAP, which
      silently truncates large SELECTs if not set explicitly.
"""
import json
import os
import urllib.parse

import requests

OUT = os.path.join("results", "catrecon")
os.makedirs(OUT, exist_ok=True)

BASE = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
S = requests.Session()
S.headers["User-Agent"] = "seti-catrecon/3.0 (metadata reconnaissance)"


def log(*a):
    print(*a, flush=True)


def write(name, text):
    if text is None:
        return
    with open(os.path.join(OUT, name), "w", encoding="utf-8",
              errors="replace") as f:
        f.write(text)
    log("  wrote %s (%d bytes)" % (name, len(text)))


def tap(query, extra=None, timeout=600):
    p = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
         "QUERY": query}
    if extra:
        p.update(extra)
    try:
        r = S.get(BASE + "/sync?" + urllib.parse.urlencode(p), timeout=timeout)
        if r.status_code == 200:
            return r.text
        log("  HTTP %d :: %s" % (r.status_code, r.text[:200]))
    except Exception as e:  # noqa: BLE001
        log("  %r" % e)
    return None


# ------------------------------------------------- (c) MAXREC / row-cap probe
log("=== sync-query row cap (MAXREC) probe ===")
cap = {}
for tag, extra in [("default", None), ("MAXREC=200000", {"MAXREC": "200000"}),
                   ("MAXREC=-1", {"MAXREC": "-1"})]:
    txt = tap('SELECT objID FROM "II/298/fis"', extra=extra)
    n = (len(txt.strip().splitlines()) - 1) if txt else -1
    over = bool(txt and "OVERFLOW" in txt.upper())
    cap[tag] = {"rows_returned": n, "overflow_marker": over}
    log("  %-16s rows=%s overflow=%s" % (tag, n, over))
# also confirm TOP works as an explicit cap
txt = tap('SELECT TOP 100000 objID FROM "II/298/fis"')
cap["TOP 100000"] = {"rows_returned":
                     (len(txt.strip().splitlines()) - 1) if txt else -1}
log("  TOP 100000       rows=%s" % cap["TOP 100000"]["rows_returned"])
write("rowcap_probe.json", json.dumps(cap, indent=2))

# ------------------------------------------ (b) secondary IRAS catalogue rows
log("=== secondary catalogue row counts ===")
sec = {}
for t in ["VII/73/irassss", "II/126/sources", "II/126/assoc", "II/274/iras_r",
          "II/275/assoc", "II/327/ysoc", "II/338/catalog", "III/197/lrs"]:
    txt = tap('SELECT COUNT(*) AS n FROM "%s"' % t)
    sec[t] = (txt or "FAILED").strip().splitlines()[-1]
    log("  %-18s %s" % (t, sec[t]))
write("secondary_rowcounts.json", json.dumps(sec, indent=2))

# column headers for the secondary catalogues that matter
for t in ["VII/73/irassss", "II/126/sources", "II/275/fsr", "II/338/catalog"]:
    txt = tap('SELECT TOP 2 * FROM "%s"' % t)
    if txt:
        write("sample_%s.csv" % t.replace("/", "_"), txt)

# ------------------------------------------------ (a) AKARI detection limits
log("=== AKARI release notes / site ===")
URLS = [
    ("fis_rn_v1.pdf", "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/"
                      "Public/RN/AKARI-FIS_BSC_V1_RN.pdf"),
    ("fis_rn_v2.pdf", "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/"
                      "Public/RN/AKARI-FIS_BSC_V2_RN.pdf"),
    ("irc_rn_v1.pdf", "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/"
                      "Public/RN/AKARI-IRC_PSC_V1_RN.pdf"),
    ("akari_psc_index.html",
     "https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/"),
    ("akari_index.html", "https://www.ir.isas.jaxa.jp/AKARI/"),
    ("darts_akari.html", "https://darts.isas.jaxa.jp/astro/akari/"),
    ("darts_cas.html", "https://darts.isas.jaxa.jp/astro/akari/cas.html"),
]
st = {}
for name, url in URLS:
    try:
        r = S.get(url, timeout=180, allow_redirects=True)
        st[name] = "HTTP %d, %d bytes, url=%s" % (
            r.status_code, len(r.content), r.url)
        if r.status_code == 200 and len(r.content) > 500:
            with open(os.path.join(OUT, name), "wb") as f:
                f.write(r.content)
    except Exception as e:  # noqa: BLE001
        st[name] = repr(e)
    log("  %-22s %s" % (name, st[name]))
write("release_notes_status3.json", json.dumps(st, indent=2))

log("=== DONE ===")

#!/usr/bin/env python3
"""Dyson/IR-excess literature survey, pass 2: full-text mining for exact cuts.

Pass 1 (scripts/litcheck_dyson.py) collected metadata + abstracts. Abstracts do
not carry the numbers this survey needs -- sample sizes, SNR/chi thresholds,
colour-excess cuts, dust-temperature and covering-fraction ranges. This pass
pulls arXiv full-text HTML (falling back to the abs page) for every paper in the
pass-1 harvest whose title/abstract marks it as relevant, plus an explicit
must-have list, and greps each one for the quantitative selection language.

Outputs under results/litcheck_dyson/fulltext/:
  <arxivid>.html        - full text (arXiv HTML5, else abs page)
  extract_<arxivid>.txt - lines matching the quantitative-cut regexes
  relevant.json         - the papers selected, with why
  summary2.json         - fetch status
"""
from __future__ import annotations

import glob
import html
import json
import os
import pathlib
import re
import time
import urllib.parse
import urllib.request

BASE = pathlib.Path("results/litcheck_dyson")
OUT = BASE / "fulltext"
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-litcheck-dyson2/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []
NSP = re.compile(r"\s+")


def get(url: str, dest: pathlib.Path, tries: int = 2, pause: float = 4.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:34s} {len(data):9d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


def clean(s: str) -> str:
    return NSP.sub(" ", html.unescape(s)).strip()


# --- 1. Harvest candidate arXiv IDs from pass-1 outputs ----------------------
RELEVANT = re.compile(
    r"dyson|dysonian|megastructur|megaswarm|waste[- ]heat|technosignature|"
    r"kardashev|extraterrestrial|infrared excess|IR excess|mid-infrared excess|"
    r"debris disc|debris disk|circumstellar dust|exozodi|disk detective|"
    r"WISE|unWISE|CatWISE|NEOWISE|AllWISE", re.I)
# Reject obvious off-topic hits that share a keyword (e.g. "wise" inside words is
# handled by the \b in WISE above only partially; add an explicit veto list).
VETO = re.compile(r"piecewise|likewise|otherwise|wisely", re.I)


def harvest() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for f in sorted(glob.glob(str(BASE / "*.atom"))):
        d = open(f, encoding="utf-8", errors="replace").read()
        for e in re.findall(r"<entry>(.*?)</entry>", d, re.S):
            def g(tag):
                m = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
                return clean(m.group(1)) if m else ""
            idu = g("id")
            if "/abs/" not in idu:
                continue
            aid = idu.split("/abs/")[-1]
            aid_nov = re.sub(r"v\d+$", "", aid)
            blob = g("title") + " " + g("summary")
            if VETO.search(g("title")) or not RELEVANT.search(blob):
                continue
            found.setdefault(aid_nov, {"arxiv": aid_nov, "title": g("title"),
                                       "why": "pass1-atom", "src": os.path.basename(f)})
    for f in sorted(glob.glob(str(BASE / "*.json"))):
        if os.path.basename(f) in ("summary.json", "summary2.json"):
            continue
        try:
            d = json.load(open(f, encoding="utf-8", errors="replace"))
        except Exception:
            continue
        rows = []
        if isinstance(d, dict):
            if "data" in d:
                rows = [r.get("citingPaper") or r.get("paper") or r for r in d["data"]]
            elif "results" in d:
                rows = d["results"]
            else:
                rows = [d]
        for p in rows:
            if not isinstance(p, dict):
                continue
            title = p.get("title") or p.get("display_name") or ""
            abst = p.get("abstract") or ""
            if VETO.search(title) or not RELEVANT.search(title + " " + abst):
                continue
            aid = (p.get("externalIds") or {}).get("ArXiv") or ""
            if not aid:
                loc = (p.get("primary_location") or {}).get("landing_page_url") or ""
                m = re.search(r"arxiv\.org/abs/([\d.]+)", loc)
                aid = m.group(1) if m else ""
            if not aid:
                continue
            aid = re.sub(r"v\d+$", "", aid)
            found.setdefault(aid, {"arxiv": aid, "title": title,
                                   "why": "pass1-json", "src": os.path.basename(f)})
    return found


# Explicit must-haves (fetched even if pass 1 missed them).
MUST = {
    "1409.5104": "Patel Metchev Heinze 2014 WISE debris disks (verify id)",
    "1606.01755": "Cotten & Song 2016 census of IR excess stars (verify id)",
    "1211.7095": "Kennedy & Wyatt exozodi luminosity function (verify id)",
    "1510.05610": "Silverberg+ Disk Detective (verify id)",
    "1907.07829": "Wright, Zackrisson & Lisse - Technosignatures in the Thermal IR",
    "2110.13887": "Huston & Wright 2021 - Dyson sphere feedback",
    "1807.00077": "Lacki 2018 - Sunscreen",
    "2504.21151": "Lacki 2025 - Ground to Dust",
    "2606.08373": "Lacki 2026 - Dust to Dust",
    "2005.13221": "Ivanov, Beamin, Caceres, Minniti 2020 - qualitative classification of ETCs",
    "1909.08851": "Osmanov & Berezhiani 2019 - anomalous variability of Dyson megastructures",
    "2602.23270": "Amiri 2026 - Dyson spheres on HR diagram",
    "2607.03619": "Ren+ 2026 - archival diagnostics of Hephaistos contaminants",
    "2605.21093": "Vidal+ 2026 - Search for Technosignatures review",
    # --- discovered in pass 3 (OpenAlex/Crossref); the core of the long tail ---
    "2403.18941": "Contardo & Hogg 2024 - data-driven mid-IR excess, 5M FGK stars (AJ)",
    "2601.07297": "Huang, Tao & Zhang 2026 - WISE/CatWISE Dysonian waste heat, nearby galaxies (AJ)",
    "2409.11447": "Blain 2024 - Did WISE detect Dyson Spheres/Structures?",
    "2405.14921": "Ren, Garrett & Siemion 2024 - background contamination (RNAAS)",
    "2501.05152": "Ren+ 2025 - high-res radio imaging of Hephaistos candidate G (MNRASL)",
    "1503.04376": "Semiz & Ogur 2015 - Dyson spheres around white dwarfs",
    "2412.02671": "Baghram 2024 - Dyson-sphere-like structures around PBHs (ApJ)",
    "2512.07924": "Baghram 2025 - microlensing signatures of Dyson-sphere-like structures",
    "2604.21886": "Curtis+ 2026 - Dyson Minds 2025 workshop, SETI around black holes",
    "1610.05293": "Silverberg+ 2016 - M dwarf debris disk candidate, Disk Detective (ApJL)",
    "2007.15735": "Schutte+ 2020 - nearby young brown dwarf disk (Disk Detective)",
    "2109.11443": "Smith 2021 - viability of a Dyson swarm",
    "1503.01509": "Lacki 2015 - SETI at Planck energy",
    "2504.21157": "Lacki 2025 - flickers, bursts and dips (g2 autocorrelation)",
}

cands = harvest()
for aid, why in MUST.items():
    cands.setdefault(aid, {"arxiv": aid, "title": "", "why": "must-have: " + why,
                           "src": "explicit"})
print(f"{len(cands)} relevant papers selected for full-text mining", flush=True)
(OUT / "relevant.json").write_text(json.dumps(list(cands.values()), indent=1))

# --- 2. Fetch full text ------------------------------------------------------
# Quantitative-cut language we care about.
GREP = re.compile(
    r"(covering fraction|filling factor|\bf\s*=\s*0\.\d|"
    r"temperature range|\bT\s*(?:dust|d|DS)?\s*[=~<>]\s*\d{2,4}\s*K|\d{2,4}\s*K\b|"
    r"W3|W4|W1|W2|12\s*(?:um|micron|\\mu m)|22\s*(?:um|micron|\\mu m)|"
    r"signal-to-noise|S/N\s*[><=]|SNR\s*[><=]|\bchi\b|significance|"
    r"we (?:select|require|impose|reject|exclude|retain|find)|"
    r"final sample|sample of [\d,]+|[\d,]+ (?:stars|sources|objects|targets)|"
    r"cut|threshold|criteri|excess (?:significance|ratio)|"
    r"candidates? (?:remain|survive|pass)|magnitude limit)", re.I)

TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
TAGS = re.compile(r"<[^>]+>")


def textify(raw: bytes) -> str:
    s = raw.decode("utf-8", errors="replace")
    s = TAG.sub(" ", s)
    s = TAGS.sub(" ", s)
    return html.unescape(s)


order = sorted(cands.values(), key=lambda c: (0 if c["why"].startswith("must") else 1,
                                              c["arxiv"]))
for c in order:
    aid = c["arxiv"]
    dest = OUT / f"{aid}.html"
    if dest.exists() and dest.stat().st_size > 20000:
        pass
    else:
        ok = get(f"https://arxiv.org/html/{aid}v1", dest)
        if not ok or dest.stat().st_size < 5000:
            time.sleep(2)
            ok = get(f"https://arxiv.org/html/{aid}", dest)
        if not ok or dest.stat().st_size < 5000:
            time.sleep(2)
            get(f"https://arxiv.org/abs/{aid}", dest)
        time.sleep(2.5)
    if not dest.exists():
        continue
    txt = textify(dest.read_bytes())
    lines = [NSP.sub(" ", ln).strip() for ln in re.split(r"(?<=[.;])\s+|\n", txt)]
    hits = [ln for ln in lines if 25 < len(ln) < 600 and GREP.search(ln)]
    seen, uniq = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    (OUT / f"extract_{aid}.txt").write_text(
        f"# {aid}  {c.get('title','')}\n# why: {c.get('why')}\n\n" + "\n".join(uniq[:400]))

(BASE / "summary2.json").write_text(json.dumps(STATUS, indent=2))
n_ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n{n_ok}/{len(STATUS)} fetches succeeded")

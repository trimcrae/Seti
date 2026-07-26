#!/usr/bin/env python3
"""Targeted fetcher: does VASCO ever cross-match vanished sources to the INFRARED?

Prior-art question S33: VASCO-vanished optical sources that HAVE a WISE/AllWISE
infrared counterpart would be "enshrouded rather than destroyed". If the VASCO
series never ran that test, the test is ours to run.

This resolves it mechanically rather than by recollection: pull the full text of
every VASCO paper (arXiv HTML, falling back to the abs page and the LaTeX
source), then grep each one for infrared-catalogue keywords and report, per
paper, which keywords appear and in what sentence. Also pulls the central
published critique (Hambly & Blair) and the 2026 replication preprints so their
authorship and peer-review status can be checked rather than assumed.

Runs on the GitHub Actions runner: the sandbox egress policy blocks arxiv.org.

Outputs under results/disaplit2/:
  ft_<id>.html / ft_<id>.abs.html   - full text / abstract page per paper
  arxiv_q_<name>.atom               - resolver searches (Hambly&Blair, 2026 preprints)
  keyword_report.json               - per-paper keyword hits with context
  keyword_report.txt                - human-readable version
  summary.json                      - fetch status for every URL
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/disaplit2")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Seti-disaplit2/1.0 (mailto:trimcrae@gmail.com)"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:44s} {len(data):8d}B", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


ARXIV = "http://export.arxiv.org/api/query?"

# --- 1. The VASCO series. Full text where arXiv has HTML, abs page always. ----
# label -> arXiv ID.  IDs verified by a prior agent against search indices; the
# resolver searches in section 3 re-derive them independently.
VASCO = {
    "2016_our_sky_now_and_then": "1606.08992",
    "2020_vasco_I_usno": "1911.05068",
    "2022_launching_citizen_science": "2009.10813",
    "2021_nine_transients_1950": "2106.11780",
    "2022_glint_in_the_eye": "2110.15217",
    "2024_triple_transient_1952": "2310.09035",
    "2025_image_profiles_rebuttal": "2507.15896",
    # critique + replications (IDs to be confirmed by section 3 searches)
    "2026_repl_nuclear_earthshadow": "2604.00056",
    "2026_repl_vanishing_sources": "2604.04810",
    "2026_storm_suppression": "2604.06234",
    "2026_fast_transients_plates": "2603.20407",
    "2026_plate_sensitivity_control": "2604.16470",
}
for label, aid in VASCO.items():
    stem = f"ft_{label}"
    # arXiv HTML (only exists for newer papers); then the abs page as fallback.
    for ver in ("v2", "v1", ""):
        if get(f"https://arxiv.org/html/{aid}{ver}", OUT / f"{stem}.html", tries=1, pause=1.5):
            break
        time.sleep(1.5)
    get(f"https://arxiv.org/abs/{aid}", OUT / f"{stem}.abs.html", tries=2)
    time.sleep(2.5)

# --- 2. Machine-readable candidate tables (does a public catalogue exist?) ----
for name, url in {
    "vizier_vasco_2020": "https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/AJ/159/8",
    "vizier_solano_2022": "https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/MNRAS/515/1380",
    "vizier_find_vasco": "https://vizier.cds.unistra.fr/viz-bin/VizieR?-words=vanishing+appearing+sources&-meta.all=1",
}.items():
    get(url, OUT / f"{name}.xml", tries=2)
    time.sleep(2.5)

# --- 3. Resolver searches: the critique, the replications, and the IR question -
QUERIES = {
    "hambly_blair": 'all:"Hambly" AND all:"Palomar" AND (all:transient OR all:"emulsion")',
    "hambly_any": 'au:"Hambly" AND abs:transient',
    "poss_emulsion_flaw": 'abs:"emulsion" AND abs:"Palomar" AND abs:transient',
    "vasco_replication": 'abs:"POSS-I" AND (abs:replication OR abs:"independent")',
    "vasco_all2": 'all:"vanishing and appearing sources"',
    "vanished_infrared_check": 'abs:vanish AND abs:star AND (abs:"WISE" OR abs:"2MASS" OR abs:"infrared counterpart")',
    "enshrouded_star": 'abs:"dust-enshrouded" AND abs:star AND (abs:disappear OR abs:vanish OR abs:"optically thick")',
    "obscuration_event_star": 'abs:"obscuration event" AND abs:star AND abs:dust',
}
for name, q in QUERIES.items():
    get(ARXIV + urllib.parse.urlencode({"search_query": q, "start": 0,
                                        "max_results": 40, "sortBy": "relevance"}),
        OUT / f"arxiv_q_{name}.atom")
    time.sleep(3.2)

# --- 4. Keyword audit: which papers mention infrared / radio / PM checks? -----
KEYWORDS = {
    "infrared": [r"\b2MASS\b", r"\bWISE\b", r"\bAllWISE\b", r"\bNEOWISE\b", r"\bIRAS\b",
                 r"\bAKARI\b", r"\bSpitzer\b", r"\bHerschel\b", r"\bUKIDSS\b", r"\bVISTA\b",
                 r"\binfrared\b", r"\bnear-infrared\b", r"\bmid-infrared\b"],
    "radio": [r"\bNVSS\b", r"\bFIRST survey\b", r"\bVLASS\b", r"\bLoTSS\b", r"\bLOFAR\b",
              r"\bradio\b"],
    "proper_motion": [r"\bproper motion\b", r"\bhigh-proper-motion\b", r"\bmas/yr\b",
                      r"\bmas yr\b"],
    "xray_uv": [r"\bROSAT\b", r"\bGALEX\b", r"\bXMM\b", r"\bChandra\b", r"\bultraviolet\b"],
    "variability": [r"\bZTF\b", r"\bASAS-SN\b", r"\bCatalina\b", r"\bCRTS\b", r"\bPTF\b"],
    "enshroud": [r"\benshroud", r"\bobscur", r"\bextinct", r"\breddened\b", r"\bdust\b"],
}
TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


def detag(raw: bytes) -> str:
    txt = html.unescape(TAGS.sub(" ", raw.decode("utf-8", "ignore")))
    return re.sub(r"\s+", " ", txt)


report: dict[str, dict] = {}
for label in VASCO:
    for suffix in (".html", ".abs.html"):
        p = OUT / f"ft_{label}{suffix}"
        if not p.exists() or p.stat().st_size < 2000:
            continue
        text = detag(p.read_bytes())
        entry = report.setdefault(label, {"files": [], "chars": 0, "hits": {}})
        entry["files"].append(p.name)
        entry["chars"] = max(entry["chars"], len(text))
        for cat, pats in KEYWORDS.items():
            for pat in pats:
                for m in re.finditer(pat, text, re.I):
                    s = max(0, m.start() - 180)
                    ctx = text[s:m.end() + 180].strip()
                    entry["hits"].setdefault(cat, [])
                    if len(entry["hits"][cat]) < 12:
                        entry["hits"][cat].append({"pattern": pat, "context": ctx})

(OUT / "keyword_report.json").write_text(json.dumps(report, indent=2))
with (OUT / "keyword_report.txt").open("w") as fh:
    for label, e in sorted(report.items()):
        fh.write(f"\n{'='*78}\n{label}  ({e['chars']} chars from {', '.join(e['files'])})\n{'='*78}\n")
        if not e["hits"]:
            fh.write("  NO KEYWORD HITS in any category\n")
        for cat in KEYWORDS:
            hits = e["hits"].get(cat, [])
            fh.write(f"\n  -- {cat}: {len(hits)} hit(s)"
                     f"{' (capped at 12)' if len(hits) == 12 else ''}\n")
            for h in hits:
                fh.write(f"     [{h['pattern']}] ...{h['context']}...\n")

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n=== disaplit2 done: {ok}/{len(STATUS)} fetches OK; "
      f"{len(report)} papers keyword-audited ===")
for label, e in sorted(report.items()):
    ir = len(e["hits"].get("infrared", []))
    print(f"  {label:38s} infrared_hits={ir}")

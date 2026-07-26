#!/usr/bin/env python3
"""Prior-art dossier for Zackrisson et al. 2018, "SETI with Gaia" (arXiv:1804.08351).

Mandatory prior-art engagement for the underluminosity / distance-discrepancy
technosignature channel. The sandbox egress policy 403s arxiv.org / ar5iv / IOP /
OpenAlex, so this runs on the GitHub Actions runner (CLAUDE.md acquisition
pattern) and commits verbatim full text + the citation graph back to the branch.

Outputs under results/zacklit/:
  txt_1804.08351.txt       - pdftotext -layout rendering of the published PDF
  ar5iv_1804.08351.html    - ar5iv HTML rendering (equations/sections survive)
  src_1804.08351/          - unpacked arXiv LaTeX e-print source (verbatim ground truth)
  oa_work.json             - OpenAlex record for doi:10.3847/1538-4357/aac386
  oa_citing_pNN.json       - every OpenAlex work citing it (cursor-paginated)
  s2_citations.json        - Semantic Scholar citation list (cross-check)
  ads_*.json               - NASA ADS if a token happens to be present (optional)
  summary.json             - fetch status for every URL
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request

AID = "1804.08351"
DOI = "10.3847/1538-4357/aac386"
MAILTO = "trimcrae@gmail.com"

OUT = pathlib.Path("results/zacklit")
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": f"Seti-zacklit/1.0 (mailto:{MAILTO})"}
STATUS: list[dict] = []


def get(url: str, dest: pathlib.Path, tries: int = 4, pause: float = 3.0) -> bool:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:32s} {len(data):9d}B  {url}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return False


# --------------------------------------------------------------- 1. metadata
get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": AID, "max_results": 1}),
    OUT / "arxiv_meta.atom")
time.sleep(2)

# ------------------------------------------------------- 2. PDF -> plain text
# Grab both the default (latest) version and v1, in case wording differs.
for tag in ("", "v1", "v2"):
    pdf = OUT / f"pdf_{AID}{tag}.pdf"
    if get(f"https://arxiv.org/pdf/{AID}{tag}", pdf):
        txt = OUT / f"txt_{AID}{tag or ''}.txt"
        try:
            subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)],
                           check=True, timeout=300)
            print(f"     -> {txt.name} {txt.stat().st_size}B", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"     pdftotext failed ({tag or 'latest'}): {e}", flush=True)
        # keep the v-less PDF only; text is what we need downstream
        pdf.unlink(missing_ok=True)
    time.sleep(3)

# ------------------------------------------------ 3. ar5iv / arXiv HTML render
get(f"https://ar5iv.labs.arxiv.org/html/{AID}", OUT / f"ar5iv_{AID}.html")
time.sleep(3)
get(f"https://ar5iv.org/abs/{AID}", OUT / f"ar5iv_abs_{AID}.html")
time.sleep(3)
get(f"https://arxiv.org/abs/{AID}", OUT / f"abs_{AID}.html")
time.sleep(3)

# --------------------------------------------- 4. LaTeX e-print (ground truth)
src = OUT / f"eprint_{AID}.tar.gz"
if get(f"https://arxiv.org/e-print/{AID}", src):
    dest = OUT / f"src_{AID}"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(src, "r:*") as tf:
            for m in tf.getmembers():
                # keep only text-ish sources; skip figures and anything path-escaping
                if m.isfile() and not m.name.startswith(("/", "..")) and \
                        m.name.lower().endswith((".tex", ".bbl", ".bib", ".txt", ".cls", ".sty")):
                    tf.extract(m, dest)
        print("     unpacked:", [p.name for p in dest.rglob("*") if p.is_file()], flush=True)
    except tarfile.ReadError:
        # single-file gzip'd .tex
        import gzip
        try:
            raw = gzip.decompress(src.read_bytes())
            (dest / f"{AID}.tex").write_bytes(raw)
            print(f"     unpacked single-file tex {len(raw)}B", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"     e-print unpack failed: {e}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"     e-print unpack failed: {e}", flush=True)
    src.unlink(missing_ok=True)
time.sleep(3)

# -------------------------------------------------- 5. OpenAlex work + citers
if get(f"https://api.openalex.org/works/doi:{DOI}?mailto={MAILTO}", OUT / "oa_work.json"):
    try:
        wid = json.loads((OUT / "oa_work.json").read_text())["id"].rsplit("/", 1)[-1]
    except Exception:  # noqa: BLE001
        wid = None
else:
    wid = None
print(f"OpenAlex work id = {wid}", flush=True)
time.sleep(2)

if wid:
    cursor, page = "*", 0
    while cursor and page < 20:
        q = urllib.parse.urlencode({
            "filter": f"cites:{wid}",
            "per-page": 200,
            "cursor": cursor,
            "select": "id,doi,title,publication_year,cited_by_count,type,"
                      "primary_location,authorships",
            "mailto": MAILTO,
        })
        dest = OUT / f"oa_citing_p{page:02d}.json"
        if not get(f"https://api.openalex.org/works?{q}", dest):
            break
        try:
            j = json.loads(dest.read_text())
            cursor = j.get("meta", {}).get("next_cursor")
            print(f"     page {page}: {len(j.get('results', []))} results, "
                  f"total={j.get('meta', {}).get('count')}", flush=True)
            if not j.get("results"):
                break
        except Exception as e:  # noqa: BLE001
            print(f"     citing-page parse failed: {e}", flush=True)
            break
        page += 1
        time.sleep(2)

# ------------------------------------------- 6. Semantic Scholar cross-check
s2f = ("title,year,externalIds,abstract,venue,citationCount,authors")
get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}/citations"
    f"?fields={s2f}&limit=1000", OUT / "s2_citations.json")
time.sleep(3)
get(f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{AID}"
    f"?fields=title,abstract,year,citationCount,externalIds", OUT / "s2_paper.json")
time.sleep(3)

# --------------------------------------------------- 7. optional NASA ADS
tok = os.environ.get("ADS_TOKEN") or os.environ.get("ADS_API_TOKEN")
if tok:
    for name, q in {
        "ads_citations": f'citations(doi:"{DOI}")',
        "ads_paper": f'doi:"{DOI}"',
    }.items():
        url = ("https://api.adsabs.harvard.edu/v1/search/query?" +
               urllib.parse.urlencode({"q": q, "rows": 200,
                                       "fl": "title,bibcode,year,abstract,doi,citation_count"}))
        try:
            req = urllib.request.Request(url, headers={**UA, "Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                (OUT / f"{name}.json").write_bytes(r.read())
            print(f"OK   {name}.json (ADS)", flush=True)
            STATUS.append({"url": url, "file": f"{name}.json", "ok": True})
        except Exception as e:  # noqa: BLE001
            print(f"ADS {name} failed: {e}", flush=True)
        time.sleep(2)
else:
    print("no ADS token in env; skipping ADS", flush=True)

(OUT / "summary.json").write_text(json.dumps(STATUS, indent=2))
ok = sum(1 for s in STATUS if s.get("ok"))
print(f"\n=== {ok}/{len(STATUS)} fetches OK ===", flush=True)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size}B", flush=True)

#!/usr/bin/env python3
"""Acquire INSTRUMENTAL + SYSTEMATICS documentation for an AKARI-FIS / IRAS
far-infrared x optical-NIR crossmatch design.

The sandbox has no egress (every astronomy host answers 403 at the policy
proxy); the GitHub Actions runner does. This script therefore runs on the
runner and commits its harvest back to the branch.

Targets, in priority order:
  A. AKARI FIS / IRC Bright Source Catalogue Release Notes (JAXA, PDF)
  B. VizieR machine-readable ReadMe files -- the authoritative byte-by-byte
     column definitions for II/298 (AKARI FIS BSC), II/297 (AKARI IRC PSC),
     II/125 (IRAS PSC), II/156A (IRAS FSC)
  C. IRSA IRAS Explanatory Supplement (crawled) + IRSA Gator data dictionaries
     for iraspsc / akari_fis / akari_irc
  D. arXiv: instrument + survey papers (Kawada 2007 FIS, Doi 2015 maps,
     Ishihara 2010 IRC, Yamamura 2010 BSC, Jeong 2005 / Kiss 2005 confusion,
     SFD98 dust) -- abstract via API plus ar5iv HTML full text for tables
  E. OpenAlex: what crossmatch radius do published AKARI-FIS / IRAS
     crossmatch papers actually adopt

Outputs under results/farir_docs/:
  raw/<name>.{pdf,html,txt,json}   - everything fetched, verbatim
  text/<name>.txt                  - extracted plain text (PDF -> text)
  digest_<topic>.txt               - keyword-in-context extracts
  summary.json                     - per-URL fetch status
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/farir_docs")
RAW = OUT / "raw"
TXT = OUT / "text"
for d in (OUT, RAW, TXT):
    d.mkdir(parents=True, exist_ok=True)

MAIL = "trimcrae@gmail.com"
UA = {
    "User-Agent": f"Mozilla/5.0 (compatible; Seti-farir/1.0; mailto:{MAIL})",
    "Accept": "*/*",
}
STATUS: list[dict] = []


def get(url: str, name: str, tries: int = 3, pause: float = 2.0, timeout: int = 90):
    """Fetch url -> raw/<name>. Returns bytes or None."""
    dest = RAW / name
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                ctype = r.headers.get("Content-Type", "")
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": name, "bytes": len(data),
                           "ctype": ctype, "ok": True})
            print(f"OK   {name:56s} {len(data):9d}B  {ctype}", flush=True)
            return data
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} ({i+1}/{tries}) {url}", flush=True)
            if e.code in (404, 403, 410):
                break
            time.sleep(pause * (i + 1))
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}/{tries}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": name, "ok": False})
    print(f"FAIL {name:56s} {url}", flush=True)
    return None


def html_to_text(b: bytes) -> str:
    try:
        s = b.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        s = str(b)
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</(p|tr|div|h[1-6]|li|table)>", "\n", s)
    s = re.sub(r"(?is)</t[dh]>", "\t", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    for a, c in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'"), ("&mu;", "u"), ("&micro;", "u"),
                 ("&plusmn;", "+/-"), ("&times;", "x"), ("&deg;", "deg"),
                 ("&prime;", "'"), ("&Prime;", '"')):
        s = s.replace(a, c)
    s = re.sub(r"&[a-zA-Z]+;", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def pdf_to_text(path: pathlib.Path) -> str:
    """Try pdftotext (layout preserved -> tables survive), then pypdf."""
    outp = path.with_suffix(".pdftotext.txt")
    for args in (["pdftotext", "-layout", str(path), str(outp)],
                 ["pdftotext", str(path), str(outp)]):
        try:
            subprocess.run(args, check=True, capture_output=True, timeout=180)
            if outp.exists():
                t = outp.read_text("utf-8", "replace")
                if len(t.strip()) > 200:
                    return t
        except Exception:  # noqa: BLE001
            pass
    try:
        import pypdf  # type: ignore
        rd = pypdf.PdfReader(str(path))
        return "\n".join((pg.extract_text() or "") for pg in rd.pages)
    except Exception as e:  # noqa: BLE001
        return f"[PDF EXTRACTION FAILED: {e}]"


def save_text(name: str, text: str):
    (TXT / f"{name}.txt").write_text(text, "utf-8")
    print(f"     -> text/{name}.txt  ({len(text)} chars)", flush=True)


def materialize(name: str, data: bytes | None):
    """Turn a fetched blob into text/<name>.txt."""
    if not data:
        return
    p = RAW / name
    if name.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        if not p.suffix == ".pdf":
            p2 = p.with_suffix(".pdf")
            p2.write_bytes(data)
            p = p2
        save_text(pathlib.Path(name).stem, pdf_to_text(p))
    elif name.lower().endswith((".html", ".htm")) or b"<html" in data[:4000].lower():
        save_text(pathlib.Path(name).stem, html_to_text(data))
    else:
        save_text(pathlib.Path(name).stem, data.decode("utf-8", "replace"))


# ---------------------------------------------------------------- A. AKARI RN
print("\n" + "=" * 78 + "\nA. AKARI release notes (JAXA)\n" + "=" * 78, flush=True)

AKARI_DOCS = [
    # (url, output name)
    ("https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/AKARI-FIS_BSC_V1_RN.pdf",
     "akari_fis_bsc_v1_rn.pdf"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/AKARI-IRC_PSC_V1_RN.pdf",
     "akari_irc_psc_v1_rn.pdf"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/AKARI-FIS_BSC_V2_RN.pdf",
     "akari_fis_bsc_v2_rn.pdf"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/RN/",
     "akari_rn_index.html"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/",
     "akari_psc_public_index.html"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Observation/",
     "akari_observation_index.html"),
    ("https://darts.isas.jaxa.jp/astro/akari/",
     "darts_akari_index.html"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Archive/Catalogues/PSC/",
     "akari_archive_psc.html"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Archive/Catalogues/PSC/RN/AKARI-FIS_BSC_V1_RN.pdf",
     "akari_fis_bsc_v1_rn_alt.pdf"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Archive/Catalogues/PSC/RN/AKARI-FIS_BSC_V2_RN.pdf",
     "akari_fis_bsc_v2_rn_alt.pdf"),
    ("https://www.ir.isas.jaxa.jp/AKARI/Archive/Catalogues/PSC/RN/AKARI-IRC_PSC_V1_RN.pdf",
     "akari_irc_psc_v1_rn_alt.pdf"),
]
for url, name in AKARI_DOCS:
    materialize(name, get(url, name))

# Follow any PDF/HTML links found in the AKARI index pages (depth 1).
seen = {n for _, n in AKARI_DOCS}
for idx in ("akari_rn_index.html", "akari_psc_public_index.html",
            "akari_archive_psc.html"):
    p = RAW / idx
    if not p.exists():
        continue
    base = next(u for u, n in AKARI_DOCS if n == idx)
    body = p.read_bytes().decode("utf-8", "replace")
    for href in re.findall(r'href="([^"]+)"', body):
        if not href.lower().endswith((".pdf", ".txt")):
            continue
        full = urllib.parse.urljoin(base, href)
        nm = "akari_" + re.sub(r"[^A-Za-z0-9._-]", "_", href.split("/")[-1])
        if nm in seen:
            continue
        seen.add(nm)
        materialize(nm, get(full, nm))

# ------------------------------------------------------------- B. VizieR ReadMe
print("\n" + "=" * 78 + "\nB. VizieR ReadMe (byte-by-byte column definitions)\n"
      + "=" * 78, flush=True)

VIZ_CATS = {
    "II_298_akari_fis_bsc": "II/298",
    "II_297_akari_irc_psc": "II/297",
    "II_125_iras_psc": "II/125",
    "II_156A_iras_fsc": "II/156A",
    "II_275_iras_psc_fsc": "II/275",
}
VIZ_HOSTS = ["https://cdsarc.cds.unistra.fr/ftp",
             "https://cdsarc.u-strasbg.fr/ftp",
             "https://vizier.cfa.harvard.edu/ftp"]
for name, cat in VIZ_CATS.items():
    for host in VIZ_HOSTS:
        d = get(f"{host}/{cat}/ReadMe", f"readme_{name}.txt")
        if d:
            materialize(f"readme_{name}.txt", d)
            break
    # directory listing -> tells us the data file names + sizes
    for host in VIZ_HOSTS:
        d = get(f"{host}/{cat}/", f"vizdir_{name}.html")
        if d:
            materialize(f"vizdir_{name}.html", d)
            break
    # VizieR HTML metadata page (units + descriptions as rendered)
    d = get("https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source="
            + urllib.parse.quote(cat) + "&-meta.all", f"vizmeta_{name}.html")
    materialize(f"vizmeta_{name}.html", d)

# ------------------------------------------------ C. IRSA exp.sup + Gator DDs
print("\n" + "=" * 78 + "\nC. IRAS Explanatory Supplement + IRSA data dictionaries\n"
      + "=" * 78, flush=True)

# Gator data dictionaries: exact column names/descriptions as IRSA serves them.
for cat in ("iraspsc", "irasfsc", "akari_fis", "akari_irc", "iraspscz"):
    for mode in ("html", "ascii"):
        d = get(f"https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?catalog={cat}"
                f"&mode={mode}", f"gator_dd_{cat}_{mode}.html")
        if d:
            materialize(f"gator_dd_{cat}_{mode}.html", d)
            break

EXPSUP_ROOT = "https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/"
d = get(EXPSUP_ROOT, "expsup_index.html")
materialize("expsup_index.html", d)

# Crawl the Explanatory Supplement two levels deep, staying under exp.sup/.
crawled: set[str] = set()
frontier: list[str] = []
if d:
    body = d.decode("utf-8", "replace")
    for href in re.findall(r'href="([^"]+)"', body):
        full = urllib.parse.urljoin(EXPSUP_ROOT, href)
        if full.startswith(EXPSUP_ROOT) and full not in crawled:
            frontier.append(full)

# Known-important chapters, seeded explicitly in case the index is JS-driven.
for ch in ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "ch9",
           "ch10", "ch11", "ch12", "ch13", "ch14"):
    for leaf in ("", "A.html", "B.html", "C.html", "D.html", "E.html", "F.html",
                 "G.html", "H.html", "C1.html", "C2.html", "C3.html", "C4.html",
                 "C5.html", "C6.html", "C7.html", "C8.html"):
        frontier.append(f"{EXPSUP_ROOT}{ch}/{leaf}")

DEADLINE = time.time() + 1500  # 25 min hard cap for the crawl
for depth in (0, 1):
    nxt: list[str] = []
    for url in frontier:
        if time.time() > DEADLINE:
            print("!! exp.sup crawl deadline reached", flush=True)
            frontier = []
            break
        if url in crawled:
            continue
        crawled.add(url)
        rel = url[len(EXPSUP_ROOT):].strip("/") or "root"
        nm = "expsup_" + re.sub(r"[^A-Za-z0-9._-]", "_", rel)
        if not nm.endswith((".html", ".htm")):
            nm += ".html"
        blob = get(url, nm, tries=1, pause=0.5, timeout=45)
        if not blob:
            continue
        materialize(nm, blob)
        if depth == 0:
            for href in re.findall(r'href="([^"]+)"',
                                   blob.decode("utf-8", "replace")):
                full = urllib.parse.urljoin(url, href)
                full = full.split("#")[0]
                if full.startswith(EXPSUP_ROOT) and full not in crawled:
                    nxt.append(full)
    frontier = nxt

# IRSA mission/catalogue overview pages
for url, nm in [
    ("https://irsa.ipac.caltech.edu/Missions/iras.html", "irsa_iras_mission.html"),
    ("https://irsa.ipac.caltech.edu/Missions/akari.html", "irsa_akari_mission.html"),
    ("https://irsa.ipac.caltech.edu/data/AKARI/docs/AKARI_FIS_BSC_V1_RN.pdf",
     "irsa_akari_fis_rn.pdf"),
    ("https://irsa.ipac.caltech.edu/data/AKARI/docs/", "irsa_akari_docs.html"),
]:
    materialize(nm, get(url, nm))

# ----------------------------------------------------------------- D. arXiv
print("\n" + "=" * 78 + "\nD. arXiv instrument / survey / confusion papers\n"
      + "=" * 78, flush=True)

ARXIV_API = "http://export.arxiv.org/api/query?"
PAPERS = {
    "kawada2007_fis": 'ti:"Far-Infrared Surveyor" AND ti:"AKARI"',
    "doi2015_maps": 'ti:"AKARI Far-Infrared All-Sky Survey Maps"',
    "ishihara2010_irc": 'ti:"AKARI/IRC mid-infrared all-sky survey"',
    "murakami2007_akari": 'ti:"Infrared Astronomical Mission AKARI"',
    "yamamura2010_bsc": 'all:"AKARI/FIS All-Sky Survey Bright Source Catalogue"',
    "takita2010_fis_detlim": 'ti:"AKARI" AND ti:"far-infrared" AND abs:"detection limit"',
    "jeong2005_confusion": 'ti:"Far-infrared detection limits" AND abs:"confusion"',
    "kiss2005_confusion": 'ti:"Sky confusion noise in the far-infrared"',
    "sfd1998_dust": 'ti:"Maps of Dust Infrared Emission for Use in Estimation of Reddening"',
    "pollo2010_akari_fis": 'abs:"AKARI" AND abs:"FIS" AND abs:"bright source catalogue"',
    "toth2014_akari_yso": 'ti:"AKARI" AND abs:"young stellar object" AND abs:"far-infrared"',
    "abrahamyan2015_iras": 'ti:"IRAS" AND abs:"PSC" AND abs:"FSC" AND abs:"catalogue"',
    "kawada2007_fis2": 'abs:"FIS" AND abs:"AKARI" AND abs:"far-infrared surveyor"',
    "doi2012_akari_maps": 'abs:"AKARI" AND abs:"all-sky" AND abs:"far-infrared" AND abs:"maps"',
    "akari_gaia_xmatch": 'abs:"AKARI" AND abs:"Gaia" AND abs:"cross-match"',
}
arxiv_ids: dict[str, str] = {}
for key, q in PAPERS.items():
    params = {"search_query": q, "max_results": "8",
              "sortBy": "relevance", "sortOrder": "descending"}
    blob = get(ARXIV_API + urllib.parse.urlencode(params), f"arxiv_q_{key}.xml")
    if not blob:
        continue
    xml = blob.decode("utf-8", "replace")
    entries = re.findall(r"(?s)<entry>(.*?)</entry>", xml)
    lines = []
    for e in entries:
        t = re.search(r"(?s)<title>(.*?)</title>", e)
        i = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
        s = re.search(r"(?s)<summary>(.*?)</summary>", e)
        pub = re.search(r"<published>([^<]+)</published>", e)
        jr = re.search(r"(?s)<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", e)
        if not (t and i):
            continue
        title = re.sub(r"\s+", " ", t.group(1)).strip()
        aid = i.group(1)
        lines.append(f"### {title}\narXiv: {aid}\npublished: "
                     f"{pub.group(1) if pub else '?'}\n"
                     f"journal_ref: {re.sub(chr(92)+'s+',' ',jr.group(1)).strip() if jr else '-'}\n"
                     f"{re.sub(r'  +', ' ', s.group(1)).strip() if s else ''}\n")
        arxiv_ids.setdefault(f"{key}__{aid.split('v')[0].replace('/', '_')}", aid)
    save_text(f"arxiv_q_{key}", "\n".join(lines))
    time.sleep(3.1)  # arXiv API politeness

# Explicit IDs we want regardless of search recall.
for aid in ["1503.02958", "0708.0110", "1003.0270", "astro-ph/9710327",
            "0708.1796", "0906.2761", "astro-ph/0507085", "astro-ph/0412248",
            "1109.6300", "1503.06617"]:
    arxiv_ids.setdefault("explicit__" + aid.replace("/", "_"), aid)

json.dump(arxiv_ids, (OUT / "arxiv_ids.json").open("w"), indent=1)

# Full text: ar5iv renders LaTeX (incl. tables) to HTML; fall back to PDF.
FULLTEXT_KEYS = [k for k in arxiv_ids
                 if any(t in k for t in ("kawada", "doi2015", "doi2012",
                                         "ishihara", "yamamura", "jeong",
                                         "kiss", "takita", "murakami",
                                         "abrahamyan", "explicit"))]
for key in FULLTEXT_KEYS[:28]:
    aid = arxiv_ids[key]
    base = aid.split("v")[0]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)
    got = get(f"https://ar5iv.labs.arxiv.org/html/{base}", f"ar5iv_{safe}.html",
              tries=2, timeout=120)
    if got and len(got) > 5000:
        materialize(f"ar5iv_{safe}.html", got)
        continue
    got = get(f"https://arxiv.org/pdf/{base}", f"pdf_{safe}.pdf", tries=2, timeout=120)
    materialize(f"pdf_{safe}.pdf", got)
    time.sleep(1.0)

# --------------------------------------------------------------- E. OpenAlex
print("\n" + "=" * 78 + "\nE. OpenAlex: adopted crossmatch radii in the literature\n"
      + "=" * 78, flush=True)

OA = "https://api.openalex.org/works?"
OA_QUERIES = {
    "akari_fis_counterpart": "AKARI far-infrared all-sky survey counterpart identification cross-match",
    "akari_fis_2mass": "AKARI FIS bright source catalogue 2MASS counterpart search radius",
    "iras_psc_xmatch": "IRAS point source catalog cross-identification search radius positional uncertainty",
    "iras_cirrus_flag": "IRAS point source catalog cirrus flag CIRR3 contamination",
    "akari_positional_accuracy": "AKARI far-infrared surveyor positional accuracy point source catalogue",
    "fir_confusion_limit": "far-infrared confusion limit cirrus source confusion 90 micron 160 micron",
    "akari_fis_stars": "AKARI far-infrared bright source catalogue stars debris disk cross-match Gaia",
    "iras_beam_size": "IRAS survey detector aperture beam size 12 25 60 100 micron point spread",
}
for name, phrase in OA_QUERIES.items():
    params = {"search": phrase, "per-page": "50",
              "sort": "cited_by_count:desc", "mailto": MAIL}
    blob = get(OA + urllib.parse.urlencode(params), f"oa_{name}.json")
    if not blob:
        continue
    try:
        d = json.loads(blob)
    except Exception:  # noqa: BLE001
        continue
    lines = []
    for w in d.get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        if inv:
            pos = {}
            for word, idxs in inv.items():
                for ix in idxs:
                    pos[ix] = word
            abstract = " ".join(pos[k] for k in sorted(pos))
        else:
            abstract = ""
        loc = (w.get("primary_location") or {}).get("source") or {}
        lines.append(
            f"### {w.get('title')}\n"
            f"year={w.get('publication_year')} cites={w.get('cited_by_count')} "
            f"doi={w.get('doi')} venue={loc.get('display_name')}\n"
            f"{abstract[:2600]}\n")
    save_text(f"oa_{name}", "\n".join(lines))
    time.sleep(1.2)

# ------------------------------------------------------------------ digests
print("\n" + "=" * 78 + "\nF. Keyword-in-context digests\n" + "=" * 78, flush=True)

TOPICS = {
    "beam": [r"\bFWHM\b", r"beam size", r"point spread", r"\bPSF\b",
             r"pixel scale", r"pixel size", r"arcsec", r"in-?scan",
             r"cross-?scan", r"detector aperture", r"aperture size",
             r"angular resolution", r"resolution of"],
    "position": [r"positional? (accuracy|uncertaint|error)", r"error ellipse",
                 r"POSANG", r"posErr", r"major axis", r"minor axis",
                 r"astrometric accuracy", r"position error", r"1-?sigma",
                 r"search radius", r"matching radius", r"cross-?match(ing)? radius",
                 r"within \d+ ?(arcsec|\"|arcmin)"],
    "sensitivity": [r"detection limit", r"sensitivit", r"5\s*-?\s*sigma",
                    r"5\s*σ", r"\bJy\b", r"mJy", r"flux limit",
                    r"completeness", r"limiting flux"],
    "confusion": [r"confusion", r"cirrus", r"CIRR[123]", r"MConf", r"NScan",
                  r"MJy/?\s*sr", r"galactic latitude", r"foreground",
                  r"source density", r"per beam", r"structure noise"],
    "flags": [r"\bflag\b", r"FQUAL", r"quality", r"CIRR", r"MConf", r"NScan",
              r"Ndens", r"confused", r"\bMONF\b", r"reliability"],
}
for topic, pats in TOPICS.items():
    rx = re.compile("|".join(pats), re.I)
    chunks = []
    for f in sorted(TXT.glob("*.txt")):
        body = f.read_text("utf-8", "replace")
        lines = body.splitlines()
        hits = []
        last = -99
        for i, ln in enumerate(lines):
            if rx.search(ln):
                lo = max(0, i - 2)
                if lo <= last:
                    lo = last + 1
                hi = min(len(lines), i + 3)
                if lo < hi:
                    hits.append("\n".join(lines[lo:hi]))
                    last = hi - 1
        if hits:
            chunks.append(f"\n{'='*76}\n## {f.name}  ({len(hits)} hits)\n"
                          f"{'='*76}\n" + "\n---\n".join(hits[:200]))
    (OUT / f"digest_{topic}.txt").write_text("\n".join(chunks), "utf-8")
    print(f"digest_{topic}.txt  {sum(len(c) for c in chunks)} chars", flush=True)

ok = sum(1 for s in STATUS if s.get("ok"))
json.dump({"n_urls": len(STATUS), "n_ok": ok, "status": STATUS},
          (OUT / "summary.json").open("w"), indent=1)
print(f"\nDONE: {ok}/{len(STATUS)} fetches OK", flush=True)
sys.exit(0)

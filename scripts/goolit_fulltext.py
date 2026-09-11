#!/usr/bin/env python3
"""GOO full-text pass, run on the GitHub runner.

Abstracts are not enough: the rates the contact-graph and propagation models
need (rocks ejected from a system, the interstellar dust flux, ecophagy
limits, capture cross-sections, the fractions the Dyson surveys exclude) live
in the bodies of the papers.  This script fetches the FULL TEXT of every
reference the two sweep scripts name and of a handful of web-only sources:

  arXiv papers    the e-print source (LaTeX, tar.gz or gz) from
                  https://arxiv.org/e-print/<id>, de-TeXed to plain text; if the
                  e-print is a PDF, or the source fails, https://arxiv.org/pdf/<id>
                  through pypdf.
  DOI-only papers the Unpaywall API (api.unpaywall.org/v2/<doi>) for an
                  open-access PDF; through pypdf if found.  Paywalled papers are
                  recorded as NO_OA_TEXT, never silently skipped.
  web sources     Freitas 2000 (ecophagy), the FHI 2008 survey, Bostrom 2002,
                  Hanson 1998, fetched as HTML/PDF.

For every text it writes the sentences that carry a number next to a
rate/density/probability word, and separately the sentences matching the
per-reference QUERY terms (what the model actually needs from that paper).

Committed outputs (results/goolit_fulltext/):
  fulltext/<key>.txt      plain text of arXiv e-prints and open-access PDFs
                          (arXiv and OA licences permit this; paywalled papers
                          are NOT stored, only their matching sentences)
  numbers.json            per key: source, licence class, n_chars, matching sentences
  REPORT.md               per key: status and the matching sentences
  summary.json            HTTP status of every URL
"""
from __future__ import annotations

import gzip
import importlib.util
import io
import json
import os
import pathlib
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path(os.environ.get("GOOLIT_FULLTEXT_OUT", "results/goolit_fulltext"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fulltext").mkdir(exist_ok=True)
UA = {"User-Agent": "Seti-goolit-fulltext/1.0 (mailto:trimcrae@gmail.com)"}
PAUSE = float(os.environ.get("GOOLIT_PAUSE", "3.5"))
EMAIL = "trimcrae@gmail.com"
STATUS: list[dict] = []
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    os.environ.setdefault("GOOLIT_OUT", str(OUT / "_scratch"))
    spec.loader.exec_module(m)
    return m


def _write_summary() -> None:
    (OUT / "summary.json").write_text(json.dumps(
        {"n_urls": len(STATUS), "n_ok": sum(1 for s in STATUS if s["ok"]),
         "n_failed": sum(1 for s in STATUS if not s["ok"]), "status": STATUS}, indent=2))


def fetch(url: str, timeout: int = 120) -> bytes | None:
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                code = int(getattr(r, "status", 0) or 0)
                ctype = r.headers.get("Content-Type", "")
            STATUS.append({"url": url, "ok": True, "http_status": code, "bytes": len(data), "content_type": ctype})
            print(f"  ok  HTTP {code} {len(data):>9,}B {ctype[:30]:30s} {url[:90]}")
            _write_summary()
            time.sleep(PAUSE)
            return data
        except urllib.error.HTTPError as exc:
            STATUS.append({"url": url, "ok": False, "http_status": int(exc.code), "error": repr(exc)})
            print(f"  HTTP {exc.code} {url[:90]}")
            _write_summary()
            if exc.code in (403, 404, 410):
                return None
            time.sleep(PAUSE * (i + 1))
        except Exception as exc:  # noqa: BLE001
            STATUS.append({"url": url, "ok": False, "http_status": None, "error": repr(exc)})
            print(f"  failed {exc!r} {url[:90]}")
            _write_summary()
            time.sleep(PAUSE * (i + 1))
    return None


# ----------------------------------------------------------------------------
# text extraction
# ----------------------------------------------------------------------------
def detex(tex: str) -> str:
    tex = re.sub(r"(?m)^%.*$", "", tex)
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\(cite[pt]?|citealt|ref|eqref|label|input|include|bibliography|bibliographystyle|usepackage|documentclass)\*?(\[[^\]]*\])?\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", " ", tex, flags=re.S)
    tex = re.sub(r"\\(textbf|textit|emph|section|subsection|subsubsection|caption|title|author)\*?\{", "{", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?", " ", tex)
    tex = re.sub(r"[{}~]", " ", tex)
    tex = re.sub(r"\$+", " ", tex)
    return re.sub(r"\s+", " ", tex)


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(data))
    parts = []
    for p in r.pages:
        try:
            parts.append(p.extract_text() or "")
        except Exception:  # noqa: BLE001
            parts.append("")
    t = "\n".join(parts)
    t = re.sub(r"-\n(?=[a-z])", "", t)     # de-hyphenate line breaks
    return re.sub(r"\s+", " ", t)


def html_text(data: bytes) -> str:
    t = data.decode("utf-8", "ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = re.sub(r"&nbsp;", " ", t)
    t = re.sub(r"&amp;", "&", t)
    return re.sub(r"\s+", " ", t)


def arxiv_fulltext(aid: str) -> tuple[str, str] | None:
    """Returns (text, how) or None."""
    data = fetch(f"https://arxiv.org/e-print/{aid}")
    if data:
        try:
            if data[:4] == b"%PDF":
                return pdf_text(data), "arxiv-eprint-pdf"
            if data[:2] == b"\x1f\x8b":
                try:
                    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                        texs = []
                        for m in tf.getmembers():
                            if m.isfile() and m.name.lower().endswith(".tex"):
                                texs.append(tf.extractfile(m).read().decode("utf-8", "ignore"))
                        if texs:
                            texs.sort(key=len, reverse=True)
                            return detex("\n".join(texs)), "arxiv-eprint-tex"
                except tarfile.TarError:
                    raw = gzip.decompress(data)
                    if raw[:4] == b"%PDF":
                        return pdf_text(raw), "arxiv-eprint-pdf"
                    return detex(raw.decode("utf-8", "ignore")), "arxiv-eprint-tex-single"
        except Exception as exc:  # noqa: BLE001
            print(f"  e-print parse failed: {exc!r}")
    data = fetch(f"https://arxiv.org/pdf/{aid}")
    if data and data[:4] == b"%PDF":
        return pdf_text(data), "arxiv-pdf"
    return None


def unpaywall_pdf(doi: str) -> tuple[str, str] | None:
    data = fetch(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe='/')}?email={EMAIL}", timeout=60)
    if not data:
        return None
    try:
        j = json.loads(data)
    except Exception:  # noqa: BLE001
        return None
    urls = []
    best = j.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        urls.append(best["url_for_pdf"])
    for loc in j.get("oa_locations") or []:
        if loc.get("url_for_pdf") and loc["url_for_pdf"] not in urls:
            urls.append(loc["url_for_pdf"])
    for u in urls[:3]:
        pdf = fetch(u)
        if pdf and pdf[:4] == b"%PDF":
            return pdf_text(pdf), f"unpaywall:{u}"
    return None


# ----------------------------------------------------------------------------
# what the model needs from each paper: per-key query terms (regex, case-insensitive)
# ----------------------------------------------------------------------------
NUMBER_WORDS = r"(rate|per (year|yr|Myr|Gyr|million years|century)|number density|density|cross[- ]section|probability|fraction|capture|impact|encounter|ejected|per star|AU\^?-?3|au\^\{?-3|pc\^?-?3|km ?s|timescale|within|lifetime|flux|survive|months|years|watts|\bW\b|kg)"
QUERIES: dict[str, list[str]] = {
    "Freitas2000": [r"hypsithermal", r"months", r"10\^\{?1[0-9]\}? ?W|\d+ ?TW|terawatt", r"ecophag", r"detect", r"replication time|replicat\w+ (rate|time)", r"limit"],
    "PhoenixDrexler2004": [r"runaway", r"grey goo|gray goo", r"replicat"],
    "Sandberg2008": [r"nanotech", r"median", r"\d+ ?%"],
    "Bostrom2002": [r"nanotech", r"goo"],
    "Hanson1998": [r"filter"],
    "Melosh2003": [r"ejected from the (solar|Solar) [Ss]ystem", r"fraction", r"probab", r"per (Myr|million)", r"\d+ ?%", r"capture"],
    "Worth2013": [r"ejected", r"rocks", r"fraction", r"escape", r"Solar System|solar system", r"captured"],
    "AdamsNapier2022": [r"cross section", r"capture rate", r"probab", r"per (Myr|Gyr|star)", r"rocks"],
    "Belbruno2012": [r"transferred", r"10\^", r"E1[0-9]", r"probab", r"fraction"],
    "AdamsSpergel2005": [r"cross section", r"capture", r"per cluster", r"probab"],
    "Ginsburg2018": [r"capture", r"cross section", r"number of", r"lifetime", r"probab"],
    "SirajLoeb2020grazing": [r"captured", r"10\^", r"binar", r"probab"],
    "Napier2004": [r"flux", r"zodiacal", r"per (year|yr)", r"kg", r"grains", r"radiation pressure"],
    "Grun1993": [r"flux", r"m\^?-?2", r"interstellar", r"mass"],
    "Landgraf2000": [r"flux", r"m\^?-?2", r"filter", r"micron|μm|\\mu m", r"mass"],
    "Kruger2019": [r"flux", r"m\^?-?2", r"mass", r"interstellar"],
    "Jones1996": [r"lifetime", r"yr", r"shatter|sputter", r"silicate|graphite|carbon"],
    "Slavin2015": [r"lifetime", r"yr", r"destruct"],
    "Mileikowsky2000": [r"dose", r"Gy", r"survive", r"years"],
    "Do2018": [r"number density", r"au\^?-?3|AU\^?-?3", r"per star", r"ejected", r"mass density"],
    "JewittSeligman2023": [r"number density", r"per star", r"ejected", r"10\^\{?2[0-9]"],
    "PortegiesZwart2018": [r"within 100", r"per year", r"density", r"pc\^?-?3"],
    "HandsDehnen2020": [r"capture rate", r"au\^?3", r"steady", r"10\^"],
    "Dehnen2022": [r"capture", r"per (year|yr|kyr|1000)", r"fall into", r"within 5"],
    "Napier2021capture": [r"cross section", r"capture rate", r"au\^?2", r"per (year|yr|Myr)"],
    "SirajLoeb2019trapped": [r"number density", r"trapped", r"thousands|10\^"],
    "LingamLoeb2018": [r"capture", r"per (year|yr|Myr)", r"number of"],
    "Sumi2023": [r"per star", r"FFP|free-floating"],
    "Mroz2017": [r"per (main-sequence )?star", r"upper limit"],
    "GoulinskiRibak2018": [r"capture", r"cross section", r"per cent|%|percent", r"lifetime"],
    "PeretsKouwenhoven2012": [r"capture", r"%|per cent"],
    "BailerJones2018": [r"per Myr", r"within 1 pc", r"rate", r"quadratic"],
    "GarciaSanchez2001": [r"per Myr", r"within", r"rate"],
    "Levison2010": [r"captured", r"fraction", r"%|per cent", r"Oort"],
    "Wallner2016": [r"Myr", r"supernova", r"deposit", r"pc"],
    "Knie2004": [r"Myr", r"supernova", r"pc"],
    "Koll2019": [r"flux", r"atoms", r"interstellar", r"per"],
    "Frankel2018": [r"kpc", r"sqrt|\\sqrt", r"migration"],
    "SellwoodBinney2002": [r"kpc", r"migration"],
    "Wyatt2007": [r"f_?\{?max|maximum", r"r\^\{?7/3", r"t_?age"],
    "Wyatt2008": [r"maximum", r"fractional luminosity", r"t\^?-1|1/t"],
    "KennedyWyatt2013": [r"1 in|one in", r"10\^?4|10,000", r"occurrence", r"old|Gyr", r"young|Myr"],
    "KennedyWyatt2012": [r"1:1000|1 in 1000|fewer than", r"factor of five|factor of 5", r"excess"],
    "Trilling2008": [r"%", r"incidence", r"24|70 ?\\?mu"],
    "Griffith2015": [r"\\gamma|γ|gamma", r"0\.85|85", r"0\.5|50", r"0\.25|25", r"galaxies", r"none|no galax"],
    "Wright2014b": [r"\\gamma|γ|gamma", r"0\.85|85", r"galaxies"],
    "Zackrisson2015": [r"%|per cent", r"Kardashev|Type III", r"excluded|upper limit"],
    "Garrett2015": [r"%|per cent", r"Kardashev|Type III", r"upper limit"],
    "Carrigan2009": [r"candidates", r"sources", r"IRAS", r"upper limit|fraction"],
    "Suazo2022": [r"upper limit", r"covering|\\gamma|γ", r"stars", r"fraction"],
    "Suazo2024": [r"candidates", r"stars", r"million|10\^6", r"covering|\\gamma|γ", r"temperature"],
    "Ren2024": [r"candidates", r"background|galax", r"contaminat"],
    "Zackrisson2026": [r"candidates", r"galax", r"MIRI|JWST", r"redshift|z ?="],
    "Lacki2025": [r"collisional time", r"covering fraction", r"orbital period", r"P_?\{?orb", r"cascade"],
    "Lacki2016": [r"galaxies", r"%|per cent|fraction", r"Type III"],
    "ArmstrongSandberg2013": [r"galaxies", r"10\^\{?9|billion", r"99%|0\.99|80%|50%", r"hours"],
    "Olson2015": [r"expansion speed|v ?=|0\.[0-9]+ ?c", r"appearance rate", r"Gpc"],
    "Hanson2021": [r"expansion speed|0\.[0-9]+ ?c", r"grabby", r"per"],
    "CarrollNellenback2019": [r"front|settlement", r"speed", r"fraction", r"steady"],
    "Sandberg2018": [r"%|per cent", r"alone", r"Milky Way", r"observable universe"],
    "SnyderBeattie2019": [r"1 in|one in", r"per year", r"14,?000|87,?000|870,?000"],
    "Cirkovic2010": [r"underestimat", r"shadow", r"probab"],
    "BarOn2018": [r"Gt C|gigaton", r"550", r"biomass"],
    "Kreidberg2019": [r"bare rock", r"albedo", r"K\b"],
    "Greene2023": [r"K\b", r"bare rock", r"albedo"],
    "Zieba2023": [r"K\b", r"bare rock", r"albedo"],
    "Moore2003": [r"8\.5", r"75,?000", r"doubl", r"minutes"],
    "Adamala2024": [r"mirror", r"risk", r"replicat"],
    "Pan2024": [r"%|per cent", r"replicat", r"trials|success"],
    "Stevens2016": [r"grey goo|gray goo", r"nanotech", r"signature", r"replicat"],
    "SaganNewman1983": [r"self-reproducing|von Neumann", r"danger", r"Tipler"],
    "Tipler1980": [r"10\^|million years|Myr", r"von Neumann", r"galaxy"],
    "Kowald2015": [r"error", r"generations|copies", r"catastroph"],
    "Chen2022": [r"Lotka|predator|prey", r"equilibrium|steady"],
    "Osmanov2019": [r"replication", r"HII", r"years", r"speed"],
    "Ellery2022a": [r"replicat", r"imminent", r"lunar|Moon"],
    "Ellery2022b": [r"curb", r"replicat", r"genetic|limiter"],
    "Ellery2025": [r"technosignature", r"replicat", r"asteroid|lunar"],
}
DEFAULT_QUERY = [NUMBER_WORDS]

WEB_SOURCES: dict[str, list[str]] = {
    "Freitas2000": ["https://www.rfreitas.com/Nano/Ecophagy.htm", "https://foresight.org/nano/Ecophagy.html", "https://foresight.org/nano/Ecophagy.php"],
    "Sandberg2008": ["https://www.fhi.ox.ac.uk/reports/2008-1.pdf", "https://www.global-catastrophic-risks.com/docs/2008-1.pdf"],
    "Bostrom2002": ["https://jetpress.org/volume9/risks.html", "https://nickbostrom.com/existential/risks.html", "https://nickbostrom.com/existential/risks"],
    "Hanson1998": ["http://mason.gmu.edu/~rhanson/greatfilter.html", "https://mason.gmu.edu/~rhanson/greatfilter.html"],
    "Napier2004": ["https://academic.oup.com/mnras/article-pdf/348/1/46/3391004/348-1-46.pdf"],
}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text) if len(s.strip()) > 20]


def _norm(s: str) -> str:
    s = re.sub(r"\s*\^\s*", "^", s)
    s = re.sub(r"\^\s*-\s*", "^-", s)
    s = re.sub(r"\^\s*\{\s*", "^{", s)
    return s


def matching(text: str, terms: list[str]) -> list[str]:
    out = []
    for s in sentences(text):
        s = _norm(s)
        if not re.search(r"\d", s):
            continue
        if any(re.search(t, s, re.I) for t in terms) and len(s) < 900:
            out.append(s)
    return out[:80]


def main() -> None:
    g1 = _load("goolit_fetch")
    g2 = _load("goolit_contact_fetch")
    refs: dict[str, dict] = {}
    refs.update(g1.REFS)
    refs.update(g2.REFS)
    # identifiers from the earlier verification records
    ids: dict[str, dict] = {}
    for vf in (ROOT / "results/goolit/verification.json", ROOT / "results/goolit_contact/verification.json"):
        if vf.exists():
            for r in json.loads(vf.read_text())["refs"]:
                ids.setdefault(r["key"], {}).update({k: r[k] for k in ("arxiv_id", "doi") if r.get(k)})
    results: dict[str, dict] = {}
    for key, spec in refs.items():
        print(f"== {key}")
        rec = {"key": key, "status": "NO_TEXT", "how": None, "n_chars": 0, "stored": False,
               "query_terms": QUERIES.get(key, DEFAULT_QUERY), "sentences": [], "number_sentences": []}
        text, how = None, None
        aid = spec.get("id") or (ids.get(key, {}).get("arxiv_id") or "")
        aid = aid.rsplit("/abs/", 1)[-1] if "/abs/" in aid else aid
        aid = re.sub(r"v\d+$", "", aid)
        doi = ids.get(key, {}).get("doi")
        if aid:
            got = arxiv_fulltext(aid)
            if got:
                text, how = got
        if text is None and key in WEB_SOURCES:
            for u in WEB_SOURCES[key]:
                data = fetch(u)
                if data:
                    text = pdf_text(data) if data[:4] == b"%PDF" else html_text(data)
                    how = f"web:{u}"
                    if len(text) > 2000:
                        break
                    text = None
        if text is None and doi:
            got = unpaywall_pdf(doi)
            if got:
                text, how = got
        if text is None:
            rec["status"] = "NO_OA_TEXT" if (doi or aid) else "NO_IDENTIFIER"
        else:
            rec.update({"status": "TEXT", "how": how, "n_chars": len(text)})
            rec["sentences"] = matching(text, QUERIES.get(key, DEFAULT_QUERY))
            rec["number_sentences"] = matching(text, [NUMBER_WORDS])[:40]
            store = how.startswith("arxiv") or how.startswith("unpaywall") or how.startswith("web")
            if store:
                (OUT / "fulltext" / f"{key}.txt").write_text(text)
                rec["stored"] = True
        results[key] = rec
        (OUT / "numbers.json").write_text(json.dumps(results, indent=1))
    L = ["# GOO full-text pass", "",
         f"URLs fetched: {sum(1 for s in STATUS if s['ok'])} ok / {len(STATUS)} total.  "
         f"Texts obtained: {sum(1 for r in results.values() if r['status'] == 'TEXT')} of {len(results)}.", "",
         "| key | status | how | chars |", "|---|---|---|---|"]
    for k, r in results.items():
        L.append(f"| {k} | {r['status']} | {(r['how'] or '')[:60]} | {r['n_chars']:,} |")
    L += ["", "## Sentences matching each paper's query terms (verbatim from the full text)", ""]
    for k, r in results.items():
        if r["status"] != "TEXT":
            continue
        L.append(f"### {k}  ({r['how']}; {r['n_chars']:,} chars)")
        for s in r["sentences"]:
            L.append(f"- {s}")
        L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(json.dumps({s: sum(1 for r in results.values() if r["status"] == s) for s in ("TEXT", "NO_OA_TEXT", "NO_IDENTIFIER", "NO_TEXT")}))


if __name__ == "__main__":
    main()

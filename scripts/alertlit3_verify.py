#!/usr/bin/env python3
"""Third-pass verification: resolve arXiv ids found via web search to REAL titles.

Everything cited in the prior-art report must have its title confirmed against
the arXiv API (a guessed id resolves to an unrelated paper often enough that
un-verified citation is unacceptable). Also resolves a handful of paper titles
that are journal-only, and pulls full text for the three papers whose
technosignature framing (or absence of it) decides a verdict:
  * Anomaly Hunter for Alerts (AHA) -- does the ZTF alert-stream anomaly
    pipeline mention technosignatures at all?
  * the executed monochromatic-light imaging survey along the galactic plane
  * SNAD's Rubin-field anomaly hunt

Runs on the GitHub Actions runner; the dev sandbox blocks arxiv.org.

Outputs under results/alertlit3/:
  ids_verified.json / verify.txt   id -> real title/date/authors
  oa_*.json                        OpenAlex title resolutions
  txt_*.txt                        full text for the decisive papers
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/alertlit3")
OUT.mkdir(parents=True, exist_ok=True)
MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-alertlit3/1.0 (mailto:{MAIL})"}
LINES: list[str] = []


def get(url: str, dest: pathlib.Path, tries: int = 3) -> bytes | None:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"OK   {dest.name:48s} {len(data):9d}B", flush=True)
            return data
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(3.0 * (i + 1))
    print(f"FAIL {url}", flush=True)
    return None


ARXIV = "http://export.arxiv.org/api/query?"


def entries(blob: bytes) -> list[dict]:
    txt = blob.decode("utf-8", "replace")
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", txt, re.S):
        def f(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
            return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip()) if m else ""
        auth = [html.unescape(re.sub(r"\s+", " ", a).strip())
                for a in re.findall(r"<author>\s*<name>(.*?)</name>", e, re.S)]
        out.append({"id": f("id").replace("http://arxiv.org/abs/", ""),
                    "title": f("title"), "published": f("published")[:10],
                    "authors": auth[:6], "summary": f("summary")})
    return out


# ids harvested from web search that MUST be verified before citation
IDS = [
    "2602.12955",  # suspected Anomaly Hunter for Alerts (AHA)
    "2404.01235",  # suspected anomaly detection + similarity search, real-time streams
    "2603.20407",  # suspected fast transients in archival plates
    "2606.08319",  # suspected plate optical-aberration follow-up
    "2401.08763",  # suspected weird+wonderful solar system / serendipity in LSST
    "2203.08968",  # suspected technosignature opportunities in Astro2020
    "2402.11037",  # suspected SN 1987A SETI Ellipsoid with TESS
    "2405.04560",  # suspected detectability of solar panels
    "2411.03258",  # suspected ZTF satellite/debris in alert stream
    "2310.17322",  # suspected rate of satellite glints ZTF/LSST
    "2202.05719",  # suspected impact of satellite glints ZTF
    "2011.03497",  # suspected sub-second flares from geosynchronous satellites
    "2011.02495",  # suspected orbital foregrounds for ultra-short transients
    "2508.03964",  # suspected systematic dipper search ZTF
    "2406.17259",  # suspected Yuti transit simulator
    "2605.17098",  # suspected Loeb-Turner alpha slope TNOs
    "2606.08373",  # suspected Dust to Dust passive technosignatures
    "2607.07781",  # suspected stellar J-harvesting
    "2403.04942",  # suspected expected impact of glints from space debris in LSST
    "2507.22156",  # suspected early transient discovery LSST via DECam DIA
    "2109.09637",  # suspected TAIGA-HiSCORE nanosecond optical transients
    "2012.02316",  # suspected drift-scan high-cadence transient limits
    "2506.13459",  # suspected BL 27 eclipsing exoplanets
    "1907.07829",  # suspected technosignatures in the thermal infrared
    "2606.13797",  # Solar System Technosignatures (re-verify)
    "2103.01536",  # concepts for future missions (re-verify)
    "2110.13887",  # suspected Dyson sphere feedback
    "1306.1672",   # suspected class A stellar engines transit curves
    "astro-ph/0506758",  # suspected optical SETI with Cherenkov telescopes
    "2604.06234",  # suspected storm-driven plate transient detections at GEO
    "2204.06091",  # suspected high-albedo objects in geosynchronous orbits
]

# journal-only / uncertain-id papers: resolve titles through OpenAlex instead
OA_TITLES = [
    ("exotica", "One of Everything: The Breakthrough Listen Exotica Catalog"),
    ("mono_swath", "A search for transient, monochromatic light in a 6-deg swath "
                   "along the galactic plane"),
    ("aligned_poss", "Aligned, Multiple-transient Events in the First Palomar Sky Survey"),
    ("multiscale_astrobio", "Multiscale astrobiology with the Vera C. Rubin Observatory "
                            "Legacy Survey of Space and Time"),
    ("evryscope_efte", "The Evryscope Fast Transient Engine: Real-time Detection for "
                       "Rapidly Evolving Transients"),
    ("triple_transient", "A bright triple transient that vanished within 50 minutes"),
    ("tomoe_glints", "Second-timescale Glints from Satellites and Space Debris Detected "
                     "with Tomo-e Gozen"),
    ("weird_detector", "The weird detector: flagging periodic, coherent signals of "
                       "arbitrary shape in time-series photometry"),
    ("fink_anomaly", "Anomaly detection in the Fink broker"),
    ("anomaly_lsst_timeseries", "Anomaly detection to identify transients in LSST "
                                "time series data"),
]

FULLTEXT = {"aha_ztf_alerts": "2602.12955",
            "snad_dr23": "2507.06217",
            "weird_wonderful_lsst": "2401.08763",
            "astro2020_technosig": "2203.08968"}


def strip_html(b: bytes) -> str:
    t = b.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", t))
    return re.sub(r"\n\s*\n\s*\n+", "\n\n",
                  re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", t))).strip()


def main() -> None:
    verified: dict[str, dict] = {}
    for k in range(0, len(IDS), 15):
        chunk = IDS[k:k + 15]
        blob = get(ARXIV + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}),
            OUT / f"ax_verify_{k//15}.atom")
        time.sleep(3.2)
        if blob:
            for x in entries(blob):
                verified[re.sub(r"v\d+$", "", x["id"])] = x
    LINES.append("### arXiv id -> REAL title")
    for i in IDS:
        v = verified.get(i)
        LINES.append(f"  {i:20s} {v['published'] if v else '':10s} "
                     f"{v['title'] if v else '*** NOT FOUND ***'}")
        if v:
            LINES.append(f"  {'':31s} {', '.join(v['authors'])}")

    LINES.append("\n### OpenAlex title resolution (journal-only / uncertain ids)")
    for slug, title in OA_TITLES:
        blob = get("https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"search": title, "per-page": "4", "mailto": MAIL}), OUT / f"oa_{slug}.json",
            tries=2)
        time.sleep(1.3)
        if not blob:
            LINES.append(f"  {slug:22s} FETCH FAILED")
            continue
        try:
            res = json.loads(blob).get("results") or []
        except Exception:  # noqa: BLE001
            res = []
        LINES.append(f"  {slug}: query={title!r}")
        for w in res:
            loc = ((w.get("primary_location") or {}).get("source") or {}) or {}
            ids = w.get("ids") or {}
            LINES.append(f"     {w.get('publication_year')}  {(w.get('title') or '')[:120]}"
                         f"  [{(loc.get('display_name') or '')[:38]}]  doi={w.get('doi')}")

    for slug, aid in FULLTEXT.items():
        for url in (f"https://arxiv.org/html/{aid}",
                    f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            b = get(url, OUT / f"html_{slug}.html", tries=2)
            if b and len(b) > 25000:
                (OUT / f"txt_{slug}.txt").write_text(strip_html(b))
                break
        time.sleep(1.0)

    (OUT / "ids_verified.json").write_text(json.dumps(verified, indent=1))
    (OUT / "verify.txt").write_text("\n".join(LINES))
    print("done", flush=True)


if __name__ == "__main__":
    main()

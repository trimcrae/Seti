#!/usr/bin/env python3
"""Second-pass prior-art adjudication for alert-stream technosignature angles.

Pass 1 (scripts/alertlit_priorart.py) ran arXiv+OpenAlex query batteries. This
pass does the two things that actually settle DONE vs PROPOSED vs UNOCCUPIED:

  (1) FULL TEXT of every candidate-relevant review / proposal, so the searches
      they claim as *planned* can be enumerated verbatim instead of guessed.
  (2) CITATION GRAPH: for each proposal paper (e.g. Lacki 2019's specular-glint
      method, Davenport 2019's spatio-temporal SETI, Gray/Greenstreet 2025's
      alert-broker paper), pull every work that CITES it from OpenAlex. If a
      proposal has never been executed, the citing set contains no paper that
      applies it to data -- that is the emptiness evidence.

Runs on the GitHub Actions runner; the dev sandbox blocks arxiv.org and
api.openalex.org.

Outputs under results/alertlit2/:
  ax_verify_*.atom      arXiv Atom for id verification (real titles)
  ids_verified.json     id -> {title, published, summary}
  html_<slug>.html      arxiv/ar5iv full-text HTML
  txt_<slug>.txt        text-extracted full text
  cites_<slug>.json     OpenAlex works citing the proposal
  cites_digest.txt      flattened citing lists
  summary.json          fetch status for every URL
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path("results/alertlit2")
OUT.mkdir(parents=True, exist_ok=True)

MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-alertlit2/1.0 (mailto:{MAIL})"}
STATUS: list[dict] = []
DIGEST: list[str] = []


def get(url: str, dest: pathlib.Path, tries: int = 3, pause: float = 3.0) -> bytes | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            STATUS.append({"url": url, "file": dest.name, "bytes": len(data), "ok": True})
            print(f"OK   {dest.name:56s} {len(data):9d}B", flush=True)
            return data
        except Exception as e:  # noqa: BLE001
            print(f"RETRY({i+1}) {url} -> {e}", flush=True)
            time.sleep(pause * (i + 1))
    STATUS.append({"url": url, "file": dest.name, "ok": False})
    print(f"FAIL {url}", flush=True)
    return None


def strip_html(b: bytes) -> str:
    t = b.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", re.sub(r"[ \t]*\n[ \t]*", "\n", t)).strip()


# --------------------------------------------------------------------------
# every arXiv id we might cite, keyed by slug. Titles are VERIFIED, never assumed.
# --------------------------------------------------------------------------
PAPERS: dict[str, str] = {
    # --- A7 reviews / the community's planned-search lists ---
    "alert_brokers": "2506.14744",
    "rubin_astrobio_technosig": "2606.00574",
    "review_possibilities": "2605.21093",
    "framing_possibility_space": "2603.17741",
    "technosig_solar_system": "2606.13797",
    "iso_technosig": "2508.16825",
    "rubin_tvs_roadmap": "2208.04499",
    "seti_2020": "2107.07512",
    "seti_2022": "2410.08253",
    "exotica": "2010.15577",
    "nasa_technosig_report": "1812.08681",
    "future_missions_technosig": "2103.01536",
    # --- A1 glint / specular ---
    "lacki_shiny_glint": "1903.05839",
    "jaiswal_specular": "2306.07859",
    "villarroel_glint_plates": "2110.15217",
    # --- A2 recurrence / spatio-temporal ---
    "davenport_spatiotemporal": "1907.04443",
    "seti_ellipsoid_gaia": "2206.04092",
    "seti_ellipsoid_2306": "2306.03118",
    # --- A3 / A6 transit + megastructure ---
    "g_search_iv_megastructures": "1510.04606",
    "technosignatures_in_transit": "1907.07830",
    "cloaking_device": "1603.08928",
    "bl_anomalous_transits_kepler": "2312.07903",
    # --- A4 optical pulse in surveys ---
    "ps1_fast_transients": "1307.5324",
    "seti_small_telescopes": "2109.11005",
    "short_pulse_limits_oseti": "1804.01251",
    "ztf_dirty_fireballs": "2201.12366",
    # --- A5 ISO / non-grav accel ---
    "loeb_scale": "2508.09167",
    "iso_review": "2304.00568",
    "ata_3iatlas": "2512.18142",
    "fast_3iatlas_narrowband": "2603.19023",
    "fast_3iatlas_periodic": "2607.01666",
    # --- A8 anomaly detection ---
    "snad_ztf_dr3": "2012.01419",
    "snad_review": "2410.18875",
    "snad_dr23_rubin": "2507.06217",
    "alertissimo": "2601.10454",
    "blind_spot_accel": "2607.07413",
    "blink_miss_it": "2509.23632",
}

ARXIV = "http://export.arxiv.org/api/query?"


def _atom_entries(blob: bytes) -> list[dict]:
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
                    "authors": auth[:8], "summary": f("summary")})
    return out


def verify_ids() -> dict[str, dict]:
    ids = list(dict.fromkeys(PAPERS.values()))
    verified: dict[str, dict] = {}
    for k in range(0, len(ids), 20):
        chunk = ids[k:k + 20]
        url = ARXIV + urllib.parse.urlencode({"id_list": ",".join(chunk),
                                              "max_results": len(chunk)})
        blob = get(url, OUT / f"ax_verify_{k//20}.atom")
        time.sleep(3.2)
        if blob:
            for x in _atom_entries(blob):
                verified[x["id"].split("v")[0]] = x
    (OUT / "ids_verified.json").write_text(json.dumps(verified, indent=1))
    DIGEST.append("### arXiv id verification (slug / id / REAL title / date / authors)")
    for slug, aid in PAPERS.items():
        v = verified.get(aid)
        if v:
            DIGEST.append(f"  {slug:32s} {aid:12s} {v['published']}  {v['title']}")
            DIGEST.append(f"  {'':32s} {'':12s} authors: {', '.join(v['authors'])}")
        else:
            DIGEST.append(f"  {slug:32s} {aid:12s} *** NOT FOUND ON ARXIV ***")
    return verified


# --------------------------------------------------------------------------
# OpenAlex citation graph
# --------------------------------------------------------------------------
OA = "https://api.openalex.org/works"

# Proposals whose citing sets we mine for "did anyone actually execute this?"
CITED_TARGETS = ["lacki_shiny_glint", "jaiswal_specular", "villarroel_glint_plates",
                 "davenport_spatiotemporal", "alert_brokers", "g_search_iv_megastructures",
                 "cloaking_device", "technosignatures_in_transit", "snad_ztf_dr3",
                 "seti_ellipsoid_gaia", "bl_anomalous_transits_kepler"]


def oa_by_arxiv(aid: str, slug: str) -> str | None:
    """Resolve an arXiv id to an OpenAlex work id via a title search fallback."""
    url = (OA + "?" + urllib.parse.urlencode(
        {"filter": f"locations.landing_page_url.search:arxiv.org/abs/{aid}",
         "mailto": MAIL}))
    blob = get(url, OUT / f"oaid_{slug}.json", tries=2)
    time.sleep(1.2)
    if blob:
        try:
            res = json.loads(blob).get("results") or []
            if res:
                return res[0]["id"].rsplit("/", 1)[-1]
        except Exception:  # noqa: BLE001
            pass
    return None


def oa_by_title(title: str, slug: str) -> str | None:
    url = OA + "?" + urllib.parse.urlencode(
        {"search": title, "per-page": "5", "mailto": MAIL})
    blob = get(url, OUT / f"oatitle_{slug}.json", tries=2)
    time.sleep(1.2)
    if not blob:
        return None
    try:
        res = json.loads(blob).get("results") or []
    except Exception:  # noqa: BLE001
        return None
    tl = title.lower()[:50]
    for w in res:
        if (w.get("title") or "").lower()[:50] == tl:
            return w["id"].rsplit("/", 1)[-1]
    return res[0]["id"].rsplit("/", 1)[-1] if res else None


def cited_by(oa_id: str, slug: str) -> list[dict]:
    got: list[dict] = []
    for page in (1, 2, 3):
        url = OA + "?" + urllib.parse.urlencode(
            {"filter": f"cites:{oa_id}", "per-page": "200", "page": str(page),
             "sort": "publication_date:desc", "mailto": MAIL})
        blob = get(url, OUT / f"cites_{slug}_p{page}.json", tries=2)
        time.sleep(1.3)
        if not blob:
            break
        try:
            res = json.loads(blob).get("results") or []
        except Exception:  # noqa: BLE001
            break
        got.extend(res)
        if len(res) < 200:
            break
    recs = [{"year": w.get("publication_year"), "title": w.get("title"),
             "venue": (((w.get("primary_location") or {}).get("source") or {}) or {}
                       ).get("display_name"),
             "doi": w.get("doi"), "cites": w.get("cited_by_count")} for w in got]
    (OUT / f"cites_{slug}.json").write_text(json.dumps(recs, indent=1))
    DIGEST.append(f"\n### CITED-BY  {slug}  (OpenAlex {oa_id})  n={len(recs)}")
    for r in recs:
        DIGEST.append(f"    {r['year']}  {(r['title'] or '')[:135]}  [{(r['venue'] or '')[:40]}]")
    return recs


def main() -> None:
    print("=== verify arXiv ids ===", flush=True)
    verified = verify_ids()

    print("=== full text ===", flush=True)
    for slug, aid in PAPERS.items():
        ok = False
        for url in (f"https://arxiv.org/html/{aid}",
                    f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            b = get(url, OUT / f"html_{slug}.html", tries=2)
            if b and len(b) > 25000:
                (OUT / f"txt_{slug}.txt").write_text(strip_html(b))
                ok = True
                break
        if not ok:
            b = get(f"https://arxiv.org/abs/{aid}", OUT / f"abs_{slug}.html", tries=2)
            if b:
                (OUT / f"txt_{slug}.txt").write_text(strip_html(b))
        time.sleep(1.0)

    print("=== citation graph ===", flush=True)
    for slug in CITED_TARGETS:
        aid = PAPERS[slug]
        oa_id = oa_by_arxiv(aid, slug)
        if not oa_id:
            t = (verified.get(aid) or {}).get("title")
            if t:
                oa_id = oa_by_title(t, slug)
        if oa_id:
            cited_by(oa_id, slug)
        else:
            DIGEST.append(f"\n### CITED-BY  {slug}  *** could not resolve OpenAlex id ***")

    (OUT / "cites_digest.txt").write_text("\n".join(DIGEST))
    (OUT / "summary.json").write_text(json.dumps(STATUS, indent=1))
    ok = sum(1 for s in STATUS if s.get("ok"))
    print(f"\ndone: {ok}/{len(STATUS)} fetches ok", flush=True)


if __name__ == "__main__":
    main()

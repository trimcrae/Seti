"""OCCULT probe: what microlensing photometry is actually reachable from a runner.

Runner-only (the sandbox has no archive egress).  It fetches a handful of
index pages and per-event files from the three public microlensing surveys
(OGLE-IV EWS, KMTNet alerts, MOA alerts), and records for every URL: HTTP
status, size, elapsed time, content type, the first lines verbatim and the
links the page carries.  Nothing is inferred beyond that: the probe's job is
to let the acquisition code be written against the formats that exist, not
the formats we remember.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

OGLE_ROOT = "https://ogle.astrouw.edu.pl/ogle4/ews/"
KMT_ROOTS = ("https://kmtnet.kasi.re.kr/~ulens/event/", "https://kmtnet.kasi.re.kr/ulens/event/")
MOA_ROOT = "https://www.massey.ac.nz/~iabond/moa/"

_HREF = re.compile(r"""href\s*=\s*["']?([^"' >]+)""", re.IGNORECASE)


def extract_links(text: str, base: str, limit: int = 120) -> list[str]:
    """Unique absolute hrefs in page order (pure)."""
    out, seen = [], set()
    for m in _HREF.finditer(text or ""):
        u = urljoin(base, m.group(1))
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= limit:
            break
    return out


def fetch(session, url: str, timeout: float = 60.0, head_chars: int = 1500) -> dict:
    t = time.time()
    rec = {"url": url}
    try:
        r = session.get(url, timeout=timeout)
        body = r.content
        rec.update(status=r.status_code, bytes=len(body), final_url=r.url,
                   content_type=r.headers.get("content-type", ""))
        try:
            text = body.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = ""
        rec["head"] = text[:head_chars]
        rec["n_lines"] = text.count("\n")
        rec["_text"] = text
    except Exception as exc:  # noqa: BLE001
        rec.update(status=None, error=repr(exc)[:400])
        rec["_text"] = ""
    rec["elapsed_s"] = round(time.time() - t, 2)
    return rec


def _strip(rec: dict, links: bool = True) -> dict:
    text = rec.pop("_text", "")
    if links and rec.get("content_type", "").startswith("text/html"):
        rec["links"] = extract_links(text, rec.get("final_url") or rec["url"])
    return rec


def run_probe(out_dir: Path) -> dict:
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "seti-occult-probe/1.0 (research; github.com/trimcrae/Seti)"
    recs: dict[str, list] = {"ogle": [], "kmt": [], "moa": []}

    # ---- OGLE-IV EWS -------------------------------------------------------
    recs["ogle"].append(_strip(fetch(s, OGLE_ROOT)))
    recs["ogle"].append(_strip(fetch(s, OGLE_ROOT + "ews.html")))
    for yr in range(2011, 2027):
        base = f"{OGLE_ROOT}{yr}/"
        idx = _strip(fetch(s, base))
        idx["links"] = idx.get("links", [])[:40]
        recs["ogle"].append(idx)
        for name in ("lenses.par", "blg-0001/phot.dat", "blg-0001/params.dat"):
            r = fetch(s, base + name)
            r.pop("_text", None)
            recs["ogle"].append(r)
    for name in ("2019/blg-0001.tar.gz", "2019/phot.tar.gz", "2019/blg-0001/",
                 "2019/blg-0001.html"):
        recs["ogle"].append(_strip(fetch(s, OGLE_ROOT + name)))

    # ---- KMTNet ------------------------------------------------------------
    for root in KMT_ROOTS:
        recs["kmt"].append(_strip(fetch(s, root)))
    for yr in (2016, 2019, 2022, 2025, 2026):
        root = KMT_ROOTS[0]
        idx = _strip(fetch(s, f"{root}{yr}/"))
        links = idx.get("links", [])
        idx["links"] = links[:60]
        idx["n_links"] = len(links)
        recs["kmt"].append(idx)
        yy = str(yr)[2:]
        ev = f"KB{yy}0001"
        for u in (f"{root}{yr}/view.php?event=KMT-{yr}-BLG-0001",
                  f"{root}{yr}/data/{ev}/",
                  f"{root}{yr}/data/{ev}/pysis/"):
            rr = _strip(fetch(s, u))
            rr["links"] = rr.get("links", [])[:80]
            recs["kmt"].append(rr)
        # try the pysis files the listing (or the known naming) offers
        cand = [lk for rr in recs["kmt"][-2:] for lk in rr.get("links", [])
                if lk.endswith(".pysis") or lk.endswith(".dat") or lk.endswith(".tar.gz")]
        if not cand:
            cand = [f"{root}{yr}/data/{ev}/pysis/KMT{site}{fld}_I.pysis"
                    for site in "ACS" for fld in ("01", "41", "02", "42")]
        for u in cand[:8]:
            r = fetch(s, u, head_chars=800)
            r.pop("_text", None)
            recs["kmt"].append(r)

    # ---- MOA ---------------------------------------------------------------
    for yr in (2016, 2019, 2022, 2025):
        for u in (f"{MOA_ROOT}alert{yr}/alert.php", f"{MOA_ROOT}alert{yr}/"):
            rr = _strip(fetch(s, u))
            links = rr.get("links", [])
            rr["links"] = links[:60]
            recs["moa"].append(rr)
        disp = [lk for rr in recs["moa"][-2:] for lk in rr.get("links", []) if "display" in lk]
        for u in disp[:1]:
            rr = _strip(fetch(s, u))
            rr["links"] = rr.get("links", [])[:60]
            recs["moa"].append(rr)
            for lk in [x for x in rr["links"] if "fetchtxt" in x or x.endswith(".dat")][:2]:
                r = fetch(s, lk, head_chars=800)
                r.pop("_text", None)
                recs["moa"].append(r)

    def _ok(lst, pred):
        return sum(1 for r in lst if r.get("status") == 200 and pred(r))

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ogle_phot_ok": _ok(recs["ogle"], lambda r: r["url"].endswith("phot.dat")),
        "ogle_par_ok": _ok(recs["ogle"], lambda r: r["url"].endswith("lenses.par")),
        "kmt_pysis_ok": _ok(recs["kmt"], lambda r: r["url"].endswith(".pysis")),
        "moa_ok": _ok(recs["moa"], lambda r: True),
        "records": recs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(summary, indent=1))
    return summary

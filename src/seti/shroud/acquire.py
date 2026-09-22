"""SHROUD acquisition — runner only (the sandbox has no archive egress).

Four routes to the sample, every one recording exactly what it did so a
degraded run reports its degradation as a first-class field:

1. **The Solano+2022 VO archive** (SVO ``svocats``): ``vanish-neowise`` (the
   optically-absent / IR-present sample, 171 753 rows live) and
   ``vanish-possi`` (the 5 399 no-counterpart control).  Both URLs are quoted
   verbatim in Watters et al. 2026 Table 1 and in the ``jannefi/vasco`` README.
   Run 30203741898 (2026-07-26) found the host answering **nothing at the TCP
   level** (``curl`` exit 000 after 60 s) and then burned its whole 70-minute
   budget in a 300 s-timeout retry ladder before reaching any fallback.  The
   probes here are short (``probe_timeout_s``) and only a *confirmed-live* root
   gets the long bulk-fetch timeout.

2. **VizieR ``TAP_SCHEMA`` discovery.**  ``J/MNRAS/515/1380`` was "not found"
   on 2026-07-26, but a catalogue can be ingested under any name, so the
   service is *asked* (keyword search over ``table_name`` and ``description``,
   with the double quotes VizieR wraps names in stripped — the ``tailings``
   channel lost a dispatch to that).  ``J/AJ/159/8`` (Villarroel+2020, 127
   rows) is fetched regardless: it is real, small and published.

3. **Reconstruct the selection from USNO-B1.0** (``I/284/out``).  USNO-B1.0 is
   the PMM scan of the very POSS-I plates: ``R1mag`` is the POSS-I E (red)
   detection, ``B1mag`` the O plate, ``B2/R2/I`` the second-epoch plates, and
   ``Ndet`` counts the plates the object was found on.  ``Ndet = 1`` with
   ``R1mag`` present is therefore *"seen on the POSS-I red plate and on no
   other plate"* — the Solano+2022 parent selection, owned rather than
   inherited, in a deterministic grid of high-latitude fields north of
   dec = -25 (south of that the first-epoch plates are SERC/ESO, not POSS-I).
   Modern-optical absence (Gaia DR3, Pan-STARRS DR1) and infrared presence
   (AllWISE, CatWISE, 2MASS) are then established by uploaded crossmatch.

4. **Local cache** when the network is disabled.

Bulk crossmatching goes through the **CDS X-Match** service, which takes an
uploaded table; per-object cone searches at this scale are not viable and
would themselves become the systematic.

Nothing in this module fabricates a row.  Every function returns
``(DataFrame, provenance)`` and the provenance carries the URL actually used,
the HTTP status, and the row count.
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_UA = {"User-Agent": "Seti-SHROUD/1.1 (mailto:trimcrae@gmail.com)"}

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
VIZIER_ASU = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
XMATCH_URLS = (
    "https://cdsxmatch.cds.unistra.fr/xmatch/api/v1/sync",
    "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync",
)

_URL_RE = re.compile(
    r"https?://[\w.\-]*(?:cab\.inta-csic\.es|svo\d?\.[\w.\-]+)[\w/\-.~%?=&+#]*", re.I)
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.I)


@dataclass
class Provenance:
    """What actually happened, so a degraded run can say so."""

    route: str = ""
    url: str = ""
    status: str = "not_attempted"
    n_rows: int = 0
    attempts: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record(self, url: str, ok: bool, detail: str = "", n: int = 0) -> None:
        self.attempts.append({"url": url, "ok": bool(ok), "detail": detail,
                              "n_rows": int(n)})

    def as_dict(self) -> dict:
        return {"route": self.route, "url": self.url, "status": self.status,
                "n_rows": int(self.n_rows), "attempts": self.attempts,
                "notes": self.notes}


# --- low-level HTTP ---------------------------------------------------------
def http_get(url: str, timeout: int = 300, retries: int = 4,
             backoff: float = 8.0, data: bytes | None = None,
             headers: dict | None = None) -> tuple[bytes | None, str]:
    """GET/POST with retries.  Returns ``(body, detail)``; body is None on failure."""
    hdr = dict(_UA)
    if headers:
        hdr.update(headers)
    last = ""
    for i in range(max(retries, 1)):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=data, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                return body, f"HTTP {r.status} ({len(body)} B, {time.time() - t0:.1f}s)"
        except urllib.error.HTTPError as e:                    # noqa: PERF203
            last = f"HTTP {e.code} ({time.time() - t0:.1f}s)"
            try:
                snippet = e.read(400).decode("utf-8", "replace")
                last += " " + re.sub(r"\s+", " ", snippet)[:200]
            except Exception:                                  # noqa: BLE001
                pass
        except Exception as e:                                 # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:160]} ({time.time() - t0:.1f}s)"
        if i + 1 < max(retries, 1):
            time.sleep(backoff * (i + 1))
    return None, last


def _probe(url: str, cfg: dict, data: bytes | None = None,
           headers: dict | None = None) -> tuple[bytes | None, str]:
    """A *short* request: the reachability ladder must never eat the budget."""
    a = cfg.get("acquire", {})
    return http_get(url, int(a.get("probe_timeout_s", 25)),
                    retries=int(a.get("probe_retries", 1)), backoff=3.0,
                    data=data, headers=headers)


def _votable_to_frame(body: bytes) -> pd.DataFrame:
    """Parse a VOTable without requiring astropy's XML stack to be importable."""
    try:
        from astropy.io.votable import parse_single_table

        tab = parse_single_table(io.BytesIO(body))
        return tab.to_table().to_pandas()
    except Exception:                                          # noqa: BLE001
        return _votable_regex_frame(body)


def _votable_regex_frame(body: bytes) -> pd.DataFrame:
    """Minimal TABLEDATA parser: enough for the small VizieR candidate tables."""
    txt = body.decode("utf-8", "replace")
    fields = re.findall(r'<FIELD[^>]*\bname="([^"]+)"', txt)
    rows = re.findall(r"<TR>(.*?)</TR>", txt, re.S)
    recs = []
    for r in rows:
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<TD>(.*?)</TD>", r, re.S)]
        if len(cells) == len(fields):
            recs.append(dict(zip(fields, cells, strict=False)))
    return pd.DataFrame(recs)


def votable_tables(body: bytes) -> dict[str, pd.DataFrame]:
    """Split a multi-TABLE VOTable into ``{table_name: frame}``."""
    txt = body.decode("utf-8", "replace")
    out: dict[str, pd.DataFrame] = {}
    for chunk in re.split(r"(?=<TABLE )", txt):
        if not chunk.startswith("<TABLE"):
            continue
        m = re.search(r'name="([^"]+)"', chunk)
        name = m.group(1) if m else f"table{len(out)}"
        df = _votable_regex_frame(chunk.encode())
        if len(df):
            out[name] = df
    return out


def sexagesimal_to_deg(ra_str: str, dec_str: str) -> tuple[float, float]:
    """'00 11 19.43', '-03 09 45.22' -> degrees."""
    def _parts(s):
        return [float(x) for x in re.split(r"[\s:]+", str(s).strip()) if x]
    h, m, s = (_parts(ra_str) + [0, 0])[:3]
    ra = 15.0 * (h + m / 60.0 + s / 3600.0)
    d = _parts(dec_str)
    sign = -1.0 if str(dec_str).strip().startswith("-") else 1.0
    dd, dm, ds = (list(map(abs, d)) + [0, 0])[:3]
    dec = sign * (dd + dm / 60.0 + ds / 3600.0)
    return ra, dec


def unquote_table(name: str) -> str:
    """Strip the double quotes VizieR's ``TAP_SCHEMA`` wraps table names in.

    ``TAP_SCHEMA.tables.table_name`` comes back as ``"J/AJ/159/8/table2"`` —
    *including* the quote characters; interpolating that into a query yields
    ``""J/AJ/159/8/table2""``, which every table rejects.  Run 35653059591
    showed the CSV form of the same answer wrapped in SINGLE quotes
    (``'J/AJ/159/8/table2'``), and three follow-up pulls were mis-spelled.
    """
    return str(name).strip().strip("\"'").strip("\"'").strip()


# --- route 0: ask the IVOA registry where the service actually is -----------
#: RegTAP mirrors.  The registry is the authoritative index of every published
#: VO service, so it answers "where is the VASCO cone search" without guessing
#: URL paths --- which is all runs 30203741898 and 35653059591 could do, and
#: they spent themselves on 404s from ``svo2`` and TCP timeouts from
#: ``svocats``.  A moved or renamed service is found here or nowhere.
REGTAP_ENDPOINTS = (
    "https://reg.g-vo.org/tap/sync",
    "http://dc.g-vo.org/tap/sync",
    "https://registry.euro-vo.org/regtap/tap/sync",
)

#: The RegTAP idiom: resource, its capabilities, their interfaces.  Matching is
#: on title/description/ivoid because the SVO publishes under names this
#: channel cannot predict.
REGTAP_ADQL = (
    "SELECT DISTINCT ivoid, short_name, res_title, access_url, standard_id "
    "FROM rr.resource NATURAL JOIN rr.capability NATURAL JOIN rr.interface "
    "WHERE res_title LIKE '%anish%' OR res_title LIKE '%VASCO%' "
    "OR res_description LIKE '%VASCO%' OR res_description LIKE '%anishing%' "
    "OR ivoid LIKE '%vanish%' OR ivoid LIKE '%vasco%'"
)


def tap_sync_at(base: str, adql: str, cfg: dict, timeout: int | None = None
                ) -> tuple[pd.DataFrame, str, str]:
    """One synchronous ADQL query against an arbitrary TAP ``/sync``."""
    a = cfg.get("acquire", {})
    q = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                "FORMAT": "csv", "MAXREC": 2000, "QUERY": adql})
    full = f"{base}?{q}"
    body, detail = http_get(full, int(timeout or a.get("tap_timeout_s", 120)),
                            retries=1, backoff=5.0)
    if body is None:
        return pd.DataFrame(), full, detail
    try:
        df = pd.read_csv(io.StringIO(body.decode("utf-8", "replace")))
    except Exception as e:                                     # noqa: BLE001
        return pd.DataFrame(), full, f"{detail}; parse: {e}"
    return df, full, detail


def _root_of_access_url(url: str) -> str:
    """A VO ``access_url`` reduced to the root :func:`_svo_urls` expects.

    A cone-search access URL is a base that already ends at the query string
    (``.../vanish-possi/cs.php?``); the probe builds its own query, so the
    trailing ``?``/``&`` and the script name come off again.
    """
    u = str(url).split("#")[0].rstrip("&?")
    if "?" in u:
        u = u.split("?")[0]
    u = re.sub(r"/(cs|cs\.php|conesearch|scs\.php|search|query)$", "", u, flags=re.I)
    return u.rstrip("/")


def discover_registry_services(cfg: dict) -> tuple[list[str], Provenance]:
    """Roots for anything the IVOA registry knows about VASCO / vanishing.

    Returns ``(roots, provenance)``; cone-search interfaces come first because
    those are the ones :func:`probe_svo_catalog` can actually exercise.  Every
    mirror tried and every error is recorded verbatim --- an unreachable
    registry is a statement about the registry, never about the sky.
    """
    a = cfg.get("acquire", {})
    prov = Provenance(route="registry_regtap_discovery")
    endpoints = list(a.get("regtap_endpoints", REGTAP_ENDPOINTS))
    cone: list[str] = []
    other: list[str] = []
    for base in endpoints:
        df, url, detail = tap_sync_at(base, a.get("regtap_adql", REGTAP_ADQL), cfg)
        prov.record(url, len(df) > 0, detail, len(df))
        if not len(df) or "access_url" not in df.columns:
            continue
        for _, row in df.iterrows():
            std = str(row.get("standard_id", "")).lower()
            root = _root_of_access_url(row.get("access_url", ""))
            if not root:
                continue
            (cone if "conesearch" in std else other).append(root)
        prov.notes.append("registry rows: " + "; ".join(
            f"{str(r.get('short_name') or r.get('ivoid'))[:40]} "
            f"[{str(r.get('standard_id', '')).split('/')[-1][:20]}] "
            f"{str(r.get('access_url'))[:90]}"
            for _, r in df.head(12).iterrows())[:1500])
        break                     # the first mirror that answers is enough
    roots = list(dict.fromkeys(cone + other))
    # "empty" only if a mirror actually ANSWERED and knew of no such service;
    # if none answered, the verdict is about the registry, not about the sky.
    answered = any(a["ok"] for a in prov.attempts)
    prov.status = "ok" if roots else ("empty" if answered else "unreachable")
    prov.n_rows = len(roots)
    if not roots:
        prov.notes.append("no VO registry mirror returned a vanishing/VASCO service")
    return roots, prov


# --- route 1: the Solano+2022 VO archive ------------------------------------
def discover_vo_archive(cfg: dict, catalog: str = "vanish_neowise",
                        out_dir: Path | None = None
                        ) -> tuple[list[str], Provenance]:
    """Candidate roots for one VASCO SVO catalogue, configured plus discovered.

    The configured roots are the published ones; discovery scrapes only the
    SVO archive index pages (short timeout) for any ``vanish``/``vasco`` link,
    in case the service moved.  Nothing is written to disk: the previous
    version saved a 44 kB arXiv abstract page into ``results/`` and mined
    nothing from it.
    """
    a = cfg.get("acquire", {})
    prov = Provenance(route=f"discover_vo_archive:{catalog}")
    roots: list[str] = list(a.get("svo_catalogs", {}).get(catalog, []))
    key = catalog.replace("_", "-")

    for url in list(a.get("svo_index_urls", [])):
        body, detail = _probe(url, cfg)
        prov.record(url, body is not None, detail, 0)
        if body is None:
            continue
        txt = body.decode("utf-8", "replace")
        hits = {u.rstrip(".,);\"'") for u in _URL_RE.findall(txt)}
        for h in _HREF_RE.findall(txt):
            if re.search(r"vanish|vasco", h, re.I):
                hits.add(urllib.parse.urljoin(url, h))
        prio = sorted(h for h in hits
                      if re.search(rf"{re.escape(key)}|vanish|vasco", h, re.I))
        roots.extend(h.rstrip("/") for h in prio)
        if prio:
            prov.notes.append(f"{len(prio)} vanish/vasco URL(s) mined from {url}: "
                              f"{prio[:6]}")

    seen, ordered = set(), []
    for r in roots:
        r = r.rstrip("/")
        if r and r not in seen:
            seen.add(r)
            ordered.append(r)
    prov.status = "ok" if ordered else "no_candidates"
    prov.n_rows = len(ordered)
    return ordered, prov


def _svo_urls(root: str, cfg: dict, ra: float | None = None,
              dec: float | None = None, sr: float | None = None) -> list[str]:
    """Query URLs for an SVO ``svocats`` catalogue, most-likely form first.

    With no cone given this builds the whole-sky dump (``SR=180`` about
    ``(180, 0)``, ``format=ascii``) — the form that production code uses
    against other svocats catalogues and the only sane way to pull 1.7x10^5
    rows.
    """
    a = cfg.get("acquire", {})
    if ra is None:
        queries = list(a.get("svo_allsky_queries", [
            "RA=180.000000&DEC=0.000000&SR=180.000000&VERB=2&format=ascii"]))
    else:
        queries = [urllib.parse.urlencode(
            {"RA": f"{ra:.6f}", "DEC": f"{dec:.6f}", "SR": f"{sr:.4f}",
             "VERB": 2}) + "&format=ascii"]
    urls = []
    for q in queries:
        for path in a.get("svo_paths", ["cs.php", "", "cs"]):
            urls.append(f"{root}/{path}?{q}" if path else f"{root}?{q}")
    return urls


def _looks_tabular(body: bytes) -> bool:
    if not body or len(body) < 40:
        return False
    head = body[:600].lower()
    if b"<table" in body[:200000] or b"<tr>" in body[:200000]:
        return True
    if b"<!doctype html" in head or b"<html" in head:
        return False
    first = body[:8000].decode("utf-8", "replace")
    lines = [ln for ln in first.splitlines() if ln.strip()
             and not ln.lstrip().startswith("#")]
    return len(lines) >= 2 and any(sep in lines[0] for sep in (",", "|", "\t"))


def probe_svo_catalog(name: str, roots, cfg: dict) -> tuple[str | None, str, Provenance]:
    """Find the live root for one named SVO catalogue.

    Probes the index page and a small cone with SHORT timeouts, records the
    HTTP status of every form and mines the index for bulk-download links.
    Returns ``(root, working_url_form, provenance)``.
    """
    prov = Provenance(route=f"probe_svo:{name}")
    for root in roots:
        body, detail = _probe(root + "/", cfg)
        prov.record(root + "/", body is not None, detail, 0)
        if body is not None:
            links = sorted({urllib.parse.urljoin(root + "/", h)
                            for h in _HREF_RE.findall(body.decode("utf-8", "replace"))
                            if re.search(r"cs\.php|download|\.csv|\.vot|\.fits|\.gz|"
                                         r"\.tsv|\.dat|search", h, re.I)})
            if links:
                prov.notes.append(f"links on {root}/: {links[:12]}")
        for url in _svo_urls(root, cfg, ra=180.0, dec=0.0, sr=5.0):
            body, detail = _probe(url, cfg)
            ok = _looks_tabular(body) if body else False
            prov.record(url, ok, detail, 0)
            if ok:
                prov.status, prov.url = "ok", url
                return root, url, prov
    prov.status = "unreachable"
    prov.notes.append(f"no root for '{name}' answered a cone search")
    return None, "", prov


def fetch_svo_catalog(name: str, root: str, cfg: dict, out_dir: Path
                      ) -> tuple[pd.DataFrame, Provenance]:
    """Pull an entire SVO catalogue from a root that answered the probe.

    Tries the single whole-sky dump first; if that is refused or truncated,
    falls back to a declination-band sweep and concatenates.
    """
    a = cfg.get("acquire", {})
    prov = Provenance(route=f"fetch_svo:{name}", url=root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for url in _svo_urls(root, cfg):
        body, detail = http_get(url, int(a.get("http_timeout_s", 300)),
                                retries=int(a.get("retries", 2)),
                                backoff=float(a.get("retry_backoff_s", 8.0)))
        if body is None or not _looks_tabular(body):
            prov.record(url, False, detail, 0)
            continue
        (out_dir / f"svo_{name}_allsky.raw").write_bytes(body)
        df = _votable_to_frame(body) if b"<TABLE" in body[:200000] else _read_csvish(body)
        has_pos = len(df) > 0 and _ra_dec_columns(df)[0] is not None
        prov.record(url, has_pos, detail if has_pos else
                    f"{detail}; {len(df)} rows but no RA/DEC column "
                    f"(cols: {list(df.columns)[:12]})", len(df))
        if has_pos:
            prov.status, prov.n_rows, prov.url = "ok", len(df), url
            _check_expected_rows(name, len(df), cfg, prov)
            df.to_parquet(out_dir / f"svo_{name}_raw.parquet", index=False)
            return df, prov

    prov.notes.append("whole-sky dump refused; falling back to a band sweep")
    frames, step = [], 6.0
    for dec in np.arange(-90.0 + step / 2, 90.0, step):
        n_ra = max(int(np.ceil(360.0 * max(np.cos(np.radians(dec)), 0.02) / step)), 1)
        for ra in np.linspace(0.0, 360.0, n_ra, endpoint=False):
            for url in _svo_urls(root, cfg, float(ra), float(dec), step):
                body, detail = http_get(url, int(a.get("http_timeout_s", 300)),
                                        retries=2, backoff=4.0)
                if body is None or not _looks_tabular(body):
                    continue
                df = (_votable_to_frame(body) if b"<TABLE" in body[:200000]
                      else _read_csvish(body))
                if len(df):
                    frames.append(df)
                break
    if not frames:
        prov.status = "empty"
        return pd.DataFrame(), prov
    df = _dedupe_positions(pd.concat(frames, ignore_index=True))
    prov.status, prov.n_rows = "ok", len(df)
    _check_expected_rows(name, len(df), cfg, prov)
    df.to_parquet(out_dir / f"svo_{name}_raw.parquet", index=False)
    return df, prov


def _check_expected_rows(name: str, n: int, cfg: dict, prov: Provenance) -> None:
    """Validate the row count against the published size; never silently accept."""
    exp = cfg.get("acquire", {}).get("expected_rows", {}).get(name)
    if not exp:
        return
    tol = float(cfg.get("acquire", {}).get("expected_rows_tolerance_frac", 0.05))
    frac = abs(n - exp) / float(exp)
    if frac <= tol:
        prov.notes.append(f"row count {n} within {tol:.0%} of the published {exp}")
    else:
        prov.notes.append(
            f"WARNING: row count {n} differs from the published {exp} by "
            f"{frac:.1%} - the fetch may be truncated or the archive changed")


def _read_csvish(body: bytes) -> pd.DataFrame:
    txt = body.decode("utf-8", "replace")
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()
    for sep in (",", "|", "\t", None):
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep,
                             engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:                                      # noqa: BLE001
            continue
    return pd.DataFrame()


def _dedupe_positions(df: pd.DataFrame, tol_deg: float = 1.0 / 3600.0) -> pd.DataFrame:
    ra, dec = _ra_dec_columns(df)
    if ra is None:
        return df.drop_duplicates()
    key = (np.round(df[ra].astype(float) / tol_deg).astype("int64").astype(str)
           + "_" + np.round(df[dec].astype(float) / tol_deg).astype("int64").astype(str))
    return df.loc[~key.duplicated()].reset_index(drop=True)


def _ra_dec_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Locate the coordinate columns whatever the archive chose to call them."""
    cols = {c.lower(): c for c in df.columns}
    named_ra = ("ra_neowise", "ra", "raj2000", "ra_icrs", "_raj2000", "radeg",
                "ra_deg", "alpha", "_ra")
    named_dec = ("dec_neowise", "de_neowise", "dec", "de", "dej2000", "de_icrs",
                 "_dej2000", "dedeg", "dec_deg", "delta", "_de")
    r = next((cols[k] for k in named_ra if k in cols), None)
    d = next((cols[k] for k in named_dec if k in cols), None)
    if r is None:
        r = next((cols[k] for k in cols if k.startswith(("ra", "_ra"))), None)
    if d is None:
        d = next((cols[k] for k in cols
                  if k.startswith(("dec", "de_", "_de")) or k == "de"), None)
    return (r, d) if r is not None and d is not None else (None, None)


# --- route 2: VizieR --------------------------------------------------------
def tap_sync(adql: str, cfg: dict, timeout: int | None = None
             ) -> tuple[pd.DataFrame, str, str]:
    """One synchronous ADQL query against TAPVizieR; ``(frame, url, detail)``."""
    a = cfg.get("acquire", {})
    url = a.get("vizier_tap", VIZIER_TAP)
    q = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                "FORMAT": "csv", "MAXREC": 5000, "QUERY": adql})
    full = f"{url}?{q}"
    body, detail = http_get(full, int(timeout or a.get("tap_timeout_s", 120)),
                            retries=2, backoff=5.0)
    if body is None:
        return pd.DataFrame(), full, detail
    try:
        df = pd.read_csv(io.StringIO(body.decode("utf-8", "replace")))
    except Exception as e:                                     # noqa: BLE001
        return pd.DataFrame(), full, f"{detail}; parse: {e}"
    return df, full, detail


def discover_vizier_tables(cfg: dict) -> tuple[pd.DataFrame, Provenance]:
    """Ask VizieR's own ``TAP_SCHEMA`` for anything that looks like the sample.

    Keyword search over ``table_name`` *and* ``description``: the catalogue
    might have been ingested under any bibcode-derived name, and the 2026-07-26
    "not found" for ``J/MNRAS/515/1380`` was a single-name probe.
    """
    a = cfg.get("acquire", {})
    prov = Provenance(route="vizier_tap_schema_discovery")
    kws = list(a.get("tap_schema_keywords",
                     ["vanish", "VASCO", "Villarroel", "MNRAS/515/1380", "AJ/159/8"]))
    clauses = []
    for k in kws:
        for variant in dict.fromkeys((k, k.lower(), k.upper(), k.capitalize())):
            clauses.append(f"description LIKE '%{variant}%'")
            clauses.append(f"table_name LIKE '%{variant}%'")
    adql = ("SELECT TOP 200 table_name, description FROM TAP_SCHEMA.tables WHERE "
            + " OR ".join(dict.fromkeys(clauses)))
    df, url, detail = tap_sync(adql, cfg)
    prov.record(url, len(df) > 0, detail, len(df))
    if not len(df) or "table_name" not in df.columns:
        prov.status = "empty" if df is not None else "unreachable"
        prov.notes.append("TAP_SCHEMA returned no matching table")
        return pd.DataFrame(columns=["table_name", "description"]), prov
    df = df.copy()
    df["table_name"] = df["table_name"].map(unquote_table)
    prov.status, prov.n_rows = "ok", len(df)
    prov.notes.append("tables: " + "; ".join(
        f"{t} ({str(d)[:60]})" for t, d in zip(df["table_name"], df["description"],
                                                strict=False))[:1500])
    return df, prov


def _solano_candidates(tables: pd.DataFrame) -> list[str]:
    """Discovered tables that could be the Solano+2022 sample (not J/AJ/159/8)."""
    out = []
    for t, d in zip(tables.get("table_name", []), tables.get("description", []),
                    strict=False):
        t, d = unquote_table(t), str(d)
        if "J/AJ/159/8" in t:
            continue
        # Word-bounded and case-sensitive for VASCO: the keyword search also
        # returns "Vasco D." and "Vasconcelos" (run 35653059591).
        if (re.search(r"vanish", t + " " + d, re.I) or re.search(r"\bVASCO\b", t + " " + d)
                or "515/1380" in t or re.search(r"Solano E\.", d)):
            out.append(t)
    return out


def probe_second_digitisation(cfg: dict) -> Provenance:
    """Is an INDEPENDENT scan of the same POSS-I plates reachable?

    The single strongest kill available to this channel is not astrophysical.
    A POSS-I-red-only detection that a *second, independent digitisation of
    the same glass* does not see is almost certainly a scan artefact of the
    first digitisation rather than a source that was on the plate --- which is
    precisely how Solano+2022 classified 3 592 of their 298 165 objects, using
    SuperCOSMOS against the USNO-B1.0-era scans.  Hambly & Blair 2024 argue
    the VASCO transients are emulsion artefacts, so a plate-level confirmation
    is the difference between a candidate and a speck of dust.

    This is a REACHABILITY probe only: it reports which of the candidate
    routes answers, and touches no science.  The kill itself needs a route
    that answers, and the answer has to be recorded before it can be used.

    Note the asymmetry that governs how the result may ever be read.  Presence
    in a second digitisation CONFIRMS a plate image.  Absence is informative
    only where the second scan is demonstrably deeper than the magnitude
    claimed for the source --- a source missing from a shallower catalogue is
    not missing, and treating it as missing is the same depth error the
    modern-optical kill already closes.
    """
    a = cfg.get("acquire", {})
    s = a.get("second_digitisation", {})
    prov = Provenance(route="second_digitisation_probe")
    found: list[str] = []

    kws = list(s.get("discovery_keywords",
                     ["SuperCOSMOS", "Hambly", "MNRAS/326/1279"]))
    clauses = []
    for k in kws:
        for variant in dict.fromkeys((k, k.lower(), k.upper())):
            clauses.append(f"description LIKE '%{variant}%'")
            clauses.append(f"table_name LIKE '%{variant}%'")
    adql = ("SELECT TOP 100 table_name, description FROM TAP_SCHEMA.tables WHERE "
            + " OR ".join(dict.fromkeys(clauses)))
    df, url, detail = tap_sync(adql, cfg)
    prov.record(url, len(df) > 0, detail, len(df))
    if len(df) and "table_name" in df.columns:
        names = [unquote_table(t) for t in df["table_name"]]
        found.extend(names)
        prov.notes.append("VizieR TAP_SCHEMA: " + "; ".join(
            f"{t} ({str(d)[:60]})"
            for t, d in zip(names, df.get("description", names), strict=False))[:1200])
    else:
        prov.notes.append("VizieR TAP_SCHEMA knows no SuperCOSMOS-like table")

    # The WFAU SuperCOSMOS Science Archive is served from Edinburgh, not CDS,
    # so it is probed on its own endpoints rather than assumed absent.
    for base in s.get("ssa_tap_urls", []):
        q = urllib.parse.urlencode(
            {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "MAXREC": 1,
             "QUERY": "SELECT TOP 1 table_name FROM TAP_SCHEMA.tables"})
        full = f"{base}?{q}"
        body, detail = http_get(full, int(s.get("probe_timeout_s", 25)),
                                retries=1, backoff=3.0)
        ok = bool(body) and _looks_tabular(body)
        prov.record(full, ok, detail, 1 if ok else 0)
        if ok:
            found.append(base)
            break

    # Any literal VizieR id the keyword sweep cannot reach, asked by name.
    for cat in s.get("candidate_tables", []):
        table = str(cat.get("table", cat) if isinstance(cat, dict) else cat)
        tabs, p_m = vizier_catalogue_meta(table, cfg)
        prov.attempts.extend(p_m.attempts)
        if tabs:
            found.extend(tabs)
            prov.notes.append(f"{table} is in VizieR as {tabs[:6]}")
        else:
            prov.notes.append(f"{table}: {p_m.status}")

    prov.n_rows = len(dict.fromkeys(found))
    prov.status = "ok" if found else "unreachable"
    if not found:
        prov.notes.append("no independent digitisation of the POSS-I plates "
                          "answered; the plate-confirmation kill is NOT "
                          "available and no source may be vetoed for lacking it")
    return prov


def vizier_catalogue_meta(cat: str, cfg: dict) -> tuple[list[str], Provenance]:
    """Does VizieR hold ``cat`` at all, and under what table names?

    ``-meta.all`` returns the catalogue's tables and columns instead of rows.
    This separates the two answers a bare row query confuses: *the catalogue
    is not in VizieR* and *the catalogue is there but the query was wrong*.
    Used for the literal ids keyword discovery cannot reach --- the
    Solano+2022 by-product tables are published under a bibcode-derived name
    whose description carries neither 'vanish' nor a bare 'VASCO', so the
    TAP_SCHEMA keyword sweep of run 35653059591 returned only 'Vasco D.' and
    'Vasconcelos' and never looked the catalogue up by name.
    """
    a = cfg.get("acquire", {})
    prov = Provenance(route=f"vizier_meta:{cat}")
    url = (f"{a.get('vizier_asu', VIZIER_ASU)}?"
           f"-source={urllib.parse.quote(cat, safe='/')}&-meta.all&-out.form=TSV")
    body, detail = http_get(url, int(a.get("tap_timeout_s", 120)), retries=1, backoff=5.0)
    prov.record(url, body is not None, detail)
    if body is None:
        prov.status = "unreachable"
        return [], prov
    txt = body.decode("utf-8", "replace")
    errs = [ln.strip() for ln in txt.splitlines()
            if ln.startswith("#***") or ln.startswith("****")]
    tables = sorted({unquote_table(m) for m in
                     re.findall(rf"{re.escape(cat)}/[A-Za-z0-9_.+-]+", txt)})
    if errs:
        prov.notes.append("ASU: " + " | ".join(errs)[:400])
    prov.status = "ok" if tables else ("asu_error" if errs else "absent")
    prov.n_rows = len(tables)
    prov.notes.append(f"tables reported for {cat}: {tables[:20] or 'none'}")
    return tables, prov


def fetch_vizier_table_asu(table: str, cfg: dict, out_dir: Path,
                           max_rows: int = 400000) -> tuple[pd.DataFrame, Provenance]:
    """Pull a whole VizieR table through ASU (tab-separated), all columns."""
    from ..metronome.acquire import parse_asu_tsv  # noqa: PLC0415  (shared parser)

    a = cfg.get("acquire", {})
    prov = Provenance(route=f"vizier_asu:{table}")
    q = urllib.parse.urlencode([("-source", table), ("-out.all", ""),
                                ("-out.max", str(int(max_rows)))],
                               quote_via=urllib.parse.quote)
    url = f"{a.get('vizier_asu', VIZIER_ASU)}?{q}"
    body, detail = http_get(url, int(a.get("http_timeout_s", 300)), retries=2,
                            backoff=6.0)
    prov.record(url, body is not None, detail)
    if body is None:
        prov.status = "unreachable"
        return pd.DataFrame(), prov
    df = parse_asu_tsv(body.decode("utf-8", "replace"))
    errs = df.attrs.get("asu_errors", [])
    if errs:
        prov.notes.append("ASU errors: " + " | ".join(map(str, errs))[:400])
    prov.status = "ok" if len(df) else "empty"
    prov.n_rows = len(df)
    if len(df):
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"\W+", "_", table)
        df.to_parquet(out_dir / f"vizier_{stem}.parquet", index=False)
    return df, prov


def fetch_vizier_vasco2020(cfg: dict, out_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    """Villarroel+2020 J/AJ/159/8 — table2 (~100 candidates) + table3."""
    a = cfg.get("acquire", {})
    prov = Provenance(route="vizier_J/AJ/159/8")
    url = a.get("vizier_votable",
                "https://vizier.cds.unistra.fr/viz-bin/votable?-source={cat}&-out.max={maxrows}"
                ).format(cat=a.get("vizier_vasco2020_cat", "J/AJ/159/8"),
                         maxrows=100000)
    body, detail = http_get(url, int(a.get("http_timeout_s", 300)),
                            retries=int(a.get("retries", 2)),
                            backoff=float(a.get("retry_backoff_s", 8.0)))
    prov.record(url, body is not None, detail)
    if body is None:
        prov.status = "unreachable"
        return pd.DataFrame(), prov
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vizier_JAJ159_8.xml").write_bytes(body)
    return parse_vizier_vasco2020(body, prov)


def parse_vizier_vasco2020(body: bytes, prov: Provenance | None = None
                           ) -> tuple[pd.DataFrame, Provenance]:
    """Parse the J/AJ/159/8 VOTable into a uniform position table (offline-safe)."""
    prov = prov or Provenance(route="vizier_J/AJ/159/8")
    tabs = votable_tables(body)
    recs = []
    for name, df in tabs.items():
        cols = {c.lower(): c for c in df.columns}
        rc, dc = cols.get("raj2000"), cols.get("dej2000")
        if rc is None or dc is None:
            continue
        tag = "table3" if name.endswith("table3") else (
            "table2" if name.endswith("table2") else name)
        for _, row in df.iterrows():
            try:
                ra, dec = sexagesimal_to_deg(row[rc], row[dc])
            except Exception:                                  # noqa: BLE001
                continue
            rmag = row.get(cols.get("rmag", ""), None)
            try:
                rmag = float(rmag)
            except (TypeError, ValueError):
                rmag = float("nan")
            recs.append({"source_id": f"VASCO2020-{tag}-{len(recs):04d}",
                         "ra_deg": ra, "dec_deg": dec, "poss1_e": rmag,
                         "vizier_table": tag,
                         "sample": "vasco2020_surviving_candidates"})
    df = pd.DataFrame(recs)
    prov.status = "ok" if len(df) else "empty"
    prov.n_rows = len(df)
    if not len(df):
        prov.notes.append("VOTable parsed but held no RAJ2000/DEJ2000 rows")
    return df, prov


# --- route 3: reconstruct the selection from USNO-B1.0 ----------------------
def field_grid(n_fields: int, cfg: dict, seed: int = 0) -> pd.DataFrame:
    """A deterministic, well-spread set of field centres.

    Fibonacci-sphere points restricted to the POSS-I footprint that USNO-B1.0
    labels as POSS-I (``dec >= usnob1_dec_min``; south of -30 the first-epoch
    plates are SERC-J/ESO-R) and to ``|b| >= abs_glat_min`` where the WISE
    chance-match rate and the plate crowding are both low.  The seed only
    shuffles which of the spread points are taken first, so ``n_fields`` can
    grow between runs without moving the fields already done.
    """
    from .classify import galactic_latitude

    r = cfg.get("acquire", {}).get("reconstruct", {})
    dec_min = float(r.get("dec_min_deg", -25.0))
    dec_max = float(r.get("dec_max_deg", 88.0))
    b_min = float(r.get("abs_glat_min_deg", 20.0))
    n_cand = 6000
    i = np.arange(n_cand) + 0.5
    dec = np.degrees(np.arcsin(1.0 - 2.0 * i / n_cand))
    ra = (np.degrees(i * math.pi * (3.0 - math.sqrt(5.0)))) % 360.0
    glat = galactic_latitude(ra, dec)
    keep = (dec >= dec_min) & (dec <= dec_max) & (np.abs(glat) >= b_min)
    ra, dec, glat = ra[keep], dec[keep], glat[keep]
    order = np.random.default_rng(seed).permutation(len(ra))
    take = order[:max(int(n_fields), 0)]
    return pd.DataFrame({"field_id": np.arange(len(take)),
                         "ra_deg": ra[take], "dec_deg": dec[take],
                         "glat_deg": glat[take]})


#: Columns the reconstruction needs from I/284/out, in VizieR's spelling.
USNOB1_COLUMNS = ("USNO-B1.0", "RAJ2000", "DEJ2000", "Epoch", "pmRA", "pmDE", "muPr",
                  "Ndet", "Flags", "B1mag", "R1mag", "R1S", "R1f", "R1s/g", "B2mag",
                  "R2mag", "Imag")

#: Query forms for one USNO-B1.0 field, most selective first.  Which form the
#: service actually honours is DISCOVERED on the first field, not assumed:
#: run 35653059591 got HTTP 200 and zero rows from every field with the
#: fully-constrained form and nothing else was tried.  A form's server-side
#: constraint is always re-applied locally, so a form that ignores a
#: constraint (and returns more) is merely slower, never wrong.
USNOB1_QUERY_FORMS = ("ndet+r1", "ndet", "r1", "none", "none_allcols")


def usnob1_field_url(ra: float, dec: float, radius_deg: float, cfg: dict,
                     form: str = "ndet+r1", columns=None) -> str:
    """ASU query for POSS-I-red-only USNO-B1.0 objects inside one cone.

    ``-c`` carries an explicit sign and is percent-encoded: a literal ``+``
    decodes to a space and VizieR then does not read the pair as a position
    (the IGNITION channel lost a dispatch to exactly that).
    """
    a = cfg.get("acquire", {})
    r = a.get("reconstruct", {})
    params = [
        ("-source", r.get("usnob1_table", "I/284/out")),
        ("-c", f"{ra:.6f} {dec:+.6f}"),
        ("-c.eq", "J2000"),
        ("-c.rd", f"{radius_deg:.4f}"),
        ("-out.max", str(int(r.get("max_rows_per_field", 200000)))),
    ]
    if form == "none_allcols":
        params.append(("-out.all", ""))
    else:
        # REPEATED ``-out=`` per column, not one comma-joined value: the
        # comma-joined spelling is what the first sweep sent and every field
        # came back HTTP 200 with a header and no rows.  Repeated ``-out=`` is
        # the spelling the METRONOME route has actually been answered on.
        params.extend(("-out", c) for c in (columns or USNOB1_COLUMNS))
    # ``asu-tsv`` names the format in the path, but the ASU dispatcher only
    # emits the tab-separated body when the form is asked for explicitly; a
    # request without it can be answered as metadata alone.
    params.append(("-out.form", "TSV"))
    if form in ("ndet+r1", "ndet"):
        params.append(("Ndet", str(int(r.get("ndet", 1)))))
    if form in ("ndet+r1", "r1"):
        params.append(("R1mag", f"<={float(r.get('r1_max_mag', 19.3)):.2f}"))
    return f"{a.get('vizier_asu', VIZIER_ASU)}?" + urllib.parse.urlencode(
        params, quote_via=urllib.parse.quote)


def usnob1_meta_url(cfg: dict) -> str:
    """``-meta.all`` for I/284/out: the catalogue's own column spelling.

    Run 35653059591 asked for columns this channel had *assumed*; one wrong
    name is enough for ASU to answer with a header and no rows, and nothing in
    the ledger could tell that apart from an empty sky.  The reconstruction
    now reads the real names off the service before it trusts a zero.
    """
    a = cfg.get("acquire", {})
    r = a.get("reconstruct", {})
    src = str(r.get("usnob1_table", "I/284/out"))
    return (f"{a.get('vizier_asu', VIZIER_ASU)}?"
            f"-source={urllib.parse.quote(src, safe='/')}&-meta.all&-out.form=TSV")


def usnob1_meta_probe(cfg: dict) -> dict:
    """Ask I/284/out for its own column list before believing any zero.

    Returns ``{url, detail, columns, missing, body_head}``.

    ``missing`` is the subset of :data:`USNOB1_COLUMNS` this body does not
    mention.  Read it as a HINT, never as a fact about the catalogue: the
    ``-meta.all`` + ``-out.form=TSV`` body carries ``#Column`` lines for the
    catalogue's DEFAULT output columns, and I/284/out's defaults are the eight
    astrometric ones, so run 35738062833's probe reported B1mag, R1mag, R2mag,
    Imag and Ndet as "missing" from a catalogue that plainly has them.  That
    probe then edited the request and every field came back as bare positions;
    it now reports only.  A wrong column name is handled where it shows up ---
    the ladder falls through to the ``-out.all`` rung, which names none.
    """
    url = usnob1_meta_url(cfg)
    r = cfg.get("acquire", {}).get("reconstruct", {})
    body, detail = http_get(url, int(r.get("meta_timeout_s", 60)), retries=1, backoff=5.0)
    out: dict = {"url": url, "detail": detail, "columns": [], "missing": [],
                 "body_head": ""}
    if body is None:
        return out
    from ..metronome.acquire import asu_body_head  # noqa: PLC0415

    txt = body.decode("utf-8", "replace")
    out["body_head"] = asu_body_head(txt, 1500)
    names: list[str] = []
    for ln in txt.splitlines():
        if ln.startswith("#Column"):
            cells = [c.strip() for c in ln.split("\t") if c.strip()]
            for c in cells[1:]:
                if not c.startswith(("(", "[")):
                    names.append(unquote_table(c))
                    break
    out["columns"] = sorted(dict.fromkeys(names))
    if names:
        have = {n.lower() for n in names}
        out["missing"] = [c for c in USNOB1_COLUMNS if c.lower() not in have]
    return out


def fetch_usnob1_field(ra: float, dec: float, radius_deg: float, cfg: dict,
                       forms=USNOB1_QUERY_FORMS, columns=None
                       ) -> tuple[pd.DataFrame, str, list[dict]]:
    """One field through the query-form ladder; ``(raw, form_used, attempts)``.

    The head of every empty or failed body is kept, so "zero rows" is a
    reading of what VizieR said rather than a guess.  Returns the first form
    that yields rows (or the last attempt's empty frame).
    """
    from ..metronome.acquire import asu_body_head, parse_asu_tsv  # noqa: PLC0415

    r = cfg.get("acquire", {}).get("reconstruct", {})
    attempts: list[dict] = []
    raw = pd.DataFrame()
    for form in forms:
        url = usnob1_field_url(ra, dec, radius_deg, cfg, form, columns)
        t0 = time.time()
        body, detail = http_get(url, int(r.get("field_timeout_s", 240)),
                                retries=int(r.get("field_retries", 2)), backoff=10.0)
        rec = {"form": form, "url": url, "detail": detail,
               "seconds": round(time.time() - t0, 1), "n_raw": 0}
        if body is None:
            attempts.append(rec)
            continue
        txt = body.decode("utf-8", "replace")
        raw = parse_asu_tsv(txt)
        errs = raw.attrs.get("asu_errors", [])
        rec["n_raw"] = int(len(raw))
        if errs:
            rec["asu_errors"] = [str(e)[:200] for e in errs[:5]]
        if not len(raw):
            rec["body_head"] = asu_body_head(txt, 1200)
        rec["columns"] = [str(c) for c in raw.columns][:40]
        # ROWS ARE NOT ENOUGH.  Run 35738062833 got 1380-5899 rows from every
        # field and reconstructed ZERO sources, because the answer carried
        # positions and no photometry at all: the POSS-I-red-only mask is a
        # statement about which plate magnitudes are present, so a frame
        # without them cannot express the selection and its "no survivors" is
        # an artefact of the request, not of the sky.  A rung that answers
        # without the columns the selection needs is not accepted; the ladder
        # falls through to the ``-out.all`` rung, which names no columns.
        need = [str(c) for c in r.get("required_columns", ("RAJ2000", "DEJ2000", "R1mag"))]
        missing = [c for c in need if c not in raw.columns]
        rec["missing_required"] = missing
        attempts.append(rec)
        if len(raw) and not missing:
            return raw, form, attempts
    return pd.DataFrame(), "", attempts


def normalise_usnob1_frame(df: pd.DataFrame, field_id: int = -1,
                           r1_max: float | None = None,
                           ndet: int | None = None) -> pd.DataFrame:
    """USNO-B1.0 columns -> channel schema, keeping only POSS-I-E-only rows.

    ``R1mag`` present and ``B1/B2/R2/I`` absent is the definition; ``Ndet``
    is re-checked rather than trusted from the query.  The plate epoch, the
    diffraction-spike flag and the star/galaxy index are carried for the vet.
    """
    if not len(df):
        return pd.DataFrame()
    low = {c.lower(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n.lower() in low:
                return pd.to_numeric(df[low[n.lower()]], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        "ra_deg": col("RAJ2000", "_RAJ2000", "RA_ICRS"),
        "dec_deg": col("DEJ2000", "_DEJ2000", "DE_ICRS"),
        "poss1_e": col("R1mag"), "poss1_o": col("B1mag"),
        "poss2_j": col("B2mag"), "poss2_f": col("R2mag"), "poss2_n": col("Imag"),
        "epoch_poss1": col("Epoch"), "usnob_ndet": col("Ndet"),
        "usnob_pmra": col("pmRA"), "usnob_pmde": col("pmDE"), "usnob_mupr": col("muPr"),
        "r1_sg": col("R1s/g"), "r1_survey": col("R1S"), "r1_field": col("R1f"),
    })
    idc = low.get("usno-b1.0")
    out["usnob_id"] = df[idc].astype(str) if idc else [f"f{field_id}-{i}" for i in df.index]
    fl = low.get("flags")
    # pandas 3 string dtype keeps NaN through astype(str); fill first.
    out["usnob_flags"] = (df[fl].fillna("").astype(str).replace({"nan": "", "None": ""})
                          if fl else "")
    out["field_id"] = int(field_id)
    only = (out["poss1_e"].notna() & out["poss1_o"].isna() & out["poss2_j"].isna()
            & out["poss2_f"].isna() & out["poss2_n"].isna())
    # The server-side constraints are re-applied here, so a query form that
    # ignored one of them still yields exactly the intended selection.
    if r1_max is not None:
        only &= out["poss1_e"] <= float(r1_max)
    if ndet is not None:
        only &= out["usnob_ndet"].isna() | (out["usnob_ndet"] == int(ndet))
    out = out[only & out["ra_deg"].notna() & out["dec_deg"].notna()].copy()
    out["source_id"] = "USNOB-" + out["usnob_id"].astype(str)
    out["sample"] = "usnob1_poss1_red_only"
    return out.drop_duplicates(subset=["source_id"]).reset_index(drop=True)


def reconstruct_from_usnob1(cfg: dict, out_dir: Path, n_fields: int | None = None,
                            radius_deg: float | None = None, seed: int | None = None,
                            deadline_s: float | None = None
                            ) -> tuple[pd.DataFrame, Provenance, list[dict]]:
    """Pull POSS-I-red-only USNO-B1.0 objects field by field, checkpointing.

    Returns ``(sample, provenance, field_ledger)``.  A field already on disk
    is reused, so a re-dispatch with more fields only fetches the new ones.
    """
    a = cfg.get("acquire", {})
    r = a.get("reconstruct", {})
    n_fields = int(n_fields if n_fields is not None else r.get("n_fields", 40))
    radius_deg = float(radius_deg if radius_deg is not None
                       else r.get("field_radius_deg", 0.5))
    seed = int(seed if seed is not None else r.get("seed", 0))
    prov = Provenance(route="usnob1_reconstruction", url=a.get("vizier_asu", VIZIER_ASU))
    grid = field_grid(n_fields, cfg, seed)
    raw_dir = out_dir / "usnob1_fields"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ledger: list[dict] = []
    frames: list[pd.DataFrame] = []
    t_start = time.time()
    n_fail = 0
    r1_max = float(r.get("r1_max_mag", 19.3))
    ndet = int(r.get("ndet", 1))
    # The query form is discovered on the first live field and then reused;
    # a form that returned rows there is tried first everywhere after.
    forms: list[str] = list(USNOB1_QUERY_FORMS)
    form_used: str | None = None
    ndet_hist: dict = {}
    r1_hist: dict = {}
    meta = usnob1_meta_probe(cfg)
    prov.notes.append(
        f"I/284/out -meta.all: {meta['detail']}; "
        + (f"{len(meta['columns'])} column(s) reported: {meta['columns'][:40]}"
           if meta["columns"] else "no column names parsed")
        + (f"; not mentioned by the probe: {sorted(meta['missing'])}"
           if meta["missing"] else "; every requested column mentioned")
        + " (reported only -- the request is not edited from this; the "
          "-meta.all body lists the catalogue's DEFAULT output columns, so "
          "'not mentioned' does NOT mean 'absent from the catalogue')")
    # The probe REPORTS; it does not edit the request.  Run 35738062833 had it
    # strip every name the -meta.all body failed to mention, and that body
    # mentioned 8 columns out of ~30 --- so B1mag/R1mag/R2mag/Ndet were all
    # dropped, the fields came back as bare positions, and the selection could
    # not be expressed.  A bad column name is handled where it shows up: the
    # ladder falls through to the ``-out.all`` rung.
    columns = None
    for _, f in grid.iterrows():
        fid = int(f["field_id"])
        rec = {"field_id": fid, "ra_deg": float(f["ra_deg"]),
               "dec_deg": float(f["dec_deg"]), "glat_deg": float(f["glat_deg"]),
               "radius_deg": radius_deg, "n_raw": 0, "n_poss1_only": 0,
               "status": "", "detail": "", "seconds": 0.0, "form": ""}
        ck = raw_dir / f"field_{fid:04d}.parquet"
        if ck.exists():
            df = pd.read_parquet(ck)
            rec.update({"status": "cached", "n_raw": int(len(df)),
                        "n_poss1_only": int(len(df))})
            frames.append(df)
            ledger.append(rec)
            continue
        if deadline_s is not None and time.time() - t_start > deadline_s:
            rec["status"] = "skipped_deadline"
            ledger.append(rec)
            continue
        raw, form, attempts = fetch_usnob1_field(float(f["ra_deg"]), float(f["dec_deg"]),
                                                 radius_deg, cfg, forms, columns)
        rec["seconds"] = round(sum(a["seconds"] for a in attempts), 1)
        rec["detail"] = attempts[-1]["detail"] if attempts else ""
        rec["form"] = form
        rec["attempts"] = attempts
        if not raw.attrs.get("asu_errors") and not len(raw) and all(
                a.get("detail", "").startswith(("URLError", "HTTP 5", "TimeoutError"))
                or "HTTP 200" not in a.get("detail", "") for a in attempts):
            rec["status"] = "unreachable"
            n_fail += 1
            for a in attempts:
                prov.record(a["url"], False, a["detail"], 0)
            ledger.append(rec)
            if n_fail >= int(r.get("max_consecutive_failures", 6)) and not frames:
                prov.notes.append("aborting the field sweep: the first "
                                  f"{n_fail} fields all failed")
                break
            continue
        errs = raw.attrs.get("asu_errors", []) if hasattr(raw, "attrs") else []
        rec["n_raw"] = int(len(raw))
        df = normalise_usnob1_frame(raw, fid, r1_max=r1_max, ndet=ndet)
        rec["n_poss1_only"] = int(len(df))
        rec["status"] = "ok" if len(raw) else ("asu_error" if errs else "empty")
        if len(raw) and "Ndet" in raw.columns:
            h = pd.to_numeric(raw["Ndet"], errors="coerce").value_counts().to_dict()
            for k, v in h.items():
                ndet_hist[str(int(k))] = ndet_hist.get(str(int(k)), 0) + int(v)
        if len(raw) and "R1mag" in raw.columns:
            h = pd.cut(pd.to_numeric(raw["R1mag"], errors="coerce"),
                       [0, 12, 14, 16, 17, 18, 19, 19.3, 20, 25]).value_counts().to_dict()
            for k, v in h.items():
                r1_hist[str(k)] = r1_hist.get(str(k), 0) + int(v)
        if form and form_used != form:
            form_used = form
            forms = [form] + [x for x in USNOB1_QUERY_FORMS if x != form]
            prov.notes.append(f"query form '{form}' returned rows on field {fid}; "
                              "using it first from here on")
        prov.record(attempts[-1]["url"] if attempts else "", len(raw) > 0,
                    rec["detail"], len(df))
        if len(df):
            df.to_parquet(ck, index=False)
            frames.append(df)
        elif len(raw) > 0:
            # A real cone with no POSS-I-red-only object: recorded, cached.
            pd.DataFrame(columns=["source_id", "ra_deg", "dec_deg"]).to_parquet(ck, index=False)
        ledger.append(rec)
        time.sleep(float(r.get("pause_s", 0.5)))

    (out_dir / "field_ledger.json").write_text(json.dumps(
        {"n_fields_requested": n_fields, "radius_deg": radius_deg, "seed": seed,
         "area_deg2_per_field": math.pi * radius_deg ** 2,
         "n_fields_ok": sum(1 for x in ledger if x["status"] in ("ok", "cached")),
         "query_form_used": form_used, "r1_max_mag": r1_max, "ndet": ndet,
         "ndet_histogram_raw": ndet_hist, "r1mag_histogram_raw": r1_hist,
         "meta_probe": meta, "columns_requested": list(columns or USNOB1_COLUMNS),
         "fields": ledger}, indent=1, default=str))
    if not frames:
        prov.status = "unreachable" if n_fail else "empty"
        return pd.DataFrame(), prov, ledger
    df = pd.concat([x for x in frames if len(x)], ignore_index=True) \
        if any(len(x) for x in frames) else pd.DataFrame()
    prov.status = "ok" if len(df) else "empty"
    prov.n_rows = int(len(df))
    prov.notes.append(f"{sum(1 for x in ledger if x['status'] in ('ok', 'cached'))} "
                      f"field(s) fetched, {sum(x['n_raw'] for x in ledger)} raw rows, "
                      f"{len(df)} POSS-I-red-only objects")
    return df, prov, ledger


# --- X-Match ----------------------------------------------------------------
def xmatch_upload(positions: pd.DataFrame, catalog: str, radius_arcsec: float,
                  cfg: dict, ra_col: str = "ra_deg", dec_col: str = "dec_deg",
                  ) -> tuple[pd.DataFrame, Provenance]:
    """Crossmatch an uploaded position list against a VizieR catalogue.

    Uses the CDS X-Match service, chunked.  ``source_id`` is uploaded alongside
    the coordinates and echoed back by the service, so results join
    unambiguously even when a source has several matches.
    """
    a = cfg.get("acquire", {})
    urls = list(a.get("xmatch_urls", XMATCH_URLS))
    if a.get("xmatch_url"):
        urls = [a["xmatch_url"], *[u for u in urls if u != a["xmatch_url"]]]
    chunk = int(a.get("xmatch_chunk_rows", 25000))
    prov = Provenance(route=f"xmatch:{catalog}", url=urls[0])
    frames = []
    cols = [ra_col, dec_col] + (["source_id"] if "source_id" in positions else [])
    for start in range(0, len(positions), chunk):
        sub = positions.iloc[start:start + chunk]
        csv = sub[cols].rename(
            columns={ra_col: "ra", dec_col: "dec"}).to_csv(index=False)
        body_parts, boundary = _multipart({
            "request": "xmatch", "distMaxArcsec": f"{radius_arcsec:g}",
            "RESPONSEFORMAT": "csv", "cat2": catalog,
            "colRA1": "ra", "colDec1": "dec", "selection": "all",
        }, {"cat1": ("positions.csv", csv)})
        body = None
        for url in urls:
            body, detail = http_get(
                url, int(a.get("xmatch_timeout_s", 600)),
                retries=int(a.get("retries", 2)),
                backoff=float(a.get("retry_backoff_s", 8.0)),
                data=body_parts,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            if body is not None:
                break
            prov.record(url, False, detail, 0)
        if body is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(body.decode("utf-8", "replace")),
                             low_memory=False)
        except Exception as e:                                 # noqa: BLE001
            prov.record(url, False, f"parse: {e}", 0)
            continue
        if "source_id" not in df.columns:
            prov.record(url, False, f"{detail}; no source_id echoed "
                        f"(cols: {list(df.columns)[:10]})", len(df))
            continue
        frames.append(df)
        prov.record(url, True, detail, len(df))
        time.sleep(1.0)
    if not frames:
        prov.status = "unreachable" if not prov.attempts or not any(
            x["ok"] for x in prov.attempts) else "empty"
        return pd.DataFrame(), prov
    out = pd.concat(frames, ignore_index=True)
    prov.status, prov.n_rows = "ok", len(out)
    return out, prov


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = "----SetiShroud" + str(int(time.time() * 1e6))
    buf = io.BytesIO()
    for k, v in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        buf.write(f"{v}\r\n".encode())
    for k, (fname, content) in files.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{k}"; '
                  f'filename="{fname}"\r\n'.encode())
        buf.write(b"Content-Type: text/csv\r\n\r\n")
        buf.write(content.encode())
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), boundary


# VizieR column names -> channel schema.  Resolved case-insensitively with
# fallbacks, because VizieR renames columns between catalogue versions.
_COLMAP: dict[str, dict[str, tuple[str, ...]]] = {
    "allwise": {"w1": ("w1mag",), "w2": ("w2mag",), "w3": ("w3mag",),
                "w4": ("w4mag",), "w1_err": ("e_w1mag",), "w2_err": ("e_w2mag",),
                "w3_err": ("e_w3mag",), "w4_err": ("e_w4mag",),
                "ir_ext_flag": ("ex", "extflg", "ext_flg"),
                "wise_ccf": ("ccf",), "wise_qph": ("qph",), "wise_id": ("allwise",)},
    "catwise": {"w1": ("w1mpropm", "w1mpro", "w1mag"),
                "w2": ("w2mpropm", "w2mpro", "w2mag"),
                "w1_err": ("e_w1mpropm", "e_w1mpro"),
                "w2_err": ("e_w2mpropm", "e_w2mpro")},
    "twomass": {"2mass_j": ("jmag",), "2mass_h": ("hmag",),
                "2mass_ks": ("kmag", "ksmag"), "2mass_j_err": ("e_jmag",),
                "2mass_h_err": ("e_hmag",), "2mass_ks_err": ("e_kmag",),
                "twomass_qflg": ("qflg",)},
    "ps1": {"ps1_g": ("gmag",), "ps1_r": ("rmag",), "ps1_i": ("imag",),
            "ps1_z": ("zmag",), "ps1_y": ("ymag",), "ps1_id": ("objid",)},
    "gaia": {"gaia_g": ("gmag",), "gaia_bp": ("bpmag",), "gaia_rp": ("rpmag",),
             "pmra": ("pmra",), "pmdec": ("pmde", "pmdec"),
             "parallax": ("plx", "parallax"), "gaia_source": ("source", "source_id_gaia")},
    "usnob1": {"poss1_o": ("b1mag",), "poss1_e": ("r1mag",)},
}
_DIST_COLS = ("angDist", "angdist", "ang_dist", "_r")
# Bands whose catalogue value is a 95% upper limit when the error is null
# (AllWISE: ``qph`` = U; 2MASS: ``Qflg`` = U).  Passing those as detections
# would fabricate W3/W4 "photometry" for nearly every faint source.
_LIMIT_ON_NULL_ERR = {"allwise": ("w1", "w2", "w3", "w4"),
                      "twomass": ("2mass_j", "2mass_h", "2mass_ks"),
                      "catwise": ("w1", "w2")}


def _limits_from_null_errors(sub: pd.DataFrame, name: str) -> pd.DataFrame:
    """Move catalogue upper limits out of the detection columns.

    AllWISE lists a 95% upper limit in ``W3mag``/``W4mag`` (and 2MASS in
    ``Jmag``…) whenever the band is undetected, with a null error and a ``U``
    in the quality string.  Applied only when the table actually carries the
    error or quality column — with neither there is nothing to judge by, and
    a synthetic frame without error columns must not lose its detections.
    """
    bands = _LIMIT_ON_NULL_ERR.get(name, ())
    qcol = "wise_qph" if name == "allwise" else ("twomass_qflg" if name == "twomass" else None)
    qual = sub[qcol].astype(str) if qcol and qcol in sub.columns else None
    for i, band in enumerate(bands):
        if band not in sub.columns:
            continue
        mag = pd.to_numeric(sub[band], errors="coerce")
        has_err = f"{band}_err" in sub.columns
        is_lim = pd.Series(False, index=sub.index)
        if has_err:
            err = pd.to_numeric(sub[f"{band}_err"], errors="coerce")
            is_lim |= mag.notna() & err.isna()
        if qual is not None:
            is_lim |= mag.notna() & (qual.str.len() > i) & (qual.str[i:i + 1].str.upper() == "U")
        if not has_err and qual is None:
            continue
        sub[f"{band}_lim"] = mag.where(is_lim)
        sub[band] = mag.where(~is_lim)
    return sub


def _pick(df: pd.DataFrame, names) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def _dist_col(df: pd.DataFrame) -> str | None:
    return _pick(df, [c.lower() for c in _DIST_COLS])


def nearest_per_source(xm: pd.DataFrame, id_col: str = "source_id") -> pd.DataFrame:
    """Keep the closest match per uploaded source."""
    if not len(xm) or id_col not in xm.columns:
        return xm
    d = _dist_col(xm)
    if d is None:
        return xm.drop_duplicates(subset=[id_col])
    return (xm.sort_values(d).drop_duplicates(subset=[id_col])
            .reset_index(drop=True))


def join_xmatch_photometry(positions: pd.DataFrame,
                           xmatches: dict[str, pd.DataFrame],
                           cfg: dict,
                           phot_radius: dict[str, float] | None = None
                           ) -> pd.DataFrame:
    """Fold X-Match results into the channel photometry schema.

    ``phot_radius`` caps, per catalogue, the separation inside which a match
    counts as *the counterpart* (default: the crossmatch radius).  Gaia is
    pulled wide for the proper-motion test, and a star 40" away is not a
    modern optical counterpart — it is a neighbour.  Catalogue values whose
    error is null are upper limits and go to ``<band>_lim``, never ``<band>``.

    Also derives, from the *unreduced* match tables, the two quantities the
    contamination model needs: the number of infrared sources inside the WISE
    PSF and the local infrared source density; and, from the wide Gaia pull,
    the brightest star nearby (a halo / diffraction-spike veto).
    """
    out = positions.copy()
    v = cfg.get("vet", {})
    psf = float(v.get("wise_psf_arcsec", 6.0))
    r_match = float(cfg.get("crossmatch", {}).get("radius_arcsec", 5.0))
    phot_radius = phot_radius or {}

    for name, xm in xmatches.items():
        if xm is None or not len(xm) or "source_id" not in xm.columns:
            continue
        best = nearest_per_source(xm)
        d = _dist_col(best)
        r_cat = float(phot_radius.get(name, r_match))
        if d is not None:
            best = best[pd.to_numeric(best[d], errors="coerce") <= r_cat]
        if not len(best):
            continue
        cols = {}
        for target, options in _COLMAP.get(name, {}).items():
            src = _pick(best, options)
            if src is not None:
                cols[target] = src
        if not cols:
            continue
        sub = best[["source_id", *cols.values()]].rename(
            columns={v_: k for k, v_ in cols.items()})
        sub = _limits_from_null_errors(sub, name)
        if d is not None:
            sub[f"{name}_sep_arcsec"] = best[d].to_numpy()
        # Do not overwrite a column an earlier (deeper) catalogue supplied.
        dup = [c for c in sub.columns if c != "source_id" and c in out.columns]
        sub = sub.drop(columns=dup)
        out = out.merge(sub, on="source_id", how="left")

    # Infrared crowding and local density, measured from the match tables.
    ir_tables = [xmatches[k] for k in ("allwise", "catwise")
                 if k in xmatches and xmatches[k] is not None
                 and len(xmatches[k]) and "source_id" in xmatches[k].columns]
    if ir_tables:
        xm = max(ir_tables, key=len)
        d = _dist_col(xm)
        inside = xm if d is None else xm[pd.to_numeric(xm[d], errors="coerce") <= psf]
        n = inside.groupby("source_id").size().rename("n_ir_neighbours")
        out = out.merge(n, on="source_id", how="left")
        out["n_ir_neighbours"] = out["n_ir_neighbours"].fillna(0).astype(int)
        area = math.pi * (max(psf, r_match) / 3600.0) ** 2
        out["ir_local_density_per_deg2"] = out["n_ir_neighbours"] / area

    # Brightest Gaia star in the wide pull: plate halos and diffraction spikes.
    g = xmatches.get("gaia")
    if g is not None and len(g) and "source_id" in g.columns:
        gm = _pick(g, ("gmag",))
        d = _dist_col(g)
        if gm is not None:
            gg = g[["source_id", gm] + ([d] if d else [])].copy()
            gg[gm] = pd.to_numeric(gg[gm], errors="coerce")
            gg = gg.dropna(subset=[gm]).sort_values(gm).drop_duplicates("source_id")
            gg = gg.rename(columns={gm: "bright_nb_gmag"})
            if d:
                gg = gg.rename(columns={d: "bright_nb_sep_arcsec"})
            out = out.merge(gg, on="source_id", how="left")
    return out


def modern_optical_depth(positions: pd.DataFrame, provs: dict, cfg: dict
                         ) -> tuple[pd.Series, pd.Series]:
    """Per source: the deepest modern-optical limit actually ESTABLISHED there.

    ``(depth_mag, catalogues)``.  A catalogue contributes only if its X-Match
    really answered (``ok``/``cached``) **and** the source lies inside that
    survey's footprint --- Pan-STARRS stops at dec = -30, so a southern source
    has only Gaia behind its "absence", three magnitudes shallower.

    This exists because absence is the whole signature, and an absence is only
    as good as the search that failed to find it.  Without this, a Pan-STARRS
    X-Match that simply errored would hand every source in the run an empty
    ``ps1_r`` --- and empty reads as *gone*.  A catalogue that was never
    successfully queried has established nothing, so its sources get NaN here
    and are excluded from the disappearance count rather than counted as
    disappearances.
    """
    mo = cfg.get("modern_optical", {})
    limits = mo.get("limits", {})
    dec = pd.to_numeric(positions.get("dec_deg"), errors="coerce")
    depth = pd.Series(np.nan, index=positions.index, dtype=float)
    names = pd.Series("", index=positions.index, dtype=object)
    for name, spec in limits.items():
        st = str((provs.get(name) or {}).get("status", "")).lower()
        if st not in ("ok", "cached"):
            continue
        inside = ((dec >= float(spec.get("dec_min_deg", -90.0)))
                  & (dec <= float(spec.get("dec_max_deg", 90.0))))
        mag = float(spec.get("mag", np.nan))
        if not np.isfinite(mag):
            continue
        better = inside & (depth.isna() | (mag > depth))
        depth = depth.where(~better, mag)
        names = names.where(~inside, names.str.cat(pd.Series(
            [name] * len(names), index=names.index), sep=",").str.strip(","))
    return depth, names


def build_photometry_table(positions: pd.DataFrame, cfg: dict, out_dir: Path
                           ) -> tuple[pd.DataFrame, dict]:
    """Attach POSS-I, modern-optical and infrared photometry to a position list.

    Returns the merged table plus a provenance dict keyed by catalogue.  Any
    catalogue that fails leaves its columns absent and its status recorded —
    never imputed.  Each X-Match result is checkpointed, so a re-run after a
    partial failure only refetches what is missing.
    """
    a = cfg.get("acquire", {})
    cats = a.get("catalogs", {})
    r_match = float(cfg.get("crossmatch", {}).get("radius_arcsec", 5.0))
    r_gaia = float(cfg.get("proper_motion", {}).get("gaia_search_radius_arcsec", 60.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    provs: dict[str, dict] = {}
    xmatches: dict[str, pd.DataFrame] = {}
    have_plate = "poss1_e" in positions.columns and positions["poss1_e"].notna().all()

    plan = [
        # AllWISE first: it is the only catalogue with W3/W4, which the
        # published NeoWISE-matched table lacks and the energy budget needs.
        ("allwise", cats.get("allwise", "vizier:II/328/allwise"), r_match),
        ("catwise", cats.get("catwise", "vizier:II/365/catwise"), r_match),
        ("twomass", cats.get("twomass", "vizier:II/246/out"), r_match),
        ("ps1", cats.get("ps1_dr1", "vizier:II/349/ps1"), r_match),
        # Gaia is pulled WIDE so proper-motion runaways are found: a 200 mas/yr
        # star sits 12.6" away 63 years later and a 5" cone would miss it.
        ("gaia", cats.get("gaia_dr3", "vizier:I/355/gaiadr3"), max(r_match, r_gaia)),
    ]
    if not have_plate:
        plan.append(("usnob1", cats.get("usnob1", "vizier:I/284/out"), r_match))
    for name, cat, radius in plan:
        ck = out_dir / f"xmatch_{name}.parquet"
        if ck.exists():
            df = pd.read_parquet(ck)
            provs[name] = {"route": f"xmatch:{cat}", "status": "cached",
                           "n_rows": int(len(df)), "radius_arcsec": radius}
        else:
            df, prov = xmatch_upload(positions, cat, radius, cfg)
            provs[name] = {**prov.as_dict(), "radius_arcsec": radius}
            if len(df):
                df.to_parquet(ck, index=False)
        xmatches[name] = df
    merged = join_xmatch_photometry(positions, xmatches, cfg,
                                    phot_radius={"gaia": r_match})
    # How deep the search that found nothing actually went, per source.  An
    # absence is only as good as the search behind it.
    depth, cats = modern_optical_depth(merged, provs, cfg)
    merged["modern_depth_mag"] = depth.to_numpy()
    merged["modern_depth_cats"] = cats.to_numpy()
    plate = pd.to_numeric(merged.get("poss1_e"), errors="coerce")
    if "poss1_o" in merged.columns:
        plate = plate.fillna(pd.to_numeric(merged["poss1_o"], errors="coerce"))
    merged["modern_depth_margin_mag"] = (depth.to_numpy()
                                         - plate.to_numpy())
    provs["_modern_optical_depth"] = {
        "route": "derived", "status": "ok",
        "n_rows": int(np.isfinite(depth.to_numpy()).sum()),
        "catalogues_that_answered": sorted(
            n for n in cfg.get("modern_optical", {}).get("limits", {})
            if str((provs.get(n) or {}).get("status", "")).lower() in ("ok", "cached")),
        "n_sources_with_no_modern_coverage": int((~np.isfinite(depth.to_numpy())).sum()),
    }
    (out_dir / "acquire_provenance.json").write_text(
        json.dumps(provs, indent=2, default=str))
    return merged, provs


# --- orchestration ----------------------------------------------------------
def acquire_sample(cfg: dict, out_dir: Path, allow_network: bool = True,
                   n_fields: int | None = None, field_radius_deg: float | None = None,
                   field_seed: int | None = None, deadline_s: float | None = None
                   ) -> tuple[pd.DataFrame, dict]:
    """Get the SHROUD working sample by every route that actually works.

    The returned provenance always carries a ``verdict``:
    ``VO_ARCHIVE`` / ``VO_ARCHIVE_PARTIAL`` / ``VIZIER_SOLANO_TABLE`` /
    ``USNOB1_RECONSTRUCTION`` / ``VIZIER_FALLBACK`` / ``LOCAL_CACHE`` /
    ``NO_DATA_REACHED``, plus ``per_sample_rows`` so the report can say which
    rows came from where.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prov: dict = {"verdict": "NO_DATA_REACHED", "routes": []}
    t_start = time.time()

    if not allow_network:
        prov["routes"].append({"route": "network_disabled", "status": "skipped"})
        cached = out_dir / "sample_positions.parquet"
        if cached.exists():
            df = pd.read_parquet(cached)
            prov["verdict"] = "LOCAL_CACHE"
            prov["n_rows"] = len(df)
            return df, prov
        prov["note"] = ("network disabled and no cached sample on disk; "
                        "no rows were invented")
        return pd.DataFrame(), prov

    frames: list[pd.DataFrame] = []
    got: dict[str, int] = {}
    routes_ok: list[str] = []

    # Route 0: the IVOA registry — where the service says it lives, rather
    # than where this channel guessed it lived.
    reg_roots: list[str] = []
    if cfg.get("acquire", {}).get("try_registry", True):
        reg_roots, p_reg = discover_registry_services(cfg)
        prov["routes"].append(p_reg.as_dict())
        prov["registry_roots"] = reg_roots[:40]

    # Route 1: the published SVO catalogues (short probes; long fetch only if live).
    if cfg.get("acquire", {}).get("try_svo", True):
        for cat, sample in (("vanish_neowise", "solano2022_ir_present"),
                            ("vanish_possi", "solano2022_no_counterpart")):
            roots, p_disc = discover_vo_archive(cfg, cat, out_dir)
            prov["routes"].append(p_disc.as_dict())
            # A registry-published root is tried FIRST: it is the only one of
            # these that any service actually claims to be at.
            roots = list(dict.fromkeys(reg_roots + list(roots)))
            root, _, p_probe = probe_svo_catalog(cat, roots, cfg)
            prov["routes"].append(p_probe.as_dict())
            if not root:
                continue
            df_c, p_fetch = fetch_svo_catalog(cat, root, cfg, out_dir)
            prov["routes"].append(p_fetch.as_dict())
            if len(df_c):
                df_c = normalise_vo_frame(df_c, sample=sample)
                got[sample] = len(df_c)
                frames.append(df_c)
        if got:
            routes_ok.append("VO_ARCHIVE" if len(got) == 2 else "VO_ARCHIVE_PARTIAL")

    # Route 2: VizieR — TAP_SCHEMA discovery, a Solano table if one exists,
    # and the Villarroel+2020 candidates regardless.
    tables, p_tap = discover_vizier_tables(cfg)
    prov["routes"].append(p_tap.as_dict())
    prov["vizier_tables_discovered"] = tables.to_dict("records") if len(tables) else []
    # Look the literal ids up by NAME as well: a keyword sweep can only find a
    # catalogue whose description happens to carry the keyword.
    named: list[str] = []
    for cat in cfg.get("acquire", {}).get("vizier_direct_catalogues", []):
        found, p_m = vizier_catalogue_meta(str(cat), cfg)
        prov["routes"].append(p_m.as_dict())
        named.extend(found)
    candidates = list(dict.fromkeys(_solano_candidates(tables) + named))
    prov["vizier_direct_tables_found"] = named
    for t in candidates[:8]:
        df_t, p_t = fetch_vizier_table_asu(t, cfg, out_dir)
        prov["routes"].append(p_t.as_dict())
        if len(df_t):
            df_t = normalise_vo_frame(df_t, sample=f"vizier:{t}")
            if len(df_t):
                got[f"vizier:{t}"] = len(df_t)
                frames.append(df_t)
                routes_ok.append("VIZIER_SOLANO_TABLE")
    df_v, p_viz = fetch_vizier_vasco2020(cfg, out_dir)
    prov["routes"].append(p_viz.as_dict())
    if len(df_v):
        got["vasco2020_surviving_candidates"] = len(df_v)
        frames.append(df_v)

    # Route 2b: is a SECOND, independent digitisation of the same POSS-I glass
    # reachable?  Reported, never assumed: the plate-confirmation kill is the
    # strongest one this channel could have and it may only be applied on a
    # route that has actually answered.
    if cfg.get("acquire", {}).get("second_digitisation", {}).get("probe", True):
        prov["routes"].append(probe_second_digitisation(cfg).as_dict())

    # Route 3: own the selection function — USNO-B1.0 POSS-I-red-only objects.
    if cfg.get("acquire", {}).get("reconstruct", {}).get("enabled", True):
        left = None if deadline_s is None else max(deadline_s - (time.time() - t_start), 60.0)
        df_u, p_u, _ledger = reconstruct_from_usnob1(
            cfg, out_dir, n_fields=n_fields, radius_deg=field_radius_deg,
            seed=field_seed, deadline_s=left)
        prov["routes"].append(p_u.as_dict())
        if len(df_u):
            got["usnob1_poss1_red_only"] = len(df_u)
            frames.append(df_u)
            routes_ok.append("USNOB1_RECONSTRUCTION")

    if not frames:
        prov["note"] = "no archive route returned rows; nothing was fabricated"
        return pd.DataFrame(), prov

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.loc[:, ~df.columns.duplicated()]
    prov["per_sample_rows"] = got
    prov["n_rows"] = int(len(df))
    if routes_ok:
        prov["verdict"] = routes_ok[0]
        prov["routes_with_rows"] = routes_ok
    else:
        prov["verdict"] = "VIZIER_FALLBACK"
    notes = []
    if "VO_ARCHIVE" not in routes_ok:
        notes.append("the Solano+2022 SVO archive did not answer a cone probe")
    if "USNOB1_RECONSTRUCTION" in routes_ok:
        notes.append("the working sample is the USNO-B1.0 POSS-I-red-only "
                     "reconstruction (own selection function, stated fields)")
    if prov["verdict"] == "VIZIER_FALLBACK":
        notes.append("only the Villarroel+2020 surviving-candidate list (127 rows) "
                     "was retrieved; population fractions from it are indicative only")
    prov["note"] = "; ".join(notes)
    df.to_parquet(out_dir / "sample_positions.parquet", index=False)
    return df, prov


def normalise_vo_frame(df: pd.DataFrame, sample: str = "solano2022_vo_archive"
                       ) -> pd.DataFrame:
    """Map whatever the VO archive calls its columns onto the channel schema.

    The ``vanish-neowise`` table carries ``RA_NEOWISE``/``DEC_NEOWISE`` (decimal
    degrees) plus a NeoWISE ``ph_qual``; ``vanish-possi`` carries plain
    positions and an object ID used for the published cutout filenames.  Both
    layouts, and the generic RA/DEC case, are handled.
    """
    out = pd.DataFrame()
    lower = {c.lower(): c for c in df.columns}
    ra_col = next((lower[k] for k in ("ra_neowise", "ra", "raj2000", "ra_icrs",
                                      "_raj2000", "radeg", "ra_deg")
                   if k in lower), None)
    dec_col = next((lower[k] for k in ("dec_neowise", "dec", "de", "dej2000",
                                       "de_icrs", "_dej2000", "dedeg", "dec_deg")
                    if k in lower), None)
    if ra_col is None or dec_col is None:
        ra_col, dec_col = _ra_dec_columns(df)
    if ra_col is None:
        return df
    out["ra_deg"] = pd.to_numeric(df[ra_col], errors="coerce")
    out["dec_deg"] = pd.to_numeric(df[dec_col], errors="coerce")
    for target, options in {
        "poss1_e": ("rmag", "r1mag", "e_plate", "rposs", "mag_r", "emag", "r"),
        "poss1_o": ("bmag", "b1mag", "o_plate", "bposs", "mag_b", "omag"),
        "w1": ("w1mpro", "w1mag", "w1"), "w2": ("w2mpro", "w2mag", "w2"),
        "w3": ("w3mpro", "w3mag", "w3"), "w4": ("w4mpro", "w4mag", "w4"),
        "w1_err": ("w1sigmpro", "e_w1mag"), "w2_err": ("w2sigmpro", "e_w2mag"),
        "2mass_j": ("jmag", "j"), "2mass_h": ("hmag", "h"),
        "2mass_ks": ("kmag", "ksmag", "ks"),
        "epoch_poss1": ("epoch", "obsdate", "mjd", "plate_epoch", "plate_date"),
    }.items():
        for o in options:
            if o in lower:
                out[target] = pd.to_numeric(df[lower[o]], errors="coerce")
                break
    for keep in ("ph_qual", "plate", "objid", "id", "field"):
        if keep in lower:
            out[keep] = df[lower[keep]].astype(str)
    tag = "IR" if "ir_present" in sample else ("NC" if "no_counterpart" in sample
                                               else "VO")
    out["source_id"] = [f"VASCO-{tag}-{i:07d}" for i in range(len(out))]
    out["sample"] = sample
    return out.dropna(subset=["ra_deg", "dec_deg"]).reset_index(drop=True)


__all__ = [
    "Provenance", "acquire_sample", "build_photometry_table",
    "discover_vizier_tables", "discover_vo_archive", "fetch_svo_catalog",
    "fetch_vizier_table_asu", "fetch_vizier_vasco2020", "field_grid", "http_get",
    "join_xmatch_photometry", "nearest_per_source", "normalise_usnob1_frame",
    "normalise_vo_frame", "parse_vizier_vasco2020", "probe_svo_catalog",
    "reconstruct_from_usnob1", "sexagesimal_to_deg", "tap_sync", "unquote_table",
    "usnob1_field_url", "votable_tables", "xmatch_upload",
]

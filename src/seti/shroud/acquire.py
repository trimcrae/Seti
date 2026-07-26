"""SHROUD acquisition — runner only (the sandbox has no archive egress).

Three routes to the sample, tried in order, each recording exactly what it did
so a degraded run reports its degradation as a first-class field:

1. **The Solano+2022 VO archive.**  MNRAS 515, 1380 states that the 5 399
   unidentified transients and the **172 163 sources not detected in the optical
   but identified in the infrared** "are available from a Virtual Observatory
   compliant archive".  The archive is an SVO ``vocats`` service.  Rather than
   trust a guessed URL, :func:`discover_vo_archive` *finds* it: it scrapes the
   SVO vocats index, and independently regex-mines the paper's own full text
   for any ``svo``/``vocats``/``cab.inta-csic.es`` URL.  Every candidate root is
   then probed for a live cone-search response.

2. **VizieR.**  ``J/AJ/159/8`` (Villarroel+2020) is present and holds ``table2``
   (99 rows, the surviving candidates) and ``table3`` (28 rows, the most
   interesting).  ``J/MNRAS/515/1380`` (Solano+2022) is **not** in VizieR —
   verified on the runner 2026-07-26, the server answers "Table or Catalog not
   found".  Do not re-derive that; it is recorded in ``docs/shroud.md``.

3. **Reconstruct the crossmatch.**  USNO-B1.0 POSS-I detections in a sky window,
   crossmatched against AllWISE/CatWISE/2MASS for an IR counterpart and against
   Gaia DR3/Pan-STARRS for modern optical absence.  This is what the channel
   falls back to when neither archive is reachable, and it is the honest way to
   own the selection function rather than inherit it.

Bulk crossmatching of 10^5 positions goes through the **CDS X-Match** service,
which takes an uploaded table; per-object cone searches at this scale are not
viable and would themselves become the systematic.

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

_UA = {"User-Agent": "Seti-SHROUD/1.0 (mailto:trimcrae@gmail.com)"}

# Full text of the paper that published the sample; mined for the archive URL.
_SOLANO2022_URLS = (
    "https://arxiv.org/html/2206.00907v2",
    "https://arxiv.org/html/2206.00907v1",
    "https://arxiv.org/abs/2206.00907",
    "https://academic.oup.com/mnras/article/515/1/1380/6608880",
)
_URL_RE = re.compile(
    r"https?://[\w.\-]*(?:cab\.inta-csic\.es|svo\d?\.[\w.\-]+|vizier[\w.\-]*)"
    r"[\w/\-.~%?=&+#]*", re.I)


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
        try:
            req = urllib.request.Request(url, data=data, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), f"HTTP {r.status}"
        except urllib.error.HTTPError as e:                    # noqa: PERF203
            last = f"HTTP {e.code}"
        except Exception as e:                                 # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if i + 1 < max(retries, 1):
            time.sleep(backoff * (i + 1))
    return None, last


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


# --- route 1: the Solano+2022 VO archive ------------------------------------
def discover_vo_archive(cfg: dict, catalog: str = "vanish_neowise",
                        out_dir: Path | None = None
                        ) -> tuple[list[str], Provenance]:
    """Candidate roots for one VASCO SVO catalogue, configured plus discovered.

    The configured roots are verified (quoted in Watters et al. 2026 and in the
    jannefi/vasco README), so discovery is a *belt-and-braces* extra: it also
    scrapes the SVO archive index and the text of Solano+2022 for any
    ``vanish``/``vocat`` URL, in case the service is relocated.
    """
    a = cfg.get("acquire", {})
    prov = Provenance(route=f"discover_vo_archive:{catalog}")
    roots: list[str] = list(a.get("svo_catalogs", {}).get(catalog, []))
    key = catalog.replace("_", "-")

    for url in list(a.get("svo_index_urls", [])) + list(_SOLANO2022_URLS):
        body, detail = http_get(url, int(a.get("http_timeout_s", 300)),
                                retries=2, backoff=4.0)
        prov.record(url, body is not None, detail, 0)
        if body is None:
            continue
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = re.sub(r"\W+", "_", url)[-90:]
            (out_dir / f"discover_{stem}.html").write_bytes(body[:2_000_000])
        txt = body.decode("utf-8", "replace")
        hits = {u.rstrip(".,);\"'") for u in _URL_RE.findall(txt)}
        # Prefer anything that smells like this catalogue specifically.
        prio = sorted(h for h in hits
                      if re.search(rf"{re.escape(key)}|vanish|vasco|vocat", h, re.I))
        roots.extend(h.rstrip("/") for h in prio)
        if prio:
            prov.notes.append(f"{len(prio)} vanish/vasco URL(s) mined from {url}")

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
    ``(180, 0)``, ``nocoor=1``, ``format=ascii``) — the form that production
    code uses against other svocats catalogues and the only sane way to pull
    1.7x10^5 rows.
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

    Probes with a small cone first (cheap); returns ``(root, working_url_form,
    provenance)``.
    """
    a = cfg.get("acquire", {})
    prov = Provenance(route=f"probe_svo:{name}")
    for root in roots:
        for url in _svo_urls(root, cfg, ra=180.0, dec=0.0, sr=5.0):
            body, detail = http_get(url, int(a.get("http_timeout_s", 300)),
                                    retries=2, backoff=4.0)
            ok = _looks_tabular(body) if body else False
            prov.record(url, ok, detail, 0)
            if ok:
                prov.status, prov.url = "ok", url
                return root, url, prov
        time.sleep(0.5)
    prov.status = "unreachable"
    prov.notes.append(f"no root for '{name}' answered a cone search")
    return None, "", prov


def fetch_svo_catalog(name: str, root: str, cfg: dict, out_dir: Path
                      ) -> tuple[pd.DataFrame, Provenance]:
    """Pull an entire SVO catalogue.

    Tries the single whole-sky dump first; if that is refused or truncated,
    falls back to a declination-band sweep and concatenates.
    """
    a = cfg.get("acquire", {})
    prov = Provenance(route=f"fetch_svo:{name}", url=root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for url in _svo_urls(root, cfg):
        body, detail = http_get(url, int(a.get("http_timeout_s", 300)),
                                retries=int(a.get("retries", 4)),
                                backoff=float(a.get("retry_backoff_s", 8.0)))
        if body is None or not _looks_tabular(body):
            prov.record(url, False, detail, 0)
            continue
        (out_dir / f"svo_{name}_allsky.raw").write_bytes(body)
        df = _votable_to_frame(body) if b"<TABLE" in body[:200000] else _read_csvish(body)
        # A table without usable coordinates cannot be analysed; reject it and
        # try the next query form rather than accepting a useless response.
        has_pos = len(df) > 0 and _ra_dec_columns(df)[0] is not None
        prov.record(url, has_pos, detail if has_pos else
                    f"{detail}; {len(df)} rows but no RA/DEC column "
                    f"(cols: {list(df.columns)[:12]})", len(df))
        if has_pos:
            prov.status, prov.n_rows, prov.url = "ok", len(df), url
            _check_expected_rows(name, len(df), cfg, prov)
            df.to_parquet(out_dir / f"svo_{name}_raw.parquet", index=False)
            return df, prov

    # Fallback: declination-band sweep.
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
def fetch_vizier_vasco2020(cfg: dict, out_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    """Villarroel+2020 J/AJ/159/8 — table2 (~100 candidates) + table3."""
    a = cfg.get("acquire", {})
    prov = Provenance(route="vizier_J/AJ/159/8")
    url = a.get("vizier_votable",
                "https://vizier.cds.unistra.fr/viz-bin/votable?-source={cat}&-out.max={maxrows}"
                ).format(cat=a.get("vizier_vasco2020_cat", "J/AJ/159/8"),
                         maxrows=100000)
    body, detail = http_get(url, int(a.get("http_timeout_s", 300)),
                            retries=int(a.get("retries", 4)),
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


# --- route 3: reconstruct the crossmatch ------------------------------------
def xmatch_upload(positions: pd.DataFrame, catalog: str, radius_arcsec: float,
                  cfg: dict, ra_col: str = "ra_deg", dec_col: str = "dec_deg",
                  ) -> tuple[pd.DataFrame, Provenance]:
    """Crossmatch an uploaded position list against a VizieR catalogue.

    Uses the CDS X-Match service, chunked.  This is the only route that scales
    to the 1.7x10^5 positions of the Solano+2022 infrared sample.  ``source_id``
    is uploaded alongside the coordinates and echoed back by the service, so
    results join unambiguously even when a source has several matches.
    """
    a = cfg.get("acquire", {})
    url = a.get("xmatch_url", "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync")
    chunk = int(a.get("xmatch_chunk_rows", 25000))
    prov = Provenance(route=f"xmatch:{catalog}", url=url)
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
        body, detail = http_get(
            url, int(a.get("http_timeout_s", 300)),
            retries=int(a.get("retries", 4)),
            backoff=float(a.get("retry_backoff_s", 8.0)),
            data=body_parts,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        if body is None:
            prov.record(url, False, detail, 0)
            continue
        try:
            df = pd.read_csv(io.StringIO(body.decode("utf-8", "replace")))
        except Exception as e:                                 # noqa: BLE001
            prov.record(url, False, f"parse: {e}", 0)
            continue
        # X-Match numbers uploaded rows from 1 within each chunk.
        if "ra" in df.columns and "dec" in df.columns:
            df["_chunk_start"] = start
        frames.append(df)
        prov.record(url, True, detail, len(df))
        time.sleep(1.0)
    if not frames:
        prov.status = "unreachable"
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
                "ir_ext_flag": ("ex", "extflg", "ext_flg")},
    "catwise": {"w1": ("w1mpropm", "w1mpro", "w1mag"),
                "w2": ("w2mpropm", "w2mpro", "w2mag"),
                "w1_err": ("e_w1mpropm", "e_w1mpro"),
                "w2_err": ("e_w2mpropm", "e_w2mpro")},
    "twomass": {"2mass_j": ("jmag",), "2mass_h": ("hmag",),
                "2mass_ks": ("kmag", "ksmag"), "2mass_j_err": ("e_jmag",),
                "2mass_h_err": ("e_hmag",), "2mass_ks_err": ("e_kmag",)},
    "ps1": {"ps1_g": ("gmag",), "ps1_r": ("rmag",), "ps1_i": ("imag",),
            "ps1_z": ("zmag",), "ps1_y": ("ymag",)},
    "gaia": {"gaia_g": ("gmag",), "gaia_bp": ("bpmag",), "gaia_rp": ("rpmag",),
             "pmra": ("pmra",), "pmdec": ("pmde", "pmdec"),
             "parallax": ("plx", "parallax")},
    "usnob1": {"poss1_o": ("b1mag",), "poss1_e": ("r1mag",)},
}
_DIST_COLS = ("angDist", "angdist", "ang_dist", "_r")


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
                           cfg: dict) -> pd.DataFrame:
    """Fold X-Match results into the channel photometry schema.

    Also derives, from the *unreduced* match tables, the two quantities the
    contamination model needs: the number of infrared sources inside the WISE
    PSF and the local infrared source density.  Offline-testable: it takes
    frames, not URLs.
    """
    out = positions.copy()
    psf = float(cfg.get("vet", {}).get("wise_psf_arcsec", 6.0))
    r_match = float(cfg.get("crossmatch", {}).get("radius_arcsec", 5.0))

    for name, xm in xmatches.items():
        if xm is None or not len(xm) or "source_id" not in xm.columns:
            continue
        best = nearest_per_source(xm)
        cols = {}
        for target, options in _COLMAP.get(name, {}).items():
            src = _pick(best, options)
            if src is not None:
                cols[target] = src
        if not cols:
            continue
        sub = best[["source_id", *cols.values()]].rename(
            columns={v: k for k, v in cols.items()})
        d = _dist_col(best)
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
        inside = xm if d is None else xm[xm[d].astype(float) <= psf]
        n = inside.groupby("source_id").size().rename("n_ir_neighbours")
        out = out.merge(n, on="source_id", how="left")
        out["n_ir_neighbours"] = out["n_ir_neighbours"].fillna(0).astype(int)
        area = math.pi * (max(psf, r_match) / 3600.0) ** 2
        out["ir_local_density_per_deg2"] = out["n_ir_neighbours"] / area
    return out


def build_photometry_table(positions: pd.DataFrame, cfg: dict, out_dir: Path
                           ) -> tuple[pd.DataFrame, dict]:
    """Attach POSS-I, modern-optical and infrared photometry to a position list.

    Returns the merged table plus a provenance dict keyed by catalogue.  Any
    catalogue that fails leaves its columns absent and its status recorded —
    never imputed.
    """
    a = cfg.get("acquire", {})
    cats = a.get("catalogs", {})
    r_match = float(cfg.get("crossmatch", {}).get("radius_arcsec", 5.0))
    r_pm = float(cfg.get("proper_motion", {}).get("mu_max_mas_yr", 3000.0)) \
        * abs(cfg.get("epochs", {}).get("gaia_dr3", 2016.0)
              - cfg.get("epochs", {}).get("poss1_default", 1953.0)) / 1000.0

    out_dir.mkdir(parents=True, exist_ok=True)
    provs: dict[str, dict] = {}
    xmatches: dict[str, pd.DataFrame] = {}

    plan = [
        # AllWISE first: it is the only catalogue with W3/W4, which the
        # published NeoWISE-matched table lacks and the energy budget needs.
        ("allwise", cats.get("allwise", "vizier:II/328/allwise"), r_match),
        ("catwise", cats.get("catwise", "vizier:II/365/catwise"), r_match),
        ("twomass", cats.get("twomass", "vizier:II/246/out"), r_match),
        ("ps1", cats.get("ps1_dr1", "vizier:II/349/ps1"), r_match),
        ("usnob1", cats.get("usnob1", "vizier:I/284/out"), r_match),
        # Gaia is pulled with a WIDE radius so proper-motion runaways are found:
        # a 200 mas/yr star sits 12.6" away 63 years later and a 5" cone would
        # miss it entirely.
        ("gaia", cats.get("gaia_dr3", "vizier:I/355/gaiadr3"),
         max(r_match, min(r_pm, 180.0))),
    ]
    for name, cat, radius in plan:
        df, prov = xmatch_upload(positions, cat, radius, cfg)
        provs[name] = prov.as_dict()
        xmatches[name] = df
        if len(df):
            df.to_parquet(out_dir / f"xmatch_{name}.parquet", index=False)
    merged = join_xmatch_photometry(positions, xmatches, cfg)
    (out_dir / "acquire_provenance.json").write_text(
        json.dumps(provs, indent=2, default=str))
    return merged, provs


# --- orchestration ----------------------------------------------------------
def acquire_sample(cfg: dict, out_dir: Path, allow_network: bool = True
                   ) -> tuple[pd.DataFrame, dict]:
    """Get the SHROUD working sample by the best route that actually works.

    The returned provenance always carries a ``verdict``:
    ``VO_ARCHIVE`` / ``VIZIER_FALLBACK`` / ``LOCAL_CACHE`` / ``NO_DATA_REACHED``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prov: dict = {"verdict": "NO_DATA_REACHED", "routes": []}

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

    # Both published catalogues: the infrared sample AND its no-counterpart
    # control.  The channel exists to compare them, so a run that gets only one
    # says so rather than quietly analysing half the measurement.
    frames: list[pd.DataFrame] = []
    got: dict[str, int] = {}
    for cat, sample in (("vanish_neowise", "solano2022_ir_present"),
                        ("vanish_possi", "solano2022_no_counterpart")):
        roots, p_disc = discover_vo_archive(cfg, cat, out_dir)
        prov["routes"].append(p_disc.as_dict())
        root, _, p_probe = probe_svo_catalog(cat, roots, cfg)
        prov["routes"].append(p_probe.as_dict())
        if not root:
            continue
        df_c, p_fetch = fetch_svo_catalog(cat, root, cfg, out_dir)
        prov["routes"].append(p_fetch.as_dict())
        if len(df_c):
            df_c = normalise_vo_frame(df_c, sample=sample)
            got[cat] = len(df_c)
            frames.append(df_c)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        prov["verdict"] = ("VO_ARCHIVE" if len(got) == 2 else "VO_ARCHIVE_PARTIAL")
        prov["n_rows"] = len(df)
        prov["per_catalog_rows"] = got
        if len(got) < 2:
            prov["note"] = ("only one of the two published catalogues was "
                            "retrieved; the obscuration-vs-destruction ratio "
                            "needs both and is reported as unavailable")
        df.to_parquet(out_dir / "sample_positions.parquet", index=False)
        return df, prov

    df, p_viz = fetch_vizier_vasco2020(cfg, out_dir)
    prov["routes"].append(p_viz.as_dict())
    if len(df):
        prov["verdict"] = "VIZIER_FALLBACK"
        prov["n_rows"] = len(df)
        prov["note"] = ("the Solano+2022 VO archive did not answer; the sample "
                        "is the Villarroel+2020 surviving-candidate list, which "
                        "is ~3 orders of magnitude smaller. Population "
                        "fractions from it are indicative only.")
        df.to_parquet(out_dir / "sample_positions.parquet", index=False)
        return df, prov

    prov["note"] = "no archive route returned rows; nothing was fabricated"
    return pd.DataFrame(), prov


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
    "discover_vo_archive", "fetch_svo_catalog", "fetch_vizier_vasco2020",
    "http_get", "join_xmatch_photometry", "nearest_per_source",
    "normalise_vo_frame", "parse_vizier_vasco2020", "probe_svo_catalog",
    "sexagesimal_to_deg", "votable_tables", "xmatch_upload",
]

"""Runner-only archive access for GROWTH, plus the (pure) Kepler <-> TESS join.

The sandbox has no archive egress, so nothing network-bound here runs in a
test; every fetch takes an injectable ``query_fn`` / ``gaia_fn`` so a failed or
empty archive can be simulated offline, and the join is a pure function over
three DataFrames.

Endpoints
---------
NASA Exoplanet Archive TAP **sync** endpoint (reached from the runner by the
necrofrontier probe, entries ``exoarchive_koi_depth`` / ``exoarchive_toi_depth``
/ ``exoarchive_tess_kepler_xmatch``):

``https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=<ADQL>&format=csv``

* ``cumulative`` --- the KOI cumulative table: ``koi_depth`` [ppm], ``koi_ror``,
  ``koi_impact``, ``koi_duration`` [h], ``koi_dor`` (a/R*), the four
  ``koi_fpflag_*`` and the dispositions.  The 2009--2013 epoch.
* ``toi`` --- the TESS Objects of Interest table: ``pl_trandep`` [ppm],
  ``pl_trandurh`` [h], ``tid`` (TIC id), ``tfopwg_disp``.  The 2018--2026 epoch.
* ``ps`` --- Planetary Systems: ``pl_name`` <-> ``hostname`` <-> ``tic_id`` for
  rows with ``disc_facility LIKE '%Kepler%'``; ``pl_trandep`` there is in
  **percent**, converted to ppm on read.

Gaia DR3 neighbours (dilution term) come from the Gaia TAP through an
upload-table crossmatch (chunked, retried); a chunk that fails leaves its
targets ``neighbours_status = "not_checked"`` rather than pretending they are
isolated.

Two facts are never collapsed: ``QUERY_FAILED`` (the service did not answer or
errored) and ``QUERY_RETURNED_ZERO_ROWS`` (it answered, with nothing).
"""

from __future__ import annotations

import io
import json
import time as _time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

EXOARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
USER_AGENT = "Seti-growth/1.0 (mailto:trimcrae@gmail.com)"

STATUS_OK = "OK"
STATUS_FAILED = "QUERY_FAILED"
STATUS_ZERO = "QUERY_RETURNED_ZERO_ROWS"

#: Columns pulled from each table.  A name the server rejects 400s the whole
#: query, so the lists are the documented column names as of the 2026-09 probe.
KOI_COLUMNS: tuple[str, ...] = (
    "kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition",
    "koi_period", "koi_period_err1", "koi_depth", "koi_depth_err1", "koi_depth_err2",
    "koi_ror", "koi_ror_err1", "koi_ror_err2", "koi_impact", "koi_impact_err1",
    "koi_impact_err2", "koi_duration", "koi_duration_err1", "koi_duration_err2",
    "koi_ingress", "koi_dor", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
    "ra", "dec", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_tce_delivname", "koi_count",
)
TOI_COLUMNS: tuple[str, ...] = (
    "toi", "tid", "tfopwg_disp", "pl_orbper", "pl_trandep", "pl_trandeperr1",
    "pl_trandeperr2", "pl_trandurh", "pl_trandurherr1", "pl_trandurherr2", "pl_rade",
    "st_teff", "st_logg", "st_rad", "st_tmag", "ra", "dec",
)
PS_COLUMNS: tuple[str, ...] = (
    "pl_name", "hostname", "tic_id", "default_flag", "pl_ratror", "pl_trandep",
    "pl_orbper", "disc_facility", "pl_refname", "ra", "dec",
)


# ---------------------------------------------------------------------------
# Acquisition log (same discipline as seti.metronome.acquire.AcquisitionLog)
# ---------------------------------------------------------------------------
@dataclass
class AcquisitionLog:
    stages: list[dict] = field(default_factory=list)
    prefix: str = "growth/acquire"

    def record(self, stage: str, query: str, *, rows: int | None = None,
               error: str | None = None, extra: dict | None = None) -> None:
        if error is not None or rows is None:
            status = STATUS_FAILED
        elif rows == 0:
            status = STATUS_ZERO
        else:
            status = STATUS_OK
        rec = {"stage": stage, "status": status, "rows": int(rows or 0),
               "query": str(query)[:2000]}
        if error:
            rec["error"] = str(error)[:500]
        if extra:
            rec.update(extra)
        self.stages.append(rec)
        print(f"[{self.prefix}] {stage}: {status} rows={rec['rows']}"
              + (f" error={rec.get('error')}" if error else ""))

    def as_dict(self) -> dict:
        n_fail = sum(1 for s in self.stages if s["status"] == STATUS_FAILED)
        n_zero = sum(1 for s in self.stages if s["status"] == STATUS_ZERO)
        n_ok = sum(1 for s in self.stages if s["status"] == STATUS_OK)
        return {"stages": self.stages, "n_stages": len(self.stages), "n_ok": n_ok,
                "n_query_failed": n_fail, "n_query_returned_zero_rows": n_zero,
                "any_query_failed": bool(n_fail > 0),
                "total_rows": int(sum(s["rows"] for s in self.stages))}

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))


# ---------------------------------------------------------------------------
# TAP primitives
# ---------------------------------------------------------------------------
def _retry(fn, retries: int = 3, label: str = "query", base_sleep: float = 4.0):
    last = None
    for attempt in range(int(retries)):
        try:
            return fn()
        except Exception as exc:                          # noqa: BLE001
            last = exc
            print(f"[growth/acquire] {label} attempt {attempt + 1}/{retries} failed: {exc!r}")
            if attempt + 1 < retries:
                _time.sleep(base_sleep * (attempt + 1))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


def _default_transport(url: str, timeout: float = 600.0) -> str:
    import requests  # noqa: PLC0415  runner-only

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def build_url(adql: str, *, url: str = EXOARCHIVE_TAP, fmt: str = "csv") -> str:
    q = urllib.parse.urlencode({"query": adql, "format": fmt}, quote_via=urllib.parse.quote)
    return f"{url}?{q}"


def parse_tap_csv(text: str) -> pd.DataFrame:
    """CSV body -> DataFrame; an archive error page raises instead of parsing."""
    head = (text or "").lstrip()[:400]
    if not head:
        return pd.DataFrame()
    if head.startswith("ERROR") or head.lower().startswith("<!doctype") or head.startswith("<html"):
        raise RuntimeError(f"archive returned an error body: {head[:200]!r}")
    return pd.read_csv(io.StringIO(text))


def tap_sync(adql: str, *, url: str = EXOARCHIVE_TAP, retries: int = 3,
             transport=None, timeout: float = 600.0) -> pd.DataFrame:
    """One ADQL query against the Exoplanet Archive sync endpoint, retried."""
    transport = transport or _default_transport
    full = build_url(adql, url=url)
    return _retry(lambda: parse_tap_csv(transport(full, timeout)), retries=retries,
                  label="exoarchive TAP")


def table_query(table: str, columns, where: str | None = None) -> str:
    cols = ",".join(columns)
    q = f"select {cols} from {table}"
    if where:
        q += f" where {where}"
    return q


# ---------------------------------------------------------------------------
# The three tables
# ---------------------------------------------------------------------------
def _fetch(name: str, adql: str, *, query_fn, log: AcquisitionLog) -> tuple[pd.DataFrame, str]:
    try:
        df = query_fn(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record(f"fetch_{name}", adql, error=repr(exc))
        return pd.DataFrame(), STATUS_FAILED
    if df is None:
        log.record(f"fetch_{name}", adql, error="query_fn returned None")
        return pd.DataFrame(), STATUS_FAILED
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    log.record(f"fetch_{name}", adql, rows=int(len(df)))
    return df, (STATUS_OK if len(df) else STATUS_ZERO)


def fetch_koi(*, query_fn=None, log: AcquisitionLog | None = None,
              columns=KOI_COLUMNS) -> tuple[pd.DataFrame, str]:
    log = log or AcquisitionLog()
    query_fn = query_fn or tap_sync
    return _fetch("koi_cumulative", table_query("cumulative", columns), query_fn=query_fn, log=log)


def fetch_toi(*, query_fn=None, log: AcquisitionLog | None = None,
              columns=TOI_COLUMNS) -> tuple[pd.DataFrame, str]:
    log = log or AcquisitionLog()
    query_fn = query_fn or tap_sync
    return _fetch("toi", table_query("toi", columns), query_fn=query_fn, log=log)


def fetch_ps_kepler(*, query_fn=None, log: AcquisitionLog | None = None,
                    columns=PS_COLUMNS) -> tuple[pd.DataFrame, str]:
    """``ps`` rows for Kepler-discovered planets; ``pl_trandep`` % -> ppm."""
    log = log or AcquisitionLog()
    query_fn = query_fn or tap_sync
    df, status = _fetch("ps_kepler", table_query("ps", columns, "disc_facility like '%Kepler%'"),
                        query_fn=query_fn, log=log)
    if len(df) and "pl_trandep" in df:
        # The Planetary Systems table quotes transit depth in PERCENT; the KOI
        # and TOI tables quote ppm.  Convert here, once, and say so in the column.
        df["pl_trandep_ppm"] = pd.to_numeric(df["pl_trandep"], errors="coerce") * 1.0e4
    return df, status


# ---------------------------------------------------------------------------
# The join (pure)
# ---------------------------------------------------------------------------
def _norm_name(s) -> str:
    return " ".join(str(s).split()).strip().lower() if s is not None and s == s else ""


def period_match(p_kepler: float, p_tess: float, *, tol: float = 1e-3,
                 alias_max: int = 4) -> tuple[bool, float, str]:
    """Same period within ``tol`` or an integer alias ``n`` / ``1/n`` (n <= alias_max).

    Returns ``(matched, ratio, alias)`` with ``alias`` in ``{"1", "n", "1/n"}``
    formatted as the integer, e.g. ``"2"`` when TESS found twice the Kepler
    period (alternate transits) or ``"1/2"`` when it found half of it.
    """
    if not (np.isfinite(p_kepler) and np.isfinite(p_tess)) or p_kepler <= 0 or p_tess <= 0:
        return False, float("nan"), ""
    r = float(p_tess / p_kepler)
    if abs(r - 1.0) < tol:
        return True, r, "1"
    for n in range(2, int(alias_max) + 1):
        if abs(r - n) < n * tol:
            return True, r, str(n)
        if abs(r - 1.0 / n) < tol / n:
            return True, r, f"1/{n}"
    return False, r, ""


def _unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    v1 = _unit_vectors([ra1], [dec1])[0]
    v2 = _unit_vectors([ra2], [dec2])[0]
    d = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)) * 3600.0)


def join_kepler_tess(koi: pd.DataFrame, toi: pd.DataFrame, ps: pd.DataFrame | None = None, *,
                     pos_radius_arcsec: float = 2.0, period_tol: float = 1e-3,
                     alias_max: int = 4) -> tuple[pd.DataFrame, dict]:
    """Every KOI that a TOI re-detects, with the route that joined it.

    Route ``tic_id``: ``cumulative.kepler_name == ps.pl_name`` gives the TIC id,
    ``toi.tid == tic_id`` gives the TOIs on that star, and the period picks the
    planet.  Route ``position_period``: for KOIs the ``ps`` table cannot key
    (candidates without a Kepler name, or no ``tic_id`` row), the nearest TOI
    within ``pos_radius_arcsec`` whose period matches (or is an integer alias).
    A TOI is used at most once; the counts by route and the reasons a star
    matched without a planet are returned in the report.
    """
    rep: dict = {"n_koi": int(len(koi)), "n_toi": int(len(toi)),
                 "n_ps_kepler_rows": int(len(ps)) if ps is not None else 0,
                 "joined_by_route": {"tic_id": 0, "position_period": 0},
                 "tic_matched_no_period_match": 0, "koi_with_tic_id": 0,
                 "position_matched_no_period_match": 0}
    if not len(koi) or not len(toi):
        return pd.DataFrame(), rep

    koi = koi.copy()
    toi = toi.copy()
    for c in ("koi_period", "ra", "dec"):
        koi[c] = pd.to_numeric(koi.get(c), errors="coerce")
    for c in ("pl_orbper", "ra", "dec", "tid"):
        toi[c] = pd.to_numeric(toi.get(c), errors="coerce")

    # --- kepler_name -> tic_id through the ps table -------------------------
    name_to_tic: dict[str, int] = {}
    ps_default: dict[str, dict] = {}
    if ps is not None and len(ps) and "pl_name" in ps and "tic_id" in ps:
        p = ps.copy()
        p["tic_id"] = pd.to_numeric(p["tic_id"], errors="coerce")
        p["_n"] = p["pl_name"].map(_norm_name)
        p = p.dropna(subset=["tic_id"])
        # default_flag row first so the reference ratio/depth come from it
        if "default_flag" in p:
            p = p.sort_values("default_flag", ascending=False)
        for _, r in p.iterrows():
            if r["_n"] and r["_n"] not in name_to_tic:
                name_to_tic[r["_n"]] = int(r["tic_id"])
                ps_default[r["_n"]] = {
                    "ps_pl_ratror": pd.to_numeric(r.get("pl_ratror"), errors="coerce"),
                    "ps_pl_trandep_ppm": pd.to_numeric(r.get("pl_trandep_ppm"), errors="coerce"),
                    "ps_refname": r.get("pl_refname")}
    koi["_n"] = koi.get("kepler_name", pd.Series([None] * len(koi))).map(_norm_name)
    koi["tic_id_ps"] = koi["_n"].map(name_to_tic)
    rep["koi_with_tic_id"] = int(koi["tic_id_ps"].notna().sum())

    toi_by_tid: dict[int, list[int]] = {}
    for i, t in toi["tid"].items():
        if np.isfinite(t):
            toi_by_tid.setdefault(int(t), []).append(i)

    used_toi: set[int] = set()
    rows: list[dict] = []

    def _emit(ki, ti, route, ratio, alias, sep):
        # Only ``ra``/``dec`` collide between the two tables; every other KOI
        # column is ``koi_*`` / ``kepid`` / ``kepoi_name`` / ``kepler_name`` and
        # every TOI column is ``toi`` / ``tid`` / ``tfopwg_disp`` / ``pl_*`` / ``st_*``.
        k, t = koi.loc[ki], toi.loc[ti]
        rec = {("ra_kepler" if c == "ra" else "dec_kepler" if c == "dec" else c): k[c]
               for c in koi.columns if c != "_n"}
        rec.update({("ra_tess" if c == "ra" else "dec_tess" if c == "dec" else c): t[c]
                    for c in toi.columns})
        rec.update({"join_route": route, "period_ratio": ratio, "period_alias": alias,
                    "join_sep_arcsec": sep,
                    "tic_id": k.get("tic_id_ps") if route == "tic_id" else t.get("tid")})
        rec.update(ps_default.get(k["_n"], {}))
        rows.append(rec)
        used_toi.add(ti)

    # --- route 1: tic_id ------------------------------------------------------
    joined_koi: set[int] = set()
    for ki, k in koi.iterrows():
        tic = k["tic_id_ps"]
        if not np.isfinite(tic):
            continue
        cands = [ti for ti in toi_by_tid.get(int(tic), []) if ti not in used_toi]
        if not cands:
            continue
        best = None
        for ti in cands:
            ok, ratio, alias = period_match(float(k["koi_period"]), float(toi.loc[ti, "pl_orbper"]),
                                            tol=period_tol, alias_max=alias_max)
            if ok and (best is None or alias == "1"):
                best = (ti, ratio, alias)
                if alias == "1":
                    break
        if best is None:
            rep["tic_matched_no_period_match"] += 1
            continue
        ti, ratio, alias = best
        _emit(ki, ti, "tic_id", ratio, alias,
              _sep_arcsec(k["ra"], k["dec"], toi.loc[ti, "ra"], toi.loc[ti, "dec"]))
        joined_koi.add(ki)
        rep["joined_by_route"]["tic_id"] += 1

    # --- route 2: position + period -----------------------------------------
    rest = koi.loc[[i for i in koi.index if i not in joined_koi]]
    rest = rest.dropna(subset=["ra", "dec"])
    tpos = toi.dropna(subset=["ra", "dec"])
    if len(rest) and len(tpos):
        from scipy.spatial import cKDTree  # noqa: PLC0415

        tree = cKDTree(_unit_vectors(tpos["ra"], tpos["dec"]))
        chord = 2.0 * np.sin(np.deg2rad(pos_radius_arcsec / 3600.0) / 2.0)
        hits = tree.query_ball_point(_unit_vectors(rest["ra"], rest["dec"]), r=chord)
        for (ki, k), idxs in zip(rest.iterrows(), hits, strict=True):
            if not idxs:
                continue
            best = None
            for j in idxs:
                ti = tpos.index[j]
                if ti in used_toi:
                    continue
                ok, ratio, alias = period_match(float(k["koi_period"]),
                                                float(toi.loc[ti, "pl_orbper"]),
                                                tol=period_tol, alias_max=alias_max)
                if ok and (best is None or alias == "1"):
                    best = (ti, ratio, alias)
                    if alias == "1":
                        break
            if best is None:
                rep["position_matched_no_period_match"] += 1
                continue
            ti, ratio, alias = best
            _emit(ki, ti, "position_period", ratio, alias,
                  _sep_arcsec(k["ra"], k["dec"], toi.loc[ti, "ra"], toi.loc[ti, "dec"]))
            rep["joined_by_route"]["position_period"] += 1

    out = pd.DataFrame(rows)
    if len(out):
        out["planet_key"] = out["kepoi_name"].astype(str)
        # The Kepler position is the target position for the Gaia cones.
        out["ra"] = out["ra_kepler"]
        out["dec"] = out["dec_kepler"]
    rep["n_joined"] = int(len(out))
    rep["n_toi_used"] = int(len(used_toi))
    return out, rep


# ---------------------------------------------------------------------------
# Gaia DR3 neighbours (runner-only default; injectable)
# ---------------------------------------------------------------------------
def gaia_neighbours_upload(targets: pd.DataFrame, radius_arcsec: float, *,
                           retries: int = 3) -> pd.DataFrame:
    """Gaia TAP upload crossmatch: every DR3 source within ``radius_arcsec``.

    ``targets`` carries integer ``key``, ``ra``, ``dec``.  Returns
    ``key, source_id, ra, dec, phot_g_mean_mag, sep_arcsec``.  Async with
    backoff, sync on the last attempt (the herdsman pattern).
    """
    from astropy.table import Table  # noqa: PLC0415
    from astroquery.gaia import Gaia  # noqa: PLC0415

    up = Table.from_pandas(targets[["key", "ra", "dec"]].reset_index(drop=True))
    r_deg = float(radius_arcsec) / 3600.0
    adql = (
        "SELECT t.key AS key, g.source_id, g.ra, g.dec, g.phot_g_mean_mag, "
        "DISTANCE(POINT('ICRS', t.ra, t.dec), POINT('ICRS', g.ra, g.dec)) * 3600.0 AS sep_arcsec "
        "FROM tap_upload.targets AS t JOIN gaiadr3.gaia_source AS g "
        f"ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', t.ra, t.dec, {r_deg:.8f}))")
    last = None
    for attempt in range(int(retries)):
        try:
            if attempt + 1 == retries:
                job = Gaia.launch_job(adql, upload_resource=up, upload_table_name="targets")
            else:
                job = Gaia.launch_job_async(adql, upload_resource=up, upload_table_name="targets")
            return job.get_results().to_pandas()
        except Exception as exc:                          # noqa: BLE001
            last = exc
            print(f"[growth/acquire] Gaia upload xmatch attempt {attempt + 1}/{retries} "
                  f"failed: {exc!r}")
            _time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"Gaia neighbours failed after {retries} attempts: {last!r}")


def fetch_gaia_neighbours(stars: pd.DataFrame, *, radius_arcsec: float = 21.0,
                          chunk: int = 300, gaia_fn=None, log: AcquisitionLog | None = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Neighbours for every star in ``stars`` (``planet_key, ra, dec``), chunked.

    Returns ``(neighbours, status)``: one neighbour row per (target, Gaia
    source) and one status row per target (``OK`` / ``QUERY_FAILED``).  A
    chunk that fails marks its targets ``QUERY_FAILED`` --- they are later
    reported as ``not_checked`` rather than assumed isolated.
    """
    log = log or AcquisitionLog()
    gaia_fn = gaia_fn or gaia_neighbours_upload
    st = stars.dropna(subset=["ra", "dec"]).drop_duplicates("planet_key").reset_index(drop=True)
    st["key"] = np.arange(len(st), dtype=int)
    neigh_frames, status = [], []
    n_chunks = int(np.ceil(len(st) / max(int(chunk), 1))) if len(st) else 0
    for i in range(n_chunks):
        part = st.iloc[i * chunk:(i + 1) * chunk]
        label = f"gaia_neighbours_chunk_{i + 1}of{n_chunks}"
        try:
            df = gaia_fn(part[["key", "ra", "dec"]], radius_arcsec)
        except Exception as exc:                          # noqa: BLE001
            log.record(label, f"gaia upload xmatch r={radius_arcsec}\" n={len(part)}",
                       error=repr(exc))
            status.extend({"planet_key": k, "neighbours_status": STATUS_FAILED}
                          for k in part["planet_key"])
            continue
        df = df if df is not None else pd.DataFrame()
        df = df.rename(columns={c: str(c).lower() for c in df.columns})
        log.record(label, f"gaia upload xmatch r={radius_arcsec}\" n={len(part)}",
                   rows=int(len(df)))
        if len(df):
            df["key"] = pd.to_numeric(df["key"], errors="coerce").astype(int)
            df = df.merge(part[["key", "planet_key", "ra", "dec"]].rename(
                columns={"ra": "ra_target", "dec": "dec_target"}), on="key", how="left")
            if "sep_arcsec" not in df:
                df["sep_arcsec"] = [_sep_arcsec(a, b, c, d) for a, b, c, d in
                                    zip(df["ra_target"], df["dec_target"], df["ra"], df["dec"],
                                        strict=True)]
            neigh_frames.append(df)
        status.extend({"planet_key": k, "neighbours_status": STATUS_OK} for k in part["planet_key"])
    neigh = pd.concat(neigh_frames, ignore_index=True) if neigh_frames else pd.DataFrame(
        columns=["key", "planet_key", "source_id", "ra", "dec", "phot_g_mean_mag", "sep_arcsec"])
    return neigh, pd.DataFrame(status, columns=["planet_key", "neighbours_status"])


__all__ = [
    "EXOARCHIVE_TAP", "KOI_COLUMNS", "PS_COLUMNS", "STATUS_FAILED", "STATUS_OK", "STATUS_ZERO",
    "TOI_COLUMNS", "AcquisitionLog", "build_url", "fetch_gaia_neighbours", "fetch_koi",
    "fetch_ps_kepler", "fetch_toi", "gaia_neighbours_upload", "join_kepler_tess",
    "parse_tap_csv", "period_match", "table_query", "tap_sync",
]

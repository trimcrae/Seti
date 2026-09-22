"""Archive access for LANTERN -- runner-only (the sandbox has no egress).

Reuses the MAST path that worked for :mod:`seti.jwst_bio` (``timeseries``
data products, ``x1dints`` selection via :func:`seti.jwst_bio.run._select_x1dints`,
``download_file(dataURI)`` because ``download_products`` trips a MAST
server-side bug) and generalises it from one target to every JWST time-series
observation of a transiting planet:

1. :func:`fetch_transiting_planets`  NASA Exoplanet Archive ``pscomppars``
   (transiting planets with ephemerides) via TAP, CSV.
2. :func:`query_jwst_timeseries`     every JWST ``timeseries`` observation, per
   instrument, with ``dataRights`` kept so proprietary rows are recorded rather
   than silently dropped.
3. :func:`match_observations`        sky cross-match of the observations to the
   planet hosts (KD-tree on unit vectors; the match radius is generous because
   JWST ``s_ra``/``s_dec`` are epoch-of-observation and hosts are nearby,
   high-proper-motion stars).
4. :func:`list_x1dints`              product lists in batches, ``x1dints`` only,
   segments grouped per exposure.
5. :func:`download_x1dints` / :func:`read_x1dints`   one product at a time,
   read into a ``(n_int, n_wl)`` stack with BJD_TDB mid-times, then deleted.

Every function returns an empty / ``None`` result on failure and prints why;
the caller degrades honestly (``NO_DATA_REACHED``), never fabricates.
"""

from __future__ import annotations

import io
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..jwst_bio.run import _select_x1dints

EXOARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
PLANET_COLUMNS = [
    "pl_name", "hostname", "ra", "dec", "sy_pmra", "sy_pmdec", "sy_dist",
    "pl_orbper", "pl_orbpererr1", "pl_tranmid", "pl_tranmiderr1", "pl_trandur",
    "pl_orbeccen", "pl_orblper", "pl_ratror", "pl_ratdor", "pl_imppar",
    "pl_rade", "pl_bmasse", "pl_eqt", "st_teff", "st_rad", "tran_flag",
]
_INSTRUMENT_PATTERNS = {
    "NIRSPEC": "NIRSPEC*", "NIRISS": "NIRISS*", "NIRCAM": "NIRCAM*", "MIRI": "MIRI*",
}


def _retry(fn, retries: int = 3, pause: float = 5.0, label: str = ""):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[lantern] {label} attempt {i + 1}/{retries} failed: {exc!r}")
            time.sleep(pause * (i + 1))
    print(f"[lantern] {label} gave up: {last!r}")
    return None


# --- 1. planets ---------------------------------------------------------------------
def fetch_transiting_planets(retries: int = 3, pause: float = 5.0,
                             timeout: float = 180.0) -> pd.DataFrame:
    """All transiting planets in ``pscomppars`` with the columns LANTERN needs."""
    import requests

    q = (f"select {','.join(PLANET_COLUMNS)} from pscomppars where tran_flag=1")

    def go():
        r = requests.get(EXOARCHIVE_TAP, params={"query": q, "format": "csv"},
                         timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "pl_name" not in df.columns:
            raise RuntimeError(f"unexpected TAP response: {r.text[:200]!r}")
        return df

    df = _retry(go, retries, pause, "exoplanet-archive pscomppars")
    if df is None:
        return pd.DataFrame(columns=PLANET_COLUMNS)
    for c in PLANET_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    print(f"[lantern] exoplanet archive: {len(df)} transiting planets, "
          f"{df['hostname'].nunique()} hosts")
    return df


# --- 2. MAST observations ---------------------------------------------------------
def query_jwst_timeseries(instruments=("NIRSPEC", "NIRISS", "NIRCAM", "MIRI"),
                          retries: int = 3, pause: float = 5.0) -> pd.DataFrame:
    """Every JWST ``timeseries`` observation for the given instruments.

    One ``query_criteria`` per instrument (a single monolithic query is the
    kind that times out).  ``dataRights`` / ``calib_level`` / ``t_min`` /
    ``t_max`` are kept.  Returns an empty frame if nothing is reachable.
    """
    try:
        from astroquery.mast import Observations
    except Exception as exc:  # noqa: BLE001
        print(f"[lantern] astroquery unavailable: {exc!r}")
        return pd.DataFrame()
    frames = []
    for inst in instruments:
        pat = _INSTRUMENT_PATTERNS.get(inst.upper(), f"{inst.upper()}*")

        def go(pat=pat):
            t = Observations.query_criteria(obs_collection="JWST",
                                            dataproduct_type="timeseries",
                                            instrument_name=pat)
            return t.to_pandas() if t is not None and len(t) else pd.DataFrame()

        df = _retry(go, retries, pause, f"MAST query_criteria {pat}")
        n = 0 if df is None else len(df)
        print(f"[lantern] MAST {pat}: {n} timeseries observations")
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    obs = pd.concat(frames, ignore_index=True)
    if "obsid" in obs.columns:
        obs = obs.drop_duplicates(subset=["obsid"]).reset_index(drop=True)
    return obs


# --- 3. cross-match ------------------------------------------------------------------
def _unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def match_observations(obs: pd.DataFrame, planets: pd.DataFrame,
                       radius_arcsec: float = 30.0) -> pd.DataFrame:
    """Attach the nearest planet HOST (within ``radius_arcsec``) to each observation.

    Returns the observations that matched, with ``hostname`` and ``sep_arcsec``
    columns.  Hosts with several planets are resolved later, per planet, by
    which planet's event falls inside the observation window.
    """
    if not len(obs) or not len(planets):
        return pd.DataFrame()
    from scipy.spatial import cKDTree

    hosts = planets.dropna(subset=["ra", "dec"]).groupby("hostname", as_index=False) \
        .agg(ra=("ra", "median"), dec=("dec", "median"))
    o = obs.dropna(subset=["s_ra", "s_dec"]).copy()
    if not len(o):
        return pd.DataFrame()
    tree = cKDTree(_unit_vectors(hosts["ra"], hosts["dec"]))
    chord = 2.0 * np.sin(0.5 * np.radians(radius_arcsec / 3600.0))
    d, j = tree.query(_unit_vectors(o["s_ra"], o["s_dec"]), distance_upper_bound=chord)
    ok = np.isfinite(d)
    o = o[ok].copy()
    o["hostname"] = hosts["hostname"].to_numpy()[j[ok]]
    o["sep_arcsec"] = np.degrees(2.0 * np.arcsin(np.clip(d[ok] / 2.0, 0, 1))) * 3600.0
    print(f"[lantern] cross-match: {len(o)}/{len(obs)} observations within "
          f"{radius_arcsec}\" of a transiting-planet host ({o['hostname'].nunique()} hosts)")
    return o.reset_index(drop=True)


# --- 4. product lists -----------------------------------------------------------------
_SEG_RE = re.compile(r"[-_]seg\d{3}", re.I)


def exposure_key(filename: str) -> str:
    """Group key for one exposure: the product filename with its segment tag removed."""
    base = os.path.basename(str(filename))
    return _SEG_RE.sub("", base).replace("_x1dints.fits", "")


def list_x1dints(obs_table, batch: int = 15, retries: int = 3, pause: float = 5.0) -> pd.DataFrame:
    """``x1dints`` products for an astropy Observations table OR a list of obsid
    strings, fetched in batches (``get_product_list`` accepts either).

    Keeps ``dataURI``, ``productFilename``, ``size``, ``calib_level``,
    ``dataRights``, ``parent_obsid``/``obsID``; adds ``exposure_key``.  A batch
    that fails after retries is logged and skipped (never silently).
    """
    try:
        from astroquery.mast import Observations
    except Exception as exc:  # noqa: BLE001
        print(f"[lantern] astroquery unavailable: {exc!r}")
        return pd.DataFrame()
    frames = []
    n = len(obs_table)
    for s in range(0, n, batch):
        sub = obs_table[s:s + batch]

        def go(sub=sub):
            p = Observations.get_product_list(sub)
            return p.to_pandas() if p is not None and len(p) else pd.DataFrame()

        df = _retry(go, retries, pause, f"MAST get_product_list [{s}:{s + batch}]")
        if df is None or not len(df):
            continue
        x = _select_x1dints(df)
        if len(x):
            frames.append(x)
    if not frames:
        return pd.DataFrame()
    prod = pd.concat(frames, ignore_index=True)
    if "productFilename" in prod.columns:
        # Only the FITS tables.  MAST lists a `_x1dints.png` preview beside
        # every product; the first screen scheduled 3,384 of them as their
        # own "exposures" and every one came back `read_failed`.
        fn = prod["productFilename"].astype(str)
        prod = prod[fn.str.lower().str.endswith(".fits")].reset_index(drop=True)
        prod = prod.drop_duplicates(subset=["productFilename"]).reset_index(drop=True)
        prod["exposure_key"] = prod["productFilename"].map(exposure_key)
    return prod


# --- 5. download + read -----------------------------------------------------------------
def download_x1dints(uri: str, local: Path, retries: int = 3, pause: float = 5.0) -> str:
    """Download one product by ``dataURI`` (``download_file``, not
    ``download_products``).  Returns ``COMPLETE``, ``PROPRIETARY``, or an error string."""
    try:
        from astroquery.mast import Observations
    except Exception as exc:  # noqa: BLE001
        return f"astroquery unavailable: {exc!r}"
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for i in range(retries):
        try:
            status, msg, _url = Observations.download_file(uri, local_path=str(local),
                                                           cache=False)
            st = str(status).upper()
            if st == "COMPLETE" and local.exists() and local.stat().st_size > 0:
                return "COMPLETE"
            last = f"{st}: {msg}"
            if "401" in str(msg) or "403" in str(msg) or "proprietary" in str(msg).lower():
                return "PROPRIETARY"
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            if "401" in last or "403" in last:
                return "PROPRIETARY"
        time.sleep(pause * (i + 1))
    return f"FAILED: {last}"


_META_KEYS = ("INSTRUME", "GRATING", "FILTER", "PUPIL", "EXP_TYPE", "DETECTOR",
              "TARGPROP", "TARGNAME", "PROGRAM", "OBSERVTN", "VISIT", "EXPOSURE",
              "NINTS", "EFFINTTM", "EXPSTART", "EXPEND", "DATE-OBS", "TSOVISIT",
              "SUBARRAY", "CAL_VER", "CRDS_CTX")
# Per-row time columns of the table-per-segment layout, in order of preference.
_TDB_MID_PATTERNS = (("TDB", "MID"), ("BJD", "MID"), ("TDB", "AVG"), ("BJD", "AVG"))
_MJD_MID_PATTERNS = (("MJD", "AVG"), ("MJD", "MID"))


def _find_col(names: dict, patterns) -> str | None:
    """First column whose upper-cased name contains every token of a pattern."""
    for toks in patterns:
        for up, real in names.items():
            if all(t in up for t in toks):
                return real
    return None


def _grid_key(wl: np.ndarray) -> tuple:
    """Hashable identity of a wavelength grid (finite endpoints + length)."""
    fin = wl[np.isfinite(wl)]
    if fin.size == 0:
        return (wl.size, None, None)
    return (int(wl.size), round(float(fin[0]), 5), round(float(fin[-1]), 5))


def _int_times_lookup(hdul) -> tuple[dict | None, str | None]:
    """``INT_TIMES`` as ``{integration_number: mid_time}`` plus its provenance."""
    try:
        it = hdul["INT_TIMES"].data
    except Exception:  # noqa: BLE001
        return None, None
    names = {c.upper(): c for c in it.columns.names}
    key = _find_col(names, (("BJD", "MID"), ("TDB", "MID")))
    source = "int_times_bjd_tdb"
    if key is None:
        key = _find_col(names, (("MJD", "MID"), ("MJD", "AVG")))
        source = "int_times_mjd_utc_no_barycentric"
    if key is None:
        return None, None
    t = np.asarray(it[key], float)
    num_col = _find_col(names, (("INTEGRATION", "NUMBER"), ("INT_NUM",), ("INT", "NUM")))
    nums = (np.asarray(it[num_col]).astype(int) if num_col is not None
            else np.arange(1, t.size + 1))
    return dict(zip(nums.tolist(), t.tolist(), strict=True)), source


def _hdu_block(hdu, names: dict, dtype) -> dict | None:
    """One ``EXTRACT1D`` HDU as ``(n_int, n_wl)`` arrays, whichever layout it uses.

    Layout A (calwebb builds up to ~2024): one HDU per integration, 1-D
    columns.  Layout B (``TSOMultiSpecModel``, the layout every product in
    MAST now carries): one HDU per segment (and per spectral order), a table
    with one ROW per integration, spectral columns 2-D ``(n_int, n_wl)`` and
    per-row time columns (``INT_NUM``, ``MJD-BEG/AVG/END``,
    ``TDB-BEG/MID/END``).  The first screen read layout B as layout A: every
    row block became one "integration" and every single-HDU file failed.
    """
    data = hdu.data
    w = np.asarray(data[names["WAVELENGTH"]], float)
    f = np.asarray(data[names["FLUX"]], float)
    e = np.asarray(data[names["FLUX_ERROR"]], float) if "FLUX_ERROR" in names else None
    row_times, row_nums, row_src = None, None, None
    if f.ndim == 2:                                   # layout B
        n_int, n_wl = f.shape
        wl = w[0] if w.ndim == 2 else w.ravel()
        if wl.size != n_wl:
            return None
        flux = f.astype(dtype)
        err = e.astype(dtype) if e is not None and e.shape == f.shape else None
        tcol = _find_col(names, _TDB_MID_PATTERNS)
        if tcol is not None:
            row_times = np.asarray(data[tcol], float).ravel()
            row_src = "row_bjd_tdb"
        else:
            tcol = _find_col(names, _MJD_MID_PATTERNS)
            if tcol is not None:
                row_times = np.asarray(data[tcol], float).ravel()
                row_src = "row_mjd_utc_no_barycentric"
        ncol = _find_col(names, (("INT_NUM",), ("INTEGRATION", "NUMBER")))
        if ncol is not None:
            row_nums = np.asarray(data[ncol]).astype(int).ravel()
        if row_times is not None and (row_times.size != n_int or not np.all(np.isfinite(row_times))):
            row_times, row_src = None, None
    elif f.ndim == 1:                                 # layout A
        wl = w.ravel()
        if f.size != wl.size:
            return None
        n_int = 1
        flux = f[None, :].astype(dtype)
        err = e[None, :].astype(dtype) if e is not None and e.size == f.size else None
    else:
        return None
    h = hdu.header
    return {"wavelength": wl, "flux": flux, "flux_err": err, "n_int": int(n_int),
            "row_times": row_times, "row_nums": row_nums, "row_time_source": row_src,
            "sporder": h.get("SPORDER"), "detector": h.get("DETECTOR"),
            "segment": h.get("SEGMENT") or h.get("EXSEGNUM"), "extver": h.get("EXTVER")}


def read_x1dints_grids(path: str | Path, dtype=np.float32) -> list[dict]:
    """Read one JWST ``x1dints`` file into spectral time series, one per
    wavelength grid.

    Every ``EXTRACT1D`` HDU is decoded by :func:`_hdu_block` (either layout),
    then blocks are grouped by their wavelength grid: a NIRISS/SOSS file holds
    three spectral orders, a level-3 NIRSpec file both detectors, and each
    grid is its own time series (different sampling, different artefacts).
    Within a grid, blocks are concatenated in time order.  Times per
    integration come from, in order: the per-row ``TDB-MID`` column (BJD_TDB),
    ``INT_TIMES`` matched on integration number (or ordinal), the per-row
    ``MJD-AVG`` (UTC, no barycentric correction, flagged), else a linear
    interpolation of ``EXPSTART``/``EXPEND`` (flagged); MJD-based values are
    converted to JD.  Returns a list of stacks with ``wavelength`` (micron,
    ascending), ``flux`` ``(n_int, n_wl)``, ``flux_err``, ``times`` (BJD_TDB),
    ``time_source``, ``grid`` (order/detector/HDU count) and header ``meta``;
    an empty list if nothing was recognised.  The largest grid comes first.
    """
    try:
        from astropy.io import fits
    except Exception as exc:  # noqa: BLE001
        print(f"[lantern] astropy.io.fits unavailable: {exc!r}")
        return []
    try:
        with fits.open(path, memmap=True) as hdul:
            h0 = hdul[0].header
            meta = {k: h0.get(k) for k in _META_KEYS}
            meta["hdu_layout"] = []
            blocks: dict[tuple, list[dict]] = {}
            ordinal = 0
            for hdu in hdul:
                cols = getattr(getattr(hdu, "columns", None), "names", None)
                if not cols or getattr(hdu, "data", None) is None:
                    continue
                names = {c.upper(): c for c in cols}
                if "WAVELENGTH" not in names or "FLUX" not in names:
                    continue
                b = _hdu_block(hdu, names, dtype)
                if b is None:
                    meta["hdu_layout"].append({"ext": hdu.name, "unrecognised": True})
                    continue
                b["ordinal"] = np.arange(ordinal + 1, ordinal + 1 + b["n_int"])
                ordinal += b["n_int"]
                blocks.setdefault(_grid_key(b["wavelength"]), []).append(b)
                if len(meta["hdu_layout"]) < 12:
                    meta["hdu_layout"].append({
                        "ext": hdu.name, "extver": b["extver"], "n_int": b["n_int"],
                        "n_wl": int(b["wavelength"].size), "sporder": b["sporder"],
                        "detector": b["detector"], "row_time": b["row_time_source"],
                        "columns": [c for c in cols][:16]})
            if not blocks:
                print(f"[lantern] no EXTRACT1D table with WAVELENGTH+FLUX in {path}")
                return []
            it_map, it_src = _int_times_lookup(hdul)
            t0, t1 = meta.get("EXPSTART"), meta.get("EXPEND")
            out = []
            for key, bl in blocks.items():
                n_tot = sum(b["n_int"] for b in bl)
                times = np.full(n_tot, np.nan)
                src = None
                pos = 0
                for b in bl:
                    n = b["n_int"]
                    seg_t = None
                    if b["row_times"] is not None and b["row_time_source"] == "row_bjd_tdb":
                        seg_t, src = b["row_times"], src or "row_bjd_tdb"
                    elif it_map is not None:
                        nums = b["row_nums"] if b["row_nums"] is not None else b["ordinal"]
                        cand = np.array([it_map.get(int(k), np.nan) for k in nums])
                        if np.all(np.isfinite(cand)):
                            seg_t, src = cand, src or it_src
                    if seg_t is None and b["row_times"] is not None:
                        seg_t, src = b["row_times"], src or b["row_time_source"]
                    if seg_t is not None:
                        times[pos:pos + n] = seg_t
                    pos += n
                if not np.all(np.isfinite(times)):
                    if t0 is not None and t1 is not None and float(t1) > float(t0):
                        times = (np.linspace(float(t0), float(t1), n_tot + 1)[:-1]
                                 + 0.5 * (float(t1) - float(t0)) / n_tot)
                        src = "header_linear"
                    else:
                        times = np.arange(n_tot, dtype=float)
                        src = "index_only"
                if src != "index_only" and np.nanmedian(times) < 2.4e6:
                    times = times + 2400000.5            # MJD-based -> JD-based
                flux = np.vstack([b["flux"] for b in bl])
                errs = [b["flux_err"] for b in bl]
                err = np.vstack(errs) if all(e is not None for e in errs) else None
                order = np.argsort(times, kind="stable")
                flux, times = flux[order], times[order]
                err = err[order] if err is not None else None
                wl = bl[0]["wavelength"].copy()
                fin = np.flatnonzero(np.isfinite(wl))
                if fin.size > 1 and wl[fin[0]] > wl[fin[-1]]:
                    wl = wl[::-1]
                    flux = flux[:, ::-1]
                    err = err[:, ::-1] if err is not None else None
                out.append({"wavelength": wl, "flux": flux, "flux_err": err, "times": times,
                            "time_source": src, "meta": dict(meta), "path": str(path),
                            "grid": {"key": [key[0], key[1], key[2]],
                                     "sporder": bl[0]["sporder"], "detector": bl[0]["detector"],
                                     "n_hdus": len(bl), "n_int": int(n_tot),
                                     "n_wl": int(wl.size)}})
            out.sort(key=lambda s: -(s["flux"].shape[0] * s["flux"].shape[1]))
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[lantern] FITS read failed for {path}: {exc!r}")
        return []


def read_x1dints(path: str | Path, dtype=np.float32) -> dict | None:
    """The largest wavelength grid of :func:`read_x1dints_grids` (``None`` if none)."""
    grids = read_x1dints_grids(path, dtype)
    return grids[0] if grids else None


def concatenate_segments(stacks: list[dict]) -> dict | None:
    """Concatenate segment stacks of ONE grid along the integration axis."""
    stacks = [s for s in stacks if s is not None]
    if not stacks:
        return None
    stacks.sort(key=lambda s: float(np.nanmin(s["times"])))
    wl = stacks[0]["wavelength"]
    keep = [s for s in stacks if s["wavelength"].size == wl.size
            and np.allclose(s["wavelength"], wl, rtol=0, atol=1e-6, equal_nan=True)]
    if len(keep) != len(stacks):
        print(f"[lantern] dropped {len(stacks) - len(keep)} segment(s) with a different grid")
    if not keep:
        return None
    out = dict(keep[0])
    out["flux"] = np.vstack([s["flux"] for s in keep])
    errs = [s["flux_err"] for s in keep]
    out["flux_err"] = np.vstack(errs) if all(e is not None for e in errs) else None
    out["times"] = np.concatenate([s["times"] for s in keep])
    order = np.argsort(out["times"], kind="stable")
    out["flux"], out["times"] = out["flux"][order], out["times"][order]
    out["flux_err"] = out["flux_err"][order] if out["flux_err"] is not None else None
    out["n_segments"] = len(keep)
    out["time_source"] = keep[0]["time_source"]
    return out


def group_segments_by_grid(stacks: list[dict]) -> list[dict]:
    """Concatenate every product's grids of one exposure: stacks sharing a
    wavelength grid are joined in time; distinct grids stay separate.  The
    largest result comes first."""
    groups: dict[tuple, list[dict]] = {}
    for s in stacks:
        if s is None:
            continue
        groups.setdefault(_grid_key(np.asarray(s["wavelength"], float)), []).append(s)
    out = [concatenate_segments(g) for g in groups.values()]
    out = [o for o in out if o is not None]
    out.sort(key=lambda s: -(s["flux"].shape[0] * s["flux"].shape[1]))
    return out


def bin_integrations(stack: dict, factor: int) -> dict:
    """Co-add ``factor`` consecutive integrations (NaN-aware mean; errors in
    quadrature over the count).  Keeps a very long, finely sampled time series
    within memory while preserving the eclipse test: the bin length is chosen
    by the caller to stay far below the ingress duration."""
    k = int(max(1, factor))
    if k == 1:
        return stack
    f = np.asarray(stack["flux"])
    n = f.shape[0] // k * k
    if n < k:
        return stack
    fb = f[:n].reshape(n // k, k, f.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cnt = np.sum(np.isfinite(fb), axis=1)
        flux = np.nansum(fb, axis=1) / np.maximum(cnt, 1)
        flux = np.where(cnt > 0, flux, np.nan).astype(f.dtype)
        err = None
        if stack.get("flux_err") is not None:
            eb = np.asarray(stack["flux_err"])[:n].reshape(n // k, k, f.shape[1])
            err = (np.sqrt(np.nansum(eb ** 2, axis=1)) / np.maximum(cnt, 1)).astype(f.dtype)
            err = np.where(cnt > 0, err, np.nan).astype(f.dtype)
    out = dict(stack)
    out["flux"], out["flux_err"] = flux, err
    out["times"] = np.asarray(stack["times"], float)[:n].reshape(n // k, k).mean(axis=1)
    out["binned_by"] = k
    out["n_int_raw"] = int(f.shape[0])
    return out


__all__ = ["fetch_transiting_planets", "query_jwst_timeseries", "match_observations",
           "list_x1dints", "exposure_key", "download_x1dints", "read_x1dints",
           "read_x1dints_grids", "concatenate_segments", "group_segments_by_grid",
           "bin_integrations", "PLANET_COLUMNS"]

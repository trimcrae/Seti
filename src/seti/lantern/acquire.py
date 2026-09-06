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


def read_x1dints(path: str | Path, dtype=np.float32) -> dict | None:
    """Read one JWST ``x1dints`` file into a spectral time series.

    Derived from :func:`seti.jwst_bio.run._read_x1dints`, extended with the
    metadata LANTERN needs (grating/filter, exposure times, NINTS) and a
    complete time path: ``INT_TIMES`` ``int_mid_BJD_TDB`` (converted from the
    MJD-based value the pipeline stores to a JD when needed), else a linear
    interpolation between ``EXPSTART``/``EXPEND`` with no barycentric
    correction, flagged as ``time_source='header_linear'``.  Returns
    ``wavelength`` (micron), ``flux`` ``(n_int, n_wl)``, ``flux_err``, ``times``
    (BJD_TDB), and header metadata; ``None`` if the structure is not recognised.
    """
    try:
        from astropy.io import fits
    except Exception as exc:  # noqa: BLE001
        print(f"[lantern] astropy.io.fits unavailable: {exc!r}")
        return None
    try:
        with fits.open(path, memmap=True) as hdul:
            h0 = hdul[0].header
            meta = {k: h0.get(k) for k in ("INSTRUME", "GRATING", "FILTER", "PUPIL",
                                            "EXP_TYPE", "DETECTOR", "TARGPROP", "TARGNAME",
                                            "PROGRAM", "OBSERVTN", "VISIT", "EXPOSURE",
                                            "NINTS", "EFFINTTM", "EXPSTART", "EXPEND",
                                            "DATE-OBS", "TSOVISIT", "SUBARRAY")}
            wl, flux_rows, err_rows = None, [], []
            for hdu in hdul:
                if getattr(hdu, "data", None) is None:
                    continue
                cols = getattr(getattr(hdu, "columns", None), "names", None)
                if not cols:
                    continue
                names = {c.upper(): c for c in cols}
                if "WAVELENGTH" not in names or "FLUX" not in names:
                    continue
                w = np.asarray(hdu.data[names["WAVELENGTH"]], float).ravel()
                f = np.asarray(hdu.data[names["FLUX"]], float).ravel()
                if wl is None:
                    wl = w
                if f.size != wl.size:
                    continue
                flux_rows.append(f.astype(dtype))
                if "FLUX_ERROR" in names:
                    err_rows.append(np.asarray(hdu.data[names["FLUX_ERROR"]],
                                               float).ravel().astype(dtype))
            if wl is None or len(flux_rows) < 2:
                return None
            flux = np.vstack(flux_rows)
            err = np.vstack(err_rows) if len(err_rows) == len(flux_rows) else None
            n_int = flux.shape[0]
            times, source = None, None
            try:
                it = hdul["INT_TIMES"].data
                names = list(it.columns.names)
                key = [c for c in names if "BJD" in c.upper() and "MID" in c.upper()]
                if not key:
                    key = [c for c in names if "MJD" in c.upper() and "MID" in c.upper()]
                    source = "int_times_mjd_utc_no_barycentric"
                else:
                    source = "int_times_bjd_tdb"
                if key:
                    t = np.asarray(it[key[0]], float)
                    if t.size >= n_int:
                        times = t[:n_int]
            except Exception:  # noqa: BLE001
                times = None
            if times is None or not np.all(np.isfinite(times)):
                t0, t1 = meta.get("EXPSTART"), meta.get("EXPEND")
                if t0 is not None and t1 is not None:
                    times = np.linspace(float(t0), float(t1), n_int + 1)[:-1] \
                        + 0.5 * (float(t1) - float(t0)) / n_int
                    source = "header_linear"
                else:
                    times = np.arange(n_int, dtype=float)
                    source = "index_only"
            if source != "index_only" and np.nanmedian(times) < 2.4e6:
                times = times + 2400000.5           # MJD-based -> JD-based
            # Wavelength grid may be descending (NIRISS); make it ascending.
            if wl.size > 1 and wl[0] > wl[-1]:
                wl = wl[::-1]
                flux = flux[:, ::-1]
                err = err[:, ::-1] if err is not None else None
            return {"wavelength": wl, "flux": flux, "flux_err": err, "times": times,
                    "time_source": source, "meta": meta, "path": str(path)}
    except Exception as exc:  # noqa: BLE001
        print(f"[lantern] FITS read failed for {path}: {exc!r}")
        return None


def concatenate_segments(stacks: list[dict]) -> dict | None:
    """Concatenate segment stacks of one exposure along the integration axis."""
    stacks = [s for s in stacks if s is not None]
    if not stacks:
        return None
    stacks.sort(key=lambda s: float(np.nanmin(s["times"])))
    wl = stacks[0]["wavelength"]
    keep = [s for s in stacks if s["wavelength"].size == wl.size
            and np.allclose(s["wavelength"], wl, rtol=0, atol=1e-6)]
    if len(keep) != len(stacks):
        print(f"[lantern] dropped {len(stacks) - len(keep)} segment(s) with a different grid")
    if not keep:
        return None
    out = dict(keep[0])
    out["flux"] = np.vstack([s["flux"] for s in keep])
    errs = [s["flux_err"] for s in keep]
    out["flux_err"] = np.vstack(errs) if all(e is not None for e in errs) else None
    out["times"] = np.concatenate([s["times"] for s in keep])
    out["n_segments"] = len(keep)
    out["time_source"] = keep[0]["time_source"]
    return out


__all__ = ["fetch_transiting_planets", "query_jwst_timeseries", "match_observations",
           "list_x1dints", "exposure_key", "download_x1dints", "read_x1dints",
           "concatenate_segments", "PLANET_COLUMNS"]

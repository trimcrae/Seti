"""Gaia DR3 seeds and the positional join for SPARK.

Network only through injected ``query_fn`` callables (``adql -> DataFrame``);
the geometry is pure and offline-tested.  Two routes to the same rows:

* **IRSA's copy** (``gaia_dr3_source`` — the table name is checked against
  ``TAP_SCHEMA`` at probe time, never assumed), which is the same service
  the Euclid and SPHEREx tables live on;
* **ESA** (``gaiadr3.gaia_source``) through the transport ladder IGNITION
  built (:func:`seti.ignition.sample.run_gaia_query`).

Whichever answered is recorded in the ledger — a parent drawn from two
copies is stated, never silently mixed.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

GAIA_COLS = ("source_id", "ra", "dec", "pmra", "pmdec", "parallax", "parallax_over_error",
             "phot_g_mean_mag", "bp_rp", "ruwe", "phot_variable_flag", "phot_bp_mean_mag",
             "phot_rp_mean_mag")

ESA_TABLE = "gaiadr3.gaia_source"


def gaia_cone_adql(table: str, ra: float, dec: float, radius_deg: float, g_max: float,
                   plx_over_error_min: float | None = None, columns=GAIA_COLS,
                   top: int | None = None) -> str:
    """One cone, the columns SPARK needs, optional parallax-quality cut."""
    cols = ", ".join(columns)
    q = (f"SELECT {'TOP ' + str(int(top)) + ' ' if top else ''}{cols} FROM {table} WHERE "
         f"1 = CONTAINS(POINT('ICRS', ra, dec), "
         f"CIRCLE('ICRS', {float(ra):.6f}, {float(dec):.6f}, {float(radius_deg):.6f})) "
         f"AND phot_g_mean_mag < {float(g_max):.3f}")
    if plx_over_error_min is not None:
        q += f" AND parallax_over_error > {float(plx_over_error_min):.3f}"
    return q


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).lower() for c in df.columns})


def fetch_gaia_cone(ra: float, dec: float, radius_deg: float, g_max: float, *,
                    irsa_tables=(), irsa_query_fn=None, esa_query_fn=None,
                    plx_over_error_min: float | None = None, label: str = "gaia",
                    log: list | None = None) -> tuple[pd.DataFrame, dict]:
    """IRSA's Gaia copy first (if the probe found one), ESA next.  Never raises.

    Returns ``(df, record)``; ``record["route"]`` is ``irsa:<table>``, ``esa``
    or ``None`` (``status`` = ``QUERY_FAILED`` / ``QUERY_RETURNED_ZERO_ROWS``).
    """
    rec: dict = {"label": label, "route": None, "status": "QUERY_FAILED", "n_rows": 0,
                 "attempts": []}
    routes: list[tuple[str, str, object]] = []
    for t in irsa_tables or ():
        if irsa_query_fn is not None:
            routes.append((f"irsa:{t}", t, irsa_query_fn))
    if esa_query_fn is not None:
        routes.append(("esa", ESA_TABLE, esa_query_fn))
    for route, table, fn in routes:
        adql = gaia_cone_adql(table, ra, dec, radius_deg, g_max, plx_over_error_min)
        t0 = _time.monotonic()
        try:
            df = fn(adql)
            if isinstance(df, tuple):
                df = df[0]
            df = _lower(pd.DataFrame(df))
            n = int(len(df))
            rec["attempts"].append({"route": route, "ok": True, "n_rows": n,
                                    "seconds": round(_time.monotonic() - t0, 1)})
            rec.update(route=route, n_rows=n,
                       status="OK" if n else "QUERY_RETURNED_ZERO_ROWS", query=adql[:1000])
            if log is not None:
                log.append(dict(rec))
            return df, rec
        except Exception as exc:                        # noqa: BLE001
            rec["attempts"].append({"route": route, "ok": False, "error": repr(exc)[:400],
                                    "seconds": round(_time.monotonic() - t0, 1)})
            rec["error"] = repr(exc)[:400]
            print(f"[spark] {label}: {route} failed: {exc!r}"[:300], flush=True)
    if log is not None:
        log.append(dict(rec))
    return pd.DataFrame(columns=list(GAIA_COLS)), rec


# --------------------------------------------------------------------------
# Pure geometry
# --------------------------------------------------------------------------
def propagate(df: pd.DataFrame, from_epoch: float, to_epoch: float) -> pd.DataFrame:
    """Positions moved by proper motion (mas/yr, ``pmra`` includes cos δ); NaN pm → 0."""
    out = df.copy()
    if not len(out):
        out["ra_ep"], out["dec_ep"] = [], []
        return out
    dt = float(to_epoch - from_epoch)
    ra = pd.to_numeric(out["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(out["dec"], errors="coerce").to_numpy(float)
    pmra = np.nan_to_num(pd.to_numeric(out.get("pmra", 0.0), errors="coerce").to_numpy(float))
    pmdec = np.nan_to_num(pd.to_numeric(out.get("pmdec", 0.0), errors="coerce").to_numpy(float))
    cosd = np.cos(np.radians(dec))
    cosd = np.where(np.abs(cosd) < 1e-6, 1e-6, cosd)
    out["ra_ep"] = ra + (pmra * dt / 3.6e6) / cosd
    out["dec_ep"] = dec + pmdec * dt / 3.6e6
    return out


def _unit(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def crossmatch(ra1, dec1, ra2, dec2, radius_arcsec: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest neighbour of every (ra1, dec1) in (ra2, dec2) within the radius.

    Returns ``(idx2, sep_arcsec)`` aligned with the first list; ``idx2 = -1``
    and ``sep = inf`` when nothing is inside the radius.
    """
    ra1 = np.atleast_1d(np.asarray(ra1, float))
    n1 = ra1.size
    if n1 == 0 or np.asarray(ra2).size == 0:
        return np.full(n1, -1, int), np.full(n1, np.inf)
    from scipy.spatial import cKDTree
    tree = cKDTree(_unit(ra2, dec2))
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    d, j = tree.query(_unit(ra1, dec1), k=1, distance_upper_bound=chord)
    ok = np.isfinite(d)
    sep = np.full(n1, np.inf)
    sep[ok] = np.degrees(2.0 * np.arcsin(np.clip(d[ok] / 2.0, 0, 1))) * 3600.0
    idx = np.where(ok, j, -1)
    return idx, sep


def neighbours_within(ra1, dec1, ra2, dec2, radius_arcsec: float) -> list[list[int]]:
    """Indices into list 2 within the radius of each point of list 1 (KD-tree)."""
    ra1 = np.atleast_1d(np.asarray(ra1, float))
    if ra1.size == 0 or np.asarray(ra2).size == 0:
        return [[] for _ in range(ra1.size)]
    from scipy.spatial import cKDTree
    tree = cKDTree(_unit(ra2, dec2))
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    return [list(x) for x in tree.query_ball_point(_unit(ra1, dec1), r=chord)]


def sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation in arcsec (haversine)."""
    r1, d1 = np.radians(np.asarray(ra1, float)), np.radians(np.asarray(dec1, float))
    r2, d2 = np.radians(np.asarray(ra2, float)), np.radians(np.asarray(dec2, float))
    a = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))) * 3600.0


def isolation(seeds: pd.DataFrame, field: pd.DataFrame, *, outer_arcsec: float,
              outer_dmag: float, inner_arcsec: float, inner_dmag: float) -> pd.DataFrame:
    """Per seed: brightest neighbour inside each radius and whether the seed is isolated.

    ``field`` is every Gaia source in the box (to the neighbour depth) and
    usually contains the seeds themselves — a source is never its own
    neighbour (matched on ``source_id`` when present, else by zero separation).
    """
    out = seeds.copy()
    n = len(out)
    out["n_within_outer"] = 0
    out["brightest_outer_dmag"] = np.nan
    out["brightest_inner_dmag"] = np.nan
    out["isolated"] = True
    if n == 0 or not len(field):
        return out
    idx_lists = neighbours_within(out["ra"], out["dec"], field["ra"], field["dec"], outer_arcsec)
    fg = pd.to_numeric(field["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    fra = field["ra"].to_numpy(float)
    fdec = field["dec"].to_numpy(float)
    fid = field["source_id"].to_numpy() if "source_id" in field else None
    for k in range(n):
        g0 = float(out["phot_g_mean_mag"].iloc[k])
        sid = out["source_id"].iloc[k] if "source_id" in out else None
        js = [j for j in idx_lists[k]
              if not (fid is not None and sid is not None and fid[j] == sid)]
        if not js:
            continue
        seps = sep_arcsec(out["ra"].iloc[k], out["dec"].iloc[k], fra[js], fdec[js])
        keep = seps > 1e-3 if fid is None else np.ones(len(js), bool)
        js = [j for j, kp in zip(js, keep, strict=True) if kp]
        seps = seps[keep]
        if not js:
            continue
        dm = fg[js] - g0
        out.loc[out.index[k], "n_within_outer"] = int(len(js))
        out.loc[out.index[k], "brightest_outer_dmag"] = float(np.nanmin(dm))
        inner = seps <= inner_arcsec
        if inner.any():
            out.loc[out.index[k], "brightest_inner_dmag"] = float(np.nanmin(dm[inner]))
        bad = bool(np.nanmin(dm) < outer_dmag) or (inner.any() and bool(np.nanmin(dm[inner]) < inner_dmag))
        out.loc[out.index[k], "isolated"] = not bad
    return out

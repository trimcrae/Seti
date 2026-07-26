"""Runner-only archive access for RUST: paired ZTF g+r light curves.

The sandbox has no archive egress (every IRSA/Gaia call returns
``CONNECT tunnel failed, response 403``); everything here runs on a GitHub
Actions runner.  The primitives themselves are inherited --- ``fetch_ztf_region``
and ``fetch_ztf_lightcurve`` from :mod:`seti.dimming.acquire` are already proven
at 250k-star scale --- so this module imports them rather than reimplementing
them, and adds the one thing RUST needs that ``dimming`` did not:

**the two bands paired per source, from the same request pair.**

The repository ledger is unambiguous that a single-band ZTF anomaly is an
artefact until confirmed in a second band, and for a *second-moment* search that
is not a follow-up nicety --- it is the primary discriminant.  A blended
neighbour, a bad reference image, or a ghost affects one band's photometry;
macroscopic debris crossing the star affects both, achromatically.  So the
channel never scores a star it cannot measure in both g and r.

ZTF assigns different ``oid`` values to the same star in different filters, so
the pairing is positional: median position per ``oid`` in each band, matched
with a KD-tree at a sub-arcsecond-to-arcsecond tolerance.

ATLAS is deliberately *not* wired.  Its forced-photometry service requires a
per-user account token that this repository does not hold, so wiring it would
produce a code path that can only ever fail on the runner.  ASAS-SN is wired for
survivors only (:func:`fetch_asassn_lc`), as the cross-survey confirmation that
a ZTF-internal trend is not a ZTF artefact.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from ..dimming.acquire import ZTF_LC_URL, fetch_ztf_lightcurve, fetch_ztf_region

# Sub-arcsecond is too tight for ZTF's per-band astrometry on faint sources;
# 1.5" is comfortably inside the ~2" PSF and outside the astrometric jitter.
PAIR_TOL_ARCSEC = 1.5


def pair_bands(lcs_g: dict[str, pd.DataFrame], lcs_r: dict[str, pd.DataFrame],
               tol_arcsec: float = PAIR_TOL_ARCSEC) -> list[dict]:
    """Positionally match per-``oid`` light curves between two bands.

    Returns a list of ``{"oid_g", "oid_r", "ra", "dec", "ccd", "lc_g", "lc_r",
    "sep_arcsec"}``.  Each g source is matched to its nearest r source within
    ``tol_arcsec``; ties are broken by separation and an r source is used once.
    Pure function --- no network --- so the pairing logic is unit-tested offline.
    """
    if not lcs_g or not lcs_r:
        return []
    from scipy.spatial import cKDTree

    def _pos(lcs):
        oids, ras, decs = [], [], []
        for oid, lc in lcs.items():
            ra = float(np.nanmedian(lc["ra"])) if "ra" in lc else np.nan
            dec = float(np.nanmedian(lc["dec"])) if "dec" in lc else np.nan
            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue
            oids.append(oid)
            ras.append(ra)
            decs.append(dec)
        return oids, np.asarray(ras), np.asarray(decs)

    og, rag, decg = _pos(lcs_g)
    orr, rar, decr = _pos(lcs_r)
    if not og or not orr:
        return []
    # Tangent-plane projection about the field centre: fine over a ZTF tile.
    dec0 = float(np.median(np.concatenate([decg, decr])))
    cosd = max(np.cos(np.radians(dec0)), 1e-3)
    xy_g = np.column_stack([rag * cosd, decg])
    xy_r = np.column_stack([rar * cosd, decr])
    tol_deg = tol_arcsec / 3600.0
    tree = cKDTree(xy_r)
    dist, idx = tree.query(xy_g, k=1, distance_upper_bound=tol_deg)

    order = np.argsort(dist)
    used: set[int] = set()
    out: list[dict] = []
    for i in order:
        j = int(idx[i])
        if not np.isfinite(dist[i]) or j >= len(orr) or j in used:
            continue
        used.add(j)
        lc_g, lc_r = lcs_g[og[i]], lcs_r[orr[j]]
        out.append({
            "oid_g": og[i], "oid_r": orr[j],
            "source_id": f"{og[i]}_{orr[j]}",
            "ra": float(rag[i]), "dec": float(decg[i]),
            "ccd_g": lc_g.attrs.get("ccd", "x"),
            "ccd_r": lc_r.attrs.get("ccd", "x"),
            "ccd": lc_r.attrs.get("ccd", lc_g.attrs.get("ccd", "x")),
            "sep_arcsec": float(dist[i] * 3600.0),
            "lc_g": lc_g, "lc_r": lc_r,
        })
    return out


def fetch_ztf_region_2band(ra: float, dec: float, box_deg: float = 0.12,
                           min_epochs: int = 60, timeout_s: float = 120.0,
                           tol_arcsec: float = PAIR_TOL_ARCSEC) -> list[dict]:
    """Bulk-fetch one sky box in g and r and return positionally paired sources.

    ``min_epochs`` is higher than the ``dimming`` default on purpose: a
    second-moment statistic across >=4 seasons needs order 8+ epochs *per season*
    in *each* band, so a 30-epoch curve is not usable here.
    """
    lcs_g = fetch_ztf_region(ra, dec, box_deg=box_deg, band="g",
                             timeout_s=timeout_s, min_epochs=min_epochs)
    lcs_r = fetch_ztf_region(ra, dec, box_deg=box_deg, band="r",
                             timeout_s=timeout_s, min_epochs=min_epochs)
    pairs = pair_bands(lcs_g, lcs_r, tol_arcsec=tol_arcsec)
    print(f"[rust] box ({ra:.4f},{dec:.4f}): g={len(lcs_g)} r={len(lcs_r)} "
          f"-> {len(pairs)} paired")
    return pairs


def iter_region_2band(ra: float, dec: float, radius_deg: float = 1.0,
                      box_deg: float = 0.12, min_epochs: int = 60,
                      time_budget_s: float = 2400.0, max_boxes: int | None = None):
    """Tile a field and yield every g/r-paired ZTF source in it.

    Bounded by ``time_budget_s`` so a slow IRSA degrades to partial coverage
    rather than a lost run; whatever was fetched is used and the coverage is
    reported as a first-class field by :mod:`seti.rust.run`.
    """
    cos_d = max(np.cos(np.radians(dec)), 0.05)
    n_side = max(1, int(np.ceil(2 * radius_deg / box_deg)))
    offs = (np.arange(n_side) - (n_side - 1) / 2.0) * box_deg
    boxes = [(ra + dx / cos_d, dec + dy) for dy in offs for dx in offs]
    if max_boxes is not None:
        boxes = boxes[:max_boxes]
    t0 = _time.monotonic()
    seen: set[str] = set()
    n_box = n_src = 0
    for bra, bdec in boxes:
        if _time.monotonic() - t0 > time_budget_s:
            print(f"[rust] time budget reached after {n_box}/{len(boxes)} boxes "
                  f"({n_src} paired sources)")
            break
        n_box += 1
        try:
            pairs = fetch_ztf_region_2band(bra, bdec, box_deg=box_deg,
                                           min_epochs=min_epochs)
        except Exception as exc:                      # noqa: BLE001
            print(f"[rust] box ({bra:.4f},{bdec:.4f}) failed: {exc!r}")
            continue
        for p in pairs:
            if p["source_id"] in seen:
                continue
            seen.add(p["source_id"])
            n_src += 1
            yield p
    print(f"[rust] region sweep: {n_src} paired g/r sources from {n_box} boxes")


def fetch_ztf_2band(ra: float, dec: float, radius_arcsec: float = 2.0):
    """Single-target g and r light curves (survivor follow-up)."""
    g = fetch_ztf_lightcurve(ra, dec, band="g", radius_arcsec=radius_arcsec)
    r = fetch_ztf_lightcurve(ra, dec, band="r", radius_arcsec=radius_arcsec)
    return g, r


def fetch_asassn_lc(ra: float, dec: float, radius_arcsec: float = 5.0):
    """Independent ASAS-SN Sky Patrol light curve --- the cross-survey check.

    A ZTF-internal trend in the second moment is only believable if a *different*
    telescope, different pipeline and different cadence sees it too.  Returns
    ``(mjd, mag, magerr)`` or ``None`` if the service is unreachable or the
    client is not installed --- a degradation the caller must report, not hide.
    """
    try:
        from pyasassn.client import SkyPatrolClient
    except Exception:                                  # noqa: BLE001
        print("[rust] pyasassn not installed; ASAS-SN cross-check unavailable")
        return None
    try:
        client = SkyPatrolClient()
        lcs = client.cone_search(ra_deg=ra, dec_deg=dec,
                                 radius=radius_arcsec / 3600.0,
                                 catalog="master_list", download=True)
        if lcs is None or not len(lcs.data):
            return None
        df = lcs.data
        df = df[(pd.to_numeric(df.get("mag_err"), errors="coerce") < 0.2)
                & (pd.to_numeric(df.get("mag"), errors="coerce") > 0)]
        if not len(df):
            return None
        return (df["jd"].to_numpy() - 2400000.5, df["mag"].to_numpy(),
                df["mag_err"].to_numpy())
    except Exception as exc:                           # noqa: BLE001
        print(f"[rust] ASAS-SN query failed: {exc!r}")
        return None


def fetch_gaia_context(positions: pd.DataFrame, radius_arcsec: float = 5.0) -> pd.DataFrame:
    """Gaia DR3 context for a shortlist: quality flags **and** the crowding census.

    Returns one row per input position with the nearest source's astrometry plus
    ``n_neighbors_5as`` / ``brightest_neighbor_dg`` --- the blending diagnostic.
    A variable neighbour inside the ZTF PSF leaks its variability into the target
    and is the most mundane way to manufacture rising scatter, so the neighbour
    census is a funnel stage, not an afterthought.
    """
    from astroquery.gaia import Gaia

    rows = []
    for _, p in positions.iterrows():
        ra, dec = float(p["ra"]), float(p["dec"])
        q = f"""
            SELECT source_id, ra, dec, parallax, parallax_over_error, ruwe,
                   phot_g_mean_mag, bp_rp, phot_variable_flag, non_single_star,
                   phot_bp_rp_excess_factor, astrometric_excess_noise,
                   DISTANCE(POINT('ICRS', ra, dec),
                            POINT('ICRS', {ra}, {dec})) AS d
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                               CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / 3600.0}))
            ORDER BY d ASC
        """
        try:
            df = Gaia.launch_job_async(q).get_results().to_pandas()
        except Exception as exc:                       # noqa: BLE001
            print(f"[rust] Gaia context failed at ({ra:.5f},{dec:.5f}): {exc!r}")
            rows.append({"source_id": p.get("source_id"), "gaia_ok": False})
            continue
        if df.empty:
            rows.append({"source_id": p.get("source_id"), "gaia_ok": False})
            continue
        df = df.rename(columns={c: c.lower() for c in df.columns})
        t = df.iloc[0]
        g0 = float(t.get("phot_g_mean_mag") or np.nan)
        nb = df.iloc[1:]
        dg = (pd.to_numeric(nb.get("phot_g_mean_mag"), errors="coerce") - g0
              if len(nb) else pd.Series(dtype=float))
        rows.append({
            "source_id": p.get("source_id"), "gaia_ok": True,
            "gaia_source_id": int(t.get("source_id")),
            "match_arcsec": float(t.get("d") or np.nan) * 3600.0,
            "g_mag": g0, "bp_rp": float(t.get("bp_rp") or np.nan),
            "parallax": float(t.get("parallax") or np.nan),
            "parallax_over_error": float(t.get("parallax_over_error") or np.nan),
            "ruwe": float(t.get("ruwe") or np.nan),
            "non_single_star": int(t.get("non_single_star") or 0),
            "phot_variable_flag": str(t.get("phot_variable_flag") or ""),
            "bp_rp_excess_factor": float(t.get("phot_bp_rp_excess_factor") or np.nan),
            "astrometric_excess_noise": float(t.get("astrometric_excess_noise") or np.nan),
            "n_neighbors_5as": int(len(nb)),
            "brightest_neighbor_dg": float(dg.min()) if len(dg) and dg.notna().any()
            else float("nan"),
        })
    return pd.DataFrame(rows)


__all__ = ["PAIR_TOL_ARCSEC", "ZTF_LC_URL", "fetch_asassn_lc", "fetch_gaia_context",
           "fetch_ztf_2band", "fetch_ztf_region_2band", "iter_region_2band",
           "pair_bands"]

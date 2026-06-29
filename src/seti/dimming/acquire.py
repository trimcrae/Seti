"""Target selection and light-curve acquisition for the dimming search.

The hunt for Boyajian's-star analogues needs two things the interactive sandbox
cannot reach but a GitHub runner can: a stellar target list (Gaia DR3 via TAP)
and per-target optical light curves (ZTF via the IRSA light-curve API).

Target strategy.  Deep, irregular dippers are rare, so a blind scan of random
stars is mostly wasted exposure.  We bias the sample toward fertile ground by
optionally restricting to Gaia ``phot_variable_flag = 'VARIABLE'`` sources --- a
population already known to vary but, for millions of them, *unclassified*.
Boyajian's star itself is a low-amplitude variable whose defining feature is the
deep *aperiodic* dips; selecting Gaia variables and asking ZTF which ones dip
deeply and aperiodically is a tractable, under-examined search.  The sky is
tiled by repeated runs over (RA, Dec, radius) fields so each runner job stays
bounded.

Everything is defensive: a target whose light curve cannot be fetched is simply
skipped, never an aborted run.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator

import numpy as np
import pandas as pd

ZTF_LC_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"


def fetch_gaia_targets(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 1.5,
    g_min: float = 13.0,
    g_max: float = 18.5,
    variable_only: bool = True,
    limit: int = 5000,
) -> pd.DataFrame:
    """Gaia DR3 stellar targets in a sky field, ordered for a dimming search.

    Selects sources in a cone within a magnitude window (bright enough for clean
    ZTF photometry, faint enough to be unsaturated), optionally restricted to
    Gaia-flagged variables.  Returns a frame keyed on ``source_id`` with the sky
    position and a few context columns.
    """
    from astroquery.gaia import Gaia

    var_clause = ("AND phot_variable_flag = 'VARIABLE'" if variable_only else "")
    query = f"""
        SELECT TOP {int(limit)}
               source_id, ra, dec, phot_g_mean_mag, bp_rp,
               parallax, parallax_over_error, phot_variable_flag
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                           CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))
          AND phot_g_mean_mag BETWEEN {g_min} AND {g_max}
          AND dec > -28.0
          {var_clause}
    """
    job = Gaia.launch_job_async(query)
    df = job.get_results().to_pandas()
    # Normalise column case (TAP backends differ) to the lower-case schema.
    df = df.rename(columns={c: c.lower() for c in df.columns})
    print(f"[dimming] Gaia targets: {len(df)} sources in field "
          f"({ra_deg:.3f}, {dec_deg:.3f}) r={radius_deg} deg, "
          f"variable_only={variable_only}")
    return df.reset_index(drop=True)


def fetch_ztf_lightcurve(
    ra: float,
    dec: float,
    band: str = "r",
    radius_arcsec: float = 2.0,
    timeout_s: float = 25.0,
    bad_catflags_mask: int = 32768,
) -> pd.DataFrame | None:
    """One ZTF light curve (time, mag, magerr) from the IRSA cone-search API.

    Returns a tidy frame with columns ``mjd``, ``mag``, ``magerr`` (quality-bad
    epochs already filtered by ``BAD_CATFLAGS_MASK``), or ``None`` if nothing was
    returned.  Defensive: any error yields ``None``.
    """
    import requests

    rad_deg = radius_arcsec / 3600.0
    params = {
        "POS": f"CIRCLE {float(ra):.6f} {float(dec):.6f} {rad_deg:.6f}",
        "BANDNAME": band,
        "FORMAT": "CSV",
        "BAD_CATFLAGS_MASK": str(bad_catflags_mask),
    }
    try:
        resp = requests.get(ZTF_LC_URL, params=params, timeout=timeout_s)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        lc = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        print(f"[dimming] ZTF fetch failed at ({ra:.5f},{dec:.5f}): {exc!r}")
        return None
    if lc.empty or "mag" not in lc.columns:
        return None
    tcol = next((c for c in ("mjd", "hjd", "bjd") if c in lc.columns), None)
    if tcol is None:
        return None
    out = pd.DataFrame({
        "mjd": pd.to_numeric(lc[tcol], errors="coerce"),
        "mag": pd.to_numeric(lc["mag"], errors="coerce"),
        "magerr": (pd.to_numeric(lc["magerr"], errors="coerce")
                   if "magerr" in lc.columns else np.nan),
    })
    out = out[np.isfinite(out["mjd"]) & np.isfinite(out["mag"])]
    return out.reset_index(drop=True) if len(out) else None


def fetch_ztf_region(
    ra: float,
    dec: float,
    box_deg: float = 0.12,
    band: str = "r",
    timeout_s: float = 90.0,
    bad_catflags_mask: int = 32768,
    min_epochs: int = 30,
) -> dict[str, pd.DataFrame]:
    """Bulk-fetch *every* ZTF light curve in a small sky box in one request.

    The IRSA light-curve service returns all epochs of all sources inside a
    ``BOX`` region; grouping by object id (``oid``) yields one light curve per
    source.  A single box request replaces hundreds of per-object cone queries,
    the throughput unlock that lets a runner search tens of thousands of stars.

    Returns ``{oid: lightcurve_frame}`` for sources with at least ``min_epochs``
    good epochs; each frame carries ``mjd``, ``mag``, ``magerr``, ``ra``, ``dec``.
    Defensive: any error yields an empty dict.
    """
    import requests

    # The IRSA ZTF light-curve service reliably supports a CIRCLE cone for POS
    # (the proven per-object primitive); a cone of radius ~box_deg/2 returns every
    # source in the tile in one request, which is the bulk unlock we want.
    radius_deg = box_deg / 2.0
    params = {
        "POS": f"CIRCLE {float(ra):.5f} {float(dec):.5f} {radius_deg:.5f}",
        "BANDNAME": band,
        "FORMAT": "CSV",
        "BAD_CATFLAGS_MASK": str(bad_catflags_mask),
    }
    try:
        resp = requests.get(ZTF_LC_URL, params=params, timeout=timeout_s)
        if resp.status_code != 200 or not resp.text.strip():
            print(f"[dimming] ZTF region cone ({ra:.4f},{dec:.4f}) r={radius_deg:.4f}: "
                  f"HTTP {resp.status_code}, {len(resp.text)} bytes")
            return {}
        lc = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        print(f"[dimming] ZTF region fetch failed at ({ra:.4f},{dec:.4f}): {exc!r}")
        return {}
    if lc.empty or "mag" not in lc.columns or "oid" not in lc.columns:
        return {}
    tcol = next((c for c in ("mjd", "hjd", "bjd") if c in lc.columns), None)
    if tcol is None:
        return {}
    lc = lc.assign(
        _mjd=pd.to_numeric(lc[tcol], errors="coerce"),
        _mag=pd.to_numeric(lc["mag"], errors="coerce"),
        _magerr=(pd.to_numeric(lc["magerr"], errors="coerce")
                 if "magerr" in lc.columns else np.nan),
    )
    # ZTF systematics (zeropoint / reference-image drift) act per readout channel,
    # so record each source's field/CCD/quadrant for a per-CCD common-mode detrend.
    fcol = next((c for c in ("field", "fieldid") if c in lc.columns), None)
    ccol = next((c for c in ("ccdid", "ccd_id", "ccd") if c in lc.columns), None)
    qcol = next((c for c in ("qid", "quadrant") if c in lc.columns), None)
    out: dict[str, pd.DataFrame] = {}
    for oid, g in lc.groupby("oid"):
        good = g[np.isfinite(g["_mjd"]) & np.isfinite(g["_mag"])]
        if len(good) < min_epochs:
            continue
        parts = [str(good[c].iloc[0]) if (c and c in good) else "x"
                 for c in (fcol, ccol, qcol)]
        ccd = "_".join(parts)
        df = pd.DataFrame({
            "mjd": good["_mjd"].to_numpy(), "mag": good["_mag"].to_numpy(),
            "magerr": good["_magerr"].to_numpy(),
            "ra": float(np.nanmedian(good["ra"])) if "ra" in good else np.nan,
            "dec": float(np.nanmedian(good["dec"])) if "dec" in good else np.nan,
        })
        df.attrs["ccd"] = ccd
        out[str(oid)] = df
    return out


def iter_region_lightcurves(
    ra: float,
    dec: float,
    radius_deg: float = 1.0,
    box_deg: float = 0.12,
    band: str = "r",
    min_epochs: int = 30,
    time_budget_s: float = 2400.0,
    max_boxes: int | None = None,
):
    """Tile a field into boxes and yield ``(meta, lightcurve)`` for every ZTF source.

    Covers a square field of half-width ``radius_deg`` with a grid of ``box_deg``
    boxes (declination-corrected in RA), bulk-fetching each.  ``meta`` carries the
    ZTF ``oid`` and the source position so candidates can later be matched to Gaia
    for the HR cut.  Bounded by ``time_budget_s`` and optionally ``max_boxes``.
    """
    import time

    cos_d = max(np.cos(np.radians(dec)), 0.05)
    n_side = max(1, int(np.ceil(2 * radius_deg / box_deg)))
    offs = (np.arange(n_side) - (n_side - 1) / 2.0) * box_deg
    boxes = [(ra + dx / cos_d, dec + dy) for dy in offs for dx in offs]
    if max_boxes is not None:
        boxes = boxes[:max_boxes]
    t0 = time.monotonic()
    n_box = n_src = 0
    seen: set[str] = set()
    for bra, bdec in boxes:
        if time.monotonic() - t0 > time_budget_s:
            print(f"[dimming] region time budget reached after {n_box} boxes "
                  f"({n_src} sources)")
            break
        n_box += 1
        lcs = fetch_ztf_region(bra, bdec, box_deg=box_deg, band=band,
                               min_epochs=min_epochs)
        for oid, lc in lcs.items():
            if oid in seen:        # boxes can overlap at edges; de-duplicate
                continue
            seen.add(oid)
            n_src += 1
            meta = {"source_id": oid, "ra": lc["ra"].iloc[0] if len(lc) else bra,
                    "dec": lc["dec"].iloc[0] if len(lc) else bdec,
                    "ccd": lc.attrs.get("ccd", "x")}
            yield meta, lc
    print(f"[dimming] region sweep: {n_src} ZTF light curves from {n_box} boxes")


def iter_lightcurves(
    targets: pd.DataFrame,
    band: str = "r",
    radius_arcsec: float = 2.0,
    timeout_s: float = 25.0,
    time_budget_s: float = 1800.0,
    max_targets: int | None = None,
) -> Iterator[tuple[dict, pd.DataFrame]]:
    """Yield ``(target_row_dict, lightcurve_frame)`` for each target with data.

    Bounded by ``time_budget_s`` (wall clock) and optionally ``max_targets`` so a
    slow IRSA never stalls the runner; whatever was fetched before the budget is
    used and the rest are left unsearched.
    """
    sub = targets if max_targets is None else targets.head(max_targets)
    t0 = time.monotonic()
    n_tried = n_got = 0
    for _, r in sub.iterrows():
        if time.monotonic() - t0 > time_budget_s:
            print(f"[dimming] time budget ({time_budget_s:.0f}s) reached after "
                  f"{n_tried} targets ({n_got} with light curves)")
            break
        n_tried += 1
        lc = fetch_ztf_lightcurve(float(r["ra"]), float(r["dec"]), band=band,
                                  radius_arcsec=radius_arcsec, timeout_s=timeout_s)
        if lc is None or lc.empty:
            continue
        n_got += 1
        yield r.to_dict(), lc
    print(f"[dimming] fetched {n_got} light curves from {n_tried} targets")


__all__ = ["fetch_gaia_targets", "fetch_ztf_lightcurve", "iter_lightcurves",
           "fetch_ztf_region", "iter_region_lightcurves", "ZTF_LC_URL"]

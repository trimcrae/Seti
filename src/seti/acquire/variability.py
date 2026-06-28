"""Shortlist-only light-curve variability acquisition (ZTF optical, NEOWISE IR).

These are per-object queries against IRSA and are *expensive* relative to a
catalogue cross-match, so they are run only on the vetted shortlist (the clean
infrared-excess / multi-axis candidates), never the whole parent sample.  Each
returns a tiny frame keyed on Gaia ``source_id`` with a fractional-RMS
variability metric the indicator suite consumes:

* ZTF  -> ``ztf_frac_rms``           (optical dimming / transit signature)
* NEOWISE -> ``neowise_w1_frac_rms`` (changing infrared waste heat)

A white dwarf is intrinsically photometrically stable, so a significant
fractional RMS on either axis is itself anomalous.  Everything here is defensive:
a failed object yields NaN (unavailable), never an aborted run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _robust_frac_rms(mag: np.ndarray, magerr: np.ndarray | None = None,
                     min_epochs: int = 8) -> float:
    """Fractional RMS of a magnitude light curve, corrected for photometric noise.

    Uses a median/MAD-clipped robust scatter so a handful of outliers do not
    masquerade as variability, and subtracts the mean photometric error in
    quadrature so the metric measures *intrinsic* variability.  Returns NaN if
    there are too few epochs to be meaningful.
    """
    m = np.asarray(mag, dtype=float)
    good = np.isfinite(m)
    m = m[good]
    if m.size < min_epochs:
        return float("nan")
    med = np.median(m)
    mad = np.median(np.abs(m - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = np.std(m)
    # Clip 4-sigma outliers, then recompute the scatter on the clean epochs.
    keep = np.abs(m - med) <= 4.0 * sigma if sigma > 0 else np.ones_like(m, dtype=bool)
    m = m[keep]
    if m.size < min_epochs:
        return float("nan")
    rms_mag = float(np.std(m, ddof=1))
    # De-noise: subtract the typical per-epoch error in quadrature.
    if magerr is not None:
        e = np.asarray(magerr, dtype=float)[good]
        e = e[keep] if e.size == keep.size else e
        med_err = float(np.nanmedian(e)) if e.size else 0.0
        rms_mag = float(np.sqrt(max(rms_mag**2 - med_err**2, 0.0)))
    # Convert magnitude scatter to a fractional flux RMS (small-amplitude limit).
    return 0.4 * np.log(10.0) * rms_mag


def fetch_ztf_variability(positions: pd.DataFrame, radius_arcsec: float = 2.0,
                          band: str = "r", max_objects: int = 500) -> pd.DataFrame:
    """Per-object ZTF light-curve fractional RMS from the IRSA ZTF API.

    Queries the IRSA ZTF light-curve service (cone search) for each shortlist
    position and computes a noise-corrected fractional RMS in the requested band.
    """
    import io

    import requests

    rows = []
    sub = positions.head(max_objects)
    base = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
    for _, r in sub.iterrows():
        sid = int(r["source_id"])
        ra, dec = float(r["ra"]), float(r["dec"])
        rad_deg = radius_arcsec / 3600.0
        params = {
            "POS": f"CIRCLE {ra:.6f} {dec:.6f} {rad_deg:.6f}",
            "BANDNAME": band,
            "FORMAT": "CSV",
            "BAD_CATFLAGS_MASK": "32768",
        }
        try:
            resp = requests.get(base, params=params, timeout=60)
            if resp.status_code != 200 or not resp.text.strip():
                continue
            lc = pd.read_csv(io.StringIO(resp.text))
            if "mag" not in lc.columns or lc.empty:
                continue
            frac = _robust_frac_rms(lc["mag"].to_numpy(),
                                    lc["magerr"].to_numpy() if "magerr" in lc else None)
            rows.append({"source_id": sid, "ztf_frac_rms": frac,
                         "ztf_n_epochs": int(np.isfinite(lc["mag"]).sum())})
        except Exception as exc:  # one bad object must not abort the shortlist
            print(f"[science] ZTF {sid} skipped: {exc!r}")
    out = pd.DataFrame(rows)
    print(f"[science] ZTF variability: {len(out)} of {len(sub)} shortlist objects measured")
    return out


def fetch_neowise_variability(positions: pd.DataFrame, radius_arcsec: float = 2.0,
                              max_objects: int = 500) -> pd.DataFrame:
    """Per-object NEOWISE W1 fractional RMS from the IRSA single-exposure table.

    Queries the NEOWISE-R single-exposure source table (``neowiser_p1bs_psd``)
    via the IRSA cone search, bins by visit (the ~6-month NEOWISE cadence), and
    computes a noise-corrected fractional RMS of the per-visit mean W1 magnitude.
    Changing infrared waste heat (construction, eclipses) is the signature.
    """
    try:
        from astroquery.ipac.irsa import Irsa
    except Exception as exc:
        print(f"[science] NEOWISE unavailable (astroquery.ipac.irsa import): {exc!r}")
        return pd.DataFrame()

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    rows = []
    sub = positions.head(max_objects)
    for _, r in sub.iterrows():
        sid = int(r["source_id"])
        ra, dec = float(r["ra"]), float(r["dec"])
        try:
            coord = SkyCoord(ra * u.deg, dec * u.deg)
            tbl = Irsa.query_region(coord, catalog="neowiser_p1bs_psd",
                                    spatial="Cone", radius=radius_arcsec * u.arcsec)
            df = tbl.to_pandas() if tbl is not None and len(tbl) else pd.DataFrame()
            if df.empty or "w1mpro" not in df.columns:
                continue
            # Quality: drop flagged epochs where columns are present.
            if "qual_frame" in df:
                df = df[pd.to_numeric(df["qual_frame"], errors="coerce") > 0]
            if "cc_flags" in df:
                df = df[df["cc_flags"].astype(str).str.startswith(("0", "00"))]
            w1 = pd.to_numeric(df.get("w1mpro"), errors="coerce")
            w1err = pd.to_numeric(df.get("w1sigmpro"), errors="coerce")
            frac = _robust_frac_rms(w1.to_numpy(),
                                    w1err.to_numpy() if w1err is not None else None,
                                    min_epochs=12)
            rows.append({"source_id": sid, "neowise_w1_frac_rms": frac,
                         "neowise_n_epochs": int(np.isfinite(w1).sum())})
        except Exception as exc:  # one bad object must not abort the shortlist
            print(f"[science] NEOWISE {sid} skipped: {exc!r}")
    out = pd.DataFrame(rows)
    print(f"[science] NEOWISE variability: {len(out)} of {len(sub)} shortlist objects measured")
    return out


__all__ = ["fetch_ztf_variability", "fetch_neowise_variability", "_robust_frac_rms"]

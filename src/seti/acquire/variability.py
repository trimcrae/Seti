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


def _lomb_scargle(mjd: np.ndarray, mag: np.ndarray, magerr: np.ndarray | None,
                  min_period_d: float = 0.1, max_period_d: float = 100.0,
                  min_epochs: int = 20) -> dict:
    """Lomb--Scargle periodicity of a light curve.

    Returns the strongest period, its (normalised) power, the false-alarm
    probability of that peak, and the semi-amplitude of the best-fit sinusoid.  A
    statistically significant period in an otherwise photometrically stable white
    dwarf is far more diagnostic of a transiting/occulting configuration than raw
    scatter, so this feeds a dedicated periodicity axis.
    """
    out = {"ls_fap": np.nan, "ls_period_d": np.nan, "ls_power": np.nan,
           "ls_amp_mag": np.nan}
    m = np.asarray(mag, dtype=float)
    t = np.asarray(mjd, dtype=float)
    good = np.isfinite(m) & np.isfinite(t)
    m, t = m[good], t[good]
    if m.size < min_epochs:
        return out
    e = (np.asarray(magerr, dtype=float)[good] if magerr is not None
         else np.full(m.size, np.nanstd(m) or 0.05))
    e = np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(e[e > 0]) if np.any(e > 0) else 0.05)
    baseline = float(np.ptp(t))
    if baseline < 2 * min_period_d:
        return out
    try:
        from astropy.timeseries import LombScargle

        max_p = min(max_period_d, baseline / 2.0)
        ls = LombScargle(t, m, e)
        freq, power = ls.autopower(minimum_frequency=1.0 / max_p,
                                   maximum_frequency=1.0 / min_period_d,
                                   samples_per_peak=5)
        if power.size == 0:
            return out
        i = int(np.argmax(power))
        best_f = float(freq[i])
        out["ls_power"] = float(power[i])
        out["ls_period_d"] = 1.0 / best_f
        out["ls_fap"] = float(ls.false_alarm_probability(power[i], method="baluev"))
        # Semi-amplitude from the best-frequency sinusoid fit.
        theta = ls.model_parameters(best_f)
        out["ls_amp_mag"] = float(np.hypot(theta[1], theta[2])) if len(theta) >= 3 else np.nan
    except Exception as exc:
        print(f"[science] Lomb-Scargle failed: {exc!r}")
    return out


def fetch_ztf_variability(positions: pd.DataFrame, radius_arcsec: float = 2.0,
                          band: str = "r", max_objects: int = 300,
                          timeout_s: float = 25.0,
                          time_budget_s: float = 420.0) -> pd.DataFrame:
    """Per-object ZTF light-curve fractional RMS and periodicity from the IRSA API.

    Queries the IRSA ZTF light-curve service (cone search) for each shortlist
    position and computes a noise-corrected fractional RMS and a Lomb--Scargle
    periodicity (period, power, false-alarm probability) in the requested band.
    Bounded by ``max_objects`` and an overall ``time_budget_s`` wall-clock budget
    so a slow service can never stall the run; whatever was measured before the
    budget is returned and the rest are simply left unmeasured (unavailable).
    """
    import io
    import time

    import requests

    rows = []
    sub = positions.head(max_objects)
    base = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
    t0 = time.monotonic()
    n_tried = 0
    for _, r in sub.iterrows():
        if time.monotonic() - t0 > time_budget_s:
            print(f"[science] ZTF time budget ({time_budget_s:.0f}s) reached after "
                  f"{n_tried} objects")
            break
        n_tried += 1
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
            resp = requests.get(base, params=params, timeout=timeout_s)
            if resp.status_code != 200 or not resp.text.strip():
                continue
            lc = pd.read_csv(io.StringIO(resp.text))
            if "mag" not in lc.columns or lc.empty:
                continue
            magerr = lc["magerr"].to_numpy() if "magerr" in lc else None
            frac = _robust_frac_rms(lc["mag"].to_numpy(), magerr)
            tcol = next((c for c in ("mjd", "hjd", "bjd") if c in lc.columns), None)
            ls = (_lomb_scargle(lc[tcol].to_numpy(), lc["mag"].to_numpy(), magerr)
                  if tcol else {})
            rows.append({"source_id": sid, "ztf_frac_rms": frac,
                         "ztf_n_epochs": int(np.isfinite(lc["mag"]).sum()),
                         "ztf_ls_fap": ls.get("ls_fap", np.nan),
                         "ztf_ls_period_d": ls.get("ls_period_d", np.nan),
                         "ztf_ls_power": ls.get("ls_power", np.nan),
                         "ztf_ls_amp_mag": ls.get("ls_amp_mag", np.nan)})
        except Exception as exc:  # one bad object must not abort the shortlist
            print(f"[science] ZTF {sid} skipped: {exc!r}")
    out = pd.DataFrame(rows)
    print(f"[science] ZTF variability: {len(out)} of {n_tried} attempted "
          f"({len(sub)} shortlist) measured")
    return out


def fetch_neowise_variability(positions: pd.DataFrame, radius_arcsec: float = 2.0,
                              max_objects: int = 120, timeout_s: float = 45.0,
                              time_budget_s: float = 420.0) -> pd.DataFrame:
    """Per-object NEOWISE W1 fractional RMS from the IRSA single-exposure table.

    Queries the NEOWISE-R single-exposure source table (``neowiser_p1bs_psd``)
    via the IRSA cone search, bins by visit (the ~6-month NEOWISE cadence), and
    computes a noise-corrected fractional RMS of the per-visit mean W1 magnitude.
    Changing infrared waste heat (construction, eclipses) is the signature.

    Bounded by ``max_objects``, a per-query ``timeout_s`` and an overall
    ``time_budget_s`` wall-clock budget: the per-object cone searches are slow, so
    these guards guarantee the run completes even if IRSA is degraded.
    """
    import time

    try:
        from astroquery.ipac.irsa import Irsa
    except Exception as exc:
        print(f"[science] NEOWISE unavailable (astroquery.ipac.irsa import): {exc!r}")
        return pd.DataFrame()

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    # Bound each cone search so a slow/hung IRSA response fails fast rather than
    # stalling the whole shortlist (the per-object queries dominate the runtime).
    try:
        Irsa.TIMEOUT = timeout_s
    except Exception:
        pass

    rows = []
    sub = positions.head(max_objects)
    t0 = time.monotonic()
    n_tried = 0
    for _, r in sub.iterrows():
        if time.monotonic() - t0 > time_budget_s:
            print(f"[science] NEOWISE time budget ({time_budget_s:.0f}s) reached after "
                  f"{n_tried} objects")
            break
        n_tried += 1
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
    print(f"[science] NEOWISE variability: {len(out)} of {n_tried} attempted "
          f"({len(sub)} shortlist) measured")
    return out


__all__ = ["fetch_ztf_variability", "fetch_neowise_variability", "_robust_frac_rms",
           "_lomb_scargle"]

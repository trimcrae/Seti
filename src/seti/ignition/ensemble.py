"""Ensemble zero-point correction for the IGNITION screen.  Pure.

Run 35039105536 screened 172 stars and labelled **97 of them FADING** --- a
5-sigma *negative* slope in both bands on more than half of a sample chosen to
be constant dwarfs.  Nothing on the sky does that to half of a field; a survey
does.  NEOWISE's W1/W2 zero points are not constant over 2014--2024 (the
explanatory supplement documents mission-long calibration trends and the
per-scan bias corrections applied to the single-exposure photometry), and the
channel's epoch errors are ~0.005--0.01 mag, so a common drift of a few
hundredths over the decade is a 5-sigma slope on every star at once.

The correction is the standard ensemble one: per band and per time bin, the
median over the shard's stars of ``mag - <mag>_star`` is the zero-point offset
of that bin, and it is subtracted from every star's epochs in the bin.  A
genuine riser among dozens of constant neighbours moves the median by
nothing; a drift shared by all of them is removed exactly.  The offsets are
returned so the screen can record them, and the correction is only applied to
bins holding at least ``min_stars`` stars --- a bin with too few is left as it
is and said so.

This changes no threshold of the rise test.  It changes what the test is
applied to, from the survey's photometry to the survey's photometry relative
to its own constant stars, which is the quantity the claim is about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ENSEMBLE: dict = {"bin_yr": 0.25, "min_stars": 8}


def ensemble_offsets(epochs: pd.DataFrame, bin_yr: float = 0.25,
                     min_stars: int = 8) -> dict:
    """Per-band zero-point offsets by time bin from a long epoch table.

    ``epochs`` has columns ``source_id, band, t_yr, mag`` (``err`` optional).
    Returns ``{band: {"bin_lo": [...], "offset": [...], "n_stars": [...],
    "applied": [...]}}`` where ``applied`` says whether the bin met
    ``min_stars``.  Offsets are medians of ``mag - median_star(mag)``.
    """
    out: dict = {}
    if epochs is None or not len(epochs):
        return out
    d = epochs.copy()
    d["source_id"] = d["source_id"].astype(str)
    d["t_yr"] = pd.to_numeric(d["t_yr"], errors="coerce")
    d["mag"] = pd.to_numeric(d["mag"], errors="coerce")
    d = d.dropna(subset=["t_yr", "mag"])
    for band, g in d.groupby("band"):
        med = g.groupby("source_id")["mag"].transform("median")
        resid = g["mag"] - med
        b = np.floor(g["t_yr"].to_numpy(float) / float(bin_yr)) * float(bin_yr)
        tab = pd.DataFrame({"bin": b, "resid": resid.to_numpy(float),
                            "sid": g["source_id"].to_numpy()})
        rows = tab.groupby("bin").agg(offset=("resid", "median"), n_stars=("sid", "nunique"))
        rows = rows.reset_index().sort_values("bin")
        out[str(band)] = {
            "bin_lo": [round(float(x), 3) for x in rows["bin"]],
            "offset": [round(float(x), 5) for x in rows["offset"]],
            "n_stars": [int(x) for x in rows["n_stars"]],
            "applied": [bool(n >= int(min_stars)) for n in rows["n_stars"]],
        }
    return out


def apply_ensemble(epochs: pd.DataFrame, offsets: dict, bin_yr: float = 0.25) -> pd.DataFrame:
    """Subtract the applied offsets; adds ``mag_raw`` and ``ensemble_applied`` columns."""
    d = epochs.copy()
    d["mag"] = pd.to_numeric(d["mag"], errors="coerce")
    d["mag_raw"] = d["mag"]
    d["ensemble_applied"] = False
    if not offsets:
        return d
    t = pd.to_numeric(d["t_yr"], errors="coerce").to_numpy(float)
    b = np.floor(t / float(bin_yr)) * float(bin_yr)
    for band, o in offsets.items():
        lut = {round(float(lo), 3): float(off) for lo, off, ap in
               zip(o["bin_lo"], o["offset"], o["applied"], strict=False) if ap}
        if not lut:
            continue
        mask = (d["band"].astype(str) == str(band)).to_numpy()
        keys = np.round(b, 3)
        corr = np.array([lut.get(float(k), np.nan) for k in keys])
        hit = mask & np.isfinite(corr)
        d.loc[hit, "mag"] = d.loc[hit, "mag"].to_numpy(float) - corr[hit]
        d.loc[hit, "ensemble_applied"] = True
    return d


def summarise_offsets(offsets: dict) -> dict:
    """Peak-to-peak and end-to-end drift per band, for the summary."""
    out = {}
    for band, o in offsets.items():
        off = np.asarray(o["offset"], float)
        ap = np.asarray(o["applied"], bool)
        if not ap.any():
            out[band] = {"n_bins": int(len(off)), "n_applied": 0}
            continue
        v = off[ap]
        lo = np.asarray(o["bin_lo"], float)[ap]
        out[band] = {"n_bins": int(len(off)), "n_applied": int(ap.sum()),
                     "ptp_mag": round(float(v.max() - v.min()), 4),
                     "first_to_last_mag": round(float(v[-1] - v[0]), 4),
                     "span_yr": round(float(lo[-1] - lo[0]), 2)}
    return out


__all__ = ["DEFAULT_ENSEMBLE", "apply_ensemble", "ensemble_offsets", "summarise_offsets"]

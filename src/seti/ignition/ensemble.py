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


# ---------------------------------------------------------------------------
# The stratified correction (from the re-vet of tiles run 35740159635)
# ---------------------------------------------------------------------------
#: W1/W2 magnitude bins and |ecliptic latitude| bands of the stratified ensemble.
#: Run 35860165284 measured the NEOWISE drift over 169,749 stars: W1 0.24 ->
#: 1.45 mmag/yr and W2 1.7 -> 3.5 mmag/yr from W1 8-9 to 11.5-12, so one median
#: per time bin over-corrects the bright end.  Cadence differs with |beta|.
MAG_EDGES = [6.0, 8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 16.0]
BETA_EDGES = [0.0, 40.0, 65.0, 90.1]
STRATUM_MIN_STARS = 40


def stratified_offsets(ep: pd.DataFrame, beta_by_sid: pd.Series | None, bin_yr: float = 0.25,
                       mag_edges=MAG_EDGES, beta_edges=BETA_EDGES,
                       min_stars: int = STRATUM_MIN_STARS) -> tuple[pd.DataFrame, dict]:
    """Median (mag - star median) per (band, magnitude bin, |beta| band, time bin).

    A star's magnitude bin is set by its own median in that band.  Where a
    (stratum, time bin) cell has fewer than ``min_stars`` stars the correction
    falls back to the magnitude bin over all |beta| ("coarse"), then to the
    plain per-time-bin median of the band ("global", the old correction), and
    only then leaves the epoch uncorrected; the fractions are reported.
    Returns ``(epochs, report)`` with ``mag`` corrected, ``mag_raw`` kept, and
    ``ensemble_applied`` / ``ensemble_level`` per epoch.
    """
    d = ep.copy()
    d["source_id"] = d["source_id"].astype(str)
    d["t_yr"] = pd.to_numeric(d["t_yr"], errors="coerce")
    d["mag"] = pd.to_numeric(d["mag"], errors="coerce")
    d = d.dropna(subset=["t_yr", "mag"]).reset_index(drop=True)
    med = d.groupby(["source_id", "band"])["mag"].transform("median")
    d["resid"] = d["mag"] - med
    d["mbin"] = np.digitize(med.to_numpy(float), mag_edges)
    if beta_by_sid is not None and len(beta_by_sid):
        bb = pd.Series(beta_by_sid)
        bb.index = bb.index.astype(str)
        bb = bb[~bb.index.duplicated()]
        beta = bb.reindex(d["source_id"]).to_numpy(float)
    else:
        beta = np.full(len(d), np.nan)
    # an unknown |beta| is its own band (len(edges)); it falls back to coarse
    d["bband"] = np.where(np.isfinite(beta), np.digitize(np.nan_to_num(beta), beta_edges),
                          len(beta_edges) + 1)
    d["tbin"] = np.round(np.floor(d["t_yr"].to_numpy(float) / bin_yr) * bin_yr, 3)
    keys = {"fine": ["band", "mbin", "bband", "tbin"], "coarse": ["band", "mbin", "tbin"],
            "global": ["band", "tbin"]}
    for lvl, k in keys.items():
        t = (d.groupby(k).agg(**{f"off_{lvl}": ("resid", "median"),
                                 f"n_{lvl}": ("source_id", "nunique")}).reset_index())
        d = d.merge(t, on=k, how="left")
    use_f = (d["n_fine"] >= min_stars).to_numpy()
    use_c = ~use_f & (d["n_coarse"] >= min_stars).to_numpy()
    use_g = ~use_f & ~use_c & (d["n_global"] >= min(8, min_stars)).to_numpy()
    corr = np.where(use_f, d["off_fine"], np.where(use_c, d["off_coarse"],
                                                   np.where(use_g, d["off_global"], np.nan)))
    applied = np.isfinite(corr)
    d["mag_raw"] = d["mag"]
    d["mag"] = np.where(applied, d["mag"].to_numpy(float) - np.nan_to_num(corr), d["mag"])
    d["ensemble_applied"] = applied
    d["ensemble_level"] = np.where(use_f, "fine", np.where(use_c, "coarse",
                                                           np.where(use_g, "global", "none")))
    n = max(len(d), 1)
    rep = {"mode": "stratified", "bin_yr": bin_yr, "mag_edges": list(mag_edges),
           "beta_edges": list(beta_edges), "min_stars": int(min_stars),
           "frac_epochs_fine": round(float(use_f.sum()) / n, 4),
           "frac_epochs_coarse": round(float(use_c.sum()) / n, 4),
           "frac_epochs_global": round(float(use_g.sum()) / n, 4),
           "frac_epochs_uncorrected": round(float((~applied).sum()) / n, 4)}
    drift = []
    coarse = d.drop_duplicates(["band", "mbin", "tbin"])[["band", "mbin", "tbin", "off_coarse",
                                                          "n_coarse"]]
    for (band, mbin), g in coarse[coarse["n_coarse"] >= min_stars].groupby(["band", "mbin"]):
        g = g.sort_values("tbin")
        if len(g) < 8:
            continue
        slope = np.polyfit(g["tbin"].to_numpy(float), g["off_coarse"].to_numpy(float), 1)[0]
        lo = mag_edges[mbin - 1] if 0 < mbin <= len(mag_edges) - 1 else None
        hi = mag_edges[mbin] if 0 <= mbin < len(mag_edges) else None
        drift.append({"band": str(band), "mag_lo": lo, "mag_hi": hi,
                      "n_stars_median": int(np.median(g["n_coarse"])),
                      "offset_slope_mmag_yr": round(1e3 * float(slope), 3),
                      "offset_ptp_mmag": round(1e3 * float(g["off_coarse"].max()
                                                         - g["off_coarse"].min()), 2)})
    rep["drift_by_mag"] = drift
    keep = [c for c in ep.columns if c not in ("mag",)] + ["mag", "mag_raw",
                                                          "ensemble_applied", "ensemble_level"]
    keep = list(dict.fromkeys(c for c in keep if c in d.columns))
    return d[keep], rep


__all__ = ["BETA_EDGES", "DEFAULT_ENSEMBLE", "MAG_EDGES", "STRATUM_MIN_STARS", "apply_ensemble",
           "ensemble_offsets", "stratified_offsets", "summarise_offsets"]

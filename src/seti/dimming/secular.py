"""Secular (long-term monotonic) fading --- a second, artifact-robust signature.

KIC 8462852 shows not only deep aperiodic dips but a slow *secular* dimming ---
it faded by a few per cent over the Kepler mission and by ~15% over a century
(Schaefer 2016).  A star steadily being enshrouded (a Dyson swarm under
construction, growing dust/debris) would fade monotonically over years with no
periodicity and no infrared excess if the obscuring material is cool or
optically grey.

Crucially this signature is far more robust to the single-epoch photometric
artefacts that swamp a deep-dip search: a trend is measured from *season medians*
across many epochs, so a handful of bad points cannot manufacture it.  We bin a
light curve into observing seasons, require several seasons, fit a weighted line
to the season medians, and score a significant, large, monotonic fade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SecularStats:
    n_epochs: int
    baseline_yr: float
    n_seasons: int
    slope_mag_yr: float      # >0 means fading (getting fainter) in magnitudes/yr
    slope_sigma: float       # significance of the slope (|slope|/error)
    total_change_mag: float  # slope * baseline: total magnitude change over the data
    monotonic_frac: float    # fraction of season-to-season steps in the fading sense
    rms_resid_mag: float     # scatter of season medians about the linear trend
    score: float             # [0,1] secular-fade likeness (fading only)

    def as_dict(self) -> dict:
        return {k: (int(v) if isinstance(v, int) else float(v))
                for k, v in self.__dict__.items()}


def _season_bins(t: np.ndarray, season_days: float = 180.0) -> np.ndarray:
    """Integer season label for each epoch (gap-aware ~half-year bins)."""
    return np.floor((t - t.min()) / season_days).astype(int)


def detect_secular_fade(time: np.ndarray, mag: np.ndarray,
                        magerr: np.ndarray | None = None,
                        min_epochs: int = 40, min_seasons: int = 3,
                        season_days: float = 180.0) -> SecularStats | None:
    """Fit a long-term trend to a light curve's season medians.

    Returns ``None`` if there are too few epochs or seasons.  ``slope_mag_yr`` is
    positive for a *fading* star; the score rewards a significant, large,
    monotonic fade and is zero for brightening or flat/periodic curves.
    """
    t = np.asarray(time, dtype=float)
    m = np.asarray(mag, dtype=float)
    good = np.isfinite(t) & np.isfinite(m)
    t, m = t[good], m[good]
    if t.size < min_epochs:
        return None
    e = (np.asarray(magerr, dtype=float)[good] if magerr is not None
         else np.full(m.size, np.nanstd(m) or 0.02))
    e = np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(e[e > 0]) if np.any(e > 0)
                 else 0.02)

    labels = _season_bins(t, season_days)
    uniq = np.unique(labels)
    if uniq.size < min_seasons:
        return None
    # Season medians and their robust uncertainties (MAD / sqrt N).
    s_t, s_m, s_w = [], [], []
    for lab in uniq:
        sel = labels == lab
        if sel.sum() < 3:                 # ignore sparsely-sampled seasons
            continue
        tt, mm = t[sel], m[sel]
        med = float(np.median(mm))
        mad = float(np.median(np.abs(mm - med))) * 1.4826
        n = mm.size
        err = (mad / np.sqrt(n)) if mad > 0 else (float(np.std(mm)) / np.sqrt(n) or 0.01)
        s_t.append(float(np.median(tt)))
        s_m.append(med)
        s_w.append(1.0 / max(err, 1e-3) ** 2)
    if len(s_t) < min_seasons:
        return None
    s_t = np.asarray(s_t)
    s_m = np.asarray(s_m)
    s_w = np.asarray(s_w)

    # Weighted linear fit of season-median magnitude vs time (years).
    yr = (s_t - s_t.min()) / 365.25
    W = s_w
    Sw = W.sum()
    Sx = (W * yr).sum()
    Sy = (W * s_m).sum()
    Sxx = (W * yr * yr).sum()
    Sxy = (W * yr * s_m).sum()
    denom = Sw * Sxx - Sx * Sx
    if denom <= 0:
        return None
    slope = (Sw * Sxy - Sx * Sy) / denom
    intercept = (Sy - slope * Sx) / Sw
    resid = s_m - (intercept + slope * yr)
    dof = max(len(s_t) - 2, 1)
    chi2 = float((W * resid ** 2).sum())
    slope_var = Sw / denom
    slope_err = np.sqrt(slope_var * max(chi2 / dof, 1.0))   # inflate by reduced chi2
    slope_sigma = float(abs(slope) / slope_err) if slope_err > 0 else 0.0
    baseline_yr = float(yr.max() - yr.min())
    total_change = float(slope * baseline_yr)
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))

    # Monotonicity: fraction of consecutive season-median steps in the fading sense.
    steps = np.diff(s_m)
    sign = np.sign(slope) if slope != 0 else 1.0
    monotonic_frac = float(np.mean(np.sign(steps) == sign)) if steps.size else 0.0

    # Score (fading only): significant, large, monotonic, and well-fit by a line.
    if slope <= 0:                        # brightening or flat -> not a fade
        score = 0.0
    else:
        sig_term = np.clip((slope_sigma - 3.0) / 5.0, 0, 1)
        amp_term = np.clip(abs(total_change) / 0.20, 0, 1)   # ~0.2 mag fade saturates
        mono_term = np.clip((monotonic_frac - 0.5) / 0.4, 0, 1)
        score = float(np.clip(0.4 * sig_term + 0.35 * amp_term + 0.25 * mono_term,
                              0, 1))
    return SecularStats(
        n_epochs=int(t.size), baseline_yr=baseline_yr, n_seasons=int(len(s_t)),
        slope_mag_yr=float(slope), slope_sigma=slope_sigma,
        total_change_mag=total_change, monotonic_frac=monotonic_frac,
        rms_resid_mag=rms_resid, score=score)


__all__ = ["SecularStats", "detect_secular_fade"]

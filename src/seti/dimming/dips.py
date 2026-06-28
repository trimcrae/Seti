"""Dip statistics for the anomalous-dimming (Boyajian-star) search.

Given a light curve (time, magnitude, error) we measure the brightness baseline
robustly from the bright state, identify epochs dimmed significantly below it, and
summarise the dimming with metrics that separate the interesting case --- deep,
irregular, aperiodic dips --- from the mundane ones: a smooth pulsation, or the
strictly periodic, repeating eclipses of a binary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DipStats:
    n_epochs: int
    max_depth: float        # deepest fractional flux dip below baseline
    n_dips: int             # epochs dimmed > depth threshold and > k-sigma
    dip_duty_cycle: float   # fraction of epochs in a dip
    asymmetry: float        # |median(dip) skew|: dimming asymmetry about baseline
    best_period_d: float    # strongest dip period (Lomb-Scargle on the dips)
    period_power: float     # its normalised power (high => periodic => binary-like)
    score: float            # [0,1] Boyajian-likeness

    def as_dict(self) -> dict:
        return {k: (float(v) if not isinstance(v, int) else int(v))
                for k, v in self.__dict__.items()}


def _robust_baseline(mag: np.ndarray) -> float:
    """Bright-state baseline: a dimming star spends most time near maximum light,
    so the bright (low-magnitude) quantile is the un-dimmed level."""
    return float(np.nanpercentile(mag, 20))   # 20th pct in mag = bright state


def detect_dips(time: np.ndarray, mag: np.ndarray, magerr: np.ndarray | None = None,
                depth_min: float = 0.05, k_sigma: float = 3.0,
                min_epochs: int = 30) -> DipStats | None:
    """Compute dip statistics for one light curve.

    ``depth_min`` is the minimum fractional flux drop (e.g. 0.05 = 5%) and
    ``k_sigma`` the per-epoch significance below the baseline for an epoch to count
    as a dip.  Returns ``None`` if there are too few epochs.
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

    base = _robust_baseline(m)
    # Fractional flux dip relative to the bright baseline (mag fainter => positive).
    dmag = m - base
    frac_dip = 1.0 - 10.0 ** (-0.4 * dmag)        # >0 when fainter than baseline
    # An epoch is a dip if it is both deep enough and significant given its error.
    sig = dmag / (0.4 * np.log(10.0) * e + 1e-9)  # mag drop in sigma
    is_dip = (frac_dip >= depth_min) & (sig >= k_sigma)

    n_dips = int(is_dip.sum())
    max_depth = float(np.nanmax(frac_dip)) if frac_dip.size else 0.0
    duty = n_dips / t.size
    # Asymmetry about the *median* level: a Boyajian-like light curve has deep
    # excursions only to the faint side, so the faint-side absolute deviation far
    # exceeds the bright-side; a symmetric oscillation or flat noise gives ~1.
    med = float(np.nanmedian(m))
    faint = np.clip(m - med, 0, None)
    bright = np.clip(med - m, 0, None)
    asym = float((np.sum(faint) + 1e-9) / (np.sum(bright) + 1e-9))

    # Periodicity of the dips: strong, clean periodicity => eclipsing binary (mundane).
    best_p, power = _dip_periodicity(t, frac_dip, e)

    # Boyajian-likeness: deep + several dips + asymmetric + NOT strongly periodic.
    depth_term = np.clip((max_depth - depth_min) / 0.15, 0, 1)
    count_term = np.clip(n_dips / 10.0, 0, 1)
    asym_term = np.clip((asym - 1.0) / 3.0, 0, 1)
    aperiodic_term = float(np.clip(1.0 - power / 0.5, 0, 1))
    score = float(np.clip(0.4 * depth_term + 0.2 * count_term
                          + 0.2 * asym_term + 0.2 * aperiodic_term, 0, 1))
    return DipStats(n_epochs=int(t.size), max_depth=max_depth, n_dips=n_dips,
                    dip_duty_cycle=float(duty), asymmetry=asym,
                    best_period_d=float(best_p), period_power=float(power),
                    score=score)


def _dip_periodicity(t: np.ndarray, frac_dip: np.ndarray, e: np.ndarray) -> tuple:
    """Lomb-Scargle power of the dip signal; high, clean power => periodic (binary)."""
    if t.size < 30 or np.ptp(t) < 4:
        return np.nan, 0.0
    try:
        from astropy.timeseries import LombScargle
        y = np.clip(frac_dip, 0, None)
        if np.all(y == 0):
            return np.nan, 0.0
        ls = LombScargle(t, y)
        freq, power = ls.autopower(minimum_frequency=1.0 / min(np.ptp(t) / 2, 200.0),
                                   maximum_frequency=1.0 / 0.1, samples_per_peak=5)
        if power.size == 0:
            return np.nan, 0.0
        i = int(np.argmax(power))
        return 1.0 / float(freq[i]), float(power[i])
    except Exception:
        return np.nan, 0.0


__all__ = ["DipStats", "detect_dips"]

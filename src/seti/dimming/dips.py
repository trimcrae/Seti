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
    n_dip_events: int = 0   # discrete dip *events* sustained over >= 2 epochs
    out_of_dip_rms: float = 0.0  # robust scatter of the NON-dipped epochs (quiescence)
    max_event_depth: float = 0.0  # deepest fractional dip inside a sustained event

    def as_dict(self) -> dict:
        return {k: (float(v) if not isinstance(v, int) else int(v))
                for k, v in self.__dict__.items()}


def _dip_events(t: np.ndarray, is_dip: np.ndarray, frac_dip: np.ndarray,
                merge_gap_d: float = 5.0, min_run: int = 2) -> tuple[int, float]:
    """Discrete dip *events* and the deepest *sustained* dip.

    Contiguous dipped epochs (adjacent in the time-sorted series and within
    ``merge_gap_d`` days) form one event.  Only events spanning at least
    ``min_run`` epochs count: a single isolated dipped epoch is, at the faint end
    of a ground-based survey, far more likely a photometric outlier than a real
    occultation, which lasts long enough to be caught by several visits.

    Returns ``(n_events, max_event_depth)`` where ``max_event_depth`` is the
    deepest fractional dip occurring *inside* a valid (multi-epoch) event --- so a
    lone noisy point can never qualify a candidate by depth alone.
    """
    idx = np.flatnonzero(is_dip)
    if idx.size == 0:
        return 0, 0.0
    # Split the dipped-epoch indices into runs.
    runs: list[list[int]] = [[int(idx[0])]]
    for a, b in zip(idx[:-1], idx[1:], strict=False):
        if b != a + 1 or (t[b] - t[a]) > merge_gap_d:
            runs.append([int(b)])
        else:
            runs[-1].append(int(b))
    n_events = 0
    max_event_depth = 0.0
    for run in runs:
        if len(run) >= min_run:
            n_events += 1
            max_event_depth = max(max_event_depth, float(np.max(frac_dip[run])))
    return n_events, max_event_depth


def _count_events(t: np.ndarray, is_dip: np.ndarray, merge_gap_d: float = 5.0,
                  min_run: int = 2) -> int:
    """Back-compatible event count (see :func:`_dip_events`)."""
    frac = np.where(is_dip, 1.0, 0.0)
    return _dip_events(t, is_dip, frac, merge_gap_d=merge_gap_d, min_run=min_run)[0]


def _out_of_dip_rms(mag: np.ndarray, err: np.ndarray) -> float:
    """Noise-corrected fractional RMS of the non-dipped epochs (baseline quiescence).

    A Boyajian-like star is stable between dips (small RMS); a pulsator or
    eclipsing binary varies continuously (large RMS even outside the deepest dips).
    """
    m = np.asarray(mag, dtype=float)
    if m.size < 5:
        return 0.0
    med = np.median(m)
    mad = np.median(np.abs(m - med))
    sigma_mag = 1.4826 * mad if mad > 0 else float(np.std(m))
    if err is not None and np.size(err):
        med_err = float(np.nanmedian(err))
        sigma_mag = float(np.sqrt(max(sigma_mag**2 - med_err**2, 0.0)))
    return 0.4 * np.log(10.0) * sigma_mag   # magnitude scatter -> fractional flux RMS


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

    # Sort by time so contiguous dip epochs can be grouped into discrete events.
    order = np.argsort(t)
    t, m, e = t[order], m[order], e[order]

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
    # Discrete dip *events*: contiguous runs of dipped epochs, merged across short
    # sampling gaps, counting only events sustained over >= 2 epochs.  This is the
    # decisive separation of the Boyajian profile (a handful of deep, sustained
    # events) from both a high-amplitude *periodic* variable (hundreds of epochs)
    # and faint-end single-epoch photometric noise (lone outliers).
    n_events, max_event_depth = _dip_events(t, is_dip, frac_dip)
    # Quiescence: a Boyajian-like star is photometrically stable *between* dips,
    # whereas a pulsator/eclipsing binary varies continuously.  Measure the robust
    # scatter of the non-dipped epochs, de-noised by the typical photometric error.
    out_rms = _out_of_dip_rms(m[~is_dip], e[~is_dip])
    # Asymmetry about the *median* level: a Boyajian-like light curve has deep
    # excursions only to the faint side, so the faint-side absolute deviation far
    # exceeds the bright-side; a symmetric oscillation or flat noise gives ~1.
    med = float(np.nanmedian(m))
    faint = np.clip(m - med, 0, None)
    bright = np.clip(med - m, 0, None)
    asym = float((np.sum(faint) + 1e-9) / (np.sum(bright) + 1e-9))

    # Periodicity of the dips: strong, clean periodicity => eclipsing binary (mundane).
    best_p, power = _dip_periodicity(t, frac_dip, e)

    # Boyajian-likeness: deep + a *few discrete* events + asymmetric + quiescent
    # between dips + NOT strongly periodic.  The event term peaks for a handful of
    # events (~2-12) and is suppressed both for a single marginal dip and for the
    # hundreds-of-epochs signature of a continuous high-amplitude variable.
    # Depth is scored from the deepest *sustained* event, not a lone outlier.
    depth_term = np.clip((max_event_depth - depth_min) / 0.15, 0, 1)
    event_term = float(np.clip(n_events / 6.0, 0, 1)
                       * np.clip((40 - n_events) / 28.0, 0, 1))
    asym_term = np.clip((asym - 1.0) / 3.0, 0, 1)
    aperiodic_term = float(np.clip(1.0 - power / 0.5, 0, 1))
    # Quiescence: reward a stable out-of-dip baseline (small fractional RMS).
    quiescence_term = float(np.clip(1.0 - out_rms / 0.05, 0, 1))
    score = float(np.clip(0.34 * depth_term + 0.18 * event_term
                          + 0.18 * asym_term + 0.15 * aperiodic_term
                          + 0.15 * quiescence_term, 0, 1))
    return DipStats(n_epochs=int(t.size), max_depth=max_depth, n_dips=n_dips,
                    dip_duty_cycle=float(duty), asymmetry=asym,
                    best_period_d=float(best_p), period_power=float(power),
                    n_dip_events=int(n_events), out_of_dip_rms=float(out_rms),
                    max_event_depth=float(max_event_depth),
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

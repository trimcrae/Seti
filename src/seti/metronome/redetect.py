"""Flare re-detection in MAST light curves for the METRONOME shortlist.

The catalogue stages inherit somebody else's flare finder: its threshold,
its cadence, its quarter coverage.  For the stars that matter --- every star
at ``watch`` or better after the assess stage, plus the most active stars in
each catalogue --- this stage goes back to the pixels' own light curve, finds
the brief brightenings again with one simple, stated detector, and runs the
identical clock statistic on the re-detected peak times with the observing
windows read off the light curve itself (which is the one place the true
windows are known: every cadence that was downlinked is in the file).

Why this is worth a MAST round trip per star:

* a clock in the catalogue that is NOT in the light curve is a catalogue
  artefact (a pipeline that snapped times, a quarter-boundary duplication);
* a clock in the light curve at ticks the catalogue's threshold missed is the
  sub-threshold regime the catalogue search cannot see;
* the fraction of catalogued flares the detector recovers is the detector's
  own calibration, measured on the star in hand.

The detector (:func:`find_flares`) is the classical one: a running-median
baseline over ``detrend_window_days`` inside each contiguous run of cadences,
a robust (MAD) sigma per run, and a flare is ``>= n_consecutive`` consecutive
cadences above ``sigma_lo`` with the peak above ``sigma_hi``.  Its peak time
is the cadence of maximum residual.  Nothing about the catalogue's own
detection is reused, so the two measurements are independent.

Everything network-facing goes through :mod:`seti.growth.stage2`'s bounded
light-curve fetch (``lightkurve`` first, ``astroquery.mast`` + FITS when it is
absent), with an injectable ``lc_fn`` / ``kepler_lc_fn`` so the offline tests
never open a socket.
"""

from __future__ import annotations

import json
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import STATUS_FAILED, STATUS_OK, STATUS_ZERO, AcquisitionLog
from .clock import DEFAULT_SCAN, analyze_star
from .vet import quality_pass
from .windows import Windows

DEFAULT_REDETECT: dict = {
    "enabled": True,
    "max_stars": 40,                 # MAST round trips per run
    "tiers": ["candidate", "interest", "watch"],
    "top_by_events": 12,             # the most flare-rich stars per catalogue, tier or not
    "detrend_window_days": 0.5,
    "sigma_lo": 2.5,
    "sigma_hi": 3.5,
    "n_consecutive": 3,
    "gap_days": 0.5,                 # a run of cadences breaks at a gap this long
    "cadence": "long",               # Kepler: long cadence only (what the catalogues used)
    "budget_s": 5400.0,
    "per_target_budget_s": 900.0,
    "max_quarters": 60,
    "retries": 2,
    "match_tol_days": 0.1,           # a catalogued flare is "recovered" within this
    "period_tol": 0.02,
    "period_harmonics": [1.0, 0.5, 2.0, 1.0 / 3.0, 3.0],
    "confirm_p_max": 0.05,
    "confirm_Q_min": 0.85,
    "confirm_jitter_max": 0.05,
    # The photometric-periodicity veto: the longest period the flux is asked
    # about.  A re-detected "clock" at the star's own photometric period is
    # the residual of an oscillation the running median could not flatten.
    "phot_max_period_days": 50.0,
    # The threshold-free catalogue check (catalogue_epoch_response).  A star's
    # catalogued epochs must carry more flux than random epochs inside the same
    # windows before any clock built from them is a statement about the star.
    "epoch_n_control": 4000,
    "epoch_min_n": 8,          # below this the test has no power; it says None
    "epoch_alpha": 0.01,
    "epoch_sigma_min": 2.0,    # median peak residual at the epochs, in run sigmas
    # a catalogue in the wrong time system announces itself as a stack peak at
    # a non-zero shift rather than as an empty catalogue
    "epoch_offsets_days": [-2457000.0, -2400000.5, -2.0, -1.0, -0.5, -0.1,
                           0.1, 0.5, 1.0, 2.0, 2400000.5, 2457000.0],
}

STATUS_NO_LC = "NO_LIGHTCURVE"
STATUS_TOO_FEW = "TOO_FEW_FLARES"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# the detector (pure)
# ---------------------------------------------------------------------------
def contiguous_runs(t, *, gap_days: float) -> list[tuple[int, int]]:
    """``[(i0, i1)]`` half-open index ranges of cadences with no gap >= ``gap_days``."""
    t = np.asarray(t, dtype=float)
    if not len(t):
        return []
    breaks = np.where(np.diff(t) >= float(gap_days))[0] + 1
    edges = np.concatenate([[0], breaks, [len(t)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=False) if b > a]


def detrend_residuals(t, f, *, cadence_days: float, window_days: float = 0.5,
                      gap_days: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Residual from a running-median baseline, and a robust sigma per point.

    The median is taken over ``window_days / cadence_days`` cadences inside
    each contiguous run, so a flare a few cadences long does not move it; the
    sigma is ``1.4826 * MAD`` of the residual over the run, which a handful of
    flare points cannot inflate.
    """
    from scipy.ndimage import median_filter

    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    resid = np.full(len(t), np.nan)
    sig = np.full(len(t), np.nan)
    size = int(round(float(window_days) / max(float(cadence_days), 1e-6)))
    size = max(3, size | 1)
    for i0, i1 in contiguous_runs(t, gap_days=gap_days):
        seg = f[i0:i1]
        if len(seg) < 3:
            continue
        base = median_filter(seg, size=min(size, len(seg) | 1), mode="nearest")
        r = seg - base
        mad = float(np.nanmedian(np.abs(r - np.nanmedian(r))))
        s = 1.4826 * mad if mad > 0 else float(np.nanstd(r))
        if not np.isfinite(s) or s <= 0:
            continue
        resid[i0:i1] = r
        sig[i0:i1] = s
    return resid, sig


def find_flares(t, f, *, cadence_days: float, window_days: float = 0.5, sigma_lo: float = 2.5,
                sigma_hi: float = 3.5, n_consecutive: int = 3, gap_days: float = 0.5
                ) -> pd.DataFrame:
    """Brief brightenings: ``>= n_consecutive`` consecutive cadences above
    ``sigma_lo`` whose peak clears ``sigma_hi``.  Returns one row per flare:
    ``t_peak, t_start, t_end, amplitude, peak_sigma, n_points, equiv_dur_s``."""
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    order = np.argsort(t)
    t, f = t[order], f[order]
    resid, sig = detrend_residuals(t, f, cadence_days=cadence_days, window_days=window_days,
                                   gap_days=gap_days)
    z = np.where(np.isfinite(sig) & (sig > 0), resid / np.where(sig > 0, sig, 1.0), np.nan)
    above = np.isfinite(z) & (z > float(sigma_lo))
    rows = []
    n = len(t)
    i = 0
    max_step = 1.5 * float(cadence_days)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1] and (t[j + 1] - t[j]) <= max_step:
            j += 1
        if (j - i + 1) >= int(n_consecutive):
            k = i + int(np.nanargmax(resid[i:j + 1]))
            if z[k] >= float(sigma_hi):
                rows.append({"t_peak": float(t[k]), "t_start": float(t[i]), "t_end": float(t[j]),
                             "amplitude": float(resid[k]), "peak_sigma": float(z[k]),
                             "n_points": int(j - i + 1),
                             "equiv_dur_s": float(np.nansum(resid[i:j + 1]) * cadence_days
                                                  * 86400.0)})
        i = j + 1
    return pd.DataFrame(rows, columns=["t_peak", "t_start", "t_end", "amplitude", "peak_sigma",
                                       "n_points", "equiv_dur_s"])


def lightcurve_windows(t, *, cadence_days: float, gap_days: float = 0.5,
                       label: str = "lightcurve") -> Windows:
    """The observed windows as the light curve itself states them: every
    stretch of cadences, broken at any gap of ``gap_days`` or more."""
    t = np.sort(np.asarray(t, dtype=float))
    t = t[np.isfinite(t)]
    if not len(t):
        return Windows(np.zeros(0), np.zeros(0), cadence_days=cadence_days, label=label)
    starts, stops = [], []
    for i0, i1 in contiguous_runs(t, gap_days=gap_days):
        starts.append(float(t[i0]) - 0.5 * cadence_days)
        stops.append(float(t[i1 - 1]) + 0.5 * cadence_days)
    return Windows(np.array(starts), np.array(stops), cadence_days=cadence_days, label=label,
                   t_ref=float(t[0]))


def _segment_cadence_days(seg: dict) -> float:
    e = seg.get("exptime_s")
    try:
        e = float(e)
    except (TypeError, ValueError):
        e = float("nan")
    if np.isfinite(e) and e > 0:
        return e / 86400.0
    t = np.asarray(seg.get("time"), dtype=float)
    return float(np.median(np.diff(np.sort(t)))) if len(t) > 2 else float("nan")


def stitch_segments(segments, *, cadence: str = "long") -> tuple[np.ndarray, np.ndarray, dict]:
    """One normalised light curve from the fetched segments.

    ``cadence="long"`` keeps the >5-min products when any exist (the Kepler
    flare catalogues were built on long cadence, and mixing 1-min and 30-min
    sampling would give the detector two different sigmas on one star).  Each
    segment is divided by its own median.
    """
    segs = [s for s in (segments or []) if s is not None
            and len(np.asarray(s.get("time"), dtype=float))]
    if not segs:
        return np.zeros(0), np.zeros(0), {"n_segments": 0}
    cads = np.array([_segment_cadence_days(s) for s in segs])
    keep = np.ones(len(segs), dtype=bool)
    if cadence == "long" and np.isfinite(cads).any():
        long = cads > 300.0 / 86400.0
        if long.any():
            keep = long
    ts, fs = [], []
    for s, k in zip(segs, keep, strict=False):
        if not k:
            continue
        t = np.asarray(s.get("time"), dtype=float)
        f = np.asarray(s.get("flux"), dtype=float)
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok]
        med = float(np.median(f)) if len(f) else float("nan")
        if not np.isfinite(med) or med == 0:
            continue
        ts.append(t)
        fs.append(f / med)
    if not ts:
        return np.zeros(0), np.zeros(0), {"n_segments": 0}
    t = np.concatenate(ts)
    f = np.concatenate(fs)
    order = np.argsort(t)
    t, f = t[order], f[order]
    # duplicate cadences (a quarter delivered twice) collapse to one
    uniq = np.concatenate([[True], np.diff(t) > 1e-6])
    t, f = t[uniq], f[uniq]
    cad = float(np.nanmedian(cads[keep])) if keep.any() else float("nan")
    return t, f, {"n_segments": int(keep.sum()), "n_segments_fetched": len(segs),
                  "cadence_days": cad, "n_points": int(len(t)),
                  "segments": [int(s.get("sector")) if s.get("sector") is not None else None
                               for s, k in zip(segs, keep, strict=False) if k]}


# ---------------------------------------------------------------------------
# per star
# ---------------------------------------------------------------------------
def _period_agrees(p_new: float, p_cat: float, harmonics, tol: float) -> tuple[bool, float | None]:
    if not (np.isfinite(p_new) and np.isfinite(p_cat) and p_cat > 0):
        return False, None
    for h in harmonics:
        if abs(p_new / (p_cat * float(h)) - 1.0) <= float(tol):
            return True, float(h)
    return False, None


def photometric_period(t, f, *, cadence_days: float, min_period_days: float = 0.05,
                       max_period_days: float = 50.0, samples_per_peak: int = 8,
                       mask=None) -> dict:
    """The strongest periodicity in the star's own FLUX, flares masked out.

    This is the check the re-detection could not do without it.  The flare
    finder subtracts a running median over ``detrend_window_days`` (0.5 d) and
    calls positive excursions flares.  A running median of that length cannot
    remove a photometric oscillation whose period is comparable to it, and if
    that oscillation has a NARROW maximum --- a pulsator's sawtooth, a
    heartbeat brightening, a contact binary --- the bulk of the cycle sets the
    robust sigma, the peak clears it every cycle, and the detector returns a
    perfect clock at the photometric period.  (A pure sinusoid does not do
    this, and the test suite records that: its maxima sit at ~1.4 sigma of a
    sigma its own residual defines.)

    So the star's flux is asked directly.  ``period`` is the Lomb-Scargle peak
    of the normalised flux; ``amplitude_frac`` is the peak-to-peak of the
    best-fit sinusoid there, in units of the median flux.  When the star's
    DOMINANT photometric periodicity is the re-detected clock period (or a low
    harmonic of it), the events are that periodicity and not a flare clock.

    The flux is used unmasked, deliberately.  Masking the detected events would
    punch a hole at exactly the period under test and imprint it on the window
    function.  The cost is that a genuine flare clock also contributes some
    power at its own period; the size of that contribution is bounded by the
    events' duty cycle, which ``redetect_star`` reports beside this.
    """
    out = {"period": float("nan"), "power": float("nan"), "amplitude_frac": float("nan"),
           "n_points": 0}
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    if mask is not None:
        ok &= ~np.asarray(mask, dtype=bool)
    t, f = t[ok], f[ok]
    if len(t) < 50:
        return out
    med = float(np.nanmedian(f))
    if not np.isfinite(med) or med == 0.0:
        return out
    y = f / med - 1.0
    span = float(t[-1] - t[0])
    p_max = float(min(max_period_days, max(span / 3.0, 2.0 * cadence_days)))
    p_min = float(max(min_period_days, 2.0 * cadence_days))
    if not (p_max > p_min):
        return out
    try:
        from astropy.timeseries import LombScargle
    except Exception:                                     # noqa: BLE001
        return out
    ls = LombScargle(t, y)
    freq = np.arange(1.0 / p_max, 1.0 / p_min,
                     1.0 / (float(samples_per_peak) * max(span, 1.0)))
    if not len(freq):
        return out
    power = ls.power(freq)
    i = int(np.nanargmax(power))
    p = float(1.0 / freq[i])
    model = ls.model(t, freq[i])
    out.update({"period": p, "power": float(power[i]), "n_points": int(len(t)),
                "amplitude_frac": float(np.nanmax(model) - np.nanmin(model))})
    return out


def catalogue_epoch_response(t, f, epochs, windows: Windows, *, cadence_days: float,
                             window_days: float = 0.5, gap_days: float = 0.5,
                             tol_days: float | None = None, n_control: int = 4000,
                             offsets=None, rng=None) -> dict:
    """Does the light curve actually brighten at the CATALOGUED epochs?

    The re-detector answers a harder question than the channel needs.  It has
    its own threshold, and on a star where it recovers few of the catalogued
    flares a non-confirmation says as much about the detector as about the
    star.  This says nothing about thresholds: it reads the detrended
    residual, in units of the run's own robust sigma, at the times the
    catalogue put its flares, and compares that to the same statistic at
    random times inside the same observing windows.

    The question it settles is the one a clock built from catalogue times
    cannot answer for itself --- whether those times are events at all.  If
    the catalogue's epochs carry no more flux than random epochs do, then
    whatever pattern they form is a pattern in the catalogue, not in the star,
    and no amount of phase coherence changes that.

    ``offsets`` additionally stacks at a list of time shifts.  A catalogue
    whose times are in a different system, or shifted by a fixed amount,
    announces itself as a peak of the stack at a non-zero offset instead of
    looking like an empty catalogue.

    Returns the median and 90th percentile of the epoch sigma, the same for
    the controls, the fraction of epochs above 3 sigma against the control
    fraction, a one-sided empirical p for the median, and the best offset.
    """
    out = {"n_epochs": 0, "n_control": 0, "epoch_sigma_median": float("nan"),
           "epoch_sigma_p90": float("nan"), "control_sigma_median": float("nan"),
           "epoch_frac_above_3": float("nan"), "control_frac_above_3": float("nan"),
           "p_empirical": float("nan"), "best_offset_days": 0.0,
           "best_offset_sigma_median": float("nan"), "best_offset_n_epochs": 0}
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    ep = np.asarray(epochs, dtype=float)
    ep = ep[np.isfinite(ep)]
    if len(t) < 50 or not len(ep) or windows is None or not windows.n:
        return out
    # detrend_residuals reads gaps off np.diff, and _stack searchsorts, so
    # both need time order; the caller's stitched series already has it, but
    # this function is public and cheap to make safe
    order = np.argsort(t)
    t, f = t[order], f[order]
    resid, sig = detrend_residuals(t, f, cadence_days=cadence_days,
                                   window_days=window_days, gap_days=gap_days)
    z = resid / sig
    ok = np.isfinite(z)
    if ok.sum() < 50:
        return out
    tz, zz = t[ok], z[ok]
    tol = float(tol_days) if tol_days is not None else 1.5 * float(cadence_days)

    def _stack(times):
        """Peak sigma within +-tol of each time; NaN where nothing is covered."""
        lo = np.searchsorted(tz, times - tol, side="left")
        hi = np.searchsorted(tz, times + tol, side="right")
        vals = np.full(len(times), np.nan)
        for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
            if b > a:
                vals[i] = float(np.max(zz[a:b]))
        return vals

    ep_all = np.sort(ep)
    ep = ep_all[windows.contains(ep_all)]
    v = _stack(ep)
    v = v[np.isfinite(v)]
    rng = rng if rng is not None else np.random.default_rng(0)
    ctrl_t = np.sort(windows.sample(int(n_control), rng))
    cv = _stack(ctrl_t)
    cv = cv[np.isfinite(cv)]
    if not len(cv):
        return out
    # An empty epoch stack is still an answer -- and the commonest reason for
    # one is a time system, which the offset scan below is here to name.  So
    # the per-epoch statistics are filled only when there is something to fill
    # them with, and the scan runs either way.
    med = float(np.median(v)) if len(v) else float("nan")
    if len(v):
        # one-sided empirical p: how often does a random draw of len(v) control
        # epochs reach this median?  Bootstrapped from the control stack
        # itself, so it needs no distributional assumption about the residuals.
        n_boot = 400
        draws = rng.integers(0, len(cv), size=(n_boot, len(v)))
        boot = np.median(cv[draws], axis=1)
        out.update({
            "epoch_sigma_median": med,
            "epoch_sigma_p90": float(np.percentile(v, 90)),
            "epoch_frac_above_3": float(np.mean(v >= 3.0)),
            "p_empirical": float((1.0 + np.sum(boot >= med)) / (1.0 + n_boot)),
        })
    out.update({
        "n_epochs": int(len(v)), "n_control": int(len(cv)),
        "control_sigma_median": float(np.median(cv)),
        "control_frac_above_3": float(np.mean(cv >= 3.0)),
    })
    if offsets is not None and len(offsets):
        # the shift is applied to EVERY catalogued time, not only those already
        # inside a window: a catalogue in the wrong time system has none of its
        # epochs inside, and that is the case this is here to catch
        best_o = 0.0
        best_m = med if np.isfinite(med) else -np.inf
        best_n = int(len(v))
        for o in offsets:
            if o == 0.0:
                continue
            shifted = np.sort(ep_all + float(o))
            shifted = shifted[windows.contains(shifted)]
            if len(shifted) < max(4, len(ep_all) // 10):
                continue
            vo = _stack(shifted)
            vo = vo[np.isfinite(vo)]
            if len(vo) >= max(4, len(ep_all) // 10):
                m = float(np.median(vo))
                if m > best_m:
                    best_o, best_m, best_n = float(o), m, int(len(vo))
        out.update({"best_offset_days": best_o,
                    "best_offset_sigma_median": (float(best_m) if np.isfinite(best_m)
                                                 else float("nan")),
                    "best_offset_n_epochs": best_n})
    return out


def redetect_star(segments, conf: dict, *, scan_conf: dict | None = None,
                  null_conf: dict | None = None, vet_conf: dict | None = None, rng=None,
                  catalogue_times=None, period_catalogue: float = float("nan")) -> dict:
    """Detect flares in one star's light curve and run the clock test on them."""
    c = dict(DEFAULT_REDETECT, **(conf or {}))
    t, f, meta = stitch_segments(segments, cadence=str(c["cadence"]))
    rec: dict = {"lc_" + k: v for k, v in meta.items()}
    if not len(t):
        rec.update({"status": STATUS_NO_LC, "n_flares": 0})
        return rec
    cad = float(meta.get("cadence_days") or np.nan)
    if not np.isfinite(cad) or cad <= 0:
        cad = float(np.median(np.diff(t)))
    fl = find_flares(t, f, cadence_days=cad, window_days=float(c["detrend_window_days"]),
                     sigma_lo=float(c["sigma_lo"]), sigma_hi=float(c["sigma_hi"]),
                     n_consecutive=int(c["n_consecutive"]), gap_days=float(c["gap_days"]))
    w = lightcurve_windows(t, cadence_days=cad, gap_days=float(c["gap_days"]))
    rec.update({"n_flares": int(len(fl)), "lc_observed_days": round(w.total, 3),
                "lc_n_windows": w.n, "lc_span_days": round(w.span, 3),
                "lc_t_first": float(t[0]), "lc_t_last": float(t[-1])})
    # detector calibration against the catalogue's own list, on this star
    if catalogue_times is not None and len(catalogue_times):
        ct = np.sort(np.asarray(catalogue_times, dtype=float))
        ct = ct[np.isfinite(ct)]
        in_lc = w.contains(ct)
        rec["n_catalogue_flares_in_lc"] = int(in_lc.sum())
        if len(fl) and in_lc.any():
            fp = np.sort(fl["t_peak"].to_numpy())
            idx = np.clip(np.searchsorted(fp, ct[in_lc]), 1, len(fp) - 1) if len(fp) > 1 \
                else np.zeros(int(in_lc.sum()), dtype=int)
            near = np.minimum(np.abs(ct[in_lc] - fp[idx]),
                              np.abs(ct[in_lc] - fp[np.maximum(idx - 1, 0)]))
            rec["catalogue_recovery_frac"] = float(np.mean(near <= float(c["match_tol_days"])))
        else:
            rec["catalogue_recovery_frac"] = 0.0 if in_lc.any() else float("nan")
        # Threshold-free: is there flux at the catalogue's own epochs?  This
        # runs BEFORE the n_min gate, because it is exactly the stars whose
        # light curve yields too few flares for the re-detection that most
        # need an answer about their catalogue times.
        er = catalogue_epoch_response(
            t, f, ct, w, cadence_days=cad, window_days=float(c["detrend_window_days"]),
            gap_days=float(c["gap_days"]), n_control=int(c["epoch_n_control"]),
            offsets=list(c["epoch_offsets_days"]), rng=rng)
        rec.update({"cat_" + k: v for k, v in er.items()})
        n_ep = int(er["n_epochs"])
        if n_ep >= int(c["epoch_min_n"]):
            rec["catalogue_epochs_are_brightenings"] = bool(
                np.isfinite(er["p_empirical"])
                and er["p_empirical"] <= float(c["epoch_alpha"])
                and er["epoch_sigma_median"] >= float(c["epoch_sigma_min"]))
        else:
            rec["catalogue_epochs_are_brightenings"] = None
    if len(fl) < int(dict(DEFAULT_SCAN, **(scan_conf or {}))["n_min"]):
        rec.update({"status": STATUS_TOO_FEW})
        return rec
    a = analyze_star(fl["t_peak"].to_numpy(), w, fl["amplitude"].to_numpy(),
                     scan_conf or {}, null_conf or {}, rng)
    a.pop("windows", None)
    rec.update({("rd_" + k if k in ("n_events", "period", "Q", "jitter", "f_in_window",
                                      "gap_integer_frac", "n_gaps_used", "jitter_core",
                                      "n_core", "gap_integer_frac_core", "n_gaps_core",
                                      "h_max", "p_window", "p_window_source", "p_shuffle",
                                      "cycle_occupancy", "t0", "mean_phase", "null_computed",
                                      "n_freq")
                 else k): v for k, v in a.items()})
    rec["status"] = a.get("status", STATUS_OK)
    agrees, h = _period_agrees(float(a.get("period", np.nan)), float(period_catalogue),
                               c["period_harmonics"], float(c["period_tol"]))
    rec["period_catalogue"] = float(period_catalogue)
    rec["period_agrees_with_catalogue"] = bool(agrees)
    rec["period_harmonic_of_catalogue"] = h
    p = float(a.get("p_window", np.nan))
    # the SAME strict clock quality the catalogue tiers use (rms route or
    # core route), so "confirmed" means the light curve passes the gate the
    # catalogue passed, not a private one
    vconf = dict(vet_conf or {}, Q_min=float(c["confirm_Q_min"]),
                 jitter_max=float(c["confirm_jitter_max"]))
    ok_strict, why = quality_pass(a, vconf, strict=True)
    rec["rd_strict_quality_why"] = ";".join(why)
    # Is the "clock" the detrending residual of a photometric oscillation?
    # A running median over detrend_window_days cannot flatten a signal whose
    # period is comparable to it, and the surviving maxima are detected as a
    # flare train at exactly the photometric period.
    ph = photometric_period(t, f, cadence_days=cad,
                            max_period_days=float(c.get("phot_max_period_days", 50.0)))
    rec.update({"phot_period": ph["period"], "phot_power": ph["power"],
                "phot_amplitude_frac": ph["amplitude_frac"]})
    ph_hit, ph_h = _period_agrees(float(a.get("period", np.nan)), float(ph["period"]),
                                  c["period_harmonics"], float(c["period_tol"]))
    rec["period_is_photometric"] = bool(ph_hit)
    rec["period_photometric_harmonic"] = ph_h
    # Event SHAPE, the other discriminator between a flare and a photometric
    # maximum re-detected as one: a flare rises in about a cadence and decays
    # over several (rise_frac well below 0.5) and occupies a small part of the
    # cycle; a symmetric photometric maximum has rise_frac ~ 0.5 and a duty
    # cycle of order the oscillation's own width.  Reported, not vetoed on:
    # at 30-min cadence a 3-point event cannot resolve the asymmetry, so this
    # is evidence for a reader and for the docs, not a rule.
    dur = (fl["t_end"] - fl["t_start"]).to_numpy(dtype=float) + cad
    rise = (fl["t_peak"] - fl["t_start"]).to_numpy(dtype=float) + 0.5 * cad
    good = np.isfinite(dur) & (dur > 0)
    rec["rd_duration_days_median"] = float(np.median(dur[good])) if good.any() else float("nan")
    rec["rd_duty_cycle"] = (float(np.median(dur[good])) / float(a.get("period", np.nan))
                            if good.any() else float("nan"))
    rec["rd_rise_frac_median"] = float(np.median(rise[good] / dur[good])) if good.any() \
        else float("nan")
    rec["clock_in_lightcurve"] = bool(a.get("status") == "scanned" and np.isfinite(p)
                                      and p <= float(c["confirm_p_max"]) and ok_strict
                                      and not ph_hit)
    rec["confirms_catalogue_clock"] = bool(rec["clock_in_lightcurve"] and agrees)
    return rec


# ---------------------------------------------------------------------------
# target selection and the stage
# ---------------------------------------------------------------------------
def select_targets(vetted: pd.DataFrame, conf: dict, *, max_stars: int | None = None
                   ) -> list[dict]:
    """Every star at a listed tier, then the most flare-rich per catalogue."""
    c = dict(DEFAULT_REDETECT, **(conf or {}))
    cap = int(max_stars if max_stars is not None else c["max_stars"])
    if vetted is None or not len(vetted):
        return []
    df = vetted.copy()
    df["star_id"] = df["star_id"].astype(str)
    out: list[dict] = []
    seen: set = set()

    def _push(r, why):
        key = str(r.get("star_key"))
        if key in seen:
            return
        seen.add(key)
        out.append({"star_key": key, "star_id": str(r.get("star_id")),
                    "mission": str(r.get("mission")), "catalogue": str(r.get("catalogue")),
                    "tier": str(r.get("tier")), "why": why,
                    "n_events": int(r.get("n_events", 0) or 0),
                    "period_catalogue": float(r.get("period", np.nan)),
                    "p_window_catalogue": float(r.get("p_window", np.nan)),
                    "Q_catalogue": float(r.get("Q", np.nan)),
                    "jitter_catalogue": float(r.get("jitter", np.nan))})

    rank = {"candidate": 0, "interest": 1, "watch": 2}
    tiers = [t for t in c["tiers"] if t in rank]
    sel = df[df["tier"].isin(tiers)].copy() if "tier" in df else df.iloc[0:0]
    if len(sel):
        sel["_rank"] = sel["tier"].map(rank)
        sel = sel.sort_values(["_rank", "p_window"], na_position="last")
        for _, r in sel.iterrows():
            _push(r, f"tier:{r['tier']}")
    k = int(c["top_by_events"])
    if k > 0 and "n_events" in df:
        for cat, g in df.groupby("catalogue"):
            for _, r in g.sort_values("n_events", ascending=False).head(k).iterrows():
                _push(r, f"top_events:{cat}")
    return out[:cap] if cap > 0 else out


def _catalogue_times(out: Path, catalogue: str, star_id: str) -> np.ndarray:
    p = out / "data" / f"{catalogue}_events.parquet"
    if not p.exists():
        return np.zeros(0)
    try:
        ev = pd.read_parquet(p, columns=["star_id", "t_peak"])
    except Exception:                                     # noqa: BLE001
        return np.zeros(0)
    ev = ev[ev["star_id"].astype(str) == str(star_id)]
    return ev["t_peak"].to_numpy(dtype=float)


def stage_redetect(conf: dict, out: Path, *, lc_fn=None, kepler_lc_fn=None,
                   max_stars: int | None = None, seed: int = 20260921, log=None,
                   targets: list[dict] | None = None, budget_s: float | None = None) -> dict:
    """Re-detect flares for the shortlist and run the clock test on them."""
    from ..growth.stage2 import Deadline, MastParams, fetch_kepler_lightcurves, fetch_lightcurves

    rc = dict(DEFAULT_REDETECT, **(conf.get("redetect") or {}))
    sc, nc = conf.get("scan") or {}, conf.get("null") or {}
    log = log or AcquisitionLog(prefix="metronome/redetect")
    out = Path(out)
    if targets is None:
        vp = out / "stars_vetted.csv"
        vetted = pd.read_csv(vp, dtype={"star_id": str, "star_key": str}) if vp.exists() \
            else pd.DataFrame()
        targets = select_targets(vetted, rc, max_stars=max_stars)
    if max_stars is not None and max_stars > 0:
        targets = targets[:int(max_stars)]
    params = MastParams(per_target_budget_s=float(rc["per_target_budget_s"]),
                        kepler_per_target_budget_s=float(rc["per_target_budget_s"]),
                        max_sectors=int(rc["max_quarters"]),
                        kepler_max_quarters=int(rc["max_quarters"]),
                        retries=int(rc["retries"]), kepler_retries=int(rc["retries"]),
                        download_dir=str(out / "lc_cache"))
    deadline = Deadline(budget_s=float(budget_s if budget_s is not None else rc["budget_s"]))
    rng = np.random.default_rng(int(seed))
    records: list[dict] = []
    t_start = _time.monotonic()
    n_fetched = n_no_lc = n_failed = n_budget = 0
    for i, tg in enumerate(targets):
        rec = dict(tg)
        if deadline.expired():
            rec.update({"status": "NOT_ATTEMPTED", "note": "stage budget exhausted"})
            n_budget += 1
            records.append(rec)
            continue
        sid = str(tg["star_id"])
        mission = str(tg.get("mission", "")).lower()
        try:
            if mission.startswith("kep"):
                segs, status, route, _prov = fetch_kepler_lightcurves(
                    sid, lc_fn=kepler_lc_fn, params=params, log=log, deadline=deadline,
                    key=tg["star_key"])
            else:
                segs, status, route = fetch_lightcurves(
                    sid, lc_fn=lc_fn, params=params, log=log, deadline=deadline,
                    key=tg["star_key"])
        except Exception as exc:                          # noqa: BLE001
            segs, status, route = [], STATUS_FAILED, ""
            log.record(f"lightcurves_{tg['star_key']}", sid, error=repr(exc))
        rec["fetch_status"] = status
        rec["fetch_route"] = route
        if status != STATUS_OK:
            rec["status"] = STATUS_NO_LC if status == STATUS_ZERO else "FETCH_FAILED"
            if status == STATUS_ZERO:
                n_no_lc += 1
            else:
                n_failed += 1
            records.append(rec)
            continue
        n_fetched += 1
        ct = _catalogue_times(out, str(tg.get("catalogue")), sid)
        rec.update(redetect_star(segs, rc, scan_conf=sc, null_conf=nc,
                                 vet_conf=conf.get("vet") or {}, rng=rng, catalogue_times=ct,
                                 period_catalogue=float(tg.get("period_catalogue", np.nan))))
        records.append(rec)
        print(f"[metronome/redetect] {i + 1}/{len(targets)} {tg['star_key']}: {rec['status']} "
              f"n_flares={rec.get('n_flares')} P={rec.get('rd_period')} "
              f"p={rec.get('rd_p_window')} confirms={rec.get('confirms_catalogue_clock')} "
              f"({_time.monotonic() - t_start:.0f}s)")
        # checkpoint after every star
        pd.DataFrame(records).to_csv(out / "stars_redetect.csv", index=False)
    df = pd.DataFrame(records)
    if len(df):
        df.to_csv(out / "stars_redetect.csv", index=False)
    scanned = [r for r in records if r.get("status") == "scanned"]
    confirmed = [r for r in records if r.get("confirms_catalogue_clock")]
    lc_clocks = [r for r in records if r.get("clock_in_lightcurve")]
    photometric = [r for r in records if r.get("period_is_photometric")]
    if not targets:
        verdict = "NO_TARGETS"
    elif n_fetched == 0:
        verdict = "NO_DATA_REACHED" if n_failed else "NO_LIGHTCURVES_FOUND"
    elif confirmed:
        verdict = f"REDETECT_CONFIRMS_{len(confirmed)}"
    elif lc_clocks:
        verdict = f"LIGHTCURVE_CLOCK_WITHOUT_CATALOGUE_AGREEMENT_{len(lc_clocks)}"
    else:
        verdict = "REDETECT_CONFIRMS_NONE"
    epochs_absent = [r for r in records
                     if r.get("catalogue_epochs_are_brightenings") is False]
    if photometric:
        verdict += f"; PHOTOMETRIC_OSCILLATION_{len(photometric)}"
    if epochs_absent:
        verdict += f"; CATALOGUE_EPOCHS_ABSENT_{len(epochs_absent)}"
    rec_frac = [float(r.get("catalogue_recovery_frac", np.nan)) for r in records]
    rec_frac = [x for x in rec_frac if np.isfinite(x)]
    rep = {"stage": "redetect", "generated_utc": _now(), "verdict": verdict,
           "n_targets": len(targets), "n_fetched": n_fetched, "n_no_lightcurve": n_no_lc,
           "n_fetch_failed": n_failed, "n_not_attempted_budget": n_budget,
           "n_scanned": len(scanned), "n_too_few_flares": int(sum(
               1 for r in records if r.get("status") == STATUS_TOO_FEW)),
           "n_clock_in_lightcurve": len(lc_clocks), "n_confirms_catalogue_clock": len(confirmed),
           "n_period_is_photometric": len(photometric),
           "n_catalogue_epochs_absent": len(epochs_absent),
           "n_catalogue_epochs_confirmed": int(sum(
               1 for r in records if r.get("catalogue_epochs_are_brightenings") is True)),
           "catalogue_epoch_sigma_median": float(np.median([
               float(r["cat_epoch_sigma_median"]) for r in records
               if np.isfinite(float(r.get("cat_epoch_sigma_median", np.nan) or np.nan))]))
           if any(np.isfinite(float(r.get("cat_epoch_sigma_median", np.nan) or np.nan))
                  for r in records) else None,
           "catalogue_recovery_frac_median": float(np.median(rec_frac)) if rec_frac else None,
           "detector": {k: rc[k] for k in ("detrend_window_days", "sigma_lo", "sigma_hi",
                                           "n_consecutive", "gap_days", "cadence")},
           "targets": [{k: r.get(k) for k in (
               "star_key", "catalogue", "tier", "why", "status", "n_events", "n_flares",
               "n_catalogue_flares_in_lc", "catalogue_recovery_frac", "period_catalogue",
               "rd_period", "rd_Q", "rd_jitter", "rd_f_in_window", "rd_jitter_core",
               "rd_n_core", "rd_p_window", "rd_p_window_source",
               "rd_gap_integer_frac", "rd_gap_integer_frac_core", "rd_strict_quality_why",
               "period_agrees_with_catalogue",
               # the photometric-oscillation answer is the whole reason the
               # light curve was fetched; it belongs in the report, not only
               # in stars_redetect.csv
               "phot_period", "phot_power", "phot_amplitude_frac",
               "period_is_photometric", "period_photometric_harmonic",
               "rd_duty_cycle", "rd_rise_frac_median", "rd_duration_days_median",
               # the threshold-free question: is there flux at the catalogue's
               # own epochs at all?
               "catalogue_epochs_are_brightenings", "cat_n_epochs",
               "cat_epoch_sigma_median", "cat_control_sigma_median",
               "cat_epoch_frac_above_3", "cat_control_frac_above_3",
               "cat_p_empirical", "cat_best_offset_days",
               "cat_best_offset_sigma_median",
               "clock_in_lightcurve", "confirms_catalogue_clock", "lc_n_segments",
               "lc_observed_days", "fetch_route")} for r in records],
           "elapsed_s": round(_time.monotonic() - t_start, 1),
           "acquisition": log.as_dict(),
           "note": ("a catalogue clock is CONFIRMED only when an independent detector on the "
                    "star's own light curve finds a clock at the same period (or a low "
                    "harmonic) with strict quality; a light-curve clock the catalogue did not "
                    "show is reported separately and is not a candidate until vetted")}
    rep["reconciliation"] = reconcile_summary(out, records, verdict=verdict)
    (out / "redetect.json").write_text(json.dumps(rep, indent=2, default=_json_default))
    print(f"[metronome/redetect] {verdict}: {n_fetched}/{len(targets)} fetched, "
          f"{len(scanned)} scanned, {len(confirmed)} confirmed")
    if rep["reconciliation"].get("demoted"):
        print("[metronome/redetect] demoted by the light curve: "
              + ", ".join(rep["reconciliation"]["demoted"]))
    return rep


# ---------------------------------------------------------------------------
# reconciliation: the light curve has the last word on the channel verdict
# ---------------------------------------------------------------------------
#: What the light curve says about a shortlisted star, carried into
#: ``summary.json`` and ``candidates.json`` so the headline verdict cannot
#: claim a candidate the photometry has already explained.
RECONCILE_KEYS = ("status", "n_flares", "catalogue_recovery_frac", "rd_period",
                  "rd_p_window", "period_agrees_with_catalogue", "clock_in_lightcurve",
                  "confirms_catalogue_clock", "period_is_photometric", "phot_period",
                  "phot_amplitude_frac", "rd_duty_cycle", "rd_rise_frac_median",
                  "catalogue_epochs_are_brightenings", "cat_n_epochs",
                  "cat_epoch_sigma_median", "cat_control_sigma_median", "cat_p_empirical",
                  "cat_best_offset_days", "cat_best_offset_sigma_median")

#: The two ways the light curve can take a claim away, most mundane first.
#: ``catalogue_epochs_absent`` is first because it is the larger statement:
#: the events themselves are not in the photometry, so there is nothing for
#: the photometric veto to be about.
LIGHTCURVE_VETOES = ("catalogue_epochs_absent", "photometric_oscillation")


def _lightcurve_veto(rd: dict | None) -> str | None:
    """Which light-curve veto, if any, this star's re-detection record trips."""
    if not rd:
        return None
    if rd.get("catalogue_epochs_are_brightenings") is False:
        return "catalogue_epochs_absent"
    if rd.get("period_is_photometric"):
        return "photometric_oscillation"
    return None


def reconcile_summary(out: Path, records, *, verdict: str = "") -> dict:
    """Fold the light-curve answer back into the channel's own verdict files.

    The assess stage runs before any light curve is fetched, so a star can be
    called ``candidate`` in ``summary.json`` while its own photometry says the
    "clock" is the star's dominant flux oscillation re-detected as a flare
    train.  Left alone, the headline verdict would overclaim -- the redetect
    answer would sit in a second file nobody read.  So:

    * every shortlisted star at ``candidate`` or ``interest`` gets a
      ``redetect`` block in ``candidates.json``, including the honest
      ``not_attempted`` for stars the budget never reached;
    * a star whose re-detected period IS its photometric period is demoted to
      ``none`` with the named flag ``photometric_oscillation``, and the counts
      and the verdict string are recomputed from the demoted tiers.

    Demotion only ever removes a claim.  Nothing here can promote a star: the
    light curve confirming a clock is reported (``confirms_catalogue_clock``)
    and left for the vet, because confirmation is not the same as having
    passed the contamination gauntlet.
    """
    res: dict = {"status": "NO_SUMMARY", "n_annotated": 0, "demoted": []}
    sp, cp = out / "summary.json", out / "candidates.json"
    if not sp.exists():
        return res
    by_key = {str(r.get("star_key")): {k: r.get(k) for k in RECONCILE_KEYS}
              for r in records}
    try:
        summary = json.loads(sp.read_text())
    except (OSError, ValueError) as exc:                  # noqa: BLE001
        res["status"] = f"SUMMARY_UNREADABLE:{exc!r}"[:200]
        return res
    demoted: list[str] = []
    cands: list[dict] = []
    if cp.exists():
        try:
            cj = json.loads(cp.read_text())
        except (OSError, ValueError):
            cj = None
        if isinstance(cj, dict):
            for bucket in ("candidates", "watch"):
                for row in cj.get(bucket) or []:
                    key = str(row.get("star_key"))
                    rd = by_key.get(key)
                    row["redetect"] = rd or {"status": "not_attempted"}
                    veto = _lightcurve_veto(rd)
                    if veto and str(row.get("tier")) in ("candidate", "interest"):
                        row["tier"] = "none"
                        row["first_veto"] = veto
                        row["flags"] = ";".join(
                            [f for f in str(row.get("flags") or "").split(";") if f] + [veto])
                        demoted.append(f"{key}:{veto}")
                    if bucket == "candidates":
                        cands.append(row)
            cp.write_text(json.dumps(cj, indent=2, default=_json_default))
            res["n_annotated"] = sum(len(cj.get(b) or []) for b in ("candidates", "watch"))
    if demoted:
        summary["n_candidates"] = int(sum(1 for r in cands if r.get("tier") == "candidate"))
        summary["n_interest"] = int(sum(1 for r in cands if r.get("tier") == "interest"))
        f = summary.get("funnel") or {}
        f["stars_candidate"] = summary["n_candidates"]
        f["stars_interest"] = summary["n_interest"]
        f["stars_demoted_by_lightcurve"] = len(demoted)
        summary["funnel"] = f
        base = str(summary.get("verdict") or "")
        summary["verdict"] = f"{base}; REDETECT_DEMOTED_{len(demoted)}"
    summary["redetect"] = {
        "verdict": str(verdict), "n_demoted": len(demoted), "demoted": demoted,
        "vetoes": list(LIGHTCURVE_VETOES), "per_star": by_key,
        "note": ("the light curve has the last word.  catalogue_epochs_absent: the flux at "
                 "the catalogue's own epochs is no higher than at random epochs in the same "
                 "windows, so the events the clock is built from are not in the photometry.  "
                 "photometric_oscillation: the re-detected period IS the star's dominant "
                 "photometric period, so the 'flares' are a detrending residual.  Either "
                 "demotes, whatever the catalogue statistics said"),
    }
    sp.write_text(json.dumps(summary, indent=2, default=_json_default))
    res.update({"status": "OK", "demoted": demoted})
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


__all__ = ["DEFAULT_REDETECT", "STATUS_NO_LC", "STATUS_TOO_FEW", "contiguous_runs",
           "detrend_residuals", "find_flares", "lightcurve_windows", "photometric_period",
           "reconcile_summary", "redetect_star", "select_targets", "stage_redetect",
           "stitch_segments"]

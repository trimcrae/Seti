"""GROWTH stage 2 --- measure the TESS depth from the LIGHT CURVE, not a catalogue.

Why this module exists
----------------------
Stage 1 (:mod:`seti.growth.drift`) compares two **heterogeneous catalogue**
numbers: the Kepler-era ``cumulative.koi_depth`` (a DV fit on Q1--Q17 DR25
long-cadence PDCSAP) against the TESS-era ``toi.pl_trandep`` (whichever of
SPOC / TESS-SPOC / QLP delivered the TOI, on a different cadence, with a
different crowding model, revised as sectors accumulate).  Run 35038510064
returned exactly one ``GROWTH_CANDIDATE`` --- Kepler-718 b (K00897.01, KIC
7849854, TIC 268924036, TOI 4490.01), 14,281 +/- 18 ppm in the Kepler era
against 34,476 +/- 2,349 ppm in the TOI table, ``z = 8.67`` --- and that run's
own population says the ``z`` cannot be read as a sigma count: **31 of 108
measured planets sit above 5 sigma and 37 above 3 sigma** of the population
median.  A sample in which a third of the objects are five-sigma outliers has
formal errors that do not describe the pipeline-to-pipeline scatter.

So stage 2 does not argue about the error model.  It **measures the TESS depth
itself** and then compares THREE numbers instead of two:

======================================  =========================================
``depth_kepler_ppm``                    the KOI catalogue depth (Kepler band)
``depth_toi_ppm``                       the TOI catalogue depth (TESS band)
``depth_measured_ppm``                  this module's fit to the TESS light curve
======================================  =========================================

with the Kepler depth carried into the TESS band by the limb-darkening band
ratio :func:`seti.growth.drift.band_ratio` before it is compared (the two
catalogues are in different bandpasses; comparing them raw would charge the
band difference to "growth").  The outcome is one of

``MEASURED_DEPTH_MATCHES_KEPLER``
    the measured TESS depth agrees with the Kepler-era depth and disagrees with
    the TOI value: **the TOI number was wrong**, this candidate dies, and the
    channel has learned that its stage-1 error model is catalogue-driven.
``MEASURED_DEPTH_MATCHES_TOI``
    the measured depth agrees with the deep TOI value and disagrees with the
    Kepler-era depth: the depth really did change, and the candidate survives
    to the centroid test (an eclipsing binary on a different star inside the
    TESS pixel makes a transit DEEPER and is *not* excluded by anything here).
``MEASURED_DEPTH_MATCHES_NEITHER``
    the measurement sits outside both.  Said, not resolved.
``MEASURED_DEPTH_MATCHES_BOTH``
    the two catalogue values are not separated by this measurement's precision.
    This is the "cannot distinguish" branch, and it is **never** a pick: it is
    reported with ``z_toi_vs_kepler`` so a reader can see whether the failure
    is the measurement's precision or the references' own agreement.
``UNMEASURED``
    no depth was fitted.  ``unmeasured_reason`` keeps ``QUERY_FAILED``
    (the archive did not answer) and ``QUERY_RETURNED_ZERO_ROWS`` (it answered
    with nothing) apart, and adds ``NO_USABLE_TRANSIT`` / ``BUDGET_EXHAUSTED``
    / ``EPHEMERIS_UNAVAILABLE``.  **A target whose light curve could not be
    fetched is never reported as agreeing with anything.**

The ephemeris, and the one arithmetic that must not be got wrong
----------------------------------------------------------------
The period and epoch are the KOI's own (``koi_period``, ``koi_time0bk``), and
the two missions count days from different zero points:

* Kepler **BKJD** = BJD - 2454833.0
* TESS  **BTJD** = BJD - 2457000.0

so ``t_BTJD = t_BKJD - 2167.0`` exactly (:data:`BKJD_MINUS_BTJD`).  A KOI epoch
of 170.0 BKJD is BJD 2455003.0 is **-1997.0 BTJD** --- the hand-worked value the
test suite checks.  The epoch is then propagated forward by an integer number
of periods to the TESS window, and the accumulated uncertainty
``sqrt(sigma_T0^2 + (n sigma_P)^2)`` is reported in minutes beside the
duration: an ephemeris whose drift is a sizeable fraction of the transit makes
a fitted depth shallow, and that has to be visible rather than absorbed.

The fit
-------
Deliberately simple, and fixed to the KOI solution --- this module measures a
**depth**, it does not re-derive an ephemeris.  Per target:

1. each sector is normalised separately by its own robust median;
2. every predicted transit gets a local window ``|dt| <= w T14``; the baseline
   is fitted as a straight line in ``dt`` over the out-of-transit part of that
   window only (the transit is **masked**, with a guard band);
3. the depth of that transit is ``1 - <flux/baseline>`` over a **core** window
   ``|dt| <= f T14 / 2`` shrunk by half the exposure time, so a point whose
   integration straddles ingress is not counted as flat-bottom flux;
4. per-transit depths are combined by inverse variance; the quoted error is a
   **bootstrap over transits** when there are enough of them and the analytic
   propagation otherwise, and ``depth_err_method`` always says which;
5. the depth is reported **per sector** as well as combined --- a depth that
   differs between sectors is a systematic, not growth --- along with the
   out-of-transit scatter and the **odd-even** depth difference, which is the
   classic eclipsing-binary signature and is nearly free once the fold exists.

A cadence that smears the transit (30-minute FFI photometry on a 2-hour
transit) is **flagged**, never quietly corrected: ``smeared`` and
``exptime_over_duration`` are columns, and ``depth_is_lower_bound`` marks the
case where no flat core survives the integration at all.

Everything that touches a service takes an injectable callable (``query_fn``
for the Exoplanet Archive, ``lc_fn`` for the light curves) and carries a
**wall-clock budget**, because an unbounded stage has already cost this channel
a whole run (``config/growth.yaml``, ``gaia.cone_budget_s``).  Nothing here
reaches the network inside a test.
"""

from __future__ import annotations

import argparse
import json
import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import STATUS_FAILED, STATUS_OK, STATUS_ZERO, AcquisitionLog, tap_sync
from .drift import band_ratio, sym_err

# ---------------------------------------------------------------------------
# Time systems.  BKJD = BJD - 2454833; BTJD = BJD - 2457000.
# ---------------------------------------------------------------------------
#: Kepler barycentric Julian date offset (``koi_time0bk`` is in BKJD).
BKJD_OFFSET = 2454833.0
#: TESS barycentric Julian date offset (TESS light-curve ``TIME`` is in BTJD).
BTJD_OFFSET = 2457000.0
#: ``t_BTJD = t_BKJD + BKJD_MINUS_BTJD``.  Exactly -2167.0 days.
BKJD_MINUS_BTJD = BKJD_OFFSET - BTJD_OFFSET

# ---------------------------------------------------------------------------
# Verdict vocabulary.  Per target (three-way), then per run.
# ---------------------------------------------------------------------------
MATCH_KEPLER = "MEASURED_DEPTH_MATCHES_KEPLER"
MATCH_TOI = "MEASURED_DEPTH_MATCHES_TOI"
MATCH_NEITHER = "MEASURED_DEPTH_MATCHES_NEITHER"
MATCH_BOTH = "MEASURED_DEPTH_MATCHES_BOTH"
MATCH_UNMEASURED = "UNMEASURED"
TARGET_VERDICTS = (MATCH_KEPLER, MATCH_TOI, MATCH_NEITHER, MATCH_BOTH, MATCH_UNMEASURED)

#: Why a target has no depth.  ``QUERY_FAILED`` (the service errored) and
#: ``QUERY_RETURNED_ZERO_ROWS`` (it answered with nothing) are different facts
#: and are never collapsed.
UNMEASURED_QUERY_FAILED = STATUS_FAILED
UNMEASURED_ZERO_ROWS = STATUS_ZERO
UNMEASURED_NO_EPHEMERIS = "EPHEMERIS_UNAVAILABLE"
UNMEASURED_NO_TRANSIT = "NO_USABLE_TRANSIT"
UNMEASURED_BUDGET = "BUDGET_EXHAUSTED"
#: Neither catalogue depth is usable, so there is nothing to agree or disagree
#: with.  The measured depth is still reported --- what is missing is the
#: comparison, not the measurement, and "matches neither" would be a false
#: statement when there is no "neither" to match.
UNMEASURED_NO_REFERENCE = "NO_REFERENCE_DEPTH"
UNMEASURED_REASONS = (UNMEASURED_QUERY_FAILED, UNMEASURED_ZERO_ROWS, UNMEASURED_NO_EPHEMERIS,
                      UNMEASURED_NO_TRANSIT, UNMEASURED_BUDGET, UNMEASURED_NO_REFERENCE)

RUN_NO_DATA = "NO_DATA_REACHED"
RUN_REFUTED = "STAGE1_DEPTH_CHANGE_REFUTED"
RUN_CONFIRMED = "STAGE1_DEPTH_CHANGE_CONFIRMED"
RUN_UNRESOLVED = "STAGE1_DEPTH_CHANGE_UNRESOLVED"
RUN_VERDICTS = (RUN_NO_DATA, RUN_REFUTED, RUN_CONFIRMED, RUN_UNRESOLVED)

STAGES = ("probe", "measure", "assess")

#: Light-curve authors, best cadence first.  SPOC 2-minute where it exists,
#: TESS-SPOC / QLP FFI photometry otherwise --- and which one supplied each
#: sector is recorded per target, because a 30-minute FFI depth on a 2-hour
#: transit is a different measurement from a 2-minute one.
DEFAULT_AUTHORS: tuple[str, ...] = ("SPOC", "TESS-SPOC", "QLP")

#: Flux columns tried in a TESS light-curve FITS product, in order.  SPOC and
#: TESS-SPOC give ``PDCSAP_FLUX``; QLP has used ``KSPSAP_FLUX`` and, in later
#: deliveries, ``DET_FLUX``.  Which one was read is recorded per sector.
FLUX_COLUMNS: tuple[str, ...] = ("PDCSAP_FLUX", "KSPSAP_FLUX", "DET_FLUX", "SAP_FLUX")

ROUTE_LIGHTKURVE = "lightkurve"
ROUTE_MAST_FITS = "astroquery_mast_fits"

KOI_EPH_COLUMNS: tuple[str, ...] = (
    "kepoi_name", "kepid", "kepler_name", "koi_disposition", "koi_period", "koi_period_err1",
    "koi_period_err2", "koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2", "koi_duration",
    "koi_duration_err1", "koi_duration_err2", "koi_depth", "koi_depth_err1", "koi_depth_err2",
    "koi_ror", "koi_impact", "koi_dor", "koi_steff", "koi_slogg", "ra", "dec",
)
TOI_DEPTH_COLUMNS: tuple[str, ...] = (
    "toi", "tid", "tfopwg_disp", "pl_orbper", "pl_trandep", "pl_trandeperr1", "pl_trandeperr2",
    "pl_trandurh", "pl_trandurherr1", "pl_trandurherr2",
)


# ---------------------------------------------------------------------------
# Epoch arithmetic
# ---------------------------------------------------------------------------
def bkjd_to_btjd(t_bkjd):
    """Kepler BKJD -> TESS BTJD.  ``t - 2167.0`` (BJD - 2454833 -> BJD - 2457000)."""
    return np.asarray(t_bkjd, dtype=float) + BKJD_MINUS_BTJD if np.ndim(t_bkjd) else (
        float(t_bkjd) + BKJD_MINUS_BTJD)


def btjd_to_bkjd(t_btjd):
    """TESS BTJD -> Kepler BKJD."""
    return np.asarray(t_btjd, dtype=float) - BKJD_MINUS_BTJD if np.ndim(t_btjd) else (
        float(t_btjd) - BKJD_MINUS_BTJD)


def bkjd_to_bjd(t_bkjd):
    """Kepler BKJD -> full BJD."""
    return np.asarray(t_bkjd, dtype=float) + BKJD_OFFSET if np.ndim(t_bkjd) else (
        float(t_bkjd) + BKJD_OFFSET)


def btjd_to_bjd(t_btjd):
    """TESS BTJD -> full BJD."""
    return np.asarray(t_btjd, dtype=float) + BTJD_OFFSET if np.ndim(t_btjd) else (
        float(t_btjd) + BTJD_OFFSET)


def propagate_epoch(t0_btjd: float, period_days: float, t_ref_btjd: float, *,
                    t0_err_days: float = float("nan"),
                    period_err_days: float = float("nan")) -> dict:
    """Carry a transit epoch forward (or back) to the epoch nearest ``t_ref_btjd``.

    Returns the nearest mid-transit time, the integer number of periods
    crossed, and the **accumulated** ephemeris uncertainty
    ``sqrt(sigma_T0^2 + (n sigma_P)^2)`` in days and minutes.  The KOI epoch is
    ~2010 and the TESS window is ~2019--2026, so ``n`` is thousands for a
    short-period planet and a 1e-7 d period error still accumulates only
    seconds --- but it is reported rather than assumed.
    """
    p = float(period_days)
    if not (np.isfinite(t0_btjd) and np.isfinite(p) and p > 0):
        return {"t0_btjd": float("nan"), "n_epochs": 0, "sigma_days": float("nan"),
                "sigma_minutes": float("nan")}
    n = int(round((float(t_ref_btjd) - float(t0_btjd)) / p))
    t0n = float(t0_btjd) + n * p
    s0 = float(t0_err_days) if np.isfinite(t0_err_days) else 0.0
    sp = float(period_err_days) if np.isfinite(period_err_days) else 0.0
    sig = math.sqrt(s0 * s0 + (abs(n) * sp) ** 2)
    return {"t0_btjd": t0n, "n_epochs": n, "sigma_days": sig, "sigma_minutes": sig * 1440.0}


def fold(time_btjd, period_days: float, t0_btjd: float) -> tuple[np.ndarray, np.ndarray]:
    """``(epoch, dt)``: the integer transit number and the signed days from its centre."""
    t = np.asarray(time_btjd, dtype=float)
    p = float(period_days)
    epoch = np.round((t - float(t0_btjd)) / p)
    dt = t - (float(t0_btjd) + epoch * p)
    return epoch.astype(int), dt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class FitParams:
    """Everything the depth fit is allowed to decide (all from ``config/growth.yaml``)."""

    core_fraction: float = 0.7          # core window = core_fraction * T14, centred
    baseline_window_durations: float = 1.5   # local window half-width, in T14
    baseline_guard_durations: float = 0.75   # baseline must lie beyond this, in T14
    min_core_points: int = 2
    min_baseline_points: int = 8
    require_baseline_both_sides: bool = True
    smear_fraction: float = 0.2         # exptime > this * T14 -> `smeared`
    bootstrap_draws: int = 2000
    min_transits_for_bootstrap: int = 8
    err_method: str = "auto"            # auto | bootstrap | analytic | chi2_scaled
    fold_bins: int = 121
    odd_even_sigma: float = 3.0         # |odd - even| / sigma at or above this is flagged
    sector_scatter_chi2_per_dof: float = 3.0
    seed: int = 7

    @classmethod
    def from_config(cls, conf: dict | None) -> FitParams:
        f = ((conf or {}).get("stage2") or {}).get("fit") or {}
        d = cls()
        for k in d.__dataclass_fields__:
            if k in f and f[k] is not None:
                setattr(d, k, type(getattr(d, k))(f[k]))
        return d


@dataclass
class CompareParams:
    """The three-way comparison's thresholds."""

    n_agree: float = 3.0                # |z| below this is "agrees with"
    sigma_sys_ln: float = 0.05          # stated floor on ln-depth heterogeneity
    apply_band_ratio: bool = True       # carry the Kepler depth into the TESS band

    @classmethod
    def from_config(cls, conf: dict | None) -> CompareParams:
        c = ((conf or {}).get("stage2") or {}).get("compare") or {}
        d = cls()
        for k in d.__dataclass_fields__:
            if k in c and c[k] is not None:
                setattr(d, k, type(getattr(d, k))(c[k]))
        return d


@dataclass
class MastParams:
    """Wall-clock ceilings on every network stage, and which products to ask for."""

    authors: tuple[str, ...] = DEFAULT_AUTHORS
    target_timeout_s: float = 600.0     # one target's whole light-curve fetch
    per_target_budget_s: float = 900.0
    budget_s: float = 5400.0            # the WHOLE measure stage
    retries: int = 2
    retry_pause_s: float = 5.0          # backoff between attempts, bounded by the budget
    max_sectors: int = 60
    quality_bitmask: str = "default"
    download_dir: str | None = None
    archive_timeout_s: float = 300.0    # the Exoplanet Archive ephemeris pulls
    archive_retries: int = 3

    @classmethod
    def from_config(cls, conf: dict | None) -> MastParams:
        m = ((conf or {}).get("stage2") or {}).get("mast") or {}
        d = cls()
        if m.get("authors"):
            d.authors = tuple(str(a) for a in m["authors"])
        for k in ("target_timeout_s", "per_target_budget_s", "budget_s", "archive_timeout_s",
                  "retry_pause_s"):
            if m.get(k) is not None:
                setattr(d, k, float(m[k]))
        for k in ("retries", "max_sectors", "archive_retries"):
            if m.get(k) is not None:
                setattr(d, k, int(m[k]))
        for k in ("quality_bitmask", "download_dir"):
            if m.get(k) is not None:
                setattr(d, k, str(m[k]))
        return d


@dataclass
class Deadline:
    """A monotonic wall-clock ceiling.  ``None`` budget means no ceiling."""

    budget_s: float | None = None
    started: float = field(default_factory=_time.monotonic)

    def expired(self) -> bool:
        return self.budget_s is not None and (_time.monotonic() - self.started) > self.budget_s

    def remaining(self) -> float:
        if self.budget_s is None:
            return float("inf")
        return max(0.0, self.budget_s - (_time.monotonic() - self.started))

    def elapsed(self) -> float:
        return _time.monotonic() - self.started


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mad = float(np.median(np.abs(x - np.median(x))))
    s = 1.4826 * mad
    return s if s > 0 else float(np.std(x, ddof=1))


def _wls_line(dt: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least squares ``y = a + b dt``.  Falls back to the weighted mean."""
    sw = float(np.sum(w))
    if sw <= 0 or dt.size < 2:
        return (float(np.mean(y)) if y.size else float("nan")), 0.0
    sx = float(np.sum(w * dt))
    sy = float(np.sum(w * y))
    sxx = float(np.sum(w * dt * dt))
    sxy = float(np.sum(w * dt * y))
    den = sw * sxx - sx * sx
    if not np.isfinite(den) or abs(den) < 1e-30:
        return sy / sw, 0.0
    b = (sw * sxy - sx * sy) / den
    a = (sy - b * sx) / sw
    return float(a), float(b)


def core_half_width(duration_days: float, exptime_days: float, params: FitParams
                    ) -> tuple[float, bool]:
    """The half-width of the flat-bottom window, shrunk by half the exposure.

    A point integrated over ``exptime`` reports the mean flux across that
    interval, so a point whose interval straddles ingress is not flat-bottom
    flux.  Returns ``(half_width_days, no_flat_core)``; ``no_flat_core`` is
    True when the integration is long enough that nothing survives the shrink,
    in which case the unshrunk window is used and the depth is a **lower
    bound** on the true one.
    """
    half = 0.5 * float(params.core_fraction) * float(duration_days)
    e = float(exptime_days) if np.isfinite(exptime_days) else 0.0
    shrunk = half - 0.5 * e
    if shrunk <= 0:
        return half, True
    return shrunk, False


def fit_transits(time_btjd, flux, flux_err, *, period_days: float, t0_btjd: float,
                 duration_days: float, exptime_days: float = float("nan"),
                 sector: int | None = None, params: FitParams | None = None) -> list[dict]:
    """Per-transit depths with a locally fitted, transit-masked baseline.

    The ephemeris is FIXED.  For each predicted transit inside the data: fit
    ``a + b dt`` to the out-of-transit part of the local window (the transit is
    masked with a guard band), divide the core points by it, and take
    ``1 - <ratio>`` as that transit's depth.  A transit without enough baseline
    or core points is skipped and says why.
    """
    params = params or FitParams()
    t = np.asarray(time_btjd, dtype=float)
    f = np.asarray(flux, dtype=float)
    fe = (np.asarray(flux_err, dtype=float) if flux_err is not None
          else np.full(t.shape, np.nan, dtype=float))
    ok = np.isfinite(t) & np.isfinite(f)
    t, f, fe = t[ok], f[ok], fe[ok]
    if t.size == 0 or not (np.isfinite(period_days) and period_days > 0
                           and np.isfinite(duration_days) and duration_days > 0):
        return []
    epoch, dt = fold(t, period_days, t0_btjd)
    half_core, _no_flat = core_half_width(duration_days, exptime_days, params)
    w_guard = float(params.baseline_guard_durations) * float(duration_days)
    w_out = float(params.baseline_window_durations) * float(duration_days)
    exp_d = float(exptime_days) if np.isfinite(exptime_days) and exptime_days > 0 else 0.0
    # THE CADENCE SETS THE WINDOW, not the other way round.  A window fixed in
    # units of T14 holds plenty of baseline points at 2-minute cadence and far
    # too few at 30-minute FFI cadence, where the transit would then be thrown
    # away for want of a baseline rather than measured and flagged.  So the
    # outer window is widened, when it has to be, until it can hold the
    # required number of baseline samples at the ACTUAL cadence.
    if exp_d > 0:
        need_per_side = 0.5 * int(params.min_baseline_points) + 1.0
        w_out = max(w_out, w_guard + need_per_side * exp_d)
    # Likewise the core requirement: a 30-minute integration cannot put two
    # samples inside a 40-minute flat bottom however much one would like it to.
    # The requirement drops to one sample when the cadence cannot supply more
    # --- such a sector is already carrying the `smeared` flag.
    min_core = int(params.min_core_points)
    if exp_d > 0:
        min_core = max(1, min(min_core, int(math.floor(2.0 * half_core / exp_d))))
    rows: list[dict] = []
    for ep in np.unique(epoch):
        m = (epoch == ep) & (np.abs(dt) <= w_out)
        if not np.any(m):
            continue
        d_i, f_i, e_i = dt[m], f[m], fe[m]
        core = np.abs(d_i) <= half_core
        base = np.abs(d_i) >= w_guard
        rec: dict = {"epoch": int(ep), "sector": sector,
                     "t_mid_btjd": float(t0_btjd + ep * period_days),
                     "n_core": int(core.sum()), "n_baseline": int(base.sum()),
                     "n_baseline_before": int((base & (d_i < 0)).sum()),
                     "n_baseline_after": int((base & (d_i > 0)).sum()),
                     "depth": float("nan"), "depth_err": float("nan"),
                     "oot_scatter": float("nan"), "used": False, "reject_reason": "",
                     "half_core_days": float(half_core), "window_days": float(w_out),
                     "min_core_points_required": int(min_core)}
        if rec["n_core"] < min_core:
            rec["reject_reason"] = "too_few_core_points"
            rows.append(rec)
            continue
        if rec["n_baseline"] < int(params.min_baseline_points):
            rec["reject_reason"] = "too_few_baseline_points"
            rows.append(rec)
            continue
        if params.require_baseline_both_sides and (rec["n_baseline_before"] == 0
                                                   or rec["n_baseline_after"] == 0):
            rec["reject_reason"] = "baseline_on_one_side_only"
            rows.append(rec)
            continue
        wb = np.where(np.isfinite(e_i[base]) & (e_i[base] > 0), 1.0 / np.maximum(
            e_i[base], 1e-30) ** 2, 1.0)
        a, b = _wls_line(d_i[base], f_i[base], wb)
        pred_base = a + b * d_i[base]
        pred_core = a + b * d_i[core]
        if not np.all(np.isfinite(pred_core)) or np.any(pred_core == 0):
            rec["reject_reason"] = "baseline_fit_failed"
            rows.append(rec)
            continue
        resid = f_i[base] / pred_base - 1.0
        sig = _robust_sigma(resid)
        ratio = f_i[core] / pred_core
        depth = 1.0 - float(np.mean(ratio))
        if np.isfinite(sig):
            err = sig * math.sqrt(1.0 / rec["n_core"] + 1.0 / rec["n_baseline"])
        else:
            err = float("nan")
        rec.update({"depth": depth, "depth_err": err, "oot_scatter": sig, "used": True})
        rows.append(rec)
    return rows


def combine_transit_depths(rows, *, params: FitParams | None = None,
                           rng: np.random.Generator | None = None) -> dict:
    """Inverse-variance combination of per-transit depths, with a stated error.

    Reports the analytic propagation, the chi2-scaled version and a bootstrap
    over transits; ``depth_err_method`` names the one promoted to
    ``depth_err_ppm``.  ``auto`` bootstraps once there are enough transits for
    the resampling to mean anything and falls back to the analytic error below
    that.
    """
    params = params or FitParams()
    used = [r for r in rows if r.get("used") and np.isfinite(r.get("depth", np.nan))
            and np.isfinite(r.get("depth_err", np.nan)) and r["depth_err"] > 0]
    out = {"n_transits": int(len(used)), "n_transits_seen": int(len(rows)),
           "depth_ppm": float("nan"), "depth_err_ppm": float("nan"),
           "depth_err_analytic_ppm": float("nan"), "depth_err_chi2_scaled_ppm": float("nan"),
           "depth_err_bootstrap_ppm": float("nan"), "depth_err_method": "none",
           "chi2": float("nan"), "chi2_per_dof": float("nan"),
           "oot_scatter_ppm": float("nan")}
    if not used:
        return out
    d = np.array([r["depth"] for r in used], dtype=float)
    e = np.array([r["depth_err"] for r in used], dtype=float)
    w = 1.0 / e**2
    mean = float(np.sum(w * d) / np.sum(w))
    err = float(1.0 / math.sqrt(np.sum(w)))
    chi2 = float(np.sum(((d - mean) / e) ** 2))
    dof = max(len(d) - 1, 1)
    out.update({"depth_ppm": mean * 1e6, "depth_err_analytic_ppm": err * 1e6,
                "chi2": chi2, "chi2_per_dof": chi2 / dof,
                "depth_err_chi2_scaled_ppm": err * math.sqrt(max(chi2 / dof, 1.0)) * 1e6,
                "oot_scatter_ppm": float(np.nanmedian(
                    [r.get("oot_scatter", np.nan) for r in used])) * 1e6})
    if len(d) >= 2:
        g = rng or np.random.default_rng(int(params.seed))
        draws = int(params.bootstrap_draws)
        idx = g.integers(0, len(d), size=(draws, len(d)))
        dd, ww = d[idx], w[idx]
        means = np.sum(ww * dd, axis=1) / np.sum(ww, axis=1)
        out["depth_err_bootstrap_ppm"] = float(np.std(means, ddof=1)) * 1e6
    method = str(params.err_method).lower()
    if method == "auto":
        method = ("bootstrap" if len(d) >= int(params.min_transits_for_bootstrap)
                  and np.isfinite(out["depth_err_bootstrap_ppm"]) else "analytic")
    chosen = {"bootstrap": out["depth_err_bootstrap_ppm"],
              "analytic": out["depth_err_analytic_ppm"],
              "chi2_scaled": out["depth_err_chi2_scaled_ppm"]}.get(
                  method, out["depth_err_analytic_ppm"])
    if not np.isfinite(chosen) or chosen <= 0:
        chosen, method = out["depth_err_analytic_ppm"], "analytic"
    out["depth_err_ppm"] = float(chosen)
    out["depth_err_method"] = method
    return out


def binned_fold(rows_time, rows_flux, *, period_days: float, t0_btjd: float,
                duration_days: float, params: FitParams | None = None) -> pd.DataFrame:
    """A phase-binned fold over ``|dt| <= w T14``, for a human to look at."""
    params = params or FitParams()
    t = np.asarray(rows_time, dtype=float)
    f = np.asarray(rows_flux, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if not t.size:
        return pd.DataFrame(columns=["dt_hours", "flux", "flux_err", "n"])
    _, dt = fold(t, period_days, t0_btjd)
    w = float(params.baseline_window_durations) * float(duration_days)
    m = np.abs(dt) <= w
    if not np.any(m):
        return pd.DataFrame(columns=["dt_hours", "flux", "flux_err", "n"])
    nb = max(int(params.fold_bins), 5)
    edges = np.linspace(-w, w, nb + 1)
    idx = np.clip(np.digitize(dt[m], edges) - 1, 0, nb - 1)
    recs = []
    for i in range(nb):
        s = idx == i
        n = int(s.sum())
        if not n:
            continue
        v = f[m][s]
        recs.append({"dt_hours": float(0.5 * (edges[i] + edges[i + 1]) * 24.0),
                     "flux": float(np.mean(v)),
                     "flux_err": float(np.std(v, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                     "n": n})
    return pd.DataFrame(recs, columns=["dt_hours", "flux", "flux_err", "n"])


def dedupe_sectors(sectors, *, authors=DEFAULT_AUTHORS) -> tuple[list[dict], list[dict]]:
    """One light curve per sector: the shortest exposure wins, ties by author order.

    MAST serves a sector under several pipelines and they are reductions of the
    SAME PIXELS. Stacking two of them counts every transit twice, which does
    not improve the depth but does shrink its quoted error by sqrt(2) --- a
    measurement that looks more precise than the photons allow. Returns
    ``(kept, dropped)``; ``dropped`` records each discarded light curve with the
    one that displaced it, so the choice is auditable rather than silent.
    """
    order = {a: i for i, a in enumerate(authors)}

    def rank(s: dict) -> tuple:
        e = float(s.get("exptime_s") or np.inf)
        return (e if np.isfinite(e) else np.inf,
                order.get(str(s.get("author")), len(order)))

    best: dict = {}
    for s in sectors or []:
        key = s.get("sector")
        if key is None:                                   # no sector id: never merged away
            best[f"__unkeyed_{len(best)}"] = s
            continue
        cur = best.get(key)
        if cur is None or rank(s) < rank(cur):
            best[key] = s
    kept_ids = {id(v) for v in best.values()}
    dropped = [{"sector": s.get("sector"), "author": s.get("author"),
                "exptime_s": s.get("exptime_s"),
                "reason": "same sector already covered by a shorter-cadence pipeline",
                "kept_author": str((best.get(s.get("sector")) or {}).get("author")),
                "kept_exptime_s": (best.get(s.get("sector")) or {}).get("exptime_s")}
               for s in (sectors or []) if id(s) not in kept_ids]
    return list(best.values()), dropped


def measure_target(sectors, *, period_days: float, t0_btjd: float, duration_days: float,
                   params: FitParams | None = None) -> dict:
    """Fit every sector of one target and combine: depth, per-sector, odd-even.

    ``sectors`` is a list of dicts with ``time`` (BTJD), ``flux``, ``flux_err``
    and metadata (``sector``, ``author``, ``exptime_s``, ``flux_column``).
    Each sector is normalised by its own robust median FIRST, so a sector-level
    flux scale cannot leak into the depth.

    **One sector is counted once.** MAST serves the same sector under more than
    one pipeline --- SPOC at 120 s and TESS-SPOC at 200 or 600 s are different
    REDUCTIONS OF THE SAME PIXELS, not independent observations. Run 35041932130
    measured K00897.01 over "14 sectors" that were seven sectors twice, so every
    transit entered the stack twice and the quoted error was too small by a
    factor sqrt(2). :func:`dedupe_sectors` keeps the best-cadence pipeline per
    sector and records what it dropped.
    """
    params = params or FitParams()
    sectors, dropped = dedupe_sectors(sectors)
    all_rows: list[dict] = []
    sec_recs: list[dict] = []
    norm_time: list[np.ndarray] = []
    norm_flux: list[np.ndarray] = []
    smeared_any = False
    lower_bound_any = False
    for s in sectors or []:
        t = np.asarray(s.get("time"), dtype=float)
        f = np.asarray(s.get("flux"), dtype=float)
        fe = (np.asarray(s.get("flux_err"), dtype=float) if s.get("flux_err") is not None
              else np.full(t.shape, np.nan))
        ok = np.isfinite(t) & np.isfinite(f)
        t, f, fe = t[ok], f[ok], fe[ok]
        med = float(np.median(f)) if t.size else float("nan")
        if not np.isfinite(med) or med == 0:
            sec_recs.append({"sector": s.get("sector"), "author": s.get("author"),
                             "exptime_s": s.get("exptime_s"), "n_points": int(t.size),
                             "status": "NOT_NORMALISABLE", "n_transits": 0,
                             "depth_ppm": float("nan"), "depth_err_ppm": float("nan")})
            continue
        f, fe = f / med, fe / med
        exptime_s = float(s.get("exptime_s") or np.nan)
        exptime_d = exptime_s / 86400.0 if np.isfinite(exptime_s) else float("nan")
        _, no_flat = core_half_width(duration_days, exptime_d, params)
        smeared = bool(np.isfinite(exptime_d)
                       and exptime_d > float(params.smear_fraction) * float(duration_days))
        smeared_any = smeared_any or smeared
        lower_bound_any = lower_bound_any or no_flat
        rows = fit_transits(t, f, fe, period_days=period_days, t0_btjd=t0_btjd,
                            duration_days=duration_days, exptime_days=exptime_d,
                            sector=s.get("sector"), params=params)
        all_rows.extend(rows)
        norm_time.append(t)
        norm_flux.append(f)
        c = combine_transit_depths(rows, params=params)
        sec_recs.append({"sector": s.get("sector"), "author": s.get("author"),
                         "exptime_s": exptime_s, "flux_column": s.get("flux_column"),
                         "n_points": int(t.size), "status": STATUS_OK,
                         "median_flux": med, "smeared": smeared,
                         "no_flat_core": bool(no_flat),
                         "exptime_over_duration": (float(exptime_d / duration_days)
                                                   if np.isfinite(exptime_d) else float("nan")),
                         "n_transits": c["n_transits"], "n_transits_seen": c["n_transits_seen"],
                         "depth_ppm": c["depth_ppm"], "depth_err_ppm": c["depth_err_ppm"],
                         "depth_err_method": c["depth_err_method"],
                         "oot_scatter_ppm": c["oot_scatter_ppm"]})
    comb = combine_transit_depths(all_rows, params=params)
    odd = combine_transit_depths([r for r in all_rows if int(r.get("epoch", 0)) % 2 != 0],
                                 params=params)
    even = combine_transit_depths([r for r in all_rows if int(r.get("epoch", 0)) % 2 == 0],
                                  params=params)
    oe_diff = odd["depth_ppm"] - even["depth_ppm"]
    oe_sig = math.sqrt(odd["depth_err_ppm"] ** 2 + even["depth_err_ppm"] ** 2) if (
        np.isfinite(odd["depth_err_ppm"]) and np.isfinite(even["depth_err_ppm"])) else float("nan")
    oe_z = abs(oe_diff) / oe_sig if np.isfinite(oe_sig) and oe_sig > 0 else float("nan")
    sec_df = pd.DataFrame(sec_recs)
    sec_meas = sec_df[np.isfinite(pd.to_numeric(sec_df.get("depth_ppm", pd.Series(dtype=float)),
                                                errors="coerce"))] if len(sec_df) else sec_df
    sec_chi2, sec_dof = float("nan"), 0
    if len(sec_meas) >= 2 and np.isfinite(comb["depth_ppm"]):
        dd = pd.to_numeric(sec_meas["depth_ppm"], errors="coerce").to_numpy(float)
        ee = pd.to_numeric(sec_meas["depth_err_ppm"], errors="coerce").to_numpy(float)
        g = np.isfinite(dd) & np.isfinite(ee) & (ee > 0)
        if g.sum() >= 2:
            sec_chi2 = float(np.sum(((dd[g] - comb["depth_ppm"]) / ee[g]) ** 2))
            sec_dof = int(g.sum() - 1)
    flags: list[str] = []
    if smeared_any:
        flags.append("smeared")
    if lower_bound_any:
        flags.append("depth_is_lower_bound")
    if np.isfinite(oe_z) and oe_z >= float(params.odd_even_sigma):
        flags.append("odd_even_significant")
    if sec_dof and np.isfinite(sec_chi2) and (sec_chi2 / sec_dof) >= float(
            params.sector_scatter_chi2_per_dof):
        flags.append("sector_scatter")
    fold_df = (binned_fold(np.concatenate(norm_time), np.concatenate(norm_flux),
                           period_days=period_days, t0_btjd=t0_btjd,
                           duration_days=duration_days, params=params)
               if norm_time else pd.DataFrame(columns=["dt_hours", "flux", "flux_err", "n"]))
    return {
        **{k: comb[k] for k in comb},
        "n_sectors": int(len(sec_recs)),
        "n_sectors_measured": int(len(sec_meas)),
        "sectors": sec_df,
        "sector_list": ",".join(str(r.get("sector")) for r in sec_recs if r.get("sector")
                                is not None),
        "authors": ",".join(sorted({str(r.get("author")) for r in sec_recs
                                    if r.get("author")})),
        "exptimes_s": ",".join(sorted({f"{float(r['exptime_s']):.0f}" for r in sec_recs
                                       if r.get("exptime_s") and np.isfinite(
                                           float(r["exptime_s"]))})),
        # Same sector, another pipeline: dropped so no transit is counted twice.
        "n_sectors_dropped_duplicate": int(len(dropped)),
        "duplicate_sectors_dropped": ";".join(
            f"{d['sector']}:{d['author']}@{d['exptime_s']}s->kept "
            f"{d['kept_author']}@{d['kept_exptime_s']}s" for d in dropped),
        "smeared": bool(smeared_any), "depth_is_lower_bound": bool(lower_bound_any),
        "depth_odd_ppm": odd["depth_ppm"], "depth_odd_err_ppm": odd["depth_err_ppm"],
        "n_transits_odd": odd["n_transits"],
        "depth_even_ppm": even["depth_ppm"], "depth_even_err_ppm": even["depth_err_ppm"],
        "n_transits_even": even["n_transits"],
        "odd_even_diff_ppm": oe_diff, "odd_even_sigma": oe_z,
        "sector_scatter_chi2": sec_chi2, "sector_scatter_dof": sec_dof,
        "sector_scatter_chi2_per_dof": (sec_chi2 / sec_dof) if sec_dof else float("nan"),
        "flags": ";".join(flags),
        "transits": pd.DataFrame(all_rows),
        "fold": fold_df,
    }


# ---------------------------------------------------------------------------
# The three-way comparison
# ---------------------------------------------------------------------------
def _z_ln(d1: float, e1: float, d2: float, e2: float, sigma_sys: float) -> tuple[float, float]:
    """``(z, sigma)`` for ``ln(d1/d2)`` with fractional errors added in quadrature."""
    if not (np.isfinite(d1) and np.isfinite(d2) and d1 > 0 and d2 > 0):
        return float("nan"), float("nan")
    f1 = (float(e1) / float(d1)) if np.isfinite(e1) and e1 > 0 else 0.0
    f2 = (float(e2) / float(d2)) if np.isfinite(e2) and e2 > 0 else 0.0
    sig = math.sqrt(f1 * f1 + f2 * f2 + float(sigma_sys) ** 2)
    if sig <= 0:
        return float("nan"), float("nan")
    return float(math.log(d1 / d2) / sig), float(sig)


def compare_three_depths(depth_measured_ppm: float, err_measured_ppm: float,
                         depth_kepler_ppm: float, err_kepler_ppm: float,
                         depth_toi_ppm: float, err_toi_ppm: float, *,
                         ld_band_ratio: float = 1.0,
                         params: CompareParams | None = None,
                         unmeasured_reason: str | None = None) -> dict:
    """The decisive comparison: measured TESS depth against BOTH catalogue depths.

    The Kepler depth is multiplied by ``ld_band_ratio`` (=
    :func:`seti.growth.drift.band_ratio`, ``F_TESS(b)/F_Kepler(b)``) so both
    references sit in the TESS band before anything is compared.  Returns the
    verdict, both ``z``s, and ``z_toi_vs_kepler`` --- the separation between the
    two references, without which "agrees with both" cannot be read.
    """
    params = params or CompareParams()
    out: dict = {"verdict": MATCH_UNMEASURED, "unmeasured_reason": unmeasured_reason or "",
                 "depth_measured_ppm": float(depth_measured_ppm),
                 "depth_measured_err_ppm": float(err_measured_ppm),
                 "depth_kepler_ppm": float(depth_kepler_ppm),
                 "depth_kepler_in_tess_band_ppm": float("nan"),
                 "depth_toi_ppm": float(depth_toi_ppm),
                 "ld_band_ratio": float(ld_band_ratio),
                 "z_vs_kepler": float("nan"), "sigma_vs_kepler": float("nan"),
                 "z_vs_toi": float("nan"), "sigma_vs_toi": float("nan"),
                 "z_toi_vs_kepler": float("nan"),
                 "agrees_with_kepler": False, "agrees_with_toi": False,
                 "references_separated": False, "n_references_usable": 0,
                 "n_agree": float(params.n_agree),
                 "sigma_sys_ln": float(params.sigma_sys_ln)}
    out["n_references_usable"] = int(sum(
        1 for d in (depth_kepler_ppm, depth_toi_ppm) if np.isfinite(d) and d > 0))
    fr = float(ld_band_ratio) if (params.apply_band_ratio and np.isfinite(ld_band_ratio)
                                  and ld_band_ratio > 0) else 1.0
    dk = float(depth_kepler_ppm) * fr
    ek = float(err_kepler_ppm) * fr if np.isfinite(err_kepler_ppm) else float("nan")
    out["depth_kepler_in_tess_band_ppm"] = dk
    zrk, _s = _z_ln(float(depth_toi_ppm), float(err_toi_ppm), dk, ek, params.sigma_sys_ln)
    out["z_toi_vs_kepler"] = zrk
    out["references_separated"] = bool(np.isfinite(zrk) and abs(zrk) >= params.n_agree)
    if unmeasured_reason or not (np.isfinite(depth_measured_ppm) and depth_measured_ppm > 0):
        out["verdict"] = MATCH_UNMEASURED
        out["unmeasured_reason"] = unmeasured_reason or UNMEASURED_NO_TRANSIT
        return out
    if out["n_references_usable"] == 0:
        # A depth WAS measured; there is simply nothing to compare it with.
        # "matches neither" would assert a disagreement with values that do not
        # exist, so the comparison is the thing reported as unmeasured.
        out["verdict"] = MATCH_UNMEASURED
        out["unmeasured_reason"] = UNMEASURED_NO_REFERENCE
        return out
    zk, sk = _z_ln(float(depth_measured_ppm), float(err_measured_ppm), dk, ek,
                   params.sigma_sys_ln)
    zt, st = _z_ln(float(depth_measured_ppm), float(err_measured_ppm), float(depth_toi_ppm),
                   float(err_toi_ppm), params.sigma_sys_ln)
    out.update({"z_vs_kepler": zk, "sigma_vs_kepler": sk, "z_vs_toi": zt, "sigma_vs_toi": st})
    ak = bool(np.isfinite(zk) and abs(zk) < params.n_agree)
    at = bool(np.isfinite(zt) and abs(zt) < params.n_agree)
    out["agrees_with_kepler"], out["agrees_with_toi"] = ak, at
    if ak and at:
        out["verdict"] = MATCH_BOTH
    elif ak:
        out["verdict"] = MATCH_KEPLER
    elif at:
        out["verdict"] = MATCH_TOI
    else:
        out["verdict"] = MATCH_NEITHER
    return out


def run_verdict(measurements: pd.DataFrame) -> tuple[str, str]:
    """``(verdict, reason)`` for the whole stage from the per-target verdicts."""
    if measurements is None or not len(measurements) or "verdict" not in measurements:
        return RUN_NO_DATA, "no_targets"
    v = measurements["verdict"].fillna(MATCH_UNMEASURED).astype(str)
    measured = v[v != MATCH_UNMEASURED]
    if not len(measured):
        reasons = sorted({str(r) for r in measurements.get(
            "unmeasured_reason", pd.Series(dtype=str)).fillna("unknown") if str(r)})
        return RUN_NO_DATA, "no_target_measured:" + ",".join(r for r in reasons if r)
    if (measured == MATCH_TOI).any():
        return RUN_CONFIRMED, f"{int((measured == MATCH_TOI).sum())} target(s) matched the TOI depth"
    if (measured == MATCH_KEPLER).all():
        return RUN_REFUTED, f"all {len(measured)} measured target(s) matched the Kepler depth"
    return RUN_UNRESOLVED, (f"{int((measured == MATCH_NEITHER).sum())} matched neither, "
                            f"{int((measured == MATCH_BOTH).sum())} matched both, "
                            f"{int((measured == MATCH_KEPLER).sum())} matched Kepler")


# ---------------------------------------------------------------------------
# Synthetic light curves (injection: the test gate, and the runner's self-check)
# ---------------------------------------------------------------------------
def trapezoid_transit(dt, duration_days: float, depth: float, ingress_frac: float = 0.1):
    """A trapezoidal transit: flat bottom ``depth``, ingress/egress ``ingress_frac * T14``."""
    d = np.abs(np.asarray(dt, dtype=float))
    half = 0.5 * float(duration_days)
    tau = max(float(ingress_frac) * float(duration_days), 1e-12)
    out = np.zeros_like(d)
    flat = d <= (half - tau)
    slope = (d > (half - tau)) & (d < half)
    out[flat] = 1.0
    out[slope] = (half - d[slope]) / tau
    return 1.0 - float(depth) * out


def synth_lightcurve(*, period_days: float, t0_btjd: float, duration_days: float,
                     depth: float, exptime_s: float = 120.0, n_transits: int = 12,
                     span_durations: float = 3.0, noise_ppm: float = 500.0,
                     sector: int = 1, author: str = "SPOC", seed: int = 11,
                     odd_depth: float | None = None, ingress_frac: float = 0.1,
                     oversample: int = 11) -> dict:
    """One synthetic sector with a KNOWN injected depth, integrated over the cadence.

    The transit is trapezoidal and is **integrated across each exposure** by
    oversampling, so a long cadence really does smear the transit the way an
    FFI light curve does --- that is what makes the smearing flag testable.
    ``odd_depth`` gives odd-numbered epochs a different depth: a synthetic
    eclipsing binary for the odd-even test.
    """
    rng = np.random.default_rng(int(seed))
    exp_d = float(exptime_s) / 86400.0
    half_span = float(span_durations) * float(duration_days)
    times, fluxes = [], []
    for k in range(int(n_transits)):
        centre = float(t0_btjd) + k * float(period_days)
        n = max(int(round(2 * half_span / exp_d)), 8)
        t = centre + np.linspace(-half_span, half_span, n)
        sub = np.linspace(-0.5, 0.5, max(int(oversample), 1)) * exp_d
        dt_grid = (t[:, None] + sub[None, :]) - centre
        dep = float(depth) if (k % 2 == 0 or odd_depth is None) else float(odd_depth)
        model = trapezoid_transit(dt_grid, duration_days, dep, ingress_frac).mean(axis=1)
        times.append(t)
        fluxes.append(model)
    t = np.concatenate(times)
    f = np.concatenate(fluxes)
    f = f + rng.normal(0.0, float(noise_ppm) * 1e-6, size=f.shape)
    return {"sector": int(sector), "author": str(author), "exptime_s": float(exptime_s),
            "flux_column": "PDCSAP_FLUX", "time": t, "flux": f,
            "flux_err": np.full(f.shape, float(noise_ppm) * 1e-6), "n_points": int(t.size)}


# ---------------------------------------------------------------------------
# MAST access (runner only; injectable everywhere)
# ---------------------------------------------------------------------------
def mast_probe() -> dict:
    """Which route to MAST this machine actually has.  Imports only; no query."""
    rep = {"lightkurve": {"importable": False}, "astroquery_mast": {"importable": False},
           "astropy_io_fits": {"importable": False}, "route_preferred": None}
    try:
        import lightkurve as lk  # noqa: PLC0415

        rep["lightkurve"] = {"importable": True, "version": str(getattr(lk, "__version__", "?"))}
    except Exception as exc:                              # noqa: BLE001
        rep["lightkurve"] = {"importable": False, "error": repr(exc)[:300]}
    try:
        import astroquery  # noqa: PLC0415
        from astroquery.mast import Observations  # noqa: PLC0415, F401

        rep["astroquery_mast"] = {"importable": True,
                                  "version": str(getattr(astroquery, "__version__", "?"))}
    except Exception as exc:                              # noqa: BLE001
        rep["astroquery_mast"] = {"importable": False, "error": repr(exc)[:300]}
    try:
        from astropy.io import fits  # noqa: PLC0415, F401

        rep["astropy_io_fits"] = {"importable": True}
    except Exception as exc:                              # noqa: BLE001
        rep["astropy_io_fits"] = {"importable": False, "error": repr(exc)[:300]}
    if rep["lightkurve"]["importable"]:
        rep["route_preferred"] = ROUTE_LIGHTKURVE
    elif rep["astroquery_mast"]["importable"] and rep["astropy_io_fits"]["importable"]:
        rep["route_preferred"] = ROUTE_MAST_FITS
    rep["note"] = ("import reachability only; whether MAST ANSWERS is established by the "
                   "measure stage and recorded per target as OK / QUERY_FAILED / "
                   "QUERY_RETURNED_ZERO_ROWS")
    return rep


def _exptime_from_header(hdr) -> float:
    """Cadence in SECONDS from a TESS light-curve header.

    ``TIMEDEL`` is the time between samples in **days**; ``FRAMETIM * NUM_FRM``
    is the co-added frame time in **seconds** and is the fallback.  Nothing
    here guesses from the data: a header that says neither returns NaN, and a
    NaN exposure means the smearing test cannot fire, which the record shows.
    """
    try:
        v = hdr.get("TIMEDEL")
        if v is not None and np.isfinite(float(v)) and float(v) > 0:
            return float(v) * 86400.0
    except Exception:                                     # noqa: BLE001
        pass
    try:
        ft = float(hdr.get("FRAMETIM") or np.nan)
        nf = float(hdr.get("NUM_FRM") or np.nan)
        if np.isfinite(ft) and np.isfinite(nf) and ft > 0 and nf > 0:
            return ft * nf
    except Exception:                                     # noqa: BLE001
        pass
    return float("nan")


def read_tess_lc_fits(path, *, flux_columns=FLUX_COLUMNS) -> dict | None:
    """One TESS light-curve FITS product -> ``{time (BTJD), flux, flux_err, meta}``.

    ``TIME`` is converted through the file's own ``BJDREFI``/``BJDREFF`` rather
    than assumed to be BTJD, and the flux column that was actually found is
    recorded.  Quality is masked when a ``QUALITY`` column exists.
    """
    from astropy.io import fits  # noqa: PLC0415

    with fits.open(str(path), memmap=False) as hdul:
        hdu = None
        for h in hdul:
            try:
                names = [str(n).upper() for n in h.columns.names]
            except AttributeError:
                continue
            if "TIME" in names:
                hdu = h
                break
        if hdu is None:
            return None
        data, hdr = hdu.data, hdu.header
        cols = [str(n).upper() for n in hdu.columns.names]
        fcol = next((c for c in flux_columns if c in cols), None)
        if fcol is None:
            return None
        bjdrefi = float(hdr.get("BJDREFI", hdul[0].header.get("BJDREFI", BTJD_OFFSET)) or
                        BTJD_OFFSET)
        bjdreff = float(hdr.get("BJDREFF", hdul[0].header.get("BJDREFF", 0.0)) or 0.0)
        t = np.asarray(data["TIME"], dtype=float) + (bjdrefi + bjdreff) - BTJD_OFFSET
        f = np.asarray(data[fcol], dtype=float)
        ecol = fcol + "_ERR"
        fe = (np.asarray(data[ecol], dtype=float) if ecol in cols
              else np.full(f.shape, np.nan, dtype=float))
        n_before = int(t.size)
        if "QUALITY" in cols:
            q = np.asarray(data["QUALITY"])
            keep = (q == 0)
            t, f, fe = t[keep], f[keep], fe[keep]
        ok = np.isfinite(t) & np.isfinite(f)
        t, f, fe = t[ok], f[ok], fe[ok]
        sector = hdul[0].header.get("SECTOR", hdr.get("SECTOR"))
        author = str(hdul[0].header.get("PROCVER", "") or hdul[0].header.get("ORIGIN", "")
                     or "unknown")
        return {"sector": int(sector) if sector is not None else None, "author": author,
                "exptime_s": _exptime_from_header(hdr), "flux_column": fcol,
                "time": t, "flux": f, "flux_err": fe, "n_points": int(t.size),
                "n_points_before_quality_mask": n_before,
                "bjdref": bjdrefi + bjdreff}


def lightkurve_lc_fn(tic_id, *, authors=DEFAULT_AUTHORS, max_sectors: int = 60,
                     download_dir: str | None = None, **_kw) -> list[dict]:
    """Light curves for one TIC through ``lightkurve`` (runner only).

    The time axis is taken as ``Time.jd - 2457000`` rather than trusting the
    object's format string, and the author, sector and exposure time of every
    sector are carried through so the record says what was measured on what.
    """
    import lightkurve as lk  # noqa: PLC0415

    sr = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
    if sr is None or len(sr) == 0:
        return []
    try:
        tbl = sr.table
        auth = np.array([str(a) for a in tbl["author"]])
        keep = np.isin(auth, list(authors))
        if keep.any():
            sr = sr[keep]
    except Exception:                                     # noqa: BLE001
        pass
    out: list[dict] = []
    for i in range(min(len(sr), int(max_sectors))):
        try:
            lc = sr[i].download(download_dir=download_dir)
        except Exception:                                 # noqa: BLE001
            continue
        if lc is None:
            continue
        try:
            lc = lc.remove_nans()
        except Exception:                                 # noqa: BLE001
            pass
        try:
            t = np.asarray(lc.time.jd, dtype=float) - BTJD_OFFSET
        except Exception:                                 # noqa: BLE001
            t = np.asarray(lc.time.value, dtype=float)
        f = np.asarray(getattr(lc.flux, "value", lc.flux), dtype=float)
        fe = np.asarray(getattr(getattr(lc, "flux_err", None), "value",
                                np.full(f.shape, np.nan)), dtype=float)
        meta = dict(getattr(lc, "meta", {}) or {})
        exptime = meta.get("TIMEDEL")
        exptime_s = float(exptime) * 86400.0 if exptime else float("nan")
        if not np.isfinite(exptime_s) and t.size > 2:
            exptime_s = float(np.median(np.diff(np.sort(t)))) * 86400.0
        out.append({"sector": int(meta.get("SECTOR")) if meta.get("SECTOR") else None,
                    "author": str(meta.get("AUTHOR") or meta.get("ORIGIN") or "unknown"),
                    "exptime_s": exptime_s, "flux_column": str(meta.get("FLUX_ORIGIN")
                                                               or "PDCSAP_FLUX"),
                    "time": t, "flux": f, "flux_err": fe, "n_points": int(t.size)})
    return out


def mast_fits_lc_fn(tic_id, *, authors=DEFAULT_AUTHORS, max_sectors: int = 60,
                    download_dir: str | None = None, **_kw) -> list[dict]:
    """Light curves for one TIC through ``astroquery.mast`` + FITS (runner only).

    The fallback when ``lightkurve`` is not installed: query the TESS
    timeseries observations for the TIC, take the ``LC`` products, download by
    ``dataURI`` and read them with :func:`read_tess_lc_fits`.
    """
    from astroquery.mast import Observations  # noqa: PLC0415

    obs = Observations.query_criteria(obs_collection="TESS", dataproduct_type="timeseries",
                                      target_name=str(int(tic_id)))
    if obs is None or len(obs) == 0:
        return []
    prod = Observations.get_product_list(obs)
    if prod is None or len(prod) == 0:
        return []
    p = prod.to_pandas()
    if "productSubGroupDescription" in p.columns:
        p = p[p["productSubGroupDescription"].astype(str).str.upper() == "LC"]
    if "provenance_name" in p.columns and len(authors):
        want = {str(a).upper() for a in authors}
        sel = p["provenance_name"].astype(str).str.upper().isin(want)
        if sel.any():
            p = p[sel]
    p = p.head(int(max_sectors))
    root = Path(download_dir or ".") / "mast_lc"
    root.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for _, r in p.iterrows():
        uri = str(r.get("dataURI") or "")
        if not uri:
            continue
        local = root / str(r.get("productFilename") or Path(uri).name)
        try:
            status, _msg, _url = Observations.download_file(uri, local_path=str(local),
                                                            cache=False)
            if str(status).upper() != "COMPLETE" or not local.exists():
                continue
            rec = read_tess_lc_fits(local)
        except Exception:                                 # noqa: BLE001
            continue
        if rec is None:
            continue
        prov = str(r.get("provenance_name") or "").strip()
        if prov:
            rec["author"] = prov
        out.append(rec)
    return out


def default_lc_fn(tic_id, **kw) -> list[dict]:
    """``lightkurve`` if it is installed, else ``astroquery.mast`` + FITS."""
    try:
        import lightkurve  # noqa: PLC0415, F401

        return lightkurve_lc_fn(tic_id, **kw)
    except ImportError:
        return mast_fits_lc_fn(tic_id, **kw)


def fetch_lightcurves(tic_id, *, lc_fn=None, params: MastParams | None = None,
                      log: AcquisitionLog | None = None, deadline: Deadline | None = None,
                      key: str = "") -> tuple[list[dict], str, str]:
    """One target's TESS light curves, bounded and recorded.

    Returns ``(sectors, status, route)``; ``status`` is ``OK`` /
    ``QUERY_RETURNED_ZERO_ROWS`` / ``QUERY_FAILED``, and a failure is **never**
    turned into an empty-but-successful answer.  Retries stop as soon as the
    wall-clock budget is gone.
    """
    params = params or MastParams()
    log = log or AcquisitionLog()
    label = f"lightcurves_{key or tic_id}"
    if deadline is not None and deadline.expired():
        log.record(label, f"TIC {tic_id}", error="budget_exhausted_before_request")
        return [], STATUS_FAILED, ""
    if lc_fn is not None:
        route = "injected"
    else:
        route = mast_probe().get("route_preferred") or ""
    fn = lc_fn or default_lc_fn
    last = ""
    own = Deadline(budget_s=float(params.per_target_budget_s))
    n_attempts = max(int(params.retries), 1)
    for attempt in range(n_attempts):
        if own.expired() or (deadline is not None and deadline.expired()):
            last = "budget_exhausted"
            break
        if attempt and float(params.retry_pause_s) > 0:
            # A backoff, but never one that outlives the budget it sits inside.
            _time.sleep(min(float(params.retry_pause_s) * attempt, own.remaining(),
                            deadline.remaining() if deadline is not None else float("inf")))
        try:
            secs = fn(tic_id, authors=tuple(params.authors), max_sectors=int(params.max_sectors),
                      download_dir=params.download_dir)
        except Exception as exc:                          # noqa: BLE001
            last = repr(exc)[:400]
            continue
        secs = [s for s in (secs or []) if s is not None and len(np.asarray(s.get("time"),
                                                                           dtype=float))]
        if not secs:
            log.record(label, f"TIC {tic_id}", rows=0,
                       extra={"route": route, "attempt": attempt + 1})
            return [], STATUS_ZERO, route
        log.record(label, f"TIC {tic_id}", rows=int(sum(int(s.get("n_points") or 0)
                                                        for s in secs)),
                   extra={"route": route, "n_sectors": len(secs), "attempt": attempt + 1})
        return secs, STATUS_OK, route
    log.record(label, f"TIC {tic_id}", error=last or "unknown")
    return [], STATUS_FAILED, route


# ---------------------------------------------------------------------------
# Ephemerides from the Exoplanet Archive
# ---------------------------------------------------------------------------
def _quote_list(values) -> str:
    return ",".join("'" + str(v).replace("'", "") + "'" for v in values)


def fetch_koi_ephemerides(kepoi_names, *, query_fn=None, log: AcquisitionLog | None = None,
                          table: str = "cumulative", columns=KOI_EPH_COLUMNS
                          ) -> tuple[pd.DataFrame, str]:
    """``koi_period`` / ``koi_time0bk`` / ``koi_duration`` for the shortlist.

    ``koi_time0bk`` is NOT in the stage-1 column list, so stage 2 must pull it:
    the epoch is what makes a fold possible and it cannot be reconstructed from
    ``joined.csv``.  A failure is ``QUERY_FAILED``; an empty answer is
    ``QUERY_RETURNED_ZERO_ROWS``; neither becomes an ephemeris.
    """
    log = log or AcquisitionLog()
    names = [str(n) for n in kepoi_names if str(n) and str(n).lower() != "nan"]
    if not names:
        return pd.DataFrame(), STATUS_ZERO
    adql = (f"select {','.join(columns)} from {table} "
            f"where kepoi_name in ({_quote_list(names)})")
    qf = query_fn or tap_sync
    try:
        df = qf(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record("fetch_koi_ephemerides", adql, error=repr(exc))
        return pd.DataFrame(), STATUS_FAILED
    if df is None:
        log.record("fetch_koi_ephemerides", adql, error="query_fn returned None")
        return pd.DataFrame(), STATUS_FAILED
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    log.record("fetch_koi_ephemerides", adql, rows=int(len(df)))
    return df, (STATUS_OK if len(df) else STATUS_ZERO)


def fetch_toi_depths(tids, *, query_fn=None, log: AcquisitionLog | None = None,
                     table: str = "toi", columns=TOI_DEPTH_COLUMNS) -> tuple[pd.DataFrame, str]:
    """The TOI rows for the shortlist's TIC ids (the deep catalogue value under test)."""
    log = log or AcquisitionLog()
    ids = []
    for t in tids:
        try:
            ids.append(str(int(float(t))))
        except Exception:                                 # noqa: BLE001
            continue
    if not ids:
        return pd.DataFrame(), STATUS_ZERO
    adql = f"select {','.join(columns)} from {table} where tid in ({','.join(sorted(set(ids)))})"
    qf = query_fn or tap_sync
    try:
        df = qf(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record("fetch_toi_depths", adql, error=repr(exc))
        return pd.DataFrame(), STATUS_FAILED
    if df is None:
        log.record("fetch_toi_depths", adql, error="query_fn returned None")
        return pd.DataFrame(), STATUS_FAILED
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    log.record("fetch_toi_depths", adql, rows=int(len(df)))
    return df, (STATUS_OK if len(df) else STATUS_ZERO)


# ---------------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------------
def load_shortlist(conf: dict, *, candidates_csv: Path | None = None) -> pd.DataFrame:
    """The stage-2 shortlist: ``config/growth.yaml`` ``stage2.shortlist``, plus
    (when ``stage2.shortlist_from_candidates_csv``) the stage-1 candidate rows.

    Config entries win on a duplicate ``kepoi_name``.  Every row carries where
    it came from in ``shortlist_source``.
    """
    s2 = (conf or {}).get("stage2") or {}
    rows: list[dict] = []
    for e in s2.get("shortlist") or []:
        r = dict(e)
        r["shortlist_source"] = "config"
        rows.append(r)
    if s2.get("shortlist_from_candidates_csv", True) and candidates_csv is not None:
        p = Path(candidates_csv)
        if p.exists() and p.stat().st_size:
            try:
                c = pd.read_csv(p, low_memory=False)
            except Exception:                             # noqa: BLE001
                c = pd.DataFrame()
            for _, r in c.iterrows():
                rows.append({"kepoi_name": r.get("kepoi_name"),
                             "kepler_name": r.get("kepler_name"), "kepid": r.get("kepid"),
                             "tic_id": r.get("tic_id"), "toi": r.get("toi"),
                             "shortlist_source": "candidates_csv"})
    if not rows:
        return pd.DataFrame(columns=["kepoi_name", "kepler_name", "kepid", "tic_id", "toi",
                                     "shortlist_source"])
    df = pd.DataFrame(rows)
    df["kepoi_name"] = df["kepoi_name"].map(str)
    df = df.drop_duplicates(subset=["kepoi_name"], keep="first").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def stage2_probe(conf: dict, out: Path) -> dict:
    """Which route to MAST exists on this machine, written to ``probe.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"stage": "probe", "generated_utc": _now(), "mast": mast_probe(),
           "shortlist_size": int(len(load_shortlist(conf,
                                                    candidates_csv=_candidates_csv(out))))}
    _write(out / "probe.json", rep)
    print(f"[growth-stage2] probe: MAST route = {rep['mast'].get('route_preferred')}; "
          f"shortlist {rep['shortlist_size']}")
    return rep


def _candidates_csv(out: Path) -> Path:
    """Stage 1's ``candidates.csv``: ``results/growth/candidates.csv`` by default."""
    p = Path(out)
    for cand in (p / "candidates.csv", p.parent / "candidates.csv"):
        if cand.exists():
            return cand
    return p.parent / "candidates.csv"


def measure_one(entry: dict, koi_row: dict | None, toi_row: dict | None, *,
                lc_fn=None, mast: MastParams, fit: FitParams, compare: CompareParams,
                ld_table: dict | None = None, log: AcquisitionLog | None = None,
                deadline: Deadline | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Everything stage 2 has to say about one target.

    Returns ``(measurement, sectors_df, fold_df)``.  A target whose light curve
    could not be fetched comes back ``UNMEASURED`` with the archive's own
    status as the reason --- never as agreeing with anything.
    """
    log = log or AcquisitionLog()
    key = str(entry.get("kepoi_name") or entry.get("toi") or entry.get("tic_id"))
    rec: dict = {"kepoi_name": entry.get("kepoi_name"), "kepler_name": entry.get("kepler_name"),
                 "kepid": entry.get("kepid"), "tic_id": entry.get("tic_id"),
                 "toi": entry.get("toi"), "shortlist_source": entry.get("shortlist_source"),
                 "lc_status": "", "lc_route": "", "n_sectors": 0, "n_sectors_measured": 0,
                 "sector_list": "", "authors": "", "exptimes_s": "",
                 "period_days": float("nan"), "t0_bkjd": float("nan"),
                 "t0_btjd_kepler_epoch": float("nan"), "t0_btjd_used": float("nan"),
                 "n_epochs_propagated": 0, "ephemeris_sigma_minutes": float("nan"),
                 "duration_hours": float("nan"), "flags": ""}
    empty_sec = pd.DataFrame()
    empty_fold = pd.DataFrame(columns=["dt_hours", "flux", "flux_err", "n"])

    depth_k = _f((koi_row or {}).get("koi_depth"))
    err_k = sym_err(_f((koi_row or {}).get("koi_depth_err1")),
                    _f((koi_row or {}).get("koi_depth_err2")))
    depth_t = _f((toi_row or {}).get("pl_trandep"))
    err_t = sym_err(_f((toi_row or {}).get("pl_trandeperr1")),
                    _f((toi_row or {}).get("pl_trandeperr2")))
    b = _f((koi_row or {}).get("koi_impact"))
    fr, ld_meta = band_ratio(_f((koi_row or {}).get("koi_steff")),
                             _f((koi_row or {}).get("koi_slogg")),
                             b if np.isfinite(b) else 0.0, ld_table)
    rec["ld_note"] = ld_meta.get("ld_note", "")

    period = _f((koi_row or {}).get("koi_period"))
    t0_bkjd = _f((koi_row or {}).get("koi_time0bk"))
    dur_h = _f((koi_row or {}).get("koi_duration"))
    rec.update({"period_days": period, "t0_bkjd": t0_bkjd, "duration_hours": dur_h,
                "b_kepler": b})
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0_bkjd)
            and np.isfinite(dur_h) and dur_h > 0):
        rec.update(compare_three_depths(float("nan"), float("nan"), depth_k, err_k, depth_t,
                                        err_t, ld_band_ratio=fr, params=compare,
                                        unmeasured_reason=UNMEASURED_NO_EPHEMERIS))
        rec["lc_status"] = "not_attempted"
        return rec, empty_sec, empty_fold

    t0_btjd = float(bkjd_to_btjd(t0_bkjd))
    rec["t0_btjd_kepler_epoch"] = t0_btjd
    tic = entry.get("tic_id")
    secs, status, route = fetch_lightcurves(tic, lc_fn=lc_fn, params=mast, log=log,
                                            deadline=deadline, key=key)
    rec["lc_status"], rec["lc_route"] = status, route
    if status != STATUS_OK:
        reason = (UNMEASURED_BUDGET if deadline is not None and deadline.expired()
                  and status == STATUS_FAILED else status)
        rec.update(compare_three_depths(float("nan"), float("nan"), depth_k, err_k, depth_t,
                                        err_t, ld_band_ratio=fr, params=compare,
                                        unmeasured_reason=reason))
        return rec, empty_sec, empty_fold

    t_all = np.concatenate([np.asarray(s["time"], dtype=float) for s in secs])
    prop = propagate_epoch(t0_btjd, period, float(np.nanmedian(t_all)),
                           t0_err_days=sym_err(_f((koi_row or {}).get("koi_time0bk_err1")),
                                               _f((koi_row or {}).get("koi_time0bk_err2"))),
                           period_err_days=sym_err(_f((koi_row or {}).get("koi_period_err1")),
                                                   _f((koi_row or {}).get("koi_period_err2"))))
    rec.update({"t0_btjd_used": prop["t0_btjd"], "n_epochs_propagated": prop["n_epochs"],
                "ephemeris_sigma_minutes": prop["sigma_minutes"]})
    dur_d = dur_h / 24.0
    m = measure_target(secs, period_days=period, t0_btjd=prop["t0_btjd"],
                       duration_days=dur_d, params=fit)
    flags = [f for f in str(m["flags"]).split(";") if f]
    if np.isfinite(prop["sigma_minutes"]) and prop["sigma_minutes"] > 0.1 * dur_h * 60.0:
        flags.append("ephemeris_drift_over_10pct_of_duration")
    # The fitted quantities keep their own names; `compare_three_depths` then
    # adds depth_measured_ppm / depth_measured_err_ppm as the canonical pair.
    rec["depth_measured_err_method"] = m["depth_err_method"]
    for k in ("depth_err_analytic_ppm", "depth_err_bootstrap_ppm", "depth_err_chi2_scaled_ppm",
              "chi2", "chi2_per_dof", "n_transits", "n_transits_seen", "oot_scatter_ppm",
              "n_sectors", "n_sectors_measured", "sector_list", "authors", "exptimes_s",
              "smeared", "depth_is_lower_bound", "depth_odd_ppm", "depth_odd_err_ppm",
              "n_transits_odd", "depth_even_ppm", "depth_even_err_ppm", "n_transits_even",
              "odd_even_diff_ppm", "odd_even_sigma", "sector_scatter_chi2",
              "sector_scatter_dof", "sector_scatter_chi2_per_dof"):
        rec[k] = m[k]
    reason = None if (m["n_transits"] and np.isfinite(m["depth_ppm"])) else UNMEASURED_NO_TRANSIT
    rec.update(compare_three_depths(m["depth_ppm"], m["depth_err_ppm"], depth_k, err_k,
                                    depth_t, err_t, ld_band_ratio=fr, params=compare,
                                    unmeasured_reason=reason))
    rec["flags"] = ";".join(flags)
    sec = m["sectors"].copy()
    if len(sec):
        sec.insert(0, "kepoi_name", rec["kepoi_name"])
        sec.insert(1, "tic_id", rec["tic_id"])
    fold_df = m["fold"].copy()
    return rec, sec, fold_df


def stage2_measure(conf: dict, out: Path, *, query_fn=None, lc_fn=None,
                   shortlist: pd.DataFrame | None = None, log: AcquisitionLog | None = None
                   ) -> dict:
    """Fetch the ephemerides and the light curves, fit every shortlisted target."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = log or AcquisitionLog(prefix="growth/stage2")
    mast = MastParams.from_config(conf)
    fit = FitParams.from_config(conf)
    cmp_p = CompareParams.from_config(conf)
    ld_table = conf.get("limb_darkening")
    if shortlist is None:
        shortlist = load_shortlist(conf, candidates_csv=_candidates_csv(out))
    arch = conf.get("archive") or {}
    url = str(arch.get("exoarchive_tap")
              or "https://exoplanetarchive.ipac.caltech.edu/TAP/sync")
    qf = query_fn or (lambda adql: tap_sync(adql, url=url, retries=int(mast.archive_retries),
                                            timeout=float(mast.archive_timeout_s)))
    koi, s_koi = fetch_koi_ephemerides(shortlist.get("kepoi_name", pd.Series(dtype=str)),
                                       query_fn=qf, log=log)
    toi, s_toi = fetch_toi_depths(shortlist.get("tic_id", pd.Series(dtype=float)),
                                  query_fn=qf, log=log)
    koi_by = {str(r.get("kepoi_name")): r for r in koi.to_dict(orient="records")} if len(
        koi) else {}
    toi_rows = toi.to_dict(orient="records") if len(toi) else []

    deadline = Deadline(budget_s=float(mast.budget_s) if mast.budget_s else None)
    recs, secs, folds = [], [], {}
    for entry in shortlist.to_dict(orient="records"):
        k = str(entry.get("kepoi_name"))
        koi_row = koi_by.get(k)
        toi_row = _pick_toi_row(toi_rows, entry, koi_row)
        rec, sec, fold_df = measure_one(entry, koi_row, toi_row, lc_fn=lc_fn, mast=mast,
                                        fit=fit, compare=cmp_p, ld_table=ld_table, log=log,
                                        deadline=deadline)
        # The table's status and THIS target's status are different facts: a
        # query that succeeded but returned no row for this KOI is ZERO_ROWS
        # for the target even though the table is OK.
        rec["koi_ephemeris_status"] = (STATUS_OK if koi_row is not None else
                                       (s_koi if s_koi != STATUS_OK else STATUS_ZERO))
        rec["toi_row_status"] = STATUS_OK if toi_row is not None else STATUS_ZERO
        recs.append(rec)
        if len(sec):
            secs.append(sec)
        if len(fold_df):
            folds[k] = fold_df
        print(f"[growth-stage2] {k}: lc={rec['lc_status']} verdict={rec.get('verdict')} "
              f"depth={rec.get('depth_measured_ppm')} +/- {rec.get('depth_measured_err_ppm')} ppm "
              f"(Kepler {rec.get('depth_kepler_in_tess_band_ppm')}, TOI {rec.get('depth_toi_ppm')})")
    meas = pd.DataFrame(recs)
    meas.to_csv(out / "measurements.csv", index=False)
    sec_df = pd.concat(secs, ignore_index=True) if secs else pd.DataFrame()
    sec_df.to_csv(out / "sectors.csv", index=False)
    fdir = out / "folds"
    fdir.mkdir(parents=True, exist_ok=True)
    for k, v in folds.items():
        v.to_csv(fdir / f"{str(k).replace('/', '_')}.csv", index=False)
    rep = {"stage": "measure", "generated_utc": _now(),
           "n_shortlist": int(len(shortlist)),
           "koi_ephemeris_status": s_koi, "toi_status": s_toi,
           "n_measured": int((meas.get("verdict", pd.Series(dtype=str)) != MATCH_UNMEASURED
                              ).sum()) if len(meas) else 0,
           "lc_status_counts": (meas["lc_status"].map(str).value_counts().to_dict()
                                if len(meas) else {}),
           "lc_routes": (meas["lc_route"].map(str).value_counts().to_dict()
                         if len(meas) else {}),
           "budget_s": mast.budget_s, "elapsed_s": round(deadline.elapsed(), 1),
           "budget_exhausted": bool(deadline.expired()),
           "acquisition": log.as_dict()}
    _write(out / "acquire.json", rep)
    log.write(out / "acquisition_log.json")
    print(f"[growth-stage2] measure: {rep['n_measured']}/{rep['n_shortlist']} measured "
          f"in {rep['elapsed_s']}s (budget {mast.budget_s}s)")
    return rep


def _pick_toi_row(toi_rows, entry: dict, koi_row: dict | None) -> dict | None:
    """The TOI row for this shortlist entry: by ``toi`` id, else by TIC + period."""
    want = _f(entry.get("toi"))
    if np.isfinite(want):
        for r in toi_rows:
            if np.isfinite(_f(r.get("toi"))) and abs(_f(r.get("toi")) - want) < 1e-6:
                return r
    try:
        tid = int(float(entry.get("tic_id")))
    except (TypeError, ValueError):
        tid = None
    same = [r for r in toi_rows
            if tid is not None and np.isfinite(_f(r.get("tid"))) and int(_f(r.get("tid"))) == tid]
    if not same:
        return None
    p = _f((koi_row or {}).get("koi_period"))
    if np.isfinite(p) and p > 0:
        best, bd = None, np.inf
        for r in same:
            pt = _f(r.get("pl_orbper"))
            if np.isfinite(pt) and pt > 0 and abs(pt - p) / p < bd:
                best, bd = r, abs(pt - p) / p
        if best is not None and bd < 1e-2:
            return best
    return same[0]


def stage2_assess(conf: dict, out: Path, *, measurements: pd.DataFrame | None = None,
                  acquire_report: dict | None = None) -> dict:
    """The three-way verdict per target, the run verdict, ``summary.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if measurements is None:
        p = out / "measurements.csv"
        measurements = (pd.read_csv(p, low_memory=False) if p.exists() and p.stat().st_size
                        else pd.DataFrame())
    if acquire_report is None and (out / "acquire.json").exists():
        try:
            acquire_report = json.loads((out / "acquire.json").read_text())
        except Exception:                                 # noqa: BLE001
            acquire_report = None
    verdict, reason = run_verdict(measurements)
    v = (measurements["verdict"].fillna(MATCH_UNMEASURED).astype(str)
         if len(measurements) and "verdict" in measurements else pd.Series(dtype=str))
    counts = {k: int((v == k).sum()) for k in TARGET_VERDICTS}
    unmeasured = (measurements.loc[v == MATCH_UNMEASURED, "unmeasured_reason"]
                  .fillna("").astype(str).value_counts().to_dict() if len(measurements)
                  and "unmeasured_reason" in measurements else {})
    flags = {}
    if len(measurements) and "flags" in measurements:
        # pandas 3 keeps NaN as a float through .astype(str), so every value
        # is coerced here rather than assumed to be a string.
        for row in measurements["flags"]:
            for fl in str(row if row == row else "").split(";"):
                if fl:
                    flags[fl] = flags.get(fl, 0) + 1
    cmp_p = CompareParams.from_config(conf)
    fit = FitParams.from_config(conf)
    mast = MastParams.from_config(conf)
    summary = {
        "verdict": verdict, "reason": reason, "generated_utc": _now(),
        "n_shortlist": int(len(measurements)),
        "n_measured": int(counts[MATCH_KEPLER] + counts[MATCH_TOI] + counts[MATCH_NEITHER]
                          + counts[MATCH_BOTH]),
        "target_verdicts": counts,
        "unmeasured_reasons": unmeasured,
        "flags": flags,
        "targets": (measurements[[c for c in SUMMARY_TARGET_COLUMNS
                                  if c in measurements.columns]].to_dict(orient="records")
                    if len(measurements) else []),
        "acquisition": {k: (acquire_report or {}).get(k) for k in
                        ("koi_ephemeris_status", "toi_status", "lc_status_counts", "lc_routes",
                         "budget_s", "elapsed_s", "budget_exhausted")},
        "config": {"compare": {"n_agree": cmp_p.n_agree, "sigma_sys_ln": cmp_p.sigma_sys_ln,
                               "apply_band_ratio": cmp_p.apply_band_ratio},
                   "fit": {k: getattr(fit, k) for k in fit.__dataclass_fields__},
                   "mast": {"authors": list(mast.authors), "budget_s": mast.budget_s,
                            "per_target_budget_s": mast.per_target_budget_s,
                            "target_timeout_s": mast.target_timeout_s}},
        "checks_not_performed": [
            "per_pixel_centroid_test (a deeper TESS transit can ORIGINATE on a different "
            "star inside the pixel; nothing here excludes that)",
            "kepler_era_lightcurve_refit (the Kepler depth is still the catalogue's)",
            "achromaticity (one band per epoch, as at stage 1)",
        ],
        "note": ("stage 2 measures the TESS depth FROM THE LIGHT CURVE and compares it with "
                 "BOTH catalogue depths; MEASURED_DEPTH_MATCHES_TOI means the depth really "
                 "changed and the candidate goes on to the centroid test, it is NOT a "
                 "detection; NO_DATA_REACHED is not a null result and is not written up "
                 "(CLAUDE.md)"),
    }
    _write(out / "summary.json", summary)
    print(f"[growth-stage2] assess: {verdict} — {reason}; {counts}")
    return summary


SUMMARY_TARGET_COLUMNS = (
    "kepoi_name", "kepler_name", "kepid", "tic_id", "toi", "verdict", "unmeasured_reason",
    "lc_status", "lc_route", "n_sectors", "n_sectors_measured", "sector_list", "authors",
    "exptimes_s", "period_days", "t0_bkjd", "t0_btjd_used", "n_epochs_propagated",
    "ephemeris_sigma_minutes", "duration_hours", "n_transits", "depth_measured_ppm",
    "depth_measured_err_ppm", "depth_measured_err_method", "depth_kepler_ppm",
    "depth_kepler_in_tess_band_ppm", "depth_toi_ppm", "ld_band_ratio", "z_vs_kepler",
    "z_vs_toi", "z_toi_vs_kepler", "references_separated", "n_references_usable",
    "agrees_with_kepler", "agrees_with_toi", "smeared", "depth_is_lower_bound",
    "odd_even_diff_ppm", "odd_even_sigma", "sector_scatter_chi2_per_dof", "flags",
)


def stage2_run(stage: str = "all", *, out_dir=None, conf: dict | None = None, query_fn=None,
               lc_fn=None, shortlist: pd.DataFrame | None = None) -> dict:
    """Run one stage, a comma list, or all.  Returns the last stage's report."""
    from .run import load_growth_config  # noqa: PLC0415

    conf = conf if conf is not None else load_growth_config()
    out = Path(out_dir) if out_dir else Path("results") / "growth" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage2_probe(conf, out)
        elif s == "measure":
            rep = stage2_measure(conf, out, query_fn=query_fn, lc_fn=lc_fn, shortlist=shortlist)
        elif s == "assess":
            rep = stage2_assess(conf, out)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="seti growth-stage2",
        description="GROWTH stage 2 (S57): measure the TESS depth from the light curve and "
                    "compare it with BOTH catalogue depths")
    p.add_argument("--stage", default="all", choices=("probe", "measure", "assess", "all"))
    p.add_argument("--out-dir", default="results/growth/stage2", help="results directory")
    a = p.parse_args(argv)
    rep = stage2_run(a.stage, out_dir=a.out_dir)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[growth-stage2] verdict: {v}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BKJD_MINUS_BTJD", "BKJD_OFFSET", "BTJD_OFFSET", "CompareParams", "DEFAULT_AUTHORS",
    "Deadline", "FitParams", "MATCH_BOTH", "MATCH_KEPLER", "MATCH_NEITHER", "MATCH_TOI",
    "MATCH_UNMEASURED", "MastParams", "RUN_CONFIRMED", "RUN_NO_DATA", "RUN_REFUTED",
    "RUN_UNRESOLVED", "RUN_VERDICTS", "STAGES", "TARGET_VERDICTS", "UNMEASURED_BUDGET",
    "UNMEASURED_NO_EPHEMERIS", "UNMEASURED_NO_REFERENCE", "UNMEASURED_NO_TRANSIT",
    "UNMEASURED_QUERY_FAILED",
    "UNMEASURED_REASONS", "UNMEASURED_ZERO_ROWS", "binned_fold", "bkjd_to_bjd", "bkjd_to_btjd",
    "btjd_to_bjd", "btjd_to_bkjd", "combine_transit_depths", "compare_three_depths",
    "core_half_width", "default_lc_fn", "fetch_koi_ephemerides", "fetch_lightcurves",
    "fetch_toi_depths", "fit_transits", "fold", "lightkurve_lc_fn", "load_shortlist", "main",
    "mast_fits_lc_fn", "mast_probe", "measure_one", "measure_target", "propagate_epoch",
    "read_tess_lc_fits", "run_verdict", "stage2_assess", "stage2_measure", "stage2_probe",
    "stage2_run", "synth_lightcurve", "trapezoid_transit",
]

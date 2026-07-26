"""Does the second moment grow?  Regression of season scatter on calendar time.

Given the bias-corrected per-season excess variance from :mod:`seti.rust.scatter`,
this module asks the channel's actual question --- *is the scatter increasing
over the decade* --- and then spends most of its lines trying to talk itself out
of a yes.

The statistic is a weighted linear fit of **excess variance** (mag^2), not of
amplitude (mag).  Variance is the additive, unbiased quantity: the excess
variance of independent contributions adds, and it is allowed to be negative, so
the regression is unbiased at the noise floor.  Amplitudes are reported for
interpretation, derived from the *fitted* variances.

Four independent guards, because a single positive slope is worth nothing:

* ``rank_rho`` --- Spearman correlation of excess variance with season index.  A
  distribution-free confirmation that survives one wild season.
* ``slope_sigma_loo_min`` --- the weakest slope significance over all
  leave-one-season-out refits.  A trend carried by a single season fails this.
* ``monotonic_frac`` --- fraction of season-to-season steps in the rising sense.
* ``ensemble_detrend_scatter`` --- the per-CCD, per-season common mode in the
  *second* moment, removed before scoring.  This is the fourth and most
  important layer of the cadence/calibration defence described in
  :mod:`seti.rust.scatter`, and it is the direct analogue of what
  ``seti.dimming.run._ensemble_detrend_secular`` does for the first moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scatter import SeasonScatter


@dataclass
class RustStats:
    """The RUST detection statistic for one light curve in one band."""

    n_epochs: int
    n_seasons: int
    baseline_yr: float
    mag_med: float
    slope_var_yr: float        # d(excess variance)/dt, mag^2/yr; >0 = getting messier
    slope_sigma: float         # |slope| / slope_err, reduced-chi2 inflated
    v_first: float             # fitted excess variance at the first season
    v_last: float              # fitted excess variance at the last season
    amp_first_mmag: float      # sqrt of the above, in mmag (0 if fitted negative)
    amp_last_mmag: float
    amp_growth: float          # amp_last / amp_first (inf if amp_first == 0)
    monotonic_frac: float
    rank_rho: float            # Spearman rho of v_exc vs season index
    rank_p: float              # exact one-sided p-value for that rho
    slope_sigma_loo_min: float
    frac_seasons_positive: float
    score: float               # [0,1] rust-likeness (rising scatter only)

    def as_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = int(v) if isinstance(v, int) and not isinstance(v, bool) else float(v)
        return out


def _weighted_line(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least-squares line fit; returns ``(slope, intercept, slope_err)``.

    The slope error is inflated by ``sqrt(max(chi2/dof, 1))`` --- the same
    discipline ``seti.dimming.secular.fit_secular`` uses --- so a star whose
    season-to-season scatter exceeds its formal errors cannot buy significance it
    has not earned.
    """
    Sw = float(w.sum())
    Sx = float((w * x).sum())
    Sy = float((w * y).sum())
    Sxx = float((w * x * x).sum())
    Sxy = float((w * x * y).sum())
    denom = Sw * Sxx - Sx * Sx
    if not np.isfinite(denom) or denom <= 0:
        return None
    slope = (Sw * Sxy - Sx * Sy) / denom
    intercept = (Sy - slope * Sx) / Sw
    resid = y - (intercept + slope * x)
    dof = max(x.size - 2, 1)
    chi2 = float((w * resid ** 2).sum())
    slope_err = np.sqrt((Sw / denom) * max(chi2 / dof, 1.0))
    return float(slope), float(intercept), float(slope_err)


def _spearman(y: np.ndarray) -> float:
    """Spearman rho of ``y`` against its own (time-ordered) index."""
    n = y.size
    if n < 3:
        return 0.0
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    idx = np.arange(1, n + 1, dtype=float)
    ry, ri = ranks - ranks.mean(), idx - idx.mean()
    den = np.sqrt(float((ry ** 2).sum()) * float((ri ** 2).sum()))
    return float((ry * ri).sum() / den) if den > 0 else 0.0


_RHO_NULL: dict[int, np.ndarray] = {}


def _rho_null(n: int) -> np.ndarray:
    """Exact null distribution of Spearman rho by enumerating all ``n!`` orderings."""
    if n in _RHO_NULL:
        return _RHO_NULL[n]
    from itertools import permutations

    idx = np.arange(1, n + 1, dtype=float)
    ri = idx - idx.mean()
    den = float((ri ** 2).sum())          # equal for any permutation of 1..n
    perms = np.array(list(permutations(idx)), dtype=float)
    vals = np.sort(((perms - idx.mean()) @ ri) / den)
    _RHO_NULL[n] = vals
    return vals


def spearman_p(rho: float, n: int) -> float:
    """One-sided p-value for a *positive* Spearman rho: ``P(rho_null >= rho)``.

    Exact by permutation enumeration for ``n <= 8`` (a decade of ZTF gives 7-8
    seasons, so the exact case is the normal case); Student-t approximation
    above.  This is the channel's primary significance because the growth law of
    a collisional cascade is **not known** --- Lacki gives a collisional
    timescale, not a light-curve shape --- so assuming the second moment rises
    *linearly* would be assuming a result.  Monotonicity is the physical claim;
    a rank test is the statistic that tests exactly that and nothing more.
    """
    if n < 3 or not np.isfinite(rho):
        return 1.0
    if rho <= 0:
        return 1.0
    if n <= 8:
        null = _rho_null(n)
        return float(np.mean(null >= rho - 1e-12))
    if rho >= 1.0:
        rho = 1.0 - 1e-12
    from scipy import stats

    t = rho * np.sqrt((n - 2) / max(1.0 - rho ** 2, 1e-15))
    return float(stats.t.sf(t, n - 2))


def fit_scatter_trend(ss: SeasonScatter, n_epochs: int = 0,
                      min_seasons: int = 4) -> RustStats | None:
    """Regress bias-corrected season excess variance against calendar time."""
    if ss is None or len(ss) < min_seasons:
        return None
    yr = (ss.t - ss.t.min()) / 365.25
    v = ss.v_exc
    w = 1.0 / np.maximum(ss.v_err, 1e-12) ** 2
    fit = _weighted_line(yr, v, w)
    if fit is None:
        return None
    slope, intercept, slope_err = fit
    slope_sigma = abs(slope) / slope_err if slope_err > 0 else 0.0

    v_first = intercept + slope * float(yr.min())
    v_last = intercept + slope * float(yr.max())
    amp_first = float(np.sqrt(max(v_first, 0.0)))
    amp_last = float(np.sqrt(max(v_last, 0.0)))
    amp_growth = (amp_last / amp_first) if amp_first > 0 else float("inf")

    steps = np.diff(v)
    sgn = 1.0 if slope >= 0 else -1.0
    monotonic_frac = float(np.mean(np.sign(steps) == sgn)) if steps.size else 0.0
    rho = _spearman(v)
    rho_p = spearman_p(rho, len(ss))
    frac_pos = float(np.mean(v > 0))

    # Leave-one-season-out: the weakest slope significance over all refits that
    # keep the sign.  A trend created by one anomalous season collapses here.
    loo = []
    for i in range(len(ss)):
        keep = np.ones(len(ss), dtype=bool)
        keep[i] = False
        f = _weighted_line(yr[keep], v[keep], w[keep])
        if f is None:
            loo.append(0.0)
            continue
        s_i, _, e_i = f
        loo.append((s_i / e_i) if (e_i > 0 and np.sign(s_i) == sgn) else 0.0)
    loo_min = float(min(loo)) if loo else 0.0

    if slope <= 0:
        score = 0.0
    else:
        # The rank p-value carries the most weight: it is the only term that does
        # not assume the cascade grows linearly.
        rank_term = float(np.clip((-np.log10(max(rho_p, 1e-6)) - 1.0) / 3.0, 0, 1))
        sig_term = float(np.clip((slope_sigma - 3.0) / 5.0, 0, 1))
        loo_term = float(np.clip((loo_min - 2.0) / 4.0, 0, 1))
        growth_term = float(np.clip((amp_growth - 1.3) / 1.7, 0, 1)) \
            if np.isfinite(amp_growth) else 1.0
        mono_term = float(np.clip((monotonic_frac - 0.5) / 0.4, 0, 1))
        score = float(np.clip(0.35 * rank_term + 0.25 * sig_term + 0.20 * loo_term
                              + 0.15 * growth_term + 0.05 * mono_term, 0, 1))

    return RustStats(
        n_epochs=int(n_epochs or int(ss.n.sum())), n_seasons=len(ss),
        baseline_yr=float(yr.max() - yr.min()),
        mag_med=float(np.median(ss.mag_med)),
        slope_var_yr=float(slope), slope_sigma=float(slope_sigma),
        v_first=float(v_first), v_last=float(v_last),
        amp_first_mmag=1e3 * amp_first, amp_last_mmag=1e3 * amp_last,
        amp_growth=float(amp_growth), monotonic_frac=monotonic_frac,
        rank_rho=rho, rank_p=rho_p, slope_sigma_loo_min=loo_min,
        frac_seasons_positive=frac_pos, score=score,
    )


def detect_rust(time, mag, magerr=None, *, season_days: float = 365.25,
                min_epochs_season: int = 8, min_seasons: int = 4,
                equalize_n: bool = False) -> RustStats | None:
    """Convenience: season-scatter then trend, in one call.  ``None`` if unusable."""
    from .scatter import season_scatter

    ss = season_scatter(time, mag, magerr, season_days=season_days,
                        min_epochs_season=min_epochs_season,
                        min_seasons=min_seasons, equalize_n=equalize_n)
    if ss is None:
        return None
    n_good = int(np.sum(np.isfinite(np.asarray(time, float))
                        & np.isfinite(np.asarray(mag, float))))
    return fit_scatter_trend(ss, n_epochs=n_good, min_seasons=min_seasons)


# --------------------------------------------------------------------------
# Ensemble common mode in the second moment
# --------------------------------------------------------------------------

def ensemble_error_scale(rows: list[dict], min_stars: int = 8) -> dict:
    """Per-CCD, per-season empirical error-scale factor ``kappa`` from the field.

    ZTF's reported ``magerr`` is a model, and the model drifts: seeing,
    background, the ZTF-I -> ZTF-II cadence change and reference-image rebuilds
    all shift the true-to-reported error ratio *as a function of calendar time*.
    An uncorrected drift in that ratio is exactly a spurious secular trend in
    excess variance --- the single most dangerous residual systematic left after
    the per-season bias correction.

    The field ensemble measures it directly.  For every star on a readout
    channel, ``sigma_obs^2 / sigma_null^2`` should be ~1 in every season if the
    errors are right and the star is constant.  The **median** of that ratio over
    many stars is dominated by the constant majority, so it estimates the
    season's error-scale factor ``kappa`` regardless of a few real variables.

    Returns ``{ccd: {season_label: kappa}}`` plus a ``"__global__"`` fallback for
    thinly-populated channels.
    """
    from collections import defaultdict

    ratios: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    glob: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        ss = r.get("_ss")
        if ss is None or len(ss) == 0:
            continue
        ccd = str(r.get("_ccd", "x"))
        for lab, so, sn in zip(ss.label, ss.sigma_obs, ss.sigma_null, strict=False):
            if not (np.isfinite(so) and np.isfinite(sn) and sn > 0):
                continue
            k = float(so ** 2 / sn ** 2)
            ratios[ccd][int(lab)].append(k)
            glob[int(lab)].append(k)
    out: dict = {ccd: {lab: float(np.median(v)) for lab, v in seasons.items()
                       if len(v) >= min_stars}
                 for ccd, seasons in ratios.items()}
    out["__global__"] = {lab: float(np.median(v)) for lab, v in glob.items()
                         if len(v) >= min_stars}
    return out


def apply_error_scale(ss: SeasonScatter, kappa_by_season: dict) -> SeasonScatter:
    """Recompute excess variance with the ensemble error-scale factor applied."""
    k = np.array([float(kappa_by_season.get(int(lab), 1.0)) for lab in ss.label])
    k = np.where(np.isfinite(k) & (k > 0), k, 1.0)
    sigma_null = ss.sigma_null * np.sqrt(k)
    v_exc = ss.sigma_obs ** 2 - np.where(np.isfinite(sigma_null), sigma_null, 0.0) ** 2
    return SeasonScatter(label=ss.label, t=ss.t, n=ss.n, mag_med=ss.mag_med,
                         sigma_obs=ss.sigma_obs, sigma_null=sigma_null,
                         v_exc=v_exc, v_err=ss.v_err)


def additive_common_mode(rows: list[dict], min_stars: int = 8) -> dict:
    """Residual additive common mode in excess variance, per CCD and season.

    Run *after* the multiplicative error-scale correction to mop up whatever it
    did not explain (a season of poor subtractions, a shared blending event).
    Each star contributes ``v_exc(season) - median_over_seasons(v_exc)`` so its
    own mean level cancels and only the shared seasonal shape survives.
    """
    from collections import defaultdict

    off: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    glob: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        ss = r.get("_ss")
        if ss is None or len(ss) == 0:
            continue
        base = float(np.median(ss.v_exc))
        ccd = str(r.get("_ccd", "x"))
        for lab, v in zip(ss.label, ss.v_exc, strict=False):
            if not np.isfinite(v):
                continue
            off[ccd][int(lab)].append(float(v) - base)
            glob[int(lab)].append(float(v) - base)
    out: dict = {ccd: {lab: float(np.median(v)) for lab, v in seasons.items()
                       if len(v) >= min_stars}
                 for ccd, seasons in off.items()}
    out["__global__"] = {lab: float(np.median(v)) for lab, v in glob.items()
                         if len(v) >= min_stars}
    return out


def apply_additive(ss: SeasonScatter, cm_by_season: dict) -> SeasonScatter:
    """Subtract the additive common mode from a star's season excess variances."""
    c = np.array([float(cm_by_season.get(int(lab), 0.0)) for lab in ss.label])
    c = np.where(np.isfinite(c), c, 0.0)
    return SeasonScatter(label=ss.label, t=ss.t, n=ss.n, mag_med=ss.mag_med,
                         sigma_obs=ss.sigma_obs, sigma_null=ss.sigma_null,
                         v_exc=ss.v_exc - c, v_err=ss.v_err)


def ensemble_detrend_scatter(rows: list[dict], min_stars: int = 8,
                             min_seasons: int = 4) -> dict:
    """Full ensemble detrend of the second moment, in place on ``rows``.

    Each row must carry ``_ss`` (a :class:`SeasonScatter`) and optionally
    ``_ccd``.  Applies the multiplicative error-scale correction then the
    additive common mode, refits every star, and writes the corrected statistics
    back onto the row.  Returns a small diagnostic dict describing how large the
    corrections were --- if they are big, say so in the write-up.
    """
    kap = ensemble_error_scale(rows, min_stars=min_stars)
    kap_glob = kap.get("__global__", {})
    for r in rows:
        ss = r.get("_ss")
        if ss is None:
            continue
        by = {**kap_glob, **kap.get(str(r.get("_ccd", "x")), {})}
        r["_ss"] = apply_error_scale(ss, by)

    add = additive_common_mode(rows, min_stars=min_stars)
    add_glob = add.get("__global__", {})
    for r in rows:
        ss = r.get("_ss")
        if ss is None:
            continue
        by = {**add_glob, **add.get(str(r.get("_ccd", "x")), {})}
        ss = apply_additive(ss, by)
        r["_ss"] = ss
        stat = fit_scatter_trend(ss, n_epochs=int(r.get("_nepoch", 0)),
                                 min_seasons=min_seasons)
        r["_stat"] = stat
        if stat is not None:
            r.update({f"det_{k}": v for k, v in stat.as_dict().items()})

    kappas = [v for d in kap.values() for v in d.values()]
    adds = [v for d in add.values() for v in d.values()]
    # A thin field cannot support an ensemble measurement, and a run in that
    # state is NOT ensemble-corrected however much its column names suggest it
    # is.  Say so as a first-class field rather than emitting NaNs that a reader
    # might mistake for "the correction was zero".
    applied = bool(kappas)
    n_glob = len(kap.get("__global__", {}))
    return {
        "ensemble_correction_applied": applied,
        "ensemble_verdict": ("APPLIED" if applied else
                             "NOT_APPLIED_TOO_FEW_STARS"),
        "n_ccd_channels": len([k for k in kap if k != "__global__"]),
        "n_seasons_with_global_kappa": int(n_glob),
        "n_stars": int(len(rows)),
        "kappa_median": float(np.median(kappas)) if kappas else float("nan"),
        "kappa_min": float(np.min(kappas)) if kappas else float("nan"),
        "kappa_max": float(np.max(kappas)) if kappas else float("nan"),
        "additive_cm_max_mmag": (float(1e3 * np.sqrt(np.max(np.abs(adds))))
                                 if adds else float("nan")),
    }


__all__ = ["RustStats", "additive_common_mode", "apply_additive", "apply_error_scale",
           "detect_rust", "ensemble_detrend_scatter", "ensemble_error_scale",
           "fit_scatter_trend", "spearman_p"]

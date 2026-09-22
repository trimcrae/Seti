"""The Menzel-gap step model --- a STEP, never a trend.

The Harvard plate patrol stopped in 1953/54 under HCO director Donald Menzel
and resumed around 1969/70 with different telescopes, emulsions and observing
programmes.  Lund, Pepper, Stassun & Hippke 2016 (arXiv:1605.02760) and Hippke
et al. 2016 showed that the apparent century-long dimming of KIC 8462852 in
DASCH is what a **photometric offset between the pre-gap and post-gap plates**
looks like when it is fitted with a straight line.  Any DASCH statistic that
compares early to late has to carry that offset as a nuisance parameter, and
the correct model for it is a step at the gap, because the plates on either
side are different physical objects.

So every "does this quantity change with calendar time" question in this
channel is fitted as

    q(t) = q0 + s * (t - t_ref) + D * H(t - t_gap)

by weighted least squares, and the answer is the slope ``s`` **with the step
``D`` free**.  Two further numbers are always reported next to it:

* ``s_naive`` --- the slope with ``D`` frozen at zero.  Its difference from
  ``s`` is the Hippke/Lund number: how much of the "trend" was the gap.
* ``s_pre``, ``s_post`` --- the slopes fitted inside each segment alone.  A
  real secular change is present *within* the pre-gap segment (sixty years of
  leverage); a trend that exists only across the gap is not a trend.

All pure functions of arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MENZEL_GAP_START = 1954.0
MENZEL_GAP_END = 1970.0


def segment(year, gap=(MENZEL_GAP_START, MENZEL_GAP_END)) -> np.ndarray:
    """0 before the gap, 1 after it, -1 inside it."""
    y = np.asarray(year, dtype=float)
    out = np.full(y.shape, -1, dtype=int)
    out[y < float(gap[0])] = 0
    out[y >= float(gap[1])] = 1
    return out


@dataclass
class StepFit:
    """Weighted level + slope + step fit with per-segment slopes."""

    n: int = 0
    n_pre: int = 0
    n_post: int = 0
    t_ref: float = float("nan")
    level: float = float("nan")
    slope: float = float("nan")          # per year
    slope_err: float = float("nan")
    slope_sigma: float = float("nan")
    step: float = float("nan")
    step_err: float = float("nan")
    step_sigma: float = float("nan")
    chi2: float = float("nan")
    dof: int = 0
    chi2_dof: float = float("nan")
    slope_naive: float = float("nan")    # step frozen at zero
    slope_naive_err: float = float("nan")
    slope_naive_sigma: float = float("nan")
    slope_pre: float = float("nan")
    slope_pre_err: float = float("nan")
    slope_pre_sigma: float = float("nan")
    slope_post: float = float("nan")
    slope_post_err: float = float("nan")
    slope_post_sigma: float = float("nan")
    span_pre_yr: float = float("nan")
    span_post_yr: float = float("nan")
    step_identifiable: bool = False      # both segments populated
    ok: bool = False

    def as_dict(self) -> dict:
        return {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                    else (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)))
                for k, v in self.__dict__.items()}


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares; returns ``(beta, cov, chi2, dof)`` with the
    covariance inflated by ``max(chi2/dof, 1)``, the repository's discipline."""
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    try:
        cov = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        return None
    beta = cov @ (Xw.T @ yw)
    resid = yw - Xw @ beta
    chi2 = float(resid @ resid)
    dof = max(len(y) - X.shape[1], 1)
    infl = max(chi2 / dof, 1.0)
    return beta, cov * infl, chi2, dof


def _slope_only(x, y, w):
    if len(x) < 3 or np.ptp(x) <= 0:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x - x.mean()])
    r = _wls(X, y, w)
    if r is None:
        return float("nan"), float("nan")
    beta, cov, _, _ = r
    return float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0)))


def fit_step_slope(year, q, err=None, *, gap=(MENZEL_GAP_START, MENZEL_GAP_END),
                   min_points: int = 6, min_per_segment: int = 3) -> StepFit:
    """Fit ``q0 + s (t - t_ref) + D H(t - gap)`` to ``(year, q)`` with weights
    ``1/err^2``.

    If only one segment is populated the step is unidentifiable: it is frozen
    at zero, ``step_identifiable`` is False, and ``slope`` equals ``slope_naive``
    --- which is the honest statement that no gap correction was possible.
    """
    x = np.asarray(year, dtype=float)
    y = np.asarray(q, dtype=float)
    e = (np.ones_like(x) if err is None else np.asarray(err, dtype=float))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    x, y, e = x[m], y[m], e[m]
    out = StepFit(n=int(len(x)))
    if len(x) < int(min_points) or np.ptp(x) <= 0:
        return out
    w = 1.0 / e ** 2
    seg = segment(x, gap)
    pre, post = seg == 0, seg == 1
    out.n_pre, out.n_post = int(pre.sum()), int(post.sum())
    out.t_ref = float(np.average(x, weights=w))
    xc = x - out.t_ref
    out.step_identifiable = bool(out.n_pre >= min_per_segment and out.n_post >= min_per_segment)

    # naive (no step)
    Xn = np.column_stack([np.ones_like(xc), xc])
    rn = _wls(Xn, y, w)
    if rn is None:
        return out
    bn, cn, _, _ = rn
    out.slope_naive = float(bn[1])
    out.slope_naive_err = float(np.sqrt(max(cn[1, 1], 0.0)))
    out.slope_naive_sigma = (abs(out.slope_naive) / out.slope_naive_err
                             if out.slope_naive_err > 0 else float("nan"))

    if out.step_identifiable:
        H = (x >= float(gap[1])).astype(float)
        X = np.column_stack([np.ones_like(xc), xc, H])
        r = _wls(X, y, w)
        if r is None:
            return out
        beta, cov, chi2, dof = r
        out.level, out.slope, out.step = float(beta[0]), float(beta[1]), float(beta[2])
        out.slope_err = float(np.sqrt(max(cov[1, 1], 0.0)))
        out.step_err = float(np.sqrt(max(cov[2, 2], 0.0)))
        out.chi2, out.dof, out.chi2_dof = float(chi2), int(dof), float(chi2 / dof)
    else:
        beta, cov, chi2, dof = rn
        out.level, out.slope, out.step = float(beta[0]), float(beta[1]), 0.0
        out.slope_err = out.slope_naive_err
        out.step_err = float("nan")
        out.chi2, out.dof, out.chi2_dof = float(chi2), int(dof), float(chi2 / dof)
    out.slope_sigma = abs(out.slope) / out.slope_err if out.slope_err > 0 else float("nan")
    out.step_sigma = (abs(out.step) / out.step_err
                      if (np.isfinite(out.step_err) and out.step_err > 0) else float("nan"))

    if out.n_pre >= min_per_segment:
        out.slope_pre, out.slope_pre_err = _slope_only(x[pre], y[pre], w[pre])
        out.span_pre_yr = float(np.ptp(x[pre]))
    if out.n_post >= min_per_segment:
        out.slope_post, out.slope_post_err = _slope_only(x[post], y[post], w[post])
        out.span_post_yr = float(np.ptp(x[post]))
    out.slope_pre_sigma = (abs(out.slope_pre) / out.slope_pre_err
                           if np.isfinite(out.slope_pre_err) and out.slope_pre_err > 0
                           else float("nan"))
    out.slope_post_sigma = (abs(out.slope_post) / out.slope_post_err
                            if np.isfinite(out.slope_post_err) and out.slope_post_err > 0
                            else float("nan"))
    out.ok = True
    return out


def segment_consistent(fit: StepFit, *, sigma: float = 2.5) -> bool:
    """Is the joint slope consistent with the slope inside the pre-gap segment?

    A century trend that the sixty pre-gap years do not show on their own is
    a statement about the gap, not about the star.
    """
    if not fit.ok or not np.isfinite(fit.slope_pre):
        return False
    d = fit.slope - fit.slope_pre
    s = float(np.hypot(fit.slope_err, fit.slope_pre_err))
    return bool(s > 0 and abs(d) <= sigma * s)


__all__ = ["MENZEL_GAP_END", "MENZEL_GAP_START", "StepFit", "fit_step_slope", "segment",
           "segment_consistent"]

"""Secular fade on the century baseline --- with the step, never the trend.

The statistic is the slope of the star's annual median magnitude against
calendar year, fitted **with the Menzel-gap step free** (:mod:`.step`), and
three robustness refits that each remove one named plate systematic:

* **plate-limit margin** --- only plates whose limit is at least ``margin``
  deeper than the star's median are used, and the fit is repeated at a deeper
  margin.  A fade that weakens as shallow plates are removed was the censoring
  bias of shallow plates (only bright excursions get measured near the limit),
  and the plate-depth history is a calendar-time history;
* **single series** --- the fit is repeated inside the dominant plate series.
  Emulsion changes are series changes;
* **pre-gap segment alone** --- the sixty pre-gap years must show the trend on
  their own (:func:`seti.century.step.segment_consistent`).

For catalogued variables the margin is raised to ``amplitude + 0.5`` so the
faint phase is never censored on the plates that are kept, and the annual
median is taken over the whole cycle.

The field common mode (every star on the same plates shares the same emulsion,
depth and calibration history) is removed at assess time from the annual
tables of all stars in the field, per magnitude bin; :func:`fade_from_annual`
is what both the per-star screen and the corrected assess call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lightcurve import CenturyLC, annual_table
from .step import MENZEL_GAP_END, MENZEL_GAP_START, StepFit, fit_step_slope, segment_consistent


@dataclass
class FadeResult:
    status: str = "insufficient_data"
    n_years: int = 0
    n_years_pre: int = 0
    n_years_post: int = 0
    span_yr: float = float("nan")
    margin_used: float = float("nan")
    frac_plates_kept: float = float("nan")
    median_mag: float = float("nan")
    slope_mag_per_century: float = float("nan")
    slope_sigma: float = float("nan")
    slope_naive_mag_per_century: float = float("nan")
    slope_naive_sigma: float = float("nan")
    step_mag: float = float("nan")
    step_sigma: float = float("nan")
    step_identifiable: bool = False
    slope_pre_mag_per_century: float = float("nan")
    slope_pre_sigma: float = float("nan")
    slope_post_mag_per_century: float = float("nan")
    slope_post_sigma: float = float("nan")
    segment_consistent: bool = False
    chi2_dof: float = float("nan")
    slope_deep_mag_per_century: float = float("nan")
    slope_deep_sigma: float = float("nan")
    deep_consistent: bool = False
    series_name: str = ""
    slope_series_mag_per_century: float = float("nan")
    slope_series_sigma: float = float("nan")
    series_consistent: bool = False
    series_span_yr: float = float("nan")
    blend_frac_trend: float = float("nan")
    lim_trend_mag_per_century: float = float("nan")
    is_fade: bool = False
    is_brightening: bool = False
    flags: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _per_century(fit: StepFit, attr: str) -> tuple[float, float]:
    v = getattr(fit, attr)
    s = getattr(fit, attr + "_sigma")
    return (float(v) * 100.0 if np.isfinite(v) else float("nan"),
            float(s) if np.isfinite(s) else float("nan"))


def _consistent(a: float, sa: float, b: float, sb: float, nsig: float = 2.5) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    ea = abs(a) / sa if (np.isfinite(sa) and sa > 0) else float("nan")
    eb = abs(b) / sb if (np.isfinite(sb) and sb > 0) else float("nan")
    if not (np.isfinite(ea) and np.isfinite(eb)):
        return False
    s = float(np.hypot(ea, eb))
    return bool(s > 0 and abs(a - b) <= nsig * s)


def fade_from_annual(tab: pd.DataFrame, *, min_years: int = 15, min_span_yr: float = 30.0,
                     gap=(MENZEL_GAP_START, MENZEL_GAP_END)) -> StepFit:
    """Step + slope fit of an annual table (``year, med, err``)."""
    if tab is None or not len(tab):
        return StepFit()
    fit = fit_step_slope(tab["year"].to_numpy(dtype=float) + 0.5,
                         tab["med"].to_numpy(dtype=float),
                         tab["err"].to_numpy(dtype=float), gap=gap)
    if fit.n < int(min_years) or (np.ptp(tab["year"].to_numpy(dtype=float)) < min_span_yr):
        fit.ok = False
    return fit


def analyze_fade(lc: CenturyLC, *, amp_cat: float = 0.0, margin: float = 1.0,
                 deep_extra: float = 1.0, min_per_year: int = 3, min_years: int = 15,
                 min_span_yr: float = 30.0, fade_sigma_min: float = 5.0,
                 pre_sigma_min: float = 3.0, series_min_years: int = 10,
                 series_min_span_yr: float = 25.0,
                 gap=(MENZEL_GAP_START, MENZEL_GAP_END)) -> tuple[FadeResult, pd.DataFrame]:
    """The per-star secular-fade test.  Returns ``(result, annual_table)``; the
    table (margin-cut) is what the assess stage corrects with the field ensemble."""
    res = FadeResult()
    if lc.n_det == 0 or not lc.good.any():
        return res, pd.DataFrame()
    marg = float(max(margin, (amp_cat if np.isfinite(amp_cat) else 0.0) + 0.5))
    res.margin_used = marg
    res.median_mag = lc.median_mag()
    m = lc.margin_mask(marg)
    res.frac_plates_kept = float(m.sum() / max(lc.good.sum(), 1))
    tab = annual_table(lc, mask=m, min_per_year=min_per_year)
    res.n_years = int(len(tab))
    if not len(tab):
        res.flags = "no_years_pass_margin"
        return res, tab
    fit = fade_from_annual(tab, min_years=min_years, min_span_yr=min_span_yr, gap=gap)
    res.span_yr = float(np.ptp(tab["year"].to_numpy(dtype=float)))
    res.n_years_pre, res.n_years_post = fit.n_pre, fit.n_post
    if not fit.ok:
        res.status = "insufficient_data"
        res.flags = "too_few_years"
        return res, tab
    res.slope_mag_per_century, res.slope_sigma = _per_century(fit, "slope")
    res.slope_naive_mag_per_century, res.slope_naive_sigma = _per_century(fit, "slope_naive")
    res.step_mag, res.step_sigma = float(fit.step), float(fit.step_sigma)
    res.step_identifiable = bool(fit.step_identifiable)
    res.slope_pre_mag_per_century, res.slope_pre_sigma = _per_century(fit, "slope_pre")
    res.slope_post_mag_per_century, res.slope_post_sigma = _per_century(fit, "slope_post")
    res.segment_consistent = segment_consistent(fit)
    res.chi2_dof = float(fit.chi2_dof)

    # Plate-limit and blend histories of the kept plates, as trends.
    yrs = tab["year"].to_numpy(dtype=float) + 0.5
    lt = fit_step_slope(yrs, tab["lim_med"].to_numpy(dtype=float), None, gap=gap)
    res.lim_trend_mag_per_century = float(lt.slope) * 100.0 if lt.ok else float("nan")
    bf = tab["blend_frac"].to_numpy(dtype=float)
    if np.isfinite(bf).sum() >= 6:
        bt = fit_step_slope(yrs[np.isfinite(bf)], bf[np.isfinite(bf)], None, gap=gap)
        res.blend_frac_trend = float(bt.slope) * 100.0 if bt.ok else float("nan")

    # Robustness 1: deeper plates only.
    tab_deep = annual_table(lc, mask=lc.margin_mask(marg + deep_extra), min_per_year=min_per_year)
    fd = fade_from_annual(tab_deep, min_years=max(min_years // 2, 6),
                          min_span_yr=min_span_yr * 0.6, gap=gap)
    if fd.ok:
        res.slope_deep_mag_per_century, res.slope_deep_sigma = _per_century(fd, "slope")
        res.deep_consistent = _consistent(res.slope_mag_per_century,
                                          res.slope_mag_per_century / max(res.slope_sigma, 1e-9),
                                          res.slope_deep_mag_per_century,
                                          res.slope_deep_mag_per_century
                                          / max(res.slope_deep_sigma, 1e-9))
    # Robustness 2: the dominant series alone.
    name, _frac = lc.dominant_series(m)
    res.series_name = name
    if name:
        tab_s = annual_table(lc, mask=m & (lc.series.astype(str) == name),
                             min_per_year=min_per_year)
        if len(tab_s):
            res.series_span_yr = float(np.ptp(tab_s["year"].to_numpy(dtype=float)))
        fs = fade_from_annual(tab_s, min_years=series_min_years, min_span_yr=series_min_span_yr,
                              gap=gap)
        if fs.ok:
            res.slope_series_mag_per_century, res.slope_series_sigma = _per_century(fs, "slope")
            res.series_consistent = _consistent(
                res.slope_mag_per_century, res.slope_mag_per_century / max(res.slope_sigma, 1e-9),
                res.slope_series_mag_per_century,
                res.slope_series_mag_per_century / max(res.slope_series_sigma, 1e-9))

    fading = res.slope_mag_per_century > 0
    sig_ok = np.isfinite(res.slope_sigma) and res.slope_sigma >= float(fade_sigma_min)
    pre_ok = (np.isfinite(res.slope_pre_sigma) and res.slope_pre_sigma >= float(pre_sigma_min)
              and np.sign(res.slope_pre_mag_per_century) == np.sign(res.slope_mag_per_century))
    flags = []
    if not sig_ok:
        flags.append("slope_not_significant")
    if not pre_ok:
        flags.append("pre_gap_segment_not_significant")
    if not res.segment_consistent:
        flags.append("segment_inconsistent")
    if np.isfinite(res.slope_deep_mag_per_century) and not res.deep_consistent:
        flags.append("depends_on_shallow_plates")
    if np.isfinite(res.slope_series_mag_per_century) and not res.series_consistent:
        flags.append("depends_on_series")
    if not res.step_identifiable:
        flags.append("step_unidentifiable_single_segment")
    if (np.isfinite(res.slope_naive_sigma) and res.slope_naive_sigma >= fade_sigma_min
            and not sig_ok):
        flags.append("naive_trend_was_the_gap")
    strict = (sig_ok and pre_ok and res.segment_consistent
              and (not np.isfinite(res.slope_deep_mag_per_century) or res.deep_consistent)
              and (not np.isfinite(res.slope_series_mag_per_century) or res.series_consistent))
    res.is_fade = bool(strict and fading)
    res.is_brightening = bool(strict and not fading)
    res.status = ("fade" if res.is_fade else "brightening" if res.is_brightening
                  else "no_secular_change" if not sig_ok else "rejected")
    res.flags = ";".join(flags)
    return res, tab


__all__ = ["FadeResult", "analyze_fade", "fade_from_annual"]

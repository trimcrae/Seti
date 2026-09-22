"""Rising season scatter on the century baseline --- RUST on plates.

The second-moment statistic is inherited whole from :mod:`seti.rust.scatter`
(bias-corrected excess variance per season, at that season's own epoch count
and error vector) and :mod:`seti.rust.trend` (exact rank test, chi2-inflated
slope, leave-one-out).  What changes on plates:

* **The Menzel step.**  Post-gap plates are different emulsions with different
  scatter, so the excess variance is fitted as level + slope + step
  (:mod:`.step`), and the rank test is run **inside the pre-gap segment**,
  where sixty years of leverage live, and separately inside the post-gap one.
* **Plate-limit margin.**  Near the plate limit, scatter is set by the plate
  rather than the star; only plates deeper than the star by a margin are used,
  and the pre-gap rise must survive a deeper margin.
* **Series restriction.**  The pre-gap rise must survive inside the dominant
  plate series; a rise produced by a change of plate scale (blending) or
  emulsion is series-clustered.
* **Periodic stars are residualised** at the reference frequency (two
  harmonics fitted per season) before the scatter is taken, so the statistic
  stays aperiodic.  This removes a few degrees of freedom the null table does
  not know about; the minimum epochs per season is raised for those stars and
  the approximation is recorded.

Pure functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rust.scatter import SeasonScatter, season_scatter
from ..rust.trend import RustStats, fit_scatter_trend
from .lightcurve import CenturyLC, mjd_to_year
from .step import MENZEL_GAP_END, MENZEL_GAP_START, StepFit, fit_step_slope, segment


def residualise_periodic(t, y, e, f_ref: float, *, n_harm: int = 2,
                         season_days: float = 365.25) -> np.ndarray:
    """Subtract a per-season fixed-frequency harmonic model (keeping the level)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    if not (np.isfinite(f_ref) and f_ref > 0) or len(t) == 0:
        return y.copy()
    out = y.copy()
    lab = np.floor((t - t.min()) / float(season_days)).astype(int)
    for k in np.unique(lab):
        m = lab == k
        n = int(m.sum())
        ncoef = 1 + 2 * n_harm
        if n < ncoef + 3:
            continue
        tt = t[m]
        cols = [np.ones(n)]
        for h in range(1, n_harm + 1):
            ph = 2.0 * np.pi * h * f_ref * tt
            cols += [np.cos(ph), np.sin(ph)]
        X = np.column_stack(cols)
        w = 1.0 / np.clip(e[m], 1e-6, None)
        try:
            beta, *_ = np.linalg.lstsq(X * w[:, None], y[m] * w, rcond=None)
        except np.linalg.LinAlgError:
            continue
        model = X @ beta
        out[m] = y[m] - (model - beta[0])
    return out


def season_scatter_from_dict(d: dict) -> SeasonScatter | None:
    """Inverse of :meth:`seti.rust.scatter.SeasonScatter.as_dict` (plus labels)."""
    if not d or not d.get("season_t"):
        return None
    n = len(d["season_t"])
    lab = d.get("season_label") or list(range(n))
    return SeasonScatter(
        label=np.asarray(lab, dtype=int), t=np.asarray(d["season_t"], dtype=float),
        n=np.asarray(d["n_per_season"], dtype=int),
        mag_med=np.asarray(d["season_mag_med"], dtype=float),
        sigma_obs=np.asarray(d["sigma_obs_mmag"], dtype=float) * 1e-3,
        sigma_null=np.asarray(d["sigma_null_mmag"], dtype=float) * 1e-3,
        v_exc=np.asarray(d["v_exc"], dtype=float), v_err=np.asarray(d["v_err"], dtype=float),
    )


def season_scatter_to_dict(ss: SeasonScatter) -> dict:
    d = ss.as_dict()
    d["season_label"] = [int(v) for v in ss.label]
    return d


def subset_scatter(ss: SeasonScatter, mask) -> SeasonScatter:
    m = np.asarray(mask, dtype=bool)
    return SeasonScatter(label=ss.label[m], t=ss.t[m], n=ss.n[m], mag_med=ss.mag_med[m],
                         sigma_obs=ss.sigma_obs[m], sigma_null=ss.sigma_null[m],
                         v_exc=ss.v_exc[m], v_err=ss.v_err[m])


@dataclass
class ScatterResult:
    status: str = "insufficient_data"
    n_epochs: int = 0
    n_seasons: int = 0
    n_seasons_pre: int = 0
    n_seasons_post: int = 0
    baseline_yr: float = float("nan")
    margin_used: float = float("nan")
    residualised: bool = False
    slope_var_per_century: float = float("nan")
    slope_sigma: float = float("nan")
    slope_naive_var_per_century: float = float("nan")
    slope_naive_sigma: float = float("nan")
    step_var: float = float("nan")
    step_sigma: float = float("nan")
    step_identifiable: bool = False
    amp_first_mmag: float = float("nan")
    amp_last_mmag: float = float("nan")
    pre_rank_rho: float = float("nan")
    pre_rank_p: float = float("nan")
    pre_slope_sigma: float = float("nan")
    pre_loo_min: float = float("nan")
    pre_amp_growth: float = float("nan")
    pre_score: float = float("nan")
    post_rank_p: float = float("nan")
    post_slope_sigma: float = float("nan")
    post_score: float = float("nan")
    series_name: str = ""
    series_pre_rank_p: float = float("nan")
    series_pre_slope_sigma: float = float("nan")
    series_pre_slope_positive: bool = False
    deep_pre_rank_p: float = float("nan")
    deep_pre_slope_sigma: float = float("nan")
    deep_pre_slope_positive: bool = False
    is_rust: bool = False
    flags: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _fill_rust(res: ScatterResult, prefix: str, st: RustStats | None) -> None:
    if st is None:
        return
    if prefix == "pre":
        res.pre_rank_rho, res.pre_rank_p = float(st.rank_rho), float(st.rank_p)
        res.pre_slope_sigma = float(st.slope_sigma if st.slope_var_yr > 0 else 0.0)
        res.pre_loo_min, res.pre_amp_growth = float(st.slope_sigma_loo_min), float(st.amp_growth)
        res.pre_score = float(st.score)
    elif prefix == "post":
        res.post_rank_p = float(st.rank_p)
        res.post_slope_sigma = float(st.slope_sigma if st.slope_var_yr > 0 else 0.0)
        res.post_score = float(st.score)


def scatter_table(lc: CenturyLC, mask, *, f_ref: float = float("nan"),
                  season_years: float = 1.0, min_epochs_season: int = 8,
                  min_seasons: int = 6, n_harm: int = 2) -> tuple[SeasonScatter | None, bool]:
    """The per-season table for ``lc[mask]``; residualised when ``f_ref`` is finite."""
    m = np.asarray(mask, dtype=bool) & lc.good
    if m.sum() < min_epochs_season * min_seasons:
        return None, False
    t, y, e = lc.t[m], lc.mag[m], lc.err[m]
    order = np.argsort(t)
    t, y, e = t[order], y[order], e[order]
    resid = bool(np.isfinite(f_ref) and f_ref > 0)
    sd = float(season_years) * 365.25
    if resid:
        y = residualise_periodic(t, y, e, f_ref, n_harm=n_harm, season_days=sd)
        min_epochs_season = max(int(min_epochs_season), 2 * (1 + 2 * n_harm) + 2)
    ss = season_scatter(t, y, e, season_days=sd, min_epochs_season=int(min_epochs_season),
                        min_seasons=int(min_seasons))
    return ss, resid


def scatter_from_table(ss: SeasonScatter, *, gap=(MENZEL_GAP_START, MENZEL_GAP_END),
                       min_seasons_segment: int = 4) -> tuple[StepFit, RustStats | None,
                                                              RustStats | None]:
    """Joint step fit plus per-segment RUST statistics."""
    yr = mjd_to_year(ss.t)
    fit = fit_step_slope(yr, ss.v_exc, ss.v_err, gap=gap)
    seg = segment(yr, gap)
    pre = subset_scatter(ss, seg == 0)
    post = subset_scatter(ss, seg == 1)
    st_pre = fit_scatter_trend(pre, min_seasons=min_seasons_segment) \
        if len(pre) >= min_seasons_segment else None
    st_post = fit_scatter_trend(post, min_seasons=min_seasons_segment) \
        if len(post) >= min_seasons_segment else None
    return fit, st_pre, st_post


def analyze_scatter(lc: CenturyLC, *, f_ref: float = float("nan"), amp_cat: float = 0.0,
                    margin: float = 1.0, deep_extra: float = 1.0, season_years: float = 1.0,
                    min_epochs_season: int = 8, min_seasons: int = 6,
                    slope_sigma_min: float = 3.0, pre_rank_p_max: float = 0.01,
                    pre_slope_sigma_min: float = 3.0, pre_loo_min: float = 2.0,
                    robust_sigma_min: float = 2.0,
                    gap=(MENZEL_GAP_START, MENZEL_GAP_END)
                    ) -> tuple[ScatterResult, SeasonScatter | None]:
    """The per-star rising-scatter test.  Returns ``(result, season_table)``."""
    res = ScatterResult()
    if lc.n_det == 0 or not lc.good.any():
        return res, None
    marg = float(max(margin, (amp_cat if np.isfinite(amp_cat) else 0.0) + 0.5))
    res.margin_used = marg
    m = lc.margin_mask(marg)
    ss, resid = scatter_table(lc, m, f_ref=f_ref, season_years=season_years,
                              min_epochs_season=min_epochs_season, min_seasons=min_seasons)
    res.residualised = resid
    if ss is None:
        res.flags = "too_few_seasons"
        return res, None
    res.n_epochs, res.n_seasons = int(ss.n.sum()), len(ss)
    fit, st_pre, st_post = scatter_from_table(ss, gap=gap)
    yr = mjd_to_year(ss.t)
    res.baseline_yr = float(np.ptp(yr))
    res.n_seasons_pre, res.n_seasons_post = fit.n_pre, fit.n_post
    if fit.ok:
        res.slope_var_per_century = float(fit.slope) * 100.0
        res.slope_sigma = float(fit.slope_sigma)
        res.slope_naive_var_per_century = float(fit.slope_naive) * 100.0
        res.slope_naive_sigma = float(fit.slope_naive_sigma)
        res.step_var, res.step_sigma = float(fit.step), float(fit.step_sigma)
        res.step_identifiable = bool(fit.step_identifiable)
        v0 = fit.level + fit.slope * (yr.min() - fit.t_ref)
        v1 = fit.level + fit.slope * (yr.max() - fit.t_ref) + (fit.step if fit.step_identifiable
                                                                 else 0.0)
        res.amp_first_mmag = 1e3 * float(np.sqrt(max(v0, 0.0)))
        res.amp_last_mmag = 1e3 * float(np.sqrt(max(v1, 0.0)))
    _fill_rust(res, "pre", st_pre)
    _fill_rust(res, "post", st_post)

    # Robustness: dominant series, pre-gap only.
    name, _ = lc.dominant_series(m)
    res.series_name = name
    if name:
        ss_s, _ = scatter_table(lc, m & (lc.series.astype(str) == name), f_ref=f_ref,
                                season_years=season_years, min_epochs_season=min_epochs_season,
                                min_seasons=4)
        if ss_s is not None:
            _, sp, _ = scatter_from_table(ss_s, gap=gap)
            if sp is not None:
                res.series_pre_rank_p = float(sp.rank_p)
                res.series_pre_slope_sigma = float(sp.slope_sigma)
                res.series_pre_slope_positive = bool(sp.slope_var_yr > 0)
    # Robustness: deeper plates only, pre-gap only.
    ss_d, _ = scatter_table(lc, lc.margin_mask(marg + deep_extra), f_ref=f_ref,
                            season_years=season_years, min_epochs_season=min_epochs_season,
                            min_seasons=4)
    if ss_d is not None:
        _, dp, _ = scatter_from_table(ss_d, gap=gap)
        if dp is not None:
            res.deep_pre_rank_p = float(dp.rank_p)
            res.deep_pre_slope_sigma = float(dp.slope_sigma)
            res.deep_pre_slope_positive = bool(dp.slope_var_yr > 0)

    flags = []
    joint_ok = (fit.ok and fit.slope > 0 and np.isfinite(res.slope_sigma)
                and res.slope_sigma >= slope_sigma_min)
    if not joint_ok:
        flags.append("joint_slope_not_rising")
    pre_ok = (st_pre is not None and st_pre.slope_var_yr > 0
              and res.pre_rank_p <= pre_rank_p_max and res.pre_slope_sigma >= pre_slope_sigma_min
              and res.pre_loo_min >= pre_loo_min)
    if not pre_ok:
        flags.append("pre_gap_segment_not_rising")
    ser_testable = np.isfinite(res.series_pre_slope_sigma)
    ser_ok = (not ser_testable) or (res.series_pre_slope_positive
                                    and res.series_pre_slope_sigma >= robust_sigma_min)
    if not ser_ok:
        flags.append("depends_on_series")
    deep_testable = np.isfinite(res.deep_pre_slope_sigma)
    deep_ok = (not deep_testable) or (res.deep_pre_slope_positive
                                      and res.deep_pre_slope_sigma >= robust_sigma_min)
    if not deep_ok:
        flags.append("depends_on_shallow_plates")
    if fit.ok and not fit.step_identifiable:
        flags.append("step_unidentifiable_single_segment")
    if (fit.ok and fit.slope_naive > 0 and np.isfinite(res.slope_naive_sigma)
            and res.slope_naive_sigma >= slope_sigma_min and not joint_ok):
        flags.append("naive_trend_was_the_gap")
    res.is_rust = bool(joint_ok and pre_ok and ser_ok and deep_ok)
    res.status = ("rising_scatter" if res.is_rust else
                  "no_rise" if not joint_ok else "rejected")
    res.flags = ";".join(flags)
    return res, ss


__all__ = ["ScatterResult", "analyze_scatter", "residualise_periodic", "scatter_from_table",
           "scatter_table", "season_scatter_from_dict", "season_scatter_to_dict",
           "subset_scatter"]

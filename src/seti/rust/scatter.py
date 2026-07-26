"""The bias-corrected season-scatter statistic --- RUST's second moment.

The channel asks whether a star's *aperiodic variability amplitude* grows over
the survey decade.  The naive implementation of that question is a trap, and the
trap is worth stating plainly because it is the whole reason this module exists:

    Robust scale estimators are biased **low** at small N.  For Gaussian data
    ``1.4826 x MAD`` recovers only ~74% of sigma at N = 5 and ~93% at N = 20.
    The number of epochs per season is set by the survey's cadence, and survey
    cadence *trends with calendar time* (ZTF's public survey went from a 3-day
    to a 2-day cadence in 2020; ZTF-I/II/III have different field rosters and
    different seasonal depths).  A perfectly constant star observed with a
    rising cadence therefore shows a **rising measured scatter**.  A search that
    skips this correction does not measure astrophysics.  It measures ZTF's
    operations calendar.

The fix implemented here has three layers, and none of them is optional.

1. **Exact null expectation per season.**  For every season we compute what the
   estimator *would* read if the star were perfectly constant, using that
   season's own ``N`` and that season's own per-epoch error vector.  Because the
   null carries the same N as the data, the N-dependence cancels in the ratio
   by construction rather than by hope.  Two ingredients:

   * ``mad_null_table`` --- a Monte-Carlo table of the finite-N bias ``b(N)`` and
     the relative sampling scatter ``u(N)`` of ``1.4826 x MAD`` on Gaussian data.
   * ``mixture_mad_sigma`` --- the asymptotic MAD scale of a *heteroscedastic*
     zero-mean Gaussian mixture, so a season whose epochs have unequal errors
     gets the right noise floor rather than a naive quadrature mean.

2. **Excess variance, subtracted not divided.**  ``v_exc = sigma_obs^2 -
   sigma_null^2`` in mag^2.  The excess is allowed to go **negative**; clipping
   it at zero would rectify noise into a spurious positive trend for faint stars.

3. **Exact per-season Monte Carlo** (``season_scatter_mc``) for survivors, which
   drops the asymptotic approximation in layer 1 entirely and simulates the
   season's actual error vector.  Cheap for a shortlist, too slow for a sweep.

A fourth layer --- the per-CCD ensemble common mode --- lives in
:mod:`seti.rust.trend`, and a fifth --- equal-N subsampling, which throws data
away until every season has identical N --- is ``equalize_n`` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf

# 1/Phi^-1(0.75): the asymptotic constant that makes MAD a consistent estimator
# of the Gaussian standard deviation.
MAD_TO_SIGMA = 1.482602218505602

# Monte-Carlo table parameters.  Deterministic seed so the correction is
# reproducible and the offline tests are stable.
_TABLE_N_MAX = 200
_TABLE_TRIALS = 4000
_TABLE_SEED = 20260726
# Above this N the estimator's even/odd alternation has died away and b(N) is
# smooth, so the tabulated values there are replaced by their own analytic fit.
_SMOOTH_FROM = 30

# Asymptotic relative scatter of the MAD scale estimator on Gaussian data:
# SD(sigma_MAD)/sigma -> 1.1664/sqrt(N)  (ARE 36.7% relative to the sample SD).
_U_ASYMPTOTIC = 1.1664

_TABLES: dict[bool, tuple[np.ndarray, np.ndarray]] = {}


def mad_scale(x: np.ndarray) -> float:
    """``1.4826 x MAD`` --- the raw (uncorrected) robust scale of ``x``."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def detrend_line(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals of ``y`` about an ordinary least-squares line in ``t``.

    Removing a *line* per season rather than merely the season mean is what
    makes this channel's statistic genuinely **aperiodic**.  Without it, any
    smooth within-season drift inflates the season's scatter, and the most
    important such drift is the one the sibling `dimming` channel selects on:
    a star with an *accelerating* secular fade drifts more within each season
    than the last, which is a rising second moment produced entirely by a
    first-moment phenomenon.  A megaswarm cascade is short-timescale and
    irregular; a fade is smooth.  Subtracting the line separates them.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 3 or np.ptp(t) <= 0:
        return y - np.median(y)
    x = (t - t.mean()) / max(np.ptp(t), 1e-9)
    sxx = float((x * x).sum())
    if sxx <= 0:
        return y - np.median(y)
    slope = float((x * (y - y.mean())).sum()) / sxx
    return y - (y.mean() + slope * x)


def mad_null_table(n_max: int = _TABLE_N_MAX, n_trials: int = _TABLE_TRIALS,
                   seed: int = _TABLE_SEED,
                   detrended: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo finite-N behaviour of ``1.4826 x MAD`` on Gaussian data.

    Returns ``(b, u)``, arrays indexed by N (0..n_max, entries below N=2 are
    NaN), where

    * ``b[N] = E[1.4826 MAD_N] / sigma`` --- the multiplicative low bias.  It is
      strongly non-monotonic: MAD alternates between even and odd N because the
      sample median does, so no smooth analytic correction is adequate.
    * ``u[N] = SD[1.4826 MAD_N] / E[1.4826 MAD_N]`` --- the relative sampling
      scatter, which becomes the season's measurement error.

    ``detrended=True`` returns the table for residuals about a fitted **line**
    rather than about the mean --- a different and *stronger* suppression,
    because a line fit removes two degrees of freedom and, at irregular sampling,
    an N-dependent amount of the noise.  Using the un-detrended table on
    detrended data would leave exactly the kind of N-dependent (hence
    calendar-time-dependent) offset this module exists to remove.  Simulated at
    uniformly random epoch positions, to match ZTF's irregular cadence.

    Computed once per variant and cached at module level.
    """
    cached = _TABLES.get(bool(detrended))
    if cached is not None and cached[0].size >= n_max + 1:
        return cached
    rng = np.random.default_rng(seed + (1 if detrended else 0))
    b = np.full(n_max + 1, np.nan)
    u = np.full(n_max + 1, np.nan)
    for n in range(2, n_max + 1):
        draws = rng.standard_normal((n_trials, n))
        if detrended and n >= 3:
            x = np.sort(rng.random((n_trials, n)), axis=1)
            x = x - x.mean(axis=1, keepdims=True)
            sxx = (x * x).sum(axis=1, keepdims=True)
            slope = (x * (draws - draws.mean(axis=1, keepdims=True))
                     ).sum(axis=1, keepdims=True) / np.maximum(sxx, 1e-12)
            draws = draws - (draws.mean(axis=1, keepdims=True) + slope * x)
        med = np.median(draws, axis=1, keepdims=True)
        s = MAD_TO_SIGMA * np.median(np.abs(draws - med), axis=1)
        mean_s = float(np.mean(s))
        b[n] = mean_s
        u[n] = float(np.std(s)) / mean_s if mean_s > 0 else np.nan

    # Above ``_SMOOTH_FROM`` the even/odd alternation has died out and b(N) is a
    # smooth 1 + a/N + c/N^2, so the only thing left in the tabulated values is
    # Monte-Carlo jitter --- and jitter indexed by N is jitter indexed by
    # *cadence*, which in this channel means jitter indexed by calendar time.
    # Replacing the tail with its own least-squares fit removes it.
    ns = np.arange(_SMOOTH_FROM, n_max + 1, dtype=float)
    design = np.column_stack([np.ones_like(ns), 1.0 / ns, 1.0 / ns ** 2])
    coef, *_ = np.linalg.lstsq(design, b[_SMOOTH_FROM:], rcond=None)
    b[_SMOOTH_FROM:] = design @ coef
    # u(N) -> 1.1664/sqrt(N); fit the leading correction the same way.
    du = np.column_stack([1.0 / np.sqrt(ns), 1.0 / ns ** 1.5])
    cu, *_ = np.linalg.lstsq(du, u[_SMOOTH_FROM:], rcond=None)
    u[_SMOOTH_FROM:] = du @ cu

    _TABLES[bool(detrended)] = (b, u)
    return b, u


def bias_factor(n: int | np.ndarray, detrended: bool = False) -> np.ndarray:
    """``E[1.4826 MAD_N]/sigma`` for Gaussian data, from the cached MC table."""
    b, _ = mad_null_table(detrended=detrended)
    n_arr = np.atleast_1d(np.asarray(n, dtype=int))
    out = np.ones(n_arr.shape, dtype=float)
    inside = (n_arr >= 2) & (n_arr < b.size)
    out[inside] = b[n_arr[inside]]
    out[n_arr < 2] = np.nan
    return out


def rel_scatter(n: int | np.ndarray, detrended: bool = False) -> np.ndarray:
    """``SD/mean`` of ``1.4826 MAD_N``; asymptotic ``1.1664/sqrt(N)`` past the table."""
    _, u = mad_null_table(detrended=detrended)
    n_arr = np.atleast_1d(np.asarray(n, dtype=int))
    out = np.where(n_arr >= 2, _U_ASYMPTOTIC / np.sqrt(np.maximum(n_arr, 1)), np.nan)
    inside = (n_arr >= 2) & (n_arr < u.size)
    out[inside] = u[n_arr[inside]]
    return out


def mixture_mad_sigma(errs: np.ndarray) -> float:
    """Asymptotic ``1.4826 x MAD`` of a zero-mean heteroscedastic Gaussian mixture.

    A season's epochs generally do *not* share one photometric error: seeing,
    airmass and moon phase vary within a season.  Under the null, the residuals
    are draws from ``N(0, e_j^2)`` with the ``e_j`` all different, so the
    population MAD is the ``c`` solving

        (1/N) sum_j erf( c / (e_j sqrt(2)) ) = 1/2

    which is **not** ``0.6745 x rms(e)``.  Getting this wrong biases the noise
    floor in a magnitude-dependent way, and magnitude correlates with everything.
    Solved by bisection; returns ``1.4826 c``.
    """
    e = np.asarray(errs, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    if e.size == 0:
        return float("nan")
    lo, hi = 0.0, float(5.0 * np.max(e))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        frac = float(np.mean(erf(mid / (e * np.sqrt(2.0)))))
        if frac < 0.5:
            lo = mid
        else:
            hi = mid
    return float(MAD_TO_SIGMA * 0.5 * (lo + hi))


@dataclass
class SeasonScatter:
    """Per-season second-moment measurements for one light curve, one band."""

    label: np.ndarray = field(default_factory=lambda: np.array([], int))
    t: np.ndarray = field(default_factory=lambda: np.array([], float))
    n: np.ndarray = field(default_factory=lambda: np.array([], int))
    mag_med: np.ndarray = field(default_factory=lambda: np.array([], float))
    sigma_obs: np.ndarray = field(default_factory=lambda: np.array([], float))
    sigma_null: np.ndarray = field(default_factory=lambda: np.array([], float))
    v_exc: np.ndarray = field(default_factory=lambda: np.array([], float))
    v_err: np.ndarray = field(default_factory=lambda: np.array([], float))

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def amp(self) -> np.ndarray:
        """Signed excess amplitude in mag: ``sign(v) sqrt(|v|)``."""
        return np.sign(self.v_exc) * np.sqrt(np.abs(self.v_exc))

    def as_dict(self) -> dict:
        return {
            "n_seasons": len(self),
            "n_per_season": [int(v) for v in self.n],
            "season_t": [float(v) for v in self.t],
            "season_mag_med": [float(v) for v in self.mag_med],
            "sigma_obs_mmag": [float(1e3 * v) for v in self.sigma_obs],
            "sigma_null_mmag": [float(1e3 * v) for v in self.sigma_null],
            "v_exc": [float(v) for v in self.v_exc],
            "v_err": [float(v) for v in self.v_err],
        }


def season_labels(t: np.ndarray, season_days: float = 365.25,
                  origin: float | None = None) -> np.ndarray:
    """Integer season index for each epoch.

    Default bin is a **full year**, not the half-year used by
    ``seti.dimming.secular``: a scatter estimate needs many more epochs than a
    median does, and ZTF's seasonal visibility gap already separates years
    cleanly for most fields.
    """
    t = np.asarray(t, dtype=float)
    t0 = float(np.min(t)) if origin is None else float(origin)
    return np.floor((t - t0) / float(season_days)).astype(int)


def _default_err(mag: np.ndarray, magerr: np.ndarray | None) -> np.ndarray:
    """Per-epoch errors, with a defensible fallback when the archive omits them."""
    if magerr is None:
        return np.full(mag.shape, np.nan)
    e = np.asarray(magerr, dtype=float)
    if e.shape != mag.shape:
        return np.full(mag.shape, np.nan)
    return e


def season_scatter(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray | None = None,
    *,
    season_days: float = 365.25,
    min_epochs_season: int = 8,
    min_seasons: int = 4,
    clip_sigma: float = 6.0,
    detrend_season: bool = True,
    equalize_n: bool = False,
    equalize_draws: int = 32,
    rng: np.random.Generator | None = None,
) -> SeasonScatter | None:
    """Bias-corrected excess variance per observing season.

    Parameters
    ----------
    detrend_season
        Remove a fitted **line** from each season rather than only its median
        (default, and the scientifically correct choice).  This is what makes
        the statistic *aperiodic*: a star with an accelerating secular fade
        drifts further within each successive season, which would otherwise read
        as a rising second moment produced entirely by a first-moment
        phenomenon --- i.e. the sibling ``dimming`` channel's population leaking
        straight into this one.  The null table is switched to its
        line-detrended variant to match, so the correction stays exact.
    equalize_n
        If True, every season is randomly subsampled to the smallest season's
        epoch count (averaged over ``equalize_draws`` draws) *before* the
        statistic is formed.  This throws data away, but it makes the estimator
        bias literally identical in every season, so any surviving trend cannot
        be a cadence artefact.  Used as the decisive cross-check on survivors,
        not as the sweep default.

    Returns ``None`` if the light curve cannot support the statistic (too few
    epochs, too few adequately-sampled seasons) --- an honest refusal, never a
    fabricated measurement.
    """
    t = np.asarray(time, dtype=float)
    m = np.asarray(mag, dtype=float)
    e = _default_err(m, magerr)
    good = np.isfinite(t) & np.isfinite(m)
    t, m, e = t[good], m[good], e[good]
    if t.size < min_epochs_season * min_seasons:
        return None

    # A single catastrophic outlier (cosmic ray, bad subtraction) can dominate a
    # season.  Clip against a *global* robust scale, generously, so real dips are
    # retained: this channel wants variability, it just does not want garbage.
    gmed = float(np.median(m))
    gscale = mad_scale(m)
    if np.isfinite(gscale) and gscale > 0:
        keep = np.abs(m - gmed) <= clip_sigma * gscale
        # Never clip away more than 5% of the curve; if the cut is that
        # aggressive the star is genuinely variable and clipping would eat it.
        if keep.mean() >= 0.95:
            t, m, e = t[keep], m[keep], e[keep]

    labels = season_labels(t, season_days)
    rng = rng if rng is not None else np.random.default_rng(_TABLE_SEED)

    rows: list[tuple[int, float, int, float, float, float, float, float]] = []
    per_season: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for lab in np.unique(labels):
        sel = labels == lab
        if int(sel.sum()) < min_epochs_season:
            continue
        per_season.append((int(lab), t[sel], m[sel], e[sel]))
    if len(per_season) < min_seasons:
        return None

    n_target = min(len(tt) for _lab, tt, _mm, _ee in per_season) if equalize_n else None

    for lab, tt, mm, ee in per_season:
        if equalize_n and n_target is not None and len(tt) > n_target:
            # Average the *squared* statistic over independent subsamples of fixed
            # size (squares, because the variance is what gets differenced below).
            sig2, null2 = [], []
            for _ in range(equalize_draws):
                idx = np.sort(rng.choice(len(tt), size=n_target, replace=False))
                res = detrend_line(tt[idx], mm[idx]) if detrend_season else mm[idx]
                sig2.append(mad_scale(res) ** 2)
                null2.append(_null_sigma(ee[idx], n_target, detrend_season) ** 2)
            sigma_obs = float(np.sqrt(np.mean(sig2)))
            sigma_null = float(np.sqrt(np.mean(null2)))
            n_eff = int(n_target)
        else:
            res = detrend_line(tt, mm) if detrend_season else mm
            sigma_obs = mad_scale(res)
            n_eff = int(len(mm))
            sigma_null = _null_sigma(ee, n_eff, detrend_season)

        # Second-order bias.  The estimator's *square* is not the square of its
        # expectation: E[s^2] = (E[s])^2 + Var(s) = (b sigma)^2 (1 + u^2).  Since
        # u(N) falls as N grows and N tracks the survey cadence, skipping this
        # term leaves a residual N-dependent -- hence calendar-time-dependent --
        # offset in the excess variance.  It is small, it is systematic, and it
        # is exactly the class of error this channel exists to avoid.
        u = float(rel_scatter(n_eff, detrend_season)[0])
        v_null = (sigma_null ** 2 * (1.0 + u ** 2)) if np.isfinite(sigma_null) else 0.0
        v_exc = sigma_obs ** 2 - v_null
        # Delta-method error on the variance: d(sigma^2) = 2 sigma d(sigma), with
        # d(sigma) = u(N) sigma from the sampling table.  Floored so a season that
        # happens to read exactly zero scatter cannot acquire infinite weight.
        scale = max(sigma_obs, sigma_null if np.isfinite(sigma_null) else 0.0)
        v_err = float(2.0 * u * max(scale, 1e-4) ** 2)
        rows.append((int(lab), float(np.median(tt)), n_eff, float(np.median(mm)),
                     float(sigma_obs), float(sigma_null), float(v_exc), v_err))

    if len(rows) < min_seasons:
        return None
    arr = list(zip(*rows, strict=True))
    return SeasonScatter(
        label=np.asarray(arr[0], dtype=int), t=np.asarray(arr[1], dtype=float),
        n=np.asarray(arr[2], dtype=int), mag_med=np.asarray(arr[3], dtype=float),
        sigma_obs=np.asarray(arr[4], dtype=float),
        sigma_null=np.asarray(arr[5], dtype=float),
        v_exc=np.asarray(arr[6], dtype=float), v_err=np.asarray(arr[7], dtype=float),
    )


def _null_sigma(errs: np.ndarray, n: int, detrended: bool = False) -> float:
    """Expected ``1.4826 x MAD`` of ``n`` pure-noise epochs with errors ``errs``.

    ``mixture_mad_sigma`` supplies the asymptotic (large-N) value for the actual
    heteroscedastic error vector; ``bias_factor(n, detrended)`` supplies the
    finite-N correction *for the same estimator that was applied to the data*.
    Their product is the season's null expectation **at that season's own epoch
    count** --- which is the whole point.
    """
    e = np.asarray(errs, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    if e.size == 0:
        return float("nan")
    return float(mixture_mad_sigma(e) * bias_factor(n, detrended)[0])


def season_scatter_mc(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    *,
    season_days: float = 365.25,
    min_epochs_season: int = 8,
    min_seasons: int = 4,
    n_trials: int = 400,
    seed: int = _TABLE_SEED,
    detrend_season: bool = True,
) -> SeasonScatter | None:
    """Exact per-season null by direct Monte Carlo --- the survivor-grade version.

    Drops every asymptotic approximation in :func:`season_scatter`: for each
    season it simulates ``n_trials`` pure-noise realisations using that season's
    **actual** error vector and **actual** epoch count, and takes the mean and
    scatter of the estimator over those realisations.  There is no table, no
    mixture solve, and no finite-N formula --- so if the fast path and this path
    disagree for a candidate, the fast path is wrong and this one is right.

    Too slow for millions of stars; exactly right for a shortlist.
    """
    t = np.asarray(time, dtype=float)
    m = np.asarray(mag, dtype=float)
    e = np.asarray(magerr, dtype=float)
    good = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[good], m[good], e[good]
    if t.size < min_epochs_season * min_seasons:
        return None
    labels = season_labels(t, season_days)
    rng = np.random.default_rng(seed)
    rows = []
    for lab in np.unique(labels):
        sel = labels == lab
        n = int(sel.sum())
        if n < min_epochs_season:
            continue
        tt, mm, ee = t[sel], m[sel], e[sel]
        sim = rng.standard_normal((n_trials, n)) * ee[None, :]
        if detrend_season and n >= 3:
            # Apply to the simulated noise the *identical* estimator applied to
            # the data, including the line removal, at this season's own epoch
            # positions.  Nothing about the null is approximated here.
            x = tt - tt.mean()
            sxx = float((x * x).sum())
            if sxx > 0:
                slope = ((sim - sim.mean(axis=1, keepdims=True)) * x[None, :]
                         ).sum(axis=1, keepdims=True) / sxx
                sim = sim - (sim.mean(axis=1, keepdims=True) + slope * x[None, :])
        sim -= np.median(sim, axis=1, keepdims=True)
        s_null = MAD_TO_SIGMA * np.median(np.abs(sim), axis=1)
        mu_null = float(np.mean(s_null))
        sd_null = float(np.std(s_null))
        sigma_obs = mad_scale(detrend_line(tt, mm) if detrend_season else mm)
        # E[s^2] straight from the simulation -- no delta-method approximation,
        # so the second-order N-dependent bias is removed exactly here.
        v_exc = sigma_obs ** 2 - float(np.mean(s_null ** 2))
        u = (sd_null / mu_null) if mu_null > 0 else float(rel_scatter(n)[0])
        v_err = float(2.0 * u * max(sigma_obs, mu_null, 1e-4) ** 2)
        rows.append((int(lab), float(np.median(tt)), n, float(np.median(mm)),
                     float(sigma_obs), mu_null, float(v_exc), v_err))
    if len(rows) < min_seasons:
        return None
    arr = list(zip(*rows, strict=True))
    return SeasonScatter(
        label=np.asarray(arr[0], dtype=int), t=np.asarray(arr[1], dtype=float),
        n=np.asarray(arr[2], dtype=int), mag_med=np.asarray(arr[3], dtype=float),
        sigma_obs=np.asarray(arr[4], dtype=float),
        sigma_null=np.asarray(arr[5], dtype=float),
        v_exc=np.asarray(arr[6], dtype=float), v_err=np.asarray(arr[7], dtype=float),
    )


__all__ = [
    "MAD_TO_SIGMA", "SeasonScatter", "bias_factor", "mad_null_table", "mad_scale",
    "mixture_mad_sigma", "rel_scatter", "season_labels", "season_scatter",
    "season_scatter_mc",
]

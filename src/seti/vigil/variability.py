"""Mid-infrared variability statistics for VIGIL.

Pure functions --- no network --- so every estimator here is unit-tested offline.

The input is NEOWISE single-exposure photometry (``neowiser_p1bs_psd``): W1/W2
magnitudes with per-exposure errors, sampled in ~1-day **visits** roughly twice
per year since 2014.  Two things about that cadence drive the design:

1. **The visit structure is a free noise calibrator.**  Each visit contains
   order 10-20 exposures inside ~1 day, during which no plausible circumstellar
   source varies.  The scatter *within* a visit therefore measures the true
   per-exposure noise, empirically, per star --- while the scatter *between*
   visit means measures variability on the ~6-month timescale we care about.
   NEOWISE quoted ``w?sigmpro`` values are known to be optimistic for bright
   sources, and an optimistic error is *exactly* how a variability search
   manufactures candidates.  So the per-epoch error used here is the quoted one
   rescaled by a per-star, per-band factor fitted from the within-visit scatter
   (:func:`fit_error_scale`), and the fitted factor is reported, not hidden.

2. **Cadence bias.**  The number of exposures per visit and the number of visits
   both vary strongly with ecliptic latitude (the NEOWISE scan pattern piles up
   at the poles).  Any statistic built from a *ratio* of scatter to noise is
   biased at small N, and if N trends with position then an uncorrected search
   maps the NEOWISE scan pattern rather than astrophysics.  Three corrections:
   the excess-variance form (unbiased in expectation) is the primary quantity,
   its uncertainty carries the exact N dependence (Vaughan et al. 2003), and
   :func:`equalize_visits` truncates every visit to a common exposure count so a
   candidate can be re-measured with the finite-N bias made identical everywhere.

A third systematic --- the per-visit zero-point wander common to all stars in a
field (moon, scan angle, thermal state) --- is removed by
:func:`ensemble_common_mode`, the same ensemble logic RUST uses in its second
moment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Defaults.  Mirrored in config/vigil.yaml; the module constants are the
# fallback so the estimators work with no config file present.
# --------------------------------------------------------------------------
VISIT_GAP_DAYS = 30.0          # NEOWISE revisits a field ~every 6 months
MIN_EXP_PER_VISIT = 3          # below this the within-visit noise calibration dies
MIN_VISITS = 8                 # ~4 yr of NEOWISE at 2 visits/yr
ERR_SCALE_MIN = 0.5            # never let the fitted rescale *shrink* errors much
ERR_SCALE_MAX = 5.0


@dataclass
class Visit:
    """One NEOWISE visit, collapsed to a mean and an *empirical* error."""

    t_mjd: float
    mag: float
    err_quoted: float           # error on the mean from the quoted per-exposure sigmas
    err_within: float           # error on the mean from the within-visit scatter
    n_exp: int
    scatter_within: float       # sample sd of the exposures in the visit


@dataclass
class MidIRVar:
    """Mid-IR variability of one star in one band."""

    band: str
    n_epochs: int
    n_visits: int
    span_yr: float
    mean_mag: float
    err_scale: float            # fitted per-exposure error rescale factor
    sigma_typ_mag: float        # typical error on a visit mean, after rescaling
    nxs: float                  # normalised excess variance (unbiased estimator)
    nxs_err: float
    f_var: float                # fractional rms variability amplitude
    f_var_err: float
    f_var_sigma: float          # significance of f_var against pure noise
    chi2_red: float
    chi2_p: float               # p-value of the constant-flux hypothesis
    amp_ptp: float              # peak-to-peak fractional flux amplitude (robust)
    amp_ptp_err: float
    median_visit_gap_d: float
    common_mode_removed: bool = False
    n_visits_flagged: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Visit construction and the empirical noise model
# --------------------------------------------------------------------------
def bin_visits(mjd, mag, magerr, gap_days: float = VISIT_GAP_DAYS,
               min_exp: int = MIN_EXP_PER_VISIT,
               clip_sigma: float = 5.0) -> list[Visit]:
    """Collapse single-exposure photometry into visits.

    Exposures are grouped by time gaps larger than ``gap_days``.  Inside a visit
    the mean is inverse-variance weighted after a robust clip, and *two*
    independent errors on that mean are kept: the one the catalogue's quoted
    sigmas imply, and the one the within-visit scatter implies.  Their ratio is
    what :func:`fit_error_scale` turns into the per-star noise calibration.

    Visits with fewer than ``min_exp`` exposures are dropped: without several
    exposures the within-visit error is meaningless and the visit would enter the
    statistic with an uncalibrated error bar.
    """
    t = np.asarray(mjd, dtype=float)
    m = np.asarray(mag, dtype=float)
    e = np.asarray(magerr, dtype=float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if t.size == 0:
        return []
    order = np.argsort(t)
    t, m, e = t[order], m[order], e[order]

    edges = np.nonzero(np.diff(t) > gap_days)[0] + 1
    groups = np.split(np.arange(t.size), edges)

    visits: list[Visit] = []
    for g in groups:
        if g.size < min_exp:
            continue
        tv, mv, ev = t[g], m[g], e[g]
        # Robust clip against cosmic rays / a single bad frame, using the MAD so
        # one outlier cannot set its own rejection threshold.
        med = np.median(mv)
        mad = 1.4826 * np.median(np.abs(mv - med))
        if mad > 0:
            keep = np.abs(mv - med) <= clip_sigma * mad
            if keep.sum() >= min_exp:
                tv, mv, ev = tv[keep], mv[keep], ev[keep]
        n = int(mv.size)
        w = 1.0 / ev**2
        mean = float(np.sum(w * mv) / np.sum(w))
        err_q = float(np.sqrt(1.0 / np.sum(w)))
        sd = float(np.std(mv, ddof=1)) if n > 1 else float(np.mean(ev))
        err_w = sd / np.sqrt(n) if n > 1 else float(np.mean(ev))
        visits.append(Visit(t_mjd=float(np.mean(tv)), mag=mean, err_quoted=err_q,
                            err_within=float(err_w), n_exp=n, scatter_within=sd))
    return visits


def fit_error_scale(visits: list[Visit], lo: float = ERR_SCALE_MIN,
                    hi: float = ERR_SCALE_MAX) -> float:
    """Per-star factor by which the quoted per-exposure errors are wrong.

    Estimated as the median over visits of ``err_within / err_quoted``.  On a
    genuinely constant star with correct errors this is 1.  For NEOWISE bright
    sources it is routinely >1, and *using it is the difference between a
    variability search and a bright-star selection function*: intrinsic
    variability on ~6-month timescales does not inflate the scatter inside a
    single 1-day visit, so this calibration is not self-cancelling for the
    signals VIGIL is after.

    The one signal it *would* suppress is variability faster than a day.  That is
    stated as a bound, not fixed: NEOWISE cannot separate sub-day variability
    from noise without an external noise model, and VIGIL does not claim to.
    """
    r = [v.err_within / v.err_quoted for v in visits
         if v.n_exp > 2 and v.err_quoted > 0 and np.isfinite(v.err_within)]
    if not r:
        return 1.0
    return float(np.clip(np.median(r), lo, hi))


def equalize_visits(visits: list[Visit], rng=None) -> list[Visit]:
    """Truncate every visit to the smallest visit's exposure count.

    The finite-N bias of the within-visit error is then *identical* in every
    visit, so a variability trend cannot be manufactured by a trend in exposure
    count.  Approximated at the summary level (the exposures themselves are not
    retained): the visit's error on the mean is rescaled to the common N and the
    within-visit sd is left as the best available estimate.
    """
    if not visits:
        return []
    n_min = min(v.n_exp for v in visits)
    out = []
    for v in visits:
        s = np.sqrt(v.n_exp / n_min)
        out.append(Visit(t_mjd=v.t_mjd, mag=v.mag, err_quoted=v.err_quoted * s,
                         err_within=v.err_within * s, n_exp=n_min,
                         scatter_within=v.scatter_within))
    return out


# --------------------------------------------------------------------------
# The ensemble common mode
# --------------------------------------------------------------------------
def ensemble_common_mode(per_star: dict[str, list[Visit]],
                         gap_days: float = VISIT_GAP_DAYS,
                         min_stars: int = 8) -> tuple[dict[str, list[Visit]], dict]:
    """Remove the per-visit zero-point wander shared by every star in a field.

    Visits are matched across stars by epoch (same ``gap_days`` grouping applied
    to visit times), and the median residual from each star's own mean is
    subtracted epoch by epoch.  Returns the corrected visit lists and a diagnostic
    dict whose ``applied`` field must be propagated to the summary: if the field
    was too thin to measure a common mode, every statistic from it is
    **uncorrected**, and that has to be visible at the top level rather than
    assumed away.
    """
    diag = {"applied": False, "n_stars": len(per_star), "n_epochs": 0,
            "rms_common_mode_mag": float("nan")}
    if len(per_star) < min_stars:
        return per_star, diag

    all_t = np.sort(np.concatenate(
        [np.array([v.t_mjd for v in vs]) for vs in per_star.values() if vs]
        or [np.array([])]))
    if all_t.size == 0:
        return per_star, diag
    edges = np.nonzero(np.diff(all_t) > gap_days)[0] + 1
    centres = np.array([np.mean(c) for c in np.split(all_t, edges) if c.size])

    resid: dict[int, list[float]] = {i: [] for i in range(centres.size)}
    assign: dict[str, list[int]] = {}
    for sid, vs in per_star.items():
        if not vs:
            assign[sid] = []
            continue
        mags = np.array([v.mag for v in vs])
        ref = float(np.median(mags))
        idx = [int(np.argmin(np.abs(centres - v.t_mjd))) for v in vs]
        assign[sid] = idx
        for k, v in zip(idx, vs, strict=False):
            resid[k].append(v.mag - ref)

    offs = np.array([np.median(resid[k]) if len(resid[k]) >= min_stars else 0.0
                     for k in range(centres.size)])
    n_used = int(sum(1 for k in range(centres.size) if len(resid[k]) >= min_stars))
    if n_used == 0:
        return per_star, diag

    out = {}
    for sid, vs in per_star.items():
        idx = assign.get(sid, [])
        out[sid] = [Visit(t_mjd=v.t_mjd, mag=v.mag - offs[k], err_quoted=v.err_quoted,
                          err_within=v.err_within, n_exp=v.n_exp,
                          scatter_within=v.scatter_within)
                    for v, k in zip(vs, idx, strict=False)]
    diag.update({"applied": True, "n_epochs": n_used,
                 "rms_common_mode_mag": float(np.std(offs[offs != 0.0]))
                 if np.any(offs != 0.0) else 0.0})
    return out, diag


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------
def _mag_to_relflux(mag: np.ndarray, magerr: np.ndarray, ref: float):
    """Magnitudes -> flux relative to ``ref``, with propagated errors."""
    f = 10.0 ** (-0.4 * (mag - ref))
    ferr = 0.4 * np.log(10.0) * f * magerr
    return f, ferr


def midir_variability(visits: list[Visit], band: str = "W1",
                      min_visits: int = MIN_VISITS,
                      use_error_scale: bool = True,
                      err_scale: float | None = None,
                      common_mode_removed: bool = False) -> MidIRVar | None:
    """Fractional variability of a visit-binned light curve, errors propagated.

    Returns ``None`` --- never a zero --- when there are too few visits to
    measure anything.  "Not measurable" and "measured to be constant" are
    different statements and the funnel must not confuse them.

    The primary quantity is the **normalised excess variance**

    ``nxs = (S^2 - <sigma^2>) / <f>^2``

    which is unbiased in expectation at any N (the biased quantity is its square
    root, ``f_var``, which is reported for interpretability with the Vaughan et
    al. 2003 uncertainty).  ``f_var_sigma`` is the significance of the *excess
    variance*, not of ``f_var``, precisely because the former's null distribution
    is the well-behaved one.
    """
    if len(visits) < min_visits:
        return None
    t = np.array([v.t_mjd for v in visits], dtype=float)
    m = np.array([v.mag for v in visits], dtype=float)
    order = np.argsort(t)
    t, m, visits = t[order], m[order], [visits[i] for i in order]

    s = float(err_scale) if err_scale is not None else (
        fit_error_scale(visits) if use_error_scale else 1.0)
    e = np.array([v.err_quoted for v in visits], dtype=float) * s

    ref = float(np.median(m))
    f, ferr = _mag_to_relflux(m, e, ref)
    n = f.size
    fbar = float(np.mean(f))
    s2 = float(np.var(f, ddof=1))
    msig2 = float(np.mean(ferr**2))

    nxs = (s2 - msig2) / fbar**2
    # Vaughan et al. 2003, eq. (11): the uncertainty on the normalised excess
    # variance, carrying the exact 1/N dependence that the cadence imposes.
    nxs_err = np.sqrt(
        (np.sqrt(2.0 / n) * msig2 / fbar**2) ** 2
        + (np.sqrt(msig2 / n) * 2.0 * np.sqrt(max(nxs, 0.0)) / fbar) ** 2
    )
    f_var = float(np.sqrt(nxs)) if nxs > 0 else 0.0
    # Vaughan et al. 2003, eq. (B2).
    if f_var > 0:
        f_var_err = float(np.sqrt(
            (np.sqrt(1.0 / (2.0 * n)) * msig2 / (fbar**2 * f_var)) ** 2
            + (np.sqrt(msig2 / n) / fbar) ** 2))
    else:
        f_var_err = float(np.sqrt(msig2 / n) / fbar)
    f_var_sigma = float(nxs / nxs_err) if nxs_err > 0 else 0.0

    w = 1.0 / e**2
    mw = float(np.sum(w * m) / np.sum(w))
    chi2 = float(np.sum(w * (m - mw) ** 2))
    dof = max(n - 1, 1)
    from scipy import stats as _st
    chi2_p = float(_st.chi2.sf(chi2, dof))

    # Robust peak-to-peak: the 5th-95th percentile span, so one bad visit cannot
    # set the amplitude.  Its error is the quadrature sum of the two edge errors.
    lo, hi = np.percentile(f, [5.0, 95.0])
    amp_ptp = float((hi - lo) / fbar)
    amp_ptp_err = float(np.sqrt(2.0) * np.median(ferr) / fbar)

    return MidIRVar(
        band=band, n_epochs=int(sum(v.n_exp for v in visits)), n_visits=n,
        span_yr=float((t[-1] - t[0]) / 365.25), mean_mag=float(mw),
        err_scale=float(s), sigma_typ_mag=float(np.median(e)),
        nxs=float(nxs), nxs_err=float(nxs_err), f_var=f_var,
        f_var_err=float(f_var_err), f_var_sigma=f_var_sigma,
        chi2_red=float(chi2 / dof), chi2_p=chi2_p,
        amp_ptp=amp_ptp, amp_ptp_err=amp_ptp_err,
        median_visit_gap_d=float(np.median(np.diff(t))) if n > 1 else float("nan"),
        common_mode_removed=bool(common_mode_removed),
    )


def visit_flux_series(visits: list[Visit], band: str = "W1",
                      err_scale: float | None = None):
    """``(t_yr, rel_flux, rel_flux_err)`` for the shape statistics.

    Fluxes are relative to the light curve's median, which is the natural
    normalisation for the morphology tests --- they care about the *shape* of the
    modulation, not its zero point.
    """
    if not visits:
        return (np.array([]),) * 3
    t = np.array([v.t_mjd for v in visits], dtype=float)
    m = np.array([v.mag for v in visits], dtype=float)
    order = np.argsort(t)
    t, m = t[order], m[order]
    vs = [visits[i] for i in order]
    s = float(err_scale) if err_scale is not None else fit_error_scale(vs)
    e = np.array([v.err_quoted for v in vs], dtype=float) * s
    f, ferr = _mag_to_relflux(m, e, float(np.median(m)))
    return (t - t[0]) / 365.25, f, ferr


@dataclass
class VariabilityPair:
    """W1 and W2 variability of one star, plus the two-band consistency check."""

    w1: MidIRVar | None = None
    w2: MidIRVar | None = None
    band_ratio: float = float("nan")       # f_var(W2)/f_var(W1)
    band_ratio_err: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.w1 is not None and self.w2 is not None

    def as_dict(self) -> dict:
        d: dict = {"band_ratio": self.band_ratio, "band_ratio_err": self.band_ratio_err,
                   "notes": ";".join(self.notes)}
        for tag, s in (("w1", self.w1), ("w2", self.w2)):
            if s is None:
                d[f"{tag}_measured"] = False
                continue
            d[f"{tag}_measured"] = True
            d.update({f"{tag}_{k}": v for k, v in s.as_dict().items() if k != "band"})
        return d


def pair_variability(w1: MidIRVar | None, w2: MidIRVar | None) -> VariabilityPair:
    """Combine the two bands and record why a single-band result is not enough.

    The repository ledger is unambiguous that a single-band anomaly is an
    artefact until confirmed in a second band.  For VIGIL the second band is not
    only a confirmation: the *ratio* of the two amplitudes is the colour of the
    varying component and therefore the input to the temperature-stability test.
    """
    p = VariabilityPair(w1=w1, w2=w2)
    if w1 is None or w2 is None:
        p.notes.append("single_band_only")
        return p
    if w1.f_var > 0 and w2.f_var > 0:
        p.band_ratio = float(w2.f_var / w1.f_var)
        p.band_ratio_err = float(p.band_ratio * np.sqrt(
            (w2.f_var_err / w2.f_var) ** 2 + (w1.f_var_err / w1.f_var) ** 2))
    return p


__all__ = ["ERR_SCALE_MAX", "ERR_SCALE_MIN", "MIN_EXP_PER_VISIT", "MIN_VISITS",
           "VISIT_GAP_DAYS", "MidIRVar", "VariabilityPair", "Visit", "bin_visits",
           "ensemble_common_mode", "equalize_visits", "fit_error_scale",
           "midir_variability", "pair_variability", "visit_flux_series"]

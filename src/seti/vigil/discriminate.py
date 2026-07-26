"""The discriminator: separating a duty-cycled radiator from an extreme debris disk.

This module is where the channel either is a search or is an EDD catalogue.

The confounder
--------------
Extreme debris disks vary strongly in the mid-IR and are optically flat --- Moor
et al. 2021 (arXiv:2103.00568) monitored a sample of warm EDDs selected from
AllWISE and report that "all monitored stars were stable with flat light curves"
in the optical.  Optical constancy therefore buys **nothing** against the
dominant confounder; it only kills YSOs, dippers and AGN.  A search that stops at
"mid-IR variable, optically constant" has produced an EDD catalogue.

Three discriminants, in order of how much work they do
------------------------------------------------------
1. **Modulation index** ``m = A_obs / A_max(tau)``.  An EDD is "extreme" precisely
   because ``tau ~ 1e-2``; its variability is a perturbation on that large steady
   excess, so only a modest fraction of the excess modulates.  A load-following
   radiator has no steady floor to speak of --- the excess *is* the modulation ---
   so ``m -> 1``.  This is the brief's "high variability at LOW fractional excess"
   made dimensionless, and it is strictly better than a raw ``tau`` cut because it
   does not depend on the absolute excess scale, on distance, or on the star's
   luminosity.  Both are computed; ``m`` is primary and ``tau`` is retained as the
   brief's stated cut so the two can be compared on the runner.
2. **Duty-cycle morphology.**  A collisional cascade is a smooth secular decay
   (the dust column drains) punctuated by stochastic brightening events.  A
   compute load has no reason to do that: it can be square, repeating, or
   two-state.  So the morphology statistics score *against* a monotone decay and
   *for* two-state structure with fast transitions.
3. **Colour-temperature stability during variation.**  A cascade changes the
   amount *and* the temperature of the dust (fresh small grains are hot, then
   spread and cool).  A radiator whose output tracks load changes the amount at
   roughly fixed temperature.  The test is on the *varying component's* W2/W1
   colour, not on the total colour --- the photosphere would dilute the latter.

Every statistic here is a pure function of arrays.  Nothing in this module
touches the network, and each one is exercised by ``tests/test_vigil.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..photometry import band_freq_hz, planck_bnu
from .excess import ExcessMeasurement, max_amplitude_for_frac_excess

# --------------------------------------------------------------------------
# Thresholds (mirrored in config/vigil.yaml)
# --------------------------------------------------------------------------
FVAR_SIGMA_MIN = 5.0            # mid-IR variability significance
FVAR_MIN = 0.02                 # 2% fractional rms: below this NEOWISE systematics rule
OPTICAL_RATIO_MIN = 4.0         # f_var(mid-IR) / f_var(optical)
OPTICAL_FVAR_MAX = 0.02         # optical constancy in absolute terms
TAU_MAX = 5.0e-3               # the brief's low-fractional-excess cut
                               # (EDDs sit at ~1e-2; see docs/vigil.md)
MODULATION_MIN = 0.5            # at least half the inferred excess must be switching
MODULATION_MAX = 1.6            # above this the variability is not the excess
DECAY_R2_MAX = 0.5              # a decay-dominated light curve is a cascade
TWO_STATE_DBIC_MIN = 6.0        # positive evidence for two states
COLOUR_DRIFT_SIGMA_MAX = 3.0    # temperature of the varying component must be stable


# --------------------------------------------------------------------------
# Morphology
# --------------------------------------------------------------------------
@dataclass
class ShapeStats:
    """Duty-cycle morphology of a visit-binned flux series."""

    n: int
    trend_slope_per_yr: float
    trend_r2: float             # fraction of variance explained by a straight line
    kendall_tau: float
    kendall_p: float
    two_state_dbic: float       # BIC(1 state) - BIC(2 states); >0 favours two states
    state_sep_sigma: float
    duty_fraction: float        # fraction of visits in the high state
    square_fraction: float      # fraction of visits within 2 sigma of a state mean
    transition_rate: float      # state changes per visit interval
    ls_power: float
    ls_period_yr: float
    burst_skew: float           # positive = rare brightenings on a quiet floor
    morphology: str

    def as_dict(self) -> dict:
        return asdict(self)


def _weighted_line(x, y, e):
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    sw = np.sum(w)
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    sxx = np.sum(w * (x - mx) ** 2)
    if sxx <= 0:
        return 0.0, my, 0.0
    slope = np.sum(w * (x - mx) * (y - my)) / sxx
    inter = my - slope * mx
    pred = inter + slope * x
    ss_tot = np.sum(w * (y - my) ** 2)
    ss_res = np.sum(w * (y - pred) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(inter), float(max(r2, 0.0))


def _two_state_fit(y, e):
    """Best two-state (high/low) split by likelihood, with per-point errors.

    Returns ``(dbic, sep_sigma, labels, mu_lo, mu_hi)``.  ``dbic`` is
    ``BIC(1 state) - BIC(2 states)``, so positive favours two states.  The split
    is over the sorted values, which is exact for a one-dimensional two-component
    problem with a hard assignment.
    """
    n = y.size
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    mu1 = np.sum(w * y) / np.sum(w)
    chi1 = float(np.sum(w * (y - mu1) ** 2))
    bic1 = chi1 + 1.0 * np.log(n)

    order = np.argsort(y)
    ys, ws = y[order], w[order]
    best = None
    for k in range(1, n):
        lo, hi = slice(0, k), slice(k, n)
        m_lo = np.sum(ws[lo] * ys[lo]) / np.sum(ws[lo])
        m_hi = np.sum(ws[hi] * ys[hi]) / np.sum(ws[hi])
        chi2 = float(np.sum(ws[lo] * (ys[lo] - m_lo) ** 2)
                     + np.sum(ws[hi] * (ys[hi] - m_hi) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, k, m_lo, m_hi)
    chi2, k, m_lo, m_hi = best
    bic2 = chi2 + 3.0 * np.log(n)      # two means + one split point
    thr = 0.5 * (m_lo + m_hi)
    labels = (y > thr).astype(int)
    e_typ = float(np.median(e))
    sep = float(abs(m_hi - m_lo) / e_typ) if e_typ > 0 else 0.0
    return float(bic1 - bic2), sep, labels, float(m_lo), float(m_hi)


def shape_stats(t_yr, flux, flux_err) -> ShapeStats | None:
    """Morphology of the modulation --- decay versus duty cycle."""
    t = np.asarray(t_yr, dtype=float)
    y = np.asarray(flux, dtype=float)
    e = np.asarray(flux_err, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    t, y, e = t[ok], y[ok], e[ok]
    n = t.size
    if n < 6:
        return None

    slope, _inter, r2 = _weighted_line(t, y, e)
    from scipy import stats as _st
    kt = _st.kendalltau(t, y)
    dbic, sep, labels, m_lo, m_hi = _two_state_fit(y, e)

    hi = labels == 1
    duty = float(hi.mean())
    mu = np.where(hi, m_hi, m_lo)
    square = float(np.mean(np.abs(y - mu) <= 2.0 * e))
    trans = float(np.mean(np.abs(np.diff(labels)))) if n > 1 else 0.0

    try:
        from astropy.timeseries import LombScargle
        span = float(t[-1] - t[0])
        if span > 0:
            freq = np.linspace(1.0 / span, n / (2.0 * span), 400)
            power = LombScargle(t, y, e).power(freq)
            j = int(np.argmax(power))
            ls_power, ls_period = float(power[j]), float(1.0 / freq[j])
        else:
            ls_power, ls_period = 0.0, float("nan")
    except Exception:                                  # noqa: BLE001
        ls_power, ls_period = float("nan"), float("nan")

    med = float(np.median(y))
    mad = 1.4826 * float(np.median(np.abs(y - med)))
    skew = float(np.mean(((y - med) / mad) ** 3)) if mad > 0 else 0.0

    if r2 >= DECAY_R2_MAX and slope < 0:
        morph = "secular_decay"
    elif r2 >= DECAY_R2_MAX and slope > 0:
        morph = "secular_rise"
    elif dbic >= TWO_STATE_DBIC_MIN and square >= 0.7:
        morph = "two_state"
    elif skew > 1.5:
        morph = "bursty"
    else:
        morph = "stochastic"

    return ShapeStats(n=int(n), trend_slope_per_yr=slope, trend_r2=r2,
                      kendall_tau=float(kt.statistic), kendall_p=float(kt.pvalue),
                      two_state_dbic=dbic, state_sep_sigma=sep, duty_fraction=duty,
                      square_fraction=square, transition_rate=trans,
                      ls_power=ls_power, ls_period_yr=ls_period,
                      burst_skew=skew, morphology=morph)


# --------------------------------------------------------------------------
# Colour temperature of the varying component
# --------------------------------------------------------------------------
def temperature_from_w2_w1_ratio(ratio: float, t_lo: float = 100.0,
                                 t_hi: float = 2500.0, n: int = 2000) -> float:
    """Dust temperature implied by an in-band W2/W1 *flux-density* ratio.

    ``B_nu(T, nu_W2)/B_nu(T, nu_W1)`` decreases monotonically with temperature,
    so a grid inversion is exact to the grid spacing.  Returns NaN outside the
    representable range rather than clipping silently to an edge --- a clipped
    temperature masquerading as a measurement is exactly the failure mode the
    ledger warns about.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return float("nan")
    grid = np.linspace(t_lo, t_hi, n)
    r = planck_bnu(grid, band_freq_hz("W2")) / planck_bnu(grid, band_freq_hz("W1"))
    if ratio > r.max() or ratio < r.min():
        return float("nan")
    return float(np.interp(ratio, r[::-1], grid[::-1]))


@dataclass
class ColourStats:
    """Is the varying component isothermal while it varies?"""

    n: int
    t_var_k: float              # temperature of the varying component
    t_var_err_k: float
    isothermal_chi2_red: float
    isothermal_p: float
    t_drift_k_per_yr: float
    t_drift_sigma: float
    ratio_mean: float
    ratio_scatter: float
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def colour_stability(t_yr, exc_w1, err_w1, exc_w2, err_w2,
                     min_snr: float = 3.0) -> ColourStats | None:
    """Temperature stability of the *varying* component, epoch by epoch.

    Inputs are the per-epoch **excess** fluxes (photosphere already subtracted)
    in the two bands, in the same units.  Epochs where the excess is not detected
    at ``min_snr`` in both bands are dropped: a ratio of two noise values is
    noise, and letting it in would fabricate a temperature drift.
    """
    t = np.asarray(t_yr, dtype=float)
    a = np.asarray(exc_w1, dtype=float)
    ea = np.asarray(err_w1, dtype=float)
    b = np.asarray(exc_w2, dtype=float)
    eb = np.asarray(err_w2, dtype=float)
    ok = (np.isfinite(t) & np.isfinite(a) & np.isfinite(b) & (ea > 0) & (eb > 0)
          & (a > min_snr * ea) & (b > min_snr * eb))
    t, a, ea, b, eb = t[ok], a[ok], ea[ok], b[ok], eb[ok]
    n = t.size
    if n < 4:
        return None

    rho = b / a
    rho_err = rho * np.sqrt((ea / a) ** 2 + (eb / b) ** 2)
    w = 1.0 / rho_err**2
    rho_bar = float(np.sum(w * rho) / np.sum(w))
    chi2 = float(np.sum(w * (rho - rho_bar) ** 2))
    dof = max(n - 1, 1)
    from scipy import stats as _st
    p = float(_st.chi2.sf(chi2, dof))

    slope, _i, _r2 = _weighted_line(t, rho, rho_err)
    sxx = float(np.sum(w * (t - np.sum(w * t) / np.sum(w)) ** 2))
    slope_err = float(np.sqrt(1.0 / sxx)) if sxx > 0 else float("nan")

    t_var = temperature_from_w2_w1_ratio(rho_bar)
    t_hi = temperature_from_w2_w1_ratio(rho_bar + float(np.mean(rho_err)))
    t_lo = temperature_from_w2_w1_ratio(max(rho_bar - float(np.mean(rho_err)), 1e-6))
    t_err = float(abs(t_lo - t_hi) / 2.0) if np.isfinite(t_lo) and np.isfinite(t_hi) \
        else float("nan")

    # Translate the ratio drift into a temperature drift via the local derivative.
    d_rho = 0.02 * rho_bar
    t_plus = temperature_from_w2_w1_ratio(rho_bar + d_rho)
    t_minus = temperature_from_w2_w1_ratio(rho_bar - d_rho)
    dtdrho = ((t_plus - t_minus) / (2.0 * d_rho)
              if np.isfinite(t_plus) and np.isfinite(t_minus) else float("nan"))
    t_drift = float(slope * dtdrho) if np.isfinite(dtdrho) else float("nan")
    drift_sigma = float(abs(slope) / slope_err) if np.isfinite(slope_err) and \
        slope_err > 0 else float("nan")

    if not np.isfinite(drift_sigma):
        verdict = "undetermined"
    elif drift_sigma <= COLOUR_DRIFT_SIGMA_MAX and p > 0.01:
        verdict = "isothermal"
    elif drift_sigma > COLOUR_DRIFT_SIGMA_MAX:
        verdict = "temperature_drifting"
    else:
        verdict = "colour_scatter"

    return ColourStats(n=int(n), t_var_k=float(t_var), t_var_err_k=t_err,
                       isothermal_chi2_red=float(chi2 / dof), isothermal_p=p,
                       t_drift_k_per_yr=t_drift, t_drift_sigma=drift_sigma,
                       ratio_mean=rho_bar,
                       ratio_scatter=float(np.std(rho, ddof=1)) if n > 1 else 0.0,
                       verdict=verdict)


# --------------------------------------------------------------------------
# The cut
# --------------------------------------------------------------------------
@dataclass
class Discrimination:
    """The verdict, with every input that produced it kept alongside."""

    verdict: str
    is_candidate: bool
    reasons: list[str]
    modulation_index: float
    modulation_index_err: float
    tau_ref: float
    tau_upper: float
    frac_excess: float
    amp_ptp: float
    f_var: float
    f_var_sigma: float
    optical_ratio: float
    morphology: str
    colour_verdict: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = ";".join(self.reasons)
        return d


def modulation_index(amp_ptp: float, amp_ptp_err: float, exc: ExcessMeasurement
                     ) -> tuple[float, float]:
    """``m = A_obs (1 + f) / (2 f)`` --- the fraction of the excess that is switching.

    Written against the measured **band** excess ``f``, so the dust temperature,
    the stellar temperature and the distance all cancel (see
    :func:`seti.vigil.excess.max_amplitude_for_frac_excess`).  Returns
    ``(m, sigma_m)``; ``m = inf`` when a significant amplitude sits on a
    *negative* measured excess, which is not a candidate but a falsification ---
    the variability cannot be the excess.
    """
    if exc is None or not exc.measured or not np.isfinite(exc.frac_excess):
        return float("nan"), float("nan")
    f, sf = float(exc.frac_excess), float(exc.frac_excess_err)
    if f <= 0:
        return (float("inf") if np.isfinite(amp_ptp) and amp_ptp > 0
                else float("nan")), float("nan")
    a_max = max_amplitude_for_frac_excess(f)
    if a_max <= 0:
        return float("nan"), float("nan")
    m = float(amp_ptp / a_max)
    dm_da = (1.0 + f) / (2.0 * f)
    dm_df = -amp_ptp / (2.0 * f**2)
    var = 0.0
    if np.isfinite(amp_ptp_err):
        var += (dm_da * amp_ptp_err) ** 2
    if np.isfinite(sf):
        var += (dm_df * sf) ** 2
    return m, float(np.sqrt(var)) if var > 0 else float("nan")


def discriminate(var_w1, var_w2, exc_w1: ExcessMeasurement | None,
                 exc_w2: ExcessMeasurement | None,
                 shape: ShapeStats | None = None,
                 colour: ColourStats | None = None,
                 optical_fvar: float | None = None,
                 optical_measured: bool = False,
                 cfg: dict | None = None) -> Discrimination:
    """Apply the VIGIL cut to one star.

    Order matters.  The measurability gates fire first, so a star that could not
    be characterised is never reported as a low-excess object --- "not measured"
    and "measured low" are different, and conflating them is how a channel turns
    its failures into candidates.
    """
    c = cfg or {}
    fvar_sigma_min = float(c.get("fvar_sigma_min", FVAR_SIGMA_MIN))
    fvar_min = float(c.get("fvar_min", FVAR_MIN))
    opt_ratio_min = float(c.get("optical_ratio_min", OPTICAL_RATIO_MIN))
    opt_fvar_max = float(c.get("optical_fvar_max", OPTICAL_FVAR_MAX))
    tau_max = float(c.get("tau_max", TAU_MAX))
    mod_min = float(c.get("modulation_min", MODULATION_MIN))
    mod_max = float(c.get("modulation_max", MODULATION_MAX))
    decay_r2_max = float(c.get("decay_r2_max", DECAY_R2_MAX))
    from .excess import EXCESS_PRECISION_MAX
    prec_max = float(c.get("excess_precision_max", EXCESS_PRECISION_MAX))

    reasons: list[str] = []
    prim = var_w2 if var_w2 is not None else var_w1
    exc = exc_w2 if (exc_w2 is not None and exc_w2.measured) else exc_w1

    def _bail(verdict: str, m_val: float = float("nan"),
              m_e: float = float("nan")) -> Discrimination:
        return Discrimination(
            verdict=verdict, is_candidate=False, reasons=reasons,
            modulation_index=m_val, modulation_index_err=m_e,
            tau_ref=exc.tau_ref if exc is not None and exc.measured else float("nan"),
            tau_upper=exc.tau_upper if exc is not None and exc.measured else float("nan"),
            frac_excess=exc.frac_excess if exc is not None and exc.measured
            else float("nan"),
            amp_ptp=prim.amp_ptp if prim is not None else float("nan"),
            f_var=prim.f_var if prim is not None else float("nan"),
            f_var_sigma=prim.f_var_sigma if prim is not None else float("nan"),
            optical_ratio=float("nan"),
            morphology=shape.morphology if shape is not None else "unmeasured",
            colour_verdict=colour.verdict if colour is not None else "unmeasured")

    # --- measurability -----------------------------------------------------
    if var_w1 is None or var_w2 is None:
        reasons.append("mid_ir_variability_not_measured_in_both_bands")
        return _bail("NOT_MEASURED")
    if exc is None or not exc.measured:
        reasons.append("no_photosphere_prediction_so_excess_unmeasured")
        return _bail("NOT_MEASURED")
    # An UNMEASURED excess is not a LOW excess.  The excess must be pinned
    # precisely enough that its 2-sigma upper limit could in principle land below
    # the threshold; otherwise the star is uncharacterised, not a candidate.
    if not np.isfinite(exc.tau_err) or 2.0 * exc.tau_err > tau_max:
        reasons.append(
            f"excess_precision_2sigma_tau={2.0 * exc.tau_err:.2e}_cannot_bound_"
            f"below_{tau_max:.1e}")
        return _bail("NOT_MEASURED")
    if not np.isfinite(exc.frac_excess_err) or exc.frac_excess_err > prec_max:
        reasons.append(f"excess_precision_{exc.frac_excess_err:.3f}_worse_than_{prec_max}")
        return _bail("NOT_MEASURED")

    # --- the mid-IR variability itself -------------------------------------
    if prim.f_var_sigma < fvar_sigma_min or prim.f_var < fvar_min:
        reasons.append("mid_ir_constant")
        return _bail("NOT_VARIABLE")
    # Two-band requirement: a single-band mid-IR excursion is an artefact.
    if var_w1.f_var_sigma < 3.0 or var_w2.f_var_sigma < 3.0:
        reasons.append("variable_in_one_band_only")
        return _bail("SINGLE_BAND")

    # --- optical constancy: kills YSOs, dippers, AGN; NOT the EDD ----------
    opt_ratio = float("nan")
    if optical_measured and optical_fvar is not None and np.isfinite(optical_fvar):
        opt_ratio = float(prim.f_var / optical_fvar) if optical_fvar > 0 else np.inf
        if optical_fvar > opt_fvar_max and opt_ratio < opt_ratio_min:
            reasons.append("optically_variable")
            return _bail("OPTICALLY_VARIABLE")
    else:
        reasons.append("optical_constancy_untested")

    # --- the discriminator -------------------------------------------------
    m, m_err = modulation_index(prim.amp_ptp, prim.amp_ptp_err, exc)
    tau = exc.tau_ref
    tau_up = exc.tau_upper

    # A significant amplitude riding on an excess too small (or negative) to
    # produce it is not a technosignature, it is a falsification: something other
    # than the circumstellar excess is moving the flux.
    m_lo = m - m_err if np.isfinite(m_err) else m
    if np.isfinite(m_lo) and m_lo > mod_max:
        reasons.append(f"amplitude_exceeds_what_the_excess_allows_m={m:.2f}")
        return _bail("AMPLITUDE_EXCESS_INCONSISTENT", m, m_err)
    if np.isinf(m):
        reasons.append("significant_amplitude_on_a_negative_measured_excess")
        return _bail("AMPLITUDE_EXCESS_INCONSISTENT", m, m_err)
    if np.isnan(m):
        reasons.append("modulation_index_undefined")
        return _bail("NOT_MEASURED", m, m_err)

    # The brief's cut: LOW fractional excess.  Applied to the 2-sigma upper limit
    # so that "low" is a bound, not a point estimate that happened to scatter low.
    low_excess = bool(np.isfinite(tau_up) and tau_up <= tau_max)
    if not low_excess:
        reasons.append(f"fractional_excess_tau_upper={tau_up:.2e}_above_"
                       f"{tau_max:.1e}_EDD_regime")

    # The sharpened, temperature-free cut.  Rejects only when the modulation
    # index is *significantly* below threshold -- for the confounder population
    # (a large, well-measured excess) it is tightly constrained and decisive,
    # while for a marginal excess it is honestly unconstrained and abstains.
    m_hi = m + m_err if np.isfinite(m_err) else m
    if np.isfinite(m_hi) and m_hi < mod_min:
        reasons.append(f"modulation_index_m={m:.2f}+-{m_err:.2f}_significantly_below_"
                       f"{mod_min}_small_perturbation_on_a_steady_excess")
        return _bail("EXTREME_DEBRIS_DISK_LIKE", m, m_err)
    if not low_excess:
        return _bail("EXTREME_DEBRIS_DISK_LIKE", m, m_err)
    if not np.isfinite(m_err) or m_err > 0.5 * max(m, 1e-6):
        reasons.append("modulation_index_unconstrained_excess_cut_carries_the_decision")

    # --- morphology --------------------------------------------------------
    if shape is not None:
        if shape.morphology == "secular_decay" and shape.trend_r2 >= decay_r2_max:
            reasons.append("secular_decay_morphology_collisional_cascade")
            return _bail("DECAY_MORPHOLOGY", m, m_err)
    else:
        reasons.append("morphology_untested")

    # --- colour temperature ------------------------------------------------
    if colour is not None and colour.verdict == "temperature_drifting":
        reasons.append("varying_component_temperature_drifts_cascade_like")
        return _bail("TEMPERATURE_DRIFT", m, m_err)
    if colour is None:
        reasons.append("colour_temperature_untested")

    reasons.append("low_excess_high_modulation")
    if shape is not None and shape.morphology == "two_state":
        reasons.append("two_state_duty_cycle")
    if colour is not None and colour.verdict == "isothermal":
        reasons.append("isothermal_during_variation")

    return Discrimination(
        verdict="VIGIL_CANDIDATE", is_candidate=True, reasons=reasons,
        modulation_index=m, modulation_index_err=m_err, tau_ref=tau,
        tau_upper=tau_up, frac_excess=exc.frac_excess,
        amp_ptp=prim.amp_ptp, f_var=prim.f_var, f_var_sigma=prim.f_var_sigma,
        optical_ratio=opt_ratio,
        morphology=shape.morphology if shape is not None else "unmeasured",
        colour_verdict=colour.verdict if colour is not None else "unmeasured")


__all__ = ["COLOUR_DRIFT_SIGMA_MAX", "DECAY_R2_MAX", "FVAR_MIN", "FVAR_SIGMA_MIN",
           "MODULATION_MAX", "MODULATION_MIN", "OPTICAL_FVAR_MAX",
           "OPTICAL_RATIO_MIN", "TAU_MAX", "TWO_STATE_DBIC_MIN", "ColourStats",
           "Discrimination", "ShapeStats", "colour_stability", "discriminate",
           "modulation_index", "shape_stats", "temperature_from_w2_w1_ratio"]

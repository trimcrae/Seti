"""Shape statistics in temperature space — the ISOTHERM discriminants.

Every executed waste-heat search selects on how MUCH infrared there is.  This
module computes what SHAPE it has.  Four statistics, in decreasing order of how
cleanly novel they are (see ``docs/isotherm.md`` for the audited prior art):

1. **Emissivity index ``beta``, fitted as a free parameter.**  Real grains have
   ``Q_abs ~ lam**(-beta)`` for ``lam > 2*pi*a``, so their SED falls off *faster*
   than a Planck function in the Rayleigh-Jeans tail; astronomical silicate and
   carbon sit at ``beta ~ 1-2``.  ``beta = 0`` — a true Planck function —
   requires emitting bodies large compared with every observed wavelength.  No
   technosignature search has ever fitted beta.
2. **Width of the temperature distribution, ``dT/T``.**  Dust on a radial
   gradient obeys ``T ~ r**(-1/2)``, hence ``dT/T = 0.5 * dr/r``: the SED is
   *necessarily* a broad superposition.  Reported as an upper limit, and
   compared with the narrowest radial extent nature is known to produce.
3. **Component multiplicity in geometric progression.**  >= 3 resolved discrete
   temperatures with a constant ratio.  Natural radial structure is continuous;
   two-temperature (warm belt + cold belt) systems are common and are NOT
   anomalous, so the test is >= 3 AND geometric AND resolved AND beating the
   continuous-gradient model.
4. **Silicate-feature equivalent width at 9.7 and 18 micron.**  A featureless
   thermal continuum is not ordinary dust.

Honesty about (4): the *idea* is not new.  Wright et al. 2014 (Gh II, sec. 3.3)
proposed absence of PAH/silicate features as a waste-heat discriminant but never
executed it, and Carrigan 2009 rejected his largest single group (3058 IRAS LRS
sources classed "E") on the presence of a 9.7 micron feature.  What is new here
is running it *forward* as a quantitative selection statistic over a spectral
archive rather than backward as candidate vetting, and the 18 micron band, which
appears nowhere in that literature.

Contamination: background-galaxy confusion killed every Project Hephaistos
candidate (JWST/MIRI resolved two of them as a Hot DOG at z~0.9 and a dusty
starburst at z~0.4, both within ~1 arcsec).  ``extragalactic_interloper`` is a
first-class stage here, not an afterthought: a redshifted dusty galaxy has a
feature-rich, multi-component SED, so these very statistics are among the few
that can reject it from the spectrum alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sed_model import (
    LAM0_UM,
    T_MAX_K,
    T_MIN_K,
    chi2_one_component,
    fit_discrete,
    fit_gradient,
    select_n_components,
)

# --- physical reference values ---------------------------------------------

# Wien displacement (F_nu peak, not F_lam): lam_peak * T = 5099 micron K.
_WIEN_NU_UMK = 5099.44

# Narrowest radial extent nature is known to produce in a debris ring.  The
# tightest resolved rings (HR 4796A, Fomalhaut) sit at dr/r ~ 0.06-0.18; 0.05 is
# below every published value and is used as the ABSOLUTE floor.  dT/T = dr/2r,
# so the absolute temperature-width floor is 0.025.
NATURAL_DR_OVER_R_ABSOLUTE = 0.05
NATURAL_DR_OVER_R_NOMINAL = 0.20

# A second, usually larger, natural width: a realistic grain-size distribution
# puts grains of different radius at different temperatures at the SAME orbital
# radius, since T_grain ~ a**(-beta/(4+beta)).  Over a >= 1 decade of grain size
# at beta ~ 1 this alone is dT/T ~ 0.1-0.2.  Quoted, not imposed, because a
# beta = 0 emitter has no such spread by construction — which is precisely why
# the width test and the beta test have to be passed together.
GRAIN_SIZE_DT_OVER_T = 0.10

# Above this the "dust" is hotter than grains survive -> unresolved companion
# photosphere (repo contamination ledger).
T_COMPANION_K = 1800.0


def natural_floor_dt_over_t(dr_over_r: float = NATURAL_DR_OVER_R_ABSOLUTE) -> float:
    """dT/T implied by a radial extent dr/r, from T ~ r**(-1/2)."""
    return 0.5 * float(dr_over_r)


# --- spectral features -----------------------------------------------------

# name -> (band_lo, band_hi, blue anchor, red anchor) in micron, rest frame.
FEATURES: dict[str, tuple] = {
    "silicate_10": (8.2, 12.3, (5.6, 7.4), (13.0, 14.2)),
    "silicate_18": (15.5, 21.5, (14.0, 15.0), (24.0, 26.5)),
    "pah_6.2": (6.0, 6.5, (5.4, 5.9), (6.6, 7.0)),
    "pah_7.7": (7.2, 8.2, (6.6, 7.1), (8.9, 9.2)),
    "pah_11.3": (11.0, 11.7, (10.3, 10.9), (11.9, 12.4)),
    "pah_12.7": (12.4, 13.0, (11.9, 12.3), (13.2, 13.8)),
    "ice_h2o_6.0": (5.7, 6.4, (5.2, 5.6), (6.6, 7.0)),
    "ice_co2_15.2": (14.9, 15.6, (14.2, 14.8), (15.8, 16.6)),
    "c2h2_13.7": (13.5, 14.0, (12.9, 13.4), (14.2, 14.8)),
    "forsterite_23.7": (22.8, 24.6, (21.3, 22.5), (25.4, 26.6)),
    "forsterite_33.6": (32.4, 34.6, (30.4, 31.6), (35.4, 36.6)),
}

# Features whose presence is, on its own, proof of ordinary astrophysical dust.
DUST_VETO_FEATURES = ("silicate_10", "silicate_18", "pah_6.2", "pah_7.7",
                      "pah_11.3", "pah_12.7", "ice_h2o_6.0", "ice_co2_15.2",
                      "c2h2_13.7", "forsterite_23.7", "forsterite_33.6")

# Redshift-coherence test uses only the strong, unambiguous bands.
_Z_TEST_FEATURES = ("pah_6.2", "pah_7.7", "pah_11.3", "silicate_10", "silicate_18")


# Order of the local continuum polynomial in log(F_nu) vs log(lam).
#
# CALIBRATED, not guessed.  A featureless Planck function is CURVED in log-log,
# so a straight-line continuum through anchor windows manufactures a spurious
# "silicate" feature: on a pure 250 K blackbody a linear continuum yields a
# false EW of +1.67 micron at 9.7 micron — as large as a real silicate band
# (+3.0 micron for a 35% bump).  Quadratic reduces this to -0.23 micron; cubic
# to -0.006 micron, while still recovering +0.69 micron for the real bump.
# Cubic it is, with a ~100x margin between the systematic floor and the veto
# threshold.  This bug would have vetoed every genuine candidate.
_CONTINUUM_ORDER = 3
_MIN_ANCHOR_PER_SIDE = 4

# Systematic error on an equivalent width, as a fraction of the band width,
# from continuum placement.  Measured at ~0.0015 for a cubic on a featureless
# Planck function; 0.01 is deliberately conservative.
CONTINUUM_EW_SYSTEMATIC_FRAC = 0.01


def _local_continuum(lam: np.ndarray, flux: np.ndarray, err: np.ndarray,
                     blue: tuple, red: tuple) -> tuple[np.ndarray, bool, int]:
    """Local continuum: a polynomial in log-log through the two anchor windows."""
    mb = (lam >= blue[0]) & (lam <= blue[1])
    mr = (lam >= red[0]) & (lam <= red[1])
    finite = np.isfinite(flux) & (flux > 0) & np.isfinite(err)
    mb &= finite
    mr &= finite
    n_blue, n_red = int(mb.sum()), int(mr.sum())
    if n_blue < 2 or n_red < 2:
        return np.full_like(lam, np.nan), False, 0
    m = mb | mr
    # Drop the polynomial order when the anchors cannot support it.
    order = _CONTINUUM_ORDER
    while order > 1 and (min(n_blue, n_red) < _MIN_ANCHOR_PER_SIDE
                         or int(m.sum()) < order + 3):
        order -= 1
    w = 1.0 / np.clip(err[m] / np.abs(flux[m]), 1e-6, None) ** 2
    try:
        coef = np.polyfit(np.log10(lam[m]), np.log10(flux[m]), order, w=np.sqrt(w))
    except Exception:  # noqa: BLE001 - degenerate anchor geometry
        return np.full_like(lam, np.nan), False, 0
    return 10.0 ** np.polyval(coef, np.log10(lam)), True, order


def feature_equivalent_width(lam_um, flux, err, band: tuple, blue: tuple,
                             red: tuple) -> dict:
    """Equivalent width of one feature against a local power-law continuum.

    ``ew_um = Integral[(F - F_cont)/F_cont dlam]``, positive in emission and
    negative in absorption.  Both count as "dust is present": emission means
    optically thin silicate, absorption means an embedded/optically thick line
    of sight.  The veto therefore uses ``|ew|``.
    """
    lam = np.asarray(lam_um, float)
    f = np.asarray(flux, float)
    e = np.asarray(err, float)
    out = {"ew_um": float("nan"), "ew_err_um": float("nan"),
           "significance": float("nan"), "covered": False,
           "n_points": 0, "peak_frac_dev": float("nan"), "continuum_order": 0}

    ok = np.isfinite(lam) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    lam, f, e = lam[ok], f[ok], e[ok]
    if lam.size < 6:
        return out
    # HARD requirement: the ENTIRE span blue[0] .. red[1] must lie inside the
    # observed range.  A truncated anchor window makes the continuum polynomial
    # extrapolate off the end of the spectrum, which fabricated 0.25 micron
    # "features" on a pure blackbody and vetoed every candidate.  No
    # extrapolation, ever.
    span_lo, span_hi = min(blue[0], band[0]), max(red[1], band[1])
    if lam.min() > span_lo + 1e-6 or lam.max() < span_hi - 1e-6:
        return out
    inband = (lam >= band[0]) & (lam <= band[1])
    if inband.sum() < 4:
        return out
    cont, okc, order = _local_continuum(lam, f, e, blue, red)
    if not okc:
        return out
    out["continuum_order"] = int(order)

    x = lam[inband]
    dev = (f[inband] - cont[inband]) / cont[inband]
    sig = e[inband] / np.abs(cont[inband])
    if x.size < 4 or not np.all(np.isfinite(dev)):
        return out
    ew = float(np.trapezoid(dev, x))
    # Trapezoid weights, treating pixels as independent (they are, post-binning).
    wts = np.gradient(x)
    ew_stat = float(np.sqrt(np.sum((wts * sig) ** 2)))
    # Continuum-placement systematic.  A cubic through the anchors leaves a
    # residual curvature term on a featureless Planck function measured at
    # ~0.15% of the band width; 1% is a ~7x-conservative floor.  Without it,
    # high-SNR spectra report 20-sigma "features" with EW ~ 0.05 micron, two
    # orders of magnitude below a real silicate band.
    ew_sys = CONTINUUM_EW_SYSTEMATIC_FRAC * float(band[1] - band[0])
    ew_err = float(np.hypot(ew_stat, ew_sys))
    out.update(ew_um=ew, ew_err_um=ew_err, ew_err_stat_um=ew_stat,
               ew_err_sys_um=ew_sys,
               significance=float(ew / ew_err) if ew_err > 0 else float("nan"),
               covered=True, n_points=int(x.size),
               peak_frac_dev=float(dev[np.argmax(np.abs(dev))]))
    return out


def measure_features(lam_um, flux, err, z: float = 0.0,
                     features: dict | None = None) -> dict[str, dict]:
    """Measure every catalogued feature, optionally redshifted by ``z``."""
    feats = features or FEATURES
    s = 1.0 + float(z)
    return {
        name: feature_equivalent_width(
            lam_um, flux, err,
            (band[0] * s, band[1] * s), (blue[0] * s, blue[1] * s),
            (red[0] * s, red[1] * s))
        for name, (band, blue, red) in
        ((n, (v[0:2], v[2], v[3])) for n, v in feats.items())
    }


# --- statistic 1: beta -----------------------------------------------------

def wavelength_leverage(lam_um, t_k: float) -> dict:
    """Can beta and T be separated at all from this coverage?

    On the Rayleigh-Jeans tail ``F_nu ~ T * lam**(-(2+beta))``, so beta and T
    are perfectly degenerate unless the Wien peak is *inside* the band with
    points on both sides.  This gates the beta statistic honestly: outside the
    leverage window we report ``beta`` as unconstrained rather than pretending.
    """
    lam = np.asarray(lam_um, float)
    lam = lam[np.isfinite(lam) & (lam > 0)]
    out = {"peak_um": float("nan"), "in_band": False, "n_blue": 0,
           "n_red": 0, "beta_constrained": False,
           "t_window_k": [float("nan"), float("nan")]}
    if lam.size < 8 or not np.isfinite(t_k) or t_k <= 0:
        return out
    lo, hi = float(lam.min()), float(lam.max())
    peak = _WIEN_NU_UMK / float(t_k)
    n_blue = int((lam < peak).sum())
    n_red = int((lam > peak).sum())
    out.update(peak_um=peak, in_band=bool(lo < peak < hi),
               n_blue=n_blue, n_red=n_red,
               t_window_k=[_WIEN_NU_UMK / hi, _WIEN_NU_UMK / lo])
    # Need real lever arms: >= 5 elements each side and a decade of tail.
    out["beta_constrained"] = bool(out["in_band"] and n_blue >= 5 and n_red >= 5
                                   and hi / peak >= 1.6)
    return out


def profile_scan(objective, grid) -> dict:
    """Delta-chi2 profile over a 1-D grid; returns min and confidence bounds."""
    vals = np.asarray([objective(float(g)) for g in grid], float)
    good = np.isfinite(vals)
    if good.sum() < 3:
        return {"best": float("nan"), "chi2_min": float("nan"),
                "lo_1sig": float("nan"), "hi_1sig": float("nan"),
                "upper_95": float("nan"), "grid": list(map(float, grid)),
                "chi2": [float(v) for v in vals]}
    g = np.asarray(grid, float)[good]
    v = vals[good]
    i = int(np.argmin(v))
    dchi = v - v[i]

    def crossing(target, direction):
        idx = range(i + 1, len(g)) if direction > 0 else range(i - 1, -1, -1)
        prev = i
        for j in idx:
            if dchi[j] >= target:
                # Linear interpolation in the grid variable.
                d0, d1 = dchi[prev], dchi[j]
                if d1 == d0:
                    return float(g[j])
                return float(g[prev] + (g[j] - g[prev]) * (target - d0) / (d1 - d0))
            prev = j
        return float("nan")

    return {"best": float(g[i]), "chi2_min": float(v[i]),
            "lo_1sig": crossing(1.0, -1), "hi_1sig": crossing(1.0, +1),
            "upper_95": crossing(3.84, +1),
            "grid": [float(x) for x in g], "chi2": [float(x) for x in v]}


def estimate_beta(lam_um, flux, err, lam0_um: float = LAM0_UM,
                  beta_grid=None) -> dict:
    """Fit the single-component emissivity index with a delta-chi2 uncertainty.

    VALID ONLY for a source the data prefer to describe with one component.  A
    single modified blackbody fitted to a broad radial gradient returns
    ``beta ~ 0`` regardless of the true grain emissivity — verified: a
    ``beta = 1`` gradient disk reports ``beta = 0.00 +/- 0.001`` here, a false
    "true Planck function".  ``classify`` therefore reads beta off the model the
    data select, never off this function alone.
    """
    grid = np.asarray(beta_grid if beta_grid is not None
                      else np.arange(-0.6, 3.01, 0.15), float)

    def obj(b):
        return chi2_one_component(lam_um, flux, err, float(b), lam0_um)[0]

    prof = profile_scan(obj, grid)
    free = fit_discrete(lam_um, flux, err, 1, beta=None, lam0_um=lam0_um)
    lev = wavelength_leverage(lam_um, free.temps_k[0] if free.temps_k else np.nan)
    lo, hi = prof["lo_1sig"], prof["hi_1sig"]
    sig = (float(hi - lo) / 2.0 if np.isfinite(lo) and np.isfinite(hi)
           else float("nan"))
    return {
        "beta": float(prof["best"]), "beta_err": sig,
        "beta_lo_1sig": float(lo), "beta_hi_1sig": float(hi),
        "beta_free_fit": float(free.beta) if free.success else float("nan"),
        "t_k": float(free.temps_k[0]) if free.temps_k else float("nan"),
        "chi2": float(prof["chi2_min"]),
        "reduced_chi2": float(free.reduced_chi2) if free.success else float("nan"),
        # "Consistent with a true Planck function" = beta = 0 inside 1 sigma.
        "planck_consistent": bool(np.isfinite(lo) and np.isfinite(hi)
                                  and lo <= 0.0 <= hi),
        "dusty_beta": bool(np.isfinite(lo) and lo >= 0.5),
        **{f"leverage_{k}": v for k, v in lev.items()},
    }


def estimate_beta_in_model(lam_um, flux, err, temps_seed, lam0_um: float = LAM0_UM,
                           beta_grid=None) -> dict:
    """Profile beta WITHIN an N-component model, temperatures re-optimised.

    Required whenever the data prefer more than one component: the
    single-component beta of a 3-shell cascade pins at the grid edge (-0.6) and
    carries no usable uncertainty, so reading ``planck_consistent`` off it would
    throw away every genuine cascade.
    """
    temps = np.asarray(list(temps_seed), float)
    n = temps.size
    grid = np.asarray(beta_grid if beta_grid is not None
                      else np.arange(-0.6, 3.01, 0.15), float)
    out = {"beta": float("nan"), "beta_lo_1sig": float("nan"),
           "beta_hi_1sig": float("nan"), "beta_err": float("nan"),
           "planck_consistent": False, "at_grid_edge": True,
           "n_components": int(n)}
    if n == 0:
        return out

    def obj(b):
        return fit_discrete(lam_um, flux, err, n, beta=float(b),
                            lam0_um=lam0_um, seed_temps=temps).chi2

    prof = profile_scan(obj, grid)
    if not np.isfinite(prof["best"]):
        return out
    lo, hi = prof["lo_1sig"], prof["hi_1sig"]
    edge = bool(prof["best"] <= grid[0] + 1e-9 or prof["best"] >= grid[-1] - 1e-9)
    out.update(beta=float(prof["best"]), beta_lo_1sig=float(lo),
               beta_hi_1sig=float(hi), at_grid_edge=edge,
               beta_err=(float(hi - lo) / 2.0
                         if np.isfinite(lo) and np.isfinite(hi) else float("nan")),
               planck_consistent=bool(np.isfinite(lo) and np.isfinite(hi)
                                      and lo <= 0.0 <= hi))
    return out


# --- statistic 2: temperature-distribution width ---------------------------

def temperature_width(lam_um, flux, err, beta: float | None = None,
                      lam0_um: float = LAM0_UM) -> dict:
    """Width of the fitted temperature distribution, with a 95% upper limit.

    The gradient model is fitted, then ``log10(T_in/T_out)`` is profiled.  For a
    genuinely isothermal source the profile minimum sits at zero width and the
    informative number is the UPPER LIMIT, which is what gets compared with the
    natural floor.
    """
    grad = fit_gradient(lam_um, flux, err, beta=beta, lam0_um=lam0_um)
    out = {"t_in_k": float(grad.t_in_k), "t_out_k": float(grad.t_out_k),
           "p_index": float(grad.p_index), "beta_gradient": float(grad.beta),
           "chi2_gradient": float(grad.chi2),
           "dt_over_t": float("nan"), "dt_over_t_upper95": float("nan"),
           "dr_over_r": float("nan"), "dr_over_r_upper95": float("nan"),
           "narrower_than_absolute_floor": False,
           "narrower_than_nominal_floor": False,
           "floor_absolute": natural_floor_dt_over_t(NATURAL_DR_OVER_R_ABSOLUTE),
           "floor_nominal": natural_floor_dt_over_t(NATURAL_DR_OVER_R_NOMINAL),
           "grain_size_floor": GRAIN_SIZE_DT_OVER_T,
           "success": bool(grad.success)}
    if not grad.success or not np.isfinite(grad.t_in_k):
        return out

    def width(t_in, t_out):
        return 2.0 * (t_in - t_out) / (t_in + t_out)

    out["dt_over_t"] = float(width(grad.t_in_k, grad.t_out_k))
    out["dr_over_r"] = 2.0 * out["dt_over_t"]

    # Profile the log width, re-optimising T_out, p and (optionally) beta.
    tmid = float(np.sqrt(grad.t_in_k * grad.t_out_k))

    # beta is held at the gradient fit's value while the width is profiled:
    # beta and width both flatten the Rayleigh-Jeans tail, so letting both float
    # lets a wide component masquerade as a narrow one with low beta.
    beta_fix = float(grad.beta) if np.isfinite(grad.beta) else 0.0

    def obj(dl):
        dl = float(abs(dl))
        best = np.inf
        for scale in (0.7, 1.0, 1.4):
            t_out = np.clip(tmid * scale / 10 ** (dl / 2), T_MIN_K, T_MAX_K)
            t_in = np.clip(t_out * 10**dl, T_MIN_K, T_MAX_K)
            for p in (0.0, 1.0, 2.0):
                f = fit_discrete_gradient_fixed(lam_um, flux, err, t_in, t_out,
                                                p, beta_fix, lam0_um)
                best = min(best, f)
        return best

    dl_grid = np.concatenate([[0.0], np.geomspace(0.005, 2.0, 20)])
    prof = profile_scan(obj, dl_grid)
    if np.isfinite(prof["upper_95"]):
        r = 10.0 ** float(prof["upper_95"])
        w = 2.0 * (r - 1.0) / (r + 1.0)
        out["dt_over_t_upper95"] = float(w)
        out["dr_over_r_upper95"] = float(2.0 * w)
        out["narrower_than_absolute_floor"] = bool(w < out["floor_absolute"])
        out["narrower_than_nominal_floor"] = bool(w < out["floor_nominal"])
    return out


def fit_discrete_gradient_fixed(lam_um, flux, err, t_in, t_out, p_index,
                                beta, lam0_um) -> float:
    """chi2 of the gradient model at fixed (T_in, T_out, p), amplitude by NNLS."""
    from .sed_model import _solve_amps, gradient_sed

    col = gradient_sed(np.asarray(lam_um, float), t_in, t_out, p_index,
                       0.0 if beta is None else float(beta), lam0_um)
    mx = col.max() if col.size else 0.0
    if not np.isfinite(mx) or mx <= 0:
        return float("inf")
    if beta is None:
        best = np.inf
        for b in (0.0, 0.5, 1.0, 1.5, 2.0):
            c = gradient_sed(np.asarray(lam_um, float), t_in, t_out, p_index,
                             b, lam0_um)
            m = c.max()
            if m > 0:
                best = min(best, _solve_amps((c / m)[:, None],
                                             np.asarray(flux, float),
                                             np.asarray(err, float))[1])
        return float(best)
    return float(_solve_amps((col / mx)[:, None], np.asarray(flux, float),
                             np.asarray(err, float))[1])


def component_width_limit(lam_um, flux, err, temps_k, beta: float | None = None,
                          lam0_um: float = LAM0_UM) -> dict:
    """Upper limit on the width of EACH component in a multi-component fit.

    This is the test that separates N engineered shells from N natural belts.
    Three warm/cold *belts* are three radial gradients; three shells are three
    narrow features.  A shared fractional width is profiled across all
    components, so the test costs one parameter regardless of N.
    """
    from .sed_model import fit_multi_gradient

    out = {"width_dex": float("nan"), "dt_over_t": float("nan"),
           "dt_over_t_upper95": float("nan"),
           "components_narrower_than_nominal_floor": False,
           "components_narrower_than_absolute_floor": False,
           "n_components": len(list(temps_k)), "success": False}
    temps = np.asarray(list(temps_k), float)
    if temps.size == 0 or np.any(~np.isfinite(temps)):
        return out

    def obj(w):
        return fit_multi_gradient(lam_um, flux, err, temps, float(abs(w)),
                                  beta=beta, lam0_um=lam0_um)

    grid = np.concatenate([[0.0], np.geomspace(0.01, 1.5, 14)])
    prof = profile_scan(obj, grid)
    if not np.isfinite(prof["best"]):
        return out

    def to_width(dex):
        r = 10.0 ** float(dex)
        return 2.0 * (r - 1.0) / (r + 1.0)

    out["width_dex"] = float(prof["best"])
    out["dt_over_t"] = float(to_width(prof["best"]))
    out["success"] = True
    if np.isfinite(prof["upper_95"]):
        w = float(to_width(prof["upper_95"]))
        out["dt_over_t_upper95"] = w
        out["components_narrower_than_nominal_floor"] = bool(
            w < natural_floor_dt_over_t(NATURAL_DR_OVER_R_NOMINAL))
        out["components_narrower_than_absolute_floor"] = bool(
            w < natural_floor_dt_over_t(NATURAL_DR_OVER_R_ABSOLUTE))
    return out


def component_coverage(temps_k, lam_min_um: float, lam_max_um: float) -> list[dict]:
    """Flag components whose Wien peak falls outside the observed band.

    Such a temperature is an extrapolation from one flank of the Planck curve
    and is far less certain than an in-band component; it must not be quoted
    with the same confidence.
    """
    out = []
    for t in temps_k:
        peak = _WIEN_NU_UMK / float(t) if t and np.isfinite(t) and t > 0 else np.nan
        out.append({"t_k": float(t), "peak_um": float(peak),
                    "peak_in_band": bool(np.isfinite(peak)
                                         and lam_min_um <= peak <= lam_max_um)})
    return out


# --- statistic 3: component multiplicity and geometric progression ---------

def geometric_progression_test(temps_k, tol_dex: float = 0.06) -> dict:
    """Are >= 3 temperatures in geometric progression?

    A staged cascade in which each shell sits at a fixed radius multiple of the
    previous one gives ``T_k = T_1 * R**(-(k-1)/2)`` — geometric in temperature.
    The statistic is the RMS scatter of ``log10`` of successive ratios.
    """
    t = np.sort(np.asarray(temps_k, float))[::-1]
    out = {"n": int(t.size), "ratios": [], "mean_ratio": float("nan"),
           "log_ratio_rms_dex": float("nan"), "is_geometric": False,
           "temps_k": [float(x) for x in t]}
    if t.size < 3 or np.any(t <= 0):
        return out
    ratios = t[:-1] / t[1:]
    lr = np.log10(ratios)
    out["ratios"] = [float(r) for r in ratios]
    out["mean_ratio"] = float(10 ** lr.mean())
    # Population RMS about the mean (not the sample sd) — with 2 ratios the
    # sample sd is undefined but the deviation is still meaningful.
    out["log_ratio_rms_dex"] = float(np.sqrt(np.mean((lr - lr.mean()) ** 2)))
    out["is_geometric"] = bool(out["log_ratio_rms_dex"] <= tol_dex)
    return out


def energy_ladder(lum_frac, temps_k) -> dict:
    """Luminosity apportionment across components.

    An energy-conserving cascade with partial covering radiates a comparable
    luminosity per stage: shell k intercepts what leaked past shell k-1.  A
    natural warm-belt/cold-belt pair is usually lopsided instead.  Reported, not
    thresholded hard, because covering fractions are free.
    """
    f = np.asarray(lum_frac, float)
    t = np.asarray(temps_k, float)
    out = {"lum_frac": [float(x) for x in f], "max_over_min": float("nan"),
           "monotonic_with_t": False, "n": int(f.size)}
    if f.size < 2 or np.any(~np.isfinite(f)) or f.min() <= 0:
        return out
    out["max_over_min"] = float(f.max() / f.min())
    order = np.argsort(t)[::-1]
    out["monotonic_with_t"] = bool(np.all(np.diff(f[order]) <= 0)
                                   or np.all(np.diff(f[order]) >= 0))
    return out


# --- contamination: the background galaxy that kills everyone --------------

def extragalactic_interloper(lam_um, flux, err, z_max: float = 1.6,
                             dz: float = 0.02, min_sig: float = 5.0,
                             min_abs_ew_um: float = 0.15) -> dict:
    """Search for a coherent redshifted PAH/silicate system.

    Every Project Hephaistos candidate died of background-galaxy confusion.  A
    dusty galaxy at z ~ 0.4-0.9 has strong PAH and silicate bands that land at
    ``(1+z) * lam_rest``; requiring >= 2 features coherent at ONE redshift is
    the spectral analogue of the repo's existing "3 lines at one z -> galaxy"
    rule.  A z ~ 0 solution is not reported as extragalactic — that is just the
    ordinary rest-frame feature veto.

    The scan is ~80 redshifts x 5 features, so the per-feature bar is 5 sigma
    AND a physically meaningful ``|EW| >= min_abs_ew_um``.  Under a
    significance-only 3-sigma cut this test flagged a pure blackbody at every
    redshift it was offered — 20-sigma "detections" with EW ~ 0.05 micron,
    two orders of magnitude below a real silicate band.
    """
    zs = np.arange(0.0, float(z_max) + 1e-9, float(dz))
    best = {"z": float("nan"), "n_features": 0, "combined_sig": 0.0,
            "features": {}}
    for z in zs:
        feats = measure_features(lam_um, flux, err, z=float(z),
                                 features={k: FEATURES[k] for k in _Z_TEST_FEATURES})
        hits = {k: v for k, v in feats.items()
                if v["covered"] and np.isfinite(v["significance"])
                and abs(v["significance"]) >= min_sig
                and abs(v.get("ew_um", 0.0)) >= min_abs_ew_um}
        if not hits:
            continue
        comb = float(np.sqrt(np.sum([v["significance"] ** 2 for v in hits.values()])))
        if (len(hits), comb) > (best["n_features"], best["combined_sig"]):
            best = {"z": float(z), "n_features": len(hits),
                    "combined_sig": comb,
                    "features": {k: float(v["significance"]) for k, v in hits.items()}}
    flagged = bool(best["n_features"] >= 2 and np.isfinite(best["z"])
                   and best["z"] >= 0.03)
    return {**best, "extragalactic_flag": flagged,
            "note": ("coherent multi-feature system at z >= 0.03 -> redshifted "
                     "dusty galaxy" if flagged else "no coherent redshifted system"),
            "z_max_searched": float(z_max),
            "limitation": "PAH 6.2/7.7 leave a 5-38 um band above z ~ 1.5-3.9; "
                          "high-z interlopers are not excluded by this test"}


def order_step(lam_um, flux, boundary_um: float = 14.2,
               window_um: float = 0.6) -> dict:
    """Flux discontinuity at an IRS module boundary (the dominant systematic).

    Spitzer/IRS SL and LL have different slit widths, so a mispointed or
    spatially extended source shows a step at ~14.2 micron.  A large step means
    the two halves of the spectrum are not the same object's flux, which
    manufactures spurious multi-component structure.
    """
    lam = np.asarray(lam_um, float)
    f = np.asarray(flux, float)
    ok = np.isfinite(lam) & np.isfinite(f)
    lam, f = lam[ok], f[ok]
    lo = (lam >= boundary_um - window_um) & (lam < boundary_um)
    hi = (lam >= boundary_um) & (lam < boundary_um + window_um)
    if lo.sum() < 2 or hi.sum() < 2:
        return {"step_frac": float("nan"), "covered": False}
    a, b = float(np.median(f[lo])), float(np.median(f[hi]))
    if not np.isfinite(a) or not np.isfinite(b) or (a + b) == 0:
        return {"step_frac": float("nan"), "covered": False}
    return {"step_frac": float(2.0 * (b - a) / (a + b)), "covered": True}


# --- the composite verdict -------------------------------------------------

@dataclass
class ShapeVerdict:
    verdict: str
    flags: list[str] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "flags": list(self.flags),
                "vetoes": list(self.vetoes), "stats": self.stats}


DEFAULT_THRESHOLDS = {
    "min_points": 25,
    "min_median_snr": 5.0,
    "max_reduced_chi2": 8.0,
    "feature_veto_sigma": 4.0,
    "feature_veto_abs_ew_um": 0.15,
    "beta_planck_max": 0.35,      # beta upper 1-sigma bound to call it grey
    "delta_bic_component": 10.0,
    "delta_bic_vs_gradient": 10.0,
    "geometric_tol_dex": 0.06,
    "max_order_step_frac": 0.25,
    "n_components_max": 4,
    "t_companion_k": T_COMPANION_K,
}


def compute_shape_stats(lam_um, flux, err, thresholds: dict | None = None,
                        lam0_um: float = LAM0_UM) -> dict:
    """Run every shape statistic on one spectrum.  Pure; no I/O."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    lam = np.asarray(lam_um, float)
    f = np.asarray(flux, float)
    e = np.asarray(err, float)
    ok = np.isfinite(lam) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (lam > 0)
    lam, f, e = lam[ok], f[ok], e[ok]

    stats: dict = {"n_points": int(lam.size)}
    if lam.size < th["min_points"]:
        stats["usable"] = False
        stats["reason"] = "insufficient_points"
        return stats
    snr = float(np.median(np.abs(f) / e))
    stats["median_snr"] = snr
    stats["lam_min_um"] = float(lam.min())
    stats["lam_max_um"] = float(lam.max())
    if snr < th["min_median_snr"]:
        stats["usable"] = False
        stats["reason"] = "low_snr"
        return stats
    stats["usable"] = True

    best, ladder = select_n_components(lam, f, e, n_max=int(th["n_components_max"]),
                                       delta_bic=float(th["delta_bic_component"]),
                                       beta=None, lam0_um=lam0_um)
    grad = fit_gradient(lam, f, e, beta=None, lam0_um=lam0_um)

    stats["fit_best"] = best.to_dict()
    stats["fit_ladder"] = [x.to_dict() for x in ladder]
    stats["fit_gradient"] = grad.to_dict()
    stats["delta_bic_discrete_minus_gradient"] = (
        float(best.bic - grad.bic) if np.isfinite(best.bic) and np.isfinite(grad.bic)
        else float("nan"))

    # WHICH MODEL DOES THE DATA PREFER?  This gate matters more than any single
    # statistic.  A single modified blackbody fitted to a broad radial gradient
    # returns a biased beta near 0 — a false "Planck-consistent" result — so
    # beta must be read off the model the data actually select, never off a
    # single-component fit imposed on a multi-temperature source.
    dbic = (float(best.bic - grad.bic)
            if np.isfinite(best.bic) and np.isfinite(grad.bic) else float("nan"))
    stats["delta_bic_discrete_minus_gradient"] = dbic
    gradient_preferred = bool(np.isfinite(dbic) and dbic > 0)
    stats["preferred_model"] = "gradient" if gradient_preferred else "discrete"
    stats["discrete_beats_gradient"] = bool(
        np.isfinite(dbic) and dbic < -float(th["delta_bic_vs_gradient"]))

    stats["beta"] = estimate_beta(lam, f, e, lam0_um=lam0_um)
    stats["beta"]["beta_gradient_model"] = float(grad.beta)
    stats["beta"]["beta_best_model"] = float(best.beta)
    stats["beta"]["model_dependent"] = bool(
        np.isfinite(grad.beta) and np.isfinite(stats["beta"]["beta"])
        and abs(grad.beta - stats["beta"]["beta"]) > 0.5)
    # For a multi-component source the single-component beta profile is
    # meaningless (it pins at the grid edge with no uncertainty), so profile
    # beta inside the selected model instead.
    if best.n_components >= 2 and best.temps_k:
        stats["beta_in_model"] = estimate_beta_in_model(
            lam, f, e, best.temps_k, lam0_um=lam0_um)

    stats["width"] = temperature_width(lam, f, e, beta=None, lam0_um=lam0_um)
    stats["features"] = measure_features(lam, f, e, z=0.0)
    stats["geometric"] = geometric_progression_test(
        best.temps_k, tol_dex=float(th["geometric_tol_dex"]))
    stats["energy"] = energy_ladder(best.lum_frac, best.temps_k)
    stats["coverage"] = component_coverage(best.temps_k, float(lam.min()),
                                           float(lam.max()))
    stats["extragalactic"] = extragalactic_interloper(lam, f, e)
    stats["order_step"] = order_step(lam, f)
    # Per-component width only matters when multiplicity is claimed, and it is
    # the expensive statistic, so it is computed only then.
    if best.n_components >= 3:
        stats["component_width"] = component_width_limit(
            lam, f, e, best.temps_k, beta=float(best.beta), lam0_um=lam0_um)
    return stats


def screen_spectrum(lam_um, flux, err, thresholds: dict | None = None,
                    lam0_um: float = LAM0_UM) -> dict:
    """Stage-1 screen: cheap statistics that reject the bulk of any archive.

    The full shape analysis costs ~10-30 s per spectrum, which does not scale to
    a 13,000-spectrum atlas.  This runs in ~1 s and rejects on the two things
    that are both cheap and decisive: a strong rest-frame dust feature, and a
    dust-like emissivity index.

    It is deliberately CONSERVATIVE — it may only reject, never promote.  The
    single-component beta is biased *towards* zero for multi-temperature
    sources, so a high beta here is solid evidence of dust and a low beta
    proves nothing; that asymmetry is what makes the screen safe for the S6
    cascade case, whose single-component beta fits at or below zero.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    lam = np.asarray(lam_um, float)
    f = np.asarray(flux, float)
    e = np.asarray(err, float)
    ok = np.isfinite(lam) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (lam > 0)
    lam, f, e = lam[ok], f[ok], e[ok]

    out = {"n_points": int(lam.size), "pass": False, "reason": ""}
    if lam.size < th["min_points"]:
        out["reason"] = "insufficient_points"
        return out
    snr = float(np.median(np.abs(f) / e))
    out["median_snr"] = snr
    out["lam_min_um"] = float(lam.min())
    out["lam_max_um"] = float(lam.max())
    if snr < th["min_median_snr"]:
        out["reason"] = "low_snr"
        return out

    feats = measure_features(lam, f, e, z=0.0)
    hit = [n for n in DUST_VETO_FEATURES
           if feats.get(n, {}).get("covered")
           and np.isfinite(feats[n].get("significance", np.nan))
           and abs(feats[n]["significance"]) >= th["feature_veto_sigma"]
           and abs(feats[n].get("ew_um", 0.0)) >= th["feature_veto_abs_ew_um"]]
    out["features_detected"] = hit
    out["feature_ew_um"] = {n: float(feats[n]["ew_um"]) for n in hit}
    if hit:
        out["reason"] = "dust_features:" + ",".join(hit)
        return out

    # beta from a coarse profile: reject only when the LOWER bound is dust-like.
    grid = np.arange(-0.5, 3.01, 0.25)
    prof = profile_scan(
        lambda b: chi2_one_component(lam, f, e, float(b), lam0_um)[0], grid)
    out["beta_coarse"] = float(prof["best"])
    out["beta_lo_1sig"] = float(prof["lo_1sig"])
    if np.isfinite(prof["lo_1sig"]) and prof["lo_1sig"] >= 0.5:
        out["reason"] = "beta_dustlike"
        return out

    step = order_step(lam, f)
    out["order_step_frac"] = float(step.get("step_frac", np.nan))
    if step.get("covered") and np.isfinite(step["step_frac"]) \
            and abs(step["step_frac"]) > th["max_order_step_frac"]:
        out["reason"] = "order_stitching_step"
        return out

    out["pass"] = True
    out["reason"] = "passed_screen"
    return out


def classify(stats: dict, thresholds: dict | None = None) -> ShapeVerdict:
    """Turn shape statistics into flags, vetoes and a verdict.

    Deliberately NOT "a single blackbody fits well => artificial".  S5 requires
    a temperature width below the natural floor AND a Planck-consistent beta AND
    no dust features AND the isothermal model beating the continuous-gradient
    null.  S6 requires >= 3 resolved components in geometric progression AND
    beating the gradient null.  Any dust feature, any coherent redshifted
    system, or a too-hot component vetoes regardless.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    flags: list[str] = []
    vetoes: list[str] = []

    if not stats.get("usable", False):
        return ShapeVerdict(verdict="INSUFFICIENT_DATA",
                            vetoes=[stats.get("reason", "unusable")],
                            stats=stats)

    # --- vetoes ---
    feats = stats.get("features", {})
    for name in DUST_VETO_FEATURES:
        v = feats.get(name)
        if not v or not v.get("covered"):
            continue
        if (np.isfinite(v.get("significance", np.nan))
                and abs(v["significance"]) >= th["feature_veto_sigma"]
                and abs(v.get("ew_um", 0.0)) >= th["feature_veto_abs_ew_um"]):
            vetoes.append(f"feature_{name}")

    xg = stats.get("extragalactic", {})
    if xg.get("extragalactic_flag"):
        vetoes.append(f"extragalactic_z{xg.get('z'):.2f}")

    step = stats.get("order_step", {})
    if step.get("covered") and np.isfinite(step.get("step_frac", np.nan)) \
            and abs(step["step_frac"]) > th["max_order_step_frac"]:
        vetoes.append("order_stitching_step")

    best = stats.get("fit_best", {})
    temps = best.get("temps_k", [])
    if temps and max(temps) > th["t_companion_k"]:
        vetoes.append("component_hotter_than_grain_survival")
    if np.isfinite(best.get("reduced_chi2", np.nan)) \
            and best["reduced_chi2"] > th["max_reduced_chi2"]:
        vetoes.append("no_acceptable_model")

    # --- shape flags ---
    beta = stats.get("beta", {})
    width = stats.get("width", {})
    geom = stats.get("geometric", {})
    cwidth = stats.get("component_width", {})
    n_comp = int(best.get("n_components", 0))
    beats_gradient = bool(stats.get("discrete_beats_gradient"))
    gradient_preferred = stats.get("preferred_model") == "gradient"

    # Read beta off the model the data SELECT.  A continuous gradient preferred
    # means this is a disk and the single-component beta is biased to zero.
    bim = stats.get("beta_in_model", {})
    if gradient_preferred:
        flags.append("gradient_model_preferred")
        beta_for_verdict = float(beta.get("beta_gradient_model", np.nan))
        beta_hi = float(beta.get("beta_gradient_model", np.nan))
        planck_consistent = False
    elif n_comp >= 2 and bim:
        beta_for_verdict = float(bim.get("beta", np.nan))
        beta_hi = float(bim.get("beta_hi_1sig", np.nan))
        planck_consistent = bool(bim.get("planck_consistent"))
    else:
        beta_for_verdict = float(beta.get("beta", np.nan))
        beta_hi = float(beta.get("beta_hi_1sig", np.nan))
        planck_consistent = bool(beta.get("planck_consistent"))

    beta_measured = bool(beta.get("leverage_beta_constrained"))
    beta_ok = bool(np.isfinite(beta_for_verdict)
                   and abs(beta_for_verdict) <= th["beta_planck_max"]
                   and planck_consistent
                   and np.isfinite(beta_hi) and beta_hi <= th["beta_planck_max"])
    if beta_ok and beta_measured:
        flags.append("beta_planck_consistent")
    if beta.get("dusty_beta") or (np.isfinite(beta_for_verdict)
                                  and beta_for_verdict >= 0.5):
        flags.append("beta_dustlike")
    if not beta_measured:
        flags.append("beta_unconstrained_by_coverage")
    if beta.get("model_dependent"):
        flags.append("beta_model_dependent")

    # The NOMINAL floor (dr/r = 0.2, i.e. dT/T = 0.1) is the discriminating
    # threshold: it is already narrower than nearly every resolved debris ring,
    # and the grain-size spread alone contributes ~0.1.  The ABSOLUTE floor
    # (dr/r = 0.05) is a second, high-confidence tier.
    narrow = bool(width.get("narrower_than_nominal_floor"))
    narrow_hard = bool(width.get("narrower_than_absolute_floor"))
    if narrow_hard:
        flags.append("width_below_absolute_natural_floor")
    elif narrow:
        flags.append("width_below_nominal_natural_floor")

    comps_narrow = bool(cwidth.get("components_narrower_than_nominal_floor"))
    if comps_narrow:
        flags.append("components_individually_narrow")
    if n_comp >= 3:
        flags.append(f"n_components_{n_comp}")
    if geom.get("is_geometric"):
        flags.append("geometric_progression")
    if beats_gradient:
        flags.append("discrete_beats_gradient")

    no_features = not any(v.startswith("feature_") for v in vetoes)
    # S6 needs ALL of: >= 3 components, geometric spacing, and the discrete
    # model beating the continuous-gradient null.  "Multi-component" alone is
    # NOT anomalous: two-temperature debris disks (warm belt + cold belt) are
    # common and natural.
    #
    # Per-component narrowness is a HIGH-CONFIDENCE TIER, not a gate, and the
    # reason is a hard property of the 5-38 micron band, not a modelling
    # choice.  Three Wien peaks all inside 5-38 micron force a temperature
    # ratio <= 2.8, and at ratio ~2 adjacent Planck functions overlap so
    # heavily that opening each component to dT/T ~ 0.3 costs almost no chi2.
    # Requiring narrowness there would reject every cascade the band can
    # actually contain.  Extending the baseline with far-IR photometry
    # (AKARI FIS, IRAS 60/100) is what re-enables the tier.
    cascade = bool(n_comp >= 3 and geom.get("is_geometric") and beats_gradient)
    # S5 needs a single component whose temperature width is bounded below the
    # natural floor.  It does NOT require "a single blackbody fits well" — that
    # is true of hundreds of ordinary debris disks.
    isothermal = bool(n_comp == 1 and narrow and not gradient_preferred)

    # --- verdict ladder ---
    if vetoes:
        verdict = "REJECTED_NATURAL"
    elif cascade and no_features and beta_ok and beta_measured:
        verdict = "S6_MATRIOSHKA_CASCADE_REVIEW"
    elif isothermal and no_features and beta_ok and beta_measured:
        verdict = "S5_ISOTHERMAL_REVIEW"
    elif (cascade or isothermal) and no_features and beta_ok and not beta_measured:
        verdict = "shape_anomalous_but_beta_unconstrained"
    else:
        verdict = "no_shape_anomaly"

    tier = "none"
    if verdict.startswith(("S5", "S6")):
        tier = "high_confidence" if (narrow_hard or comps_narrow) else "candidate"
    stats["tier"] = tier
    return ShapeVerdict(verdict=verdict, flags=flags, vetoes=vetoes, stats=stats)


def analyse_spectrum(lam_um, flux, err, thresholds: dict | None = None,
                     lam0_um: float = LAM0_UM) -> ShapeVerdict:
    """Convenience: compute every statistic and classify in one call."""
    return classify(compute_shape_stats(lam_um, flux, err, thresholds, lam0_um),
                    thresholds)


__all__ = [
    "DEFAULT_THRESHOLDS", "DUST_VETO_FEATURES", "FEATURES",
    "GRAIN_SIZE_DT_OVER_T", "NATURAL_DR_OVER_R_ABSOLUTE",
    "NATURAL_DR_OVER_R_NOMINAL", "ShapeVerdict", "T_COMPANION_K",
    "analyse_spectrum", "classify", "component_coverage", "component_width_limit",
    "compute_shape_stats", "energy_ladder", "estimate_beta",
    "extragalactic_interloper", "feature_equivalent_width",
    "geometric_progression_test", "measure_features", "natural_floor_dt_over_t",
    "order_step", "profile_scan", "screen_spectrum", "temperature_width",
    "wavelength_leverage",
]

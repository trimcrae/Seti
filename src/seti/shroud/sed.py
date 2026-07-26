"""SHROUD SED physics: photometric system, competing models, the energy budget.

Everything here is a pure function of arrays and dictionaries so the whole
physics layer is exercised offline by ``tests/test_shroud.py``.

The central measurement
-----------------------
An object detected on a POSS-I plate and absent from every modern optical survey
has lost a bolometric flux

    dF_bol = F_bol(POSS-I epoch) - F_bol(modern optical)

If it was *enshrouded* rather than destroyed, that energy was absorbed by
circumstellar material and re-radiated in the thermal infrared, so the observed
infrared flux must satisfy

    eta = F_IR(now) / dF_bol  ~  1

**eta is a pure flux ratio, so the distance cancels exactly.**  That is what
makes the test usable on a catalogue with no parallaxes: nothing here needs to
know how far away the object is.

Three regimes, all informative:

* ``eta ~ 1``            energy-conserving obscuration — enshrouded, not destroyed.
* ``eta << 1``           the infrared is far too faint to be the missing optical
                         light.  The object did not simply get obscured.
* ``eta >> 1``           the object is intrinsically infrared-luminous (dusty
                         AGB, YSO, galaxy) or the plate "source" and the IR
                         source are unrelated.

Because a vanished source usually has only *one* historical band, ``F_bol(then)``
is not uniquely determined.  Two statistics are therefore reported:

``eta_range``   the interval spanned by marginalising over the allowed
                progenitor temperature grid, and
``eta_max``     the value obtained with the temperature that *minimises*
                ``F_bol(then)``, i.e. the most generous possible value.
                **If ``eta_max`` < 1 the infrared cannot account for the missing
                optical light for any progenitor temperature at all** — an
                assumption-free rejection of simple obscuration.

The Forés-Toribio & Kochanek (2026, arXiv:2604.05019) progenitor-to-remnant
luminosity discriminant is the *same measurement read at different thresholds*:
merger remnants sit at 10-100x their progenitor luminosity, genuine
disappearance remnants at ~0.1x.  ``vet.py`` applies those cuts to ``eta``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- physical constants (SI) ------------------------------------------------
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23

# --- photometric system -----------------------------------------------------
# (lambda_eff [um], zero-point flux density F0 [Jy], A_lambda/A_V, system)
#
# Sources:
#   POSS-I O/E   photographic; treated as Johnson B / Cousins R equivalents
#                (Monet+2003 USNO-B1.0 calibration is onto those systems).
#   Gaia EDR3    Riello et al. 2021 Vega zero points.
#   Pan-STARRS   AB system (F0 = 3631 Jy in every band).
#   2MASS        Cohen, Wheaton & Megeath 2003.
#   WISE         Wright et al. 2010 / Jarrett et al. 2011.
#   A_lambda/A_V Wang & Chen 2019 (R_V = 3.1) for Gaia/2MASS/WISE; Cardelli+1989
#                for the photographic and Pan-STARRS bands.
BANDS: dict[str, tuple[float, float, float, str]] = {
    # historical photographic (POSS-I)
    "poss1_o":  (0.4250, 4000.0, 1.324, "vega"),
    "poss1_e":  (0.6450, 3064.0, 0.748, "vega"),
    # modern optical
    "gaia_bp":  (0.5110, 3552.01, 1.002, "vega"),
    "gaia_g":   (0.6218, 3228.75, 0.789, "vega"),
    "gaia_rp":  (0.7769, 2554.95, 0.589, "vega"),
    "ps1_g":    (0.4810, 3631.0, 1.023, "ab"),
    "ps1_r":    (0.6170, 3631.0, 0.733, "ab"),
    "ps1_i":    (0.7520, 3631.0, 0.543, "ab"),
    "ps1_z":    (0.8660, 3631.0, 0.426, "ab"),
    "ps1_y":    (0.9620, 3631.0, 0.351, "ab"),
    # near-infrared
    "2mass_j":  (1.2350, 1594.0, 0.243, "vega"),
    "2mass_h":  (1.6620, 1024.0, 0.131, "vega"),
    "2mass_ks": (2.1590, 666.7, 0.078, "vega"),
    # mid-infrared
    "w1":       (3.3526, 309.540, 0.039, "vega"),
    "w2":       (4.6028, 171.787, 0.026, "vega"),
    "w3":       (11.5608, 31.674, 0.040, "vega"),
    "w4":       (22.0883, 8.363, 0.020, "vega"),
}

HISTORICAL_BANDS = ("poss1_o", "poss1_e")
OPTICAL_BANDS = ("gaia_bp", "gaia_g", "gaia_rp",
                 "ps1_g", "ps1_r", "ps1_i", "ps1_z", "ps1_y")
IR_BANDS = ("2mass_j", "2mass_h", "2mass_ks", "w1", "w2", "w3", "w4")
MIDIR_BANDS = ("w1", "w2", "w3", "w4")

JY = 1.0e-26  # 1 Jy in W m^-2 Hz^-1


def lambda_eff_um(band: str) -> float:
    return BANDS[band][0]


def nu_hz(band: str) -> float:
    """Effective frequency of a band, Hz."""
    return _C / (BANDS[band][0] * 1e-6)


def mag_to_fnu(band: str, mag: float) -> float:
    """Magnitude -> flux density in W m^-2 Hz^-1 (AB or Vega as appropriate)."""
    return BANDS[band][1] * JY * 10.0 ** (-0.4 * float(mag))


def fnu_to_mag(band: str, fnu: float) -> float:
    if fnu <= 0:
        return float("inf")
    return -2.5 * math.log10(fnu / (BANDS[band][1] * JY))


def magerr_to_fnuerr(band: str, mag: float, magerr: float) -> float:
    """Symmetrised flux error from a magnitude error (dF/F = 0.921 * dmag)."""
    return mag_to_fnu(band, mag) * 0.9210340372 * float(magerr)


def extinction_factor(band: str, a_v: float) -> float:
    """Multiplicative attenuation 10^(-0.4 A_lambda) for a given A_V."""
    return 10.0 ** (-0.4 * BANDS[band][2] * float(a_v))


def planck_fnu(t_k: float, lam_um: float) -> float:
    """Planck B_nu(T) in W m^-2 Hz^-1 sr^-1 (shape only; scale is fitted)."""
    if t_k <= 0:
        return 0.0
    lam = lam_um * 1e-6
    nu = _C / lam
    x = _H * nu / (_KB * t_k)
    if x > 500.0:                      # Wien tail underflow guard
        return 0.0
    return (2.0 * _H * nu ** 3 / _C ** 2) / math.expm1(x)


def _planck_fnu_arr(t_k: float, lam_um: np.ndarray) -> np.ndarray:
    return np.array([planck_fnu(t_k, float(x)) for x in np.atleast_1d(lam_um)])


def blackbody_bolometric_from_band(t_k: float, band: str, fnu_obs: float) -> float:
    """Bolometric flux of a blackbody normalised to reproduce ``fnu_obs``.

    For a blackbody the ratio F_bol / F_nu(lambda) is a pure function of T, so
    a single band plus an assumed temperature fixes the bolometric flux.  With
    ``F_bol = sigma T^4 * Omega / pi`` and ``F_nu = B_nu(T) * Omega``:

        F_bol = F_nu_obs * (sigma T^4 / pi) / B_nu(T, lambda)
    """
    b = planck_fnu(t_k, BANDS[band][0])
    if b <= 0.0:
        return float("inf")
    sigma = 5.670374419e-8
    return float(fnu_obs) * (sigma * t_k ** 4 / math.pi) / b


def bolometric_correction_factor(t_k: float, band: str) -> float:
    """F_bol / F_nu(band) for a blackbody of temperature ``t_k`` (units Hz)."""
    b = planck_fnu(t_k, BANDS[band][0])
    if b <= 0.0:
        return float("inf")
    sigma = 5.670374419e-8
    return (sigma * t_k ** 4 / math.pi) / b


def temperature_minimising_fbol(band: str, teff_grid) -> float:
    """The grid temperature giving the *smallest* F_bol for a fixed band flux.

    This is the progenitor temperature most favourable to the obscuration
    hypothesis: it demands the least missing energy, hence the largest eta.
    For a blackbody it is the temperature whose ``nu B_nu`` peaks in the band.
    """
    grid = list(teff_grid)
    return min(grid, key=lambda t: bolometric_correction_factor(float(t), band))


# --- SED container ----------------------------------------------------------
@dataclass
class SED:
    """Photometry for one object.

    ``mags``/``errs``  detected bands (magnitudes in each band's native system)
    ``limits``         non-detections as 1-sigma-equivalent magnitude limits;
                       a model is penalised only if it exceeds the limit.
    """

    source_id: str = ""
    mags: dict[str, float] = field(default_factory=dict)
    errs: dict[str, float] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # -- derived views ------------------------------------------------------
    def detected(self, group=None) -> list[str]:
        keys = [b for b in self.mags if b in BANDS and np.isfinite(self.mags[b])]
        if group is not None:
            keys = [b for b in keys if b in group]
        return sorted(keys, key=lambda b: BANDS[b][0])

    def detected_modern(self) -> list[str]:
        """Detected bands *excluding* the 1949-1958 photographic points.

        The plate magnitude belongs to a different epoch by ~60 years and must
        never enter a fit of the object's present-day SED; it enters only the
        energy budget, as the "then" side of the comparison.
        """
        return [b for b in self.detected() if b not in HISTORICAL_BANDS]

    def fnu(self, band: str) -> float:
        return mag_to_fnu(band, self.mags[band])

    def sigma_fnu(self, band: str, floor_phot: float = 0.30,
                  floor_modern: float = 0.05) -> float:
        floor = floor_phot if band in HISTORICAL_BANDS else floor_modern
        e = float(self.errs.get(band, 0.0) or 0.0)
        return magerr_to_fnuerr(band, self.mags[band], math.hypot(e, floor))

    def has_ir(self, min_bands: int = 2) -> bool:
        return len(self.detected(IR_BANDS)) >= min_bands


# --- model evaluation -------------------------------------------------------
def photosphere_fnu(bands, teff: float, scale: float, a_v: float) -> np.ndarray:
    """Reddened single-temperature photosphere, F_nu per band."""
    lam = np.array([BANDS[b][0] for b in bands])
    b = _planck_fnu_arr(teff, lam)
    ext = np.array([extinction_factor(b_, a_v) for b_ in bands])
    return scale * b * ext


def obscured_plus_dust_fnu(bands, teff: float, scale_star: float, a_v: float,
                           t_dust: float, scale_dust: float,
                           a_v_ism: float = 0.0) -> np.ndarray:
    """Obscured photosphere + warm-dust blackbody.

    ``a_v`` is the *total* column seen by the star (circumstellar + ISM);
    ``a_v_ism`` is the foreground-only column, which is all the dust emission
    itself suffers.
    """
    lam = np.array([BANDS[b][0] for b in bands])
    star = scale_star * _planck_fnu_arr(teff, lam) * np.array(
        [extinction_factor(b_, a_v) for b_ in bands])
    dust = scale_dust * _planck_fnu_arr(t_dust, lam) * np.array(
        [extinction_factor(b_, a_v_ism) for b_ in bands])
    return star + dust


def _chi2(model: np.ndarray, obs: np.ndarray, sig: np.ndarray) -> float:
    return float(np.sum(((obs - model) / sig) ** 2))


def _limit_penalty(sed: SED, bands_lim, model_fn, huge: float = 1e3) -> float:
    """Penalty for a model that violates a non-detection.

    A non-detection at magnitude ``m_lim`` means F_nu < F(m_lim).  Models below
    the limit are unpenalised; models above it are penalised quadratically in
    units of the limit flux.
    """
    if not bands_lim:
        return 0.0
    lim_f = np.array([mag_to_fnu(b, sed.limits[b]) for b in bands_lim])
    mod = model_fn(bands_lim)
    over = np.clip((mod - lim_f) / np.maximum(lim_f, 1e-300), 0.0, None)
    return float(np.sum(np.minimum(over ** 2, huge)))


@dataclass
class FitResult:
    model: str
    teff_k: float
    a_v_mag: float
    t_dust_k: float
    scale_star: float
    scale_dust: float
    chi2: float
    n_bands: int
    dof: int
    ok: bool = True
    note: str = ""

    @property
    def chi2_red(self) -> float:
        return self.chi2 / max(self.dof, 1)


def _linear_scale(model_shape: np.ndarray, obs: np.ndarray,
                  sig: np.ndarray) -> float:
    """Closed-form least-squares amplitude for a single-component model."""
    w = 1.0 / sig ** 2
    denom = float(np.sum(w * model_shape ** 2))
    if denom <= 0:
        return 0.0
    return max(float(np.sum(w * model_shape * obs)) / denom, 0.0)


def _linear_scales_two(shape_a: np.ndarray, shape_b: np.ndarray,
                       obs: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    """Non-negative least squares for a two-component linear model."""
    w = 1.0 / sig ** 2
    aa = float(np.sum(w * shape_a * shape_a))
    bb = float(np.sum(w * shape_b * shape_b))
    ab = float(np.sum(w * shape_a * shape_b))
    ay = float(np.sum(w * shape_a * obs))
    by = float(np.sum(w * shape_b * obs))
    det = aa * bb - ab * ab
    if det > 0:
        sa = (bb * ay - ab * by) / det
        sb = (aa * by - ab * ay) / det
        if sa >= 0 and sb >= 0:
            return float(sa), float(sb)
    # Boundary solutions (one component switched off).
    cand = []
    if aa > 0:
        s = max(ay / aa, 0.0)
        cand.append((_chi2(s * shape_a, obs, sig), s, 0.0))
    if bb > 0:
        s = max(by / bb, 0.0)
        cand.append((_chi2(s * shape_b, obs, sig), 0.0, s))
    if not cand:
        return 0.0, 0.0
    _, sa, sb = min(cand)
    return float(sa), float(sb)


def fit_photosphere(sed: SED, teff_grid, av_grid,
                    floor_phot: float = 0.30,
                    floor_modern: float = 0.05,
                    bands: list[str] | None = None) -> FitResult:
    """Model (a): a single reddened stellar photosphere.

    Fitted to the *present-day* bands only (``SED.detected_modern``) unless an
    explicit band list is given — the POSS-I point is 60+ years older and
    belongs to the budget, not to the current SED.
    """
    bands = list(bands) if bands is not None else sed.detected_modern()
    if len(bands) < 2:
        return FitResult("photosphere", float("nan"), float("nan"), float("nan"),
                         0.0, 0.0, float("inf"), len(bands), 1, ok=False,
                         note="fewer than 2 detected bands")
    obs = np.array([sed.fnu(b) for b in bands])
    sig = np.array([sed.sigma_fnu(b, floor_phot, floor_modern) for b in bands])
    lim_bands = [b for b in sed.limits if b in BANDS and b not in bands]

    best = None
    for teff in teff_grid:
        for a_v in av_grid:
            shape = photosphere_fnu(bands, float(teff), 1.0, float(a_v))
            if not np.any(shape > 0):
                continue
            s = _linear_scale(shape, obs, sig)
            c = _chi2(s * shape, obs, sig)
            c += _limit_penalty(
                sed, lim_bands,
                lambda bb, t=teff, a=a_v, s_=s: photosphere_fnu(bb, float(t), s_, float(a)))
            if best is None or c < best[0]:
                best = (c, float(teff), float(a_v), s)
    if best is None:
        return FitResult("photosphere", float("nan"), float("nan"), float("nan"),
                         0.0, 0.0, float("inf"), len(bands), 1, ok=False,
                         note="no viable grid point")
    c, teff, a_v, s = best
    return FitResult("photosphere", teff, a_v, float("nan"), s, 0.0, c,
                     len(bands), max(len(bands) - 3, 1))


def fit_obscured_dust(sed: SED, teff_grid, av_grid, tdust_grid,
                      a_v_ism: float = 0.0,
                      floor_phot: float = 0.30,
                      floor_modern: float = 0.05,
                      bands: list[str] | None = None) -> FitResult:
    """Model (b): an obscured photosphere plus a warm-dust blackbody.

    Fitted to the present-day bands only (see ``fit_photosphere``).
    """
    bands = list(bands) if bands is not None else sed.detected_modern()
    if len(bands) < 3:
        return FitResult("obscured_dust", float("nan"), float("nan"), float("nan"),
                         0.0, 0.0, float("inf"), len(bands), 1, ok=False,
                         note="fewer than 3 detected bands")
    obs = np.array([sed.fnu(b) for b in bands])
    sig = np.array([sed.sigma_fnu(b, floor_phot, floor_modern) for b in bands])
    lim_bands = [b for b in sed.limits if b in BANDS and b not in bands]

    best = None
    for teff in teff_grid:
        for a_v in av_grid:
            shape_s = photosphere_fnu(bands, float(teff), 1.0, float(a_v))
            for t_d in tdust_grid:
                if float(t_d) >= float(teff):
                    continue          # dust cannot outshine-in-temperature the star
                shape_d = photosphere_fnu(bands, float(t_d), 1.0, float(a_v_ism))
                if not (np.any(shape_s > 0) or np.any(shape_d > 0)):
                    continue
                ss, sd = _linear_scales_two(shape_s, shape_d, obs, sig)
                c = _chi2(ss * shape_s + sd * shape_d, obs, sig)
                c += _limit_penalty(
                    sed, lim_bands,
                    lambda bb, t=teff, a=a_v, td=t_d, s1=ss, s2=sd:
                    obscured_plus_dust_fnu(bb, float(t), s1, float(a), float(td),
                                           s2, a_v_ism))
                if best is None or c < best[0]:
                    best = (c, float(teff), float(a_v), float(t_d), ss, sd)
    if best is None:
        return FitResult("obscured_dust", float("nan"), float("nan"), float("nan"),
                         0.0, 0.0, float("inf"), len(bands), 1, ok=False,
                         note="no viable grid point")
    c, teff, a_v, t_d, ss, sd = best
    return FitResult("obscured_dust", teff, a_v, t_d, ss, sd, c,
                     len(bands), max(len(bands) - 5, 1))


# --- the energy budget ------------------------------------------------------
@dataclass
class EnergyBudget:
    verdict: str
    eta_max: float                 # most generous eta (T minimising F_bol_then)
    eta_lo: float                  # over the progenitor temperature grid
    eta_hi: float
    eta_trapz_max: float           # model-free IR integral (a lower bound on F_IR)
    f_ir_model: float
    f_ir_trapz: float
    f_bol_then_min: float
    f_bol_then_lo: float
    f_bol_then_hi: float
    f_bol_now_optical: float
    t_dust_k: float
    n_ir_bands: int
    note: str = ""

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k, v in d.items():
            if isinstance(v, float) and not np.isfinite(v):
                d[k] = None
        return d


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed it
    return float(fn(y, x))


def integrate_ir_trapz(sed: SED) -> tuple[float, int]:
    """Model-free ``int F_nu dnu`` over the detected infrared bands.

    Trapezoidal in frequency across the detected points only — no extrapolation
    beyond the observed range, so this is a strict *lower bound* on the true
    infrared flux and therefore a conservative eta.
    """
    bands = sed.detected(IR_BANDS)
    if len(bands) < 2:
        return 0.0, len(bands)
    nu = np.array([nu_hz(b) for b in bands])
    f = np.array([sed.fnu(b) for b in bands])
    order = np.argsort(nu)
    return _trapz(f[order], nu[order]), len(bands)


def integrate_blackbody_flux(t_k: float, scale: float) -> float:
    """Bolometric flux of a blackbody component with fitted solid-angle scale.

    ``scale`` multiplies ``B_nu`` (i.e. it *is* the solid angle in sr), so
    ``F_bol = scale * sigma T^4 / pi``.
    """
    if not np.isfinite(t_k) or t_k <= 0 or scale <= 0:
        return 0.0
    sigma = 5.670374419e-8
    return float(scale) * sigma * float(t_k) ** 4 / math.pi


def energy_budget(sed: SED, cfg: dict, fit_dust: FitResult | None = None) -> EnergyBudget:
    """The obscuration-vs-destruction test.

    Returns eta = F_IR(now) / [F_bol(then) - F_bol(now, optical)] together with
    the verdict.  Distance cancels; nothing here needs a parallax.
    """
    sed_cfg = cfg.get("sed", {})
    eb_cfg = cfg.get("energy_budget", {})
    teff_grid = sed_cfg.get("teff_star_grid_k", [3000, 4000, 5800, 9000])
    min_ir = int(sed_cfg.get("min_ir_bands", 2))

    hist = sed.detected(HISTORICAL_BANDS)
    ir = sed.detected(IR_BANDS)
    nan = float("nan")
    if not hist:
        return EnergyBudget("NO_HISTORICAL_PHOTOMETRY", nan, nan, nan, nan,
                            nan, nan, nan, nan, nan, nan, nan, len(ir),
                            note="no POSS-I magnitude: the budget is undefined")
    if len(ir) < min_ir:
        return EnergyBudget("INSUFFICIENT_IR", nan, nan, nan, nan,
                            nan, nan, nan, nan, nan, nan, nan, len(ir),
                            note=f"{len(ir)} IR band(s) < {min_ir} required")

    # --- F_bol at the POSS-I epoch, marginalised over progenitor temperature.
    # Use the reddest available historical band (least sensitive to extinction).
    hband = hist[-1]
    f_hist = sed.fnu(hband)
    bcs = {float(t): bolometric_correction_factor(float(t), hband) for t in teff_grid}
    bcs = {t: v for t, v in bcs.items() if np.isfinite(v)}
    if not bcs:
        return EnergyBudget("NO_VIABLE_PROGENITOR", nan, nan, nan, nan,
                            nan, nan, nan, nan, nan, nan, nan, len(ir),
                            note="bolometric correction diverged on the whole grid")
    f_then_vals = {t: f_hist * v for t, v in bcs.items()}
    f_then_min = min(f_then_vals.values())
    f_then_lo = np.percentile(list(f_then_vals.values()), 0)
    f_then_hi = np.percentile(list(f_then_vals.values()), 100)

    # --- modern optical flux (detections; limits contribute nothing).
    opt = sed.detected(OPTICAL_BANDS)
    if opt:
        oband = opt[len(opt) // 2]
        # assume the same progenitor temperature that minimises F_bol(then)
        t_min = min(f_then_vals, key=f_then_vals.get)
        f_now_opt = sed.fnu(oband) * bolometric_correction_factor(t_min, oband)
    else:
        f_now_opt = 0.0

    # --- observed infrared flux.
    f_ir_trapz, n_ir = integrate_ir_trapz(sed)
    t_dust = nan
    f_ir_model = f_ir_trapz
    if fit_dust is not None and fit_dust.ok and np.isfinite(fit_dust.t_dust_k):
        f_dust = integrate_blackbody_flux(fit_dust.t_dust_k, fit_dust.scale_dust)
        f_star_att = integrate_blackbody_flux(fit_dust.teff_k, fit_dust.scale_star)
        # The dust component is the re-radiated energy; the attenuated star is
        # what still leaks through.  Only the dust term can pay the optical debt.
        if f_dust > 0:
            f_ir_model = f_dust
            t_dust = fit_dust.t_dust_k
        del f_star_att

    denom_min = f_then_min - f_now_opt
    denom_lo = f_then_lo - f_now_opt
    denom_hi = f_then_hi - f_now_opt
    if denom_min <= 0:
        return EnergyBudget("NO_DEFICIT", nan, nan, nan, nan, f_ir_model,
                            f_ir_trapz, f_then_min, f_then_lo, f_then_hi,
                            f_now_opt, t_dust, n_ir,
                            note="modern optical flux is not below the POSS-I flux")

    eta_max = f_ir_model / denom_min
    eta_hi = f_ir_model / max(denom_lo, 1e-300)
    eta_lo = f_ir_model / max(denom_hi, 1e-300)
    eta_trapz_max = f_ir_trapz / denom_min
    lo, hi = sorted((eta_lo, eta_hi))

    cons_lo = float(eb_cfg.get("eta_conserving_lo", 0.30))
    cons_hi = float(eb_cfg.get("eta_conserving_hi", 3.0))
    too_faint = float(eb_cfg.get("eta_too_faint", 0.10))

    if eta_max < too_faint:
        verdict = "IR_TOO_FAINT"
        note = ("even the progenitor temperature that minimises the missing "
                "bolometric flux leaves the infrared short by "
                f"{1.0 / max(eta_max, 1e-300):.0f}x: not simple obscuration")
    elif eta_max < cons_lo:
        verdict = "IR_TOO_FAINT_MARGINAL"
        note = "infrared short of the missing optical for every progenitor T"
    elif lo > cons_hi:
        verdict = "IR_EXCEEDS_MISSING"
        note = ("infrared exceeds the missing optical for every progenitor T: "
                "intrinsically IR-luminous source or an unrelated IR match")
    elif hi >= cons_lo and lo <= cons_hi:
        verdict = "ENERGY_CONSERVING_OBSCURATION"
        note = ("the infrared accounts for the missing optical luminosity: "
                "enshrouded, energy conserved")
    else:
        verdict = "INDETERMINATE"
        note = "eta range does not cleanly select a regime"

    return EnergyBudget(verdict, float(eta_max), float(lo), float(hi),
                        float(eta_trapz_max), float(f_ir_model), float(f_ir_trapz),
                        float(f_then_min), float(f_then_lo), float(f_then_hi),
                        float(f_now_opt), float(t_dust), int(n_ir), note)


def luminosity_ratio(budget: EnergyBudget) -> float:
    """Remnant / progenitor bolometric ratio (Forés-Toribio & Kochanek 2026).

    With the modern optical negligible, the present-day bolometric output is
    dominated by the reprocessed infrared, so the F-T&K progenitor-to-remnant
    ratio *is* eta.  Returned separately so ``vet.py`` can apply their
    thresholds without re-deriving the identity.
    """
    return budget.eta_max


def fit_both(sed: SED, cfg: dict) -> tuple[FitResult, FitResult]:
    s = cfg.get("sed", {})
    return (
        fit_photosphere(sed, s.get("teff_star_grid_k", [3000, 4000, 5800]),
                        s.get("av_grid_mag", [0.0, 1.0, 5.0]),
                        s.get("sys_floor_mag_photographic", 0.30),
                        s.get("sys_floor_mag_modern", 0.05)),
        fit_obscured_dust(sed, s.get("teff_star_grid_k", [3000, 4000, 5800]),
                          s.get("av_grid_mag", [0.0, 1.0, 5.0]),
                          s.get("tdust_grid_k", [200, 500, 1000]),
                          0.0,
                          s.get("sys_floor_mag_photographic", 0.30),
                          s.get("sys_floor_mag_modern", 0.05)),
    )


__all__ = [
    "BANDS", "HISTORICAL_BANDS", "IR_BANDS", "MIDIR_BANDS", "OPTICAL_BANDS",
    "SED", "EnergyBudget", "FitResult",
    "blackbody_bolometric_from_band", "bolometric_correction_factor",
    "energy_budget", "extinction_factor", "fit_both", "fit_obscured_dust",
    "fit_photosphere", "fnu_to_mag", "integrate_blackbody_flux",
    "integrate_ir_trapz", "lambda_eff_um", "luminosity_ratio", "mag_to_fnu",
    "magerr_to_fnuerr", "nu_hz", "obscured_plus_dust_fnu", "photosphere_fnu",
    "planck_fnu", "temperature_minimising_fbol",
]

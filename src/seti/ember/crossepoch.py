"""The cross-epoch excess-difference statistic.

EMBER asks whether a mid-infrared excess that a survey saw at one epoch is
*gone* at a later one. The whole difficulty is that "gone" has half a dozen
instrumental explanations that are individually more likely than the
astrophysical one, so this module is built around four ideas:

**1. The photosphere is empirical, not modelled.**
A blackbody at ``T_eff`` is *not* a stellar atmosphere: extrapolating a
2MASS Ks flux to 12 micron with a Planck function over-predicts the mid-IR by
~0.3 mag at 5000 K (H- opacity means the Ks continuum forms deeper and hotter
than a blackbody implies). Instead the photospheric colour ``Ks - band`` is
fitted **from the sample itself** as a robust function of an optical colour.
This absorbs, in one step and exactly:

* stellar-atmosphere error,
* each survey's absolute calibration and zero-point scale,
* the response-curve error in this module's band model.

That last point matters more than it sounds. Liu (2020, arXiv:2008.12611)
re-examined Rhee et al.'s IRAS-detected debris hosts with WISE and attributed
the IRAS-WISE flux discrepancies to *calibration*. A per-band empirical locus
makes any such constant multiplicative offset unobservable in the differential
statistic, which is precisely the correct behaviour: a calibration offset moves
every star, and EMBER only believes stars that move against their own locus.

**2. The Eddington/Malmquist bias is one-directional and must be removed.**
The early epoch is flux-limited. Near its threshold only upward noise
excursions are catalogued, so the true flux is systematically *lower* than the
measured one; a deeper later survey then measures the truth and the pair
manufactures a fade. This bias points only one way -- toward false cessation --
so it cannot be dismissed as scatter. It is handled twice over: analytically,
by deboosting the early flux using the source-count slope measured from the
catalogue itself; and empirically, by the null calibration below.

**3. The statistic is two-sided, and the rising half is the control.**
Excess *appearance* is astrophysically as (im)plausible as disappearance and is
produced by every symmetric systematic in the same measure -- but **not** by
Eddington bias, which only fades. So the negative tail of the same statistic is
a matched, same-sample, same-S/N null for the positive tail, and the asymmetry
between them *is* the residual systematic budget. Thresholds are set on that
empirical distribution rather than on a Gaussian assumption.

**4. Three epochs, not two.**
IRAS (1983), AKARI (2006-07) and WISE (2010) are the only surveys carrying
12-25 micron information -- NEOWISE flies W1/W2 only and cannot see 100-300 K
dust at all. With three epochs the failure modes separate:

===================  ==============================================
IRAS / AKARI / WISE  interpretation
===================  ==============================================
high / high / low    real fade 2007-2010, or a WISE artefact
high / low  / low    real fade 1983-2006, or (more often) IRAS-side
                     blending or a spurious IRAS source
high / low  / high   incoherent -> systematic
all consistent       constant
===================  ==============================================

Two epochs cannot tell those apart; three can, and so the three-epoch
adjudication is the primary funnel rather than a follow-up.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .bands import BANDS, transfer, transfer_with_systematic

#: Fractional systematic floor on the photospheric prediction, added in
#: quadrature to the measurement error. Covers residual scatter of the empirical
#: colour locus about a single star (unresolved companions, metallicity,
#: activity). 0.05 = 5% ~ 0.054 mag.
PHOT_SYS_FRAC = 0.05

#: Prior range on the excess (dust / radiator) temperature when it cannot be
#: fitted from two same-epoch bands. 150 K is set by the longest usable
#: wavelength (25 micron); 1500 K by silicate sublimation.
T_DUST_PRIOR_K = (150.0, 1500.0)
T_DUST_DEFAULT_K = 400.0


# --------------------------------------------------------------------------
# 1. The empirical photospheric locus
# --------------------------------------------------------------------------
@dataclass
class PhotosphereLocus:
    """Robust empirical relation ``(Ks - band)_0`` as a function of a colour.

    Fitted by binned **low quantiles** rather than least squares or even
    medians. The contaminating population -- stars with a genuine excess -- is
    strictly one-sided, since an excess can only make the band brighter. A mean
    is destroyed by it; a median survives only while the contaminated fraction
    stays under 50%, which is not guaranteed in an infrared-selected catalogue,
    where excess sources are exactly what got selected. Anchoring on the 25th
    percentile and reconstructing the ridge centre under a Gaussian core pushes
    the breakdown point to ~75% contamination.
    """

    band: str
    colour_name: str
    knots: np.ndarray  # colour values at bin centres
    values: np.ndarray  # median (Ks - band) in each bin
    scatter: np.ndarray  # 1.4826 * MAD in each bin
    n_per_bin: np.ndarray
    n_calib: int = 0
    fallback_scatter: float = 0.05
    degraded: bool = False
    note: str = ""

    def predict(self, colour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Ks - band)_0`` and its scatter at the given colours."""
        col = np.atleast_1d(np.asarray(colour, dtype=float))
        if self.knots.size == 0:
            return (np.full(col.shape, np.nan), np.full(col.shape, self.fallback_scatter))
        val = np.interp(col, self.knots, self.values,
                        left=self.values[0], right=self.values[-1])
        sca = np.interp(col, self.knots, self.scatter,
                        left=self.scatter[0], right=self.scatter[-1])
        return val, np.maximum(sca, 1e-3)


#: Quantile anchoring the photospheric ridge. Excess sources contaminate only
#: the upper tail, so a low quantile is uncontaminated until they dominate.
LOCUS_QUANTILE = 0.25
#: Gaussian quantile constants for reconstructing the ridge centre and width
#: from two *lower* quantiles (0.10 and 0.25), neither of which a one-sided
#: positive tail can reach until it exceeds ~75% of the sample.
_Z25 = -0.6744897501960817  # Phi^-1(0.25)
_Z10 = -1.2815515655446004  # Phi^-1(0.10)


def _ridge_from_low_quantiles(y: np.ndarray) -> tuple[float, float]:
    """Recover (centre, sigma) of a Gaussian core from its 10th/25th percentiles.

    Both anchors lie below the median, so a positive contaminating tail cannot
    move them until it makes up most of the sample. Falls back to the median and
    the MAD when the quantile spread is degenerate.
    """
    q10, q25 = np.percentile(y, [10.0, 25.0])
    spread = q25 - q10
    if not np.isfinite(spread) or spread <= 0:
        med = float(np.median(y))
        mad = float(1.4826 * np.median(np.abs(y - med)))
        return med, max(mad, 1e-3)
    sigma = spread / (_Z25 - _Z10)
    centre = q25 - _Z25 * sigma
    return float(centre), float(max(sigma, 1e-3))


def fit_photosphere_locus(colour: np.ndarray, ks_mag: np.ndarray, band_mag: np.ndarray,
                          band: str, colour_name: str = "bp_rp",
                          n_bins: int = 20, min_per_bin: int = 25,
                          clip_iters: int = 3) -> PhotosphereLocus:
    """Fit the bare-photosphere colour locus for one band.

    ``band_mag`` must be on the same magnitude system used downstream; for the
    Jy-native catalogues (IRAS, AKARI) pass ``-2.5*log10(F_Jy)`` with any fixed
    zero point -- the locus absorbs the constant, and with it every per-survey
    calibration offset.

    Within each colour bin the ridge is reconstructed from the 10th and 25th
    percentiles, which a one-sided excess population cannot reach. Iterative
    clipping is then deliberately **asymmetric** -- high outliers rejected at
    3 sigma, low outliers only at 5 sigma -- so the fit tracks the bare
    photosphere rather than being dragged up by the very population under study.
    """
    col = np.asarray(colour, dtype=float)
    y = np.asarray(ks_mag, dtype=float) - np.asarray(band_mag, dtype=float)
    good = np.isfinite(col) & np.isfinite(y)
    if good.sum() < max(min_per_bin, 10):
        return PhotosphereLocus(band=band, colour_name=colour_name,
                                knots=np.array([]), values=np.array([]),
                                scatter=np.array([]), n_per_bin=np.array([]),
                                n_calib=int(good.sum()), degraded=True,
                                note="too few stars to fit a locus")
    col, y = col[good], y[good]
    keep = np.ones(col.size, dtype=bool)

    knots = values = scatter = counts = np.array([])
    for _ in range(max(1, clip_iters)):
        edges = np.unique(np.quantile(col[keep], np.linspace(0, 1, n_bins + 1)))
        if edges.size < 3:
            break
        idx = np.clip(np.digitize(col, edges) - 1, 0, edges.size - 2)
        k, v, sg, c = [], [], [], []
        for b in range(edges.size - 1):
            sel = keep & (idx == b)
            if sel.sum() < min_per_bin:
                continue
            centre, sigma = _ridge_from_low_quantiles(y[sel])
            k.append(float(np.median(col[sel])))
            v.append(centre)
            sg.append(sigma)
            c.append(int(sel.sum()))
        if not k:
            break
        knots, values = np.asarray(k), np.asarray(v)
        scatter, counts = np.asarray(sg), np.asarray(c)
        pred = np.interp(col, knots, values, left=values[0], right=values[-1])
        sca = np.interp(col, knots, scatter, left=scatter[0], right=scatter[-1])
        resid = y - pred
        keep = (resid < 3.0 * sca) & (resid > -5.0 * sca)
        if keep.sum() < min_per_bin:
            break

    if knots.size == 0:
        return PhotosphereLocus(band=band, colour_name=colour_name,
                                knots=np.array([]), values=np.array([]),
                                scatter=np.array([]), n_per_bin=np.array([]),
                                n_calib=int(col.size), degraded=True,
                                note="binning failed; using fallback scatter")
    return PhotosphereLocus(band=band, colour_name=colour_name, knots=knots,
                            values=values, scatter=scatter, n_per_bin=counts,
                            n_calib=int(keep.sum()))


# --------------------------------------------------------------------------
# 2. Excess measurement
# --------------------------------------------------------------------------
@dataclass
class ExcessMeasurement:
    """An excess flux in one band at one epoch, with its full error budget."""

    band: str
    epoch_year: float
    f_obs_jy: float
    f_obs_err_jy: float
    f_phot_jy: float
    f_phot_err_jy: float
    f_exc_jy: float
    f_exc_err_jy: float
    chi: float
    saturated: bool = False
    upper_limit: bool = False
    eddington_shift_jy: float = 0.0

    @property
    def frac_excess(self) -> float:
        """``F_obs / F_phot - 1``; the dimensionless excess."""
        if self.f_phot_jy <= 0:
            return float("nan")
        return self.f_obs_jy / self.f_phot_jy - 1.0


def eddington_deboost(f_obs_jy: np.ndarray, sigma_jy: np.ndarray,
                      count_slope: float) -> np.ndarray:
    """Remove the flux-limited selection bias from an early-epoch flux.

    For differential source counts ``dN/dS ~ S**(-gamma)`` and Gaussian errors,
    the posterior mean of the true flux given a measurement is, to first order,

        S_true ~= S_obs - gamma * sigma**2 / S_obs

    Only upward noise excursions clear a detection threshold, so the measured
    flux of a near-threshold source is biased **high**. In a two-epoch
    comparison where the early survey is the shallow one, that bias points
    entirely toward a false fade -- it is the one contaminant that a two-sided
    null does not calibrate away, which is why it is corrected explicitly here
    *and* why the null-calibration in :func:`calibrate_null` measures what is
    left.

    ``count_slope`` should be measured from the catalogue in the relevant flux
    range by :func:`measure_count_slope`, not assumed.
    """
    f = np.atleast_1d(np.asarray(f_obs_jy, dtype=float))
    s = np.atleast_1d(np.asarray(sigma_jy, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = float(count_slope) * s**2 / f
    shift = np.where(np.isfinite(shift), shift, 0.0)
    # Never deboost by more than half the flux; below that the first-order
    # expansion is meaningless and the source should simply be cut on S/N.
    shift = np.minimum(shift, 0.5 * np.abs(f))
    return f - shift


def measure_count_slope(flux_jy: np.ndarray, lo: float, hi: float) -> float:
    """Fit ``gamma`` in ``dN/dS ~ S**(-gamma)`` over ``[lo, hi]`` from the data.

    Uses the cumulative counts, which are less noisy than a binned differential
    fit: ``N(>S) ~ S**(1-gamma)``, so the slope of ``log N(>S)`` against
    ``log S`` is ``1 - gamma``. Returns a Euclidean-ish default when the sample
    is too small to fit.
    """
    f = np.asarray(flux_jy, dtype=float)
    f = f[np.isfinite(f) & (f > 0)]
    sel = f[(f >= lo) & (f <= hi)]
    if sel.size < 50:
        return 2.5
    grid = np.geomspace(lo, hi, 12)
    n_gt = np.array([(f >= g).sum() for g in grid], dtype=float)
    ok = n_gt > 0
    if ok.sum() < 4:
        return 2.5
    slope = np.polyfit(np.log10(grid[ok]), np.log10(n_gt[ok]), 1)[0]
    gamma = 1.0 - slope
    return float(np.clip(gamma, 1.0, 4.0))


def measure_excess(band_key: str, f_obs_jy: float, f_obs_err_jy: float,
                   ks_jy: float, ks_err_jy: float, colour: float,
                   locus: PhotosphereLocus,
                   count_slope: float | None = None,
                   phot_sys_frac: float = PHOT_SYS_FRAC) -> ExcessMeasurement:
    """Measure the excess in one band against the empirical photospheric locus.

    ``count_slope`` triggers Eddington deboosting of ``f_obs_jy``; pass ``None``
    for a deep late-epoch band where the bias is negligible.
    """
    band = BANDS[band_key]
    colour_term, locus_scatter = locus.predict(np.array([colour]))
    # (Ks - band)_0 in magnitudes -> the photospheric FLUX ratio F_band / F_Ks.
    #
    # Sign matters and is easy to invert. With C = Ks - band in magnitudes,
    # m_band = m_Ks - C, so F_band / F_Ks = 10**(-0.4 * (m_band - m_Ks))
    #                                     = 10**(+0.4 * C).
    # Getting this backwards biases the predicted photosphere by 10**(0.8*C),
    # and because C differs from band to band it biases each epoch by a
    # *different* amount -- which manufactures apparent fades wholesale rather
    # than cancelling out.
    ratio = 10.0 ** (0.4 * float(colour_term[0]))
    f_phot = float(ks_jy) * ratio
    rel_locus = 0.4 * np.log(10.0) * float(locus_scatter[0])
    rel_ks = abs(ks_err_jy / ks_jy) if ks_jy else 0.0
    f_phot_err = abs(f_phot) * float(np.hypot(np.hypot(rel_locus, rel_ks), phot_sys_frac))

    f_used, shift = float(f_obs_jy), 0.0
    if count_slope is not None:
        f_deb = float(eddington_deboost(np.array([f_obs_jy]),
                                        np.array([f_obs_err_jy]), count_slope)[0])
        shift, f_used = float(f_obs_jy) - f_deb, f_deb

    f_exc = f_used - f_phot
    # The deboost correction is itself uncertain; carry half of it as an error.
    f_exc_err = float(np.sqrt(f_obs_err_jy**2 + f_phot_err**2 + (0.5 * shift) ** 2))
    chi = f_exc / f_exc_err if f_exc_err > 0 else float("nan")
    sat = band.sat_jy is not None and f_obs_jy >= band.sat_jy
    return ExcessMeasurement(band=band_key, epoch_year=band.epoch_year,
                             f_obs_jy=float(f_obs_jy), f_obs_err_jy=float(f_obs_err_jy),
                             f_phot_jy=f_phot, f_phot_err_jy=f_phot_err,
                             f_exc_jy=f_exc, f_exc_err_jy=f_exc_err, chi=float(chi),
                             saturated=bool(sat), eddington_shift_jy=shift)


# --------------------------------------------------------------------------
# 3. Excess temperature
# --------------------------------------------------------------------------
def fit_dust_temperature(measurements: list[ExcessMeasurement],
                         t_grid: np.ndarray | None = None
                         ) -> tuple[float, float, float, str]:
    """Fit the excess colour temperature from two same-epoch bands.

    Returns ``(T, T_lo, T_hi, source)``. With two significant excess
    measurements at one epoch the ratio pins ``T`` and the transfer to the late
    band becomes a measured quantity rather than a marginalised nuisance -- the
    difference between a 20% and a 400% systematic for the 9-to-12 micron pair.
    ``source`` is ``"fitted"`` or ``"prior"``.
    """
    usable = [m for m in measurements if np.isfinite(m.f_exc_jy) and m.f_exc_err_jy > 0]
    if len(usable) < 2 or not all(m.f_exc_jy > 2 * m.f_exc_err_jy for m in usable):
        return (T_DUST_DEFAULT_K, T_DUST_PRIOR_K[0], T_DUST_PRIOR_K[1], "prior")

    grid = t_grid if t_grid is not None else np.geomspace(*T_DUST_PRIOR_K, 240)
    ref = usable[0]
    chi2 = np.zeros_like(grid)
    for t_i, temp in enumerate(grid):
        # Scale to the reference band, then compare the predicted others.
        pred = [ref.f_exc_jy * transfer(BANDS[ref.band], BANDS[m.band], float(temp))
                for m in usable]
        chi2[t_i] = float(np.sum([((p - m.f_exc_jy) / m.f_exc_err_jy) ** 2
                                  for p, m in zip(pred, usable, strict=True)]))
    best = int(np.argmin(chi2))
    within = grid[chi2 <= chi2[best] + 1.0]
    lo = float(within.min()) if within.size else T_DUST_PRIOR_K[0]
    hi = float(within.max()) if within.size else T_DUST_PRIOR_K[1]
    return float(grid[best]), lo, hi, "fitted"


# --------------------------------------------------------------------------
# 4. The cessation statistic
# --------------------------------------------------------------------------
@dataclass
class CessationResult:
    """Outcome of comparing an early-epoch excess with a later-epoch one."""

    early_band: str
    late_band: str
    baseline_yr: float
    t_dust_k: float
    t_dust_source: str
    exc_early_jy: float
    exc_early_err_jy: float
    exc_late_pred_jy: float
    exc_late_pred_err_jy: float
    exc_late_obs_jy: float
    exc_late_obs_err_jy: float
    delta_jy: float
    delta_err_jy: float
    z: float
    f_cess: float
    f_cess_err: float
    verdict: str
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def cessation(early: ExcessMeasurement, late: ExcessMeasurement,
              t_dust_k: float = T_DUST_DEFAULT_K,
              t_dust_lo: float = T_DUST_PRIOR_K[0],
              t_dust_hi: float = T_DUST_PRIOR_K[1],
              t_dust_source: str = "prior",
              min_early_snr: float = 5.0) -> CessationResult:
    """Compare an early excess with a later one in a different band.

    The early excess is transported to the late band assuming the excess is a
    blackbody at ``t_dust_k``, applying **both** bands' quoting conventions and
    response functions. The transfer uncertainty is the span of the transfer
    over ``[t_dust_lo, t_dust_hi]`` combined with the bandpass systematic -- so
    a source whose dust temperature could not be fitted is *penalised*, rather
    than silently assigned a convenient temperature.

    ``f_cess`` is the fraction of the excess that vanished: 1 means it switched
    off completely, 0 means it is unchanged, negative means it grew.
    """
    b_e, b_l = BANDS[early.band], BANDS[late.band]
    flags: list[str] = []

    c_nom, c_bpsys = transfer_with_systematic(b_e, b_l, t_dust_k)
    c_lo = transfer(b_e, b_l, t_dust_lo)
    c_hi = transfer(b_e, b_l, t_dust_hi)
    c_temp_sys = 0.5 * abs(c_hi - c_lo)
    c_err = float(np.hypot(c_bpsys, c_temp_sys))

    pred = c_nom * early.f_exc_jy
    pred_err = float(np.hypot(c_nom * early.f_exc_err_jy, c_err * abs(early.f_exc_jy)))

    delta = pred - late.f_exc_jy
    delta_err = float(np.hypot(pred_err, late.f_exc_err_jy))
    z = delta / delta_err if delta_err > 0 else float("nan")

    if pred_err > 0 and abs(pred) > 0:
        f_cess = 1.0 - late.f_exc_jy / pred if pred != 0 else float("nan")
        # Ratio-of-normals: only meaningful when the denominator is well measured.
        f_cess_err = abs(1.0 - f_cess) * float(np.hypot(
            late.f_exc_err_jy / late.f_exc_jy if late.f_exc_jy else np.inf,
            pred_err / pred)) if late.f_exc_jy else abs(late.f_exc_err_jy / pred)
    else:
        f_cess = f_cess_err = float("nan")

    early_snr = early.f_exc_jy / early.f_exc_err_jy if early.f_exc_err_jy > 0 else 0.0
    if early_snr < min_early_snr:
        verdict = "no_early_excess"
        flags.append(f"early excess S/N {early_snr:.1f} < {min_early_snr}")
    elif early.saturated:
        verdict = "early_saturated"
        flags.append(f"{early.band} at/above saturation onset")
    elif late.saturated:
        # A saturated late band UNDER-reports flux and manufactures a fade.
        verdict = "late_saturated"
        flags.append(f"{late.band} at/above saturation onset -- fade not believable")
    elif not np.isfinite(z):
        verdict = "undetermined"
    elif z >= 5.0 and np.isfinite(f_cess) and f_cess > 0.5:
        verdict = "cessation"
    elif z >= 3.0:
        verdict = "fade"
    elif z <= -3.0:
        verdict = "rise"
    else:
        verdict = "constant"

    if t_dust_source == "prior":
        flags.append("dust temperature not fitted; transfer marginalised over "
                     f"{t_dust_lo:.0f}-{t_dust_hi:.0f} K")
    if c_err / abs(c_nom) > 0.25 if c_nom else False:
        flags.append(f"transfer systematic {100 * c_err / abs(c_nom):.0f}% -- "
                     "poorly conditioned band pair")

    return CessationResult(
        early_band=early.band, late_band=late.band,
        baseline_yr=round(b_l.epoch_year - b_e.epoch_year, 1),
        t_dust_k=float(t_dust_k), t_dust_source=t_dust_source,
        exc_early_jy=early.f_exc_jy, exc_early_err_jy=early.f_exc_err_jy,
        exc_late_pred_jy=float(pred), exc_late_pred_err_jy=float(pred_err),
        exc_late_obs_jy=late.f_exc_jy, exc_late_obs_err_jy=late.f_exc_err_jy,
        delta_jy=float(delta), delta_err_jy=float(delta_err), z=float(z),
        f_cess=float(f_cess), f_cess_err=float(f_cess_err),
        verdict=verdict, flags=flags)


def cessation_mc(early: ExcessMeasurement, late: ExcessMeasurement,
                 t_dust_lo: float = T_DUST_PRIOR_K[0],
                 t_dust_hi: float = T_DUST_PRIOR_K[1],
                 n_draws: int = 4000, seed: int = 0) -> dict:
    """Monte-Carlo version of :func:`cessation`, for shortlisted sources.

    The analytic propagation linearises the temperature dependence of the
    transfer, which is a poor approximation for the 9-to-12 micron pair where
    the transfer moves by a factor of ~5 across the prior. This draws the dust
    temperature log-uniformly and returns the empirical distribution of the
    statistic. Used to confirm every candidate that the fast path flags.
    """
    rng = np.random.default_rng(seed)
    b_e, b_l = BANDS[early.band], BANDS[late.band]
    temps = np.exp(rng.uniform(np.log(t_dust_lo), np.log(t_dust_hi), n_draws))
    # Transfer is smooth in log T; tabulate once and interpolate.
    grid = np.geomspace(t_dust_lo, t_dust_hi, 48)
    tab = np.array([transfer(b_e, b_l, float(t)) for t in grid])
    c = np.interp(np.log(temps), np.log(grid), tab)

    e_early = rng.normal(early.f_exc_jy, max(early.f_exc_err_jy, 1e-12), n_draws)
    e_late = rng.normal(late.f_exc_jy, max(late.f_exc_err_jy, 1e-12), n_draws)
    delta = c * e_early - e_late
    with np.errstate(divide="ignore", invalid="ignore"):
        f_cess = 1.0 - e_late / (c * e_early)
    f_cess = f_cess[np.isfinite(f_cess)]
    return {
        "delta_median_jy": float(np.median(delta)),
        "delta_p16_jy": float(np.percentile(delta, 16)),
        "delta_p84_jy": float(np.percentile(delta, 84)),
        "p_delta_gt_0": float(np.mean(delta > 0)),
        "f_cess_median": float(np.median(f_cess)) if f_cess.size else float("nan"),
        "f_cess_p16": float(np.percentile(f_cess, 16)) if f_cess.size else float("nan"),
        "f_cess_p84": float(np.percentile(f_cess, 84)) if f_cess.size else float("nan"),
        "n_draws": int(n_draws),
    }


# --------------------------------------------------------------------------
# 5. Three-epoch adjudication
# --------------------------------------------------------------------------
#: Morphology labels for the IRAS / AKARI / WISE excess pattern.
LADDER_VERDICTS = (
    "fade_2007_2010",     # IRAS high, AKARI high, WISE low  <- TYC 8241 morphology
    "fade_1983_2006",     # IRAS high, AKARI low,  WISE low
    "monotone_decline",   # each epoch below the last
    "incoherent",         # high / low / high -- a systematic, not a source
    "constant",
    "rise",
    "no_mid_epoch",       # AKARI absent: cannot separate real fade from IRAS artefact
    "insufficient_ir",
)


def adjudicate_ladder(iras: CessationResult | None,
                      akari: CessationResult | None,
                      akari_to_wise: CessationResult | None) -> tuple[str, list[str]]:
    """Classify a source from its position on the IRAS -> AKARI -> WISE ladder.

    ``iras`` compares IRAS with WISE, ``akari`` compares IRAS with AKARI, and
    ``akari_to_wise`` compares AKARI with WISE. Any may be ``None`` when the
    source is undetected or uncovered in that survey.

    The discriminating power is in the *middle* epoch. Without AKARI, an
    IRAS-high / WISE-low source is degenerate between a genuine 27-year fade and
    an IRAS blend or spurious entry -- and the prior strongly favours the
    latter. That degeneracy is why ``no_mid_epoch`` is a verdict in its own
    right rather than a weak detection.
    """
    notes: list[str] = []
    if iras is None and akari_to_wise is None:
        return "insufficient_ir", ["no usable early-epoch excess"]

    if akari is None and akari_to_wise is None:
        if iras is not None and iras.verdict in ("cessation", "fade"):
            notes.append("no AKARI epoch: 27-yr fade is degenerate with an IRAS "
                         "blend or spurious source; not adjudicable")
            return "no_mid_epoch", notes
        return ("rise" if iras is not None and iras.verdict == "rise"
                else "constant"), notes

    faded = ("cessation", "fade")
    iras_akari_faded = akari is not None and akari.verdict in faded
    akari_wise_faded = akari_to_wise is not None and akari_to_wise.verdict in faded
    iras_akari_rose = akari is not None and akari.verdict == "rise"
    akari_wise_rose = akari_to_wise is not None and akari_to_wise.verdict == "rise"

    if iras_akari_faded and akari_wise_faded:
        notes.append("excess declined across both intervals")
        return "monotone_decline", notes
    if akari_wise_faded and not iras_akari_faded and not iras_akari_rose:
        notes.append("excess stable 1983-2006 then dropped by 2010 "
                     "(TYC 8241 2652 1 morphology); WISE-side artefact must be excluded")
        return "fade_2007_2010", notes
    if iras_akari_faded and not akari_wise_faded and not akari_wise_rose:
        notes.append("drop occurred between IRAS and AKARI; IRAS blending and "
                     "spurious-source tests are decisive here")
        return "fade_1983_2006", notes
    if (iras_akari_faded and akari_wise_rose) or (iras_akari_rose and akari_wise_faded):
        notes.append("non-monotonic across three instruments -- systematic")
        return "incoherent", notes
    if iras_akari_rose or akari_wise_rose:
        return "rise", notes
    return "constant", notes


# --------------------------------------------------------------------------
# 6. IRAS beam-sum consistency
# --------------------------------------------------------------------------
def beam_sum_consistency(iras_flux_jy: float, iras_err_jy: float,
                         neighbour_late_fluxes_jy: list[float],
                         transfer_ratio: float,
                         tolerance_sigma: float = 3.0) -> dict:
    """Test whether the early flux is explained by *everything* in its beam.

    The IRAS 12-micron beam is roughly 0.75' x 4.5' -- some 300 times the solid
    angle of WISE W3. An IRAS flux is therefore the sum over every source in
    that footprint, so comparing it with the WISE flux of the single nearest
    counterpart is guaranteed to manufacture fades wherever the field is
    crowded. The only defensible comparison sums **all** late-epoch sources
    inside the early beam:

        F_early  vs  sum_over_beam( F_late ) * transfer

    If the summed late flux accounts for the early flux, there is no fade: the
    early epoch simply could not resolve the field. A real cessation must
    survive with the whole neighbourhood counted in.

    This test costs one extra cone search per source and removes the single
    largest contaminant of the 27-year pair, so it runs on every source rather
    than only on shortlisted ones.
    """
    summed = float(np.nansum(np.asarray(neighbour_late_fluxes_jy, dtype=float)))
    predicted_early = summed / transfer_ratio if transfer_ratio else float("nan")
    resid = float(iras_flux_jy) - predicted_early
    sigma = float(max(iras_err_jy, 1e-12))
    z = resid / sigma
    explained = z < tolerance_sigma
    return {
        "n_neighbours": int(len(neighbour_late_fluxes_jy)),
        "late_sum_jy": summed,
        "early_predicted_from_sum_jy": predicted_early,
        "residual_jy": resid,
        "z_unexplained": float(z),
        "beam_explained": bool(explained),
        "note": ("early flux is accounted for by the summed late-epoch sources in "
                 "the beam -- no fade, just resolution" if explained else
                 "early flux exceeds the summed late-epoch flux in the beam"),
    }


# --------------------------------------------------------------------------
# 7. Empirical null calibration
# --------------------------------------------------------------------------
def calibrate_null(z_values: np.ndarray, quantile: float = 0.999) -> dict:
    """Set the detection threshold from the sample's own rising tail.

    Every symmetric systematic -- photometric scatter, locus error, transfer
    error, cross-calibration -- populates the fading (``z > 0``) and rising
    (``z < 0``) tails equally. Eddington bias does not: it fades only. So the
    **rising tail is a matched null for the fading tail**, drawn from the same
    stars, the same fluxes and the same pipeline, and the asymmetry between them
    is the residual one-directional systematic.

    The returned ``threshold`` is the magnitude of the rising-tail quantile: a
    fading source must beat the most extreme rises the sample produces. This is
    strictly more conservative than a Gaussian threshold whenever the real error
    distribution has tails, which it always does.
    """
    z = np.asarray(z_values, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 100:
        return {"threshold": 5.0, "n": int(z.size), "degraded": True,
                "note": "too few sources for an empirical null; Gaussian 5 sigma used"}
    rise = -z[z < 0]
    if rise.size < 30:
        return {"threshold": 5.0, "n": int(z.size), "degraded": True,
                "note": "rising tail too sparse; Gaussian 5 sigma used"}
    thr = float(np.quantile(rise, quantile))
    n_fade = int((z > thr).sum())
    n_rise = int((rise > thr).sum())
    return {
        "threshold": max(thr, 3.0),
        "n": int(z.size),
        "n_rise": int(rise.size),
        "rise_quantile": float(thr),
        "n_fade_above_threshold": n_fade,
        "n_rise_above_threshold": n_rise,
        "asymmetry_excess": n_fade - n_rise,
        "degraded": False,
        "note": ("threshold set by the sample's own rising tail; the excess of "
                 "faders over risers above it is the only quantity that can "
                 "contain a real signal"),
    }

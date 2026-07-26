"""Modified-blackbody SED models and multi-component decomposition.

This is the physics layer of the ISOTHERM channel.  Everything here is a pure
function of numpy arrays: no network, no config, no I/O, so the whole detector
is exercised offline by ``tests/test_isotherm.py``.

The models
----------
Three nested families, all written in ``F_nu`` (arbitrary flux units) against
wavelength in micron.

1. **Modified blackbody** (one temperature)::

       F_nu(lam) = A * (lam / lam0)**(-beta) * B_nu(lam, T)

   ``beta`` is the dust emissivity index: real grains have an absorption
   efficiency ``Q_abs ~ lam**(-beta)`` for ``lam > 2*pi*a``, with ``beta ~ 1-2``
   for astronomical silicate/carbon.  ``beta = 0`` is a *true Planck function* —
   an emitter that is large compared with every wavelength observed.  That is
   easy for an engineered radiator and hard for dust in quantity.

2. **Discrete N-component**::

       F_nu(lam) = sum_k A_k * (lam / lam0)**(-beta) * B_nu(lam, T_k)

   A staged computational cascade (the "Matrioshka" architecture: each shell
   harvests the previous shell's waste heat and re-radiates cooler) produces a
   *discrete* set of temperatures.  ``A_k >= 0`` is enforced, so components are
   emission, never a fitted absorption trough.

3. **Continuous radial gradient** — THE NATURAL NULL.  Optically thin dust in a
   disk with surface density ``Sigma ~ r**(-p)`` and the equilibrium profile
   ``T ~ r**(-1/2)`` emits::

       F_nu(lam) ~ lam**(-beta) * Integral[ dT * T**(2p-5) * B_nu(lam, T) ]

   over ``T_out < T < T_in``.  The change of variables is
   ``r = r_in (T/T_in)**(-2)``, so ``r**(1-p) dr ~ T**(2p-5) dT``: a radial
   gradient is *necessarily* a broad superposition of Planck functions, and its
   width is set by the radial extent through ``dT/T = 0.5 * dr/r``.

The discriminating comparison is (2) versus (3) — NOT "does a single blackbody
fit well".  Debris-disk practice already quotes a single ``T_dust`` for hundreds
of disks and two-temperature (warm belt + cold belt) systems are common and
entirely natural, so neither a good single-blackbody fit nor a two-component
decomposition is anomalous on its own.  What has no natural counterpart is a
temperature distribution *narrower than any physical radial extent allows*, or
>= 3 resolved components in geometric progression, at ``beta = 0``, with no
silicate emission.  See ``shape_stats.py`` for those statistics.

Model selection
---------------
Adding components always lowers chi2, so component count is chosen by an
information criterion, not by fit quality.  Two guards matter:

* **BIC uses the number of resolution elements, not the number of pixels.**
  Spitzer/IRS low-resolution spectra are oversampled ~2x, so raw-pixel BIC
  over-rewards complexity by ~ln(2) per parameter.  ``bin_to_resolution``
  rebins onto an R-spaced grid first.
* **A fractional systematic floor** is added in quadrature.  The dominant IRS
  systematic is order-stitching mismatch between the SL and LL modules, which
  is correlated and not captured by the pipeline error array.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, nnls

# --- constants -------------------------------------------------------------

_HC_OVER_K_UMK = 14387.768775039337   # h*c/k in micron * kelvin
_TWO_H_OVER_C2 = 1.4745007e-47        # 2h/c^2 in cgs (erg s^2 / cm^2)
_C_UM_HZ = 2.99792458e14              # c in micron * Hz
_EXP_CLIP = 700.0                     # exp() overflow guard for float64

# Wavelength at which the emissivity law is normalised.  Sitting the pivot in
# the middle of the Spitzer/IRS band decorrelates the amplitude from beta.
LAM0_UM = 24.0

# Temperature search range.  The floor is where a 38 micron cutoff still sees
# the Wien peak; the ceiling is above dust sublimation, deliberately, so that a
# too-hot component is *fitted* and then vetoed as a companion photosphere
# (repo contamination ledger) rather than being excluded by construction.
T_MIN_K = 20.0
T_MAX_K = 3000.0

# Two fitted components closer than this ratio are degenerate: NNLS cannot
# apportion flux between them and the "discrete" decomposition is meaningless.
MIN_T_RATIO = 1.35


# --- Planck and emissivity -------------------------------------------------

def planck_nu(lam_um: np.ndarray, t_k: float) -> np.ndarray:
    """Planck function ``B_nu`` in cgs (erg / s / cm^2 / Hz / sr).

    Underflows to 0 on the Wien side rather than overflowing.
    """
    lam = np.asarray(lam_um, dtype=float)
    if not np.isfinite(t_k) or t_k <= 0:
        return np.zeros_like(lam)
    nu = _C_UM_HZ / lam
    x = np.clip(_HC_OVER_K_UMK / (lam * t_k), 1e-12, _EXP_CLIP)
    with np.errstate(over="ignore", invalid="ignore"):
        out = _TWO_H_OVER_C2 * nu**3 / np.expm1(x)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def emissivity(lam_um: np.ndarray, beta: float, lam0_um: float = LAM0_UM) -> np.ndarray:
    """Emissivity law ``Q(lam) = (lam / lam0)**(-beta)``; ``beta = 0`` is grey."""
    return (np.asarray(lam_um, dtype=float) / float(lam0_um)) ** (-float(beta))


def mbb(lam_um: np.ndarray, t_k: float, beta: float = 0.0,
        lam0_um: float = LAM0_UM) -> np.ndarray:
    """Modified blackbody ``Q(lam) * B_nu(lam, T)``, unnormalised."""
    return emissivity(lam_um, beta, lam0_um) * planck_nu(lam_um, t_k)


def gradient_sed(lam_um: np.ndarray, t_in_k: float, t_out_k: float,
                 p_index: float = 1.0, beta: float = 0.0,
                 lam0_um: float = LAM0_UM, n_t: int = 64) -> np.ndarray:
    """Continuous radial-temperature-gradient SED (the natural null).

    Emission-measure weighting ``dE/dT ~ T**(2p - 5)`` follows from a surface
    density ``Sigma ~ r**(-p)`` and ``T ~ r**(-1/2)``.  ``p_index = 1`` (the
    canonical debris-disk value) gives ``dE/dT ~ T**-3``.

    Degenerates smoothly to a single modified blackbody as ``t_in -> t_out``.
    """
    lam = np.asarray(lam_um, dtype=float)
    t_hi, t_lo = max(t_in_k, t_out_k), min(t_in_k, t_out_k)
    if t_lo <= 0 or not np.isfinite(t_hi) or not np.isfinite(t_lo):
        return np.zeros_like(lam)
    if t_hi / t_lo < 1.02:                       # isothermal limit
        return mbb(lam, np.sqrt(t_hi * t_lo), beta, lam0_um)

    q = 2.0 * float(p_index) - 5.0
    lnt = np.linspace(np.log(t_lo), np.log(t_hi), int(n_t))
    temps = np.exp(lnt)
    # Integral over dT becomes an integral over d(lnT) with one extra power.
    weights = temps ** (q + 1.0)
    planck_stack = np.stack([planck_nu(lam, float(t)) for t in temps], axis=0)
    integ = np.trapezoid(planck_stack * weights[:, None], lnt, axis=0)
    return emissivity(lam, beta, lam0_um) * integ


def bolometric(shape: np.ndarray, lam_um: np.ndarray) -> float:
    """Integrate ``F_nu`` over frequency: ``Int F_nu dnu = Int F_nu c/lam^2 dlam``.

    Units are arbitrary but consistent, so ratios between components — which is
    all the energy-ladder test needs — are exact.
    """
    lam = np.asarray(lam_um, dtype=float)
    f = np.asarray(shape, dtype=float)
    ok = np.isfinite(f) & np.isfinite(lam) & (lam > 0)
    if ok.sum() < 2:
        return 0.0
    return float(np.trapezoid(f[ok] * _C_UM_HZ / lam[ok] ** 2, lam[ok]))


def component_bolometric(t_k: float, beta: float = 0.0, lam0_um: float = LAM0_UM,
                         lam_lo: float = 0.3, lam_hi: float = 5000.0,
                         n: int = 900) -> float:
    """Bolometric integral of a *unit-amplitude* modified blackbody.

    Multiplying by a fitted amplitude converts it to a component luminosity in
    the same arbitrary units for every component, so ``L_k / sum L`` is exact.
    """
    grid = np.geomspace(lam_lo, lam_hi, n)
    return bolometric(mbb(grid, t_k, beta, lam0_um), grid)


# --- data preparation ------------------------------------------------------

def bin_to_resolution(lam_um: np.ndarray, flux: np.ndarray, err: np.ndarray,
                      resolution: float = 100.0) -> tuple[np.ndarray, ...]:
    """Rebin onto a grid of independent resolution elements.

    Spitzer/IRS low-res is R ~ 60-130 but sampled ~2 pixels per element, so
    treating pixels as independent inflates the evidence for extra components.
    Returns ``(lam, flux, err, n_per_bin)`` with inverse-variance weighting and
    an error that never falls below the mean input error / sqrt(N).
    """
    lam = np.asarray(lam_um, float)
    f = np.asarray(flux, float)
    e = np.asarray(err, float)
    ok = np.isfinite(lam) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (lam > 0)
    lam, f, e = lam[ok], f[ok], e[ok]
    if lam.size == 0:
        z = np.array([])
        return z, z, z, z
    order = np.argsort(lam)
    lam, f, e = lam[order], f[order], e[order]

    step = 1.0 + 1.0 / float(resolution)
    edges = [lam[0] / np.sqrt(step)]
    while edges[-1] < lam[-1] * np.sqrt(step):
        edges.append(edges[-1] * step)
    edges = np.asarray(edges)
    idx = np.clip(np.digitize(lam, edges) - 1, 0, len(edges) - 2)

    out_l, out_f, out_e, out_n = [], [], [], []
    for b in np.unique(idx):
        m = idx == b
        w = 1.0 / e[m] ** 2
        wsum = w.sum()
        out_l.append(float((lam[m] * w).sum() / wsum))
        out_f.append(float((f[m] * w).sum() / wsum))
        out_e.append(float(np.sqrt(1.0 / wsum)))
        out_n.append(int(m.sum()))
    return (np.asarray(out_l), np.asarray(out_f),
            np.asarray(out_e), np.asarray(out_n))


def apply_systematic_floor(flux: np.ndarray, err: np.ndarray,
                           sys_frac: float = 0.05) -> np.ndarray:
    """Add a fractional systematic in quadrature to the pipeline error."""
    f = np.abs(np.asarray(flux, float))
    e = np.asarray(err, float)
    return np.sqrt(e**2 + (float(sys_frac) * f) ** 2)


# --- fit results -----------------------------------------------------------

@dataclass
class SEDFit:
    """Result of one SED fit.  ``kind`` is ``"discrete"`` or ``"gradient"``."""

    kind: str
    chi2: float
    n_data: int
    n_params: int
    beta: float
    temps_k: list[float] = field(default_factory=list)
    amps: list[float] = field(default_factory=list)
    lum_frac: list[float] = field(default_factory=list)
    t_in_k: float = float("nan")
    t_out_k: float = float("nan")
    p_index: float = float("nan")
    model: np.ndarray | None = None
    success: bool = True
    message: str = ""

    @property
    def n_components(self) -> int:
        return len(self.temps_k)

    @property
    def dof(self) -> int:
        return max(self.n_data - self.n_params, 1)

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof

    @property
    def bic(self) -> float:
        return self.chi2 + self.n_params * np.log(max(self.n_data, 2))

    @property
    def aicc(self) -> float:
        k, n = self.n_params, self.n_data
        pen = 2 * k + (2 * k * (k + 1) / (n - k - 1) if n > k + 1 else 1e6)
        return self.chi2 + pen

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "chi2": float(self.chi2),
            "reduced_chi2": float(self.reduced_chi2),
            "n_data": int(self.n_data), "n_params": int(self.n_params),
            "bic": float(self.bic), "aicc": float(self.aicc),
            "beta": float(self.beta),
            "n_components": int(self.n_components),
            "temps_k": [float(t) for t in self.temps_k],
            "lum_frac": [float(x) for x in self.lum_frac],
            "t_in_k": float(self.t_in_k), "t_out_k": float(self.t_out_k),
            "p_index": float(self.p_index),
            "success": bool(self.success), "message": self.message,
        }


# --- the fitting machinery -------------------------------------------------

def _design(lam: np.ndarray, temps, beta: float, lam0_um: float) -> np.ndarray:
    """Columns = unit-max modified blackbodies (conditions the NNLS)."""
    cols = []
    for t in temps:
        c = mbb(lam, float(t), beta, lam0_um)
        mx = c.max() if c.size else 0.0
        cols.append(c / mx if mx > 0 else np.zeros_like(lam))
    return np.stack(cols, axis=1) if cols else np.zeros((lam.size, 0))


def _solve_amps(design: np.ndarray, flux: np.ndarray,
                err: np.ndarray) -> tuple[np.ndarray, float]:
    """Non-negative least squares for the linear amplitudes; returns chi2."""
    if design.shape[1] == 0:
        return np.array([]), float(np.sum((flux / err) ** 2))
    a_w = design / err[:, None]
    b_w = flux / err
    try:
        amps, rnorm = nnls(a_w, b_w, maxiter=4000)
    except Exception:  # noqa: BLE001 - singular design; report as unusable
        return np.zeros(design.shape[1]), 1e18
    return amps, float(rnorm**2)


def _discrete_chi2(logts: np.ndarray, beta: float, lam, flux, err,
                   lam0_um: float, min_ratio: float) -> float:
    if not np.all(np.isfinite(logts)) or not np.isfinite(beta):
        return 1e18
    lo, hi = np.log10(T_MIN_K), np.log10(T_MAX_K)
    pen = 0.0
    for v in logts:
        if v < lo:
            pen += 1e6 * (lo - v)
        if v > hi:
            pen += 1e6 * (v - hi)
    if beta < -1.0:
        pen += 1e6 * (-1.0 - beta)
    if beta > 4.0:
        pen += 1e6 * (beta - 4.0)
    srt = np.sort(logts)
    gap = np.log10(min_ratio)
    for a, b in zip(srt[:-1], srt[1:], strict=True):
        if b - a < gap:
            pen += 1e6 * (gap - (b - a))
    if pen > 0:
        return 1e12 + pen
    temps = 10.0 ** srt
    des = _design(lam, temps, beta, lam0_um)
    return _solve_amps(des, flux, err)[1]


def _package_discrete(logts, beta, lam, flux, err, lam0_um, n_data,
                      success=True, message="") -> SEDFit:
    temps = np.sort(10.0 ** np.asarray(logts, float))
    des = _design(lam, temps, beta, lam0_um)
    amps, chi2 = _solve_amps(des, flux, err)
    keep = amps > 0
    temps, amps = temps[keep], amps[keep]
    des = des[:, keep] if des.shape[1] else des
    # Amplitudes are per unit-max column; undo that for the bolometric weight.
    lums = []
    for t, a in zip(temps, amps, strict=True):
        col = mbb(lam, float(t), beta, lam0_um)
        mx = col.max() if col.size else 1.0
        lums.append(a / mx * component_bolometric(float(t), beta, lam0_um)
                    if mx > 0 else 0.0)
    tot = float(np.sum(lums))
    frac = [float(x / tot) if tot > 0 else 0.0 for x in lums]
    n_par = 2 * len(temps) + 1                    # T_k, A_k, and beta
    return SEDFit(kind="discrete", chi2=float(chi2), n_data=int(n_data),
                  n_params=int(n_par), beta=float(beta),
                  temps_k=[float(t) for t in temps],
                  amps=[float(a) for a in amps], lum_frac=frac,
                  model=(des @ amps if des.shape[1] else np.zeros_like(lam)),
                  success=success, message=message)


def fit_discrete(lam_um: np.ndarray, flux: np.ndarray, err: np.ndarray,
                 n_components: int = 1, beta: float | None = None,
                 lam0_um: float = LAM0_UM, min_t_ratio: float = MIN_T_RATIO,
                 n_grid: int = 26, seed_temps=None) -> SEDFit:
    """Fit ``n_components`` modified blackbodies with a shared emissivity index.

    ``beta=None`` fits beta; a float holds it fixed.  Amplitudes are solved
    exactly by NNLS at every step, so the nonlinear search is only over
    ``n_components`` temperatures (+ beta) and is robust with multiple starts.
    """
    lam = np.asarray(lam_um, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    n_data = lam.size
    if n_data < 2 * n_components + 2:
        return SEDFit(kind="discrete", chi2=float("inf"), n_data=n_data,
                      n_params=2 * n_components + 1, beta=float("nan"),
                      success=False, message="insufficient_points")

    beta_free = beta is None
    grid = np.log10(np.geomspace(T_MIN_K, T_MAX_K, n_grid))
    beta_seeds = [0.0, 1.0, 2.0] if beta_free else [float(beta)]

    # --- seed: exhaustive for N=1, greedy addition for N>1 ---
    if seed_temps is not None:
        starts = [(np.log10(np.asarray(seed_temps, float)), b) for b in beta_seeds]
    elif n_components == 1:
        starts = [(np.array([g]), b) for g in grid for b in beta_seeds]
    else:
        prev = fit_discrete(lam, flux, err, n_components - 1, beta,
                            lam0_um, min_t_ratio, n_grid)
        if not prev.success or not prev.temps_k:
            return SEDFit(kind="discrete", chi2=float("inf"), n_data=n_data,
                          n_params=2 * n_components + 1, beta=float("nan"),
                          success=False, message="seed_failed")
        base = np.log10(np.asarray(prev.temps_k, float))
        starts = []
        for g in grid:
            if np.min(np.abs(base - g)) < np.log10(min_t_ratio):
                continue
            starts.append((np.append(base, g), prev.beta))
        if not starts:
            return SEDFit(kind="discrete", chi2=float("inf"), n_data=n_data,
                          n_params=2 * n_components + 1, beta=float("nan"),
                          success=False, message="no_separable_seed")

    def obj(theta):
        lt = theta[:n_components]
        b = theta[n_components] if beta_free else float(beta)
        return _discrete_chi2(lt, b, lam, flux, err, lam0_um, min_t_ratio)

    scored = []
    for lt, b in starts:
        th = np.append(lt, b) if beta_free else np.asarray(lt, float)
        scored.append((obj(th), th))
    scored.sort(key=lambda x: x[0])
    if not np.isfinite(scored[0][0]) or scored[0][0] >= 1e12:
        return SEDFit(kind="discrete", chi2=float("inf"), n_data=n_data,
                      n_params=2 * n_components + 1, beta=float("nan"),
                      success=False, message="no_feasible_start")

    best_val, best_th = scored[0]
    for _, th0 in scored[: min(6, len(scored))]:
        res = minimize(obj, th0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
        if res.fun < best_val:
            best_val, best_th = float(res.fun), np.asarray(res.x, float)

    lt = best_th[:n_components]
    b = float(best_th[n_components]) if beta_free else float(beta)
    return _package_discrete(lt, b, lam, flux, err, lam0_um, n_data)


def fit_gradient(lam_um: np.ndarray, flux: np.ndarray, err: np.ndarray,
                 beta: float | None = None, lam0_um: float = LAM0_UM,
                 p_bounds: tuple[float, float] = (-1.0, 3.0)) -> SEDFit:
    """Fit the continuous radial-gradient model — the natural-disk null.

    Free parameters: ``T_in``, ``T_out``, the surface-density index ``p``,
    ``beta``, and one amplitude.  ``T_in >= T_out`` is enforced by construction
    (the search is over ``log T_out`` and ``log(T_in / T_out) >= 0``).
    """
    lam = np.asarray(lam_um, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    n_data = lam.size
    beta_free = beta is None
    n_par = 5 if beta_free else 4
    if n_data < n_par + 2:
        return SEDFit(kind="gradient", chi2=float("inf"), n_data=n_data,
                      n_params=n_par, beta=float("nan"), success=False,
                      message="insufficient_points")

    def unpack(theta):
        lt_out = theta[0]
        dl = abs(theta[1])                       # log10(T_in / T_out) >= 0
        p = theta[2]
        b = theta[3] if beta_free else float(beta)
        return lt_out, dl, p, b

    def obj(theta):
        lt_out, dl, p, b = unpack(theta)
        lo, hi = np.log10(T_MIN_K), np.log10(T_MAX_K)
        pen = 0.0
        if lt_out < lo:
            pen += 1e6 * (lo - lt_out)
        if lt_out + dl > hi:
            pen += 1e6 * (lt_out + dl - hi)
        if dl > 3.0:
            pen += 1e6 * (dl - 3.0)
        if p < p_bounds[0]:
            pen += 1e6 * (p_bounds[0] - p)
        if p > p_bounds[1]:
            pen += 1e6 * (p - p_bounds[1])
        if b < -1.0:
            pen += 1e6 * (-1.0 - b)
        if b > 4.0:
            pen += 1e6 * (b - 4.0)
        if pen > 0:
            return 1e12 + pen
        col = gradient_sed(lam, 10 ** (lt_out + dl), 10**lt_out, p, b, lam0_um)
        mx = col.max()
        if not np.isfinite(mx) or mx <= 0:
            return 1e18
        des = (col / mx)[:, None]
        return _solve_amps(des, flux, err)[1]

    starts = []
    for lt_out in np.log10(np.geomspace(T_MIN_K, T_MAX_K / 2, 9)):
        for dl in (0.05, 0.3, 0.7, 1.2):
            for p in (0.0, 1.0, 2.0):
                base = [lt_out, dl, p]
                starts.append(np.asarray(base + ([1.0] if beta_free else []),
                                         float))
    scored = sorted(((obj(s), s) for s in starts), key=lambda x: x[0])
    if not scored or scored[0][0] >= 1e12:
        return SEDFit(kind="gradient", chi2=float("inf"), n_data=n_data,
                      n_params=n_par, beta=float("nan"), success=False,
                      message="no_feasible_start")

    best_val, best_th = scored[0]
    for _, th0 in scored[:8]:
        res = minimize(obj, th0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-6})
        if res.fun < best_val:
            best_val, best_th = float(res.fun), np.asarray(res.x, float)

    lt_out, dl, p, b = unpack(best_th)
    t_out, t_in = 10**lt_out, 10 ** (lt_out + dl)
    col = gradient_sed(lam, t_in, t_out, p, b, lam0_um)
    mx = col.max() if col.size else 1.0
    des = (col / (mx if mx > 0 else 1.0))[:, None]
    amps, chi2 = _solve_amps(des, flux, err)
    return SEDFit(kind="gradient", chi2=float(chi2), n_data=n_data,
                  n_params=n_par, beta=float(b),
                  temps_k=[float(np.sqrt(t_in * t_out))],
                  amps=[float(amps[0])] if amps.size else [], lum_frac=[1.0],
                  t_in_k=float(t_in), t_out_k=float(t_out), p_index=float(p),
                  model=(des @ amps) if amps.size else np.zeros_like(lam),
                  success=True)


def select_n_components(lam_um: np.ndarray, flux: np.ndarray, err: np.ndarray,
                        n_max: int = 4, delta_bic: float = 10.0,
                        beta: float | None = None,
                        lam0_um: float = LAM0_UM) -> tuple[SEDFit, list[SEDFit]]:
    """Choose component count by BIC; a component must *earn* ``delta_bic``.

    Returns ``(best, ladder)``.  ``delta_bic = 10`` is "very strong" on the
    Kass & Raftery scale — deliberately conservative, because the entire claim
    of this channel rests on component multiplicity being real.
    """
    ladder = [fit_discrete(lam_um, flux, err, n, beta, lam0_um)
              for n in range(1, int(n_max) + 1)]
    usable = [f for f in ladder if f.success and np.isfinite(f.chi2)]
    if not usable:
        return ladder[0], ladder
    best = usable[0]
    for f in usable[1:]:
        if f.n_components > best.n_components and f.bic < best.bic - delta_bic:
            best = f
    return best, ladder


__all__ = [
    "LAM0_UM", "MIN_T_RATIO", "SEDFit", "T_MAX_K", "T_MIN_K",
    "apply_systematic_floor", "bin_to_resolution", "bolometric",
    "component_bolometric", "emissivity", "fit_discrete", "fit_gradient",
    "gradient_sed", "mbb", "planck_nu", "select_n_components",
]

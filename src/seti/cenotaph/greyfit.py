"""Joint (grey, reddening) decomposition of a multi-band magnitude residual.

The physics
-----------
An occulter of geometric covering fraction ``f`` in front of a star removes the
same *fraction* of photons at every wavelength, so it adds

    Δm_b = −2.5 log10(1 − f) ≡ g          (identical in every band b)

An interstellar dust column adds ``A_V · R_b`` where ``R_b = A_band/A_V`` spans
a factor ≈38 over Gaia BP → WISE W2 (×88 including GALEX NUV). The two are
therefore *near-orthogonal vectors in magnitude space* and can be fitted
simultaneously — extinction is never assumed to be zero, it is measured and
divided out.

The design matrix is ``X = [1, R]`` and the model is ``ΔM = X·(g, A_V)``.

The one thing that is NOT orthogonal to grey
--------------------------------------------
A distance error is *exactly* a grey offset: a parallax error δϖ shifts every
absolute magnitude by the same ``5/ln10 · δϖ/ϖ``. There is no photometric way
to tell it from an occulter. This module therefore does **generalised** least
squares with a rank-1 fully-correlated noise term

    C = diag(σ_b²) + σ_μ² · 1 1ᵀ

where σ_μ is the distance-modulus uncertainty. The consequences fall out
automatically and correctly:

* ``σ_g`` inflates to include the parallax term — no candidate can be
  manufactured by a bad parallax;
* ``A_V`` is *immune* to it, because the rank-1 term lies entirely along the
  grey direction (colours do not care how far away the star is).

Treating this as a diagonal problem would silently overstate every
significance in the channel; that is the single most dangerous error available
here, so it is handled in the covariance rather than in a footnote.

Sign convention
---------------
``ΔM_b = M_b,observed − M_b,intrinsic``. Positive = fainter than it should be.
An occulter gives ``g > 0``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .extinction import a_over_av, covering_fraction_from_grey

# Cross-survey zero-point / bandpass-transformation systematics. Even with
# perfect photon statistics, GALEX-to-Gaia-to-2MASS-to-WISE ties are good to a
# few hundredths of a magnitude, and the twin construction cannot remove a
# survey-level offset that is common to target and twins alike only if the
# colour distributions differ. This floor is added in quadrature per band.
DEFAULT_SYS_FLOOR_MAG: float = 0.02

# Below this the design matrix is too collinear to separate grey from
# reddening: the fit will return huge, anticorrelated errors and any
# "significance" is an artefact of the band set, not the star.
MIN_LEVER_ARM: float = 0.30
"""Minimum required span of ``R_b`` (max − min) across the fitted bands."""

MAX_ABS_RHO: float = 0.995
"""Maximum tolerated |correlation| between the fitted ``g`` and ``A_V``."""


@dataclass
class GreyFit:
    """Result of the joint grey/reddening decomposition."""

    grey_mag: float
    grey_err: float
    av: float
    av_err: float
    rho_grey_av: float
    """Correlation coefficient between the two fitted parameters."""
    chi2: float
    dof: int
    n_bands: int
    bands: tuple[str, ...] = ()
    lever_arm: float = 0.0
    """``max(R_b) − min(R_b)`` over the fitted bands — the separating power."""
    av_at_boundary: bool = False
    """True if the unconstrained A_V went negative and was pinned at the prior/0."""
    delta_chi2_grey: float = 0.0
    """χ² improvement from adding the grey term to a reddening-only model."""
    verdict: str = "ok"
    """``ok`` | ``degenerate_band_set`` | ``too_few_bands`` | ``bad_sed_fit``"""
    notes: list[str] = field(default_factory=list)

    @property
    def significance(self) -> float:
        """Signed significance of the grey term, ``g / σ_g``."""
        if not math.isfinite(self.grey_err) or self.grey_err <= 0:
            return 0.0
        return self.grey_mag / self.grey_err

    @property
    def covering_fraction(self) -> float:
        """Geometric covering fraction implied by ``grey_mag`` (may be < 0)."""
        return covering_fraction_from_grey(self.grey_mag)

    @property
    def covering_fraction_err(self) -> float:
        """1σ on ``f`` propagated from ``σ_g``: ``df/dg = 0.4 ln10 (1−f)``."""
        f = self.covering_fraction
        return 0.4 * math.log(10.0) * max(1.0 - f, 0.0) * self.grey_err

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bands"] = list(self.bands)
        d["significance"] = self.significance
        d["covering_fraction"] = self.covering_fraction
        d["covering_fraction_err"] = self.covering_fraction_err
        d["reduced_chi2"] = self.reduced_chi2
        return d


def _gls(x_design: np.ndarray, y: np.ndarray, cov: np.ndarray):
    """Generalised least squares. Returns (theta, param_cov, chi2)."""
    cinv = np.linalg.inv(cov)
    a = x_design.T @ cinv @ x_design
    b = x_design.T @ cinv @ y
    pcov = np.linalg.inv(a)
    theta = pcov @ b
    resid = y - x_design @ theta
    chi2 = float(resid @ cinv @ resid)
    return theta, pcov, chi2


def fit_grey_reddening(
    bands: list[str],
    dmag: np.ndarray | list[float],
    sigma: np.ndarray | list[float],
    *,
    dist_modulus_sigma: float = 0.0,
    av_prior: tuple[float, float] | None = None,
    sys_floor_mag: float = DEFAULT_SYS_FLOOR_MAG,
    enforce_av_nonneg: bool = True,
) -> GreyFit:
    """Decompose per-band absolute-magnitude residuals into grey + reddening.

    Parameters
    ----------
    bands
        Band names, keys of :data:`~seti.cenotaph.extinction.BAND_INDEX`.
    dmag
        ``M_b,observed − M_b,intrinsic`` per band (mag). Positive = fainter.
    sigma
        Per-band 1σ uncertainty on ``dmag`` (mag), *excluding* the distance term.
    dist_modulus_sigma
        1σ on the distance modulus (mag). Enters as a fully correlated rank-1
        component — the term that is degenerate with grey.
    av_prior
        Optional ``(A_V, σ)`` from a 3D dust map, added as a pseudo-observation.
        Use it; a line of sight with a known column is the cheapest way to keep
        the fit off the grey/reddening ridge for a short band set.
    sys_floor_mag
        Per-band systematic floor added in quadrature.
    enforce_av_nonneg
        If the unconstrained A_V lands below zero (or below the prior mean minus
        3σ), refit with A_V pinned at the boundary. Negative dust is a fitting
        artefact, and letting it float steals flux from the grey term.
    """
    bands = list(bands)
    y = np.asarray(dmag, dtype=float)
    sig = np.asarray(sigma, dtype=float)

    good = np.isfinite(y) & np.isfinite(sig) & (sig > 0)
    bands = [b for b, k in zip(bands, good, strict=False) if k]
    y = y[good]
    sig = sig[good]
    n = len(bands)

    if n < 3:
        return GreyFit(
            grey_mag=float("nan"), grey_err=float("inf"), av=float("nan"),
            av_err=float("inf"), rho_grey_av=float("nan"), chi2=float("nan"),
            dof=0, n_bands=n, bands=tuple(bands), verdict="too_few_bands",
            notes=[f"need >=3 finite bands to fit 2 parameters + test the fit; got {n}"],
        )

    r = np.array([a_over_av(b) for b in bands], dtype=float)
    lever = float(r.max() - r.min())

    sig_tot = np.sqrt(sig**2 + sys_floor_mag**2)
    cov = np.diag(sig_tot**2)
    if dist_modulus_sigma and dist_modulus_sigma > 0:
        # Fully correlated across bands: this *is* the grey direction.
        cov = cov + (dist_modulus_sigma**2) * np.ones((n, n))

    x_design = np.column_stack([np.ones(n), r])

    if av_prior is not None:
        av0, av_sig = float(av_prior[0]), float(av_prior[1])
        if av_sig > 0:
            # Pseudo-observation row: 0*g + 1*A_V = av0 with σ = av_sig.
            x_design = np.vstack([x_design, [0.0, 1.0]])
            y = np.append(y, av0)
            cov_p = np.zeros((n + 1, n + 1))
            cov_p[:n, :n] = cov
            cov_p[n, n] = av_sig**2
            cov = cov_p

    n_eff = x_design.shape[0]
    theta, pcov, chi2 = _gls(x_design, y, cov)
    grey, av = float(theta[0]), float(theta[1])
    var_g, var_a = float(pcov[0, 0]), float(pcov[1, 1])
    cov_ga = float(pcov[0, 1])
    denom = math.sqrt(var_g * var_a)
    rho = cov_ga / denom if denom > 0 else float("nan")

    notes: list[str] = []
    verdict = "ok"
    if lever < MIN_LEVER_ARM:
        verdict = "degenerate_band_set"
        notes.append(
            f"lever arm max(R)-min(R) = {lever:.3f} < {MIN_LEVER_ARM}: this band set "
            "cannot separate grey from reddening; a blue band is required"
        )
    if math.isfinite(rho) and abs(rho) > MAX_ABS_RHO:
        verdict = "degenerate_band_set"
        notes.append(f"|rho(g, A_V)| = {abs(rho):.4f} > {MAX_ABS_RHO}")

    av_pinned = False
    # Tolerance so an exact-zero fit is not treated as a boundary case; a
    # genuinely negative A_V is unphysical, a -1e-17 one is floating point.
    if enforce_av_nonneg and av < -1e-9:
        floor = 0.0
        if av_prior is not None:
            floor = max(0.0, float(av_prior[0]) - 3.0 * float(av_prior[1]))
        # Refit with A_V pinned: subtract the fixed reddening, fit grey alone.
        x_g = x_design[:, :1]
        y_pin = y - x_design[:, 1] * floor
        theta_p, pcov_p, chi2 = _gls(x_g, y_pin, cov)
        grey = float(theta_p[0])
        var_g = float(pcov_p[0, 0])
        av, var_a, rho = floor, 0.0, 0.0
        av_pinned = True
        notes.append(f"unconstrained A_V < 0 (unphysical); pinned at {floor:.3f}")
        dof = n_eff - 1
    else:
        dof = n_eff - 2

    grey_err = math.sqrt(max(var_g, 0.0))
    av_err = math.sqrt(max(var_a, 0.0))

    # Nested-model test: how much does the grey term buy over reddening alone?
    # For a linear model this equals (g/σ_g)², computed independently as a check.
    x_red = x_design[:, 1:]
    _, _, chi2_red = _gls(x_red, y, cov)
    dchi2 = float(chi2_red - chi2)

    if dof > 0 and chi2 / dof > 6.0:
        # The SED is not describable as grey + a standard reddening law. Either
        # the twin match is wrong, the photometry is blended, or there is real
        # spectral structure (an excess). Not a grey-occulter candidate.
        if verdict == "ok":
            verdict = "bad_sed_fit"
        notes.append(f"chi2/dof = {chi2 / dof:.1f} > 6: SED not grey+R_V=3.1 reddening")

    return GreyFit(
        grey_mag=grey, grey_err=grey_err, av=av, av_err=av_err, rho_grey_av=rho,
        chi2=chi2, dof=int(dof), n_bands=n, bands=tuple(bands), lever_arm=lever,
        av_at_boundary=av_pinned, delta_chi2_grey=dchi2, verdict=verdict, notes=notes,
    )


def grey_significance_floor(
    twin_scatter_mag: float, parallax_over_error: float, n_twins: int = 50,
    phot_err_mag: float = 0.02,
) -> float:
    """The 1σ floor on ``g`` for a star — i.e. what this channel can *never* beat.

    Three irreducible terms:

    * the intrinsic spread of the twin population at fixed parameters
      (``twin_scatter``), which the target draws from once;
    * the uncertainty of the twin median itself (``twin_scatter/√N``);
    * the distance modulus, ``5/ln10 · σ_ϖ/ϖ``, which is *exactly* grey.

    At ``parallax_over_error = 20`` the distance term alone is 0.109 mag, which
    is comparable to the whole signal at f = 0.10 (g = 0.114). This function
    exists so that the sensitivity claim in the docs is computed, not asserted.
    """
    sig_dist = (5.0 / math.log(10.0)) / max(parallax_over_error, 1e-9)
    sig_twin = twin_scatter_mag * math.sqrt(1.0 + 1.0 / max(n_twins, 1))
    return math.sqrt(sig_dist**2 + sig_twin**2 + phot_err_mag**2)


def minimum_detectable_f(
    twin_scatter_mag: float, parallax_over_error: float, n_sigma: float = 3.0,
    n_twins: int = 50, phot_err_mag: float = 0.02,
) -> float:
    """Smallest covering fraction detectable at ``n_sigma`` for one star."""
    g = n_sigma * grey_significance_floor(
        twin_scatter_mag, parallax_over_error, n_twins, phot_err_mag
    )
    return covering_fraction_from_grey(g)


__all__ = [
    "DEFAULT_SYS_FLOOR_MAG",
    "GreyFit",
    "fit_grey_reddening",
    "grey_significance_floor",
    "minimum_detectable_f",
]

"""The validation target: Griffith et al. 2021's fifteen Na-enhanced stars.

Why this module exists
----------------------
A sparse-anomaly statistic that cannot recover a *known, published* sparse
anomaly is not a detector, it is a threshold. There is exactly one published
population with the right morphology, and it is the one this channel cites as
its proof of concept — Griffith, Weinberg, Buder et al. 2021
(arXiv:2110.06240), GALAH+ DR3, 82,910 disk stars, verbatim from the abstract:

    "As one example of a population with distinctive abundance patterns, we
    identify **15 stars that have 0.3-0.6 dex enhancements of Na but normal
    abundances of other elements from O to Ni** and positive average residuals
    of Cu, Zn, Y, and Ba."

Those fifteen stars were found *incidentally*, by eye, while inspecting large
two-process residuals. TAILINGS claims to find that morphology **on purpose**.
The claim is only worth making if the statistic demonstrably recovers it, so
this module injects the Griffith signature into a GALAH-DR3-like synthetic
population and measures the recovery fraction with the channel's own detection
code — ``manifold.to_xh`` -> ``manifold.fit_manifold`` -> ``manifold.zscores``
-> ``sparse.sparse_statistics`` — at the shipped thresholds.

The harness is deliberately *not* allowed to tune anything to make the answer
come out right. If the recovery fraction is low, that is a result about the
statistic, and it is reported as one.

What is modelled, and why each piece is there
---------------------------------------------
* **Abundances are generated in [X/H] and published as [X/Fe].** The observed
  ``[X/Fe] = [X/H]_true - [Fe/H]_observed + noise_X``, with ``[Fe/H]_observed``
  carrying its own measurement error. This reproduces Weinberg's *measurement
  aberration* exactly: a single error in the star's own iron abundance moves
  every published ``[X/Fe]`` coherently. :func:`aberration_comparison` measures
  what that costs, and returns a result the channel's own docstrings do not
  predict — ``[X/H]`` and ``[X/Fe]`` give **identical** residuals here, because
  ``[Fe/H]`` is a linear column of the manifold's design matrix. What actually
  protects this pipeline is *regressing on* ``[Fe/H]``, not the ``to_xh``
  conversion. The aberration is real and costly in ``[X/Mg]``, where the
  normaliser is **not** a predictor and its error therefore survives into every
  element's residual with the same sign.
* **A low-dimensional chemical core.** One latent alpha axis (anti-correlated
  with [Fe/H]) and one latent s-process axis shared by Sr/Y/Zr and Ba/La/Ce, so
  "natural processes move a family" is a property of the synthetic data rather
  than an assumption of the test.
* **Per-element noise with a systematic floor and a photon term**,
  ``sigma_X = sqrt(floor_X^2 + (k_X/SNR)^2)``, so the empirical scatter table
  measured in (SNR, Teff) bins has something real to measure. The floors are
  set so the population residual RMS reproduces Griffith's stated
  "RMS residuals <~ 0.07 dex for well-measured elements".
* **Realistic per-element missingness.** GALAH DR3 does not measure thirty
  elements in every dwarf. The number of *measured and quiet* elements is the
  evidence in this channel, so getting the element count right is not cosmetic.
* **Unflagged bad measurements.** Griffith inspected their own large residuals
  and concluded "roughly 40% of the large deviations are physical and 60% are
  caused by problematic data". A fraction of measurements are therefore drawn
  from an inflated distribution; most carry a per-element quality flag (removed
  by :func:`~seti.tailings.acquire.apply_element_flags`, the real code) and the
  rest survive as the false-positive background.

Schema note
-----------
The canonical dataframe is the one ``acquire.normalize`` emits: ``star_id, ra,
dec, teff, logg, fe_h, snr, chi2, ruwe, vbroad, rv_scatter, rv, fiber,
field_id, survey`` plus ``<El>``, ``e_<El>`` and the per-element quality flag.
The flag column is written ``f_<El>``, which is what
``acquire.apply_element_flags`` consumes; ``flag_<el>_fe`` is GALAH's *raw*
name and is mapped onto ``f_<El>`` by ``acquire.resolve_abundance_columns``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import manifold as M
from . import sparse as S
from . import vet as V
from .acquire import apply_element_flags

#: The published target, quoted so a reader never has to trust a paraphrase.
GRIFFITH_QUOTE = (
    "we identify 15 stars that have 0.3-0.6 dex enhancements of Na but normal "
    "abundances of other elements from O to Ni and positive average residuals "
    "of Cu, Zn, Y, and Ba"
)
GRIFFITH_REF = "Griffith, Weinberg, Buder et al. 2021, arXiv:2110.06240 (GALAH+ DR3)"

#: The Griffith 15 are Na-enhanced by this much, in dex.
GRIFFITH_AMPLITUDE_RANGE: tuple[float, float] = (0.3, 0.6)
GRIFFITH_N_STARS: int = 15

#: "normal abundances of other elements from O to Ni" — the elements that must
#: stay untouched for the injection to be the Griffith signature rather than a
#: generic metal-rich star. Atomic number 8 through 28.
O_THROUGH_NI: tuple[str, ...] = (
    "O", "Na", "Mg", "Al", "Si", "P", "S", "K", "Ca", "Sc", "Ti", "TiII",
    "V", "Cr", "Mn", "Co", "Ni",
)

#: The secondary detail of the Griffith morphology: "positive average residuals
#: of Cu, Zn, Y, and Ba". Small, and an *average* over the fifteen rather than a
#: per-star statement, so it is off by default and exercised as a variant.
GRIFFITH_HEAVY_ELEMENTS: tuple[str, ...] = ("Cu", "Zn", "Y", "Ba")


# ---------------------------------------------------------------------------
# The element table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ElementSpec:
    """Generative parameters for one element of the synthetic population.

    ``[X/H] = a0 + (1 + b_feh) * [Fe/H] + c_alpha * alpha + s_coef * s_axis
              + d_teff * (Teff - 5200)/1000 + g_logg * (logg - 4.35)
              + q_feh * [Fe/H]^2 + intrinsic + noise``

    The Teff and log g terms are not astrophysics: they are the pipeline
    systematic that ``manifold.fit_element`` exists to regress out, and putting
    them in means the manifold has to earn its residuals.
    """

    name: str
    b_feh: float = 0.0
    c_alpha: float = 0.0
    s_coef: float = 0.0
    d_teff: float = 0.0
    g_logg: float = 0.0
    q_feh: float = 0.0
    sigma_floor: float = 0.04
    """Systematic residual floor in dex: line list, NLTE, continuum, and the
    genuine star-to-star chemical individuality that survives conditioning."""
    sigma_photon: float = 2.0
    """Photon term coefficient: the noise contribution is ``sigma_photon/SNR``."""
    sigma_intrinsic_family: float = 0.02
    """Amplitude of the family-correlated individuality component."""
    measured_frac: float = 0.95
    """Fraction of stars with a usable measurement of this element."""
    err_factor: float = 0.85
    """Catalogue errors are formal fit errors and are known to be optimistic
    (see ``manifold``); the reported ``e_<El>`` is this multiple of the truth."""


#: A GALAH-DR3-like element set. Floors are chosen so that the *population*
#: residual RMS lands at Griffith's "<~ 0.07 dex for well-measured elements",
#: and the ordering (alpha and Fe-peak tight; Cu/Zn/Ba/n-capture loose) follows
#: the per-element caveat table in ``vet.ELEMENT_CAVEATS``.
GALAH_DR3_SPECS: tuple[ElementSpec, ...] = (
    # --- alpha ---
    ElementSpec("O",   b_feh=-0.35, c_alpha=0.9, d_teff=0.035, g_logg=0.04,
                sigma_floor=0.055, sigma_photon=2.5, measured_frac=0.85),
    ElementSpec("Mg",  b_feh=-0.30, c_alpha=1.0, d_teff=0.015, g_logg=0.02,
                sigma_floor=0.030, sigma_photon=1.2, measured_frac=0.99),
    ElementSpec("Si",  b_feh=-0.25, c_alpha=0.9, d_teff=0.012, g_logg=0.02,
                sigma_floor=0.028, sigma_photon=1.1, measured_frac=0.99),
    ElementSpec("Ca",  b_feh=-0.22, c_alpha=0.8, d_teff=0.018, g_logg=0.03,
                sigma_floor=0.032, sigma_photon=1.3, measured_frac=0.98),
    ElementSpec("Ti",  b_feh=-0.25, c_alpha=0.85, d_teff=0.022, g_logg=0.03,
                sigma_floor=0.035, sigma_photon=1.5, measured_frac=0.98),
    ElementSpec("TiII", b_feh=-0.25, c_alpha=0.85, d_teff=0.020, g_logg=0.05,
                sigma_floor=0.040, sigma_photon=1.8, measured_frac=0.95),
    # --- odd-Z (Na is the element the whole exercise turns on) ---
    ElementSpec("Na",  b_feh=0.10, c_alpha=0.15, d_teff=0.030, g_logg=0.04,
                q_feh=0.10, sigma_floor=0.045, sigma_photon=2.0,
                measured_frac=0.95),
    ElementSpec("Al",  b_feh=0.05, c_alpha=0.5, d_teff=0.025, g_logg=0.03,
                sigma_floor=0.040, sigma_photon=1.6, measured_frac=0.93),
    ElementSpec("K",   b_feh=-0.10, c_alpha=0.4, d_teff=0.040, g_logg=0.05,
                sigma_floor=0.060, sigma_photon=2.5, measured_frac=0.80),
    ElementSpec("Sc",  b_feh=-0.05, c_alpha=0.4, d_teff=0.020, g_logg=0.03,
                sigma_floor=0.042, sigma_photon=1.7, measured_frac=0.92),
    # --- Fe-peak ---
    ElementSpec("V",   b_feh=0.02, c_alpha=0.25, d_teff=0.030, g_logg=0.04,
                sigma_floor=0.060, sigma_photon=2.6, measured_frac=0.70),
    ElementSpec("Cr",  b_feh=0.01, c_alpha=0.05, d_teff=0.018, g_logg=0.02,
                sigma_floor=0.035, sigma_photon=1.5, measured_frac=0.95),
    ElementSpec("Mn",  b_feh=0.20, c_alpha=-0.20, d_teff=0.020, g_logg=0.03,
                sigma_floor=0.038, sigma_photon=1.6, measured_frac=0.96),
    ElementSpec("Co",  b_feh=0.05, c_alpha=0.20, d_teff=0.028, g_logg=0.04,
                sigma_floor=0.055, sigma_photon=2.4, measured_frac=0.78),
    ElementSpec("Ni",  b_feh=0.03, c_alpha=0.15, d_teff=0.012, g_logg=0.02,
                sigma_floor=0.030, sigma_photon=1.2, measured_frac=0.97),
    ElementSpec("Cu",  b_feh=0.25, c_alpha=0.10, d_teff=0.035, g_logg=0.05,
                sigma_floor=0.070, sigma_photon=3.0, measured_frac=0.75),
    ElementSpec("Zn",  b_feh=-0.10, c_alpha=0.35, s_coef=0.25, d_teff=0.030,
                g_logg=0.04, sigma_floor=0.065, sigma_photon=2.8,
                measured_frac=0.80),
    # --- s-process, light and heavy ---
    ElementSpec("Sr",  b_feh=-0.05, s_coef=0.7, d_teff=0.040, g_logg=0.06,
                sigma_floor=0.085, sigma_photon=3.5, measured_frac=0.25),
    ElementSpec("Y",   b_feh=-0.05, s_coef=0.8, d_teff=0.030, g_logg=0.04,
                sigma_floor=0.060, sigma_photon=2.6, measured_frac=0.75),
    ElementSpec("Zr",  b_feh=-0.05, s_coef=0.7, d_teff=0.038, g_logg=0.05,
                sigma_floor=0.080, sigma_photon=3.2, measured_frac=0.30),
    ElementSpec("Ba",  b_feh=-0.10, s_coef=1.0, d_teff=0.035, g_logg=0.06,
                sigma_floor=0.070, sigma_photon=2.5, measured_frac=0.90),
    ElementSpec("La",  b_feh=-0.05, s_coef=0.9, d_teff=0.040, g_logg=0.05,
                sigma_floor=0.085, sigma_photon=3.5, measured_frac=0.35),
    ElementSpec("Ce",  b_feh=-0.05, s_coef=0.9, d_teff=0.042, g_logg=0.05,
                sigma_floor=0.090, sigma_photon=3.5, measured_frac=0.40),
    # --- r-process / mixed ---
    ElementSpec("Nd",  b_feh=0.00, s_coef=0.5, d_teff=0.040, g_logg=0.05,
                sigma_floor=0.090, sigma_photon=3.5, measured_frac=0.30),
    ElementSpec("Sm",  b_feh=0.00, s_coef=0.4, d_teff=0.045, g_logg=0.05,
                sigma_floor=0.100, sigma_photon=4.0, measured_frac=0.20),
    ElementSpec("Eu",  b_feh=-0.20, c_alpha=0.5, d_teff=0.035, g_logg=0.05,
                sigma_floor=0.075, sigma_photon=3.0, measured_frac=0.45),
    # --- naturally sparse: excluded from the statistic by construction ---
    ElementSpec("Li",  d_teff=0.30, g_logg=0.10, sigma_floor=0.060,
                sigma_photon=2.5, sigma_intrinsic_family=0.45,
                measured_frac=0.55),
    ElementSpec("C",   b_feh=-0.05, d_teff=0.030, g_logg=0.05,
                sigma_floor=0.050, sigma_photon=2.0, measured_frac=0.60),
    ElementSpec("N",   b_feh=0.05, d_teff=0.040, g_logg=0.06,
                sigma_floor=0.060, sigma_photon=2.5, measured_frac=0.30),
    ElementSpec("Be",  d_teff=0.10, sigma_floor=0.100, sigma_photon=4.0,
                sigma_intrinsic_family=0.30, measured_frac=0.20),
    ElementSpec("B",   d_teff=0.10, sigma_floor=0.120, sigma_photon=4.5,
                sigma_intrinsic_family=0.30, measured_frac=0.15),
)

#: Latent-axis membership, so a family genuinely co-moves in the synthetic data.
_FAMILY_AXIS = {
    "alpha": "alpha",
    "odd_z": "oddz",
    "fe_peak": "fepeak",
    "s_light": "sproc",
    "s_heavy": "sproc",
    "r_mixed": "rproc",
    "r_process": "rproc",
    "cno": "cno",
    "light": "light",
}


# ---------------------------------------------------------------------------
# Population synthesis
# ---------------------------------------------------------------------------
def synthesise_population(
    n: int = 6000,
    *,
    rng: np.random.Generator | None = None,
    seed: int = 20260726,
    specs: tuple[ElementSpec, ...] = GALAH_DR3_SPECS,
    feh_error_floor: float = 0.020,
    feh_error_photon: float = 1.0,
    bad_measurement_frac: float = 6.0e-4,
    bad_measurement_inflation: float = 4.0,
    bad_measurement_flagged_frac: float = 0.6,
    survey: str = "GALAH",
) -> pd.DataFrame:
    """A GALAH-DR3-like cool-dwarf catalogue in the canonical TAILINGS schema.

    The sample box is the channel's own: ``4000 < Teff < 6000 K``,
    ``log g > 4.0``, ``SNR >= 40``, ``[Fe/H] >= -1.0`` (``config/thresholds.yaml``).
    Abundances come back as ``[X/Fe]``, exactly as a survey publishes them, with
    the iron-normalisation aberration built in.
    """
    rng = rng or np.random.default_rng(seed)

    teff = rng.uniform(4100.0, 5950.0, n)
    logg = rng.uniform(4.05, 4.65, n)
    feh_true = np.clip(rng.normal(-0.08, 0.24, n), -0.98, 0.48)
    snr = 10 ** rng.uniform(np.log10(40.0), np.log10(320.0), n)

    # One latent alpha axis and one latent s-process axis: this is what makes
    # "a natural process moves a family" true of the data and not just of the
    # docstring. The alpha axis is bimodal because the disk is -- a thick-disk
    # sequence offset by ~0.15 dex at fixed [Fe/H] -- and that genuine spread is
    # what keeps the alpha proxy from being a pure restatement of [Fe/H].
    thick = rng.random(n) < 0.18
    alpha = np.clip(
        -0.22 * feh_true + 0.14 * thick + rng.normal(0.0, 0.045, n), -0.10, 0.55
    )
    s_axis = rng.normal(0.0, 0.09, n)

    # Family-correlated chemical individuality (Weinberg et al. find the
    # residual noise itself is element-correlated).
    # sorted(), not set(): set iteration order depends on PYTHONHASHSEED, which
    # would make the whole population -- and therefore every recovery number in
    # this module -- irreproducible between processes at a fixed seed.
    axes = {a: rng.normal(0.0, 1.0, n) for a in sorted(set(_FAMILY_AXIS.values()))}

    # The star's own [Fe/H] carries measurement error, and that error is what
    # propagates into every published [X/Fe] at once.
    feh_err = np.sqrt(feh_error_floor**2 + (feh_error_photon / snr) ** 2)
    feh_obs = feh_true + rng.normal(0.0, 1.0, n) * feh_err

    df = pd.DataFrame(
        {
            "star_id": [f"SYN{i:07d}" for i in range(n)],
            "ra": rng.uniform(0.0, 360.0, n),
            "dec": np.degrees(np.arcsin(rng.uniform(-1.0, 0.35, n))),
            "teff": teff,
            "logg": logg,
            "fe_h": feh_obs,
            "snr": snr,
            "chi2": rng.gamma(20.0, 0.05, n),
            "ruwe": 0.95 + rng.gamma(2.0, 0.06, n),
            "vbroad": rng.uniform(2.0, 9.0, n),
            "rv_scatter": rng.gamma(2.0, 0.12, n),
            "rv": rng.normal(0.0, 38.0, n),
            "fiber": rng.integers(1, 401, n),
            "field_id": rng.integers(0, 60, n),
            "survey": survey,
        }
    )
    df["e_fe_h"] = feh_err

    for spec in specs:
        fam = M.element_family(spec.name)
        axis = axes[_FAMILY_AXIS.get(fam, "alpha")]
        # Truth in [X/H]: this is the space the manifold works in.
        xh = (
            (1.0 + spec.b_feh) * feh_true
            + spec.c_alpha * alpha
            + spec.s_coef * s_axis
            + spec.q_feh * feh_true**2
            + spec.d_teff * (teff - 5200.0) / 1000.0
            + spec.g_logg * (logg - 4.35)
            + spec.sigma_intrinsic_family * axis
        )
        sigma = np.sqrt(spec.sigma_floor**2 + (spec.sigma_photon / snr) ** 2)
        noise = rng.normal(0.0, 1.0, n) * sigma

        # Unflagged and flagged bad measurements: Griffith inspected their own
        # large residuals and found ~60% were data problems, not stars.
        bad = rng.random(n) < bad_measurement_frac
        noise = noise + bad * rng.normal(0.0, 1.0, n) * sigma * bad_measurement_inflation
        flagged = bad & (rng.random(n) < bad_measurement_flagged_frac)

        # Published quantity: [X/Fe] against the *observed* iron abundance.
        df[spec.name] = xh - feh_obs + noise
        df[f"e_{spec.name}"] = sigma * spec.err_factor
        df[f"f_{spec.name}"] = flagged.astype(int)

        missing = rng.random(n) >= spec.measured_frac
        df.loc[missing, spec.name] = np.nan
        df.loc[missing, f"e_{spec.name}"] = np.nan

    df.attrs["injected_ids"] = []
    return apply_element_flags(df, [s.name for s in specs])


# ---------------------------------------------------------------------------
# Injections
# ---------------------------------------------------------------------------
def inject_griffith_na(
    df: pd.DataFrame,
    *,
    n_stars: int = GRIFFITH_N_STARS,
    amplitude_range: tuple[float, float] = GRIFFITH_AMPLITUDE_RANGE,
    rng: np.random.Generator | None = None,
    seed: int = 4242,
    heavy_residual_dex: float = 0.0,
    element: str = "Na",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject the Griffith morphology: ``element`` only, +0.3 to +0.6 dex.

    Every other element is left exactly as synthesised, so "normal abundances
    of other elements from O to Ni" is true by construction rather than by
    accident. ``heavy_residual_dex`` optionally adds the secondary detail of the
    published morphology — "positive average residuals of Cu, Zn, Y, and Ba" —
    which is an average over the fifteen, not a per-star claim, and is therefore
    off by default.

    Returns ``(df, injection_table)``. The injection table carries the star id,
    the amplitude in dex, and which stars were eligible.
    """
    rng = rng or np.random.default_rng(seed)
    out = df.copy()

    # Only stars that actually have the element measured can carry the signal —
    # injecting into a NaN would silently reduce the denominator.
    eligible = np.flatnonzero(out[element].notna().to_numpy())
    if eligible.size < n_stars:
        raise ValueError(f"only {eligible.size} stars have {element} measured")
    idx = rng.choice(eligible, size=n_stars, replace=False)
    amps = rng.uniform(amplitude_range[0], amplitude_range[1], n_stars)

    col = out.columns.get_loc(element)
    for k, i in enumerate(idx):
        out.iat[i, col] = float(out.iat[i, col]) + float(amps[k])
        if heavy_residual_dex:
            for el in GRIFFITH_HEAVY_ELEMENTS:
                if el in out.columns and np.isfinite(out.at[out.index[i], el]):
                    out.at[out.index[i], el] = (
                        float(out.at[out.index[i], el]) + heavy_residual_dex
                    )

    table = pd.DataFrame(
        {
            "star_id": out["star_id"].to_numpy()[idx],
            "row": idx,
            "element": element,
            "amplitude_dex": amps,
            "snr": out["snr"].to_numpy()[idx],
            "teff": out["teff"].to_numpy()[idx],
            "fe_h": out["fe_h"].to_numpy()[idx],
        }
    )
    out.attrs["injected_ids"] = list(table["star_id"])
    return out, table


def inject_dense_control(
    df: pd.DataFrame,
    *,
    n_stars: int = 15,
    amplitude_dex: float = 0.40,
    elements: tuple[str, ...] = O_THROUGH_NI,
    rng: np.random.Generator | None = None,
    seed: int = 909,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The negative control: raise **all** of O-through-Ni together.

    This is a metal-rich star, or a mis-set continuum, or a wrong [Fe/H] — a
    *dense* anomaly, which is exactly what this channel is built to reject.
    A statistic that flags this is an amplitude detector wearing a sparsity
    costume.
    """
    rng = rng or np.random.default_rng(seed)
    out = df.copy()
    pool = np.flatnonzero(out["Mg"].notna().to_numpy() & out["Ni"].notna().to_numpy())
    idx = rng.choice(pool, size=n_stars, replace=False)
    present = [e for e in elements if e in out.columns]
    for i in idx:
        for el in present:
            v = out.at[out.index[i], el]
            if np.isfinite(v):
                out.at[out.index[i], el] = float(v) + amplitude_dex
    table = pd.DataFrame(
        {
            "star_id": out["star_id"].to_numpy()[idx],
            "row": idx,
            "amplitude_dex": amplitude_dex,
            "n_elements_raised": len(present),
        }
    )
    return out, table


def inject_light_element_control(
    df: pd.DataFrame,
    *,
    n_stars: int = 15,
    amplitude_dex: float = 1.20,
    elements: tuple[str, ...] = ("Li", "Be", "B"),
    rng: np.random.Generator | None = None,
    seed: int = 1717,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The a-priori-exclusion control: a huge, genuinely single-element Li/Be/B spike.

    Lithium varies by orders of magnitude among otherwise identical cool dwarfs
    through convective depletion — Sun et al. measure 0.35-0.6 dex of intrinsic
    scatter at fixed parameters, and Spina et al. document a co-natal pair
    differing by 1.9 dex. A known single-element variable cannot be evidence for
    an unknown one, so Li, Be and B are excluded from the statistic *by
    construction* (``manifold.NATURALLY_SPARSE_ELEMENTS``), and this control
    checks that the exclusion is real rather than a label.
    """
    rng = rng or np.random.default_rng(seed)
    out = df.copy()
    present = [e for e in elements if e in out.columns]
    if not present:
        return out, pd.DataFrame(columns=["star_id", "row", "amplitude_dex"])
    pool = np.arange(len(out))
    idx = rng.choice(pool, size=n_stars, replace=False)
    for i in idx:
        for el in present:
            v = out.at[out.index[i], el]
            if not np.isfinite(v):
                # Force a measurement so the spike is actually testable.
                out.at[out.index[i], el] = 0.0
                out.at[out.index[i], f"e_{el}"] = 0.10
                v = 0.0
            out.at[out.index[i], el] = float(v) + amplitude_dex
    table = pd.DataFrame(
        {
            "star_id": out["star_id"].to_numpy()[idx],
            "row": idx,
            "amplitude_dex": amplitude_dex,
            "elements": ",".join(present),
        }
    )
    return out, table


# ---------------------------------------------------------------------------
# Detection — the channel's real statistic, nothing reimplemented
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """Everything the detection stage produced, keyed by the input row order."""

    elements: list[str]
    manifold: M.Manifold
    Z: pd.DataFrame
    sigma: pd.DataFrame
    stats: pd.DataFrame
    joined: pd.DataFrame
    flag_rates: pd.DataFrame


def detect(
    stars: pd.DataFrame,
    *,
    sparse_cfg: S.SparseConfig | None = None,
    in_xh: bool = True,
    min_rows: int = 200,
    degree: int = 2,
    clip: float = 4.0,
    n_iter: int = 4,
    scatter_min_count: int = 40,
    scatter_floor: float = 0.005,
) -> Detection:
    """Run the channel's detection statistic, mirroring ``run.reduce_survey``.

    ``in_xh=True`` is the shipped behaviour: convert the published ``[X/Fe]`` to
    ``[X/H]`` *before* fitting, so the residuals are in the space the channel
    reports. ``in_xh=False`` exists so :func:`aberration_comparison` can measure
    what that choice is worth — and the measured answer is "nothing, given that
    ``[Fe/H]`` is already a predictor", which is worth knowing before the claim
    is written up.
    """
    cfg = sparse_cfg or S.SparseConfig()
    elements = [c for c in stars.columns if M.element_family(c) != "other"]
    elements = [e for e in elements if int(stars[e].notna().sum()) >= min_rows]

    frame = M.to_xh(stars, elements, feh_col="fe_h") if in_xh else stars
    with warnings.catch_warnings():
        # A star with no alpha element measured yields an all-NaN leave-one-out
        # proxy; manifold.alpha_proxy already documents that it returns NaN and
        # leaves the decision to the caller, so the numpy warning is noise.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return _fit_and_score(frame, stars, elements, cfg, degree, clip, n_iter,
                              min_rows, scatter_min_count, scatter_floor)


def _fit_and_score(frame, stars, elements, cfg, degree, clip, n_iter, min_rows,
                   scatter_min_count, scatter_floor) -> Detection:
    mani = M.fit_manifold(
        frame,
        elements,
        teff_col="teff",
        logg_col="logg",
        feh_col="fe_h",
        snr_col="snr",
        degree=degree,
        clip=clip,
        n_iter=n_iter,
        min_rows=min_rows,
        min_count=scatter_min_count,
        floor=scatter_floor,
    )
    Z, sig = M.zscores(frame, mani, err_prefix="e_")
    stats = S.sparse_statistics(Z, cfg=cfg)
    rates = S.element_flag_rates(stats, Z, cfg=cfg)

    keep = [c for c in ("star_id", "ra", "dec", "teff", "logg", "fe_h", "snr",
                        "chi2", "ruwe", "vbroad", "rv_scatter", "rv", "fiber",
                        "field_id", "survey") if c in stars.columns]
    joined = pd.concat(
        [stars[keep].reset_index(drop=True), stats.reset_index(drop=True)], axis=1
    )
    for el in Z.columns:
        joined[f"z_{el}"] = Z[el].to_numpy()
    return Detection(
        elements=elements, manifold=mani, Z=Z, sigma=sig, stats=stats,
        joined=joined, flag_rates=rates,
    )


# ---------------------------------------------------------------------------
# The validation harness
# ---------------------------------------------------------------------------
def _recovery_rows(det: Detection, inj: pd.DataFrame, element: str) -> list[dict]:
    rows = []
    for _, r in inj.iterrows():
        k = int(r["row"])
        st = det.stats.iloc[k]
        z_el = float(det.Z[element].iloc[k]) if element in det.Z.columns else float("nan")
        rows.append(
            {
                "star_id": str(r["star_id"]),
                "amplitude_dex": round(float(r["amplitude_dex"]), 4),
                "snr": round(float(r["snr"]), 1),
                f"z_{element}": round(z_el, 3),
                "sigma_used_dex": round(float(det.sigma[element].iloc[k]), 4)
                if element in det.sigma.columns else None,
                "z_max": round(float(st["z_max"]), 3),
                "element_max": st["element_max"],
                "n_elements": int(st["n_elements"]),
                "n_active": int(st["n_active"]),
                "n_quiet": int(st["n_quiet"]),
                "contrast": round(float(st["contrast"]), 3),
                "classification": str(st["classification"]),
                "reason": str(st["reason"]),
                "recovered": bool(
                    st["classification"] == S.SPARSE and st["element_max"] == element
                ),
            }
        )
    return rows


def _miss_breakdown(rows: list[dict], element: str, z_flag: float) -> dict[str, int]:
    """Why each missed injection was missed. This is the diagnostic that matters."""
    out = {
        "amplitude_below_threshold": 0,
        "background_not_quiet": 0,
        "family_co_moves": 0,
        "contrast_too_low": 0,
        "too_few_elements": 0,
        "carried_by_another_element": 0,
        "other": 0,
    }
    for r in rows:
        if r["recovered"]:
            continue
        z = r.get(f"z_{element}")
        if z is not None and np.isfinite(z) and abs(z) < z_flag:
            out["amplitude_below_threshold"] += 1
            continue
        if r["element_max"] != element:
            out["carried_by_another_element"] += 1
            continue
        reason = r["reason"]
        if "not quiet" in reason:
            out["background_not_quiet"] += 1
        elif "family co-moves" in reason:
            out["family_co_moves"] += 1
        elif reason.startswith("contrast"):
            out["contrast_too_low"] += 1
        elif "measured elements" in reason:
            out["too_few_elements"] += 1
        else:
            out["other"] += 1
    return out


#: Threshold sets re-scored alongside the shipped one. The first is the
#: shipped configuration's two binding constraints relaxed to what the
#: injection-recovery measurement says they should be: ``z_flag`` 6 -> 5,
#: because ``6 * sigma_Na ~ 0.35 dex`` cuts off the bottom third of Griffith's
#: published 0.3-0.6 dex range by arithmetic; and ``max_quiet_excess`` 1 -> 3,
#: because that rule is an absolute count applied to a vector of ~20 measured
#: elements in which ~0.9 elements exceed ``z_quiet`` by chance.
DEFAULT_ALT_CONFIGS: tuple[dict[str, Any], ...] = (
    {"z_flag": 5.0, "max_quiet_excess": 3},
)


def _rescore(det: Detection, cfg: S.SparseConfig, inj_rows, is_field, element: str) -> dict:
    stats = S.sparse_statistics(det.Z, cfg=cfg)
    lab = stats["classification"].to_numpy()
    emax = stats["element_max"].to_numpy()
    rec = int(sum(1 for k in inj_rows if lab[k] == S.SPARSE and emax[k] == element))
    fp = int(((lab == S.SPARSE) & is_field).sum())
    return {
        "z_flag": cfg.z_flag,
        "z_quiet": cfg.z_quiet,
        "max_quiet_excess": cfg.max_quiet_excess,
        "min_contrast": cfg.min_contrast,
        "n_recovered": rec,
        "n_injected": len(inj_rows),
        "fraction": round(rec / max(len(inj_rows), 1), 4),
        "n_false_positive": fp,
        "false_positive_rate": round(fp / max(int(is_field.sum()), 1), 6),
    }


def validate_griffith(
    *,
    seed: int = 20260726,
    n_field: int = 6000,
    n_injected: int = GRIFFITH_N_STARS,
    amplitude_range: tuple[float, float] = GRIFFITH_AMPLITUDE_RANGE,
    element: str = "Na",
    heavy_residual_dex: float = 0.0,
    sparse_cfg: S.SparseConfig | None = None,
    with_controls: bool = True,
    n_control: int = 15,
    run_vet: bool = True,
    specs: tuple[ElementSpec, ...] = GALAH_DR3_SPECS,
    alt_configs: tuple[dict[str, Any], ...] = DEFAULT_ALT_CONFIGS,
) -> dict[str, Any]:
    """Inject the Griffith 15 and measure what the channel's statistic recovers.

    Returns a JSON-serialisable dict suitable for dropping into
    ``results/tailings/summary.json`` under a ``validation`` key.

    The verdict is **not** softened if the recovery is poor. Per the standing
    instruction that motivates this module: if the pipeline cannot recover the
    published population, the statistic is wrong, and the harness says so.
    """
    cfg = sparse_cfg or S.SparseConfig()
    rng = np.random.default_rng(seed)

    df = synthesise_population(n_field, rng=rng, specs=specs)
    n_pre = len(df)
    df, inj = inject_griffith_na(
        df,
        n_stars=n_injected,
        amplitude_range=amplitude_range,
        rng=rng,
        heavy_residual_dex=heavy_residual_dex,
        element=element,
    )
    dense_tab = pd.DataFrame()
    light_tab = pd.DataFrame()
    if with_controls:
        df, dense_tab = inject_dense_control(df, n_stars=n_control, rng=rng)
        df, light_tab = inject_light_element_control(df, n_stars=n_control, rng=rng)

    det = detect(df, sparse_cfg=cfg)

    # --- recovery of the injected fifteen -------------------------------------
    rows = _recovery_rows(det, inj, element)
    n_rec = int(sum(r["recovered"] for r in rows))

    # --- false positives among everything that was NOT injected ---------------
    control_rows = set()
    for tab in (dense_tab, light_tab):
        if len(tab):
            control_rows.update(int(v) for v in tab["row"])
    injected_rows = {int(v) for v in inj["row"]}
    is_field = np.ones(len(df), dtype=bool)
    for k in injected_rows | control_rows:
        is_field[k] = False

    lab = det.stats["classification"].to_numpy()
    fp_mask = is_field & (lab == S.SPARSE)
    fp_elements = (
        det.stats.loc[fp_mask, "element_max"].value_counts().to_dict() if fp_mask.any() else {}
    )
    n_field_stars = int(is_field.sum())

    # --- controls --------------------------------------------------------------
    controls: dict[str, Any] = {}
    if with_controls:
        d_idx = [int(v) for v in dense_tab["row"]]
        d_lab = det.stats["classification"].to_numpy()[d_idx]
        controls["dense_multi_element"] = {
            "description": (
                "all of O-through-Ni raised by 0.40 dex together: a metal-rich star, "
                "not a refined one"
            ),
            "n": len(d_idx),
            "n_flagged_sparse": int((d_lab == S.SPARSE).sum()),
            "classifications": {
                k: int(v) for k, v in pd.Series(d_lab).value_counts().to_dict().items()
            },
            "median_n_discrepant": float(
                np.median(det.stats["n_discrepant"].to_numpy()[d_idx])
            ),
        }
        l_idx = [int(v) for v in light_tab["row"]]
        l_lab = det.stats["classification"].to_numpy()[l_idx]
        l_max = det.stats["element_max"].to_numpy()[l_idx]
        controls["light_element_exclusion"] = {
            "description": (
                "Li/Be/B spiked by 1.20 dex: genuinely single-element, and excluded "
                "a priori because it is a KNOWN natural single-element variable"
            ),
            "n": len(l_idx),
            "n_flagged_sparse": int((l_lab == S.SPARSE).sum()),
            "n_carried_by_a_light_element": int(
                sum(1 for e in l_max if e in cfg.exclude_elements)
            ),
            "excluded_elements": list(cfg.exclude_elements),
            "excluded_from_statistic": [
                e for e in det.Z.columns if e in cfg.exclude_elements
            ],
        }

    # --- optional: the catalogue-level contamination funnel ---------------------
    vet_block: dict[str, Any] = {}
    if run_vet and n_rec:
        cand = det.joined[det.joined["classification"] == S.SPARSE].copy()
        vcfg = V.VetConfig()
        cand = V.vet_candidates(cand, cfg=vcfg, survey="GALAH")
        cand = V.element_rate_veto(cand, det.flag_rates, cfg=vcfg)
        flagged = det.stats["n_discrepant"].to_numpy() > 0
        cand = V.field_rate_veto(cand, det.joined, flagged, cfg=vcfg)
        inj_ids = set(inj["star_id"])
        passed = cand[cand["vet_pass"]]
        vet_block = {
            "n_sparse_before_vetting": int(len(cand)),
            "n_sparse_after_vetting": int(len(passed)),
            "n_injected_surviving_vetting": int(
                sum(1 for s in passed["star_id"] if s in inj_ids)
            ),
            "note": (
                "GALAH's Na caveat is absent from vet.ELEMENT_CAVEATS (only APOGEE Na "
                "is listed), so an injected GALAH Na candidate carries no demotion"
            ),
        }

    # --- manifold quality, for comparison against the published number ---------
    resid = M.residuals(M.to_xh(df, det.elements, feh_col="fe_h"), det.manifold)
    rms = {
        el: round(float(np.sqrt(np.nanmean(resid[el].to_numpy() ** 2))), 4)
        for el in det.elements
    }

    # --- what the same data would give under other thresholds -----------------
    inj_rows = sorted(injected_rows)
    alt = [
        _rescore(det, S.SparseConfig(**kw), inj_rows, is_field, element)
        for kw in alt_configs
    ]

    frac = n_rec / max(n_injected, 1)
    verdict = _verdict(n_rec, n_injected, frac)

    return {
        "target": {
            "reference": GRIFFITH_REF,
            "quote": GRIFFITH_QUOTE,
            "n_stars": n_injected,
            "element": element,
            "amplitude_range_dex": list(amplitude_range),
            "other_elements": "O through Ni left untouched (single-element by construction)",
        },
        "config": {
            "seed": seed,
            "n_field_stars": n_pre,
            "z_flag": cfg.z_flag,
            "z_quiet": cfg.z_quiet,
            "max_discrepant": cfg.max_discrepant,
            "max_quiet_excess": cfg.max_quiet_excess,
            "min_contrast": cfg.min_contrast,
            "min_elements": cfg.min_elements,
            "family_max_mean_z": cfg.family_max_mean_z,
            "excluded_elements": list(cfg.exclude_elements),
            "space": "[X/H] residuals against the fitted manifold (never [X/Fe] or [X/Mg])",
        },
        "population": {
            "n_total": int(len(df)),
            "n_elements_on_manifold": len(det.elements),
            "elements": det.elements,
            # Note: sparse.sparse_statistics computes BOTH `n_elements` and
            # `n_elements_all` over the already-filtered element list, so the
            # latter does not in fact count the excluded ones. The true total is
            # taken from Z directly.
            "median_measured_elements": float(
                np.median(det.Z.notna().sum(axis=1).to_numpy())
            ),
            "median_measured_elements_in_statistic": float(
                np.median(det.stats["n_elements"])
            ),
            "rms_residual_dex": rms,
            f"sigma_{element}_median_dex": round(
                float(np.nanmedian(det.sigma[element])), 4
            )
            if element in det.sigma.columns
            else None,
        },
        "recovery": {
            "n_injected": n_injected,
            "n_recovered": n_rec,
            "fraction": round(frac, 4),
            "miss_breakdown": _miss_breakdown(rows, element, cfg.z_flag),
            "per_star": rows,
        },
        "alternative_thresholds": alt,
        "false_positives": {
            "n_uninjected_stars": n_field_stars,
            "n_flagged_sparse": int(fp_mask.sum()),
            "rate": round(float(fp_mask.sum()) / max(n_field_stars, 1), 6),
            "elements": {str(k): int(v) for k, v in fp_elements.items()},
        },
        "controls": controls,
        "vetting": vet_block,
        "verdict": verdict,
    }


def _verdict(n_rec: int, n_inj: int, frac: float) -> str:
    if frac >= 0.8:
        return (
            f"VALIDATION_PASSED: {n_rec}/{n_inj} of the Griffith Na population recovered "
            "at the shipped thresholds. The statistic detects, on purpose, the one "
            "published population with this morphology."
        )
    if frac >= 0.5:
        return (
            f"VALIDATION_MARGINAL: only {n_rec}/{n_inj} of the Griffith Na population "
            "recovered at the shipped thresholds. The statistic finds the morphology but "
            "the thresholds cost a large fraction of it — see recovery.miss_breakdown "
            "before any occurrence statement is made."
        )
    return (
        f"VALIDATION_FAILED: {n_rec}/{n_inj} of the Griffith Na population recovered. "
        "The channel cannot recover the published population it cites as its proof of "
        "concept, so the statistic — not the sky — is what needs changing."
    )


# ---------------------------------------------------------------------------
# Diagnostics that explain a shortfall rather than hiding it
# ---------------------------------------------------------------------------
def threshold_scan(
    *,
    seed: int = 20260726,
    n_field: int = 6000,
    z_flags: tuple[float, ...] = (4.0, 5.0, 6.0, 7.0),
    quiet_excess: tuple[int, ...] = (1, 2, 3),
    n_injected: int = GRIFFITH_N_STARS,
    element: str = "Na",
) -> pd.DataFrame:
    """Recovery and false-positive rate over the two thresholds that bind.

    Built once, scored many times: the population and the manifold are shared
    across the grid, so only the classification rules vary. That is the point —
    it isolates the *rules* from the data.
    """
    rng = np.random.default_rng(seed)
    df = synthesise_population(n_field, rng=rng)
    df, inj = inject_griffith_na(df, n_stars=n_injected, rng=rng, element=element)

    base = detect(df)
    inj_rows = [int(v) for v in inj["row"]]
    is_field = np.ones(len(df), dtype=bool)
    is_field[inj_rows] = False

    out = []
    for zf in z_flags:
        for qe in quiet_excess:
            cfg = S.SparseConfig(z_flag=float(zf), max_quiet_excess=int(qe))
            stats = S.sparse_statistics(base.Z, cfg=cfg)
            lab = stats["classification"].to_numpy()
            emax = stats["element_max"].to_numpy()
            rec = int(
                sum(
                    1
                    for k in inj_rows
                    if lab[k] == S.SPARSE and emax[k] == element
                )
            )
            fp = int(((lab == S.SPARSE) & is_field).sum())
            out.append(
                {
                    "z_flag": float(zf),
                    "max_quiet_excess": int(qe),
                    "n_recovered": rec,
                    "recovery_fraction": rec / n_injected,
                    "n_false_positive": fp,
                    "false_positive_rate": fp / int(is_field.sum()),
                }
            )
    return pd.DataFrame(out)


def to_ratio_space(
    df: pd.DataFrame, *, reference: str = "Mg", elements: list[str] | None = None
) -> pd.DataFrame:
    """Re-normalise every abundance onto ``[X/ref]`` and blank the reference.

    This is the space Weinberg et al. warn about, and the reason it is dangerous
    is specific: the reference element is **not** one of the manifold's
    predictors, so its measurement error cannot be absorbed by the regression.
    It lands in every element's residual at once, with the same sign — which is
    exactly how a sparse anomaly is converted into a weak dense one.
    """
    out = df.copy()
    els = elements or [c for c in df.columns if M.element_family(c) != "other"]
    if reference not in out.columns:
        return out
    ref = out[reference].to_numpy(dtype=float)
    for el in els:
        if el == reference:
            continue
        out[el] = out[el].to_numpy(dtype=float) - ref
        ecol, eref = f"e_{el}", f"e_{reference}"
        if ecol in out.columns and eref in out.columns:
            out[ecol] = np.sqrt(
                out[ecol].to_numpy(dtype=float) ** 2 + out[eref].to_numpy(dtype=float) ** 2
            )
    out[reference] = np.nan
    return out


def aberration_comparison(
    *,
    seed: int = 20260726,
    n_field: int = 6000,
    n_injected: int = GRIFFITH_N_STARS,
    element: str = "Na",
    reference: str = "Mg",
) -> dict[str, Any]:
    """Measure what the choice of abundance normalisation actually costs.

    The same injected population is scored three ways:

    ``xh``
        the shipped path — ``manifold.to_xh`` then fit.
    ``xfe``
        the published ``[X/Fe]`` fitted directly, with no conversion.
    ``xmg``
        ``[X/Mg]`` fitted directly — ratio space against an element that is
        *not* a manifold predictor.

    The result is not the one the channel's docstrings assert, and it is stated
    plainly here because it matters for how the claim is written up: ``xh`` and
    ``xfe`` give **identical** residuals. They must, algebraically — ``[Fe/H]``
    is a linear column of the design matrix, so subtracting it from the target
    only shifts that column's coefficient by one and leaves ``y - X beta``
    unchanged. The protection against Weinberg's measurement aberration in this
    pipeline comes from *regressing on ``[Fe/H]``*, not from the conversion.

    ``xmg`` is where the aberration is real, because ``[Mg/H]`` is not in the
    design matrix and its error therefore survives into every element's residual
    with the same sign.
    """
    rng = np.random.default_rng(seed)
    df = synthesise_population(n_field, rng=rng)
    df, inj = inject_griffith_na(df, n_stars=n_injected, rng=rng, element=element)
    inj_rows = [int(v) for v in inj["row"]]
    ratio = to_ratio_space(df, reference=reference)

    result: dict[str, Any] = {}
    for label, frame, in_xh in (
        ("xh", df, True),
        ("xfe", df, False),
        ("xmg", ratio, False),
    ):
        det = detect(frame, in_xh=in_xh)
        lab = det.stats["classification"].to_numpy()
        emax = det.stats["element_max"].to_numpy()
        rec = int(sum(1 for k in inj_rows if lab[k] == S.SPARSE and emax[k] == element))
        z_inj = np.array([abs(float(det.Z[element].iloc[k])) for k in inj_rows])
        result[label] = {
            "n_recovered": rec,
            "fraction": rec / n_injected,
            "median_z_injected": round(float(np.median(z_inj)), 3),
            "median_sigma_used_dex": round(float(np.nanmedian(det.sigma[element])), 4),
            "mean_n_active_all_stars": round(float(np.mean(det.stats["n_active"])), 3),
            "n_elements_on_manifold": len(det.elements),
        }
    result["space_used_by_the_pipeline"] = "xh"
    result["finding"] = (
        "[X/H] and [X/Fe] are residual-identical because [Fe/H] is a linear manifold "
        "predictor; the aberration bites only when the normaliser is NOT a predictor, "
        f"which is what the [X/{reference}] column measures."
    )
    return result


__all__ = [
    "GALAH_DR3_SPECS",
    "GRIFFITH_AMPLITUDE_RANGE",
    "GRIFFITH_HEAVY_ELEMENTS",
    "GRIFFITH_N_STARS",
    "GRIFFITH_QUOTE",
    "GRIFFITH_REF",
    "O_THROUGH_NI",
    "Detection",
    "ElementSpec",
    "aberration_comparison",
    "detect",
    "inject_dense_control",
    "inject_griffith_na",
    "inject_light_element_control",
    "synthesise_population",
    "threshold_scan",
    "validate_griffith",
]

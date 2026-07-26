"""Stage 4 — co-natal wide binaries and the engulfed-planet mass budget.

The idea
--------
Two stars in a Gaia wide binary formed from the same molecular-cloud material
at the same time. Their birth composition was identical to well inside the
measurement precision — high-precision differential work on co-natal pairs
routinely reaches 0.01-0.02 dex, and the internal chemical coherence of a
single open cluster is ~0.03 dex (Bovy 2016; Cheng et al. 2020; Casamiquela
et al. 2021; Patil et al. 2022). So a *differential* abundance between the two
components is not a statement about Galactic chemical evolution at all. It is a
statement about what happened to one of them afterwards, and the differential
analysis cancels most pipeline systematics because the two stars are analysed
identically.

Almost everything that happens afterwards is known:

* **Atomic diffusion** — settles heavy elements out of the envelope, a smooth
  function of stellar mass, and it moves everything refractory together.
* **Planet engulfment** — dumps rock into the convective envelope, raising
  every refractory in proportion to its abundance in rock, i.e. along a
  **condensation-temperature trend**. This is now an established, measured
  phenomenon: Liu et al. 2024 (Nature) found >=7 new ingestion instances among
  91 co-natal pairs, an ~8% occurrence rate.

Both are **dense** signatures. They move a family. The engulfment channel is
the strongest natural competitor for TAILINGS, so stage 4 is designed to say
exactly when a pair difference exceeds it — in two independent ways.

Test A: the mass budget
-----------------------
Dissolving a mass ``M_rock`` of rock into a convective envelope of mass
``M_cz`` raises element X by

    d[X/H] = log10( 1 + M_rock * f_X^rock / (M_cz * Z_X,sun * 10^[Fe/H]) )

with ``f_X^rock`` the mass fraction of X in the rock and ``Z_X,sun`` the mass
fraction of X in solar-composition stellar material. For the Sun
(``M_cz = 0.021 M_sun``) this gives ~0.015 dex in [Fe/H] per Earth mass, the
canonical number. Inverting it turns an observed differential into an
**implied engulfed rocky mass**, and the question becomes whether any
planetary system could have supplied it.

The calibration point that matters is the most extreme case actually claimed in
the literature: the Kronos/Krios pair (HD 240430/240429), with a differential
metallicity of ~0.20-0.23 dex, for which the published inferred ingested mass
is ~28 Earth masses and was described as exceptionally large. The default
budget ceiling here is **100 Earth masses** — more than three times that, and
above the ~10-30 Earth-mass core at which runaway gas accretion turns a rocky
body into a gas giant, so a system able to deliver more rock than this to its
star is outside every planet-formation model as well as outside every
observation.

The conservatism is deliberate and it runs one way. ``M_cz`` is scaled *down*
by a safety factor by default, which makes the pollutant less diluted, the
implied mass smaller, and the "unexplainable" verdict harder to reach.
Thermohaline mixing after accretion dilutes a real signature further still,
which again means the true required mass is larger than computed here. Every
approximation in this module is chosen so that a star only clears the bar if it
would clear it under the most engulfment-friendly assumptions.

Test B: composition, independent of mass
----------------------------------------
Rock is not a single element. Any engulfment moves Fe, Mg, Si, Ni, Ca, Al, Cr,
Ti and more, together, in fixed proportion, along a condensation-temperature
trend. So a pair whose difference is **one element with the rest identical** is
not engulfment *at any mass* — no amount of rock produces that pattern. That is
the same sparsity argument as the main channel, applied differentially, and it
is the stronger of the two tests because it does not depend on a convective
envelope model at all.

Honest limits
-------------
* ``M_cz(Teff)`` is a coarse tabulation of standard main-sequence models. It is
  good to a factor of ~2 in the F-G range and worse for M dwarfs; the safety
  factor exists because of that.
* Rock composition is taken as bulk Earth. A differentiated core-rich fragment
  is more siderophile-rich and would produce a *different* pattern — which is
  precisely the hypothesis Huang et al. 2026 test in polluted white dwarfs, and
  it is a **dense**, >=5-element template, so it is caught by the family and
  Tcond tests here rather than being confused with a sparse anomaly.
* Non-co-natal contaminants (chance alignments, hierarchical triples) are a
  real background and are handled in ``vet``, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

M_EARTH_IN_MSUN = 3.0027e-6

# ---------------------------------------------------------------------------
# 50% condensation temperatures for a solar-composition gas (Lodders 2003), K.
# The engulfment signature is a positive trend of d[X/H] against these.
# ---------------------------------------------------------------------------
T_COND: dict[str, float] = {
    "C": 40.0,
    "N": 123.0,
    "O": 180.0,
    "S": 664.0,
    "Zn": 726.0,
    "Na": 958.0,
    "K": 1006.0,
    "Cu": 1037.0,
    "Mn": 1158.0,
    "P": 1229.0,
    "Cr": 1296.0,
    "Si": 1310.0,
    "Fe": 1334.0,
    "Mg": 1336.0,
    "Li": 1142.0,
    "Eu": 1356.0,
    "Co": 1352.0,
    "Ni": 1353.0,
    "Ce": 1478.0,
    "V": 1429.0,
    "Ba": 1455.0,
    "Sr": 1464.0,
    "Ca": 1517.0,
    "La": 1578.0,
    "Ti": 1582.0,
    "Sm": 1590.0,
    "Nd": 1602.0,
    "Al": 1653.0,
    "Sc": 1659.0,
    "Y": 1659.0,
    "Zr": 1741.0,
}

#: Refractory means condensing above this temperature. Engulfment raises these.
REFRACTORY_TCOND_K = 1200.0

# ---------------------------------------------------------------------------
# Mass fractions. Solar values are Asplund et al. (2009) photospheric
# abundances converted to mass fraction of total stellar material; bulk-Earth
# values follow McDonough-style bulk-Earth compositions. Both are approximate
# at the 10-30% level for the minor elements, which is far below the factor of
# several that separates a plausible from an implausible engulfed mass.
# ---------------------------------------------------------------------------
SOLAR_MASS_FRACTION: dict[str, float] = {
    "C": 2.36e-3,
    "N": 6.93e-4,
    "O": 5.73e-3,
    "Na": 2.9e-5,
    "Mg": 7.08e-4,
    "Al": 5.6e-5,
    "Si": 6.65e-4,
    "P": 6.1e-6,
    "S": 3.09e-4,
    "K": 3.1e-6,
    "Ca": 6.41e-5,
    "Sc": 4.6e-8,
    "Ti": 3.1e-6,
    "V": 3.8e-7,
    "Cr": 1.72e-5,
    "Mn": 1.12e-5,
    "Fe": 1.29e-3,
    "Co": 4.2e-6,
    "Ni": 7.3e-5,
    "Cu": 7.3e-7,
    "Zn": 1.8e-6,
    "Sr": 4.5e-8,
    "Y": 1.0e-8,
    "Zr": 2.5e-8,
    "Ba": 1.6e-8,
    "La": 2.1e-9,
    "Ce": 5.5e-9,
    "Nd": 4.0e-9,
    "Sm": 1.3e-9,
    "Eu": 5.0e-10,
    "Li": 1.0e-8,
}

BULK_EARTH_MASS_FRACTION: dict[str, float] = {
    "C": 7.3e-4,
    "N": 6.0e-6,
    "O": 0.297,
    "Na": 1.80e-3,
    "Mg": 0.154,
    "Al": 1.59e-2,
    "Si": 0.161,
    "P": 7.2e-4,
    "S": 6.35e-3,
    "K": 1.60e-4,
    "Ca": 1.71e-2,
    "Sc": 1.0e-5,
    "Ti": 8.2e-4,
    "V": 9.5e-5,
    "Cr": 4.72e-3,
    "Mn": 8.0e-4,
    "Fe": 0.321,
    "Co": 8.8e-4,
    "Ni": 1.82e-2,
    "Cu": 6.0e-5,
    "Zn": 4.0e-5,
    "Sr": 1.3e-5,
    "Y": 2.6e-6,
    "Zr": 6.4e-6,
    "Ba": 4.5e-6,
    "La": 4.4e-7,
    "Ce": 1.1e-6,
    "Nd": 8.4e-7,
    "Sm": 2.7e-7,
    "Eu": 1.1e-7,
    "Li": 1.5e-6,
}

# ---------------------------------------------------------------------------
# Convective envelope mass vs Teff for main-sequence dwarfs at roughly solar
# metallicity and age. Coarse tabulation of standard models; good to a factor
# of ~2 in F-G, worse for M dwarfs (where the star becomes fully convective
# below ~3500 K and M_cz saturates at the stellar mass).
# ---------------------------------------------------------------------------
_MCZ_TEFF = np.array(
    [3000, 3200, 3400, 3600, 3800, 4000, 4200, 4500, 4800, 5000, 5200, 5400,
     5600, 5800, 6000, 6200, 6500],
    dtype=float,
)
_MCZ_MSUN = np.array(
    [0.20, 0.28, 0.35, 0.45, 0.42, 0.34, 0.27, 0.20, 0.14, 0.10, 0.072, 0.050,
     0.033, 0.021, 0.0028, 0.0012, 0.0005],
    dtype=float,
)


def convective_envelope_mass(teff: np.ndarray | float, *, safety_factor: float = 0.5) -> np.ndarray:
    """Convective envelope mass in solar masses, log-interpolated in Teff.

    ``safety_factor`` scales the result *down*. A smaller envelope dilutes an
    accreted pollutant less, so a given observed differential implies a
    *smaller* engulfed mass — i.e. the default 0.5 makes the "no planetary
    system could supply this" verdict strictly harder to reach.
    """
    t = np.asarray(teff, dtype=float)
    logm = np.interp(t, _MCZ_TEFF, np.log10(_MCZ_MSUN), left=np.log10(_MCZ_MSUN[0]),
                     right=np.log10(_MCZ_MSUN[-1]))
    return safety_factor * 10.0**logm


def delta_from_rock(
    mass_earth: np.ndarray | float,
    element: str,
    *,
    teff: np.ndarray | float,
    feh: np.ndarray | float = 0.0,
    safety_factor: float = 0.5,
    rock: dict[str, float] | None = None,
) -> np.ndarray:
    """d[X/H] produced by dissolving ``mass_earth`` of rock into the envelope."""
    rock = rock or BULK_EARTH_MASS_FRACTION
    f_rock = rock.get(element)
    z_sun = SOLAR_MASS_FRACTION.get(element)
    if f_rock is None or z_sun is None:
        return np.full(np.shape(np.asarray(mass_earth, dtype=float)), np.nan)
    mcz = convective_envelope_mass(teff, safety_factor=safety_factor)
    m_pol = np.asarray(mass_earth, dtype=float) * M_EARTH_IN_MSUN * f_rock
    m_env = mcz * z_sun * 10.0 ** np.asarray(feh, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(1.0 + m_pol / m_env)


def implied_engulfed_mass(
    delta: np.ndarray | float,
    element: str,
    *,
    teff: np.ndarray | float,
    feh: np.ndarray | float = 0.0,
    safety_factor: float = 0.5,
    rock: dict[str, float] | None = None,
) -> np.ndarray:
    """Invert :func:`delta_from_rock`: Earth masses needed to make ``delta``.

    Returns NaN for non-positive ``delta`` (a *deficit* is not engulfment) and
    for elements without a tabulated composition.
    """
    rock = rock or BULK_EARTH_MASS_FRACTION
    f_rock = rock.get(element)
    z_sun = SOLAR_MASS_FRACTION.get(element)
    d = np.asarray(delta, dtype=float)
    if f_rock is None or z_sun is None:
        return np.full(d.shape, np.nan)
    mcz = convective_envelope_mass(teff, safety_factor=safety_factor)
    m_env = mcz * z_sun * 10.0 ** np.asarray(feh, dtype=float)
    with np.errstate(invalid="ignore"):
        need = (10.0**d - 1.0) * m_env / (M_EARTH_IN_MSUN * f_rock)
    return np.where(d > 0, need, np.nan)


# ---------------------------------------------------------------------------
# Pair statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TwinConfig:
    """Thresholds for the co-natal pair stage."""

    max_engulfed_earth_masses: float = 100.0
    """Budget ceiling. ~3.5x the largest engulfed mass ever inferred
    (Kronos/Krios, ~28 Me) and above the runaway-accretion core mass."""

    mcz_safety_factor: float = 0.5
    """Scales the convective envelope down; makes the budget test conservative."""

    z_flag: float = 4.0
    """Per-element significance for a pair difference to count as real.
    Lower than the field threshold because a differential analysis of two
    co-natal stars cancels most systematics."""

    z_quiet: float = 2.0
    max_discrepant: int = 2
    min_elements: int = 6
    min_refractories: int = 4
    """Refractories needed before a Tcond trend can be fitted at all."""

    tcond_slope_sigma: float = 3.0
    """Significance above which a Tcond trend counts as present (= engulfment-like)."""


NO_DIFFERENCE = "NO_DIFFERENCE"
ENGULFMENT_CONSISTENT = "ENGULFMENT_CONSISTENT"
ENGULFMENT_EXCESSIVE = "ENGULFMENT_EXCESSIVE"
SPARSE_UNEXPLAINABLE = "SPARSE_UNEXPLAINABLE"
DENSE_NOT_ENGULFMENT = "DENSE_NOT_ENGULFMENT"
INSUFFICIENT = "INSUFFICIENT"


def tcond_trend(
    deltas: dict[str, float],
    sigmas: dict[str, float] | None = None,
) -> dict:
    """Weighted linear fit of d[X/H] against condensation temperature.

    Returns the slope in dex per 1000 K, its uncertainty, the significance, and
    the RMS about the fit. A positive, significant slope is the engulfment
    fingerprint; a flat trend with one element off it is not.
    """
    els = [e for e in deltas if e in T_COND and np.isfinite(deltas[e])]
    if len(els) < 3:
        return {"slope": np.nan, "slope_err": np.nan, "slope_sigma": np.nan,
                "rms": np.nan, "n": len(els)}
    x = np.array([T_COND[e] for e in els]) / 1000.0
    y = np.array([deltas[e] for e in els])
    if sigmas:
        s = np.array([max(float(sigmas.get(e, np.nan)), 1e-4) for e in els])
        s = np.where(np.isfinite(s), s, np.nanmedian(s[np.isfinite(s)]) if np.isfinite(s).any() else 1.0)
    else:
        s = np.ones_like(y)
    w = 1.0 / s**2
    A = np.column_stack([np.ones_like(x), x])
    ATA = A.T @ (w[:, None] * A)
    try:
        cov = np.linalg.inv(ATA)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate design
        return {"slope": np.nan, "slope_err": np.nan, "slope_sigma": np.nan,
                "rms": np.nan, "n": len(els)}
    beta = cov @ (A.T @ (w * y))
    resid = y - A @ beta
    slope = float(beta[1])
    slope_err = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {
        "slope": slope,
        "slope_err": slope_err,
        "slope_sigma": float(slope / slope_err) if slope_err > 0 else np.nan,
        "rms": float(np.sqrt(np.mean(resid**2))),
        "n": len(els),
    }


def pair_verdict(
    deltas: dict[str, float],
    sigmas: dict[str, float],
    *,
    teff: float,
    feh: float = 0.0,
    cfg: TwinConfig | None = None,
) -> dict:
    """Classify one co-natal pair's differential abundance vector.

    ``deltas[X]`` is d[X/H] = (polluted component) - (comparison component) and
    ``sigmas[X]`` its uncertainty. ``teff`` is the Teff of the component being
    treated as polluted (it sets the convective envelope).
    """
    cfg = cfg or TwinConfig()
    els = [e for e in deltas if np.isfinite(deltas.get(e, np.nan))]
    if len(els) < cfg.min_elements:
        return {"verdict": INSUFFICIENT,
                "reason": f"only {len(els)} elements measured in both components",
                "n_elements": len(els)}

    z = {e: deltas[e] / max(float(sigmas.get(e, np.nan)) if np.isfinite(sigmas.get(e, np.nan))
                            else np.inf, 1e-6) for e in els}
    absz = {e: abs(v) for e, v in z.items()}
    disc = [e for e in els if absz[e] >= cfg.z_flag]
    active = [e for e in els if absz[e] >= cfg.z_quiet]

    refr = [e for e in els if T_COND.get(e, 0.0) >= REFRACTORY_TCOND_K]
    trend = tcond_trend({e: deltas[e] for e in els}, {e: sigmas.get(e, np.nan) for e in els})

    # Implied engulfed mass: the robust median over refractories with a
    # positive differential, plus the single most demanding element.
    implied = {}
    for e in refr:
        if deltas[e] > 0:
            m = implied_engulfed_mass(deltas[e], e, teff=teff, feh=feh,
                                      safety_factor=cfg.mcz_safety_factor)
            if np.isfinite(m):
                implied[e] = float(m)
    m_med = float(np.median(list(implied.values()))) if implied else float("nan")
    m_max = float(np.max(list(implied.values()))) if implied else float("nan")

    base = {
        "n_elements": len(els),
        "n_discrepant": len(disc),
        "n_active": len(active),
        "discrepant_elements": sorted(disc, key=lambda e: -absz[e]),
        "n_refractories": len(refr),
        "tcond_slope_dex_per_kK": trend["slope"],
        "tcond_slope_sigma": trend["slope_sigma"],
        "tcond_rms_dex": trend["rms"],
        "implied_engulfed_mass_earth_median": m_med,
        "implied_engulfed_mass_earth_max": m_max,
        "budget_earth_masses": cfg.max_engulfed_earth_masses,
        "z_max": float(max(absz.values())) if absz else float("nan"),
        "element_max": max(absz, key=absz.get) if absz else None,
    }

    if not disc:
        return {**base, "verdict": NO_DIFFERENCE,
                "reason": f"no element differs at |z| >= {cfg.z_flag}"}

    has_trend = np.isfinite(trend["slope_sigma"]) and trend["slope_sigma"] >= cfg.tcond_slope_sigma
    sparse = (
        len(disc) <= cfg.max_discrepant
        and len(active) <= len(disc) + 1
        and not has_trend
    )

    if sparse:
        # No amount of rock produces a one-element offset with a flat Tcond
        # trend. This verdict does not depend on the convective-envelope model.
        return {**base, "verdict": SPARSE_UNEXPLAINABLE,
                "reason": (f"{len(disc)} element(s) differ ({', '.join(base['discrepant_elements'])}) "
                           f"with no condensation-temperature trend "
                           f"(slope {trend['slope_sigma']:.1f} sigma): rock of any mass "
                           "moves the whole refractory family, so engulfment cannot "
                           "produce this pattern")}

    if has_trend and trend["slope"] > 0:
        if np.isfinite(m_med) and m_med > cfg.max_engulfed_earth_masses:
            return {**base, "verdict": ENGULFMENT_EXCESSIVE,
                    "reason": (f"condensation-temperature trend present but requires "
                               f"{m_med:.0f} Earth masses of rock, above the "
                               f"{cfg.max_engulfed_earth_masses:.0f} Me budget ceiling")}
        return {**base, "verdict": ENGULFMENT_CONSISTENT,
                "reason": (f"positive Tcond trend at {trend['slope_sigma']:.1f} sigma, "
                           f"implied mass {m_med:.1f} Me within budget")}

    return {**base, "verdict": DENSE_NOT_ENGULFMENT,
            "reason": (f"{len(disc)} elements differ with no positive Tcond trend "
                       "-- not sparse, and not the engulfment pattern either; "
                       "check co-natality and the pipeline before anything else")}


def pair_table(
    pairs: pd.DataFrame,
    elements: list[str],
    *,
    a_prefix: str = "a_",
    b_prefix: str = "b_",
    err_suffix: str = "_err",
    teff_col_a: str = "a_teff",
    teff_col_b: str = "b_teff",
    feh_col_a: str = "a_fe_h",
    cfg: TwinConfig | None = None,
) -> pd.DataFrame:
    """Run :func:`pair_verdict` over a table of co-natal pairs.

    The component with the *higher* mean refractory abundance is treated as the
    polluted one, because that is the direction accretion works; the sign is
    recorded so a refractory *deficit* — which engulfment cannot produce at all
    — is visible rather than silently absorbed.
    """
    cfg = cfg or TwinConfig()
    rows = []
    for idx, r in pairs.iterrows():
        d_raw, s_raw = {}, {}
        for e in elements:
            ca, cb = f"{a_prefix}{e}", f"{b_prefix}{e}"
            if ca not in pairs.columns or cb not in pairs.columns:
                continue
            va, vb = float(r[ca]), float(r[cb])
            if not (np.isfinite(va) and np.isfinite(vb)):
                continue
            ea = float(r.get(f"{ca}{err_suffix}", np.nan))
            eb = float(r.get(f"{cb}{err_suffix}", np.nan))
            sig = np.sqrt(np.nansum([ea**2, eb**2]))
            d_raw[e] = va - vb
            s_raw[e] = sig if sig > 0 else np.nan

        refr = [e for e in d_raw if T_COND.get(e, 0.0) >= REFRACTORY_TCOND_K]
        mean_refr = float(np.mean([d_raw[e] for e in refr])) if refr else 0.0
        flip = mean_refr < 0
        deltas = {e: (-v if flip else v) for e, v in d_raw.items()}
        teff = float(r[teff_col_b] if flip else r[teff_col_a])
        feh = float(r.get(feh_col_a, 0.0))

        v = pair_verdict(deltas, s_raw, teff=teff, feh=feh, cfg=cfg)
        v["pair_id"] = r.get("pair_id", idx)
        v["polluted_component"] = "b" if flip else "a"
        v["mean_refractory_delta_dex"] = abs(mean_refr)
        rows.append(v)
    return pd.DataFrame(rows)


__all__ = [
    "BULK_EARTH_MASS_FRACTION",
    "DENSE_NOT_ENGULFMENT",
    "ENGULFMENT_CONSISTENT",
    "ENGULFMENT_EXCESSIVE",
    "INSUFFICIENT",
    "M_EARTH_IN_MSUN",
    "NO_DIFFERENCE",
    "REFRACTORY_TCOND_K",
    "SOLAR_MASS_FRACTION",
    "SPARSE_UNEXPLAINABLE",
    "T_COND",
    "TwinConfig",
    "convective_envelope_mass",
    "delta_from_rock",
    "implied_engulfed_mass",
    "pair_table",
    "pair_verdict",
    "tcond_trend",
]

"""Dark-companion search from Gaia DR3 astrometric *orbits* (the BH1/2/3 method).

An acceleration alone cannot fix a companion's mass, but a full astrometric orbit
can.  Gaia DR3 ``nss_two_body_orbit`` gives the Thiele-Innes elements (A, B, F, G)
and period of the photocentre orbit.  From them:

* the photocentre semi-major axis  a0 (mas) -> a0 (AU) via the parallax;
* the **astrometric mass function**  f = a0_AU^3 / P_yr^2 = M2^3 / (M1 + M2)^2
  (valid when the secondary is dark, so the photocentre traces the primary);
* solving for M2 given the primary mass M1 (from the absolute magnitude) yields
  the companion mass.

A companion with M2 well above the mass a luminous star could have while staying
invisible (a few Msun) is a **dormant compact object** --- exactly how Gaia found
the nearest black holes.  Recovering those is validation; a *new* such system, or
one whose companion mass is impossible to reconcile with any stellar/compact
object, is the remarkable candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def a0_mas_from_thiele_innes(A: np.ndarray, B: np.ndarray, F: np.ndarray,
                             G: np.ndarray) -> np.ndarray:
    """Photocentre semi-major axis (mas) from the Thiele-Innes constants."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    F = np.asarray(F, float)
    G = np.asarray(G, float)
    u = 0.5 * (A * A + B * B + F * F + G * G)
    v = A * G - B * F
    disc = np.clip(u * u - v * v, 0.0, None)
    return np.sqrt(np.clip(u + np.sqrt(disc), 0.0, None))


def primary_mass_from_absg(abs_g: np.ndarray) -> np.ndarray:
    """Rough main-sequence primary mass (Msun) from absolute G.  Monotonic
    inversion of a crude M_G(M) relation; only used to scale the mass function."""
    mg = np.asarray(abs_g, float)
    # M_G ~ 4.8 - 8 log10(M)  ->  M = 10^((4.8 - M_G)/8), clipped to a sane range.
    return np.clip(10.0 ** ((4.8 - mg) / 8.0), 0.1, 20.0)


def companion_mass(mass_function: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Solve f = M2^3 / (M1 + M2)^2 for M2 (Msun), element-wise (Newton)."""
    f = np.asarray(mass_function, float)
    m1 = np.asarray(m1, float)
    m2 = np.maximum(f, 0.1).astype(float)          # initial guess
    for _ in range(60):
        g = m2 ** 3 - f * (m1 + m2) ** 2
        gp = 3 * m2 ** 2 - 2 * f * (m1 + m2)
        step = np.where(np.abs(gp) > 1e-12, g / gp, 0.0)
        m2 = np.clip(m2 - step, 1e-3, 1e4)
    return m2


def analyze_orbits(df: pd.DataFrame) -> pd.DataFrame:
    """Add a0, mass function, primary/companion mass and a compact-object flag.

    Expects Thiele-Innes columns ``a_thiele_innes``/``b_thiele_innes``/
    ``f_thiele_innes``/``g_thiele_innes`` (mas), ``period`` (days), ``parallax``
    (mas), ``phot_g_mean_mag``.  Column names are matched flexibly.
    """
    out = df.copy()

    def col(*names):
        for n in names:
            if n in out.columns:
                return pd.to_numeric(out[n], errors="coerce")
        return pd.Series(np.nan, index=out.index)

    A = col("a_thiele_innes", "a_ti", "athieleinnes")
    B = col("b_thiele_innes", "b_ti", "bthieleinnes")
    F = col("f_thiele_innes", "f_ti", "fthieleinnes")
    Gc = col("g_thiele_innes", "g_ti", "gthieleinnes")
    P_d = col("period")
    plx = col("parallax")
    g = col("phot_g_mean_mag")

    a0_mas = a0_mas_from_thiele_innes(A, B, F, Gc)
    out["a0_mas"] = a0_mas
    out["a0_au"] = a0_mas / plx                    # mas / (mas) * AU... = a0_mas/plx_mas
    out["period_yr"] = P_d / 365.25
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mass_function"] = out["a0_au"] ** 3 / out["period_yr"] ** 2
    out["abs_g"] = g + 5.0 * np.log10(np.where(plx > 0, plx / 100.0, np.nan))
    out["m1_msun"] = primary_mass_from_absg(out["abs_g"])
    out["m2_msun"] = companion_mass(out["mass_function"].to_numpy(),
                                    out["m1_msun"].to_numpy())
    # Astrometric Mass Ratio Function + Shahaf triage: the rigorous test of whether
    # the companion can be a luminous star at all, or must be compact.
    out["amrf"] = amrf(a0_mas, plx, out["period_yr"].to_numpy(),
                       out["m1_msun"].to_numpy())
    out["triage_class"] = triage_class(out["amrf"].to_numpy())
    # A companion heavier than ~3 Msun that is not a luminous star is a compact
    # object (white dwarf < 1.4, neutron star ~1.4-2.2, black hole > ~3).
    out["compact_object_candidate"] = (out["m2_msun"] >= 3.0) & (out["triage_class"] >= 2)
    return out


# The three published Gaia astrometric black holes (source_ids) -- recovering
# them validates the pipeline; anything else at similar mass is a fresh candidate.
KNOWN_GAIA_BH = {
    4373465352415301632,   # Gaia BH1
    5870569352746779008,   # Gaia BH2
    4318465066420528000,   # Gaia BH3
}


def amrf(a0_mas: np.ndarray, parallax_mas: np.ndarray, period_yr: np.ndarray,
         m1_msun: np.ndarray) -> np.ndarray:
    """Astrometric Mass Ratio Function (Shahaf et al. 2019, 2023):
    ``A = (a0/parallax) * P^(-2/3) * M1^(-1/3)``  (a0, parallax same units).

    For a *dark* companion A = q(1+q)^(-2/3); a *luminous* companion reduces A
    because its own light drags the photocentre toward it, so a large A cannot be
    produced by an ordinary stellar companion."""
    a0 = np.asarray(a0_mas, float)
    plx = np.asarray(parallax_mas, float)
    P = np.asarray(period_yr, float)
    m1 = np.asarray(m1_msun, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a0 / plx) * P ** (-2.0 / 3.0) * m1 ** (-1.0 / 3.0)


def _amrf_ms_single_max(ml_exp: float = 3.5) -> float:
    """Maximum AMRF a single main-sequence companion can produce.

    A_MS(q) = (1+q)^(1/3) [ q/(1+q) - S/(1+S) ],  S = q^ml_exp (mass-luminosity).
    A companion brighter than this ceiling cannot be a single MS star."""
    q = np.linspace(1e-3, 1.0, 4000)
    s = q ** ml_exp
    a = (1.0 + q) ** (1.0 / 3.0) * (q / (1.0 + q) - s / (1.0 + s))
    return float(np.nanmax(a))


# The single-MS ceiling (~0.36) and a generous 'inner-binary of two MS stars'
# ceiling: above the latter the companion is essentially forced to be compact.
_AMRF_MS1 = _amrf_ms_single_max()
_AMRF_MS2 = _AMRF_MS1 * 2.0 ** (1.0 / 3.0)     # two equal MS stars, ~1.26x


def triage_class(A: np.ndarray) -> np.ndarray:
    """Shahaf-style class: 1 = a single MS companion suffices; 2 = needs an inner
    binary or a compact object; 3 = requires a *compact* companion (A above what
    even two MS stars can produce)."""
    A = np.asarray(A, float)
    cls = np.ones_like(A, dtype=int)
    cls = np.where(A > _AMRF_MS1, 2, cls)
    cls = np.where(A > _AMRF_MS2, 3, cls)
    return cls


def rank_dark_companions(df: pd.DataFrame, m2_min: float = 3.0, m2_max: float = 25.0,
                         max_dist_pc: float = 1000.0, ruwe_min: float = 3.0
                         ) -> pd.DataFrame:
    """Rank systems whose invisible companion is massive (compact-object-like),
    with physical-consistency cuts.

    A genuine massive dark companion produces a *large* astrometric wobble, so a
    high implied mass paired with a *low* RUWE is a degenerate/spurious orbital
    fit (these dominate the short-period tail where a0^3/P^2 blows up) --- require
    ``ruwe >= ruwe_min``.  Cap the mass at ``m2_max``: a stellar-mass compact
    object is <~ 20 Msun, so a larger value flags a bad solution, not a discovery.
    """
    d = analyze_orbits(df) if "m2_msun" not in df.columns else df
    plx = pd.to_numeric(d.get("parallax"), errors="coerce")
    ruwe = pd.to_numeric(d.get("ruwe"), errors="coerce")
    dist = 1000.0 / plx
    tclass = pd.to_numeric(d.get("triage_class"), errors="coerce").fillna(0)
    sel = ((d["m2_msun"] >= m2_min) & (d["m2_msun"] <= m2_max)
           & (dist <= max_dist_pc) & np.isfinite(d["m2_msun"])
           & (ruwe >= ruwe_min)
           & (tclass >= 2))          # AMRF: companion cannot be a single MS star
    out = d[sel].copy()
    out["dist_pc"] = dist[sel]
    out["known_gaia_bh"] = out["source_id"].isin(KNOWN_GAIA_BH)
    # Rank the *new* nearby companions that MUST be compact (class 3) highest;
    # known BHs are validation and pushed down.
    out["rank_score"] = (np.clip(out["m2_msun"] / 15.0, 0, 1)
                         * np.clip(1.0 - out["dist_pc"] / max_dist_pc, 0, 1)
                         * np.where(out["triage_class"] >= 3, 1.0, 0.5)
                         * np.where(out["known_gaia_bh"], 0.2, 1.0))
    return out.sort_values(["known_gaia_bh", "triage_class", "rank_score"],
                           ascending=[True, False, False])


__all__ = ["a0_mas_from_thiele_innes", "companion_mass", "primary_mass_from_absg",
           "analyze_orbits", "rank_dark_companions", "amrf", "triage_class",
           "KNOWN_GAIA_BH"]

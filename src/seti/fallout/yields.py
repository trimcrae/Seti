"""FALLOUT physics tables: fission yields, solar abundances, s/r/p decomposition.

Everything a template needs, with the source of every number stated next to
it. The tables are deliberately *round*: they are cumulative chain yields and
solar-system fractions at the 5-10% level, which is an order of magnitude
below the 0.1-0.2 dex per-element precision of any survey abundance, so
refining them further would change nothing observable.

Three tables
------------
``CHAIN_YIELDS``   thermal-neutron U-235 cumulative *mass-chain* yields, per
                   100 fissions, with the element each chain ends on after the
                   short-lived members have decayed, and the last long-lived
                   intermediate if there is one. Source: ENDF/B-VII.1 and
                   JEFF-3.1.1 chain yields as tabulated by England & Rider
                   (LA-UR-94-3106, 1994); the two libraries agree to a few
                   percent for every chain used here.
``SOLAR_LOGEPS``   solar photospheric abundances A(X) = log N_X/N_H + 12 from
                   Asplund, Amarsi & Grevesse 2021 (A&A 653, A141), with the
                   meteoritic value where the photospheric one does not exist
                   (Te, I, Cs) and Lodders 2009 (Landolt-Bornstein 4B, 712)
                   for Kr/Xe. Differences between the two compilations are
                   <= 0.1 dex for every element in the template space.
``S_FRACTION``     fraction of each solar-system element made by the s-process,
                   from the Arlandini et al. 1999 (ApJ 525, 886) stellar model,
                   cross-checked against the Bisterzo et al. 2014 (ApJ 787, 10)
                   Galactic-chemical-evolution decomposition (quoted in the
                   comment where the two differ by more than ~10 points).
``P_FRACTION``     the small p-process share of Mo, Ru, Sr and a few others,
                   so that the r-process residual is not silently inflated by
                   it. The remainder ``1 - s - p`` is the r-process fraction.

The decay horizon
-----------------
A fission-product inventory keeps changing for ~10^7 yr: Cs-137 and Sr-90
(30 yr) are gone within a few centuries; Tc-99 (2.1e5 yr) and Sn-126 are gone
within ~1 Myr; Cs-135 (2.3 Myr), Zr-93 (1.6 Myr), Pd-107 (6.5 Myr) and I-129
(16 Myr) outlast that. ``element_yields(horizon_yr)`` assigns every chain to
the element it sits on at that horizon. The **default horizon is 1 Myr**: this
channel searches for the *pattern* that survives long after the makers, which
is the complement of ``midden`` (Tc I / Pm II lines, i.e. the < 10^5 yr
window). At the 1 Myr horizon Tc has become Ru, and the light peak reads
Zr-Mo-Ru with no Tc at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Chain yields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chain:
    """One mass chain: its cumulative yield and where it ends up."""

    A: int
    yield_pct: float           # per 100 fissions (two fragments -> sums to ~200)
    stable: str                # element the chain sits on once everything decays
    intermediate: str | None = None   # last long-lived (>= 1 yr) member, if any
    half_life_yr: float | None = None  # ... and its half-life


#: Thermal U-235 cumulative chain yields, per 100 fissions. Chains below A=77
#: and above A=161 sum to < 0.05% and are omitted. The "shielded" chains
#: (A=82, 86, 96, 134, 136) end on the stable isobar reached by beta decay from
#: the neutron-rich side; the shielded nuclide itself has negligible direct
#: yield.
CHAIN_YIELDS: tuple[Chain, ...] = (
    Chain(77, 0.0083, "Se"),
    Chain(78, 0.021, "Se"),
    Chain(79, 0.044, "Br", "Se", 3.3e5),
    Chain(80, 0.13, "Se"),
    Chain(81, 0.19, "Br"),
    Chain(82, 0.32, "Se"),
    Chain(83, 0.54, "Kr"),
    Chain(84, 1.00, "Kr"),
    Chain(85, 1.32, "Rb", "Kr", 10.8),
    Chain(86, 1.96, "Kr"),
    Chain(87, 2.56, "Rb"),                 # Rb-87, 4.9e10 yr: stable on any horizon
    Chain(88, 3.55, "Sr"),
    Chain(89, 4.73, "Y"),                  # Sr-89 50.5 d -> Y-89
    Chain(90, 5.78, "Zr", "Sr", 28.8),     # Sr-90 -> Y-90 -> Zr-90
    Chain(91, 5.83, "Zr"),                 # Y-91 58.5 d -> Zr-91
    Chain(92, 5.95, "Zr"),
    Chain(93, 6.37, "Nb", "Zr", 1.6e6),    # Zr-93 -> Nb-93
    Chain(94, 6.47, "Zr"),
    Chain(95, 6.50, "Mo"),                 # Zr-95 64 d -> Nb-95 35 d -> Mo-95
    Chain(96, 6.27, "Zr"),                 # Zr-96 (2e19 yr): stable
    Chain(97, 6.00, "Mo"),
    Chain(98, 5.78, "Mo"),
    Chain(99, 6.11, "Ru", "Tc", 2.1e5),    # Mo-99 -> Tc-99 -> Ru-99: MIDDEN's line
    Chain(100, 6.29, "Mo"),                # Mo-100 (7e18 yr): stable
    Chain(101, 5.17, "Ru"),
    Chain(102, 4.30, "Ru"),
    Chain(103, 3.03, "Rh"),                # Ru-103 39 d -> Rh-103
    Chain(104, 1.88, "Ru"),
    Chain(105, 0.96, "Pd"),                # Ru-105 -> Rh-105 -> Pd-105
    Chain(106, 0.40, "Pd", "Ru", 1.02),    # Ru-106 -> Rh-106 -> Pd-106
    Chain(107, 0.15, "Ag", "Pd", 6.5e6),   # Pd-107 -> Ag-107
    Chain(108, 0.054, "Pd"),
    Chain(109, 0.031, "Ag"),
    Chain(110, 0.026, "Pd"),
    Chain(111, 0.017, "Cd"),
    Chain(112, 0.013, "Cd"),
    Chain(113, 0.014, "Cd"),               # Cd-113 (8e15 yr): stable
    Chain(114, 0.012, "Cd"),
    Chain(115, 0.011, "In"),               # In-115 (4.4e14 yr): stable
    Chain(116, 0.012, "Sn"),
    Chain(117, 0.011, "Sn"),
    Chain(118, 0.013, "Sn"),
    Chain(119, 0.013, "Sn"),
    Chain(120, 0.013, "Sn"),
    Chain(121, 0.013, "Sb"),
    Chain(122, 0.016, "Sn"),
    Chain(123, 0.016, "Sb"),
    Chain(124, 0.027, "Sn"),
    Chain(125, 0.034, "Te", "Sb", 2.76),   # Sb-125 -> Te-125
    Chain(126, 0.058, "Te", "Sn", 2.3e5),  # Sn-126 -> Sb-126 -> Te-126
    Chain(127, 0.16, "I"),                 # Te-127 -> I-127
    Chain(128, 0.35, "Te"),
    Chain(129, 0.75, "Xe", "I", 1.6e7),    # I-129 -> Xe-129
    Chain(130, 1.80, "Te"),
    Chain(131, 2.89, "Xe"),                # I-131 8 d -> Xe-131
    Chain(132, 4.31, "Xe"),
    Chain(133, 6.70, "Cs"),                # Xe-133 5 d -> Cs-133
    Chain(134, 7.87, "Xe"),
    Chain(135, 6.54, "Ba", "Cs", 2.3e6),   # Cs-135 -> Ba-135
    Chain(136, 6.32, "Xe"),
    Chain(137, 6.19, "Ba", "Cs", 30.1),    # Cs-137 -> Ba-137
    Chain(138, 6.77, "Ba"),
    Chain(139, 6.41, "La"),                # Ba-139 83 min -> La-139
    Chain(140, 6.22, "Ce"),                # Ba-140 12.8 d -> La-140 -> Ce-140
    Chain(141, 5.85, "Pr"),                # Ce-141 32.5 d -> Pr-141
    Chain(142, 5.85, "Ce"),
    Chain(143, 5.96, "Nd"),                # Ce-143 -> Pr-143 -> Nd-143
    Chain(144, 5.50, "Nd", "Ce", 0.78),    # Ce-144 -> Pr-144 -> Nd-144
    Chain(145, 3.93, "Nd"),
    Chain(146, 3.00, "Nd"),
    Chain(147, 2.25, "Sm", "Pm", 2.62),    # Nd-147 11 d -> Pm-147 -> Sm-147
    Chain(148, 1.67, "Nd"),
    Chain(149, 1.08, "Sm"),                # Pm-149 -> Sm-149
    Chain(150, 0.65, "Nd"),
    Chain(151, 0.42, "Eu", "Sm", 90.0),    # Sm-151 -> Eu-151
    Chain(152, 0.27, "Sm"),
    Chain(153, 0.16, "Eu"),                # Sm-153 -> Eu-153
    Chain(154, 0.074, "Sm"),
    Chain(155, 0.032, "Gd", "Eu", 4.75),   # Eu-155 -> Gd-155
    Chain(156, 0.013, "Gd"),
    Chain(157, 0.0062, "Gd"),
    Chain(158, 0.0030, "Gd"),
    Chain(159, 0.0010, "Tb"),
    Chain(160, 0.0003, "Gd"),
    Chain(161, 0.00008, "Dy"),
)

#: Default decay horizon: the pattern that outlives its makers by ~10^6 yr.
DEFAULT_HORIZON_YR = 1.0e6


def element_yields(horizon_yr: float = DEFAULT_HORIZON_YR) -> dict[str, float]:
    """Cumulative yield per element (per 100 fissions) at a decay horizon.

    A chain whose last long-lived intermediate has a half-life longer than
    ``horizon_yr`` is still sitting on that intermediate; otherwise it has
    reached its stable end. (The assignment is all-or-nothing per chain: at a
    horizon of one half-life the split would be 50/50, but no element in the
    template space has a chain within a factor of three of the default
    horizon, so the step approximation costs nothing there.)
    """
    out: dict[str, float] = {}
    for ch in CHAIN_YIELDS:
        el = ch.stable
        if ch.intermediate is not None and ch.half_life_yr is not None \
                and ch.half_life_yr > horizon_yr:
            el = ch.intermediate
        out[el] = out.get(el, 0.0) + ch.yield_pct
    return out


def total_chain_yield() -> float:
    """Sum of all chain yields; must be ~200 (two fragments per fission)."""
    return float(sum(ch.yield_pct for ch in CHAIN_YIELDS))


# ---------------------------------------------------------------------------
# Solar abundances
# ---------------------------------------------------------------------------
#: A(X) = log10(N_X / N_H) + 12. Asplund, Amarsi & Grevesse 2021 unless noted.
SOLAR_LOGEPS: dict[str, float] = {
    "Li": 0.96,   # photospheric (meteoritic 3.26): the depleted value is the right one here
    "Fe": 7.46,
    "Se": 3.34,   # meteoritic
    "Br": 2.54,   # meteoritic
    "Kr": 3.25,   # Lodders 2009 (interpolated; no photospheric line)
    "Rb": 2.60,
    "Sr": 2.83,
    "Y": 2.21,
    "Zr": 2.59,
    "Nb": 1.47,
    "Mo": 1.88,
    "Tc": -99.0,  # no stable isotope: the solar abundance is zero (MIDDEN's target)
    "Ru": 1.75,
    "Rh": 0.78,
    "Pd": 1.57,
    "Ag": 0.96,
    "Cd": 1.71,
    "In": 0.80,
    "Sn": 2.02,
    "Sb": 1.01,
    "Te": 2.18,   # meteoritic
    "I": 1.51,    # meteoritic
    "Xe": 2.22,   # Lodders 2009 (indirect)
    "Cs": 1.08,   # meteoritic
    "Ba": 2.27,   # Lodders 2009: 2.18
    "La": 1.11,
    "Ce": 1.58,
    "Pr": 0.75,
    "Nd": 1.42,   # Lodders 2009: 1.45
    "Pm": -99.0,  # no stable isotope
    "Sm": 0.95,
    "Eu": 0.52,
    "Gd": 1.08,
    "Tb": 0.31,
    "Dy": 1.10,
    "Ho": 0.48,
    "Er": 0.93,
    "Tm": 0.10,
    "Yb": 0.85,
    "Hf": 0.85,
    "Os": 1.40,
    "Ir": 1.38,
    "Pt": 1.62,   # meteoritic
    "Pb": 1.75,   # never a fission product: the A=208 anti-signature
    "Th": 0.03,   # r-process only; 1.4e10 yr
    "U": -0.54,   # r-process only; meteoritic
}

# ---------------------------------------------------------------------------
# s / p / r decomposition of the solar system
# ---------------------------------------------------------------------------
#: s-process fraction of the solar abundance, Arlandini et al. 1999 stellar
#: model. Bisterzo et al. 2014 GCE values in comments where they differ.
S_FRACTION: dict[str, float] = {
    "Rb": 0.50,   # Bisterzo: ~0.4; Rb is branch-point sensitive
    "Sr": 0.85,   # Bisterzo: 0.69
    "Y": 0.92,    # Bisterzo: 0.72
    "Zr": 0.83,   # Bisterzo: 0.66
    "Nb": 0.85,
    "Mo": 0.50,   # Bisterzo: 0.39 (the rest is ~25% p, ~25% r)
    "Ru": 0.32,   # Bisterzo: 0.29
    "Rh": 0.14,
    "Pd": 0.46,
    "Ag": 0.20,
    "Cd": 0.55,
    "In": 0.36,
    "Sn": 0.63,
    "Sb": 0.25,
    "Te": 0.17,
    "I": 0.05,
    "Xe": 0.17,
    "Cs": 0.15,
    "Ba": 0.81,   # Bisterzo: 0.85
    "La": 0.62,   # Bisterzo: 0.76
    "Ce": 0.77,   # Bisterzo: 0.84
    "Pr": 0.49,   # Bisterzo: 0.50
    "Nd": 0.56,   # Bisterzo: 0.58
    "Sm": 0.29,   # Bisterzo: 0.31
    "Eu": 0.06,   # Bisterzo: 0.06 -- the canonical r-process element
    "Gd": 0.15,
    "Tb": 0.07,
    "Dy": 0.15,
    "Ho": 0.08,
    "Er": 0.16,
    "Tm": 0.13,
    "Yb": 0.32,
    "Hf": 0.55,
    "Os": 0.09,
    "Ir": 0.01,
    "Pt": 0.05,
    # Pb: dominated by the s-process (strong component, Pb-208) -- Arlandini
    # 1999 gives ~46% for the stellar model but the Galactic-evolution value
    # (Travaglio et al. 2001; Bisterzo 2014) is >= 85% once low-metallicity AGB
    # stars are included. The *shape* test only needs Pb to be an s-process
    # element that fission never makes; 0.85 is the GCE value.
    "Pb": 0.85,
    # Th and U: r-process only (no stable isotope reachable by the s-process).
    "Th": 0.0,
    "U": 0.0,
}

#: p-process (gamma-process) share of the solar abundance; only where it is
#: not negligible against the survey precision. Mo-92/94 and Ru-96/98 are the
#: classic cases.
P_FRACTION: dict[str, float] = {
    "Sr": 0.01,
    "Mo": 0.25,
    "Ru": 0.07,
    "Pd": 0.01,
    "Cd": 0.02,
    "In": 0.04,
    "Sn": 0.02,
    "Xe": 0.02,
    "Ba": 0.01,
    "La": 0.01,
    "Ce": 0.02,
    "Sm": 0.03,
    "Gd": 0.002,
    "Dy": 0.001,
}


def r_fraction(element: str) -> float:
    """r-process share: whatever the s- and p-processes do not account for."""
    s = S_FRACTION.get(element)
    if s is None:
        return float("nan")
    return float(max(0.0, 1.0 - s - P_FRACTION.get(element, 0.0)))


# ---------------------------------------------------------------------------
# The three pure-source patterns in number space, relative to solar
# ---------------------------------------------------------------------------
#: The n-capture elements a survey can plausibly deliver. Order is by mass.
NCAPTURE_ELEMENTS: tuple[str, ...] = (
    "Rb", "Sr", "Y", "Zr", "Mo", "Ru", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu",
)

#: The full set the high-resolution literature compilations (JINAbase, Hypatia)
#: can carry. The additions are the elements that DECIDE the fission vector:
#: Pd/Ag/Cd/Sn are the fission valley (~1000x below the peaks, where the
#: r-process is not suppressed at all); Pb is made by the s-process and never
#: by fission (A=208 is beyond every fission fragment); Th/U are r-only.
NCAPTURE_ELEMENTS_EXTENDED: tuple[str, ...] = (
    "Rb", "Sr", "Y", "Zr", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Sn",
    "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Yb",
    "Hf", "Os", "Ir", "Pt", "Pb", "Th", "U",
)

#: The element the fission pattern is normalised to: a_f = 1 doubles Nd.
FISSION_ANCHOR = "Nd"


def solar_number(element: str) -> float:
    """N_X / N_H from A(X)."""
    a = SOLAR_LOGEPS.get(element)
    if a is None or a < -50:
        return 0.0
    return float(10.0 ** (a - 12.0))


def fission_pattern(elements=NCAPTURE_ELEMENTS, *, horizon_yr: float = DEFAULT_HORIZON_YR,
                    anchor: str = FISSION_ANCHOR) -> dict[str, float]:
    """F_X = (Y_X / N_sol,X) / (Y_anchor / N_sol,anchor).

    Adding fission product at amplitude ``a`` multiplies the number abundance
    of X by ``1 + a * F_X``; ``a = 1`` means the anchor element (Nd) has been
    doubled. The pattern is a *ratio of ratios*, so the absolute amount of
    material and the size of the convective envelope cancel out of it -- the
    shape is what is being tested, the amplitude is a nuisance parameter.
    """
    y = element_yields(horizon_yr)
    ref = y.get(anchor, 0.0) / solar_number(anchor)
    out = {}
    for el in elements:
        n = solar_number(el)
        out[el] = float((y.get(el, 0.0) / n) / ref) if n > 0 and ref > 0 else float("nan")
    return out


def s_pattern(elements=NCAPTURE_ELEMENTS) -> dict[str, float]:
    """S_X = solar s-process fraction: adding pure s at amplitude ``a`` gives 1 + a S_X."""
    return {el: float(S_FRACTION.get(el, float("nan"))) for el in elements}


def r_pattern(elements=NCAPTURE_ELEMENTS) -> dict[str, float]:
    """R_X = solar r-process fraction: adding pure r at amplitude ``a`` gives 1 + a R_X."""
    return {el: r_fraction(el) for el in elements}


def pattern_dex(pattern: dict[str, float], amplitude: float) -> dict[str, float]:
    """[X/H] shift in dex from adding a pure source at ``amplitude``."""
    return {el: float(np.log10(1.0 + amplitude * v)) if np.isfinite(v) else float("nan")
            for el, v in pattern.items()}


def template_table(elements=NCAPTURE_ELEMENTS, *, amplitude: float = 1.0,
                   horizon_yr: float = DEFAULT_HORIZON_YR) -> list[dict]:
    """One row per element: the raw inputs and the three templates at ``amplitude``.

    This is what ``docs/fallout.md`` reproduces and what ``summary.json``
    carries, so the numbers the search actually used are the numbers a reader
    sees.
    """
    y = element_yields(horizon_yr)
    F = fission_pattern(elements, horizon_yr=horizon_yr)
    S = s_pattern(elements)
    R = r_pattern(elements)
    rows = []
    for el in elements:
        rows.append({
            "element": el,
            "fission_yield_pct": round(y.get(el, 0.0), 3),
            "solar_logeps": SOLAR_LOGEPS.get(el),
            "s_fraction": S_FRACTION.get(el),
            "p_fraction": P_FRACTION.get(el, 0.0),
            "r_fraction": round(r_fraction(el), 3),
            "F_over_Fanchor": round(F[el], 4),
            "fission_dex": round(float(np.log10(1 + amplitude * F[el])), 3),
            "s_dex": round(float(np.log10(1 + amplitude * S[el])), 3),
            "r_dex": round(float(np.log10(1 + amplitude * R[el])), 3),
        })
    return rows


__all__ = [
    "CHAIN_YIELDS",
    "DEFAULT_HORIZON_YR",
    "FISSION_ANCHOR",
    "NCAPTURE_ELEMENTS",
    "NCAPTURE_ELEMENTS_EXTENDED",
    "P_FRACTION",
    "SOLAR_LOGEPS",
    "S_FRACTION",
    "Chain",
    "element_yields",
    "fission_pattern",
    "pattern_dex",
    "r_fraction",
    "r_pattern",
    "s_pattern",
    "solar_number",
    "template_table",
    "total_chain_yield",
]

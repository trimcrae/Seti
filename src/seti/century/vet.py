"""The contamination gauntlet for CENTURY --- every verdict a pure function.

Ordered so the cheapest and most lethal kills run first, and every kill is
named after the systematic it removes, not after a quality cut:

``high_pm``               a star that moved arcseconds across the century is
                          measured against different sky, different blends and,
                          on the earliest plates, possibly a different aperture;
``long_period_giant``     Miras / SR / L variables and red giants have huge,
                          slow, irregular amplitude and mean changes (Tang et
                          al.'s DASCH K giants); they are excluded by class
                          and by colour;
``too_bright``            photographic saturation compresses amplitude
                          non-linearly and is series-dependent;
``vanished_not_ceased``   a star not recovered on plates deep enough to show
                          it is a vanishing-source claim --- the contested
                          2026 VASCO-adjacent class --- not a cessation;
``blend_transition``      the blend fraction jumps at the transition;
``series_disjoint``       pre and post blocks share no plate series;
``same_series_fails``     the cessation does not hold inside a common series;
``menzel_gap_transition`` the transition falls in the 1954--1970 hole (kept as
                          a lower-confidence class, not killed);
``mean_flux_changed``     after the field-ensemble step correction, the mean
                          moved across the transition (a fade is a fade);
``single_plate_evidence`` any statistic resting on fewer than the required
                          number of plates.
"""

from __future__ import annotations

import re

import numpy as np

from .pulsators import LPV_CLASSES, pulsator_class

LPV_TYPE_RE = re.compile(r"(?i)^(M|SR|SRA|SRB|SRC|SRD|SRS|L|LB|LC|LPV|OSARG|MIRA)(\b|[:/|+])")
PERIODIC_TYPE_RE = re.compile(
    r"(?i)^(RR|RRAB|RRC|RRD|DCEP|CEP|CW|CWA|CWB|RV|RVA|RVB|DSCT|HADS|SXPHE|EA|EB|EW|E\b|"
    r"BCEP|ACV|ELL|RS|BY|ROT|ACYG|SPB|GDOR|ZZ|BLAP|EC|ED|ESD|ACEP|ANRRD|T2CEP)")

MUNDANE_KEYS = ("high_pm", "long_period_giant", "too_bright", "vanished_not_ceased",
                "blend_transition", "series_disjoint", "same_series_fails",
                "mean_flux_changed", "single_plate_evidence", "not_periodic_class",
                "catalogue_period_suspect", "geometric_period")

# An eclipse or ellipsoidal period is an ORBIT.  It cannot stop without the
# system being destroyed, so a "cessation" in one is a statement about the
# photometry (docs/century.md §6.3).
GEOMETRIC_TYPE_RE = re.compile(r"(?i)^(EA|EB|EW|E|ELL|EC|ED|ESD|EP|EL)(\b|[:/|+(])")


def is_geometric_type(vtype: str) -> bool:
    return bool(vtype) and GEOMETRIC_TYPE_RE.match(str(vtype).strip()) is not None


def is_lpv_type(vtype: str) -> bool:
    return bool(vtype) and LPV_TYPE_RE.match(str(vtype).strip()) is not None


def is_periodic_type(vtype: str) -> bool:
    return bool(vtype) and PERIODIC_TYPE_RE.match(str(vtype).strip()) is not None


def vet_row(row: dict, conf: dict | None = None) -> dict:
    """Apply the gauntlet to one screened star.  Returns flags and a verdict.

    ``row`` is a flat dict of the screen output (keys prefixed ``cess_``,
    ``fade_``, ``rust_``) plus any context (``pm_total_masyr``, ``bp_rp``,
    ``vtype``, ``mag_cat``).  ``verdict`` is one of ``survivor`` (some
    candidate statistic passed and nothing killed it), ``killed:<key>`` or
    ``not_candidate``.
    """
    c = conf or {}
    pm_max = float(c.get("pm_max_masyr", 50.0))
    bright = float(c.get("bright_limit_mag", 8.0))
    red = float(c.get("lpv_colour_min", 1.5))
    mean_max = float(c.get("mean_shift_max_mag", 0.10))
    kills: list[str] = []
    notes: list[str] = []

    def g(k, default=float("nan")):
        v = row.get(k, default)
        try:
            return float(v) if v is not None and v != "" else default
        except (TypeError, ValueError):
            return default

    cand_cess = str(row.get("cess_status", "")) == "cessation"
    cand_fade = bool(row.get("fade_is_fade", False))
    cand_rust = bool(row.get("rust_is_rust", False))
    any_cand = cand_cess or cand_fade or cand_rust

    pm = g("pm_total_masyr")
    if np.isfinite(pm) and pm > pm_max:
        kills.append("high_pm")
    vtype = str(row.get("vtype", "") or "")
    bprp = g("bp_rp")
    pcls = str(row.get("pulsator_class", "") or "") or pulsator_class(vtype)
    if is_lpv_type(vtype) or (np.isfinite(bprp) and bprp > red and g("period_cat") > 50):
        if cand_cess and not (cand_fade or cand_rust) and pcls in LPV_CLASSES:
            # A catalogued Mira / SRa is in the cessation population on
            # purpose.  Its slow irregular mean and amplitude changes are what
            # this kill exists for, and they poison the FADE and SCATTER
            # questions --- but a regular LPV whose period STOPPED is exactly
            # the cessation question.  The known astrophysical mimics (thermal
            # pulses: R Hya, T UMi, W Dra; Mira -> SR transitions) are named
            # here as the first thing to trace, not silently absorbed.
            notes.append("lpv_cessation_trace_to_period_evolution")
        else:
            kills.append("long_period_giant")
    mag = g("mag_cat", g("median_mag"))
    if np.isfinite(mag) and mag < bright:
        kills.append("too_bright")

    if cand_cess:
        if is_geometric_type(vtype) and not pcls:
            kills.append("geometric_period")
        if vtype and not is_periodic_type(vtype) and not pcls:
            notes.append("not_periodic_class")
        if str(row.get("cess_status", "")) == "vanished_not_ceased" or \
                "vanished_not_ceased" in str(row.get("cess_flags", "")):
            kills.append("vanished_not_ceased")
        if "blend_transition" in str(row.get("cess_flags", "")):
            kills.append("blend_transition")
        if "series_disjoint" in str(row.get("cess_flags", "")):
            kills.append("series_disjoint")
        sss = str(row.get("same_series_status", ""))
        if sss and sss not in ("cessation", "untestable", ""):
            kills.append("same_series_fails")
        if sss == "untestable":
            notes.append("same_series_untestable")
        if "transition_at_menzel_gap" in str(row.get("cess_flags", "")):
            notes.append("menzel_gap_transition")
        ms = g("cess_mean_shift_corrected", g("cess_mean_shift_mag"))
        if np.isfinite(ms) and abs(ms) > mean_max:
            kills.append("mean_flux_changed")
        if g("cess_n_post_informative", 0) < 2 or g("cess_n_pre_detected", 0) < 2:
            kills.append("single_plate_evidence")
    if cand_fade:
        if g("fade_n_years", 0) < 15:
            kills.append("single_plate_evidence")
        if "depends_on_series" in str(row.get("fade_flags", "")):
            kills.append("series_disjoint")
    if cand_rust:
        if g("rust_n_seasons", 0) < 8:
            kills.append("single_plate_evidence")

    kills = list(dict.fromkeys(kills))
    if not any_cand:
        verdict = "not_candidate"
    elif kills:
        verdict = "killed:" + kills[0]
    else:
        verdict = "survivor"
    return {"verdict": verdict, "kills": ";".join(kills), "notes": ";".join(notes),
            "candidate_cessation": cand_cess, "candidate_fade": cand_fade,
            "candidate_rust": cand_rust}


__all__ = ["GEOMETRIC_TYPE_RE", "LPV_TYPE_RE", "MUNDANE_KEYS", "PERIODIC_TYPE_RE",
           "is_geometric_type", "is_lpv_type", "is_periodic_type", "vet_row"]

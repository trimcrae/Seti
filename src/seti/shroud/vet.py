"""SHROUD vetting — the kill-tests every candidate must survive.

Four independent families, all of which must pass:

**1. It moved.**  A star with mu = 200 mas/yr travels 12.6" between the POSS-I
epoch and Gaia DR3.  At a 5" match radius it is simply *absent* from its 1953
position, and it dominates any naive "vanished source" list — Solano+2022
removed 180 by proper motion, which for a 3x10^5 source list is implausibly few.
:func:`epoch_propagation_check` searches a radius set by the largest proper
motion considered and propagates every modern neighbour *back* to the plate
epoch, rather than propagating the plate position forward with a proper motion
the vanished source by construction does not have.

**2. The infrared match is a coincidence.**  At the published 5" radius the
chance-match probability against AllWISE is ~1% at the Galactic pole and of
order unity in the plane: ``P = 1 - exp(-n * pi * r^2)``.  This is not a footnote
— it is the single largest systematic in the sample, and
:func:`chance_match_probability` makes it a per-object number driven by the
*locally measured* source density.

**3. The repository contamination ledger** (docs/channel-brief.md §4), inherited
and not re-derived: a single-band anomaly is an artefact; a W4-only excess is
cirrus; a negative W1-W2 is a blend; fitted dust hotter than 1800 K is a
companion photosphere.

**4. The Forés-Toribio & Kochanek (2026, arXiv:2604.05019) discriminant.**
Merger remnants are 10-100x MORE luminous than their progenitors at late
phases; genuine disappearance remnants are ~10x DIMMER.  Asymmetric dust cannot
manufacture a factor ~100.  Because the present-day bolometric output of an
obscured object is dominated by its reprocessed infrared, that ratio *is* the
energy-budget eta — the same measurement read at different thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import sed as sedmod

_ARCSEC_PER_DEG = 3600.0


# --- geometry ---------------------------------------------------------------
def angular_separation_arcsec(ra1, dec1, ra2, dec2):
    """Haversine separation in arcsec; broadcasts over arrays."""
    ra1, dec1 = np.radians(np.asarray(ra1, float)), np.radians(np.asarray(dec1, float))
    ra2, dec2 = np.radians(np.asarray(ra2, float)), np.radians(np.asarray(dec2, float))
    d = 2.0 * np.arcsin(np.sqrt(np.clip(
        np.sin((dec2 - dec1) / 2.0) ** 2
        + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2.0) ** 2, 0.0, 1.0)))
    return np.degrees(d) * _ARCSEC_PER_DEG


def propagate_position(ra_deg, dec_deg, pmra_mas_yr, pmdec_mas_yr, dt_yr):
    """Move a position by its proper motion over ``dt_yr`` years.

    ``pmra_mas_yr`` is mu_alpha* (already including the cos(dec) factor), the
    Gaia convention.  Returns ``(ra_deg, dec_deg)``.
    """
    dec = np.asarray(dec_deg, float)
    ddec = np.asarray(pmdec_mas_yr, float) * dt_yr / 1000.0 / _ARCSEC_PER_DEG
    cosd = np.cos(np.radians(np.clip(dec, -89.999, 89.999)))
    dra = np.asarray(pmra_mas_yr, float) * dt_yr / 1000.0 / _ARCSEC_PER_DEG / cosd
    return (np.asarray(ra_deg, float) + dra) % 360.0, dec + ddec


def epoch_propagation_check(ra_deg: float, dec_deg: float,
                            neighbours: pd.DataFrame, cfg: dict,
                            epoch_poss1: float | None = None,
                            neighbour_epoch: float | None = None) -> dict:
    """Did a high-proper-motion star simply move off the POSS-I position?

    ``neighbours`` must carry ``ra_deg``, ``dec_deg``, ``pmra``, ``pmdec``
    (mas/yr) at ``neighbour_epoch``.  Each is propagated *back* to the plate
    epoch and tested against the vanished position.
    """
    ep = cfg.get("epochs", {})
    pm = cfg.get("proper_motion", {})
    t0 = float(epoch_poss1 if epoch_poss1 is not None
               else ep.get("poss1_default", 1953.0))
    t1 = float(neighbour_epoch if neighbour_epoch is not None
               else ep.get("gaia_dr3", 2016.0))
    r_match = float(pm.get("match_radius_arcsec", 3.0))
    min_disp = float(pm.get("min_displacement_arcsec", 2.0))

    out = {"pm_recovered": False, "pm_back_propagated_sep_arcsec": np.nan,
           "pm_total_mas_yr": np.nan, "pm_n_neighbours": 0,
           "pm_displacement_arcsec": np.nan}
    if neighbours is None or not len(neighbours):
        return out
    need = {"ra_deg", "dec_deg", "pmra", "pmdec"}
    if not need <= set(neighbours.columns):
        return out
    out["pm_n_neighbours"] = int(len(neighbours))

    dt = t0 - t1                                  # negative: propagate backwards
    ra_b, dec_b = propagate_position(neighbours["ra_deg"].to_numpy(),
                                     neighbours["dec_deg"].to_numpy(),
                                     np.nan_to_num(neighbours["pmra"].to_numpy(), nan=0.0),
                                     np.nan_to_num(neighbours["pmdec"].to_numpy(), nan=0.0),
                                     dt)
    sep = angular_separation_arcsec(ra_deg, dec_deg, ra_b, dec_b)
    sep_now = angular_separation_arcsec(ra_deg, dec_deg,
                                        neighbours["ra_deg"].to_numpy(),
                                        neighbours["dec_deg"].to_numpy())
    mu = np.hypot(np.nan_to_num(neighbours["pmra"].to_numpy(), nan=0.0),
                  np.nan_to_num(neighbours["pmdec"].to_numpy(), nan=0.0))
    i = int(np.nanargmin(sep))
    out["pm_back_propagated_sep_arcsec"] = float(sep[i])
    out["pm_total_mas_yr"] = float(mu[i])
    out["pm_displacement_arcsec"] = float(sep_now[i])
    # It counts as "it moved" only if the star is now genuinely elsewhere:
    # a source that never left the match radius explains nothing.
    out["pm_recovered"] = bool(sep[i] <= r_match and sep_now[i] >= min_disp)
    return out


# --- chance coincidence -----------------------------------------------------
def chance_match_probability(radius_arcsec: float, density_per_deg2: float) -> float:
    """P(at least one unrelated source inside the radius) for a Poisson field."""
    if not np.isfinite(density_per_deg2) or density_per_deg2 <= 0:
        return 0.0
    area_deg2 = math.pi * (float(radius_arcsec) / _ARCSEC_PER_DEG) ** 2
    return float(1.0 - math.exp(-density_per_deg2 * area_deg2))


def local_source_density(n_sources_in_radius: float, radius_arcsec: float) -> float:
    """Sources per square degree from a count inside a search radius."""
    area = math.pi * (float(radius_arcsec) / _ARCSEC_PER_DEG) ** 2
    if area <= 0:
        return float("nan")
    return float(n_sources_in_radius) / area


def expected_chance_matches(n_sources: int, radius_arcsec: float,
                            density_per_deg2: float) -> float:
    """How many of ``n_sources`` matches are expected to be pure coincidence."""
    return float(n_sources) * chance_match_probability(radius_arcsec,
                                                       density_per_deg2)


def offset_positions(df: pd.DataFrame, sep_arcsec: float, seed: int = 0,
                     ra_col: str = "ra_deg", dec_col: str = "dec_deg"
                     ) -> pd.DataFrame:
    """Rotate every position by ``sep_arcsec`` in a random direction.

    The offset-position null is the correct way to measure the chance-match
    rate: it samples the *real* local source density along the same sightlines,
    with the same plate coverage and the same Galactic structure.  Assuming a
    spatially uniform random background instead is precisely the error Watters
    et al. 2026 identified in the VASCO Earth-shadow analysis, so this channel
    does not repeat it.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    d_deg = float(sep_arcsec) / _ARCSEC_PER_DEG
    dec = df[dec_col].to_numpy(float)
    ddec = d_deg * np.sin(theta)
    cosd = np.cos(np.radians(np.clip(dec, -89.999, 89.999)))
    dra = d_deg * np.cos(theta) / np.maximum(cosd, 1e-6)
    out = df.copy()
    out[ra_col] = (df[ra_col].to_numpy(float) + dra) % 360.0
    out[dec_col] = np.clip(dec + ddec, -90.0, 90.0)
    out["offset_realisation"] = seed
    return out


def chance_match_rate_from_null(n_real_matched: int, n_real: int,
                                n_null_matched: int, n_null: int) -> dict:
    """Genuinely-associated fraction from the offset-position null.

    ``f_chance`` is measured, not modelled; ``f_true`` is the excess of the real
    match fraction over it, which is the fraction of the sample whose infrared
    counterpart is a real physical association.
    """
    if n_real <= 0 or n_null <= 0:
        return {"f_match": None, "f_chance": None, "f_true": None,
                "n_expected_chance": None, "significance_sigma": None}
    f_m = n_real_matched / n_real
    f_c = n_null_matched / n_null
    # Binomial errors on both fractions.
    s_m = math.sqrt(max(f_m * (1 - f_m), 1e-12) / n_real)
    s_c = math.sqrt(max(f_c * (1 - f_c), 1e-12) / n_null)
    sig = (f_m - f_c) / math.hypot(s_m, s_c) if math.hypot(s_m, s_c) > 0 else None
    return {"f_match": f_m, "f_chance": f_c, "f_true": f_m - f_c,
            "n_expected_chance": f_c * n_real,
            "significance_sigma": sig}


# --- Forés-Toribio & Kochanek 2026 discriminant -----------------------------
def ftk_verdict(eta: float, cfg: dict) -> tuple[str, str]:
    """Progenitor-to-remnant luminosity ratio verdict."""
    eb = cfg.get("energy_budget", {})
    merger = float(eb.get("ftk_merger_ratio_min", 10.0))
    disap = float(eb.get("ftk_disappearance_ratio_max", 0.30))
    if not np.isfinite(eta):
        return ("UNDETERMINED", "no usable luminosity ratio")
    if eta >= merger:
        return ("MERGER_REMNANT_LIKE",
                f"remnant/progenitor = {eta:.1f} >= {merger:g}: the "
                "Forés-Toribio & Kochanek 2026 signature of a merger remnant, "
                "not a disappearance")
    if eta <= disap:
        return ("DISAPPEARANCE_LIKE",
                f"remnant/progenitor = {eta:.3f} <= {disap:g}: the object is "
                "genuinely dimmer than its progenitor; asymmetric dust cannot "
                "manufacture this")
    return ("OBSCURATION_LIKE",
            f"remnant/progenitor = {eta:.2f}: consistent with energy-conserving "
            "reprocessing")


# --- the ledger -------------------------------------------------------------
def ledger_vetoes(row, cfg: dict, fit_dust=None) -> list[str]:
    """Inherited contamination rules; returns the list of tripped vetoes."""
    v = cfg.get("vet", {})
    s = cfg.get("sed", {})
    p = cfg.get("plate", {})
    flags: list[str] = []

    def val(k):
        try:
            x = float(row.get(k, np.nan))
        except (TypeError, ValueError):
            return np.nan
        return x

    ir = {b: val(b) for b in ("2mass_j", "2mass_h", "2mass_ks",
                              "w1", "w2", "w3", "w4")}
    det = [b for b, m in ir.items() if np.isfinite(m)]
    if v.get("require_two_ir_bands", True) and len(det) < 2:
        flags.append("SINGLE_IR_BAND")
    if v.get("reject_w4_only", True) and det == ["w4"]:
        flags.append("W4_ONLY")
    if (v.get("reject_negative_w1w2", True) and np.isfinite(ir["w1"])
            and np.isfinite(ir["w2"])
            and (ir["w1"] - ir["w2"]) < float(
                cfg.get("classify", {}).get("blend_w1w2_max", -0.10))):
        flags.append("NEGATIVE_W1W2_BLEND")

    n_neigh = val("n_ir_neighbours")
    if np.isfinite(n_neigh) and n_neigh > float(v.get("max_ir_neighbours_in_psf", 1)):
        flags.append("IR_CONFUSION")

    # Plate-quality flags: the emulsion, not the sky.
    plate = val("poss1_e")
    lim = float(p.get("poss1_e_limit_mag", 20.0))
    if not np.isfinite(plate):
        plate, lim = val("poss1_o"), float(p.get("poss1_o_limit_mag", 21.0))
    if np.isfinite(plate):
        if lim - plate <= float(p.get("limit_proximity_mag", 0.7)):
            flags.append("PLATE_LIMIT_PROXIMITY")
        if plate <= float(p.get("saturation_mag", 12.0)):
            flags.append("PLATE_SATURATED")

    p_chance = val("p_chance_match")
    if np.isfinite(p_chance):
        cm = cfg.get("crossmatch", {})
        if p_chance >= float(cm.get("chance_match_reject_prob", 0.30)):
            flags.append("CHANCE_MATCH_LIKELY")
        elif p_chance >= float(cm.get("chance_match_warn_prob", 0.05)):
            flags.append("CHANCE_MATCH_POSSIBLE")

    if fit_dust is not None and getattr(fit_dust, "ok", False):
        t_d = float(getattr(fit_dust, "t_dust_k", np.nan))
        if np.isfinite(t_d) and t_d > float(s.get("tdust_max_physical_k", 1800.0)):
            flags.append("TDUST_UNPHYSICAL_COMPANION")
    return flags


# --- one-stop vetting -------------------------------------------------------
@dataclass
class VetResult:
    source_id: str
    survives: bool
    flags: list[str] = field(default_factory=list)
    ftk_class: str = "UNDETERMINED"
    ftk_reason: str = ""
    budget_verdict: str = ""
    eta_max: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k, val in d.items():
            if isinstance(val, float) and not np.isfinite(val):
                d[k] = None
        return d


def vet_object(row, cfg: dict, budget=None, fit_dust=None) -> VetResult:
    """Apply every kill-test to one classified object."""
    from .classify import _flag

    sid = str(row.get("source_id", ""))
    flags = ledger_vetoes(row, cfg, fit_dust)
    if _flag(row, "pm_recovered"):
        flags.append("HIGH_PM_RECOVERED")
    klass = str(row.get("class", ""))
    if klass and klass != "RESIDUAL_UNEXPLAINED":
        flags.append(f"CLASS_{klass}")

    ftk_class, ftk_reason = "UNDETERMINED", "no energy budget"
    verdict, eta = "", float("nan")
    if budget is not None:
        verdict = budget.verdict
        eta = float(budget.eta_max)
        ftk_class, ftk_reason = ftk_verdict(eta, cfg)
        if ftk_class == "MERGER_REMNANT_LIKE":
            flags.append("FTK_MERGER_REMNANT")
        if verdict in ("INSUFFICIENT_IR", "NO_HISTORICAL_PHOTOMETRY",
                       "NO_DEFICIT", "NO_VIABLE_PROGENITOR"):
            flags.append(f"BUDGET_{verdict}")

    survives = not flags
    return VetResult(sid, survives, flags, ftk_class, ftk_reason, verdict, eta)


def vet_table(df: pd.DataFrame, cfg: dict, budgets: dict | None = None,
              fits: dict | None = None) -> pd.DataFrame:
    """Vet a classified table; adds ``survives``, ``vet_flags``, FTK columns."""
    if not len(df):
        out = df.copy()
        for c, dt in (("survives", bool), ("vet_flags", object),
                      ("ftk_class", object), ("ftk_reason", object)):
            out[c] = pd.Series(dtype=dt)
        return out
    budgets, fits = budgets or {}, fits or {}
    rows = []
    for _, row in df.iterrows():
        sid = str(row.get("source_id", ""))
        r = vet_object(row, cfg, budgets.get(sid), fits.get(sid))
        rows.append({"survives": r.survives, "vet_flags": ";".join(r.flags),
                     "ftk_class": r.ftk_class, "ftk_reason": r.ftk_reason,
                     "budget_verdict": r.budget_verdict, "eta_max": r.eta_max})
    out = df.copy().reset_index(drop=True)
    for k in ("survives", "vet_flags", "ftk_class", "ftk_reason",
              "budget_verdict", "eta_max"):
        out[k] = [r[k] for r in rows]
    return out


def build_sed(row) -> sedmod.SED:
    """Turn a catalogue row into an :class:`~seti.shroud.sed.SED`."""
    mags, errs, limits = {}, {}, {}
    for b in sedmod.BANDS:
        try:
            m = float(row.get(b, np.nan))
        except (TypeError, ValueError):
            m = np.nan
        if np.isfinite(m):
            mags[b] = m
            try:
                e = float(row.get(f"{b}_err", np.nan))
            except (TypeError, ValueError):
                e = np.nan
            if np.isfinite(e):
                errs[b] = e
        else:
            try:
                lim = float(row.get(f"{b}_lim", np.nan))
            except (TypeError, ValueError):
                lim = np.nan
            if np.isfinite(lim):
                limits[b] = lim
    return sedmod.SED(source_id=str(row.get("source_id", "")), mags=mags,
                      errs=errs, limits=limits,
                      meta={"ra_deg": row.get("ra_deg"),
                            "dec_deg": row.get("dec_deg")})


__all__ = [
    "VetResult", "angular_separation_arcsec", "build_sed",
    "chance_match_probability", "chance_match_rate_from_null",
    "epoch_propagation_check", "expected_chance_matches", "ftk_verdict",
    "ledger_vetoes", "local_source_density", "offset_positions",
    "propagate_position", "vet_object", "vet_table",
]

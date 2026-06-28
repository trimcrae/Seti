"""Quantitative false-positive budget for the Gaia--WISE co-movement cut.

This substantiates the paper's central methodological claim.  Chance-aligned
background WISE sources within the cross-match radius are the dominant source of
spurious infrared excess (the Project Hephaistos failure mode).  We compute, for
the realistic white-dwarf population:

  * the expected number of chance-aligned CatWISE2020 sources per white dwarf,
    lambda = n_bg * pi * r^2;
  * the probability the co-movement cut removes such a contaminant, given the
    white dwarf's proper motion -- via the position test (the white dwarf has
    moved away from a static background by mu*Delta_t) OR the proper-motion
    test (the background's ~zero proper motion disagrees with the Gaia value);
  * the resulting rejection factor and residual contamination across the sample.

A background object is essentially static in proper motion compared with a
nearby, fast-moving white dwarf, so the cut's power grows with white-dwarf
proper motion -- which we quantify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ARCSEC2_PER_DEG2 = 3600.0**2


def chance_alignment_rate(n_bg_per_deg2: float, radius_arcsec: float) -> float:
    """Expected number of chance-aligned background sources within the radius."""
    area_deg2 = np.pi * radius_arcsec**2 / ARCSEC2_PER_DEG2
    return n_bg_per_deg2 * area_deg2


def _sigma_pm(w1mag: np.ndarray, pm_sigma0: float, ref_w1: float) -> np.ndarray:
    """CatWISE proper-motion uncertainty (mas/yr), degrading toward faint W1."""
    return pm_sigma0 * np.maximum(1.0, 10.0 ** (0.4 * (np.asarray(w1mag) - ref_w1)))


def removal_probability(
    pop: pd.DataFrame,
    thresholds: dict,
    bg_cfg: dict,
    dt_years: float,
) -> pd.Series:
    """Probability the co-movement cut removes a chance-aligned static background.

    A static background contaminant fails the cut if EITHER the epoch-propagated
    position offset (mu * dt) exceeds the position tolerance, OR the white-dwarf
    proper motion exceeds ``pm_sigma_max`` times the combined PM uncertainty (the
    background has ~zero proper motion, so the disagreement is ~mu).
    """
    cm = thresholds["contamination"]["comovement"]
    mu = np.hypot(pop["pmra"].to_numpy(), pop["pmdec"].to_numpy())  # mas/yr

    # Position test: WD has moved mu*dt from the (static) background position.
    pos_removed = (mu * abs(dt_years)) > (cm["max_position_offset_arcsec"] * 1000.0)

    # PM test: background mu ~ 0, Gaia mu large -> |Delta mu|/sigma > threshold.
    sig_w = _sigma_pm(pop["W1mag"].to_numpy(), bg_cfg["pm_sigma0_mas_yr"],
                      bg_cfg["pm_sigma_ref_w1"])
    sig_g = np.hypot(pop.get("pmra_error", 1.0), pop.get("pmdec_error", 1.0))
    sig_comb = np.hypot(sig_w, sig_g)
    pm_removed = (mu / sig_comb) > cm["pm_consistency_sigma_max"]

    return pd.Series((pos_removed | pm_removed).astype(float), index=pop.index)


def contamination_budget(cfg, pop: pd.DataFrame) -> dict:
    """Aggregate chance-alignment contamination before and after the cut."""
    bg = cfg.population["background"]
    epochs = cfg.catalogs.get("models", {}).get("epochs", {})
    dt = epochs.get("wise_mean_epoch", 2010.5) - epochs.get("gaia_ref_epoch", 2016.0)

    detected = pop[pop["detected"]].copy()
    n_wd = len(detected)
    lam = chance_alignment_rate(bg["source_density_per_deg2"], bg["match_radius_arcsec"])

    removed = removal_probability(detected, cfg.thresholds, bg, dt)
    n_before = lam * n_wd
    n_after = lam * float((1.0 - removed).sum())
    # Scale to the real 100 pc WISE-detected sample.
    per_draw = cfg.population["catalog"]["n_within_100pc"] / cfg.population["population"]["n_draw"]

    mu = np.hypot(detected["pmra"], detected["pmdec"])
    return {
        "lambda_per_wd": float(lam),
        "n_wd_detected_draw": int(n_wd),
        "chance_aligned_before_draw": float(n_before),
        "chance_aligned_after_draw": float(n_after),
        "rejection_factor": float(n_before / n_after) if n_after > 0 else float("inf"),
        "removed_fraction": float(removed.mean()),
        "chance_aligned_before_real": float(n_before * per_draw),
        "chance_aligned_after_real": float(n_after * per_draw),
        "median_pm_mas_yr": float(np.median(mu)),
    }


def efficacy_vs_pm(cfg, pop: pd.DataFrame, pm_bins=None) -> pd.DataFrame:
    """Removed fraction of chance-aligned contaminants in proper-motion bins."""
    bg = cfg.population["background"]
    epochs = cfg.catalogs.get("models", {}).get("epochs", {})
    dt = epochs.get("wise_mean_epoch", 2010.5) - epochs.get("gaia_ref_epoch", 2016.0)
    detected = pop[pop["detected"]].copy()
    removed = removal_probability(detected, cfg.thresholds, bg, dt)
    mu = np.hypot(detected["pmra"], detected["pmdec"]).to_numpy()

    pm_bins = pm_bins if pm_bins is not None else [0, 25, 50, 100, 200, 400, 1000]
    rows = []
    for lo, hi in zip(pm_bins[:-1], pm_bins[1:], strict=False):
        m = (mu >= lo) & (mu < hi)
        if m.sum() == 0:
            continue
        rows.append({"pm_lo": lo, "pm_hi": hi, "pm_mid": 0.5 * (lo + hi),
                     "n": int(m.sum()), "removed_fraction": float(removed[m].mean())})
    return pd.DataFrame(rows)


__all__ = ["chance_alignment_rate", "removal_probability", "contamination_budget",
           "efficacy_vs_pm"]

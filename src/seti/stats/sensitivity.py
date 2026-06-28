"""Projected occurrence-rate sensitivity for the white-dwarf search.

Given the realistic 100 pc white-dwarf population (:mod:`seti.population`) and the
injection-recovery completeness of the detection pipeline, we forecast the
occurrence-rate upper limit a *null* all-sky search would place on white-dwarf
Dyson-swarm-like infrared excess, as a function of waste-heat temperature
``T_dust`` and covering fraction ``tau``.

This is a forecast built from a population *model* calibrated to the published
Gaia EDR3 white-dwarf statistics and the CatWISE2020 depth -- not a measurement.
The same detection code path is exercised, so the forecast is self-consistent
with what the empirical search would do once archive access is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..population import generate_population
from .completeness import completeness_map
from .upper_limit import poisson_upper_limit


def forecast_sensitivity(
    cfg: Config,
    t_grid=None,
    tau_grid=None,
    seed: int = 11,
) -> pd.DataFrame:
    """Forecast N_eff and the 95% occurrence-rate upper limit per (T_dust, tau).

    Returns a tidy DataFrame with columns:
    ``t_dust_k, tau, recovered_fraction, n_detected_real, n_eff, f_upper_95``.
    """
    pop = generate_population(cfg, seed=seed)
    detected = pop[pop["detected"]].reset_index(drop=True)

    # Each Monte-Carlo draw represents (N_100pc / n_draw) real white dwarfs.
    n_draw = int(cfg.population["population"]["n_draw"])
    n_100pc = float(cfg.population["catalog"]["n_within_100pc"])
    per_draw = n_100pc / n_draw
    n_detected_real = len(detected) * per_draw

    inj = cfg.thresholds["stats"]["injection"]
    t_grid = t_grid if t_grid is not None else inj["t_dust_grid_k"]
    tau_grid = tau_grid if tau_grid is not None else inj["tau_grid"]

    cmap = completeness_map(detected, cfg.thresholds, t_grid=t_grid, tau_grid=tau_grid)

    mu0 = poisson_upper_limit(0, cfg.thresholds["stats"]["upper_limit_confidence"])
    cmap = cmap.copy()
    cmap["n_detected_real"] = n_detected_real
    cmap["n_eff"] = cmap["recovered_fraction"] * n_detected_real
    cmap["f_upper_95"] = np.where(cmap["n_eff"] > 0, mu0 / cmap["n_eff"], np.inf)
    return cmap


def minimum_detectable_tau(forecast: pd.DataFrame, recovered_threshold: float = 0.5):
    """For each temperature, the smallest tau recovered above ``recovered_threshold``."""
    rows = []
    for t in sorted(forecast["t_dust_k"].unique()):
        sub = forecast[forecast["t_dust_k"] == t].sort_values("tau")
        ok = sub[sub["recovered_fraction"] >= recovered_threshold]
        tau_min = float(ok["tau"].min()) if len(ok) else np.nan
        rows.append({"t_dust_k": float(t), "tau_min_detectable": tau_min})
    return pd.DataFrame(rows)


def headline_limit(forecast: pd.DataFrame) -> dict:
    """Best (smallest) forecast occurrence-rate upper limit and where it occurs."""
    finite = forecast[np.isfinite(forecast["f_upper_95"])]
    if finite.empty:
        return {"f_upper_95": float("inf")}
    best = finite.loc[finite["f_upper_95"].idxmin()]
    return {
        "f_upper_95": float(best["f_upper_95"]),
        "at_t_dust_k": float(best["t_dust_k"]),
        "at_tau": float(best["tau"]),
        "n_eff": float(best["n_eff"]),
        "n_detected_real": float(best["n_detected_real"]),
    }


__all__ = ["forecast_sensitivity", "minimum_detectable_tau", "headline_limit"]

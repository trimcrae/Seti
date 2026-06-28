"""Injection-recovery completeness map C(T_dust, tau).

We inject synthetic blackbody excesses onto real (or sample) white-dwarf SEDs
across a grid of dust temperature and fractional luminosity, run the same excess
detection, and record the recovered fraction per grid cell.  The resulting
completeness feeds the occurrence-rate limit so the published constraint is an
*intrinsic* one.  Fully self-contained -- no external data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..photometry import (
    band_freq_hz,
    flux_jy_to_mag,
    mag_to_flux_jy,
    planck_bnu,
)
from ..sed.excess import compute_excess, select_excess
from ..sed.predict import predict_photosphere


def _inject_excess(df: pd.DataFrame, t_dust: float, tau: float) -> pd.DataFrame:
    """Add a blackbody excess of bolometric fractional luminosity ``tau`` and
    temperature ``T_dust`` to W1/W2, consistent with ``sed.excess.fit_dust`` and
    ``seti.sample`` (Omega_dust = tau * Omega_WD * (T_WD/T_dust)^4)."""
    out = df.copy()
    teff = out["teff"].to_numpy()
    scale = out["sed_scale"].to_numpy() if "sed_scale" in out else None
    if scale is None:
        from ..sed.predict import predict_photosphere
        scale = predict_photosphere(out)["sed_scale"].to_numpy()
    omega_dust = tau * scale * (teff / t_dust) ** 4
    for b in ("W1", "W2"):
        nu = band_freq_hz(b)
        phot_jy = mag_to_flux_jy(out[f"{b}mag"].to_numpy(), b)
        excess_jy = omega_dust * np.pi * planck_bnu(t_dust, nu) * 1e26
        out[f"{b}mag"] = flux_jy_to_mag(phot_jy + excess_jy, b)
    return out


def completeness_map(
    base_df: pd.DataFrame,
    thresholds: dict,
    t_grid=None,
    tau_grid=None,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Return a tidy DataFrame of recovered fraction per (T_dust, tau) cell."""
    inj = thresholds["stats"]["injection"]
    t_grid = t_grid if t_grid is not None else inj["t_dust_grid_k"]
    tau_grid = tau_grid if tau_grid is not None else inj["tau_grid"]

    # Clean photospheric sample only (no pre-existing excess) for injection.
    # Predict the photospheric scale once so the injection is self-consistent.
    clean = predict_photosphere(base_df.copy())
    rows = []
    for t_dust in t_grid:
        for tau in tau_grid:
            injected = _inject_excess(clean, float(t_dust), float(tau))
            pred = predict_photosphere(injected)
            ex = compute_excess(pred, thresholds)
            recovered = select_excess(ex, thresholds)
            frac = float(recovered.mean()) if len(recovered) else 0.0
            rows.append({"t_dust_k": float(t_dust), "tau": float(tau),
                         "recovered_fraction": frac, "n": int(len(clean))})
    return pd.DataFrame(rows)

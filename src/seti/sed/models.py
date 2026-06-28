"""White-dwarf photospheric models for predicting W1/W2 photometry.

Two interchangeable predictors:

* ``BlackbodyModel`` -- a fully self-contained blackbody of temperature Teff,
  scaled by a single solid-angle factor fit to anchor bands.  This is exact in
  the Rayleigh-Jeans regime that WISE samples for white dwarfs and needs no
  external data, so the pipeline always runs.

* ``BergeronModel`` -- interpolates synthetic absolute magnitudes (Gaia/2MASS/
  WISE) from the Montreal/Bergeron WD photometry tables as a function of
  (Teff, logg).  A frozen snapshot lives under ``src/seti/data_assets/``.  The
  committed snapshot is a *synthetic stand-in* generated from blackbody SEDs
  (see ``scripts/make_bergeron_asset.py``); replace it with the real Montreal
  table (https://www.astro.umontreal.ca/~bergeron/CoolingModels/) for science
  runs.  Both models are reported in the manuscript as a robustness cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from ..photometry import band_freq_hz, planck_bnu

ANCHOR_BANDS_DEFAULT = ("J", "H", "Ks")
PREDICT_BANDS = ("W1", "W2")


@dataclass
class PredictedSED:
    """Predicted photospheric flux densities (Jy) and the fit scale factor."""

    fluxes_jy: dict[str, float]
    scale: float          # solid angle Omega = (R/d)^2 in steradian
    anchor_chi2: float    # goodness of the anchor fit


class BlackbodyModel:
    """Blackbody photosphere scaled to anchor-band fluxes."""

    def __init__(self, anchor_bands: tuple[str, ...] = ANCHOR_BANDS_DEFAULT):
        self.anchor_bands = anchor_bands

    def predict(
        self,
        teff_k: float,
        anchor_flux_jy: dict[str, float],
        anchor_flux_err_jy: dict[str, float],
        predict_bands: tuple[str, ...] = PREDICT_BANDS,
    ) -> PredictedSED:
        """Fit one scale factor to the anchor bands, then predict W1/W2."""
        bands = [b for b in self.anchor_bands if b in anchor_flux_jy]
        if not bands:
            raise ValueError("no anchor bands available for SED fit")

        # Model band flux per unit solid angle: Omega * pi * B_nu(T) (Jy).
        model_unit = {b: np.pi * planck_bnu(teff_k, band_freq_hz(b)) * 1e26 for b in bands}
        obs = np.array([anchor_flux_jy[b] for b in bands])
        err = np.array([max(anchor_flux_err_jy.get(b, 0.0), 1e-6) for b in bands])
        mod = np.array([model_unit[b] for b in bands])

        # Weighted least-squares scale (closed form for a single linear parameter).
        w = 1.0 / err**2
        scale = float(np.sum(w * obs * mod) / np.sum(w * mod**2))
        resid = obs - scale * mod
        anchor_chi2 = float(np.sum((resid / err) ** 2))

        fluxes = {
            b: float(scale * np.pi * planck_bnu(teff_k, band_freq_hz(b)) * 1e26)
            for b in predict_bands
        }
        return PredictedSED(fluxes_jy=fluxes, scale=scale, anchor_chi2=anchor_chi2)


class BergeronModel:
    """Interpolator over a synthetic-magnitude grid keyed by (Teff, logg)."""

    def __init__(self, table: pd.DataFrame):
        self._bands = [c for c in PREDICT_BANDS if f"M_{c}" in table.columns]
        pts = table[["teff", "logg"]].to_numpy()
        self._interp = {
            b: LinearNDInterpolator(pts, table[f"M_{b}"].to_numpy())
            for b in self._bands
        }

    @classmethod
    def from_asset(cls, path: Path | str) -> BergeronModel:
        from astropy.table import Table

        tbl = Table.read(path).to_pandas()
        return cls(tbl)

    def absolute_mag(self, teff_k: float, logg: float, band: str) -> float:
        return float(self._interp[band](teff_k, logg))

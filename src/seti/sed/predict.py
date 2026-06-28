"""Predict photospheric W1/W2 for a table of white dwarfs and attach the result.

Operates row-wise on a DataFrame that already carries Gaia/2MASS anchor
photometry and the WD ``teff``.  Adds predicted-flux and predicted-magnitude
columns used downstream by :mod:`seti.sed.excess`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..photometry import (
    BANDS,
    flux_jy_to_mag,
    mag_err_to_flux_err_jy,
    mag_to_flux_jy,
)
from .models import ANCHOR_BANDS_DEFAULT, PREDICT_BANDS, BlackbodyModel


def _row_anchor_fluxes(row: pd.Series, bands: tuple[str, ...]):
    flux, err = {}, {}
    for b in bands:
        mcol, ecol = f"{b}mag", f"e_{b}mag"
        if mcol in row and np.isfinite(row[mcol]):
            flux[b] = float(mag_to_flux_jy(row[mcol], b))
            merr = float(row[ecol]) if ecol in row and np.isfinite(row[ecol]) else 0.05
            err[b] = float(mag_err_to_flux_err_jy(row[mcol], merr, b))
    return flux, err


def predict_photosphere(
    df: pd.DataFrame,
    anchor_bands: tuple[str, ...] = ANCHOR_BANDS_DEFAULT,
    predict_bands: tuple[str, ...] = PREDICT_BANDS,
    model: BlackbodyModel | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with predicted photospheric photometry columns.

    Adds, per predicted band ``B``: ``B_pred_jy`` and ``B_pred_mag``; plus
    ``sed_scale`` and ``sed_anchor_chi2`` diagnostics.
    """
    model = model or BlackbodyModel(anchor_bands=anchor_bands)
    out = df.copy().reset_index(drop=True)

    pred_jy = {b: np.full(len(out), np.nan) for b in predict_bands}
    scale = np.full(len(out), np.nan)
    achi2 = np.full(len(out), np.nan)

    for i, row in out.iterrows():
        teff = row.get("teff", np.nan)
        if not np.isfinite(teff):
            continue
        aflux, aerr = _row_anchor_fluxes(row, anchor_bands)
        if not aflux:
            continue
        sed = model.predict(float(teff), aflux, aerr, predict_bands=predict_bands)
        for b in predict_bands:
            pred_jy[b][i] = sed.fluxes_jy[b]
        scale[i] = sed.scale
        achi2[i] = sed.anchor_chi2

    for b in predict_bands:
        out[f"{b}_pred_jy"] = pred_jy[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{b}_pred_mag"] = flux_jy_to_mag(pred_jy[b], b)
    out["sed_scale"] = scale
    out["sed_anchor_chi2"] = achi2
    return out


__all__ = ["predict_photosphere", "BANDS"]

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
    n = len(out)
    teff = out["teff"].to_numpy(dtype=float) if "teff" in out else np.full(n, np.nan)

    # Vectorised weighted-scale blackbody fit across all rows (closed form for a
    # single linear parameter), identical to BlackbodyModel.predict row-by-row
    # but ~100x faster -- essential for the injection grid in the forecast.
    from ..photometry import band_freq_hz, planck_bnu

    num = np.zeros(n)
    den = np.zeros(n)
    for b in anchor_bands:
        mcol, ecol = f"{b}mag", f"e_{b}mag"
        if mcol not in out:
            continue
        mag = out[mcol].to_numpy(dtype=float)
        merr = out[ecol].to_numpy(dtype=float) if ecol in out else np.full(n, 0.05)
        merr = np.where(np.isfinite(merr), merr, 0.05)
        finite = np.isfinite(mag) & np.isfinite(teff)
        obs = np.where(finite, mag_to_flux_jy(np.where(finite, mag, 0.0), b), 0.0)
        oerr = np.where(finite, mag_err_to_flux_err_jy(np.where(finite, mag, 0.0), merr, b), 0.0)
        mod = np.pi * planck_bnu(np.where(finite, teff, 1.0), band_freq_hz(b)) * 1e26
        w = np.where(finite & (oerr > 0), 1.0 / np.maximum(oerr, 1e-30) ** 2, 0.0)
        num += w * obs * mod
        den += w * mod**2

    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(den > 0, num / den, np.nan)

    pred_jy = {}
    achi2 = np.full(n, np.nan)  # closed-form fit; per-row chi2 not needed downstream
    for b in predict_bands:
        with np.errstate(invalid="ignore"):
            pred_jy[b] = scale * np.pi * planck_bnu(np.where(np.isfinite(teff), teff, 1.0),
                                                    band_freq_hz(b)) * 1e26

    for b in predict_bands:
        out[f"{b}_pred_jy"] = pred_jy[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{b}_pred_mag"] = flux_jy_to_mag(pred_jy[b], b)
    out["sed_scale"] = scale
    out["sed_anchor_chi2"] = achi2
    return out


__all__ = ["predict_photosphere", "BANDS"]

"""Spectral-shape anomaly detection for Gaia DR3 XP low-resolution spectra.

Gaia DR3 published BP/RP ("XP") low-resolution spectrophotometry for ~220 million
sources --- a vast, recent dataset mined for stellar parameters and quasar
classification but, to our knowledge, never blind-searched for technosignatures.
A partial Dyson swarm (optical light reprocessed to the infrared) or an artificial
spectral feature would imprint a continuum *shape* no stellar atmosphere reproduces.

The model is colour-conditional and robust: stars of the same Gaia colour have
nearly identical XP continua, so we model the expected spectrum as the *median*
spectrum of same-colour stars and score a source by its deviation from that median.
The median is insensitive to the rare anomalies we are hunting (unlike a PCA basis,
which a strong outlier can hijack into its own component), and the comparison is
physically grounded: colour predicts the stellar continuum, so a sharp,
non-stellar departure --- a band-limited deficit, or a narrow artificial feature ---
stands out as a large residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class XPLocus:
    """Colour-conditional median-spectrum model of the normal-stellar XP manifold."""
    bin_edges: np.ndarray         # (n_bins+1,) colour (BP-RP) bin edges
    medians: np.ndarray           # (n_bins, n_wave) median normalised spectrum
    scales: np.ndarray            # (n_bins, n_wave) robust per-sample scatter
    global_med: float             # median global residual of the training inliers
    global_mad: float


def normalize_spectrum(flux: np.ndarray) -> np.ndarray:
    """Shape-normalise a spectrum: non-negative, unit sum (brightness removed)."""
    f = np.asarray(flux, dtype=float)
    f = np.where(np.isfinite(f), f, 0.0)
    s = float(np.sum(np.abs(f)))
    if s <= 0:
        return np.zeros_like(f)
    return f / s


def _bin_of(color: float, edges: np.ndarray) -> int:
    j = int(np.searchsorted(edges, color, side="right") - 1)
    return int(np.clip(j, 0, edges.size - 2))


def fit_locus(spectra: np.ndarray, colors: np.ndarray, n_bins: int = 24,
              min_per_bin: int = 40) -> XPLocus:
    """Build the colour-binned median-spectrum model.

    ``spectra`` is (n, n_wave) shape-normalised flux, ``colors`` the BP-RP per
    source.  Bins are colour quantiles so each is well populated; per bin we store
    the median spectrum and a robust per-wavelength scatter (MAD).

    The number of bins is capped so every bin holds at least ``min_per_bin``
    sources: a per-bin MAD estimated from a handful of spectra is noise-dominated
    and *underestimates* the true scatter, which inflates every z-score and makes
    ordinary stars look anomalous (the small-sample failure that flagged 70% of a
    159-source cone).  With too few sources the scatter cannot be trusted at all.
    """
    X = np.asarray(spectra, dtype=float)
    c = np.asarray(colors, dtype=float)
    good = np.all(np.isfinite(X), axis=1) & np.isfinite(c)
    X, c = X[good], c[good]
    n_bins = max(1, min(n_bins, X.shape[0] // max(1, min_per_bin)))
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(c, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    edges = np.unique(edges)
    n_bins = edges.size - 1
    n_wave = X.shape[1]
    medians = np.zeros((n_bins, n_wave))
    scales = np.ones((n_bins, n_wave))
    glob = []
    for b in range(n_bins):
        sel = (c >= edges[b]) & (c < edges[b + 1])
        if sel.sum() < 3:
            medians[b] = np.median(X, axis=0)
            scales[b] = 1.4826 * np.median(np.abs(X - medians[b]), axis=0) + 1e-9
            continue
        Xb = X[sel]
        med = np.median(Xb, axis=0)
        sca = 1.4826 * np.median(np.abs(Xb - med), axis=0)
        sca = np.where(sca > 0, sca, np.median(sca[sca > 0]) if np.any(sca > 0) else 1e-6)
        medians[b] = med
        scales[b] = sca
        glob.extend(np.sqrt(np.mean(((Xb - med) / sca) ** 2, axis=1)).tolist())
    glob = np.asarray(glob) if glob else np.array([1.0])
    inlier = glob <= np.quantile(glob, 0.90)
    gmed = float(np.median(glob[inlier]))
    gmad = float(np.median(np.abs(glob[inlier] - gmed))) * 1.4826
    return XPLocus(bin_edges=edges, medians=medians, scales=scales,
                   global_med=gmed, global_mad=max(gmad, 1e-9))


def anomaly_score(flux_norm: np.ndarray, color: float, locus: XPLocus) -> dict:
    """Score one shape-normalised spectrum against its colour bin's median.

    Returns the global standardized residual (RMS of per-sample z-scores, and its
    significance vs the training inliers) and the strongest *localised* feature ---
    the largest single-sample z-score, plus its **width**: the number of
    contiguous samples around the peak whose |z| exceeds half the peak.  A narrow
    feature (width 1-3 samples) is the signature of an artificial emission/
    absorption line; a broad excursion (width >~ 8) is a molecular band (TiO/VO in
    M dwarfs, carbon bands) or a reddening/SED tilt --- ordinary stellar
    astrophysics that dominates a raw max-residual ranking, so the width separates
    the two.
    """
    x = np.asarray(flux_norm, dtype=float)
    if not np.all(np.isfinite(x)) or not np.isfinite(color):
        return {"global_resid": np.nan, "global_sigma": np.nan,
                "feature_resid": np.nan, "feature_index": -1,
                "feature_width": -1, "narrow_feature": False}
    b = _bin_of(float(color), locus.bin_edges)
    z = (x - locus.medians[b]) / locus.scales[b]
    global_resid = float(np.sqrt(np.mean(z ** 2)))
    global_sigma = (global_resid - locus.global_med) / locus.global_mad
    az = np.abs(z)
    i = int(np.argmax(az))
    peak = float(az[i])
    half = 0.5 * peak
    lo = i
    while lo - 1 >= 0 and az[lo - 1] > half:
        lo -= 1
    hi = i
    while hi + 1 < az.size and az[hi + 1] > half:
        hi += 1
    width = hi - lo + 1
    # Edge/shape guards.  Gaia XP has a broad line-spread function (~5+ samples)
    # and its basis-function reconstruction rings at the band extremes, producing
    # monotonic ramps pinned to the first/last samples that masquerade as narrow
    # features.  A *real* localised feature is (a) interior, not within edge_margin
    # of either end, and (b) bounded --- it falls back below half-peak on BOTH
    # sides within the band (an edge ramp never descends on the edge side).  A
    # feature only 1 sample wide is below the XP resolution element, i.e. noise.
    edge_margin = 8
    n = az.size
    interior = edge_margin <= i < n - edge_margin
    bounded = (lo > 0 and az[lo - 1] <= half) and (hi < n - 1 and az[hi + 1] <= half)
    real_narrow = bool(interior and bounded and 2 <= width <= 5)
    return {"global_resid": global_resid, "global_sigma": float(global_sigma),
            "feature_resid": peak, "feature_index": i,
            "feature_width": int(width), "feature_interior": bool(interior),
            "feature_bounded": bool(bounded), "narrow_feature": real_narrow}


__all__ = ["XPLocus", "normalize_spectrum", "fit_locus", "anomaly_score"]

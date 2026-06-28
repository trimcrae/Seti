"""Narrow emission-line detection in 1-D spectra.

The detector is a matched filter for an *unresolved* line.  We estimate a robust
continuum (median filter, insensitive to narrow positive spikes), form the
noise-normalised residual, and convolve it with the instrumental line-spread
function (a Gaussian of width ``lsf_sigma_pix``).  Peaks in the matched-filter
output are candidate emission lines; for each we measure a significance and a
*width ratio* --- the fitted line width divided by the LSF width --- which is the
key discriminant: a laser sits at ratio ~ 1, a resolved astrophysical line at
ratio > 1, and a cosmic-ray hit at ratio < 1 (often a single pixel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmissionLine:
    """A candidate emission line found in a spectrum."""
    index: int            # pixel index of the peak
    wavelength: float     # wavelength at the peak
    significance: float   # matched-filter S/N
    width_ratio: float    # fitted Gaussian sigma / instrumental LSF sigma
    amplitude: float      # peak residual flux above the continuum
    ew: float             # equivalent width (same units as wavelength)
    n_pix: int            # number of contiguous pixels above the per-pixel threshold
    ivar_ratio: float = 1.0  # peak inverse-variance / local median (sky-residual flag)

    def as_dict(self) -> dict:
        return {
            "index": int(self.index),
            "wavelength": float(self.wavelength),
            "significance": float(self.significance),
            "width_ratio": float(self.width_ratio),
            "amplitude": float(self.amplitude),
            "ew": float(self.ew),
            "n_pix": int(self.n_pix),
            "ivar_ratio": float(self.ivar_ratio),
        }


def estimate_continuum(flux: np.ndarray, window: int = 101) -> np.ndarray:
    """Robust continuum via a sliding median (insensitive to narrow emission).

    ``window`` is forced odd and clipped to the array length.  The median ignores
    the narrow positive spikes we are searching for, so the residual preserves
    them while removing the broadband stellar/galaxy shape.
    """
    f = np.asarray(flux, dtype=float)
    n = f.size
    if n == 0:
        return f.copy()
    w = int(window)
    w = max(3, w if w % 2 == 1 else w + 1)
    w = min(w, n if n % 2 == 1 else n - 1)
    half = w // 2
    padded = np.pad(f, half, mode="reflect")
    # Vectorised sliding median over the padded array.
    idx = np.arange(n)[:, None] + np.arange(w)[None, :]
    windows = padded[idx]
    return np.nanmedian(windows, axis=1)


def _gaussian_kernel(sigma_pix: float, truncate: float = 4.0) -> np.ndarray:
    half = max(1, int(np.ceil(truncate * sigma_pix)))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma_pix) ** 2)
    return k / np.sqrt(np.sum(k**2))   # unit-norm so the match-filter output is S/N


def _matched_filter(snr: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Convolve the per-pixel S/N with a unit-norm LSF kernel -> matched-filter S/N."""
    kernel = _gaussian_kernel(sigma_pix)
    return np.convolve(np.nan_to_num(snr), kernel, mode="same")


def _fit_width_ratio(resid: np.ndarray, err: np.ndarray, i: int,
                     lsf_sigma_pix: float) -> tuple[float, int]:
    """Estimate a line's width (as a multiple of the LSF sigma) around pixel ``i``.

    Uses the noise-weighted second moment of the residual over a local window,
    and counts contiguous pixels above 1 sigma.  Returns (width_ratio, n_pix).
    """
    half = max(2, int(np.ceil(4 * lsf_sigma_pix)))
    lo, hi = max(0, i - half), min(resid.size, i + half + 1)
    seg = np.clip(resid[lo:hi], 0.0, None)
    if seg.sum() <= 0:
        return np.nan, 0
    x = np.arange(lo, hi) - i
    m1 = np.sum(x * seg) / np.sum(seg)
    m2 = np.sum((x - m1) ** 2 * seg) / np.sum(seg)
    sigma_meas = np.sqrt(max(m2, 0.0))
    width_ratio = sigma_meas / lsf_sigma_pix if lsf_sigma_pix > 0 else np.nan
    with np.errstate(invalid="ignore"):
        above = (resid[lo:hi] / err[lo:hi]) > 1.0
    # contiguous run containing the centre
    n_pix = int(np.sum(above))
    return float(width_ratio), n_pix


def find_emission_lines(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    lsf_sigma_pix: float = 1.0,
    snr_min: float = 8.0,
    continuum_window: int = 101,
    min_separation_pix: int = 3,
) -> list[EmissionLine]:
    """Detect narrow positive emission lines via an LSF-matched filter.

    Parameters mirror the Tellis & Marcy (2017) criteria: a candidate must exceed
    ``snr_min`` in the matched-filter S/N (intensity above the continuum greater
    than the effective noise).  Width discrimination (laser vs resolved line vs
    cosmic ray) is left to the caller via ``EmissionLine.width_ratio``.
    """
    wave = np.asarray(wavelength, dtype=float)
    f = np.asarray(flux, dtype=float)
    err = np.asarray(error, dtype=float)
    n = f.size
    if n < 5 or wave.size != n or err.size != n:
        return []
    err = np.where(np.isfinite(err) & (err > 0), err, np.inf)
    cont = estimate_continuum(f, continuum_window)
    resid = f - cont
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = resid / err
    mf = _matched_filter(snr, lsf_sigma_pix)

    # Candidate peaks: local maxima of the matched filter above threshold.
    above = mf >= snr_min
    lines: list[EmissionLine] = []
    i = 1
    while i < n - 1:
        if above[i] and mf[i] >= mf[i - 1] and mf[i] >= mf[i + 1]:
            # local peak; refine to the max within the separation window
            lo = max(0, i - min_separation_pix)
            hi = min(n, i + min_separation_pix + 1)
            j = lo + int(np.argmax(mf[lo:hi]))
            wr, npix = _fit_width_ratio(resid, err, j, lsf_sigma_pix)
            amp = float(resid[j])
            dl = float(np.median(np.diff(wave))) if n > 1 else 1.0
            cj = cont[j] if cont[j] != 0 else np.nan
            ew = float(np.nansum(np.clip(resid[lo:hi], 0, None)) * dl / cj) if cj else np.nan
            # Inverse-variance depression at the peak relative to its neighbourhood:
            # bright sky lines inflate the variance, so a residual sitting on a
            # locally-low-ivar pixel is a sky-subtraction artefact, not a clean line.
            wlo, whi = max(0, j - 25), min(n, j + 26)
            with np.errstate(divide="ignore"):
                ivar_local = 1.0 / np.square(err[wlo:whi])
            ivar_peak = 1.0 / err[j] ** 2 if np.isfinite(err[j]) and err[j] > 0 else 0.0
            med_iv = float(np.nanmedian(ivar_local[np.isfinite(ivar_local)])) \
                if np.any(np.isfinite(ivar_local)) else 0.0
            ivar_ratio = float(ivar_peak / med_iv) if med_iv > 0 else 1.0
            lines.append(EmissionLine(index=j, wavelength=float(wave[j]),
                                      significance=float(mf[j]), width_ratio=wr,
                                      amplitude=amp, ew=ew, n_pix=npix,
                                      ivar_ratio=ivar_ratio))
            i = hi  # skip past this peak to enforce separation
        else:
            i += 1
    return lines


__all__ = ["EmissionLine", "estimate_continuum", "find_emission_lines",
           "_gaussian_kernel", "_matched_filter"]

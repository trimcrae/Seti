"""Blind search for anomalous narrow **absorption** lines in survey spectra.

All spectral-SETI to date looks for *emission* (laser lines; Tellis & Marcy 2017
and successors).  This is the absorption-mode analogue, which --- as far as the
published literature goes --- nobody has run as a blind technosignature search: a
sharp, *unresolved* absorption feature at a wavelength that matches **no** stellar
photospheric line, no interstellar line, no diffuse interstellar band, no telluric
band and no night-sky residual, and that does **not** recur across sightlines.

Physically it is a distinct technosignature from waste heat or a beacon: a narrow
engineered absorber (a resonant shroud, a partial swarm with a spectral
resonance, an artificial narrowband filter) in front of the star would imprint a
monochromatic absorption line the photosphere cannot make.  Cool stars are line
forests, so the cleanest ground is a nearly featureless continuum (hot stars /
white dwarfs) where any narrow line is conspicuous --- but the funnel is
wavelength-list driven and runs on any spectrum.

The detector is the emission matched filter run on the *inverted* residual
(continuum minus flux); the rejection reuses the emission funnel's width, profile,
sky, telluric and recurrent cuts, with an absorption-specific line list.
"""

from __future__ import annotations

import numpy as np

from .detect import (
    EmissionLine,
    _fit_width_ratio,
    _fwhm_pixels,
    _gaussian_kernel,
    _matched_filter,
    _profile_shape,
    estimate_continuum,
)
from .reject import SKY_LINES, air_to_vacuum, in_telluric, is_near

# Comprehensive stellar photospheric absorption lines: hydrogen Balmer & Paschen,
# He I/II, Ca II H&K + IR triplet, Na D, Mg b, the G band, K I, and the strong
# Fe I / Ca I / Mg I lines that dominate FGK spectra.  Literature (air) values,
# converted to the vacuum scale of the survey spectra at definition time.
REST_ABSORPTION_LINES = air_to_vacuum(np.array([
    # Balmer
    3835.4, 3889.1, 3970.1, 4101.7, 4340.5, 4861.3, 6562.8,
    # Paschen (red)
    8598.4, 8665.0, 8750.5, 8862.8, 9014.9, 9229.0,
    # He I / He II (hot stars)
    4026.2, 4471.5, 4541.6, 4685.7, 5875.6, 6678.2, 4685.7,
    # Ca II H&K and IR triplet
    3933.7, 3968.5, 8498.0, 8542.1, 8662.1,
    # Na D, Mg b triplet, Mg I, G band (CH), K I
    5889.9, 5895.9, 5167.3, 5172.7, 5183.6, 5528.4, 4308.0, 7664.9, 7699.0,
    # Strong Fe I / Ca I / Cr / Mn lines common in FGK photospheres
    4045.8, 4063.6, 4071.7, 4132.1, 4143.9, 4226.7, 4271.8, 4383.5, 4404.8,
    4415.1, 4528.6, 4668.1, 4920.5, 4957.6, 5183.6, 5250.2, 5269.5, 5328.0,
    5371.5, 5397.1, 5405.8, 5434.5, 5446.9, 5455.6, 6122.2, 6162.2, 6439.1,
    6494.9,
    # TiO / molecular band heads (M dwarfs) --- approximate heads
    7053.0, 7589.0, 7666.0, 8432.0,
]))

# Interstellar medium lines and the strongest diffuse interstellar bands (DIBs),
# literature air values converted to vacuum.  These sit at (near) zero velocity
# regardless of the star, so they must be rejected in the *observed* frame, not
# the stellar frame.
ISM_LINES = air_to_vacuum(
    np.array([5889.9, 5895.9, 3933.7, 3968.5, 7664.9, 7699.0, 4300.3]))
DIB_LINES = air_to_vacuum(np.array([
    4428.0, 5780.5, 5797.1, 6196.0, 6203.0, 6269.8, 6283.8, 6379.3, 6613.6,
    6660.7, 6993.2, 5705.1, 5849.8, 4726.8, 4963.9,
]))


def find_absorption_lines(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    lsf_sigma_pix: float = 1.0,
    snr_min: float = 8.0,
    continuum_window: int = 101,
    min_separation_pix: int = 3,
    max_depth_frac: float = 0.95,
) -> list[EmissionLine]:
    """Detect narrow *absorption* lines via an LSF-matched filter on -residual.

    Mirrors :func:`seti.spectra.detect.find_emission_lines` but searches for flux
    deficits (continuum minus flux).  ``amplitude`` is the absorption depth; a line
    deeper than ``max_depth_frac`` of the continuum (i.e. flux ~ 0) is a bad-pixel
    / masked region, not a resonant line, and is skipped.
    """
    wave = np.asarray(wavelength, dtype=float)
    f = np.asarray(flux, dtype=float)
    err = np.asarray(error, dtype=float)
    n = f.size
    if n < 5 or wave.size != n or err.size != n:
        return []
    err = np.where(np.isfinite(err) & (err > 0), err, np.inf)
    cont = estimate_continuum(f, continuum_window)
    resid = cont - f                                # absorption -> positive
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = resid / err
    mf = _matched_filter(snr, lsf_sigma_pix)

    above = mf >= snr_min
    lines: list[EmissionLine] = []
    i = 1
    while i < n - 1:
        if above[i] and mf[i] >= mf[i - 1] and mf[i] >= mf[i + 1]:
            lo = max(0, i - min_separation_pix)
            hi = min(n, i + min_separation_pix + 1)
            j = lo + int(np.argmax(mf[lo:hi]))
            # Reject saturated / zero-flux troughs (bad pixels, not real lines).
            cj = cont[j]
            if not np.isfinite(cj) or cj <= 0 or (cont[j] - f[j]) >= max_depth_frac * cj:
                i = hi
                continue
            wr, npix = _fit_width_ratio(resid, err, j, lsf_sigma_pix)
            amp = float(resid[j])
            dl = float(np.median(np.diff(wave))) if n > 1 else 1.0
            ew = float(np.nansum(np.clip(resid[lo:hi], 0, None)) * dl / cj)
            wlo, whi = max(0, j - 150), min(n, j + 151)
            with np.errstate(divide="ignore"):
                ivar_local = 1.0 / np.square(err[wlo:whi])
            ivar_peak = 1.0 / err[j] ** 2 if np.isfinite(err[j]) and err[j] > 0 else 0.0
            finite_iv = ivar_local[np.isfinite(ivar_local)]
            med_iv = float(np.nanmedian(finite_iv)) if finite_iv.size else 0.0
            ivar_ratio = float(ivar_peak / med_iv) if med_iv > 0 else 1.0
            fwhm = _fwhm_pixels(resid, j)
            min_adj, asym = _profile_shape(resid, j)
            lines.append(EmissionLine(index=j, wavelength=float(wave[j]),
                                      significance=float(mf[j]), width_ratio=wr,
                                      amplitude=amp, ew=ew, n_pix=npix,
                                      ivar_ratio=ivar_ratio, fwhm_pix=fwhm,
                                      min_adjacent=min_adj, asymmetry=asym))
            i = hi
        else:
            i += 1
    return lines


def classify_absorption_line(
    line: EmissionLine,
    redshift: float = 0.0,
    sky_tol: float = 2.0,
    stellar_tol: float = 3.0,
    ism_tol: float = 2.0,
    dib_tol: float = 4.0,
    width_lo: float = 0.6,
    width_hi: float = 1.8,
    fwhm_min: float = 2.0,
    min_adjacent_floor: float = -0.30,
    asymmetry_max: float = 4.0,
) -> str | None:
    """Reject an absorption line with any natural origin; ``None`` if it survives.

    Order: unresolved-spike / resolved / single-pixel / profile quality (shared
    with the emission funnel), then night-sky and telluric (observed frame), then
    stellar photospheric lines (shifted to the *observed* frame by the stellar
    redshift/RV) and interstellar / DIB lines (observed frame, ~zero velocity).
    """
    wr = line.width_ratio
    if np.isfinite(wr) and wr < width_lo:
        return "cosmic_ray"
    if np.isfinite(wr) and wr > width_hi:
        return "resolved_line"
    if getattr(line, "n_pix", 2) < 2:
        return "single_pixel"
    fwhm = getattr(line, "fwhm_pix", fwhm_min)
    if np.isfinite(fwhm) and fwhm < fwhm_min:
        return "unresolved_spike"
    min_adj = getattr(line, "min_adjacent", 0.0)
    if np.isfinite(min_adj) and min_adj < min_adjacent_floor:
        return "emission_flanked"           # a trough beside an emission spike
    asym = getattr(line, "asymmetry", 1.0)
    if np.isfinite(asym) and asym > asymmetry_max:
        return "asymmetric_profile"
    lam = line.wavelength
    if is_near(lam, SKY_LINES, sky_tol):
        return "sky_line"
    if in_telluric(lam):
        return "telluric"
    # Stellar photosphere: shift the rest-frame absorption list to the observed
    # frame by the star's redshift/RV.
    if is_near(lam, REST_ABSORPTION_LINES * (1.0 + redshift), stellar_tol):
        return "stellar_line"
    if is_near(lam, ISM_LINES, ism_tol):
        return "ism_line"
    if is_near(lam, DIB_LINES, dib_tol):
        return "diffuse_band"
    return None


def reject_absorption(lines, redshift: float = 0.0, **kwargs):
    """Apply the absorption funnel; return (survivors, reason histogram)."""
    survivors, counts = [], {}
    for ln in lines:
        reason = classify_absorption_line(ln, redshift=redshift, **kwargs)
        if reason is None:
            survivors.append(ln)
        else:
            counts[reason] = counts.get(reason, 0) + 1
    return survivors, counts


__all__ = ["find_absorption_lines", "classify_absorption_line", "reject_absorption",
           "REST_ABSORPTION_LINES", "ISM_LINES", "DIB_LINES",
           "_gaussian_kernel"]

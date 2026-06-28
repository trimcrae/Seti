r"""Injection--recovery completeness for the laser-line search.

We inject synthetic, unresolved laser lines (Gaussians at the instrumental LSF
width) into real survey spectra at a controlled peak signal-to-noise and a random
clean wavelength, run the full detection-plus-rejection funnel, and record whether
the injected line is recovered as a surviving candidate.  The recovered fraction
as a function of injected S/N is the completeness C(S/N) that converts the raw
occurrence-rate upper limit into an intrinsic one, exactly as in the white-dwarf
injection--recovery and following Tellis & Marcy (2017).
"""

from __future__ import annotations

import numpy as np

from .vet import _lsf_sigma_pix, process_spectrum


def inject_laser_line(wave: np.ndarray, flux: np.ndarray, ivar: np.ndarray,
                      lam_c: float, snr: float, lsf_sigma_pix: float) -> np.ndarray:
    """Return ``flux`` with an unresolved line of peak ``snr`` added at ``lam_c``."""
    flux = np.asarray(flux, dtype=float).copy()
    i = int(np.argmin(np.abs(wave - lam_c)))
    err_i = 1.0 / np.sqrt(ivar[i]) if ivar[i] > 0 else np.nan
    if not np.isfinite(err_i):
        return flux
    amp = snr * err_i
    dlam = float(np.median(np.abs(np.diff(wave))))
    sig = lsf_sigma_pix * dlam
    flux += amp * np.exp(-0.5 * ((wave - lam_c) / sig) ** 2)
    return flux


def injection_recovery(spectra: list[dict], snr_grid=(8, 10, 15, 20, 30, 50),
                       n_per_spectrum: int = 1, rng_seed: int = 0,
                       snr_min: float = 8.0) -> dict:
    """Completeness vs injected peak S/N over a sample of real spectra.

    For each spectrum and each grid S/N, inject one line at a random clean
    (finite-ivar, away from the edges) wavelength and test whether a surviving
    candidate lands within a couple of pixels of it.
    """
    # Vary the draw deterministically by index (no global RNG state).
    recovered = {s: 0 for s in snr_grid}
    injected = {s: 0 for s in snr_grid}
    for k, spec in enumerate(spectra):
        wave = np.asarray(spec["wave"], dtype=float)
        ivar = np.asarray(spec["ivar"], dtype=float)
        lsf = _lsf_sigma_pix(wave, float(spec.get("resolution", 2000.0) or 2000.0))
        good = np.where(ivar > 0)[0]
        if good.size < 50:
            continue
        rng = np.random.default_rng(rng_seed + k)
        for snr in snr_grid:
            for _ in range(n_per_spectrum):
                j = int(good[rng.integers(20, max(21, good.size - 20))])
                lam_c = float(wave[j])
                flux_inj = inject_laser_line(wave, spec["flux"], ivar, lam_c, snr, lsf)
                cands, _ = process_spectrum(
                    spec.get("spec_id", str(k)), wave, flux_inj, ivar,
                    redshift=float(spec.get("redshift", 0.0) or 0.0),
                    resolution=float(spec.get("resolution", 2000.0) or 2000.0),
                    snr_min=snr_min, mask=spec.get("mask"))
                injected[snr] += 1
                dlam = float(np.median(np.abs(np.diff(wave))))
                if any(abs(c.wavelength - lam_c) <= 2.5 * dlam for c in cands):
                    recovered[snr] += 1
    completeness = {int(s): (recovered[s] / injected[s] if injected[s] else 0.0)
                    for s in snr_grid}
    return {"completeness": completeness,
            "n_injected": {int(s): injected[s] for s in snr_grid}}


__all__ = ["inject_laser_line", "injection_recovery"]

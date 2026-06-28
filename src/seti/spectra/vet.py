"""Per-spectrum laser-candidate vetting, scoring and the search orchestration.

Each spectrum is reduced to zero or more *surviving* emission lines (those that
pass the contamination funnel).  A surviving line is scored on independent axes
--- detection significance, agreement of its width with the instrumental LSF, and
its monochromatic *isolation* (a laser is a single line; a spectrum littered with
co-detected emission is far more likely an emission-line galaxy/AGN whose lines
happened to dodge the mask) --- mirroring the multi-axis philosophy of the
white-dwarf search.  ``search_spectra`` runs the whole funnel over a sample and
returns ranked candidates plus the counts needed for an occurrence-rate limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import EmissionLine, find_emission_lines
from .reject import reject_lines


def _lsf_sigma_pix(wave: np.ndarray, resolution: float) -> float:
    """Instrumental LSF sigma in pixels from a resolving power ``R``.

    FWHM_lambda = lambda / R; sigma = FWHM / 2.355; divide by the local dispersion.
    Evaluated at the middle of the grid (adequate for a near-linear dispersion).
    """
    n = wave.size
    if n < 2 or not np.isfinite(resolution) or resolution <= 0:
        return 1.0
    mid = n // 2
    dlam = float(np.median(np.abs(np.diff(wave))))
    if dlam <= 0:
        return 1.0
    fwhm_lam = wave[mid] / resolution
    sigma_lam = fwhm_lam / 2.3548
    return max(0.6, sigma_lam / dlam)


def score_line(line: EmissionLine, n_survivors: int, snr_ref: float = 10.0) -> float:
    """Composite [0, 1] laser-likeness score for a surviving line."""
    # Significance: saturating in S/N above the reference.
    s_sig = 1.0 - np.exp(-max(line.significance, 0.0) / snr_ref)
    # Width agreement with the LSF: peaks at width_ratio == 1.
    wr = line.width_ratio if np.isfinite(line.width_ratio) else 1.0
    s_width = float(np.exp(-((wr - 1.0) ** 2) / (2 * 0.25**2)))
    # Isolation: a single surviving line is ideal; many surviving lines is suspect.
    s_iso = 1.0 / float(max(n_survivors, 1))
    return float(np.clip(0.45 * s_sig + 0.35 * s_width + 0.20 * s_iso, 0.0, 1.0))


@dataclass
class LaserCandidate:
    spec_id: str
    wavelength: float
    significance: float
    width_ratio: float
    score: float
    redshift: float
    n_survivors: int
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "spec_id": self.spec_id,
            "wavelength": float(self.wavelength),
            "significance": float(self.significance),
            "width_ratio": float(self.width_ratio),
            "score": float(self.score),
            "redshift": float(self.redshift),
            "n_survivors": int(self.n_survivors),
        }
        d.update(self.meta)
        return d


def process_spectrum(
    spec_id: str,
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    redshift: float = 0.0,
    resolution: float = 2000.0,
    snr_min: float = 8.0,
    meta: dict | None = None,
    mask: np.ndarray | None = None,
) -> tuple[list[LaserCandidate], dict[str, int]]:
    """Detect, reject and score laser candidates in one spectrum.

    ``ivar`` is the inverse variance (the survey-native error representation);
    pixels with ivar <= 0 are treated as infinite error (masked).  ``mask`` is the
    survey per-pixel data-quality mask (nonzero = bad/sky-affected); those pixels
    are blanked so the dense red airglow forest cannot generate false lines.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)
    if mask is not None:
        m = np.asarray(mask, dtype=float)
        if m.size == err.size:
            err = np.where(m != 0, np.inf, err)
    lsf = _lsf_sigma_pix(wave, resolution)
    lines = find_emission_lines(wave, flux, err, lsf_sigma_pix=lsf, snr_min=snr_min)
    survivors, counts = reject_lines(lines, redshift=redshift)
    cands = [
        LaserCandidate(spec_id=spec_id, wavelength=ln.wavelength,
                       significance=ln.significance, width_ratio=ln.width_ratio,
                       score=score_line(ln, len(survivors)), redshift=redshift,
                       n_survivors=len(survivors), meta=dict(meta or {}))
        for ln in survivors
    ]
    return cands, counts


def search_spectra(spectra: list[dict], snr_min: float = 8.0) -> dict:
    """Run the laser-line funnel over a list of spectrum dicts.

    Each dict needs ``spec_id, wave, flux, ivar`` and optionally ``redshift,
    resolution, meta``.  Returns ranked candidates, aggregate rejection counts and
    the searched-sample size for an occurrence-rate limit.
    """
    all_cands: list[LaserCandidate] = []
    total_counts: dict[str, int] = {}
    n_searched = 0
    for s in spectra:
        if "wave" not in s or "flux" not in s or "ivar" not in s:
            continue
        n_searched += 1
        cands, counts = process_spectrum(
            s.get("spec_id", str(n_searched)), s["wave"], s["flux"], s["ivar"],
            redshift=float(s.get("redshift", 0.0) or 0.0),
            resolution=float(s.get("resolution", 2000.0) or 2000.0),
            snr_min=snr_min, meta=s.get("meta"), mask=s.get("mask"))
        all_cands.extend(cands)
        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + v
    all_cands.sort(key=lambda c: c.score, reverse=True)
    return {
        "n_searched": n_searched,
        "n_candidates": len(all_cands),
        "candidates": [c.as_dict() for c in all_cands],
        "rejection_counts": total_counts,
    }


__all__ = ["LaserCandidate", "process_spectrum", "search_spectra", "score_line"]

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

from .absorb import find_absorption_lines, reject_absorption
from .detect import EmissionLine, find_emission_lines
from .reject import reject_lines

# Absorption mode: max detected lines before a spectrum is deemed a cool
# line-forest star and skipped (a hot/clean continuum has only a handful).
_MAX_ABSORPTION_LINES = 25


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
    mode: str = "emission",
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
    # Bad pixels: the survey mask, zero/non-finite flux, and zero ivar.  Their
    # *edges* generate sharp false residuals (single-pixel spikes next to a chip
    # gap or a zeroed cosmic ray), so dilate the bad set by +/-2 px and blank it.
    bad = ~np.isfinite(flux) | (flux == 0.0) | ~(ivar > 0)
    if mask is not None:
        m = np.asarray(mask, dtype=float)
        if m.size == bad.size:
            bad |= (m != 0)
    if bad.any():
        d = bad.copy()
        for s in (1, 2):
            d[s:] |= bad[:-s]
            d[:-s] |= bad[s:]
        err = np.where(d, np.inf, err)
    lsf = _lsf_sigma_pix(wave, resolution)
    if mode == "absorption":
        lines = find_absorption_lines(wave, flux, err, lsf_sigma_pix=lsf,
                                      snr_min=snr_min)
        # A conspicuous anomalous absorber is only detectable on a *clean*
        # continuum (hot star / white dwarf: a few broad lines).  A cool star is a
        # line forest of hundreds of real metal lines -- intractable and a hopeless
        # host -- so skip any spectrum with too many detected lines entirely.
        if len(lines) > _MAX_ABSORPTION_LINES:
            return [], {"line_forest_skipped": 1}
        survivors, counts = reject_absorption(lines, redshift=redshift)
    else:
        lines = find_emission_lines(wave, flux, err, lsf_sigma_pix=lsf,
                                    snr_min=snr_min)
        survivors, counts = reject_lines(lines, redshift=redshift)
    cands = []
    for ln in survivors:
        # Capture a compact window of the spectrum around the line so the actual
        # candidate can be inspected/plotted later (the bulk arrays are discarded).
        lo, hi = max(0, ln.index - 40), min(wave.size, ln.index + 41)
        m = dict(meta or {})
        m["win_wave"] = [round(float(w), 2) for w in wave[lo:hi]]
        m["win_flux"] = [round(float(f), 4) for f in flux[lo:hi]]
        m["n_lines_in_spectrum"] = len(lines)  # crowding: a laser star is otherwise quiet
        cands.append(LaserCandidate(
            spec_id=spec_id, wavelength=ln.wavelength, significance=ln.significance,
            width_ratio=ln.width_ratio, score=score_line(ln, len(survivors)),
            redshift=redshift, n_survivors=len(survivors), meta=m))
    return cands, counts


def reject_recurrent(candidates: list[dict], bin_width: float = 2.0,
                     min_spectra: int = 3) -> tuple[list[dict], int]:
    """Reject candidate lines whose wavelength recurs across many spectra.

    A monochromatic laser appears in a single spectrum; a sky-airglow or
    instrumental residual appears at the *same observed wavelength* in many
    unrelated sightlines.  Binning candidate wavelengths and dropping any bin
    populated by ``>= min_spectra`` distinct spectra removes the OH-airglow forest
    and fixed-pattern artefacts without any hand-built line atlas --- the empirical
    analogue of requiring a line to be absent from the sky.
    """
    from collections import defaultdict
    by_bin: dict[int, set] = defaultdict(set)
    for c in candidates:
        by_bin[round(float(c["wavelength"]) / bin_width)].add(c.get("spec_id"))
    recurrent = {b for b, s in by_bin.items() if len(s) >= min_spectra}
    survivors, n_rej = [], 0
    for c in candidates:
        if round(float(c["wavelength"]) / bin_width) in recurrent:
            n_rej += 1
        else:
            survivors.append(c)
    return survivors, n_rej


def search_spectra(spectra: list[dict], snr_min: float = 8.0,
                   mode: str = "emission") -> dict:
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
            snr_min=snr_min, meta=s.get("meta"), mask=s.get("mask"), mode=mode)
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

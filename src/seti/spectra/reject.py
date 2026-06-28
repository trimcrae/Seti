"""Contamination funnel for candidate emission lines.

A narrow emission line survives as a laser candidate only if it is inconsistent
with every known natural origin.  We reject, in order:

* **cosmic rays** --- sharper than the instrumental LSF (``width_ratio`` well below
  unity), the dominant non-astrophysical false positive;
* **resolved astrophysical lines** --- broader than the LSF (``width_ratio`` well
  above unity);
* **night-sky and telluric features** --- at fixed *observed* wavelengths from
  airglow / atmospheric bands, where sky-subtraction leaves residuals;
* **known astrophysical emission lines** --- Balmer, forbidden nebular and strong
  stellar lines, evaluated in the *observed* frame by shifting the rest-frame line
  list by the source redshift.

Everything is wavelength-list driven and config-tolerant so it is reproducible and
re-tunable, mirroring the white-dwarf contamination funnel.
"""

from __future__ import annotations

import numpy as np

from .detect import EmissionLine

# Strong night-sky emission lines (observed/topocentric frame, air, Angstrom):
# [OI], NaD airglow, and representative bright OH airglow doublets across the
# optical.  A real survey pipeline carries the full OH atlas; this captures the
# brightest residual-prone features for the offline funnel.
SKY_LINES = np.array([
    5577.34, 5867.5, 5889.95, 5895.92, 6300.30, 6363.78, 6498.7, 6533.0,
    6863.9, 6923.2, 6948.9, 7276.4, 7340.9, 7358.7, 7392.2, 7822.0, 7841.3,
    7913.7, 7964.0, 7993.3, 8344.6, 8399.2, 8430.2, 8465.4, 8505.0, 8791.2,
    8827.1, 8885.9, 8919.6,
])

# Telluric absorption/emission band edges (observed frame, Angstrom): O2 B-band,
# O2 A-band, and the strong H2O bands.  Lines inside these are atmospheric.
TELLURIC_BANDS = [(6860.0, 6920.0), (7590.0, 7700.0), (7160.0, 7340.0),
                  (8100.0, 8400.0), (8900.0, 9800.0)]

# Common astrophysical emission lines, rest-frame vacuum wavelengths (Angstrom):
# Balmer series, forbidden nebular lines, He, and strong stellar features that can
# appear in emission (Ca II H&K, Na D, Mg b region).
REST_EMISSION_LINES = np.array([
    3727.09, 3729.88,          # [O II] doublet
    3868.76, 3967.47,          # [Ne III]
    3934.78, 3969.59,          # Ca II H & K
    4102.89, 4341.68, 4862.68, # H-delta, H-gamma, H-beta
    4960.30, 5008.24,          # [O III]
    5176.7,                    # Mg b
    5890.0, 5896.0,            # Na D
    6549.86, 6564.61, 6585.27, # [N II], H-alpha, [N II]
    6718.29, 6732.67,          # [S II]
    5877.25, 4472.7, 6679.99,  # He I
])


def is_near(obs_wave: float, line_list: np.ndarray, tol: float) -> bool:
    return bool(np.any(np.abs(line_list - obs_wave) <= tol))


def in_telluric(obs_wave: float) -> bool:
    return any(lo <= obs_wave <= hi for lo, hi in TELLURIC_BANDS)


def is_astrophysical(obs_wave: float, redshift: float, tol: float) -> bool:
    """Whether the observed wavelength matches any rest emission line at ``z``."""
    shifted = REST_EMISSION_LINES * (1.0 + redshift)
    return is_near(obs_wave, shifted, tol)


def classify_line(
    line: EmissionLine,
    redshift: float = 0.0,
    sky_tol: float = 1.5,
    astro_tol: float = 2.0,
    width_lo: float = 0.6,
    width_hi: float = 1.6,
) -> str | None:
    """Return a rejection reason, or ``None`` if the line survives as a candidate.

    Width gates use the LSF-relative ``width_ratio``: below ``width_lo`` is a
    cosmic ray (sub-LSF), above ``width_hi`` is a resolved astrophysical line.
    """
    wr = line.width_ratio
    if np.isfinite(wr) and wr < width_lo:
        return "cosmic_ray"
    if np.isfinite(wr) and wr > width_hi:
        return "resolved_line"
    if is_near(line.wavelength, SKY_LINES, sky_tol):
        return "sky_line"
    if in_telluric(line.wavelength):
        return "telluric"
    if is_astrophysical(line.wavelength, redshift, astro_tol):
        return "astrophysical_line"
    return None


def reject_lines(
    lines: list[EmissionLine],
    redshift: float = 0.0,
    **kwargs,
) -> tuple[list[EmissionLine], dict[str, int]]:
    """Apply the funnel; return surviving candidates and a reason histogram."""
    survivors: list[EmissionLine] = []
    counts: dict[str, int] = {}
    for ln in lines:
        reason = classify_line(ln, redshift=redshift, **kwargs)
        if reason is None:
            survivors.append(ln)
        else:
            counts[reason] = counts.get(reason, 0) + 1
    return survivors, counts


__all__ = ["classify_line", "reject_lines", "is_astrophysical", "in_telluric",
           "is_near", "SKY_LINES", "TELLURIC_BANDS", "REST_EMISSION_LINES"]

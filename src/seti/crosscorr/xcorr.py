"""High-resolution cross-correlation biosignature engine (pure logic).

The transmission-spectroscopy cross-correlation method extracts a planet's
atmospheric absorption from ground-based high-resolution spectra even when no
single line is detectable.  During transit the planet's atmosphere imprints a
*forest* of molecular lines on the stellar spectrum, all Doppler-shifted together
by the planet's orbital radial velocity ``v_p = Vsys + Kp sin(2*pi*phase)``.  A
template of the molecule's line positions is cross-correlated with each in-transit
residual spectrum; because the planet moves several km/s across the transit while
the star and the (much stronger) telluric lines stay put, summing the individual
cross-correlation functions (CCFs) along the planet's velocity track coherently
stacks the planet lines and averages the stationary contaminants down.  Scanning
over trial ``(Kp, Vsys)`` builds the standard Kp-Vsys detection map, whose peak --
if the molecule is present -- lands at the planet's true ``(Kp, Vsys)``.

The Doppler separation from telluric O2 is exactly what makes an O2 A-band
(0.76 um) search possible from the ground: the planetary O2 lines slide off the
stationary telluric O2 lines by ``v_p``, so a template that is blind to the
telluric rest frame recovers only the planet.

Everything here is pure numpy: line-list templates (with an explicit, correct
air->vacuum conversion, since literature line lists are on the air scale while
archive echelle spectra are on the vacuum scale), the per-exposure CCF, the
planet velocity track, and the shift-and-add Kp-Vsys map with its detection
significance.  Acquisition and detrending live in :mod:`seti.crosscorr.run`; this
module is unit-tested offline by injecting a synthetic planet signal.
"""

from __future__ import annotations

import numpy as np

# Speed of light (km/s); all radial velocities in this module are km/s.
C_KMS = 299792.458

# --- O2 A-band rotational structure (b1Sigma+_g v'=0 <- X3Sigma-_g v''=0) ----
# The A-band line comb is generated from the rigid-rotor term values rather than
# hard-coded, so the template is a physically-motivated set of true line
# positions.  Constants (cm^-1): band origin and the lower/upper rotational
# constants B'' (X state) and B' (b state) from the HITRAN O2 A-band parameters.
_O2A_NU0_CM = 13122.0        # band origin, cm^-1  (~762 nm)
_O2A_BX_CM = 1.437676        # B'' lower (X 3Sigma) rotational constant, cm^-1
_O2A_BB_CM = 1.391226        # B'  upper (b 1Sigma) rotational constant, cm^-1
_HC_OVER_K_CM_K = 1.438776   # second radiation constant hc/k, cm*K

# Representative strong optical/near-IR H2O line centres, quoted on the **air**
# wavelength scale (Angstrom) as most line atlases are.  This is a deliberately
# small, illustrative comb across the 0.65-0.95 um H2O bands; a production search
# would load the full HITRAN H2O list.  It is converted to vacuum at build time
# so it can be compared against archive spectra directly.
H2O_LINES_AIR = np.array([
    6543.9, 6552.6, 6561.1, 6572.0, 6574.8, 7168.0, 7174.2, 7185.6, 7196.9,
    7205.8, 7226.2, 7234.0, 8227.0, 8235.5, 8244.9, 8267.9, 9328.0, 9337.5,
], dtype=float)


def air_to_vacuum(wave_air):
    """Convert air wavelengths (Angstrom) to vacuum, Morton (1991)/IAU dispersion.

    Ground-based echelle spectra (ESPRESSO/HARPS/NIRPS/IGRINS pipelines) are on
    the **vacuum** wavelength scale, but published molecular line atlases are
    almost always tabulated in **air**.  The offset is ~1.8-2.3 A across the
    optical -- larger than a high-resolution CCF pixel -- so mixing the two shifts
    every template line and destroys the correlation.  We therefore convert every
    air line list to vacuum explicitly at definition time.
    """
    w = np.asarray(wave_air, dtype=float)
    s2 = (1e4 / w) ** 2
    n = (1.0 + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s2)
         + 0.0001599740894897 / (38.92568793293 - s2))
    return w * n


def vacuum_to_air(wave_vac):
    """Convert vacuum wavelengths (Angstrom) to air (inverse of :func:`air_to_vacuum`).

    Provided for completeness; the dispersion relation is written for air input,
    so we invert it once by Newton iteration (two passes converge to < 1e-6 A).
    """
    w = np.asarray(wave_vac, dtype=float)
    air = w.copy()
    for _ in range(3):
        air = w / (air_to_vacuum(air) / air)
    return air


def _o2_a_band_air():
    """Generate the O2 A-band line comb (air Angstrom) and relative depths.

    Rigid-rotor combination differences give the R- and P-branch line wavenumbers
    ``nu = nu0 + (B'+B'')m + (B'-B'')m^2`` with ``m = J+1`` (R) and ``m = -J`` (P).
    Relative depths follow the ground-state Boltzmann population
    ``(2J+1) exp(-B'' J(J+1) hc/kT)`` at a representative planetary ``T ~ 250 K``.
    Wavenumbers are inverse-vacuum by construction, so we return the *air*
    equivalent (via :func:`vacuum_to_air`) to keep every template on one footing
    and route it through the same explicit air->vacuum step as the H2O list.
    """
    temp_k = 250.0
    lam_vac, depth = [], []
    for jj in range(0, 34):
        for m in ((jj + 1), (-jj if jj > 0 else None)):   # R branch, P branch
            if m is None:
                continue
            nu = _O2A_NU0_CM + (_O2A_BB_CM + _O2A_BX_CM) * m \
                + (_O2A_BB_CM - _O2A_BX_CM) * m * m
            if nu <= 0:
                continue
            lam = 1e8 / nu                                 # cm^-1 -> vacuum Angstrom
            if 7585.0 <= lam <= 7700.0:                    # A-band optical window
                pop = (2 * jj + 1) * np.exp(
                    -_O2A_BX_CM * jj * (jj + 1) * _HC_OVER_K_CM_K / temp_k)
                lam_vac.append(lam)
                depth.append(pop)
    lam_vac = np.asarray(lam_vac)
    depth = np.asarray(depth)
    order = np.argsort(lam_vac)
    lam_vac, depth = lam_vac[order], depth[order]
    depth = depth / depth.max() if depth.size else depth
    return vacuum_to_air(lam_vac), depth


def molecular_template(species: str = "O2", wl_min: float | None = None,
                       wl_max: float | None = None):
    """Delta-function molecular line-list template on the **vacuum** scale.

    Returns ``(wavelength_vac_angstrom, depth)`` for ``species`` in ``{"O2","H2O"}``.
    ``O2`` is the 0.76 um A-band comb generated from its rotational constants;
    ``H2O`` is the illustrative optical/near-IR line list above.  Both are built in
    (or converted to) air and then passed through :func:`air_to_vacuum`, so the
    conversion is explicit and identical for every species.  ``wl_min``/``wl_max``
    (Angstrom, vacuum) optionally crop the comb to a spectrograph order.
    """
    key = species.upper()
    if key == "O2":
        air, depth = _o2_a_band_air()
    elif key == "H2O":
        air = H2O_LINES_AIR
        depth = np.ones_like(air)
    else:
        raise ValueError(f"unknown species {species!r}; use 'O2' or 'H2O'")
    wl_vac = air_to_vacuum(air)
    if wl_min is not None:
        mask = wl_vac >= wl_min
        wl_vac, depth = wl_vac[mask], depth[mask]
    if wl_max is not None:
        mask = wl_vac <= wl_max
        wl_vac, depth = wl_vac[mask], depth[mask]
    return wl_vac, depth


def o2_a_band_template(wl_min: float | None = None, wl_max: float | None = None):
    """Convenience wrapper: the O2 0.76 um A-band template (vacuum Angstrom)."""
    return molecular_template("O2", wl_min, wl_max)


def h2o_template(wl_min: float | None = None, wl_max: float | None = None):
    """Convenience wrapper: the illustrative H2O template (vacuum Angstrom)."""
    return molecular_template("H2O", wl_min, wl_max)


def planet_rv_track(phase, Kp: float, Vsys: float = 0.0):
    """Planet radial velocity (km/s) versus orbital phase for a circular orbit.

    ``v_p(phase) = Vsys + Kp sin(2*pi*phase)`` with ``phase`` in orbital *cycles*
    measured from mid-transit (inferior conjunction), where the planet's radial
    velocity relative to the system crosses zero.  ``Kp`` is the planet's
    radial-velocity semi-amplitude (the projected orbital speed); ``Vsys`` the
    systemic velocity.  Accepts a scalar or array ``phase``.
    """
    ph = np.asarray(phase, dtype=float)
    return Vsys + Kp * np.sin(2.0 * np.pi * ph)


def doppler_shift(wavelength, rv_kms: float):
    """Shift ``wavelength`` by radial velocity ``rv_kms`` (km/s): ``lam*(1+v/c)``.

    Positive ``rv_kms`` (recession) moves lines redward, the non-relativistic
    Doppler relation adequate at planetary velocities (v/c ~ 1e-4).
    """
    return np.asarray(wavelength, dtype=float) * (1.0 + rv_kms / C_KMS)


def cross_correlation(wavelength, residual_flux, template_wl, template_depth,
                      rv_grid):
    """Cross-correlate one residual spectrum against a Doppler-shifted template.

    For each trial velocity ``v`` in ``rv_grid`` the template line list is shifted
    to ``template_wl*(1+v/c)`` and the residual is sampled there (linear interp,
    zero outside coverage); the CCF is the depth-weighted sum
    ``CCF(v) = sum_i depth_i * (-residual(shifted_i))``.

    Convention: ``residual_flux`` is continuum-normalised with **absorption as
    negative** dips (flux below the continuum), and ``template_depth`` is
    **positive**.  The minus sign makes a planetary absorption line aligning with
    a template line contribute positively, so a real molecule produces a positive
    CCF peak at the planet's velocity.  The per-exposure CCF is median-subtracted
    so exposures with different continua stack without a baseline offset.
    """
    wave = np.asarray(wavelength, dtype=float)
    resid = np.asarray(residual_flux, dtype=float)
    twl = np.asarray(template_wl, dtype=float)
    depth = np.asarray(template_depth, dtype=float)
    rv = np.asarray(rv_grid, dtype=float)
    ccf = np.empty(rv.size, dtype=float)
    if wave.size < 2 or twl.size == 0:
        return np.zeros(rv.size)
    for k, v in enumerate(rv):
        shifted = twl * (1.0 + v / C_KMS)
        sampled = np.interp(shifted, wave, resid, left=0.0, right=0.0)
        ccf[k] = -float(np.sum(depth * sampled))
    ccf -= np.median(ccf)
    return ccf


def kp_vsys_map(ccfs, phases, rv_grid, kp_grid, vsys_grid,
                exclude_kms: float = 15.0):
    """Shift-and-add per-exposure CCFs along trial ``(Kp, Vsys)`` planet tracks.

    ``ccfs`` is ``(n_exposure, n_rv)`` -- one CCF per in-transit exposure on the
    shared ``rv_grid`` -- and ``phases`` the matching orbital phases (cycles from
    mid-transit).  For every ``(Kp, Vsys)`` the expected planet velocity in each
    exposure is ``planet_rv_track(phase, Kp, Vsys)``; each CCF is sampled there and
    the samples are summed.  A molecule present at the true ``(Kp, Vsys)`` stacks
    coherently while stationary telluric/stellar residuals do not.

    Returns a dict with the co-added map, its standardised S/N map, the peak
    ``(Kp, Vsys)`` and the **detection significance** = ``(peak - mean)/std`` of
    the map evaluated *away from the peak* -- columns within ``exclude_kms`` of the
    peak Vsys are withheld from the noise estimate so the peak does not inflate it.
    """
    ccfs = np.asarray(ccfs, dtype=float)
    if ccfs.ndim != 2:
        raise ValueError("ccfs must be a 2-D array (n_exposure, n_rv)")
    phases = np.asarray(phases, dtype=float)
    rv_grid = np.asarray(rv_grid, dtype=float)
    kp_grid = np.asarray(kp_grid, dtype=float)
    vsys_grid = np.asarray(vsys_grid, dtype=float)
    n_exp = ccfs.shape[0]
    if phases.size != n_exp:
        raise ValueError("phases length must match number of CCFs")

    coadd = np.zeros((kp_grid.size, vsys_grid.size), dtype=float)
    for i, kp in enumerate(kp_grid):
        for j, vs in enumerate(vsys_grid):
            vtrack = planet_rv_track(phases, kp, vs)               # (n_exp,)
            total = 0.0
            for e in range(n_exp):
                total += np.interp(vtrack[e], rv_grid, ccfs[e],
                                   left=0.0, right=0.0)
            coadd[i, j] = total

    # Peak of the co-added map.
    pk = np.unravel_index(int(np.argmax(coadd)), coadd.shape)
    kp_peak = float(kp_grid[pk[0]])
    vsys_peak = float(vsys_grid[pk[1]])
    peak_val = float(coadd[pk])

    # Noise from cells away from the peak Vsys column (exclude a +-exclude_kms band).
    away = np.abs(vsys_grid - vsys_peak) > exclude_kms
    if np.count_nonzero(away) < 2:
        away = np.ones(vsys_grid.size, dtype=bool)
        away[pk[1]] = False
    noise_region = coadd[:, away]
    mu = float(np.mean(noise_region))
    sigma = float(np.std(noise_region))
    snr_map = (coadd - mu) / sigma if sigma > 0 else np.zeros_like(coadd)
    significance = (peak_val - mu) / sigma if sigma > 0 else 0.0

    return {
        "coadd": coadd,
        "snr_map": snr_map,
        "kp_grid": kp_grid,
        "vsys_grid": vsys_grid,
        "kp_peak": kp_peak,
        "vsys_peak": vsys_peak,
        "peak_snr": float(snr_map[pk]),
        "significance": float(significance),
        "noise_mean": mu,
        "noise_std": sigma,
    }


__all__ = [
    "C_KMS", "H2O_LINES_AIR",
    "air_to_vacuum", "vacuum_to_air", "molecular_template",
    "o2_a_band_template", "h2o_template",
    "planet_rv_track", "doppler_shift", "cross_correlation", "kp_vsys_map",
]

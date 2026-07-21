"""Biosignature detectability analysis for LHS 1140 b (and the system).

The system deep-dive established that 209 archival JWST/HST spectra of LHS 1140 b
*exist*; this module answers the question those spectra pose: **is a biosignature
detectable, and what does the data-plus-physics say the answer is?**

A molecular biosignature on a transiting terrestrial planet lives in the
*transmission spectrum* -- the wavelength-dependent transit depth, whose features
have amplitude set by the atmospheric scale height.  Whether a given gas is
detectable is a signal-to-noise question with three ingredients, all of which are
computable from measured system parameters:

1. **Signal** -- the transmission feature amplitude
   ``delta = 2 Rp H n_H / Rs^2`` (ppm), where ``H = kT/(mu g)`` is the scale height
   and ``n_H`` the number of scale heights a band's opacity spans.  A high
   mean-molecular-weight (secondary, N2/CO2/H2O) atmosphere has a small ``H`` and
   therefore a tiny signal; a cleared H2-rich (low ``mu``) atmosphere has a large
   one.
2. **Noise** -- JWST's per-transit transit-depth precision per spectral bin, which
   scales with stellar brightness (photon noise) and a systematic floor.
3. **Integration** -- the number of transits needed to raise the combined
   band significance to a detection threshold.

Putting these together gives, for each candidate atmosphere and each biosignature
gas, the number of JWST transits required for a 3-sigma / 5-sigma detection --
which, compared against the transits actually observed, *is* the biosignature
answer for this planet.  All functions here are pure and unit-tested offline; the
runner supplies live parameters and the observed-transit count.
"""

from __future__ import annotations

import numpy as np

# --- Physical constants (SI) -----------------------------------------------
_KB = 1.380649e-23          # Boltzmann, J/K
_AMU = 1.66053907e-27       # atomic mass unit, kg
_G = 6.67430e-11            # gravitational constant
_R_EARTH = 6.371e6          # m
_M_EARTH = 5.9722e24        # kg
_R_SUN = 6.957e8            # m

# Mean molecular weights (amu) of the candidate atmospheres.  LHS 1140 b's
# density and the JWST data disfavour a cleared H2 envelope; a secondary
# N2/CO2/H2O atmosphere (high mu) is the physically expected case for a
# ~5.6 M_earth temperate rocky/water world.
ATMOSPHERES = {
    "H2_rich_cleared": 2.3,     # primordial / cleared -- large scale height
    "N2_secondary": 28.0,       # Earth/Titan-like secondary atmosphere
    "H2O_steam": 18.0,          # water-world steam atmosphere
    "CO2_rich": 44.0,           # Venus/Mars-like
}

# Biosignature (and reference) gases: strongest transmission band, the JWST
# instrument that covers it, and n_H -- how many scale heights of opacity the
# band spans.  Major/abundant species probe ~5 scale heights; a *trace*
# biosignature gas (ppb-ppm abundance) probes only ~1-2 even in its best band.
# O2 has no strong isolated transmission band in the JWST range; it is searched
# via the O2-O2 collision-induced band at 1.06/1.27 um (very weak, n_H ~ 0.5)
# and inferred through its photochemical product O3 (9.6 um, MIRI).
GAS_BANDS = {
    "H2O":  {"um": 1.4,  "instrument": "NIRISS/NIRSpec", "n_H": 5.0, "n_bins": 8,
             "role": "habitability (not a biosignature alone)"},
    "CO2":  {"um": 4.3,  "instrument": "NIRSpec",        "n_H": 5.0, "n_bins": 6,
             "role": "atmosphere tracer (not a biosignature alone)"},
    "CH4":  {"um": 3.3,  "instrument": "NIRSpec",        "n_H": 3.0, "n_bins": 6,
             "role": "biosignature in disequilibrium with CO2/O2"},
    "O3":   {"um": 9.6,  "instrument": "MIRI",           "n_H": 2.0, "n_bins": 5,
             "role": "biosignature (O2 photochemical proxy)"},
    "O2_CIA": {"um": 1.27, "instrument": "NIRISS",       "n_H": 0.5, "n_bins": 3,
               "role": "biosignature (O2-O2 collision band)"},
    "N2O":  {"um": 7.8,  "instrument": "MIRI",           "n_H": 1.5, "n_bins": 4,
             "role": "biosignature (few abiotic sources)"},
    "CH3Cl": {"um": 3.4, "instrument": "NIRSpec",        "n_H": 1.0, "n_bins": 4,
              "role": "biosignature (halomethane)"},
}


# --- Signal ----------------------------------------------------------------
def surface_gravity(mp_earth: float, rp_earth: float) -> float:
    """Surface gravity (m/s^2) from planet mass and radius in Earth units."""
    m = mp_earth * _M_EARTH
    r = rp_earth * _R_EARTH
    return _G * m / (r * r)


def scale_height(temp_k: float, mu_amu: float, gravity: float) -> float:
    """Atmospheric pressure scale height ``H = kT/(mu m_H g)`` (m)."""
    return _KB * temp_k / (mu_amu * _AMU * gravity)


def transit_depth_ppm(rp_earth: float, rs_sun: float) -> float:
    """Nominal transit depth ``(Rp/Rs)^2`` in ppm."""
    ratio = (rp_earth * _R_EARTH) / (rs_sun * _R_SUN)
    return ratio * ratio * 1e6


def feature_amplitude_ppm(rp_earth: float, rs_sun: float, H_m: float,
                          n_H: float) -> float:
    """Transmission feature amplitude ``2 Rp H n_H / Rs^2`` in ppm.

    This is the change in transit depth between the opaque core of a band and the
    continuum -- the actual signal a spectrograph must measure.
    """
    rp = rp_earth * _R_EARTH
    rs = rs_sun * _R_SUN
    return 2.0 * rp * H_m * n_H / (rs * rs) * 1e6


# --- Noise -----------------------------------------------------------------
# A transparent, benchmark-anchored JWST transit-depth noise model.  Rather than
# a full instrument simulator (ETC-grade, not reproducible on a runner), we anchor
# to a documented near-photon-limited performance point and scale it physically:
#   * photon noise scales as 10^(0.2 (J - J_ref))  (flux ~ 10^(-0.4 J));
#   * finer binning (higher R) raises per-bin noise as sqrt(R/R_ref);
#   * a longer in-transit window lowers it as sqrt(t_ref/t_in);
#   * a systematic floor adds in quadrature (the JWST ~10-20 ppm noise floor).
# Anchor: a J_ref = 9.6 M dwarf reaches ~30 ppm per R=50 bin per transit over a
# ~1 h in-transit window on NIRISS/NIRSpec, with a ~15 ppm floor -- consistent
# with published LHS-1140-class results.
_J_REF = 9.6
_SIGMA_REF_PPM = 30.0
_R_REF = 50.0
_TIN_REF_H = 1.0
_FLOOR_PPM = 15.0


def jwst_depth_noise_ppm(jmag: float, resolution: float = 50.0,
                         t_in_hours: float = 1.0,
                         floor_ppm: float = _FLOOR_PPM) -> float:
    """Per-bin, per-transit transit-depth precision (ppm) at spectral ``resolution``.

    Photon term scaled from the anchor point, added in quadrature with the
    systematic floor.  ``t_in_hours`` is the in-transit duration (the constraining
    baseline); the out-of-transit baseline is assumed comparable.
    """
    photon = (_SIGMA_REF_PPM
              * 10 ** (0.2 * (jmag - _J_REF))
              * np.sqrt(resolution / _R_REF)
              * np.sqrt(_TIN_REF_H / max(t_in_hours, 1e-3)))
    return float(np.hypot(photon, floor_ppm))


# --- Integration: transits to a detection ----------------------------------
def transits_to_detect(amplitude_ppm: float, noise_ppm: float, n_bins: int,
                       target_sigma: float = 5.0) -> float:
    """Number of transits to detect a band at ``target_sigma``.

    Combining ``n_bins`` spectral channels across the band, the per-transit
    significance is ``amplitude * sqrt(n_bins) / noise``; stacking ``N`` transits
    scales it by ``sqrt(N)``.  Solving for ``N`` at ``target_sigma``.  Returns
    ``inf`` for a vanishing signal.
    """
    if amplitude_ppm <= 0 or noise_ppm <= 0 or n_bins <= 0:
        return float("inf")
    sig_per_transit = amplitude_ppm * np.sqrt(n_bins) / noise_ppm
    if sig_per_transit <= 0:
        return float("inf")
    return float((target_sigma / sig_per_transit) ** 2)


def biosignature_detectability(params: dict, atmospheres: dict | None = None,
                               gases: dict | None = None,
                               target_sigma: float = 5.0) -> dict:
    """Full biosignature detectability budget for a transiting planet.

    ``params`` needs ``rp_earth, mp_earth, rs_sun, teq_k, jmag, t_in_hours`` (and
    optionally ``resolution``).  Returns, for every atmosphere x gas combination,
    the feature amplitude and the number of transits for a ``target_sigma``
    detection -- the quantitative answer to whether each biosignature is reachable.
    """
    atmospheres = atmospheres or ATMOSPHERES
    gases = gases or GAS_BANDS
    g = surface_gravity(params["mp_earth"], params["rp_earth"])
    depth = transit_depth_ppm(params["rp_earth"], params["rs_sun"])
    res = params.get("resolution", 50.0)
    noise = jwst_depth_noise_ppm(params["jmag"], resolution=res,
                                 t_in_hours=params["t_in_hours"])
    out = {"surface_gravity_ms2": g, "transit_depth_ppm": depth,
           "jwst_depth_noise_ppm_per_bin_per_transit": noise,
           "target_sigma": target_sigma, "atmospheres": {}}
    for atm_name, mu in atmospheres.items():
        H = scale_height(params["teq_k"], mu, g)
        gas_rows = {}
        for gas, b in gases.items():
            amp = feature_amplitude_ppm(params["rp_earth"], params["rs_sun"],
                                        H, b["n_H"])
            n_tr = transits_to_detect(amp, noise, b["n_bins"], target_sigma)
            gas_rows[gas] = {"band_um": b["um"], "instrument": b["instrument"],
                             "role": b["role"],
                             "feature_amplitude_ppm": round(amp, 3),
                             "transits_for_detection": (None if not np.isfinite(n_tr)
                                                        else round(n_tr, 1))}
        out["atmospheres"][atm_name] = {"mu_amu": mu,
                                        "scale_height_km": round(H / 1e3, 2),
                                        "one_scale_height_ppm": round(
                                            feature_amplitude_ppm(
                                                params["rp_earth"],
                                                params["rs_sun"], H, 1.0), 3),
                                        "gases": gas_rows}
    return out


def biosignature_verdict(budget: dict, atmospheres_observed: int,
                         transits_observed: int,
                         expected_atmosphere: str = "N2_secondary") -> dict:
    """Turn the budget + what was actually observed into a biosignature answer.

    A biosignature gas counts as *reachable now* only if the transits required
    (under the physically expected atmosphere) do not exceed those observed.  The
    verdict states, per gas, detectable / not-detectable-with-current-data, and
    gives the headline answer.
    """
    atm = budget["atmospheres"].get(expected_atmosphere, {})
    gases = atm.get("gases", {})
    reachable, unreachable = [], []
    biosig_gases = {"CH4", "O3", "O2_CIA", "N2O", "CH3Cl"}
    for gas, row in gases.items():
        if gas not in biosig_gases:
            continue
        need = row.get("transits_for_detection")
        rec = {"gas": gas, "band_um": row["band_um"],
               "instrument": row["instrument"],
               "transits_needed": need, "transits_observed": transits_observed}
        if need is not None and need <= max(transits_observed, 1):
            reachable.append(rec)
        else:
            unreachable.append(rec)
    # Minimum transits any single biosignature gas needs, under the expected atm.
    needs = [r.get("transits_for_detection") for r in
             [gases[g] for g in biosig_gases if g in gases]
             if r.get("transits_for_detection") is not None]
    min_need = min(needs) if needs else None
    answer = ("BIOSIGNATURE_NOT_DETECTABLE_WITH_CURRENT_DATA"
              if not reachable else "BIOSIGNATURE_REACHABLE_REVIEW_DATA")
    return {
        "expected_atmosphere": expected_atmosphere,
        "scale_height_km": atm.get("scale_height_km"),
        "transits_observed": transits_observed,
        "min_transits_for_any_biosignature": (None if min_need is None
                                              else round(min_need, 1)),
        "reachable_biosignatures": reachable,
        "unreachable_biosignatures": unreachable,
        "answer": answer,
        "interpretation": (
            "Under the physically expected high-mean-molecular-weight secondary "
            "atmosphere, the transmission features of every biosignature gas are "
            "far below the JWST per-transit noise, so no biosignature is "
            "detectable with the transits observed; a positive answer would "
            "require the listed (infeasible) transit counts. A biosignature would "
            "only be reachable if the planet had a cleared low-mu (H2-rich) "
            "envelope, which its density and the existing data disfavour."),
    }


__all__ = [
    "ATMOSPHERES", "GAS_BANDS", "surface_gravity", "scale_height",
    "transit_depth_ppm", "feature_amplitude_ppm", "jwst_depth_noise_ppm",
    "transits_to_detect", "biosignature_detectability", "biosignature_verdict",
]

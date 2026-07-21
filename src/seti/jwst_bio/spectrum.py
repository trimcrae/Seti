"""Pure-logic scorers for the JWST/HST biosignature spectrum analysis of
LHS 1140 b (unit-tested offline).

Everything here is a deterministic NumPy function operating on arrays the runner
extracts from archival JWST products; there is no network access and no fitted
instrument model.  The chain is:

1. :func:`build_transmission_spectrum` -- from a stack of per-integration
   flux(wavelength) spectra plus an in-/out-of-transit mask (derived from the
   *known* ephemeris), form the transit depth per wavelength bin,
   ``depth = 1 - <F_in> / <F_out>``, with propagated errors.  This is a robust,
   detection-level estimator: it needs no full transit light-curve (limb
   darkening / systematics) model, only a clean split of integrations, so it can
   over-estimate the error on a bin dominated by red noise -- honest but
   conservative.
2. :func:`molecular_feature_detect` -- for each molecular band, measure the
   feature amplitude (in-band transit depth above the local continuum) and its
   significance.
3. :func:`disequilibrium_biosignature` -- the actual biosignature logic: a robust
   flag requires a redox-*disequilibrium pair* (CH4+CO2, or O2+CH4, or the
   low-abiotic-source gas N2O), never a single gas.
4. :func:`abiotic_false_positive` -- M-dwarf discipline: O2/O3 can be abiotic
   (H2O photolysis + H escape, or CO2 photolysis giving O2 + CO), so any
   O2/O3-based claim is gated by this test.
5. :func:`eclipse_brightness_temperature` -- the cleanest atmosphere-vs-rock
   discriminant: a MIRI secondary-eclipse depth -> day-side brightness
   temperature -> atmosphere-present (redistributed / cool) vs bare-rock
   (day-side near the no-redistribution maximum).
6. :func:`laser_line_scan` -- reuses the XP-proven narrow-line guards to search
   the (higher-resolution than Gaia XP) JWST spectrum for an unresolved emission
   line (a laser technosignature).

Limitations are stated in each docstring; the headline caveat is that this is a
*detection-level screen*, not an atmospheric retrieval -- a positive result is a
reason to trigger a full retrieval, not a publication on its own.
"""

from __future__ import annotations

import numpy as np

# Reuse the XP-resolution narrow-line guards verbatim for the laser scan.
from ..panspermia.dossier import narrow_feature_scan  # noqa: F401

# --- Physical constants (SI) -----------------------------------------------
_H_PLANCK = 6.62607015e-34      # J s
_C_LIGHT = 2.99792458e8         # m/s
_KB = 1.380649e-23              # J/K

# Molecular bands searched in the JWST transmission spectrum.  Each gas maps to a
# list of ``(center_um, half_width_um)`` windows (the strongest accessible band,
# and for O2 the two collision-induced/near-IR windows).  These centers/widths are
# deliberately generous detection windows, not line lists.
DEFAULT_BANDS: dict[str, list[tuple[float, float]]] = {
    "H2O": [(1.40, 0.15)],      # habitability tracer (not a biosignature alone)
    "CH4": [(3.30, 0.15)],      # biosignature in disequilibrium with CO2/O2
    "CO2": [(4.30, 0.15)],      # atmosphere / redox tracer
    "CO":  [(4.70, 0.12)],      # CO2-photolysis product -> abiotic-O2 tracer
    "O2":  [(0.76, 0.02), (1.27, 0.04)],   # O2 A-band + O2-O2 CIA
    "O3":  [(9.60, 0.40)],      # MIRI; O2 photochemical proxy
    "N2O": [(7.80, 0.30)],      # biosignature with few abiotic sources
}

# Which gases are redox-disequilibrium biosignature *pairs*.  A single gas is
# never a biosignature; N2O is listed as a stand-alone robust biosignature only
# because it has very few abiotic sources (still reported distinctly).
_REDOX_PAIRS = [("CH4", "CO2"), ("O2", "CH4"), ("O3", "CH4")]
_OXYGEN_SPECIES = ("O2", "O3")


# --- 1. Transmission spectrum ----------------------------------------------
def transit_mask_from_ephemeris(times, t0: float, period: float,
                                duration: float,
                                fraction: float = 1.0) -> np.ndarray:
    """Boolean in-transit mask for integration mid-times from a known ephemeris.

    ``times``, ``t0`` and ``period`` share one unit (days); ``duration`` is the
    full (T14) transit duration in the same unit.  A point is in-transit if its
    orbital phase falls within ``+/- fraction * duration / 2`` of mid-transit.
    Using ``fraction < 1`` (e.g. 0.8) excludes ingress/egress for a cleaner depth.
    """
    times = np.asarray(times, float)
    if period <= 0 or duration <= 0:
        return np.zeros(times.shape, bool)
    phase = ((times - t0) / period + 0.5) % 1.0 - 0.5      # in (-0.5, 0.5] orbits
    dt = np.abs(phase) * period                            # time from mid-transit
    return dt <= 0.5 * fraction * duration


def build_transmission_spectrum(wavelength, flux_series, transit_mask,
                                flux_err=None) -> dict:
    """Transit depth per wavelength bin from stacked per-integration spectra.

    ``flux_series`` is ``(n_integrations, n_wavelength)``; ``transit_mask`` is a
    length-``n_integrations`` boolean (True == in transit).  For each wavelength
    bin the depth is ``1 - <F_in>/<F_out>``.  Per-bin errors on the two means come
    from ``flux_err`` if given (propagated as ``sqrt(sum e^2)/n``) and otherwise
    from the sample scatter across integrations (``std/sqrt(n)``); the depth error
    is the standard ratio propagation.  Returns a dict with ``wavelength``,
    ``depth`` and ``depth_err`` (fractional, multiply by 1e6 for ppm), plus
    ``n_in``/``n_out``.

    This estimates the depth from a simple in/out split -- no limb-darkening or
    systematics model -- so it is a robust *detection-level* spectrum; its error
    bars are conservative for bins dominated by correlated (red) noise.
    """
    wavelength = np.asarray(wavelength, float)
    flux = np.asarray(flux_series, float)
    if flux.ndim != 2:
        raise ValueError("flux_series must be 2D (n_integrations, n_wavelength)")
    mask = np.asarray(transit_mask, bool)
    if mask.shape[0] != flux.shape[0]:
        raise ValueError("transit_mask length must match n_integrations")
    n_in, n_out = int(mask.sum()), int((~mask).sum())
    n_wl = flux.shape[1]
    if n_in == 0 or n_out == 0:
        return {"wavelength": wavelength, "depth": np.full(n_wl, np.nan),
                "depth_err": np.full(n_wl, np.nan), "n_in": n_in, "n_out": n_out,
                "note": "need >=1 in-transit and >=1 out-of-transit integration"}

    f_in, f_out = flux[mask], flux[~mask]
    mean_in = np.nanmean(f_in, axis=0)
    mean_out = np.nanmean(f_out, axis=0)

    if flux_err is not None:
        err = np.asarray(flux_err, float)
        e_in = np.sqrt(np.nansum(err[mask] ** 2, axis=0)) / max(n_in, 1)
        e_out = np.sqrt(np.nansum(err[~mask] ** 2, axis=0)) / max(n_out, 1)
    else:
        e_in = (np.nanstd(f_in, axis=0, ddof=1) / np.sqrt(n_in)
                if n_in > 1 else np.zeros(n_wl))
        e_out = (np.nanstd(f_out, axis=0, ddof=1) / np.sqrt(n_out)
                 if n_out > 1 else np.zeros(n_wl))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mean_in / mean_out
        depth = 1.0 - ratio
        rel = np.sqrt((e_in / mean_in) ** 2 + (e_out / mean_out) ** 2)
        depth_err = np.abs(ratio) * rel
    # Floor the error so a (near-)noiseless synthetic input cannot manufacture an
    # infinite significance downstream.
    scale = np.nanmedian(np.abs(depth)) if np.any(np.isfinite(depth)) else 0.0
    floor = 1e-9 + 1e-6 * (scale if np.isfinite(scale) else 0.0)
    depth_err = np.where(np.isfinite(depth_err), depth_err, np.nan)
    depth_err = np.fmax(depth_err, floor)
    return {"wavelength": wavelength, "depth": depth, "depth_err": depth_err,
            "n_in": n_in, "n_out": n_out}


# --- 2. Molecular feature detection ----------------------------------------
def _weighted_mean(values, errs):
    """Inverse-variance weighted mean and its error; falls back to plain mean."""
    values = np.asarray(values, float)
    errs = np.asarray(errs, float)
    good = np.isfinite(values) & np.isfinite(errs) & (errs > 0)
    if not np.any(good):
        v = values[np.isfinite(values)]
        if v.size == 0:
            return np.nan, np.inf
        return float(np.mean(v)), np.inf
    w = 1.0 / errs[good] ** 2
    mean = float(np.sum(w * values[good]) / np.sum(w))
    err = float(np.sqrt(1.0 / np.sum(w)))
    return mean, err


def molecular_feature_detect(wavelength, depth, depth_err,
                             band_centers: dict | None = None) -> dict:
    """Per-band feature amplitude and significance vs a local continuum.

    ``depth``/``depth_err`` are the (fractional) transit depth and error from
    :func:`build_transmission_spectrum`; ``band_centers`` maps a gas name to a
    list of ``(center_um, half_width_um)`` windows (defaults to
    :data:`DEFAULT_BANDS`).  For each gas the in-band inverse-variance-weighted
    depth is compared to a *local continuum* (points within ~3 half-widths of the
    band but outside it), and the amplitude is reported in ppm with its
    significance ``amplitude / err``.  A molecular absorption band raises the
    transit depth, so a real feature has positive amplitude.

    This is a matched-window detector, not a line-by-line fit; blended bands and
    a sloped continuum are only approximately handled, which is why a positive
    result is a *screen*, not a retrieval.
    """
    wavelength = np.asarray(wavelength, float)
    depth = np.asarray(depth, float)
    depth_err = np.asarray(depth_err, float)
    bands = band_centers or DEFAULT_BANDS
    out: dict[str, dict] = {}
    for gas, windows in bands.items():
        in_band = np.zeros(wavelength.shape, bool)
        cont = np.zeros(wavelength.shape, bool)
        for center, hw in windows:
            in_band |= np.abs(wavelength - center) <= hw
            cont |= np.abs(wavelength - center) <= 3.0 * hw
        cont &= ~in_band
        n_bins = int(np.count_nonzero(in_band & np.isfinite(depth)))
        if n_bins == 0:
            out[gas] = {"center_um": windows[0][0], "n_bins": 0,
                        "amplitude_ppm": None, "amplitude_err_ppm": None,
                        "significance": 0.0, "detected": False,
                        "note": "band not covered by the wavelength grid"}
            continue
        in_mean, in_err = _weighted_mean(depth[in_band], depth_err[in_band])
        if np.count_nonzero(cont & np.isfinite(depth)) >= 2:
            base_mean, base_err = _weighted_mean(depth[cont], depth_err[cont])
        else:
            # No local continuum -> use the global median as a flat baseline.
            fin = np.isfinite(depth)
            base_mean = float(np.median(depth[fin])) if np.any(fin) else 0.0
            base_err = 0.0
        amp = in_mean - base_mean
        amp_err = float(np.hypot(in_err, base_err))
        sig = amp / amp_err if np.isfinite(amp_err) and amp_err > 0 else 0.0
        out[gas] = {
            "center_um": windows[0][0], "n_bins": n_bins,
            "amplitude_ppm": round(amp * 1e6, 3),
            "amplitude_err_ppm": (round(amp_err * 1e6, 3)
                                  if np.isfinite(amp_err) else None),
            "significance": round(float(sig), 3),
            "detected": bool(np.isfinite(sig) and sig >= 3.0 and amp > 0),
        }
    return out


# --- 3. Disequilibrium biosignature logic ----------------------------------
def disequilibrium_biosignature(detections: dict, n_sigma: float = 3.0) -> dict:
    """Flag a robust biosignature only if a redox-disequilibrium pair coexists.

    ``detections`` is the output of :func:`molecular_feature_detect`.  A single
    gas is never a biosignature; a robust flag requires either a co-existing redox
    pair -- CH4+CO2, O2+CH4, or O3(as O2 proxy)+CH4, each member at
    ``>= n_sigma`` -- or the low-abiotic-source gas N2O at ``>= n_sigma``.  The
    reported ``joint_significance`` is the quadrature sum of the members'
    significances (the significance of the *coincidence*), while the gating uses
    the *weaker* member, so both gases must be independently significant.

    Detecting a disequilibrium pair is necessary but not sufficient: O2/O3 pairs
    must still survive :func:`abiotic_false_positive`.  That gate is applied by the
    caller and echoed here only as a reminder.
    """
    def sig(gas):
        d = detections.get(gas, {})
        s = d.get("significance", 0.0)
        return float(s) if s is not None and np.isfinite(s) else 0.0

    candidates = []
    for a, b in _REDOX_PAIRS:
        sa, sb = sig(a), sig(b)
        if sa >= n_sigma and sb >= n_sigma:
            candidates.append({
                "pair": f"{a}+{b}", "members": {a: sa, b: sb},
                "limiting_significance": round(min(sa, sb), 3),
                "joint_significance": round(float(np.hypot(sa, sb)), 3),
                "kind": "redox_disequilibrium",
            })
    sn2o = sig("N2O")
    if sn2o >= n_sigma:
        candidates.append({
            "pair": "N2O", "members": {"N2O": sn2o},
            "limiting_significance": round(sn2o, 3),
            "joint_significance": round(sn2o, 3),
            "kind": "low_abiotic_source_gas",
        })

    candidates.sort(key=lambda c: -c["limiting_significance"])
    best = candidates[0] if candidates else None
    return {
        "is_biosignature": bool(candidates),
        "n_sigma_threshold": n_sigma,
        "best_pair": (best["pair"] if best else None),
        "best": best,
        "all_candidates": candidates,
        "oxygen_involved": bool(best and any(g in best["members"]
                                             for g in _OXYGEN_SPECIES)),
        "note": ("robust disequilibrium biosignature pair detected -- an O2/O3 "
                 "pair must still pass abiotic_false_positive"
                 if candidates else
                 "no redox-disequilibrium pair; a single gas is not a "
                 "biosignature"),
    }


# --- 4. Abiotic false-positive gate (M-dwarf discipline) -------------------
def abiotic_false_positive(detections: dict, n_sigma: float = 3.0) -> dict:
    """Gate O2/O3 claims against known M-dwarf abiotic oxygen mechanisms.

    Around an M dwarf, O2/O3 can accumulate abiotically: (i) H2O photolysis
    followed by hydrogen escape leaves O2 without any biology; (ii) CO2 photolysis
    produces O2 together with its tracer CO; (iii) dense O2 shows up via the
    O2-O2 (O4) collision-induced band rather than a biological source.  This
    function flags a *likely-abiotic* oxygen detection when O2 or O3 is present
    (``>= n_sigma``) and EITHER there is no co-existing CH4 disequilibrium, OR CO
    is co-detected, OR the O2 evidence is the CIA/O4 window.  It must gate any
    O2-based biosignature claim.

    A clean ``abiotic_flag=False`` requires oxygen accompanied by CH4 (genuine
    redox disequilibrium) and no CO photolysis tracer -- the only regime in which
    O2/O3 is hard to explain abiotically.
    """
    def det(gas):
        d = detections.get(gas, {})
        s = d.get("significance", 0.0)
        s = float(s) if s is not None and np.isfinite(s) else 0.0
        return s >= n_sigma

    o2, o3, ch4, co = det("O2"), det("O3"), det("CH4"), det("CO")
    oxygen = o2 or o3
    # Hard triggers -- an oxygen detection these explain abiotically.
    hard: list[str] = []
    if oxygen and not ch4:
        hard.append("O2/O3 present without CH4 disequilibrium -- consistent with "
                    "H2O photolysis + H escape (abiotic O2)")
    if oxygen and co:
        hard.append("CO co-detected with O2/O3 -- CO2-photolysis abiotic-O2 tracer")
    # Advisory caveat -- the near-IR O2 evidence in DEFAULT_BANDS includes the
    # O2-O2 (O4) CIA window, a pressure-induced pattern to verify.  This does NOT
    # by itself brand a detection abiotic (that would reject the genuine O2+CH4
    # disequilibrium case), but it must always be checked.
    caveats = (["O2 evidence includes the O2-O2 (O4) CIA window -- verify it is "
                "not a pressure-induced abiotic pattern"] if o2 else [])
    reasons = hard + caveats
    return {
        "oxygen_present": bool(oxygen),
        "o2": o2, "o3": o3, "ch4": ch4, "co": co,
        "abiotic_flag": bool(hard),
        "reasons": reasons,
        "gates_oxygen_claim": True,
        "note": ("oxygen detection is consistent with abiotic M-dwarf pathways -- "
                 "do NOT claim a biosignature on O2/O3 alone"
                 if hard else
                 "no unresolved abiotic-oxygen concern (either no oxygen, or "
                 "oxygen with CH4 disequilibrium and no CO tracer)"),
    }


# --- 5. Eclipse brightness-temperature discriminant ------------------------
def eclipse_brightness_temperature(eclipse_depth_ppm: float, rp_rs: float,
                                   teff_star: float, a_rs: float,
                                   bond_albedo: float = 0.0) -> dict:
    """Day-side brightness temperature from a secondary-eclipse depth, classified.

    In the Rayleigh-Jeans limit appropriate to the MIRI mid-IR, the eclipse depth
    is ``(Rp/Rs)^2 * (T_day / T_eff)``, so
    ``T_day = T_eff * depth / (Rp/Rs)^2``.  This is compared to two references:

    * ``T_bare_rock`` -- the no-heat-redistribution, zero-albedo day-side maximum
      ``T_eff * sqrt(1/a_rs) * (2/3)^(1/4)`` (a bare rock re-radiates instantly and
      runs hot); and
    * ``T_full_redist`` -- the full-redistribution equilibrium temperature
      ``T_eff * sqrt(1/(2 a_rs)) * (1 - A)^(1/4)`` (an atmosphere carries heat to
      the night side and lowers the day-side).

    A measured day-side within 10% of ``T_bare_rock`` is classified ``bare_rock``;
    one significantly cooler is ``atmosphere`` (heat redistribution and/or albedo);
    in between is ``ambiguous``.  This is the cleanest single atmosphere-vs-rock
    discriminant -- but the RJ approximation and the unknown emissivity/albedo mean
    the temperatures are indicative, not retrieval-grade.
    """
    depth = float(eclipse_depth_ppm) * 1e-6
    if rp_rs <= 0 or a_rs <= 0 or teff_star <= 0:
        return {"t_day_brightness_k": None, "classification": "invalid_input"}
    t_day = teff_star * depth / (rp_rs ** 2)
    t_bare = teff_star * np.sqrt(1.0 / a_rs) * (2.0 / 3.0) ** 0.25
    t_full = (teff_star * np.sqrt(1.0 / (2.0 * a_rs))
              * (1.0 - bond_albedo) ** 0.25)
    if t_day >= 0.9 * t_bare:
        cls = "bare_rock"
    elif t_day <= 1.1 * t_full:
        cls = "atmosphere"
    else:
        cls = "ambiguous"
    # bare_rock (checked first) wins at the hot end, atmosphere at the cool end;
    # the gap between 1.1*T_full and 0.9*T_bare is the honest "ambiguous" band.
    return {
        "t_day_brightness_k": round(float(t_day), 1),
        "t_bare_rock_max_k": round(float(t_bare), 1),
        "t_full_redist_k": round(float(t_full), 1),
        "classification": cls,
        "note": ("day-side near the no-redistribution maximum -> no atmosphere "
                 "(bare rock)" if cls == "bare_rock" else
                 "day-side significantly cooler than a bare rock -> heat "
                 "redistribution / albedo (atmosphere present)"
                 if cls == "atmosphere" else
                 "day-side between the bare-rock and full-redistribution limits "
                 "-> inconclusive"),
    }


# --- 6. Laser-line scan (technosignature) ----------------------------------
def laser_line_scan(wavelength, flux, sigma_min: float = 6.0,
                    min_from_edge: int = 8, max_width: int = 5) -> dict:
    """Search the JWST spectrum for an unresolved emission line (laser).

    Delegates to :func:`seti.panspermia.dossier.narrow_feature_scan`, reusing its
    guards (interior, narrow, bounded positive spike over a smooth continuum).
    JWST resolves finer than Gaia XP, so a genuine monochromatic laser is even
    more sharply localised here; the same guards reject sub-resolution single-bin
    spikes and edge/reconstruction artefacts.  ``flux`` should be a stellar or
    planetary spectrum (e.g. an out-of-transit stellar spectrum, or the residual
    after the transmission model) on the ``wavelength`` grid.
    """
    flux = np.asarray(flux, float)
    result = narrow_feature_scan(wavelength, flux, sigma_min=sigma_min,
                                 min_from_edge=min_from_edge, max_width=max_width)
    return {"laser_line_flag": bool(result.get("xp_feature_flag")),
            "peak": result.get("peak"), "reasons": result.get("reasons", [])}


__all__ = [
    "DEFAULT_BANDS",
    "transit_mask_from_ephemeris",
    "build_transmission_spectrum",
    "molecular_feature_detect",
    "disequilibrium_biosignature",
    "abiotic_false_positive",
    "eclipse_brightness_temperature",
    "laser_line_scan",
    "narrow_feature_scan",
]

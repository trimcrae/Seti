"""Exhaustive per-target signature sweep for the two directed-travel candidates.

The population search narrowed the directed-travel destinations to two standout
hycean-analog systems: **LTT 3780** and **K2-3** (nearby M dwarfs with
sub-Neptunes, closest slow approaches to K2-18).  With the field this small we can
stop doing statistics and instead pull *every archive a runner can reach* for each
star and run every signature detector this repo already has against it:

* **Gaia DR3 astrometry** -> hidden-companion diagnostics (RUWE, astrometric
  excess noise, image-parameter multi-peak fraction, non_single_star): a massive
  unseen companion or an anomalous acceleration.
* **WISE photometry** -> infrared excess vs the stellar photosphere (W1 anchor):
  warm dust or waste-heat re-emission (a Dyson-like signature).
* **ZTF light curves (g, r)** -> aperiodic deep dips, secular fades, and achromatic
  glints via the existing :mod:`seti.dimming` detectors: megastructure transits,
  slow enshrouding, specular reflections.
* **Gaia XP spectra** -> a narrow, unresolved emission feature (a laser line) that
  no smooth stellar continuum reproduces.

Pure-logic scorers (companion diagnostics, IR-colour excess, light-curve verdict,
XP narrow-feature scan) are unit-tested offline; the acquisition runs on the
runner and writes one dossier per target under ``results/panspermia/dossier/``.

Nothing here *assumes* a signature; the point is an honest, complete check that
records "clean" where the data are clean and flags anything that is not.
"""

from __future__ import annotations

import numpy as np

# The two candidates (Gaia DR3), from the reachable-target ranking.
TARGETS = [
    {"name": "LTT 3780", "source_id": 3767281845873242112,
     "ra": 154.64485, "dec": -11.71784},
    {"name": "K2-3", "source_id": 3796690380302214272,
     "ra": 172.33538, "dec": -1.45515},
]

# --- Gaia hidden-companion diagnostics ------------------------------------
# RUWE > 1.4 is the standard binarity flag; astrometric_excess_noise_sig > 2 with
# a non-trivial amplitude indicates an unmodelled photocentre wobble; a high
# image-parameter multi-peak fraction indicates a resolved/partially-resolved
# companion in the Gaia window.
_RUWE_FLAG = 1.4
_EXNOISE_SIG_FLAG = 2.0
_IPD_MULTIPEAK_FLAG = 0.10


def companion_diagnostics(row: dict) -> dict:
    """Interpret Gaia astrometric/quality columns as unseen-companion evidence."""
    def g(*names, default=np.nan):
        for n in names:
            if n in row and row[n] is not None and not (
                    isinstance(row[n], float) and np.isnan(row[n])):
                return float(row[n])
        return default

    ruwe = g("ruwe")
    exn = g("astrometric_excess_noise")
    exn_sig = g("astrometric_excess_noise_sig")
    ipd_mp = g("ipd_frac_multi_peak")
    nss = g("non_single_star", default=0.0)
    reasons = []
    if np.isfinite(ruwe) and ruwe > _RUWE_FLAG:
        reasons.append(f"RUWE={ruwe:.2f}>{_RUWE_FLAG}")
    if np.isfinite(exn_sig) and exn_sig > _EXNOISE_SIG_FLAG and np.isfinite(exn) and exn > 0.1:
        reasons.append(f"astrometric_excess_noise={exn:.2f}mas (sig {exn_sig:.1f})")
    if np.isfinite(ipd_mp) and ipd_mp / 100.0 > _IPD_MULTIPEAK_FLAG:
        reasons.append(f"ipd_frac_multi_peak={ipd_mp:.0f}%")
    if np.isfinite(nss) and nss > 0:
        reasons.append("non_single_star>0 (Gaia NSS solution)")
    return {"ruwe": ruwe, "astrometric_excess_noise": exn,
            "astrometric_excess_noise_sig": exn_sig,
            "ipd_frac_multi_peak": ipd_mp, "non_single_star": nss,
            "companion_flag": bool(reasons), "reasons": reasons}


# --- WISE infrared excess (waste-heat screen) -----------------------------
# For an M-dwarf photosphere the WISE bands sit on the Rayleigh-Jeans tail, so
# W1-W2/W1-W3/W1-W4 are all near zero (Vega mags).  Warm circumstellar dust or
# engineered re-emission raises the longer bands.  Conservative thresholds keep
# molecular-band and calibration scatter from masquerading as excess.
_IR_EXCESS_MIN = {"W1_W2": 0.35, "W1_W3": 0.6, "W1_W4": 0.9}   # mag over photosphere
_IR_EXCESS_SIG = 3.0


def ir_color_excess(phot: dict) -> dict:
    """Colour excesses relative to a bare (W1-anchored) photosphere, with sig."""
    def val(k):
        v = phot.get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    w1, w2, w3, w4 = (val("w1mpro"), val("w2mpro"), val("w3mpro"), val("w4mpro"))
    e1, e2 = val("w1sigmpro"), val("w2sigmpro")
    e3, e4 = val("w3sigmpro"), val("w4sigmpro")
    out = {"w1mpro": w1, "w2mpro": w2, "w3mpro": w3, "w4mpro": w4}
    flags = []
    for key, (a, b, ea, eb) in {
        "W1_W2": (w1, w2, e1, e2), "W1_W3": (w1, w3, e1, e3),
        "W1_W4": (w1, w4, e1, e4)}.items():
        col = a - b
        sig = col / np.sqrt(np.nansum([ea ** 2, eb ** 2]) + 1e-6)
        out[key] = col
        out[key + "_sig"] = sig
        if (np.isfinite(col) and col > _IR_EXCESS_MIN[key]
                and np.isfinite(sig) and sig > _IR_EXCESS_SIG):
            flags.append(f"{key}={col:.2f} ({sig:.1f} sigma)")
    out["ir_excess_flag"] = bool(flags)
    out["reasons"] = flags
    return out


# --- Light-curve verdict (reuses the dimming detectors) --------------------
def lightcurve_verdict(dip: dict | None, secular: dict | None,
                       glint: dict | None) -> dict:
    """Combine the dip/secular/glint detector outputs into anomaly flags."""
    dip = dip or {}
    secular = secular or {}
    glint = glint or {}
    reasons = []
    # Aperiodic deep dipper (Boyajian-like): sustained event, deep, non-periodic.
    if (dip.get("n_dip_events", 0) >= 1 and (dip.get("max_event_depth", 0) or 0) >= 0.10
            and (dip.get("score", 0) or 0) >= 0.5
            and (dip.get("period_power", 1.0) or 1.0) < 0.5):
        reasons.append(f"aperiodic deep dips (score {dip.get('score', 0):.2f}, "
                       f"max event depth {dip.get('max_event_depth', 0):.2f})")
    sig = secular.get("slope_sigma", 0) or 0
    amp = abs(secular.get("total_change_mag", 0) or 0)
    if sig > 5 and amp > 0.05:
        reasons.append(f"secular fade {amp:.2f} mag ({sig:.1f} sigma)")
    if (glint.get("n_glint_events", 0) or 0) >= 1 and (glint.get("score", 0) or 0) >= 0.5:
        reasons.append(f"{glint['n_glint_events']} glint event(s), "
                       f"max brighten {glint.get('max_brighten', 0):.2f}")
    return {"lightcurve_flag": bool(reasons), "reasons": reasons,
            "dip": dip, "secular": secular, "glint": glint}


# --- Gaia XP narrow-feature (laser-line) scan ------------------------------
def narrow_feature_scan(wavelength: np.ndarray, flux: np.ndarray,
                        sigma_min: float = 6.0, min_from_edge: int = 8,
                        max_width: int = 5) -> dict:
    """Flag a narrow, interior, bounded positive spike over a smooth continuum.

    Encodes the XP-resolution guards proven necessary in the spectra channel: a
    real localised feature must be interior (>=``min_from_edge`` samples from
    either end), narrow (<=``max_width`` samples above half-peak), and bounded
    (falls back to the continuum on both sides) -- a 1-sample spike is
    sub-resolution noise and an edge ramp is a basis-reconstruction artefact.
    """
    flux = np.asarray(flux, float)
    n = len(flux)
    if n < 2 * min_from_edge + 3:
        return {"xp_feature_flag": False, "reasons": ["spectrum too short"]}
    # Smooth continuum via a low-order polynomial (a real stellar continuum and
    # its molecular bands are smooth on the XP wavelength grid, so a narrow line
    # sits in the residual).  A polynomial avoids the staircase residuals a moving
    # median leaves on a smooth curve.
    x = np.linspace(-1.0, 1.0, n)
    deg = min(6, max(3, n // 15))
    try:
        cont = np.polyval(np.polyfit(x, flux, deg), x)
    except Exception:  # noqa: BLE001
        cont = np.full(n, np.median(flux))
    resid = flux - cont
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    # Floor the robust scatter to the flux scale so a (near-)noiseless smooth
    # input cannot manufacture a high-sigma spike out of rounding residuals.
    mad = max(mad, 1e-3 * np.median(np.abs(flux)) + 1e-9)
    z = resid / mad
    reasons = []
    peak = None
    for i in range(min_from_edge, n - min_from_edge):
        if z[i] < sigma_min or z[i] != np.max(z[i - 2:i + 3]):
            continue
        half = z[i] / 2.0
        left = i
        while left > 0 and z[left] > half:
            left -= 1
        right = i
        while right < n - 1 and z[right] > half:
            right += 1
        width = right - left - 1
        bounded = z[left] <= half and z[right] <= half
        if 1 < width <= max_width and bounded:
            peak = {"index": int(i), "sigma": float(z[i]), "width": int(width),
                    "wavelength": (float(wavelength[i]) if wavelength is not None
                                   and i < len(wavelength) else None)}
            reasons.append(f"narrow feature at index {i} ({z[i]:.1f} sigma, "
                           f"width {width})")
            break
    return {"xp_feature_flag": peak is not None, "peak": peak, "reasons": reasons}


def dossier_verdict(parts: dict) -> dict:
    """Roll the per-channel flags into one verdict for a target."""
    flags = {k: bool(v.get(f"{k}_flag") or v.get("companion_flag")
                     or v.get("ir_excess_flag") or v.get("lightcurve_flag")
                     or v.get("xp_feature_flag"))
             for k, v in parts.items()}
    any_flag = any(flags.values())
    return {"any_signature_flag": any_flag,
            "channel_flags": flags,
            "verdict": "ANOMALY_FLAGGED" if any_flag else "clean_all_channels"}


__all__ = ["TARGETS", "companion_diagnostics", "ir_color_excess",
           "lightcurve_verdict", "narrow_feature_scan", "dossier_verdict"]

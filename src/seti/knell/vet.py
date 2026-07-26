"""The KNELL contamination gauntlet --- pure functions, one verdict per star.

Every rejection class below corresponds to a *named mechanism by which a real
clock stops or appears to stop*, not to a generic quality cut.  They are applied
in a fixed order --- instrumental first, then survey-detectability, then the
astrophysical cessation mechanisms --- so a star that trips several is reported
under the most mundane one.  ``clean_cessation`` is what is left.

The astrophysical cessation mechanisms this channel must survive
---------------------------------------------------------------
* **Third-body orbital precession** --- the canonical eclipse-cessation case,
  SS Lacertae, whose eclipses ceased in the mid-20th century because a third
  star precessed the inner binary's orbital plane out of the line of sight.  Two
  observables distinguish it: the eclipse depth *declines* over years before the
  eclipses vanish (a stop caused by geometry is gradual, a stop caused by the
  mechanism ending need not be), and the third body is detectable astrometrically
  --- Gaia ``ruwe`` and the non-single-star solutions.  Both are funnel stages.
* **Blazhko-like amplitude modulation** --- an RR Lyrae whose amplitude passes
  through a deep minimum has not stopped; it will come back.  Handled at the
  statistic level (a later detection breaks the required transition pattern) and
  again here through the pre-transition modulation index and the post-transition
  baseline length.
* **Pulsation mode switching** --- power moves to a different frequency.  The
  blind per-block detector fires on *any* frequency, so a mode switch leaves the
  late blocks detected and no transition exists.
* **Spot-cycle evolution in rotational variables** --- a starspot pattern decays
  and the rotational modulation fades.  This is a genuine and common astrophysical
  cessation; it is separated by amplitude, colour, period (rotation periods for
  the relevant stars are days to tens of days), and by activity indicators, and
  where it cannot be separated it is reported as the leading interpretation
  rather than being quietly folded into the survivor count.
* **Cataclysmic-variable disc states** --- a VY Scl star in a low state, or a
  dwarf nova between outbursts, changes its variability character wholesale.
  Rejected on SIMBAD class and on the mean-flux requirement, since disc-state
  changes move the mean.
* **AGN masquerading as a variable star** --- red-noise power is not a clock, but
  over a short block it can produce a formally significant periodogram peak that
  does not repeat.  Rejected on class; the strict pre-transition pattern
  (``min_pre_blocks`` consecutive detections at a *consistent* frequency) is the
  statistical defence.
"""

from __future__ import annotations

import numpy as np

# Instrumental walls.  Both of ZTF's photometric limits manufacture exactly this
# channel's signature: a saturating star's amplitude is compressed non-linearly,
# and a star near the faint limit loses its periodogram peak whenever the survey
# depth dips --- which is a cadence effect wearing a photometric costume.
ZTF_BRIGHT_LIMIT = 13.5
ZTF_FAINT_LIMIT_G = 20.3
ZTF_FAINT_LIMIT_R = 20.1

CROWD_RADIUS_ARCSEC = 5.0
CROWD_DELTA_G = 2.5
RUWE_MAX = 1.4
ASTROMETRIC_EXCESS_NOISE_MAX = 1.0

# Rotational modulation from starspots lives here; a cessation with a period in
# this window and a low amplitude gets the spot-cycle interpretation attached.
SPOT_PERIOD_LO_DAYS = 0.5
SPOT_PERIOD_HI_DAYS = 60.0
SPOT_AMP_MAX_MMAG = 60.0

_CV_TYPES = ("CV*", "DN*", "NL*", "No*", "AM*", "DQ*", "Nova", "CataclyV*",
             "Symbiotic", "SymbSt")
# NB: these are matched as case-insensitive substrings, so every entry must be
# specific enough not to fire on an unrelated class.  A bare "G" here would have
# matched "RGB*", "AGB*" and "GlCl" and silently rejected ordinary giants as AGN.
_AGN_TYPES = ("AGN", "QSO", "Sy1", "Sy2", "Seyfert", "BLLac", "Blazar",
              "LINER", "Galaxy", "GinCl", "EmG", "RadioG")
_YSO_TYPES = ("YSO", "TTau", "Or*", "Ae*", "HerbigAe", "pMS*", "Y*O", "TT*")
_RRL_TYPES = ("RRLyr", "RR*", "RRab", "RRc")


def _has(otype, keys) -> bool:
    if otype is None:
        return False
    s = str(otype)
    if not s or s.lower() in ("nan", "none"):
        return False
    return any(k.lower() in s.lower() for k in keys)


def _f(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def vet_row(res_g, res_r, combined: dict, context: dict | None = None,
            *, cfg: dict | None = None) -> dict:
    """Adjudicate one star.  Returns a dict with ``knell_verdict`` and its reasons."""
    ctx = dict(context or {})
    c = dict(cfg or {})
    bright = float(c.get("ztf_bright_limit", ZTF_BRIGHT_LIMIT))
    faint_g = float(c.get("ztf_faint_limit_g", ZTF_FAINT_LIMIT_G))
    faint_r = float(c.get("ztf_faint_limit_r", ZTF_FAINT_LIMIT_R))
    crowd_dg = float(c.get("crowd_delta_g", CROWD_DELTA_G))
    ruwe_max = float(c.get("ruwe_max", RUWE_MAX))
    aen_max = float(c.get("astrometric_excess_noise_max", ASTROMETRIC_EXCESS_NOISE_MAX))

    reasons: list[str] = []
    out: dict = {
        "knell_verdict": "unset",
        "n_bands_cessation": int(combined.get("n_bands_cessation", 0)),
        "two_band_cessation": bool(combined.get("two_band_cessation", False)),
    }

    # ---- 0. did the test run at all -------------------------------------
    if res_g.status == "insufficient_data" or res_r.status == "insufficient_data":
        out["knell_verdict"] = "insufficient_data"
        out["reasons"] = "one or both bands had too few usable epoch blocks"
        return out

    # ---- 1. instrumental walls ------------------------------------------
    mg = _f(ctx.get("mean_mag_g"))
    mr = _f(ctx.get("mean_mag_r"))
    if (np.isfinite(mg) and mg < bright) or (np.isfinite(mr) and mr < bright):
        reasons.append(f"brighter than the ZTF saturation wall ({bright})")
        out["knell_verdict"] = "saturated"
        out["reasons"] = "; ".join(reasons)
        return out
    if (np.isfinite(mg) and mg > faint_g) or (np.isfinite(mr) and mr > faint_r):
        reasons.append("within the ZTF faint wall, where survey depth sets detectability")
        out["knell_verdict"] = "near_faint_limit"
        out["reasons"] = "; ".join(reasons)
        return out

    # ---- 2. a fade is a different, mundane phenomenon --------------------
    # Checked BEFORE the efficiency gate on purpose: a star that faded also has
    # low efficiency (it is fainter and noisier), and of the two true statements
    # the physical one is more informative than the instrumental one.
    if "mean_flux_changed" in res_g.flags or "mean_flux_changed" in res_r.flags:
        out["knell_verdict"] = "faded_not_ceased"
        out["reasons"] = (f"mean flux moved by {_f(res_g.mean_shift_mag):+.3f} (g) / "
                          f"{_f(res_r.mean_shift_mag):+.3f} (r) mag across the "
                          "transition; the signal sank rather than stopped")
        return out

    # ---- 3. the detectability confounder, before any astrophysics --------
    if "low_efficiency" in res_g.flags or "low_efficiency" in res_r.flags:
        out["knell_verdict"] = "low_efficiency"
        out["reasons"] = ("the post-transition blocks could not have detected the "
                          "signal even if it had persisted (injection-measured "
                          f"eta_min = {min(_f(res_g.eta_min_post, 9), _f(res_r.eta_min_post, 9)):.3f})"
                          " -- this is a cadence/sensitivity change, not a cessation")
        return out

    # ---- 4. still periodic / mode switch ---------------------------------
    if res_g.status == "still_periodic" and res_r.status == "still_periodic":
        out["knell_verdict"] = "still_periodic"
        out["reasons"] = ("every block detects a period; if the frequency moved this "
                          "is mode switching, not cessation")
        return out
    if ("variance_conserved_mode_switch_like" in res_g.flags
            or "variance_conserved_mode_switch_like" in res_r.flags):
        out["knell_verdict"] = "mode_switch"
        out["reasons"] = ("the periodogram peak went away but the integrated excess "
                          f"variance did not (ratio {_f(res_g.excess_var_ratio):.2f} g / "
                          f"{_f(res_r.excess_var_ratio):.2f} r): the power moved in "
                          "frequency rather than stopping, which is mode switching")
        return out
    if not combined.get("n_bands_cessation", 0):
        out["knell_verdict"] = "no_cessation"
        out["reasons"] = (f"g: {res_g.status}; r: {res_r.status}; "
                          + "; ".join(sorted(set(res_g.flags) | set(res_r.flags))))
        return out

    # ---- 5. the two-band rule --------------------------------------------
    if not combined.get("two_band_cessation", False):
        if combined.get("n_bands_cessation", 0) == 1:
            out["knell_verdict"] = "single_band_only"
            out["reasons"] = ("cessation in one band only; the ledger's first rule "
                              "makes this an artefact until confirmed")
        else:
            out["knell_verdict"] = "transition_mismatch"
            out["reasons"] = ("both bands ceased but not at the same epoch or the "
                              f"same period (same_block="
                              f"{combined.get('same_transition_block')}, same_period="
                              f"{combined.get('same_period_both_bands')})")
        return out

    # ---- 6. blending ------------------------------------------------------
    nn = _f(ctx.get("n_neighbors_5as"))
    dg = _f(ctx.get("brightest_neighbor_dg"))
    if np.isfinite(nn) and nn >= 1 and np.isfinite(dg) and dg <= crowd_dg:
        out["knell_verdict"] = "blended"
        out["reasons"] = (f"a Gaia neighbour within {CROWD_RADIUS_ARCSEC}\" is only "
                          f"{dg:.2f} mag fainter; its own variability, and its own "
                          "reference-image history, leak into the ZTF aperture")
        return out

    # ---- 7. named astrophysical cessation mechanisms ----------------------
    ot = ctx.get("simbad_otype")
    if _has(ot, _AGN_TYPES):
        out["knell_verdict"] = "agn_red_noise"
        out["reasons"] = f"SIMBAD type {ot!r}: red noise, not a clock"
        return out
    if _has(ot, _CV_TYPES):
        out["knell_verdict"] = "cataclysmic_disc_state"
        out["reasons"] = f"SIMBAD type {ot!r}: disc-state change is the known mechanism"
        return out
    if _has(ot, _YSO_TYPES):
        out["knell_verdict"] = "yso_variability"
        out["reasons"] = f"SIMBAD type {ot!r}: accretion/occultation variability"
        return out

    ruwe = _f(ctx.get("ruwe"))
    nss = _f(ctx.get("non_single_star"), 0.0)
    aen = _f(ctx.get("astrometric_excess_noise"), 0.0)
    third_body = ((np.isfinite(ruwe) and ruwe > ruwe_max) or nss > 0
                  or (np.isfinite(aen) and aen > aen_max))
    decline = ("pre_decline_precession_like" in res_g.flags
               or "pre_decline_precession_like" in res_r.flags)
    if third_body and decline:
        out["knell_verdict"] = "third_body_precession"
        out["reasons"] = (f"astrometric companion evidence (ruwe={ruwe:.2f}, "
                          f"non_single_star={int(nss)}) *and* a pre-cessation "
                          "amplitude decline: the SS Lacertae mechanism")
        return out
    if decline:
        out["knell_verdict"] = "precession_like"
        out["reasons"] = ("amplitude declined significantly through the pre-transition "
                          f"blocks (trend {_f(res_g.pre_decline_sigma):.1f} sigma in g, "
                          f"{_f(res_r.pre_decline_sigma):.1f} in r): a geometric stop, "
                          "as in SS Lac, rather than a mechanism that ended")
        return out
    if third_body:
        out["knell_verdict"] = "astrometric_companion"
        out["reasons"] = (f"ruwe={ruwe:.2f}, non_single_star={int(nss)}, "
                          f"astrometric_excess_noise={aen:.2f}: an unresolved companion "
                          "is both a blend and a candidate precession driver")
        return out

    if ("pre_amplitude_modulated" in res_g.flags
            or "pre_amplitude_modulated" in res_r.flags) or _has(ot, _RRL_TYPES):
        out["knell_verdict"] = "amplitude_modulated"
        out["reasons"] = ("the pre-transition amplitude was already strongly modulated "
                          f"(index {_f(res_g.amp_modulation):.2f} g / "
                          f"{_f(res_r.amp_modulation):.2f} r) or the star is an RR Lyrae: "
                          "a Blazhko minimum is not a cessation")
        return out

    # ---- 8. survivor, with the leading benign interpretation attached ----
    per = _f(res_r.ref_period, _f(res_g.ref_period))
    amp = _f(res_g.amp_pre_mmag)
    spot_like = (np.isfinite(per) and SPOT_PERIOD_LO_DAYS <= per <= SPOT_PERIOD_HI_DAYS
                 and np.isfinite(amp) and amp <= SPOT_AMP_MAX_MMAG)
    out["knell_verdict"] = "clean_cessation"
    out["spot_cycle_plausible"] = bool(spot_like)
    out["reasons"] = (
        "two-band cessation at a common period and a common epoch, with the "
        "post-transition blocks injection-demonstrated to have been sensitive to "
        "the pre-transition signal, at unchanged mean flux"
        + ("; NOTE period and amplitude are consistent with a decaying starspot "
           "pattern on a rotational variable, which is the leading benign "
           "interpretation and must be addressed before any other claim"
           if spot_like else ""))
    return out


def summarise_flags(res_g, res_r) -> str:
    return ";".join(sorted(set(res_g.flags) | set(res_r.flags)))


__all__ = ["ASTROMETRIC_EXCESS_NOISE_MAX", "CROWD_DELTA_G", "CROWD_RADIUS_ARCSEC",
           "RUWE_MAX", "SPOT_AMP_MAX_MMAG", "SPOT_PERIOD_HI_DAYS",
           "SPOT_PERIOD_LO_DAYS", "ZTF_BRIGHT_LIMIT", "ZTF_FAINT_LIMIT_G",
           "ZTF_FAINT_LIMIT_R", "summarise_flags", "vet_row"]

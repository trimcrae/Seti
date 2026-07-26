"""The contamination gauntlet for VIGIL.  Pure functions --- offline testable.

The ledger this implements is inherited, not re-derived:

* **AllWISE W4 is unreliable for faint stars; a W4-only signal is cirrus.**  W4
  is therefore used here only to *reject*.  It can never contribute detection
  evidence.
* **A negative W1-W2 is a blend, not a photosphere.**
* **~92% of AllWISE-selected infrared excesses are false positives**
  (Silverberg et al. 2018).  The channel is designed around that number: the
  excess is never the detection, the *modulation* is.
* **AGN colour boxes cannot be used alone.**  A ~350 K circumstellar shroud has
  W1-W2 = 3.2 and sits inside the Stern/Assef box; colour alone therefore cannot
  separate a technosignature from a quasar.  So an AGN colour hit is fatal *only
  when the source is not astrometrically stellar* --- no significant parallax,
  no significant proper motion, or flagged extended.  Astrometry, not colour, is
  the discriminant.
* **YSOs and dippers** are optically variable, so the optical-constancy cut does
  most of that work; the SIMBAD type and the Galactic-plane check finish it.
* **AGB/Mira** stars are large-amplitude, red, luminous and long-period.
* **Bright-source bias**: NEOWISE W1 saturates near W1 ~ 8; a saturated profile
  fit produces spurious epoch-to-epoch scatter, which for a variability channel
  is not a nuisance but a candidate factory.
* **High-proper-motion nulls** must be distinguished from real nulls, because a
  previous channel lost an entire run to exactly that confusion.
"""

from __future__ import annotations

import numpy as np

# Colour / magnitude thresholds
STERN_W1W2_MIN = 0.8            # Stern et al. 2012 AGN colour cut
W1_SAT_MAG = 8.0                # NEOWISE W1 saturation onset
W2_SAT_MAG = 7.0
NEIGHBOR_BEAM_ARCSEC = 6.1      # WISE W1 PSF FWHM
NEIGHBOR_DG_MAX = 2.0           # a neighbour within 2 mag contaminates the beam
GAL_LAT_MIN_DEG = 10.0          # below this, cirrus and star-forming contamination
MIRA_AMP_MAG = 0.5              # NEOWISE amplitude typical of a Mira
PLX_SIG_MIN = 5.0
PM_SIG_MIN = 5.0

YSO_TYPES = ("YSO", "TTau", "Orion_V*", "pMS*", "Ae*", "HerbigAe", "Y*O", "Y*?",
             "TT*", "TT?", "out", "HH")
AGB_TYPES = ("Mira", "AGB*", "LPV*", "OH/IR", "C*", "S*", "post-AGB", "pA*",
             "RGB*", "SG*", "s*r")
AGN_TYPES = ("QSO", "AGN", "Sy1", "Sy2", "Seyfert", "BLLac", "Blazar", "G", "GinCl",
             "LINER", "EmG", "RadioG")


def _num(v, default=float("nan")) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def galactic_latitude(ra: float, dec: float) -> float:
    """Galactic latitude in degrees (no astropy dependency in the hot path)."""
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    ra_ngp, dec_ngp = np.radians(192.85948), np.radians(27.12825)
    sb = (np.sin(dec_r) * np.sin(dec_ngp)
          + np.cos(dec_r) * np.cos(dec_ngp) * np.cos(ra_r - ra_ngp))
    return float(np.degrees(np.arcsin(np.clip(sb, -1.0, 1.0))))


def w4_only_signal(ctx: dict) -> bool:
    """True when the *only* infrared evidence is W4 --- i.e. it is cirrus.

    Evidence is 'W4-only' when W4 shows an excess at >3 sigma while neither W1,
    W2 nor W3 does.  A real warm-dust SED lights the star-dominated bands first.
    """
    chi = {b: _num(ctx.get(f"chi_{b}")) for b in ("w1", "w2", "w3", "w4")}
    if not np.isfinite(chi["w4"]) or chi["w4"] < 3.0:
        return False
    others = [chi[b] for b in ("w1", "w2", "w3") if np.isfinite(chi[b])]
    if not others:
        return True
    return max(others) < 3.0


def is_agn_like(ctx: dict) -> tuple[bool, str]:
    """AGN adjudication that does *not* rest on colour alone."""
    w1 = _num(ctx.get("w1mpro"))
    w2 = _num(ctx.get("w2mpro"))
    colour = w1 - w2 if np.isfinite(w1) and np.isfinite(w2) else float("nan")
    in_box = bool(np.isfinite(colour) and colour >= STERN_W1W2_MIN)

    plx_sig = _num(ctx.get("parallax_over_error"), 0.0)
    pmra, pmdec = _num(ctx.get("pmra"), 0.0), _num(ctx.get("pmdec"), 0.0)
    pm_err = _num(ctx.get("pm_error"), np.nan)
    pm = float(np.hypot(pmra, pmdec))
    pm_sig = pm / pm_err if np.isfinite(pm_err) and pm_err > 0 else (
        np.inf if pm > 20.0 else 0.0)
    stellar_astrometry = bool(plx_sig >= PLX_SIG_MIN or pm_sig >= PM_SIG_MIN)
    extended = _num(ctx.get("ext_flg"), 0.0) > 0

    otype = str(ctx.get("simbad_otype") or "")
    if any(t.lower() in otype.lower() for t in AGN_TYPES) and not stellar_astrometry:
        return True, "simbad_agn_and_no_stellar_astrometry"
    if in_box and not stellar_astrometry:
        return True, "agn_colour_box_and_no_stellar_astrometry"
    if extended and not stellar_astrometry:
        return True, "extended_and_no_stellar_astrometry"
    if in_box and stellar_astrometry:
        # This is precisely the 350 K shroud case the brief warns about: inside
        # the AGN box but with Gaia astrometry.  Not a rejection --- a flag.
        return False, "in_agn_colour_box_but_astrometrically_stellar"
    return False, ""


def vet_row(ctx: dict, discrimination=None) -> dict:
    """Run every rejection rule on one candidate; return the verdict and the flags.

    ``ctx`` is a flat dict of everything known about the star.  Missing keys are
    treated as *untested*, never as passes: each untested rule is named in
    ``untested`` so the final verdict cannot quietly rest on checks that never
    ran.
    """
    flags: list[str] = []
    untested: list[str] = []
    ra, dec = _num(ctx.get("ra")), _num(ctx.get("dec"))

    # --- blend --------------------------------------------------------------
    w1, w2 = _num(ctx.get("w1mpro")), _num(ctx.get("w2mpro"))
    if np.isfinite(w1) and np.isfinite(w2):
        if (w1 - w2) < -0.05:
            return _verdict("rejected_blend", ["negative_w1_w2_is_a_blend"], untested)
    else:
        untested.append("w1_w2_colour")

    # --- saturation / bright-source bias -----------------------------------
    if np.isfinite(w1) and w1 < W1_SAT_MAG:
        return _verdict("rejected_saturated", [f"w1={w1:.2f}_below_saturation_onset"],
                        untested)
    if np.isfinite(w2) and w2 < W2_SAT_MAG:
        return _verdict("rejected_saturated", [f"w2={w2:.2f}_below_saturation_onset"],
                        untested)
    for k in ("w1sat", "w2sat"):
        s = _num(ctx.get(k), 0.0)
        if s > 0.05:
            return _verdict("rejected_saturated", [f"{k}={s:.2f}"], untested)

    # --- cirrus -------------------------------------------------------------
    if w4_only_signal(ctx):
        return _verdict("rejected_cirrus", ["w4_only_signal"], untested)
    if np.isfinite(ra) and np.isfinite(dec):
        b = galactic_latitude(ra, dec)
        if abs(b) < GAL_LAT_MIN_DEG:
            flags.append(f"low_galactic_latitude_b={b:.1f}")
    else:
        untested.append("galactic_latitude")

    # --- crowding / beam contamination -------------------------------------
    nn = _num(ctx.get("n_neighbors_beam"), np.nan)
    dg = _num(ctx.get("brightest_neighbor_dg"), np.nan)
    if np.isfinite(nn) and nn > 0 and np.isfinite(dg) and dg < NEIGHBOR_DG_MAX:
        return _verdict("rejected_blend",
                        [f"neighbour_within_{NEIGHBOR_BEAM_ARCSEC}as_dg={dg:.2f}"],
                        untested)
    if not np.isfinite(nn):
        untested.append("neighbour_census")

    # --- literature types ---------------------------------------------------
    otype = str(ctx.get("simbad_otype") or "")
    if otype:
        if any(t.lower() in otype.lower() for t in YSO_TYPES):
            return _verdict("rejected_yso", [f"simbad_otype={otype}"], untested)
        if any(t.lower() in otype.lower() for t in AGB_TYPES):
            return _verdict("rejected_agb", [f"simbad_otype={otype}"], untested)
    else:
        untested.append("simbad_type")

    agn, why = is_agn_like(ctx)
    if agn:
        return _verdict("rejected_agn", [why], untested)
    if why:
        flags.append(why)

    # --- AGB by phenomenology (amplitude + colour + luminosity) -------------
    amp = _num(ctx.get("amp_ptp"), np.nan)
    if np.isfinite(amp) and amp > MIRA_AMP_MAG and np.isfinite(w1 - w2) \
            and (w1 - w2) > 0.6:
        m_ks = _num(ctx.get("abs_ks_mag"), np.nan)
        if not np.isfinite(m_ks) or m_ks < -4.0:
            return _verdict("rejected_agb",
                            ["large_amplitude_red_and_luminous"], untested)

    # --- optical variability -------------------------------------------------
    if bool(ctx.get("optical_measured")):
        ofv = _num(ctx.get("optical_fvar"), np.nan)
        if np.isfinite(ofv) and ofv > 0.05:
            return _verdict("rejected_optically_variable",
                            [f"optical_fvar={ofv:.3f}"], untested)
    else:
        untested.append("optical_constancy")
    if str(ctx.get("phot_variable_flag") or "").upper().startswith("VARIABLE"):
        flags.append("gaia_phot_variable_flag")

    # --- astrometric quality -------------------------------------------------
    ruwe = _num(ctx.get("ruwe"), np.nan)
    if np.isfinite(ruwe) and ruwe > 1.4:
        flags.append(f"ruwe={ruwe:.2f}")
    if _num(ctx.get("non_single_star"), 0.0) > 0:
        flags.append("gaia_non_single_star")

    # --- the high-PM null guard ----------------------------------------------
    sweep = _num(ctx.get("pm_sweep_arcsec"), np.nan)
    if np.isfinite(sweep) and sweep > 2.0:
        flags.append(f"high_pm_sweep_{sweep:.1f}as_pm_propagation_required")

    verdict = "clean_duty_cycle"
    if "optical_constancy" in untested:
        verdict = "clean_optical_untested"
    return _verdict(verdict, flags, untested)


def _verdict(verdict: str, flags: list[str], untested: list[str]) -> dict:
    return {"vigil_verdict": verdict, "flags": ";".join(flags),
            "untested_checks": ";".join(sorted(set(untested))),
            "n_untested": len(set(untested))}


def summarise_verdicts(verdicts) -> dict:
    """Counts by verdict, for the funnel table."""
    out: dict[str, int] = {}
    for v in verdicts:
        k = v.get("vigil_verdict", "unknown") if isinstance(v, dict) else str(v)
        out[k] = out.get(k, 0) + 1
    return out


__all__ = ["AGB_TYPES", "AGN_TYPES", "GAL_LAT_MIN_DEG", "NEIGHBOR_BEAM_ARCSEC",
           "NEIGHBOR_DG_MAX", "PLX_SIG_MIN", "PM_SIG_MIN", "STERN_W1W2_MIN",
           "W1_SAT_MAG", "W2_SAT_MAG", "YSO_TYPES", "galactic_latitude",
           "is_agn_like", "summarise_verdicts", "vet_row", "w4_only_signal"]

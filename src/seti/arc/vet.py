"""The contamination gauntlet for a star above the spot-energy ceiling --- pure.

Every rejection is a named mechanism with its own counter, applied
most-mundane-first so a star that trips several is reported under the
dullest one (docs/arc.md §5):

``no_ceiling``          no amplitude / radius / Teff, so no bound could be
                        computed (unassessable, not a pass)
``below_ceiling``       xi_conservative <= 0: the spots can pay for the flare
``catalogue_doubtful``  the catalogue's own flag marks the flare or star as
                        doubtful / binary / contaminated (Yang & Liu 2019 note
                        earlier catalogues are "seriously polluted")
``companion_suspect``   Gaia RUWE > 1.4, a Gaia DR3 non-single-star solution,
                        or a catalogue binary flag: an unresolved companion
                        --- an M-dwarf flarer hidden in a G-dwarf's aperture is
                        the reason the slow-rotator superflare count fell
                        between Maehara+2012 and Notsu+2019
``blend``               a Gaia neighbour within 12" brighter than G_target + 3:
                        the aperture flux is normalised to the blend, the
                        flare may belong to the neighbour, and the amplitude
                        is diluted (both push xi upward)
``evolved``             logg < 4.0 or R > 1.6 R_sun: a subgiant's ceiling is
                        not the dwarf's, and its "amplitude" is often
                        granulation / pulsation rather than spots
``pulsator_or_eb``      a Gaia DR3 variability class that is not spot
                        modulation (eclipsing binary, delta Scuti, gamma Dor,
                        RR Lyrae, Cepheid, ...): the "flares" are eclipses or
                        pulsation cycles chopped by the flare finder, and the
                        "amplitude" is not a spot

Report-only flags never reject: ``single_flare_above`` (only one independent
flare exceeds the bound --- an isolated event is a measurement problem until
it repeats), ``gaia_unreached``, ``stellar_params_assumed`` (solar Teff /
radius used because the catalogue gave none), ``amplitude_unit_guessed``,
``no_peak_times`` (independence could not be established from timing).

The decisive test the catalogues cannot give --- a pixel-centroid shift
during the flare from the Kepler target pixel file or a TESS FFI cutout ---
is recorded as ``centroid: not_checked`` in stage 1, with the exact quarters
/ sectors each candidate needs pulled in stage 2 (docs/arc.md §6).

Tiers
-----
``none``       unassessable, below the ceiling, or a hard veto tripped
``watch``      above the NOMINAL bound only (xi_nominal > 0 >= xi_conservative)
``interest``   above the conservative bound, no hard veto, but a single
               flare, assumed parameters, or Gaia not reached
``candidate``  >= 2 independent flares above the conservative bound, every
               veto applied and passed; PENDING the stage-2 centroid test
"""

from __future__ import annotations

import json
import re

import numpy as np

from ..metronome.windows import KEPLER_QUARTERS_BKJD

DEFAULT_VET: dict = {
    "ruwe_max": 1.4,
    "logg_min": 4.0,
    "radius_max_rsun": 1.6,
    "neighbour_radius_arcsec": 12.0,
    "neighbour_dg_max": 3.0,
    "min_flares_above": 2,
    "pulsator_classes": ["ECL", "DSCT", "GDOR", "SXPHE", "RR", "CEP", "T2CEP", "ACEP",
                         "ACV", "CP", "MCP", "ROAM", "ROAP", "SXARI", "WD", "AGN", "YSO",
                         "BE", "GCAS", "SDB", "MICROLENSING", "ELL"],
    "doubtful_flag_patterns": ["doubt", "bin", "contam", "eclips", "pollut", "bad", "\\bn\\b",
                               "^[fF]$", "^[bB]$", "^[dD]$"],
}

HARD_VETO_ORDER = ("catalogue_doubtful", "companion_suspect", "blend", "evolved",
                   "pulsator_or_eb")
REPORT_FLAGS = ("single_flare_above", "gaia_unreached", "stellar_params_assumed",
                "amplitude_unit_guessed", "no_peak_times")
NOT_ABOVE = ("no_ceiling", "below_ceiling")


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def companion_suspect(ruwe: float, nss, catalogue_binary: bool = False,
                      ruwe_max: float = 1.4) -> tuple[bool, str | None]:
    """RUWE above threshold, a Gaia NSS solution, or a catalogue binary flag."""
    r = _f(ruwe)
    if np.isfinite(r) and r > float(ruwe_max):
        return True, f"RUWE={r:.2f}>{ruwe_max}"
    n = _f(nss)
    if np.isfinite(n) and n != 0:
        return True, f"gaia_nss={int(n)}"
    if catalogue_binary:
        return True, "catalogue_binary_flag"
    return False, None


def blend(neighbours, g_target: float, *, radius_arcsec: float = 12.0, dg_max: float = 3.0
          ) -> tuple[bool, str | None]:
    """``neighbours`` is an iterable of ``(separation_arcsec, gmag)`` excluding
    the target.  A neighbour inside ``radius_arcsec`` brighter than
    ``g_target + dg_max`` contaminates the aperture."""
    gt = _f(g_target)
    for sep, g in neighbours or []:
        s, gg = _f(sep), _f(g)
        if not np.isfinite(s) or s > float(radius_arcsec):
            continue
        if not np.isfinite(gg):
            continue
        if not np.isfinite(gt) or gg <= gt + float(dg_max):
            return True, f"neighbour {s:.1f}\" G={gg:.2f} (target G={gt:.2f})"
    return False, None


def evolved(logg: float, radius_rsun: float, *, logg_min: float = 4.0,
            radius_max: float = 1.6) -> tuple[bool, str | None]:
    lg, r = _f(logg), _f(radius_rsun)
    if np.isfinite(lg) and lg < float(logg_min):
        return True, f"logg={lg:.2f}<{logg_min}"
    if np.isfinite(r) and r > float(radius_max):
        return True, f"R={r:.2f}>{radius_max}Rsun"
    return False, None


def pulsator_or_eb(vari_class: str | None, classes=None) -> tuple[bool, str | None]:
    classes = DEFAULT_VET["pulsator_classes"] if classes is None else classes
    if not vari_class:
        return False, None
    vc = str(vari_class).strip().upper()
    if not vc or vc in ("NAN", "NONE", ""):
        return False, None
    for c in classes:
        cu = str(c).upper()
        if vc == cu or vc.startswith(cu + "|") or ("|" + cu) in vc or vc.startswith(cu):
            # Gaia classes are single tokens (ECL, DSCT|GDOR|SXPHE, RS, SOLAR_LIKE, ...);
            # RS (RS CVn) and SOLAR_LIKE are spot modulation and must NOT trip.
            if vc.startswith("RS") or vc.startswith("SOLAR"):
                return False, None
            return True, vc
    return False, None


def catalogue_doubtful(flag, patterns=None) -> tuple[bool, str | None]:
    """A catalogue flag column whose value marks the row as doubtful / binary."""
    patterns = DEFAULT_VET["doubtful_flag_patterns"] if patterns is None else patterns
    if flag is None:
        return False, None
    s = str(flag).strip()
    if not s or s.lower() in ("nan", "none", "0", "ok", "good", "y", "yes", "true", "1"):
        return False, None
    for p in patterns:
        if re.search(p, s, flags=re.I):
            return True, s[:40]
    return False, None


# ---------------------------------------------------------------------------
# stage-2 pull list
# ---------------------------------------------------------------------------
def kepler_quarter_of(t_bkjd: float) -> int | None:
    t = _f(t_bkjd)
    if not np.isfinite(t):
        return None
    for q, lo, hi in KEPLER_QUARTERS_BKJD:
        if lo - 1.0 <= t <= hi + 1.0:
            return int(q)
    return None


def stage2_pulls(rec: dict, mission: str) -> list[dict]:
    """The quarters / sectors a stage-2 centroid test must pull for this star:
    one entry per independent flare above the conservative bound."""
    out = []
    flares = rec.get("flares_above") or []
    if isinstance(flares, str):
        # records that round-tripped through xi_<cat>.csv carry it as JSON
        try:
            flares = json.loads(flares) or []
        except ValueError:
            flares = []
    for fl in flares:
        if not isinstance(fl, dict):
            continue
        t = fl.get("t_peak")
        entry = {"t_peak": t, "e_flare_erg": fl.get("e_flare_erg"),
                 "xi_conservative": fl.get("xi_conservative")}
        if str(mission).lower() == "kepler":
            entry["product"] = "kepler_tpf"
            entry["quarter"] = kepler_quarter_of(t) if t is not None else None
        else:
            entry["product"] = "tess_ffi_cutout"
            entry["sector"] = fl.get("sector")
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# the gauntlet
# ---------------------------------------------------------------------------
def vet_star(rec: dict, context: dict | None = None, conf: dict | None = None) -> dict:
    """Apply the gauntlet to one star's ceiling record plus its Gaia context.

    ``rec`` is :func:`seti.arc.ceiling.star_ceiling` output plus ``mission``,
    ``params_assumed``, ``amplitude_unit_guessed``, ``has_peak_times``,
    ``catalogue_flag``.  ``context`` carries ``gaia_reached`` (bool), ``ruwe``,
    ``nss``, ``gmag``, ``neighbours`` (list of ``(sep_arcsec, gmag)``),
    ``vari_class``, ``logg``, ``radius_rsun``, ``catalogue_binary`` (bool).
    """
    c = dict(DEFAULT_VET, **(conf or {}))
    ctx = context or {}
    flags: list[str] = []
    detail: dict = {}
    out = {"tier": "none", "first_veto": None, "flags": flags, "veto_detail": detail,
           "centroid": "not_checked", "stage2_pulls": []}

    x_con = _f(rec.get("xi_conservative_max"))
    x_nom = _f(rec.get("xi_nominal_max"))
    if not rec.get("assessable", np.isfinite(x_con)) or not np.isfinite(x_con):
        out["first_veto"] = "no_ceiling"
        return out
    if x_con <= 0.0:
        out["first_veto"] = "below_ceiling"
        out["tier"] = "watch" if np.isfinite(x_nom) and x_nom > 0.0 else "none"
        return out

    hit, d = catalogue_doubtful(rec.get("catalogue_flag"), c.get("doubtful_flag_patterns"))
    if hit:
        flags.append("catalogue_doubtful")
        detail["catalogue_doubtful"] = d
    reached = bool(ctx.get("gaia_reached", False))
    hit, d = companion_suspect(ctx.get("ruwe"), ctx.get("nss"),
                               bool(ctx.get("catalogue_binary", False)), float(c["ruwe_max"]))
    if hit:
        flags.append("companion_suspect")
        detail["companion_suspect"] = d
    hit, d = blend(ctx.get("neighbours") or [], ctx.get("gmag"),
                   radius_arcsec=float(c["neighbour_radius_arcsec"]),
                   dg_max=float(c["neighbour_dg_max"]))
    if hit:
        flags.append("blend")
        detail["blend"] = d
    logg = ctx.get("logg", rec.get("logg"))
    radius = ctx.get("radius_rsun", rec.get("radius_rsun"))
    hit, d = evolved(logg, radius, logg_min=float(c["logg_min"]),
                     radius_max=float(c["radius_max_rsun"]))
    if hit:
        flags.append("evolved")
        detail["evolved"] = d
    hit, d = pulsator_or_eb(ctx.get("vari_class"), c.get("pulsator_classes"))
    if hit:
        flags.append("pulsator_or_eb")
        detail["pulsator_or_eb"] = d

    # report-only
    if int(rec.get("n_above_conservative", 0) or 0) < int(c["min_flares_above"]):
        flags.append("single_flare_above")
    if not reached:
        flags.append("gaia_unreached")
    if bool(rec.get("params_assumed", False)):
        flags.append("stellar_params_assumed")
    if bool(rec.get("amplitude_unit_guessed", False)):
        flags.append("amplitude_unit_guessed")
    if not bool(rec.get("has_peak_times", True)):
        flags.append("no_peak_times")

    out["stage2_pulls"] = stage2_pulls(rec, str(ctx.get("mission", rec.get("mission", ""))))
    hard = [f for f in HARD_VETO_ORDER if f in flags]
    if hard:
        out["first_veto"] = hard[0]
        out["tier"] = "none"
        return out
    complete = not ({"single_flare_above", "gaia_unreached", "stellar_params_assumed"}
                    & set(flags))
    out["tier"] = "candidate" if complete else "interest"
    return out


def assign_tiers(records: list[dict], contexts: dict | None = None,
                 conf: dict | None = None) -> list[dict]:
    """The gauntlet per star; returns new dicts with ``tier``, ``first_veto``,
    ``flags``, ``centroid`` and ``stage2_pulls`` added."""
    contexts = contexts or {}
    out = []
    for r in records:
        rec = dict(r)
        v = vet_star(rec, contexts.get(rec.get("star_key"), {"mission": rec.get("mission")}), conf)
        rec.update({"tier": v["tier"], "first_veto": v["first_veto"],
                    "flags": ";".join(v["flags"]), "veto_detail": v["veto_detail"],
                    "centroid": v["centroid"], "stage2_pulls": v["stage2_pulls"]})
        out.append(rec)
    return out


def rejection_counters(vetted: list[dict]) -> dict:
    first: dict[str, int] = {}
    every: dict[str, int] = {}
    tiers = {"none": 0, "watch": 0, "interest": 0, "candidate": 0}
    for r in vetted:
        fv = r.get("first_veto") or "passed"
        first[fv] = first.get(fv, 0) + 1
        for f in str(r.get("flags", "") or "").split(";"):
            if f:
                every[f] = every.get(f, 0) + 1
        t = str(r.get("tier", "none"))
        tiers[t] = tiers.get(t, 0) + 1
    for name in NOT_ABOVE + HARD_VETO_ORDER:
        first.setdefault(name, 0)
    for name in HARD_VETO_ORDER + REPORT_FLAGS:
        every.setdefault(name, 0)
    return {"first_veto": first, "flags_raised": every, "tiers": tiers}


def funnel(vetted: list[dict]) -> dict:
    """Stars with xi_conservative > 0 before and after each veto, in order."""
    above = [r for r in vetted if np.isfinite(_f(r.get("xi_conservative_max")))
             and _f(r.get("xi_conservative_max")) > 0]
    out = {"xi_conservative_positive": int(len(above))}
    remaining = list(above)
    for name in HARD_VETO_ORDER:
        remaining = [r for r in remaining
                     if name not in str(r.get("flags", "") or "").split(";")]
        out[f"after_{name}"] = int(len(remaining))
    out["after_all_hard_vetoes"] = int(len(remaining))
    out["with_two_or_more_flares_above"] = int(sum(
        1 for r in remaining if "single_flare_above" not in str(r.get("flags", "")).split(";")))
    out["candidate"] = int(sum(1 for r in remaining if r.get("tier") == "candidate"))
    out["interest"] = int(sum(1 for r in remaining if r.get("tier") == "interest"))
    return out


__all__ = ["DEFAULT_VET", "HARD_VETO_ORDER", "NOT_ABOVE", "REPORT_FLAGS", "assign_tiers",
           "blend", "catalogue_doubtful", "companion_suspect", "evolved", "funnel",
           "kepler_quarter_of", "pulsator_or_eb", "rejection_counters", "stage2_pulls",
           "vet_star"]

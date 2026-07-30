"""Calibrate the momentum ceiling against every published non-gravitational fit.

Two jobs, both independent of how old the Rubin survey is, which is why they can
run today while the residual path waits for a second apparition.

**1. Turn the ceiling's efficiency from an argument into a measured distribution.**
``nongrav.calibration_table`` anchors the gate on three objects (Bennu, 2005 ES70,
2009 BD) with realised thermal-recoil efficiency ``eps_eff = 0.020-0.079``, and
seven more from the live ALeRCE mirror landed at 0.017-0.027.  Ten objects is an
argument.  JPL's Small-Body Database holds a fitted ``A2`` for several hundred
asteroids, each with ``H``, and many with a measured ``diameter`` and ``albedo``
— so the same quantity can be computed for the whole published population.  That
converts "``eps = 0.1`` is a generous envelope" from a claim into a number with a
99th percentile attached, and it does something a small calibration set cannot: it
shows what the *tail* looks like, which is exactly the regime the gate operates in.

Anything that comes out above ``eps = 1`` in that population is either a dark
comet, a body whose diameter is badly wrong, or a fit artefact — and finding out
which is a result either way.

**2. Verify the mirror's ``yarkovsky`` unit against an independent source.**
``lsst_mpc_orbits.yarkovsky`` is *documented* as ``1e-10 au/day^2``, and this
channel's every acceleration depends on that being true — read it raw and every
object jumps ten orders of magnitude above every ceiling.  Documentation is not
verification.  The twelve objects in the mirror with a genuine non-zero value can
be cross-matched to JPL's own ``A2`` for the same objects, and the ratio *measures*
the scale factor.  If it comes back 1e-10, the assumption is confirmed on real
data; if it comes back anything else, the channel has been wrong about its
principal quantity and needs to know.

Network access lives in :func:`fetch_sbdb`, which runs only on the GitHub runner.
Everything that does arithmetic is a pure function and is unit-tested offline.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .nongrav import (
    RHO_TYPICAL_KG_M3,
    SI_TO_AU_PER_DAY2,
    YARKOVSKY_COL_UNIT,
    amr_sphere,
    diameter_m_from_h,
    momentum_ceiling_si,
)

SBDB_QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

# Fields to request.  `A1`/`A2`/`A3` are JPL's Marsden non-gravitational
# parameters in au/day^2; `diameter` is in km where measured; `albedo` is the
# geometric albedo; `rms` is the orbit-fit residual RMS in arcsec.
# VERIFIED WORKING against the live API on 2026-07-30 (939 rows returned).  Do not
# add to this tuple without a live run; put speculative names in
# SBDB_OPTIONAL_FIELDS instead, where a rejection costs nothing.
SBDB_CORE_FIELDS: tuple[str, ...] = (
    "full_name", "pdes", "name", "class", "neo", "pha", "H", "diameter",
    "albedo", "diameter_sigma", "a", "e", "i", "A1", "A2", "A3", "DT", "rms",
    "n_obs_used", "n_del_obs_used", "data_arc", "condition_code",
)

# Fields whose exact spelling is NOT known.  The uncertainty on A2 is the one that
# matters most -- without it an exceedance has no signal-to-noise and is an
# unmeasured number rather than a candidate -- and its name could reasonably be any
# of several forms.  Guessing one and putting it in the main list is what broke the
# second calibration run: a single invalid name 400s the ENTIRE query, so one wrong
# guess cost every field.  These are requested optimistically and dropped
# individually when the API rejects them, which it does by name.
SBDB_OPTIONAL_FIELDS: tuple[str, ...] = (
    "sigma_A2", "A2_sigma", "sigma_a2",
    "sigma_A1", "sigma_A3",
    "first_obs", "last_obs", "epoch", "producer", "two_body", "n_opp",
)

SBDB_FIELDS = ",".join((*SBDB_CORE_FIELDS, *SBDB_OPTIONAL_FIELDS))

# Every spelling the A2 uncertainty might arrive under.  Read in order until one
# is present, because which one the API serves is discovered at run time and must
# not be re-guessed downstream.  MEASURED 2026-07-30: `A2_sigma`.
A2_SIGMA_KEYS: tuple[str, ...] = ("A2_sigma", "sigma_A2", "sigma_a2", "a2_sigma")

# The API reports a bad field by name: {"code":"400","message":"invalid field
# specified: 'sigma_a2'"}.  That is enough to repair the request automatically.
_INVALID_FIELD = re.compile(r"invalid field specified:\s*'([^']+)'")

# Del Vigna et al. (2018) call a Yarkovsky detection reliable only on TWO
# conditions, not one: S/N >= 3 AND agreement with a size-scaled expectation.  The
# gates below are the first condition plus the orbit-quality terms that drive the
# spurious rate -- short arcs, few observations, inflated residuals and a poor
# uncertainty parameter are the documented causes.
MIN_A2_SNR = 3.0
MAX_ORBIT_RMS_ARCSEC = 0.8          # Catalina's astrometric RMS is ~0.69
MIN_DATA_ARC_DAYS = 3650.0          # a decade; Yarkovsky needs many apparitions
MIN_OBS_USED = 100
MAX_CONDITION_CODE = 2.0            # MPC U parameter, 0 = best

# Candidate constraint forms.  The SBDB query API's `sb-cdata` grammar is not
# something to guess at silently: the workflow tries each in order, records what
# every one returned, and uses the first that yields rows.  Same discipline the
# ALeRCE probe applies, and for the same reason -- a wrong guess here returns an
# empty result rather than an error, which reads as "no objects have a fitted A2".
SBDB_CONSTRAINTS: tuple[tuple[str, dict], ...] = (
    ("a2_defined", {"sb-cdata": json.dumps({"AND": ["A2|DF"]})}),
    ("a2_defined_asteroids", {"sb-kind": "a",
                              "sb-cdata": json.dumps({"AND": ["A2|DF"]})}),
    ("neos_all", {"sb-group": "neo"}),
)


# ---------------------------------------------------------------------------
# The arithmetic (pure, offline-tested)
# ---------------------------------------------------------------------------
def diameter_m(h, diameter_km=None, albedo=None,
               default_albedo: float = 0.14) -> np.ndarray:
    """Best available diameter in metres, preferring a *measured* one.

    A measured diameter is used wherever it exists, because the whole point of
    this calibration is to remove the albedo assumption that the H-derived size
    carries.  Where none exists the object's own albedo is used if known, and only
    then does a population mean enter.
    """
    d_km = np.asarray(diameter_km, dtype=float) if diameter_km is not None \
        else np.full(np.shape(h), np.nan)
    out = d_km * 1000.0
    need = ~np.isfinite(out)
    if np.any(need):
        alb = (np.asarray(albedo, dtype=float) if albedo is not None
               else np.full(np.shape(h), np.nan))
        alb = np.where(np.isfinite(alb) & (alb > 0), alb, default_albedo)
        h_arr = np.atleast_1d(np.asarray(h, dtype=float))
        derived = np.array([float(diameter_m_from_h(hi, albedo=float(ai)))
                            for hi, ai in zip(np.ravel(h_arr),
                                              np.ravel(np.broadcast_to(
                                                  alb, h_arr.shape)),
                                              strict=True)])
        derived = derived.reshape(np.shape(out))
        out = np.where(need, derived, out)
    return out


def epsilon_effective(a2_au_day2, diameter_metres,
                      rho_kg_m3: float = RHO_TYPICAL_KG_M3) -> np.ndarray:
    """Realised fraction of the radiation momentum budget an object is using.

    ``|A2|`` divided by the ``epsilon = 1`` ceiling for a body of that size.  This
    is the single number the whole gate rests on: it is bounded above by 1 for any
    radiation-driven process and by 2 for a perfect specular reflector, and the
    calibration set says real thermal recoil realises 2-8% of it.
    """
    a2 = np.abs(np.asarray(a2_au_day2, dtype=float))
    d = np.asarray(diameter_metres, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ceiling = (momentum_ceiling_si(amr_sphere(rho_kg_m3, d), r_au=1.0,
                                       epsilon=1.0) * SI_TO_AU_PER_DAY2)
        return np.where(np.isfinite(ceiling) & (ceiling > 0), a2 / ceiling, np.nan)


# JPL orbit-class codes for comets.  A comet's acceleration is driven by MASS
# LOSS, not by radiation, so it is not bound by the radiation momentum budget at
# all and routinely exceeds it by orders of magnitude.  Mixing comets into the
# asteroid distribution is not a detail: it turns the single most important
# validation the ceiling has -- that it separates the two populations cleanly --
# into a summary statistic that looks like a failure.
COMET_CLASSES = frozenset({"COM", "CTc", "JFc", "JFC", "HTC", "ETc", "CTC",
                           "PAR", "HYP", "ENC"})


def is_comet(row: dict) -> bool:
    """Is this a comet?  By orbit class, or by the ``nP/Name`` designation form."""
    cls = str(row.get("class") or "").strip()
    if cls in COMET_CLASSES:
        return True
    name = str(row.get("full_name") or row.get("pdes") or "")
    # 1P/Halley, 133P/Elst-Pizarro, 75D/Kohoutek, C/2014 UN271, A/2017 U1.
    return bool(re.match(r"^\s*\d*\s*[PDCXAI]/", name))


# Objects that are inactive by classification but are known to be anomalous, so an
# exceedance list that contains them is the gate WORKING rather than failing.  Kept
# separate from `control.py`'s sets because these are not controls for the Rubin
# screen -- they are the published population the ceiling should recover, and
# recovering them is how the ceiling earns the right to flag anything else.
KNOWN_ANOMALOUS: dict[str, str] = {
    "AOUMUAMUA": "1I/'Oumuamua -- interstellar, non-grav acceleration, no coma",
    "A2017U1": "1I/'Oumuamua -- interstellar, non-grav acceleration, no coma",
    "2017U1": "1I/'Oumuamua -- interstellar, non-grav acceleration, no coma",
    "2008GO98": "362P -- quasi-Hilda active asteroid (comet in all but number)",
    "457175": "362P -- quasi-Hilda active asteroid (comet in all but number)",
    "3200": "(3200) Phaethon -- active asteroid, Geminid parent",
    "1983TB": "(3200) Phaethon -- active asteroid, Geminid parent",
    "133P": "133P/Elst-Pizarro -- main-belt comet",
    "1979OW7": "133P/Elst-Pizarro -- main-belt comet",
}


def annotate(name: str) -> str:
    """What is this object already known to be, if anything?

    Turns the exceedance list from a set of designations into an account of itself.
    An exceedance that is a published dark comet, an active asteroid or an
    interstellar object is the ceiling recovering a known anomaly; one that is none
    of those is the only kind worth a second look.
    """
    from .control import control_index, normalise_designation

    keys = {normalise_designation(name)}
    for m in re.finditer(r"\(([^)]+)\)", name):
        keys.add(normalise_designation(m.group(1)))
    m = re.match(r"^\s*\D*(\d+)", name)
    if m:
        keys.add(str(int(m.group(1))))
    keys.add(re.sub(r"[^A-Z0-9]", "", name.upper()))
    for m in re.finditer(r"\(([^)]+)\)", name):
        keys.add(re.sub(r"[^A-Z0-9]", "", m.group(1).upper()))
    # Punctuation-free variants of everything, so "A/2017 U1" and "'Oumuamua"
    # match keys written without the slash or the apostrophe.
    keys |= {re.sub(r"[^A-Z0-9]", "", k.upper()) for k in list(keys)}
    keys.discard("")

    for k in keys:
        if k in KNOWN_ANOMALOUS:
            return KNOWN_ANOMALOUS[k]
    idx = control_index()
    for k in keys:
        hit = idx.get(k)
        if hit:
            return f"{hit['control_set']}: {hit.get('identification') or k}"
    return ""


# Bulk density is the one input the ceiling cannot measure and cannot avoid, and
# calling any single value "generous" hides how much work it is doing.  Measured
# asteroid densities run ~1200-1900 kg/m^3 for rubble piles, ~1300 for C-types and
# ~2700 for S-types.  1000 is therefore NOT a neutral or conservative default --
# it is the extreme low end, chosen to make an exceedance hard to claim, and an
# object that falls below the ceiling only at 1000 has not been ruled out.  It has
# been ruled out under an assumption almost no asteroid satisfies.
DENSITY_GRID_KG_M3: tuple[float, ...] = (1000.0, 1500.0, 2000.0, 2500.0, 3000.0)
ALBEDO_GRID: tuple[float, ...] = (0.05, 0.14, 0.25)

# Del Vigna et al. (2018) second reliability condition: the measured A2 compared
# with a Bennu-scaled expectation for the object's size (Yarkovsky goes as 1/D).
# R <= 2 is a reliable Yarkovsky detection; far above it, the fit is either
# spurious or the object is not doing Yarkovsky.
BENNU_D_M = 490.0
BENNU_A2 = 4.62e-14
MAX_DEL_VIGNA_R = 2.0


# Semimajor axis of Jupiter, for the Tisserand parameter.
JUPITER_A_AU = 5.2044


def tisserand_j(a_au, e, i_deg) -> float:
    """Tisserand parameter with respect to Jupiter.

    THE discriminator this analysis was missing.  ``T_J < 3`` is comet-like
    dynamics — the object is on a Jupiter-crossing or Jupiter-coupled orbit, which
    is where dark comets live and which makes hidden outgassing the natural
    reading.  ``T_J > 3`` is asteroidal dynamics, and an unexplained acceleration
    on such an orbit is harder to attribute to a volatile reservoir that should
    have been depleted long ago.

    It is computed from ``a``, ``e`` and ``i`` alone — all three measured to many
    digits for any multi-apparition object — so unlike epsilon it depends on no
    assumed density and no assumed albedo.
    """
    a, ecc, inc = _f(a_au), _f(e), _f(i_deg)
    if not (math.isfinite(a) and math.isfinite(ecc) and math.isfinite(inc)) or a <= 0:
        return float("nan")
    return (JUPITER_A_AU / a
            + 2.0 * math.cos(math.radians(inc))
            * math.sqrt((a / JUPITER_A_AU) * (1.0 - ecc * ecc)))


def del_vigna_ratio(a2_au_day2, diameter_metres) -> float:
    """``|A2|`` over the Bennu-scaled Yarkovsky expectation for that size."""
    d = _f(diameter_metres)
    if not math.isfinite(d) or d <= 0:
        return float("nan")
    return abs(_f(a2_au_day2)) / (BENNU_A2 * (BENNU_D_M / d))


def sensitivity_grid(a2_au_day2, h, diameter_metres=None,
                     densities=DENSITY_GRID_KG_M3,
                     albedos=ALBEDO_GRID) -> dict:
    """How the exceedance verdict moves across the assumptions it rests on.

    An object is above the ceiling or not *given a density and a size*, and for
    most objects neither is measured.  Reporting one number hides that; reporting
    the grid makes the dependence checkable, and makes it obvious when a
    conclusion rests on the corner of the grid rather than on the object.
    """
    out: dict = {"del_vigna_R": float("nan"),
                 "max_del_vigna_R_for_reliable": MAX_DEL_VIGNA_R,
                 "diameter_measured": diameter_metres is not None,
                 "grid": []}
    sizes = ([(None, _f(diameter_metres))] if diameter_metres is not None
             else [(a, float(diameter_m_from_h(h, albedo=a))) for a in albedos])
    for albedo, d in sizes:
        for rho in densities:
            out["grid"].append({
                "albedo": albedo, "rho_kg_m3": float(rho),
                "diameter_m": float(d),
                "epsilon": float(epsilon_effective(a2_au_day2, d, rho_kg_m3=rho)),
            })
    eps = [g["epsilon"] for g in out["grid"] if math.isfinite(g["epsilon"])]
    if eps:
        out["epsilon_min"] = min(eps)
        out["epsilon_max"] = max(eps)
        out["fraction_of_grid_above_ceiling"] = (
            sum(1 for v in eps if v > 1.0) / len(eps))
        out["robust_above_ceiling"] = min(eps) > 1.0
        out["robust_below_ceiling"] = max(eps) <= 1.0
    d_ref = (_f(diameter_metres) if diameter_metres is not None
             else float(diameter_m_from_h(h, albedo=0.14)))
    out["del_vigna_R"] = del_vigna_ratio(a2_au_day2, d_ref)
    return out


def vet_exceedance(row: dict) -> dict:
    """Is this object's fitted ``A2`` reliable enough to be worth anything?

    An object above the momentum ceiling is making a strong claim, and the base
    rate says most such claims are wrong: a blind search for Yarkovsky signal in
    minor-planet astrometry returns a *majority* of spurious detections at nominal
    S/N > 3, with short arcs, sparse or isolated astrometry and incomplete
    dynamical models the usual causes.  That is why Del Vigna et al. require two
    conditions rather than one.

    Every gate here is named in the output, so an object that fails says which
    test it failed and an object that survives says what it survived.  Nothing is
    scored on absence: a missing uncertainty is ``no_a2_uncertainty``, not a pass.
    """
    reasons: list[str] = []
    a2 = _f(row.get("A2"))
    # Whichever spelling the API actually served.  The request discovers this at
    # run time (measured 2026-07-30: `A2_sigma`), so reading a single hard-coded
    # key here reintroduces exactly the guess the self-repairing request removed --
    # and it did: every object came back with no signal-to-noise and was rejected
    # as `no_a2_uncertainty` while the value sat in the row under another name.
    sigma = float("nan")
    for key in A2_SIGMA_KEYS:
        sigma = _f(row.get(key))
        if math.isfinite(sigma):
            break
    snr = abs(a2) / sigma if (math.isfinite(a2) and math.isfinite(sigma)
                              and sigma > 0) else float("nan")
    if not math.isfinite(sigma) or sigma <= 0:
        reasons.append("no_a2_uncertainty")
    elif snr < MIN_A2_SNR:
        reasons.append(f"a2_snr_{snr:.1f}_below_{MIN_A2_SNR:g}")

    rms = _f(row.get("rms"))
    if math.isfinite(rms) and rms > MAX_ORBIT_RMS_ARCSEC:
        reasons.append(f"orbit_rms_{rms:.2f}as_above_{MAX_ORBIT_RMS_ARCSEC:g}")

    arc = _f(row.get("data_arc"))
    if not math.isfinite(arc):
        reasons.append("no_data_arc")
    elif arc < MIN_DATA_ARC_DAYS:
        reasons.append(f"arc_{arc:.0f}d_below_{MIN_DATA_ARC_DAYS:.0f}d")

    nobs = _f(row.get("n_obs_used"))
    if math.isfinite(nobs) and nobs < MIN_OBS_USED:
        reasons.append(f"only_{nobs:.0f}_observations")

    cc = _f(row.get("condition_code"))
    if math.isfinite(cc) and cc > MAX_CONDITION_CODE:
        reasons.append(f"condition_code_{cc:.0f}_above_{MAX_CONDITION_CODE:g}")

    if str(row.get("two_body") or "").strip() in ("Y", "T", "1", "true"):
        reasons.append("two_body_solution_only")

    return {"a2_snr": snr, "orbit_rms_arcsec": rms, "data_arc_days": arc,
            "n_obs_used": nobs, "condition_code": cc,
            "reliable": not reasons, "fails": reasons}


@dataclass
class EpsilonSummary:
    """The measured distribution of realised efficiency, with its exceedances."""

    n: int = 0
    quantiles: dict = field(default_factory=dict)
    n_above_realistic: int = 0
    n_above_hard: int = 0
    n_above_specular: int = 0
    rho_assumed: float = RHO_TYPICAL_KG_M3
    kind: str = "asteroid"
    n_comets_in_source: int = 0
    n_exceedances_known: int = 0
    n_exceedances_unexplained: int = 0
    n_survivors: int = 0
    survivors: list = field(default_factory=list)
    ok: bool = False
    reason: str = ""
    exceedances: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n": self.n, "kind": self.kind,
                "n_comets_in_source": self.n_comets_in_source,
                "n_above_hard_already_known": self.n_exceedances_known,
                "n_above_hard_unexplained": self.n_exceedances_unexplained,
                "n_above_hard_unexplained_and_reliable": self.n_survivors,
                "survivors": self.survivors,
                "quantiles": self.quantiles,
                "n_above_realistic_0.1": self.n_above_realistic,
                "n_above_hard_1.0": self.n_above_hard,
                "n_above_specular_2.0": self.n_above_specular,
                "rho_assumed_kg_m3": self.rho_assumed, "ok": self.ok,
                "reason": self.reason, "exceedances": self.exceedances}


def summarise_epsilon(rows: list[dict], rho_kg_m3: float = RHO_TYPICAL_KG_M3,
                      default_albedo: float = 0.14,
                      max_exceedances: int = 40,
                      kind: str = "asteroid") -> EpsilonSummary:
    """Realised-efficiency distribution over objects with a fitted ``A2``.

    ``kind`` selects ``"asteroid"``, ``"comet"`` or ``"all"``, and the split is
    the point rather than a refinement.  A comet's acceleration comes from mass
    loss, so it is not bound by the radiation momentum budget and exceeds it by
    orders of magnitude; an asteroid's comes from re-radiated sunlight, so it is.
    Reporting the two together produced a headline of "92 objects above the hard
    ceiling", which reads as the gate failing when it is in fact the gate working
    exactly as designed — every one of those 92 was a periodic comet.

    Objects above ``epsilon = 1`` are listed individually rather than counted,
    because each one is a claim that sunlight cannot drive it.
    """
    out = EpsilonSummary(rho_assumed=float(rho_kg_m3))
    out.kind = kind
    a2, h, dkm, alb, names, kept = [], [], [], [], [], []
    n_comet = 0
    for r in rows:
        v = _f(r.get("A2"))
        if not math.isfinite(v) or v == 0.0:
            continue
        comet = is_comet(r)
        n_comet += int(comet)
        if kind == "asteroid" and comet:
            continue
        if kind == "comet" and not comet:
            continue
        a2.append(v)
        h.append(_f(r.get("H")))
        dkm.append(_f(r.get("diameter")))
        alb.append(_f(r.get("albedo")))
        names.append(str(r.get("full_name") or r.get("pdes") or "?").strip())
        kept.append(r)
    out.n_comets_in_source = n_comet
    if len(a2) < 10:
        out.reason = f"only {len(a2)} objects with a non-zero fitted A2"
        return out
    a2 = np.array(a2)
    d = diameter_m(np.array(h), np.array(dkm), np.array(alb),
                   default_albedo=default_albedo)
    eps = epsilon_effective(a2, d, rho_kg_m3=rho_kg_m3)
    good = np.isfinite(eps) & (eps > 0)
    out.n = int(good.sum())
    if out.n < 10:
        out.reason = f"only {out.n} objects have both A2 and a usable diameter"
        return out
    e = eps[good]
    out.quantiles = {f"p{int(q * 100)}": float(np.quantile(e, q))
                     for q in (0.5, 0.9, 0.99)}
    out.quantiles["max"] = float(e.max())
    out.quantiles["min"] = float(e.min())
    out.n_above_realistic = int((e > 0.1).sum())
    out.n_above_hard = int((e > 1.0).sum())
    out.n_above_specular = int((e > 2.0).sum())
    idx = np.flatnonzero(good)[np.argsort(-e)][:max_exceedances]
    out.exceedances = [
        {"name": names[i], "A2_au_day2": float(a2[i]),
         "diameter_m": float(d[i]), "H": float(h[i]),
         "epsilon_effective": float(eps[i]),
         "diameter_measured": bool(math.isfinite(_f(dkm[i]))),
         "known_as": annotate(names[i]),
         # Dynamical and non-gravitational context, so an exceedance can be read
         # without going back to the source table.  `tisserand_j` is the one
         # discriminator here that depends on NO assumed density and NO assumed
         # albedo: below 3 is comet-like dynamics and hidden outgassing is the
         # natural reading; above 3 is asteroidal and harder to explain away.
         "a_au": _f(kept[i].get("a")), "e": _f(kept[i].get("e")),
         "i_deg": _f(kept[i].get("i")),
         "tisserand_j": tisserand_j(kept[i].get("a"), kept[i].get("e"),
                                    kept[i].get("i")),
         "comet_like_dynamics": (tisserand_j(kept[i].get("a"), kept[i].get("e"),
                                             kept[i].get("i")) < 3.0
                                 if math.isfinite(tisserand_j(
                                     kept[i].get("a"), kept[i].get("e"),
                                     kept[i].get("i"))) else None),
         "A1_au_day2": _f(kept[i].get("A1")), "A3_au_day2": _f(kept[i].get("A3")),
         "albedo": _f(kept[i].get("albedo")),
         "del_vigna_R": del_vigna_ratio(a2[i], d[i]),
         **vet_exceedance(kept[i])}
        for i in idx if eps[i] > 0.1]
    above_hard = [x for x in out.exceedances if x["epsilon_effective"] > 1.0]
    out.n_exceedances_known = sum(1 for x in above_hard if x["known_as"])
    unexplained = [x for x in above_hard if not x["known_as"]]
    out.n_exceedances_unexplained = len(unexplained)
    # The only ones that mean anything: above the ceiling, not already explained,
    # and with an orbit solution good enough for the A2 to be a measurement.
    out.survivors = [x for x in unexplained if x["reliable"]]
    out.n_survivors = len(out.survivors)
    out.ok = True
    return out


# ---------------------------------------------------------------------------
# Unit verification for the mirror's `yarkovsky` column
# ---------------------------------------------------------------------------
def verify_yarkovsky_unit(mirror: list[dict], jpl: list[dict]) -> dict:
    """Measure the scale factor between ``lsst_mpc_orbits.yarkovsky`` and JPL ``A2``.

    Documentation says ``1e-10 au/day^2``.  Every acceleration in this channel
    depends on that, so it is measured rather than trusted: cross-match on
    designation and take the ratio.  A consistent ratio across objects *is* the
    unit; a scattered one means the two sources are not fitting the same quantity
    and the column cannot be used at all.
    """
    from .control import normalise_designation

    by_desig: dict[str, float] = {}
    for r in jpl:
        v = _f(r.get("A2"))
        if not math.isfinite(v) or v == 0.0:
            continue
        for key in ("pdes", "full_name", "name"):
            if r.get(key):
                by_desig.setdefault(normalise_designation(r[key]), v)
        # JPL's `pdes` for a NUMBERED object is its number, not its provisional
        # designation, so an object the mirror calls "1937 UB" is "69230" here and
        # a designation-only match silently misses it -- 9 of 12 in the first run.
        # `full_name` carries both, e.g. "69230 Hermes (1937 UB)", so any
        # parenthesised provisional designation inside it is indexed too.
        for m in re.finditer(r"\(([^)]+)\)", str(r.get("full_name") or "")):
            by_desig.setdefault(normalise_designation(m.group(1)), v)

    pairs = []
    for r in mirror:
        col = _f(r.get("yarkovsky"))
        if not math.isfinite(col) or col == 0.0:
            continue
        desig = (r.get("unpacked_primary_provisional_designation")
                 or r.get("designation"))
        jpl_a2 = by_desig.get(normalise_designation(desig)) if desig else None
        if jpl_a2 is None:
            continue
        pairs.append({"designation": str(desig), "mirror_yarkovsky": col,
                      "jpl_a2_au_day2": jpl_a2, "ratio": jpl_a2 / col})

    out: dict = {"n_matched": len(pairs), "pairs": pairs,
                 "documented_unit_au_day2": YARKOVSKY_COL_UNIT}
    if len(pairs) < 3:
        out["verdict"] = "TOO_FEW_MATCHES"
        out["note"] = ("fewer than three objects matched between the mirror and "
                       "JPL, so the unit could not be measured.  The documented "
                       "scale is assumed and every acceleration in the channel "
                       "still depends on it.")
        return out
    ratios = np.array([p["ratio"] for p in pairs])
    med = float(np.median(ratios))
    spread = float(np.median(np.abs(ratios - med)) / abs(med)) if med else float("nan")
    out["median_ratio"] = med
    out["fractional_scatter"] = spread
    out["implied_unit_au_day2"] = med
    # A consistent ratio is the unit.  A scattered one means the two sources are
    # not fitting the same quantity, which is a stronger and worse result than a
    # wrong scale factor.
    if not math.isfinite(spread) or spread > 0.2:
        out["verdict"] = "INCONSISTENT_ACROSS_OBJECTS"
        out["note"] = (f"the mirror-to-JPL ratio scatters by {spread:.0%} about "
                       f"its median, so the two are not reporting the same "
                       f"quantity and `yarkovsky` cannot be converted at all")
    elif abs(med / YARKOVSKY_COL_UNIT - 1.0) <= 0.2:
        out["verdict"] = "DOCUMENTED_UNIT_CONFIRMED"
        out["note"] = (f"measured scale {med:.3g} au/day^2 per count, against the "
                       f"documented {YARKOVSKY_COL_UNIT:.3g}: confirmed on real "
                       f"data")
    else:
        out["verdict"] = "UNIT_DIFFERS_FROM_DOCUMENTATION"
        out["note"] = (f"measured scale is {med:.3g} au/day^2 per count, NOT the "
                       f"documented {YARKOVSKY_COL_UNIT:.3g} -- every acceleration "
                       f"in the channel is wrong by a factor "
                       f"{med / YARKOVSKY_COL_UNIT:.4g} until this is applied")
    return out


# ---------------------------------------------------------------------------
# Network (runner only)
# ---------------------------------------------------------------------------
def fetch_sbdb(fields: str = SBDB_FIELDS, timeout: float = 120.0,
               constraints=SBDB_CONSTRAINTS, on_result=None) -> dict:
    """Pull objects with fitted non-gravitational parameters from JPL's SBDB.

    Two kinds of uncertainty are handled by measurement rather than by guessing,
    because guessing has already cost two runs here.

    *Which constraint grammar works* — each candidate is tried in turn and every
    attempt's status and row count recorded, so an API change appears as a diff
    rather than as an empty result reading "no asteroid has a fitted A2".

    *Which field names exist* — a single invalid name returns 400 for the WHOLE
    query, so one wrong guess costs every field.  The API names the field it
    objected to, so the request repairs itself: drop that name, retry, record the
    drop.  The surviving list is reported, which is how the correct spelling of
    the A2 uncertainty gets discovered rather than assumed.
    """
    import requests

    out: dict = {"url": SBDB_QUERY_URL, "attempts": {}, "rows": [],
                 "verdict": "NO_DATA_REACHED", "dropped_fields": []}
    # Persistent across constraints: a field rejected once is rejected always, and
    # rediscovering that per constraint would waste a request each time.
    current = [f for f in fields.split(",") if f]
    for name, extra in constraints:
        rec: dict = {"params": dict(extra)}
        resp = None
        for _ in range(len(SBDB_OPTIONAL_FIELDS) + 2):
            params = {"fields": ",".join(current), "limit": "20000", **extra}
            try:
                resp = requests.get(SBDB_QUERY_URL, params=params, timeout=timeout)
            except Exception as exc:                          # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                resp = None
                break
            if resp.status_code == 400:
                m = _INVALID_FIELD.search(resp.text or "")
                if m and m.group(1) in current:
                    current.remove(m.group(1))
                    out["dropped_fields"].append(m.group(1))
                    continue
            break
        rec["fields_used"] = list(current)
        if resp is not None:
            rec["status"] = resp.status_code
            if resp.status_code != 200:
                rec["body"] = resp.text[:300]
            else:
                try:
                    payload = resp.json()
                    cols = payload.get("fields") or []
                    data = payload.get("data") or []
                    rec["n_rows"] = len(data)
                    rec["fields"] = cols
                    if data and not out["rows"]:
                        out["rows"] = [dict(zip(cols, row, strict=False))
                                       for row in data]
                        out["used_constraint"] = name
                        out["verdict"] = "OK"
                except Exception as exc:                      # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["attempts"][name] = rec
        if on_result is not None:
            on_result(name, rec)
        if out["rows"]:
            break
    out["n_rows"] = len(out["rows"])
    return out


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")

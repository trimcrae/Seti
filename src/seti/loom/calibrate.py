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
SBDB_FIELDS = (
    "full_name,pdes,name,class,neo,pha,H,diameter,albedo,diameter_sigma,"
    "a,e,i,A1,A2,A3,DT,rms,n_obs_used,n_del_obs_used,data_arc,condition_code"
)

# Candidate constraint forms.  The SBDB query API's `sb-cdata` grammar is not
# something to guess at silently: the workflow tries each in order, records what
# every one returned, and uses the first that yields rows.  Same discipline the
# ALeRCE probe applies, and for the same reason -- a wrong guess here returns an
# empty result rather than an error, which reads as "no objects have a fitted A2".
SBDB_CONSTRAINTS: tuple[tuple[str, dict], ...] = (
    ("a2_defined", {"sb-cdata": json.dumps({"AND": ["A2|DF"]})}),
    ("a2_defined_asteroids", {"sb-kind": "a",
                              "sb-cdata": json.dumps({"AND": ["A2|DF"]})}),
    ("a2_nonzero", {"sb-cdata": json.dumps({"AND": ["A2|NZ"]})}),
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
    ok: bool = False
    reason: str = ""
    exceedances: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n": self.n, "kind": self.kind,
                "n_comets_in_source": self.n_comets_in_source,
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
    a2, h, dkm, alb, names = [], [], [], [], []
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
         "diameter_measured": bool(math.isfinite(_f(dkm[i])))}
        for i in idx if eps[i] > 0.1]
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

    Tries each candidate constraint form in turn and records what every one
    returned, so a grammar that has changed shows up as a diff rather than as an
    empty result that reads like "no asteroid has a fitted A2".
    """
    import requests

    out: dict = {"url": SBDB_QUERY_URL, "attempts": {}, "rows": [],
                 "verdict": "NO_DATA_REACHED"}
    for name, extra in constraints:
        params = {"fields": fields, "limit": "20000", **extra}
        rec: dict = {"params": {k: v for k, v in params.items() if k != "fields"}}
        try:
            resp = requests.get(SBDB_QUERY_URL, params=params, timeout=timeout)
            rec["status"] = resp.status_code
            if resp.status_code != 200:
                rec["body"] = resp.text[:300]
            else:
                payload = resp.json()
                cols = payload.get("fields") or []
                data = payload.get("data") or []
                rec["n_rows"] = len(data)
                rec["fields"] = cols
                if data:
                    rows = [dict(zip(cols, row, strict=False)) for row in data]
                    if not out["rows"]:
                        out["rows"] = rows
                        out["used_constraint"] = name
                        out["verdict"] = "OK"
        except Exception as exc:                              # noqa: BLE001
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

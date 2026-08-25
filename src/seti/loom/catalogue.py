"""The momentum-ceiling screen, run over the WHOLE small-body catalogue.

What this is, and why it is not `calibrate.py` again
----------------------------------------------------
``seti.loom.calibrate`` asked one question — "what does realised thermal-recoil
efficiency look like?" — and answered it from a single SBDB query,
``{"AND": ["A2|DF"]}``, which returned 939 rows on 2026-07-30.  Out of that came
the two exceedances this repository has been carrying ever since:
``875163 (1998 SH2)`` at ``eps_eff = 1.58`` and ``428209 (2006 VC)`` at
``eps_eff = 1.30`` — above the hard radiation momentum ceiling, reliably fitted,
and not found in the searched literature.

Two objects above a ceiling mean **nothing at all** without a denominator.  A
tail with two entries and a tail with two hundred are the same list until you
know how many objects were screened, what the bulk of the distribution looks
like, and how many objects the tail was *expected* to contain.  This module
supplies that denominator, and it supplies it by measurement rather than by
assumption, in four ways that the calibration run did not:

**1. The census is complete and is checked, not asserted.**  JPL's SBDB query API
will serve ``fields=spkid,full_name,pdes,kind,class,A1,A2,A3`` for *every* small
body with no constraint at all — 1,553,263 asteroids and 4,069 comets, measured
by this repository's DERELICT channel on the same API (``results/derelict/
completeness.json``).  So the number of objects with each non-gravitational
parameter fitted is a *counted* quantity, and the constrained queries
(``A1|DF``, ``A2|DF``, ``A3|DF``) can be checked against it object by object.  A
constrained query that silently misses rows — the failure mode that makes a
population screen a lie — shows up here as a set difference, exactly as it did
for DERELICT's ``A1`` census.

**2. All three components, not ``A2`` alone.**  The momentum ceiling bounds the
*total* non-gravitational acceleration, and JPL's model is
``a = (A1 r_hat + A2 t_hat + A3 n_hat) g(r)`` — three orthogonal components.
Screening on ``|A2|`` alone understates the acceleration of every object with a
fitted radial or normal term, which is most comets and a handful of asteroids.
This screen uses ``|A| = sqrt(A1^2 + A2^2 + A3^2)`` over whichever components
were fitted, and says which those were.

**3. The distance at which the comparison is made is stated, and it matters for
comets only.**  ``A1/A2/A3`` are the coefficients of ``g(r)``, and *both* of
JPL's ``g`` laws are normalised to ``g(1 au) = 1`` — the radiation law
``(1 au / r)^d`` trivially, and the water-ice sublimation law by construction of
its leading constant ``alpha = 0.1112620426`` (verified numerically in
:func:`g_normalisation`, and pinned by the tests).  So ``|A|`` *is* the
acceleration at 1 au whichever law was fitted, and the ceiling at 1 au is
``eps * (Phi/c) * AMR``: the comparison is law-independent, and it is also
independent of whether JPL used ``d = 2`` or ``d = 2.25`` for a Yarkovsky fit.

That law-independence holds **only at 1 au**.  For the radiation law the ratio is
distance-independent everywhere, which is the whole reason ``A2`` is the right
quantity for Yarkovsky.  For the sublimation law it is not: ``eps(r) =
eps(1 au) * g_comet(r) * r^2``, which at ``r = 3 au`` is 65 times smaller and at
``r = 0.5 au`` is 14% larger.  A comet whose perihelion is 3 au and which never
comes near 1 au is therefore being compared at a distance it never visits.  Both
numbers are reported — ``epsilon_1au`` and ``epsilon_at_perihelion`` — with the
law that was actually fitted flagged as *not served by the query API* and
therefore inferred from the object's kind.  For the two standing exceedances,
both asteroids on radiation-law fits, the distinction does not arise.

**4. Every reliability cut reports what it removed.**  A cut that quietly eats
90% of the catalogue changes the meaning of everything downstream, so
:func:`cut_ledger` reports, for each gate, both the *marginal* count (how many
objects fail this gate alone) and the *sequential* count (how many it removes in
pipeline order).  The two differ whenever cuts are correlated, and the gap is
itself informative: ``data_arc >= 3650 d`` and ``n_obs_used >= 100`` overlap
almost completely for comets and hardly at all for recently-discovered NEOs.

What this module is not allowed to do
-------------------------------------
It cannot promote anything to ``candidate``.  LOOM's tier ladder
(``docs/loom.md`` §3.5) requires an *artificiality* channel — an anomalous
area-to-mass ratio, or an acceleration independent of heliocentric distance —
and neither is available from SBDB's query fields.  "Inactive small body with an
acceleration above Yarkovsky expectations" is Seligman et al. (2023)'s dark
comets, an occupied field with an accepted explanation, and the dark-comet
population is known to be incomplete, so a magnitude cut *will* recover objects
that are simply dark comets nobody has published yet.  The ceiling therefore
produces ``interest``, and the objects that reach it are handed to
``seti.loom.litcheck`` with their Tisserand parameter and their comet-like
dynamics flag attached — which is the same discipline ``calibrate.summarise_
epsilon`` applies, carried to catalogue scale.

The tier assignment is not reimplemented here: :func:`record_from_entry` builds a
:class:`seti.loom.screen.ObjectRecord` and calls
:func:`seti.loom.screen.assign_tier`, so this screen and the Rubin screen cannot
drift apart.

Network access lives in :func:`fetch_catalogue` and the ``fetch_*`` helpers,
which run only on the GitHub runner.  Everything that does arithmetic is a pure
function and is unit-tested offline against synthetic catalogues.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .calibrate import (
    _INVALID_FIELD,
    A2_SIGMA_KEYS,
    MAX_CONDITION_CODE,
    MAX_DEL_VIGNA_R,
    MAX_ORBIT_RMS_ARCSEC,
    MIN_A2_SNR,
    MIN_DATA_ARC_DAYS,
    MIN_OBS_USED,
    SBDB_QUERY_URL,
    annotate,
    del_vigna_ratio,
    diameter_m,
    epsilon_effective,
    is_comet,
    sensitivity_grid,
    tisserand_j,
    vet_exceedance,
)
from .nongrav import EPSILON_HARD, EPSILON_INVIOLABLE, EPSILON_REALISTIC, RHO_TYPICAL_KG_M3
from .residuals import g_comet, g_radiation
from .screen import ObjectRecord, Thresholds, assign_tier

# ---------------------------------------------------------------------------
# What to ask JPL for
# ---------------------------------------------------------------------------
# The census pull.  MEASURED WORKING on this exact API by the DERELICT channel
# (results/derelict/completeness.json, 2026-07-30): an unconstrained
# `sb-kind=a` query with these seven fields returned 1,553,263 rows, and
# `sb-kind=c` returned 4,069.  Keeping the field list this short is not
# fastidiousness -- it is what makes a 1.55-million-row response a ~100 MB
# download instead of a gigabyte, and it is the difference between a job that
# finishes and a job that is cancelled before its commit step.
CENSUS_FIELDS: tuple[str, ...] = ("spkid", "full_name", "pdes", "kind", "class",
                                  "A1", "A2", "A3")

# The detail pull, for the tiny subset with a fitted non-gravitational parameter.
# VERIFIED WORKING (939 rows, 2026-07-30, results/loom/calibration.json for the
# LOOM subset; results/derelict/queries.json for `spkid`/`q`/`moid`/`kind`).  Do
# not add a name here without evidence that the server accepts it: ONE invalid
# field returns 400 for the WHOLE query, so a single wrong guess costs every
# field and the run comes back reading "no object has a fitted A2".
DETAIL_CORE_FIELDS: tuple[str, ...] = (
    "spkid", "full_name", "pdes", "name", "kind", "class", "neo", "pha",
    "H", "diameter", "diameter_sigma", "albedo",
    "a", "e", "i", "q", "om", "w", "moid",
    "A1", "A2", "A3", "DT",
    "rms", "n_obs_used", "data_arc", "condition_code", "epoch",
)

# Names whose exact spelling is NOT established.  The per-component uncertainties
# are the ones that matter: without them an exceedance has no signal-to-noise and
# is an unmeasured number rather than a lead.  MEASURED: `A2_sigma` is accepted
# and `sigma_A2` / `sigma_A1` / `sigma_A3` / `sigma_DT` are all rejected
# (results/derelict/queries.json shows the 400s by name), so `A1_sigma` and
# `A3_sigma` are the plausible forms and are requested optimistically.  A
# rejection costs one retry and is recorded, never silently absorbed.
DETAIL_OPTIONAL_FIELDS: tuple[str, ...] = (
    "A1_sigma", "A2_sigma", "A3_sigma", "DT_sigma",
    "n_del_obs_used", "n_opp", "first_obs", "last_obs", "producer", "two_body",
    "rot_per", "GM", "spec_B", "spec_T",
)

# Every spelling each component's uncertainty might arrive under, read in order.
# `A2_SIGMA_KEYS` is imported from `calibrate` rather than restated, so the two
# modules cannot disagree about which name won at run time.
SIGMA_KEYS: dict[str, tuple[str, ...]] = {
    "A1": ("A1_sigma", "sigma_A1", "a1_sigma"),
    "A2": A2_SIGMA_KEYS,
    "A3": ("A3_sigma", "sigma_A3", "a3_sigma"),
}

NONGRAV_COMPONENTS: tuple[str, ...] = ("A1", "A2", "A3")

# `sb-kind`: 'a' = asteroids, 'c' = comets.  Split deliberately rather than
# pulled together, because the two are bound by DIFFERENT PHYSICS -- a comet
# accelerates by shedding mass and is not subject to the radiation momentum
# budget at all -- and because the census row counts are only checkable per kind.
KINDS: tuple[tuple[str, str], ...] = (("a", "asteroid"), ("c", "comet"))

# The row cap the API applies.  A response of exactly this length is a TRUNCATION
# SIGNAL, not a row count, and must never be read as a population.
SBDB_ROW_LIMIT = 200_000

# ---------------------------------------------------------------------------
# The two objects this whole screen exists to put in context
# ---------------------------------------------------------------------------
# Keyed on the permanent number, which is what SBDB's `pdes` carries for a
# numbered object.  Recorded here so the output states plainly where they land in
# the full distribution -- including, and especially, if the answer is that the
# tail is crowded and they are unremarkable within it.
STANDING_EXCEEDANCES: dict[str, str] = {
    "875163": "875163 (1998 SH2) -- LOOM standing exceedance, eps_eff 1.58 from A2 alone "
              "on the 939-row A2|DF pull (2026-07-30); not found in the searched literature",
    "428209": "428209 (2006 VC) -- LOOM standing exceedance, eps_eff 1.30 from A2 alone "
              "on the 939-row A2|DF pull (2026-07-30); not found in the searched literature",
}


# ---------------------------------------------------------------------------
# Small numeric helpers.  `_fz` mirrors `screen._fz`: exactly zero is MISSING.
# ---------------------------------------------------------------------------
def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fz(v) -> float:
    """Like :func:`_f`, but exactly zero is missing.

    SBDB serves ``null`` for an unfitted parameter, so this is belt-and-braces
    here rather than load-bearing as it is against the ALeRCE mirror (where 0.0
    *is* the fill value).  It is applied anyway because the cost of the two
    readings diverging is the cost of treating an unfitted non-gravitational term
    as a measured zero, which is the strongest possible statement that an object
    is ordinary — the exact confusion this channel is built to avoid.
    """
    x = _f(v)
    return float("nan") if x == 0.0 else x


def _fin(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


# ---------------------------------------------------------------------------
# The g(r) normalisation — why a 1 au comparison is law-independent
# ---------------------------------------------------------------------------
def g_normalisation() -> dict:
    """Evaluate both of JPL's ``g(r)`` laws at 1 au.

    The load-bearing fact behind this entire screen.  ``A1/A2/A3`` are the
    coefficients of ``g(r)`` in ``a = (A1 r_hat + A2 t_hat + A3 n_hat) g(r)``, so
    "the acceleration at 1 au" equals ``|A|`` **only if** ``g(1 au) = 1``.  For
    the radiation law ``(1 au / r)^d`` that is true for any ``d``, which is why
    the ``d = 2`` versus ``d = 2.25`` ambiguity in JPL's Yarkovsky fits does not
    touch the ceiling comparison.  For the water-ice sublimation law it is true
    only because ``alpha = 0.1112620426`` was chosen to make it so — a fact worth
    *computing* rather than believing, since if it were false every comet in this
    screen would be compared against the wrong ceiling by a constant factor.

    Both are recomputed from :mod:`seti.loom.residuals`, so a change to those
    constants shows up here and in the test suite rather than in a candidate list.
    """
    return {
        "g_radiation_d2_at_1au": float(g_radiation(1.0, d=2.0)),
        "g_radiation_d225_at_1au": float(g_radiation(1.0, d=2.25)),
        "g_comet_at_1au": float(g_comet(1.0)),
        "note": ("both of JPL's g(r) laws are normalised to 1 at 1 au, so |A| is the "
                 "acceleration at 1 au whichever law was fitted and the ceiling "
                 "comparison at 1 au is law-independent"),
    }


def epsilon_at_distance(epsilon_1au, r_au, law: str = "radiation") -> float:
    """Realised efficiency evaluated at ``r`` rather than at 1 au.

    ``eps(r) = |A| g(r) r^2 / [(Phi/c) AMR au^2]`` = ``eps(1 au) * g(r) * r^2``,
    because the ceiling itself falls as ``r^-2``.

    For the radiation law the ``r^2`` exactly cancels ``g``, so the ratio is the
    same everywhere — which is the reason ``A2`` is the natural quantity for a
    Yarkovsky screen and the reason the 1 au number needs no defence for an
    asteroid.  For the sublimation law it does not cancel: the acceleration falls
    off far faster than the sunlight does, so a comet compared at 1 au when its
    perihelion is 3 au is being credited with an efficiency 65 times its actual
    one at the only distance it ever occupies.

    Which law was fitted is **not** served by the SBDB query API.  Callers pass
    what they inferred from the object's kind and the output says so.
    """
    eps = _f(epsilon_1au)
    r = _f(r_au)
    if not (_fin(eps) and _fin(r)) or r <= 0:
        return float("nan")
    if law == "radiation":
        return eps
    if law == "sublimation":
        return eps * float(g_comet(r)) * r * r
    raise ValueError(f"unknown g(r) law {law!r}")


# ---------------------------------------------------------------------------
# The non-gravitational magnitude and its signal-to-noise
# ---------------------------------------------------------------------------
@dataclass
class NonGravVector:
    """The fitted non-gravitational acceleration of one object, at 1 au.

    ``components`` names which of ``A1``/``A2``/``A3`` were actually fitted, and
    is reported rather than implied: an object with only ``A2`` and an object
    whose ``A1`` happens to be tiny are the same number and completely different
    statements, and the second one has had its radial term *measured*.
    """

    magnitude: float = float("nan")
    sigma: float = float("nan")
    snr: float = float("nan")
    components: tuple[str, ...] = ()
    missing_sigma: tuple[str, ...] = ()
    values: dict = field(default_factory=dict)
    sigmas: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"A_magnitude_au_day2": self.magnitude, "A_sigma_au_day2": self.sigma,
                "A_snr": self.snr, "components_fitted": list(self.components),
                "components_without_uncertainty": list(self.missing_sigma),
                "A_values": dict(self.values), "A_sigmas": dict(self.sigmas)}


def nongrav_vector(row: dict) -> NonGravVector:
    """``|A| = sqrt(A1^2 + A2^2 + A3^2)`` over the components that were fitted.

    The uncertainty is propagated first-order,
    ``sigma_|A|^2 = sum (A_i / |A|)^2 sigma_i^2``, which **ignores the
    covariances** between the components.  SBDB's query API does not serve the
    covariance matrix (the per-object ``sbdb.api`` endpoint does, one object at a
    time, which is affordable only for the tail — see :func:`fetch_covariance`).
    The components of a non-gravitational solution are correlated, often strongly,
    so this signal-to-noise is an approximation and is labelled as one.  It is
    used only as a *gate*, never as a detection statistic, and for the
    overwhelming majority of asteroids -- where ``A2`` is the only fitted
    component -- it is exact and reduces to ``|A2| / sigma_A2``, the quantity
    ``calibrate.vet_exceedance`` already gates on.

    A component fitted without an uncertainty poisons the whole magnitude's
    signal-to-noise: it is named in ``missing_sigma`` and ``snr`` is NaN.  A
    parameter with no uncertainty has no signal-to-noise, and substituting a
    default would silently promote it.
    """
    vec = NonGravVector()
    vals: dict[str, float] = {}
    sigs: dict[str, float] = {}
    missing: list[str] = []
    for comp in NONGRAV_COMPONENTS:
        v = _fz(row.get(comp))
        if not _fin(v):
            continue
        vals[comp] = v
        sigma = float("nan")
        for key in SIGMA_KEYS[comp]:
            sigma = _fz(row.get(key))
            if _fin(sigma) and sigma > 0:
                break
        if _fin(sigma) and sigma > 0:
            sigs[comp] = sigma
        else:
            missing.append(comp)
    vec.values, vec.sigmas = vals, sigs
    vec.components = tuple(vals)
    vec.missing_sigma = tuple(missing)
    if not vals:
        return vec
    mag = math.sqrt(sum(v * v for v in vals.values()))
    vec.magnitude = mag
    if missing or mag <= 0:
        return vec
    var = sum((vals[c] / mag) ** 2 * sigs[c] ** 2 for c in vals)
    vec.sigma = math.sqrt(var)
    vec.snr = mag / vec.sigma if vec.sigma > 0 else float("nan")
    return vec


# ---------------------------------------------------------------------------
# Reliability, generalised from calibrate.vet_exceedance
# ---------------------------------------------------------------------------
# The A2-specific reasons `calibrate.vet_exceedance` emits.  They are stripped and
# replaced with the magnitude's equivalent rather than deleted: an object with a
# fitted A1 and no A2 is not "missing an uncertainty", it is being gated on a
# parameter it never had, and the first version of this screen rejected every
# comet in the catalogue for exactly that reason.
_A2_SNR_REASONS = ("a2_snr_", "no_a2_uncertainty")


def vet_catalogue_row(row: dict, vec: NonGravVector | None = None,
                      min_snr: float = MIN_A2_SNR) -> dict:
    """Is this object's non-gravitational solution reliable enough to mean anything?

    **Reuses** :func:`seti.loom.calibrate.vet_exceedance` for every orbit-quality
    gate — ``rms <= 0.8``, ``data_arc >= 3650 d``, ``n_obs_used >= 100``,
    ``condition_code <= 2``, no two-body-only solution — so the thresholds and
    their justification (Del Vigna et al. 2018: a blind Yarkovsky search returns a
    *majority* of spurious detections at nominal S/N > 3, and short arcs with
    sparse astrometry are the usual cause) live in exactly one place.

    The single thing it replaces is the parameter signal-to-noise, which
    ``vet_exceedance`` computes from ``A2`` alone.  Here it is computed from the
    magnitude of whichever components were fitted.  When ``A2`` is the only fitted
    component the two are identical by construction, and
    ``test_vet_reduces_to_calibrate_for_a2_only_rows`` pins that: if this function
    ever disagrees with ``calibrate.vet_exceedance`` on an A2-only row, the suite
    fails rather than the candidate list changing.
    """
    vec = vec or nongrav_vector(row)
    base = vet_exceedance(row)
    fails = [r for r in base["fails"] if not r.startswith(_A2_SNR_REASONS)]

    if not vec.components:
        fails.append("no_fitted_nongrav_parameter")
    elif vec.missing_sigma:
        fails.append("no_uncertainty_for_" + "_".join(vec.missing_sigma))
    elif not _fin(vec.snr):
        fails.append("no_nongrav_uncertainty")
    elif vec.snr < min_snr:
        fails.append(f"a_snr_{vec.snr:.1f}_below_{min_snr:g}")

    out = dict(base)
    out["fails"] = fails
    out["reliable"] = not fails
    out["a_snr"] = vec.snr
    out["components_fitted"] = list(vec.components)
    # `a2_snr` is kept verbatim from `vet_exceedance` so the two screens' outputs
    # remain directly comparable object by object.
    return out


# ---------------------------------------------------------------------------
# Per-object screening
# ---------------------------------------------------------------------------
def _row_key(row: dict) -> str:
    """A stable identity for merging the constrained pulls and the census.

    ``spkid`` is JPL's own primary key and is preferred wherever it is served.
    ``pdes`` and ``full_name`` are fallbacks, in that order, because ``pdes`` for
    a numbered object is its NUMBER (so "1937 UB" appears as "69230") and
    ``full_name`` carries both forms — the asymmetry that cost the first unit
    verification nine of its twelve cross-matches.
    """
    for key in ("spkid", "pdes", "full_name", "name"):
        v = row.get(key)
        if v not in (None, ""):
            return str(v).strip()
    return ""


def _display_name(row: dict) -> str:
    return str(row.get("full_name") or row.get("pdes") or row.get("name") or "?").strip()


def _standing_key(row: dict) -> str | None:
    """Which standing exceedance is this row, if any?

    Matched on the permanent number wherever it appears — bare in ``pdes``, or
    as the leading number of ``full_name`` — because that is the one form that is
    stable across SBDB's designation columns.
    """
    keys: set[str] = set()
    pdes = str(row.get("pdes") or "").strip()
    if pdes.isdigit():
        keys.add(str(int(pdes)))
    m = re.match(r"^\s*(\d{3,7})\b", str(row.get("full_name") or ""))
    if m:
        keys.add(str(int(m.group(1))))
    for k in keys:
        if k in STANDING_EXCEEDANCES:
            return k
    return None


def screen_entry(row: dict, rho_kg_m3: float = RHO_TYPICAL_KG_M3,
                 default_albedo: float = 0.14,
                 min_snr: float = MIN_A2_SNR) -> dict:
    """Screen one SBDB row against the radiation momentum ceiling.

    Everything physical here is delegated:

    * the diameter, preferring a **measured** one over an ``H``-derived one, to
      :func:`seti.loom.calibrate.diameter_m`;
    * the efficiency itself to :func:`seti.loom.calibrate.epsilon_effective`,
      which is ``|A|`` over the ``epsilon = 1`` ceiling for a sphere of that size
      — the same function that produced the 1.58 and 1.30 this screen is putting
      in context;
    * the Del Vigna size-scaled reliability ratio and the Tisserand parameter to
      :mod:`seti.loom.calibrate`;
    * the assumption sensitivity to :func:`seti.loom.calibrate.sensitivity_grid`,
      because an object is above the ceiling *given a density and a size* and
      reporting one number hides which corner of that grid the verdict rests on.

    What is computed here and nowhere else is the magnitude over all three
    components, the perihelion-distance sensitivity for objects fitted with the
    sublimation law, and the identity bookkeeping.
    """
    vec = nongrav_vector(row)
    comet = is_comet(row)
    h = _f(row.get("H"))
    d_km = _f(row.get("diameter"))
    albedo = _f(row.get("albedo"))
    d_m = float(diameter_m(np.array([h]), np.array([d_km]), np.array([albedo]),
                           default_albedo=default_albedo)[0])
    eps = float(epsilon_effective(vec.magnitude, d_m, rho_kg_m3=rho_kg_m3)) \
        if _fin(vec.magnitude) else float("nan")

    # Which g(r) law JPL fitted is NOT a served field.  Inferred from kind, and
    # the inference is reported so a reader can discount it: a "dark comet" is an
    # asteroid by classification whose fit may well have used the comet law.
    law = "sublimation" if comet else "radiation"
    q = _f(row.get("q"))
    a_au, e, i_deg = _f(row.get("a")), _f(row.get("e")), _f(row.get("i"))
    if not _fin(q) and _fin(a_au) and _fin(e):
        q = a_au * (1.0 - e)
    tj = tisserand_j(a_au, e, i_deg)

    vet = vet_catalogue_row(row, vec=vec, min_snr=min_snr)
    entry = {
        "key": _row_key(row),
        "name": _display_name(row),
        "pdes": str(row.get("pdes") or "").strip() or None,
        "kind": str(row.get("kind") or "").strip() or None,
        "class": str(row.get("class") or "").strip() or None,
        "is_comet": comet,
        "neo": str(row.get("neo") or "").strip() or None,
        "H": h,
        "diameter_m": d_m,
        "diameter_measured": bool(_fin(d_km)),
        "albedo": albedo,
        "a_au": a_au, "e": e, "i_deg": i_deg, "q_au": q,
        "moid_au": _f(row.get("moid")),
        "tisserand_j": tj,
        # The one discriminator here that assumes NO density and NO albedo.  Below
        # 3 is comet-like dynamics and hidden outgassing is the natural reading.
        "comet_like_dynamics": (tj < 3.0) if _fin(tj) else None,
        "epsilon_1au": eps,
        "rho_assumed_kg_m3": float(rho_kg_m3),
        "g_law_assumed": law,
        "epsilon_at_perihelion": epsilon_at_distance(eps, q, law=law),
        "epsilon_at_perihelion_is_meaningful": law == "sublimation" and _fin(q),
        "del_vigna_R": del_vigna_ratio(vec.magnitude, d_m),
        "max_del_vigna_R_for_reliable": MAX_DEL_VIGNA_R,
        "known_as": annotate(_display_name(row)),
        "standing_exceedance": _standing_key(row),
        **vec.as_dict(),
        **vet,
    }
    return entry


def record_from_entry(entry: dict, th: Thresholds) -> ObjectRecord:
    """Put a screened entry on **LOOM's** tier ladder, not on a new one.

    The ratios that :func:`seti.loom.screen.assign_tier` reads are ``|A|`` over
    the ceiling at each ``epsilon``, and ``ceiling_ratio`` scales exactly as
    ``1/epsilon`` — so ``ratio_hard = eps``, ``ratio_realistic = eps /
    epsilon_realistic`` and ``ratio_inviolable = eps / epsilon_inviolable``
    reproduce that function's own arithmetic from the efficiency this screen
    already measured with the better (measured-diameter) size.

    Going through ``assign_tier`` rather than writing a fresh ladder buys three
    things that matter: the tier names mean the same thing in both screens; the
    rule that ``candidate`` requires an artificiality channel is enforced by the
    same code that enforces it for Rubin, so this screen *cannot* promote on
    magnitude alone even by accident; and the orbit-quality failures are folded
    into the reasons list in the same order and the same words.
    """
    rec = ObjectRecord(key=entry.get("key") or "", designation=entry.get("name"),
                       path="sbdb_catalogue")
    rec.h = _f(entry.get("H"))
    rec.diameter_m = _f(entry.get("diameter_m"))
    rec.a, rec.e, rec.i = _f(entry.get("a_au")), _f(entry.get("e")), _f(entry.get("i_deg"))
    rec.tisserand_j = _f(entry.get("tisserand_j"))
    rec.a2_au_day2 = _f(entry.get("A_magnitude_au_day2"))
    rec.a2_unc_au_day2 = _f(entry.get("A_sigma_au_day2"))
    rec.a2_snr = _f(entry.get("A_snr"))
    rec.arc_days = _f(entry.get("data_arc_days"))
    rec.n_opp = _f(entry.get("n_opp"))
    eps = _f(entry.get("epsilon_1au"))
    if _fin(eps):
        rec.ratio_hard = eps / th.epsilon_hard
        rec.ratio_realistic = eps / th.epsilon_realistic
        rec.ratio_inviolable = eps / th.epsilon_inviolable
    # The orbit-quality verdict comes from the SAME gates `calibrate.vet_exceedance`
    # applies, so a badly determined orbit vetoes promotion here exactly as it does
    # in the Rubin screen.  The parameter-SNR reasons are excluded: `assign_tier`
    # applies its own SNR gate to `a2_snr`, and passing the reason in as well would
    # double-count it into `systematic` and silently demote every low-SNR object
    # from `untestable` to `ordinary`.
    quality_fails = [r for r in entry.get("fails", [])
                     if not (r.startswith("a_snr_") or r.startswith("no_uncertainty_for_")
                             or r in ("no_nongrav_uncertainty", "no_fitted_nongrav_parameter"))]
    rec.orbit_ok = not quality_fails
    rec.orbit_reasons = list(quality_fails)
    return assign_tier(rec, th)


# ---------------------------------------------------------------------------
# The cut ledger — what each reliability gate actually removed
# ---------------------------------------------------------------------------
#: The gates, in the order the screen applies them, each with a predicate that is
#: True when the object PASSES.  Order matters only for the sequential column;
#: the marginal column is order-free by construction.
def _cut_definitions(min_snr: float = MIN_A2_SNR) -> list[tuple[str, str, object]]:
    return [
        ("has_nongrav_fit",
         "at least one of A1/A2/A3 fitted and non-zero",
         lambda e: bool(e.get("components_fitted"))),
        ("has_usable_size",
         "a measured diameter, or an H from which one can be derived",
         lambda e: _fin(_f(e.get("diameter_m"))) and _f(e.get("diameter_m")) > 0),
        ("nongrav_snr",
         f"|A| / sigma_|A| >= {min_snr:g} (Del Vigna et al. 2018 condition 1)",
         lambda e: _fin(_f(e.get("A_snr"))) and _f(e.get("A_snr")) >= min_snr),
        ("orbit_rms",
         f"orbit-fit residual RMS <= {MAX_ORBIT_RMS_ARCSEC:g} arcsec",
         lambda e: not (_fin(_f(e.get("orbit_rms_arcsec")))
                        and _f(e.get("orbit_rms_arcsec")) > MAX_ORBIT_RMS_ARCSEC)),
        ("data_arc",
         f"observed arc >= {MIN_DATA_ARC_DAYS:.0f} d (a decade; Yarkovsky needs "
         f"many apparitions)",
         lambda e: _fin(_f(e.get("data_arc_days")))
         and _f(e.get("data_arc_days")) >= MIN_DATA_ARC_DAYS),
        ("n_obs_used",
         f"astrometric observations used >= {MIN_OBS_USED:.0f}",
         lambda e: not (_fin(_f(e.get("n_obs_used")))
                        and _f(e.get("n_obs_used")) < MIN_OBS_USED)),
        ("condition_code",
         f"MPC condition code <= {MAX_CONDITION_CODE:g}",
         lambda e: not (_fin(_f(e.get("condition_code")))
                        and _f(e.get("condition_code")) > MAX_CONDITION_CODE)),
    ]


def cut_ledger(entries: Sequence[dict], min_snr: float = MIN_A2_SNR) -> dict:
    """How many objects each reliability gate removes, marginally and in sequence.

    Both columns are reported because they answer different questions and only
    together do they say what a cut *means*.

    * **marginal** — how many of the objects entering the ledger fail this gate,
      ignoring every other gate.  Order-free, and the right number for "is this
      cut doing anything?".
    * **sequential** — how many this gate removes when applied in pipeline order,
      after the earlier ones.  The right number for "where did the population
      go?", and it is what sums to the total.

    A large marginal count with a small sequential one means the gate is
    redundant against an earlier cut; the reverse is impossible.  The gap between
    them is the correlation between the cuts, which for a small-body catalogue is
    severe — short arcs, few observations and a poor condition code are three
    views of the same objects — and pretending otherwise would make each cut look
    independently expensive.
    """
    cuts = _cut_definitions(min_snr=min_snr)
    n_in = len(entries)
    surviving = list(entries)
    rows = []
    for name, why, passes in cuts:
        marginal = sum(1 for e in entries if not passes(e))
        before = len(surviving)
        surviving = [e for e in surviving if passes(e)]
        rows.append({
            "cut": name, "criterion": why,
            "n_failing_alone": int(marginal),
            "fraction_failing_alone": (marginal / n_in) if n_in else float("nan"),
            "n_removed_in_sequence": int(before - len(surviving)),
            "n_surviving_after": int(len(surviving)),
        })
    return {"n_entering": int(n_in), "n_surviving": int(len(surviving)),
            "fraction_surviving": (len(surviving) / n_in) if n_in else float("nan"),
            "cuts": rows,
            "note": ("`n_failing_alone` ignores every other cut; "
                     "`n_removed_in_sequence` is what this cut removed after the "
                     "ones above it.  The two differ wherever the cuts are "
                     "correlated, which for orbit quality they heavily are.")}


# ---------------------------------------------------------------------------
# The distribution and its tail
# ---------------------------------------------------------------------------
def _normal_sf(z: float) -> float:
    """Upper-tail probability of the standard normal, without scipy."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _poisson_sf(k: int, lam: float) -> float:
    """``P(N >= k)`` for a Poisson with mean ``lam``, by direct summation.

    Direct rather than via a special function because ``lam`` here is small (the
    whole point of the statistic is that the expected tail count is a handful),
    and a summation that is exact for small ``lam`` is preferable to a library
    dependency this repository does not otherwise need in this module.
    """
    if lam <= 0:
        return 1.0 if k <= 0 else 0.0
    if k <= 0:
        return 1.0
    # P(N < k) = sum_{i<k} e^-lam lam^i / i!
    term = math.exp(-lam)
    total = term
    for i in range(1, int(k)):
        term *= lam / i
        total += term
    return max(0.0, min(1.0, 1.0 - total))


def epsilon_distribution(eps: Iterable[float], n_bins_per_dex: int = 4) -> dict:
    """Quantiles and a log-spaced histogram of realised efficiency.

    The histogram is the deliverable, not the quantiles: two exceedances are only
    interpretable against the *shape* of the thing they are in the tail of, and a
    p99 hides whether the distribution falls off a cliff at ``eps = 0.3`` or
    trails smoothly through 1 into 10.
    """
    v = np.asarray(list(eps), dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    out: dict = {"n": int(v.size)}
    if v.size == 0:
        out["reason"] = "no object has both a fitted |A| and a usable size"
        return out
    out["quantiles"] = {f"p{int(q * 100)}": float(np.quantile(v, q))
                        for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95,
                                  0.99, 0.999)}
    out["quantiles"]["min"] = float(v.min())
    out["quantiles"]["max"] = float(v.max())
    for label, thr in (("realistic", EPSILON_REALISTIC), ("hard", EPSILON_HARD),
                       ("specular", EPSILON_INVIOLABLE)):
        n_above = int((v > thr).sum())
        out[f"n_above_{label}_{thr:g}"] = n_above
        out[f"fraction_above_{label}_{thr:g}"] = float(n_above / v.size)
    lg = np.log10(v)
    lo = math.floor(float(lg.min()) * n_bins_per_dex) / n_bins_per_dex
    hi = math.ceil(float(lg.max()) * n_bins_per_dex) / n_bins_per_dex
    edges = np.arange(lo, hi + 0.5 / n_bins_per_dex, 1.0 / n_bins_per_dex)
    if edges.size < 2:
        edges = np.array([lo, lo + 1.0 / n_bins_per_dex])
    counts, _ = np.histogram(lg, bins=edges)
    out["histogram"] = {"log10_epsilon_edges": [float(x) for x in edges],
                        "counts": [int(c) for c in counts]}
    return out


def tail_expectation(eps: Iterable[float],
                     thresholds: Sequence[float] = (EPSILON_REALISTIC, EPSILON_HARD,
                                                    EPSILON_INVIOLABLE),
                     core: tuple[float, float] = (0.25, 0.50)) -> dict:
    """How many objects the tail was *expected* to hold, and how many it holds.

    Ordinary thermal recoil produces a realised efficiency that is positive,
    multiplicative in several roughly independent factors (size, thermal inertia,
    obliquity, spin rate) and spans a decade — so a log-normal is the natural
    reference, and the population LOOM measured on 2026-07-30 (median 0.074,
    p90 0.143 at rho = 2000) is consistent with one.

    The parameters are fitted **from the lower half only** — the median, and the
    scale from the gap between the 25th percentile and the median — because the
    upper half is the thing under test.  Fitting a scale that includes the tail
    would let the objects under test inflate their own expectation, which is the
    classic way to make an excess disappear.

    *How robust that is, stated exactly rather than implied.*  The estimator is
    completely immune to the tail's **values**: an object at ``eps = 50`` and one
    at ``eps = 5e4`` move nothing.  It is immune to the tail's **count** only to
    first order, because every added object shifts the quantile *positions*: a
    tail holding a fraction ``f`` of the sample moves the 25th percentile to the
    core's ``0.25/(1-f)`` and the median to its ``0.5/(1-f)``, which at
    ``f = 0.05`` inflates ``sigma`` by about 3.5% and therefore inflates the
    expected tail count.  That bias is conservative — it makes an excess harder to
    claim, not easier — and at ``f = 0.05`` the screen's verdict is
    ``TAIL_DENSELY_POPULATED`` anyway, at which point a log-normal reference has
    stopped being the interesting question.

    Three things are reported and all three are needed:

    * ``expected`` versus ``observed`` above each threshold, with a Poisson
      probability of seeing at least the observed count.  This is a *descriptive*
      comparison, not a detection statistic — it has no trials correction and the
      log-normal is a reference, not a theory.
    * ``model_check`` — expected versus observed at the sample's own 90th and 99th
      percentiles.  If the log-normal already misses there, its extrapolation to
      ``eps = 1`` means nothing and the reader must be able to see that without
      taking anyone's word for it.
    * the fitted parameters themselves, so the extrapolation can be redone.

    A tail that is *denser* than the reference is not evidence of anything
    exotic — it is the expected signature of a real population of dark comets
    with no published membership list — and it is exactly the finding that would
    make the two standing exceedances unremarkable.  Reporting it plainly is the
    point of the function.
    """
    v = np.asarray(list(eps), dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    out: dict = {"n": int(v.size), "core_quantiles": list(core)}
    if v.size < 20:
        out["ok"] = False
        out["reason"] = f"only {int(v.size)} objects with a usable efficiency"
        return out
    lg = np.log10(v)
    mu = float(np.median(lg))
    q_lo, q_hi = (float(np.quantile(lg, core[0])), float(np.quantile(lg, core[1])))
    # The quantile gap divided by the same gap in a standard normal.  Written
    # generally rather than with the interquartile constant hard-coded, so the
    # default (0.25, 0.50) and a caller's (0.25, 0.75) are the same arithmetic and
    # neither can be right while the other silently is not.
    denom = _normal_quantile(core[1]) - _normal_quantile(core[0])
    sigma = (q_hi - q_lo) / denom if (q_hi > q_lo and denom > 0) else float("nan")
    out["log10_mu"] = mu
    out["log10_sigma"] = sigma
    out["ok"] = bool(_fin(sigma) and sigma > 0)
    if not out["ok"]:
        out["reason"] = "the core of the log-efficiency distribution has zero width"
        return out

    def _expected(thr: float) -> float:
        return float(v.size) * _normal_sf((math.log10(thr) - mu) / sigma)

    rows = []
    for thr in thresholds:
        obs = int((v > thr).sum())
        exp = _expected(thr)
        rows.append({"threshold": float(thr), "observed": obs,
                     "expected_lognormal": exp,
                     "excess": obs - exp,
                     "poisson_p_at_least_observed": _poisson_sf(obs, exp)})
    out["thresholds"] = rows

    check = []
    for q in (0.90, 0.99):
        thr = float(np.quantile(v, q))
        check.append({"sample_quantile": q, "threshold": thr,
                      "observed": int((v > thr).sum()),
                      "expected_lognormal": _expected(thr)})
    out["model_check"] = check
    out["note"] = ("the log-normal is a REFERENCE for what an ordinary "
                   "thermal-recoil population looks like, not a theory of one; a "
                   "Poisson p here carries no trials correction and is descriptive")
    return out


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF, Acklam's rational approximation.

    Needed only so :func:`tail_expectation` can accept a non-default core
    quantile pair without dragging in scipy.  Accurate to ~1e-9 in the relative
    error of the CDF, which is far beyond what a robust scale estimate needs.
    """
    if not 0.0 < p < 1.0:
        return float("nan")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# ---------------------------------------------------------------------------
# The whole screen
# ---------------------------------------------------------------------------
@dataclass
class CatalogueScreen:
    """The full-catalogue momentum-ceiling screen and its denominator."""

    n_rows: int = 0
    n_unique: int = 0
    entries: list = field(default_factory=list)
    populations: dict = field(default_factory=dict)
    tail: list = field(default_factory=list)
    survivors: list = field(default_factory=list)
    standing: dict = field(default_factory=dict)
    verdict: str = "NOT_RUN"
    headline: str = ""
    notes: list = field(default_factory=list)

    def as_dict(self, max_entries: int = 0) -> dict:
        out = {"n_rows": self.n_rows, "n_unique_objects": self.n_unique,
               "populations": self.populations, "tail": self.tail,
               "survivors": self.survivors, "standing_exceedances": self.standing,
               "verdict": self.verdict, "headline": self.headline,
               "notes": self.notes,
               "g_normalisation": g_normalisation()}
        if max_entries:
            out["entries_sample"] = self.entries[:max_entries]
        return out


def screen_catalogue(rows: Sequence[dict], th: Thresholds | None = None,
                     densities: Sequence[float] = (RHO_TYPICAL_KG_M3, 1000.0),
                     default_albedo: float = 0.14,
                     min_snr: float = MIN_A2_SNR,
                     max_tail: int = 400) -> CatalogueScreen:
    """Screen the whole fetched non-gravitational catalogue.  Pure; no network.

    The primary density is 2000 kg/m^3 — the same one ``calibrate`` reports its
    headline at, so the numbers are comparable object by object with the
    939-row run — and 1000 kg/m^3 runs alongside as the generous case.  Density
    is the one input the ceiling can neither measure nor avoid: at rho = 1000 a
    body of the same size has half the mass per unit area and therefore twice the
    permitted acceleration, so an exceedance that exists only at rho = 2000 has
    not been established, it has been assumed.  Both are computed for every
    object and the tail lists both.
    """
    th = th or Thresholds()
    # De-duplicate: an object with both A1 and A2 fitted comes back from two
    # constrained queries, and counting it twice would inflate the denominator and
    # (worse) double any tail entry.
    by_key: dict[str, dict] = {}
    for r in rows:
        key = _row_key(r)
        if not key:
            key = f"__anon_{len(by_key)}"
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = dict(r)
        else:
            # Merge: a later pull may carry a field the earlier one lacked.
            for k, v in r.items():
                if prev.get(k) in (None, "") and v not in (None, ""):
                    prev[k] = v
    unique_rows = list(by_key.values())

    scr = CatalogueScreen(n_rows=len(rows), n_unique=len(unique_rows))
    rho_primary = float(densities[0])
    entries = [screen_entry(r, rho_kg_m3=rho_primary, default_albedo=default_albedo,
                            min_snr=min_snr) for r in unique_rows]
    # The alternate-density efficiencies, attached to the same entry so a reader
    # never has to join two tables to see how much the verdict moved.
    for rho in densities[1:]:
        alt = [screen_entry(r, rho_kg_m3=rho, default_albedo=default_albedo,
                            min_snr=min_snr) for r in unique_rows]
        for e, a in zip(entries, alt, strict=True):
            e[f"epsilon_1au_rho_{int(rho)}"] = a["epsilon_1au"]
    for e in entries:
        rec = record_from_entry(e, th)
        e["tier"] = rec.tier
        e["tier_reasons"] = list(rec.reasons)
    scr.entries = entries

    for kind_label, sel in (("asteroid", lambda e: not e["is_comet"]),
                            ("comet", lambda e: e["is_comet"]),
                            ("all", lambda e: True)):
        subset = [e for e in entries if sel(e)]
        eps_all = [e["epsilon_1au"] for e in subset]
        reliable = [e for e in subset if e["reliable"]]
        block = {
            "n_objects": len(subset),
            "n_with_nongrav_fit": sum(1 for e in subset if e["components_fitted"]),
            "n_by_component": {c: sum(1 for e in subset if c in e["components_fitted"])
                               for c in NONGRAV_COMPONENTS},
            "n_reliable": len(reliable),
            "cut_ledger": cut_ledger(subset, min_snr=min_snr),
            "epsilon_all_screened": epsilon_distribution(eps_all),
            "epsilon_reliable_only": epsilon_distribution(
                [e["epsilon_1au"] for e in reliable]),
            "tail_expectation_all_screened": tail_expectation(eps_all),
            "tail_expectation_reliable_only": tail_expectation(
                [e["epsilon_1au"] for e in reliable]),
            "tiers": {},
        }
        for tier in ("untestable", "ordinary", "watch", "interest", "candidate"):
            block["tiers"][tier] = sum(1 for e in subset if e.get("tier") == tier)
        scr.populations[kind_label] = block

    # The tail, ranked.  Everything above the hard ceiling, whether or not it is
    # reliable and whether or not it is already known — because a tail listing that
    # silently drops the unreliable entries cannot be used to judge how crowded the
    # tail is, which is the question this run exists to answer.
    above = [e for e in entries if _fin(_f(e["epsilon_1au"])) and _f(e["epsilon_1au"]) > 1.0]
    above.sort(key=lambda e: -_f(e["epsilon_1au"]))
    for rank, e in enumerate(above, start=1):
        e["tail_rank"] = rank
    scr.tail = [_tail_entry(e) for e in above[:max_tail]]
    if len(above) > max_tail:
        scr.notes.append(f"{len(above)} objects exceed the hard ceiling; the "
                         f"{max_tail} largest are listed individually and the rest "
                         f"are counted only")

    # Survivors: above the ceiling, reliably fitted, and not already explained by
    # what the object is known to be.  These are what `seti.loom.litcheck` reads.
    # NOTHING here is a candidate: promotion needs an artificiality channel and
    # SBDB serves none, so `tier` for every one of these is `interest` at best.
    scr.survivors = [t for t in scr.tail if t["reliable"] and not t["known_as"]]

    scr.standing = _standing_report(entries, above)
    scr.verdict, scr.headline = _verdict(scr)
    return scr


def _tail_entry(e: dict) -> dict:
    """One tail object, with everything needed to judge it without a re-query."""
    keep = ("tail_rank", "key", "name", "pdes", "kind", "class", "is_comet", "neo",
            "H", "diameter_m", "diameter_measured", "albedo",
            "a_au", "e", "i_deg", "q_au", "moid_au",
            "tisserand_j", "comet_like_dynamics",
            "epsilon_1au", "g_law_assumed", "epsilon_at_perihelion",
            "epsilon_at_perihelion_is_meaningful",
            "A_magnitude_au_day2", "A_sigma_au_day2", "A_snr", "components_fitted",
            "components_without_uncertainty", "A_values",
            "del_vigna_R", "known_as", "standing_exceedance",
            "reliable", "fails", "a2_snr", "orbit_rms_arcsec", "data_arc_days",
            "n_obs_used", "condition_code", "tier", "tier_reasons",
            "rho_assumed_kg_m3")
    out = {k: e.get(k) for k in keep}
    for k in e:
        if k.startswith("epsilon_1au_rho_"):
            out[k] = e[k]
    # The assumption grid, so a reader can see whether the exceedance survives the
    # density and albedo it rests on or lives only in one corner of it.
    d_m = _f(e.get("diameter_m"))
    out["sensitivity"] = sensitivity_grid(
        _f(e.get("A_magnitude_au_day2")), _f(e.get("H")),
        diameter_metres=d_m if e.get("diameter_measured") and _fin(d_m) else None)
    return out


def _standing_report(entries: Sequence[dict], above: Sequence[dict]) -> dict:
    """Where the two standing exceedances land in the full distribution.

    The question this run was commissioned to answer, so it is answered
    explicitly rather than left to be read off a ranked list: for each of
    ``875163 (1998 SH2)`` and ``428209 (2006 VC)``, its efficiency now that all
    three components are used, its rank among everything above the ceiling, how
    many objects sit above it, and its percentile in the whole screened
    population.  If either is no longer in the screened set at all — a re-fit
    since 2026-07-30 could remove its ``A2`` or change its arc — that is reported
    as ``NOT_IN_SCREENED_POPULATION``, which is a finding and not an absence.
    """
    eps_all = np.array([_f(e["epsilon_1au"]) for e in entries], dtype=float)
    eps_all = eps_all[np.isfinite(eps_all) & (eps_all > 0)]
    out: dict = {}
    for key, description in STANDING_EXCEEDANCES.items():
        hit = next((e for e in entries if e.get("standing_exceedance") == key), None)
        rec: dict = {"expected": description}
        if hit is None:
            rec["status"] = "NOT_IN_SCREENED_POPULATION"
            rec["note"] = ("the object carries no fitted non-gravitational parameter "
                           "in this pull, or its designation did not match; either "
                           "way its 2026-07-30 exceedance is not reproduced here and "
                           "the reason must be established before it is quoted again")
            out[key] = rec
            continue
        eps = _f(hit["epsilon_1au"])
        rec["status"] = "SCREENED"
        rec["name"] = hit["name"]
        rec["epsilon_1au"] = eps
        rec["A_snr"] = hit["A_snr"]
        rec["components_fitted"] = hit["components_fitted"]
        rec["reliable"] = hit["reliable"]
        rec["fails"] = hit["fails"]
        rec["tier"] = hit.get("tier")
        rec["tisserand_j"] = hit["tisserand_j"]
        rec["comet_like_dynamics"] = hit["comet_like_dynamics"]
        rec["tail_rank"] = hit.get("tail_rank")
        rec["n_objects_above_it"] = int(sum(1 for e in above
                                            if _f(e["epsilon_1au"]) > eps))
        if eps_all.size and _fin(eps):
            rec["percentile_in_screened_population"] = float(
                (eps_all <= eps).sum() / eps_all.size)
        rec["n_above_hard_ceiling_total"] = int(len(above))
        out[key] = rec
    return out


def _verdict(scr: CatalogueScreen) -> tuple[str, str]:
    """One verdict, stated so that a crowded tail cannot read as a detection.

    Per ``CLAUDE.md`` a clean null is a reason to change the question, never a
    result to write up — and the failure mode this guards against is the opposite
    one: a tail with hundreds of entries reported as "N objects above the
    ceiling!" when what it means is that the dark-comet population is large and
    unpublished and the ceiling alone cannot separate it.
    """
    ast = scr.populations.get("asteroid", {})
    dist = ast.get("epsilon_all_screened", {})
    n_ast = int(dist.get("n", 0) or 0)
    n_above = int(dist.get(f"n_above_hard_{EPSILON_HARD:g}", 0) or 0)
    n_surv = len(scr.survivors)
    exp = None
    for row in (ast.get("tail_expectation_all_screened", {}).get("thresholds") or []):
        if abs(float(row["threshold"]) - EPSILON_HARD) < 1e-9:
            exp = float(row["expected_lognormal"])
    if n_ast == 0:
        return ("NO_SCREENABLE_POPULATION",
                "no asteroid in the pull has both a fitted non-gravitational "
                "parameter and a usable size; the screen did not run")
    frac = n_above / n_ast if n_ast else float("nan")
    head = (f"{n_ast} asteroids screened against the radiation momentum ceiling; "
            f"{n_above} ({frac:.2%}) exceed it, "
            f"{n_surv} of those are reliably fitted and not already identified")
    if exp is not None:
        head += f"; a log-normal core fit expects {exp:.1f}"
    if n_above == 0:
        return "NOTHING_ABOVE_CEILING", head
    # "Crowded" is defined on the fraction, not on the count: a tail of 40 in
    # 600 is a population, a tail of 40 in 60,000 is a tail.
    if frac > 0.05:
        return ("TAIL_DENSELY_POPULATED", head + " -- the tail holds more than 5% of "
                "the screened asteroids, so exceeding the ceiling is a common "
                "property of this population and no individual exceedance, "
                "including the two standing ones, is remarkable on magnitude alone")
    if n_surv == 0:
        return ("ALL_EXCEEDANCES_ALREADY_IDENTIFIED", head + " -- every object above "
                "the ceiling is either unreliably fitted or already known to be "
                "anomalous; nothing new survives")
    return "TAIL_SPARSE_SURVIVORS_PRESENT", head


# ---------------------------------------------------------------------------
# The census: the denominator, counted rather than assumed
# ---------------------------------------------------------------------------
def census_counts(rows: Sequence[dict]) -> dict:
    """Count fitted non-gravitational parameters over an unconstrained census pull.

    Given the cheap seven-field pull for a whole ``sb-kind``, this is the exact
    number of objects with each component fitted, plus the union — the
    denominator every fraction in this channel is quoted against.  Zero is
    treated as missing (``_fz``) for the same reason it is everywhere else.
    """
    keys = {c: set() for c in NONGRAV_COMPONENTS}
    any_key: set[str] = set()
    n = 0
    for r in rows:
        n += 1
        key = _row_key(r) or f"__anon_{n}"
        for c in NONGRAV_COMPONENTS:
            if _fin(_fz(r.get(c))):
                keys[c].add(key)
                any_key.add(key)
    return {"n_rows": n,
            "n_with_A1": len(keys["A1"]), "n_with_A2": len(keys["A2"]),
            "n_with_A3": len(keys["A3"]), "n_with_any": len(any_key),
            "fraction_with_any": (len(any_key) / n) if n else float("nan"),
            "keys_with_any": sorted(any_key)}


def completeness_check(census: dict, fetched_keys: Iterable[str]) -> dict:
    """Did the constrained queries return every object the census says exists?

    The check DERELICT ran on its ``A1`` census and the reason its result could be
    believed.  A ``sb-cdata`` constraint that quietly misses rows turns a
    population screen into a statement about the query, and the failure is
    invisible from inside the constrained result — there is nothing to compare it
    to.  Comparing against the unconstrained census makes it a set difference.
    """
    have = {str(k) for k in fetched_keys}
    want = set(census.get("keys_with_any") or [])
    missing = sorted(want - have)
    extra = sorted(have - want)
    out = {"n_expected_from_census": len(want), "n_fetched": len(have),
           "n_missing_from_constrained": len(missing),
           "n_extra_in_constrained": len(extra),
           "missing_sample": missing[:50], "extra_sample": extra[:50]}
    if not want:
        out["verdict"] = "NO_CENSUS"
        out["note"] = ("no census pull succeeded, so the constrained queries could "
                       "not be checked and their completeness is ASSUMED")
    elif not missing:
        out["verdict"] = "CONSTRAINT_COMPLETE"
    else:
        out["verdict"] = "CONSTRAINT_INCOMPLETE"
        out["note"] = (f"{len(missing)} objects have a fitted non-gravitational "
                       f"parameter in the unconstrained census but did not come back "
                       f"from any constrained query; every population fraction below "
                       f"is a lower bound until this is explained")
    return out


# ---------------------------------------------------------------------------
# Network (runner only)
# ---------------------------------------------------------------------------
def _sbdb_get(params: dict, timeout: float, session=None):
    """One HTTP GET against the SBDB query endpoint.

    ``session`` is anything with a ``requests``-shaped ``get``, and ``requests``
    itself is imported only when none is supplied.  That is what makes the
    self-repair loop, the truncation guard and the chunk/resume logic testable
    offline against a fake transport, which is the whole reason the tests can
    cover the network layer in a sandbox with no egress.
    """
    if session is None:
        import requests

        session = requests
    return session.get(SBDB_QUERY_URL, params=params, timeout=timeout)


def sbdb_query(params: dict, fields: Sequence[str], timeout: float = 300.0,
               session=None, max_repairs: int = 12) -> dict:
    """One SBDB query, repairing itself when the server rejects a field by name.

    A single invalid field name returns 400 for the WHOLE query, so one wrong
    guess costs every field and the run comes back reading "no object has a
    fitted A2".  The API names the field it objected to, so the request drops
    that name and retries, and every drop is recorded — which is how the correct
    spelling of ``A1_sigma`` gets *discovered* rather than assumed.  Identical in
    behaviour to ``calibrate.fetch_sbdb``'s repair loop, and it uses that
    module's compiled pattern so the two cannot diverge.
    """
    current = [f for f in fields if f]
    rec: dict = {"params": {k: v for k, v in params.items() if k != "fields"},
                 "dropped_fields": [], "status": None, "n_rows": 0, "rows": []}
    resp = None
    for _ in range(max_repairs):
        try:
            resp = _sbdb_get({**params, "fields": ",".join(current),
                              "limit": str(SBDB_ROW_LIMIT)}, timeout, session)
        except Exception as exc:                              # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return rec
        if resp.status_code == 400:
            m = _INVALID_FIELD.search(resp.text or "")
            if m and m.group(1) in current:
                current.remove(m.group(1))
                rec["dropped_fields"].append(m.group(1))
                continue
        break
    rec["fields_used"] = list(current)
    if resp is None:
        rec["error"] = "no response"
        return rec
    rec["status"] = resp.status_code
    if resp.status_code != 200:
        rec["body"] = (resp.text or "")[:300]
        return rec
    try:
        payload = resp.json()
    except Exception as exc:                                  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return rec
    cols = payload.get("fields") or []
    data = payload.get("data") or []
    rec["fields"] = cols
    rec["n_rows"] = len(data)
    # The server's own count, when it gives one.  A mismatch against the number of
    # rows actually parsed is a truncated or interrupted transfer, and it must not
    # be read as a population.
    if payload.get("count") is not None:
        try:
            rec["server_count"] = int(payload["count"])
        except (TypeError, ValueError):
            rec["server_count"] = payload["count"]
    rec["rows"] = [dict(zip(cols, r, strict=False)) for r in data]
    if len(data) >= SBDB_ROW_LIMIT:
        rec["truncated"] = True
        rec["note"] = (f"exactly {SBDB_ROW_LIMIT} rows returned: this is the row cap, "
                       f"not a row count, and the result is TRUNCATED")
    return rec


def fetch_field_names(timeout: float = 60.0, session=None) -> dict:
    """Ask the server which field names it accepts, rather than guessing them.

    JPL documents an ``info=field`` probe on the query endpoint.  When it answers,
    the optional field list can be filtered before the first real request instead
    of being discovered one 400 at a time; when it does not, the repair loop in
    :func:`sbdb_query` still handles it, so this is an optimisation and never a
    dependency.
    """
    out: dict = {"available": None, "notes": []}
    for params in ({"fields": "full_name", "limit": "1", "info": "field"},
                   {"fields": "full_name", "limit": "1"}):
        try:
            resp = _sbdb_get(params, timeout, session)
            payload = resp.json()
        except Exception as exc:                              # noqa: BLE001
            out["notes"].append(f"probe {params} failed: {type(exc).__name__}: {exc}"[:200])
            continue
        for key in ("field", "fields", "available_fields"):
            val = payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                names = sorted({str(d.get("name")) for d in val if d.get("name")})
                if names:
                    out["available"] = names
                    out["notes"].append(f"discovered {len(names)} field names via '{key}'")
                    return out
        out["notes"].append(f"probe {params} returned keys {sorted(payload)[:12]}")
    return out


def fetch_census(kind: str, timeout: float = 900.0, session=None) -> dict:
    """The unconstrained pull for one ``sb-kind``: the denominator.

    ~1.55 million rows for ``kind='a'``, seven short fields each.  This is the
    single most expensive request in the channel and the single most important
    one, because everything downstream is a fraction of it.  It is issued last in
    :func:`fetch_catalogue` for exactly that reason: a job that dies here still
    has the screen on disk, whereas a job that dies in the detail pull has
    nothing.
    """
    rec = sbdb_query({"sb-kind": kind, "full-prec": "1"}, CENSUS_FIELDS,
                     timeout=timeout, session=session)
    rec["kind"] = kind
    if rec.get("rows"):
        rec["counts"] = census_counts(rec["rows"])
        # The row bodies are enormous and are never needed again once counted.
        rec["counts_only"] = True
    return rec


def fetch_detail(kind: str, component: str, fields: Sequence[str],
                 timeout: float = 300.0, session=None) -> dict:
    """Every object of one kind with ``component`` fitted, with the full field set."""
    rec = sbdb_query({"sb-kind": kind, "full-prec": "1",
                      "sb-cdata": json.dumps({"AND": [f"{component}|DF"]})},
                     fields, timeout=timeout, session=session)
    rec["kind"] = kind
    rec["component"] = component
    return rec


def fetch_covariance(spkid: str, timeout: float = 60.0, session=None) -> dict:
    """Per-object solution detail, including the non-gravitational covariance.

    Affordable only for the tail — one request per object — and used for nothing
    but honesty about :func:`nongrav_vector`'s signal-to-noise, which propagates
    the component uncertainties as if they were independent.  Recorded when it is
    available, and its absence is recorded too.
    """
    if session is None:
        import requests

        session = requests
    url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    out: dict = {"spkid": str(spkid)}
    try:
        resp = session.get(
            url, params={"sstr": str(spkid), "cov": "mat", "full-prec": "1",
                         "phys-par": "1"}, timeout=timeout)
        out["status"] = resp.status_code
        if resp.status_code == 200:
            payload = resp.json()
            out["orbit_covariance_present"] = bool(
                (payload.get("orbit") or {}).get("covariance"))
            out["model_pars"] = [p.get("name") for p in
                                 ((payload.get("orbit") or {}).get("model_pars") or [])]
        else:
            out["body"] = (resp.text or "")[:200]
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def fetch_catalogue(timeout: float = 300.0, census_timeout: float = 900.0,
                    session=None, on_result=None, do_census: bool = True,
                    resume_rows: Sequence[dict] | None = None,
                    done_chunks: Iterable[str] = ()) -> dict:
    """Pull the whole non-gravitational catalogue, chunk by chunk and resumably.

    Six small detail queries (``{A1, A2, A3} x {asteroid, comet}``) followed by
    two large census queries.  Each chunk calls ``on_result`` the moment it
    returns, so the orchestrator can write the record to disk between chunks and
    a job that is cancelled part way through loses only the chunk in flight — the
    discipline ``loom-probe`` established after a cancelled Actions job, which
    never runs its commit step, cost TOCSIN a three-hour backfill.

    ``done_chunks`` lets a re-run skip what a previous run already committed;
    ``resume_rows`` carries that run's rows back in.  Ordering is deliberate: the
    detail pulls come first because they are what the screen needs, and the census
    — a ~1.55-million-row transfer — comes last because it only adds a
    denominator to a result that is already on disk.
    """
    done = set(done_chunks)
    out: dict = {"chunks": {}, "rows": list(resume_rows or []), "census": {},
                 "verdict": "NO_DATA_REACHED"}
    fields = fetch_field_names(session=session)
    out["field_discovery"] = fields
    optional = list(DETAIL_OPTIONAL_FIELDS)
    if fields.get("available"):
        known = set(fields["available"])
        rejected = [f for f in optional if f not in known]
        optional = [f for f in optional if f in known]
        out["field_discovery"]["optional_dropped_before_request"] = rejected
    ask = [*DETAIL_CORE_FIELDS, *optional]

    for kind, _label in KINDS:
        for comp in NONGRAV_COMPONENTS:
            name = f"detail:{kind}:{comp}"
            if name in done:
                out["chunks"][name] = {"skipped": "already done in a previous run"}
                continue
            rec = fetch_detail(kind, comp, ask, timeout=timeout, session=session)
            rows = rec.pop("rows", [])
            out["rows"].extend(rows)
            # A field the server rejected stays rejected; re-requesting it in the
            # next chunk would spend a whole retry cycle rediscovering that.
            for f in rec.get("dropped_fields", []):
                if f in ask:
                    ask.remove(f)
            out["chunks"][name] = rec
            if rec.get("status") == 200:
                out["verdict"] = "OK"
            if on_result is not None:
                on_result(name, rec)

    if do_census:
        for kind, label in KINDS:
            name = f"census:{kind}"
            if name in done:
                out["census"][label] = {"skipped": "already done in a previous run"}
                continue
            rec = fetch_census(kind, timeout=census_timeout, session=session)
            rec.pop("rows", None)
            out["census"][label] = rec
            if on_result is not None:
                on_result(name, rec)
    out["n_rows"] = len(out["rows"])
    return out


# ---------------------------------------------------------------------------
# Orchestration.  Lives here rather than in `run.py` because `run.py` is not this
# module's to edit; the shape (checkpoint after every chunk, verdict never
# implied by an empty table) is copied from `run.calibrate` deliberately.
# ---------------------------------------------------------------------------
def run_catalogue(cfg=None, out_dir=None, do_census: bool = True,
                  max_tail: int = 400, write_objects_csv: bool = True) -> dict:
    """The full-catalogue momentum-ceiling screen, end to end.  Runner-only.

    Writes ``catalogue.json`` after every fetched chunk and again after the
    screen, plus ``catalogue_objects.csv`` — one row per screened object, so the
    distribution can be re-derived, re-binned or re-plotted without re-querying
    JPL.
    """
    from pathlib import Path

    from .run import _utc, _write_json, load_loom_config, thresholds_from_config

    conf = load_loom_config(cfg)
    th = thresholds_from_config(conf)
    root = Path(cfg.root) if cfg is not None else Path(__file__).resolve().parents[3]
    out = Path(out_dir) if out_dir else root / conf["report"]["results_dir"]
    path = out / "catalogue.json"

    rec: dict = {"screened_at_utc": _utc(), "verdict": "NOT_RUN",
                 "g_normalisation": g_normalisation(),
                 "thresholds": {"epsilon_realistic": th.epsilon_realistic,
                                "epsilon_hard": th.epsilon_hard,
                                "epsilon_inviolable": th.epsilon_inviolable,
                                "min_snr": MIN_A2_SNR,
                                "max_orbit_rms_arcsec": MAX_ORBIT_RMS_ARCSEC,
                                "min_data_arc_days": MIN_DATA_ARC_DAYS,
                                "min_obs_used": MIN_OBS_USED,
                                "max_condition_code": MAX_CONDITION_CODE},
                 "fetch": {}}

    def checkpoint() -> None:
        _write_json(path, rec)

    # Resume: a previous run's committed chunks are not re-fetched.
    done: list[str] = []
    resume: list[dict] = []
    if path.exists():
        try:
            prev = json.loads(path.read_text())
            done = list(prev.get("completed_chunks") or [])
            resume = list(prev.get("resume_rows") or [])
        except Exception as exc:                              # noqa: BLE001
            rec.setdefault("notes", []).append(
                f"could not read the previous {path.name} for resume: {exc!r}")
    rec["resumed_chunks"] = done
    checkpoint()

    progress: dict = {}
    rec["fetch"] = progress
    completed: list[str] = list(done)

    def _record(name, value):
        progress[name] = {k: v for k, v in value.items() if k != "rows"}
        completed.append(name)
        rec["completed_chunks"] = completed
        checkpoint()

    fetched = fetch_catalogue(on_result=_record, do_census=do_census,
                              resume_rows=resume, done_chunks=done)
    rows = fetched["rows"]
    rec["fetch"] = {k: v for k, v in fetched.items() if k not in ("rows",)}
    rec["n_rows_fetched"] = len(rows)
    rec["verdict"] = fetched["verdict"]
    checkpoint()

    if not rows:
        rec["note"] = ("JPL SBDB returned no rows for any of the six constrained "
                       "non-gravitational queries.  This is a DEAD FETCH, not a "
                       "null result: the two standing exceedances are neither "
                       "confirmed nor withdrawn by it")
        checkpoint()
        print(f"[loom] catalogue verdict={rec['verdict']} n_rows=0")
        return rec

    scr = screen_catalogue(rows, th=th, max_tail=max_tail)
    rec.update(scr.as_dict())
    rec["n_rows_fetched"] = len(rows)

    # The census, and the completeness check it makes possible.
    census_totals = {"n_rows": 0, "n_with_A1": 0, "n_with_A2": 0, "n_with_A3": 0,
                     "n_with_any": 0}
    census_keys: set[str] = set()
    for label, block in (fetched.get("census") or {}).items():
        counts = block.get("counts") or {}
        for k in ("n_rows", "n_with_A1", "n_with_A2", "n_with_A3", "n_with_any"):
            census_totals[k] += int(counts.get(k) or 0)
        census_keys |= set(counts.get("keys_with_any") or [])
        rec.setdefault("census", {})[label] = {
            k: v for k, v in {**block, "counts": {kk: vv for kk, vv in counts.items()
                                                  if kk != "keys_with_any"}}.items()}
    if census_totals["n_rows"]:
        census_totals["fraction_with_any"] = (census_totals["n_with_any"]
                                              / census_totals["n_rows"])
        rec.setdefault("census", {})["total"] = census_totals
        rec["completeness"] = completeness_check(
            {"keys_with_any": sorted(census_keys)},
            [_row_key(r) for r in rows])
    else:
        rec["completeness"] = completeness_check({"keys_with_any": []},
                                                 [_row_key(r) for r in rows])
    checkpoint()

    # What litcheck should be asked about.  Fed by name, exactly as
    # `run.litcheck(names=...)` expects, and deliberately NOT called "candidates":
    # every one is `interest` at best, because the momentum ceiling alone cannot
    # separate an engineered object from a dark comet nobody has published.
    rec["litcheck_input"] = [s["name"] for s in scr.survivors]
    rec["completed_chunks"] = completed
    # Rows are carried forward for a resume, but only the small detail pulls --
    # the census is counted in flight and its rows are never retained.
    rec["resume_rows"] = rows
    checkpoint()

    if write_objects_csv:
        _write_objects_csv(out / "catalogue_objects.csv", scr.entries)

    print(f"[loom] catalogue verdict={rec['verdict']} "
          f"n_screened={scr.n_unique} n_tail={len(scr.tail)} "
          f"n_survivors={len(scr.survivors)}")
    return rec


def _write_objects_csv(path, entries: Sequence[dict]) -> None:
    """One row per screened object: the distribution, re-derivable offline."""
    import csv

    cols = ["key", "name", "pdes", "kind", "class", "is_comet", "neo", "H",
            "diameter_m", "diameter_measured", "albedo", "a_au", "e", "i_deg",
            "q_au", "moid_au", "tisserand_j", "comet_like_dynamics",
            "A_magnitude_au_day2", "A_sigma_au_day2", "A_snr",
            "epsilon_1au", "epsilon_1au_rho_1000", "g_law_assumed",
            "epsilon_at_perihelion", "del_vigna_R", "a2_snr",
            "orbit_rms_arcsec", "data_arc_days", "n_obs_used", "condition_code",
            "reliable", "tier", "known_as", "standing_exceedance"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([*cols, "components_fitted", "fails"])
        for e in entries:
            row = []
            for c in cols:
                v = e.get(c)
                row.append("" if v is None or (isinstance(v, float) and not math.isfinite(v))
                           else v)
            w.writerow([*row, "|".join(e.get("components_fitted") or []),
                        "|".join(e.get("fails") or [])])

"""SEXTANT residuals: observed-minus-predicted minor-planet astrometry at Gaia precision.

LOOM's observable is the ephemeris residual of a *known* minor planet, decomposed
so that the along-track component --- the one a non-gravitational acceleration
displaces --- is separated from everything that has no directional preference.
Rubin serves that residual pre-computed, in arcsec, at a ~10 mas per-epoch
precision.  Gaia does not serve it at all: it serves the *ingredients*, at
milliarcsecond precision, for 46,264,083 observations of 156,823 objects
(``gaiafpr.sso_observation``) plus 23,336,467 of 158,152 (``gaiadr3``).

This module computes the residual from those ingredients.  It is where the
channel is won or lost, because at mas precision four things that Rubin's
arcsecond-scale product could hide become first-order:

1. **Circularity.**  Gaia's own SSO astrometry was used to fit Gaia's own orbit
   solutions, so a residual taken against those orbits is minimised by
   construction and measures nothing at all.  Handled twice over: the API
   refuses an orbit source it has not been told the provenance of
   (:class:`OrbitSource`, :func:`require_independent_prediction`), *and* the
   estimator is built to be invariant to whatever an orbit fit could have done
   --- see "Immunity by construction" below.
2. **The along-scan axis is the entire error model.**  Gaia's precision is
   strongly anisotropic; along-scan (AL) is the good axis and across-scan (AC)
   is roughly an order of magnitude worse.  ``position_angle_scan`` gives the
   direction.  Pooling the two axes would let AC noise masquerade as structure,
   so every residual here is projected onto (AL, AC) and the default fit uses
   **AL only**, one scalar equation per observation --- which is what Gaia's own
   astrometric solution does, and for the same reason.
3. **The astrometric chain is no longer optional.**  At arcsecond scale one can
   be sloppy about light time, stellar aberration and solar light deflection.
   At mas scale, light time is ~10 arcmin of motion, stellar aberration is
   ~20.5 arcsec, and gravitational deflection by the Sun is ~10 mas at Gaia's
   45-degree solar aspect angle.  All three are computed here, explicitly, and
   the *conventions* they imply about the archive columns are **measured from
   the data** rather than assumed (:func:`resolve_conventions`).
4. **The ephemeris has an error too, and it is smooth in time.**  A
   badly-determined orbit produces a residual that drifts, which is exactly what
   an acceleration produces.  The separation is that an orbit error lives in the
   six-dimensional space spanned by the partial derivatives of the predicted sky
   position with respect to the target's state, and a non-gravitational
   acceleration does not.  So the six partials are carried as *nuisance
   regressors* and marginalised, and what is reported is the component of the
   residual orthogonal to any possible orbit-element error.

Immunity by construction
------------------------
Point (4) is also the complete answer to point (1), and it is worth stating
plainly because it is the reason this channel can survive using a JPL orbit that
*does* contain Gaia astrometry in its fit (as every current JPL solution for a
numbered object does --- Gaia DR2 and DR3 SSO astrometry were delivered to the
MPC and are used, at high weight, in every modern fit).

An orbit fit can only change the prediction by changing the six state
components.  Whatever weight it gave the Gaia observations, the *only* thing it
could have absorbed out of the residual is a vector in the span of those six
partials.  Marginalising that span therefore removes exactly the part of the
signal a fit could have eaten, and what is left is untouched by the fit's
existence.  The cost is a loss of sensitivity, which is *reported* as
``absorbed_fraction`` rather than silently taken.

Two exceptions this does not cover, and both are enforced rather than trusted:

* an orbit whose fit **itself carried a non-gravitational parameter** (JPL fits
  ``A1``/``A2``/``A3`` for a small minority of objects) has had the signal
  itself fitted out, not merely the six elements.  ``OrbitSource`` records this
  and :func:`require_independent_prediction` refuses to treat such an object's
  residual as a blind measurement.
* Gaia's *own* orbit solutions (``gaiadr3.sso_orbits`` and successors) are
  fitted to precisely these observations with no other data.  That is
  ``CIRCULAR`` and is refused outright, in every mode, with no override.

What the residual actually is
-----------------------------
For observation ``i`` at barycentric-dynamical time ``t_i``, with Gaia at
barycentric position ``R_i`` and velocity ``V_i`` (both columns of the archive,
so no model of Gaia's orbit is needed anywhere in this module):

* solve the light time ``tau_i`` for ``|r_target(t_i - tau_i) - R_i| = c tau_i``;
* form the geometric direction ``u = (r_target(t_i - tau_i) - R_i)/|...|``;
* deflect it in the Sun's field (finite-distance formula, ~10 mas at Gaia's
  scanning geometry);
* aberrate it with ``V_i`` (special-relativistic form; the second-order term is
  ~2 mas and therefore not negligible);
* subtract from the observed direction in the local tangent plane, giving
  ``(d_east, d_north)`` in mas;
* rotate into ``(AL, AC)`` using ``position_angle_scan``.

The signal model is an along-track *physical* displacement ``S(t)`` in km,
mapped onto the sky by the object's own geometry --- so the observing geometry is
never left in the signal, exactly as LOOM insists.

The factor of three, which LOOM does not carry
-----------------------------------------------
``seti.loom.residuals.drift_fit`` converts a fitted quadratic to an acceleration
as ``a = 2*c2``, i.e. it reads the along-track displacement as the kinematic
``a t^2 / 2``.  That is not what a transverse force does to an orbit.  A
transverse acceleration ``a_T`` raises the semimajor axis (``da/dt = 2 a_T/n``),
which *lowers* the mean motion (``dn/dt = -3 a_T/a``), so the along-track
displacement is

    d^2 S / dt^2 = -3 a_T(t),      S(t) = -3 * A2 * IntInt g(r(t)) dt dt

--- three times larger than the kinematic reading and of the opposite sign.
(Checked against the orbit-averaged secular formula LOOM itself uses for
``da/dt``: for a circular orbit the two agree exactly.  See
:func:`along_track_displacement_basis`.)  This module uses the orbital form.
The discrepancy is recorded here rather than patched into LOOM, because LOOM's
number feeds a published calibration and changing it is a decision for the
channel's owner, not a side effect of this port.

Everything in this module is pure numpy and has no network dependency: the live
parts (the ephemeris, the orbit source, the binary catalogue) enter as inputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

# --- constants --------------------------------------------------------------
AU_KM = 1.495978707e8
AU_M = 1.495978707e11
DAY_S = 86400.0
#: Speed of light in au/day (IAU 2009 exact c over the exact au).
C_AU_PER_DAY = 173.1446326846693
MAS_PER_RAD = 648000.0 / math.pi * 1000.0          # 206264806.24709636
ARCSEC_PER_RAD = 648000.0 / math.pi
#: Schwarzschild radius of the Sun in au: 2 GM_sun / c^2.  The constant used by
#: ERFA's light-deflection routine, and the scale of the ~10 mas deflection Gaia
#: sees at its 45-degree solar aspect angle.
SUN_SCHWARZSCHILD_AU = 1.97412574336e-8
#: Heliocentric gravitational parameter in au^3/day^2 (Gauss's constant squared).
GM_SUN_AU3_DAY2 = 0.01720209895 ** 2
#: TCB - TDB linear rate and its origin (IAU 2006 Resolution B3).  ~17 s in 2015,
#: which at a main-belt sky rate of 30 arcsec/hour is ~0.14 arcsec --- 140 times
#: the signal this channel is looking for, so the time scale is not a detail.
L_B = 1.550519768e-8
TCB_TDB_ORIGIN_JD = 2443144.5003725


# ---------------------------------------------------------------------------
# 1. Orbit-source provenance: the independence requirement, made unbypassable
# ---------------------------------------------------------------------------
#: The three states an orbit source can be in with respect to *this* dataset.
INDEPENDENT = "INDEPENDENT"
PARTIAL_SELF_FIT = "PARTIAL_SELF_FIT"
CIRCULAR = "CIRCULAR"


class CircularOrbitSourceError(RuntimeError):
    """Raised when a residual would be taken against an orbit fitted to itself.

    This is not a defensive nicety.  A residual against Gaia's own SSO orbit
    solution is minimised by construction: the fit *chose* the elements that make
    it small.  Such a residual would have the right units, the right magnitude
    and no meaning whatsoever, and nothing downstream could tell.
    """


class UnknownProvenanceError(RuntimeError):
    """Raised when an orbit source's relationship to Gaia's astrometry is unstated.

    "I do not know whether this fit used the observations I am about to compare
    it against" is not a usable input.  It is refused rather than defaulted,
    because every plausible default is wrong for some source.
    """


@dataclass(frozen=True)
class OrbitSource:
    """Where a prediction came from, and what it was fitted to.

    Frozen, and required by every function in this module that produces a
    residual.  There is deliberately **no** default and no ``from_name`` shortcut
    that guesses: the only way to obtain a residual is to state the provenance,
    which is what makes the independence requirement impossible to violate by
    accident.

    Fields
    ------
    name, provider, solution_reference
        Identity, for the record written to disk.
    dynamical_model
        What integrated the orbit.  A two-body propagation of osculating
        elements is *not* good enough for a mas-scale residual --- planetary
        perturbations reach arcseconds over a six-year arc --- so anything other
        than a full-force model is refused unless ``allow_approximate`` is set,
        which exists for the synthetic tests and nowhere else.
    gaia_sso_astrometry_in_fit
        Whether the fit ingested Gaia SSO astrometry.  ``None`` means "not
        established", which raises: see :class:`UnknownProvenanceError`.
    nongrav_parameters_fitted
        Whether the fit carried ``A1``/``A2``/``A3`` (or an equivalent).  If it
        did *and* it saw the Gaia data, the signal itself has been fitted out
        and the residual is not a blind measurement --- it is a residual to
        somebody else's non-gravitational solution, which is a different and
        much weaker statement.
    fit_arc_end_utc
        The last observation in the fit.  An orbit whose arc ends before Gaia's
        first SSO epoch is independent as a matter of chronology, which is the
        cleanest kind of independence there is.
    """

    name: str
    provider: str
    dynamical_model: str
    solution_reference: str
    gaia_sso_astrometry_in_fit: bool | None
    nongrav_parameters_fitted: bool | None = None
    fit_arc_end_utc: str | None = None
    retrieved_utc: str | None = None
    notes: str = ""

    @property
    def independence(self) -> str:
        """``INDEPENDENT`` / ``PARTIAL_SELF_FIT`` / ``CIRCULAR``."""
        p = f"{self.provider} {self.name}".lower()
        if "gaia" in p and ("sso_orbit" in p or "gaia_solution" in p
                            or "gaiadr3.sso_orbits" in p):
            return CIRCULAR
        if self.gaia_sso_astrometry_in_fit is None:
            raise UnknownProvenanceError(
                f"orbit source {self.name!r} does not state whether Gaia SSO "
                f"astrometry entered its fit; establish it before using it as a "
                f"prediction (see docs/sextant.md)")
        return PARTIAL_SELF_FIT if self.gaia_sso_astrometry_in_fit else INDEPENDENT

    def as_dict(self) -> dict:
        d = {
            "name": self.name, "provider": self.provider,
            "dynamical_model": self.dynamical_model,
            "solution_reference": self.solution_reference,
            "gaia_sso_astrometry_in_fit": self.gaia_sso_astrometry_in_fit,
            "nongrav_parameters_fitted": self.nongrav_parameters_fitted,
            "fit_arc_end_utc": self.fit_arc_end_utc,
            "retrieved_utc": self.retrieved_utc, "notes": self.notes,
        }
        try:
            d["independence"] = self.independence
        except UnknownProvenanceError as exc:
            d["independence"] = "UNKNOWN"
            d["independence_error"] = str(exc)
        return d


#: Sources whose provenance has been established.  Kept in code, not config, so
#: the classification travels with the function that enforces it.  ``notes``
#: carries the evidence, because a provenance claim without a reason is a guess
#: with a dataclass around it.
KNOWN_ORBIT_SOURCES: dict[str, OrbitSource] = {
    "jpl_horizons": OrbitSource(
        name="jpl_horizons",
        provider="JPL Horizons (VECTORS, CENTER=@0)",
        dynamical_model="jpl_nbody_de44x_plus_perturbers",
        solution_reference="JPL SBDB small-body solution, current",
        gaia_sso_astrometry_in_fit=True,
        nongrav_parameters_fitted=None,
        fit_arc_end_utc=None,
        notes=("Gaia DR2 (2018) and DR3 (2022) SSO astrometry were delivered to "
               "the MPC and are used, at high weight, in current JPL fits.  So a "
               "current JPL orbit is PARTIAL_SELF_FIT, not independent, and the "
               "estimator's marginalisation over the six state partials is what "
               "makes the measurement survive that.  nongrav_parameters_fitted "
               "must be filled per object from the SBDB record: an object whose "
               "own fit carried A1/A2/A3 has had the signal removed."),
    ),
    "mpcorb_pre_gaia_snapshot": OrbitSource(
        name="mpcorb_pre_gaia_snapshot",
        provider="MPC MPCORB archival snapshot",
        dynamical_model="mpc_nbody",
        solution_reference="MPCORB.DAT snapshot dated before 2014-07",
        gaia_sso_astrometry_in_fit=False,
        nongrav_parameters_fitted=False,
        fit_arc_end_utc="2014-07-01",
        notes=("The gold path.  An orbit fitted before Gaia's first SSO epoch "
               "cannot contain Gaia data as a matter of chronology, so the "
               "residual is a genuine prediction.  The cost is a larger "
               "ephemeris uncertainty, which the nuisance marginalisation "
               "absorbs and which shows up as a larger absorbed_fraction."),
    ),
    "gaia_sso_orbits": OrbitSource(
        name="gaia_sso_orbits",
        provider="gaiadr3.sso_orbits (Gaia's own solution)",
        dynamical_model="gaia_dpac",
        solution_reference="Gaia DR3 SSO orbital solutions",
        gaia_sso_astrometry_in_fit=True,
        nongrav_parameters_fitted=False,
        notes=("CIRCULAR and refused in every mode.  Fitted to precisely these "
               "observations and to essentially nothing else, so the residual is "
               "minimised by construction and measures the fit, not the sky."),
    ),
}

#: Dynamical models good enough for a mas-scale prediction.
FULL_FORCE_MODELS = ("jpl_nbody_de44x_plus_perturbers", "mpc_nbody", "openorb_nbody",
                     "rebound_nbody", "assist_nbody")


def require_independent_prediction(source: OrbitSource, *,
                                   allow_partial: bool = False,
                                   allow_approximate: bool = False) -> str:
    """Gate an orbit source, returning its independence class.

    ``CIRCULAR`` always raises --- there is no flag that permits it.
    ``PARTIAL_SELF_FIT`` raises unless the caller says, in the call, that it
    accepts a partially self-fitted prediction; that acceptance is then stamped
    into the residual record so a reader of the output cannot fail to see it.
    """
    klass = source.independence                     # may raise UnknownProvenance
    if klass == CIRCULAR:
        raise CircularOrbitSourceError(
            f"orbit source {source.name!r} ({source.provider}) is Gaia's own "
            f"solution, fitted to the very observations the residual is taken "
            f"from; the residual is minimised by construction and measures "
            f"nothing.  There is no override for this.")
    if klass == PARTIAL_SELF_FIT and not allow_partial:
        raise CircularOrbitSourceError(
            f"orbit source {source.name!r} was fitted with Gaia SSO astrometry "
            f"included, so part of the signal has been absorbed into its "
            f"elements.  Pass allow_partial=True to proceed --- the estimator "
            f"marginalises the six state partials, which is exactly the "
            f"subspace an orbit fit can move in, and the residual sensitivity "
            f"lost is reported as absorbed_fraction.")
    if source.nongrav_parameters_fitted and source.gaia_sso_astrometry_in_fit:
        raise CircularOrbitSourceError(
            f"orbit source {source.name!r} carried fitted non-gravitational "
            f"parameters AND saw the Gaia astrometry, so the signal itself was "
            f"fitted out.  A residual against it is a residual to somebody "
            f"else's A2 solution, not a blind measurement; screen such objects "
            f"on the published A2 instead (seti.loom.nongrav).")
    if source.dynamical_model not in FULL_FORCE_MODELS and not allow_approximate:
        raise ValueError(
            f"dynamical model {source.dynamical_model!r} is not a full-force "
            f"integration; planetary perturbations reach arcseconds over a "
            f"six-year arc, which is a thousand times the signal.  "
            f"allow_approximate=True exists only for synthetic tests.")
    return klass


# ---------------------------------------------------------------------------
# 2. Time scales, and the epoch convention that could not be settled offline
# ---------------------------------------------------------------------------
# STATE OF KNOWLEDGE, SAID LOUDLY.  `gaiafpr.sso_observation` carries BOTH
# `epoch` and `epoch_utc`, and the sandbox has no egress, so the archive's own
# column documentation could not be read while this was written.  What `epoch`
# is measured in --- TCB (Gaia's mission time scale) or TDB --- and whether its
# zero point is JD 2455197.5 (2010-01-01, the Gaia reference epoch) or a plain
# Julian date, is therefore NOT SETTLED HERE.  It is a probe question, listed as
# such in docs/sextant.md.
#
# It is also, mercifully, a question the data answers by itself, because every
# candidate is wrong by SECONDS and a second of time is enormous at mas
# precision.  TCB - UTC in 2015 is about 84.8 s (67.184 s of TT - UTC plus
# ~17.6 s of TCB - TT); a main-belt object moves ~30 arcsec/hour, so 84.8 s of
# time-scale error is ~0.7 arcsec of along-track offset --- seven hundred times
# the signal, and PROPORTIONAL TO SKY RATE, which is a shape nothing else in the
# problem has.  `resolve_conventions` therefore measures the convention instead
# of assuming it, and `fit_common_time_offset` reports the residual timing error
# in seconds so a near-miss (a leap second, a 32.184 s TT-TAI) is legible as
# itself rather than as an acceleration.
#
# A wrong guess here would not fail; it would produce a large, coherent,
# rate-proportional along-track residual on every object in the sample, which is
# precisely what this channel would otherwise call a population-wide detection.
#: Leap seconds (TAI - UTC) at the start of each UTC interval, as (JD_UTC, secs).
#: Covers Gaia's SSO window with margin; anything outside raises rather than
#: silently extrapolating, because a missing leap second is a 1 s error and 1 s
#: is ~8 mas of along-track motion for a main-belt object.
LEAP_SECONDS: tuple[tuple[float, float], ...] = (
    (2453736.5, 33.0),   # 2006-01-01
    (2454832.5, 34.0),   # 2009-01-01
    (2456109.5, 35.0),   # 2012-07-01
    (2457204.5, 36.0),   # 2015-07-01
    (2457754.5, 37.0),   # 2017-01-01
)
LEAP_VALID_FROM = 2453736.5
#: No leap second has been inserted since 2017-01-01; the table is valid until
#: one is, and the guard below expresses that as a date rather than as trust.
LEAP_VALID_UNTIL = 2461041.5   # 2026-01-01, comfortably past Gaia's SSO window


def tai_minus_utc(jd_utc) -> np.ndarray:
    """TAI - UTC in seconds, from the pinned table; NaN outside its validity."""
    jd = np.asarray(jd_utc, dtype=float)
    out = np.full(jd.shape, np.nan)
    for start, secs in LEAP_SECONDS:
        out = np.where(jd >= start, secs, out)
    return np.where((jd >= LEAP_VALID_FROM) & (jd <= LEAP_VALID_UNTIL), out, np.nan)


def tdb_minus_tt_seconds(jd_tt) -> np.ndarray:
    """Leading periodic term of TDB - TT, in seconds (amplitude 1.657 ms).

    Included for completeness rather than necessity: 1.7 ms at a main-belt sky
    rate is 1.4e-5 arcsec, far below anything measurable here.  It is written
    down so that a reader can see it was considered and found negligible, which
    is not the same as it having been forgotten.
    """
    jd = np.asarray(jd_tt, dtype=float)
    g = np.radians(357.53 + 0.9856003 * (jd - 2451545.0))
    return 0.001657 * np.sin(g) + 0.000022 * np.sin(
        np.radians(246.11 + 0.90251792 * (jd - 2451545.0)))


@dataclass(frozen=True)
class EpochConvention:
    """One candidate reading of the archive's time column."""

    column: str
    scale: str            # "TCB" | "TDB" | "TT" | "UTC"
    jd_zero: float        # add to the column value to get a Julian date

    @property
    def label(self) -> str:
        return f"{self.column}:{self.scale}:{self.jd_zero:.1f}"


#: Gaia's mission reference epoch, 2010-01-01T00:00:00, as a Julian date.
GAIA_JD_ZERO = 2455197.5

#: Every reading of the two time columns that is plausible enough to be worth
#: testing.  The resolver picks between them by measurement; the list exists so
#: that "we guessed" can never be the answer.
CANDIDATE_EPOCH_CONVENTIONS: tuple[EpochConvention, ...] = (
    EpochConvention("epoch", "TCB", GAIA_JD_ZERO),
    EpochConvention("epoch", "TDB", GAIA_JD_ZERO),
    EpochConvention("epoch", "TT", GAIA_JD_ZERO),
    EpochConvention("epoch", "TCB", 0.0),
    EpochConvention("epoch", "TDB", 0.0),
    EpochConvention("epoch_utc", "UTC", GAIA_JD_ZERO),
    EpochConvention("epoch_utc", "UTC", 0.0),
    EpochConvention("epoch_utc", "TDB", GAIA_JD_ZERO),
)


def epoch_to_jd_tdb(values, convention: EpochConvention) -> np.ndarray:
    """Convert an archive time column to JD(TDB), the scale an ephemeris wants.

    TCB is the big one: it runs fast relative to TDB by 1.55e-8, which is ~17 s
    by 2015 and would appear downstream as a rate-proportional along-track
    residual of ~0.14 arcsec on a main-belt object.
    """
    jd = np.asarray(values, dtype=float) + float(convention.jd_zero)
    scale = convention.scale.upper()
    if scale == "TDB":
        return jd
    if scale == "TCB":
        # TDB = TCB - L_B (JD_TCB - T0) * 86400 s, expressed back in days.
        return jd - L_B * (jd - TCB_TDB_ORIGIN_JD)
    if scale == "TT":
        return jd + tdb_minus_tt_seconds(jd) / DAY_S
    if scale == "UTC":
        dtai = tai_minus_utc(jd)
        jd_tt = jd + (dtai + 32.184) / DAY_S
        return jd_tt + tdb_minus_tt_seconds(jd_tt) / DAY_S
    raise ValueError(f"unknown time scale {convention.scale!r}")


@dataclass(frozen=True)
class Conventions:
    """Every archive convention this module cannot read off a schema.

    ``resolved_by`` is the honest field: ``"ASSUMED_NOT_MEASURED"`` until
    :func:`resolve_conventions` has run on real data, and the residual record
    carries it so a result computed under an assumed convention is legible as
    such.  The defaults are the *most likely* reading, not a settled one.
    """

    epoch: EpochConvention = CANDIDATE_EPOCH_CONVENTIONS[0]
    apply_light_time: bool = True
    apply_stellar_aberration: bool = True
    apply_solar_deflection: bool = True
    #: True: position_angle_scan is measured from North towards East.
    scan_pa_north_to_east: bool = True
    resolved_by: str = "ASSUMED_NOT_MEASURED"
    resolution_margin: float = float("nan")
    notes: str = ""

    def as_dict(self) -> dict:
        return {"epoch": self.epoch.label,
                "apply_light_time": self.apply_light_time,
                "apply_stellar_aberration": self.apply_stellar_aberration,
                "apply_solar_deflection": self.apply_solar_deflection,
                "scan_pa_north_to_east": self.scan_pa_north_to_east,
                "resolved_by": self.resolved_by,
                "resolution_margin": self.resolution_margin,
                "notes": self.notes}


# ---------------------------------------------------------------------------
# 3. Spherical geometry
# ---------------------------------------------------------------------------
def unit_from_radec(ra_deg, dec_deg) -> np.ndarray:
    """Unit vectors (N, 3) in the equatorial frame from RA/Dec in degrees."""
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    return np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])


def radec_from_unit(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RA/Dec in degrees from unit vectors (N, 3)."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    ra = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(u[:, 2], -1.0, 1.0)))
    return ra, dec


def tangent_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local East and North unit vectors at each direction ``u``.

    East is ``d/d(RA)`` normalised, North is ``d/d(Dec)``.  Degenerate at the
    poles, where East is undefined; Gaia's SSO sample is ecliptic-confined so
    this is never reached, but the degeneracy propagates as NaN rather than as a
    quietly wrong basis.
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, u)
    norm = np.linalg.norm(east, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        east = np.where(norm[:, None] > 1e-12, east / norm[:, None], np.nan)
    north = np.cross(u, east)
    return east, north


def tangent_offset_mas(u_obs: np.ndarray, u_pred: np.ndarray) -> np.ndarray:
    """Observed-minus-predicted offset in the tangent plane at the prediction.

    Returned as (N, 2) = ``(delta_east, delta_north)`` in mas, with the
    ``cos(dec)`` factor already inside the East component by construction --- the
    projection onto a local orthonormal basis is the definition of a great-circle
    offset and carries no RA-wrapping trap at all, which is why it is done in
    vectors rather than in coordinate differences.
    """
    u_o = np.atleast_2d(np.asarray(u_obs, dtype=float))
    u_p = np.atleast_2d(np.asarray(u_pred, dtype=float))
    east, north = tangent_basis(u_p)
    d = u_o - u_p
    # Remove the (second-order, but not negligible at large offsets) radial part.
    d = d - np.sum(d * u_p, axis=1)[:, None] * u_p
    return np.column_stack([np.sum(d * east, axis=1) * MAS_PER_RAD,
                            np.sum(d * north, axis=1) * MAS_PER_RAD])


def project_to_tangent(vec: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Components of a 3-vector in the local (East, North) basis at ``u``."""
    v = np.atleast_2d(np.asarray(vec, dtype=float))
    east, north = tangent_basis(np.atleast_2d(np.asarray(u, dtype=float)))
    return np.column_stack([np.sum(v * east, axis=1), np.sum(v * north, axis=1)])


# ---------------------------------------------------------------------------
# 4. The astrometric chain: light time, deflection, aberration
# ---------------------------------------------------------------------------
def solve_light_time(target_state: Callable[[np.ndarray], np.ndarray],
                     jd_tdb: np.ndarray, r_obs_au: np.ndarray,
                     *, max_iter: int = 5, tol_days: float = 1e-11
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate the light time and return ``(tau_days, r_target, v_target)``.

    ``target_state(jd)`` must return an (N, 6) barycentric state in au and au/day
    --- the same frame and origin as the observer's ``x_gaia...vz_gaia`` columns.
    Mixing a heliocentric target with a barycentric observer is an 0.008 au error
    and would show up as a ~1 arcsec residual, so the frame is a documented
    precondition rather than an inference.

    Convergence is quadratic and three iterations are ample (the correction after
    one is ~1e-8 day); ``tol_days`` of 1e-11 is ~1 microsecond, at which the
    target has moved ~1e-11 au = 1.5 m, i.e. ~2e-6 mas.
    """
    t = np.asarray(jd_tdb, dtype=float)
    R = np.atleast_2d(np.asarray(r_obs_au, dtype=float))
    tau = np.zeros(t.shape, dtype=float)
    st = np.atleast_2d(np.asarray(target_state(t), dtype=float))
    for _ in range(int(max_iter)):
        st = np.atleast_2d(np.asarray(target_state(t - tau), dtype=float))
        rho = st[:, :3] - R
        dist = np.linalg.norm(rho, axis=1)
        new = dist / C_AU_PER_DAY
        if np.all(np.abs(new - tau) < tol_days):
            tau = new
            st = np.atleast_2d(np.asarray(target_state(t - tau), dtype=float))
            break
        tau = new
    return tau, st[:, :3], st[:, 3:6]


def light_deflection(u: np.ndarray, r_obs_from_sun: np.ndarray,
                     r_target_from_sun: np.ndarray,
                     bm: float = 1.0, dlim: float = 1e-6) -> np.ndarray:
    """Deflect a direction in the Sun's gravitational field (finite distance).

    The ERFA ``ld`` algorithm, which is the standard finite-distance form: for a
    source at a finite distance the deflection is reduced relative to the
    stellar case, and for Gaia's solar-system targets that reduction is a factor
    of order unity, not a factor that lets the term be dropped.

    Magnitude, because it decides whether this matters: 4.07 mas at 90 degrees
    solar elongation, rising as ``cot(theta/2)`` --- so ~9.8 mas at Gaia's
    45-degree solar aspect angle.  That is ten times the per-observation AL
    precision and it varies smoothly along the arc, which is exactly the shape
    an acceleration has.  Omitting it would be the single largest systematic in
    the channel.
    """
    p = np.atleast_2d(np.asarray(u, dtype=float))
    e_vec = np.atleast_2d(np.asarray(r_obs_from_sun, dtype=float))
    em = np.linalg.norm(e_vec, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_hat = e_vec / em[:, None]
    q_vec = np.atleast_2d(np.asarray(r_target_from_sun, dtype=float))
    qn = np.linalg.norm(q_vec, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_hat = q_vec / qn[:, None]
    qdqpe = np.sum(q_hat * (q_hat + e_hat), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = float(bm) * SUN_SCHWARZSCHILD_AU / em / np.maximum(qdqpe, float(dlim))
    pdq = np.sum(p * q_hat, axis=1)
    edp = np.sum(e_hat * p, axis=1)
    out = p + w[:, None] * (pdq[:, None] * e_hat - edp[:, None] * q_hat)
    n = np.linalg.norm(out, axis=1)
    return out / n[:, None]


def stellar_aberration(u: np.ndarray, v_obs_au_per_day: np.ndarray) -> np.ndarray:
    """Special-relativistic aberration of a direction by the observer's velocity.

    Gaia's barycentric speed is ~30 km/s, so ``|v|/c ~ 1e-4`` and the first-order
    displacement is ~20.5 arcsec --- twenty thousand times the signal.  The
    second-order term is ~2 mas, which is *also* above the noise floor, so the
    classical formula is not good enough and the relativistic one is used.
    """
    p = np.atleast_2d(np.asarray(u, dtype=float))
    v = np.atleast_2d(np.asarray(v_obs_au_per_day, dtype=float)) / C_AU_PER_DAY
    v2 = np.sum(v * v, axis=1)
    bm1 = np.sqrt(np.maximum(1.0 - v2, 0.0))          # 1/gamma
    pdv = np.sum(p * v, axis=1)
    w1 = 1.0 + pdv / (1.0 + bm1)
    out = p * bm1[:, None] + w1[:, None] * v
    n = np.linalg.norm(out, axis=1)
    return out / n[:, None]


def apparent_direction(target_state: Callable[[np.ndarray], np.ndarray],
                       jd_tdb: np.ndarray, r_obs_au: np.ndarray,
                       v_obs_au_per_day: np.ndarray,
                       *, conv: Conventions,
                       sun_state: Callable[[np.ndarray], np.ndarray] | None = None
                       ) -> dict:
    """The full predicted direction, with every step switchable and recorded.

    Returns a dict carrying the direction plus the geometry the downstream fit
    needs: the topocentric distance (which converts angle to kilometres), the
    heliocentric distance (which the ``g(r)`` law needs), the target's
    heliocentric velocity direction (the along-track basis), and the apparent
    sky rate (which converts a timing error into an angle).

    Each of the three corrections is a switch, and none of them is a switch
    because it is optional --- they are switches because *whether the archive's
    reduced positions already contain them* is a convention question this module
    resolves by measurement (:func:`resolve_conventions`), and a wrong answer is
    an arcsecond-scale error, not a subtle one.
    """
    t = np.asarray(jd_tdb, dtype=float)
    R = np.atleast_2d(np.asarray(r_obs_au, dtype=float))
    V = np.atleast_2d(np.asarray(v_obs_au_per_day, dtype=float))
    if conv.apply_light_time:
        tau, r_t, v_t = solve_light_time(target_state, t, R)
    else:
        st = np.atleast_2d(np.asarray(target_state(t), dtype=float))
        tau = np.zeros(t.shape)
        r_t, v_t = st[:, :3], st[:, 3:6]
    rho = r_t - R
    delta = np.linalg.norm(rho, axis=1)
    u = rho / delta[:, None]

    if sun_state is not None:
        sun = np.atleast_2d(np.asarray(sun_state(t), dtype=float))
        r_sun, v_sun = sun[:, :3], sun[:, 3:6]
    else:
        # The Sun's barycentric offset reaches ~0.008 au and its velocity
        # ~7.5e-6 au/day.  For the DIRECTION of the along-track basis that is a
        # 0.08% error and for g(r) a ~0.3% error; both are recorded rather than
        # hidden, and a caller with the Sun's state should pass it.
        r_sun = np.zeros_like(R)
        v_sun = np.zeros_like(V)
    r_helio = r_t - r_sun
    r_au = np.linalg.norm(r_helio, axis=1)
    v_helio = v_t - v_sun
    vn = np.linalg.norm(v_helio, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_hat = v_helio / vn[:, None]

    if conv.apply_solar_deflection:
        u = light_deflection(u, R - r_sun, r_helio)
    if conv.apply_stellar_aberration:
        u = stellar_aberration(u, V)

    # Apparent sky rate: needed only for the error budget (epoch_err) and for the
    # timing diagnostic, so the first-order kinematic form is ample.
    rel_v = v_t - V
    with np.errstate(divide="ignore", invalid="ignore"):
        dudt = (rel_v - np.sum(rel_v * u, axis=1)[:, None] * u) / delta[:, None]
    return {
        "u": u, "tau_days": tau, "delta_au": delta, "r_au": r_au,
        "v_hat_helio": v_hat, "rate_mas_per_day": dudt * MAS_PER_RAD,
        "r_target_bary": r_t, "v_target_bary": v_t,
    }


# ---------------------------------------------------------------------------
# 5. The scan frame: Gaia's error model IS the along-scan axis
# ---------------------------------------------------------------------------
# Gaia measures a transit as a one-dimensional time of crossing: along-scan (AL)
# is the good axis by construction, and across-scan (AC) is roughly an order of
# magnitude worse and, on the published Gaia-CRF3 minor-planet comparison,
# strongly non-Gaussian.  Pooling the two would let AC noise masquerade as
# structure, so this module keeps them apart everywhere and the default fit uses
# AL alone --- one scalar equation per observation, which is what Gaia's own
# astrometric solution does and for exactly this reason.
#
# `position_angle_scan` gives the direction.  Which way it is measured is a
# convention the schema states and the sandbox could not read, so it is VERIFIED
# AGAINST THE DATA by `verify_scan_convention`: the archive's own ra/dec
# covariance is anisotropic with its minor axis along-scan, so the angle between
# the covariance's minor eigenvector and the PA-implied direction is a free,
# assumption-free check that the convention is right.  A reflected convention
# does not merely flip a sign, it mixes AL into AC, so this check is not optional.
def scan_basis(position_angle_scan_deg, *, north_to_east: bool = True
               ) -> tuple[np.ndarray, np.ndarray]:
    """Along-scan and across-scan unit vectors in the local (East, North) basis.

    With ``north_to_east`` the position angle is measured from North towards
    East, so ``e_AL = (sin PA, cos PA)`` and ``e_AC = (cos PA, -sin PA)``.  The
    alternative handedness is provided because it is a real candidate, not
    because it is expected: :func:`verify_scan_convention` decides between them
    on data.
    """
    pa = np.radians(np.asarray(position_angle_scan_deg, dtype=float))
    s = np.sin(pa) if north_to_east else -np.sin(pa)
    c = np.cos(pa)
    e_al = np.column_stack([s, c])
    e_ac = np.column_stack([c, -s])
    return e_al, e_ac


def radec_covariance(ra_err, dec_err, ra_dec_corr) -> np.ndarray:
    """Assemble (N, 2, 2) covariances in (East, North) from mas errors + correlation.

    The archive's ``ra_error`` is already an error in ``ra*cos(dec)`` (the Gaia
    convention throughout), so it is the East component directly and no
    ``cos(dec)`` factor is applied here.  Applying one would be a silent
    ``sec(dec)`` inflation of every East error, and Gaia's SSO sample is
    ecliptic-confined, so it would be a smooth function of ecliptic longitude ---
    i.e. it would look like sky-coherent structure.
    """
    sa = np.asarray(ra_err, dtype=float)
    sd = np.asarray(dec_err, dtype=float)
    rho = np.asarray(ra_dec_corr, dtype=float)
    rho = np.clip(np.where(np.isfinite(rho), rho, 0.0), -1.0, 1.0)
    cov = np.empty((sa.size, 2, 2), dtype=float)
    cov[:, 0, 0] = sa * sa
    cov[:, 1, 1] = sd * sd
    cov[:, 0, 1] = cov[:, 1, 0] = rho * sa * sd
    return cov


def project_covariance(cov: np.ndarray, e_a: np.ndarray, e_b: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variances along ``e_a``/``e_b`` and their covariance, for (N, 2, 2) inputs."""
    cov = np.asarray(cov, dtype=float)

    def quad(x, y):
        return (x[:, 0] * (cov[:, 0, 0] * y[:, 0] + cov[:, 0, 1] * y[:, 1])
                + x[:, 1] * (cov[:, 1, 0] * y[:, 0] + cov[:, 1, 1] * y[:, 1]))

    return quad(e_a, e_a), quad(e_b, e_b), quad(e_a, e_b)


def verify_scan_convention(ra_err, dec_err, ra_dec_corr,
                           position_angle_scan_deg) -> dict:
    """Is ``position_angle_scan`` measured the way :func:`scan_basis` assumes?

    Free, assumption-free, and decisive.  Gaia's per-observation (ra, dec)
    covariance is strongly anisotropic with its *minor* axis along-scan, so the
    minor eigenvector of the covariance must coincide with the PA-implied AL
    direction.  Both handednesses are scored and the better one is reported with
    its margin.

    Both conventions scoring badly is itself a finding --- it would mean the
    error columns are not anisotropic in the way the mission's error model says,
    which would invalidate the whole AL/AC split --- so it is reported as
    ``NEITHER_CONVENTION_MATCHES`` rather than resolved to the lesser evil.
    """
    cov = radec_covariance(ra_err, dec_err, ra_dec_corr)
    good = np.all(np.isfinite(cov.reshape(cov.shape[0], 4)), axis=1)
    pa = np.asarray(position_angle_scan_deg, dtype=float)
    good &= np.isfinite(pa)
    out: dict = {"n": int(good.sum())}
    if good.sum() < 20:
        out["verdict"] = "TOO_FEW_OBSERVATIONS"
        return out
    c = cov[good]
    w, v = np.linalg.eigh(c)
    minor = v[:, :, 0]                       # eigenvector of the smallest eigenvalue
    out["median_axis_ratio"] = float(np.median(np.sqrt(
        np.maximum(w[:, 1], 0.0) / np.maximum(w[:, 0], 1e-30))))
    scores = {}
    for label, n2e in (("north_to_east", True), ("north_to_west", False)):
        e_al, _ = scan_basis(pa[good], north_to_east=n2e)
        # Axes, not vectors: |cos| removes the irrelevant 180-degree ambiguity.
        cos = np.abs(np.sum(e_al * minor, axis=1))
        scores[label] = float(np.median(np.degrees(np.arccos(np.clip(cos, 0, 1)))))
    out["median_misalignment_deg"] = scores
    best = min(scores, key=lambda k: scores[k])
    other = "north_to_west" if best == "north_to_east" else "north_to_east"
    out["preferred"] = best
    out["margin_deg"] = float(scores[other] - scores[best])
    if scores[best] > 15.0:
        out["verdict"] = "NEITHER_CONVENTION_MATCHES"
        out["note"] = ("the covariance minor axis does not align with either "
                       "reading of position_angle_scan; the AL/AC split cannot "
                       "be trusted and nothing downstream should be believed")
        return out
    out["verdict"] = "OK"
    out["north_to_east"] = best == "north_to_east"
    return out


# ---------------------------------------------------------------------------
# 6. Two-body propagation --- used ONLY for partial derivatives and for tests
# ---------------------------------------------------------------------------
# This is not the ephemeris.  A two-body propagation of osculating elements is
# arcseconds wrong over a six-year arc, which is a thousand times the signal, and
# `require_independent_prediction` refuses it as a dynamical model.  It is here
# for two jobs where only a *derivative* is needed and a per-cent error in the
# derivative is irrelevant:
#
#   * the six partial derivatives of predicted sky position with respect to the
#     target's state, which span the subspace any orbit fit can move in and are
#     therefore the nuisance basis this channel marginalises over;
#   * generating synthetic observations with a known injected acceleration, which
#     is the only honest way to test the estimator offline.
def _stumpff(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stumpff functions c2, c3, valid for elliptic, parabolic and hyperbolic psi."""
    psi = np.asarray(psi, dtype=float)
    c2 = np.empty_like(psi)
    c3 = np.empty_like(psi)
    pos = psi > 1e-8
    neg = psi < -1e-8
    mid = ~(pos | neg)
    if np.any(pos):
        s = np.sqrt(psi[pos])
        c2[pos] = (1.0 - np.cos(s)) / psi[pos]
        c3[pos] = (s - np.sin(s)) / s ** 3
    if np.any(neg):
        s = np.sqrt(-psi[neg])
        c2[neg] = (np.cosh(s) - 1.0) / (-psi[neg])
        c3[neg] = (np.sinh(s) - s) / s ** 3
    if np.any(mid):
        c2[mid] = 0.5 - psi[mid] / 24.0
        c3[mid] = 1.0 / 6.0 - psi[mid] / 120.0
    return c2, c3


def propagate_two_body(state0: np.ndarray, dt_days,
                       mu: float = GM_SUN_AU3_DAY2, *, max_iter: int = 60,
                       tol: float = 1e-12) -> np.ndarray:
    """Universal-variable Kepler propagation of one state to many epochs.

    ``state0`` is (6,) heliocentric position/velocity in au and au/day; returns
    (N, 6).  Universal variables rather than a mean-anomaly solve so the routine
    does not silently break on a hyperbolic or near-parabolic test case.
    """
    s0 = np.asarray(state0, dtype=float).reshape(6)
    dt = np.atleast_1d(np.asarray(dt_days, dtype=float))
    r0v, v0v = s0[:3], s0[3:]
    r0 = float(np.linalg.norm(r0v))
    v0 = float(np.linalg.norm(v0v))
    sqmu = math.sqrt(mu)
    rdotv = float(r0v @ v0v)
    alpha = 2.0 / r0 - v0 * v0 / mu                # 1/a
    chi = sqmu * dt * alpha if alpha > 1e-10 else np.sign(dt) * math.sqrt(1.0 / max(
        abs(alpha), 1e-12)) * np.ones_like(dt)
    chi = np.asarray(chi, dtype=float) * np.ones_like(dt)
    r = np.full(dt.shape, r0)
    c2 = np.full(dt.shape, 0.5)
    c3 = np.full(dt.shape, 1.0 / 6.0)
    for _ in range(int(max_iter)):
        psi = chi * chi * alpha
        c2, c3 = _stumpff(psi)
        r = (chi * chi * c2 + rdotv / sqmu * chi * (1.0 - psi * c3)
             + r0 * (1.0 - psi * c2))
        f = (rdotv / sqmu * chi * chi * c2 + (1.0 - alpha * r0) * chi ** 3 * c3
             + r0 * chi - sqmu * dt)
        step = -f / np.where(np.abs(r) > 1e-14, r, 1e-14)
        chi = chi + step
        if np.all(np.abs(step) < tol):
            psi = chi * chi * alpha
            c2, c3 = _stumpff(psi)
            r = (chi * chi * c2 + rdotv / sqmu * chi * (1.0 - psi * c3)
                 + r0 * (1.0 - psi * c2))
            break
    fq = 1.0 - chi * chi * c2 / r0
    gq = dt - chi ** 3 * c3 / sqmu
    gdot = 1.0 - chi * chi * c2 / r
    fdot = sqmu / (r * r0) * chi * (chi * chi * alpha * c3 - 1.0)
    pos = fq[:, None] * r0v[None, :] + gq[:, None] * v0v[None, :]
    vel = fdot[:, None] * r0v[None, :] + gdot[:, None] * v0v[None, :]
    return np.column_stack([pos, vel])


def state_partial_basis(state0: np.ndarray, jd0: float, jd_tdb: np.ndarray,
                        r_obs_au: np.ndarray, u_pred: np.ndarray,
                        e_al: np.ndarray, delta_au: np.ndarray,
                        *, mu: float = GM_SUN_AU3_DAY2,
                        dr: float = 1e-7, dv: float = 1e-9) -> np.ndarray:
    """Along-scan sky displacement per unit change in each of the six state components.

    Returns (N, 6) in mas per au (positions) and mas per au/day (velocities).

    **Why this is the load-bearing defence.**  An orbit fit has exactly six free
    parameters.  Whatever data it used and whatever weight it gave them, the only
    thing it could have removed from these residuals is a vector in the span of
    these six columns.  Carrying them as nuisance regressors therefore makes the
    non-gravitational estimate invariant to the fit's existence --- which is what
    lets a JPL orbit that *does* contain Gaia astrometry still be used, and what
    turns the circularity problem from an argument into a projection.

    Central differences of a two-body propagation: the partials themselves are
    smooth functions whose per-cent-level error changes the *shape* of the
    marginalised subspace negligibly, and using the real integrator would require
    re-running it twelve times per object for no measurable gain.
    """
    s0 = np.asarray(state0, dtype=float).reshape(6)
    t = np.asarray(jd_tdb, dtype=float)
    dt = t - float(jd0)
    R = np.atleast_2d(np.asarray(r_obs_au, dtype=float))
    u = np.atleast_2d(np.asarray(u_pred, dtype=float))
    east, north = tangent_basis(u)
    e = np.atleast_2d(np.asarray(e_al, dtype=float))
    dd = np.asarray(delta_au, dtype=float)
    cols = np.empty((t.size, 6), dtype=float)
    for k in range(6):
        h = dr if k < 3 else dv
        sp, sm = s0.copy(), s0.copy()
        sp[k] += h
        sm[k] -= h
        rp = propagate_two_body(sp, dt, mu=mu)[:, :3] - R
        rm = propagate_two_body(sm, dt, mu=mu)[:, :3] - R
        with np.errstate(divide="ignore", invalid="ignore"):
            dvec = (rp / np.linalg.norm(rp, axis=1)[:, None]
                    - rm / np.linalg.norm(rm, axis=1)[:, None]) / (2.0 * h)
        # Project the direction perturbation onto the local tangent plane, then
        # onto the scan axis.  `delta_au` is unused in the ratio but kept in the
        # signature so a caller cannot forget the geometry is per-observation.
        de = np.sum(dvec * east, axis=1) * MAS_PER_RAD
        dn = np.sum(dvec * north, axis=1) * MAS_PER_RAD
        cols[:, k] = de * e[:, 0] + dn * e[:, 1]
    _ = dd
    return cols


# ---------------------------------------------------------------------------
# 7. Ingestion and the residual table
# ---------------------------------------------------------------------------
#: Columns the acquisition stage is measured to deliver.  Named here so a schema
#: change is a KeyError with a column name in it, not a silent NaN column.
REQUIRED_COLUMNS = ("ra", "dec", "ra_error_random", "dec_error_random",
                    "position_angle_scan")
OPTIONAL_COLUMNS = ("ra_error_systematic", "dec_error_systematic",
                    "ra_dec_correlation_random", "ra_dec_correlation_systematic",
                    "epoch", "epoch_utc", "epoch_err", "astrometric_outcome_ccd",
                    "astrometric_outcome_transit", "is_rejected", "number_mp",
                    "denomination", "observation_id", "transit_id",
                    "x_gaia", "y_gaia", "z_gaia", "vx_gaia", "vy_gaia", "vz_gaia")


def as_columns(obs) -> dict[str, np.ndarray]:
    """Normalise a DataFrame / list of dicts / dict of arrays to a dict of arrays.

    Duck-typed on ``.columns`` so pandas is never imported here: this module has
    numpy as its only dependency, which is what lets the whole of it be exercised
    offline.
    """
    if hasattr(obs, "columns") and hasattr(obs, "__getitem__"):
        return {str(c): np.asarray(obs[c]) for c in obs.columns}
    if isinstance(obs, dict):
        return {str(k): np.atleast_1d(np.asarray(v)) for k, v in obs.items()}
    rows = list(obs)
    if not rows:
        return {}
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(str(k))
    out: dict[str, np.ndarray] = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        try:
            out[k] = np.array([float(v) if v is not None else np.nan for v in vals],
                              dtype=float)
        except (TypeError, ValueError):
            out[k] = np.array(vals, dtype=object)
    return out


def _col(cols: dict, name: str, n: int, default=np.nan) -> np.ndarray:
    v = cols.get(name)
    if v is None:
        return np.full(n, default, dtype=float)
    arr = np.asarray(v)
    if arr.dtype == object or arr.dtype.kind in "SU":
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError):
            return np.full(n, default, dtype=float)
    if arr.dtype == bool:
        return arr.astype(float)
    return arr.astype(float)


@dataclass
class ResidualSeries:
    """One object's observed-minus-predicted astrometry, in the scan frame.

    Every array is per observation and **nothing is dropped**: rejected and
    outcome-flagged rows are carried with their flags, because the rejection
    pattern is itself an observable (``seti.sextant.screen``) and a module that
    filtered them out at ingest would destroy the denominator that makes it
    measurable.
    """

    key: str = ""
    n: int = 0
    jd_tdb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    al_mas: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ac_mas: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_al_random: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_ac_random: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_al_systematic: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_ac_systematic: np.ndarray = field(default_factory=lambda: np.zeros(0))
    delta_au: np.ndarray = field(default_factory=lambda: np.zeros(0))
    r_au: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phase_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    elongation_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: mas of along-scan displacement per km of along-track physical displacement.
    sensitivity_mas_per_km: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: |projection of the along-track direction onto the scan axis|, 0 to 1.
    track_scan_projection: np.ndarray = field(default_factory=lambda: np.zeros(0))
    along_scan_rate_mas_per_day: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Projection onto the scan axis of the on-sky direction from the object
    #: towards the Sun.  Both competing GEOMETRIC explanations --- an
    #: illumination photocentre shift and residual solar light deflection ---
    #: act along this axis and along no other, which is what makes them
    #: separable from a force law rather than merely different in amplitude.
    sun_scan_projection: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: (n, 6) along-scan sky displacement per unit change in each component of
    #: the target's state --- the subspace ANY orbit fit can move in, computed
    #: at ingest while the full geometry is in hand and marginalised by every
    #: fit downstream.  Empty only if the reference state was unavailable.
    state_partials: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    #: Per force law, the along-scan design column in mas per (au/day^2):
    #: the exact linearised displacement response projected onto the sky and
    #: then onto the scan axis.  Computed at ingest because that is where the
    #: predicted directions and the observer state still exist.
    signal_columns: dict = field(default_factory=dict)
    transit_id: np.ndarray = field(default_factory=lambda: np.zeros(0))
    is_rejected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    outcome_ccd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    outcome_transit: np.ndarray = field(default_factory=lambda: np.zeros(0))
    state0: np.ndarray = field(default_factory=lambda: np.zeros(6))
    jd0: float = float("nan")
    orbit_source: dict = field(default_factory=dict)
    conventions: dict = field(default_factory=dict)
    independence: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def arc_days(self) -> float:
        t = self.jd_tdb[np.isfinite(self.jd_tdb)]
        return float(t.max() - t.min()) if t.size > 1 else float("nan")

    def usable(self, *, include_rejected: bool = False) -> np.ndarray:
        """Mask of observations the astrometric fit may use."""
        m = (np.isfinite(self.jd_tdb) & np.isfinite(self.al_mas)
             & np.isfinite(self.sigma_al_random) & (self.sigma_al_random > 0))
        if not include_rejected:
            m &= ~self.is_rejected
        return m

    def summary(self) -> dict:
        m = self.usable()
        return {"key": self.key, "n_observations": int(self.n),
                "n_usable": int(m.sum()),
                "n_rejected": int(np.count_nonzero(self.is_rejected)),
                "arc_days": self.arc_days,
                "n_transits": int(np.unique(self.transit_id[m]).size) if m.any() else 0,
                "median_sigma_al_mas": float(np.median(self.sigma_al_random[m]))
                if m.any() else float("nan"),
                "rms_al_mas": float(np.sqrt(np.mean(self.al_mas[m] ** 2)))
                if m.any() else float("nan"),
                "rms_ac_mas": float(np.sqrt(np.nanmean(self.ac_mas[m] ** 2)))
                if m.any() else float("nan"),
                "orbit_source": self.orbit_source,
                "independence": self.independence,
                "conventions": self.conventions,
                "notes": list(self.notes)}


def compute_residuals(obs, target_state: Callable[[np.ndarray], np.ndarray],
                      orbit_source: OrbitSource, *,
                      conv: Conventions | None = None,
                      state0: np.ndarray | None = None,
                      jd0: float | None = None,
                      sun_state: Callable[[np.ndarray], np.ndarray] | None = None,
                      allow_partial: bool = False,
                      allow_approximate: bool = False,
                      key: str = "") -> ResidualSeries:
    """The channel's observable: O-C in the scan frame, with its error model.

    ``orbit_source`` is required and is gated before anything is computed, so
    there is no code path that produces a residual without recording what the
    prediction was and whether it was independent of the data.
    """
    conv = conv or Conventions()
    klass = require_independent_prediction(
        orbit_source, allow_partial=allow_partial, allow_approximate=allow_approximate)
    cols = as_columns(obs)
    if not cols:
        return ResidualSeries(key=key, orbit_source=orbit_source.as_dict(),
                              conventions=conv.as_dict(), independence=klass,
                              notes=["no_observations"])
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise KeyError(f"observation table is missing required columns: {missing}")
    n = int(np.asarray(cols["ra"]).size)
    notes: list[str] = []

    jd = epoch_to_jd_tdb(_col(cols, conv.epoch.column, n), conv.epoch)
    if not np.any(np.isfinite(jd)):
        notes.append(f"epoch_column_{conv.epoch.label}_unusable")
    R = np.column_stack([_col(cols, c, n) for c in ("x_gaia", "y_gaia", "z_gaia")])
    V = np.column_stack([_col(cols, c, n) for c in ("vx_gaia", "vy_gaia", "vz_gaia")])
    if not np.any(np.isfinite(R)):
        raise KeyError("observer state (x_gaia..vz_gaia) is absent; this channel "
                       "uses Gaia's own state vector and has no model of Gaia's "
                       "orbit to fall back on")

    pred = apparent_direction(target_state, jd, R, V, conv=conv, sun_state=sun_state)
    u_pred = pred["u"]
    u_obs = unit_from_radec(_col(cols, "ra", n), _col(cols, "dec", n))
    d_en = tangent_offset_mas(u_obs, u_pred)

    pa = _col(cols, "position_angle_scan", n)
    e_al, e_ac = scan_basis(pa, north_to_east=conv.scan_pa_north_to_east)
    al = d_en[:, 0] * e_al[:, 0] + d_en[:, 1] * e_al[:, 1]
    ac = d_en[:, 0] * e_ac[:, 0] + d_en[:, 1] * e_ac[:, 1]

    cov_r = radec_covariance(_col(cols, "ra_error_random", n),
                             _col(cols, "dec_error_random", n),
                             _col(cols, "ra_dec_correlation_random", n, 0.0))
    cov_s = radec_covariance(_col(cols, "ra_error_systematic", n, 0.0),
                             _col(cols, "dec_error_systematic", n, 0.0),
                             _col(cols, "ra_dec_correlation_systematic", n, 0.0))
    var_al_r, var_ac_r, _ = project_covariance(cov_r, e_al, e_ac)
    var_al_s, var_ac_s, _ = project_covariance(cov_s, e_al, e_ac)

    # The time uncertainty is a POSITION uncertainty along the direction of
    # motion, so it belongs in the along-scan error budget with the geometry
    # applied, not as a separate caveat.  epoch_err of 1 ms at a main-belt sky
    # rate of ~720000 mas/day is ~0.008 mas: usually negligible, occasionally not.
    rate = pred["rate_mas_per_day"]
    rate_en = project_to_tangent(rate, u_pred)
    rate_al = rate_en[:, 0] * e_al[:, 0] + rate_en[:, 1] * e_al[:, 1]
    t_err = _col(cols, "epoch_err", n, 0.0)
    t_err = np.where(np.isfinite(t_err), t_err, 0.0)
    var_al_r = var_al_r + (rate_al * t_err) ** 2

    # Geometry the law-discrimination stage needs, computed once here because it
    # is free given the vectors already in hand.
    r_helio_t = pred["r_target_bary"]
    if sun_state is not None:
        sun = np.atleast_2d(np.asarray(sun_state(jd), dtype=float))[:, :3]
    else:
        sun = np.zeros_like(r_helio_t)
    r_obj = r_helio_t - sun
    r_obs_helio = R - sun
    rn = np.linalg.norm(r_obj, axis=1)
    dn = pred["delta_au"]
    on = np.linalg.norm(r_obs_helio, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_phase = (rn ** 2 + dn ** 2 - on ** 2) / (2.0 * rn * dn)
        cos_elong = (on ** 2 + dn ** 2 - rn ** 2) / (2.0 * on * dn)
    phase = np.degrees(np.arccos(np.clip(cos_phase, -1.0, 1.0)))
    elong = np.degrees(np.arccos(np.clip(cos_elong, -1.0, 1.0)))

    # mas of along-scan signal per km of along-track physical displacement:
    # the object's heliocentric velocity direction, projected onto the sky and
    # then onto the scan axis, divided by the topocentric distance.  This is what
    # keeps the observing geometry OUT of the fitted quantity.
    with np.errstate(divide="ignore", invalid="ignore"):
        sun_hat = -r_obj / rn[:, None]
    sun_en = project_to_tangent(sun_hat, u_pred)
    sun_scan = sun_en[:, 0] * e_al[:, 0] + sun_en[:, 1] * e_al[:, 1]

    v_hat_en = project_to_tangent(pred["v_hat_helio"], u_pred)
    track_dot_scan = v_hat_en[:, 0] * e_al[:, 0] + v_hat_en[:, 1] * e_al[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = MAS_PER_RAD / (dn * AU_KM) * track_dot_scan

    rej = cols.get("is_rejected")
    if rej is None:
        is_rej = np.zeros(n, dtype=bool)
        notes.append("is_rejected_absent__treated_as_all_retained")
    else:
        arr = np.asarray(rej)
        is_rej = (arr.astype(bool) if arr.dtype != object
                  else np.array([bool(x) for x in arr]))

    tid = _col(cols, "transit_id", n)
    if not np.any(np.isfinite(tid)):
        # Without transit ids the systematic cannot be blocked by transit and is
        # treated as independent per observation, which UNDERSTATES the drift
        # uncertainty.  Recorded, not silently assumed.
        tid = np.arange(n, dtype=float)
        notes.append("no_transit_id__systematic_treated_as_per_observation")

    if state0 is None or jd0 is None:
        # The state at the middle of the arc, taken from the ephemeris itself, is
        # the reference the nuisance partials are differenced about.
        finite = np.isfinite(jd)
        jd0 = float(np.median(jd[finite])) if finite.any() else float("nan")
        st0 = np.atleast_2d(np.asarray(target_state(np.array([jd0])), dtype=float))
        state0 = st0[0] - (np.concatenate([sun[0], np.zeros(3)])
                           if sun_state is not None else np.zeros(6))

    signal_columns: dict[str, np.ndarray] = {}
    try:
        s0r = np.asarray(state0, dtype=float).reshape(6)
        for _law in FORCE_LAWS:
            resp = variational_response(s0r, float(jd0), jd, law=_law)
            resp_en = project_to_tangent(resp, u_pred)
            with np.errstate(divide="ignore", invalid="ignore"):
                signal_columns[_law] = (
                    (resp_en[:, 0] * e_al[:, 0] + resp_en[:, 1] * e_al[:, 1])
                    / dn * MAS_PER_RAD)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        notes.append(f"signal_columns_unavailable__{type(exc).__name__}")

    try:
        partials = state_partial_basis(np.asarray(state0, dtype=float).reshape(6),
                                       float(jd0), jd, R, u_pred, e_al, dn)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        partials = np.zeros((n, 6))
        notes.append(f"state_partials_unavailable__{type(exc).__name__}")

    return ResidualSeries(
        key=key, n=n, jd_tdb=jd, al_mas=al, ac_mas=ac,
        sigma_al_random=np.sqrt(np.maximum(var_al_r, 0.0)),
        sigma_ac_random=np.sqrt(np.maximum(var_ac_r, 0.0)),
        sigma_al_systematic=np.sqrt(np.maximum(var_al_s, 0.0)),
        sigma_ac_systematic=np.sqrt(np.maximum(var_ac_s, 0.0)),
        delta_au=dn, r_au=rn, phase_deg=phase, elongation_deg=elong,
        sensitivity_mas_per_km=sens, track_scan_projection=np.abs(track_dot_scan),
        sun_scan_projection=sun_scan, state_partials=partials,
        signal_columns=signal_columns,
        along_scan_rate_mas_per_day=rate_al, transit_id=tid, is_rejected=is_rej,
        outcome_ccd=_col(cols, "astrometric_outcome_ccd", n),
        outcome_transit=_col(cols, "astrometric_outcome_transit", n),
        state0=np.asarray(state0, dtype=float).reshape(6), jd0=float(jd0),
        orbit_source=orbit_source.as_dict(), conventions=conv.as_dict(),
        independence=klass, notes=notes)


# ---------------------------------------------------------------------------
# 8. Competing models for the same residual series
# ---------------------------------------------------------------------------
# THE QUESTION THIS CHANNEL ASKS IS *WHICH LAW*, NOT *HOW MUCH FORCE*.
#
# Every published screen of this dataset assumes a functional form and fits its
# amplitude: Dziadura, Bartczak & Oszkiewicz (A&A 693, A31; arXiv:2411.09750)
# fit six elements plus a Yarkovsky A2 to 446 NEAs and 54,094 inner-belt and
# Mars-crossing objects on Gaia FPR and report no main-belt detection; Liberato
# et al. (arXiv:2605.22702) fit a binary photocentre wobble --- along-scan
# projection, an FPR-consistent noise model, a Monte-Carlo noise-only control and
# a period search --- and select 343 binary candidates.  Both ask "how big is the
# amplitude of the law I already chose".  Neither asks which law the residual
# actually prefers.
#
# The models below are therefore built to be compared against OBSERVING
# GEOMETRY, not merely against each other in amplitude:
#
#   force laws, which act on the orbit and appear DOUBLE-INTEGRATED in time
#     radiation    g(r) = (1 au / r)^2      Yarkovsky, SRP: recoil from sunlight
#     sublimation  JPL's water-ice g(r)     knee at 2.8 au; steeply falling
#     constant     g(r) = 1                 distance-independent: nothing natural
#
#   geometric explanations, which act on the MEASUREMENT and appear undifferenced
#     illumination  along the sunward axis, shape sin(phase)    photocentre shift
#     deflection    along the sunward axis, shape cot(elong/2)  unmodelled bending
#     timing        along the scan axis, shape = along-scan rate  clock/shutter
#
# The two families are separable *by shape*, not by size: an acceleration's
# signature grows quadratically in time and is indifferent to where the Sun is,
# while an illumination or deflection residue tracks the sunward direction on the
# sky and reverses with it, and a timing error is proportional to sky rate.  A
# station-keeping or trajectory-correcting object cannot mimic the geometric
# family by construction --- its acceleration does not know where the Sun is ---
# and that is the one discriminant amplitude cannot express.
G_COMET_ALPHA = 0.1112620426
G_COMET_R0 = 2.808
G_COMET_M = 2.15
G_COMET_N = 5.093
G_COMET_K = 4.6142


def g_sublimation(r_au) -> np.ndarray:
    """JPL's standard water-ice scaling ``g(r)``, normalised to 1 at 1 au."""
    r = np.asarray(r_au, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = r / G_COMET_R0
        return G_COMET_ALPHA * x ** (-G_COMET_M) * (1.0 + x ** G_COMET_N) ** (-G_COMET_K)


def g_radiation(r_au, d: float = 2.0) -> np.ndarray:
    """Radiation-driven scaling ``(1 au / r)^d``; ``d = 2`` for Yarkovsky and SRP."""
    r = np.asarray(r_au, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, r ** (-float(d)), np.nan)


def g_constant(r_au) -> np.ndarray:
    """A distance-independent acceleration: what nothing natural does."""
    r = np.asarray(r_au, dtype=float)
    return np.where(np.isfinite(r), 1.0, np.nan)


FORCE_LAWS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "radiation": g_radiation,
    "sublimation": g_sublimation,
    "constant": g_constant,
}

#: The factor of three, and its sign.  A transverse acceleration ``a_T`` raises
#: the semimajor axis (``da/dt = 2 a_T / n``), which lowers the mean motion
#: (``dn/dt = -3 a_T / a``), so the along-track displacement obeys
#: ``d2S/dt2 = -3 a_T`` --- three times the kinematic reading, with the opposite
#: sign.  Verified against the orbit-averaged secular formula for a circular
#: orbit in ``test_sextant_residuals.py``.
ALONG_TRACK_FEEDBACK = -3.0


def along_track_displacement_basis(jd_tdb: np.ndarray, r_au: np.ndarray,
                                   law: str = "radiation", *,
                                   t0: float | None = None,
                                   n_grid: int = 4096) -> np.ndarray:
    """Along-track displacement (au) per unit ``A2`` (au/day^2), per observation.

    ``-3 * IntInt g(r(t)) dt dt`` on a dense uniform grid, interpolated back to
    the observation epochs.  Double integration on the grid rather than a closed
    form because ``r(t)`` is whatever the real ephemeris says it is, including
    the eccentricity modulation that makes the sublimation and radiation laws
    separable in the first place.

    Over a Gaia SSO arc (2014-2020) a main-belt object completes one to three
    revolutions, so ``r`` genuinely varies and the laws genuinely separate ---
    which is the one thing LOOM could never get from a one-month Rubin baseline,
    where ``law_discrimination`` returned ``INSUFFICIENT_R_SPAN`` on every object.
    """
    t = np.asarray(jd_tdb, dtype=float)
    r = np.asarray(r_au, dtype=float)
    good = np.isfinite(t) & np.isfinite(r) & (r > 0)
    out = np.full(t.shape, np.nan)
    if good.sum() < 3:
        return out
    ts, rs = t[good], r[good]
    order = np.argsort(ts)
    ts, rs = ts[order], rs[order]
    ts_u, idx = np.unique(ts, return_index=True)
    rs_u = rs[idx]
    t_ref = float(t0) if t0 is not None else float(np.median(ts_u))
    lo, hi = float(ts_u.min()), float(ts_u.max())
    grid = np.linspace(lo, hi, int(n_grid))
    r_grid = np.interp(grid, ts_u, rs_u)
    g = np.asarray(FORCE_LAWS[law](r_grid), dtype=float)
    dt = np.diff(grid)
    # Trapezoidal cumulative integral, twice.
    v = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dt)])
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    # Re-reference so the basis vanishes (in value and slope) at t_ref: the
    # constant and linear parts belong to the nuisance basis, not to the signal,
    # and leaving them here would make the signal column partly degenerate with
    # an orbit-element error.
    s0 = float(np.interp(t_ref, grid, s))
    v0 = float(np.interp(t_ref, grid, v))
    basis = ALONG_TRACK_FEEDBACK * (s - s0 - v0 * (grid - t_ref))
    out[good] = np.interp(t[good], grid, basis)
    return out


def transverse_unit(state: np.ndarray) -> np.ndarray:
    """JPL's transverse direction: in the orbit plane, perpendicular to ``r``.

    ``A1``/``A2``/``A3`` are defined on the radial / transverse / normal triad,
    NOT on the velocity direction.  For a circular orbit the two coincide; for an
    eccentric one they differ by the flight-path angle, and using the velocity
    direction would misdefine the very parameter being reported.
    """
    st = np.atleast_2d(np.asarray(state, dtype=float))
    r, v = st[:, :3], st[:, 3:6]
    h = np.cross(r, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_hat = h / np.linalg.norm(h, axis=1)[:, None]
        r_hat = r / np.linalg.norm(r, axis=1)[:, None]
    return np.cross(h_hat, r_hat)


def variational_response(state0: np.ndarray, jd0: float, jd_eval: np.ndarray,
                         law: str = "radiation", *, mu: float = GM_SUN_AU3_DAY2,
                         step_days: float = 1.0) -> np.ndarray:
    """Exact linear displacement response (au, (N, 3)) per unit ``A2`` (au/day^2).

    THE SCALAR SECULAR FORMULA IS NOT GOOD ENOUGH, AND THIS IS WHY IT MATTERS.
    ``d2S/dt2 = -3 a_T`` is the along-track secular response, and it reproduces a
    direct numerical integration of a *circular* orbit to under two per cent.
    But the true displacement is a three-vector: it has a radial part, periodic
    along-track terms of order ``e``, and a normal part.  The residual is a
    projection of that whole vector onto the sky, so a scalar model of only the
    along-track part is an incomplete model --- and an incomplete model does not
    merely add noise once six orbit-error columns are marginalised, it BIASES the
    fitted amplitude, because the unmodelled part is absorbed asymmetrically.

    Measured on synthetic data with a known injected ``A2``: the scalar basis
    recovers the amplitude to 2% with no nuisance columns, and is 49% low once
    the six state partials are marginalised.  This function recovers it to a few
    per cent in both cases.  That is the difference between an estimator and a
    number.

    The equation integrated is the linearised (variational) form about the
    unperturbed two-body trajectory,

        d2(dr)/dt2 = -mu/r^3 dr + 3 mu (r.dr) r / r^5 + g(r) t_hat(t)

    with ``dr = 0`` and ``d(dr)/dt = 0`` at ``jd0``.  Those initial conditions are
    not a choice that has to be defended: changing them adds a homogeneous
    solution of the same equation, which is exactly a state perturbation, which
    is exactly what the six nuisance partials span --- so the fitted amplitude is
    invariant to them.  ``test_variational_basis_reference_epoch_invariance``
    pins that.
    """
    s0 = np.asarray(state0, dtype=float).reshape(6)
    t = np.asarray(jd_eval, dtype=float)
    good = np.isfinite(t)
    out = np.full((t.size, 3), np.nan)
    if not good.any():
        return out
    lo = min(float(np.min(t[good])), float(jd0))
    hi = max(float(np.max(t[good])), float(jd0))
    if hi <= lo:
        return np.zeros((t.size, 3))
    n_steps = max(int(math.ceil((hi - lo) / max(float(step_days), 1e-6))), 8)
    h = (hi - lo) / n_steps
    # Unperturbed reference trajectory on the grid AND its half-steps, so the
    # RK4 stages never need a fresh Kepler solve.
    grid = lo + h * np.arange(2 * n_steps + 1) * 0.5
    ref = propagate_two_body(s0, grid - float(jd0), mu=mu)
    r_ref = ref[:, :3]
    rn = np.linalg.norm(r_ref, axis=1)
    t_hat = transverse_unit(ref)
    g = np.asarray(FORCE_LAWS[law](rn), dtype=float)
    forcing = g[:, None] * t_hat

    def deriv(k: int, y: np.ndarray) -> np.ndarray:
        dr, dv = y[:3], y[3:]
        rv, rr = r_ref[k], rn[k]
        acc = (-mu / rr ** 3 * dr + 3.0 * mu * float(rv @ dr) / rr ** 5 * rv
               + forcing[k])
        return np.concatenate([dv, acc])

    # Integrate outward from jd0 in both directions so the reference epoch is
    # where the response vanishes, whichever end of the arc it sits in.
    i0 = int(round((float(jd0) - lo) / h))
    i0 = int(np.clip(i0, 0, n_steps))
    states = np.zeros((n_steps + 1, 6))
    for direction in (+1, -1):
        y = np.zeros(6)
        k = i0
        while 0 <= k + direction <= n_steps:
            kk = 2 * k
            step = direction * h
            k1 = deriv(kk, y)
            k2 = deriv(kk + direction, y + 0.5 * step * k1)
            k3 = deriv(kk + direction, y + 0.5 * step * k2)
            k4 = deriv(kk + 2 * direction, y + step * k3)
            y = y + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            k += direction
            states[k] = y
    node_t = lo + h * np.arange(n_steps + 1)
    for j in range(3):
        out[good, j] = np.interp(t[good], node_t, states[:, j])
    return out


def geometric_driver(series: ResidualSeries, name: str) -> np.ndarray:
    """A competing *measurement* explanation, as an along-scan design column (mas).

    Each returns the shape only; the amplitude is fitted, so the comparison is a
    model-selection question and not a threshold on a number nobody measured.
    """
    if name == "illumination":
        # An illuminated body's photocentre sits towards the Sun by an amount
        # that grows with phase angle; the displacement is along the sunward axis
        # on the sky and nowhere else.
        return series.sun_scan_projection * np.sin(np.radians(series.phase_deg))
    if name == "deflection":
        # Residual (mis-modelled or double-applied) solar light bending: same
        # axis, different shape --- cot(elongation/2), which diverges towards the
        # Sun where the illumination term is instead maximal in a different way.
        with np.errstate(divide="ignore", invalid="ignore"):
            half = np.radians(np.clip(series.elongation_deg, 1e-3, 180.0)) / 2.0
            return series.sun_scan_projection / np.tan(half)
    if name == "timing":
        # A clock or shutter error dt displaces every object by rate*dt along its
        # own direction of motion; in the scan frame that is the along-scan rate.
        return series.along_scan_rate_mas_per_day
    raise ValueError(f"unknown geometric driver {name!r}")


GEOMETRIC_DRIVERS = ("illumination", "deflection", "timing")


# ---------------------------------------------------------------------------
# 9. Generalised least squares with a block-correlated systematic
# ---------------------------------------------------------------------------
# The archive gives the random and systematic error terms SEPARATELY, and the
# distinction is not decoration.  A systematic that is shared inside a transit
# (nine CCD crossings behind one attitude solution and one calibration) does not
# average down across those nine, so adding it in quadrature per observation and
# then fitting as though the points were independent understates the uncertainty
# on any fitted drift by up to sqrt(9).  The covariance is therefore built with
# the systematic as a per-block random effect and inverted with Woodbury, which
# is exact and costs nothing.
#
# `correlation_scale` chooses the block:
#   "transit"  the nominal model: shared within a transit_id, independent across.
#   "time"     a pessimistic model: shared within `block_days` of wall clock, so
#              a slowly drifting calibration projects onto the linear and
#              quadratic terms rather than averaging away.  Reported alongside,
#              and the LARGER of the two uncertainties is the one the screen uses.
#   "none"     random only; diagnostic, never a result.
def _block_ids(series: ResidualSeries, mask: np.ndarray, scale: str,
               block_days: float) -> np.ndarray:
    if scale == "transit":
        return series.transit_id[mask]
    if scale == "time":
        t = series.jd_tdb[mask]
        return np.floor((t - np.nanmin(t)) / max(float(block_days), 1e-9))
    if scale == "none":
        return np.arange(int(mask.sum()), dtype=float)
    raise ValueError(f"unknown correlation scale {scale!r}")


def _gls_normal_equations(A: np.ndarray, y: np.ndarray, sig_r: np.ndarray,
                          sig_s: np.ndarray, blocks: np.ndarray):
    """Return ``(A^T V^-1 A, A^T V^-1 y, y^T V^-1 y)`` for a block-diagonal V.

    ``V_b = diag(sig_r^2) + s_b^2 * ones``.  Woodbury gives, for any u, v:
    ``u^T V_b^-1 v = sum(u v / sig_r^2) - s^2 (sum u/sig_r^2)(sum v/sig_r^2)
    / (1 + s^2 sum 1/sig_r^2)``.
    """
    k = A.shape[1]
    ata = np.zeros((k, k))
    aty = np.zeros(k)
    yty = 0.0
    for b in np.unique(blocks):
        sel = blocks == b
        w = 1.0 / (sig_r[sel] ** 2)
        s2 = float(np.mean(sig_s[sel])) ** 2
        Ab, yb = A[sel], y[sel]
        Aw = Ab * w[:, None]
        ata += Ab.T @ Aw
        aty += Aw.T @ yb
        yty += float(yb @ (yb * w))
        if s2 > 0:
            denom = 1.0 + s2 * float(np.sum(w))
            sa = Aw.sum(axis=0)
            sy = float(np.sum(yb * w))
            ata -= s2 * np.outer(sa, sa) / denom
            aty -= s2 * sa * sy / denom
            yty -= s2 * sy * sy / denom
    return ata, aty, yty


@dataclass
class ModelFit:
    """One model fitted to one residual series, with its degradation named."""

    name: str
    amplitude: float = float("nan")
    amplitude_err: float = float("nan")
    amplitude_err_pessimistic: float = float("nan")
    snr: float = float("nan")
    chi2: float = float("nan")
    chi2_nuisance_only: float = float("nan")
    delta_chi2: float = float("nan")
    dof: int = 0
    n: int = 0
    arc_days: float = float("nan")
    absorbed_fraction: float = float("nan")
    units: str = ""
    ok: bool = False
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def chi2_reduced(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def as_dict(self) -> dict:
        d = {"model": self.name, "amplitude": self.amplitude,
             "amplitude_err": self.amplitude_err,
             "amplitude_err_pessimistic": self.amplitude_err_pessimistic,
             "snr": self.snr, "chi2": self.chi2,
             "chi2_reduced": self.chi2_reduced,
             "chi2_nuisance_only": self.chi2_nuisance_only,
             "delta_chi2": self.delta_chi2, "dof": self.dof, "n": self.n,
             "arc_days": self.arc_days, "units": self.units,
             "absorbed_fraction": self.absorbed_fraction,
             "ok": self.ok, "reason": self.reason}
        d.update(self.detail)
        return d


def nuisance_design(series: ResidualSeries, mask: np.ndarray) -> np.ndarray:
    """The six state partials for the selected observations, as design columns.

    This is the subspace an orbit fit lives in.  Marginalising it is what makes
    the estimate invariant to the fact that Gaia's own astrometry is inside every
    modern JPL solution --- see the module docstring.  They are computed once, at
    ingest, where the observer state and the predicted directions are still in
    hand; recomputing them from the series alone would need geometry the series
    deliberately does not keep.

    An empty (n, 0) design is returned when the partials could not be built, and
    that is *not* silently equivalent to "no orbit error is possible": callers
    that care check ``series.state_partials.size`` and refuse.
    """
    m = np.asarray(mask, dtype=bool)
    P = np.asarray(series.state_partials, dtype=float)
    if P.size == 0 or P.shape[0] != series.n:
        return np.zeros((int(m.sum()), 0))
    out = P[m]
    return out[:, np.all(np.isfinite(out), axis=0)]


def fit_model(series: ResidualSeries, column: np.ndarray, *, name: str,
              units: str, nuisance: np.ndarray | None = None,
              include_rejected: bool = False,
              correlation_scale: str = "transit",
              block_days: float = 180.0,
              min_observations: int = 20,
              min_arc_days: float = 365.0) -> ModelFit:
    """Fit one model column against the along-scan residuals, marginalising nuisance.

    ``column`` is the model's along-scan shape in mas per unit amplitude;
    ``nuisance`` is a (n, k) design whose coefficients are fitted and thrown
    away.  The fitted amplitude is therefore the component of the residual
    ORTHOGONAL to every orbit-element error, which is the whole architecture.
    """
    m = series.usable(include_rejected=include_rejected) & np.isfinite(column)
    fit = ModelFit(name=name, units=units, n=int(m.sum()))
    if fit.n < int(min_observations):
        fit.reason = f"only_{fit.n}_usable_observations_below_{min_observations}"
        return fit
    t = series.jd_tdb[m]
    fit.arc_days = float(t.max() - t.min())
    if fit.arc_days < float(min_arc_days):
        fit.reason = (f"arc_{fit.arc_days:.0f}d_below_{min_arc_days:.0f}d__a_drift "
                      f"fitted_over_a_short_arc_is_an_extrapolation")
        return fit
    y = series.al_mas[m]
    sig_r = series.sigma_al_random[m]
    sig_s = series.sigma_al_systematic[m]
    sig_s = np.where(np.isfinite(sig_s), sig_s, 0.0)
    x = np.asarray(column, dtype=float)[m]

    N = nuisance if nuisance is not None else np.zeros((fit.n, 0))
    N = np.asarray(N, dtype=float)
    if N.shape[0] != fit.n:
        raise ValueError("nuisance design has the wrong number of rows")
    A_full = np.column_stack([N, x]) if N.size else x[:, None]
    # Column scaling is not cosmetic here.  The six state partials span ten
    # orders of magnitude between the position and velocity components, and the
    # model columns range from ~1e-9 (an acceleration basis) to ~1e6 (a sky rate
    # in mas/day).  Unscaled, the normal equations lose every significant digit
    # and return NEGATIVE chi-squared improvements for nested models -- which is
    # arithmetically impossible and was exactly what the first version did.
    scales = np.linalg.norm(A_full, axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    A_s = A_full / scales

    def _solve(design, blocks):
        ata, aty, yty = _gls_normal_equations(design, y, sig_r, sig_s, blocks)
        try:
            p_hat = np.linalg.solve(ata, aty)
        except np.linalg.LinAlgError:
            p_hat = np.linalg.pinv(ata) @ aty
        # chi2 = y'V^-1y - p'A'V^-1y for the exact GLS solution.  Written this
        # way rather than as the three-term expansion, which cancels catastrophi-
        # cally when the model explains most of the variance.
        return p_hat, np.linalg.pinv(ata), float(yty - float(p_hat @ aty))

    results = {}
    for scale_name in ("transit", "time"):
        eff = "none" if (scale_name == "transit" and correlation_scale == "none") \
            else scale_name
        blocks = _block_ids(series, m, eff, block_days)
        try:
            p, cov, chi2 = _solve(A_s, blocks)
            N_s = (N / np.linalg.norm(N, axis=0)) if N.size else np.ones((fit.n, 1))
            _, _, chi2_0 = _solve(N_s, blocks)
        except (np.linalg.LinAlgError, ValueError):
            fit.reason = "normal_equations_singular"
            return fit
        results[scale_name] = (p, cov, chi2, chi2_0)

    p, cov, chi2, chi2_0 = results["transit"]
    j = A_s.shape[1] - 1
    fit.amplitude = float(p[j]) / scales[j]
    var = float(cov[j, j]) / (scales[j] ** 2)
    fit.amplitude_err = math.sqrt(var) if var > 0 else float("nan")
    var_p = float(results["time"][1][j, j]) / (scales[j] ** 2)
    fit.amplitude_err_pessimistic = math.sqrt(var_p) if var_p > 0 else float("nan")
    fit.chi2 = float(chi2)
    fit.chi2_nuisance_only = float(chi2_0)
    fit.delta_chi2 = float(chi2_0 - chi2)
    fit.dof = int(fit.n - A_s.shape[1])
    # The honest error is the larger of the two correlation models, and the
    # signal-to-noise is quoted on it.  A result that survives only under the
    # optimistic model is not a result.
    err = max(fit.amplitude_err, fit.amplitude_err_pessimistic)
    # Rescale by the reduced chi-squared about the fitted model, never below 1:
    # the standard remedy for a common error-scale error, and the same guard
    # LOOM had to add after `scatter/sqrt(n)` inflated every S/N five-fold.
    scale = max(math.sqrt(fit.chi2_reduced) if fit.dof > 0
                and math.isfinite(fit.chi2_reduced) else 1.0, 1.0)
    fit.detail["error_scale_applied"] = scale
    err = err * scale
    fit.detail["amplitude_err_used"] = err
    fit.snr = abs(fit.amplitude) / err if err > 0 and math.isfinite(err) else float("nan")
    # How much of this model's signal the six state partials could have removed.
    fit.absorbed_fraction = absorbed_fraction(x, N, sig_r)
    fit.ok = True
    return fit


def absorbed_fraction(column: np.ndarray, nuisance: np.ndarray,
                      sigma: np.ndarray) -> float:
    """Fraction of a model column that lies in the span of the nuisance design.

    This is the sensitivity an orbit fit could have taken, whatever data it used
    --- it is the *quantitative* form of the circularity worry, and reporting it
    is what turns "the orbit might have absorbed the signal" from an objection
    into a number.  1.0 means the model is entirely degenerate with an orbit
    error and nothing can be measured; 0.0 means an orbit fit could not have
    touched it.
    """
    x = np.asarray(column, dtype=float)
    N = np.asarray(nuisance, dtype=float)
    if N.size == 0:
        return 0.0
    w = 1.0 / np.asarray(sigma, dtype=float)
    xw, Nw = x * w, N * w[:, None]
    denom = float(xw @ xw)
    if denom <= 0:
        return float("nan")
    coef, *_ = np.linalg.lstsq(Nw, xw, rcond=None)
    resid = xw - Nw @ coef
    return float(1.0 - (resid @ resid) / denom)


# ---------------------------------------------------------------------------
# 10. The discriminator: which law does this residual prefer?
# ---------------------------------------------------------------------------
def model_comparison(series: ResidualSeries, *,
                     include_rejected: bool = False,
                     correlation_scale: str = "transit",
                     block_days: float = 180.0,
                     min_observations: int = 20,
                     min_arc_days: float = 365.0,
                     min_delta_chi2: float = 16.0) -> dict:
    """Fit every competing model to one series and report which the data prefers.

    Six models, all marginalised over the same six-dimensional orbit-error
    subspace and all fitted with the same block-correlated error model, so the
    comparison is between *shapes* and not between different treatments:

    * three force laws --- radiation, sublimation, distance-independent --- whose
      along-scan signature is the double time-integral of ``g(r(t))`` and which
      therefore grow with the arc and are indifferent to where the Sun is;
    * three geometric explanations --- illumination photocentre, residual solar
      deflection, timing --- which track observing geometry and do not grow.

    The verdict names the best model and the margin over the second best.  A
    force law winning is a statement no amplitude fit can make; a geometric
    driver winning is a rejection with a named cause rather than a shrug.
    ``NO_MODEL_PREFERRED`` is a first-class outcome and by far the most common.
    """
    out: dict = {"key": series.key, "n": int(series.usable(
        include_rejected=include_rejected).sum()),
        "independence": series.independence,
        "orbit_source": series.orbit_source.get("name"),
        "conventions_resolved_by": series.conventions.get("resolved_by")}
    m = series.usable(include_rejected=include_rejected)
    if m.sum() < int(min_observations):
        out["verdict"] = "TOO_FEW_OBSERVATIONS"
        return out
    nuis = nuisance_design(series, m)
    out["n_nuisance_columns"] = int(nuis.shape[1])
    if nuis.shape[1] < 6:
        # Fewer than six means the orbit-error subspace is not fully removed and
        # an element error can leak into the signal.  Named, never assumed away.
        out["nuisance_incomplete"] = True

    fits: dict[str, dict] = {}
    for law in FORCE_LAWS:
        col = series.signal_columns.get(law)
        if col is None:
            # Fall back to the scalar secular basis, and SAY SO: it is biased low
            # once the nuisance is marginalised, so a result computed on it is a
            # lower bound and must be labelled one.
            basis = along_track_displacement_basis(series.jd_tdb, series.r_au,
                                                   law=law, t0=series.jd0)
            col = series.sensitivity_mas_per_km * AU_KM * basis
            out.setdefault("degraded", []).append(f"scalar_basis_used_for_{law}")
        f = fit_model(series, np.asarray(col, dtype=float),
                      name=f"force:{law}", units="au/day^2",
                      nuisance=nuis, include_rejected=include_rejected,
                      correlation_scale=correlation_scale, block_days=block_days,
                      min_observations=min_observations, min_arc_days=min_arc_days)
        fits[f.name] = f.as_dict()
    for drv in GEOMETRIC_DRIVERS:
        col = geometric_driver(series, drv)
        unit = {"illumination": "mas", "deflection": "mas",
                "timing": "days"}[drv]
        f = fit_model(series, col, name=f"geometry:{drv}", units=unit,
                      nuisance=nuis, include_rejected=include_rejected,
                      correlation_scale=correlation_scale, block_days=block_days,
                      min_observations=min_observations, min_arc_days=min_arc_days)
        if drv == "timing" and math.isfinite(f.amplitude):
            f.detail["dt_seconds"] = f.amplitude * DAY_S
            fits[f.name] = f.as_dict()
        else:
            fits[f.name] = f.as_dict()
    out["fits"] = fits

    usable = {k: v for k, v in fits.items()
              if v.get("ok") and math.isfinite(v.get("delta_chi2", float("nan")))}
    if len(usable) < 2:
        out["verdict"] = "MODELS_NOT_EVALUABLE"
        return out
    ranked = sorted(usable, key=lambda k: -usable[k]["delta_chi2"])
    best, second = ranked[0], ranked[1]
    out["best_model"] = best
    out["next_model"] = second
    out["delta_chi2_best"] = usable[best]["delta_chi2"]
    out["delta_chi2_margin"] = usable[best]["delta_chi2"] - usable[second]["delta_chi2"]
    out["best_family"] = best.split(":", 1)[0]
    out["best_snr"] = usable[best].get("snr")
    if usable[best]["delta_chi2"] < float(min_delta_chi2):
        out["verdict"] = "NO_MODEL_PREFERRED"
        out["note"] = (f"the best model improves chi-squared by only "
                       f"{usable[best]['delta_chi2']:.1f} over the orbit-error "
                       f"subspace alone; below {min_delta_chi2} that is not a "
                       f"preference, it is a fit")
        return out
    if out["delta_chi2_margin"] < float(min_delta_chi2) / 2.0:
        out["verdict"] = "MODELS_DEGENERATE"
        out["note"] = ("two models fit this arc equally well; the observing "
                       "geometry does not separate them and no law is preferred")
        return out
    out["verdict"] = ("FORCE_LAW_PREFERRED" if out["best_family"] == "force"
                      else "GEOMETRIC_EXPLANATION_PREFERRED")
    return out


# ---------------------------------------------------------------------------
# 11. Diagnostics that decide whether any of the above may be believed
# ---------------------------------------------------------------------------
def scan_axis_partition(series: ResidualSeries, *,
                        include_rejected: bool = False) -> dict:
    """Are the AL and AC residuals each consistent with their OWN error model?

    The two axes are never pooled, and this is why.  Across-scan is roughly an
    order of magnitude worse than along-scan and, on the published Gaia-CRF3
    minor-planet comparison, strongly non-Gaussian --- so an anomaly that appears
    equally in both axes in units of sigma is an error-model failure, not a
    displacement.  A real along-track displacement appears in AL and AC in the
    ratio their geometry dictates, which is what ``expected_ac_over_al`` states.
    """
    m = series.usable(include_rejected=include_rejected)
    out: dict = {"n": int(m.sum())}
    if m.sum() < 10:
        out["verdict"] = "TOO_FEW_OBSERVATIONS"
        return out
    al, ac = series.al_mas[m], series.ac_mas[m]
    sal = series.sigma_al_random[m]
    sac = series.sigma_ac_random[m]
    good_ac = np.isfinite(ac) & np.isfinite(sac) & (sac > 0)
    out["rms_al_mas"] = float(np.sqrt(np.mean(al ** 2)))
    out["median_sigma_al_mas"] = float(np.median(sal))
    out["chi_al"] = float(np.sqrt(np.mean((al / sal) ** 2)))
    if good_ac.sum() >= 10:
        out["rms_ac_mas"] = float(np.sqrt(np.mean(ac[good_ac] ** 2)))
        out["median_sigma_ac_mas"] = float(np.median(sac[good_ac]))
        out["chi_ac"] = float(np.sqrt(np.mean((ac[good_ac] / sac[good_ac]) ** 2)))
        out["sigma_ratio_ac_over_al"] = (out["median_sigma_ac_mas"]
                                         / out["median_sigma_al_mas"])
        out["chi_ratio_ac_over_al"] = out["chi_ac"] / out["chi_al"] \
            if out["chi_al"] > 0 else float("nan")
    else:
        out["note_ac"] = "across_scan_errors_absent_or_unusable"
    out["median_track_scan_projection"] = float(np.median(
        series.track_scan_projection[m]))
    out["verdict"] = "OK"
    return out


def fit_common_time_offset(series_list: Sequence[ResidualSeries]) -> dict:
    """One clock/shutter offset fitted across the whole sample, in seconds.

    A timing error puts every object off by ``rate * dt`` along its own direction
    of motion: a property of the *spacecraft* multiplied by a property of the
    *object*, so it is identifiable across a population even though it is
    perfectly degenerate with an acceleration within any single object.

    It doubles as the epoch-convention meter.  TCB read as TDB is ~17 s; TT read
    as UTC is 67.2 s; a missing leap second is 1 s.  A fitted ``dt`` landing on
    one of those numbers names the mistake instead of leaving it to be discovered
    as a population-wide detection.
    """
    xs, ys, ws = [], [], []
    for s in series_list:
        m = s.usable()
        if not m.any():
            continue
        xs.append(s.along_scan_rate_mas_per_day[m])
        ys.append(s.al_mas[m])
        ws.append(s.sigma_al_random[m])
    out: dict = {"n_objects": len(series_list)}
    if not xs:
        out["verdict"] = "NO_USABLE_OBSERVATIONS"
        return out
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    out["n"] = int(good.sum())
    if good.sum() < 100:
        out["verdict"] = "TOO_FEW_OBSERVATIONS"
        return out
    x, y, iv = x[good], y[good], 1.0 / w[good] ** 2
    denom = float(np.sum(iv * x * x))
    if denom <= 0:
        out["verdict"] = "RATE_IDENTICALLY_ZERO"
        return out
    slope = float(np.sum(iv * x * y)) / denom          # days
    resid = y - slope * x
    out["dt_seconds"] = slope * DAY_S
    out["dt_seconds_err"] = math.sqrt(1.0 / denom) * DAY_S
    ss_tot = float(np.sum(iv * y * y))
    out["variance_explained"] = (1.0 - float(np.sum(iv * resid * resid)) / ss_tot
                                 if ss_tot > 0 else float("nan"))
    known = {"leap_second": 1.0, "tt_minus_tai": 32.184, "tt_minus_utc_2015": 67.184,
             "tcb_minus_tt_2015": 17.6, "tcb_minus_utc_2015": 84.8}
    dt = abs(out["dt_seconds"])
    near = [k for k, v in known.items() if v > 0 and abs(dt - v) < 0.25 * v]
    out["matches_known_time_scale_error"] = near
    out["verdict"] = ("TIMING_OFFSET_DETECTED" if abs(out["dt_seconds"]) >
                      3.0 * out["dt_seconds_err"] else "NO_TIMING_OFFSET")
    return out


# ---------------------------------------------------------------------------
# 12. Resolving the conventions by measurement rather than by assumption
# ---------------------------------------------------------------------------
def resolve_conventions(obs, target_state: Callable[[np.ndarray], np.ndarray],
                        orbit_source: OrbitSource, *,
                        sun_state: Callable[[np.ndarray], np.ndarray] | None = None,
                        epoch_candidates: Sequence[EpochConvention] | None = None,
                        allow_partial: bool = True,
                        allow_approximate: bool = False,
                        max_rows: int = 4000) -> dict:
    """Measure the archive's conventions instead of guessing them.

    Runs the residual chain under every candidate reading of the time column and
    every on/off combination of the three astrometric corrections, and reports
    the one that minimises the robust along-scan residual.

    **This works because every wrong answer is enormous.**  A mis-read time scale
    is seconds, i.e. hundreds of mas of along-track motion; omitted stellar
    aberration is 20.5 arcsec; omitted light time is minutes of motion; omitted
    solar deflection is ~10 mas.  The correct combination lands at the mas level
    and every wrong one lands at least an order of magnitude above it, so this is
    a measurement with a margin of orders of magnitude, not a fit that could go
    either way.  ``margin`` reports that ratio, and a margin below 3 is returned
    as ``AMBIGUOUS`` rather than resolved --- if two conventions really do fit
    equally well then something is wrong with the premise, not with the winner.

    The scan-angle handedness is resolved separately and independently by
    :func:`verify_scan_convention`, which uses the archive's own covariance and
    needs no ephemeris at all; it is folded into the returned ``Conventions``.
    """
    cols = as_columns(obs)
    n = int(np.asarray(cols["ra"]).size)
    if n > int(max_rows):
        step = int(math.ceil(n / float(max_rows)))
        cols = {k: (v[::step] if isinstance(v, np.ndarray) and v.size == n else v)
                for k, v in cols.items()}
    scan = verify_scan_convention(_col(cols, "ra_error_random", n),
                                  _col(cols, "dec_error_random", n),
                                  _col(cols, "ra_dec_correlation_random", n, 0.0),
                                  _col(cols, "position_angle_scan", n))
    north_to_east = bool(scan.get("north_to_east", True))

    candidates = tuple(epoch_candidates or CANDIDATE_EPOCH_CONVENTIONS)
    trials: list[dict] = []
    for ep in candidates:
        if ep.column not in cols:
            continue
        for lt in (True, False):
            for ab in (True, False):
                for df in (True, False):
                    conv = Conventions(epoch=ep, apply_light_time=lt,
                                       apply_stellar_aberration=ab,
                                       apply_solar_deflection=df,
                                       scan_pa_north_to_east=north_to_east)
                    try:
                        s = compute_residuals(
                            cols, target_state, orbit_source, conv=conv,
                            sun_state=sun_state, allow_partial=allow_partial,
                            allow_approximate=allow_approximate)
                    except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
                        trials.append({"convention": conv.as_dict(),
                                       "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    m = s.usable()
                    if m.sum() < 10:
                        continue
                    # Median absolute residual: robust to the tail the published
                    # Gaia-CRF3 comparison shows is emphatically not Gaussian.
                    score = float(np.median(np.abs(s.al_mas[m])))
                    trials.append({"convention": conv.as_dict(), "median_abs_al_mas":
                                   score, "n": int(m.sum())})
    scored = [t for t in trials if math.isfinite(t.get("median_abs_al_mas", np.nan))]
    out: dict = {"n_trials": len(trials), "scan_convention": scan,
                 "trials": sorted(scored, key=lambda t: t["median_abs_al_mas"])[:8]}
    if len(scored) < 2:
        out["verdict"] = "NOT_ENOUGH_TRIALS"
        out["conventions"] = Conventions(
            scan_pa_north_to_east=north_to_east).as_dict()
        return out
    ordered = sorted(scored, key=lambda t: t["median_abs_al_mas"])
    best, second = ordered[0], ordered[1]
    margin = (second["median_abs_al_mas"] / best["median_abs_al_mas"]
              if best["median_abs_al_mas"] > 0 else float("inf"))
    out["best_median_abs_al_mas"] = best["median_abs_al_mas"]
    out["margin"] = margin
    ep_label = best["convention"]["epoch"]
    col, scale, zero = ep_label.split(":")
    conv = Conventions(
        epoch=EpochConvention(col, scale, float(zero)),
        apply_light_time=best["convention"]["apply_light_time"],
        apply_stellar_aberration=best["convention"]["apply_stellar_aberration"],
        apply_solar_deflection=best["convention"]["apply_solar_deflection"],
        scan_pa_north_to_east=north_to_east,
        resolved_by="MEASURED_ON_DATA", resolution_margin=float(margin))
    if margin < 3.0:
        out["verdict"] = "AMBIGUOUS"
        out["note"] = ("two convention combinations fit within a factor of three "
                       "of each other; every wrong reading here should be an "
                       "order of magnitude worse, so the premise is wrong "
                       "somewhere and no convention is adopted")
        out["conventions"] = Conventions(
            scan_pa_north_to_east=north_to_east,
            resolved_by="AMBIGUOUS_NOT_ADOPTED").as_dict()
        return out
    out["verdict"] = "RESOLVED"
    out["conventions"] = conv.as_dict()
    out["resolved"] = conv
    return out

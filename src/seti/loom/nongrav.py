"""The anomaly boundary: which non-gravitational accelerations are unphysical.

This module is the scientific core of LOOM, and the channel is worthless without
it.  Every solar-system body has non-gravitational acceleration — the Yarkovsky
effect (anisotropic thermal re-radiation) and solar radiation pressure act on all
of them.  So "shows non-gravitational acceleration" is not a technosignature; it
is a description of asteroids.  Worse, "shows an acceleration *larger than
Yarkovsky predicts*" is not a technosignature either: Seligman et al. (2023, PSJ
4, 35) already identified seven inactive small bodies with exactly that property
— the "dark comets" — and the accepted reading is hidden outgassing.  A channel
that flags large ``|A2|`` rediscovers dark comets.

So the gate here is not a magnitude.  It is a **momentum budget**, which is a
theorem rather than a fit, and the discriminants that follow it are about
*direction* and *time structure*, where the dark-comet explanation makes
predictions that an engineered object does not share.

The radiation momentum ceiling
------------------------------
Yarkovsky acceleration is recoil from re-radiating absorbed sunlight.  The
thermal photons therefore cannot carry away more momentum than the intercepted
beam delivered, so for a body of area-to-mass ratio ``AMR`` at heliocentric
distance ``r``::

    |a_NG| <= epsilon * (Phi_1au / c) * AMR * (1 au / r)^2

with ``Phi_1au / c = 4.5398e-6 N/m^2``.  ``epsilon = 1`` means every absorbed
photon's momentum is re-emitted in a single direction — physically unreachable;
``epsilon = 2`` is the specular-reflection limit and is inviolable for any
radiation-driven process whatsoever.  Calibrated against three objects with
independently measured ``A2`` (Bennu, 2005 ES70, 2009 BD) the *realised* value is
``epsilon_eff = 0.02–0.08``, so ``epsilon = 0.1`` is already a generous envelope
for real thermal recoil (see :func:`calibration_table`, which the test suite
checks against those measurements).

An object above the ``epsilon = 1`` curve for its size cannot be driven by
sunlight at all.  It requires either mass loss — outgassing, which is the dark
comet reading and is testable, see below — or a bulk density no rock has.

Area-to-mass ratio: the cleanest artificiality discriminant there is
--------------------------------------------------------------------
Radiation pressure enters an orbit fit as ``beta``, the ratio of radiation force
to solar gravity, and MPC reports it as an area-to-mass ratio.  With
``Phi/c = 4.5398e-6 N/m^2`` and ``GM/r^2 = 5.9301e-3 m/s^2`` at 1 au,

    beta = 7.656e-4 * C_R * AMR[m^2/kg]

and a sphere has ``AMR = 3 / (2 rho D)``.  The natural and artificial
populations are cleanly separated in this one number:

===========================  ====================  =============
object                       AMR (m^2/kg)          implied rho*D
===========================  ====================  =============
2009 BD (natural, H~28.4)    (2.97 +- 0.33)e-4     5050 kg/m^2
2011 MD (natural, D~6 m)     ~2.3-3.9e-4           2100-3600
J002E3 (Apollo 12 S-IVB)     7.9e-3                190
WT1190F (lunar-origin body)  (1.18 +- 0.05)e-2     127
1I/'Oumuamua (if pure SRP)   1.08                  1.4
===========================  ====================  =============

Identified artificial debris sits 20-40x above the natural small-body locus, and
``rho*D ~ 130-190 kg/m^2`` implies ``rho <~ 100 kg/m^3`` for a metre-scale body:
diagnostic of a hollow shell, impossible for rock.  This is how J002E3 and
WT1190F were classified, and it is the discriminant LOOM leans on hardest,
because — unlike acceleration magnitude — outgassing does *not* reproduce it.
Mass loss makes an object accelerate; it does not make it a thin shell.

Units in the ALeRCE mirror
--------------------------
Verified against the LSST v11.1 ``mpc_orbits`` Avro schema:
``lsst_mpc_orbits.yarkovsky`` is in units of ``1e-10 au/day^2`` (so Bennu's
``A2 = -4.62e-14`` appears as ``-4.6e-4``), and ``srp`` is in ``m^2/ton``
(1 m^2/ton = 1e-3 m^2/kg).  The ``a1``/``a2``/``a3`` columns are *also*
documented ``m^2/ton``, which is dimensionally wrong for Marsden accelerations;
those three are therefore treated as unit-unverified and are calibrated against
objects with published JPL solutions before use, never trusted raw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- physical constants, SI -------------------------------------------------
AU_M = 1.495978707e11
DAY_S = 86400.0
# Solar radiation momentum flux at 1 au, Phi/c.  N/m^2.
PHI_OVER_C_1AU = 4.5398e-6
# Solar gravitational acceleration at 1 au, GM/r^2.  m/s^2.
GM_OVER_R2_1AU = 5.9301e-3
# beta = (Phi/c) * C_R * AMR / (GM/r^2); the r^-2 cancels, so beta is distance
# independent, which is why it is the parameter an orbit fit can actually carry.
BETA_PER_CR_AMR = PHI_OVER_C_1AU / GM_OVER_R2_1AU        # 7.656e-4 kg/m^2

# m/s^2 -> au/day^2.
SI_TO_AU_PER_DAY2 = DAY_S ** 2 / AU_M                     # 0.04990

# `lsst_mpc_orbits.yarkovsky` is quoted in 1e-10 au/day^2 (schema-documented).
YARKOVSKY_COL_UNIT = 1.0e-10
# `lsst_mpc_orbits.srp` is quoted in m^2/ton = 1e-3 m^2/kg (schema-documented).
SRP_COL_UNIT = 1.0e-3

# --- the three ceilings -----------------------------------------------------
# Realised thermal-recoil efficiency, measured on objects with independent A2
# (see calibration_table): 0.02 - 0.08.  0.1 is therefore already generous for
# any real Yarkovsky effect.
EPSILON_REALISTIC = 0.1
# All absorbed momentum re-emitted in one direction.  Unreachable in practice.
EPSILON_HARD = 1.0
# Perfect specular reflection: the absolute limit for ANY radiation-driven
# process.  Above this, sunlight is not the cause, whatever the object is.
EPSILON_INVIOLABLE = 2.0

# Assumed bulk density and albedo when only H is known.  Both are chosen to make
# the ceiling GENEROUS (i.e. hard to exceed, so a flag means something): a low
# density and a high albedo both shrink the implied mass per unit area and so
# raise the permitted acceleration.
RHO_GENEROUS_KG_M3 = 1000.0
ALBEDO_GENEROUS = 0.25
# A middling pair, for reporting the realistic ratio alongside the generous one.
RHO_TYPICAL_KG_M3 = 2000.0
ALBEDO_TYPICAL = 0.15

# Area-to-mass ratio (m^2/kg) above which no solid body of plausible size works.
# The natural small-NEA locus is ~3e-4; identified artificial objects are
# 8e-3 - 1.2e-2.  1e-3 is the midpoint of the gap in log space and is used only
# as a reporting label, never as the gate (the gate is AMR vs the object's own
# size, which is strictly stronger).
AMR_ARTIFICIAL_FLOOR = 1.0e-3


# ---------------------------------------------------------------------------
# Size, area-to-mass ratio, beta
# ---------------------------------------------------------------------------
def diameter_m_from_h(h, albedo: float = ALBEDO_TYPICAL) -> np.ndarray:
    """Diameter (metres) from absolute magnitude ``H`` and geometric albedo.

    ``D = 1329 / sqrt(p) * 10^(-H/5)`` km, the standard relation.  A *higher*
    albedo gives a *smaller* body and hence a larger area-to-mass ratio, so the
    albedo assumption is not neutral: which direction is conservative depends on
    what the number is used for, and every caller here states its choice.
    """
    h = np.asarray(h, dtype=float)
    return 1329.0e3 / math.sqrt(float(albedo)) * 10.0 ** (-h / 5.0)


def amr_sphere(rho_kg_m3, diameter_m) -> np.ndarray:
    """Area-to-mass ratio (m^2/kg) of a solid sphere: ``3 / (2 rho D)``.

    Cross-section ``pi D^2 / 4`` over mass ``rho pi D^3 / 6``.
    """
    rho = np.asarray(rho_kg_m3, dtype=float)
    d = np.asarray(diameter_m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 3.0 / (2.0 * rho * d)


def beta_from_amr(amr, c_r: float = 1.5) -> np.ndarray:
    """Radiation-pressure parameter ``beta`` from area-to-mass ratio (m^2/kg)."""
    return BETA_PER_CR_AMR * float(c_r) * np.asarray(amr, dtype=float)


def amr_from_beta(beta, c_r: float = 1.5) -> np.ndarray:
    """Area-to-mass ratio (m^2/kg) from ``beta``."""
    return np.asarray(beta, dtype=float) / (BETA_PER_CR_AMR * float(c_r))


def amr_from_srp_column(srp_col) -> np.ndarray:
    """AMR in m^2/kg from ``lsst_mpc_orbits.srp``, which is in m^2/ton."""
    return np.asarray(srp_col, dtype=float) * SRP_COL_UNIT


def a2_from_yarkovsky_column(yark_col) -> np.ndarray:
    """``A2`` in au/day^2 from ``lsst_mpc_orbits.yarkovsky`` (1e-10 au/day^2)."""
    return np.asarray(yark_col, dtype=float) * YARKOVSKY_COL_UNIT


# ---------------------------------------------------------------------------
# The momentum ceiling
# ---------------------------------------------------------------------------
def momentum_ceiling_si(amr, r_au=1.0, epsilon: float = EPSILON_HARD) -> np.ndarray:
    """Largest radiation-driven acceleration (m/s^2) a body of this AMR allows.

    ``epsilon * (Phi_1au / c) * AMR * (1 au / r)^2``.  This is a momentum budget,
    not a thermophysical model: it holds whatever the object's spin, obliquity,
    thermal inertia or surface roughness, which is exactly why it is the gate.
    """
    amr = np.asarray(amr, dtype=float)
    r = np.asarray(r_au, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(epsilon) * PHI_OVER_C_1AU * amr / np.where(r > 0, r * r, np.nan)


def momentum_ceiling_a2(h, albedo: float = ALBEDO_GENEROUS,
                        rho_kg_m3: float = RHO_GENEROUS_KG_M3,
                        epsilon: float = EPSILON_HARD) -> np.ndarray:
    """Ceiling on ``|A2|`` (au/day^2) for a body of absolute magnitude ``H``.

    ``A2`` is defined at ``r = 1 au`` with ``g(r) = (1 au / r)^2``, so the
    distance factor cancels and the ceiling is a function of size alone.  The
    defaults deliberately make the ceiling generous — low density, high albedo,
    both of which raise the permitted acceleration — so that exceeding it is a
    statement about the object rather than about the assumptions.
    """
    d = diameter_m_from_h(h, albedo=albedo)
    return momentum_ceiling_si(amr_sphere(rho_kg_m3, d), r_au=1.0,
                               epsilon=epsilon) * SI_TO_AU_PER_DAY2


def ceiling_ratio(h, a2, albedo: float = ALBEDO_GENEROUS,
                  rho_kg_m3: float = RHO_GENEROUS_KG_M3,
                  epsilon: float = EPSILON_HARD) -> np.ndarray:
    """``|A2|`` over the momentum ceiling for that ``H``.  ``> 1`` is unphysical.

    NaN propagates: an object with no usable ``H`` or no fitted ``A2`` is
    *untestable*, and must never be scored as ordinary.
    """
    lvl = momentum_ceiling_a2(h, albedo=albedo, rho_kg_m3=rho_kg_m3,
                             epsilon=epsilon)
    v = np.abs(np.asarray(a2, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(lvl) & (lvl > 0), v / lvl, np.nan)


def amr_ceiling_ratio(h, amr, albedo: float = ALBEDO_GENEROUS,
                      rho_min_kg_m3: float = 500.0) -> np.ndarray:
    """Measured AMR over the largest AMR a solid body of that ``H`` can have.

    The strongest single test in the channel when ``srp`` is populated, because
    it is where outgassing and engineering part company: mass loss raises the
    *acceleration* but leaves the object a rock, whereas a thin shell, a panel or
    a sail is anomalous in area-to-mass ratio itself.  ``rho_min`` is set to
    500 kg/m^3 — already at the bottom of the measured rubble-pile range — so a
    ratio above 1 is not an argument about porosity.
    """
    d = diameter_m_from_h(h, albedo=albedo)
    lvl = amr_sphere(rho_min_kg_m3, d)
    v = np.abs(np.asarray(amr, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(lvl) & (lvl > 0), v / lvl, np.nan)


# ---------------------------------------------------------------------------
# A2 <-> semimajor-axis drift
# ---------------------------------------------------------------------------
def a2_to_dadt_au_per_day(a2, a_au, e) -> np.ndarray:
    """Secular drift ``<da/dt>`` (au/day) from ``A2`` (au/day^2).

    From ``dE/dt = v.F`` with purely transverse forcing and
    ``<r^-3> = a^-3 (1-e^2)^-3/2``:

        <da/dt> = 2 A2 r0^2 / [ n a^2 (1 - e^2) ]

    with ``n`` the mean motion in rad/day and ``r0 = 1 au``.  Checked against
    Bennu: the measured ``da/dt = -19.0e-4 au/Myr`` inverts to
    ``A2 = -4.55e-14``, against JPL's fitted ``-4.62e-14`` — 1.5%, the residual
    being the ``d = 2`` vs ``d = 2.25`` choice of ``g(r)`` exponent.
    """
    a2 = np.asarray(a2, dtype=float)
    a = np.asarray(a_au, dtype=float)
    e = np.asarray(e, dtype=float)
    # Gaussian gravitational constant: n = k / a^1.5 rad/day.
    k = 0.01720209895
    with np.errstate(divide="ignore", invalid="ignore"):
        n = k / a ** 1.5
        return 2.0 * a2 / (n * a * a * (1.0 - e * e))


def dadt_au_per_myr(a2, a_au, e) -> np.ndarray:
    """Secular drift in au/Myr, the unit the Yarkovsky literature quotes."""
    return a2_to_dadt_au_per_day(a2, a_au, e) * 365.25e6


def a2_from_dadt_au_per_myr(dadt, a_au, e) -> np.ndarray:
    """Invert :func:`dadt_au_per_myr`: ``A2`` (au/day^2) from ``da/dt`` (au/Myr)."""
    a = np.asarray(a_au, dtype=float)
    e = np.asarray(e, dtype=float)
    k = 0.01720209895
    with np.errstate(divide="ignore", invalid="ignore"):
        n = k / a ** 1.5
        return np.asarray(dadt, dtype=float) / 365.25e6 * n * a * a * (1.0 - e * e) / 2.0


def calibration_table() -> list[dict]:
    """Realised thermal-recoil efficiency for objects with measured ``A2``.

    Three objects spanning 4 m to 490 m, each with an independently published
    ``A2`` and a measured or well-constrained diameter and density.  The point of
    keeping this in code rather than in prose is that the test suite recomputes
    it: if a refactor changes the ceiling by a factor of anything, these numbers
    move and the suite says so.
    """
    rows = [
        {"name": "(101955) Bennu", "d_m": 490.0, "rho": 1190.0,
         "a2_au_day2": 4.62e-14, "source": "JPL fitted A2; da/dt -19.0e-4 au/Myr"},
        {"name": "2005 ES70", "d_m": 60.0, "rho": 1500.0,
         "a2_au_day2": 1.2848e-13, "source": "Del Vigna et al. 2018"},
        {"name": "2009 BD", "d_m": 4.0, "rho": 1500.0,
         "a2_au_day2": 1.14329e-12, "source": "Del Vigna et al. 2018; AMR-fitted"},
    ]
    for r in rows:
        amr = float(amr_sphere(r["rho"], r["d_m"]))
        ceil = float(momentum_ceiling_si(amr, 1.0, EPSILON_HARD)) * SI_TO_AU_PER_DAY2
        r["amr_m2_kg"] = amr
        r["ceiling_a2_hard"] = ceil
        r["epsilon_effective"] = r["a2_au_day2"] / ceil if ceil > 0 else float("nan")
    return rows


# ---------------------------------------------------------------------------
# The empirical, self-calibrating envelope (secondary; cross-checks the ceiling)
# ---------------------------------------------------------------------------
@dataclass
class Envelope:
    """A quantile of |value| as a function of ``H``, fitted to the sample itself.

    ``edges``/``level`` describe a step function in ``H``; ``n_per_bin`` records
    how many objects each level rests on, because a level fitted to four objects
    is not a boundary and callers must be able to see that.

    This runs *alongside* the momentum ceiling, not instead of it.  The ceiling
    says "no body of this size can do this"; the envelope says "no other object
    of this size in this survey does this".  The first is physics and is the gate;
    the second catches the case where the physics assumption (density, albedo,
    the ``H`` scale itself) is systematically off, because a survey-wide
    calibration error moves the empirical envelope and leaves the ceiling alone.
    Where the two disagree, that disagreement is a finding about the sample.
    """

    edges: np.ndarray
    level: np.ndarray
    n_per_bin: np.ndarray
    quantile: float
    min_per_bin: int
    ok: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    def evaluate(self, h) -> np.ndarray:
        """Envelope level at each ``H``; NaN where no bin could be fitted."""
        h = np.atleast_1d(np.asarray(h, dtype=float))
        out = np.full(h.shape, np.nan)
        if not self.ok:
            return out
        idx = np.clip(np.digitize(h, self.edges) - 1, 0, self.level.size - 1)
        good = np.isfinite(h)
        out[good] = self.level[idx[good]]
        return out


def fit_envelope(h, value, quantile: float = 0.999, n_bins: int = 8,
                 min_per_bin: int = 200) -> Envelope:
    """Fit the ``quantile`` of ``|value|`` per ``H`` bin, on the sample's own data.

    Asks "is this object extreme *for its size cohort*", which is answerable
    without knowing any object's albedo, spin or thermal inertia.  Bins are
    equal-count in ``H`` so the level is estimated with comparable precision
    everywhere, and a bin with fewer than ``min_per_bin`` objects is refused
    rather than fitted: a 99.9th percentile estimated from 50 objects is a
    maximum wearing a quantile's name.
    """
    h = np.asarray(h, dtype=float)
    v = np.abs(np.asarray(value, dtype=float))
    good = np.isfinite(h) & np.isfinite(v)
    env = Envelope(np.array([]), np.array([]), np.array([]), float(quantile),
                   int(min_per_bin))
    if good.sum() < max(min_per_bin, 50):
        env.reason = f"only {int(good.sum())} objects with both H and a value"
        return env
    hs = h[good]
    qs = np.linspace(0, 1, int(n_bins) + 1)
    edges = np.quantile(hs, qs)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    # Collapse duplicate edges (a spiked H distribution) so bins stay non-empty.
    edges = np.unique(edges)
    if edges.size < 3:
        env.reason = "H distribution too degenerate to bin"
        return env
    idx = np.digitize(hs, edges) - 1
    levels, counts = [], []
    for b in range(edges.size - 1):
        sel = idx == b
        n = int(sel.sum())
        counts.append(n)
        levels.append(float(np.quantile(v[good][sel], quantile)) if n >= min_per_bin
                      else np.nan)
    env.edges = edges
    env.level = np.array(levels, dtype=float)
    env.n_per_bin = np.array(counts, dtype=int)
    if not np.any(np.isfinite(env.level)):
        env.reason = (f"no H bin reached min_per_bin={min_per_bin}; the sample is "
                      f"too small to define an empirical envelope")
        return env
    # Fill refused bins from the nearest fitted one so an object in a sparse bin
    # is still testable, and record that it was borrowed.
    fin = np.isfinite(env.level)
    if not fin.all():
        idxs = np.arange(env.level.size)
        env.level[~fin] = np.interp(idxs[~fin], idxs[fin], env.level[fin])
        env.notes.append(f"{int((~fin).sum())} of {env.level.size} H bins had "
                         f"fewer than {min_per_bin} objects and borrowed their "
                         f"level from neighbouring bins")
    env.ok = True
    return env


def anomaly_ratio(h, value, envelope: Envelope) -> np.ndarray:
    """``|value|`` divided by the envelope level at that ``H``.

    Greater than 1 means the object exceeds the quantile of its own size cohort.
    NaN propagates as NaN: an object with no usable ``H`` is untestable, and must
    not be silently scored as ordinary.
    """
    lvl = envelope.evaluate(h)
    v = np.abs(np.asarray(value, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(lvl) & (lvl > 0), v / lvl, np.nan)


# ---------------------------------------------------------------------------
# Fit-quality gating
# ---------------------------------------------------------------------------
# A blind search for Yarkovsky signal in minor-planet astrometry returns a
# MAJORITY of spurious detections at nominal S/N > 3 -- the well-documented
# reason Del Vigna et al. (2018) require BOTH S/N >= 3 and agreement with a
# size-scaled expectation before calling a detection reliable.  The usual
# culprits are short arcs, few oppositions, isolated old astrometry and
# incomplete dynamical models, all of which inflate the fit residuals and so
# show up in `normalized_rms`.  These defaults encode that discipline.
MIN_SNR_A2 = 3.0
MAX_NORMALIZED_RMS = 1.5
MIN_ARC_DAYS = 180.0
MIN_OPPOSITIONS = 2


@dataclass
class FitQuality:
    """Whether an orbit solution is good enough for its ``A2`` to mean anything."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    snr: float = float("nan")


def orbit_quality(normalized_rms, arc_days, n_opp,
                  max_normalized_rms: float = MAX_NORMALIZED_RMS,
                  min_arc_days: float = MIN_ARC_DAYS,
                  min_oppositions: int = MIN_OPPOSITIONS) -> FitQuality:
    """Quality of the orbit *solution itself*, independent of any one parameter.

    Kept separate from the per-parameter signal-to-noise because the two gate
    different things and conflating them was a real error: an object with a
    well-determined orbit and a well-measured area-to-mass ratio but no fitted
    Yarkovsky term is perfectly testable on the radiation-pressure channel, and a
    combined gate would reject it for lacking an ``A2`` it never needed.
    """
    reasons: list[str] = []
    rms = _f(normalized_rms)
    if not math.isfinite(rms):
        reasons.append("no_normalized_rms")
    elif rms > max_normalized_rms:
        reasons.append(f"normalized_rms_{rms:.2f}_above_{max_normalized_rms}")
    arc = _f(arc_days)
    if not math.isfinite(arc):
        reasons.append("no_arc_length")
    elif arc < min_arc_days:
        reasons.append(f"arc_{arc:.0f}d_below_{min_arc_days:.0f}d")
    opp = _f(n_opp)
    if not math.isfinite(opp):
        reasons.append("no_opposition_count")
    elif opp < min_oppositions:
        reasons.append(f"n_opp_{opp:.0f}_below_{min_oppositions}")
    return FitQuality(ok=not reasons, reasons=reasons)


def parameter_snr(value, uncertainty) -> float:
    """Signal-to-noise of one fitted non-gravitational parameter.

    NaN where either is missing: a parameter with no uncertainty has no
    signal-to-noise, and substituting a default would silently promote it.
    """
    v, u = _f(value), _f(uncertainty)
    if not math.isfinite(v) or not math.isfinite(u) or u <= 0:
        return float("nan")
    return abs(v) / u


def fit_quality(a2, a2_unc, normalized_rms, arc_days, n_opp,
                min_snr: float = MIN_SNR_A2,
                max_normalized_rms: float = MAX_NORMALIZED_RMS,
                min_arc_days: float = MIN_ARC_DAYS,
                min_oppositions: int = MIN_OPPOSITIONS) -> FitQuality:
    """Gate one object's non-gravitational solution on the orbit's own quality.

    ``normalized_rms`` is the chi-like unitless fit residual: near 1 means the
    residuals match the assigned weights, and a value well above 1 means the
    reported uncertainties — including ``a2_unc`` — are underestimated, so any
    signal-to-noise built from them is overstated.  Rejecting on it is therefore
    not fussiness; it is the difference between a detection and an artefact.
    """
    reasons: list[str] = []
    snr = float("nan")
    v, u = _f(a2), _f(a2_unc)
    if not math.isfinite(v):
        reasons.append("no_fitted_a2")
    if not math.isfinite(u) or u <= 0:
        reasons.append("no_a2_uncertainty")
    else:
        snr = abs(v) / u if math.isfinite(v) else float("nan")
        if math.isfinite(snr) and snr < min_snr:
            reasons.append(f"a2_snr_{snr:.1f}_below_{min_snr}")
    rms = _f(normalized_rms)
    if not math.isfinite(rms):
        reasons.append("no_normalized_rms")
    elif rms > max_normalized_rms:
        reasons.append(f"normalized_rms_{rms:.2f}_above_{max_normalized_rms}")
    arc = _f(arc_days)
    if not math.isfinite(arc):
        reasons.append("no_arc_length")
    elif arc < min_arc_days:
        reasons.append(f"arc_{arc:.0f}d_below_{min_arc_days:.0f}d")
    opp = _f(n_opp)
    if not math.isfinite(opp):
        reasons.append("no_opposition_count")
    elif opp < min_oppositions:
        reasons.append(f"n_opp_{opp:.0f}_below_{min_oppositions}")
    return FitQuality(ok=not reasons, reasons=reasons, snr=snr)


def _f(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")

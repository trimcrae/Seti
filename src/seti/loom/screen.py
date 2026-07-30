"""Per-object screening: from an orbit row to a tier, with every rejection named.

Two independent paths reach the same tier ladder, and running both is the point.

**Path A — the fitted non-gravitational solution.**  ``lsst_mpc_orbits`` carries
``yarkovsky`` (A2) and ``srp`` (area-to-mass ratio) where MPC has fitted them.
This path is cheap and precise, and it is *biased*: MPC fits those terms only for
objects somebody already thought were interesting, so the subset is not a
population a rate can be computed against.

**Path B — the per-detection residual time series.**  Every solar-system alert
carries its own observed-minus-predicted along-track offset, whether or not
anyone ever fitted a non-gravitational term.  This path is expensive and noisier,
and it is *unbiased*, which is why it is the one the population statistics run on.

Where both are available for the same object they must agree; a disagreement is a
finding about the mirror or about the orbit epoch, not something to average over.

The tier ladder
---------------
``untestable``
    No usable ``H``, or no acceleration measurement at all.  Recorded as its own
    outcome and never folded in with "ordinary" — the difference between "we
    looked and it was normal" and "we could not look" is the difference between a
    limit and nothing.
``ordinary``
    Below the realistic thermal-recoil envelope (``epsilon = 0.1``).  This is
    where essentially every asteroid sits.
``watch``
    Above ``epsilon = 0.1`` but below the hard ceiling.  Real Yarkovsky can
    plausibly reach here for an extreme spin state or obliquity.
``interest``
    Above the ``epsilon = 1`` hard ceiling: sunlight cannot drive this, so it is
    either mass loss (a dark comet) or an object that is not a rock.
``candidate``
    Above the ``epsilon = 2`` specular limit, **or** anomalous in area-to-mass
    ratio, **and** with the systematic explanations excluded: the fit is clean,
    the residual is not the sample's common timing offset, the acceleration does
    not follow the sublimation law, and the object is not sitting in a
    sky-coherent residual patch.

Promotion to ``candidate`` requires the AMR channel or the law channel to speak,
not magnitude alone.  That is deliberate and it is the whole novelty position:
"large non-gravitational acceleration in an inactive small body" is Seligman et
al.'s dark comets, a populated field with an accepted explanation.  Magnitude
gets an object onto the list; only area-to-mass ratio or time structure gets it
off the list of things outgassing explains.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .nongrav import (
    ALBEDO_GENEROUS,
    ALBEDO_TYPICAL,
    AMR_ARTIFICIAL_FLOOR,
    EPSILON_HARD,
    EPSILON_INVIOLABLE,
    EPSILON_REALISTIC,
    RHO_GENEROUS_KG_M3,
    RHO_TYPICAL_KG_M3,
    a2_from_yarkovsky_column,
    amr_ceiling_ratio,
    amr_from_srp_column,
    ceiling_ratio,
    dadt_au_per_myr,
    diameter_m_from_h,
    fit_quality,
    orbit_quality,
    parameter_snr,
)

TIERS = ("untestable", "ordinary", "watch", "interest", "candidate")


@dataclass
class Thresholds:
    """Every LOOM selection number in one place; pinned to config by the tests."""

    # Momentum-ceiling assumptions.  Generous by construction: a low density and
    # a high albedo both raise the permitted acceleration, so exceeding the
    # ceiling is a statement about the object rather than about the assumptions.
    albedo_generous: float = ALBEDO_GENEROUS
    rho_generous_kg_m3: float = RHO_GENEROUS_KG_M3
    albedo_typical: float = ALBEDO_TYPICAL
    rho_typical_kg_m3: float = RHO_TYPICAL_KG_M3
    rho_min_kg_m3: float = 500.0
    epsilon_realistic: float = EPSILON_REALISTIC
    epsilon_hard: float = EPSILON_HARD
    epsilon_inviolable: float = EPSILON_INVIOLABLE
    # Orbit-fit quality: a majority of nominal S/N > 3 Yarkovsky detections in
    # blind searches are spurious, and short arcs with inflated residuals are the
    # usual cause, so these are not optional refinements.
    min_snr_a2: float = 3.0
    max_normalized_rms: float = 1.5
    min_arc_days: float = 180.0
    min_oppositions: int = 2
    # Area-to-mass ratio.  The artificial locus is 8e-3 to 1.2e-2 against a
    # natural ~3e-4; the gate is AMR against the object's OWN size, which is
    # strictly stronger than a flat floor, and the floor is a reporting label.
    amr_artificial_floor: float = AMR_ARTIFICIAL_FLOOR
    min_amr_ceiling_ratio: float = 1.0
    # Residual path.
    min_detections_for_drift: int = 8
    min_accel_snr: float = 5.0
    max_timing_correlation: float = 0.5
    min_delta_chi2_law: float = 9.0
    max_sky_variance_explained: float = 0.3
    # A transverse force displaces an object ALONG its track; star-catalogue bias
    # and mis-association have no directional preference.  An isotropic residual
    # is therefore not an acceleration, whatever its size.
    min_along_cross_power_ratio: float = 3.0
    # Monotone growth of the per-apparition offset.  Required where it can be
    # measured (three or more apparitions); silent where it cannot.
    min_apparition_spearman: float = 0.8
    # Resolved extent is a coma: the outgassing explanation showing itself
    # directly, and the cheapest possible rejection of a dark comet.
    max_extendedness: float = 0.5
    # Population tests.
    population_n_null: int = 2000
    population_seed: int = 20260730


@dataclass
class ObjectRecord:
    """One screened object.  Every rejection is named, never implied by absence."""

    key: str = ""
    designation: str | None = None
    h: float = float("nan")
    diameter_m: float = float("nan")
    a: float = float("nan")
    e: float = float("nan")
    i: float = float("nan")
    node: float = float("nan")
    argperi: float = float("nan")
    # Path A
    a2_au_day2: float = float("nan")
    a2_unc_au_day2: float = float("nan")
    a2_snr: float = float("nan")
    dadt_au_myr: float = float("nan")
    amr_m2_kg: float = float("nan")
    amr_unc_m2_kg: float = float("nan")
    ratio_realistic: float = float("nan")
    ratio_hard: float = float("nan")
    ratio_inviolable: float = float("nan")
    amr_ratio: float = float("nan")
    # Path B
    accel_au_day2_residual: float = float("nan")
    accel_snr_residual: float = float("nan")
    ratio_hard_residual: float = float("nan")
    n_detections: int = 0
    mjd_min: float = float("nan")
    mjd_max: float = float("nan")
    mean_along_arcsec: float = float("nan")
    mean_cross_arcsec: float = float("nan")
    along_cross_power_ratio: float = float("nan")
    apparition_spearman: float = float("nan")
    n_apparitions: int = 0
    # Photometric axis (ssObject); independent of the dynamical selection.
    h_g: float = float("nan")
    h_r: float = float("nan")
    h_i: float = float("nan")
    h_z: float = float("nan")
    g12_r: float = float("nan")
    extendedness_median: float = float("nan")
    moid_earth: float = float("nan")
    tisserand_j: float = float("nan")
    # Fit quality / systematics
    normalized_rms: float = float("nan")
    arc_days: float = float("nan")
    n_opp: float = float("nan")
    orbit_ok: bool = False
    orbit_reasons: list[str] = field(default_factory=list)
    amr_snr: float = float("nan")
    fit_ok: bool = False
    fit_reasons: list[str] = field(default_factory=list)
    timing_correlation: float = float("nan")
    best_law: str | None = None
    delta_chi2_law: float = float("nan")
    sky_variance_explained: float = float("nan")
    breakpoint_p: float = float("nan")
    breakpoint_mjd: float = float("nan")
    # Outcome
    score: float = float("nan")
    tier: str = "untestable"
    reasons: list[str] = field(default_factory=list)
    path: str = "none"

    def as_dict(self) -> dict:
        return asdict(self)


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fz(v) -> float:
    """Like :func:`_f`, but **exactly zero is missing**.

    Measured on the live mirror 2026-07-30: ``srp``, ``a1``, ``a2``, ``a3`` and
    ``dt`` are non-NULL for 1812 of 130,909 orbit rows and every one of those
    values is identically 0.0 — fill, not measurement — and the same pattern
    appears in ``a``, ``mean_motion``, ``period``, ``not_normalized_rms`` and
    ``arc_length_total`` on rows where the quantity was not determined.

    So ``COUNT(col)`` reports these columns as populated and a naive read treats an
    unfitted parameter as a *measured zero*, which for a non-gravitational term is
    the strongest possible statement that the object is ordinary.  That is the
    difference between "untestable" and "we looked and it was fine", and this
    channel is built on not confusing those two.  For every quantity this is
    applied to, zero is not physically meaningful: an orbit has non-zero semimajor
    axis, a moving object has non-zero sky rate, and a fit residual of identically
    zero is an absence.
    """
    x = _f(v)
    return float("nan") if x == 0.0 else x


def _fin(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def screen_orbit_row(row: dict, th: Thresholds,
                     key_column: str = "ssobjectid") -> ObjectRecord:
    """Path A: screen one ``lsst_mpc_orbits`` row.

    The unit conversions are the load-bearing part and they are not done here:
    ``yarkovsky`` is in ``1e-10 au/day^2`` and ``srp`` in ``m^2/ton``, and both
    conversions live in :mod:`seti.loom.nongrav` so there is exactly one place
    they can be wrong.  Reading ``yarkovsky`` as au/day^2 would overstate every
    acceleration by ten orders of magnitude and flag the entire catalogue.
    """
    # Prefer the UNPACKED designation.  Measured 2026-07-30: `lsst_ss_detection`
    # carries the packed form ("J97L01J", "K16Cd3G") while `lsst_mpc_orbits` carries
    # both, and the control set is written in unpacked form ("2020 SO").  Matching
    # the packed string against it finds nothing, silently, which would make the
    # positive control report NO_CONTROLS_PRESENT for an object that was right there.
    desig = (row.get("unpacked_primary_provisional_designation")
             or row.get("designation"))
    rec = ObjectRecord(key=str(row.get(key_column, "") or ""),
                       designation=str(desig) if desig else None,
                       path="mpc_orbits")
    rec.h = _fz(row.get("h"))
    rec.a, rec.e, rec.i = _fz(row.get("a")), _f(row.get("e")), _f(row.get("i"))
    rec.node, rec.argperi = _f(row.get("node")), _f(row.get("argperi"))
    rec.normalized_rms = _fz(row.get("normalized_rms"))
    rec.arc_days = _fz(row.get("arc_length_total"))
    rec.n_opp = _f(row.get("nopp"))

    if _fin(rec.h):
        rec.diameter_m = float(diameter_m_from_h(rec.h, albedo=th.albedo_typical))

    # `_fz`, not `_f`: an unfitted non-gravitational term arrives as exactly 0.0 in
    # this mirror, and reading it as a measured zero would be the strongest possible
    # statement that the object is ordinary.
    yark = _fz(row.get("yarkovsky"))
    yark_unc = _fz(row.get("yarkovsky_unc"))
    if _fin(yark):
        rec.a2_au_day2 = float(a2_from_yarkovsky_column(yark))
    if _fin(yark_unc):
        rec.a2_unc_au_day2 = float(a2_from_yarkovsky_column(yark_unc))
    srp = _fz(row.get("srp"))
    srp_unc = _fz(row.get("srp_unc"))
    if _fin(srp):
        rec.amr_m2_kg = float(amr_from_srp_column(srp))
    if _fin(srp_unc):
        rec.amr_unc_m2_kg = float(amr_from_srp_column(srp_unc))

    # Orbit quality and per-parameter signal-to-noise are gated SEPARATELY.  An
    # object with a clean orbit and a well-measured area-to-mass ratio but no
    # fitted Yarkovsky term is perfectly testable on the radiation-pressure
    # channel, and a combined gate would reject it for lacking an ``A2`` it never
    # needed -- which is not a hypothetical, it is what the first version did.
    oq = orbit_quality(rec.normalized_rms, rec.arc_days, rec.n_opp,
                       max_normalized_rms=th.max_normalized_rms,
                       min_arc_days=th.min_arc_days,
                       min_oppositions=th.min_oppositions)
    rec.orbit_ok, rec.orbit_reasons = oq.ok, list(oq.reasons)
    rec.a2_snr = parameter_snr(rec.a2_au_day2, rec.a2_unc_au_day2)
    rec.amr_snr = parameter_snr(rec.amr_m2_kg, rec.amr_unc_m2_kg)
    q = fit_quality(rec.a2_au_day2, rec.a2_unc_au_day2, rec.normalized_rms,
                    rec.arc_days, rec.n_opp, min_snr=th.min_snr_a2,
                    max_normalized_rms=th.max_normalized_rms,
                    min_arc_days=th.min_arc_days,
                    min_oppositions=th.min_oppositions)
    rec.fit_ok, rec.fit_reasons = q.ok, list(q.reasons)

    if _fin(rec.a2_au_day2) and _fin(rec.a) and _fin(rec.e):
        rec.dadt_au_myr = float(dadt_au_per_myr(rec.a2_au_day2, rec.a, rec.e))
    if _fin(rec.a2_au_day2) and _fin(rec.h):
        kw = {"albedo": th.albedo_generous, "rho_kg_m3": th.rho_generous_kg_m3}
        rec.ratio_realistic = float(ceiling_ratio(
            rec.h, rec.a2_au_day2, epsilon=th.epsilon_realistic, **kw))
        rec.ratio_hard = float(ceiling_ratio(
            rec.h, rec.a2_au_day2, epsilon=th.epsilon_hard, **kw))
        rec.ratio_inviolable = float(ceiling_ratio(
            rec.h, rec.a2_au_day2, epsilon=th.epsilon_inviolable, **kw))
    if _fin(rec.amr_m2_kg) and _fin(rec.h):
        rec.amr_ratio = float(amr_ceiling_ratio(
            rec.h, rec.amr_m2_kg, albedo=th.albedo_generous,
            rho_min_kg_m3=th.rho_min_kg_m3))
    return rec


def assign_tier(rec: ObjectRecord, th: Thresholds) -> ObjectRecord:
    """Place a screened object on the tier ladder and record why.

    ``candidate`` requires an *artificiality* channel — area-to-mass ratio, or a
    time structure that sublimation does not produce — not just a large
    acceleration.  A magnitude-only promotion would rediscover dark comets, which
    is what the literature already contains.
    """
    reasons: list[str] = []
    # Each channel is gated on ITS OWN parameter's signal-to-noise.  A fitted value
    # without an uncertainty has no signal-to-noise at all, and substituting a
    # default would silently promote it.
    a2_usable = _fin(rec.a2_snr) and rec.a2_snr >= th.min_snr_a2
    amr_usable = _fin(rec.amr_snr) and rec.amr_snr >= th.min_snr_a2
    ratios = [r for r in ((rec.ratio_hard if a2_usable else float("nan")),
                          rec.ratio_hard_residual) if _fin(r)]
    score = max(ratios) if ratios else float("nan")
    if _fin(rec.amr_ratio):
        score = rec.amr_ratio if not _fin(score) else max(score, rec.amr_ratio)
    rec.score = score

    if not _fin(rec.h):
        rec.tier, rec.reasons = "untestable", ["no_absolute_magnitude"]
        return rec
    if not ratios and not _fin(rec.amr_ratio):
        rec.tier, rec.reasons = "untestable", ["no_acceleration_measurement"]
        if _fin(rec.a2_au_day2) and not a2_usable:
            rec.reasons = [f"a2_snr_{rec.a2_snr:.1f}_below_{th.min_snr_a2}"
                           if _fin(rec.a2_snr) else "no_a2_uncertainty"]
        return rec

    # Systematic explanations, evaluated once and reused by every threshold.
    if _fin(rec.timing_correlation) and abs(rec.timing_correlation) > th.max_timing_correlation:
        reasons.append(f"timing_correlated_{rec.timing_correlation:.2f}")
    if (rec.best_law == "sublimation" and _fin(rec.delta_chi2_law)
            and rec.delta_chi2_law >= th.min_delta_chi2_law):
        reasons.append("sublimation_law_preferred")
    if (_fin(rec.sky_variance_explained)
            and rec.sky_variance_explained > th.max_sky_variance_explained):
        reasons.append(f"sky_coherent_{rec.sky_variance_explained:.2f}")
    if (_fin(rec.along_cross_power_ratio)
            and rec.along_cross_power_ratio < th.min_along_cross_power_ratio):
        reasons.append(f"residual_isotropic_along_cross_"
                       f"{rec.along_cross_power_ratio:.1f}")
    if _fin(rec.extendedness_median) and rec.extendedness_median > th.max_extendedness:
        reasons.append(f"resolved_extent_{rec.extendedness_median:.2f}__coma")
    # Monotone growth across apparitions is required where three or more
    # apparitions exist; where they do not, the axis is silent rather than
    # permissive, and the record shows which of the two happened.
    if rec.n_apparitions >= 3 and _fin(rec.apparition_spearman) \
            and rec.apparition_spearman < th.min_apparition_spearman:
        reasons.append(f"apparition_offset_not_monotone_"
                       f"{rec.apparition_spearman:.2f}")
    # Orbit-solution quality vetoes every channel; a badly determined orbit makes
    # both the fitted parameters and the ephemeris prediction untrustworthy.
    if not rec.orbit_ok:
        reasons.extend(rec.orbit_reasons)
    systematic = bool(reasons)

    hard = max(ratios, default=float("nan"))
    realistic = rec.ratio_realistic
    if not _fin(realistic) and _fin(hard):
        # ratio scales as 1/epsilon, so the realistic ratio is recoverable.
        realistic = hard * th.epsilon_hard / th.epsilon_realistic

    # The two artificiality channels.  An anomalous area-to-mass ratio stands on
    # its own -- outgassing raises an acceleration but does not turn a rock into a
    # thin shell.  A distance-independent acceleration needs the magnitude channel
    # too, because "constant" is also what a mis-modelled perturbation looks like
    # over a short arc.
    amr_anomalous = (amr_usable and _fin(rec.amr_ratio)
                     and rec.amr_ratio >= th.min_amr_ceiling_ratio)
    if (_fin(rec.amr_ratio) and rec.amr_ratio >= th.min_amr_ceiling_ratio
            and not amr_usable):
        reasons.append("amr_anomalous_but_uncertainty_missing_or_low_snr")
    law_anomalous = (rec.best_law == "constant" and _fin(rec.delta_chi2_law)
                     and rec.delta_chi2_law >= th.min_delta_chi2_law)

    if not systematic and (amr_anomalous or (_fin(hard) and hard >= 1.0
                                             and law_anomalous)):
        rec.tier = "candidate"
        if amr_anomalous:
            reasons.append(f"amr_{rec.amr_m2_kg:.2e}_is_{rec.amr_ratio:.1f}x_"
                           f"the_max_for_a_solid_body_of_this_size")
        if law_anomalous:
            reasons.append("acceleration_independent_of_heliocentric_distance")
    elif _fin(hard) and hard >= 1.0:
        rec.tier = "interest"
        reasons.append("above_hard_momentum_ceiling")
        if not amr_anomalous and not law_anomalous:
            reasons.append("magnitude_only__consistent_with_a_dark_comet")
    elif _fin(realistic) and realistic >= 1.0:
        rec.tier = "watch"
        reasons.append("above_realistic_thermal_recoil_envelope")
    else:
        rec.tier = "ordinary"
    rec.reasons = reasons
    return rec


def screen_orbits(rows: list[dict], th: Thresholds,
                  key_column: str = "ssobjectid") -> tuple[list[ObjectRecord], dict]:
    """Path A over the whole parent population, with a funnel count."""
    recs = [assign_tier(screen_orbit_row(r, th, key_column=key_column), th)
            for r in rows]
    funnel: dict = {"n_rows": len(rows)}
    for t in TIERS:
        funnel[f"n_{t}"] = sum(1 for r in recs if r.tier == t)
    funnel["n_with_yarkovsky"] = sum(1 for r in recs if _fin(r.a2_au_day2))
    funnel["n_with_srp"] = sum(1 for r in recs if _fin(r.amr_m2_kg))
    funnel["n_fit_ok"] = sum(1 for r in recs if r.fit_ok)
    funnel["n_amr_anomalous"] = sum(
        1 for r in recs if _fin(r.amr_ratio) and r.amr_ratio >= th.min_amr_ceiling_ratio)
    # A funnel where nothing has a fitted non-gravitational term is not a null
    # result, it is a dead path, and it must not be reported as the former.
    if funnel["n_with_yarkovsky"] == 0 and funnel["n_with_srp"] == 0:
        funnel["verdict"] = "NONGRAV_COLUMNS_EMPTY"
        funnel["note"] = ("no object in the sample has a fitted yarkovsky or srp "
                          "value; Path A is unavailable in this mirror and only "
                          "the per-detection residual path can run")
    else:
        funnel["verdict"] = "OK"
    return recs, funnel


def covariate_labels(recs: list[ObjectRecord], n_bins: int = 4) -> np.ndarray:
    """Strata for the matched null: what drives both detectability and residual size.

    An anomalous object is preferentially small, faint, short-arc and
    poorly-observed, and objects like that are *not* uniformly distributed in the
    belt.  Matching the null on those covariates is what stops the population
    statistics from firing on the selection function instead of on the sky, which
    is the same failure TOCSIN hit when its stratified null shipped inert and four
    deep-drilling-field artefacts were promoted.
    """
    cols = []
    for name in ("h", "arc_days", "n_opp", "normalized_rms"):
        v = np.array([getattr(r, name) for r in recs], dtype=float)
        if not np.any(np.isfinite(v)):
            continue
        q = np.linspace(0, 1, int(n_bins) + 1)[1:-1]
        edges = np.nanquantile(v[np.isfinite(v)], q) if np.isfinite(v).sum() > 4 else []
        idx = np.digitize(v, edges) if len(edges) else np.zeros(v.size, dtype=int)
        idx = np.where(np.isfinite(v), idx, -1)
        cols.append(idx.astype(int))
    if not cols:
        return np.zeros(len(recs), dtype=int)
    lab = np.zeros(len(recs), dtype=np.int64)
    for c in cols:
        lab = lab * (int(c.max()) + 2) + (c + 1)
    return lab

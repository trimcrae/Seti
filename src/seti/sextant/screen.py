"""SEXTANT's screen: the discard pile, and what the surviving astrometry says.

WHICH QUESTION THIS ASKS, AND WHY IT IS NOT THE OBVIOUS ONE
===========================================================
The obvious screen --- rank ``gaiafpr.sso_observation`` objects by anomalous
post-fit astrometric residual --- is occupied, twice over, on this exact table:

* **Liberato, Tanga, Mary, Lallemand, Liu, Carry, Desmars, Hestroffer, Minker &
  Siakas**, *Follow the wobble: Statistical methods to detect astrometric binary
  asteroids in Gaia FPR* (arXiv:2605.22702, May 2026).  Along-scan projection of
  post-fit residuals, an FPR-consistent noise model, a Monte-Carlo noise-only
  control, trend detection, a period search, improved statistical selection ---
  343 binary candidates over 410 windows.  That is the population-scale
  anomalous-residual screen, with our null-control design, already run.
* **Dziadura, Bartczak & Oszkiewicz**, *Assessing the detection of the Yarkovsky
  effect using Gaia DR3 and FPR catalogues* (A&A 693, A31; arXiv:2411.09750).
  OrbFit solutions for six elements **plus a non-gravitational A2** for 446 NEAs
  and 54,094 inner-main-belt and Mars-crossing objects on Gaia FPR.  No
  main-belt detection.  The amplitude question was asked at our scale and came
  back null.

Their earlier companion, *Binary asteroid candidates in Gaia DR3 astrometry*
(A&A 688, A50; arXiv:2406.07195), is the contamination catalogue subtracted
below.  And the observed-minus-computed construction itself is executed and
published: the Gaia-CRF3 comparison against planetary ephemerides over 1001
asteroids puts ~96% of along-scan residuals inside +-5 mas, ~52% sub-mas, with a
strongly non-Gaussian across-scan tail.

**Every one of those searches operates post-fit.**  They screen what the
astrometric solution successfully modelled.  So an object whose astrometry is
*systematically un-fittable* --- one that the pipeline keeps throwing away --- is
invisible to all of them, by construction, and has never been looked at.

So the observable here is the **rejection pattern**: ``is_rejected``,
``astrometric_outcome_ccd`` and ``astrometric_outcome_transit``, as a per-object
rate against the attempts denominator.  Published context for the scale: the
outlier fraction is ~0.58% in DR3 and ~1% in DR2, so the ordinary rate is small,
well measured, and has structure that can be modelled --- which is what makes an
excess meaningful rather than merely large.

**This is a measurement before it is a search.**  Nobody has published the
per-object distribution of Gaia SSO astrometric rejection rate, so the first
output is that distribution and its dependence on observing conditions.  The
search is what remains after the conditions are divided out.

Why the second stage exists
---------------------------
A rejection excess is an *anomaly detector*, not an explanation.  What it
surfaces still has to be explained, and the explanation is where the second
novel discriminant lives: for the observations of a flagged object that the
pipeline *did* keep, ``seti.sextant.residuals.model_comparison`` asks **which
law** the residual prefers --- a force acting on the orbit (radiation,
sublimation, distance-independent) against an artefact of the measurement
(illumination photocentre, residual solar deflection, timing).  Both papers
above assume a functional form and fit its amplitude; neither asks which form
the data prefers.  A station-keeping or trajectory-correcting object cannot
imitate the geometric family, because its acceleration does not know where the
Sun is.

The contamination discipline, which is the whole job
----------------------------------------------------
A rejection-rate excess has a shelf of mundane readings and each has to be
excluded, not waved at:

===============  =======================================================
crowding         a source near the galactic plane has more neighbours, so
                 more transits are confused and discarded
apparent motion  a fast mover smears along scan and fails the point-source
                 window; this is the reading that most resembles the signal
brightness       saturation at the bright end, low signal at the faint end
scan geometry    a transit at an unfavourable scan angle or an unusual
                 across-scan position is discarded more often
phase coverage   an object observed only at extreme phase angles has a
                 systematically offset photocentre
sampling         an object with three attempts has no measurable rate at all
===============  =======================================================

Each is carried as a covariate of the null, the excess is recomputed with each
covariate individually dropped, and an excess that evaporates under any one of
them is reported as that covariate's, by name.  The null itself is not
parametric: it is matched random subsets of the same screened sample, which is
LOOM's architecture and is what makes the statistic immune to any contaminant
that does not itself cluster in the matched covariates.

And the decision is on the SET, not on one object
--------------------------------------------------
LOOM's premise is unchanged and is not weakened here: a self-replicating probe is
defined by a *population sharing an origin*, so a single object with an odd
rejection rate is a curiosity and a *set* of them sharing orbital structure is
the claim.  The population stage reuses ``seti.loom.replication`` --- element
clustering, orbital-pole coherence, inclination isotropy, resonance
concentration --- against matched random subsets of this screened sample, with
the mis-linkage collapse running before any statistic sees the set.

Coverage limit, stated once and meant
-------------------------------------
Gaia SSO astrometry is archival: the mission's 2014-2020 window.  This tests a
*static* population, which is exactly the question LOOM asks, and it cannot
detect a new event or substitute for a nightly cadence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..loom.replication import replication_tests
from ..tocsin.population import matched_draws
from .residuals import ResidualSeries, model_comparison, scan_axis_partition

TIERS = ("untestable", "ordinary", "watch", "interest", "candidate")

#: Published outlier fractions, used as a sanity band on the measured rate: a
#: sample-wide rate far from these means the flags are not being read correctly,
#: which is a finding about the columns and not about the sky.
PUBLISHED_OUTLIER_FRACTION_DR3 = 0.0058
PUBLISHED_OUTLIER_FRACTION_DR2 = 0.010


@dataclass
class Thresholds:
    """Every SEXTANT selection number in one place."""

    # --- the rejection screen -------------------------------------------------
    #: An object with too few attempts has no measurable rate.  30 attempts at a
    #: 0.6% base rate expects 0.18 rejections, so a single rejection is already a
    #: 5x "excess" --- which is why the floor is on ATTEMPTS and not on the rate.
    min_attempts: int = 60
    min_transits: int = 8
    #: Standardised excess (quasi-binomial, over-dispersion estimated per
    #: stratum) above which an object enters the shortlist.
    min_rejection_z: float = 4.0
    #: Fraction of the excess that must survive dropping any single covariate
    #: from the stratification.  An excess that halves when crowding is removed
    #: from the null was crowding.
    min_covariate_survival: float = 0.5
    #: Rank correlation of the excess with any single covariate above which the
    #: screen is measuring the survey rather than the sky.
    max_quality_correlation: float = 0.3
    n_covariate_bins: int = 4
    #: Randomisation depth for the empirical null on the maximum excess.
    n_null: int = 2000
    seed: int = 20260825

    # --- the second stage: which law -----------------------------------------
    min_observations_for_model: int = 40
    min_arc_days: float = 365.0
    min_delta_chi2: float = 16.0
    #: Signal-to-noise on the fitted force amplitude, quoted on the LARGER of the
    #: two systematic-correlation models.
    min_force_snr: float = 5.0
    #: Above this, an orbit fit could have removed essentially all of the signal
    #: and the measurement is not a measurement.
    max_absorbed_fraction: float = 0.995

    # --- contamination --------------------------------------------------------
    #: Galactic latitude below which crowding dominates and a rejection excess is
    #: not attributable to the object.
    min_abs_galactic_latitude_deg: float = 10.0
    #: Across-scan to along-scan sigma ratio outside which the error model is not
    #: behaving as the mission's anisotropy says it should.
    min_scan_anisotropy: float = 3.0
    #: Population stage.
    population_n_null: int = 2000
    min_parent_for_population: int = 200
    min_anomalies_for_population: int = 5


# ---------------------------------------------------------------------------
# 1. The observable: attempts, rejections, and the covariates of both
# ---------------------------------------------------------------------------
#: Which values of `astrometric_outcome_*` mean "the attempt succeeded".
#: THE ENCODING IS NOT SETTLED.  The archive documents these columns and the
#: sandbox has no egress, so 0 is taken as nominal and everything else as a
#: non-nominal outcome.  This is a PROBE QUESTION (docs/sextant.md): the runner
#: must histogram both columns before any rate computed from them is believed,
#: because a wrong reading here does not fail, it changes the denominator.
NOMINAL_OUTCOME_VALUES: tuple[float, ...] = (0.0,)

#: Covariates of the rejection rate.  Each is a documented mundane reading of an
#: excess, and each therefore has to be in the null rather than in the discussion.
DEFAULT_COVARIATES = ("apparent_motion_mas_per_day", "abs_galactic_latitude_deg",
                      "magnitude", "median_phase_deg", "n_attempts")


@dataclass
class RejectionRecord:
    """One object's discard pattern, with its denominator and its covariates."""

    key: str = ""
    number_mp: float = float("nan")
    denomination: str | None = None
    n_attempts: int = 0
    n_rejected: int = 0
    n_transits: int = 0
    n_transits_rejected: int = 0
    n_outcome_nonnominal_ccd: int = 0
    n_outcome_nonnominal_transit: int = 0
    rate: float = float("nan")
    transit_rate: float = float("nan")
    expected_rate: float = float("nan")
    expected_n: float = float("nan")
    excess_z: float = float("nan")
    excess_z_covariate_min: float = float("nan")
    covariate_survival: float = float("nan")
    binding_covariate: str | None = None
    stratum: int = -1
    # covariates
    apparent_motion_mas_per_day: float = float("nan")
    abs_galactic_latitude_deg: float = float("nan")
    magnitude: float = float("nan")
    median_phase_deg: float = float("nan")
    arc_days: float = float("nan")
    # orbit, for the population stage
    a: float = float("nan")
    e: float = float("nan")
    i: float = float("nan")
    node: float = float("nan")
    argperi: float = float("nan")
    h: float = float("nan")
    mjd_min: float = float("nan")
    mjd_max: float = float("nan")
    # stage two
    model_verdict: str = ""
    best_force_model: str | None = None
    best_geometric_model: str | None = None
    family_margin: float = float("nan")
    force_amplitude: float = float("nan")
    force_snr: float = float("nan")
    absorbed_fraction: float = float("nan")
    law_verdict: str = ""
    # contamination
    known_binary: bool = False
    binary_reference: str | None = None
    scan_anisotropy: float = float("nan")
    # outcome
    tier: str = "untestable"
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _as_bool(v, n: int) -> np.ndarray:
    if v is None:
        return np.zeros(n, dtype=bool)
    a = np.asarray(v)
    if a.dtype == object:
        return np.array([bool(x) for x in a], dtype=bool)
    return a.astype(bool)


def rejection_counts(obs, *, key: str = "",
                     nominal_outcomes: tuple[float, ...] = NOMINAL_OUTCOME_VALUES
                     ) -> RejectionRecord:
    """Count one object's attempts and discards, keeping the denominator.

    The denominator is the point.  A rejection *count* is meaningless --- a
    well-observed asteroid has hundreds of attempts and a badly-placed one has a
    dozen --- and every published treatment of these flags uses them to filter
    rather than to measure, which is exactly why the rate has never been looked
    at.  Both the per-observation and the per-transit rate are kept: rejections
    cluster inside a transit (nine CCD crossings behind one attitude solution
    and one confusion), so the per-observation count is over-dispersed relative
    to binomial and the per-transit count is closer to independent.
    """
    from .residuals import as_columns

    cols = as_columns(obs)
    rec = RejectionRecord(key=key)
    if not cols or "ra" not in cols:
        return rec
    n = int(np.asarray(cols["ra"]).size)
    rej = _as_bool(cols.get("is_rejected"), n)
    rec.n_attempts = n
    rec.n_rejected = int(np.count_nonzero(rej))
    rec.rate = rec.n_rejected / n if n else float("nan")

    tid = cols.get("transit_id")
    if tid is None:
        tids = np.arange(n)
    else:
        tids = np.asarray(tid)
    uniq = np.unique(tids)
    rec.n_transits = int(uniq.size)
    rec.n_transits_rejected = int(sum(1 for t in uniq if np.all(rej[tids == t])))
    rec.transit_rate = (rec.n_transits_rejected / rec.n_transits
                        if rec.n_transits else float("nan"))

    for col, attr in (("astrometric_outcome_ccd", "n_outcome_nonnominal_ccd"),
                      ("astrometric_outcome_transit", "n_outcome_nonnominal_transit")):
        v = cols.get(col)
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        bad = np.isfinite(arr) & ~np.isin(arr, np.asarray(nominal_outcomes, float))
        setattr(rec, attr, int(np.count_nonzero(bad)))

    for c, attr in (("number_mp", "number_mp"),):
        v = cols.get(c)
        if v is not None and np.asarray(v).size:
            try:
                setattr(rec, attr, float(np.asarray(v, dtype=float)[0]))
            except (TypeError, ValueError):
                pass
    d = cols.get("denomination")
    if d is not None and np.asarray(d).size:
        rec.denomination = str(np.asarray(d)[0])
    return rec


def sample_rejection_summary(records: list[RejectionRecord]) -> dict:
    """The measurement that has to come before the search.

    Nobody has published the per-object distribution of Gaia SSO astrometric
    rejection rate, so this is the first output of the channel and it stands on
    its own.  The sample-wide rate is compared with the published DR2/DR3 outlier
    fractions: landing far from them means the flags are not being read the way
    the mission means them, which is a finding about the columns and must stop
    the run rather than be folded into a result.
    """
    att = np.array([r.n_attempts for r in records], dtype=float)
    rej = np.array([r.n_rejected for r in records], dtype=float)
    out: dict = {"n_objects": len(records), "n_attempts": float(att.sum()),
                 "n_rejected": float(rej.sum())}
    if att.sum() <= 0:
        out["verdict"] = "NO_ATTEMPTS"
        return out
    rate = float(rej.sum() / att.sum())
    out["sample_rejection_rate"] = rate
    out["published_dr3_outlier_fraction"] = PUBLISHED_OUTLIER_FRACTION_DR3
    out["published_dr2_outlier_fraction"] = PUBLISHED_OUTLIER_FRACTION_DR2
    per = rej[att > 0] / att[att > 0]
    if per.size:
        qs = [0.5, 0.9, 0.99, 0.999, 1.0]
        out["per_object_rate_quantiles"] = {
            f"q{q}": float(np.quantile(per, q)) for q in qs}
        out["n_objects_with_zero_rejections"] = int(np.count_nonzero(per == 0))
    lo = 0.2 * min(PUBLISHED_OUTLIER_FRACTION_DR2, PUBLISHED_OUTLIER_FRACTION_DR3)
    hi = 5.0 * max(PUBLISHED_OUTLIER_FRACTION_DR2, PUBLISHED_OUTLIER_FRACTION_DR3)
    if not (lo <= rate <= hi):
        out["verdict"] = "RATE_INCONSISTENT_WITH_PUBLISHED_OUTLIER_FRACTION"
        out["note"] = (f"sample rejection rate {rate:.4f} is outside "
                       f"[{lo:.4f}, {hi:.4f}]; the flag columns are probably not "
                       f"being read as the mission means them.  Histogram "
                       f"is_rejected and both astrometric_outcome columns before "
                       f"anything computed from them is believed.")
        return out
    out["verdict"] = "OK"
    return out


# ---------------------------------------------------------------------------
# 2. The null: the survey's own rejection statistics, stratified
# ---------------------------------------------------------------------------
def covariate_strata(records: list[RejectionRecord],
                     covariates: tuple[str, ...] = DEFAULT_COVARIATES,
                     n_bins: int = 4) -> np.ndarray:
    """Equal-count strata in the covariates that drive the rejection rate.

    Equal-count rather than equal-width because every one of these covariates is
    heavily skewed --- apparent motion spans four orders of magnitude across the
    NEO-to-outer-belt range --- and an equal-width binning would put 99% of the
    sample in one bin and call it a control.

    An object missing a covariate gets its own bin rather than being dropped:
    "we could not control for this" and "this was controlled for" must not be
    the same row.
    """
    cols = []
    for name in covariates:
        v = np.array([getattr(r, name, float("nan")) for r in records], dtype=float)
        if not np.any(np.isfinite(v)):
            continue
        fin = np.isfinite(v)
        if fin.sum() > 4 * n_bins:
            q = np.linspace(0, 1, int(n_bins) + 1)[1:-1]
            edges = np.unique(np.nanquantile(v[fin], q))
            idx = np.digitize(v, edges)
        else:
            idx = np.zeros(v.size, dtype=int)
        cols.append(np.where(fin, idx, -1).astype(int))
    if not cols:
        return np.zeros(len(records), dtype=np.int64)
    lab = np.zeros(len(records), dtype=np.int64)
    for c in cols:
        lab = lab * (int(c.max()) + 2) + (c + 1)
    return lab


def _excess_z(records: list[RejectionRecord], labels: np.ndarray,
              min_attempts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quasi-binomial standardised excess per object, given a stratification.

    Returns ``(z, expected_rate, dispersion)``.  The dispersion factor is
    estimated **within each stratum** from the observed scatter of the rate
    across its objects, because rejections cluster --- inside a transit, inside a
    field, inside a scan --- so the binomial variance is an underestimate and a
    naive z would flag a large fraction of the catalogue.  This is the same
    mistake LOOM made once with ``scatter/sqrt(n)``, in a different costume.
    """
    att = np.array([r.n_attempts for r in records], dtype=float)
    rej = np.array([r.n_rejected for r in records], dtype=float)
    z = np.full(att.size, np.nan)
    exp_rate = np.full(att.size, np.nan)
    disp = np.full(att.size, np.nan)
    usable = att >= float(min_attempts)
    for lab in np.unique(labels):
        sel = (labels == lab) & usable
        n_in = int(sel.sum())
        if n_in < 5:
            continue
        p = float(rej[sel].sum() / att[sel].sum()) if att[sel].sum() > 0 else 0.0
        if p <= 0:
            # A stratum with no rejections at all cannot calibrate an excess; the
            # honest expected rate is an upper bound from the attempts it holds.
            p = 1.0 / (att[sel].sum() + 1.0)
        e = att[sel] * p
        var = np.maximum(e * (1.0 - p), 1e-12)
        raw = (rej[sel] - e) / np.sqrt(var)
        # Quasi-binomial dispersion, estimated robustly so the very objects being
        # searched for do not inflate the yardstick that is supposed to find them.
        mad = float(np.median(np.abs(raw - np.median(raw))))
        phi = max(1.4826 * mad, 1.0)
        z[sel] = (rej[sel] - e) / (np.sqrt(var) * phi)
        exp_rate[sel] = p
        disp[sel] = phi
    return z, exp_rate, disp


def rejection_excess(records: list[RejectionRecord], th: Thresholds,
                     covariates: tuple[str, ...] = DEFAULT_COVARIATES) -> dict:
    """Score every object's discard rate against the survey's own statistics.

    Three things happen here and the third is the one that matters.

    1. The excess is computed under the FULL stratification.
    2. It is recomputed with each covariate dropped in turn.  An excess that
       collapses when crowding leaves the null was crowding; the covariate whose
       removal costs the most is recorded by name as ``binding_covariate``, and
       ``covariate_survival`` is the fraction of the excess that survives the
       worst case.  This is the difference between controlling for a confounder
       and mentioning it.
    3. The maximum excess is calibrated against matched random subsets of the
       same screened sample, so the p-value already carries the trials factor
       over a catalogue of >100,000 objects rather than acquiring it later.
    """
    labels = covariate_strata(records, covariates, th.n_covariate_bins)
    z, exp_rate, disp = _excess_z(records, labels, th.min_attempts)
    out: dict = {"n_objects": len(records), "covariates": list(covariates),
                 "n_strata": int(np.unique(labels).size)}

    # Drop-one sensitivity.
    drop: dict[str, np.ndarray] = {}
    for name in covariates:
        reduced = tuple(c for c in covariates if c != name)
        if not reduced:
            continue
        lab_r = covariate_strata(records, reduced, th.n_covariate_bins)
        z_r, _, _ = _excess_z(records, lab_r, th.min_attempts)
        drop[name] = z_r
    if drop:
        stack = np.vstack([drop[k] for k in drop])
        with np.errstate(invalid="ignore"):
            worst_idx = np.nanargmin(np.where(np.isfinite(stack), stack, np.inf),
                                     axis=0) if stack.size else None
        z_min = np.nanmin(stack, axis=0) if stack.size else np.full(z.shape, np.nan)
        names = list(drop)
    else:
        z_min = z.copy()
        worst_idx = None
        names = []

    for k, r in enumerate(records):
        r.stratum = int(labels[k])
        r.excess_z = float(z[k])
        r.expected_rate = float(exp_rate[k])
        r.expected_n = (float(exp_rate[k]) * r.n_attempts
                        if math.isfinite(exp_rate[k]) else float("nan"))
        r.excess_z_covariate_min = float(z_min[k]) if z_min.size else float("nan")
        if math.isfinite(r.excess_z) and r.excess_z > 0 and math.isfinite(
                r.excess_z_covariate_min):
            r.covariate_survival = float(max(r.excess_z_covariate_min, 0.0)
                                         / r.excess_z)
        if worst_idx is not None and names and math.isfinite(z_min[k]):
            r.binding_covariate = names[int(worst_idx[k])]

    finite = np.isfinite(z)
    out["n_scored"] = int(finite.sum())
    if finite.sum() < 20:
        out["verdict"] = "TOO_FEW_SCORED_OBJECTS"
        return out
    out["z_quantiles"] = {f"q{q}": float(np.nanquantile(z[finite], q))
                          for q in (0.5, 0.9, 0.99, 0.999, 1.0)}
    out["median_dispersion"] = float(np.nanmedian(disp[finite]))
    out["n_above_threshold"] = int(np.count_nonzero(z[finite] >= th.min_rejection_z))

    # Empirical null on the maximum, via matched draws over the same strata.
    rng = np.random.default_rng(th.seed)
    mask = np.zeros(z.size, dtype=bool)
    top = np.argsort(np.where(finite, z, -np.inf))[-max(
        1, out["n_above_threshold"]):]
    mask[top] = True
    draws = matched_draws(labels, mask & finite, int(th.n_null), rng)
    null_max = np.array([float(np.nanmax(z[draws[i] & finite]))
                         if np.any(draws[i] & finite) else np.nan
                         for i in range(int(th.n_null))])
    null_max = null_max[np.isfinite(null_max)]
    if null_max.size >= 20:
        z_obs = float(np.nanmax(z[finite]))
        out["z_max"] = z_obs
        out["p_max_matched"] = float((int((null_max >= z_obs).sum()) + 1)
                                     / (null_max.size + 1))
        out["p_resolution_floor"] = 1.0 / (null_max.size + 1)
        out["null_max_median"] = float(np.median(null_max))
    else:
        out["note_null"] = "matched null did not populate"
    out["verdict"] = "OK"
    return out


def quality_independence(records: list[RejectionRecord],
                         covariates: tuple[str, ...] = DEFAULT_COVARIATES) -> dict:
    """Does the excess track any single covariate by rank?

    The failure this prevents is the one KNELL documents and every blind search
    walks into: rank objects by anomaly and you rank them by how badly they were
    observed.  Rank correlation rather than Pearson, because every covariate here
    is heavily skewed and one outer-belt object would otherwise set the slope.
    """
    z = np.array([r.excess_z for r in records], dtype=float)
    out: dict = {"n": int(np.isfinite(z).sum()), "correlations": {}}
    if np.isfinite(z).sum() < 20:
        out["verdict"] = "TOO_FEW_OBJECTS"
        return out
    worst, worst_name = 0.0, None
    for name in covariates:
        v = np.array([getattr(r, name, float("nan")) for r in records], dtype=float)
        m = np.isfinite(z) & np.isfinite(v)
        if m.sum() < 20:
            out["correlations"][name] = None
            continue
        rx = np.argsort(np.argsort(z[m])).astype(float)
        ry = np.argsort(np.argsort(v[m])).astype(float)
        if np.std(rx) <= 0 or np.std(ry) <= 0:
            out["correlations"][name] = None
            continue
        r = float(np.corrcoef(rx, ry)[0, 1])
        out["correlations"][name] = r
        if abs(r) > abs(worst):
            worst, worst_name = r, name
    out["max_abs_correlation"] = abs(worst)
    out["max_correlated_with"] = worst_name
    out["verdict"] = "OK"
    return out


# ---------------------------------------------------------------------------
# 3. Binaries: the contaminant that produces the exact signal
# ---------------------------------------------------------------------------
# An unresolved binary asteroid's photocentre wobbles about the barycentre, which
# is an anomalous astrometric residual with the right amplitude, the right
# timescale and the right dataset.  It is simultaneously the nearest prior art
# and the contamination catalogue.
#
#   Liberato, Tanga et al., "Binary asteroid candidates in Gaia DR3 astrometry",
#   A&A 688, A50 (2024) = arXiv:2406.07195   -- VizieR J/A+A/688/A50
#   Liberato, Tanga, Mary et al., "Follow the wobble: ... Gaia FPR",
#   arXiv:2605.22702 (2026)                  -- 343 candidates on THIS table
#
# VizieR is unreachable from the sandbox, so the catalogue is an INPUT: the
# runner pulls it (docs/sextant.md gives the exact query) and hands it in as a
# list of rows.  Structuring it that way is not a convenience --- it means the
# rejection path is exercised by the offline tests with a synthetic catalogue,
# instead of being a code path that has never run.
#
# There is also a rejection that needs no catalogue at all, and it is the
# stronger one: a photocentre wobble is BOUNDED and PERIODIC.  Its amplitude
# cannot exceed the primary-to-barycentre distance, which for a small-body binary
# is at most a few hundred kilometres, and it does not grow.  A secular quadratic
# displacement over a six-year arc is therefore not a wobble whatever the
# catalogue says --- which is why the screen keeps the amplitude bound as a
# first-class test rather than relying on catalogue completeness.
#: Generous upper bound on a photocentre-barycentre offset for a small-body
#: binary, in km.  The mutual orbit of a ~100 km primary with a large secondary
#: puts the photocentre at most a few hundred km from the barycentre; 1000 km is
#: chosen to be indefensibly generous, so exceeding it is a statement about the
#: object.
MAX_PHOTOCENTRE_OFFSET_KM = 1000.0


@dataclass
class BinaryCatalogue:
    """Known and candidate binary asteroids, as an input rather than a fetch."""

    rows: list[dict] = field(default_factory=list)
    reference: str = ("Liberato et al. 2024, A&A 688, A50 (arXiv:2406.07195), "
                      "VizieR J/A+A/688/A50; extended by Liberato et al. 2026, "
                      "arXiv:2605.22702 (Gaia FPR)")
    retrieved_utc: str | None = None

    def _index(self) -> tuple[set[int], set[str]]:
        nums: set[int] = set()
        names: set[str] = set()
        for r in self.rows:
            for k in ("number_mp", "Number", "num", "number"):
                v = r.get(k)
                if v is None:
                    continue
                try:
                    nums.add(int(float(v)))
                except (TypeError, ValueError):
                    pass
            for k in ("denomination", "Name", "name", "designation"):
                v = r.get(k)
                if v:
                    names.add(str(v).strip().lower())
        return nums, names

    def match(self, rec: RejectionRecord) -> bool:
        """Is this object in the catalogue?  Matched on number first, then name."""
        nums, names = self._index()
        if math.isfinite(rec.number_mp) and int(rec.number_mp) in nums:
            return True
        if rec.denomination and str(rec.denomination).strip().lower() in names:
            return True
        return False


def photocentre_bound(series: ResidualSeries, amplitude_au_day2: float,
                      max_offset_km: float = MAX_PHOTOCENTRE_OFFSET_KM) -> dict:
    """Could a bounded photocentre wobble produce this much displacement?

    The fitted force amplitude implies a physical along-track displacement over
    the arc.  A binary's photocentre cannot supply more than
    ``max_offset_km``, and unlike an acceleration it does not accumulate.  So a
    displacement larger than the bound is not a wobble, and a displacement
    smaller than it is not evidence of anything --- which is the honest reading
    and the one that stops this channel from rediscovering 343 binaries.
    """
    out: dict = {"max_offset_km": float(max_offset_km)}
    m = series.usable()
    if not m.any() or not math.isfinite(amplitude_au_day2):
        out["verdict"] = "NOT_EVALUABLE"
        return out
    col = series.signal_columns.get("radiation")
    if col is None:
        out["verdict"] = "NO_SIGNAL_COLUMN"
        return out
    mas = np.abs(np.asarray(col, dtype=float)[m] * amplitude_au_day2)
    span = float(np.nanmax(mas) - np.nanmin(mas)) if mas.size else float("nan")
    out["implied_along_scan_span_mas"] = span
    delta = series.delta_au[m]
    med = float(np.nanmedian(delta)) if delta.size else float("nan")
    from .residuals import AU_KM, MAS_PER_RAD
    out["implied_displacement_km"] = (span / MAS_PER_RAD * med * AU_KM
                                      if math.isfinite(span) and math.isfinite(med)
                                      else float("nan"))
    d = out["implied_displacement_km"]
    if not math.isfinite(d):
        out["verdict"] = "NOT_EVALUABLE"
        return out
    out["ratio_to_bound"] = d / float(max_offset_km)
    out["verdict"] = ("EXCEEDS_PHOTOCENTRE_BOUND" if d > float(max_offset_km)
                      else "WITHIN_PHOTOCENTRE_BOUND")
    return out


# ---------------------------------------------------------------------------
# 4. Stage two: what does the surviving astrometry of a flagged object say?
# ---------------------------------------------------------------------------
def characterise(rec: RejectionRecord, series: ResidualSeries, th: Thresholds,
                 binaries: BinaryCatalogue | None = None) -> RejectionRecord:
    """Run the law discriminator on a shortlisted object and record the outcome.

    This is the second novel axis and it is deliberately downstream of the
    rejection screen: the discard pattern says *which objects to look at*, and
    the model comparison says *what the look means*.  Neither is a threshold on
    an amplitude, which is what both published treatments of this table are.
    """
    reasons: list[str] = list(rec.reasons)
    part = scan_axis_partition(series)
    rec.scan_anisotropy = part.get("sigma_ratio_ac_over_al", float("nan"))
    if (math.isfinite(rec.scan_anisotropy)
            and rec.scan_anisotropy < th.min_scan_anisotropy):
        reasons.append(f"scan_anisotropy_{rec.scan_anisotropy:.1f}_below_"
                       f"{th.min_scan_anisotropy}__error_model_not_as_documented")

    mc = model_comparison(series, min_observations=th.min_observations_for_model,
                          min_arc_days=th.min_arc_days,
                          min_delta_chi2=th.min_delta_chi2)
    rec.model_verdict = mc.get("verdict", "")
    rec.law_verdict = mc.get("law_verdict", "")
    rec.best_force_model = mc.get("best_force_model")
    rec.best_geometric_model = mc.get("best_geometric_model")
    rec.family_margin = float(mc.get("family_margin", float("nan")))
    bf = mc.get("best_force_model")
    if bf and bf in mc.get("fits", {}):
        f = mc["fits"][bf]
        rec.force_amplitude = float(f.get("amplitude", float("nan")))
        rec.force_snr = float(f.get("snr", float("nan")))
        rec.absorbed_fraction = float(f.get("absorbed_fraction", float("nan")))
    if (math.isfinite(rec.absorbed_fraction)
            and rec.absorbed_fraction > th.max_absorbed_fraction):
        reasons.append(f"absorbed_fraction_{rec.absorbed_fraction:.4f}__an_orbit_"
                       f"fit_could_have_removed_essentially_all_of_this")
    if rec.model_verdict == "GEOMETRIC_EXPLANATION_PREFERRED":
        reasons.append(f"geometric_explanation_preferred__{rec.best_geometric_model}")

    if binaries is not None and binaries.match(rec):
        rec.known_binary = True
        rec.binary_reference = binaries.reference
        reasons.append("known_or_candidate_binary__photocentre_wobble")
    pb = photocentre_bound(series, rec.force_amplitude)
    if pb.get("verdict") == "WITHIN_PHOTOCENTRE_BOUND":
        reasons.append("displacement_within_photocentre_wobble_bound")
    rec.reasons = _dedup(reasons)
    return rec


def assign_tier(rec: RejectionRecord, th: Thresholds) -> RejectionRecord:
    """Place a screened object on the tier ladder and name every reason.

    ``untestable`` is a first-class outcome and is never folded into
    ``ordinary``: "we could not measure this object's discard rate" and "we
    measured it and it was normal" are different statements, and the whole
    channel rests on not confusing them.
    """
    reasons: list[str] = list(rec.reasons)
    if rec.n_attempts < th.min_attempts or rec.n_transits < th.min_transits:
        rec.tier = "untestable"
        rec.reasons = _dedup([*reasons,
                              f"only_{rec.n_attempts}_attempts_in_"
                              f"{rec.n_transits}_transits"])
        return rec
    if not math.isfinite(rec.excess_z):
        rec.tier = "untestable"
        rec.reasons = _dedup([*reasons, "stratum_too_sparse_to_calibrate_an_excess"])
        return rec
    if rec.excess_z < th.min_rejection_z:
        rec.tier = "ordinary"
        rec.reasons = _dedup(reasons)
        return rec

    # From here the object HAS an excess.  Everything below decides whether the
    # excess is the object's or the observing conditions'.
    if (math.isfinite(rec.covariate_survival)
            and rec.covariate_survival < th.min_covariate_survival):
        reasons.append(f"excess_collapses_when_{rec.binding_covariate}_leaves_the_"
                       f"null__survival_{rec.covariate_survival:.2f}")
        rec.tier = "watch"
        rec.reasons = _dedup(reasons)
        return rec
    if (math.isfinite(rec.abs_galactic_latitude_deg)
            and rec.abs_galactic_latitude_deg < th.min_abs_galactic_latitude_deg):
        reasons.append(f"galactic_latitude_{rec.abs_galactic_latitude_deg:.1f}deg"
                       f"__crowding_not_excluded")
        rec.tier = "watch"
        rec.reasons = _dedup(reasons)
        return rec

    systematic = bool(reasons)
    if systematic:
        rec.tier = "watch"
        rec.reasons = _dedup(reasons)
        return rec

    rec.tier = "interest"
    reasons.append(f"rejection_excess_z_{rec.excess_z:.1f}_survives_every_"
                   f"covariate_control")

    # Promotion to candidate needs the SECOND axis to speak.  A discard-rate
    # excess alone is an anomaly detector; it is not an explanation, and
    # promoting on it would be exactly the "large number, therefore interesting"
    # move this repository refuses everywhere else.
    force_ok = (rec.model_verdict == "FORCE_LAW_PREFERRED"
                and math.isfinite(rec.force_snr) and rec.force_snr >= th.min_force_snr
                and math.isfinite(rec.absorbed_fraction)
                and rec.absorbed_fraction <= th.max_absorbed_fraction)
    if force_ok and not rec.known_binary:
        rec.tier = "candidate"
        reasons.append(f"surviving_astrometry_prefers_a_force_law__"
                       f"{rec.best_force_model}_over_{rec.best_geometric_model}")
        if rec.law_verdict == "LAW_PREFERRED":
            reasons.append(f"law_discriminated__{rec.best_force_model}")
        else:
            reasons.append(f"law_not_discriminated__{rec.law_verdict}")
    rec.reasons = _dedup(reasons)
    return rec


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# 5. The decision is on the SET
# ---------------------------------------------------------------------------
def population_decision(records: list[RejectionRecord], th: Thresholds,
                        tiers: tuple[str, ...] = ("interest", "candidate")) -> dict:
    """Does the SET of flagged objects have the structure replication implies?

    LOOM's premise, unweakened: a self-replicating probe is *defined* by a
    population sharing an origin, so the claim is never about one object.  The
    machinery is reused rather than reimplemented --- element clustering,
    orbital-pole coherence (an orientation tensor, because poles are axes and a
    vector mean would cancel antipodal ones), inclination isotropy, resonance
    concentration --- each against **matched random subsets of this screened
    sample**, with the MPC mis-linkage collapse running before any statistic sees
    the set.

    The matching covariates are SEXTANT's, not LOOM's, and that substitution is
    the only change: an object flagged here is preferentially fast-moving, faint
    and near the galactic plane, and objects like that are not uniformly
    distributed in the belt.  Matching on the covariates of the *rejection rate*
    is what stops the population statistics from firing on the scanning law.
    """
    rows = [{"a": r.a, "e": r.e, "i": r.i, "node": r.node, "h": r.h,
             "mjd_min": r.mjd_min, "mjd_max": r.mjd_max, "key": r.key}
            for r in records]
    mask = np.array([r.tier in tiers for r in records], dtype=bool)
    labels = covariate_strata(records, DEFAULT_COVARIATES, th.n_covariate_bins)
    out = replication_tests(rows, mask, labels, n_null=th.population_n_null,
                            seed=th.seed)
    out["matched_on"] = list(DEFAULT_COVARIATES)
    out["note_matching"] = (
        "strata are SEXTANT's rejection-rate covariates (apparent motion, "
        "galactic latitude, magnitude, phase coverage, attempts), not LOOM's "
        "orbit-quality ones; an object flagged by a discard-rate excess is "
        "preferentially fast, faint and crowded, and those are not uniformly "
        "distributed in the belt")
    if "photometric_homogeneity" in str(out.get("tests", "")):
        out["note_photometry"] = (
            "the photometric homogeneity test needs per-band absolute magnitudes, "
            "which gaiafpr.sso_observation does not carry; it degrades rather "
            "than silently passing")
    return out


# ---------------------------------------------------------------------------
# 6. The orchestrator
# ---------------------------------------------------------------------------
def screen(records: list[RejectionRecord], th: Thresholds, *,
           series_by_key: dict[str, ResidualSeries] | None = None,
           binaries: BinaryCatalogue | None = None,
           covariates: tuple[str, ...] = DEFAULT_COVARIATES) -> dict:
    """Run the whole screen and return the funnel, the excess report and the tiers.

    Order matters and is not negotiable: the sample-wide rate is checked against
    the published outlier fractions *first*, because if the flags are not being
    read as the mission means them then every number after that is a
    misinterpretation with error bars.
    """
    out: dict = {"n_objects": len(records)}
    out["sample"] = sample_rejection_summary(records)
    if out["sample"].get("verdict") != "OK":
        out["verdict"] = out["sample"]["verdict"]
        out["note"] = out["sample"].get("note")
        return out

    out["excess"] = rejection_excess(records, th, covariates=covariates)
    out["quality_independence"] = quality_independence(records, covariates)
    qi = out["quality_independence"]
    if (qi.get("verdict") == "OK"
            and qi.get("max_abs_correlation", 0.0) > th.max_quality_correlation):
        out["quality_warning"] = (
            f"the excess correlates with {qi['max_correlated_with']} at rank "
            f"rho={qi['max_abs_correlation']:.2f}; above "
            f"{th.max_quality_correlation} the screen is ranking objects by how "
            f"they were observed, not by what they are")

    series_by_key = series_by_key or {}
    for r in records:
        assign_tier(r, th)
    # Only shortlisted objects get the (expensive) second stage.  Characterising
    # everything would be honest but pointless: the model comparison needs a long
    # arc and many surviving observations, which most objects do not have.
    for r in records:
        if r.tier in ("interest", "candidate"):
            s = series_by_key.get(r.key)
            if s is None:
                r.reasons = _dedup([*r.reasons, "no_residual_series_supplied__"
                                    "second_stage_not_run"])
                continue
            characterise(r, s, th, binaries=binaries)
            assign_tier(r, th)

    funnel = {f"n_{t}": sum(1 for r in records if r.tier == t) for t in TIERS}
    funnel["n_known_binary"] = sum(1 for r in records if r.known_binary)
    funnel["n_characterised"] = sum(1 for r in records if r.model_verdict)
    out["funnel"] = funnel
    out["tiers"] = {t: [r.key for r in records if r.tier == t] for t in TIERS}

    if (len(records) < th.min_parent_for_population
            or funnel["n_interest"] + funnel["n_candidate"]
            < th.min_anomalies_for_population):
        out["population"] = {
            "verdict": "INSUFFICIENT_POPULATION",
            "note": (f"the population stage needs >= "
                     f"{th.min_parent_for_population} screened objects and >= "
                     f"{th.min_anomalies_for_population} flagged ones; below "
                     f"that the matched null cannot be populated and any p-value "
                     f"would be noise")}
    else:
        out["population"] = population_decision(records, th)

    pv = out["population"].get("verdict")
    if pv == "REPLICATION_STRUCTURE_DETECTED":
        out["verdict"] = "REPLICATION_STRUCTURE_DETECTED"
    elif funnel["n_candidate"]:
        out["verdict"] = "CANDIDATES_WITHOUT_POPULATION_STRUCTURE"
    elif funnel["n_interest"]:
        out["verdict"] = "REJECTION_EXCESS_UNEXPLAINED"
    else:
        out["verdict"] = "NO_UNEXPLAINED_EXCESS"
    return out

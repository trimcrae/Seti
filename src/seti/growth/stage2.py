"""GROWTH stage 2 --- measure the TESS depth from the LIGHT CURVE, not a catalogue.

Why this module exists
----------------------
Stage 1 (:mod:`seti.growth.drift`) compares two **heterogeneous catalogue**
numbers: the Kepler-era ``cumulative.koi_depth`` (a DV fit on Q1--Q17 DR25
long-cadence PDCSAP) against the TESS-era ``toi.pl_trandep`` (whichever of
SPOC / TESS-SPOC / QLP delivered the TOI, on a different cadence, with a
different crowding model, revised as sectors accumulate).  Run 35038510064
returned exactly one ``GROWTH_CANDIDATE`` --- Kepler-718 b (K00897.01, KIC
7849854, TIC 268924036, TOI 4490.01), 14,281 +/- 18 ppm in the Kepler era
against 34,476 +/- 2,349 ppm in the TOI table, ``z = 8.67`` --- and that run's
own population says the ``z`` cannot be read as a sigma count: **31 of 108
measured planets sit above 5 sigma and 37 above 3 sigma** of the population
median.  A sample in which a third of the objects are five-sigma outliers has
formal errors that do not describe the pipeline-to-pipeline scatter.

So stage 2 does not argue about the error model.  It **measures the TESS depth
itself** and then compares THREE numbers instead of two:

======================================  =========================================
``depth_kepler_ppm``                    the KOI catalogue depth (Kepler band)
``depth_toi_ppm``                       the TOI catalogue depth (TESS band)
``depth_measured_ppm``                  this module's fit to the TESS light curve
======================================  =========================================

with the Kepler depth carried into the TESS band by the limb-darkening band
ratio :func:`seti.growth.drift.band_ratio` before it is compared (the two
catalogues are in different bandpasses; comparing them raw would charge the
band difference to "growth").  The outcome is one of

``MEASURED_DEPTH_MATCHES_KEPLER``
    the measured TESS depth agrees with the Kepler-era depth and disagrees with
    the TOI value: **the TOI number was wrong**, this candidate dies, and the
    channel has learned that its stage-1 error model is catalogue-driven.
``MEASURED_DEPTH_MATCHES_TOI``
    the measured depth agrees with the deep TOI value and disagrees with the
    Kepler-era depth: the depth really did change, and the candidate survives
    to the centroid test (an eclipsing binary on a different star inside the
    TESS pixel makes a transit DEEPER and is *not* excluded by anything here).
``MEASURED_DEPTH_MATCHES_NEITHER``
    the measurement sits outside both.  Said, not resolved.
``MEASURED_DEPTH_MATCHES_BOTH``
    the two catalogue values are not separated by this measurement's precision.
    This is the "cannot distinguish" branch, and it is **never** a pick: it is
    reported with ``z_toi_vs_kepler`` so a reader can see whether the failure
    is the measurement's precision or the references' own agreement.
``UNMEASURED``
    no depth was fitted.  ``unmeasured_reason`` keeps ``QUERY_FAILED``
    (the archive did not answer) and ``QUERY_RETURNED_ZERO_ROWS`` (it answered
    with nothing) apart, and adds ``NO_USABLE_TRANSIT`` / ``BUDGET_EXHAUSTED``
    / ``EPHEMERIS_UNAVAILABLE``.  **A target whose light curve could not be
    fetched is never reported as agreeing with anything.**

The ephemeris, and the one arithmetic that must not be got wrong
----------------------------------------------------------------
The period and epoch are the KOI's own (``koi_period``, ``koi_time0bk``), and
the two missions count days from different zero points:

* Kepler **BKJD** = BJD - 2454833.0
* TESS  **BTJD** = BJD - 2457000.0

so ``t_BTJD = t_BKJD - 2167.0`` exactly (:data:`BKJD_MINUS_BTJD`).  A KOI epoch
of 170.0 BKJD is BJD 2455003.0 is **-1997.0 BTJD** --- the hand-worked value the
test suite checks.  The epoch is then propagated forward by an integer number
of periods to the TESS window, and the accumulated uncertainty
``sqrt(sigma_T0^2 + (n sigma_P)^2)`` is reported in minutes beside the
duration: an ephemeris whose drift is a sizeable fraction of the transit makes
a fitted depth shallow, and that has to be visible rather than absorbed.

The fit
-------
Deliberately simple, and fixed to the KOI solution --- this module measures a
**depth**, it does not re-derive an ephemeris.  Per target:

1. each sector is normalised separately by its own robust median;
2. every predicted transit gets a local window ``|dt| <= w T14``; the baseline
   is fitted as a straight line in ``dt`` over the out-of-transit part of that
   window only (the transit is **masked**, with a guard band);
3. the depth of that transit is ``1 - <flux/baseline>`` over a **core** window
   ``|dt| <= f T14 / 2`` shrunk by half the exposure time, so a point whose
   integration straddles ingress is not counted as flat-bottom flux;
4. per-transit depths are combined by inverse variance; the quoted error is a
   **bootstrap over transits** when there are enough of them and the analytic
   propagation otherwise, and ``depth_err_method`` always says which;
5. the depth is reported **per sector** as well as combined --- a depth that
   differs between sectors is a systematic, not growth --- along with the
   out-of-transit scatter and the **odd-even** depth difference, which is the
   classic eclipsing-binary signature and is nearly free once the fold exists.

A cadence that smears the transit (30-minute FFI photometry on a 2-hour
transit) is **flagged**, never quietly corrected: ``smeared`` and
``exptime_over_duration`` are columns, and ``depth_is_lower_bound`` marks the
case where no flat core survives the integration at all.

The asymmetry this module used to carry, and the like-for-like verdict
--------------------------------------------------------------------
Run 35041932130 measured the TESS era from the light curve and then compared it
against ``koi_depth`` --- **a catalogue number**, produced by a different
pipeline, in a different decade, through a different aperture.  That is exactly
the heterogeneity that broke stage 1's error model, so the comparison inherited
the flaw it was built to remove.  Stage 2 therefore measures **both eras with
the same fitter**: the same fold, the same locally fitted transit-masked
baseline, the same core-window shrink, the same bootstrap over transits, the
same :func:`dedupe_sectors`.  Only the archive product and the time system
differ.  Two more numbers join the table:

======================================  =========================================
``depth_kepler_measured_ppm``           this module's fit to the KEPLER light curve
``depth_kepler_measured_in_tess_band_ppm``  the same, carried into the TESS band
======================================  =========================================

and the **primary** verdict is now the like-for-like one
(:data:`LIKE_FOR_LIKE_VERDICTS`):

``MEASURED_DEPTH_CHANGED``
    our Kepler-era fit and our TESS-era fit disagree at or beyond ``n_agree``
    sigma once the band ratio is applied.  The change survives a comparison in
    which nothing but the sky differs, and is much harder to dismiss.
``MEASURED_DEPTH_UNCHANGED``
    they agree, **and** the comparison had the power to have seen the change
    under test.  Nothing changed: the KOI catalogue depth is simply wrong for
    this object, the candidate dies, and stage 1 has learned something about
    its own inputs.
``MEASURED_DEPTH_UNRESOLVED``
    they agree, but the comparison could not have detected the change being
    claimed (``detectable_ln_ratio > ln_ratio_under_test``).  An agreement
    without power is not a refutation and is never reported as one.
``MEASURED_DEPTH_ERA_UNMEASURED``
    one or both eras produced no depth.  ``like_for_like_unmeasured_reason``
    names which era and why, and **an era that was not measured never agrees
    with the other one.**

The old catalogue-vs-measurement verdicts stay, and are now a diagnostic of the
**CATALOGUES** rather than of the sky: ``MEASURED_DEPTH_MATCHES_TOI`` says the
TOI table is right about the TESS era, ``MEASURED_DEPTH_MATCHES_KEPLER`` says
the KOI table is right about it.  ``summary.json`` says which verdict is
primary.  Separately, our Kepler-era fit is compared with ``koi_depth`` **in
the Kepler band, with no band ratio** --- if they disagree at high significance
that is a finding about the KOI table (``KOI_DEPTH_CONTRADICTED``) and it is
reported as one, not buried.

Time systems, and the bug this arrangement invites
--------------------------------------------------
``koi_time0bk`` is in BKJD.  **Kepler light curves are also in BKJD**, so the
Kepler era applies *no* conversion; the TESS era subtracts 2167 days.  Applying
the shift on the Kepler side too is the obvious bug --- it would put the fold
2167 days from any transit and return a depth of zero --- so the choice is made
by one named function, :func:`epoch_in_era`, and the test suite checks both
branches.

The reduction ensemble, and the error that was not describing the scatter
------------------------------------------------------------------------
Run 35041932130 measured the **same seven TESS sectors twice**, once from the
2-minute SPOC reduction and once from the FFI TESS-SPOC reduction of the *same
pixels* (``results/growth/stage2/sectors.csv`` at commit ``050402a``).  SPOC
gave 25,425 to 35,498 ppm; TESS-SPOC gave 33,078 to 39,798 ppm.  **Two
reductions of identical photons disagreeing by 20 to 30 % is a systematic far
larger than the 1,521 ppm error the bootstrap quotes**, and at G = 15.23 the
out-of-transit scatter is 414 ppm in Kepler against 67,631 ppm in TESS --- a
factor of 163.  Stage 1's whole failure was a quoted error that did not
describe the real scatter; repeating it one level down, on a single object,
would be worse.

So stage 2 does not pick a reduction.  It measures the depth under **every
available reduction** of the same data --- every (pipeline author, flux column)
combination the archive serves for that era --- and reports the spread over
that ensemble as a systematic that **enters the comparison**:

======================================  =========================================
``depth_reduction_spread_ppm``          half the 16th-to-84th percentile range
                                        over the ensemble members
``depth_measured_total_err_ppm``        ``sqrt(stat^2 + reduction_spread^2)``
``sap_minus_pdcsap_ppm``                the background/crowding test, by number
======================================  =========================================

and **the like-for-like verdict uses the TOTAL error**.  A ratio that survives
the ensemble is a much stronger statement than one that survives only the
bootstrap; a ratio that does not survive it comes back
``MEASURED_DEPTH_UNRESOLVED``, and that is the correct answer rather than a
disappointment.

Three things about this that are easy to get wrong, and are therefore named:

* **the ensemble never moves the primary depth.**  :func:`dedupe_sectors` still
  governs the measurement: one sector is counted once, by the best-cadence
  pipeline.  The ensemble is a *systematic estimate* --- it sets the error, and
  only the error.  Re-stacking the duplicates into the depth is exactly the
  double-counting that dedupe exists to stop.
* **a member that could not be fetched is recorded, not dropped quietly.**  An
  ensemble of two that was meant to be six has a smaller spread than the truth,
  so every unavailable member keeps its own reason (``QUERY_FAILED`` /
  ``QUERY_RETURNED_ZERO_ROWS`` / ``FLUX_COLUMN_NOT_PRESENT`` /
  ``BUDGET_EXHAUSTED``, all different facts) and the era is flagged
  ``ensemble_incomplete``.
* **``SAP_FLUX`` versus ``PDCSAP_FLUX`` is measured, never skipped.**  PDC is
  where the crowding and background corrections are applied, so a faint star in
  a crowded aperture with an over-subtracted background yields an *inflated*
  transit depth --- a specific, testable, entirely mundane explanation for this
  whole result.  It is answered with a number
  (``sap_minus_pdcsap_ppm`` and its significance) and a named finding when the
  two disagree beyond their errors.

Everything that touches a service takes an injectable callable (``query_fn``
for the Exoplanet Archive, ``lc_fn`` for the TESS light curves, ``kepler_lc_fn``
for the Kepler ones, ``ensemble_lc_fn`` / ``kepler_ensemble_lc_fn`` for the
reduction ensemble) and carries a **wall-clock budget**, because an unbounded
stage has already cost this channel a whole run (``config/growth.yaml``,
``gaia.cone_budget_s``).  Nothing here reaches the network inside a test.
"""

from __future__ import annotations

import argparse
import json
import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import STATUS_FAILED, STATUS_OK, STATUS_ZERO, AcquisitionLog, tap_sync
from .drift import band_ratio, sym_err

# ---------------------------------------------------------------------------
# Time systems.  BKJD = BJD - 2454833; BTJD = BJD - 2457000.
# ---------------------------------------------------------------------------
#: Kepler barycentric Julian date offset (``koi_time0bk`` is in BKJD).
BKJD_OFFSET = 2454833.0
#: TESS barycentric Julian date offset (TESS light-curve ``TIME`` is in BTJD).
BTJD_OFFSET = 2457000.0
#: ``t_BTJD = t_BKJD + BKJD_MINUS_BTJD``.  Exactly -2167.0 days.
BKJD_MINUS_BTJD = BKJD_OFFSET - BTJD_OFFSET

#: The two eras this stage fits with the SAME fitter.  The era decides one
#: thing only: the time system the light curves are in (and therefore whether
#: ``koi_time0bk`` needs converting at all -- see :func:`epoch_in_era`).
ERA_KEPLER = "kepler"
ERA_TESS = "tess"
ERAS = (ERA_KEPLER, ERA_TESS)

# ---------------------------------------------------------------------------
# Verdict vocabulary.  Per target (three-way), then per run.
# ---------------------------------------------------------------------------
MATCH_KEPLER = "MEASURED_DEPTH_MATCHES_KEPLER"
MATCH_TOI = "MEASURED_DEPTH_MATCHES_TOI"
MATCH_NEITHER = "MEASURED_DEPTH_MATCHES_NEITHER"
MATCH_BOTH = "MEASURED_DEPTH_MATCHES_BOTH"
MATCH_UNMEASURED = "UNMEASURED"
TARGET_VERDICTS = (MATCH_KEPLER, MATCH_TOI, MATCH_NEITHER, MATCH_BOTH, MATCH_UNMEASURED)

#: Why a target has no depth.  ``QUERY_FAILED`` (the service errored) and
#: ``QUERY_RETURNED_ZERO_ROWS`` (it answered with nothing) are different facts
#: and are never collapsed.
UNMEASURED_QUERY_FAILED = STATUS_FAILED
UNMEASURED_ZERO_ROWS = STATUS_ZERO
UNMEASURED_NO_EPHEMERIS = "EPHEMERIS_UNAVAILABLE"
UNMEASURED_NO_TRANSIT = "NO_USABLE_TRANSIT"
UNMEASURED_BUDGET = "BUDGET_EXHAUSTED"
#: Neither catalogue depth is usable, so there is nothing to agree or disagree
#: with.  The measured depth is still reported --- what is missing is the
#: comparison, not the measurement, and "matches neither" would be a false
#: statement when there is no "neither" to match.
UNMEASURED_NO_REFERENCE = "NO_REFERENCE_DEPTH"
#: The Kepler era was switched off in config (``stage2.mast.kepler_enabled``).
#: It is a SEPARATE fact from "the archive did not answer": nothing was asked.
#: ``MastParams.kepler_enabled`` defaults to False so that a caller which has
#: not thought about the Kepler era cannot silently reach the network; the
#: repository config turns it on, and ``growth_stage2.yml`` fails the run if the
#: summary comes back with the primary comparison switched off.
UNMEASURED_KEPLER_DISABLED = "KEPLER_ERA_NOT_ATTEMPTED"
UNMEASURED_REASONS = (UNMEASURED_QUERY_FAILED, UNMEASURED_ZERO_ROWS, UNMEASURED_NO_EPHEMERIS,
                      UNMEASURED_NO_TRANSIT, UNMEASURED_BUDGET, UNMEASURED_NO_REFERENCE,
                      UNMEASURED_KEPLER_DISABLED)

RUN_NO_DATA = "NO_DATA_REACHED"
RUN_REFUTED = "STAGE1_DEPTH_CHANGE_REFUTED"
RUN_CONFIRMED = "STAGE1_DEPTH_CHANGE_CONFIRMED"
RUN_UNRESOLVED = "STAGE1_DEPTH_CHANGE_UNRESOLVED"
RUN_VERDICTS = (RUN_NO_DATA, RUN_REFUTED, RUN_CONFIRMED, RUN_UNRESOLVED)

# ---------------------------------------------------------------------------
# THE PRIMARY VERDICT: our Kepler-era fit against our TESS-era fit.
# Same fitter, same fold, same baseline treatment, same bootstrap; only the
# archive product and the time system differ.  Nothing catalogue-derived enters
# it except the (fixed) ephemeris and the limb-darkening band ratio.
# ---------------------------------------------------------------------------
LL_CHANGED = "MEASURED_DEPTH_CHANGED"
LL_UNCHANGED = "MEASURED_DEPTH_UNCHANGED"
#: Agreement WITHOUT the power to have seen the change under test.  Not a
#: refutation, and never reported as one.
LL_UNRESOLVED = "MEASURED_DEPTH_UNRESOLVED"
LL_ERA_UNMEASURED = "MEASURED_DEPTH_ERA_UNMEASURED"
LIKE_FOR_LIKE_VERDICTS = (LL_CHANGED, LL_UNCHANGED, LL_UNRESOLVED, LL_ERA_UNMEASURED)

RUN_LL_NO_DATA = "LIKE_FOR_LIKE_NO_DATA"
RUN_LL_CHANGED = "MEASURED_DEPTH_CHANGE_CONFIRMED"
RUN_LL_UNCHANGED = "MEASURED_DEPTH_CHANGE_REFUTED"
RUN_LL_UNRESOLVED = "MEASURED_DEPTH_CHANGE_UNRESOLVED"
RUN_LL_VERDICTS = (RUN_LL_NO_DATA, RUN_LL_CHANGED, RUN_LL_UNCHANGED, RUN_LL_UNRESOLVED)

# ---------------------------------------------------------------------------
# Our Kepler-era fit against the KOI CATALOGUE depth.  Same band (both Kepler),
# so no band ratio is applied.  A disagreement here is a finding about the KOI
# table -- stage 1's own input -- and is reported as one.
# ---------------------------------------------------------------------------
KOI_DEPTH_CONFIRMED = "KOI_DEPTH_CONFIRMED"
KOI_DEPTH_CONTRADICTED = "KOI_DEPTH_CONTRADICTED"
KOI_DEPTH_UNCHECKED = "KOI_DEPTH_UNCHECKED"
KOI_DEPTH_VERDICTS = (KOI_DEPTH_CONFIRMED, KOI_DEPTH_CONTRADICTED, KOI_DEPTH_UNCHECKED)

# ---------------------------------------------------------------------------
# THE REDUCTION ENSEMBLE.  One member per (pipeline author, flux column); the
# spread over the members is the systematic that enters the total error.
# ---------------------------------------------------------------------------
#: The requested flux column is not in that product (asking QLP for
#: ``PDCSAP_FLUX``, or SPOC for ``KSPSAP_FLUX``).  This is a DIFFERENT fact from
#: "the archive did not answer" and from "the archive answered with nothing":
#: the product exists and simply does not carry that column, so the member is
#: absent by construction rather than by failure.  Silently letting such a
#: member fall back to whatever column the file does carry would put the SAME
#: reduction into the ensemble twice and shrink the spread.
ENSEMBLE_NO_FLUX_COLUMN = "FLUX_COLUMN_NOT_PRESENT"
#: The archive answered, but with ANOTHER pipeline's products.  Run bcc670c made
#: this real: QLP serves nothing for TIC 268924036, and ``lightkurve_lc_fn``
#: drops its author filter when nothing matches it (``if keep.any()``), so the
#: three QLP members came back carrying SPOC's light curves and entered the
#: ensemble as byte-identical copies of the SPOC members --- 29,255 / 11,196 ppm
#: twice over.  That double-weights one reduction in the percentile spread and
#: inflates ``sap_vs_pdcsap_n_pairs`` from 2 to 3, shrinking the very error the
#: ensemble exists to size.  Same failure as ``FLUX_COLUMN_NOT_PRESENT``, one
#: axis over, and it is caught the same way: the member is only measured if the
#: products REALLY carry the author that was asked for.
ENSEMBLE_NO_AUTHOR = "AUTHOR_NOT_SERVED"
#: The ensemble was switched off in config, or the era it belongs to produced no
#: depth at all.  Nothing was asked; that is not a failure either.
ENSEMBLE_NOT_ATTEMPTED = "ENSEMBLE_NOT_ATTEMPTED"
ENSEMBLE_MEMBER_STATUSES = (STATUS_OK, STATUS_FAILED, STATUS_ZERO, UNMEASURED_NO_TRANSIT,
                            UNMEASURED_BUDGET, ENSEMBLE_NO_FLUX_COLUMN, ENSEMBLE_NO_AUTHOR,
                            ENSEMBLE_NOT_ATTEMPTED)


def _same_author(a, b) -> bool:
    """Author names compared as the archive spells them, not byte for byte.

    ``TESS-SPOC``/``TESS_SPOC``/``tess spoc`` are the same pipeline.  Only an
    actual mismatch may drop a member, so the normalisation is deliberately
    generous; what the fetch really returned is kept in ``authors_returned``.
    """
    norm = lambda s: "".join(ch for ch in str(s).upper() if ch.isalnum())  # noqa: E731
    return norm(a) == norm(b)

#: The spread over the ensemble was computed.
SPREAD_OK = "OK"
#: Fewer members came back than ``min_members_for_spread``: there IS no spread
#: to quote.  It is reported as NaN with this status rather than as zero --- a
#: zero spread would say "every reduction agreed" when in truth only one was
#: reached, which is precisely the overconfidence this ensemble exists to stop.
SPREAD_TOO_FEW_MEMBERS = "TOO_FEW_MEMBERS"
SPREAD_NOT_ATTEMPTED = ENSEMBLE_NOT_ATTEMPTED
SPREAD_STATUSES = (SPREAD_OK, SPREAD_TOO_FEW_MEMBERS, SPREAD_NOT_ATTEMPTED)

#: ``SAP_FLUX`` vs ``PDCSAP_FLUX``: the direct test of the hypothesis that the
#: PDC background/crowding correction is what inflates the depth.
BG_AGREE = "BACKGROUND_TEST_AGREES"
BG_DISAGREE = "BACKGROUND_TEST_DISAGREES"
BG_UNAVAILABLE = "BACKGROUND_TEST_UNAVAILABLE"
BACKGROUND_VERDICTS = (BG_AGREE, BG_DISAGREE, BG_UNAVAILABLE)
#: Which way round the disagreement goes.  ``PDC_DEEPER_THAN_SAP`` is the
#: mundane explanation under test: the corrected photometry is deeper than the
#: raw aperture photometry, i.e. the correction is adding depth.
BG_PDC_DEEPER = "PDC_DEEPER_THAN_SAP"
BG_SAP_DEEPER = "SAP_DEEPER_THAN_PDC"

STAGES = ("probe", "measure", "assess")

#: Light-curve authors, best cadence first.  SPOC 2-minute where it exists,
#: TESS-SPOC / QLP FFI photometry otherwise --- and which one supplied each
#: sector is recorded per target, because a 30-minute FFI depth on a 2-hour
#: transit is a different measurement from a 2-minute one.
DEFAULT_AUTHORS: tuple[str, ...] = ("SPOC", "TESS-SPOC", "QLP")

#: Flux columns tried in a TESS light-curve FITS product, in order.  SPOC and
#: TESS-SPOC give ``PDCSAP_FLUX``; QLP has used ``KSPSAP_FLUX`` and, in later
#: deliveries, ``DET_FLUX``.  Which one was read is recorded per sector.
FLUX_COLUMNS: tuple[str, ...] = ("PDCSAP_FLUX", "KSPSAP_FLUX", "DET_FLUX", "SAP_FLUX")

#: Kepler light-curve authors at MAST.  There is only one official reduction
#: (the Kepler pipeline itself); the community products (``K2SFF`` and friends)
#: are K2, not Kepler, and are deliberately not asked for -- the point of this
#: stage is that BOTH eras go through THIS fitter, not another pipeline's.
KEPLER_AUTHORS: tuple[str, ...] = ("Kepler",)

#: Flux columns in a Kepler light-curve FITS product, in order.  PDCSAP first,
#: exactly as on the TESS side; which one was read is recorded per quarter.
KEPLER_FLUX_COLUMNS: tuple[str, ...] = ("PDCSAP_FLUX", "SAP_FLUX")

#: Kepler quality is in ``SAP_QUALITY`` (TESS calls the same column ``QUALITY``).
KEPLER_QUALITY_COLUMNS: tuple[str, ...] = ("SAP_QUALITY", "QUALITY")

#: Kepler cadences, in seconds.  Long cadence is 270 co-added 6.02 s frames =
#: 1765.5 s = **29.4 minutes**, which on a 2.08-hour transit is 0.24 T14 --- over
#: ``fit.smear_fraction`` and therefore ``smeared``, which is exactly what the
#: ``core_half_width`` shrink and the flag exist for.  Short cadence is 9 frames
#: = 58.85 s and is preferred wherever it exists.
KEPLER_LONG_CADENCE_S = 1765.5
KEPLER_SHORT_CADENCE_S = 58.85
#: Exposures at or below this are called ``short``; above it, ``long``.
KEPLER_SHORT_CADENCE_MAX_S = 300.0

#: MAST product sub-groups for Kepler time series: long- and short-cadence
#: light curves.  ``LLC`` / ``SLC`` are the sub-group descriptions; target
#: pixel files (``TPF``) are not asked for.
KEPLER_PRODUCT_SUBGROUPS: tuple[str, ...] = ("LLC", "SLC")

ROUTE_LIGHTKURVE = "lightkurve"
ROUTE_MAST_FITS = "astroquery_mast_fits"

KOI_EPH_COLUMNS: tuple[str, ...] = (
    "kepoi_name", "kepid", "kepler_name", "koi_disposition", "koi_period", "koi_period_err1",
    "koi_period_err2", "koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2", "koi_duration",
    "koi_duration_err1", "koi_duration_err2", "koi_depth", "koi_depth_err1", "koi_depth_err2",
    "koi_ror", "koi_impact", "koi_dor", "koi_steff", "koi_slogg", "ra", "dec",
)
TOI_DEPTH_COLUMNS: tuple[str, ...] = (
    "toi", "tid", "tfopwg_disp", "pl_orbper", "pl_trandep", "pl_trandeperr1", "pl_trandeperr2",
    "pl_trandurh", "pl_trandurherr1", "pl_trandurherr2",
)


# ---------------------------------------------------------------------------
# Epoch arithmetic
# ---------------------------------------------------------------------------
def bkjd_to_btjd(t_bkjd):
    """Kepler BKJD -> TESS BTJD.  ``t - 2167.0`` (BJD - 2454833 -> BJD - 2457000)."""
    return np.asarray(t_bkjd, dtype=float) + BKJD_MINUS_BTJD if np.ndim(t_bkjd) else (
        float(t_bkjd) + BKJD_MINUS_BTJD)


def btjd_to_bkjd(t_btjd):
    """TESS BTJD -> Kepler BKJD."""
    return np.asarray(t_btjd, dtype=float) - BKJD_MINUS_BTJD if np.ndim(t_btjd) else (
        float(t_btjd) - BKJD_MINUS_BTJD)


def bkjd_to_bjd(t_bkjd):
    """Kepler BKJD -> full BJD."""
    return np.asarray(t_bkjd, dtype=float) + BKJD_OFFSET if np.ndim(t_bkjd) else (
        float(t_bkjd) + BKJD_OFFSET)


def btjd_to_bjd(t_btjd):
    """TESS BTJD -> full BJD."""
    return np.asarray(t_btjd, dtype=float) + BTJD_OFFSET if np.ndim(t_btjd) else (
        float(t_btjd) + BTJD_OFFSET)


def epoch_in_era(t0_bkjd: float, era: str) -> float:
    """``koi_time0bk`` expressed in the time system of ``era``'s light curves.

    This exists so the conversion is a *decision made once, by name*, rather
    than a shift copied into two call sites:

    * ``ERA_TESS`` --- TESS light curves are BTJD, so subtract 2167 days.
    * ``ERA_KEPLER`` --- **Kepler light curves are already BKJD, and so is**
      ``koi_time0bk``.  **No conversion at all.**

    Converting on the Kepler side too is the obvious bug in a two-era module:
    it would place the fold 2167 days from any transit and return a depth of
    zero (or nothing at all), which reads as "the Kepler era did not change"
    when in truth nothing was measured.  Both branches are tested.
    """
    e = str(era).lower()
    if e == ERA_KEPLER:
        return float(t0_bkjd)
    if e == ERA_TESS:
        return float(bkjd_to_btjd(t0_bkjd))
    raise ValueError(f"unknown era {era!r}; choose from {ERAS}")


def propagate_epoch(t0_btjd: float, period_days: float, t_ref_btjd: float, *,
                    t0_err_days: float = float("nan"),
                    period_err_days: float = float("nan")) -> dict:
    """Carry a transit epoch forward (or back) to the epoch nearest ``t_ref_btjd``.

    Returns the nearest mid-transit time, the integer number of periods
    crossed, and the **accumulated** ephemeris uncertainty
    ``sqrt(sigma_T0^2 + (n sigma_P)^2)`` in days and minutes.  The KOI epoch is
    ~2010 and the TESS window is ~2019--2026, so ``n`` is thousands for a
    short-period planet and a 1e-7 d period error still accumulates only
    seconds --- but it is reported rather than assumed.
    """
    p = float(period_days)
    if not (np.isfinite(t0_btjd) and np.isfinite(p) and p > 0):
        return {"t0_btjd": float("nan"), "n_epochs": 0, "sigma_days": float("nan"),
                "sigma_minutes": float("nan")}
    n = int(round((float(t_ref_btjd) - float(t0_btjd)) / p))
    t0n = float(t0_btjd) + n * p
    s0 = float(t0_err_days) if np.isfinite(t0_err_days) else 0.0
    sp = float(period_err_days) if np.isfinite(period_err_days) else 0.0
    sig = math.sqrt(s0 * s0 + (abs(n) * sp) ** 2)
    return {"t0_btjd": t0n, "n_epochs": n, "sigma_days": sig, "sigma_minutes": sig * 1440.0}


def fold(time_btjd, period_days: float, t0_btjd: float) -> tuple[np.ndarray, np.ndarray]:
    """``(epoch, dt)``: the integer transit number and the signed days from its centre."""
    t = np.asarray(time_btjd, dtype=float)
    p = float(period_days)
    epoch = np.round((t - float(t0_btjd)) / p)
    dt = t - (float(t0_btjd) + epoch * p)
    return epoch.astype(int), dt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class FitParams:
    """Everything the depth fit is allowed to decide (all from ``config/growth.yaml``)."""

    core_fraction: float = 0.7          # core window = core_fraction * T14, centred
    baseline_window_durations: float = 1.5   # local window half-width, in T14
    baseline_guard_durations: float = 0.75   # baseline must lie beyond this, in T14
    min_core_points: int = 2
    min_baseline_points: int = 8
    require_baseline_both_sides: bool = True
    smear_fraction: float = 0.2         # exptime > this * T14 -> `smeared`
    bootstrap_draws: int = 2000
    min_transits_for_bootstrap: int = 8
    err_method: str = "auto"            # auto | bootstrap | analytic | chi2_scaled
    fold_bins: int = 121
    odd_even_sigma: float = 3.0         # |odd - even| / sigma at or above this is flagged
    sector_scatter_chi2_per_dof: float = 3.0
    seed: int = 7

    @classmethod
    def from_config(cls, conf: dict | None) -> FitParams:
        f = ((conf or {}).get("stage2") or {}).get("fit") or {}
        d = cls()
        for k in d.__dataclass_fields__:
            if k in f and f[k] is not None:
                setattr(d, k, type(getattr(d, k))(f[k]))
        return d


@dataclass
class CompareParams:
    """The three-way comparison's thresholds."""

    n_agree: float = 3.0                # |z| below this is "agrees with"
    sigma_sys_ln: float = 0.05          # stated floor on ln-depth heterogeneity
    apply_band_ratio: bool = True       # carry the Kepler depth into the TESS band
    # The like-for-like comparison's POWER floor.  Two measured depths that
    # agree only say "unchanged" if the comparison could have detected the
    # change being claimed.  ``ln_ratio_under_test`` (the catalogue
    # TOI-vs-KOI ln ratio) is used when it is available; this is the fallback
    # when it is not.  0.10 in ln depth is a 10 % depth change.
    min_detectable_ln_ratio: float = 0.10
    # Our Kepler-era fit against ``koi_depth``: both are in the KEPLER band, so
    # NO band ratio is applied, and the same sigma_sys_ln floor is used.
    koi_check_n_agree: float = 3.0

    @classmethod
    def from_config(cls, conf: dict | None) -> CompareParams:
        c = ((conf or {}).get("stage2") or {}).get("compare") or {}
        d = cls()
        for k in d.__dataclass_fields__:
            if k in c and c[k] is not None:
                setattr(d, k, type(getattr(d, k))(c[k]))
        return d


@dataclass
class MastParams:
    """Wall-clock ceilings on every network stage, and which products to ask for."""

    authors: tuple[str, ...] = DEFAULT_AUTHORS
    target_timeout_s: float = 600.0     # one target's whole light-curve fetch
    per_target_budget_s: float = 900.0
    budget_s: float = 5400.0            # the WHOLE measure stage
    retries: int = 2
    retry_pause_s: float = 5.0          # backoff between attempts, bounded by the budget
    max_sectors: int = 60
    quality_bitmask: str = "default"
    download_dir: str | None = None
    archive_timeout_s: float = 300.0    # the Exoplanet Archive ephemeris pulls
    archive_retries: int = 3
    # --- the KEPLER era ----------------------------------------------------
    # Off by default ON PURPOSE.  A caller that has not thought about the
    # Kepler era must not silently open a socket to MAST; the repository
    # config turns it on and the workflow fails the run if it comes back off,
    # so a dropped key is loud rather than a quietly one-sided comparison.
    kepler_enabled: bool = False
    kepler_authors: tuple[str, ...] = KEPLER_AUTHORS
    #: Wall clock for the WHOLE Kepler side of the measure stage, counted
    #: separately from ``budget_s`` so that a slow Kepler fetch cannot eat the
    #: TESS budget (or the reverse).  Same discipline, same reason as
    #: ``gaia.cone_budget_s`` in ``config/growth.yaml``.
    kepler_budget_s: float = 5400.0
    kepler_per_target_budget_s: float = 1800.0
    kepler_target_timeout_s: float = 1200.0
    kepler_max_quarters: int = 60
    kepler_retries: int = 2

    @classmethod
    def from_config(cls, conf: dict | None) -> MastParams:
        m = ((conf or {}).get("stage2") or {}).get("mast") or {}
        d = cls()
        if m.get("authors"):
            d.authors = tuple(str(a) for a in m["authors"])
        if m.get("kepler_authors"):
            d.kepler_authors = tuple(str(a) for a in m["kepler_authors"])
        if m.get("kepler_enabled") is not None:
            d.kepler_enabled = bool(m["kepler_enabled"])
        for k in ("target_timeout_s", "per_target_budget_s", "budget_s", "archive_timeout_s",
                  "retry_pause_s", "kepler_budget_s", "kepler_per_target_budget_s",
                  "kepler_target_timeout_s"):
            if m.get(k) is not None:
                setattr(d, k, float(m[k]))
        for k in ("retries", "max_sectors", "archive_retries", "kepler_max_quarters",
                  "kepler_retries"):
            if m.get(k) is not None:
                setattr(d, k, int(m[k]))
        for k in ("quality_bitmask", "download_dir"):
            if m.get(k) is not None:
                setattr(d, k, str(m[k]))
        return d

    def kepler_view(self) -> MastParams:
        """The same ceilings, expressed for the Kepler fetch.

        :func:`fetch_kepler_lightcurves` runs through the same bounded loop as
        the TESS one, so it is handed a params object whose per-target budget,
        timeout, retry count and product cap are the Kepler ones.
        """
        import dataclasses  # noqa: PLC0415

        return dataclasses.replace(
            self, authors=tuple(self.kepler_authors),
            per_target_budget_s=float(self.kepler_per_target_budget_s),
            target_timeout_s=float(self.kepler_target_timeout_s),
            max_sectors=int(self.kepler_max_quarters), retries=int(self.kepler_retries))


@dataclass
class EnsembleParams:
    """The reduction ensemble: which reductions to enumerate, and how they bind.

    **Enumerated, not chosen.**  For each era the members are the full cross
    product of pipeline author and flux column --- ``SPOC`` / ``TESS-SPOC`` /
    ``QLP`` against ``PDCSAP_FLUX`` / ``KSPSAP_FLUX`` / ``DET_FLUX`` /
    ``SAP_FLUX`` on the TESS side, ``Kepler`` against ``PDCSAP_FLUX`` /
    ``SAP_FLUX`` on the Kepler side.  Most combinations do not exist (QLP has
    never served ``PDCSAP_FLUX``); those come back
    ``FLUX_COLUMN_NOT_PRESENT``, which is a recorded absence and not a failure.
    ``SAP_FLUX`` is in the grid deliberately and is never skipped: it is the one
    member that tests the background/crowding correction directly.

    ``enabled`` is **False in the code's own defaults, on purpose**, exactly as
    :attr:`MastParams.kepler_enabled` is: the ensemble multiplies the number of
    archive requests by the size of the grid, and a caller that has not thought
    about it must not silently open a socket (``tests/conftest.py`` raises on
    any socket use).  ``config/growth.yaml`` turns it on and
    ``growth_stage2.yml`` fails the run if the summary comes back with it off,
    so a dropped key is loud rather than a quietly over-precise error bar.

    ``spread_percentile`` sets the robust spread: the half range between the
    ``100 - p`` and ``p`` percentiles of the member depths, 16-84 by default.
    """

    enabled: bool = False
    tess_authors: tuple[str, ...] = DEFAULT_AUTHORS
    tess_flux_columns: tuple[str, ...] = FLUX_COLUMNS
    kepler_authors: tuple[str, ...] = KEPLER_AUTHORS
    kepler_flux_columns: tuple[str, ...] = KEPLER_FLUX_COLUMNS
    #: Wall clock for the WHOLE TESS ensemble, counted separately from
    #: ``MastParams.budget_s`` so the ensemble cannot eat the primary
    #: measurement's budget (or the reverse).  Same discipline, same reason as
    #: ``gaia.cone_budget_s`` in ``config/growth.yaml``.
    budget_s: float = 5400.0
    #: The same, for the whole Kepler ensemble.
    kepler_budget_s: float = 5400.0
    per_member_budget_s: float = 600.0
    member_timeout_s: float = 600.0
    retries: int = 2
    max_sectors: int = 60
    spread_percentile: float = 84.0
    min_members_for_spread: int = 2
    #: |z| at or above this on ``SAP - PDCSAP`` is ``BACKGROUND_TEST_DISAGREES``.
    background_n_agree: float = 3.0

    @classmethod
    def from_config(cls, conf: dict | None) -> EnsembleParams:
        e = ((conf or {}).get("stage2") or {}).get("ensemble") or {}
        d = cls()
        for k in ("tess_authors", "tess_flux_columns", "kepler_authors",
                  "kepler_flux_columns"):
            if e.get(k):
                setattr(d, k, tuple(str(v) for v in e[k]))
        if e.get("enabled") is not None:
            d.enabled = bool(e["enabled"])
        for k in ("budget_s", "kepler_budget_s", "per_member_budget_s", "member_timeout_s",
                  "spread_percentile", "background_n_agree"):
            if e.get(k) is not None:
                setattr(d, k, float(e[k]))
        for k in ("retries", "max_sectors", "min_members_for_spread"):
            if e.get(k) is not None:
                setattr(d, k, int(e[k]))
        return d

    def authors_for(self, era: str) -> tuple[str, ...]:
        return tuple(self.kepler_authors if str(era).lower() == ERA_KEPLER else self.tess_authors)

    def flux_columns_for(self, era: str) -> tuple[str, ...]:
        return tuple(self.kepler_flux_columns if str(era).lower() == ERA_KEPLER
                     else self.tess_flux_columns)

    def budget_for(self, era: str) -> float:
        return float(self.kepler_budget_s if str(era).lower() == ERA_KEPLER else self.budget_s)

    def member_view(self, mast: MastParams, *, author: str) -> MastParams:
        """``mast``'s ceilings, narrowed to ONE ensemble member's fetch.

        One author, the ensemble's own per-member budget and timeout, and the
        ensemble's own retry count --- so a member that hangs cannot consume the
        whole grid's wall clock, and the primary measurement's budget is
        untouched either way.
        """
        import dataclasses  # noqa: PLC0415

        return dataclasses.replace(
            mast, authors=(str(author),), kepler_authors=(str(author),),
            per_target_budget_s=float(self.per_member_budget_s),
            target_timeout_s=float(self.member_timeout_s),
            max_sectors=int(self.max_sectors), retries=int(self.retries),
            kepler_per_target_budget_s=float(self.per_member_budget_s),
            kepler_target_timeout_s=float(self.member_timeout_s),
            kepler_max_quarters=int(self.max_sectors), kepler_retries=int(self.retries))


@dataclass
class Deadline:
    """A monotonic wall-clock ceiling.  ``None`` budget means no ceiling."""

    budget_s: float | None = None
    started: float = field(default_factory=_time.monotonic)

    def expired(self) -> bool:
        return self.budget_s is not None and (_time.monotonic() - self.started) > self.budget_s

    def remaining(self) -> float:
        if self.budget_s is None:
            return float("inf")
        return max(0.0, self.budget_s - (_time.monotonic() - self.started))

    def elapsed(self) -> float:
        return _time.monotonic() - self.started


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mad = float(np.median(np.abs(x - np.median(x))))
    s = 1.4826 * mad
    return s if s > 0 else float(np.std(x, ddof=1))


def _wls_line(dt: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least squares ``y = a + b dt``.  Falls back to the weighted mean."""
    sw = float(np.sum(w))
    if sw <= 0 or dt.size < 2:
        return (float(np.mean(y)) if y.size else float("nan")), 0.0
    sx = float(np.sum(w * dt))
    sy = float(np.sum(w * y))
    sxx = float(np.sum(w * dt * dt))
    sxy = float(np.sum(w * dt * y))
    den = sw * sxx - sx * sx
    if not np.isfinite(den) or abs(den) < 1e-30:
        return sy / sw, 0.0
    b = (sw * sxy - sx * sy) / den
    a = (sy - b * sx) / sw
    return float(a), float(b)


def core_half_width(duration_days: float, exptime_days: float, params: FitParams
                    ) -> tuple[float, bool]:
    """The half-width of the flat-bottom window, shrunk by half the exposure.

    A point integrated over ``exptime`` reports the mean flux across that
    interval, so a point whose interval straddles ingress is not flat-bottom
    flux.  Returns ``(half_width_days, no_flat_core)``; ``no_flat_core`` is
    True when the integration is long enough that nothing survives the shrink,
    in which case the unshrunk window is used and the depth is a **lower
    bound** on the true one.
    """
    half = 0.5 * float(params.core_fraction) * float(duration_days)
    e = float(exptime_days) if np.isfinite(exptime_days) else 0.0
    shrunk = half - 0.5 * e
    if shrunk <= 0:
        return half, True
    return shrunk, False


def fit_transits(time_btjd, flux, flux_err, *, period_days: float, t0_btjd: float,
                 duration_days: float, exptime_days: float = float("nan"),
                 sector: int | None = None, params: FitParams | None = None) -> list[dict]:
    """Per-transit depths with a locally fitted, transit-masked baseline.

    The ephemeris is FIXED.  For each predicted transit inside the data: fit
    ``a + b dt`` to the out-of-transit part of the local window (the transit is
    masked with a guard band), divide the core points by it, and take
    ``1 - <ratio>`` as that transit's depth.  A transit without enough baseline
    or core points is skipped and says why.
    """
    params = params or FitParams()
    t = np.asarray(time_btjd, dtype=float)
    f = np.asarray(flux, dtype=float)
    fe = (np.asarray(flux_err, dtype=float) if flux_err is not None
          else np.full(t.shape, np.nan, dtype=float))
    ok = np.isfinite(t) & np.isfinite(f)
    t, f, fe = t[ok], f[ok], fe[ok]
    if t.size == 0 or not (np.isfinite(period_days) and period_days > 0
                           and np.isfinite(duration_days) and duration_days > 0):
        return []
    epoch, dt = fold(t, period_days, t0_btjd)
    half_core, _no_flat = core_half_width(duration_days, exptime_days, params)
    w_guard = float(params.baseline_guard_durations) * float(duration_days)
    w_out = float(params.baseline_window_durations) * float(duration_days)
    exp_d = float(exptime_days) if np.isfinite(exptime_days) and exptime_days > 0 else 0.0
    # THE CADENCE SETS THE WINDOW, not the other way round.  A window fixed in
    # units of T14 holds plenty of baseline points at 2-minute cadence and far
    # too few at 30-minute FFI cadence, where the transit would then be thrown
    # away for want of a baseline rather than measured and flagged.  So the
    # outer window is widened, when it has to be, until it can hold the
    # required number of baseline samples at the ACTUAL cadence.
    if exp_d > 0:
        need_per_side = 0.5 * int(params.min_baseline_points) + 1.0
        w_out = max(w_out, w_guard + need_per_side * exp_d)
    # Likewise the core requirement: a 30-minute integration cannot put two
    # samples inside a 40-minute flat bottom however much one would like it to.
    # The requirement drops to one sample when the cadence cannot supply more
    # --- such a sector is already carrying the `smeared` flag.
    min_core = int(params.min_core_points)
    if exp_d > 0:
        min_core = max(1, min(min_core, int(math.floor(2.0 * half_core / exp_d))))
    rows: list[dict] = []
    for ep in np.unique(epoch):
        m = (epoch == ep) & (np.abs(dt) <= w_out)
        if not np.any(m):
            continue
        d_i, f_i, e_i = dt[m], f[m], fe[m]
        core = np.abs(d_i) <= half_core
        base = np.abs(d_i) >= w_guard
        rec: dict = {"epoch": int(ep), "sector": sector,
                     "t_mid_btjd": float(t0_btjd + ep * period_days),
                     "n_core": int(core.sum()), "n_baseline": int(base.sum()),
                     "n_baseline_before": int((base & (d_i < 0)).sum()),
                     "n_baseline_after": int((base & (d_i > 0)).sum()),
                     "depth": float("nan"), "depth_err": float("nan"),
                     "oot_scatter": float("nan"), "used": False, "reject_reason": "",
                     "half_core_days": float(half_core), "window_days": float(w_out),
                     "min_core_points_required": int(min_core)}
        if rec["n_core"] < min_core:
            rec["reject_reason"] = "too_few_core_points"
            rows.append(rec)
            continue
        if rec["n_baseline"] < int(params.min_baseline_points):
            rec["reject_reason"] = "too_few_baseline_points"
            rows.append(rec)
            continue
        if params.require_baseline_both_sides and (rec["n_baseline_before"] == 0
                                                   or rec["n_baseline_after"] == 0):
            rec["reject_reason"] = "baseline_on_one_side_only"
            rows.append(rec)
            continue
        wb = np.where(np.isfinite(e_i[base]) & (e_i[base] > 0), 1.0 / np.maximum(
            e_i[base], 1e-30) ** 2, 1.0)
        a, b = _wls_line(d_i[base], f_i[base], wb)
        pred_base = a + b * d_i[base]
        pred_core = a + b * d_i[core]
        if not np.all(np.isfinite(pred_core)) or np.any(pred_core == 0):
            rec["reject_reason"] = "baseline_fit_failed"
            rows.append(rec)
            continue
        resid = f_i[base] / pred_base - 1.0
        sig = _robust_sigma(resid)
        ratio = f_i[core] / pred_core
        depth = 1.0 - float(np.mean(ratio))
        if np.isfinite(sig):
            err = sig * math.sqrt(1.0 / rec["n_core"] + 1.0 / rec["n_baseline"])
        else:
            err = float("nan")
        rec.update({"depth": depth, "depth_err": err, "oot_scatter": sig, "used": True})
        rows.append(rec)
    return rows


def combine_transit_depths(rows, *, params: FitParams | None = None,
                           rng: np.random.Generator | None = None) -> dict:
    """Inverse-variance combination of per-transit depths, with a stated error.

    Reports the analytic propagation, the chi2-scaled version and a bootstrap
    over transits; ``depth_err_method`` names the one promoted to
    ``depth_err_ppm``.  ``auto`` bootstraps once there are enough transits for
    the resampling to mean anything and falls back to the analytic error below
    that.
    """
    params = params or FitParams()
    used = [r for r in rows if r.get("used") and np.isfinite(r.get("depth", np.nan))
            and np.isfinite(r.get("depth_err", np.nan)) and r["depth_err"] > 0]
    out = {"n_transits": int(len(used)), "n_transits_seen": int(len(rows)),
           "depth_ppm": float("nan"), "depth_err_ppm": float("nan"),
           "depth_err_analytic_ppm": float("nan"), "depth_err_chi2_scaled_ppm": float("nan"),
           "depth_err_bootstrap_ppm": float("nan"), "depth_err_method": "none",
           "chi2": float("nan"), "chi2_per_dof": float("nan"),
           "oot_scatter_ppm": float("nan")}
    if not used:
        return out
    d = np.array([r["depth"] for r in used], dtype=float)
    e = np.array([r["depth_err"] for r in used], dtype=float)
    w = 1.0 / e**2
    mean = float(np.sum(w * d) / np.sum(w))
    err = float(1.0 / math.sqrt(np.sum(w)))
    chi2 = float(np.sum(((d - mean) / e) ** 2))
    dof = max(len(d) - 1, 1)
    out.update({"depth_ppm": mean * 1e6, "depth_err_analytic_ppm": err * 1e6,
                "chi2": chi2, "chi2_per_dof": chi2 / dof,
                "depth_err_chi2_scaled_ppm": err * math.sqrt(max(chi2 / dof, 1.0)) * 1e6,
                "oot_scatter_ppm": float(np.nanmedian(
                    [r.get("oot_scatter", np.nan) for r in used])) * 1e6})
    if len(d) >= 2:
        g = rng or np.random.default_rng(int(params.seed))
        draws = int(params.bootstrap_draws)
        idx = g.integers(0, len(d), size=(draws, len(d)))
        dd, ww = d[idx], w[idx]
        means = np.sum(ww * dd, axis=1) / np.sum(ww, axis=1)
        out["depth_err_bootstrap_ppm"] = float(np.std(means, ddof=1)) * 1e6
    method = str(params.err_method).lower()
    if method == "auto":
        method = ("bootstrap" if len(d) >= int(params.min_transits_for_bootstrap)
                  and np.isfinite(out["depth_err_bootstrap_ppm"]) else "analytic")
    chosen = {"bootstrap": out["depth_err_bootstrap_ppm"],
              "analytic": out["depth_err_analytic_ppm"],
              "chi2_scaled": out["depth_err_chi2_scaled_ppm"]}.get(
                  method, out["depth_err_analytic_ppm"])
    if not np.isfinite(chosen) or chosen <= 0:
        chosen, method = out["depth_err_analytic_ppm"], "analytic"
    out["depth_err_ppm"] = float(chosen)
    out["depth_err_method"] = method
    return out


def binned_fold(rows_time, rows_flux, *, period_days: float, t0_btjd: float,
                duration_days: float, params: FitParams | None = None) -> pd.DataFrame:
    """A phase-binned fold over ``|dt| <= w T14``, for a human to look at."""
    params = params or FitParams()
    t = np.asarray(rows_time, dtype=float)
    f = np.asarray(rows_flux, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if not t.size:
        return pd.DataFrame(columns=["dt_hours", "flux", "flux_err", "n"])
    _, dt = fold(t, period_days, t0_btjd)
    w = float(params.baseline_window_durations) * float(duration_days)
    m = np.abs(dt) <= w
    if not np.any(m):
        return pd.DataFrame(columns=["dt_hours", "flux", "flux_err", "n"])
    nb = max(int(params.fold_bins), 5)
    edges = np.linspace(-w, w, nb + 1)
    idx = np.clip(np.digitize(dt[m], edges) - 1, 0, nb - 1)
    recs = []
    for i in range(nb):
        s = idx == i
        n = int(s.sum())
        if not n:
            continue
        v = f[m][s]
        recs.append({"dt_hours": float(0.5 * (edges[i] + edges[i + 1]) * 24.0),
                     "flux": float(np.mean(v)),
                     "flux_err": float(np.std(v, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                     "n": n})
    return pd.DataFrame(recs, columns=["dt_hours", "flux", "flux_err", "n"])


def dedupe_sectors(sectors, *, authors=DEFAULT_AUTHORS) -> tuple[list[dict], list[dict]]:
    """One light curve per sector: the shortest exposure wins, ties by author order.

    MAST serves a sector under several pipelines and they are reductions of the
    SAME PIXELS. Stacking two of them counts every transit twice, which does
    not improve the depth but does shrink its quoted error by sqrt(2) --- a
    measurement that looks more precise than the photons allow. Returns
    ``(kept, dropped)``; ``dropped`` records each discarded light curve with the
    one that displaced it, so the choice is auditable rather than silent.
    """
    order = {a: i for i, a in enumerate(authors)}

    def rank(s: dict) -> tuple:
        e = float(s.get("exptime_s") or np.inf)
        return (e if np.isfinite(e) else np.inf,
                order.get(str(s.get("author")), len(order)))

    best: dict = {}
    for s in sectors or []:
        key = s.get("sector")
        if key is None:                                   # no sector id: never merged away
            best[f"__unkeyed_{len(best)}"] = s
            continue
        cur = best.get(key)
        if cur is None or rank(s) < rank(cur):
            best[key] = s
    kept_ids = {id(v) for v in best.values()}
    dropped = [{"sector": s.get("sector"), "author": s.get("author"),
                "exptime_s": s.get("exptime_s"),
                "reason": "same sector already covered by a shorter-cadence pipeline",
                "kept_author": str((best.get(s.get("sector")) or {}).get("author")),
                "kept_exptime_s": (best.get(s.get("sector")) or {}).get("exptime_s")}
               for s in (sectors or []) if id(s) not in kept_ids]
    return list(best.values()), dropped


def measure_target(sectors, *, period_days: float, t0_btjd: float, duration_days: float,
                   params: FitParams | None = None) -> dict:
    """Fit every sector of one target and combine: depth, per-sector, odd-even.

    ``sectors`` is a list of dicts with ``time`` (BTJD), ``flux``, ``flux_err``
    and metadata (``sector``, ``author``, ``exptime_s``, ``flux_column``).
    Each sector is normalised by its own robust median FIRST, so a sector-level
    flux scale cannot leak into the depth.

    **One sector is counted once.** MAST serves the same sector under more than
    one pipeline --- SPOC at 120 s and TESS-SPOC at 200 or 600 s are different
    REDUCTIONS OF THE SAME PIXELS, not independent observations. Run 35041932130
    measured K00897.01 over "14 sectors" that were seven sectors twice, so every
    transit entered the stack twice and the quoted error was too small by a
    factor sqrt(2). :func:`dedupe_sectors` keeps the best-cadence pipeline per
    sector and records what it dropped.
    """
    params = params or FitParams()
    sectors, dropped = dedupe_sectors(sectors)
    all_rows: list[dict] = []
    sec_recs: list[dict] = []
    norm_time: list[np.ndarray] = []
    norm_flux: list[np.ndarray] = []
    smeared_any = False
    lower_bound_any = False
    for s in sectors or []:
        t = np.asarray(s.get("time"), dtype=float)
        f = np.asarray(s.get("flux"), dtype=float)
        fe = (np.asarray(s.get("flux_err"), dtype=float) if s.get("flux_err") is not None
              else np.full(t.shape, np.nan))
        ok = np.isfinite(t) & np.isfinite(f)
        t, f, fe = t[ok], f[ok], fe[ok]
        med = float(np.median(f)) if t.size else float("nan")
        if not np.isfinite(med) or med == 0:
            sec_recs.append({"sector": s.get("sector"), "author": s.get("author"),
                             "exptime_s": s.get("exptime_s"), "n_points": int(t.size),
                             "status": "NOT_NORMALISABLE", "n_transits": 0,
                             "depth_ppm": float("nan"), "depth_err_ppm": float("nan")})
            continue
        f, fe = f / med, fe / med
        exptime_s = float(s.get("exptime_s") or np.nan)
        exptime_d = exptime_s / 86400.0 if np.isfinite(exptime_s) else float("nan")
        _, no_flat = core_half_width(duration_days, exptime_d, params)
        smeared = bool(np.isfinite(exptime_d)
                       and exptime_d > float(params.smear_fraction) * float(duration_days))
        smeared_any = smeared_any or smeared
        lower_bound_any = lower_bound_any or no_flat
        rows = fit_transits(t, f, fe, period_days=period_days, t0_btjd=t0_btjd,
                            duration_days=duration_days, exptime_days=exptime_d,
                            sector=s.get("sector"), params=params)
        all_rows.extend(rows)
        norm_time.append(t)
        norm_flux.append(f)
        c = combine_transit_depths(rows, params=params)
        sec_recs.append({"sector": s.get("sector"), "author": s.get("author"),
                         "exptime_s": exptime_s, "flux_column": s.get("flux_column"),
                         "n_points": int(t.size), "status": STATUS_OK,
                         "median_flux": med, "smeared": smeared,
                         "no_flat_core": bool(no_flat),
                         "exptime_over_duration": (float(exptime_d / duration_days)
                                                   if np.isfinite(exptime_d) else float("nan")),
                         "n_transits": c["n_transits"], "n_transits_seen": c["n_transits_seen"],
                         "depth_ppm": c["depth_ppm"], "depth_err_ppm": c["depth_err_ppm"],
                         "depth_err_method": c["depth_err_method"],
                         "oot_scatter_ppm": c["oot_scatter_ppm"]})
    comb = combine_transit_depths(all_rows, params=params)
    odd = combine_transit_depths([r for r in all_rows if int(r.get("epoch", 0)) % 2 != 0],
                                 params=params)
    even = combine_transit_depths([r for r in all_rows if int(r.get("epoch", 0)) % 2 == 0],
                                  params=params)
    oe_diff = odd["depth_ppm"] - even["depth_ppm"]
    oe_sig = math.sqrt(odd["depth_err_ppm"] ** 2 + even["depth_err_ppm"] ** 2) if (
        np.isfinite(odd["depth_err_ppm"]) and np.isfinite(even["depth_err_ppm"])) else float("nan")
    oe_z = abs(oe_diff) / oe_sig if np.isfinite(oe_sig) and oe_sig > 0 else float("nan")
    sec_df = pd.DataFrame(sec_recs)
    sec_meas = sec_df[np.isfinite(pd.to_numeric(sec_df.get("depth_ppm", pd.Series(dtype=float)),
                                                errors="coerce"))] if len(sec_df) else sec_df
    sec_chi2, sec_dof = float("nan"), 0
    if len(sec_meas) >= 2 and np.isfinite(comb["depth_ppm"]):
        dd = pd.to_numeric(sec_meas["depth_ppm"], errors="coerce").to_numpy(float)
        ee = pd.to_numeric(sec_meas["depth_err_ppm"], errors="coerce").to_numpy(float)
        g = np.isfinite(dd) & np.isfinite(ee) & (ee > 0)
        if g.sum() >= 2:
            sec_chi2 = float(np.sum(((dd[g] - comb["depth_ppm"]) / ee[g]) ** 2))
            sec_dof = int(g.sum() - 1)
    flags: list[str] = []
    if smeared_any:
        flags.append("smeared")
    if lower_bound_any:
        flags.append("depth_is_lower_bound")
    if np.isfinite(oe_z) and oe_z >= float(params.odd_even_sigma):
        flags.append("odd_even_significant")
    if sec_dof and np.isfinite(sec_chi2) and (sec_chi2 / sec_dof) >= float(
            params.sector_scatter_chi2_per_dof):
        flags.append("sector_scatter")
    fold_df = (binned_fold(np.concatenate(norm_time), np.concatenate(norm_flux),
                           period_days=period_days, t0_btjd=t0_btjd,
                           duration_days=duration_days, params=params)
               if norm_time else pd.DataFrame(columns=["dt_hours", "flux", "flux_err", "n"]))
    return {
        **{k: comb[k] for k in comb},
        "n_sectors": int(len(sec_recs)),
        "n_sectors_measured": int(len(sec_meas)),
        "sectors": sec_df,
        "sector_list": ",".join(str(r.get("sector")) for r in sec_recs if r.get("sector")
                                is not None),
        "authors": ",".join(sorted({str(r.get("author")) for r in sec_recs
                                    if r.get("author")})),
        "exptimes_s": ",".join(sorted({f"{float(r['exptime_s']):.0f}" for r in sec_recs
                                       if r.get("exptime_s") and np.isfinite(
                                           float(r["exptime_s"]))})),
        # Same sector, another pipeline: dropped so no transit is counted twice.
        "n_sectors_dropped_duplicate": int(len(dropped)),
        "duplicate_sectors_dropped": ";".join(
            f"{d['sector']}:{d['author']}@{d['exptime_s']}s->kept "
            f"{d['kept_author']}@{d['kept_exptime_s']}s" for d in dropped),
        "smeared": bool(smeared_any), "depth_is_lower_bound": bool(lower_bound_any),
        "depth_odd_ppm": odd["depth_ppm"], "depth_odd_err_ppm": odd["depth_err_ppm"],
        "n_transits_odd": odd["n_transits"],
        "depth_even_ppm": even["depth_ppm"], "depth_even_err_ppm": even["depth_err_ppm"],
        "n_transits_even": even["n_transits"],
        "odd_even_diff_ppm": oe_diff, "odd_even_sigma": oe_z,
        "sector_scatter_chi2": sec_chi2, "sector_scatter_dof": sec_dof,
        "sector_scatter_chi2_per_dof": (sec_chi2 / sec_dof) if sec_dof else float("nan"),
        "flags": ";".join(flags),
        "transits": pd.DataFrame(all_rows),
        "fold": fold_df,
    }


# ---------------------------------------------------------------------------
# The reduction ensemble: the spread over reductions, and what it does to the
# error.  These four functions are PURE --- they take member records and return
# numbers, so the load-bearing behaviour (a disagreeing ensemble inflating the
# error until a CHANGED verdict becomes UNRESOLVED) is testable without any
# fetch at all.
# ---------------------------------------------------------------------------
def ensemble_member_grid(era: str, params: EnsembleParams | None = None) -> list[dict]:
    """Every (author, flux column) combination to be measured for ``era``.

    The full cross product, **enumerated rather than chosen**.  Most entries do
    not exist at the archive --- QLP has never served ``PDCSAP_FLUX``, SPOC has
    never served ``KSPSAP_FLUX`` --- and those come back
    ``FLUX_COLUMN_NOT_PRESENT``, which is recorded.  The point is that the grid
    is written down in one place and the *archive* decides what is in it, not
    the code's opinion about which reduction is best.
    """
    params = params or EnsembleParams()
    e = str(era).lower()
    if e not in ERAS:
        raise ValueError(f"unknown era {era!r}; choose from {ERAS}")
    return [{"era": e, "author": str(a), "flux_column": str(c)}
            for a in params.authors_for(e) for c in params.flux_columns_for(e)]


def total_depth_error(stat_err_ppm: float, reduction_spread_ppm: float) -> float:
    """``sqrt(statistical^2 + reduction_spread^2)`` --- the error a verdict may use.

    The bootstrap over transits describes the photon noise and the
    transit-to-transit scatter.  It says **nothing** about two reductions of the
    same pixels disagreeing by 20 %, which is what run 35041932130's duplicated
    sectors actually showed.  A spread that could not be measured contributes
    nothing (it is not silently treated as zero *confidence*; the era carries
    ``depth_reduction_spread_status`` saying so).
    """
    s = float(stat_err_ppm) if np.isfinite(stat_err_ppm) else float("nan")
    if not np.isfinite(s):
        return float("nan")
    r = float(reduction_spread_ppm) if np.isfinite(reduction_spread_ppm) else 0.0
    return float(math.sqrt(s * s + r * r))


def reduction_spread(members, *, primary_depth_ppm: float = float("nan"),
                     params: EnsembleParams | None = None) -> dict:
    """The spread over the reduction ensemble, and what is missing from it.

    The spread is **half the range between the 16th and 84th percentiles** of
    the measured members' depths (``spread_percentile``), which is the robust
    analogue of one standard deviation and is not moved by one pathological
    reduction the way ``max - min`` is.  ``min``, ``max``, the percentiles and
    the full per-member table are reported beside it, so a reader can see
    whether the spread is a broad scatter or two clusters.

    **A member that did not come back is counted, not dropped.**  An ensemble of
    two that was meant to be six has a smaller spread than the truth, so
    ``n_members_unavailable`` and ``ensemble_unavailable_reasons`` are part of
    the record and ``ensemble_incomplete`` is True.  Below
    ``min_members_for_spread`` measured members there is no spread at all: the
    value is NaN with ``depth_reduction_spread_status = TOO_FEW_MEMBERS``,
    never 0, because a zero spread would assert that every reduction agreed.
    """
    params = params or EnsembleParams()
    rows = [dict(r) for r in (members or [])]
    usable = [bool(str(r.get("status")) == STATUS_OK and np.isfinite(_f(r.get("depth_ppm")))
                   and _f(r.get("depth_ppm")) > 0) for r in rows]
    meas = [r for r, ok in zip(rows, usable, strict=True) if ok]
    bad = [r for r, ok in zip(rows, usable, strict=True) if not ok]
    reasons: dict = {}
    for r in bad:
        k = str(r.get("status") or "unknown")
        reasons[k] = reasons.get(k, 0) + 1
    out = {
        "n_members": int(len(rows)),
        "n_members_measured": int(len(meas)),
        "n_members_unavailable": int(len(bad)),
        "ensemble_incomplete": bool(bad),
        "ensemble_unavailable_reasons": ";".join(f"{k}:{v}" for k, v in sorted(reasons.items())),
        "ensemble_members": ";".join(
            f"{r.get('author')}/{r.get('flux_column')}="
            + (f"{_f(r.get('depth_ppm')):.0f}ppm" if ok else str(r.get("status")))
            for r, ok in zip(rows, usable, strict=True)),
        "depth_reduction_spread_ppm": float("nan"),
        "depth_reduction_spread_fraction": float("nan"),
        "depth_reduction_spread_status": (SPREAD_NOT_ATTEMPTED if not rows
                                          else SPREAD_TOO_FEW_MEMBERS),
        "depth_reduction_p16_ppm": float("nan"), "depth_reduction_p84_ppm": float("nan"),
        "depth_reduction_min_ppm": float("nan"), "depth_reduction_max_ppm": float("nan"),
        "depth_reduction_median_ppm": float("nan"),
        "depth_reduction_max_offset_from_primary_ppm": float("nan"),
        "spread_percentile": float(params.spread_percentile),
    }
    if len(meas) < max(int(params.min_members_for_spread), 2):
        return out
    d = np.array([_f(r.get("depth_ppm")) for r in meas], dtype=float)
    hi = float(params.spread_percentile)
    lo = 100.0 - hi
    p_lo, p_hi = (float(v) for v in np.percentile(d, [lo, hi]))
    spread = 0.5 * (p_hi - p_lo)
    med = float(np.median(d))
    out.update({
        "depth_reduction_spread_ppm": float(spread),
        "depth_reduction_spread_status": SPREAD_OK,
        "depth_reduction_p16_ppm": p_lo, "depth_reduction_p84_ppm": p_hi,
        "depth_reduction_min_ppm": float(np.min(d)), "depth_reduction_max_ppm": float(np.max(d)),
        "depth_reduction_median_ppm": med,
    })
    ref = (float(primary_depth_ppm) if np.isfinite(primary_depth_ppm)
           and primary_depth_ppm > 0 else med)
    if ref > 0:
        out["depth_reduction_spread_fraction"] = float(spread / ref)
        out["depth_reduction_max_offset_from_primary_ppm"] = float(np.max(np.abs(d - ref)))
    return out


def sap_vs_pdcsap(members, *, params: EnsembleParams | None = None) -> dict:
    """The background test, answered with a number: ``SAP - PDCSAP``.

    PDC is where the crowding and background corrections are applied, so "the
    PDC background subtraction inflates the depth" is a specific mundane claim
    about this object --- a faint star in a crowded aperture with an
    over-subtracted background gives an INFLATED transit depth.  It deserves a
    measurement rather than a worry.

    The comparison is made **within a pipeline author**, because SPOC's SAP and
    TESS-SPOC's PDCSAP differ in the aperture as well as in the correction; the
    per-author differences are then combined by inverse variance.  A negative
    ``sap_minus_pdcsap_ppm`` means the corrected photometry is DEEPER than the
    raw aperture photometry --- ``PDC_DEEPER_THAN_SAP``, the direction the
    mundane explanation predicts.
    """
    params = params or EnsembleParams()
    rows = [r for r in (members or []) if str(r.get("status")) == STATUS_OK
            and np.isfinite(_f(r.get("depth_ppm"))) and _f(r.get("depth_ppm")) > 0]
    by_author: dict = {}
    for r in rows:
        by_author.setdefault(str(r.get("author")), {})[str(r.get("flux_column")).upper()] = r
    pairs = []
    for author in sorted(by_author):
        cols = by_author[author]
        sap, pdc = cols.get("SAP_FLUX"), cols.get("PDCSAP_FLUX")
        if sap is None or pdc is None:
            continue
        ds, dp = _f(sap.get("depth_ppm")), _f(pdc.get("depth_ppm"))
        es, ep = _f(sap.get("depth_err_ppm")), _f(pdc.get("depth_err_ppm"))
        err = math.sqrt((es if np.isfinite(es) and es > 0 else 0.0) ** 2
                        + (ep if np.isfinite(ep) and ep > 0 else 0.0) ** 2)
        pairs.append({"author": author, "sap_ppm": ds, "pdcsap_ppm": dp,
                      "diff_ppm": ds - dp, "diff_err_ppm": err if err > 0 else float("nan")})
    out = {
        "sap_vs_pdcsap_verdict": BG_UNAVAILABLE,
        "sap_vs_pdcsap_n_pairs": int(len(pairs)),
        "sap_vs_pdcsap_authors": ",".join(p["author"] for p in pairs),
        "sap_minus_pdcsap_ppm": float("nan"), "sap_minus_pdcsap_err_ppm": float("nan"),
        "sap_minus_pdcsap_z": float("nan"),
        "sap_minus_pdcsap_fraction": float("nan"),
        "background_direction": "",
        "background_n_agree": float(params.background_n_agree),
        "sap_vs_pdcsap_pairs": pairs,
    }
    if not pairs:
        return out
    diffs = np.array([p["diff_ppm"] for p in pairs], dtype=float)
    errs = np.array([p["diff_err_ppm"] for p in pairs], dtype=float)
    good = np.isfinite(errs) & (errs > 0)
    if good.all():
        w = 1.0 / errs**2
        diff = float(np.sum(w * diffs) / np.sum(w))
        err = float(1.0 / math.sqrt(np.sum(w)))
    else:
        # No usable per-member errors: the scatter over the pairs IS the error,
        # and a single pair without an error cannot be given a significance.
        diff = float(np.mean(diffs))
        err = (float(np.std(diffs, ddof=1) / math.sqrt(diffs.size)) if diffs.size > 1
               else float("nan"))
    out["sap_minus_pdcsap_ppm"] = diff
    out["sap_minus_pdcsap_err_ppm"] = err
    pdc_mean = float(np.mean([p["pdcsap_ppm"] for p in pairs]))
    if pdc_mean > 0:
        out["sap_minus_pdcsap_fraction"] = diff / pdc_mean
    out["background_direction"] = BG_PDC_DEEPER if diff < 0 else BG_SAP_DEEPER
    if not (np.isfinite(err) and err > 0):
        return out                                        # a difference without a significance
    z = diff / err
    out["sap_minus_pdcsap_z"] = float(z)
    out["sap_vs_pdcsap_verdict"] = (BG_DISAGREE if abs(z) >= float(params.background_n_agree)
                                    else BG_AGREE)
    return out


# ---------------------------------------------------------------------------
# The three-way comparison
# ---------------------------------------------------------------------------
def _z_ln(d1: float, e1: float, d2: float, e2: float, sigma_sys: float) -> tuple[float, float]:
    """``(z, sigma)`` for ``ln(d1/d2)`` with fractional errors added in quadrature."""
    if not (np.isfinite(d1) and np.isfinite(d2) and d1 > 0 and d2 > 0):
        return float("nan"), float("nan")
    f1 = (float(e1) / float(d1)) if np.isfinite(e1) and e1 > 0 else 0.0
    f2 = (float(e2) / float(d2)) if np.isfinite(e2) and e2 > 0 else 0.0
    sig = math.sqrt(f1 * f1 + f2 * f2 + float(sigma_sys) ** 2)
    if sig <= 0:
        return float("nan"), float("nan")
    return float(math.log(d1 / d2) / sig), float(sig)


def compare_three_depths(depth_measured_ppm: float, err_measured_ppm: float,
                         depth_kepler_ppm: float, err_kepler_ppm: float,
                         depth_toi_ppm: float, err_toi_ppm: float, *,
                         ld_band_ratio: float = 1.0,
                         params: CompareParams | None = None,
                         unmeasured_reason: str | None = None,
                         measured_reduction_spread_ppm: float = float("nan")) -> dict:
    """The decisive comparison: measured TESS depth against BOTH catalogue depths.

    The Kepler depth is multiplied by ``ld_band_ratio`` (=
    :func:`seti.growth.drift.band_ratio`, ``F_TESS(b)/F_Kepler(b)``) so both
    references sit in the TESS band before anything is compared.  Returns the
    verdict, both ``z``s, and ``z_toi_vs_kepler`` --- the separation between the
    two references, without which "agrees with both" cannot be read.

    ``measured_reduction_spread_ppm`` is the reduction-ensemble systematic
    (:func:`reduction_spread`).  When it is available it is added in quadrature
    to the statistical error and **the comparison uses the total**, so this
    diagnostic cannot be more confident about the TESS era than the ensemble
    allows either.  Both errors stay on the record.
    """
    params = params or CompareParams()
    err_total = total_depth_error(err_measured_ppm, measured_reduction_spread_ppm)
    out: dict = {"verdict": MATCH_UNMEASURED, "unmeasured_reason": unmeasured_reason or "",
                 "depth_measured_ppm": float(depth_measured_ppm),
                 "depth_measured_err_ppm": float(err_measured_ppm),
                 "depth_measured_total_err_ppm": float(err_total),
                 "depth_measured_reduction_spread_ppm": float(measured_reduction_spread_ppm),
                 "depth_kepler_ppm": float(depth_kepler_ppm),
                 "depth_kepler_in_tess_band_ppm": float("nan"),
                 "depth_toi_ppm": float(depth_toi_ppm),
                 "ld_band_ratio": float(ld_band_ratio),
                 "z_vs_kepler": float("nan"), "sigma_vs_kepler": float("nan"),
                 "z_vs_toi": float("nan"), "sigma_vs_toi": float("nan"),
                 "z_toi_vs_kepler": float("nan"),
                 "agrees_with_kepler": False, "agrees_with_toi": False,
                 "references_separated": False, "n_references_usable": 0,
                 "n_agree": float(params.n_agree),
                 "sigma_sys_ln": float(params.sigma_sys_ln)}
    out["n_references_usable"] = int(sum(
        1 for d in (depth_kepler_ppm, depth_toi_ppm) if np.isfinite(d) and d > 0))
    fr = float(ld_band_ratio) if (params.apply_band_ratio and np.isfinite(ld_band_ratio)
                                  and ld_band_ratio > 0) else 1.0
    dk = float(depth_kepler_ppm) * fr
    ek = float(err_kepler_ppm) * fr if np.isfinite(err_kepler_ppm) else float("nan")
    out["depth_kepler_in_tess_band_ppm"] = dk
    zrk, _s = _z_ln(float(depth_toi_ppm), float(err_toi_ppm), dk, ek, params.sigma_sys_ln)
    out["z_toi_vs_kepler"] = zrk
    out["references_separated"] = bool(np.isfinite(zrk) and abs(zrk) >= params.n_agree)
    if unmeasured_reason or not (np.isfinite(depth_measured_ppm) and depth_measured_ppm > 0):
        out["verdict"] = MATCH_UNMEASURED
        out["unmeasured_reason"] = unmeasured_reason or UNMEASURED_NO_TRANSIT
        return out
    if out["n_references_usable"] == 0:
        # A depth WAS measured; there is simply nothing to compare it with.
        # "matches neither" would assert a disagreement with values that do not
        # exist, so the comparison is the thing reported as unmeasured.
        out["verdict"] = MATCH_UNMEASURED
        out["unmeasured_reason"] = UNMEASURED_NO_REFERENCE
        return out
    zk, sk = _z_ln(float(depth_measured_ppm), float(err_total), dk, ek,
                   params.sigma_sys_ln)
    zt, st = _z_ln(float(depth_measured_ppm), float(err_total), float(depth_toi_ppm),
                   float(err_toi_ppm), params.sigma_sys_ln)
    out.update({"z_vs_kepler": zk, "sigma_vs_kepler": sk, "z_vs_toi": zt, "sigma_vs_toi": st})
    ak = bool(np.isfinite(zk) and abs(zk) < params.n_agree)
    at = bool(np.isfinite(zt) and abs(zt) < params.n_agree)
    out["agrees_with_kepler"], out["agrees_with_toi"] = ak, at
    if ak and at:
        out["verdict"] = MATCH_BOTH
    elif ak:
        out["verdict"] = MATCH_KEPLER
    elif at:
        out["verdict"] = MATCH_TOI
    else:
        out["verdict"] = MATCH_NEITHER
    return out


def compare_measured_eras(depth_kepler_measured_ppm: float, err_kepler_measured_ppm: float,
                          depth_tess_measured_ppm: float, err_tess_measured_ppm: float, *,
                          ld_band_ratio: float = 1.0,
                          ln_ratio_under_test: float = float("nan"),
                          params: CompareParams | None = None,
                          kepler_unmeasured_reason: str | None = None,
                          tess_unmeasured_reason: str | None = None,
                          kepler_reduction_spread_ppm: float = float("nan"),
                          tess_reduction_spread_ppm: float = float("nan")) -> dict:
    """**The primary comparison**: OUR Kepler-era depth against OUR TESS-era depth.

    Both numbers come out of :func:`measure_target` --- the same fold, the same
    transit-masked local baseline, the same exposure-shrunk core window, the
    same inverse-variance combination and bootstrap over transits.  Nothing
    catalogue-derived enters except the fixed ephemeris and the limb-darkening
    band ratio, which is applied to the Kepler-era measurement exactly as
    :func:`compare_three_depths` applies it to the Kepler *catalogue* depth, so
    the bandpass difference is not charged to "growth".

    ``z = ln(D_TESS / D_Kepler,in TESS band) / sigma`` with the fractional
    errors and ``sigma_sys_ln`` added in quadrature; positive means deeper in
    the TESS era.

    The four branches, and the one that is easy to get wrong:

    * ``MEASURED_DEPTH_CHANGED`` --- ``|z| >= n_agree``.
    * ``MEASURED_DEPTH_UNCHANGED`` --- ``|z| < n_agree`` **and the change under
      test is excluded**: the observed log ratio sits at least ``n_agree``
      sigma BELOW ``ln_ratio_under_test`` (the catalogue TOI-vs-KOI separation,
      or ``params.min_detectable_ln_ratio`` when that is unavailable), i.e.
      ``claim_excluded_sigma = (target - |ln ratio observed|) / sigma >=
      n_agree``.  It is not enough for ``n_agree * sigma`` to be smaller than
      the claim: a comparison that measures a ratio of 2.16 and merely cannot
      call it significant has not refuted anything, and "UNCHANGED" says the
      candidate DIES.  The distinction is dormant while the errors are small
      (the observed ratio is then either significant or near zero) and becomes
      load-bearing the moment a reduction systematic is folded in --- which is
      exactly when an inflated error bar could otherwise buy a refutation.
    * ``MEASURED_DEPTH_UNRESOLVED`` --- ``|z| < n_agree`` but the comparison
      could not have seen the claimed change anyway.  **An agreement without
      power is not a refutation**, and reporting it as one would be the same
      error in the opposite direction from the one this stage exists to fix.
    * ``MEASURED_DEPTH_ERA_UNMEASURED`` --- one or both eras produced no depth.
      The reason names the era, and an unmeasured era never agrees with
      anything.

    **The verdict uses the TOTAL error.**  Each era's statistical error (the
    bootstrap over transits) is added in quadrature to that era's
    reduction-ensemble spread (:func:`reduction_spread`) before any ``z`` is
    formed, because two reductions of the same pixels disagreeing by 20 % is a
    real uncertainty on the depth and the bootstrap does not know about it.
    ``z_measured_eras_stat_only`` keeps the bootstrap-only number beside it, so
    the effect of the systematic is visible rather than merely applied.  Note
    the consequence, which is intended: a large ensemble spread not only pulls
    ``|z|`` below ``n_agree`` but also raises ``detectable_ln_ratio``, so a
    comparison drowned in reduction systematics returns
    ``MEASURED_DEPTH_UNRESOLVED`` --- an honest "this measurement cannot tell"
    --- and never ``MEASURED_DEPTH_UNCHANGED``.
    """
    params = params or CompareParams()
    fr = float(ld_band_ratio) if (params.apply_band_ratio and np.isfinite(ld_band_ratio)
                                  and ld_band_ratio > 0) else 1.0
    dk = float(depth_kepler_measured_ppm) * fr
    ek_stat = (float(err_kepler_measured_ppm) * fr if np.isfinite(err_kepler_measured_ppm)
               else float("nan"))
    ek_tot = total_depth_error(err_kepler_measured_ppm, kepler_reduction_spread_ppm)
    ek = ek_tot * fr if np.isfinite(ek_tot) else float("nan")
    et_tot = total_depth_error(err_tess_measured_ppm, tess_reduction_spread_ppm)
    applied = bool(np.isfinite(kepler_reduction_spread_ppm)
                   or np.isfinite(tess_reduction_spread_ppm))
    out: dict = {
        "like_for_like_verdict": LL_ERA_UNMEASURED,
        "like_for_like_unmeasured_reason": "",
        "depth_kepler_measured_ppm": float(depth_kepler_measured_ppm),
        "depth_kepler_measured_err_ppm": float(err_kepler_measured_ppm),
        "depth_kepler_measured_total_err_ppm": float(ek_tot),
        "depth_kepler_measured_in_tess_band_ppm": dk,
        "depth_tess_measured_ppm": float(depth_tess_measured_ppm),
        "depth_tess_measured_err_ppm": float(err_tess_measured_ppm),
        "depth_tess_measured_total_err_ppm": float(et_tot),
        "kepler_reduction_spread_ppm": float(kepler_reduction_spread_ppm),
        "tess_reduction_spread_ppm": float(tess_reduction_spread_ppm),
        "reduction_systematic_applied": applied,
        "like_for_like_ld_band_ratio": float(ld_band_ratio),
        "z_measured_eras": float("nan"), "sigma_measured_eras": float("nan"),
        "z_measured_eras_stat_only": float("nan"),
        "sigma_measured_eras_stat_only": float("nan"),
        "measured_depth_ratio": float("nan"),
        "measured_eras_agree": False,
        "ln_ratio_under_test": float(ln_ratio_under_test),
        "detectable_ln_ratio": float("nan"),
        "claim_excluded_sigma": float("nan"),
        "like_for_like_n_agree": float(params.n_agree),
    }
    missing = []
    if kepler_unmeasured_reason or not (np.isfinite(depth_kepler_measured_ppm)
                                        and depth_kepler_measured_ppm > 0):
        missing.append(f"KEPLER_ERA:{kepler_unmeasured_reason or UNMEASURED_NO_TRANSIT}")
    if tess_unmeasured_reason or not (np.isfinite(depth_tess_measured_ppm)
                                      and depth_tess_measured_ppm > 0):
        missing.append(f"TESS_ERA:{tess_unmeasured_reason or UNMEASURED_NO_TRANSIT}")
    if missing:
        out["like_for_like_unmeasured_reason"] = ";".join(missing)
        return out
    z, sig = _z_ln(float(depth_tess_measured_ppm), float(et_tot), dk, ek, params.sigma_sys_ln)
    z0, sig0 = _z_ln(float(depth_tess_measured_ppm), float(err_tess_measured_ppm), dk, ek_stat,
                     params.sigma_sys_ln)
    out["z_measured_eras"], out["sigma_measured_eras"] = z, sig
    out["z_measured_eras_stat_only"], out["sigma_measured_eras_stat_only"] = z0, sig0
    if dk > 0:
        out["measured_depth_ratio"] = float(depth_tess_measured_ppm) / dk
    if not np.isfinite(z):
        out["like_for_like_unmeasured_reason"] = "Z_NOT_COMPUTABLE"
        return out
    detectable = float(params.n_agree) * float(sig)
    out["detectable_ln_ratio"] = detectable
    target = (abs(float(ln_ratio_under_test)) if np.isfinite(ln_ratio_under_test)
              and ln_ratio_under_test != 0 else float(params.min_detectable_ln_ratio))
    out["ln_ratio_under_test"] = float(target)
    if abs(z) >= float(params.n_agree):
        out["like_for_like_verdict"] = LL_CHANGED
        return out
    out["measured_eras_agree"] = True
    # How far BELOW the claimed change the observation sits, in sigma.  Note
    # |z| * sigma is |ln ratio observed|, so this is target/sigma - |z|.  It is
    # at least as strict as `detectable <= target`: agreeing with zero is not
    # the same as excluding the claim, and only the second one kills a candidate.
    sep = (abs(float(target)) - abs(z) * float(sig)) / float(sig) if sig > 0 else float("nan")
    out["claim_excluded_sigma"] = sep
    out["like_for_like_verdict"] = (LL_UNCHANGED if np.isfinite(sep)
                                    and sep >= float(params.n_agree) else LL_UNRESOLVED)
    return out


def compare_kepler_to_catalogue(depth_measured_ppm: float, err_measured_ppm: float,
                                koi_depth_ppm: float, koi_depth_err_ppm: float, *,
                                params: CompareParams | None = None,
                                unchecked_reason: str | None = None) -> dict:
    """OUR Kepler-era fit against ``cumulative.koi_depth``.  **Same band, no ratio.**

    Both numbers describe the Kepler bandpass, so applying
    :func:`seti.growth.drift.band_ratio` here would be wrong --- the band ratio
    exists to move a Kepler-band number into the TESS band, and there is no
    band change to correct.  What is left is a direct test of the KOI table on
    this object, and if it fails at high significance **that is a finding about
    the catalogue stage 1 is built on** and is reported as
    ``KOI_DEPTH_CONTRADICTED``, not absorbed into the depth story.
    """
    params = params or CompareParams()
    out = {"koi_depth_verdict": KOI_DEPTH_UNCHECKED, "koi_depth_unchecked_reason": "",
           "z_kepler_measured_vs_koi": float("nan"), "sigma_kepler_measured_vs_koi": float("nan"),
           "kepler_measured_over_koi_depth": float("nan"),
           "koi_check_n_agree": float(params.koi_check_n_agree)}
    if unchecked_reason or not (np.isfinite(depth_measured_ppm) and depth_measured_ppm > 0):
        out["koi_depth_unchecked_reason"] = unchecked_reason or UNMEASURED_NO_TRANSIT
        return out
    if not (np.isfinite(koi_depth_ppm) and koi_depth_ppm > 0):
        out["koi_depth_unchecked_reason"] = UNMEASURED_NO_REFERENCE
        return out
    z, sig = _z_ln(float(depth_measured_ppm), float(err_measured_ppm), float(koi_depth_ppm),
                   float(koi_depth_err_ppm), params.sigma_sys_ln)
    out["z_kepler_measured_vs_koi"], out["sigma_kepler_measured_vs_koi"] = z, sig
    out["kepler_measured_over_koi_depth"] = float(depth_measured_ppm) / float(koi_depth_ppm)
    if not np.isfinite(z):
        out["koi_depth_unchecked_reason"] = "Z_NOT_COMPUTABLE"
        return out
    out["koi_depth_verdict"] = (KOI_DEPTH_CONTRADICTED if abs(z) >= float(params.koi_check_n_agree)
                                else KOI_DEPTH_CONFIRMED)
    return out


def run_verdict(measurements: pd.DataFrame) -> tuple[str, str]:
    """``(verdict, reason)`` for the whole stage from the per-target verdicts."""
    if measurements is None or not len(measurements) or "verdict" not in measurements:
        return RUN_NO_DATA, "no_targets"
    v = measurements["verdict"].fillna(MATCH_UNMEASURED).astype(str)
    measured = v[v != MATCH_UNMEASURED]
    if not len(measured):
        reasons = sorted({str(r) for r in measurements.get(
            "unmeasured_reason", pd.Series(dtype=str)).fillna("unknown") if str(r)})
        return RUN_NO_DATA, "no_target_measured:" + ",".join(r for r in reasons if r)
    if (measured == MATCH_TOI).any():
        return RUN_CONFIRMED, f"{int((measured == MATCH_TOI).sum())} target(s) matched the TOI depth"
    if (measured == MATCH_KEPLER).all():
        return RUN_REFUTED, f"all {len(measured)} measured target(s) matched the Kepler depth"
    return RUN_UNRESOLVED, (f"{int((measured == MATCH_NEITHER).sum())} matched neither, "
                            f"{int((measured == MATCH_BOTH).sum())} matched both, "
                            f"{int((measured == MATCH_KEPLER).sum())} matched Kepler")


def run_like_for_like_verdict(measurements: pd.DataFrame) -> tuple[str, str]:
    """``(verdict, reason)`` for the **primary** comparison over the whole stage.

    ``MEASURED_DEPTH_UNRESOLVED`` rows are never counted as refutations: a
    stage in which every target agreed *without the power to disagree* is
    ``MEASURED_DEPTH_CHANGE_UNRESOLVED``, not ``..._REFUTED``.
    """
    col = "like_for_like_verdict"
    if measurements is None or not len(measurements) or col not in measurements:
        return RUN_LL_NO_DATA, "no_targets"
    v = measurements[col].fillna(LL_ERA_UNMEASURED).astype(str)
    resolved = v[v != LL_ERA_UNMEASURED]
    if not len(resolved):
        reasons = sorted({str(r) for r in measurements.get(
            "like_for_like_unmeasured_reason", pd.Series(dtype=str)).fillna("") if str(r)})
        return RUN_LL_NO_DATA, "no_target_compared_like_for_like:" + ",".join(reasons)
    n_ch = int((resolved == LL_CHANGED).sum())
    n_un = int((resolved == LL_UNCHANGED).sum())
    n_ur = int((resolved == LL_UNRESOLVED).sum())
    if n_ch:
        return RUN_LL_CHANGED, (f"{n_ch} target(s) changed depth between OUR Kepler-era and OUR "
                                f"TESS-era fit of the light curves")
    if n_un and not n_ur:
        return RUN_LL_UNCHANGED, (f"all {n_un} like-for-like comparison(s) agree, with the power "
                                  f"to have seen the change under test")
    return RUN_LL_UNRESOLVED, (f"{n_ur} comparison(s) agreed without the power to detect the "
                               f"change under test, {n_un} agreed with it")


# ---------------------------------------------------------------------------
# Synthetic light curves (injection: the test gate, and the runner's self-check)
# ---------------------------------------------------------------------------
def trapezoid_transit(dt, duration_days: float, depth: float, ingress_frac: float = 0.1):
    """A trapezoidal transit: flat bottom ``depth``, ingress/egress ``ingress_frac * T14``."""
    d = np.abs(np.asarray(dt, dtype=float))
    half = 0.5 * float(duration_days)
    tau = max(float(ingress_frac) * float(duration_days), 1e-12)
    out = np.zeros_like(d)
    flat = d <= (half - tau)
    slope = (d > (half - tau)) & (d < half)
    out[flat] = 1.0
    out[slope] = (half - d[slope]) / tau
    return 1.0 - float(depth) * out


def synth_lightcurve(*, period_days: float, t0_btjd: float, duration_days: float,
                     depth: float, exptime_s: float = 120.0, n_transits: int = 12,
                     span_durations: float = 3.0, noise_ppm: float = 500.0,
                     sector: int = 1, author: str = "SPOC", seed: int = 11,
                     odd_depth: float | None = None, ingress_frac: float = 0.1,
                     oversample: int = 11) -> dict:
    """One synthetic sector with a KNOWN injected depth, integrated over the cadence.

    The transit is trapezoidal and is **integrated across each exposure** by
    oversampling, so a long cadence really does smear the transit the way an
    FFI light curve does --- that is what makes the smearing flag testable.
    ``odd_depth`` gives odd-numbered epochs a different depth: a synthetic
    eclipsing binary for the odd-even test.
    """
    rng = np.random.default_rng(int(seed))
    exp_d = float(exptime_s) / 86400.0
    half_span = float(span_durations) * float(duration_days)
    times, fluxes = [], []
    for k in range(int(n_transits)):
        centre = float(t0_btjd) + k * float(period_days)
        n = max(int(round(2 * half_span / exp_d)), 8)
        t = centre + np.linspace(-half_span, half_span, n)
        sub = np.linspace(-0.5, 0.5, max(int(oversample), 1)) * exp_d
        dt_grid = (t[:, None] + sub[None, :]) - centre
        dep = float(depth) if (k % 2 == 0 or odd_depth is None) else float(odd_depth)
        model = trapezoid_transit(dt_grid, duration_days, dep, ingress_frac).mean(axis=1)
        times.append(t)
        fluxes.append(model)
    t = np.concatenate(times)
    f = np.concatenate(fluxes)
    f = f + rng.normal(0.0, float(noise_ppm) * 1e-6, size=f.shape)
    return {"sector": int(sector), "author": str(author), "exptime_s": float(exptime_s),
            "flux_column": "PDCSAP_FLUX", "time": t, "flux": f,
            "flux_err": np.full(f.shape, float(noise_ppm) * 1e-6), "n_points": int(t.size)}


# ---------------------------------------------------------------------------
# MAST access (runner only; injectable everywhere)
# ---------------------------------------------------------------------------
def mast_probe() -> dict:
    """Which route to MAST this machine actually has.  Imports only; no query."""
    rep = {"lightkurve": {"importable": False}, "astroquery_mast": {"importable": False},
           "astropy_io_fits": {"importable": False}, "route_preferred": None}
    try:
        import lightkurve as lk  # noqa: PLC0415

        rep["lightkurve"] = {"importable": True, "version": str(getattr(lk, "__version__", "?"))}
    except Exception as exc:                              # noqa: BLE001
        rep["lightkurve"] = {"importable": False, "error": repr(exc)[:300]}
    try:
        import astroquery  # noqa: PLC0415
        from astroquery.mast import Observations  # noqa: PLC0415, F401

        rep["astroquery_mast"] = {"importable": True,
                                  "version": str(getattr(astroquery, "__version__", "?"))}
    except Exception as exc:                              # noqa: BLE001
        rep["astroquery_mast"] = {"importable": False, "error": repr(exc)[:300]}
    try:
        from astropy.io import fits  # noqa: PLC0415, F401

        rep["astropy_io_fits"] = {"importable": True}
    except Exception as exc:                              # noqa: BLE001
        rep["astropy_io_fits"] = {"importable": False, "error": repr(exc)[:300]}
    if rep["lightkurve"]["importable"]:
        rep["route_preferred"] = ROUTE_LIGHTKURVE
    elif rep["astroquery_mast"]["importable"] and rep["astropy_io_fits"]["importable"]:
        rep["route_preferred"] = ROUTE_MAST_FITS
    rep["note"] = ("import reachability only; whether MAST ANSWERS is established by the "
                   "measure stage and recorded per target as OK / QUERY_FAILED / "
                   "QUERY_RETURNED_ZERO_ROWS. The SAME route serves both eras: TESS by "
                   "TIC (search_lightcurve('TIC n', mission='TESS') / "
                   "query_criteria(obs_collection='TESS')) and KEPLER by KIC "
                   "(search_lightcurve('KIC n', mission='Kepler') / "
                   "query_criteria(obs_collection='Kepler')), so an importable route does "
                   "NOT imply that both collections answer — that is a per-era measure-stage "
                   "fact, recorded as lc_status and kepler_lc_status")
    rep["eras"] = list(ERAS)
    return rep


def _exptime_from_header(hdr) -> float:
    """Cadence in SECONDS from a TESS light-curve header.

    ``TIMEDEL`` is the time between samples in **days**; ``FRAMETIM * NUM_FRM``
    is the co-added frame time in **seconds** and is the fallback.  Nothing
    here guesses from the data: a header that says neither returns NaN, and a
    NaN exposure means the smearing test cannot fire, which the record shows.
    """
    try:
        v = hdr.get("TIMEDEL")
        if v is not None and np.isfinite(float(v)) and float(v) > 0:
            return float(v) * 86400.0
    except Exception:                                     # noqa: BLE001
        pass
    try:
        ft = float(hdr.get("FRAMETIM") or np.nan)
        nf = float(hdr.get("NUM_FRM") or np.nan)
        if np.isfinite(ft) and np.isfinite(nf) and ft > 0 and nf > 0:
            return ft * nf
    except Exception:                                     # noqa: BLE001
        pass
    return float("nan")


def read_tess_lc_fits(path, *, flux_columns=FLUX_COLUMNS) -> dict | None:
    """One TESS light-curve FITS product -> ``{time (BTJD), flux, flux_err, meta}``.

    ``TIME`` is converted through the file's own ``BJDREFI``/``BJDREFF`` rather
    than assumed to be BTJD, and the flux column that was actually found is
    recorded.  Quality is masked when a ``QUALITY`` column exists.
    """
    from astropy.io import fits  # noqa: PLC0415

    with fits.open(str(path), memmap=False) as hdul:
        hdu = None
        for h in hdul:
            try:
                names = [str(n).upper() for n in h.columns.names]
            except AttributeError:
                continue
            if "TIME" in names:
                hdu = h
                break
        if hdu is None:
            return None
        data, hdr = hdu.data, hdu.header
        cols = [str(n).upper() for n in hdu.columns.names]
        fcol = next((c for c in flux_columns if c in cols), None)
        if fcol is None:
            return None
        bjdrefi = float(hdr.get("BJDREFI", hdul[0].header.get("BJDREFI", BTJD_OFFSET)) or
                        BTJD_OFFSET)
        bjdreff = float(hdr.get("BJDREFF", hdul[0].header.get("BJDREFF", 0.0)) or 0.0)
        t = np.asarray(data["TIME"], dtype=float) + (bjdrefi + bjdreff) - BTJD_OFFSET
        f = np.asarray(data[fcol], dtype=float)
        ecol = fcol + "_ERR"
        fe = (np.asarray(data[ecol], dtype=float) if ecol in cols
              else np.full(f.shape, np.nan, dtype=float))
        n_before = int(t.size)
        if "QUALITY" in cols:
            q = np.asarray(data["QUALITY"])
            keep = (q == 0)
            t, f, fe = t[keep], f[keep], fe[keep]
        ok = np.isfinite(t) & np.isfinite(f)
        t, f, fe = t[ok], f[ok], fe[ok]
        sector = hdul[0].header.get("SECTOR", hdr.get("SECTOR"))
        author = str(hdul[0].header.get("PROCVER", "") or hdul[0].header.get("ORIGIN", "")
                     or "unknown")
        return {"sector": int(sector) if sector is not None else None, "author": author,
                "exptime_s": _exptime_from_header(hdr), "flux_column": fcol,
                "time": t, "flux": f, "flux_err": fe, "n_points": int(t.size),
                "n_points_before_quality_mask": n_before,
                "bjdref": bjdrefi + bjdreff}


def kepler_cadence_label(exptime_s: float) -> str:
    """``"short"`` / ``"long"`` / ``"unknown"`` from the exposure in seconds."""
    e = float(exptime_s) if exptime_s is not None else float("nan")
    if not np.isfinite(e) or e <= 0:
        return "unknown"
    return "short" if e <= KEPLER_SHORT_CADENCE_MAX_S else "long"


def read_kepler_lc_fits(path, *, flux_columns=KEPLER_FLUX_COLUMNS) -> dict | None:
    """One Kepler light-curve FITS product -> ``{time (BKJD), flux, flux_err, meta}``.

    The mirror of :func:`read_tess_lc_fits`, with the three differences that
    actually exist between the two products:

    * the time axis is converted through the file's own ``BJDREFI``/``BJDREFF``
      to **BKJD** (``BJD - 2454833``) --- the system ``koi_time0bk`` is already
      in, so nothing downstream converts it (:func:`epoch_in_era`);
    * quality lives in ``SAP_QUALITY``, not ``QUALITY``;
    * the segment identifier is the **quarter**, and the cadence (29.4-minute
      long or 58.85-second short) is recorded, because the two are different
      measurements of the same transit and the long one is ``smeared``.

    ``sector`` carries the quarter as well, so :func:`dedupe_sectors` and
    :func:`measure_target` work on Kepler data without being changed.
    """
    from astropy.io import fits  # noqa: PLC0415

    with fits.open(str(path), memmap=False) as hdul:
        hdu = None
        for h in hdul:
            try:
                names = [str(n).upper() for n in h.columns.names]
            except AttributeError:
                continue
            if "TIME" in names:
                hdu = h
                break
        if hdu is None:
            return None
        data, hdr = hdu.data, hdu.header
        cols = [str(n).upper() for n in hdu.columns.names]
        fcol = next((c for c in flux_columns if c in cols), None)
        if fcol is None:
            return None
        bjdrefi = float(hdr.get("BJDREFI", hdul[0].header.get("BJDREFI", BKJD_OFFSET)) or
                        BKJD_OFFSET)
        bjdreff = float(hdr.get("BJDREFF", hdul[0].header.get("BJDREFF", 0.0)) or 0.0)
        # -> BKJD.  NOT BTJD: this is the Kepler era.
        t = np.asarray(data["TIME"], dtype=float) + (bjdrefi + bjdreff) - BKJD_OFFSET
        f = np.asarray(data[fcol], dtype=float)
        ecol = fcol + "_ERR"
        fe = (np.asarray(data[ecol], dtype=float) if ecol in cols
              else np.full(f.shape, np.nan, dtype=float))
        n_before = int(t.size)
        qcol = next((c for c in KEPLER_QUALITY_COLUMNS if c in cols), None)
        if qcol is not None:
            keep = (np.asarray(data[qcol]) == 0)
            t, f, fe = t[keep], f[keep], fe[keep]
        ok = np.isfinite(t) & np.isfinite(f)
        t, f, fe = t[ok], f[ok], fe[ok]
        quarter = hdul[0].header.get("QUARTER", hdr.get("QUARTER"))
        exptime_s = _exptime_from_header(hdr)
        return {"sector": int(quarter) if quarter is not None else None,
                "quarter": int(quarter) if quarter is not None else None,
                "era": ERA_KEPLER, "author": "Kepler",
                "exptime_s": exptime_s, "cadence": kepler_cadence_label(exptime_s),
                "obsmode": str(hdul[0].header.get("OBSMODE", hdr.get("OBSMODE", "")) or ""),
                "flux_column": fcol, "quality_column": qcol,
                "time": t, "flux": f, "flux_err": fe, "n_points": int(t.size),
                "n_points_before_quality_mask": n_before,
                "bjdref": bjdrefi + bjdreff}


def merge_kepler_segments(segments) -> tuple[list[dict], list[dict]]:
    """Kepler month-files -> one light curve per (quarter, cadence).

    **Why this has to happen before :func:`dedupe_sectors`.**  Kepler
    short-cadence data is delivered *per month*: three files carry one quarter,
    all stamped with the same ``QUARTER``.  ``dedupe_sectors`` keys on the
    segment identifier and keeps one light curve per key, so handing it three
    month-files of Q9 unchanged would throw two thirds of the short-cadence
    data away.  Merging first makes the key mean what dedupe assumes it means:
    *one light curve per quarter per pipeline product*.  Long and short cadence
    of the same quarter stay separate records at this point --- they are the
    same pixels and it is dedupe's job, not this function's, to prefer the
    shorter one.

    Each segment is divided by **its own robust median before concatenation**,
    so a month-to-month flux scale cannot survive into the merged quarter (the
    per-sector normalisation in :func:`measure_target` then finds a series that
    is already at unity).  Returns ``(merged, provenance)``.
    """
    groups: dict = {}
    order: list = []
    for s in segments or []:
        q = s.get("quarter", s.get("sector"))
        e = s.get("exptime_s")
        try:
            ekey = round(float(e)) if e is not None and np.isfinite(float(e)) else None
        except (TypeError, ValueError):
            ekey = None
        key = (q, ekey) if q is not None else ("__unkeyed__", len(order))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s)
    merged: list[dict] = []
    prov: list[dict] = []
    for key in order:
        parts = groups[key]
        times, fluxes, errs = [], [], []
        for s in parts:
            t = np.asarray(s.get("time"), dtype=float)
            f = np.asarray(s.get("flux"), dtype=float)
            fe = (np.asarray(s.get("flux_err"), dtype=float) if s.get("flux_err") is not None
                  else np.full(t.shape, np.nan))
            ok = np.isfinite(t) & np.isfinite(f)
            t, f, fe = t[ok], f[ok], fe[ok]
            med = float(np.median(f)) if t.size else float("nan")
            if not np.isfinite(med) or med == 0:
                continue
            times.append(t)
            fluxes.append(f / med)
            errs.append(fe / med)
        if not times:
            merged.append(dict(parts[0]))
            continue
        idx = np.argsort(np.concatenate(times))
        base = dict(parts[0])
        base.update({"time": np.concatenate(times)[idx],
                     "flux": np.concatenate(fluxes)[idx],
                     "flux_err": np.concatenate(errs)[idx],
                     "n_points": int(idx.size), "n_files_merged": int(len(parts)),
                     "normalised_per_file": True})
        base["n_points"] = int(base["time"].size)
        merged.append(base)
        prov.append({"quarter": base.get("quarter", base.get("sector")),
                     "cadence": base.get("cadence"), "exptime_s": base.get("exptime_s"),
                     "n_files_merged": int(len(parts)), "n_points": int(base["n_points"])})
    return merged, prov


def _select_flux_column(lc, flux_columns):
    """Switch a ``lightkurve`` light curve to a REQUESTED flux column, or fail.

    Returns ``(lc, column_name)``, or ``(None, "")`` when the product does not
    carry the requested column.  Failing is the point: an ensemble member that
    asked for ``SAP_FLUX`` and silently got ``PDCSAP_FLUX`` would put the same
    reduction into the ensemble twice and shrink the spread, which is the exact
    overconfidence the ensemble exists to remove.  ``flux_columns=None`` leaves
    the product's own default alone (the primary measurement's behaviour).
    """
    if not flux_columns:
        meta = dict(getattr(lc, "meta", {}) or {})
        return lc, str(meta.get("FLUX_ORIGIN") or "PDCSAP_FLUX").upper()
    for col in flux_columns:
        name = str(col)
        try:
            if name.lower() not in {str(c).lower() for c in getattr(lc, "columns", [])}:
                continue
            return lc.select_flux(name.lower()), name.upper()
        except Exception:                                 # noqa: BLE001
            continue
    return None, ""


def lightkurve_lc_fn(tic_id, *, authors=DEFAULT_AUTHORS, max_sectors: int = 60,
                     download_dir: str | None = None, flux_columns=None,
                     **_kw) -> list[dict]:
    """Light curves for one TIC through ``lightkurve`` (runner only).

    The time axis is taken as ``Time.jd - 2457000`` rather than trusting the
    object's format string, and the author, sector and exposure time of every
    sector are carried through so the record says what was measured on what.
    ``flux_columns``, when given (the reduction ensemble gives exactly one),
    selects that column and **drops the product when it does not have it**.
    """
    import lightkurve as lk  # noqa: PLC0415

    sr = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
    if sr is None or len(sr) == 0:
        return []
    try:
        tbl = sr.table
        auth = np.array([str(a) for a in tbl["author"]])
        keep = np.isin(auth, list(authors))
        if keep.any():
            sr = sr[keep]
    except Exception:                                     # noqa: BLE001
        pass
    out: list[dict] = []
    for i in range(min(len(sr), int(max_sectors))):
        try:
            lc = sr[i].download(download_dir=download_dir)
        except Exception:                                 # noqa: BLE001
            continue
        if lc is None:
            continue
        lc, fcol = _select_flux_column(lc, flux_columns)
        if lc is None:
            continue                       # this product has no such column: absent, not failed
        try:
            lc = lc.remove_nans()
        except Exception:                                 # noqa: BLE001
            pass
        try:
            t = np.asarray(lc.time.jd, dtype=float) - BTJD_OFFSET
        except Exception:                                 # noqa: BLE001
            t = np.asarray(lc.time.value, dtype=float)
        f = np.asarray(getattr(lc.flux, "value", lc.flux), dtype=float)
        fe = np.asarray(getattr(getattr(lc, "flux_err", None), "value",
                                np.full(f.shape, np.nan)), dtype=float)
        meta = dict(getattr(lc, "meta", {}) or {})
        exptime = meta.get("TIMEDEL")
        exptime_s = float(exptime) * 86400.0 if exptime else float("nan")
        if not np.isfinite(exptime_s) and t.size > 2:
            exptime_s = float(np.median(np.diff(np.sort(t)))) * 86400.0
        out.append({"sector": int(meta.get("SECTOR")) if meta.get("SECTOR") else None,
                    "author": str(meta.get("AUTHOR") or meta.get("ORIGIN") or "unknown"),
                    "exptime_s": exptime_s, "flux_column": fcol,
                    "time": t, "flux": f, "flux_err": fe, "n_points": int(t.size)})
    return out


def mast_fits_lc_fn(tic_id, *, authors=DEFAULT_AUTHORS, max_sectors: int = 60,
                    download_dir: str | None = None, flux_columns=None, **_kw) -> list[dict]:
    """Light curves for one TIC through ``astroquery.mast`` + FITS (runner only).

    The fallback when ``lightkurve`` is not installed: query the TESS
    timeseries observations for the TIC, take the ``LC`` products, download by
    ``dataURI`` and read them with :func:`read_tess_lc_fits`.
    """
    from astroquery.mast import Observations  # noqa: PLC0415

    obs = Observations.query_criteria(obs_collection="TESS", dataproduct_type="timeseries",
                                      target_name=str(int(tic_id)))
    if obs is None or len(obs) == 0:
        return []
    prod = Observations.get_product_list(obs)
    if prod is None or len(prod) == 0:
        return []
    p = prod.to_pandas()
    if "productSubGroupDescription" in p.columns:
        p = p[p["productSubGroupDescription"].astype(str).str.upper() == "LC"]
    if "provenance_name" in p.columns and len(authors):
        want = {str(a).upper() for a in authors}
        sel = p["provenance_name"].astype(str).str.upper().isin(want)
        if sel.any():
            p = p[sel]
    p = p.head(int(max_sectors))
    root = Path(download_dir or ".") / "mast_lc"
    root.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for _, r in p.iterrows():
        uri = str(r.get("dataURI") or "")
        if not uri:
            continue
        local = root / str(r.get("productFilename") or Path(uri).name)
        try:
            status, _msg, _url = Observations.download_file(uri, local_path=str(local),
                                                            cache=False)
            if str(status).upper() != "COMPLETE" or not local.exists():
                continue
            rec = read_tess_lc_fits(local, flux_columns=tuple(flux_columns or FLUX_COLUMNS))
        except Exception:                                 # noqa: BLE001
            continue
        if rec is None:
            continue
        prov = str(r.get("provenance_name") or "").strip()
        if prov:
            rec["author"] = prov
        out.append(rec)
    return out


def default_lc_fn(tic_id, **kw) -> list[dict]:
    """``lightkurve`` if it is installed, else ``astroquery.mast`` + FITS."""
    try:
        import lightkurve  # noqa: PLC0415, F401

        return lightkurve_lc_fn(tic_id, **kw)
    except ImportError:
        return mast_fits_lc_fn(tic_id, **kw)


# ---------------------------------------------------------------------------
# The KEPLER era, reached the same two ways
# ---------------------------------------------------------------------------
def lightkurve_kepler_lc_fn(kepid, *, authors=KEPLER_AUTHORS, max_sectors: int = 60,
                            download_dir: str | None = None, flux_columns=None,
                            **_kw) -> list[dict]:
    """Kepler light curves for one KIC through ``lightkurve`` (runner only).

    ``search_lightcurve("KIC <kepid>", mission="Kepler")``, both cadences.  The
    time axis is taken as ``Time.jd - 2454833`` (**BKJD**) rather than trusting
    the object's format string, exactly as the TESS route takes ``jd - 2457000``;
    quarter, cadence, flux column and exposure are carried through so the record
    says what was measured on what.  Short cadence is *not* filtered out here:
    :func:`merge_kepler_segments` merges its month-files into quarters and
    :func:`dedupe_sectors` then prefers it over the 29.4-minute long cadence.
    """
    import lightkurve as lk  # noqa: PLC0415

    sr = lk.search_lightcurve(f"KIC {int(kepid)}", mission="Kepler")
    if sr is None or len(sr) == 0:
        return []
    try:
        tbl = sr.table
        auth = np.array([str(a) for a in tbl["author"]])
        keep = np.isin(auth, list(authors))
        if keep.any():
            sr = sr[keep]
    except Exception:                                     # noqa: BLE001
        pass
    out: list[dict] = []
    for i in range(min(len(sr), int(max_sectors))):
        try:
            lc = sr[i].download(download_dir=download_dir)
        except Exception:                                 # noqa: BLE001
            continue
        if lc is None:
            continue
        lc, fcol = _select_flux_column(lc, flux_columns)
        if lc is None:
            continue                       # this product has no such column: absent, not failed
        try:
            lc = lc.remove_nans()
        except Exception:                                 # noqa: BLE001
            pass
        try:
            t = np.asarray(lc.time.jd, dtype=float) - BKJD_OFFSET
        except Exception:                                 # noqa: BLE001
            t = np.asarray(lc.time.value, dtype=float)
        f = np.asarray(getattr(lc.flux, "value", lc.flux), dtype=float)
        fe = np.asarray(getattr(getattr(lc, "flux_err", None), "value",
                                np.full(f.shape, np.nan)), dtype=float)
        meta = dict(getattr(lc, "meta", {}) or {})
        exptime = meta.get("TIMEDEL")
        exptime_s = float(exptime) * 86400.0 if exptime else float("nan")
        if not np.isfinite(exptime_s) and t.size > 2:
            exptime_s = float(np.median(np.diff(np.sort(t)))) * 86400.0
        quarter = meta.get("QUARTER")
        out.append({"sector": int(quarter) if quarter is not None else None,
                    "quarter": int(quarter) if quarter is not None else None,
                    "era": ERA_KEPLER,
                    "author": str(meta.get("AUTHOR") or meta.get("MISSION") or "Kepler"),
                    "exptime_s": exptime_s, "cadence": kepler_cadence_label(exptime_s),
                    "obsmode": str(meta.get("OBSMODE") or ""), "flux_column": fcol,
                    "time": t, "flux": f, "flux_err": fe, "n_points": int(t.size)})
    return out


#: The spellings of a Kepler target name tried at MAST, in order.  ``kplr`` +
#: the nine-digit zero-padded KIC is the archive's own form; the bare integer is
#: the fallback.  WHICH ONE ANSWERED cannot be established in this sandbox and
#: is established at runtime -- a spelling that returns nothing is
#: ``QUERY_RETURNED_ZERO_ROWS``, which is not ``QUERY_FAILED``, so the loop
#: moves on to the next spelling instead of calling the target unreachable.
KEPLER_TARGET_NAME_FORMS: tuple[str, ...] = ("kplr{kepid:09d}", "{kepid:d}", "KIC {kepid:d}")


def mast_kepler_fits_lc_fn(kepid, *, authors=KEPLER_AUTHORS, max_sectors: int = 60,
                           download_dir: str | None = None, flux_columns=None,
                           **_kw) -> list[dict]:
    """Kepler light curves for one KIC through ``astroquery.mast`` + FITS.

    The fallback when ``lightkurve`` is not installed: query the Kepler
    timeseries observations for the KIC, keep the ``LLC`` / ``SLC`` products,
    download by ``dataURI`` and read them with :func:`read_kepler_lc_fits`.
    """
    from astroquery.mast import Observations  # noqa: PLC0415

    obs = None
    for form in KEPLER_TARGET_NAME_FORMS:
        try:
            cand = Observations.query_criteria(obs_collection="Kepler",
                                               dataproduct_type="timeseries",
                                               target_name=form.format(kepid=int(kepid)))
        except Exception:                                 # noqa: BLE001
            continue
        if cand is not None and len(cand):
            obs = cand
            break
    if obs is None or len(obs) == 0:
        return []
    prod = Observations.get_product_list(obs)
    if prod is None or len(prod) == 0:
        return []
    p = prod.to_pandas()
    if "productSubGroupDescription" in p.columns:
        want = {s.upper() for s in KEPLER_PRODUCT_SUBGROUPS}
        sel = p["productSubGroupDescription"].astype(str).str.upper().isin(want)
        if sel.any():
            p = p[sel]
    if "provenance_name" in p.columns and len(authors):
        wanted = {str(a).upper() for a in authors}
        sel = p["provenance_name"].astype(str).str.upper().isin(wanted)
        if sel.any():
            p = p[sel]
    p = p.head(int(max_sectors))
    root = Path(download_dir or ".") / "mast_kepler_lc"
    root.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for _, r in p.iterrows():
        uri = str(r.get("dataURI") or "")
        if not uri:
            continue
        local = root / str(r.get("productFilename") or Path(uri).name)
        try:
            status, _msg, _url = Observations.download_file(uri, local_path=str(local),
                                                            cache=False)
            if str(status).upper() != "COMPLETE" or not local.exists():
                continue
            rec = read_kepler_lc_fits(
                local, flux_columns=tuple(flux_columns or KEPLER_FLUX_COLUMNS))
        except Exception:                                 # noqa: BLE001
            continue
        if rec is None:
            continue
        out.append(rec)
    return out


def default_kepler_lc_fn(kepid, **kw) -> list[dict]:
    """``lightkurve`` if it is installed, else ``astroquery.mast`` + FITS."""
    try:
        import lightkurve  # noqa: PLC0415, F401

        return lightkurve_kepler_lc_fn(kepid, **kw)
    except ImportError:
        return mast_kepler_fits_lc_fn(kepid, **kw)


def _fetch_lcs(target_id, *, lc_fn, default_fn, params: MastParams, log: AcquisitionLog,
               deadline: Deadline | None, label: str, query: str,
               flux_columns=None) -> tuple[list[dict], str, str]:
    """The bounded, recorded fetch loop shared by both eras.

    Returns ``(segments, status, route)``; ``status`` is ``OK`` /
    ``QUERY_RETURNED_ZERO_ROWS`` / ``QUERY_FAILED``, and a failure is **never**
    turned into an empty-but-successful answer.  Retries stop as soon as the
    wall-clock budget is gone.  One loop, so the Kepler era cannot drift away
    from the TESS era's error handling -- and so the reduction ensemble's member
    fetches go through exactly the same error accounting as the primary one.

    ``flux_columns`` is passed to the fetch function only when it is given, so
    the primary measurement's call is byte-for-byte what it always was; an
    ensemble member asks for ONE column and nothing else.
    """
    if deadline is not None and deadline.expired():
        log.record(label, query, error="budget_exhausted_before_request")
        return [], STATUS_FAILED, ""
    route = "injected" if lc_fn is not None else (mast_probe().get("route_preferred") or "")
    fn = lc_fn or default_fn
    last = ""
    own = Deadline(budget_s=float(params.per_target_budget_s))
    n_attempts = max(int(params.retries), 1)
    for attempt in range(n_attempts):
        if own.expired() or (deadline is not None and deadline.expired()):
            last = "budget_exhausted"
            break
        if attempt and float(params.retry_pause_s) > 0:
            # A backoff, but never one that outlives the budget it sits inside.
            _time.sleep(min(float(params.retry_pause_s) * attempt, own.remaining(),
                            deadline.remaining() if deadline is not None else float("inf")))
        kw = {"authors": tuple(params.authors), "max_sectors": int(params.max_sectors),
              "download_dir": params.download_dir}
        if flux_columns is not None:
            kw["flux_columns"] = tuple(flux_columns)
        try:
            secs = fn(target_id, **kw)
        except Exception as exc:                          # noqa: BLE001
            last = repr(exc)[:400]
            continue
        secs = [s for s in (secs or []) if s is not None and len(np.asarray(s.get("time"),
                                                                           dtype=float))]
        if not secs:
            log.record(label, query, rows=0, extra={"route": route, "attempt": attempt + 1})
            return [], STATUS_ZERO, route
        log.record(label, query, rows=int(sum(int(s.get("n_points") or 0) for s in secs)),
                   extra={"route": route, "n_sectors": len(secs), "attempt": attempt + 1})
        return secs, STATUS_OK, route
    log.record(label, query, error=last or "unknown")
    return [], STATUS_FAILED, route


def fetch_lightcurves(tic_id, *, lc_fn=None, params: MastParams | None = None,
                      log: AcquisitionLog | None = None, deadline: Deadline | None = None,
                      key: str = "", flux_columns=None) -> tuple[list[dict], str, str]:
    """One target's TESS light curves, bounded and recorded."""
    params = params or MastParams()
    log = log or AcquisitionLog()
    return _fetch_lcs(tic_id, lc_fn=lc_fn, default_fn=default_lc_fn, params=params, log=log,
                      deadline=deadline, label=f"lightcurves_{key or tic_id}",
                      query=f"TIC {tic_id}", flux_columns=flux_columns)


def fetch_kepler_lightcurves(kepid, *, lc_fn=None, params: MastParams | None = None,
                             log: AcquisitionLog | None = None, deadline: Deadline | None = None,
                             key: str = "", flux_columns=None
                             ) -> tuple[list[dict], str, str, list[dict]]:
    """One target's KEPLER light curves, through the same bounded loop.

    Returns ``(quarters, status, route, merge_provenance)``.  The archive
    delivers short cadence per *month*, so the segments are passed through
    :func:`merge_kepler_segments` before they are returned: what comes back is
    one light curve per (quarter, cadence), which is the unit
    :func:`dedupe_sectors` assumes when it prefers the shorter cadence.  The
    Kepler ceilings (``kepler_per_target_budget_s``, ``kepler_max_quarters``,
    ``kepler_retries``) are applied through :meth:`MastParams.kepler_view`.
    """
    params = params or MastParams()
    log = log or AcquisitionLog()
    segs, status, route = _fetch_lcs(kepid, lc_fn=lc_fn, default_fn=default_kepler_lc_fn,
                                     params=params.kepler_view(), log=log, deadline=deadline,
                                     label=f"kepler_lightcurves_{key or kepid}",
                                     query=f"KIC {kepid}", flux_columns=flux_columns)
    if status != STATUS_OK:
        return [], status, route, []
    merged, prov = merge_kepler_segments(segs)
    merged = [s for s in merged if len(np.asarray(s.get("time"), dtype=float))]
    if not merged:
        return [], STATUS_ZERO, route, prov
    return merged, STATUS_OK, route, prov


# ---------------------------------------------------------------------------
# Ephemerides from the Exoplanet Archive
# ---------------------------------------------------------------------------
def _quote_list(values) -> str:
    return ",".join("'" + str(v).replace("'", "") + "'" for v in values)


def fetch_koi_ephemerides(kepoi_names, *, query_fn=None, log: AcquisitionLog | None = None,
                          table: str = "cumulative", columns=KOI_EPH_COLUMNS
                          ) -> tuple[pd.DataFrame, str]:
    """``koi_period`` / ``koi_time0bk`` / ``koi_duration`` for the shortlist.

    ``koi_time0bk`` is NOT in the stage-1 column list, so stage 2 must pull it:
    the epoch is what makes a fold possible and it cannot be reconstructed from
    ``joined.csv``.  A failure is ``QUERY_FAILED``; an empty answer is
    ``QUERY_RETURNED_ZERO_ROWS``; neither becomes an ephemeris.
    """
    log = log or AcquisitionLog()
    names = [str(n) for n in kepoi_names if str(n) and str(n).lower() != "nan"]
    if not names:
        return pd.DataFrame(), STATUS_ZERO
    adql = (f"select {','.join(columns)} from {table} "
            f"where kepoi_name in ({_quote_list(names)})")
    qf = query_fn or tap_sync
    try:
        df = qf(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record("fetch_koi_ephemerides", adql, error=repr(exc))
        return pd.DataFrame(), STATUS_FAILED
    if df is None:
        log.record("fetch_koi_ephemerides", adql, error="query_fn returned None")
        return pd.DataFrame(), STATUS_FAILED
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    log.record("fetch_koi_ephemerides", adql, rows=int(len(df)))
    return df, (STATUS_OK if len(df) else STATUS_ZERO)


def fetch_toi_depths(tids, *, query_fn=None, log: AcquisitionLog | None = None,
                     table: str = "toi", columns=TOI_DEPTH_COLUMNS) -> tuple[pd.DataFrame, str]:
    """The TOI rows for the shortlist's TIC ids (the deep catalogue value under test)."""
    log = log or AcquisitionLog()
    ids = []
    for t in tids:
        try:
            ids.append(str(int(float(t))))
        except Exception:                                 # noqa: BLE001
            continue
    if not ids:
        return pd.DataFrame(), STATUS_ZERO
    adql = f"select {','.join(columns)} from {table} where tid in ({','.join(sorted(set(ids)))})"
    qf = query_fn or tap_sync
    try:
        df = qf(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record("fetch_toi_depths", adql, error=repr(exc))
        return pd.DataFrame(), STATUS_FAILED
    if df is None:
        log.record("fetch_toi_depths", adql, error="query_fn returned None")
        return pd.DataFrame(), STATUS_FAILED
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    log.record("fetch_toi_depths", adql, rows=int(len(df)))
    return df, (STATUS_OK if len(df) else STATUS_ZERO)


# ---------------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------------
def load_shortlist(conf: dict, *, candidates_csv: Path | None = None) -> pd.DataFrame:
    """The stage-2 shortlist: ``config/growth.yaml`` ``stage2.shortlist``, plus
    (when ``stage2.shortlist_from_candidates_csv``) the stage-1 candidate rows.

    Config entries win on a duplicate ``kepoi_name``.  Every row carries where
    it came from in ``shortlist_source``.
    """
    s2 = (conf or {}).get("stage2") or {}
    rows: list[dict] = []
    for e in s2.get("shortlist") or []:
        r = dict(e)
        r["shortlist_source"] = "config"
        rows.append(r)
    if s2.get("shortlist_from_candidates_csv", True) and candidates_csv is not None:
        p = Path(candidates_csv)
        if p.exists() and p.stat().st_size:
            try:
                c = pd.read_csv(p, low_memory=False)
            except Exception:                             # noqa: BLE001
                c = pd.DataFrame()
            for _, r in c.iterrows():
                rows.append({"kepoi_name": r.get("kepoi_name"),
                             "kepler_name": r.get("kepler_name"), "kepid": r.get("kepid"),
                             "tic_id": r.get("tic_id"), "toi": r.get("toi"),
                             "shortlist_source": "candidates_csv"})
    if not rows:
        return pd.DataFrame(columns=["kepoi_name", "kepler_name", "kepid", "tic_id", "toi",
                                     "shortlist_source"])
    df = pd.DataFrame(rows)
    df["kepoi_name"] = df["kepoi_name"].map(str)
    df = df.drop_duplicates(subset=["kepoi_name"], keep="first").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def stage2_probe(conf: dict, out: Path) -> dict:
    """Which route to MAST exists on this machine, written to ``probe.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"stage": "probe", "generated_utc": _now(), "mast": mast_probe(),
           "shortlist_size": int(len(load_shortlist(conf,
                                                    candidates_csv=_candidates_csv(out))))}
    _write(out / "probe.json", rep)
    print(f"[growth-stage2] probe: MAST route = {rep['mast'].get('route_preferred')}; "
          f"shortlist {rep['shortlist_size']}")
    return rep


def _candidates_csv(out: Path) -> Path:
    """Stage 1's ``candidates.csv``: ``results/growth/candidates.csv`` by default."""
    p = Path(out)
    for cand in (p / "candidates.csv", p.parent / "candidates.csv"):
        if cand.exists():
            return cand
    return p.parent / "candidates.csv"


#: The measure_target outputs carried into the per-target record for an era.
_ERA_FIT_FIELDS: tuple[str, ...] = (
    "depth_err_analytic_ppm", "depth_err_bootstrap_ppm", "depth_err_chi2_scaled_ppm",
    "chi2", "chi2_per_dof", "n_transits", "n_transits_seen", "oot_scatter_ppm",
    "n_sectors", "n_sectors_measured", "sector_list", "authors", "exptimes_s",
    "smeared", "depth_is_lower_bound", "depth_odd_ppm", "depth_odd_err_ppm",
    "n_transits_odd", "depth_even_ppm", "depth_even_err_ppm", "n_transits_even",
    "odd_even_diff_ppm", "odd_even_sigma", "sector_scatter_chi2",
    "sector_scatter_dof", "sector_scatter_chi2_per_dof", "n_sectors_dropped_duplicate",
    "duplicate_sectors_dropped",
)


def _measure_era(segments, *, era: str, period_days: float, t0_bkjd: float,
                 duration_hours: float, koi_row: dict | None, fit: FitParams
                 ) -> tuple[dict, dict, str | None]:
    """Fold and fit ONE era's light curves with :func:`measure_target`.

    The ONLY thing ``era`` changes is the time system the epoch is expressed in
    (:func:`epoch_in_era`): Kepler light curves are BKJD, so ``koi_time0bk`` is
    used as it stands; TESS light curves are BTJD, so it is shifted by -2167 d.
    Everything after that --- the dedupe, the fold, the masked baseline, the
    exposure-shrunk core, the bootstrap --- is the same code for both, which is
    the whole point of measuring both eras here.

    Returns ``(fit_result, ephemeris_record, unmeasured_reason_or_None)``.
    """
    t0_era = epoch_in_era(t0_bkjd, era)
    t_all = np.concatenate([np.asarray(s["time"], dtype=float) for s in segments])
    prop = propagate_epoch(t0_era, period_days, float(np.nanmedian(t_all)),
                           t0_err_days=sym_err(_f((koi_row or {}).get("koi_time0bk_err1")),
                                               _f((koi_row or {}).get("koi_time0bk_err2"))),
                           period_err_days=sym_err(_f((koi_row or {}).get("koi_period_err1")),
                                                   _f((koi_row or {}).get("koi_period_err2"))))
    dur_d = float(duration_hours) / 24.0
    m = measure_target(segments, period_days=period_days, t0_btjd=prop["t0_btjd"],
                       duration_days=dur_d, params=fit)
    eph = {"t0_epoch_used": prop["t0_btjd"], "t0_era_epoch": t0_era,
           "n_epochs_propagated": prop["n_epochs"],
           "ephemeris_sigma_minutes": prop["sigma_minutes"]}
    reason = None if (m["n_transits"] and np.isfinite(m["depth_ppm"])) else UNMEASURED_NO_TRANSIT
    return m, eph, reason


#: The per-member columns written to ``reductions.csv``.
ENSEMBLE_MEMBER_COLUMNS: tuple[str, ...] = (
    "kepoi_name", "tic_id", "era", "author", "flux_column", "status", "lc_status", "lc_route",
    "n_segments", "segment_list", "authors_returned", "exptimes_s", "n_transits",
    "depth_ppm", "depth_err_ppm", "depth_err_method", "oot_scatter_ppm", "smeared",
    "n_sectors_dropped_duplicate", "is_primary_reduction",
)


def measure_reduction_ensemble(target_id, *, era: str, period_days: float, t0_bkjd: float,
                               duration_hours: float, koi_row: dict | None = None,
                               lc_fn=None, mast: MastParams | None = None,
                               fit: FitParams | None = None,
                               ensemble: EnsembleParams | None = None,
                               primary_depth_ppm: float = float("nan"),
                               primary_author: str = "", primary_flux_column: str = "",
                               log: AcquisitionLog | None = None,
                               deadline: Deadline | None = None,
                               enabled: bool | None = None, key: str = ""
                               ) -> tuple[list[dict], dict, pd.DataFrame]:
    """Measure ONE era's depth under EVERY available reduction of the same data.

    This is the systematic estimate, and it is worth being explicit about what
    it is not: **it does not touch the primary depth.**  Each member is fetched,
    deduplicated and fitted on its own --- ``SPOC``'s ``PDCSAP_FLUX`` is one
    measurement, ``TESS-SPOC``'s ``PDCSAP_FLUX`` of the same pixels is another
    --- and what leaves this function is the *spread* over those numbers.  The
    depth the stage reports is still the deduplicated one from
    :func:`measure_one`; :func:`dedupe_sectors` still guarantees that one sector
    enters the depth once.  Stacking the members instead would count every
    transit as many times as the archive serves it, which is exactly the error
    run 35041932130 made.

    Returns ``(member_records, summary, member_sectors)``.  ``member_sectors``
    is each member's PER-SEGMENT breakdown, which goes into ``sectors.csv``
    beside the primary rows with ``scope = "ensemble"``: it is the same table
    that made the problem visible in the first place (``sectors.csv`` at commit
    ``050402a``, where sector 41 read 29,810 ppm under SPOC and 33,078 ppm
    under TESS-SPOC), and it is what distinguishes a uniform pipeline offset
    from sector-specific noise.

    Every grid point produces a record, including the ones that produced no
    depth, and each keeps its own status:

    ``QUERY_FAILED``                the archive errored for that member
    ``QUERY_RETURNED_ZERO_ROWS``    it answered with nothing
    ``AUTHOR_NOT_SERVED``           it answered with ANOTHER pipeline's products
    ``FLUX_COLUMN_NOT_PRESENT``     the product does not carry that column
    ``NO_USABLE_TRANSIT``           it was fetched but no transit could be fitted
    ``BUDGET_EXHAUSTED``            the ensemble's wall clock ran out first

    all of which are different facts and none of which is allowed to vanish: an
    ensemble that quietly lost four of its six members would report a spread
    that is too small, which is the same overconfidence one level down.
    """
    mast = mast or MastParams()
    fit = fit or FitParams()
    ensemble = ensemble or EnsembleParams()
    log = log or AcquisitionLog()
    e = str(era).lower()
    run = bool(ensemble.enabled if enabled is None else enabled)
    if not run:
        # Nothing was ASKED of the archive.  Not a failure, and the spread comes
        # back NaN with ENSEMBLE_NOT_ATTEMPTED rather than 0.
        return [], {**reduction_spread([], params=ensemble),
                    **sap_vs_pdcsap([], params=ensemble)}, pd.DataFrame()
    members: list[dict] = []
    sec_frames: list[pd.DataFrame] = []
    for g in ensemble_member_grid(e, ensemble):
        author, col = str(g["author"]), str(g["flux_column"])
        rec = {"era": e, "author": author, "flux_column": col, "status": "",
               "lc_status": "", "lc_route": "", "n_segments": 0, "segment_list": "",
               "authors_returned": "", "exptimes_s": "", "n_transits": 0,
               "depth_ppm": float("nan"), "depth_err_ppm": float("nan"),
               "depth_err_method": "", "oot_scatter_ppm": float("nan"), "smeared": False,
               "n_sectors_dropped_duplicate": 0,
               "is_primary_reduction": bool(str(primary_author) == author
                                            and str(primary_flux_column).upper() == col.upper())}
        if deadline is not None and deadline.expired():
            rec["status"] = UNMEASURED_BUDGET
            members.append(rec)
            continue
        mp = ensemble.member_view(mast, author=author)
        mkey = f"ens_{e}_{author}_{col}_{key or target_id}"
        if e == ERA_KEPLER:
            segs, status, route, _prov = fetch_kepler_lightcurves(
                target_id, lc_fn=lc_fn, params=mp, log=log, deadline=deadline, key=mkey,
                flux_columns=(col,))
        else:
            segs, status, route = fetch_lightcurves(
                target_id, lc_fn=lc_fn, params=mp, log=log, deadline=deadline, key=mkey,
                flux_columns=(col,))
        rec["lc_status"], rec["lc_route"] = status, route
        if status != STATUS_OK:
            rec["status"] = (UNMEASURED_BUDGET if deadline is not None and deadline.expired()
                             and status == STATUS_FAILED else status)
            members.append(rec)
            continue
        # The products must REALLY be the reduction that was asked for, on BOTH
        # axes.  A member that fell back to another author or another column is
        # a duplicate of a member already in the ensemble: it double-weights one
        # reduction in the percentile spread and adds a phantom SAP/PDCSAP pair,
        # shrinking the very error this ensemble exists to size.  Which one it
        # was is recorded; neither is a failure, and neither is the other.
        rec["authors_returned"] = ",".join(sorted({str(s.get("author")) for s in segs
                                                   if s.get("author")}))
        by_author = [s for s in segs if _same_author(s.get("author"), author)]
        if not by_author:
            rec["status"] = ENSEMBLE_NO_AUTHOR
            members.append(rec)
            continue
        kept = [s for s in by_author if str(s.get("flux_column") or "").upper() == col.upper()]
        if not kept:
            rec["status"] = ENSEMBLE_NO_FLUX_COLUMN
            members.append(rec)
            continue
        m, _eph, reason = _measure_era(kept, era=e, period_days=period_days, t0_bkjd=t0_bkjd,
                                       duration_hours=duration_hours, koi_row=koi_row, fit=fit)
        rec.update({"n_segments": int(m["n_sectors"]), "segment_list": m["sector_list"],
                    "exptimes_s": m["exptimes_s"], "n_transits": int(m["n_transits"]),
                    "depth_ppm": float(m["depth_ppm"]),
                    "depth_err_ppm": float(m["depth_err_ppm"]),
                    "depth_err_method": str(m["depth_err_method"]),
                    "oot_scatter_ppm": float(m["oot_scatter_ppm"]),
                    "smeared": bool(m["smeared"]),
                    "n_sectors_dropped_duplicate": int(m["n_sectors_dropped_duplicate"]),
                    "status": reason or STATUS_OK})
        members.append(rec)
        ms = m["sectors"].copy()
        if len(ms):
            ms.insert(0, "era", e)
            ms.insert(1, "scope", "ensemble")
            ms.insert(2, "reduction_author", author)
            ms.insert(3, "reduction_flux_column", col)
            sec_frames.append(ms)
    summary = {**reduction_spread(members, primary_depth_ppm=primary_depth_ppm,
                                  params=ensemble),
               **sap_vs_pdcsap(members, params=ensemble)}
    secs = pd.concat(sec_frames, ignore_index=True) if sec_frames else pd.DataFrame()
    return members, summary, secs


#: The reduction-ensemble fields carried into the per-target record for an era.
#: ``sap_vs_pdcsap_pairs`` is a list and lives in ``summary.json`` instead of a
#: CSV cell.
_ENSEMBLE_RECORD_DROP: frozenset = frozenset({"sap_vs_pdcsap_pairs"})


def _ensemble_fields(summary: dict, prefix: str = "") -> dict:
    """An ensemble summary flattened into ``prefix``-named record columns."""
    return {f"{prefix}{k}": v for k, v in (summary or {}).items()
            if k not in _ENSEMBLE_RECORD_DROP}


def _primary_reduction(m: dict) -> tuple[str, str]:
    """``(author, flux_column)`` of the reduction the PRIMARY depth came from.

    Recorded so the ensemble table can mark which of its members is the one the
    stage actually reports (``is_primary_reduction``) --- which is how a reader
    checks that the ensemble set the error and did not move the depth.
    """
    sec = m.get("sectors")
    author = str(m.get("authors") or "").split(",")[0]
    col = ""
    if sec is not None and len(sec) and "flux_column" in sec:
        col = str(sec["flux_column"].iloc[0] or "")
    return author, col


def measure_one(entry: dict, koi_row: dict | None, toi_row: dict | None, *,
                lc_fn=None, kepler_lc_fn=None, ensemble_lc_fn=None,
                kepler_ensemble_lc_fn=None, mast: MastParams, fit: FitParams,
                compare: CompareParams, ensemble: EnsembleParams | None = None,
                ld_table: dict | None = None,
                log: AcquisitionLog | None = None, deadline: Deadline | None = None,
                kepler_deadline: Deadline | None = None,
                ensemble_deadline: Deadline | None = None,
                kepler_ensemble_deadline: Deadline | None = None
                ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Everything stage 2 has to say about one target, in BOTH eras.

    Returns ``(measurement, sectors_df, fold_df, reductions_df)``; the sector,
    fold and reduction frames carry an ``era`` column and hold the Kepler and
    TESS rows together.  An era whose light curves could not be fetched comes
    back with its own ``QUERY_FAILED`` / ``QUERY_RETURNED_ZERO_ROWS`` and is
    **never** reported as agreeing with the other one.

    The depth each era reports is the **deduplicated** one --- one sector, one
    reduction, counted once.  The reduction ensemble
    (:func:`measure_reduction_ensemble`) is run separately on the same era and
    sets that depth's **error**, never its value: its spread is added in
    quadrature to the bootstrap and the like-for-like verdict is formed from the
    total.
    """
    log = log or AcquisitionLog()
    ensemble = ensemble or EnsembleParams()
    key = str(entry.get("kepoi_name") or entry.get("toi") or entry.get("tic_id"))
    rec: dict = {"kepoi_name": entry.get("kepoi_name"), "kepler_name": entry.get("kepler_name"),
                 "kepid": entry.get("kepid"), "tic_id": entry.get("tic_id"),
                 "toi": entry.get("toi"), "shortlist_source": entry.get("shortlist_source"),
                 "lc_status": "", "lc_route": "", "n_sectors": 0, "n_sectors_measured": 0,
                 "sector_list": "", "authors": "", "exptimes_s": "",
                 "period_days": float("nan"), "t0_bkjd": float("nan"),
                 "t0_btjd_kepler_epoch": float("nan"), "t0_btjd_used": float("nan"),
                 "n_epochs_propagated": 0, "ephemeris_sigma_minutes": float("nan"),
                 "duration_hours": float("nan"), "flags": "",
                 # --- the Kepler era, measured with the SAME fitter ---------
                 "kepler_lc_status": "", "kepler_lc_route": "", "kepler_n_quarters": 0,
                 "kepler_n_quarters_measured": 0, "kepler_quarter_list": "",
                 "kepler_cadences": "", "kepler_has_short_cadence": False,
                 "kepler_n_files_merged": 0, "kepler_t0_bkjd_used": float("nan"),
                 "kepler_n_epochs_propagated": 0,
                 "kepler_ephemeris_sigma_minutes": float("nan"),
                 "kepler_depth_measured_err_method": "", "kepler_flags": "",
                 "tess_era_unmeasured_reason": "", "kepler_era_unmeasured_reason": ""}
    # Every ensemble column exists on every row, measured or not, so a run in
    # which the ensemble never fired still says so in the CSV rather than
    # dropping the columns and leaving a reader to guess.
    rec.update(_ensemble_fields({**reduction_spread([], params=ensemble),
                                 **sap_vs_pdcsap([], params=ensemble)}))
    rec.update(_ensemble_fields({**reduction_spread([], params=ensemble),
                                 **sap_vs_pdcsap([], params=ensemble)}, "kepler_"))
    empty_sec = pd.DataFrame()
    empty_fold = pd.DataFrame(columns=["era", "dt_hours", "flux", "flux_err", "n"])
    empty_red = pd.DataFrame(columns=list(ENSEMBLE_MEMBER_COLUMNS))
    red_rows: list[dict] = []
    ens_t: dict = {}
    ens_k: dict = {}

    depth_k = _f((koi_row or {}).get("koi_depth"))
    err_k = sym_err(_f((koi_row or {}).get("koi_depth_err1")),
                    _f((koi_row or {}).get("koi_depth_err2")))
    depth_t = _f((toi_row or {}).get("pl_trandep"))
    err_t = sym_err(_f((toi_row or {}).get("pl_trandeperr1")),
                    _f((toi_row or {}).get("pl_trandeperr2")))
    b = _f((koi_row or {}).get("koi_impact"))
    fr, ld_meta = band_ratio(_f((koi_row or {}).get("koi_steff")),
                             _f((koi_row or {}).get("koi_slogg")),
                             b if np.isfinite(b) else 0.0, ld_table)
    rec["ld_note"] = ld_meta.get("ld_note", "")

    period = _f((koi_row or {}).get("koi_period"))
    t0_bkjd = _f((koi_row or {}).get("koi_time0bk"))
    dur_h = _f((koi_row or {}).get("koi_duration"))
    rec.update({"period_days": period, "t0_bkjd": t0_bkjd, "duration_hours": dur_h,
                "b_kepler": b})
    # The catalogue separation the like-for-like comparison has to be able to
    # see: ln(TOI / KOI-in-TESS-band).  This is the change stage 1 claimed, and
    # a like-for-like agreement only refutes it if it could have detected it.
    fr_applied = float(fr) if (compare.apply_band_ratio and np.isfinite(fr) and fr > 0) else 1.0
    ln_under_test = float("nan")
    if np.isfinite(depth_k) and depth_k > 0 and np.isfinite(depth_t) and depth_t > 0:
        ln_under_test = math.log(depth_t / (depth_k * fr_applied))
    rec["catalogue_ln_ratio_under_test"] = ln_under_test

    def _unmeasured(reason: str, *, kepler_reason: str) -> tuple[dict, pd.DataFrame,
                                                                 pd.DataFrame, pd.DataFrame]:
        rec.update(compare_three_depths(float("nan"), float("nan"), depth_k, err_k, depth_t,
                                        err_t, ld_band_ratio=fr, params=compare,
                                        unmeasured_reason=reason))
        rec.update(compare_measured_eras(float("nan"), float("nan"), float("nan"), float("nan"),
                                         ld_band_ratio=fr, ln_ratio_under_test=ln_under_test,
                                         params=compare, kepler_unmeasured_reason=kepler_reason,
                                         tess_unmeasured_reason=reason))
        rec.update(compare_kepler_to_catalogue(float("nan"), float("nan"), depth_k, err_k,
                                               params=compare, unchecked_reason=kepler_reason))
        rec["tess_era_unmeasured_reason"] = reason
        rec["kepler_era_unmeasured_reason"] = kepler_reason
        return rec, empty_sec, empty_fold, empty_red

    if not (np.isfinite(period) and period > 0 and np.isfinite(t0_bkjd)
            and np.isfinite(dur_h) and dur_h > 0):
        rec["lc_status"] = rec["kepler_lc_status"] = "not_attempted"
        return _unmeasured(UNMEASURED_NO_EPHEMERIS, kepler_reason=UNMEASURED_NO_EPHEMERIS)

    rec["t0_btjd_kepler_epoch"] = float(epoch_in_era(t0_bkjd, ERA_TESS))
    sec_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    flags = []

    # ---------------- the TESS era ----------------------------------------
    tess_reason: str | None = None
    m_t: dict | None = None
    secs, status, route = fetch_lightcurves(entry.get("tic_id"), lc_fn=lc_fn, params=mast,
                                            log=log, deadline=deadline, key=key)
    rec["lc_status"], rec["lc_route"] = status, route
    if status != STATUS_OK:
        tess_reason = (UNMEASURED_BUDGET if deadline is not None and deadline.expired()
                       and status == STATUS_FAILED else status)
    else:
        m_t, eph_t, tess_reason = _measure_era(secs, era=ERA_TESS, period_days=period,
                                               t0_bkjd=t0_bkjd, duration_hours=dur_h,
                                               koi_row=koi_row, fit=fit)
        rec.update({"t0_btjd_used": eph_t["t0_epoch_used"],
                    "n_epochs_propagated": eph_t["n_epochs_propagated"],
                    "ephemeris_sigma_minutes": eph_t["ephemeris_sigma_minutes"]})
        flags = [f for f in str(m_t["flags"]).split(";") if f]
        if (np.isfinite(eph_t["ephemeris_sigma_minutes"])
                and eph_t["ephemeris_sigma_minutes"] > 0.1 * dur_h * 60.0):
            flags.append("ephemeris_drift_over_10pct_of_duration")
        # The fitted quantities keep their own names; `compare_three_depths`
        # then adds depth_measured_ppm / depth_measured_err_ppm as the pair.
        rec["depth_measured_err_method"] = m_t["depth_err_method"]
        for k in _ERA_FIT_FIELDS:
            rec[k] = m_t[k]
        sec_t = m_t["sectors"].copy()
        if len(sec_t):
            sec_t.insert(0, "era", ERA_TESS)
            # `scope` separates the rows the DEPTH came from (the deduplicated
            # primary reduction) from the ensemble's rows, which exist only to
            # size the error.  Without it a reader could mistake the ensemble's
            # duplicated sectors for extra data.
            sec_t.insert(1, "scope", "primary")
            sec_frames.append(sec_t)
        if len(m_t["fold"]):
            ft = m_t["fold"].copy()
            ft.insert(0, "era", ERA_TESS)
            fold_frames.append(ft)
        # --- the TESS reduction ensemble.  The primary depth above is already
        # fixed; what follows only sets its error.
        if tess_reason is None:
            p_author, p_col = _primary_reduction(m_t)
            t_members, ens_t, t_secs = measure_reduction_ensemble(
                entry.get("tic_id"), era=ERA_TESS, period_days=period, t0_bkjd=t0_bkjd,
                duration_hours=dur_h, koi_row=koi_row, lc_fn=ensemble_lc_fn, mast=mast,
                fit=fit, ensemble=ensemble, primary_depth_ppm=m_t["depth_ppm"],
                primary_author=p_author, primary_flux_column=p_col,
                log=log, deadline=ensemble_deadline, key=key,
                enabled=(True if ensemble_lc_fn is not None else None))
            red_rows.extend(t_members)
            if len(t_secs):
                sec_frames.append(t_secs)
            rec.update(_ensemble_fields(ens_t))

    # ---------------- the KEPLER era, SAME fitter --------------------------
    kepler_reason: str | None = None
    m_k: dict | None = None
    if kepler_lc_fn is None and not bool(mast.kepler_enabled):
        # Nothing was asked of the archive: a different fact from an archive
        # that did not answer, and it is named as such.
        kepler_reason = UNMEASURED_KEPLER_DISABLED
        rec["kepler_lc_status"] = "not_attempted"
    else:
        kepid = entry.get("kepid")
        if kepid is None or not np.isfinite(_f(kepid)):
            kepler_reason = UNMEASURED_ZERO_ROWS
            rec["kepler_lc_status"] = STATUS_ZERO
        else:
            ksecs, kstatus, kroute, kprov = fetch_kepler_lightcurves(
                int(_f(kepid)), lc_fn=kepler_lc_fn, params=mast, log=log,
                deadline=kepler_deadline, key=key)
            rec["kepler_lc_status"], rec["kepler_lc_route"] = kstatus, kroute
            rec["kepler_n_files_merged"] = int(sum(int(p.get("n_files_merged") or 0)
                                                   for p in kprov))
            if kstatus != STATUS_OK:
                kepler_reason = (UNMEASURED_BUDGET if kepler_deadline is not None
                                 and kepler_deadline.expired() and kstatus == STATUS_FAILED
                                 else kstatus)
            else:
                cad = sorted({str(s.get("cadence") or kepler_cadence_label(s.get("exptime_s")))
                              for s in ksecs})
                rec["kepler_cadences"] = ",".join(cad)
                rec["kepler_has_short_cadence"] = bool("short" in cad)
                m_k, eph_k, kepler_reason = _measure_era(ksecs, era=ERA_KEPLER,
                                                         period_days=period, t0_bkjd=t0_bkjd,
                                                         duration_hours=dur_h, koi_row=koi_row,
                                                         fit=fit)
                rec.update({"kepler_t0_bkjd_used": eph_k["t0_epoch_used"],
                            "kepler_n_epochs_propagated": eph_k["n_epochs_propagated"],
                            "kepler_ephemeris_sigma_minutes": eph_k["ephemeris_sigma_minutes"]})
                rec["kepler_depth_measured_err_method"] = m_k["depth_err_method"]
                rec["kepler_n_quarters"] = m_k["n_sectors"]
                rec["kepler_n_quarters_measured"] = m_k["n_sectors_measured"]
                rec["kepler_quarter_list"] = m_k["sector_list"]
                for k in _ERA_FIT_FIELDS:
                    rec[f"kepler_{k}"] = m_k[k]
                kflags = [f for f in str(m_k["flags"]).split(";") if f]
                if (np.isfinite(eph_k["ephemeris_sigma_minutes"])
                        and eph_k["ephemeris_sigma_minutes"] > 0.1 * dur_h * 60.0):
                    kflags.append("ephemeris_drift_over_10pct_of_duration")
                rec["kepler_flags"] = ";".join(kflags)
                sec_k = m_k["sectors"].copy()
                if len(sec_k):
                    sec_k.insert(0, "era", ERA_KEPLER)
                    sec_k.insert(1, "scope", "primary")
                    sec_frames.append(sec_k)
                if len(m_k["fold"]):
                    fk = m_k["fold"].copy()
                    fk.insert(0, "era", ERA_KEPLER)
                    fold_frames.append(fk)
                # --- the KEPLER reduction ensemble, same code, same rules.
                if kepler_reason is None:
                    kp_author, kp_col = _primary_reduction(m_k)
                    k_members, ens_k, k_secs = measure_reduction_ensemble(
                        int(_f(kepid)), era=ERA_KEPLER, period_days=period, t0_bkjd=t0_bkjd,
                        duration_hours=dur_h, koi_row=koi_row, lc_fn=kepler_ensemble_lc_fn,
                        mast=mast, fit=fit, ensemble=ensemble,
                        primary_depth_ppm=m_k["depth_ppm"],
                        primary_author=kp_author, primary_flux_column=kp_col,
                        log=log, deadline=kepler_ensemble_deadline, key=key,
                        enabled=(True if kepler_ensemble_lc_fn is not None else None))
                    red_rows.extend(k_members)
                    if len(k_secs):
                        sec_frames.append(k_secs)
                    rec.update(_ensemble_fields(ens_k, "kepler_"))

    rec["tess_era_unmeasured_reason"] = tess_reason or ""
    rec["kepler_era_unmeasured_reason"] = kepler_reason or ""

    # ---------------- the comparisons --------------------------------------
    d_t = m_t["depth_ppm"] if m_t is not None else float("nan")
    e_t = m_t["depth_err_ppm"] if m_t is not None else float("nan")
    d_k = m_k["depth_ppm"] if m_k is not None else float("nan")
    e_k = m_k["depth_err_ppm"] if m_k is not None else float("nan")
    # The reduction systematic each era carries into its error.  NaN when the
    # ensemble was not attempted or came back with too few members, in which
    # case the total error is the statistical one and the record says why.
    spread_t = _f(ens_t.get("depth_reduction_spread_ppm"))
    spread_k = _f(ens_k.get("depth_reduction_spread_ppm"))
    # Secondary, and now a diagnostic of the CATALOGUES: our TESS depth
    # against both catalogue numbers.
    rec.update(compare_three_depths(d_t, e_t, depth_k, err_k, depth_t, err_t,
                                    ld_band_ratio=fr, params=compare,
                                    unmeasured_reason=tess_reason,
                                    measured_reduction_spread_ppm=spread_t))
    # PRIMARY: our Kepler-era fit against our TESS-era fit, on TOTAL errors.
    rec.update(compare_measured_eras(d_k, e_k, d_t, e_t, ld_band_ratio=fr,
                                     ln_ratio_under_test=ln_under_test, params=compare,
                                     kepler_unmeasured_reason=kepler_reason,
                                     tess_unmeasured_reason=tess_reason,
                                     kepler_reduction_spread_ppm=spread_k,
                                     tess_reduction_spread_ppm=spread_t))
    # And what our Kepler-era fit says about the KOI table itself.
    rec.update(compare_kepler_to_catalogue(d_k, e_k, depth_k, err_k, params=compare,
                                           unchecked_reason=kepler_reason))
    # The ensemble's own flags, per era.  A spread that exceeds the bootstrap is
    # the headline one: it says the quoted error was NOT describing the real
    # scatter, which is the failure stage 1 made and this ensemble exists to
    # catch one level down.
    def _ensemble_flags(ens: dict, stat_err_ppm: float) -> list[str]:
        marks: list[str] = []
        sp = _f(ens.get("depth_reduction_spread_ppm"))
        if ens.get("ensemble_incomplete"):
            marks.append("ensemble_incomplete")
        if np.isfinite(sp) and np.isfinite(stat_err_ppm) and stat_err_ppm > 0 and sp > stat_err_ppm:
            marks.append("reduction_spread_exceeds_statistical_error")
        if str(ens.get("sap_vs_pdcsap_verdict")) == BG_DISAGREE:
            marks.append("sap_pdcsap_disagree")
        return marks

    flags.extend(_ensemble_flags(ens_t, e_t))
    rec["kepler_flags"] = ";".join(
        [f for f in str(rec.get("kepler_flags") or "").split(";") if f]
        + _ensemble_flags(ens_k, e_k))
    rec["flags"] = ";".join(flags)

    sec = (pd.concat(sec_frames, ignore_index=True) if sec_frames else empty_sec)
    if len(sec):
        sec.insert(0, "kepoi_name", rec["kepoi_name"])
        sec.insert(1, "tic_id", rec["tic_id"])
    fold_df = (pd.concat(fold_frames, ignore_index=True) if fold_frames else empty_fold)
    red = empty_red
    if red_rows:
        red = pd.DataFrame(red_rows)
        red.insert(0, "kepoi_name", rec["kepoi_name"])
        red.insert(1, "tic_id", rec["tic_id"])
        red = red[[c for c in ENSEMBLE_MEMBER_COLUMNS if c in red.columns]]
    return rec, sec, fold_df, red


def stage2_measure(conf: dict, out: Path, *, query_fn=None, lc_fn=None, kepler_lc_fn=None,
                   ensemble_lc_fn=None, kepler_ensemble_lc_fn=None,
                   shortlist: pd.DataFrame | None = None, log: AcquisitionLog | None = None
                   ) -> dict:
    """Fetch the ephemerides and BOTH eras' light curves, fit every shortlisted target.

    The two eras get **separate wall-clock budgets** (``stage2.mast.budget_s``
    and ``stage2.mast.kepler_budget_s``) so that a slow fetch on one side
    cannot silently starve the other and leave the like-for-like comparison
    one-sided.  The reduction ensemble gets **two more** of its own
    (``stage2.ensemble.budget_s`` / ``kepler_budget_s``): the grid multiplies
    the number of archive requests, and a systematic estimate must never be
    able to consume the budget of the measurement it is qualifying.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = log or AcquisitionLog(prefix="growth/stage2")
    mast = MastParams.from_config(conf)
    fit = FitParams.from_config(conf)
    cmp_p = CompareParams.from_config(conf)
    ens_p = EnsembleParams.from_config(conf)
    ld_table = conf.get("limb_darkening")
    if shortlist is None:
        shortlist = load_shortlist(conf, candidates_csv=_candidates_csv(out))
    arch = conf.get("archive") or {}
    url = str(arch.get("exoarchive_tap")
              or "https://exoplanetarchive.ipac.caltech.edu/TAP/sync")
    qf = query_fn or (lambda adql: tap_sync(adql, url=url, retries=int(mast.archive_retries),
                                            timeout=float(mast.archive_timeout_s)))
    koi, s_koi = fetch_koi_ephemerides(shortlist.get("kepoi_name", pd.Series(dtype=str)),
                                       query_fn=qf, log=log)
    toi, s_toi = fetch_toi_depths(shortlist.get("tic_id", pd.Series(dtype=float)),
                                  query_fn=qf, log=log)
    koi_by = {str(r.get("kepoi_name")): r for r in koi.to_dict(orient="records")} if len(
        koi) else {}
    toi_rows = toi.to_dict(orient="records") if len(toi) else []

    deadline = Deadline(budget_s=float(mast.budget_s) if mast.budget_s else None)
    kepler_deadline = Deadline(budget_s=float(mast.kepler_budget_s)
                               if mast.kepler_budget_s else None)
    ens_deadline = Deadline(budget_s=float(ens_p.budget_s) if ens_p.budget_s else None)
    ens_kepler_deadline = Deadline(budget_s=float(ens_p.kepler_budget_s)
                                   if ens_p.kepler_budget_s else None)
    recs, secs, folds, reds = [], [], {}, []
    for entry in shortlist.to_dict(orient="records"):
        k = str(entry.get("kepoi_name"))
        koi_row = koi_by.get(k)
        toi_row = _pick_toi_row(toi_rows, entry, koi_row)
        rec, sec, fold_df, red_df = measure_one(
            entry, koi_row, toi_row, lc_fn=lc_fn, kepler_lc_fn=kepler_lc_fn,
            ensemble_lc_fn=ensemble_lc_fn, kepler_ensemble_lc_fn=kepler_ensemble_lc_fn,
            mast=mast, fit=fit, compare=cmp_p, ensemble=ens_p, ld_table=ld_table, log=log,
            deadline=deadline, kepler_deadline=kepler_deadline,
            ensemble_deadline=ens_deadline, kepler_ensemble_deadline=ens_kepler_deadline)
        # The table's status and THIS target's status are different facts: a
        # query that succeeded but returned no row for this KOI is ZERO_ROWS
        # for the target even though the table is OK.
        rec["koi_ephemeris_status"] = (STATUS_OK if koi_row is not None else
                                       (s_koi if s_koi != STATUS_OK else STATUS_ZERO))
        rec["toi_row_status"] = STATUS_OK if toi_row is not None else STATUS_ZERO
        recs.append(rec)
        if len(sec):
            secs.append(sec)
        if len(fold_df):
            folds[k] = fold_df
        if len(red_df):
            reds.append(red_df)
        print(f"[growth-stage2] {k}: reduction ensemble — TESS "
              f"{rec.get('n_members_measured')}/{rec.get('n_members')} members, spread "
              f"{rec.get('depth_reduction_spread_ppm')} ppm "
              f"[{rec.get('depth_reduction_spread_status')}]; KEPLER "
              f"{rec.get('kepler_n_members_measured')}/{rec.get('kepler_n_members')} members, "
              f"spread {rec.get('kepler_depth_reduction_spread_ppm')} ppm "
              f"[{rec.get('kepler_depth_reduction_spread_status')}]; SAP-PDCSAP "
              f"{rec.get('sap_minus_pdcsap_ppm')} ppm (z={rec.get('sap_minus_pdcsap_z')}) "
              f"{rec.get('sap_vs_pdcsap_verdict')}")
        print(f"[growth-stage2] {k}: PRIMARY {rec.get('like_for_like_verdict')} "
              f"(z={rec.get('z_measured_eras')}) | our Kepler-era "
              f"{rec.get('depth_kepler_measured_ppm')} +/- "
              f"{rec.get('depth_kepler_measured_err_ppm')} ppm [{rec.get('kepler_lc_status')}] "
              f"vs our TESS-era {rec.get('depth_measured_ppm')} +/- "
              f"{rec.get('depth_measured_err_ppm')} ppm [{rec.get('lc_status')}]")
        print(f"[growth-stage2] {k}: catalogues — {rec.get('verdict')}; KOI table "
              f"{rec.get('koi_depth_verdict')} (koi_depth {rec.get('depth_kepler_ppm')}, "
              f"TOI {rec.get('depth_toi_ppm')} ppm)")
    meas = pd.DataFrame(recs)
    meas.to_csv(out / "measurements.csv", index=False)
    sec_df = pd.concat(secs, ignore_index=True) if secs else pd.DataFrame()
    sec_df.to_csv(out / "sectors.csv", index=False)
    # The ensemble's own table: one row per (era, author, flux column), measured
    # or not, with the reason when not.  This is the audit trail behind the
    # spread, and it is written even when it is empty so a reader can tell a
    # run without an ensemble from a run whose ensemble found nothing.
    red_df = (pd.concat(reds, ignore_index=True) if reds
              else pd.DataFrame(columns=list(ENSEMBLE_MEMBER_COLUMNS)))
    red_df.to_csv(out / "reductions.csv", index=False)
    fdir = out / "folds"
    fdir.mkdir(parents=True, exist_ok=True)
    for k, v in folds.items():
        v.to_csv(fdir / f"{str(k).replace('/', '_')}.csv", index=False)
    rep = {"stage": "measure", "generated_utc": _now(),
           "n_shortlist": int(len(shortlist)),
           "koi_ephemeris_status": s_koi, "toi_status": s_toi,
           "n_measured": int((meas.get("verdict", pd.Series(dtype=str)) != MATCH_UNMEASURED
                              ).sum()) if len(meas) else 0,
           "lc_status_counts": (meas["lc_status"].map(str).value_counts().to_dict()
                                if len(meas) else {}),
           "lc_routes": (meas["lc_route"].map(str).value_counts().to_dict()
                         if len(meas) else {}),
           "budget_s": mast.budget_s, "elapsed_s": round(deadline.elapsed(), 1),
           "budget_exhausted": bool(deadline.expired()),
           # The Kepler era is a SEPARATE network stage with a SEPARATE budget,
           # and its statuses are reported separately: a run in which only the
           # TESS side answered has no like-for-like comparison at all, and
           # that must be visible here rather than inferred from the verdicts.
           "kepler_era_enabled": bool(mast.kepler_enabled or kepler_lc_fn is not None),
           "kepler_lc_status_counts": (meas["kepler_lc_status"].map(str).value_counts().to_dict()
                                       if len(meas) and "kepler_lc_status" in meas else {}),
           "kepler_lc_routes": (meas["kepler_lc_route"].map(str).value_counts().to_dict()
                                if len(meas) and "kepler_lc_route" in meas else {}),
           "kepler_budget_s": mast.kepler_budget_s,
           "kepler_elapsed_s": round(kepler_deadline.elapsed(), 1),
           "kepler_budget_exhausted": bool(kepler_deadline.expired()),
           "n_compared_like_for_like": (
               int((meas.get("like_for_like_verdict", pd.Series(dtype=str))
                    != LL_ERA_UNMEASURED).sum()) if len(meas) else 0),
           # --- the REDUCTION ENSEMBLE, a third and fourth network stage with
           # their own budgets.  An ensemble that was never attempted, or that
           # lost members, must be visible here rather than inferred from a
           # suspiciously small error bar.
           "ensemble_enabled": bool(ens_p.enabled or ensemble_lc_fn is not None
                                    or kepler_ensemble_lc_fn is not None),
           "ensemble_budget_s": ens_p.budget_s,
           "ensemble_elapsed_s": round(ens_deadline.elapsed(), 1),
           "ensemble_budget_exhausted": bool(ens_deadline.expired()),
           "ensemble_kepler_budget_s": ens_p.kepler_budget_s,
           "ensemble_kepler_elapsed_s": round(ens_kepler_deadline.elapsed(), 1),
           "ensemble_kepler_budget_exhausted": bool(ens_kepler_deadline.expired()),
           "n_reduction_members": int(len(red_df)),
           "reduction_member_status_counts": (red_df["status"].map(str).value_counts().to_dict()
                                              if len(red_df) and "status" in red_df else {}),
           "acquisition": log.as_dict()}
    _write(out / "acquire.json", rep)
    log.write(out / "acquisition_log.json")
    print(f"[growth-stage2] measure: {rep['n_measured']}/{rep['n_shortlist']} TESS-era measured "
          f"in {rep['elapsed_s']}s (budget {mast.budget_s}s); "
          f"{rep['n_compared_like_for_like']}/{rep['n_shortlist']} compared like-for-like "
          f"in {rep['kepler_elapsed_s']}s of Kepler budget {mast.kepler_budget_s}s")
    return rep


def _pick_toi_row(toi_rows, entry: dict, koi_row: dict | None) -> dict | None:
    """The TOI row for this shortlist entry: by ``toi`` id, else by TIC + period."""
    want = _f(entry.get("toi"))
    if np.isfinite(want):
        for r in toi_rows:
            if np.isfinite(_f(r.get("toi"))) and abs(_f(r.get("toi")) - want) < 1e-6:
                return r
    try:
        tid = int(float(entry.get("tic_id")))
    except (TypeError, ValueError):
        tid = None
    same = [r for r in toi_rows
            if tid is not None and np.isfinite(_f(r.get("tid"))) and int(_f(r.get("tid"))) == tid]
    if not same:
        return None
    p = _f((koi_row or {}).get("koi_period"))
    if np.isfinite(p) and p > 0:
        best, bd = None, np.inf
        for r in same:
            pt = _f(r.get("pl_orbper"))
            if np.isfinite(pt) and pt > 0 and abs(pt - p) / p < bd:
                best, bd = r, abs(pt - p) / p
        if best is not None and bd < 1e-2:
            return best
    return same[0]


def reduction_ensemble_report(measurements: pd.DataFrame,
                              params: EnsembleParams | None = None) -> dict:
    """The ensemble block of ``summary.json``: the spread, and the background test.

    Two things are stated here in words as well as numbers, because both are
    claims a reader would otherwise have to reconstruct:

    * **which error the verdict used.**  The like-for-like ``z`` is formed from
      ``sqrt(bootstrap^2 + reduction_spread^2)``, and the bootstrap-only ``z``
      is shown beside it, so the effect of the systematic is visible.
    * **what SAP minus PDCSAP says.**  "The PDC background subtraction inflates
      the depth" is a specific mundane explanation for a deeper TESS transit,
      and it is answered with a number and a direction rather than left as a
      worry.  ``PDC_DEEPER_THAN_SAP`` beyond ``background_n_agree`` sigma is the
      case that supports the mundane explanation, and it is reported as a
      finding.
    """
    params = params or EnsembleParams()
    rows = measurements.to_dict(orient="records") if len(measurements) else []
    per_target = []
    disagreeing = []
    for r in rows:
        per_target.append({
            "kepoi_name": r.get("kepoi_name"),
            "tess": {k: r.get(k) for k in
                     ("n_members", "n_members_measured", "n_members_unavailable",
                      "depth_reduction_spread_ppm", "depth_reduction_spread_fraction",
                      "depth_reduction_spread_status", "depth_reduction_min_ppm",
                      "depth_reduction_max_ppm", "ensemble_incomplete",
                      "ensemble_unavailable_reasons", "ensemble_members",
                      "depth_tess_measured_err_ppm", "depth_tess_measured_total_err_ppm")},
            "kepler": {k[len("kepler_"):] if k.startswith("kepler_") else k: r.get(k) for k in
                       ("kepler_n_members", "kepler_n_members_measured",
                        "kepler_n_members_unavailable", "kepler_depth_reduction_spread_ppm",
                        "kepler_depth_reduction_spread_fraction",
                        "kepler_depth_reduction_spread_status", "kepler_ensemble_incomplete",
                        "kepler_ensemble_unavailable_reasons", "kepler_ensemble_members",
                        "depth_kepler_measured_err_ppm",
                        "depth_kepler_measured_total_err_ppm")},
            "background_test": {k: r.get(k) for k in
                                ("sap_vs_pdcsap_verdict", "sap_vs_pdcsap_n_pairs",
                                 "sap_vs_pdcsap_authors", "sap_minus_pdcsap_ppm",
                                 "sap_minus_pdcsap_err_ppm", "sap_minus_pdcsap_z",
                                 "sap_minus_pdcsap_fraction", "background_direction")},
            "verdict_errors": {k: r.get(k) for k in
                               ("z_measured_eras", "z_measured_eras_stat_only",
                                "sigma_measured_eras", "sigma_measured_eras_stat_only",
                                "detectable_ln_ratio", "ln_ratio_under_test",
                                "reduction_systematic_applied", "like_for_like_verdict")},
        })
        if str(r.get("sap_vs_pdcsap_verdict")) == BG_DISAGREE:
            disagreeing.append({"kepoi_name": r.get("kepoi_name"),
                                "sap_minus_pdcsap_ppm": r.get("sap_minus_pdcsap_ppm"),
                                "z": r.get("sap_minus_pdcsap_z"),
                                "direction": r.get("background_direction")})
    # A row from an older measurements.csv has no ensemble columns at all; that
    # is "not attempted", not "attempted and silent".
    def _attempted(r: dict, key: str) -> bool:
        v = r.get(key)
        return bool(v) and str(v) not in (SPREAD_NOT_ATTEMPTED, "nan", "None")

    n_attempted = sum(1 for r in rows
                      if _attempted(r, "depth_reduction_spread_status")
                      or _attempted(r, "kepler_depth_reduction_spread_status"))
    finding = ""
    if disagreeing:
        pdc = [d for d in disagreeing if str(d.get("direction")) == BG_PDC_DEEPER]
        finding = (
            f"{len(disagreeing)} target(s): SAP_FLUX and PDCSAP_FLUX disagree at or beyond "
            f"{params.background_n_agree} sigma. "
            + (f"{len(pdc)} of them are {BG_PDC_DEEPER} — the corrected photometry is DEEPER "
               "than the raw aperture photometry, which is what 'the PDC background/crowding "
               "correction inflates the depth' predicts, and it is a mundane explanation for "
               "the whole result that must be settled before anything else is claimed."
               if pdc else
               "All of them are SAP_DEEPER_THAN_PDC, which is the OPPOSITE of the "
               "background-inflation hypothesis: the correction is making the transit "
               "shallower, not deeper."))
    return {
        "what_it_is": ("the depth measured under EVERY available reduction of the same data — "
                       "each (pipeline author, flux column) combination the archive serves — "
                       "with the spread over those members entering the total error as a "
                       "systematic"),
        "what_it_is_not": ("it does NOT move the depth. The reported depth is still the "
                           "DEDUPLICATED one (dedupe_sectors: one sector, one reduction, "
                           "counted once); re-stacking the ensemble members would count every "
                           "transit as many times as the archive serves it, which is the error "
                           "run 35041932130 made. The ensemble sets the ERROR, and only the "
                           "error."),
        "spread_definition": (f"half the range between the {100.0 - params.spread_percentile:.0f}"
                              f"th and {params.spread_percentile:.0f}th percentiles of the "
                              "measured members' depths; NaN with status TOO_FEW_MEMBERS below "
                              f"{params.min_members_for_spread} members, never 0"),
        "how_it_enters": ("total_err = sqrt(statistical^2 + reduction_spread^2), per era, and "
                          "the like-for-like verdict is formed from the TOTAL. A spread large "
                          "enough to swamp the ratio also raises detectable_ln_ratio, so the "
                          "verdict becomes MEASURED_DEPTH_UNRESOLVED — never "
                          "MEASURED_DEPTH_UNCHANGED."),
        "why": ("run 35041932130 measured the same seven TESS sectors under two pipelines: SPOC "
                "2-minute gave 25,425-35,498 ppm and TESS-SPOC FFI gave 33,078-39,798 ppm for "
                "the SAME PIXELS — a 20-30 % disagreement against a quoted error of 1,521 ppm. "
                "The out-of-transit scatter is 414 ppm in Kepler and 67,631 ppm in TESS. A "
                "quoted error that does not describe the real scatter is exactly what broke "
                "stage 1."),
        "n_targets_with_an_ensemble": int(n_attempted),
        "background_test_is": ("SAP_FLUX minus PDCSAP_FLUX, paired WITHIN a pipeline author and "
                               "combined by inverse variance. PDC is where the crowding and "
                               "background corrections are applied, so this is the direct test "
                               "of 'an over-subtracted background inflates the depth'."),
        "background_test_finding": finding,
        "targets_where_sap_and_pdcsap_disagree": disagreeing,
        "targets": per_target,
        "config": {k: getattr(params, k) for k in
                   ("enabled", "spread_percentile", "min_members_for_spread",
                    "background_n_agree", "budget_s", "kepler_budget_s",
                    "per_member_budget_s", "retries")}
        | {"tess_authors": list(params.tess_authors),
           "tess_flux_columns": list(params.tess_flux_columns),
           "kepler_authors": list(params.kepler_authors),
           "kepler_flux_columns": list(params.kepler_flux_columns)},
    }


def stage2_assess(conf: dict, out: Path, *, measurements: pd.DataFrame | None = None,
                  acquire_report: dict | None = None) -> dict:
    """The three-way verdict per target, the run verdict, ``summary.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if measurements is None:
        p = out / "measurements.csv"
        measurements = (pd.read_csv(p, low_memory=False) if p.exists() and p.stat().st_size
                        else pd.DataFrame())
    if acquire_report is None and (out / "acquire.json").exists():
        try:
            acquire_report = json.loads((out / "acquire.json").read_text())
        except Exception:                                 # noqa: BLE001
            acquire_report = None
    verdict, reason = run_verdict(measurements)
    ll_verdict, ll_reason = run_like_for_like_verdict(measurements)
    v = (measurements["verdict"].fillna(MATCH_UNMEASURED).astype(str)
         if len(measurements) and "verdict" in measurements else pd.Series(dtype=str))
    counts = {k: int((v == k).sum()) for k in TARGET_VERDICTS}
    llv = (measurements["like_for_like_verdict"].fillna(LL_ERA_UNMEASURED).astype(str)
           if len(measurements) and "like_for_like_verdict" in measurements
           else pd.Series(dtype=str))
    ll_counts = {k: int((llv == k).sum()) for k in LIKE_FOR_LIKE_VERDICTS}
    koiv = (measurements["koi_depth_verdict"].fillna(KOI_DEPTH_UNCHECKED).astype(str)
            if len(measurements) and "koi_depth_verdict" in measurements
            else pd.Series(dtype=str))
    koi_counts = {k: int((koiv == k).sum()) for k in KOI_DEPTH_VERDICTS}
    unmeasured = (measurements.loc[v == MATCH_UNMEASURED, "unmeasured_reason"]
                  .fillna("").astype(str).value_counts().to_dict() if len(measurements)
                  and "unmeasured_reason" in measurements else {})
    ll_unmeasured = (measurements.loc[llv == LL_ERA_UNMEASURED,
                                      "like_for_like_unmeasured_reason"]
                     .fillna("").astype(str).value_counts().to_dict()
                     if len(measurements) and "like_for_like_unmeasured_reason" in measurements
                     else {})
    flags = {}
    if len(measurements):
        # pandas 3 keeps NaN as a float through .astype(str), so every value
        # is coerced here rather than assumed to be a string.  Both eras'
        # flags are counted; the Kepler ones are prefixed so a smeared Kepler
        # long cadence is never read as a smeared TESS sector.
        for col, pre in (("flags", ""), ("kepler_flags", "kepler:")):
            if col not in measurements:
                continue
            for row in measurements[col]:
                for fl in str(row if row == row else "").split(";"):
                    if fl:
                        flags[pre + fl] = flags.get(pre + fl, 0) + 1
    cmp_p = CompareParams.from_config(conf)
    fit = FitParams.from_config(conf)
    mast = MastParams.from_config(conf)
    ens_p = EnsembleParams.from_config(conf)
    n_contra = koi_counts[KOI_DEPTH_CONTRADICTED]
    koi_check = {
        "verdict_counts": koi_counts,
        "what_it_compares": ("OUR fit to the KEPLER light curve against cumulative.koi_depth, "
                             "BOTH in the Kepler band, so NO limb-darkening band ratio is "
                             "applied and nothing but the two numbers is under test"),
        "targets_contradicting_the_koi_table": [
            {"kepoi_name": r.get("kepoi_name"),
             "koi_depth_ppm": r.get("depth_kepler_ppm"),
             "our_kepler_depth_ppm": r.get("depth_kepler_measured_ppm"),
             "our_kepler_depth_err_ppm": r.get("depth_kepler_measured_err_ppm"),
             "ratio": r.get("kepler_measured_over_koi_depth"),
             "z": r.get("z_kepler_measured_vs_koi")}
            for r in (measurements.to_dict(orient="records") if len(measurements) else [])
            if str(r.get("koi_depth_verdict")) == KOI_DEPTH_CONTRADICTED],
        "finding": ("" if not n_contra else
                    f"{n_contra} target(s): our fit to the Kepler light curve DISAGREES with "
                    f"cumulative.koi_depth at or beyond {cmp_p.koi_check_n_agree} sigma. That "
                    f"is a finding about the KOI TABLE — the stage-1 input — and is reported "
                    f"as one, not absorbed into the depth story."),
    }
    ensemble_block = reduction_ensemble_report(measurements, params=ens_p)
    summary = {
        # --- the PRIMARY verdict ------------------------------------------
        "primary_verdict": ll_verdict,
        "primary_reason": ll_reason,
        "primary_verdict_field": "like_for_like_verdict",
        "primary_verdict_is": ("OUR fit to the Kepler light curve against OUR fit to the TESS "
                               "light curve — same fitter, same fold, same masked baseline, "
                               "same exposure-shrunk core, same bootstrap; only the archive "
                               "product and the time system differ"),
        "like_for_like_verdicts": ll_counts,
        "like_for_like_unmeasured_reasons": ll_unmeasured,
        "n_compared_like_for_like": int(ll_counts[LL_CHANGED] + ll_counts[LL_UNCHANGED]
                                        + ll_counts[LL_UNRESOLVED]),
        # --- the SECONDARY verdict, now a diagnostic of the CATALOGUES -----
        "verdict": verdict, "reason": reason,
        "verdict_is_primary": False,
        "verdict_is": ("our TESS-era fit against the two CATALOGUE depths. With both eras now "
                       "fitted here, this says which catalogue is right about the TESS era — "
                       "it is a diagnostic of the CATALOGUES, not of the sky"),
        "generated_utc": _now(),
        "n_shortlist": int(len(measurements)),
        "n_measured": int(counts[MATCH_KEPLER] + counts[MATCH_TOI] + counts[MATCH_NEITHER]
                          + counts[MATCH_BOTH]),
        "target_verdicts": counts,
        "unmeasured_reasons": unmeasured,
        "koi_catalogue_check": koi_check,
        "reduction_ensemble": ensemble_block,
        "flags": flags,
        "targets": (measurements[[c for c in SUMMARY_TARGET_COLUMNS
                                  if c in measurements.columns]].to_dict(orient="records")
                    if len(measurements) else []),
        "acquisition": {k: (acquire_report or {}).get(k) for k in
                        ("koi_ephemeris_status", "toi_status", "lc_status_counts", "lc_routes",
                         "budget_s", "elapsed_s", "budget_exhausted", "kepler_era_enabled",
                         "kepler_lc_status_counts", "kepler_lc_routes", "kepler_budget_s",
                         "kepler_elapsed_s", "kepler_budget_exhausted",
                         "ensemble_enabled", "ensemble_budget_s", "ensemble_elapsed_s",
                         "ensemble_budget_exhausted", "ensemble_kepler_budget_s",
                         "ensemble_kepler_elapsed_s", "ensemble_kepler_budget_exhausted",
                         "n_reduction_members", "reduction_member_status_counts")},
        "config": {"compare": {"n_agree": cmp_p.n_agree, "sigma_sys_ln": cmp_p.sigma_sys_ln,
                               "apply_band_ratio": cmp_p.apply_band_ratio,
                               "min_detectable_ln_ratio": cmp_p.min_detectable_ln_ratio,
                               "koi_check_n_agree": cmp_p.koi_check_n_agree},
                   "fit": {k: getattr(fit, k) for k in fit.__dataclass_fields__},
                   "mast": {"authors": list(mast.authors), "budget_s": mast.budget_s,
                            "per_target_budget_s": mast.per_target_budget_s,
                            "target_timeout_s": mast.target_timeout_s,
                            "kepler_enabled": mast.kepler_enabled,
                            "kepler_authors": list(mast.kepler_authors),
                            "kepler_budget_s": mast.kepler_budget_s,
                            "kepler_per_target_budget_s": mast.kepler_per_target_budget_s,
                            "kepler_max_quarters": mast.kepler_max_quarters},
                   "ensemble": ensemble_block["config"]},
        "checks_not_performed": [
            "per_pixel_centroid_test (a deeper TESS transit can ORIGINATE on a different "
            "star inside the pixel; nothing here excludes that)",
            "achromaticity (one band per epoch, as at stage 1)",
            "independent_ephemeris (both eras are folded on the KOI ephemeris, which is "
            "itself a Kepler-era product; a depth is measured, an ephemeris is not)",
            "aperture_photometry_of_our_own (the ensemble measures every reduction the "
            "ARCHIVE serves; it does not re-extract the pixels with a different aperture, "
            "so a systematic common to every delivered pipeline would not show up in the "
            "spread)",
        ],
        "note": ("the PRIMARY verdict is the like-for-like one: BOTH eras are fitted here by "
                 "the SAME code, so MEASURED_DEPTH_CHANGED means the change survives a "
                 "comparison in which nothing but the sky differs, and MEASURED_DEPTH_UNCHANGED "
                 "means the KOI catalogue depth is simply wrong for this object. "
                 "MEASURED_DEPTH_UNRESOLVED is an agreement WITHOUT the power to have seen the "
                 "change under test and is never a refutation. The catalogue verdicts "
                 "(MEASURED_DEPTH_MATCHES_KEPLER / _TOI) are kept, and are now a diagnostic of "
                 "the CATALOGUES rather than of the sky — they say which table is right about "
                 "the TESS era. The primary z is formed from the TOTAL error — bootstrap and "
                 "the REDUCTION-ENSEMBLE spread in quadrature — so a ratio that survives it has "
                 "survived every reduction the archive serves, and one that does not comes back "
                 "MEASURED_DEPTH_UNRESOLVED; the DEPTH itself is still the deduplicated one and "
                 "the ensemble never moves it. A measured change is still NOT a detection: the "
                 "centroid test is outstanding. NO_DATA_REACHED / LIKE_FOR_LIKE_NO_DATA is not "
                 "a null result and is not written up (CLAUDE.md)"),
    }
    _write(out / "summary.json", summary)
    if ensemble_block.get("background_test_finding"):
        print(f"[growth-stage2] assess: {ensemble_block['background_test_finding']}")
    print(f"[growth-stage2] assess: PRIMARY {ll_verdict} — {ll_reason}; {ll_counts}")
    print(f"[growth-stage2] assess: catalogues {verdict} — {reason}; {counts}; "
          f"KOI table {koi_counts}")
    return summary


SUMMARY_TARGET_COLUMNS = (
    "kepoi_name", "kepler_name", "kepid", "tic_id", "toi",
    # --- THE PRIMARY COMPARISON: our Kepler-era fit vs our TESS-era fit ----
    "like_for_like_verdict", "like_for_like_unmeasured_reason",
    "depth_kepler_measured_ppm", "depth_kepler_measured_err_ppm",
    "depth_kepler_measured_in_tess_band_ppm", "kepler_depth_measured_err_method",
    "depth_tess_measured_ppm", "depth_tess_measured_err_ppm",
    "depth_kepler_measured_total_err_ppm", "depth_tess_measured_total_err_ppm",
    "z_measured_eras", "sigma_measured_eras", "z_measured_eras_stat_only",
    "sigma_measured_eras_stat_only", "reduction_systematic_applied",
    "measured_depth_ratio", "measured_eras_agree",
    "ln_ratio_under_test", "detectable_ln_ratio", "claim_excluded_sigma",
    "catalogue_ln_ratio_under_test",
    # --- THE REDUCTION ENSEMBLE: the systematic that enters the total error ---
    "n_members", "n_members_measured", "n_members_unavailable",
    "depth_reduction_spread_ppm", "depth_reduction_spread_fraction",
    "depth_reduction_spread_status", "depth_reduction_p16_ppm", "depth_reduction_p84_ppm",
    "depth_reduction_min_ppm", "depth_reduction_max_ppm", "depth_reduction_median_ppm",
    "depth_reduction_max_offset_from_primary_ppm", "ensemble_incomplete",
    "ensemble_unavailable_reasons", "ensemble_members",
    "kepler_n_members", "kepler_n_members_measured", "kepler_n_members_unavailable",
    "kepler_depth_reduction_spread_ppm", "kepler_depth_reduction_spread_fraction",
    "kepler_depth_reduction_spread_status", "kepler_depth_reduction_min_ppm",
    "kepler_depth_reduction_max_ppm", "kepler_ensemble_incomplete",
    "kepler_ensemble_unavailable_reasons", "kepler_ensemble_members",
    # --- the background/crowding test, by number ---------------------------
    "sap_vs_pdcsap_verdict", "sap_vs_pdcsap_n_pairs", "sap_vs_pdcsap_authors",
    "sap_minus_pdcsap_ppm", "sap_minus_pdcsap_err_ppm", "sap_minus_pdcsap_z",
    "sap_minus_pdcsap_fraction", "background_direction",
    "kepler_sap_vs_pdcsap_verdict", "kepler_sap_minus_pdcsap_ppm",
    "kepler_sap_minus_pdcsap_z", "kepler_background_direction",
    # --- what our Kepler-era fit says about the KOI TABLE ------------------
    "koi_depth_verdict", "z_kepler_measured_vs_koi", "kepler_measured_over_koi_depth",
    "koi_depth_unchecked_reason",
    # --- the Kepler era's provenance --------------------------------------
    "kepler_lc_status", "kepler_lc_route", "kepler_n_quarters", "kepler_n_quarters_measured",
    "kepler_quarter_list", "kepler_authors", "kepler_exptimes_s", "kepler_cadences",
    "kepler_has_short_cadence", "kepler_n_files_merged", "kepler_n_transits",
    "kepler_t0_bkjd_used", "kepler_n_epochs_propagated", "kepler_ephemeris_sigma_minutes",
    "kepler_smeared", "kepler_depth_is_lower_bound", "kepler_odd_even_diff_ppm",
    "kepler_odd_even_sigma", "kepler_sector_scatter_chi2_per_dof",
    "kepler_n_sectors_dropped_duplicate", "kepler_flags", "kepler_era_unmeasured_reason",
    # --- the secondary, catalogue-diagnostic comparison --------------------
    "verdict", "unmeasured_reason", "tess_era_unmeasured_reason",
    "depth_measured_total_err_ppm",
    "lc_status", "lc_route", "n_sectors", "n_sectors_measured", "sector_list", "authors",
    "exptimes_s", "period_days", "t0_bkjd", "t0_btjd_used", "n_epochs_propagated",
    "ephemeris_sigma_minutes", "duration_hours", "n_transits", "depth_measured_ppm",
    "depth_measured_err_ppm", "depth_measured_err_method", "depth_kepler_ppm",
    "depth_kepler_in_tess_band_ppm", "depth_toi_ppm", "ld_band_ratio", "z_vs_kepler",
    "z_vs_toi", "z_toi_vs_kepler", "references_separated", "n_references_usable",
    "agrees_with_kepler", "agrees_with_toi", "smeared", "depth_is_lower_bound",
    "odd_even_diff_ppm", "odd_even_sigma", "sector_scatter_chi2_per_dof", "flags",
)


def stage2_run(stage: str = "all", *, out_dir=None, conf: dict | None = None, query_fn=None,
               lc_fn=None, kepler_lc_fn=None, ensemble_lc_fn=None, kepler_ensemble_lc_fn=None,
               shortlist: pd.DataFrame | None = None) -> dict:
    """Run one stage, a comma list, or all.  Returns the last stage's report."""
    from .run import load_growth_config  # noqa: PLC0415

    conf = conf if conf is not None else load_growth_config()
    out = Path(out_dir) if out_dir else Path("results") / "growth" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage2_probe(conf, out)
        elif s == "measure":
            rep = stage2_measure(conf, out, query_fn=query_fn, lc_fn=lc_fn,
                                 kepler_lc_fn=kepler_lc_fn, ensemble_lc_fn=ensemble_lc_fn,
                                 kepler_ensemble_lc_fn=kepler_ensemble_lc_fn,
                                 shortlist=shortlist)
        elif s == "assess":
            rep = stage2_assess(conf, out)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="seti growth-stage2",
        description="GROWTH stage 2 (S57): measure BOTH the Kepler-era and the TESS-era depth "
                    "from the light curves with the SAME fitter (the primary, like-for-like "
                    "comparison) and keep the catalogue comparison as a diagnostic of the "
                    "catalogues")
    p.add_argument("--stage", default="all", choices=("probe", "measure", "assess", "all"))
    p.add_argument("--out-dir", default="results/growth/stage2", help="results directory")
    a = p.parse_args(argv)
    rep = stage2_run(a.stage, out_dir=a.out_dir)
    if isinstance(rep, dict):
        if rep.get("primary_verdict"):
            print(f"[growth-stage2] PRIMARY verdict (like-for-like, both eras fitted here): "
                  f"{rep['primary_verdict']}")
        if rep.get("verdict"):
            print(f"[growth-stage2] catalogue verdict (secondary): {rep['verdict']}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BACKGROUND_VERDICTS", "BG_AGREE", "BG_DISAGREE", "BG_PDC_DEEPER", "BG_SAP_DEEPER",
    "BG_UNAVAILABLE",
    "BKJD_MINUS_BTJD", "BKJD_OFFSET", "BTJD_OFFSET", "CompareParams", "DEFAULT_AUTHORS",
    "Deadline", "ENSEMBLE_MEMBER_COLUMNS", "ENSEMBLE_MEMBER_STATUSES",
    "ENSEMBLE_NOT_ATTEMPTED", "ENSEMBLE_NO_AUTHOR", "ENSEMBLE_NO_FLUX_COLUMN", "ERAS", "ERA_KEPLER", "ERA_TESS",
    "EnsembleParams", "FLUX_COLUMNS", "FitParams", "KEPLER_AUTHORS",
    "KEPLER_FLUX_COLUMNS", "KEPLER_LONG_CADENCE_S", "KEPLER_PRODUCT_SUBGROUPS",
    "KEPLER_QUALITY_COLUMNS", "KEPLER_SHORT_CADENCE_MAX_S", "KEPLER_SHORT_CADENCE_S",
    "KEPLER_TARGET_NAME_FORMS", "KOI_DEPTH_CONFIRMED", "KOI_DEPTH_CONTRADICTED",
    "KOI_DEPTH_UNCHECKED", "KOI_DEPTH_VERDICTS", "LIKE_FOR_LIKE_VERDICTS", "LL_CHANGED",
    "LL_ERA_UNMEASURED", "LL_UNCHANGED", "LL_UNRESOLVED",
    "MATCH_BOTH", "MATCH_KEPLER", "MATCH_NEITHER", "MATCH_TOI",
    "MATCH_UNMEASURED", "MastParams", "RUN_CONFIRMED", "RUN_LL_CHANGED", "RUN_LL_NO_DATA",
    "RUN_LL_UNCHANGED", "RUN_LL_UNRESOLVED", "RUN_LL_VERDICTS", "RUN_NO_DATA", "RUN_REFUTED",
    "RUN_UNRESOLVED", "RUN_VERDICTS", "SPREAD_NOT_ATTEMPTED", "SPREAD_OK", "SPREAD_STATUSES",
    "SPREAD_TOO_FEW_MEMBERS", "STAGES", "TARGET_VERDICTS", "UNMEASURED_BUDGET",
    "UNMEASURED_KEPLER_DISABLED",
    "UNMEASURED_NO_EPHEMERIS", "UNMEASURED_NO_REFERENCE", "UNMEASURED_NO_TRANSIT",
    "UNMEASURED_QUERY_FAILED",
    "UNMEASURED_REASONS", "UNMEASURED_ZERO_ROWS", "binned_fold", "bkjd_to_bjd", "bkjd_to_btjd",
    "btjd_to_bjd", "btjd_to_bkjd", "combine_transit_depths", "compare_kepler_to_catalogue",
    "compare_measured_eras", "compare_three_depths",
    "core_half_width", "dedupe_sectors", "default_kepler_lc_fn", "default_lc_fn",
    "ensemble_member_grid",
    "epoch_in_era", "fetch_kepler_lightcurves", "fetch_koi_ephemerides", "fetch_lightcurves",
    "fetch_toi_depths", "fit_transits", "fold", "kepler_cadence_label",
    "lightkurve_kepler_lc_fn", "lightkurve_lc_fn", "load_shortlist", "main",
    "mast_fits_lc_fn", "mast_kepler_fits_lc_fn", "mast_probe", "measure_one",
    "measure_reduction_ensemble", "measure_target",
    "merge_kepler_segments", "propagate_epoch",
    "read_kepler_lc_fits", "read_tess_lc_fits", "reduction_ensemble_report",
    "reduction_spread", "run_like_for_like_verdict", "run_verdict", "sap_vs_pdcsap",
    "stage2_assess", "stage2_measure", "stage2_probe",
    "stage2_run", "synth_lightcurve", "total_depth_error", "trapezoid_transit",
]

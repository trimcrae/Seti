"""Gaia SSO astrometry acquisition for SEXTANT.  Runner-only network code.

SEXTANT computes LOOM's observable — the ephemeris residual of a known minor
planet — from Gaia's asteroid astrometry instead of Rubin's ``ssSource.ephOffset``.
Rubin serves that residual pre-computed at **arcsecond** scale and has been off
sky since the night of 13/14 July 2026 (``docs/rubin-outage.md``, verdict
``SKY_STOPPED``).  Gaia serves the *ingredients* at **milliarcsecond** scale, for
46,264,083 observations of 156,823 objects, and it serves them today.

This module is the acquisition layer and nothing else.  It pulls rows; it does not
compute a residual, project onto an axis, or decide anything.  Its whole job is to
get every column the residual computation will need, in complete and honestly
labelled chunks, off a service that will not hand over 46 million rows in one
query.

Egress to ``gea.esac.esa.int`` is blocked in the development sandbox (CONNECT
tunnel 403), exactly as it is for the brokers, so **every network path here runs
on a GitHub Actions runner** via ``.github/workflows/sextant-probe.yml``.
Everything that can be tested without the network — chunk planning, quality cuts,
cross-release reconciliation, and the interpretation of each probe measurement —
is a pure function and is tested in ``tests/test_sextant_acquire.py``.

What the service actually holds (MEASURED on the runner, 2026-08-25)
--------------------------------------------------------------------
Service: ``https://gea.esac.esa.int/tap-server/tap`` — IVOA TAP, public, no auth.

===============================  ===========  =======  ========
table                            rows         columns  objects
===============================  ===========  =======  ========
``gaiafpr.sso_observation``      46,264,083   34       156,823
``gaiadr3.sso_observation``      23,336,467   35       158,152
``gaiafpr.sso_source``           156,823      —        —
``gaiadr3.sso_source``           158,152      —        —
===============================  ===========  =======  ========

The two observation tables share **32** columns; DR3 adds ``g_mag``/``g_flux``/
``g_flux_error`` (32 + 3 = 35) and FPR adds ``is_rejected``/``fov`` (32 + 2 = 34).
Those two sums are the reason :data:`SHARED_COLUMNS` can be trusted to be
complete, and ``test_column_inventory_reproduces_the_measured_counts`` pins them:
if anyone drops a column from that tuple the arithmetic stops matching the live
column counts and the test fails, rather than the residual computation silently
losing the observer state vector.

The columns that matter, and why each one is non-negotiable
-----------------------------------------------------------
The next stage computes O−C against **independent** JPL orbits (Gaia's own SSO
observations were used to fit Gaia's own orbit solutions, so a residual taken
against those is minimised by construction and means nothing —
``docs/substitute-surveys.md``), and projects the result onto the along-scan axis.
For that to be possible this layer must carry:

* ``ra``, ``dec`` (deg) — the observation;
* ``ra_error_random``, ``ra_error_systematic``, ``dec_error_random``,
  ``dec_error_systematic`` (mas) and the two ``ra_dec_correlation_*`` terms — the
  error model is **given, not assumed**, and it is given as two separable parts.
  The systematic part is common to observations sharing a scan and does not average
  down; treating the quadrature sum as random is the σ/√n error that cost LOOM its
  first 150 "anomalies" (``docs/loom.md`` §2.2), transposed to a new dataset;
* ``epoch``, ``epoch_utc``, ``epoch_err`` (d) — the time.  See the time-scale
  warning below, which is load-bearing at mas precision;
* ``position_angle_scan`` (deg) — **the axis**.  Gaia's precision is strongly
  anisotropic: along-scan is the good axis by more than an order of magnitude, and
  across-scan is nearly uninformative for a single CCD transit.  Every residual has
  to be projected onto this direction, and it is the direct analogue of Rubin's
  along-track/cross-track split.  Drop it and the whole channel is impossible;
* ``x_gaia … vz_gaia`` and their ``*_geocentric`` twins (AU, AU/d) — the observer's
  own state.  With these the light-time-corrected prediction needs no model of
  Gaia's orbit at all.  Drop them and the next agent has to reconstruct Gaia's
  position at L2 to ~100 km, which is not reconstructible;
* ``number_mp``, ``denomination`` — the join to JPL/MPC orbit solutions;
* ``observation_id``, ``transit_id``, ``source_id``, ``solution_id`` — identity,
  deduplication, and the independence question below;
* ``astrometric_outcome_ccd``, ``astrometric_outcome_transit`` (both tables) and
  ``is_rejected`` (FPR) — **the primary observable of this channel**, not a
  quality filter.  See the next section.

The rejected observations are the signal
----------------------------------------
This is a change of question, made 2026-08-25 after a prior-art sweep, and it
governs the whole design of this module.

Screening Gaia's *surviving* SSO residuals for a population with anomalous
non-gravitational signatures is **not novel**: it is occupied by the Gaia SSO
team's own active programme.  Liberato, Tanga, Mary et al., *Follow the wobble:
Statistical methods to detect astrometric binary asteroids in Gaia FPR*
(arXiv:2605.22702, May 2026) already projects post-fit residuals from
``gaiafpr.sso_observation`` onto the along-scan axis with a Monte-Carlo noise
model and a trend-detection stage, yielding 343 candidates; and Dziadura,
Bartczak & Oszkiewicz (A&A 693, A31; arXiv:2411.09750) already fitted
non-gravitational ``A2`` to 54,094 inner-main-belt bodies on the same data.

**Every one of those searches works post-fit.**  An object whose astrometry
*fails* to fit is invisible to all of them: it is removed before their first
statistic is computed.  So the question SEXTANT asks is whether some objects are
systematically **un-fittable** — a screen on the *rejection pattern* rather than
on the surviving residuals — and that makes ``is_rejected``,
``astrometric_outcome_ccd`` and ``astrometric_outcome_transit`` the measurement.

Three consequences bind this module:

1. **Nothing here ever filters a rejected or bad-outcome row out of a query.**
   No ADQL built by this module carries a rejection clause; the rows come back
   labelled and the caller decides.  ``QualityCuts.drop_rejected`` therefore
   defaults to ``False``, which reverses the obvious default deliberately — a
   pipeline that quietly discarded the rejected rows would make this channel
   impossible and the loss would be invisible in every output it produced.
   ``test_no_data_pull_ever_filters_on_a_rejection_flag`` pins it.
2. **The denominator travels with the numerator.**  A rejection *rate* is
   meaningless without the attempts, exactly as TOCSIN's event rate is
   meaningless without its forced-photometry denominator.
   :meth:`GaiaSSO.rejection_census` returns, per object, how many observations
   were written, how many were rejected, and the full breakdown by outcome code —
   in one ``GROUP BY`` rather than by downloading 46 million rows to count them.
   The denominator's own limit is stated at :func:`rejection_ledger`: it counts
   observations that *entered the table*, which is not the same as transits that
   *should have occurred*.
3. **The outcome codes stay data.**  Their meanings are unverified here and are
   now load-bearing rather than cosmetic, so no value is hard-coded as "good".
   The probe measures the distribution per table and per object, and
   :func:`interpret_rejection_fraction` compares the global fraction against the
   published Gaia outlier fractions (~0.58% in DR3, ~1% in DR2) — which is the
   check that says whether ``is_rejected`` marks what we think it marks.

:data:`REQUIRED_FOR_RESIDUALS` names the subset without which the downstream work
cannot proceed, and :func:`check_columns_for_residuals` refuses a pull that is
missing any of them.  A pull that quietly came back without ``position_angle_scan``
would produce a plausible, publishable, wrong answer; that is the failure mode this
repository fears most, so it is an error here rather than a NaN there.

Three traps that would silently produce a wrong result
-------------------------------------------------------
**1. One transit is not one independent measurement.**  A Gaia field-of-view
transit crosses a row of AF CCDs, so a single crossing produces up to ~9 rows in
``sso_observation``, ~4.4 s apart, sharing one attitude solution and one scan
angle.  They are *not* independent samples of the astrometric error: the
systematic component is common to all of them.  Averaging them with σ/√N
understates the transit error by up to 3×, which is precisely the bug that
manufactured LOOM's first candidate list.  The number of rows per transit is
**UNVERIFIED** here and is measured by the probe (:data:`OPEN_QUESTIONS`
``rows_per_transit``); :func:`transit_groups` exists so the next stage can collapse
to transits before it fits anything.

**2. The time scale.**  ``epoch`` is believed to be days from a 2010.0 reference in
**TCB**, and ``epoch_utc`` the same instant in UTC — but the zero point, the unit
and the scale are all **UNVERIFIED** from the sandbox.  This is not pedantry: TCB
runs ahead of TDB by ~1.55e-8 in rate, some ~10 s by the Gaia era, and a main-belt
asteroid moving at 30″/day covers 3.5 mas in 10 s.  At Gaia's precision that is a
multi-sigma systematic applied coherently to the entire catalogue — an along-track
offset shared by every object, which is exactly what an "anomalous along-scan
residual population" looks like.  The probe measures ``MIN``/``MAX`` of both
columns and of their difference, and records the TAP_SCHEMA ``unit``/``description``
verbatim; :func:`interpret_epoch_zero_point` turns those numbers into a verdict.

**3. The frame of the state vectors.**  The columns are documented in AU; the
*frame* is not stated in anything reachable from here.  Barycentric ICRS
(equatorial) and barycentric ecliptic differ by the 23.4° obliquity, and LOOM has
this exact scar: ``residuals.decompose_offset`` deliberately avoids the alert
schema's state vectors because "a 23.4° frame error would rotate along-track into
cross-track and destroy the exact quantity being measured" (``docs/loom.md`` §2.1).
Here the vectors cannot be avoided — they *are* the observer position — so the
frame has to be measured.  It can be, cheaply and decisively: Gaia orbits L2, so
in an **ecliptic** frame its barycentric ``z`` stays within a few 1e-3 AU of zero,
while in an **equatorial** frame ``z = y_ecl·sin(23.44°)`` sweeps ±0.4 AU once a
year.  ``MIN``/``MAX`` of ``z_gaia`` therefore separates the two by two orders of
magnitude.  :func:`interpret_state_vector_frame` is that test, and the probe runs
it on both tables.

Why the chunk key is the object, not the epoch
-----------------------------------------------
See :meth:`GaiaSSO.iter_observation_chunks`.  In one line: the residual of an
object is a time series and every fit needs the whole arc, so chunking on
``number_mp`` hands back complete objects while chunking on ``epoch`` hands back
truncated arcs that must be reassembled over the entire 46-million-row download
before any of them can be used.  Epoch chunking is provided too, because coverage
questions and cross-release comparisons are naturally time-sliced, and it is
labelled with what it costs.

The synchronous-query trap
--------------------------
The ESA Gaia TAP service is believed to cap **synchronous** anonymous queries at a
small number of rows (2000 is the widely-quoted figure) while allowing far more
asynchronously.  If that is true and this module used the synchronous path, every
bulk pull would come back silently truncated and the channel would screen ~0.004%
of the catalogue while reporting success — an unrecoverable, invisible failure.
So :meth:`GaiaSSO.query` is **asynchronous by default**, :meth:`GaiaSSO.query_sync`
is opt-in and used only for tiny metadata queries, every result is checked against
the requested ``maxrec`` for overflow, and the probe measures the real synchronous
cap rather than trusting the number above (:data:`OPEN_QUESTIONS` ``sync_row_cap``).

Composition, not reinvention
----------------------------
:class:`~seti.tocsin.brokers.AlerceTAP` is named for the service TOCSIN uses but
its ``query`` path is plain IVOA TAP: a ``pyvo.dal.TAPService`` built on a session
that injects a **real per-request socket timeout** (``requests.Session`` has no
honoured ``timeout`` attribute, and the hopeful version of that line once burned a
full 180-minute job on one stuck query), retries with exponential backoff, and
fails fast on ADQL errors instead of retrying a query that will fail identically
four times.  All three were paid for once already.  :class:`GaiaSSO` composes it
rather than re-deriving them, exactly as :class:`seti.loom.acquire.AlerceSSO` does,
and adds only the asynchronous path and the overflow check that this service needs.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..tocsin.brokers import AlerceTAP, BrokerError

# ---------------------------------------------------------------------------
# The service, the tables, and what was measured of them
# ---------------------------------------------------------------------------
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

DR3_OBSERVATION = "gaiadr3.sso_observation"
FPR_OBSERVATION = "gaiafpr.sso_observation"
DR3_SOURCE = "gaiadr3.sso_source"
FPR_SOURCE = "gaiafpr.sso_source"

RELEASES: tuple[str, ...] = ("gaiadr3", "gaiafpr")

OBSERVATION_TABLE: dict[str, str] = {
    "gaiadr3": DR3_OBSERVATION,
    "gaiafpr": FPR_OBSERVATION,
}
SOURCE_TABLE: dict[str, str] = {
    "gaiadr3": DR3_SOURCE,
    "gaiafpr": FPR_SOURCE,
}

# MEASURED on the runner 2026-08-25.  Held here so that a later probe run can be
# diffed against them: a row count that has moved means the archive republished,
# and a column count that has moved means the schema changed under us.
MEASURED_ROWS: dict[str, int] = {
    FPR_OBSERVATION: 46_264_083,
    DR3_OBSERVATION: 23_336_467,
    FPR_SOURCE: 156_823,
    DR3_SOURCE: 158_152,
}
MEASURED_COLUMNS: dict[str, int] = {
    FPR_OBSERVATION: 34,
    DR3_OBSERVATION: 35,
}

# ---------------------------------------------------------------------------
# The column inventory
# ---------------------------------------------------------------------------
# Present on BOTH observation tables (measured 2026-08-25).  Exactly 32 entries:
# with the 3 DR3-only and 2 FPR-only columns below that reproduces the measured
# 35 and 34, which is the only offline check available that this tuple is
# complete.  `test_column_inventory_reproduces_the_measured_counts` enforces it.
SHARED_COLUMNS: tuple[str, ...] = (
    # identity and provenance
    "observation_id", "transit_id", "source_id", "solution_id",
    "number_mp", "denomination",
    # the observation
    "ra", "dec",
    # the error model, kept in its two separable parts.  Collapsing these into one
    # sigma is wrong: the systematic part is common to observations sharing a scan
    # and does NOT average down over a transit.
    "ra_error_random", "ra_error_systematic",
    "dec_error_random", "dec_error_systematic",
    "ra_dec_correlation_random", "ra_dec_correlation_systematic",
    # time
    "epoch", "epoch_utc", "epoch_err",
    # THE AXIS.  Gaia's precision is anisotropic; every residual is projected onto
    # this direction.  Without it the channel does not exist.
    "position_angle_scan",
    # per-observation quality (meanings UNVERIFIED; see OPEN_QUESTIONS)
    "astrometric_outcome_ccd", "astrometric_outcome_transit",
    # the observer's own state, barycentric and geocentric, AU and AU/d.  With
    # these the light-time-corrected prediction needs no model of Gaia's orbit.
    "x_gaia", "y_gaia", "z_gaia", "vx_gaia", "vy_gaia", "vz_gaia",
    "x_gaia_geocentric", "y_gaia_geocentric", "z_gaia_geocentric",
    "vx_gaia_geocentric", "vy_gaia_geocentric", "vz_gaia_geocentric",
)

# DR3 only: the photometry.  Not needed for the residual, but it is the cheapest
# available handle on the binary-photocentre contaminant (an unresolved binary's
# wobble is exactly the anomaly being searched for -- docs/substitute-surveys.md),
# because a photocentre displacement correlates with the lightcurve phase.
DR3_ONLY_COLUMNS: tuple[str, ...] = ("g_mag", "g_flux", "g_flux_error")

# FPR only.  `is_rejected` is the FPR reduction's own outlier flag; `fov` is which
# of Gaia's two fields of view saw the transit, which matters because the two have
# different systematics and a residual that flips sign with `fov` is instrumental.
FPR_ONLY_COLUMNS: tuple[str, ...] = ("is_rejected", "fov")

ALL_COLUMNS: dict[str, tuple[str, ...]] = {
    "gaiadr3": SHARED_COLUMNS + DR3_ONLY_COLUMNS,
    "gaiafpr": SHARED_COLUMNS + FPR_ONLY_COLUMNS,
}

# Without every one of these the O-C-against-JPL-orbits computation cannot be done
# at all, so a pull that lacks one is an error here rather than a NaN three stages
# later.  `position_angle_scan` and the observer state vectors are on this list for
# the reasons in the module docstring, and removing either of them from a query to
# "save bandwidth" silently ends the channel.
REQUIRED_FOR_RESIDUALS: tuple[str, ...] = (
    "number_mp", "observation_id", "transit_id",
    "ra", "dec",
    "ra_error_random", "ra_error_systematic",
    "dec_error_random", "dec_error_systematic",
    "epoch", "epoch_utc",
    "position_angle_scan",
    "x_gaia", "y_gaia", "z_gaia", "vx_gaia", "vy_gaia", "vz_gaia",
)

# Columns whose values are flags with UNVERIFIED meanings.  These are the
# channel's PRIMARY OBSERVABLE, not a quality filter -- see the module docstring.
QUALITY_FLAG_COLUMNS: tuple[str, ...] = (
    "astrometric_outcome_ccd", "astrometric_outcome_transit", "is_rejected",
)

# What the rejection screen needs.  Disjoint in purpose from
# REQUIRED_FOR_RESIDUALS and checked separately, because a pull can be perfectly
# adequate for one and useless for the other: the residual path needs the
# astrometry and the axis, the rejection path needs the flags and the epoch that
# says WHEN an object stopped being fittable.  `is_rejected` is FPR-only, so it is
# not in the shared list and is checked per release.
REQUIRED_FOR_REJECTION_SCREEN: tuple[str, ...] = (
    "number_mp", "observation_id", "transit_id", "epoch",
    "astrometric_outcome_ccd", "astrometric_outcome_transit",
)

# Published Gaia SSO astrometric outlier fractions, for calibrating what
# `is_rejected` actually marks.  If the measured fraction lands near these, the
# column marks the documented outlier rejection; if it lands at 30%, it marks
# something else entirely and every rate built on it would be a rate of something
# else.  DR3 ~0.58%, DR2 ~1%.
PUBLISHED_OUTLIER_FRACTION: dict[str, float] = {"gaiadr3": 0.0058, "gaiadr2": 0.010}

# ---------------------------------------------------------------------------
# Every inferred-but-unverified fact, and the probe query that settles it
# ---------------------------------------------------------------------------
# This table is not decoration.  `test_every_open_question_is_measured_by_the_probe`
# asserts that each entry names a probe key that the probe actually produces, so a
# question cannot be recorded here and then quietly never asked.
OPEN_QUESTIONS: dict[str, dict[str, str]] = {
    "sync_row_cap": {
        "question": "How many rows will a synchronous anonymous query return "
                    "before the service silently truncates?",
        "assumed": "2000 (the widely-quoted ESA Gaia archive figure) — UNVERIFIED",
        "why_it_matters": "If this module used the synchronous path and the cap is "
                          "real, every bulk pull returns 2000 rows and looks "
                          "successful. The channel would screen 0.004% of the "
                          "catalogue and report a clean null. Hence async by "
                          "default and an overflow check on every result.",
        "probe_key": "sync_row_cap",
    },
    "async_row_cap": {
        "question": "What are the service's declared default and hard output "
                    "limits for asynchronous queries?",
        "assumed": "3,000,000 rows for anonymous users — UNVERIFIED",
        "why_it_matters": "It sets the largest chunk that can be requested, and "
                          "therefore the whole chunking plan. A chunk that hits "
                          "the cap is truncated, not empty, so it fails silently.",
        "probe_key": "service_limits",
    },
    "epoch_zero_point": {
        "question": "What are the zero point, unit and time scale of `epoch`, and "
                    "what does `epoch_utc` hold?",
        "assumed": "days since 2010-01-01T00:00 in TCB, and the same instant in "
                   "UTC — UNVERIFIED",
        "why_it_matters": "TCB leads TDB by ~10 s in the Gaia era. A main-belt "
                          "asteroid covers 3.5 mas in 10 s, so a scale error is a "
                          "coherent multi-sigma along-track offset applied to the "
                          "entire catalogue — indistinguishable from the signal.",
        "probe_key": "epoch_ranges",
    },
    "state_vector_frame": {
        "question": "Are x_gaia/y_gaia/z_gaia barycentric ECLIPTIC or barycentric "
                    "EQUATORIAL (ICRS)?",
        "assumed": "ICRS equatorial — UNVERIFIED",
        "why_it_matters": "The two differ by the 23.44 deg obliquity. LOOM avoids "
                          "its alert-schema state vectors for exactly this reason: "
                          "a frame error rotates along-track into cross-track and "
                          "destroys the quantity being measured.",
        "probe_key": "state_vector_frame",
    },
    "position_angle_scan_convention": {
        "question": "Is position_angle_scan measured North through East, and does "
                    "it wrap at 0..360 or -180..180?",
        "assumed": "degrees, North through East, 0..360 — UNVERIFIED",
        "why_it_matters": "A 90 deg convention error swaps along-scan for "
                          "across-scan, i.e. swaps the good axis for the one that "
                          "carries almost no information.",
        "probe_key": "position_angle_scan_range",
    },
    "outcome_flag_meanings": {
        "question": "What values do astrometric_outcome_ccd and "
                    "astrometric_outcome_transit take, and in what proportion?",
        "assumed": "nothing — the meanings are NOT known here, and no value is "
                   "hard-coded as 'good'",
        "why_it_matters": "These codes are the channel's PRIMARY OBSERVABLE, not "
                          "a filter: the search is for objects that are "
                          "systematically un-fittable, which every post-fit "
                          "search in the literature is blind to by construction. "
                          "Hard-coding an interpretation would bake in the very "
                          "assumption being tested, so the codes stay data and the "
                          "probe measures their distribution.",
        "probe_key": "quality_flag_distributions",
    },
    "is_rejected_semantics": {
        "question": "What type and values does gaiafpr is_rejected take, what "
                    "global fraction does it mark, and how does it relate to the "
                    "outcome codes?",
        "assumed": "boolean, and marking the documented astrometric outlier "
                   "rejection at ~0.58% (DR3) to ~1% (DR2) — UNVERIFIED, and the "
                   "type may be int or string",
        "why_it_matters": "The fraction is the calibration: near 0.6-1% and the "
                          "column marks what the papers call outlier rejection, so "
                          "the rejected set is 1e5-1e6 rows and comfortably "
                          "fetchable. Far from it and the column marks something "
                          "else, and every rate built on it is a rate of something "
                          "else. Separately: a string 'false' is truthy in Python, "
                          "so a naive drop would discard every FPR observation.",
        "probe_key": "rejection_fraction",
    },
    "rejection_denominator": {
        "question": "Does sso_observation contain a row for every transit that "
                    "Gaia's scanning law predicts, or only for those that were "
                    "successfully processed?",
        "assumed": "unknown, and this is the weak point of a rejection RATE",
        "why_it_matters": "A rejection rate needs the attempts. If a wholly failed "
                          "transit writes no row at all, the denominator is itself "
                          "censored in the same direction as the signal, and an "
                          "object that is MORE un-fittable looks like an object "
                          "that was observed less. The cross-check is the "
                          "per-object row count against whatever observation count "
                          "sso_source carries.",
        "probe_key": "rejection_census_sample",
    },
    "rows_per_transit": {
        "question": "How many observation rows does one transit_id produce?",
        "assumed": "up to ~9, one per AF CCD crossing — UNVERIFIED",
        "why_it_matters": "Rows within a transit share an attitude and a scan "
                          "angle and are not independent. Averaging N of them with "
                          "sigma/sqrt(N) understates the transit error by up to 3x "
                          "— the exact bug that produced LOOM's first 150 false "
                          "anomalies.",
        "probe_key": "rows_per_transit",
    },
    "release_overlap": {
        "question": "Is gaiafpr.sso_observation a SUPERSET of "
                    "gaiadr3.sso_observation (a re-reduction over a longer "
                    "window), or an increment to be unioned with it?",
        "assumed": "superset: 46.26M/23.34M = 1.98, and FPR's window is ~66 months "
                   "against DR3's ~34, whose ratio is 1.94 — SUGGESTIVE, NOT "
                   "MEASURED",
        "why_it_matters": "If it is a superset and a caller unions the tables, "
                          "every shared observation is counted twice and the "
                          "residual scatter of every object is wrong. If it is an "
                          "increment and a caller takes FPR alone, half the arc is "
                          "thrown away. See reconcile_observations, which REFUSES "
                          "to guess.",
        "probe_key": "release_overlap",
    },
    "observation_id_stability": {
        "question": "Is observation_id constructed identically in DR3 and FPR, so "
                    "that it can be used as the cross-release dedup key?",
        "assumed": "nothing — this is the open question that gates reconciliation",
        "why_it_matters": "If the id was re-minted in FPR then a key-based dedup "
                          "finds zero overlap and reports two disjoint datasets "
                          "that are in fact the same observations twice.",
        "probe_key": "release_overlap",
    },
    "number_mp_completeness": {
        "question": "Do all observation rows carry a number_mp, or are some "
                    "objects identified only by denomination?",
        "assumed": "the Gaia SSO samples are numbered asteroids, so number_mp is "
                   "always populated — UNVERIFIED",
        "why_it_matters": "Chunking on number_mp would silently skip every row "
                          "where it is NULL, and the objects most likely to lack a "
                          "number are the least ordinary ones.",
        "probe_key": "number_mp_completeness",
    },
    "ra_error_cosdec": {
        "question": "Is ra_error_random sigma(alpha) or sigma(alpha·cos(delta))?",
        "assumed": "true-angle, i.e. cos(delta) already applied — UNVERIFIED",
        "why_it_matters": "Getting it wrong mis-scales the RA error by 1/cos(dec) "
                          "and tilts the error ellipse, which biases any projection "
                          "onto the along-scan axis.",
        "probe_key": "ra_error_vs_dec",
    },
    "cross_schema_join": {
        "question": "Does the service accept a join between the gaiadr3 and "
                    "gaiafpr schemas in one query?",
        "assumed": "yes, same database — UNVERIFIED",
        "why_it_matters": "If not, reconciliation must be done client-side after "
                          "two separate pulls, which is what reconcile_observations "
                          "does anyway; this only decides whether the probe can "
                          "measure the overlap cheaply.",
        "probe_key": "release_overlap",
    },
    "source_catalogue_contents": {
        "question": "What do gaiadr3.sso_source and gaiafpr.sso_source carry, and "
                    "does either give a per-object observation count?",
        "assumed": "FPR adds epoch_state_vector, h_state_vector and the two "
                   "var_covar matrices — MEASURED; the rest UNVERIFIED",
        "why_it_matters": "A per-object observation count makes the chunk plan "
                          "exact instead of assuming the catalogue mean of ~295 "
                          "observations per object. Also: those state vectors come "
                          "from GAIA'S OWN orbit fit and must never be used as the "
                          "reference orbit for a residual — that is circular.",
        "probe_key": "schema",
    },
}

# Keys the probe writes at the top level of its record (as opposed to inside
# ``diagnostics``).  ``test_every_open_question_is_measured_by_the_probe`` checks
# every OPEN_QUESTIONS entry against these plus the live diagnostics key set, so a
# question cannot be recorded and then quietly never asked.
RECORD_KEYS_FROM_PROBE: tuple[str, ...] = (
    "schema", "schema_column_counts", "column_check", "service_limits",
    "answers", "reconciliation_trial", "expected", "column_inventory",
    "open_questions",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
# Mirrors the fast-fail list in seti.tocsin.brokers.AlerceTAP.query.  A malformed
# query fails identically every time, so retrying it four times only burns the
# run's budget.
_ADQL_FATAL = ("syntax", "unknown column", "unknown table", "not found",
               "no such field", "unrecognized")

# The catalogue mean, used only when a real per-object count is unavailable:
# 46,264,083 / 156,823 = 295.0 observations per object in FPR.
MEAN_OBS_PER_OBJECT: dict[str, float] = {
    "gaiafpr": MEASURED_ROWS[FPR_OBSERVATION] / MEASURED_ROWS[FPR_SOURCE],
    "gaiadr3": MEASURED_ROWS[DR3_OBSERVATION] / MEASURED_ROWS[DR3_SOURCE],
}

# JD of 2010-01-01T00:00:00, the reference epoch Gaia's SSO tables are believed to
# count from.  Used only by interpret_epoch_zero_point to *test* that belief.
JD_2010_0 = 2455197.5
MJD_2010_0 = 55197.0
JD_MINUS_MJD = 2400000.5


def _fatal_adql(message: str) -> bool:
    low = message.lower()
    return any(k in low for k in _ADQL_FATAL)


def _scalar(v):
    """One VOTable cell as a plain Python value, with NULL as ``None``.

    pyvo hands back masked constants for SQL NULL and ``bytes`` for char columns
    on some servers.  A masked value that leaks through compares equal to nothing
    and serialises to a crash; a ``bytes`` denomination never matches the ``str``
    designations the JPL side uses, and would silently find no object at all.
    """
    if v is None:
        return None
    if v is np.ma.masked:
        return None
    if isinstance(v, np.ma.core.MaskedConstant):
        return None
    if isinstance(v, np.ma.core.MaskedArray) and v.mask.all():
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip()
    if isinstance(v, np.bytes_):
        return bytes(v).decode("utf-8", "replace").strip()
    if isinstance(v, np.str_):
        return str(v).strip()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.ndarray):
        return [_scalar(x) for x in v.tolist()]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _clean(obj):
    """Recursively make a record JSON-serialisable with no NaN and no numpy."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _scalar(obj)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, sort_keys=True,
                               allow_nan=False) + "\n")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir():
            return parent
    return here.parents[3]


def _quote(value) -> str:
    """A single-quoted ADQL string literal with the quote doubled."""
    return "'" + str(value).replace("'", "''") + "'"


def _int_list(values) -> str:
    return ", ".join(str(int(v)) for v in values)


def _str_list(values) -> str:
    return ", ".join(_quote(v) for v in values)


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _truthy(v) -> bool:
    """Interpret an ``is_rejected``-shaped value whose type is UNVERIFIED.

    The column may come back as a numpy bool, as 0/1, or as the strings
    ``'true'``/``'false'`` — and ``bool('false')`` is ``True``, which would drop
    every FPR observation while looking like a working quality cut.  So the string
    case is decided by content, not by Python truthiness, and anything
    unrecognised is treated as *not* rejected, because deleting data on an
    unparsed flag is the more damaging error.
    """
    if v is None:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            return float(v) != 0.0
        except (TypeError, ValueError):
            return False
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0", "", "none", "null"):
        return False
    return False


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SSOResult:
    """What one pull returned, with its degradation stated rather than implied.

    ``truncated`` is the field that matters and the reason this class exists
    rather than a bare list: a TAP service that hits ``maxrec`` returns a short
    table and an OVERFLOW marker, and a caller that ignores it gets a subset of
    the sky with no indication that it is one.
    """

    rows: list[dict] = field(default_factory=list)
    calls: int = 0
    reached: bool = False
    truncated: bool = False
    verdict: str = "NOT_RUN"
    adql: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class Chunk:
    """One half-open slice ``[lo, hi)`` of the chunk key."""

    key: str
    lo: float
    hi: float
    index: int = 0
    depth: int = 0
    expected_rows: float = float("nan")

    def where(self, alias: str = "") -> str:
        p = f"{alias}." if alias else ""
        return f"{p}{self.key} >= {self.lo!r} AND {p}{self.key} < {self.hi!r}"

    def halves(self) -> tuple[Chunk, Chunk] | None:
        """Split for a retry after an overflow, or ``None`` if unsplittable."""
        if self.key == "number_mp":
            lo, hi = int(self.lo), int(self.hi)
            if hi - lo <= 1:
                return None
            mid = lo + (hi - lo) // 2
            return (Chunk(self.key, lo, mid, self.index, self.depth + 1),
                    Chunk(self.key, mid, hi, self.index, self.depth + 1))
        lo, hi = float(self.lo), float(self.hi)
        if not (hi > lo) or (hi - lo) < 1e-9:
            return None
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            return None
        return (Chunk(self.key, lo, mid, self.index, self.depth + 1),
                Chunk(self.key, mid, hi, self.index, self.depth + 1))


@dataclass
class ChunkResult:
    """One chunk's rows plus the bookkeeping that proves it is complete."""

    chunk: Chunk
    rows: list[dict] = field(default_factory=list)
    verdict: str = "OK"
    truncated: bool = False
    calls: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class GaiaSSO:
    """Gaia SSO astrometry over the public ESA TAP service.  No credentials.

    Composition rather than inheritance, for the same reason
    :class:`seti.loom.acquire.AlerceSSO` does it:
    :class:`~seti.tocsin.brokers.AlerceTAP` already owns the session with a real
    per-request socket timeout, the exponential-backoff retry and the fail-fast on
    ADQL errors.  Each of those was added after an unattended job lost hours to the
    failure it prevents, and re-deriving them here would mean two places for the
    same bug to come back.  What this class adds is the part that is specific to
    this service: an **asynchronous** query path (the synchronous one is believed
    to be capped at a handful of rows — see the module docstring) and an overflow
    check on every result.
    """

    def __init__(self, url: str = GAIA_TAP, timeout: float = 1800.0,
                 maxrec: int = 2_000_000):
        self.url = url
        self.maxrec = int(maxrec)
        self.tap = AlerceTAP(url=url, timeout=timeout, maxrec=maxrec)
        self.last_truncated = False

    @property
    def calls(self) -> int:
        return self.tap.calls

    # -- the two query paths ------------------------------------------------
    def query_sync(self, adql: str, maxrec: int | None = None, retries: int = 3):
        """A synchronous query.  Only for tiny metadata reads.

        Public because the probe has to *measure* the synchronous row cap, which
        it cannot do without issuing one.  Nothing that pulls data should use it:
        if the cap is real, the truncation is silent.
        """
        rows = self.tap.query(adql, maxrec=maxrec, retries=retries)
        return [{k: _scalar(v) for k, v in r.items()} for r in rows]

    def fetch(self, adql: str, maxrec: int | None = None,
              retries: int = 4) -> tuple[list[dict], bool]:
        """Run one asynchronous ADQL query; return ``(rows, truncated)``.

        The retry loop mirrors :meth:`seti.tocsin.brokers.AlerceTAP.query` clause
        for clause — backoff on transport errors, immediate raise on anything that
        looks like an ADQL error — and reuses that object's ``pyvo`` service so the
        timeout-injecting session is the same one.  Building a second service here
        would create a second session *without* the per-request deadline, which is
        the precise bug ``brokers.py`` documents having paid for.

        ``truncated`` is ``True`` when the service returned as many rows as were
        asked for.  That is deliberately conservative: a query whose true answer is
        exactly ``maxrec`` rows is reported as possibly-truncated, because the cost
        of one redundant sub-query is nothing and the cost of an unnoticed
        truncation is a silent partial catalogue.
        """
        svc = self.tap._service()
        want = int(maxrec or self.maxrec)
        last = None
        for attempt in range(retries):
            self.tap.calls += 1
            try:
                res = svc.run_async(adql, maxrec=want)
            except Exception as exc:                          # noqa: BLE001
                last = str(exc)
                if _fatal_adql(last):
                    raise BrokerError(
                        f"ADQL rejected: {last[:600]}\nquery: {adql}") from exc
                time.sleep(2.0 ** attempt)
                continue
            try:
                tab = res.to_table()
            except Exception as exc:                          # noqa: BLE001
                raise BrokerError(f"TAP result unreadable: {exc}") from exc
            rows = [{c: _scalar(row[c]) for c in tab.colnames} for row in tab]
            self.last_truncated = len(rows) >= want
            return rows, self.last_truncated
        raise BrokerError(f"TAP query failed after {retries} attempts: {last}")

    def query(self, adql: str, maxrec: int | None = None,
              retries: int = 4) -> list[dict]:
        """:meth:`fetch` without the overflow flag; sets ``last_truncated``."""
        rows, _ = self.fetch(adql, maxrec=maxrec, retries=retries)
        return rows

    # -- schema discovery ---------------------------------------------------
    def describe(self, tables: tuple[str, ...] = (
            DR3_OBSERVATION, FPR_OBSERVATION, DR3_SOURCE, FPR_SOURCE)) -> dict:
        """The live ``TAP_SCHEMA`` column list, per table, with units and UCDs.

        Committed verbatim by the probe so that a schema change shows up as a
        **diff in version control** rather than as an unexplained null months
        later.  ``unit``, ``ucd`` and ``description`` are pulled alongside the
        names deliberately: several of the open questions in
        :data:`OPEN_QUESTIONS` — the epoch's time scale, whether ``ra_error_*``
        carries ``cos(dec)``, the frame of the state vectors — may simply be
        *answered* by the archive's own column metadata, and if they are, the
        answer is in the committed record rather than in someone's memory.
        """
        names = ", ".join(_quote(t) for t in tables)
        rows = self.query(
            "SELECT table_name, column_name, datatype, unit, ucd, description "
            f"FROM TAP_SCHEMA.columns WHERE table_name IN ({names}) "
            "ORDER BY table_name, column_name", maxrec=20000)
        out: dict[str, list[dict]] = {t: [] for t in tables}
        for r in rows:
            t = str(r.get("table_name", ""))
            out.setdefault(t, []).append({
                "name": str(r.get("column_name", "")),
                "datatype": r.get("datatype"),
                "unit": r.get("unit"),
                "ucd": r.get("ucd"),
                "description": (str(r.get("description"))[:400]
                                if r.get("description") is not None else None),
            })
        return out

    def describe_names(self, tables=None) -> dict[str, list[str]]:
        """Just the column names, for callers that only need the intersection."""
        full = self.describe(tables) if tables else self.describe()
        return {t: [c["name"] for c in cols] for t, cols in full.items()}

    def available_columns(self, table: str) -> list[str]:
        """The columns a table actually has, lower-cased, from ``TAP_SCHEMA``.

        Used to resolve a wanted column list against the live schema instead of
        SELECTing a guessed name and taking the whole query down with an ADQL
        error — the discipline LOOM arrived at after ``lsst_ss_object``'s spellings
        turned out not to be the ones the science model documents.
        """
        rows = self.query(
            "SELECT column_name FROM TAP_SCHEMA.columns "
            f"WHERE table_name = {_quote(table)}", maxrec=2000, retries=2)
        return [str(r.get("column_name", "")).lower() for r in rows]

    def resolve_columns(self, table: str, wanted: tuple[str, ...] | None = None
                        ) -> tuple[list[str], list[str]]:
        """``(present, missing)`` for a wanted list against the live schema."""
        release = "gaiadr3" if table.startswith("gaiadr3") else "gaiafpr"
        want = [c.lower() for c in (wanted or ALL_COLUMNS.get(release, SHARED_COLUMNS))]
        have = set(self.available_columns(table))
        return [c for c in want if c in have], [c for c in want if c not in have]

    # -- counts and coverage -------------------------------------------------
    def count(self, table: str, where: str = "") -> int | None:
        adql = f"SELECT COUNT(*) AS n FROM {table}"
        if where:
            adql += f" WHERE {where}"
        rows = self.query(adql, maxrec=5, retries=2)
        if not rows:
            return None
        try:
            return int(rows[0].get("n"))
        except (TypeError, ValueError):
            return None

    def epoch_range(self, table: str) -> dict:
        """``MIN``/``MAX`` of ``epoch`` and ``epoch_utc`` plus their difference.

        The difference is the interesting one: if ``epoch_utc - epoch`` is a
        constant of order 2455197.5 then ``epoch_utc`` is a Julian Date and
        ``epoch`` counts days from 2010-01-01, which settles the zero point
        without needing documentation.  If it *drifts*, the two differ by a time
        scale (TCB against UTC, whose difference grows), and that drift is itself
        the measurement.  :func:`interpret_epoch_zero_point` reads the numbers.
        """
        rows = self.query(
            "SELECT MIN(epoch) AS epoch_min, MAX(epoch) AS epoch_max, "
            "MIN(epoch_utc) AS utc_min, MAX(epoch_utc) AS utc_max, "
            "MIN(epoch_utc - epoch) AS diff_min, MAX(epoch_utc - epoch) AS diff_max, "
            "MIN(epoch_err) AS err_min, MAX(epoch_err) AS err_max, "
            f"AVG(epoch_err) AS err_mean FROM {table}", maxrec=5, retries=2)
        return rows[0] if rows else {}

    def object_numbers(self, release: str = "gaiafpr",
                       maxrec: int | None = None) -> SSOResult:
        """Every ``number_mp`` in the release's source catalogue, sorted.

        This is what the chunk plan is built from.  It is one query of ~157k rows,
        which is nothing, and it makes the chunk boundaries *exact* — each chunk
        holds a known number of objects — instead of guessing a stride in an index
        whose density is wildly non-uniform (MPC numbers run past 600,000 while
        Gaia's sample is concentrated in the low-numbered, bright end).
        """
        table = SOURCE_TABLE.get(release, FPR_SOURCE)
        res = SSOResult()
        res.adql = (f"SELECT number_mp, denomination FROM {table} "
                    "ORDER BY number_mp")
        try:
            res.rows, res.truncated = self.fetch(res.adql, maxrec=maxrec or 400_000)
            res.reached = True
            res.verdict = "OK" if res.rows else "EMPTY"
            if res.truncated:
                res.verdict = "TRUNCATED"
                res.notes.append(
                    "the object catalogue came back at exactly the row limit, so "
                    "it is incomplete and the chunk plan built from it would skip "
                    "objects silently")
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"object catalogue query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    def source_catalogue(self, release: str = "gaiafpr",
                         columns: tuple[str, ...] | None = None,
                         include_arrays: bool = False,
                         maxrec: int | None = None) -> SSOResult:
        """The per-object rows of ``sso_source``, resolved against the live schema.

        **The state vectors and orbital elements in this table come from Gaia's own
        orbit solution, which was fitted to the very observations SEXTANT is about
        to take residuals of.**  A residual computed against them is minimised by
        construction and carries no information whatsoever
        (``docs/substitute-surveys.md``).  They are pulled because the *comparison*
        between Gaia's solution and an independent JPL solution is itself
        diagnostic — but the reference orbit must be the independent one, always.

        ``include_arrays`` is off by default because the covariance matrices are
        array-valued and blow up both the transfer and the JSON record.
        """
        table = SOURCE_TABLE.get(release, FPR_SOURCE)
        res = SSOResult()
        try:
            have = set(self.available_columns(table))
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"sso_source schema unavailable: {exc}"[:400])
            return res
        if not have:
            res.verdict = "TABLE_ABSENT"
            res.notes.append(f"{table} has no columns in TAP_SCHEMA")
            return res
        if columns:
            cols = [c for c in (x.lower() for x in columns) if c in have]
            missing = [c for c in (x.lower() for x in columns) if c not in have]
            if missing:
                res.notes.append(f"absent from {table}: {missing[:12]}")
        else:
            skip = () if include_arrays else (
                "epoch_state_vector", "h_state_vector",
                "h_state_vector_var_covar_matrix",
                "orbital_elements_var_covar_matrix")
            cols = sorted(c for c in have if c not in skip)
        if not cols:
            res.verdict = "NO_EXPECTED_COLUMNS"
            return res
        res.adql = f"SELECT {', '.join(cols)} FROM {table}"
        try:
            res.rows, res.truncated = self.fetch(res.adql, maxrec=maxrec or 400_000)
            res.reached = True
            res.verdict = "TRUNCATED" if res.truncated else (
                "OK" if res.rows else "EMPTY")
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"sso_source query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    # -- the bulk pulls ------------------------------------------------------
    def _observation_select(self, release: str,
                            columns: tuple[str, ...] | None = None) -> str:
        cols = columns or ALL_COLUMNS.get(release, SHARED_COLUMNS)
        return ", ".join(cols)

    def observations_for_objects(self, numbers=None, denominations=None,
                                 release: str = "gaiafpr",
                                 columns: tuple[str, ...] | None = None,
                                 epoch_lo: float | None = None,
                                 epoch_hi: float | None = None,
                                 extra_where: str = "",
                                 maxrec: int | None = None) -> SSOResult:
        """Every observation of the named objects, with the full column set.

        The per-object path.  ``numbers`` selects on ``number_mp`` and
        ``denominations`` on ``denomination``; passing both ORs them, because an
        object present under one identifier and absent under the other is exactly
        what the ``number_mp_completeness`` open question is about, and an AND
        would silently return nothing for such an object.

        ``epoch_lo``/``epoch_hi`` are here so that a single object whose row count
        somehow exceeds the service's limit can still be pulled in time slices —
        the escape hatch :meth:`iter_observation_chunks` names when a chunk cannot
        be split any further.
        """
        res = SSOResult()
        if not numbers and not denominations:
            res.verdict = "NO_TARGETS"
            res.notes.append("neither numbers nor denominations were given; "
                             "refusing to pull the whole table by accident")
            return res
        table = OBSERVATION_TABLE.get(release, FPR_OBSERVATION)
        ident: list[str] = []
        if numbers:
            ident.append(f"number_mp IN ({_int_list(numbers)})")
        if denominations:
            ident.append(f"denomination IN ({_str_list(denominations)})")
        where = ["(" + " OR ".join(ident) + ")"]
        if epoch_lo is not None:
            where.append(f"epoch >= {float(epoch_lo)!r}")
        if epoch_hi is not None:
            where.append(f"epoch < {float(epoch_hi)!r}")
        if extra_where:
            where.append(f"({extra_where})")
        res.adql = (f"SELECT {self._observation_select(release, columns)} "
                    f"FROM {table} WHERE " + " AND ".join(where))
        try:
            res.rows, res.truncated = self.fetch(res.adql, maxrec=maxrec)
            res.reached = True
            res.verdict = "TRUNCATED" if res.truncated else (
                "OK" if res.rows else "EMPTY")
            if res.truncated:
                res.notes.append(
                    "row limit reached: this object list is INCOMPLETE. Split it "
                    "or slice it with epoch_lo/epoch_hi before using it.")
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"observation query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    def observations_in_epoch_range(self, epoch_lo: float, epoch_hi: float,
                                    release: str = "gaiafpr",
                                    columns: tuple[str, ...] | None = None,
                                    extra_where: str = "",
                                    maxrec: int | None = None) -> SSOResult:
        """Every observation in ``[epoch_lo, epoch_hi)``, with the full column set.

        The time-sliced path.  **An epoch slice truncates every arc it crosses**,
        so what comes back is not a set of objects, it is a set of fragments; the
        residual fits cannot be run on it until it has been reassembled with the
        neighbouring slices.  It is the right shape for coverage questions ("how
        much of the mission does this release cover?"), for cross-release
        comparisons, and for restarting a partial download, and the wrong shape for
        the science.  Hence the default chunk key is the object, not the epoch.
        """
        table = OBSERVATION_TABLE.get(release, FPR_OBSERVATION)
        res = SSOResult()
        where = [f"epoch >= {float(epoch_lo)!r}", f"epoch < {float(epoch_hi)!r}"]
        if extra_where:
            where.append(f"({extra_where})")
        res.adql = (f"SELECT {self._observation_select(release, columns)} "
                    f"FROM {table} WHERE " + " AND ".join(where))
        try:
            res.rows, res.truncated = self.fetch(res.adql, maxrec=maxrec)
            res.reached = True
            res.verdict = "TRUNCATED" if res.truncated else (
                "OK" if res.rows else "EMPTY")
            if res.truncated:
                res.notes.append("row limit reached: this epoch slice is INCOMPLETE")
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"epoch-slice query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    def rejection_census(self, numbers=None, release: str = "gaiafpr",
                         flag_column: str = "astrometric_outcome_ccd",
                         min_number: int | None = None,
                         max_number: int | None = None,
                         maxrec: int | None = None) -> SSOResult:
        """Per-object attempts and outcomes, server-side.  The screen's denominator.

        **This is the channel's primary pull, and it is one ``GROUP BY``.**  The
        rejection screen asks whether some objects are systematically un-fittable,
        which is a *rate*: how many observations of this object were written, and
        how many of them failed.  Counting that by downloading 46 million rows
        would be absurd when the service will group them, so this returns one row
        per ``(number_mp, flag value)`` pair — of order 157k objects times a
        handful of codes — and :func:`rejection_ledger` folds it into a per-object
        ledger client-side.

        Summing the counts over the flag values for an object **is** the
        denominator, which is why the query groups rather than filters: a query
        that selected only the failures would return a numerator with no
        denominator, and a rate cannot be recovered from that afterwards.  Nothing
        here narrows on the flag at all.

        ``flag_column`` chooses which axis to break down by; call it once per
        column (``astrometric_outcome_ccd``, ``astrometric_outcome_transit``, and
        ``is_rejected`` on FPR) and merge the ledgers.  ``min_number``/
        ``max_number`` slice the object range so the census can be chunked exactly
        like an observation pull.

        **The denominator's own limit, stated because a rate is only as good as
        it:** this counts observations that entered ``sso_observation``.  If Gaia's
        pipeline can drop a transit so completely that no row is written, then the
        denominator is censored in the *same direction* as the signal — an object
        that is more un-fittable would look like an object that was simply observed
        less, and the screen would be measuring its own selection function.  That
        is ``OPEN_QUESTIONS['rejection_denominator']``, and the cross-check is the
        per-object row count against whatever observation count ``sso_source``
        carries.
        """
        table = OBSERVATION_TABLE.get(release, FPR_OBSERVATION)
        res = SSOResult()
        where: list[str] = []
        if numbers:
            where.append(f"number_mp IN ({_int_list(numbers)})")
        if min_number is not None:
            where.append(f"number_mp >= {int(min_number)}")
        if max_number is not None:
            where.append(f"number_mp < {int(max_number)}")
        res.adql = (f"SELECT number_mp, {flag_column} AS flag_value, "
                    f"COUNT(*) AS n FROM {table}")
        if where:
            res.adql += " WHERE " + " AND ".join(where)
        res.adql += f" GROUP BY number_mp, {flag_column}"
        try:
            res.rows, res.truncated = self.fetch(res.adql, maxrec=maxrec)
            res.reached = True
            res.verdict = "TRUNCATED" if res.truncated else (
                "OK" if res.rows else "EMPTY")
            for r in res.rows:
                r["flag_column"] = flag_column
            if res.truncated:
                res.notes.append(
                    "row limit reached: this census is INCOMPLETE, and an "
                    "incomplete denominator makes every rate built on it wrong "
                    "rather than noisy. Slice it with min_number/max_number.")
        except BrokerError as exc:
            res.verdict = "NO_DATA_REACHED"
            res.notes.append(f"rejection census query failed: {exc}"[:500])
        res.calls = self.tap.calls
        return res

    def iter_observation_chunks(self, chunks, release: str = "gaiafpr",
                                columns: tuple[str, ...] | None = None,
                                extra_where: str = "",
                                maxrec: int | None = None,
                                max_depth: int = 12,
                                on_chunk=None):
        """Walk a chunk plan, splitting any chunk the service truncates.

        **The chunk key is ``number_mp``, and the reason is that a residual is a
        time series.**  Every fit the next stage runs — the drift fit, the
        apparition trend, the heliocentric-distance law discrimination — needs an
        object's *whole* arc at once.  Chunking on the object hands back complete,
        immediately usable objects and lets a run be stopped and resumed at any
        chunk boundary with everything already downloaded still valid.  Chunking on
        ``epoch`` hands back fragments of every object alive at that time, so
        nothing at all can be fitted until the entire 46-million-row download has
        completed and been regrouped — which turns a resumable job into an
        all-or-nothing one, on a runner with a hard job timeout, for a table that
        cannot be pulled in one sitting.  :meth:`observations_in_epoch_range` and
        :func:`plan_epoch_chunks` exist for the questions that really are about
        time, and say what they cost.

        The other half of the design is that **a chunk plan cannot be trusted to be
        correctly sized**, because the density of Gaia's sample in ``number_mp`` is
        extremely non-uniform.  So every chunk's result is checked for overflow, and
        an overflowing chunk is bisected and retried rather than accepted: a chunk
        that hits ``maxrec`` is short, not empty, and nothing downstream could tell.
        Bisection stops at ``max_depth`` or when a single ``number_mp`` still
        overflows, and yields ``IRREDUCIBLE_OVERFLOW`` — at which point the caller
        pulls that one object in epoch slices via
        :meth:`observations_for_objects`.
        """
        pending = list(chunks)
        while pending:
            chunk = pending.pop(0)
            table = OBSERVATION_TABLE.get(release, FPR_OBSERVATION)
            where = [chunk.where()]
            if extra_where:
                where.append(f"({extra_where})")
            adql = (f"SELECT {self._observation_select(release, columns)} "
                    f"FROM {table} WHERE " + " AND ".join(where))
            out = ChunkResult(chunk=chunk)
            try:
                rows, truncated = self.fetch(adql, maxrec=maxrec)
            except BrokerError as exc:
                out.verdict = "NO_DATA_REACHED"
                out.notes.append(f"chunk query failed: {exc}"[:500])
                out.calls = self.tap.calls
                if on_chunk is not None:
                    on_chunk(out)
                yield out
                continue
            if truncated:
                halves = chunk.halves() if chunk.depth < max_depth else None
                if halves is None:
                    out.rows, out.truncated, out.verdict = rows, True, "IRREDUCIBLE_OVERFLOW"
                    out.notes.append(
                        "this chunk overflows and cannot be split further; re-pull "
                        "it with observations_for_objects(epoch_lo=..., "
                        "epoch_hi=...) in time slices")
                    out.calls = self.tap.calls
                    if on_chunk is not None:
                        on_chunk(out)
                    yield out
                    continue
                out.verdict = "SPLIT"
                out.truncated = True
                out.notes.append(f"overflowed at maxrec; split into "
                                 f"[{halves[0].lo}, {halves[0].hi}) and "
                                 f"[{halves[1].lo}, {halves[1].hi})")
                pending[:0] = list(halves)
                out.calls = self.tap.calls
                if on_chunk is not None:
                    on_chunk(out)
                yield out
                continue
            out.rows = rows
            out.verdict = "OK" if rows else "EMPTY"
            out.calls = self.tap.calls
            if on_chunk is not None:
                on_chunk(out)
            yield out

    # -- the measurements that settle the open questions ---------------------
    def service_limits(self) -> dict:
        """The service's declared TAP output limits, fetched verbatim.

        A capabilities document is cheap and authoritative, whereas measuring the
        asynchronous cap by requesting three million rows is neither.  The raw XML
        is recorded so a human can read what the parser missed.
        """
        out: dict = {"url": self.url.rstrip("/") + "/capabilities"}
        try:
            session = self.tap._service()._session          # noqa: SLF001
            resp = session.get(out["url"])
            text = resp.text
            out["status"] = int(resp.status_code)
            out["raw"] = text[:20000]
            out["output_limits"] = {
                "default": [int(x) for x in re.findall(
                    r"<default[^>]*>\s*(\d+)\s*</default>", text)][:8],
                "hard": [int(x) for x in re.findall(
                    r"<hard[^>]*>\s*(\d+)\s*</hard>", text)][:8],
            }
        except Exception as exc:                              # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return out

    def diagnostics(self, on_result=None, sample_rows: int = 20,
                    probe_numbers: tuple[int, ...] = (1, 2, 3, 4, 21, 433, 704),
                    ) -> dict:
        """Every open question, asked, with each query's error captured.

        Same discipline as ``loom.acquire.diagnostics``: each query runs on its own
        and its failure is recorded rather than raised, because the point is to come
        back with a **complete** picture in one runner pass rather than to iterate
        blind against a service that takes minutes per query.  ``on_result`` is
        called after each one so a job timeout part-way through still leaves every
        answer already obtained on disk.
        """
        out: dict = {}

        def run(name: str, adql: str, maxrec: int = 20, sync: bool = False) -> None:
            entry: dict = {"adql": adql, "sync": sync}
            try:
                if sync:
                    rows = self.query_sync(adql, maxrec=maxrec, retries=2)
                    truncated = len(rows) >= maxrec
                else:
                    rows, truncated = self.fetch(adql, maxrec=maxrec, retries=2)
                entry["rows"] = len(rows)
                entry["truncated"] = bool(truncated)
                entry["data"] = rows[:maxrec]
            except Exception as exc:                          # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"[:600]
            out[name] = entry
            if on_result is not None:
                on_result(name, entry)

        # --- sync_row_cap.  THE dangerous one.  Ask synchronously for far more
        # rows than the quoted 2000 cap and count what comes back; whatever number
        # returns IS the cap, and if it is 5000 there is no cap at this size.
        run("sync_row_cap",
            f"SELECT observation_id, epoch FROM {FPR_OBSERVATION}",
            maxrec=5000, sync=True)

        # --- row and object counts, so a later run can see the archive move.
        for table in (FPR_OBSERVATION, DR3_OBSERVATION, FPR_SOURCE, DR3_SOURCE):
            run(f"count_{table.replace('.', '_')}",
                f"SELECT COUNT(*) AS n FROM {table}", maxrec=5)

        # --- epoch_zero_point.  MIN/MAX of both time columns and of their
        # difference: a constant difference near 2455197.5 identifies the zero
        # point outright, a drifting one identifies a time-scale difference.
        for table in (FPR_OBSERVATION, DR3_OBSERVATION):
            run(f"epoch_ranges_{table.replace('.', '_')}",
                "SELECT MIN(epoch) AS epoch_min, MAX(epoch) AS epoch_max, "
                "MIN(epoch_utc) AS utc_min, MAX(epoch_utc) AS utc_max, "
                "MIN(epoch_utc - epoch) AS diff_min, "
                "MAX(epoch_utc - epoch) AS diff_max, "
                "MIN(epoch_err) AS err_min, MAX(epoch_err) AS err_max, "
                f"AVG(epoch_err) AS err_mean FROM {table}", maxrec=5)
        _epoch_key = f"epoch_ranges_{FPR_OBSERVATION.replace('.', '_')}"
        _epoch_data = (out.get(_epoch_key, {}).get("data") or [{}])[0]
        out["epoch_ranges"] = {
            "see": [k for k in out if k.startswith("epoch_ranges_")],
            "interpretation": interpret_epoch_zero_point(_epoch_data),
        }
        if on_result is not None:
            on_result("epoch_ranges", out["epoch_ranges"])

        # --- state_vector_frame.  Gaia orbits L2: in an ECLIPTIC frame its
        # barycentric z stays within a few 1e-3 AU of zero; in an EQUATORIAL frame
        # z = y_ecl*sin(23.44 deg) sweeps +/-0.4 AU annually.  MIN/MAX separates
        # the two by two orders of magnitude and costs one aggregate query.
        for table in (FPR_OBSERVATION, DR3_OBSERVATION):
            run(f"state_vector_frame_{table.replace('.', '_')}",
                "SELECT MIN(x_gaia) AS x_min, MAX(x_gaia) AS x_max, "
                "MIN(y_gaia) AS y_min, MAX(y_gaia) AS y_max, "
                "MIN(z_gaia) AS z_min, MAX(z_gaia) AS z_max, "
                "MIN(z_gaia_geocentric) AS zg_min, MAX(z_gaia_geocentric) AS zg_max, "
                "MIN(x_gaia_geocentric) AS xg_min, MAX(x_gaia_geocentric) AS xg_max "
                f"FROM {table}", maxrec=5)
        _frame_key = f"state_vector_frame_{FPR_OBSERVATION.replace('.', '_')}"
        _frame_data = (out.get(_frame_key, {}).get("data") or [{}])[0]
        out["state_vector_frame"] = {
            "see": [k for k in out if k.startswith("state_vector_frame_")],
            "interpretation": interpret_state_vector_frame(
                _frame_data.get("z_min"), _frame_data.get("z_max")),
        }
        if on_result is not None:
            on_result("state_vector_frame", out["state_vector_frame"])

        # --- position_angle_scan_convention.  The query cannot settle North-through
        # -East (that needs the data model), but it settles the wrap, which is the
        # half that silently breaks arithmetic.
        for table in (FPR_OBSERVATION, DR3_OBSERVATION):
            run(f"position_angle_scan_range_{table.replace('.', '_')}",
                "SELECT MIN(position_angle_scan) AS pa_min, "
                "MAX(position_angle_scan) AS pa_max, "
                "AVG(position_angle_scan) AS pa_mean, "
                "COUNT(position_angle_scan) AS n_nonnull, COUNT(*) AS n_total "
                f"FROM {table}", maxrec=5)
        out["position_angle_scan_range"] = {
            "see": [k for k in out if k.startswith("position_angle_scan_range_")]}
        if on_result is not None:
            on_result("position_angle_scan_range", out["position_angle_scan_range"])

        # --- outcome_flag_meanings and is_rejected_semantics.  The value
        # DISTRIBUTION is the measurement: a flag whose modal value covers 99% of
        # rows is a failure code, and one that splits the table is something else.
        for table in (FPR_OBSERVATION, DR3_OBSERVATION):
            for col in ("astrometric_outcome_ccd", "astrometric_outcome_transit"):
                run(f"flag_{col}_{table.replace('.', '_')}",
                    f"SELECT {col} AS value, COUNT(*) AS n FROM {table} "
                    f"GROUP BY {col} ORDER BY COUNT(*) DESC", maxrec=200)
        run(f"flag_is_rejected_{FPR_OBSERVATION.replace('.', '_')}",
            "SELECT is_rejected AS value, COUNT(*) AS n "
            f"FROM {FPR_OBSERVATION} GROUP BY is_rejected", maxrec=50)
        run("flag_is_rejected_vs_outcome",
            "SELECT is_rejected, astrometric_outcome_ccd, COUNT(*) AS n "
            f"FROM {FPR_OBSERVATION} "
            "GROUP BY is_rejected, astrometric_outcome_ccd "
            "ORDER BY COUNT(*) DESC", maxrec=200)
        run(f"flag_fov_{FPR_OBSERVATION.replace('.', '_')}",
            f"SELECT fov AS value, COUNT(*) AS n FROM {FPR_OBSERVATION} "
            "GROUP BY fov", maxrec=50)
        out["quality_flag_distributions"] = {
            "see": [k for k in out if k.startswith("flag_")]}
        if on_result is not None:
            on_result("quality_flag_distributions", out["quality_flag_distributions"])

        # --- rejection_fraction.  THE calibration for the new channel: does
        # is_rejected mark the documented ~0.58% (DR3) / ~1% (DR2) astrometric
        # outlier rejection, or something else?  Two COUNT(*) queries rather than
        # a SUM(CASE WHEN ...), because a CASE literal makes some TAP services
        # infer a 16-bit type and then fail VOTable serialisation for any count
        # above 32767 -- which silently lost the non-zero count for exactly the
        # columns that had one, in LOOM's first probe.
        run("rejection_fraction_total",
            f"SELECT COUNT(*) AS n FROM {FPR_OBSERVATION}", maxrec=5)
        run("rejection_fraction_rejected",
            f"SELECT COUNT(*) AS n FROM {FPR_OBSERVATION} WHERE is_rejected = 'true'",
            maxrec=5)
        # The literal above assumes a string-typed column; if it is boolean or
        # integer the query is rejected and this second form answers instead. Both
        # are recorded, and whichever succeeded also SETTLES the column's type.
        run("rejection_fraction_rejected_numeric",
            f"SELECT COUNT(*) AS n FROM {FPR_OBSERVATION} WHERE is_rejected = 1",
            maxrec=5)
        out["rejection_fraction"] = {
            "see": [k for k in out if k.startswith("rejection_fraction")],
            "published_reference": PUBLISHED_OUTLIER_FRACTION,
        }
        if on_result is not None:
            on_result("rejection_fraction", out["rejection_fraction"])

        # --- rejection_census_sample.  One GROUP BY per flag column over a handful
        # of objects: the exact shape of the channel's primary pull, run small so
        # its cost and its row count are known before it is run over 157k objects.
        # It also carries the DENOMINATOR cross-check: the per-object attempt count
        # here can be compared against whatever observation count sso_source
        # carries, which is what settles OPEN_QUESTIONS['rejection_denominator'].
        for col in ("astrometric_outcome_ccd", "astrometric_outcome_transit",
                    "is_rejected"):
            run(f"rejection_census_{col}",
                f"SELECT number_mp, {col} AS flag_value, COUNT(*) AS n "
                f"FROM {FPR_OBSERVATION} "
                f"WHERE number_mp IN ({_int_list(probe_numbers)}) "
                f"GROUP BY number_mp, {col}", maxrec=200)
        run("rejection_census_astrometric_outcome_ccd_dr3",
            "SELECT number_mp, astrometric_outcome_ccd AS flag_value, "
            f"COUNT(*) AS n FROM {DR3_OBSERVATION} "
            f"WHERE number_mp IN ({_int_list(probe_numbers)}) "
            "GROUP BY number_mp, astrometric_outcome_ccd", maxrec=200)
        out["rejection_census_sample"] = {
            "see": [k for k in out if k.startswith("rejection_census_")]}
        if on_result is not None:
            on_result("rejection_census_sample", out["rejection_census_sample"])

        # --- rows_per_transit.  Restricted to a handful of low-numbered objects so
        # it is a cheap GROUP BY rather than a scan of 46M rows; the multiplicity
        # is a property of the instrument, not of the object, so a sample answers it.
        run("rows_per_transit",
            "SELECT transit_id, COUNT(*) AS n_rows "
            f"FROM {FPR_OBSERVATION} "
            f"WHERE number_mp IN ({_int_list(probe_numbers)}) "
            "GROUP BY transit_id ORDER BY COUNT(*) DESC", maxrec=400)
        run("rows_per_transit_dr3",
            "SELECT transit_id, COUNT(*) AS n_rows "
            f"FROM {DR3_OBSERVATION} "
            f"WHERE number_mp IN ({_int_list(probe_numbers)}) "
            "GROUP BY transit_id ORDER BY COUNT(*) DESC", maxrec=400)

        # --- number_mp_completeness.  A NULL number_mp would be silently skipped by
        # every number_mp chunk, and the objects without a number are the least
        # ordinary ones in the catalogue.
        for table in (FPR_OBSERVATION, DR3_OBSERVATION):
            run(f"number_mp_completeness_{table.replace('.', '_')}",
                "SELECT COUNT(*) AS n_total, COUNT(number_mp) AS n_with_number, "
                "MIN(number_mp) AS number_min, MAX(number_mp) AS number_max, "
                f"COUNT(denomination) AS n_with_denomination FROM {table}", maxrec=5)
        run("number_mp_null_sample",
            f"SELECT number_mp, denomination, source_id FROM {FPR_OBSERVATION} "
            "WHERE number_mp IS NULL", maxrec=20)
        out["number_mp_completeness"] = {
            "see": [k for k in out if k.startswith("number_mp_")]}
        if on_result is not None:
            on_result("number_mp_completeness", out["number_mp_completeness"])

        # --- ra_error_cosdec.  If the column is a raw sigma(alpha) rather than a
        # true angle, the ratio ra_error/dec_error grows as 1/cos(dec); binning in
        # |dec| exposes that and nothing else does.
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 90)):
            run(f"ra_error_vs_dec_{lo}_{hi}",
                "SELECT AVG(ra_error_random) AS ra_err, "
                "AVG(dec_error_random) AS dec_err, COUNT(*) AS n "
                f"FROM {FPR_OBSERVATION} "
                f"WHERE ABS(dec) >= {lo} AND ABS(dec) < {hi}", maxrec=5)
        out["ra_error_vs_dec"] = {
            "see": [k for k in out if k.startswith("ra_error_vs_dec_")]}
        if on_result is not None:
            on_result("ra_error_vs_dec", out["ra_error_vs_dec"])

        # --- release_overlap and observation_id_stability.  THE reconciliation
        # question.  Three independent readings, because any one of them can be
        # explained away and together they cannot:
        #   * do the two releases hold the same epochs? (covered by epoch_ranges)
        #   * for a fixed object, how many rows does each hold, and do the
        #     observation_ids intersect at all?
        #   * does the service even accept a cross-schema join?
        run("release_overlap_counts_per_object",
            "SELECT number_mp, COUNT(*) AS n FROM " + FPR_OBSERVATION +
            f" WHERE number_mp IN ({_int_list(probe_numbers)}) GROUP BY number_mp",
            maxrec=50)
        run("release_overlap_counts_per_object_dr3",
            "SELECT number_mp, COUNT(*) AS n FROM " + DR3_OBSERVATION +
            f" WHERE number_mp IN ({_int_list(probe_numbers)}) GROUP BY number_mp",
            maxrec=50)
        run("release_overlap_join",
            "SELECT COUNT(*) AS n_matched FROM "
            f"{FPR_OBSERVATION} AS f JOIN {DR3_OBSERVATION} AS d "
            "ON f.observation_id = d.observation_id "
            f"WHERE f.number_mp IN ({_int_list(probe_numbers[:3])})", maxrec=5)
        run("release_overlap_join_transit",
            "SELECT COUNT(*) AS n_matched FROM "
            f"{FPR_OBSERVATION} AS f JOIN {DR3_OBSERVATION} AS d "
            "ON f.transit_id = d.transit_id AND f.number_mp = d.number_mp "
            f"WHERE f.number_mp IN ({_int_list(probe_numbers[:3])})", maxrec=5)
        # The verbatim ids for one object from each release, so that if the joins
        # are refused a human can still see by eye whether the ids are the same
        # objects re-minted or the same integers.
        for table, tag in ((FPR_OBSERVATION, "fpr"), (DR3_OBSERVATION, "dr3")):
            run(f"release_overlap_ids_{tag}",
                "SELECT observation_id, transit_id, source_id, solution_id, "
                f"epoch, epoch_utc FROM {table} "
                f"WHERE number_mp = {int(probe_numbers[0])} ORDER BY epoch",
                maxrec=30)
        out["release_overlap"] = {
            "see": [k for k in out if k.startswith("release_overlap")]}
        if on_result is not None:
            on_result("release_overlap", out["release_overlap"])

        # --- a real slice, verbatim, with EVERY column.  TAP_SCHEMA can list a
        # column that no row populates; only rows settle that.
        for release, table in (("gaiafpr", FPR_OBSERVATION),
                               ("gaiadr3", DR3_OBSERVATION)):
            run(f"sample_rows_{release}",
                f"SELECT {self._observation_select(release)} FROM {table} "
                f"WHERE number_mp = {int(probe_numbers[0])} ORDER BY epoch",
                maxrec=sample_rows)
        for release, table in (("gaiafpr", FPR_SOURCE), ("gaiadr3", DR3_SOURCE)):
            run(f"sample_source_{release}",
                f"SELECT * FROM {table} "
                f"WHERE number_mp IN ({_int_list(probe_numbers[:3])})", maxrec=5)
        return out


# ---------------------------------------------------------------------------
# Chunk planning — pure, offline, and tested
# ---------------------------------------------------------------------------
def plan_object_chunks(numbers, target_objects: int = 500,
                       obs_per_object: float | None = None,
                       key: str = "number_mp") -> list[Chunk]:
    """Half-open ``number_mp`` ranges holding ``target_objects`` objects each.

    The boundaries come from the **actual sorted object list**, not from a stride
    in the index, because Gaia's sample is concentrated in the low-numbered bright
    asteroids while MPC numbers run past 600,000: a fixed stride would produce a
    first chunk with a hundred thousand objects in it and a last chunk with none.
    Taking boundaries from the list makes the object count per chunk exact and the
    row count per chunk predictable, which is what lets a runner size the pull.

    The ranges tile ``[min, max]`` with no gaps and no overlaps, and the last one
    extends one past the largest number so that ``< hi`` still includes it.  A
    caller that unions every chunk gets every row exactly once — the property that
    matters, since a gap loses observations silently and an overlap double-counts
    them into a smaller apparent scatter.
    """
    vals = sorted({int(n) for n in numbers if n is not None})
    if not vals:
        return []
    per = max(1, int(target_objects))
    rate = float(obs_per_object) if obs_per_object else MEAN_OBS_PER_OBJECT["gaiafpr"]
    chunks: list[Chunk] = []
    for i in range(0, len(vals), per):
        block = vals[i:i + per]
        lo = block[0]
        hi = vals[i + per] if (i + per) < len(vals) else block[-1] + 1
        chunks.append(Chunk(key=key, lo=lo, hi=hi, index=len(chunks),
                            expected_rows=len(block) * rate))
    return chunks


def plan_epoch_chunks(epoch_lo: float, epoch_hi: float, n_chunks: int = 0,
                      width_days: float = 0.0, key: str = "epoch") -> list[Chunk]:
    """Half-open epoch ranges tiling ``[epoch_lo, epoch_hi]``.

    For coverage and cross-release questions.  Remember what it costs: an epoch
    slice cuts every arc it crosses, so the pieces are not usable until they are
    reassembled — see :meth:`GaiaSSO.iter_observation_chunks` for why the object is
    the default key.  The final chunk's upper bound is nudged past ``epoch_hi`` so
    the last observation is not lost to the half-open interval.
    """
    lo, hi = float(epoch_lo), float(epoch_hi)
    if not (hi > lo):
        return []
    if width_days and width_days > 0:
        n = max(1, int(math.ceil((hi - lo) / float(width_days))))
    else:
        n = max(1, int(n_chunks or 1))
    step = (hi - lo) / n
    chunks: list[Chunk] = []
    for i in range(n):
        a = lo + i * step
        b = hi + 1e-6 if i == n - 1 else lo + (i + 1) * step
        chunks.append(Chunk(key=key, lo=a, hi=b, index=i))
    return chunks


def chunks_cover(chunks, values) -> dict:
    """Does this plan contain every value exactly once?  Offline audit.

    Cheap insurance against the two ways a chunk plan goes wrong in silence: a gap
    loses observations and looks like a sparser catalogue, an overlap double-counts
    them and looks like a smaller residual scatter.  Both survive every downstream
    test.
    """
    vals = [v for v in values if v is not None]
    counts = []
    for v in vals:
        counts.append(sum(1 for c in chunks if c.lo <= v < c.hi))
    n_missing = sum(1 for c in counts if c == 0)
    n_double = sum(1 for c in counts if c > 1)
    return {
        "n_values": len(vals),
        "n_missing": n_missing,
        "n_duplicated": n_double,
        "complete": n_missing == 0 and n_double == 0,
        "verdict": ("COVERS_EXACTLY_ONCE" if n_missing == 0 and n_double == 0
                    else "GAPS" if n_missing and not n_double
                    else "OVERLAPS" if n_double and not n_missing
                    else "GAPS_AND_OVERLAPS"),
    }


# ---------------------------------------------------------------------------
# Interpreting the probe's measurements — pure, offline, and tested
# ---------------------------------------------------------------------------
def interpret_sync_row_cap(n_returned: int, n_requested: int) -> dict:
    """Read the synchronous-cap measurement.

    Returning fewer rows than were asked for, with a table of 46 million rows
    behind the query, means the service imposed the limit.  A round number is the
    signature of a configured cap.
    """
    n_returned, n_requested = int(n_returned), int(n_requested)
    if n_returned >= n_requested:
        return {"verdict": "NO_CAP_AT_THIS_SIZE", "cap": None,
                "note": f"{n_requested} rows requested and returned; the "
                        f"synchronous cap, if any, is above {n_requested}"}
    return {"verdict": "SYNC_CAP_MEASURED", "cap": n_returned,
            "note": f"asked for {n_requested} rows from a table of "
                    f"{MEASURED_ROWS[FPR_OBSERVATION]:,} and got {n_returned}: "
                    f"the synchronous path truncates at {n_returned}. Every bulk "
                    f"pull must be asynchronous."}


def interpret_state_vector_frame(z_min, z_max, ecliptic_max_au: float = 0.05,
                                 equatorial_min_au: float = 0.20) -> dict:
    """Ecliptic or equatorial, from the range of Gaia's barycentric ``z``.

    Gaia sits at L2, on the Sun-Earth line, which lies in the ecliptic.  So in an
    **ecliptic** frame its barycentric ``z`` never leaves a thin slab about zero
    (the Lissajous amplitude, of order 1e-3 AU).  In an **equatorial** frame the
    same vector is rotated by the 23.44 deg obliquity, so ``z = y_ecl·sin(eps)``
    sweeps roughly ``+/-0.4`` AU once a year.  The two possibilities differ by two
    orders of magnitude, so this test has no grey zone worth arguing about — and it
    matters because a frame error rotates along-scan into across-scan, which is the
    one thing that would destroy the observable outright.
    """
    lo, hi = _f(z_min), _f(z_max)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return {"verdict": "NOT_MEASURED", "max_abs_z_au": None}
    m = max(abs(lo), abs(hi))
    if m <= ecliptic_max_au:
        verdict = "ECLIPTIC_LIKE"
        note = ("|z_gaia| stays within a thin slab about the ecliptic, which is "
                "where an L2 observer sits: the vectors are ECLIPTIC, and using "
                "them as ICRS would tilt every prediction by 23.44 deg")
    elif m >= equatorial_min_au:
        verdict = "EQUATORIAL_LIKE"
        note = ("|z_gaia| sweeps far out of the ecliptic plane, which is what the "
                "obliquity does to an L2 orbit: the vectors are EQUATORIAL (ICRS)")
    else:
        verdict = "AMBIGUOUS"
        note = ("max |z_gaia| falls between the two expectations; do not assume a "
                "frame, cross-check against a JPL Horizons Gaia ephemeris before "
                "computing anything")
    return {"verdict": verdict, "max_abs_z_au": m, "note": note}


def interpret_epoch_zero_point(row: dict, tol_days: float = 1e-6) -> dict:
    """What ``epoch`` counts from, and whether ``epoch_utc`` is a different scale.

    Two readings from one aggregate query:

    * if ``epoch_utc - epoch`` is **constant** to within ``tol_days``, the two
      columns are the same clock with different zero points, and the constant
      identifies the offset — 2455197.5 would mean ``epoch_utc`` is a Julian Date
      and ``epoch`` counts days from 2010-01-01;
    * if it **drifts**, the columns are in different time scales.  The drift is the
      TCB-minus-UTC difference and is the number that matters: at 30 arcsec/day a
      10-second scale error is a 3.5 mas along-track shift applied coherently to
      every object in the catalogue.

    ``tol_days`` defaults to ``1e-6`` d — 0.086 s — and that number is set by the
    physics, not by floating point.  A main-belt asteroid moves ~30 arcsec/day, so
    0.086 s of timing error is 0.03 mas of along-track displacement, comfortably
    below Gaia's per-transit precision.  A looser tolerance would call a
    ten-second time-scale difference "constant" and wave through a 3.5 mas
    coherent shift, which is several sigma of exactly the signal being searched
    for.
    """
    out: dict = {"verdict": "NOT_MEASURED"}
    lo, hi = _f(row.get("diff_min")), _f(row.get("diff_max"))
    e_lo, e_hi = _f(row.get("epoch_min")), _f(row.get("epoch_max"))
    if math.isfinite(e_lo) and math.isfinite(e_hi):
        out["epoch_span_days"] = e_hi - e_lo
        out["mjd_if_2010_zero_point"] = [MJD_2010_0 + e_lo, MJD_2010_0 + e_hi]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return out
    spread = abs(hi - lo)
    out["offset_min"] = lo
    out["offset_max"] = hi
    out["offset_spread_days"] = spread
    out["offset_spread_seconds"] = spread * 86400.0
    if spread <= tol_days:
        out["verdict"] = "CONSTANT_OFFSET"
        mid = 0.5 * (lo + hi)
        out["offset_days"] = mid
        if abs(mid - JD_2010_0) < 1.0:
            out["note"] = ("epoch_utc - epoch is ~2455197.5 d: epoch_utc is a "
                           "Julian Date and epoch counts days from 2010-01-01, as "
                           "assumed. The TIME SCALE is still unsettled by this "
                           "query — read the TAP_SCHEMA description.")
        elif abs(mid - JD_MINUS_MJD) < 1.0:
            out["note"] = ("the offset is the JD-MJD constant: one column is MJD "
                           "and the other JD")
        else:
            out["note"] = (f"constant offset of {mid} d, which matches neither the "
                           f"2010.0 zero point nor JD-MJD: DO NOT assume either "
                           f"until the schema description is read")
    else:
        out["verdict"] = "DRIFTING_OFFSET"
        out["note"] = (
            f"epoch_utc - epoch varies by {spread * 86400.0:.3f} s across the "
            f"mission, so the two columns are in DIFFERENT TIME SCALES. An "
            f"asteroid at 30 arcsec/day moves "
            f"{spread * 86400.0 * 30.0 / 86400.0 * 1000.0:.2f} mas in that "
            f"interval — a coherent along-track shift across the whole catalogue "
            f"if the wrong column is used.")
    return out


def interpret_rows_per_transit(rows, count_key: str = "n_rows") -> dict:
    """Multiplicity of observation rows within one transit, and what it costs.

    Rows from one transit share an attitude solution and a scan angle, so their
    systematic error is common and they are not independent samples.  Averaging
    ``N`` of them as if they were understates the transit's error by ``sqrt(N)``,
    and at ``N = 9`` that is a factor of three applied to every signal-to-noise in
    the channel — the same arithmetic that manufactured LOOM's first 150
    "anomalies" (``docs/loom.md`` §2.2).
    """
    counts = [int(_f(r.get(count_key))) for r in rows
              if math.isfinite(_f(r.get(count_key)))]
    if not counts:
        return {"verdict": "NOT_MEASURED"}
    hist: dict[int, int] = {}
    for c in counts:
        hist[c] = hist.get(c, 0) + 1
    mean = sum(counts) / len(counts)
    return {
        "verdict": "MEASURED",
        "n_transits": len(counts),
        "min": min(counts),
        "max": max(counts),
        "mean": mean,
        "histogram": {str(k): hist[k] for k in sorted(hist)},
        "sigma_inflation_if_treated_independently": math.sqrt(mean),
        "note": ("rows within a transit are NOT independent: they share one "
                 f"attitude and one scan angle. Treating a mean of {mean:.2f} of "
                 f"them as independent understates the transit error by "
                 f"{math.sqrt(mean):.2f}x."),
    }


# ---------------------------------------------------------------------------
# Quality cuts — pure, offline, and tested
# ---------------------------------------------------------------------------
# The value that `astrometric_outcome_ccd` takes for a usable observation is NOT
# known here.  Zero is the conventional "no error" code and is the obvious guess,
# which is exactly why it is written down as a guess and not used by default.
ASSUMED_GOOD_OUTCOME = 0


@dataclass(frozen=True)
class QualityCuts:
    """A quality selection over the per-observation flags.

    **Every flag cut is off by default, including ``drop_rejected``, and that
    reverses the obvious default on purpose.**

    Two separate reasons, and either alone would be sufficient.

    First, the rejected and bad-outcome rows are this channel's *signal*.  Every
    published search over this dataset works post-fit, so an object that fails to
    fit is invisible to all of them; SEXTANT screens on the rejection pattern
    instead (see the module docstring).  A helper that discarded those rows by
    default would make the channel impossible, and — worse — the loss would be
    invisible, because what came out the other end would still look like a clean
    dataset.

    Second, the meanings of ``astrometric_outcome_ccd`` and
    ``astrometric_outcome_transit`` are UNVERIFIED (:data:`OPEN_QUESTIONS`), and a
    cut on a guessed good-value has two failure modes that look identical from the
    outside: if the guess is right it removes a few per cent, if it is wrong it
    removes 99%, and either way the pipeline runs to completion and reports a
    number.  So the caller passes the values explicitly, after reading what the
    probe measured, and the report always names which rules ran and what each
    removed.

    ``drop_rejected`` remains available for the residual path, where a caller
    genuinely does want the fitted subset.  :func:`_truthy` handles the column's
    unverified type by deciding string values on content rather than on Python
    truthiness — ``bool('false')`` is ``True``, and a naive version of this cut
    would have discarded every FPR observation.
    """

    keep_outcome_ccd: tuple[int, ...] | None = None
    keep_outcome_transit: tuple[int, ...] | None = None
    drop_rejected: bool = False
    require_finite_position: bool = True
    require_finite_scan_angle: bool = True
    max_error_mas: float | None = None


def summarise_quality_values(rows) -> dict:
    """Value distribution of each quality flag, as the probe measures it live.

    Offline twin of the probe's ``GROUP BY`` queries, so a local slice can be
    checked without the network and so the tests can exercise the same reading.
    """
    out: dict[str, dict[str, int]] = {}
    for col in QUALITY_FLAG_COLUMNS:
        hist: dict[str, int] = {}
        present = False
        for r in rows:
            if col not in r:
                continue
            present = True
            k = str(_scalar(r.get(col)))
            hist[k] = hist.get(k, 0) + 1
        if present:
            out[col] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    return out


def apply_quality_cuts(rows, cuts: QualityCuts | None = None
                       ) -> tuple[list[dict], dict]:
    """Apply the cuts and return ``(kept, report)``.

    The report is the point.  It names every rule, how many rows it removed, and —
    when no flag cut was configured — says so in a verdict rather than leaving the
    reader to infer that the data is unfiltered.  A quality helper that silently
    does nothing is worse than none at all, because downstream reads "cuts applied"
    and believes it.
    """
    cuts = cuts or QualityCuts()
    removed: dict[str, int] = {}
    kept: list[dict] = []
    for r in rows:
        why = None
        if cuts.drop_rejected and "is_rejected" in r and _truthy(r.get("is_rejected")):
            why = "is_rejected"
        elif (cuts.keep_outcome_ccd is not None
                and "astrometric_outcome_ccd" in r
                and _scalar(r.get("astrometric_outcome_ccd"))
                not in cuts.keep_outcome_ccd):
            why = "astrometric_outcome_ccd"
        elif (cuts.keep_outcome_transit is not None
                and "astrometric_outcome_transit" in r
                and _scalar(r.get("astrometric_outcome_transit"))
                not in cuts.keep_outcome_transit):
            why = "astrometric_outcome_transit"
        elif cuts.require_finite_position and not (
                math.isfinite(_f(r.get("ra"))) and math.isfinite(_f(r.get("dec")))):
            why = "non_finite_position"
        elif cuts.require_finite_scan_angle and not math.isfinite(
                _f(r.get("position_angle_scan"))):
            # Without the scan angle the observation cannot be projected onto the
            # along-scan axis, which is the only axis that carries the precision.
            why = "non_finite_position_angle_scan"
        elif cuts.max_error_mas is not None:
            err = max(_f(r.get("ra_error_random")), _f(r.get("dec_error_random")))
            if not math.isfinite(err) or err > float(cuts.max_error_mas):
                why = "error_above_max"
        if why is None:
            kept.append(r)
        else:
            removed[why] = removed.get(why, 0) + 1
    report = {
        "n_in": len(rows),
        "n_kept": len(kept),
        "removed_by_rule": removed,
        "cuts": {
            "keep_outcome_ccd": list(cuts.keep_outcome_ccd)
            if cuts.keep_outcome_ccd is not None else None,
            "keep_outcome_transit": list(cuts.keep_outcome_transit)
            if cuts.keep_outcome_transit is not None else None,
            "drop_rejected": cuts.drop_rejected,
            "require_finite_position": cuts.require_finite_position,
            "require_finite_scan_angle": cuts.require_finite_scan_angle,
            "max_error_mas": cuts.max_error_mas,
        },
        "values_seen": summarise_quality_values(rows),
    }
    if cuts.keep_outcome_ccd is None and cuts.keep_outcome_transit is None:
        report["verdict"] = "NO_OUTCOME_FLAG_CUT_APPLIED_MEANINGS_UNVERIFIED"
        report["note"] = (
            "astrometric_outcome_ccd / astrometric_outcome_transit were NOT used "
            "to filter: their value meanings are unverified (OPEN_QUESTIONS "
            "'outcome_flag_meanings'), and in this channel they are the OBSERVABLE "
            "rather than a filter. Read the probe's measured distribution, then "
            "pass keep_outcome_ccd=(...) explicitly if the residual path needs it.")
    else:
        report["verdict"] = "OUTCOME_FLAG_CUT_APPLIED"
        report["note"] = (
            "an outcome-code cut was applied: this is the residual path's "
            "selection, and it removes exactly the rows the rejection screen "
            "treats as signal. Do not feed this output to rejection_ledger.")
    return kept, report


# ---------------------------------------------------------------------------
# The rejection screen — pure, offline, and tested
# ---------------------------------------------------------------------------
def partition_by_rejection(rows) -> dict:
    """Split rows into fitted and rejected **without discarding either**.

    The rejection screen and the residual path want opposite halves of the same
    pull, and both halves are needed at once — the numerator and its denominator.
    So this labels rather than filters, and returns the counts alongside, which is
    the smallest unit of the observable.
    """
    fitted, rejected, unknown = [], [], []
    for r in rows:
        if "is_rejected" not in r or r.get("is_rejected") is None:
            unknown.append(r)
        elif _truthy(r.get("is_rejected")):
            rejected.append(r)
        else:
            fitted.append(r)
    n = len(rows)
    known = len(fitted) + len(rejected)
    return {
        "fitted": fitted,
        "rejected": rejected,
        "unlabelled": unknown,
        "n_total": n,
        "n_rejected": len(rejected),
        "rejected_fraction": (len(rejected) / known) if known else float("nan"),
        "note": ("`unlabelled` holds rows with no is_rejected column — every "
                 "gaiadr3 row, since the column is FPR-only. They are NOT counted "
                 "as fitted: an absent flag is not a passed one."),
    }


def rejection_ledger(census_rows, flag_column: str | None = None) -> dict:
    """Fold a ``GROUP BY (number_mp, flag)`` census into a per-object ledger.

    Offline twin of :meth:`GaiaSSO.rejection_census`.  For each object it returns
    the **attempts** (the sum over every flag value — the denominator), the count
    per code, and the modal code.  It deliberately computes no rejection *rate*:
    which codes count as a failure is unverified, so the rate is the caller's to
    define once the probe has said what the codes mean, and manufacturing one here
    would bake in the assumption the channel exists to test.

    **What the denominator is, exactly.**  It is the number of observations that
    were *written to* ``sso_observation`` for this object — not the number of
    transits Gaia's scanning law predicted.  If a transit can fail so completely
    that no row is written, the denominator is censored in the same direction as
    the signal and the screen partly measures its own selection function.  That is
    ``OPEN_QUESTIONS['rejection_denominator']`` and it has to be settled before any
    rate from this ledger is believed.
    """
    per: dict = {}
    for r in census_rows:
        num = _scalar(r.get("number_mp"))
        if num is None:
            continue
        col = flag_column or _scalar(r.get("flag_column")) or "flag"
        n = _f(r.get("n"))
        if not math.isfinite(n):
            continue
        entry = per.setdefault(num, {"number_mp": num, "attempts": 0,
                                     "by_flag": {}})
        by = entry["by_flag"].setdefault(str(col), {})
        key = str(_scalar(r.get("flag_value")))
        by[key] = by.get(key, 0) + int(n)
    for entry in per.values():
        # The attempts denominator is the sum over ONE flag column's values, not
        # over all of them: summing two columns would count every observation twice.
        first = next(iter(entry["by_flag"].values()), {})
        entry["attempts"] = sum(first.values())
        entry["modal_flag"] = (max(first.items(), key=lambda kv: kv[1])[0]
                               if first else None)
        entry["n_flag_values"] = len(first)
    return per


def interpret_rejection_fraction(n_rejected, n_total,
                                 reference: str = "gaiadr3") -> dict:
    """Does ``is_rejected`` mark what the published outlier fractions describe?

    The calibration that says whether the column means what the channel assumes.
    Gaia's documented SSO astrometric outlier fraction is ~0.58% in DR3 and ~1% in
    DR2, so a measured fraction in that neighbourhood says the column marks outlier
    rejection and that the rejected set is of order 1e5-1e6 rows — comfortably
    fetchable, which is what makes the screen possible at all.  A fraction an order
    of magnitude away says the column marks something else, and every rate built on
    it would be a rate of something else.
    """
    n_r, n_t = _f(n_rejected), _f(n_total)
    if not (math.isfinite(n_r) and math.isfinite(n_t)) or n_t <= 0:
        return {"verdict": "NOT_MEASURED"}
    frac = n_r / n_t
    ref = PUBLISHED_OUTLIER_FRACTION.get(reference, 0.0058)
    ratio = frac / ref if ref else float("inf")
    if 0.3 <= ratio <= 3.0:
        verdict, note = "CONSISTENT_WITH_PUBLISHED_OUTLIER_FRACTION", (
            f"{frac:.4%} against a published {ref:.2%}: is_rejected plausibly "
            f"marks the documented astrometric outlier rejection")
    elif frac < 1e-5:
        verdict, note = "ALMOST_NOTHING_REJECTED", (
            f"only {frac:.6%} rejected — either the column is near-empty in this "
            f"release or it marks a much rarer condition than outlier rejection")
    else:
        verdict, note = "INCONSISTENT_WITH_PUBLISHED_OUTLIER_FRACTION", (
            f"{frac:.4%} against a published {ref:.2%}, a factor of {ratio:.1f}: "
            f"is_rejected does NOT mark what the published outlier fraction "
            f"describes, so do not read a rate built on it as a rejection rate "
            f"until the column's meaning is established")
    return {"verdict": verdict, "fraction": frac, "reference_fraction": ref,
            "ratio_to_reference": ratio, "n_rejected": int(n_r),
            "n_total": int(n_t), "note": note}


def check_columns_for_rejection_screen(rows_or_columns,
                                       release: str = "gaiafpr") -> dict:
    """Refuse a pull that cannot support the rejection screen.

    Separate from :func:`check_columns_for_residuals` because a pull can be
    perfectly adequate for one and useless for the other.  ``is_rejected`` is
    FPR-only, so its absence from a DR3 pull is expected and is reported as a
    reduced capability rather than as an error.
    """
    if isinstance(rows_or_columns, dict):
        have = {str(c).lower() for c in rows_or_columns}
    elif rows_or_columns and isinstance(rows_or_columns[0], dict):
        have = {str(c).lower() for c in rows_or_columns[0]}
    else:
        have = {str(c).lower() for c in rows_or_columns}
    missing = [c for c in REQUIRED_FOR_REJECTION_SCREEN if c not in have]
    has_rejected = "is_rejected" in have
    out = {
        "ok": not missing,
        "missing": missing,
        "has_is_rejected": has_rejected,
        "verdict": "OK" if not missing else "MISSING_REQUIRED_COLUMNS",
    }
    if not has_rejected:
        out["note"] = (
            "no is_rejected column: expected for gaiadr3, where the screen runs on "
            "the outcome codes alone" if release == "gaiadr3" else
            "is_rejected is absent from a gaiafpr pull — the FPR reduction's own "
            "rejection flag is the channel's cleanest observable and this pull "
            "cannot see it")
    return out


def transit_groups(rows) -> dict:
    """Group rows by ``transit_id`` so a caller can collapse before fitting.

    Provided because the alternative — feeding per-CCD rows straight into a fit —
    is the single easiest way to get a wrong answer out of this dataset, and it
    fails in the direction of *more* significance, not less.
    """
    groups: dict[object, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(_scalar(r.get("transit_id")), []).append(i)
    sizes = [len(v) for v in groups.values()]
    return {
        "groups": groups,
        "n_transits": len(groups),
        "n_rows": len(rows),
        "mean_rows_per_transit": (sum(sizes) / len(sizes)) if sizes else float("nan"),
        "max_rows_per_transit": max(sizes) if sizes else 0,
    }


def check_columns_for_residuals(rows_or_columns) -> dict:
    """Refuse a pull that is missing anything the residual computation needs.

    ``position_angle_scan`` and the observer state vectors are the ones this is
    really guarding.  A pull without them does not fail — it produces residuals
    that cannot be projected onto the precise axis, or a prediction computed for
    the wrong observer, and both look like data.
    """
    if isinstance(rows_or_columns, dict):
        have = {str(c).lower() for c in rows_or_columns}
    elif rows_or_columns and isinstance(rows_or_columns[0], dict):
        have = {str(c).lower() for c in rows_or_columns[0]}
    else:
        have = {str(c).lower() for c in rows_or_columns}
    missing = [c for c in REQUIRED_FOR_RESIDUALS if c not in have]
    return {
        "ok": not missing,
        "missing": missing,
        "verdict": "OK" if not missing else "MISSING_REQUIRED_COLUMNS",
        "note": "" if not missing else (
            f"the residual computation cannot run without {missing}; "
            f"position_angle_scan is the along-scan axis and x_gaia..vz_gaia are "
            f"the observer state, and neither is reconstructible downstream"),
    }


# ---------------------------------------------------------------------------
# Cross-release reconciliation — pure, offline, and tested
# ---------------------------------------------------------------------------
DEDUP_KEYS = ("observation_id", "transit_ccd")


def dedup_key(row: dict, strategy: str = "observation_id",
              epoch_ndigits: int = 6):
    """The identity of one observation, for cross-release deduplication.

    ``observation_id`` is the obvious key and the default.  Whether it survives
    across releases is the open question ``observation_id_stability``: if FPR
    re-minted the ids then a key-based dedup finds no overlap at all and reports
    two disjoint datasets that are in fact the same observations twice.

    ``transit_ccd`` is the fallback that does not depend on that.  A transit
    crosses several CCDs about 4.4 s apart, so ``transit_id`` alone is not unique
    per row; the epoch, rounded to ``1e-6`` d (0.086 s), separates the CCDs while
    tolerating a re-reduction that moved the timestamp by less than that.  It keys
    on the *physical event* rather than on a catalogue id, which is what a
    re-reduction cannot change.
    """
    if strategy == "observation_id":
        return ("observation_id", _scalar(row.get("observation_id")))
    if strategy == "transit_ccd":
        e = _f(row.get("epoch"))
        return ("transit_ccd",
                _scalar(row.get("number_mp")),
                _scalar(row.get("transit_id")),
                round(e, epoch_ndigits) if math.isfinite(e) else None)
    raise ValueError(f"unknown dedup strategy {strategy!r}; expected one of "
                     f"{DEDUP_KEYS}")


@dataclass
class ReconcileResult:
    """A single deduplicated observation set, plus why it can be believed."""

    rows: list[dict] = field(default_factory=list)
    verdict: str = "NOT_RUN"
    report: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)


def _epoch_span(rows) -> tuple[float, float]:
    vals = [_f(r.get("epoch")) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return (float("nan"), float("nan"))
    return (min(vals), max(vals))


def reconcile_observations(dr3_rows, fpr_rows, policy: str = "prefer_fpr",
                           strategy: str = "observation_id",
                           epoch_ndigits: int = 6,
                           allow_unverified_union: bool = False
                           ) -> ReconcileResult:
    """One deduplicated observation set from the two releases — or a refusal.

    **The honest state of this question, stated plainly: it is not settled, and
    this function will not guess.**

    ``gaiafpr.sso_observation`` holds 46,264,083 rows and ``gaiadr3.sso_observation``
    23,336,467, a ratio of 1.98, against FPR's ~66-month window and DR3's ~34 —
    a ratio of 1.94.  That is *suggestive* that FPR is a re-reduction covering a
    longer baseline and therefore a **superset**, in which case the correct answer
    is "take FPR, and join DR3 only for its ``g_mag``/``g_flux`` photometry".  It
    is not a measurement, and the two possible errors are both severe:

    * treat an increment as a superset and half of every arc is thrown away —
      which shortens exactly the baseline the acceleration fit needs;
    * treat a superset as an increment and every shared observation is counted
      twice, which halves the apparent scatter of every object and inflates every
      signal-to-noise by ``sqrt(2)``.

    So the decision is made from **evidence in the rows themselves**.  If the dedup
    key matches across the two releases, they genuinely share observations and the
    union deduplicates correctly.  If the key matches nothing *while the epoch
    ranges overlap*, then either the ids were re-minted or the releases really are
    disjoint in content, and those two cases demand opposite handling — so the
    function returns ``KEY_DISJOINT_BUT_EPOCHS_OVERLAP`` and refuses to produce a
    union unless ``allow_unverified_union`` is passed explicitly.  Try
    ``strategy='transit_ccd'`` first: it keys on the physical event and survives a
    re-minting.

    **TODO(SEXTANT-Q7/Q8): settle this with the probe.**  ``release_overlap`` in
    :meth:`GaiaSSO.diagnostics` asks it three ways — per-object row counts in each
    release, a cross-schema join on ``observation_id``, a cross-schema join on
    ``(transit_id, number_mp)`` — and dumps the verbatim ids of one object from
    each release so a human can compare them by eye if the joins are refused.  Once
    the probe has run, the answer belongs in this docstring as a measurement and
    the default ``policy`` can become a decision rather than a negotiation.

    Policies: ``prefer_fpr`` (union, FPR wins a collision, because it is the later
    reduction), ``prefer_dr3``, ``fpr_only``, ``dr3_only``.  Every row is tagged
    ``_release`` so nothing downstream has to guess where it came from.
    """
    dr3 = [dict(r, _release="gaiadr3") for r in dr3_rows]
    fpr = [dict(r, _release="gaiafpr") for r in fpr_rows]
    res = ReconcileResult()

    if policy not in ("prefer_fpr", "prefer_dr3", "fpr_only", "dr3_only"):
        raise ValueError(f"unknown policy {policy!r}")

    def keyed(rows):
        out: dict = {}
        collisions = 0
        for r in rows:
            k = dedup_key(r, strategy, epoch_ndigits)
            if k in out:
                collisions += 1
            out[k] = r
        return out, collisions

    k_dr3, coll_dr3 = keyed(dr3)
    k_fpr, coll_fpr = keyed(fpr)
    overlap = set(k_dr3) & set(k_fpr)
    dr3_span, fpr_span = _epoch_span(dr3), _epoch_span(fpr)
    epochs_overlap = (
        math.isfinite(dr3_span[0]) and math.isfinite(fpr_span[0])
        and dr3_span[0] <= fpr_span[1] and fpr_span[0] <= dr3_span[1])

    res.report = {
        "policy": policy,
        "strategy": strategy,
        "n_dr3": len(dr3),
        "n_fpr": len(fpr),
        "n_overlap": len(overlap),
        "n_dr3_only": len(k_dr3) - len(overlap),
        "n_fpr_only": len(k_fpr) - len(overlap),
        "within_release_key_collisions": {"gaiadr3": coll_dr3, "gaiafpr": coll_fpr},
        "epoch_span_dr3": list(dr3_span),
        "epoch_span_fpr": list(fpr_span),
        "epoch_ranges_overlap": bool(epochs_overlap),
    }

    if coll_dr3 or coll_fpr:
        # The key is not unique inside a single release, so it cannot identify an
        # observation across two.  Deduplicating on it would delete real rows.
        res.verdict = "KEY_NOT_UNIQUE_WITHIN_RELEASE"
        res.notes.append(
            f"the {strategy!r} key repeats within a release "
            f"(dr3={coll_dr3}, fpr={coll_fpr}), so it does not identify an "
            f"observation; deduplicating on it would DELETE real rows. Try "
            f"strategy='transit_ccd' with a finer epoch_ndigits.")
        res.rows = []
        return res

    if policy == "fpr_only":
        res.rows, res.verdict = fpr, "FPR_ONLY"
        res.notes.append(
            "FPR only, as asked. Correct if and only if FPR is a superset of DR3 "
            "— see OPEN_QUESTIONS 'release_overlap'; if it is not, this silently "
            "discards the DR3-only part of every arc.")
        return res
    if policy == "dr3_only":
        res.rows, res.verdict = dr3, "DR3_ONLY"
        return res

    if not overlap and epochs_overlap and dr3 and fpr:
        res.verdict = "KEY_DISJOINT_BUT_EPOCHS_OVERLAP"
        res.notes.append(
            f"the two releases share epochs ({dr3_span} vs {fpr_span}) but no "
            f"{strategy!r} key matches. Either the ids were re-minted in FPR — in "
            f"which case a union DOUBLE-COUNTS every shared observation and halves "
            f"the apparent scatter of every object — or the releases really are "
            f"disjoint in content. Those need opposite handling, so no union is "
            f"returned. Retry with strategy='transit_ccd', or pass "
            f"allow_unverified_union=True if the probe has settled it.")
        if not allow_unverified_union:
            res.rows = []
            return res
        res.notes.append("allow_unverified_union=True: returning the union anyway")

    primary, secondary = ((k_fpr, k_dr3) if policy == "prefer_fpr"
                          else (k_dr3, k_fpr))
    merged = dict(secondary)
    merged.update(primary)
    # Deterministic order: by epoch then by key, so two runs over the same data
    # produce byte-identical output and a diff means the data moved.
    res.rows = sorted(
        merged.values(),
        key=lambda r: (_f(r.get("epoch")) if math.isfinite(_f(r.get("epoch")))
                       else float("inf"), str(r.get("observation_id"))))
    if res.verdict == "NOT_RUN":
        res.verdict = "UNION_DEDUPLICATED" if overlap else "UNION_NO_OVERLAP"
    res.report["n_out"] = len(res.rows)
    if not overlap and not epochs_overlap and dr3 and fpr:
        res.notes.append(
            "no key overlap and no epoch overlap: the releases cover disjoint "
            "time, so the union is simply the concatenation and is correct")
    return res


# ---------------------------------------------------------------------------
# The probe entry point
# ---------------------------------------------------------------------------
def probe(cfg=None, out_dir: str | Path | None = None,
          url: str = GAIA_TAP, timeout: float = 1800.0,
          sample_rows: int = 20) -> dict:
    """Stage 0, runner-only: record the live schema and settle the open questions.

    Writes ``results/sextant/probe.json`` **after every query**, so a job timeout
    part-way through leaves every answer already obtained on disk instead of
    losing the pass — the discipline that TOCSIN paid three hours to learn.

    Nothing in SEXTANT should be built on top of this module until this has run:
    the synchronous row cap, the frame of the state vectors, the epoch's time
    scale, the meaning of the quality flags and the DR3/FPR relationship are all
    unverified inferences today, and each of them fails silently rather than
    loudly.  :data:`OPEN_QUESTIONS` is carried into the record verbatim so the
    committed file says what was still unknown at the time it was written.
    """
    root = Path(getattr(cfg, "root", None) or _repo_root())
    out = Path(out_dir) if out_dir else root / "results" / "sextant"
    rec: dict = {
        "probed_at_utc": _utc(),
        "service": url,
        "reached": False,
        "verdict": "NOT_RUN",
        "open_questions": OPEN_QUESTIONS,
        "expected": {"rows": MEASURED_ROWS, "columns": MEASURED_COLUMNS,
                     "measured_on": "2026-08-25"},
        "column_inventory": {
            "shared": list(SHARED_COLUMNS),
            "gaiadr3_only": list(DR3_ONLY_COLUMNS),
            "gaiafpr_only": list(FPR_ONLY_COLUMNS),
            "required_for_residuals": list(REQUIRED_FOR_RESIDUALS),
        },
    }
    sso = GaiaSSO(url=url, timeout=timeout)

    def checkpoint() -> None:
        rec["verdict"] = "OK" if rec.get("reached") else "NO_DATA_REACHED"
        rec["calls"] = sso.calls
        _write_json(out / "probe.json", rec)

    checkpoint()

    # The live schema first: it is the cheapest query, it proves reachability, and
    # its unit/ucd/description fields may answer several open questions outright.
    try:
        rec["schema"] = sso.describe()
        rec["schema_column_counts"] = {t: len(c) for t, c in rec["schema"].items()}
        rec["reached"] = True
    except Exception as exc:                                  # noqa: BLE001
        rec["schema_error"] = f"{type(exc).__name__}: {exc}"[:800]
    checkpoint()

    if rec["reached"]:
        # Does the live schema still contain every column this module selects?  A
        # column silently dropped by the archive would otherwise surface as an ADQL
        # error in the middle of a six-hour bulk pull.
        rec["column_check"] = {}
        for release, table in (("gaiafpr", FPR_OBSERVATION),
                               ("gaiadr3", DR3_OBSERVATION)):
            have = {c["name"].lower() for c in rec["schema"].get(table, [])}
            want = ALL_COLUMNS[release]
            rec["column_check"][table] = {
                "n_live": len(have),
                "n_expected": MEASURED_COLUMNS.get(table),
                "missing_from_live": [c for c in want if c not in have],
                "unclaimed_by_this_module": sorted(
                    c for c in have if c not in {w.lower() for w in want}),
                "residuals_ok": check_columns_for_residuals(sorted(have)),
            }
        checkpoint()

        try:
            rec["service_limits"] = sso.service_limits()
        except Exception as exc:                              # noqa: BLE001
            rec["service_limits"] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        checkpoint()

        diag: dict = {}
        rec["diagnostics"] = diag

        def _record(name, result):
            diag[name] = result
            checkpoint()

        try:
            sso.diagnostics(on_result=_record, sample_rows=sample_rows)
        except Exception as exc:                              # noqa: BLE001
            diag["_fatal"] = f"{type(exc).__name__}: {exc}"[:800]
        checkpoint()

        # Turn the raw measurements into the verdicts the next agent needs, so the
        # committed record answers the questions rather than only containing them.
        answers: dict = {}
        d = diag.get("sync_row_cap", {})
        if "rows" in d:
            answers["sync_row_cap"] = interpret_sync_row_cap(d["rows"], 5000)
        fk = f"state_vector_frame_{FPR_OBSERVATION.replace('.', '_')}"
        fd = (diag.get(fk, {}).get("data") or [{}])[0]
        answers["state_vector_frame"] = interpret_state_vector_frame(
            fd.get("z_min"), fd.get("z_max"))
        ek = f"epoch_ranges_{FPR_OBSERVATION.replace('.', '_')}"
        ed = (diag.get(ek, {}).get("data") or [{}])[0]
        answers["epoch_zero_point"] = interpret_epoch_zero_point(ed)
        answers["rows_per_transit"] = interpret_rows_per_transit(
            diag.get("rows_per_transit", {}).get("data") or [])
        # The rejection calibration.  Whichever of the two is_rejected predicates
        # the service accepted also settles the column's type, so both are read and
        # the one that returned rows wins.
        n_total = ((diag.get("rejection_fraction_total", {}).get("data")
                    or [{}])[0]).get("n")
        n_rej = None
        for key in ("rejection_fraction_rejected",
                    "rejection_fraction_rejected_numeric"):
            d = diag.get(key, {})
            if "error" not in d and (d.get("data") or [{}])[0].get("n") is not None:
                n_rej = (d.get("data") or [{}])[0].get("n")
                answers["is_rejected_predicate_that_worked"] = key
                break
        answers["rejection_fraction"] = interpret_rejection_fraction(n_rej, n_total)
        answers["rejection_ledger_sample"] = rejection_ledger(
            (diag.get("rejection_census_astrometric_outcome_ccd", {}).get("data")
             or []), flag_column="astrometric_outcome_ccd")
        rec["answers"] = answers
        checkpoint()

        # And a real, deduplicated cross-release reconciliation on one object,
        # computed by the code that will do it at scale.  This is the measurement
        # that turns TODO(SEXTANT-Q7/Q8) into a decision.
        try:
            fpr_rows = (diag.get("sample_rows_gaiafpr", {}) or {}).get("data") or []
            dr3_rows = (diag.get("sample_rows_gaiadr3", {}) or {}).get("data") or []
            rec["reconciliation_trial"] = {}
            for strat in DEDUP_KEYS:
                r = reconcile_observations(dr3_rows, fpr_rows, strategy=strat)
                rec["reconciliation_trial"][strat] = {
                    "verdict": r.verdict, "report": r.report, "notes": r.notes}
        except Exception as exc:                              # noqa: BLE001
            rec["reconciliation_trial"] = {
                "error": f"{type(exc).__name__}: {exc}"[:400]}
        checkpoint()

    checkpoint()
    print(f"[sextant] probe verdict={rec['verdict']} calls={rec.get('calls')} "
          f"-> {out / 'probe.json'}")
    return rec

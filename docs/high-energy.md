# Archival high-energy technosignature searches

Status: **EXPLORATORY_ARCHIVAL_RESEARCH_CANDIDATE**. No detector, archive
acquisition, workflow, scheduled task, measured
coverage, sensitivity estimate or candidate exists for this track. The next
eligible work is a frozen XMM-Newton event-product feasibility design and product checks,
not a sky search. This is a research plan, not a result or a manuscript.

## Motivation and provenance

John Michael Godier, [High Energy Alien Civilizations](https://www.youtube.com/watch?v=Hit4Xnb9Sqo),
published 2025-10-08, runtime 11:20, motivates revisiting existing astronomical
surveys for engineering signatures in high-energy environments (4:10–4:27).
The video describes: X-ray/gamma-ray bands and neutron-star environments (4:39–5:14),
flashes from otherwise quiet nearby systems (5:15–5:42), hypothetical asteroid
impacts to time neutron-star bursts (6:17–7:34), X-ray-binary occultations and
population emission deficits (7:40–8:18). These are motivations, not evidence
that any anomaly is artificial. The radio source-population discussion
(8:30–10:27) is a separate possible track; brightness alone is not an origin test.

The description identifies Brian C. Lacki and Stephen DiKerby's June 2025
five-page NASA DARES white paper,
[Possibilities for SETI at High Energy, v1](https://arxiv.org/html/2506.16351v1).
The white paper advocates commensal/archival
anomaly searches in X-rays, gamma rays and neutrinos, speculative technologies,
and likely natural-astrophysics discoveries. This plan chooses X-rays first;
it does not implement neutrino searches or assert that any proposal is feasible.

The separate [Artificial Broadcasts as Galactic Populations III](https://arxiv.org/abs/2508.00249)
has a May 2026 revision. Its conditional
bounds on extremely radio-luminous populations must not become a claim about
the general rarity of life. It is provenance for the video's separate radio
branch, not validation of this X-ray design.

Sources reviewed on 2026-10-08; direct citations are provided above and below.
Raw transcript/paper snapshots are not included in this plan. No novelty claim follows from an absent channel;
an exact-observable prior-art check remains required. Mission products below
are documented archive candidates, not products reached or analyzed by this
track. The first pilot is **our implementation proposal**, not a mission/sample
specified by the video and not an executed scientific result.

## Place in the portfolio

The executable channel map is `docs/channels.md`; proposed work is indexed
separately there. `STATUS.md` records run/candidate state and is not changed by
this unrun proposal. `src/seti/confluence/registry.py` registers scored parent
samples with instrument/support tags, not research ideas: do not add this track
until it produces a defined parent, calibrated scores and provenance.

| Existing work | Reuse or overlap | Distinct proposed measurement |
|---|---|---|
| RING / S63 | Compact-object hosts, identity and offset-position controls | High-energy photon timing/spectra rather than IR excess |
| ARC / S59 | Flare energetics and unresolved-companion rejection | X-ray transients; optical spot ceilings cannot be transferred without physics validation |
| METRONOME | Event-time statistics and search-trial accounting | Photon arrival times, exposure windows and compact-object natural variability |
| OCCULT / dimming | Dip vetting and injection discipline | X-ray emitting-region geometry and event-list instrumental effects |
| BAFFLE / baffle-radio | Deficit and source-count selection-function controls | X-ray-binary population deficit; no claim that radio and X-ray deficits are equivalent |
| TOCSIN / PARALLAX4 | Association, blending and recurrence discipline | High-energy localization and cross-instrument archival confirmation |

## Hypotheses and competing explanations

All branches start with H0: the observations arise from astrophysics plus
instrument/selection effects. A structured residual is a triage item, not an
alien detection. Failure to fit one natural model does not reject H0.

| Branch | Conditional engineering motivation and proposed observable | Required natural nulls | Required instrument/selection nulls |
|---|---|---|---|
| Quiet-host flashes (later branch) | Excess short events associated with a preregistered nearby, otherwise quiet host class; quantify timing, energy and localization with uncertainty | Coronal flares, active/unresolved companions, accretion, background AGN, chance alignments, ordinary transient populations | Particle background, detector hot pixels, pile-up, dead time, chip edges, attitude motion, exposure/GTI gaps, catalog selection and crossmatch errors |
| Compact-object timing (speculative) | Repeated temporal structure beyond a fitted natural process; neutron-star impact messaging is a hypothesis, not an established viable mechanism | Pulsation, orbital modulation, accretion states, thermonuclear bursts, magnetar activity, red noise and burst clustering | Soft-proton background flares (XMM), spacecraft clock/orbit, barycentric errors, timestamp quantization, detector dead time, window aliases and trials over periods/templates |
| X-ray-binary dips | Reproducible occultation-like residuals conditional on emitting-region geometry | Eclipses, winds/absorption, accretion dips, state transitions and source confusion | Background subtraction, telemetry gaps, detector response, pile-up and pointing variations |
| Population deficits (deferred) | XRB luminosity/count residual conditional on a defensible galaxy population model | Star formation, stellar mass/age, metallicity, stochastic bright binaries and extinction | Distance, flux limits, completeness, sensitivity maps, heterogeneous observing depth and source classification |

Do not prioritize the population branch until identifiability and completeness
are demonstrated. No population deficit, periodicity, flash or unexplained dip
by itself supports artificiality. Evidence from distinct instruments is not
independent if it shares acquisition/selection/calibration errors.

## Archival dataset candidates and product gates

| Candidate service/product family | Documented role and source | Remaining gate |
|---|---|---|
| XMM-Newton Science Archive (first pilot) | [ESA archive](https://www.cosmos.esa.int/web/xmm-newton/xsa) and [archive handbook](https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/dfhb/sciencearchive.html): observations, event data and processed products | Freeze a public neutron-star/XRB sample before inspecting anomalies; verify event/time/energy schema, source/background regions, GTIs, timing mode, response and pile-up |
| Swift 2SXPS (later context/quiet-host branch) | [Catalog documentation](https://www.swift.ac.uk/2SXPS/docs.php) and [web documentation](https://www.swift.ac.uk/2SXPS/webdocs.php): source catalog, variability and light curves | Verify bins/resolution, source association, exposure/background errors and suitability for a chosen statistic |
| Fermi LAT (later extension) | [Public data access](https://fermi.gsfc.nasa.gov/ssc/data/access/): weekly all-sky photons and spacecraft pointing/history | Verify photon sparsity, localization, diffuse background, response and trials; apply [Earth-limb, quality and exposure selections](https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/lat_data_selection.html) |
| Chandra / NICER (secondary candidates, product documentation not yet reviewed) | Possible localization, timing or independent archival replication | Verify public products, availability, timing resolution, attitude/background and source-specific suitability before selecting |

The first proposed pilot screens resolved XMM event timing for bursts/dips in
neutron-star/XRB observations, with matched source/background/GTI controls,
natural timing/spectral models and held-out archival replication. Require a
model for both source events and background events, not only a light-curve cut.
[XMM EPIC external-background guidance](https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/epicextbkgd.html)
requires explicit soft-proton flare rejection: simultaneous detector responses
alone are insufficient, since a common background flare can mimic agreement.
Galaxy-wide XRB deficits remain deferred: require a population/selection model
and an independent signature before an engineering interpretation.

No release dates, source totals or sensitivity values are assumed. A catalog
row is not proof that usable time-resolved products exist. Reject any branch
whose required products are proprietary, unavailable, too poorly localized or
unable to resolve the proposed timescale. Do not substitute a different band
or product silently.

## Cheap feasibility ladder and stop rules

These are proposed future caps, not resources already consumed or approved
science runs. Adding this research plan does not launch any of these stages.

1. **Paper/product desk review, no data acquisition.** Record exact primary
   paper versions/sections, prior executed searches, justified observable,
   candidate product schema and documented limitations. Stop as
   `SOURCE_REVIEW_PENDING`, `PRIOR_ART_OVERLAP` or `NO_TESTABLE_OBSERVABLE`
   if the branch lacks a distinct, testable question. Existing natural
   variability catalogs may be better inputs than new event processing.
2. **Offline design.** Freeze one branch, target/control selection, energy/time
   ranges, quality gates, statistic, trial family and vet ladder before seeing
   anomaly scores. Model the observation window and Poisson/red-noise process.
   Simulate confounders and injections through the actual measurement pipeline;
   stop `NULL_MODEL_UNCALIBRATED` when natural/instrumental mocks produce
   uncalibrated tails. Do not tune on a selected survivor.
3. **Separately authorized archive feasibility pilot.** Metadata first; at most
   10 observations / 5 targets plus matched controls, at most 1 GiB total
   downloads, 2 CPU-hours, 4 GiB RAM, no GPU and no paid service. Count control
   products within these caps. Stop before a download/job would exceed a cap;
   do not raise caps automatically. Check licenses/terms and product sizes.
   No recursive bulk download or full-sky run. Record every attempted unit,
   service/product version, checksum, bytes, time, returned rows, failure,
   excluded interval and reason. Missing background/GTIs/calibration produce
   `INSUFFICIENT_PRODUCTS`; a transport failure produces `NO_DATA_REACHED`,
   never a scientific null. A reached but empty table is distinct from failure.
4. **Feasibility decision.** Proceed only if the necessary products are usable,
   target association holds, controls are calibrated and injection recovery is
   measured over the exact observed sampling. Report measured coverage and
   recovery with uncertainty, never assumed sensitivity. An uninformative
   window returns `INSUFFICIENT_COVERAGE`; no surviving pilot residual is an
   internal reason to reconsider the question, not a null-result paper.
5. **Separate scale decision.** A later explicit scope chooses the justified
   parent, compute budget, archival confirmation and executable integration.
   No scheduling, dispatch, subscriptions, credentials/access changes, new
   observations or outreach are implied by adding this proposal.

## Implementation and validation gates for later work

Follow `docs/channel-brief.md`: pure offline detector functions, config-held
thresholds, staged/checkpointed runner acquisition, exact shard ownership,
network-guarded pytest, explicit degradation and funnel counts. Only after
implementation would the channel gain `src/seti/<channel>/`, a CLI, config,
`tests/test_<channel>.py`, dispatch-only workflow and `results/<channel>/`.
Do not create those as empty evidence of readiness.

Required future tests: injected timing/flash/dip recovery across actual GTIs
and count regimes; natural red-noise/burst/eclipsing/flare controls; detector
background/clock/dead-time/window aliases; localization/chance-association
and duplicate observations; empty/failed/missing-product degradation;
trial-corrected false-positive calibration and held-out controls; units,
response/time conversion and barycentric conventions; reproducible seeds,
checkpoint completeness and budget termination. Every veto needs a case
that trips it. Validate version-sensitive code against runner dependencies
before any dispatch. Synthetic recovery establishes software behavior only;
archive-matched injections and calibration are needed for sensitivity claims.

A later result summary must keep `proposed`, `acquired`, `assessable`,
`excluded`, `unmeasured` and `surviving_residuals` separate. This document supports no candidate count,
scientific null, validated sensitivity or alien-detection claim. Do not overwrite historical results, queue entries or alert
state to make the proposal appear operational.

## Portfolio scope

Repository documentation only: this proposal and its proposed-track index row.
No scientific publication, source-code change, executable registry entry,
workflow, results file, scheduler/queue change or change to existing science
thresholds. Implementation, archive acquisition and scale decisions remain
separate stages subject to the feasibility and validation gates above.

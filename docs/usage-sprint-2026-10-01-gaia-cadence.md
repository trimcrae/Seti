# Gaia fixed-cadence synthetic phasing control — 2026-10-02

AI-assisted implementation by OpenAI Codex; source, mathematics, raw evidence and actual receipts independently reviewed by the sprint coordinator and a separate reviewer.

Completed one read-only control on the pinned Gaia DR3 cadence. It makes the conditioning assumptions reproducible: the uniform ten-subset model yields 8008/3268760 = 0.0024498586620002693, while the existing independent-uniform episode-phase union bound is 0.0014570756577869113. These are different conditional models; their comparison is not measured sky calibration or evidence that a scientific classifier should change.

The fixture is existing Gaia-4 DR3 source `1457486023639239296`, retrieved unmodified on 2026-09-22 through the public redistribution documented in its README. No new ESA/catalogue request was made. Git blob `06eddfdeeb713769aaa78d08843ef23b3fe02857`, 19,396 bytes, SHA256 `664c9f20d750c552c4eb1230db537b1025c7dd7e60fc6ba56efadfff9b35544f`. Time is BARYCENTER/TCB days from JD 2455197.5. Native source/transit IDs remain decimal strings in the report.

The unchanged production G quality mask admits 65 unique observations, including five variability-rejection flags that production treats as diagnostic. Adjacent gaps <= the existing one-day gap form 25 sampling blocks; representatives are their first observed epochs. They are not recovered dip peaks or independent physical episodes. The ephemeris P=1 day, duration=0.1 day, t0=0 Gaia days, tolerance=0.03, k=10 was fixed in the control design before implementation/Python execution, not fitted to flux.

| Conditional diagnostic | Actual Python result |
|---|---:|
| Alias-accepted opportunities at P, 2P, P/2 | 8 / 4 / 16 |
| Uniform ten-subset without replacement, exact alias inclusion–exclusion | 8008 / 3268760 |
| One common shift of first ten chronological block starts | 0 for each alias and their union |
| Separate phase-selected positive's shared-shift sensitivity | 0.23324861109860617 |
| Perfect-correlation toy: one epoch repeated in ten slots | 0.64; one unique epoch |

The positive deliberately selects the first ten P/2-matching starts (16 available) and tests production phasing with a fixed synthetic BLS result. It passes at P/2; the chronological negative is not phased. This is phase plumbing, not photometric completeness, BLS recovery or independent validation. The common-shift main null uses the first ten chronological starts chosen independently of phase, preserving their relative cadence. Zero belongs only to that fixed synthetic geometry, never a real-star null result.

## Exact execution and integration

Tested source `6300bcb339b226c559d5ac8c5c4d10f3f3ba10f1`, tree `f2bbe54fb92e487a8ce9d66c50c3ed6d768dada4`, isolated branch `codex/usage-sprint-2026-10-01-gaia-cadence-control`, [PR 28](https://github.com/trimcrae/Seti/pull/28). Normally merged as `a16ddd4acffde5c65b96272816d86cd97f5a37f3`, preserving fresh bot parent `169a6f117944d8930988a5e9243f4ebebb9d1fe9` and both cronwatch/watchdog commits. The integration tree includes their three operational JSON changes; all tested implementation and protected scientific/configuration blobs match exactly.

Actual focused [run 36971358336](https://github.com/trimcrae/Seti/actions/runs/36971358336), job 110725874286: Python 3.11.16, lint passed, **298 tests passed in 49.49 s**, generator passed. Its checkout `2d984f76107731ebce44f29fef3326235d8d705c` has the identical tested tree. Tests include independent finite-set enumeration, time-distance quadrature, wrap/intersection/overlap, grouping/deduplication, invalid/insufficient inputs and real-cadence positive/negative production phasing.

Actual full [push run 36971354891](https://github.com/trimcrae/Seti/actions/runs/36971354891), exact checkout `6300bcb339b226c559d5ac8c5c4d10f3f3ba10f1`: offline job 110725863247 and paper job 110725863370 completed SUCCESS, every step passed. Lint, full pytest (100% with retained skips), sample analysis/forecast reproduction, paper build and artifact uploads succeeded. The quiet full log prints no total; no full-suite case count is inferred.

The first source `541ebc5b8bf1152e51996d16f95bb51f9245eec0` failed focused lint [36971245416](https://github.com/trimcrae/Seti/actions/runs/36971245416)/110725478034: own implementation B905, missing explicit `zip(strict=...)`. Tests and generator were skipped. The repair adds `strict=True` only. Initial full run 36971245347 was cancelled; no initial full success is claimed.

[report.json](../research/parallax4/gaia-cadence-control-2026-10-02/report.json) is the exact canonical Python generator line plus LF, acquired from the original focused log: 12,959 bytes, SHA256 `9df8c08ef80f998cd1823862de8265785ea0f5efa5a9fe0930b0675fb9923023`. Both reviewers independently matched all 65 timestamps/native transit IDs/flags and every block start to original BINARY2 bytes with zero differences. Independent arithmetic agrees; the selected-shift comparison differs by 4.37e-13 from a separate float64 endpoint partition.

The [JSON receipt](usage-sprint-2026-10-01-gaia-cadence-receipt.json) pins implementation/protected blobs, exact commits/trees, original unabridged logs and report bytes. Reproduce from the repository root with `python -m seti.parallax4.cadence_control`; it prints canonical JSON/hash without writing scientific outputs. Local Python execution was unavailable and is not claimed.

Only the standalone diagnostic, tests, focused workflow and plan changed in source; production classification, thresholds, preregistrations, A4/release gates, fixtures and historical results are unchanged. Subsequent report/receipt/log/handoff commits change documentation only. Any later automatic main/docs CI is separate from the completed exact-source proof and is reported once at handoff.

## Stop and next bounded task

This diagnostic is complete; no empirical sky false-positive rate, episode independence, detection or classifier calibration is established. No remaining implementation blocker, acquisition retry, science sweep or continuing monitor is assigned.

Next distinct priority, grounded in current STATUS/brief: inspect fresh ownership/code for a demonstrably unsupported spatial-identifiability case in the DR4 photocentre ON_TARGET versus BLEND discriminator. If supported, use the already pinned real prerelease scan geometry and exact matched native transit IDs for one predeclared synthetic off-target flux/centroid coupling control. STATUS's 41-transit real Gaia-4 fit remains AMBIGUOUS with limited flux leverage; A3 failed and the A4 gate remains authoritative. Preserve those results and gates. Stop explicitly if matching/time/scan-angle inputs or a useful new degeneracy are unavailable; avoid repeating existing 21-scene controls, cadence/alias/identity/VSX cycles or unchanged source acquisition. This next task has not begun.

# Native-geometry spatial identifiability repair — 2026-10-02

AI-assisted by OpenAI Codex; independently reviewed by the coordinator, a source/evidence reviewer and a separate scientific numerical reviewer.

Completed one demonstrated confidence defect. On exact native Gaia-4 scan geometry, a predeclared synthetic affine flux trend makes both sky-blend columns indistinguishable from position/proper-motion nuisance terms. The old fit returned **ON_TARGET, ul95=0.0104477746 mas** for a cancelled hidden D=(300,−200) mas model, although arbitrary D values yield the same predictions. This is a conditional model counterexample, not an observed blend or detection.

The repair uses one direct column-scaled weighted SVD/cutoff for coefficients, covariance and rank/nullspace, with physical-unit mapping and rank-based residual DOF. Both sky coefficients must be estimable modulo nuisance parameters before finite D confidence is reported. A rank-deficient nuisance column alone does not invalidate an identifiable sky vector. Near-null covariance is computed without squaring the design in normal equations.

| Actual repaired synthetic control | Result |
|---|---|
| Exact affine, cancelled hidden blend and sky-only injection | rank 6/8; NON_ESTIMABLE; INSUFFICIENT; D/UL/confidence keys omitted |
| Predeclared near-affine perturbation 1e−6 | rank 8; sigma_D_max 8847.973810056905 mas; INSUFFICIENT |
| Predeclared nondegenerate synthetic flux, native geometry | rank 8; recovered D=(300,−200) mas; BLEND_UNRESOLVED |
| Same positive with synthetic nuisance-only parallax null | rank 7; recovered D=(300,−200) mas; BLEND_UNRESOLVED; parallax unmeasured |

## Provenance and execution

Pinned source `1457486023639239296`. Existing prerelease ZIP blob `b3d51a1506ca1d20f304a0895c294b4a9908039f`,625,651 bytes, SHA256 `07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b`; existing DR3 VOT blob `06eddfdeeb713769aaa78d08843ef23b3fe02857`,19,396 bytes, SHA256 `664c9f20d750c552c4eb1230db537b1025c7dd7e60fc6ba56efadfff9b35544f`. Public redistribution provenance is in the unchanged fixture README; no new direct ESA/catalogue fetch. Native metadata is BARYCENTER/TCB, day origin JD 2455197.5.

The exact source/transit-ID join has 63/65 photometric IDs in the prerelease, 93 collapsed usable astrometric transits and 41 retained joined rows; maximum joined time difference 1.3126321646 s. Native times/angles/parallax factors/positive centroid errors were retained; original phi/x remain separate. All 41 row/source JSON values are identical between baseline and repaired reports. Synthetic rank fields describe injected flux on native geometry, not the measured native flux model.

Baseline `ee8864210beb8279672906d0e8bb32a3c2136b76`: actual focused [36980276136](https://github.com/trimcrae/Seti/actions/runs/36980276136)/110752984520 passed 299 tests, lint, both generators; production fit was still unchanged. Analytical target-null residual 2.78e−17 and hidden-parameter prediction difference 1.42e−14 mas independently verify the witness.

Tested repair `d581914100836d629048ad2166d3a1d5b69747ca`,tree `465d1e9a2c78b93381bc3086fd611e0e9dc6bbf8`; [PR 29](https://github.com/trimcrae/Seti/pull/29). Actual focused [36982225760](https://github.com/trimcrae/Seti/actions/runs/36982225760)/110759077461 passed **313 tests in 59.03 s**, lint, both generators. Checkout `94b14b650a0977d7822aa0e87f036942452830d8` has the identical tested tree. Tests include exact/near-null confidence, nuisance-only positive, independent weighted QR covariance, mixed physical column units, N<p structural nullspace, even-median/time-translated affine refusal and invalid whitening weights.

Actual full [push run 36982221210](https://github.com/trimcrae/Seti/actions/runs/36982221210) checked out exact d581: offline job 110759357798 and paper job 110759357678/all steps completed SUCCESS. Lint, full pytest (100% with retained skips), sample reproduction, paper build and artifact uploads passed. The quiet full log prints no total; none is inferred.
Normally merged as `09d7af6e4e3ae7196a717401808e7728d6bab1fd`, preserving late watchdog parent `174f57adfcaee69bb5762373162562adb720b066`. The merge differs from the tested tree only in three retained cronwatch/watchdog operational JSONs; all implementation and protected scientific/configuration blobs match exactly.

Intermediate `cca1b0b0f82d61a1232a842d77dd4c63e6466608` failed actual focused 36981937817/110758189045 with our I001 import blank-line error. Scientific tests and generators were skipped. The repair changes two blank lines only; no environment/baseline failure is claimed.

[Baseline report](../research/parallax4/spatial-identifiability-2026-10-02/baseline-report.json) and [repaired report](../research/parallax4/spatial-identifiability-2026-10-02/report.json) are exact canonical generator payloads+LF, 19,057/25,714 bytes, SHA256 `b974300eeb4f56f06c682f0fee073328f6c80f106784d8a79c6c70dde4f412fa` / `e160061e284fed43d6691ad227faaadfdd43c1b3ae7b339372ecf1d93b402b55`. The [JSON receipt](usage-sprint-2026-10-01-spatial-identifiability-receipt.json) pins originals, source/protected blobs and unabridged logs. Separate V8 two-pass weighted QR on emitted geometry independently gives near sigma 8847.973810102147 mas (5.12e−12 relative difference); positive covariance agrees about 1e−16. This mathematical verification is separate from actual Python execution.

Reproduce from repository root with `python -m seti.parallax4.spatial_control`,which prints canonical JSON/hash without writing scientific outputs. No local Python execution is claimed. Subsequent report/log/receipt/handoff changes are documentation only; final automatic main CI is observed once separately from exact-source proof.

## Limits, stop and next bounded task

This is conditional known-design/centroid-error covariance, not empirical sky calibration. Flux-regressor uncertainty and estimated-jitter uncertainty remain uncalibrated. Jitter holds its initial numerical rank for DOF; the final weighted D guard protects loss of sky estimability. A nonestimable detector nuisance point may be a chosen gauge, not a measurement; its flag is false and z_detector is undefined. We do not claim all unestimable nuisance points are omitted. Time-translation tests cover affine refusal/even-median algebra, not full-rank covariance invariance.

Thresholds, preregistrations, A4/release gates, native fixtures and historical results are unchanged. No candidate, null-result paper, new catalogue acquisition, science sweep or continuing monitor. Current task is complete after reviewed integration; no further implementation blocker remains.

Next distinct bounded candidate, not started: inspect fresh ownership/source for a useful untested CCD-to-transit coherence/uncertainty case. The native reader records chi2_ccd and formal combined centroid errors. Only after independent evidence of a real gap and exact native CCD/flag/error bindings should one predeclared common-mode or discordant-CCD synthetic contamination control proceed. Preserve A4/results/thresholds; stop explicitly if useful inputs/gap are unavailable. Avoid repeated SVD/affine, cadence, alias, identity or VSX cycles.

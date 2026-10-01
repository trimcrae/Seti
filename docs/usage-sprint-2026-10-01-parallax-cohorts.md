# PARALLAX4 cohort integrity sprint — 2026-10-01

Branch: `codex/usage-sprint-2026-10-01-parallax-cohorts`.
Base: `5b62e3bbf2db3bfca944970c1b59213f7db4ea17`.
Scope: reducer/input integrity, offline tests and PARALLAX4 workflows.
No science runs, result rewrites or threshold changes. Main integration is
coordinated separately after independent review and passing checks.

Confirmed defects: injection tables pooled other shard widths; equal-length
histograms pooled different time grids/runs/revisions; reduce-only without a
source run purged inputs and overwrote the prior reduction with NO_SHARD_OUTPUTS.

The reducer now selects companions by accepted canonical sweep tags, checks
uniform time_edges, dimensions, finite nonnegative histograms, run/revision,
processed-file uniqueness, event counts and CSV file lineage before aggregate
writes. Incoherent or corrupt input raises CohortError and preserves existing
reduce.json and events_AB.csv. Missing shards and partial work remain usable
with DEGRADED bookkeeping, including inferred widths when --shards is omitted. Missing legacy pixel tables disable the
local veto; partial pixel denominators cannot be combined with all event rows.
New sweeps count all pixel-denominator transits independently and require exact
CSV-total agreement. Legacy records only supply searched-source transit counts
and in-grid histograms; these are lower bounds, not equal totals (unsearched
sources and out-of-grid transits remain legitimate). Legacy pixel totals below
either bound fail closed; incomplete tables above both bounds remain unprovable
without an original manifest.

Final sweep records now store SHA256 manifests for all four companion CSVs
(including absent-file markers). Checkpoints and historic artifacts lack these
manifests. Their compatibility path validates declared provenance, exact tags,
counts and file lineage, and explicitly reports `legacy_shards` and
`legacy_csv_provenance`. This does **not** prove legacy CSV byte provenance.
Equal `local`/`unknown` markers are reported as unresolved identity fields;
numeric expected GitHub source run IDs prevent that ambiguity in the workflow.

The plan rejects stage=reduce without a positive reduce_only_run_id before the
reducer can purge files. Current/prior source run IDs reach --source-run-id.
DR4/watchlist jobs preserve shard state. Reduction failure stops downstream
stages; workflow commit-back requires success.

## Receipts

Executable offline command:
`python -m pytest tests/test_parallax4.py tests/test_parallax4_cohorts.py -q -p no:cacheprovider`.

Final source code: `5fa30f2e2536bf06fcae131189ea002fef0fab62`.
Independent review by `/root/review_research` found no remaining scoped blocker
after the pixel-coverage and error-metadata repairs.

Observed [Actions run 36937937271](https://github.com/trimcrae/Seti/actions/runs/36937937271)
at that exact source SHA: success; job `110622474723` passed lint
(`All checks passed!`) and the complete offline PARALLAX4 suite. The quiet
pytest log contains 72 + 49 = **121 passing cases** and no skip/fail markers.
The independent reviewer also read these runner logs; no local Python runtime
or unseen execution is claimed.

This receipt update changes documentation only. Final head and its focused/full
CI receipts are tracked in [PR #23](https://github.com/trimcrae/Seti/pull/23);
main integration is delegated to the SETI coordinator after checks settle.
Legacy CSV byte provenance and pixel completeness remain explicitly unresolved
where old metadata cannot prove them.

## Next task

After review/CI, inspect one retained historic sweep artifact cohort read-only
against the documented legacy checks before considering any real rerun.
Confirm that its declared run/revision and compact grids agree and that every
CSV count/file lineage check passes. Record the legacy byte-provenance limit.
Do not reinterpret current survivors or change scientific thresholds.

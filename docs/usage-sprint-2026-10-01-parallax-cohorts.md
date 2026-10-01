# PARALLAX4 cohort integrity sprint — 2026-10-01

Branch: `codex/usage-sprint-2026-10-01-parallax-cohorts`.
Base: `5b62e3bbf2db3bfca944970c1b59213f7db4ea17`.
Scope: reducer/input integrity, offline tests and PARALLAX4 workflows.
No science runs, result rewrites, threshold changes or main merges.

Confirmed defects: injection tables pooled other shard widths; equal-length
histograms pooled different time grids/runs/revisions; reduce-only without a
source run purged inputs and overwrote the prior reduction with NO_SHARD_OUTPUTS.

The reducer now selects companions by accepted canonical sweep tags, checks
uniform time_edges, dimensions, finite nonnegative histograms, run/revision,
processed-file uniqueness, event counts and CSV file lineage before aggregate
writes. Incoherent or corrupt input raises CohortError and preserves existing
reduce.json and events_AB.csv. Missing shards and partial work remain usable
with existing DEGRADED bookkeeping. Missing legacy pixel tables disable the
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

A dedicated read-only PARALLAX4 offline workflow checks changed code and the
existing science suite plus new cohort/workflow regressions. First implementation `b77989d7b977da38f3ed48a43dd96d6aed76e193` passed the
[dedicated offline check](https://github.com/trimcrae/Seti/actions/runs/36937276729)
(lint and PARALLAX4 tests). Independent review identified a missing-count legacy
pixel-coverage gap and malformed-error metadata that could raise after an
aggregate write. Follow-ups require conservative pixel coverage, separate
all-transit totals from detector searched-source counts, validate error-list
metadata and finish report computation before aggregate writes. Final-head receipts
remain pending until the follow-up runner reports them.

## Next task

After review/CI, inspect one retained historic sweep artifact cohort read-only
against the documented legacy checks before considering any real rerun.
Confirm that its declared run/revision and compact grids agree and that every
CSV count/file lineage check passes. Record the legacy byte-provenance limit.
Do not reinterpret current survivors or change scientific thresholds.

# SETI usage sprint handoff

Draft PR: https://github.com/trimcrae/Seti/pull/22  
Branch: `codex/usage-sprint-2026-10-01-seti-scheduler`  
Reviewed implementation commit: `2e6e9e80dbee5bb45bd6d5ecbcb19f02064dcb09`  
Base: `5b62e3bbf2db3bfca944970c1b59213f7db4ea17`

## Delivered

TOCSIN_ZTF's two daily cron expressions now imply a 12-hour workflow cadence
and three-hour grace. Distinct schedule unions, older mature missed slots,
catch-up identity, and alert identity are consistent across fresh-slot rollover.
The actual-run drift guard, unknown-history refusal, new-schedule bounds, and
existing ledger fallback remain covered. Numeric steps such as `20/15` expand
to minutes 20, 35, and 50.

Independent source review found and corrected a duplicate-alert rollover defect,
then reported no remaining source blockers. All four implementation files were
read back at the exact implementation commit and matched the submitted content.

## Validation receipt and remaining gate

This session has GitHub connector tools but no shell or Python runtime.
**Python tests and Ruff have not been executed in this session.**

GitHub received real CI events for the implementation commit, actor
`trimcrae`: push run
https://github.com/trimcrae/Seti/actions/runs/36935989534 and pull-request run
https://github.com/trimcrae/Seti/actions/runs/36935996349.
Both were pending at the last inspection. Pending is not a pass.

Before advancing the PR, inspect the newest runs and their exact head SHA;
obtain job conclusions and failing logs if any. Targeted offline commands:

```bash
ruff check src/seti/cronwatch.py src/seti/alerts.py tests/test_cronwatch_schedules.py
pytest tests/test_cronwatch.py tests/test_cronwatch_schedules.py tests/test_alerts.py
```

The existing CI also runs repository-wide lint, the full offline test suite,
sample analysis, and manuscript compilation. Keep honest receipt links in the PR.

## Next prioritized science task: PARALLAX4 cohort integrity

The newest research scoreboard describes a catalogue-scale grey-dip search
with surviving sources still requiring binary/blend rejection and Gaia DR4
photocentre evidence. No technosignature candidate is claimed.

Fix reduction cohort validation before trusting completeness or veto counts:

1. In `src/seti/parallax4/run.py:stage_reduce`, validate consistent shard
   counts, unique valid shard IDs, run/code/config provenance, identical
   `time_edges`, and histogram lengths **before** pooling. Currently
   equal-length arrays can be summed despite incompatible time bins.
2. Select injection CSVs, event CSVs, and pixel-epoch CSVs from that same
   validated cohort. The injection glob currently includes every
   `injections_s*of*.csv`, even when `n_shards_expected` filters sweep records
   and event tables to one cohort.
3. In `.github/workflows/parallax4.yml`, refuse `stage=reduce` with blank
   `reduce_only_run_id` before purging or downloading. That input combination
   currently purges inherited shard outputs, downloads neither current nor
   prior-run artifacts, and writes `NO_SHARD_OUTPUTS`.
4. Add offline regressions for mixed cohort counts, duplicate/missing shards,
   equal-length shifted bin edges, incompatible provenance, unrelated injection
   files, valid complete/partial cohorts, and the missing reducer input. Verify
   the existing explicit degraded-partial behavior remains honest.
5. Only then trace surviving PARALLAX4 targets through the existing deep vet and
   positive controls, using committed evidence or an explicitly available
   execution environment. Do not turn unreached data or failed controls into a
   null result.

Use a separate branch and reviewable draft PR for that science integrity task.
No merge, deployment, paid search, external notification, or discovery is part
of this handoff.

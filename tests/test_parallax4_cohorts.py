"""Offline regressions for PARALLAX4 input cohorts and reducer preservation."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seti.parallax4 import cohorts as H
from seti.parallax4 import run as R

WF = Path(".github/workflows/parallax4.yml")


def _shard(out, index=0, width=2, *, run_id="9001", sha="revision-a",
           edges=None, events=True, pixel=True, inject=True, manifest=False):
    name = f"cdn-{index}.csv.gz"
    counts = {"n_events": int(events), "n_AB": int(events), "n_transits_ok_g": 100,
              "n_sources": 1, "n_searched": 1}
    record = {"stage": "sweep", "shard": index, "n_shards": width,
              "run_id": run_id, "git_sha": sha, "verdict": "SHARD_COMPLETE",
              "time_edges": edges or [1600.0, 2800.0, 2],
              "hist_transits": [50.0, 50.0], "hist_grey_episodes": [int(events), 0],
              "hist_all_episodes": [int(events), 0], "counts": counts,
              "done": [name], "n_files_done": 1, "n_files_planned": 1, "errors": []}
    tag = f"s{index}of{width}"
    if events:
        row = dict.fromkeys(R.EVENT_COLS, np.nan)
        row.update(source_id=5_000_000_000_000_000_000 + index, kind="DIP", tier="A",
                   t_peak=1700.0, rms_out_of_episode_g=0.01, delta_g=0.2,
                   n_transits_episode=2, file=name)
        for prefix in ("events", "eventsAB"):
            pd.DataFrame([row]).to_csv(out / f"{prefix}_{tag}.csv", index=False)
    if pixel:
        pd.DataFrame({"hp": [1], "tb": [17000], "n": [100]}).to_csv(
            out / f"pixel_epochs_{tag}.csv", index=False)
    if inject:
        pd.DataFrame([{"file": name, "kind": "grey", "snr": 20.0, "pre_existing_event": False,
                       "recovered": True, "tier": "A", "grey_class": "GREY",
                       "colour_testable": True}]).to_csv(out / f"injections_{tag}.csv", index=False)
    if manifest:
        record.update(artifact_schema=1, output_sha256=H.output_manifest(out, tag))
    path = out / f"sweep_{tag}.json"
    path.write_text(json.dumps(record))
    return path, record


def _save(path, record):
    path.write_text(json.dumps(record))


def _preserved(out, call, match):
    targets = (out / "reduce.json", out / "events_AB.csv")
    for target in targets:
        target.write_text(f"existing {target.name}\n")
    before = [target.read_bytes() for target in targets]
    with pytest.raises(H.CohortError, match=match):
        call()
    assert [target.read_bytes() for target in targets] == before


def test_expected_width_selects_one_cohort_for_every_csv(tmp_path):
    _shard(tmp_path)
    _shard(tmp_path, index=1, width=3, run_id="old-run")
    rep = R.stage_reduce({}, tmp_path, n_shards_expected=2, run_id_expected="9001")
    assert rep["counts"]["n_events"] == rep["n_AB"] == 1
    assert rep["n_all_events_for_local_veto"] == 1
    assert rep["completeness"]["n_trials"] == 1
    assert rep["degraded"] == ["shards_missing:1/2"]
    assert set(rep["input_cohort"]["ignored"]) == {
        f"{prefix}_s1of3.{suffix}" for prefix, suffix in [
            ("sweep", "json"), ("events", "csv"), ("eventsAB", "csv"),
            ("pixel_epochs", "csv"), ("injections", "csv")]}
    assert pd.read_csv(tmp_path / "events_AB.csv")["source_id"].tolist() == [5_000_000_000_000_000_000]


def test_inferred_cohort_rejects_mixed_widths(tmp_path):
    _shard(tmp_path)
    _shard(tmp_path, index=1, width=3)
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path), "mixed n_shards")


@pytest.mark.parametrize(("key", "value", "match"), [
    ("run_id", "different-run", "mixed run_id"),
    ("git_sha", "different-revision", "mixed git_sha"),
    ("time_edges", [1601.0, 2801.0, 2], "mismatched time_edges"),
    ("time_edges", [1600.0, 2800.0, 2.5], "nonnegative integer"),
    ("time_edges", [2800.0, 1600.0, 2], "invalid histogram grid"),
    ("hist_transits", [100.0], "invalid hist_transits"),
    ("hist_grey_episodes", [float("nan"), 0], "invalid hist_grey_episodes"),
    ("hist_all_episodes", [-1, 0], "invalid hist_all_episodes"),
    ("shard", 0, "filename/metadata mismatch"),
    ("done", ["cdn-0.csv.gz"], "multiple shards"),
    ("errors", True, "errors must be a list"),
    ("errors", {"file": "broken"}, "errors must be a list"),
])
def test_invalid_cohort_fails_before_overwriting(tmp_path, key, value, match):
    _shard(tmp_path)
    path, record = _shard(tmp_path, index=1)
    record[key] = value
    _save(path, record)
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path, n_shards_expected=2), match)


def test_explicit_source_run_rejects_even_one_wrong_run(tmp_path):
    _shard(tmp_path, run_id="wrong")
    _preserved(tmp_path, lambda: R.stage_reduce(
        {}, tmp_path, n_shards_expected=2, run_id_expected="9001"), "wrong run_id")


def test_explicit_source_without_records_preserves_aggregate(tmp_path):
    _preserved(tmp_path, lambda: R.stage_reduce(
        {}, tmp_path, n_shards_expected=2, run_id_expected="9001"), "no accepted sweep")


@pytest.mark.parametrize("prefix", H.PREFIXES)
def test_new_manifest_binds_every_csv_before_overwriting(tmp_path, prefix):
    _shard(tmp_path, manifest=True)
    with (tmp_path / f"{prefix}_s0of2.csv").open("a") as stream:
        stream.write("\n")
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path, n_shards_expected=2),
               "checksum mismatch")


def test_new_manifest_cohort_is_accepted(tmp_path):
    _shard(tmp_path, manifest=True)
    rep = R.stage_reduce({}, tmp_path, n_shards_expected=2)
    assert rep["n_AB"] == 1 and rep["input_cohort"]["legacy_shards"] == []


def test_legacy_record_checks_lineage_without_claiming_byte_provenance(tmp_path):
    _shard(tmp_path, pixel=False)
    rep = R.stage_reduce({}, tmp_path, n_shards_expected=2)
    assert rep["input_cohort"]["legacy_shards"] == ["s0of2"]
    assert "unbound bytes" in rep["input_cohort"]["legacy_csv_provenance"]
    assert rep["local_veto"].startswith("NOT_RUN")
    assert rep["n_AB"] == 1


@pytest.mark.parametrize("prefix", ["events", "eventsAB", "injections"])
def test_legacy_rows_from_another_cdn_file_are_rejected(tmp_path, prefix):
    _shard(tmp_path)
    path = tmp_path / f"{prefix}_s0of2.csv"
    frame = pd.read_csv(path)
    frame["file"] = "old-unprocessed-file.csv.gz"
    frame.to_csv(path, index=False)
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path, n_shards_expected=2),
               "outside recorded processed files")


@pytest.mark.parametrize("prefix", ["events", "eventsAB"])
def test_legacy_csv_count_must_match_record(tmp_path, prefix):
    _shard(tmp_path)
    path = tmp_path / f"{prefix}_s0of2.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame], ignore_index=True).to_csv(path, index=False)
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path, n_shards_expected=2), "row count")


def test_orphan_csv_is_not_loaded_as_cohort_evidence(tmp_path):
    _shard(tmp_path)
    (tmp_path / "injections_s1of2.csv").write_text("file\nstale.csv.gz\n")
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path, n_shards_expected=2),
               "no accepted sweep record")


def test_partial_shards_and_zero_work_shards_remain_valid(tmp_path):
    path, record = _shard(tmp_path, events=False, pixel=False, inject=False)
    record.update(verdict="SHARD_PARTIAL_TIME_BUDGET", n_files_planned=3)
    _save(path, record)
    path, empty = _shard(tmp_path, index=1, events=False, pixel=False, inject=False)
    empty.update(done=[], n_files_done=0, n_files_planned=0,
                 counts={"n_events": 0, "n_AB": 0, "n_transits_ok_g": 0},
                 hist_transits=[0, 0])
    _save(path, empty)
    rep = R.stage_reduce({}, tmp_path, n_shards_expected=2)
    assert rep["n_shards_found"] == 2 and rep["n_AB"] == 0
    assert rep["degraded"] == ["files_incomplete:1/3"]


def test_partial_pixel_denominator_is_never_combined_with_all_events(tmp_path):
    _shard(tmp_path)
    _shard(tmp_path, index=1, pixel=False)
    rep = R.stage_reduce({}, tmp_path, n_shards_expected=2)
    assert rep["n_all_events_for_local_veto"] == 2
    assert rep["local_veto"].startswith("NOT_RUN")


def test_missing_histogram_provenance_fails_closed(tmp_path):
    path, record = _shard(tmp_path)
    del record["run_id"]
    _save(path, record)
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path, n_shards_expected=2), "missing run_id")


def test_cli_explicit_one_shard_filters_other_widths(tmp_path):
    _shard(tmp_path, width=1)
    _shard(tmp_path, index=1, width=3, run_id="old-run")
    assert R.main(["--stage", "reduce", "--shards", "1", "--source-run-id", "9001",
                   "--out-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "reduce.json").read_text())["n_shards_found"] == 1


def test_no_data_does_not_continue_to_vet_or_replace_existing_aggregate(tmp_path, monkeypatch):
    (tmp_path / "reduce.json").write_text("previous reduction")
    monkeypatch.setattr(R, "stage_vet", lambda *a, **k: pytest.fail("vet after failed reduction"))
    rep = R.run("reduce,vet", out_dir=tmp_path, conf={})
    assert rep["verdict"] == "NO_SHARD_OUTPUTS"
    assert (tmp_path / "reduce.json").read_text() == "previous reduction"


def _plan(monkeypatch, *, stage="reduce", source="", shards="24"):
    step = yaml.safe_load(WF.read_text())["jobs"]["plan"]["steps"][0]
    script = step["run"].split("\n", 1)[1].rsplit("\nPY", 1)[0]
    for key, value in {"PARALLAX_STAGE": stage, "PARALLAX_SOURCE_RUN": source,
                       "PARALLAX_SHARDS": shards, "GITHUB_RUN_ID": "9100"}.items():
        monkeypatch.setenv(key, value)
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(script, "parallax4-plan", "exec"), {})
    return dict(line.split("=", 1) for line in output.getvalue().splitlines())


def test_reduce_without_source_run_is_rejected_before_purge(monkeypatch):
    with pytest.raises(SystemExit, match="requires reduce_only_run_id"):
        _plan(monkeypatch)
    jobs = yaml.safe_load(WF.read_text())["jobs"]
    assert "plan" in jobs["reduce"]["needs"]
    assert "needs.plan.result == 'success'" in jobs["reduce"]["if"]


@pytest.mark.parametrize(("stage", "source", "expected", "reduce_inputs"), [
    ("reduce", "9001", "9001", "true"),
    ("all", "", "9100", "true"),
    ("sweep", "", "9100", "true"),
    ("dr4", "", "9100", "false"),
    ("watchlist", "", "9100", "false"),
])
def test_workflow_plan_selects_source_and_protects_unrelated_stages(
        monkeypatch, stage, source, expected, reduce_inputs):
    outputs = _plan(monkeypatch, stage=stage, source=source, shards="2")
    assert outputs["source_run_id"] == expected and outputs["reduce_inputs"] == reduce_inputs
    assert outputs["n_shards"] == "2"
    assert json.loads(outputs["matrix"]) == {"include": [{"shard": 0}, {"shard": 1}]}


@pytest.mark.parametrize(("source", "shards", "match"), [
    ("", "0", "between 1 and 256"),
    ("", "257", "between 1 and 256"),
    ("abc", "2", "positive GitHub run ID"),
    ("0", "2", "positive GitHub run ID"),
])
def test_bad_workflow_inputs_are_rejected(monkeypatch, source, shards, match):
    with pytest.raises(SystemExit, match=match):
        _plan(monkeypatch, stage="all", source=source, shards=shards)


def test_workflow_binds_source_run_and_commits_only_success():
    steps = yaml.safe_load(WF.read_text())["jobs"]["reduce"]["steps"]
    purge = next(step for step in steps if step.get("name", "").startswith("Purge"))
    assert "reduce_inputs == 'true'" in purge["if"]
    reduce = next(step for step in steps if step.get("name", "").startswith("Reduce, vet"))
    assert '--source-run-id "$PARALLAX_SOURCE_RUN"' in reduce["run"]
    assert 'PARALLAX_SOURCE_RUN' in reduce["env"]
    commit = next(step for step in steps if step.get("name") == "Commit results")
    assert commit["if"] == "success()"


def test_unknown_legacy_transit_count_cannot_hide_missing_pixel_table(tmp_path):
    _shard(tmp_path)
    path, record = _shard(tmp_path, index=1, pixel=False)
    del record["counts"]["n_transits_ok_g"]
    _save(path, record)
    rep = R.stage_reduce({}, tmp_path, n_shards_expected=2)
    assert rep["local_veto"].startswith("NOT_RUN")


def test_truncated_legacy_pixel_table_fails_before_overwrite(tmp_path):
    _shard(tmp_path)
    pd.DataFrame({"hp": [1], "tb": [17000], "n": [50]}).to_csv(
        tmp_path / "pixel_epochs_s0of2.csv", index=False)
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path, n_shards_expected=2),
               "pixel transit total")


def test_legacy_pixels_can_include_unsearched_sources_and_outside_grid_transits(tmp_path):
    path, record = _shard(tmp_path)
    record["counts"]["n_transits_ok_g"] = 70
    _save(path, record)
    pd.DataFrame({"hp": [1, 1], "tb": [17000, 29000], "n": [100, 10]}).to_csv(
        tmp_path / "pixel_epochs_s0of2.csv", index=False)
    rep = R.stage_reduce({}, tmp_path, n_shards_expected=2)
    assert rep["local_veto"] == "RUN"


def test_new_exact_pixel_total_catches_loss_even_above_legacy_lower_bound(tmp_path):
    path, record = _shard(tmp_path)
    record["counts"]["n_pixel_transits"] = 110
    _save(path, record)
    _preserved(tmp_path, lambda: R.stage_reduce({}, tmp_path, n_shards_expected=2),
               "pixel transit total")


def test_inferred_width_still_reports_missing_shards(tmp_path):
    _shard(tmp_path)
    rep = R.stage_reduce({}, tmp_path)
    assert rep["n_shards_found"] == 1
    assert rep["degraded"] == ["shards_missing:1/2"]
    assert rep["verdict"].startswith("DEGRADED")

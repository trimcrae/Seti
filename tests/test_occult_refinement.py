"""Actual OCCULT orchestration witnesses; numerical fits and clock are adapters.

These synthetic fixtures test production assess_event, its gates and the
empirical reducer. They are not physical injection recovery or sky evidence.
"""
from __future__ import annotations

import copy
import gzip
import json

import numpy as np
import pytest

from seti.occult import detect as D
from seti.occult import run as R


def _assess(monkeypatch, mode):
    n = 80
    datasets = [
        {"name": "A_I", "site": "KMTA", "band": "I"},
        {"name": "C_V", "site": "KMTC", "band": "V"},
    ]
    ev = D.Event("synthetic-orchestration", 7960 + np.arange(n), np.ones(n),
                 np.full(n, 0.1), np.arange(n) % 2, datasets)
    def fit(kind, chi2):
        return D.Fit(kind, 8000.0, 25.0, 0.2, 0.002,
                     None if kind == "fspl" else 0.6, chi2, np.ones(2),
                     np.zeros(2), np.ones(n), 8)
    null = fit("fspl", 500.0)
    monkeypatch.setattr(D, "fit_fspl_clean",
                        lambda *_: (ev, ev.e, null, {"peak_snr": 20.0}))
    monkeypatch.setattr(D, "expected_map", lambda *_: {})
    monkeypatch.setattr(D, "scan_wing", lambda *_, **__: {"all": ([], [])})
    monkeypatch.setattr(D, "scan_hole", lambda *_, **__: {"all": ([], [])})
    monkeypatch.setattr(D, "_scan_maxima", lambda *_: [])
    monkeypatch.setattr(D, "hollow_centre", lambda *_: None)
    monkeypatch.setattr(D, "occultation_term", lambda *_: np.ones(n))
    monkeypatch.setattr(D, "side_tests", lambda *_, **__: {
        "rise": {"snr": 10.0}, "fall": {"snr": 10.0},
        "alpha_z": 0.0, "edges_agree": True})
    clock = [0.0]
    monkeypatch.setattr(D.time, "time", lambda: clock[0])
    calls = []
    def refine(_ev, _null, _regime, _uth, _conf, kind, _starts):
        calls.append(kind)
        if mode == "timeout_before_anti" and kind == "occult":
            clock[0] = 601.0
        if mode == "timeout_partial_anti" and kind == "anti":
            clock[0] = 601.0
        if mode in ("anti_missing", "occult_missing") and kind == mode.split("_")[0]:
            return None
        if mode == "anti_raises" and kind == "anti":
            raise ValueError("synthetic missing comparison")
        if mode == "anti_nonfinite" and kind == "anti":
            return fit(kind, float("nan"))
        if mode == "partial_anti_missing" and kind == "anti" and calls.count("anti") == 1:
            return None
        value = 480.0 if mode in ("complete_positive", "partial_anti_missing") else 500.0
        result = fit(kind, value if kind == "anti" else 0.0)
        if mode == "complete_not_converged":
            result.converged = False
        return result
    monkeypatch.setattr(D, "_refine", refine)
    rec = D.assess_event(ev, {"seed_rho_l": [0.6, 0.7], "unit_budget_s": 600.0})
    return rec, calls



def _complete(anti=0.0):
    return {
        "refinement": {
            k: {"planned": 2, "attempted": 2, "usable": 2, "finished": True}
            for k in ("occult", "anti")
        },
        "refinement_status": "COMPLETE", "dchi2_anti": anti,
    }


@pytest.mark.parametrize("mode,anti_calls", [
    ("timeout_before_anti", 0), ("timeout_partial_anti", 1),
    ("anti_missing", 2), ("anti_raises", 2), ("anti_nonfinite", 2),
    ("partial_anti_missing", 2), ("occult_missing", 2),
])
def test_incomplete_production_search_is_refused(monkeypatch, mode, anti_calls):
    rec, calls = _assess(monkeypatch, mode)
    assert calls.count("anti") == anti_calls
    assert rec["tier"] == D.TIER_INCOMPLETE
    assert rec["refinement_status"] == D.refinement_status(rec) == "INCOMPLETE"
    assert rec["rejections"] == ["REFINEMENT_COMPARISON_INCOMPLETE"]
    if mode in ("anti_missing", "anti_raises", "anti_nonfinite", "timeout_before_anti"):
        assert rec["dchi2_anti"] is None
    if mode == "partial_anti_missing":
        assert rec["dchi2_anti"] == 20.0  # observed partial result, not a completed null
    threshold = R.empirical_threshold([rec], 50.0)
    assert threshold["status"] == "UNMEASURED" and threshold["n_null"] == 0
    assert threshold["threshold"] == 50.0  # policy floor retained, never called measured


@pytest.mark.parametrize("mode,anti", [("complete_zero", 0.0), ("complete_positive", 20.0)])
def test_completed_native_numeric_results_remain_measured(monkeypatch, mode, anti):
    rec, calls = _assess(monkeypatch, mode)
    assert calls == ["occult", "occult", "anti", "anti"]
    assert rec["tier"] == D.TIER_CANDIDATE
    assert D.refinement_status(rec) == "COMPLETE"
    assert rec["dchi2_anti"] == anti
    assert rec["refinement"]["anti"] == {
        "planned": 2, "attempted": 2, "usable": 2, "finished": True}
    threshold = R.empirical_threshold([rec], 50.0)
    assert threshold["status"] == "MEASURED" and threshold["n_null"] == 1
    assert threshold["anti_max"] == anti


@pytest.mark.parametrize("anti", [None, float("nan"), float("inf"), True, "0"])
def test_unknown_or_invalid_anti_is_not_numeric_zero(anti):
    rec = _complete(anti)
    assert D.refinement_status(rec) == "INCOMPLETE"
    assert R.empirical_threshold([dict(rec, tier=D.TIER_NO_OCC)], 50)["n_null"] == 0


@pytest.mark.parametrize("change", [
    {"planned": 0}, {"attempted": 1}, {"usable": 1}, {"finished": False},
    {"planned": True}, {"usable": -1}, {"attempted": 3},
])
def test_incomplete_execution_receipts_cannot_establish_a_null(change):
    rec = _complete()
    rec["refinement"]["anti"].update(change)
    assert D.refinement_status(rec) == "INCOMPLETE"


def test_legacy_and_budget_states_are_reported_separately():
    legacy = {"tier": D.TIER_CANDIDATE, "dchi2": 900, "dchi2_anti": 0.0}
    incomplete = dict(_complete(), tier=D.TIER_NO_OCC, budget_exceeded=True)
    measured = dict(_complete(), tier=D.TIER_NO_OCC)
    assert D.refinement_status(legacy) == "UNKNOWN_LEGACY"
    threshold = R.empirical_threshold([legacy, incomplete, measured], 50)
    assert threshold["n_null"] == 1 and threshold["anti_max"] == 0.0
    assert threshold["status"] == "PARTIAL"
    assert threshold["n_unknown_legacy"] == threshold["n_incomplete"] == 1
    assert "refinement" not in legacy  # no retroactive completion invented


def _write_assess_input(root, rows):
    sdir = root / "shards"
    sdir.mkdir()
    (sdir / "screen_s0of1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (sdir / "inj_s0of1.jsonl").write_text("")
    (root / "controls.json").write_text(json.dumps({"passed": True, "verdict": "CONTROLS_PASS"}))
    with gzip.open(root / "catalog.json.gz", "wt") as fh:
        json.dump({"units": [{"unit": r["unit"]} for r in rows]}, fh)


def test_reducer_withholds_legacy_and_partial_candidates_and_preserves_receipts(tmp_path):
    rows = [
        dict(_complete(), unit="completed-zero", tier=D.TIER_CANDIDATE, dchi2=500),
        {"unit": "legacy", "tier": D.TIER_CANDIDATE, "dchi2": 900, "dchi2_anti": 0.0},
        dict(_complete(70), unit="measured-null", tier=D.TIER_NO_OCC, dchi2=1),
    ]
    partial = dict(_complete(20), unit="partial", tier=D.TIER_CANDIDATE, dchi2=800)
    partial["refinement"]["anti"].update(attempted=1, usable=1, finished=False)
    partial["refinement_status"] = "INCOMPLETE"
    rows.append(partial)
    before = copy.deepcopy(rows)
    _write_assess_input(tmp_path, rows)
    out = R.stage_assess(tmp_path, {}, 1)
    assert out["verdict"] == "ANTI_NULL_PARTIAL -- no candidate promotion"
    assert out["null_threshold"]["status"] == "PARTIAL"
    assert out["null_threshold"]["n_null"] == 2
    assert out["null_threshold"]["threshold"] == 70.0
    assert out["candidates"] == []
    assert out["funnel"]["complete_gate_passing_withheld_for_partial_null"] == 1
    assert out["funnel"]["unverified_gate_passing_withheld"] == 2
    with gzip.open(tmp_path / "screen_compact.jsonl.gz", "rt") as fh:
        compact = [json.loads(line) for line in fh]
    by = {r["unit"]: r for r in compact}
    assert by["partial"]["refinement"] == partial["refinement"]
    assert "refinement" not in by["legacy"]
    assert rows == before


def test_unmeasured_null_never_promotes_legacy_candidate(tmp_path):
    rows = [{"unit": "legacy", "tier": D.TIER_CANDIDATE, "dchi2": 900, "dchi2_anti": 0.0}]
    _write_assess_input(tmp_path, rows)
    out = R.stage_assess(tmp_path, {}, 1)
    assert out["null_threshold"]["status"] == "UNMEASURED"
    assert out["null_threshold"]["n_null"] == 0
    assert out["null_threshold"]["threshold"] == 50.0
    assert out["verdict"].startswith("ANTI_NULL_UNMEASURED")
    assert out["candidates"] == []


def _control():
    return {
        **_complete(), "name": "synthetic", "class": "finite_source_single_lens",
        "reached": True, "tier": D.TIER_NO_OCC, "sites": ["KMTA", "KMTC"],
        "injections": [{**_complete(), "expected_dchi2": 900, "recovered": True}] * 3,
    }


@pytest.mark.parametrize("which", ["named", "strong-injection"])
def test_unverified_controls_cannot_be_verified_negatives(which):
    record = _control()
    assert R.controls_verdict([record], {}, 1)["passed"] is True
    if which == "named":
        record.pop("refinement")
    else:
        record["injections"] = [dict(expected_dchi2=900, recovered=True)]
    verdict = R.controls_verdict([record], {}, 1)
    assert verdict["verdict"] == "CONTROLS_DEGRADED_REFINEMENT_INCOMPLETE"
    assert verdict["passed"] is False


def test_candidate_depth_alone_does_not_make_an_injection_recovered():
    from seti.occult import inject as I
    old = {"tier": D.TIER_CANDIDATE, "occult": {"rho_l": 0.6}}
    assert I.recovered(old, 0.6) is False
    good = dict(_complete(), **old)
    assert I.recovered(good, 0.6) is True


def test_execution_receipt_does_not_change_native_optimizer_success_policy(monkeypatch):
    rec, calls = _assess(monkeypatch, "complete_not_converged")
    assert len(calls) == 4
    assert rec["tier"] == D.TIER_CANDIDATE
    assert D.refinement_status(rec) == "COMPLETE"


def test_screen_and_control_keep_actual_producer_completion_metadata(monkeypatch, tmp_path):
    rec, _ = _assess(monkeypatch, "complete_zero")
    ev = D.Event("metadata-wire", [1, 2, 3], [1, 1, 1], [.1, .1, .1],
                 [0, 0, 0], [{"name": "A", "site": "KMTA", "band": "I"}])
    unit = {"unit": "synthetic", "kmt": None, "ogle": None}
    monkeypatch.setattr(R, "unit_event", lambda *_: (ev, {}))
    monkeypatch.setattr(D, "assess_event", lambda *_: rec)
    monkeypatch.setattr(R, "save_lc", lambda *_: "synthetic-adapter")
    monkeypatch.setattr(R.INJ, "injection_trials", lambda *_: [])
    monkeypatch.setitem(R._WORKER, "session", None)
    screen, _ = R.screen_unit(None, unit, {}, {"inject_every": 0}, 1, tmp_path)
    control = R._control_one(("finite_source_single_lens", {"name": "synthetic"}, unit, {}, []))
    for row in (screen, control):
        assert row["refinement"] == rec["refinement"]
        assert row["refinement_status"] == "COMPLETE"
        assert D.refinement_status(row) == "COMPLETE"


def test_injection_trials_keep_completion_and_unmeasured_anti(monkeypatch):
    from seti.occult import inject as I
    rec, _ = _assess(monkeypatch, "anti_missing")
    ev = D.Event("injection-wire", [1, 2, 3], [1, 1, 1], [.1, .1, .1],
                 [0, 0, 0], [{"name": "A", "site": "KMTA", "band": "I"}])
    null = D.Fit("fspl", 2, 25, .2, .002, None, 500, np.ones(1), np.zeros(1), np.ones(3), 6)
    monkeypatch.setattr(D, "fit_fspl_clean", lambda *_: (ev, ev.e, null, {}))
    monkeypatch.setattr(D, "expected_dchi2", lambda *_: 1000.0)
    monkeypatch.setattr(D, "assess_event", lambda *_, **__: rec)
    monkeypatch.setattr(I, "inject", lambda *_: ev)
    trial = I.injection_trials(ev, {}, None, [.6])[0]
    assert trial["refinement"] == rec["refinement"]
    assert trial["dchi2_anti"] is None
    assert trial["refinement_status"] == "INCOMPLETE"
    assert trial["recovered"] is False


def test_full_eligible_null_coverage_keeps_candidate_positive(tmp_path):
    rows = [
        dict(_complete(), unit="candidate", tier=D.TIER_CANDIDATE, dchi2=500),
        dict(_complete(70), unit="null", tier=D.TIER_NO_OCC, dchi2=1),
    ]
    _write_assess_input(tmp_path, rows)
    out = R.stage_assess(tmp_path, {}, 1)
    assert out["null_threshold"]["status"] == "MEASURED"
    assert out["null_threshold"]["n_null"] == out["null_threshold"]["n_eligible"] == 2
    assert out["verdict"] == "CANDIDATES_TO_VET:1"
    assert [r["unit"] for r in out["candidates"]] == ["candidate"]

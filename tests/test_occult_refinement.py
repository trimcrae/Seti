"""Actual OCCULT orchestration witnesses; numerical fits and clock are adapters.

These synthetic fixtures test production assess_event, its gates and the
empirical reducer. They are not physical injection recovery or sky evidence.
"""
from __future__ import annotations

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
        if mode == "anti_missing" and kind == "anti":
            return None
        return fit(kind, 500.0 if kind == "anti" else 0.0)
    monkeypatch.setattr(D, "_refine", refine)
    rec = D.assess_event(ev, {"seed_rho_l": [0.6, 0.7], "unit_budget_s": 600.0})
    return rec, calls


@pytest.mark.parametrize("mode,anti_calls", [
    ("timeout_before_anti", 0),
    ("timeout_partial_anti", 1),
    ("anti_missing", 2),
])
def test_baseline_incomplete_comparison_can_be_promoted(monkeypatch, mode, anti_calls):
    rec, calls = _assess(monkeypatch, mode)
    assert calls.count("anti") == anti_calls
    assert rec["dchi2"] == 500.0
    assert rec["dchi2_anti"] == 0.0
    assert rec["rejections"] == []
    assert rec["tier"] == D.TIER_CANDIDATE
    if mode.startswith("timeout"):
        assert rec["budget_exceeded"] is True
    threshold = R.empirical_threshold([rec], 50.0)
    assert threshold["n_null"] == 1
    assert threshold["threshold"] == 50.0
    print(f"BASELINE production orchestration: {mode}, anti calls={anti_calls}, "
          "incomplete candidate and synthetic-zero null admitted")


def test_baseline_completed_measured_zero_is_a_positive_control(monkeypatch):
    rec, calls = _assess(monkeypatch, "complete_zero")
    assert calls == ["occult", "occult", "anti", "anti"]
    assert rec["dchi2_anti"] == 0.0
    assert rec["tier"] == D.TIER_CANDIDATE
    assert R.empirical_threshold([rec], 50.0)["n_null"] == 1

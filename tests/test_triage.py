"""Offline tests for the observed-frame candidate triage (spectra/triage.py)."""

from __future__ import annotations

import json

import pandas as pd

from seti.spectra.reject import air_to_vacuum
from seti.spectra.triage import (
    excluded_bandwidth_frac,
    known_lines_vacuum,
    triage_candidates,
    triage_run,
)


def test_air_to_vacuum_reference_values():
    # H-alpha: 6562.79 air -> 6564.61 vacuum; [O I] sky: 5577.34 air -> 5578.89.
    assert abs(float(air_to_vacuum(6562.79)) - 6564.61) < 0.05
    assert abs(float(air_to_vacuum(5577.34)) - 5578.89) < 0.05
    # Red end: Ca II 8542.09 air -> 8544.44 vacuum (a ~2.4 A shift).
    assert abs(float(air_to_vacuum(8542.09)) - 8544.44) < 0.05


def _cand(spec_id, wave, sig=12.0, mode="emission"):
    return {"spec_id": spec_id, "wavelength": wave, "significance": sig,
            "ra": 10.0, "dec": 10.0, "search_mode": mode}


def test_triage_flags_known_line_despite_wrong_catalog_rv():
    # He I 5875.6 air = 5877.2 vacuum.  With a wrong catalogue RV the in-funnel
    # cut misses it; the observed-frame velocity window must not.
    he = float(air_to_vacuum(5875.62))
    df = pd.DataFrame([
        _cand("a", he + 0.8),          # ~40 km/s off the line: known_line_window
        _cand("b", 5389.5),            # far from every known line: survives
    ])
    out = triage_candidates(df, v_window_kms=300.0)
    assert out.loc[0, "triage_verdict"] == "known_line_window"
    assert out.loc[1, "triage_verdict"] == "survives"
    assert out.loc[0, "nearest_line_dv_kms"] < 300.0


def test_triage_flags_cross_run_recurrence_and_duplicates():
    df = pd.DataFrame([
        _cand("a", 7007.0), _cand("b", 7007.4, mode="absorption"),
        _cand("c", 7008.1),                       # 3 distinct spectra: recurrent
        _cand("d", 7420.0), _cand("d", 7420.0),   # exact duplicate row
    ])
    out = triage_candidates(df, recur_tol=3.0, recur_min=3)
    assert list(out["triage_verdict"][:3]) == ["recurrent_across_runs"] * 3
    assert out.loc[3, "triage_verdict"] == "survives"
    assert out.loc[4, "triage_verdict"] == "duplicate"
    assert out.loc[0, "n_spectra_at_wavelength"] == 3


def test_excluded_bandwidth_is_minor():
    frac = excluded_bandwidth_frac(known_lines_vacuum(), 300.0)
    # The velocity windows must leave the overwhelming majority of the band open,
    # otherwise the triage would be trading completeness for purity.
    assert 0.0 < frac < 0.35


def test_triage_run_end_to_end(tmp_path):
    lines = known_lines_vacuum()
    (tmp_path / "results" / "spectra").mkdir(parents=True)
    (tmp_path / "results" / "spectra_absorption").mkdir(parents=True)
    pd.DataFrame([_cand("a", float(lines[10])), _cand("b", 5389.5, sig=20.0)]).drop(
        columns="search_mode").to_csv(
        tmp_path / "results" / "spectra" / "laser_candidates.csv", index=False)
    pd.DataFrame([_cand("c", 6234.5, sig=9.0)]).drop(columns="search_mode").to_csv(
        tmp_path / "results" / "spectra_absorption" / "laser_candidates.csv",
        index=False)

    summary = triage_run(tmp_path)
    assert summary["n_input"] == 3
    assert summary["n_priority_targets"] == 2
    assert summary["verdict_counts"]["known_line_window"] == 1

    out_dir = tmp_path / "results" / "spectra_triage"
    pri = pd.read_csv(out_dir / "priority_targets.csv")
    # Ranked by significance, both modes present, verdicts all "survives".
    assert list(pri["spec_id"]) == ["b", "c"]
    assert set(pri["triage_verdict"]) == {"survives"}
    saved = json.loads((out_dir / "summary.json").read_text())
    assert saved["n_priority_targets"] == 2

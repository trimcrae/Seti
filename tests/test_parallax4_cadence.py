"""Independent arithmetic and fixed-cadence synthetic controls, never sky calibration."""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from seti.parallax4 import cadence_control as CC
from seti.parallax4 import deepvet as DV

FIXTURE = Path(CC.FIXTURE_PATH)


def independent_accept(times, shift, p, duration, tol):
    # Distance to nearest primary/secondary center in TIME, not the
    # implementation's centered-phase expression or interval arithmetic.
    q = p / 2
    remainder = np.remainder(np.asarray(times) + shift, q)
    return np.minimum(remainder, q - remainder) <= duration / 2 + tol * p


def test_real_fixture_pins_mask_times_and_proxy_groups():
    meta, records = CC.load_fixture(FIXTURE)
    assert meta["fixture_bytes"] == 19396
    assert meta["source_id"] == "1457486023639239296"
    assert meta["raw_rows"] == meta["eligible_rows"] == meta["eligible_unique_epochs"] == 65
    assert meta["vrej_g_eligible_diagnostic_count"] == 5
    assert meta["bad_g_count"] == 0
    assert meta["time_frame"] == {
        "refposition": "BARYCENTER", "timescale": "TCB", "timeorigin": "2455197.5"}
    assert all(isinstance(x["transit_id"], str) for x in records)
    times = [x["t_gaia_tcb_d"] for x in records]
    assert times[0] == 1693.257506198131
    assert times[-1] == 2612.3500838080454
    assert sum(np.diff(times) < 0.1) == 22
    groups = CC.sampling_blocks(times)
    assert [len(g) for g in groups] == [
        7, 4, 5, 1, 2, 1, 2, 1, 2, 3, 4, 4, 3, 2, 1, 2, 2, 1, 4, 4, 3, 1, 2, 2, 2]
    assert len(groups) == 25
    assert all(g[0] in times for g in groups)


def test_altered_fixture_refuses_before_interpretation(tmp_path):
    path = tmp_path / "altered.vot"
    path.write_bytes(FIXTURE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="PINNED_FIXTURE_SHA256_MISMATCH"):
        CC.load_fixture(path)


def test_grouping_deduplicates_sorts_and_keeps_exact_boundary():
    assert CC.sampling_blocks([2.01, 1.0, 0.0, 1.0]) == [[0.0, 1.0], [2.01]]
    assert CC.sampling_blocks([0.0, 0.5, 1.0], 0.0) == [[0.0], [0.5], [1.0]]


@pytest.mark.parametrize("times", [[], [np.nan], [np.inf], [[0.0]], [0.0, np.nan]])
def test_invalid_time_vectors_refuse(times):
    with pytest.raises(ValueError):
        CC.sampling_blocks(times)
    with pytest.raises(ValueError):
        CC.common_shift_null(times, 1.0, 0.1, 0.03)


@pytest.mark.parametrize("gap", [-1.0, np.nan, np.inf])
def test_invalid_sampling_gap_refuses(gap):
    with pytest.raises(ValueError):
        CC.sampling_blocks([0.0], gap)


@pytest.mark.parametrize("k", [0, -1, 5, 1.5, True])
def test_invalid_subset_label_count_refuses(k):
    with pytest.raises(ValueError):
        CC.subset_null([0.0, 0.1, 0.2, 0.3], k, 1.0, 0.1, 0.03)


def test_subset_duplicate_opportunities_refuse():
    with pytest.raises(ValueError, match="unique"):
        CC.subset_null([0.0, 0.0], 1, 1.0, 0.1, 0.03)


@pytest.mark.parametrize(("period", "duration", "tol", "t0"), [
    (0.0, 0.1, 0.03, 0.0), (1.0, 0.0, 0.03, 0.0),
    (1.0, 0.1, -0.03, 0.0), (1.0, 0.1, 0.03, np.nan),
])
def test_invalid_ephemeris_refuses(period, duration, tol, t0):
    with pytest.raises(ValueError):
        CC.common_shift_null([0.0], period, duration, tol, t0)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_exact_subset_inclusion_exclusion_against_independent_brute_force(k):
    times = [0.0, 0.075, 0.17, 0.25, 0.5, 0.67]
    result = CC.subset_null(times, k, 1.0, 0.1, 0.03)
    subsets = list(itertools.combinations(times, k))
    count = sum(any(independent_accept(subset, 0.0, p, 0.1, 0.03).all()
                    for p in (1.0, 2.0, 0.5)) for subset in subsets)
    assert int(result["union_subset_count"]) == count
    assert int(result["all_subset_count"]) == math.comb(len(times), k)
    assert result["probability"] == count / len(subsets)


def test_nontrivial_common_shift_intersection_and_wrap_hand_result():
    result = CC.common_shift_null([0.0, 0.075], 1.0, 0.1, 0.03)
    assert [x["probability"] for x in result["alias_trials"]] == pytest.approx(
        [0.17, 0.145, 0.22])
    assert result["probability"] == pytest.approx(0.34)
    # P trial intersects across the origin and preserves both wrapped pieces.
    intervals = result["alias_trials"][0]["accepted_shift_intervals_d"]
    assert intervals[0][0] == 0.0
    assert intervals[-1][1] == 2.0


@pytest.mark.parametrize("times", [
    [0.0], [0.0, 0.075], [0.17, 0.49, 1.08], [0.0, 0.15, 0.31], [0.3] * 10,
])
def test_common_shift_measure_against_independent_deterministic_quadrature(times):
    result = CC.common_shift_null(times, 1.0, 0.1, 0.03)
    grid_n = 200000
    shifts = (np.arange(grid_n) + 0.5) * 2 / grid_n
    accepted = []
    for trial in result["alias_trials"]:
        mask = np.ones(grid_n, dtype=bool)
        for t in times:
            mask &= independent_accept([t], shifts, trial["period_d"], 0.1, 0.03)
        accepted.append(mask)
        assert trial["probability"] == pytest.approx(mask.mean(), abs=16 / grid_n)
    assert result["probability"] == pytest.approx(
        np.any(accepted, axis=0).mean(), abs=48 / grid_n)


def test_one_common_shift_keeps_perfectly_correlated_slots_correlated():
    single = CC.common_shift_null([0.0], 1.0, 0.1, 0.03)
    repeated = CC.common_shift_null([0.0] * 10, 1.0, 0.1, 0.03)
    assert single["probability"] == repeated["probability"] == pytest.approx(0.64)
    assert repeated["slot_count"] == 10
    assert repeated["unique_epoch_count"] == 1
    iid, _ = DV._ztf_phase_chance_bound(1.0, 0.1, 0.03, 10)
    assert iid < repeated["probability"]


def test_common_shift_order_translation_and_overlap_saturation():
    times = [0.0, 0.075, 1.0]
    base = CC.common_shift_null(times, 1.0, 0.1, 0.03)
    reverse = CC.common_shift_null(times[::-1], 1.0, 0.1, 0.03)
    moved = CC.common_shift_null([t + 0.37 for t in times], 1.0, 0.1, 0.03)
    assert base["probability"] == reverse["probability"]
    assert base["probability"] == pytest.approx(moved["probability"])
    saturated = CC.common_shift_null(times, 1.0, 0.5, 0.0)
    assert saturated["probability"] == 1.0
    assert saturated["alias_trials"][0]["probability"] == 1.0
    assert saturated["alias_trials"][2]["probability"] == 1.0


def test_report_binds_distinct_selections_and_conditional_results():
    report = CC.build_report(FIXTURE)
    discrete = report["uniform_subset_null"]
    assert discrete["accepted_counts"] == [8, 4, 16]
    assert discrete["union_subset_count"] == "8008"
    assert discrete["all_subset_count"] == "3268760"
    assert discrete["probability"] == pytest.approx(0.0024498586620002693)
    chronological = report["chronological_common_shift_null"]
    positive = report["positive_control"]
    assert chronological["times_d"] == report["sampling_proxy"]["block_starts_d"][:10]
    assert chronological["probability"] == 0.0
    assert all(x["probability"] == 0.0 for x in chronological["alias_trials"])
    assert positive["times_d"] != chronological["times_d"]
    assert positive["available_matching_count"] == 16
    assert report["phase_selected_positive_shift_sensitivity"]["probability"] == pytest.approx(
        0.23324861109904305)
    assert report["iid_uniform_comparison"]["union_bound"] == pytest.approx(
        0.0014570756577869113)
    assert report["result_kind"] == "CONDITIONAL_SYNTHETIC_CONTROL"
    assert report["sampling_proxy"]["not_recovered_dip_episodes"] is True


def test_real_cadence_phase_selected_positive_and_chronological_negative(monkeypatch):
    # Fixed synthetic ephemeris isolates production phasing, deliberately no
    # BLS/real-flux recovery claim. The real cadence supplies only test times.
    class FixedBLS:
        def __init__(self, *args, **kwargs):
            pass

        def power(self, *args, **kwargs):
            n = 128
            return SimpleNamespace(
                power=np.r_[100.0, np.zeros(n - 1)], period=np.ones(n),
                transit_time=np.zeros(n), duration=np.full(n, 0.1),
                depth=np.full(n, 0.2))

    monkeypatch.setattr("astropy.timeseries.BoxLeastSquares", FixedBLS)
    z = pd.DataFrame({"mjd": DV.GAIA_T0_MJD + np.linspace(0, 40, 80),
                      "mag": 15.0, "magerr": 0.01, "filtercode": "zr", "oid": "123456789"})
    report = CC.build_report(FIXTURE)
    positive = DV.ztf_eclipse_test(z, report["positive_control"]["times_d"])
    negative = DV.ztf_eclipse_test(z, report["chronological_common_shift_null"]["times_d"])
    assert positive["ztf_class"] == "ZTF_ECLIPSING_PHASED"
    assert positive["ztf_period_d"] == 0.5
    assert negative["ztf_class"] == "ZTF_PERIODIC_NOT_PHASED"
    assert positive["phase_p_chance"] == negative["phase_p_chance"]
    assert positive["phase_p_chance_is_empirical"] is False


def test_insufficient_predeclared_positive_stops_without_tuning(monkeypatch):
    monkeypatch.setattr(CC, "load_fixture", lambda path: ({}, [
        {"t_gaia_tcb_d": 0.17 + 2 * i} for i in range(10)]))
    with pytest.raises(ValueError, match="INSUFFICIENT_PREDECLARED_CONTROL_OPPORTUNITIES"):
        CC.build_report(FIXTURE)

"""Independent mean/scatter oracles and native bindings for one CCD control."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.parallax4 import ccd_control as C
from seti.parallax4 import epochs as E


def _ccds():
    return pd.DataFrame({
        "source_id": [1] * 4, "transit_id": [2] * 4, "ccd_slot": [0, 1, 2, 3],
        "x_al": [0.0, 0.0, 0.0, 1e6], "sx_al": [1.0, 2.0, 3.0, 0.01],
        "used": [True, True, True, False], "ccd_flags": [8192, 832, 0, 6],
        "t": [3000.0] * 4, "bary_ok": [True] * 4, "theta": [0.3] * 4,
        "pf_al": [0.5] * 4, "excess_noise": [0.0] * 4,
        "g_mag": [12.0] * 4, "dist_ci": [0.0] * 4, "cf_al": [0.0] * 4,
        "multipeak": [False] * 4, "blended": [False] * 4,
    })


@pytest.mark.parametrize("delta", [0.0, 3.0, -3.0])
def test_unequal_weight_mean_scatter_oracle_and_agis_mask(delta):
    ccd = _ccds()
    reference, discordant, diagnostic = C.matched_centroids(ccd, {(1, 2): delta})
    weights = np.array([1.0, 0.25, 1 / 9])
    total = float(weights.sum())
    # Independent analytical mean and scatter; excluded tiny-error CCD contributes nothing.
    expected_chi2 = delta ** 2 * (total ** 2 / weights[0] - total)
    for arm in (reference, discordant):
        for key in set(ccd.columns) - {"x_al"}:
            pd.testing.assert_series_equal(arm[key], ccd[key])
        assert arm.loc[3, "x_al"] == 1e6
    ref, dis = E.collapse_transits(reference).iloc[0], E.collapse_transits(discordant).iloc[0]
    assert ref["n_ccd"] == dis["n_ccd"] == 3
    assert ref["x_al"] == pytest.approx(delta, abs=2e-15)
    assert dis["x_al"] == pytest.approx(delta, abs=2e-15)
    assert ref["sx_al"] == dis["sx_al"] == pytest.approx(1 / np.sqrt(total))
    assert ref["chi2_ccd"] == pytest.approx(0.0, abs=1e-28)
    assert dis["chi2_ccd"] == pytest.approx(expected_chi2, abs=1e-13)
    assert diagnostic[0]["analytic_discordant_chi2_ccd"] == pytest.approx(expected_chi2)


def test_native_slot_choice_and_collapse_do_not_depend_on_row_order():
    ccd = _ccds()
    ref, dis, _ = C.matched_centroids(ccd, {(1, 2): 3.0})
    shuffled = ccd.iloc[[2, 3, 1, 0]]
    ref2, dis2, diagnostic = C.matched_centroids(shuffled, {(1, 2): 3.0})
    assert diagnostic[0]["selected_native_slot"] == 0
    pd.testing.assert_frame_equal(ref.sort_index(), ref2.sort_index())
    pd.testing.assert_frame_equal(dis.sort_index(), dis2.sort_index())
    np.testing.assert_allclose(E.collapse_transits(dis)["chi2_ccd"],
                               E.collapse_transits(dis2)["chi2_ccd"], rtol=1e-15)


def test_one_eligible_ccd_has_no_within_transit_discordance_information():
    ccd = _ccds()
    ccd["used"] = [False, True, False, False]
    ref, dis, diagnostic = C.matched_centroids(ccd, {(1, 2): 3.0})
    pd.testing.assert_frame_equal(ref, dis)
    row = E.collapse_transits(dis).iloc[0]
    assert row["n_ccd"] == 1 and row["chi2_ccd"] == 0.0
    assert diagnostic[0]["analytic_discordant_chi2_ccd"] == 0.0


def test_invalid_or_ambiguous_control_bindings_are_refused():
    with pytest.raises(ValueError, match="MISSING_SYNTHETIC_TRANSIT_OFFSET"):
        C.matched_centroids(_ccds(), {})
    with pytest.raises(ValueError, match="NONFINITE_SYNTHETIC_TRANSIT_OFFSET"):
        C.matched_centroids(_ccds(), {(1, 2): np.nan})
    ccd = _ccds()
    ccd.loc[1, "ccd_slot"] = 0
    with pytest.raises(ValueError, match="AMBIGUOUS_NATIVE_CCD_SLOT"):
        C.matched_centroids(ccd, {(1, 2): 3.0})


@pytest.fixture(scope="module")
def native_control():
    return C.build_report()


def test_exact_native_ccd_bindings_and_analytical_collapse(native_control):
    report = native_control
    source = report["source"]
    assert source["zip_git_blob"] == "b3d51a1506ca1d20f304a0895c294b4a9908039f"
    assert source["retained_joint_transits"] == 41
    assert source["retained_native_ccd_slots"] == 410
    assert source["retained_eligible_ccds"] == 365
    assert source["processing_flag_bits_interpreted"] is False
    rows = report["native_bound_transit_rows"]
    assert len(rows) == 41 and len({row["transit_id"] for row in rows}) == 41
    for row in rows:
        assert row["source_id"] == "1457486023639239296"
        assert row["native_ccd_slots"] == list(range(10))
        mask = np.asarray(row["native_eligible_mask"], bool)
        assert int(mask.sum()) in (8, 9)
        errors = np.asarray(row["native_centroid_error_al_mas"])[mask]
        weights = 1 / errors ** 2
        expected_mean = row["synthetic_transit_offset_mas"]
        expected_error = 1 / np.sqrt(weights.sum())
        expected_chi2 = expected_mean ** 2 * (
            weights.sum() ** 2 / weights[0] - weights.sum())
        assert row["reference_x_al_mas"] == pytest.approx(expected_mean, abs=5e-13)
        assert row["discordant_x_al_mas"] == pytest.approx(expected_mean, abs=5e-13)
        assert row["reference_sx_al_mas"] == pytest.approx(expected_error, rel=1e-14)
        assert row["discordant_sx_al_mas"] == row["reference_sx_al_mas"]
        assert row["reference_chi2_ccd"] < 1e-20
        assert row["discordant_chi2_ccd"] == pytest.approx(expected_chi2, rel=2e-14, abs=1e-12)


def test_matched_native_transit_fits_have_same_bound_without_coherence_test(native_control):
    ref, dis = native_control["reference_fit"], native_control["discordant_fit"]
    assert ref["verdict"] == dis["verdict"] == "BLEND_UNRESOLVED"
    for key in ("D_ra_mas", "D_dec_mas", "sigma_D_max", "ul95_mas", "jitter_mas"):
        assert dis[key] == pytest.approx(ref[key], rel=1e-10, abs=1e-10)
    assert ref["D_ra_mas"] == pytest.approx(300.0, abs=1e-8)
    assert ref["D_dec_mas"] == pytest.approx(-200.0, abs=1e-8)
    comparison = native_control["comparison"]
    assert comparison["max_abs_transit_mean_difference_mas"] < 1e-12
    assert comparison["max_abs_formal_error_difference_mas"] == 0.0
    assert comparison["discordant_chi2_ccd_max"] > 1e5

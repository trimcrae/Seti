"""Pinned native-geometry algebra witness; synthetic, never a sky detection."""
from __future__ import annotations

import numpy as np

import pytest

from seti.parallax4 import photocentre as P
from seti.parallax4 import spatial_control as SC


def test_native_geometry_and_analytic_target_null_directions():
    report = SC.build_report()
    source = report["source"]
    assert source["source_id"] == "1457486023639239296"
    assert source["zip_bytes"] == 625651
    assert source["phot_bytes"] == 19396
    assert source["astrometry_release"] == "Gaia DR4_RC3"
    assert source["raw_phot_transits"] == 65
    assert source["native_transit_id_intersection"] == 63
    assert source["retained_joint_transits"] >= 40
    assert source["max_abs_join_time_difference_s"] < 5
    rows = report["native_geometry_rows"]
    assert len({row["transit_id"] for row in rows}) == len(rows)
    assert all(row["source_id"] == source["source_id"] for row in rows)
    assert all(np.isfinite(row["theta_rad"]) and row["sx_al_mas"] > 0 for row in rows)
    drift = report["affine_drift_control"]
    assert drift["min_total_flux_ratio"] > 0
    assert drift["hidden_D_mas"] == [300.0, -200.0]
    assert drift["max_abs_analytic_null_row_residual"] < 1e-12
    assert drift["max_abs_hidden_parameterization_difference_mas"] < 1e-10
    assert drift["synthetic_whitened_design_rank"] < drift["synthetic_whitened_design_columns"]
    # No production verdict is preordained; the first actual report supplies it.
    assert drift["cancelled_hidden_blend_result"]["n_transits"] == len(rows)


def test_affine_native_geometry_refuses_confident_target_and_blend():
    report = SC.build_report()
    drift = report["affine_drift_control"]
    for name in ("cancelled_hidden_blend_result", "sky_only_injection_result"):
        result = drift[name]
        assert result["verdict"] == "INSUFFICIENT"
        assert result["D_estimability"] == "NON_ESTIMABLE"
        assert result["astrometric_design"]["rank"] == 6
        assert not any(result["astrometric_design"]["coefficient_estimable"][5:7])
        assert "ul95_mas" not in result and "D_mas" not in result
        assert "sigma_D_max" not in result and "p_D" not in result


def test_near_affine_flux_has_honest_large_uncertainty():
    result = SC.build_report()["independent_flux_controls"]["near_affine_zero_centroid"]["result"]
    assert result["D_estimability"] == "ESTIMABLE"
    assert result["astrometric_design"]["rank"] == 8
    assert result["sigma_D_max"] > P.PhotocentreConfig().sigma_d_max_mas
    assert result["verdict"] == "INSUFFICIENT"
    assert result["ul95_mas"] > P.PhotocentreConfig().ul_on_target_mas


@pytest.mark.parametrize("name", [
    "nondegenerate_sky_positive", "nuisance_only_parallax_null_positive",
])
def test_native_geometry_positive_and_nuisance_only_null_recover_sky_vector(name):
    result = SC.build_report()["independent_flux_controls"][name]["result"]
    assert result["verdict"] == "BLEND_UNRESOLVED"
    assert result["D_estimability"] == "ESTIMABLE"
    assert result["D_ra_mas"] == pytest.approx(300.0, abs=1e-8)
    assert result["D_dec_mas"] == pytest.approx(-200.0, abs=1e-8)
    if name == "nuisance_only_parallax_null_positive":
        assert result["astrometric_design"]["rank"] == 7
        assert result["astrometric_design"]["coefficient_estimable"][2] is False
        assert result["parallax_mas"] is None
    else:
        assert result["astrometric_design"]["rank"] == 8


@pytest.mark.parametrize("even", [False, True])
def test_affine_null_algebra_survives_time_translation_and_even_median(even):
    _, joint = SC.load_geometry()
    if even:
        joint = joint.iloc[:-1].copy()
    joint["t_yr"] += 7.0
    before = joint.copy(deep=True)
    drift = SC.drift_case(joint)
    assert drift["max_abs_analytic_null_row_residual"] < 1e-12
    assert drift["max_abs_hidden_parameterization_difference_mas"] < 1e-10
    assert drift["cancelled_hidden_blend_result"]["D_estimability"] == "NON_ESTIMABLE"
    assert joint.equals(before)  # native flux/centroid remain untouched


def _full_rank_problem():
    rng = np.random.default_rng(9127)
    X = rng.normal(size=(40, 8)) * np.array([1, 0.1, 1000, 0.001, 100, 0.2, 0.3, 4])
    y = X @ np.arange(1, 9) + rng.normal(size=40) * 0.03
    s = np.linspace(0.2, 0.5, len(X))
    return X, y, s


def test_direct_svd_matches_independent_weighted_qr_fit_and_covariance():
    X, y, s = _full_rank_problem()
    beta, cov, _, _, diagnostic = P._weighted_svd(X, y, s)
    A, b = X / s[:, None], y / s
    q, r = np.linalg.qr(A, mode="reduced")
    expected_beta = np.linalg.solve(r, q.T @ b)
    inverse_r = np.linalg.inv(r)
    expected_cov = inverse_r @ inverse_r.T
    np.testing.assert_allclose(beta, expected_beta, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(cov, expected_cov, rtol=1e-8, atol=1e-8)
    assert diagnostic["rank"] == 8
    assert all(diagnostic["coefficient_estimable"])


def test_direct_svd_maps_mixed_column_units_back_to_physical_coefficients():
    X, y, s = _full_rank_problem()
    beta, cov, _, _, diagnostic = P._weighted_svd(X, y, s)
    factors = np.array([1e-6, 1e6, 1e-3, 1e3, 1e-2, 1e2, 10, 0.1])
    changed_beta, changed_cov, _, _, changed = P._weighted_svd(X * factors, y, s)
    np.testing.assert_allclose(changed_beta * factors, beta, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(changed_cov * factors[:, None] * factors[None, :],
                               cov, rtol=1e-8, atol=1e-8)
    assert changed["coefficient_estimable"] == diagnostic["coefficient_estimable"]


def test_nuisance_only_null_target_covariance_matches_reduced_qr_oracle():
    X, y, s = _full_rank_problem()
    X[:, 2] = 0.0
    beta, cov, _, _, diagnostic = P._weighted_svd(X, y, s)
    indices = [0, 1, 3, 4, 5, 6, 7]
    q, r = np.linalg.qr(X[:, indices] / s[:, None], mode="reduced")
    expected_beta = np.linalg.solve(r, q.T @ (y / s))
    inverse_r = np.linalg.inv(r)
    expected_cov = inverse_r @ inverse_r.T
    np.testing.assert_allclose(beta[indices], expected_beta, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(cov[np.ix_(indices, indices)], expected_cov, rtol=1e-8, atol=1e-8)
    assert diagnostic["rank"] == 7
    assert not diagnostic["coefficient_estimable"][2]
    assert all(diagnostic["coefficient_estimable"][5:7])


def test_null_projector_includes_missing_directions_when_rows_less_than_columns():
    X = np.c_[np.eye(3), np.zeros((3, 2))]
    _, _, _, _, diagnostic = P._weighted_svd(X, np.arange(3), np.ones(3))
    assert diagnostic["rank"] == 3
    assert diagnostic["coefficient_estimable"] == [True, True, True, False, False]


@pytest.mark.parametrize("bad_sigma", [0.0, -0.1, np.nan, np.inf])
def test_solver_refuses_invalid_whitening_weights(bad_sigma):
    X, y, s = _full_rank_problem()
    s[0] = bad_sigma
    with pytest.raises(ValueError, match="positive uncertainties"):
        P._weighted_svd(X, y, s)

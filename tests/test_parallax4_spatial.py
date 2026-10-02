"""Pinned native-geometry algebra witness; synthetic, never a sky detection."""
from __future__ import annotations

import numpy as np

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
    assert drift["native_whitened_design_rank"] < drift["native_whitened_design_columns"]
    # No production verdict is preordained; the first actual report supplies it.
    assert drift["cancelled_hidden_blend_result"]["n_transits"] == len(rows)

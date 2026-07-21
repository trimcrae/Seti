"""Offline unit tests for the LHS 1140 signature-sweep scorers."""

from __future__ import annotations

import numpy as np

from seti.lhs1140.dossier import (
    LHS1140,
    PLANETS,
    inventory_summary,
    neighbor_companion_scan,
    neighbor_ir_excess_scan,
)


def test_anchor_and_planets_defined():
    assert LHS1140["name"] == "LHS 1140"
    assert 60 < LHS1140["parallax_mas"] < 70          # ~15 pc
    names = {p["name"] for p in PLANETS}
    assert {"LHS 1140 b", "LHS 1140 c"} <= names


def test_neighbor_ir_excess_flags_only_real_excess():
    # A bare M-dwarf photosphere: WISE colours ~ 0 -> no flag.
    clean = {"source_id": 1, "ra": 10.0, "dec": -15.0,
             "w1mpro": 9.0, "w2mpro": 8.98, "w3mpro": 8.95, "w4mpro": 8.9,
             "w1sigmpro": 0.02, "w2sigmpro": 0.02,
             "w3sigmpro": 0.03, "w4sigmpro": 0.05}
    # A strong warm-dust / waste-heat excess in W3/W4 at high significance.
    excess = {"source_id": 2, "ra": 11.0, "dec": -15.0,
              "w1mpro": 9.0, "w2mpro": 8.5, "w3mpro": 7.5, "w4mpro": 6.5,
              "w1sigmpro": 0.02, "w2sigmpro": 0.02,
              "w3sigmpro": 0.03, "w4sigmpro": 0.05}
    # No WISE data at all -> no data, no flag.
    nodata = {"source_id": 3, "ra": 12.0, "dec": -15.0}
    out = neighbor_ir_excess_scan([clean, excess, nodata])
    assert out["n_sources"] == 3
    assert out["n_with_wise"] == 2
    assert out["n_ir_excess"] == 1
    assert out["flagged"][0]["source_id"] == 2


def test_neighbor_companion_scan_flags_high_ruwe_and_nss():
    single = {"source_id": 1, "ruwe": 1.05, "astrometric_excess_noise": 0.05,
              "astrometric_excess_noise_sig": 20.0, "ipd_frac_multi_peak": 0.0,
              "non_single_star": 0}
    binary = {"source_id": 2, "ruwe": 3.2, "astrometric_excess_noise": 2.0,
              "astrometric_excess_noise_sig": 50.0, "ipd_frac_multi_peak": 15.0,
              "non_single_star": 1}
    out = neighbor_companion_scan([single, binary])
    assert out["n_with_astrometry"] == 2
    assert out["n_companion_flag"] == 1
    assert out["flagged"][0]["source_id"] == 2


def test_companion_scan_ignores_tiny_high_sigma_excess():
    # A well-measured single star: tiny excess noise at very high sigma is NOT a
    # companion (the amplitude gate) -- must not flag.
    row = {"source_id": 9, "ruwe": 1.0, "astrometric_excess_noise": 0.1,
           "astrometric_excess_noise_sig": 100.0, "ipd_frac_multi_peak": 0.0,
           "non_single_star": 0}
    out = neighbor_companion_scan([row])
    assert out["n_companion_flag"] == 0


def test_inventory_detects_atmosphere_capable_spectroscopy():
    records = [
        {"instrument_name": "NIRISS", "dataproduct_type": "spectrum",
         "obs_collection": "JWST", "em_min": 0.6e-6, "em_max": 2.8e-6,
         "t_exptime": 21000.0},
        {"instrument_name": "TESS", "dataproduct_type": "timeseries",
         "obs_collection": "TESS", "t_exptime": 1200.0},
        {"instrument_name": "STIS", "dataproduct_type": "spectrum",
         "obs_collection": "HST", "em_min": 0.2e-6, "em_max": 1.0e-6},
    ]
    inv = inventory_summary(records)
    assert inv["n_observations"] == 3
    assert inv["n_spectroscopic"] == 2
    assert inv["atmosphere_capable_spectroscopy"] is True
    assert inv["per_instrument"].get("NIRISS") == 1
    assert inv["wavelength_span"]["min_um"] < 0.5
    assert inv["wavelength_span"]["max_um"] > 2.5


def test_inventory_photometry_only_is_honest():
    records = [
        {"instrument_name": "TESS", "dataproduct_type": "timeseries",
         "obs_collection": "TESS"},
        {"instrument_name": "WISE", "dataproduct_type": "image",
         "obs_collection": "WISE"},
    ]
    inv = inventory_summary(records)
    assert inv["n_spectroscopic"] == 0
    assert inv["atmosphere_capable_spectroscopy"] is False
    assert "no atmosphere-capable" in inv["note"]


def test_inventory_empty():
    inv = inventory_summary([])
    assert inv["n_observations"] == 0
    assert inv["atmosphere_capable_spectroscopy"] is False
    assert inv["wavelength_span"] is None

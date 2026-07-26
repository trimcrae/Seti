"""Offline unit tests for the LHS 1140 signature-sweep scorers."""

from __future__ import annotations

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
    # A real warm-dust / waste-heat excess lighting up the star-dominated bands
    # (W1-W2 hot band AND W1-W3 warm band), a physically bounded SED.
    excess = {"source_id": 2, "ra": 11.0, "dec": -15.0,
              "w1mpro": 9.0, "w2mpro": 8.5, "w3mpro": 8.0, "w4mpro": 7.6,
              "w1sigmpro": 0.02, "w2sigmpro": 0.02,
              "w3sigmpro": 0.03, "w4sigmpro": 0.05}
    # No WISE data at all -> no data, no flag.
    nodata = {"source_id": 3, "ra": 12.0, "dec": -15.0}
    out = neighbor_ir_excess_scan([clean, excess, nodata])
    assert out["n_sources"] == 3
    assert out["n_with_wise"] == 2
    assert out["n_ir_excess"] == 1
    assert out["flagged"][0]["source_id"] == 2


def test_neighbor_ir_excess_rejects_w4_artifact_and_blend():
    # W4-only "excess" (photospheric W1-W2/W1-W3, huge W4) = the AllWISE W4
    # faint-source/cirrus artefact -> needs_vetting, NOT a candidate.
    w4_artifact = {"source_id": 10, "ra": 10.0, "dec": -15.0,
                   "w1mpro": 10.0, "w2mpro": 9.85, "w3mpro": 9.7, "w4mpro": 7.5,
                   "w1sigmpro": 0.02, "w2sigmpro": 0.02,
                   "w3sigmpro": 0.05, "w4sigmpro": 0.05}
    # Negative W1-W2 (W2 brighter than W1) = blend/bad photometry, not a star.
    blend = {"source_id": 11, "ra": 11.0, "dec": -15.0,
             "w1mpro": 10.0, "w2mpro": 10.15, "w3mpro": 8.0, "w4mpro": 5.0,
             "w1sigmpro": 0.02, "w2sigmpro": 0.02,
             "w3sigmpro": 0.03, "w4sigmpro": 0.05}
    out = neighbor_ir_excess_scan([w4_artifact, blend])
    assert out["n_ir_excess"] == 0
    assert out["n_needs_vetting"] == 2
    whys = " ".join(v["vetting"] for v in out["needs_vetting"])
    assert "W4-only" in whys and "blend" in whys


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
    # MAST reports em_min/em_max in nanometres.
    records = [
        {"instrument_name": "NIRISS", "dataproduct_type": "spectrum",
         "obs_collection": "JWST", "em_min": 600.0, "em_max": 2800.0,
         "t_exptime": 21000.0},
        {"instrument_name": "TESS", "dataproduct_type": "timeseries",
         "obs_collection": "TESS", "t_exptime": 1200.0},
        {"instrument_name": "STIS", "dataproduct_type": "spectrum",
         "obs_collection": "HST", "em_min": 200.0, "em_max": 1000.0},
    ]
    inv = inventory_summary(records)
    assert inv["n_observations"] == 3
    assert inv["n_spectroscopic"] == 2
    assert inv["atmosphere_capable_spectroscopy"] is True
    assert inv["per_instrument"].get("NIRISS") == 1
    assert inv["wavelength_span"]["min_um"] == 0.2      # 200 nm -> 0.2 um
    assert inv["wavelength_span"]["max_um"] == 2.8      # 2800 nm -> 2.8 um


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


# --- biosignature detectability -------------------------------------------
from seti.lhs1140.biosignature import (  # noqa: E402
    biosignature_detectability,
    biosignature_verdict,
    feature_amplitude_ppm,
    scale_height,
    surface_gravity,
    transit_depth_ppm,
    transits_to_detect,
)

# LHS 1140 b (Cadieux+2024).
_LHS1140B = dict(rp_earth=1.730, mp_earth=5.60, rs_sun=0.2159, teq_k=226.0,
                 jmag=9.612, t_in_hours=2.0, resolution=50.0)


def test_surface_gravity_and_depth_match_lhs1140b():
    g = surface_gravity(_LHS1140B["mp_earth"], _LHS1140B["rp_earth"])
    assert 17 < g < 20                       # super-Earth, ~18 m/s^2
    depth = transit_depth_ppm(_LHS1140B["rp_earth"], _LHS1140B["rs_sun"])
    assert 5000 < depth < 5800               # ~0.5% transit


def test_high_mu_atmosphere_has_smaller_scale_height():
    g = surface_gravity(_LHS1140B["mp_earth"], _LHS1140B["rp_earth"])
    h_h2 = scale_height(226.0, 2.3, g)
    h_n2 = scale_height(226.0, 28.0, g)
    assert h_h2 > 10 * h_n2                   # low-mu envelope ~12x puffier
    # And the feature amplitude scales with H.
    a_h2 = feature_amplitude_ppm(1.730, 0.2159, h_h2, 1.0)
    a_n2 = feature_amplitude_ppm(1.730, 0.2159, h_n2, 1.0)
    assert a_h2 > 10 * a_n2


def test_transits_to_detect_monotonic():
    # A larger feature needs fewer transits; a vanishing one needs infinity.
    n_big = transits_to_detect(40.0, 25.0, 6, target_sigma=5.0)
    n_small = transits_to_detect(4.0, 25.0, 6, target_sigma=5.0)
    assert n_small > n_big
    assert transits_to_detect(0.0, 25.0, 6) == float("inf")


def test_biosignature_verdict_not_detectable_for_secondary_atmosphere():
    budget = biosignature_detectability(_LHS1140B)
    # Under the expected N2 secondary atmosphere, biosignatures need many transits.
    n2 = budget["atmospheres"]["N2_secondary"]["gases"]
    assert n2["CH4"]["transits_for_detection"] > 10
    assert n2["O3"]["transits_for_detection"] > 20
    # Under a cleared H2 envelope they would be trivially reachable.
    h2 = budget["atmospheres"]["H2_rich_cleared"]["gases"]
    assert h2["CH4"]["transits_for_detection"] < 2
    # With only the ~2 transits actually observed, the answer is "not detectable".
    v = biosignature_verdict(budget, atmospheres_observed=1, transits_observed=2)
    assert v["answer"] == "BIOSIGNATURE_NOT_DETECTABLE_WITH_CURRENT_DATA"
    assert v["min_transits_for_any_biosignature"] > 2
    assert not v["reachable_biosignatures"]

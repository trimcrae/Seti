"""Offline unit tests for the LHS 1140 radio/optical technosignature-limit scorers.

No network: imports only :mod:`seti.seti_archive.limits`.
"""

from __future__ import annotations

import numpy as np

from seti.seti_archive.limits import (
    ARECIBO_EIRP_W,
    beacon_capability,
    eirp_limit,
    optical_seti_limit,
    parse_observation_inventory,
    radio_band,
)


# --- eirp_limit scaling ----------------------------------------------------
def test_eirp_scales_as_distance_squared():
    a = eirp_limit(10.0, 3.0, 300.0, 10.0)
    b = eirp_limit(10.0, 3.0, 300.0, 20.0)
    # Doubling distance quadruples the EIRP limit.
    assert np.isclose(b["eirp_w"] / a["eirp_w"], 4.0, rtol=1e-6)


def test_eirp_scales_inverse_sqrt_bandwidth_time():
    base = eirp_limit(10.0, 3.0, 300.0, 15.0)
    # 4x the bandwidth-time product -> half the EIRP limit.
    wider_bw = eirp_limit(10.0, 12.0, 300.0, 15.0)
    longer_t = eirp_limit(10.0, 3.0, 1200.0, 15.0)
    assert np.isclose(wider_bw["eirp_w"] / base["eirp_w"], 0.5, rtol=1e-6)
    assert np.isclose(longer_t["eirp_w"] / base["eirp_w"], 0.5, rtol=1e-6)
    # And linearly with SEFD, sqrt with npol.
    assert np.isclose(eirp_limit(20.0, 3.0, 300.0, 15.0)["eirp_w"]
                      / base["eirp_w"], 2.0, rtol=1e-6)


def test_eirp_ballpark_far_below_arecibo():
    # GBT-like: SEFD 10 Jy, 3 Hz channel, 300 s, 15 pc, 5 sigma.
    lim = eirp_limit(10.0, 3.0, 300.0, 15.0, snr=5.0, npol=2)
    # LHS 1140 is close enough that even this modest search constrains a beacon
    # far below the ~2e13 W Arecibo planetary radar.
    assert lim["eirp_w"] < ARECIBO_EIRP_W
    assert lim["eirp_arecibo_frac"] < 0.1
    # Physically sensible order of magnitude: ~1e10-1e11 W.
    assert 1e9 < lim["eirp_w"] < 1e12


def test_eirp_rejects_nonpositive_inputs():
    for bad in (dict(sefd=0.0), dict(bandwidth_hz=-1.0), dict(distance_pc=0.0)):
        kw = dict(sefd=10.0, bandwidth_hz=3.0, integration_s=300.0,
                  distance_pc=15.0)
        kw.update(bad)
        try:
            eirp_limit(**kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_beacon_capability_yardsticks():
    cap = beacon_capability(2e10)
    assert cap["rules_out_arecibo_class"] is True
    assert "below Arecibo" in cap["capability_class"]
    big = beacon_capability(1e18)
    assert big["rules_out_arecibo_class"] is False
    assert "Kardashev" in big["capability_class"]


# --- optical limit ---------------------------------------------------------
def test_optical_limit_bigger_aperture_is_deeper():
    small = optical_seti_limit(2.4, 15.0)
    big = optical_seti_limit(10.0, 15.0)
    # A larger collector reaches a fainter (smaller) EIRP limit.
    assert big["eirp_w"] < small["eirp_w"]
    # Beamed laser power is far below the isotropic EIRP (transmitter gain).
    assert big["laser_power_w"] < big["eirp_w"]
    assert big["eirp_w"] > 0 and big["min_photon_rate_hz"] > 0


def test_optical_limit_scales_distance_squared():
    a = optical_seti_limit(10.0, 10.0)
    b = optical_seti_limit(10.0, 20.0)
    assert np.isclose(b["eirp_w"] / a["eirp_w"], 4.0, rtol=1e-6)


# --- band assignment -------------------------------------------------------
def test_radio_band_edges():
    assert radio_band(1.5) == "L"
    assert radio_band(3.0) == "S"
    assert radio_band(0.8) == "UHF"
    assert radio_band(50.0) == "unknown"


# --- parse_observation_inventory ------------------------------------------
def test_parse_inventory_summarises_and_finds_best_limit():
    records = [
        # Two L-band configs at different depth; the lower-SEFD one wins the band.
        {"telescope": "MeerKAT", "band": "L", "sefd_jy": 7.0,
         "channel_bw_hz": 3.0, "integration_s": 300.0, "mjd": 60000.0,
         "confirmed": False},
        {"telescope": "Parkes", "band": "L", "sefd_jy": 36.0,
         "channel_bw_hz": 3.0, "integration_s": 300.0, "confirmed": True},
        # An S-band config, band derived from centre frequency.
        {"telescope": "GBT", "center_freq_mhz": 3000.0, "sefd_jy": 12.0,
         "channel_bw_hz": 3.0, "integration_s": 300.0, "confirmed": False},
        # A pointing with no SEFD/bandwidth: counts as coverage, no limit.
        {"telescope": "VLA", "band": "C", "confirmed": True},
    ]
    inv = parse_observation_inventory(records, distance_pc=15.0)
    assert inv["n_observations"] == 4
    assert inv["n_confirmed_observations"] == 2
    assert inv["n_with_eirp_limit"] == 3        # VLA row has no limit
    assert inv["facilities"]["MeerKAT"] == 1
    assert set(inv["bands_observed"]) >= {"L", "S", "C"}
    # MeerKAT (SEFD 7) beats Parkes (SEFD 36) in L band.
    best_L = inv["best_eirp_limit_per_band"]["L"]
    assert best_L["facility"] == "MeerKAT"
    # Overall best across bands is the deepest single config.
    assert inv["best_eirp_limit_overall"]["eirp_w"] == best_L["eirp_w"]


def test_parse_inventory_per_record_distance_overrides():
    near = parse_observation_inventory(
        [{"telescope": "GBT", "band": "L", "sefd_jy": 10.0, "channel_bw_hz": 3.0,
          "integration_s": 300.0, "distance_pc": 10.0}], distance_pc=100.0)
    # The per-record 10 pc must be used, not the 100 pc default.
    lim10 = eirp_limit(10.0, 3.0, 300.0, 10.0)
    assert np.isclose(near["best_eirp_limit_per_band"]["L"]["eirp_w"],
                      lim10["eirp_w"], rtol=1e-9)


def test_parse_inventory_empty():
    inv = parse_observation_inventory([])
    assert inv["n_observations"] == 0
    assert inv["best_eirp_limit_overall"] is None
    assert inv["best_eirp_limit_per_band"] == {}

"""RELAY: the supplementary rest-frequency ("magic line") test.

A relay that de-drifts its carrier so the RECEIVER sees a chosen rest line puts
that line, for Earth, at ``f_rest (1 + (v_TR - v_TE)/c)``.  The offset is a
velocity difference, so the test exists only where Gaia gives both radial
velocities — and a pointing without them must come back "not tested", never
"tested and did not match", or the trials count lies in the safe direction for
a claim.  These tests pin that, the arithmetic of the window, and the price of
the extra trials.
"""

from __future__ import annotations

import numpy as np
import pytest

from seti.relay.run import (
    C_KMS,
    _magic_chance,
    _magic_line_match,
    _magic_window_mhz,
    load_relay_config,
)

HI = 1420.405751768


@pytest.fixture(scope="module")
def conf():
    return load_relay_config()


def _w(lo, hi):
    return {"n_pairs": 3, "dv_offset_kms_min": lo, "dv_offset_kms_max": hi}


def test_the_window_is_the_offset_plus_the_rv_tolerance():
    lo, hi = _magic_window_mhz(HI, _w(0.0, 0.0), 5.0)
    assert np.isclose(hi - lo, 2 * HI * 5.0 / C_KMS, rtol=1e-9)
    assert np.isclose(0.5 * (lo + hi), HI, rtol=1e-12)
    # a receding pair shifts the whole window down
    lo2, hi2 = _magic_window_mhz(HI, _w(-30.0, -30.0), 5.0)
    assert hi2 < lo
    assert np.isclose(0.5 * (lo2 + hi2), HI * (1 - 30.0 / C_KMS), rtol=1e-12)


def test_a_pointing_without_radial_velocities_is_not_a_test(conf):
    aconf = conf["assess"]
    assert _magic_window_mhz(HI, {"n_pairs": 3}, 5.0) is None
    line, tested = _magic_line_match({"freq_mhz": HI}, {"n_pairs": 3}, conf, aconf)
    assert line is None
    assert tested == 0          # not counted as a trial, and not counted as a miss


def test_a_hit_on_the_shifted_line_matches_and_one_beside_it_does_not(conf):
    aconf = conf["assess"]
    dv = 42.0
    f_obs = HI * (1 + dv / C_KMS)
    line, tested = _magic_line_match({"freq_mhz": f_obs}, _w(dv, dv), conf, aconf)
    assert line == "HI_1420"
    assert tested == 1                      # it stopped at the line that matched
    off, tested_off = _magic_line_match({"freq_mhz": f_obs + 1.0}, _w(dv, dv), conf, aconf)
    assert off is None
    assert tested_off == len(conf["drift"]["magic_lines_mhz"])   # every line was tried


def test_the_unshifted_line_does_not_match_a_fast_pair(conf):
    """The test has teeth: at 42 km/s the line is 0.2 MHz off, ten tolerances away."""
    aconf = conf["assess"]
    line, _ = _magic_line_match({"freq_mhz": HI}, _w(42.0, 42.0), conf, aconf)
    assert line is None


def test_the_oh_lines_are_distinguished(conf):
    aconf = conf["assess"]
    f = 1665.4018 * (1 + 10.0 / C_KMS)
    line, _ = _magic_line_match({"freq_mhz": f}, _w(10.0, 10.0), conf, aconf)
    assert line == "OH_1665"


def test_chance_rate_is_the_window_over_the_searched_band(conf):
    aconf = conf["assess"]
    lines = conf["drift"]["magic_lines_mhz"]
    tol = float(aconf["magic_line_window_kms"])
    band = float(aconf["magic_search_bandwidth_mhz"])
    expect = 100 * float(np.mean([2 * f * tol / C_KMS for f in lines.values()])) / band
    assert np.isclose(_magic_chance(100, conf, aconf), expect)
    assert _magic_chance(0, conf, aconf) is None
    # the rate is small but NOT negligible: it has to be printed, not waved away
    assert 1e-4 < _magic_chance(100, conf, aconf) < 1e-2


def test_a_hit_with_no_frequency_is_not_tested(conf):
    line, tested = _magic_line_match({"freq_mhz": np.nan}, _w(0.0, 0.0), conf, aconf := conf["assess"])
    assert (line, tested) == (None, 0)
    assert aconf is conf["assess"]

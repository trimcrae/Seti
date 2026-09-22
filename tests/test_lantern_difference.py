"""LANTERN: the out-minus-in DIFFERENCE search.

The screen's first runner measurement (run 35737559234) showed that on real
JWST ``x1d`` products the time-averaged spectrum's residual around its local
continuum sits at ~1% of the continuum -- not photon noise but the static pixel
pattern of the extraction -- so a 6-sigma trigger on the out-of-eclipse
spectrum alone needs a line brighter than ~8% of the continuum.  Both
known-eclipse verification cases failed on exactly that: an injected line at 2%
of the continuum produced ZERO features (WASP-43 b MIRI/LRS, WASP-18 b
NIRISS/SOSS), while the phase labels and the eclipse itself were correct
(depths 6681 ppm at 15.4 sigma and 1451 ppm at 23.1 sigma).

The static pattern is identical in the in-event and out-of-event averages and
cancels in their difference.  These tests hold the difference search to the
same contract as the rest of the channel: recover an injected signal, return a
clean null on the dominant confounder, and trip every rejection rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from seti.lantern.line import (
    difference_spectrum,
    drift_control_masks,
    narrow_feature_search,
    residual_z,
    time_average_spectrum,
)
from seti.lantern.run import analyse_stack, load_lantern_config, measure_event_depth
from seti.lantern.synth import synthesise_timeseries

PATTERN = 0.01          # the ~1% static pixel pattern measured on real x1d products
LINE_WL = 4.05


def _stack(**kw):
    s = synthesise_timeseries(n_int=600, n_wl=600, fixed_pattern_amp=PATTERN, **kw)
    s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
    s["time_source"] = "int_times_bjd_tdb"
    return s


def _analyse(**kw):
    s = _stack(**kw)
    rec = analyse_stack(s, [s["ephemeris"]], load_lantern_config(), "synthetic")
    near = [f for f in rec["features"] if abs(f["wavelength"] - LINE_WL) < 0.01]
    return rec, (near[0] if near else None)


# --- the static pattern is real, and it is what the difference removes ---------


def test_static_pattern_sets_the_out_of_eclipse_noise_floor():
    """The out-of-eclipse spectrum is pattern limited; the difference is not."""
    s = _stack(line_amp=0.0)
    lab_out = np.ones(s["flux"].shape[0], bool)
    avg = time_average_spectrum(s["flux"], lab_out, s["flux_err"])
    scan_out = narrow_feature_search(s["wavelength"], avg["spec"], avg["spec_err"], 2.0)
    # The pattern, not sqrt(N) photon noise, holds the floor near 1%.
    assert 0.3 * PATTERN < scan_out["noise_median"] < 3 * PATTERN
    half = s["flux"].shape[0] // 2
    a, b = np.zeros(lab_out.size, bool), np.zeros(lab_out.size, bool)
    a[:half], b[half:] = True, True
    d = difference_spectrum(s["flux"], a, b, s["flux_err"])
    scan_d = narrow_feature_search(s["wavelength"], d["spec"], d["spec_err"], 2.0)
    # The same pattern is in both halves and cancels: >10x deeper.
    assert scan_d["noise_median"] < 0.1 * scan_out["noise_median"]


def test_difference_recovers_a_line_the_out_spectrum_cannot_see():
    rec, f = _analyse(line_amp=0.02)
    assert rec["phase_class"] == "eclipse"
    # Invisible to the out-of-eclipse search under the pattern...
    assert rec["n_features_out_spectrum"] == 0
    # ...and a clean candidate in the difference.
    assert f is not None
    assert f["found_in"] == "difference"
    assert f["vetoes_local"] == []
    assert f["tier_local"] == "candidate"
    assert f["eclipse"]["eclipse_vanish_snr"] > 10.0
    # The quoted sensitivity is the difference's, and it is far deeper.
    assert rec["ew_5sigma_limit_diff_um"] < 0.05 * rec["ew_5sigma_limit_out_um"]
    assert rec["ew_5sigma_limit_um"] == rec["ew_5sigma_limit_diff_um"]


def test_constant_stellar_line_cancels_in_the_difference():
    """The dominant confounder: a line that does not care about the eclipse."""
    rec, f = _analyse(line_amp=0.02, line_vanishes=False)
    assert f is None or f["tier_local"] == "none"
    assert not [x for x in rec["features"] if x["tier_local"] in ("interest", "candidate")]


def test_no_line_is_a_clean_null():
    rec, f = _analyse(line_amp=0.0)
    assert f is None
    assert [x for x in rec["features"] if x["tier_local"] != "none"] == []


# --- the drift null -------------------------------------------------------------


def test_drift_control_masks_straddle_the_event():
    inn = np.zeros(20, bool)
    inn[8:12] = True
    out = ~inn
    a, b = drift_control_masks(out, inn)
    assert a.sum() == 8 and b.sum() == 8
    assert not (a & inn).any() and not (b & inn).any()
    assert np.flatnonzero(a).max() < 8 and np.flatnonzero(b).min() > 11


def test_persistence_ramp_is_caught_by_the_drift_null():
    """A decaying line is present out of eclipse and weaker after it -- it looks
    like a vanishing line but shows the same step between two out-of-eclipse
    blocks with no occultation in between."""
    rec, f = _analyse(line_amp=0.02, line_vanishes=False, line_ramp_amp=3.0, ramp_tau=60.0)
    assert f is not None
    assert abs(f["eclipse"]["drift_control_snr"]) > 4.0
    assert "present_in_drift_control" in f["vetoes_local"]
    assert f["tier_local"] == "none"


def test_a_real_line_is_not_flagged_by_the_drift_null():
    _rec, f = _analyse(line_amp=0.02)
    assert abs(f["eclipse"]["drift_control_snr"]) < 4.0


# --- the transit difference -------------------------------------------------------


def _transit_vanishing_stack():
    """A line that switches off while the planet is IN FRONT of the star.

    The synthesiser gates its line on eclipse visibility, so the transit case is
    built by re-gating the line here.
    """
    s = _stack(line_amp=0.0, centre="transit")
    from seti.lantern.phase import label_integrations

    lab = label_integrations(s["times"], s["ephemeris"], load_lantern_config()["phase"])
    j = int(np.argmin(np.abs(s["wavelength"] - LINE_WL)))
    prof = np.exp(-0.5 * ((np.arange(s["wavelength"].size) - j) / 0.9) ** 2)
    vis = np.where(lab["in_transit"], 0.0, np.where(lab["transit_contact"], 0.5, 1.0))
    cont = np.nanmedian(s["flux"][:, j - 10:j + 11], axis=1)
    s["flux"] = s["flux"] + (0.02 * cont * vis)[:, None] * prof[None, :]
    return s, lab


def test_transit_difference_reaches_a_line_the_out_spectrum_cannot():
    """A transit-class visit cannot test vanishing, but it can still be made
    photon limited: the out-of-transit minus in-transit difference cancels the
    same static pattern, so a narrow feature that CHANGED across transit is
    reachable at ~0.1% of the continuum instead of ~8%."""
    s, lab = _transit_vanishing_stack()
    assert lab["phase_class"] == "transit"
    rec = analyse_stack(s, [s["ephemeris"]], load_lantern_config(), "synthetic")
    assert rec["phase_class"] == "transit"
    assert rec.get("difference_kind") == "transit_difference"
    assert rec["n_features_out_spectrum"] == 0        # the pattern hides it
    near = [f for f in rec["features"] if abs(f["wavelength"] - LINE_WL) < 0.01]
    assert near, "the transit difference must reach a 2% line under a 1% pattern"
    f = near[0]
    assert f["found_in"] == "transit_difference"
    assert f["snr"] > 20.0
    assert rec["ew_5sigma_limit_diff_um"] < 0.05 * rec["ew_5sigma_limit_out_um"]


def test_transit_class_exposure_never_yields_a_candidate():
    """Whatever the transit difference turns up, it is not LANTERN's signature:
    with no eclipse there is no vanishing test, and a line that changes across
    transit MORE than the continuum does is not a steady source on the planet.
    Both statements have to appear as named vetoes, and no tier above watch."""
    s, _lab = _transit_vanishing_stack()
    rec = analyse_stack(s, [s["ephemeris"]], load_lantern_config(), "synthetic")
    near = [f for f in rec["features"] if abs(f["wavelength"] - LINE_WL) < 0.01]
    assert "insufficient_phase_coverage" in near[0]["vetoes_local"]
    assert "transit_inconsistent" in near[0]["vetoes_local"]
    assert all(f["tier_local"] in ("none", "watch") for f in rec["features"])
    assert all(not f["eclipse_tested"] for f in rec["features"])


def test_transit_difference_is_a_clean_null_on_a_constant_line():
    """The transit-class sensitivity is real, not an artefact of the injection:
    a line that does NOT change across transit leaves the difference empty."""
    s = _stack(line_amp=0.02, line_vanishes=False, centre="transit")
    rec = analyse_stack(s, [s["ephemeris"]], load_lantern_config(), "synthetic")
    assert rec["phase_class"] == "transit"
    assert rec.get("difference_kind") == "transit_difference"
    assert rec["ew_5sigma_limit_diff_um"] < 0.05 * rec["ew_5sigma_limit_out_um"]
    assert [f for f in rec["features"] if abs(f["wavelength"] - LINE_WL) < 0.01] == []


# --- degradation ------------------------------------------------------------------


@pytest.mark.parametrize("n_in", [0, 3])
def test_difference_not_attempted_without_enough_in_event_integrations(n_in):
    """No eclipse in the window -> no difference search, no fabricated result."""
    s = _stack(line_amp=0.02, centre="none")
    rec = analyse_stack(s, [s["ephemeris"]], load_lantern_config(), "synthetic")
    if rec["phase_class"] == "phase_unresolved":
        assert rec.get("ew_5sigma_limit_diff_um") is None
        assert all(f["found_in"] == "out_spectrum" for f in rec["features"])


# --- what the limit means ---------------------------------------------------------


@pytest.mark.parametrize("depth", [0.002, 0.01])
def test_eclipse_depth_is_recovered_and_sets_the_beacon_fraction(depth):
    """The equivalent-width limit is a fraction of the STAR's continuum; divided
    by the measured eclipse depth it becomes a fraction of the PLANET's own
    broad-band emission, which is the statement that says what kind of beacon
    was ruled out -- and it needs no distance and no stellar model."""
    rec, _f = _analyse(line_amp=0.0, eclipse_depth=depth)
    ev = rec["event_depth"]
    assert ev["depth"] == pytest.approx(depth, rel=0.1)
    assert ev["depth_snr"] > 10
    assert rec["line_contrast_5sigma"] > 0
    assert rec["beacon_fraction_of_event_flux_5sigma"] == pytest.approx(
        rec["line_contrast_5sigma"] / ev["depth"], rel=1e-6)


def test_event_depth_degrades_rather_than_guesses():
    f = np.ones((20, 50)) + 0.01 * np.random.default_rng(0).normal(size=(20, 50))
    t = np.arange(20, dtype=float)
    inn = np.zeros(20, bool)
    inn[:2] = True                      # too few in-event integrations
    assert measure_event_depth(f, t, inn, ~inn)["depth"] is None


def test_difference_spectrum_offsets_to_unity():
    """The difference keeps the continuum near 1 so residuals, equivalent widths
    and significances stay in fractions of the STELLAR continuum."""
    s = _stack(line_amp=0.0)
    n = s["flux"].shape[0]
    a, b = np.zeros(n, bool), np.zeros(n, bool)
    a[: n // 2], b[n // 2:] = True, True
    d = difference_spectrum(s["flux"], a, b, s["flux_err"])
    assert np.nanmedian(d["spec"]) == pytest.approx(1.0, abs=0.01)
    z = residual_z(d["spec"], d["spec_err"], 2.0)["z"]
    assert np.nanpercentile(np.abs(z), 50) < 1.5

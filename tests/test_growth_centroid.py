"""Offline test suite for GROWTH **stage 3** --- is the transit on the target?

No network anywhere (``conftest.py`` raises on any socket); every fetch in
:mod:`seti.growth.centroid` takes an injectable callable and every test injects
one.  What this suite is the gate for:

* **the load-bearing test: a synthetic pixel stamp with the transit injected on
  a KNOWN pixel.**  With the transit on the target the measured offset must be
  consistent with zero; with it injected on a neighbour three pixels away the
  difference-image centroid must land on that neighbour, within its own quoted
  uncertainty.  Everything else in this module is bookkeeping around that
  number, and a centroid that cannot find an injected source finds nothing.
* **the census arithmetic**: a neighbour too faint to supply the observed depth
  is excluded, and the depth it *would* have needed is reported rather than
  merely asserted; a neighbour bright enough is NOT excluded.
* **a failed fetch is ``QUERY_FAILED``**, never an on-target verdict, and
  ``QUERY_RETURNED_ZERO_ROWS`` stays a different fact.
* **an offset whose error bar spans zero without the precision to exclude the
  neighbour that matters is ``OFFSET_UNRESOLVED``** --- never
  ``TRANSIT_ON_TARGET``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.growth.acquire import STATUS_FAILED, STATUS_OK, STATUS_ZERO, AcquisitionLog
from seti.growth.centroid import (
    NO_DATA,
    OFFSET_FROM_TARGET,
    OFFSET_UNRESOLVED,
    ON_TARGET,
    REASON_BUDGET,
    REASON_NO_CENTROID,
    REASON_PRECISION,
    TARGET_VERDICTS,
    CensusParams,
    DiffParams,
    TpfParams,
    attribute_offset,
    centroid_assess,
    centroid_census_stage,
    centroid_difference_stage,
    centroid_probe,
    centroid_run,
    centroid_verdict,
    combine_sector_offsets,
    difference_image_offset,
    fetch_gaia_census,
    fetch_target_pixel_files,
    gaussian_prf,
    image_centroid,
    load_measured_depths,
    load_shortlist,
    neighbour_census,
    position_angle_deg,
    run_verdict,
    sky_sep_arcsec,
    synth_tpf,
    tpf_probe,
    transit_cadence_masks,
)
from seti.growth.centroid import (
    main as centroid_main,
)

# The stage-1/stage-2 candidate: P = 2.052349884 d, T14 = 2.0778 h, and the
# depth stage 2 fitted from the TESS light curve.
PERIOD = 2.052349884
DURATION_D = 2.0778 / 24.0
T0 = 1000.0
DEPTH_PPM = 30614.0
RA, DEC = 290.0, 43.5


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_separation_and_position_angle_are_the_textbook_values():
    # 10" due north.
    assert sky_sep_arcsec(RA, DEC, RA, DEC + 10.0 / 3600.0) == pytest.approx(10.0, abs=1e-3)
    assert position_angle_deg(RA, DEC, RA, DEC + 10.0 / 3600.0) == pytest.approx(0.0, abs=1e-6)
    # 10" due east: the RA offset is 10"/cos(dec).
    east = RA + (10.0 / 3600.0) / math.cos(math.radians(DEC))
    assert sky_sep_arcsec(RA, DEC, east, DEC) == pytest.approx(10.0, abs=1e-3)
    assert position_angle_deg(RA, DEC, east, DEC) == pytest.approx(90.0, abs=0.1)
    assert not np.isfinite(sky_sep_arcsec(np.nan, DEC, RA, DEC))


# ---------------------------------------------------------------------------
# 1. The census arithmetic --- the test that needs no pixels
# ---------------------------------------------------------------------------
def _gaia_rows(rows) -> pd.DataFrame:
    """``[(dG_from_12th_mag, sep_arcsec, pa)]`` -> a Gaia-shaped cone answer."""
    recs = []
    for i, (dg, sep, pa) in enumerate(rows):
        d = math.radians(float(pa))
        dec = DEC + (sep / 3600.0) * math.cos(d)
        ra = RA + (sep / 3600.0) * math.sin(d) / math.cos(math.radians(DEC))
        recs.append({"source_id": f"{1000 + i}", "ra": ra, "dec": dec,
                     "phot_g_mean_mag": 12.0 + float(dg)})
    return pd.DataFrame(recs)


def test_a_neighbour_too_faint_to_make_the_depth_is_excluded_by_arithmetic():
    """A 5-magnitude-fainter star cannot make a 3 % dip: say so, with the number."""
    src = _gaia_rows([(0.0, 0.1, 0.0),      # the target itself
                      (5.0, 6.0, 45.0),     # 100x fainter: arithmetically impossible
                      (1.0, 9.0, 200.0)])   # only 2.5x fainter: not excluded
    census, summary = neighbour_census(src, ra=RA, dec=DEC, observed_depth_ppm=DEPTH_PPM,
                                       params=CensusParams())
    assert summary["target_found_in_gaia"] is True
    assert summary["n_neighbours"] == 2
    faint = census[census["source_id"] == "1001"].iloc[0]
    bright = census[census["source_id"] == "1002"].iloc[0]

    # The faint one: the depth it would need is reported and is over 100 %.
    assert bool(faint["excluded_by_arithmetic"])
    assert faint["required_depth_ppm"] > 1e6
    assert "impossible" in faint["exclusion_note"]
    # It is the arithmetic, not a threshold: depth / flux-fraction, exactly.
    assert faint["required_depth_ppm"] == pytest.approx(DEPTH_PPM / faint["flux_fraction"],
                                                        rel=1e-9)
    # The bright one survives, and the depth it would need is a real number.
    assert not bool(bright["excluded_by_arithmetic"])
    assert 0.0 < bright["required_depth_ppm"] <= 1e6
    assert "NOT excluded" in bright["exclusion_note"]

    assert summary["n_neighbours_excluded"] == 1
    assert summary["n_neighbours_not_excluded"] == 1
    assert summary["census_excludes_all_neighbours"] is False
    assert summary["nearest_unexcluded_sep_arcsec"] == pytest.approx(9.0, abs=0.1)
    # The magnitude gap at which a neighbour stops being able to do it at all.
    assert summary["min_delta_g_that_can_supply_depth"] == pytest.approx(3.7, abs=0.4)
    assert "excluded by arithmetic" in summary["census_statement"]


def test_a_census_in_which_every_neighbour_is_excluded_says_so():
    src = _gaia_rows([(0.0, 0.2, 0.0), (4.5, 5.0, 10.0), (6.0, 12.0, 100.0),
                      (8.0, 18.0, 250.0)])
    census, summary = neighbour_census(src, ra=RA, dec=DEC, observed_depth_ppm=DEPTH_PPM)
    assert summary["n_neighbours"] == 3
    assert summary["n_neighbours_excluded"] == 3
    assert summary["census_excludes_all_neighbours"] is True
    assert not np.isfinite(summary["nearest_unexcluded_sep_arcsec"])
    assert bool(census[~census["is_target"]]["excluded_by_arithmetic"].all())
    assert "all 3" in summary["census_statement"]


def test_the_census_puts_the_target_in_the_denominator_and_keeps_its_own_row():
    src = _gaia_rows([(0.0, 0.1, 0.0), (3.0, 8.0, 30.0)])
    census, summary = neighbour_census(src, ra=RA, dec=DEC, observed_depth_ppm=DEPTH_PPM)
    tgt = census[census["is_target"]].iloc[0]
    assert tgt["delta_g_mag"] == pytest.approx(0.0)
    # Every flux fraction sums to one: the aperture is fully accounted for.
    assert float(census["flux_fraction"].sum()) == pytest.approx(1.0, rel=1e-9)
    # The target's own required depth is the dilution-corrected depth: slightly
    # deeper than the aperture value, never shallower.
    assert tgt["required_depth_ppm"] > DEPTH_PPM
    assert summary["contam_max"] == pytest.approx(10.0 ** (-0.4 * 3.0), rel=1e-6)


def test_a_census_with_no_sources_excludes_nothing_and_implicates_nothing():
    census, summary = neighbour_census(None, ra=RA, dec=DEC, observed_depth_ppm=DEPTH_PPM)
    assert len(census) == 0
    assert summary["n_neighbours"] == 0
    assert summary["census_excludes_all_neighbours"] is False
    assert "could not be run" in summary["census_statement"]


def test_a_failed_gaia_cone_is_query_failed_and_an_empty_one_is_zero_rows():
    log = AcquisitionLog()

    def boom(_t, _r, **_kw):
        raise RuntimeError("Error 500: statement timeout")

    df, status = fetch_gaia_census(RA, DEC, cone_fn=boom, log=log, key="K00897.01")
    assert status == STATUS_FAILED and not len(df)

    def empty(_t, _r, **_kw):
        return pd.DataFrame(columns=["source_id", "ra", "dec", "phot_g_mean_mag"]), []

    df, status = fetch_gaia_census(RA, DEC, cone_fn=empty, log=log)
    assert status == STATUS_ZERO
    assert STATUS_FAILED != STATUS_ZERO

    def unreachable(_t, _r, **_kw):
        return pd.DataFrame(), [0]

    _df, status = fetch_gaia_census(RA, DEC, cone_fn=unreachable, log=log)
    assert status == STATUS_FAILED

    def ok(_t, _r, **_kw):
        return _gaia_rows([(0.0, 0.1, 0.0), (2.0, 7.0, 90.0)]), []

    df, status = fetch_gaia_census(RA, DEC, cone_fn=ok, log=log)
    assert status == STATUS_OK and len(df) == 2


# ---------------------------------------------------------------------------
# 2. The difference image --- the load-bearing test
# ---------------------------------------------------------------------------
def _stamp(*, transit_source: int, neighbour_dx: float = 3.0, neighbour_flux: float = 0.15,
           noise: float = 4.0e-4, depth: float = 0.03, n_transits: int = 8, sector: int = 41,
           seed: int = 5) -> dict:
    """An 11x11 stamp: target at (5, 5), neighbour ``neighbour_dx`` pixels east."""
    return synth_tpf(sources=[{"x": 5.0, "y": 5.0, "flux": 1.0},
                              {"x": 5.0 + neighbour_dx, "y": 5.0, "flux": neighbour_flux}],
                     period_days=PERIOD, t0_btjd=T0, duration_days=DURATION_D, depth=depth,
                     transit_source=transit_source, target_index=0, n_transits=n_transits,
                     noise=noise, sector=sector, seed=seed)


def test_the_cadence_masks_separate_the_transit_from_a_local_baseline():
    s = _stamp(transit_source=0)
    in_t, oot, dt = transit_cadence_masks(s["time"], period_days=PERIOD, t0_btjd=T0,
                                          duration_days=DURATION_D,
                                          exptime_days=s["exptime_s"] / 86400.0)
    assert in_t.sum() > 0 and oot.sum() > 0
    assert not (in_t & oot).any()                        # the two sets are disjoint
    assert np.abs(dt[in_t]).max() <= 0.5 * DURATION_D    # in transit means in transit
    assert np.abs(dt[oot]).min() >= 0.75 * DURATION_D    # the guard band is honoured


def test_the_centroid_finds_an_injected_gaussian_where_it_was_put():
    img = gaussian_prf((11, 11), 7.25, 3.75, 0.9, 100.0)
    c = image_centroid(img, params=DiffParams(subtract_background=False))
    assert c["status"] == STATUS_OK
    assert c["x"] == pytest.approx(7.25, abs=0.05)
    assert c["y"] == pytest.approx(3.75, abs=0.05)
    # A flat image has no source: no centroid, and it says so rather than
    # returning the middle of the stamp.
    flat = image_centroid(np.ones((11, 11)), params=DiffParams())
    assert flat["status"] != STATUS_OK and not np.isfinite(flat["x"])


def test_transit_on_the_target_gives_an_offset_consistent_with_zero():
    """THE GATE (a).  The transit is on the target; the offset must be zero."""
    s = _stamp(transit_source=0)
    r = difference_image_offset(s, period_days=PERIOD, t0_btjd=T0, duration_days=DURATION_D,
                                params=DiffParams(), rng=np.random.default_rng(3))
    assert r["status"] == STATUS_OK
    assert r["n_in_transit"] >= 3 and r["n_out_of_transit"] >= 6
    assert r["n_transits_covered"] == 8
    # The difference image peaks on the TARGET pixel, which is where the
    # transit was injected.
    assert r["diff_x_px"] == pytest.approx(5.0, abs=0.25)
    assert r["diff_y_px"] == pytest.approx(5.0, abs=0.25)
    # ... and the offset from the out-of-transit centroid is consistent with
    # zero at the module's own threshold.
    assert np.isfinite(r["offset_err_px"]) and r["offset_err_px"] > 0
    assert r["offset_sigma"] < DiffParams().n_offset, (
        f"offset {r['offset_px']:.3f} +/- {r['offset_err_px']:.3f} px is not consistent "
        "with zero for a transit injected ON the target")
    assert r["offset_from_target_px"] == pytest.approx(0.0, abs=0.3)


def test_transit_on_a_neighbour_three_pixels_away_is_recovered_at_that_neighbour():
    """THE GATE (b).  The transit is on a neighbour 3 px east; recover its position."""
    s = _stamp(transit_source=1, neighbour_dx=3.0, neighbour_flux=0.25, depth=0.12)
    r = difference_image_offset(s, period_days=PERIOD, t0_btjd=T0, duration_days=DURATION_D,
                                params=DiffParams(), rng=np.random.default_rng(3))
    assert r["status"] == STATUS_OK
    # The difference image centroid IS the neighbour, to within its uncertainty.
    err = math.hypot(r["diff_x_err_px"] or 0.0, r["diff_y_err_px"] or 0.0)
    assert r["diff_x_px"] == pytest.approx(8.0, abs=max(5.0 * err, 0.3)), (
        f"difference-image centroid {r['diff_x_px']:.3f} px does not recover the injected "
        "neighbour at x = 8.0 px")
    assert r["diff_y_px"] == pytest.approx(5.0, abs=max(5.0 * err, 0.3))
    # The offset from the target's own pixel is the full three pixels.
    assert r["offset_from_target_px"] == pytest.approx(3.0, rel=0.15)
    # The offset from the out-of-transit centroid is smaller --- the direct
    # image is itself pulled toward the neighbour --- but it is significant,
    # and it is a LOWER BOUND on the displacement from the target, never an
    # upper one.  That inequality is the fact, not the exact shrinkage.
    assert 1.5 < r["offset_px"] < 3.0
    assert r["offset_px"] < r["offset_from_target_px"]
    assert r["offset_sigma"] > DiffParams().n_offset
    assert r["offset_arcsec"] == pytest.approx(r["offset_px"] * 21.0, rel=1e-9)


def test_the_offset_scales_with_the_injected_separation():
    """Two pixels in, two pixels out: the measurement is a ruler, not a flag."""
    out = {}
    for dx in (2.0, 4.0):
        s = _stamp(transit_source=1, neighbour_dx=dx, neighbour_flux=0.25, depth=0.12)
        r = difference_image_offset(s, period_days=PERIOD, t0_btjd=T0,
                                    duration_days=DURATION_D, params=DiffParams(),
                                    rng=np.random.default_rng(11))
        out[dx] = r["offset_from_target_px"]
    assert out[2.0] == pytest.approx(2.0, rel=0.2)
    assert out[4.0] == pytest.approx(4.0, rel=0.2)


def test_a_sector_without_enough_in_transit_cadences_is_rejected_not_fitted():
    s = _stamp(transit_source=0)
    # An ephemeris half a period out puts no cadence in transit.
    r = difference_image_offset(s, period_days=PERIOD, t0_btjd=T0 + 0.5 * PERIOD,
                                duration_days=DURATION_D, params=DiffParams())
    assert r["status"] != STATUS_OK
    assert r["reject_reason"] == "too_few_in_transit_cadences"
    assert not np.isfinite(r["offset_px"])


def test_a_sub_arcsecond_offset_is_not_called_an_offset_however_small_the_error():
    """The systematic floor.  Statistical significance is not astrometry.

    A thresholded first moment on 21" pixels picks up a fraction of a pixel
    from the direct image's faint wings that the difference image does not
    share.  Measured against a cadence bootstrap alone that shows up as a
    multi-sigma offset of a tenth of an arcsecond --- arithmetic, not a
    displaced transit --- so the quoted error carries a floor.
    """
    s = _stamp(transit_source=0, neighbour_flux=0.15, noise=1e-5, n_transits=10)
    r = difference_image_offset(s, period_days=PERIOD, t0_btjd=T0, duration_days=DURATION_D,
                                params=DiffParams(), rng=np.random.default_rng(2))
    assert r["status"] == STATUS_OK
    assert r["offset_arcsec"] < 1.0                       # a tenth of an arcsecond
    assert r["offset_err_stat_px"] * 21.0 < 1.0           # measured far more precisely
    assert r["centroid_sys_floor_arcsec"] == 1.0
    assert r["offset_err_arcsec"] >= 1.0
    assert r["offset_sigma"] < DiffParams().n_offset, (
        "a sub-arcsecond offset measured to milli-arcseconds is a systematic of the "
        "centroid method, and must not be reported as the transit being off target")
    # The floor is a parameter, not a constant: turn it off and the same
    # measurement does become formally significant, which is the point.
    bare = difference_image_offset(s, period_days=PERIOD, t0_btjd=T0, duration_days=DURATION_D,
                                   params=DiffParams(centroid_sys_floor_arcsec=0.0),
                                   rng=np.random.default_rng(2))
    assert bare["offset_err_arcsec"] < r["offset_err_arcsec"]


def test_combining_on_target_sectors_does_not_manufacture_an_offset():
    """Averaging raw lengths shrinks the error, not the positive bias."""
    rows = [difference_image_offset(_stamp(transit_source=0, sector=s, seed=s, noise=1e-5),
                                    period_days=PERIOD, t0_btjd=T0, duration_days=DURATION_D,
                                    params=DiffParams(), rng=np.random.default_rng(s))
            for s in (41, 54, 55, 74, 75, 81, 82)]
    comb = combine_sector_offsets(rows, params=DiffParams())
    assert comb["n_sectors_used"] == 7
    assert comb["offset_err_arcsec"] >= DiffParams().centroid_sys_floor_arcsec
    assert comb["offset_sigma"] < DiffParams().n_offset, (
        f"seven on-target sectors combined to {comb['offset_arcsec']:.3f} +/- "
        f"{comb['offset_err_arcsec']:.3f}\" --- a manufactured offset")
    v = centroid_verdict(comb, {"nearest_unexcluded_sep_arcsec": 6.0}, params=DiffParams())
    assert v["verdict"] == ON_TARGET


def test_per_sector_and_combined_offsets_are_both_reported():
    rows = [difference_image_offset(_stamp(transit_source=0, sector=s, seed=s),
                                    period_days=PERIOD, t0_btjd=T0,
                                    duration_days=DURATION_D, params=DiffParams(),
                                    rng=np.random.default_rng(s))
            for s in (41, 54, 55)]
    comb = combine_sector_offsets(rows, params=DiffParams())
    assert comb["n_sectors_used"] == 3
    assert comb["sectors_used"] == "41,54,55"
    assert comb["combined_from_magnitudes"] is True     # no WCS in a synthetic stamp
    assert np.isfinite(comb["offset_err_arcsec"]) and comb["offset_err_arcsec"] > 0
    # A consistent set of sectors is recorded as consistent; the scatter is a
    # number either way.
    assert comb["sector_scatter_dof"] == 2
    assert np.isfinite(comb["sector_scatter_chi2_per_dof"])


def test_sector_offsets_combine_as_sky_vectors_when_a_wcs_is_available():
    """Two roll angles: pixel offsets differ, the SKY offset is one vector."""
    rows = []
    for sector, sign in ((41, +1.0), (54, -1.0)):
        r = difference_image_offset(_stamp(transit_source=1, neighbour_flux=0.25, depth=0.12,
                                           sector=sector, seed=sector),
                                    period_days=PERIOD, t0_btjd=T0, duration_days=DURATION_D,
                                    params=DiffParams(), rng=np.random.default_rng(sector))
        # A stand-in for the file's own WCS: the same sky offset seen at two
        # opposite roll angles.
        r["offset_ra_arcsec"] = sign * 40.0
        r["offset_dec_arcsec"] = 0.0
        rows.append(r)
    comb = combine_sector_offsets(rows, params=DiffParams())
    assert comb["combined_from_magnitudes"] is False
    # The two sky vectors cancel: the combination is honest about that, and the
    # scatter chi2 is what says the sectors disagree.
    assert abs(comb["offset_ra_arcsec"]) < 20.0
    assert comb["sector_scatter_chi2_per_dof"] > DiffParams().sector_scatter_chi2_per_dof
    assert comb["sector_offsets_consistent"] is False


# ---------------------------------------------------------------------------
# 3. The verdict vocabulary
# ---------------------------------------------------------------------------
def test_an_error_bar_spanning_zero_without_the_precision_is_offset_unresolved():
    """The rule: an offset you cannot resolve is NEVER TRANSIT_ON_TARGET."""
    cs = {"nearest_unexcluded_sep_arcsec": 6.0, "n_neighbours": 2,
          "n_neighbours_not_excluded": 1, "census_excludes_all_neighbours": False}
    # 2 +/- 5 arcsec: consistent with zero, but 3 sigma is 15" and the
    # neighbour that matters is at 6".  Unresolved.
    v = centroid_verdict({"offset_arcsec": 2.0, "offset_err_arcsec": 5.0, "offset_sigma": 0.4,
                          "n_sectors_used": 3}, cs, params=DiffParams())
    assert v["verdict"] == OFFSET_UNRESOLVED
    assert v["reason"] == REASON_PRECISION
    assert v["offset_detectable_arcsec"] == pytest.approx(15.0)
    assert v["resolution_required_arcsec"] == pytest.approx(6.0)
    assert "UNRESOLVED" in v["verdict_statement"]

    # The same offset measured ten times better IS an on-target statement.
    good = centroid_verdict({"offset_arcsec": 0.2, "offset_err_arcsec": 0.5,
                             "offset_sigma": 0.4, "n_sectors_used": 3}, cs,
                            params=DiffParams())
    assert good["verdict"] == ON_TARGET
    assert good["offset_detectable_arcsec"] < good["resolution_required_arcsec"]


def test_a_significant_offset_is_attributed_to_the_neighbour_it_lands_on():
    src = _gaia_rows([(0.0, 0.1, 0.0), (1.5, 30.0, 90.0), (1.0, 8.0, 90.0)])
    census, cs = neighbour_census(src, ra=RA, dec=DEC, observed_depth_ppm=DEPTH_PPM,
                                  params=CensusParams(radius_arcsec=42.0,
                                                      aperture_radius_arcsec=42.0))
    v = centroid_verdict({"offset_arcsec": 8.2, "offset_err_arcsec": 0.8, "offset_sigma": 10.2,
                          "offset_pa_deg": 90.0, "n_sectors_used": 7}, cs,
                         params=DiffParams(), census=census)
    assert v["verdict"] == OFFSET_FROM_TARGET
    assert v["attributed_source_id"] == "1002"
    assert v["attributed_sep_arcsec"] == pytest.approx(8.0, abs=0.2)
    assert np.isfinite(v["attributed_required_depth_ppm"])
    assert "NOT on the target" in v["verdict_statement"]


def test_an_offset_that_lands_on_nothing_says_so():
    src = _gaia_rows([(0.0, 0.1, 0.0), (1.0, 3.0, 90.0)])
    census, _cs = neighbour_census(src, ra=RA, dec=DEC, observed_depth_ppm=DEPTH_PPM)
    a = attribute_offset(census, 15.0, 0.5, n_sigma=3.0)
    assert a["attributed_source_id"] == ""
    assert "does not land on a catalogued neighbour" in a["attribution_note"]


def test_a_failed_fetch_is_query_failed_and_never_an_on_target_verdict():
    log = AcquisitionLog()

    def boom(_tic, **_kw):
        raise RuntimeError("MAST timed out")

    secs, status, _route = fetch_target_pixel_files(268924036, tpf_fn=boom,
                                                    params=TpfParams(retries=1), log=log)
    assert status == STATUS_FAILED and secs == []
    v = centroid_verdict(None, {}, params=DiffParams(), reason=status)
    assert v["verdict"] == NO_DATA
    assert v["reason"] == STATUS_FAILED
    assert v["verdict"] != ON_TARGET

    def empty(_tic, **_kw):
        return []

    secs, status, _route = fetch_target_pixel_files(268924036, tpf_fn=empty,
                                                    params=TpfParams(retries=1), log=log)
    assert status == STATUS_ZERO and secs == []
    assert STATUS_ZERO != STATUS_FAILED
    v = centroid_verdict(None, {}, params=DiffParams(), reason=status)
    assert v["verdict"] == NO_DATA and v["reason"] == STATUS_ZERO


def test_a_missing_centroid_is_unresolved_not_on_target():
    v = centroid_verdict({"offset_arcsec": float("nan"), "offset_err_arcsec": float("nan")},
                         {}, params=DiffParams())
    assert v["verdict"] == OFFSET_UNRESOLVED
    assert v["reason"] == REASON_NO_CENTROID


def test_the_budget_stops_the_fetch_and_the_reason_is_budget_exhausted():
    from seti.growth.centroid import Deadline

    calls = []

    def slow(_tic, **_kw):
        calls.append(1)
        return [_stamp(transit_source=0)]

    d = Deadline(budget_s=0.0)
    secs, status, _route = fetch_target_pixel_files(268924036, tpf_fn=slow,
                                                    params=TpfParams(), log=AcquisitionLog(),
                                                    deadline=d)
    assert status == STATUS_FAILED and not calls, "the budget must stop the request itself"
    v = centroid_verdict(None, {}, params=DiffParams(), reason=REASON_BUDGET)
    assert v["verdict"] == NO_DATA and v["reason"] == REASON_BUDGET


def test_the_per_target_sector_cap_is_honoured():
    def many(_tic, **_kw):
        return [_stamp(transit_source=0, sector=s, seed=s) for s in range(1, 11)]

    secs, status, _route = fetch_target_pixel_files(268924036, tpf_fn=many,
                                                    params=TpfParams(max_sectors=3),
                                                    log=AcquisitionLog())
    assert status == STATUS_OK and len(secs) == 3


def test_the_run_verdict_aggregates_the_target_verdicts():
    assert run_verdict(pd.DataFrame()) == (NO_DATA, "no_targets")
    v, _r = run_verdict(pd.DataFrame([{"verdict": ON_TARGET}, {"verdict": OFFSET_UNRESOLVED}]))
    assert v == OFFSET_UNRESOLVED
    v, _r = run_verdict(pd.DataFrame([{"verdict": ON_TARGET}, {"verdict": ON_TARGET}]))
    assert v == ON_TARGET
    v, _r = run_verdict(pd.DataFrame([{"verdict": ON_TARGET},
                                      {"verdict": OFFSET_FROM_TARGET}]))
    assert v == OFFSET_FROM_TARGET
    v, r = run_verdict(pd.DataFrame([{"verdict": NO_DATA, "reason": STATUS_FAILED}]))
    assert v == NO_DATA and STATUS_FAILED in r
    for w in TARGET_VERDICTS:
        assert isinstance(w, str)


# ---------------------------------------------------------------------------
# End to end, offline
# ---------------------------------------------------------------------------
def _conf() -> dict:
    return {"centroid": {"shortlist": [{"kepoi_name": "K00897.01", "kepler_name": "Kepler-718 b",
                                        "kepid": 7849854, "tic_id": 268924036, "toi": 4490.01,
                                        "observed_depth_ppm": DEPTH_PPM,
                                        "observed_depth_err_ppm": 1195.0}],
                         "census": {"radius_arcsec": 21.0},
                         "difference": {"n_offset": 3.0},
                         "tpf": {"max_sectors": 3}}}


def _koi_query_fn(_adql: str) -> pd.DataFrame:
    return pd.DataFrame([{"kepoi_name": "K00897.01", "kepid": 7849854,
                          "kepler_name": "Kepler-718 b", "koi_disposition": "CONFIRMED",
                          "koi_period": PERIOD, "koi_period_err1": 1e-7,
                          "koi_time0bk": T0 + 2167.0, "koi_time0bk_err1": 1e-4,
                          "koi_duration": DURATION_D * 24.0, "koi_depth": 14280.6,
                          "ra": RA, "dec": DEC}])


def test_the_stages_run_end_to_end_offline_and_write_the_record(tmp_path: Path):
    out = tmp_path / "centroid"
    conf = _conf()

    def cone(_t, _r, **_kw):
        return _gaia_rows([(0.0, 0.1, 0.0), (5.0, 6.0, 45.0), (6.5, 14.0, 300.0)]), []

    def tpf(_tic, **_kw):
        return [_stamp(transit_source=0, sector=s, seed=s) for s in (41, 54, 55)]

    centroid_probe(conf, out)
    probe = json.loads((out / "probe.json").read_text())
    assert probe["shortlist_size"] == 1

    centroid_census_stage(conf, out, cone_fn=cone, query_fn=_koi_query_fn)
    census = pd.read_csv(out / "census.csv")
    assert len(census) == 3
    assert int(census["excluded_by_arithmetic"].sum()) == 2
    targets = pd.read_csv(out / "targets.csv")
    assert targets["period_days"].iloc[0] == pytest.approx(PERIOD)
    assert targets["observed_depth_ppm"].iloc[0] == pytest.approx(DEPTH_PPM)

    centroid_difference_stage(conf, out, tpf_fn=tpf, query_fn=_koi_query_fn)
    sectors = pd.read_csv(out / "sectors.csv")
    assert len(sectors) == 3 and (sectors["status"] == STATUS_OK).all()
    offsets = pd.read_csv(out / "offsets.csv")
    assert offsets["n_sectors_measured"].iloc[0] == 3

    summary = centroid_assess(conf, out)
    assert summary["verdict"] in TARGET_VERDICTS
    assert summary["n_targets"] == 1
    t = summary["targets"][0]
    # Every verdict carries the numbers that justify it.
    for k in ("offset_arcsec", "offset_err_arcsec", "offset_sigma",
              "offset_detectable_arcsec", "resolution_required_arcsec"):
        assert k in t
    assert t["census_statement"]
    assert summary["checks_not_performed"]
    # The transit was injected on the target, and no neighbour survives the
    # census arithmetic: the run must not come back OFFSET_FROM_TARGET.
    assert summary["verdict"] != OFFSET_FROM_TARGET
    assert summary["verdict"] == ON_TARGET
    assert t["census_excludes_all_neighbours"] is True
    assert t["n_neighbours"] == 2 and t["n_neighbours_not_excluded"] == 0


def test_a_run_that_reaches_no_pixels_is_no_data_reached_and_not_on_target(tmp_path: Path):
    out = tmp_path / "centroid"
    conf = _conf()

    def cone(_t, _r, **_kw):
        return _gaia_rows([(0.0, 0.1, 0.0), (1.0, 7.0, 45.0)]), []

    def boom(_tic, **_kw):
        raise RuntimeError("MAST 503")

    centroid_census_stage(conf, out, cone_fn=cone, query_fn=_koi_query_fn)
    centroid_difference_stage(conf, out, tpf_fn=boom, query_fn=_koi_query_fn)
    summary = centroid_assess(conf, out)
    assert summary["verdict"] == NO_DATA
    assert summary["target_verdicts"][ON_TARGET] == 0
    t = summary["targets"][0]
    assert t["verdict"] == NO_DATA and t["reason"] == STATUS_FAILED
    # The census still ran, and its arithmetic is still on the record.
    assert t["census_statement"]
    assert summary["targets"][0]["n_neighbours"] == 1


def test_centroid_run_dispatches_stages_and_rejects_an_unknown_one(tmp_path: Path):
    conf = _conf()
    rep = centroid_run("probe", out_dir=tmp_path / "c", conf=conf)
    assert rep["stage"] == "probe"
    with pytest.raises(SystemExit):
        centroid_run("nonsense", out_dir=tmp_path / "c", conf=conf)


def test_the_cli_entry_point_runs_the_probe(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert centroid_main(["--stage", "probe", "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "probe.json").exists()


def test_the_probe_is_imports_only_and_never_queries():
    rep = tpf_probe()
    assert set(rep) >= {"lightkurve", "astroquery_mast", "astropy_io_fits", "route_preferred"}
    assert "import reachability only" in rep["note"]


def test_params_come_from_the_config_block():
    conf = {"centroid": {"census": {"radius_arcsec": 30.0, "psf_sigma_arcsec": 9.0},
                         "difference": {"n_offset": 5.0, "in_transit_fraction": 0.5},
                         "tpf": {"max_sectors": 2, "budget_s": 60.0,
                                 "authors": ["SPOC"], "sectors": [41, 54]}}}
    cp = CensusParams.from_config(conf)
    dp = DiffParams.from_config(conf)
    tp = TpfParams.from_config(conf)
    assert cp.radius_arcsec == 30.0 and cp.psf_sigma_arcsec == 9.0
    assert dp.n_offset == 5.0 and dp.in_transit_fraction == 0.5
    assert tp.max_sectors == 2 and tp.budget_s == 60.0
    assert tp.authors == ("SPOC",) and tp.sectors == (41, 54)
    # An absent block leaves every default in place.
    assert CensusParams.from_config({}).radius_arcsec == 21.0


def test_the_shortlist_falls_back_to_stage_2s_and_the_depth_comes_from_its_measurements(
        tmp_path: Path):
    conf = {"stage2": {"shortlist": [{"kepoi_name": "K00897.01", "tic_id": 268924036}],
                       "shortlist_from_candidates_csv": False},
            "centroid": {}}
    sl = load_shortlist(conf)
    assert list(sl["kepoi_name"]) == ["K00897.01"]
    p = tmp_path / "measurements.csv"
    pd.DataFrame([{"kepoi_name": "K00897.01", "depth_measured_ppm": DEPTH_PPM,
                   "depth_measured_err_ppm": 1194.6}]).to_csv(p, index=False)
    depths = load_measured_depths(p)
    assert depths["K00897.01"][0] == pytest.approx(DEPTH_PPM)
    assert load_measured_depths(tmp_path / "nope.csv") == {}

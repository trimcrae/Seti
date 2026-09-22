"""Offline tests for the LANTERN data path rebuilt after the first archive run:
the table-per-segment ``x1dints`` layout, the level-3/level-2 planner, the
eclipse-first shard order, integration binning, the verify stage on known
eclipse windows, and checkpoint versioning (docs/lantern.md section 3.1/3.7).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.lantern import run as R
from seti.lantern.phase import Ephemeris, label_integrations, predict_phase_class
from seti.lantern.synth import synthesise_timeseries

LINE_WL = 4.05
_CONF = R.load_lantern_config()


def _layout_b_file(path, n_int, t0_mjd, wl_by_order: dict, segments: int = 2,
                   int_times: bool = True, tdb_cols: bool = True, cadence_d: float = 0.001,
                   base_int: int = 0, instrume="NIRISS", extra_header=None):
    """An x1dints file in the TSOMultiSpecModel layout: one EXTRACT1D table per
    (segment, spectral order), one ROW per integration, 2-D spectral columns,
    per-row INT_NUM / MJD-* / TDB-* columns, plus INT_TIMES."""
    from astropy.io import fits

    hdus = [fits.PrimaryHDU()]
    hdus[0].header.update({"INSTRUME": instrume, "FILTER": "CLEAR", "PUPIL": "GR700XD",
                           "NINTS": n_int * segments, "EXPSTART": t0_mjd,
                           "EXPEND": t0_mjd + cadence_d * n_int * segments, "CAL_VER": "1.19.x"})
    if extra_header:
        hdus[0].header.update(extra_header)
    all_nums, all_mid = [], []
    for s in range(segments):
        nums = base_int + s * n_int + np.arange(1, n_int + 1)
        mid = t0_mjd + cadence_d * (nums - 1) + 0.5 * cadence_d
        all_nums.append(nums)
        all_mid.append(mid)
        for order, wl in wl_by_order.items():
            n_wl = wl.size
            flux = 100.0 + nums[:, None] + 0.0 * wl[None, :]          # row i carries its INT_NUM
            cols = [fits.Column(name="INT_NUM", format="J", array=nums)]
            cols += [fits.Column(name="MJD-BEG", format="D", array=mid - 0.5 * cadence_d),
                     fits.Column(name="MJD-AVG", format="D", array=mid + 0.0003),  # UTC, offset
                     fits.Column(name="MJD-END", format="D", array=mid + 0.5 * cadence_d)]
            if tdb_cols:
                cols += [fits.Column(name="TDB-BEG", format="D", array=mid - 0.5 * cadence_d),
                         fits.Column(name="TDB-MID", format="D", array=mid),
                         fits.Column(name="TDB-END", format="D", array=mid + 0.5 * cadence_d)]
            dim = f"({n_wl})"
            cols += [fits.Column(name="WAVELENGTH", format=f"{n_wl}D", dim=dim,
                                 array=np.tile(wl, (n_int, 1))),
                     fits.Column(name="FLUX", format=f"{n_wl}D", dim=dim, array=flux),
                     fits.Column(name="FLUX_ERROR", format=f"{n_wl}D", dim=dim,
                                 array=np.full((n_int, n_wl), 1.0)),
                     fits.Column(name="DQ", format=f"{n_wl}J", dim=dim,
                                 array=np.zeros((n_int, n_wl), int))]
            h = fits.BinTableHDU.from_columns(cols, name="EXTRACT1D")
            h.header["SPORDER"] = order
            h.header["EXTVER"] = s * len(wl_by_order) + order
            hdus.append(h)
    if int_times:
        it = fits.BinTableHDU.from_columns(
            [fits.Column(name="integration_number", format="J", array=np.concatenate(all_nums)),
             fits.Column(name="int_start_MJD_UTC", format="D", array=np.concatenate(all_mid) - 0.0005),
             fits.Column(name="int_mid_BJD_TDB", format="D", array=np.concatenate(all_mid))],
            name="INT_TIMES")
        hdus.append(it)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_read_x1dints_table_per_segment_layout(tmp_path):
    """The layout every product in MAST now carries (the one the first screen
    misread as one integration per HDU): rows are integrations, grids are
    spectral orders, and times come from the per-row TDB-MID column."""
    from seti.lantern.acquire import group_segments_by_grid, read_x1dints, read_x1dints_grids

    wl1 = np.linspace(2.8, 0.85, 40)                    # order 1, descending like NIRISS
    wl2 = np.linspace(1.4, 0.6, 30)                     # order 2
    p = tmp_path / "jw01_nis_x1dints.fits"
    _layout_b_file(p, n_int=6, t0_mjd=60000.0, wl_by_order={1: wl1, 2: wl2}, segments=2)
    grids = read_x1dints_grids(p)
    assert len(grids) == 2
    g1 = grids[0]                                       # largest grid first = order 1
    assert g1["flux"].shape == (12, 40) and g1["flux_err"].shape == (12, 40)
    assert g1["grid"]["sporder"] == 1 and g1["grid"]["n_hdus"] == 2
    assert g1["time_source"] == "row_bjd_tdb"
    assert abs(g1["times"][0] - (2460000.5 + 0.0005)) < 1e-9 and np.all(np.diff(g1["times"]) > 0)
    assert g1["wavelength"][0] < g1["wavelength"][-1]   # ascending
    # Row i carries INT_NUM 100+i; the concatenation must follow the times.
    assert np.allclose(g1["flux"][:, 0], 100.0 + np.arange(1, 13))
    assert grids[1]["flux"].shape == (12, 30) and grids[1]["grid"]["sporder"] == 2
    assert read_x1dints(p)["flux"].shape == (12, 40)
    assert any(h["n_int"] == 6 for h in g1["meta"]["hdu_layout"])
    # A single-HDU file (a level-2 segment): the case that returned read_failed 656 times.
    p2 = tmp_path / "jw01-seg003_nis_x1dints.fits"
    _layout_b_file(p2, n_int=5, t0_mjd=60000.012, wl_by_order={1: wl1}, segments=1, base_int=12)
    g = read_x1dints_grids(p2)
    assert len(g) == 1 and g[0]["flux"].shape == (5, 40)
    # Segments of one exposure join per grid, in time order.
    joined = group_segments_by_grid(grids + g)
    assert len(joined) == 2 and joined[0]["flux"].shape == (17, 40)
    assert np.all(np.diff(joined[0]["times"]) > 0) and joined[0]["n_segments"] == 2
    # Without per-row TDB columns the INT_TIMES table is matched on INT_NUM.
    p3 = tmp_path / "jw02_nis_x1dints.fits"
    _layout_b_file(p3, n_int=4, t0_mjd=60001.0, wl_by_order={1: wl1}, segments=2, tdb_cols=False)
    g3 = read_x1dints_grids(p3)[0]
    assert g3["time_source"] == "int_times_bjd_tdb"
    assert abs(g3["times"][3] - (2460001.5 + 3 * 0.001 + 0.0005)) < 1e-9
    # Without either: the UTC MJD-AVG column, flagged as no barycentric correction.
    p4 = tmp_path / "jw03_nis_x1dints.fits"
    _layout_b_file(p4, n_int=4, t0_mjd=60002.0, wl_by_order={1: wl1}, segments=1,
                   tdb_cols=False, int_times=False)
    g4 = read_x1dints_grids(p4)[0]
    assert g4["time_source"] == "row_mjd_utc_no_barycentric"
    # Not a FITS file at all (the .png preview case): empty, not an exception.
    p5 = tmp_path / "x_x1dints.png"
    p5.write_bytes(b"\x89PNG\r\n")
    assert read_x1dints_grids(p5) == []


def test_known_eclipse_windows_from_the_archive_ephemerides():
    """Two published JWST eclipses through the labeller, from their MAST
    windows and archive ephemerides (inventory.json values):

    * WASP-43 b, MIRI/LRS phase curve jw01366-o011 (Bell et al. 2024): the
      window MJD 59914.028-59915.141 holds two eclipses bracketing one transit.
    * WASP-18 b, NIRISS/SOSS eclipse jw01366-o021 (Coulombe et al. 2023):
      MJD 59802.185-59802.496, one eclipse ~4.7 h after the start.
    """
    w43 = Ephemeris("WASP-43 b", 0.813475, 2455528.86774, 0.0483, 1e-6, 0.00014,
                    ecc=0.0, omega_deg=90.0, rp_rs=0.1594)
    t = np.linspace(59914.02769 + 2400000.5, 59915.14139 + 2400000.5, 2400)
    lab = label_integrations(t, w43, _CONF["phase"])
    assert lab["phase_class"] == "both"
    ecl = [e for e in lab["eclipses"] if not e["unplaceable"]]
    tr = [e for e in lab["transits"] if not e["unplaceable"]]
    assert len(ecl) == 2 and len(tr) == 1
    # Independent arithmetic: transit epoch 5392 sits inside the window, the
    # eclipses half a period either side of it.
    t_tr = 2455528.86774 + 5392 * 0.813475
    assert abs(tr[0]["mid"] - t_tr) < 1e-6
    assert abs(ecl[0]["mid"] - (t_tr - 0.5 * 0.813475)) < 1e-6
    assert abs(ecl[1]["mid"] - (t_tr + 0.5 * 0.813475)) < 1e-6
    assert (ecl[0]["mid"] - t.min()) * 24 == pytest.approx(4.6, abs=0.2)      # hours after start
    tin = t[lab["in_eclipse"]]
    assert np.all(np.min(np.abs(tin[:, None] - np.array([e["mid"] for e in ecl])[None, :]), axis=1)
                  < 0.5 * 0.0483 - ecl[0]["ingress_duration"])
    assert lab["coverage"]["n_baseline_before_eclipse_ingress"] > 100
    assert lab["in_transit"].sum() > 30 and not (lab["in_transit"] & lab["in_eclipse"]).any()
    pred = predict_phase_class([w43], t.min(), t.max(), _CONF["phase"])
    assert pred["phase_class"] == "both" and pred["rank"] == 0 and pred["n_eclipses"] == 2

    w18 = Ephemeris("WASP-18 b", 0.94145223, 2456740.8056, 0.09208333, 2.4e-7, 0.00019,
                    ecc=0.0051, omega_deg=-85.0, rp_rs=0.1018)
    t = np.linspace(59802.18480 + 2400000.5, 59802.49567 + 2400000.5, 1300)
    lab = label_integrations(t, w18, _CONF["phase"])
    assert lab["phase_class"] == "eclipse"
    ecl = [e for e in lab["eclipses"] if not e["unplaceable"]]
    assert len(ecl) == 1 and not lab["in_transit"].any()
    assert abs(ecl[0]["mid"] - (2456740.8056 + 3252.5 * 0.94145223)) < 1e-6
    assert (ecl[0]["mid"] - t.min()) * 24 == pytest.approx(4.66, abs=0.1)
    assert lab["coverage"]["n_in_eclipse"] > 50 and lab["coverage"]["n_baseline_before_eclipse_ingress"] > 100
    assert ecl[0]["timing_sigma"] < 0.002                                    # 3 minutes
    pred = predict_phase_class([w18], t.min(), t.max(), _CONF["phase"])
    assert pred["phase_class"] == "eclipse" and pred["planet"] == "WASP-18 b"
    # A window that misses both events is unresolved, and a missing window is too.
    pred = predict_phase_class([w18], 2459802.95, 2459803.10, _CONF["phase"])
    assert pred["phase_class"] == "phase_unresolved" and pred["rank"] == 2
    assert predict_phase_class([w18], None, None)["notes"] == ["no_window"]


def _fake_inventory():
    w43 = {"name": "WASP-43 b", "period": 0.813475, "t0": 2455528.86774, "duration": 0.0483,
           "period_err": 1e-6, "t0_err": 0.00014, "ecc": 0.0, "omega_deg": 90.0, "rp_rs": 0.1594,
           "notes": []}

    def item(fn, size, lvl=3, rights="PUBLIC"):
        return {"filename": fn, "uri": f"mast:JWST/product/{fn}", "size": size,
                "calib_level": lvl, "dataRights": rights}

    l3 = "jw01366-o011_t002_miri_p750l-slitlessprism"
    l2 = "jw01366011001_04103_00001_mirimage"
    obs = [
        {"obsid": "1", "obs_id": l3, "instrument": "MIRI/SLITLESS", "filters": "P750L",
         "t_min": 59914.02769, "t_max": 59915.14139, "calib_level": 3, "dataRights": "PUBLIC",
         "proposal_id": "1366", "exposures": {
             l3: [item(l3 + "_x1dints.fits", 500), item(l3 + "_x1dints.png", 1)],
             l2: [item("jw01366011001_04103_00001-seg001_mirimage_x1dints.fits", 250, 2),
                  item("jw01366011001_04103_00001-seg002_mirimage_x1dints.fits", 250, 2)]}},
        {"obsid": "2", "obs_id": "jw01366011001_04103_00001-seg001_mirimage", "instrument": "MIRI/SLITLESS",
         "filters": "P750L", "t_min": 59914.02769, "t_max": 59914.5, "calib_level": 2,
         "dataRights": "PUBLIC", "proposal_id": "1366", "exposures": {
             l2: [item("jw01366011001_04103_00001-seg001_mirimage_x1dints.fits", 250, 2)]}},
        # A transit-only visit (window centred on transit epoch 5392).
        {"obsid": "3", "obs_id": "jw01366-o099_t002_miri_p750l-slitlessprism", "instrument": "MIRI/SLITLESS",
         "filters": "P750L", "t_min": 2455528.86774 + 5392 * 0.813475 - 0.15 - 2400000.5,
         "t_max": 2455528.86774 + 5392 * 0.813475 + 0.15 - 2400000.5, "calib_level": 3,
         "dataRights": "PUBLIC", "proposal_id": "1366", "exposures": {
             "jw01366-o099_t002_miri_p750l-slitlessprism": [
                 item("jw01366-o099_t002_miri_p750l-slitlessprism_x1dints.fits", 300)]}},
        # Proprietary eclipse visit: counted, never scheduled.
        {"obsid": "4", "obs_id": "jw09999-o001_t001_miri_p750l-slitlessprism", "instrument": "MIRI/SLITLESS",
         "filters": "P750L", "t_min": 59914.02769, "t_max": 59915.14139, "calib_level": 3,
         "dataRights": "EXCLUSIVE_ACCESS", "proposal_id": "9999", "exposures": {
             "jw09999-o001_t001_miri_p750l-slitlessprism": [
                 item("jw09999-o001_t001_miri_p750l-slitlessprism_x1dints.fits", 900, 3,
                      "EXCLUSIVE_ACCESS")]}},
    ]
    return {"targets": {"WASP-43": {"host": "WASP-43", "ra": 154.9, "dec": -9.8, "planets": [w43],
                                    "observations": obs}}, "funnel": {}}


def test_plan_units_prefers_level3_drops_previews_and_orders_eclipses_first(tmp_path):
    inv = R.plan_units(_fake_inventory(), _CONF, n_shards=2)
    units = {u["exposure_key"]: u for u in inv["units"]}
    assert len(units) == 4
    l3 = units["jw01366-o011_t002_miri_p750l-slitlessprism"]
    l2 = units["jw01366011001_04103_00001_mirimage"]
    assert l3["scheduled"] and l3["n_products"] == 1 and l3["level"] == 3     # no png
    assert not l2["scheduled"] and l2["skip_reason"] == "level3_product_holds_these_segments"
    assert l2["n_products"] == 2                                              # merged across rows
    assert l3["predicted"]["phase_class"] == "both" and l3["rank"] == 0
    tr = units["jw01366-o099_t002_miri_p750l-slitlessprism"]
    assert tr["scheduled"] and tr["predicted"]["phase_class"] == "transit" and tr["rank"] == 1
    prop = units["jw09999-o001_t001_miri_p750l-slitlessprism"]
    assert not prop["scheduled"] and prop["skip_reason"] == "proprietary"
    # Eclipse-class first in the flat order; two shards, round-robin.
    order = [u["exposure_key"] for u in inv["units"] if u["scheduled"]]
    assert order[0] == l3["exposure_key"] and order[1] == tr["exposure_key"]
    assert inv["shards"] == [[l3["unit"]], [tr["unit"]]]
    assert inv["plan"]["n_scheduled"] == 2 and inv["plan"]["scheduled_bytes"] == 800
    assert inv["plan"]["by_class"]["skipped_level3_product_holds_these_segments"]["units"] == 1
    # When the level-3 file is much smaller than its segments it is missing a
    # detector: use the segments instead, never both.
    inv2 = _fake_inventory()
    inv2["targets"]["WASP-43"]["observations"][0]["exposures"][
        "jw01366-o011_t002_miri_p750l-slitlessprism"][0]["size"] = 100
    inv2 = R.plan_units(inv2, _CONF, n_shards=1)
    u2 = {u["exposure_key"]: u for u in inv2["units"]}
    assert u2["jw01366011001_04103_00001_mirimage"]["scheduled"]
    assert not u2["jw01366-o011_t002_miri_p750l-slitlessprism"]["scheduled"]
    assert u2["jw01366-o011_t002_miri_p750l-slitlessprism"]["skip_reason"].startswith("level3_bytes_0.20")
    # Re-planning an existing inventory file needs no archive.
    p = tmp_path / "old_inventory.json"
    p.write_text(json.dumps(_fake_inventory()))
    out = R.inventory(tmp_path / "out", _CONF, n_shards=3, replan_from=p)
    assert out["plan"]["n_scheduled"] == 2 and out["replanned_from"] == str(p)
    assert (tmp_path / "out" / "shards.json").exists()


def test_bin_integrations_and_factor():
    from seti.lantern.acquire import bin_integrations

    s = synthesise_timeseries(line_amp=0.02)
    b = bin_integrations(s, 3)
    assert b["flux"].shape == (100, 600) and b["binned_by"] == 3 and b["n_int_raw"] == 300
    assert np.allclose(b["times"][0], s["times"][:3].mean())
    assert np.allclose(b["flux"][0], s["flux"][:3].mean(axis=0))
    assert np.allclose(b["flux_err"][0], np.sqrt((s["flux_err"][:3] ** 2).sum(axis=0)) / 3)
    assert bin_integrations(s, 1) is s
    acq = dict(_CONF["acquire"], max_samples_in_memory=20000)
    k = R._bin_factor(s, [s["ephemeris"]], acq)
    cad = float(np.median(np.diff(s["times"])))
    assert k >= 1 and k * cad <= min(3.0 / 1440.0, 0.25 * s["ephemeris"].duration * 0.1) + 1e-12
    assert R._bin_factor(s, [s["ephemeris"]], dict(acq, max_samples_in_memory=1e9)) == 1
    # Through the exposure wrapper the binned stack still yields the candidate.
    conf = dict(_CONF, acquire=acq)
    s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
    s["time_source"] = "row_bjd_tdb"
    rec = R.analyse_exposure([s], [s["ephemeris"]], conf, "X")
    assert rec["binned_by"] > 1 and rec["n_integrations_raw"] == 300 and rec["n_grids"] == 1
    near = [f for f in rec["features"] if abs(f["wavelength"] - LINE_WL) < 0.01]
    assert near and near[0]["tier_local"] == "candidate" and near[0]["grid"] == 0


def test_analyse_exposure_merges_two_grids():
    s1 = synthesise_timeseries(line_amp=0.02)
    s2 = synthesise_timeseries(line_amp=0.0, wl_range=(1.0, 1.8), n_wl=300, seed=3)
    for s in (s1, s2):
        s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
        s["time_source"] = "row_bjd_tdb"
    rec = R.analyse_exposure([s1, s2], [s1["ephemeris"]], _CONF, "X")
    assert rec["n_grids"] == 2 and len(rec["grids"]) == 2
    assert rec["n_scanned"] == rec["grids"][0]["n_scanned"] + rec["grids"][1]["n_scanned"]
    assert {f["grid"] for f in rec["features"]} <= {0, 1}
    near = [f for f in rec["features"] if abs(f["wavelength"] - LINE_WL) < 0.01 and f["grid"] == 0]
    assert near and near[0]["tier_local"] == "candidate"


def test_verify_eclipse_stack_on_a_synthetic_eclipse():
    """The verify stage's measurement: a 0.2% eclipse in the continuum must be
    found at the predicted contacts, and an injected vanishing line recovered."""
    s = synthesise_timeseries(line_amp=0.0, eclipse_depth=0.002, ramp_amp=0.003, ramp_tau=40.0)
    s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
    s["time_source"] = "row_bjd_tdb"
    v = R.verify_eclipse_stack(s, [s["ephemeris"]], _CONF, injection_amp=0.02)
    assert v["phase_class"] == "eclipse" and v["passed"], v["checks"]
    # A linear detrend leaves part of the exponential settling ramp in the depth.
    assert v["depth"] == pytest.approx(0.002, abs=0.0006) and v["depth_snr"] > 5
    assert abs(v["free_step_offset_days"]) <= v["timing_tolerance_days"]
    assert v["injection"]["recovered"] and v["injection"]["tier"] in ("candidate", "interest")
    assert v["injection"]["vetoes"] == [] and v["checks"]["injected_line_recovered_clean"]
    # No eclipse in the window: the stage says so instead of passing.
    s2 = synthesise_timeseries(line_amp=0.0, centre="none")
    s2["time_source"] = "row_bjd_tdb"
    v2 = R.verify_eclipse_stack(s2, [s2["ephemeris"]], _CONF)
    assert not v2["passed"] and v2["reason"] == "labeller_placed_no_eclipse_in_window"
    # A flat light curve (no eclipse signal) fails the depth check honestly.
    s3 = synthesise_timeseries(line_amp=0.0, eclipse_depth=0.0)
    s3["time_source"] = "row_bjd_tdb"
    v3 = R.verify_eclipse_stack(s3, [s3["ephemeris"]], _CONF)
    assert not v3["passed"] and not v3["checks"]["depth_positive_and_significant"]


def test_verify_injection_scales_to_the_exposures_own_sensitivity():
    """A FIXED injected amplitude tests nothing on a noisy exposure: on the real
    products 2% of the continuum sits below the 5-sigma equivalent-width limit,
    so 'not recovered' is a statement about the injection, not the chain.  The
    amplitude must instead be set from the exposure's own measured noise."""
    s = synthesise_timeseries(line_amp=0.0, eclipse_depth=0.004, noise=2e-2,
                              fixed_pattern_amp=0.01, seed=11)
    s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
    s["time_source"] = "row_bjd_tdb"
    floor = 0.005
    # Fixed 0.5% of the continuum: below this exposure's limit, nothing comes back.
    weak = R.verify_eclipse_stack(s, [s["ephemeris"]], _CONF, injection_amp=floor,
                                  injection_snr_target=0.0)
    assert weak["injection"]["amp"] == pytest.approx(floor)
    assert weak["injection"]["injected_ew_over_5sigma_limit"] < 1.0
    assert not weak["injection"]["recovered"] and not weak["injection_passed"]
    # Scaled to the measured difference-spectrum noise, the same chain finds it.
    strong = R.verify_eclipse_stack(s, [s["ephemeris"]], _CONF, injection_amp=floor,
                                    injection_snr_target=12.0)
    inj = strong["injection"]
    assert inj["amp"] > floor and inj["amp"] == pytest.approx(
        12.0 * inj["baseline_noise_median"], rel=1e-6)
    assert inj["injected_ew_over_5sigma_limit"] > 1.5
    assert inj["recovered"] and strong["injection_passed"]
    assert inj["tier"] in ("candidate", "interest") and inj["vetoes"] == []
    # The phase question is answered either way -- it does not depend on the
    # injection, which is what lets the screen gate on it.
    assert weak["phase_passed"] and strong["phase_passed"]
    # The injected sigma tracks the sampling, so the line is never a
    # sub-resolution-element spike on a coarsely sampled grid (which the search
    # vetoes by design).
    spr = float(R.instrument_profile(_CONF, "NIRSPEC", "G395H", None, None)["samples_per_resel"])
    assert inj["sigma_samples"] == pytest.approx(max(0.55 * spr, 0.6))


def test_verify_stage_with_stubbed_archive(tmp_path):
    inv = _fake_inventory()
    inv["targets"]["WASP-43"]["planets"][0].update(
        {"period": 1.5, "t0": 2460000.0, "duration": 2.0 / 24, "rp_rs": 0.1})
    R._write_json(tmp_path / "inventory.json", inv)
    conf = dict(_CONF, verify={"injection_amp": 0.02, "cases": [
        {"name": "synthetic_case", "host": "WASP-43", "exposure_key": "e", "expect": "eclipse",
         "uris": ["mast:JWST/product/e_x1dints.fits"]},
        {"name": "missing_host", "host": "Nobody", "uris": ["mast:JWST/product/f_x1dints.fits"]}]})

    def dl(uri, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"x")
        return "COMPLETE"

    def rd(path):
        s = synthesise_timeseries(line_amp=0.0, eclipse_depth=0.002)
        s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
        s["time_source"] = "row_bjd_tdb"
        return [s]

    conf["acquire"] = dict(conf["acquire"])
    conf["acquire"]["retries"] = 1
    import seti.lantern.acquire as A
    orig = A.fetch_transiting_planets
    A.fetch_transiting_planets = lambda *a, **k: pd.DataFrame()
    try:
        res = R.verify(tmp_path, conf, work_dir=tmp_path / "w", download_fn=dl, read_fn=rd)
    finally:
        A.fetch_transiting_planets = orig
    assert res["verdict"] == "PHASE_PARTIALLY_VERIFIED" and res["n_passed"] == 1
    assert res["cases"]["synthetic_case"]["passed"]
    assert res["cases"]["missing_host"]["reason"] == "no_ephemeris_for_host"
    assert (tmp_path / "verify.json").exists()


def test_screen_redoes_stale_checkpoints_and_honours_the_deadline(tmp_path):
    planets = pd.DataFrame([{"pl_name": "X b", "hostname": "X", "ra": 10.0, "dec": -5.0,
                             "pl_orbper": 1.5, "pl_tranmid": 2460000.0, "pl_trandur": 2.0,
                             "pl_ratror": 0.1, "pl_orbeccen": 0.0, "pl_orblper": 90.0}])
    obs = pd.DataFrame([{"obsid": "1", "obs_id": "jw01001-o001_x", "s_ra": 10.0, "s_dec": -5.0,
                         "instrument_name": "NIRSPEC/SLIT", "filters": "G395H", "dataRights": "PUBLIC",
                         "target_name": "X", "t_min": 60015.1, "t_max": 60015.4, "t_exptime": 0,
                         "calib_level": 3, "proposal_id": "1"}])       # MJD: eclipse at JD 2460015.75
    prods = pd.DataFrame([{"parent_obsid": "1", "productFilename": "jw01001-o001_x_x1dints.fits",
                           "dataURI": "mast:JWST/a", "size": 10, "calib_level": 3, "dataRights": "PUBLIC"},
                          {"parent_obsid": "1", "productFilename": "jw01001-o001_x_x1dints.png",
                           "dataURI": "mast:JWST/a.png", "size": 1, "calib_level": 3, "dataRights": "PUBLIC"},
                          {"parent_obsid": "1", "productFilename": "jw01001001001_04101_00001-seg001_nrs1_x1dints.fits",
                           "dataURI": "mast:JWST/b", "size": 10, "calib_level": 2, "dataRights": "PUBLIC"}])
    prods["exposure_key"] = prods["productFilename"].map(
        __import__("seti.lantern.acquire", fromlist=["exposure_key"]).exposure_key)
    inv = R.inventory(tmp_path, _CONF, fetch_planets_fn=lambda: planets, query_tso_fn=lambda i: obs,
                      list_products_fn=lambda o: prods, n_shards=1)
    assert inv["plan"]["n_scheduled"] == 1                      # level-3 only; png dropped
    assert inv["units"][0]["predicted"]["phase_class"] == "eclipse"
    calls = []

    def dl(uri, local):
        calls.append(uri)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"x")
        return "COMPLETE"

    def rd(path):
        s = synthesise_timeseries(line_amp=0.02)
        s["meta"] = {"INSTRUME": "NIRSPEC", "GRATING": "G395H"}
        s["time_source"] = "row_bjd_tdb"
        return [s]

    # A stale checkpoint (older reader) is re-analysed, not trusted.
    ck = tmp_path / "obs" / "X" / "jw01001_o001_x.json"
    R._write_json(ck, {"checkpoint_version": R.CHECKPOINT_VERSION - 1, "status": "read_failed"})
    log = R.screen(tmp_path, _CONF, shard=0, n_shards=1, download_fn=dl, read_fn=rd,
                   work_dir=tmp_path / "work")
    assert log["counts"]["stale_checkpoint_redone"] == 1 and log["counts"]["analysed"] == 1
    assert calls == ["mast:JWST/a"]
    rec = json.loads(ck.read_text())
    assert rec["checkpoint_version"] == R.CHECKPOINT_VERSION and rec["predicted"]["phase_class"] == "eclipse"
    assert rec["phase_class"] == "eclipse" and rec["n_grids"] == 1
    # Current checkpoint: skipped.  Deadline already passed: deferred, not started.
    log2 = R.screen(tmp_path, _CONF, shard=0, n_shards=1, download_fn=dl, read_fn=rd,
                    work_dir=tmp_path / "work")
    assert log2["counts"]["skipped_checkpoint"] == 1
    ck.unlink()
    log3 = R.screen(tmp_path, _CONF, shard=0, n_shards=1, download_fn=dl, read_fn=rd,
                    work_dir=tmp_path / "work", deadline_minutes=1e-9)
    assert log3["counts"]["deadline_deferred"] == 1 and len(calls) == 1
    summary = R.assess(tmp_path, _CONF)
    assert summary["funnel"]["exposure_checkpoints"] == 0

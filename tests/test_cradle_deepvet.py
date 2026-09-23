"""Offline tests for the CRADLE deep vet (no network: every route is injected)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seti.cradle import deepvet as dv
from seti.cradle.deepvet import Route

C = dv.DEFAULT_DEEPVET
T = {"source_id": "42", "role": "CANDIDATE", "ra_gaia": 150.0, "dec_gaia": 40.0,
     "ra_wise": 150.0, "dec_wise": 40.0, "ra_2000": 150.0, "dec_2000": 40.0,
     "w3mpro": 9.1, "w4mpro": 8.27, "W3_excess_jy": 0.0019, "W4_excess_jy": 0.0026,
     "t_bb_k": 270.0, "log_f_fmax_1gyr": 3.4}


def fake_tap(tables: dict):
    """Route by a key appearing in the ADQL; 'fail' entries raise QUERY_FAILED."""
    def tap(url, adql, label, retries=3):
        for key, val in tables.items():
            if key in adql:
                if isinstance(val, str) and val == "fail":
                    return Route(label, "QUERY_FAILED", error="boom")
                df = pd.DataFrame(val)
                return Route(label, "OK" if len(df) else "QUERY_RETURNED_ZERO_ROWS", len(df), data=df)
        return Route(label, "QUERY_RETURNED_ZERO_ROWS", 0, data=pd.DataFrame())
    return tap


def test_simbad_failure_is_untested_and_empty_is_tested():
    fail = dv.simbad_check(T, C, fake_tap({"FROM basic": "fail"}))
    assert fail["status"] == "UNTESTED"
    empty = dv.simbad_check(T, C, fake_tap({"FROM basic": []}))
    assert empty["status"] == "TESTED" and empty["in_simbad"] is False
    v = dv.judge(T, {"simbad": empty}, C)
    assert "simbad" in v["tested"] and not v["kills"]


def test_simbad_yso_type_on_star_kills_and_debris_title_flags():
    tap = fake_tap({
        "FROM basic": [{"oid": 7, "main_id": "V* XX Foo", "otype": "*", "sp_type": "G2V",
                        "nbref": 3, "ra": 150.0, "dec": 40.0}],
        "FROM otypes": [{"otype": "*"}, {"otype": "Y*O"}],
        "FROM has_ref": [{"bibcode": "2024X", "year": 2024, "title": "An extreme debris disk"}],
        "FROM ident": [{"id": "TYC 1-2-3"}],
    })
    s = dv.simbad_check(T, C, tap)
    v = dv.judge(T, {"simbad": s}, C)
    assert any(k.startswith("simbad_type:") for k in v["kills"])
    assert any(f.startswith("literature_debris_or_excess_refs") for f in v["flags"])


def test_simbad_galaxy_in_w3_beam_kills():
    tap = fake_tap({"FROM basic": [
        {"oid": 1, "main_id": "star", "otype": "PM*", "sp_type": "", "nbref": 1, "ra": 150.0, "dec": 40.0},
        {"oid": 2, "main_id": "gal", "otype": "G", "sp_type": "", "nbref": 1,
         "ra": 150.0, "dec": 40.0 + 4.0 / 3600}]})
    s = dv.simbad_check(T, C, tap)
    assert dv.judge(T, {"simbad": s}, C)["kills"][0].startswith("simbad_blend_object_in_w3_beam")


def _ls_rows(excess_on_star: bool):
    p3 = float(dv.mag_to_jy(T["w3mpro"], "W3")) - T["W3_excess_jy"]
    star_w3 = (p3 + (T["W3_excess_jy"] if excess_on_star else 0.0)) / 3.631e-6
    gal_w3 = (0.0 if excess_on_star else T["W3_excess_jy"]) / 3.631e-6
    return [
        {"ra": 150.0, "dec": 40.0, "type": "PSF", "ref_cat": "G3", "ref_id": 42, "brick_primary": True,
         "maskbits": 0, "shape_r": 0.0, "flux_g": 1e5, "flux_r": 1e5, "flux_z": 1e5, "flux_w1": 1e4,
         "flux_w2": 1e4, "flux_w3": star_w3, "flux_w4": 100.0, "flux_ivar_r": 1.0,
         "flux_ivar_w3": 1.0 / (0.02 * star_w3) ** 2, "flux_ivar_w4": 1e-4},
        {"ra": 150.0, "dec": 40.0 + 2.5 / 3600, "type": "EXP", "ref_cat": "", "ref_id": 0,
         "brick_primary": True, "maskbits": 0, "shape_r": 1.2, "flux_g": 3.0, "flux_r": 6.0,
         "flux_z": 9.0, "flux_w1": 5.0, "flux_w2": 5.0, "flux_w3": gal_w3, "flux_w4": 100.0,
         "flux_ivar_r": 1.0, "flux_ivar_w3": 1e-4, "flux_ivar_w4": 1e-4}]


def test_ls_galaxy_carrying_the_excess_kills():
    ls = dv.ls_check(T, C, fake_tap({"ls_dr10.tractor": _ls_rows(excess_on_star=False)}))
    v = dv.judge(T, {"ls": ls}, C)
    assert any("ls_galaxy_in_w3_beam_carries" in k for k in v["kills"])
    assert any("no_w3_excess_in_deblended" in k for k in v["kills"])


def test_ls_excess_on_the_star_survives_that_check():
    ls = dv.ls_check(T, C, fake_tap({"ls_dr10.tractor": _ls_rows(excess_on_star=True)}))
    assert ls["chi_w3_forced"] > 3
    v = dv.judge(T, {"ls": ls}, C)
    assert not any("deblended" in k for k in v["kills"])


def test_ls_out_of_footprint_is_not_tested():
    ls = dv.ls_check(T, C, fake_tap({}))
    assert ls["status"] == "NOT_IN_FOOTPRINT"
    assert "legacy_surveys" in dv.judge(T, {"ls": ls}, C)["untested"]


def test_allwise_flags_and_density():
    row = {"designation": "J1", "ra": 150.0, "dec": 40.0, "nb": 1, "na": 0, "ext_flg": 0,
           "cc_flags": "0000", "ph_qual": "AAAB", "w3snr": 40.0, "w4snr": 5.6, "w3nm": 10, "w3m": 12,
           "w4nm": 3, "w4m": 12, "w3mpro": 9.1, "w3sigmpro": 0.03, "w3mag": 8.9, "w3sigm": 0.03,
           "w4mpro": 8.27, "w4sigmpro": 0.18}
    nb = {**row, "designation": "J2", "dec": 40.0 + 8 / 3600, "w3snr": 5.0}
    aw = dv.allwise_check(T, C, fake_tap({"allwise_p3as_psd WHERE CONTAINS": [row, nb]}))
    v = dv.judge(T, {"allwise": aw}, C)
    assert any(f.startswith("w4_frame_detection_fraction") for f in v["flags"])
    assert any(f.startswith("w4_at_detection_threshold") for f in v["flags"])
    assert any(f.startswith("w3_aperture_brighter") for f in v["flags"])
    assert any(f.startswith("allwise_w3_neighbour_in_w4_beam") for f in v["flags"])
    rng = np.random.default_rng(1)
    field = [{"ra": 150 + rng.uniform(-0.1, 0.1), "dec": 40 + rng.uniform(-0.1, 0.1),
              "w3mpro": 8.0, "w3snr": 20.0, "w4mpro": 6.0, "w4snr": 10.0, "ext_flg": 0}
             for _ in range(400)]
    d = dv.density_check(T, C, fake_tap({"w3snr >= 3 OR": field}))
    assert d["status"] == "TESTED"
    p = d["by_fraction_of_excess"]["frac1"]["p_blend_w3"]
    n = d["by_fraction_of_excess"]["frac1"]["n_w3_ge"]
    assert n > 0
    dens = n / (d["area_arcmin2"] * 3600)
    assert p == pytest.approx(1 - math.exp(-dens * math.pi * C["blend_radius_arcsec"] ** 2))


def _gauss(nx, x0, y0, fwhm_px, amp):
    yy, xx = np.mgrid[0:nx, 0:nx]
    s = fwhm_px / 2.355
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))


def _hdr(nx, ra, dec):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [(nx + 1) / 2, (nx + 1) / 2]
    w.wcs.cdelt = [-2.75 / 3600, 2.75 / 3600]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w.to_header()


def _images(offset_px: float, excess_frac: float):
    nx = 161
    rng = np.random.default_rng(3)
    c = (nx - 1) / 2
    field = [(30.0, 40.0), (120.0, 35.0), (50.0, 125.0), (130.0, 130.0), (80.0, 30.0)]
    imgs = {}
    for band, fw in ((1, 6.1 / 2.75), (3, 6.5 / 2.75), (4, 12.0 / 2.75)):
        img = rng.normal(0, 1.0, (nx, nx))
        img += _gauss(nx, c, c, fw, 400.0 * (1 - (excess_frac if band > 1 else 0)))
        if band > 1:
            img += _gauss(nx, c + offset_px, c, fw, 400.0 * excess_frac)
        for fx, fy in field:
            img += _gauss(nx, fx, fy, fw, 300.0)
        imgs[band] = (img, _hdr(nx, 150.0, 40.0))
    return imgs


def test_image_offset_blend_is_killed_and_centred_excess_survives():
    blend = dv.image_check(T, C, fetch_primary=lambda ra, dec, n: (_images(1.2, 0.35), {}),
                           fetch_fallback=lambda *a: ({}, {}))
    assert blend["status"] == "TESTED"
    assert blend["result"]["w3"]["offset_from_w1_arcsec"] > 0.8
    assert any(k.startswith("w3_centroid_offset") for k in dv.judge(T, {"image": blend}, C)["kills"])
    ok = dv.image_check(T, C, fetch_primary=lambda ra, dec, n: (_images(0.0, 0.35), {}),
                        fetch_fallback=lambda *a: ({}, {}))
    assert ok["result"]["w3"]["offset_from_w1_arcsec"] < 0.3
    assert not dv.judge(T, {"image": ok}, C)["kills"]


def test_image_route_failure_is_untested():
    r = dv.image_check(T, C, fetch_primary=lambda *a: ({}, {"http": 503}),
                       fetch_fallback=lambda *a: ({}, {"http": 503}))
    assert r["status"] == "UNTESTED"
    assert "image_centroid" in dv.judge(T, {"image": r}, C)["untested"]


def test_run_deepvet_end_to_end_offline(tmp_path):
    sl = pd.DataFrame([{"source_id": 42, "ra": 150.0, "dec": 40.0, "pmra": 10.0, "pmdec": -5.0,
                        "parallax": 5.0, "phot_g_mean_mag": 11.0, "w1mpro": 9.4, "w2mpro": 9.4,
                        "w3mpro": 9.1, "w4mpro": 8.27, "w3_snr": 40, "w4_snr": 6,
                        "W3_excess_jy": 0.0019, "W4_excess_jy": 0.0026, "chi_W3": 6.6, "chi_W4": 3.8,
                        "t_bb_k": 270.0, "log_f_fmax_1gyr": 3.4, "f_ir": 7e-4}])
    sl.to_csv(tmp_path / "shortlist.csv", index=False)
    (tmp_path / "summary.json").write_text(json.dumps({
        "generated_utc": "x", "verdict": "v", "funnel": {"n_ks_w1_photospheric": 1000},
        "candidates": [{"source_id": 42}], "in_cell_age_undetermined": [], "controls": []}))
    rep = dv.run_deepvet(tmp_path, routes={
        "tap": fake_tap({"FROM basic": "fail"}),
        "vizier": lambda t, c, footprint=True: {"status": "UNTESTED"},
        "image": lambda t, c: {"status": "UNTESTED"}})
    assert rep["n_candidates_in"] == 1
    assert rep["targets"][0]["verdict"] == "SURVIVES_WITH_UNTESTED"
    assert (tmp_path / "deepvet_targets.csv").exists()

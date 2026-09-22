"""Offline tests for the CRYPT screen the PDS holdings actually support:
diurnal/seasonal invariance of bolometric temperature inside a mapped PSR.

No network (``conftest.py`` raises on any socket).  The suite

* reproduces the closed-form bolometric mixing and its inverse;
* returns a CLEAN NULL on the un-injected synthetic bowl, and states its
  measured detection floor;
* RECOVERS an injected internal source of a few square metres in both
  seasons at every local time;
* rejects each dominant confounder by its own named rule — a summer-only
  source as ``seasonal``, a day-bins-only source as ``diurnal``, a patch as
  ``extended``, a row as ``stripe``, a landing site as ``human_hardware``, a
  sparsely observed pixel as ``low_coverage``;
* shows the windowed injection–recovery table agrees with the whole-map one,
  which is the only reason the table can be produced on the 2535 x 2535 grid;
* reads a REAL-FORMAT PCP ASCII table (five whitespace columns, one header
  record, empty bins omitted) onto the polar grid, at either grid origin and
  either line-axis sign, and reports a wrong registration through the
  residual and the collision fraction;
* registers the LOLA PSR raster off its label's projection offsets, not the
  array centre;
* resolves the local-time bins from an ODE file listing and falls back to the
  verified URL template when ODE cannot be reached;
* runs pcp -> assess_diurnal end to end through a SCRIPTED archive that
  serves synthetic products in the real formats, and reports
  ``NO_DATA_REACHED`` — never a science null — when the archive serves
  nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seti.crypt import acquire as A
from seti.crypt import diurnal as D
from seti.crypt import pcp as PCP
from seti.crypt.labels import Georef
from seti.crypt.run import crypt_run, load_crypt_config, resolve_pcp_products, stage_pcp

CONF = load_crypt_config()
THR = {**D.DEFAULT_THRESHOLDS, **{k: v for k, v in CONF["diurnal"].items()
                                  if k in D.DEFAULT_THRESHOLDS}}
PIX = 240.0**2


@pytest.fixture(scope="module")
def cube_psr():
    return D.synthetic_cube(120, pole="south", seed=5, n_bins=24)


def _screen(cube, psr, **kw):
    return D.screen_pole(cube, THR, external_psr=psr, **kw)


def _row(rep, i, j):
    fl = rep["flags"]
    if fl is None or not len(fl):
        return None
    hit = fl[(fl["line"] == i) & (fl["sample"] == j)]
    return hit.iloc[0].to_dict() if len(hit) else None


def _interior_pixel(cube, psr, di=0, dj=0):
    m = D.cold_trap_mask(cube, THR, external=psr)
    ii, jj = np.nonzero(m["interior"])
    k = len(ii) // 2
    return int(ii[k]) + di, int(jj[k]) + dj


# ---------------------------------------------------------------------------
# physics: bolometric mixing is exact, so the inverse is closed form
# ---------------------------------------------------------------------------
def test_mixing_and_its_inverse_are_consistent():
    T_cold, T_hot, f = 40.0, 300.0, 1.0 / PIX
    T = float(D.mix_bolometric(T_cold, f, T_hot))
    assert T > T_cold
    a = D.floor_area_bolometric(T_cold, T - T_cold, T_hot, PIX)
    assert a == pytest.approx(1.0, rel=1e-6)


def test_zero_fraction_leaves_the_pixel_untouched():
    assert float(D.mix_bolometric(45.0, 0.0, 400.0)) == pytest.approx(45.0)


def test_floor_area_is_monotonic_in_the_excess():
    areas = [D.floor_area_bolometric(40.0, d, 300.0, PIX) for d in (0.1, 0.5, 2.0, 8.0)]
    assert all(b > a for a, b in zip(areas, areas[1:], strict=False))


def test_a_source_colder_than_the_pixel_it_would_heat_is_not_a_solution():
    assert not np.isfinite(D.floor_area_bolometric(40.0, 5.0, 42.0, PIX))


# ---------------------------------------------------------------------------
# the null, and the floor that makes the null a COUNT and not a limit
# ---------------------------------------------------------------------------
def test_clean_null_on_the_uninjected_bowl(cube_psr):
    cube, psr = cube_psr
    rep = _screen(cube, psr)
    assert rep["status"] == "OK"
    assert rep["mask"]["n_interior"] > 2000
    assert rep["counts"]["candidate"] == 0


def test_the_floor_is_stated_and_finite(cube_psr):
    cube, psr = cube_psr
    rep = _screen(cube, psr)
    fl = rep["floor"]
    assert fl["T_hot_K"] == 300.0
    assert np.isfinite(fl["sigma_K"]) and fl["sigma_K"] > 0
    assert np.isfinite(fl["A_min_m2"]) and 0.0 < fl["A_min_m2"] < PIX


def test_the_curvature_bias_is_removed_before_sigma_is_measured(cube_psr):
    """A bowl's annulus mean is not its centre value; without pass 2 that
    constant bias inflates sigma and hides real sources."""
    cube, psr = cube_psr
    m = D.cold_trap_mask(cube, THR, external=psr)
    st = D.statistics(cube, m["interior"], THR)
    bg = D.local_background(st["t_floor"], m["interior"] & np.isfinite(st["t_floor"]), THR)
    raw = np.nanstd(bg["residual"][m["interior"]])
    corrected = np.nanstd(bg["excess"][m["interior"]])
    assert corrected < raw


# ---------------------------------------------------------------------------
# recovery of an internal source
# ---------------------------------------------------------------------------
def test_an_internal_source_is_recovered_as_a_candidate(cube_psr):
    cube, psr = cube_psr
    i, j = _interior_pixel(cube, psr)
    rep = _screen(D.inject(cube, [(i, j)], 3.0 / PIX, 300.0), psr)
    r = _row(rep, i, j)
    assert r is not None, "injected source was not even flagged"
    assert r["class"] == "candidate", r["reasons"]
    assert r["z_floor"] >= THR["z_min"]
    assert 0.3 < r["area_m2_300K"] / 3.0 < 3.0


def test_a_source_below_the_floor_is_not_flagged(cube_psr):
    cube, psr = cube_psr
    i, j = _interior_pixel(cube, psr, di=7)
    rep = _screen(D.inject(cube, [(i, j)], 0.02 / PIX, 300.0), psr)
    r = _row(rep, i, j)
    assert r is None or r["class"] != "candidate"


# ---------------------------------------------------------------------------
# every confounder, by its own named rule
# ---------------------------------------------------------------------------
def test_a_summer_only_source_is_rejected_as_seasonal(cube_psr):
    cube, psr = cube_psr
    i, j = _interior_pixel(cube, psr, di=3)
    rep = _screen(D.inject(cube, [(i, j)], 300.0 / PIX, 300.0, seasons=["summer"]), psr)
    r = _row(rep, i, j)
    assert r is not None and r["class"] == "seasonal", r and r["reasons"]


def test_a_day_bins_only_source_is_rejected_as_diurnal(cube_psr):
    cube, psr = cube_psr
    i, j = _interior_pixel(cube, psr, di=-3)
    day = [f"ltim{h:02d}" for h in range(7, 19)]
    rep = _screen(D.inject(cube, [(i, j)], 300.0 / PIX, 300.0, bins=day), psr)
    r = _row(rep, i, j)
    assert r is not None and r["class"] in ("diurnal", "seasonal"), r and r["reasons"]
    assert "diurnal:" in r["reasons"] or "seasonal:" in r["reasons"]


def test_an_extended_warm_patch_is_rejected(cube_psr):
    cube, psr = cube_psr
    i, j = _interior_pixel(cube, psr, dj=5)
    px = [(i + a, j + b) for a in range(-2, 3) for b in range(-2, 3)]
    rep = _screen(D.inject(cube, px, 3.0 / PIX, 300.0), psr)
    r = _row(rep, i, j)
    assert r is not None and r["class"] == "extended", r and r["reasons"]
    assert r["cluster_px"] > THR["max_cluster_px"]


def test_a_row_of_hot_pixels_is_a_stripe(cube_psr):
    cube, psr = cube_psr
    i, j = _interior_pixel(cube, psr, dj=-8)
    px = [(i, j + 4 * k) for k in range(8)]
    rep = _screen(D.inject(cube, px, 3.0 / PIX, 300.0), psr)
    r = _row(rep, i, j)
    assert r is not None and r["class"] == "stripe", r and r["reasons"]


def test_a_pixel_inside_a_landing_site_is_flagged_as_human_hardware(cube_psr):
    cube, psr = cube_psr
    i, j = _interior_pixel(cube, psr)
    lon, lat = cube.georef.pix_to_lonlat(np.array([i]), np.array([j]))
    site = [{"name": "test site", "lat": float(lat[0]), "lon": float(lon[0]), "radius_m": 3000.0}]
    rep = _screen(D.inject(cube, [(i, j)], 3.0 / PIX, 300.0), psr, hardware=site)
    r = _row(rep, i, j)
    assert r is not None and r["class"] == "human_hardware", r and r["reasons"]


def test_a_sparsely_observed_pixel_cannot_be_a_candidate(cube_psr):
    cube, psr = cube_psr
    i, j = _interior_pixel(cube, psr, di=11)
    holed = cube.copy()
    for n, key in enumerate(sorted(holed.arrays)):
        if n % 3:
            holed.arrays[key][i, j] = np.nan
    rep = _screen(D.inject(holed, [(i, j)], 3.0 / PIX, 300.0), psr)
    r = _row(rep, i, j)
    assert r is None or r["class"] != "candidate"


def test_a_single_season_cube_is_reported_as_degraded(cube_psr):
    cube, psr = cube_psr
    one = D.DiurnalCube(cube.pole, cube.georef,
                        {k: v for k, v in cube.arrays.items() if k[0] == "summer"}, {})
    rep = D.screen_pole(one, {**THR, "n_bins_min": 4}, external_psr=psr)
    assert "single_season_only" in rep["degraded"]


def test_no_external_psr_raster_is_reported_not_assumed(cube_psr):
    cube, _psr = cube_psr
    rep = D.screen_pole(cube, THR)
    assert "no_external_psr_raster" in rep["degraded"]


def test_an_external_psr_raster_of_the_wrong_shape_is_named_not_silently_dropped(cube_psr):
    cube, _psr = cube_psr
    rep = D.screen_pole(cube, THR, external_psr=np.ones((7, 7), dtype=bool))
    assert any(d.startswith("external_psr_shape_") for d in rep["degraded"])


def test_an_empty_cube_gives_no_interior_never_a_null(cube_psr):
    cube, _psr = cube_psr
    rep = D.screen_pole(D.DiurnalCube(cube.pole, cube.georef), THR)
    assert rep["status"] == "NO_PSR_INTERIOR"
    assert rep["counts"]["candidate"] == 0


# ---------------------------------------------------------------------------
# the injection-recovery table
# ---------------------------------------------------------------------------
def test_sensitivity_recovers_large_sources_and_not_tiny_ones(cube_psr):
    cube, psr = cube_psr
    s = D.sensitivity(cube, THR, [0.02, 30.0], 300.0, n_per_area=4, seed=3, external_psr=psr)
    assert s["status"] == "OK"
    lo, hi = s["rows"][0], s["rows"][1]
    assert lo["recovered_frac"] == 0.0
    assert hi["recovered_frac"] > 0.5


def test_the_window_gives_the_same_table_as_the_whole_map(cube_psr):
    """The window is what makes the table affordable on the real grid; it is
    only legitimate because every rule in the screen is local."""
    cube, psr = cube_psr
    full = D.sensitivity(cube, THR, [30.0], 300.0, n_per_area=4, seed=3, external_psr=psr)
    win = D.sensitivity(cube, THR, [30.0], 300.0, n_per_area=4, seed=3, external_psr=psr,
                        window_px=301)
    assert win["rows"][0]["recovered_frac"] == full["rows"][0]["recovered_frac"]


def test_crop_cube_keeps_the_georeference(cube_psr):
    cube, _psr = cube_psr
    sub = D.crop_cube(cube, 10, 20, 40)
    lon_s, lat_s = sub.georef.pix_to_lonlat(np.array([5]), np.array([6]))
    lon_f, lat_f = cube.georef.pix_to_lonlat(np.array([15]), np.array([26]))
    assert float(lat_s[0]) == pytest.approx(float(lat_f[0]), abs=1e-9)
    assert float(lon_s[0]) == pytest.approx(float(lon_f[0]), abs=1e-9)


# ---------------------------------------------------------------------------
# the real PCP table format
# ---------------------------------------------------------------------------
def _pcp_tab_bytes(pole: str, half_px: int, pixels, *, origin_frac: float = 0.0,
                   flip_y: bool = False) -> bytes:
    """A table in the real PCP layout: one header record, then five
    whitespace-separated ASCII columns x y lon lat T, empty bins omitted."""
    g = PCP.pcp_georef(pole, half_px)
    lines = ["PCP synthetic header record"]
    for (i, j, t) in pixels:
        x = (j - half_px + origin_frac) * PCP.PCP_SCALE_M / PCP.MOON_RADIUS_M
        yy = (i - half_px) if flip_y else (half_px - i)
        y = (yy + origin_frac) * PCP.PCP_SCALE_M / PCP.MOON_RADIUS_M
        lon, lat = g.pix_to_lonlat(np.array([i]), np.array([j]))
        lines.append(f"{x:+.8f} {y:+.8f} {float(lon[0]):10.5f} {float(lat[0]):9.5f} {t:8.3f}")
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


@pytest.mark.parametrize("origin_frac", [0.0, 0.5])
@pytest.mark.parametrize("flip_y", [False, True])
def test_a_real_format_table_lands_on_the_grid_at_either_registration(tmp_path, origin_frac, flip_y):
    half = 12
    px = [(10, 11, 41.0), (12, 13, 43.5), (14, 9, 39.25)]
    p = tmp_path / "pcp.tab"
    p.write_bytes(_pcp_tab_bytes("north", half, px, origin_frac=origin_frac, flip_y=flip_y))
    tab = PCP.read_pcp_tab(p)
    assert tab["n"] == len(px)
    r = PCP.rasterise(tab, "north", half)
    assert r["status"] == "OK"
    assert r["n_on_grid"] == len(px)
    assert r["collision_frac"] == 0.0
    assert r["flip_y"] is flip_y
    assert r["origin_frac"] == origin_frac
    assert r["georef_resid_deg"] < 1e-3
    for (i, j, t) in px:
        assert float(r["array"][i, j]) == pytest.approx(t, abs=1e-2)


def test_the_wrong_registration_is_visible_in_the_residual(tmp_path):
    half = 12
    px = [(10, 11, 41.0), (12, 13, 43.5)]
    p = tmp_path / "pcp.tab"
    p.write_bytes(_pcp_tab_bytes("north", half, px, flip_y=False))
    tab = PCP.read_pcp_tab(p)
    right = PCP.rasterise(tab, "north", half, flip_y=False)
    wrong = PCP.rasterise(tab, "north", half, flip_y=True)
    assert wrong["georef_resid_deg"] > right["georef_resid_deg"]


def test_an_empty_table_is_empty_not_an_exception(tmp_path):
    p = tmp_path / "pcp.tab"
    p.write_bytes(b"header only\r\n")
    assert PCP.read_pcp_tab(p).get("n", 0) == 0
    assert PCP.rasterise({"n": 0}, "north", 4)["status"] == "EMPTY"


def test_the_polar_grid_covers_the_products_latitude_cap():
    half = PCP.half_px_for(80.0)
    g = PCP.pcp_georef("north", half)
    _lon, lat = g.pix_to_lonlat(np.array([half]), np.array([0]))
    assert float(lat[0]) < 80.5
    assert g.lines == g.samples == 2 * half + 1
    assert g.map_scale_m == PCP.PCP_SCALE_M


def test_local_time_hours_spans_the_day():
    assert PCP.local_time_hours(1, 96) == pytest.approx(0.125)
    assert PCP.local_time_hours(96, 96) == pytest.approx(23.875)
    assert PCP.local_time_hours(1, 24) == pytest.approx(0.5)


@pytest.mark.parametrize(("spec", "n_bins", "exp"), [
    ("every:16", 96, [1, 17, 33, 49, 65, 81]),
    ("n:4", 24, [1, 7, 13, 19]),
    ([2, 5], 96, [2, 5]),
    ("3,9", 96, [3, 9]),
])
def test_bin_spec_parsing(spec, n_bins, exp):
    assert PCP.parse_bin_spec(spec, n_bins) == exp


# ---------------------------------------------------------------------------
# resolving the products: ODE first, the verified template as fallback
# ---------------------------------------------------------------------------
def _ode_pcp_files(bins_by_season: dict) -> list[dict]:
    out = []
    for pole_tag in ("poln", "pols"):
        for season_tag, bins in bins_by_season.items():
            for b in bins:
                stem = f"pcp_avg_tbol_{pole_tag}_{season_tag}_ltim{b:02d}_240"
                for ext in ("tab", "lbl"):
                    out.append({"name": f"{stem}.{ext}".upper(),
                                "url": f"https://example.invalid/pcp/{pole_tag}/{stem}.{ext}"})
    return out


def test_ode_listing_becomes_a_bin_index():
    idx = PCP.index_pcp_files(_ode_pcp_files({"sum": [1, 5, 9], "win": [1, 5, 9]}))
    assert idx[("north", "summer", 5, "tab")].endswith("poln_sum_ltim05_240.tab")
    assert PCP.available_bins(idx, "south") == [1, 5, 9]
    assert len(idx) == 2 * 2 * 3 * 2


def test_a_bin_missing_in_one_season_is_not_offered():
    idx = PCP.index_pcp_files(_ode_pcp_files({"sum": [1, 5, 9], "win": [1, 9]}))
    assert PCP.available_bins(idx, "north") == [1, 9]
    assert PCP.available_bins(idx, "north", ["summer"]) == [1, 5, 9]


def test_names_that_are_not_local_time_products_are_ignored():
    idx = PCP.index_pcp_files({
        "PCP_AVG_TBOL_POLS_LS030_SLON07_240.TAB": "https://example.invalid/a.tab",
        "PCP_AVG_TBOL_POLN_SUM_LTIM01_240_XML.ZIP": "https://example.invalid/b.zip",
        "PCP_AVG_TBOL_POLN_SUM_LTIM01_240.TAB": "https://example.invalid/c.tab",
    })
    assert list(idx) == [("north", "summer", 1, "tab")]


def test_resolution_falls_back_to_the_verified_template_when_ode_is_down(tmp_path):
    fetch = A.ScriptedFetcher(routes={})
    rep = resolve_pcp_products(CONF, fetch, tmp_path)
    assert rep["route"] == "url_template"
    assert rep["status"] == "ODE_UNAVAILABLE_USING_TEMPLATE"
    assert json.loads((tmp_path / "pcp_index.json").read_text())["route"] == "url_template"
    # the template is the URL the runner verified answers 200
    u = PCP.pcp_url("north", "summer", 1, "tab")
    assert u.endswith("data_derived_pcp/diurnal/ltim/poln/pcp_avg_tbol_poln_sum_ltim01_240.tab")


# ---------------------------------------------------------------------------
# the LOLA PSR raster: registered off the label, not the array centre
# ---------------------------------------------------------------------------
def _lpsr_georef(n: int, offset: float) -> Georef:
    return Georef(projection="polar_stereographic", center_lat=90.0, center_lon=0.0,
                  map_scale_m=240.0, line_offset=offset, sample_offset=offset,
                  lines=n, samples=n, radius_m=PCP.MOON_RADIUS_M)


def test_the_psr_mask_is_registered_off_the_label_offsets():
    """LPSR_65N_240M is 6420 x 6420 with the pole at 0-based pixel 3208.5,
    one pixel from the array centre; cropping about the centre moves the
    mapped shadow by 240 m."""
    n, half = 64, 5
    src = np.full((n, n), -20000, dtype=np.int16)
    pole_i = pole_j = 30            # deliberately NOT (n-1)/2
    src[pole_i, pole_j] = 20000
    g = _lpsr_georef(n, float(pole_i))
    m = PCP.crop_lpsr(src, half, pole="north", src_georef=g)
    assert m.shape == (2 * half + 1, 2 * half + 1)
    assert m[half, half]                       # the pole pixel lands on the pole
    assert int(m.sum()) == 1
    centred = PCP.crop_lpsr(src, half, pole="north")
    assert not centred[half, half]             # the array-centre crop misses it


def test_the_psr_crop_is_all_false_outside_the_source_raster():
    src = np.full((8, 8), 20000, dtype=np.int16)
    m = PCP.crop_lpsr(src, 12, pole="south", src_georef=_lpsr_georef(8, 3.5))
    assert m.shape == (25, 25)
    assert int(m.sum()) == 64                  # only the 8x8 that exists is true


def test_both_published_lpsr_routes_are_offered():
    assert PCP.n_lpsr_routes() == 2
    assert PCP.lpsr_url("north", "img", 0).endswith("release_2014/img/lpsr_65n_240m.img")
    assert PCP.lpsr_url("south", "img", 1).endswith("illumination/img/lpsr_65s_240m_201608.img")


# ---------------------------------------------------------------------------
# end to end through a scripted archive
# ---------------------------------------------------------------------------
#: The grid is set by the product's latitude cap, not by the test: at the real
#: cap of 80 deg it is 2535 x 2535, which is the right size to screen and the
#: wrong size to build a scripted archive for.  So the cap is raised until the
#: half-width is 40 px, and HALF is read back from the same function the stage
#: uses, so the archive and the screen can never disagree about the grid.
MIN_LAT = 89.6834
HALF = PCP.half_px_for(MIN_LAT)
#: comfortably above the floor this scripted archive measures for itself
A_INJ = 300.0
BINS = [1, 5, 9, 13, 17, 21]
SEASONS = ["summer", "winter"]


def _scripted_conf(tmp_path: Path, **diurnal) -> dict:
    conf = json.loads(json.dumps(CONF))
    conf["products"]["poles"] = ["north"]
    conf["diurnal"].update({"ltim_bins": ",".join(str(b) for b in BINS),
                            "seasons": SEASONS, "n_bins_min": 6, "n_bins_excess": 4,
                            "min_lat_deg": MIN_LAT, **diurnal})
    return conf


def _scripted_archive(*, inject_px=None, serve: bool = True) -> A.ScriptedFetcher:
    """Serves the LPSR raster and every PCP table in the real formats."""
    n = 2 * HALF + 1
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(y - HALF, x - HALF) / float(HALF)
    psr = r < 0.8
    rng = np.random.default_rng(17)
    routes: dict = {}
    if not serve:
        return A.ScriptedFetcher(routes=routes)

    img = np.where(psr, 20000, -20000).astype("<i2")
    lbl = ("PDS_VERSION_ID = \"PDS3\"\r\nRECORD_TYPE = FIXED_LENGTH\r\n"
           f"FILE_RECORDS = {n}\r\nRECORD_BYTES = {2 * n}\r\n"
           "^IMAGE = \"LPSR.IMG\"\r\nOBJECT = IMAGE\r\n"
           f"  LINES = {n}\r\n  LINE_SAMPLES = {n}\r\n"
           "  SAMPLE_TYPE = LSB_INTEGER\r\n  SAMPLE_BITS = 16\r\n"
           "  SCALING_FACTOR = 0.000025\r\n  OFFSET = 0.5\r\nEND_OBJECT = IMAGE\r\n"
           "OBJECT = IMAGE_MAP_PROJECTION\r\n"
           "  MAP_PROJECTION_TYPE = \"POLAR STEREOGRAPHIC\"\r\n"
           "  CENTER_LATITUDE = 90 <deg>\r\n  CENTER_LONGITUDE = 0 <deg>\r\n"
           "  MAP_SCALE = 240 <m/pix>\r\n  A_AXIS_RADIUS = 1737.4 <km>\r\n"
           f"  LINE_PROJECTION_OFFSET = {HALF + 1}.0 <pix>\r\n"
           f"  SAMPLE_PROJECTION_OFFSET = {HALF + 1}.0 <pix>\r\n"
           "END_OBJECT = IMAGE_MAP_PROJECTION\r\nEND\r\n")
    for route in range(PCP.n_lpsr_routes()):
        routes[PCP.lpsr_url("north", "lbl", route)] = (200, lbl)
        routes[PCP.lpsr_url("north", "img", route)] = (200, img.tobytes())

    ii, jj = np.nonzero(psr)
    for season, warm in (("summer", 1.0), ("winter", 0.0)):
        for b in BINS:
            phase = 2.0 * np.pi * (b - 1) / 24.0
            t = (38.0 + 22.0 * np.clip(r, 0, 1) ** 2
                 + warm * (0.2 + 2.2 * np.clip(r - 0.35, 0, 1) ** 2)
                 + (0.15 + 3.0 * np.clip(r - 0.35, 0, 1) ** 2)
                 * (1.0 + np.cos(phase)) * (0.35 + 0.65 * warm)
                 + rng.normal(0.0, 0.12, size=(n, n)))
            for (pi, pj, f) in (inject_px or []):
                t[pi, pj] = float(D.mix_bolometric(t[pi, pj], f, 300.0))
            px = [(int(a), int(c), float(t[a, c])) for a, c in zip(ii, jj, strict=True)]
            routes[PCP.pcp_url("north", season, b, "tab")] = (
                200, _pcp_tab_bytes("north", HALF, px))
    return A.ScriptedFetcher(routes=routes)


def test_end_to_end_through_a_scripted_archive_recovers_an_injected_source(tmp_path):
    conf = _scripted_conf(tmp_path)
    fetch = _scripted_archive(inject_px=[(HALF, HALF + 6, A_INJ / PIX)])
    reps = stage_pcp(conf, tmp_path, fetch=fetch, poles=["north"],
                     run_sensitivity=False)
    rep = reps["north"]
    assert rep["status"] == "OK", rep
    assert json.loads((tmp_path / "pcp_north.json").read_text())["half_px"] == HALF
    assert rep["acquisition"]["n_layers"] == len(BINS) * len(SEASONS)
    assert rep["acquisition"]["index_route"] == "url_template"
    assert rep["mask"]["external_psr"] is True
    pcp_rep = json.loads((tmp_path / "pcp_north.json").read_text())
    assert pcp_rep["psr"]["status"] == "OK"
    assert pcp_rep["psr"]["registration"] == "label_offsets"
    assert all(p["outcome"] == "OK" for p in pcp_rep["products"].values())
    assert all(p["collision_frac"] == 0.0 for p in pcp_rep["products"].values())
    cand = [c for c in rep["candidates"] if (c["line"], c["sample"]) == (HALF, HALF + 6)]
    assert cand, f"injected source not a candidate; counts={rep['counts']}"
    assert 0.3 < cand[0]["area_m2_300K"] / A_INJ < 3.0

    summary = crypt_run(conf, stage="assess_diurnal", out_dir=tmp_path)
    assert summary["verdict"].startswith("DIURNALLY_INVARIANT_CANDIDATES")
    assert summary["n_psr_interior_px"] > 0
    assert summary["poles_screened"] == ["north"]
    assert Path(tmp_path / "candidates.csv").exists()


def test_end_to_end_clean_archive_is_a_count_not_a_candidate(tmp_path):
    conf = _scripted_conf(tmp_path)
    stage_pcp(conf, tmp_path, fetch=_scripted_archive(), poles=["north"], run_sensitivity=False)
    summary = crypt_run(conf, stage="assess_diurnal", out_dir=tmp_path)
    assert summary["verdict"].startswith("NO_DIURNALLY_INVARIANT_SURVIVOR")
    assert summary["n_candidates"] == 0
    assert summary["n_psr_interior_px"] > 0
    assert "not an occurrence limit" in summary["note"]


def test_an_archive_that_serves_nothing_is_no_data_reached_never_a_null(tmp_path):
    conf = _scripted_conf(tmp_path)
    reps = stage_pcp(conf, tmp_path, fetch=_scripted_archive(serve=False), poles=["north"],
                     run_sensitivity=False)
    assert reps["north"]["status"] == "NO_DATA_REACHED"
    summary = crypt_run(conf, stage="assess_diurnal", out_dir=tmp_path)
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["n_psr_interior_px"] == 0
    assert summary["poles_screened"] == []
    assert "NO_DATA_REACHED is an access statement" in summary["note"]


def test_a_missing_psr_raster_is_recorded_as_degraded_not_ignored(tmp_path):
    conf = _scripted_conf(tmp_path)
    fetch = _scripted_archive()
    for route in range(PCP.n_lpsr_routes()):
        fetch.routes.pop(PCP.lpsr_url("north", "img", route))
        fetch.routes.pop(PCP.lpsr_url("north", "lbl", route))
    reps = stage_pcp(conf, tmp_path, fetch=fetch, poles=["north"], run_sensitivity=False)
    rep = reps["north"]
    assert "psr:NO_PSR_RASTER" in rep["degraded"]
    assert "no_external_psr_raster" in rep["degraded"]
    summary = crypt_run(conf, stage="assess_diurnal", out_dir=tmp_path)
    assert any("NO_PSR_RASTER" in d for d in summary["degraded"])
    assert "DEGRADED" in summary["verdict"]


def test_the_tables_are_deleted_after_they_are_rasterised(tmp_path):
    """Peak disk must be one 262 MB product, not the whole set."""
    conf = _scripted_conf(tmp_path)
    stage_pcp(conf, tmp_path, fetch=_scripted_archive(), poles=["north"], run_sensitivity=False)
    assert list((tmp_path / "data" / "north").glob("*.tab")) == []

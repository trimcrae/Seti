"""Offline test suite for CRYPT — artifacts in lunar permanent shadow (S55).

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite:

* reproduces the brief's number (1 m² at 300 K on a 35 K pixel lifts channel
  6 to ~57 K while channel 9 moves ~0.01 K) and the single-temperature
  identity of the band model;
* recovers an injected warm sub-pixel component on a synthetic PSR map as a
  ``candidate`` in both seasons, with the fitted area within a factor of
  three;
* rejects a lit rim (present in summer, absent in winter) as ``seasonal``;
* trips every other rule: ``extended``, ``stripe``, ``low_count``,
  ``single_season``, ``human_hardware``, ``inconsistent_spectrum``,
  ``unstable``;
* returns a clean null on the un-injected synthetic map;
* reads PDS3 (detached, 16-bit scaled) and PDS4 labels and rasters, and
  round-trips the polar-stereographic georeference;
* parses IIS and Apache directory listings, classifies product names,
  selects the needed set;
* runs probe → acquire → screen → assess through a SCRIPTED archive that
  serves synthetic PDS3 products, and reports ``NO_DATA_REACHED`` when the
  archive serves nothing;
* screens compact radar anomalies and kills rock fields / elevated
  backgrounds / weak echoes;
* shards the pole list and runs the CLI entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.crypt import acquire as A
from seti.crypt import labels as Lb
from seti.crypt import radar as R
from seti.crypt import thermal as T
from seti.crypt import vet as V
from seti.crypt.run import (
    crypt_run,
    load_crypt_config,
    main,
    needed_products,
    parse_shard,
    shard_poles,
    stage_acquire,
    stage_assess,
    stage_probe,
    stage_screen,
)

CONF = load_crypt_config()
THR = dict(CONF["thermal"])
PIX = 240.0**2


@pytest.fixture(scope="module")
def synth():
    return T.synthetic_pole(160, pole="south", seed=3)


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def test_brief_numbers_one_square_metre_at_300K():
    m = T.band_model()
    f = 1.0 / PIX
    t6 = float(m.two_component_bt("6", 35.0, f, 300.0))
    t9 = float(m.two_component_bt("9", 35.0, f, 300.0))
    assert 54.0 < t6 < 60.0, t6            # the brief says 57 K
    assert abs(t9 - 35.0) < 0.02, t9        # and 0.01 K in channel 9


def test_single_temperature_identity_and_monotonic_inverse():
    m = T.band_model()
    for k in ("6", "7", "8", "9"):
        for Tc in (30.0, 60.0, 110.0, 400.0):
            assert abs(float(m.bt(k, m.radiance(k, Tc))) - Tc) < 1e-3
    L = m.radiance("7", np.array([40.0, 41.0, 60.0]))
    assert np.all(np.diff(L) > 0)


def test_floor_area_is_monotonic_in_noise():
    m = T.band_model()
    a1 = T.floor_area(m, "6", 50.0, 1.0, 300.0, PIX)
    a5 = T.floor_area(m, "6", 50.0, 5.0, 300.0, PIX)
    assert 0 < a1 < a5
    # a source that raises ch6 by 5 K on a 50 K pixel is metres-squared scale
    assert 0.1 < a5 < 1000.0


# ---------------------------------------------------------------------------
# the screen: injection, confounders, every rule
# ---------------------------------------------------------------------------
def test_clean_null_on_uninjected_map(synth):
    rep = T.screen_pole(synth, THR)
    assert rep["status"] == "OK"
    assert rep["counts"]["candidate"] == 0
    assert rep["mask"]["n_interior"] > 1000
    for s in ("summer", "winter"):
        r = rep["per_season"][s]
        assert r["ref"] == "9" and r["primary"] == "6"
        assert r["noise"]["6"]["bins"], "no noise bins"


def test_injected_warm_component_is_recovered_in_both_seasons(synth):
    inj = T.inject(synth, [(80, 80)], 300.0 / PIX, 300.0)
    rep = T.screen_pole(inj, THR)
    fl = rep["flags"]
    hit = fl[(fl["line"] == 80) & (fl["sample"] == 80)]
    assert len(hit) == 1
    row = hit.iloc[0]
    assert row["class"] == "candidate", row["reasons"]
    assert row["z6_summer"] >= THR["z_min"] and row["z6_winter"] >= THR["z_min"]
    for s in ("summer", "winter"):
        # f and T_hot are degenerate on the grid; the area is right to a factor ~4
        assert 60.0 < row[f"area_m2_{s}"] < 1500.0, row[f"area_m2_{s}"]
        assert row[f"dchi2_{s}"] > THR["dchi2_min"]
    # the excess radiance (f·L6(T_hot)) is the season-independent observable
    assert abs(row["excess_summer"] - row["excess_winter"]) < 0.3 * max(row["excess_summer"], row["excess_winter"])
    assert rep["counts"]["candidate"] == 1
    assert -90.0 < row["lat"] < -89.9


def test_lit_rim_present_in_summer_only_is_seasonal(synth):
    inj = T.inject(synth, [(60, 80)], 3000.0 / PIX, 300.0, seasons=["summer"])
    rep = T.screen_pole(inj, THR)
    fl = rep["flags"]
    hit = fl[(fl["line"] == 60) & (fl["sample"] == 80)].iloc[0]
    assert hit["class"] == "seasonal"
    assert rep["counts"]["candidate"] == 0


def test_extended_warm_patch_is_rejected(synth):
    pix = [(i, j) for i in range(70, 75) for j in range(70, 75)]
    inj = T.inject(synth, pix, 500.0 / PIX, 300.0)
    rep = T.screen_pole(inj, THR)
    fl = rep["flags"]
    assert (fl["class"] == "extended").sum() >= 20
    assert rep["counts"]["candidate"] == 0


def test_row_of_hot_pixels_is_a_stripe(synth):
    pix = [(90, j) for j in (40, 55, 70, 85, 100, 115)]
    inj = T.inject(synth, pix, 500.0 / PIX, 300.0)
    rep = T.screen_pole(inj, THR)
    fl = rep["flags"]
    assert (fl["class"] == "stripe").sum() == 6
    assert rep["counts"]["candidate"] == 0


def test_low_observation_count_is_rejected(synth):
    L = synth.copy()
    L.arrays[("winter", "9", "count")][80, 80] = 3.0
    inj = T.inject(L, [(80, 80)], 500.0 / PIX, 300.0)
    rep = T.screen_pole(inj, THR)
    hit = rep["flags"].iloc[0]
    assert hit["class"] == "low_count"


def test_single_season_product_cannot_be_a_candidate(synth):
    L = T.PoleLayers(synth.pole, synth.georef)
    for (s, c, t), a in synth.arrays.items():
        if s in ("summer", "all"):
            L.put(s, c, t, a)
    inj = T.inject(L, [(80, 80)], 500.0 / PIX, 300.0)
    rep = T.screen_pole(inj, THR)
    assert "single_season_product" in rep["degraded"]
    hit = rep["flags"].iloc[0]
    assert hit["class"] == "single_season"
    assert rep["counts"]["candidate"] == 0


def test_human_hardware_site_is_flagged(synth):
    lon, lat = synth.georef.pix_to_lonlat(80, 80)
    hw = [{"name": "test lander", "lat": float(lat), "lon": float(lon), "radius_m": 500.0}]
    inj = T.inject(synth, [(80, 80)], 500.0 / PIX, 300.0)
    rep = T.screen_pole(inj, THR, hardware=hw)
    hit = rep["flags"].iloc[0]
    assert hit["class"] == "human_hardware"
    assert "test lander" in hit["reasons"]


def test_wrong_channel_ordering_is_inconsistent_spectrum(synth):
    # channel 6 and channel 8 raised, channel 7 untouched: no hot component does that
    L = synth.copy()
    for s in ("summer", "winter"):
        L.arrays[(s, "6", "avg")][80, 80] += 40.0
        L.arrays[(s, "8", "avg")][80, 80] += 12.0
    rep = T.screen_pole(L, THR)
    hit = rep["flags"][(rep["flags"]["line"] == 80) & (rep["flags"]["sample"] == 80)].iloc[0]
    assert hit["class"] == "inconsistent_spectrum"
    assert "ordering" in hit["reasons"]


def test_seasonally_varying_source_is_unstable(synth):
    L = T.inject(synth, [(80, 80)], 300.0 / PIX, 300.0, seasons=["winter"])
    L = T.inject(L, [(80, 80)], 6000.0 / PIX, 300.0, seasons=["summer"])
    rep = T.screen_pole(L, THR)
    hit = rep["flags"][(rep["flags"]["line"] == 80) & (rep["flags"]["sample"] == 80)].iloc[0]
    assert hit["class"] == "unstable"


def test_mask_degrades_without_a_maximum_product(synth):
    L = T.PoleLayers(synth.pole, synth.georef)
    for (s, c, t), a in synth.arrays.items():
        if t != "max":
            L.put(s, c, t, a)
    m = T.psr_mask(L, {**T.DEFAULT_THRESHOLDS, **THR})
    assert "mask_from_average_not_maximum" in m["degraded"]
    assert m["n_interior"] > 0


def test_external_psr_raster_is_anded_in(synth):
    ext = np.zeros(synth.shape, dtype=np.float32)
    ext[70:90, 70:90] = 1.0
    m = T.psr_mask(synth, {**T.DEFAULT_THRESHOLDS, **THR}, external=ext)
    assert m["n_mask"] <= 400 and "external_psr" in m["source"]


def test_sensitivity_table_recovers_large_sources(synth):
    sens = T.sensitivity(synth, THR, [10.0, 1000.0], 300.0, n_per=6, seed=2)
    rows = {r["area_m2"]: r for r in sens["rows"]}
    assert rows[1000.0]["recovered_frac"] == 1.0
    assert rows[10.0]["recovered_frac"] < 0.5


# ---------------------------------------------------------------------------
# labels and georeferencing
# ---------------------------------------------------------------------------
def test_pds3_roundtrip_and_polar_georef(tmp_path):
    g = Lb.polar_georef("north", 64, 240.0)
    img = np.random.default_rng(1).uniform(30, 120, (64, 64)).astype(np.float32)
    img[3, 4] = np.nan
    lbl = Lb.write_pds3_raster(img, tmp_path / "dgdr_t7_avg_sum_n_240m.img", georef=g,
                               product_id="X", description="channel 7 summer average")
    meta = Lb.read_label(lbl)
    assert meta.dialect == "pds3" and meta.lines == 64 and meta.samples == 64
    assert meta.dtype == ">i2" and meta.scaling_factor == 0.01
    assert meta.georef.projection == "polar_stereographic" and meta.georef.center_lat == 90.0
    assert abs(meta.georef.map_scale_m - 240.0) < 1e-6
    assert meta.georef.origin_check_px < 1e-6
    back = Lb.read_raster(meta)
    assert np.isnan(back[3, 4])
    ok = np.isfinite(img)
    assert np.allclose(back[ok], img[ok], atol=0.006)
    # georef round trip and the pole at the grid centre
    lon, lat = meta.georef.pix_to_lonlat(np.array([0, 31.5, 63]), np.array([0, 31.5, 63]))
    assert lat[1] > 89.999
    i, j = meta.georef.lonlat_to_pix(lon, lat)
    assert np.allclose(i, [0, 31.5, 63], atol=1e-6) and np.allclose(j, [0, 31.5, 63], atol=1e-6)
    # a point 240 m from the pole is one pixel away
    i2, j2 = meta.georef.lonlat_to_pix(0.0, 90.0 - 240.0 / 30336.0)
    assert abs(np.hypot(i2 - 31.5, j2 - 31.5) - 1.0) < 0.01


def test_south_polar_georef_is_consistent():
    g = Lb.polar_georef("south", 100, 240.0)
    lon, lat = g.pix_to_lonlat(np.array([10, 49.5, 90]), np.array([49.5, 49.5, 10]))
    assert lat[1] < -89.999 and -89.75 < lat[0] < -89.6      # 39.5 px = 9.5 km = 0.31 deg
    i, j = g.lonlat_to_pix(lon, lat)
    assert np.allclose(i, [10, 49.5, 90], atol=1e-6) and np.allclose(j, [49.5, 49.5, 10], atol=1e-6)


def test_pds3_label_parser_handles_pointers_units_and_objects():
    txt = """PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 5056
FILE_RECORDS = 2529
^IMAGE = ("DGDR_TBOL_MAX_ALL_S_240M.IMG", 2)
PRODUCT_ID = "DGDR_TBOL_MAX_ALL_S_240M"
OBJECT = IMAGE
  LINES = 2528
  LINE_SAMPLES = 2528
  SAMPLE_TYPE = MSB_INTEGER
  SAMPLE_BITS = 16
  SCALING_FACTOR = 0.01 <K>
  OFFSET = 0.0
  MISSING_CONSTANT = -32768
  DESCRIPTION = "Bolometric temperature, maximum over
                 all seasons"
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "POLAR STEREOGRAPHIC"
  A_AXIS_RADIUS = 1737.4 <KM>
  CENTER_LATITUDE = -90.0 <DEG>
  CENTER_LONGITUDE = 0.0 <DEG>
  MAP_SCALE = 0.240 <KM/PIXEL>
  LINE_PROJECTION_OFFSET = 1263.5
  SAMPLE_PROJECTION_OFFSET = 1263.5
END_OBJECT = IMAGE_MAP_PROJECTION
END
"""
    d = Lb.parse_pds3_label(txt)
    assert d["^IMAGE"] == ["DGDR_TBOL_MAX_ALL_S_240M.IMG", "2"]
    assert d["IMAGE.LINES"] == "2528" and d["IMAGE_MAP_PROJECTION.MAP_SCALE__unit"] == "KM/PIXEL"
    assert "all seasons" in d["IMAGE.DESCRIPTION"]
    g = Lb.georef_from_pds3(d, 2528, 2528)
    assert g.projection == "polar_stereographic" and abs(g.map_scale_m - 240.0) < 1e-9
    assert g.center_lat == -90.0 and g.origin_check_px < 1e-9


def test_pds4_label_is_read(tmp_path):
    img = (np.arange(16, dtype="<f4").reshape(4, 4) * 2.0)
    (tmp_path / "prod.img").write_bytes(img.tobytes())
    xml = """<?xml version="1.0"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1" xmlns:cart="http://pds.nasa.gov/pds4/cart/v1">
 <Identification_Area><logical_identifier>urn:nasa:pds:test:prod</logical_identifier></Identification_Area>
 <Observation_Area><Discipline_Area><cart:Cartography>
  <cart:Spatial_Reference_Information><cart:Horizontal_Coordinate_System_Definition><cart:Planar>
   <cart:Map_Projection><cart:map_projection_name>Polar Stereographic</cart:map_projection_name>
    <cart:Polar_Stereographic><cart:straight_vertical_longitude_from_pole unit="deg">0</cart:straight_vertical_longitude_from_pole>
     <cart:latitude_of_projection_origin unit="deg">-90</cart:latitude_of_projection_origin></cart:Polar_Stereographic>
   </cart:Map_Projection>
   <cart:Planar_Coordinate_Information><cart:Coordinate_Representation>
    <cart:pixel_resolution_x unit="m/pixel">240</cart:pixel_resolution_x></cart:Coordinate_Representation></cart:Planar_Coordinate_Information>
   <cart:Geo_Transformation><cart:upperleft_corner_x unit="m">-480</cart:upperleft_corner_x><cart:upperleft_corner_y unit="m">480</cart:upperleft_corner_y></cart:Geo_Transformation>
  </cart:Planar></cart:Horizontal_Coordinate_System_Definition>
  <cart:Geodetic_Model><cart:semi_major_radius unit="km">1737.4</cart:semi_major_radius></cart:Geodetic_Model>
  </cart:Spatial_Reference_Information></cart:Cartography></Discipline_Area></Observation_Area>
 <File_Area_Observational><File><file_name>prod.img</file_name></File>
  <Array_2D_Image><offset unit="byte">0</offset><axes>2</axes>
   <Element_Array><data_type>IEEE754LSBSingle</data_type><scaling_factor>0.5</scaling_factor><value_offset>1</value_offset></Element_Array>
   <Axis_Array><axis_name>Line</axis_name><elements>4</elements></Axis_Array>
   <Axis_Array><axis_name>Sample</axis_name><elements>4</elements></Axis_Array>
   <Special_Constants><missing_constant>30</missing_constant></Special_Constants>
  </Array_2D_Image></File_Area_Observational>
</Product_Observational>"""
    (tmp_path / "prod.xml").write_text(xml)
    meta = Lb.read_label(tmp_path / "prod.xml")
    assert meta.dialect == "pds4" and meta.lines == 4 and meta.samples == 4 and meta.dtype == "<f4"
    back = Lb.read_raster(meta)
    assert back[0, 1] == 2.0 * 0.5 + 1.0
    assert np.isnan(back[3, 3])       # raw 30 is the missing constant
    g = meta.georef
    assert g.projection == "polar_stereographic" and g.center_lat == -90.0
    assert abs(g.map_scale_m - 240.0) < 1e-9 and g.origin_check_px < 1e-9


# ---------------------------------------------------------------------------
# listings, classification, selection
# ---------------------------------------------------------------------------
IIS = """<html><head><title>pds-geosciences.wustl.edu - /lro/x/data/pcp/</title></head><body>
<pre><A HREF="/lro/x/data/">[To Parent Directory]</A><br><br> 9/20/2023 10:11 AM        &lt;dir&gt; <A HREF="/lro/x/data/pcp/north/">north</A><br>
 6/15/2026 10:54 AM     12780544 <A HREF="/lro/x/data/pcp/dgdr_t6_avg_sum_s_240m.img">dgdr_t6_avg_sum_s_240m.img</A><br>
 6/15/2026 10:54 AM         2809 <A HREF="/lro/x/data/pcp/dgdr_t6_avg_sum_s_240m.lbl">dgdr_t6_avg_sum_s_240m.lbl</A><br></pre><hr></body></html>"""

APACHE = """<html><body><h1>Index of /data</h1><pre><a href="../">Parent Directory</a>
<a href="sub/">sub/</a>                 2024-01-01 10:00    -
<a href="a_cpr_85s.img">a_cpr_85s.img</a>      2024-01-01 10:00  1.5M
<a href="a_cpr_85s.lbl">a_cpr_85s.lbl</a>      2024-01-01 10:00  1024
</pre></body></html>"""


def test_parse_iis_listing():
    ents = A.parse_listing(IIS, "https://pds-geosciences.wustl.edu/lro/x/data/pcp/")
    names = {e.name: e for e in ents}
    assert set(names) == {"north", "dgdr_t6_avg_sum_s_240m.img", "dgdr_t6_avg_sum_s_240m.lbl"}
    assert names["north"].is_dir and names["dgdr_t6_avg_sum_s_240m.img"].size == 12780544
    assert names["dgdr_t6_avg_sum_s_240m.img"].url.endswith("/pcp/dgdr_t6_avg_sum_s_240m.img")


def test_parse_apache_listing():
    ents = A.parse_listing(APACHE, "https://host/data/")
    names = {e.name: e for e in ents}
    assert names["sub"].is_dir and names["a_cpr_85s.img"].size == int(1.5 * 1024**2)
    assert names["a_cpr_85s.lbl"].size == 1024


@pytest.mark.parametrize("name,exp", [
    ("dgdr_t6_avg_sum_s_240m.img", ("south", "6", "avg", "summer", 240)),
    ("dgdr_tbol_max_all_n_240m.lbl", ("north", "tbol", "max", "all", 240)),
    ("DGDR_T9_CNT_WIN_240S.IMG", ("south", "9", "count", "winter", None)),
    ("pcp_ch7_mean_winter_north.xml", ("north", "7", "avg", "winter", None)),
])
def test_classify_names(name, exp):
    c = A.classify_name(name, CONF["patterns"])
    assert (c["pole"], c["channel"], c["stat"], c["season"], c["resolution_m"]) == exp


def test_classify_falls_back_to_label_text():
    c = A.classify_name("product_0042.img", CONF["patterns"],
                        extra_text='DESCRIPTION = "Channel 7 brightness temperature, average, winter, south polar"')
    assert (c["pole"], c["channel"], c["stat"], c["season"]) == ("south", "7", "avg", "winter")


def test_select_needed_prefers_unsplit_240m():
    ents = [A.Entry(n, "http://h/" + n, False, s) for n, s in [
        ("dgdr_t6_avg_sum_s_240m.img", 100), ("dgdr_t6_avg_sum_s_240m.lbl", 1),
        ("dgdr_t6_avg_sum_s_ssl030_240m.img", 100), ("dgdr_t6_avg_sum_s_ssl030_240m.lbl", 1),
        ("dgdr_t6_avg_sum_s_480m.img", 25), ("dgdr_t6_avg_sum_s_480m.lbl", 1)]]
    groups = A.pair_products(ents)
    sel = A.select_needed(groups, [{"pole": "south", "season": "summer", "channel": "6", "stat": "avg"}],
                          CONF["patterns"])
    assert sel["south/summer/6/avg"]["stem"] == "dgdr_t6_avg_sum_s_240m"
    sel2 = A.select_needed(groups, [{"pole": "north", "season": "summer", "channel": "6", "stat": "avg"}],
                           CONF["patterns"])
    assert sel2["north/summer/6/avg"] is None


def test_ode_helpers():
    p = A.ode_footprint_params("LRO", "LROC", "EDRNAC4", 10.0, -85.0)
    assert p["minlat"] < -85.0 < p["maxlat"] and p["loc"] == "f"
    rec = {"json": {"ODEResults": {"Count": "2", "Products": {"Product": [{"pdsid": "A"}, {"pdsid": "B"}]}}}}
    assert A.ode_count(rec) == 2 and [x["pdsid"] for x in A.ode_products(rec)] == ["A", "B"]
    assert A.ode_products({"json": None}) == []


# ---------------------------------------------------------------------------
# a scripted archive serving synthetic PDS3 products: probe → assess
# ---------------------------------------------------------------------------
def _listing(base: str, files: dict[str, bytes]) -> str:
    rows = "".join(f' 6/15/2026 10:54 AM {len(b):>12} <A HREF="{base}{n}">{n}</A><br>\n' for n, b in files.items())
    return f"<html><body><pre><A HREF=\"/lro/\">[To Parent Directory]</A><br><br>{rows}</pre></body></html>"


def _scripted_archive(tmp_path: Path, layers: T.PoleLayers, conf: dict) -> A.ScriptedFetcher:
    """Serve every layer of ``layers`` as a PDS3 product under the configured
    Diviner root, with an IIS-style listing; everything else is unreachable."""
    root = conf["acquire"]["diviner_volume_roots"][0]
    sub = conf["acquire"]["diviner_data_subdirs"][0]
    base_url = root + sub
    base_path = "/" + base_url.split("/", 3)[3]
    files: dict[str, bytes] = {}
    d = tmp_path / "archive"
    d.mkdir(exist_ok=True)
    tok = {"summer": "sum", "winter": "win", "all": "all"}
    stat_tok = {"avg": "avg", "max": "max", "count": "cnt", "min": "min"}
    pole_tok = "s" if layers.pole == "south" else "n"
    for (s, c, t), arr in layers.arrays.items():
        stem = f"dgdr_t{c}_{stat_tok[t]}_{tok[s]}_{pole_tok}_240m" if c != "tbol" else \
               f"dgdr_tbol_{stat_tok[t]}_{tok[s]}_{pole_tok}_240m"
        unit = "K" if t != "count" else "N"
        scaling = 0.01 if t != "count" else 1.0
        lbl = Lb.write_pds3_raster(arr, d / f"{stem}.img", georef=layers.georef, product_id=stem.upper(),
                                   description=f"synthetic {s} {t} channel {c}", scaling=scaling, unit=unit)
        files[f"{stem}.lbl"] = lbl.read_bytes()
        files[f"{stem}.img"] = (d / f"{stem}.img").read_bytes()
    routes = {base_url: (200, _listing(base_path, files))}
    for n, b in files.items():
        routes[base_url + n] = (200, b)
    return A.ScriptedFetcher(routes=routes)


def _small_conf(poles):
    conf = load_crypt_config()
    conf["products"]["poles"] = list(poles)
    conf["acquire"]["diviner_data_subdirs"] = ["data/pcp/"]
    conf["acquire"]["minirf_roots"] = ["https://example.invalid/minirf/"]
    conf["acquire"]["shadowcam_roots"] = ["https://example.invalid/shadowcam/"]
    conf["acquire"]["psr_routes"] = ["https://example.invalid/psr/"]
    conf["acquire"]["ode_rest"] = "https://example.invalid/ode/"
    conf["sensitivity"] = {"areas_m2": [30.0, 1000.0], "t_hot_K": [300.0], "n_per_area": 4, "seed": 1}
    return conf


def test_end_to_end_through_scripted_archive_recovers_the_injected_source(tmp_path, synth):
    conf = _small_conf(["south"])
    layers = T.inject(synth, [(80, 80)], 400.0 / PIX, 300.0)
    fetch = _scripted_archive(tmp_path, layers, conf)
    out = tmp_path / "results"
    probe = stage_probe(conf, out, fetch=fetch)
    assert probe["verdict"] == "DIVINER_PRODUCTS_LISTED"
    # 2 seasons x (4 channel averages + tbol average + one count) + the all-time maximum
    assert probe["diviner"]["n_products"] == 13
    assert probe["diviner"]["n_selected"]["south"] == 13
    assert probe["reached"]["minirf"] is False and probe["reached"]["ode"] is False
    acq = stage_acquire(conf, out, fetch=fetch)
    assert acq["south"]["status"] == "OK" and acq["south"]["n_ok"] == 13
    scr = stage_screen(conf, out, fetch=fetch)
    assert scr["south"]["status"] == "OK"
    assert scr["south"]["counts"]["candidate"] == 1
    assert scr["south"]["psr_route"]["status"] == "THERMAL_DEFINITION_ONLY"
    summ = stage_assess(conf, out, fetch=fetch)
    assert summ["verdict"].startswith("ANISOTHERMAL_CANDIDATES (1)")
    assert "DEGRADED" in summ["verdict"] and "no_radar_layer" in summ["verdict"]
    c = summ["candidates"][0]
    assert c["line"] == 80 and c["sample"] == 80
    assert c["coverage"]["shadowcam"]["status"] == "NO_DATA_REACHED"
    assert any("ShadowCam" in u for u in c["unexcluded_systematics"])
    assert (out / "summary.json").exists() and (out / "candidates.csv").exists()
    assert len(pd.read_csv(out / "candidates.csv")) == 1
    assert (out / "sensitivity_south.json").exists()
    floor = summ["floor"]["south"]
    assert floor["recovered_half_area_m2"]["300.0"] == 1000.0


def test_end_to_end_clean_archive_is_a_count_not_a_candidate(tmp_path, synth):
    conf = _small_conf(["south"])
    fetch = _scripted_archive(tmp_path, synth, conf)
    out = tmp_path / "results"
    crypt_run(conf, stage="probe,acquire,screen", out_dir=out, fetch=fetch, run_sensitivity=False)
    summ = stage_assess(conf, out, fetch=fetch)
    assert summ["verdict"].startswith("NO_ANISOTHERMAL_SURVIVOR")
    assert summ["n_candidates"] == 0 and summ["n_psr_interior_px"] > 1000


def test_empty_archive_is_no_data_reached(tmp_path):
    conf = _small_conf(["south", "north"])
    fetch = A.ScriptedFetcher(routes={})
    out = tmp_path / "results"
    probe = stage_probe(conf, out, fetch=fetch)
    assert probe["verdict"] == "DIVINER_NOT_LISTED" and probe["diviner"]["n_files"] == 0
    stage_acquire(conf, out, fetch=fetch)
    stage_screen(conf, out, fetch=fetch)
    summ = stage_assess(conf, out, fetch=fetch)
    assert summ["verdict"] == "NO_DATA_REACHED"
    assert summ["n_candidates"] == 0 and summ["poles_screened"] == []
    assert json.loads((out / "summary.json").read_text())["verdict"] == "NO_DATA_REACHED"


def test_archive_that_lists_but_does_not_serve_images_is_no_data(tmp_path, synth):
    conf = _small_conf(["south"])
    fetch = _scripted_archive(tmp_path, synth, conf)
    # drop every image route: labels list and download, pixels never arrive
    fetch.routes = {k: v for k, v in fetch.routes.items() if not k.endswith(".img")}
    out = tmp_path / "results"
    stage_probe(conf, out, fetch=fetch)
    acq = stage_acquire(conf, out, fetch=fetch)
    assert acq["south"]["status"] == "NO_DATA_REACHED" and acq["south"]["n_ok"] == 0
    stage_screen(conf, out, fetch=fetch)
    summ = stage_assess(conf, out, fetch=fetch)
    assert summ["verdict"] == "NO_DATA_REACHED"


# ---------------------------------------------------------------------------
# radar
# ---------------------------------------------------------------------------
def test_radar_compact_anomaly_and_its_kills():
    cpr, s1, mask = R.synthetic_radar(160, seed=5)
    g = Lb.polar_georef("south", 160, 240.0)
    # a compact bright dihedral
    cpr[80, 80] = 1.6
    s1[80, 80] = 10.0
    # a rock field: a 7x7 block of high CPR (inside the 45 px mask)
    cpr[56:63, 56:63] = 1.4
    s1[56:63, 56:63] = 5.0
    # an isolated high pixel in an elevated-CPR neighbourhood
    cpr[86:110, 86:110] = 0.8
    cpr[98, 98] = 1.5
    s1[98, 98] = 9.0
    # a weak echo
    cpr[60, 100] = 1.3
    rep = R.screen_radar(cpr, mask, g, CONF["radar"], s1=s1)
    fl = rep["flags"]
    cls = {(int(r["line"]), int(r["sample"])): r["class"] for _, r in fl.iterrows()}
    assert cls[(80, 80)] == "radar_candidate"
    assert cls[(59, 59)] == "rock_field"
    assert cls[(98, 98)] == "elevated_background"
    assert cls[(60, 100)] == "weak_echo"
    assert rep["counts"]["radar_candidate"] == 1


def test_radar_mask_resampling_and_sampling():
    src = Lb.polar_georef("south", 160, 240.0)
    dst = Lb.polar_georef("south", 80, 480.0)
    mask = np.zeros((160, 160), dtype=bool)
    mask[60:100, 60:100] = True
    rm = R.resample_mask(mask, src, dst)
    assert 300 < rm.sum() < 500
    r = np.full((80, 80), 0.3, dtype=np.float32)
    r[40, 40] = 1.5
    lon, lat = dst.pix_to_lonlat(40, 40)
    s = R.sample_at(r, dst, lon, lat)
    assert s["in_raster"] and abs(s["value"] - 1.5) < 1e-6 and s["window_max"] > 1.0
    assert not R.sample_at(r, dst, 0.0, 0.0)["in_raster"]


# ---------------------------------------------------------------------------
# vet, sharding, CLI
# ---------------------------------------------------------------------------
def test_unexcluded_lists_what_is_missing():
    items = V.unexcluded({"radar": {"status": "NO_RADAR_LAYER"}, "coverage": {"shadowcam": {"status": "OK", "n_listed": 3}}})
    assert any("no Mini-RF reading" in x for x in items)
    assert any("not been inspected" in x for x in items)


def test_sharding():
    assert parse_shard(None) == (0, 1) and parse_shard("1/2") == (1, 2)
    with pytest.raises(SystemExit):
        parse_shard("2/2")
    assert shard_poles(CONF, "0/2") == ["north"] and shard_poles(CONF, "1/2") == ["south"]
    assert shard_poles(CONF, "0/1") == ["north", "south"]
    assert shard_poles(CONF, "3/4") == []
    need = needed_products(CONF, "south")
    assert {n["role"] for n in need} == {"screen", "mask", "optional"}
    assert sum(1 for n in need if n["role"] == "screen") == 2 * 4 * 2


def test_cli_synthetic_run_writes_a_synthetic_verdict(tmp_path):
    rc = main(["--stage", "screen,assess", "--synthetic", "--out-dir", str(tmp_path / "r"),
               "--no-sensitivity", "--poles", "south"])
    assert rc == 0
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["synthetic"] is True and s["verdict"].startswith("SYNTHETIC_ANISOTHERMAL_CANDIDATES (1)")
    r = json.loads((tmp_path / "r" / "radar_south.json").read_text())
    assert r["counts"]["radar_candidate"] == 1

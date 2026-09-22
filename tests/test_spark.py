"""Offline suite for SPARK (S48 SPHEREx / S49 Euclid single-element excess).

No network (``conftest.py`` raises on any socket).  Per ``docs/channel-brief.md`` §5:

* an injected single-channel excess is recovered — in the Euclid line table
  (a 1.55 µm feature on a Gaia star at S/N 12 in 3 dithers survives every
  veto and carries the Er-fibre descriptor) and in synthetic SPHEREx
  cutouts rendered through a fake archive (a 30 % excess confined to one
  resolution element at 1.30 µm comes out ``candidate``);
* the dominant confounders are rejected: a galaxy's Hα/[N II]/[S II]
  pattern at z = 1.2 on a mis-seeded object trips ``galaxy_pattern``; a
  broad bump is not a single-channel excess; the same channel on three
  stars is instrumental;
* an empty / failed archive yields ``NO_DATA_REACHED``, never a candidate;
* every rejection rule has a case that trips it.
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.spark import euclid as E
from seti.spark import gaia as G
from seti.spark import lines as L
from seti.spark import run as R
from seti.spark import spherex as S

CONF = R.load_spark_config()
EC = CONF["euclid"]
SC = CONF["spherex"]


# --------------------------------------------------------------------------
# lines.py
# --------------------------------------------------------------------------
def test_air_to_vacuum_nd_yag():
    assert abs(L.air_to_vacuum_um(1.0641) - 1.0644) < 2e-4


def test_stellar_line_match_and_bands():
    m = L.stellar_line_match(1.2822, 0.003)
    assert m is not None and m.name == "Pa-beta"
    assert L.stellar_line_match(1.5500, 0.003) is None          # inside the Er band, no stellar line
    band = L.stellar_line_match(1.42, 0.003)                     # H2O 1.4 band
    assert band is not None and band.kind == "band"
    assert L.industrial_flag(1.5500, 0.003) == "Er fibre C-band"
    assert L.industrial_flag(1.0644, 0.003).startswith("Nd:YAG 1064")
    assert L.industrial_flag(1.75, 0.003) is None


def test_bands_veto_only_at_their_edges_for_a_single_channel_statistic():
    # A 0.16 um band is 5 resolution elements wide at R = 40: it cannot make an
    # excess confined to one channel, so its interior must not veto.  Its edge,
    # where the gradient is, still does.
    tol = 1.40 / 40.0
    assert L.stellar_line_match(1.40, tol, bands_edge_only=True) is None
    edge = L.stellar_line_match(1.48, 0.004, bands_edge_only=True)   # the band's upper edge
    assert edge is not None and edge.name == "H2O 1.4 band"
    interior = L.stellar_line_match(1.40, tol)                   # default: whole interior
    assert interior is not None and interior.name == "H2O 1.4 band"
    # a discrete line inside a band still vetoes under the edge-only rule
    assert L.stellar_line_match(1.4879, 1.4879 / 40.0, bands_edge_only=True) is not None


def test_clean_channel_fraction_is_a_coverage_number():
    lo = L.clean_channel_fraction(1.11, 1.64, 40.0)
    hi = L.clean_channel_fraction(1.11, 1.64, 130.0)
    assert lo["n_bins"] > 0 and 0.0 < lo["fraction"] < 0.5      # R=40: most of D2 is stellar-line territory
    assert hi["fraction"] > lo["fraction"]                       # resolution buys back clean channels
    assert L.clean_channel_fraction(1.0, 0.5, 40.0)["fraction"] is None


def test_redshift_pattern_vetoes_galaxy_and_passes_single_line():
    z = 1.2
    lam = [0.656461 * (1 + z), 0.658527 * (1 + z), 0.671829 * (1 + z)]
    r = L.redshift_pattern(lam)
    assert r["vetoed"] and abs(r["z"] - z) < 0.002 and r["n_lines"] == 3
    # a z ~ 0 nebular pair (He I + Pa-gamma) is also a pattern
    r0 = L.redshift_pattern([1.08332, 1.09411])
    assert r0["vetoed"] and abs(r0["z"]) < 0.003
    # one lone feature cannot form a pattern
    assert not L.redshift_pattern([1.55])["vetoed"]
    # two features with no common redshift
    assert not L.redshift_pattern([1.55, 1.30])["vetoed"] or L.redshift_pattern([1.55, 1.30])["n_lines"] >= 2
    ids = L.single_line_interpretations(1.55)
    assert any(d["line"] == "H-alpha" and abs(d["z"] - (1.55 / 0.656461 - 1)) < 1e-3 for d in ids)


def test_line_table_has_sources():
    rows = L.line_table()
    assert all(r["src"] for r in rows)
    assert {r["list"] for r in rows} == {"stellar", "galaxy", "industrial"}


# --------------------------------------------------------------------------
# gaia.py
# --------------------------------------------------------------------------
def test_crossmatch_with_proper_motion():
    g = pd.DataFrame({"source_id": [1, 2], "ra": [10.0, 10.1], "dec": [20.0, 20.0],
                      "pmra": [1000.0, 0.0], "pmdec": [0.0, 0.0], "phot_g_mean_mag": [12.0, 13.0]})
    gp = G.propagate(g, 2016.0, 2024.0)
    # 8 yr x 1 arcsec/yr = 8 arcsec east: the unpropagated position no longer matches
    idx, sep = G.crossmatch([10.0], [20.0], gp["ra_ep"], gp["dec_ep"], 1.0)
    assert idx[0] == -1
    idx, sep = G.crossmatch(gp["ra_ep"].iloc[:1], gp["dec_ep"].iloc[:1], gp["ra_ep"], gp["dec_ep"], 1.0)
    assert idx[0] == 0 and sep[0] < 1e-6


def test_isolation_flags_bright_neighbour():
    seeds = pd.DataFrame({"source_id": [1], "ra": [10.0], "dec": [0.0], "phot_g_mean_mag": [12.0]})
    field = pd.DataFrame({"source_id": [1, 2, 3], "ra": [10.0, 10.0 + 5 / 3600, 10.0 + 15 / 3600],
                          "dec": [0.0, 0.0, 0.0], "phot_g_mean_mag": [12.0, 20.0, 13.0]})
    iso = G.isolation(seeds, field, outer_arcsec=20, outer_dmag=3, inner_arcsec=9, inner_dmag=6)
    assert not bool(iso["isolated"].iloc[0]) and iso["n_within_outer"].iloc[0] == 2
    field2 = field[field["source_id"] != 3]
    iso2 = G.isolation(seeds, field2, outer_arcsec=20, outer_dmag=3, inner_arcsec=9, inner_dmag=6)
    assert bool(iso2["isolated"].iloc[0])


# --------------------------------------------------------------------------
# euclid.py — geometry, schema, wavelength scale, dedupe
# --------------------------------------------------------------------------
def test_field_strips_cover_the_cone_and_shards_partition():
    f = {"ra": 269.73, "dec": 66.01, "radius_deg": 2.7}
    strips = E.field_strips(f, 0.5)
    assert strips[0]["dec_lo"] <= 66.01 - 2.7 + 1e-6 and strips[-1]["dec_hi"] >= 66.01 + 2.7 - 1e-6
    assert all(s["ra_lo"] < 269.73 < s["ra_hi"] for s in strips)
    fields = {"A": f, "B": {"ra": 61.24, "dec": -48.42, "radius_deg": 2.9}}
    allu = [u for i in range(3) for u in E.units_for_shard(fields, 0.5, i, 3)]
    assert len(allu) == len(E.units_for_shard(fields, 0.5, 0, 1))


def test_roles_and_adql_use_only_seen_columns():
    cols = ["object_id", "spe_rank", "spe_line_name", "spe_line_n_dith", "spe_line_central_wl_gf",
            "spe_line_snr_gf", "spe_line_fwhm_gf", "spe_line_flux_gf"]
    rl = E.resolve_roles(cols, E.LINE_ROLES)
    assert rl["wl"] == "spe_line_central_wl_gf" and rl["snr"] == "spe_line_snr_gf" and rl["ew"] is None
    assert E.missing_required(rl, E.LINE_REQUIRED) == []
    rm = E.resolve_roles(["object_id", "right_ascension", "declination", "point_like_prob"], E.MER_ROLES)
    unit = E.units_for_shard({"F": {"ra": 100.0, "dec": 10.0, "radius_deg": 1.0}}, 0.5)[0]
    q = E.strip_adql({"lines": "L", "mer": "M"}, rl, rm, unit, snr_min=3.0)
    assert "spe_line_ew" not in q and "JOIN M m ON l.object_id = m.object_id" in q
    assert "l.spe_line_snr_gf >= 3.00" in q and "m.declination BETWEEN" in q
    assert E.missing_required(E.resolve_roles(["object_id"], E.MER_ROLES), E.MER_REQUIRED) == ["ra", "dec"]


def test_wavelength_scale_inference():
    assert E.infer_wavelength_scale([15000.0, 16000.0])[0] == 1e-4
    assert E.infer_wavelength_scale([1500.0, 1600.0])[0] == 1e-3
    assert E.infer_wavelength_scale([1.5, 1.6])[0] == 1.0
    assert E.infer_wavelength_scale([1.5], "angstrom")[0] == 1e-4


def _raw_rows(oid, wl_um, snr, *, ra, dec, n_dith=3, fwhm_um=0.003, flux=1e-16, name="Halpha",
              rank=1, point_like=0.99, spurious=0):
    return {"l_object_id": oid, "l_rank": rank, "l_line_id": 1, "l_line_name": name, "l_line_flag": 0,
            "l_n_dith": n_dith, "l_wl": wl_um * 1e4, "l_wl_err": 5.0, "l_flux": flux, "l_flux_err": flux / snr,
            "l_ew": -50.0, "l_fwhm": fwhm_um * 1e4, "l_snr": snr, "l_snr_di": snr, "l_cont": 1e-17,
            "l_qual": 0, "l_aon": snr, "m_ra": ra, "m_dec": dec, "m_point_like_prob": point_like,
            "m_point_like_flag": 1, "m_spurious_flag": spurious, "m_det_quality_flag": 0,
            "m_mumax_minus_mag": -2.6, "m_flux_h": 100.0, "m_flux_vis": 100.0, "m_segmentation_area": 30}


def test_normalise_and_dedupe_across_ranks():
    raw = pd.DataFrame([_raw_rows(1, 1.55, 12.0, ra=10.0, dec=0.0, rank=1, name="Halpha"),
                        _raw_rows(1, 1.5502, 11.0, ra=10.0, dec=0.0, rank=2, name="OIII5007"),
                        _raw_rows(1, 1.30, 6.0, ra=10.0, dec=0.0, rank=1, name="Hbeta")])
    feat, meta = E.normalise_features(raw, "F")
    assert meta["wavelength_scale"] == 1e-4 and abs(feat["wl_um"].iloc[0] - 1.55) < 1e-6
    d = E.dedupe_features(feat, 450.0)
    assert len(d) == 2
    top = d[d["wl_um"] > 1.5].iloc[0]
    assert top["snr"] == 12.0 and "OIII5007" in top["names"] and top["n_ranks"] == 2


# --------------------------------------------------------------------------
# euclid.py — the screen: injection, confounder, every veto
# --------------------------------------------------------------------------
def _gaia(rows):
    return pd.DataFrame(rows, columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax",
                                       "parallax_over_error", "phot_g_mean_mag", "bp_rp", "ruwe",
                                       "phot_variable_flag"])


def _star(sid, ra, dec, g=14.0, poe=30.0):
    return [sid, ra, dec, 0.0, 0.0, 10.0, poe, g, 1.0, 1.0, "NOT_AVAILABLE"]


def _screen(rows, gaia_rows, **kw):
    raw = pd.DataFrame(rows)
    feat, _ = E.normalise_features(raw, "F")
    conf = {**EC, **kw}
    return E.screen_features(feat, _gaia(gaia_rows), conf, n_stars_with_spectra=1000)


def test_injected_feature_on_a_star_survives_with_the_er_descriptor():
    d, f = _screen([_raw_rows(1, 1.5500, 12.0, ra=10.0, dec=0.0)], [_star(101, 10.0, 0.0)])
    assert len(d) == 1 and bool(d["survivor"].iloc[0]), d.iloc[0].to_dict()
    r = d.iloc[0]
    assert r["gaia_source_id"] == 101 and r["industrial_flag"] == "Er fibre C-band"
    assert r["p_global"] > r["p_single"] and bool(r["significant_after_trials"])   # S/N 12 survives 1000 x 200 trials
    assert f["n_survivors"] == 1 and f["n_features_not_on_gaia_star"] == 0
    z = json.loads(r["single_line_z"])
    assert any(x["line"] == "H-alpha" for x in z)


def test_galaxy_pattern_is_vetoed_even_on_a_gaia_match():
    z = 1.2
    rows = [_raw_rows(7, 0.656461 * (1 + z), 9.0, ra=10.0, dec=0.0, rank=1),
            _raw_rows(7, 0.658527 * (1 + z), 6.0, ra=10.0, dec=0.0, rank=1),
            _raw_rows(7, 0.671829 * (1 + z), 5.5, ra=10.0, dec=0.0, rank=1)]
    d, f = _screen(rows, [_star(101, 10.0, 0.0)])
    assert len(d) == 3 and d["galaxy_pattern"].all() and not d["survivor"].any()
    assert abs(float(d["galaxy_z"].iloc[0]) - z) < 0.003


def test_feature_without_a_gaia_star_is_not_in_the_star_sample():
    d, f = _screen([_raw_rows(1, 1.55, 12.0, ra=10.0, dec=0.0)], [_star(101, 11.0, 0.0)])
    assert len(d) == 0 and f["n_features_not_on_gaia_star"] == 1 and f["n_survivors"] == 0


@pytest.mark.parametrize("kw,veto", [
    ({"snr": 4.0}, "low_snr"),
    ({"n_dith": 1}, "few_dithers"),
    ({"wl_um": 1.215}, "band_edge"),
    ({"wl_um": 1.885}, "band_edge"),
    ({"fwhm_um": 0.02}, "broad"),
    ({"flux": -1e-16}, "non_positive_flux"),
    ({"wl_um": 1.28216}, "stellar_line"),
    ({"wl_um": 1.7200}, None),
    ({"spurious": 1}, "spurious"),
])
def test_each_single_feature_veto_trips(kw, veto):
    base = {"wl_um": 1.55, "snr": 12.0, "ra": 10.0, "dec": 0.0}
    base.update(kw)
    row = _raw_rows(1, base.pop("wl_um"), base.pop("snr"), **base)
    d, _ = _screen([row], [_star(101, 10.0, 0.0)])
    assert len(d) == 1
    if veto is None:
        assert bool(d["survivor"].iloc[0])
    else:
        assert bool(d[veto].iloc[0]) and not bool(d["survivor"].iloc[0])
        assert veto in d["vetoes"].iloc[0]


def test_stellar_line_veto_names_the_line_and_co_bandheads_count():
    d, _ = _screen([_raw_rows(1, 1.5582, 12.0, ra=10.0, dec=0.0)], [_star(101, 10.0, 0.0)])
    assert bool(d["stellar_line"].iloc[0]) and d["stellar_line_name"].iloc[0].startswith("CO 3-0")


def test_blend_veto_from_a_gaia_neighbour_and_dispersion_neighbour_count():
    gaia = [_star(101, 10.0, 0.0), _star(102, 10.0 + 3 / 3600, 0.0, g=15.0)]
    d, _ = _screen([_raw_rows(1, 1.55, 12.0, ra=10.0, dec=0.0)], gaia)
    assert bool(d["blend"].iloc[0]) and not bool(d["survivor"].iloc[0])
    gaia2 = [_star(101, 10.0, 0.0), _star(103, 10.0 + 60 / 3600, 0.0 + 1 / 3600, g=12.0)]
    d2, _ = _screen([_raw_rows(1, 1.55, 12.0, ra=10.0, dec=0.0)], gaia2)
    assert not bool(d2["blend"].iloc[0]) and int(d2["n_dispersion_neighbours"].iloc[0]) == 1
    assert bool(d2["survivor"].iloc[0])


def test_recurrent_wavelength_on_three_stars_is_instrumental():
    rows, gaia = [], []
    for k in range(3):
        rows.append(_raw_rows(k + 1, 1.55, 10.0, ra=10.0 + k * 0.01, dec=0.0))
        gaia.append(_star(100 + k, 10.0 + k * 0.01, 0.0))
    d, f = _screen(rows, gaia)
    assert d["recurrent_wavelength"].all() and f["n_survivors"] == 0
    assert f["recurrent_bins"] and abs(f["recurrent_bins"][0]["wl_um"] - 1.55) < 0.01


def test_point_like_is_a_descriptor_unless_required():
    row = _raw_rows(1, 1.55, 12.0, ra=10.0, dec=0.0, point_like=0.1)
    d, _ = _screen([row], [_star(101, 10.0, 0.0)])
    assert bool(d["point_like_below_min"].iloc[0]) and bool(d["survivor"].iloc[0])
    d2, _ = _screen([row], [_star(101, 10.0, 0.0)], require_point_like=True)
    assert bool(d2["not_point_like"].iloc[0]) and not bool(d2["survivor"].iloc[0])


def test_screen_on_empty_input_does_not_invent():
    feat, _ = E.normalise_features(pd.DataFrame(columns=["l_object_id", "l_wl", "l_snr"]), "F")
    d, f = E.screen_features(feat, _gaia([_star(1, 0, 0)]), EC)
    assert f["n_survivors"] == 0 and len(d) == 0


def test_fetch_strip_splits_on_truncation_and_records_failure():
    calls = []

    def q(adql, maxrec=None):
        calls.append(adql)
        m = re.search(r"BETWEEN ([-\d.]+) AND ([-\d.]+)", adql)
        lo, hi = float(m.group(1)), float(m.group(2))
        n = maxrec if hi - lo > 0.3 else 5
        return pd.DataFrame({"x": np.arange(n)})

    unit = {"field": "F", "dec_lo": 0.0, "dec_hi": 1.0, "ra_lo": 0.0, "ra_hi": 1.0}
    log: list = []
    df, rec = E.fetch_strip(lambda u: f"SELECT x WHERE d BETWEEN {u['dec_lo']} AND {u['dec_hi']}", unit, q,
                            maxrec=100, label="t", log=log)
    assert rec["status"] == "SPLIT" and len(df) == 20 and len(calls) == 7
    assert any(r["status"] == "SPLIT" for r in log) and all(r["status"] != "QUERY_FAILED" for r in log)

    def bad(adql, maxrec=None):
        raise RuntimeError("archive down")
    df2, rec2 = E.fetch_strip(lambda u: "SELECT", unit, bad, maxrec=100, label="t")
    assert rec2["status"] == "QUERY_FAILED" and len(df2) == 0


def test_stars_with_spectra_denominator():
    spec = pd.DataFrame({"m_object_id": [1, 2, 3], "m_ra": [10.0, 10.01, 10.02], "m_dec": [0.0, 0.0, 0.0]})
    n, m = E.stars_with_spectra(spec, _gaia([_star(101, 10.0, 0.0), _star(102, 10.02, 0.0)]), EC)
    assert n == 2 and set(m["gaia_source_id"]) == {101, 102}


# --------------------------------------------------------------------------
# spherex.py — discovery helpers and photometry
# --------------------------------------------------------------------------
def test_detector_from_bandpass_uri_and_em_range():
    bands = SC["detector_bands_um"]
    assert S.detector_of({"energy_bandpassname": "SPHEREx-D3"}, bands) == "D3"
    assert S.detector_of({"access_url": "ibe/data/spherex/qr2/level2/x/level2_2025W45_2A_0001_1D2_spx.fits"}, bands) == "D2"
    assert S.detector_of({"em_min": 4.5e-6, "em_max": 4.9e-6}, bands) == "D6"
    assert S.detector_of({"em_min": 1.2, "em_max": 1.5}, bands) == "D2"
    assert S.detector_of({}, bands) is None


def test_select_images_spreads_in_time_and_cutout_url():
    df = pd.DataFrame({"obs_id": [f"o{i}" for i in range(50)], "t_min": np.arange(50.0),
                       "access_url": [f"ibe/x/level2_2025_0001_1D{1 + i % 2}_spx.fits" for i in range(50)]})
    sel = S.select_images(df, SC["detector_bands_um"], 5)
    assert set(sel["detector"]) == {"D1", "D2"} and len(sel) == 10
    assert sel[sel["detector"] == "D1"]["t_min"].iloc[-1] >= 40
    u = S.cutout_url(S.product_url("https://irsa.ipac.caltech.edu/", "ibe/data/a.fits"), 270.0, 66.56, 900.0)
    assert u.startswith("https://irsa.ipac.caltech.edu/ibe/data/a.fits?center=270.000000,66.560000&size=900.0arcsec")


def _gauss_star(ny, nx, x0, y0, total, sigma=0.7):
    yy, xx = np.mgrid[0:ny, 0:nx]
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return total * g / g.sum()


def test_forced_photometry_recovers_a_point_source_and_masks_flags():
    rng = np.random.default_rng(1)
    ny = nx = 40
    omega = (6.15 / 206264.806) ** 2
    f_mjy = 50.0
    total = f_mjy / (omega * 1e9)                       # MJy/sr summed over pixels
    img = 3.0 + _gauss_star(ny, nx, 20.3, 19.6, total, sigma=0.5) + rng.normal(0, 0.05, (ny, nx))
    var = np.full((ny, nx), 0.05 ** 2)
    flags = np.zeros((ny, nx), int)
    r = S.forced_photometry(img, 20.3, 19.6, variance=var, flags=flags, zodi=np.full((ny, nx), 1.0))
    assert r["ok"] and abs(r["flux_mjy"] - f_mjy) < 3 * r["err_mjy"] and abs(r["flux_mjy"] - f_mjy) / f_mjy < 0.05
    flags[20, 20] = 4
    r2 = S.forced_photometry(img, 20.3, 19.6, variance=var, flags=flags)
    assert not r2["ok"] and r2["reason"] == "flagged" and r2["n_flagged"] == 1
    assert not S.forced_photometry(img, 2.0, 2.0)["ok"]


def test_wavemap_synthetic_and_celestial_header_stripping():
    from astropy.io import fits
    wm = S.WaveMap.synthetic(100, 200, 1.11, 1.64, R=40.0, axis="y")
    lam, bw = wm.wavelength_at([50.0, 50.0], [0.0, 199.0])
    assert abs(lam[0] - 1.11) < 1e-6 and abs(lam[1] - 1.64) < 1e-6 and abs(bw[0] - 1.11 / 40) < 1e-6
    assert wm.fingerprint()["gradient_axis"] == "y"
    h = fits.Header()
    h["NAXIS"] = 2
    h["WCSAXES"] = 3
    h["CTYPE1"], h["CTYPE2"], h["CTYPE3"] = "RA---TAN", "DEC--TAN", "WAVE-TAB"
    h["CRPIX1"], h["CRPIX2"], h["CRVAL1"], h["CRVAL2"] = 50.0, 50.0, 270.0, 66.0
    h["CDELT1"], h["CDELT2"] = -6.15 / 3600, 6.15 / 3600
    h["PS3_0"], h["PS3_1"], h["PV3_1"] = "WCS-WAVE", "WAVELENGTH", 1
    h["CTYPE3W"], h["PS3_0W"] = "WAVE-TAB", "WCS-WAVE"
    hc = S.celestial_header(h)
    assert "CTYPE3" not in hc and "PS3_0" not in hc and "CTYPE3W" not in hc and hc["WCSAXES"] == 2
    w = S.celestial_wcs(h)
    x, y = w.all_world2pix([270.0], [66.0], 0)
    assert abs(x[0] - 49.0) < 1e-6 and abs(y[0] - 49.0) < 1e-6
    assert S.cutout_offset({"LTV1": -100.0, "LTV2": -200.0}, None) == (100.0, 200.0)
    assert S.cutout_offset({"CRPIX1": 20.5, "CRPIX2": 30.5}, (1020.5, 1020.5)) == (1000.0, 990.0)
    assert S.cutout_offset({}, None) is None


# --------------------------------------------------------------------------
# spherex.py — the statistic
# --------------------------------------------------------------------------
def _synthetic_samples(rng, *, n_per_pass=90, passes=(0.0, 200.0), inject_bin=None, inject_frac=0.3,
                       bump_um=None, snr=30.0, one_position=False, detector="D2", lam_lo=1.11, lam_hi=1.64,
                       R=40.0):
    rows = []
    for p, mjd0 in enumerate(passes):
        for k in range(n_per_pass):
            lam = lam_lo * (lam_hi / lam_lo) ** rng.uniform()
            bw = lam / R
            cont = 100.0 * (1.3 / lam) ** 2
            f = cont
            in_bin = inject_bin is not None and int(math.floor(math.log(lam) * R)) == inject_bin
            if in_bin:
                f *= 1.0 + inject_frac
                if one_position:
                    lam = math.exp((inject_bin + 0.5) / R)
                    bw = lam / R
            if bump_um is not None:
                f *= 1.0 + 0.3 * math.exp(-0.5 * ((lam - bump_um) / (4 * bw)) ** 2)
            err = cont / snr
            x = 1000.0 if (one_position and in_bin) else 1000.0 + rng.uniform(-900, 900)
            y = (math.log(lam / lam_lo) / math.log(lam_hi / lam_lo)) * 2039.0
            rows.append({"source_id": 1, "detector": detector, "wl_um": lam, "bw_um": bw,
                         "flux_mjy": f + rng.normal(0, err), "err_mjy": err, "mjd": mjd0 + rng.uniform(0, 5),
                         "x_full": x, "y_full": y, "obs_id": f"p{p}k{k}"})
    return pd.DataFrame(rows)


INJECT_BIN = int(math.floor(math.log(1.30) * 40))


def test_single_channel_excess_is_recovered_and_null_is_clean():
    rng = np.random.default_rng(3)
    d = _synthetic_samples(rng, inject_bin=INJECT_BIN)
    ch, per = S.detect_single_channel_excess(d, SC, SC["detector_bands_um"])
    cands = [c for c in ch if c["tier"] == "candidate"]
    assert len(cands) == 1 and cands[0]["channel_bin"] == INJECT_BIN, [(c["channel_bin"], c["tier"], c["reasons"]) for c in ch]
    c = cands[0]
    assert c["z_combined"] >= 5 and c["n_passes_above_single"] >= 2 and c["n_positions_above_single"] >= 2
    assert abs(c["excess_fraction"] - 0.3) < 0.1
    assert per["D2"]["status"] == "ok" and per["D2"]["n_passes"] == 2
    null = _synthetic_samples(rng)
    ch0, _ = S.detect_single_channel_excess(null, SC, SC["detector_bands_um"])
    assert not [c for c in ch0 if c["tier"] == "candidate"]
    assert all(abs(c["z_combined"] or 0) < 5 for c in ch0)


def test_broad_bump_is_not_a_single_channel_excess():
    rng = np.random.default_rng(4)
    d = _synthetic_samples(rng, bump_um=1.30, snr=60.0)
    ch, _ = S.detect_single_channel_excess(d, SC, SC["detector_bands_um"])
    assert not [c for c in ch if c["tier"] == "candidate"]


def test_one_pass_or_one_position_is_not_a_candidate():
    rng = np.random.default_rng(5)
    d1 = _synthetic_samples(rng, inject_bin=INJECT_BIN, passes=(0.0,), n_per_pass=180)
    ch1, _ = S.detect_single_channel_excess(d1, SC, SC["detector_bands_um"])
    hot = [c for c in ch1 if c["channel_bin"] == INJECT_BIN]
    assert hot and hot[0]["tier"] != "candidate" and "fewer_than_two_passes" in hot[0]["reasons"]
    d2 = _synthetic_samples(rng, inject_bin=INJECT_BIN, one_position=True)
    ch2, _ = S.detect_single_channel_excess(d2, SC, SC["detector_bands_um"])
    hot2 = [c for c in ch2 if c["channel_bin"] == INJECT_BIN]
    assert hot2 and "fewer_than_two_positions" in hot2[0]["reasons"]


def test_lvf_edge_channel_is_flagged():
    rng = np.random.default_rng(6)
    edge_bin = int(math.floor(math.log(1.115) * 40))
    d = _synthetic_samples(rng, inject_bin=edge_bin, n_per_pass=200)
    ch, _ = S.detect_single_channel_excess(d, SC, SC["detector_bands_um"])
    hot = [c for c in ch if c["channel_bin"] == edge_bin]
    assert hot and "lvf_edge" in hot[0]["reasons"]


def test_insufficient_samples_is_recorded_not_tested():
    rng = np.random.default_rng(7)
    d = _synthetic_samples(rng, n_per_pass=3)
    ch, per = S.detect_single_channel_excess(d, SC, SC["detector_bands_um"])
    assert per["D2"]["status"] == "insufficient_samples" and ch == []


def test_screen_spherex_recurrence_and_stellar_line_vetoes():
    rng = np.random.default_rng(8)
    parts = []
    for sid in (1, 2, 3):
        d = _synthetic_samples(rng, inject_bin=INJECT_BIN)
        d["source_id"] = sid
        parts.append(d)
    hei_bin = int(math.floor(math.log(1.0833) * 40))
    d4 = _synthetic_samples(rng, inject_bin=hei_bin, detector="D1", lam_lo=0.75, lam_hi=1.11)
    d4["source_id"] = 4
    parts.append(d4)
    seeds = pd.DataFrame({"source_id": [1, 2, 3, 4]})
    chans, f = S.screen_spherex(pd.concat(parts, ignore_index=True), seeds, SC, SC["detector_bands_um"])
    rec = chans[(chans["channel_bin"] == INJECT_BIN) & (chans["detector"] == "D2")]
    assert len(rec) == 3 and rec["recurrent_channel"].all() and not rec["survivor"].any()
    he = chans[(chans["source_id"] == 4) & (chans["channel_bin"] == hei_bin)]
    assert len(he) == 1 and bool(he["stellar_line"].iloc[0])
    assert he["stellar_line_name"].iloc[0] in ("He I 1.083", "Pa-gamma")     # both inside one R=40 element
    assert f["n_survivors"] == 0 and f["n_candidates_before_vetoes"] >= 3 and f["n_trials"] >= 40
    assert f["recurrent_channels"][0]["n_stars"] == 3


# --------------------------------------------------------------------------
# run.py — degradation and the end-to-end fake archives
# --------------------------------------------------------------------------
def test_offline_stages_degrade_to_no_data(tmp_path):
    out = tmp_path / "spark"
    p = R.probe(CONF, out)
    assert p["status"] == "NO_QUERY_FUNCTION"
    led = R.euclid_stage(CONF, out)
    assert led["status"] == R.VERDICT_NO_DATA
    sled = R.spherex_stage(CONF, out)
    assert sled["status"] == R.VERDICT_NO_DATA
    s = R.assess(CONF, out)
    assert s["verdict"] == R.VERDICT_NO_DATA and s["stage_counts"]["euclid_survivors"] == 0
    assert (out / "summary.json").exists() and (out / "candidates.csv").exists()


class FakeIRSA:
    """An IRSA TAP that serves the Euclid Q1 schema, a joined strip, Gaia, and obscore."""

    def __init__(self, strip_rows: pd.DataFrame, spec_rows: pd.DataFrame, gaia: pd.DataFrame,
                 products: pd.DataFrame | None = None, empty: bool = False, fail_lines: bool = False):
        self.strip_rows, self.spec_rows, self.gaia, self.products = strip_rows, spec_rows, gaia, products
        self.empty, self.fail_lines = empty, fail_lines
        self.queries: list[str] = []

    def __call__(self, adql: str, maxrec=None):
        self.queries.append(adql)
        a = adql.lower()
        if "tap_schema.columns" in a:
            t = re.search(r"table_name = '([^']+)'", adql).group(1)
            cols = {"euclid_q1_spe_lines_line_features": [k for k in
                    ["object_id", "spe_rank", "spe_line_id", "spe_line_flag", "spe_line_name", "spe_line_n_dith",
                     "spe_line_central_wl_gf", "spe_line_central_wl_err_gf", "spe_line_flux_gf", "spe_line_flux_err_gf",
                     "spe_line_ew_gf", "spe_line_fwhm_gf", "spe_line_snr_gf", "spe_line_cont_gf", "spe_line_qual_gf",
                     "spe_line_snr_di", "spe_line_aon"]],
                    "euclid_q1_mer_catalogue": ["object_id", "right_ascension", "declination", "point_like_prob",
                                                "spurious_flag", "det_quality_flag", "flux_h_2fwhm_aper", "mumax_minus_mag"],
                    "euclid_q1_spectro_zcatalog_spe_classification": ["object_id", "spe_class", "spe_class_prob_star"],
                    "gaia_dr3_source": list(G.GAIA_COLS),
                    "spherex.obscore": list(S.OBSCORE_COLS)}.get(t, [])
            return pd.DataFrame({"column_name": cols, "datatype": ["x"] * len(cols), "description": [""] * len(cols)})
        if "tap_schema.tables" in a:
            return pd.DataFrame({"table_name": ["gaia_dr3_source"]})
        if "count(*)" in a and "spherex.obscore" in a:
            return pd.DataFrame({"n": [0 if self.products is None else len(self.products)]})
        if "count(*)" in a:
            return pd.DataFrame({"n": [len(self.strip_rows)]})
        if "group by" in a:
            return pd.DataFrame({"nm": ["Halpha"], "n": [len(self.strip_rows)]})
        if "from gaia_dr3_source" in a:
            m = re.search(r"CIRCLE\('ICRS', ([-\d.]+), ([-\d.]+), ([-\d.]+)\)", adql)
            ra0, dec0, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
            sep = G.sep_arcsec(self.gaia["ra"], self.gaia["dec"], ra0, dec0) / 3600.0
            gmax = float(re.search(r"phot_g_mean_mag < ([\d.]+)", adql).group(1))
            return self.gaia[(sep <= r) & (self.gaia["phot_g_mean_mag"] < gmax)].reset_index(drop=True)
        if "from spherex.obscore" in a:
            df = self.products if self.products is not None else pd.DataFrame(columns=list(S.OBSCORE_COLS))
            m = re.search(r"TOP (\d+)", adql)
            return df.head(int(m.group(1))) if m else df
        if "spherex.plane" in a:
            return pd.DataFrame({"planeid": [], "energy_bandpassname": [], "uri": []})
        if "euclid_q1_spe_lines_line_features l join" in a:
            if self.fail_lines:
                raise RuntimeError("statement timeout")
            if self.empty:
                return self.strip_rows.head(0)
            m = re.search(r"BETWEEN ([-\d.]+) AND ([-\d.]+)", adql)
            lo, hi = float(m.group(1)), float(m.group(2))
            d = self.strip_rows[(self.strip_rows["m_dec"] >= lo) & (self.strip_rows["m_dec"] < hi)]
            if "top 5" in a:
                d = d.head(5)
            return d.reset_index(drop=True)
        if "euclid_q1_spectro_zcatalog_spe_classification s join" in a:
            m = re.search(r"BETWEEN ([-\d.]+) AND ([-\d.]+)", adql)
            lo, hi = float(m.group(1)), float(m.group(2))
            return self.spec_rows[(self.spec_rows["m_dec"] >= lo) & (self.spec_rows["m_dec"] < hi)].reset_index(drop=True)
        raise AssertionError(f"unexpected ADQL: {adql[:200]}")


def _euclid_world(rng, n_bg=60):
    """A field with an injected star feature, a galaxy pattern, a stellar line and background objects."""
    f = {"ra": 100.0, "dec": 10.0, "radius_deg": 0.6}
    gaia_rows, strip, spec = [], [], []
    # the injected star: one 1.55 um feature in 3 dithers
    gaia_rows.append(_star(1001, 100.0, 10.05))
    strip.append(_raw_rows(1, 1.5500, 11.0, ra=100.0, dec=10.05))
    spec.append({"m_object_id": 1, "m_ra": 100.0, "m_dec": 10.05, "m_point_like_prob": 0.99, "s_spe_class": "GALAXY"})
    # a star with Pa-beta
    gaia_rows.append(_star(1002, 100.1, 10.1))
    strip.append(_raw_rows(2, 1.28216, 9.0, ra=100.1, dec=10.1))
    spec.append({"m_object_id": 2, "m_ra": 100.1, "m_dec": 10.1, "m_point_like_prob": 0.99, "s_spe_class": "STAR"})
    # a galaxy with a Gaia match (a mis-seeded compact galaxy with a spurious parallax)
    z = 1.2
    gaia_rows.append(_star(1003, 100.2, 9.9, poe=6.0))
    for wl in (0.656461, 0.658527, 0.671829):
        strip.append(_raw_rows(3, wl * (1 + z), 7.0, ra=100.2, dec=9.9))
    spec.append({"m_object_id": 3, "m_ra": 100.2, "m_dec": 9.9, "m_point_like_prob": 0.7, "s_spe_class": "GALAXY"})
    # background galaxies without Gaia counterparts, and stars with spectra but no lines
    for k in range(n_bg):
        ra, dec = 100.0 + rng.uniform(-0.4, 0.4), 10.0 + rng.uniform(-0.4, 0.4)
        oid = 100 + k
        strip.append(_raw_rows(oid, rng.uniform(1.25, 1.85), rng.uniform(3, 8), ra=ra, dec=dec, point_like=0.01))
        spec.append({"m_object_id": oid, "m_ra": ra, "m_dec": dec, "m_point_like_prob": 0.01, "s_spe_class": "GALAXY"})
    for k in range(20):
        ra, dec = 100.0 + rng.uniform(-0.4, 0.4), 10.0 + rng.uniform(-0.4, 0.4)
        gaia_rows.append(_star(2000 + k, ra, dec))
        spec.append({"m_object_id": 3000 + k, "m_ra": ra, "m_dec": dec, "m_point_like_prob": 0.99, "s_spe_class": "STAR"})
    return f, _gaia(gaia_rows), pd.DataFrame(strip), pd.DataFrame(spec)


def _conf_for(field: dict, boxes=None) -> dict:
    conf = json.loads(json.dumps(CONF))
    conf["euclid"]["fields"] = {"T": field}
    conf["euclid"]["strip_deg"] = 0.4
    conf["services"]["irsa_gaia_tables"] = ["gaia_dr3_source"]
    if boxes is not None:
        conf["spherex"]["boxes"] = boxes
    return conf


def test_euclid_end_to_end_on_a_fake_archive(tmp_path):
    rng = np.random.default_rng(11)
    field, gaia, strip, spec = _euclid_world(rng)
    conf = _conf_for(field)
    irsa = FakeIRSA(strip, spec, gaia)
    out = tmp_path / "spark"
    p = R.probe(conf, out, irsa_query=irsa, skip_spherex=True)
    assert p["euclid"]["roles_line"]["wl"] == "spe_line_central_wl_gf" and p["euclid"]["missing_mer_roles"] == []
    assert p["euclid"]["join_top5"]["wavelength"]["wavelength_scale"] == 1e-4
    assert p["gaia"]["route_recommended"] == "irsa:gaia_dr3_source"
    led = R.euclid_stage(conf, out, shard=0, n_shards=1, irsa_query=irsa)
    assert led["status"] == "OK" and led["n_units"] == 3 and not led["degraded"]
    assert sum(u["n_stars_with_spectra"] for u in led["units"]) == 23
    s = R.assess(conf, out)
    eu = s["euclid"]
    assert eu["n_stars_with_spectra"] == 23 and eu["n_stars_with_features"] == 3
    assert eu["n_survivors"] == 1 and eu["survivors"][0]["gaia_source_id"] == 1001
    assert eu["veto_counts"]["stellar_line"] >= 1 and eu["veto_counts"]["galaxy_pattern"] == 3
    assert s["verdict"] == R.VERDICT_CANDIDATES
    c = pd.read_csv(out / "candidates.csv")
    assert len(c) == 1 and c["industrial_flag"].iloc[0] == "Er fibre C-band"
    # resume: a second call finds the checkpoints and re-queries nothing
    n_q = len(irsa.queries)
    led2 = R.euclid_stage(conf, out, shard=0, n_shards=1, irsa_query=irsa)
    assert all(u.get("resumed") for u in led2["units"])
    assert not any(" l JOIN " in q for q in irsa.queries[n_q:])       # only schema discovery, no strip re-pulled
    # the CLI screen stage re-runs over the local files
    r = R.spark_run("screen", out, config=conf, offline=True)
    assert r["euclid_screen"]["n_survivors"] == 1


def test_euclid_empty_and_failed_archives_are_not_null_results(tmp_path):
    rng = np.random.default_rng(12)
    field, gaia, strip, spec = _euclid_world(rng)
    conf = _conf_for(field)
    out = tmp_path / "e"
    led = R.euclid_stage(conf, out, irsa_query=FakeIRSA(strip.head(0), spec.head(0), gaia, empty=True))
    assert led["status"] == R.VERDICT_NO_DATA
    s = R.assess(conf, out)
    assert s["euclid"]["verdict"] == R.VERDICT_NO_DATA and s["verdict"] == R.VERDICT_NO_DATA
    out2 = tmp_path / "f"
    led2 = R.euclid_stage(conf, out2, irsa_query=FakeIRSA(strip, spec, gaia, fail_lines=True))
    assert led2["status"] == R.VERDICT_NO_DATA and any("failed" in x for x in led2["degraded"])
    s2 = R.assess(conf, out2)
    assert s2["verdict"].startswith("DEGRADED") or s2["verdict"] == R.VERDICT_NO_DATA
    assert s2["stage_counts"]["euclid_survivors"] == 0


class FakeSpherexArchive:
    """Synthetic level-2 products: a small LVF detector, a TAN WCS, a ``WCS-WAVE`` lookup, IBE cutouts."""

    NX = NY = 360
    PIX = 6.15

    def __init__(self, rng, stars: pd.DataFrame, box: dict, *, n_per_pass=70, inject_sid=None,
                 inject_bin=None, inject_frac=0.35, detectors=("D2",), bands=None):
        self.rng, self.stars, self.box = rng, stars, box
        self.inject_sid, self.inject_bin, self.inject_frac = inject_sid, inject_bin, inject_frac
        self.bands = bands or SC["detector_bands_um"]
        rows, self.pointings = [], {}
        k = 0
        for det in detectors:
            for mjd0 in (0.0, 200.0):
                for _ in range(n_per_pass):
                    oid = f"obs{k:04d}"
                    # the box centre lands at a random place on the detector (dithers walk the LVF)
                    xc, yc = rng.uniform(80, self.NX - 80), rng.uniform(40, self.NY - 40)
                    mjd = mjd0 + rng.uniform(0, 5)
                    self.pointings[oid] = (det, xc, yc, mjd)
                    rows.append({"obs_id": oid, "obs_collection": "spherex_qr2", "dataproduct_type": "image",
                                 "calib_level": 2, "s_ra": box["ra"], "s_dec": box["dec"], "t_min": mjd, "t_max": mjd + 0.01,
                                 "em_min": self.bands[det][0] * 1e-6, "em_max": self.bands[det][1] * 1e-6,
                                 "access_url": f"ibe/data/spherex/qr2/level2/x/level2_{oid}_1{det}_spx.fits",
                                 "access_format": "application/fits", "instrument_name": f"SPHEREx-{det}"})
                    k += 1
        self.products = pd.DataFrame(rows)
        self.n_full = 0
        self.n_cut = 0

    def wavemap(self, det):
        lo, hi = self.bands[det]
        return S.WaveMap.synthetic(self.NX, self.NY, lo, hi, R=40.0, axis="y")

    def _render(self, oid):
        from astropy.io import fits
        from astropy.wcs import WCS
        det, xc, yc, mjd = self.pointings[oid]
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crpix = [xc + 1, yc + 1]
        w.wcs.crval = [self.box["ra"], self.box["dec"]]
        w.wcs.cdelt = [-self.PIX / 3600, self.PIX / 3600]
        hdr = w.to_header()
        img = np.full((self.NY, self.NX), 2.0)
        wm = self.wavemap(det)
        omega = (self.PIX / 206264.806) ** 2
        xs, ys = w.all_world2pix(self.stars["ra"].to_numpy(), self.stars["dec"].to_numpy(), 0)
        for x, y, sid in zip(xs, ys, self.stars["source_id"], strict=True):
            if not (5 < x < self.NX - 5 and 5 < y < self.NY - 5):
                continue
            lam, _ = wm.wavelength_at([x], [y])
            f_mjy = 120.0 * (1.3 / lam[0]) ** 2
            if sid == self.inject_sid and int(math.floor(math.log(lam[0]) * 40)) == self.inject_bin:
                f_mjy *= 1 + self.inject_frac
            img += _gauss_star(self.NY, self.NX, x, y, f_mjy / (omega * 1e9))
        noise = 0.02
        img += self.rng.normal(0, noise, img.shape)
        var = np.full(img.shape, noise ** 2)
        flags = np.zeros(img.shape, np.int32)
        zodi = np.full(img.shape, 1.5)
        hdr["MJD-AVG"] = mjd
        hdr["BUNIT"] = "MJy/sr"
        hdr["CTYPE3"] = "WAVE-TAB"
        hdr["PS3_0"], hdr["PS3_1"] = "WCS-WAVE", "WAVELENGTH"
        prim = fits.PrimaryHDU()
        im = fits.ImageHDU(img.astype(np.float32), header=hdr, name="IMAGE")
        fl = fits.ImageHDU(flags, name="FLAGS")
        va = fits.ImageHDU(var.astype(np.float32), header=hdr, name="VARIANCE")
        zo = fits.ImageHDU(zodi.astype(np.float32), name="ZODI")
        wave = wm.wave.reshape(1, self.NY, self.NX).astype(np.float32)
        bw = wm.bw.reshape(1, self.NY, self.NX).astype(np.float32)
        tab = fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format=f"{self.NX * self.NY}E", dim=f"({self.NX},{self.NY})", array=wave),
            fits.Column(name="BANDWIDTH", format=f"{self.NX * self.NY}E", dim=f"({self.NX},{self.NY})", array=bw)],
            name="WCS-WAVE")
        return fits.HDUList([prim, im, fl, va, zo, tab])

    def fetch(self, url: str):
        from astropy.io import fits
        oid = re.search(r"level2_(obs\d+)_", url).group(1)
        full = self._render(oid)
        if "?center=" not in url:
            self.n_full += 1
            return full
        self.n_cut += 1
        m = re.search(r"center=([-\d.]+),([-\d.]+)&size=([\d.]+)arcsec", url)
        ra, dec, size = float(m.group(1)), float(m.group(2)), float(m.group(3))
        half = int(round(size / self.PIX / 2))
        w = S.celestial_wcs(full["IMAGE"].header)
        x, y = w.all_world2pix([ra], [dec], 0)
        x0, y0 = int(round(x[0])) - half, int(round(y[0])) - half
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x0 + 2 * half, self.NX), min(y0 + 2 * half, self.NY)
        out = [fits.PrimaryHDU()]
        for name in ("IMAGE", "FLAGS", "VARIANCE", "ZODI"):
            h = full[name]
            hdr = h.header.copy()
            hdr["CRPIX1"] = float(hdr.get("CRPIX1", 0)) - x0
            hdr["CRPIX2"] = float(hdr.get("CRPIX2", 0)) - y0
            hdr["LTV1"], hdr["LTV2"] = -float(x0), -float(y0)
            out.append(fits.ImageHDU(h.data[y0:y1, x0:x1], header=hdr, name=name))
        return fits.HDUList(out)              # the cutout does NOT carry WCS-WAVE


def _spherex_world(rng, box, n_seed=6, inject=True):
    stars = []
    for k in range(n_seed):
        stars.append(_star(5000 + k, box["ra"] + rng.uniform(-5, 5) / 60 / math.cos(math.radians(box["dec"])),
                           box["dec"] + rng.uniform(-5, 5) / 60, g=11.0 + k * 0.5))
    # faint field stars (isolation test: none close enough to matter)
    for k in range(30):
        stars.append(_star(6000 + k, box["ra"] + rng.uniform(-9, 9) / 60 / math.cos(math.radians(box["dec"])),
                           box["dec"] + rng.uniform(-9, 9) / 60, g=17.5 + rng.uniform(0, 1.4)))
    gaia = _gaia(stars)
    # 1.437 um: a channel the stellar-line veto leaves available at R = 40.
    # (1.30 um is within one resolution element of [Fe II] 1.295 and Pa-beta,
    # so an injection there is correctly vetoed — see the veto tests.)
    bin_ = int(math.floor(math.log(1.437) * 40))
    arch = FakeSpherexArchive(rng, gaia[gaia["source_id"] < 6000], box, inject_sid=5000 if inject else None,
                              inject_bin=bin_)
    return gaia, arch, bin_


def test_spherex_end_to_end_recovers_an_injected_channel(tmp_path):
    rng = np.random.default_rng(21)
    box = {"name": "T-0", "ra": 270.0, "dec": 66.56}
    gaia, arch, bin_ = _spherex_world(rng, box)
    conf = _conf_for({"ra": 100.0, "dec": 10.0, "radius_deg": 0.5}, boxes=[box])
    conf["spherex"]["box_arcmin"] = 12.0
    conf["spherex"]["max_images_per_detector"] = 140
    irsa = FakeIRSA(pd.DataFrame(), pd.DataFrame(), gaia, products=arch.products)
    out = tmp_path / "spark"
    probe_rec = {"gaia": {"irsa_gaia_table_usable": ["gaia_dr3_source"]}, "spherex": {"obscore_missing_columns": []}}
    R._write_json(out / "probe.json", probe_rec)
    led = R.spherex_stage(conf, out, shard=0, n_shards=1, irsa_query=irsa, esa_query=None, fetch_fn=arch.fetch)
    assert led["status"] == "OK", led
    b = led["boxes"][0]
    assert b["seeds"]["n_seeds"] == 6 and b["cutouts"]["n_images_ok"] == 140 and b["cutouts"]["n_images_failed"] == 0
    assert b["cutouts"]["route_counts"] == {"ibe_cutout": 140} and arch.n_full == 1     # one full frame per detector for the map
    assert list(b["cutouts"]["wavemap_methods"].values()) == [140]        # one method served every cutout
    s = R.assess(conf, out)
    sx = s["spherex"]
    assert sx["n_stars_with_samples"] == 6 and sx["n_survivors"] == 1, sx
    surv = sx["survivors"][0]
    assert surv["source_id"] == 5000 and surv["channel_bin"] == bin_ and surv["detector"] == "D2"
    assert abs(surv["excess_fraction"] - 0.35) < 0.12 and surv["n_passes"] == 2
    assert s["verdict"] == R.VERDICT_CANDIDATES
    # resume: nothing re-fetched
    n_cut = arch.n_cut
    led2 = R.spherex_stage(conf, out, shard=0, n_shards=1, irsa_query=irsa, fetch_fn=arch.fetch)
    assert arch.n_cut == n_cut and led2["boxes"][0]["cutouts"]["n_images_tried"] == 0


def test_spherex_null_world_and_empty_obscore(tmp_path):
    rng = np.random.default_rng(22)
    box = {"name": "T-1", "ra": 90.0, "dec": -66.56}
    gaia, arch, _ = _spherex_world(rng, box, n_seed=3, inject=False)
    conf = _conf_for({"ra": 100.0, "dec": 10.0, "radius_deg": 0.5}, boxes=[box])
    conf["spherex"]["box_arcmin"] = 12.0
    conf["spherex"]["max_images_per_detector"] = 120
    out = tmp_path / "n"
    R._write_json(out / "probe.json", {"gaia": {"irsa_gaia_table_usable": ["gaia_dr3_source"]}})
    irsa = FakeIRSA(pd.DataFrame(), pd.DataFrame(), gaia, products=arch.products)
    led = R.spherex_stage(conf, out, irsa_query=irsa, fetch_fn=arch.fetch)
    s = R.assess(conf, out)
    assert led["status"] == "OK" and s["spherex"]["n_survivors"] == 0 and s["spherex"]["n_stars_with_samples"] == 3
    assert s["verdict"] == R.VERDICT_NONE
    out2 = tmp_path / "e"
    R._write_json(out2 / "probe.json", {"gaia": {"irsa_gaia_table_usable": ["gaia_dr3_source"]}})
    led2 = R.spherex_stage(conf, out2, irsa_query=FakeIRSA(pd.DataFrame(), pd.DataFrame(), gaia, products=None),
                           fetch_fn=arch.fetch)
    assert led2["status"] == R.VERDICT_NO_DATA and any("no products" in x for x in led2["degraded"])
    assert R.assess(conf, out2)["spherex"]["verdict"] == R.VERDICT_NO_DATA


def test_probe_on_the_fake_spherex_archive_describes_products(tmp_path):
    rng = np.random.default_rng(23)
    box = {"name": "T-2", "ra": 270.0, "dec": 66.56}
    gaia, arch, _ = _spherex_world(rng, box, n_seed=2, inject=False)
    field = {"ra": 100.0, "dec": 10.0, "radius_deg": 0.5}
    conf = _conf_for(field, boxes=[box])
    _, g2, strip, spec = _euclid_world(rng, n_bg=3)
    irsa = FakeIRSA(strip, spec, pd.concat([gaia, g2], ignore_index=True), products=arch.products)
    p = R.probe(conf, tmp_path, irsa_query=irsa, fetch_fn=arch.fetch, http_get=lambda u: "<html>cutout</html>")
    sx = p["spherex"]
    assert sx["obscore_count_box0"]["n"] == len(arch.products)
    assert sx["full_product"]["wavemap"]["status"] == "OK" and sx["full_product"]["wavemap"]["fingerprint"]["method"] in ("astropy_tab", "manual_tab")
    assert sx["cutout"]["hdus"][1]["name"] == "IMAGE" and sx["cutout"]["offset_from_ltv_or_crpix"] is not None
    assert sx["cutout"]["wavemap"]["status"] == "FAILED"          # the fake cutout drops WCS-WAVE, and the probe says so
    assert sx["second_product_same_detector"]["wavemap"]["max_abs_diff_um_vs_first"] == 0.0
    assert sx["cutout_tool_page"]["text_head"].startswith("<html>")
    assert Path(tmp_path / "probe.json").exists()


def test_cli_entry_offline(tmp_path, capsys):
    rc = R.main(["--stage", "assess", "--out-dir", str(tmp_path / "o"), "--offline"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NO_DATA_REACHED" in out


def test_fits_roundtrip_of_the_fake_cutout_is_readable_by_the_pipeline():
    """The fake archive's products go through astropy's writer/reader like a real download."""
    from astropy.io import fits
    rng = np.random.default_rng(24)
    box = {"name": "T-3", "ra": 270.0, "dec": 66.56}
    gaia, arch, _ = _spherex_world(rng, box, n_seed=2, inject=False)
    hd = arch.fetch(S.product_url("https://irsa.ipac.caltech.edu/", arch.products["access_url"].iloc[0]))
    buf = io.BytesIO()
    hd.writeto(buf)
    buf.seek(0)
    hd2 = fits.open(buf)
    wm = S.WaveMap.from_hdul(hd2)
    assert wm.fingerprint()["gradient_axis"] == "y"
    # both decoders agree with the model the file was rendered from
    img = S._find_image_hdu(hd2)
    xs = ys = np.arange(0, arch.NX, 8.0)
    truth = arch.wavemap("D2")
    for wmx in (wm, S.WaveMap._from_table(hd2, img, xs, ys, (arch.NX, arch.NY))):
        lam, bw = wmx.wavelength_at([100.0, 250.0], [200.0, 50.0])
        lt, bt = truth.wavelength_at([100.0, 250.0], [200.0, 50.0])
        assert np.allclose(lam, lt, atol=2e-4) and np.allclose(bw, bt, atol=2e-4), (wmx.method, lam, lt)
    # astropy's decode of this synthetic table is degenerate (one wavelength everywhere) and is REFUSED
    with pytest.raises(ValueError, match="degenerate"):
        S.WaveMap._from_astropy(hd2, img, xs, ys, (arch.NX, arch.NY))
    assert wm.method == "manual_tab"
    planes = S.image_planes(hd2)
    assert planes["variance"] is not None and planes["flags"] is not None and planes["zodi"] is not None


# --------------------------------------------------------------------------
# The survivor vet (the slitless neighbour, from Euclid's own catalogues)
# --------------------------------------------------------------------------
def _mer_neighbour(oid, ra, dec, flux_h=1.0):
    return {"m_object_id": oid, "m_ra": ra, "m_dec": dec, "m_flux_h": flux_h,
            "m_point_like_prob": 0.2, "m_spurious_flag": 0}


def test_classify_neighbours_separates_corridor_blend_and_redshift():
    conf = dict(R.DEFAULTS["euclid"])
    sv = {"object_id": 7, "ra": 100.0, "dec": 10.0, "wl_um": 1.5500}
    arcsec = 1.0 / 3600.0
    cosd = math.cos(math.radians(10.0))
    nbrs = pd.DataFrame([
        _mer_neighbour(7, 100.0, 10.0, 10.0),                              # the star itself
        _mer_neighbour(8, 100.0 + 60 * arcsec / cosd, 10.0, 4.0),          # in the dispersion corridor
        _mer_neighbour(9, 100.0, 10.0 + 60 * arcsec, 8.0),                 # across dispersion: not a neighbour
        _mer_neighbour(10, 100.0 + 3 * arcsec / cosd, 10.0 + 2 * arcsec, 0.05),  # inside 6", far too faint
    ])
    out = E.classify_neighbours(sv, nbrs, conf)
    # object 10 sits inside both the 6" blend radius and the corridor; object 9 is across dispersion
    assert out["n_in_corridor"] == 2 and out["n_in_blend_radius"] == 1
    assert out["neighbour_trace_overlap"] is True                          # object 8 carries 40 % of the star's H flux
    assert [o["object_id"] for o in out["offenders"]] == [8]
    # the faint one alone is not an overlap
    out2 = E.classify_neighbours(sv, nbrs.drop(index=1).reset_index(drop=True), conf)
    assert out2["neighbour_trace_overlap"] is False and out2["n_in_blend_radius"] == 1
    # a corridor neighbour at z = 1.3617 puts H-alpha at 1.55 um: that is the feature
    z = 1.5500 / 0.656461 - 1.0
    zr = pd.DataFrame([{"s_object_id": 8, "s_spe_z": z}, {"s_object_id": 99, "s_spe_z": z}])
    out3 = E.classify_neighbours(sv, nbrs, conf, z_rows=zr)
    assert out3["neighbour_line_at_z"] is True
    assert [m["object_id"] for m in out3["z_matches"]] == [8]               # 99 is not a neighbour
    assert out3["z_matches"][0]["line"] == "H-alpha"
    # a neighbour carrying a feature at the same observed wavelength
    lr = pd.DataFrame([{"l_object_id": 8, "l_wl": 1.5502}])
    out4 = E.classify_neighbours(sv, nbrs, conf, line_rows=lr)
    assert out4["neighbour_line_at_z"] is True and "same_wavelength_feature_um" in out4["z_matches"][0]


class FakeVetIRSA:
    """Serves the three vet queries; everything else is an error."""

    def __init__(self, nbrs: pd.DataFrame, z_rows: pd.DataFrame, line_rows: pd.DataFrame):
        self.nbrs, self.z_rows, self.line_rows = nbrs, z_rows, line_rows
        self.queries: list[str] = []

    def __call__(self, adql: str, maxrec=None):
        self.queries.append(adql)
        a = adql.lower()
        if "tap_schema.columns" in a:
            t = re.search(r"table_name = '([^']+)'", adql).group(1)
            cols = {"euclid_q1_spectro_zcatalog_spe_classification": ["object_id", "spe_z", "spe_class"]}.get(t, [])
            return pd.DataFrame({"column_name": cols, "datatype": ["x"] * len(cols)})
        if "from euclid_q1_mer_catalogue m" in a:
            m = re.search(r"BETWEEN ([-\d.]+) AND ([-\d.]+)", adql)
            lo, hi = float(m.group(1)), float(m.group(2))
            return self.nbrs[(self.nbrs["m_dec"] >= lo) & (self.nbrs["m_dec"] <= hi)].reset_index(drop=True)
        if "from euclid_q1_spectro_zcatalog_spe_classification s" in a:
            ids = {int(x) for x in re.findall(r"\d+", adql.split("IN (")[1])}
            return self.z_rows[self.z_rows["s_object_id"].isin(ids)].reset_index(drop=True)
        if "from euclid_q1_spe_lines_line_features l" in a:
            ids = {int(x) for x in re.findall(r"\d+", adql.split("IN (")[1])}
            return self.line_rows[self.line_rows["l_object_id"].isin(ids)].reset_index(drop=True)
        raise AssertionError(f"unexpected vet ADQL: {adql[:200]}")


def _vet_setup(tmp_path):
    out = tmp_path / "spark"
    (out / "euclid").mkdir(parents=True)
    R._write_json(out / "probe.json", {"euclid": {
        "roles_mer": {"object_id": "object_id", "ra": "right_ascension", "dec": "declination",
                      "flux_h": "flux_h_2fwhm_aper"},
        "roles_line": {"object_id": "object_id", "wl": "spe_line_central_wl_gf", "snr": "spe_line_snr_gf"},
        "spectra_table": "euclid_q1_spectro_zcatalog_spe_classification"}})
    feats = pd.DataFrame([
        {"object_id": 7, "gaia_source_id": 1007, "field": "F", "ra": 100.0, "dec": 10.0,
         "wl_um": 1.55, "snr": 11.0, "survivor": True, "low_snr": False},
        {"object_id": 11, "gaia_source_id": 1011, "field": "F", "ra": 101.0, "dec": 10.0,
         "wl_um": 1.61, "snr": 9.0, "survivor": True, "low_snr": False},
        {"object_id": 12, "gaia_source_id": 1012, "field": "F", "ra": 102.0, "dec": 10.0,
         "wl_um": 1.40, "snr": 4.0, "survivor": False, "low_snr": True}])
    feats.to_csv(out / "euclid" / "features_all.csv", index=False)
    return out


def test_vet_euclid_kills_the_trace_overlap_and_keeps_the_clean_survivor(tmp_path):
    out = _vet_setup(tmp_path)
    arcsec = 1.0 / 3600.0
    cosd = math.cos(math.radians(10.0))
    nbrs = pd.DataFrame([
        _mer_neighbour(7, 100.0, 10.0, 10.0), _mer_neighbour(8, 100.0 + 50 * arcsec / cosd, 10.0, 5.0),
        _mer_neighbour(11, 101.0, 10.0, 10.0), _mer_neighbour(13, 101.0, 10.0 + 400 * arcsec, 9.0)])
    irsa = FakeVetIRSA(nbrs, pd.DataFrame(columns=["s_object_id", "s_spe_z"]),
                       pd.DataFrame(columns=["l_object_id", "l_wl"]))
    conf = R.load_spark_config()
    rec = R.vet_euclid(conf, out, irsa_query=irsa)
    assert rec["status"] == "OK" and rec["n_vetted"] == 2
    assert rec["n_trace_overlap"] == 1 and rec["n_clean_after_vet"] == 1
    got = {r["object_id"]: r["neighbour_trace_overlap"] for r in rec["survivors"]}
    assert got == {7: True, 11: False}
    d = pd.read_csv(out / "euclid" / "features_all.csv")
    assert list(d.sort_values("object_id")["survivor_after_vet"]) == [False, True, False]


def test_vet_euclid_is_honest_when_it_cannot_run(tmp_path):
    out = _vet_setup(tmp_path)
    rec = R.vet_euclid(R.load_spark_config(), out, irsa_query=None)
    assert rec["status"] == R.VERDICT_NO_DATA and rec["degraded"]
    empty = tmp_path / "empty"
    (empty / "euclid").mkdir(parents=True)
    assert R.vet_euclid(R.load_spark_config(), empty, irsa_query=None)["status"] == "NO_FEATURE_TABLE"

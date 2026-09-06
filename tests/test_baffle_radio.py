"""BAFFLE / radio: the LoTSS void statistic, offline.

Every archive call is injected (``lotss_fetcher`` / ``target_fetcher``); the
conftest guard raises on any socket, so a test that forgot to stub one fails
here rather than only on the runner.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from seti.baffle import radio as R

# A field far from any real LoTSS coordinates matters not at all offline; this
# one sits inside the coarse 13h region so the region pre-cut keeps it.
FIELD_RA, FIELD_DEC = 200.0, 50.0


def _cfg(**over) -> dict:
    cfg = R.load_radio_config()
    cfg.update(over)
    return cfg


def uniform_field(density_per_deg2: float, ra0: float, dec0: float, half_ra: float,
                  half_dec: float, seed: int = 11, hole_arcsec: float | None = None,
                  hole_at: tuple[float, float] | None = None) -> pd.DataFrame:
    """A Poisson-uniform sky patch (uniform in RA and in sin Dec), optionally
    with every source inside ``hole_arcsec`` of ``hole_at`` removed."""
    rng = np.random.default_rng(seed)
    s_lo, s_hi = np.sin(np.radians(dec0 - half_dec)), np.sin(np.radians(dec0 + half_dec))
    area = (2 * half_ra) * np.degrees(s_hi - s_lo)
    n = rng.poisson(density_per_deg2 * area)
    ra = rng.uniform(ra0 - half_ra, ra0 + half_ra, n)
    dec = np.degrees(np.arcsin(rng.uniform(s_lo, s_hi, n)))
    flux = rng.lognormal(mean=np.log(2e-3), sigma=1.0, size=n)      # Jy, faint
    df = pd.DataFrame({"ra": ra, "dec": dec, "flux_jy": flux})
    if hole_arcsec:
        hra, hdec = hole_at if hole_at else (ra0, dec0)
        sep = R.angular_separation_arcsec(hra, hdec, df["ra"].to_numpy(), df["dec"].to_numpy())
        df = df.loc[sep > hole_arcsec].reset_index(drop=True)
    return df


def _targets(ra=FIELD_RA, dec=FIELD_DEC, source_id=1) -> pd.DataFrame:
    return pd.DataFrame({"source_id": [source_id], "ra": [ra], "dec": [dec],
                         "parallax": [50.0], "is_etz": [False]})


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_obs,lam", [(0, 8.0), (3, 20.2), (11, 19.6), (0, 0.5), (25, 25.0)])
def test_poisson_void_p_matches_scipy(n_obs, lam):
    assert R.poisson_void_p(n_obs, lam) == pytest.approx(poisson.cdf(n_obs, lam), rel=1e-10)


def test_poisson_void_p_is_vectorised_and_handles_zero_expectation():
    p = R.poisson_void_p([0, 1, 2], [0.0, 1.0, 1.0])
    assert p[0] == 1.0
    assert p[1] == pytest.approx(poisson.cdf(1, 1.0))
    assert p[2] == pytest.approx(poisson.cdf(2, 1.0))


@pytest.mark.parametrize("d_au", [500.0, 1000.0, 10000.0])
@pytest.mark.parametrize("ra,dec", [(200.0, 50.0), (0.0, 0.0), (90.0, 66.56), (270.0, -23.4)])
def test_annual_ellipse_semi_major_axis_is_1au_over_d(ra, dec, d_au):
    """The projected orbit's semi-major axis is 1 AU / d for every star (only
    the semi-minor axis depends on ecliptic latitude).  Measured along the
    principal axis of a dense sampling so Earth's 1.7 % eccentricity (which
    shifts the ellipse's centre, not its size) does not enter."""
    g = R.baffle_centre_grid(ra, dec, d_au, n_phase=360)
    xy = g[["dx_arcsec", "dy_arcsec"]].to_numpy()
    xy0 = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy0, full_matrices=False)
    proj = xy0 @ vt[0]
    semi_major = 0.5 * (proj.max() - proj.min())
    assert semi_major == pytest.approx(R.AU_ARCSEC / d_au, rel=0.01)
    # The 8-phase grid used in the screen is a subset of the same ellipse.
    g8 = R.baffle_centre_grid(ra, dec, d_au, n_phase=8)
    assert len(g8) == 8
    assert np.all(np.hypot(g8["dx_arcsec"], g8["dy_arcsec"]) <= 1.02 * R.AU_ARCSEC / d_au)


def test_annual_ellipse_semi_minor_axis_scales_with_ecliptic_latitude():
    pole = R.baffle_centre_grid(270.0, 66.56, 1000.0, n_phase=360)    # ~ north ecliptic pole
    plane = R.baffle_centre_grid(0.0, 0.0, 1000.0, n_phase=360)       # in the ecliptic
    r_pole = np.hypot(pole["dx_arcsec"], pole["dy_arcsec"])
    r_plane = np.hypot(plane["dx_arcsec"], plane["dy_arcsec"])
    assert r_pole.min() > 0.95 * R.AU_ARCSEC / 1000.0       # circle
    assert r_plane.min() < 0.05 * R.AU_ARCSEC / 1000.0      # line


def test_annulus_density_and_aperture_count_on_a_uniform_field():
    df = uniform_field(900.0, FIELD_RA, FIELD_DEC, 1.0, 0.6, seed=3)
    dens = R.annulus_density(df, FIELD_RA, FIELD_DEC, 480.0, 1200.0)
    assert dens == pytest.approx(900.0, rel=0.15)
    n = R.count_in_aperture(df, FIELD_RA, FIELD_DEC, 600.0)
    assert n == pytest.approx(900.0 * R.aperture_area_deg2(600.0), abs=4 * np.sqrt(79.0))
    assert R.count_in_aperture(df.iloc[:0], FIELD_RA, FIELD_DEC, 600.0) == 0
    assert R.annulus_density(None, FIELD_RA, FIELD_DEC, 480.0, 1200.0) == 0.0


def test_offset_position_round_trips_through_separation():
    ra, dec = R.offset_position(200.0, 50.0, 2700.0, 0.0)
    assert R.angular_separation_arcsec(200.0, 50.0, ra, dec) == pytest.approx(2700.0, rel=1e-3)
    ra, dec = R.offset_position(200.0, 50.0, 0.0, -2700.0)
    assert R.angular_separation_arcsec(200.0, 50.0, ra, dec) == pytest.approx(2700.0, rel=1e-6)


def test_ecliptic_latitude_flags_the_earth_transit_zone():
    lat = R.ecliptic_latitude_deg([0.0, 90.0, 270.0], [0.0, 23.44, 66.56])
    assert abs(lat[0]) < 0.01
    assert abs(lat[1]) < 0.05
    assert lat[2] > 89.0


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------
def test_a_hole_in_a_dense_field_is_a_candidate_at_the_matching_aperture():
    """A 210" hole at 2000 / deg^2: the 204" aperture expects ~20 and sees 0,
    p_raw ~ 2e-9, p_trials ~ 4e-7 < 1e-5.  (At 900 / deg^2 the same hole
    expects only 9 and is reported not_significant -- see the module
    docstring's sensitivity floor.)"""
    cfg = _cfg()
    df = uniform_field(2000.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=5, hole_arcsec=210.0)
    st = R.void_statistics(df, FIELD_RA, FIELD_DEC, cfg)
    assert st["veto"] == R.VETO_NONE
    assert st["best_aperture_arcsec"] == 204.0
    assert st["best_n_obs"] == 0
    assert st["best_lambda"] > 15
    assert st["p_trials"] < 1e-5
    assert st["p_raw"] < 1e-5
    # 41 centres x the apertures that expect >= 8 sources at 2000 / deg^2
    # (204", 300", 600"; the 120" aperture expects 7.0).
    n_usable = sum(2000.0 * R.aperture_area_deg2(r) >= 8.0 for r in cfg["apertures_arcsec"])
    assert n_usable == 3
    assert st["n_trials"] == (1 + 5 * 8) * n_usable
    assert st["best_dx_arcsec"] == 0.0 and st["best_dy_arcsec"] == 0.0     # centred on X


def test_the_same_field_without_the_hole_is_not_significant():
    cfg = _cfg()
    df = uniform_field(2000.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=5)
    st = R.void_statistics(df, FIELD_RA, FIELD_DEC, cfg)
    assert st["veto"] == R.VETO_NOT_SIGNIFICANT
    assert st["p_trials"] > 1e-3


def test_a_900_per_deg2_field_with_a_1au_at_1000au_hole_is_below_the_floor():
    """The honest number: a 200" hole in LoTSS-depth counts is a ~1e-4 event
    raw, so it is recorded but not a candidate."""
    cfg = _cfg()
    df = uniform_field(900.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=9, hole_arcsec=200.0)
    st = R.void_statistics(df, FIELD_RA, FIELD_DEC, cfg)
    assert st["veto"] == R.VETO_NOT_SIGNIFICANT
    assert st["best_n_obs"] <= 1
    assert 1e-6 < st["p_raw"] < 1e-2


def test_a_sparse_field_is_low_expected_count():
    """20 / deg^2 expects 1.7 in even the 600" aperture.  The footprint veto
    (300 / deg^2) fires first with the default order, so it is lowered to
    isolate this veto."""
    cfg = _cfg(min_annulus_density_per_deg2=5.0)
    df = uniform_field(20.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=1)
    st = R.void_statistics(df, FIELD_RA, FIELD_DEC, cfg)
    assert st["veto"] == R.VETO_LOW_EXPECTED
    assert st["n_trials"] == 0


def test_an_empty_annulus_is_outside_the_footprint():
    cfg = _cfg()
    empty = pd.DataFrame({"ra": [], "dec": [], "flux_jy": []})
    st = R.void_statistics(empty, FIELD_RA, FIELD_DEC, cfg)
    assert st["veto"] == R.VETO_FOOTPRINT
    assert st["annulus_density_per_deg2"] == 0.0
    # With the defaults, 20 / deg^2 is also below the footprint floor.
    sparse = uniform_field(20.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=1)
    assert R.void_statistics(sparse, FIELD_RA, FIELD_DEC, cfg)["veto"] == R.VETO_FOOTPRINT


def test_a_bright_source_nearby_masks_the_position():
    cfg = _cfg()
    df = uniform_field(2000.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=5, hole_arcsec=210.0)
    ra_b, dec_b = R.offset_position(FIELD_RA, FIELD_DEC, 1200.0, 900.0)     # 25' away
    df = pd.concat([df, pd.DataFrame({"ra": [ra_b], "dec": [dec_b], "flux_jy": [3.0]})],
                   ignore_index=True)
    st = R.void_statistics(df, FIELD_RA, FIELD_DEC, cfg)
    assert st["veto"] == R.VETO_BRIGHT
    assert st["bright_source_max_jy"] == pytest.approx(3.0)


def test_screen_targets_finds_the_hole_and_the_controls_do_not_fire():
    cfg = _cfg()
    field = uniform_field(2000.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=5, hole_arcsec=210.0)
    tiles = R.bin_sources_into_tiles(field, cfg["tile_deg"])
    assert len(tiles) >= 4
    voids, counters = R.screen_targets(_targets(), tiles, cfg)
    assert len(voids) == 1
    row = voids.iloc[0]
    assert bool(row["is_candidate"])
    assert row["veto"] == R.VETO_NONE
    assert counters["n_candidates"] == 1
    assert counters["n_targets_in_footprint"] == 1
    # Four controls 45' away, all evaluated, none a void.
    assert row["n_control_evaluated"] == 4
    assert row["n_control_fired"] == 0
    for name in ("+ra", "-ra", "+dec", "-dec"):
        assert row[f"control_{name}_veto"] == R.VETO_NOT_SIGNIFICANT
        assert row[f"control_{name}_p_trials"] > 1e-3
    assert counters["n_control_evaluated"] == 4
    assert counters["n_control_fired"] == 0
    assert counters["control_false_void_rate"] == 0.0
    assert row["p_empirical_control"] < 0.5


def test_screen_targets_marks_a_star_outside_the_footprint():
    cfg = _cfg()
    field = uniform_field(900.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=5)
    tiles = R.bin_sources_into_tiles(field, cfg["tile_deg"])
    targets = pd.concat([_targets(), _targets(ra=120.0, dec=40.0, source_id=2)],
                        ignore_index=True)
    voids, counters = R.screen_targets(targets, tiles, cfg)
    assert list(voids["veto"]) == [R.VETO_NOT_SIGNIFICANT, R.VETO_FOOTPRINT]
    assert counters[R.VETO_FOOTPRINT] == 1
    assert counters["n_targets_in_footprint"] == 1
    assert counters["n_candidates"] == 0


def test_plan_tiles_covers_every_target_and_its_neighbourhood():
    rng = np.random.default_rng(2)
    targets = pd.DataFrame({"ra": rng.uniform(0, 360, 300), "dec": rng.uniform(-80, 80, 300)})
    targets.loc[0, ["ra", "dec"]] = [359.9, 30.0]      # RA wrap
    tiles = R.plan_tiles(targets, 2.0)
    keys = {t["key"] for t in tiles}
    for ra, dec in zip(targets["ra"], targets["dec"], strict=True):
        ix, iy = R.tile_index(ra, dec, 2.0)
        assert R.tile_key(int(ix), int(iy)) in keys
        t = R.tile_bounds(int(ix), int(iy), 2.0)
        assert t["ra_min"] <= ra < t["ra_max"] and t["dec_min"] <= dec < t["dec_max"]
    assert sum(t["n_targets"] for t in tiles) == len(targets)
    padded = R.plan_tiles(targets, 2.0, pad_deg=1.25)
    assert {t["key"] for t in padded} >= keys
    assert len(padded) > len(tiles)
    # The tile the star at RA 359.9 sits in and the one across the wrap.
    pkeys = {t["key"] for t in padded}
    assert R.tile_key(179, 60) in pkeys and R.tile_key(0, 60) in pkeys


def test_in_regions_wraps_through_ra_zero():
    regs = R.DEFAULTS["targets"]["regions"]
    ok = R.in_regions([200.0, 350.0, 10.0, 60.0, 200.0], [50.0, 30.0, 30.0, 30.0, 5.0], regs)
    assert list(ok) == [True, True, True, False, False]


def test_pick_columns_uses_ucd_then_names_and_never_invents():
    cols = pd.DataFrame({
        "column_name": ["Source", "RAJ2000", "DEJ2000", "Speak", "Stotal", "e_Stotal"],
        "ucd": ["meta.id;meta.main", "pos.eq.ra;meta.main", "pos.eq.dec;meta.main",
                "phot.flux.density;em.radio", "phot.flux.density;em.radio",
                "stat.error;phot.flux.density"],
        "unit": ["", "deg", "deg", "mJy/beam", "mJy", "mJy"],
        "description": [""] * 6})
    c = R.pick_columns(cols)
    assert (c["ra_col"], c["dec_col"], c["flux_col"], c["flux_unit"]) == (
        "RAJ2000", "DEJ2000", "Stotal", "mJy")
    assert R.flux_to_jy_factor("mJy")[0] == 1e-3
    assert R.flux_to_jy_factor("Jy")[0] == 1.0
    assert "ASSUMED" in R.flux_to_jy_factor("")[1]
    # Names only, no UCDs: still found; and nothing is found where nothing is.
    c2 = R.pick_columns(pd.DataFrame({"column_name": ["ra", "dec", "total_flux"]}))
    assert (c2["ra_col"], c2["dec_col"], c2["flux_col"]) == ("ra", "dec", "total_flux")
    c3 = R.pick_columns(pd.DataFrame({"column_name": ["x", "y"]}))
    assert c3["ra_col"] is None and c3["flux_col"] is None
    disc = {"service": "s", "table": "J/A+A/659/A1/lotss_dr2", "ra_col": "RAJ2000",
            "dec_col": "DEJ2000", "flux_col": "Stotal"}
    q = R.build_tile_query(disc, R.tile_bounds(100, 70, 2.0), 40000)
    assert q.startswith('SELECT TOP 40000 RAJ2000, DEJ2000, Stotal FROM "J/A+A/659/A1/lotss_dr2"')
    assert "RAJ2000 >= 200.0 AND RAJ2000 < 202.0" in q


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def test_run_radio_stage_end_to_end_with_injected_fetchers(tmp_path):
    cfg = {"radio": _cfg()}
    field = uniform_field(2000.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=5, hole_arcsec=210.0)
    tiles = R.bin_sources_into_tiles(field, 2.0)
    calls = []

    # The plan pads each target's neighbourhood out to ~1.3 deg, so it reaches
    # tile t098_070 (RA 196-198) which no target or control actually needs.
    # That is the tile made to fail here; the one the target sits in is t100_070.
    def lotss_fetcher(tile):
        calls.append(tile["key"])
        if tile["key"] == "t098_070":
            raise RuntimeError("simulated TAP outage")
        return tiles.get(tile["key"], pd.DataFrame({"ra": [], "dec": [], "flux_jy": []}))

    def target_fetcher(rcfg):
        return pd.DataFrame({"source_id": [1, 2], "ra": [FIELD_RA, 200.0], "dec": [FIELD_DEC, 50.0],
                             "parallax": [50.0, 25.0], "pmra": [0.0, 0.0], "pmdec": [0.0, 0.0]})

    s = R.run_radio_stage(cfg, tmp_path, lotss_fetcher=lotss_fetcher,
                          target_fetcher=target_fetcher, max_targets=1)
    assert s["verdict_code"] == R.VERDICT_CANDIDATES
    assert s["n_candidates"] == 1
    assert s["targets"]["n_used"] == 1
    assert s["tiles"]["QUERY_FAILED"] == 1 and s["tiles"]["QUERY_OK"] >= 4
    assert s["funnel"]["n_masked_by_bad_tile"] == 0
    assert s["funnel"]["control_false_void_rate"] == 0.0
    assert (tmp_path / "voids.csv").exists() and (tmp_path / "candidates.csv").exists()
    assert (tmp_path / "tiles" / "t100_070.parquet").exists()
    assert not (tmp_path / "tiles" / "t098_070.parquet").exists()
    led = json.loads((tmp_path / "tiles_ledger.json").read_text())
    statuses = {e["key"]: e["status"] for e in led["tiles"]}
    assert statuses["t098_070"] == "QUERY_FAILED" and statuses["t100_070"] == "QUERY_OK"
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["verdict_code"] == R.VERDICT_CANDIDATES
    assert summ["funnel"]["n_targets_in_footprint"] == 1
    assert "control_false_void_rate" in summ and "n_etz" in summ
    cands = pd.read_csv(tmp_path / "candidates.csv")
    assert list(cands["source_id"]) == [1]
    assert cands.iloc[0]["best_aperture_arcsec"] == 204.0
    # A second run (or a resume from a downloaded artifact, which is the same
    # thing: tiles/<key>.parquet present) reloads the checkpoints instead of
    # calling the fetcher again, and the ledger says FROM_CHECKPOINT.
    n_calls = len(calls)
    s2 = R.run_radio_stage(cfg, tmp_path, lotss_fetcher=lotss_fetcher,
                           target_fetcher=target_fetcher, max_targets=1)
    assert s2["tiles"]["FROM_CHECKPOINT"] == s["tiles"]["QUERY_OK"] == s2["tiles"]["n_from_checkpoint"]
    assert s2["tiles"]["QUERY_OK"] == 0
    assert len(calls) - n_calls == 1                      # only the failed tile is retried
    led2 = json.loads((tmp_path / "tiles_ledger.json").read_text())
    st2 = {e["key"]: e for e in led2["tiles"]}
    assert st2["t100_070"]["status"] == "FROM_CHECKPOINT" and st2["t100_070"]["from_checkpoint"]
    assert st2["t100_070"]["n_rows"] > 0
    assert st2["t098_070"]["status"] == "QUERY_FAILED"
    assert s2["verdict_code"] == R.VERDICT_CANDIDATES


def test_a_failed_tile_next_to_a_target_masks_it_rather_than_making_a_void(tmp_path):
    """The failure mode the first draft of this test tripped over: with the
    target's own tile QUERY_FAILED, the surviving neighbours give the annulus
    a plausible density while the apertures are empty -- a manufactured void,
    and two of the four controls 'fired' on the missing half of the sky.  A
    position whose reach touches a failed tile must be masked, never scored."""
    cfg = {"radio": _cfg()}
    field = uniform_field(2000.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=5, hole_arcsec=210.0)
    tiles = R.bin_sources_into_tiles(field, 2.0)

    def lotss_fetcher(tile):
        if tile["key"] == "t100_070":                      # the target's own tile
            raise RuntimeError("simulated TAP outage")
        return tiles.get(tile["key"], pd.DataFrame({"ra": [], "dec": [], "flux_jy": []}))

    s = R.run_radio_stage(cfg, tmp_path, lotss_fetcher=lotss_fetcher,
                          target_fetcher=lambda rcfg: _targets())
    assert s["n_candidates"] == 0
    assert s["verdict_code"] == R.VERDICT_DEGRADED
    assert "1/8 tiles QUERY_FAILED" in s["verdict"]
    f = s["funnel"]
    assert f["n_masked_by_bad_tile"] == 1
    assert f[R.VETO_FOOTPRINT] == 1
    assert f["n_control_fired"] == 0
    # The +RA and +Dec controls reach into the failed tile (RA 200-202,
    # Dec 50-52) and are masked; the -RA and -Dec ones do not and are scored.
    assert f["n_controls_masked_by_bad_tile"] == 2
    assert f["n_control_evaluated"] == 2
    v = pd.read_csv(tmp_path / "voids.csv")
    assert bool(v.iloc[0]["masked_by_bad_tile"]) and v.iloc[0]["veto"] == R.VETO_FOOTPRINT
    assert v.iloc[0]["control_+ra_veto"] == R.VETO_FOOTPRINT
    assert v.iloc[0]["control_+dec_veto"] == R.VETO_FOOTPRINT
    assert v.iloc[0]["control_-ra_veto"] == R.VETO_NOT_SIGNIFICANT
    assert v.iloc[0]["control_-dec_veto"] == R.VETO_NOT_SIGNIFICANT
    # Directly through screen_targets, the same field with nothing marked bad
    # is the manufactured void; with the tile marked bad it is masked.
    partial = {k: d for k, d in tiles.items() if k != "t100_070"}
    voids_bad, _ = R.screen_targets(_targets(), partial, cfg["radio"], bad_tiles={"t100_070"})
    assert voids_bad.iloc[0]["veto"] == R.VETO_FOOTPRINT
    voids_naive, c_naive = R.screen_targets(_targets(), partial, cfg["radio"])
    assert c_naive["n_control_fired"] > 0 or voids_naive.iloc[0]["p_trials"] < 1e-3


def test_a_raising_fetcher_records_query_failed_and_no_data_reached(tmp_path):
    cfg = {"radio": _cfg()}

    def lotss_fetcher(tile):
        raise RuntimeError("CONNECT tunnel failed, response 403")

    s = R.run_radio_stage(cfg, tmp_path, lotss_fetcher=lotss_fetcher,
                          target_fetcher=lambda rcfg: _targets())
    assert s["verdict_code"] == R.VERDICT_NO_DATA
    assert s["verdict"].startswith(R.VERDICT_NO_DATA)
    assert "403" in s["verdict"]                          # the first error text is in the verdict
    # Eight tiles planned; five consecutive failures of one class abort the
    # pull and the other three are ledgered as not attempted.
    assert s["tiles"]["n_tiles"] == 8
    assert s["tiles"]["QUERY_FAILED"] == 5
    assert s["tiles"]["NOT_ATTEMPTED_AFTER_ABORT"] == 3
    assert s["tiles"]["QUERY_OK"] == 0
    assert s["tiles"]["failed_by_class"] == {"OTHER": 5}
    assert s["acquisition_abort"]["aborted"] and s["acquisition_abort"]["n_consecutive"] == 5
    assert s["n_candidates"] == 0
    led = json.loads((tmp_path / "tiles_ledger.json").read_text())
    failed = [e for e in led["tiles"] if e["status"] == "QUERY_FAILED"]
    assert len(failed) == 5 and all("403" in e["error"] for e in failed)
    assert (tmp_path / "summary.json").exists()
    assert not any((tmp_path / "tiles").glob("*.parquet"))


def test_a_failing_target_fetch_is_no_data_reached(tmp_path):
    def target_fetcher(rcfg):
        raise RuntimeError("Gaia archive down")

    s = R.run_radio_stage({"radio": _cfg()}, tmp_path, lotss_fetcher=lambda t: None,
                          target_fetcher=target_fetcher)
    assert s["verdict_code"] == R.VERDICT_NO_DATA
    assert s["targets"]["status"] == "QUERY_FAILED"


def test_prepare_targets_cuts_regions_flags_etz_and_sorts_by_parallax():
    raw = pd.DataFrame({"source_id": [1, 2, 3], "ra": [200.0, 60.0, 0.0],
                        "dec": [50.0, 30.0, 0.0], "parallax": [30.0, 40.0, 25.0],
                        "pmra": [1000.0, 0.0, 0.0], "pmdec": [0.0, 0.0, 0.0]})
    cfg = _cfg()
    cfg["targets"] = dict(cfg["targets"], regions=cfg["targets"]["regions"] + [
        {"name": "test", "ra_min": 350.0, "ra_max": 10.0, "dec_min": -5.0, "dec_max": 5.0}])
    t = R.prepare_targets(raw, cfg)
    assert list(t["source_id"]) == [1, 3]               # 60/30 is outside every region
    assert bool(t.loc[t["source_id"] == 3, "is_etz"].iloc[0])
    assert not bool(t.loc[t["source_id"] == 1, "is_etz"].iloc[0])
    # 1000 mas/yr for 1.5 yr = 1.5" east, divided by cos(50).
    assert (t.loc[0, "ra"] - t.loc[0, "ra_gaia"]) * 3600 * np.cos(np.radians(50)) == pytest.approx(1.5, rel=1e-3)


# ---------------------------------------------------------------------------
# Column quoting -- the failure of run 34048836401
# ---------------------------------------------------------------------------
#: LoTSS DR2 column names exactly as TAPVizieR's TAP_SCHEMA.columns serves
#: them: already double-quoted.  (The bright tier hit the same thing on
#: AKARI / IRAS / Hipparcos / 2MASS, run 34048837928.)
LOTSS_SCHEMA_AS_SERVED = ['"Source"', '"RAJ2000"', '"e_RAJ2000"', '"DEJ2000"', '"e_DEJ2000"',
                          '"Speak"', '"e_Speak"', '"Stotal"', '"e_Stotal"', '"Maj"', '"Min"',
                          '"PA"', '"rms"', '"Type"', '"Mosaic"', '"Nmosaic"', '"Masked"']
LOTSS_UCDS = ["meta.id;meta.main", "pos.eq.ra;meta.main", "stat.error;pos.eq.ra",
              "pos.eq.dec;meta.main", "stat.error;pos.eq.dec", "phot.flux.density;em.radio",
              "stat.error", "phot.flux.density;em.radio", "stat.error", "phys.angSize",
              "phys.angSize", "pos.posAng", "instr.det.noise", "meta.code", "meta.id",
              "meta.number", "meta.code"]
LOTSS_UNITS = ["", "deg", "arcsec", "deg", "arcsec", "mJy/beam", "mJy/beam", "mJy", "mJy",
               "arcsec", "arcsec", "deg", "mJy/beam", "", "", "", ""]


def _assert_each_column_once_and_never_empty(adql: str, columns: tuple[str, ...]):
    assert '""' not in adql, adql
    select = adql.split(" FROM ")[0].replace("SELECT TOP 40000 ", "")
    toks = select.split(", ")
    for c in columns:
        hits = [t for t in toks if t == c or t == f'"{c}"']
        assert len(hits) == 1, (c, toks)
    for t in toks:
        assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*|"[^"]+"', t), t


def test_tile_query_from_the_quoted_schema_vizier_serves_has_no_empty_identifier():
    """Feed the column list exactly as served -- quoted strings -- through the
    same path the run uses (pick_columns -> build_tile_query)."""
    cols = pd.DataFrame({"column_name": LOTSS_SCHEMA_AS_SERVED, "ucd": LOTSS_UCDS,
                         "unit": LOTSS_UNITS})
    c = R.pick_columns(cols)
    assert (c["ra_col"], c["dec_col"], c["flux_col"], c["flux_unit"]) == (
        "RAJ2000", "DEJ2000", "Stotal", "mJy")
    assert all('"' not in str(v) for v in (c["ra_col"], c["dec_col"], c["flux_col"]))
    disc = {"table": '"J/A+A/659/A1/lotss_dr2"', **{k: c[k] for k in ("ra_col", "dec_col", "flux_col")}}
    q = R.build_tile_query(disc, R.tile_bounds(100, 70, 2.0), 40000)
    _assert_each_column_once_and_never_empty(q, ("RAJ2000", "DEJ2000", "Stotal"))
    assert 'FROM "J/A+A/659/A1/lotss_dr2" WHERE RAJ2000 >= 200.0 AND RAJ2000 < 202.0' in q
    assert q.count('"J/A+A/659/A1/lotss_dr2"') == 1
    # Quoting is idempotent: names that arrive still quoted are quoted once, not twice.
    disc_q = {"table": "J/A+A/659/A1/lotss_dr2", "ra_col": '"RAJ2000"', "dec_col": '"DEJ2000"',
              "flux_col": '"Stotal"'}
    assert R.build_tile_query(disc_q, R.tile_bounds(100, 70, 2.0), 40000) == q
    # An empty identifier is refused by logical name, never emitted.
    with pytest.raises(RuntimeError, match="dec"):
        R.build_tile_query({"table": "t", "ra_col": "RAJ2000", "dec_col": '""'},
                           R.tile_bounds(100, 70, 2.0), 40000)
    # No flux column: the SELECT has two columns and the veto is simply off.
    q2 = R.build_tile_query({"table": "t", "ra_col": "RAJ2000", "dec_col": "DEJ2000"},
                            R.tile_bounds(100, 70, 2.0), 40000)
    assert q2.startswith("SELECT TOP 40000 RAJ2000, DEJ2000 FROM t WHERE")


def test_discover_lotss_records_bare_names_and_what_was_served(monkeypatch):
    """The whole discovery path with VizieR's TAP_SCHEMA answers replayed."""
    queries = []

    def fake_run_vizier(query, **kw):
        queries.append(query)
        if "TAP_SCHEMA.tables" in query:
            return pd.DataFrame({"table_name": ["J/A+A/659/A1/gaus", "J/A+A/659/A1/lotss_dr2"],
                                 "description": ["Gaussian components", "LoTSS DR2 sources"]})
        if "TAP_SCHEMA.columns" in query:
            assert "J/A+A/659/A1/lotss_dr2" in query
            return pd.DataFrame({"column_name": LOTSS_SCHEMA_AS_SERVED, "ucd": LOTSS_UCDS,
                                 "unit": LOTSS_UNITS, "datatype": ["double"] * 17,
                                 "description": [""] * 17})
        raise AssertionError(f"unexpected query {query}")

    monkeypatch.setattr(R.bright, "run_vizier", fake_run_vizier)
    disc = R.discover_lotss(R.load_radio_config())
    assert disc["status"] == "DISCOVERED" and disc["route"] == "vizier"
    assert disc["table"] == '"J/A+A/659/A1/lotss_dr2"' and disc["table_bare"] == "J/A+A/659/A1/lotss_dr2"
    assert (disc["ra_col"], disc["dec_col"], disc["flux_col"]) == ("RAJ2000", "DEJ2000", "Stotal")
    assert disc["flux_unit"] == "mJy" and disc["flux_to_jy"] == 1e-3
    assert disc["names_as_served"][:2] == ['"Source"', '"RAJ2000"']    # evidence, verbatim
    assert "RAJ2000" in disc["columns"] and '"RAJ2000"' not in disc["columns"]
    assert '""' not in disc["example_query"]
    assert disc["degraded_reasons"] == [] and not disc["degraded"]
    assert len([q for q in queries if "TAP_SCHEMA.tables" in q]) == 1       # VizieR answered: no ASTRON call
    # The fetcher built on it composes the same valid query and normalises the
    # result whatever spelling the TAP result comes back with.
    seen = {}

    def fake_sync(url, adql):
        seen["q"] = adql
        return pd.DataFrame({"RAJ2000": [200.5], "DEJ2000": [50.5], "Stotal": [1500.0]})

    monkeypatch.setattr(R, "_run_sync", fake_sync)
    df = R.make_lotss_fetcher(disc, R.load_radio_config())(R.tile_bounds(100, 70, 2.0))
    _assert_each_column_once_and_never_empty(seen["q"], ("RAJ2000", "DEJ2000", "Stotal"))
    assert list(df.columns) == ["ra", "dec", "flux_jy"] and df["flux_jy"].iloc[0] == pytest.approx(1.5)


def test_discover_lotss_falls_back_to_astron_and_says_so(monkeypatch):
    def fake_run_vizier(query, **kw):
        if "tapvizier" in kw.get("url", ""):
            raise RuntimeError("vizier down")
        if "TAP_SCHEMA.tables" in query:
            return pd.DataFrame({"table_name": ["lotss_dr2.main_sources", "lotss_dr2.gaussians"],
                                 "description": ["", ""]})
        return pd.DataFrame({"column_name": ["ra", "dec", "total_flux"],
                             "ucd": ["pos.eq.ra", "pos.eq.dec", "phot.flux.density"],
                             "unit": ["deg", "deg", "Jy"], "datatype": ["double"] * 3,
                             "description": [""] * 3})

    monkeypatch.setattr(R.bright, "run_vizier", fake_run_vizier)
    disc = R.discover_lotss(R.load_radio_config())
    assert disc["status"] == "DISCOVERED" and disc["route"] == "astron"
    assert disc["table"] == "lotss_dr2.main_sources"
    assert (disc["ra_col"], disc["dec_col"], disc["flux_col"]) == ("ra", "dec", "total_flux")
    assert disc["flux_to_jy"] == 1.0
    assert any("fallback route astron" in r for r in disc["degraded_reasons"]) and disc["degraded"]
    assert disc["ledger"][0]["status"] == "QUERY_FAILED"


# ---------------------------------------------------------------------------
# Fail fast on syntax, back off on throttling
# ---------------------------------------------------------------------------
def test_an_adql_syntax_error_is_not_retried(monkeypatch):
    from pyvo.dal.exceptions import DALQueryError
    calls, sleeps = [], []

    def bad_sync(url, adql):
        calls.append(adql)
        raise DALQueryError('Incorrect ADQL query:  Encountered "\\"\\"". Was expecting one of: "," "FROM"')

    monkeypatch.setattr(R, "_run_sync", bad_sync)
    monkeypatch.setattr(R, "_sleep", sleeps.append)
    with pytest.raises(R.TileQueryError) as ei:
        R.query_tile("https://example/tap", 'SELECT "" FROM x', "t", retries=3)
    assert ei.value.error_class == R.ERR_ADQL_SYNTAX
    assert ei.value.query == 'SELECT "" FROM x' and ei.value.attempts == 1
    assert len(calls) == 1 and sleeps == []


def test_a_connect_timeout_backs_off_10_30_90(monkeypatch):
    from requests.exceptions import ConnectTimeout
    calls, sleeps = [], []

    def slow_sync(url, adql):
        calls.append(adql)
        raise ConnectTimeout("HTTPSConnectionPool(host='tapvizier.cds.unistra.fr', port=443): "
                             "Max retries exceeded")

    monkeypatch.setattr(R, "_run_sync", slow_sync)
    monkeypatch.setattr(R, "_sleep", sleeps.append)
    with pytest.raises(R.TileQueryError) as ei:
        R.query_tile("https://example/tap", "SELECT 1", "t", retries=3)
    assert ei.value.error_class == R.ERR_CONNECT_TIMEOUT
    assert sleeps == [10.0, 30.0, 90.0] and len(calls) == 4
    # ...and a transient failure that clears is returned normally.
    n = {"k": 0}

    def flaky(url, adql):
        n["k"] += 1
        if n["k"] < 3:
            raise ConnectTimeout("timed out")
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(R, "_run_sync", flaky)
    assert len(R.query_tile("u", "SELECT 1", "t", retries=3)) == 1


def test_classify_error_reads_the_runner_log_texts():
    assert R.classify_error(RuntimeError(
        'DALQueryError: Incorrect ADQL query:  Encountered "\\"\\"". Was expecting one of: "," "FROM"')) == R.ERR_ADQL_SYNTAX
    assert R.classify_error(RuntimeError(
        "DALFormatError: ConnectTimeout: HTTPSConnectionPool(host='tapvizier.cds.unistra.fr', "
        "port=443): Max retries exceeded")) == R.ERR_CONNECT_TIMEOUT
    assert R.classify_error(ValueError("fetcher returned no 'ra' column")) == R.ERR_OTHER
    assert R.classify_error(R.TileQueryError(R.ERR_ADQL_SYNTAX, "x")) == R.ERR_ADQL_SYNTAX


def test_five_consecutive_syntax_failures_abort_the_pull_in_seconds(tmp_path):
    """The 5.5-hour run, replayed: every tile fails with the same syntax
    error.  Now the fifth aborts; the ledger names the class, the first
    error and its query, and the summary is NO_DATA_REACHED with that text."""
    import time
    n = {"calls": 0}

    def lotss_fetcher(tile):
        n["calls"] += 1
        raise R.TileQueryError(R.ERR_ADQL_SYNTAX,
                               'DALQueryError: Incorrect ADQL query:  Encountered "\\"\\"".',
                               query='SELECT TOP 40000 ""RAJ2000"" FROM "J/A+A/659/A1/lotss_dr2"')

    rng = np.random.default_rng(4)
    targets = pd.DataFrame({"source_id": range(40), "ra": rng.uniform(150, 250, 40),
                            "dec": rng.uniform(30, 60, 40), "parallax": 30.0})
    t0 = time.monotonic()
    s = R.run_radio_stage({"radio": _cfg()}, tmp_path, lotss_fetcher=lotss_fetcher,
                          target_fetcher=lambda rcfg: targets)
    assert time.monotonic() - t0 < 30
    assert n["calls"] == 5
    assert s["verdict_code"] == R.VERDICT_NO_DATA
    assert "Incorrect ADQL" in s["verdict"] and "ADQL_SYNTAX" in s["verdict"]
    ab = s["acquisition_abort"]
    assert ab["aborted"] and ab["error_class"] == R.ERR_ADQL_SYNTAX and ab["n_consecutive"] == 5
    assert '""RAJ2000""' in ab["first_error"]["query"]
    assert s["tiles"]["QUERY_FAILED"] == 5
    assert s["tiles"]["NOT_ATTEMPTED_AFTER_ABORT"] == s["tiles"]["n_tiles"] - 5 > 0
    led = json.loads((tmp_path / "tiles_ledger.json").read_text())
    failed = [e for e in led["tiles"] if e["status"] == "QUERY_FAILED"]
    assert all(e["error_class"] == R.ERR_ADQL_SYNTAX and '""' in e["query"] for e in failed)
    assert (tmp_path / "summary.json").exists() and (tmp_path / "voids.csv").exists()
    # Failures of DIFFERENT classes do not accumulate into one streak.
    n["calls"] = 0
    classes = [R.ERR_ADQL_SYNTAX, R.ERR_CONNECT_TIMEOUT]

    def alternating(tile):
        n["calls"] += 1
        raise R.TileQueryError(classes[n["calls"] % 2], "boom")

    cfg = {"radio": _cfg(lotss=dict(_cfg()["lotss"], max_consecutive_tile_failures=3))}
    s2 = R.run_radio_stage(cfg, tmp_path / "b", lotss_fetcher=alternating,
                           target_fetcher=lambda rcfg: targets.head(3))
    assert not s2["acquisition_abort"]["aborted"]
    assert s2["tiles"]["QUERY_FAILED"] == s2["tiles"]["n_tiles"]


def test_a_resumed_run_keeps_its_checkpoints_when_the_service_then_dies(tmp_path):
    """resume_from_run_id downloads tiles/*.parquet; if VizieR then fails on
    every remaining tile the checkpointed ones are still screened and the
    verdict is DEGRADED, not NO_DATA."""
    cfg = {"radio": _cfg(lotss=dict(_cfg()["lotss"], max_consecutive_tile_failures=1))}
    field = uniform_field(2000.0, FIELD_RA, FIELD_DEC, 2.6, 1.6, seed=5, hole_arcsec=210.0)
    tiles = R.bin_sources_into_tiles(field, 2.0)
    # "Downloaded artifact": every tile except the four outermost in RA (ix 98
    # and 101, two Dec rows each), which no target or control reaches into.
    (tmp_path / "tiles").mkdir(parents=True)
    withheld = {k for k in tiles if k.startswith(("t098_", "t101_"))}
    assert len(withheld) == 4
    for k, df in tiles.items():
        if k not in withheld:
            df.to_parquet(tmp_path / "tiles" / f"{k}.parquet", index=False)
    calls = []

    def dead(tile):
        calls.append(tile["key"])
        raise R.TileQueryError(R.ERR_CONNECT_TIMEOUT, "ConnectTimeout")

    s = R.run_radio_stage(cfg, tmp_path, lotss_fetcher=dead, target_fetcher=lambda rcfg: _targets())
    assert s["tiles"]["FROM_CHECKPOINT"] == 4 and s["tiles"]["n_rows_total"] > 0
    assert calls == ["t098_069"]                          # aborted on the first live failure
    assert s["tiles"]["QUERY_FAILED"] == 1 and s["tiles"]["NOT_ATTEMPTED_AFTER_ABORT"] == 3
    assert s["acquisition_abort"]["aborted"]
    assert any("acquisition aborted" in r for r in s["degraded_reasons"])
    # The checkpointed sky is still screened: the hole is found, the verdict
    # is not NO_DATA, and the abort is on record in the summary.
    assert s["verdict_code"] == R.VERDICT_CANDIDATES
    assert s["n_targets_in_footprint"] == 1
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["acquisition_abort"]["error_class"] == R.ERR_CONNECT_TIMEOUT


def test_config_file_and_defaults_agree_on_every_key():
    cfg = R.load_radio_config()
    for k in R.DEFAULTS:
        assert k in cfg
    assert cfg["p_min"] == 1e-5
    assert cfg["min_annulus_density_per_deg2"] == 300.0
    assert cfg["min_expected_count"] == 8.0
    assert 204.0 in cfg["apertures_arcsec"]
    assert R._radio_cfg({"radio": {"p_min": 1e-3}})["p_min"] == 1e-3
    assert R._radio_cfg({})["p_min"] == 1e-5

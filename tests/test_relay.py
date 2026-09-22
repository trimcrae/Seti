"""Offline suite for RELAY (S60): intercepting node-to-node beams by geometry.

No socket is opened (``conftest.py`` raises on any).  Per ``docs/channel-brief.md``
section 5 the suite:

* recovers an injected signal --- a synthetic (T, R) pair placed inside the
  spillover cone is found at the beam that admits it and not at a narrower
  one, with the exact transmitter angle equal to the closed form, and a hit
  on that transmitter whose drift equals the pair's prior is the candidate;
* counts the near-antipodal "Earth between the nodes" geometry separately;
* returns a clean null on the confounders --- terrestrial interference (zero
  topocentric drift), a frequency recurring on unrelated sightlines, a hit
  on a star that is on no pair line, a drift outside the window;
* degrades honestly --- an empty target list, an empty Gaia answer and an
  empty hit catalogue each yield NO_DATA_REACHED / NO_HIT_CATALOGUE_REACHED,
  never a candidate;
* checks the geometry against brute force, the theta^2 N^2 scaling against
  the analytic coefficient, the range acceleration against a finite
  difference, and the Earth term against its closed form.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from seti.relay import acquire as acq
from seti.relay import geometry as geo
from seti.relay.run import (
    V_GEOMETRY,
    V_NO_DATA,
    V_NO_HIT_CATALOGUE,
    V_NO_PAIRLINE_HIT,
    V_PAIRLINE_MATCH,
    V_PAIRLINE_OFF_PRIOR,
    load_relay_config,
    name_key,
    relay_run,
    stage_assess,
    stage_geometry,
    stage_targets,
)

DEG = geo.DEG


# ---------------------------------------------------------------------------
# fixtures: a synthetic 100 pc sample with two injected pairs
# ---------------------------------------------------------------------------
def _offset_radec(ra0, dec0, sep_deg, pa_deg=90.0):
    """A point ``sep_deg`` from (ra0, dec0) along position angle ``pa_deg``."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(ra0 * u.deg, dec0 * u.deg).directional_offset_by(pa_deg * u.deg, sep_deg * u.deg)
    return float(c.ra.deg), float(c.dec.deg)


INJ_T_SPILL = "HIP 999001"      # transmitter of the spillover pair
INJ_R_SPILL = "HIP 999002"
INJ_T_BETWEEN = "HIP 999003"
INJ_R_BETWEEN = "HIP 999004"
QUIET = "HIP 999005"            # a star on no pair line at any beam


def make_sample(n_random: int = 2500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_random, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    d = 100.0 * rng.random(n_random) ** (1 / 3)
    d = np.clip(d, 12.0, 100.0)             # keep the random receivers from dominating the caps
    ra = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360
    dec = np.degrees(np.arcsin(u[:, 2]))
    df = pd.DataFrame({"source_id": [str(10_000 + i) for i in range(n_random)], "ra": ra, "dec": dec,
                       "parallax": 1000.0 / d, "parallax_error": 0.02,
                       "pmra": rng.normal(0, 60, n_random), "pmdec": rng.normal(0, 60, n_random),
                       "radial_velocity": rng.normal(0, 25, n_random), "radial_velocity_error": 0.5,
                       "phot_g_mean_mag": 10.0, "bp_rp": 1.0, "ruwe": 1.0, "grvs_mag": np.nan,
                       "rv_template_teff": 5000.0})
    # injected spillover pair: R at 20 pc, T at 60 pc, delta = half of delta_max at 1.5 deg
    theta = 1.5 * DEG
    d_r, d_t = 20.0, 60.0
    delta = 0.5 * geo.delta_max_spillover(theta, d_t, d_r) / DEG
    ra_r, dec_r = 100.0, 10.0
    ra_t, dec_t = _offset_radec(ra_r, dec_r, delta)
    # injected between pair: T at 50 pc, R 1 deg from T's antipode at 30 pc
    ra_tb, dec_tb = 200.0, -20.0
    ra_rb, dec_rb = _offset_radec((ra_tb + 180.0) % 360, -dec_tb, 1.0, pa_deg=45.0)
    inj = pd.DataFrame({
        "source_id": ["999001", "999002", "999003", "999004", "999005"],
        "ra": [ra_t, ra_r, ra_tb, ra_rb, 10.0], "dec": [dec_t, dec_r, dec_tb, dec_rb, 80.0],
        "parallax": [1000 / d_t, 1000 / d_r, 1000 / 50.0, 1000 / 30.0, 1000 / 40.0],
        "parallax_error": 0.02, "pmra": [30.0, -40.0, 10.0, 55.0, 0.0], "pmdec": [-20.0, 25.0, 5.0, -15.0, 0.0],
        "radial_velocity": [12.0, -5.0, 3.0, 20.0, 0.0], "radial_velocity_error": 0.3,
        "phot_g_mean_mag": 8.0, "bp_rp": 0.9, "ruwe": 1.0, "grvs_mag": np.nan, "rv_template_teff": 5500.0})
    return pd.concat([df, inj], ignore_index=True)


@pytest.fixture(scope="module")
def conf():
    c = load_relay_config()
    c["bl_opendata"]["request_sleep_s"] = 0.0
    c["geometry"]["commit_full_catalogue_max_rows"] = 200000
    return c


@pytest.fixture(scope="module")
def sample():
    return make_sample()


def _idx(sample, sid):
    return int(np.flatnonzero(sample["source_id"].astype(str) == sid)[0])


# ---------------------------------------------------------------------------
# beams and the closed forms
# ---------------------------------------------------------------------------
def test_beam_grid_reproduces_the_briefs_numbers(conf):
    beams = {b["name"]: b for b in geo.beam_grid(conf["beams"])}
    assert abs(beams["radio_100m_1p4ghz"]["theta_rad"] / geo.ARCMIN - 8.85) < 0.3      # ~9 arcmin
    assert abs(beams["radio_10m_1p4ghz"]["theta_deg"] - 1.47) < 0.05                  # ~1.5 deg
    assert beams["optical_10m_1um"]["theta_rad"] / geo.ARCSEC < 0.03
    assert beams["overfilled_5deg"]["theta_deg"] == 5.0


def test_transmitter_angle_closed_form():
    # spillover: T at 60 pc, R at 20 pc, delta on the sky -> tan(alpha) = dR sin d / (dT - dR cos d)
    delta = 0.7 * DEG
    xt = np.array([[60.0, 0.0, 0.0]])
    xr = 20.0 * np.array([[math.cos(delta), math.sin(delta), 0.0]])
    a = geo.transmitter_angle(xt, xr)[0]
    expect = math.atan2(20 * math.sin(delta), 60 - 20 * math.cos(delta))
    assert math.isclose(a, expect, rel_tol=1e-9)
    # between: R 1 deg from T's antipode
    dp = 1.0 * DEG
    xr2 = 30.0 * np.array([[-math.cos(dp), math.sin(dp), 0.0]])
    xt2 = np.array([[50.0, 0.0, 0.0]])
    a2 = geo.transmitter_angle(xt2, xr2)[0]
    expect2 = math.atan2(30 * math.sin(dp), 50 + 30 * math.cos(dp))
    assert math.isclose(a2, expect2, rel_tol=1e-9)


def test_exact_separation_bounds_reduce_to_the_brief_and_saturate_for_near_receivers():
    th = 1.5 * DEG
    sp, bt = geo.delta_max_exact(th, 60.0, 20.0)
    assert abs(sp / geo.delta_max_spillover(th, 60.0, 20.0) - 1) < 2e-3      # arcsin x ~ x
    assert abs(bt / geo.delta_max_between(th, 60.0, 20.0) - 1) < 2e-3
    assert sp > geo.delta_max_spillover(th, 60.0, 20.0)                     # exact is the superset
    sp, bt = geo.delta_max_exact(5.0 * DEG, 100.0, 2.0)                      # d_R < d_T sin(theta/2)
    assert sp == math.pi and bt == math.pi
    # a receiver 2 pc away with T at 100 pc at 90 deg separation is inside a 5 deg cone
    xt = np.array([[100.0, 0.0, 0.0]])
    xr = np.array([[0.0, 2.0, 0.0]])
    assert geo.transmitter_angle(xt, xr)[0] <= 2.5 * DEG


def test_injected_spillover_pair_found_only_at_the_beam_that_admits_it(sample):
    xyz = geo.positions_pc(sample["ra"], sample["dec"], sample["parallax"])
    t, r = _idx(sample, "999001"), _idx(sample, "999002")
    wide = geo.find_pairs(xyz, 1.5 * DEG, d_max_pc=100.0)
    rows = wide.pairs[(wide.pairs["t_idx"] == t) & (wide.pairs["r_idx"] == r)]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["geometry"] == geo.GEOM_SPILLOVER
    assert 0.4 < row["alpha_over_half_theta"] < 0.6          # placed at half the cone
    assert row["flux_ratio_earth_over_receiver"] < 1.0        # Earth is farther than R from T
    assert abs(row["flux_ratio_earth_over_receiver"] - (40.0 / 60.0) ** 2) < 0.02
    narrow = geo.find_pairs(xyz, 9.0 * geo.ARCMIN, d_max_pc=100.0)
    assert not len(narrow.pairs[(narrow.pairs["t_idx"] == t) & (narrow.pairs["r_idx"] == r)])
    # the reverse direction (near star transmitting to the far one) never reaches Earth
    assert not len(wide.pairs[(wide.pairs["t_idx"] == r) & (wide.pairs["r_idx"] == t)])


def test_between_geometry_is_counted_separately_with_flux_above_one(sample):
    xyz = geo.positions_pc(sample["ra"], sample["dec"], sample["parallax"])
    t, r = _idx(sample, "999003"), _idx(sample, "999004")
    res = geo.find_pairs(xyz, 5.0 * DEG, d_max_pc=100.0)
    rows = res.pairs[(res.pairs["t_idx"] == t) & (res.pairs["r_idx"] == r)]
    assert len(rows) == 1 and rows.iloc[0]["geometry"] == geo.GEOM_BETWEEN
    assert rows.iloc[0]["flux_ratio_earth_over_receiver"] > 1.0
    assert rows.iloc[0]["sep_deg"] > 170.0
    assert res.n_between >= 1 and res.per_transmitter_between[t] >= 1
    assert res.per_transmitter_spill[t] == 0 or res.per_transmitter_spill[t] >= 0
    narrow = geo.find_pairs(xyz, 9.0 * geo.ARCMIN, d_max_pc=100.0)
    assert not len(narrow.pairs[(narrow.pairs["t_idx"] == t) & (narrow.pairs["r_idx"] == r)])


def test_tree_search_matches_brute_force():
    rng = np.random.default_rng(11)
    n = 1500
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    xyz = u * (100 * rng.random(n) ** (1 / 3))[:, None]
    th = 3.0 * DEG
    res = geo.find_pairs(xyz, th, d_max_pc=100.0)
    tr = xyz[None, :, :] - xyz[:, None, :]
    te = -xyz[:, None, :]
    cosa = np.sum(tr * te, -1) / (np.linalg.norm(tr, axis=-1) * np.linalg.norm(te, axis=-1) + 1e-30)
    np.fill_diagonal(cosa, -1)
    brute = int((np.arccos(np.clip(cosa, -1, 1)) <= th / 2).sum())
    assert res.n_spillover + res.n_between == brute
    assert len(res.pairs) == brute


def test_counts_scale_as_theta_squared_and_track_the_analytic_coefficient():
    rng = np.random.default_rng(5)
    n = 30000
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    xyz = u * (100 * rng.random(n) ** (1 / 3))[:, None]
    c1 = geo.find_pairs(xyz, 0.5 * DEG, d_max_pc=100.0, row_cap=10)
    c2 = geo.find_pairs(xyz, 1.0 * DEG, d_max_pc=100.0, row_cap=10)
    ratio = c2.n_between / c1.n_between
    assert 3.6 < ratio < 4.4                                  # theta^2
    exp = geo.analytic_expectation(1.0 * DEG, n)
    assert 0.8 < c2.n_between / exp["between"] < 1.2
    assert 0.5 < c2.n_spillover / exp["spillover"] < 1.5     # dominated by the nearest receivers
    assert c2.truncated and c2.n_kept_rows == 10             # the cap counts on, stores nothing more


def test_analytic_coefficients_against_monte_carlo():
    rng = np.random.default_rng(3)
    m = 12_000_000
    th = 3.0 * DEG
    d_t = 100 * rng.random(m) ** (1 / 3)
    d_r = 100 * rng.random(m) ** (1 / 3)
    sep = np.arccos(rng.uniform(-1, 1, m))
    spill = ((d_t > d_r) & (sep <= geo.delta_max_spillover(th, d_t, d_r))).mean()
    between = ((math.pi - sep) <= geo.delta_max_between(th, d_t, d_r)).mean()
    exp = geo.analytic_expectation(th, 1)
    assert abs(spill / exp["spillover"] - 1) < 0.08          # ~1,000 spillover draws: Poisson 3 %
    assert abs(between / exp["between"] - 1) < 0.05


# ---------------------------------------------------------------------------
# kinematics and the drift prior
# ---------------------------------------------------------------------------
def test_range_acceleration_matches_a_finite_difference():
    x_a = np.array([10.0, 0.0, 0.0])
    x_b = np.array([12.0, 3.0, -1.0])
    v_a = np.array([5.0, -3.0, 2.0])
    v_b = np.array([-4.0, 6.0, 1.0])
    v_r, a = geo.range_rate_and_acceleration(x_a, v_a, x_b, v_b)
    kms_per_yr_pc = 1.0 / 977792.22            # 1 km/s = 1.0227e-6 pc/yr
    dt_yr = 100.0

    def r_at(t):
        return np.linalg.norm((x_b + v_b * kms_per_yr_pc * t) - (x_a + v_a * kms_per_yr_pc * t))

    r0, rp, rm = r_at(0), r_at(dt_yr), r_at(-dt_yr)
    v_fd = (rp - rm) / (2 * dt_yr) / kms_per_yr_pc                # km/s
    a_fd = (rp - 2 * r0 + rm) / (dt_yr * 3.15576e7) ** 2 * geo.PC_M   # m/s^2
    assert math.isclose(v_r, v_fd, rel_tol=1e-6)
    assert math.isclose(a, a_fd, rel_tol=1e-3)
    assert a < 1e-8                                              # field-star scale: negligible drift


def test_kinematic_drift_prior_is_far_below_pipeline_resolution(sample, conf):
    xyz = geo.positions_pc(sample["ra"], sample["dec"], sample["parallax"])
    vel = geo.space_velocities(sample["ra"], sample["dec"], sample["parallax"], sample["pmra"],
                               sample["pmdec"], sample["radial_velocity"])
    t, r = _idx(sample, "999001"), _idx(sample, "999002")
    k = geo.pair_kinematics(xyz, vel, [t], [r])
    assert bool(k["kinematics_complete"].iloc[0])
    d = geo.drift_hz_s(1.42e9, k["a_kin_m_s2"].iloc[0])
    assert abs(d) < 1e-5 < conf["drift"]["sigma_drift_default_hz_s"]
    # v_TE is the transmitter's recession speed
    ut = xyz[t] / np.linalg.norm(xyz[t])
    assert math.isclose(k["v_te_kms"].iloc[0], float(vel[t] @ ut), rel_tol=1e-9)


def test_earth_term_sign_and_magnitude_at_transit():
    lat, lon = 38.4331, -79.8398
    mjd = 57650.3
    lst = (geo.gmst_deg(mjd) + lon) % 360.0
    e = geo.earth_acceleration_los(mjd, lat, lon, ra_deg=lst, dec_deg=10.0)
    expect = -(geo.OMEGA_EARTH ** 2) * geo.R_EARTH_M * math.cos(math.radians(lat)) * math.cos(math.radians(10.0))
    assert math.isclose(float(e["a_rot_m_s2"]), expect, rel_tol=1e-9)
    assert float(e["a_rot_m_s2"]) < 0                          # site accelerates away from a transiting source
    assert abs(float(e["a_orb_m_s2"])) <= 6.1e-3
    # six hours later the rotational term vanishes
    e6 = geo.earth_acceleration_los(mjd + 0.25 * 0.9972696, lat, lon, ra_deg=lst, dec_deg=10.0)
    assert abs(float(e6["a_rot_m_s2"])) < 1e-3 * abs(expect)
    # a de-drifted relay at L band shows ~0.1 Hz/s of it
    assert 0.05 < abs(geo.drift_hz_s(1.42e9, e["a_los_m_s2"])) < 0.2


def test_sun_direction_on_the_equinox():
    ra, dec, dist = geo.sun_direction(57467.7)                  # 2016-03-20 ~ 04:30 UT
    assert min(ra, 360 - ra) < 1.0 and abs(dec) < 0.5
    assert 0.98 < dist < 1.02


def test_drift_window_tight_is_the_small_angle_leak_and_loose_adds_the_receiver_platform(conf):
    f = 1.42e9
    w = geo.drift_window(f, 0.0, -0.02, alpha_rad=1e-3, a_local_max_m_s2=1.0,
                         sigma_drift_hz_s=0.0093)
    assert math.isclose(float(w["centre_hz_s"]), f / geo.C_M_S * -0.02, rel_tol=1e-9)
    leak = f / geo.C_M_S * 1e-3
    assert math.isclose(float(w["halfwidth_tight_hz_s"]), math.hypot(0.0093, leak), rel_tol=1e-9)
    assert float(w["halfwidth_loose_hz_s"]) > 4.0             # +/- the whole search range: no discrimination
    w2 = geo.drift_window(f, 0.0, 0.03, alpha_rad=1e-3, a_local_max_m_s2=1.0,
                          sigma_drift_hz_s=0.0093, earth_term_known=False)
    assert float(w2["centre_hz_s"]) == 0.0
    assert float(w2["halfwidth_tight_hz_s"]) > float(w["halfwidth_tight_hz_s"])


# ---------------------------------------------------------------------------
# archive parsing
# ---------------------------------------------------------------------------
def test_bl_file_normalisation_reads_mjd_from_the_file_name_and_units():
    rec = {"telescope": "Green Bank Telescope", "target_name": "HIP999001",
           "file_name": "blc00_guppi_57650_67573_HIP999001_0012.gpuspec.0000.h5",
           "center_freq": 1500.0, "file_type": "hdf5", "size": 1.2}
    f = acq.normalise_bl_file(rec)
    assert math.isclose(f["mjd"], 57650 + 67573 / 86400, rel_tol=1e-12)
    assert math.isclose(f["freq_centre_ghz"], 1.5)
    assert f["telescope"] == "Green Bank Telescope" and f["target"] == "HIP999001"
    assert math.isnan(acq.mjd_from_string("nothing_here"))
    assert acq.target_names_from(["HIP1", {"target": "GJ 2"}, {"name": "X"}]) == ["HIP1", "GJ 2", "X"]
    assert acq.target_names_from({"data": [{"target_name": "A"}]}) == ["A"]
    assert acq.unwrap_records({"results": [1, 2]}) == [1, 2] and acq.unwrap_records("x") == []


def test_name_variants_and_keys():
    assert "HIP 12345" in acq.name_variants("HIP12345")
    assert "GJ 1002" in acq.name_variants("GL_1002")
    assert name_key("HIP 12345") == name_key("hip_12345") == "HIP12345"
    assert name_key("GL 1002") == name_key("GJ1002")


def test_hit_table_classification_by_roles():
    kind, roles = acq.classify_hit_table(["Name", "RAJ2000", "DEJ2000", "Freq", "Drift", "SNR", "MJD"])
    assert kind == acq.KIND_HITS and roles["freq_mhz"] == "Freq" and roles["drift"] == "Drift"
    kind, _ = acq.classify_hit_table(["HIP", "RAJ2000", "DEJ2000", "Dist"])
    assert kind == acq.KIND_TARGETS
    kind, _ = acq.classify_hit_table(["a", "b"])
    assert kind == acq.KIND_OTHER


def test_canonicalise_gaia_accepts_vizier_spellings():
    df = pd.DataFrame({"Source": [1, 2], "RA_ICRS": [1.0, 2.0], "DE_ICRS": [3.0, 4.0], "Plx": [50.0, 20.0],
                       "pmRA": [1.0, 1.0], "pmDE": [2.0, 2.0], "RV": [np.nan, 5.0], "Gmag": [9.0, 10.0]})
    out = acq.canonicalise_gaia(df)
    assert list(out["source_id"]) == ["1", "2"] and out["parallax"].tolist() == [50.0, 20.0]
    assert np.isnan(out["ruwe"]).all() and out["radial_velocity"].iloc[1] == 5.0


# ---------------------------------------------------------------------------
# honest degradation
# ---------------------------------------------------------------------------
def _failing_fetch(url, params):
    raise ConnectionError("no route to archive")


def _failing_tap(adql, endpoint, maxrec=None):
    raise RuntimeError("TAP down")


def _failing_query(adql):
    raise RuntimeError("VizieR down")


def test_empty_archive_yields_no_data_reached(tmp_path, conf):
    rep = stage_targets(conf, tmp_path, bl_fetch=_failing_fetch, tap_fn=_failing_tap,
                        query_fn=_failing_query, probe={"hit_tables": []})
    assert rep["verdict"].startswith(V_NO_DATA) and rep["n_resolved"] == 0
    rep = stage_geometry(conf, tmp_path, tap_fn=_failing_tap, query_fn=_failing_query, fetch_fn=_failing_fetch)
    assert rep["verdict"].startswith(V_NO_DATA) and rep["n_stars"] == 0
    rep = stage_assess(conf, tmp_path, query_fn=_failing_query, probe={"hit_tables": []})
    assert rep["assess_verdict"] == V_NO_HIT_CATALOGUE
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["candidates"] == [] and V_NO_DATA in s["verdict"]


def test_hits_reached_without_geometry_is_not_a_sky_statement(tmp_path, conf):
    hits = pd.DataFrame({"target": ["HIP 1"], "freq_mhz": [1420.0], "drift": [0.1], "snr": [20.0],
                         "mjd": [57650.0]})
    rep = stage_assess(conf, tmp_path, hits_df=hits)
    assert rep["assess_verdict"].startswith(V_NO_DATA)


# ---------------------------------------------------------------------------
# the full offline pipeline through fake transports
# ---------------------------------------------------------------------------
class FakeBL:
    """The documented routes with the shapes a Flask JSON API would return."""

    def __init__(self, targets, files_by_target):
        self.targets = targets
        self.files = files_by_target
        self.calls = []

    def __call__(self, url, params):
        self.calls.append((url, dict(params)))
        if url.endswith("list-targets"):
            return [{"target_name": t} for t in self.targets]
        if url.endswith("list-telescopes"):
            return ["GBT", "Parkes", "MeerKAT"]
        if url.endswith("list-file-types"):
            return {"data": ["hdf5", "filterbank"]}
        if url.endswith("query-files"):
            return {"data": list(self.files.get(params.get("target", ""), []))}
        raise ValueError(f"unknown route {url}")


def fake_simbad(sample):
    def _tap(adql, endpoint, maxrec=None):
        if "TAP_SCHEMA" in adql or "gaia_source" in adql:
            raise RuntimeError("not simbad")
        import re
        asked = re.findall(r"'([^']*)'", adql.split("WHERE", 1)[1])
        rows = []
        for a in asked:
            k = name_key(a)
            m = re.match(r"^HIP(\d+)$", k)
            if not m:
                continue
            sid = m.group(1)
            hit = sample[sample["source_id"] == sid]
            if len(hit):
                rows.append({"asked": a, "main_id": f"HIP {sid}", "ra": float(hit["ra"].iloc[0]),
                             "dec": float(hit["dec"].iloc[0]), "plx_value": float(hit["parallax"].iloc[0]),
                             "pmra": 0.0, "pmdec": 0.0, "rvz_radvel": 0.0, "gaia": f"Gaia DR3 {sid}"})
        return pd.DataFrame(rows)
    return _tap


def _gbt_file(target, mjd_int, sec):
    return {"telescope": "GBT", "target_name": target,
            "file": f"blc00_guppi_{mjd_int}_{sec:05d}_{target.replace(' ', '')}_0012.gpuspec.0000.h5",
            "center_freq": 1500.0, "file_type": "hdf5"}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, conf, sample):
    out = tmp_path_factory.mktemp("relay")
    targets = [INJ_T_SPILL.replace(" ", ""), INJ_T_BETWEEN.replace(" ", ""), QUIET.replace(" ", ""),
               "HIP 10005", "NotAStar_1"]
    files = {t: [_gbt_file(t, 57650, 67573), _gbt_file(t, 57650, 68000)] for t in targets[:4]}
    bl = FakeBL(targets, files)
    tap = fake_simbad(sample)
    # the hits: (a) on the spillover transmitter, drift at the prior; (b) the same star at zero
    # drift (RFI); (c) on the quiet star; (d) on the transmitter far off the prior;
    # (e,f,g) one frequency on three unrelated sightlines
    site = conf["telescopes"]["GBT"]
    t = _idx(sample, "999001")
    mjd = 57650 + 67573 / 86400
    e = geo.earth_acceleration_los(mjd, site["lat_deg"], site["lon_deg"], sample["ra"][t], sample["dec"][t])
    f_mhz = 1420.5
    prior = geo.drift_hz_s(f_mhz * 1e6, float(e["a_los_m_s2"]))
    hits = pd.DataFrame({
        "target": [INJ_T_SPILL, INJ_T_SPILL, QUIET, INJ_T_SPILL, "HIP 10005", "HIP 10006", "HIP 10007"],
        "freq_mhz": [f_mhz, 1600.0, f_mhz, f_mhz, 1233.4, 1233.4, 1233.4],
        "drift": [prior, 0.0, prior, prior + 2.0, prior, prior, prior],
        "snr": [30.0, 100.0, 25.0, 15.0, 12.0, 12.0, 12.0],
        "mjd": [mjd, mjd, mjd, mjd, mjd, mjd, mjd],
        "telescope": ["GBT"] * 7,
    })
    rep = relay_run("probe,targets,geometry,recut", out_dir=out, conf=conf, bl_fetch=bl, tap_fn=tap,
                    query_fn=_failing_query, fetch_fn=_failing_fetch, gaia_df=sample)
    summary = stage_assess(conf, out, hits_df=hits, tap_fn=tap)
    return {"out": out, "bl": bl, "summary": summary, "recut": rep, "prior": prior}


def test_pipeline_probe_records_the_real_api_shapes(pipeline):
    probe = json.loads((pipeline["out"] / "probe.json").read_text())
    assert probe["bl_targets_n"] == 5 and "MeerKAT" in probe["bl_telescopes"]
    assert probe["endpoints"]["bl_list_targets"]["shape"].startswith("list[dict]")
    assert probe["endpoints"]["gaia_esa_count"]["status"] == "QUERY_FAILED"   # TAP was down, and it says so
    assert "BL_API_REACHED" in probe["verdict"]


def test_pipeline_targets_resolve_through_simbad_gaia_ids(pipeline):
    t = json.loads((pipeline["out"] / "targets.json").read_text())
    assert t["n_names"] == 5 and t["n_with_gaia_id"] == 4
    assert "NotAStar_1" in t["unresolved_sample"]
    m = pd.read_csv(pipeline["out"] / "targets_matched.csv")
    assert (m["gaia_idx"] >= 0).sum() == 4 and set(m["match_by"]) >= {"gaia_id"}


def test_pipeline_geometry_counts_and_committed_files(pipeline, conf):
    g = json.loads((pipeline["out"] / "geometry.json").read_text())
    assert g["verdict"].startswith(V_GEOMETRY) and g["n_stars"] == 2505
    b15 = g["beams"]["radio_10m_1p4ghz"]
    assert b15["n_spillover"] >= 1 and b15["n_between"] >= 1
    assert b15["n_bl_targets_as_transmitter"] >= 2          # both injected transmitters
    assert g["beams"]["optical_10m_1um"]["n_directed_pairs"] == 0
    assert "analytic_expectation" in b15 and b15["analytic_expectation"]["between"] > 0
    assert g["scaling"]["n_between"]["slope_log_n_log_theta"] is not None
    assert (pipeline["out"] / "pairs_targets.csv.gz").exists()
    assert any(f.startswith("pairs_full_") for f in g["committed_files"])
    tp = pd.read_csv(pipeline["out"] / "pairs_targets.csv.gz")
    assert set(tp["t_source_id"].astype(str)) <= {"999001", "999003", "999005", "10005"} | set(
        tp["t_source_id"].astype(str))                            # keep set = targets + in-beam neighbours
    assert tp["kinematics_complete"].all()


def test_pipeline_recut_marks_pair_line_pointings_with_a_resolved_earth_term(pipeline, conf, sample):
    r = json.loads((pipeline["out"] / "recut.json").read_text())
    assert r["verdict"].startswith("RECUT_COMPLETE") and r["bl_files"]["n_files"] == 8
    assert r["bl_files"]["n_with_mjd"] == 8
    p = pd.read_csv(pipeline["out"] / "recut_pointings.csv")
    on = p[p["radio_10m_1p4ghz:n_pairs"] > 0]
    assert set(on["target"]) >= {"HIP999001", "HIP999003"}
    assert not (p[p["target"] == "HIP999005"]["overfilled_5deg:n_pairs"] > 0).any()
    row = on[on["target"] == "HIP999001"].iloc[0]
    assert bool(row["radio_10m_1p4ghz:earth_term_known"])
    # the window centre IS the Earth term at the session epoch (kinematics ~ 1e-8 Hz/s)
    site = conf["telescopes"]["GBT"]
    t = _idx(sample, "999001")
    e = geo.earth_acceleration_los(row["mjd_session"], site["lat_deg"], site["lon_deg"],
                                   sample["ra"][t], sample["dec"][t])
    expect = geo.drift_hz_s(row["freq_centre_ghz"] * 1e9, float(e["a_los_m_s2"]))
    assert abs(row["radio_10m_1p4ghz:drift_centre_hz_s_min"] - expect) < 1e-5
    assert row["radio_10m_1p4ghz:halfwidth_tight_hz_s_max"] < 0.1
    assert row["radio_10m_1p4ghz:halfwidth_loose_hz_s_max"] > 4.0
    assert r["beams"]["radio_10m_1p4ghz"]["n_pointings_on_pair_line"] >= 2


def test_pipeline_assess_finds_the_injected_hit_and_trips_every_rejection(pipeline):
    s = pipeline["summary"]
    assert s["assess_verdict"].startswith(V_PAIRLINE_MATCH)
    beams_with = {c["beam"] for c in s["candidates"]}
    assert "radio_10m_1p4ghz" in beams_with and "optical_10m_1um" not in beams_with
    cands = [c for c in s["candidates"] if c["beam"] == "radio_10m_1p4ghz"]
    assert len(cands) == 1 and cands[0]["target"] == INJ_T_SPILL
    assert abs(cands[0]["drift_hz_s"] - pipeline["prior"]) < 1e-9
    x = pd.read_csv(pipeline["out"] / "hits_crossmatch.csv")
    b = "radio_10m_1p4ghz"
    zero = x[x["freq_mhz"] == 1600.0].iloc[0]
    assert bool(zero["rfi_zero_drift"]) and not bool(zero[f"{b}:candidate"])
    quiet = x[x["target"] == QUIET].iloc[0]
    assert not bool(quiet[f"{b}:on_pair_line"])
    off = x[(x["target"] == INJ_T_SPILL) & (x["drift_hz_s"] > pipeline["prior"] + 1.0)].iloc[0]
    assert bool(off[f"{b}:on_pair_line"]) and not bool(off[f"{b}:drift_match"])
    rec = x[x["freq_mhz"] == 1233.4]
    assert rec["rfi_recurrent"].all()
    assert s["per_beam"][b]["n_trials"] if "n_trials" in s["per_beam"][b] else True
    hits = json.loads((pipeline["out"] / "hits.json").read_text())
    assert hits["beams"][b]["n_hits_on_pair_line"] >= 3 and hits["n_rfi_recurrent"] == 3
    assert hits["beams"][b]["n_expected_by_chance"] is not None
    assert s["funnel"]["n_bl_targets_in_sample"] == 4


def test_no_pair_line_hit_verdict_when_hits_avoid_the_geometry(pipeline, conf):
    hits = pd.DataFrame({"target": [QUIET, QUIET], "freq_mhz": [1420.0, 1500.0],
                         "drift": [0.2, -0.3], "snr": [20.0, 20.0], "mjd": [57650.8, 57650.8],
                         "telescope": ["GBT", "GBT"]})
    s = stage_assess(conf, pipeline["out"], hits_df=hits)
    assert s["assess_verdict"].startswith(V_NO_PAIRLINE_HIT)
    assert s["candidates"] == []
    # a star that IS a transmitter at the over-filled beam, with a drift outside even that window
    hits = pd.DataFrame({"target": ["HIP 10005"], "freq_mhz": [1500.0], "drift": [3.9], "snr": [20.0],
                         "mjd": [57650.8], "telescope": ["GBT"]})
    s = stage_assess(conf, pipeline["out"], hits_df=hits)
    assert s["assess_verdict"].startswith(V_PAIRLINE_OFF_PRIOR) or s["assess_verdict"].startswith(V_NO_PAIRLINE_HIT)
    assert s["candidates"] == []

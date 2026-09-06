"""BAFFLE patch stage: parallax geometry, patch coherence, annual modulation.

Everything is synthetic and offline (conftest blocks sockets); the two fetchers
are stubbed.  The geometry is cross-checked against astropy's own parallax.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.baffle.patch import (
    ARCSEC_PER_RAD,
    FlatLocus,
    baffle_offset_arcsec,
    coherence_test,
    modulation_test,
    neighbour_query,
    neighbour_residuals,
    own_constancy,
    parallax_amplitude_arcsec,
    patch_config,
    predict_coverage,
    profile_shape,
    rho_arcsec,
    rho_grid_from_config,
    run_patch_stage,
    scan_modulation,
    visits_from_epochs,
)

ECL_POLE = (270.0, 66.5607)          # north ecliptic pole in ICRS (deg)
ON_ECLIPTIC = (0.0, 0.0)             # the vernal equinox lies on the ecliptic
CFG = {"patch": {"search_radius_arcmin": 10, "n_permutations": 300,
                 "max_modulation_neighbours": 4}}


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def make_field(rng, n=300, r_max=600.0, ra0=120.0, dec0=-20.0,
               deficit_fn=None, noise=0.03):
    """A Gaia+AllWISE+2MASS neighbour table around X = (ra0, dec0).

    ``deficit_fn(theta_arcsec) -> mag`` is added to W1 and W2 (positive =
    fainter).  Photometric errors are 0.02–0.03 mag; the locus is the module's
    flat fallback, so residuals are pure noise except for the injected deficit.
    """
    theta = r_max * np.sqrt(rng.uniform(0, 1, n))
    pa = rng.uniform(0, 2 * np.pi, n)
    ex, ey = theta * np.sin(pa), theta * np.cos(pa)
    ra = ra0 + ex / 3600.0 / np.cos(np.radians(dec0))
    dec = dec0 + ey / 3600.0
    ks = rng.uniform(9.0, 13.0, n)
    jk = rng.uniform(0.3, 0.8, n)
    d = deficit_fn(theta) if deficit_fn is not None else np.zeros(n)
    w1 = ks - 0.05 + rng.normal(0, noise, n) + d
    w2 = ks - 0.03 + rng.normal(0, noise, n) + d
    return pd.DataFrame({
        "source_id": np.arange(1, n + 1) * 1000, "ra": ra, "dec": dec,
        "pmra": np.zeros(n), "pmdec": np.zeros(n), "parallax": rng.uniform(0.5, 3.0, n),
        "phot_g_mean_mag": ks + 1.5, "j_m": ks + jk, "ks_m": ks,
        "ks_msigcom": np.full(n, 0.02), "w1mpro": w1, "w1mpro_error": np.full(n, 0.025),
        "w2mpro": w2, "w2mpro_error": np.full(n, 0.03), "ph_qual": ["AAAA"] * n,
        "cc_flags": ["0000"] * n, "ext_flag": np.zeros(n, dtype=int)})


def make_epochs(rng, visit_mjds, w1=10.0, w2=10.0, deficit=None, n_exp=12, err=0.02):
    """Single-exposure NEOWISE rows: ``n_exp`` frames inside a day at each visit."""
    rows = []
    for k, t0 in enumerate(visit_mjds):
        dm = float(deficit[k]) if deficit is not None else 0.0
        for _ in range(n_exp):
            rows.append([t0 + rng.uniform(0, 1.0), w1 + dm + rng.normal(0, err), err,
                         w2 + dm + rng.normal(0, err), err, 10, "0000", "AA", 30.0, 0, 1.0, 1.0])
    return pd.DataFrame(rows, columns=["mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro",
                                       "qual_frame", "cc_flags", "ph_qual", "saa_sep",
                                       "moon_masked", "w1rchi2", "w2rchi2"])


def neowise_visit_times(n=20, start=56700.0):
    """Twenty ~6-monthly visits, 2014–2024, with the usual few-day jitter."""
    return start + np.arange(n) * 182.625 + np.linspace(-3, 3, n)


# --------------------------------------------------------------------------
# 1. Parallax geometry
# --------------------------------------------------------------------------

def test_ecliptic_pole_offset_is_a_circle_of_radius_one_au_over_d():
    d = 1000.0
    t = 57000.0 + np.linspace(0, 365.25, 73)
    x, y = baffle_offset_arcsec(t, *ECL_POLE, d)
    r = np.hypot(x, y)
    a = parallax_amplitude_arcsec(d)
    assert a == pytest.approx(ARCSEC_PER_RAD / d)
    # Earth's orbit has e = 0.0167: |r| stays within 2 % of 1 AU / d ...
    assert np.all(np.abs(r / a - 1) < 0.02)
    # ... and the direction sweeps the full circle over the year.
    ang = np.degrees(np.unwrap(np.arctan2(y, x)))
    assert abs(abs(ang[-1] - ang[0]) - 360.0) < 2.0


def test_ecliptic_star_offset_is_a_line():
    d = 500.0
    t = 57000.0 + np.linspace(0, 365.25, 73)
    x, y = baffle_offset_arcsec(t, *ON_ECLIPTIC, d)
    pts = np.stack([x, y], axis=1)
    # Principal axes: the minor axis is < 1 % of the major one.
    u, s, _ = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    assert s[1] / s[0] < 0.01
    assert np.max(np.hypot(x, y)) == pytest.approx(parallax_amplitude_arcsec(d), rel=0.02)


def test_offsets_half_a_year_apart_are_opposite():
    d = 2000.0
    t0 = np.array([56700.0, 57100.0, 59000.0])
    for ra, dec in (ECL_POLE, ON_ECLIPTIC, (200.0, 35.0)):
        x0, y0 = baffle_offset_arcsec(t0, ra, dec, d)
        x1, y1 = baffle_offset_arcsec(t0 + 0.5 * 365.25, ra, dec, d)
        # p(t) + p(t + 0.5 yr) vanishes to within Earth's orbital eccentricity
        # (e = 0.0167: radius 2e and true-anomaly 2e -> up to ~4e = 7 % of the
        # amplitude); a ratio test would be meaningless near the node of an
        # on-ecliptic star's line.
        amp = parallax_amplitude_arcsec(d)
        assert np.all(np.hypot(x0 + x1, y0 + y1) < 0.08 * amp)
        assert np.all(np.hypot(x0, y0) <= 1.02 * amp)


def test_geometry_matches_astropy_parallax_to_one_percent():
    """The tangent-plane formula must reproduce astropy's full 3-D parallax of a
    body at heliocentric distance d, and its amplitude must be 1 AU / d."""
    import astropy.units as u
    from astropy.coordinates import GCRS, SkyCoord, get_body_barycentric
    from astropy.time import Time

    d = 1000.0
    ra, dec = 200.0, 35.0
    times = np.array([56700.0, 56800.0, 56950.0, 57050.0])
    px, py = baffle_offset_arcsec(times, ra, dec, d)
    for k, mjd in enumerate(times):
        t = Time(mjd, format="mjd", scale="utc")
        sun = get_body_barycentric("sun", t)
        far = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=1e9 * u.au, frame="icrs")
        body = SkyCoord(far.cartesian * (d / 1e9) + sun, frame="icrs")
        g_body = body.transform_to(GCRS(obstime=t))
        g_far = far.transform_to(GCRS(obstime=t))
        dra, ddec = g_far.spherical_offsets_to(g_body)
        ax, ay = dra.to_value(u.arcsec), ddec.to_value(u.arcsec)
        amp = np.hypot(ax, ay)
        assert amp == pytest.approx(np.hypot(px[k], py[k]), rel=0.01)
        assert np.hypot(ax - px[k], ay - py[k]) < 0.01 * amp
    # The semi-major axis of the ellipse over a year is 1 AU / d.
    year = 57000.0 + np.linspace(0, 365.25, 73)
    qx, qy = baffle_offset_arcsec(year, ra, dec, d)
    assert np.max(np.hypot(qx, qy)) == pytest.approx(ARCSEC_PER_RAD / d, rel=0.02)


def test_predict_coverage_top_hat_and_annual_switch():
    d, R = 1000.0, 1.0
    rho = rho_arcsec(d, R)
    assert rho == pytest.approx(206.26, rel=1e-3)
    t = neowise_visit_times()
    # X itself: at R = 2 AU it is always covered (1 AU orbit inside a 2 AU screen).
    assert predict_coverage(0.0, 0.0, t, *ECL_POLE, d, 2.0).all()
    # A neighbour far outside rho + pi_b is never covered.
    assert not predict_coverage(0.0, 3 * rho, t, *ECL_POLE, d, R).any()
    # A neighbour placed where the centre sits at t[0] is covered then and not
    # half a year later.
    px, py = baffle_offset_arcsec(t, *ECL_POLE, d)
    cov = predict_coverage(px[0], py[0], t, *ECL_POLE, d, R)
    assert cov[0] and not cov[1] and cov.sum() >= 6 and (~cov).sum() >= 6


# --------------------------------------------------------------------------
# 2. Patch coherence and profile shape
# --------------------------------------------------------------------------

def _residuals(field, ra0=120.0, dec0=-20.0, cfg=None):
    return neighbour_residuals(field, ra0, dec0, patch_config(cfg or CFG))


def test_top_hat_field_is_a_coherent_patch(tmp_path):
    rng = np.random.default_rng(1)
    field = make_field(rng, n=400, deficit_fn=lambda th: np.where(th < 200.0, 0.4, 0.0))
    res = _residuals(field)
    grid = rho_grid_from_config(patch_config(CFG))
    coh = coherence_test(res["theta_arcsec"], res["deficit"], grid)
    assert coh["coherence_p"] < 1e-6
    assert abs(coh["best_rho_arcsec"] / 200.0 - 1) < 0.2
    shape = profile_shape(res["theta_arcsec"], -res["resid"], res["resid_err"], grid)
    assert shape["shape"] == "tophat"
    assert shape["tophat_amp_mag"] == pytest.approx(0.4, abs=0.05)

    cand = pd.DataFrame({"source_id": [7], "ra": [120.0], "dec": [-20.0]})
    out = run_patch_stage(cand, tmp_path, CFG, neighbour_fetcher=lambda ra, dec, r: field,
                          neowise_fetcher=lambda *a, **k: pd.DataFrame())
    tab = pd.read_csv(tmp_path / "patches.csv")
    assert tab.loc[0, "patch_verdict"] == "COHERENT_PATCH"
    assert abs(tab.loc[0, "best_rho_arcsec"] / 200.0 - 1) < 0.2
    assert tab.loc[0, "profile_shape"] == "tophat"
    assert out["n_coherent_patch"] == 1 and out["objects"][0]["source_id"] == 7
    prof = json.loads((tmp_path / "patch_profiles.json").read_text())
    assert prof["7"]["verdict"] == "COHERENT_PATCH"
    assert len(prof["7"]["profile_bins"]) >= 3 and len(prof["7"]["neighbours"]) == 400


def test_smooth_halo_is_not_coherent(tmp_path):
    rng = np.random.default_rng(2)
    field = make_field(rng, n=400, deficit_fn=lambda th: 1.0 * (20.0 / np.maximum(th, 20.0)))
    res = _residuals(field)
    grid = rho_grid_from_config(patch_config(CFG))
    shape = profile_shape(res["theta_arcsec"], -res["resid"], res["resid_err"], grid)
    assert shape["shape"] == "smooth"
    cand = pd.DataFrame({"source_id": [8], "ra": [120.0], "dec": [-20.0]})
    run_patch_stage(cand, tmp_path, CFG, neighbour_fetcher=lambda ra, dec, r: field,
                    neowise_fetcher=lambda *a, **k: pd.DataFrame())
    tab = pd.read_csv(tmp_path / "patches.csv")
    assert tab.loc[0, "profile_shape"] == "smooth"
    assert tab.loc[0, "patch_verdict"] == "NOT_COHERENT"


def test_no_deficit_field_is_not_coherent(tmp_path):
    rng = np.random.default_rng(3)
    field = make_field(rng, n=300)
    cand = pd.DataFrame({"source_id": [9], "ra": [120.0], "dec": [-20.0]})
    out = run_patch_stage(cand, tmp_path, CFG, neighbour_fetcher=lambda ra, dec, r: field,
                          neowise_fetcher=lambda *a, **k: pd.DataFrame())
    tab = pd.read_csv(tmp_path / "patches.csv")
    assert tab.loc[0, "patch_verdict"] == "NOT_COHERENT"
    assert tab.loc[0, "n_deficit_total"] <= 2
    assert out["n_not_coherent"] == 1 and out["objects"] == []


def test_too_few_neighbours(tmp_path):
    rng = np.random.default_rng(4)
    field = make_field(rng, n=5)
    cand = pd.DataFrame({"source_id": [10], "ra": [120.0], "dec": [-20.0]})
    out = run_patch_stage(cand, tmp_path, CFG, neighbour_fetcher=lambda ra, dec, r: field,
                          neowise_fetcher=lambda *a, **k: pd.DataFrame())
    tab = pd.read_csv(tmp_path / "patches.csv")
    assert tab.loc[0, "patch_verdict"] == "INSUFFICIENT_NEIGHBOURS"
    assert out["n_insufficient_neighbours"] == 1


def test_isolated_deficit_needs_clean_field_and_constant_own_series(tmp_path):
    rng = np.random.default_rng(5)
    field = make_field(rng, n=300)
    t = neowise_visit_times()
    own = make_epochs(rng, t, w1=9.6, w2=9.6)       # 0.4 mag below its AllWISE-free photosphere
    cand = pd.DataFrame({"source_id": [11], "ra": [120.0], "dec": [-20.0],
                         "w1mpro": [9.6], "w2mpro": [9.6]})
    out = run_patch_stage(cand, tmp_path, CFG, neighbour_fetcher=lambda ra, dec, r: field,
                          neowise_fetcher=lambda ra, dec, pmra=0, pmdec=0: own)
    tab = pd.read_csv(tmp_path / "patches.csv")
    assert tab.loc[0, "own_neowise_n_visits"] == 20
    assert tab.loc[0, "own_flat_chi2"] < 3.0
    assert abs(tab.loc[0, "own_offset_from_allwise"]) < 0.05
    assert bool(tab.loc[0, "own_constant"])
    assert tab.loc[0, "patch_verdict"] == "ISOLATED_DEFICIT"
    assert out["n_own_constant"] == 1 and out["n_isolated_deficit"] == 1
    assert out["objects"][0]["patch_verdict"] == "ISOLATED_DEFICIT"


def test_candidate_itself_is_excluded_from_its_own_patch():
    rng = np.random.default_rng(6)
    field = make_field(rng, n=100)
    x = field.iloc[:1].copy()
    x["source_id"] = 12
    x["ra"], x["dec"] = 120.0, -20.0
    x["w1mpro"] = x["ks_m"] - 0.05 - 0.6
    x["w2mpro"] = x["ks_m"] - 0.03 - 0.6
    res = neighbour_residuals(pd.concat([x, field]), 120.0, -20.0, patch_config(CFG),
                              exclude_source_id=12)
    assert 12 not in set(res["source_id"])
    assert len(res) == 100


def test_locus_object_is_used_and_fallback_is_flat():
    class Locus:
        calls = []

        def predict(self, jk, lum_class, band):
            Locus.calls.append((lum_class, band))
            return (0.30, 0.05)         # a locus 0.25 mag above the flat fallback

    rng = np.random.default_rng(7)
    field = make_field(rng, n=50)
    r_fallback = neighbour_residuals(field, 120.0, -20.0, patch_config(CFG))
    r_locus = neighbour_residuals(field, 120.0, -20.0, patch_config(CFG), locus=Locus())
    assert Locus.calls and {b for _, b in Locus.calls} == {"W1", "W2"}
    assert np.allclose(r_locus["resid_w1"] - r_fallback["resid_w1"], -0.25, atol=1e-9)
    assert FlatLocus().predict(0.5, "dwarf", "W1") == (0.05, 0.06)


# --------------------------------------------------------------------------
# 3. Annual modulation
# --------------------------------------------------------------------------

def test_modulation_test_finds_an_injected_annual_switch():
    rng = np.random.default_rng(8)
    d, R = 1000.0, 1.0
    t = neowise_visit_times()
    px, py = baffle_offset_arcsec(t, *ECL_POLE, d)
    theta = (px[0], py[0])                     # neighbour under the centre at the first visit
    cov = predict_coverage(theta[0], theta[1], t, *ECL_POLE, d, R)
    assert cov.sum() >= 6 and (~cov).sum() >= 6
    epochs = make_epochs(rng, t, deficit=np.where(cov, 0.3, 0.0))
    visits = visits_from_epochs(epochs)
    assert len(visits) == 20
    mt = modulation_test(visits, cov, n_perm=2000, rng=1)
    assert mt["status"] == "OK"
    assert mt["sig"] > 5.0
    assert mt["diff"] == pytest.approx(0.3, abs=0.03)
    assert mt["null_p"] < 1e-3
    # The raw single-exposure frame is accepted too.
    assert modulation_test(epochs, cov, n_perm=100, rng=1)["sig"] > 5.0

    # A constant series shows nothing.
    flat = visits_from_epochs(make_epochs(rng, t))
    mt0 = modulation_test(flat, cov, n_perm=500, rng=2)
    assert abs(mt0["sig"]) < 3.0 and mt0["null_p"] > 0.01
    # And a schedule without contrast is reported as such, not as a detection.
    assert modulation_test(flat, np.ones(20, bool))["status"] == "NO_CONTRAST"


def test_scan_modulation_recovers_d_and_R():
    rng = np.random.default_rng(9)
    d, R = 1000.0, 1.0
    t = neowise_visit_times()
    px, py = baffle_offset_arcsec(t, *ECL_POLE, d)
    series = []
    # Three switching neighbours at different phases, plus one at 550" that a
    # (1000 AU, 1 AU) screen never reaches but a closer / larger one would: the
    # constant series it shows is what breaks the d-R degeneracy.
    for k, scale in enumerate((1.0, 0.7, -0.6, 2.67)):
        th = (scale * px[0], scale * py[0])
        cov = predict_coverage(th[0], th[1], t, *ECL_POLE, d, R)
        ep = make_epochs(rng, t + rng.uniform(-2, 2, t.size), deficit=np.where(cov, 0.3, 0.0))
        series.append({"source_id": k, "theta_ra": th[0], "theta_dec": th[1],
                       "visits": visits_from_epochs(ep)})
    assert not predict_coverage(series[3]["theta_ra"], series[3]["theta_dec"], t, *ECL_POLE, d, R).any()
    mod = scan_modulation(series, *ECL_POLE, [200, 500, 1000, 2000, 5000], [1, 2, 5],
                          n_perm=1000, seed=3)
    assert mod["best_d_au"] == 1000.0 and mod["best_R_au"] == 1.0
    assert mod["modulation_sig"] > 5.0
    assert mod["modulation_null_p"] < 5e-3
    assert (1000.0, 1.0) in mod["degenerate_grid"]
    assert 0.0 <= mod["alternation_control_sig"]
    # Without the far neighbour, (200, 1) and (1000, 1) predict the same
    # alternating schedule for everything inside rho: the module says so.
    mod3 = scan_modulation(series[:3], *ECL_POLE, [200, 500, 1000, 2000, 5000], [1, 2, 5],
                           n_perm=200, seed=3)
    assert (1000.0, 1.0) in mod3["degenerate_grid"] and len(mod3["degenerate_grid"]) > 1


def test_stage_reports_modulated_field(tmp_path):
    """Whole pipeline: a top-hat field whose edge neighbours switch annually."""
    rng = np.random.default_rng(10)
    ra0, dec0 = ECL_POLE
    d, R = 1000.0, 1.0
    rho = rho_arcsec(d, R)
    field = make_field(rng, n=300, ra0=ra0, dec0=dec0,
                       deficit_fn=lambda th: np.where(th < rho, 0.4, 0.0))
    t = neowise_visit_times()
    from seti.baffle.patch import tangent_offsets_arcsec

    def nw(ra, dec, pmra=0.0, pmdec=0.0):
        ex, ey = tangent_offsets_arcsec(ra0, dec0, [ra], [dec])
        cov = predict_coverage(float(ex[0]), float(ey[0]), t, ra0, dec0, d, R)
        return make_epochs(rng, t, w1=10.0, w2=10.0, deficit=np.where(cov, 0.4, 0.0))

    cand = pd.DataFrame({"source_id": [13], "ra": [ra0], "dec": [dec0],
                         "w1mpro": [10.4], "w2mpro": [10.4]})
    cfg = {"patch": {"search_radius_arcmin": 10, "n_permutations": 1500,
                     "max_modulation_neighbours": 6, "d_grid_au": [500, 1000, 2000],
                     "R_grid_au": [1, 2]}}
    out = run_patch_stage(cand, tmp_path, cfg, neighbour_fetcher=lambda ra, dec, r: field,
                          neowise_fetcher=nw)
    tab = pd.read_csv(tmp_path / "patches.csv")
    assert tab.loc[0, "patch_verdict"] == "MODULATED"
    assert tab.loc[0, "modulation_sig"] > 5.0 and tab.loc[0, "modulation_null_p"] < 1e-3
    assert out["n_modulated"] == 1
    prof = json.loads((tmp_path / "patch_profiles.json").read_text())
    assert prof["13"]["modulation_grid"] and prof["13"]["neighbour_visits"]
    # The true screen is among the (d, R) the sampling cannot tell apart.
    assert [1000.0, 1.0] in prof["13"]["modulation_degenerate_grid"]
    assert tab.loc[0, "best_d_au"] in (500.0, 1000.0) and tab.loc[0, "best_R_au"] == 1.0


def test_own_constancy_flags_a_variable_or_shifted_star():
    rng = np.random.default_rng(11)
    t = neowise_visit_times()
    v = visits_from_epochs(make_epochs(rng, t, w1=10.0, w2=10.0))
    ok = own_constancy(v, 10.0, 10.0)
    assert ok["constant"] and ok["flat_chi2"] < 3.0
    shifted = own_constancy(v, 10.5, 10.5)       # NEOWISE 0.5 mag brighter than AllWISE
    assert not shifted["constant"] and shifted["offset_from_allwise"] == pytest.approx(-0.5, abs=0.02)
    var = visits_from_epochs(make_epochs(rng, t, deficit=rng.normal(0, 0.2, t.size)))
    assert not own_constancy(var, 10.0, 10.0)["constant"]
    assert own_constancy(v.iloc[:3], 10.0, 10.0)["constant"] is False


# --------------------------------------------------------------------------
# 4. Fetch failures, config, and the query string
# --------------------------------------------------------------------------

def test_raising_fetcher_is_recorded_and_stage_continues(tmp_path):
    rng = np.random.default_rng(12)
    field = make_field(rng, n=200)

    def bad_neighbours(ra, dec, r):
        if ra > 100:
            raise RuntimeError("archive down")
        return field

    def bad_neowise(*a, **k):
        raise TimeoutError("irsa")

    cand = pd.DataFrame({"source_id": [1, 2], "ra": [120.0, 50.0], "dec": [-20.0, -20.0]})
    out = run_patch_stage(cand, tmp_path, CFG, neighbour_fetcher=bad_neighbours,
                          neowise_fetcher=bad_neowise)
    assert out["n_assessed"] == 2
    assert out["n_fetch_failed"] == 2          # neighbour fetch failed for 1; NEOWISE failed for both
    tab = pd.read_csv(tmp_path / "patches.csv")
    assert list(tab["status"]) == ["FETCH_FAILED", "OK"]
    assert list(tab["patch_verdict"]) == ["FETCH_FAILED", "NOT_COHERENT"]
    assert list(tab["own_status"]) == ["FETCH_FAILED", "FETCH_FAILED"]
    assert (tmp_path / "patch_profiles.json").exists()


def test_config_defaults_and_max_objects(tmp_path):
    pc = patch_config({})
    assert pc["search_radius_arcmin"] == 10 and pc["min_neighbours"] == 8
    assert pc["deficit_sig"] == 3.0 and pc["deficit_mag"] == 0.2 and pc["max_objects"] == 200
    assert pc["d_grid_au"] == [100, 200, 500, 1000, 2000, 5000, 10000]
    assert pc["R_grid_au"] == [1, 2, 5]
    grid = rho_grid_from_config(pc)
    assert grid[0] == pytest.approx(20.0) and grid[-1] == pytest.approx(600.0)
    assert patch_config({"patch": {"max_objects": 3}})["max_objects"] == 3
    assert patch_config({"patch": {"rho_grid_arcsec": [30, 100, 900]}})["rho_grid_arcsec"] == [30, 100, 900]
    assert list(rho_grid_from_config(patch_config({"patch": {"rho_grid_arcsec": [30, 100, 900]}}))) == [30, 100]

    rng = np.random.default_rng(13)
    field = make_field(rng, n=100)
    cand = pd.DataFrame({"source_id": [1, 2, 3], "ra": [120.0] * 3, "dec": [-20.0] * 3})
    out = run_patch_stage(cand, tmp_path, {}, neighbour_fetcher=lambda ra, dec, r: field,
                          neowise_fetcher=lambda *a, **k: None, max_objects=2)
    assert out["n_assessed"] == 2 and len(pd.read_csv(tmp_path / "patches.csv")) == 2


def test_neighbour_query_is_a_gaia_archive_cone():
    q = neighbour_query(120.0, -20.0, 10.0)
    assert "gaiadr3.gaia_source" in q and "gaiadr1.allwise_original_valid" in q
    assert "gaiadr1.tmass_original_valid" in q and "allwise_best_neighbour" in q
    assert "CIRCLE('ICRS', 120.0000000, -20.0000000, 0.1666667)" in q

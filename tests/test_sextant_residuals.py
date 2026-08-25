"""Offline tests for SEXTANT's residual computation and screen.

Everything here runs with no network, on synthetic observations built from a
numerically integrated orbit with a **known injected non-gravitational
acceleration**.  That is not a convenience --- it is the only honest way to test
this estimator.  There is no minor planet whose true ``A2`` is known
independently of an orbit fit, so a test against real data would be a test
against somebody else's fit; and the failure this channel most has to fear is an
estimator that returns a confident number for a signal that is not there.  So:
inject a signal and require it back, inject nothing and require nothing back.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from seti.sextant import residuals as R
from seti.sextant import screen as S

MU = R.GM_SUN_AU3_DAY2
OBLIQUITY = math.radians(23.439291)
_ROT = np.array([[1.0, 0.0, 0.0],
                 [0.0, math.cos(OBLIQUITY), -math.sin(OBLIQUITY)],
                 [0.0, math.sin(OBLIQUITY), math.cos(OBLIQUITY)]])
JD0 = 2456900.0          # 2014-08-22, inside Gaia's SSO window


# ---------------------------------------------------------------------------
# Synthetic sky
# ---------------------------------------------------------------------------
def _ecl_to_eq(v):
    return np.asarray(v) @ _ROT.T


def kepler_state(a, e, inc_deg, node_deg, argp_deg, m0_deg):
    """Heliocentric equatorial state from classical elements."""
    n = math.sqrt(MU / a ** 3)
    m = math.radians(m0_deg)
    ecc = m
    for _ in range(80):
        ecc = ecc - (ecc - e * math.sin(ecc) - m) / (1.0 - e * math.cos(ecc))
    xv = a * (math.cos(ecc) - e)
    yv = a * math.sqrt(1.0 - e * e) * math.sin(ecc)
    r = a * (1.0 - e * math.cos(ecc))
    vx = -a * a * n * math.sin(ecc) / r
    vy = a * a * n * math.sqrt(1.0 - e * e) * math.cos(ecc) / r
    i, om, w = math.radians(inc_deg), math.radians(node_deg), math.radians(argp_deg)

    def orient(x, y):
        x1 = x * math.cos(w) - y * math.sin(w)
        y1 = x * math.sin(w) + y * math.cos(w)
        return np.array([x1 * math.cos(om) - y1 * math.cos(i) * math.sin(om),
                         x1 * math.sin(om) + y1 * math.cos(i) * math.cos(om),
                         y1 * math.sin(i)])

    return np.concatenate([_ecl_to_eq(orient(xv, yv)), _ecl_to_eq(orient(vx, vy))])


def _accel(state, a2):
    r, v = state[:3], state[3:]
    rn = float(np.linalg.norm(r))
    acc = -MU * r / rn ** 3
    if a2:
        h = np.cross(r, v)
        t_hat = np.cross(h / np.linalg.norm(h), r / rn)
        acc = acc + a2 * (1.0 / rn) ** 2 * t_hat
    return np.concatenate([v, acc])


def integrate(state0, t0, t1, h, a2):
    """Fixed-step RK4.  Truth, against which the estimator is scored."""
    n = int(math.ceil((t1 - t0) / h))
    h = (t1 - t0) / n
    ts = t0 + h * np.arange(n + 1)
    ss = np.empty((n + 1, 6))
    ss[0] = state0
    s = np.array(state0, dtype=float)
    for k in range(n):
        k1 = _accel(s, a2)
        k2 = _accel(s + 0.5 * h * k1, a2)
        k3 = _accel(s + 0.5 * h * k2, a2)
        k4 = _accel(s + h * k3, a2)
        s = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        ss[k + 1] = s
    return ts, ss


def hermite(ts, ss):
    """Cubic Hermite interpolant of a state history; exact to ~mm at h = 0.2 d."""
    h = ts[1] - ts[0]

    def f(jd):
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        i = np.clip(((jd - ts[0]) / h).astype(int), 0, ts.size - 2)
        u = (jd - ts[i]) / h
        p0, p1 = ss[i, :3], ss[i + 1, :3]
        v0, v1 = ss[i, 3:], ss[i + 1, 3:]
        u2, u3 = u * u, u * u * u
        pos = ((2 * u3 - 3 * u2 + 1)[:, None] * p0 + (u3 - 2 * u2 + u)[:, None] * (h * v0)
               + (-2 * u3 + 3 * u2)[:, None] * p1 + (u3 - u2)[:, None] * (h * v1))
        vel = ((6 * u2 - 6 * u)[:, None] * p0 + (3 * u2 - 4 * u + 1)[:, None] * (h * v0)
               + (-6 * u2 + 6 * u)[:, None] * p1 + (3 * u2 - 2 * u)[:, None] * (h * v1)) / h
        return np.column_stack([pos, vel])

    return f


def observer_state(t):
    """A Gaia-like observer: circular ecliptic orbit at 1.01 au.

    Only its position and velocity are ever used --- which is the point of the
    archive carrying ``x_gaia..vz_gaia``: the channel needs no model of Gaia's
    real orbit, and this stand-in exercises the light-time and aberration
    machinery at the right magnitude without pretending to be one.
    """
    ang = 2.0 * math.pi * (np.asarray(t, dtype=float) - JD0) / 365.25
    pos = _ecl_to_eq(np.column_stack([1.01 * np.cos(ang), 1.01 * np.sin(ang),
                                      np.zeros_like(ang)]))
    vel = _ecl_to_eq(np.column_stack([-1.01 * np.sin(ang), 1.01 * np.cos(ang),
                                      np.zeros_like(ang)]) * 2 * math.pi / 365.25)
    return pos, vel


SYNTHETIC_SOURCE = R.OrbitSource(
    name="synthetic_independent", provider="test fixture",
    dynamical_model="two_body", solution_reference="known injected A2",
    gaia_sso_astrometry_in_fit=False, nongrav_parameters_fitted=False)
TEST_CONV = R.Conventions(epoch=R.EpochConvention("epoch", "TDB", 0.0),
                          resolved_by="FIXED_BY_CONSTRUCTION_IN_TEST")


def make_observations(a2_true, *, a=1.6, e=0.2, n_transits=45, seed=7,
                      sigma_al=0.4, sigma_ac=4.0, sigma_sys=0.3, span=1900.0,
                      reject_fraction=0.0):
    """Synthetic Gaia SSO observations with a known injected transverse A2."""
    s0 = kepler_state(a, e, 5.0, 40.0, 30.0, 10.0)
    ts, ss = integrate(s0, JD0, JD0 + span + 150.0, 0.2, a2_true)
    truth = hermite(ts, ss)
    rng = np.random.default_rng(seed)
    t_tr = np.sort(rng.uniform(JD0 + 5.0, JD0 + span, n_transits))
    keys = ("ra", "dec", "ra_error_random", "dec_error_random",
            "ra_dec_correlation_random", "ra_error_systematic",
            "dec_error_systematic", "ra_dec_correlation_systematic", "epoch",
            "epoch_err", "position_angle_scan", "transit_id", "is_rejected",
            "x_gaia", "y_gaia", "z_gaia", "vx_gaia", "vy_gaia", "vz_gaia")
    rows: dict[str, list] = {k: [] for k in keys}
    for ti, t_start in enumerate(t_tr):
        pa = rng.uniform(0.0, 360.0)
        sys_al = rng.normal(0.0, sigma_sys)
        rejected = rng.random() < reject_fraction
        for ccd in range(9):
            t = t_start + ccd * 4.85 / 86400.0
            pos, vel = observer_state(np.array([t]))
            ap = R.apparent_direction(truth, np.array([t]), pos, vel, conv=TEST_CONV)
            u = ap["u"][0]
            east, north = R.tangent_basis(u[None, :])
            e_al, e_ac = R.scan_basis(np.array([pa]))
            d_al = rng.normal(0.0, sigma_al) + sys_al
            d_ac = rng.normal(0.0, sigma_ac)
            de = d_al * e_al[0, 0] + d_ac * e_ac[0, 0]
            dn = d_al * e_al[0, 1] + d_ac * e_ac[0, 1]
            up = u + (de * east[0] + dn * north[0]) / R.MAS_PER_RAD
            up = up / np.linalg.norm(up)
            ra, dec = R.radec_from_unit(up[None, :])
            cr = (np.outer(e_al[0], e_al[0]) * sigma_al ** 2
                  + np.outer(e_ac[0], e_ac[0]) * sigma_ac ** 2)
            cs = np.outer(e_al[0], e_al[0]) * sigma_sys ** 2
            rows["ra"].append(ra[0])
            rows["dec"].append(dec[0])
            rows["ra_error_random"].append(math.sqrt(cr[0, 0]))
            rows["dec_error_random"].append(math.sqrt(cr[1, 1]))
            rows["ra_dec_correlation_random"].append(
                cr[0, 1] / math.sqrt(cr[0, 0] * cr[1, 1]))
            rows["ra_error_systematic"].append(math.sqrt(max(cs[0, 0], 1e-12)))
            rows["dec_error_systematic"].append(math.sqrt(max(cs[1, 1], 1e-12)))
            rows["ra_dec_correlation_systematic"].append(
                cs[0, 1] / math.sqrt(max(cs[0, 0] * cs[1, 1], 1e-24)))
            rows["epoch"].append(t)
            rows["epoch_err"].append(0.0)
            rows["position_angle_scan"].append(pa)
            rows["transit_id"].append(float(ti))
            rows["is_rejected"].append(rejected)
            for nm, val in zip(("x_gaia", "y_gaia", "z_gaia"), pos[0], strict=True):
                rows[nm].append(val)
            for nm, val in zip(("vx_gaia", "vy_gaia", "vz_gaia"), vel[0], strict=True):
                rows[nm].append(val)
    cols = {k: np.asarray(v) for k, v in rows.items()}
    predict = lambda jd: R.propagate_two_body(  # noqa: E731
        s0, np.atleast_1d(np.asarray(jd, dtype=float)) - JD0)
    return cols, predict, s0


def build_series(a2_true, **kw):
    cols, predict, s0 = make_observations(a2_true, **kw)
    return R.compute_residuals(cols, predict, SYNTHETIC_SOURCE, conv=TEST_CONV,
                               state0=s0, jd0=JD0, allow_approximate=True,
                               key="synthetic"), cols, predict, s0


# ---------------------------------------------------------------------------
# 1. The astrometric chain, term by term, at its documented magnitude
# ---------------------------------------------------------------------------
def test_two_body_propagator_returns_to_start_after_one_period():
    s0 = kepler_state(2.5, 0.15, 8.0, 100.0, 40.0, 0.0)
    period = 2.0 * math.pi * math.sqrt(2.5 ** 3 / MU)
    back = R.propagate_two_body(s0, np.array([period]))[0]
    assert np.allclose(back, s0, atol=1e-9)


def test_two_body_propagator_agrees_with_numerical_integration():
    """The propagator and the RK4 truth must agree far below the injected signal.

    They are used for different jobs --- the propagator supplies the nuisance
    partials, the integrator supplies the truth --- so an undetected disagreement
    between them would show up as a fake acceleration.
    """
    s0 = kepler_state(1.6, 0.2, 5.0, 40.0, 30.0, 10.0)
    ts, ss = integrate(s0, JD0, JD0 + 2000.0, 0.2, 0.0)
    ref = hermite(ts, ss)
    t = JD0 + np.array([500.0, 1000.0, 1500.0, 2000.0])
    kep = R.propagate_two_body(s0, t - JD0)[:, :3]
    err = np.linalg.norm(kep - ref(t)[:, :3], axis=1)
    assert err.max() < 1e-9, err          # au; the injected signal is ~1e-6 au


def test_stumpff_functions_are_continuous_through_zero():
    psi = np.array([-1e-3, -1e-9, 0.0, 1e-9, 1e-3])
    c2, c3 = R._stumpff(psi)
    assert np.all(np.abs(c2 - 0.5) < 1e-3)
    assert np.all(np.abs(c3 - 1.0 / 6.0) < 1e-3)
    assert np.all(np.diff(c2) < 0)        # c2 decreases with psi


def test_stellar_aberration_is_twenty_arcsec_and_relativistic():
    """|v|/c ~ 1e-4 gives ~20.5 arcsec; the second-order term is ~2 mas.

    Both matter: the first order is twenty thousand times the signal, and the
    second order is still above the noise floor, which is why the classical
    formula is not used.
    """
    # Small angles are measured as chord lengths, never with arccos: at 2e-8 rad
    # the cosine differs from 1 by 2e-16, which is one machine epsilon, and
    # arccos returns noise.  This bit the first version of this very test.
    u = np.array([[1.0, 1.0, 0.0]]) / math.sqrt(2.0)
    v = np.array([[0.0, 0.0173, 0.0]])    # ~30 km/s in au/day
    out = R.stellar_aberration(u, v)
    ang = float(np.linalg.norm(out[0] - u[0])) * R.ARCSEC_PER_RAD
    assert 13.0 < ang < 16.0, ang         # 20.5 arcsec x sin(45 deg)
    classical = u + v / R.C_AU_PER_DAY
    classical = classical / np.linalg.norm(classical)
    diff_mas = float(np.linalg.norm(out[0] - classical[0])) * R.MAS_PER_RAD
    assert 0.3 < diff_mas < 20.0, diff_mas


def test_solar_deflection_is_four_mas_at_ninety_degrees_elongation():
    """The textbook number, reproduced by the finite-distance formula.

    ~4.07 mas at 90 degrees, rising as cot(theta/2), so ~9.8 mas at Gaia's
    45-degree solar aspect angle.  Ten times the per-observation precision, and
    smoothly varying along an arc, which is the shape an acceleration has.
    """
    obs = np.array([[1.0, 0.0, 0.0]])
    far = np.array([[0.0, 1.0, 0.0]]) * 1e6      # effectively at infinity
    u = (far - obs)
    u = u / np.linalg.norm(u, axis=1)[:, None]
    out = R.light_deflection(u, obs, far)
    mas = float(np.linalg.norm(out[0] - u[0])) * R.MAS_PER_RAD
    assert 3.8 < mas < 4.4, mas


def test_light_time_solution_matches_range_over_c():
    s0 = kepler_state(2.5, 0.1, 5.0, 0.0, 0.0, 0.0)
    state = lambda jd: R.propagate_two_body(s0, np.asarray(jd, float) - JD0)  # noqa: E731
    obs = np.array([[1.0, 0.0, 0.0]])
    tau, r_t, _ = R.solve_light_time(state, np.array([JD0]), obs)
    dist = float(np.linalg.norm(r_t[0] - obs[0]))
    assert abs(tau[0] - dist / R.C_AU_PER_DAY) < 1e-10


def test_tcb_to_tdb_is_seventeen_seconds_in_2015():
    """The trap that would look like a population-wide detection.

    TCB runs fast on TDB by 1.55e-8, which is 18.8 s by mid-2015.  At a main-belt sky
    rate that is ~0.14 arcsec of along-track offset on EVERY object, proportional
    to sky rate --- exactly the shape `fit_common_time_offset` is built to name.
    """
    jd = 2457204.5      # 2015-07-01
    tdb = R.epoch_to_jd_tdb(np.array([jd]), R.EpochConvention("epoch", "TCB", 0.0))
    secs = (jd - float(tdb[0])) * 86400.0
    assert 18.0 < secs < 19.5, secs


def test_utc_conversion_uses_the_leap_second_table_and_refuses_outside_it():
    conv = R.EpochConvention("epoch_utc", "UTC", 0.0)
    inside = R.epoch_to_jd_tdb(np.array([2457204.5]), conv)   # 2015-07-01
    secs = (float(inside[0]) - 2457204.5) * 86400.0
    assert 67.0 < secs < 70.0, secs        # 36 leap + 32.184
    outside = R.epoch_to_jd_tdb(np.array([2440000.5]), conv)
    assert not np.isfinite(outside[0])


# ---------------------------------------------------------------------------
# 2. The scan frame
# ---------------------------------------------------------------------------
def test_scan_basis_is_orthonormal_in_both_handednesses():
    pa = np.linspace(0.0, 350.0, 36)
    for n2e in (True, False):
        e_al, e_ac = R.scan_basis(pa, north_to_east=n2e)
        assert np.allclose(np.sum(e_al * e_al, axis=1), 1.0)
        assert np.allclose(np.sum(e_ac * e_ac, axis=1), 1.0)
        assert np.allclose(np.sum(e_al * e_ac, axis=1), 0.0, atol=1e-12)


def test_scan_convention_is_recovered_from_the_covariance_alone():
    """No ephemeris, no assumption: the error model itself names the convention.

    Gaia's covariance is anisotropic with the minor axis along-scan, so the angle
    between the covariance's minor eigenvector and the PA-implied direction
    settles the handedness.  A reflected convention does not merely flip a sign,
    it mixes AL into AC, so getting this wrong destroys the quantity being
    measured and it must not be a guess.
    """
    rng = np.random.default_rng(3)
    pa = rng.uniform(0.0, 360.0, 400)
    e_al, e_ac = R.scan_basis(pa, north_to_east=True)
    sa, sd, rho = [], [], []
    for k in range(pa.size):
        c = np.outer(e_al[k], e_al[k]) * 0.4 ** 2 + np.outer(e_ac[k], e_ac[k]) * 4.0 ** 2
        sa.append(math.sqrt(c[0, 0]))
        sd.append(math.sqrt(c[1, 1]))
        rho.append(c[0, 1] / math.sqrt(c[0, 0] * c[1, 1]))
    out = R.verify_scan_convention(np.array(sa), np.array(sd), np.array(rho), pa)
    assert out["verdict"] == "OK"
    assert out["north_to_east"] is True
    assert out["median_misalignment_deg"]["north_to_east"] < 1e-6
    assert out["median_misalignment_deg"]["north_to_west"] > 10.0
    assert out["median_axis_ratio"] > 5.0


def test_scan_convention_reports_failure_when_the_errors_are_isotropic():
    n = 300
    rng = np.random.default_rng(4)
    pa = rng.uniform(0.0, 360.0, n)
    out = R.verify_scan_convention(np.ones(n), np.ones(n), np.zeros(n), pa)
    assert out["verdict"] == "NEITHER_CONVENTION_MATCHES"


# ---------------------------------------------------------------------------
# 3. Independence: the requirement the API makes it impossible to skip
# ---------------------------------------------------------------------------
def test_gaias_own_orbit_solution_is_refused_with_no_override():
    src = R.KNOWN_ORBIT_SOURCES["gaia_sso_orbits"]
    assert src.independence == R.CIRCULAR
    for kw in ({}, {"allow_partial": True}, {"allow_partial": True,
                                             "allow_approximate": True}):
        with pytest.raises(R.CircularOrbitSourceError):
            R.require_independent_prediction(src, **kw)


def test_unstated_provenance_raises_rather_than_defaulting():
    src = R.OrbitSource(name="mystery", provider="somewhere",
                        dynamical_model="mpc_nbody", solution_reference="?",
                        gaia_sso_astrometry_in_fit=None)
    with pytest.raises(R.UnknownProvenanceError):
        R.require_independent_prediction(src)


def test_a_current_jpl_orbit_is_partial_and_must_be_acknowledged_explicitly():
    """Gaia DR2/DR3 SSO astrometry is inside every modern JPL fit.

    So the honest classification of a current JPL solution is PARTIAL_SELF_FIT,
    and using it has to be an explicit act.  What makes it usable at all is the
    nuisance marginalisation, not optimism.
    """
    src = R.KNOWN_ORBIT_SOURCES["jpl_horizons"]
    assert src.independence == R.PARTIAL_SELF_FIT
    with pytest.raises(R.CircularOrbitSourceError):
        R.require_independent_prediction(src)
    assert R.require_independent_prediction(src, allow_partial=True) == \
        R.PARTIAL_SELF_FIT


def test_an_orbit_that_already_fitted_a_nongrav_term_is_refused():
    src = R.OrbitSource(name="jpl_with_a2", provider="JPL",
                        dynamical_model="jpl_nbody_de44x_plus_perturbers",
                        solution_reference="SBDB", gaia_sso_astrometry_in_fit=True,
                        nongrav_parameters_fitted=True)
    with pytest.raises(R.CircularOrbitSourceError):
        R.require_independent_prediction(src, allow_partial=True)


def test_a_pre_gaia_snapshot_is_independent():
    src = R.KNOWN_ORBIT_SOURCES["mpcorb_pre_gaia_snapshot"]
    assert R.require_independent_prediction(src) == R.INDEPENDENT


def test_two_body_prediction_is_refused_as_a_dynamical_model():
    with pytest.raises(ValueError, match="full-force"):
        R.require_independent_prediction(SYNTHETIC_SOURCE)


def test_every_residual_record_carries_its_provenance():
    series, _, _, _ = build_series(0.0, n_transits=12, span=600.0)
    assert series.orbit_source["name"] == "synthetic_independent"
    assert series.independence == R.INDEPENDENT
    assert series.conventions["epoch"] == "epoch:TDB:0.0"


# ---------------------------------------------------------------------------
# 4. The physics of the along-track response
# ---------------------------------------------------------------------------
def test_variational_response_reproduces_the_factor_of_three():
    """A transverse acceleration makes an object LAG by 3/2 a_T t^2, not lead by 1/2.

    ``da/dt = 2 a_T/n`` raises the semimajor axis, which lowers the mean motion by
    ``3 a_T/a``, so the along-track displacement is three times the kinematic
    reading and of the opposite sign.  ``seti.loom.residuals.drift_fit`` uses the
    kinematic form; this pins the orbital one on a circular orbit where the
    secular result is exact.
    """
    a = 1.0
    s0 = kepler_state(a, 0.0, 0.0, 0.0, 0.0, 0.0)
    t = JD0 + np.array([400.0])
    resp = R.variational_response(s0, JD0, t, law="radiation", step_days=0.5)
    ref = R.propagate_two_body(s0, t - JD0)
    t_hat = R.transverse_unit(ref)[0]
    along = float(resp[0] @ t_hat)
    dt = 400.0
    expected = -1.5 * (1.0 / a ** 2) * dt ** 2      # a_T = A2 g(r) = A2/a^2
    assert abs(along / expected - 1.0) < 0.05, (along, expected)
    assert along < 0                                 # it lags


def test_variational_basis_reference_epoch_invariance():
    """Moving the reference epoch adds a homogeneous solution, which the nuisance spans.

    So the fitted amplitude must not move.  If it did, the six state partials
    would not be spanning what they claim to span, and the whole
    immunity-by-construction argument would be false.
    """
    series, cols, predict, s0 = build_series(2.0e-13, n_transits=30, span=1500.0)
    m = series.usable()
    nuis = R.nuisance_design(series, m)
    amps = []
    for shift in (0.0, 400.0):
        resp = R.variational_response(s0, JD0, series.jd_tdb,
                                      law="radiation", zero_epoch=JD0 + shift)
        u_pred = R.unit_from_radec(cols["ra"], cols["dec"])
        en = R.project_to_tangent(resp, u_pred)
        e_al, _ = R.scan_basis(cols["position_angle_scan"])
        col = ((en[:, 0] * e_al[:, 0] + en[:, 1] * e_al[:, 1])
               / series.delta_au * R.MAS_PER_RAD)
        f = R.fit_model(series, col, name="r", units="au/day^2", nuisance=nuis,
                        min_arc_days=365.0)
        amps.append(f.amplitude)
    assert abs(amps[0] - amps[1]) / abs(amps[0]) < 0.02, amps


# ---------------------------------------------------------------------------
# 5. Injection and recovery --- the test the channel stands on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a2_true", [3.0e-13, -3.0e-13, 1.0e-13])
def test_injected_acceleration_is_recovered(a2_true):
    """A known A2 goes in; the same A2 must come out, sign and all.

    With the six state partials marginalised --- i.e. with the estimator given no
    help from knowing the orbit was right --- and with anisotropic per-observation
    noise and a per-transit correlated systematic.
    """
    series, _, _, _ = build_series(a2_true)
    mc = R.model_comparison(series, min_arc_days=365.0)
    assert mc["verdict"] == "FORCE_LAW_PREFERRED", mc
    assert mc["best_force_model"] == "force:radiation", mc["best_force_model"]
    fit = mc["fits"]["force:radiation"]
    rel = abs(fit["amplitude"] - a2_true) / abs(a2_true)
    assert rel < 0.15, (fit["amplitude"], a2_true, rel)
    assert fit["snr"] > 10.0
    assert np.sign(fit["amplitude"]) == np.sign(a2_true)
    # The absorbed fraction is high because a quadratic over a six-year arc is
    # largely mimicked by an element error -- and it is REPORTED, which is the
    # difference between a lost sensitivity and a hidden one.
    assert 0.5 < fit["absorbed_fraction"] < 0.9999


def test_pure_noise_recovers_nothing():
    """No signal in, no signal out, on three independent noise realisations."""
    for seed in (11, 12, 13):
        series, _, _, _ = build_series(0.0, seed=seed)
        mc = R.model_comparison(series, min_arc_days=365.0)
        assert mc["verdict"] == "NO_MODEL_PREFERRED", (seed, mc["verdict"])
        fit = mc["fits"]["force:radiation"]
        assert abs(fit["snr"]) < 4.0, (seed, fit["snr"])


def test_a_short_arc_is_refused_rather_than_extrapolated():
    """LOOM promoted four objects on 2-to-29-day arcs.  This is that gate, ported."""
    series, _, _, _ = build_series(3.0e-13, n_transits=12, span=120.0)
    mc = R.model_comparison(series, min_arc_days=365.0)
    assert mc["fits"]["force:radiation"]["ok"] is False
    assert "arc_" in mc["fits"]["force:radiation"]["reason"]


def test_a_clock_error_fits_a_force_law_but_the_family_test_rejects_it():
    """A clock error fits a force law at 12 sigma. THIS is why the family test exists.

    Injecting a 40 s error into the recorded epoch --- shifting the observer, the
    target and the aberration together, as a real clock error does --- leaves a
    678 mas along-scan residual that the six state partials cannot absorb,
    because the observer's own annual motion is not in the span of the target's
    orbital elements.  An amplitude-only search would report a 12-sigma
    non-gravitational acceleration of 1.2e-11 au/day^2, forty times Bennu's.

    The model comparison gets it right: ``geometry:timing`` wins by a factor of
    four in chi-squared with a signal-to-noise in the thousands, because the
    residual's SHAPE is the sky rate and not a double time integral.  That is the
    whole argument for asking which law rather than how much force, demonstrated
    on an injected systematic rather than asserted.
    """
    dt_true = 40.0 / 86400.0
    cols, predict, s0 = make_observations(0.0, n_transits=45)
    shifted = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in cols.items()}
    shifted["epoch"] = cols["epoch"] + dt_true
    pos, vel = observer_state(cols["epoch"] + dt_true)
    for k, nm in enumerate(("x_gaia", "y_gaia", "z_gaia")):
        shifted[nm] = pos[:, k]
    for k, nm in enumerate(("vx_gaia", "vy_gaia", "vz_gaia")):
        shifted[nm] = vel[:, k]
    series = R.compute_residuals(shifted, predict, SYNTHETIC_SOURCE, conv=TEST_CONV,
                                 state0=s0, jd0=JD0, allow_approximate=True)
    mc = R.model_comparison(series, min_arc_days=365.0)
    # The trap: a force law DOES fit, and fits well.
    assert mc["fits"]["force:radiation"]["snr"] > 5.0
    # The escape: geometry fits far better, and the verdict says so.
    assert mc["verdict"] == "GEOMETRIC_EXPLANATION_PREFERRED", mc["verdict"]
    assert mc["best_geometric_model"] == "geometry:timing"
    assert mc["family_margin"] < 0
    dt = R.fit_common_time_offset([series])
    assert dt["verdict"] == "TIMING_OFFSET_DETECTED"
    assert abs(abs(dt["dt_seconds"]) - 40.0) < 6.0, dt["dt_seconds"]


def test_laws_are_not_separated_on_a_near_circular_arc():
    """The honest outcome, and the one LOOM could never reach for a different reason.

    Over an arc that samples little heliocentric range the three force laws are
    the same curve, so a chi-squared difference between them is noise however
    large.  Separability is measured on the design columns themselves, after the
    orbit-error subspace is removed.
    """
    series, _, _, _ = build_series(3.0e-13, e=0.005)
    sep = R.law_separability(series)
    assert sep["verdict"] == "OK"
    assert sep["correlations"]["constant|radiation"] > 0.999, sep
    mc = R.model_comparison(series, min_arc_days=365.0)
    assert mc["law_verdict"] == "LAWS_NOT_SEPARABLE", mc["law_verdict"]
    # ...but the FAMILY question is still answered: it is a force, not geometry.
    assert mc["verdict"] == "FORCE_LAW_PREFERRED"


def test_scan_axis_partition_sees_the_injected_anisotropy():
    series, _, _, _ = build_series(0.0)
    part = R.scan_axis_partition(series)
    assert part["verdict"] == "OK"
    assert part["sigma_ratio_ac_over_al"] > 5.0
    assert 0.5 < part["chi_al"] < 2.0


# ---------------------------------------------------------------------------
# 6. The screen: the discard pile
# ---------------------------------------------------------------------------
def make_population(n=500, seed=5, n_injected=6, injected_factor=8.0):
    """A synthetic catalogue whose rejection rate depends on observing conditions.

    The base rate is the published DR3 outlier fraction, multiplied by a strong
    crowding term.  The point of building the confounder in deliberately is that
    a screen which cannot divide it out will flag the galactic plane, and a
    screen which divides it out too eagerly will flag nothing.
    """
    rng = np.random.default_rng(seed)
    records = []
    for k in range(n):
        att = int(rng.integers(60, 700))
        gal_b = float(rng.uniform(0.0, 70.0))
        motion = float(10.0 ** rng.uniform(4.5, 6.5))
        mag = float(rng.uniform(15.0, 21.0))
        phase = float(rng.uniform(0.0, 30.0))
        crowd = 1.0 + 3.0 * math.exp(-gal_b / 12.0)
        p = 0.0058 * crowd
        injected = k < n_injected
        if injected:
            gal_b = float(rng.uniform(35.0, 70.0))
            crowd = 1.0 + 3.0 * math.exp(-gal_b / 12.0)
            p = 0.0058 * crowd * injected_factor
            att = int(rng.integers(400, 700))
        rej = int(rng.binomial(att, min(p, 0.9)))
        r = S.RejectionRecord(
            key=f"obj{k}", number_mp=float(k + 1), n_attempts=att,
            n_rejected=rej, n_transits=max(att // 9, 1),
            rate=rej / att, apparent_motion_mas_per_day=motion,
            abs_galactic_latitude_deg=gal_b, magnitude=mag,
            median_phase_deg=phase, a=float(rng.uniform(1.8, 3.3)),
            e=float(rng.uniform(0.0, 0.3)), i=float(rng.uniform(0.0, 25.0)),
            node=float(rng.uniform(0.0, 360.0)), h=float(rng.uniform(12.0, 19.0)),
            mjd_min=56900.0, mjd_max=58800.0)
        records.append(r)
    return records


def test_rejection_counts_keep_the_denominator():
    """A rejection COUNT is meaningless; the rate against attempts is the observable."""
    cols, _, _ = make_observations(0.0, n_transits=20, reject_fraction=0.25, seed=2)
    rec = S.rejection_counts(cols, key="x")
    assert rec.n_attempts == 180
    assert rec.n_transits == 20
    assert 0.0 < rec.rate < 1.0
    # Rejections are injected per transit, so the transit-level rate must match
    # the per-observation one exactly: this is the clustering that makes the
    # per-observation count over-dispersed relative to binomial.
    assert abs(rec.transit_rate - rec.rate) < 1e-12


def test_rejected_rows_are_carried_not_filtered():
    """The discard pile is the observable, so nothing may drop it at ingest."""
    cols, predict, s0 = make_observations(0.0, n_transits=20, reject_fraction=0.3,
                                          seed=3)
    series = R.compute_residuals(cols, predict, SYNTHETIC_SOURCE, conv=TEST_CONV,
                                 state0=s0, jd0=JD0, allow_approximate=True)
    assert series.n == 180
    assert int(np.count_nonzero(series.is_rejected)) > 0
    assert series.usable().sum() < series.n
    assert series.usable(include_rejected=True).sum() == series.n


def test_sample_rate_far_from_the_published_outlier_fraction_stops_the_run():
    """If the flags are not being read as the mission means them, everything after
    is a misinterpretation with error bars.  So this check comes first."""
    recs = [S.RejectionRecord(key=f"o{k}", n_attempts=200, n_rejected=100,
                              n_transits=22) for k in range(50)]
    out = S.sample_rejection_summary(recs)
    assert out["verdict"] == "RATE_INCONSISTENT_WITH_PUBLISHED_OUTLIER_FRACTION"
    th = S.Thresholds()
    run = S.screen(recs, th)
    assert run["verdict"] == "RATE_INCONSISTENT_WITH_PUBLISHED_OUTLIER_FRACTION"
    assert "funnel" not in run


def test_the_measured_rate_lands_on_the_published_outlier_fraction():
    """The measurement that precedes the search, and stands on its own."""
    recs = make_population(n_injected=0)
    out = S.sample_rejection_summary(recs)
    assert out["verdict"] == "OK"
    assert 0.004 < out["sample_rejection_rate"] < 0.02
    assert "per_object_rate_quantiles" in out


def test_crowding_is_divided_out_rather_than_flagged():
    """The galactic plane must not be the result.

    With no injected objects, a screen that has not controlled for crowding will
    flag the low-latitude tail; one that has will flag nothing.
    """
    recs = make_population(n_injected=0, seed=21)
    th = S.Thresholds()
    run = S.screen(recs, th)
    assert run["verdict"] == "NO_UNEXPLAINED_EXCESS", run["funnel"]
    assert run["funnel"]["n_interest"] == 0
    assert run["funnel"]["n_candidate"] == 0
    qi = run["quality_independence"]
    assert abs(qi["correlations"]["abs_galactic_latitude_deg"]) < 0.3, qi


def test_an_injected_rejection_excess_is_recovered():
    """Objects discarded eight times more often than their conditions predict."""
    recs = make_population(n_injected=6, injected_factor=8.0, seed=31)
    th = S.Thresholds()
    run = S.screen(recs, th)
    injected = [r for r in recs if int(r.number_mp) <= 6]
    found = [r for r in injected if r.tier in ("interest", "watch")]
    assert len(found) >= 4, [(r.key, r.tier, r.excess_z) for r in injected]
    assert all(r.excess_z > 4.0 for r in found)
    assert run["excess"]["verdict"] == "OK"
    assert run["excess"]["n_above_threshold"] >= 4


def test_objects_with_too_few_attempts_are_untestable_not_ordinary():
    """"We could not measure this" is never folded into "we measured it and it was
    fine".  At 30 attempts and a 0.6% base rate a single rejection is already a
    fivefold excess, which is why the floor is on attempts."""
    recs = make_population(n_injected=0, seed=9)
    for r in recs[:20]:
        r.n_attempts = 20
        r.n_transits = 2
        r.n_rejected = 1
    th = S.Thresholds()
    S.screen(recs, th)
    assert all(r.tier == "untestable" for r in recs[:20])


def test_the_population_stage_refuses_to_run_on_too_few_objects():
    """LOOM's guard, ported unweakened: below this the matched null cannot be
    populated and any p-value would be noise."""
    recs = make_population(n=40, n_injected=3)
    run = S.screen(recs, S.Thresholds())
    assert run["population"]["verdict"] == "INSUFFICIENT_POPULATION"


def test_binary_catalogue_matches_on_number_and_on_name():
    """The contamination catalogue is an INPUT, so the rejection path is exercised.

    Liberato et al. 2024, A&A 688, A50 (arXiv:2406.07195) = VizieR
    J/A+A/688/A50, extended by arXiv:2605.22702 on Gaia FPR.
    """
    cat = S.BinaryCatalogue(rows=[{"Number": 41, "Name": "Daphne"},
                                  {"Name": "1998 XY "}])
    assert cat.match(S.RejectionRecord(number_mp=41.0))
    assert cat.match(S.RejectionRecord(denomination="1998 xy"))
    assert not cat.match(S.RejectionRecord(number_mp=42.0, denomination="Isis"))
    assert "2406.07195" in cat.reference


def test_photocentre_bound_separates_a_wobble_from_an_accumulating_drift():
    """A binary's photocentre offset is BOUNDED; an acceleration is not.

    This is the rejection that needs no catalogue, and it is the stronger of the
    two because catalogue completeness is not an assumption it makes.
    """
    series, _, _, _ = build_series(3.0e-13)
    big = S.photocentre_bound(series, 3.0e-11)
    small = S.photocentre_bound(series, 1.0e-16)
    assert big["verdict"] == "EXCEEDS_PHOTOCENTRE_BOUND"
    assert small["verdict"] == "WITHIN_PHOTOCENTRE_BOUND"
    assert big["implied_displacement_km"] > S.MAX_PHOTOCENTRE_OFFSET_KM
    # And the honest corollary: a Yarkovsky-scale drift is INSIDE the bound, so
    # the bound cannot be the veto and the timescale test has to do that work.
    real = S.photocentre_bound(series, 3.0e-13)
    assert real["verdict"] == "WITHIN_PHOTOCENTRE_BOUND"


def test_a_flagged_object_is_promoted_only_when_the_second_axis_speaks():
    """A discard-rate excess is an anomaly detector, not an explanation.

    Promotion to `candidate` needs the surviving astrometry to prefer a force law
    over every geometric artefact --- which is the discriminant neither published
    treatment of this table performs.
    """
    th = S.Thresholds()
    series, _, _, _ = build_series(3.0e-13)
    rec = S.RejectionRecord(key="k", n_attempts=400, n_rejected=40,
                            n_transits=45, excess_z=6.0, covariate_survival=1.0,
                            abs_galactic_latitude_deg=40.0)
    S.assign_tier(rec, th)
    assert rec.tier == "interest"
    S.characterise(rec, series, th)
    S.assign_tier(rec, th)
    assert rec.tier == "candidate", rec.reasons
    assert rec.model_verdict == "FORCE_LAW_PREFERRED"
    assert any("prefers_a_force_law" in r for r in rec.reasons)


def test_a_known_binary_is_not_promoted_however_good_the_fit():
    th = S.Thresholds()
    series, _, _, _ = build_series(3.0e-13)
    rec = S.RejectionRecord(key="k", number_mp=41.0, n_attempts=400,
                            n_rejected=40, n_transits=45, excess_z=6.0,
                            covariate_survival=1.0, abs_galactic_latitude_deg=40.0)
    cat = S.BinaryCatalogue(rows=[{"Number": 41}])
    S.characterise(rec, series, th, binaries=cat)
    S.assign_tier(rec, th)
    assert rec.known_binary is True
    assert rec.tier == "watch"
    assert any("binary" in r for r in rec.reasons)


def test_pure_noise_astrometry_does_not_promote_a_flagged_object():
    th = S.Thresholds()
    series, _, _, _ = build_series(0.0)
    rec = S.RejectionRecord(key="k", n_attempts=400, n_rejected=40, n_transits=45,
                            excess_z=6.0, covariate_survival=1.0,
                            abs_galactic_latitude_deg=40.0)
    S.characterise(rec, series, th)
    S.assign_tier(rec, th)
    assert rec.model_verdict == "NO_MODEL_PREFERRED"
    assert rec.tier == "interest"        # flagged, but not explained


# ---------------------------------------------------------------------------
# 7. Resolving conventions by measurement
# ---------------------------------------------------------------------------
def test_the_epoch_convention_is_measured_not_assumed():
    """The archive's time scale is a probe question, and the data settles it.

    Every wrong reading is wrong by seconds, and a second is ~8 mas of along-track
    motion for a main-belt object, so the correct candidate wins by orders of
    magnitude rather than by a hair.  The margin is reported, and a margin below
    three is returned as AMBIGUOUS instead of resolved --- because if two
    conventions really did fit equally well, the premise would be broken.
    """
    cols, predict, _ = make_observations(0.0, n_transits=20, span=900.0)
    cands = (R.EpochConvention("epoch", "TDB", 0.0),
             R.EpochConvention("epoch", "TCB", 0.0))
    out = R.resolve_conventions(cols, predict, SYNTHETIC_SOURCE,
                                epoch_candidates=cands, allow_approximate=True)
    assert out["verdict"] == "RESOLVED", out.get("note")
    assert out["conventions"]["epoch"] == "epoch:TDB:0.0"
    assert out["conventions"]["apply_light_time"] is True
    assert out["conventions"]["apply_stellar_aberration"] is True
    assert out["conventions"]["apply_solar_deflection"] is True
    assert out["conventions"]["resolved_by"] == "MEASURED_ON_DATA"
    assert out["margin"] > 3.0
    assert out["best_median_abs_al_mas"] < 5.0
    assert out["scan_convention"]["north_to_east"] is True


def test_dropping_stellar_aberration_costs_twenty_arcseconds():
    """The magnitude that makes the resolver a measurement rather than a fit."""
    cols, predict, s0 = make_observations(0.0, n_transits=10, span=600.0)
    good = R.compute_residuals(cols, predict, SYNTHETIC_SOURCE, conv=TEST_CONV,
                               state0=s0, jd0=JD0, allow_approximate=True,
                               compute_bases=False)
    bad_conv = R.Conventions(epoch=TEST_CONV.epoch, apply_stellar_aberration=False)
    bad = R.compute_residuals(cols, predict, SYNTHETIC_SOURCE, conv=bad_conv,
                              state0=s0, jd0=JD0, allow_approximate=True,
                              compute_bases=False)
    assert float(np.median(np.abs(good.al_mas))) < 5.0
    # 20.5 arcsec of aberration, projected onto a scan axis that points
    # somewhere different at every transit: 11-19 arcsec of along-scan residual,
    # against a signal of a few mas.  Four orders of magnitude is why the
    # convention resolver is a measurement and not a fit.
    assert float(np.median(np.abs(bad.al_mas))) > 5.0e3


def test_an_excess_that_is_not_robust_to_the_null_is_held_at_watch():
    """The excess must be present whichever way the null is stratified."""
    th = S.Thresholds()
    rec = S.RejectionRecord(key="k", n_attempts=400, n_rejected=40, n_transits=45,
                            excess_z=6.0, excess_z_covariate_min=1.0,
                            covariate_survival=1.0 / 6.0,
                            binding_covariate="apparent_motion_mas_per_day",
                            abs_galactic_latitude_deg=40.0)
    S.assign_tier(rec, th)
    assert rec.tier == "watch"
    assert any("not_robust_to_the_null" in v for v in rec.vetoes)


def test_a_low_latitude_object_is_never_promoted_on_crowding_alone():
    th = S.Thresholds()
    rec = S.RejectionRecord(key="k", n_attempts=400, n_rejected=40, n_transits=45,
                            excess_z=8.0, covariate_survival=1.2,
                            abs_galactic_latitude_deg=3.0)
    S.assign_tier(rec, th)
    assert rec.tier == "watch"
    assert any("crowding_not_excluded" in v for v in rec.vetoes)

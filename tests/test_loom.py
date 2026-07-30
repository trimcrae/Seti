"""Offline tests for the LOOM channel (solar-system artefact population search).

Runs with no network, per ``docs/channel-brief.md`` §5.  Organised around the four
requirements that document imposes:

* **recover an injected signal** --- an object with an artificial area-to-mass
  ratio, and one whose along-track residual grows quadratically, are both
  recovered; a planted cluster of anomalous objects is detected by the population
  tests;
* **return a clean null on the dominant confounder** --- a dark comet (large
  acceleration, ordinary AMR, sublimation-law time structure), a common shutter
  timing offset, a star-catalogue bias patch, a short-arc object with an inflated
  fit, and an ordinary Yarkovsky drifter are each rejected, by name;
* **degrade honestly** --- an all-NULL non-gravitational block, too few
  detections, an arc that cannot separate the distance laws, and a null with
  insufficient resolution each produce an explicit verdict, never a silent pass;
* **cover every rejection rule** with a case that trips it.

Three tests are load-bearing regressions rather than unit checks and should not be
weakened.  :func:`test_yarkovsky_column_unit_conversion` guards the ten-orders-of-
magnitude unit trap in ``lsst_mpc_orbits.yarkovsky``.
:func:`test_momentum_ceiling_matches_measured_objects` pins the gate to three real
measurements, so a refactor that moves the ceiling by any factor fails here rather
than in a candidate list.  :func:`test_breakpoint_null_absorbs_the_scan` guards the
fact that scanning for a best breakpoint finds structure in pure noise every time.
"""

from __future__ import annotations

import numpy as np
import pytest

from seti.loom import control, nongrav, replication, residuals, screen
from seti.loom.run import DEFAULTS, load_loom_config, thresholds_from_config


# ---------------------------------------------------------------------------
# The anomaly boundary
# ---------------------------------------------------------------------------
def test_yarkovsky_column_unit_conversion():
    """``lsst_mpc_orbits.yarkovsky`` is 1e-10 au/day^2, not au/day^2.

    Reading the column raw would overstate every acceleration by ten orders of
    magnitude and put the entire catalogue above every ceiling.  Bennu's fitted
    ``A2 = -4.62e-14`` must appear in the column as ``-4.6e-4``.
    """
    col_value = -4.62e-4
    assert nongrav.a2_from_yarkovsky_column(col_value) == pytest.approx(-4.62e-14)
    # And the reverse sanity: a column value of 1 would be 1e-10 au/day^2, which
    # is comet territory, four orders above any asteroid Yarkovsky.
    assert nongrav.a2_from_yarkovsky_column(1.0) == pytest.approx(1e-10)


def test_srp_column_unit_conversion():
    """``srp`` is m^2/ton; the artificial objects land where they should."""
    # J002E3's published AMR is 7.9e-3 m^2/kg, i.e. 7.9 m^2/ton.
    assert nongrav.amr_from_srp_column(7.9) == pytest.approx(7.9e-3)
    # A natural small NEA: 2009 BD at 2.97e-4 m^2/kg = 0.297 m^2/ton.
    assert nongrav.amr_from_srp_column(0.297) == pytest.approx(2.97e-4)


def test_beta_constant_matches_first_principles():
    """``beta = 7.656e-4 * C_R * AMR`` follows from (Phi/c) / (GM/r^2) at 1 au."""
    assert nongrav.BETA_PER_CR_AMR == pytest.approx(7.656e-4, rel=2e-3)
    # C_R = 1.5 gives the 1.148e-3 coefficient quoted for a diffuse reflector.
    assert float(nongrav.beta_from_amr(1.0, c_r=1.5)) == pytest.approx(1.148e-3,
                                                                      rel=2e-3)


def test_amr_sphere_reproduces_published_implied_density():
    """WT1190F's AMR implies rho*D ~ 127 kg/m^2 --- a shell, not a rock."""
    amr = 1.18e-2
    rho_d = 3.0 / (2.0 * amr)
    assert rho_d == pytest.approx(127.0, rel=0.05)


def test_momentum_ceiling_matches_measured_objects():
    """The gate, pinned to three objects with independently measured ``A2``.

    Realised thermal-recoil efficiency must come out at a few per cent for all
    three.  If a refactor moves the ceiling, these fail here rather than silently
    changing which objects are candidates.
    """
    rows = nongrav.calibration_table()
    by_name = {r["name"]: r for r in rows}
    assert by_name["(101955) Bennu"]["ceiling_a2_hard"] == pytest.approx(5.83e-13,
                                                                        rel=0.02)
    for r in rows:
        assert 0.01 < r["epsilon_effective"] < 0.10, r
    # Ordered by size: the smaller the body, the larger the AMR, so the ceiling
    # rises monotonically as diameter falls.
    ceils = [by_name[n]["ceiling_a2_hard"] for n in
             ("(101955) Bennu", "2005 ES70", "2009 BD")]
    assert ceils[0] < ceils[1] < ceils[2]


def test_a2_to_dadt_reproduces_bennu():
    """The A2 <-> da/dt conversion, validated against a published measurement.

    Bennu's measured drift of -19.0e-4 au/Myr must invert to within a couple of
    per cent of JPL's fitted ``A2 = -4.62e-14``; the residual is the ``d = 2`` vs
    ``d = 2.25`` choice of ``g(r)`` exponent, not an error in the algebra.
    """
    a2 = float(nongrav.a2_from_dadt_au_per_myr(-19.0e-4, 1.1264, 0.2037))
    assert a2 == pytest.approx(-4.62e-14, rel=0.03)
    # Round trip.
    assert float(nongrav.dadt_au_per_myr(a2, 1.1264, 0.2037)) == pytest.approx(
        -19.0e-4, rel=1e-6)


def test_ceiling_is_generous_by_construction():
    """Lower density and higher albedo both raise the permitted acceleration.

    The defaults must be on the generous side, so that exceeding the ceiling is a
    statement about the object rather than about the assumptions.
    """
    h = 20.0
    generous = float(nongrav.momentum_ceiling_a2(h, albedo=0.25, rho_kg_m3=1000.0))
    typical = float(nongrav.momentum_ceiling_a2(h, albedo=0.15, rho_kg_m3=2000.0))
    assert generous > typical


def test_untestable_h_propagates_as_nan_not_zero():
    """An object with no ``H`` is untestable and must never score as ordinary."""
    r = nongrav.ceiling_ratio([float("nan")], [1e-12])
    assert not np.isfinite(r[0])
    r2 = nongrav.amr_ceiling_ratio([float("nan")], [1e-2])
    assert not np.isfinite(r2[0])


def test_amr_discriminant_separates_natural_from_artificial():
    """The whole artificiality argument, on real numbers.

    A 4 m natural body and an artificial object of the same brightness differ by a
    factor of tens in area-to-mass ratio, and only the latter exceeds what a solid
    body of that size can be.
    """
    # H ~ 28.2 (2009 BD).  Natural: 2.97e-4 m^2/kg.  Artificial: 1.18e-2.
    nat = float(nongrav.amr_ceiling_ratio(28.2, 2.97e-4))
    art = float(nongrav.amr_ceiling_ratio(28.2, 1.18e-2))
    assert nat < 1.0 < art
    assert art / nat == pytest.approx(1.18e-2 / 2.97e-4, rel=1e-6)


# ---------------------------------------------------------------------------
# Fit quality --- the base-rate discipline
# ---------------------------------------------------------------------------
def test_fit_quality_rejects_short_arc_and_inflated_rms():
    """Every rejection rule trips, and each is named."""
    good = nongrav.fit_quality(1e-13, 1e-14, 1.0, 400.0, 4)
    assert good.ok and not good.reasons
    assert good.snr == pytest.approx(10.0)

    low_snr = nongrav.fit_quality(1e-14, 1e-14, 1.0, 400.0, 4)
    assert not low_snr.ok
    assert any("snr" in r for r in low_snr.reasons)

    inflated = nongrav.fit_quality(1e-13, 1e-14, 4.0, 400.0, 4)
    assert any("normalized_rms" in r for r in inflated.reasons)

    short = nongrav.fit_quality(1e-13, 1e-14, 1.0, 30.0, 4)
    assert any("arc_" in r for r in short.reasons)

    one_opp = nongrav.fit_quality(1e-13, 1e-14, 1.0, 400.0, 1)
    assert any("n_opp" in r for r in one_opp.reasons)

    missing = nongrav.fit_quality(None, None, None, None, None)
    assert not missing.ok
    assert "no_fitted_a2" in missing.reasons


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------
def test_arcsec_to_km_uses_topocentric_distance():
    """One arcsec at 1 au is ~725 km; geometry must not be left in the signal."""
    km = float(residuals.arcsec_to_km(1.0, 1.0))
    assert km == pytest.approx(725.27, rel=1e-3)
    # Twice as far away, twice the physical displacement for the same angle.
    assert float(residuals.arcsec_to_km(1.0, 2.0)) == pytest.approx(2 * km)


def test_drift_fit_recovers_injected_acceleration():
    """A quadratic along-track growth is recovered with the right sign and size."""
    t = np.linspace(0.0, 400.0, 40)
    accel = 3.0e-4                        # km/day^2
    y = 0.5 * accel * (t - 200.0) ** 2 + 12.0 * (t - 200.0) + 5.0
    rng = np.random.default_rng(11)
    y = y + rng.normal(0.0, 0.5, t.size)
    out = residuals.drift_fit(t, y, np.full(t.size, 0.5), epoch_mjd=200.0)
    assert out["verdict"] == "OK"
    # Checked against the fit's OWN reported uncertainty rather than a fixed
    # tolerance: a hard-coded relative tolerance either passes by luck or fails on
    # a 1.4-sigma noise draw, and neither says whether the estimator is unbiased.
    assert abs(out["accel_km_per_day2"] - accel) < 3.0 * out["accel_km_per_day2_err"]
    assert out["accel_snr"] > 5.0
    assert out["delta_chi2_quadratic"] > 25.0


def test_drift_fit_null_on_pure_linear_series():
    """A wrong mean motion is LINEAR in time and must not read as acceleration."""
    t = np.linspace(0.0, 400.0, 40)
    rng = np.random.default_rng(12)
    y = 4.0 * t + rng.normal(0.0, 0.5, t.size)
    out = residuals.drift_fit(t, y, np.full(t.size, 0.5))
    assert out["verdict"] == "OK"
    assert out["delta_chi2_quadratic"] < 9.0
    assert not (out["accel_snr"] > 5.0)


def test_drift_fit_degrades_on_too_few_epochs():
    out = residuals.drift_fit([1.0, 2.0], [1.0, 2.0], [0.1, 0.1])
    assert out["verdict"] == "TOO_FEW_EPOCHS"


def test_common_timing_offset_is_identifiable_across_objects():
    """A shutter error is linear in the object's rate, and the slope IS ``dt``.

    This is the dominant confounder in the residual path and the reason the test
    is done at population level: within one object it is perfectly degenerate with
    a real along-track acceleration.
    """
    rng = np.random.default_rng(13)
    rate = rng.uniform(0.1, 30.0, 400)     # arcsec/min
    dt_true = 0.4                          # seconds
    along = rate * dt_true / 60.0 + rng.normal(0.0, 0.002, rate.size)
    sol = residuals.fit_common_timing(along, rate)
    assert sol.ok
    assert sol.dt_seconds == pytest.approx(dt_true, rel=0.05)
    assert sol.variance_explained > 0.9
    # And removing it leaves nothing correlated with rate.
    corrected = residuals.subtract_timing(along, rate, sol.dt_seconds)
    assert abs(float(np.corrcoef(rate, corrected)[0, 1])) < 0.3


def test_common_timing_null_when_residuals_are_not_rate_driven():
    rng = np.random.default_rng(14)
    rate = rng.uniform(0.1, 30.0, 400)
    along = rng.normal(0.0, 0.1, rate.size)
    sol = residuals.fit_common_timing(along, rate)
    assert sol.ok
    assert sol.variance_explained < 0.2


def test_per_object_timing_correlation_flags_the_artefact():
    rng = np.random.default_rng(15)
    rate = rng.uniform(1.0, 20.0, 30)
    along = rate * 0.005 + rng.normal(0.0, 0.001, rate.size)
    assert residuals.per_object_rate_correlation(along, rate) > 0.9


def test_sublimation_and_radiation_laws_are_distinguishable():
    """The novelty test: which heliocentric-distance law does the drift follow?

    A drift generated by the water-ice sublimation law must be preferred over the
    inverse-square and constant alternatives, and vice versa, provided the arc
    samples enough heliocentric range.
    """
    rng = np.random.default_rng(16)
    t = np.linspace(0.0, 1400.0, 60)
    # r sweeps 1.2 -> 3.5 au, spanning the sublimation knee at 2.8 au.
    r = np.linspace(1.2, 3.5, t.size)

    for law_name, law in (("sublimation", residuals.g_comet),
                          ("radiation", residuals.g_radiation)):
        g = np.asarray(law(r), dtype=float)
        # Integrate the drift rate to get displacement.  The trapezoid rule, not a
        # left-endpoint sum: the test fits the rate against g at the *midpoint* of
        # each interval, so a left-endpoint integral would inject a discretisation
        # error far larger than the noise and both laws would fit badly.
        y = np.concatenate([[0.0],
                            np.cumsum(0.5 * (g[:-1] + g[1:]) * np.diff(t))]) * 1000.0
        y = y + rng.normal(0.0, 0.5, t.size)
        out = residuals.law_discrimination(t, y, np.full(t.size, 0.5), r)
        assert out["verdict"] == "LAW_PREFERRED", (law_name, out)
        assert out["best_law"] == law_name, out


def test_law_discrimination_refuses_a_short_r_span():
    """Honest degradation: over one apparition the three laws are not separable."""
    t = np.linspace(0.0, 60.0, 30)
    r = np.full(t.size, 2.5) + np.linspace(0, 0.05, t.size)
    y = np.linspace(0.0, 100.0, t.size)
    out = residuals.law_discrimination(t, y, np.full(t.size, 0.5), r)
    assert out["verdict"] == "INSUFFICIENT_R_SPAN"


def test_breakpoint_null_absorbs_the_scan():
    """Scanning for the best breakpoint finds structure in pure noise every time.

    So the null must be generated with the same scan.  A smooth quadratic series
    must NOT be flagged; without the resampled null this test fails, and so would
    most objects in the survey.
    """
    rng = np.random.default_rng(17)
    t = np.linspace(0.0, 400.0, 40)
    y = 0.001 * (t - 200) ** 2 + rng.normal(0.0, 1.0, t.size)
    out = residuals.breakpoint_scan(t, y, np.full(t.size, 1.0), n_null=200,
                                    rng=np.random.default_rng(18))
    assert out["verdict"] == "OK"
    assert out["p_value"] > 0.05, out


def test_breakpoint_detects_an_acceleration_that_stops():
    """The derelict signature: a drift that switches off at a discrete epoch."""
    t = np.linspace(0.0, 400.0, 60)
    y = np.where(t < 200.0, 2.0 * t, 400.0)
    rng = np.random.default_rng(19)
    y = y + rng.normal(0.0, 1.0, t.size)
    out = residuals.breakpoint_scan(t, y, np.full(t.size, 1.0), n_null=200,
                                    rng=np.random.default_rng(20))
    assert out["verdict"] == "OK"
    assert out["p_value"] <= 0.02, out
    assert 150.0 < out["break_mjd"] < 250.0


def test_sky_coherence_separates_a_catalogue_bias_patch():
    """Star-catalogue bias is coherent with sky position, not with the object."""
    rng = np.random.default_rng(21)
    n = 600
    ra = rng.uniform(0.0, 40.0, n)
    dec = rng.uniform(-20.0, 20.0, n)
    # A systematic that depends only on where on the sky the measurement was made.
    patch = np.where(ra < 20.0, 0.5, -0.5)
    value = patch + rng.normal(0.0, 0.05, n)
    out = residuals.sky_coherence(ra, dec, value, bin_deg=10.0)
    assert out["verdict"] == "OK"
    assert out["variance_explained_by_sky_bin"] > 0.8

    # An object-driven residual, randomly placed, explains nothing by position.
    value2 = rng.normal(0.0, 0.5, n)
    out2 = residuals.sky_coherence(ra, dec, value2, bin_deg=10.0)
    assert out2["variance_explained_by_sky_bin"] < 0.2


def test_along_cross_partition_rejects_an_isotropic_residual():
    """Amplitude is not the discriminant; geometry is.

    A transverse force displaces an object along its track.  Star-catalogue bias
    and mis-association have no directional preference, so an isotropic residual is
    not an acceleration however large it is.
    """
    rng = np.random.default_rng(31)
    n = 200
    # Isotropic: equal power in both components.
    iso = residuals.along_cross_partition(rng.normal(0, 0.5, n),
                                          rng.normal(0, 0.5, n))
    assert iso["verdict"] == "OK"
    assert iso["power_ratio"] == pytest.approx(1.0, rel=0.35)

    # Along-track dominated: what an acceleration produces.
    acc = residuals.along_cross_partition(rng.normal(1.5, 0.1, n),
                                          rng.normal(0, 0.1, n))
    assert acc["power_ratio"] > 10.0
    assert acc["along_coherence"] > 5.0


def test_apparition_trend_separates_growth_from_wander():
    """An acceleration accumulates; an orbit-fit error wanders."""
    # Five apparitions, ~500 days apart, offset growing quadratically.
    t, y = [], []
    for k in range(5):
        base = 60000.0 + 500.0 * k
        t.extend(base + np.linspace(0, 20, 8))
        y.extend(np.full(8, 0.05 * k * k))
    out = residuals.apparition_trend(np.array(t), np.array(y))
    assert out["verdict"] == "OK"
    assert out["n_apparitions"] == 5
    assert out["spearman"] > 0.9

    # Same sampling, offsets that wander in sign: not a trend.
    y2 = []
    for v in (0.3, -0.2, 0.25, -0.3, 0.1):
        y2.extend(np.full(8, v))
    out2 = residuals.apparition_trend(np.array(t), np.array(y2))
    assert out2["verdict"] == "OK"
    assert abs(out2["spearman"]) < 0.7
    assert out2["sign_consistent"] is False


def test_apparition_trend_refuses_a_single_apparition():
    """Many detections in one apparition are one measurement of the trend."""
    t = 60000.0 + np.linspace(0, 25, 40)
    out = residuals.apparition_trend(t, np.linspace(0, 1, 40))
    assert out["verdict"] == "TOO_FEW_APPARITIONS"


def test_quality_independence_catches_a_score_that_tracks_arc_length():
    """The failure every blind non-gravitational search runs into."""
    rng = np.random.default_rng(32)
    arc = rng.uniform(100.0, 8000.0, 300)
    # A "signal" that is really 1/arc: short arcs get big residuals.
    score = 1000.0 / arc + rng.normal(0, 0.01, arc.size)
    out = residuals.quality_independence(score, {"arc_days": arc})
    assert out["verdict"] == "OK"
    assert out["max_abs_correlation"] > 0.8
    assert out["max_correlated_with"] == "arc_days"

    clean = rng.normal(0, 1, arc.size)
    out2 = residuals.quality_independence(clean, {"arc_days": arc})
    assert out2["max_abs_correlation"] < 0.25


def test_quality_independence_reports_h_without_gating_on_it():
    """Absolute magnitude is the signature's own shape, not a confounder.

    The momentum ceiling is a function of ``H`` by construction, so an anomalous
    set being systematically faint is expected; the matched null already stratifies
    on it.  It must be reported and must not invalidate a result — whereas the same
    correlation with arc length must.
    """
    rng = np.random.default_rng(33)
    h = rng.uniform(16.0, 28.0, 300)
    score = -h + rng.normal(0, 0.2, h.size)      # perfectly H-driven
    arc = rng.uniform(200.0, 6000.0, h.size)
    out = residuals.quality_independence(score, {"h": h, "arc_days": arc},
                                         gate_keys=["arc_days"])
    assert abs(out["correlations"]["h"]) > 0.9
    assert out["max_correlated_with"] == "arc_days"
    assert out["max_abs_correlation"] < 0.25
    assert out["reported_only"] == ["h"]


def test_residual_significance_does_not_reward_bad_orbits():
    """A large residual on a badly determined orbit is the expected outcome."""
    good = float(residuals.residual_significance(0.5, 1.0, 3000.0))
    bad = float(residuals.residual_significance(0.5, 4.0, 30.0))
    assert good > bad


# ---------------------------------------------------------------------------
# Screening and the tier ladder
# ---------------------------------------------------------------------------
def _orbit_row(**kw) -> dict:
    row = {"ssobjectid": "1", "designation": "TEST 1", "h": 20.0,
           "a": 2.4, "e": 0.15, "i": 5.0, "node": 100.0, "argperi": 40.0,
           "normalized_rms": 1.0, "arc_length_total": 3000.0, "nopp": 5,
           "yarkovsky": None, "yarkovsky_unc": None, "srp": None, "srp_unc": None}
    row.update(kw)
    return row


def test_ordinary_yarkovsky_drifter_is_ordinary():
    """Bennu-like: a real, published, entirely natural non-gravitational drift."""
    th = screen.Thresholds()
    # A2 = 4.62e-14 -> column value 4.62e-4.
    rec = screen.assign_tier(screen.screen_orbit_row(
        _orbit_row(h=20.2, yarkovsky=4.62e-4, yarkovsky_unc=1e-5,
                   a=1.1264, e=0.2037), th), th)
    assert rec.tier == "ordinary", (rec.tier, rec.score, rec.reasons)
    assert rec.dadt_au_myr == pytest.approx(-19.0e-4, rel=0.05, abs=2e-4) or \
        rec.dadt_au_myr == pytest.approx(19.0e-4, rel=0.05, abs=2e-4)


def test_dark_comet_reaches_interest_but_not_candidate():
    """The channel's hardest confuser, and the whole novelty position in one test.

    A large acceleration with an ordinary area-to-mass ratio is Seligman et al.'s
    dark comet.  It MUST be flagged --- that is what makes a magnitude cut
    insufficient --- and it must NOT be promoted, because outgassing explains it.
    """
    th = screen.Thresholds()
    # Three times the hard ceiling for its size: sunlight cannot do this, so the
    # object is either losing mass or is not a rock.  Derived from the ceiling
    # rather than hard-coded, so the test states the relationship it means.
    ceiling = float(nongrav.momentum_ceiling_a2(
        24.0, albedo=th.albedo_generous, rho_kg_m3=th.rho_generous_kg_m3,
        epsilon=th.epsilon_hard))
    col = 3.0 * ceiling / nongrav.YARKOVSKY_COL_UNIT
    rec = screen.assign_tier(screen.screen_orbit_row(
        _orbit_row(h=24.0, yarkovsky=col, yarkovsky_unc=col / 30.0), th), th)
    assert rec.ratio_hard == pytest.approx(3.0)
    assert rec.tier == "interest"
    assert any("dark_comet" in r for r in rec.reasons)


def test_artificial_area_to_mass_ratio_is_promoted():
    """An injected artificial object: WT1190F's AMR at its brightness."""
    th = screen.Thresholds()
    rec = screen.assign_tier(screen.screen_orbit_row(
        _orbit_row(h=28.2, srp=11.8, srp_unc=0.5,
                   yarkovsky=1.0e-2, yarkovsky_unc=1e-4), th), th)
    assert rec.amr_m2_kg == pytest.approx(1.18e-2)
    assert rec.amr_ratio > 1.0
    assert rec.tier == "candidate"
    assert any("amr_" in r for r in rec.reasons)


def test_systematic_explanations_block_promotion():
    """Each veto is named and each blocks promotion on its own."""
    th = screen.Thresholds()

    base = _orbit_row(h=28.2, srp=11.8, srp_unc=0.5)
    rec = screen.screen_orbit_row(base, th)
    rec.timing_correlation = 0.9
    screen.assign_tier(rec, th)
    assert rec.tier != "candidate"
    assert any("timing_correlated" in r for r in rec.reasons)

    rec = screen.screen_orbit_row(base, th)
    rec.best_law, rec.delta_chi2_law = "sublimation", 40.0
    screen.assign_tier(rec, th)
    assert rec.tier != "candidate"
    assert "sublimation_law_preferred" in rec.reasons

    rec = screen.screen_orbit_row(base, th)
    rec.sky_variance_explained = 0.8
    screen.assign_tier(rec, th)
    assert rec.tier != "candidate"
    assert any("sky_coherent" in r for r in rec.reasons)


def test_geometry_and_coma_vetoes_block_promotion():
    """Two more vetoes, each named, each sufficient on its own."""
    th = screen.Thresholds()
    base = _orbit_row(h=28.2, srp=11.8, srp_unc=0.5)

    rec = screen.screen_orbit_row(base, th)
    rec.along_cross_power_ratio = 1.1          # isotropic: not an acceleration
    screen.assign_tier(rec, th)
    assert rec.tier != "candidate"
    assert any("residual_isotropic" in r for r in rec.reasons)

    rec = screen.screen_orbit_row(base, th)
    rec.extendedness_median = 0.9              # resolved: a coma
    screen.assign_tier(rec, th)
    assert rec.tier != "candidate"
    assert any("coma" in r for r in rec.reasons)

    rec = screen.screen_orbit_row(base, th)
    rec.n_apparitions, rec.apparition_spearman = 5, 0.1
    screen.assign_tier(rec, th)
    assert rec.tier != "candidate"
    assert any("not_monotone" in r for r in rec.reasons)

    # And with too few apparitions the axis is SILENT, not permissive: the same
    # weak Spearman must not veto when it could not be measured properly.
    rec = screen.screen_orbit_row(base, th)
    rec.n_apparitions, rec.apparition_spearman = 2, 0.1
    screen.assign_tier(rec, th)
    assert rec.tier == "candidate"


def test_no_measurement_is_untestable_not_ordinary():
    """"We could not look" must never be recorded as "we looked and it was fine"."""
    th = screen.Thresholds()
    rec = screen.assign_tier(screen.screen_orbit_row(_orbit_row(), th), th)
    assert rec.tier == "untestable"
    assert rec.reasons == ["no_acceleration_measurement"]

    rec = screen.assign_tier(screen.screen_orbit_row(
        _orbit_row(h=None, yarkovsky=1.0, yarkovsky_unc=1e-3), th), th)
    assert rec.tier == "untestable"
    assert rec.reasons == ["no_absolute_magnitude"]


def test_empty_nongrav_block_is_a_dead_path_not_a_null():
    """An all-NULL non-gravitational block must say so, explicitly."""
    th = screen.Thresholds()
    rows = [_orbit_row(ssobjectid=str(i)) for i in range(50)]
    recs, funnel = screen.screen_orbits(rows, th)
    assert funnel["verdict"] == "NONGRAV_COLUMNS_EMPTY"
    assert funnel["n_untestable"] == 50
    assert all(r.tier == "untestable" for r in recs)


def test_watch_tier_sits_between_the_two_epsilons():
    """Above the realistic thermal-recoil envelope, below the hard ceiling."""
    th = screen.Thresholds()
    ceiling = float(nongrav.momentum_ceiling_a2(
        22.0, albedo=th.albedo_generous, rho_kg_m3=th.rho_generous_kg_m3,
        epsilon=th.epsilon_hard))
    a2 = 0.3 * ceiling                        # 3x the realistic envelope
    rec = screen.assign_tier(screen.screen_orbit_row(
        _orbit_row(h=22.0, yarkovsky=a2 / nongrav.YARKOVSKY_COL_UNIT,
                   yarkovsky_unc=a2 / nongrav.YARKOVSKY_COL_UNIT / 20.0), th), th)
    assert rec.tier == "watch"
    assert "above_realistic_thermal_recoil_envelope" in rec.reasons


# ---------------------------------------------------------------------------
# Population structure --- where the channel actually decides
# ---------------------------------------------------------------------------
def _population(n: int = 600, seed: int = 5) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = [{"a": float(a), "e": float(e), "i": float(i), "node": float(nd),
             "h": float(h)}
            for a, e, i, nd, h in zip(
                rng.uniform(2.0, 3.3, n), rng.uniform(0.0, 0.3, n),
                rng.uniform(0.0, 20.0, n), rng.uniform(0.0, 360.0, n),
                rng.uniform(16.0, 24.0, n), strict=True)]
    labels = np.zeros(n, dtype=int)
    return rows, labels


def test_replication_tests_detect_a_planted_family():
    """A set with a common origin is recovered: clustered elements, aligned poles."""
    rows, labels = _population()
    rng = np.random.default_rng(6)
    mask = np.zeros(len(rows), dtype=bool)
    for k in range(30):
        rows[k].update({"a": 2.60 + float(rng.normal(0, 0.004)),
                        "e": 0.12 + float(rng.normal(0, 0.004)),
                        "i": 8.0 + float(rng.normal(0, 0.15)),
                        "node": 210.0 + float(rng.normal(0, 1.5))})
        mask[k] = True
    out = replication.replication_tests(rows, mask, labels, n_null=2000)
    assert out["verdict"] == "REPLICATION_STRUCTURE_DETECTED", out
    by = {t["name"]: t for t in out["tests"]}
    assert by["element_clustering"]["p_value"] < 0.01
    assert by["pole_coherence"]["p_value"] < 0.01


def test_replication_tests_clean_null_on_a_random_subset():
    """A random subset of the same population must return no structure."""
    rows, labels = _population()
    rng = np.random.default_rng(7)
    mask = np.zeros(len(rows), dtype=bool)
    mask[rng.choice(len(rows), 30, replace=False)] = True
    out = replication.replication_tests(rows, mask, labels, n_null=2000)
    assert out["verdict"] == "NO_STRUCTURE", out


def test_replication_resolution_guard():
    """A null too coarse to resolve the Bonferroni threshold refuses to report."""
    rows, labels = _population()
    mask = np.zeros(len(rows), dtype=bool)
    mask[:20] = True
    out = replication.replication_tests(rows, mask, labels, n_null=100)
    assert out["verdict"] == "INSUFFICIENT_RESOLUTION", out


def test_replication_refuses_a_tiny_population():
    rows, labels = _population(n=40)
    mask = np.zeros(len(rows), dtype=bool)
    mask[:6] = True
    out = replication.replication_tests(rows, mask, labels, n_null=500)
    assert out["verdict"] == "INSUFFICIENT_POPULATION"


def test_inclination_isotropy_detects_a_dynamically_hot_subset():
    """A captured population keeps no memory of this disc's plane."""
    rows, labels = _population()
    rng = np.random.default_rng(8)
    mask = np.zeros(len(rows), dtype=bool)
    for k in range(40):
        # Isotropic inclinations: cos i uniform on [-1, 1].
        rows[k]["i"] = float(np.degrees(np.arccos(rng.uniform(-1, 1))))
        mask[k] = True
    inc = np.array([r["i"] for r in rows])
    st = replication.inclination_isotropy(inc, mask, labels, n_null=2000)
    assert st.ok
    assert st.p_value < 0.01
    assert st.detail["retrograde_fraction"] > 0.2


def test_linkage_duplicates_collapse_one_object_seen_as_three():
    """The contaminant that looks exactly like a family, and it is not hypothetical.

    A strongly accelerating object breaks MPC linking and can enter the catalogue
    several times.  Near-identical elements with *disjoint* observation epochs is
    that, not replication; near-identical elements observed *at the same time* is a
    real pair and must survive.
    """
    rows = [{"a": 2.5, "e": 0.1, "i": 6.0, "mjd_min": 60000.0, "mjd_max": 60100.0},
            {"a": 2.5001, "e": 0.1002, "i": 6.01, "mjd_min": 60400.0,
             "mjd_max": 60500.0},
            {"a": 2.4999, "e": 0.0999, "i": 5.99, "mjd_min": 60800.0,
             "mjd_max": 60900.0},
            {"a": 3.1, "e": 0.2, "i": 12.0, "mjd_min": 60000.0, "mjd_max": 60900.0}]
    mask = np.ones(4, dtype=bool)
    dup = replication.linkage_duplicates(rows, mask)
    assert dup["n_groups"] == 1
    assert dup["n_collapsed"] == 2
    collapsed = replication.collapse_duplicates(mask, dup)
    assert collapsed.sum() == 2

    # Contemporaneous twins are a real pair: they were both on the sky at once.
    rows2 = [dict(r) for r in rows[:2]]
    rows2[1].update({"mjd_min": 60010.0, "mjd_max": 60090.0})
    dup2 = replication.linkage_duplicates(rows2, np.ones(2, dtype=bool))
    assert dup2["n_groups"] == 0


def test_replication_reports_and_collapses_mislinkage():
    """The collapse happens before any statistic sees the set, and is reported."""
    rows, labels = _population()
    mask = np.zeros(len(rows), dtype=bool)
    for k in range(6):
        rows[k].update({"a": 2.7, "e": 0.11, "i": 7.0,
                        "mjd_min": 60000.0 + 400.0 * k,
                        "mjd_max": 60100.0 + 400.0 * k})
        mask[k] = True
    out = replication.replication_tests(rows, mask, labels, n_null=500)
    assert out["n_anomaly_raw"] == 6
    assert out["linkage_duplicates"]["n_collapsed"] == 5
    assert out["n_anomaly"] == 1
    # One object left is not a population, and the channel must say so.
    assert out["verdict"] == "INSUFFICIENT_POPULATION"


def test_photometric_homogeneity_detects_a_common_surface():
    """The independent second axis --- and what it cannot decide, tested too."""
    rows, labels = _population()
    rng = np.random.default_rng(41)
    for r in rows:
        r["h_g"] = float(rng.uniform(16, 22))
        r["h_r"] = r["h_g"] - float(rng.uniform(0.2, 0.9))
        r["h_i"] = r["h_r"] - float(rng.uniform(0.0, 0.5))
        r["g12_r"] = float(rng.uniform(0.1, 0.9))
    mask = np.zeros(len(rows), dtype=bool)
    for k in range(30):
        rows[k]["h_r"] = rows[k]["h_g"] - 0.55 + float(rng.normal(0, 0.01))
        rows[k]["h_i"] = rows[k]["h_r"] - 0.25 + float(rng.normal(0, 0.01))
        rows[k]["g12_r"] = 0.5 + float(rng.normal(0, 0.01))
        mask[k] = True
    st = replication.photometric_homogeneity(rows, mask, labels, n_null=2000)
    assert st.ok and st.p_value < 0.01
    assert "manufactured vs collisional" in st.detail["cannot_decide"]


def test_photometric_homogeneity_degrades_without_photometry():
    rows, labels = _population()
    mask = np.zeros(len(rows), dtype=bool)
    mask[:10] = True
    st = replication.photometric_homogeneity(rows, mask, labels, n_null=100)
    assert not st.ok
    assert st.reason == "fewer_than_two_usable_photometric_columns"


def test_resonance_locations_reproduce_the_kirkwood_gaps():
    """The 3:1 gap sits at 2.50 au and the 2:1 at 3.28; Hildas at 3.97."""
    locs = replication.resonance_locations()
    assert locs["3:1"] == pytest.approx(2.50, abs=0.02)
    assert locs["5:2"] == pytest.approx(2.82, abs=0.03)
    assert locs["2:1"] == pytest.approx(3.28, abs=0.03)
    assert locs["3:2"] == pytest.approx(3.97, abs=0.03)
    assert locs["1:1"] == pytest.approx(5.2044, abs=1e-3)


def test_resonance_concentration_detects_parking():
    """Signature S29: objects sitting where a natural body least often arrives."""
    rows, labels = _population()
    rng = np.random.default_rng(9)
    locs = sorted(replication.resonance_locations().values())
    targets = [v for v in locs if 2.0 <= v <= 3.3]
    mask = np.zeros(len(rows), dtype=bool)
    for k in range(30):
        rows[k]["a"] = float(targets[k % len(targets)] + rng.normal(0, 0.004))
        mask[k] = True
    a = np.array([r["a"] for r in rows])
    st = replication.resonance_concentration(a, mask, labels, n_null=2000)
    assert st.ok and st.p_value < 0.01


def test_size_distribution_is_reported_not_gated():
    """Selection shapes the anomalous set's sizes, so this cannot be a gate."""
    st = replication.size_distribution(np.linspace(18.0, 24.0, 60))
    assert st.ok
    assert st.gate is False
    assert st.as_dict()["is_gate"] is False


def test_pole_coherence_uses_the_orientation_tensor():
    """Axial data: a vector mean would cancel antipodal poles and hide alignment."""
    inc = np.array([10.0, 10.0, 170.0, 170.0, 10.0, 170.0])
    node = np.array([0.0, 180.0, 0.0, 180.0, 90.0, 270.0])
    P = replication.orbital_poles(inc, node)
    # Unit vectors.
    assert np.allclose(np.linalg.norm(P, axis=1), 1.0)
    T = (P[:, :, None] * P[:, None, :]).mean(axis=0)
    lam = float(np.linalg.eigvalsh(T)[-1])
    # Strongly axial (all near the +/- z axis) despite a vector mean near zero.
    assert lam > 0.9
    assert np.linalg.norm(P.mean(axis=0)) < 0.2


# ---------------------------------------------------------------------------
# Positive controls
# ---------------------------------------------------------------------------
def test_designation_normalisation_matches_across_catalogues():
    n = control.normalise_designation
    assert n("2020 SO") == n("2020SO") == n("(2020 SO)")
    assert n("(101955)") == n("101955") == n("101955 Bennu") == "101955"
    assert n(None) == ""


def test_control_validation_recovers_a_planted_artificial_object():
    """If the screen does not recover J002E3, it does not work."""
    rows = [{"designation": f"FILLER {i}", "score": 0.01 * i, "flagged": False}
            for i in range(100)]
    rows.append({"designation": "J002E3", "score": 50.0, "flagged": True})
    out = control.validate(rows, score_key="score")
    assert out["verdict"] == "SCREEN_VALIDATED"
    assert out["n_artificial_present"] == 1
    found = out["controls_found"][0]
    assert found["percentile"] > 99.0
    assert found["expected_amr_m2_kg"] == pytest.approx(7.9e-3)


def test_control_validation_reports_an_insensitive_screen():
    """A present artificial object that was not flagged invalidates any null."""
    rows = [{"designation": f"FILLER {i}", "score": 0.01 * i, "flagged": False}
            for i in range(100)]
    rows.append({"designation": "2020 SO", "score": 0.02, "flagged": False})
    out = control.validate(rows, score_key="score")
    assert out["verdict"] == "SCREEN_INSENSITIVE"


def test_no_controls_present_is_unexercised_not_passed():
    """The expected common case, and it must not read as a pass."""
    rows = [{"designation": f"FILLER {i}", "score": 0.01 * i, "flagged": False}
            for i in range(20)]
    out = control.validate(rows, score_key="score")
    assert out["verdict"] == "NO_CONTROLS_PRESENT"
    assert "NOT passed" in out["note"]


def test_disputed_control_excluded_from_arithmetic():
    """1991 VG is genuinely disputed and must not decide a pass or a fail."""
    rows = [{"designation": f"FILLER {i}", "score": 0.01 * i, "flagged": False}
            for i in range(20)]
    rows.append({"designation": "1991 VG", "score": 99.0, "flagged": True})
    out = control.validate(rows, score_key="score")
    assert out["verdict"] == "NO_CONTROLS_PRESENT"
    assert any(c["designation"] == "1991 VG" for c in out["controls_found"])


def test_dark_comets_are_carried_as_confusers():
    """All seven Seligman objects present, with the source recorded."""
    assert len(control.DARK_COMETS) == 7
    idx = control.control_index()
    for d in ("1998 KY26", "2003 RM", "2006 RH120"):
        assert idx[control.normalise_designation(d)]["control_set"] == "dark_comet"
    assert "Seligman" in control.DARK_COMET_SOURCE


def test_unverified_control_values_are_none_not_invented():
    """Provenance discipline: a number that could not be verified stays absent."""
    by = {e["designation"]: e for e in control.ARTIFICIAL}
    assert by["2020 SO"]["amr_m2_kg"] is None
    assert "not in a reachable source" in by["2020 SO"]["source"]
    assert by["1991 VG"]["confidence"] == "disputed"


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------
def test_thresholds_dataclass_matches_config():
    """The failure that let TOCSIN screen a full backlog with a stale threshold.

    Every value in the ``screen`` block of ``config/loom.yaml`` must be a field of
    :class:`~seti.loom.screen.Thresholds`, and the dataclass defaults must equal
    the configured values --- otherwise the code and the documented configuration
    silently disagree.
    """
    conf = load_loom_config()
    th = thresholds_from_config(conf)
    defaults = screen.Thresholds()
    for key, value in conf["screen"].items():
        assert hasattr(defaults, key), key
        assert getattr(th, key) == pytest.approx(value) if isinstance(value, float) \
            else getattr(th, key) == value, key
        assert getattr(defaults, key) == pytest.approx(value) \
            if isinstance(value, float) else getattr(defaults, key) == value, key


def test_unknown_config_key_is_an_error():
    """A renamed config entry must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError):
        thresholds_from_config({"screen": {"not_a_real_threshold": 1.0}})


def test_defaults_cover_every_config_section():
    conf = load_loom_config()
    for section in DEFAULTS:
        assert section in conf
        for key in DEFAULTS[section]:
            assert key in conf[section], (section, key)


def test_epsilon_ladder_is_ordered():
    """realistic < hard < inviolable, or the tier ladder is meaningless."""
    th = screen.Thresholds()
    assert th.epsilon_realistic < th.epsilon_hard < th.epsilon_inviolable


def test_acquire_module_imports_without_network():
    """The module must be importable in the sandbox; only its calls need egress."""
    from seti.loom import acquire
    assert "ephoffsetalongtrack" in acquire.SS_DETECTION_COLUMNS
    assert "yarkovsky" in acquire.MPC_ORBIT_COLUMNS
    assert "srp" in acquire.MPC_ORBIT_COLUMNS
    # The offset decomposition is the channel's primary observable; if any of
    # these disappears from the schema the channel has no signal.
    for col in ("ephoffset", "ephoffsetra", "ephoffsetdec",
                "ephoffsetcrosstrack", "ephrate", "heliorange", "toporange",
                "diadistancerank", "designation", "ssobjectid"):
        assert col in acquire.SS_DETECTION_COLUMNS, col


def test_assess_end_to_end_recovers_a_planted_family(tmp_path):
    """The whole offline half of the channel, on an injected signal.

    Screen a synthetic parent population containing a planted family of objects
    with WT1190F-like area-to-mass ratios and clustered elements, write the table
    the way the screen stage does, and check the decision stage recovers it — and
    that it reports the positive control as unexercised rather than passed.
    """
    import pandas as pd

    from seti.loom.run import assess

    th = screen.Thresholds()
    rng = np.random.default_rng(77)
    rows = [_orbit_row(ssobjectid=str(i), designation=f"OBJ {i}",
                       h=float(rng.uniform(17, 25)),
                       a=float(rng.uniform(2.0, 3.3)),
                       e=float(rng.uniform(0, 0.3)),
                       i=float(rng.uniform(0, 20)),
                       node=float(rng.uniform(0, 360)),
                       normalized_rms=float(rng.uniform(0.5, 1.2)),
                       arc_length_total=float(rng.uniform(300, 6000)),
                       nopp=int(rng.integers(2, 12)),
                       yarkovsky=float(rng.normal(0, 1e-4)), yarkovsky_unc=1e-5)
            for i in range(800)]
    for k in range(25):
        rows[k].update({"h": 28.0 + float(rng.normal(0, 0.3)),
                        "srp": 11.8, "srp_unc": 0.4,
                        "a": 2.6 + float(rng.normal(0, 0.004)),
                        "e": 0.12 + float(rng.normal(0, 0.004)),
                        "i": 8.0 + float(rng.normal(0, 0.15)),
                        "node": 210.0 + float(rng.normal(0, 1.5))})
    recs, funnel = screen.screen_orbits(rows, th)
    assert funnel["verdict"] == "OK"
    assert funnel["n_candidate"] == 25

    df = pd.DataFrame([r.as_dict() for r in recs])
    for col in ("fit_reasons", "reasons", "orbit_reasons"):
        df[col] = df[col].apply(lambda v: "|".join(v) if isinstance(v, list) else v)
    (tmp_path / "objects.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path / "objects.csv", index=False)

    rec = assess(out_dir=tmp_path)
    assert rec["verdict"] == "REPLICATION_STRUCTURE_DETECTED", rec["verdict"]
    assert rec["n_anomalous"] == 25
    by = {t["name"]: t for t in rec["replication"]["tests"]}
    assert by["element_clustering"]["p_value"] < 0.01
    assert by["pole_coherence"]["p_value"] < 0.01
    # The anomaly score must not be tracking observation quality.
    assert "warning" not in rec["quality_independence"]
    # And the positive control is unexercised, which is honest, not a pass.
    assert rec["controls"]["verdict"] == "NO_CONTROLS_PRESENT"
    assert (tmp_path / "assessment.json").exists()


def test_assess_without_a_sample_degrades_explicitly(tmp_path):
    from seti.loom.run import assess

    rec = assess(out_dir=tmp_path)
    assert rec["verdict"] == "NO_SCREENED_SAMPLE"


def test_covariate_labels_stratify_on_detectability():
    """The matched null must control for what drives both detection and residual."""
    th = screen.Thresholds()
    rows = [_orbit_row(ssobjectid=str(i), h=16.0 + 0.02 * i,
                       arc_length_total=200.0 + 20.0 * i, nopp=2 + i % 7,
                       normalized_rms=0.5 + 0.005 * i) for i in range(400)]
    recs, _ = screen.screen_orbits(rows, th)
    labels = screen.covariate_labels(recs)
    assert labels.size == len(recs)
    assert len(set(labels.tolist())) > 8

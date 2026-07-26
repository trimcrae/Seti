"""Offline tests for the DERELICT channel.  No network.

Coverage required by the channel brief:

* the beta / area-to-mass conversions validated against 'Oumuamua's published
  A1 reproducing Bialy & Loeb 2018's ~0.1 g/cm^2 surface density;
* a natural 100 m asteroid that must NOT flag;
* a dark-comet-like object with large A2/A3 that must NOT pass the A1-only
  screen (this is the whole novelty argument, so it is tested directly);
* a negative-A1 case;
* honest degradation when the API returns nothing or an unexpected schema.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.derelict import acquire, census, radiation, screen, vet
from seti.derelict.radiation import (
    AMR_PER_A1,
    AMR_PER_BETA,
    BETA_PER_A1,
    OUMUAMUA_A1_AU_DAY2,
    REFERENCE_OBJECTS,
    amr_from_a1,
    amr_from_beta,
    amr_natural,
    areal_density_from_a1,
    beta_from_a1,
    beta_from_amr,
    diameter_from_h,
    r_statistic,
)
from seti.derelict.run import derelict_run
from seti.derelict.screen import ScreenParams, run_screens
from seti.derelict.vet import VetParams, vet_object


# =============================================================================
# 1. The conversions
# =============================================================================
def test_conversion_constants_match_published_values():
    """beta = 3379.4 * A1 and AMR = 4.4137e6 * A1, as derived in docs/derelict.md."""
    assert BETA_PER_A1 == pytest.approx(3379.4, rel=1e-4)
    assert AMR_PER_A1 == pytest.approx(4.4137e6, rel=1e-4)
    assert AMR_PER_BETA == pytest.approx(1306.1, rel=1e-4)


def test_oumuamua_reproduces_bialy_loeb_surface_density():
    """The calibration anchor of the entire channel.

    JPL/Micheli et al. 2018 give 1I/'Oumuamua a radial non-grav acceleration of
    4.92e-6 m/s^2 at 1 au.  Pushing that through A1 -> beta -> AMR must land on
    Bialy & Loeb 2018's quoted m/A ~ 0.1 g/cm^2 (agreement to within 10%).
    """
    beta = beta_from_a1(OUMUAMUA_A1_AU_DAY2)
    amr = amr_from_a1(OUMUAMUA_A1_AU_DAY2)
    sigma_g_cm2 = areal_density_from_a1(OUMUAMUA_A1_AU_DAY2) / 10.0  # kg/m^2 -> g/cm^2

    assert beta == pytest.approx(8.3e-4, rel=0.02)
    assert amr == pytest.approx(1.08, rel=0.02)
    assert sigma_g_cm2 == pytest.approx(0.1, rel=0.10), (
        f"implied surface density {sigma_g_cm2:.4f} g/cm^2 must match "
        "Bialy & Loeb 2018's ~0.1 g/cm^2")


def test_reference_table_matches_the_brief():
    """Every row of the docs/derelict.md benchmark table."""
    expect = {
        "natural_sphere_D100m_rho2000": (7.5e-6, 5.7e-9),
        "natural_sphere_D10m_rho2000": (7.5e-5, 5.7e-8),
        "oumuamua_if_pure_srp": (1.08, 8.3e-4),
        "ikaros_sailcraft": (0.63, 9.7e-4),
        "bare_mylar_1um": (714.0, 1.09),
    }
    got = {r.name: (r.amr_m2_kg, r.beta) for r in REFERENCE_OBJECTS}
    assert set(got) == set(expect)
    for name, (amr, beta) in expect.items():
        assert got[name][0] == pytest.approx(amr, rel=0.02), name
        assert got[name][1] == pytest.approx(beta, rel=0.02), name


def test_ikaros_and_oumuamua_sit_at_the_same_beta():
    """The headline claim: a real sailcraft and 'Oumuamua are indistinguishable
    in beta, and both are >=4 orders of magnitude above any natural 10-100 m body."""
    ikaros = {r.name: r for r in REFERENCE_OBJECTS}["ikaros_sailcraft"].beta
    oum = {r.name: r for r in REFERENCE_OBJECTS}["oumuamua_if_pure_srp"].beta
    nat100 = {r.name: r for r in REFERENCE_OBJECTS}["natural_sphere_D100m_rho2000"].beta
    nat10 = {r.name: r for r in REFERENCE_OBJECTS}["natural_sphere_D10m_rho2000"].beta

    assert 0.5 < ikaros / oum < 2.0, "IKAROS and 'Oumuamua must share a beta decade"
    assert oum / nat100 > 1e4
    assert oum / nat10 > 1e3


def test_conversions_round_trip():
    for a1 in (1e-9, 2.4551e-7, 1e-5):
        assert radiation.a1_from_beta(beta_from_a1(a1)) == pytest.approx(a1, rel=1e-12)
    for amr in (7.5e-6, 1.08, 714.0):
        for q in (1.0, 2.0):
            assert amr_from_beta(beta_from_amr(amr, q), q) == pytest.approx(amr, rel=1e-12)


def test_q_pr_two_halves_the_implied_area_to_mass():
    """A reflective sail needs only half the area-to-mass for the same beta, so
    reporting the Q_pr = 1 value is the conservative choice."""
    assert amr_from_a1(1e-7, q_pr=2.0) == pytest.approx(amr_from_a1(1e-7, q_pr=1.0) / 2)


def test_amr_natural_formula_and_h_diameter():
    assert amr_natural(100.0, 2000.0) == pytest.approx(7.5e-6)
    assert amr_natural(10.0, 2000.0) == pytest.approx(7.5e-5)
    # 1329 km / sqrt(p) * 10^(-H/5); H=20, p=0.15 -> ~343 m.
    assert diameter_from_h(20.0, 0.15) == pytest.approx(343.1, rel=0.01)
    # D ~ p^(-1/2)
    assert diameter_from_h(20.0, 0.60) == pytest.approx(diameter_from_h(20.0, 0.15) / 2,
                                                        rel=1e-9)


def test_conversions_are_vectorised():
    a1 = np.array([1e-9, 1e-8, 1e-7])
    assert np.allclose(beta_from_a1(a1), a1 * BETA_PER_A1)
    assert np.allclose(amr_from_a1(a1), a1 * AMR_PER_A1)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        amr_from_beta(1.0, q_pr=0.0)
    with pytest.raises(ValueError):
        amr_natural(0.0)
    with pytest.raises(ValueError):
        diameter_from_h(20.0, albedo=0.0)


# =============================================================================
# 2. The R statistic
# =============================================================================
def test_r_statistic_is_unity_for_a_natural_body():
    """A body whose A1 is exactly what its size and density predict has R = 1."""
    d, rho = 100.0, 1000.0
    amr_nat = amr_natural(d, rho)
    a1 = amr_nat / AMR_PER_A1  # the A1 such a body would show
    st = r_statistic(a1, diameter_m=d, rho_kg_m3=rho)
    assert st.valid
    assert st.r == pytest.approx(1.0, rel=1e-9)


def test_r_statistic_flags_oumuamua_enormously():
    st = r_statistic(OUMUAMUA_A1_AU_DAY2, diameter_m=100.0, rho_kg_m3=1000.0)
    assert st.valid
    assert st.r > 1e4
    assert st.areal_density_kg_m2 == pytest.approx(0.92, rel=0.05)


def test_r_statistic_albedo_bracket_orders_correctly():
    """R is linear in D, so a DARKER assumed albedo (bigger body) gives a
    LARGER R: the same acceleration on more mass is more anomalous."""
    st = r_statistic(1e-8, h_mag=22.0)
    assert st.valid and st.diameter_source == "H_albedo_assumed"
    assert st.r_lo < st.r < st.r_hi
    # The bracket must correspond to the configured albedo range.
    assert st.r_hi / st.r_lo == pytest.approx(np.sqrt(0.60 / 0.05), rel=1e-9)


def test_r_statistic_degrades_without_a_size():
    st = r_statistic(1e-8)
    assert not st.valid and "cannot normalise" in st.reason


def test_r_statistic_refuses_negative_a1():
    st = r_statistic(-1e-8, diameter_m=100.0)
    assert not st.valid
    assert "radiation pressure" in st.reason
    assert st.beta < 0


def test_r_statistic_refuses_only_on_real_outgassing_evidence():
    st = r_statistic(1e-7, diameter_m=100.0, outgassing_evidence=True)
    assert not st.valid and "outgassing" in st.reason


def test_has_outgassing_model_evidence():
    """Only DT is evidence; the Marsden shape parameters are not."""
    assert not radiation.has_outgassing_model_evidence(["A1", "A2"])
    assert not radiation.has_outgassing_model_evidence(None)
    assert not radiation.has_outgassing_model_evidence(["A1", "aln", "nm", "nn"])
    assert radiation.has_outgassing_model_evidence(["A1", "DT"])


def test_marsden_g_is_normalised_to_one_at_1au():
    """THE fact that makes this channel work on JPL's cometary-parameterised
    fits: the standard Marsden g(r) equals 1 at 1 au, exactly like 1/r^2, so A1
    is the radial acceleration at 1 au under BOTH laws."""
    assert radiation.marsden_g(1.0) == pytest.approx(1.0, abs=1e-6)
    # ...but the laws diverge sharply further out, which is what an astrometric
    # refit could exploit to separate them.
    assert radiation.marsden_g(5.0) < 1e-4 < 1.0 / 5.0**2


def test_g_at_1au_is_one_for_both_laws():
    assert radiation.g_at_1au(None) == pytest.approx(1.0)
    assert radiation.g_at_1au({"A1": None, "A2": None}) == pytest.approx(1.0)
    # The Marsden shape parameters, at their standard values, still give 1.
    assert radiation.g_at_1au({"ALN": None, "NM": None, "NN": None,
                               "NK": None, "R0": None}) == pytest.approx(1.0, abs=1e-6)


def test_marsden_parameterisation_does_not_change_the_implied_beta():
    """A Marsden-parameterised fit must yield the SAME beta as an
    inverse-square one for the same A1 -- that is the whole point."""
    a = r_statistic(1e-7, diameter_m=100.0, g_1au=1.0)
    b = r_statistic(1e-7, diameter_m=100.0, g_1au=radiation.g_at_1au({"ALN": None}))
    assert a.valid and b.valid
    assert b.beta == pytest.approx(a.beta, rel=1e-6)


# =============================================================================
# 3. The screens
# =============================================================================
def _params() -> ScreenParams:
    return ScreenParams()


def _row(**kw) -> dict:
    """A well-observed default object; override what the test cares about."""
    base = dict(full_name="(test)", pdes="test", kind="an", **{"class": "APO"},
                H=20.0, diameter=np.nan, albedo=np.nan,
                a=1.5, e=0.3, i=8.0,
                n_obs_used=500, data_arc=3000.0, condition_code=0, rms=0.4,
                first_obs="1998-01-01", last_obs="2024-01-01",
                A1=np.nan, A2=np.nan, A3=np.nan, DT=np.nan,
                sigma_A1=np.nan, sigma_A2=np.nan, sigma_A3=np.nan, sigma_DT=np.nan)
    base.update(kw)
    return base


def test_natural_100m_asteroid_does_not_flag():
    """MUST NOT FLAG: a 100 m body whose A1 is exactly its own radiation pressure.

    Its A1 is real and significant, so it legitimately passes screen 1 -- and
    that is precisely why screen 1 alone is not the discriminant.  The R
    statistic must place it at ~1 and no R screen may fire.
    """
    d_m = 100.0
    a1 = amr_natural(d_m, 1000.0) / AMR_PER_A1
    df = pd.DataFrame([_row(A1=a1, sigma_A1=a1 / 10.0,
                            A2=0.0, sigma_A2=1e-12, A3=0.0, sigma_A3=1e-12,
                            diameter=d_m / 1000.0)])
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["screen_a1_only"], "a real, well-measured A1 does pass the pre-filter"
    assert r["R"] == pytest.approx(1.0, rel=1e-6)
    assert not r["screen_r_flag"], "R ~ 1 must not raise any anomaly flag"
    assert not r["screen_r_strong"] and not r["screen_r_extreme"]
    assert res.funnel["screen2_r_flag"] == 0


def test_dark_comet_like_object_fails_the_a1_only_screen():
    """MUST NOT PASS: the population the dark-comet papers actually selected.

    Large non-radial A2/A3 is the signature that *excludes* radiation pressure,
    so an object like that must be rejected by screen 1 -- this is the entire
    complement-set argument, executed.
    """
    df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9,     # A1 significant at 10 sigma
                            A2=5e-9, sigma_A2=2e-10,    # A2 significant at 25 sigma
                            A3=3e-9, sigma_A3=2e-10)])  # A3 significant at 15 sigma
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["a2_state"] == screen.SIGNIFICANT
    assert r["a3_state"] == screen.SIGNIFICANT
    assert r["s1_a1_significant"], "its A1 is significant..."
    assert not r["s1_a2_zero"] and not r["s1_a3_zero"], "...but A2/A3 are not zero"
    assert not r["screen_a1_only"]
    assert res.funnel["screen1_a1_only"] == 0


def test_a1_only_complement_object_passes():
    """The target population: significant A1, A2 and A3 fitted and consistent
    with zero, clean orbit."""
    df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9,
                            A2=1e-12, sigma_A2=5e-11,
                            A3=1e-12, sigma_A3=5e-11)])
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["screen_a1_only"]
    assert r["screen_a1_only_strict"], "A2/A3 were actually fitted -> strict pass"
    assert r["nonradial_constrained"]
    assert res.funnel["screen1_a1_only_strict"] == 1


def test_unfitted_a2_a3_passes_but_is_not_counted_as_constrained():
    """'A2 was never fitted' is NOT evidence that A2 is zero, and the funnel must
    keep the two cases apart."""
    df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9)])  # A2/A3 absent
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["a2_state"] == screen.NOT_FITTED
    assert r["screen_a1_only"]
    assert not r["screen_a1_only_strict"]
    assert not r["nonradial_constrained"]
    assert res.funnel["screen1_a1_only"] == 1
    assert res.funnel["screen1_a1_only_strict"] == 0


def test_negative_a1_case():
    """Radiation pressure cannot push sunward: a significant A1 < 0 must be
    caught by screen 3 and must never reach screen 1."""
    df = pd.DataFrame([_row(A1=-3e-8, sigma_A1=3e-9)])
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["screen_negative_a1"]
    assert not r["screen_a1_only"]
    assert not r["R_valid"], "R is meaningless for a sunward acceleration"
    assert r["beta_implied"] < 0
    assert res.funnel["screen3_negative_a1"] == 1
    assert res.funnel["screen1_a1_only"] == 0


def test_toutatis_measures_the_empirical_false_positive_floor():
    """MEASURED, run 30204137011: the real negative-A1 census returned exactly
    one object, 4179 Toutatis, at A1/sigma = -3.39 SUNWARD.

    Radiation pressure cannot push sunward, so this is a clean measurement of
    the systematic floor of JPL's non-grav fits -- on one of the best-observed
    asteroids in the sky (92-year arc, 7141 observations, condition code 0).
    It converts to |R| ~ 5, and the channel's flag threshold is R = 10, so the
    thresholds sit a factor of ~2 ABOVE the measured noise rather than inside
    it.  If that margin ever closes, this test fails and the thresholds must be
    revisited.
    """
    a1, sig, d_km = -3.148561365334624e-13, 9.279e-14, 5.4
    df = pd.DataFrame([_row(full_name="  4179 Toutatis (1989 AC)", pdes="4179",
                            **{"class": "APO"}, H=15.29, diameter=d_km,
                            n_obs_used=7141, data_arc=33698, condition_code=0.0,
                            A1=a1, sigma_A1=sig)])
    p = _params()
    res = run_screens(df, p)
    r = res.table.iloc[0]

    assert r["screen_negative_a1"], "a 3.4-sigma sunward A1 must be caught"
    assert not r["screen_a1_only"], "and must never reach the science screen"
    assert r["a1_snr"] == pytest.approx(-3.39, abs=0.01)

    floor_r = abs(amr_from_a1(a1)) / amr_natural(d_km * 1000.0, p.rho_natural_kg_m3)
    assert floor_r == pytest.approx(5.0, rel=0.02)
    assert p.r_flag > floor_r, (
        f"flag threshold R={p.r_flag} must sit above the measured systematic "
        f"floor |R|={floor_r:.2f}")


def test_short_arc_garbage_is_killed_by_the_quality_gate():
    df = pd.DataFrame([_row(A1=5e-7, sigma_A1=1e-8, data_arc=3.0,
                            condition_code=9, n_obs_used=12)])
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["s1_a1_significant"] and not r["s1_quality"]
    assert not r["screen_a1_only"]


def test_missing_orbit_quality_fails_closed():
    """Absent metadata is not a pass: a spurious A1 is usually a short arc."""
    df = pd.DataFrame([_row(A1=5e-8, sigma_A1=5e-9, data_arc=np.nan,
                            condition_code=np.nan, n_obs_used=np.nan)])
    res = run_screens(df, _params())
    assert not res.table.iloc[0]["screen_a1_only"]


def test_comet_is_excluded_by_designation_and_by_DT():
    for kw in ({"full_name": "1P/Halley"}, {"full_name": "C/2017 U1"},
               {"full_name": "D/1993 F2"}, {"kind": "cn"},
               {"class": "HYP"}, {"DT": 12.0}):
        df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9, **kw)])
        res = run_screens(df, _params())
        assert not res.table.iloc[0]["screen_a1_only"], kw


def test_interstellar_I_prefix_is_not_treated_as_a_coma_report():
    """REGRESSION: 1I/'Oumuamua must not be excluded from its own channel.

    The "I" designation prefix means *interstellar*, not comet.  1I is
    designated 1I precisely because it showed NO coma -- it is the calibration
    anchor for every conversion in this package.  A cometary-prefix regex that
    swallows I/ would silently delete the exemplar, and any 'Oumuamua-like
    interloper, from the search.
    """
    df = pd.DataFrame([_row(full_name="1I/'Oumuamua (2017 U1)", **{"class": "HYA"},
                            diameter=0.100,
                            A1=OUMUAMUA_A1_AU_DAY2, sigma_A1=OUMUAMUA_A1_AU_DAY2 / 10,
                            A2=0.0, sigma_A2=1e-14, A3=0.0, sigma_A3=1e-14)])
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["s1_no_coma"], "the I/ prefix is not a coma report"
    assert r["screen_a1_only"], "'Oumuamua must pass its own channel's screen 1"
    assert r["screen_r_extreme"], "and must land in the extreme-R bin"
    assert r["R"] > 1e4


def test_genuinely_cometary_interstellar_object_is_still_excluded():
    """2I/Borisov IS a comet -- caught by kind/class even though I/ is allowed."""
    df = pd.DataFrame([_row(full_name="2I/Borisov (C/2019 Q4)", kind="cu",
                            A1=2e-8, sigma_A1=2e-9)])
    assert not run_screens(df, _params()).table.iloc[0]["screen_a1_only"]


def test_marsden_parameterisation_is_recorded_but_does_NOT_reject():
    """REGRESSION for run 30204137011, which discarded 20 of 22 objects.

    JPL's DEFAULT parameterisation for any object it fits A1 to -- including
    every dark comet -- is the Marsden g(r).  Treating its mere presence as
    "this object outgasses" removes the entire target population by
    construction.  g(1 au) = 1 for both laws, so the conversion is valid.
    """
    df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9, diameter=0.050,
                            A2=0.0, sigma_A2=1e-14, A3=0.0, sigma_A3=1e-14)])
    df["model_pars"] = [["A1", "A2", "ALN", "NM", "NN", "NK"]]
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["nongrav_law"] == "marsden_g"
    assert r["g_1au"] == pytest.approx(1.0, abs=1e-6)
    assert not r["outgassing_evidence"], "a parameterisation is not an observation"
    assert r["screen_a1_only"], "it must still be screened, not discarded"
    assert r["R_valid"]
    assert res.funnel["law_marsden_g"] == 1
    assert res.funnel["outgassing_evidence_excluded"] == 0


def test_fitted_time_delay_DT_does_reject():
    """A lagged response cannot be radiation pressure -- this one IS evidence."""
    df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9)])
    df["model_pars"] = [["A1", "A2", "ALN", "DT"]]
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["outgassing_evidence"]
    assert not r["screen_a1_only"]
    assert not r["R_valid"]
    assert res.funnel["outgassing_evidence_excluded"] == 1


def test_albedo_screen():
    df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9, albedo=0.85),
                       _row(A1=2e-8, sigma_A1=2e-9, albedo=0.20)])
    res = run_screens(df, _params())
    assert list(res.table["screen_albedo"]) == [True, False]
    assert res.funnel["screen4_albedo"] == 1


def test_albedo_screen_requires_significance_when_sigma_present():
    df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9, albedo=0.72),
                       _row(A1=2e-8, sigma_A1=2e-9, albedo=0.90)])
    df["albedo_sigma"] = [0.20, 0.02]   # first is 0.1 sigma above the cut, second 10
    res = run_screens(df, _params())
    assert list(res.table["screen_albedo"]) == [False, True]


def test_high_amr_film_flags_all_the_way_through():
    """Injected-signal recovery: an IKAROS-class sail at 30 m must reach R_extreme."""
    ikaros_amr = 196.0 / 310.0
    a1 = ikaros_amr / AMR_PER_A1
    df = pd.DataFrame([_row(A1=a1, sigma_A1=a1 / 20.0,
                            A2=0.0, sigma_A2=1e-12, A3=0.0, sigma_A3=1e-12,
                            diameter=0.030)])
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert r["screen_a1_only"] and r["screen_r_extreme"]
    assert r["R"] > 1e3
    assert r["amr_implied_m2_kg"] == pytest.approx(ikaros_amr, rel=1e-9)


# =============================================================================
# 4. Vetting
# =============================================================================
def test_known_human_hardware_is_flagged_not_celebrated():
    """2020 SO is a Centaur upper stage.  A pipeline that does not catch it is
    broken, so this doubles as a positive control."""
    row = pd.Series(_row(full_name="(2020 SO)", A1=1e-6, sigma_A1=1e-8,
                         a=1.06, e=0.03, i=0.5, R=5e4))
    v = vet_object(row, None, VetParams())
    assert v.verdict == vet.ARTIFICIAL_HUMAN
    assert "known_artificial" in v.flags


def test_earthlike_orbit_is_an_artificial_suspect():
    row = pd.Series(_row(full_name="(2099 ZZ)", A1=1e-6, sigma_A1=1e-8,
                         a=1.01, e=0.02, i=1.0, R=1e5, diameter_m=8.0))
    v = vet_object(row, None, VetParams())
    assert v.verdict == vet.ARTIFICIAL_SUSPECT
    assert "earthlike_orbit" in v.flags


def test_short_arc_survivor_is_vetted_out():
    row = pd.Series(_row(full_name="(x)", A1=1e-6, sigma_A1=1e-8,
                         a=2.5, e=0.4, i=15.0, data_arc=10.0, condition_code=7, R=1e5))
    assert vet_object(row, None, VetParams()).verdict == vet.SHORT_ARC


def test_fitted_DT_in_detail_gives_outgassing():
    detail = {"ok": True, "orbit": {"model_pars": [
        {"name": "A1", "value": "1e-8"}, {"name": "DT", "value": "12.0"}]}}
    row = pd.Series(_row(full_name="(y)", A1=1e-6, sigma_A1=1e-8, a=3.0, e=0.5, i=20.0,
                         R=1e5))
    assert vet_object(row, detail, VetParams()).verdict == vet.OUTGASSING


def test_marsden_pars_in_detail_are_flagged_but_do_not_reject():
    detail = {"ok": True, "orbit": {
        "model_pars": [{"name": "A1", "value": "1e-8", "sigma": "1e-10"},
                       {"name": "ALN", "value": "0.111"},
                       {"name": "A2", "value": "1e-13", "sigma": "1e-12"}],
        "covariance": {"labels": ["A1", "A2"], "data": [[1.0, 0.05], [0.05, 1.0]]}}}
    row = pd.Series(_row(full_name="(y2)", A1=1e-6, sigma_A1=1e-8, a=3.0, e=0.5,
                         i=20.0, R=1e5, diameter_m=200.0))
    v = vet_object(row, detail, VetParams())
    assert v.verdict != vet.OUTGASSING
    assert "marsden_g_parameterisation" in v.flags


def test_a1_a2_degeneracy_is_caught_from_the_covariance():
    detail = {"ok": True, "orbit": {
        "model_pars": [{"name": "A1", "value": "1e-8"}, {"name": "A2", "value": "1e-9"}],
        "covariance": {"labels": ["A1", "A2"],
                       "data": [[1.0, 0.99], [0.99, 1.0]]}}}
    row = pd.Series(_row(full_name="(z)", A1=1e-6, sigma_A1=1e-8, a=2.5, e=0.4, i=15.0,
                         R=1e5))
    v = vet_object(row, detail, VetParams())
    assert v.verdict == vet.YARKOVSKY_LEAK
    assert "a1_a2_degenerate" in v.flags


def test_ordinary_body_is_vetted_natural():
    row = pd.Series(_row(full_name="(w)", A1=1e-9, sigma_A1=1e-10, a=2.5, e=0.4,
                         i=15.0, R=1.2))
    assert vet_object(row, None, VetParams()).verdict == vet.NATURAL


def test_unexplained_requires_every_route_to_be_checked():
    detail = {"ok": True, "orbit": {
        "model_pars": [{"name": "A1", "value": "1e-8"}, {"name": "A2", "value": "1e-12"}],
        "covariance": {"labels": ["A1", "A2"], "data": [[1.0, 0.05], [0.05, 1.0]]}}}
    row = pd.Series(_row(full_name="(q)", A1=1e-6, sigma_A1=1e-8, a=2.5, e=0.4,
                         i=15.0, R=5e4, diameter_m=200.0))
    v = vet_object(row, detail, VetParams())
    assert v.verdict == vet.UNEXPLAINED
    assert "no_covariance" not in v.flags


def test_dedupe():
    df = pd.DataFrame({"spkid": [1, 1, 2], "A1": [1.0, 1.0, 2.0]})
    assert len(vet.dedupe(df)) == 2


def test_dedupe_uses_a_composite_key_and_never_merges_distinct_objects():
    """A degenerate identifier column must not delete real rows."""
    df = pd.DataFrame({"pdes": ["x", "x", "x"],
                       "full_name": ["(a)", "(b)", "(c)"], "A1": [1.0, 2.0, 3.0]})
    assert len(vet.dedupe(df)) == 3


# =============================================================================
# 5. Honest degradation
# =============================================================================
def _fake_transport(mapping: dict, default: bytes | Exception | None = None):
    def _t(url: str, timeout: float = 0.0) -> bytes:
        for key, val in mapping.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return val
        if isinstance(default, Exception):
            raise default
        if default is None:
            raise OSError("blocked")
        return default
    return _t


def test_fetch_degrades_to_no_data_reached_when_the_api_is_unreachable():
    res = acquire.fetch_nongrav_table(
        transport=_fake_transport({}, ConnectionError("CONNECT tunnel failed, 403")),
        tries=1)
    assert res.status == "NO_DATA_REACHED"
    assert not res.ok and res.n_rows == 0
    assert res.errors and any("403" in e for e in res.errors)


def test_fetch_falls_back_when_the_constraint_syntax_is_rejected():
    """If sb-cdata is not understood, the run must degrade to the unconstrained
    pull and filter client-side rather than reporting an empty sky."""
    payload = json.dumps({
        "signature": {"version": "1.0"},
        "fields": ["full_name", "A1", "sigma_A1"],
        "data": [["(a)", "1e-8", "1e-9"], ["(b)", None, None]],
    }).encode()

    def _t(url: str, timeout: float = 0.0) -> bytes:
        if "cdata" in url:
            raise OSError("400 Bad Request: invalid constraint")
        return payload

    res = acquire.fetch_nongrav_table(transport=_t, tries=1)
    assert res.ok
    assert res.strategy == "unconstrained_full_pull"
    assert res.n_rows == 1, "the row with a null A1 must be filtered client-side"
    assert any("client-side A1 filter" in e for e in res.errors)


def test_unconstrained_fallback_requests_only_the_minimal_field_set():
    """A full-column pull over ~1.4M rows would OOM the runner, so the last
    resort must ask for the identity + A1 columns only."""
    seen: list[str] = []
    payload = json.dumps({"fields": ["full_name", "A1"], "data": [["(a)", "1e-8"]]}).encode()

    def _t(url: str, timeout: float = 0.0) -> bytes:
        seen.append(url)
        if "cdata" in url:
            raise OSError("400 Bad Request")
        return payload

    res = acquire.fetch_nongrav_table(transport=_t, tries=1)
    assert res.strategy == "unconstrained_full_pull"
    final = seen[-1]
    assert "diameter" not in final and "condition_code" not in final
    assert "A1" in final and "full_name" in final
    assert any("must be enriched per object" in e for e in res.errors)


def test_enrich_from_details_fills_missing_columns_without_overwriting():
    df = pd.DataFrame([{"full_name": "(a)", "A1": 1e-8, "H": np.nan, "a": np.nan,
                        "condition_code": np.nan, "data_arc": np.nan,
                        "n_obs_used": np.nan, "diameter": 0.5}])
    details = {"(a)": {"ok": True,
                       "orbit": {"data_arc": "4000", "condition_code": "1",
                                 "n_obs_used": "812",
                                 "elements": [{"name": "a", "value": "2.4"},
                                              {"name": "e", "value": "0.3"}]},
                       "phys_par": [{"name": "H", "value": "19.5"},
                                    {"name": "diameter", "value": "9.9"}]}}
    out = acquire.enrich_from_details(df, details)
    assert out.loc[0, "data_arc"] == 4000
    assert out.loc[0, "condition_code"] == 1
    assert out.loc[0, "n_obs_used"] == 812
    assert out.loc[0, "H"] == 19.5
    assert out.loc[0, "a"] == 2.4
    assert out.loc[0, "diameter"] == 0.5, "an existing bulk value is never overwritten"


def test_enrich_from_details_ignores_failed_records():
    df = pd.DataFrame([{"full_name": "(a)", "A1": 1e-8, "data_arc": np.nan}])
    out = acquire.enrich_from_details(df, {"(a)": {"ok": False, "error": "blocked"}})
    assert pd.isna(out.loc[0, "data_arc"])


def test_invalid_field_error_is_parsed_from_the_server_body():
    """MEASURED against the real API: a single bad field name 400s the whole
    query, so the offending name must be recoverable from the error body."""
    body = ("HTTP 400 for https://...: "
            '{"message":"invalid field specified: \'sigma_A1\'","code":"400"}')
    assert acquire._invalid_fields_from_error(body) == {"sigma_A1"}
    multi = '{"message":"invalid fields specified: \'sigma_A1\', \'sigma_A2\'"}'
    assert acquire._invalid_fields_from_error(multi) == {"sigma_A1", "sigma_A2"}
    assert acquire._invalid_fields_from_error("some other error") == set()
    assert acquire._invalid_fields_from_error(None) == set()


def test_fetch_self_heals_by_dropping_the_field_the_server_rejects():
    """REGRESSION for run 30203392288: `sigma_A1` is not a valid SBDB field, and
    one bad name turned the whole search into '0 rows in the database'.  The
    fetch must drop it and retry rather than reporting an empty sky."""
    good = json.dumps({"fields": ["full_name", "A1"],
                       "data": [["(a)", "1e-8"], ["(b)", "2e-8"]]}).encode()

    def _t(url: str, timeout: float = 0.0) -> bytes:
        if "sigma_A1" in url:
            raise OSError('HTTP 400: {"message":"invalid field specified: '
                          "'sigma_A1'\",\"code\":\"400\"}")
        if "sigma_A2" in url:
            raise OSError('HTTP 400: {"message":"invalid field specified: '
                          "'sigma_A2'\",\"code\":\"400\"}")
        if "sigma_A3" in url:
            raise OSError('HTTP 400: {"message":"invalid field specified: '
                          "'sigma_A3'\",\"code\":\"400\"}")
        if "sigma_DT" in url:
            raise OSError('HTTP 400: {"message":"invalid field specified: '
                          "'sigma_DT'\",\"code\":\"400\"}")
        return good

    res = acquire.fetch_nongrav_table(transport=_t, tries=1)
    assert res.ok, res.errors
    assert res.n_rows == 2
    assert set(res.fields_rejected) >= {"sigma_A1"}
    assert res.strategy == "cdata_A1_defined", "self-healed on the FIRST strategy"


def test_fetch_never_drops_a_required_field():
    """If the server rejects A1 itself there is nothing to search for; the
    strategy must fail rather than silently querying without it."""
    def _t(url: str, timeout: float = 0.0) -> bytes:
        raise OSError('HTTP 400: {"message":"invalid field specified: \'A1\'"}')

    res = acquire.fetch_nongrav_table(transport=_t, tries=1)
    assert res.status == acquire.STATUS_UNREACHABLE
    assert "A1" not in res.fields_rejected


def test_missing_a1_column_is_reported_as_a_query_defect_not_an_empty_sky():
    payload = json.dumps({"fields": ["full_name"], "data": [["(a)"], ["(b)"]]}).encode()
    res = acquire.fetch_nongrav_table(transport=_fake_transport({"": payload}), tries=1)
    assert res.status == acquire.STATUS_A1_FIELD_REJECTED
    assert res.n_rows_raw == 2, "the server DID return rows -- the query is at fault"
    assert res.a1_column_present is False
    assert not res.ok
    assert any("QUERY defect" in e for e in res.errors)
    # ...and the screens must then refuse to score it rather than emit anything.
    assert len(run_screens(res.table, _params()).table) == 0


def test_all_null_a1_column_is_distinguished_from_a_missing_one():
    payload = json.dumps({"fields": ["full_name", "A1"],
                          "data": [["(a)", None], ["(b)", None]]}).encode()

    def _t(url: str, timeout: float = 0.0) -> bytes:
        if "cdata" in url:
            raise OSError("400 Bad Request")
        return payload

    res = acquire.fetch_nongrav_table(transport=_t, tries=1)
    assert res.status == acquire.STATUS_A1_ALL_NULL
    assert res.a1_column_present is True
    assert res.n_rows_raw == 2 and res.n_rows == 0


def test_enrich_pulls_nongrav_sigmas_from_model_pars():
    """The bulk query rejects sigma_A1, so orbit.model_pars is the ONLY source
    of the uncertainties every SNR screen depends on."""
    df = pd.DataFrame([{"full_name": "(a)", "A1": 1e-8}])
    details = {"(a)": {"ok": True, "orbit": {"model_pars": [
        {"name": "A1", "value": "1.5e-8", "sigma": "2e-10"},
        {"name": "A2", "value": "-3e-12", "sigma": "4e-12"},
    ]}}}
    out = acquire.enrich_from_details(df, details)
    assert out.loc[0, "A1"] == 1e-8, "the bulk value wins where it exists"
    assert out.loc[0, "sigma_A1"] == pytest.approx(2e-10)
    assert out.loc[0, "A2"] == pytest.approx(-3e-12)
    assert out.loc[0, "sigma_A2"] == pytest.approx(4e-12)
    assert out.loc[0, "model_pars"] == ["A1", "A2"]


def test_enriched_sigmas_make_the_screens_runnable_end_to_end():
    """Bulk pull with NO sigmas -> enrich -> screens work. This is the whole
    recovery path from run 30203392288."""
    df = pd.DataFrame([{"full_name": "(film)", "pdes": "film", "kind": "an",
                        "class": "APO", "A1": 2e-7, "diameter": 0.030,
                        "a": 2.4, "e": 0.45, "i": 22.0}])
    details = {"(film)": {"ok": True, "orbit": {
        "data_arc": "2000", "condition_code": "1", "n_obs_used": "300",
        "model_pars": [{"name": "A1", "value": "2e-7", "sigma": "1e-8"},
                       {"name": "A2", "value": "0", "sigma": "1e-13"},
                       {"name": "A3", "value": "0", "sigma": "1e-13"}]}}}
    enriched = acquire.enrich_from_details(df, details)
    res = run_screens(enriched, _params())
    r = res.table.iloc[0]
    assert r["screen_a1_only"], res.funnel
    assert r["screen_a1_only_strict"], "A2/A3 sigmas came from model_pars"
    assert r["screen_r_extreme"]


def test_fetch_reports_an_unexpected_schema_rather_than_inventing_rows():
    bad = json.dumps({"message": "invalid field: A1"}).encode()
    res = acquire.fetch_nongrav_table(transport=_fake_transport({"": bad}), tries=1)
    assert res.status == "NO_DATA_REACHED"
    assert any("SBDB error" in e or "ValueError" in e for e in res.errors)


def test_object_detail_degrades_without_raising():
    d = acquire.fetch_object_detail(
        "433", transport=_fake_transport({}, OSError("blocked")), tries=1)
    assert d["ok"] is False and d["error"]


def test_run_screens_on_empty_input():
    res = run_screens(pd.DataFrame(), _params())
    assert res.funnel == {"input": 0}
    assert res.notes and len(res.table) == 0


def test_run_screens_without_an_A1_column():
    res = run_screens(pd.DataFrame({"full_name": ["x"], "H": [20.0]}), _params())
    assert len(res.table) == 0
    assert any("no A1 column" in n for n in res.notes)


def test_derelict_run_emits_no_data_reached_when_the_archive_is_blocked(tmp_path):
    from seti.config import load_config

    cfg = load_config()
    cfg.root = tmp_path              # write results into a scratch tree
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    summary = derelict_run(cfg, transport=_fake_transport({}, OSError("blocked")),
                           skip_control=True, tries=1)
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["funnel"]["input"] == 0
    # The funnel must let a reader tell a query failure from an empty sky.
    assert summary["funnel"]["rows_returned_by_server"] == 0
    assert summary["funnel"]["a1_column_present"] is False
    assert any("QUERY DEFECT" in d for d in summary["degradation"])
    out = json.loads((tmp_path / "results" / "derelict" / "summary.json").read_text())
    assert out["verdict"] == "NO_DATA_REACHED"
    assert json.loads((tmp_path / "results" / "derelict" / "candidates.json").read_text()) == []


def test_empty_offline_input_is_never_an_OK_verdict(tmp_path):
    """An empty table cannot be a success, whatever the fetch wrapper reports."""
    from seti.config import load_config

    src = tmp_path / "empty.csv"
    pd.DataFrame(columns=["full_name", "A1", "sigma_A1"]).to_csv(src, index=False)
    cfg = load_config()
    cfg.root = tmp_path
    summary = derelict_run(cfg, offline_input=str(src), skip_control=True)
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["funnel"]["input"] == 0


def test_derelict_run_offline_input_end_to_end(tmp_path):
    """The full funnel on a synthetic table containing one of each species."""
    from seti.config import load_config

    rows = [
        # a natural 100 m body: R = 1, must not flag
        _row(full_name="(natural100)", diameter=0.100,
             A1=amr_natural(100.0, 1000.0) / AMR_PER_A1,
             sigma_A1=amr_natural(100.0, 1000.0) / AMR_PER_A1 / 10,
             A2=0.0, sigma_A2=1e-12, A3=0.0, sigma_A3=1e-12),
        # a dark-comet-like object: large A2/A3, must not pass screen 1
        _row(full_name="(darkcomet)", A1=2e-8, sigma_A1=2e-9,
             A2=5e-9, sigma_A2=2e-10, A3=3e-9, sigma_A3=2e-10),
        # a negative-A1 object: screen 3 only
        _row(full_name="(sunward)", A1=-3e-8, sigma_A1=3e-9),
        # a film-like object on a non-Earth orbit: the target
        _row(full_name="(film)", diameter=0.030, a=2.4, e=0.45, i=22.0,
             A1=(196.0 / 310.0) / AMR_PER_A1, sigma_A1=(196.0 / 310.0) / AMR_PER_A1 / 20,
             A2=0.0, sigma_A2=1e-14, A3=0.0, sigma_A3=1e-14),
    ]
    src = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(src, index=False)

    cfg = load_config()
    cfg.root = tmp_path
    summary = derelict_run(cfg, offline_input=str(src), skip_control=True)

    f = summary["funnel"]
    assert f["input"] == 4
    assert f["a1_fitted"] == 4
    assert f["screen1_a1_only"] == 2          # natural100 and film
    assert f["screen2_r_extreme"] == 1        # only film
    assert f["outgassing_evidence_excluded"] == 0
    assert f["screen3_negative_a1"] == 1      # only sunward
    assert summary["verdict"] in {"CANDIDATES_UNEXPLAINED", "ALL_SURVIVORS_EXPLAINED"}

    cands = json.loads((tmp_path / "results" / "derelict" / "candidates.json").read_text())
    names = [c["full_name"] for c in cands]
    assert "(film)" in names and "(darkcomet)" not in names
    top = cands[0]
    assert top["full_name"] == "(film)"
    assert top["verdict"] == vet.UNEXPLAINED
    assert (tmp_path / "results" / "derelict" / "REPORT.md").read_text().startswith(
        "# DERELICT")


# =============================================================================
# 6. The query log — QUERY_FAILED must never look like QUERY_RETURNED_ZERO_ROWS
# =============================================================================
def _payload(fields: list[str], data: list[list], count: int | None = None) -> bytes:
    body: dict = {"signature": {"version": "1.0"}, "fields": fields, "data": data}
    if count is not None:
        body["count"] = count
    return json.dumps(body).encode()


def test_a_failed_query_is_recorded_as_QUERY_FAILED_with_its_http_status():
    """THE guard. A 400 on a mistyped field and an empty database produced the
    same visible outcome in run 30203392288. They must now be distinguishable
    from the log alone, without reading any prose."""
    log = acquire.QueryLog()

    def _t(url: str, timeout: float = 0.0) -> bytes:
        raise OSError('HTTP 400 Bad Request: {"message":"invalid field '
                      "specified: 'A1'\",\"code\":\"400\"}")

    res = acquire.fetch_nongrav_table(transport=_t, tries=1, log=log)
    assert res.status == acquire.STATUS_UNREACHABLE
    assert log.records, "every query must be recorded"
    assert all(r.status == acquire.STATUS_QUERY_FAILED for r in log.records)
    assert all(r.http_status == 400 for r in log.records), (
        "the HTTP code must survive into the log even from a non-HTTPError "
        "transport, or 'the server said no' is indistinguishable from 'the "
        "network never got there'")
    # The literal URL is recorded, unredacted, so the query can be reproduced.
    assert all("ssd-api.jpl.nasa.gov" in r.url and "fields=" in r.url
               for r in log.records)
    assert res.queries and res.queries[0]["status"] == acquire.STATUS_QUERY_FAILED


def test_a_successful_but_empty_query_is_QUERY_RETURNED_ZERO_ROWS():
    """The other half of the distinction: the server answered, and the answer
    was 'no rows'. That is a real measurement, not a failure."""
    log = acquire.QueryLog()
    empty = _payload(["full_name", "A1"], [], count=0)
    res = acquire.fetch_nongrav_table(
        transport=_fake_transport({"": empty}), tries=1, log=log)

    assert [r.status for r in log.records] == [
        acquire.STATUS_ZERO_ROWS] * len(log.records)
    assert all(r.http_status == 200 and r.n_rows == 0 for r in log.records)
    assert acquire.STATUS_QUERY_FAILED not in log.counts()
    # ...and the fetch as a whole still refuses to call an empty sky a success.
    assert not res.ok


def test_transport_failure_records_no_http_status_rather_than_inventing_one():
    log = acquire.QueryLog()
    acquire.fetch_nongrav_table(
        transport=_fake_transport({}, ConnectionError("CONNECT tunnel failed")),
        tries=1, log=log)
    assert all(r.http_status is None for r in log.records)
    assert all(r.status == acquire.STATUS_QUERY_FAILED for r in log.records)


def test_object_detail_records_an_unresolvable_designation_as_zero_rows():
    """'No such object' is an answer; 'the archive is down' is not."""
    log = acquire.QueryLog()
    missing = json.dumps({"message": "specified object was not found"}).encode()
    d = acquire.fetch_object_detail("(9999 XX)", transport=_fake_transport({"": missing}),
                                    tries=1, log=log)
    assert d["ok"] is False
    assert log.records[-1].status == acquire.STATUS_ZERO_ROWS
    assert log.records[-1].http_status == 200

    log2 = acquire.QueryLog()
    acquire.fetch_object_detail("433", transport=_fake_transport({}, OSError("blocked")),
                                tries=1, log=log2)
    assert log2.records[-1].status == acquire.STATUS_QUERY_FAILED


def test_summary_carries_the_full_query_log(tmp_path):
    from seti.config import load_config

    cfg = load_config()
    cfg.root = tmp_path
    summary = derelict_run(cfg, transport=_fake_transport({}, OSError("blocked")),
                           skip_control=True, tries=1)
    assert summary["queries"], "summary.json must carry the queries list"
    first = summary["queries"][0]
    assert set(first) >= {"label", "url", "http_status", "status", "n_rows"}
    assert summary["query_status_counts"][acquire.STATUS_QUERY_FAILED] > 0
    # The complete, untruncated log is on disk as well.
    full = json.loads((tmp_path / "results" / "derelict" / "queries.json").read_text())
    assert len(full) >= len(summary["queries"])


# =============================================================================
# 7. Completeness of the A1|DF constraint
# =============================================================================
def test_completeness_probe_confirms_a_complete_constraint():
    """The decisive check: 22 rows is only a census if the unconstrained pull
    finds exactly the same 22 objects with a non-null A1."""
    constrained = pd.DataFrame([
        {"spkid": "2000433", "pdes": "433", "full_name": "433 Eros", "A1": 1e-10},
        {"spkid": "3000001", "pdes": "2005 VL1", "full_name": "(2005 VL1)",
         "A1": -8.3e-10},
    ])
    everything = _payload(
        ["spkid", "full_name", "pdes", "A1"],
        [["2000433", "433 Eros", "433", "1e-10"],
         ["3000001", "(2005 VL1)", "2005 VL1", "-8.3e-10"],
         ["2000001", "1 Ceres", "1", None],          # no A1 -> not in the census
         ["2000004", "4 Vesta", "4", None]],
        count=4)
    res = census.completeness_probe(
        "a", constrained, transport=_fake_transport({"": everything}), tries=1)

    assert res.verdict == census.CONSTRAINT_COMPLETE
    assert res.n_constrained == 2
    assert res.n_unconstrained_rows == 4
    assert res.n_unconstrained_nonnull_A1 == 2
    assert res.missing_from_constrained == []
    assert res.extra_in_constrained == []


def test_completeness_probe_catches_an_incomplete_constraint():
    """If the unconstrained pull finds an A1 the constraint missed, the census
    is NOT complete and the primary path has to change."""
    constrained = pd.DataFrame([
        {"spkid": "2000433", "pdes": "433", "full_name": "433 Eros", "A1": 1e-10}])
    everything = _payload(
        ["spkid", "full_name", "pdes", "A1"],
        [["2000433", "433 Eros", "433", "1e-10"],
         ["2000999", "(2099 ZZ)", "2099 ZZ", "5e-9"]],   # missed by the constraint
        count=2)
    res = census.completeness_probe(
        "a", constrained, transport=_fake_transport({"": everything}), tries=1)

    assert res.verdict == census.CONSTRAINT_INCOMPLETE
    assert res.missing_from_constrained == ["2000999"]
    assert any("must switch to the" in n for n in res.notes)


def test_completeness_probe_does_not_confuse_identifier_FORMATS_with_absences():
    """SBDB prints the same object as '(2005 VL1)' and '2005 VL1'. A formatting
    difference must never be reported as a missing object."""
    constrained = pd.DataFrame([{"full_name": "(2005 VL1)", "A1": 1e-10}])
    everything = _payload(["pdes", "A1"], [["2005 VL1", "1e-10"]], count=1)
    res = census.completeness_probe(
        "a", constrained, transport=_fake_transport({"": everything}), tries=1)
    assert res.verdict == census.CONSTRAINT_COMPLETE


def test_completeness_probe_matches_on_ANY_identifier_not_just_the_primary():
    """REGRESSION. The two pulls need not lead with the same identifier column:
    the constrained one may carry only `pdes` while the unconstrained one leads
    with `spkid`. Comparing raw key sets would then report a perfectly matched
    object as BOTH missing and extra, manufacturing a false
    CONSTRAINT_INCOMPLETE out of a schema difference."""
    constrained = pd.DataFrame([{"pdes": "433", "full_name": "433 Eros",
                                 "A1": 1e-10}])
    everything = _payload(
        ["spkid", "full_name", "pdes", "A1"],
        [["2000433", "433 Eros", "433", "1e-10"]], count=1)
    res = census.completeness_probe(
        "a", constrained, transport=_fake_transport({"": everything}), tries=1)
    assert res.missing_from_constrained == []
    assert res.extra_in_constrained == []
    assert res.verdict == census.CONSTRAINT_COMPLETE


def test_completeness_probe_reports_PROBE_FAILED_rather_than_agreement():
    """An unreachable probe is UNTESTED. It must never be reported as 'the sets
    agree' just because no disagreement was observed."""
    res = census.completeness_probe(
        "a", pd.DataFrame([{"pdes": "433", "A1": 1e-10}]),
        transport=_fake_transport({}, OSError("blocked")), tries=1,
        class_chunks=("APO", "AMO"))
    assert res.verdict == census.PROBE_FAILED
    assert any("UNTESTED" in n for n in res.notes)


def test_completeness_probe_flags_a_truncated_pull():
    """A server count that disagrees with the parsed row count means the pull
    may be truncated -- and a 'complete' verdict would then be unsafe."""
    constrained = pd.DataFrame([{"pdes": "433", "A1": 1e-10}])
    everything = _payload(["pdes", "A1"], [["433", "1e-10"]], count=1400000)
    res = census.completeness_probe(
        "a", constrained, transport=_fake_transport({"": everything}), tries=1)
    assert any("truncated" in n for n in res.notes)
    assert res.server_count == 1400000


def test_completeness_probe_falls_back_to_class_chunks():
    calls: list[str] = []

    def _t(url: str, timeout: float = 0.0) -> bytes:
        calls.append(url)
        if "sb-class" not in url:
            raise OSError("504 Gateway Timeout")
        cls = url.split("sb-class=")[1].split("&")[0]
        if cls == "APO":
            return _payload(["pdes", "A1"], [["433", "1e-10"]], count=1)
        return _payload(["pdes", "A1"], [], count=0)

    res = census.completeness_probe(
        "a", pd.DataFrame([{"pdes": "433", "A1": 1e-10}]), transport=_t, tries=1,
        class_chunks=("APO", "AMO", "MBA"))
    assert res.strategy == "unconstrained_by_class"
    assert res.verdict == census.CONSTRAINT_COMPLETE
    assert res.n_unconstrained_nonnull_A1 == 1


# =============================================================================
# 8. The dark-comet named-target census
# =============================================================================
def _detail(**model_pars) -> dict:
    """An sbdb.api record carrying the fitted non-grav parameters."""
    pars = [{"name": n, "value": str(v), "sigma": str(s)}
            for n, (v, s) in model_pars.items()]
    return {"ok": True,
            "object": {"fullname": "(2016 NJ33)", "des": "2016 NJ33",
                       "spkid": "3752158", "kind": "au",
                       "orbit_class": {"code": "AMO"}},
            "orbit": {"data_arc": "2261", "condition_code": "3",
                      "n_obs_used": "90", "model_pars": pars,
                      "elements": [{"name": "a", "value": "1.31"},
                                   {"name": "e", "value": "0.21"},
                                   {"name": "i", "value": "6.64"}]},
            "phys_par": [{"name": "H", "value": "25.53"}]}


def test_dark_comet_R_reproduces_the_published_numbers_for_2016_NJ33():
    """A KNOWN dark comet, end to end, against numbers read verbatim from
    Seligman et al. 2024 Table 2 (A1 = 9.48e-10 +/- 2.93e-10 au/d^2, sigma =
    3.24) and JPL's own H = 25.53.

    Two things are asserted at once. (1) The A1 -> beta -> AMR -> R chain lands
    on R ~ 75 for this object. (2) It still FAILS screen 1 -- its A3 is a 5-sigma
    non-radial acceleration, which is exactly the signature that excludes
    radiation pressure. That is the channel's whole complement-set argument,
    executed on a real member of the published dark-comet sample.
    """
    detail = _detail(A1=(9.475402e-10, 2.928e-10),
                     A2=(-5.486101e-13, 1.909e-13),
                     A3=(8.49e-11, 1.63e-11))

    def _fetcher(desig, **kw):
        return detail

    table, summ = census.dark_comet_census(
        [("2016 NJ33", "seligman2023_psj")], params=_params(),
        detail_fetcher=_fetcher)
    r = table.iloc[0]

    assert bool(r["resolved"])
    assert r["A1"] == pytest.approx(9.475402e-10)
    assert r["sigma_A1"] == pytest.approx(2.928e-10)
    assert r["a1_snr"] == pytest.approx(3.24, abs=0.01), (
        "must reproduce the significance the paper itself prints")
    assert r["R"] == pytest.approx(74.95, rel=0.01)
    assert r["amr_implied_m2_kg"] == pytest.approx(
        4.4137e6 * 9.475402e-10, rel=1e-3)

    assert r["a3_state"] == screen.SIGNIFICANT
    assert not r["screen_a1_only"], (
        "a 5-sigma NON-RADIAL acceleration is the dark-comet signature and "
        "excludes radiation pressure -- this object is the complement, not the "
        "target")
    assert summ["n_with_A1_and_sigma"] == 1
    assert summ["n_a1_only"] == 0
    assert summ["n_unresolved"] == 0


def test_dark_comet_with_no_fitted_A1_is_reported_not_dropped():
    """Several of the 14 may have no JPL A1 at all. That is a finding about
    JPL's solutions, and it has to be visible."""
    def _fetcher(desig, **kw):
        return {"ok": True, "object": {"fullname": "(2001 ME1)", "des": "2001 ME1"},
                "orbit": {"data_arc": "8000", "condition_code": "0",
                          "n_obs_used": "500", "model_pars": []}}

    table, summ = census.dark_comet_census(
        [("2001 ME1", "seligman2024_pnas")], params=_params(),
        detail_fetcher=_fetcher)
    assert summ["n_resolved"] == 1
    assert summ["n_with_A1_and_sigma"] == 0
    assert summ["no_A1_fitted"] == ["2001 ME1"]
    assert len(table) == 1, "the target stays in the census table"


def test_unresolvable_dark_comet_designation_is_reported_not_silently_dropped():
    """A designation that SBDB cannot resolve is a defect in our list -- exactly
    the class of error that put two wrong papers in this repository."""
    def _fetcher(desig, **kw):
        return {"ok": False, "error": "specified object was not found"}

    table, summ = census.dark_comet_census(
        [("2016 RH120", "typo_in_the_2023_abstract")], params=_params(),
        detail_fetcher=_fetcher)
    assert summ["n_unresolved"] == 1
    assert summ["unresolved"] == ["2016 RH120"]
    assert len(table) == 1
    assert not bool(table.iloc[0]["resolved"])
    assert "not found" in str(table.iloc[0]["detail_error"])


def test_row_from_detail_lifts_the_sigmas_that_only_sbdb_api_has():
    row = census.row_from_detail("2016 NJ33", _detail(A1=(9.5e-10, 2.9e-10)))
    assert row["resolved"] is True
    assert row["sigma_A1"] == pytest.approx(2.9e-10)
    assert row["condition_code"] == 3
    assert row["n_obs_used"] == 90
    assert row["H"] == pytest.approx(25.53)
    assert row["class"] == "AMO"
    assert row["a"] == pytest.approx(1.31)


def test_configured_dark_comet_list_matches_the_fetched_literature():
    """The list is CONFIG, with provenance, because two identifiers recalled
    from memory have already cost this repository two wrong papers."""
    import yaml as _yaml

    from seti.config import load_config
    cfg = load_config()
    d = _yaml.safe_load((cfg.root / "config" / "derelict.yaml").read_text())
    cp = census.CensusParams.from_config(d)
    names = {n for n, _ in cp.dark_comets}

    assert len(cp.dark_comets) == 14, "Seligman et al. 2024 report 14 dark comets"
    # The seven of Seligman et al. 2023 Table 1, read verbatim from the fetched
    # text in results/derelictlit/txt_seligman2023_dark_comets.txt.
    assert {"1998 KY26", "2005 VL1", "2016 NJ33", "2010 VL65", "2010 RF12",
            "2006 RH120", "2003 RM"} <= names
    # ...and the seven added by the 2024 PNAS paper.
    assert {"2001 ME1", "2005 UY6", "1998 FR11", "2012 UR158", "2013 BA74",
            "2016 GW221", "2013 XY20"} <= names
    assert "2016 RH120" not in names, (
        "the 2023 ABSTRACT prints 2016 RH120; its own Table 1 prints 2006 RH120, "
        "and the typo must not propagate into the census")
    assert all(src in {"seligman2023_psj", "seligman2024_pnas"}
               for _, src in cp.dark_comets), "every target carries its source"


# =============================================================================
# 9. The negative-A1 census as a rate with its denominator
# =============================================================================
def test_negative_a1_census_reports_a_rate_with_its_denominator():
    """A count is uninterpretable: 1 in 22 and 1 in 272 are different
    measurements of the systematic floor."""
    ast = pd.DataFrame([
        _row(full_name="4179 Toutatis (1989 AC)", diameter=5.4,
             A1=-3.148561e-13, sigma_A1=9.279e-14),
        _row(full_name="(ordinary)", A1=2e-8, sigma_A1=2e-9),
    ])
    com = pd.DataFrame([
        _row(full_name="1P/Halley", kind="cn", A1=-4e-9, sigma_A1=5e-10),
        _row(full_name="2P/Encke", kind="cn", A1=3e-9, sigma_A1=5e-10),
        _row(full_name="9P/Tempel", kind="cn", A1=-1e-9, sigma_A1=1e-9),  # 1 sigma
    ])
    p = _params()
    ast_s = run_screens(ast, p).table
    com_s = run_screens(com, p).table

    table, summ = census.negative_a1_census(
        {"asteroid": ast_s, "comet": com_s}, p)

    assert set(table["population"]) == {"asteroid", "comet"}
    a = summ["populations"]["asteroid"]
    c = summ["populations"]["comet"]
    assert a["n_negative"] == 1 and a["n_a1_fitted"] == 2
    assert a["rate"] == pytest.approx(0.5)
    assert c["n_negative"] == 1 and c["n_a1_fitted"] == 3
    assert c["rate"] == pytest.approx(1 / 3)
    # Toutatis is the measured floor and it must come out at |R| ~ 5.
    tou = table[table["full_name"].str.contains("Toutatis")].iloc[0]
    assert tou["abs_R"] == pytest.approx(5.0, rel=0.02)
    assert summ["floor_abs_R_max"] >= 5.0
    assert summ["flag_threshold_above_floor"] is True or p.r_flag > 5.0


def test_negative_a1_census_records_an_empty_population_as_such():
    summ = census.negative_a1_census({"comet": pd.DataFrame()}, _params())[1]
    assert summ["populations"]["comet"]["n_negative"] == 0
    assert summ["populations"]["comet"]["rate"] is None


# =============================================================================
# 10. The independent, catalogue-wide albedo screen
# =============================================================================
def test_high_albedo_census_is_independent_of_A1_and_filters_client_side():
    """Screen 4 asks a question that has nothing to do with whether an orbit fit
    included a non-gravitational term -- so it must not be restricted to the A1
    sample, and it must not trust the server's comparison operator either."""
    payload = _payload(
        ["spkid", "full_name", "pdes", "albedo", "condition_code", "data_arc",
         "n_obs_used", "H"],
        [["1", "(bright)", "2020 AA", "0.85", "0", "3000", "500", "20.0"],
         ["2", "(brighter)", "2020 BB", "0.95", "7", "12", "18", "27.0"],
         # The server returned this despite the constraint: filter it out here.
         ["3", "(ordinary)", "2020 CC", "0.20", "0", "3000", "500", "18.0"]])
    res = census.high_albedo_census(
        params=_params(), transport=_fake_transport({"": payload}), tries=1,
        crosscheck=False)

    assert res.status == acquire.STATUS_OK
    assert res.n_rows_returned == 3
    assert res.n_above_cut == 2, "the 0.20 row must be dropped client-side"
    assert list(res.table["albedo"]) == [0.95, 0.85]
    # Orbit quality travels with every row: short-arc albedo fits are the
    # expected dominant contaminant and must be visible, not assumed away.
    assert {"condition_code", "data_arc", "n_obs_used"} <= set(res.table.columns)
    assert res.albedo_sigma_available is False
    assert any("UNTESTED" in n for n in res.notes)


def test_high_albedo_census_requires_a_significant_excess_when_sigma_exists():
    payload = _payload(
        ["full_name", "pdes", "albedo", "albedo_sigma"],
        [["(marginal)", "2020 AA", "0.72", "0.20"],     # 0.1 sigma above the cut
         ["(solid)", "2020 BB", "0.90", "0.02"]])       # 10 sigma above
    res = census.high_albedo_census(
        params=_params(), transport=_fake_transport({"": payload}), tries=1,
        crosscheck=False)
    assert res.albedo_sigma_available is True
    assert res.n_above_cut == 2 and res.n_significant == 1
    sig = res.table.set_index("pdes")["albedo_excess_significant"]
    assert bool(sig["2020 BB"]) and not bool(sig["2020 AA"])


def test_high_albedo_census_degrades_honestly_when_every_strategy_fails():
    res = census.high_albedo_census(
        params=_params(), transport=_fake_transport({}, OSError("blocked")),
        tries=1, crosscheck=False)
    assert res.status == acquire.STATUS_QUERY_FAILED
    assert res.n_above_cut == 0
    assert any("UNTESTED" in n for n in res.notes)


def test_high_albedo_zero_rows_is_a_measurement_not_a_failure():
    """REGRESSION. If every strategy answers with zero rows, the screen must NOT
    be relabelled QUERY_FAILED -- that is the exact conflation this channel
    exists to avoid."""
    res = census.high_albedo_census(
        params=_params(),
        transport=_fake_transport({"": _payload(["full_name", "albedo"], [])}),
        tries=1, crosscheck=False)
    assert res.status == acquire.STATUS_ZERO_ROWS
    assert not any("failed" in n for n in res.notes)
    assert any("measurement, not a failure" in n for n in res.notes)


def test_high_albedo_query_that_works_but_finds_nothing_bright_stays_OK():
    """The query succeeded; the sky is what is empty. `n_above_cut` carries the
    screen result, and the query status must not be dragged down with it."""
    payload = _payload(["full_name", "pdes", "albedo"],
                       [["(dull)", "2020 CC", "0.20"]])
    res = census.high_albedo_census(
        params=_params(), transport=_fake_transport({"": payload}), tries=1,
        crosscheck=False)
    assert res.status == acquire.STATUS_OK
    assert res.n_rows_returned == 1 and res.n_above_cut == 0
    assert any("the sky is what is empty" in n for n in res.notes)


def test_irsa_crosscheck_confirms_a_second_independent_albedo():
    """A single-source albedo above 0.7 is a fit artefact until something else
    agrees. The table and column names are DISCOVERED, because a renamed table
    looks exactly like an unreachable archive."""
    def _searcher(query: str) -> pd.DataFrame:
        if "TAP_SCHEMA.tables" in query:
            return pd.DataFrame({"table_name": ["neowise_diam_alb"]})
        if "TAP_SCHEMA.columns" in query:
            return pd.DataFrame({"column_name": ["pdes", "albedo", "albedo_err"]})
        return pd.DataFrame({"pdes": ["2020 AA"], "albedo": [0.88],
                             "albedo_err": [0.03]})

    payload = _payload(["full_name", "pdes", "albedo"],
                       [["(a)", "2020 AA", "0.85"], ["(b)", "2020 BB", "0.92"]])
    res = census.high_albedo_census(
        params=_params(), transport=_fake_transport({"": payload}), tries=1,
        searcher=_searcher)

    assert res.crosscheck["status"] == census.XCHECK_OK
    assert res.crosscheck["table"] == "neowise_diam_alb"
    t = res.table.set_index("pdes")
    assert t.loc["2020 AA", "neowise_albedo"] == pytest.approx(0.88)
    assert bool(t.loc["2020 AA", "albedo_confirmed_two_sources"])
    assert not bool(t.loc["2020 BB", "albedo_confirmed_two_sources"]), (
        "no independent measurement means UNCONFIRMED, not confirmed")
    assert res.crosscheck["n_confirmed_above_cut"] == 1


def test_irsa_crosscheck_matches_across_designation_renderings():
    """IRSA renders '2005 VL1' as '2005VL1'. An unmatched row is
    indistinguishable from a body with no independent albedo, so a whitespace
    convention must not be able to turn a real confirmation into a silent
    'nothing agreed'."""
    def _searcher(query: str) -> pd.DataFrame:
        if "TAP_SCHEMA.tables" in query:
            return pd.DataFrame({"table_name": ["neowise_diam_alb"]})
        if "TAP_SCHEMA.columns" in query:
            return pd.DataFrame({"column_name": ["designation", "pv"]})
        return pd.DataFrame({"designation": ["2005VL1"], "pv": [0.91]})

    payload = _payload(["full_name", "pdes", "albedo"],
                       [["(2005 VL1)", "2005 VL1", "0.85"]])
    res = census.high_albedo_census(
        params=_params(), transport=_fake_transport({"": payload}), tries=1,
        searcher=_searcher)
    assert res.crosscheck["status"] == census.XCHECK_OK
    assert res.table.iloc[0]["neowise_albedo"] == pytest.approx(0.91)
    assert bool(res.table.iloc[0]["albedo_confirmed_two_sources"])


def test_irsa_crosscheck_unreachable_is_untested_never_confirmed():
    def _searcher(query: str) -> pd.DataFrame:
        raise OSError("CONNECT tunnel failed, response 403")

    payload = _payload(["full_name", "pdes", "albedo"], [["(a)", "2020 AA", "0.85"]])
    res = census.high_albedo_census(
        params=_params(), transport=_fake_transport({"": payload}), tries=1,
        searcher=_searcher)
    assert res.crosscheck["status"] == census.XCHECK_UNREACHED
    assert not bool(res.table.iloc[0]["albedo_confirmed_two_sources"])
    assert res.table.iloc[0]["albedo_crosscheck"] == census.XCHECK_UNREACHED


def test_full_run_wires_every_census_stage_together(tmp_path):
    """End to end through derelict_run with a synthetic SBDB: the completeness
    probe, the catalogue-wide albedo screen, the negative-A1 rate and the query
    log must all land on disk from one invocation."""
    from seti.config import load_config

    constrained = [["3752158", "(2016 NJ33)", "2016 NJ33", "au", "AMO",
                    "9.475402e-10", "-5.486101e-13", "8.49e-11"]]
    detail = json.dumps({
        "object": {"fullname": "(2016 NJ33)", "des": "2016 NJ33",
                   "spkid": "3752158", "kind": "au",
                   "orbit_class": {"code": "AMO"}},
        "orbit": {"data_arc": "2261", "condition_code": "3", "n_obs_used": "90",
                  "model_pars": [
                      {"name": "A1", "value": "9.475402e-10", "sigma": "2.928e-10"},
                      {"name": "A2", "value": "-5.486101e-13", "sigma": "1.909e-13"},
                      {"name": "A3", "value": "8.49e-11", "sigma": "1.63e-11"}]},
        "phys_par": [{"name": "H", "value": "25.53"}]}).encode()

    def _t(url: str, timeout: float = 0.0) -> bytes:
        if "sbdb.api" in url:
            return detail
        if "albedo" in url:
            return _payload(["spkid", "full_name", "pdes", "albedo",
                             "condition_code", "data_arc", "n_obs_used"],
                            [["9", "(shiny)", "2020 AA", "0.88", "0", "3000", "500"]])
        cols = ["spkid", "full_name", "pdes", "kind", "class", "A1", "A2", "A3"]
        if "sb-cdata" in url:
            return _payload(cols, constrained, count=1)
        return _payload(cols, constrained
                        + [["2000001", "1 Ceres", "1", "au", "MBA", None, None, None]],
                        count=2)

    cfg = load_config()
    cfg.root = tmp_path
    summary = derelict_run(cfg, transport=_t, tries=1, skip_control=True,
                           searcher=lambda q: (_ for _ in ()).throw(OSError("no irsa")))

    out = tmp_path / "results" / "derelict"
    for name in ("completeness.json", "queries.json", "high_albedo.csv",
                 "negative_a1.csv", "screened.csv", "summary.json"):
        assert (out / name).exists(), name

    # The constraint returned every object that has an A1: complete.
    comp = json.loads((out / "completeness.json").read_text())
    assert comp["verdict"] == census.CONSTRAINT_COMPLETE
    assert comp["asteroid"]["n_unconstrained_nonnull_A1"] == 1
    assert comp["asteroid"]["queries"], "completeness.json is self-contained"

    # Screen 4 ran catalogue-wide and found the bright object, unconfirmed.
    assert summary["funnel"]["screen4_albedo_catalogue_wide"] == 1
    assert summary["funnel"]["screen4_albedo_confirmed_two_sources"] == 0

    # Screen 3 is reported with its denominator even when the count is zero.
    assert summary["funnel"]["screen3_denominator_asteroid"] == 1
    assert summary["negative_a1_census"]["populations"]["asteroid"]["rate"] == 0.0

    assert summary["query_status_counts"], "the query log reached the summary"
    assert acquire.STATUS_QUERY_FAILED not in summary["query_status_counts"]


def test_a_standalone_stage_never_clobbers_the_main_summary(tmp_path):
    """A partial run must not be able to replace a full one and make the
    channel look like it regressed."""
    from seti.config import load_config

    cfg = load_config()
    cfg.root = tmp_path
    out = tmp_path / "results" / "derelict"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text('{"verdict": "A_PREVIOUS_FULL_RUN"}')

    summary = derelict_run(cfg, stage="dark_comets", tries=1,
                           transport=_fake_transport({}, OSError("blocked")))
    assert summary["verdict"] == "DARK_COMET_CENSUS_ONLY"
    assert (out / "summary_dark_comets.json").exists()
    assert json.loads((out / "summary.json").read_text())["verdict"] == \
        "A_PREVIOUS_FULL_RUN"


def test_screen_params_from_config_reads_the_yaml():
    import yaml as _yaml

    from seti.config import load_config
    cfg = load_config()
    d = _yaml.safe_load((cfg.root / "config" / "derelict.yaml").read_text())
    p = ScreenParams.from_config(d)
    assert p.a1_snr_min == 3.0 and p.a2_snr_max == 1.0 and p.a3_snr_max == 1.0
    assert p.q_pr == 1.0 and p.rho_natural_kg_m3 == 1000.0
    vp = VetParams.from_config(d)
    assert vp.space_age_year == 1957

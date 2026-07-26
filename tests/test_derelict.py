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

from seti.derelict import acquire, radiation, screen, vet
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
    nongrav_law_is_inverse_square,
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


def test_r_statistic_refuses_a_cometary_law():
    st = r_statistic(1e-7, diameter_m=100.0, nongrav_law_is_inverse_square=False)
    assert not st.valid and "cometary" in st.reason


def test_nongrav_law_detection():
    assert nongrav_law_is_inverse_square(["A1", "A2"])
    assert nongrav_law_is_inverse_square(None)
    assert not nongrav_law_is_inverse_square(["A1", "DT"])
    assert not nongrav_law_is_inverse_square(["A1", "aln", "nm", "nn"])


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


def test_cometary_nongrav_law_blocks_the_conversion():
    df = pd.DataFrame([_row(A1=2e-8, sigma_A1=2e-9)])
    df["model_pars"] = [["A1", "A2", "ALN", "NM"]]
    res = run_screens(df, _params())
    r = res.table.iloc[0]
    assert not r["nongrav_inverse_square"]
    assert not r["screen_a1_only"]
    assert not r["R_valid"]
    assert res.funnel["cometary_law_excluded"] == 1


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


def test_cometary_model_pars_in_detail_gives_outgassing():
    detail = {"ok": True, "orbit": {"model_pars": [
        {"name": "A1", "value": "1e-8"}, {"name": "ALN", "value": "0.1"}]}}
    row = pd.Series(_row(full_name="(y)", A1=1e-6, sigma_A1=1e-8, a=3.0, e=0.5, i=20.0,
                         R=1e5))
    assert vet_object(row, detail, VetParams()).verdict == vet.OUTGASSING


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
                           skip_control=True)
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["funnel"]["input"] == 0
    # The funnel must let a reader tell a query failure from an empty sky.
    assert summary["funnel"]["rows_returned_by_server"] == 0
    assert summary["funnel"]["a1_column_present"] is False
    assert any("QUERY DEFECT" in d for d in summary["degradation"])
    out = json.loads((tmp_path / "results" / "derelict" / "summary.json").read_text())
    assert out["verdict"] == "NO_DATA_REACHED"
    assert json.loads((tmp_path / "results" / "derelict" / "candidates.json").read_text()) == []


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

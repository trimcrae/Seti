"""Offline tests for the SHROUD channel (CI gate; no network).

Every synthetic object below is built from the channel's own physics rather
than from hand-chosen magnitudes: the enshrouded star's infrared photometry is
computed by re-radiating exactly the bolometric flux implied by its POSS-I
magnitude, so "the infrared accounts for the missing optical" is true by
construction and the detector has to find it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from seti.config import load_config
from seti.shroud import acquire as acq
from seti.shroud import classify as cls
from seti.shroud import run as runmod
from seti.shroud import sed as S
from seti.shroud import vet as V

SIGMA_SB = 5.670374419e-8


@pytest.fixture(scope="module")
def sc():
    cfg = load_config()
    with (cfg.root / "config" / "shroud.yaml").open() as fh:
        return yaml.safe_load(fh)


# --- synthetic object factories --------------------------------------------
def _enshrouded_row(plate_mag=17.0, t_dust=350.0, ir_scale_factor=1.0,
                    source_id="ENSHROUDED", teff_grid=None):
    """A star that vanished optically and re-radiates its light as warm dust.

    ``ir_scale_factor`` < 1 makes the infrared too faint to pay the optical
    debt, which is the second required test case.
    """
    teff_grid = teff_grid or [2500, 2800, 3200, 3600, 4000, 4500, 5000, 5800,
                              6500, 7500, 9000, 11000, 15000, 20000]
    bc_min = min(S.bolometric_correction_factor(float(t), "poss1_e")
                 for t in teff_grid)
    f_bol_then = S.mag_to_fnu("poss1_e", plate_mag) * bc_min
    scale = ir_scale_factor * f_bol_then * math.pi / (SIGMA_SB * t_dust ** 4)
    row = {"source_id": source_id, "ra_deg": 190.0, "dec_deg": 42.0,
           "poss1_e": plate_mag, "sample": "solano2022_ir_present",
           "n_ir_neighbours": 1, "ir_local_density_per_deg2": 2000.0}
    for b in ("w1", "w2", "w3", "w4"):
        row[b] = S.fnu_to_mag(b, scale * S.planck_fnu(t_dust, S.BANDS[b][0]))
        row[f"{b}_err"] = 0.03
    row["2mass_ks_lim"] = 15.3          # a real 2MASS non-detection
    row["ps1_r_lim"] = 23.2             # the modern optical upper limit
    return row


def _plate_defect_row(source_id="DEFECT"):
    """Emulsion artefact: on the plate, absent from every catalogue ever since."""
    return {"source_id": source_id, "ra_deg": 12.0, "dec_deg": 70.0,
            "poss1_e": 19.8, "sample": "solano2022_no_counterpart",
            "n_ir_neighbours": 0, "ir_local_density_per_deg2": 1500.0}


def _high_pm_case(cfg_epochs):
    """A star that simply moved off its 1953 position."""
    ra0, dec0 = 200.0, 10.0
    pmra, pmdec = 200.0, -150.0                      # mas/yr
    dt = cfg_epochs["gaia_dr3"] - cfg_epochs["poss1_default"]
    ra_now, dec_now = V.propagate_position(ra0, dec0, pmra, pmdec, dt)
    neigh = pd.DataFrame([{"ra_deg": float(ra_now), "dec_deg": float(dec_now),
                           "pmra": pmra, "pmdec": pmdec}])
    return ra0, dec0, neigh


# ===========================================================================
# 1. REQUIRED: the enshrouded star must be flagged energy-conserving.
# ===========================================================================
def test_enshrouded_star_is_energy_conserving(sc):
    row = _enshrouded_row()
    sed = V.build_sed(row)
    assert sed.detected_modern() == ["w1", "w2", "w3", "w4"]
    assert "poss1_e" not in sed.detected_modern(), \
        "the 1953 plate point must never enter the present-day SED fit"

    _, fit_dust = S.fit_both(sed, sc)
    assert fit_dust.ok
    assert fit_dust.t_dust_k == pytest.approx(350.0, abs=1e-6), \
        "the warm-dust temperature must be recovered"

    b = S.energy_budget(sed, sc, fit_dust)
    assert b.verdict == "ENERGY_CONSERVING_OBSCURATION", b.note
    assert b.eta_max == pytest.approx(1.0, rel=0.15), b.eta_max
    assert b.eta_lo < 1.0 and b.eta_hi == pytest.approx(1.0, rel=0.15)
    # A shroud is a curved blackbody, NOT the power law of an AGN.
    pl, info = S.ir_shape_prefers_powerlaw(sed, sc["sed"]["tdust_grid_k"])
    assert pl is False
    assert info["t_ir_blackbody_k"] == pytest.approx(350.0)

    ftk, _ = V.ftk_verdict(b.eta_max, sc)
    assert ftk == "OBSCURATION_LIKE"


def test_enshrouded_star_survives_the_full_funnel(sc):
    df = pd.DataFrame([_enshrouded_row()])
    df = runmod.stage_classify(df, sc)
    assert df.loc[0, "class"] == "RESIDUAL_UNEXPLAINED", df.loc[0, "class_reason"]
    df, budgets, fits = runmod.stage_budget(df, sc)
    df = V.vet_table(df, sc, budgets, fits)
    assert bool(df.loc[0, "survives"]), df.loc[0, "vet_flags"]
    assert df.loc[0, "budget_verdict"] == "ENERGY_CONSERVING_OBSCURATION"


# ===========================================================================
# 2. REQUIRED: infrared far too faint -> NOT simple obscuration.
# ===========================================================================
def test_ir_too_faint_is_not_obscuration(sc):
    row = _enshrouded_row(ir_scale_factor=0.01, source_id="TOO_FAINT")
    sed = V.build_sed(row)
    b = S.energy_budget(sed, sc, S.fit_both(sed, sc)[1])
    assert b.verdict == "IR_TOO_FAINT", (b.verdict, b.note)
    assert b.eta_max < sc["energy_budget"]["eta_too_faint"]
    # The claim must hold for EVERY allowed progenitor temperature.
    assert b.eta_hi < 1.0, "eta_max is the most generous value and must be < 1"


def test_ir_far_too_bright_is_flagged_and_killed_by_ftk(sc):
    """An IR-luminous source: the F-T&K merger-remnant discriminant must fire."""
    row = _enshrouded_row(ir_scale_factor=50.0, source_id="TOO_BRIGHT")
    sed = V.build_sed(row)
    b = S.energy_budget(sed, sc, S.fit_both(sed, sc)[1])
    assert b.verdict == "IR_EXCEEDS_MISSING", (b.verdict, b.note)
    ftk, reason = V.ftk_verdict(b.eta_max, sc)
    assert ftk == "MERGER_REMNANT_LIKE", reason
    r = V.vet_object({**row, "class": "RESIDUAL_UNEXPLAINED"}, sc, b)
    assert not r.survives
    assert "FTK_MERGER_REMNANT" in r.flags


def test_two_band_ir_cannot_support_a_deficit_verdict(sc):
    """W1/W2 alone under-integrate a thermal SED; no verdict may be issued.

    This is the normal case for the published ``vanish-neowise`` table, so the
    guard is what stops the channel manufacturing its headline result out of
    missing photometry.
    """
    full = _enshrouded_row(source_id="FULL")
    two = {k: v for k, v in full.items() if k not in ("w3", "w4", "w3_err", "w4_err")}
    b = S.energy_budget(V.build_sed(two), sc, None)
    assert b.verdict == "IR_UNDERSAMPLED", (b.verdict, b.note)
    assert b.n_ir_bands == 2
    assert "AllWISE" in b.note
    # With W3/W4 restored the same object is correctly energy-conserving.
    assert S.energy_budget(V.build_sed(full), sc,
                           S.fit_both(V.build_sed(full), sc)[1]
                           ).verdict == "ENERGY_CONSERVING_OBSCURATION"


def test_ftk_disappearance_regime(sc):
    """~10x dimmer than the progenitor is the genuine-disappearance signature."""
    ftk, _ = V.ftk_verdict(0.1, sc)
    assert ftk == "DISAPPEARANCE_LIKE"
    assert V.ftk_verdict(float("nan"), sc)[0] == "UNDETERMINED"


# ===========================================================================
# 3. REQUIRED: high proper motion must be rejected by epoch propagation.
# ===========================================================================
def test_high_pm_star_rejected_by_epoch_propagation(sc):
    ra0, dec0, neigh = _high_pm_case(sc["epochs"])
    res = V.epoch_propagation_check(ra0, dec0, neigh, sc)
    assert res["pm_recovered"] is True
    assert res["pm_back_propagated_sep_arcsec"] < 0.5
    assert res["pm_displacement_arcsec"] > 10.0
    assert res["pm_total_mas_yr"] == pytest.approx(250.0, rel=1e-3)

    row = {"source_id": "PM", "ra_deg": ra0, "dec_deg": dec0, "poss1_e": 15.0,
           "w1": 12.0, "w2": 11.9, "n_ir_neighbours": 1, **res}
    klass, why = cls.classify_source(row, sc)
    assert klass == "HIGH_PM_STAR", why
    assert not V.vet_object({**row, "class": klass}, sc).survives


def test_a_stationary_star_is_not_called_high_pm(sc):
    """A source that never moved cannot be explained by proper motion."""
    neigh = pd.DataFrame([{"ra_deg": 200.0, "dec_deg": 10.0,
                           "pmra": 1.0, "pmdec": -1.0}])
    res = V.epoch_propagation_check(200.0, 10.0, neigh, sc)
    assert res["pm_recovered"] is False
    assert V.epoch_propagation_check(200.0, 10.0, pd.DataFrame(), sc)[
        "pm_recovered"] is False


def test_propagation_round_trip():
    ra, dec = V.propagate_position(100.0, -30.0, 500.0, 400.0, 60.0)
    back = V.propagate_position(ra, dec, 500.0, 400.0, -60.0)
    # Not exact: the return leg uses the shifted declination in cos(dec).
    assert float(back[0]) == pytest.approx(100.0, abs=1e-5)
    assert float(back[1]) == pytest.approx(-30.0, abs=1e-12)
    # 500 mas/yr for 60 yr = 30" in RA*, 24" in Dec.
    sep = V.angular_separation_arcsec(100.0, -30.0, ra, dec)
    assert float(sep) == pytest.approx(math.hypot(30.0, 24.0), rel=1e-3)


# ===========================================================================
# 4. REQUIRED: a plate defect with no IR counterpart must be rejected.
# ===========================================================================
def test_plate_defect_rejected(sc):
    row = _plate_defect_row()
    klass, why = cls.classify_source(row, sc)
    assert klass == "PLATE_DEFECT", why
    sed = V.build_sed(row)
    b = S.energy_budget(sed, sc, None)
    assert b.verdict == "INSUFFICIENT_IR"
    r = V.vet_object({**row, "class": klass}, sc, b)
    assert not r.survives
    assert "SINGLE_IR_BAND" in r.flags
    assert "CLASS_PLATE_DEFECT" in r.flags


def test_plate_defect_near_the_plate_limit_is_flagged(sc):
    """Hambly & Blair 2024: emulsion noise lives within a magnitude of the limit."""
    row = {**_plate_defect_row(), "poss1_e": 19.9, "w1": 16.0, "w2": 15.7}
    assert "PLATE_LIMIT_PROXIMITY" in V.ledger_vetoes(row, sc)
    row_bright = {**row, "poss1_e": 16.0}
    assert "PLATE_LIMIT_PROXIMITY" not in V.ledger_vetoes(row_bright, sc)
    assert "PLATE_SATURATED" in V.ledger_vetoes({**row, "poss1_e": 10.0}, sc)


# ===========================================================================
# 5. REQUIRED: honest degradation when the VO archive is unreachable.
# ===========================================================================
def test_degrades_honestly_without_network(sc, tmp_path):
    df, prov = acq.acquire_sample(sc, tmp_path, allow_network=False)
    assert len(df) == 0
    assert prov["verdict"] == "NO_DATA_REACHED"
    assert "no rows were invented" in prov["note"]


def test_run_reports_no_data_reached(tmp_path, monkeypatch):
    cfg = load_config()
    sc_local = runmod.load_shroud_config(cfg)
    out = tmp_path / "shroud"
    out.mkdir()
    summary = runmod.stage_report(
        cfg, sc_local, pd.DataFrame(),
        {"verdict": "NO_DATA_REACHED", "routes": [{"route": "probe_svo:x",
                                                   "status": "unreachable",
                                                   "attempts": []}],
         "note": "no archive route returned rows; nothing was fabricated"},
        out)
    assert summary["verdict"] == "NO_DATA_REACHED"
    assert summary["n_sample"] == 0
    assert summary["degraded"] is True
    body = (out / "REPORT.md").read_text()
    assert "nothing was analysed and nothing was" in body
    assert json.loads((out / "summary.json").read_text())["n_sample"] == 0


def test_local_cache_route(sc, tmp_path):
    """A cached sample is used and labelled as such, never silently as fresh."""
    pd.DataFrame([{"source_id": "X", "ra_deg": 1.0, "dec_deg": 2.0}]).to_parquet(
        tmp_path / "sample_positions.parquet", index=False)
    df, prov = acq.acquire_sample(sc, tmp_path, allow_network=False)
    assert prov["verdict"] == "LOCAL_CACHE"
    assert len(df) == 1


# ===========================================================================
# Mundane-population subtraction: one case per class.
# ===========================================================================
def _base(**kw):
    row = {"source_id": "T", "ra_deg": 190.0, "dec_deg": 42.0, "poss1_e": 18.0,
           "n_ir_neighbours": 1}
    row.update(kw)
    return row


def test_asteroid_class(sc):
    # Near the ecliptic, nothing anywhere else.
    ra, dec = 60.0, 20.0                      # |ecliptic lat| small
    assert abs(float(cls.ecliptic_latitude(ra, dec))) < 20.0
    klass, _ = cls.classify_source(_base(ra_deg=ra, dec_deg=dec), sc)
    assert klass == "ASTEROID"


def test_modern_optical_match_and_variable_star(sc):
    same, _ = cls.classify_source(_base(ps1_r=18.1, w1=14.0, w2=13.8), sc)
    assert same == "MODERN_OPTICAL_MATCH"
    var, _ = cls.classify_source(_base(ps1_r=21.5, w1=14.0, w2=13.8), sc)
    assert var == "VARIABLE_STAR"


def _powerlaw_agn_row(beta=-1.0, amp=1e-15, **kw):
    """A genuine AGN: F_nu is a power law across W1-W4."""
    row = _base(**kw)
    for b in ("w1", "w2", "w3", "w4"):
        row[b] = S.fnu_to_mag(b, amp * S.nu_hz(b) ** beta)
        row[f"{b}_err"] = 0.03
    return row


def test_agn_needs_a_power_law_not_just_a_red_colour(sc):
    """The Stern colour cut alone would delete the shroud population."""
    agn = _powerlaw_agn_row()
    assert agn["w1"] - agn["w2"] >= sc["classify"]["agn_w1w2_min"]
    klass, why = cls.classify_source(agn, sc)
    assert klass == "AGN_QSO", why

    # A 350 K shroud is redder still, yet must NOT be called an AGN.
    shroud = _enshrouded_row()
    assert shroud["w1"] - shroud["w2"] > 3.0
    assert cls.classify_source(shroud, sc)[0] == "RESIDUAL_UNEXPLAINED"


def test_two_ir_bands_cannot_decide_shape(sc):
    """With W1/W2 only the AGN and shroud hypotheses are formally degenerate."""
    klass, why = cls.classify_source(_base(w1=15.0, w2=13.9), sc)  # W1-W2 = 1.1
    assert klass == "AGN_QSO_COLOUR_ONLY", why
    assert "not separable" in why


def test_galaxy_class(sc):
    klass, _ = cls.classify_source(_base(w1=15.0, w2=14.8, ir_ext_flag=1), sc)
    assert klass == "GALAXY"


def test_dusty_agb_class(sc):
    klass, why = cls.classify_source(
        _base(w1=6.0, w2=5.5, w3=3.0, w4=1.5, **{"2mass_ks": 6.5}), sc)
    assert klass == "DUSTY_AGB", why


def test_yso_class(sc):
    # A sightline in the Galactic plane with a rising mid-IR SED.
    ra, dec = 283.0, 0.0
    assert abs(float(cls.galactic_latitude(ra, dec))) < 5.0
    klass, why = cls.classify_source(
        _base(ra_deg=ra, dec_deg=dec, w1=11.0, w2=10.5, w3=7.0), sc)
    assert klass == "YSO", why


def test_blend_confusion_class(sc):
    klass, _ = cls.classify_source(
        _base(w1=14.0, w2=13.8, n_ir_neighbours=3), sc)
    assert klass == "BLEND_CONFUSION"


def test_population_breakdown_and_ratio(sc):
    rows = [_enshrouded_row(source_id="A"), _plate_defect_row("B"),
            _powerlaw_agn_row(source_id="C", sample="solano2022_ir_present")]
    df = cls.classify_table(pd.DataFrame(rows), sc)
    pop = cls.population_breakdown(df)
    assert set(pop["class"]) == {"RESIDUAL_UNEXPLAINED", "PLATE_DEFECT", "AGN_QSO"}
    assert pop["fraction"].sum() == pytest.approx(1.0)
    r = cls.obscuration_vs_destruction_ratio(df)
    assert r["n_no_counterpart"] == 1
    assert r["n_residual_with_ir"] == 1
    assert r["ratio_after_subtraction"] == pytest.approx(1.0)
    assert cls.population_breakdown(pd.DataFrame()).empty


# ===========================================================================
# Contamination ledger.
# ===========================================================================
def test_w4_only_and_single_band_vetoes(sc):
    assert "W4_ONLY" in V.ledger_vetoes(_base(w4=7.0), sc)
    assert "SINGLE_IR_BAND" in V.ledger_vetoes(_base(w1=14.0), sc)
    assert "SINGLE_IR_BAND" not in V.ledger_vetoes(_base(w1=14.0, w2=13.8), sc)


def test_negative_w1w2_is_a_blend(sc):
    assert "NEGATIVE_W1W2_BLEND" in V.ledger_vetoes(_base(w1=14.0, w2=14.5), sc)
    assert "NEGATIVE_W1W2_BLEND" not in V.ledger_vetoes(_base(w1=14.0, w2=13.9), sc)


def test_ir_confusion_veto(sc):
    assert "IR_CONFUSION" in V.ledger_vetoes(
        _base(w1=14.0, w2=13.8, n_ir_neighbours=4), sc)


def test_unphysically_hot_dust_is_a_companion(sc):
    hot = S.FitResult("obscured_dust", 5000.0, 1.0, 2500.0, 1.0, 1.0, 1.0, 5, 1)
    assert "TDUST_UNPHYSICAL_COMPANION" in V.ledger_vetoes(
        _base(w1=14.0, w2=13.8), sc, hot)
    cool = S.FitResult("obscured_dust", 5000.0, 1.0, 400.0, 1.0, 1.0, 1.0, 5, 1)
    assert "TDUST_UNPHYSICAL_COMPANION" not in V.ledger_vetoes(
        _base(w1=14.0, w2=13.8), sc, cool)


# ===========================================================================
# Chance coincidence — the dominant systematic of this sample.
# ===========================================================================
def test_chance_match_probability_scales_correctly():
    p_lo = V.chance_match_probability(5.0, 1500.0)      # high galactic latitude
    p_hi = V.chance_match_probability(5.0, 45800.0)     # CatWISE all-sky mean
    assert 0.005 < p_lo < 0.02
    assert 0.2 < p_hi < 0.3
    assert V.chance_match_probability(2.0, 45800.0) < p_hi
    assert V.chance_match_probability(5.0, 0.0) == 0.0
    # 172 163 sources at the all-sky mean density: tens of thousands of
    # coincidences are expected, which is why the null is measured.
    assert V.expected_chance_matches(172163, 5.0, 45800.0) > 3e4


def test_chance_match_veto_fires(sc):
    assert "CHANCE_MATCH_LIKELY" in V.ledger_vetoes(
        _base(w1=14.0, w2=13.8, p_chance_match=0.5), sc)
    assert "CHANCE_MATCH_POSSIBLE" in V.ledger_vetoes(
        _base(w1=14.0, w2=13.8, p_chance_match=0.10), sc)
    assert not [f for f in V.ledger_vetoes(
        _base(w1=14.0, w2=13.8, p_chance_match=0.01), sc)
        if f.startswith("CHANCE_MATCH")]


def test_offset_position_null_preserves_separation():
    df = pd.DataFrame({"ra_deg": [10.0, 200.0, 350.0], "dec_deg": [0.0, 60.0, -30.0]})
    off = V.offset_positions(df, 45.0, seed=3)
    sep = V.angular_separation_arcsec(df["ra_deg"], df["dec_deg"],
                                      off["ra_deg"], off["dec_deg"])
    assert np.allclose(np.asarray(sep), 45.0, rtol=2e-3)


def test_chance_rate_from_null_recovers_the_true_fraction():
    # 60% matched for real, 20% matched at offset positions -> 40% genuine.
    st = V.chance_match_rate_from_null(6000, 10000, 2000, 10000)
    assert st["f_true"] == pytest.approx(0.40, abs=1e-9)
    assert st["n_expected_chance"] == pytest.approx(2000.0)
    assert st["significance_sigma"] > 10
    assert V.chance_match_rate_from_null(0, 0, 0, 0)["f_true"] is None


# ===========================================================================
# SED machinery.
# ===========================================================================
def test_magnitude_flux_round_trip():
    for band in S.BANDS:
        f = S.mag_to_fnu(band, 12.34)
        assert S.fnu_to_mag(band, f) == pytest.approx(12.34, abs=1e-12)


def test_bolometric_correction_minimum_is_physical():
    """F_bol/F_nu is minimised where x = h*nu/kT ~ 3.92, not at the Wien peak."""
    grid = np.arange(2000.0, 30000.0, 25.0)
    t_min = S.temperature_minimising_fbol("poss1_e", grid)
    nu = S.nu_hz("poss1_e")
    x = 6.62607015e-34 * nu / (1.380649e-23 * t_min)
    assert x == pytest.approx(3.921, abs=0.02)


def test_extinction_is_chromatic_and_ordered():
    """Reddening must be strongly wavelength dependent (unlike a grey occulter)."""
    a_v = 2.0
    fac = {b: S.extinction_factor(b, a_v) for b in
           ("poss1_o", "poss1_e", "2mass_ks", "w1", "w2")}
    assert fac["poss1_o"] < fac["poss1_e"] < fac["2mass_ks"] < fac["w2"]
    assert fac["w2"] > 0.95 and fac["poss1_o"] < 0.15


def test_photosphere_fit_recovers_an_injected_star(sc):
    """A reddened photosphere with no dust must be fitted by model (a)."""
    teff, a_v, scale = 4500.0, 1.0, 3.0e-19
    bands = ["2mass_j", "2mass_h", "2mass_ks", "w1", "w2"]
    f = S.photosphere_fnu(bands, teff, scale, a_v)
    sed = S.SED("STAR", mags={b: S.fnu_to_mag(b, x) for b, x in zip(bands, f, strict=True)},
                errs=dict.fromkeys(bands, 0.02))
    fit = S.fit_photosphere(sed, sc["sed"]["teff_star_grid_k"],
                            sc["sed"]["av_grid_mag"])
    assert fit.ok
    assert fit.teff_k == pytest.approx(teff, abs=1e-6)
    assert fit.a_v_mag == pytest.approx(a_v, abs=1e-6)
    assert fit.chi2_red < 1e-6


def test_upper_limits_penalise_overbright_models(sc):
    """A non-detection is information: models brighter than the limit are punished."""
    bands = ["w1", "w2", "w3"]
    sed = S.SED("L", mags={"w1": 14.0, "w2": 13.6, "w3": 12.0},
                errs=dict.fromkeys(bands, 0.05))
    free = S.fit_photosphere(sed, [1000, 2000, 3000], [0.0])
    sed_lim = S.SED("L", mags=dict(sed.mags), errs=dict(sed.errs),
                    limits={"2mass_ks": 15.3})
    with_lim = S.fit_photosphere(sed_lim, [1000, 2000, 3000], [0.0])
    assert with_lim.chi2 >= free.chi2


def test_budget_requires_historical_photometry(sc):
    sed = S.SED("N", mags={"w1": 14.0, "w2": 13.5, "w3": 11.0})
    assert S.energy_budget(sed, sc).verdict == "NO_HISTORICAL_PHOTOMETRY"


def test_budget_reports_no_deficit_when_nothing_faded(sc):
    """A source as bright now as on the plate has no missing luminosity."""
    row = _enshrouded_row()
    row["ps1_r"] = 12.0                      # far brighter than the plate
    sed = V.build_sed(row)
    b = S.energy_budget(sed, sc, None)
    assert b.verdict == "NO_DEFICIT", b.note


def test_trapz_integral_is_a_lower_bound(sc):
    """The model-free IR integral must never exceed the fitted blackbody flux."""
    row = _enshrouded_row()
    sed = V.build_sed(row)
    fit = S.fit_both(sed, sc)[1]
    f_model = S.integrate_blackbody_flux(fit.t_dust_k, fit.scale_dust)
    f_trapz, n = S.integrate_ir_trapz(sed)
    assert n == 4
    assert 0 < f_trapz < f_model


def test_luminosity_ratio_is_eta(sc):
    row = _enshrouded_row()
    sed = V.build_sed(row)
    b = S.energy_budget(sed, sc, S.fit_both(sed, sc)[1])
    assert S.luminosity_ratio(b) == b.eta_max


# ===========================================================================
# Coordinates and archive parsing.
# ===========================================================================
def test_coordinate_transforms():
    # Galactic centre.
    assert float(cls.galactic_latitude(266.405, -28.936)) == pytest.approx(0.0, abs=0.05)
    assert float(cls.galactic_longitude(266.405, -28.936)) == pytest.approx(0.0, abs=0.05)
    # North Galactic Pole.
    assert float(cls.galactic_latitude(192.85948, 27.12825)) == pytest.approx(90.0, abs=1e-4)
    # Vernal equinox lies on the ecliptic.
    assert float(cls.ecliptic_latitude(0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    # North ecliptic pole.
    assert float(cls.ecliptic_latitude(270.0, 66.5607)) == pytest.approx(90.0, abs=0.01)


def test_sexagesimal_parsing():
    ra, dec = acq.sexagesimal_to_deg("00 11 19.43", "-03 09 45.22")
    assert ra == pytest.approx(2.83096, abs=1e-4)
    assert dec == pytest.approx(-3.16256, abs=1e-4)
    ra2, dec2 = acq.sexagesimal_to_deg("12 00 00.0", "+45 30 00.0")
    assert ra2 == pytest.approx(180.0) and dec2 == pytest.approx(45.5)


def test_parse_vizier_vasco2020_offline():
    """Parse the real committed VizieR VOTable for J/AJ/159/8 (99 + 28 rows)."""
    cfg = load_config()
    p = cfg.root / "results" / "disaplit2" / "vizier_vasco_2020.xml"
    if not p.exists():
        pytest.skip("committed VizieR VOTable not present")
    df, prov = acq.parse_vizier_vasco2020(p.read_bytes())
    assert prov.status == "ok"
    assert len(df) == 127, len(df)
    assert set(df["vizier_table"]) == {"table2", "table3"}
    assert int((df["vizier_table"] == "table2").sum()) == 99
    assert df["ra_deg"].between(0, 360).all()
    assert df["dec_deg"].between(-90, 90).all()
    # table2 carries r magnitudes in the published 13.57-19.49 range.
    r = df.loc[df["vizier_table"] == "table2", "poss1_e"].dropna()
    assert len(r) == 99
    assert 13.5 <= r.min() and r.max() <= 19.5


def test_normalise_vo_frame_handles_neowise_columns():
    raw = pd.DataFrame({"RA_NEOWISE": [10.0, 20.0], "DEC_NEOWISE": [1.0, 2.0],
                        "w1mpro": [15.0, 16.0], "w2mpro": [14.5, 15.5],
                        "ph_qual": ["AB", "AU"]})
    out = acq.normalise_vo_frame(raw, sample="solano2022_ir_present")
    assert list(out["ra_deg"]) == [10.0, 20.0]
    assert list(out["w1"]) == [15.0, 16.0]
    assert out["source_id"].iloc[0].startswith("VASCO-IR-")
    assert (out["sample"] == "solano2022_ir_present").all()


def test_votable_multi_table_split():
    body = (b'<VOTABLE><TABLE name="t/a"><FIELD name="x"/><DATA><TABLEDATA>'
            b"<TR><TD>1</TD></TR></TABLEDATA></DATA></TABLE>"
            b'<TABLE name="t/b"><FIELD name="y"/><DATA><TABLEDATA>'
            b"<TR><TD>2</TD></TR><TR><TD>3</TD></TR>"
            b"</TABLEDATA></DATA></TABLE></VOTABLE>")
    tabs = acq.votable_tables(body)
    assert set(tabs) == {"t/a", "t/b"}
    assert len(tabs["t/b"]) == 2


def test_expected_row_counts_are_the_published_ones(sc):
    """Guard the published sizes so a truncated fetch cannot pass silently."""
    exp = sc["acquire"]["expected_rows"]
    assert exp["vanish_neowise"] == 171753   # live archive (Watters et al. 2026)
    assert exp["vanish_possi"] == 5399       # Solano et al. 2022
    prov = acq.Provenance(route="t")
    acq._check_expected_rows("vanish_possi", 5399, sc, prov)
    assert any("within" in n for n in prov.notes)
    prov2 = acq.Provenance(route="t")
    acq._check_expected_rows("vanish_possi", 100, sc, prov2)
    assert any("WARNING" in n for n in prov2.notes)


def test_config_endpoints_are_the_verified_hosts(sc):
    roots = sc["acquire"]["svo_catalogs"]
    assert any("svocats.cab.inta-csic.es/vanish-neowise" in r
               for r in roots["vanish_neowise"])
    assert any("svocats.cab.inta-csic.es/vanish-possi" in r
               for r in roots["vanish_possi"])


# ===========================================================================
# End-to-end on a mixed synthetic population.
# ===========================================================================
def test_end_to_end_mixed_population(sc, tmp_path):
    ra0, dec0, neigh = _high_pm_case(sc["epochs"])
    pm = V.epoch_propagation_check(ra0, dec0, neigh, sc)
    rows = [
        _enshrouded_row(source_id="SHROUD_A"),
        _enshrouded_row(source_id="TOO_FAINT_B", ir_scale_factor=0.01),
        _plate_defect_row("DEFECT_C"),
        {"source_id": "PM_D", "ra_deg": ra0, "dec_deg": dec0, "poss1_e": 16.0,
         "w1": 12.0, "w2": 11.9, "n_ir_neighbours": 1,
         "sample": "solano2022_ir_present", **pm},
        _powerlaw_agn_row(source_id="AGN_E", ra_deg=190.0, dec_deg=40.0,
                          poss1_e=18.5, sample="solano2022_ir_present"),
    ]
    df = pd.DataFrame(rows)
    df = runmod.stage_classify(df, sc)
    df, budgets, fits = runmod.stage_budget(df, sc)
    df = V.vet_table(df, sc, budgets, fits)

    got = dict(zip(df["source_id"], df["class"], strict=True))
    assert got["SHROUD_A"] == "RESIDUAL_UNEXPLAINED"
    assert got["DEFECT_C"] == "PLATE_DEFECT"
    assert got["PM_D"] == "HIGH_PM_STAR"
    assert got["AGN_E"] == "AGN_QSO"

    # "survives" means it beat every CONTAMINATION kill-test.  The energy
    # budget then separates the survivors into two physically different
    # findings; an IR-too-faint object is not contamination, it is the more
    # extreme claim (the light did not simply get reprocessed).
    surv = df[df["survives"]]
    assert list(surv["source_id"]) == ["SHROUD_A", "TOO_FAINT_B"], \
        list(surv["source_id"])
    by_id = dict(zip(df["source_id"], df["budget_verdict"], strict=True))
    assert by_id["SHROUD_A"] == "ENERGY_CONSERVING_OBSCURATION"
    assert by_id["TOO_FAINT_B"] == "IR_TOO_FAINT"

    cfg = load_config()
    summary = runmod.stage_report(cfg, sc, df, {"verdict": "VO_ARCHIVE"},
                                  tmp_path, null_stats={"f_match": 0.6,
                                                        "f_chance": 0.2,
                                                        "f_true": 0.4})
    assert summary["n_sample"] == 5
    assert summary["n_survivors"] == 2
    assert summary["n_energy_conserving"] == 1
    assert summary["n_ir_too_faint"] == 1
    # PM_D has only W1/W2: no verdict is issued from an undersampled SED.
    assert summary["n_ir_undersampled"] == 1
    assert summary["obscuration_vs_destruction"]["n_no_counterpart"] == 1
    md = (tmp_path / "REPORT.md").read_text()
    assert "Population breakdown" in md
    assert "Obscuration vs destruction" in md
    assert "Chance-match null" in md
    assert (tmp_path / "population.csv").exists()
    assert (tmp_path / "survivors.csv").exists()


def test_empty_frames_do_not_crash(sc):
    empty = pd.DataFrame()
    assert cls.classify_table(empty, sc).empty
    assert V.vet_table(empty, sc).empty
    assert cls.obscuration_vs_destruction_ratio(empty)["ratio_raw"] is None


# ===========================================================================
# Photometry join (offline: synthetic X-Match outputs, no network).
# ===========================================================================
def test_join_xmatch_photometry_builds_full_seds(sc):
    """AllWISE W3/W4 + 2MASS must be joined onto the W1/W2-only published table."""
    pos = pd.DataFrame([{"source_id": "S1", "ra_deg": 10.0, "dec_deg": 5.0},
                        {"source_id": "S2", "ra_deg": 11.0, "dec_deg": 5.0}])
    allwise = pd.DataFrame([
        # S1 has two AllWISE sources in the PSF: the nearer one must win.
        {"source_id": "S1", "angDist": 1.2, "W1mag": 15.0, "W2mag": 13.0,
         "W3mag": 9.0, "W4mag": 7.5, "ex": 0},
        {"source_id": "S1", "angDist": 4.4, "W1mag": 16.9, "W2mag": 16.5,
         "W3mag": 12.0, "W4mag": 9.0, "ex": 0},
        {"source_id": "S2", "angDist": 0.8, "W1mag": 14.0, "W2mag": 13.9,
         "W3mag": 12.5, "W4mag": 8.9, "ex": 1},
    ])
    twomass = pd.DataFrame([{"source_id": "S2", "angDist": 0.9, "Jmag": 15.0,
                             "Hmag": 14.4, "Kmag": 14.1}])
    out = acq.join_xmatch_photometry(
        pos, {"allwise": allwise, "twomass": twomass}, sc)
    r1 = out.set_index("source_id").loc["S1"]
    assert r1["w1"] == 15.0 and r1["w3"] == 9.0 and r1["w4"] == 7.5
    assert r1["n_ir_neighbours"] == 2          # both are inside the 6" PSF
    assert out.set_index("source_id").loc["S2", "2mass_ks"] == 14.1
    assert out.set_index("source_id").loc["S2", "ir_ext_flag"] == 1
    assert (out["ir_local_density_per_deg2"] > 0).all()

    # The crowded source must then be classified as a blend, not a candidate.
    done = runmod.stage_classify(out, sc)
    assert done.set_index("source_id").loc["S1", "class"] == "BLEND_CONFUSION"
    assert (done["p_chance_match"] > 0).all()


def test_join_is_a_no_op_without_matches(sc):
    pos = pd.DataFrame([{"source_id": "S1", "ra_deg": 1.0, "dec_deg": 2.0}])
    out = acq.join_xmatch_photometry(pos, {"allwise": pd.DataFrame()}, sc)
    assert list(out.columns) == ["source_id", "ra_deg", "dec_deg"]


def test_nearest_per_source_picks_the_closest():
    xm = pd.DataFrame([{"source_id": "A", "angDist": 3.0, "W1mag": 1.0},
                       {"source_id": "A", "angDist": 0.4, "W1mag": 2.0},
                       {"source_id": "B", "angDist": 1.0, "W1mag": 3.0}])
    best = acq.nearest_per_source(xm)
    assert len(best) == 2
    assert float(best.set_index("source_id").loc["A", "W1mag"]) == 2.0


def test_epoch_propagation_applied_across_a_table(sc):
    ra0, dec0, neigh = _high_pm_case(sc["epochs"])
    gaia = neigh.rename(columns={"ra_deg": "RA_ICRS", "dec_deg": "DE_ICRS",
                                 "pmra": "pmRA", "pmdec": "pmDE"})
    gaia["source_id"] = "PM1"
    df = pd.DataFrame([{"source_id": "PM1", "ra_deg": ra0, "dec_deg": dec0},
                       {"source_id": "STILL", "ra_deg": 5.0, "dec_deg": 5.0}])
    out = runmod.apply_epoch_propagation(df, gaia, sc)
    assert bool(out.set_index("source_id").loc["PM1", "pm_recovered"]) is True
    assert bool(out.set_index("source_id").loc["STILL", "pm_recovered"]) is False
    # A Gaia table without proper motions must not silently claim recovery.
    assert "pm_recovered" not in runmod.apply_epoch_propagation(
        df, gaia.drop(columns=["pmRA"]), sc).columns


def test_ra_dec_column_detection():
    """The archive's own column names must be found, whatever they are."""
    for cols in (["RA_NEOWISE", "DEC_NEOWISE"], ["RAJ2000", "DEJ2000"],
                 ["ra", "dec"], ["_RAJ2000", "_DEJ2000"], ["RA_ICRS", "DE_ICRS"],
                 ["radeg", "dedeg"]):
        df = pd.DataFrame({cols[0]: [1.0], cols[1]: [2.0], "junk": ["x"]})
        assert acq._ra_dec_columns(df) == (cols[0], cols[1]), cols
    assert acq._ra_dec_columns(pd.DataFrame({"w1mpro": [1.0]})) == (None, None)


def test_allsky_query_forms_are_tried_in_order(sc):
    """`nocoor=1` is undocumented, so several query forms must be generated."""
    urls = acq._svo_urls("http://host/vanish-neowise", sc)
    assert len(urls) >= 4
    assert any("VERB=2&format=ascii" in u for u in urls)
    assert any("nocoor=1" in u for u in urls)
    assert all(u.startswith("http://host/vanish-neowise") for u in urls)
    assert any("/cs.php?" in u for u in urls)
    # A cone query is a single form.
    cone = acq._svo_urls("http://host/x", sc, ra=10.0, dec=-5.0, sr=1.0)
    assert all("SR=1.0000" in u and "RA=10.000000" in u for u in cone)

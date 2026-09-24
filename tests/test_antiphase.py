"""Offline suite for ANTIPHASE (docs/antiphase.md).

(i) an injected grey fade + balanced IR rise is recovered as a candidate;
(ii) the dominant confounders (reddening dust, line-of-sight dust that also
dims the IR, the NEOWISE drift, independent noise paired at random) come out
clean; (iii) empty / failed archives degrade to explicit verdicts; (iv) every
rule of the ladder is tripped by a case built to trip it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.antiphase import acquire as acq
from seti.antiphase.classify import (
    blend_explains,
    blend_prediction,
    final_verdict,
    natural_flags,
    periodogram_features,
)
from seti.antiphase.coupling import (
    assess_coupling,
    bin_at_epochs,
    classify_chroma,
    colour_slope,
    ir_rise_prescore,
    year_to_mjd,
)
from seti.antiphase.energy import bc_g, energy_budget, f_bol, fit_blackbody, planck_nu
from seti.antiphase.null import align_to, inject, ir_excess_dmag, run_null, score_fap
from seti.antiphase.pipeline import evaluate_pack, make_pack
from seti.antiphase.run import (
    control_outcome,
    controls_gate,
    load_config,
    run_controls,
    run_reduce,
    run_shard,
)

SUN = {"phot_g_mean_mag": 13.5, "bp_rp": 0.82, "teff_gspphot": 5780.0, "w1mpro": 11.95,
       "w2mpro": 12.0, "w3mpro": 11.95, "ra": 200.0, "dec": 40.0, "parallax": 8.0,
       "pmra": 30.0, "pmdec": -20.0}
BLEND_OK = {"status": "OK", "explains": False}
CONF = load_config()


# ---------------------------------------------------------------------------
# synthetic stars
# ---------------------------------------------------------------------------
def ir_epochs(rng, w1=11.95, w2=12.0, err=0.008, t0=2014.1, t1=2024.55):
    t = np.arange(t0, t1, 0.5) + rng.normal(0, 0.01, int(np.ceil((t1 - t0) / 0.5)))[:len(np.arange(t0, t1, 0.5))]
    rows = []
    for b, m0 in (("W1", w1), ("W2", w2)):
        for tt in t:
            rows.append({"band": b, "t_yr": float(tt), "mag": m0 + rng.normal(0, err), "err": err})
    return pd.DataFrame(rows)


def ztf_points(rng, g0=14.2, r0=13.8, err=0.012, t0=2018.3, t1=2024.8, cadence_d=3.0):
    mjd = np.arange(year_to_mjd(t0), year_to_mjd(t1), cadence_d)
    # seasonal gaps: a field is unobservable ~3 months a year
    phase = ((mjd - mjd[0]) / 365.25) % 1.0
    mjd = mjd[phase < 0.75]
    out = {}
    for b, m0 in (("g", g0), ("r", r0)):
        out[b] = {"mjd": mjd.copy(), "mag": m0 + rng.normal(0, err, mjd.size),
                  "err": np.full(mjd.size, err), "n": int(mjd.size)}
    return out


def fade(opt, t_lo, t_hi, dmag):
    """Apply a fade ``dmag`` (per band dict or scalar) between t_lo..t_hi (yr)."""
    out = {}
    for b, v in opt.items():
        t = 2000.0 + (v["mjd"] - 51544.5) / 365.25
        d = dmag[b] if isinstance(dmag, dict) else dmag
        m = v["mag"].copy()
        m[(t >= t_lo) & (t <= t_hi)] += d
        out[b] = {**v, "mag": m}
    return out


def ir_step(ep, t_lo, t_hi, dmag: dict):
    e = ep.copy()
    for b, d in dmag.items():
        sel = (e["band"] == b) & (e["t_yr"] >= t_lo) & (e["t_yr"] <= t_hi)
        e.loc[sel, "mag"] = e.loc[sel, "mag"] + d
    return e


def grey_balanced(rng, frac=0.06, t_k=600.0, meta=SUN, t_lo=2020.8, t_hi=2021.9):
    ep = ir_epochs(rng)
    opt = ztf_points(rng)
    dm = -2.5 * np.log10(1 - frac)
    opt = fade(opt, t_lo, t_hi, dm)
    dir_ = ir_excess_dmag(meta, frac, t_k)
    ep = ir_step(ep, t_lo, t_hi, dir_)
    return ep, opt


def run_one(ep, opt, meta=SUN, blend=BLEND_OK, conf=CONF):
    pack = make_pack(ep, opt, meta, conf["coupling"])
    return evaluate_pack(pack, conf, blend=blend)


# ---------------------------------------------------------------------------
# (i) recovery
# ---------------------------------------------------------------------------
def test_an_injected_grey_balanced_fade_is_a_candidate():
    rec = run_one(*grey_balanced(np.random.default_rng(1)))
    assert rec["coupling_label"] == "COUPLED"
    assert rec["chroma"] == "GREY"
    assert rec["energy_verdict"] == "BALANCED"
    assert rec["verdict"] == "ANTIPHASE_CANDIDATE"
    assert 400 < rec["t_bb_k"] < 900
    assert rec["best_lag"] in (-1, 0, 1)


def test_a_habitable_zone_reradiator_is_recovered_without_a_w1_rise():
    """300 K: W2 answers, W1 barely moves; the W2-led rule must not reject it."""
    rec = run_one(*grey_balanced(np.random.default_rng(2), frac=0.08, t_k=300.0))
    assert rec["coupling_label"] == "COUPLED"
    assert not rec["ir_two_band"]
    assert rec["energy_verdict"] == "BALANCED"
    assert rec["verdict"] == "ANTIPHASE_CANDIDATE"


def test_injection_helper_matches_the_energy_budget():
    d = ir_excess_dmag(SUN, 0.05, 600.0)
    assert d["W2"] < d["W1"] < 0              # 600 K: W2 gains more than W1
    ex_w2 = 10 ** (-0.4 * d["W2"]) - 1
    assert ex_w2 > 0.3


# ---------------------------------------------------------------------------
# (ii) the dominant confounders
# ---------------------------------------------------------------------------
def test_ism_like_dust_reddening_is_natural_even_with_an_ir_rise():
    rng = np.random.default_rng(3)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2020.8, 2021.9, {"g": 0.30 * 1.41, "r": 0.30})
    ep = ir_step(ep, 2020.8, 2021.9, ir_excess_dmag(SUN, 0.24, 700.0))
    rec = run_one(ep, opt)
    assert rec["coupling_label"] == "COUPLED"
    assert rec["chroma"].startswith("REDDENING")
    assert rec["verdict"] == "NATURAL" and rec["natural_class"] == "NATURAL_CHROMATIC"


def test_line_of_sight_dust_that_also_dims_the_ir_is_not_coupled():
    """Brief section 4: an ordinary dust fade dims NEOWISE at ~6% of the optical rate."""
    rng = np.random.default_rng(4)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2020.8, 2021.9, {"g": 0.6 * 1.41, "r": 0.6})
    ep = ir_step(ep, 2020.8, 2021.9, {"W1": 0.6 * 0.06 * 1.2, "W2": 0.6 * 0.06})
    rec = run_one(ep, opt)
    assert rec["coupling_label"] in ("FADE_IR_FADES", "FADE_IR_FLAT")
    assert rec["verdict"] != "ANTIPHASE_CANDIDATE"


def test_the_neowise_drift_alone_is_not_a_fade():
    rng = np.random.default_rng(5)
    ep = ir_epochs(rng)
    ep.loc[:, "mag"] = ep["mag"] + 0.0056 * (ep["t_yr"] - 2014.0)   # W2-sized drift, fading
    rec = run_one(ep, ztf_points(rng))
    assert rec["coupling_label"] == "NO_FADE"


def test_a_flat_star_is_clean():
    rng = np.random.default_rng(6)
    rec = run_one(ir_epochs(rng), ztf_points(rng))
    assert rec["coupling_label"] in ("NO_FADE", "IR_RISE_NO_FADE")
    assert rec["verdict"] != "ANTIPHASE_CANDIDATE"


def test_random_pairs_of_independent_stars_are_not_coupled():
    rng = np.random.default_rng(7)
    packs = {}
    for k in range(40):
        ep, opt = ir_epochs(rng), ztf_points(rng)
        packs[str(k)] = make_pack(ep, opt, SUN, CONF["coupling"])
    nl = run_null(packs, CONF["coupling"], n_rounds=3, shifts=(-2, 2))
    assert nl["status"] == "OK"
    assert nl["pair_coupled_mean"] == 0.0
    assert sum(nl["shift_coupled"].values()) == 0


def test_a_coupled_star_is_not_coupled_to_other_stars_optical():
    rng = np.random.default_rng(8)
    packs = {}
    for k in range(20):
        ep, opt = ir_epochs(rng), ztf_points(rng)
        packs[str(k)] = make_pack(ep, opt, SUN, CONF["coupling"])
    ep, opt = grey_balanced(rng)
    packs["event"] = make_pack(ep, opt, SUN, CONF["coupling"])
    nl = run_null(packs, CONF["coupling"], n_rounds=5, shifts=(-4, 4))
    assert nl["pair_coupled_mean"] <= 0.2       # the event's own IR paired elsewhere fades nowhere
    assert score_fap(50.0, nl["pair_scores"], 21 * 5) < 0.05


# ---------------------------------------------------------------------------
# (iii) honest degradation
# ---------------------------------------------------------------------------
def test_no_optical_is_an_explicit_verdict():
    rng = np.random.default_rng(9)
    rec = run_one(ir_epochs(rng), {})
    assert rec["coupling_label"] == "NO_OPTICAL"
    assert rec["verdict"] == "NO_OPTICAL"


def test_south_of_ztf_is_not_in_footprint_without_a_request():
    assert acq.fetch_ztf(120.0, -40.0)["status"] == "NOT_IN_FOOTPRINT"


def test_insufficient_overlap_is_explicit():
    rng = np.random.default_rng(10)
    opt = ztf_points(rng, t0=2023.5, t1=2024.8)
    rec = run_one(ir_epochs(rng), opt)
    assert rec["coupling_label"] == "INSUFFICIENT_MATCHED"


def test_reduce_with_no_shards_says_no_data(tmp_path):
    s = run_reduce(CONF, tmp_path, 3, fetchers={"gaia_cone": lambda *a: {"status": "FAILED"}})
    assert "NO_DATA_REACHED" in s["verdict"]
    assert "shards_missing:3" in s["verdict"]
    assert "CONTROLS_NOT_RUN" in s["verdict"]


def test_shard_without_ignition_input_says_so(tmp_path):
    rep = run_shard(CONF, tmp_path / "none", tmp_path / "out", 0, 2,
                    ztf_fetch=lambda *a, **k: {"status": "FAILED"})
    assert rep["status"] == "NO_IGNITION_INPUT"


def test_controls_with_every_archive_down_are_untestable_not_passed():
    down = {k: (lambda *a, **kw: {"status": "FAILED", "error": "down"})
            for k in ("sesame", "ztf", "gaia_alert", "asassn", "gaia_cone", "allwise")}
    down["neowise"] = lambda *a, **kw: (pd.DataFrame(), {"status": "QUERY_FAILED"})
    res = run_controls({**CONF, "controls": CONF["controls"][:3]}, down)
    assert res["gate"] == "UNTESTED"
    assert all(v.startswith("UNTESTABLE") for v in res["outcomes"].values())


# ---------------------------------------------------------------------------
# (iv) every rule
# ---------------------------------------------------------------------------
def test_yso_flag_from_pre_existing_excess_and_region():
    cd = {"depth_mag": 0.1}
    assert "YSO_DIPPER" in natural_flags({**SUN, "w1mpro": 11.7, "w2mpro": 11.4}, cd, None,
                                         CONF["classify"])["flags"]
    assert "YSO_DIPPER" in natural_flags({**SUN, "ra": 68.0, "dec": 25.0}, cd, None,
                                         CONF["classify"])["flags"]
    assert "YSO_DIPPER" not in natural_flags(SUN, cd, None, CONF["classify"])["flags"]


def test_rcrb_like_needs_a_giant_and_a_deep_fade():
    giant = {**SUN, "phot_g_mean_mag": 10.0, "parallax": 0.2}
    assert "RCRB_LIKE" in natural_flags(giant, {"depth_mag": 2.0}, None)["flags"]
    assert "RCRB_LIKE" not in natural_flags(giant, {"depth_mag": 0.2}, None)["flags"]
    assert "RCRB_LIKE" not in natural_flags(SUN, {"depth_mag": 2.0}, None)["flags"]


def test_periodic_variables_are_lpv_or_eb_and_a_single_event_is_neither():
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(year_to_mjd(2018.3), year_to_mjd(2024.8), 700))
    e = np.full(t.size, 0.01)
    lpv = periodogram_features(t, 14 + 0.3 * np.sin(2 * np.pi * t / 310.0) + rng.normal(0, .01, t.size), e)
    assert "LPV" in natural_flags(SUN, {"depth_mag": 0.3}, lpv)["flags"]
    eb = periodogram_features(t, 14 + 0.05 * np.sin(2 * np.pi * t / 7.3) + rng.normal(0, .01, t.size), e)
    assert "EB_DUSTY_DISK" in natural_flags(SUN, {"depth_mag": 0.05}, eb)["flags"]
    m = 14 + rng.normal(0, 0.01, t.size)
    ty = 2000 + (t - 51544.5) / 365.25
    m[(ty > 2021) & (ty < 2021.8)] += 0.2
    single = periodogram_features(t, m, e)
    f = natural_flags(SUN, {"depth_mag": 0.2}, single)["flags"]
    assert "LPV" not in f and "EB_DUSTY_DISK" not in f


def test_chroma_classes():
    assert classify_chroma(1.0, 0.05) == "GREY"
    assert classify_chroma(1.41, 0.05) == "REDDENING_ISM_LIKE"
    assert classify_chroma(2.2, 0.05) == "REDDENING_STEEP"
    assert classify_chroma(0.6, 0.05) == "BLUEING"
    assert classify_chroma(1.2, 0.3) == "AMBIGUOUS"
    k, ke = colour_slope([0.1, 0.2, 0.3], [0.01] * 3, [0.1, 0.2, 0.3], [0.01] * 3)
    assert abs(k - 1) < 1e-6


def test_blueing_is_natural():
    rng = np.random.default_rng(12)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2020.8, 2021.9, {"g": 0.10, "r": 0.20})
    ep = ir_step(ep, 2020.8, 2021.9, ir_excess_dmag(SUN, 0.15, 700.0))
    rec = run_one(ep, opt)
    assert rec["chroma"] == "BLUEING" and rec["natural_class"] == "NATURAL_CHROMATIC"


def test_stellar_temperature_ir_is_a_companion():
    rng = np.random.default_rng(13)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2020.8, 2021.9, 0.07)
    ep = ir_step(ep, 2020.8, 2021.9, {"W1": -0.20, "W2": -0.20})     # photospheric colour
    rec = run_one(ep, opt)
    assert rec["coupling_label"] == "COUPLED"
    assert rec["natural_class"] == "STELLAR_IR"


def test_an_occulter_on_our_line_of_sight_only_is_an_ir_deficit():
    """ASASSN-24fw-like: a deep grey fade with a small IR answer."""
    rng = np.random.default_rng(14)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2020.8, 2021.9, 1.0)
    ep = ir_step(ep, 2020.8, 2021.9, ir_excess_dmag(SUN, 0.6, 700.0, ratio=0.02))
    rec = run_one(ep, opt)
    assert rec["coupling_label"] == "COUPLED"
    assert rec["natural_class"] == "GREY_IR_DEFICIT"


def test_an_ir_surplus_is_not_the_absorbed_light():
    rng = np.random.default_rng(15)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2020.8, 2021.9, 0.04)
    ep = ir_step(ep, 2020.8, 2021.9, ir_excess_dmag(SUN, 0.036, 800.0, ratio=40.0))
    rec = run_one(ep, opt)
    assert rec["coupling_label"] == "COUPLED"
    assert rec["natural_class"] == "GREY_IR_SURPLUS"


def test_an_ir_rise_that_leads_the_fade_is_lagged():
    """ASASSN-21qj: the IR brightened ~2.5 yr before the optical dimmed."""
    rng = np.random.default_rng(16)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2022.0, 2023.4, 0.3)
    dir_ = ir_excess_dmag(SUN, 0.24, 600.0)
    ep = ir_step(ep, 2019.4, 2020.9, dir_)                         # the afterglow, earlier
    ep = ir_step(ep, 2022.0, 2023.4, {k: 0.12 * v for k, v in dir_.items()})
    rec = run_one(ep, opt)
    assert rec["best_lag"] < -1
    assert rec["coupling_label"] in ("COUPLED", "LAGGED_COUPLING")
    assert rec["verdict"] == "NATURAL" and "LAGGED" in rec["reasons"]


def test_an_afterglow_that_has_faded_by_the_transit_is_a_lagged_coupling():
    """The ASASSN-21qj shape seen in run 36030068228: the IR is back at (or
    below) its reference when the optical dims, so the simultaneous test says
    FADE_IR_FADES / FLAT; the lag scan must still name it."""
    rng = np.random.default_rng(25)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2021.9, 2023.4, 0.25)
    ep = ir_step(ep, 2019.4, 2020.9, ir_excess_dmag(SUN, 0.2, 600.0))
    rec = run_one(ep, opt)
    assert rec["coupling_label"] == "LAGGED_COUPLING"
    assert rec["lag_ir_sigma"] >= 3 and rec["best_lag"] <= -2
    assert rec["verdict"] == "NATURAL" and "LAGGED" in rec["reasons"]


def test_w1_only_and_disagreeing_bands_and_flat_ir():
    rng = np.random.default_rng(17)
    ep = ir_epochs(rng)
    opt = fade(ztf_points(rng), 2020.8, 2021.9, 0.08)
    assert run_one(ir_step(ep, 2020.8, 2021.9, {"W1": -0.1}), opt)["coupling_label"] == "W1_ONLY"
    r = run_one(ir_step(ep, 2020.8, 2021.9, {"W1": 0.1, "W2": -0.1}), opt)
    assert r["coupling_label"] == "IR_BANDS_DISAGREE"
    assert run_one(ep, opt)["coupling_label"] == "FADE_IR_FLAT"


def test_not_proportional():
    """W2 up at the faded epochs but more so at unfaded ones: not a response to the fade."""
    t = np.arange(2014.1, 2024.6, 0.5)
    n = t.size
    rng = np.random.default_rng(18)
    opt = {b: (np.where(t > 2018.2, 14 + rng.normal(0, 0.003, n), np.nan),
               np.where(t > 2018.2, 0.008, np.nan)) for b in "gr"}
    k = np.nonzero(t > 2018.2)[0]
    for b in "gr":
        opt[b][0][k[3]] += 0.1
    w2 = 12 + rng.normal(0, 0.003, n)
    w2[k[3]] -= 0.05
    for j in (k[5], k[7], k[9], k[10]):
        w2[j] -= 0.3
    ir = {"W1": (11.95 + rng.normal(0, 0.003, n), np.full(n, 0.008)), "W2": (w2, np.full(n, 0.008))}
    r = assess_coupling(t, opt, ir, CONF["coupling"])
    assert r.label == "NOT_PROPORTIONAL"


def test_one_optical_band_is_coupled_but_incomplete():
    rng = np.random.default_rng(19)
    ep, opt = grey_balanced(rng)
    cc = {**CONF, "coupling": {**CONF["coupling"], "min_optical_bands": 1}}
    rec = run_one(ep, {"g": opt["g"]}, conf=cc)
    assert rec["coupling_label"] == "COUPLED"
    assert rec["verdict"] == "COUPLED_INCOMPLETE"


def test_blend_model_predicts_an_approaching_neighbour():
    tgt = {"ra": 200.0, "dec": 40.0, "pmra": 0.0, "pmdec": 0.0, "w1mpro": 12.0, "w2mpro": 12.0}
    # a bright neighbour 8" east moving 700 mas/yr west: ~1.5" closer by 2024
    nb = [{"ra": 200.0 + 8.0 / 3600 / np.cos(np.radians(40)), "dec": 40.0, "pmra": -700.0,
           "pmdec": 0.0, "phot_g_mean_mag": 12.5, "bp_rp": 2.5}]
    t = np.arange(2014.1, 2024.6, 0.5)
    p = blend_prediction(tgt, nb, t)
    assert p["status"] == "OK" and p["max_brightening"]["W1"] > 0.02
    faded = [float(x) for x in t[-3:]]
    be = blend_explains(p, faded, t, {"W1": -0.03, "W2": -0.03})
    assert be["explains"]
    fv = final_verdict({"label": "COUPLED", "chroma": "GREY", "best_lag": 0}, {"verdict": "BALANCED"},
                       {"flags": [], "untested": []}, be)
    assert fv["natural_class"] == "BLEND_PREDICTED"
    far = blend_prediction(tgt, [{**nb[0], "pmra": 0.0, "ra": 200.0 + 30 / 3600}], t)
    assert not blend_explains(far, faded, t, {"W1": -0.03, "W2": -0.03})["explains"]


def test_untested_blend_is_listed_not_passed():
    fv = final_verdict({"label": "COUPLED", "chroma": "GREY", "best_lag": 0},
                       {"verdict": "BALANCED"}, {"flags": [], "untested": []}, None)
    assert "neighbour_blend" in fv["untested"]


# ---------------------------------------------------------------------------
# physics and parsers
# ---------------------------------------------------------------------------
def test_bolometric_correction_and_flux():
    assert abs(bc_g(5772.0) - 0.06) < 1e-9
    # the Sun: G = -26.90, BC 0.06 -> m_bol -26.84 -> ~1.36e3 W/m^2 (the solar constant)
    assert 1.2e3 < f_bol(-26.90, 5772.0) < 1.5e3


def test_blackbody_fit_recovers_temperature():
    from seti.antiphase.energy import JY, WISE_BANDS, C
    t_true, omega = 700.0, 1e-18
    ex = {b: omega * float(planck_nu(C / (WISE_BANDS[b]["lam_um"] * 1e-6), t_true)) / JY
          for b in ("W1", "W2")}
    bb = fit_blackbody(ex, {b: 0.01 * ex[b] for b in ex})
    assert abs(bb["t_best_k"] - t_true) / t_true < 0.05


def test_energy_budget_verdicts():
    base = {"W1": 11.95, "W2": 12.0}
    bal = energy_budget(13.5, 5780.0, 0.05, 0.005, base, ir_excess_dmag(SUN, 0.05, 600.0),
                        {"W1": 0.01, "W2": 0.01})
    assert bal["verdict"] == "BALANCED" and 0.5 < bal["ratio_best"] < 2.0
    none = energy_budget(13.5, 5780.0, 0.05, 0.005, base, {"W1": 0.01, "W2": 0.01},
                         {"W1": 0.01, "W2": 0.01})
    assert none["verdict"] == "UNDETERMINED"


def test_binning_and_prescore():
    rng = np.random.default_rng(20)
    t = np.arange(year_to_mjd(2019.0), year_to_mjd(2020.0), 2.0)
    b = bin_at_epochs(t, 14 + rng.normal(0, 0.01, t.size), np.full(t.size, 0.01),
                      [year_to_mjd(2019.5), year_to_mjd(2023.0)])
    assert np.isfinite(b["mag"][0]) and np.isnan(b["mag"][1])
    assert abs(b["mag"][0] - 14) < 0.01
    tt = np.arange(2014.1, 2024.6, 0.5)
    w = np.full(tt.size, 12.0)
    w[tt > 2021] -= 0.2
    ir = {"W1": (w, np.full(tt.size, 0.01)), "W2": (w, np.full(tt.size, 0.01))}
    assert ir_rise_prescore(tt, ir) > 10


def test_align_to_nearest():
    out = align_to([2019.0, 2019.5, 2030.0], [2019.02, 2019.48], {"g": ([1.0, 2.0], [0.1, 0.2])})
    assert out["g"][0][0] == 1.0 and out["g"][0][1] == 2.0 and np.isnan(out["g"][0][2])


def test_inject_does_not_mutate_the_source():
    rng = np.random.default_rng(21)
    p = make_pack(ir_epochs(rng), ztf_points(rng), SUN, CONF["coupling"])
    g0 = p["opt"]["g"][0].copy()
    inject(p, 0.1, 600.0, 2, 2)
    assert np.array_equal(p["opt"]["g"][0], g0, equal_nan=True)


def test_parse_ztf_keeps_good_flags_and_dominant_oid():
    csv = ("oid,mjd,mag,magerr,filtercode,catflags\n"
           "1,59000,14.0,0.01,zg,0\n1,59001,14.1,0.01,zg,0\n2,59002,15.0,0.01,zg,0\n"
           "3,59000,12.0,0.01,zr,0\n3,59001,13.0,0.01,zr,32768\n")
    b = acq.parse_ztf_csv(csv, bright_limit=12.5)
    assert b["g"]["n"] == 2 and b["g"]["oid"] == "1"
    assert b["r"]["n"] == 1 and b["r"]["saturation_risk"]


def test_parse_gaia_alert_and_asu():
    txt = "Gaia20ehk\n#Date,JD,averagemag.\n2020-01-01,2458849.5,15.20\n2020-02-01,2458880.5,null\n"
    d = acq.parse_gaia_alert_csv(txt)
    assert len(d) == 1 and abs(d["mjd"].iloc[0] - 58849.0) < 1e-6
    asu = "#c\nSource\tGmag\n\t\n------\t----\n123\t12.3\n"
    assert acq.parse_asu_tsv(asu)["Gmag"].iloc[0] == "12.3"


def test_control_outcomes_and_gate():
    c = {"kind": "coupled_event"}
    assert control_outcome(c, {"verdict": "NATURAL", "coupling_label": "COUPLED"}, CONF) \
        == "RECOVERED_NATURAL"
    assert control_outcome(c, {"verdict": "ANTIPHASE_CANDIDATE", "coupling_label": "COUPLED"},
                           CONF).startswith("FAIL")
    assert control_outcome(c, {"verdict": "FADE_IR_FLAT", "coupling_label": "FADE_IR_FLAT",
                               "n_matched": 10, "n_faded": 2}, CONF) == "MISSED_FADE_IR_FLAT"
    assert control_outcome(c, {"verdict": "NO_FADE", "coupling_label": "NO_FADE",
                               "n_matched": 10, "n_faded": 0}, CONF) \
        == "UNTESTABLE_NO_FADE_IN_WINDOW"
    kinds = {"a": "coupled_event", "b": "natural_yso"}
    assert controls_gate({"a": "RECOVERED_NATURAL", "b": "NOT_COUPLED_NO_FADE"}, kinds) == "PASS"
    assert controls_gate({"a": "MISSED_FADE_IR_FLAT", "b": "RECOVERED_NATURAL"}, kinds) == "FAIL"
    assert controls_gate({"a": "UNTESTABLE_NO_DATA", "b": "UNTESTABLE_NO_DATA"}, kinds) == "UNTESTED"
    assert controls_gate({"a": "RECOVERED_NATURAL", "b": "FAIL_NATURAL_PASSED_AS_CANDIDATE"},
                         kinds) == "FAIL"


# ---------------------------------------------------------------------------
# end to end, offline: IGNITION-shaped input -> shard -> reduce
# ---------------------------------------------------------------------------
def _ignition_dir(tmp_path, rng, n_stars=60, event_sid="999"):
    d = tmp_path / "ign"
    d.mkdir()
    rows, eps, ztf = [], [], {}
    for k in range(n_stars):
        sid = event_sid if k == 0 else str(1000 + k)
        ra, dec = 150.0 + k * 0.5, 30.0 + (k % 7)
        rows.append({"source_id": int(sid), "ra": ra, "dec": dec, "b": 50.0, "parallax": 8.0,
                     "pmra": 20.0, "pmdec": -10.0, "phot_g_mean_mag": 13.5, "bp_rp": 0.82,
                     "teff_gspphot": 5780.0, "w1mpro": 11.95, "w2mpro": 12.0, "w3mpro": 11.95})
        if k == 0:
            ep, opt = grey_balanced(rng)
        else:
            ep, opt = ir_epochs(rng), ztf_points(rng)
        ep = ep.assign(source_id=sid, epoch=0, t_mjd=year_to_mjd(ep["t_yr"]), n_exp=12,
                       scatter=0.02)
        eps.append(ep)
        ztf[(round(ra, 5), round(dec, 5))] = opt
    # one southern star (not in the ZTF footprint)
    rows.append({**rows[-1], "source_id": 5, "dec": -45.0})
    eps.append(eps[-1].assign(source_id="5"))
    pd.DataFrame(rows).to_parquet(d / "parent_s0of1.parquet")
    pd.concat(eps, ignore_index=True)[["source_id", "band", "epoch", "t_mjd", "t_yr", "mag", "err",
                                       "n_exp", "scatter"]].to_csv(d / "epochs_s0of1.csv", index=False)
    return d, ztf


def test_shard_and_reduce_end_to_end(tmp_path):
    rng = np.random.default_rng(22)
    idir, ztf = _ignition_dir(tmp_path, rng)

    def fake_ztf(ra, dec, pmra=0.0, pmdec=0.0, **kw):
        if dec < acq.ZTF_DEC_MIN:
            return {"status": "NOT_IN_FOOTPRINT"}
        o = ztf.get((round(ra, 5), round(dec, 5)))
        if o is None:
            return {"status": "NO_ROWS"}
        return {"status": "OK", "bands": {b: {**v, "median_mag": float(np.median(v["mag"])),
                                              "saturation_risk": False} for b, v in o.items()}}

    conf = {**CONF, "survey": {**CONF["survey"], "null_rounds": 3, "injection_max_stars": 10,
                               "injection_depths": [0.1], "injection_temps_k": [600.0],
                               "ztf_workers": 2}}
    out = tmp_path / "out"
    rep = run_shard(conf, idir, out, 0, 1, ztf_fetch=fake_ztf, budget_s=600)
    assert rep["status"] == "OK"
    assert rep["funnel"]["n_parent"] == 61 and rep["funnel"]["n_ztf_footprint"] == 60
    assert rep["ztf"]["counts"].get("OK") == 60
    st = pd.read_csv(out / "stars_s0of1.csv", dtype={"source_id": str})
    ev = st[st["source_id"] == "999"].iloc[0]
    assert ev["verdict"] == "ANTIPHASE_CANDIDATE"
    assert (st["verdict"] == "ANTIPHASE_CANDIDATE").sum() == 1
    inj = rep["injection"]["grid"][0]
    assert inj["n"] > 0 and inj["frac_coupled"] >= 0.8
    # resume: nothing is re-fetched
    rep2 = run_shard(conf, idir, out, 0, 1, ztf_fetch=lambda *a, **k: 1 / 0, budget_s=600)
    assert rep2["n_resumed_done"] == 60 and not rep2["ztf"]["counts"]
    (out / "controls.json").write_text(json.dumps({"gate": "PASS", "outcomes": {}}))

    def fake_cone(ra, dec, r):
        return {"status": "OK", "rows": [{"source_id": "999", "ra": 150.0, "dec": 30.0,
                                          "pmra": 20.0, "pmdec": -10.0,
                                          "phot_g_mean_mag": 13.5, "bp_rp": 0.82,
                                          "sep_arcsec": 0.0}]}
    s = run_reduce(conf, out, 1, fetchers={"gaia_cone": fake_cone}, run_id="test")
    assert s["self_consistency"]["labels_sum_to_evaluated"]
    assert s["self_consistency"]["shards_found_equals_expected"]
    assert s["funnel"]["n_evaluated"] == 60
    # the reduce re-runs the ladder on every faded star from stored series, same answer
    assert s["reevaluated_in_reduce"]["n"] >= 1 and s["reevaluated_in_reduce"]["n_relabelled"] == 0
    cands = pd.read_csv(out / "candidates.csv", dtype={"source_id": str})
    row = cands[cands["source_id"] == "999"].iloc[0]
    # a 60-star shard's null cannot resolve FAP x N < 0.1: the candidate is held
    assert row["verdict_final"] in ("ANTIPHASE_CANDIDATE", "NOT_SIGNIFICANT_VS_NULL")
    assert s["controls_gate"] == "PASS"
    assert s["run_id"] == "test" and s["generated_utc"]


def test_batched_ztf_route_matches_ids_and_assembles_bands():
    stars = pd.DataFrame({"source_id": ["a", "b", "c"], "ra": [10.0, 20.0, 30.0],
                          "dec": [5.0, 6.0, 7.0], "pmra": [0.0, 0.0, 0.0],
                          "pmdec": [0.0, 0.0, 0.0]})

    def upload(q, tbl, timeout):
        assert "TAP_UPLOAD.pos" in q and "ztf_objects_dr24" in q
        rows = []
        for k in range(len(tbl)):
            ra, dec = float(tbl["ra"][k]), float(tbl["dec"][k])
            if k == 2:
                continue                                    # star c: no ZTF object
            rows += [{"sid": k, "ra_p": ra, "dec_p": dec, "oid": 100 + k, "ra": ra, "dec": dec,
                      "filtercode": "zg", "ngoodobsrel": 300},
                     {"sid": k, "ra_p": ra, "dec_p": dec, "oid": 200 + k, "ra": ra, "dec": dec,
                      "filtercode": "zr", "ngoodobsrel": 400},
                     # a fainter neighbour 1" away with fewer epochs loses
                     {"sid": k, "ra_p": ra, "dec_p": dec, "oid": 900 + k, "ra": ra,
                      "dec": dec + 1.0 / 3600, "filtercode": "zr", "ngoodobsrel": 20}]
        return pd.DataFrame(rows)

    def ids(oids):
        return {"status": "OK", "lcs": {o: {"mjd": np.arange(10.0) + 58500, "mag": np.full(10, 14.0),
                                            "err": np.full(10, 0.01)} for o in oids}}
    got = {}
    log = acq.fetch_ztf_batched(stars, table="ztf_objects_dr24", upload_fn=upload, ids_fn=ids,
                                on_result=lambda s, r: got.__setitem__(s, r))
    assert log["counts"] == {"OK": 2, "NO_ROWS": 1}
    assert got["a"]["bands"]["g"]["oid"] == "100" and got["a"]["bands"]["r"]["oid"] == "200"
    assert got["c"]["status"] == "NO_ROWS"


def test_batched_route_that_cannot_upload_hands_back_to_the_per_star_route():
    stars = pd.DataFrame({"source_id": ["a"], "ra": [10.0], "dec": [5.0]})

    def bad(q, tbl, timeout):
        raise RuntimeError("no such column ngoodobsrel")
    log = acq.fetch_ztf_batched(stars, table="t", upload_fn=bad, ids_fn=lambda i: {})
    assert log["route_failed"] and "ngoodobsrel" in log["error"]


def test_natural_sample_is_classified_and_a_leak_is_counted():
    from seti.antiphase.run import run_natural

    rng = np.random.default_rng(24)
    ep_dust, opt_dust = ir_epochs(rng), None
    opt_dust = fade(ztf_points(rng), 2020.8, 2021.9, {"g": 0.4 * 1.41, "r": 0.4})
    ep_dust = ir_step(ep_dust, 2020.8, 2021.9, ir_excess_dmag(SUN, 0.3, 700.0))
    ep_grey, opt_grey = grey_balanced(rng, frac=0.15)

    def vsx(vt, **kw):
        return {"status": "OK", "rows": [{"name": f"{vt} A", "ra": 10.0, "dec": 5.0,
                                          "vsx_type": vt, "max": "13", "period": ""}]}
    series = {"vsx0": (ep_dust, opt_dust), "vsx1": (ep_grey, opt_grey)}

    def neo(objs):
        return {s: series[s][0].assign(source_id=s) for s in objs["source_id"]}, {"chunks": []}

    def ztf_b(have, table, budget_s, on_result):
        for s in have["source_id"]:
            on_result(s, {"status": "OK", "bands": series[s][1]})
        return {"counts": {"OK": len(have)}}
    meta_row = [{"source_id": "x", "ra": 10.0, "dec": 5.0, "pmra": 0.0, "pmdec": 0.0,
                 "parallax": 8.0, "phot_g_mean_mag": 13.5, "bp_rp": 0.82, "sep_arcsec": 0.1}]
    fx = {"vsx": vsx, "neowise_many": neo, "ztf_table": lambda: {"status": "OK", "table": "t"},
          "ztf_batched": ztf_b, "gaia_cone": lambda *a: {"status": "OK", "rows": meta_row},
          "allwise": lambda *a: {"status": "OK", "w1mpro": 11.95, "w2mpro": 12.0,
                                 "w3mpro": 11.95}}
    conf = {**CONF, "natural": {"types": {"RCB": 1, "UXOR": 1}, "ztf_route": "batched"}}
    res = run_natural(conf, fx)
    s = res["summary"]
    assert s["by_type"]["RCB"]["natural_classes"].get("NATURAL_CHROMATIC") == 1
    # the grey balanced one, filed under UXOR, is exactly what a leak looks like
    assert s["n_leaks"] == 1 and s["leaks"] == ["UXOR A"]


def test_multi_id_csv_parser():
    csv = ("oid,mjd,mag,magerr,filtercode,catflags\n5,1,14,0.01,zg,0\n5,2,14,0.01,zg,32768\n"
           "6,1,13,0.01,zr,0\n")
    d = acq.parse_ztf_multi(csv)
    assert len(d["5"]["mjd"]) == 1 and len(d["6"]["mjd"]) == 1


def test_a_control_with_non_overlapping_bands_is_judged_on_the_best_single_band():
    """ASAS-SN V ended in 2018 as its g began: V+g together match no epoch."""
    from seti.antiphase.run import choose_optical

    rng = np.random.default_rng(23)
    ep, opt = grey_balanced(rng)
    v = ztf_points(rng, t0=2014.2, t1=2018.3)["g"]
    rec, pack, tried = choose_optical(ep, {"asassn_V": v, "asassn_g": opt["g"]}, SUN, CONF)
    assert rec["optical_bands"] == "asassn_g"
    assert rec["coupling_label"] == "COUPLED"
    assert any(t["bands"] == ["asassn_V", "asassn_g"] and t["n_matched"] == 0 for t in tried)


@pytest.mark.parametrize("name", ["Gaia-GIC-1", "ASASSN-21qj", "ASASSN-24fw"])
def test_configured_controls_carry_references(name):
    c = {x["name"]: x for x in CONF["controls"]}[name]
    assert c.get("ref") and c.get("kind")

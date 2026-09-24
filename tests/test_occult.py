"""OCCULT offline suite: the opaque-lens detector on synthetic multi-site photometry.

No network.  Covers: injected signals recovered (wing and central-hole
regimes), clean nulls on the dominant confounders (plain FSPL, a binary-lens
brightening, a one-sided "disc" step, a chromatic or single-site impostor),
honest degradation on an empty archive, every gate rule, the parsers of the
formats the runner probe measured, and the injection / efficiency machinery.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import tarfile
from datetime import datetime, timezone

import numpy as np
import pytest

from seti.occult import acquire as A
from seti.occult import detect as D
from seti.occult import inject as INJ
from seti.occult import run as R
from seti.roman.lens import magnifications, occulted_magnification_point, paczynski_magnification

# --------------------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------------------


def test_minor_image_hidden_beyond_u_c_and_blend_degeneracy():
    rl = 0.6
    uc = D.u_crit(rl)
    u = np.array([0.5 * uc, 0.99 * uc, 1.01 * uc, 2 * uc])
    ap, am = magnifications(u)
    occ = occulted_magnification_point(u, rl)
    assert np.allclose(occ[:2], (ap + am)[:2])
    assert np.allclose(occ[2:], ap[2:])
    # A_+ = (A + 1)/2 exactly: with u0 > u_c the occulted curve IS a blended Paczynski curve
    assert np.allclose(ap, (paczynski_magnification(u) + 1) / 2)


def test_rho_l_inverses():
    for rl in (0.3, 0.6, 0.95):
        assert D.rho_l_from_uc(D.u_crit(rl)) == pytest.approx(rl, rel=1e-12)
    for rl in (1.05, 1.3, 2.5):
        assert D.rho_l_from_uh(D.u_crit(rl)) == pytest.approx(rl, rel=1e-12)


# --------------------------------------------------------------------------------------
# detector, end to end
# --------------------------------------------------------------------------------------

FAST = {"n_samples": 32}


def _ev(**kw):
    kw.setdefault("cadence_d", 0.33)
    return D.synth_event(**kw)


def test_injected_wing_occultation_is_recovered():
    rec = D.assess_event(_ev(rho_l=0.6, seed=2), FAST)
    assert rec["tier"] == D.TIER_CANDIDATE, rec["rejections"]
    assert rec["regime"] == "wing"
    assert rec["occult"]["rho_l"] == pytest.approx(0.6, rel=0.02)
    assert rec["alpha"]["alpha"] == pytest.approx(1.0, abs=0.1)
    assert len(rec["sites_seeing_step"]) >= 2


def test_injected_central_hole_is_recovered():
    rec = D.assess_event(_ev(rho_l=1.3, seed=4), FAST)
    assert rec["tier"] == D.TIER_CANDIDATE, rec["rejections"]
    assert rec["regime"] == "hole"
    assert rec["occult"]["rho_l"] == pytest.approx(1.3, rel=0.05)


def test_plain_fspl_is_a_clean_null():
    rec = D.assess_event(_ev(rho_l=None, seed=1), FAST)
    assert rec["tier"] == D.TIER_NO_OCC
    assert rec["dchi2"] < 50


def test_finite_source_peak_is_not_an_occultation():
    rec = D.assess_event(_ev(rho_l=None, u0=0.002, rho=0.03, seed=7), FAST)
    assert rec["tier"] != D.TIER_CANDIDATE
    assert rec["fspl"]["rho_star"] == pytest.approx(0.03, rel=0.15)


def test_binary_caustic_brightening_is_not_an_occultation():
    def bump(t):   # a caustic-crossing-like U: two spikes with an elevated floor between
        return 1.2 * ((t > 8002.0) & (t < 8004.0)) + 4.0 * (np.exp(-0.5 * ((t - 8002) / 0.05) ** 2)
                                                             + np.exp(-0.5 * ((t - 8004) / 0.05) ** 2))

    rec = D.assess_event(_ev(rho_l=None, seed=5, extra=bump), FAST)
    assert rec["tier"] != D.TIER_CANDIDATE


def test_one_sided_step_the_disc_twin_is_not_a_candidate():
    def one_side(t):
        u = np.sqrt(0.2**2 + ((t - 8000) / 25.0) ** 2)
        _, am = magnifications(u)
        return -3.0 * am * ((t > 8000) & (u > D.u_crit(0.6)))

    rec = D.assess_event(_ev(rho_l=None, seed=6, extra=one_side), FAST)
    assert rec["tier"] != D.TIER_CANDIDATE
    if rec["tier"] == D.TIER_REJECTED:
        assert {"ONE_SIDED_STEP", "SIDES_DIFFER_IN_DEPTH", "DEPTH_NOT_TIED_TO_RHO_L"} & set(rec["rejections"])


def test_single_site_occultation_is_rejected():
    rec = D.assess_event(_ev(rho_l=0.6, seed=8, sites=("KMTC",)), FAST)
    assert rec["tier"] == D.TIER_REJECTED
    assert "FEWER_THAN_TWO_SITES_SEE_IT" in rec["rejections"]


def test_chromatic_impostor_is_rejected():
    rec = D.assess_event(_ev(rho_l=0.6, seed=9, bands=("I", "V"), band_depth={"V": 0.0},
                             cadence_d=0.1), FAST)
    assert rec["tier"] == D.TIER_REJECTED
    assert "CHROMATIC" in rec["rejections"]


def test_one_site_only_impostor_is_not_a_candidate():
    """A step in one site's photometry only (a site systematic) never passes."""
    rec = D.assess_event(_ev(rho_l=0.6, seed=10, site_depth={"KMTA": 0.0, "KMTS": 0.0}), FAST)
    assert rec["tier"] != D.TIER_CANDIDATE
    if rec["tier"] == D.TIER_REJECTED:
        assert {"SITES_DISAGREE", "FEWER_THAN_TWO_SITES_SEE_IT"} & set(rec["rejections"])


def test_empty_event_degrades_honestly():
    ev = D.build_event("empty", [])
    rec = D.assess_event(ev)
    assert rec["tier"] == D.TIER_NO_DATA
    few = D.build_event("few", [({"name": "X", "site": "KMTA", "band": "I"}, [1, 2, 3], [1, 1, 1],
                                 [0.1, 0.1, 0.1])])
    assert D.assess_event(few)["tier"] == D.TIER_NO_DATA


def test_flat_star_is_not_lensing():
    rng = np.random.default_rng(3)
    t = np.arange(0, 200, 0.2)
    ev = D.build_event("flat", [({"name": f"KMT{s}_I", "site": f"KMT{s}", "band": "I"}, t + k / 3,
                                 1 + 0.01 * rng.normal(size=t.size), np.full(t.size, 0.01))
                                for k, s in enumerate("ACS")])
    assert D.assess_event(ev, FAST, {"t0": 100, "tE": 20, "u0": 0.3})["tier"] == D.TIER_NOT_LENSING


# --------------------------------------------------------------------------------------
# every gate rule, on crafted diagnostics
# --------------------------------------------------------------------------------------

def _passing():
    return {
        "regime": "wing", "dchi2": 500.0, "dchi2_anti": 20.0,
        "alpha": {"alpha": 1.0, "sigma": 0.05},
        "symmetry": {"rise": {"snr": 10.0}, "fall": {"snr": 10.0}, "alpha_z": 0.5, "edges_agree": True},
        "sites_seeing_step": ["KMTA", "KMTC"],
        "sites_consistency": {"consistent": True}, "bands_consistency": {"consistent": True},
        "jackknife": {"dchi2": 500.0, "min_drop_night": 450.0, "max_night_share": 0.1,
                      "min_drop_dataset": 300.0},
        "steps": [{"bracketed": True}, {"bracketed": True}],
        "positive_bump": {"max_sigma": 2.0}, "binned_redchi2_after": 1.1,
    }


def test_the_passing_diagnostics_pass():
    assert D.gate_rejections(_passing(), {}) == []


@pytest.mark.parametrize("mutate,rule", [
    (lambda d: d["alpha"].update(alpha=0.4), "DEPTH_NOT_TIED_TO_RHO_L"),
    (lambda d: d["alpha"].update(alpha=None), "DEPTH_NOT_TIED_TO_RHO_L"),
    (lambda d: d["symmetry"]["fall"].update(snr=1.0), "ONE_SIDED_STEP"),
    (lambda d: d["symmetry"].update(alpha_z=5.0), "SIDES_DIFFER_IN_DEPTH"),
    (lambda d: d["symmetry"].update(edges_agree=False), "SIDES_DISAGREE_ON_U_C"),
    (lambda d: d.update(sites_seeing_step=["KMTC"]), "FEWER_THAN_TWO_SITES_SEE_IT"),
    (lambda d: d["sites_consistency"].update(consistent=False), "SITES_DISAGREE"),
    (lambda d: d["bands_consistency"].update(consistent=False), "CHROMATIC"),
    (lambda d: d["jackknife"].update(min_drop_night=100.0), "ONE_NIGHT_DOMINATES"),
    (lambda d: d["jackknife"].update(max_night_share=0.8), "ONE_NIGHT_DOMINATES"),
    (lambda d: d["jackknife"].update(min_drop_dataset=100.0), "ONE_DATASET_DOMINATES"),
    (lambda d: d["steps"][1].update(bracketed=False), "STEP_IN_DATA_GAP"),
    (lambda d: d.update(steps=[]), "STEP_IN_DATA_GAP"),
    (lambda d: d.update(regime="hole", n_in_hole=1, hole_below_baseline=True), "HOLE_UNSAMPLED"),
    (lambda d: d.update(regime="hole", n_in_hole=50, hole_below_baseline=False),
     "HOLE_NOT_BELOW_BASELINE"),
    (lambda d: d["positive_bump"].update(max_sigma=9.0), "BRIGHTENING_LEFT_CAUSTIC_LIKE"),
    (lambda d: d.update(binned_redchi2_after=4.0), "RESIDUAL_STRUCTURE"),
    (lambda d: d.update(dchi2_anti=300.0), "ANTI_TEMPLATE_COMPARABLE"),
])
def test_each_gate_rule_trips(mutate, rule):
    d = _passing()
    mutate(d)
    assert rule in D.gate_rejections(d, {})


# --------------------------------------------------------------------------------------
# parsers (formats as measured by the runner probe)
# --------------------------------------------------------------------------------------

OGLE_PAR = """    Event     Field   StarNo  RA(J2000)   Dec(J2000)   Tmax(HJD)   Tmax(UT)      tau     umin  Amax  Dmag   fbl  I_bl    I0

2019-BLG-0001 BLG500.01 179275 17:53:50.46 -29:08:57.9 2458529.584 2019-02-15.08 54.55   0.176  5.737 1.897 0.161 17.267 19.252
2019-BLG-0161 BLG500.02 152053 17:30:52.66 -28:57:26.4 2458540.210 2019-02-10.96 2.061   0.524  2.101 0.806 1.000 15.999 15.999
"""

KMT_LIST = """KMT-2019-BLG-0001 BLG11M0308.009408 1 1 17:30:52.66 -28:57:26.42  8540.21006   24.04  0.120  21.37  19.62  19.62 1   2.74  OB190161
KMT-2019-BLG-0002 BLG01K0506.001741 1 1 17:51:52.00 -30:24:32.62  8559.52139   33.64  0.157  16.18  16.17  16.17 1   2.46
"""

PHOT = """2457416.87761 17.298 0.025 7.47 1620.0
2457424.86801 17.262 0.013 5.94 705.0
"""

PYSIS = """# HJD  \\Delta_flux flux_err  mag  mag_err  fwhm  sky  secz
2460727.23090   -7137.2279   774.6577    16.7603    0.0272   10.54  1383  1.74
2460728.23437     -43.3113   557.7957    17.0391    0.0253    5.71  3245  1.67
"""


def test_parse_ogle_lenses():
    rows = A.parse_ogle_lenses(OGLE_PAR, 2019)
    assert [r["name"] for r in rows] == ["OGLE-2019-BLG-0001", "OGLE-2019-BLG-0161"]
    assert rows[0]["t0"] == pytest.approx(8529.584)
    assert rows[0]["tE"] == pytest.approx(54.55)
    assert rows[0]["ra"] == pytest.approx(15 * (17 + 53 / 60 + 50.46 / 3600))
    assert rows[0]["dec"] == pytest.approx(-(29 + 8 / 60 + 57.9 / 3600))


def test_parse_kmt_listpage_and_units():
    k = A.parse_kmt_listpage(KMT_LIST, 2019)
    assert k[0]["name"] == "KMT-2019-BLG-0001" and k[0]["related"] == ["OB190161"]
    assert k[0]["t0"] == pytest.approx(8540.21006)
    assert A.ogle_name_from_token("OB190161") == "OGLE-2019-BLG-0161"
    units = A.build_units(A.parse_ogle_lenses(OGLE_PAR, 2019) + k)
    by = {u["unit"]: u for u in units}
    assert by["KMT-2019-BLG-0001"]["ogle"]["name"] == "OGLE-2019-BLG-0161"
    assert by["KMT-2019-BLG-0002"]["ogle"] is None
    assert "OGLE-2019-BLG-0001" in by and by["OGLE-2019-BLG-0001"]["kmt"] is None
    assert len(units) == 3
    assert A.kmt_tar_url(k[0]).endswith("/2019/data/KB190001/pysis/pysis.tar.gz")
    assert A.ogle_phot_url(by["OGLE-2019-BLG-0001"]["ogle"]).endswith("/2019/blg-0001/phot.dat")


def test_kmt_listpage_with_a_missing_class_column_is_read_by_coordinate_shape():
    line = ("KMT-2016-BLG-2586 SAO42M0805.086923 3 17:54:41.62 -28:34:00.05 7544.69315  109.77  "
            "0.089  22.81  20.00  25.00 3   1.99\n")
    r = A.parse_kmt_listpage(line, 2016)[0]
    assert r["dec"] == pytest.approx(-(28 + 34 / 60 + 0.05 / 3600))
    assert r["t0"] == pytest.approx(7544.69315) and r["tE"] == pytest.approx(109.77)


def test_parse_phot_and_pysis_and_tar():
    p = A.parse_phot_dat(PHOT)
    assert p["t"][0] == pytest.approx(7416.87761) and p["mag"][1] == pytest.approx(17.262)
    q = A.parse_pysis(PYSIS)
    assert q["t"][0] == pytest.approx(10727.23090) and q["flux"][1] == pytest.approx(-43.3113)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in ("KMTC01_I.pysis", "KMTS41_V.pysis", "README"):
            data = PYSIS.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    got = A.parse_pysis_tar(buf.getvalue())
    assert sorted(d["name"] for d, _ in got) == ["KMTC01_I", "KMTS41_V"]
    assert {d["site"] for d, _ in got} == {"KMTC", "KMTS"}


def test_public_seasons_respect_the_proprietary_year():
    s = A.public_seasons(datetime(2026, 9, 24, tzinfo=timezone.utc), last=2026)
    assert s[0] == 2016 and s[-1] == 2025 and 2026 not in s
    assert 2025 not in A.public_seasons(datetime(2026, 6, 30, tzinfo=timezone.utc), last=2026)


def test_quality_masks():
    tab = {"t": np.arange(20.0), "flux": np.ones(20), "err": np.full(20, 0.1),
           "fwhm": np.r_[np.full(19, 4.0), 20.0], "sky": np.full(20, 100.0)}
    assert A.quality_mask_kmt(tab).sum() == 19


# --------------------------------------------------------------------------------------
# injection and sensitivity
# --------------------------------------------------------------------------------------

def test_injection_into_a_baseline_is_recovered_and_counted():
    ev = _ev(rho_l=None, seed=12)
    trials = INJ.injection_trials(ev, FAST, None, [0.6])
    assert len(trials) == 1 and trials[0]["recovered"], trials
    tab = INJ.efficiency_table(trials * 5)
    cell = tab["cells"]["0.6|0.2-0.35"]
    assert cell["n"] == 5 and cell["eff"] == 1.0


def test_expected_map_is_zero_where_blend_degenerate():
    ev = _ev(rho_l=None, seed=13, u0=0.9)
    ev2, e, f0, _ = D.fit_fspl_clean(ev, D.conf_with(FAST))
    m = INJ.expected_map(ev2, f0, e)
    assert m["0.8"] == 0.0            # u_c(0.8) = 0.45 < u0
    assert m["0.3"] > 0.0             # u_c(0.3) = 3.03 > u0


def test_recovered_requires_candidate_and_matching_rho():
    ok = {"tier": D.TIER_CANDIDATE, "occult": {"rho_l": 0.61}}
    assert INJ.recovered(ok, 0.6)
    assert not INJ.recovered({"tier": D.TIER_REJECTED, "occult": {"rho_l": 0.6}}, 0.6)
    assert not INJ.recovered({"tier": D.TIER_CANDIDATE, "occult": {"rho_l": 1.2}}, 0.6)


# --------------------------------------------------------------------------------------
# stages degrade honestly
# --------------------------------------------------------------------------------------

def test_screen_without_catalogue_says_no_data(tmp_path):
    res = R.stage_screen(tmp_path, {}, 0, 2, 10.0)
    assert res["verdict"] == "NO_DATA_REACHED"


def test_screen_end_to_end_with_a_fake_archive(tmp_path, monkeypatch):
    """Catalogue -> screen (inline worker) -> assess, with the network replaced.

    One unit carries an injected opaque lens, one is plain FSPL, one returns
    no photometry at all (an unreachable archive for that event).
    """
    units = [{"unit": "KMT-2019-BLG-0001", "kmt": {"t0": 8000.0, "tE": 25.0, "u0": 0.2}, "ogle": None},
             {"unit": "KMT-2019-BLG-0002", "kmt": {"t0": 8000.0, "tE": 25.0, "u0": 0.2}, "ogle": None},
             {"unit": "KMT-2019-BLG-0003", "kmt": {"t0": 8000.0, "tE": 25.0, "u0": 0.2}, "ogle": None}]
    with gzip.open(tmp_path / "catalog.json.gz", "wt") as fh:
        json.dump({"units": units}, fh)

    def fake_fetch(s, unit, conf=None):
        if unit["unit"].endswith("3"):
            return {"series": [], "status": {"kmt": {"http": 404}}}
        ev = _ev(rho_l=0.6 if unit["unit"].endswith("1") else None, seed=21)
        series = []
        for k, d in enumerate(ev.datasets):
            m = ev.ds == k
            series.append((d, ev.t[m], ev.f[m], ev.e[m]))
        return {"series": series, "status": {"kmt": {"http": 200}}}

    monkeypatch.setattr(A, "fetch_unit_series", fake_fetch)
    monkeypatch.setattr(A, "session", lambda: None)
    cfg = {"detect": FAST, "screen": {"inject_every": 0}}
    res = R.stage_screen(tmp_path, cfg, 0, 1, 3600.0, workers=1)
    assert res["n_done"] == 3
    rows = [json.loads(x) for x in (tmp_path / "shards" / "screen_s0of1.jsonl").read_text().splitlines()]
    tiers = {r["unit"]: r["tier"] for r in rows}
    assert tiers == {"KMT-2019-BLG-0001": D.TIER_CANDIDATE, "KMT-2019-BLG-0002": D.TIER_NO_OCC,
                     "KMT-2019-BLG-0003": D.TIER_NO_DATA}
    # resume: a second call does no new work
    assert R.stage_screen(tmp_path, cfg, 0, 1, 3600.0, workers=1)["n_new"] == 0
    (tmp_path / "controls.json").write_text(json.dumps({"passed": True, "verdict": "CONTROLS_PASS"}))
    s = R.stage_assess(tmp_path, cfg, 1)
    assert s["funnel"]["units_with_photometry"] == 2
    assert s["self_consistency"]["ok"]


def test_assess_without_shards_says_no_data(tmp_path):
    res = R.stage_assess(tmp_path, {}, 2)
    assert res["verdict"] == "NO_DATA_REACHED"
    assert json.loads((tmp_path / "summary.json").read_text())["verdict"] == "NO_DATA_REACHED"


def test_controls_verdict_logic():
    gate = {"min_recovery_strong": 0.7, "min_controls_reached_frac": 0.7}
    assert R.controls_verdict([], gate, 3)["verdict"] == "NO_DATA_REACHED"
    good = [{"name": "a", "class": "finite_source_single_lens", "reached": True,
             "tier": D.TIER_NO_OCC,
             "injections": [{"expected_dchi2": 900, "recovered": True}] * 3}]
    assert R.controls_verdict(good, gate, 1)["verdict"] == "CONTROLS_PASS"
    fp = [dict(good[0], tier=D.TIER_CANDIDATE)]
    assert R.controls_verdict(fp, gate, 1)["verdict"] == "CONTROLS_FAIL_FALSE_POSITIVE"
    low = [dict(good[0], injections=[{"expected_dchi2": 900, "recovered": False}] * 3)]
    assert R.controls_verdict(low, gate, 1)["verdict"] == "CONTROLS_FAIL_LOW_RECOVERY"


def test_assess_threshold_from_anti_null_and_funnel(tmp_path):
    sdir = tmp_path / "shards"
    sdir.mkdir()
    rows = [{"unit": f"U{i}", "tier": D.TIER_NO_OCC, "dchi2": 5.0, "dchi2_anti": float(i)}
            for i in range(60)]
    rows.append({"unit": "C1", "tier": D.TIER_CANDIDATE, "dchi2": 400.0, "dchi2_anti": 3.0})
    rows.append({"unit": "C2", "tier": D.TIER_CANDIDATE, "dchi2": 55.0, "dchi2_anti": 1.0})
    (sdir / "screen_s0of1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (sdir / "inj_s0of1.jsonl").write_text("")
    (tmp_path / "controls.json").write_text(json.dumps({"passed": True, "verdict": "CONTROLS_PASS"}))
    with gzip.open(tmp_path / "catalog.json.gz", "wt") as fh:
        json.dump({"units": [{"unit": r["unit"]} for r in rows]}, fh)
    s = R.stage_assess(tmp_path, {}, 1)
    assert s["null_threshold"]["threshold"] == pytest.approx(59.0)
    assert s["funnel"]["gate_passing_above_null_threshold"] == 1
    assert s["funnel"]["gate_passing_below_null_threshold"] == 1
    assert s["verdict"] == "CANDIDATES_TO_VET:1"
    assert s["self_consistency"]["ok"]
    assert math.isfinite(s["null_threshold"]["q999"])

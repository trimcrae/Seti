"""Offline suite for ARC stage 2 --- is the flare on the target?

No network (``conftest.py`` raises on any socket).  Per ``docs/channel-brief.md``
§5 the suite:

* recovers the injected signal: a synthetic Kepler-like target pixel file
  with the flare on the TARGET gives a difference-image centroid on the
  target, a centroid shift consistent with zero and the outcome
  ``on_target``; the same flare on a NEIGHBOUR pixel gives the centroid on
  the neighbour, the predicted shift direction and ``on_neighbour``;
* degrades honestly: a missing pixel file, a missing light curve, an
  exhausted budget and a failed archive each leave the star
  ``centroid_untestable`` with the reason, never ``flare_on_target``;
* checks the flare finder (injected flares recovered, no false positives on
  noise), the Shibayama energy against a hand computation, the match by time
  window and by energy rank, the quarter amplitude (``Rvar = 2 sqrt(2) Sph``),
  the census arithmetic, the ``Begin`` role fix and the Santos ``Sph``
  rescaling in stage 1, and the parameter merge across a positions-only and
  a parameters-only table;
* runs the whole stage end to end through scripted archive callables and
  checks ``summary.json`` / ``stars.json`` / ``flares.csv`` and the stage-1
  re-assess picking the verdict up.
"""

from __future__ import annotations

import json
import math
import re
import time as _t

import numpy as np
import pandas as pd
import pytest

from seti.arc import acquire as acq
from seti.arc import flares as F
from seti.arc import pixels as P
from seti.arc import stage2 as S
from seti.arc.ceiling import R_SUN_CM, SIGMA_SB, spot_area
from seti.arc.run import build_star_context, load_arc_config, stage_assess

CAD = 29.4244 / 1440.0


# ---------------------------------------------------------------------------
# flare physics
# ---------------------------------------------------------------------------
def test_shibayama_energy_by_hand_and_band_fraction_matches_config():
    # one cadence, C' = 1e-3, Sun, T_fl = 9000 K
    ratio = F.band_integral(5772.0) / F.band_integral(9000.0)
    a = 1e-3 * math.pi * R_SUN_CM ** 2 * ratio
    e = SIGMA_SB * 9000.0 ** 4 * a * CAD * 86400.0
    r = F.flare_energy_shibayama([100.0], [1e-3], teff_k=5772.0, radius_rsun=1.0, cadence_days=CAD)
    assert math.isclose(r["e_bol_erg"], e, rel_tol=1e-9)
    assert 0.1 < ratio < 0.4                       # a 9000 K blackbody is much brighter in the band
    assert 1e33 < r["e_bol_erg"] < 1e34            # a 0.1 % half-hour flare on the Sun: 2e33 erg
    # the configured band fraction 0.35 (verify) is what the response integral gives
    assert abs(F.band_fraction(1.0, 5800.0) - 0.35) < 0.03
    assert F.flare_energy_band(10.0, teff_k=5800.0, radius_rsun=1.0) == pytest.approx(
        10.0 * F.band_luminosity(1.0, 5800.0))
    assert np.isnan(F.flare_energy_band(float("nan"), teff_k=5800.0, radius_rsun=1.0))


def _lightcurve(seed=1, flares=((5.0, 0.004), (17.3, 0.002)), noise=1e-4, n_days=30.0,
                prot=5.5, amp=0.002):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, n_days, CAD)
    f = 1.0 + amp * np.sin(2 * np.pi * t / prot) + rng.normal(0, noise, t.size)
    for tp, a in flares:
        dt = t - tp
        f = f * (1 + a * np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / 0.04), 0.0))
    return t, f


def test_flare_finder_recovers_injected_flares_and_is_quiet_on_noise():
    t, f = _lightcurve()
    fl, info = F.detect_flares(t, f)
    assert [round(x["t_peak"], 1) for x in fl] == [5.0, 17.3]
    assert fl[0]["amplitude"] == pytest.approx(0.004, rel=0.2)
    assert fl[0]["ed_s"] > fl[1]["ed_s"] > 0 and fl[0]["n_points"] >= 3
    assert fl[0]["t_rise_days"] <= CAD + 1e-9 and fl[0]["t_decay_days"] > 0
    assert info["sigma"] == pytest.approx(1e-4, rel=0.3)
    # a flagged cadence is not a flare
    q = np.zeros(t.size, dtype=int)
    q[np.argmin(np.abs(t - 5.0))] = 1
    fl2, _ = F.detect_flares(t, f, quality=q)
    assert all(abs(x["t_peak"] - 5.0) > 2 * CAD for x in fl2) or fl2[0]["n_points"] < fl[0]["n_points"]
    # pure noise: nothing
    t0, f0 = _lightcurve(seed=3, flares=())
    assert F.detect_flares(t0, f0)[0] == []
    # too short: nothing, and the info says so
    assert F.detect_flares(t[:10], f[:10])[1]["n_points"] == 10


def test_quarter_amplitude_range_over_std_is_two_root_two():
    t, f = _lightcurve(flares=())
    a = F.rotational_amplitude(t, f)
    assert a["rvar"] == pytest.approx(0.004, rel=0.05)
    assert a["rvar"] / a["sph"] == pytest.approx(2 * math.sqrt(2), rel=0.05)
    assert F.rotational_amplitude(t[:20], f[:20])["n_points"] == 20
    assert np.isnan(F.rotational_amplitude(t[:20], f[:20])["rvar"])


def test_match_by_time_window_then_by_energy_rank():
    det = [{"t_start": 4.99, "t_peak": 5.0, "t_end": 5.2, "ed_s": 30.0},
           {"t_start": 17.28, "t_peak": 17.3, "t_end": 17.4, "ed_s": 10.0},
           {"t_start": 22.0, "t_peak": 22.01, "t_end": 22.1, "ed_s": 3.0}]
    rows = [{"t_start": 5.0, "t_end": 5.15, "energy_erg": 1e34},
            {"t_start": float("nan"), "t_end": float("nan"), "energy_erg": 5e33},
            {"t_start": float("nan"), "t_end": float("nan"), "energy_erg": 2e33},
            {"t_start": 9.0, "t_end": 9.1, "energy_erg": 1e33}]
    m = F.match_flares(rows, det, cadence_days=CAD)
    assert m[0]["match_method"] == F.MATCH_TIME and m[0]["match"]["t_peak"] == 5.0
    assert m[1]["match_method"] == F.MATCH_RANK and m[1]["match"]["t_peak"] == 17.3
    assert m[2]["match_method"] == F.MATCH_RANK and m[2]["match"]["t_peak"] == 22.01
    assert m[3]["match_method"] == F.MATCH_NONE and m[3]["match"] is None


# ---------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------
SRC = [{"id": "T", "x": 4.0, "y": 4.0, "flux": 1e5, "gmag": 11.5},
       {"id": "N", "x": 6.0, "y": 4.5, "flux": 1e5 * 10 ** (-0.4 * 4.7), "gmag": 16.2}]


def _run_pixels(flare_source, amplitude, noise=3.0, drift=0.0, sources=SRC, seed=11):
    tpf = P.synth_flare_tpf(sources=sources, flare_source=flare_source, t_peak=100.0,
                            amplitude=amplitude, noise=noise, decay_days=0.04,
                            drift_px_per_day=drift, seed=seed)
    inm, bm, core = P.flare_cadence_masks(tpf["time"], 100.0 - 0.02, 100.0 + 0.1, cadence_days=CAD)
    sh = P.centroid_shift(tpf["time"], tpf["flux"], tpf["aperture"], inm, bm)
    diff = P.difference_image(tpf["time"], tpf["flux"], core, bm, err_cube=tpf["flux_err"])
    cen, summ = P.census([dict(s) for s in sources], target_index=0, aperture=tpf["aperture"],
                         prf_sigma_px=0.7, aperture_amplitude=sh["a"])
    anc = P.anchor_sources(cen, target_index=0, baseline_xy=(sh["x0"], sh["y0"]))
    att = P.attribute_flare(diff, anc, target_index=0, shift=sh)
    return tpf, sh, diff, cen, summ, anc, att


def test_pixel_gaussian_and_capture_fractions():
    g = P.pixel_gaussian((15, 15), 7.0, 7.0, 0.7)
    assert g.sum() == pytest.approx(1.0, abs=1e-6)          # the stamp holds the whole PRF
    ap = np.hypot(*np.mgrid[0:15, 0:15][::-1] - 7.0) <= 2.0
    c_in = P.aperture_capture(7.0, 7.0, ap, 0.7)
    c_out = P.aperture_capture(11.0, 7.0, ap, 0.7)
    assert c_in > 0.95 and c_out < 0.01
    assert P.circular_capture(0.0, 3.0, 0.7) == pytest.approx(1.0, abs=1e-3)
    assert P.circular_capture(6.0, 3.0, 0.7) < 1e-3
    assert np.isnan(P.aperture_capture(float("nan"), 1.0, ap, 0.7))


def test_flare_on_target_gives_zero_shift_and_on_target_outcome():
    tpf, sh, diff, cen, summ, anc, att = _run_pixels(0, 0.01)
    assert sh["ok"] and 0.002 < sh["a"] < 0.01        # the in-flare MEAN of a 1 % peak
    # the flux-weighted shift is consistent with the target's tiny prediction
    tgt = next(s for s in att["sources"] if s["is_target"])
    assert abs(sh["dx"] - tgt["pred_dx"]) < 3 * sh["dx_err"] + 1e-3
    assert abs(sh["dx"]) < 0.01 and abs(sh["dy"]) < 0.01
    assert diff["snr_peak"] > 10
    assert att["outcome"] == P.OUTCOME_ON_TARGET
    assert att["d_target_px"] < 0.1 and att["z_target"] < 3
    assert "N" not in att["consistent_neighbours"]
    assert summ["n_neighbours"] == 1 and cen[0]["is_target"] and cen[0]["flux_fraction"] > 0.98
    assert P.star_verdict([att["outcome"]]) == (P.VERDICT_ON_TARGET, "all 1 attributable flares on the target")


def test_flare_on_neighbour_moves_the_centroid_toward_it():
    tpf, sh, diff, cen, summ, anc, att = _run_pixels(1, 0.5)
    assert att["outcome"] == P.OUTCOME_ON_NEIGHBOUR and att["neighbour"] == "N"
    assert abs(att["x_diff"] - 6.0) < 0.15 and abs(att["y_diff"] - 4.5) < 0.15
    assert att["z_target"] > 3
    # the shift points toward the neighbour and matches the neighbour's prediction
    nb = next(s for s in att["sources"] if s["id"] == "N")
    assert sh["dx"] > 0 and nb["pred_dx"] > 0
    assert abs(sh["dx"] - nb["pred_dx"]) < 5 * sh["dx_err"] + 2e-3
    tg = next(s for s in att["sources"] if s["is_target"])
    assert nb["shift_chi2"] < tg["shift_chi2"]
    # the census says the neighbour would have to brighten by ~ a / f_N
    n = next(c for c in cen if c["id"] == "N")
    assert not n["excluded_by_arithmetic"] and n["required_amplitude"] == pytest.approx(
        sh["a"] / n["flux_fraction"], rel=1e-6)
    assert P.star_verdict([P.OUTCOME_ON_TARGET, att["outcome"]])[0] == P.VERDICT_ON_NEIGHBOUR


def test_pointing_drift_is_removed_by_the_baseline_trend():
    _, sh, _, _, _, _, att = _run_pixels(0, 0.01, drift=0.05)
    assert att["outcome"] == P.OUTCOME_ON_TARGET
    assert abs(sh["dx"]) < 0.01


def test_census_arithmetic_excludes_a_neighbour_that_cannot_supply_the_flare():
    faint = [dict(SRC[0]), {"id": "F", "x": 6.0, "y": 4.5, "flux": 1.0, "gmag": 21.0}]
    tpf, sh, diff, cen, summ, anc, att = _run_pixels(0, 0.01, sources=faint)
    f = next(c for c in cen if c["id"] == "F")
    assert f["excluded_by_arithmetic"] and f["required_amplitude"] > 20
    assert summ["n_neighbours_excluded"] == 1 and "excluded by arithmetic" in summ["statement"]
    assert att["excluded_neighbours"] == ["F"] and att["outcome"] == P.OUTCOME_ON_TARGET


def test_unresolved_neighbour_makes_the_flare_ambiguous():
    close = [dict(SRC[0]), {"id": "C", "x": 4.3, "y": 4.1, "flux": 3e4, "gmag": 12.8}]
    _, _, _, _, _, _, att = _run_pixels(0, 0.01, sources=close)
    assert att["outcome"] == P.OUTCOME_AMBIGUOUS and att["unresolved_neighbours"] == ["C"]
    assert P.star_verdict([att["outcome"]])[0] == P.VERDICT_AMBIGUOUS


def test_undetected_and_missing_difference_image_are_never_on_target():
    tpf = P.synth_flare_tpf(sources=SRC, flare_source=0, t_peak=100.0, amplitude=1e-5, noise=30.0)
    inm, bm, core = P.flare_cadence_masks(tpf["time"], 99.98, 100.1, cadence_days=CAD)
    diff = P.difference_image(tpf["time"], tpf["flux"], core, bm, err_cube=tpf["flux_err"])
    att = P.attribute_flare(diff, [dict(s, x_est=s["x"], y_est=s["y"]) for s in SRC], target_index=0)
    assert att["outcome"] == P.OUTCOME_UNDETECTED
    att2 = P.attribute_flare({"ok": False}, [], target_index=0)
    assert att2["outcome"] == P.OUTCOME_UNTESTABLE
    assert P.star_verdict([P.OUTCOME_UNDETECTED, P.OUTCOME_UNTESTABLE])[0] == P.VERDICT_UNTESTABLE
    assert P.star_verdict([])[0] == P.VERDICT_UNTESTABLE
    # too few baseline cadences: no difference image, no shift
    few = np.zeros(tpf["time"].size, dtype=bool)
    few[:3] = True
    assert not P.difference_image(tpf["time"], tpf["flux"], core, few)["ok"]
    assert not P.centroid_shift(tpf["time"], tpf["flux"], tpf["aperture"], core, few)["ok"]


def test_a_dead_pixel_does_not_veto_the_whole_cadence():
    """A NaN column in the stamp must not cost the star its centroid test.

    MEASURED (results/arc/stage2/flares.csv, the first stage-2 run): 11 of 30
    shortlisted stars came back "no difference image (too few in-flare or
    baseline cadences)" while the SAME flares carried 5-7 in-flare and 68-82
    baseline cadences in the aperture centroid, which masks per pixel.  The
    difference image required EVERY pixel of a cadence to be finite, so one
    permanently-NaN pixel -- routine in a Kepler postage stamp -- discarded
    every cadence there was.
    """
    tpf = P.synth_flare_tpf(sources=SRC, flare_source=0, t_peak=100.0, amplitude=0.02,
                            noise=3.0, decay_days=0.04, seed=11)
    inm, bm, core = P.flare_cadence_masks(tpf["time"], 99.98, 100.1, cadence_days=CAD)
    good = P.difference_image(tpf["time"], tpf["flux"], core, bm, err_cube=tpf["flux_err"])
    assert good["ok"] and good["n_in"] >= 1 and good["n_base"] >= 4

    holed = np.array(tpf["flux"], dtype=float)
    holed[:, 0, 0] = np.nan                       # a column outside the downloaded mask
    holed[:, -1, -1] = np.nan
    d = P.difference_image(tpf["time"], holed, core, bm, err_cube=tpf["flux_err"])
    assert d["ok"], d
    assert d["n_in"] == good["n_in"] and d["n_base"] == good["n_base"]
    assert d["n_pixels_used"] == holed[0].size - 2
    assert not np.isfinite(d["image"][0, 0]) and not np.isfinite(d["image"][-1, -1])
    # and the flare is still attributed to the target, at the same position
    sh = P.centroid_shift(tpf["time"], holed, tpf["aperture"], inm, bm)
    cen, _ = P.census([dict(s) for s in SRC], target_index=0, aperture=tpf["aperture"],
                      prf_sigma_px=0.7, aperture_amplitude=sh["a"])
    anc = P.anchor_sources(cen, target_index=0, baseline_xy=(sh["x0"], sh["y0"]))
    att = P.attribute_flare(d, anc, target_index=0, shift=sh)
    assert att["outcome"] == P.OUTCOME_ON_TARGET, att["reason"]

    # a stamp with NO usable pixel at all is still honestly untestable, and the
    # reason now names the count that actually failed
    dead = np.full_like(holed, np.nan)
    empty = P.difference_image(tpf["time"], dead, core, bm)
    assert not empty["ok"] and empty["n_pixels_used"] == 0
    att_empty = P.attribute_flare(empty, anc, target_index=0, shift=sh)
    assert att_empty["outcome"] == P.OUTCOME_UNTESTABLE
    assert "usable pixels 0" in att_empty["reason"]


# ---------------------------------------------------------------------------
# stage-1 fixes
# ---------------------------------------------------------------------------
def test_begin_resolves_as_the_start_time_role():
    roles = acq.resolve_arc_columns(["recno", "KIC", "Q", "Begin", "End", "logE"], "flares")
    assert roles["t_start"] == "Begin" and roles["t_end"] == "End"
    assert roles["energy"] == "logE" and roles["sector"] == "Q"
    assert acq.score_table(["recno", "KIC", "Q", "Begin", "End", "logE"], "flares")[0] > 10


def test_santos_sph_is_rescaled_to_a_range_and_recorded():
    fl = pd.DataFrame({"star_id": ["a", "b"], "energy": [1e34, 1e34]})
    mcq = pd.DataFrame({"star_id": ["a"], "rot_amplitude": [1000.0]})
    santos = pd.DataFrame({"star_id": ["a", "b"], "rot_amplitude": [1000.0, 500.0]})
    ctx = build_star_context(fl, [("mcquillan2014", mcq, "ppm"),
                                  ("santos2021", santos, "ppm", 2.828)], {}).set_index("star_id")
    # a: Santos 1000 ppm x 2.828 beats McQuillan's 1000 ppm
    assert ctx.loc["a", "amplitude_source"] == "santos2021"
    assert ctx.loc["a", "amplitude_frac"] == pytest.approx(2.828e-3)
    assert ctx.loc["a", "amplitude_scale"] == pytest.approx(2.828)
    assert ctx.loc["b", "amplitude_frac"] == pytest.approx(1.414e-3, rel=1e-3)
    # the ceiling moves by the scale^1.5
    a1 = float(spot_area(1e-3, 1.0, 5800.0))
    a2 = float(spot_area(2.828e-3, 1.0, 5800.0))
    assert math.log10((a2 / a1) ** 1.5) == pytest.approx(0.677, abs=0.01)
    # the config carries the scale for santos2021
    conf = load_arc_config()
    s = next(x for x in conf["star_catalogues"]["kepler"] if x["name"] == "santos2021")
    assert s["amplitude_scale"] == pytest.approx(2.828)
    assert conf["param_tables"]["kepler"][0] == "J/AJ/159/280/table2"


class _TAP:
    """A scripted VizieR: ``tables`` maps a table name to a frame."""

    def __init__(self, tables, fail=()):
        self.tables, self.fail, self.calls = tables, set(fail), []

    def __call__(self, adql):
        self.calls.append(adql)
        if "TAP_SCHEMA.columns" in adql:
            t = re.search(r"table_name = '([^']*)'", adql).group(1)
            if t in self.fail:
                raise RuntimeError("503")
            cols = list(self.tables[t].columns) if t in self.tables else []
            return pd.DataFrame({"column_name": cols})
        t = re.search(r'FROM "([^"]+)"', adql).group(1)
        if t in self.fail:
            raise RuntimeError("503")
        df = self.tables[t]
        sel = re.search(r"SELECT\s+(.*?)\s+FROM", adql, re.S).group(1)
        cols = [c.strip().strip('"') for c in sel.split(",")]
        out = df[cols].copy()
        w = re.search(r'WHERE "([^"]+)" IN \((.*)\)', adql)
        if w:
            ids = [s.strip().strip("'") for s in w.group(2).split(",")]
            out = out[out[w.group(1)].astype(str).isin(ids)]
        w = re.search(r'WHERE "([^"]+)" = \'?([^\']+)\'?$', adql)
        if w:
            out = out[out[w.group(1)].astype(str) == w.group(2)]
        return out.reset_index(drop=True)


def test_params_merge_across_a_parameters_only_and_a_positions_only_table():
    t2 = pd.DataFrame({"KIC": ["1", "2"], "Teff": [6000.0, 5500.0], "R*": [1.3, 0.9],
                       "logg": [4.2, 4.5], "Mass": [1.1, 0.9], "Evol": ["MS", "MS"]})
    t1 = pd.DataFrame({"KIC": ["1", "2", "3"], "_RA": [290.0, 291.0, 292.0],
                       "_DE": [44.0, 45.0, 46.0], "RUWE": [0.9, 1.2, 1.0],
                       "GaiaDR2": ["111", "222", "333"]})
    kic = pd.DataFrame({"KIC": ["3"], "RAJ2000": [292.0], "DEJ2000": [46.0], "Teff": [5000.0],
                        "R*": [0.8], "logg": [4.6]})
    tap = _TAP({"J/AJ/159/280/table2": t2, "J/AJ/159/280/table1": t1, "V/133/kic": kic})
    pos, rec = acq.fetch_star_params_by_id(["1", "2", "3"], "kepler", query_fn=tap,
                                           tables={"kepler": ["J/AJ/159/280/table2",
                                                              "J/AJ/159/280/table1",
                                                              "V/133/kic"]})
    pos = pos.set_index("star_id")
    assert pos.loc["1", "radius"] == 1.3 and pos.loc["1", "ra"] == 290.0
    assert pos.loc["1", "radius_source"] == "J/AJ/159/280/table2"
    assert pos.loc["1", "ra_source"] == "J/AJ/159/280/table1"
    assert pos.loc["3", "radius"] == 0.8 and pos.loc["3", "teff_source"] == "V/133/kic"
    assert pos.loc["1", "gaia_id"] == "111" and pos.loc["2", "ruwe"] == 1.2
    # the KIC was asked only about star 3 (the others were complete)
    kic_calls = [c for c in tap.calls if "V/133/kic" in c and "IN (" in c]
    assert len(kic_calls) == 1 and "IN (3)" in kic_calls[0]
    st = {r["table"]: r for r in rec}
    assert st["J/AJ/159/280/table2"]["has_positions"] is False
    assert st["J/AJ/159/280/table2"]["has_parameters"] is True
    # a failed table is recorded, not fatal
    tap2 = _TAP({"V/133/kic": kic}, fail={"J/AJ/159/280/table2"})
    pos2, rec2 = acq.fetch_star_params_by_id(["3"], "kepler", query_fn=tap2,
                                             tables={"kepler": ["J/AJ/159/280/table2", "V/133/kic"]})
    assert rec2[0]["status"] == "QUERY_FAILED" and len(pos2) == 1


# ---------------------------------------------------------------------------
# the stage, end to end through scripted archives
# ---------------------------------------------------------------------------
E_CON = 5.85e33
E_NOM = 1.13e33


def _entry(star_id="11507705", tier="interest", **kw):
    e = {"record_key": f"kepler_yang2019:kepler:{star_id}", "star_key": f"kepler:{star_id}",
         "star_id": star_id, "catalogue": "kepler_yang2019", "mission": "kepler", "tier": tier,
         "xi_conservative_max": 0.44, "xi_nominal_max": 1.16, "n_above_conservative": 2,
         "e_mag_conservative_erg": E_CON, "e_mag_nominal_erg": E_NOM,
         "amplitude_frac": 0.000116, "amplitude_source": "santos2021", "teff_k": 6367.0,
         "radius_rsun": 1.0, "logg": 4.22, "params_assumed": True, "centroid": "not_checked",
         "stage2_pulls": [{"t_peak": None, "e_flare_erg": 1.6e34, "quarter": None}],
         "context": {"gaia_source": "2129762445439045888", "gmag": 11.56}}
    e.update(kw)
    return e


def _world():
    """A scripted sky: two flares in quarter 5 on the target, one in quarter 9
    on the neighbour; Yang & Liu rows with Begin/End; Berger + FLAME + Gaia."""
    ra, dec = 290.0, 49.0
    # neighbour 6.7" east: dRA = 6.7"/cos(dec)
    nra = ra + 6.7 / 3600.0 / math.cos(math.radians(dec))
    yang = pd.DataFrame({"KIC": ["11507705"] * 4, "Q": [5, 5, 9, 9],
                         "Begin": [460.0, 500.0, 850.0, 880.0],
                         "End": [460.15, 500.12, 850.1, 880.1],
                         "logE": [34.2, 33.95, 34.0, 33.6]})
    t2 = pd.DataFrame({"KIC": ["11507705"], "Teff": [6300.0], "R*": [1.25], "logg": [4.25],
                       "Mass": [1.2], "Evol": ["MS"]})
    t1 = pd.DataFrame({"KIC": ["11507705"], "_RA": [ra], "_DE": [dec], "RUWE": [0.84],
                       "GaiaDR3": ["2129762445439045888"]})
    flame = pd.DataFrame({"Source": ["2129762445439045888"], "Rad-Flame": [1.24],
                          "Mass-Flame": [1.19], "Lum-Flame": [2.2], "Age-Flame": [2.5],
                          "Evol": [300]})
    tables = {"J/ApJS/241/29/table2": yang, "J/AJ/159/280/table2": t2,
              "J/AJ/159/280/table1": t1, "I/355/paramp": flame}
    gaia = pd.DataFrame({"RA_ICRS": [ra, nra], "DE_ICRS": [dec, dec], "Gmag": [11.56, 16.2],
                         "Source": ["2129762445439045888", "999"], "RUWE": [0.84, 1.1],
                         "Plx": [2.6, 0.5]})
    return {"ra": ra, "dec": dec, "nra": nra, "tables": tables, "gaia": gaia}


def _pix_fn_factory(ra0, dec0, x0=4.0, y0=4.0):
    def fn(ra, dec):
        dx = (ra - ra0) * math.cos(math.radians(dec0)) * 3600.0 / P.KEPLER_PIXEL_ARCSEC
        dy = (dec - dec0) * 3600.0 / P.KEPLER_PIXEL_ARCSEC
        return x0 + dx, y0 + dy
    return fn


def _make_tpf(world, segment, flares):
    """flares: list of (t_peak, amplitude, source_index)."""
    ra, dec = world["ra"], world["dec"]
    fn = _pix_fn_factory(ra, dec)
    nx, ny = fn(world["nra"], dec)
    sources = [{"x": 4.0, "y": 4.0, "flux": 1e5}, {"x": nx, "y": ny, "flux": 1e5 * 10 ** (-0.4 * 4.64)}]
    t0 = min(f[0] for f in flares) - 2.0
    t1 = max(f[0] for f in flares) + 2.0
    n = int(round((t1 - t0) / CAD)) + 1
    t = t0 + np.arange(n) * CAD
    cube = np.zeros((n, 9, 9))
    rng = np.random.default_rng(int(segment))
    for j, s in enumerate(sources):
        gain = np.ones(n)
        for tp, amp, src in flares:
            if src == j:
                dt = t - tp
                prof = np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / 0.04), 0.0)
                gain = gain * (1 + amp * prof)
        base = P.pixel_gaussian((9, 9), s["x"], s["y"], 0.7, s["flux"])
        cube += base[None] * gain[:, None, None]
    cube += rng.normal(0, 3.0, cube.shape)
    yy, xx = np.mgrid[0:9, 0:9]
    ap = np.hypot(xx - 4.0, yy - 4.0) <= 2.0
    return {"mission": "kepler", "segment": segment, "time": t, "flux": cube,
            "flux_err": np.full(cube.shape, 3.0), "quality": np.zeros(n, dtype=int),
            "aperture": ap, "aperture_source": "pipeline", "pix_fn": fn,
            "pixel_scale_arcsec": P.KEPLER_PIXEL_ARCSEC, "exptime_s": CAD * 86400,
            "author": "synthetic", "n_cadences": n}


def _make_lc(world, segment, flares, seed=5):
    tpf = _make_tpf(world, segment, flares)
    lc = S.lc_from_pixels(tpf)
    f, e = lc["flux_columns"]["aperture_sum"]
    lc["flux_columns"] = {"pdcsap_flux": (f, e)}
    return lc


def _scripted(world, *, tpf_missing_segments=(), lc_missing=False, tpf_fail=False):
    flares_by_seg = {5: [(460.05, 0.006, 0), (500.03, 0.004, 0)],
                     9: [(850.02, 0.9, 1), (880.02, 0.3, 1)]}

    def lc_fn(star_id, *, mission, segments, **kw):
        if lc_missing:
            return []
        return [_make_lc(world, s, flares_by_seg[s]) for s in segments if s in flares_by_seg]

    def tpf_fn(star_id, *, mission, segments, **kw):
        if tpf_fail:
            raise RuntimeError("MAST 503")
        return [_make_tpf(world, s, flares_by_seg[s]) for s in segments
                if s in flares_by_seg and s not in set(tpf_missing_segments)]

    def cone_fn(table, ra, dec, r):
        return world["gaia"]

    return _TAP(world["tables"]), cone_fn, lc_fn, tpf_fn


def _conf(tmp_path):
    conf = load_arc_config()
    conf["_arc_dir"] = str(tmp_path / "arc")
    conf["stage2"]["budget_s"] = 600.0
    return conf


def test_stage2_end_to_end_attributes_flares_and_recomputes_xi(tmp_path):
    world = _world()
    tap, cone, lc_fn, tpf_fn = _scripted(world)
    conf = _conf(tmp_path)
    params = S.Stage2Params.from_config(conf)
    (tmp_path / "arc").mkdir()
    (tmp_path / "arc" / "acquire.json").write_text(json.dumps(
        {"catalogues": {"kepler_yang2019": {"table": "J/ApJS/241/29/table2"}}}))
    out = tmp_path / "arc" / "stage2"
    entry = _entry()
    entry["e_mag_conservative_erg"] = 10 ** 33.9         # rows at 34.2, 33.95, 34.0 exceed
    s = S.stage2_run(conf, out, params=params, query_fn=tap, cone_fn=cone, lc_fn=lc_fn,
                     tpf_fn=tpf_fn, shortlist=[entry])
    stars = json.loads((out / "stars.json").read_text())["stars"]
    st = stars[0]
    # rows: Begin/End resolved, in BKJD, quarters from the table
    assert st["statuses"]["flare_rows"] == "OK" and st["n_catalogue_rows"] == 4
    assert st["flare_rows_info"]["roles"]["t_start"] == "Begin"
    assert st["flare_rows_info"]["time_system"] == "BKJD"
    assert st["segments"] == [5, 9] and st["n_flares_to_test"] == 3
    # parameters measured: Berger radius, FLAME beside it
    assert st["params"]["measured"] and st["params"]["radius_rsun"] == 1.25
    assert st["params"]["radius_source"] == "J/AJ/159/280/table2"
    assert st["params"]["flame_status"] == "OK" and st["params"]["flame"]["radius_flame"] == 1.24
    # census: one neighbour at 6.7" that could supply a small flare, in the 12" not the 4" radius
    assert st["census"]["aperture"]["summary"]["n_neighbours"] == 1
    one, ap_ = st["census"]["one_pixel"]["sources"][1], st["census"]["aperture"]["sources"][1]
    assert one["capture_fraction"] < 0.2 < ap_["capture_fraction"]
    assert one["required_amplitude"] > ap_["required_amplitude"]
    # flares: the two quarter-5 flares are on the target, the quarter-9 one on the neighbour
    by_t = {round(f["t_start"], 1): f for f in st["flares"]}
    assert by_t[460.0]["match_method"] == "time_overlap"
    assert by_t[460.0]["outcome"] == P.OUTCOME_ON_TARGET
    assert by_t[500.0]["outcome"] == P.OUTCOME_ON_TARGET
    assert by_t[850.0]["outcome"] == P.OUTCOME_ON_NEIGHBOUR and by_t[850.0]["neighbour"] == "999"
    assert by_t[850.0]["z_target"] > 3 and by_t[460.0]["z_target"] < 3
    assert abs(by_t[460.0]["wcs_anchor_dx_px"]) < 0.05
    assert by_t[460.0]["e_remeasured_bol_erg"] > 0 and by_t[460.0]["ed_s"] > 0
    assert st["verdict"] == P.VERDICT_ON_NEIGHBOUR
    assert st["n_flares_tested"] == 3
    # xi on measured parameters: the Sph is rescaled, the quarter amplitude measured
    x = st["xi"]
    assert x["params_measured"] and x["amplitude_catalogue_scale_applied"] == pytest.approx(2.828)
    assert x["amplitude_catalogue_as_range"] == pytest.approx(0.000116 * 2.828)
    assert np.isfinite(x["amplitude_quarter_rvar"]) and x["amplitude_used_source"] in (
        "catalogue_amplitude", "quarter_rvar")
    assert np.isfinite(x["xi_conservative_measured"])
    assert x["with_max_amplitude"]["n_independent"] == 4          # Begin times make them independent
    assert x["with_max_amplitude_remeasured_energy"]["assessable"]
    # summary
    assert s["verdict"] == S.VERDICT_S2_NEIGHBOUR
    assert s["verdict_counts"] == {P.VERDICT_ON_NEIGHBOUR: 1}
    assert s["n_params_measured"] == 1 and not s["budget_exhausted"]
    fl = pd.read_csv(out / "flares.csv")
    assert len(fl) == 3 and set(fl["outcome"]) == {P.OUTCOME_ON_TARGET, P.OUTCOME_ON_NEIGHBOUR}
    cs = pd.read_csv(out / "census.csv")
    assert len(cs) == 6 and cs["is_target"].sum() == 3
    assert (out / "acquisition_log.json").exists() and (out / "summary.json").exists()

    # a stage-1 re-assess picks the verdict up into the centroid column
    xi_rows = pd.DataFrame([{"record_key": entry["record_key"], "star_key": entry["star_key"],
                             "star_id": entry["star_id"], "catalogue": "kepler_yang2019",
                             "mission": "kepler", "assessable": True, "xi_conservative_max": 0.4,
                             "xi_nominal_max": 1.1, "n_above_conservative": 2, "n_independent": 4,
                             "n_flares": 4, "amplitude_frac": 3e-4, "radius_rsun": 1.25,
                             "teff_k": 6300.0, "logg": 4.25, "params_assumed": False,
                             "has_peak_times": True, "catalogue_flag": "", "flares_above": "[]",
                             "energies_json": "", "t_peaks_json": "", "amplitude_unit_guessed": False}])
    xi_rows.to_csv(tmp_path / "arc" / "xi_kepler_yang2019.csv", index=False)
    s1 = stage_assess(conf, tmp_path / "arc", offline=True)
    assert s1["stage2_verdicts_applied"] == {P.VERDICT_ON_NEIGHBOUR: 1}
    c = pd.read_csv(tmp_path / "arc" / "candidates.csv")
    assert c.iloc[0]["centroid"] == P.VERDICT_ON_NEIGHBOUR


def test_stage2_all_on_target_and_xi_above_is_the_pending_spectroscopy_verdict(tmp_path):
    world = _world()
    tap, cone, lc_fn, tpf_fn = _scripted(world)
    conf = _conf(tmp_path)
    out = tmp_path / "s2"
    # only quarter 5 exceeds: both flares on the target
    entry = _entry(e_mag_conservative_erg=10 ** 34.1, amplitude_frac=1e-5)
    s = S.stage2_run(conf, out, params=S.Stage2Params.from_config(conf), query_fn=tap,
                     cone_fn=cone, lc_fn=lc_fn, tpf_fn=tpf_fn, shortlist=[entry])
    st = json.loads((out / "stars.json").read_text())["stars"][0]
    assert st["verdict"] == P.VERDICT_ON_TARGET and st["segments"] == [5]
    assert st["xi"]["xi_status"] == "above_ceiling"
    assert s["verdict"] == S.VERDICT_S2_ON_TARGET
    assert "PENDING" in s["verdict"] and "unresolved" in s["note"]


def test_stage2_dissolves_when_measured_parameters_close_the_excess(tmp_path):
    world = _world()
    tap, cone, lc_fn, tpf_fn = _scripted(world, tpf_missing_segments=(5, 9))
    conf = _conf(tmp_path)
    out = tmp_path / "s2"
    # a huge catalogue amplitude: every flare far below the ceiling on measured R
    entry = _entry(e_mag_conservative_erg=10 ** 34.1, amplitude_frac=0.05,
                   amplitude_source="mcquillan2014")
    s = S.stage2_run(conf, out, params=S.Stage2Params.from_config(conf), query_fn=tap,
                     cone_fn=cone, lc_fn=lc_fn, tpf_fn=tpf_fn, shortlist=[entry])
    st = json.loads((out / "stars.json").read_text())["stars"][0]
    assert st["verdict"] == P.VERDICT_UNTESTABLE and "pixels:QUERY_RETURNED_ZERO_ROWS" in st["degraded"]
    assert st["xi"]["amplitude_catalogue_scale_applied"] == 1.0
    assert st["xi"]["xi_status"] == "below_ceiling"
    assert s["verdict"] == S.VERDICT_S2_DISSOLVED


def test_the_sph_range_scale_is_applied_once_whichever_stage_applied_it(tmp_path):
    """Stage 1 now scales a Santos Sph itself; stage 2 must not scale it again.

    Double-scaling raises the ceiling by 2.828^1.5 = 4.75 and pushes xi DOWN
    by 0.68 dex — it hides a candidate rather than inventing one, which is
    the direction that would never be noticed.  A record that carries the
    scale stage 1 applied is taken as done; only a record from before that
    fix (no amplitude_scale, no amplitude_scaled) is scaled here.
    """
    world = _world()
    conf = _conf(tmp_path)
    prm = S.Stage2Params.from_config(conf)

    def _xi(entry):
        tap, cone, lc_fn, tpf_fn = _scripted(world, tpf_missing_segments=(5, 9))
        out = tmp_path / f"s2_{abs(hash(json.dumps(entry, sort_keys=True, default=str)))}"
        S.stage2_run(conf, out, params=prm, query_fn=tap, cone_fn=cone, lc_fn=lc_fn,
                     tpf_fn=tpf_fn, shortlist=[entry])
        return json.loads((out / "stars.json").read_text())["stars"][0]["xi"]

    # an OLD record: no scale on it at all -> stage 2 applies 2.828 itself
    old = _xi(_entry(amplitude_frac=0.000116, amplitude_source="santos2021"))
    assert old["amplitude_catalogue_scale_applied"] == pytest.approx(prm.sph_to_range)
    assert old["amplitude_catalogue_as_range"] == pytest.approx(0.000116 * prm.sph_to_range)
    assert old["amplitude_scale_from_stage1"] is None

    # a NEW record: stage 1 scaled it and says so, twice over
    for extra in ({"amplitude_scale": 2.828}, {"amplitude_scaled": True},
                  {"amplitude_scale": 2.828, "amplitude_scaled": True}):
        new = _xi(_entry(amplitude_frac=0.000116 * prm.sph_to_range,
                         amplitude_source="santos2021", **extra))
        assert new["amplitude_catalogue_scale_applied"] == 1.0, extra
        assert new["amplitude_catalogue_as_range"] == pytest.approx(
            old["amplitude_catalogue_as_range"])
        # the same amplitude reaches the ceiling, so the same xi comes out
        assert new["xi_conservative_measured"] == pytest.approx(
            old["xi_conservative_measured"], abs=1e-9), extra


def test_missing_pixel_file_is_untestable_never_on_target(tmp_path):
    world = _world()
    conf = _conf(tmp_path)
    for kw, expect in (({"tpf_missing_segments": (5, 9)}, "pixels:QUERY_RETURNED_ZERO_ROWS"),
                       ({"tpf_fail": True}, "pixels:QUERY_FAILED")):
        tap, cone, lc_fn, tpf_fn = _scripted(world, **kw)
        out = tmp_path / f"s2_{expect}"
        entry = _entry(e_mag_conservative_erg=10 ** 33.9)
        s = S.stage2_run(conf, out, params=S.Stage2Params.from_config(conf), query_fn=tap,
                         cone_fn=cone, lc_fn=lc_fn, tpf_fn=tpf_fn, shortlist=[entry])
        st = json.loads((out / "stars.json").read_text())["stars"][0]
        assert st["verdict"] == P.VERDICT_UNTESTABLE and expect in st["degraded"]
        assert all(f["outcome"] == P.OUTCOME_UNTESTABLE for f in st["flares"])
        assert "no pixel file" in st["verdict_reason"]
        # the light curve side still ran: energies re-measured, amplitude measured
        assert st["flares"][0]["e_remeasured_bol_erg"] > 0
        assert np.isfinite(st["xi"]["amplitude_quarter_rvar"])
        assert s["verdict"] in (S.VERDICT_S2_UNTESTABLE, S.VERDICT_S2_DISSOLVED)
        assert s["verdict_counts"] == {P.VERDICT_UNTESTABLE: 1}


def test_missing_light_curve_falls_back_to_the_pixel_aperture_sum(tmp_path):
    world = _world()
    tap, cone, lc_fn, tpf_fn = _scripted(world, lc_missing=True)
    conf = _conf(tmp_path)
    out = tmp_path / "s2"
    entry = _entry(e_mag_conservative_erg=10 ** 34.1)
    S.stage2_run(conf, out, params=S.Stage2Params.from_config(conf), query_fn=tap, cone_fn=cone,
                 lc_fn=lc_fn, tpf_fn=tpf_fn, shortlist=[entry])
    st = json.loads((out / "stars.json").read_text())["stars"][0]
    assert "lightcurve:aperture_sum_of_pixels_used" in st["degraded"]
    assert st["verdict"] == P.VERDICT_ON_TARGET
    assert st["flares"][0]["flux_column"] == "aperture_sum"


def test_failed_archive_everywhere_is_no_data_reached(tmp_path):
    world = _world()
    conf = _conf(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("CONNECT tunnel failed, response 403")

    out = tmp_path / "s2"
    entry = _entry()
    s = S.stage2_run(conf, out, params=S.Stage2Params.from_config(conf), query_fn=boom,
                     cone_fn=boom, lc_fn=boom, tpf_fn=boom, shortlist=[entry])
    st = json.loads((out / "stars.json").read_text())["stars"][0]
    assert st["verdict"] == P.VERDICT_UNTESTABLE
    assert st["statuses"]["flare_rows"] == "QUERY_FAILED"
    assert st["statuses"]["gaia_census"] == "QUERY_FAILED"
    assert not st["params"]["measured"] and "stellar_params:not_measured" in st["degraded"]
    # the pull list kept the energies, so the flare list is not empty, and nothing is on target
    assert st["n_flares_to_test"] == 1 and st["flares"][0]["outcome"] == P.OUTCOME_UNTESTABLE
    assert s["verdict"] == S.VERDICT_S2_NO_DATA and "not a null result" in s["note"]
    assert s["acquisition"]["any_query_failed"]
    del world


def test_a_hung_archive_fetch_is_abandoned_inside_its_own_clock(tmp_path):
    """The stage budget is checked BETWEEN stars, so a fetch that never
    answers has to be bounded by itself.

    MEASURED (run 35738785437): the stage-2 loop sat inside one star's fetch
    from the moment its 9,000 s budget was spent until the 240-minute job cap,
    so the budget check it was meant to obey was never reached.
    """
    world = _world()
    tap, cone, lc_fn, tpf_fn = _scripted(world)
    conf = _conf(tmp_path)
    params = S.Stage2Params.from_config(conf)
    params.product_timeout_s = 0.3
    params.retries = 1
    calls = {"n": 0}

    def hung(*a, **k):
        calls["n"] += 1
        _t.sleep(30.0)                                    # never answers in time
        raise AssertionError("the hung fetch was waited out")

    out = tmp_path / "s2"
    t0 = _t.monotonic()
    s = S.stage2_run(conf, out, params=params, query_fn=tap, cone_fn=cone, lc_fn=hung,
                     tpf_fn=hung, shortlist=[_entry()])
    elapsed = _t.monotonic() - t0
    # two products (light curve, pixels) abandoned at 0.3 s each; waiting them
    # out would cost 2 x 30 s on its own, before anything else the star needs
    assert calls["n"] >= 2 and elapsed < 30.0
    st = json.loads((out / "stars.json").read_text())["stars"][0]
    assert st["verdict"] == P.VERDICT_UNTESTABLE
    assert st["statuses"]["lightcurve"] == "QUERY_FAILED"
    assert st["statuses"]["pixels"] == "QUERY_FAILED"
    # a timed-out fetch is a FAILED fetch, never "the archive holds nothing"
    assert "QUERY_RETURNED_ZERO_ROWS" not in json.dumps(st["statuses"])
    assert s["verdict"] != ""
    del world


def test_the_default_tap_and_cone_callables_are_wrapped_in_a_wall_clock(tmp_path, monkeypatch):
    """Stage 2's TAP path was reaching pyvo's unbounded run_async directly.

    MEASURED (run 35744902798): the stage sat in the run step for half an hour
    past its own 9,000 s budget, which is only checked BETWEEN stars, with the
    per-star checkpoint frozen at the last star that finished.
    """
    seen = {}

    def fake_query(base_fn=None, *, timeout_s):
        seen["query"] = timeout_s
        return lambda adql, **kw: pd.DataFrame()

    def fake_cone(base_fn=None, *, timeout_s):
        seen["cone"] = timeout_s
        return lambda table, ra, dec, r, **kw: pd.DataFrame()

    monkeypatch.setattr(acq, "timeout_query_fn", fake_query)
    monkeypatch.setattr(acq, "timeout_cone_fn", fake_cone)
    conf = _conf(tmp_path)
    params = S.Stage2Params.from_config(conf)
    params.budget_s = 1e-9                                # stop before any star
    S.stage2_run(conf, tmp_path / "s2", params=params, shortlist=[_entry()])
    assert seen == {"query": params.query_timeout_s, "cone": params.query_timeout_s}
    assert params.query_timeout_s > 0

    # an injected callable is left exactly as passed
    seen.clear()
    S.stage2_run(conf, tmp_path / "s2b", params=params, shortlist=[_entry()],
                 query_fn=lambda adql, **kw: pd.DataFrame(),
                 cone_fn=lambda *a, **kw: pd.DataFrame())
    assert seen == {}


def test_an_in_time_fetch_and_its_exception_pass_straight_through():
    def ok(star_id, **kw):
        return [{"star_id": star_id, "kw": kw}]

    assert S._bounded_fetch(ok, "9418692", timeout_s=30.0, mission="kepler")[0][
        "star_id"] == "9418692"
    assert S._bounded_fetch(ok, "9418692", timeout_s=None, mission="kepler")[0][
        "star_id"] == "9418692"

    def boom(star_id, **kw):
        raise RuntimeError("403 from the archive")

    with pytest.raises(RuntimeError, match="403"):
        S._bounded_fetch(boom, "9418692", timeout_s=30.0, mission="kepler")

    def hung(star_id, **kw):
        _t.sleep(30.0)

    with pytest.raises(S.ArcProductTimeout, match="no answer in"):
        S._bounded_fetch(hung, "9418692", timeout_s=0.2, mission="kepler")


def test_budget_exhausted_marks_the_rest_untestable(tmp_path):
    world = _world()
    tap, cone, lc_fn, tpf_fn = _scripted(world)
    conf = _conf(tmp_path)
    params = S.Stage2Params.from_config(conf)
    params.budget_s = 1e-9
    out = tmp_path / "s2"
    s = S.stage2_run(conf, out, params=params, query_fn=tap, cone_fn=cone, lc_fn=lc_fn,
                     tpf_fn=tpf_fn, shortlist=[_entry(), _entry(star_id="8487271", tier="watch")])
    assert s["budget_exhausted"] and s["verdict_counts"] == {P.VERDICT_UNTESTABLE: 2}
    st = json.loads((out / "stars.json").read_text())["stars"]
    assert all(x["verdict_reason"] == "budget_exhausted" for x in st)


def test_shortlist_orders_interest_before_watch_and_caps(tmp_path):
    d = {"candidates": [_entry(star_id="1", tier="interest", xi_conservative_max=0.1),
                        _entry(star_id="2", tier="interest", xi_conservative_max=0.4)],
         "watch": [_entry(star_id="3", tier="watch", xi_conservative_max=-0.1),
                   _entry(star_id="4", tier="watch", xi_conservative_max=-0.05, mission="tess")]}
    (tmp_path / "candidates.json").write_text(json.dumps(d))
    p = S.Stage2Params(max_stars=3)
    keys = [e["star_id"] for e in S.load_shortlist(tmp_path, params=p)]
    assert keys == ["2", "1", "4"]
    p2 = S.Stage2Params(missions=("kepler",))
    assert [e["star_id"] for e in S.load_shortlist(tmp_path, params=p2)] == ["2", "1", "3"]
    assert S.load_shortlist(tmp_path / "nowhere", params=p) == []


def test_a_hard_vetoed_ceiling_excess_star_is_tested_first_not_dropped(tmp_path):
    """companion_suspect is a suspicion; the pixels are what can settle it.

    MEASURED (run 35738218021): KIC 9418692 is the ONLY star above the
    conservative ceiling on measured parameters (xi = +0.462, 4 flares) and
    its first_veto is companion_suspect, so it sits in no tier and reached
    no stage-2 shortlist at all.
    """
    d = {"candidates": [_entry(star_id="1", tier="interest", xi_conservative_max=0.1)],
         "watch": [_entry(star_id="3", tier="watch", xi_conservative_max=-0.1)]}
    (tmp_path / "candidates.json").write_text(json.dumps(d))
    pd.DataFrame([
        # above the ceiling but hard-vetoed: belongs in stage 2, first
        {"star_key": "kepler:9418692", "star_id": "9418692", "mission": "kepler",
         "catalogue": "kepler_yang2019", "tier": "none", "first_veto": "companion_suspect",
         "xi_conservative_max": 0.462, "amplitude_frac": 2.01e-4,
         "amplitude_source": "santos2021", "amplitude_scale": 2.828},
        # above the ceiling but vetoed for a reason the pixels cannot touch
        {"star_key": "kepler:1", "star_id": "1", "mission": "kepler",
         "catalogue": "kepler_yang2019", "tier": "none", "first_veto": "evolved",
         "xi_conservative_max": 0.9, "amplitude_frac": 1e-3,
         "amplitude_source": "mcquillan2014", "amplitude_scale": 1.0},
        # below the ceiling and vetoed: not a ceiling excess, stays out
        {"star_key": "kepler:7", "star_id": "7", "mission": "kepler",
         "catalogue": "kepler_yang2019", "tier": "none", "first_veto": "companion_suspect",
         "xi_conservative_max": -2.0, "amplitude_frac": 1e-3,
         "amplitude_source": "mcquillan2014", "amplitude_scale": 1.0},
    ]).to_csv(tmp_path / "xi_table.csv", index=False)

    p = S.Stage2Params(max_stars=10)
    short = S.load_shortlist(tmp_path, params=p)
    assert [e["star_id"] for e in short] == ["9418692", "1", "3"]
    assert short[0]["tier"] == "vetoed_excess"
    assert short[0]["amplitude_scale"] == pytest.approx(2.828)
    # "1" keeps its interest row (deduped by star_key), it is not re-tiered
    assert short[1]["tier"] == "interest"

    off = S.Stage2Params(max_stars=10, include_vetoed_excess=False)
    assert [e["star_id"] for e in S.load_shortlist(tmp_path, params=off)] == ["1", "3"]

    # the config turns it on with the measurement that motivated it
    conf = load_arc_config()
    assert conf["stage2"]["include_vetoed_excess"] is True
    assert "companion_suspect" in conf["stage2"]["vetoed_first_vetoes"]
    assert S.Stage2Params.from_config(conf).include_vetoed_excess is True


def _on_target_star(**kw):
    s = {"star_key": "kepler:9418692", "tier": "vetoed_excess",
         "verdict": P.VERDICT_ON_TARGET, "verdict_reason": "all 5 on the target",
         "n_flares_tested": 5, "outcomes": {"on_target": 5}, "statuses": {}, "degraded": [],
         "params": {"teff_k": 5677.4, "radius_rsun": 1.089, "gaia_ruwe": 1.5562,
                    "measured": True},
         "xi": {"xi_conservative_stage1": 0.715, "xi_conservative_measured": 0.292,
                "xi_status": "above_ceiling", "params_measured": True,
                "with_max_amplitude_remeasured_energy": {"xi_conservative_max": 0.045}},
         "flares": [{"outcome": "on_target", "e_ratio_remeasured_over_catalogue": 0.472},
                    {"outcome": "on_target", "e_ratio_remeasured_over_catalogue": 0.566}]}
    s.update(kw)
    return s


def test_an_on_target_excess_with_an_unresolved_companion_is_not_reported_as_cleared():
    """The centroid clears only Gaia-RESOLVED neighbours.

    MEASURED (run 35744902798): KIC 9418692 came back flare_on_target with
    xi = +0.292 and Gaia RUWE = 1.5562, and the run verdict read
    CEILING_EXCESS_ON_TARGET_PENDING_SPECTROSCOPY — which sounds like a
    clearance. A companion inside ~0.1" is invisible to Gaia and to a 4"
    Kepler pixel alike, and an M dwarf there is the standard mundane reading
    of a superflare on a solar-type star.
    """
    ruwe = _on_target_star()
    hit, why = S.companion_unresolved(ruwe)
    assert hit and "RUWE" in why
    v, r = S._overall_verdict([ruwe])
    assert v == S.VERDICT_S2_ON_TARGET_COMPANION
    assert "cannot settle" in r and "kepler:9418692" in r

    # the stage-1 veto alone is enough, with no RUWE measured at all
    vetoed = _on_target_star(params={"measured": True}, first_veto="companion_suspect")
    assert S.companion_unresolved(vetoed)[0]
    assert S._overall_verdict([vetoed])[0] == S.VERDICT_S2_ON_TARGET_COMPANION

    # a clean star still gets the plain on-target verdict
    clean = _on_target_star(star_key="kepler:1", first_veto="below_ceiling",
                            params={"teff_k": 5700.0, "radius_rsun": 1.0, "gaia_ruwe": 1.02,
                                    "measured": True})
    assert not S.companion_unresolved(clean)[0]
    assert S._overall_verdict([clean])[0] == S.VERDICT_S2_ON_TARGET
    # and one clean star among flagged ones still reports as on target
    assert S._overall_verdict([ruwe, clean])[0] == S.VERDICT_S2_ON_TARGET


def test_a_cancelled_run_can_be_summarised_from_its_checkpoint(tmp_path):
    """A cancelled run writes stars.json after every star but never reaches
    the end of stage2_run, so summary.json stayed behind from an older run."""
    stars = [_on_target_star(),
             {"star_key": "kepler:2", "tier": "watch", "verdict": P.VERDICT_UNTESTABLE,
              "verdict_reason": "no light curve", "statuses": {}, "degraded": ["x"],
              "params": {}, "xi": {"xi_status": "not_recomputed"}, "flares": []}]
    out = tmp_path / "s2"
    out.mkdir()
    (out / "stars.json").write_text(json.dumps({"generated_utc": "2026-09-22T15:26:18Z",
                                                "stars": stars}))
    assert S.main(["--stage", "summarise", "--out-dir", str(out),
                   "--source-run", "run 35744902798"]) == 0
    s = json.loads((out / "summary.json").read_text())
    # the summary describes THESE stars, and says where it came from
    assert s["n_stars"] == 2 and s["verdict"] == S.VERDICT_S2_ON_TARGET_COMPANION
    assert "15:26:18" in s["summary_source"] and "35744902798" in s["summary_source"]
    assert s["verdict_counts"] == {P.VERDICT_ON_TARGET: 1, P.VERDICT_UNTESTABLE: 1}
    # the energy scale that xi rides on is reported, not assumed away
    e = s["flare_energy_scale"]
    assert e["n"] == 2 and e["remeasured_over_catalogue_median"] == pytest.approx(0.519)
    assert e["median_dex"] == pytest.approx(math.log10(0.519), abs=1e-6)
    # and both readings of xi are side by side
    row = s["stars"][0]
    assert row["xi_conservative_measured"] == pytest.approx(0.292)
    assert row["xi_conservative_remeasured_energy"] == pytest.approx(0.045)
    assert row["companion_unresolved"] and row["gaia_ruwe"] == pytest.approx(1.5562)
    # a star with no re-measurement writes null, not NaN (NaN is not valid JSON)
    assert s["stars"][1]["xi_conservative_remeasured_energy"] is None
    assert "NaN" not in (out / "summary.json").read_text()
    assert (out / "flares.csv").exists()


def test_a_named_star_is_tested_whatever_tier_it_ended_in(tmp_path):
    """A star can leave every tier the moment better parameters arrive.

    MEASURED (run 35741271294): KIC 8487271 went xi +0.104 -> -0.920 on
    Berger+2020's 1.301 Rsun and dropped out of `interest`, and with it out of
    every shortlist -- so the centroid test that had failed on it before the
    NaN-pixel fix could never be retried.
    """
    d = {"candidates": [], "watch": [_entry(star_id="3", tier="watch",
                                            xi_conservative_max=-0.1)]}
    (tmp_path / "candidates.json").write_text(json.dumps(d))
    pd.DataFrame([
        {"star_key": "kepler:8487271", "star_id": "8487271", "mission": "kepler",
         "catalogue": "kepler_yang2019", "tier": "none", "first_veto": "below_ceiling",
         "xi_conservative_max": -0.920, "amplitude_frac": 1.07e-3,
         "amplitude_source": "santos2021", "amplitude_scale": 2.828},
        # the same star in a second catalogue: the larger xi wins
        {"star_key": "kepler:8487271", "star_id": "8487271", "mission": "kepler",
         "catalogue": "kepler_shibayama2013", "tier": "none", "first_veto": "below_ceiling",
         "xi_conservative_max": -1.9, "amplitude_frac": 2e-3,
         "amplitude_source": "own", "amplitude_scale": 1.0},
        {"star_key": "kepler:3", "star_id": "3", "mission": "kepler",
         "catalogue": "kepler_yang2019", "tier": "watch", "first_veto": "below_ceiling",
         "xi_conservative_max": -0.1, "amplitude_frac": 1e-3,
         "amplitude_source": "mcquillan2014", "amplitude_scale": 1.0},
    ]).to_csv(tmp_path / "xi_table.csv", index=False)

    p = S.Stage2Params(max_stars=10, stars=("kepler:8487271",))
    short = S.load_shortlist(tmp_path, params=p)
    assert [e["star_id"] for e in short] == ["8487271", "3"]
    assert short[0]["tier"] == "named" and short[0]["catalogue"] == "kepler_yang2019"
    # a bare id works too
    assert S.load_shortlist(tmp_path, params=S.Stage2Params(
        max_stars=10, stars=("8487271",)))[0]["star_id"] == "8487271"
    # naming nothing changes nothing; naming an unknown star adds nothing
    assert [e["star_id"] for e in S.load_shortlist(
        tmp_path, params=S.Stage2Params(max_stars=10))] == ["3"]
    assert [e["star_id"] for e in S.load_shortlist(
        tmp_path, params=S.Stage2Params(max_stars=10, stars=("kepler:404",)))] == ["3"]
    assert S.named_rows(tmp_path / "nowhere", ["kepler:8487271"]) == []
    # the config carries no named star by default: this is an explicit request
    conf = load_arc_config()
    assert S.Stage2Params.from_config(conf).stars == ()


def test_probe_writes_the_route_and_shortlist(tmp_path):
    conf = _conf(tmp_path)
    (tmp_path / "arc").mkdir()
    (tmp_path / "arc" / "candidates.json").write_text(json.dumps({"candidates": [_entry()], "watch": []}))
    rep = S.stage2_probe(conf, tmp_path / "arc" / "stage2")
    assert rep["shortlist_size"] == 1 and "route_preferred" in rep["mast"]
    assert rep["mast"]["lightkurve"]["importable"] in (True, False)
    assert (tmp_path / "arc" / "stage2" / "probe.json").exists()


def test_flare_rows_time_system_and_energy_conversion():
    world = _world()
    tap = _TAP(world["tables"])
    rows, status, info = S.fetch_flare_rows("11507705", {"mission": "kepler",
                                                        "energy": {"kind": "bolometric",
                                                                   "log10": "auto"}},
                                            query_fn=tap, table="J/ApJS/241/29/table2")
    assert status == "OK" and len(rows) == 4
    assert rows[0]["energy_erg"] == pytest.approx(10 ** 34.2) and rows[0]["segment"] == 5
    assert rows[0]["t_start"] == 460.0 and rows[0]["t_peak"] == 460.0 and rows[0]["t_end"] == 460.15
    assert info["energy"]["log10_input"] is True
    rows0, status0, _ = S.fetch_flare_rows("1", {"mission": "kepler"}, query_fn=tap,
                                           table="J/ApJS/241/29/table2")
    assert status0 == "QUERY_RETURNED_ZERO_ROWS" and rows0 == []
    _, sf, _ = S.fetch_flare_rows("1", {"mission": "kepler"}, query_fn=_TAP({}, fail={"x"}),
                                  table="x")
    assert sf == "QUERY_FAILED"


def test_gaia_flame_roles_and_no_source():
    world = _world()
    tap = _TAP(world["tables"])
    fl, st = S.fetch_gaia_flame("2129762445439045888", query_fn=tap)
    assert st == "OK" and fl["radius_flame"] == 1.24 and fl["mass_flame"] == 1.19
    assert fl["roles"]["radius_flame"] == "Rad-Flame"
    assert S.fetch_gaia_flame(None, query_fn=tap)[1] == "QUERY_RETURNED_ZERO_ROWS"
    assert S.fetch_gaia_flame("5", query_fn=tap)[1] == "QUERY_RETURNED_ZERO_ROWS"

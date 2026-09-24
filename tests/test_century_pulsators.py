"""CENTURY's cessation population: all-sky catalogued PULSATORS (docs/century.md §6.4).

The first real target selection (run 35748748365) drew 24 field variables,
dominated by eclipsing binaries --- whose geometric period cannot cease.  These
tests pin the retargeting: pulsators by type, all sky, with a real catalogued
amplitude, bright enough for the plates; fade and scatter kept on the bright
field sample; and the sweep's two wall clocks inside the job timeout.  Offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seti.century.pulsators import (
    b_mean_estimate,
    filter_pulsators,
    normalise_catalogue,
    pulsator_class,
    vsx_adql,
    vsx_amplitude,
)
from seti.century.run import DEFAULTS, field_common_mode, stage_targets
from seti.century.vet import vet_row

ROOT = Path(__file__).resolve().parents[1]


def _conf():
    return json.loads(json.dumps(DEFAULTS))


def test_pulsator_class_admits_oscillators_and_refuses_orbits_and_doubt():
    assert pulsator_class("DCEP") == "cepheid" and pulsator_class("CWB") == "cepheid"
    assert pulsator_class("DCEP(B)") == "cepheid" and pulsator_class("DCEP+EA") == "cepheid"
    assert pulsator_class("RRAB") == pulsator_class("RRC") == pulsator_class("RRD") == "rr_lyrae"
    assert pulsator_class("HADS") == "delta_scuti" and pulsator_class("SXPHE") == "delta_scuti"
    assert pulsator_class("RVA") == "rv_tauri"
    assert pulsator_class("M") == "mira" and pulsator_class("SRA") == "semiregular"
    # Orbits cannot cease; doubt cannot carry a cessation; irregulars have no clock.
    for t in ("EA", "EB", "EW", "ELL", "EA/DCEP", "RS", "ROT", "BY", "RRAB:", "DCEP:", "MISC",
              "SRB", "SRC", "SR", "L", "LB", "UG", "", None, float("nan")):
        assert pulsator_class(t) == "", t


def test_vsx_amplitude_is_nan_unless_established_in_one_band():
    df = pd.DataFrame({"max": [10.0, 10.0, 10.0, 10.0, 10.0],
                       "min": [0.8, 11.0, 11.0, np.nan, 12.0],
                       "f_min": ["(", "", "", "", ""],
                       "n_max": ["V", "V", "V", "V", "V"],
                       "n_min": ["V", "V", "B", "", ""],
                       "l_min": ["", "", "", "", "<"]})
    amp = vsx_amplitude(df)
    assert amp[0] == pytest.approx(0.8)          # "(" : min IS the amplitude
    assert amp[1] == pytest.approx(1.0)          # same band: min - max
    assert np.isnan(amp[2])                      # different bands: not an amplitude
    assert np.isnan(amp[3])                      # missing: NaN, never a pass
    assert np.isnan(amp[4])                      # a fainter-than limit


def test_b_mean_estimate_moves_v_by_the_class_colour_and_refuses_red_bands():
    bm = b_mean_estimate(np.array([12.0, 12.0, 12.0, 12.0]), np.array([1.0, 1.0, 1.0, 1.0]),
                         np.array(["V", "B", "I", "p"]),
                         np.array(["rr_lyrae", "rr_lyrae", "rr_lyrae", "mira"]))
    assert bm[0] == pytest.approx(12.5 + 0.35)
    assert bm[1] == pytest.approx(12.5)
    assert np.isnan(bm[2])
    assert bm[3] == pytest.approx(12.5)


def _vsx_rows() -> pd.DataFrame:
    rows = [
        # name, V flag, ra, dec, type, max, n_max, f_min, min, n_min, period
        ("RR bright", 0, 10.0, 10.0, "RRAB", 10.0, "V", "", 11.0, "V", 0.55),
        ("RR dup", 0, 10.0 + 1.0 / 3600, 10.0, "RRAB", 10.1, "V", "", 11.1, "V", 0.55),
        ("RR faint", 0, 20.0, 10.0, "RRAB", 13.8, "V", "", 14.8, "V", 0.60),
        ("RR doubt", 0, 30.0, 10.0, "RRAB:", 10.0, "V", "", 11.0, "V", 0.50),
        ("RR noamp", 0, 40.0, 10.0, "RRC", 10.0, "V", "", np.nan, "", 0.30),
        ("RR smallamp", 0, 50.0, 10.0, "RRC", 10.0, "V", "(", 0.1, "V", 0.30),
        ("EB", 0, 60.0, 10.0, "EB", 9.0, "V", "", 9.8, "V", 1.2),
        ("Cep", 0, 70.0, 10.0, "DCEP", 8.5, "V", "", 9.4, "V", 5.4),
        ("Cep P out", 0, 80.0, 10.0, "DCEP", 8.5, "V", "", 9.4, "V", 150.0),
        ("Mira", 0, 90.0, 10.0, "M", 7.5, "V", "", 13.0, "V", 300.0),
        ("Mira red", 0, 100.0, 10.0, "M", 7.5, "I", "", 13.0, "I", 300.0),
        ("HADS", 0, 110.0, 10.0, "HADS", 11.0, "V", "(", 0.5, "V", 0.10),
        ("HADS short", 0, 120.0, 10.0, "HADS", 11.0, "V", "(", 0.5, "V", 0.04),
        ("suspect", 1, 130.0, 10.0, "RRAB", 10.0, "V", "", 11.0, "V", 0.5),
        ("too bright", 0, 140.0, 10.0, "DCEP", 4.0, "V", "", 4.8, "V", 7.0),
    ]
    cols = ["Name", "V", "RAJ2000", "DEJ2000", "Type", "max", "n_max", "f_min", "min", "n_min",
            "Period"]
    return pd.DataFrame(rows, columns=cols)


def test_filter_pulsators_keeps_only_bright_certain_pulsators_with_real_amplitudes():
    cat = normalise_catalogue(_vsx_rows(), "vsx")
    sel, f = filter_pulsators(cat)
    names = set(sel["name"])
    assert names == {"RR bright", "Cep", "Mira", "HADS"}, names
    assert f["rejected_not_a_pulsator_type"] == 2          # EB, RRAB:
    assert f["rejected_amplitude_not_established"] == 1   # the NaN amplitude is NOT a pass
    assert f["rejected_amplitude_below_plate_floor"] == 1
    assert f["rejected_period_outside_class_range"] == 2  # Cep 150 d, HADS 0.04 d
    assert f["rejected_band_not_convertible_to_B"] == 1   # I-band Mira
    assert f["rejected_too_faint_for_plates"] == 1
    assert f["rejected_too_bright_saturates"] == 1
    assert f["rejected_not_a_confirmed_variable"] == 1
    assert f["rejected_duplicate_position"] == 1          # 1" apart: measured once
    accounted = sum(v for k, v in f.items() if k.startswith("rejected_"))
    assert accounted + f["n_selected"] == f["n_catalogue_rows"]
    assert set(sel["pulsator_class"]) == {"rr_lyrae", "cepheid", "mira", "delta_scuti"}
    assert np.all(np.isfinite(sel["amp_cat"])) and np.all(sel["amp_cat"] >= 0.3)


def test_filter_pulsators_caps_round_robin_so_one_class_cannot_crowd_out_the_rest():
    rng = np.random.default_rng(3)
    n = 60
    rows = pd.DataFrame({
        "Name": [f"s{i}" for i in range(3 * n)], "V": 0,
        "RAJ2000": rng.uniform(0, 360, 3 * n), "DEJ2000": rng.uniform(-60, 60, 3 * n),
        "Type": ["M"] * n + ["RRAB"] * n + ["DCEP"] * n,
        "max": rng.uniform(8.0, 11.0, 3 * n), "n_max": "V", "f_min": "(", "min": 1.0,
        "n_min": "V", "Period": [300.0] * n + [0.5] * n + [5.0] * n})
    cat = normalise_catalogue(rows, "vsx")
    sel, _ = filter_pulsators(cat, max_total=30)
    assert sel["pulsator_class"].value_counts().to_dict() == {"mira": 10, "rr_lyrae": 10,
                                                              "cepheid": 10}
    # Within a class the brightest are the ones kept.
    rr_all = np.sort(cat.loc[cat["vtype"] == "RRAB", "mag_max"].to_numpy() + 0.5 + 0.35)
    rr_sel = np.sort(sel.loc[sel["pulsator_class"] == "rr_lyrae", "mag_cat"].to_numpy())
    np.testing.assert_allclose(rr_sel, rr_all[:10])


def test_vsx_adql_quotes_reserved_words_and_the_table():
    q = vsx_adql(max_mag_max=14.0, period_min=0.07, period_max=450.0)
    assert 'FROM "B/vsx/vsx"' in q and '"max" <= 14.000' in q and '"min"' in q
    assert "\"Type\" LIKE 'RR%'" in q and "\"Type\" LIKE 'M%'" in q


def test_stage_targets_puts_pulsators_all_sky(tmp_path):
    cat = normalise_catalogue(_vsx_rows(), "vsx")
    df = stage_targets(_conf(), tmp_path, fields=[], include_bright=False, max_variables=0,
                       pulsator_catalogue_fn=lambda: (cat, "vsx_tap"))
    assert set(df["kind"]) == {"pulsator"}
    assert set(df["field"]) == {"allsky_pulsators"}
    assert df["target_id"].tolist() == list(range(len(df)))
    s = json.loads((tmp_path / "targets_summary.json").read_text())
    assert s["n_pulsators"] == 4 and s["n_variables"] == 0
    assert s["n_pulsators_by_class"] == {"cepheid": 1, "delta_scuti": 1, "mira": 1,
                                         "rr_lyrae": 1}
    assert s["pulsators"]["catalogue_source"] == "vsx_tap"
    back = pd.read_csv(tmp_path / "targets.csv")
    assert "pulsator_class" in back.columns and back["pulsator_class"].notna().all()


def test_stage_targets_reports_an_unreached_catalogue_as_none_not_an_empty_sky(tmp_path):
    empty = normalise_catalogue(pd.DataFrame(), "none")
    df = stage_targets(_conf(), tmp_path, fields=[], include_bright=False, max_variables=0,
                       pulsator_catalogue_fn=lambda: (empty, "none"))
    assert len(df) == 0
    s = json.loads((tmp_path / "targets_summary.json").read_text())
    assert s["pulsators"]["catalogue_source"] == "none" and s["n_targets"] == 0


def test_lpv_cessation_is_a_note_but_lpv_fade_is_killed_and_an_eclipse_cannot_cease():
    base = {"cess_status": "cessation", "cess_n_post_informative": 3, "cess_n_pre_detected": 4,
            "cess_flags": "", "mag_cat": 11.0}
    r = vet_row(dict(base, vtype="M", pulsator_class="mira", period_cat=300.0))
    assert r["verdict"] == "survivor"
    assert "lpv_cessation_trace_to_period_evolution" in r["notes"]
    assert "not_periodic_class" not in r["notes"]
    r = vet_row({"fade_is_fade": True, "fade_n_years": 40, "vtype": "M",
                 "pulsator_class": "mira", "period_cat": 300.0})
    assert r["verdict"] == "killed:long_period_giant"
    assert vet_row(dict(base, vtype="EA", period_cat=2.0))["verdict"] == "killed:geometric_period"
    assert vet_row(dict(base, vtype="RRAB", period_cat=0.5))["verdict"] == "survivor"


def test_pulsators_do_not_enter_the_field_ensemble():
    ann = {"year": list(range(1900, 1950)), "med": [11.0] * 50}
    rows = [{"field": "kepler", "kind": "bright", "median_mag": 11.2, "annual": ann}
            for _ in range(8)]
    rows += [{"field": "allsky_pulsators", "kind": "pulsator", "median_mag": 11.2,
              "annual": ann} for _ in range(20)]
    assert set(field_common_mode(rows, min_stars=8, mag_bin=1.0)) == {"kepler"}


def test_sweep_budgets_fit_inside_the_job_timeout():
    """acquire + screen share ONE sweep job: their clocks, plus install, gate,
    the one star each clock always lets through and the upload, must fit
    timeout-minutes, or the runner kills the shard with nothing uploaded."""
    wf = yaml.safe_load((ROOT / ".github/workflows/century.yml").read_text())
    inputs = wf.get("on", wf.get(True))["workflow_dispatch"]["inputs"]
    acq = float(inputs["time_budget_s"]["default"])
    scr = float(inputs["screen_budget_s"]["default"])
    timeout_min = float(wf["jobs"]["sweep"]["timeout-minutes"])
    assert acq + scr <= 300 * 60
    assert timeout_min * 60 - (acq + scr) >= 45 * 60
    conf = yaml.safe_load((ROOT / "config/century.yaml").read_text())
    assert float(conf["acquire"]["time_budget_s"]) + float(conf["screen"]["time_budget_s"]) \
        <= timeout_min * 60 - 45 * 60
    assert DEFAULTS["acquire"]["time_budget_s"] + DEFAULTS["screen"]["time_budget_s"] \
        <= timeout_min * 60 - 45 * 60
    # The shard count comes from the real target count, not a guess.
    assert inputs["n_shards"]["default"] == "auto"
    assert re.search(r"math\.ceil\(n_t / per\)",
                     (ROOT / ".github/workflows/century.yml").read_text())

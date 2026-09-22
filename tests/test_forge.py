"""Offline test suite for FORGE --- hot exozodis as ~1500 K swarm candidates.

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite:

* recovers an injected signal: a Planck 1500 K excess with consistent K and N
  is a ``candidate`` and its temperature is recovered;
* returns a clean null on the confounder: a K-bright / N-faint nano-grain
  system is never a candidate (``nano_preferred`` or ``inconclusive``);
* a missing N band is ``N_UNTESTED``, never a candidate, however strong the
  K excess;
* a hot photosphere (H/K/N in the ratio of a 5000 K body) is
  ``COMPANION_TEMPERATURE``; a catalogued companion is ``KNOWN_COMPANION``;
  embedded values the archive did not confirm are ``UNVERIFIED_INPUT``;
* degrades honestly: a dead VizieR yields ``NO_DATA_REACHED`` and no
  candidate, with every embedded row ``table_not_reached``;
* the broadband leg flags an injected 10 % hot component, leaves a 1 % one
  alone (below its sensitivity, which it reports), and reads cold dust as
  ``BROADBAND_W3_TOO_BRIGHT``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.forge import acquire as acq
from seti.forge.physics import (
    BAND_WAVELENGTH_UM,
    BROADBAND_NORMAL,
    BROADBAND_SATURATED,
    BROADBAND_W3_BRIGHT,
    TIER_CANDIDATE,
    TIER_COMPANION_T,
    TIER_INTEREST,
    TIER_KNOWN_COMPANION,
    TIER_N_UNTESTED,
    TIER_NANO,
    TIER_NO_NIR,
    TIER_UNVERIFIED,
    Measurement,
    StarContext,
    assess_star,
    colour_shift,
    excess_fraction,
    f_k_for_shift,
    planck_extrapolation,
    variability,
)
from seti.forge.run import (
    VERDICT_CANDIDATES,
    VERDICT_NO_DATA,
    forge_run,
    load_forge_config,
)
from seti.forge.tables import (
    embedded_measurements,
    load_asset,
    load_targets,
    normalise_name,
    teff_from_sptype,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _meas(teff, t, f, *, bands=(("H", 0.25), ("K", 0.2), ("N", 0.4)), a=None, beta=0.0,
          verified=True, source="x"):
    out = []
    for b, s in bands:
        v = float(excess_fraction(BAND_WAVELENGTH_UM[b], t, f, teff, a_um=a, beta=beta))
        out.append(Measurement(b, BAND_WAVELENGTH_UM[b], v, s, instrument="inst" + b, epoch=b,
                               source=source, verified=verified))
    return out


CTX = StarContext("test", 6000.0, companion_assessed=True)


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def test_planck_extrapolation_matches_the_brief():
    # "a grey 1500 K body at 1 % K gives ~8 % at 10 um" (A star)
    assert 7.0 < planck_extrapolation(1.0, 9600.0, 1500.0, 10.5) < 9.0
    assert 6.0 < planck_extrapolation(1.0, 6000.0, 1500.0, 10.5) < 7.5
    # the emissivity family is K-bright / N-faint relative to grey
    grey = excess_fraction(10.5, 1500.0, 1.0, 6000.0)
    nano = excess_fraction(10.5, 1500.0, 1.0, 6000.0, a_um=0.1, beta=1.0)
    assert nano < 0.4 * grey


def test_injected_planck_swarm_is_recovered_as_candidate():
    r = assess_star(_meas(6000.0, 1500.0, 1.0), CTX)
    assert r["tier"] == TIER_CANDIDATE
    assert abs(r["grey"]["t_k"] - 1500.0) / 1500.0 < 0.15
    assert abs(r["grey"]["f_ref_pct"] - 1.0) < 0.2
    assert r["delta_chi2"] >= 9.0
    assert r["n_band"]["consistent_with_grey"]
    # the caveat is on the record: a cooler nano population is not excluded by K + N
    assert "delta_chi2_unrestricted" in r and "1000" in r["delta_chi2_by_nano_t_floor"]


def test_a_star_swarm_is_recovered_too():
    ctx = StarContext("A", 9600.0, companion_assessed=True)
    r = assess_star(_meas(9600.0, 1500.0, 1.0), ctx)
    assert r["tier"] == TIER_CANDIDATE
    assert r["n_band"]["separation_sigma"] > 3.0


@pytest.mark.parametrize("a_um,beta", [(0.01, 2.0), (0.1, 2.0), (0.05, 1.0)])
def test_nano_grain_system_is_never_a_candidate(a_um, beta):
    r = assess_star(_meas(6000.0, 1500.0, 1.0, a=a_um, beta=beta, bands=(("K", 0.2), ("N", 0.4))), CTX)
    assert r["tier"] != TIER_CANDIDATE
    assert r["delta_chi2"] < 0


def test_the_typical_real_hot_exozodi_is_nano_preferred():
    # 1 % in K, an N-band null at 0.3 +/- 0.4 %: the population's signature
    ms = [Measurement("K", 2.2, 1.0, 0.2, verified=True, source="a", epoch="1"),
          Measurement("N", 11.1, 0.3, 0.4, verified=True, source="b", epoch="2")]
    r = assess_star(ms, CTX)
    assert r["tier"] == TIER_NANO
    assert r["delta_chi2"] <= -9.0


def test_missing_n_band_is_untested_never_candidate():
    ms = _meas(6000.0, 1500.0, 3.0, bands=(("H", 0.2), ("K", 0.1)))   # a 30 sigma K excess
    r = assess_star(ms, CTX)
    assert r["tier"] == TIER_N_UNTESTED
    assert "grey" not in r
    assert r["planck_1500k_prediction_pct"]["N"] > 15


def test_no_nir_detection_is_no_nir_excess():
    ms = [Measurement("K", 2.2, 0.2, 0.25, verified=True), Measurement("N", 11.1, 5.0, 0.4, verified=True)]
    assert assess_star(ms, CTX)["tier"] == TIER_NO_NIR


def test_hot_photosphere_is_companion_temperature():
    # a 5000 K body at 3 % in K (Altair's regime): H ~ K, N ~ 1.2 K -- nothing
    # solid does that.  At 1 % the same shape is within 2 sigma of a 2000 K
    # large grain, so the gate only fires where it is decisive.
    ms = _meas(6000.0, 5000.0, 3.0, bands=(("H", 0.15), ("K", 0.15), ("N", 0.3)))
    r = assess_star(ms, CTX)
    assert r["tier"] == TIER_COMPANION_T
    assert r["grey_free"]["t_k"] > 1800.0
    weak = assess_star(_meas(6000.0, 5000.0, 1.0, bands=(("H", 0.15), ("K", 0.15), ("N", 0.3))), CTX)
    assert weak["tier"] != TIER_CANDIDATE


def test_known_companion_kills():
    ctx = StarContext("c", 6000.0, known_companion=True, companion_note="WDS 0.4 arcsec",
                      companion_assessed=True)
    r = assess_star(_meas(6000.0, 1500.0, 1.0), ctx)
    assert r["tier"] == TIER_KNOWN_COMPANION


def test_unverified_driving_values_never_promote():
    ms = _meas(6000.0, 1500.0, 1.0, verified=False)
    r = assess_star(ms, CTX)
    assert r["tier"] == TIER_UNVERIFIED
    # verifying only the K anchor is not enough: the N value drives the verdict
    ms[1].verified = True
    assert assess_star(ms, CTX)["tier"] == TIER_UNVERIFIED
    ms[2].verified = True
    assert assess_star(ms, CTX)["tier"] == TIER_CANDIDATE


def test_planck_consistent_but_insignificant_n_is_interest_not_candidate():
    ms = [Measurement("K", 2.2, 0.6, 0.15, verified=True, source="a", epoch="1"),
          Measurement("N", 11.1, 4.0, 1.6, verified=True, source="b", epoch="2")]
    r = assess_star(ms, CTX)
    assert r["tier"] in (TIER_INTEREST, "inconclusive")
    assert r["tier"] != TIER_CANDIDATE


def test_variability_term_flags_a_changing_excess():
    ms = [Measurement("K", 2.2, 1.0, 0.2, instrument="FLUOR", epoch="2008", verified=True),
          Measurement("K", 2.2, 2.6, 0.2, instrument="JouFLU", epoch="2016", verified=True)]
    v = variability(ms)
    assert v["variable"] and v["z_max"] > 3
    steady = [ms[0], Measurement("K", 2.2, 1.1, 0.2, instrument="JouFLU", epoch="2016", verified=True)]
    assert not variability(steady)["variable"]


def test_colour_shift_is_monotonic_and_sensitivity_is_worse_than_interferometry():
    s1, s10 = colour_shift(5000.0, 1.0), colour_shift(5000.0, 10.0)
    assert 0 < s1["d_k_w1"] < s10["d_k_w1"]
    assert 0 < s1["d_w1_w2"] < s10["d_w1_w2"]
    assert s1["d_w2_w3"] > s1["d_w1_w2"]
    # a 3 sigma K-W1 residual at a 0.03 mag locus scatter is a ~6 % K excess
    assert f_k_for_shift(5000.0, 0.09) > 3.0


# ---------------------------------------------------------------------------
# tables and identifiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [("HD 102647", "hd102647"), ("HD 102647", "HD 102647.0"),
                                 ("bet Leo", "beta Leo"), ("β Leo", "bet Leo"),
                                 ("* bet Leo", "bet Leo"), ("HIP 57632", "hip 057632")])
def test_normalise_name_unifies_spellings(a, b):
    assert normalise_name(a) == normalise_name(b)


def test_targets_asset_indexes_every_alias():
    tt = load_targets()
    assert tt.key_for("Vega") == "HD 172167"
    assert tt.key_for("alpha Lyr") == "HD 172167"
    assert tt.key_for("HIP 91262") == "HD 172167"
    assert tt.key_for("HD 172167") == "HD 172167"
    assert tt.key_for("HD 999999") is None
    assert teff_from_sptype("A0V") > 9000 > teff_from_sptype("G8V") > teff_from_sptype("K5V")
    assert 5000 < tt.teff("HD 10700") < 5600


def test_embedded_asset_values_are_unverified_and_membership_rows_produce_nothing():
    ex = load_asset("excess")
    assert (ex["verify"].isin(["unverified", "membership"])).all()
    ms = embedded_measurements(ex, load_targets())
    assert "HD 172167" in ms and not ms["HD 172167"][0].verified
    assert "HD 7788" not in ms                     # membership row, no value
    # every embedded numeric row cites its publication
    numeric = ex[ex["value_pct"].str.strip() != ""]
    assert numeric["source"].str.len().min() > 10


def test_resolve_excess_columns_handles_the_published_spellings():
    r = acq.resolve_excess_columns(["recno", "HD", "Name", "SpType", "fCSE", "e_fCSE", "chi"])
    assert r["hd"] == "HD" and r["excess"] == "fCSE" and r["excess_err"] == "e_fCSE"
    assert r["significance"] == "chi"
    r = acq.resolve_excess_columns(["HIP", "Star", "Excess", "e_Excess", "l_Excess"])
    assert r["excess"] == "Excess" and r["excess_err"] == "e_Excess" and r["limit_flag"] == "l_Excess"
    r = acq.resolve_excess_columns(["HD", "Null", "e_Null", "zodi"])
    assert r["excess"] == "Null" and r["excess_err"] == "e_Null" and r["zodi"] == "zodi"
    score, _, why = acq.score_excess_table(["HD", "zodi", "e_zodi"])
    assert score == 0 and "zodi" in why
    score, _, _ = acq.score_excess_table(["Name", "fCSE", "e_fCSE"])
    assert score > 10


# ---------------------------------------------------------------------------
# a fake archive for the end-to-end run
# ---------------------------------------------------------------------------
FAKE_TABLES = {
    "J/A+A/555/A104/table3": {
        "cols": ["recno", "HD", "Name", "SpType", "fCSE", "e_fCSE", "chi"],
        "rows": [
            (1, 172167, "Vega", "A0V", 1.26, 0.27, 4.7),
            (2, 10700, "tau Cet", "G8V", 0.98, 0.19, 5.2),
            (3, 177724, "zet Aql", "A0V", 1.50, 0.27, 5.6),       # embedded says 1.69: discrepant
            (4, 102647, "bet Leo", "A3V", 0.94, 0.26, 3.6),
            (5, 999001, "SWARM", "A0V", 1.5, 0.2, 7.5),
            (6, 999002, "NANO", "F5V", 1.2, 0.2, 6.0),
            (7, 999003, "COMP", "G0V", 1.0, 0.15, 6.7),
            (8, 999004, "WDSCOMP", "A0V", 1.5, 0.2, 7.5),
            (9, 22049, "eps Eri", "K2V", 0.2, 0.3, 0.7),
        ]},
    "J/A+A/570/A128/table2": {
        "cols": ["recno", "HD", "fCSE", "e_fCSE"],
        "rows": [(1, 999001, 0.55, 0.25), (2, 999003, 0.95, 0.15), (3, 999004, 0.55, 0.25),
                 (4, 7788, 1.4, 0.3)]},
    "J/AJ/159/177/table4": {
        "cols": ["recno", "HD", "Name", "Excess", "e_Excess"],
        "rows": [(1, 172167, "Vega", 0.4, 0.5), (2, 102647, "bet Leo", 1.7, 0.3),
                 (3, 999001, "SWARM", 12.0, 0.8), (4, 999002, "NANO", 0.3, 0.4),
                 (5, 999003, "COMP", 1.2, 0.3), (6, 999004, "WDSCOMP", 12.0, 0.8),
                 (7, 22049, "eps Eri", 0.9, 0.3)]},
}
FAKE_CATALOGUES = {"J/A+A/555/A104": ["J/A+A/555/A104/table3"],
                   "J/A+A/570/A128": ["J/A+A/570/A128/table2"],
                   "J/AJ/159/177": ["J/AJ/159/177/table4"]}


def fake_query(adql: str) -> pd.DataFrame:
    a = adql.strip()
    m = re.search(r"TAP_SCHEMA\.tables\s+WHERE\s+table_name\s+LIKE\s+'%([^%']+)%'", a)
    if m:
        pat = m.group(1)
        names = [t for c, ts in FAKE_CATALOGUES.items() if pat in c for t in ts]
        return pd.DataFrame({"table_name": [f'"{n}"' for n in names],
                             "description": ["fake"] * len(names)})
    if "TAP_SCHEMA.tables" in a and "description LIKE" in a:
        return pd.DataFrame(columns=["table_name", "description"])
    m = re.search(r"TAP_SCHEMA\.columns\s+WHERE\s+table_name\s*=\s*'([^']+)'", a)
    if m:
        t = m.group(1)
        return pd.DataFrame({"column_name": [f'"{c}"' for c in FAKE_TABLES[t]["cols"]]})
    m = re.search(r'COUNT\(\*\)\s+AS\s+n\s+FROM\s+"([^"]+)"', a)
    if m:
        return pd.DataFrame({"n": [len(FAKE_TABLES[m.group(1)]["rows"])]})
    m = re.search(r'SELECT\s+(?:TOP\s+\d+\s+)?(.+?)\s+FROM\s+"([^"]+)"', a, re.S)
    if m:
        t = m.group(2)
        cols = [c.strip().strip('"') for c in m.group(1).split(",")]
        df = pd.DataFrame(FAKE_TABLES[t]["rows"], columns=FAKE_TABLES[t]["cols"])
        return df[cols]
    raise RuntimeError(f"fake archive cannot answer {adql!r}")


def dead_query(adql: str) -> pd.DataFrame:
    raise RuntimeError("VizieR: 503 at every endpoint")


SPTYPES = {"HD 172167": "A0V", "HD 10700": "G8V", "HD 177724": "A0V", "HD 102647": "A3V",
           "HD 999001": "A0V", "HD 999002": "F5V", "HD 999003": "G0V", "HD 999004": "A0V",
           "HD 22049": "K2V", "HD 7788": "F6V", "HD 216956": "A4V"}


def fake_simbad(adql: str, url: str) -> pd.DataFrame:
    ids = re.findall(r"'([^']+)'", adql)
    rows = []
    for i, k in enumerate(ids):
        if k not in SPTYPES:
            continue
        rows.append({"asked": k, "main_id": k, "ra": 10.0 + i, "dec": -20.0 + i,
                     "sp_type": SPTYPES[k], "plx_value": 50.0})
    return pd.DataFrame(rows)


def fake_cone(table: str, ra: float, dec: float, radius: float) -> pd.DataFrame:
    if table.startswith("B/wds"):
        # HD 999004 sits at index 7 in the SPTYPES order -> ra 17, dec -13
        if abs(ra - 17.0) < 0.01 and abs(dec + 13.0) < 0.01:
            return pd.DataFrame({"Sep2": [0.4], "mag1": [4.0], "mag2": [7.0], "Comp": ["AB"]})
        return pd.DataFrame()
    if table.startswith("I/355"):
        return pd.DataFrame()
    if table.startswith("II/246"):
        return pd.DataFrame({"RAJ2000": [ra], "DEJ2000": [dec], "Kmag": [2.0], "e_Kmag": [0.2],
                             "Qflg": ["DDD"]})
    if table.startswith("II/328"):
        return pd.DataFrame({"RAJ2000": [ra], "DEJ2000": [dec], "W1mag": [2.0], "e_W1mag": [0.1],
                             "W2mag": [1.5], "e_W2mag": [0.1], "W3mag": [2.0], "e_W3mag": [0.02],
                             "ccf": ["0000"], "ex": [0]})
    return pd.DataFrame()


def _synthetic_population(seed=3, n=2500):
    rng = np.random.default_rng(seed)
    bp_rp = rng.uniform(0.5, 3.0, n)
    # a smooth photospheric locus in the WISE colours (mag), with scatter
    k_w1 = 0.02 + 0.05 * (bp_rp - 0.5) + rng.normal(0, 0.03, n)
    w1_w2 = -0.02 + 0.03 * (bp_rp - 0.5) ** 2 + rng.normal(0, 0.03, n)
    w2_w3 = 0.01 + 0.02 * (bp_rp - 0.5) + rng.normal(0, 0.08, n)
    ks = rng.uniform(5.0, 9.0, n)
    df = pd.DataFrame({"source_id": np.arange(n), "ra": rng.uniform(0, 360, n),
                       "dec": rng.uniform(-89, 89, n), "parallax": rng.uniform(34, 100, n),
                       "parallax_over_error": 50.0, "phot_g_mean_mag": ks + 1.5 + bp_rp,
                       "bp_rp": bp_rp, "ruwe": 1.0, "non_single_star": 0, "teff_gspphot": np.nan,
                       "ks": ks, "e_ks": 0.02, "w1": ks - k_w1, "e_w1": 0.02, "w2": np.nan, "e_w2": 0.02,
                       "w3": np.nan, "e_w3": 0.05, "wise_qual": "AAAA", "wise_cc": "0000", "wise_ext": 0,
                       "tmass_qual": "AAA", "kind": "normal"})
    df["w2"] = df["w1"] - w1_w2
    df["w3"] = df["w2"] - w2_w3
    # injections: a hot 1500 K component at 10 % (detectable), 1 % (not), and cold dust
    def inject(idx, f_k, t_k, kind):
        for i in idx:
            teff = float(np.interp(df.loc[i, "bp_rp"], [0.5, 1.0, 2.0, 3.0], [6600, 5400, 3900, 3200]))
            sh = colour_shift(teff, f_k, t_k)
            df.loc[i, "w1"] -= sh["d_k_w1"]
            df.loc[i, "w2"] = df.loc[i, "w1"] - (df.loc[i, "w1"] - df.loc[i, "w2"]) - sh["d_w1_w2"]
            df.loc[i, "w3"] = df.loc[i, "w2"] - (df.loc[i, "w2"] - df.loc[i, "w3"]) - sh["d_w2_w3"]
            df.loc[i, "kind"] = kind
    inject(range(0, 6), 10.0, 1500.0, "hot10")
    inject(range(6, 10), 1.0, 1500.0, "hot1")
    inject(range(10, 14), 10.0, 300.0, "cold")      # a 300 K body: W3 far too bright for its W1/W2
    return df


POP = _synthetic_population()


def fake_gaia(query: str) -> pd.DataFrame:
    q = query.strip()
    if q.upper().startswith("SELECT TOP 1 * FROM"):
        t = q.split()[-1]
        if "allwise" in t:
            return pd.DataFrame(columns=["designation", "w1mpro", "w1mpro_error", "w2mpro", "w2mpro_error",
                                         "w3mpro", "w3mpro_error", "w4mpro", "w4mpro_error", "ph_qual",
                                         "cc_flags", "ext_flag"])
        return pd.DataFrame(columns=["designation", "j_m", "h_m", "ks_m", "ks_msigcom", "ph_qual"])
    m = re.search(r"g\.dec >= ([-\d.]+) AND g\.dec < ([-\d.]+)", q)
    lo, hi = float(m.group(1)), float(m.group(2))
    if "tmass_psc_xsc_join" in q:
        return pd.DataFrame()      # the bridge chain matches nothing in this fake mirror
    sel = POP[(POP["dec"] >= lo) & (POP["dec"] < hi)]
    return sel.drop(columns=["kind"]).reset_index(drop=True)


def _conf():
    conf = load_forge_config()
    conf["probe"]["budget_s"] = 0
    return conf


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
def test_end_to_end_recovers_the_injected_swarm_and_verifies_the_asset(tmp_path):
    out = tmp_path / "forge"
    res = forge_run("all", out, _conf(), query_fn=fake_query, simbad_fn=fake_simbad,
                    cone_fn=fake_cone, gaia_fn=fake_gaia)
    probe = res["probe"]
    assert probe["tables"]["absil2013"]["status"] == acq.STATUS_OK
    assert probe["tables"]["absil2013"]["table"] == "J/A+A/555/A104/table3"
    assert probe["tables"]["absil2013"]["roles"]["excess"] == "fCSE"
    assert probe["tables"]["ertel2020"]["roles"]["excess"] == "Excess"
    assert probe["tables"]["nunez2017"]["status"] == acq.STATUS_ZERO     # not in the fake archive
    assert probe["n_tables_usable"] == 3
    a = res["acquire"]
    assert a["tables"]["absil2013"]["n_rows"] == 9
    assert a["tables"]["absil2013"]["n_added"] == 4        # the four fake stars joined the sample
    ver = a["verification"]
    assert ver["verified"] >= 2 and ver["discrepant"] == 1 and ver["membership"] == 8
    vdf = pd.read_csv(out / "forge_excess_verified.csv", dtype=str, keep_default_na=False)
    assert set(vdf.loc[vdf["key"] == "HD 177724", "verify"]) == {"discrepant"}
    assert set(vdf.loc[vdf["key"] == "HD 172167", "verify"]) == {"verified"}
    assert set(vdf.loc[vdf["key"] == "HD 216956", "verify"]) <= {"no_archive_table", "table_not_reached"}
    # the population pull fell back to the direct 2MASS chain and says so
    assert a["population"]["chain"] == "direct" and a["population"]["n_rows"] == len(POP)
    scr = res["screen"]
    tier = {s["key"]: s["tier"] for s in scr["stars"]}
    assert tier["HD 999001"] == TIER_CANDIDATE
    assert tier["HD 999002"] == TIER_NANO
    assert tier["HD 999003"] == TIER_COMPANION_T
    assert tier["HD 999004"] == TIER_KNOWN_COMPANION
    assert tier["HD 10700"] == TIER_N_UNTESTED
    assert tier["HD 102647"] == TIER_NANO           # 0.94 % K, 1.7 % N: K-bright / N-faint
    assert tier["HD 172167"] != TIER_CANDIDATE
    assert tier["HD 22049"] == TIER_NO_NIR
    # the zeta Aql row is discrepant: the archive value (1.50) is what the fit used
    zaq = next(s for s in scr["stars"] if s["key"] == "HD 177724")
    assert abs(zaq["anchor"]["value_pct"] - 1.50) < 1e-6 and zaq["anchor"]["origin"] == "archive"
    # the swarm's temperature came back
    sw = next(s for s in scr["stars"] if s["key"] == "HD 999001")
    assert 1300 < sw["grey"]["t_k"] < 1750
    assert sw["teff_k"] > 9000                     # A0V from the fake Simbad
    # broadband: the sample is saturated (bright stars), the population finds the 10 % injections
    bs = pd.read_csv(out / "broadband_sample.csv")
    assert (bs["class"] == BROADBAND_SATURATED).all()
    bp = json.loads((out / "broadband_population.json").read_text())
    assert bp["status"] == "OK"
    assert bp["n_hot_excess"] >= 4
    flagged = pd.read_csv(out / "broadband_population_flagged.csv")
    assert set(flagged["source_id"]) <= set(range(0, 6))
    assert bp["median_f_k_3sigma_pct"] > 3.0        # an order of magnitude worse than interferometry
    # cold dust reads as W3 too bright, not as a hot excess
    star_cls = dict(zip(POP["source_id"], POP["kind"], strict=True))
    classes = bp["classes"]
    assert classes.get(BROADBAND_W3_BRIGHT, 0) >= 3
    assert classes.get(BROADBAND_NORMAL, 0) > 2000
    assert all(star_cls[i] == "hot10" for i in flagged["source_id"])
    s = res["assess"]
    assert s["verdict"].startswith(VERDICT_CANDIDATES)
    assert s["funnel"]["n_candidates"] == 1 and s["funnel"]["n_N_UNTESTED"] >= 1
    cand = s["candidates"][0]
    assert cand["key"] == "HD 999001"
    assert any("companion" in x for x in cand["systematics_not_excluded"])
    assert (out / "summary.json").exists() and (out / "star_table.csv").exists()
    st = pd.read_csv(out / "star_table.csv")
    assert set(st.columns) >= {"key", "tier", "delta_chi2", "n_separation_sigma", "reason"}


def test_dead_archive_degrades_to_no_data_and_promotes_nothing(tmp_path):
    out = tmp_path / "forge"
    res = forge_run("all", out, _conf(), query_fn=dead_query, simbad_fn=fake_simbad,
                    cone_fn=fake_cone, gaia_fn=fake_gaia, skip_population=True,
                    skip_sample_phot=True)
    assert res["probe"]["n_tables_usable"] == 0
    a = res["acquire"]
    assert a["verification"].get("verified", 0) == 0
    assert a["verification"].get("table_not_reached", 0) >= 5
    s = res["assess"]
    assert s["verdict"].startswith(VERDICT_NO_DATA)
    assert "DEGRADED" in s["verdict"]
    assert s["funnel"]["n_candidates"] == 0
    tiers = {st["key"]: st["tier"] for st in res["screen"]["stars"]}
    # the embedded Fomalhaut rows (K + KIN N8) exist but are unverified: never a candidate
    assert tiers["HD 216956"] != TIER_CANDIDATE
    assert all(t != TIER_CANDIDATE for t in tiers.values())


def test_probe_budget_marks_untouched_tables_not_attempted(tmp_path):
    from seti.forge.run import stage_probe
    conf = _conf()
    conf["probe"]["budget_s"] = -1.0           # exhausted before the first table
    rec = stage_probe(tmp_path, conf, query_fn=fake_query, simbad_fn=fake_simbad, gaia_fn=fake_gaia)
    assert all(v["status"] == acq.STATUS_NOT_ATTEMPTED for v in rec["tables"].values())
    assert rec["endpoints"]["simbad_tap"]["status"] == acq.STATUS_OK


def test_verify_embedded_distinguishes_every_outcome():
    tt = load_targets()
    asset = pd.DataFrame([
        {"key": "HD 172167", "survey": "absil2013", "band": "K", "value_pct": "1.26", "err_pct": "0.27",
         "vizier": "J/A+A/555/A104"},
        {"key": "HD 10700", "survey": "absil2013", "band": "K", "value_pct": "0.98", "err_pct": "0.19",
         "vizier": "J/A+A/555/A104"},
        {"key": "HD 177724", "survey": "absil2013", "band": "K", "value_pct": "1.69", "err_pct": "0.27",
         "vizier": "J/A+A/555/A104"},
        {"key": "HD 216956", "survey": "absil2009", "band": "K", "value_pct": "0.88", "err_pct": "0.12",
         "vizier": ""},
        {"key": "HD 7788", "survey": "ertel2014", "band": "H", "value_pct": "", "err_pct": "",
         "vizier": "J/A+A/570/A128"},
        {"key": "HD 102647", "survey": "ertel2020", "band": "N", "value_pct": "1.7", "err_pct": "0.3",
         "vizier": "J/AJ/159/177"},
    ])
    archive = {"HD 172167": [Measurement("K", 2.2, 1.26, 0.27, source="absil2013")],
               "HD 177724": [Measurement("K", 2.2, 1.50, 0.27, source="absil2013")]}
    rows, counts = acq.verify_embedded(asset, archive, {"absil2013": acq.STATUS_OK,
                                                        "ertel2020": acq.STATUS_FAILED}, tt)
    got = {r["key"]: r["verify"] for r in rows}
    assert got == {"HD 172167": "verified", "HD 10700": "not_in_archive", "HD 177724": "discrepant",
                   "HD 216956": "no_archive_table", "HD 7788": "membership",
                   "HD 102647": "table_not_reached"}
    assert counts["verified"] == 1


def test_config_and_assets_are_consistent():
    conf = load_forge_config()
    assert conf["physics"]["nano_t_range_k"][0] >= conf["physics"]["swarm_t_range_k"][0]
    surveys = set(load_asset("excess")["survey"])
    for s in surveys:
        assert s in conf["tables"] or s in ("absil2009", "mennesson2013"), s
    assert (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "forge.yml").exists()
    assert math.isfinite(planck_extrapolation(1.0, 6000.0))

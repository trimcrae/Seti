"""Offline tests for CRADLE --- the shattered cradle (S52) and slag, not glass (S53).

No network.  A synthetic archive answers every ADQL the stages send.  The
decisive four: an injected 300 K excess at log(f/f_max) = 4 on a 3 Gyr star is
recovered as a candidate; the same excess on a Sco-Cen star is vetoed; a galaxy
blend is vetoed; an empty archive is NO_DATA_REACHED.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.cradle import acquire as acq
from seti.cradle import excess as E
from seti.cradle.ages import assess_ages
from seti.cradle.mineralogy import parse_ipac_table, score_spectrum, slag_verdict
from seti.cradle.run import STAGES, cradle_run, load_cradle_config, parse_shard
from seti.cradle.sample import (
    CONTROLS,
    DEFAULT_SAMPLE,
    JOINED_SHAPES,
    SHAPE_FULL,
    SHAPE_GAIA_ONLY,
    SHAPE_LEAN,
    build_query,
    healpix_units,
    parent_columns,
    select_parent,
    shard_units,
    unit_children,
    unit_range,
)
from seti.cradle.vet import DEFAULT_VET, apply_rules, classify
from seti.ignition.sample import GaiaQueryFailed
from seti.photometry import band_freq_hz, mag_to_flux_jy, planck_bnu
from seti.vigil.acquire import QueryResult

ZP = {"W1": 309.540, "W2": 171.787, "W3": 31.674, "W4": 8.363}
LOCUS = {"W1": 0.05, "W2": 0.03, "W3": 0.06, "W4": 0.05}          # true K_s - W_i
UNIT_K = 5                                                          # every synthetic star lives here
K = 4.740470446


def _pm_for(v_kms: float, d_pc: float) -> float:
    return v_kms * 1000.0 / (K * d_pc)


def _g_minus_ks(bp_rp: float) -> float:
    """A monotonic G - Ks for an FGK dwarf (solar: bp_rp 0.82 -> 1.58)."""
    return 0.23 + 1.65 * float(bp_rp)


def _ms_abs_g(bp_rp: float) -> float:
    """A main-sequence M_G that sits above the dwarf cut at every colour used."""
    return 2.8 + 4.3 * (float(bp_rp) - 0.6)


def _plx_for_dwarf(ks: float, bp_rp: float) -> float:
    """The parallax that puts a star of this K_s and colour on the main sequence."""
    g = float(ks) + _g_minus_ks(bp_rp)
    return float(10.0 ** ((_ms_abs_g(bp_rp) - g + 10.0) / 5.0))


def make_star(i: int, rng, *, ks: float = 6.5, bp_rp: float = 0.85, plx: float = 20.0,
              ra: float = 45.0, dec: float = 30.0, l: float = 160.0, b: float = -30.0,
              v_tan_kms: float = 8.0, rv=np.nan, disk=None, age=(4.0, 2.0, 7.0),
              ext_flag: int = 0, cc_flags: str = "0000", noise: float = 0.02, **over) -> dict:
    lo, _hi = unit_range(3, UNIT_K)
    d = 1000.0 / plx
    pm = _pm_for(v_tan_kms, d)
    row = {"source_id": lo + 1000 + i, "ra": ra, "dec": dec, "l": l, "b": b, "parallax": plx,
           "parallax_error": plx / 100.0, "parallax_over_error": 100.0, "pmra": pm * 0.8,
           "pmra_error": 0.03, "pmdec": pm * 0.6, "pmdec_error": 0.03, "radial_velocity": rv,
           "radial_velocity_error": 1.0, "phot_g_mean_mag": ks + _g_minus_ks(bp_rp),
           "phot_bp_mean_mag": ks + _g_minus_ks(bp_rp) + 0.5 * bp_rp,
           "phot_rp_mean_mag": ks + _g_minus_ks(bp_rp) - 0.5 * bp_rp,
           "bp_rp": bp_rp, "ruwe": 1.0, "astrometric_excess_noise": 0.1,
           "ipd_frac_multi_peak": 0, "phot_variable_flag": "NOT_AVAILABLE", "non_single_star": 0,
           "teff_gspphot": 5600.0, "logg_gspphot": 4.4, "mh_gspphot": 0.0, "ag_gspphot": 0.02,
           "random_index": i, "age_flame": age[0], "age_flame_lower": age[1], "age_flame_upper": age[2],
           "mass_flame": 1.0, "mass_flame_lower": 0.95, "mass_flame_upper": 1.05, "lum_flame": np.nan,
           "lum_flame_lower": np.nan, "lum_flame_upper": np.nan, "radius_flame": 1.0,
           "evolstage_flame": 300, "flags_flame": "0", "teff_gspspec": np.nan, "logg_gspspec": np.nan,
           "mh_gspspec": np.nan, "alphafe_gspspec": np.nan, "alphafe_gspspec_lower": np.nan,
           "alphafe_gspspec_upper": np.nan, "flags_gspspec": "", "activityindex_espcs": np.nan,
           "activityindex_espcs_uncertainty": np.nan, "spectraltype_esphs": "G",
           "allwise_designation": f"J{i:010d}", "allwise_ra": ra, "allwise_dec": dec,
           "cc_flags": cc_flags, "ext_flag": ext_flag, "var_flag": "0000", "ph_qual": "AAAA",
           "allwise_sep_arcsec": 0.2, "allwise_n_neighbours": 1, "allwise_n_mates": 0,
           "tmass_designation": f"T{i:010d}", "j_m": ks + 0.5, "j_msigcom": 0.02, "h_m": ks + 0.1,
           "h_msigcom": 0.02, "ks_m": ks, "ks_msigcom": 0.02, "tmass_ph_qual": "AAA",
           "tmass_sep_arcsec": 0.1, "tmass_n_neighbours": 1, "gaia_rot_period_d": np.nan}
    abs_g = row["phot_g_mean_mag"] + 5 * np.log10(plx) - 10
    # the absolute magnitude must satisfy the dwarf cut; M_G for a G dwarf at these K_s is fine
    assert abs_g > 2.0 + 3.3 * (bp_rp - 0.6)
    mags = {bnd: ks - LOCUS[bnd] + rng.normal(0, noise) for bnd in LOCUS}
    if disk is not None:
        t_k, f = disk
        lum = 10 ** (-0.4 * (abs_g - row["ag_gspphot"] + float(E.bc_g(np.array([5600.0]))[0]) - 4.74))
        fb = E.fbol_w_m2(lum, plx)
        om = np.pi * f * fb / (E.SIGMA_SB * t_k ** 4)
        for bnd in ("W2", "W3", "W4"):
            phot = mag_to_flux_jy(mags[bnd], bnd)
            ex = om * planck_bnu(t_k, band_freq_hz(bnd)) * 1e26
            mags[bnd] = -2.5 * np.log10((phot + ex) / ZP[bnd])
    for bnd in LOCUS:
        lb = bnd.lower()
        row[f"{lb}mpro"] = mags[bnd]
        row[f"{lb}mpro_error"] = 0.02 if bnd in ("W1", "W2") else (0.03 if bnd == "W3" else 0.08)
    row.update(over)
    return row


def make_archive(rng, n_clean: int = 700, extra: list[dict] | None = None) -> pd.DataFrame:
    rows = []
    for i in range(n_clean):
        ks = rng.uniform(4.8, 7.2)
        bp_rp = rng.uniform(0.5, 1.45)
        rows.append(make_star(i, rng, ks=ks, bp_rp=bp_rp, plx=_plx_for_dwarf(ks, bp_rp),
                              ra=rng.uniform(40, 50), dec=rng.uniform(25, 35),
                              v_tan_kms=rng.uniform(3, 40)))
    for r in (extra or []):
        rows.append(r)
    df = pd.DataFrame(rows)
    df["source_id"] = df["source_id"].astype(np.int64)
    return df


class FakeArchive:
    """Answers the ADQL of every stage from one DataFrame.  Counts what it was asked."""

    def __init__(self, df: pd.DataFrame, *, fail_units: set | None = None, timeout_units: set | None = None):
        self.df = df
        self.calls: list[str] = []
        self.fail_units = fail_units or set()
        self.timeout_units = timeout_units or set()

    def _range(self, adql):
        m = re.search(r"g\.source_id >= (\d+) AND g\.source_id < (\d+)", adql)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _cone(self, adql):
        m = re.search(r"CIRCLE\('ICRS', ([-\d.]+), ([-\d.]+), ([-\d.e]+)\)", adql)
        return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None

    def gaia(self, adql: str):
        self.calls.append(adql)
        if "TOP 1 *" in adql:
            return pd.DataFrame([{c: 0 for c in ("designation", "w4mpro_error", "ext_flag",
                                                  "original_ext_source_id", "best_rotation_period",
                                                  "age_flame")}]), {"transport": "fake"}
        if "TAP_UPLOAD" in adql:
            raise GaiaQueryFailed({"status": "QUERY_FAILED", "error": "no upload here"})
        sel = self.df
        rng_ = self._range(adql)
        if rng_ is not None:
            lo, hi = rng_
            for u in self.timeout_units:
                ulo, uhi = unit_range(u[0], u[1])
                if lo == ulo and hi == uhi:
                    raise GaiaQueryFailed({"status": "TIMED_OUT", "error": "no answer within 1 s"})
            for u in self.fail_units:
                ulo, uhi = unit_range(u[0], u[1])
                if lo == ulo and hi == uhi:
                    raise GaiaQueryFailed({"status": "QUERY_FAILED", "error": "boom"})
            sel = sel[(sel["source_id"] >= lo) & (sel["source_id"] < hi)]
        cone = self._cone(adql)
        if cone is not None:
            ra0, dec0, rad = cone
            sep = np.hypot((sel["ra"] - ra0) * np.cos(np.radians(dec0)), sel["dec"] - dec0)
            sel = sel[sep <= rad]
        if "nb_source_id" in adql:            # neighbour cone
            ra0, dec0, _ = cone
            out = pd.DataFrame({f"nb_{c}": sel[c] for c in acq.NEIGHBOUR_COLS})
            out["sep_arcsec"] = np.hypot((sel["ra"] - ra0) * np.cos(np.radians(dec0)), sel["dec"] - dec0) * 3600
            return out, {}
        if "gaiadr1.allwise_original_valid" in adql and "w4mpro_error" in adql.split("WHERE")[-1]:
            sel = sel[(sel["w3mpro_error"] < 0.2172) & (sel["w4mpro_error"] < 0.2172)]
        if "COUNT(*)" in adql:
            return pd.DataFrame({"n": [len(sel)]}), {}
        if "FROM gaiadr3.gaia_source AS g" in adql and "allwise" not in adql:
            gcols = [c for c in sel.columns if c in ("source_id", "ra", "dec", "l", "b", "parallax",
                                                     "parallax_error", "parallax_over_error", "pmra",
                                                     "pmra_error", "pmdec", "pmdec_error", "radial_velocity",
                                                     "radial_velocity_error", "phot_g_mean_mag", "bp_rp",
                                                     "ruwe", "phot_variable_flag", "non_single_star",
                                                     "teff_gspphot", "random_index")]
            sel = sel[gcols]
        elif SHAPE_LEAN in adql or "tmass" not in adql:
            sel = sel.drop(columns=[c for c in sel.columns if c.startswith(("j_", "h_", "ks_", "tmass"))
                                    or c == "gaia_rot_period_d"], errors="ignore")
        m = re.search(r"SELECT TOP (\d+)", adql)
        if m:
            sel = sel.head(int(m.group(1)))
        return sel.reset_index(drop=True), {"transport": "fake"}

    def irsa(self, adql: str):
        self.calls.append("IRSA:" + adql[:80])
        if "TAP_SCHEMA" in adql:
            if "irs_enhv211" in adql:
                return pd.DataFrame({"column_name": ["reqkey", "tn", "object", "ra", "dec", "irs16", "irs22"]}), {}
            return pd.DataFrame({"column_name": list(acq.IRSA_ALLWISE_EXTRA)}), {}
        if "COUNT(*)" in adql:
            return pd.DataFrame({"n": [3]}), {}
        if "irs_enhv211" in adql:
            return pd.DataFrame([{"reqkey": 12345, "tn": 1, "object": "SYNTH", "ra": 45.0, "dec": 30.0,
                                  "irs16": 0.1, "irs22": 0.2}]), {}
        if "designation IN" in adql:
            des = re.findall(r"'(J\d+)'", adql)
            return pd.DataFrame({"designation": des, "w3rchi2": 1.1, "w4rchi2": 1.0, "nb": 1, "na": 0,
                                 "w3snr": 20.0, "w4snr": 10.0, "ext_flg": 0, "cc_flags": "0000"}), {}
        return pd.DataFrame(), {}


def _neowise(ra, dec, pmra, pmdec, radius_arcsec=2.5, **_k):
    rng = np.random.default_rng(int(abs(ra * 1000)) % 2 ** 31)
    n = 120
    mjd = np.sort(np.concatenate([57000 + 183 * k + rng.uniform(0, 2, 10) for k in range(12)]))
    df = pd.DataFrame({"ra": ra, "dec": dec, "mjd": mjd, "w1mpro": 8.0 + rng.normal(0, 0.01, n),
                       "w1sigmpro": 0.01, "w2mpro": 8.0 + rng.normal(0, 0.01, n), "w2sigmpro": 0.01,
                       "qual_frame": 10, "saa_sep": 30.0, "moon_masked": "00", "cc_flags": "0000",
                       "ph_qual": "AA", "nb": 1, "na": 0})
    return QueryResult(label="fake", service="fake", status="OK", n_rows=n, data=df)


def _backends(arch: FakeArchive, **over) -> acq.Backends:
    kw = dict(gaia=arch.gaia, irsa=arch.irsa, simbad=lambda ra, dec: "", simbad_resolve=lambda name: None,
              ebv=lambda pos: pd.DataFrame({"source_id": pos["source_id"].astype(str), "ebv_sfd": 0.02}),
              neowise=_neowise, asu=lambda url: (_ for _ in ()).throw(RuntimeError("no vizier")),
              http=lambda url: (404, ""))
    kw.update(over)
    return acq.Backends(**kw)


def _conf(tmp_path: Path) -> dict:
    conf = load_cradle_config(Path("config/cradle.yaml"))
    conf["excess"]["mc_draws"] = 60
    conf["sample"]["count_parent"] = True
    return conf


def _run_all(tmp_path, arch, conf, n_shards=1):
    b = _backends(arch)
    out = tmp_path / "cradle"
    for i in range(n_shards):
        cradle_run("acquire", out_dir=out, shard=i, n_shards=n_shards, conf=conf, backends=b)
    cradle_run("screen", out_dir=out, conf=conf, backends=b)
    for i in range(n_shards):
        cradle_run("ages", out_dir=out, shard=i, n_shards=n_shards, conf=conf, backends=b)
    return cradle_run("assess", out_dir=out, n_shards=n_shards, conf=conf, backends=b), out


# ---------------------------------------------------------------------------
# sample / ADQL
# ---------------------------------------------------------------------------
def test_healpix_units_tile_source_id_space_without_gaps():
    units = healpix_units(3)
    assert len(units) == 768
    assert units[0]["sid_lo"] == 0
    for a, b in zip(units[:-1], units[1:], strict=False):
        assert a["sid_hi"] == b["sid_lo"]
    assert units[-1]["sid_hi"] == 12 * 4 ** 12 * 2 ** 35
    kids = unit_children(units[7])
    assert kids[0]["sid_lo"] == units[7]["sid_lo"] and kids[-1]["sid_hi"] == units[7]["sid_hi"]
    assert sum(k["sid_hi"] - k["sid_lo"] for k in kids) == units[7]["sid_hi"] - units[7]["sid_lo"]
    sh = [shard_units(units, i, 8) for i in range(8)]
    assert sum(len(s) for s in sh) == 768 and len({u["k"] for s in sh for u in s}) == 768


def test_build_query_carries_every_cut_and_the_shapes_differ_only_in_joins():
    unit = healpix_units(3)[3]
    q = build_query(DEFAULT_SAMPLE, unit=unit, shape=SHAPE_FULL)
    assert f"g.source_id >= {unit['sid_lo']}" in q.replace("gs.", "g.")
    for frag in ("phot_g_mean_mag < 13.5", "parallax > 2.0", "ABS(gs.b) > 10.0", "ruwe < 1.4",
                 "bp_rp > 0.45", "5 * LOG10(gs.parallax) - 10 >", "w3mpro_error < 0.21715",
                 "w4mpro_error < 0.21715", "allwise_best_neighbour", "astrophysical_parameters",
                 "tmass_psc_xsc_best_neighbour", "tmass_original_valid", "vari_rotation_modulation",
                 "age_flame_lower", "alphafe_gspspec", "activityindex_espcs"):
        assert frag in q, frag
    lean = build_query(DEFAULT_SAMPLE, unit=unit, shape=SHAPE_LEAN)
    assert "tmass" not in lean and "vari_rotation" not in lean and "astrophysical_parameters" in lean
    go = build_query(DEFAULT_SAMPLE, unit=unit, shape=SHAPE_GAIA_ONLY)
    assert "allwise" not in go and "JOIN" not in go
    cnt = build_query(DEFAULT_SAMPLE, unit=unit, shape=SHAPE_GAIA_ONLY, count_only=True)
    assert cnt.startswith("SELECT COUNT(*)") and "TOP" not in cnt
    ctrl = build_query(DEFAULT_SAMPLE, cone={"ra": 28.7, "dec": 21.3, "radius_deg": 0.001},
                       shape=SHAPE_FULL, controls=True)
    assert "CIRCLE" in ctrl and "w4mpro_error <" not in ctrl
    with pytest.raises(ValueError):
        build_query(DEFAULT_SAMPLE, unit=unit, shape="nope")
    cols = parent_columns()
    assert "ks_m" in cols and "w4mpro_error" in cols and "gaia_rot_period_d" in cols


def test_select_parent_reapplies_the_cuts_and_exempts_controls_from_the_wise_cut():
    rng = np.random.default_rng(3)
    good = make_star(1, rng)
    giant = make_star(2, rng, phot_g_mean_mag=2.0)      # far above the dwarf line
    faint_w4 = make_star(3, rng, w4mpro_error=0.5)
    ctrl = make_star(4, rng, w4mpro_error=0.5, is_control=True)
    df = pd.DataFrame([good, giant, faint_w4, ctrl])
    kept, counters = select_parent(df, DEFAULT_SAMPLE)
    assert counters["cut_dwarf"] == 1 and counters["cut_w3_w4_5sigma"] == 1
    assert set(kept["source_id"].astype(int)) == {good["source_id"], ctrl["source_id"]}
    assert counters["n_controls"] == 1


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def test_wyatt_fmax_matches_the_catalogue_form():
    assert E.wyatt_fmax(1.0, 1000.0, 1.0, 1.0) == pytest.approx(1.6e-7)
    # r^{7/3}: a 0.5 AU ring is 5.04x lower; L^{-1/2}: a 4 L_sun star halves it
    assert E.wyatt_fmax(0.5, 1000.0) / E.wyatt_fmax(1.0, 1000.0) == pytest.approx(0.5 ** (7 / 3))
    assert E.wyatt_fmax(1.0, 1000.0, 1.0, 4.0) / E.wyatt_fmax(1.0, 1000.0) == pytest.approx(0.5)
    assert E.blackbody_radius_au(278.3, 1.0) == pytest.approx(1.0)
    assert E.blackbody_radius_au(300.0, 1.0) == pytest.approx((278.3 / 300) ** 2)
    assert abs(float(E.bc_g(np.array([5772.0]))[0]) - 0.06) < 1e-6


def test_injected_300k_excess_at_4dex_is_recovered_and_placed_in_the_cell():
    rng = np.random.default_rng(11)
    f_inj = 10 ** 4.0 * E.wyatt_fmax((278.3 / 300) ** 2 * np.sqrt(0.9), 1000.0, 1.0, 0.9)
    star = make_star(999, rng, ks=6.5, bp_rp=0.85, disk=(300.0, f_inj), noise=0.0)
    arch = make_archive(rng, extra=[star])
    parent, _ = select_parent(arch, DEFAULT_SAMPLE)
    d = E.harmonise(parent)
    loci = E.fit_loci(d, E.DEFAULT_EXCESS)
    assert all(loc.n_bins >= 8 for loc in loci.values())
    fit = E.fit_disk(E.excess_table(d, loci, E.DEFAULT_EXCESS), {**E.DEFAULT_EXCESS, "mc_draws": 80})
    r = fit[fit["source_id"].astype(int) == star["source_id"]].iloc[0]
    assert r["excess_significant"] and r["chi_W3"] > 5 and r["chi_W4"] > 5
    assert 270 < r["t_bb_k"] < 330 and r["in_t_cell"]
    assert 3.6 < r["log_f_fmax_1gyr"] < 4.4 and r["above_fmax_3dex_16pct"]
    assert r["t_bb_lo_k"] < r["t_bb_k"] < r["t_bb_hi_k"]
    # the clean stars: no significant excess manufactured by the locus
    clean = fit[fit["source_id"].astype(int) != star["source_id"]]
    assert clean["excess_significant"].mean() < 0.02


def test_kinematic_indicator_certifies_old_by_excess_velocity_only():
    rng = np.random.default_rng(5)
    slow = make_star(1, rng, v_tan_kms=5.0, age=(np.nan, np.nan, np.nan))
    fast = make_star(2, rng, v_tan_kms=70.0, age=(np.nan, np.nan, np.nan))
    df = pd.DataFrame([slow, fast])
    out = assess_ages(df, load_cradle_config(Path("config/cradle.yaml"))["ages"])
    assert list(out["age_kin_verdict"]) == ["UNDETERMINED", "OLD"]
    assert list(out["age_class"]) == ["AGE_UNDETERMINED", "AGE_UNDETERMINED"]   # one indicator only
    assert out["n_old_indicators"].tolist() == [0, 1]


def test_two_independent_indicators_make_a_mature_star_and_one_does_not():
    rng = np.random.default_rng(6)
    conf = load_cradle_config(Path("config/cradle.yaml"))["ages"]
    both = make_star(1, rng, v_tan_kms=70.0, age=(3.0, 2.0, 5.0))
    iso_only = make_star(2, rng, v_tan_kms=5.0, age=(3.0, 2.0, 5.0))
    alpha_iso = make_star(3, rng, v_tan_kms=5.0, age=(3.0, 2.0, 5.0), alphafe_gspspec=0.25,
                          alphafe_gspspec_lower=0.2, alphafe_gspspec_upper=0.3, mh_gspspec=-0.5,
                          flags_gspspec="0000000000000")
    out = assess_ages(pd.DataFrame([both, iso_only, alpha_iso]), conf)
    assert list(out["age_class"]) == ["MATURE_2PLUS", "AGE_UNDETERMINED", "MATURE_2PLUS"]
    assert out["old_indicators"].tolist()[2] == "iso+alpha"


def test_young_group_rotation_and_activity_are_vetoes():
    rng = np.random.default_rng(7)
    conf = load_cradle_config(Path("config/cradle.yaml"))["ages"]
    rot = make_star(1, rng, v_tan_kms=70.0, age=(3.0, 2.0, 5.0), gaia_rot_period_d=2.0)
    act = make_star(2, rng, v_tan_kms=70.0, age=(3.0, 2.0, 5.0), activityindex_espcs=0.8)
    # AB Dor member at the group centroid, UVW of the group (with RV so the test is 3-D)
    abdor = make_star(3, rng, age=(3.0, 2.0, 5.0))
    out = assess_ages(pd.DataFrame([rot, act]), conf)
    assert list(out["age_class"]) == ["YOUNG_VETO", "YOUNG_VETO"]
    assert out["age_gyro_verdict"].iloc[0] == "YOUNG" and out["age_act_verdict"].iloc[1] == "YOUNG"
    # place a star inside AB Dor by construction: X,Y,Z from the group centre
    from seti.cradle.ages import galactic_xyz
    from seti.ossuary.kinematics import _A_ICRS_TO_GAL
    g = next(x for x in conf["young_groups"] if x["name"] == "AB_Dor")
    xyz = np.asarray(g["xyz_pc"], float)
    d = np.linalg.norm(xyz)
    icrs = _A_ICRS_TO_GAL.T @ (xyz / d)
    ra = float(np.degrees(np.arctan2(icrs[1], icrs[0])) % 360)
    dec = float(np.degrees(np.arcsin(icrs[2])))
    abdor.update(ra=ra, dec=dec, parallax=1000.0 / d, radial_velocity=0.0)
    # tangential velocity = projection of the group UVW onto the sky, radial = the rest
    from seti.ossuary.kinematics import _triad
    r_hat, a_hat, d_hat = _triad(np.array([ra]), np.array([dec]))
    uvw = np.asarray(g["uvw_kms"], float)
    v_r = float((_A_ICRS_TO_GAL @ r_hat)[:, 0] @ uvw)
    v_a = float((_A_ICRS_TO_GAL @ a_hat)[:, 0] @ uvw)
    v_d = float((_A_ICRS_TO_GAL @ d_hat)[:, 0] @ uvw)
    abdor.update(radial_velocity=v_r, pmra=v_a * 1000 / (K * d), pmdec=v_d * 1000 / (K * d))
    out = assess_ages(pd.DataFrame([abdor]), conf)
    assert out["young_group"].iloc[0] == "AB_Dor" and out["age_class"].iloc[0] == "YOUNG_VETO"
    assert np.allclose(galactic_xyz(np.array([ra]), np.array([dec]), np.array([d]))[:, 0], xyz, atol=0.5)


# ---------------------------------------------------------------------------
# vet
# ---------------------------------------------------------------------------
def test_every_kill_rule_trips_on_its_own_case_and_has_a_counter():
    rng = np.random.default_rng(8)
    base = dict(disk=(300.0, 2e-3))
    cases = {
        "ks_w1_not_photospheric": dict(ks_m=6.5, w1mpro=6.0),
        "registration_gt_1arcsec": dict(allwise_sep_arcsec=1.8),
        "ext_flag": dict(ext_flag=2),
        "cc_flags_w3w4": dict(cc_flags="00D0"),
        "w3_saturated": dict(w3mpro=2.0),
        "allwise_shared_by_gaia_mates": dict(allwise_n_mates=1),
        "gaia_beam_neighbour": dict(n_gaia_beam_neighbours=1),
        "wide_companion_1000au": dict(n_wide_companions=1),
        "non_single_star": dict(non_single_star=1),
        "wise_profile_misfit": dict(w4rchi2=8.0),
        "agn_colour": dict(w1mpro=7.0, w2mpro=6.3),
        "galaxy_colour": dict(w1mpro=7.0, w2mpro=6.65, w3mpro=2.9),
        "simbad_type": dict(simbad_otype="G"),
        "cirrus_ebv": dict(ebv_sfd=0.5),
        "young_group": dict(young_group="beta_Pic"),
        "young_region": dict(young_region="Sco-Cen_US"),
        "young_indicator": dict(n_young_indicators=1),
    }
    rows = [make_star(i, rng, **base, **over) for i, over in enumerate(cases.values())]
    df = pd.DataFrame(rows)
    df["ks_w1"] = df["ks_m"] - df["w1mpro"]
    for col in ("n_gaia_beam_neighbours", "n_wide_companions", "w4rchi2", "ebv_sfd", "n_young_indicators"):
        if col not in df:
            df[col] = np.nan
    vetted, counters = apply_rules(df, DEFAULT_VET)
    for i, name in enumerate(cases):
        assert vetted[f"rule_{name}"].iloc[i], name
        assert counters["kills"][name] >= 1, name
    assert vetted["killed"].all()
    # a clean star trips nothing, and the untested vetoes are named, not passed
    clean = pd.DataFrame([make_star(99, rng, **base)])
    clean["ks_w1"] = clean["ks_m"] - clean["w1mpro"]
    v, c = apply_rules(clean, DEFAULT_VET)
    assert not v["killed"].iloc[0] and "gaia_beam_neighbour" in v["untested"].iloc[0]
    assert c["untested"]["cirrus_ebv"] == 1


def test_classify_names_every_cell_outcome():
    df = pd.DataFrame({
        "ks_m": [6.0, 6.0, 6.0, 6.0, 6.0, np.nan, 6.0],
        "excess_significant": [True, True, True, True, True, True, False],
        "log_f_fmax_1gyr": [4, 4, 4, 4, 2, 4, np.nan], "log_f_fmax_1gyr_lo": [3.5, 3.5, 3.5, 3.5, 1.5, 3.5, np.nan],
        "t_bb_k": [300, 300, 300, 500, 300, 300, np.nan],
        "killed": [False, False, True, False, False, False, False],
        "age_class": ["MATURE_2PLUS", "AGE_UNDETERMINED", "MATURE_2PLUS", "MATURE_2PLUS", "MATURE_2PLUS",
                      "MATURE_2PLUS", "MATURE_2PLUS"],
        "is_control": [False, False, False, False, False, False, True]})
    out = classify(df, DEFAULT_VET)
    assert out["cradle_class"].tolist() == ["CANDIDATE", "IN_CELL_AGE_UNDETERMINED", "IN_CELL_KILLED",
                                            "ABOVE_FMAX_HOT", "BELOW_FMAX", "KS_MISSING",
                                            "CONTROL:NOT_SIGNIFICANT"]


# ---------------------------------------------------------------------------
# mineralogy
# ---------------------------------------------------------------------------
def _spectrum(featured: bool, noise: float = 0.01):
    w = np.linspace(5.2, 37.0, 700)
    cont = 2.0 * (w / 10.0) ** 0.5
    f = cont.copy()
    if featured:
        f *= 1 + 0.6 * np.exp(-0.5 * ((w - 9.9) / 1.0) ** 2) + 0.25 * np.exp(-0.5 * ((w - 18.0) / 1.5) ** 2)
    rng = np.random.default_rng(1)
    f = f * (1 + rng.normal(0, noise, len(w)))
    return w, f, np.full_like(f, noise * cont)


def test_silicate_feature_is_scored_and_a_flat_band_is_featureless():
    w, f, e = _spectrum(True)
    s = score_spectrum(w, f, e)
    assert s["verdict"] == "SILICATE_FEATURED" and s["feat10"]["contrast"] > 0.2 and s["feat10_snr"] > 3
    assert 9.5 < s["feat10"]["peak_um"] < 10.5
    assert slag_verdict(s, True).startswith("NATURAL_SILICATE")
    w, f, e = _spectrum(False)
    s = score_spectrum(w, f, e)
    assert s["verdict"] == "FEATURELESS"
    assert slag_verdict(s, True).startswith("SLAG_CONSISTENT")
    assert slag_verdict(s, False) == "NO_EXCESS_TO_SCORE"
    assert score_spectrum(w[:5], f[:5])["verdict"] == "NO_COVERAGE"


def test_ipac_table_parser_reads_a_spectrum():
    text = ("\\ comment\n| wavelength | flux_density | error |\n| double | double | double |\n"
            + "\n".join(f" {5 + 0.1 * i:.3f} {1.0 + 0.01 * i:.4f} 0.01" for i in range(40)))
    df = parse_ipac_table(text)
    assert list(df.columns) == ["wavelength", "flux_density", "error"] and len(df) == 40
    from seti.cradle.mineralogy import spectrum_from_table
    w, f, e = spectrum_from_table(df)
    assert len(w) == 40 and w[0] == pytest.approx(5.0)


def test_irs_url_ladder_prefers_a_file_column_and_falls_back_to_patterns():
    urls = acq.irs_spectrum_urls({"reqkey": 4931840, "tn": 1, "spec_file": "x/y.tbl"})
    assert urls[0] == acq.IRS_PRODUCT_BASE + "x/y.tbl" and any("04931840" in u for u in urls)
    assert len(urls) == len(set(urls))


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_end_to_end_recovers_the_injected_cradle_on_a_mature_star(tmp_path):
    rng = np.random.default_rng(21)
    f_inj = 10 ** 4.0 * E.wyatt_fmax((278.3 / 300) ** 2 * np.sqrt(0.9), 1000.0, 1.0, 0.9)
    star = make_star(999, rng, ks=6.5, bp_rp=0.85, disk=(300.0, f_inj), noise=0.0,
                     v_tan_kms=70.0, age=(3.0, 2.0, 5.0))
    arch = FakeArchive(make_archive(rng, extra=[star]))
    conf = _conf(tmp_path)
    rep, out = _run_all(tmp_path, arch, conf, n_shards=2)
    assert rep["verdict"] == "CRADLE_CANDIDATES", rep
    assert rep["n_candidates"] == 1
    c = rep["candidates"][0]
    assert int(c["source_id"]) == star["source_id"]
    assert c["age_class"] == "MATURE_2PLUS" and c["old_indicators"] == "iso+kin"
    assert 270 < c["t_bb_k"] < 330 and c["log_f_fmax_1gyr"] > 3.5
    assert any("mantle" in s for s in c["not_excluded"])
    assert rep["stage_counts"]["screened"] > 600 and rep["funnel"]["n_in_cell"] == 1
    cand = pd.read_csv(out / "candidates.csv")
    assert (cand["cradle_class"] == "CANDIDATE").sum() == 1
    assert (out / "controls.json").exists() and (out / "screen.json").exists()
    screen = json.loads((out / "screen.json").read_text())
    assert screen["coverage"]["n_units_done"] == 768 and screen["funnel"]["n_in_t_cell_photometric"] == 1
    assert c["neowise_measured"] is True or c["neowise_measured"] == 1


def test_the_same_excess_on_a_sco_cen_star_is_vetoed(tmp_path):
    rng = np.random.default_rng(22)
    f_inj = 10 ** 4.0 * E.wyatt_fmax((278.3 / 300) ** 2 * np.sqrt(0.9), 1000.0, 1.0, 0.9)
    # Upper Sco sky box at 140 pc; a FLAME age that (wrongly) says 3 Gyr and a
    # kinematic tangential velocity that is small: position + distance veto it.
    star = make_star(998, rng, ks=6.5, bp_rp=0.85, disk=(300.0, f_inj), noise=0.0, plx=11.0,
                     l=350.0, b=20.0, ra=243.0, dec=-22.0, v_tan_kms=70.0, age=(3.0, 2.0, 5.0))
    arch = FakeArchive(make_archive(rng, extra=[star]))
    rep, out = _run_all(tmp_path, arch, _conf(tmp_path))
    assert rep["verdict"] == "NO_CRADLE_CANDIDATE", rep["verdict"]
    assert rep["funnel"]["n_in_cell"] == 1 and rep["funnel"]["n_in_cell_killed"] == 1
    assert "young_region" in rep["funnel"]["in_cell_kill_reasons"]
    cand = pd.read_csv(out / "candidates.csv")
    assert cand["cradle_class"].iloc[0] == "IN_CELL_KILLED" and cand["young_region"].iloc[0] == "Sco-Cen_US"


def test_a_galaxy_blend_is_vetoed(tmp_path):
    rng = np.random.default_rng(23)
    f_inj = 10 ** 4.0 * E.wyatt_fmax((278.3 / 300) ** 2 * np.sqrt(0.9), 1000.0, 1.0, 0.9)
    star = make_star(997, rng, ks=6.5, bp_rp=0.85, disk=(300.0, f_inj), noise=0.0, v_tan_kms=70.0,
                     age=(3.0, 2.0, 5.0), ext_flag=3)
    # and a faint Gaia neighbour 3" away inside the W3 beam
    nb = make_star(996, rng, ks=9.5, ra=star["ra"] + 3 / 3600 / np.cos(np.radians(star["dec"])),
                   dec=star["dec"], phot_g_mean_mag=12.0)
    arch = FakeArchive(make_archive(rng, extra=[star, nb]))
    rep, out = _run_all(tmp_path, arch, _conf(tmp_path))
    assert rep["verdict"] == "NO_CRADLE_CANDIDATE"
    reasons = rep["funnel"]["in_cell_kill_reasons"]
    assert "ext_flag" in reasons and "gaia_beam_neighbour" in reasons


def test_a_single_indicator_is_age_undetermined_never_a_candidate(tmp_path):
    rng = np.random.default_rng(24)
    f_inj = 10 ** 4.0 * E.wyatt_fmax((278.3 / 300) ** 2 * np.sqrt(0.9), 1000.0, 1.0, 0.9)
    star = make_star(995, rng, ks=6.5, bp_rp=0.85, disk=(300.0, f_inj), noise=0.0, v_tan_kms=5.0,
                     age=(3.0, 2.0, 5.0))
    arch = FakeArchive(make_archive(rng, extra=[star]))
    rep, out = _run_all(tmp_path, arch, _conf(tmp_path))
    assert rep["verdict"] == "NO_CRADLE_CANDIDATE"
    assert rep["funnel"]["n_in_cell_age_undetermined"] == 1 and rep["n_candidates"] == 0
    assert rep["in_cell_age_undetermined"][0]["old_indicators"] == "iso"


def test_empty_archive_is_no_data_reached(tmp_path):
    arch = FakeArchive(make_archive(np.random.default_rng(1), n_clean=0))
    rep, out = _run_all(tmp_path, arch, _conf(tmp_path))
    assert rep["verdict"] == "NO_DATA_REACHED" and rep["n_candidates"] == 0
    assert rep["stage_counts"]["screened"] == 0


def test_missing_ages_shards_are_never_a_clean_null(tmp_path):
    rng = np.random.default_rng(25)
    f_inj = 10 ** 4.0 * E.wyatt_fmax((278.3 / 300) ** 2 * np.sqrt(0.9), 1000.0, 1.0, 0.9)
    star = make_star(994, rng, ks=6.5, disk=(300.0, f_inj), noise=0.0, v_tan_kms=70.0, age=(3.0, 2.0, 5.0))
    arch = FakeArchive(make_archive(rng, extra=[star]))
    conf = _conf(tmp_path)
    b = _backends(arch)
    out = tmp_path / "cradle"
    cradle_run("acquire", out_dir=out, conf=conf, backends=b)
    cradle_run("screen", out_dir=out, conf=conf, backends=b)
    rep = cradle_run("assess", out_dir=out, n_shards=2, conf=conf, backends=b)
    assert rep["verdict"] == "NO_DATA_REACHED" and "no_ages_shard_outputs" in rep["degraded"]
    # one of two shards present: degraded, not clean
    cradle_run("ages", out_dir=out, shard=0, n_shards=2, conf=conf, backends=b)
    rep = cradle_run("assess", out_dir=out, n_shards=2, conf=conf, backends=b)
    assert rep["verdict"].startswith("DEGRADED (ages_shards_missing:1/2")


def test_acquire_splits_a_timed_out_unit_and_records_failures(tmp_path):
    rng = np.random.default_rng(26)
    arch = FakeArchive(make_archive(rng, n_clean=50), timeout_units={(3, UNIT_K)}, fail_units={(3, 9)})
    conf = _conf(tmp_path)
    conf["sample"]["healpix_split_max_level"] = 4
    conf["acquire"]["irsa_fallback"] = False
    b = _backends(arch)
    out = tmp_path / "cradle"
    rep = cradle_run("acquire", out_dir=out, conf=conf, backends=b, max_units=12)
    assert rep["n_failed"] == 1 and rep["n_units_done"] == 12
    prog = json.loads((out / "acquire_s0of1.json").read_text())
    split = next(r for r in prog["records"] if r["label"] == f"hp3_{UNIT_K}")
    assert split["split"] is True and split["status"] == "OK" and split["n_rows"] == 50
    failed = next(r for r in prog["records"] if r["label"] == "hp3_9")
    assert failed["status"] == "QUERY_FAILED"
    # resume: nothing re-fetched
    n_calls = len(arch.calls)
    rep2 = cradle_run("acquire", out_dir=out, conf=conf, backends=b, max_units=12)
    assert rep2["n_units_done"] == 12 and len(arch.calls) == n_calls + 0


def test_probe_writes_a_record_and_finds_the_controls_by_position(tmp_path):
    rng = np.random.default_rng(27)
    bd = make_star(993, rng, ks=7.0, ra=CONTROLS[0]["ra"], dec=CONTROLS[0]["dec"], disk=(450.0, 3e-2))
    arch = FakeArchive(make_archive(rng, n_clean=30, extra=[bd]))
    conf = _conf(tmp_path)
    conf["probe"]["unit_k"] = UNIT_K
    out = tmp_path / "cradle"
    rep = cradle_run("probe", out_dir=out, conf=conf, backends=_backends(arch))
    assert rep["verdict"] == "PARENT_ROUTE_REACHABLE" and rep["shape_working"] == SHAPE_FULL
    assert rep["controls_found"] == 1
    assert [c["method"] for c in rep["controls"]] == ["asserted", None, "asserted"]
    assert rep["irsa"]["irs"]["n_columns"] == 7 and rep["irsa"]["irs_count"] == 3
    assert (out / "probe.json").exists()
    # and the acquire stage reuses the shape the probe recorded
    from seti.cradle.run import _shape_order
    assert _shape_order(conf, out)[0] == SHAPE_FULL and set(_shape_order(conf, out)) == set(JOINED_SHAPES)


def test_probe_with_every_route_dead_is_no_data_reached_not_a_crash(tmp_path):
    def dead(adql):
        raise GaiaQueryFailed({"status": "QUERY_FAILED", "error": "down"})
    b = acq.Backends(gaia=dead, irsa=dead, simbad_resolve=lambda n: None,
                     asu=lambda url: (_ for _ in ()).throw(RuntimeError("down")),
                     neowise=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    rep = cradle_run("probe", out_dir=tmp_path / "c", conf=_conf(tmp_path), backends=b)
    assert rep["verdict"] == "NO_DATA_REACHED" and rep["shape_working"] is None
    assert all(s["status"] == "QUERY_FAILED" for s in rep["gaia_shapes"])


def test_controls_land_where_the_literature_puts_them(tmp_path):
    rng = np.random.default_rng(28)
    # BD+20 307: 450 K, f = 3e-2 --- hot, far above f_max, outside the T cell
    bd = make_star(993, rng, ks=7.0, ra=CONTROLS[0]["ra"], dec=CONTROLS[0]["dec"], disk=(450.0, 3e-2),
                   noise=0.0, v_tan_kms=40.0, age=(1.5, 1.1, 2.5))
    arch = FakeArchive(make_archive(rng, extra=[bd]))
    rep, out = _run_all(tmp_path, arch, _conf(tmp_path))
    ctrl = {c["name"]: c for c in rep["controls"]}
    assert ctrl["BD+20 307"]["found"] and ctrl["BD+20 307"]["cradle_class"] == "CONTROL:ABOVE_FMAX_HOT"
    assert ctrl["BD+20 307"]["t_bb_k"] > 350 and ctrl["BD+20 307"]["log_f_fmax_1gyr"] > 5
    assert not ctrl["TYC 4479-3-1"]["found"]
    assert rep["verdict"] == "NO_CRADLE_CANDIDATE"      # the control is never a candidate


def test_parse_shard_and_config_and_files_exist():
    assert parse_shard("3/8") == (3, 8) and parse_shard(None) == (0, 1)
    with pytest.raises(ValueError):
        parse_shard("8/8")
    conf = load_cradle_config(Path("config/cradle.yaml"))
    assert conf["excess"]["t_cell_k"] == [250.0, 350.0] and conf["excess"]["log_f_fmax_min"] == 3.0
    assert len(conf["ages"]["young_groups"]) >= 15 and any(g["name"] == "Upper_Sco" for g in conf["ages"]["young_groups"])
    assert STAGES == ("probe", "acquire", "screen", "ages", "assess")
    for p in ("config/cradle.yaml", ".github/workflows/cradle.yml", "docs/cradle.md"):
        assert Path(p).exists(), p

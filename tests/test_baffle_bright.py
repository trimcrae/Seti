"""Offline tests for the BAFFLE bright tier (AKARI/IRAS deficits on G < 7.5 stars).

No network: every fetcher is a stub over one synthetic sky.  The sky carries a
realistic Ks − [9] scatter (0.15 mag), 20 % AKARI-18 / IRAS coverage, and a
handful of planted stars whose fate the tests assert one by one — including a
1″/yr star that only matches if the epoch propagation is right.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from seti.baffle import bright
from seti.baffle.bright import (
    BANDS,
    VERDICT_CAND,
    VERDICT_NO_DATA,
    VERDICT_NULL,
    fit_bright_locus,
    flux_to_mag,
    in_iras_ellipse,
    load_bright_config,
    match_within,
    propagate_to_epoch,
    residuals,
    screen_bright,
)

ZP = {"s09": 56.26, "s18": 12.00, "f12": 28.3, "f25": 6.73}
# The injected photospheric relations Ks - m_b = a + c * (J - Ks).
REL = {"s09": (0.05, 0.30), "s18": (0.10, 0.40), "f12": (0.08, 0.35), "f25": (0.12, 0.45)}


@pytest.fixture(scope="module")
def cfg():
    c = load_bright_config()
    c["targets"]["hipparcos"]["enabled"] = False
    return c


# --------------------------------------------------------------------------
# Synthetic sky
# --------------------------------------------------------------------------
def make_sky(n: int = 3000, seed: int = 3, second_frac: float = 0.2) -> pd.DataFrame:
    """A truth table: one row per star with every catalogue's view of it."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    jk = rng.uniform(-0.1, 1.25, n)
    ks = rng.uniform(4.5, 7.0, n)
    df = pd.DataFrame({
        "source_id": np.arange(1, n + 1, dtype=np.int64),
        "ra": ra, "dec": dec, "pmra": rng.normal(0, 30, n), "pmdec": rng.normal(0, 30, n),
        "parallax": rng.uniform(3, 40, n), "ks": ks, "j": ks + jk, "e_ks": 0.03, "e_j": 0.03,
        "g": ks + 1.2 + 1.5 * jk, "bp_rp": 0.4 + 1.6 * jk,
        "variable": "NOT_AVAILABLE", "nss": 0,
        "has_2mass": True, "has_akari": True,
        "has_s18": rng.uniform(0, 1, n) < second_frac,
        "has_iras": rng.uniform(0, 1, n) < second_frac,
        "deficit": 0.0, "deficit_bands": "all", "qflg": "AAA", "rflg": "222",
        "iras_q": 3,
    })
    for b in BANDS:
        a, c = REL[b]
        m = ks - (a + c * jk) + rng.normal(0, 0.15, n)
        df[f"true_m_{b}"] = m
    return df


def _plant(sky: pd.DataFrame, i: int, **kw) -> None:
    for k, v in kw.items():
        sky.loc[i, k] = v


def plant_specials(sky: pd.DataFrame) -> dict[str, int]:
    """Plant the stars each test asserts on; returns name -> row index."""
    ids = {}
    # 0: two-band 1-mag deficit (AKARI 9 + 18) -> candidate
    _plant(sky, 0, deficit=1.0, deficit_bands="all", has_s18=True, has_iras=False)
    ids["two_band"] = 0
    # 1: 9-um-only deficit, nothing else measured -> single_band_only
    _plant(sky, 1, deficit=1.0, deficit_bands="s09", has_s18=False, has_iras=False)
    ids["single"] = 1
    # 2: LPV-coloured deficit -> deferred
    _plant(sky, 2, deficit=1.0, has_s18=True, bp_rp=3.4)
    sky.loc[2, "j"] = sky.loc[2, "ks"] + 1.2
    ids["lpv"] = 2
    # 3: crowded (a G = 8 companion 15" away) -> crowded
    _plant(sky, 3, deficit=1.0, has_s18=True, has_iras=True)
    ids["crowded"] = 3
    # 4: predicted S09 = 5 Jy, no AKARI and no IRAS, high |b| -> missing list
    _plant(sky, 4, ra=200.0, dec=60.0, has_akari=False, has_iras=False, has_s18=False)
    a, c = REL["s09"]
    jk4 = 0.4
    m9 = -2.5 * np.log10(5.0 / ZP["s09"])
    sky.loc[4, "ks"] = m9 + a + c * jk4
    sky.loc[4, "j"] = sky.loc[4, "ks"] + jk4
    ids["missing"] = 4
    # 5: Gaia-variable with a two-band deficit -> variable veto
    _plant(sky, 5, deficit=1.0, has_s18=True, variable="VARIABLE")
    ids["variable"] = 5
    # 6: a 1"/yr star, no deficit, everything measured -> must still match
    _plant(sky, 6, pmra=800.0, pmdec=-600.0, has_s18=True, has_iras=True, dec=20.0, ra=50.0)
    ids["fast"] = 6
    # 7: two-instrument deficit (AKARI 9 + IRAS 12/25), no 18 -> candidate
    _plant(sky, 7, deficit=1.2, has_s18=False, has_iras=True)
    ids["two_instrument"] = 7
    # 8: IRAS quality-1 upper limit far below the photosphere
    _plant(sky, 8, has_iras=True, iras_q=1, has_s18=False)
    ids["iras_ul"] = 8
    # 9: bright Read-1 star (Ks = 2.5, e_Ks = 0.25), no deficit
    _plant(sky, 9, ks=2.5, e_ks=0.25, has_s18=True)
    sky.loc[9, "j"] = 2.5 + 0.3
    sky.loc[9, "rflg"] = "111"
    ids["read1"] = 9
    # Every plant other than the LPV and the missing star gets an ordinary colour
    # so that only the veto under test can fire.
    for name, i in ids.items():
        if name not in ("lpv", "missing", "read1"):
            sky.loc[i, "j"] = sky.loc[i, "ks"] + 0.5
    # Re-derive the true magnitudes of every planted star from its (possibly
    # changed) Ks and colour so the plant is on the relation before the deficit.
    rng = np.random.default_rng(99)
    for i in ids.values():
        jk = sky.loc[i, "j"] - sky.loc[i, "ks"]
        for b in BANDS:
            a, c = REL[b]
            sky.loc[i, f"true_m_{b}"] = sky.loc[i, "ks"] - (a + c * jk) + rng.normal(0, 0.02)
    return ids


def _obs_flux(sky: pd.DataFrame, b: str) -> np.ndarray:
    d = sky["deficit"].to_numpy(float)
    only = sky["deficit_bands"].to_numpy()
    applied = np.where((only == "all") | (only == b), d, 0.0)
    return ZP[b] * 10 ** (-0.4 * (sky[f"true_m_{b}"].to_numpy(float) + applied))


def _b1950(ra, dec):
    import astropy.units as u
    from astropy.coordinates import FK4, SkyCoord

    c = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg, frame="icrs")
    f = c.transform_to(FK4(equinox="B1950", obstime="B1950"))
    return f.ra.deg, f.dec.deg


class FakeArchive:
    """Every fetcher the bright tier takes, served from one truth table."""

    def __init__(self, sky: pd.DataFrame, cfg: dict, companions: pd.DataFrame | None = None,
                 fail: set[str] = frozenset()):
        self.sky, self.cfg, self.fail = sky, cfg, set(fail)
        self.companions = companions if companions is not None else pd.DataFrame(
            columns=["target_id", "source_id", "ra", "dec", "phot_g_mean_mag"])
        self.calls: list[str] = []
        ep = cfg["epochs"]
        self.pos = {}
        for key in ("tmass", "akari", "iras"):
            self.pos[key] = propagate_to_epoch(sky["ra"], sky["dec"], sky["pmra"], sky["pmdec"],
                                               from_epoch=ep["gaia"], to_epoch=ep[key])

    # -- Gaia -------------------------------------------------------------
    def gaia(self, query: str, label: str) -> pd.DataFrame:
        self.calls.append(label)
        if "gaia" in self.fail:
            raise RuntimeError("gaia archive down")
        m = re.search(r"ra >= ([\d.]+) AND (?:t\.)?ra < ([\d.]+)", query)
        lo, hi = float(m.group(1)), float(m.group(2))
        if label.startswith("neighbours"):
            if "neighbours_join" in self.fail:
                raise RuntimeError("self-join timed out")
            sub = self.sky[(self.sky["ra"] >= lo) & (self.sky["ra"] < hi)]
            return self.companions[self.companions["target_id"].isin(sub["source_id"])].copy()
        s = self.sky[(self.sky["ra"] >= lo) & (self.sky["ra"] < hi)]
        from seti.baffle.bright import ecliptic_and_galactic
        ecl, gl, gb = ecliptic_and_galactic(s["ra"], s["dec"])
        return pd.DataFrame({
            "source_id": s["source_id"].to_numpy(), "ra": s["ra"].to_numpy(),
            "dec": s["dec"].to_numpy(), "l": gl, "b": gb, "ecl_lat": ecl,
            "parallax": s["parallax"].to_numpy(), "parallax_over_error": 50.0,
            "pmra": s["pmra"].to_numpy(), "pmdec": s["pmdec"].to_numpy(), "ruwe": 1.0,
            "phot_g_mean_mag": s["g"].to_numpy(), "bp_rp": s["bp_rp"].to_numpy(),
            "phot_variable_flag": s["variable"].to_numpy(), "non_single_star": s["nss"].to_numpy(),
        })

    # -- 2MASS (upload cone) ---------------------------------------------
    xmatch_style = False

    def tmass(self, positions: pd.DataFrame, radius_arcsec: float, label: str) -> pd.DataFrame:
        self.calls.append(label)
        if "tmass" in self.fail:
            raise RuntimeError("VizieR upload refused")
        if self.xmatch_style:
            return self._tmass_xmatch(positions, radius_arcsec)
        cat = self.sky[self.sky["has_2mass"]]
        cra, cde = self.pos["tmass"]
        cat_pos = pd.DataFrame({"ra": cra[cat.index], "dec": cde[cat.index]})
        m = match_within(positions, cat_pos, radius_arcsec)
        rows = cat.iloc[m["cat_index"].to_numpy()]
        return pd.DataFrame({
            "source_id": positions["source_id"].to_numpy()[m["pos_index"].to_numpy()],
            "RAJ2000": cat_pos["ra"].to_numpy()[m["cat_index"]],
            "DEJ2000": cat_pos["dec"].to_numpy()[m["cat_index"]],
            "2MASS": ["J" + str(i) for i in rows["source_id"]],
            "Jmag": rows["j"].to_numpy(), "e_Jmag": rows["e_j"].to_numpy(),
            "Hmag": rows["j"].to_numpy() - 0.1, "e_Hmag": 0.03,
            "Kmag": rows["ks"].to_numpy(), "e_Kmag": rows["e_ks"].to_numpy(),
            "Qflg": rows["qflg"].to_numpy(), "Rflg": rows["rflg"].to_numpy(),
            "Bflg": "111", "Cflg": "000", "Xflg": 0, "Aflg": 0,
        })

    def _tmass_xmatch(self, positions: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
        """Exactly the CDS X-Match CSV columns of run 34048837928, source_id as float."""
        cat = self.sky[self.sky["has_2mass"]]
        cra, cde = self.pos["tmass"]
        cat_pos = pd.DataFrame({"ra": cra[cat.index], "dec": cde[cat.index]})
        m = match_within(positions, cat_pos, radius_arcsec)
        rows = cat.iloc[m["cat_index"].to_numpy()]
        pi = m["pos_index"].to_numpy()
        return pd.DataFrame({
            "angDist": m["sep_arcsec"].to_numpy(),
            "source_id": positions["source_id"].to_numpy()[pi].astype(float),
            "ra": positions["ra"].to_numpy()[pi], "dec": positions["dec"].to_numpy()[pi],
            "2MASS": ["J" + str(i) for i in rows["source_id"]],
            "RAJ2000": cat_pos["ra"].to_numpy()[m["cat_index"]],
            "DEJ2000": cat_pos["dec"].to_numpy()[m["cat_index"]],
            "errHalfMaj": 0.06, "errHalfMin": 0.06, "errPosAng": 90,
            "Jmag": rows["j"].to_numpy(), "Hmag": rows["j"].to_numpy() - 0.1, "Kmag": rows["ks"].to_numpy(),
            "e_Jmag": rows["e_j"].to_numpy(), "e_Hmag": 0.03, "e_Kmag": rows["e_ks"].to_numpy(),
            "Qfl": rows["qflg"].to_numpy(), "Rfl": rows["rflg"].to_numpy(), "X": 0,
            "MeasureJD": 2451000.5,
        })

    # -- AKARI (RA slice) --------------------------------------------------
    def akari(self, ra_lo: float, ra_hi: float, label: str) -> pd.DataFrame:
        self.calls.append(label)
        if "akari" in self.fail:
            raise RuntimeError("VizieR akari slice failed")
        cra, cde = self.pos["akari"]
        sel = self.sky["has_akari"].to_numpy() & (cra >= ra_lo) & (cra < ra_hi)
        s = self.sky[sel]
        s09 = _obs_flux(self.sky, "s09")[sel]
        s18 = _obs_flux(self.sky, "s18")[sel]
        has18 = s["has_s18"].to_numpy(bool)
        return pd.DataFrame({
            "objID": s["source_id"].to_numpy(), "objName": "AKARI-IRC-V1_J", "errMaj": 1.5,
            "errMin": 1.0, "errPA": 0.0,
            "S09": s09, "e_S09": 0.04 * s09, "q_S09": 3,
            "S18": np.where(has18, s18, np.nan), "e_S18": np.where(has18, 0.06 * s18, np.nan),
            "q_S18": np.where(has18, 3, 0),
            "RAJ2000": cra[sel], "DEJ2000": cde[sel],
        })

    # -- IRAS (RA slice, B1950 positions, percent errors) ---------------
    def iras(self, ra_lo: float, ra_hi: float, label: str) -> pd.DataFrame:
        self.calls.append(label)
        if "iras" in self.fail:
            raise RuntimeError("VizieR iras slice failed")
        cra, cde = self.pos["iras"]
        ra50, de50 = _b1950(cra, cde)
        sel = self.sky["has_iras"].to_numpy() & (ra50 >= ra_lo) & (ra50 < ra_hi)
        s = self.sky[sel]
        f12 = _obs_flux(self.sky, "f12")[sel]
        f25 = _obs_flux(self.sky, "f25")[sel]
        q = s["iras_q"].to_numpy()
        # A quality-1 row carries an UPPER LIMIT, here a tenth of the photosphere.
        f12 = np.where(q == 1, 0.1 * f12, f12)
        f25 = np.where(q == 1, 0.1 * f25, f25)
        return pd.DataFrame({
            "IRAS": ["I" + str(i) for i in s["source_id"]], "RA1950": ra50[sel], "DE1950": de50[sel],
            "Major": 25.0, "Minor": 8.0, "PosAng": 90.0, "NHcon": 3,
            "Fnu_12": f12, "e_Fnu_12": 6.0, "q_Fnu_12": q,
            "Fnu_25": f25, "e_Fnu_25": 8.0, "q_Fnu_25": q,
            "Fnu_60": 0.4, "e_Fnu_60": 30, "q_Fnu_60": 1, "Fnu_100": 1.0, "e_Fnu_100": 30,
            "q_Fnu_100": 1, "Cirr3": 2.0, "Confuse": 0, "Var": 0,
        })


def _run(cfg, tmp_path, sky, **kw):
    arch = FakeArchive(sky, cfg, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = bright.run_bright_stage(cfg, tmp_path, gaia_fetcher=arch.gaia, tmass_fetcher=arch.tmass,
                                    akari_fetcher=arch.akari, iras_fetcher=arch.iras)
    return s, arch


@pytest.fixture(scope="module")
def planted(cfg, tmp_path_factory):
    sky = make_sky()
    ids = plant_specials(sky)
    t = sky.loc[ids["crowded"]]
    comp = pd.DataFrame({"target_id": [t["source_id"]], "source_id": [999999],
                         "ra": [t["ra"] + 15.0 / 3600 / np.cos(np.radians(t["dec"]))],
                         "dec": [t["dec"]], "phot_g_mean_mag": [8.0]})
    out = tmp_path_factory.mktemp("planted")
    s, arch = _run(cfg, out, sky, companions=comp)
    resid = pd.read_csv(out / "bright_residuals.csv")
    cands = pd.read_csv(out / "candidates.csv")
    missing = pd.read_csv(out / "missing_bright_candidates.csv")
    return {"sky": sky, "ids": ids, "summary": s, "out": out, "resid": resid, "cands": cands,
            "missing": missing, "arch": arch}


# --------------------------------------------------------------------------
# Pure functions
# --------------------------------------------------------------------------
def test_propagate_to_epoch_matches_astropy_over_16_years():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    ra, dec, pmra, pmdec = 10.0, 40.0, 800.0, -600.0     # 1"/yr total
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, pm_ra_cosdec=pmra * u.mas / u.yr,
                     pm_dec=pmdec * u.mas / u.yr, obstime=Time(2016.0, format="jyear"), frame="icrs")
        ref = c.apply_space_motion(new_obstime=Time(2000.0, format="jyear"))
    r2, d2 = propagate_to_epoch([ra], [dec], [pmra], [pmdec], from_epoch=2016.0, to_epoch=2000.0)
    moved = SkyCoord(ra=r2[0] * u.deg, dec=d2[0] * u.deg, frame="icrs")
    assert abs(c.separation(moved).arcsec - 16.0) < 0.05
    assert ref.separation(moved).arcsec < 0.05
    assert r2[0] < ra and d2[0] > dec          # backwards in time: opposite to the motion


def test_propagate_handles_missing_pm_and_wraps_ra():
    r, d = propagate_to_epoch([359.9999], [0.0], [np.nan], [np.nan], to_epoch=1983.5)
    assert r[0] == pytest.approx(359.9999) and d[0] == 0.0
    r, d = propagate_to_epoch([359.9999], [0.0], [3.6e6 * 0.001 / 33], [0.0], from_epoch=2016.0,
                              to_epoch=2049.0)
    assert 0 <= r[0] < 0.01


def test_flux_to_mag_with_config_zero_points(cfg):
    zp = cfg["akari"]["zero_points_jy"]["s09"]
    m, e = flux_to_mag([zp, zp / 100.0, 0.0, np.nan], zp, [0.1 * zp, 0.0, 0.0, 0.0])
    assert m[0] == pytest.approx(0.0) and m[1] == pytest.approx(5.0)
    assert np.isnan(m[2]) and np.isnan(m[3])
    assert e[0] == pytest.approx(0.1086, abs=1e-3)
    assert cfg["iras"]["zero_points_jy"]["f12"] == 28.3


def test_locus_fit_recovers_an_injected_relation(cfg):
    rng = np.random.default_rng(1)
    jk = rng.uniform(-0.1, 1.0, 4000)
    y = 0.05 + 0.3 * jk + rng.normal(0, 0.15, 4000)
    y[:40] += 2.0                                       # an excess tail must not bias the median
    loc = fit_bright_locus(jk, y, cfg["locus"])
    assert loc["ok"] and len(loc["bin_centres"]) > 15
    x = np.array(loc["bin_centres"])
    assert np.max(np.abs(np.array(loc["median"]) - (0.05 + 0.3 * x))) < 0.04
    assert abs(loc["scatter_global"] - 0.15) < 0.03
    assert not fit_bright_locus(jk[:30], y[:30], cfg["locus"])["ok"]


def test_match_within_returns_nearest_inside_radius_only():
    pos = pd.DataFrame({"ra": [10.0, 20.0, 30.0], "dec": [0.0, 0.0, 0.0]})
    cat = pd.DataFrame({"ra": [10.0 + 2 / 3600, 10.0 + 1 / 3600, 20.0 + 20 / 3600], "dec": [0.0, 0.0, 0.0]})
    m = match_within(pos, cat, 5.0)
    assert m["pos_index"].tolist() == [0]
    assert m["cat_index"].tolist() == [1]
    assert m["sep_arcsec"].iloc[0] == pytest.approx(1.0, abs=1e-3)
    assert len(match_within(pos.iloc[:0], cat, 5.0)) == 0


def test_iras_error_ellipse_floor_and_orientation():
    # Major axis along PA = 90 (east-west): a 40" east offset is inside a 50" x 8" ellipse
    assert in_iras_ellipse([40.0], [0.0], [50.0], [8.0], [90.0], 30.0, 120.0)[0]
    # ... but a 40" north offset is outside (minor axis floored to 30")
    assert not in_iras_ellipse([0.0], [40.0], [50.0], [8.0], [90.0], 30.0, 120.0)[0]
    # missing ellipse -> circle of the floor radius
    assert in_iras_ellipse([20.0], [20.0], [np.nan], [np.nan], [np.nan], 30.0, 120.0)[0]


def test_query_builders_only_name_columns_that_exist():
    q = bright.gaia_targets_query(7.5, 0.0, 30.0)
    for c in bright.GAIA_COLUMNS:
        assert c in q
    assert "phot_g_mean_mag < 7.5" in q
    assert "n.source_id != t.source_id" in bright.gaia_neighbours_query(7.5, 10.0, 60.0, 0.0, 30.0)
    assert bright.resolve_aliases(["RAJ2000", "S09", "e_S09", "q_S09", "objID"], bright.AKARI_ALIASES) == {
        "akari_id": "objID", "akari_ra": "RAJ2000", "s09": "S09", "e_s09": "e_S09", "q_s09": "q_S09"}
    assert bright._adql_col("2MASS") == '"2MASS"' and bright._adql_col("Kmag") == "Kmag"


# --------------------------------------------------------------------------
# The synthetic population, end to end
# --------------------------------------------------------------------------
def test_verdict_is_candidates_pending_vet_and_summary_is_complete(planted):
    s = planted["summary"]
    assert s["verdict"] == VERDICT_CAND
    assert s["n_targets"] == 3000
    assert s["n_with_2mass"] >= 2990
    assert s["n_with_akari"] >= 2900 and s["n_with_iras"] > 400
    for k in ("counters", "tail_asymmetry", "sensitivity", "denominators", "locus",
              "brightest_300_not_covered", "ledger_counts"):
        assert k in s
    assert s["brightest_300_not_covered"] is True          # Hipparcos disabled in the fixture
    assert (planted["out"] / "ledger.json").exists()
    led = json.loads((planted["out"] / "ledger.json").read_text())["entries"]
    assert {e["stage"] for e in led} >= {"gaia_targets", "gaia_neighbours", "tmass", "akari", "iras"}
    assert all(e["status"] in ("QUERY_OK", "QUERY_RETURNED_ZERO_ROWS") for e in led)
    assert not (planted["out"] / "targets.parquet").read_bytes() == b""


def test_two_band_and_two_instrument_deficits_are_recovered(planted):
    ids = planted["ids"]
    got = set(planted["cands"]["source_id"].astype(int))
    assert ids["two_band"] + 1 in got
    assert ids["two_instrument"] + 1 in got
    row = planted["cands"].set_index("source_id").loc[ids["two_band"] + 1]
    assert row["primary_band"] == "s09" and "s18" in row["agreeing_bands"]
    assert row["resid_s09"] < -0.8 and row["sig_s09"] < -4
    row = planted["cands"].set_index("source_id").loc[ids["two_instrument"] + 1]
    assert "f12" in row["agreeing_bands"] or "f25" in row["agreeing_bands"]


def test_the_only_candidates_are_the_planted_ones(planted):
    got = set(planted["cands"]["source_id"].astype(int))
    assert got == {planted["ids"]["two_band"] + 1, planted["ids"]["two_instrument"] + 1}


def test_vetoes_land_on_the_right_planted_stars(planted):
    ids = planted["ids"]
    r = planted["resid"].set_index("source_id")
    assert r.loc[ids["single"] + 1, "veto"] == "single_band_only"
    assert r.loc[ids["lpv"] + 1, "veto"] == "lpv_colour"
    assert r.loc[ids["crowded"] + 1, "veto"] == "crowded"
    assert r.loc[ids["crowded"] + 1, "n_neigh_30"] == 1
    assert r.loc[ids["variable"] + 1, "veto"] == "variable"
    c = planted["summary"]["counters"]
    assert c["single_band_only"] >= 1 and c["lpv_colour"] == 1 and c["crowded"] == 1
    assert c["variable"] == 1 and c["deferred_lpv"] == 1
    assert c["n_candidates"] == 2


def test_fast_star_is_matched_in_every_catalogue_because_positions_are_propagated(planted):
    r = planted["resid"].set_index("source_id").loc[planted["ids"]["fast"] + 1]
    assert r["has_tmass"] and r["has_akari"] and r["has_iras"]
    assert r["tmass_sep_arcsec"] < 0.5 and r["akari_sep_arcsec"] < 0.5
    assert r["veto"] == "not_deficit"


def test_star_with_predicted_5jy_and_no_midir_lands_in_missing_list(planted):
    m = planted["missing"]
    assert planted["ids"]["missing"] + 1 in set(m["source_id"].astype(int))
    row = m.set_index("source_id").loc[planted["ids"]["missing"] + 1]
    assert 3.5 < row["pred_s09_jy"] < 7.0
    assert row["predicted_over_akari_limit"] > 10
    assert "coverage" in row["caveat"]
    assert planted["summary"]["n_missing_bright_candidates"] == len(m)


def test_iras_upper_limit_below_photosphere_is_reported_separately(planted):
    sid = planted["ids"]["iras_ul"] + 1
    r = planted["resid"].set_index("source_id").loc[sid]
    assert r["iras_upper_limit_below_photosphere"]
    assert r["veto"] == "not_deficit"                 # an upper limit is not a detection
    assert sid in planted["summary"]["iras_upper_limit_below_photosphere_source_ids"]
    assert planted["summary"]["counters"]["iras_upper_limit_below_photosphere"] >= 1


def test_read1_regime_is_flagged_and_its_error_is_carried(planted):
    r = planted["resid"].set_index("source_id").loc[planted["ids"]["read1"] + 1]
    assert r["tmass_read1_regime"] and r["e_ks"] == pytest.approx(0.25)
    assert r["err_s09"] >= 0.25                       # the star's own error, not the locus scatter
    assert planted["summary"]["n_with_2mass_read1_regime"] >= 1


def test_iras_positions_were_precessed_from_b1950(planted):
    acq = planted["summary"]["acquisition"]["iras"]
    assert acq["frame"] == "fk4_b1950_precessed_to_icrs"
    assert acq["columns"]["iras_ra"] == "RA1950"
    r = planted["resid"]
    assert r["has_iras"].sum() > 400 and r.loc[r["has_iras"], "iras_sep_arcsec"].max() < 5


def test_excess_tail_is_the_control_and_sensitivity_is_reported(planted):
    s = planted["summary"]
    t = s["tail_asymmetry"]["s09"]
    assert t["n"] > 2500 and abs(t["scatter_global"] - 0.15) < 0.03
    sens = s["sensitivity"]
    assert sens["n_base"] > 2500
    assert sens["2"]["recovered_two_band"] > sens["0.5"]["recovered_two_band"]
    # A 2-mag (near-opaque) deficit is recovered wherever a second band exists.
    assert sens["2"]["recovered_two_band"] > 0.30
    assert sens["1"]["recovered_primary_only"] > 0.95
    assert sens["1"]["recovered_two_band_given_second_band"] > 0.90
    assert sens["n_with_second_band"] > 800


def test_clean_null_gives_zero_candidates_and_the_null_verdict(cfg, tmp_path):
    sky = make_sky(n=1500, seed=11)
    s, _ = _run(cfg, tmp_path, sky)
    assert s["verdict"] == VERDICT_NULL
    assert s["n_candidates"] == 0
    assert s["counters"]["n_primary_deficit"] <= 2       # 4-sigma tail of 1500 draws
    assert pd.read_csv(tmp_path / "candidates.csv").empty


def test_raising_gaia_fetcher_gives_no_data_reached_with_a_written_summary(cfg, tmp_path):
    sky = make_sky(n=200, seed=5)
    s, arch = _run(cfg, tmp_path, sky, fail={"gaia"})
    assert s["verdict"] == VERDICT_NO_DATA
    assert s["n_targets"] == 0
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["verdict"] == VERDICT_NO_DATA
    led = json.loads((tmp_path / "ledger.json").read_text())["entries"]
    assert len(led) == 12 and all(e["status"] == "QUERY_FAILED" for e in led)


def test_a_dead_midir_archive_is_a_degraded_source_not_a_null(cfg, tmp_path):
    sky = make_sky(n=1200, seed=8)
    s, _ = _run(cfg, tmp_path, sky, fail={"iras"})
    assert s["verdict"].startswith("DEGRADED_SOURCE (") and "iras:QUERY_FAILED" in s["verdict"]
    assert s["science_verdict"] == VERDICT_NULL
    assert s["n_with_akari"] > 1000 and s["n_with_iras"] == 0


def test_neighbour_self_join_failure_falls_back_to_bulk_counting(cfg, tmp_path):
    sky = make_sky(n=300, seed=21)
    arch = FakeArchive(sky, cfg, fail={"neighbours_join"})

    def gaia(query, label):
        if label.endswith("_bulk"):
            m = re.search(r"ra >= ([-\d.]+) AND ra < ([-\d.]+)", query)
            lo, hi = float(m.group(1)), float(m.group(2))
            t = sky.iloc[0]
            comp = pd.DataFrame({"source_id": [777777], "ra": [t["ra"] + 10 / 3600],
                                 "dec": [t["dec"]], "phot_g_mean_mag": [9.0]})
            return comp[(comp["ra"] >= lo) & (comp["ra"] < hi)]
        return arch.gaia(query, label)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = bright.run_bright_stage(cfg, tmp_path, gaia_fetcher=gaia, tmass_fetcher=arch.tmass,
                                    akari_fetcher=arch.akari, iras_fetcher=arch.iras)
    r = pd.read_csv(tmp_path / "bright_residuals.csv").set_index("source_id")
    assert r.loc[1, "n_neigh_30"] == 1
    assert s["acquisition"]["neighbours"]["failed"] == 0


def test_checkpoints_are_reused_on_rerun(cfg, tmp_path):
    sky = make_sky(n=300, seed=4)
    s1, a1 = _run(cfg, tmp_path, sky)
    s2, a2 = _run(cfg, tmp_path, sky)
    assert s2["verdict"] == s1["verdict"]
    assert not any(c.startswith(("targets", "akari", "iras", "tmass")) for c in a2.calls)
    assert s2["ledger_counts"]["FROM_CHECKPOINT"] >= 12 + 12 + 12 + 1


def test_max_targets_keeps_the_brightest(cfg, tmp_path):
    sky = make_sky(n=400, seed=6)
    arch = FakeArchive(sky, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = bright.run_bright_stage(cfg, tmp_path, gaia_fetcher=arch.gaia, tmass_fetcher=arch.tmass,
                                    akari_fetcher=arch.akari, iras_fetcher=arch.iras, max_targets=50)
    assert s["n_targets"] == 50
    t = pd.read_parquet(tmp_path / "targets.parquet")
    assert t["phot_g_mean_mag"].max() <= np.sort(sky["g"].to_numpy())[49] + 1e-9


def test_hipparcos_supplement_adds_only_stars_gaia_lacks(cfg, tmp_path):
    c = json.loads(json.dumps(cfg))
    c["targets"]["hipparcos"]["enabled"] = True
    sky = make_sky(n=300, seed=9)
    arch = FakeArchive(sky, c)
    dup = sky.iloc[0]

    def hip(query, label):
        assert "Hpmag" in query or "hip2" in query
        return pd.DataFrame({"HIP": [91262, 32349, 1], "RArad": [279.23473, 101.28716, dup["ra"]],
                             "DErad": [38.78369, -16.71612, dup["dec"]], "Plx": [130.2, 379.2, 10.0],
                             "e_Plx": [0.4, 1.6, 0.5], "pmRA": [200.9, -546.0, dup["pmra"]],
                             "pmDE": [286.2, -1223.1, dup["pmdec"]], "Hpmag": [0.09, -1.09, 2.5],
                             "B-V": [0.0, 0.0, 0.5]})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = bright.run_bright_stage(c, tmp_path, gaia_fetcher=arch.gaia, tmass_fetcher=arch.tmass,
                                    akari_fetcher=arch.akari, iras_fetcher=arch.iras, hip_fetcher=hip)
    assert s["n_hipparcos_added"] == 2 and s["brightest_300_not_covered"] is False
    assert s["acquisition"]["targets"]["hipparcos"]["n_duplicates_of_gaia"] == 1
    t = pd.read_parquet(tmp_path / "targets.parquet")
    h = t[t["origin"] == "hipparcos2"]
    assert set(h["hip"].astype(int)) == {91262, 32349}
    assert np.isfinite(h["ecl_lat"]).all() and (h["source_id"] < 0).all()


def test_hipparcos_failure_is_recorded_not_fatal(cfg, tmp_path):
    c = json.loads(json.dumps(cfg))
    c["targets"]["hipparcos"]["enabled"] = True
    sky = make_sky(n=300, seed=10)
    arch = FakeArchive(sky, c)

    def hip(query, label):
        raise RuntimeError("VizieR down")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = bright.run_bright_stage(c, tmp_path, gaia_fetcher=arch.gaia, tmass_fetcher=arch.tmass,
                                    akari_fetcher=arch.akari, iras_fetcher=arch.iras, hip_fetcher=hip)
    assert s["brightest_300_not_covered"] is True
    assert "hipparcos:QUERY_FAILED" in s["verdict"]
    assert s["n_targets"] == 300


# --------------------------------------------------------------------------
# Regressions from runner run 34048837928 (NO_DATA_REACHED)
# --------------------------------------------------------------------------
# TAP_SCHEMA.columns on TAPVizieR serves column names ALREADY double-quoted.
AKARI_SCHEMA = ['"objID"', '"objName"', '"errMaj"', '"errMin"', '"errPA"', '"S09"', '"e_S09"',
                '"q_S09"', '"S18"', '"e_S18"', '"q_S18"', '"f09"', '"f18"', '"Ndet"',
                '"RAJ2000"', '"DEJ2000"']
IRAS_SCHEMA = ['"recno"', '"IRAS"', '"RA1950"', '"DE1950"', '"Major"', '"Minor"', '"PosAng"',
               '"NHcon"', '"Fnu_12"', '"e_Fnu_12"', '"q_Fnu_12"', '"Fnu_25"', '"e_Fnu_25"',
               '"q_Fnu_25"', '"Fnu_60"', '"q_Fnu_60"', '"Fnu_100"', '"q_Fnu_100"', '"Var"',
               '"Confuse"', '"Cirr3"']
HIP_SCHEMA = ['"recno"', '"HIP"', '"RArad"', '"e_RArad"', '"DErad"', '"Plx"', '"e_Plx"',
              '"pmRA"', '"pmDE"', '"Hpmag"', '"B-V"', '"e_B-V"', '"V-I"']
TMASS_SCHEMA = ['"RAJ2000"', '"DEJ2000"', '"2MASS"', '"Jmag"', '"e_Jmag"', '"Hmag"', '"e_Hmag"',
                '"Kmag"', '"e_Kmag"', '"Qflg"', '"Rflg"', '"Bflg"', '"Cflg"', '"Xflg"', '"Aflg"']


def _assert_valid_adql_select(select: str):
    assert '""' not in select, select
    assert select and not select.startswith(",") and ",," not in select
    for tok in select.split(", "):
        tok = tok.split(".")[-1]
        assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*|"[^"]+"', tok), tok


@pytest.mark.parametrize("schema,aliases,ra_key,expect", [
    (AKARI_SCHEMA, bright.AKARI_ALIASES, "akari_ra", ("RAJ2000", "S09", "e_S09", "q_S18")),
    (IRAS_SCHEMA, bright.IRAS_ALIASES, "iras_ra", ("RA1950", "Fnu_12", "e_Fnu_12", "q_Fnu_25")),
    (HIP_SCHEMA, bright.HIP_ALIASES, "hip_ra", ("HIP", "RArad", "Hpmag", '"B-V"')),
    (TMASS_SCHEMA, bright.TMASS_ALIASES, "tmass_ra", ('"2MASS"', "Kmag", "Qflg")),
], ids=["akari", "iras", "hipparcos", "tmass"])
def test_quoted_schema_names_compose_a_valid_select(schema, aliases, ra_key, expect):
    res = bright.resolve_aliases(schema, aliases)
    assert ra_key in res and all('"' not in v for v in res.values())
    select = bright.select_list(res, required=(ra_key,))
    _assert_valid_adql_select(select)
    for e in expect:
        assert e in select.split(", "), (e, select)
    # Quoting is idempotent: an already-quoted name is quoted exactly once.
    assert bright._adql_col('"B-V"') == '"B-V"' and bright._adql_col('"RAJ2000"') == "RAJ2000"
    assert bright._adql_col("2MASS") == '"2MASS"'


def test_slice_fetcher_query_from_quoted_schema_has_no_empty_identifier(monkeypatch):
    seen = {}

    def fake_discover(table, url=None):
        return {"table": table, "names": [bright._unquote(c) for c in AKARI_SCHEMA],
                "meta": {}, "route": "TAP_SCHEMA", "errors": []}

    def fake_run(query, **kw):
        seen["q"] = query
        return pd.DataFrame({"S09": [1.0]})

    monkeypatch.setattr(bright, "discover_columns", fake_discover)
    monkeypatch.setattr(bright, "run_vizier", fake_run)
    f = bright.VizierSliceFetcher('"II/297/irc"', bright.AKARI_ALIASES, "akari_ra")
    f(0.0, 30.0, "akari_ra00")
    q = seen["q"]
    assert '""' not in q and 'FROM "II/297/irc" WHERE RAJ2000 >= 0.0 AND RAJ2000 < 30.0' in q
    assert "S09, e_S09, q_S09" in q


def test_discover_columns_strips_the_quotes_tap_schema_serves(monkeypatch):
    def fake_run(query, **kw):
        assert "TAP_SCHEMA.columns" in query
        return pd.DataFrame({"column_name": ['"RAJ2000"', '"B-V"'], "ucd": ["pos.eq.ra", ""],
                             "unit": ["deg", "mag"], "datatype": ["double", "float"],
                             "description": ["", ""]})

    monkeypatch.setattr(bright, "run_vizier", fake_run)
    d = bright.discover_columns('"I/311/hip2"')
    assert d["names"] == ["RAJ2000", "B-V"] and d["meta"]["RAJ2000"]["unit"] == "deg"
    assert d["names_as_served"] == ['"RAJ2000"', '"B-V"']


def test_empty_identifier_is_refused_loudly():
    with pytest.raises(ValueError, match="hpmag"):
        bright._adql_col('""', "hpmag")
    with pytest.raises(RuntimeError, match="akari_ra"):
        bright.select_list({"s09": "S09"}, required=("akari_ra",))
    with pytest.raises(RuntimeError, match="q_s09"):
        bright.select_list({"s09": "S09", "q_s09": '""'})


def test_xmatch_columns_attach_and_absent_flags_are_unknown(cfg):
    """Exactly the X-Match CSV header of run 34048837928; source_id as float."""
    raw = pd.DataFrame({
        "angDist": [0.4, 2.1, 0.2], "source_id": [11.0, 11.0, 12.0],
        "ra": [10.0, 10.0, 20.0], "dec": [1.0, 1.0, 2.0],
        "2MASS": ["J1", "J1b", "J2"], "RAJ2000": [10.0001, 10.0006, 20.00005],
        "DEJ2000": [1.0, 1.0, 2.0], "errHalfMaj": 0.06, "errHalfMin": 0.06, "errPosAng": 90,
        "Jmag": [5.0, 9.0, 4.0], "Hmag": [4.8, 8.8, 3.7], "Kmag": [4.6, 8.6, 3.5],
        "e_Jmag": 0.03, "e_Hmag": 0.03, "e_Kmag": [0.03, 0.05, 0.25],
        "Qfl": ["AAA", "AAA", "EAB"], "Rfl": ["222", "222", "111"], "X": 0, "MeasureJD": 2451000.5,
    })
    df, res = bright.normalise_columns(raw, bright.TMASS_ALIASES, keep=("source_id",))
    assert res["tmass_qflg"] == "Qfl" and res["tmass_rflg"] == "Rfl" and res["tmass_xflg"] == "X"
    pos = pd.DataFrame({"source_id": [11, 12, 13], "ra": [10.0, 20.0, 30.0], "dec": [1.0, 2.0, 3.0]})
    att = bright.attach_tmass(df, pos, 3.0)
    assert att["source_id"].dtype == np.int64 and att["source_id"].tolist() == [12, 11]
    assert att.loc[att["source_id"] == 11, "tmass_id"].iloc[0] == "J1"      # closest by angDist
    q_ok, known, read1 = bright.tmass_quality_masks(att, cfg["tmass"])
    assert known.all() and q_ok.tolist() == [False, True] and read1.tolist() == [True, False]
    # No flag column at all: unknown, not bad.
    q_ok2, known2, _ = bright.tmass_quality_masks(att.drop(columns=["tmass_qflg", "tmass_rflg"]),
                                                   cfg["tmass"])
    assert q_ok2.all() and not known2.any()


def test_xmatch_style_2mass_route_gives_a_2mass_anchor_end_to_end(cfg, tmp_path):
    sky = make_sky(n=400, seed=13)
    arch = FakeArchive(sky, cfg)
    arch.xmatch_style = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = bright.run_bright_stage(cfg, tmp_path, gaia_fetcher=arch.gaia, tmass_fetcher=arch.tmass,
                                    akari_fetcher=arch.akari, iras_fetcher=arch.iras)
    assert s["n_with_2mass"] >= 395 and s["n_with_2mass_quality_unknown"] == 0
    assert s["verdict"] == VERDICT_NULL
    cols = s["acquisition"]["tmass"]["columns_attached"]
    assert "tmass_qflg" in cols and "tmass_angdist" in cols and "source_id" in cols


# --------------------------------------------------------------------------
# Screen logic on a hand-built frame
# --------------------------------------------------------------------------
def _frame_for_screen(cfg, n=400, seed=2):
    rng = np.random.default_rng(seed)
    jk = rng.uniform(0.0, 0.9, n)
    ks = rng.uniform(4.5, 6.5, n)
    df = pd.DataFrame({"source_id": np.arange(n), "j_m": ks + jk, "ks_m": ks, "e_ks": 0.03,
                       "phot_g_mean_mag": ks + 1.5, "bp_rp": 0.5 + jk, "parallax": 10.0,
                       "phot_variable_flag": "NOT_AVAILABLE", "non_single_star": 0,
                       "ecl_lat": 5.0, "b": 30.0, "n_neigh_30": 0.0, "n_neigh_60": 0.0,
                       "lum_class": "dwarf", "has_akari": True, "has_iras": True,
                       "tmass_read1_regime": False})
    for b in BANDS:
        a, c = REL[b]
        m = ks - (a + c * jk) + rng.normal(0, 0.15, n)
        df[b] = ZP[b] * 10 ** (-0.4 * m)
        df[f"e_{b}"] = 0.05 * df[b]
        df[f"q_{b}"] = 3
        df[f"m_{b}"] = m
    return df


def test_akari_saturation_regime_moves_the_primary_band_to_iras(cfg):
    df = _frame_for_screen(cfg)
    loci = bright.fit_all_loci(df, cfg)
    # A star whose photosphere predicts 60 Jy at 9 um (Vega-like) with a deficit
    # in AKARI only: the AKARI deficit is exactly what saturation produces.
    df.loc[0, "ks_m"] = 0.0
    df.loc[0, "j_m"] = 0.3
    for b in BANDS:
        a, c = REL[b]
        df.loc[0, f"m_{b}"] = 0.0 - (a + c * 0.3)
        df.loc[0, b] = ZP[b] * 10 ** (-0.4 * df.loc[0, f"m_{b}"])
    df.loc[0, "s09"] *= 10 ** (-0.4 * 1.5)                 # AKARI 9 depressed, IRAS normal
    df.loc[0, "s18"] *= 10 ** (-0.4 * 1.5)
    res = residuals(df, loci, cfg)
    assert res.loc[0, "pred_s09_jy"] > cfg["akari"]["saturation_9_jy"]
    cands, flagged, counters = screen_bright(res, cfg)
    assert flagged.loc[0, "primary_band"] == "f12"
    assert not flagged.loc[0, "primary_deficit"]
    assert 0 not in set(cands["source_id"])
    # The same star with IRAS 12 AND 25 depressed too is a candidate on IRAS alone.
    df.loc[0, "f12"] *= 10 ** (-0.4 * 1.5)
    df.loc[0, "f25"] *= 10 ** (-0.4 * 1.5)
    res = residuals(df, loci, cfg)
    cands, flagged, counters = screen_bright(res, cfg)
    assert flagged.loc[0, "primary_deficit"] and 0 in set(cands["source_id"])
    assert "f25" in flagged.loc[0, "agreeing_bands"]
    assert counters["n_primary_akari_sat_regime"] == 1


def test_non_single_star_and_poor_akari_quality_are_vetoes(cfg):
    df = _frame_for_screen(cfg)
    loci = bright.fit_all_loci(df, cfg)
    for i in (1, 2):
        df.loc[i, "s09"] *= 10 ** (-0.4 * 1.2)
        df.loc[i, "s18"] *= 10 ** (-0.4 * 1.2)
    df.loc[1, "non_single_star"] = 1
    df.loc[2, "q_s09"] = 2
    res = residuals(df, loci, cfg)
    _, flagged, counters = screen_bright(res, cfg)
    assert flagged.loc[1, "veto"] == "non_single_star"
    assert flagged.loc[2, "veto"] == "poor_akari_quality"
    assert counters["non_single_star"] == 1 and counters["poor_akari_quality"] == 1


def test_luminosity_class_split_by_absolute_g(cfg):
    cls = bright.luminosity_class([5.0, 5.0, 5.0, np.nan], [1.2, 1.2, 0.5, 1.0],
                                  [1.0, 100.0, 1.0, 10.0], cfg["locus"])
    assert list(cls) == ["giant", "dwarf", "dwarf", "unknown"]


def test_cli_probe_and_run_write_summaries_offline(cfg, tmp_path, monkeypatch):
    """The CLI entry points, with the archives stubbed at the module level."""
    sky = make_sky(n=150, seed=12)
    arch = FakeArchive(sky, cfg)
    monkeypatch.setattr(bright, "default_gaia_fetcher", arch.gaia)
    monkeypatch.setattr(bright, "VizierSliceFetcher",
                        lambda table, aliases, ra_key, **kw: arch.akari if "297" in table else arch.iras)
    monkeypatch.setattr(bright, "VizierUploadMatcher", lambda c, **kw: arch.tmass)
    monkeypatch.setattr(bright, "load_bright_config", lambda p=None: cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert bright.main(["--stage", "run", "--out", str(tmp_path), "--max-targets", "60"]) == 0
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["n_targets"] == 60 and s["verdict"] in (VERDICT_NULL, VERDICT_CAND)

    def dead_vizier(query, **kw):
        raise RuntimeError("no egress")

    monkeypatch.setattr(bright, "run_vizier", dead_vizier)
    rep = bright.probe_bright(cfg, tmp_path / "probe", gaia_fetcher=arch.gaia)
    assert (tmp_path / "probe" / "probe.json").exists()
    assert rep["verdict"] == VERDICT_NO_DATA
    assert all(v["status"] == "QUERY_FAILED" for v in rep["tables"].values())

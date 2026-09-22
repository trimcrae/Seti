"""Offline tests for RING (S63).  No network: every fetcher is injected.

The suite must (a) recover an injected 500 K ring on a white-dwarf SED, (b)
classify a debris-disk-shaped and a companion-shaped excess as such, (c) trip
every rejection rule with a case, and (d) degrade to NO_DATA_REACHED on an
empty archive.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.photometry import band_freq_hz, flux_jy_to_mag, mag_to_flux_jy, planck_bnu
from seti.ring import acquire as racq
from seti.ring import assess as rass
from seti.ring import physics as ph
from seti.ring import run as rrun
from seti.ring import screen as rscr

_WBANDS = ("W1", "W2", "W3", "W4")


@pytest.fixture(scope="module")
def cfg():
    return rrun.load_ring_config()


# ==========================================================================
# Physics
# ==========================================================================

def test_blackbody_colour_inverts_to_temperature():
    for t in (300.0, 500.0, 700.0, 1500.0):
        c = ph.blackbody_colour(t)
        assert ph.colour_to_temperature(c) == pytest.approx(t, rel=0.02)
    # A 500 K ring is red in W1-W2 by more than a magnitude; a hot photosphere is ~0.
    assert ph.blackbody_colour(500.0) > 1.0
    assert abs(ph.blackbody_colour(10000.0)) < 0.1
    # Bluer than the Rayleigh-Jeans limit has no blackbody temperature.
    assert np.isnan(ph.colour_to_temperature(-1.0))


def test_ring_flux_is_bolometrically_consistent():
    """Integrated over frequency the ring flux must equal f L / 4 pi d^2."""
    l_w, f, t, d = 1e26, 0.1, 500.0, 100.0
    nu = np.geomspace(1e11, 1e16, 20000)
    from seti.photometry import C_LIGHT, H_PLANCK, K_BOLTZ
    x = H_PLANCK * nu / (K_BOLTZ * t)
    bnu = (2 * H_PLANCK * nu ** 3 / C_LIGHT ** 2) / np.expm1(x)
    bol = f * l_w / (4 * np.pi * (d * ph.PC_M) ** 2)
    fnu = bol * np.pi * bnu / (ph.SIGMA_SB * t ** 4)
    assert np.trapezoid(fnu, nu) == pytest.approx(bol, rel=1e-3)
    # The band value from the helper agrees with the direct evaluation at nu_W2.
    direct = bol * np.pi * float(planck_bnu(t, band_freq_hz("W2"))) / (ph.SIGMA_SB * t ** 4)
    assert ph.ring_flux_density_jy(l_w, f, t, d, "W2") == pytest.approx(direct * 1e26)


def test_sensitivity_and_radius_scale_correctly():
    f1 = ph.min_intercept_fraction(1e26, 500.0, 100.0, "W2", 7.6e-5)
    f2 = ph.min_intercept_fraction(1e26, 500.0, 200.0, "W2", 7.6e-5)
    assert f2 == pytest.approx(4.0 * f1, rel=1e-6)          # inverse-square
    r = ph.ring_equilibrium_radius_au(ph.L_SUN_W, 279.0)
    assert r == pytest.approx(1.0, rel=0.03)                 # Earth's blackbody radius
    assert ph.ring_temperature_k(ph.L_SUN_W, 1.0) == pytest.approx(279.0, rel=0.03)


def test_host_luminosities():
    # Crab: P = 33.4 ms, Pdot = 4.2e-13 -> Edot ~ 4.5e38 erg/s.
    edot = ph.spin_down_luminosity_w(0.0334, 4.2e-13)
    assert edot == pytest.approx(4.5e31, rel=0.15)
    assert np.isnan(ph.spin_down_luminosity_w(1.0, -1e-15))
    # A 0.6 Msun, log g = 8, 10 kK white dwarf: R ~ 0.0128 Rsun, L ~ 1.5e-3 Lsun.
    r = ph.wd_radius_m(8.0, 0.6) / ph.R_SUN_M
    assert r == pytest.approx(0.0128, rel=0.05)
    assert ph.wd_luminosity_w(1e4, 8.0, 0.6) / ph.L_SUN_W == pytest.approx(1.5e-3, rel=0.1)


def test_cooling_ceiling_is_a_ceiling():
    # 13 MJ at 100 Myr sits near log L ~ -4.7; the ceiling adds the margin.
    c = ph.planetary_cooling_ceiling_log_lsun(0.1, margin_dex=0.5)
    assert -4.9 + 0.5 < c < -4.4 + 0.5
    # Older is fainter; heavier is brighter.
    assert ph.planetary_cooling_ceiling_log_lsun(1.0) < ph.planetary_cooling_ceiling_log_lsun(0.1)
    assert ph.burrows_luminosity_lsun(0.05, 1.0) > ph.burrows_luminosity_lsun(0.02, 1.0)


def test_shape_classification(cfg):
    assert ph.classify_excess_shape(500.0, 0.01, cfg) == "ring_band"
    assert ph.classify_excess_shape(1200.0, 1e-3, cfg) == "debris_disk"
    assert ph.classify_excess_shape(2600.0, 0.5, cfg) == "companion"
    assert ph.classify_excess_shape(150.0, 0.01, cfg) == "cold_dust"
    assert ph.classify_excess_shape(1200.0, 5.0, cfg) == "warm_ambiguous"   # too bright for the locus
    assert ph.classify_excess_shape(np.nan, np.nan, cfg) == "unfit"


# ==========================================================================
# White-dwarf leg
# ==========================================================================

def _wd_mags(teff: float, ks: float) -> dict:
    """Blackbody photosphere magnitudes anchored at Ks."""
    omega = mag_to_flux_jy(ks, "Ks") / (np.pi * planck_bnu(teff, band_freq_hz("Ks")) * 1e26)
    out = {}
    for b in ("J", "H", "Ks", "W1", "W2", "W3", "W4", "G", "BP", "RP"):
        f = omega * np.pi * planck_bnu(teff, band_freq_hz(b)) * 1e26
        out[b] = float(flux_jy_to_mag(f, b))
    return out, omega


def _add_ring(mags: dict, omega_wd: float, teff: float, t_ring: float, tau: float) -> dict:
    """Add a blackbody ring of fractional luminosity ``tau`` in flux space."""
    om_ring = tau * omega_wd * (teff / t_ring) ** 4
    out = dict(mags)
    for b in _WBANDS:
        f = mag_to_flux_jy(out[b], b) + om_ring * np.pi * planck_bnu(t_ring, band_freq_hz(b)) * 1e26
        out[b] = float(flux_jy_to_mag(f, b))
    return out


def make_wd_sample(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dt = 2010.5 - 2016.0
    for i in range(n):
        teff = float(rng.uniform(7000.0, 20000.0))
        ks = float(rng.uniform(13.0, 15.5))
        mags, omega = _wd_mags(teff, ks)
        pmra, pmdec = float(rng.normal(0, 120)), float(rng.normal(0, 120))
        ra, dec = float(rng.uniform(20, 340)), float(rng.uniform(-60, 60))
        r = {"source_id": 1000 + i, "wd_name": f"WDJ{i:04d}", "ra": ra, "dec": dec,
             "l": 100.0, "b": 55.0, "parallax": 20.0, "parallax_error": 0.05,
             "parallax_over_error": 400.0, "pmra": pmra, "pmdec": pmdec,
             "pmra_error": 0.05, "pmdec_error": 0.05, "phot_g_mean_mag": mags["G"],
             "bp_rp": mags["BP"] - mags["RP"], "ruwe": 1.0, "pwd": 0.99,
             "teff_h": teff, "logg_h": 8.0, "mass_h": 0.6, "chisq_h": 1.0,
             "teff_he": np.nan, "logg_he": np.nan, "mass_he": np.nan, "chisq_he": np.nan,
             "ph_qual": "AAUU", "cc_flags": "0000", "ext_flag": 0,
             "number_of_neighbours": 1, "number_of_mates": 0, "wise_angdist": 0.3,
             "ra_wise": ra + (pmra * dt / 1000.0) / 3600.0 / np.cos(np.radians(dec)),
             "dec_wise": dec + (pmdec * dt / 1000.0) / 3600.0,
             "_omega": omega, "_teff": teff}
        for b in ("J", "H", "Ks"):
            r[f"{b}mag"] = mags[b] + rng.normal(0, 0.02)
            r[f"e_{b}mag"] = 0.02
        r["W1mag"] = mags["W1"] + rng.normal(0, 0.03)
        r["e_W1mag"] = 0.03
        r["W2mag"] = mags["W2"] + rng.normal(0, 0.05)
        r["e_W2mag"] = 0.05
        r["W3mag"] = mags["W3"] - 3.0        # an upper limit (ph_qual U)
        r["e_W3mag"] = np.nan
        r["W4mag"] = mags["W4"] - 4.0
        r["e_W4mag"] = np.nan
        r["_mags"] = mags
        rows.append(r)
    return pd.DataFrame(rows)


def _inject(df: pd.DataFrame, i: int, t_ring: float, tau: float) -> None:
    mags = _add_ring(df.loc[i, "_mags"], df.loc[i, "_omega"], df.loc[i, "_teff"], t_ring, tau)
    for b in ("W1", "W2"):
        df.loc[i, f"{b}mag"] = mags[b]


def _screen(df: pd.DataFrame, cfg: dict):
    work = racq.harmonise_wd(df.drop(columns=["_mags", "_omega", "_teff"]))
    return rscr.screen_wd(work, cfg, rng=np.random.default_rng(1))


def test_clean_white_dwarfs_produce_no_ring_candidates(cfg):
    df = make_wd_sample()
    out, s = _screen(df, cfg)
    assert s["status"] == "OK"
    assert s["sed_anchor_counts"].get("nir", 0) == len(df)
    assert s["n_excess_flagged"] <= 0.02 * len(df)
    assert s["n_ring_candidates"] == 0
    assert (out["W3_detected"] == False).all()          # noqa: E712  upper limits blanked
    assert out["W3mag"].isna().all()


def test_injected_500K_ring_is_recovered_and_classified(cfg):
    df = make_wd_sample()
    _inject(df, 0, t_ring=500.0, tau=0.02)
    out, s = _screen(df, cfg)
    row = out.iloc[0]
    assert bool(row["excess_flag"]), "the ring excess was not flagged"
    assert 400.0 < float(row["t_ring_k"]) < 650.0, row["t_ring_k"]
    assert float(row["t_ring_lo_k"]) <= float(row["t_ring_k"]) <= float(row["t_ring_hi_k"])
    assert float(row["tau_ring"]) == pytest.approx(0.02, rel=0.5)
    assert row["shape_class"] == "ring_band"
    assert row["verdict"] == "surviving", row["gate_reason"]
    assert bool(row["ring_candidate"])
    assert s["n_ring_candidates"] == 1
    # The equilibrium radius for that temperature around that white dwarf is tiny.
    assert 0.001 < float(row["ring_radius_au"]) < 0.2


def test_debris_disk_and_companion_shapes_are_named(cfg):
    df = make_wd_sample()
    _inject(df, 1, t_ring=1300.0, tau=3e-3)    # a WD debris disk
    _inject(df, 2, t_ring=2800.0, tau=0.3)     # an unresolved cool companion
    out, s = _screen(df, cfg)
    disk, comp = out.iloc[1], out.iloc[2]
    assert bool(disk["excess_flag"]) and disk["shape_class"] == "debris_disk"
    assert not bool(disk["ring_candidate"])
    assert bool(comp["excess_flag"]) and comp["shape_class"] == "companion"
    assert comp["verdict"] == "rejected" and comp["gate_reason"] == "unresolved_companion"
    assert s["shape_counts"].get("debris_disk", 0) >= 1
    assert s["shape_counts"].get("companion", 0) >= 1
    assert s["n_ring_candidates"] == 0


def test_ledger_rules_trip_on_a_white_dwarf(cfg):
    df = make_wd_sample()
    # A blend: negative W1-W2 with a formally significant W1 excess.
    df.loc[3, "W1mag"] = df.loc[3, "W1mag"] - 1.0
    # A static background source under a fast white dwarf.
    _inject(df, 4, t_ring=500.0, tau=0.02)
    df.loc[4, "pmra"], df.loc[4, "pmdec"] = 600.0, -400.0
    df.loc[4, "ra_wise"], df.loc[4, "dec_wise"] = df.loc[4, "ra"], df.loc[4, "dec"]
    # An ambiguous cross-match (a mate).
    _inject(df, 5, t_ring=500.0, tau=0.02)
    df.loc[5, "number_of_mates"] = 2
    out, _ = _screen(df, cfg)
    assert not bool(out.iloc[3]["ring_candidate"])
    if bool(out.iloc[3]["excess_flag"]):
        assert out.iloc[3]["gate_reason"] == "ledger"
    assert out.iloc[4]["gate_reason"] == "astrometric_registration"
    assert out.iloc[5]["gate_reason"] == "wise_quality"


def test_empty_wd_sample_degrades_honestly(cfg):
    out, s = rscr.screen_wd(pd.DataFrame(), cfg)
    assert s["status"] == "NO_DATA_REACHED" and len(out) == 0


def test_wd_followup_requires_every_test_to_be_made(cfg):
    df = make_wd_sample(40)
    _inject(df, 0, t_ring=500.0, tau=0.02)
    out, _ = _screen(df, cfg)
    short = out[out["ring_candidate"].astype(bool)]
    assert len(short) == 1

    def xm_ok(pos, table, radius):
        # CatWISE co-moving with Gaia (arcsec/yr in the VizieR convention).
        return pd.DataFrame({"source_id": pos["source_id"], "angDist": 0.2,
                             "pmRA": short["pmra"].to_numpy() / 1000.0,
                             "pmDE": short["pmdec"].to_numpy() / 1000.0,
                             "e_pmRA": 0.01, "e_pmDE": 0.01})

    def xm_static(pos, table, radius):
        return pd.DataFrame({"source_id": pos["source_id"], "angDist": 0.2,
                             "pmRA": 0.0, "pmDE": 0.0, "e_pmRA": 0.01, "e_pmDE": 0.01})

    fu = rass.wd_followup(short, cfg, fetch_known_disks=lambda p: set(),
                          fetch_neighbours=lambda ra, dec: pd.DataFrame(),
                          fetch_simbad=lambda p: pd.DataFrame(), xmatch_fn=xm_ok)
    assert fu["followup_verdict"].iloc[0] == "surviving", fu["followup_reason"].iloc[0]
    fu = rass.wd_followup(short, cfg, fetch_known_disks=lambda p: set(),
                          fetch_neighbours=lambda ra, dec: pd.DataFrame(),
                          fetch_simbad=lambda p: pd.DataFrame(), xmatch_fn=xm_static)
    assert fu["followup_reason"].iloc[0] == "background_source_no_comovement"
    # A known debris disk is subtracted; a service failure leaves a test untested.
    fu = rass.wd_followup(short, cfg, fetch_known_disks=lambda p: {int(short["source_id"].iloc[0])},
                          fetch_neighbours=lambda ra, dec: pd.DataFrame(),
                          fetch_simbad=lambda p: pd.DataFrame(), xmatch_fn=xm_ok)
    assert fu["followup_reason"].iloc[0] == "known_debris_disk"

    def boom(*a, **k):
        raise RuntimeError("archive down")
    fu = rass.wd_followup(short, cfg, fetch_known_disks=boom, fetch_neighbours=boom,
                          fetch_simbad=boom, xmatch_fn=boom)
    assert fu["followup_reason"].iloc[0] == "comovement_untested"


# ==========================================================================
# Pulsar leg
# ==========================================================================

_PSRCAT = """#CATALOGUE 2.6.2
PSRJ     J1300+1240                    wdf92
PSRB     B1257+12                      wdf92
RAJ      13:00:03.5767                 4    kon03
DECJ     +12:40:55.1927                7    kon03
PMRA     46.44                         0.09  kon03
PMDEC    -84.87                        0.11  kon03
POSEPOCH 49750.0
P0       0.006218531938                1    kon03
P1       1.1433E-19                    8    kon03
DIST_DM  0.71
ASSOC    PLANET
BINARY   ??
@-----------------------------------------------------------------
PSRJ     J0534+2200                    ls68
PSRB     B0531+21                      ls68
RAJ      05:34:31.973                  5    lpg+95
DECJ     +22:00:52.06                  6    lpg+95
POSEPOCH 40675
P0       0.0333924123                  5    lpg+95
P1       4.209E-13                     1    lpg+95
DIST_A   2.0
ASSOC    SNR:Crab,PWN
@-----------------------------------------------------------------
PSRJ     J1623-2631
PSRB     B1620-26
RAJ      16:23:38.2218                 2
DECJ     -26:31:53.769                 4
P0       0.011075750
P1       -6.7E-19
DIST_A   1.85
ASSOC    GC:M4,PLANET
BINARY   DD
BINCOMP  CO
@-----------------------------------------------------------------
PSRJ     J0000+0001
RAJ      00:00:10.0                    1
DECJ     +00:01:00
P0       1.0
P1       1.0E-15
@-----------------------------------------------------------------
PSRJ     J0001+0002
RAJ      00:01:10.00                   1
DECJ     +00:02:00.0                   1
P0       0.5
P1       1.0E-14
DIST_DM  1.0
@-----------------------------------------------------------------
"""


def test_psrcat_parser_and_normalisation():
    raw = racq.parse_psrcat_db(_PSRCAT)
    psr = racq.normalise_pulsars(raw, "tarball")
    assert list(psr["jname"]) == ["J1300+1240", "J0534+2200", "J1623-2631", "J0000+0001",
                                  "J0001+0002"]
    b = psr.set_index("jname").loc["J1300+1240"]
    assert b["ra"] == pytest.approx(195.0149, abs=1e-3)
    assert b["dec"] == pytest.approx(12.68200, abs=1e-3)
    assert b["pmra"] == pytest.approx(46.44) and b["pmdec"] == pytest.approx(-84.87)
    assert 1994.0 < b["posepoch_yr"] < 1996.0
    assert b["pos_err_arcsec"] < 0.01                       # milliarcsecond timing position
    assert b["dist_kpc"] == pytest.approx(0.71)
    # A position given to 0.1 s of time and whole arcseconds is known to arcseconds.
    c = psr.set_index("jname").loc["J0000+0001"]
    assert 1.0 < c["pos_err_arcsec"] < 3.0
    assert psr.set_index("jname").loc["J0534+2200"]["bname"] == "B0531+21"


def test_control_offsets_ring_the_target():
    raw = racq.parse_psrcat_db(_PSRCAT)
    psr = racq.normalise_pulsars(raw, "tarball")
    cfg = rrun.load_ring_config()
    pos = racq.pulsar_match_positions(psr, cfg, 2010.5)
    assert len(pos) == len(psr) * 17
    tgt = pos[pos["source_id"] == "J1300+1240|t"].iloc[0]
    # Propagated ~15.5 yr at -84.87 mas/yr in Dec: -1.3 arcsec.
    assert (tgt["dec"] - 12.68200) * 3600.0 == pytest.approx(-1.32, abs=0.1)
    c0 = pos[pos["source_id"] == "J1300+1240|c0"].iloc[0]
    sep = np.hypot((c0["ra"] - tgt["ra"]) * np.cos(np.radians(tgt["dec"])),
                   c0["dec"] - tgt["dec"]) * 3600.0
    assert sep == pytest.approx(60.0, rel=0.01)


def _pulsar_fixture():
    raw = racq.parse_psrcat_db(_PSRCAT)
    psr = racq.normalise_pulsars(raw, "tarball")
    ring_colour = float(ph.blackbody_colour(500.0))
    rows = [
        # J0001+0002: a clean 500 K counterpart, no control hits.
        {"source_id": "J0001+0002|t", "match_dist_arcsec": 0.8, "designation": "J000110.00+000200.0",
         "W1mag": 15.0, "e_W1mag": 0.05, "W2mag": 15.0 - ring_colour, "e_W2mag": 0.08,
         "ph_qual": "AAUU", "cc_flags": "0000", "ext_flag": 0},
        # J0000+0001: a counterpart in a crowded field -- 12 of 16 controls hit.
        {"source_id": "J0000+0001|t", "match_dist_arcsec": 1.5, "designation": "x",
         "W1mag": 14.0, "e_W1mag": 0.05, "W2mag": 13.9, "e_W2mag": 0.05,
         "ph_qual": "AAAA", "cc_flags": "0000", "ext_flag": 0},
        # Crab: a bright counterpart (the nebula) -- catalogued, vetoed.
        {"source_id": "J0534+2200|t", "match_dist_arcsec": 0.5, "designation": "crab",
         "W1mag": 9.0, "e_W1mag": 0.02, "W2mag": 8.0, "e_W2mag": 0.02,
         "ph_qual": "AAAA", "cc_flags": "0000", "ext_flag": 5},
    ]
    for k in range(12):
        rows.append({"source_id": f"J0000+0001|c{k}", "match_dist_arcsec": 1.0,
                     "designation": f"c{k}", "W1mag": 15.0, "e_W1mag": 0.1,
                     "W2mag": 14.8, "e_W2mag": 0.1, "ph_qual": "AAUU", "cc_flags": "0000",
                     "ext_flag": 0})
    return psr, {"allwise": pd.DataFrame(rows), "catwise": pd.DataFrame()}


def test_pulsar_screen_measures_colour_chance_and_provenance(cfg):
    psr, matches = _pulsar_fixture()
    out, s = rscr.screen_pulsars(psr, matches, cfg)
    o = out.set_index("jname")
    good = o.loc["J0001+0002"]
    assert bool(good["allwise_match"]) and good["allwise_n_control_hits"] == 0
    assert good["allwise_p_chance"] < cfg["pulsar"]["chance_p_max"]
    assert good["t_colour_k"] == pytest.approx(500.0, rel=0.05)
    assert good["shape_class"] == "ring_band" and good["verdict"] == "surviving"
    assert bool(good["ring_candidate"])
    crowded = o.loc["J0000+0001"]
    assert crowded["allwise_n_control_hits"] == 12
    assert crowded["veto_reason"] == "chance_coincidence"
    crab = o.loc["J0534+2200"]
    assert crab["veto_reason"] == "catalogued_counterpart"
    assert bool(crab["assoc_veto"])
    planets = o.loc["J1300+1240"]
    assert planets["verdict"] == "no_counterpart"
    m4 = o.loc["J1623-2631"]
    assert bool(m4["globular_cluster"]) and bool(m4["companion_veto"])
    # Sensitivity: the Crab's spin-down power makes a 500 K ring visible at f << 1.
    assert crab["f_min_500K_W2"] < 1e-3
    assert np.isfinite(crab["ring_radius_500K_au"]) and crab["ring_radius_500K_au"] > 10.0
    assert s["n_ring_candidates"] == 1 and s["n_with_any_counterpart"] == 3
    assert s["veto_reasons"]["chance_coincidence"] == 1


def test_two_target_vet_reports_both_systems(cfg):
    psr, matches = _pulsar_fixture()
    out, _ = rscr.screen_pulsars(psr, matches, cfg)
    tv = rass.two_target_vet(out, cfg)
    assert [t["jname"] for t in tv] == ["J1300+1240", "J1623-2631"]
    assert all(t["status"] == "VETTED" for t in tv)
    assert tv[0]["verdict"] == "no_counterpart"
    assert tv[0]["f_min_500K_W2"] is not None
    assert rass.two_target_vet(pd.DataFrame(), cfg)[0]["status"] == "NO_DATA_REACHED"


def test_empty_pulsar_table_degrades_honestly(cfg):
    out, s = rscr.screen_pulsars(pd.DataFrame(), {}, cfg)
    assert s["status"] == "NO_DATA_REACHED"


# ==========================================================================
# Brown-dwarf leg
# ==========================================================================

def _bd_epochs(rng, sid, n, switch: bool):
    t = 2014.0 + np.arange(n) * 0.5
    base = 14.0 + rng.normal(0, 0.02, n)
    if switch:
        state = (np.arange(n) % 3 == 0)
        base = base - 0.6 * state
    rows = []
    for band, off, noise in (("W1", 2.5, 0.06), ("W2", 0.0, 0.02)):
        m = base + off + rng.normal(0, noise, n) if band == "W2" else \
            14.0 + off + rng.normal(0, noise, n)
        for k in range(n):
            rows.append({"source_id": sid, "band": band, "epoch": k, "t_mjd": 0.0,
                         "t_yr": t[k], "mag": m[k], "err": noise, "n_exp": 12,
                         "scatter": noise})
    return pd.DataFrame(rows)


def test_bd_duty_cycle_flags_a_switching_w2_series_only(cfg):
    rng = np.random.default_rng(4)
    targets = pd.DataFrame({"source_id": [f"Y{i}" for i in range(12)], "spt": "Y0",
                            "spt_num": 30.0})
    eps = [_bd_epochs(rng, f"Y{i}", 16, switch=(i == 0)) for i in range(10)]
    eps.append(_bd_epochs(rng, "Y10", 3, switch=True))         # too few epochs
    epochs = pd.concat(eps, ignore_index=True)                   # Y11: no epochs
    out, s = rscr.screen_bd(epochs, targets, cfg)
    o = out.set_index("source_id")
    assert bool(o.loc["Y0", "duty_cycle_flag"])
    assert o.loc["Y0", "w2_duty_cycle_high"] == pytest.approx(1 / 3, abs=0.1)
    assert o.loc["Y0", "w2_two_state_sep_mag"] == pytest.approx(0.6, abs=0.1)
    assert not o.loc[[f"Y{i}" for i in range(1, 10)], "duty_cycle_flag"].any()
    assert o.loc["Y10", "status"] == "TOO_FEW_EPOCHS"
    assert o.loc["Y11", "status"] == "NO_EPOCHS"
    assert s["n_duty_cycle_flags"] == 1 and s["n_tested"] == 10
    assert s["flagged"][0]["source_id"] == "Y0"


def test_bd_no_epochs_degrades(cfg):
    targets = pd.DataFrame({"source_id": ["Y0"], "spt": "Y0", "spt_num": 30.0})
    out, s = rscr.screen_bd(pd.DataFrame(), targets, cfg)
    assert s["status"] == "NO_DATA_REACHED" and s["n_with_epochs"] == 0


def test_spt_parsing_and_role_resolution():
    assert racq.spt_to_numeric("T7.5") == 27.5
    assert racq.spt_to_numeric(">=Y1") == 31.0
    assert racq.spt_to_numeric("L3 gamma") == 13.0
    assert np.isnan(racq.spt_to_numeric("--"))
    roles = racq.resolve_roles(["Name", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "plx", "SpT",
                                "W1mag", "W2mag"], racq.BD_ROLES)
    assert roles["ra"] == "RAJ2000" and roles["dec"] == "DEJ2000" and roles["spt"] == "SpT"
    assert roles["pmra"] == "pmRA" and roles["w2"] == "W2mag"


# ==========================================================================
# Free-floating planets
# ==========================================================================

def test_ffp_screen_flags_only_planetary_mass_objects_above_the_ceiling(cfg):
    df = pd.DataFrame({
        "name": ["hot-planet", "normal-planet", "hot-bd", "no-age", "group-age"],
        "spt": ["L4", "L7", "L2", "T2", "L5"],
        "group": ["TWA", "TWA", "TWA", "", "beta Pic"],
        "age": [10.0, 10.0, 10.0, np.nan, np.nan],           # Myr
        "lbol": [-2.6, -4.0, -2.6, -3.0, -3.3],
        "mass": [8.0, 8.0, 40.0, 8.0, 10.0],                 # M_J
    })
    out, s = rscr.screen_ffp(df, cfg)
    o = out.set_index("name")
    assert bool(o.loc["hot-planet", "flag"])
    assert o.loc["hot-planet", "systematics_not_excluded"].startswith("age_misassignment")
    assert not bool(o.loc["normal-planet", "flag"])
    assert bool(o.loc["hot-bd", "hotter_than_cooling"]) and not bool(o.loc["hot-bd", "flag"])
    assert not bool(o.loc["no-age", "testable"])
    assert o.loc["group-age", "age_source"] == "group_fallback"
    assert o.loc["group-age", "age_gyr"] == pytest.approx(0.024)
    assert s["n_flags_planetary_and_hot"] == 1 and s["n_testable"] == 4


# ==========================================================================
# Stages end to end, offline
# ==========================================================================

def _fetchers(empty: bool = False):
    if empty:
        return {"wd": lambda d, c: (pd.DataFrame(), {"route": "none"}),
                "pulsar": lambda c: (pd.DataFrame(), {"route": "none"}),
                "bd_targets": lambda c: (pd.DataFrame(), {"route": "none"}),
                "ffp": lambda c: (pd.DataFrame(), {"route": "none"})}
    psr, matches = _pulsar_fixture()
    wd = make_wd_sample(60).drop(columns=["_mags", "_omega", "_teff"])
    rng = np.random.default_rng(9)
    targets = pd.DataFrame({"source_id": ["Y0", "Y1"], "name": ["Y0", "Y1"], "spt": "Y0",
                            "ra": [10.0, 20.0], "dec": [0.0, 0.0], "pmra": 0.0, "pmdec": 0.0,
                            "spt_num": 30.0})

    def bd_neowise(t, d, c, shard, n):
        ep = pd.concat([_bd_epochs(rng, "Y0", 16, False), _bd_epochs(rng, "Y1", 16, True)])
        ep.to_csv(d / f"epochs_bd_shard{shard}.csv", index=False)
        return {"route": "injected", "n_ok": 2}

    ffp = pd.DataFrame({"name": ["a"], "spt": ["L4"], "group": ["TWA"], "age": [10.0],
                        "lbol": [-3.5], "mass": [8.0]})
    return {"wd": lambda d, c: (wd, {"route": "injected"}),
            "pulsar": lambda c: (psr, {"route": "injected"}),
            "pulsar_matches": lambda p, c: (matches, {"allwise": {"status": "OK"}}),
            "bd_targets": lambda c: (targets, {"route": "injected"}),
            "bd_neowise": bd_neowise,
            "ffp": lambda c: (ffp, {"route": "injected"})}


def test_stages_end_to_end_offline(cfg, tmp_path):
    out = tmp_path / "ring"
    f = _fetchers()
    for leg in rrun.LEGS:
        rrun.stage_acquire(cfg, out, leg, fetchers=f)
        rrun.stage_screen(cfg, out, leg, rng=np.random.default_rng(1))
    s = rrun.stage_assess(cfg, out, followup=True, followup_fetchers={
        "fetch_known_disks": lambda p: set(),
        "fetch_neighbours": lambda ra, dec: pd.DataFrame(),
        "fetch_simbad": lambda p: pd.DataFrame(),
        "xmatch_fn": lambda pos, t, r: pd.DataFrame()})
    assert (out / "summary.json").exists() and (out / "REPORT.md").exists()
    js = json.loads((out / "summary.json").read_text())
    assert js["verdict"].startswith("RING_CANDIDATES_PENDING_VET")   # the pulsar fixture
    assert js["legs"]["pulsar"]["n_ring_candidates"] == 1
    assert js["legs"]["bd"]["n_duty_cycle_flags"] == 1
    assert js["legs"]["ffp"]["n_flags_planetary_and_hot"] == 0
    assert js["two_target_vet"][0]["status"] == "VETTED"
    assert (out / "pulsar" / "counterparts.csv").exists()
    assert (out / "wd" / "screened.parquet").exists()
    assert "Rings around the dead" in (out / "REPORT.md").read_text()


def test_empty_archives_give_no_data_reached(cfg, tmp_path):
    out = tmp_path / "ring"
    f = _fetchers(empty=True)
    for leg in rrun.LEGS:
        rrun.stage_acquire(cfg, out, leg, fetchers=f)
        rrun.stage_screen(cfg, out, leg)
    s = rrun.stage_assess(cfg, out, followup=False)
    assert s["verdict"] == "NO_DATA_REACHED"
    assert set(s["degraded_legs"]) == set(rrun.LEGS)


def test_a_missing_leg_is_named_not_hidden(cfg, tmp_path):
    out = tmp_path / "ring"
    f = _fetchers()
    for leg in ("pulsar", "ffp"):
        rrun.stage_acquire(cfg, out, leg, fetchers=f)
        rrun.stage_screen(cfg, out, leg)
    s = rrun.stage_assess(cfg, out, followup=False)
    assert s["verdict"].startswith("DEGRADED (")
    assert "wd" in s["degraded_legs"] and "bd" in s["degraded_legs"]
    assert "RING_CANDIDATES_PENDING_VET" in s["verdict"]

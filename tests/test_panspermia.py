"""Tests for the K2-18 panspermia close-encounter channel.

Decisive checks: (1) 6D phase space round-trips (observables -> Galactic 6D and
back); (2) a co-moving pair has ~zero relative velocity in the Galactic frame;
(3) linear closest-approach recovers an injected flyby's time, distance and
relative speed; (4) the transfer score prefers closer/slower *past* encounters and
zeroes future ones; (5) an injected close-slow recipient is recovered at the top
of the shortlist by the full run, among random background stars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from seti.panspermia.encounters import (
    closest_approach,
    flag_comoving,
    regime_summary,
    transfer_regime,
    transfer_score,
)
from seti.panspermia.kinematics import _A_ICRS_TO_GAL, _K_AUYR_KMS, phase_space_6d
from seti.panspermia.run import panspermia_run


def _observables_from_galactic(X, Y, Z, U, V, W, source_id=1):
    """Inverse of phase_space_6d: Galactic 6D (pc, km/s) -> Gaia observables.

    Lets a test place a star at any physical state and check the forward pipeline
    reconstructs it."""
    At = _A_ICRS_TO_GAL.T
    r_icrs = At @ np.array([X, Y, Z], float)
    v_icrs = At @ np.array([U, V, W], float)
    dist = np.linalg.norm(r_icrs)
    r_hat = r_icrs / dist
    dec = np.arcsin(r_hat[2])
    ra = np.arctan2(r_hat[1], r_hat[0]) % (2 * np.pi)
    ca, sa, cd, sd = np.cos(ra), np.sin(ra), np.cos(dec), np.sin(dec)
    a_hat = np.array([-sa, ca, 0.0])
    d_hat = np.array([-sd * ca, -sd * sa, cd])
    rv = float(v_icrs @ r_hat)
    v_a = float(v_icrs @ a_hat)
    v_d = float(v_icrs @ d_hat)
    scale = _K_AUYR_KMS * dist / 1000.0
    return {
        "source_id": source_id, "ra": np.degrees(ra), "dec": np.degrees(dec),
        "parallax": 1000.0 / dist, "pmra": v_a / scale, "pmdec": v_d / scale,
        "radial_velocity": rv, "phot_g_mean_mag": 10.0, "bp_rp": 1.5,
    }


def test_phase_space_roundtrip():
    truth = dict(X=12.0, Y=-30.0, Z=5.0, U=-11.0, V=7.0, W=-3.0)
    obs = _observables_from_galactic(**truth)
    ps = phase_space_6d(pd.DataFrame([obs])).iloc[0]
    for k, key in [("X", "X_pc"), ("Y", "Y_pc"), ("Z", "Z_pc"),
                   ("U", "U_kms"), ("V", "V_kms"), ("W", "W_kms")]:
        assert np.isclose(ps[key], truth[k], rtol=1e-6, atol=1e-6)


def test_missing_rv_gives_nan_velocity():
    obs = _observables_from_galactic(10, 10, 10, 5, 5, 5)
    obs["radial_velocity"] = np.nan
    ps = phase_space_6d(pd.DataFrame([obs])).iloc[0]
    assert np.isfinite(ps["X_pc"]) and not np.isfinite(ps["U_kms"])


def test_comoving_pair_zero_relative_velocity():
    anchor_obs = _observables_from_galactic(10, -20, 4, -8, 12, -1, source_id=0)
    mate_obs = _observables_from_galactic(11, -19, 4.5, -8, 12, -1, source_id=1)
    a = phase_space_6d(pd.DataFrame([anchor_obs])).iloc[0]
    anchor = {k: float(a[k]) for k in ("X_pc", "Y_pc", "Z_pc", "U_kms", "V_kms", "W_kms")}
    df = phase_space_6d(pd.DataFrame([mate_obs]))
    enc = closest_approach(anchor, df).iloc[0]
    assert enc["v_rel_kms"] < 1e-6
    assert enc["sep_now_pc"] < 2.0


def test_closest_approach_recovers_injected_flyby():
    # Anchor at rest at the origin (Galactic frame).
    anchor = dict(X_pc=0.0, Y_pc=0.0, Z_pc=0.0, U_kms=0.0, V_kms=0.0, W_kms=0.0)
    # A star that grazed to d_min=0.2 pc at t=-4 Myr, moving in +X at 15 km/s.
    kms_to_pcmyr = 1.0227121651
    vx_kms = 15.0
    vx_pcmyr = vx_kms * kms_to_pcmyr
    t_enc_true = -4.0            # Myr (past)
    d_min_true = 0.2            # pc, offset along Y at closest approach
    # position now (t=0): closest-approach point + velocity * (0 - t_enc)
    x_now = 0.0 + vx_pcmyr * (0.0 - t_enc_true)
    obs = _observables_from_galactic(x_now, d_min_true, 0.0, vx_kms, 0.0, 0.0)
    df = phase_space_6d(pd.DataFrame([obs]))
    enc = closest_approach(anchor, df).iloc[0]
    assert np.isclose(enc["t_enc_myr"], t_enc_true, atol=1e-3)
    assert np.isclose(enc["d_min_pc"], d_min_true, atol=1e-3)
    assert np.isclose(enc["v_rel_kms"], vx_kms, atol=1e-3)


def test_transfer_score_orders_and_gates():
    df = pd.DataFrame({
        "d_min_pc":  [0.1, 1.0, 0.1, 0.1],
        "v_rel_kms": [1.0, 1.0, 5.0, 1.0],
        "t_enc_myr": [-2.0, -2.0, -2.0, +2.0],   # last is a FUTURE encounter
    })
    scored = transfer_score(df, t_max_myr=10.0)
    s = scored["transfer_score"].to_numpy()
    assert s[0] > s[1]          # closer beats farther
    assert s[0] > s[2]          # slower beats faster
    assert s[3] == 0.0          # future encounter scores zero
    # Out-of-window past encounter is also gated to zero.
    far = transfer_score(pd.DataFrame({"d_min_pc": [0.1], "v_rel_kms": [1.0],
                                       "t_enc_myr": [-50.0]}), t_max_myr=10.0)
    assert far["transfer_score"].iloc[0] == 0.0


def test_flag_comoving():
    df = pd.DataFrame({"v_rel_kms": [0.5, 10.0, 2.0], "sep_now_pc": [3.0, 1.0, 40.0]})
    out = flag_comoving(df, v_rel_max_kms=3.0, sep_now_max_pc=5.0)
    assert list(out["comoving"]) == [True, False, False]


def test_transfer_regime_capture_vs_interception():
    # Three past encounters: a slow near-graze, a fast direct hit, a fast far pass.
    df = pd.DataFrame({
        "v_rel_kms": [0.05, 80.0, 80.0],
        "d_min_pc":  [1.0, 0.001, 1.0],
        "t_enc_myr": [-1.0, -1.0, -1.0],
    })
    reg = transfer_regime(df, donor_mass_msun=0.36, reservoir_pc=0.2)
    # Row 0: within the reservoir (1 pc > 0.2? no) -- 1 pc is OUTSIDE 0.2 pc, so
    # no material there; slow but nothing to capture -> no transfer.
    assert not reg["within_reservoir"].iloc[0] and not reg["transfers"].iloc[0]
    # Row 1: a direct hit (0.001 pc, inside the reservoir) but 80 km/s is far too
    # fast to bind material at that distance -> no transfer.
    assert reg["within_reservoir"].iloc[1] and not reg["capturable"].iloc[1]
    assert not reg["transfers"].iloc[1]
    # Row 2: fast AND far -> neither condition.
    assert not reg["transfers"].iloc[2]
    # Focusing collapses to ~1 for the fast passes.
    assert reg["focusing_factor"].iloc[1] < 1.01

    # A genuinely slow, close pass DOES transfer (both conditions met).
    good = transfer_regime(pd.DataFrame({"v_rel_kms": [0.01], "d_min_pc": [0.001],
                                         "t_enc_myr": [-1.0]}),
                           donor_mass_msun=0.36, reservoir_pc=0.2)
    assert good["transfers"].iloc[0]


def test_regime_summary_counts_only_past():
    df = pd.DataFrame({
        "v_rel_kms": [0.05, 80.0],
        "d_min_pc":  [1.0, 1.0],
        "t_enc_myr": [-1.0, +1.0],      # second is a FUTURE encounter -> excluded
    })
    s = regime_summary(df, donor_mass_msun=0.36, reservoir_pc=0.2)
    assert s["n_past"] == 1
    # The single past encounter is slow but at 1 pc (outside 0.2 pc reservoir):
    # nothing to capture there, so no transfer.
    assert s["n_within_reservoir"] == 0
    assert s["n_transfers"] == 0


def test_crossing_times_and_full_connectivity():
    from seti.panspermia.reachability import crossing_times
    df = pd.DataFrame({"d_min_pc": [0.9033]})     # the real closest approach
    ct = crossing_times(df, speeds_c=(0.001, 0.01, 0.1))
    # 0.9 pc at 0.1c ~ 29 yr; at 0.001c ~ 2946 yr -- trivial on stellar timescales.
    assert 25 < ct["cross_yr_0.1c"].iloc[0] < 35
    assert 2500 < ct["cross_yr_0.001c"].iloc[0] < 3500


def test_destination_quality_prefers_cool_ms_for_hycean():
    from seti.panspermia.reachability import destination_quality
    # At 10 pc, abs_G == apparent G. Use realistic MS absolute mags: M2.5 dwarf
    # ~11.3, Sun-analog ~4.67; a red giant at the same M-dwarf colour is far
    # over-luminous (abs_G ~ 1).
    df = pd.DataFrame({
        "phot_g_mean_mag": [11.3, 4.67, 1.0, 4.67],
        "dist_pc":         [10.0, 10.0, 10.0, 10.0],
        "bp_rp":           [2.6, 0.82, 2.6, 0.82],  # M dwarf, Sun, M-colour giant, Sun
    })
    q = destination_quality(df, target="hycean")
    assert q["lum_class"].iloc[0] == "main_sequence"
    assert q["lum_class"].iloc[2] == "giant"
    # Hycean prior: the cool M dwarf outscores the Sun-like G dwarf and the giant.
    assert q["dest_score"].iloc[0] > q["dest_score"].iloc[3]
    assert q["dest_score"].iloc[0] > q["dest_score"].iloc[2]


def test_hycean_vs_classical_prior_differ():
    from seti.panspermia.reachability import destination_quality
    df = pd.DataFrame({"phot_g_mean_mag": [11.3, 4.67], "dist_pc": [10.0, 10.0],
                       "bp_rp": [2.6, 0.82]})       # M dwarf vs Sun-analog
    hy = destination_quality(df, target="hycean")["dest_score"].to_numpy()
    cl = destination_quality(df, target="classical")["dest_score"].to_numpy()
    assert hy[0] > hy[1]        # hycean prior favours the M dwarf
    assert cl[1] > cl[0]        # classical prior favours the Sun-analog


def test_exohost_hycean_crossmatch():
    from seti.panspermia.exohosts import crossmatch_hosts
    neighbors = pd.DataFrame({"source_id": [111, 222, 333],
                              "ra": [10.0, 20.0, 30.0], "dec": [0.0, 0.0, 0.0]})
    planets = pd.DataFrame({
        "hostname": ["Hy A", "Rocky B"],
        "gaia_id": ["Gaia DR3 111", "Gaia DR3 222"],
        "ra": [10.0, 20.0], "dec": [0.0, 0.0], "sy_dist": [12.0, 15.0],
        "pl_name": ["Hy A b", "Rocky B b"],
        "pl_rade": [2.4, 1.0],           # sub-Neptune (hycean) vs Earth-size
        "pl_bmasse": [8.0, 1.0],
        "pl_orbper": [33.0, 300.0],
        "pl_eqt": [280.0, 255.0],
        "pl_insol": [1.2, 1.0],
        "st_teff": [3500.0, 5700.0],
    })
    out = crossmatch_hosts(neighbors, planets)
    assert bool(out.loc[out["source_id"] == 111, "has_hycean_candidate"].iloc[0])
    # The Earth-size, Earth-insolation planet is classical-temperate but NOT hycean.
    assert bool(out.loc[out["source_id"] == 222, "has_temperate_planet"].iloc[0])
    assert not bool(out.loc[out["source_id"] == 222, "has_hycean_candidate"].iloc[0])
    assert not bool(out.loc[out["source_id"] == 333, "known_planet_host"].iloc[0])


def test_run_recovers_injected_recipient(tmp_path):
    from seti.config import load_config

    rng = np.random.default_rng(3)
    # Anchor: K2-18-like distance, arbitrary but fixed velocity.
    anchor_obs = _observables_from_galactic(20.0, -30.0, 8.0, -12.0, 5.0, -4.0,
                                            source_id=999)
    # Background: random 6D stars within the shell, generally fast/far encounters.
    rows = []
    for i in range(300):
        X = 20.0 + rng.uniform(-30, 30)
        Y = -30.0 + rng.uniform(-30, 30)
        Z = 8.0 + rng.uniform(-30, 30)
        U, V, W = rng.uniform(-40, 40, 3)
        rows.append(_observables_from_galactic(X, Y, Z, U, V, W, source_id=i))
    # Injected recipient: a genuine close+slow PAST flyby of the anchor.
    kms_to_pcmyr = 1.0227121651
    vrelx = 4.0                                    # km/s, slow
    x_now = 20.0 + vrelx * kms_to_pcmyr * 3.0      # was at closest approach 3 Myr ago
    recip = _observables_from_galactic(x_now, -30.0 + 0.15, 8.0,
                                       -12.0 + vrelx, 5.0, -4.0, source_id=42)
    table = pd.DataFrame(rows + [recip])

    cfg = load_config()
    cfg.root = tmp_path                            # write results under tmp
    summary = panspermia_run(cfg, anchor=dict(anchor_obs), table=table,
                             search_pc=60.0, t_max_myr=10.0, d_min_max_pc=2.0)
    assert summary["n_shortlist"] >= 1
    assert summary["top_recipients"][0]["source_id"] == 42
    assert summary["closest_approach_pc"] < 0.5
    assert (tmp_path / "results" / "panspermia" / "recipient_candidates.csv").exists()

"""HERDSMAN offline tests: unit physics + end-to-end synthetic herd recovery.

The end-to-end test constructs a background field with realistic velocity
dispersion, injects an 8-star "herd" engineered (by backward integration from
the assembly point) to rendezvous within ~0.1 pc at t = +8 Myr, converts
everything to noisy Gaia-like observables, and requires the full pipeline —
zero-point correction, 6D build, precision cut, detection, mocks, vetting — to
recover the herd and nothing comparable in the pure-background control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seti.galactic.orbits import (
    KMS_TO_KPCMYR,
    R0_KPC,
    VSUN_GC_KMS,
    Z_SUN_KPC,
    acceleration,
)
from seti.herdsman.acquire import apply_rv_zero_point, scalar_velocity_error
from seti.herdsman.convergence import (
    ConvergenceParams,
    _components_from_pairs,
    _dedupe,
    _effective_units,
    propagate,
)
from seti.herdsman.mocks import shuffle_velocities
from seti.herdsman.vet import chemistry_vet
from seti.panspermia.kinematics import _A_ICRS_TO_GAL

_K = 4.740470446


# ----------------------------------------------------------------- helpers --
def _integrate_state(pos_kpc, vel_kpcmyr, t_myr, dt_myr, direction):
    """Leapfrog returning final (pos, vel) with forward-time velocity."""
    pos = np.atleast_2d(np.asarray(pos_kpc, float)).copy()
    vel = np.atleast_2d(np.asarray(vel_kpcmyr, float)).copy() * direction
    acc = acceleration(pos)
    for _ in range(int(round(t_myr / dt_myr))):
        vel += 0.5 * dt_myr * acc
        pos += dt_myr * vel
        acc = acceleration(pos)
        vel += 0.5 * dt_myr * acc
    return pos, vel * direction


def _gc_to_helio(pos_kpc, vel_kpcmyr):
    """Invert heliocentric_to_galactocentric."""
    X = (pos_kpc[:, 0] + R0_KPC) * 1e3
    Y = pos_kpc[:, 1] * 1e3
    Z = (pos_kpc[:, 2] - Z_SUN_KPC) * 1e3
    U = vel_kpcmyr[:, 0] / KMS_TO_KPCMYR - VSUN_GC_KMS[0]
    V = vel_kpcmyr[:, 1] / KMS_TO_KPCMYR - VSUN_GC_KMS[1]
    W = vel_kpcmyr[:, 2] / KMS_TO_KPCMYR - VSUN_GC_KMS[2]
    return X, Y, Z, U, V, W


def _helio_to_gaia_obs(X, Y, Z, U, V, W, rng, sig_rv=0.2, sig_pm=0.02,
                       sig_plx=0.02):
    """Heliocentric 6D -> noisy Gaia-like observable table."""
    r_gal = np.stack([X, Y, Z], axis=0)          # (3, N) pc
    v_gal = np.stack([U, V, W], axis=0)          # (3, N) km/s
    r_icrs = _A_ICRS_TO_GAL.T @ r_gal
    v_icrs = _A_ICRS_TO_GAL.T @ v_gal
    d = np.sqrt((r_icrs ** 2).sum(0))
    dec = np.arcsin(r_icrs[2] / d)
    ra = np.arctan2(r_icrs[1], r_icrs[0]) % (2 * np.pi)
    ca, sa, cd, sd = np.cos(ra), np.sin(ra), np.cos(dec), np.sin(dec)
    r_hat = np.stack([cd * ca, cd * sa, sd])
    a_hat = np.stack([-sa, ca, np.zeros_like(sa)])
    d_hat = np.stack([-sd * ca, -sd * sa, cd])
    rv = (v_icrs * r_hat).sum(0)
    v_a = (v_icrs * a_hat).sum(0)
    v_d = (v_icrs * d_hat).sum(0)
    pmra = v_a * 1000.0 / (_K * d)
    pmdec = v_d * 1000.0 / (_K * d)
    n = len(d)
    plx = 1000.0 / d + rng.standard_normal(n) * sig_plx
    return pd.DataFrame({
        "source_id": np.arange(n, dtype=np.int64),
        "ra": np.degrees(ra), "dec": np.degrees(dec),
        "parallax": plx, "parallax_error": np.full(n, sig_plx),
        "parallax_over_error": plx / sig_plx,
        "pmra": pmra + rng.standard_normal(n) * sig_pm,
        "pmra_error": np.full(n, sig_pm),
        "pmdec": pmdec + rng.standard_normal(n) * sig_pm,
        "pmdec_error": np.full(n, sig_pm),
        "radial_velocity": rv + rng.standard_normal(n) * sig_rv,
        "radial_velocity_error": np.full(n, sig_rv),
        "rv_nb_transits": np.full(n, 20), "rv_template_teff": np.full(n, 5500.0),
        "grvs_mag": np.full(n, 9.0), "phot_g_mean_mag": np.full(n, 10.0),
        "bp_rp": np.full(n, 0.8), "ruwe": np.full(n, 1.0),
        "mh_gspphot": rng.normal(-0.1, 0.25, n),
        "teff_gspphot": np.full(n, 5600.0), "logg_gspphot": np.full(n, 4.4),
    })


def _background(n, radius_pc, rng):
    """Uniform sphere of field stars with a disk-ish velocity ellipsoid."""
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = radius_pc * rng.random(n) ** (1 / 3)
    pos = u * r[:, None]
    vel = np.stack([rng.normal(0, 25, n), rng.normal(0, 20, n),
                    rng.normal(0, 12, n)], axis=1)
    return pos, vel


def _make_herd(n_members, t_meet_myr, rng, sig_arrive_kms=3.0):
    """Now-states (gc) of stars that rendezvous within ~0.1 pc at +t_meet."""
    sun_pos = np.array([[-R0_KPC, 0.0, Z_SUN_KPC]])
    sun_vel = np.array([[v * KMS_TO_KPCMYR for v in VSUN_GC_KMS]])
    P, V_P = _integrate_state(sun_pos, sun_vel, t_meet_myr, 0.05, +1)
    pos_meet = P + rng.standard_normal((n_members, 3)) * (0.1 / 1e3)
    vel_meet = V_P + rng.standard_normal((n_members, 3)) \
        * (sig_arrive_kms * KMS_TO_KPCMYR)
    return _integrate_state(pos_meet, vel_meet, t_meet_myr, 0.05, -1)


# -------------------------------------------------------------- unit tests --
def test_rv_zero_point_endpoints():
    df = pd.DataFrame({"radial_velocity": [10.0, 10.0, 10.0],
                       "grvs_mag": [9.0, 11.0, 14.0]})
    out = apply_rv_zero_point(df)
    corr = out["rv_zeropoint_corr_kms"].to_numpy()
    assert corr[0] == 0.0                       # bright: no correction
    assert abs(corr[1]) < 0.02                  # ~0 at G_RVS = 11
    assert 0.30 < corr[2] < 0.45                # ~+0.39 km/s at G_RVS = 14


def test_scalar_velocity_error_dominated_by_rv():
    df = pd.DataFrame({"parallax": [10.0], "radial_velocity_error": [0.4],
                       "pmra": [50.0], "pmdec": [50.0],
                       "pmra_error": [0.02], "pmdec_error": [0.02],
                       "parallax_over_error": [500.0]})
    sig = scalar_velocity_error(df, astro_floor_kms=0.3)
    assert 0.45 < sig[0] < 0.60                 # sqrt(0.4^2+0.3^2) ~ 0.5


def test_propagate_roundtrip():
    rng = np.random.default_rng(3)
    pos = np.array([[-R0_KPC + 0.05, 0.02, 0.01]])
    vel = np.array([[10.0, 240.0, 5.0]]) * KMS_TO_KPCMYR
    p1, v1 = _integrate_state(pos, vel, 10.0, 0.05, +1)
    p0, v0 = _integrate_state(p1, v1, 10.0, 0.05, -1)
    assert np.allclose(p0, pos, atol=1e-6)
    assert np.allclose(v0, vel, atol=1e-9)
    del rng


def test_propagate_stream_matches_endpoint():
    pos = np.array([[-R0_KPC, 0.03, 0.0]])
    vel = np.array([[0.0, 233.0, 3.0]]) * KMS_TO_KPCMYR
    last = None
    for t, p in propagate(pos, vel, 5.0, 0.25, +1, 2):
        last = (t, p.copy())
    pe, _ = _integrate_state(pos, vel, 5.0, 0.25, +1)
    assert last is not None and abs(last[0] - 5.0) < 1e-9
    assert np.allclose(last[1], pe, atol=1e-9)


def test_components_and_effective_units():
    pairs = np.array([[0, 1], [1, 2], [5, 6]])
    comps = _components_from_pairs(pairs, 3)
    assert len(comps) == 1 and set(comps[0]) == {0, 1, 2}
    pos_now = np.array([[0, 0, 0], [0.3, 0, 0], [50, 0, 0], [0, 80, 0.0]])
    assert _effective_units(pos_now, np.arange(4), 1.0) == 3   # 0+1 collapse


def test_dedupe_merges_and_counts_persistence():
    a = {"t_myr": 5.0, "members": np.array([1, 2, 3, 4]), "surprise": 6.0}
    b = {"t_myr": 5.5, "members": np.array([1, 2, 3, 5]), "surprise": 4.0}
    c = {"t_myr": 9.0, "members": np.array([7, 8, 9, 10]), "surprise": 5.0}
    for d in (a, b, c):
        d.update({"m": 4, "m_eff": 4, "r_ball_pc": 2.0, "lambda": 0.1,
                  "rms_now_pc": 30.0, "rms_meet_pc": 1.0, "focus": 30.0,
                  "med_now_pc": 40.0, "sig_v_internal_kms": 2.0,
                  "centroid_gc_kpc": [0, 0, 0]})
    out = _dedupe([a, b, c], 0.5)
    assert len(out) == 2
    top = out[0]
    assert top["surprise"] == 6.0 and top["n_epochs_seen"] == 2


def test_shuffle_preserves_positions_and_velocity_multiset():
    rng = np.random.default_rng(11)
    pos, vel = _background(500, 100.0, rng)
    sig = rng.uniform(0.2, 0.6, 500)
    vel_s, sig_s = shuffle_velocities(pos, vel, sig, 40.0, rng)
    assert vel_s.shape == vel.shape
    assert np.allclose(np.sort(vel_s.ravel()), np.sort(vel.ravel()))
    assert np.allclose(np.sort(sig_s), np.sort(sig))
    assert not np.allclose(vel_s, vel)          # something actually moved


def test_chemistry_vet_flags():
    het = chemistry_vet(np.array([-0.4, 0.0, 0.3, 0.1, -0.2]))
    assert het["heterogeneous"] is True and het["co_natal_possible"] is False
    con = chemistry_vet(np.array([0.01, 0.02, -0.01, 0.0]))
    assert con["co_natal_possible"] is True


# -------------------------------------------------- end-to-end injection --
def _synthetic_catalog(seed=42):
    """Background field + injected 8-star herd, as a noisy Gaia-like table."""
    rng = np.random.default_rng(seed)
    t_meet = 8.0
    bg_pos, bg_vel = _background(2500, 120.0, rng)
    herd_pos_kpc, herd_vel_kpcmyr = _make_herd(8, t_meet, rng)
    hX, hY, hZ, hU, hV, hW = _gc_to_helio(herd_pos_kpc, herd_vel_kpcmyr)
    X = np.concatenate([bg_pos[:, 0], hX])
    Y = np.concatenate([bg_pos[:, 1], hY])
    Z = np.concatenate([bg_pos[:, 2], hZ])
    U = np.concatenate([bg_vel[:, 0], hU])
    V = np.concatenate([bg_vel[:, 1], hV])
    W = np.concatenate([bg_vel[:, 2], hW])
    table = _helio_to_gaia_obs(X, Y, Z, U, V, W, rng)
    herd_ids = set(range(2500, 2508))
    # Give the herd a wide, field-like metallicity spread (gathered stars).
    table.loc[list(herd_ids), "mh_gspphot"] = \
        [-0.45, -0.25, -0.1, 0.0, 0.1, 0.2, 0.3, -0.35]
    return table, herd_ids, t_meet


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    from seti.config import load_config
    from seti.herdsman.run import herdsman_run

    table, herd_ids, t_meet = _synthetic_catalog()
    cfg = load_config()
    cfg.root = tmp_path_factory.mktemp("herdsman_out")
    summary = herdsman_run(
        cfg, t_max_myr=12.0, dt_myr=0.25, rec_every=2, sigv_max_kms=0.8,
        astro_floor_kms=0.2, n_min=4, r_now_min_pc=20.0, focus_min=3.0,
        surprise_min=3.0, n_mocks=2, table=table)
    import json
    cands = json.loads((cfg.root / "results" / "herdsman"
                        / "candidates_forward.json").read_text())
    return summary, cands, herd_ids, t_meet


def test_synthetic_herd_recovered(synthetic_run):
    summary, cands, herd_ids, t_meet = synthetic_run
    assert summary["directions"]["forward"]["n_candidates"] >= 1
    top = cands["candidates"][0]
    got = set(top["member_source_ids"]) & herd_ids
    assert len(got) >= 4, f"herd not recovered: {top['member_source_ids']}"
    assert abs(top["t_myr"] - t_meet) < 2.5
    assert top["chemistry"]["heterogeneous"] is True
    assert top["rendezvous_mc"]["p_rms_lt_5pc"] > 0.5


def test_background_produces_no_comparable_forward_candidate(synthetic_run):
    summary, cands, herd_ids, _ = synthetic_run
    # Any non-herd candidate must be far weaker than the injected herd.
    top_surprise = cands["candidates"][0]["surprise"]
    for c in cands["candidates"][1:]:
        if not (set(c["member_source_ids"]) & herd_ids):
            assert c["surprise"] < top_surprise
    # The time-reversal control must not out-score the injection.
    assert summary["directions"]["backward"]["best_surprise"] \
        < summary["directions"]["forward"]["best_surprise"]


def test_v2_cuts_recover_cold_herd_and_kill_hot_background():
    """v2 herd-physics cuts: a dynamically cold, dwelling herd survives;
    field-dispersion transient crossings (the entire v1 background) cannot."""
    from seti.galactic.orbits import heliocentric_to_galactocentric
    from seti.herdsman.acquire import apply_rv_zero_point, scalar_velocity_error
    from seti.herdsman.convergence import ConvergenceParams, detect_convergences
    from seti.panspermia.kinematics import phase_space_6d

    rng = np.random.default_rng(7)
    t_meet = 8.0
    bg_pos, bg_vel = _background(2500, 120.0, rng)
    herd_pos_kpc, herd_vel_kpcmyr = _make_herd(8, t_meet, rng,
                                               sig_arrive_kms=1.5)
    hX, hY, hZ, hU, hV, hW = _gc_to_helio(herd_pos_kpc, herd_vel_kpcmyr)
    table = _helio_to_gaia_obs(
        np.concatenate([bg_pos[:, 0], hX]), np.concatenate([bg_pos[:, 1], hY]),
        np.concatenate([bg_pos[:, 2], hZ]), np.concatenate([bg_vel[:, 0], hU]),
        np.concatenate([bg_vel[:, 1], hV]), np.concatenate([bg_vel[:, 2], hW]),
        rng)
    df = phase_space_6d(apply_rv_zero_point(table))
    df["sigv_kms"] = scalar_velocity_error(df, astro_floor_kms=0.2)
    pos_kpc, vel = heliocentric_to_galactocentric(
        df["X_pc"].to_numpy(float), df["Y_pc"].to_numpy(float),
        df["Z_pc"].to_numpy(float), df["U_kms"].to_numpy(float),
        df["V_kms"].to_numpy(float), df["W_kms"].to_numpy(float))
    params = ConvergenceParams(t_max_myr=12.0, n_min=4, focus_min=3.0,
                               surprise_min=3.0, sigv_int_max_kms=5.0,
                               min_epochs=2)
    res = detect_convergences(pos_kpc, vel, df["sigv_kms"].to_numpy(float),
                              +1, params)
    herd_ids = set(range(2500, 2508))
    hits = [c for c in res["candidates"]
            if len(set(c["members"]) & herd_ids) >= 4]
    assert hits, "cold herd not recovered under v2 cuts"
    assert hits[0]["sig_v_internal_kms"] < 5.0
    assert hits[0]["n_epochs_seen"] >= 2
    # Every surviving candidate must be the herd — the hot chance background
    # (internal dispersion ~ field, single-epoch) is fully removed.
    for c in res["candidates"]:
        assert set(c["members"]) & herd_ids, \
            f"hot-background candidate survived v2 cuts: {c['surprise']:.1f}"


def test_staged_pipeline_recovers_herd(tmp_path):
    """fetch -> scan(real) -> scan(mock shard) -> reduce on the synthetic sky.

    Exercises the checkpointed CI path: parquet round-trip, per-scan shard
    files, tolerant aggregation, and identical herd recovery to the monolith.
    """
    import json

    from seti.config import load_config
    from seti.herdsman import stages

    table, herd_ids, t_meet = _synthetic_catalog()
    cfg = load_config()
    cfg.root = tmp_path
    stages.fetch_stage(cfg, table=table, sigv_max_kms=0.8, astro_floor_kms=0.2)
    assert (tmp_path / "results" / "herdsman" / "sample.parquet").exists()

    params = ConvergenceParams(t_max_myr=12.0, n_min=4, focus_min=3.0,
                               surprise_min=3.0)
    stages.scan_stage(cfg, mode="real", params=params)
    stages.scan_stage(cfg, mode="mock", shard=0, mocks_per_shard=2,
                      params=params)
    shards = tmp_path / "results" / "herdsman" / "shards"
    assert (shards / "real_forward.json").exists()
    assert len(list(shards.glob("mock_*_forward.json"))) == 2

    summary = stages.reduce_stage(cfg, n_mocks_expected=2,
                                  astro_floor_kms=0.2)
    fwd = summary["directions"]["forward"]
    assert fwd["n_candidates"] >= 1 and fwd["p_global"] is not None
    cands = json.loads((tmp_path / "results" / "herdsman"
                        / "candidates_forward.json").read_text())
    top = cands["candidates"][0]
    assert len(set(top["member_source_ids"]) & herd_ids) >= 4
    assert abs(top["t_myr"] - t_meet) < 2.5


def test_median_pairwise_subsamples_giant_components():
    """Giant percolation components must not allocate O(N^2) memory.

    Run 30199588771 died allocating 42.7 GiB for a 43,692-star component; the
    strided-subsample estimator has to stay accurate and bounded above the cap.
    """
    from seti.herdsman.convergence import _MEDIAN_PAIRWISE_CAP, _median_pairwise

    rng = np.random.default_rng(7)
    small = rng.uniform(0.0, 100.0, size=(500, 3))
    exact = _median_pairwise(small)

    big = rng.uniform(0.0, 100.0, size=(6 * _MEDIAN_PAIRWISE_CAP, 3))
    approx = _median_pairwise(big)
    # Same uniform-cube distribution: subsampled median must agree with the
    # exact small-N median to a few percent, deterministically.
    assert abs(approx - exact) / exact < 0.05
    assert _median_pairwise(big) == approx

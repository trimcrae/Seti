"""Tests for the population-level anomaly-clustering test.

The decisive checks: (1) a *random* anomaly subset of the parent is NOT flagged as
over-clustered (no false positive, even though the parent itself is spatially
structured); (2) an injected co-moving spatial cluster IS recovered at high
significance; (3) the matched null defeats the brightness confound --- anomalies
drawn preferentially from bright/nearby stars are not flagged just for that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from seti.cluster.clustering import friends_of_friends, matched_null_clustering
from seti.cluster.phase_space import galactic_xyz, tangential_velocity


def _background(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    # Exponential-ish distance profile within ~500 pc.
    dist = rng.uniform(50, 500, n)
    plx = 1000.0 / dist
    g = rng.uniform(12, 19, n)
    bp_rp = rng.uniform(0.0, 2.5, n)
    return pd.DataFrame({"ra": ra, "dec": dec, "parallax": plx,
                         "pmra": rng.normal(0, 10, n), "pmdec": rng.normal(0, 10, n),
                         "phot_g_mean_mag": g, "bp_rp": bp_rp})


def test_random_anomaly_set_not_over_clustered():
    df = galactic_xyz(_background())
    rng = np.random.default_rng(0)
    mask = np.zeros(len(df), bool)
    mask[rng.choice(len(df), 60, replace=False)] = True
    res = matched_null_clustering(df, mask, ["X_pc", "Y_pc", "Z_pc"],
                                  n_null=200, seed=7)
    assert res["insufficient"] is False
    assert res["p_value"] > 0.05          # no false detection


def test_injected_comoving_cluster_recovered():
    bg = _background(n=3000)
    # Inject a tight co-moving group: 40 stars in a 5 pc ball at ~200 pc.
    rng = np.random.default_rng(3)
    cz = rng.normal(0, 5, 40)             # ~5 pc depth scatter along the line of sight
    # Place the ball at a fixed sky position/distance.
    clump = pd.DataFrame({
        "ra": 123.0 + rng.normal(0, 0.3, 40),
        "dec": 20.0 + rng.normal(0, 0.3, 40),
        "parallax": 1000.0 / (200.0 + cz),
        "pmra": rng.normal(15, 0.5, 40), "pmdec": rng.normal(-8, 0.5, 40),
        "phot_g_mean_mag": rng.uniform(12, 19, 40),
        "bp_rp": rng.uniform(0.0, 2.5, 40)})
    df = pd.concat([bg, clump], ignore_index=True)
    df = galactic_xyz(df)
    mask = np.zeros(len(df), bool)
    mask[len(bg):] = True                 # the injected cluster is the anomaly set
    res = matched_null_clustering(df, mask, ["X_pc", "Y_pc", "Z_pc"],
                                  n_null=300, seed=11)
    assert res["over_clustered"] is True
    assert res["p_value"] < 0.01
    # Friends-of-friends should recover a group.
    labels = friends_of_friends(df[mask], ["X_pc", "Y_pc", "Z_pc"],
                                linking_length=0.5, min_size=5)
    assert (labels >= 0).sum() >= 20


def test_brightness_confound_defeated():
    # Anomalies drawn preferentially from bright, nearby stars -- which ARE more
    # clustered in distance -- must NOT be flagged, because the matched null draws
    # from the same magnitude/distance strata.
    df = galactic_xyz(_background(n=4000, seed=5))
    bright_near = (df["phot_g_mean_mag"] < 14) & (df["dist_pc"] < 200)
    idx = np.where(bright_near.to_numpy())[0]
    rng = np.random.default_rng(2)
    sel = rng.choice(idx, size=min(60, len(idx)), replace=False)
    mask = np.zeros(len(df), bool)
    mask[sel] = True
    # Match on the confounders that drove selection (magnitude AND parallax);
    # detectability correlating with distance is exactly why parallax must be a
    # matching feature in real use.
    res = matched_null_clustering(df, mask, ["X_pc", "Y_pc", "Z_pc"],
                                  feature_cols=("phot_g_mean_mag", "parallax"),
                                  n_null=300, seed=9)
    assert res["p_value"] > 0.02          # matched null removes the confound


def test_tangential_velocity_columns():
    df = tangential_velocity(_background(n=50))
    assert {"vtan_ra_kms", "vtan_dec_kms", "vtan_kms"} <= set(df.columns)
    assert np.isfinite(df["vtan_kms"]).any()


def test_cluster_run_offline_recovers_excess_group(tmp_path):
    from seti.cluster.run import cluster_run, ir_excess_indicator
    from seti.config import load_config

    bg = _background(n=2500, seed=8)
    # Normal stars: W1-W2 ~ 0 with scatter.
    rng = np.random.default_rng(4)
    bg["w1mpro"] = 12.0 - 0.0 + rng.normal(0, 0.03, len(bg))
    bg["w2mpro"] = 12.0 + rng.normal(0, 0.03, len(bg))
    bg["parallax_over_error"] = 50.0
    bg["ruwe"] = 1.0
    # An IR-excess co-moving group: strong W1-W2 excess, tight in space.
    n = 30
    clump = pd.DataFrame({
        "ra": 200.0 + rng.normal(0, 0.2, n), "dec": -10.0 + rng.normal(0, 0.2, n),
        "parallax": 1000.0 / (150.0 + rng.normal(0, 3, n)),
        "pmra": rng.normal(12, 0.5, n), "pmdec": rng.normal(5, 0.5, n),
        "phot_g_mean_mag": rng.uniform(12, 16, n), "bp_rp": rng.uniform(0.5, 1.5, n),
        "w1mpro": 11.0 + rng.normal(0, 0.03, n),
        "w2mpro": 11.0 + 0.8 + rng.normal(0, 0.03, n),   # W1-W2 ~ -0.8? make excess:
        "parallax_over_error": 50.0, "ruwe": 1.0})
    # Make W1-W2 strongly POSITIVE (excess): w2 brighter numerically => set w2 lower.
    clump["w2mpro"] = clump["w1mpro"] - 0.9
    tbl = pd.concat([bg, clump], ignore_index=True)

    ind = ir_excess_indicator(tbl)
    assert np.nanmax(ind["ir_excess_z"]) > 4.0     # the group is a clear excess

    cfg = load_config()
    cfg.root = tmp_path
    s = cluster_run(cfg, table=tbl, excess_z_min=4.0)
    assert s["n_ir_excess"] >= 20
    assert s["clustering"]["over_clustered"] is True
    assert s["n_groups"] >= 1

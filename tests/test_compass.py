"""Offline tests for the COMPASS axial-statistics core (no network)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seti.compass.axial import (
    bingham_stat,
    ecliptic_latitude_deg,
    orientation_tensor,
    pole_axes,
    principal_axis,
    scan_coherence,
    shuffle_axes_within_bands,
    tangent_basis,
)

RNG = np.random.default_rng(20260726)


def _random_axes(n, rng):
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[v[:, 2] < 0] *= -1
    return v


def test_tangent_basis_is_orthonormal():
    ra = np.array([0.0, 123.4, 271.0])
    dec = np.array([-67.0, 5.5, 42.0])
    e, n, los = tangent_basis(ra, dec)
    for a in (e, n, los):
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose((e * n).sum(1), 0.0, atol=1e-12)
    assert np.allclose((e * los).sum(1), 0.0, atol=1e-12)
    assert np.allclose((n * los).sum(1), 0.0, atol=1e-12)


def test_face_on_orbit_pole_is_the_line_of_sight():
    """i = 0 means the orbit plane is the plane of sky: pole along the LOS."""
    ra, dec = np.array([200.0]), np.array([-30.0])
    _, _, los = tangent_basis(ra, dec)
    from seti.compass.axial import _ICRS_TO_GAL

    p = pole_axes(ra, dec, np.array([0.0]), np.array([77.0]))
    los_gal = los @ _ICRS_TO_GAL.T
    dot = float((p * los_gal).sum())
    assert abs(abs(dot) - 1.0) < 1e-9


def test_pole_axis_sign_ambiguity_is_harmless():
    """i and 180-i (the astrometric ambiguity) give the same axis statistic."""
    ra = np.full(50, 150.0) + RNG.uniform(-5, 5, 50)
    dec = np.full(50, 20.0) + RNG.uniform(-5, 5, 50)
    inc = RNG.uniform(0, 180, 50)
    node = RNG.uniform(0, 180, 50)
    p1 = pole_axes(ra, dec, inc, node)
    p2 = pole_axes(ra, dec, 180.0 - inc, (node + 180.0) % 360.0)
    assert bingham_stat(p1) == pytest.approx(bingham_stat(p2), rel=1e-9)


def test_bingham_stat_calibrates_on_isotropy_and_fires_on_alignment():
    iso = _random_axes(4000, np.random.default_rng(1))
    s_iso = bingham_stat(iso)
    # chi^2_5 mean is 5; at N=4000 the statistic should be O(10), not O(100).
    assert s_iso < 30.0

    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    aligned = axis + 0.15 * np.random.default_rng(2).standard_normal((100, 3))
    aligned /= np.linalg.norm(aligned, axis=1, keepdims=True)
    # Perfect alignment at N=100 gives exactly 5N = 500; 0.15-rad scatter
    # lands just below it, still two orders above the isotropic O(5).
    assert bingham_stat(aligned) > 300.0
    assert abs(float(principal_axis(aligned) @ axis)) > 0.99


def test_orientation_tensor_trace_is_one():
    t = orientation_tensor(_random_axes(300, np.random.default_rng(3)))
    assert np.trace(t) == pytest.approx(1.0)


def test_scan_finds_injected_patch_and_shuffle_null_kills_it():
    """End-to-end: an aligned 25-star patch beats every shuffle maximum."""
    rng = np.random.default_rng(11)
    n_field = 1200
    pos = rng.uniform(-200, 200, (n_field, 3))
    axes = _random_axes(n_field, rng)

    axis = np.array([0.0, 0.6, 0.8])
    patch = rng.uniform(-15, 15, (25, 3)) + np.array([80.0, -40.0, 10.0])
    pax = axis + 0.12 * rng.standard_normal((25, 3))
    pax /= np.linalg.norm(pax, axis=1, keepdims=True)
    pos_all = np.vstack([pos, patch])
    axes_all = np.vstack([axes, pax])
    axes_all[axes_all[:, 2] < 0] *= -1

    hits = scan_coherence(pos_all, axes_all, radius_pc=30.0, n_min=8)
    assert hits, "no neighbourhoods found"
    top = hits[0]
    injected = set(range(n_field, n_field + 25))
    assert len(set(top["members"]) & injected) >= 15
    assert abs(float(np.array(top["axis"]) @ axis)) > 0.95

    # Shuffle null: permuting axes globally must never reproduce the top stat.
    band = np.zeros(len(axes_all), dtype=int)
    null_max = []
    for k in range(20):
        sh = shuffle_axes_within_bands(axes_all, band,
                                       np.random.default_rng(100 + k))
        nh = scan_coherence(pos_all, sh, radius_pc=30.0, n_min=8)
        null_max.append(nh[0]["stat"] if nh else 0.0)
    assert top["stat"] > max(null_max)


def test_shuffle_preserves_band_membership():
    axes = _random_axes(200, np.random.default_rng(5))
    band = np.repeat(np.arange(4), 50)
    sh = shuffle_axes_within_bands(axes, band, np.random.default_rng(6))
    for b in range(4):
        idx = band == b
        # same multiset of axes within each band
        a0 = np.sort(axes[idx].round(12).view([('', float)] * 3), axis=0)
        a1 = np.sort(sh[idx].round(12).view([('', float)] * 3), axis=0)
        assert np.array_equal(a0, a1)


def test_ecliptic_latitude_known_points():
    # North ecliptic pole: ra=270, dec=66.56 -> beta ~ +90.
    assert ecliptic_latitude_deg(270.0, 66.5607) == pytest.approx(90.0, abs=0.01)
    # A point on the ecliptic: ra=0, dec=0 -> beta = 0.
    assert ecliptic_latitude_deg(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Thiele-Innes inversion (DR3 Orbital solutions publish A,B,F,G, not i/Omega)
# ---------------------------------------------------------------------------

def test_thiele_innes_round_trip_recovers_inclination_and_axial_node():
    from seti.compass.orbit import (
        geometric_to_thiele_innes,
        thiele_innes_to_geometric,
    )

    rng = np.random.default_rng(8)
    n = 500
    a0 = rng.uniform(0.1, 10.0, n)
    inc = rng.uniform(1.0, 179.0, n)
    node = rng.uniform(0.0, 360.0, n)
    omega = rng.uniform(0.0, 360.0, n)

    A, B, F, G = geometric_to_thiele_innes(a0, inc, node, omega)
    a0_r, inc_r, node_r, _ = thiele_innes_to_geometric(A, B, F, G)

    assert np.allclose(a0_r, a0, rtol=1e-9)
    # The inversion cannot distinguish (i, O) from (180-i, O+180): both give
    # identical (A,B,F,G) up to the omega redefinition. What IS recoverable —
    # and all the detector needs — is the POLE AXIS. Check axis equality.
    from seti.compass.axial import pole_axes
    ra = np.full(n, 120.0)
    dec = np.full(n, -25.0)
    p_true = pole_axes(ra, dec, inc, node % 180.0)
    p_rec = pole_axes(ra, dec, inc_r, node_r)
    dots = np.abs((p_true * p_rec).sum(1))
    assert np.all(dots > 1.0 - 1e-9), f"min |dot| {dots.min()}"


def test_thiele_innes_face_on_and_edge_on_limits():
    from seti.compass.orbit import (
        geometric_to_thiele_innes,
        thiele_innes_to_geometric,
    )

    # Face-on: i = 0 -> k2 = 0, cos i = +1.
    A, B, F, G = geometric_to_thiele_innes(2.0, 0.0, 45.0, 30.0)
    _, inc, _, _ = thiele_innes_to_geometric(A, B, F, G)
    assert float(inc) == pytest.approx(0.0, abs=1e-6)

    # Edge-on: i = 90 -> k1 = k2, cos i = 0.
    A, B, F, G = geometric_to_thiele_innes(2.0, 90.0, 45.0, 30.0)
    _, inc, node, _ = thiele_innes_to_geometric(A, B, F, G)
    assert float(inc) == pytest.approx(90.0, abs=1e-6)
    assert float(node) == pytest.approx(45.0, abs=1e-6)


def test_batched_stats_match_per_group_scan():
    from seti.compass.axial import bingham_stats_batch, group_matrix

    rng = np.random.default_rng(21)
    pos = rng.uniform(-100, 100, (400, 3))
    axes = _random_axes(400, rng)
    m, counts, centers = group_matrix(pos, radius_pc=40.0, n_min=8)
    stats = bingham_stats_batch(m, counts, axes)

    hits = scan_coherence(pos, axes, radius_pc=40.0, n_min=8)
    assert len(hits) == len(stats)
    assert max(h["stat"] for h in hits) == pytest.approx(float(stats.max()),
                                                         rel=1e-9)


def test_compass_run_end_to_end_with_injected_patch(tmp_path):
    """Full offline pipeline: synthetic NSS table -> scan -> shuffle null.

    The injected aligned patch must surface as the max statistic with a
    globally significant p; the isotropic remainder must not.
    """
    from seti.compass.orbit import geometric_to_thiele_innes
    from seti.compass.run import compass_run
    from seti.config import load_config

    rng = np.random.default_rng(31)
    n_field = 600

    # Field: isotropic poles. Draw random axes, convert to (i, node) per
    # star... simpler: random (i, node) uniform in cos i and node gives an
    # isotropic pole distribution in the tangent frame.
    ra = rng.uniform(0, 360, n_field)
    dec = np.degrees(np.arcsin(rng.uniform(-0.95, 0.95, n_field)))
    inc = np.degrees(np.arccos(rng.uniform(-1, 1, n_field)))
    node = rng.uniform(0, 180, n_field)
    parallax = rng.uniform(2.5, 20.0, n_field)     # 50-400 pc

    # Patch: 20 stars in a small sky region at a common distance sharing a
    # pole (same i/node works because they share a tangent frame closely).
    n_p = 20
    ra_p = 40.0 + rng.uniform(-2, 2, n_p)
    dec_p = -10.0 + rng.uniform(-2, 2, n_p)
    plx_p = np.full(n_p, 10.0) + rng.uniform(-0.2, 0.2, n_p)   # ~100 pc
    inc_p = np.full(n_p, 55.0) + rng.normal(0, 3.0, n_p)
    node_p = np.full(n_p, 120.0) + rng.normal(0, 3.0, n_p)

    ra = np.concatenate([ra, ra_p])
    dec = np.concatenate([dec, dec_p])
    inc = np.concatenate([inc, inc_p])
    node = np.concatenate([node, node_p])
    parallax = np.concatenate([parallax, plx_p])
    n = len(ra)
    a, b, f, g = geometric_to_thiele_innes(
        rng.uniform(0.5, 5.0, n), inc, node, rng.uniform(0, 360, n))

    nss = pd.DataFrame({
        "source_id": np.arange(n, dtype=np.int64),
        "nss_solution_type": "Orbital",
        "a_thiele_innes": a, "b_thiele_innes": b,
        "f_thiele_innes": f, "g_thiele_innes": g,
        "period": rng.uniform(100, 900, n),
        "eccentricity": rng.uniform(0, 0.6, n),
        "significance": rng.uniform(10, 60, n),
        "ra": ra, "dec": dec, "parallax": parallax,
        "parallax_over_error": np.full(n, 20.0),
        "pmra": rng.normal(0, 20, n), "pmdec": rng.normal(0, 20, n),
        "radial_velocity": rng.normal(0, 25, n),
        "phot_g_mean_mag": rng.uniform(8, 15, n),
        "mh_gspphot": rng.normal(-0.1, 0.3, n),
        "teff_gspphot": rng.uniform(4500, 7500, n),
        "random_index": np.arange(n, dtype=np.int64),
    })

    cfg = load_config()
    cfg.root = tmp_path
    out = tmp_path / "results" / "compass"
    out.mkdir(parents=True)
    nss.to_parquet(out / "nss_sample.parquet", index=False)   # fetch checkpoint

    summary = compass_run(cfg, radii_pc=(30.0,), n_min=8, n_shuffles=60,
                          d_max_pc=1000.0)
    r = summary["radii"]["r30"]
    assert r["n_groups"] > 0
    assert r["p_global"] <= 1.0 / 61 + 1e-9        # patch beats every shuffle

    cands = json.loads((out / "candidates.json").read_text())
    top = cands["top_per_radius"][0]
    injected = set(range(600, 620))
    assert len(set(top["member_source_ids"]) & injected) >= 12
    assert top["above_null_p99"]

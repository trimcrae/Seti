"""Offline tests for the COMPASS axial-statistics core (no network)."""

from __future__ import annotations

import numpy as np
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

"""Offline tests for the ISO back-tracking dynamics (no network).

We construct synthetic ISOs whose velocity is set to point *exactly* back at a
synthetic target star a few pc away and verify the back-track recovers a small
closest-approach distance; a mis-directed ISO must not.  We also pin the
radiant->velocity sign convention (an ISO whose radiant is at the anti-apex must
move *toward* the apex).
"""

from __future__ import annotations

import numpy as np

from seti.galactic.orbits import (
    KMS_TO_KPCMYR,
    heliocentric_to_galactocentric,
)
from seti.iso.backtrack import (
    backtrack_closest_approach,
    mc_backtrack,
    radiant_to_velocity,
)
from seti.panspermia.kinematics import _A_ICRS_TO_GAL

# 1 km/s in pc/Myr (for turning a required relative velocity into a speed).
_KMS_TO_PCMYR = KMS_TO_KPCMYR * 1000.0


def _target_gc(pos_pc, vel_kms):
    """Galactocentric (pos, vel) of a synthetic target from heliocentric inputs."""
    pos, vel = heliocentric_to_galactocentric(
        pos_pc[0], pos_pc[1], pos_pc[2], vel_kms[0], vel_kms[1], vel_kms[2])
    return pos[0], vel[0]


def test_radiant_velocity_sign_points_toward_apex():
    # Solar-apex-like direction and its antipode.
    ra_apex, dec_apex = 270.0, 30.0
    ra_anti, dec_anti = 90.0, -30.0
    v_inf = 30.0
    v = radiant_to_velocity(v_inf, ra_anti, dec_anti)
    # Expected: the ISO moves toward the apex, i.e. along the Galactic apex vector.
    ra_r, dec_r = np.radians(ra_apex), np.radians(dec_apex)
    apex_icrs = np.array([np.cos(dec_r) * np.cos(ra_r),
                          np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)])
    apex_gal = _A_ICRS_TO_GAL @ apex_icrs
    assert np.allclose(np.linalg.norm(v), v_inf, rtol=1e-6)     # speed preserved
    assert np.allclose(v / v_inf, apex_gal, atol=1e-6)          # direction = apex


def test_backtrack_recovers_aimed_iso():
    # Target 4 pc toward the Galactic centre, at rest heliocentrically.
    P = np.array([4.0, 0.0, 0.0])
    T = 3.0                                                     # Myr lookback
    tgt_pos, tgt_vel = _target_gc(P, [0.0, 0.0, 0.0])
    # ISO at origin; to have coincided with the target T Myr ago, its heliocentric
    # velocity must be -P/T (relative velocity closes the gap going backward).
    v_iso_kms = (-P / T) / _KMS_TO_PCMYR
    res = backtrack_closest_approach(v_iso_kms, tgt_pos, tgt_vel,
                                     t_max_myr=10.0, dt_myr=0.1)
    assert res["d_min_pc"] < 0.3                                # passes ~0 pc
    assert -4.0 < res["t_enc_myr"] < -2.0                       # ~ -3 Myr, in past


def test_backtrack_rejects_misdirected_isos():
    # Same target, but ISOs fired off in random directions rarely pass close.
    P = np.array([4.0, 0.0, 0.0])
    tgt_pos, tgt_vel = _target_gc(P, [0.0, 0.0, 0.0])
    rng = np.random.default_rng(7)
    d_mins = []
    for _ in range(20):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v_iso_kms = 1.3 * u                                     # ~ same speed
        res = backtrack_closest_approach(v_iso_kms, tgt_pos, tgt_vel,
                                         t_max_myr=10.0, dt_myr=0.2)
        d_mins.append(res["d_min_pc"])
    assert np.median(d_mins) > 1.0                              # typically far


def test_mc_backtrack_aimed_is_consistent_but_caveated():
    # Build a synthetic ISO from a real-ish radiant, then place the target exactly
    # where that ISO's back-track goes, so a low-uncertainty MC must recover it.
    ra_rad, dec_rad, v_inf = 279.8, 33.996, 26.0
    v_gal = radiant_to_velocity(v_inf, ra_rad, dec_rad)        # (U,V,W) km/s
    T = 1.0                                                     # Myr
    # Target sits toward the radiant (the direction the ISO came from).
    P = (-v_gal * _KMS_TO_PCMYR) * T
    tgt_pos, tgt_vel = _target_gc(P, [0.0, 0.0, 0.0])
    iso = {"name": "synthetic", "v_inf_kms": v_inf,
           "ra_radiant": ra_rad, "dec_radiant": dec_rad}
    target = {"name": "synthetic-star", "pos_gc": tgt_pos, "vel_gc": tgt_vel}
    out = mc_backtrack(iso, target, sigma_v=0.5, sigma_radiant_deg=0.5, n=300,
                       t_max_myr=10.0, dt_myr=0.1, d_close_pc=1.0, seed=1)
    assert out["consistent_with_origin"] is True
    assert out["d_min_pc"]["p50"] < 1.0
    assert out["frac_past"] > 0.9
    # The honesty caveat must always ship as a first-class field.
    assert "necessary_not_sufficient" in out
    assert "not exclud" in out["necessary_not_sufficient"].lower()


def test_mc_backtrack_wrong_radiant_not_consistent():
    # A target placed one way, an ISO aimed 90 deg off -> not consistent.
    ra_rad, dec_rad, v_inf = 279.8, 33.996, 26.0
    v_gal = radiant_to_velocity(v_inf, ra_rad, dec_rad)
    P = (-v_gal * _KMS_TO_PCMYR) * 1.0
    tgt_pos, tgt_vel = _target_gc(P, [0.0, 0.0, 0.0])
    iso = {"name": "wrong", "v_inf_kms": v_inf,
           "ra_radiant": ra_rad + 90.0, "dec_radiant": dec_rad}
    target = {"name": "star", "pos_gc": tgt_pos, "vel_gc": tgt_vel}
    out = mc_backtrack(iso, target, sigma_v=0.5, sigma_radiant_deg=0.5, n=300,
                       t_max_myr=10.0, dt_myr=0.2, d_close_pc=1.0, seed=2)
    assert out["consistent_with_origin"] is False

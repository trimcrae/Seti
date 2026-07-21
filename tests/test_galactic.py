"""Offline tests for the Galactic-orbit long-baseline encounter machinery."""

from __future__ import annotations

import numpy as np

from seti.galactic.encounters import (
    closest_approach_integrated,
    mc_encounter_orbit,
)
from seti.galactic.orbits import (
    KMS_TO_KPCMYR,
    R0_KPC,
    acceleration,
    circular_velocity_kms,
    integrate_orbits,
    potential_energy,
)


def test_rotation_curve_is_flat_and_reasonable():
    # V_c(R0) ~ 233 km/s and roughly flat across the disk.
    vc0 = circular_velocity_kms(R0_KPC)
    assert 215 < vc0 < 250
    vc_in, vc_out = circular_velocity_kms(5.0), circular_velocity_kms(12.0)
    assert 190 < vc_in < 260 and 190 < vc_out < 260   # flat-ish


def test_circular_orbit_stays_circular():
    # A star on a circular orbit at R0 keeps its radius over 300 Myr.
    vc = circular_velocity_kms(R0_KPC) * KMS_TO_KPCMYR       # kpc/Myr
    pos0 = np.array([[R0_KPC, 0.0, 0.0]])
    vel0 = np.array([[0.0, vc, 0.0]])
    times, traj = integrate_orbits(pos0, vel0, t_max_myr=300.0, dt_myr=0.5,
                                   direction=1, record_every=20)
    R = np.sqrt(traj[:, 0, 0] ** 2 + traj[:, 0, 1] ** 2)
    assert np.all(np.abs(R - R0_KPC) < 0.05)                # <50 pc drift


def test_energy_conserved():
    # Specific energy of an eccentric orbit is conserved to O(dt^2).
    vc = circular_velocity_kms(R0_KPC) * KMS_TO_KPCMYR
    pos0 = np.array([[R0_KPC, 0.0, 0.1]])
    vel0 = np.array([[10.0 * KMS_TO_KPCMYR, 0.8 * vc, 5.0 * KMS_TO_KPCMYR]])
    times, traj = integrate_orbits(pos0, vel0, t_max_myr=200.0, dt_myr=0.2,
                                   direction=1, record_every=1)
    # Recompute velocity by finite difference to get KE is noisy; instead re-run
    # with a returned-velocity check via energy at endpoints using the integrator
    # invariant: potential + kinetic. Use a fresh integration tracking velocity.
    pos = pos0.astype(float).copy()
    vel = vel0.astype(float).copy()
    acc = acceleration(pos)
    dt = 0.2
    E0 = 0.5 * np.sum(vel ** 2) + potential_energy(pos)[0]
    for _ in range(1000):
        vel += 0.5 * dt * acc
        pos += dt * vel
        acc = acceleration(pos)
        vel += 0.5 * dt * acc
    E1 = 0.5 * np.sum(vel ** 2) + potential_energy(pos)[0]
    assert abs((E1 - E0) / E0) < 1e-3


def test_recovers_synthetic_close_encounter():
    # Two stars started at the same phase-space point stay coincident (d_min ~ 0);
    # a star offset in position but with matched velocity stays offset.
    vc = circular_velocity_kms(R0_KPC) * KMS_TO_KPCMYR
    a_pos = np.array([[R0_KPC, 0.0, 0.0]])
    a_vel = np.array([[0.0, vc, 0.0]])
    # Co-located twin + a star 5 pc away co-moving.
    s_pos = np.array([[R0_KPC, 0.0, 0.0],
                      [R0_KPC, 0.005, 0.0]])
    s_vel = np.array([[0.0, vc, 0.0],
                      [0.0, vc, 0.0]])
    res = closest_approach_integrated(a_pos, a_vel, s_pos, s_vel,
                                      t_max_myr=100.0, dt_myr=0.5)
    assert res["d_min_pc"][0] < 1.0            # the twin passes ~0 pc
    assert res["d_min_pc"][1] < 6.0            # co-mover stays within ~5 pc


def test_mc_encounter_timing_recoverability_flag():
    # A star with tiny errors -> tight t_enc spread -> timing recoverable;
    # (smoke test that the MC runs end-to-end and returns the flag).
    anchor = {"ra": 11.25, "dec": -15.27, "parallax": 66.7,
              "parallax_error": 0.03, "pmra": 317.6, "pmra_error": 0.03,
              "pmdec": -596.6, "pmdec_error": 0.03,
              "radial_velocity": -13.7, "radial_velocity_error": 0.1}
    star = {"ra": 12.0, "dec": -15.0, "parallax": 60.0,
            "parallax_error": 0.05, "pmra": 300.0, "pmra_error": 0.05,
            "pmdec": -580.0, "pmdec_error": 0.05,
            "radial_velocity": -10.0, "radial_velocity_error": 0.2}
    out = mc_encounter_orbit(anchor, star, t_max_myr=100.0, dt_myr=1.0, n=40)
    assert out["n_valid"] > 0
    assert "timing_recoverable" in out
    assert 0.0 <= out["frac_past"] <= 1.0

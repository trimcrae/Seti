"""Axisymmetric Milky Way potential + vectorised leapfrog orbit integrator.

Frame and units.  We work in **Galactocentric Cartesian** coordinates with
positions in kpc, velocities in kpc/Myr, and masses in solar masses (so
``G = 4.4985e-12 kpc^3 Msun^-1 Myr^-2``).  The potential is a two-component
axisymmetric model -- a Miyamoto-Nagai disk plus a spherical logarithmic halo --
tuned to a flat rotation curve with ``V_c(R0) ~ 233 km/s``.  That is the standard,
well-behaved model for *local disk* dynamics over a few hundred Myr; it captures
the differential rotation and vertical tide that bend a straight-line encounter,
without pretending to a precision Galactic mass model the data cannot constrain.

Everything is vectorised over an array of ``N`` stars so the whole encounter
sample integrates on one time grid at once.
"""

from __future__ import annotations

import numpy as np

# --- constants -------------------------------------------------------------
G = 4.498502151e-12          # kpc^3 Msun^-1 Myr^-2
KMS_TO_KPCMYR = 1.0227121651e-3   # 1 km/s in kpc/Myr
KPCMYR_TO_KMS = 1.0 / KMS_TO_KPCMYR

# Sun's Galactocentric frame (GRAVITY 2019 R0; Schoenrich-ish solar motion).
R0_KPC = 8.178
Z_SUN_KPC = 0.0208
VSUN_GC_KMS = (11.1, 245.0, 7.25)     # (toward GC, rotation, NGP)

# --- potential parameters (flat rotation curve, V_c(R0) ~ 233 km/s) --------
_MN_M = 6.5e10       # Msun, Miyamoto-Nagai disk mass
_MN_A = 3.0          # kpc, disk scale length
_MN_B = 0.28         # kpc, disk scale height
_HALO_V0 = 164.0 * KMS_TO_KPCMYR   # kpc/Myr, logarithmic-halo asymptotic speed
_HALO_RC = 1.0       # kpc, halo core radius


def acceleration(pos: np.ndarray) -> np.ndarray:
    """Galactic acceleration (kpc/Myr^2) at Cartesian positions ``pos`` (N,3) kpc."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R2 = x * x + y * y
    R = np.sqrt(R2)
    # Miyamoto-Nagai disk.
    zb = np.sqrt(z * z + _MN_B * _MN_B)
    D = _MN_A + zb
    denom = np.power(R2 + D * D, 1.5) + 1e-12
    aR_disk = -G * _MN_M * R / denom
    az_disk = -G * _MN_M * D * (z / zb) / denom
    # Logarithmic halo: Phi = 0.5 v0^2 ln(Rc^2 + R^2 + z^2).
    s = _HALO_RC * _HALO_RC + R2 + z * z
    aR_halo = -_HALO_V0 * _HALO_V0 * R / s
    az_halo = -_HALO_V0 * _HALO_V0 * z / s
    aR = aR_disk + aR_halo
    az = az_disk + az_halo
    with np.errstate(invalid="ignore", divide="ignore"):
        ax = np.where(R > 0, aR * x / R, 0.0)
        ay = np.where(R > 0, aR * y / R, 0.0)
    return np.stack([ax, ay, az], axis=1)


def potential_energy(pos: np.ndarray) -> np.ndarray:
    """Specific potential energy (kpc/Myr)^2 at ``pos`` (N,3) -- for energy tests."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R2 = x * x + y * y
    zb = np.sqrt(z * z + _MN_B * _MN_B)
    phi_disk = -G * _MN_M / np.sqrt(R2 + (_MN_A + zb) ** 2)
    phi_halo = 0.5 * _HALO_V0 * _HALO_V0 * np.log(_HALO_RC ** 2 + R2 + z * z)
    return phi_disk + phi_halo


def circular_velocity_kms(R_kpc: float) -> float:
    """Circular speed (km/s) at Galactocentric radius ``R_kpc`` in the disk plane."""
    pos = np.array([[R_kpc, 0.0, 0.0]])
    a = acceleration(pos)[0]
    aR = -a[0]                       # inward radial acceleration magnitude
    vc = np.sqrt(max(aR, 0.0) * R_kpc)
    return float(vc * KPCMYR_TO_KMS)


def integrate_orbits(pos0: np.ndarray, vel0: np.ndarray, t_max_myr: float,
                     dt_myr: float = 0.2, direction: int = -1,
                     record_every: int = 1):
    """Leapfrog-integrate ``N`` orbits, returning sampled trajectories.

    ``pos0`` (N,3) kpc, ``vel0`` (N,3) kpc/Myr.  ``direction=-1`` integrates into
    the past.  Returns ``(times, traj)`` where ``times`` is (T,) Myr (signed, so
    negative into the past) and ``traj`` is (T,N,3) kpc sampled every
    ``record_every`` steps.  Velocity-Verlet (symplectic), so energy is conserved
    to O(dt^2) over the whole baseline.
    """
    pos = np.asarray(pos0, float).copy()
    vel = np.asarray(vel0, float).copy() * direction  # integrate |t|, flip v for past
    n_steps = int(np.ceil(abs(t_max_myr) / dt_myr))
    acc = acceleration(pos)
    times = [0.0]
    traj = [pos.copy()]
    for i in range(1, n_steps + 1):
        vel += 0.5 * dt_myr * acc
        pos += dt_myr * vel
        acc = acceleration(pos)
        vel += 0.5 * dt_myr * acc
        if i % record_every == 0 or i == n_steps:
            times.append(direction * i * dt_myr)
            traj.append(pos.copy())
    return np.array(times), np.stack(traj, axis=0)


def heliocentric_to_galactocentric(X_pc, Y_pc, Z_pc, U_kms, V_kms, W_kms):
    """Convert heliocentric Galactic (pc, km/s) to Galactocentric (kpc, kpc/Myr).

    Uses the module's Sun frame.  Heliocentric X is toward the Galactic centre, so
    the GC sits at heliocentric X = +R0; Galactocentric x = X_helio - R0 puts the
    Sun at x = -R0.  Velocities add the Sun's Galactocentric motion.
    """
    x = np.asarray(X_pc, float) / 1e3 - R0_KPC
    y = np.asarray(Y_pc, float) / 1e3
    z = np.asarray(Z_pc, float) / 1e3 + Z_SUN_KPC
    vx = (np.asarray(U_kms, float) + VSUN_GC_KMS[0]) * KMS_TO_KPCMYR
    vy = (np.asarray(V_kms, float) + VSUN_GC_KMS[1]) * KMS_TO_KPCMYR
    vz = (np.asarray(W_kms, float) + VSUN_GC_KMS[2]) * KMS_TO_KPCMYR
    pos = np.stack([x, y, z], axis=-1)
    vel = np.stack([vx, vy, vz], axis=-1)
    return np.atleast_2d(pos), np.atleast_2d(vel)


__all__ = ["G", "KMS_TO_KPCMYR", "KPCMYR_TO_KMS", "R0_KPC", "Z_SUN_KPC",
           "VSUN_GC_KMS", "acceleration", "potential_energy",
           "circular_velocity_kms", "integrate_orbits",
           "heliocentric_to_galactocentric"]

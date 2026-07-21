"""Pure-logic ISO back-tracking dynamics (offline unit-tested).

Given an interstellar object's asymptotic incoming speed ``v_infinity`` and its
radiant (the sky direction it *came from*), we build the ISO's heliocentric
Galactic velocity vector, place the ISO at the Solar System (heliocentric origin,
since it passed through it ~now on Myr timescales), and integrate its orbit
**backward through the Galactic potential** shared with the ``galactic`` channel
-- reusing :func:`seti.galactic.encounters.closest_approach_integrated` and the
:mod:`seti.galactic.orbits` integrator rather than duplicating any dynamics.

The headline product is a Monte-Carlo distribution of the closest-approach
distance between the ISO's back-track and a target star (LHS 1140), propagating
the *large* velocity + radiant uncertainties.

READ THIS BEFORE BELIEVING ANY NUMBER HERE.  A small closest-approach distance is
**necessary but not sufficient** for a common origin:

* The radiant of a fast ISO is only known to a few degrees, and a few degrees at
  the ~10-20 pc distance of the nearest stars is already several parsecs of
  transverse smear -- growing with lookback time as phase mixing sets in.
* 'Oumuamua's radiant lies near the solar apex, the direction the Sun itself moves
  toward, so a *large fraction of all nearby disk stars* lie along its back-track
  purely by projection.  Passing close to one of them is the expected null, not a
  detection.
* The astrophysical prior overwhelmingly favours a generic Galactic-disk origin
  (the phase-space density of unremarkable field stars along any track dwarfs the
  prior on one named system).

So :func:`mc_backtrack` returns a ``consistent_with_origin`` flag that can be
``True`` only when a large fraction of Monte-Carlo draws pass within a tight
distance in a plausible past transit time -- and *even then* it means "not
excluded", never "originated at".  All functions are pure NumPy.
"""

from __future__ import annotations

import numpy as np

from ..galactic.encounters import closest_approach_integrated
from ..galactic.orbits import (
    KMS_TO_KPCMYR,
    heliocentric_to_galactocentric,
)
from ..panspermia.kinematics import _A_ICRS_TO_GAL

__all__ = [
    "radiant_to_velocity",
    "iso_galactocentric",
    "backtrack_closest_approach",
    "mc_backtrack",
]


def radiant_to_velocity(v_inf_kms, ra_radiant_deg, dec_radiant_deg) -> np.ndarray:
    """Convert an ISO's incoming speed + radiant into a heliocentric Galactic
    velocity vector ``(U, V, W)`` in km/s.

    SIGN CONVENTION (the whole point of this function).  A *radiant* is the
    direction on the sky the object **came from** (the meteor-astronomy
    convention): the ISO's velocity points in the *opposite* direction, toward the
    anti-radiant.  So with the ICRS radiant unit vector

        ``r_hat = [cos(dec) cos(ra), cos(dec) sin(ra), sin(dec)]``

    the ISO's heliocentric velocity in ICRS Cartesian is ``-v_inf * r_hat``, which
    we then rotate into the Galactic frame with the same ICRS->Galactic matrix the
    kinematics module uses (so this channel shares one Galactic frame with the
    encounter search).  ``v_inf`` is the *heliocentric* asymptotic speed (relative
    to the Sun), which is exactly the frame :func:`heliocentric_to_galactocentric`
    expects before it adds the Sun's Galactocentric motion.

    Returns a length-3 array ``(U, V, W)`` km/s.  Broadcasts over array inputs,
    returning shape ``(N, 3)``.
    """
    v_inf = np.asarray(v_inf_kms, float)
    ra = np.radians(np.asarray(ra_radiant_deg, float))
    dec = np.radians(np.asarray(dec_radiant_deg, float))
    # ICRS unit vector toward the radiant (where it came from).
    r_hat = np.stack([np.cos(dec) * np.cos(ra),
                      np.cos(dec) * np.sin(ra),
                      np.sin(dec)], axis=-1)                    # (..., 3)
    # Velocity points AWAY from the radiant: anti-radiant direction.
    v_icrs = -v_inf[..., None] * r_hat if r_hat.ndim > 1 else -v_inf * r_hat
    # Rotate ICRS -> Galactic (matrix acts on the last axis).
    v_gal = v_icrs @ _A_ICRS_TO_GAL.T
    return v_gal


def iso_galactocentric(vel_uvw, pos_xyz_pc=(0.0, 0.0, 0.0)):
    """Galactocentric phase space of an ISO from its heliocentric Galactic velocity.

    ``vel_uvw`` is ``(U, V, W)`` km/s; ``pos_xyz_pc`` its heliocentric Galactic
    position in pc (default the Solar System origin -- an ISO passing through the
    inner Solar System is at heliocentric ~0 on Myr timescales).  Returns
    ``(pos, vel)`` each shaped ``(1, 3)`` in kpc and kpc/Myr, ready for the shared
    orbit integrator.
    """
    vel_uvw = np.asarray(vel_uvw, float)
    X, Y, Z = pos_xyz_pc
    U, V, W = vel_uvw[..., 0], vel_uvw[..., 1], vel_uvw[..., 2]
    pos, vel = heliocentric_to_galactocentric(X, Y, Z, U, V, W)
    return pos, vel


def backtrack_closest_approach(iso_vel_uvw, target_pos_gc, target_vel_gc,
                               t_max_myr: float = 200.0, dt_myr: float = 0.5,
                               iso_pos_xyz_pc=(0.0, 0.0, 0.0)) -> dict:
    """Minimum separation between an ISO's back-track and a target star's orbit.

    Integrates both the ISO (starting at ``iso_pos_xyz_pc`` heliocentric with
    heliocentric Galactic velocity ``iso_vel_uvw`` km/s) and the target star
    (Galactocentric ``target_pos_gc`` kpc, ``target_vel_gc`` kpc/Myr) *backward*
    through the Galactic potential on one shared time grid, via the reused
    :func:`closest_approach_integrated`.  Returns the running-minimum separation
    ``d_min_pc``, the signed lookback time ``t_enc_myr`` (<0 = in the past) at
    which it occurs, and the present-day separation ``sep_now_pc``.

    A small ``d_min_pc`` is necessary-not-sufficient for a common origin (see the
    module docstring); this function reports geometry only, no origin claim.
    """
    iso_pos, iso_vel = iso_galactocentric(iso_vel_uvw, iso_pos_xyz_pc)
    tgt_pos = np.atleast_2d(np.asarray(target_pos_gc, float))
    tgt_vel = np.atleast_2d(np.asarray(target_vel_gc, float))
    res = closest_approach_integrated(tgt_pos, tgt_vel, iso_pos, iso_vel,
                                      t_max_myr=t_max_myr, dt_myr=dt_myr,
                                      direction=-1)
    return {"d_min_pc": float(res["d_min_pc"][0]),
            "t_enc_myr": float(res["t_enc_myr"][0]),
            "sep_now_pc": float(res["sep_now_pc"][0])}


def _pct(x, ps=(2.5, 16, 50, 84, 97.5)):
    return {f"p{p:g}".replace(".", "_"): float(np.percentile(x, p)) for p in ps}


def mc_backtrack(iso: dict, target: dict, sigma_v: float = 2.0,
                 sigma_radiant_deg: float = 3.0, n: int = 500,
                 t_max_myr: float = 200.0, dt_myr: float = 1.0,
                 d_close_pc: float = 1.0, frac_thresh: float = 0.5,
                 transit_max_myr: float | None = None, seed: int = 0) -> dict:
    """Monte-Carlo the ISO->target closest approach over the ISO's uncertainties.

    ``iso`` carries ``v_inf_kms``, ``ra_radiant`` and ``dec_radiant`` (deg).
    ``target`` carries Galactocentric ``pos_gc`` (3,) kpc and ``vel_gc`` (3,)
    kpc/Myr (e.g. LHS 1140, resolved once from Gaia and transformed).  We draw
    ``n`` realisations of the ISO's asymptotic speed (Gaussian, ``sigma_v`` km/s,
    truncated at 0) and radiant (isotropic Gaussian scatter of ``sigma_radiant_deg``
    on the sphere), convert each to a Galactic velocity, and integrate all draws
    against the (fixed) target in one vectorised backward integration.

    Returns percentiles of ``d_min`` (pc) and ``t_enc`` (Myr, <0 past), the
    fraction of draws that are a *past* encounter, the fraction passing within
    ``d_close_pc`` in a plausible transit time, and a ``consistent_with_origin``
    flag.

    THE FLAG IS NOT A DETECTION.  ``consistent_with_origin`` is ``True`` only when
    more than ``frac_thresh`` of the draws pass within ``d_close_pc`` of the target
    *in the past* within ``transit_max_myr`` -- i.e. when a common origin is not
    *excluded* by the geometry.  It is necessary-not-sufficient: the solar-apex
    projection effect and the overwhelming Galactic-disk prior mean that even a
    ``True`` flag leaves a generic field-star origin far more probable than the
    named target.  Never report this as "the ISO came from" the target.
    """
    rng = np.random.default_rng(seed)
    transit_max_myr = abs(t_max_myr) if transit_max_myr is None else abs(transit_max_myr)

    # --- draw v_inf and scatter the radiant on the sphere -------------------
    v_inf = np.clip(rng.normal(float(iso["v_inf_kms"]), float(sigma_v), n), 1e-3, None)
    ra0 = np.radians(float(iso["ra_radiant"]))
    dec0 = np.radians(float(iso["dec_radiant"]))
    sig = np.radians(float(sigma_radiant_deg))
    # ICRS local triad at the nominal radiant: radial + east(+RA) + north(+Dec).
    r_hat = np.array([np.cos(dec0) * np.cos(ra0),
                      np.cos(dec0) * np.sin(ra0), np.sin(dec0)])
    e_hat = np.array([-np.sin(ra0), np.cos(ra0), 0.0])         # +RA (east)
    n_hat = np.array([-np.sin(dec0) * np.cos(ra0),
                      -np.sin(dec0) * np.sin(ra0), np.cos(dec0)])  # +Dec (north)
    de = rng.normal(0.0, sig, n)
    dn = rng.normal(0.0, sig, n)
    # Small-angle tangent-plane scatter, renormalised to the unit sphere.
    rad = (r_hat[None, :] + de[:, None] * e_hat[None, :]
           + dn[:, None] * n_hat[None, :])
    rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    # ISO velocity = anti-radiant * speed, rotated to Galactic.
    v_icrs = -v_inf[:, None] * rad
    v_gal = v_icrs @ _A_ICRS_TO_GAL.T                          # (n, 3) km/s (U,V,W)

    # --- integrate all draws against the fixed target -----------------------
    iso_pos, iso_vel = heliocentric_to_galactocentric(
        np.zeros(n), np.zeros(n), np.zeros(n),
        v_gal[:, 0], v_gal[:, 1], v_gal[:, 2])
    tgt_pos = np.atleast_2d(np.asarray(target["pos_gc"], float))
    tgt_vel = np.atleast_2d(np.asarray(target["vel_gc"], float))
    res = closest_approach_integrated(tgt_pos, tgt_vel, iso_pos, iso_vel,
                                      t_max_myr=t_max_myr, dt_myr=dt_myr,
                                      direction=-1)
    d_min = res["d_min_pc"]
    t_enc = res["t_enc_myr"]

    past = t_enc < 0
    close = (d_min < d_close_pc) & past & (np.abs(t_enc) < transit_max_myr)
    frac_close = float(np.mean(close))
    consistent = bool(frac_close > frac_thresh)

    out = {
        "name": iso.get("name", "ISO"),
        "target": target.get("name", "target"),
        "n_draws": int(n),
        "sigma_v_kms": float(sigma_v),
        "sigma_radiant_deg": float(sigma_radiant_deg),
        "t_max_myr": float(t_max_myr),
        "d_close_pc": float(d_close_pc),
        "d_min_pc": _pct(d_min),
        "t_enc_myr": _pct(t_enc),
        "frac_past": float(np.mean(past)),
        "frac_within_d_close": frac_close,
        "consistent_with_origin": consistent,
        "necessary_not_sufficient": (
            "A close pass is NECESSARY BUT NOT SUFFICIENT for a common origin. "
            "The radiant is uncertain by degrees (parsecs of transverse smear), "
            "fast-ISO radiants project near the solar apex so many disk stars lie "
            "along the track, and the Galactic-disk prior overwhelmingly favours a "
            "generic field-star origin. consistent_with_origin=True means 'not "
            "excluded', never 'originated at'."),
    }
    return out

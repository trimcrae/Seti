"""Closest approach of stars to an anchor over Galactic-orbit-integrated paths.

Given an anchor (LHS 1140 / K2-18) and a sample of stars, all with Galactocentric
phase space, we integrate every orbit backward on one shared time grid and track,
per star, the *running minimum* separation to the anchor and the lookback time at
which it occurs.  Storing only the running minimum keeps the memory flat even for
thousands of stars over a few hundred Myr at fine time resolution.

Because differential rotation makes the separation oscillate (stars on nearby
guiding radii lap each other), the "closest approach" is the global minimum over
the window, and its *timing* is only meaningful while the Monte-Carlo spread stays
small -- which :func:`mc_encounter_orbit` measures directly, giving an honest
recoverability horizon rather than a false precise date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..panspermia.kinematics import phase_space_6d
from ..panspermia.uncertainty import _sample
from .orbits import acceleration, heliocentric_to_galactocentric

_KPC_TO_PC = 1000.0


def closest_approach_integrated(anchor_pos, anchor_vel, star_pos, star_vel,
                                t_max_myr: float, dt_myr: float = 0.2,
                                direction: int = -1) -> dict:
    """Running-minimum separation of each star to the anchor over integrated orbits.

    ``anchor_pos/vel`` are (3,) or (1,3); ``star_pos/vel`` are (N,3), all in kpc
    and kpc/Myr (Galactocentric).  Integrates both with a shared velocity-Verlet
    step and returns per-star ``d_min_pc``, ``t_enc_myr`` (signed lookback), and
    ``sep_now_pc``.
    """
    a_pos = np.atleast_2d(anchor_pos).astype(float).copy()
    a_vel = np.atleast_2d(anchor_vel).astype(float).copy() * direction
    s_pos = np.atleast_2d(star_pos).astype(float).copy()
    s_vel = np.atleast_2d(star_vel).astype(float).copy() * direction
    # Stack anchor as row 0 so one integration advances everything together.
    pos = np.concatenate([a_pos, s_pos], axis=0)
    vel = np.concatenate([a_vel, s_vel], axis=0)

    sep0 = np.linalg.norm(pos[1:] - pos[0], axis=1) * _KPC_TO_PC
    d_min = sep0.copy()
    t_enc = np.zeros(len(s_pos))
    n_steps = int(np.ceil(abs(t_max_myr) / dt_myr))
    acc = acceleration(pos)
    for i in range(1, n_steps + 1):
        # Analytic closest approach *within* this step (from relative pos+vel),
        # so a fast encounter between sampled points is not stepped over.
        seg_sep, seg_frac = _segment_min(pos, vel, dt_myr)
        t_here = direction * ((i - 1) + seg_frac) * dt_myr
        hit = seg_sep < d_min
        d_min = np.where(hit, seg_sep, d_min)
        t_enc = np.where(hit, t_here, t_enc)
        vel += 0.5 * dt_myr * acc
        pos += dt_myr * vel
        acc = acceleration(pos)
        vel += 0.5 * dt_myr * acc
    # Final endpoint separation.
    sep = np.linalg.norm(pos[1:] - pos[0], axis=1) * _KPC_TO_PC
    hit = sep < d_min
    d_min = np.where(hit, sep, d_min)
    t_enc = np.where(hit, direction * n_steps * dt_myr, t_enc)
    return {"d_min_pc": d_min, "t_enc_myr": t_enc, "sep_now_pc": sep0}


def _segment_min(pos, vel, dt_myr):
    """Min separation of stars (rows 1:) to anchor (row 0) over [0, dt] of the
    local straight-line segment, from current relative position and velocity.

    Returns ``(sep_pc, frac)`` where ``frac`` in [0,1] is where in the step the
    minimum falls.  This resolves a sub-step close approach a fast flyby would
    otherwise skip between sampled points."""
    dr = (pos[1:] - pos[0])                         # kpc
    dv = (vel[1:] - vel[0])                         # kpc/Myr
    dv2 = np.einsum("ij,ij->i", dv, dv)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstar = -np.einsum("ij,ij->i", dr, dv) / dv2
    tstar = np.clip(np.where(dv2 > 0, tstar, 0.0), 0.0, dt_myr)
    closest = dr + dv * tstar[:, None]
    sep = np.linalg.norm(closest, axis=1) * _KPC_TO_PC
    return sep, tstar / dt_myr


def closest_approach_from_helio(anchor: dict, stars: pd.DataFrame,
                                t_max_myr: float = 300.0,
                                dt_myr: float = 0.2) -> pd.DataFrame:
    """Orbit-integrated closest approach to ``anchor`` for a table of stars.

    ``anchor`` and ``stars`` carry heliocentric Galactic ``X_pc,Y_pc,Z_pc`` and
    ``U_kms,V_kms,W_kms`` (from :func:`phase_space_6d`).  Adds ``sep_now_pc``,
    ``d_min_pc`` and ``t_enc_myr`` (lookback, <0 past) integrated in the Galactic
    potential.
    """
    a_pos, a_vel = heliocentric_to_galactocentric(
        anchor["X_pc"], anchor["Y_pc"], anchor["Z_pc"],
        anchor["U_kms"], anchor["V_kms"], anchor["W_kms"])
    s_pos, s_vel = heliocentric_to_galactocentric(
        stars["X_pc"].to_numpy(float), stars["Y_pc"].to_numpy(float),
        stars["Z_pc"].to_numpy(float), stars["U_kms"].to_numpy(float),
        stars["V_kms"].to_numpy(float), stars["W_kms"].to_numpy(float))
    res = closest_approach_integrated(a_pos, a_vel, s_pos, s_vel,
                                      t_max_myr=t_max_myr, dt_myr=dt_myr)
    out = stars.copy()
    out["sep_now_pc"] = res["sep_now_pc"]
    out["d_min_pc"] = res["d_min_pc"]
    out["t_enc_myr"] = res["t_enc_myr"]
    return out


def mc_encounter_orbit(anchor: dict, star: dict, t_max_myr: float = 300.0,
                       dt_myr: float = 0.5, n: int = 200, seed: int = 0) -> dict:
    """Monte-Carlo the orbit-integrated encounter of one ``star`` with ``anchor``.

    Resamples both from their Gaia (parallax, pmra, pmdec, RV) errors, integrates
    each realisation, and reports percentiles of ``d_min`` and ``t_enc`` plus a
    **timing-recoverable** flag: the encounter *time* is trustworthy only if the
    16-84% spread in ``t_enc`` is a small fraction of the lookback window (else
    phase mixing has erased it, even though ``d_min`` may still be robust).
    """
    rng = np.random.default_rng(seed)
    a = phase_space_6d(_sample(anchor, n, rng))
    s = phase_space_6d(_sample(star, n, rng))
    good = (np.isfinite(a[["U_kms", "V_kms", "W_kms"]].to_numpy()).all(1)
            & np.isfinite(s[["U_kms", "V_kms", "W_kms"]].to_numpy()).all(1))
    a, s = a[good].reset_index(drop=True), s[good].reset_index(drop=True)
    if not len(a):
        return {"n_valid": 0}
    a_pos, a_vel = heliocentric_to_galactocentric(
        a["X_pc"], a["Y_pc"], a["Z_pc"], a["U_kms"], a["V_kms"], a["W_kms"])
    s_pos, s_vel = heliocentric_to_galactocentric(
        s["X_pc"], s["Y_pc"], s["Z_pc"], s["U_kms"], s["V_kms"], s["W_kms"])
    m = len(a_pos)
    # Integrate each (anchor_i, star_i) realisation pair and difference i-to-i.
    d_min, t_enc = _pairwise_min(a_pos, a_vel, s_pos, s_vel, t_max_myr, dt_myr)

    def pct(x):
        return [float(np.percentile(x, p)) for p in (16, 50, 84)]
    dlo, dmed, dhi = pct(d_min)
    tlo, tmed, thi = pct(t_enc)
    t_spread = thi - tlo
    return {
        "n_valid": int(m),
        "d_min_p16": dlo, "d_min_p50": dmed, "d_min_p84": dhi,
        "t_enc_p16": tlo, "t_enc_p50": tmed, "t_enc_p84": thi,
        "t_enc_spread_myr": float(t_spread),
        "frac_past": float(np.mean(t_enc < 0)),
        "timing_recoverable": bool(abs(t_spread) < 0.25 * abs(t_max_myr)),
    }


def _pairwise_min(a_pos, a_vel, s_pos, s_vel, t_max_myr, dt_myr, direction=-1):
    """Per-realisation (i-to-i) running-min separation over integrated orbits."""
    pos = np.concatenate([a_pos, s_pos], axis=0).astype(float).copy()
    vel = np.concatenate([a_vel, s_vel], axis=0).astype(float).copy() * direction
    m = len(a_pos)
    sep0 = np.linalg.norm(pos[m:] - pos[:m], axis=1) * _KPC_TO_PC
    d_min = sep0.copy()
    t_enc = np.zeros(m)
    n_steps = int(np.ceil(abs(t_max_myr) / dt_myr))
    acc = acceleration(pos)
    for i in range(1, n_steps + 1):
        dr = pos[m:] - pos[:m]
        dv = vel[m:] - vel[:m]
        dv2 = np.einsum("ij,ij->i", dv, dv)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstar = np.clip(np.where(dv2 > 0,
                                     -np.einsum("ij,ij->i", dr, dv) / dv2, 0.0),
                            0.0, dt_myr)
        sep = np.linalg.norm(dr + dv * tstar[:, None], axis=1) * _KPC_TO_PC
        hit = sep < d_min
        d_min = np.where(hit, sep, d_min)
        t_enc = np.where(hit, direction * ((i - 1) + tstar / dt_myr) * dt_myr, t_enc)
        vel += 0.5 * dt_myr * acc
        pos += dt_myr * vel
        acc = acceleration(pos)
        vel += 0.5 * dt_myr * acc
    return d_min, t_enc


__all__ = ["closest_approach_integrated", "closest_approach_from_helio",
           "mc_encounter_orbit"]

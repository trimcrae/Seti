"""Close stellar encounters with a fixed anchor star, and a transfer score.

The panspermia question, made concrete: *which stars passed close to K2-18, at
low relative speed, in the recent past?*  Those are the only systems into which
K2-18-origin material (impact ejecta, dormant spores, free-floating bodies of the
'Oumuamua/Borisov class) could plausibly have been delivered -- the transfer
vector need not be a continuous bridge, only a close, slow flyby whose geometry
lets one star's outer reservoir dump material the other can capture.

Method.  Over the recent past (a few Myr) the Galactic tide is negligible on the
scale of the local neighbourhood, so each star moves on a straight line at
constant velocity to good approximation -- the standard treatment for the Sun's
own encounter list (Garcia-Sanchez 2001; Bailer-Jones 2015+).  For a star with
relative position ``dr`` (pc) and relative velocity ``dv`` (pc/Myr) with respect
to the anchor, the separation ``|dr + dv t|`` is minimised at

    t_enc = - (dr . dv) / (dv . dv)          [Myr; negative = in the past]
    d_min = | dr + dv * t_enc |              [pc]

and the relative speed at closest approach equals ``|dv|`` (constant, straight
line).  ``t_enc < 0`` is a *past* encounter -- the only kind that could already
have seeded a neighbour.

The transfer score is deliberately a *ranking* heuristic, not a probability:
material capture during a flyby grows as the encounter is closer (smaller
``d_min``) and slower (capture cross-section rises steeply as the relative speed
falls toward the reservoir escape speed).  We use

    score = (d_ref / d_min) * (v_ref / v_rel)^2

gated to past encounters inside a viability window ``|t_enc| <= t_max_myr``.  The
squared velocity dependence mirrors the gravitational-capture cross-section; the
constants ``d_ref``, ``v_ref`` only set the scale of an ordinal score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 1 km/s expressed in pc/Myr (1 km/s = 1.0227121651 pc/Myr).
_KMS_TO_PC_PER_MYR = 1.0227121651


def closest_approach(anchor: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Linear closest-approach of every star in ``df`` to ``anchor``.

    ``anchor`` and ``df`` must carry Galactic Cartesian position ``X_pc,Y_pc,Z_pc``
    (pc) and velocity ``U_kms,V_kms,W_kms`` (km/s).  Adds:

    * ``sep_now_pc``   present-day separation (pc);
    * ``v_rel_kms``    relative speed (km/s), constant along the straight line;
    * ``t_enc_myr``    time of closest approach (Myr; <0 past, >0 future);
    * ``d_min_pc``     closest-approach separation (pc).
    """
    out = df.copy()
    r0 = np.array([anchor["X_pc"], anchor["Y_pc"], anchor["Z_pc"]], float)
    v0 = np.array([anchor["U_kms"], anchor["V_kms"], anchor["W_kms"]], float)

    r = df[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)                 # (N,3) pc
    v = df[["U_kms", "V_kms", "W_kms"]].to_numpy(float)             # (N,3) km/s

    dr = r - r0                                                     # pc
    dv = (v - v0) * _KMS_TO_PC_PER_MYR                              # pc/Myr

    v_rel_kms = np.linalg.norm(v - v0, axis=1)
    dv2 = np.einsum("ij,ij->i", dv, dv)                            # (pc/Myr)^2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_enc = -np.einsum("ij,ij->i", dr, dv) / dv2               # Myr
    t_enc = np.where(dv2 > 0, t_enc, np.nan)
    closest = dr + dv * t_enc[:, None]
    d_min = np.linalg.norm(closest, axis=1)

    out["sep_now_pc"] = np.linalg.norm(dr, axis=1)
    out["v_rel_kms"] = v_rel_kms
    out["t_enc_myr"] = t_enc
    out["d_min_pc"] = d_min
    return out


def transfer_score(df: pd.DataFrame, t_max_myr: float = 10.0,
                   d_ref_pc: float = 1.0, v_ref_kms: float = 1.0,
                   d_floor_pc: float = 0.01, v_floor_kms: float = 0.1) -> pd.DataFrame:
    """Ordinal panspermia-transfer score for past close encounters.

    Requires ``d_min_pc``, ``v_rel_kms``, ``t_enc_myr`` (from
    :func:`closest_approach`).  Only *past* encounters (``t_enc_myr < 0``) within
    ``|t_enc| <= t_max_myr`` score above zero; everything else is 0.  Floors on
    ``d_min`` and ``v_rel`` keep a (numerically) grazing zero-velocity match from
    producing an infinite score.
    """
    out = df.copy()
    d = np.maximum(pd.to_numeric(out["d_min_pc"], errors="coerce"), d_floor_pc)
    vrel = np.maximum(pd.to_numeric(out["v_rel_kms"], errors="coerce"), v_floor_kms)
    t = pd.to_numeric(out["t_enc_myr"], errors="coerce")

    past = np.isfinite(t) & (t < 0) & (t >= -t_max_myr)
    score = (d_ref_pc / d) * (v_ref_kms / vrel) ** 2
    out["transfer_score"] = np.where(past, score, 0.0)
    out["past_encounter"] = past
    return out


_GM_SUN_KM3_S2 = 1.32712440018e11     # G * M_sun, km^3 s^-2
_PC_KM = 3.0856775815e13              # 1 pc in km
_AU_PER_PC = 206264.806


def transfer_regime(df: pd.DataFrame, donor_mass_msun: float = 0.36,
                    reservoir_pc: float = 0.2) -> pd.DataFrame:
    """Classify each encounter by which *passive* transfer mode (if any) works.

    Two physically distinct channels deliver donor material to a passing star,
    and they scale oppositely with relative speed:

    * **Capture** (the slow channel): the passer gravitationally binds loosely-held
      donor material.  Feasible only when the relative speed is below the escape
      speed at the reservoir edge, ``v_esc = sqrt(2 G M / r)`` -- for a low-mass
      M-dwarf donor this threshold is *metres per second* at Oort distances, so it
      demands an almost co-moving pass.
    * **Interception** (the speed-independent channel): the passer physically
      ploughs through the donor's reservoir.  Feasible only when the
      closest-approach distance ``d_min`` is smaller than the reservoir radius --
      independent of how fast the pass is.  This is the mode a *fast* encounter
      could still satisfy, which is why it is worth separating out.

    Also reports the gravitational-focusing enhancement of the capture
    cross-section, ``1 + (v_esc/v_rel)^2``; for a fast pass this collapses to ~1
    (pure geometry, no focusing help).  Requires ``v_rel_kms`` and ``d_min_pc``.
    """
    out = df.copy()
    vrel = pd.to_numeric(out["v_rel_kms"], errors="coerce").to_numpy(float)
    dmin = pd.to_numeric(out["d_min_pc"], errors="coerce").to_numpy(float)
    # Escape speed at the ACTUAL closest-approach distance: the passer can bind
    # donor material located where it passed only if it is slower than this.
    with np.errstate(divide="ignore", invalid="ignore"):
        v_esc_dmin = np.sqrt(2.0 * _GM_SUN_KM3_S2 * donor_mass_msun
                             / (dmin * _PC_KM))                  # km/s
        focus = 1.0 + (v_esc_dmin / vrel) ** 2
    out["d_min_au"] = dmin * _AU_PER_PC
    out["v_esc_at_dmin_kms"] = v_esc_dmin
    # Two independent necessary conditions; a real passive transfer needs BOTH:
    out["within_reservoir"] = dmin < reservoir_pc          # donor has material this far out
    out["capturable"] = vrel < v_esc_dmin                  # slow enough to bind it there
    out["focusing_factor"] = focus
    out["transfers"] = out["within_reservoir"] & out["capturable"]
    return out


def regime_summary(df: pd.DataFrame, donor_mass_msun: float = 0.36,
                   reservoir_pc: float = 0.2) -> dict:
    """Aggregate :func:`transfer_regime` over past encounters into a verdict."""
    reg = transfer_regime(df, donor_mass_msun, reservoir_pc)
    past = reg[pd.to_numeric(reg["t_enc_myr"], errors="coerce") < 0]
    vrel = pd.to_numeric(past["v_rel_kms"], errors="coerce")
    dmin = pd.to_numeric(past["d_min_pc"], errors="coerce")
    # At the closest pass in the sample, how many times too fast was it to capture?
    if len(past):
        closest = past.loc[dmin.idxmin()]
        speed_excess = float(closest["v_rel_kms"] / closest["v_esc_at_dmin_kms"])
    else:
        speed_excess = None
    return {
        "donor_mass_msun": donor_mass_msun,
        "reservoir_pc": reservoir_pc,
        "n_past": int(len(past)),
        "v_rel_min_kms": float(vrel.min()) if len(past) else None,
        "v_rel_median_kms": float(vrel.median()) if len(past) else None,
        "d_min_min_pc": float(dmin.min()) if len(past) else None,
        "d_min_min_au": float(dmin.min() * _AU_PER_PC) if len(past) else None,
        "n_within_reservoir": int(past["within_reservoir"].sum()),
        "n_capturable": int(past["capturable"].sum()),
        "n_transfers": int(past["transfers"].sum()),
        "closest_pass_speed_excess": speed_excess,   # v_rel / v_esc(d_min) at the closest pass
        "max_focusing_factor": float(past["focusing_factor"].replace(
            [np.inf, -np.inf], np.nan).max()) if len(past) else None,
    }


def flag_comoving(df: pd.DataFrame, v_rel_max_kms: float = 3.0,
                  sep_now_max_pc: float = 5.0) -> pd.DataFrame:
    """Tag stars that are *currently* close and share the anchor's velocity.

    A persistent low-relative-velocity companion (a shared kinematic stream, a
    dissolving natal cluster, a wide co-moving pair) is the strongest panspermia
    bridge of all: the two reservoirs stay within reach for a long time rather
    than for one fleeting flyby.  This is the concrete form of "life need not be
    confined to a planet" -- material shed into a shared, slowly-drifting halo has
    a standing chance of capture, not a one-shot one.
    """
    out = df.copy()
    vrel = pd.to_numeric(out.get("v_rel_kms"), errors="coerce")
    sep = pd.to_numeric(out.get("sep_now_pc"), errors="coerce")
    out["comoving"] = (vrel <= v_rel_max_kms) & (sep <= sep_now_max_pc)
    return out


__all__ = ["closest_approach", "transfer_score", "flag_comoving",
           "transfer_regime", "regime_summary"]

"""Per-candidate vetting: ancestry (chemistry) and an error-honest rendezvous test.

Two independent axes, per ``docs/herdsman.md``:

* **Ancestry.**  Every natural mechanism that focuses stellar orbits acts on
  co-natal stars (cluster birth, tail epicycles, traceback of moving groups) or
  on exactly two discrete populations (cluster–cluster collisions).  The Gaia
  GSP-Phot metallicities of a real herd of *gathered* stars should span the
  field distribution; a co-natal group is chemically homogeneous at the
  ~0.05 dex level.  We flag, never cut — a chemically uniform convergence is
  downgraded, not deleted, because GSP-Phot has type-dependent systematics.

* **Rendezvous Monte Carlo.**  The detector's ball statistic asks "are these
  stars co-located beyond chance"; this test asks the complementary question,
  "is a common space-time point actually consistent with the *measurements*".
  Members are re-propagated under draws of their full astrometric + RV errors;
  the distribution of the minimum rms radius and its epoch quantifies how
  point-like the meeting can be and how well its time is determined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..galactic.orbits import heliocentric_to_galactocentric
from ..panspermia.kinematics import phase_space_6d
from .convergence import propagate


def chemistry_vet(mh: np.ndarray) -> dict:
    """Metallicity-spread verdict for one candidate's members."""
    x = np.asarray(mh, float)
    x = x[np.isfinite(x)]
    out = {"n_mh": int(len(x)), "mh_mad_dex": float("nan"),
           "mh_range_dex": float("nan"), "co_natal_possible": None,
           "heterogeneous": None}
    if len(x) >= 3:
        med = np.median(x)
        out["mh_mad_dex"] = float(1.4826 * np.median(np.abs(x - med)))
        out["mh_range_dex"] = float(x.max() - x.min())
        out["co_natal_possible"] = bool(out["mh_mad_dex"] < 0.05
                                        and out["mh_range_dex"] < 0.15)
        out["heterogeneous"] = bool(out["mh_mad_dex"] > 0.12
                                    or out["mh_range_dex"] > 0.35)
    return out


def rendezvous_mc(members: pd.DataFrame, direction: int, t_scan_myr: float,
                  dt_myr: float = 0.25, rec_every: int = 2, n_draws: int = 400,
                  astro_floor_kms: float = 0.3, seed: int = 7) -> dict:
    """Monte-Carlo the members' meeting under their measurement errors.

    Returns the distribution of the minimum rms radius over the scan window and
    of the epoch at which it occurs.  ``t_scan_myr`` should comfortably bracket
    the detection epoch (the caller passes ~1.5x |t_detect|).
    """
    rng = np.random.default_rng(seed)
    m = len(members)
    reps = pd.concat([members] * n_draws, ignore_index=True)

    def _num(col):
        return pd.to_numeric(reps[col], errors="coerce").to_numpy(float)

    def _err(col):
        if col in reps.columns:
            e = pd.to_numeric(reps[col], errors="coerce").to_numpy(float)
            return np.where(np.isfinite(e), e, 0.0)
        return np.zeros(len(reps))

    pert = reps.copy()
    pert["parallax"] = np.maximum(
        _num("parallax") + rng.standard_normal(len(reps)) * _err("parallax_error"),
        0.05)
    pert["pmra"] = _num("pmra") + rng.standard_normal(len(reps)) * _err("pmra_error")
    pert["pmdec"] = _num("pmdec") + rng.standard_normal(len(reps)) * _err("pmdec_error")
    rv_sig = np.sqrt(_err("radial_velocity_error") ** 2 + astro_floor_kms ** 2)
    pert["radial_velocity"] = (_num("radial_velocity")
                               + rng.standard_normal(len(reps)) * rv_sig)

    ps = phase_space_6d(pert)
    pos_kpc, vel = heliocentric_to_galactocentric(
        ps["X_pc"].to_numpy(), ps["Y_pc"].to_numpy(), ps["Z_pc"].to_numpy(),
        ps["U_kms"].to_numpy(), ps["V_kms"].to_numpy(), ps["W_kms"].to_numpy())

    best_rms = np.full(n_draws, np.inf)
    best_t = np.zeros(n_draws)
    for t, pos in propagate(pos_kpc, vel, t_scan_myr, dt_myr, direction, rec_every):
        p3 = pos.reshape(n_draws, m, 3) * 1000.0   # pc
        c = p3.mean(axis=1, keepdims=True)
        rms = np.sqrt(((p3 - c) ** 2).sum(-1).mean(axis=1))
        better = rms < best_rms
        best_rms[better] = rms[better]
        best_t[better] = t
    ok = np.isfinite(best_rms)
    q16, q50, q84 = np.percentile(best_rms[ok], [16, 50, 84])
    return {
        "n_draws": int(n_draws),
        "rms_min_pc_p16": float(q16), "rms_min_pc_p50": float(q50),
        "rms_min_pc_p84": float(q84),
        "t_min_myr_p16": float(np.percentile(best_t[ok], 16)),
        "t_min_myr_p50": float(np.percentile(best_t[ok], 50)),
        "t_min_myr_p84": float(np.percentile(best_t[ok], 84)),
        "p_rms_lt_2pc": float(np.mean(best_rms[ok] < 2.0)),
        "p_rms_lt_5pc": float(np.mean(best_rms[ok] < 5.0)),
    }


def vet_candidate(cand: dict, table: pd.DataFrame, direction: int,
                  dt_myr: float = 0.25, n_draws: int = 400,
                  astro_floor_kms: float = 0.3) -> dict:
    """Attach chemistry + rendezvous-MC verdicts to one detector candidate."""
    members = table.iloc[cand["members"]].reset_index(drop=True)
    mh = pd.to_numeric(members.get("mh_gspphot"), errors="coerce").to_numpy(float) \
        if "mh_gspphot" in members.columns else np.full(len(members), np.nan)
    out = dict(cand)
    out["chemistry"] = chemistry_vet(mh)
    t_scan = max(abs(cand["t_myr"]) * 1.5, 2.0)
    out["rendezvous_mc"] = rendezvous_mc(members, direction, t_scan,
                                         dt_myr=dt_myr, n_draws=n_draws,
                                         astro_floor_kms=astro_floor_kms)
    out["member_source_ids"] = [int(s) for s in
                                pd.to_numeric(members["source_id"],
                                              errors="coerce").fillna(-1)] \
        if "source_id" in members.columns else []
    return out


__all__ = ["chemistry_vet", "rendezvous_mc", "vet_candidate"]

"""Chance calibration: velocity-shuffled mock catalogues.

The null hypothesis is "positions and velocities are uncorrelated beyond the
smooth local phase-space structure".  Each mock keeps every star exactly where
it is and permutes the velocity *vectors* (with their attached uncertainties)
among stars within the same spatial cell.  That preserves the one-point spatial
density and the local velocity-ellipsoid — including its gradients on scales
above the cell size — while destroying the specific position–velocity phasing
any real convergence relies on.  Running the *identical* detector on each mock
absorbs the full look-elsewhere effect (all epochs, all ball placements, all
dedupe choices) into one number: the null distribution of the best surprise.

The second, assumption-free control is not in this module: the backward time
direction of the real data, scanned with the same detector, is a matched
astrophysical null for the forward scan (phase-mixed dynamics is statistically
time-symmetric; deliberate future assembly is not).
"""

from __future__ import annotations

import numpy as np

from .convergence import ConvergenceParams, detect_convergences


def shuffle_velocities(pos_pc: np.ndarray, vel: np.ndarray, sigv: np.ndarray,
                       cell_pc: float, rng: np.random.Generator):
    """Permute (velocity, sigma_v) jointly among stars sharing a spatial cell."""
    keys = np.floor(np.asarray(pos_pc, float) / cell_pc).astype(np.int64)
    # Stable 1-D cell id.
    kmin = keys.min(axis=0)
    span = keys.max(axis=0) - kmin + 1
    cid = ((keys[:, 0] - kmin[0]) * span[1] + (keys[:, 1] - kmin[1])) * span[2] \
        + (keys[:, 2] - kmin[2])
    order = np.argsort(cid, kind="stable")
    vel_s = np.asarray(vel, float).copy()
    sig_s = np.asarray(sigv, float).copy()
    cid_sorted = cid[order]
    starts = np.flatnonzero(np.r_[True, cid_sorted[1:] != cid_sorted[:-1]])
    ends = np.r_[starts[1:], len(cid_sorted)]
    for s, e in zip(starts, ends, strict=True):
        idx = order[s:e]
        if len(idx) < 2:
            continue
        perm = rng.permutation(len(idx))
        vel_s[idx] = vel[idx[perm]]
        sig_s[idx] = sigv[idx[perm]]
    return vel_s, sig_s


def run_mocks(pos0_kpc: np.ndarray, vel0_kpcmyr: np.ndarray, sigv_kms: np.ndarray,
              direction: int, params: ConvergenceParams, n_mocks: int,
              cell_pc: float = 40.0, seed: int = 20260725) -> dict:
    """Detector null distribution from ``n_mocks`` velocity-shuffled catalogues."""
    pos_pc = np.asarray(pos0_kpc, float) * 1000.0
    per_mock = []
    for k in range(n_mocks):
        rng = np.random.default_rng(seed + 1000 * k + (0 if direction > 0 else 1))
        vel_s, sig_s = shuffle_velocities(pos_pc, vel0_kpcmyr, sigv_kms,
                                          cell_pc, rng)
        res = detect_convergences(pos0_kpc, vel_s, sig_s, direction, params,
                                  rng=np.random.default_rng(seed + k))
        best = max((c["surprise"] for c in res["candidates"]), default=0.0)
        per_mock.append({"n_candidates": len(res["candidates"]),
                         "max_surprise": float(best),
                         "n_raw": res["n_raw_detections"],
                         "t_horizon_myr": res["t_horizon_myr"]})
        print(f"[herdsman] mock {k + 1}/{n_mocks} (dir {direction:+d}): "
              f"{per_mock[-1]['n_candidates']} candidates, "
              f"max surprise {per_mock[-1]['max_surprise']:.2f}")
    maxes = [m["max_surprise"] for m in per_mock]
    return {"n_mocks": n_mocks, "cell_pc": cell_pc, "per_mock": per_mock,
            "max_surprise_dist": maxes,
            "mean_candidates": float(np.mean([m["n_candidates"] for m in per_mock]))
            if per_mock else 0.0}


def global_p_value(observed_best: float, mock_result: dict) -> float:
    """P(any mock produces a candidate at least as surprising as the best real)."""
    maxes = mock_result.get("max_surprise_dist", [])
    if not maxes:
        return float("nan")
    n_ge = sum(1 for m in maxes if m >= observed_best)
    # Add-one (conservative) so a zero count never claims p = 0.
    return float((n_ge + 1) / (len(maxes) + 1))


__all__ = ["shuffle_velocities", "run_mocks", "global_p_value"]

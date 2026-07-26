"""Core detector: N-star orbital convergences in a propagated 6D sample.

The statistic
-------------
Integrate every star's orbit through the Galactic potential and, at each
recorded epoch ``t``, look for sets of ``>= n_min`` stars that occupy a common
ball whose radius matches the *propagated measurement uncertainty*,

    R(t) = r0 + kappa * sigma_v_ref * |t|          (pc; sigma_v in pc/Myr)

A true point-rendezvous appears as a blob of exactly this size, because each
member's present-day velocity error ``sigma_v`` displaces its propagated
position by ``sigma_v * t``.  Each such set is scored by its Poisson surprise,

    S = -log10 P( X >= m | lambda ),   lambda = n_local * (4/3) pi R(t)^3,

i.e. how unlikely ``m`` stars are to share the ball by chance given the local
propagated number density ``n_local``.  Sets must also be *dispersed today*
(median pairwise separation > ``r_now_min``) and *focusing* (rms radius shrinks
by ``>= focus_min``), so present-day clusters, moving groups, and wide binaries
never enter; the detector only fires on stars that are far apart now and
co-located later (forward direction) or earlier (backward direction).

The self-limiting horizon
-------------------------
Because R(t) grows linearly while chance occupancy grows as R^3, there is an
epoch beyond which a ball of radius R(t) is expected to contain field stars by
chance (lambda ~ 1) and the statistic loses meaning.  The scan therefore stops
when the *typical* lambda reaches ``lambda_cap``; the reached horizon is a
measured property of the sample (its density and velocity precision), reported
as ``t_horizon_myr`` rather than assumed.  Sensitivity claims must quote it.

Look-elsewhere is handled globally: the identical detector runs on
velocity-shuffled mock catalogues (``mocks.py``) and, as a matched astrophysical
control, on the time-reversed direction of the real data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import poisson

from ..galactic.orbits import KMS_TO_KPCMYR, acceleration

PC_PER_KPC = 1000.0
# 1 km/s of velocity error displaces a propagated position by this many pc/Myr.
KMS_AS_PC_PER_MYR = KMS_TO_KPCMYR * PC_PER_KPC


@dataclass
class ConvergenceParams:
    """Tunables for the convergence detector (defaults are the search defaults)."""

    t_max_myr: float = 20.0      # scan horizon per direction (may stop earlier)
    dt_myr: float = 0.25         # leapfrog step
    rec_every: int = 2           # detection every rec_every steps (0.5 Myr)
    r0_pc: float = 1.0           # meeting radius at t=0 (floor)
    kappa: float = 1.0           # error-growth factor in R(t)
    lambda_cap: float = 0.5      # stop scanning when typical ball occupancy hits this
    n_min: int = 4               # minimum stars meeting simultaneously
    r_now_min_pc: float = 20.0   # median present-day pairwise separation must exceed
    now_collapse_pc: float = 1.0 # members closer than this today count as one unit
    focus_min: float = 3.0       # rms_now / rms_meet must exceed
    surprise_min: float = 3.0    # -log10 Poisson tail to record a candidate
    density_r_pc: float = 25.0   # radius of the local-density estimate
    density_sample: int = 2000   # stars sampled for the typical-density estimate
    jaccard_dedupe: float = 0.5  # member overlap that merges epoch duplicates


class _UnionFind:
    """Minimal union-find over integer labels."""

    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        p = self.parent
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def _components_from_pairs(pairs: np.ndarray, size_min: int) -> list[np.ndarray]:
    """Connected components with >= ``size_min`` members of the pair graph.

    Works on the compacted node set actually touched by pairs, with scipy's
    C-level connected_components, so it stays fast at mock-catalogue scale.
    """
    if len(pairs) == 0:
        return []
    nodes = np.unique(pairs.ravel())
    idx = np.searchsorted(nodes, pairs)
    m = len(nodes)
    adj = coo_matrix((np.ones(len(idx)), (idx[:, 0], idx[:, 1])), shape=(m, m))
    _, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    out = []
    for lab in np.where(sizes >= size_min)[0]:
        out.append(nodes[labels == lab])
    return out


def propagate(pos0_kpc: np.ndarray, vel0_kpcmyr: np.ndarray, t_max_myr: float,
              dt_myr: float, direction: int, rec_every: int):
    """Yield ``(t_signed_myr, pos_kpc)`` at recorded epochs (t=0 not yielded).

    Symplectic leapfrog identical to ``galactic.orbits.integrate_orbits`` but
    streamed, so the full trajectory tensor is never held in memory.
    """
    pos = np.asarray(pos0_kpc, float).copy()
    vel = np.asarray(vel0_kpcmyr, float).copy() * direction
    n_steps = int(np.ceil(abs(t_max_myr) / dt_myr))
    acc = acceleration(pos)
    for i in range(1, n_steps + 1):
        vel += 0.5 * dt_myr * acc
        pos += dt_myr * vel
        acc = acceleration(pos)
        vel += 0.5 * dt_myr * acc
        if i % rec_every == 0 or i == n_steps:
            yield direction * i * dt_myr, pos


def _effective_units(pos_now_pc: np.ndarray, members: np.ndarray,
                     collapse_pc: float) -> int:
    """Count members after collapsing sub-pc (binary/co-moving) subgroups."""
    sub = pos_now_pc[members]
    tree = cKDTree(sub)
    pairs = tree.query_pairs(collapse_pc, output_type="ndarray")
    if len(pairs) == 0:
        return len(members)
    uf = _UnionFind(len(members))
    for i, j in pairs:
        uf.union(int(i), int(j))
    return len({uf.find(i) for i in range(len(members))})


def _median_pairwise(x: np.ndarray) -> float:
    d = x[:, None, :] - x[None, :, :]
    dd = np.sqrt((d * d).sum(-1))
    iu = np.triu_indices(len(x), k=1)
    return float(np.median(dd[iu]))


def _rms_radius(x: np.ndarray) -> float:
    c = x.mean(0)
    return float(np.sqrt(((x - c) ** 2).sum(-1).mean()))


def detect_convergences(pos0_kpc: np.ndarray, vel0_kpcmyr: np.ndarray,
                        sigv_kms: np.ndarray, direction: int,
                        params: ConvergenceParams | None = None,
                        rng: np.random.Generator | None = None) -> dict:
    """Scan one time direction for N-star convergences.

    ``pos0_kpc``/``vel0_kpcmyr`` are Galactocentric; ``sigv_kms`` is each star's
    scalar velocity uncertainty (measurement + astrophysical floor).
    ``direction`` is +1 (future) or -1 (past).  Returns a dict with the
    candidate list (member *indices* into the input arrays), the reached
    horizon, and scan metadata.  Deterministic given ``rng`` (used only for the
    density subsample).
    """
    p = params or ConvergenceParams()
    rng = rng or np.random.default_rng(0)
    n = len(pos0_kpc)
    pos_now_pc = np.asarray(pos0_kpc, float) * PC_PER_KPC
    sigv_ref_pcmyr = float(np.median(sigv_kms)) * KMS_AS_PC_PER_MYR

    raw: list[dict] = []
    t_horizon = None
    epochs_scanned = 0
    dens_idx = rng.choice(n, size=min(p.density_sample, n), replace=False)
    vol_dens_kpc3 = (4.0 / 3.0) * np.pi * (p.density_r_pc / PC_PER_KPC) ** 3

    for t, pos in propagate(pos0_kpc, vel0_kpcmyr, p.t_max_myr, p.dt_myr,
                            direction, p.rec_every):
        r_t_pc = p.r0_pc + p.kappa * sigv_ref_pcmyr * abs(t)
        r_t_kpc = r_t_pc / PC_PER_KPC
        tree = cKDTree(pos)
        # Typical chance occupancy of an R(t) ball; stop when it saturates.
        cnt = tree.query_ball_point(pos[dens_idx], r=p.density_r_pc / PC_PER_KPC,
                                    return_length=True)
        n_typ_kpc3 = float(np.median(cnt)) / vol_dens_kpc3
        lam_typ = n_typ_kpc3 * (4.0 / 3.0) * np.pi * r_t_kpc ** 3
        if lam_typ > p.lambda_cap:
            t_horizon = abs(t)
            break
        epochs_scanned += 1

        pairs = tree.query_pairs(r_t_kpc, output_type="ndarray")
        if len(pairs) == 0:
            continue
        for members in _components_from_pairs(pairs, p.n_min):
            m_eff = _effective_units(pos_now_pc, members, p.now_collapse_pc)
            if m_eff < p.n_min:
                continue
            med_now = _median_pairwise(pos_now_pc[members])
            if med_now < p.r_now_min_pc:
                continue
            rms_now = _rms_radius(pos_now_pc[members])
            rms_t = _rms_radius(pos[members] * PC_PER_KPC)
            if rms_t <= 0 or rms_now / max(rms_t, 1e-6) < p.focus_min:
                continue
            # Ball-likeness: linking can chain filaments whose extent exceeds
            # the ball the Poisson score assumes; reject non-compact shapes.
            if rms_t > 1.5 * r_t_pc:
                continue
            centroid = pos[members].mean(0)
            n_loc = tree.query_ball_point(centroid, r=p.density_r_pc / PC_PER_KPC,
                                          return_length=True) / vol_dens_kpc3
            lam = max(n_loc, n_typ_kpc3, 1e-12) * (4.0 / 3.0) * np.pi * r_t_kpc ** 3
            surprise = float(-poisson.logsf(len(members) - 1, lam) / np.log(10.0))
            if not np.isfinite(surprise) or surprise < p.surprise_min:
                continue
            # Internal velocity spread at the meeting (arrival coherence).
            v = vel0_kpcmyr[members] / KMS_TO_KPCMYR   # km/s, conserved shape
            sig_int = float(np.sqrt(((v - v.mean(0)) ** 2).sum(-1).mean()))
            raw.append({
                "t_myr": float(t), "members": members, "m": int(len(members)),
                "m_eff": int(m_eff), "r_ball_pc": float(r_t_pc),
                "lambda": float(lam), "surprise": surprise,
                "rms_now_pc": rms_now, "rms_meet_pc": rms_t,
                "focus": float(rms_now / max(rms_t, 1e-6)),
                "med_now_pc": med_now, "sig_v_internal_kms": sig_int,
                "centroid_gc_kpc": [float(c) for c in centroid],
            })

    if t_horizon is None:
        t_horizon = p.t_max_myr
    candidates = _dedupe(raw, p.jaccard_dedupe)
    return {
        "direction": int(direction),
        "n_stars": int(n),
        "sigv_ref_kms": float(np.median(sigv_kms)),
        "t_horizon_myr": float(t_horizon),
        "epochs_scanned": int(epochs_scanned),
        "n_raw_detections": len(raw),
        "candidates": candidates,
        "params": asdict(p),
    }


def _dedupe(raw: list[dict], jaccard_min: float) -> list[dict]:
    """Merge epoch-duplicates of the same physical set; keep the best epoch.

    Two detections merge when their member sets' Jaccard overlap exceeds
    ``jaccard_min``.  The survivor carries the maximum surprise, the number of
    epochs on which the set was seen (persistence — long dwell favours a true
    co-moving rendezvous over a chance crossing), and the epoch span.
    """
    groups: list[dict] = []
    for det in sorted(raw, key=lambda d: -d["surprise"]):
        s = set(det["members"].tolist())
        for g in groups:
            inter = len(s & g["member_set"])
            union = len(s | g["member_set"])
            if union and inter / union >= jaccard_min:
                g["n_epochs_seen"] += 1
                g["t_first_myr"] = min(g["t_first_myr"], det["t_myr"], key=abs)
                g["t_last_myr"] = max(g["t_last_myr"], det["t_myr"], key=abs)
                g["member_set"] |= s
                break
        else:
            best = dict(det)
            best["member_set"] = s
            best["n_epochs_seen"] = 1
            best["t_first_myr"] = det["t_myr"]
            best["t_last_myr"] = det["t_myr"]
            groups.append(best)
    out = []
    for g in sorted(groups, key=lambda d: -d["surprise"]):
        g = dict(g)
        g["members"] = sorted(int(i) for i in g.pop("member_set"))
        out.append(g)
    return out


__all__ = ["ConvergenceParams", "detect_convergences", "propagate",
           "KMS_AS_PC_PER_MYR", "PC_PER_KPC"]

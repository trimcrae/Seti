"""The ANTIPHASE null and the injection efficiency.  Pure.

**Null.**  A coupled event is a statement about *one* star's optical and IR
series moving against each other.  If the detector's COUPLED rate among real
stars is no higher than among pairs whose optical and IR cannot be physically
linked, the detections are chance alignments of independent noise and
systematics (a ZTF season offset meeting a NEOWISE visit that scattered
bright).  Two nulls are built from the same stars:

* **pair** --- star i's IR series against star j's optical series (j != i,
  random), j's bins aligned to i's NEOWISE epochs by nearest time (<= 90 d);
* **shift** --- star i's own optical series moved by +-``shifts`` epochs
  against its own IR (a year or more of lag destroys a simultaneous coupling
  but keeps each series' own structure).

Each null round yields a COUPLED count over the same number of stars as the
real sample, and a pooled distribution of the coupling ``score`` (the minimum
of the four coupling sigmas); a real candidate's FAP is the fraction of null
scores at or above its own, and its expected false count is FAP times the
number of stars tested.

**Injection.**  Into real stars' binned series: a grey fade of depth ``f``
over ``dur`` consecutive matched epochs, and in the same epochs the IR excess a
single-temperature re-radiator of luminosity ``R * f * F_bol`` adds to W1 and
W2 (``R = 1`` is exact energy balance).  Recovered = the full ladder calls it
COUPLED, GREY and BALANCED.
"""

from __future__ import annotations

import numpy as np

from .coupling import assess_coupling
from .energy import SIGMA_SB, WISE_BANDS, C, f_bol, planck_nu, star_teff, wise_fnu_jy


def align_to(t_target, t_src, arrs: dict, tol_yr: float = 90.0 / 365.25) -> dict:
    """Re-index ``arrs`` ({band: (mag, err)} on ``t_src``) onto ``t_target`` by nearest time."""
    tt = np.asarray(t_target, float)
    ts = np.asarray(t_src, float)
    out = {}
    if ts.size == 0:
        return {b: (np.full(tt.size, np.nan), np.full(tt.size, np.nan)) for b in arrs}
    order = np.argsort(ts)
    ts_sorted = ts[order]
    pos = np.clip(np.searchsorted(ts_sorted, tt), 1, max(ts_sorted.size - 1, 1))
    left = ts_sorted[np.maximum(pos - 1, 0)]
    right = ts_sorted[np.minimum(pos, ts_sorted.size - 1)]
    pick = np.where(np.abs(tt - left) <= np.abs(right - tt), pos - 1, pos)
    pick = np.clip(pick, 0, ts_sorted.size - 1)
    near = np.abs(ts_sorted[pick] - tt) <= tol_yr
    src_idx = order[pick]
    for b, (m, e) in arrs.items():
        m, e = np.asarray(m, float), np.asarray(e, float)
        mm = np.where(near, m[src_idx], np.nan)
        ee = np.where(near, e[src_idx], np.nan)
        out[b] = (mm, ee)
    return out


def shift_arrays(arrs: dict, s: int) -> dict:
    out = {}
    for b, (m, e) in arrs.items():
        m, e = np.asarray(m, float), np.asarray(e, float)
        mm = np.full(m.size, np.nan)
        ee = np.full(e.size, np.nan)
        if s > 0:
            mm[s:], ee[s:] = m[:-s], e[:-s]
        elif s < 0:
            mm[:s], ee[:s] = m[-s:], e[-s:]
        else:
            mm, ee = m.copy(), e.copy()
        out[b] = (mm, ee)
    return out


def run_null(packs: dict, conf: dict | None = None, *, n_rounds: int = 10, shifts=(-4, -2, 2, 4),
             seed: int = 20260924, max_stars: int | None = None) -> dict:
    """Pair and shift nulls over ``packs`` ({sid: {"t", "opt", "ir"}})."""
    rng = np.random.default_rng(seed)
    sids = [s for s in packs if packs[s].get("opt") and packs[s].get("ir")]
    if max_stars and len(sids) > max_stars:
        sids = list(rng.choice(sids, size=int(max_stars), replace=False))
    n = len(sids)
    out = {"n_stars": n, "n_rounds_pair": 0, "pair_coupled_per_round": [],
           "shift_coupled": {}, "pair_scores": [], "shift_scores": []}
    if n < 3:
        out["status"] = "TOO_FEW_STARS"
        return out
    for _ in range(int(n_rounds)):
        perm = rng.permutation(n)
        # a derangement-ish: any fixed point is swapped with its neighbour
        for k in range(n):
            if perm[k] == k:
                j = (k + 1) % n
                perm[k], perm[j] = perm[j], perm[k]
        nc = 0
        for k in range(n):
            pi, pj = packs[sids[k]], packs[sids[perm[k]]]
            opt = align_to(pi["t"], pj["t"], pj["opt"])
            r = assess_coupling(pi["t"], opt, pi["ir"], conf)
            if r.is_coupled:
                nc += 1
            if np.isfinite(r.score) and r.n_faded:
                out["pair_scores"].append(round(float(r.score), 3))
        out["pair_coupled_per_round"].append(nc)
        out["n_rounds_pair"] += 1
    for s in shifts:
        nc = 0
        for sid in sids:
            p = packs[sid]
            r = assess_coupling(p["t"], shift_arrays(p["opt"], int(s)), p["ir"], conf)
            if r.is_coupled:
                nc += 1
            if np.isfinite(r.score) and r.n_faded:
                out["shift_scores"].append(round(float(r.score), 3))
        out["shift_coupled"][str(s)] = nc
    pc = np.asarray(out["pair_coupled_per_round"], float)
    out["pair_coupled_mean"] = float(pc.mean()) if pc.size else float("nan")
    sc = np.asarray(list(out["shift_coupled"].values()), float)
    out["shift_coupled_mean"] = float(sc.mean()) if sc.size else float("nan")
    out["status"] = "OK"
    return out


def score_fap(score: float, null_scores, n_null_trials: int) -> float:
    """Fraction of null *trials* (star-pairs tested) with score >= ``score``.

    ``null_scores`` holds only trials that produced a faded set; the
    denominator is every trial, so a trial that never faded counts as a
    non-exceedance, which is what it is.
    """
    ns = np.asarray(null_scores, float)
    if not np.isfinite(score) or n_null_trials <= 0:
        return float("nan")
    k = int(np.sum(ns >= float(score)))
    return float((k + 1) / (n_null_trials + 1))


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------
def ir_excess_dmag(meta: dict, frac: float, t_k: float, ratio: float = 1.0) -> dict:
    """W1/W2 magnitude change from re-radiating ``ratio * frac * F_bol`` at ``t_k``."""
    teff, _ = star_teff(meta.get("teff_gspphot"), meta.get("bp_rp"))
    fb = f_bol(float(meta.get("phot_g_mean_mag", np.nan)), teff)
    out = {}
    if not np.isfinite(fb):
        return {"W1": float("nan"), "W2": float("nan")}
    omega = ratio * frac * fb * np.pi / (SIGMA_SB * t_k ** 4)
    for b in ("W1", "W2"):
        base = meta.get(f"{b.lower()}mpro")
        if base is None or not np.isfinite(float(base)):
            out[b] = float("nan")
            continue
        nu = C / (WISE_BANDS[b]["lam_um"] * 1e-6)
        ex_jy = omega * float(planck_nu(nu, t_k)) / 1e-26
        out[b] = float(-2.5 * np.log10(1.0 + ex_jy / wise_fnu_jy(float(base), b)))
    return out


def inject(pack: dict, frac: float, t_k: float, start: int, dur: int = 2,
           ratio: float = 1.0) -> dict:
    """A copy of ``pack`` with a grey fade + balanced IR rise over matched epochs."""
    t = np.asarray(pack["t"], float)
    have = np.isfinite(t)
    for v in list(pack["opt"].values()) + list(pack["ir"].values()):
        have &= np.isfinite(np.asarray(v[0], float))
    idx = np.nonzero(have)[0]
    sel = idx[start:start + dur]
    dm_opt = -2.5 * np.log10(1.0 - frac)
    dir_ = ir_excess_dmag(pack.get("meta", {}), frac, t_k, ratio)
    opt = {}
    for b, (m, e) in pack["opt"].items():
        mm = np.array(m, float, copy=True)
        mm[sel] += dm_opt
        opt[b] = (mm, np.asarray(e, float))
    ir = {}
    for b, (m, e) in pack["ir"].items():
        mm = np.array(m, float, copy=True)
        if np.isfinite(dir_.get(b, np.nan)):
            mm[sel] += dir_[b]
        ir[b] = (mm, np.asarray(e, float))
    return {**pack, "opt": opt, "ir": ir, "injected_epochs": [float(x) for x in t[sel]],
            "injected_ir_dmag": dir_}


def injection_efficiency(packs: dict, evaluate, *, depths=(0.01, 0.02, 0.05, 0.1),
                         temps=(300.0, 600.0, 1000.0), dur: int = 2, max_stars: int = 200,
                         seed: int = 7) -> dict:
    """Recovery fraction on real series.  ``evaluate(pack) -> dict`` must return
    ``{"coupled": bool, "grey": bool, "balanced": bool}`` for one pack; it is
    the channel's own full ladder, passed in so this module stays pure."""
    rng = np.random.default_rng(seed)
    sids = [s for s in packs if packs[s].get("opt") and packs[s].get("ir")]
    if len(sids) > max_stars:
        sids = list(rng.choice(sids, size=int(max_stars), replace=False))
    grid = []
    for f in depths:
        for tk in temps:
            n = rec_c = rec_all = 0
            for sid in sids:
                p = packs[sid]
                t = np.asarray(p["t"], float)
                have = np.isfinite(t)
                for v in list(p["opt"].values()) + list(p["ir"].values()):
                    have &= np.isfinite(np.asarray(v[0], float))
                k = int(have.sum())
                if k < dur + 4:
                    continue
                start = int(rng.integers(0, k - dur + 1))
                q = inject(p, float(f), float(tk), start, dur)
                r = evaluate(q)
                n += 1
                rec_c += int(bool(r.get("coupled")))
                rec_all += int(bool(r.get("coupled")) and bool(r.get("grey"))
                               and bool(r.get("balanced")))
            grid.append({"depth": float(f), "t_k": float(tk), "n": n,
                         "frac_coupled": round(rec_c / n, 4) if n else None,
                         "frac_full_ladder": round(rec_all / n, 4) if n else None})
    return {"n_stars": len(sids), "dur_epochs": dur, "grid": grid}


__all__ = ["align_to", "inject", "injection_efficiency", "ir_excess_dmag", "run_null",
           "score_fap", "shift_arrays"]

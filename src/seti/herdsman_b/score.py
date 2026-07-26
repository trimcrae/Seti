"""Per-cluster ancestry scoring: co-natal vs field-sampled.

Statistics are deliberately *self-calibrating*: GSP-Phot metallicities carry
Teff/extinction-correlated systematics that inflate raw spreads, so no
absolute Bayes factor is trusted. Instead each cluster's excess spread is

1. computed on **Teff-detrended** residuals (a systematic trend across the
   CMD is removed; a genuine field-sampled spread is Teff-independent and
   survives detrending),
2. **interloper-trimmed** (the membership probabilities predict how many
   field contaminants to expect; that many + 2 most-deviant members are
   removed before the spread is trusted — a real assembly's spread is a
   property of the core, not of a few outliers),
3. ranked as a **z-score against the census itself** (clusters of comparable
   member count define what "normal" spread — including shared systematics —
   looks like; an assembly must be an outlier against its own peers),
4. required to be **unimodal** (a two-population gap points to the natural
   heterogeneous channel — stripped nuclei / cluster mergers — or to a
   catalog chance overlap, all interesting but not the target), and
5. compared with the **local field spread** at the cluster's Galactocentric
   radius (an assembly of gathered local stars should *mirror* the field,
   not merely exceed co-natal homogeneity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..panspermia.kinematics import _A_ICRS_TO_GAL

R0_KPC = 8.178

QUALITY = {"prob_min": 0.7, "teff_lo": 4000.0, "teff_hi": 7500.0,
           "mh_sigma_max": 0.4, "n_min": 8}
CANDIDATE = {"z_min": 4.0, "x_trim_min": 2.0, "field_likeness_min": 0.5,
             "n_trim_min": 8}
CONATAL_FLOOR_DEX = 0.03


def _mad_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def radec_to_rgal(ra_deg, dec_deg, dist_kpc) -> np.ndarray:
    """Galactocentric cylindrical radius (kpc) from sky position + distance."""
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    d = np.asarray(dist_kpc, float)
    r_icrs = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                       np.sin(dec)], axis=0) * d
    r_gal = _A_ICRS_TO_GAL @ r_icrs          # heliocentric: X toward GC
    x = r_gal[0] - R0_KPC                    # galactocentric (GC at 0)
    return np.hypot(x, r_gal[1])


def detrend_teff(mh: np.ndarray, teff: np.ndarray) -> np.ndarray:
    """Remove a robust linear mh(Teff) trend (one clip pass); returns residuals.

    Kills the dominant GSP-Phot systematic within a cluster while barely
    touching a genuine field-sampled spread (which is uncorrelated with Teff).
    """
    ok = np.isfinite(mh) & np.isfinite(teff)
    if ok.sum() < 10 or np.ptp(teff[ok]) < 300.0:
        return mh - np.nanmedian(mh)
    c = np.polyfit(teff[ok], mh[ok], 1)
    r = mh - np.polyval(c, teff)
    s = _mad_std(r[ok])
    keep = ok & (np.abs(r - np.nanmedian(r[ok])) < 3 * max(s, 1e-3))
    if keep.sum() >= 10:
        c = np.polyfit(teff[keep], mh[keep], 1)
        r = mh - np.polyval(c, teff)
    return r - np.nanmedian(r[ok])


def build_field_spread(field: pd.DataFrame, r_lo: float = 3.0,
                       r_hi: float = 15.0, width: float = 1.0):
    """Robust field [M/H] spread vs Galactocentric radius, from the field pull."""
    edges = np.arange(r_lo, r_hi + width, width)
    spreads, counts = [], []
    r = field["r_gal"].to_numpy(float)
    mh = field["mh"].to_numpy(float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi) & np.isfinite(mh)
        spreads.append(_mad_std(mh[sel]) if sel.sum() >= 200 else np.nan)
        counts.append(int(sel.sum()))
    return edges, np.array(spreads), np.array(counts)


def _field_spread_at(edges, spreads, r_gal: float) -> float:
    if not np.isfinite(r_gal):
        return float(np.nanmedian(spreads))
    i = int(np.clip(np.searchsorted(edges, r_gal) - 1, 0, len(spreads) - 1))
    s = spreads[i]
    return float(s) if np.isfinite(s) else float(np.nanmedian(spreads))


def score_cluster(g: pd.DataFrame) -> dict | None:
    """Spread statistics for one cluster's member table (post-quality-cut)."""
    q = QUALITY
    sel = ((g["prob"].to_numpy(float) >= q["prob_min"])
           & np.isfinite(g["mh"].to_numpy(float))
           & (g["teff"].to_numpy(float) >= q["teff_lo"])
           & (g["teff"].to_numpy(float) <= q["teff_hi"])
           & (g["mh_sigma"].to_numpy(float) < q["mh_sigma_max"]))
    gg = g[sel]
    if len(gg) < q["n_min"]:
        return None
    mh = gg["mh"].to_numpy(float)
    teff = gg["teff"].to_numpy(float)
    prob = gg["prob"].to_numpy(float)
    sig = gg["mh_sigma"].to_numpy(float)

    resid = detrend_teff(mh, teff)
    s_raw = _mad_std(mh)
    s_det = _mad_std(resid)
    e_c = float(np.sqrt(np.median(sig) ** 2 + CONATAL_FLOOR_DEX ** 2))

    # Interloper trim: expected contaminants from the membership probabilities.
    k = int(np.ceil(np.sum(1.0 - prob))) + 2
    order = np.argsort(-np.abs(resid))
    kept = np.sort(order[k:]) if k < len(resid) else np.array([], dtype=int)
    if len(kept) < q["n_min"]:
        return None
    s_trim = _mad_std(resid[kept])

    # Two-population test: find the best split (each side holding >=20% of
    # members) and compare the gap there with the *within-mode* spread — the
    # full-sample MAD is useless here because a symmetric bimodal has
    # MAD ~ separation.
    srt = np.sort(resid[np.isfinite(resid)])
    gap, two_pop = 0.0, False
    if len(srt) >= 12:
        j_lo = max(3, int(np.ceil(0.2 * len(srt))))
        j_hi = min(len(srt) - 3, int(np.floor(0.8 * len(srt))))
        if j_hi > j_lo:
            diffs = np.diff(srt)
            j = j_lo + int(np.argmax(diffs[j_lo - 1: j_hi]))
            gap = float(diffs[j - 1])
            s_within = max(_mad_std(srt[:j]), _mad_std(srt[j:]), 0.05)
            two_pop = bool(gap > 4.0 * s_within)

    dist_kpc = float(np.nanmedian(1.0 / gg["parallax"].to_numpy(float)))
    r_gal = float(radec_to_rgal(np.nanmedian(gg["ra"]), np.nanmedian(gg["dec"]),
                                dist_kpc))
    return {"n_members": int(len(g)), "n_used": int(len(gg)),
            "n_trim": int(len(kept)), "k_trimmed": int(k),
            "s_raw": s_raw, "s_det": s_det, "s_trim": s_trim,
            "err_floor": e_c, "x_trim": float(s_trim / e_c) if e_c > 0 else np.nan,
            "two_pop": two_pop, "gap": gap,
            "dist_kpc": dist_kpc, "r_gal_kpc": r_gal,
            "mh_median": float(np.median(mh))}


def score_census(members: pd.DataFrame, field: pd.DataFrame) -> pd.DataFrame:
    """Score every cluster; attach census-relative z and field-likeness.

    ``members`` columns: cluster, source_id, prob, mh, mh_sigma, teff, gmag,
    ra, dec, parallax.  ``field`` columns: mh, r_gal.
    """
    edges, spreads, _ = build_field_spread(field)
    rows = []
    for name, g in members.groupby("cluster"):
        st = score_cluster(g)
        if st is None:
            continue
        st["cluster"] = name
        st["s_field"] = _field_spread_at(edges, spreads, st["r_gal_kpc"])
        a, b = st["s_trim"], st["s_field"]
        st["field_likeness"] = float(min(a, b) / max(a, b)) \
            if a > 0 and b > 0 else np.nan
        rows.append(st)
    tab = pd.DataFrame(rows)
    if not len(tab):
        return tab

    # Census self-calibration: z of ln(x_trim) against comparable-N clusters.
    lx = np.log(tab["x_trim"].to_numpy(float))
    n = tab["n_used"].to_numpy(float)
    z = np.full(len(tab), np.nan)
    for i in range(len(tab)):
        peers = (n >= n[i] / 2) & (n <= n[i] * 2) & np.isfinite(lx)
        if peers.sum() < 20:
            peers = np.isfinite(lx)
        mu = np.median(lx[peers])
        sd = _mad_std(lx[peers])
        z[i] = (lx[i] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    tab["z_census"] = z

    c = CANDIDATE
    tab["assembly_candidate"] = ((tab["z_census"] >= c["z_min"])
                                 & (tab["x_trim"] >= c["x_trim_min"])
                                 & (~tab["two_pop"])
                                 & (tab["field_likeness"] >= c["field_likeness_min"])
                                 & (tab["n_trim"] >= c["n_trim_min"]))
    return tab.sort_values("z_census", ascending=False).reset_index(drop=True)


__all__ = ["score_census", "score_cluster", "build_field_spread",
           "detrend_teff", "radec_to_rgal", "QUALITY", "CANDIDATE"]

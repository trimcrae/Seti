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

v2 additions (each a failure mode the v1 run demonstrated on real data —
Hogg_4, corr(mh, G) = -0.68 at 4.2 kpc; a top-10 of globulars and distant
reddened open clusters):

6. **magnitude-correlation kill switch** — |corr(mh_resid, G)| > 0.4 marks
   the spread as a GSP-Phot extinction/magnitude systematic
   (``mag_systematic``), which vetoes candidacy; the correlation is always
   reported,
7. **distance gate** — candidacy requires dist < 2.5 kpc (the GSP-Phot trust
   region); farther clusters are still scored but flagged
   ``beyond_phot_trust``,
8. **distance-matched field baseline** — the field spread is binned by both
   Galactocentric radius and heliocentric distance, so ``s_field`` comes
   from field stars whose GSP-Phot systematics match the cluster's (falls
   back to the R_gal-only bin when the matched bin has <200 stars), and
9. **globular-cluster heuristic** — ``gc_like`` (metal-poor and rich, or
   very distant) vetoes candidacy; GCs are ancient accreted systems where
   XP metallicities degrade most, not disk assemblies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..panspermia.kinematics import _A_ICRS_TO_GAL

R0_KPC = 8.178

QUALITY = {"prob_min": 0.7, "teff_lo": 4000.0, "teff_hi": 7500.0,
           "mh_sigma_max": 0.4, "n_min": 8}
CANDIDATE = {"z_min": 4.0, "x_trim_min": 2.0, "field_likeness_min": 0.5,
             "n_trim_min": 8}
VETO = {"mag_corr_max": 0.4,     # |corr(mh_resid, G)| above this = systematic
        "dist_trust_kpc": 2.5,   # GSP-Phot trust region (heliocentric)
        "gc_mh_max": -0.8,       # gc_like: metal-poor ...
        "gc_n_min": 100,         # ... and populous, or
        "gc_dist_kpc": 4.0}      # ... simply very distant
CONATAL_FLOOR_DEX = 0.03
FIELD_BIN_N_MIN = 200


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


@dataclass
class FieldSpread:
    """Field [M/H] spread lookup: R_gal bins x heliocentric-distance bins.

    The 2D (distance-matched) grid is the v2 default — each cluster's
    reference spread comes from field stars at comparable distance, so a
    distance-driven GSP-Phot systematic inflates cluster and reference
    alike instead of only the reference denominator. Bins with fewer than
    ``FIELD_BIN_N_MIN`` stars fall back to the R_gal-only marginal.
    """

    r_edges: np.ndarray
    d_edges: np.ndarray
    spreads_r: np.ndarray     # (n_r,)  R_gal-only marginal
    counts_r: np.ndarray
    spreads_2d: np.ndarray    # (n_r, n_d)  R_gal x distance
    counts_2d: np.ndarray


DIST_EDGES_KPC = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 8.0, 15.0)


def build_field_spread(field: pd.DataFrame, r_lo: float = 3.0,
                       r_hi: float = 15.0, width: float = 1.0,
                       d_edges=DIST_EDGES_KPC) -> FieldSpread:
    """Robust field [M/H] spread vs (R_gal, heliocentric distance).

    ``field`` needs columns ``mh`` and ``r_gal``; ``dist_kpc`` is optional —
    without it every 2D bin is empty and lookups use the R_gal marginal.
    """
    r_edges = np.arange(r_lo, r_hi + width, width)
    d_edges = np.asarray(d_edges, float)
    r = field["r_gal"].to_numpy(float)
    mh = field["mh"].to_numpy(float)
    d = (field["dist_kpc"].to_numpy(float) if "dist_kpc" in field.columns
         else np.full(len(field), np.nan))
    n_r, n_d = len(r_edges) - 1, len(d_edges) - 1
    spreads_r = np.full(n_r, np.nan)
    counts_r = np.zeros(n_r, dtype=int)
    spreads_2d = np.full((n_r, n_d), np.nan)
    counts_2d = np.zeros((n_r, n_d), dtype=int)
    ok = np.isfinite(mh)
    for i in range(n_r):
        sel_r = ok & (r >= r_edges[i]) & (r < r_edges[i + 1])
        counts_r[i] = int(sel_r.sum())
        if counts_r[i] >= FIELD_BIN_N_MIN:
            spreads_r[i] = _mad_std(mh[sel_r])
        for j in range(n_d):
            sel = sel_r & (d >= d_edges[j]) & (d < d_edges[j + 1])
            counts_2d[i, j] = int(sel.sum())
            if counts_2d[i, j] >= FIELD_BIN_N_MIN:
                spreads_2d[i, j] = _mad_std(mh[sel])
    return FieldSpread(r_edges, d_edges, spreads_r, counts_r,
                       spreads_2d, counts_2d)


def _field_spread_at(fs: FieldSpread, r_gal: float,
                     dist_kpc: float = np.nan) -> tuple[float, bool]:
    """Reference spread at (R_gal, distance); returns (spread, matched).

    ``matched`` is True when the distance-matched 2D bin supplied the value;
    False means the R_gal-only marginal (or the global median) was used.
    """
    if not np.isfinite(r_gal):
        return float(np.nanmedian(fs.spreads_r)), False
    i = int(np.clip(np.searchsorted(fs.r_edges, r_gal) - 1,
                    0, len(fs.spreads_r) - 1))
    if np.isfinite(dist_kpc):
        j = int(np.clip(np.searchsorted(fs.d_edges, dist_kpc) - 1,
                        0, fs.spreads_2d.shape[1] - 1))
        s2 = fs.spreads_2d[i, j]
        if np.isfinite(s2):
            return float(s2), True
    s = fs.spreads_r[i]
    return (float(s), False) if np.isfinite(s) \
        else (float(np.nanmedian(fs.spreads_r)), False)


def _safe_corr(a: np.ndarray, b: np.ndarray, n_min: int = 8) -> float:
    """Pearson correlation, NaN-safe; NaN when degenerate/underpopulated."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < n_min:
        return float("nan")
    aa, bb = a[ok], b[ok]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def two_pop_split(resid: np.ndarray) -> tuple[float, bool]:
    """Best >=20/80 split gap vs within-mode spread; (gap, two_pop) flag."""
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
    return gap, two_pop


def census_z(x: np.ndarray, n: np.ndarray, peer_min: int = 20) -> np.ndarray:
    """z of ln(x) against comparable-N peers (self-calibrating census rank)."""
    x = np.asarray(x, float)
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.where(x > 0, np.log(x), np.nan)
    z = np.full(len(lx), np.nan)
    for i in range(len(lx)):
        peers = (n >= n[i] / 2) & (n <= n[i] * 2) & np.isfinite(lx)
        if peers.sum() < peer_min:
            peers = np.isfinite(lx)
        mu = np.median(lx[peers])
        sd = _mad_std(lx[peers])
        z[i] = (lx[i] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return z


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
    gap, two_pop = two_pop_split(resid)

    # Magnitude-correlation kill switch (v2): a residual [M/H] trend with
    # apparent G is the fingerprint of the GSP-Phot extinction/magnitude
    # systematic that manufactured Hogg_4 in v1 — real chemistry does not
    # know how bright a star looks from Earth.
    corr_mh_gmag = _safe_corr(resid, gg["gmag"].to_numpy(float))
    mag_systematic = bool(np.isfinite(corr_mh_gmag)
                          and abs(corr_mh_gmag) > VETO["mag_corr_max"])

    dist_kpc = float(np.nanmedian(1.0 / gg["parallax"].to_numpy(float)))
    r_gal = float(radec_to_rgal(np.nanmedian(gg["ra"]), np.nanmedian(gg["dec"]),
                                dist_kpc))
    mh_median = float(np.median(mh))
    beyond_phot_trust = bool(np.isfinite(dist_kpc)
                             and dist_kpc >= VETO["dist_trust_kpc"]) \
        or not np.isfinite(dist_kpc)
    gc_like = bool((mh_median < VETO["gc_mh_max"]
                    and len(gg) >= VETO["gc_n_min"])
                   or (np.isfinite(dist_kpc)
                       and dist_kpc > VETO["gc_dist_kpc"]))
    return {"n_members": int(len(g)), "n_used": int(len(gg)),
            "n_trim": int(len(kept)), "k_trimmed": int(k),
            "s_raw": s_raw, "s_det": s_det, "s_trim": s_trim,
            "err_floor": e_c, "x_trim": float(s_trim / e_c) if e_c > 0 else np.nan,
            "two_pop": two_pop, "gap": gap,
            "corr_mh_gmag": corr_mh_gmag, "mag_systematic": mag_systematic,
            "dist_kpc": dist_kpc, "r_gal_kpc": r_gal,
            "beyond_phot_trust": beyond_phot_trust, "gc_like": gc_like,
            "mh_median": mh_median}


def score_census(members: pd.DataFrame, field: pd.DataFrame) -> pd.DataFrame:
    """Score every cluster; attach census-relative z and field-likeness.

    ``members`` columns: cluster, source_id, prob, mh, mh_sigma, teff, gmag,
    ra, dec, parallax.  ``field`` columns: mh, r_gal (+ optional dist_kpc for
    the distance-matched baseline).
    """
    fs = build_field_spread(field)
    rows = []
    for name, g in members.groupby("cluster"):
        st = score_cluster(g)
        if st is None:
            continue
        st["cluster"] = name
        st["s_field"], st["field_bin_matched"] = _field_spread_at(
            fs, st["r_gal_kpc"], st["dist_kpc"])
        a, b = st["s_trim"], st["s_field"]
        st["field_likeness"] = float(min(a, b) / max(a, b)) \
            if a > 0 and b > 0 else np.nan
        rows.append(st)
    tab = pd.DataFrame(rows)
    if not len(tab):
        return tab

    # Census self-calibration: z of ln(x_trim) against comparable-N clusters.
    tab["z_census"] = census_z(tab["x_trim"].to_numpy(float),
                               tab["n_used"].to_numpy(float))

    c = CANDIDATE
    tab["assembly_candidate"] = ((tab["z_census"] >= c["z_min"])
                                 & (tab["x_trim"] >= c["x_trim_min"])
                                 & (~tab["two_pop"])
                                 & (tab["field_likeness"] >= c["field_likeness_min"])
                                 & (tab["n_trim"] >= c["n_trim_min"])
                                 & (~tab["mag_systematic"])
                                 & (~tab["beyond_phot_trust"])
                                 & (~tab["gc_like"]))
    return tab.sort_values("z_census", ascending=False).reset_index(drop=True)


__all__ = ["score_census", "score_cluster", "build_field_spread",
           "FieldSpread", "census_z", "two_pop_split", "detrend_teff",
           "radec_to_rgal", "QUALITY", "CANDIDATE", "VETO",
           "CONATAL_FLOOR_DEX", "FIELD_BIN_N_MIN"]

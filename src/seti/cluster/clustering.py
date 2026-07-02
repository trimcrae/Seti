"""Matched-random-null clustering statistic + friends-of-friends group finder.

The central problem: an anomaly set will *look* clustered for boring reasons ---
anomalies are often preferentially bright, nearby, or in crowded sky regions, all
of which correlate with position.  Comparing to a *uniform* random field would
therefore fake a detection.  The fix is a **matched** null: random subsets of the
same parent catalogue drawn with the anomaly set's own distribution in the
confounders (magnitude, colour, sky cell).  Any residual clustering beyond that is
physical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score each column by the sample scatter so position (pc) and velocity
    (km/s) contribute on a common scale to Euclidean distances."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def _knn_stat(coords: np.ndarray, k: int = 1) -> float:
    """Median distance to the k-th nearest neighbour within a point set (smaller =
    more clustered).  Uses a KD-tree when SciPy is available, else a brute force
    fallback."""
    n = coords.shape[0]
    if n <= k:
        return float("nan")
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        d, _ = tree.query(coords, k=k + 1)   # first neighbour is self (0)
        return float(np.median(d[:, k]))
    except Exception:  # noqa: BLE001
        # Brute force: pairwise distances.
        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        kth = np.sort(dists, axis=1)[:, k - 1]
        return float(np.median(kth))


def _strata(parent: pd.DataFrame, feature_cols, n_feature_bins: int,
            sky_bins: int) -> np.ndarray:
    """Assign every parent row a stratum id from quantile bins of each feature
    plus a coarse RA/Dec grid, so matched draws preserve the anomaly set's
    magnitude/colour/sky-density distribution."""
    codes = np.zeros(len(parent), dtype=np.int64)
    mult = 1
    for c in feature_cols:
        v = pd.to_numeric(parent[c], errors="coerce")
        try:
            b = pd.qcut(v, q=n_feature_bins, labels=False, duplicates="drop")
        except Exception:  # noqa: BLE001
            b = pd.cut(v, bins=n_feature_bins, labels=False)
        b = pd.Series(b).fillna(-1).astype(int).to_numpy()
        codes = codes * (n_feature_bins + 1) + (b + 1)
        mult *= (n_feature_bins + 1)
    if sky_bins and {"ra", "dec"} <= set(parent.columns):
        ra = pd.to_numeric(parent["ra"], errors="coerce").to_numpy()
        dec = pd.to_numeric(parent["dec"], errors="coerce").to_numpy()
        ri = np.clip((ra / 360.0 * sky_bins).astype(int), 0, sky_bins - 1)
        di = np.clip(((dec + 90) / 180.0 * sky_bins).astype(int), 0, sky_bins - 1)
        codes = codes * (sky_bins * sky_bins) + (ri * sky_bins + di)
    return codes


def matched_null_clustering(parent: pd.DataFrame, anomaly_mask, space_cols,
                            feature_cols=("phot_g_mean_mag", "bp_rp", "parallax"),
                            n_null: int = 500, k: int = 1, n_feature_bins: int = 4,
                            sky_bins: int = 8, seed: int = 12345) -> dict:
    """Test whether the anomaly subset is more clustered in ``space_cols`` than a
    magnitude/colour/sky-matched random subset of the same parent.

    Returns ``S_obs`` (median k-NN distance of the anomalies), the null
    distribution summary, a left-tail ``p_value`` (small = over-clustered) and a
    ``z`` score.  ``space_cols`` are e.g. ['X_pc','Y_pc','Z_pc'] for spatial or
    those + velocity columns for full phase space.
    """
    parent = parent.reset_index(drop=True)
    mask = np.asarray(anomaly_mask, bool)
    space = _standardize(parent[list(space_cols)].to_numpy(float))
    good = np.all(np.isfinite(space), axis=1)
    mask = mask & good
    n_anom = int(mask.sum())
    if n_anom <= k:
        return {"n_anom": n_anom, "S_obs": float("nan"), "p_value": float("nan"),
                "z": float("nan"), "insufficient": True}

    s_obs = _knn_stat(space[mask], k=k)

    strata = _strata(parent, list(feature_cols), n_feature_bins, sky_bins)
    # Per-stratum indices among good, non-anomaly parent rows to draw the null from.
    pool = {}
    for st in np.unique(strata[good]):
        pool[st] = np.where(good & (strata == st))[0]
    want = {}
    for st in np.unique(strata[mask]):
        want[st] = int((strata[mask] == st).sum())

    rng = np.random.default_rng(seed)
    s_null = np.empty(n_null)
    for j in range(n_null):
        pick = []
        for st, cnt in want.items():
            cand = pool.get(st, np.array([], dtype=int))
            if len(cand) == 0:
                continue
            repl = len(cand) < cnt
            pick.append(rng.choice(cand, size=cnt, replace=repl))
        idx = np.concatenate(pick) if pick else np.array([], dtype=int)
        s_null[j] = _knn_stat(space[idx], k=k) if len(idx) > k else np.nan

    s_null = s_null[np.isfinite(s_null)]
    if not len(s_null):
        return {"n_anom": n_anom, "S_obs": s_obs, "p_value": float("nan"),
                "z": float("nan"), "insufficient": True}
    # Left tail: observed clustering (small S) more extreme than null.
    p_value = float((np.sum(s_null <= s_obs) + 1) / (len(s_null) + 1))
    mu, sd = float(np.mean(s_null)), float(np.std(s_null))
    z = float((s_obs - mu) / sd) if sd > 0 else float("nan")
    return {"n_anom": n_anom, "S_obs": float(s_obs), "S_null_mean": mu,
            "S_null_std": sd, "p_value": p_value, "z": z,
            "over_clustered": bool(p_value < 0.05), "insufficient": False}


def friends_of_friends(df: pd.DataFrame, space_cols, linking_length: float,
                       min_size: int = 3, standardize: bool = True) -> pd.Series:
    """Friends-of-friends grouping: link points within ``linking_length`` (in the
    same standardized units as the clustering statistic) and return an integer
    group label per row (-1 = ungrouped, groups smaller than ``min_size`` also
    -1).  Surfaces the concrete candidate co-moving groups behind an over-density.
    """
    X = df[list(space_cols)].to_numpy(float)
    if standardize:
        X = _standardize(X)
    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)
    good = np.all(np.isfinite(X), axis=1)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(X[good])
        pairs = tree.query_pairs(r=linking_length, output_type="ndarray")
        gi = np.where(good)[0]
        parent = np.arange(gi.size)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for u, v in pairs:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru
        roots = np.array([find(i) for i in range(gi.size)])
    except Exception:  # noqa: BLE001
        return pd.Series(labels, index=df.index)

    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    gid = -np.ones(gi.size, dtype=int)
    next_id = 0
    for u_i in range(len(uniq)):
        if counts[u_i] >= min_size:
            gid[inv == u_i] = next_id
            next_id += 1
    labels[gi] = gid
    return pd.Series(labels, index=df.index)


__all__ = ["matched_null_clustering", "friends_of_friends"]

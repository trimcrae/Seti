"""CONFLUENCE statistics: do tails of independent channels share stars?

Everything here is a pure function of in-memory tables, so the whole
statistic is testable offline.

The question
------------
Channel A and channel B each score stars from their own parent sample.  Take
the stars present in BOTH parents (the *joint parent*, J).  Each channel marks
its own tail (the top-q of its score over its OWN parent, or its own flag).
The observed confluence is

    O_AB = | tail_A  ∩  tail_B  ∩  J |.

The null is that, *once the stars are matched on everything that makes an
instrument misbehave* (brightness, colour, crowding, latitude, astrometric
quality, scan coverage, WISE depth), membership of A's tail says nothing about
membership of B's tail.  The matched null is the stratified permutation null:
inside each covariate cell c of J, shuffle B's tail labels over the n_c stars
of the cell while A's labels stay put.  Inside a cell that permutation makes
the overlap exactly hypergeometric (n_c, a_c, b_c); across cells the overlaps
are independent, so the null distribution of O_AB is the *convolution* of the
per-cell hypergeometrics.  It is computed exactly here -- no Monte-Carlo
error, no permutation count to choose -- and the p-value is P(O >= O_obs).

For three channels the per-cell count |A ∩ B ∩ C| under independent shuffles
is a hypergeometric mixture: X = |A ∩ B| ~ H(n, a, b), then |X ∩ C| ~ H(n, X, c).
That too is exact and convolved over cells.

Why cells and not a regression
------------------------------
A shared systematic (a blend, a bad scan, a bright-star artefact) moves a star
into the tail of two channels at once *because of* its covariates.  Holding the
covariate cell fixed removes exactly that route and nothing else; a regression
would impose a functional form on how the systematic acts.  The price is that a
cell holding one or two joint stars can absorb a true overlap into its own
expectation (a star alone in its cell that is in both tails is expected there
with probability 1).  That is the conservative direction and it is reported
(``n_cells``, ``frac_joint_in_singletons``) rather than hidden.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# ---------------------------------------------------------------------------
# Tails
# ---------------------------------------------------------------------------


def top_q_tail(score: pd.Series, q: float) -> pd.Series:
    """Boolean tail: the top-``q`` fraction of finite scores (higher = more anomalous).

    Ties at the threshold are all included, so the realised fraction can
    exceed ``q`` slightly on a discrete score; it can never silently fall to
    zero on a small parent (at least one star is in any non-empty tail).
    Non-finite scores are never in the tail.
    """
    s = pd.to_numeric(score, errors="coerce").astype(float)
    fin = np.isfinite(s.to_numpy())
    out = pd.Series(False, index=score.index)
    n = int(fin.sum())
    if n == 0 or q <= 0:
        return out
    k = max(1, int(math.ceil(q * n)))
    thr = np.sort(s.to_numpy()[fin])[::-1][k - 1]
    out[fin] = s.to_numpy()[fin] >= thr
    return out


def percentile_rank(score: pd.Series) -> pd.Series:
    """Upper-tail rank in (0, 1]: 1/N for the single most anomalous star."""
    s = pd.to_numeric(score, errors="coerce").astype(float)
    r = s.rank(ascending=False, method="max")
    n = int(np.isfinite(s.to_numpy()).sum())
    return r / max(n, 1)


# ---------------------------------------------------------------------------
# Covariate cells
# ---------------------------------------------------------------------------

#: Bin edges per covariate.  Deliberately coarse: a cell must hold enough joint
#: stars for the permutation to have somewhere to go.  Every edge list is
#: open-ended at both extremes (values outside fall into the end bins).
DEFAULT_EDGES: dict[str, Sequence[float]] = {
    "phot_g_mean_mag": (8.0, 10.0, 12.0, 14.0, 16.0, 18.0),
    "bp_rp": (0.5, 1.0, 1.5, 2.5),
    "abs_b": (10.0, 20.0, 40.0),
    "ruwe": (1.2, 1.4, 2.0),
    "log_density": None,    # tertiles of the joint parent
    "scan_coverage": None,  # tertiles (Gaia visibility_periods_used)
    "wise_depth": None,     # tertiles (|ecliptic latitude| when nothing better)
}


def assign_cells(cov: pd.DataFrame, covariates: Sequence[str] | None = None,
                 edges: dict[str, Sequence[float] | None] | None = None,
                 quantiles: int = 3) -> tuple[pd.Series, dict]:
    """Cell label per row of ``cov`` from the named covariates.

    A covariate with ``None`` edges is cut at the joint parent's own
    ``quantiles`` (tertiles by default).  A missing value gets its own bin
    ("na") rather than being dropped: a star whose RUWE is unknown is a
    different kind of star from one whose RUWE is 1.0, and dropping it would
    silently shrink the joint parent.  A covariate absent from ``cov`` is
    skipped and recorded in the returned report -- the null is then *not*
    matched on it, and the report says so.
    """
    edges = {**DEFAULT_EDGES, **(edges or {})}
    covariates = list(covariates if covariates is not None else DEFAULT_EDGES)
    used, skipped = [], []
    parts = []
    for c in covariates:
        if c not in cov.columns:
            skipped.append(c)
            continue
        v = pd.to_numeric(cov[c], errors="coerce").astype(float).to_numpy()
        fin = np.isfinite(v)
        if fin.sum() == 0:
            skipped.append(c)
            continue
        e = edges.get(c)
        if e is None:
            qs = np.nanquantile(v[fin], np.linspace(0, 1, quantiles + 1)[1:-1])
            e = np.unique(qs)
        lab = np.where(fin, np.digitize(np.where(fin, v, 0.0), np.asarray(e, float)).astype(str),
                       "na")
        parts.append(pd.Series(lab, index=cov.index, dtype=object).radd(c[:3] + "="))
        used.append(c)
    if parts:
        cell = parts[0]
        for p in parts[1:]:
            cell = cell + "|" + p
    else:
        cell = pd.Series("all", index=cov.index, dtype=object)
    return cell, {"covariates_used": used, "covariates_missing": skipped}


def knn_propensity(cov: pd.DataFrame, y: pd.Series, covariates: Sequence[str],
                   k: int = 100) -> np.ndarray:
    """P(in tail | covariates), estimated as the tail fraction among the star's k
    nearest neighbours in robust-standardised covariate space (itself excluded).

    Nonparametric on purpose: a systematic can switch on sharply (a saturation
    limit, a crowding threshold) and a coarse bin edge in the wrong place leaves
    it inside a cell.  A missing covariate value is set to the median, which is
    a 'typical star' placement rather than a drop.
    """
    from scipy.spatial import cKDTree

    cols = [c for c in covariates if c in cov.columns]
    yv = np.asarray(y, dtype=float)
    n = len(yv)
    if not cols or n < 3:
        return np.full(n, yv.mean() if n else np.nan)
    X = np.column_stack([pd.to_numeric(cov[c], errors="coerce").to_numpy(float) for c in cols])
    for j in range(X.shape[1]):
        col = X[:, j]
        fin = np.isfinite(col)
        if not fin.any():
            X[:, j] = 0.0
            continue
        med = np.median(col[fin])
        iqr = np.subtract(*np.percentile(col[fin], [75, 25])) or np.std(col[fin]) or 1.0
        col = np.where(fin, col, med)
        X[:, j] = (col - med) / iqr
    kk = int(min(max(k, 1), n - 1))
    tree = cKDTree(X)
    out = np.empty(n)
    step = 20_000     # bounded memory: n x (k+1) indices per block
    for s0 in range(0, n, step):
        rows = np.arange(s0, min(n, s0 + step))
        _, idx = tree.query(X[rows], k=kk + 1)
        # drop the star itself by identity (with tied covariates it need not be
        # the first neighbour); where it is absent, drop the farthest instead
        is_self = idx == rows[:, None]
        has = is_self.any(axis=1)
        is_self[~has, -1] = True
        keep = idx[~is_self].reshape(len(rows), kk)
        out[rows] = yv[keep].mean(axis=1)
    return out


def poisson_binomial_sf(p: np.ndarray, o: int) -> float:
    """P(sum of independent Bernoulli(p_i) >= o), exact, truncated DP."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if o <= 0:
        return 1.0
    if p.size == 0:
        return 0.0
    if o > 5000:
        # far outside the regime this channel meets; a refined normal tail
        from scipy.stats import norm
        mu, sd = p.sum(), np.sqrt((p * (1 - p)).sum())
        return float(norm.sf((o - 0.5 - mu) / max(sd, 1e-12)))
    # dist[j] = P(count == j) for j < o; mass that reaches o is absorbed
    dist = np.zeros(o)
    dist[0] = 1.0
    reached = 0.0
    for pi in np.clip(p, 0.0, 1.0):
        reached += dist[-1] * pi
        dist[1:] = dist[1:] * (1 - pi) + dist[:-1] * pi
        dist[0] *= (1 - pi)
    return float(min(1.0, max(0.0, reached)))


def conditional_overlap(tails: pd.DataFrame, cov: pd.DataFrame, channels: Sequence[str],
                        covariates: Sequence[str], k: int = 100) -> OverlapResult:
    """Headline null: conditional independence given the covariates.

    Each star's probability of sitting in channel X's tail given its covariates,
    p_X(i), is estimated nonparametrically (``knn_propensity``).  If the tails
    are independent once the covariates are fixed, star i sits in all of them
    with probability prod_X p_X(i), and the overlap count is Poisson-binomial
    with those probabilities -- computed exactly.  A shared systematic raises
    both p's for the same stars and is absorbed; an overlap the covariates
    cannot predict is not.
    """
    chans = tuple(channels)
    t = tails[list(chans)].astype(bool)
    cols = [c for c in covariates if c in cov.columns
            and pd.to_numeric(cov[c], errors="coerce").notna().any()]
    covJ = cov.reindex(t.index)
    prod = np.ones(len(t))
    for ch in chans:
        prod *= knn_propensity(covJ, t[ch], cols, k=k)
    both = t.all(axis=1)
    o = int(both.sum())
    lam = float(prod.sum())
    return OverlapResult(
        channels=chans, n_joint=int(len(t)), tail_counts={c: int(t[c].sum()) for c in chans},
        observed=o, expected=lam, p_value=poisson_binomial_sf(prod, o),
        ratio=(o / lam) if lam > 0 else (np.inf if o else np.nan), n_cells=0,
        frac_joint_in_singletons=0.0, members=list(t.index[both]),
        null_sd=float(np.sqrt((prod * (1 - prod)).sum())))


def propensity_cells(cov: pd.DataFrame, tails: pd.DataFrame, channels: Sequence[str],
                     covariates: Sequence[str], k: int = 100, bins: int | None = None
                     ) -> tuple[pd.Series, dict]:
    """Strata from each channel's covariate-predicted tail propensity.

    Each star gets, per channel, the kNN-estimated probability of sitting in
    that channel's tail given its covariates; the stratum is the joint quantile
    bin of those propensities (5 x 5 for a pair, 3 x 3 x 3 for a triple).  A
    shared systematic raises both propensities together and is absorbed; an
    overlap the covariates cannot predict is not.
    """
    bins = bins or (5 if len(channels) == 2 else 3)
    cols = [c for c in covariates if c in cov.columns
            and pd.to_numeric(cov[c], errors="coerce").notna().any()]
    lab = pd.Series("", index=tails.index, dtype=object)
    for ch in channels:
        p = knn_propensity(cov.reindex(tails.index), tails[ch], cols, k=k)
        r = pd.Series(p, index=tails.index).rank(method="average", pct=True)
        b = np.minimum((r.to_numpy() * bins).astype(int), bins - 1)
        lab = lab + f"{ch[:4]}{b}|"
    return lab, {"covariates_used": cols,
                 "covariates_missing": [c for c in covariates if c not in cols],
                 "k": k, "bins": bins}


# ---------------------------------------------------------------------------
# Exact stratified-permutation nulls
# ---------------------------------------------------------------------------


def _conv(pmfs: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([1.0])
    for p in pmfs:
        out = np.convolve(out, p)
        # trim numerical dust at the far tail to keep the vector short
        nz = np.flatnonzero(out > 1e-300)
        out = out[: nz[-1] + 1] if nz.size else np.array([1.0])
    return out


def pair_cell_pmf(n: int, a: int, b: int) -> np.ndarray:
    """P(|A ∩ B| = k) for random subsets of sizes a, b of n (hypergeometric)."""
    lo, hi = max(0, a + b - n), min(a, b)
    k = np.arange(0, hi + 1)
    pmf = np.zeros(hi + 1)
    if hi >= lo:
        pmf[lo:] = hypergeom.pmf(k[lo:], n, a, b)
    return pmf


def triple_cell_pmf(n: int, a: int, b: int, c: int) -> np.ndarray:
    """P(|A ∩ B ∩ C| = k) for independent random subsets of sizes a, b, c of n."""
    px = pair_cell_pmf(n, a, b)
    hi = min(a, b, c)
    out = np.zeros(hi + 1)
    for x, w in enumerate(px):
        if w <= 0:
            continue
        out[: min(x, c) + 1] += w * pair_cell_pmf(n, x, c)[: hi + 1]
    return out


@dataclass
class OverlapResult:
    channels: tuple[str, ...]
    n_joint: int
    tail_counts: dict[str, int]
    observed: int
    expected: float
    p_value: float
    ratio: float
    n_cells: int
    frac_joint_in_singletons: float
    members: list = field(default_factory=list)
    null_sd: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "channels": list(self.channels), "n_joint": self.n_joint,
            "tail_counts": self.tail_counts, "observed": self.observed,
            "expected": round(self.expected, 4), "null_sd": round(self.null_sd, 4),
            "ratio": (round(self.ratio, 3) if np.isfinite(self.ratio) else None),
            "p_value": self.p_value, "n_cells": self.n_cells,
            "frac_joint_in_singletons": round(self.frac_joint_in_singletons, 4),
            "n_members": len(self.members),
        }


def stratified_overlap(tails: pd.DataFrame, cells: pd.Series,
                       channels: Sequence[str]) -> OverlapResult:
    """Exact stratified-permutation test for 2 or 3 channels.

    ``tails`` is indexed by star and holds one boolean column per channel,
    restricted to the joint parent already (every row is in every named
    channel's parent).  ``cells`` gives each row's covariate cell.
    """
    chans = tuple(channels)
    if len(chans) not in (2, 3):
        raise ValueError("stratified_overlap handles pairs and triples")
    t = tails[list(chans)].astype(bool)
    both = t.all(axis=1)
    observed = int(both.sum())
    pmfs = []
    df = t.copy()
    df["_cell"] = cells.reindex(t.index).fillna("na").to_numpy()
    grp = df.groupby("_cell", sort=False)
    sizes = grp.size()
    counts = grp[list(chans)].sum()
    for cell, n in sizes.items():
        n = int(n)
        cnt = [int(counts.loc[cell, c]) for c in chans]
        if min(cnt) == 0:
            continue
        pmfs.append(pair_cell_pmf(n, *cnt) if len(chans) == 2 else triple_cell_pmf(n, *cnt))
    null = _conv(pmfs)
    k = np.arange(null.size)
    expected = float((k * null).sum())
    var = float(((k - expected) ** 2 * null).sum())
    p = float(null[observed:].sum()) if observed < null.size else 0.0
    p = min(1.0, max(p, 0.0))
    singles = int(sizes[sizes <= 1].sum())
    return OverlapResult(
        channels=chans, n_joint=int(len(t)),
        tail_counts={c: int(t[c].sum()) for c in chans},
        observed=observed, expected=expected, p_value=p,
        ratio=(observed / expected) if expected > 0 else (np.inf if observed else np.nan),
        n_cells=int(len(sizes)),
        frac_joint_in_singletons=(singles / len(t)) if len(t) else 0.0,
        members=list(t.index[both]), null_sd=math.sqrt(max(var, 0.0)))


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """BH q-values (monotone), NaN-safe."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    m = pv.size
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def inject_subthreshold(scores: dict[str, pd.Series], stars: Sequence, q: float,
                        alert_q: float, rng: np.random.Generator) -> dict[str, pd.Series]:
    """Plant anomalies that NO single channel would alert on.

    For each named channel the chosen ``stars`` get scores drawn uniformly
    between the channel's own upper-``q`` and upper-``alert_q`` quantiles
    (``alert_q < q``): inside the confluence tail, below the channel's alert
    level.  Returns modified copies; the inputs are untouched.
    """
    if not alert_q < q:
        raise ValueError("alert_q must be stricter than q")
    out = {}
    for ch, s in scores.items():
        s2 = pd.to_numeric(s, errors="coerce").astype(float).copy()
        fin = s2[np.isfinite(s2.to_numpy())]
        # margins keep a planted star inside the q-tail after the plant itself
        # shifts the ranks, and outside the alert tail
        lo, hi = np.quantile(fin, 1 - 0.8 * q), np.quantile(fin, 1 - 1.5 * alert_q)
        idx = [x for x in stars if x in s2.index]
        if hi <= lo:
            hi = lo + 1e-9
        s2.loc[idx] = rng.uniform(lo, hi, size=len(idx))
        out[ch] = s2
    return out

"""Population-level tests on the accumulated event set.

Everything else in TOCSIN asks *"is this star special?"*.  That question dies on
per-object contamination, and it died three times on real data: two candidates
were a deep-drilling field's observing strategy, three were low-amplitude
variable stars.  Any single grey repeater will always have a mundane reading
available.

This module asks a question that is **immune to that failure mode**: is the
*rate* of grey events structured across the target population in a way a
contaminant population cannot produce?  Contaminants — flares, subtraction
residuals, cosmic rays, satellite glints — trace the ordinary distribution of
observed stars, so they can inflate the rate everywhere but they cannot
manufacture a coherent gradient with distance, a sharp edge, or over-clustering
in phase space.

It is the same argument, and the same engines, that ``seti.cluster`` and
``seti.tidemark`` already make for the archival channels
(``docs/tidemark.md``) — reused here rather than reimplemented, on a population
those channels could never reach because it only exists in a live stream.

Three statistics, one null
--------------------------
The null is the load-bearing part.  It is **not** "uniform sky": it is random
subsets of the *same screened target population*, matched in the covariates that
would otherwise make any subset look structured —

* ``n_visits`` — a star looked at 48 times has far more chance to produce an
  event than one looked at 3 times, and visit count is strongly structured on
  the sky (deep-drilling fields);
* ``local_rate`` — the alert density of the star's own sky bin;
* ``g_mag`` / ``bp_rp`` — brightness and colour drive both the detectability of
  a small fractional amplitude and the intrinsic flare rate;
* ``parallax`` — distance, which is what a radial gradient is measured against
  and so must be matched *out* of the null.

Without that matching every statistic below would fire on the survey's own
footprint. With it, they measure the residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Covariates matched out of the null.  See the module docstring: each one is a
# confounder that independently structures the event rate across the sky.
DEFAULT_COVARIATES = ("n_visits", "local_rate", "phot_g_mean_mag", "bp_rp")


@dataclass
class PopulationResult:
    """One population statistic, with its degradation stated."""

    name: str
    statistic: float = float("nan")
    p_value: float = float("nan")
    z: float = float("nan")
    n_anomaly: int = 0
    n_parent: int = 0
    n_null: int = 0
    detail: dict = field(default_factory=dict)
    ok: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        d = {
            "name": self.name, "statistic": self.statistic,
            "p_value": self.p_value, "z": self.z,
            "n_anomaly": self.n_anomaly, "n_parent": self.n_parent,
            "n_null": self.n_null, "ok": self.ok, "reason": self.reason,
        }
        d.update(self.detail)
        return d


def _finite_cols(rows: list[dict], cols) -> np.ndarray:
    """Covariate matrix, with non-finite entries left as NaN for the binner."""
    out = np.full((len(rows), len(cols)), np.nan)
    for j, c in enumerate(cols):
        for i, r in enumerate(rows):
            v = r.get(c)
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                out[i, j] = f
    return out


def _strata_labels(cov: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin each covariate and label the joint cell.

    A star with a missing covariate goes into its own ``-1`` stratum rather than
    being dropped: dropping it would quietly shrink the parent population that
    the null is drawn from, which is the population the p-value is about.
    """
    n, k = cov.shape
    codes = np.zeros((n, k), dtype=int)
    for j in range(k):
        x = cov[:, j]
        good = np.isfinite(x)
        if good.sum() < n_bins:
            codes[:, j] = np.where(good, 0, -1)
            continue
        edges = np.quantile(x[good], np.linspace(0, 1, n_bins + 1)[1:-1])
        codes[:, j] = np.where(good, np.digitize(x, edges), -1)
    labels = np.zeros(n, dtype=np.int64)
    for j in range(k):
        labels = labels * (n_bins + 2) + (codes[:, j] + 1)
    return labels


def matched_draws(labels: np.ndarray, mask: np.ndarray, n_null: int,
                  rng: np.random.Generator) -> np.ndarray:
    """``n_null`` random masks with the same per-stratum counts as ``mask``.

    This is the whole defence.  Drawing uniformly from the parent would let any
    statistic that correlates with visit count or sky density fire on the
    survey's footprint; drawing *within strata* holds those constant, so what
    survives is structure the covariates do not explain.
    """
    out = np.zeros((n_null, mask.size), dtype=bool)
    by_stratum: dict[int, np.ndarray] = {}
    for lab in np.unique(labels):
        by_stratum[int(lab)] = np.nonzero(labels == lab)[0]
    wanted = {int(lab): int((mask & (labels == lab)).sum())
              for lab in np.unique(labels)}
    for d in range(n_null):
        for lab, k in wanted.items():
            if k == 0:
                continue
            pool = by_stratum[lab]
            take = rng.choice(pool, size=min(k, pool.size), replace=False)
            out[d, take] = True
    return out


# ---------------------------------------------------------------------------
# 1. Over-clustering in phase space
# ---------------------------------------------------------------------------
def clustering_test(coords: np.ndarray, mask: np.ndarray, labels: np.ndarray,
                    n_null: int = 500, k: int = 1,
                    rng: np.random.Generator | None = None) -> PopulationResult:
    """Are the event-bearing targets closer together than matched random ones?

    The statistic is the median k-nearest-neighbour distance among the anomalies
    in standardised coordinates; a **left** tail means over-clustered.  A
    technological population spreading from an origin over-clusters; flares,
    blends and noise trace the parent density and do not.
    """
    res = PopulationResult("phase_space_clustering", n_parent=int(mask.size),
                           n_anomaly=int(mask.sum()), n_null=int(n_null))
    X = np.asarray(coords, float)
    good = np.all(np.isfinite(X), axis=1)
    m = np.asarray(mask, bool) & good
    if m.sum() <= k + 1:
        res.reason = "too_few_anomalies_with_coordinates"
        return res
    sd = np.nanstd(X[good], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - np.nanmean(X[good], axis=0)) / sd

    def stat(sel: np.ndarray) -> float:
        pts = Xs[sel]
        if pts.shape[0] <= k:
            return float("nan")
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(np.median(np.sort(d, axis=1)[:, k - 1]))

    s_obs = stat(m)
    rng = rng or np.random.default_rng(31337)
    draws = matched_draws(labels, m, n_null, rng)
    null = np.array([stat(draws[i] & good) for i in range(n_null)])
    null = null[np.isfinite(null)]
    if null.size < 20 or not math.isfinite(s_obs):
        res.reason = "null_did_not_populate"
        return res
    # Add-one: with n_null draws the smallest resolvable p-value is 1/(n+1).
    res.statistic = s_obs
    res.p_value = (int((null <= s_obs).sum()) + 1) / (null.size + 1)
    res.z = float((s_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    res.detail = {"null_median": float(np.median(null)), "n_null_used": int(null.size)}
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# 2. A gradient in the event rate along a coordinate
# ---------------------------------------------------------------------------
def gradient_test(coord: np.ndarray, mask: np.ndarray, labels: np.ndarray,
                  name: str = "coord", n_bins: int = 8, n_null: int = 500,
                  rng: np.random.Generator | None = None) -> PopulationResult:
    """Does the event rate vary monotonically along ``coord``?

    The statistic is Spearman correlation between bin centre and rate excess.
    Reported against the matched null, so a gradient produced by the survey's
    own depth or cadence structure cancels.
    """
    res = PopulationResult(f"gradient_{name}", n_parent=int(mask.size),
                           n_anomaly=int(mask.sum()), n_null=int(n_null))
    x = np.asarray(coord, float)
    good = np.isfinite(x)
    m = np.asarray(mask, bool) & good
    if m.sum() < 6 or good.sum() < 4 * n_bins:
        res.reason = "too_few_for_a_gradient"
        return res

    # The statistic is the mean RANK of the coordinate among the anomalies,
    # centred so 0 is no trend.  Ranking individual targets rather than binning
    # them keeps all the information: an earlier binned-Spearman version needed
    # roughly an order of magnitude more signal to reach the same p-value,
    # because with ~8 bins the statistic is coarse and its null is wide.
    ranks = np.full(x.size, np.nan)
    idx = np.nonzero(good)[0]
    ranks[idx] = np.argsort(np.argsort(x[idx])).astype(float) / max(1, idx.size - 1)

    def stat(sel: np.ndarray) -> float:
        r = ranks[sel & good]
        if r.size < 3:
            return float("nan")
        return float(r.mean() - 0.5)

    s_obs = stat(m)
    rng = rng or np.random.default_rng(4242)
    draws = matched_draws(labels, m, n_null, rng)
    null = np.array([stat(draws[i]) for i in range(n_null)])
    null = null[np.isfinite(null)]
    if null.size < 20 or not math.isfinite(s_obs):
        res.reason = "null_did_not_populate"
        return res
    res.statistic = s_obs
    # Two-sided: a gradient in either direction is a claim.
    res.p_value = (int((np.abs(null) >= abs(s_obs)).sum()) + 1) / (null.size + 1)
    res.z = float((s_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    res.detail = {"coordinate": name,
                  "statistic_is": "mean_rank_of_coordinate_among_anomalies_minus_0.5",
                  "null_absmedian": float(np.median(np.abs(null)))}
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# 3. A sharp edge in the event rate
# ---------------------------------------------------------------------------
def edge_test(coord: np.ndarray, mask: np.ndarray, labels: np.ndarray,
              name: str = "coord", n_null: int = 500, min_side: int = 30,
              rng: np.random.Generator | None = None) -> PopulationResult:
    """Is there a location along ``coord`` where the rate steps?

    Carrigan 2010, Landis 1998 and Hanson et al. 2021 all predict a *boundary*
    rather than a gradient — a colonised volume has an edge.  The statistic is
    the maximum absolute rate contrast over all split points with enough targets
    on each side; the null is the same scan on matched draws, which is what
    stops the maximum-over-splits from being significant by construction.
    """
    res = PopulationResult(f"edge_{name}", n_parent=int(mask.size),
                           n_anomaly=int(mask.sum()), n_null=int(n_null))
    x = np.asarray(coord, float)
    good = np.isfinite(x)
    m = np.asarray(mask, bool) & good
    if good.sum() < 4 * min_side or m.sum() < 6:
        res.reason = "too_few_for_an_edge"
        return res
    order = np.argsort(x[good])
    idx = np.nonzero(good)[0][order]
    splits = range(min_side, idx.size - min_side, max(1, idx.size // 200))

    def stat(sel: np.ndarray) -> tuple[float, float]:
        s = sel[idx].astype(float)
        cum = np.concatenate([[0.0], np.cumsum(s)])
        best, best_at = 0.0, float("nan")
        for c in splits:
            r_in = cum[c] / c
            r_out = (cum[-1] - cum[c]) / (idx.size - c)
            contrast = abs(r_in - r_out)
            if contrast > best:
                best, best_at = contrast, float(x[idx[c]])
        return best, best_at

    s_obs, at = stat(m)
    rng = rng or np.random.default_rng(909)
    draws = matched_draws(labels, m, n_null, rng)
    null = np.array([stat(draws[i])[0] for i in range(n_null)])
    null = null[np.isfinite(null)]
    if null.size < 20 or not math.isfinite(s_obs):
        res.reason = "null_did_not_populate"
        return res
    res.statistic = s_obs
    res.p_value = (int((null >= s_obs).sum()) + 1) / (null.size + 1)
    res.z = float((s_obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    res.detail = {"coordinate": name, "edge_at": at,
                  "null_median": float(np.median(null))}
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def population_tests(rows: list[dict], mask, covariates=DEFAULT_COVARIATES,
                     n_null: int = 500, n_strata_bins: int = 3,
                     seed: int = 20260730) -> dict:
    """Run all three population statistics on the screened target population.

    ``rows`` is one dict per screened target (the parent), ``mask`` marks those
    that produced at least one kept event.  Coordinates are taken from whatever
    is present, so a run whose target table lacks parallax simply reports the
    radial tests as untestable instead of failing.
    """
    n = len(rows)
    out: dict = {"n_parent": n, "n_anomaly": int(np.asarray(mask, bool).sum()),
                 "covariates": list(covariates), "n_null": int(n_null),
                 "tests": []}
    if n < 50 or out["n_anomaly"] < 6:
        out["verdict"] = "INSUFFICIENT_POPULATION"
        out["note"] = ("population tests need a parent of >=50 screened targets "
                       "and >=6 with events; below that the matched null cannot "
                       "be populated and any p-value would be noise")
        return out
    m = np.asarray(mask, bool)
    labels = _strata_labels(_finite_cols(rows, covariates), n_strata_bins)
    out["n_strata"] = int(np.unique(labels).size)
    rng = np.random.default_rng(seed)

    ra = np.array([float(r.get("ra", np.nan) or np.nan) for r in rows])
    dec = np.array([float(r.get("dec", np.nan) or np.nan) for r in rows])
    plx = np.array([float(r.get("parallax", np.nan) or np.nan) for r in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(plx > 0, 1000.0 / plx, np.nan)

    # Galactic-ish 3-D position for the clustering test; falls back to sky-only
    # when parallax is absent, and says so.
    cd = np.cos(np.radians(dec))
    xyz = np.column_stack([dist * cd * np.cos(np.radians(ra)),
                           dist * cd * np.sin(np.radians(ra)),
                           dist * np.sin(np.radians(dec))])
    out["tests"].append(
        clustering_test(xyz, m, labels, n_null=n_null, rng=rng).as_dict())
    for coord, nm in ((dist, "distance_pc"), (dec, "declination"),
                      (np.abs(np.sin(np.radians(dec))), "abs_sin_dec")):
        out["tests"].append(
            gradient_test(coord, m, labels, name=nm, n_null=n_null,
                          rng=rng).as_dict())
    out["tests"].append(
        edge_test(dist, m, labels, name="distance_pc", n_null=n_null,
                  rng=rng).as_dict())

    usable = [t for t in out["tests"] if t.get("ok")]
    out["n_tests_usable"] = len(usable)
    if not usable:
        out["verdict"] = "NO_TEST_COULD_RUN"
        return out
    # Multiple-testing across the statistics themselves: five correlated tests
    # on one population, so the smallest p-value is compared to a Bonferroni
    # threshold rather than to alpha.  Correlated tests make this conservative,
    # which is the right direction.
    pmin = min(t["p_value"] for t in usable)
    out["p_min"] = pmin
    out["bonferroni_threshold"] = 0.05 / len(usable)
    # RESOLUTION GUARD.  With `n_null` randomisations the smallest p-value that
    # can be measured at all is 1/(n_null+1).  If that floor is not comfortably
    # below the Bonferroni threshold, then a statistic sitting *at* the floor
    # would clear the threshold without the randomisations having demonstrated
    # anything -- the claim would be an artefact of how few draws were taken.
    # Say so instead, in the same spirit as quoting an injection-limited rate as
    # an inequality.
    out["p_resolution_floor"] = 1.0 / (int(n_null) + 1)
    if out["p_resolution_floor"] > 0.2 * out["bonferroni_threshold"]:
        out["verdict"] = "INSUFFICIENT_RESOLUTION"
        out["note"] = (
            f"n_null={n_null} resolves p only to {out['p_resolution_floor']:.4f}, "
            f"which is not comfortably below the Bonferroni threshold "
            f"{out['bonferroni_threshold']:.4f} for {len(usable)} tests; raise "
            f"n_null to at least {int(5 * len(usable) / 0.05)} before believing "
            f"any detection here")
        return out
    out["verdict"] = ("STRUCTURE_DETECTED" if pmin <= out["bonferroni_threshold"]
                      else "NO_STRUCTURE")
    return out


# ---------------------------------------------------------------------------
# Reconstructing the parent population
# ---------------------------------------------------------------------------
def build_parent_population(targets, bin_trials: dict, bin_deg: float = 1.0,
                            event_targets: dict | None = None
                            ) -> tuple[list[dict], np.ndarray]:
    """The screened parent population and the event mask, from committed state.

    The ledger deliberately stores only targets that produced events — keeping
    254k rows of "nothing happened" would balloon the file for no information.
    So the parent has to be rebuilt, and it can be rebuilt EXACTLY, because of
    how the footprint denominator works: every target in a 1-degree sky bin
    shares that bin's observed nights. Therefore

        nights observed per target in bin B = bin_trials[B] / (targets in bin B)

    which is exact rather than modelled, given that ``bin_trials`` was
    accumulated by counting one trial per (target, night) over the same bins.

    Returns ``(rows, mask)`` ready for :func:`population_tests`.  A target whose
    bin has no recorded trials was never screened and is excluded — including it
    would dilute the parent with stars the survey never looked at.
    """
    import pandas as pd

    if targets is None or not len(targets):
        return [], np.zeros(0, dtype=bool)
    df = targets if isinstance(targets, pd.DataFrame) else pd.DataFrame(targets)
    ra = np.asarray(df["ra"], float)
    dec = np.asarray(df["dec"], float)
    keys = [f"{math.floor(a / bin_deg)},{math.floor(d / bin_deg)}"
            for a, d in zip(ra, dec, strict=True)]
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1

    ids = (np.asarray(df["source_id"]).astype(str) if "source_id" in df
           else np.arange(len(df)).astype(str))
    ev = {str(k): v for k, v in (event_targets or {}).items()}

    def col(name):
        return np.asarray(df[name], float) if name in df else np.full(len(df), np.nan)

    plx, gmag, bprp = col("parallax"), col("phot_g_mean_mag"), col("bp_rp")
    rows, mask = [], []
    for i, key in enumerate(keys):
        trials = int(bin_trials.get(key, 0))
        if trials <= 0:
            continue
        n_bin = counts[key]
        tid = str(ids[i])
        rec = ev.get(tid)
        # Exact per-target nights for event-bearing targets; the bin quotient for
        # the rest.  They agree by construction, but preferring the stored value
        # keeps the anomalies' own denominator authoritative.
        n_visits = int(rec["n_visits"]) if rec and rec.get("n_visits") \
            else max(1, trials // max(1, n_bin))
        rows.append({
            "target_id": tid, "ra": float(ra[i]), "dec": float(dec[i]),
            "parallax": float(plx[i]), "phot_g_mean_mag": float(gmag[i]),
            "bp_rp": float(bprp[i]), "n_visits": n_visits,
            "local_rate": float(rec.get("local_rate") or np.nan) if rec
            else float(trials and np.nan),
            "sky_bin": key,
        })
        mask.append(bool(rec and int(rec.get("n_events") or 0) > 0))
    # `local_rate` is only stored for event-bearing targets, so fill the rest
    # from the bin they share — otherwise the covariate would be perfectly
    # correlated with being an anomaly and the matching would be degenerate.
    by_bin: dict[str, float] = {}
    for r in rows:
        lr = r["local_rate"]
        if lr is not None and math.isfinite(lr):
            by_bin[r["sky_bin"]] = lr
    for r in rows:
        if not math.isfinite(r["local_rate"] if r["local_rate"] is not None else np.nan):
            r["local_rate"] = by_bin.get(r["sky_bin"], float("nan"))
    return rows, np.asarray(mask, dtype=bool)

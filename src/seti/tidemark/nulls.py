"""Parent-matched resampling nulls --- the selection function, made explicit.

The whole scientific difficulty of a spatial technosignature test is that the
*detectability* of an anomaly varies with position.  Stars far away are fainter,
more crowded, more reddened, and less likely to have a spectrum; the survey
footprint is not the sky.  Any raw "anomaly rate vs Galactocentric radius" curve
is therefore mostly a map of the survey, not of the Galaxy.

This module builds the correction the same way ``seti.cluster`` builds its
clustering null --- by **resampling the parent catalogue itself** --- but
generalised from "is this subset clumped?" to "does this subset's *rate* vary
across a coordinate?".

The construction
----------------
Stratify the parent sample on the covariates that control **detectability**
(apparent magnitude, colour, distance, extinction, crowding, spectroscopic
coverage, per-object measurement error, metallicity...) and *never* on the
coordinate under test.  Within a stratum ``s`` the parent has ``N_s`` rows of
which ``c_s`` are anomalies.  Give every parent row the weight

    w_i = c_{s(i)} / N_{s(i)}

which is the empirical, non-parametric probability that a star with row ``i``'s
detectability is flagged.  Then for **any** region ``W`` of any coordinate,

    E(W) = sum_{i in W} w_i

is the number of anomalies that region would contain if the anomaly rate were
constant in the coordinate, given the observed detectability.  Note
``sum_i w_i == n_anom`` exactly, so the correction redistributes but never
invents anomalies.  The selection-corrected rate ratio is

    rho(W) = n_obs(W) / E(W)

and ``rho == 1`` everywhere is the null hypothesis.  ``w_i`` is the deliverable's
credibility: it is written out per star so a reader can audit the correction.

Why the coordinate must not be a stratification covariate is obvious, but the
converse deserves a warning, so ``MatchedNull`` **refuses** to stratify on a
column listed in ``forbid_cols``.  A silently self-cancelling test is worse than
no test.

Distance is the hard case: it is both a detectability covariate *and* carries
spatial information (Galactocentric radius is a function of distance and
direction).  Two modes are therefore supported and both are reported:

* ``strict``  --- distance is a stratification covariate.  A radial gradient
  must then show up as a *difference between directions at matched heliocentric
  distance*, which is immune to every distance-dependent selection effect.  This
  is the primary claim.
* ``permissive`` --- distance is not stratified on.  More sensitive, but a
  distance-dependent detectability leaks straight into the radial gradient.
  Reported as a cross-check only.

Monte Carlo draws are stratified (exactly ``c_s`` rows drawn from stratum ``s``),
so every realisation matches the anomaly set's covariate distribution *exactly*,
not just in expectation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A coordinate must never be used to match the null it is tested against.
DEFAULT_FORBIDDEN = (
    "R_gal_kpc", "z_gal_kpc", "abs_z_gal_kpc", "l_deg", "b_deg", "abs_b_deg",
    "X_pc", "Y_pc", "Z_pc", "X_gal_kpc", "Y_gal_kpc", "ra", "dec",
)


def quantile_bins(values, n_bins: int) -> np.ndarray:
    """Quantile-bin ``values`` into at most ``n_bins`` codes; NaN -> code -1.

    Missing covariates get their own stratum rather than being dropped: a star
    with no extinction estimate is a real member of the parent sample whose
    detectability is simply less well known, and pretending otherwise would bias
    the very correction we are building.
    """
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    out = np.full(v.shape, -1, dtype=np.int64)
    if ok.sum() < max(2, n_bins):
        out[ok] = 0
        return out
    edges = np.unique(np.quantile(v[ok], np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size < 3:
        out[ok] = 0
        return out
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, edges.size - 2)
    out[ok] = idx[ok]
    return out


def _codes_for(parent: pd.DataFrame, cols, n_bins: int,
               categorical: set) -> list[np.ndarray]:
    codes = []
    for c in cols:
        if c in categorical:
            v = parent[c].astype("object").where(parent[c].notna(), "__nan__")
            _, inv = np.unique(v.to_numpy().astype(str), return_inverse=True)
            codes.append(inv.astype(np.int64))
        else:
            codes.append(quantile_bins(
                pd.to_numeric(parent[c], errors="coerce").to_numpy(), n_bins))
    return codes


def _combine(codes: list[np.ndarray]) -> np.ndarray:
    """Fold a list of per-column bin codes into a single dense stratum id."""
    if not codes:
        return np.zeros(0, dtype=np.int64)
    acc = codes[0].astype(np.int64) + 1
    for c in codes[1:]:
        span = max(int(c.max()) + 2, 2)
        acc = acc * span + (c.astype(np.int64) + 1)
    _, inv = np.unique(acc, return_inverse=True)
    return inv.astype(np.int64)


@dataclass
class NullDiagnostics:
    """Everything a reader needs to judge whether the matching actually worked."""

    n_parent: int
    n_anom: int
    n_strata: int
    n_strata_with_anomaly: int
    min_pool: int
    median_pool: float
    frac_anom_in_thin_strata: float
    max_weight: float
    effective_parent_size: float
    collapse_levels_used: dict
    balance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return {k: (v if not isinstance(v, np.generic) else v.item())
                for k, v in d.items()}


class MatchedNull:
    """Stratified parent-matched null for a rate-vs-coordinate test.

    Parameters
    ----------
    parent
        Every star actually searched.  Not the survivors --- the *parent*.
    anomaly_mask
        Boolean, aligned to ``parent`` rows.
    covariate_cols
        Detectability covariates to match on.  Ordered by priority: when a
        stratum is too thin the *last* covariate is dropped first.
    forbid_cols
        Coordinates under test.  Passing one of these as a covariate raises ---
        matching on the test coordinate would silently null the test.
    n_bins
        Quantile bins per continuous covariate.
    min_pool
        Minimum parent rows per stratum.  Thinner strata are progressively
        collapsed onto coarser stratifications (dropping covariates from the end
        of ``covariate_cols``) until they are populated enough to resample from.
    """

    def __init__(self, parent: pd.DataFrame, anomaly_mask, covariate_cols,
                 *, forbid_cols=DEFAULT_FORBIDDEN, n_bins: int = 5,
                 min_pool: int = 25, categorical_cols=(), seed: int = 20260726,
                 tilt: np.ndarray | None = None):
        self.parent = parent.reset_index(drop=True)
        self.mask = np.asarray(anomaly_mask, bool)
        if self.mask.size != len(self.parent):
            raise ValueError("anomaly_mask length must match the parent sample")
        forbidden = {c.lower() for c in (forbid_cols or ())}
        clash = [c for c in covariate_cols if c.lower() in forbidden]
        if clash:
            raise ValueError(
                f"refusing to stratify the null on the coordinate(s) under test: {clash}. "
                "Matching on the tested coordinate cancels the signal by construction.")
        missing = [c for c in covariate_cols if c not in self.parent.columns]
        if missing:
            raise KeyError(f"covariate column(s) absent from the parent sample: {missing}")
        self.covariate_cols = list(covariate_cols)
        self.n_bins = int(n_bins)
        self.min_pool = int(min_pool)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

        self.strata, self._levels = self._build_strata(set(categorical_cols))
        self._index_strata()
        self.set_tilt(tilt)

    # ------------------------------------------------------------------ setup
    def _build_strata(self, categorical: set) -> tuple[np.ndarray, dict]:
        """Nested stratifications, finest first; each row is assigned to the
        finest level at which its stratum has at least ``min_pool`` parent rows."""
        cols = self.covariate_cols
        n = len(self.parent)
        if not cols:
            return np.zeros(n, dtype=np.int64), {0: n}
        all_codes = _codes_for(self.parent, cols, self.n_bins, categorical)
        levels = []                                   # finest -> coarsest
        for depth in range(len(cols), 0, -1):
            levels.append(_combine(all_codes[:depth]))
        levels.append(np.zeros(n, dtype=np.int64))    # everything in one stratum

        assigned_level = np.full(n, len(levels) - 1, dtype=np.int64)
        remaining = np.ones(n, bool)
        last = len(levels) - 1
        for li, code in enumerate(levels):
            if not remaining.any():
                break
            # Count only among rows still unassigned: a stratum's population is
            # the rows that actually end up in it, not the rows that share its
            # coarse code but were already claimed by a finer level.  Counting
            # the latter would leave final strata below ``min_pool`` and make the
            # resampling pool for those rows uselessly thin.
            counts = np.bincount(code[remaining], minlength=int(code.max()) + 1)
            ok = remaining & (counts[code] >= self.min_pool)
            if li == last:
                ok = remaining                     # coarsest level takes the rest
            assigned_level[ok] = li
            remaining &= ~ok
        # Final id = (level, code within that level), densified.
        packed = assigned_level * (int(max(c.max() for c in levels)) + 2)
        packed = packed + np.take_along_axis(
            np.stack(levels, axis=1), assigned_level[:, None], axis=1).ravel()
        _, strata = np.unique(packed, return_inverse=True)
        used = {int(li): int((assigned_level == li).sum()) for li in np.unique(assigned_level)}
        return strata.astype(np.int64), used

    def _index_strata(self) -> None:
        s = self.strata
        n_s = int(s.max()) + 1 if s.size else 0
        self.parent_count = np.bincount(s, minlength=n_s).astype(float)
        self.anom_count = np.bincount(s[self.mask], minlength=n_s).astype(float)
        order = np.argsort(s, kind="stable")
        self._sorted_idx = order
        self._starts = np.searchsorted(s[order], np.arange(n_s), side="left")
        self._ends = np.searchsorted(s[order], np.arange(n_s), side="right")
        self.active_strata = np.where(self.anom_count > 0)[0]

    # ----------------------------------------------------------------- tilt
    def set_tilt(self, tilt) -> MatchedNull:
        """Reweight *within* each stratum by a positive per-row factor.

        Used by the edge test: tilting the null by the measured smooth radial
        trend makes the null "the matched selection function *plus* the fitted
        gradient", so a strong smooth gradient cannot masquerade as a sharp
        edge.  With ``tilt=None`` the draw is uniform within a stratum.
        """
        n = len(self.parent)
        if tilt is None:
            t = np.ones(n)
        else:
            t = np.asarray(tilt, float)
            if t.shape != (n,):
                raise ValueError("tilt must be one positive value per parent row")
            t = np.where(np.isfinite(t) & (t > 0), t, 0.0)
        self._tilt = t
        # Per-stratum tilt sums, then per-row weights w_i = c_s * t_i / sum_s t.
        tsum = np.bincount(self.strata, weights=t,
                           minlength=self.parent_count.size).astype(float)
        # A stratum whose tilt sums to zero falls back to uniform.
        bad = tsum <= 0
        if bad.any():
            self._tilt = np.where(bad[self.strata], 1.0, self._tilt)
            tsum = np.bincount(self.strata, weights=self._tilt,
                               minlength=self.parent_count.size).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.weights = np.where(tsum[self.strata] > 0,
                                    self.anom_count[self.strata] * self._tilt
                                    / np.maximum(tsum[self.strata], 1e-300), 0.0)
        # Per-stratum draw probabilities, cached for the Monte Carlo.
        self._probs = {}
        for s in self.active_strata:
            sl = self._sorted_idx[self._starts[s]:self._ends[s]]
            p = self._tilt[sl]
            tot = p.sum()
            self._probs[int(s)] = (sl, (p / tot) if tot > 0 else None)
        return self

    def copy_with_tilt(self, tilt) -> MatchedNull:
        """A view of this null with different within-stratum draw probabilities.

        The stratification (the expensive part) is shared; only the weights and
        the per-stratum draw probabilities are rebuilt.
        """
        other = MatchedNull.__new__(MatchedNull)
        other.__dict__.update(self.__dict__)
        other._rng = np.random.default_rng(self.seed)
        return other.set_tilt(tilt)

    # ------------------------------------------------------------ expectation
    @property
    def n_anom(self) -> int:
        return int(self.mask.sum())

    def expected(self, region) -> float:
        """Expected anomaly count in a boolean region, under a constant rate."""
        r = np.asarray(region, bool)
        return float(self.weights[r].sum())

    def expected_binned(self, bin_index: np.ndarray, n_bins: int) -> np.ndarray:
        """Expected counts per bin, exactly (no Monte Carlo needed)."""
        b = np.asarray(bin_index, np.int64)
        ok = b >= 0
        return np.bincount(b[ok], weights=self.weights[ok], minlength=n_bins).astype(float)

    def observed_binned(self, bin_index: np.ndarray, n_bins: int) -> np.ndarray:
        b = np.asarray(bin_index, np.int64)
        ok = (b >= 0) & self.mask
        return np.bincount(b[ok], minlength=n_bins).astype(float)

    # ------------------------------------------------------------------ draws
    def draw(self, rng=None) -> np.ndarray:
        """One stratified matched draw: parent row indices, ``n_anom`` of them."""
        rng = rng or self._rng
        picks = []
        for s in self.active_strata:
            sl, p = self._probs[int(s)]
            cnt = int(self.anom_count[s])
            if sl.size == 0 or cnt == 0:
                continue
            picks.append(rng.choice(sl, size=cnt, replace=True, p=p))
        return np.concatenate(picks) if picks else np.empty(0, dtype=np.int64)

    def draws(self, n_draw: int, rng=None):
        """Generator over ``n_draw`` matched draws."""
        rng = rng or np.random.default_rng(self.seed + 1)
        for _ in range(int(n_draw)):
            yield self.draw(rng)

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self, extra_balance_cols=()) -> NullDiagnostics:
        pool = self.parent_count[self.active_strata] if self.active_strata.size else np.array([0.0])
        thin = self.active_strata[self.parent_count[self.active_strata] < self.min_pool]
        frac_thin = (float(self.anom_count[thin].sum()) / max(self.n_anom, 1)
                     if thin.size else 0.0)
        w = self.weights
        eff = float(w.sum() ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
        bal = {}
        for c in list(self.covariate_cols) + list(extra_balance_cols):
            if c not in self.parent.columns:
                continue
            v = pd.to_numeric(self.parent[c], errors="coerce").to_numpy()
            ok = np.isfinite(v)
            if ok.sum() < 10 or not (self.mask & ok).any():
                continue
            m_a = float(np.average(v[self.mask & ok]))
            wt = w[ok]
            m_n = float(np.average(v[ok], weights=wt)) if wt.sum() > 0 else float("nan")
            sd = float(np.std(v[ok])) or 1.0
            bal[c] = {"anomaly_mean": m_a, "matched_null_mean": m_n,
                      "std_diff": (m_a - m_n) / sd}
        return NullDiagnostics(
            n_parent=len(self.parent), n_anom=self.n_anom,
            n_strata=int(self.parent_count.size),
            n_strata_with_anomaly=int(self.active_strata.size),
            min_pool=int(pool.min()), median_pool=float(np.median(pool)),
            frac_anom_in_thin_strata=frac_thin, max_weight=float(w.max() if w.size else 0.0),
            effective_parent_size=eff, collapse_levels_used=self._levels, balance=bal)


def empirical_p(observed: float, null_values, *, tail: str = "two") -> float:
    """Monte Carlo p-value with the standard +1 correction (never returns 0)."""
    v = np.asarray([x for x in np.ravel(null_values) if np.isfinite(x)], float)
    if v.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "greater":
        k = int(np.sum(v >= observed))
    elif tail == "less":
        k = int(np.sum(v <= observed))
    else:
        c = float(np.median(v))
        k = int(np.sum(np.abs(v - c) >= abs(observed - c)))
    return float((k + 1) / (v.size + 1))


__all__ = ["MatchedNull", "NullDiagnostics", "empirical_p", "quantile_bins",
           "DEFAULT_FORBIDDEN"]

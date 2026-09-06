"""FALLOUT detector: fit each star's n-capture vector as solar + source mixtures.

The observable is a **vector**, not an element. Fission product, after its
short-lived members decay, has a two-humped mass distribution (light peak
A~90-105: Sr/Y/Zr/Mo/Ru; heavy peak A~133-145: Xe/Cs/Ba/La/Ce/Pr/Nd) with a
~1000x valley between them and almost nothing beyond A~155. Folded against
solar abundances that gives a fixed *shape* in ``[X/H]`` space, and the shape
is what natural nucleosynthesis cannot reproduce:

* relative to solar, fission gives ``[Nd/Ba] >> 0``, ``[Ce/Ba] > 0``,
  ``[La/Ba] > 0``, ``[Mo/Zr] > 0``, ``[Ru/Zr] > 0`` and ``[Eu/Nd] < 0``;
* the s-process gives ``[Nd/Ba] < 0`` and ``[Mo/Zr] < 0`` (it makes Ba and Zr
  *well*), and moves Sr/Y strongly where fission barely touches them;
* the r-process gives ``[Eu/Nd] > 0`` -- Eu is the r-process element, and
  fission makes almost none of it.

So each star is fitted five ways -- nothing added; pure s; pure r; s + r;
pure fission -- with the amplitude of each source a free non-negative
parameter, and the statistic is the log-likelihood ratio of the fission-only
fit against the **best natural** fit. The natural alternative has two free
amplitudes against fission's one, so the comparison is biased *against*
fission; that is the conservative direction.

Everything here is a pure function of arrays, offline-testable. The only
survey-specific step is assembling the vectors from a normalised table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..tailings import manifold as M
from . import yields as Y

# ---------------------------------------------------------------------------
# Classification labels and veto names
# ---------------------------------------------------------------------------
NORMAL = "NORMAL"
S_PROCESS = "S_PROCESS"
R_PROCESS = "R_PROCESS"
S_PLUS_R = "S_PLUS_R"
FISSION = "FISSION"
AMBIGUOUS = "AMBIGUOUS"
INSUFFICIENT = "INSUFFICIENT"

#: Every veto, in the order they are applied. Each is a named counter in
#: ``summary.json``; ``first_veto`` records which one a star hit first.
VETOES: tuple[str, ...] = (
    "low_snr_or_flagged",
    "s_process_star",
    "r_process_star",
    "young_ba_enhancement",
    "nlte_saturated_lines",
    "single_element_driver",
    "teff_peer_residual",
)


@dataclass(frozen=True)
class PatternConfig:
    """Thresholds; every value is mirrored in ``config/fallout.yaml``."""

    elements: tuple[str, ...] = Y.NCAPTURE_ELEMENTS
    horizon_yr: float = Y.DEFAULT_HORIZON_YR
    # amplitude grid: 0 plus a log grid from amp_min to amp_max
    amp_min: float = 0.03
    amp_max: float = 100.0
    n_amp: int = 40
    n_amp_2d: int = 26
    # error model: sigma = sqrt(reported^2 + floor^2); reported missing -> default
    systematic_floor_dex: float = 0.05
    error_default_dex: float = 0.15
    min_elements: int = 5
    # decision thresholds (ln likelihood ratios; Delta chi2 = 2 * lr)
    lr_min: float = 8.0
    enrich_min: float = 12.5
    ambiguity_margin: float = 2.0
    # the heavy-peak anchors that must remain when Ba is excluded
    ba_element: str = "Ba"
    core_elements: tuple[str, ...] = ("Y", "Zr", "Ba", "La", "Ce", "Nd", "Sm", "Eu")
    # vetoes
    snr_min: float = 40.0
    s_ba_fe_min: float = 0.5
    s_light_fe_min: float = 0.3
    r_eu_fe_min: float = 0.3
    young_ali_min: float = 2.3
    young_age_max_gyr: float = 1.0
    young_ba_fe_min: float = 0.2
    raw_consistency_fraction: float = 0.5
    # nulls
    null_quantile: float = 0.999
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
@dataclass
class Templates:
    """The three pure-source patterns over a fixed element order."""

    elements: list[str]
    F: np.ndarray   # fission, normalised so a_f = 1 doubles the anchor (Nd)
    S: np.ndarray   # solar s-process fraction
    R: np.ndarray   # solar r-process fraction
    horizon_yr: float

    def index(self, element: str) -> int:
        return self.elements.index(element)

    def predict(self, a_f: float, a_s: float, a_r: float) -> np.ndarray:
        """[X/H] shift, dex, from adding the three sources at these amplitudes."""
        return np.log10(1.0 + a_f * self.F + a_s * self.S + a_r * self.R)

    def to_dict(self, amplitude: float = 1.0) -> dict:
        return {
            "elements": list(self.elements),
            "horizon_yr": self.horizon_yr,
            "amplitude": amplitude,
            "fission_dex": [round(float(v), 4) for v in np.log10(1 + amplitude * self.F)],
            "s_dex": [round(float(v), 4) for v in np.log10(1 + amplitude * self.S)],
            "r_dex": [round(float(v), 4) for v in np.log10(1 + amplitude * self.R)],
            "F": [round(float(v), 5) for v in self.F],
            "S": [round(float(v), 4) for v in self.S],
            "R": [round(float(v), 4) for v in self.R],
        }


def build_templates(elements, *, horizon_yr: float = Y.DEFAULT_HORIZON_YR) -> Templates:
    """Templates over the elements a survey actually delivers (unknowns dropped)."""
    F = Y.fission_pattern(elements, horizon_yr=horizon_yr)
    S = Y.s_pattern(elements)
    R = Y.r_pattern(elements)
    keep = [e for e in elements
            if np.isfinite(F.get(e, np.nan)) and np.isfinite(S.get(e, np.nan))
            and np.isfinite(R.get(e, np.nan))]
    return Templates(
        elements=keep,
        F=np.array([F[e] for e in keep], dtype=float),
        S=np.array([S[e] for e in keep], dtype=float),
        R=np.array([R[e] for e in keep], dtype=float),
        horizon_yr=horizon_yr,
    )


def discriminant_ratios(vec: dict[str, float]) -> dict[str, float]:
    """The named ratios of the brief, from any {element: dex} vector."""
    def d(a, b):
        va, vb = vec.get(a), vec.get(b)
        if va is None or vb is None or not (np.isfinite(va) and np.isfinite(vb)):
            return float("nan")
        return float(va - vb)
    return {"Nd/Ba": d("Nd", "Ba"), "Ce/Ba": d("Ce", "Ba"), "La/Ba": d("La", "Ba"),
            "Mo/Zr": d("Mo", "Zr"), "Ru/Zr": d("Ru", "Zr"), "Eu/Nd": d("Eu", "Nd"),
            "Sr/Nd": d("Sr", "Nd")}


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def amplitude_grid(cfg: PatternConfig, n: int | None = None) -> np.ndarray:
    n = int(n or cfg.n_amp)
    return np.concatenate([[0.0], np.logspace(np.log10(cfg.amp_min), np.log10(cfg.amp_max), n)])


def _weights(obs: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sig) & (sig > 0)
    o = np.where(mask, obs, 0.0)
    w = np.where(mask, 1.0 / np.where(mask, sig, 1.0) ** 2, 0.0)
    return o, w, mask


def _chi2_at(o, w, pattern: np.ndarray, a: np.ndarray, fixed=None) -> np.ndarray:
    """chi2 per star for a per-star amplitude ``a`` (N,) on ``pattern`` (K,)."""
    lin = 1.0 + a[:, None] * pattern[None, :]
    if fixed is not None:
        lin = lin + fixed
    pred = np.log10(np.maximum(lin, 1e-12))
    return ((o - pred) ** 2 * w).sum(axis=1)


def _refine(o, w, pattern: np.ndarray, a: np.ndarray, chi2: np.ndarray, *, fixed=None,
            factor: float = 1.25, n_iter: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic refinement of a per-star amplitude in log space, vectorised.

    The grid is log-spaced at ~20% steps, so the grid minimum is within one
    step of the true one but the fitted amplitude (and, weakly, the chi2) is
    quantised. Three bracketed parabolic steps remove that. Amplitudes at zero
    stay at zero: the refinement only moves stars that already prefer the
    source.
    """
    a = np.asarray(a, dtype=float).copy()
    chi2 = np.asarray(chi2, dtype=float).copy()
    pos = a > 0
    if not pos.any():
        return a, chi2
    f = float(factor)
    for _ in range(int(n_iter)):
        lo = a / f
        hi = a * f
        c_lo = _chi2_at(o, w, pattern, lo, fixed)
        c_hi = _chi2_at(o, w, pattern, hi, fixed)
        # parabola through (log lo, c_lo), (log a, chi2), (log hi, c_hi)
        denom = c_lo - 2.0 * chi2 + c_hi
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = 0.5 * (c_lo - c_hi) / denom      # in units of log(f)
        shift = np.where(np.isfinite(shift) & (denom > 0), np.clip(shift, -1.0, 1.0), 0.0)
        cand = a * f ** shift
        c_cand = _chi2_at(o, w, pattern, cand, fixed)
        # take whichever of the four is lowest, only for stars with a > 0
        stack_c = np.stack([chi2, c_lo, c_hi, c_cand])
        stack_a = np.stack([a, lo, hi, cand])
        k = stack_c.argmin(axis=0)
        new_c = stack_c[k, np.arange(len(a))]
        new_a = stack_a[k, np.arange(len(a))]
        a = np.where(pos, new_a, a)
        chi2 = np.where(pos, new_c, chi2)
        f = f ** 0.5
    return a, chi2


def _scan_1d(o, w, pattern: np.ndarray, grid: np.ndarray, *, refine: bool = True
             ) -> tuple[np.ndarray, np.ndarray]:
    best = np.full(o.shape[0], np.inf)
    best_a = np.zeros(o.shape[0])
    for a in grid:
        pred = np.log10(1.0 + a * pattern)
        chi2 = ((o - pred) ** 2 * w).sum(axis=1)
        better = chi2 < best
        best = np.where(better, chi2, best)
        best_a = np.where(better, a, best_a)
    if refine:
        best_a, best = _refine(o, w, pattern, best_a, best)
    return best, best_a


def _scan_2d(o, w, p1: np.ndarray, p2: np.ndarray, grid: np.ndarray, *, refine: bool = True
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best = np.full(o.shape[0], np.inf)
    best_1 = np.zeros(o.shape[0])
    best_2 = np.zeros(o.shape[0])
    for a1 in grid:
        base = a1 * p1
        for a2 in grid:
            pred = np.log10(1.0 + base + a2 * p2)
            chi2 = ((o - pred) ** 2 * w).sum(axis=1)
            better = chi2 < best
            best = np.where(better, chi2, best)
            best_1 = np.where(better, a1, best_1)
            best_2 = np.where(better, a2, best_2)
    if refine:
        # coordinate-wise: refine a1 holding a2, then a2 holding a1
        for _ in range(2):
            best_1, best = _refine(o, w, p1, best_1, best, fixed=best_2[:, None] * p2[None, :])
            best_2, best = _refine(o, w, p2, best_2, best, fixed=best_1[:, None] * p1[None, :])
    return best, best_1, best_2


def fit_patterns(obs, sig, T: Templates, cfg: PatternConfig) -> pd.DataFrame:
    """Fit the five hypotheses to every row of ``obs`` (N x K, NaN = unmeasured).

    Returns one row per star with the best chi2 and amplitude of each
    hypothesis, the two likelihood ratios and the classification:

    ``fission_lr``  = 0.5 * (chi2_best_natural - chi2_fission): positive means
                      the fission-only fit beats the best of s, r and s+r;
    ``enrich_lr``   = 0.5 * (chi2_null - min(chi2_natural, chi2_fission)):
                      how strongly *anything* is added at all.
    """
    o, w, mask = _weights(obs, sig)
    n_meas = mask.sum(axis=1)
    g1 = amplitude_grid(cfg, cfg.n_amp)
    g2 = amplitude_grid(cfg, cfg.n_amp_2d)

    chi2_null = (o ** 2 * w).sum(axis=1)
    chi2_s, a_s = _scan_1d(o, w, T.S, g1)
    chi2_r, a_r = _scan_1d(o, w, T.R, g1)
    chi2_sr, a_sr_s, a_sr_r = _scan_2d(o, w, T.S, T.R, g2)
    chi2_f, a_f = _scan_1d(o, w, T.F, g1)
    # The 2-D grid is coarser than the 1-D ones, so the pure-s / pure-r fits
    # can beat "s+r"; the natural alternative is the best of the three.
    chi2_nat = np.minimum(chi2_sr, np.minimum(chi2_s, chi2_r))
    chi2_best = np.minimum(chi2_nat, chi2_f)

    fission_lr = 0.5 * (chi2_nat - chi2_f)
    enrich_lr = 0.5 * (chi2_null - chi2_best)

    # Natural sub-class: prefer the one-parameter fits unless s+r is clearly better.
    nat = np.where(chi2_s <= chi2_r, S_PROCESS, R_PROCESS).astype(object)
    both = chi2_sr < np.minimum(chi2_s, chi2_r) - cfg.ambiguity_margin
    nat[both] = S_PLUS_R

    cls = np.full(len(o), NORMAL, dtype=object)
    enriched = enrich_lr >= cfg.enrich_min
    cls[enriched] = nat[enriched]
    cls[enriched & (fission_lr >= cfg.lr_min)] = FISSION
    cls[enriched & (fission_lr > 0) & (fission_lr < cfg.lr_min)] = AMBIGUOUS
    cls[n_meas < cfg.min_elements] = INSUFFICIENT

    return pd.DataFrame({
        "n_measured": n_meas.astype(int),
        "chi2_null": chi2_null,
        "chi2_s": chi2_s, "a_s": a_s,
        "chi2_r": chi2_r, "a_r": a_r,
        "chi2_sr": chi2_sr, "a_sr_s": a_sr_s, "a_sr_r": a_sr_r,
        "chi2_f": chi2_f, "a_f": a_f,
        "chi2_natural": chi2_nat,
        "fission_lr": fission_lr,
        "enrich_lr": enrich_lr,
        "natural_class": nat,
        "classification": cls,
    })


def fission_lr_only(obs, sig, T: Templates, cfg: PatternConfig) -> np.ndarray:
    """Just the fission-vs-natural ratio (the hot loop of LOO and the nulls)."""
    o, w, _ = _weights(obs, sig)
    g1 = amplitude_grid(cfg, cfg.n_amp)
    g2 = amplitude_grid(cfg, cfg.n_amp_2d)
    chi2_s, _ = _scan_1d(o, w, T.S, g1)
    chi2_r, _ = _scan_1d(o, w, T.R, g1)
    chi2_sr, _, _ = _scan_2d(o, w, T.S, T.R, g2)
    chi2_f, _ = _scan_1d(o, w, T.F, g1)
    return 0.5 * (np.minimum(chi2_sr, np.minimum(chi2_s, chi2_r)) - chi2_f)


# ---------------------------------------------------------------------------
# Leave-one-out: is it a pattern or one element?
# ---------------------------------------------------------------------------
def leave_one_out(obs, sig, T: Templates, cfg: PatternConfig) -> pd.DataFrame:
    """``fission_lr`` with each element removed in turn.

    This is the test that separates FALLOUT from a one-element anomaly: if
    dropping any single element destroys the fission preference, the star was
    carried by that element and there is no pattern. Columns ``lr_without_X``
    for every X, plus ``lr_loo_min`` / ``lr_loo_driver``.
    """
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    out = {}
    for k, el in enumerate(T.elements):
        o2 = obs.copy()
        o2[:, k] = np.nan
        out[f"lr_without_{el}"] = fission_lr_only(o2, sig, T, cfg)
    df = pd.DataFrame(out)
    arr = df.to_numpy()
    df["lr_loo_min"] = arr.min(axis=1)
    df["lr_loo_driver"] = [T.elements[i] for i in arr.argmin(axis=1)]
    return df


def lr_without(obs, sig, T: Templates, cfg: PatternConfig, drop: tuple[str, ...]) -> np.ndarray:
    """``fission_lr`` with a named set of elements excluded (e.g. Ba)."""
    o2 = np.asarray(obs, dtype=float).copy()
    for el in drop:
        if el in T.elements:
            o2[:, T.index(el)] = np.nan
    return fission_lr_only(o2, sig, T, cfg)


# ---------------------------------------------------------------------------
# Vectors from a normalised table
# ---------------------------------------------------------------------------
def _flag_is_bad(v: pd.Series) -> np.ndarray:
    if v.dtype == object:
        return ~v.fillna("").astype(str).str.strip().isin(["", "0", "0.0", "nan", "False"]).to_numpy()
    return (pd.to_numeric(v, errors="coerce").fillna(0) != 0).to_numpy()


def assemble_vectors(df: pd.DataFrame, elements: list[str], *, value_prefix: str = "",
                     err_prefix: str = "e_", flag_prefix: str = "f_",
                     cfg: PatternConfig | None = None,
                     fallback_sigma: dict[str, float] | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(obs, sig, flagged) matrices over ``elements`` from a canonical table.

    A flagged measurement is *excluded* (NaN) and counted, never used. The
    error is ``sqrt(reported^2 + floor^2)``; a missing reported error falls
    back to the element's empirical scatter if given, else the default.
    """
    cfg = cfg or PatternConfig()
    n = len(df)
    obs = np.full((n, len(elements)), np.nan)
    sig = np.full((n, len(elements)), np.nan)
    flagged = np.zeros((n, len(elements)), dtype=bool)
    info = {"elements": list(elements), "value_columns": {}, "error_columns": {},
            "flag_columns": {}, "missing": []}
    for k, el in enumerate(elements):
        col = f"{value_prefix}{el}"
        if col not in df.columns:
            info["missing"].append(el)
            continue
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        info["value_columns"][el] = col
        ecol = f"{err_prefix}{el}"
        if ecol in df.columns:
            e = pd.to_numeric(df[ecol], errors="coerce").to_numpy(dtype=float)
            info["error_columns"][el] = ecol
        else:
            e = np.full(n, np.nan)
        fb = (fallback_sigma or {}).get(el, cfg.error_default_dex)
        e = np.where(np.isfinite(e) & (e > 0), e, fb)
        fcol = f"{flag_prefix}{el}"
        if fcol in df.columns:
            bad = _flag_is_bad(df[fcol])
            info["flag_columns"][el] = fcol
            flagged[:, k] = bad & np.isfinite(v)
            v = np.where(bad, np.nan, v)
        obs[:, k] = v
        sig[:, k] = np.sqrt(e ** 2 + cfg.systematic_floor_dex ** 2)
    return obs, sig, flagged, info


# ---------------------------------------------------------------------------
# Peer residuals: abundances relative to a Teff/logg/[Fe/H]-matched surface
# ---------------------------------------------------------------------------
def peer_residuals(df: pd.DataFrame, elements: list[str], *, teff_col: str = "teff",
                   logg_col: str = "logg", feh_col: str = "fe_h", degree: int = 2,
                   clip: float = 4.0, n_iter: int = 4, min_rows: int = 200,
                   ) -> tuple[pd.DataFrame, dict[str, float], dict]:
    """``[X/H]`` residuals against a robust polynomial in ([Fe/H], Teff, logg[, alpha]).

    Reuses TAILINGS' ``fit_element`` (iteratively clipped least squares, so a
    genuine enrichment cannot drag its own reference surface) and its
    leave-one-out alpha proxy when alpha elements are present. Working in
    ``[X/H]`` with ``[Fe/H]`` as a predictor removes the normalisation
    aberration (an error in the star's own [Fe/H] otherwise shifts every
    [X/Fe] together, which is a *coherent* vector and exactly what a pattern
    fit must not be fed).

    Returns the residual table (same index as ``df``), the per-element robust
    scatter of the fit, and a notes dict naming any element that fell back to
    a median because it had too few rows.
    """
    feh = pd.to_numeric(df[feh_col], errors="coerce").to_numpy(dtype=float)
    cols = [feh, pd.to_numeric(df[teff_col], errors="coerce").to_numpy(dtype=float) / 1000.0,
            pd.to_numeric(df[logg_col], errors="coerce").to_numpy(dtype=float)]
    have_alpha = any(a in df.columns for a in M.ALPHA_ELEMENTS)
    resid = {}
    scatter: dict[str, float] = {}
    notes: dict = {"fallback_median": [], "alpha_proxy": have_alpha, "n_fit": {}}
    for el in elements:
        if el not in df.columns:
            continue
        xfe = pd.to_numeric(df[el], errors="coerce").to_numpy(dtype=float)
        y = xfe + feh
        P = list(cols)
        if have_alpha:
            P.append(M.alpha_proxy(df, exclude=el))
        P = np.column_stack(P)
        # Rows with an unmeasured alpha proxy would drop out of the fit; give
        # them the sample median so the residual still exists for them.
        if have_alpha:
            ap = P[:, -1]
            P[:, -1] = np.where(np.isfinite(ap), ap, np.nanmedian(ap) if np.isfinite(ap).any() else 0.0)
        try:
            fit = M.fit_element(P, y, element=el, degree=degree, clip=clip, n_iter=n_iter,
                                min_rows=min_rows)
            r = y - fit.predict(P)
            scatter[el] = float(fit.robust_scatter)
            notes["n_fit"][el] = int(fit.n_fit)
        except ValueError:
            r = y - np.nanmedian(y)
            scatter[el] = float(M.robust_sigma(r[np.isfinite(r)])) if np.isfinite(r).any() else float("nan")
            notes["fallback_median"].append(el)
        resid[el] = r
    if not resid:
        return pd.DataFrame(index=df.index), scatter, notes
    return pd.DataFrame(resid, index=df.index), scatter, notes


def raw_vectors(df: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    """``[X/Fe]`` minus the sample median: no Teff/logg regression at all.

    The raw vector is scored alongside the peer residual so the run can say how
    many raw-space "patterns" the peer regression removed -- the
    ``teff_peer_residual`` counter.
    """
    out = {}
    for el in elements:
        if el in df.columns:
            v = pd.to_numeric(df[el], errors="coerce").to_numpy(dtype=float)
            out[el] = v - (np.nanmedian(v) if np.isfinite(v).any() else 0.0)
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Vetoes
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def _finite_gt(a: np.ndarray, x: float) -> np.ndarray:
    return np.isfinite(a) & (a > x)


def _finite_lt(a: np.ndarray, x: float) -> np.ndarray:
    return np.isfinite(a) & (a < x)


def apply_vetoes(cand: pd.DataFrame, *, cfg: PatternConfig, lr_threshold: float,
                 flagged_core: np.ndarray | None = None) -> pd.DataFrame:
    """Add ``veto_<name>`` columns, ``first_veto`` and ``vet_pass`` to a candidate table.

    ``cand`` must carry the raw ``[X/Fe]`` columns (``Ba``, ``Nd``, ...), the
    fit columns from :func:`fit_patterns`, ``lr_noba`` (from :func:`lr_without`),
    ``lr_loo_min`` (from :func:`leave_one_out`) and ``fission_lr_raw`` (the same
    statistic on the raw vector). Missing columns disable the veto that needs
    them -- disabled, and said so, rather than silently passed.

    Veto semantics:

    * ``low_snr_or_flagged``   flag_sp != 0, SNR below the floor, too few
      elements, or a flagged *core* element (a heavy-peak anchor that the
      pipeline itself distrusts).
    * ``s_process_star``       the Ba/CH/CEMP-s morphology: [Ba/Fe] high with
      [Nd/Ba] < 0, or the whole s-process (Y, Zr and Ba) up together; or a
      binary flag if the catalogue has one.
    * ``r_process_star``       [Eu/Fe] > 0.3 with [Eu/Nd] > 0.
    * ``young_ba_enhancement`` Ba is over-estimated in young/active dwarfs:
      Li-rich (A(Li) above the threshold) or age below the cap, with Ba up.
    * ``nlte_saturated_lines`` the pattern must hold with Ba excluded (the
      Ba II 5853/6141/6496 lines saturate in metal-rich cool dwarfs);
      ``lr_noba`` is reported next to ``fission_lr`` for every candidate.
    * ``single_element_driver`` removing any ONE element must not destroy the
      preference: ``lr_loo_min`` >= threshold.
    * ``teff_peer_residual``    the preference must also be visible in the raw
      vector at a fraction of the threshold; a pattern that exists only after
      the regression was made by the regression.
    """
    out = cand.copy()
    n = len(out)
    thr = float(lr_threshold)

    ba = _col(out, "Ba")
    nd = _col(out, "Nd")
    eu = _col(out, "Eu")
    y = _col(out, "Y")
    zr = _col(out, "Zr")
    li = _col(out, "Li")
    feh = _col(out, "fe_h")
    snr = _col(out, "snr")
    flag_sp = _col(out, "flag_sp")
    age = _col(out, "age")
    binary = _col(out, "binary_flag")
    n_meas = _col(out, "n_measured")

    # 1. quality
    v = np.zeros(n, dtype=bool)
    v |= _finite_gt(np.abs(flag_sp), 0.0)
    v |= _finite_lt(snr, cfg.snr_min)
    v |= _finite_lt(n_meas, cfg.min_elements)
    if flagged_core is not None:
        v |= np.asarray(flagged_core, dtype=bool)
    out["veto_low_snr_or_flagged"] = v

    # 2. s-process star
    nd_ba = nd - ba
    v = _finite_gt(ba, cfg.s_ba_fe_min) & (_finite_lt(nd_ba, 0.0)
                                            | (_finite_gt(y, cfg.s_light_fe_min)
                                               & _finite_gt(zr, cfg.s_light_fe_min)))
    v |= _finite_gt(np.abs(binary), 0.0)
    out["veto_s_process_star"] = v

    # 3. r-process star
    out["veto_r_process_star"] = _finite_gt(eu, cfg.r_eu_fe_min) & _finite_gt(eu - nd, 0.0)

    # 4. young Ba enhancement. A(Li) = [Li/Fe] + [Fe/H] + A(Li)_sun.
    ali = li + np.where(np.isfinite(feh), feh, 0.0) + Y.SOLAR_LOGEPS["Li"]
    young = _finite_gt(ali, cfg.young_ali_min) | _finite_lt(age, cfg.young_age_max_gyr)
    out["veto_young_ba_enhancement"] = young & _finite_gt(ba, cfg.young_ba_fe_min)

    # 5. NLTE / saturation: hold without Ba
    lr_noba = _col(out, "lr_noba")
    out["veto_nlte_saturated_lines"] = _finite_lt(lr_noba, thr) | ~np.isfinite(lr_noba)

    # 6. single-element driver
    loo = _col(out, "lr_loo_min")
    out["veto_single_element_driver"] = _finite_lt(loo, thr) | ~np.isfinite(loo)

    # 7. raw-space consistency
    raw = _col(out, "fission_lr_raw")
    if np.isfinite(raw).any():
        out["veto_teff_peer_residual"] = _finite_lt(raw, cfg.raw_consistency_fraction * thr)
    else:
        out["veto_teff_peer_residual"] = np.zeros(n, dtype=bool)

    first = np.full(n, "", dtype=object)
    for name in VETOES:
        col = out[f"veto_{name}"].to_numpy(dtype=bool)
        first = np.where((first == "") & col, name, first)
    out["first_veto"] = first
    out["vet_pass"] = first == ""
    out["veto_reasons"] = [";".join(nm for nm in VETOES if row[f"veto_{nm}"])
                           for _, row in out.iterrows()] if n else []
    return out


def veto_counters(vetted: pd.DataFrame) -> dict[str, int]:
    """How many candidates each veto removed (independently) plus first-veto funnel."""
    out = {}
    for name in VETOES:
        col = f"veto_{name}"
        out[name] = int(vetted[col].sum()) if col in vetted.columns else 0
    out["first_veto"] = {name: int((vetted["first_veto"] == name).sum()) for name in VETOES} \
        if "first_veto" in vetted.columns else {}
    out["n_pass"] = int(vetted["vet_pass"].sum()) if "vet_pass" in vetted.columns else 0
    return out


# ---------------------------------------------------------------------------
# Nulls and sensitivity
# ---------------------------------------------------------------------------
def shuffled_null(obs, sig, T: Templates, cfg: PatternConfig, *, n_perm: int = 3,
                  max_rows: int = 20000, rng: np.random.Generator | None = None) -> dict:
    """Permute the element labels within each star and re-score.

    Each star's (value, error) pairs are shuffled across its element slots, so
    the amplitude structure and the per-element noise are kept and only the
    *alignment with the fission shape* is destroyed. The distribution of
    ``fission_lr`` under that shuffle is how often noise-plus-natural-spread
    makes the vector by accident; its quantile sets the threshold.
    """
    rng = rng or np.random.default_rng(20260906)
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    n = obs.shape[0]
    if n == 0:
        return {"n_rows": 0, "n_perm": 0, "quantiles": {}, "lr": np.array([])}
    idx = np.arange(n) if n <= max_rows else rng.choice(n, size=max_rows, replace=False)
    o = obs[idx]
    s = sig[idx]
    lrs = []
    for _ in range(int(n_perm)):
        keys = rng.random(o.shape)
        order = np.argsort(keys, axis=1)
        lrs.append(fission_lr_only(np.take_along_axis(o, order, 1),
                                   np.take_along_axis(s, order, 1), T, cfg))
    lr = np.concatenate(lrs) if lrs else np.array([])
    qs = (0.5, 0.9, 0.99, 0.999, 0.9999)
    quant = {f"q{q}": float(np.quantile(lr, q)) for q in qs if lr.size} if lr.size else {}
    return {"n_rows": int(len(idx)), "n_perm": int(n_perm), "quantiles": quant, "lr": lr,
            "frac_above_lr_min": float((lr >= cfg.lr_min).mean()) if lr.size else float("nan")}


def sample_null(fission_lr: np.ndarray) -> dict:
    """Quantiles of the statistic over the whole (un-shuffled) sample."""
    lr = np.asarray(fission_lr, dtype=float)
    lr = lr[np.isfinite(lr)]
    if lr.size == 0:
        return {"n": 0, "quantiles": {}}
    qs = (0.5, 0.9, 0.99, 0.999, 0.9999)
    return {"n": int(lr.size), "quantiles": {f"q{q}": float(np.quantile(lr, q)) for q in qs},
            "max": float(lr.max())}


def derive_threshold(cfg: PatternConfig, shuffled: dict) -> tuple[float, str]:
    """The working threshold: the config floor, raised to the shuffled-null quantile."""
    q = (shuffled.get("quantiles") or {}).get(f"q{cfg.null_quantile}")
    if q is None or not np.isfinite(q):
        return float(cfg.lr_min), "config lr_min (no shuffled null available)"
    if q > cfg.lr_min:
        return float(q), f"shuffled-null q{cfg.null_quantile} = {q:.2f} exceeds config lr_min"
    return float(cfg.lr_min), f"config lr_min (shuffled-null q{cfg.null_quantile} = {q:.2f} is below it)"


def sensitivity_curve(obs, sig, T: Templates, cfg: PatternConfig, *, lr_threshold: float,
                      amplitudes=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0), n_inject: int = 1500,
                      rng: np.random.Generator | None = None, with_loo: bool = True) -> list[dict]:
    """Inject the fission pattern at each amplitude into real vectors and re-score.

    Reports, per amplitude, the Nd shift in dex and the fraction of injected
    stars that (a) pass the LR threshold and (b) also pass the single-element
    driver test. This is the channel's sensitivity statement: the enrichment
    level at which the *vector* becomes detectable given the survey's errors.
    """
    rng = rng or np.random.default_rng(7)
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    n = obs.shape[0]
    if n == 0:
        return []
    idx = np.arange(n) if n <= n_inject else rng.choice(n, size=n_inject, replace=False)
    base = obs[idx]
    s = sig[idx]
    rows = []
    for a in amplitudes:
        shift = np.log10(1.0 + a * T.F)
        o = base + shift[None, :]
        lr = fission_lr_only(o, s, T, cfg)
        rec = {"a_f": float(a),
               "nd_dex": round(float(shift[T.index("Nd")]) if "Nd" in T.elements else float("nan"), 3),
               "n": int(len(idx)),
               "frac_lr_pass": float((lr >= lr_threshold).mean()),
               "median_lr": float(np.median(lr))}
        if with_loo:
            loo = leave_one_out(o, s, T, cfg)
            rec["frac_lr_and_loo_pass"] = float(((lr >= lr_threshold)
                                                 & (loo["lr_loo_min"].to_numpy() >= lr_threshold)).mean())
        rows.append(rec)
    return rows


__all__ = [
    "AMBIGUOUS", "FISSION", "INSUFFICIENT", "NORMAL", "R_PROCESS", "S_PLUS_R", "S_PROCESS",
    "VETOES", "PatternConfig", "Templates", "amplitude_grid", "apply_vetoes",
    "assemble_vectors", "build_templates", "derive_threshold", "discriminant_ratios",
    "fission_lr_only", "fit_patterns", "leave_one_out", "lr_without", "peer_residuals",
    "raw_vectors", "sample_null", "sensitivity_curve", "shuffled_null", "veto_counters",
]

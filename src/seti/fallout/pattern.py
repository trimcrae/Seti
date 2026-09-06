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

What the first real GALAH DR4 run taught (2026-09-06, results/fallout/)
----------------------------------------------------------------------
* **Quoted errors are not the scatter.** The shuffled-element null put its
  99.9th percentile at ln LR 17 (dwarfs) against 2.3 on a synthetic population
  built from the quoted errors, because GALAH's per-element errors understate
  the measured peer-residual scatter by factors of 2-4 (Ce 0.28 dex, Rb 0.40,
  Eu 0.32). The error is therefore **floored at the measured peer scatter of
  that element in that sample** (:func:`error_floors`), the same rescaling
  LOOM applies through sqrt(reduced chi2); the floors and their ratio to the
  quoted errors are recorded.
* **Winning is not fitting.** Both giant "survivors" beat the natural models
  with chi2_f = 230 on 9 elements. A star whose *best* model has reduced chi2
  above ``max_reduced_chi2`` is ``UNEXPLAINED_BY_ALL_TEMPLATES``: counted and
  listed separately, never a fission candidate.
* **One hot element must not carry a pattern.** Both were La-driven (peer
  La +1.2 dex against a template prediction of +0.3). ``heavy_peak_incoherent``
  requires at least two of La/Ce/Nd individually >= 2 sigma in the fission
  direction; ``la_cn_blend`` distrusts La in cool giants where the La II lines
  sit in CN-blended regions, once the La residual is shown to track C/N or Teff
  (:func:`la_diagnostics`).

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
UNEXPLAINED = "UNEXPLAINED_BY_ALL_TEMPLATES"

#: Every veto, in the order they are applied. Each is a named counter in
#: ``summary.json``; ``first_veto`` records which one a star hit first.
VETOES: tuple[str, ...] = (
    "low_snr_or_flagged",
    "unexplained_by_all_templates",
    "s_process_star",
    "r_process_star",
    "young_ba_enhancement",
    "nlte_saturated_lines",
    "single_element_driver",
    "heavy_peak_incoherent",
    "la_cn_blend",
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
    # error model: sigma = sqrt(max(reported, floor)^2 + sys^2); reported missing -> default
    systematic_floor_dex: float = 0.05
    error_default_dex: float = 0.15
    error_floor_mode: str = "peer_scatter"     # "peer_scatter" | "none"
    min_elements: int = 5
    # decision thresholds (ln likelihood ratios; Delta chi2 = 2 * lr)
    lr_min: float = 8.0
    enrich_min: float = 12.5
    ambiguity_margin: float = 2.0
    # absolute goodness of fit: the BEST model must fit this well or the star is unexplained
    max_reduced_chi2: float = 3.0
    # the heavy-peak anchors that must remain when Ba is excluded
    ba_element: str = "Ba"
    core_elements: tuple[str, ...] = ("Y", "Zr", "Ba", "La", "Ce", "Nd", "Sm", "Eu")
    # heavy-peak coherence: at least this many of these individually >= this many sigma up
    heavy_peak_elements: tuple[str, ...] = ("La", "Ce", "Nd")
    heavy_peak_min_coherent: int = 2
    heavy_peak_sigma: float = 2.0
    # "testable": enough heavy-peak elements measured for the pattern to be decidable
    min_heavy_measured: int = 2
    # La in cool giants: CN blends on the La II lines
    la_cn_teff_max: float = 4800.0
    la_cn_corr_min: float = 0.2
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
        denom = c_lo - 2.0 * chi2 + c_hi
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = 0.5 * (c_lo - c_hi) / denom      # in units of log(f)
        shift = np.where(np.isfinite(shift) & (denom > 0), np.clip(shift, -1.0, 1.0), 0.0)
        cand = a * f ** shift
        c_cand = _chi2_at(o, w, pattern, cand, fixed)
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
        for _ in range(2):
            best_1, best = _refine(o, w, p1, best_1, best, fixed=best_2[:, None] * p2[None, :])
            best_2, best = _refine(o, w, p2, best_2, best, fixed=best_1[:, None] * p1[None, :])
    return best, best_1, best_2


def fit_patterns(obs, sig, T: Templates, cfg: PatternConfig) -> pd.DataFrame:
    """Fit the five hypotheses to every row of ``obs`` (N x K, NaN = unmeasured).

    Returns one row per star with the best chi2 and amplitude of each
    hypothesis, the two likelihood ratios, the absolute goodness of fit and the
    classification:

    ``fission_lr``  = 0.5 * (chi2_best_natural - chi2_fission): positive means
                      the fission-only fit beats the best of s, r and s+r;
    ``enrich_lr``   = 0.5 * (chi2_null - min(chi2_natural, chi2_fission)):
                      how strongly *anything* is added at all;
    ``reduced_chi2_best`` = chi2 of the best model per degree of freedom.
                      Above ``max_reduced_chi2`` the star is
                      ``UNEXPLAINED_BY_ALL_TEMPLATES`` whatever the ratios say.
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
    chi2_nat = np.minimum(chi2_sr, np.minimum(chi2_s, chi2_r))
    chi2_best = np.minimum(chi2_nat, chi2_f)

    fission_lr = 0.5 * (chi2_nat - chi2_f)
    enrich_lr = 0.5 * (chi2_null - chi2_best)
    dof = np.maximum(n_meas - 1, 1)
    red = chi2_best / dof

    nat = np.where(chi2_s <= chi2_r, S_PROCESS, R_PROCESS).astype(object)
    both = chi2_sr < np.minimum(chi2_s, chi2_r) - cfg.ambiguity_margin
    nat[both] = S_PLUS_R

    cls = np.full(len(o), NORMAL, dtype=object)
    enriched = enrich_lr >= cfg.enrich_min
    cls[enriched] = nat[enriched]
    cls[enriched & (fission_lr >= cfg.lr_min)] = FISSION
    cls[enriched & (fission_lr > 0) & (fission_lr < cfg.lr_min)] = AMBIGUOUS
    # A star nothing fits is not a candidate for anything: "winning" against
    # worse models is not the same as fitting.
    cls[enriched & (red > cfg.max_reduced_chi2)] = UNEXPLAINED
    cls[n_meas < cfg.min_elements] = INSUFFICIENT

    return pd.DataFrame({
        "n_measured": n_meas.astype(int),
        "chi2_null": chi2_null,
        "chi2_s": chi2_s, "a_s": a_s,
        "chi2_r": chi2_r, "a_r": a_r,
        "chi2_sr": chi2_sr, "a_sr_s": a_sr_s, "a_sr_r": a_sr_r,
        "chi2_f": chi2_f, "a_f": a_f,
        "chi2_natural": chi2_nat,
        "chi2_best": chi2_best,
        "reduced_chi2_best": red,
        "reduced_chi2_f": chi2_f / dof,
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
# Error model and vectors from a normalised table
# ---------------------------------------------------------------------------
def _flag_is_bad(v: pd.Series) -> np.ndarray:
    if v.dtype == object:
        return ~v.fillna("").astype(str).str.strip().isin(["", "0", "0.0", "nan", "False"]).to_numpy()
    return (pd.to_numeric(v, errors="coerce").fillna(0) != 0).to_numpy()


def error_floors(df: pd.DataFrame, elements: list[str], scatter: dict[str, float], *,
                 err_prefix: str = "e_", cfg: PatternConfig | None = None) -> dict[str, dict]:
    """Per-element error floor = the measured peer-residual scatter of that sample.

    Catalogue errors are formal fit errors; on the first real run they
    understated the residual scatter by 2-4x and the shuffled null's tail
    stretched to ln LR 17. Flooring every element's error at the robust width
    of its own peer residual (in this sample) is the same correction LOOM makes
    by rescaling sigma with sqrt(reduced chi2), applied per element. Returns
    ``{el: {floor, median_quoted, inflation}}`` so the correction is on the
    record.
    """
    cfg = cfg or PatternConfig()
    out: dict[str, dict] = {}
    for el in elements:
        ecol = f"{err_prefix}{el}"
        med = float("nan")
        if ecol in df.columns:
            e = pd.to_numeric(df[ecol], errors="coerce").to_numpy(dtype=float)
            e = e[np.isfinite(e) & (e > 0)]
            if e.size:
                med = float(np.median(e))
        sc = scatter.get(el)
        floor = float(sc) if (cfg.error_floor_mode == "peer_scatter" and sc is not None
                              and np.isfinite(sc) and sc > 0) else 0.0
        out[el] = {"floor_dex": round(floor, 4), "median_quoted_dex": round(med, 4) if np.isfinite(med) else None,
                   "inflation": round(floor / med, 2) if (np.isfinite(med) and med > 0 and floor > 0) else None}
    return out


def assemble_vectors(df: pd.DataFrame, elements: list[str], *, value_prefix: str = "",
                     err_prefix: str = "e_", flag_prefix: str = "f_",
                     cfg: PatternConfig | None = None,
                     fallback_sigma: dict[str, float] | None = None,
                     sigma_floor: dict[str, float] | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(obs, sig, flagged) matrices over ``elements`` from a canonical table.

    A flagged measurement is *excluded* (NaN) and counted, never used. The
    error is ``sqrt(max(reported, floor)^2 + systematic^2)``; a missing reported
    error falls back to the element's empirical scatter if given, else the
    default. ``sigma_floor`` is the per-element floor from :func:`error_floors`.
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
        fl = (sigma_floor or {}).get(el)
        if fl is not None and np.isfinite(fl) and fl > 0:
            e = np.maximum(e, float(fl))
        fcol = f"{flag_prefix}{el}"
        if fcol in df.columns:
            bad = _flag_is_bad(df[fcol])
            info["flag_columns"][el] = fcol
            flagged[:, k] = bad & np.isfinite(v)
            v = np.where(bad, np.nan, v)
        obs[:, k] = v
        sig[:, k] = np.sqrt(e ** 2 + cfg.systematic_floor_dex ** 2)
    return obs, sig, flagged, info


def testable_mask(obs, T: Templates, cfg: PatternConfig) -> np.ndarray:
    """Stars on which the pattern is *decidable*: enough elements, enough heavy peak.

    On the first real run 79,690 of 101,928 dwarfs were INSUFFICIENT (Rb, Sr,
    Ru, Eu are rarely measured in dwarfs), and an injection into them cannot
    succeed. Completeness is reported on this mask, with the testable fraction
    stated next to it.
    """
    obs = np.asarray(obs, dtype=float)
    meas = np.isfinite(obs)
    n_meas = meas.sum(axis=1)
    heavy = [T.index(e) for e in cfg.heavy_peak_elements if e in T.elements]
    n_heavy = meas[:, heavy].sum(axis=1) if heavy else np.zeros(len(obs), dtype=int)
    return (n_meas >= cfg.min_elements) & (n_heavy >= cfg.min_heavy_measured)


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


def la_diagnostics(frame: pd.DataFrame, *, residual_col: str = "peer_La",
                   covariates: tuple[str, ...] = ("teff", "logg", "C", "N", "vsini"),
                   cfg: PatternConfig | None = None, min_rows: int = 200) -> dict:
    """Does the La residual track what a CN blend would track?

    Spearman correlations of the La peer residual against Teff, log g, [C/Fe],
    [N/Fe] and vsini over the sample. ``la_cn_suspect`` is true when the
    residual correlates with C or N above ``la_cn_corr_min``, or
    anti-correlates with Teff by that much (blends strengthen toward cooler
    atmospheres). When the diagnostic cannot be computed the verdict is
    ``suspect = True`` with ``reason = "not computable"`` -- La in a cool giant
    is distrusted until shown clean, not the other way round.
    """
    cfg = cfg or PatternConfig()
    out: dict = {"residual": residual_col, "n": 0, "correlations": {}, "la_cn_suspect": True,
                 "reason": "not computable", "covariates_absent": []}
    if residual_col not in frame.columns:
        out["reason"] = f"{residual_col} absent"
        return out
    r = pd.to_numeric(frame[residual_col], errors="coerce")
    out["n"] = int(r.notna().sum())
    if out["n"] < min_rows:
        out["reason"] = f"only {out['n']} La residuals (need {min_rows})"
        return out
    trig = []
    for c in covariates:
        if c not in frame.columns:
            out["covariates_absent"].append(c)
            continue
        x = pd.to_numeric(frame[c], errors="coerce")
        ok = r.notna() & x.notna()
        if ok.sum() < min_rows:
            out["correlations"][c] = None
            continue
        rho = float(r[ok].corr(x[ok], method="spearman"))
        out["correlations"][c] = round(rho, 4) if np.isfinite(rho) else None
        if c in ("C", "N") and np.isfinite(rho) and abs(rho) >= cfg.la_cn_corr_min:
            trig.append(f"{c} rho={rho:+.2f}")
        if c == "teff" and np.isfinite(rho) and rho <= -cfg.la_cn_corr_min:
            trig.append(f"teff rho={rho:+.2f}")
    computed = [c for c, v in out["correlations"].items() if v is not None]
    if not computed:
        out["reason"] = "no covariate had enough rows"
        return out
    out["la_cn_suspect"] = bool(trig)
    out["reason"] = ("La residual tracks " + ", ".join(trig)) if trig else \
        f"no correlation above {cfg.la_cn_corr_min} with {', '.join(computed)}"
    return out


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


def heavy_peak_z(cand: pd.DataFrame, cfg: PatternConfig) -> pd.DataFrame:
    """Per-element z = peer residual / sigma for the heavy-peak elements.

    Uses ``z_<El>`` if present, else ``peer_<El> / sig_<El>``. Elements without
    both are NaN and do not count toward coherence.
    """
    out = {}
    for el in cfg.heavy_peak_elements:
        if f"z_{el}" in cand.columns:
            out[el] = _col(cand, f"z_{el}")
        else:
            p = _col(cand, f"peer_{el}")
            s = _col(cand, f"sig_{el}")
            with np.errstate(divide="ignore", invalid="ignore"):
                out[el] = np.where(np.isfinite(p) & np.isfinite(s) & (s > 0), p / s, np.nan)
    return pd.DataFrame(out, index=cand.index)


def apply_vetoes(cand: pd.DataFrame, *, cfg: PatternConfig, lr_threshold: float,
                 flagged_core: np.ndarray | None = None,
                 la_cn_suspect: bool = True) -> pd.DataFrame:
    """Add ``veto_<name>`` columns, ``first_veto`` and ``vet_pass`` to a candidate table.

    ``cand`` must carry the raw ``[X/Fe]`` columns (``Ba``, ``Nd``, ...), the
    fit columns from :func:`fit_patterns`, ``lr_noba`` (from :func:`lr_without`),
    ``lr_loo_min`` / ``lr_without_La`` / ``lr_loo_driver`` (from
    :func:`leave_one_out`), ``fission_lr_raw`` (the same statistic on the raw
    vector) and, for the coherence veto, ``peer_<El>`` with ``sig_<El>`` (or
    ``z_<El>``). Missing columns disable the veto that needs them -- disabled,
    and said so, rather than silently passed.

    Veto semantics:

    * ``low_snr_or_flagged``   flag_sp != 0, SNR below the floor, too few
      elements, or a flagged *core* element.
    * ``unexplained_by_all_templates``  the BEST model's reduced chi2 exceeds
      ``max_reduced_chi2``: nothing fits, so nothing is preferred.
    * ``s_process_star``       Ba/CH/CEMP-s morphology, or a binary flag.
    * ``r_process_star``       [Eu/Fe] > 0.3 with [Eu/Nd] > 0.
    * ``young_ba_enhancement`` Li-rich or young with Ba up.
    * ``nlte_saturated_lines`` the pattern must hold with Ba excluded.
    * ``single_element_driver`` removing any ONE element must not destroy the
      preference: ``lr_loo_min`` >= threshold.
    * ``heavy_peak_incoherent`` fewer than ``heavy_peak_min_coherent`` of
      La/Ce/Nd individually >= ``heavy_peak_sigma`` in the fission direction.
    * ``la_cn_blend``          a cool giant (Teff < ``la_cn_teff_max``) whose
      preference rests on La, in a sample where the La residual tracks C/N or
      Teff (or where that could not be checked).
    * ``teff_peer_residual``    the preference must also be visible in the raw
      vector at a fraction of the threshold.
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
    teff = _col(out, "teff")

    # 1. quality
    v = np.zeros(n, dtype=bool)
    v |= _finite_gt(np.abs(flag_sp), 0.0)
    v |= _finite_lt(snr, cfg.snr_min)
    v |= _finite_lt(n_meas, cfg.min_elements)
    if flagged_core is not None:
        v |= np.asarray(flagged_core, dtype=bool)
    out["veto_low_snr_or_flagged"] = v

    # 2. absolute goodness of fit
    red = _col(out, "reduced_chi2_best")
    if not np.isfinite(red).any():
        best = np.fmin(_col(out, "chi2_f"), _col(out, "chi2_natural"))
        red = best / np.maximum(n_meas - 1, 1)
    out["reduced_chi2_best"] = red
    out["veto_unexplained_by_all_templates"] = _finite_gt(red, cfg.max_reduced_chi2)

    # 3. s-process star
    nd_ba = nd - ba
    v = _finite_gt(ba, cfg.s_ba_fe_min) & (_finite_lt(nd_ba, 0.0)
                                            | (_finite_gt(y, cfg.s_light_fe_min)
                                               & _finite_gt(zr, cfg.s_light_fe_min)))
    v |= _finite_gt(np.abs(binary), 0.0)
    out["veto_s_process_star"] = v

    # 4. r-process star
    out["veto_r_process_star"] = _finite_gt(eu, cfg.r_eu_fe_min) & _finite_gt(eu - nd, 0.0)

    # 5. young Ba enhancement. A(Li) = [Li/Fe] + [Fe/H] + A(Li)_sun.
    ali = li + np.where(np.isfinite(feh), feh, 0.0) + Y.SOLAR_LOGEPS["Li"]
    young = _finite_gt(ali, cfg.young_ali_min) | _finite_lt(age, cfg.young_age_max_gyr)
    out["veto_young_ba_enhancement"] = young & _finite_gt(ba, cfg.young_ba_fe_min)

    # 6. NLTE / saturation: hold without Ba
    lr_noba = _col(out, "lr_noba")
    out["veto_nlte_saturated_lines"] = _finite_lt(lr_noba, thr) | ~np.isfinite(lr_noba)

    # 7. single-element driver
    loo = _col(out, "lr_loo_min")
    out["veto_single_element_driver"] = _finite_lt(loo, thr) | ~np.isfinite(loo)

    # 8. heavy-peak coherence
    z = heavy_peak_z(out, cfg)
    zarr = z.to_numpy(dtype=float) if len(z.columns) else np.full((n, 0), np.nan)
    n_coh = (np.isfinite(zarr) & (zarr >= cfg.heavy_peak_sigma)).sum(axis=1)
    any_z = np.isfinite(zarr).any(axis=1) if zarr.shape[1] else np.zeros(n, dtype=bool)
    out["n_heavy_coherent"] = n_coh
    # no z at all -> cannot be shown coherent -> vetoed (distrust, not silence)
    out["veto_heavy_peak_incoherent"] = (n_coh < cfg.heavy_peak_min_coherent) | ~any_z

    # 9. La in cool giants
    sample = out["sample"].astype(str).to_numpy() if "sample" in out.columns else np.full(n, "", dtype=object)
    lr_wo_la = _col(out, "lr_without_La")
    driver = out["lr_loo_driver"].astype(str).to_numpy() if "lr_loo_driver" in out.columns \
        else np.full(n, "", dtype=object)
    la_carries = _finite_lt(lr_wo_la, thr) | (driver == "La")
    z_la = _col(z, "La") if "La" in z.columns else np.full(n, np.nan)
    la_carries |= np.isfinite(z_la) & (z_la >= cfg.heavy_peak_sigma) & (n_coh <= 1)
    is_cool_giant = (sample == "giant") & _finite_lt(teff, cfg.la_cn_teff_max)
    out["veto_la_cn_blend"] = bool(la_cn_suspect) & is_cool_giant & la_carries

    # 10. raw-space consistency
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


def _inject_and_score(base, s, T, cfg, a, lr_threshold, with_loo):
    shift = np.log10(1.0 + a * T.F)
    o = base + shift[None, :]
    lr = fission_lr_only(o, s, T, cfg)
    passed = lr >= lr_threshold
    rec = {"frac_lr_pass": float(passed.mean()) if len(lr) else float("nan"),
           "median_lr": float(np.median(lr)) if len(lr) else float("nan")}
    if with_loo and len(lr):
        loo = leave_one_out(o, s, T, cfg)
        rec["frac_lr_and_loo_pass"] = float((passed & (loo["lr_loo_min"].to_numpy() >= lr_threshold)).mean())
    return rec, shift


def sensitivity_curve(obs, sig, T: Templates, cfg: PatternConfig, *, lr_threshold: float,
                      amplitudes=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0), n_inject: int = 1500,
                      rng: np.random.Generator | None = None, with_loo: bool = True,
                      testable: np.ndarray | None = None) -> list[dict]:
    """Inject the fission pattern at each amplitude into real vectors and re-score.

    Two completeness numbers per amplitude: over **testable** stars
    (:func:`testable_mask` -- enough elements and enough heavy peak measured
    for the pattern to be decidable) and over **all** stars. The first is the
    channel's sensitivity; the second is what the raw survey delivers once the
    untestable fraction is folded in. The testable fraction is reported with
    the curve so neither number can be read without the other.
    """
    rng = rng or np.random.default_rng(7)
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    n = obs.shape[0]
    if n == 0:
        return []
    if testable is None:
        testable = testable_mask(obs, T, cfg)
    testable = np.asarray(testable, dtype=bool)
    t_idx = np.flatnonzero(testable)
    idx_all = np.arange(n) if n <= n_inject else rng.choice(n, size=n_inject, replace=False)
    idx_t = t_idx if t_idx.size <= n_inject else rng.choice(t_idx, size=n_inject, replace=False)
    rows = []
    for a in amplitudes:
        rec_all, shift = _inject_and_score(obs[idx_all], sig[idx_all], T, cfg, a, lr_threshold, with_loo)
        if idx_t.size:
            rec_t, _ = _inject_and_score(obs[idx_t], sig[idx_t], T, cfg, a, lr_threshold, with_loo)
        else:
            rec_t = {"frac_lr_pass": float("nan"), "median_lr": float("nan"),
                     "frac_lr_and_loo_pass": float("nan")}
        rows.append({
            "a_f": float(a),
            "nd_dex": round(float(shift[T.index("Nd")]) if "Nd" in T.elements else float("nan"), 3),
            "n_testable": int(idx_t.size),
            "testable_fraction": float(testable.mean()),
            "frac_lr_pass_testable": rec_t["frac_lr_pass"],
            "frac_lr_and_loo_pass_testable": rec_t.get("frac_lr_and_loo_pass", float("nan")),
            "median_lr_testable": rec_t["median_lr"],
            "n_all": int(idx_all.size),
            "frac_lr_pass_all": rec_all["frac_lr_pass"],
            "frac_lr_and_loo_pass_all": rec_all.get("frac_lr_and_loo_pass", float("nan")),
            # backward-compatible names = the all-star numbers
            "frac_lr_pass": rec_all["frac_lr_pass"],
            "frac_lr_and_loo_pass": rec_all.get("frac_lr_and_loo_pass", float("nan")),
            "median_lr": rec_all["median_lr"],
        })
    return rows


__all__ = [
    "AMBIGUOUS", "FISSION", "INSUFFICIENT", "NORMAL", "R_PROCESS", "S_PLUS_R", "S_PROCESS",
    "UNEXPLAINED", "VETOES", "PatternConfig", "Templates", "amplitude_grid", "apply_vetoes",
    "assemble_vectors", "build_templates", "derive_threshold", "discriminant_ratios",
    "error_floors", "fission_lr_only", "fit_patterns", "heavy_peak_z", "la_diagnostics",
    "leave_one_out", "lr_without", "peer_residuals", "raw_vectors", "sample_null",
    "sensitivity_curve", "shuffled_null", "testable_mask", "veto_counters",
]

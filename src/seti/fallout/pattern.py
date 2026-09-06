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
  fission makes almost none of it;
* and where a catalogue carries them, the DECISIVE elements: ``[Pb/Nd]``
  (the s-process makes Pb, fission never does), ``[Ag/Nd]`` and ``[Pd/Nd]``
  (the fission valley, ~1000x below the peaks, where the r-process is not
  suppressed).

So each star is fitted five ways -- nothing added; pure s; pure r; s + r;
pure fission -- with the amplitude of each source a free non-negative
parameter, and the statistic is the log-likelihood ratio of the fission-only
fit against the **best natural** fit. The natural alternative has two free
amplitudes against fission's one, so the comparison is biased *against*
fission; that is the conservative direction.

Upper limits are **censored data, not missing data**. A Pb upper limit far
below what the s-process fit predicts is evidence *for* fission-not-s and
enters the likelihood as ``-2 ln Phi((limit - prediction) / sigma)``; a
detection enters as the usual chi2 term. Dropping limits would throw away the
one measurement that decides the question.

What the first real GALAH DR4 run taught (2026-09-06, results/fallout/)
----------------------------------------------------------------------
* **Quoted errors are not the scatter.** The shuffled-element null put its
  99.9th percentile at ln LR 17 (dwarfs) against 2.3 on a synthetic population
  built from the quoted errors, because GALAH's per-element errors understate
  the measured peer-residual scatter by factors of 2-4. The error is therefore
  **floored at the measured peer scatter of that element in that sample**
  (:func:`error_floors`); the floors and their ratio to the quoted errors are
  recorded. For literature compilations the floor is the larger of that and
  the star-to-star scatter of duplicate entries (``literature_heterogeneity``).
* **Winning is not fitting.** Both giant "survivors" beat the natural models
  with chi2_f = 230 on 9 elements. A star whose *best* model has reduced chi2
  above ``max_reduced_chi2`` is ``UNEXPLAINED_BY_ALL_TEMPLATES``.
* **One hot element must not carry a pattern.** ``heavy_peak_incoherent``
  requires at least two of La/Ce/Nd individually >= 2 sigma up; ``la_cn_blend``
  distrusts La in cool giants once the La residual is shown to track C/N.

Everything here is a pure function of arrays, offline-testable. The only
survey-specific step is assembling the vectors from a normalised table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

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

#: Limit codes in a ``limits`` matrix: 0 detection, -1 upper limit, +1 lower limit.
DETECTION, UPPER_LIMIT, LOWER_LIMIT = 0, -1, 1

#: Every veto, in the order they are applied. Each is a named counter in
#: ``summary.json``; ``first_veto`` records which one a star hit first.
VETOES: tuple[str, ...] = (
    "low_snr_or_flagged",
    "unexplained_by_all_templates",
    "literature_heterogeneity",
    "s_process_star",
    "r_process_star",
    "young_ba_enhancement",
    "nlte_saturated_lines",
    "single_element_driver",
    "heavy_peak_incoherent",
    "la_cn_blend",
    "teff_peer_residual",
)

#: The ratios that decide the vector where a catalogue carries them.
DECISIVE_RATIOS: tuple[tuple[str, str], ...] = (("Pb", "Nd"), ("Ag", "Nd"), ("Pd", "Nd"), ("Eu", "Nd"))


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
    # literature compilations: a star whose own duplicate entries disagree by more
    # than this in a pattern element is not a candidate
    hetero_max_dex: float = 0.3
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


def decisive_ratios(vec: dict[str, float], limits: dict[str, int] | None = None) -> dict:
    """[Pb/Nd], [Ag/Nd], [Pd/Nd], [Eu/Nd] with the limit sense carried along.

    A Pb *upper* limit makes ``[Pb/Nd]`` an upper limit (``"<"``); an Nd upper
    limit would make it a lower limit. The numbers are the same as any
    difference of two dex values; the annotation is what a reader needs.
    """
    limits = limits or {}
    out = {}
    for a, b in DECISIVE_RATIOS:
        va, vb = vec.get(a), vec.get(b)
        key = f"{a}/{b}"
        if va is None or vb is None or not (np.isfinite(va) and np.isfinite(vb)):
            out[key] = float("nan")
            out[f"{key}_limit"] = ""
            continue
        out[key] = float(va - vb)
        la, lb = int(limits.get(a, 0) or 0), int(limits.get(b, 0) or 0)
        sense = ""
        if la == UPPER_LIMIT or lb == LOWER_LIMIT:
            sense = "<"
        if la == LOWER_LIMIT or lb == UPPER_LIMIT:
            sense = ">" if not sense else "?"
        out[f"{key}_limit"] = sense
    return out


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def amplitude_grid(cfg: PatternConfig, n: int | None = None) -> np.ndarray:
    n = int(n or cfg.n_amp)
    return np.concatenate([[0.0], np.logspace(np.log10(cfg.amp_min), np.log10(cfg.amp_max), n)])


@dataclass
class _Obs:
    """Prepared observation matrices: values, weights, sigmas, limit codes."""

    o: np.ndarray        # value (or the limit value) where measured, else 0
    w: np.ndarray        # 1/sigma^2 for detections, 0 for limits and unmeasured
    s: np.ndarray        # sigma where measured, 1 elsewhere
    lim: np.ndarray      # 0 / -1 / +1
    mask: np.ndarray     # measured (detection or limit)
    has_lim: bool


def _prep(obs, sig, limits=None) -> _Obs:
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sig) & (sig > 0)
    if limits is None:
        lim = np.zeros(obs.shape, dtype=int)
    else:
        lim = np.where(mask, np.nan_to_num(np.asarray(limits, dtype=float), nan=0.0).astype(int), 0)
    det = mask & (lim == 0)
    o = np.where(mask, obs, 0.0)
    w = np.where(det, 1.0 / np.where(mask, sig, 1.0) ** 2, 0.0)
    s = np.where(mask, sig, 1.0)
    return _Obs(o=o, w=w, s=s, lim=lim, mask=mask, has_lim=bool((lim != 0).any()))


def _loss(ob: _Obs, pred: np.ndarray) -> np.ndarray:
    """-2 ln L per star, up to a constant: chi2 for detections, censored terms for limits.

    ``pred`` broadcasts against ``ob.o`` ((K,) or (N, K)). For an upper limit
    the stored value is the limit ``u`` and the term is ``-2 ln Phi((u - pred)
    / sigma)``: a prediction far above the limit is heavily penalised, one far
    below costs nothing. A lower limit is the mirror image.
    """
    r = ob.o - pred
    chi2 = (r * r * ob.w).sum(axis=1)
    if ob.has_lim:
        z = r / ob.s
        up = ob.lim == UPPER_LIMIT
        lo = ob.lim == LOWER_LIMIT
        chi2 = chi2 + np.where(up, -2.0 * log_ndtr(z), 0.0).sum(axis=1) \
            + np.where(lo, -2.0 * log_ndtr(-z), 0.0).sum(axis=1)
    return chi2


def _chi2_at(ob: _Obs, pattern: np.ndarray, a: np.ndarray, fixed=None) -> np.ndarray:
    """Loss per star for a per-star amplitude ``a`` (N,) on ``pattern`` (K,)."""
    lin = 1.0 + a[:, None] * pattern[None, :]
    if fixed is not None:
        lin = lin + fixed
    return _loss(ob, np.log10(np.maximum(lin, 1e-12)))


def _refine(ob: _Obs, pattern: np.ndarray, a: np.ndarray, chi2: np.ndarray, *, fixed=None,
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
        c_lo = _chi2_at(ob, pattern, lo, fixed)
        c_hi = _chi2_at(ob, pattern, hi, fixed)
        denom = c_lo - 2.0 * chi2 + c_hi
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = 0.5 * (c_lo - c_hi) / denom      # in units of log(f)
        shift = np.where(np.isfinite(shift) & (denom > 0), np.clip(shift, -1.0, 1.0), 0.0)
        cand = a * f ** shift
        c_cand = _chi2_at(ob, pattern, cand, fixed)
        stack_c = np.stack([chi2, c_lo, c_hi, c_cand])
        stack_a = np.stack([a, lo, hi, cand])
        k = stack_c.argmin(axis=0)
        new_c = stack_c[k, np.arange(len(a))]
        new_a = stack_a[k, np.arange(len(a))]
        a = np.where(pos, new_a, a)
        chi2 = np.where(pos, new_c, chi2)
        f = f ** 0.5
    return a, chi2


def _scan_1d(ob: _Obs, pattern: np.ndarray, grid: np.ndarray, *, refine: bool = True
             ) -> tuple[np.ndarray, np.ndarray]:
    n = ob.o.shape[0]
    best = np.full(n, np.inf)
    best_a = np.zeros(n)
    for a in grid:
        chi2 = _loss(ob, np.log10(1.0 + a * pattern))
        better = chi2 < best
        best = np.where(better, chi2, best)
        best_a = np.where(better, a, best_a)
    if refine:
        best_a, best = _refine(ob, pattern, best_a, best)
    return best, best_a


def _scan_2d(ob: _Obs, p1: np.ndarray, p2: np.ndarray, grid: np.ndarray, *, refine: bool = True
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = ob.o.shape[0]
    best = np.full(n, np.inf)
    best_1 = np.zeros(n)
    best_2 = np.zeros(n)
    for a1 in grid:
        base = a1 * p1
        for a2 in grid:
            chi2 = _loss(ob, np.log10(1.0 + base + a2 * p2))
            better = chi2 < best
            best = np.where(better, chi2, best)
            best_1 = np.where(better, a1, best_1)
            best_2 = np.where(better, a2, best_2)
    if refine:
        for _ in range(2):
            best_1, best = _refine(ob, p1, best_1, best, fixed=best_2[:, None] * p2[None, :])
            best_2, best = _refine(ob, p2, best_2, best, fixed=best_1[:, None] * p1[None, :])
    return best, best_1, best_2


def fit_patterns(obs, sig, T: Templates, cfg: PatternConfig, limits=None) -> pd.DataFrame:
    """Fit the five hypotheses to every row of ``obs`` (N x K, NaN = unmeasured).

    ``limits`` (N x K, optional) marks censored values: -1 = the value is an
    upper limit, +1 = a lower limit, 0 = detection. Returns one row per star
    with the best loss and amplitude of each hypothesis, the two likelihood
    ratios, the absolute goodness of fit and the classification:

    ``fission_lr``  = 0.5 * (chi2_best_natural - chi2_fission): positive means
                      the fission-only fit beats the best of s, r and s+r;
    ``enrich_lr``   = 0.5 * (chi2_null - min(chi2_natural, chi2_fission)):
                      how strongly *anything* is added at all;
    ``reduced_chi2_best`` = loss of the best model per degree of freedom.
                      Above ``max_reduced_chi2`` the star is
                      ``UNEXPLAINED_BY_ALL_TEMPLATES`` whatever the ratios say.
    """
    ob = _prep(obs, sig, limits)
    n_meas = ob.mask.sum(axis=1)
    n_det = (ob.mask & (ob.lim == 0)).sum(axis=1)
    g1 = amplitude_grid(cfg, cfg.n_amp)
    g2 = amplitude_grid(cfg, cfg.n_amp_2d)

    chi2_null = _loss(ob, np.zeros(ob.o.shape[1]))
    chi2_s, a_s = _scan_1d(ob, T.S, g1)
    chi2_r, a_r = _scan_1d(ob, T.R, g1)
    chi2_sr, a_sr_s, a_sr_r = _scan_2d(ob, T.S, T.R, g2)
    chi2_f, a_f = _scan_1d(ob, T.F, g1)
    chi2_nat = np.minimum(chi2_sr, np.minimum(chi2_s, chi2_r))
    chi2_best = np.minimum(chi2_nat, chi2_f)

    fission_lr = 0.5 * (chi2_nat - chi2_f)
    enrich_lr = 0.5 * (chi2_null - chi2_best)
    dof = np.maximum(n_meas - 1, 1)
    red = chi2_best / dof

    nat = np.where(chi2_s <= chi2_r, S_PROCESS, R_PROCESS).astype(object)
    both = chi2_sr < np.minimum(chi2_s, chi2_r) - cfg.ambiguity_margin
    nat[both] = S_PLUS_R

    cls = np.full(len(ob.o), NORMAL, dtype=object)
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
        "n_detected": n_det.astype(int),
        "n_limits": (n_meas - n_det).astype(int),
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


def fission_lr_only(obs, sig, T: Templates, cfg: PatternConfig, limits=None) -> np.ndarray:
    """Just the fission-vs-natural ratio (the hot loop of LOO and the nulls)."""
    ob = _prep(obs, sig, limits)
    g1 = amplitude_grid(cfg, cfg.n_amp)
    g2 = amplitude_grid(cfg, cfg.n_amp_2d)
    chi2_s, _ = _scan_1d(ob, T.S, g1)
    chi2_r, _ = _scan_1d(ob, T.R, g1)
    chi2_sr, _, _ = _scan_2d(ob, T.S, T.R, g2)
    chi2_f, _ = _scan_1d(ob, T.F, g1)
    return 0.5 * (np.minimum(chi2_sr, np.minimum(chi2_s, chi2_r)) - chi2_f)


# ---------------------------------------------------------------------------
# Leave-one-out: is it a pattern or one element?
# ---------------------------------------------------------------------------
def leave_one_out(obs, sig, T: Templates, cfg: PatternConfig, limits=None) -> pd.DataFrame:
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
        out[f"lr_without_{el}"] = fission_lr_only(o2, sig, T, cfg, limits)
    df = pd.DataFrame(out)
    arr = df.to_numpy()
    df["lr_loo_min"] = arr.min(axis=1)
    df["lr_loo_driver"] = [T.elements[i] for i in arr.argmin(axis=1)]
    return df


def lr_without(obs, sig, T: Templates, cfg: PatternConfig, drop: tuple[str, ...],
               limits=None) -> np.ndarray:
    """``fission_lr`` with a named set of elements excluded (e.g. Ba)."""
    o2 = np.asarray(obs, dtype=float).copy()
    for el in drop:
        if el in T.elements:
            o2[:, T.index(el)] = np.nan
    return fission_lr_only(o2, sig, T, cfg, limits)


# ---------------------------------------------------------------------------
# Error model and vectors from a normalised table
# ---------------------------------------------------------------------------
def _flag_is_bad(v: pd.Series) -> np.ndarray:
    if v.dtype == object:
        return ~v.fillna("").astype(str).str.strip().isin(["", "0", "0.0", "nan", "False"]).to_numpy()
    return (pd.to_numeric(v, errors="coerce").fillna(0) != 0).to_numpy()


def error_floors(df: pd.DataFrame, elements: list[str], scatter: dict[str, float], *,
                 err_prefix: str = "e_", cfg: PatternConfig | None = None,
                 duplicate_scatter: dict[str, float] | None = None) -> dict[str, dict]:
    """Per-element error floor = the measured peer-residual scatter of that sample.

    Catalogue errors are formal fit errors; on the first real run they
    understated the residual scatter by 2-4x and the shuffled null's tail
    stretched to ln LR 17. Flooring every element's error at the robust width
    of its own peer residual (in this sample) is the same correction LOOM makes
    by rescaling sigma with sqrt(reduced chi2), applied per element. For a
    literature compilation the floor is the larger of that and the
    star-to-star scatter of duplicate entries (``duplicate_scatter``), which is
    the measured heterogeneity of the analyses that were mixed. Returns
    ``{el: {floor_dex, median_quoted_dex, inflation, source}}`` so the
    correction is on the record.
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
        source = "peer_scatter" if floor > 0 else "none"
        dup = (duplicate_scatter or {}).get(el)
        if dup is not None and np.isfinite(dup) and dup > floor:
            floor = float(dup)
            source = "duplicate_scatter"
        out[el] = {"floor_dex": round(floor, 4),
                   "median_quoted_dex": round(med, 4) if np.isfinite(med) else None,
                   "inflation": round(floor / med, 2) if (np.isfinite(med) and med > 0 and floor > 0) else None,
                   "source": source}
    return out


def assemble_vectors(df: pd.DataFrame, elements: list[str], *, value_prefix: str = "",
                     err_prefix: str = "e_", flag_prefix: str = "f_",
                     cfg: PatternConfig | None = None,
                     fallback_sigma: dict[str, float] | None = None,
                     sigma_floor: dict[str, float] | None = None,
                     limit_prefix: str | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(obs, sig, flagged) matrices over ``elements`` from a canonical table.

    A flagged measurement is *excluded* (NaN) and counted, never used. The
    error is ``sqrt(max(reported, floor)^2 + systematic^2)``; a missing reported
    error falls back to the element's empirical scatter if given, else the
    default. ``sigma_floor`` is the per-element floor from :func:`error_floors`.
    With ``limit_prefix`` the info dict also carries ``limits`` (N x K int
    matrix from ``<limit_prefix><El>`` columns; absent columns mean detections).
    """
    cfg = cfg or PatternConfig()
    n = len(df)
    obs = np.full((n, len(elements)), np.nan)
    sig = np.full((n, len(elements)), np.nan)
    flagged = np.zeros((n, len(elements)), dtype=bool)
    lims = np.zeros((n, len(elements)), dtype=int)
    info = {"elements": list(elements), "value_columns": {}, "error_columns": {},
            "flag_columns": {}, "limit_columns": {}, "missing": []}
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
        if limit_prefix is not None and f"{limit_prefix}{el}" in df.columns:
            lims[:, k] = np.nan_to_num(pd.to_numeric(df[f"{limit_prefix}{el}"], errors="coerce")
                                       .to_numpy(dtype=float), nan=0.0).astype(int)
            info["limit_columns"][el] = f"{limit_prefix}{el}"
        obs[:, k] = v
        sig[:, k] = np.sqrt(e ** 2 + cfg.systematic_floor_dex ** 2)
    if limit_prefix is not None:
        info["limits"] = lims
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
                   exclude_from_fit: dict[str, np.ndarray] | None = None,
                   ) -> tuple[pd.DataFrame, dict[str, float], dict]:
    """``[X/H]`` residuals against a robust polynomial in ([Fe/H], Teff, logg[, alpha]).

    Reuses TAILINGS' ``fit_element`` (iteratively clipped least squares, so a
    genuine enrichment cannot drag its own reference surface) and its
    leave-one-out alpha proxy when alpha elements are present. Working in
    ``[X/H]`` with ``[Fe/H]`` as a predictor removes the normalisation
    aberration (an error in the star's own [Fe/H] otherwise shifts every
    [X/Fe] together, which is a *coherent* vector and exactly what a pattern
    fit must not be fed).

    ``exclude_from_fit[el]`` (boolean per row) keeps rows out of the surface
    fit -- upper limits, which would bias it -- while still giving them a
    residual against the fitted surface.

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
        y_fit = y
        if exclude_from_fit and el in exclude_from_fit:
            y_fit = np.where(np.asarray(exclude_from_fit[el], dtype=bool), np.nan, y)
        P = list(cols)
        if have_alpha:
            P.append(M.alpha_proxy(df, exclude=el))
        P = np.column_stack(P)
        if have_alpha:
            ap = P[:, -1]
            P[:, -1] = np.where(np.isfinite(ap), ap, np.nanmedian(ap) if np.isfinite(ap).any() else 0.0)
        try:
            fit = M.fit_element(P, y_fit, element=el, degree=degree, clip=clip, n_iter=n_iter,
                                min_rows=min_rows)
            r = y - fit.predict(P)
            scatter[el] = float(fit.robust_scatter)
            notes["n_fit"][el] = int(fit.n_fit)
        except ValueError:
            med = np.nanmedian(y_fit) if np.isfinite(y_fit).any() else np.nan
            r = y - med
            rr = (y_fit - med)[np.isfinite(y_fit)]
            scatter[el] = float(M.robust_sigma(rr)) if rr.size else float("nan")
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
    both are NaN and do not count toward coherence. An upper limit
    (``lim_<El> == -1``) can never count as "up".
    """
    out = {}
    for el in cfg.heavy_peak_elements:
        if f"z_{el}" in cand.columns:
            z = _col(cand, f"z_{el}")
        else:
            p = _col(cand, f"peer_{el}")
            s = _col(cand, f"sig_{el}")
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(np.isfinite(p) & np.isfinite(s) & (s > 0), p / s, np.nan)
        lim = _col(cand, f"lim_{el}")
        z = np.where(np.isfinite(lim) & (lim == UPPER_LIMIT), np.nan, z)
        out[el] = z
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
    * ``literature_heterogeneity``  the star's own duplicate literature entries
      disagree by more than ``hetero_max_dex`` in a pattern element
      (``hetero_max_dex`` column, from the compilation's duplicates).
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

    # 3. literature heterogeneity (compilations only; column absent -> no veto)
    het = _col(out, "hetero_max_dex")
    out["veto_literature_heterogeneity"] = _finite_gt(het, cfg.hetero_max_dex)

    # 4. s-process star
    nd_ba = nd - ba
    v = _finite_gt(ba, cfg.s_ba_fe_min) & (_finite_lt(nd_ba, 0.0)
                                            | (_finite_gt(y, cfg.s_light_fe_min)
                                               & _finite_gt(zr, cfg.s_light_fe_min)))
    v |= _finite_gt(np.abs(binary), 0.0)
    out["veto_s_process_star"] = v

    # 5. r-process star
    out["veto_r_process_star"] = _finite_gt(eu, cfg.r_eu_fe_min) & _finite_gt(eu - nd, 0.0)

    # 6. young Ba enhancement. A(Li) = [Li/Fe] + [Fe/H] + A(Li)_sun.
    ali = li + np.where(np.isfinite(feh), feh, 0.0) + Y.SOLAR_LOGEPS["Li"]
    young = _finite_gt(ali, cfg.young_ali_min) | _finite_lt(age, cfg.young_age_max_gyr)
    out["veto_young_ba_enhancement"] = young & _finite_gt(ba, cfg.young_ba_fe_min)

    # 7. NLTE / saturation: hold without Ba
    lr_noba = _col(out, "lr_noba")
    out["veto_nlte_saturated_lines"] = _finite_lt(lr_noba, thr) | ~np.isfinite(lr_noba)

    # 8. single-element driver
    loo = _col(out, "lr_loo_min")
    out["veto_single_element_driver"] = _finite_lt(loo, thr) | ~np.isfinite(loo)

    # 9. heavy-peak coherence
    z = heavy_peak_z(out, cfg)
    zarr = z.to_numpy(dtype=float) if len(z.columns) else np.full((n, 0), np.nan)
    n_coh = (np.isfinite(zarr) & (zarr >= cfg.heavy_peak_sigma)).sum(axis=1)
    any_z = np.isfinite(zarr).any(axis=1) if zarr.shape[1] else np.zeros(n, dtype=bool)
    out["n_heavy_coherent"] = n_coh
    out["veto_heavy_peak_incoherent"] = (n_coh < cfg.heavy_peak_min_coherent) | ~any_z

    # 10. La in cool giants
    sample = out["sample"].astype(str).to_numpy() if "sample" in out.columns else np.full(n, "", dtype=object)
    lr_wo_la = _col(out, "lr_without_La")
    driver = out["lr_loo_driver"].astype(str).to_numpy() if "lr_loo_driver" in out.columns \
        else np.full(n, "", dtype=object)
    la_carries = _finite_lt(lr_wo_la, thr) | (driver == "La")
    z_la = _col(z, "La") if "La" in z.columns else np.full(n, np.nan)
    la_carries |= np.isfinite(z_la) & (z_la >= cfg.heavy_peak_sigma) & (n_coh <= 1)
    is_cool_giant = (sample == "giant") & _finite_lt(teff, cfg.la_cn_teff_max)
    out["veto_la_cn_blend"] = bool(la_cn_suspect) & is_cool_giant & la_carries

    # 11. raw-space consistency
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
                  max_rows: int = 20000, rng: np.random.Generator | None = None,
                  limits=None) -> dict:
    """Permute the element labels within each star and re-score.

    Each star's (value, error, limit-code) triples are shuffled across its
    element slots, so the amplitude structure and the per-element noise are
    kept and only the *alignment with the fission shape* is destroyed. The
    distribution of ``fission_lr`` under that shuffle is how often
    noise-plus-natural-spread makes the vector by accident; its quantile sets
    the threshold.
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
    lm = np.asarray(limits)[idx] if limits is not None else None
    lrs = []
    for _ in range(int(n_perm)):
        keys = rng.random(o.shape)
        order = np.argsort(keys, axis=1)
        lrs.append(fission_lr_only(np.take_along_axis(o, order, 1),
                                   np.take_along_axis(s, order, 1), T, cfg,
                                   np.take_along_axis(lm, order, 1) if lm is not None else None))
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


def _inject_and_score(base, s, T, cfg, a, lr_threshold, with_loo, lim=None):
    shift = np.log10(1.0 + a * T.F)
    o = base + shift[None, :]
    lr = fission_lr_only(o, s, T, cfg, lim)
    passed = lr >= lr_threshold
    rec = {"frac_lr_pass": float(passed.mean()) if len(lr) else float("nan"),
           "median_lr": float(np.median(lr)) if len(lr) else float("nan")}
    if with_loo and len(lr):
        loo = leave_one_out(o, s, T, cfg, lim)
        rec["frac_lr_and_loo_pass"] = float((passed & (loo["lr_loo_min"].to_numpy() >= lr_threshold)).mean())
    return rec, shift


def sensitivity_curve(obs, sig, T: Templates, cfg: PatternConfig, *, lr_threshold: float,
                      amplitudes=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0), n_inject: int = 1500,
                      rng: np.random.Generator | None = None, with_loo: bool = True,
                      testable: np.ndarray | None = None, limits=None) -> list[dict]:
    """Inject the fission pattern at each amplitude into real vectors and re-score.

    Two completeness numbers per amplitude: over **testable** stars
    (:func:`testable_mask` -- enough elements and enough heavy peak measured
    for the pattern to be decidable) and over **all** stars. The first is the
    channel's sensitivity; the second is what the raw survey delivers once the
    untestable fraction is folded in. The testable fraction is reported with
    the curve so neither number can be read without the other. Upper limits
    travel with the injection: a limit shifted up is still a limit.
    """
    rng = rng or np.random.default_rng(7)
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    lim = np.asarray(limits) if limits is not None else None
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
        rec_all, shift = _inject_and_score(obs[idx_all], sig[idx_all], T, cfg, a, lr_threshold, with_loo,
                                           lim[idx_all] if lim is not None else None)
        if idx_t.size:
            rec_t, _ = _inject_and_score(obs[idx_t], sig[idx_t], T, cfg, a, lr_threshold, with_loo,
                                         lim[idx_t] if lim is not None else None)
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
    "AMBIGUOUS", "DECISIVE_RATIOS", "DETECTION", "FISSION", "INSUFFICIENT", "LOWER_LIMIT", "NORMAL",
    "R_PROCESS", "S_PLUS_R", "S_PROCESS", "UNEXPLAINED", "UPPER_LIMIT", "VETOES", "PatternConfig",
    "Templates", "amplitude_grid", "apply_vetoes", "assemble_vectors", "build_templates",
    "decisive_ratios", "derive_threshold", "discriminant_ratios", "error_floors",
    "fission_lr_only", "fit_patterns", "heavy_peak_z", "la_diagnostics", "leave_one_out",
    "lr_without", "peer_residuals", "raw_vectors", "sample_null", "sensitivity_curve",
    "shuffled_null", "testable_mask", "veto_counters",
]

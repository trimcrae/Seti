"""The sparse-anomaly statistic, and its explicit dense contrast.

The discriminant
----------------
Given the standardised residual vector ``z`` from ``manifold.zscores`` — one
number per element, already conditioned on ([Fe/H], Teff, log g, alpha) and
divided by the empirical scatter at that star's (SNR, Teff) — the question is
not "is any element extreme?" but "is *exactly one* element extreme while
everything else is ordinary?"

That distinction is the whole channel, because it maps directly onto the
physics:

* **Dense anomaly = natural.** Every astrophysical process that changes a
  photosphere's composition changes a *family*. AGB pollution raises the whole
  s-process (Sr, Y, Zr and Ba, La, Ce together). A Type Ia deficit lowers the
  whole Fe-peak. Accretion of rock raises every refractory along a condensation
  -temperature trend. A bad spectral fit moves many elements at once, because
  they all share Teff, log g, the continuum and the microturbulence.
* **Sparse anomaly = not natural.** Refining is *defined* by separating one
  element from its chemical neighbours. Isotope separation, ore beneficiation,
  fission-product partitioning — all of them produce a stream enriched in one
  element and depleted in nothing else in particular.

So the statistic is deliberately *anti*-correlated with amplitude-based outlier
detection. A star with 15 elements at 4 sigma is a strong global outlier and a
**rejection** here. A star with 1 element at 7 sigma and 24 elements inside
2 sigma is an unremarkable global outlier and the **target**.

Reported contrast
-----------------
Every star gets both numbers, and the run reports the joint distribution:

* ``z_max`` — the amplitude of the largest deviation;
* ``z_rest_rms`` — the RMS over every element except the largest, i.e. how
  *dense* the anomaly is;
* ``contrast = z_max / max(z_background_rms, 1)`` — the sparsity itself, where
  the background excludes the discrepant set so a legitimate two-element
  anomaly is not penalised by its own second element.

A natural chemically peculiar star and a pipeline failure both sit at high
``z_max`` *and* high ``z_rest_rms``. Only the artificial hypothesis predicts
high ``z_max`` with ``z_rest_rms ~ 1``. Publishing the two-dimensional
distribution is what makes the claim falsifiable rather than a threshold
choice.

Two vetoes beyond the counting rule
-----------------------------------
1. **Family coherence.** Even when only one element crosses the hard threshold,
   if its nucleosynthetic siblings are all leaning the same way at 2-3 sigma,
   that is a family event caught early, not a sparse one. The mean |z| over the
   *other* members of the flagged element's family is required to be small.
2. **The quiet elements are the evidence, not the loud one.** Huang, Tao &
   Zhang (2026) showed quantitatively, for polluted white dwarfs, that a
   record with only one or two *measured* elements can produce a large Bayes
   factor while being information-starved -- their high-evidence one- and
   two-element records are the ones they then disqualify. That criticism is
   real and it is the reason ``n_quiet`` (elements measured and inside
   ``z_quiet``) is carried as a first-class statistic. Their low-N_det records
   are cases where the other elements were **never measured**; here the other
   ~20-30 elements are measured *and quiet*, and it is exactly that
   information which their archival compilation lacks. A candidate is only as
   strong as its ``n_quiet``.
3. **Naturally sparse elements are excluded.** Lithium varies by orders of
   magnitude among otherwise identical cool dwarfs through convective
   depletion; C and N are pipeline-fragile and evolutionarily mixed. A known
   single-element variable cannot be evidence for an unknown one. They are
   still measured and reported — a Li excess is the classic engulfment tracer,
   so it is diagnostic in the *opposite* direction (see ``twins``).

Thresholds
----------
The default hard threshold is 6 sigma against the *empirical* scatter, not the
catalogue error. Calibration context from the chemical-tagging literature:
intrinsic star-to-star scatter within a birth cluster is 0.01-0.05 dex
depending on element (Bovy 2016; Cheng et al. 2020; Patil et al. 2022;
Casamiquela et al. 2021), and survey-scale residual scatter is a few times
that. Six sigma on a 0.03-0.05 dex empirical width is a 0.2-0.3 dex
single-element excursion — an order of magnitude above the intrinsic
chemical individuality of co-natal stars, and comfortably above the 0.03 dex
precision at which ~0.3% of unrelated field-star pairs are already
indistinguishable (Ness et al. 2018).

The residual distribution is emphatically **not** Gaussian in the tails, so 6
sigma is not a p-value and must never be quoted as one. The false-positive
control is empirical and comes later: per-element flag rates, per-field flag
rates, cross-survey confirmation and raw-spectrum re-measurement (see ``vet``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .manifold import NATURALLY_SPARSE_ELEMENTS, element_family


@dataclass(frozen=True)
class SparseConfig:
    """Thresholds for the sparse/dense classification."""

    z_flag: float = 6.0
    """Hard threshold: |z| at or above this counts as discrepant."""

    z_quiet: float = 2.0
    """'Everything else is normal' means |z| below this."""

    max_discrepant: int = 2
    """At most this many elements may be discrepant (the 'one or two' rule)."""

    max_quiet_excess: int = 1
    """How many elements beyond the discrepant set may exceed ``z_quiet``."""

    min_elements: int = 8
    """Below this many measured elements, sparsity is not a meaningful claim."""

    min_contrast: float = 3.0
    """Required ``z_max / max(z_background_rms, 1)``."""

    family_max_mean_z: float = 2.0
    """Mean |z| over the flagged element's other family members must stay below this."""

    dense_min_discrepant: int = 3
    """At or above this many discrepant elements the star is labelled DENSE."""

    exclude_elements: tuple[str, ...] = NATURALLY_SPARSE_ELEMENTS
    """Elements that cannot carry a candidacy (known natural single-element variables)."""


SPARSE = "SPARSE"
DENSE = "DENSE"
NORMAL = "NORMAL"
INSUFFICIENT = "INSUFFICIENT"


def _order_desc(absz: np.ndarray) -> np.ndarray:
    """Indices sorting |z| descending, NaNs last."""
    filled = np.where(np.isfinite(absz), absz, -np.inf)
    return np.argsort(-filled, kind="stable")


def sparse_statistics(
    Z: pd.DataFrame,
    *,
    cfg: SparseConfig | None = None,
) -> pd.DataFrame:
    """Per-star sparsity statistics and classification.

    ``Z`` has one row per star and one column per element, holding the
    standardised residual. NaN means "not measured" and is simply skipped —
    the element count is carried so a star measured in 9 elements is never
    compared against one measured in 30 without that being visible.
    """
    cfg = cfg or SparseConfig()
    elements = [c for c in Z.columns if c not in cfg.exclude_elements]
    excluded = [c for c in Z.columns if c in cfg.exclude_elements]

    Zv = Z[elements].to_numpy(dtype=float)
    A = np.abs(Zv)
    n_el = np.isfinite(A).sum(axis=1)

    n_rows = Zv.shape[0]
    rows = []
    for k in range(n_rows):
        a = A[k]
        finite = np.isfinite(a)
        m = int(finite.sum())
        if m == 0:
            rows.append(_empty_row(cfg))
            continue

        order = _order_desc(a)
        order = [o for o in order if finite[o]]
        i0 = order[0]
        z_max = float(a[i0])
        el_max = elements[i0]
        signed_max = float(Zv[k, i0])
        z_second = float(a[order[1]]) if len(order) > 1 else float("nan")
        el_second = elements[order[1]] if len(order) > 1 else None

        n_disc = int((a[finite] >= cfg.z_flag).sum())
        n_active = int((a[finite] >= cfg.z_quiet).sum())

        # Two background statistics, for two different jobs.
        #
        # z_rest_rms excludes ONLY the single largest element, so it answers
        # "how much of the rest of the abundance vector moved?" — a dense
        # anomaly is loud here by construction and a sparse one is not. This is
        # the honest density diagnostic and it is what the contrast table
        # reports.
        rest_idx = order[1:]
        z_rest_rms = (
            float(np.sqrt(np.mean(a[rest_idx] ** 2))) if rest_idx else float("nan")
        )
        # z_background_rms excludes the whole discrepant set, so a legitimate
        # two-element sparse anomaly is not penalised by its own second
        # element. This is what the contrast cut uses.
        bg_idx = order[max(1, n_disc):]
        z_background_rms = (
            float(np.sqrt(np.mean(a[bg_idx] ** 2))) if bg_idx else float("nan")
        )
        contrast = z_max / max(
            z_background_rms if np.isfinite(z_background_rms) else 0.0, 1.0
        )

        fam = element_family(el_max)
        sibs = [
            idx
            for idx, e in enumerate(elements)
            if e != el_max and element_family(e) == fam and finite[idx]
        ]
        fam_mean_z = float(np.mean(a[sibs])) if sibs else float("nan")
        n_fam_active = int((a[sibs] >= cfg.z_quiet).sum()) if sibs else 0

        label, reason = classify(
            n_elements=m,
            n_discrepant=n_disc,
            n_active=n_active,
            contrast=contrast,
            family_mean_z=fam_mean_z,
            cfg=cfg,
        )

        rows.append(
            {
                "n_elements": m,
                "z_max": z_max,
                "z_max_signed": signed_max,
                "element_max": el_max,
                "family_max": fam,
                "z_second": z_second,
                "element_second": el_second,
                "n_discrepant": n_disc,
                "n_active": n_active,
                "n_quiet": m - n_active,
                "z_rest_rms": z_rest_rms,
                "z_background_rms": z_background_rms,
                "contrast": contrast,
                "family_mean_z": fam_mean_z,
                "n_family_active": n_fam_active,
                "n_family_siblings": len(sibs),
                "classification": label,
                "reason": reason,
            }
        )

    out = pd.DataFrame(rows, index=Z.index)
    out["n_elements_all"] = n_el
    for e in excluded:
        out[f"z_{e}_excluded"] = Z[e].to_numpy(dtype=float)
    return out


def _empty_row(cfg: SparseConfig) -> dict:
    return {
        "n_elements": 0,
        "z_max": float("nan"),
        "z_max_signed": float("nan"),
        "element_max": None,
        "family_max": None,
        "z_second": float("nan"),
        "element_second": None,
        "n_discrepant": 0,
        "n_active": 0,
        "n_quiet": 0,
        "z_rest_rms": float("nan"),
        "z_background_rms": float("nan"),
        "contrast": float("nan"),
        "family_mean_z": float("nan"),
        "n_family_active": 0,
        "n_family_siblings": 0,
        "classification": INSUFFICIENT,
        "reason": "no measured elements",
    }


def classify(
    *,
    n_elements: int,
    n_discrepant: int,
    n_active: int,
    contrast: float,
    family_mean_z: float,
    cfg: SparseConfig | None = None,
) -> tuple[str, str]:
    """Apply the sparse/dense rules in order, returning ``(label, reason)``.

    Order matters and is deliberate: the density rules fire *before* the
    contrast rule, so a natural family event is always labelled DENSE rather
    than merely failing a contrast cut. That keeps the rejection reason
    physically meaningful in the report instead of collapsing everything into
    "did not pass".
    """
    cfg = cfg or SparseConfig()
    if n_elements < cfg.min_elements:
        return INSUFFICIENT, f"only {n_elements} measured elements (< {cfg.min_elements})"
    if n_discrepant == 0:
        return NORMAL, f"no element at |z| >= {cfg.z_flag}"
    if n_discrepant >= cfg.dense_min_discrepant:
        return DENSE, f"{n_discrepant} elements discrepant: a family/global event, not sparse"
    if n_discrepant > cfg.max_discrepant:
        return DENSE, f"{n_discrepant} elements discrepant (> {cfg.max_discrepant})"
    if n_active > n_discrepant + cfg.max_quiet_excess:
        return DENSE, (
            f"{n_active - n_discrepant} background elements above |z| = {cfg.z_quiet}: "
            "the rest of the abundance vector is not quiet"
        )
    if np.isfinite(family_mean_z) and family_mean_z >= cfg.family_max_mean_z:
        return DENSE, (
            f"nucleosynthetic family co-moves (mean |z| of siblings = {family_mean_z:.2f} "
            f">= {cfg.family_max_mean_z})"
        )
    if not np.isfinite(contrast) or contrast < cfg.min_contrast:
        return DENSE, f"contrast {contrast:.2f} < {cfg.min_contrast}"
    return SPARSE, "single-element excursion with a quiet background"


def global_statistics(Z: pd.DataFrame, *, cfg: SparseConfig | None = None) -> pd.DataFrame:
    """The statistics the *rest of the field* uses, for direct comparison.

    Every executed abundance-outlier method reduces the residual vector to a
    single global distance: a reduced chi-squared, a Mahalanobis distance, an
    autoencoder reconstruction error, a random-forest proximity. Those are all
    monotone in how *many* elements deviate, so a dense anomaly beats a sparse
    one of the same per-element amplitude, and a sparse anomaly is diluted by
    the quiet elements it is defined by.

    ``chi2_leave_one_out`` reproduces the convention Weinberg et al. adopted
    when they searched residuals star by star: omit the element making the
    largest contribution, on the reasoning that a lone deviant element is
    usually an artifact. That choice is defensible and it is also an explicit
    exclusion of the TAILINGS signal -- it *requires* at least two anomalous
    abundances. Computing it here alongside the sparse statistic makes the
    contrast measurable rather than asserted, and the injection-recovery
    comparison in the test suite is the demonstration.
    """
    cfg = cfg or SparseConfig()
    elements = [c for c in Z.columns if c not in cfg.exclude_elements]
    A = np.abs(Z[elements].to_numpy(dtype=float))
    rows = []
    for k in range(A.shape[0]):
        a = A[k][np.isfinite(A[k])]
        if a.size == 0:
            rows.append({"n_used": 0, "chi2_reduced": np.nan,
                         "chi2_leave_one_out": np.nan, "z_sum_abs": np.nan})
            continue
        order = np.argsort(-a)
        drop = a[order[1:]] if a.size > 1 else np.array([])
        rows.append({
            "n_used": int(a.size),
            "chi2_reduced": float(np.mean(a**2)),
            "chi2_leave_one_out": float(np.mean(drop**2)) if drop.size else 0.0,
            "z_sum_abs": float(np.sum(a)),
        })
    return pd.DataFrame(rows, index=Z.index)


def contrast_table(stats: pd.DataFrame, *, z_bins: tuple[float, ...] = (2, 3, 4, 6, 8, 12)) -> pd.DataFrame:
    """The sparse/dense contrast, binned in ``z_max`` — the headline diagnostic.

    This is the table that decides whether the channel found anything. If the
    SPARSE fraction is flat with ``z_max`` the sample is behaving like noise
    plus systematics; a genuine population of artificial anomalies appears as a
    SPARSE excess that *survives* to high ``z_max`` while DENSE dominates at
    moderate ``z_max`` (where real chemical peculiarity lives).
    """
    edges = list(z_bins) + [np.inf]
    rows = []
    zm = stats["z_max"].to_numpy(dtype=float)
    lab = stats["classification"].to_numpy()
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        sel = np.isfinite(zm) & (zm >= lo) & (zm < hi)
        n = int(sel.sum())
        rows.append(
            {
                "z_max_lo": lo,
                "z_max_hi": hi,
                "n": n,
                "n_sparse": int((sel & (lab == SPARSE)).sum()),
                "n_dense": int((sel & (lab == DENSE)).sum()),
                "n_normal": int((sel & (lab == NORMAL)).sum()),
                "sparse_frac": (float((sel & (lab == SPARSE)).sum()) / n) if n else float("nan"),
                "median_z_rest_rms": (
                    float(np.nanmedian(stats.loc[sel, "z_rest_rms"])) if n else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def element_flag_rates(stats: pd.DataFrame, Z: pd.DataFrame, cfg: SparseConfig | None = None) -> pd.DataFrame:
    """Per-element rate of crossing ``z_flag``, and the share of SPARSE labels it carries.

    A pipeline systematic in one element shows up here as a flag rate far above
    every other element. If one element carries most of the candidates, that is
    a statement about the line list, not about the Galaxy — and it is reported
    as such rather than discovered later.
    """
    cfg = cfg or SparseConfig()
    A = Z.abs()
    n_meas = A.notna().sum()
    n_flag = (A >= cfg.z_flag).sum()
    carried = stats.loc[stats["classification"] == SPARSE, "element_max"].value_counts()
    rows = []
    for el in Z.columns:
        rows.append(
            {
                "element": el,
                "family": element_family(el),
                "n_measured": int(n_meas.get(el, 0)),
                "n_flagged": int(n_flag.get(el, 0)),
                "flag_rate": (
                    float(n_flag.get(el, 0)) / float(n_meas.get(el, 0))
                    if n_meas.get(el, 0)
                    else float("nan")
                ),
                "n_sparse_candidates_carried": int(carried.get(el, 0)),
                "excluded_by_construction": el in cfg.exclude_elements,
            }
        )
    return pd.DataFrame(rows).sort_values("flag_rate", ascending=False, ignore_index=True)


__all__ = [
    "DENSE",
    "INSUFFICIENT",
    "NORMAL",
    "SPARSE",
    "SparseConfig",
    "classify",
    "contrast_table",
    "element_flag_rates",
    "global_statistics",
    "sparse_statistics",
]

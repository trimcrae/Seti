"""The natural abundance manifold, and the empirical scatter around it.

Why a manifold and not raw abundances
-------------------------------------
A star's ``[X/Fe]`` is not free. It is set, to first order, by where the star
sits in the Galaxy's chemical-evolution history: its ``[Fe/H]`` fixes how much
Type Ia iron has been added, and its alpha enhancement fixes how much
core-collapse material it inherited. Conditioning on those two removes almost
all of the variance for almost every element — this is the content of the
two-process model (Weinberg et al.) and of the repeated finding that abundance
space has only a handful of independent dimensions (Ting et al.,
Price-Jones & Bovy).

The other two predictors are not astrophysics, they are the pipeline. A
spectroscopic ``[X/Fe]`` is a *fitted* quantity, and its systematic error is a
smooth function of ``Teff`` and ``log g`` (line strength, blending, continuum
placement, NLTE departures all track the atmosphere). Regressing them out means
the residual is dominated by genuine star-to-star chemical individuality plus
noise, not by a trend across the HR diagram. It also means a candidate cannot
be manufactured by simply sitting at an unusual Teff.

So: for each element X, fit

    [X/Fe] = f([Fe/H], Teff, log g, alpha_proxy) + r_X

with ``f`` a low-order polynomial fitted robustly (iterative sigma clipping, so
the very anomalies being searched for cannot drag the manifold toward
themselves), and keep the residual ``r_X``.

The alpha proxy is **leave-one-out**: when fitting Mg, the proxy is built from
the other alpha elements. Otherwise an element would partly predict itself and
its residual would be artificially crushed.

Why the scatter has to be empirical
-----------------------------------
Catalogue-reported abundance uncertainties are formal fit errors and are
routinely too small — they do not know about line-list error, unresolved
blends, or continuum systematics. Using them to define a "6-sigma" outlier
would manufacture candidates by the thousand. Instead the scatter is measured:
the robust (MAD-based) width of the residual distribution in bins of
(SNR, Teff), which is exactly the two variables that control abundance
precision. That empirical width already contains the measurement error, so it
is the honest denominator.

The reported per-star error is still used, but only defensively:
``sigma_used = max(sigma_empirical(SNR, Teff), sigma_reported)``. A star whose
own fit was unusually bad does not get to be a candidate on the strength of the
population's typical precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Nucleosynthetic families. The whole claim rests on these: natural processes
# move a family, refining moves a member. Assignments follow standard practice
# (see e.g. the two-process decompositions of APOGEE abundances); elements with
# genuinely mixed origin are given their own labels rather than being forced.
# ---------------------------------------------------------------------------
NUCLEO_FAMILIES: dict[str, tuple[str, ...]] = {
    "alpha": ("O", "Mg", "Si", "S", "Ca", "Ti", "TiII"),
    "odd_z": ("Na", "Al", "K", "Sc", "P"),
    "fe_peak": ("V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"),
    "s_light": ("Sr", "Y", "Zr"),
    "s_heavy": ("Ba", "La", "Ce"),
    "r_mixed": ("Nd", "Pr", "Sm"),
    "r_process": ("Eu", "Gd", "Dy"),
    "cno": ("C", "N"),
    "light": ("Li", "Be", "B"),
}

#: Elements whose alpha-element behaviour makes them the natural proxy.
ALPHA_ELEMENTS: tuple[str, ...] = ("Mg", "Si", "Ca", "Ti", "O", "S")

#: Elements excluded from the sparse statistic by construction. Lithium is
#: destroyed by convective mixing and varies by orders of magnitude among
#: otherwise identical cool dwarfs — it is a *known* single-element variable,
#: so it can never be evidence for an *unknown* one. It is still measured and
#: reported, because a Li excess is the classic engulfment tracer and is
#: therefore diagnostic in the opposite direction (see ``twins``).
NATURALLY_SPARSE_ELEMENTS: tuple[str, ...] = ("Li", "Be", "B", "C", "N")


def element_family(element: str) -> str:
    """Return the nucleosynthetic family label for ``element`` (or ``other``)."""
    el = element.strip()
    for fam, members in NUCLEO_FAMILIES.items():
        if el in members:
            return fam
    return "other"


def family_members(element: str, available: list[str]) -> list[str]:
    """The *other* members of ``element``'s family that are present in ``available``."""
    fam = element_family(element)
    if fam == "other":
        return []
    return [e for e in available if e != element and element_family(e) == fam]


def alpha_proxy(
    df: pd.DataFrame,
    *,
    exclude: str | None = None,
    prefix: str = "",
    alpha_elements: tuple[str, ...] = ALPHA_ELEMENTS,
) -> np.ndarray:
    """Leave-one-out alpha enhancement, as the mean of the available alpha ``[X/Fe]``.

    Returns NaN where no alpha element other than ``exclude`` is measured; the
    caller decides whether to drop those rows or fall back to a three-predictor
    fit.
    """
    cols = [f"{prefix}{e}" for e in alpha_elements if e != exclude and f"{prefix}{e}" in df.columns]
    if not cols:
        return np.full(len(df), np.nan)
    block = df[cols].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(block, axis=1)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------
def poly_design(P: np.ndarray, degree: int = 2) -> tuple[np.ndarray, list[str]]:
    """Polynomial design matrix in the columns of ``P``.

    Degree 2 gives the constant, the linear terms and all pairwise products
    (including squares). Degree 3 adds the pure cubes only — full cubic
    interactions buy nothing over a smooth chemical-evolution trend and cost
    conditioning.
    """
    n, k = P.shape
    cols = [np.ones(n)]
    names = ["1"]
    for i in range(k):
        cols.append(P[:, i])
        names.append(f"x{i}")
    if degree >= 2:
        for i in range(k):
            for j in range(i, k):
                cols.append(P[:, i] * P[:, j])
                names.append(f"x{i}x{j}")
    if degree >= 3:
        for i in range(k):
            cols.append(P[:, i] ** 3)
            names.append(f"x{i}^3")
    return np.column_stack(cols), names


def robust_sigma(x: np.ndarray) -> float:
    """MAD-based sigma, NaN-tolerant. Returns NaN for fewer than 3 finite values."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


@dataclass
class ElementFit:
    """A robust polynomial fit of one element's ``[X/Fe]`` against the predictors."""

    element: str
    term_names: list[str]
    coeffs: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    degree: int
    n_fit: int
    n_clipped: int
    robust_scatter: float
    predictor_names: list[str] = field(default_factory=list)

    def predict(self, P: np.ndarray) -> np.ndarray:
        """Predicted ``[X/Fe]`` for a raw (unstandardised) predictor block."""
        Ps = (np.asarray(P, dtype=float) - self.center) / self.scale
        X, _ = poly_design(Ps, degree=self.degree)
        return X @ self.coeffs


def fit_element(
    P: np.ndarray,
    y: np.ndarray,
    *,
    element: str = "X",
    degree: int = 2,
    clip: float = 4.0,
    n_iter: int = 4,
    min_rows: int = 50,
    predictor_names: list[str] | None = None,
) -> ElementFit:
    """Robustly regress ``y`` on ``P`` with iterative sigma clipping.

    Clipping matters for a reason specific to this search: the manifold is
    fitted on the same stars that are then tested against it. Without clipping,
    a genuine strong anomaly pulls the surface toward itself and partially
    hides. With clipping at 4 sigma over a few iterations, a 6-8 sigma outlier
    is excluded from its own reference model.
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    if good.sum() < max(min_rows, P.shape[1] + 2):
        raise ValueError(f"{element}: only {int(good.sum())} usable rows (need >= {min_rows})")

    center = np.nanmedian(P[good], axis=0)
    scale = np.array([robust_sigma(P[good, i]) for i in range(P.shape[1])])
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

    Ps = (P - center) / scale
    X, names = poly_design(Ps, degree=degree)

    keep = good.copy()
    coeffs = np.zeros(X.shape[1])
    sigma = float("nan")
    for _ in range(max(1, n_iter)):
        coeffs, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        resid = y - X @ coeffs
        sigma = robust_sigma(resid[keep])
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_keep = good & (np.abs(resid - np.nanmedian(resid[keep])) < clip * sigma)
        if new_keep.sum() < max(min_rows, X.shape[1] + 2):
            break
        if bool(np.array_equal(new_keep, keep)):
            keep = new_keep
            break
        keep = new_keep

    return ElementFit(
        element=element,
        term_names=names,
        coeffs=coeffs,
        center=center,
        scale=scale,
        degree=degree,
        n_fit=int(keep.sum()),
        n_clipped=int(good.sum() - keep.sum()),
        robust_scatter=float(sigma),
        predictor_names=list(predictor_names or []),
    )


# ---------------------------------------------------------------------------
# Empirical scatter as a function of (SNR, Teff)
# ---------------------------------------------------------------------------
DEFAULT_SNR_EDGES: tuple[float, ...] = (0.0, 30.0, 50.0, 80.0, 120.0, 200.0, 1e9)
DEFAULT_TEFF_EDGES: tuple[float, ...] = (2800.0, 3800.0, 4300.0, 4700.0, 5100.0, 5500.0, 6000.0)


@dataclass
class ScatterTable:
    """Robust residual width on a (SNR, Teff) grid, with graceful fallbacks."""

    element: str
    snr_edges: np.ndarray
    teff_edges: np.ndarray
    sigma: np.ndarray  # (n_snr, n_teff)
    counts: np.ndarray
    sigma_teff: np.ndarray  # marginal over SNR, per Teff bin
    sigma_global: float
    floor: float

    def sigma_for(self, snr: np.ndarray, teff: np.ndarray) -> np.ndarray:
        """Look up sigma per star, falling back Teff-marginal then global."""
        snr = np.asarray(snr, dtype=float)
        teff = np.asarray(teff, dtype=float)
        i = np.clip(np.digitize(snr, self.snr_edges) - 1, 0, self.sigma.shape[0] - 1)
        j = np.clip(np.digitize(teff, self.teff_edges) - 1, 0, self.sigma.shape[1] - 1)
        out = self.sigma[i, j]
        bad = ~np.isfinite(out)
        if bad.any():
            out = np.where(bad, self.sigma_teff[j], out)
        bad = ~np.isfinite(out)
        if bad.any():
            out = np.where(bad, self.sigma_global, out)
        return np.maximum(out, self.floor)


def build_scatter_table(
    resid: np.ndarray,
    snr: np.ndarray,
    teff: np.ndarray,
    *,
    element: str = "X",
    snr_edges: tuple[float, ...] = DEFAULT_SNR_EDGES,
    teff_edges: tuple[float, ...] = DEFAULT_TEFF_EDGES,
    min_count: int = 40,
    floor: float = 0.005,
) -> ScatterTable:
    """Measure the robust residual width in each (SNR, Teff) cell.

    Cells with fewer than ``min_count`` stars are left NaN and resolved at
    lookup time by the Teff-marginal then the global width — a thin cell must
    never produce an optimistically small denominator.
    """
    resid = np.asarray(resid, dtype=float)
    snr = np.asarray(snr, dtype=float)
    teff = np.asarray(teff, dtype=float)
    se = np.asarray(snr_edges, dtype=float)
    te = np.asarray(teff_edges, dtype=float)
    ns, nt = len(se) - 1, len(te) - 1

    i = np.clip(np.digitize(snr, se) - 1, 0, ns - 1)
    j = np.clip(np.digitize(teff, te) - 1, 0, nt - 1)
    finite = np.isfinite(resid) & np.isfinite(snr) & np.isfinite(teff)

    sigma = np.full((ns, nt), np.nan)
    counts = np.zeros((ns, nt), dtype=int)
    for a in range(ns):
        for b in range(nt):
            sel = finite & (i == a) & (j == b)
            counts[a, b] = int(sel.sum())
            if counts[a, b] >= min_count:
                sigma[a, b] = robust_sigma(resid[sel])

    sigma_teff = np.full(nt, np.nan)
    for b in range(nt):
        sel = finite & (j == b)
        if sel.sum() >= min_count:
            sigma_teff[b] = robust_sigma(resid[sel])

    g = robust_sigma(resid[finite])
    return ScatterTable(
        element=element,
        snr_edges=se,
        teff_edges=te,
        sigma=sigma,
        counts=counts,
        sigma_teff=sigma_teff,
        sigma_global=float(g if np.isfinite(g) else floor),
        floor=float(floor),
    )


# ---------------------------------------------------------------------------
# The manifold
# ---------------------------------------------------------------------------
@dataclass
class Manifold:
    """Per-element fits plus per-element empirical scatter tables."""

    elements: list[str]
    fits: dict[str, ElementFit]
    scatter: dict[str, ScatterTable]
    abund_prefix: str
    teff_col: str
    logg_col: str
    feh_col: str
    snr_col: str
    degree: int
    meta: dict = field(default_factory=dict)

    def to_summary(self) -> list[dict]:
        rows = []
        for el in self.elements:
            f = self.fits[el]
            s = self.scatter[el]
            rows.append(
                {
                    "element": el,
                    "family": element_family(el),
                    "n_fit": f.n_fit,
                    "n_clipped": f.n_clipped,
                    "robust_scatter_dex": round(float(f.robust_scatter), 5),
                    "sigma_global_dex": round(float(s.sigma_global), 5),
                    "sigma_best_cell_dex": (
                        round(float(np.nanmin(s.sigma)), 5) if np.isfinite(s.sigma).any() else None
                    ),
                }
            )
        return rows


def predictor_block(
    df: pd.DataFrame,
    element: str,
    *,
    abund_prefix: str,
    teff_col: str,
    logg_col: str,
    feh_col: str,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the four predictors for one element, with a leave-one-out alpha proxy."""
    ap = alpha_proxy(df, exclude=element, prefix=abund_prefix)
    P = np.column_stack(
        [
            df[feh_col].to_numpy(dtype=float),
            df[teff_col].to_numpy(dtype=float) / 1000.0,
            df[logg_col].to_numpy(dtype=float),
            ap,
        ]
    )
    return P, [feh_col, f"{teff_col}/1000", logg_col, "alpha_proxy_loo"]


def fit_manifold(
    df: pd.DataFrame,
    elements: list[str],
    *,
    abund_prefix: str = "",
    teff_col: str = "teff",
    logg_col: str = "logg",
    feh_col: str = "fe_h",
    snr_col: str = "snr",
    degree: int = 2,
    clip: float = 4.0,
    n_iter: int = 4,
    min_rows: int = 50,
    snr_edges: tuple[float, ...] = DEFAULT_SNR_EDGES,
    teff_edges: tuple[float, ...] = DEFAULT_TEFF_EDGES,
    min_count: int = 40,
    floor: float = 0.005,
) -> Manifold:
    """Fit the natural manifold and the empirical scatter for every element."""
    fits: dict[str, ElementFit] = {}
    scat: dict[str, ScatterTable] = {}
    used: list[str] = []
    skipped: dict[str, str] = {}

    snr = df[snr_col].to_numpy(dtype=float) if snr_col in df.columns else np.full(len(df), 100.0)
    teff = df[teff_col].to_numpy(dtype=float)

    for el in elements:
        col = f"{abund_prefix}{el}"
        if col not in df.columns:
            skipped[el] = "column absent"
            continue
        P, pnames = predictor_block(
            df,
            el,
            abund_prefix=abund_prefix,
            teff_col=teff_col,
            logg_col=logg_col,
            feh_col=feh_col,
        )
        y = df[col].to_numpy(dtype=float)
        try:
            f = fit_element(
                P,
                y,
                element=el,
                degree=degree,
                clip=clip,
                n_iter=n_iter,
                min_rows=min_rows,
                predictor_names=pnames,
            )
        except ValueError as exc:
            skipped[el] = str(exc)
            continue
        r = y - f.predict(P)
        scat[el] = build_scatter_table(
            r,
            snr,
            teff,
            element=el,
            snr_edges=snr_edges,
            teff_edges=teff_edges,
            min_count=min_count,
            floor=floor,
        )
        fits[el] = f
        used.append(el)

    return Manifold(
        elements=used,
        fits=fits,
        scatter=scat,
        abund_prefix=abund_prefix,
        teff_col=teff_col,
        logg_col=logg_col,
        feh_col=feh_col,
        snr_col=snr_col,
        degree=degree,
        meta={"n_stars": int(len(df)), "skipped": skipped},
    )


def residuals(df: pd.DataFrame, manifold: Manifold) -> pd.DataFrame:
    """Residual ``r_X`` of every element against the fitted manifold, in dex."""
    out = {}
    for el in manifold.elements:
        col = f"{manifold.abund_prefix}{el}"
        P, _ = predictor_block(
            df,
            el,
            abund_prefix=manifold.abund_prefix,
            teff_col=manifold.teff_col,
            logg_col=manifold.logg_col,
            feh_col=manifold.feh_col,
        )
        out[el] = df[col].to_numpy(dtype=float) - manifold.fits[el].predict(P)
    return pd.DataFrame(out, index=df.index)


def zscores(
    df: pd.DataFrame,
    manifold: Manifold,
    *,
    err_prefix: str | None = None,
    use_reported_errors: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardised residuals ``z_X`` and the sigma actually used, per element.

    ``sigma_used = max(sigma_empirical(SNR, Teff), sigma_reported)`` — the
    empirical width is the floor because catalogue errors are known to be
    optimistic, and the reported error takes over when a particular fit was bad.
    """
    r = residuals(df, manifold)
    snr = (
        df[manifold.snr_col].to_numpy(dtype=float)
        if manifold.snr_col in df.columns
        else np.full(len(df), 100.0)
    )
    teff = df[manifold.teff_col].to_numpy(dtype=float)

    z = {}
    sig = {}
    for el in manifold.elements:
        s = manifold.scatter[el].sigma_for(snr, teff)
        if use_reported_errors and err_prefix is not None:
            ecol = f"{err_prefix}{el}"
            if ecol in df.columns:
                e = df[ecol].to_numpy(dtype=float)
                s = np.where(np.isfinite(e), np.maximum(s, e), s)
        sig[el] = s
        z[el] = r[el].to_numpy(dtype=float) / s
    return pd.DataFrame(z, index=df.index), pd.DataFrame(sig, index=df.index)


__all__ = [
    "ALPHA_ELEMENTS",
    "DEFAULT_SNR_EDGES",
    "DEFAULT_TEFF_EDGES",
    "NATURALLY_SPARSE_ELEMENTS",
    "NUCLEO_FAMILIES",
    "ElementFit",
    "Manifold",
    "ScatterTable",
    "alpha_proxy",
    "build_scatter_table",
    "element_family",
    "family_members",
    "fit_element",
    "fit_manifold",
    "poly_design",
    "predictor_block",
    "residuals",
    "robust_sigma",
    "zscores",
]

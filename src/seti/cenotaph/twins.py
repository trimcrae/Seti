"""Parameter twins: a differential absolute magnitude that cancels pipelines.

Why not isochrones
------------------
The obvious estimator — predict ``M`` from an isochrone at the spectroscopic
(Teff, logg, [Fe/H]) and difference — is limited by Teff systematics, not by
photon noise. On the lower main sequence ``dM_G/dTeff ≈ −0.0022 mag/K``, so a
100 K zero-point error between the survey and the isochrone scale injects
0.22 mag: *twice* the entire signal at f = 0.10. There is no version of that
estimator that is safe at the 0.1-mag level.

The twin estimator
------------------
Compare a star only to the ≳50 stars nearest to it in
``(Teff, log g, [M/H], [α/Fe])``, measured by the *same pipeline*, and take the
statistic to be

    ΔM_Ks = M_Ks(target) − median{ M_Ks(twins) }

Every pipeline systematic that is a smooth function of the parameters —
Teff scale, log g scale, the [Fe/H] scale, the model atmospheres, the very
grid — is shared by the target and its twins and cancels to first order in the
difference. What survives is the local intrinsic spread, which is measured
from the twins themselves rather than assumed.

Two properties make the *negative* tail the clean one:

* **Metal-poor subdwarfs cannot leak in.** A subdwarf is underluminous relative
  to solar-metallicity stars, but [M/H] is one of the matching axes, so its
  twins are subdwarfs too and the deficit cancels by construction. (What must
  still be caught is the *edge* of the metallicity distribution, where a star
  has no symmetric twin set — see ``param_edge``.)
* **Unresolved binaries scatter the wrong way.** Adding an unseen companion
  makes a star *brighter*, up to 0.75 mag for an equal-mass pair. The
  overluminous tail is a binary sequence; the underluminous tail is not.

The dangerous residual binary case is a *hot* companion (white dwarf, sdB):
it biases the composite Teff upward, so the star is compared to hotter,
brighter twins and appears underluminous. That one is caught in ``vet.py`` by
its ultraviolet excess and by the SED goodness-of-fit, not here.

Band choice
-----------
``Ks`` (or ``W1``) is the luminosity band because ``A_Ks/A_V = 0.078``: for
``A_V < 0.3`` the extinction term is under 0.025 mag, so the statistic cannot
be manufactured by a dust column that the 3D maps missed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Matching scales: roughly the per-star measurement precision of Gaia GSP-Spec /
# LAMOST / APOGEE. A scaled distance of 1 means "different by about one
# measurement error", i.e. genuinely indistinguishable stars.
DEFAULT_SCALES: dict[str, float] = {
    "teff": 100.0,      # K
    "logg": 0.15,       # dex
    "mh": 0.10,         # dex
    "alphafe": 0.05,    # dex
}

DEFAULT_PARAMS: tuple[str, ...] = ("teff", "logg", "mh", "alphafe")

MAX_SCALED_DIST: float = 1.5
"""Twins farther than this in quadrature-scaled parameter space are not twins."""

PARAM_EDGE_TOL: float = 0.5
"""Max tolerated mean scaled offset of the twin set from the target, per axis."""


@dataclass
class TwinConfig:
    params: tuple[str, ...] = DEFAULT_PARAMS
    scales: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCALES))
    n_twins: int = 50
    n_twins_min: int = 20
    max_scaled_dist: float = MAX_SCALED_DIST
    param_edge_tol: float = PARAM_EDGE_TOL
    band: str = "ks"
    chunk: int = 20_000


def absolute_magnitude(mag: np.ndarray, parallax_mas: np.ndarray) -> np.ndarray:
    """``M = m + 5 log10(ϖ[mas]) − 10``. Non-positive parallax -> NaN."""
    mag = np.asarray(mag, dtype=float)
    plx = np.asarray(parallax_mas, dtype=float)
    out = np.full(mag.shape, np.nan)
    ok = np.isfinite(mag) & np.isfinite(plx) & (plx > 0)
    out[ok] = mag[ok] + 5.0 * np.log10(plx[ok]) - 10.0
    return out


def distance_modulus_sigma(parallax_mas, parallax_error_mas) -> np.ndarray:
    """1σ on the distance modulus, ``5/ln10 · σ_ϖ/ϖ`` — the grey-degenerate term."""
    plx = np.asarray(parallax_mas, dtype=float)
    err = np.asarray(parallax_error_mas, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (5.0 / math.log(10.0)) * np.abs(err / plx)


def _scaled_matrix(df: pd.DataFrame, cfg: TwinConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the scaled parameter matrix and a finite-row mask."""
    cols = []
    finite = np.ones(len(df), dtype=bool)
    for p in cfg.params:
        if p not in df.columns:
            raise KeyError(f"twin parameter {p!r} missing from the table")
        v = pd.to_numeric(df[p], errors="coerce").to_numpy(float)
        finite &= np.isfinite(v)
        cols.append(v / float(cfg.scales[p]))
    return np.column_stack(cols), finite


def twin_statistics(df: pd.DataFrame, cfg: TwinConfig | None = None,
                    mag_col: str | None = None,
                    verbose: bool = True) -> pd.DataFrame:
    """Compute the twin-differential absolute magnitude for every row.

    ``df`` must carry the twin parameters (``cfg.params``), ``parallax``,
    ``parallax_error`` and the luminosity magnitude column (default
    ``<band>_mag``, e.g. ``ks_mag``).

    Returns a frame aligned to ``df.index`` with:

    ``m_abs``            the star's own absolute magnitude in the band
    ``dm_twin``          ``M − median(M_twins)`` — the headline statistic
    ``dm_twin_err``      1σ including twin spread, median error, parallax, photometry
    ``z_twin``           ``dm_twin / dm_twin_err``, positive = underluminous
    ``dm_local``         the same, against a *local linear* fit in parameter
                         space (removes first-order edge bias)
    ``n_twins``          twins actually used
    ``twin_scatter``     robust σ about the *local linear* model (the true
                         irreducible spread; this is the sensitivity floor)
    ``twin_scatter_median`` robust σ about the twin median — larger, because it
                         also contains the parameter gradient across the box
    ``z_local``          the local-linear statistic in σ units
    ``param_edge``       twin set is lopsided in ≥1 parameter -> gradient bias
    ``twin_verdict``     ``ok`` | ``no_twins`` | ``bad_input``
    """
    from scipy.spatial import cKDTree

    cfg = cfg or TwinConfig()
    mag_col = mag_col or f"{cfg.band}_mag"
    n = len(df)

    out = pd.DataFrame(index=df.index)
    out["m_abs"] = np.nan
    out["dm_twin"] = np.nan
    out["dm_twin_err"] = np.nan
    out["z_twin"] = np.nan
    out["dm_local"] = np.nan
    out["n_twins"] = 0
    out["twin_scatter"] = np.nan
    out["param_edge"] = False
    out["twin_verdict"] = "bad_input"

    if n == 0:
        return out

    xs, finite_par = _scaled_matrix(df, cfg)
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(float)
    mag_err = pd.to_numeric(
        df.get(f"{cfg.band}_mag_error", pd.Series(np.full(n, 0.02))), errors="coerce"
    ).to_numpy(float)
    mag_err = np.where(np.isfinite(mag_err), mag_err, 0.02)
    plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
    plx_err = pd.to_numeric(df["parallax_error"], errors="coerce").to_numpy(float)

    m_abs = absolute_magnitude(mag, plx)
    sig_mu = distance_modulus_sigma(plx, plx_err)
    out["m_abs"] = m_abs

    usable = finite_par & np.isfinite(m_abs)
    idx_usable = np.flatnonzero(usable)
    if idx_usable.size < cfg.n_twins_min + 1:
        if verbose:
            print(f"[cenotaph.twins] only {idx_usable.size} usable rows; "
                  "cannot build a twin population")
        return out

    tree = cKDTree(xs[idx_usable])
    k = min(cfg.n_twins + 1, idx_usable.size)

    dm_twin = np.full(n, np.nan)
    dm_err = np.full(n, np.nan)
    dm_local = np.full(n, np.nan)
    n_tw = np.zeros(n, dtype=int)
    scat = np.full(n, np.nan)
    scat_med = np.full(n, np.nan)
    edge = np.zeros(n, dtype=bool)
    verdict = np.array(["bad_input"] * n, dtype=object)

    m_abs_u = m_abs[idx_usable]
    xs_u = xs[idx_usable]

    for start in range(0, idx_usable.size, cfg.chunk):
        block = idx_usable[start:start + cfg.chunk]
        dists, nb = tree.query(xs[block], k=k, workers=-1)
        dists = np.atleast_2d(dists)
        nb = np.atleast_2d(nb)

        for row, gi in enumerate(block):
            d_row = dists[row]
            n_row = nb[row]
            # Drop self (the exact zero-distance match to its own tree index).
            self_pos = start + row
            keep = n_row != self_pos
            d_row, n_row = d_row[keep], n_row[keep]
            keep = np.isfinite(d_row) & (d_row <= cfg.max_scaled_dist)
            n_row = n_row[keep]
            d_row = d_row[keep]
            n_tw[gi] = n_row.size
            if n_row.size < cfg.n_twins_min:
                verdict[gi] = "no_twins"
                continue

            mt = m_abs_u[n_row]
            med = float(np.median(mt))
            dm = float(m_abs[gi] - med)

            # Lopsidedness of the twin cloud: a star at the edge of the
            # parameter distribution has twins on one side only, so the median
            # carries a gradient bias. Flag it and correct it locally.
            dx = xs_u[n_row] - xs[gi]
            mean_off = dx.mean(axis=0)
            edge[gi] = bool(np.any(np.abs(mean_off) > cfg.param_edge_tol))

            # Local linear model M = a0 + sum_p a_p * (x_p - x_p,target).
            # Evaluated at the target, the intercept a0 is the twin prediction
            # with the first-order gradient bias removed.
            resid = mt - med
            try:
                design = np.column_stack([np.ones(n_row.size), dx])
                coef, *_ = np.linalg.lstsq(design, mt, rcond=None)
                dm_local[gi] = float(m_abs[gi] - coef[0])
                resid = mt - design @ coef
            except np.linalg.LinAlgError:
                dm_local[gi] = dm

            # The irreducible spread is the scatter about the *local linear
            # model*, not about the median. Scatter about the median also
            # contains the parameter gradient across the twin box, which is a
            # deterministic trend that the local fit removes; using it would
            # inflate every error bar by the box width times dM/dTeff and throw
            # away most of the channel's sensitivity.
            sigma_int = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
            if not np.isfinite(sigma_int) or sigma_int <= 0:
                sigma_int = float(np.std(resid)) or 0.05
            sigma_med = 1.4826 * float(np.median(np.abs(mt - med)))

            # Target draws once from the intrinsic spread; the twin prediction
            # itself is uncertain by sigma_int/sqrt(N); the distance modulus is
            # grey; the photometry adds its own error.
            err = math.sqrt(
                sigma_int**2 * (1.0 + 1.0 / n_row.size)
                + sig_mu[gi] ** 2
                + mag_err[gi] ** 2
            )
            dm_twin[gi] = dm
            dm_err[gi] = err
            scat[gi] = sigma_int
            scat_med[gi] = sigma_med
            verdict[gi] = "ok"

        if verbose and (start // cfg.chunk) % 20 == 0:
            print(f"[cenotaph.twins] {min(start + cfg.chunk, idx_usable.size)}"
                  f"/{idx_usable.size} stars", flush=True)

    out["dm_twin"] = dm_twin
    out["dm_twin_err"] = dm_err
    with np.errstate(divide="ignore", invalid="ignore"):
        out["z_twin"] = dm_twin / dm_err
    out["dm_local"] = dm_local
    out["n_twins"] = n_tw
    out["twin_scatter"] = scat
    out["twin_scatter_median"] = scat_med
    with np.errstate(divide="ignore", invalid="ignore"):
        out["z_local"] = dm_local / dm_err
    out["param_edge"] = edge
    out["twin_verdict"] = verdict
    return out


def band_residuals(df: pd.DataFrame, twins: pd.DataFrame, bands: list[str],
                   cfg: TwinConfig | None = None) -> pd.DataFrame:
    """Per-band absolute-magnitude residuals ``ΔM_b`` — the greyfit input.

    Built as ``ΔM_b = ΔM_ref + (colour excess relative to the twin median)``,
    which is algebraically identical to differencing per-band absolute
    magnitudes but keeps the *reference-band* statistic and the *colour*
    statistic explicit — the first carries the grey signal and the distance
    degeneracy, the second carries the reddening and is distance-immune.

    Requires that ``<band>_col_med`` (twin-median colour ``b − ref``) has been
    attached; :func:`twin_colour_medians` does that.
    """
    cfg = cfg or TwinConfig()
    ref = cfg.band
    out = pd.DataFrame(index=df.index)
    for b in bands:
        if b == ref:
            out[f"dm_{b}"] = twins["dm_twin"]
            continue
        col_obs = (pd.to_numeric(df.get(f"{b}_mag"), errors="coerce")
                   - pd.to_numeric(df.get(f"{ref}_mag"), errors="coerce"))
        col_med = pd.to_numeric(df.get(f"{b}_col_med"), errors="coerce")
        out[f"dm_{b}"] = twins["dm_twin"] + (col_obs - col_med)
    return out


def twin_colour_medians(df: pd.DataFrame, cfg: TwinConfig | None = None,
                        bands: list[str] | None = None,
                        verbose: bool = True) -> pd.DataFrame:
    """Twin-median intrinsic colours ``median{b − ref}`` for each star.

    Separate pass from :func:`twin_statistics` so that a star missing (say)
    GALEX still gets its Ks statistic; only the bands it has are fitted.
    """
    from scipy.spatial import cKDTree

    cfg = cfg or TwinConfig()
    ref = cfg.band
    bands = [b for b in (bands or []) if b != ref]
    out = pd.DataFrame(index=df.index)
    for b in bands:
        out[f"{b}_col_med"] = np.nan
        out[f"{b}_col_scatter"] = np.nan
    if not bands:
        return out

    xs, finite_par = _scaled_matrix(df, cfg)
    ref_mag = pd.to_numeric(df.get(f"{ref}_mag"), errors="coerce").to_numpy(float)
    idx_usable = np.flatnonzero(finite_par & np.isfinite(ref_mag))
    if idx_usable.size < cfg.n_twins_min + 1:
        return out

    tree = cKDTree(xs[idx_usable])
    k = min(cfg.n_twins + 1, idx_usable.size)

    for b in bands:
        bm = pd.to_numeric(df.get(f"{b}_mag"), errors="coerce").to_numpy(float)
        colour = bm - ref_mag
        col_u = colour[idx_usable]
        med = np.full(len(df), np.nan)
        sca = np.full(len(df), np.nan)
        for start in range(0, idx_usable.size, cfg.chunk):
            block = idx_usable[start:start + cfg.chunk]
            dists, nb = tree.query(xs[block], k=k, workers=-1)
            dists, nb = np.atleast_2d(dists), np.atleast_2d(nb)
            for row, gi in enumerate(block):
                sel = nb[row][(nb[row] != start + row)
                              & (dists[row] <= cfg.max_scaled_dist)]
                vals = col_u[sel]
                vals = vals[np.isfinite(vals)]
                if vals.size < cfg.n_twins_min:
                    continue
                m = float(np.median(vals))
                med[gi] = m
                sca[gi] = 1.4826 * float(np.median(np.abs(vals - m)))
        out[f"{b}_col_med"] = med
        out[f"{b}_col_scatter"] = sca
        if verbose:
            print(f"[cenotaph.twins] colour medians done for {b}: "
                  f"{int(np.isfinite(med).sum())} stars", flush=True)
    return out


__all__ = [
    "DEFAULT_PARAMS",
    "DEFAULT_SCALES",
    "TwinConfig",
    "absolute_magnitude",
    "band_residuals",
    "distance_modulus_sigma",
    "twin_colour_medians",
    "twin_statistics",
]

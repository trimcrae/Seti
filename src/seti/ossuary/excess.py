"""Photosphere-anchored infrared excess with propagated errors.

The problem with metal-poor stars
---------------------------------
Every published infrared-excess search predicts the photosphere from a synthetic
stellar atmosphere at solar (or fitted) metallicity.  For a sample deliberately
selected to be metal-poor, that model is being extrapolated exactly where it is
least calibrated, and a systematic colour offset of a few hundredths of a
magnitude -- entirely plausible at [Fe/H] < -1 -- would manufacture a false
excess in *every* star at once.

So this channel does not use a synthetic photosphere at all.  It builds the
photosphere locus **empirically from the sample itself**: the robust median
``Ks - W1``, ``Ks - W2``, ``Ks - W3`` colour as a function of ``BP - RP``,
computed in colour bins, optionally split by metallicity bin.  An excess is then
a *redward outlier from the metal-poor stars' own locus*, and the significance
is measured against the empirical star-to-star scatter rather than an assumed
model error.  Any metallicity-dependent colour offset is absorbed, not assumed
away.  The median has a 50% breakdown point, so the (rare) genuine excesses
cannot drag the locus they are measured against.

Anchoring on Ks, not on G, is deliberate twice over:

* Ks (2.16 um) is the longest-wavelength band still overwhelmingly photospheric
  for a >4000 K dwarf, so it fixes the flux scale with the shortest lever arm to
  W1 -- the extrapolation error is minimal.
* An unresolved cool companion -- the classic false excess -- contributes at Ks
  too, which *inflates* the anchor and therefore *suppresses* the inferred W1
  excess.  The anchor choice is conservative against the dominant confounder.

The W1-W2 colour excess is computed directly from W1 and W2, so the anchor's
error cancels out of it entirely.  That is the statistic the contamination ledger
requires a detection to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..photometry import (
    BANDS,
    band_freq_hz,
    mag_err_to_flux_err_jy,
    mag_to_flux_jy,
    planck_bnu,
)

_SIGMA_SB = 5.670374419e-8          # W m^-2 K^-4
_MAD_TO_SIGMA = 1.4826


@dataclass
class ColourLocus:
    """Empirical ``anchor - band`` colour vs a driving colour, in bins."""

    band: str
    anchor: str
    centres: np.ndarray = field(default_factory=lambda: np.array([]))
    median: np.ndarray = field(default_factory=lambda: np.array([]))
    scatter: np.ndarray = field(default_factory=lambda: np.array([]))
    counts: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_bins(self) -> int:
        return int(len(self.centres))

    def predict(self, colour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate (colour, scatter) at ``colour``; NaN outside the fit range.

        Refusing to extrapolate is the point: a star bluer or redder than any
        well-populated bin has no empirical photosphere and must not be assigned
        one.  It leaves the funnel as ``no_locus``, not as a candidate.
        """
        c = np.asarray(colour, float)
        if self.n_bins == 0:
            nan = np.full(c.shape, np.nan)
            return nan, nan
        lo, hi = self.centres.min(), self.centres.max()
        med = np.interp(c, self.centres, self.median)
        sca = np.interp(c, self.centres, self.scatter)
        outside = ~np.isfinite(c) | (c < lo) | (c > hi)
        med = np.where(outside, np.nan, med)
        sca = np.where(outside, np.nan, sca)
        return med, sca

    def to_dict(self) -> dict:
        return {"band": self.band, "anchor": self.anchor,
                "n_bins": self.n_bins,
                "centres": [float(x) for x in self.centres],
                "median": [float(x) for x in self.median],
                "scatter": [float(x) for x in self.scatter],
                "counts": [int(x) for x in self.counts]}


def fit_colour_locus(df: pd.DataFrame, band: str, anchor: str = "Ks",
                     colour_col: str = "bp_rp", *, bin_width: float = 0.10,
                     min_per_bin: int = 25, scatter_floor: float = 0.02,
                     clip_sigma: float = 3.0, n_iter: int = 2) -> ColourLocus:
    """Robust running median of ``anchor - band`` vs ``colour_col``.

    Two sigma-clipping iterations remove the redward tail (the excesses
    themselves, plus blends) from the locus they are measured against, while the
    median keeps the fit from moving even before clipping.
    """
    a = pd.to_numeric(df.get(f"{anchor}mag"), errors="coerce").to_numpy(float)
    b = pd.to_numeric(df.get(f"{band}mag"), errors="coerce").to_numpy(float)
    c = pd.to_numeric(df.get(colour_col), errors="coerce").to_numpy(float)
    y = a - b
    ok = np.isfinite(y) & np.isfinite(c)
    if not ok.any():
        return ColourLocus(band=band, anchor=anchor)

    lo = np.floor(np.nanmin(c[ok]) / bin_width) * bin_width
    hi = np.ceil(np.nanmax(c[ok]) / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.array([lo, lo + bin_width])

    centres, meds, scas, counts = [], [], [], []
    idx = np.digitize(c, edges) - 1
    for k in range(len(edges) - 1):
        sel = ok & (idx == k)
        if sel.sum() < min_per_bin:
            continue
        vals = y[sel]
        for _ in range(n_iter):
            m = np.median(vals)
            s = _MAD_TO_SIGMA * np.median(np.abs(vals - m))
            s = max(float(s), scatter_floor)
            keep = np.abs(vals - m) <= clip_sigma * s
            if keep.sum() < max(5, min_per_bin // 3):
                break
            vals = vals[keep]
        m = float(np.median(vals))
        s = max(float(_MAD_TO_SIGMA * np.median(np.abs(vals - m))), scatter_floor)
        centres.append(0.5 * (edges[k] + edges[k + 1]))
        meds.append(m)
        scas.append(s)
        counts.append(int(sel.sum()))

    return ColourLocus(band=band, anchor=anchor,
                       centres=np.asarray(centres, float),
                       median=np.asarray(meds, float),
                       scatter=np.asarray(scas, float),
                       counts=np.asarray(counts, int))


def fit_loci(df: pd.DataFrame, cfg: dict, bands=("W1", "W2", "W3", "W4"),
             anchor: str = "Ks", colour_col: str = "bp_rp") -> dict:
    """Fit one :class:`ColourLocus` per band from the (clean) reference sample."""
    return {b: fit_colour_locus(df, b, anchor=anchor, colour_col=colour_col,
                                bin_width=cfg["locus_color_bin"],
                                min_per_bin=cfg["locus_min_per_bin"],
                                scatter_floor=cfg["locus_scatter_floor_mag"])
            for b in bands}


def _mag_err(df: pd.DataFrame, band: str, default: float = 0.05) -> np.ndarray:
    e = pd.to_numeric(df.get(f"e_{band}mag"), errors="coerce").to_numpy(float)
    return np.where(np.isfinite(e) & (e > 0), e, default)


def compute_excess(df: pd.DataFrame, loci: dict, cfg: dict, *,
                   anchor: str = "Ks", colour_col: str = "bp_rp") -> pd.DataFrame:
    """Per-band predicted photosphere, excess flux, and significance.

    Adds, for every band in ``loci``: ``{b}_pred_mag``, ``{b}_pred_jy``,
    ``{b}_obs_jy``, ``{b}_excess_jy``, ``{b}_excess_err_jy``, ``chi_{b}`` and
    ``{b}_excess_mag``.  Also the anchor-independent colour excesses
    ``w1_w2_excess`` / ``chi_w1_w2`` and ``w1_w3_excess`` / ``chi_w1_w3``.
    """
    out = df.copy()
    sys_floor = float(cfg["sys_floor_mag"])
    colour = pd.to_numeric(out.get(colour_col), errors="coerce").to_numpy(float)
    a_mag = pd.to_numeric(out.get(f"{anchor}mag"), errors="coerce").to_numpy(float)
    a_err = _mag_err(out, anchor, default=0.03)

    pred_colour, pred_scatter = {}, {}
    for b, loc in loci.items():
        med, sca = loc.predict(colour)
        pred_colour[b], pred_scatter[b] = med, sca

        pred_mag = a_mag - med
        pred_mag_err = np.sqrt(a_err ** 2 + np.nan_to_num(sca, nan=np.inf) ** 2
                               + sys_floor ** 2)
        obs_mag = pd.to_numeric(out.get(f"{b}mag"), errors="coerce").to_numpy(float)
        obs_err = _mag_err(out, b)

        pred_jy = mag_to_flux_jy(pred_mag, b)
        pred_jy_err = mag_err_to_flux_err_jy(pred_mag, pred_mag_err, b)
        obs_jy = mag_to_flux_jy(obs_mag, b)
        obs_jy_err = mag_err_to_flux_err_jy(obs_mag, obs_err, b)

        exc = obs_jy - pred_jy
        exc_err = np.sqrt(obs_jy_err ** 2 + pred_jy_err ** 2)

        out[f"{b}_pred_mag"] = pred_mag
        out[f"{b}_pred_jy"] = pred_jy
        out[f"{b}_obs_jy"] = obs_jy
        out[f"{b}_excess_jy"] = exc
        out[f"{b}_excess_err_jy"] = exc_err
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"chi_{b}"] = exc / exc_err
            # Excess expressed in magnitudes of the photosphere (interpretable).
            out[f"{b}_excess_mag"] = -2.5 * np.log10(
                np.where(pred_jy > 0, np.maximum(obs_jy, 1e-30) / pred_jy, np.nan))

    # Anchor-independent colour excesses.  Ks drops out of a W1-W2 difference,
    # so these do not inherit the anchor's error -- and the ledger requires the
    # detection to appear here, in the star-dominated bands.
    for c1, c2, name in (("W1", "W2", "w1_w2"), ("W1", "W3", "w1_w3")):
        if c1 not in loci or c2 not in loci:
            continue
        m1 = pd.to_numeric(out.get(f"{c1}mag"), errors="coerce").to_numpy(float)
        m2 = pd.to_numeric(out.get(f"{c2}mag"), errors="coerce").to_numpy(float)
        e1, e2 = _mag_err(out, c1), _mag_err(out, c2)
        # locus colour (c1 - c2) = (anchor - c2) - (anchor - c1)
        loc_colour = pred_colour[c2] - pred_colour[c1]
        loc_err = np.hypot(np.nan_to_num(pred_scatter[c1], nan=np.inf),
                           np.nan_to_num(pred_scatter[c2], nan=np.inf))
        obs_colour = m1 - m2
        err = np.sqrt(e1 ** 2 + e2 ** 2 + loc_err ** 2
                      + (np.sqrt(2.0) * sys_floor) ** 2)
        out[f"{name}_obs"] = obs_colour
        out[f"{name}_locus"] = loc_colour
        out[f"{name}_excess"] = obs_colour - loc_colour
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"chi_{name}"] = (obs_colour - loc_colour) / err
    return out


# --------------------------------------------------------------------------
# Warm-dust characterisation
# --------------------------------------------------------------------------

def _bb_flux_jy(temp_k: float, band: str, omega: float) -> float:
    """Blackbody flux density (Jy) in ``band`` for solid angle ``omega`` (sr)."""
    return float(omega * planck_bnu(temp_k, band_freq_hz(band)) * 1e26)


_COARSE_GRID = np.geomspace(50.0, 3000.0, 300)


def _model_matrix(bands, t_grid: np.ndarray) -> np.ndarray:
    """Blackbody flux density (Jy per sr) on a temperature grid: (n_T, n_bands)."""
    t = np.asarray(t_grid, float)
    return np.stack([planck_bnu(t, band_freq_hz(b)) * 1e26 for b in bands], axis=1)


def _fit_grid(f: np.ndarray, s: np.ndarray, model: np.ndarray, t_grid: np.ndarray):
    """Closed-form weighted fit of one blackbody scale per (draw, temperature).

    ``f`` and ``s`` are ``(n_draws, n_bands)``; ``model`` is ``(n_T, n_bands)``.
    For a linear scale the least-squares solution is ``omega = B / C`` with
    ``B = sum(f * model * w)`` and ``C = sum(model^2 * w)``, and the residual chi2 is
    ``A - B^2 / C`` with ``A = sum(f^2 * w)``.  Every draw and every temperature are
    evaluated in one matrix product, which is what turned a per-star cost of
    ~10^5 Python calls (the loop that timed out the first catalogue-scale run) into
    three array operations.  Returns ``(t_best, omega_best, chi2_best)`` per draw;
    draws whose best scale is not positive come back NaN.
    """
    w = 1.0 / s ** 2                                    # (n_draws, n_bands)
    a = (f ** 2 * w).sum(axis=1)                        # (n_draws,)
    b = (f * w) @ model.T                               # (n_draws, n_T)
    c = (model ** 2) @ w.T                              # (n_T, n_draws)
    c = c.T                                             # (n_draws, n_T)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = b / c
        chi2 = a[:, None] - b ** 2 / c
    chi2 = np.where((omega > 0) & np.isfinite(chi2), chi2, np.inf)
    k = np.argmin(chi2, axis=1)
    rows = np.arange(len(k))
    best_chi2 = chi2[rows, k]
    ok = np.isfinite(best_chi2)
    t_best = np.where(ok, t_grid[k], np.nan)
    om_best = np.where(ok, omega[rows, k], np.nan)
    return t_best, om_best, np.where(ok, best_chi2, np.nan)


def fit_excess_blackbody_draws(fluxes: np.ndarray, errors: np.ndarray, bands,
                               *, refine: bool = True):
    """Vectorised blackbody fit for many flux draws sharing one band set.

    ``fluxes``/``errors`` are ``(n_draws, n_bands)`` in Jy.  The coarse grid is
    scanned for every draw at once; when ``refine`` is set the first draw (the
    point estimate) is re-scanned on a fine grid around its coarse minimum so the
    grid spacing, not the data, cannot dominate the residual chi2.
    """
    f = np.asarray(fluxes, float)
    s = np.asarray(errors, float)
    model = _model_matrix(bands, _COARSE_GRID)
    t, om, chi2 = _fit_grid(f, s, model, _COARSE_GRID)
    if refine and len(t) and np.isfinite(t[0]):
        fine = np.linspace(t[0] / 1.05, t[0] * 1.05, 60)
        tf, omf, c2f = _fit_grid(f[:1], s[:1], _model_matrix(bands, fine), fine)
        if np.isfinite(c2f[0]) and c2f[0] < chi2[0]:
            t[0], om[0], chi2[0] = tf[0], omf[0], c2f[0]
    return t, om, chi2


def fit_excess_blackbody(fluxes_jy: dict, errors_jy: dict, *,
                         t_grid: np.ndarray | None = None) -> tuple[float, float, float]:
    """Least-squares single-temperature blackbody fit to the excess fluxes.

    Returns ``(T_dust_K, omega_sr, chi2)``.  With two bands the fit is exact; with
    three or four it is over-determined and the chi2 is a genuine goodness-of-fit
    (a large chi2 means the excess is not a single-temperature blackbody at all --
    which is itself diagnostic of a blend rather than a dust population).
    """
    bands = [b for b in fluxes_jy
             if np.isfinite(fluxes_jy[b]) and np.isfinite(errors_jy.get(b, np.nan))
             and errors_jy[b] > 0]
    if len(bands) < 2:
        return np.nan, np.nan, np.nan
    f = np.array([[fluxes_jy[b] for b in bands]], float)
    s = np.array([[errors_jy[b] for b in bands]], float)
    if t_grid is not None:
        grid = np.asarray(t_grid, float)
        t, om, chi2 = _fit_grid(f, s, _model_matrix(bands, grid), grid)
        return float(t[0]), float(om[0]), float(chi2[0])
    t, om, chi2 = fit_excess_blackbody_draws(f, s, bands, refine=True)
    return float(t[0]), float(om[0]), float(chi2[0])


def _star_solid_angle(anchor_mag: float, anchor_band: str, teff_k: float) -> float:
    """Stellar solid angle (sr) from an anchor magnitude and T_eff."""
    if not (np.isfinite(anchor_mag) and np.isfinite(teff_k) and teff_k > 0):
        return np.nan
    f_jy = float(mag_to_flux_jy(anchor_mag, anchor_band))
    b_nu = float(planck_bnu(teff_k, band_freq_hz(anchor_band))) * 1e26
    return f_jy / b_nu if b_nu > 0 else np.nan


def characterise(df: pd.DataFrame, cfg: dict, *, anchor: str = "Ks",
                 bands=("W1", "W2", "W3", "W4"),
                 rng: np.random.Generator | None = None,
                 mc_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Fit ``(T_dust, tau)`` per row with Monte-Carlo uncertainties.

    ``tau = L_dust / L_star`` from the ratio of blackbody bolometric outputs,
    ``(Omega_d T_d^4) / (Omega_* T_eff^4)``, so distance and radius cancel.  The
    MC redraws every excess flux from its own error and refits, giving honest
    16/84 percentiles rather than a formal curvature error -- which matters here
    because the blackbody fit is strongly non-linear near the low-T end.

    ``mc_mask`` (boolean, aligned to ``df``) restricts the Monte-Carlo to the rows
    where it can matter -- those still alive after the cheap catalogue gates.
    Every row gets the point estimate; a row outside the mask gets NaN
    percentiles rather than a guess.
    """
    rng = rng or np.random.default_rng(20260726)
    n_mc = int(cfg.get("mc_draws", 400))
    out = df.copy()
    n = len(out)
    do_mc = np.ones(n, bool) if mc_mask is None else np.asarray(mc_mask, bool)
    t_dust = np.full(n, np.nan)
    tau = np.full(n, np.nan)
    t_lo = np.full(n, np.nan)
    t_hi = np.full(n, np.nan)
    tau_lo = np.full(n, np.nan)
    tau_hi = np.full(n, np.nan)
    chi2 = np.full(n, np.nan)
    nb = np.zeros(n, int)

    a_mag = pd.to_numeric(out.get(f"{anchor}mag"), errors="coerce").to_numpy(float)
    teff = pd.to_numeric(out.get("teff"), errors="coerce").to_numpy(float)
    if not np.isfinite(teff).any():
        for alt in ("teff_gspspec", "teff_gspphot"):
            if alt in out.columns:
                teff = pd.to_numeric(out[alt], errors="coerce").to_numpy(float)
                if np.isfinite(teff).any():
                    break

    have = [b for b in bands
            if f"{b}_excess_jy" in out.columns and f"{b}_excess_err_jy" in out.columns]
    fx = {b: pd.to_numeric(out[f"{b}_excess_jy"], errors="coerce").to_numpy(float)
          for b in have}
    ex = {b: pd.to_numeric(out[f"{b}_excess_err_jy"], errors="coerce").to_numpy(float)
          for b in have}

    for i in range(n):
        # Only bands with a positive, significant excess constrain the dust; a
        # band consistent with zero would drag the fit toward an arbitrary T.
        use = [b for b in have
               if np.isfinite(fx[b][i]) and fx[b][i] > 0 and np.isfinite(ex[b][i])
               and ex[b][i] > 0 and fx[b][i] / ex[b][i] >= 1.0]
        nb[i] = len(use)
        if len(use) < 2:
            continue
        f0 = np.array([fx[b][i] for b in use], float)
        e0 = np.array([ex[b][i] for b in use], float)
        n_draw = n_mc if do_mc[i] else 0
        draws = np.vstack([f0[None, :],
                           f0[None, :] + rng.normal(0.0, 1.0, (n_draw, len(use))) * e0])
        errs = np.broadcast_to(e0, draws.shape)
        tk, omk, c2k = fit_excess_blackbody_draws(draws, errs, use, refine=True)
        if not np.isfinite(tk[0]):
            continue
        omega_star = _star_solid_angle(a_mag[i], anchor, teff[i])
        have_star = np.isfinite(omega_star) and omega_star > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            tauk = (omk * tk ** 4) / (omega_star * teff[i] ** 4) if have_star \
                else np.full(len(tk), np.nan)
        t_dust[i], tau[i], chi2[i] = tk[0], tauk[0], c2k[0]
        if n_draw:
            ts = tk[1:][np.isfinite(tk[1:])]
            taus = tauk[1:][np.isfinite(tauk[1:])]
            if ts.size:
                t_lo[i], t_hi[i] = np.percentile(ts, [16, 84])
            if taus.size:
                tau_lo[i], tau_hi[i] = np.percentile(taus, [16, 84])

    out["t_dust_k"] = t_dust
    out["t_dust_lo_k"] = t_lo
    out["t_dust_hi_k"] = t_hi
    out["tau"] = tau
    out["tau_lo"] = tau_lo
    out["tau_hi"] = tau_hi
    out["dust_fit_chi2"] = chi2
    out["n_excess_bands"] = nb
    return out


def wien_peak_k(band: str) -> float:
    """Dust temperature whose Wien peak falls in ``band``.

    W1 -> 852 K, W2 -> 630 K, W3 -> 241 K, W4 -> 132 K.  This sets the channel's
    honest reach: W3 is the workhorse for warm dust, and everything below ~200 K
    is accessible only through W4 -- the shallowest, most confusion-limited band,
    whose depth has been frozen since the 2010 cryogenic mission ended (NEOWISE-R,
    CatWISE2020 and the deep unWISE coadds are W1/W2 only).  There is no route to
    a deeper 12/22 um measurement, so the sensitivity floor is instrumental, not
    a choice.
    """
    b = 2.897771955e3  # Wien displacement constant, um K
    return float(b / BANDS[band]["lambda_um"])


def cascade_timescale_yr(period_yr: float, covering_fraction: float) -> float:
    """Lacki (2025) collisional-cascade timescale: t ~ P / f.

    A swarm with geometric covering fraction ``f`` self-collides on roughly the
    orbital period divided by ``f`` once station-keeping stops.  For f = 1e-3 at
    1 AU that is ~1e3 yr -- instantaneous next to any stellar age, which is why
    the *dust*, not the intact swarm, is the observable relic.  Provided so the
    channel's headline claim can be evaluated numerically rather than asserted.
    """
    if not (np.isfinite(period_yr) and np.isfinite(covering_fraction)) \
            or covering_fraction <= 0:
        return np.nan
    return float(period_yr / covering_fraction)


def select_excess(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Ledger-compliant excess selection.

    Requires *both* a significant flux excess in a star-dominated band (W1/W2/W3,
    never W4 alone) *and* a significant red colour excess in W1-W2 or W1-W3.  The
    colour requirement is what makes the detection anchor-independent, and it is
    the exact test that killed the previous channel's W4-only candidates.
    """
    chi_min = float(cfg["chi_min"])
    band_ok = pd.Series(False, index=df.index)
    for b in cfg["require_bands"]:
        col = f"chi_{b}"
        if col in df.columns:
            band_ok = band_ok | ((df[col] >= chi_min)
                                 & (df.get(f"{b}_excess_jy", 0) > 0))

    col_min = float(cfg["color_excess_sigma_min"])
    colour_ok = pd.Series(False, index=df.index)
    for name in ("w1_w2", "w1_w3"):
        c = f"chi_{name}"
        if c in df.columns:
            colour_ok = colour_ok | (df[c] >= col_min)

    return (band_ok & colour_ok).fillna(False)


__all__ = ["ColourLocus", "fit_colour_locus", "fit_loci", "compute_excess",
           "fit_excess_blackbody", "fit_excess_blackbody_draws", "characterise",
           "select_excess",
           "cascade_timescale_yr", "wien_peak_k", "BANDS"]

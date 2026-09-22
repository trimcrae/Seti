"""The S52 statistic: T_bb, f = L_IR/L_* and log(f / f_max) per star.

Photosphere
-----------
An **empirical** locus, not a synthetic one: the robust running median of
``K_s - W_i`` against ``BP - RP`` over the parent sample's own bright, clean
stars (OSSUARY's estimator, reused: :func:`seti.ossuary.excess.fit_colour_locus`).
The anchor is K_s (2.16 um), the longest band still photospheric for every FGK
dwarf, and ``K_s - W1 < 0.2`` is required so that the anchor itself is not
already contaminated.  The locus for W4 is fitted on stars bright enough that a
bare photosphere is a >= 5-sigma W4 detection (``locus_ks_max``): at fainter
K_s the W4-detected subsample is *selected for excess* and its median would be
biased --- so the locus is fitted where the median is a photosphere and applied
everywhere.

Dust
----
The excess fluxes in W3 and W4 (both required significant) with W2 as a third,
constraining band are fitted by a single-temperature blackbody
``F_nu,exc = Omega_d B_nu(T_bb)`` on a temperature grid, vectorised over the grid
so the whole parent can be fitted and Monte-Carlo'd in seconds.  Then

* ``F_IR = Omega_d sigma T_bb^4 / pi`` (W m^-2) and ``f = F_IR / F_bol``, with
  ``F_bol = L_* / 4 pi d^2`` and ``L_*`` from FLAME where it exists, else from
  ``M_G`` and the Andrae et al. (2018) Gaia bolometric correction;
* the blackbody radius ``r_bb = (278.3 K / T_bb)^2 sqrt(L_*/L_sun)`` AU;
* Wyatt et al. (2007) eq. 20 with dr/r = 0.5, D_c = 2000 km, Q_D* = 200 J/kg,
  e = 0.05, in Moór's catalogue form with M_* and L_*::

      f_max = 0.16e-3 r^{7/3} M_*^{-5/6} L_*^{-1/2} t^{-1}   (r in AU, t in Myr)

  --- ``log_f_fmax_1gyr`` is evaluated at t = 1000 Myr, the youngest age the
  cell admits, so it is a LOWER bound on log(f/f_max) for every star that
  really is older than 1 Gyr; ``log_f_fmax_adopted`` uses the star's adopted
  age where one exists.

Colour corrections for a 250-350 K blackbody in W3/W4 (Wright et al. 2010,
of order 5-10 %) are NOT applied; they move T_bb by ~10 K and f by < 0.05 dex,
against a threshold of 3 dex.  Stated in docs/cradle.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ossuary.excess import ColourLocus, compute_excess, fit_colour_locus
from ..photometry import BANDS, band_freq_hz, mag_to_flux_jy, planck_bnu

SIGMA_SB = 5.670374419e-8      # W m^-2 K^-4
L_SUN_W = 3.828e26
PC_M = 3.085677581491367e16
AU_M = 1.495978707e11
M_BOL_SUN = 4.74

#: Andrae et al. 2018 (A&A 616, A8) Table 4: BC_G(Teff) for 4000 <= Teff <= 8000 K.
_BC_G_COEF = (6.000e-02, 6.731e-05, -6.647e-08, 2.859e-11, -7.197e-15)
_BC_G_COEF_COOL = (1.749e+00, 1.977e-03, 3.737e-07, -8.966e-11, -4.183e-14)  # 3300-4000 K

DEFAULT_EXCESS: dict = {
    "locus_color_bin": 0.10,
    "locus_min_per_bin": 20,
    "locus_scatter_floor_mag": 0.03,
    "locus_ks_max": 7.5,            # W4 photosphere >= 5 sigma below this K_s
    "locus_ks_w1_max": 0.2,
    "sys_floor_mag": 0.03,
    "chi_min": 3.0,                 # W3 AND W4 excess significance
    "ks_w1_max": 0.2,               # the mission's photosphere cut
    "t_grid_k": [60.0, 2000.0, 400],
    "mc_draws": 300,
    "t_cell_k": [250.0, 350.0],
    "log_f_fmax_min": 3.0,
    "fmax_prefactor": 0.16e-3,
    "fmax_age_floor_myr": 1000.0,
    "shortlist_t_k": [180.0, 450.0],   # enriched and reported; the strict cell is applied at assess
}

_BANDS = ("W1", "W2", "W3", "W4")


def _num(d: pd.DataFrame, col: str) -> np.ndarray:
    """A numeric column as float array; all-NaN when the column is absent."""
    if col not in d.columns:
        return np.full(len(d), np.nan, dtype=float)
    return pd.to_numeric(d[col], errors="coerce").to_numpy(float)


# ---------------------------------------------------------------------------
# Harmonisation onto the OSSUARY column contract
# ---------------------------------------------------------------------------
def harmonise(df: pd.DataFrame) -> pd.DataFrame:
    """``w3mpro`` -> ``W3mag`` etc., ``ks_m`` -> ``Ksmag``, plus ``teff``.

    The archive spellings are **kept alongside** the OSSUARY ones: the vetting
    rules, the shortlist contract and the candidate table all speak
    ``ks_m`` / ``w3mpro``, and a rename would silently turn every row into
    ``KS_MISSING`` downstream.
    """
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    ren = {}
    for b in _BANDS:
        lb = b.lower()
        ren[f"{lb}mpro"] = f"{b}mag"
        ren[f"{lb}mpro_error"] = f"e_{b}mag"
    ren.update({"ks_m": "Ksmag", "ks_msigcom": "e_Ksmag", "j_m": "Jmag", "j_msigcom": "e_Jmag",
                "h_m": "Hmag", "h_msigcom": "e_Hmag"})
    for src, dst in ren.items():
        if src in out.columns:
            out[dst] = out[src]
    for b in _BANDS + ("Ks", "J", "H"):
        if f"{b}mag" not in out:
            out[f"{b}mag"] = np.nan
        if f"e_{b}mag" not in out:
            out[f"e_{b}mag"] = np.nan
    teff = pd.Series(np.nan, index=out.index, dtype=float)
    for col in ("teff_gspspec", "teff_gspphot"):
        if col in out:
            v = pd.to_numeric(out[col], errors="coerce")
            teff = teff.fillna(v)
    out["teff"] = teff
    return out


# ---------------------------------------------------------------------------
# Locus
# ---------------------------------------------------------------------------
def locus_reference_mask(d: pd.DataFrame, cfg: dict) -> pd.Series:
    """Stars the photosphere locus is fitted on: bright, clean, anchored."""
    c = {**DEFAULT_EXCESS, **(cfg or {})}
    ks = pd.to_numeric(d.get("Ksmag"), errors="coerce")
    w1 = pd.to_numeric(d.get("W1mag"), errors="coerce")
    m = ks.notna() & (ks < float(c["locus_ks_max"])) & ((ks - w1) < float(c["locus_ks_w1_max"]))
    if "tmass_ph_qual" in d:
        m &= d["tmass_ph_qual"].astype(str).str.upper().str[2:3].eq("A")
    if "ext_flag" in d:
        m &= pd.to_numeric(d["ext_flag"], errors="coerce").fillna(0) == 0
    if "cc_flags" in d:
        m &= d["cc_flags"].astype(str).str.strip().str.replace(r"^0+$", "0000", regex=True) \
            .isin(["0000", "0"])
    if "is_control" in d:
        m &= ~d["is_control"].fillna(False).astype(bool)
    return m.fillna(False)


def fit_loci(d: pd.DataFrame, cfg: dict) -> dict[str, ColourLocus]:
    c = {**DEFAULT_EXCESS, **(cfg or {})}
    ref = d[locus_reference_mask(d, c)]
    return {b: fit_colour_locus(ref, b, anchor="Ks", colour_col="bp_rp",
                                bin_width=float(c["locus_color_bin"]),
                                min_per_bin=int(c["locus_min_per_bin"]),
                                scatter_floor=float(c["locus_scatter_floor_mag"]))
            for b in _BANDS}


def loci_summary(loci: dict) -> dict:
    return {b: {"n_bins": int(loc.n_bins),
                "centres": [round(float(x), 3) for x in loc.centres],
                "median": [round(float(x), 4) for x in loc.median],
                "scatter": [round(float(x), 4) for x in loc.scatter],
                "counts": [int(x) for x in loc.counts]} for b, loc in loci.items()}


def excess_table(d: pd.DataFrame, loci: dict, cfg: dict) -> pd.DataFrame:
    """OSSUARY's per-band excess (Jy, chi) over the CRADLE loci."""
    c = {**DEFAULT_EXCESS, **(cfg or {})}
    return compute_excess(d, loci, {"sys_floor_mag": float(c["sys_floor_mag"])},
                          anchor="Ks", colour_col="bp_rp")


# ---------------------------------------------------------------------------
# Stellar luminosity and mass
# ---------------------------------------------------------------------------
def bc_g(teff: np.ndarray) -> np.ndarray:
    """Gaia G-band bolometric correction, Andrae et al. 2018 (mag)."""
    t = np.asarray(teff, float)
    dt = t - 5772.0
    warm = sum(a * dt ** i for i, a in enumerate(_BC_G_COEF))
    cool = sum(a * dt ** i for i, a in enumerate(_BC_G_COEF_COOL))
    out = np.where(t >= 4000.0, warm, cool)
    out = np.where((t < 3300.0) | (t > 8000.0), np.nan, out)
    return out


def stellar_luminosity(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(L/L_sun, source)``: FLAME where finite, else M_G + BC_G(Teff)."""
    lum = _num(d, "lum_flame")
    abs_g = _num(d, "abs_g")
    ag = np.nan_to_num(_num(d, "ag_gspphot"), nan=0.0)
    teff = _num(d, "teff")
    mbol = abs_g - ag + bc_g(teff)
    l_phot = 10.0 ** (-0.4 * (mbol - M_BOL_SUN))
    src = np.where(np.isfinite(lum) & (lum > 0), "flame",
                   np.where(np.isfinite(l_phot), "mg_bc", "none")).astype(object)
    out = np.where(np.isfinite(lum) & (lum > 0), lum, l_phot)
    return out, src


def stellar_mass(d: pd.DataFrame, lum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(M/M_sun, source)``: FLAME where finite, else the main-sequence L ~ M^4."""
    m = _num(d, "mass_flame")
    ms = np.where(np.isfinite(lum) & (lum > 0), np.clip(np.asarray(lum, float), 1e-3, 1e3) ** 0.25,
                  np.nan)
    src = np.where(np.isfinite(m) & (m > 0), "flame", np.where(np.isfinite(ms), "ml_relation",
                                                                  "none")).astype(object)
    return np.where(np.isfinite(m) & (m > 0), m, ms), src


# ---------------------------------------------------------------------------
# The blackbody fit, vectorised over a temperature grid
# ---------------------------------------------------------------------------
def _t_grid(cfg: dict) -> np.ndarray:
    lo, hi, n = cfg.get("t_grid_k", DEFAULT_EXCESS["t_grid_k"])
    return np.geomspace(float(lo), float(hi), int(n))


def model_matrix(t_grid: np.ndarray, bands=_BANDS) -> np.ndarray:
    """``B_nu(T, band)`` in Jy per steradian: shape ``(n_T, n_bands)``."""
    return np.column_stack([planck_bnu(t_grid, band_freq_hz(b)) * 1e26 for b in bands])


def fit_bb_grid(flux_jy: np.ndarray, err_jy: np.ndarray, t_grid: np.ndarray,
                mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted single-temperature fit of ``flux_jy`` (rows) over the grid.

    ``flux_jy``/``err_jy`` are ``(n_rows, n_bands)`` with NaN where a band is
    not used.  Returns ``(T, Omega_sr, chi2)`` per row --- the grid minimum,
    then a parabolic refinement in log T so the grid spacing does not dominate.
    """
    f = np.asarray(flux_jy, float)
    e = np.asarray(err_jy, float)
    use = np.isfinite(f) & np.isfinite(e) & (e > 0)
    w = np.where(use, 1.0 / np.where(use, e, 1.0) ** 2, 0.0)
    fz = np.where(use, f, 0.0)
    # omega(T) = sum(M f w) / sum(M^2 w), per row and per T: (n_rows, n_T)
    num = fz * w @ mm.T
    den = w @ (mm.T ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(den > 0, num / den, np.nan)
    omega = np.where(omega > 0, omega, np.nan)
    # chi2(T) = sum w (f - omega M)^2
    resid = fz[:, None, :] - omega[:, :, None] * mm[None, :, :]
    chi2 = np.nansum(w[:, None, :] * resid ** 2, axis=2)
    chi2 = np.where(np.isfinite(omega), chi2, np.inf)
    j = np.argmin(chi2, axis=1)
    rows = np.arange(f.shape[0])
    t_best = t_grid[j].astype(float)
    om_best = omega[rows, j]
    c_best = chi2[rows, j]
    # Parabolic refinement in log T around the minimum (interior points only).
    ok = (j > 0) & (j < len(t_grid) - 1) & np.isfinite(c_best)
    if ok.any():
        lt = np.log(t_grid)
        jm, j0, jp = j[ok] - 1, j[ok], j[ok] + 1
        r = rows[ok]
        y0, ym, yp = chi2[r, j0], chi2[r, jm], chi2[r, jp]
        denom = (ym - 2 * y0 + yp)
        good = np.isfinite(ym) & np.isfinite(yp) & (denom > 0)
        shift = np.where(good, 0.5 * (ym - yp) / np.where(good, denom, 1.0), 0.0)
        shift = np.clip(shift, -1.0, 1.0)
        t_ref = np.exp(lt[j0] + shift * (lt[jp] - lt[j0]))
        t_best[ok] = np.where(good, t_ref, t_best[ok])
    bad = ~np.isfinite(c_best) | (use.sum(axis=1) < 2)
    t_best[bad] = np.nan
    om_best[bad] = np.nan
    c_best[bad] = np.nan
    return t_best, om_best, c_best


def blackbody_radius_au(t_bb_k, lum_lsun) -> np.ndarray:
    """``r_bb = (278.3 / T)^2 sqrt(L)`` AU (a blackbody grain in equilibrium)."""
    t = np.asarray(t_bb_k, float)
    lum = np.asarray(lum_lsun, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (278.3 / t) ** 2 * np.sqrt(lum)


def wyatt_fmax(r_au, t_myr, mass_msun=1.0, lum_lsun=1.0, prefactor: float = 0.16e-3) -> np.ndarray:
    """Wyatt et al. 2007 eq. 20, Moór's form: ``0.16e-3 r^{7/3} M^{-5/6} L^{-1/2} / t``."""
    r = np.asarray(r_au, float)
    t = np.asarray(t_myr, float)
    m = np.asarray(mass_msun, float)
    lum = np.asarray(lum_lsun, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return prefactor * r ** (7.0 / 3.0) * m ** (-5.0 / 6.0) * lum ** (-0.5) / t


def disk_fraction(omega_sr, t_bb_k, fbol_w_m2) -> np.ndarray:
    """``f = L_IR / L_* = (Omega sigma T^4 / pi) / F_bol``."""
    om = np.asarray(omega_sr, float)
    t = np.asarray(t_bb_k, float)
    fb = np.asarray(fbol_w_m2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (om * SIGMA_SB * t ** 4 / np.pi) / fb


def log_f_over_fmax(f, r_au, t_myr, mass_msun=1.0, lum_lsun=1.0,
                    prefactor: float = 0.16e-3) -> np.ndarray:
    fm = wyatt_fmax(r_au, t_myr, mass_msun, lum_lsun, prefactor)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.asarray(f, float) / fm)


def fbol_w_m2(lum_lsun, parallax_mas) -> np.ndarray:
    d_m = 1000.0 / np.asarray(parallax_mas, float) * PC_M
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(lum_lsun, float) * L_SUN_W / (4.0 * np.pi * d_m ** 2)


# ---------------------------------------------------------------------------
# Per-star: the whole statistic
# ---------------------------------------------------------------------------
def fit_disk(d: pd.DataFrame, cfg: dict, *, rng: np.random.Generator | None = None,
             age_gyr: np.ndarray | None = None, mc_all: bool = False) -> pd.DataFrame:
    """T_bb, f, r_bb, log(f/f_max) with Monte-Carlo 16/84 bounds, per row.

    Requires the ``{b}_excess_jy`` / ``{b}_excess_err_jy`` columns of
    :func:`excess_table`.  Bands used: W3 and W4 always (a row without both
    significant is still fitted, but flagged ``excess_significant=False``),
    W2 as a constraint (its excess, of whatever sign, with its error).  W1 is
    the anchor-side check and never enters the fit.
    """
    c = {**DEFAULT_EXCESS, **(cfg or {})}
    rng = rng or np.random.default_rng(20260921)
    n = len(d)
    out = d.copy()
    t_grid = _t_grid(c)
    fit_bands = ("W2", "W3", "W4")
    mm = model_matrix(t_grid, fit_bands)
    f = np.column_stack([_num(d, f"{b}_excess_jy")
                         for b in fit_bands])
    e = np.column_stack([_num(d, f"{b}_excess_err_jy")
                         for b in fit_bands])
    chi3 = _num(d, "chi_W3")
    chi4 = _num(d, "chi_W4")
    sig = (chi3 >= float(c["chi_min"])) & (chi4 >= float(c["chi_min"]))
    # W3 and W4 must both be present for a fit at all.
    fit_ok = np.isfinite(f[:, 1]) & np.isfinite(f[:, 2]) & np.isfinite(e[:, 1]) & np.isfinite(e[:, 2])

    t_bb = np.full(n, np.nan)
    omega = np.full(n, np.nan)
    chi2 = np.full(n, np.nan)
    if fit_ok.any():
        tb, om, c2 = fit_bb_grid(f[fit_ok], e[fit_ok], t_grid, mm)
        t_bb[fit_ok], omega[fit_ok], chi2[fit_ok] = tb, om, c2

    lum, lsrc = stellar_luminosity(out)
    mass, msrc = stellar_mass(out, lum)
    plx = _num(out, "parallax")
    fb = fbol_w_m2(lum, plx)
    f_ir = disk_fraction(omega, t_bb, fb)
    r_bb = blackbody_radius_au(t_bb, lum)
    t_floor = float(c["fmax_age_floor_myr"])
    lff_1gyr = log_f_over_fmax(f_ir, r_bb, t_floor, mass, lum, float(c["fmax_prefactor"]))
    if age_gyr is None:
        age_gyr = _num(out, "age_adopted_gyr") \
            if "age_adopted_gyr" in out else np.full(n, np.nan)
    age_myr = np.asarray(age_gyr, float) * 1000.0
    lff_age = log_f_over_fmax(f_ir, r_bb, np.where(np.isfinite(age_myr), age_myr, np.nan),
                              mass, lum, float(c["fmax_prefactor"]))

    # Monte Carlo: redraw the excess fluxes from their errors and refit, only
    # where a fit exists (cheap: one vectorised call per draw over all rows).
    n_mc = int(c.get("mc_draws", 300))
    t_lo = np.full(n, np.nan)
    t_hi = np.full(n, np.nan)
    l_lo = np.full(n, np.nan)
    l_hi = np.full(n, np.nan)
    f_lo = np.full(n, np.nan)
    f_hi = np.full(n, np.nan)
    idx = np.nonzero(fit_ok & np.isfinite(t_bb) & (sig | bool(mc_all)))[0]
    if len(idx) and n_mc > 0:
        ts = np.full((n_mc, len(idx)), np.nan)
        fs = np.full((n_mc, len(idx)), np.nan)
        ls = np.full((n_mc, len(idx)), np.nan)
        fi, ei = f[idx], e[idx]
        for k in range(n_mc):
            draw = fi + rng.normal(0.0, 1.0, fi.shape) * np.nan_to_num(ei, nan=0.0)
            tk, omk, _ = fit_bb_grid(draw, ei, t_grid, mm)
            ts[k] = tk
            fk = disk_fraction(omk, tk, fb[idx])
            fs[k] = fk
            ls[k] = log_f_over_fmax(fk, blackbody_radius_au(tk, lum[idx]), t_floor,
                                    mass[idx], lum[idx], float(c["fmax_prefactor"]))
        with np.errstate(invalid="ignore"):
            t_lo[idx], t_hi[idx] = np.nanpercentile(ts, [16, 84], axis=0)
            f_lo[idx], f_hi[idx] = np.nanpercentile(fs, [16, 84], axis=0)
            l_lo[idx], l_hi[idx] = np.nanpercentile(ls, [16, 84], axis=0)

    out["excess_significant"] = sig & fit_ok
    out["t_bb_k"] = t_bb
    out["t_bb_lo_k"] = t_lo
    out["t_bb_hi_k"] = t_hi
    out["omega_dust_sr"] = omega
    out["bb_fit_chi2"] = chi2
    out["lum_lsun"] = lum
    out["lum_source"] = lsrc
    out["mass_msun"] = mass
    out["mass_source"] = msrc
    out["fbol_w_m2"] = fb
    out["f_ir"] = f_ir
    out["f_ir_lo"] = f_lo
    out["f_ir_hi"] = f_hi
    out["r_bb_au"] = r_bb
    out["fmax_1gyr"] = wyatt_fmax(r_bb, t_floor, mass, lum, float(c["fmax_prefactor"]))
    out["log_f_fmax_1gyr"] = lff_1gyr
    out["log_f_fmax_1gyr_lo"] = l_lo
    out["log_f_fmax_1gyr_hi"] = l_hi
    out["log_f_fmax_adopted"] = lff_age
    tc = c["t_cell_k"]
    out["in_t_cell"] = (t_bb >= float(tc[0])) & (t_bb <= float(tc[1]))
    out["above_fmax_3dex"] = np.isfinite(lff_1gyr) & (lff_1gyr > float(c["log_f_fmax_min"]))
    out["above_fmax_3dex_16pct"] = np.isfinite(l_lo) & (l_lo > float(c["log_f_fmax_min"]))
    return out


def photosphere_flux_jy(mag, band: str) -> np.ndarray:
    return mag_to_flux_jy(np.asarray(mag, float), band)


__all__ = ["BANDS", "DEFAULT_EXCESS", "bc_g", "blackbody_radius_au", "disk_fraction",
           "excess_table", "fbol_w_m2", "fit_bb_grid", "fit_disk", "fit_loci", "harmonise",
           "loci_summary", "locus_reference_mask", "log_f_over_fmax", "model_matrix",
           "photosphere_flux_jy", "stellar_luminosity", "stellar_mass", "wyatt_fmax"]

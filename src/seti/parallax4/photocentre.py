"""The astrometric half: does the photocentre move with the brightness?

Physics.  Two point sources, target T and neighbour N at sky offset
rho * u (u a unit vector, rho in mas), fluxes F_T and F_N, beta = F_N / F.
The unresolved photocentre sits at beta * rho * u from T.  If N varies,
d(photocentre) = rho (1 - beta) (dF/F) u; if T varies,
d(photocentre) = -rho beta (dF/F) u.  Either way a blend makes the along-scan
residual linear in the fractional flux residual phi with a fixed SKY vector

    r_i = psi_i * (D_x sin theta_i + D_y cos theta_i),  psi = phi / (1 + phi)   (1)

(phi = F/F_median - 1; psi makes (1) exact for any depth, with D = rho (1-beta)
if N varies and -rho beta if T varies; theta the scan position angle; (sin, cos) the AL unit vector in
(alpha*, delta), verified on the DR4 prerelease).  A variation ON the source
that dominates the photocentre gives D = 0.  This is varstrometry
(Hwang et al. 2020, arXiv:1908.02292) done per transit instead of through the
excess-noise summary statistic.

The fit.  Per source, weighted least squares of the transit-level AL
centroid on the five astrometric columns [sin, cos, parallax factor,
t sin, t cos] plus the blend pair [phi sin, phi cos] plus one detector-frame
column [phi]: a flux-dependent centroid bias that does NOT flip with scan
direction (charge-transfer inefficiency, PSF calibration error vs magnitude).
A sky displacement flips sign between theta and theta + 180 deg; a detector
effect does not.  An extra jitter is solved for so that chi^2/nu = 1 (orbital
motion, attitude noise), so significances are not inflated by unmodelled
motion.

Systematics that correlate photometry and astrometry by scan direction:

* SCAN_ANGLE_FLUX -- phi is predictable from theta (harmonics k = 1..3) or
  from a known neighbour's window membership.  A static neighbour at ~1"
  entering the 0.7" x 2.1" window on some scans adds flux AND pulls the
  centroid: it mimics (1) exactly, but its "variability" is a function of
  scan geometry, not time.
* DETECTOR_FRAME -- the phi column is significant.
* SECTOR_INCONSISTENT -- per-scan-angle-sector slopes disagree with the
  single sky vector D of (1).

Verdicts, in precedence order: INSUFFICIENT, NO_FLUX_VARIATION, SCAN_ANGLE_FLUX, DETECTOR_FRAME,
SECTOR_INCONSISTENT, BLEND_NEIGHBOUR, BLEND_UNRESOLVED, ON_TARGET, AMBIGUOUS.
Finite D confidence requires both sky coefficients to be estimable modulo all nuisance
columns in the same direct weighted SVD used for the fit/covariance. Missing sky
directions return INSUFFICIENT, without a finite D upper bound. Nuisance-only
rank deficiency can leave D estimable.
ON_TARGET means: no photocentre motion, and the varying light lies within
``ul95`` mas (x (1-beta)^-1) of the photocentre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Gaia AF window half-sizes for G <~ 16 (12 x 12 samples; AL pixel 59 mas,
#: AC 177 mas): AL +-354 mas, AC +-1062 mas.  Fainter sources get 12 x 6
#: windows with the same AL extent.
WINDOW_HALF_AL_MAS = 354.0
WINDOW_HALF_AC_MAS = 1062.0


@dataclass(frozen=True)
class PhotocentreConfig:
    n_min: int = 12
    p_blend: float = 1e-4          # Delta chi^2 (2 dof) p-value for "D != 0"
    p_null: float = 0.01           # D consistent with 0 above this
    ul_on_target_mas: float = 20.0  # 1000x finer than a 21" TESS pixel
    sigma_d_max_mas: float = 200.0  # leverage floor: sigma(D) above this is INSUFFICIENT
    p_scan: float = 1e-3
    r2_scan: float = 0.3
    z_detector: float = 4.0
    p_sector: float = 1e-3
    n_sectors: int = 6
    min_per_sector: int = 4
    neighbour_perp_sigma: float = 3.0
    harmonics: int = 3
    p_phot_variable: float = 1e-3  # flux must vary beyond its errors at this p

    @classmethod
    def from_dict(cls, d: dict | None) -> PhotocentreConfig:
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _chi2_sf(x: float, k: int) -> float:
    from scipy.stats import chi2

    return float(chi2.sf(x, k))


def _f_sf(f: float, d1: int, d2: int) -> float:
    from scipy.stats import f as fdist

    return float(fdist.sf(f, d1, d2))


def _weighted_svd(X, y, s):
    """One direct, column-scaled weighted SVD for fit, covariance and estimability.

    Squaring the design in normal equations can discard near-null directions
    before lstsq does. This uses the same retained singular directions for all
    three quantities and maps coefficients/covariance back to physical units.
    A finite pseudoinverse variance is NOT a bound on a nonestimable coefficient.
    """
    X, y, s = np.asarray(X, float), np.asarray(y, float), np.asarray(s, float)
    if X.ndim != 2 or y.shape != (len(X),) or s.shape != (len(X),) \
            or not np.isfinite(X).all() or not np.isfinite(y).all() \
            or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("weighted design requires finite data and positive uncertainties")
    A, b = X / s[:, None], y / s
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 0, scale, 1.0)  # zero nuisance columns remain explicit
    B = A / scale
    u, singular, vt = np.linalg.svd(B, full_matrices=False)
    cutoff = np.finfo(float).eps * max(B.shape) * (singular[0] if len(singular) else 0.0)
    keep = singular > cutoff
    v = vt[keep].T
    gamma = v @ ((u[:, keep].T @ b) / singular[keep])
    beta = gamma / scale
    inverse_basis = v / singular[keep]
    cov_scaled = inverse_basis @ inverse_basis.T
    cov = cov_scaled / (scale[:, None] * scale[None, :])
    rank = int(keep.sum())
    # I - V_retained V_retained^T includes missing directions even when N < p.
    null_projector = np.eye(X.shape[1]) - v @ v.T
    null_norm = np.linalg.norm(null_projector, axis=1)
    # Dimensionless orthogonal-projector roundoff allowance, NOT a physical
    # leverage threshold. The physical uncertainty floor remains cfg unchanged.
    projector_tol = np.finfo(float).eps * max(B.shape) * X.shape[1]
    estimable = (null_norm <= projector_tol) if rank < X.shape[1] else np.ones(
        X.shape[1], dtype=bool)
    residual = y - X @ beta
    chi2 = float(np.sum((residual / s) ** 2))
    diagnostic = {
        "method": "DIRECT_COLUMN_SCALED_WEIGHTED_SVD",
        "rank": rank, "n_columns": int(X.shape[1]), "rank_tolerance": float(cutoff),
        "scaled_singular_values": singular.tolist(), "column_scales": scale.tolist(),
        "coefficient_estimable": estimable.tolist(),
        "coefficient_null_projection_norm": null_norm.tolist(),
        "projector_roundoff_tolerance": float(projector_tol),
    }
    return beta, cov, residual, chi2, diagnostic


def _wls(X, y, s):
    beta, cov, residual, chi2, _ = _weighted_svd(X, y, s)
    return beta, cov, residual, chi2

def _jitter(X, y, s0):
    """Extra per-transit noise making chi^2/nu = 1 (0 if already <= 1)."""
    _, _, _, c0, design = _weighted_svd(X, y, s0)
    dof = len(y) - design["rank"]
    if dof <= 0:
        return 0.0
    if c0 / dof <= 1.0:
        return 0.0
    lo, hi = 0.0, float(np.std(y) * 10 + np.max(s0) * 10 + 1.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        _, _, _, c = _wls(X, y, np.sqrt(s0 ** 2 + mid ** 2))
        if c / dof > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def neighbour_offsets(ra, dec, nb: pd.DataFrame) -> pd.DataFrame:
    """Neighbour table (ra, dec, phot_g_mean_flux or phot_g_mean_mag) ->
    offsets in mas from the target, position angle and G flux."""
    if nb is None or not len(nb):
        return pd.DataFrame(columns=["dra_mas", "ddec_mas", "rho_mas", "flux_g", "source_id"])
    d = nb.copy()
    cosd = np.cos(np.deg2rad(dec))
    dra = (d["ra"].to_numpy(float) - ra)
    dra = (dra + 180.0) % 360.0 - 180.0
    d["dra_mas"] = dra * cosd * 3.6e6
    d["ddec_mas"] = (d["dec"].to_numpy(float) - dec) * 3.6e6
    d["rho_mas"] = np.hypot(d["dra_mas"], d["ddec_mas"])
    if "flux_g" not in d:
        if "phot_g_mean_flux" in d:
            d["flux_g"] = d["phot_g_mean_flux"]
        elif "phot_g_mean_mag" in d:
            d["flux_g"] = 10 ** (-0.4 * (d["phot_g_mean_mag"].to_numpy(float) - 25.6874))
    return d


def window_membership(theta: np.ndarray, dra_mas: float, ddec_mas: float) -> np.ndarray:
    """Is the neighbour inside the target's AF window on each transit?"""
    al = dra_mas * np.sin(theta) + ddec_mas * np.cos(theta)
    ac = -dra_mas * np.cos(theta) + ddec_mas * np.sin(theta)
    return (np.abs(al) <= WINDOW_HALF_AL_MAS) & (np.abs(ac) <= WINDOW_HALF_AC_MAS)


def prepare_joint(joined: pd.DataFrame) -> pd.DataFrame:
    """Add phi (fractional G flux residual about the median) and its error
    sphi to a photometry x astrometry join (``epochs.join_photometry_astrometry``);
    unusable photometric transits are dropped."""
    j = joined.copy()
    ok = np.isfinite(j["f_g"]) & (j["f_g"] > 0) & ~j["bad_g"].astype(bool)
    j = j[ok]
    med = float(np.median(j["f_g"])) if len(j) else np.nan
    j["phi"] = j["f_g"] / med - 1.0
    j["sphi"] = j["e_g"] / med
    return j


def fit_photocentre(tr: pd.DataFrame, *, target_flux_g: float | None = None,
                    neighbours: pd.DataFrame | None = None,
                    cfg: PhotocentreConfig | None = None) -> dict:
    """One source's joint photometry-astrometry test.

    ``tr`` needs t_yr, theta (rad), pf_al, x_al (mas), sx_al (mas), phi
    (fractional G flux residual).  ``neighbours`` (optional) carries dra_mas,
    ddec_mas, flux_g (see ``neighbour_offsets``)."""
    cfg = cfg or PhotocentreConfig()
    d = tr.copy()
    m = np.ones(len(d), bool)
    for c in ("t_yr", "theta", "pf_al", "x_al", "sx_al", "phi"):
        m &= np.isfinite(d[c].to_numpy(float))
    d = d[m]
    n = len(d)
    out: dict = {"n_transits": int(n), "verdict": "INSUFFICIENT", "reasons": []}
    if n < cfg.n_min:
        out["reasons"].append(f"n_transits<{cfg.n_min}")
        return out
    th = d["theta"].to_numpy(float)
    s, c = np.sin(th), np.cos(th)
    t = d["t_yr"].to_numpy(float)
    pf = d["pf_al"].to_numpy(float)
    y = d["x_al"].to_numpy(float)
    s0 = d["sx_al"].to_numpy(float)
    phi_raw = d["phi"].to_numpy(float)
    if np.any(1.0 + phi_raw <= 0) or np.any(s0 <= 0):
        out["reasons"].append("nonpositive total flux or centroid uncertainty")
        return out
    phi_raw = (1.0 + phi_raw) / (1.0 + np.median(phi_raw)) - 1.0     # re-referenced to the median
    # The exact blend regressor.  With F the total flux, phi = F/F_med - 1:
    # neighbour varies -> offset = rho beta_med + rho (1 - beta_med) psi,
    # target varies    -> offset = rho beta_med - rho beta_med psi,
    # with psi = phi / (1 + phi) in both cases.  Linear in psi exactly, so
    # a deep eclipse does not leave a curvature residual for the sector test.
    phi = phi_raw / (1.0 + phi_raw)
    base = np.column_stack([s, c, pf, t * s, t * c])
    X = np.column_stack([base, phi * s, phi * c, phi])
    _, _, _, _, design = _weighted_svd(X, y, s0)
    out["astrometric_design"] = design
    if not all(design["coefficient_estimable"][5:7]):
        out["D_estimability"] = "NON_ESTIMABLE"
        out["reasons"].append("sky-blend coefficients are not identifiable modulo nuisance parameters")
        return out
    jit = _jitter(X, y, s0)
    sig = np.sqrt(s0 ** 2 + jit ** 2)
    beta, cov, r, chi2, design = _weighted_svd(X, y, sig)
    out["astrometric_design"] = design
    if not all(design["coefficient_estimable"][5:7]):
        out["D_estimability"] = "NON_ESTIMABLE"
        out["reasons"].append("sky-blend coefficients lost estimability in the jitter-weighted design")
        return out
    out["D_estimability"] = "ESTIMABLE"
    D = beta[5:7]
    cD = cov[5:7, 5:7]
    b_det, sb = float(beta[7]), float(np.sqrt(cov[7, 7]))
    try:
        dchi = float(D @ np.linalg.solve(cD, D))
    except np.linalg.LinAlgError:
        dchi = 0.0
    p_D = _chi2_sf(dchi, 2)
    ev = np.linalg.eigvalsh(cD)
    sig_max = float(np.sqrt(max(ev.max(), 0.0)))
    Dmag = float(np.hypot(*D))
    ul95 = Dmag + float(np.sqrt(5.991)) * sig_max
    out.update(jitter_mas=float(jit), chi2=chi2, dof=int(n - design["rank"]),
               D_ra_mas=float(D[0]), D_dec_mas=float(D[1]), D_mas=Dmag,
               D_pa_deg=float(np.degrees(np.arctan2(D[0], D[1])) % 360.0),
               sigma_D_ra=float(np.sqrt(cD[0, 0])), sigma_D_dec=float(np.sqrt(cD[1, 1])),
               sigma_D_max=sig_max, dchi2_D=dchi, p_D=p_D, ul95_mas=ul95,
               b_detector_mas=b_det, z_detector=(b_det / sb if sb > 0 and design["coefficient_estimable"][7] else np.nan),
               parallax_mas=float(beta[2]) if design["coefficient_estimable"][2] else None,
               pmra=float(beta[3]) if design["coefficient_estimable"][3] else None,
               pmdec=float(beta[4]) if design["coefficient_estimable"][4] else None,
               phi_rms=float(np.std(phi_raw)))
    # --- flux predictable from scan angle? -------------------------------
    H = [np.ones(n)]
    for k in range(1, cfg.harmonics + 1):
        H += [np.cos(k * th), np.sin(k * th)]
    H = np.column_stack(H)
    bh, *_ = np.linalg.lstsq(H, phi, rcond=None)
    rss1 = float(np.sum((phi - H @ bh) ** 2))
    rss0 = float(np.sum((phi - phi.mean()) ** 2))
    k1 = H.shape[1] - 1
    if rss0 > 0 and n - H.shape[1] > 0:
        F = ((rss0 - rss1) / k1) / (rss1 / (n - H.shape[1])) if rss1 > 0 else np.inf
        p_scan = _f_sf(F, k1, n - H.shape[1]) if np.isfinite(F) else 0.0
        r2 = 1.0 - rss1 / rss0
    else:
        p_scan, r2 = 1.0, 0.0
    out.update(p_scan_flux=p_scan, r2_scan_flux=r2)
    # --- neighbours --------------------------------------------------------
    nb_rows = []
    window_hit = None
    if neighbours is not None and len(neighbours):
        uvec_cov = cD
        for _, nr in neighbours.iterrows():
            rho = float(nr["rho_mas"])
            if not np.isfinite(rho) or rho <= 0:
                continue
            u = np.array([nr["dra_mas"], nr["ddec_mas"]], float) / rho
            up = np.array([-u[1], u[0]])
            fn = float(nr.get("flux_g", np.nan))
            bet = fn / (fn + target_flux_g) if (target_flux_g and np.isfinite(fn)) else np.nan
            dpar = float(D @ u)
            dperp = float(D @ up)
            spar = float(np.sqrt(u @ uvec_cov @ u))
            sperp = float(np.sqrt(up @ uvec_cov @ up))
            exp_nb = rho * (1 - bet) if np.isfinite(bet) else rho
            exp_tg = -rho * bet if np.isfinite(bet) else 0.0
            lo, hi = min(exp_tg, 0.0), max(exp_nb, 0.0)
            in_range = (dpar >= lo - 3 * spar) and (dpar <= hi + 3 * spar)
            perp_ok = abs(dperp) <= cfg.neighbour_perp_sigma * sperp
            inwin = window_membership(th, float(nr["dra_mas"]), float(nr["ddec_mas"]))
            p_win = np.nan
            if 0 < inwin.sum() < n:
                a, b_ = phi[inwin], phi[~inwin]
                from scipy.stats import mannwhitneyu

                try:
                    p_win = float(mannwhitneyu(a, b_, alternative="two-sided").pvalue)
                except ValueError:
                    p_win = np.nan
            row = {"neighbour_source_id": nr.get("source_id"), "rho_mas": rho,
                   "pa_deg": float(np.degrees(np.arctan2(u[0], u[1])) % 360.0),
                   "beta": bet, "expected_D_if_neighbour_varies_mas": exp_nb,
                   "expected_D_if_target_varies_mas": exp_tg, "D_parallel_mas": dpar,
                   "D_perp_mas": dperp, "sigma_parallel": spar, "sigma_perp": sperp,
                   "consistent": bool(in_range and perp_ok),
                   "frac_transits_in_window": float(inwin.mean()), "p_window_flux": p_win}
            nb_rows.append(row)
            if np.isfinite(p_win) and p_win < cfg.p_scan:
                window_hit = row
    out["neighbours"] = nb_rows
    # --- sector consistency -------------------------------------------------
    res0 = y - base @ beta[:5]
    edges = np.linspace(-np.pi, np.pi, cfg.n_sectors + 1)
    thw = (th + np.pi) % (2 * np.pi) - np.pi
    chi_sec, n_sec, sectors = 0.0, 0, []
    for k in range(cfg.n_sectors):
        mk = (thw >= edges[k]) & (thw < edges[k + 1])
        if mk.sum() < cfg.min_per_sector or np.std(phi[mk]) <= 0:
            continue
        w = 1.0 / sig[mk] ** 2
        pk = phi[mk]
        sk = float(np.sum(w * pk * res0[mk]) / np.sum(w * pk * pk))
        ssk = float(1.0 / np.sqrt(np.sum(w * pk * pk)))
        # the model's own prediction, weighted exactly as the slope is
        # (by w * psi^2), so a sector whose leverage sits at one edge is
        # compared at that edge, not at its mean angle
        lev = w * pk * pk
        tk = float(np.arctan2(np.sum(lev * np.sin(th[mk])), np.sum(lev * np.cos(th[mk]))))
        pred = float(np.sum(lev * (D[0] * np.sin(th[mk]) + D[1] * np.cos(th[mk]) + b_det))
                     / np.sum(lev))
        chi_sec += ((sk - pred) / ssk) ** 2
        n_sec += 1
        sectors.append({"theta_deg": float(np.degrees(tk)), "n": int(mk.sum()), "slope": sk,
                        "sigma": ssk, "predicted": pred})
    p_sector = _chi2_sf(chi_sec, n_sec - 3) if n_sec > 3 else np.nan
    out.update(sectors=sectors, chi2_sector=chi_sec, n_sectors_used=n_sec, p_sector=p_sector)
    # --- verdict ---------------------------------------------------------
    # --- is there any brightness variation to correlate with? ----------------
    if "sphi" in d:
        sp = d["sphi"].to_numpy(float)
        okp = np.isfinite(sp) & (sp > 0)
        if okp.sum() > 3:
            chi_phot = float(np.sum((phi_raw[okp] / sp[okp]) ** 2))
            p_var = _chi2_sf(chi_phot, int(okp.sum()) - 1)
            out.update(chi2_phot=chi_phot, p_phot_variable=p_var)
            if p_var > cfg.p_phot_variable:
                out["reasons"].append(f"flux not variable beyond its errors (p={p_var:.2g})")
                out["verdict"] = "NO_FLUX_VARIATION"
                return out
    if sig_max > cfg.sigma_d_max_mas:
        out["reasons"].append(f"sigma_D_max={sig_max:.1f}mas>{cfg.sigma_d_max_mas}")
        out["verdict"] = "INSUFFICIENT"
        return out
    if (p_scan < cfg.p_scan and r2 > cfg.r2_scan) or window_hit is not None:
        out["verdict"] = "SCAN_ANGLE_FLUX"
        if window_hit is not None:
            out["reasons"].append(f"flux tracks window membership of neighbour at "
                                  f"{window_hit['rho_mas']:.0f} mas (p={window_hit['p_window_flux']:.1e})")
        else:
            out["reasons"].append(f"flux predictable from scan angle (p={p_scan:.1e}, R2={r2:.2f})")
        return out
    if np.isfinite(out["z_detector"]) and abs(out["z_detector"]) > cfg.z_detector:
        out["verdict"] = "DETECTOR_FRAME"
        out["reasons"].append(f"flux-centroid coupling in the detector frame, z={out['z_detector']:.1f}")
        return out
    if np.isfinite(p_sector) and p_sector < cfg.p_sector:
        out["verdict"] = "SECTOR_INCONSISTENT"
        out["reasons"].append(f"per-sector slopes disagree with one sky vector (p={p_sector:.1e})")
        return out
    if p_D < cfg.p_blend:
        match = [r for r in nb_rows if r["consistent"]]
        if match:
            best = min(match, key=lambda r: abs(r["D_perp_mas"]) / max(r["sigma_perp"], 1e-9))
            out["verdict"] = "BLEND_NEIGHBOUR"
            out["matched_neighbour"] = best
            out["reasons"].append(f"photocentre moves with flux toward neighbour at "
                                  f"{best['rho_mas']:.0f} mas, PA {best['pa_deg']:.0f}")
        else:
            out["verdict"] = "BLEND_UNRESOLVED"
            out["reasons"].append(f"photocentre moves with flux ({Dmag:.2f} mas per unit dF/F, "
                                  f"p={p_D:.1e}); no catalogued neighbour on that vector")
        return out
    if p_D > cfg.p_null and ul95 <= cfg.ul_on_target_mas:
        out["verdict"] = "ON_TARGET"
        out["reasons"].append(f"no photocentre motion; varying light within {ul95:.2f} mas "
                              f"(x (1-beta)^-1) of the photocentre")
        return out
    out["verdict"] = "AMBIGUOUS"
    out["reasons"].append(f"p_D={p_D:.2e}, ul95={ul95:.1f} mas")
    return out

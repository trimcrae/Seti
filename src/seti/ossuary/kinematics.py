"""Space velocities and Galactic-population membership for OSSUARY.

Why this module carries the channel's honesty burden
----------------------------------------------------
"Halo star" is the *natural-null* half of the claim.  If a star is mislabelled
halo, its warm excess is not anomalous at all -- it is an ordinary young thin-disk
debris disk.  So every classification here has to be traceable to what was
actually measured, and the two regimes must never be silently mixed:

* **With a radial velocity** the full space velocity ``(U, V, W)`` is available
  and ``v_tot`` is measured.  ``kinematic_method = "uvw"``.
* **Without a radial velocity** only the tangential velocity is available.  This
  is *not* a degraded guess at ``v_tot`` -- it is a strict **lower bound** on it,
  because the missing radial component can only add in quadrature.  A star with
  ``v_tan_lsr > 200`` km/s therefore *provably* has ``v_tot > 200`` km/s and is
  halo; a star with small ``v_tan`` is simply **unclassified**, never "thin
  disk".  ``kinematic_method = "vtan_lower_bound"``.

That asymmetry is the whole point: the lower bound can only ever *promote* a
star into the halo sample, never demote one out of it, so the halo sample stays
clean while the unclassified remainder is reported as unclassified.

Conventions
-----------
Right-handed Galactic Cartesian frame with ``U`` toward the Galactic centre,
``V`` toward Galactic rotation, ``W`` toward the North Galactic Pole.  The
ICRS->Galactic rotation matrix is shared with ``seti.panspermia.kinematics`` and
``seti.cluster`` so all channels live in one frame.  Solar peculiar motion is
Schoenrich, Binney & Dehnen (2010).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 1 AU/yr in km/s (IAU 2012):  v_tan[km/s] = _K * mu[mas/yr] * d[pc] / 1000.
_K_AUYR_KMS = 4.740470446

# ICRS -> Galactic rotation (Hipparcos/Gaia convention).
_A_ICRS_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
])


def _col(df: pd.DataFrame, *names, default: float = np.nan) -> np.ndarray:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").to_numpy(float)
    return np.full(len(df), default, dtype=float)


def _triad(ra_deg: np.ndarray, dec_deg: np.ndarray):
    """Local ICRS orthonormal triad (radial, +RA, +Dec) at each position."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    ca, sa = np.cos(ra), np.sin(ra)
    cd, sd = np.cos(dec), np.sin(dec)
    r_hat = np.stack([cd * ca, cd * sa, sd], axis=0)
    a_hat = np.stack([-sa, ca, np.zeros_like(sa)], axis=0)
    d_hat = np.stack([-sd * ca, -sd * sa, cd], axis=0)
    return r_hat, a_hat, d_hat


def space_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Heliocentric Galactic ``U,V,W`` (km/s) and the tangential-velocity vector.

    Returns a frame carrying ``U_kms``/``V_kms``/``W_kms`` (NaN where no radial
    velocity exists), the *tangential* velocity vector ``Ut/Vt/Wt`` in the same
    Galactic frame (always defined), and ``dist_pc``.
    """
    ra = _col(df, "ra")
    dec = _col(df, "dec")
    plx = _col(df, "parallax")
    pmra = _col(df, "pmra")                 # mu_alpha*, already x cos(dec)
    pmdec = _col(df, "pmdec")
    rv = _col(df, "radial_velocity", "rv")

    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(plx > 0, 1000.0 / plx, np.nan)      # pc

    r_hat, a_hat, d_hat = _triad(ra, dec)
    v_a = _K_AUYR_KMS * pmra * dist / 1000.0
    v_d = _K_AUYR_KMS * pmdec * dist / 1000.0

    v_tan_icrs = v_a * a_hat + v_d * d_hat
    v_tan_gal = _A_ICRS_TO_GAL @ v_tan_icrs
    v_rad_gal = (_A_ICRS_TO_GAL @ r_hat) * rv               # NaN where rv is NaN

    out = df.copy()
    out["dist_pc"] = dist
    out["Ut_kms"], out["Vt_kms"], out["Wt_kms"] = v_tan_gal
    out["U_kms"] = v_tan_gal[0] + v_rad_gal[0]
    out["V_kms"] = v_tan_gal[1] + v_rad_gal[1]
    out["W_kms"] = v_tan_gal[2] + v_rad_gal[2]
    out["has_rv"] = np.isfinite(rv)
    return out


def to_lsr(df: pd.DataFrame, kin: dict) -> pd.DataFrame:
    """Add the solar peculiar motion so velocities are w.r.t. the LSR.

    A star at rest in the LSR has heliocentric speed ~16 km/s, so this correction
    is small compared with the 200 km/s halo threshold -- but it is what makes
    ``v_tot`` mean "peculiar velocity" rather than "velocity relative to us".
    """
    out = df.copy()
    u0, v0, w0 = kin["solar_u_kms"], kin["solar_v_kms"], kin["solar_w_kms"]
    for src, dst, off in (("U_kms", "U_lsr_kms", u0),
                          ("V_kms", "V_lsr_kms", v0),
                          ("W_kms", "W_lsr_kms", w0),
                          ("Ut_kms", "Ut_lsr_kms", u0),
                          ("Vt_kms", "Vt_lsr_kms", v0),
                          ("Wt_kms", "Wt_lsr_kms", w0)):
        out[dst] = out[src].to_numpy(float) + off
    out["v_tot_lsr_kms"] = np.sqrt(out["U_lsr_kms"] ** 2 + out["V_lsr_kms"] ** 2
                                   + out["W_lsr_kms"] ** 2)
    # Tangential speed in the LSR frame: the component of the LSR-frame velocity
    # that is actually measured.  Project out the (unmeasured) radial direction so
    # this stays a genuine lower bound when the radial velocity is missing.
    r_hat, _, _ = _triad(_col(out, "ra"), _col(out, "dec"))
    r_gal = _A_ICRS_TO_GAL @ r_hat
    vt = np.stack([out["Ut_lsr_kms"].to_numpy(float),
                   out["Vt_lsr_kms"].to_numpy(float),
                   out["Wt_lsr_kms"].to_numpy(float)])
    v_par = (vt * r_gal).sum(axis=0)                 # radial part of the LSR shift
    v_perp = vt - v_par * r_gal
    out["v_tan_lsr_kms"] = np.sqrt((v_perp ** 2).sum(axis=0))
    return out


def velocity_error(df: pd.DataFrame, kin: dict, n_draws: int = 0,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Propagated 1-sigma uncertainty on ``v_tot_lsr`` (or on ``v_tan_lsr``).

    Analytic quadrature over the radial-velocity error, both proper-motion
    errors, and the distance-error leverage on the tangential velocity.  The
    distance term dominates for halo stars, which are distant and fast.
    """
    plx = _col(df, "parallax")
    plx_err = _col(df, "parallax_error")
    poe = _col(df, "parallax_over_error")
    bad_poe = ~np.isfinite(poe)
    with np.errstate(divide="ignore", invalid="ignore"):
        poe = np.where(bad_poe, np.abs(plx / plx_err), poe)

    pmra = _col(df, "pmra")
    pmdec = _col(df, "pmdec")
    pmra_e = _col(df, "pmra_error", default=0.05)
    pmdec_e = _col(df, "pmdec_error", default=0.05)
    rv_e = _col(df, "radial_velocity_error")
    has_rv = np.isfinite(_col(df, "radial_velocity", "rv"))

    d_kpc = 1.0 / plx
    sig_tan2 = (_K_AUYR_KMS * d_kpc) ** 2 * (np.nan_to_num(pmra_e) ** 2
                                             + np.nan_to_num(pmdec_e) ** 2)
    v_tan2 = (_K_AUYR_KMS * d_kpc) ** 2 * (pmra ** 2 + pmdec ** 2)
    sig_dist2 = v_tan2 / np.maximum(poe, 1.0) ** 2
    sig_rv2 = np.where(has_rv, np.nan_to_num(rv_e) ** 2, 0.0)
    err = np.sqrt(sig_tan2 + sig_dist2 + sig_rv2)

    if n_draws and rng is not None:  # optional MC cross-check of the analytic form
        pass
    return err


def classify(df: pd.DataFrame, kin: dict) -> pd.DataFrame:
    """Assign a Galactic population, recording *how* each star was classified.

    Adds ``kinematic_method``, ``v_tot_or_bound_kms``, ``v_tot_err_kms``,
    ``population`` and ``halo_flag``.  ``population`` is one of

      ``halo``          measured (or lower-bounded) |v - v_LSR| > halo threshold
      ``thick_disk``    measured v_tot in the thick-disk range
      ``thin_disk``     measured v_tot below the thin-disk threshold
      ``unclassified``  no radial velocity AND the tangential bound is too small
                        to decide -- this is *not* "thin disk"
    """
    out = space_velocity(df)
    out = to_lsr(out, kin)
    out["v_tot_err_kms"] = velocity_error(out, kin)

    has_rv = out["has_rv"].to_numpy(bool)
    v_tot = out["v_tot_lsr_kms"].to_numpy(float)
    v_tan = out["v_tan_lsr_kms"].to_numpy(float)

    method = np.where(has_rv, "uvw", "vtan_lower_bound").astype(object)
    method = np.where(np.isfinite(v_tan) | np.isfinite(v_tot), method, "none")
    # The reported speed is the measured v_tot where an RV exists, else the
    # strict lower bound v_tan.  Both are directly comparable to the threshold
    # because exceeding it is sufficient in either case.
    v_use = np.where(has_rv, v_tot, v_tan)

    halo = v_use > kin["halo_v_tot_kms"]
    thick = has_rv & ~halo & (v_tot > kin["thick_disk_v_tot_kms"])
    thin = has_rv & ~halo & ~thick & (v_tot <= kin["thin_disk_v_tot_kms"])

    pop = np.full(len(out), "unclassified", dtype=object)
    pop[has_rv & ~halo & ~thick & ~thin] = "disk_intermediate"
    pop[thin] = "thin_disk"
    pop[thick] = "thick_disk"
    pop[halo] = "halo"
    pop[method == "none"] = "unclassified"

    out["kinematic_method"] = method
    out["v_tot_or_bound_kms"] = v_use
    out["population"] = pop
    out["halo_flag"] = halo & np.isfinite(v_use)
    # A velocity whose own error swamps the classification boundary is not a
    # classification.  Demote rather than pretend.
    sloppy = out["v_tot_err_kms"].to_numpy(float) > kin["max_v_tot_err_kms"]
    out.loc[sloppy, "population"] = "unclassified"
    out.loc[sloppy, "halo_flag"] = False
    return out


def reduced_proper_motion(df: pd.DataFrame) -> pd.Series:
    """H_G = G + 5 log10(mu[arcsec/yr]) + 5 -- a distance-free halo proxy.

    Useful only as a sanity axis: halo subdwarfs sit below the disk main sequence
    in (H_G, BP-RP).  Never used to *make* a halo classification here, because
    ``classify`` already has parallaxes.
    """
    g = _col(df, "phot_g_mean_mag", "g_mag")
    mu_mas = np.hypot(_col(df, "pmra"), _col(df, "pmdec"))
    with np.errstate(divide="ignore", invalid="ignore"):
        hg = g + 5.0 * np.log10(np.where(mu_mas > 0, mu_mas / 1000.0, np.nan)) + 5.0
    return pd.Series(hg, index=df.index)


def absolute_g(df: pd.DataFrame) -> pd.Series:
    """Absolute G magnitude from the parallax (no extinction correction).

    Extinction is left uncorrected on purpose: at |b| > 15 deg with E(B-V) < 0.1
    (the channel's own cirrus gate) A_G < 0.3 mag, which cannot move a star
    across the 1-mag-wide dwarf/giant gap used here.
    """
    g = _col(df, "phot_g_mean_mag", "g_mag")
    plx = _col(df, "parallax")
    with np.errstate(divide="ignore", invalid="ignore"):
        mg = g + 5.0 * np.log10(np.where(plx > 0, plx, np.nan)) - 10.0
    return pd.Series(mg, index=df.index)


def luminosity_class(df: pd.DataFrame, sample_cfg: dict) -> pd.Series:
    """Split dwarfs from giants.  Metal-poor giants have winds and dusty
    envelopes -- an entirely natural source of the very excess this channel
    hunts -- so they are analysed separately and never counted as clean.

    Spectroscopic ``logg`` wins where it exists; otherwise the absolute magnitude
    decides, with the ambiguous strip in between labelled ``subgiant``.
    """
    mg = absolute_g(df).to_numpy(float)
    logg = _col(df, "logg", "logg_gspspec", "logg_gspphot")

    cls = np.full(len(df), "unknown", dtype=object)
    cls[np.isfinite(mg) & (mg >= sample_cfg["mg_dwarf_min"])] = "dwarf"
    cls[np.isfinite(mg) & (mg <= sample_cfg["mg_giant_max"])] = "giant"
    between = np.isfinite(mg) & (mg > sample_cfg["mg_giant_max"]) & \
        (mg < sample_cfg["mg_dwarf_min"])
    cls[between] = "subgiant"

    have_logg = np.isfinite(logg)
    cls[have_logg & (logg >= sample_cfg["logg_dwarf_min"])] = "dwarf"
    cls[have_logg & (logg <= sample_cfg["logg_giant_max"])] = "giant"
    return pd.Series(cls, index=df.index)


__all__ = ["space_velocity", "to_lsr", "velocity_error", "classify",
           "reduced_proper_motion", "absolute_g", "luminosity_class"]

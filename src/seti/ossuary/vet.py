"""The OSSUARY contamination gauntlet.

This channel lives or dies here, and the repository has already been burned:
15 of 15 raw infrared-excess candidates in an earlier channel were WISE W4 /
blend systematics, and the entire published warm-Dyson candidate list
(Hephaistos A-J) has now collapsed under JWST/MIRI follow-up -- candidates D and
E resolved into a Hot DOG at z~0.9 and a dusty starburst at z~0.4, both within
~1 arcsec (Hephaistos IV, arXiv:2607.09460).

So background-galaxy confusion is treated here as a **funnel stage with a number
attached**, not as follow-up.  The three legs:

1. **Astrometric registration.**  Propagate the Gaia position to the AllWISE mean
   epoch using the Gaia proper motion and require the WISE centroid to sit there
   to sub-arcsecond precision.  Halo stars have *large* proper motions -- that is
   what makes them halo stars -- so over the 5.5 yr Gaia-to-AllWISE baseline a
   200 mas/yr star moves 1.1 arcsec.  The test therefore has real leverage on
   exactly the sample this channel selects, and a static background galaxy fails
   it.  (Skipping the propagation does not merely weaken the test: it makes the
   crossmatch itself silently return nothing.  That bug cost a previous channel a
   whole run.)
2. **Co-movement.**  Where CatWISE2020 measures its own proper motion, it must
   agree with Gaia's.  A zero-proper-motion infrared source sitting on a
   high-proper-motion star is a background object, full stop.
3. **Chance-superposition prior.**  For each candidate, compute the sky density
   of AllWISE sources at least as bright as the *excess itself* and turn it into
   an a-priori probability of an interloper inside the registration radius.  A
   candidate whose excess could be supplied by a run-of-the-mill background
   source at p > 1e-3 is not evidence of anything.

A note on Galactic latitude, stated honestly because it is easy to overclaim:
the halo sample sits at high |b|, and that is a large real advantage against
**Galactic cirrus** and against **stellar** blending, because both fall steeply
away from the plane.  It is *not* an advantage against extragalactic confusion --
if anything high |b| means slightly more visible galaxies per square degree, not
fewer.  Legs 1-3 above are what handle the extragalactic case; latitude is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..discriminate.blend import _gmag_to_w1_approx, _vega_mag_to_flux

MAS_PER_DEG = 3.6e6


# --------------------------------------------------------------------------
# 1. Catalogue quality
# --------------------------------------------------------------------------

def wise_quality_gate(df: pd.DataFrame, cfg: dict,
                      bands=("W1", "W2")) -> pd.DataFrame:
    """AllWISE photometric-quality flags, per band, plus crossmatch ambiguity.

    ``ph_qual`` is a 4-character string (W1W2W3W4); ``cc_flags`` likewise, where a
    letter (D/P/H/O) marks contamination by a diffraction spike, persistence
    artefact, scattered-light halo, or optical ghost.
    """
    q = cfg["wise_quality"]
    n = len(df)
    ph = df.get("ph_qual", pd.Series([""] * n, index=df.index)).astype(str)
    cc = df.get("cc_flags", pd.Series([""] * n, index=df.index)).astype(str)
    allowed = set(q["allowed_ph_qual"])
    forbidden = set(q["forbidden_cc"].upper())
    order = ["W1", "W2", "W3", "W4"]

    out = pd.DataFrame(index=df.index)
    ok = pd.Series(True, index=df.index)
    for b in bands:
        i = order.index(b)
        pb = ph.str.slice(i, i + 1).str.upper()
        cb = cc.str.slice(i, i + 1).str.upper()
        # An absent flag string is unknown, not bad: do not silently reject.
        pb_ok = pb.isin(allowed) | (ph.str.len() < i + 1)
        cb_ok = ~cb.isin(forbidden) | (cc.str.len() < i + 1)
        out[f"ph_qual_{b}_ok"] = pb_ok
        out[f"cc_flag_{b}_ok"] = cb_ok
        ok &= pb_ok & cb_ok

    w1 = pd.to_numeric(df.get("W1mag"), errors="coerce")
    out["not_saturated"] = ~(w1 < q["w1_bright_limit_mag"]).fillna(False)
    ok &= out["not_saturated"]

    mates = pd.to_numeric(df.get("number_of_mates"), errors="coerce")
    neigh = pd.to_numeric(df.get("number_of_neighbours"), errors="coerce")
    out["xmatch_unambiguous"] = (
        (mates.fillna(0) <= q["max_number_of_mates"])
        & (neigh.fillna(1) <= q["max_number_of_neighbours"]))
    ok &= out["xmatch_unambiguous"]

    ext = pd.to_numeric(df.get("ext_flag"), errors="coerce")
    out["not_extended"] = (ext.fillna(0) <= 0)
    ok &= out["not_extended"]

    out["wise_quality_ok"] = ok
    return out


# --------------------------------------------------------------------------
# 2. Inherited contamination ledger
# --------------------------------------------------------------------------

def ledger_gate(df: pd.DataFrame, excess_cfg: dict, sample_cfg: dict) -> pd.DataFrame:
    """The rules the repository already paid for, applied verbatim.

    * A W4-only excess is cirrus.  A real warm-dust SED lights the star-dominated
      bands first, so the detection must appear in W1-W2 or W1-W3.
    * A negative W1-W2 is a blend, not a photosphere.
    * G >~ 18 is WISE-confusion-limited; small bp_rp leaves the empirical
      photosphere locus unanchored.
    * W1-W2 above the AGN wedge means the beam is dominated by something that is
      not the star.
    """
    out = pd.DataFrame(index=df.index)
    chi_min = float(excess_cfg["chi_min"])

    star_band = pd.Series(False, index=df.index)
    for b in excess_cfg["require_bands"]:
        if f"chi_{b}" in df.columns:
            star_band = star_band | ((df[f"chi_{b}"] >= chi_min)
                                     & (df.get(f"{b}_excess_jy", 0) > 0))
    w4 = pd.Series(False, index=df.index)
    if "chi_W4" in df.columns:
        w4 = ((df["chi_W4"] >= chi_min) & (df.get("W4_excess_jy", 0) > 0)).fillna(False)

    out["w4_only"] = (w4 & ~star_band).fillna(False)
    out["star_band_excess"] = star_band.fillna(False)

    w1w2 = pd.to_numeric(df.get("w1_w2_obs"), errors="coerce")
    if w1w2.isna().all() and {"W1mag", "W2mag"} <= set(df.columns):
        w1w2 = pd.to_numeric(df["W1mag"], errors="coerce") - \
            pd.to_numeric(df["W2mag"], errors="coerce")
    out["w1_w2_physical"] = (w1w2 >= excess_cfg["w1_w2_min"]).fillna(False)
    out["not_agn_wedge"] = (w1w2 <= excess_cfg["w1_w2_agn_max"]).fillna(False)

    g = pd.to_numeric(df.get("phot_g_mean_mag", df.get("g_mag")), errors="coerce")
    out["not_confusion_limited"] = (g <= sample_cfg["g_max"]).fillna(False)

    bprp = pd.to_numeric(df.get("bp_rp"), errors="coerce")
    out["locus_anchored"] = ((bprp >= sample_cfg["bp_rp_min"])
                             & (bprp <= sample_cfg["bp_rp_max"])).fillna(False)

    out["ledger_ok"] = (out["star_band_excess"] & ~out["w4_only"]
                        & out["w1_w2_physical"] & out["not_agn_wedge"]
                        & out["not_confusion_limited"] & out["locus_anchored"])
    return out


# --------------------------------------------------------------------------
# 3. Unresolved cool companion
# --------------------------------------------------------------------------

def companion_gate(df: pd.DataFrame, excess_cfg: dict) -> pd.DataFrame:
    """Reject unresolved cool companions masquerading as dust.

    Two independent handles:

    * a fitted excess temperature above ~1800 K is hotter than silicate or
      carbonaceous grains survive, so it is a companion photosphere;
    * a companion also emits in J/H/Ks, so a genuine circumstellar excess must
      leave the near-infrared bands *on* the photosphere locus.  A significant
      J/H excess is a companion signature, and it is independent of the
      temperature test.
    """
    out = pd.DataFrame(index=df.index)
    t = pd.to_numeric(df.get("t_dust_k"), errors="coerce")
    out["t_dust_too_hot"] = (t > excess_cfg["t_dust_max_k"]).fillna(False)
    out["t_dust_too_cold"] = (t < excess_cfg["t_dust_min_k"]).fillna(False)

    nir = pd.Series(False, index=df.index)
    for b in ("J", "H"):
        c = f"chi_{b}"
        if c in df.columns:
            nir = nir | (pd.to_numeric(df[c], errors="coerce")
                         > excess_cfg["nir_excess_sigma_max"]).fillna(False)
    out["nir_excess"] = nir
    out["companion_ok"] = ~(out["t_dust_too_hot"] | out["nir_excess"])
    return out


# --------------------------------------------------------------------------
# 4. Astrometric registration + co-movement (the extragalactic test)
# --------------------------------------------------------------------------

def registration_offset_arcsec(df: pd.DataFrame, astro_cfg: dict,
                               propagate: bool = True) -> pd.Series:
    """Offset (arcsec) between the WISE centroid and the Gaia position.

    With ``propagate=True`` the Gaia position is first moved to the AllWISE mean
    epoch using the Gaia proper motion -- which is mandatory, not optional, for a
    high-proper-motion halo sample.  ``propagate=False`` returns the naive offset
    and exists so the pipeline can *report* how much the propagation mattered
    rather than assert that it did.
    """
    dt = astro_cfg["wise_mean_epoch"] - astro_cfg["gaia_ref_epoch"]
    ra = pd.to_numeric(df.get("ra"), errors="coerce").to_numpy(float)
    dec = pd.to_numeric(df.get("dec"), errors="coerce").to_numpy(float)
    ra_w = pd.to_numeric(df.get("ra_wise"), errors="coerce").to_numpy(float)
    dec_w = pd.to_numeric(df.get("dec_wise"), errors="coerce").to_numpy(float)
    pmra = pd.to_numeric(df.get("pmra"), errors="coerce").to_numpy(float)
    pmdec = pd.to_numeric(df.get("pmdec"), errors="coerce").to_numpy(float)

    cosd = np.cos(np.radians(dec))
    obs_dra = (ra_w - ra) * MAS_PER_DEG * cosd
    obs_ddec = (dec_w - dec) * MAS_PER_DEG
    if propagate:
        obs_dra = obs_dra - np.nan_to_num(pmra) * dt
        obs_ddec = obs_ddec - np.nan_to_num(pmdec) * dt
    return pd.Series(np.hypot(obs_dra, obs_ddec) / 1000.0, index=df.index)


def astrometry_gate(df: pd.DataFrame, astro_cfg: dict) -> pd.DataFrame:
    """Sub-arcsecond registration at the WISE epoch, plus Gaia/CatWISE co-movement."""
    out = pd.DataFrame(index=df.index)
    have_pos = {"ra_wise", "dec_wise"} <= set(df.columns)
    if not have_pos:
        # Cannot test.  Say so; do not silently pass and do not silently reject.
        out["registration_arcsec"] = np.nan
        out["registration_arcsec_unpropagated"] = np.nan
        out["registration_tested"] = False
        out["registration_ok"] = False
        out["pm_leverage_mas"] = np.nan
        out["comovement_sigma"] = np.nan
        out["comovement_ok"] = True
        out["astrometry_ok"] = False
        return out

    off = registration_offset_arcsec(df, astro_cfg, propagate=True)
    off_raw = registration_offset_arcsec(df, astro_cfg, propagate=False)
    out["registration_arcsec"] = off
    out["registration_arcsec_unpropagated"] = off_raw
    out["registration_tested"] = True
    out["registration_ok"] = (off <= astro_cfg["max_registration_arcsec"]).fillna(False)

    dt = abs(astro_cfg["wise_mean_epoch"] - astro_cfg["gaia_ref_epoch"])
    mu = np.hypot(pd.to_numeric(df.get("pmra"), errors="coerce"),
                  pd.to_numeric(df.get("pmdec"), errors="coerce"))
    out["pm_leverage_mas"] = mu * dt
    out["pm_has_leverage"] = (mu >= astro_cfg["min_pm_for_leverage_mas_yr"]).fillna(False)

    sigma = pd.Series(np.nan, index=df.index)
    if {"pmra_wise", "pmdec_wise"} <= set(df.columns):
        e_r = pd.to_numeric(df.get("e_pmra_wise"), errors="coerce").fillna(50.0)
        e_d = pd.to_numeric(df.get("e_pmdec_wise"), errors="coerce").fillna(50.0)
        g_r = pd.to_numeric(df.get("pmra_error"), errors="coerce").fillna(1.0)
        g_d = pd.to_numeric(df.get("pmdec_error"), errors="coerce").fillna(1.0)
        d_r = (pd.to_numeric(df["pmra"], errors="coerce")
               - pd.to_numeric(df["pmra_wise"], errors="coerce")) / np.hypot(e_r, g_r)
        d_d = (pd.to_numeric(df["pmdec"], errors="coerce")
               - pd.to_numeric(df["pmdec_wise"], errors="coerce")) / np.hypot(e_d, g_d)
        sigma = np.hypot(d_r, d_d)
    out["comovement_sigma"] = sigma
    cm_ok = pd.Series(True, index=df.index)
    have = sigma.notna()
    cm_ok[have] = sigma[have] <= astro_cfg["pm_consistency_sigma_max"]
    out["comovement_ok"] = cm_ok
    out["astrometry_ok"] = out["registration_ok"] & cm_ok
    return out


# --------------------------------------------------------------------------
# 5. Chance-superposition prior
# --------------------------------------------------------------------------

def allwise_source_density_per_arcsec2(w1_mag: np.ndarray, cfg: dict) -> np.ndarray:
    """Cumulative AllWISE source density brighter than ``w1_mag``, per arcsec^2.

    ``log10 N(<W1) [deg^-2] = norm + slope * (W1 - ref)``, anchored on the
    catalogue's own totals (747e6 sources over 41253 deg^2 at the W1 5-sigma
    depth of 16.9 mag).  The slope is the shallow-counts compromise between the
    Euclidean galaxy slope and the flatter high-latitude stellar counts.
    """
    e = cfg["extragalactic"]
    log_n_deg2 = e["counts_log10_norm_per_deg2"] + \
        e["counts_slope_per_mag"] * (np.asarray(w1_mag, float) - e["counts_ref_w1_mag"])
    return 10.0 ** log_n_deg2 / (3600.0 ** 2)


def chance_superposition_p(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Probability an unrelated AllWISE source bright enough to supply the excess
    falls inside the registration radius by chance.

    The relevant interloper is not "any WISE source" but one at least as bright as
    the *excess flux itself* -- a fainter one cannot produce the signal.  Turning
    the excess into an equivalent W1 magnitude and reading the source counts there
    is what makes this a per-candidate number instead of a global hand-wave.
    """
    from ..photometry import BANDS

    exc_jy = pd.to_numeric(df.get("W1_excess_jy"), errors="coerce").to_numpy(float)
    # If W1 carries no excess, use the largest positive short-band excess.
    for b in ("W2", "W3"):
        col = f"{b}_excess_jy"
        if col in df.columns:
            alt = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            exc_jy = np.where(np.isfinite(exc_jy) & (exc_jy > 0), exc_jy, alt)
    with np.errstate(divide="ignore", invalid="ignore"):
        w1_equiv = -2.5 * np.log10(np.where(exc_jy > 0, exc_jy, np.nan)
                                   / BANDS["W1"]["zp_jy"])

    dens = allwise_source_density_per_arcsec2(w1_equiv, cfg)
    r = float(cfg["astrometry"]["max_registration_arcsec"])
    area = np.pi * r ** 2
    p = 1.0 - np.exp(-dens * area)
    return pd.Series(p, index=df.index)


def extragalactic_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Chance-superposition prior + the AllWISE extended-source flag."""
    out = pd.DataFrame(index=df.index)
    p = chance_superposition_p(df, cfg)
    out["chance_superposition_p"] = p
    out["chance_superposition_ok"] = (
        p <= cfg["extragalactic"]["max_chance_superposition_p"]).fillna(False)
    ext = pd.to_numeric(df.get("ext_flag"), errors="coerce")
    out["not_wise_extended"] = (ext.fillna(0) <= 0)
    out["extragalactic_ok"] = out["chance_superposition_ok"] & out["not_wise_extended"]
    return out


# --------------------------------------------------------------------------
# 6. Galactic cirrus
# --------------------------------------------------------------------------

def cirrus_gate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reject sightlines where the 12/22 um background is Galactic cirrus.

    Per-object: an SFD reddening above the threshold, or a low Galactic latitude,
    means the WISE long-band flux is dominated by diffuse emission rather than by
    the star.  ``ebv_sfd`` is fetched per candidate on the runner; where it is
    absent the object is marked untested rather than passed.
    """
    c = cfg["cirrus"]
    out = pd.DataFrame(index=df.index)
    ebv = pd.to_numeric(df.get("ebv_sfd"), errors="coerce")
    b = pd.to_numeric(df.get("b"), errors="coerce")
    if b.isna().all() and "gal_b" in df.columns:
        b = pd.to_numeric(df["gal_b"], errors="coerce")

    out["ebv_sfd"] = ebv
    out["cirrus_tested"] = ebv.notna()
    out["ebv_ok"] = (ebv <= c["max_ebv_sfd"]).fillna(False)
    out["gal_lat_ok"] = (b.abs() >= c["min_gal_lat_deg"]).fillna(False)
    out["cirrus_ok"] = out["ebv_ok"] & out["gal_lat_ok"]
    return out


def cirrus_correlation_test(df: pd.DataFrame, flag_col: str = "excess_flag",
                            ebv_col: str = "ebv_sfd") -> dict:
    """Population-level cirrus test: does the flag rate track the reddening?

    A per-object threshold cannot detect a *statistical* cirrus leak.  If the
    candidates preferentially occupy dusty sightlines, the excess is Galactic
    foreground however clean each individual object looks.  Reported as a
    Spearman correlation with its p-value plus the flag rate in reddening
    quartiles, so the leak is visible even when it is not significant.
    """
    from scipy import stats

    sub = df[[c for c in (flag_col, ebv_col) if c in df.columns]].dropna()
    if flag_col not in df.columns or ebv_col not in df.columns or len(sub) < 20:
        return {"tested": False, "reason": "insufficient E(B-V) coverage",
                "n": int(len(sub))}
    flag = sub[flag_col].astype(float).to_numpy()
    ebv = sub[ebv_col].astype(float).to_numpy()
    if flag.std() == 0 or ebv.std() == 0:
        return {"tested": False, "reason": "degenerate (no variance)",
                "n": int(len(sub))}
    rho, p = stats.spearmanr(ebv, flag)
    try:
        q = pd.qcut(pd.Series(ebv), 4, labels=False, duplicates="drop")
        rates = [float(flag[q == k].mean()) for k in sorted(pd.unique(q))
                 if np.isfinite(k)]
    except ValueError:
        rates = []
    return {"tested": True, "n": int(len(sub)), "spearman_rho": float(rho),
            "p_value": float(p), "flag_rate_by_ebv_quartile": rates}


# --------------------------------------------------------------------------
# 6b. Astrophysical impostors specific to metal-poor / halo samples
# --------------------------------------------------------------------------

# The largest globular clusters by angular size, as an offline fallback for the
# cluster veto when the VizieR Harris catalogue cannot be reached.  (name, RA,
# Dec, half-light radius in arcmin.)
_BRIGHT_GLOBULARS = [
    ("NGC 5139 (omega Cen)", 201.697, -47.480, 5.00),
    ("NGC 104 (47 Tuc)", 6.024, -72.081, 3.17),
    ("NGC 6121 (M4)", 245.897, -26.526, 4.33),
    ("NGC 6397", 265.175, -53.674, 2.90),
    ("NGC 6656 (M22)", 279.100, -23.905, 3.36),
    ("NGC 6752", 287.717, -59.985, 1.91),
    ("NGC 5272 (M3)", 205.548, +28.377, 2.31),
    ("NGC 6205 (M13)", 250.422, +36.460, 1.69),
    ("NGC 7078 (M15)", 322.493, +12.167, 1.00),
    ("NGC 7089 (M2)", 323.363, -0.823, 1.06),
    ("NGC 5904 (M5)", 229.638, +2.081, 1.77),
    ("NGC 362", 15.809, -70.849, 0.82),
    ("NGC 3201", 154.403, -46.412, 3.10),
    ("NGC 6218 (M12)", 251.809, -1.949, 1.77),
    ("NGC 6254 (M10)", 254.288, -4.100, 1.95),
    ("NGC 288", 13.189, -26.583, 2.23),
    ("NGC 6809 (M55)", 294.999, -30.965, 2.83),
    ("NGC 6541", 272.010, -43.715, 1.06),
    ("NGC 1851", 78.528, -40.047, 0.51),
    ("NGC 2808", 138.013, -64.863, 0.80),
]


def globular_cluster_veto(df: pd.DataFrame, cfg: dict,
                          clusters: pd.DataFrame | None = None) -> pd.DataFrame:
    """Veto sightlines toward globular clusters.

    Boyer et al. 2010 (arXiv:1002.1348) demonstrated that a *published*
    red-giant-branch-wide infrared excess across 47 Tuc was entirely stellar
    blending and imaging artefacts, using the same archival Spitzer imagery as
    the original claim.  Metal-poor stars concentrate toward globular clusters,
    which are the most crowded fields on the sky, so this channel does not try to
    vet cluster sightlines -- it removes them.
    """
    g = cfg["globular_cluster"]
    if clusters is None or not len(clusters):
        clusters = pd.DataFrame(_BRIGHT_GLOBULARS,
                                columns=["name", "ra", "dec", "rh_arcmin"])
    out = pd.DataFrame(index=df.index)
    ra = pd.to_numeric(df.get("ra"), errors="coerce").to_numpy(float)
    dec = pd.to_numeric(df.get("dec"), errors="coerce").to_numpy(float)

    nearest = np.full(len(df), np.inf)
    which = np.full(len(df), "", dtype=object)
    for _, c in clusters.iterrows():
        rh = float(c.get("rh_arcmin", np.nan))
        veto_arcmin = max(float(g["veto_radius_arcmin"]),
                          g["veto_radius_multiplier"] * rh if np.isfinite(rh) else 0.0)
        dra = (ra - float(c["ra"])) * np.cos(np.radians(dec))
        dde = dec - float(c["dec"])
        sep_arcmin = np.hypot(dra, dde) * 60.0
        closer = sep_arcmin < nearest
        nearest = np.where(closer, sep_arcmin, nearest)
        which = np.where(closer & (sep_arcmin < veto_arcmin), str(c["name"]), which)
    out["nearest_globular_arcmin"] = nearest
    out["globular_cluster"] = which
    out["globular_ok"] = which == ""
    return out


def impostor_gate(df: pd.DataFrame, sample_cfg: dict) -> pd.DataFrame:
    """The three impostors this sample selects for, from the literature sweep.

    * **lambda Bootis stars** -- A/early-F stars with a metal-depleted *surface*
      from accreting gas-depleted ISM, 21 of 34 of which carry infrared excesses
      (Murphy et al. 2020).  Some were previously catalogued as blue horizontal
      branch stars, so they arrive wearing exactly this channel's badge.  They are
      hot; a T_eff ceiling removes them.
    * **Blue stragglers / merger products** -- Yong et al. 2016 found two to three
      alpha-rich metal-poor "young" giants with debris-disk-like excesses that are
      evolved blue stragglers.  Flagged where a star sits blueward and brighter
      than the metal-poor turnoff.
    * **sdA halo binaries** -- Brown et al. 2017 showed the majority of sdA stars
      are metal-poor halo A-F stars with ~0.8 Msun companions, several with
      infrared excess.  The T_eff ceiling and the RUWE cut both bear on these.
    """
    out = pd.DataFrame(index=df.index)
    teff = pd.to_numeric(df.get("teff"), errors="coerce")
    if teff.isna().all():
        for alt in ("teff_gspspec", "teff_gspphot"):
            if alt in df.columns:
                teff = pd.to_numeric(df[alt], errors="coerce")
                if teff.notna().any():
                    break
    out["teff"] = teff
    hot = (teff > sample_cfg["teff_max_k"]).fillna(False)
    cold = (teff < sample_cfg["teff_min_k"]).fillna(False)
    out["lambda_boo_risk"] = hot
    out["teff_in_range"] = ~(hot | cold)

    # Blue straggler locus: brighter than M_G ~ 4.5 while bluer than BP-RP ~ 0.65
    # is above and blueward of the metal-poor main-sequence turnoff.
    mg = pd.to_numeric(df.get("M_G"), errors="coerce")
    bprp = pd.to_numeric(df.get("bp_rp"), errors="coerce")
    out["blue_straggler_risk"] = ((mg < 4.5) & (bprp < 0.65)).fillna(False)

    otype = df.get("simbad_otype", pd.Series("", index=df.index)).astype(str).str.lower()
    out["simbad_impostor"] = otype.str.contains(
        "lam boo|lambda boo|rr lyr|bs\\*|blue strag|agb|post-agb|c\\*|carbon|"
        "yso|t tau|herbig|em\\*|agn|qso|seyfert|galaxy|glob", regex=True)

    out["impostor_ok"] = (out["teff_in_range"] & ~out["blue_straggler_risk"]
                          & ~out["simbad_impostor"])
    return out


def expected_chance_alignments(n_stars: int, cfg: dict,
                               w1_excess_mag: float = 12.0) -> dict:
    """Expected number of chance extragalactic alignments for the whole sample.

    Two independent estimates, because the Hephaistos post-mortem gives a
    directly measured density for the specific population that killed it:

    * **Hot DOGs** at 9e-6 arcsec^-2 -- the density that "can probably account for
      the contamination of all 7" Hephaistos candidates (arXiv:2405.14921), two of
      which JWST/MIRI later resolved into a z~0.9 Hot DOG and a z~0.4 dusty
      starburst within ~1 arcsec (arXiv:2607.09460).
    * **All AllWISE sources** bright enough to supply the excess, from the counts
      law in :func:`allwise_source_density_per_arcsec2`.

    If the expectation is much less than 1 the sample is in a genuinely strong
    position; if it is much greater than 1, nothing survives without the
    astrometric-registration stage.
    """
    r = float(cfg["astrometry"]["max_registration_arcsec"])
    area = np.pi * r ** 2
    hot_dog_density = float(cfg["extragalactic"].get(
        "hot_dog_density_per_arcsec2", 9.0e-6))
    n_hd = n_stars * hot_dog_density * area
    dens = float(allwise_source_density_per_arcsec2(np.array([w1_excess_mag]), cfg)[0])
    n_all = n_stars * dens * area
    # For contrast: what the same sample would suffer without the registration cut.
    beam = float(cfg["beam"]["wise_beam_arcsec"])
    n_all_beam = n_stars * dens * np.pi * beam ** 2
    return {
        "n_stars": int(n_stars),
        "registration_radius_arcsec": r,
        "expected_hot_dogs": float(n_hd),
        "expected_allwise_interlopers": float(n_all),
        "expected_allwise_interlopers_in_full_beam": float(n_all_beam),
        "assumed_excess_equivalent_w1_mag": float(w1_excess_mag),
        "leverage_of_registration_cut": float(n_all_beam / n_all) if n_all > 0 else np.nan,
    }


# --------------------------------------------------------------------------
# 7. Beam blending
# --------------------------------------------------------------------------

def beam_blend_verdict(candidate: dict, neighbours: pd.DataFrame,
                       cfg: dict) -> dict:
    """Can a Gaia neighbour inside the WISE beam supply the observed excess?

    Unlike the white-dwarf version of this test (``discriminate.blend``), the
    denominator here is the star's *measured* excess flux rather than its total
    W1 flux: a main-sequence star dominates its own beam easily, so the question
    is never "does a neighbour outshine it" but "can a neighbour account for the
    small extra flux we are calling a technosignature".  A neighbour 4 magnitudes
    fainter is irrelevant to the total and decisive for a 3% excess.
    """
    beam = float(cfg["beam"]["wise_beam_arcsec"])
    frac_max = float(cfg["beam"]["max_neighbour_flux_frac"])
    ra0, dec0 = float(candidate["ra"]), float(candidate["dec"])
    excess_jy = float(candidate.get("W1_excess_jy", np.nan))
    w1_obs_jy = float(candidate.get("W1_obs_jy", np.nan))

    if neighbours is None or not len(neighbours):
        return {"blend_verdict": "isolated", "n_beam_neighbours": 0,
                "neighbour_flux_frac": 0.0, "neighbour_over_excess": 0.0,
                "nearest_neighbour_arcsec": np.nan}

    nb = neighbours.copy()
    dra = (pd.to_numeric(nb["ra"], errors="coerce") - ra0) * np.cos(np.radians(dec0))
    dde = pd.to_numeric(nb["dec"], errors="coerce") - dec0
    sep = np.hypot(dra, dde) * 3600.0
    nb = nb.assign(_sep=sep)
    if "source_id" in nb.columns and "source_id" in candidate:
        nb = nb[nb["source_id"] != candidate["source_id"]]
    inbeam = nb[nb["_sep"] <= beam]
    if not len(inbeam):
        return {"blend_verdict": "isolated", "n_beam_neighbours": 0,
                "neighbour_flux_frac": 0.0, "neighbour_over_excess": 0.0,
                "nearest_neighbour_arcsec": float(np.nanmin(sep)) if len(sep) else np.nan}

    ng = pd.to_numeric(inbeam["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    nc = pd.to_numeric(inbeam.get("bp_rp"), errors="coerce").to_numpy(float)
    nb_w1_jy = _vega_mag_to_flux(_gmag_to_w1_approx(ng, nc)) * 309.540  # -> Jy
    total_nb = float(np.nansum(nb_w1_jy))

    frac = total_nb / w1_obs_jy if np.isfinite(w1_obs_jy) and w1_obs_jy > 0 else np.inf
    over = total_nb / excess_jy if np.isfinite(excess_jy) and excess_jy > 0 else np.inf

    if over >= 1.0 or frac >= frac_max:
        verdict = "beam_blend"
    else:
        verdict = "clean"
    return {"blend_verdict": verdict, "n_beam_neighbours": int(len(inbeam)),
            "neighbour_flux_frac": float(frac), "neighbour_over_excess": float(over),
            "nearest_neighbour_arcsec": float(inbeam["_sep"].min())}


# --------------------------------------------------------------------------
# 8. The gauntlet
# --------------------------------------------------------------------------

_ORDER = [
    ("wise_quality_ok", "wise_quality"),
    ("ledger_ok", "ledger"),
    ("companion_ok", "unresolved_companion"),
    ("astrometry_ok", "astrometric_registration"),
    ("extragalactic_ok", "background_source"),
    ("cirrus_ok", "galactic_cirrus"),
    ("globular_ok", "globular_cluster_sightline"),
    ("impostor_ok", "lambda_boo_or_blue_straggler"),
    ("luminosity_ok", "giant_or_unclassified"),
    ("kinematics_ok", "not_a_null_reservoir_host"),
]


def vet(df: pd.DataFrame, cfg: dict, sample_cfg: dict,
        excess_cfg: dict, kin_cfg: dict) -> pd.DataFrame:
    """Run every gate and assign a single verdict plus the first failing reason.

    The verdict is ``surviving`` only when every applicable gate passes.  The
    ``reject_reason`` names the *first* gate that failed in a fixed order, so the
    funnel is reportable as counts per stage rather than as a single number.
    """
    out = df.copy()
    for frame in (wise_quality_gate(df, cfg),
                  ledger_gate(df, excess_cfg, sample_cfg),
                  companion_gate(df, excess_cfg),
                  astrometry_gate(df, cfg["astrometry"]),
                  extragalactic_gate(df, cfg),
                  cirrus_gate(df, cfg),
                  globular_cluster_veto(df, cfg),
                  impostor_gate(df, sample_cfg)):
        for c in frame.columns:
            out[c] = frame[c]

    lum = out.get("luminosity_class", pd.Series("unknown", index=out.index))
    out["luminosity_ok"] = (lum == "dwarf")

    # The sample's whole point: the host must have no natural reservoir.  A star
    # that is neither metal-poor nor halo-kinematic is not part of the claim even
    # if its excess is real.
    feh = pd.to_numeric(out.get("feh"), errors="coerce")
    metal_poor = (feh <= sample_cfg["feh_max"]).fillna(False)
    halo = out.get("halo_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["metal_poor"] = metal_poor
    out["null_reservoir_host"] = metal_poor | halo
    out["kinematics_ok"] = out["null_reservoir_host"]

    reason = pd.Series("", index=out.index, dtype=object)
    for col, name in _ORDER:
        if col not in out.columns:
            continue
        failed = ~out[col].fillna(False).astype(bool) & (reason == "")
        reason[failed] = name
    out["reject_reason"] = reason
    out["verdict"] = np.where(reason == "", "surviving", "rejected")
    return out


def funnel_counts(df: pd.DataFrame) -> dict:
    """Survivors after each gate, and how many each gate removed.

    Silverberg et al. 2018 measured that at most 7.9 +/- 0.2 % of AllWISE-selected
    infrared excesses are good disk candidates -- a ~92% false-positive rate, with
    the McDonald and Marton searches above 70% and *all thirteen* Theissen & West
    candidates with W4 S/N > 3 spurious.  So roughly nine in ten raw flags here are
    expected to be junk, and the funnel is only credible if it can say which stage
    removed them.  Hence the per-stage removals, not just the running total.
    """
    counts = {"input": int(len(df))}
    removed = {}
    alive = pd.Series(True, index=df.index)
    for col, name in _ORDER:
        before = int(alive.sum())
        if col in df.columns:
            alive = alive & df[col].fillna(False).astype(bool)
        counts[f"after_{name}"] = int(alive.sum())
        removed[name] = before - int(alive.sum())
    counts["surviving"] = int(alive.sum())
    counts["removed_by_stage"] = removed
    n_in = counts["input"]
    counts["fraction_removed"] = float(1.0 - counts["surviving"] / n_in) if n_in else 0.0
    return counts


__all__ = ["wise_quality_gate", "ledger_gate", "companion_gate",
           "registration_offset_arcsec", "astrometry_gate",
           "allwise_source_density_per_arcsec2", "chance_superposition_p",
           "extragalactic_gate", "cirrus_gate", "cirrus_correlation_test",
           "globular_cluster_veto", "impostor_gate", "expected_chance_alignments",
           "beam_blend_verdict", "vet", "funnel_counts"]

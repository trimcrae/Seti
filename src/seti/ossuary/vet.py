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
                  cirrus_gate(df, cfg)):
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
    """Survivors remaining after each gate, applied cumulatively in order."""
    counts = {"input": int(len(df))}
    alive = pd.Series(True, index=df.index)
    for col, name in _ORDER:
        if col not in df.columns:
            counts[f"after_{name}"] = int(alive.sum())
            continue
        alive = alive & df[col].fillna(False).astype(bool)
        counts[f"after_{name}"] = int(alive.sum())
    counts["surviving"] = int(alive.sum())
    return counts


__all__ = ["wise_quality_gate", "ledger_gate", "companion_gate",
           "registration_offset_arcsec", "astrometry_gate",
           "allwise_source_density_per_arcsec2", "chance_superposition_p",
           "extragalactic_gate", "cirrus_gate", "cirrus_correlation_test",
           "beam_blend_verdict", "vet", "funnel_counts"]

"""Independent follow-up characterization of a top dimming candidate.

Runs the decisive checks a survivor of the ZTF funnel still needs before it can
be taken seriously: the Gaia DR3 astrometric/variability record (to rule out an
unresolved binary, a blend, or an already-classified variable, and to place it on
the HR diagram with a real distance), and --- where reachable --- an independent
survey light curve (ASAS-SN) to confirm the fade is not a ZTF-specific artefact.

Everything is defensive: an unreachable service yields ``None`` for that axis.
"""

from __future__ import annotations

import numpy as np


def fetch_gaia_dr3(ra: float, dec: float, radius_arcsec: float = 3.0) -> dict | None:
    """Nearest Gaia DR3 source: astrometry, photometry, variability, binarity."""
    from astroquery.gaia import Gaia

    q = f"""
        SELECT TOP 5 source_id, ra, dec, parallax, parallax_over_error, ruwe,
               pmra, pmdec, pmra_error, pmdec_error, phot_g_mean_mag, bp_rp,
               phot_variable_flag, non_single_star,
               DISTANCE(POINT('ICRS', ra, dec),
                        POINT('ICRS', {ra}, {dec})) AS d
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                           CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600.0}))
        ORDER BY d ASC
    """
    try:
        df = Gaia.launch_job_async(q).get_results().to_pandas()
    except Exception as exc:
        print(f"[characterize] Gaia DR3 query failed: {exc!r}")
        return None
    if df.empty:
        return None
    df = df.rename(columns={c: c.lower() for c in df.columns})
    r = df.iloc[0]
    plx = float(r.get("parallax") or np.nan)
    g = float(r.get("phot_g_mean_mag") or np.nan)
    dist_pc = (1000.0 / plx) if np.isfinite(plx) and plx > 0 else np.nan
    m_g = (g + 5.0 * np.log10(plx / 100.0)) if np.isfinite(plx) and plx > 0 else np.nan
    return {
        "gaia_source_id": int(r.get("source_id")),
        "parallax_mas": plx, "parallax_over_error": float(r.get("parallax_over_error") or np.nan),
        "distance_pc": dist_pc, "abs_g": m_g,
        "ruwe": float(r.get("ruwe") or np.nan),
        "pmra": float(r.get("pmra") or np.nan), "pmdec": float(r.get("pmdec") or np.nan),
        "bp_rp": float(r.get("bp_rp") or np.nan), "g_mag": g,
        "phot_variable_flag": str(r.get("phot_variable_flag") or ""),
        "non_single_star": int(r.get("non_single_star") or 0),
        "match_arcsec": float(r.get("d") or np.nan) * 3600.0,
    }


def fetch_asassn(ra: float, dec: float, radius_arcsec: float = 5.0) -> dict | None:
    """Independent ASAS-SN Sky Patrol v2 light curve, if the service is reachable."""
    try:
        from pyasassn.client import SkyPatrolClient
    except Exception:
        print("[characterize] pyasassn not installed; skipping ASAS-SN")
        return None
    try:
        client = SkyPatrolClient()
        lcs = client.cone_search(ra_deg=ra, dec_deg=dec, radius=radius_arcsec / 3600.0,
                                 catalog="master_list", download=True)
        if lcs is None or not len(lcs.data):
            return None
        df = lcs.data
        df = df[(df.get("mag_err", 1) < 0.2) & (df.get("mag", 0) > 0)]
        t = df["jd"].to_numpy() - 2400000.5
        m = df["mag"].to_numpy()
        if t.size < 20:
            return {"n_epochs": int(t.size), "note": "too few ASAS-SN epochs"}
        from .secular import detect_secular_fade
        s = detect_secular_fade(t, m, df.get("mag_err"))
        return {"n_epochs": int(t.size),
                "asassn_slope_mag_yr": s.slope_mag_yr if s else None,
                "asassn_slope_sigma": s.slope_sigma if s else None,
                "asassn_total_mag": s.total_change_mag if s else None}
    except Exception as exc:
        print(f"[characterize] ASAS-SN query failed: {exc!r}")
        return None


def fetch_neowise(ra: float, dec: float, radius_arcsec: float = 3.0) -> dict | None:
    """NEOWISE-R multi-epoch W1/W2 photometry -> secular mid-IR trend.

    The decisive test for a secular optical fader: circumstellar dust that dims
    the star must reprocess the absorbed light and *brighten* in W1/W2 as the
    optical fades.  ~2 visits/yr since 2013 give a >decade mid-IR baseline.
    """
    try:
        from astroquery.ipac.irsa import Irsa
    except Exception:
        print("[characterize] astroquery IRSA not installed; skipping NEOWISE")
        return None
    try:
        q = f"""
            SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qual_frame, cc_flags
            FROM neowiser_p1bs_psd
            WHERE CONTAINS(POINT('ICRS', ra, dec),
                           CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600.0})) = 1
        """
        df = Irsa.query_tap(q).to_table().to_pandas()
    except Exception as exc:
        print(f"[characterize] NEOWISE query failed: {exc!r}")
        return None
    if df.empty:
        return None
    # Keep clean frames free of known artifact contamination.
    if "qual_frame" in df.columns:
        df = df[df["qual_frame"] > 0]
    if "cc_flags" in df.columns:
        df = df[df["cc_flags"].astype(str).str.startswith("00")]
    out: dict = {"n_epochs": int(len(df))}
    from .secular import detect_secular_fade
    for band in ("w1", "w2"):
        m = df.get(f"{band}mpro")
        e = df.get(f"{band}sigmpro")
        if m is None:
            continue
        s = detect_secular_fade(df["mjd"].to_numpy(), m.to_numpy(),
                                e.to_numpy() if e is not None else None,
                                min_epochs=30)
        out[f"{band}_slope_mag_yr"] = s.slope_mag_yr if s else None
        out[f"{band}_slope_sigma"] = s.slope_sigma if s else None
        out[f"{band}_n_seasons"] = s.n_seasons if s else 0
    return out


def ir_counterpart_verdict(optical_slope_mag_yr: float | None,
                           neowise: dict | None,
                           sig_min: float = 2.0) -> str:
    """Pure verdict on what the mid-IR does while the optical fades.

    * ``ir_brightens_dusty``      --- W1/W2 brighten as the optical fades: the
                                      absorbed light reappears as thermal dust
                                      emission; a mundane enshrouding event.
    * ``ir_fades_gray_occulter``  --- the mid-IR fades *with* the optical at a
                                      comparable rate: a gray (large-particle or
                                      solid) occulter with no thermal signature;
                                      resists every dust explanation.
    * ``ir_flat_chromatic_fade``  --- optical fades, mid-IR flat: reddening-like,
                                      small-grain dust along the line of sight.
    * ``insufficient_ir``         --- no usable NEOWISE trend.
    """
    if not neowise or optical_slope_mag_yr is None or optical_slope_mag_yr <= 0:
        return "insufficient_ir"
    slopes = [(neowise.get(f"{b}_slope_mag_yr"), neowise.get(f"{b}_slope_sigma"))
              for b in ("w1", "w2")]
    slopes = [(s, sig) for s, sig in slopes if s is not None and sig is not None]
    if not slopes:
        return "insufficient_ir"
    if any(s < 0 and sig >= sig_min for s, sig in slopes):
        return "ir_brightens_dusty"
    fading = [(s, sig) for s, sig in slopes if s > 0 and sig >= sig_min]
    if fading and any(s >= 0.3 * optical_slope_mag_yr for s, _ in fading):
        return "ir_fades_gray_occulter"
    return "ir_flat_chromatic_fade"


def characterize(ra: float, dec: float) -> dict:
    """Full independent follow-up of one candidate position."""
    gaia = fetch_gaia_dr3(ra, dec)
    asassn = fetch_asassn(ra, dec)
    neowise = fetch_neowise(ra, dec)
    out: dict = {"ra": ra, "dec": dec, "gaia_dr3": gaia, "asassn": asassn,
                 "neowise": neowise}
    # Quick interpretation flags.
    flags = []
    if gaia:
        if np.isfinite(gaia["ruwe"]) and gaia["ruwe"] > 1.4:
            flags.append("high_ruwe_possible_binary")
        if gaia["non_single_star"]:
            flags.append("gaia_non_single_star")
        if "VARIABLE" in gaia["phot_variable_flag"]:
            flags.append("gaia_flagged_variable")
    optical_slope = None
    if asassn and asassn.get("asassn_slope_sigma"):
        if asassn["asassn_slope_mag_yr"] and asassn["asassn_slope_mag_yr"] > 0 \
           and asassn["asassn_slope_sigma"] > 2:
            flags.append("asassn_confirms_fade")
            optical_slope = asassn["asassn_slope_mag_yr"]
        else:
            flags.append("asassn_does_not_confirm_fade")
    verdict = ir_counterpart_verdict(optical_slope, neowise)
    out["ir_counterpart_verdict"] = verdict
    if verdict != "insufficient_ir":
        flags.append(verdict)
    out["flags"] = flags
    return out


__all__ = ["characterize", "fetch_gaia_dr3", "fetch_asassn", "fetch_neowise",
           "ir_counterpart_verdict"]

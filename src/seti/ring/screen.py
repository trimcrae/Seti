"""The RING measurements, one function per host class.  Pure and offline.

White dwarfs
------------
The photosphere is the WD IR-excess channel's blackbody scaled to J/H/Ks (Gaia
G/BP/RP as the fallback, with a larger systematic), the excess statistics are
that channel's, and the *shape* of the excess is then fitted with OSSUARY's
matrix blackbody fit for a temperature and a fractional luminosity.  The
question this channel asks of the shape is new: is it in the Osmanov ring band
(250-800 K) rather than in the white-dwarf debris-disk locus (sublimation-
limited, 800-1800 K) or above the grain-survival ceiling (a companion)?

Pulsars
-------
A pulsar has no photosphere in the infrared, so a counterpart is an excess by
construction; the measurement is its W1-W2 colour temperature, its chance
probability from the offset-position controls, and its provenance (a nebula,
a cluster, a stellar companion, a catalogued optical counterpart).  For every
pulsar the covering fraction a 300/500/700 K ring would need to reach the
AllWISE depth is computed from the spin-down power and the distance -- the
per-host sensitivity that makes a null a number.

Brown dwarfs
------------
Per-epoch NEOWISE series: an excess-variance test on the W2 (Wien-peak) epoch
means against the quoted errors AND against the population's own median
reduced chi2 (the systematic floor), with the amplitude and a two-state duty
cycle for anything that passes.

Free-floating planets
---------------------
The catalogued bolometric luminosity against the cooling ceiling for a 13 M_J
object at the group's age (with the analytic law's margin); a flag names the
three systematics that can produce it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ossuary import excess as oex
from ..ossuary import vet as ovet
from ..photometry import BANDS, mag_err_to_flux_err_jy, mag_to_flux_jy
from . import physics as ph

# --------------------------------------------------------------------------
# White dwarfs
# --------------------------------------------------------------------------

_WISE_ORDER = ("W1", "W2", "W3", "W4")


def _ph_qual_ok(df: pd.DataFrame, band: str, allowed=("A", "B", "C")) -> np.ndarray:
    """A band is a detection only when its ph_qual letter is a detection grade.

    AllWISE gives an upper limit (``ph_qual = U``) with a NULL error; treating
    that magnitude as a measurement would manufacture excesses in W3/W4 for
    every faint white dwarf.
    """
    i = _WISE_ORDER.index(band)
    ph_ = df.get("ph_qual", pd.Series("", index=df.index)).astype(str)
    letter = ph_.str.slice(i, i + 1).str.upper()
    e = pd.to_numeric(df.get(f"e_{band}mag"), errors="coerce")
    ok = letter.isin(allowed) | (ph_.str.len() < i + 1)
    return (ok & e.notna() & (e > 0)).to_numpy(bool)


def screen_wd(df: pd.DataFrame, cfg: dict, *, rng=None) -> tuple[pd.DataFrame, dict]:
    """Photosphere, excess, ring fit, shape class and the catalogue gates."""
    from ..sed.excess import compute_excess, select_excess
    from ..sed.predict import predict_photosphere

    w = cfg["wd"]
    c = cfg["contamination"]
    if df is None or not len(df):
        return pd.DataFrame(), {"status": "NO_DATA_REACHED", "n_input": 0}

    work = df.copy().reset_index(drop=True)
    for b in _WISE_ORDER:
        if f"{b}mag" in work.columns:
            det = _ph_qual_ok(work, b, tuple(c["wise_quality"]["allowed_ph_qual"]))
            work[f"{b}_detected"] = det
            # Non-detections carry an upper limit in the magnitude column: blank
            # it so no stage downstream can mistake it for a measurement.
            work.loc[~det, f"{b}mag"] = np.nan
            work.loc[~det, f"e_{b}mag"] = np.nan
    work = predict_photosphere(work, anchor_bands=("J", "H", "Ks"),
                               predict_bands=tuple(b for b in _WISE_ORDER
                                                   if f"{b}mag" in work.columns),
                               fallback_bands=("G", "BP", "RP"))
    n_anchor = {k: int(v) for k, v in work["sed_anchor"].value_counts().items()}

    # Excess statistics with an anchor-dependent systematic floor.
    floor = np.where(work["sed_anchor"] == "nir", float(w["sys_floor_mag"]),
                     float(w["sys_floor_mag_gaia_anchor"]))
    parts = []
    for anchor, sf in (("nir", float(w["sys_floor_mag"])),
                       ("gaia", float(w["sys_floor_mag_gaia_anchor"]))):
        sub = work[work["sed_anchor"] == anchor]
        if not len(sub):
            continue
        th = {"excess": {"sys_floor_mag": sf, "chi_w1_min": float(w["chi_w1_min"]),
                         "chi_w2_min": float(w["chi_w2_min"]),
                         "color_excess_sigma_min": float(w["color_excess_sigma_min"])}}
        ex = compute_excess(sub, th, bands=("W1", "W2"))
        ex["excess_flag"] = select_excess(ex, th)
        parts.append(ex)
    none = work[work["sed_anchor"] == "none"].copy()
    if len(none):
        for col in ("chi_W1", "chi_W2", "chi_color", "W1_excess_jy", "W2_excess_jy"):
            none[col] = np.nan
        none["excess_flag"] = False
        parts.append(none)
    work = pd.concat(parts).sort_index()
    work["sys_floor_mag_used"] = floor[work.index] if len(work) else floor

    # Excess errors per band for the ring fit (the WD channel keeps only chi).
    for b in _WISE_ORDER:
        if f"{b}_pred_jy" not in work.columns or f"{b}mag" not in work.columns:
            continue
        obs = pd.to_numeric(work[f"{b}mag"], errors="coerce").to_numpy(float)
        oerr = pd.to_numeric(work.get(f"e_{b}mag"), errors="coerce").to_numpy(float)
        pred = pd.to_numeric(work[f"{b}_pred_jy"], errors="coerce").to_numpy(float)
        with np.errstate(invalid="ignore"):
            obs_jy = mag_to_flux_jy(obs, b)
            obs_err = mag_err_to_flux_err_jy(obs, np.where(np.isfinite(oerr), oerr, 0.1), b)
        sys_jy = 0.4 * np.log(10.0) * work["sys_floor_mag_used"].to_numpy(float) * np.abs(pred)
        exc_ = obs_jy - pred
        err = np.sqrt(obs_err ** 2 + sys_jy ** 2)
        work[f"{b}_obs_jy"] = obs_jy
        work[f"{b}_excess_jy"] = exc_
        work[f"{b}_excess_err_jy"] = err
        with np.errstate(divide="ignore", invalid="ignore"):
            work[f"chi_{b}"] = exc_ / err
    work["w1_w2_obs"] = pd.to_numeric(work.get("W1mag"), errors="coerce") - \
        pd.to_numeric(work.get("W2mag"), errors="coerce")

    # The ring fit on the flagged rows (matrix fit; Monte-Carlo percentiles).
    flagged = work[work["excess_flag"].fillna(False).astype(bool)].copy()
    for col in ("t_ring_k", "t_ring_lo_k", "t_ring_hi_k", "tau_ring", "tau_lo", "tau_hi",
                "ring_fit_chi2", "n_excess_bands", "shape_class", "ring_radius_au",
                "l_wd_w"):
        work[col] = np.nan if col != "shape_class" else "not_flagged"
    if len(flagged):
        fit = oex.characterise(flagged, {"mc_draws": int(w["mc_draws"])}, anchor="Ks",
                               bands=_WISE_ORDER, rng=rng)
        # tau from the SED's own solid angle (the photosphere fit), not the anchor mag.
        teff = pd.to_numeric(fit["teff"], errors="coerce").to_numpy(float)
        scale = pd.to_numeric(fit["sed_scale"], errors="coerce").to_numpy(float)
        # characterise's tau used the Ks anchor; recompute from the omega ratio.
        # The fitted omega is in "flux / B_nu" units (no pi) while the SED
        # scale from predict_photosphere multiplies pi B_nu, hence the 1/pi.
        om_ring = _omega_from_fit(fit)
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = om_ring * fit["t_dust_k"].to_numpy(float) ** 4 / \
                (np.pi * scale * teff ** 4)
            tau_lo = tau * (fit["tau_lo"] / fit["tau"]).to_numpy(float)
            tau_hi = tau * (fit["tau_hi"] / fit["tau"]).to_numpy(float)
        lwd = ph.wd_luminosity_w(teff, pd.to_numeric(fit["logg"], errors="coerce"),
                                 pd.to_numeric(fit["mass"], errors="coerce").fillna(0.6))
        work.loc[fit.index, "t_ring_k"] = fit["t_dust_k"]
        work.loc[fit.index, "t_ring_lo_k"] = fit["t_dust_lo_k"]
        work.loc[fit.index, "t_ring_hi_k"] = fit["t_dust_hi_k"]
        work.loc[fit.index, "tau_ring"] = tau
        work.loc[fit.index, "tau_lo"] = tau_lo
        work.loc[fit.index, "tau_hi"] = tau_hi
        work.loc[fit.index, "ring_fit_chi2"] = fit["dust_fit_chi2"]
        work.loc[fit.index, "n_excess_bands"] = fit["n_excess_bands"]
        work.loc[fit.index, "l_wd_w"] = lwd
        work.loc[fit.index, "ring_radius_au"] = ph.ring_equilibrium_radius_au(
            lwd, fit["t_dust_k"].to_numpy(float))
        work.loc[fit.index, "shape_class"] = [
            ph.classify_excess_shape(t, ta, cfg)
            for t, ta in zip(fit["t_dust_k"].to_numpy(float), tau, strict=False)]

    # Catalogue gates, inherited from OSSUARY and driven by this channel's config.
    gates = _wd_gates(work, cfg)
    for col in gates.columns:
        work[col] = gates[col]
    flagged_mask = work["excess_flag"].fillna(False).astype(bool)
    work["verdict"] = np.where(~flagged_mask, "no_excess",
                               np.where(work["gate_reason"] == "", "surviving", "rejected"))
    work["ring_candidate"] = flagged_mask & (work["verdict"] == "surviving") & \
        (work["shape_class"] == "ring_band")

    # Per-host sensitivity: the covering fraction a 500 K ring needs at the W2 depth.
    teff_all = pd.to_numeric(work["teff"], errors="coerce").to_numpy(float)
    lwd_all = ph.wd_luminosity_w(teff_all, pd.to_numeric(work.get("logg"), errors="coerce"),
                                 pd.to_numeric(work.get("mass"), errors="coerce").fillna(0.6))
    work["l_wd_w"] = np.where(np.isfinite(work["l_wd_w"]), work["l_wd_w"], lwd_all)
    for t in cfg["ring"]["t_nominal_k"]:
        work[f"f_min_{int(t)}K_W2"] = ph.min_intercept_fraction(
            lwd_all, t, work["dist_pc"].to_numpy(float), "W2",
            float(cfg["survey_limits_jy"]["W2"]))
    f500 = work["f_min_500K_W2"].to_numpy(float)
    shape_counts = {k: int(v) for k, v in work.loc[flagged_mask, "shape_class"]
                    .value_counts().items()}
    summary = {
        "status": "OK",
        "n_input": int(len(df)),
        "sed_anchor_counts": n_anchor,
        "n_w1_detected": int(work.get("W1_detected", pd.Series(False)).sum()),
        "n_w2_detected": int(work.get("W2_detected", pd.Series(False)).sum()),
        "n_excess_flagged": int(flagged_mask.sum()),
        "shape_counts": shape_counts,
        "n_surviving_gates": int((flagged_mask & (work["verdict"] == "surviving")).sum()),
        "n_ring_candidates": int(work["ring_candidate"].sum()),
        "gate_reasons": {k: int(v) for k, v in
                         work.loc[flagged_mask, "gate_reason"].value_counts().items() if k},
        "sensitivity": {
            "n_hosts_with_500K_ring_detectable_at_f_lt_1":
                int(np.isfinite(f500).sum() and (f500 < 1.0).sum()),
            "median_f_min_500K": float(np.nanmedian(f500)) if np.isfinite(f500).any()
            else None,
            "n_hosts_with_500K_ring_detectable_at_f_lt_0p01": int((f500 < 0.01).sum()),
        },
    }
    return work, summary


def _omega_from_fit(fit: pd.DataFrame) -> np.ndarray:
    """Recover the fitted ring solid angle from characterise's outputs.

    ``characterise`` reports tau = omega_d T_d^4 / (omega_* T_eff^4) with
    omega_* from the Ks anchor; multiplying back gives omega_d.  Where the
    anchor was absent tau is NaN, so refit omega from the W-band excesses.
    """
    a_mag = pd.to_numeric(fit.get("Ksmag"), errors="coerce").to_numpy(float)
    teff = pd.to_numeric(fit.get("teff"), errors="coerce").to_numpy(float)
    om_star = np.array([oex._star_solid_angle(a, "Ks", t) for a, t in
                        zip(a_mag, teff, strict=False)])
    with np.errstate(divide="ignore", invalid="ignore"):
        om = fit["tau"].to_numpy(float) * om_star * teff ** 4 / \
            fit["t_dust_k"].to_numpy(float) ** 4
    need = ~np.isfinite(om)
    if need.any():
        for i in np.nonzero(need)[0]:
            f = {b: float(fit[f"{b}_excess_jy"].iloc[i]) for b in _WISE_ORDER
                 if f"{b}_excess_jy" in fit.columns}
            e = {b: float(fit[f"{b}_excess_err_jy"].iloc[i]) for b in _WISE_ORDER
                 if f"{b}_excess_err_jy" in fit.columns}
            use = {b: v for b, v in f.items()
                   if np.isfinite(v) and v > 0 and e.get(b, 0) > 0 and v / e[b] >= 1.0}
            if len(use) >= 2:
                _, omk, _ = oex.fit_excess_blackbody(use, {b: e[b] for b in use})
                om[i] = omk
    return om


def _wd_gates(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The inherited ledger for a white-dwarf host, in gate order."""
    w, c = cfg["wd"], cfg["contamination"]
    out = pd.DataFrame(index=work.index)
    wq = ovet.wise_quality_gate(work, c, bands=("W1", "W2"))
    out["wise_quality_ok"] = wq["wise_quality_ok"]
    excess_cfg = {"chi_min": float(w["chi_w1_min"]), "require_bands": ["W1", "W2"],
                  "w1_w2_min": float(w["w1_w2_min"]), "w1_w2_agn_max": float(w["w1_w2_agn_max"]),
                  "t_dust_max_k": float(cfg["companion_t_min_k"]),
                  "t_dust_min_k": float(cfg["ring"]["t_min_k"]), "nir_excess_sigma_max": 3.0}
    sample_cfg = {"g_max": float(w["g_max"]), "bp_rp_min": float(w["bp_rp_min"]),
                  "bp_rp_max": float(w["bp_rp_max"])}
    lg = ovet.ledger_gate(work, excess_cfg, sample_cfg)
    out["ledger_ok"] = lg["ledger_ok"]
    out["w4_only"] = lg["w4_only"]
    tmp = work.copy()
    tmp["t_dust_k"] = work["t_ring_k"]
    cg = ovet.companion_gate(tmp, excess_cfg)
    out["companion_ok"] = cg["companion_ok"]
    ag = ovet.astrometry_gate(work, c["astrometry"])
    out["registration_arcsec"] = ag["registration_arcsec"]
    out["registration_tested"] = ag["registration_tested"]
    out["astrometry_ok"] = ag["astrometry_ok"]
    eg = ovet.extragalactic_gate(work, c)
    out["chance_superposition_p"] = eg["chance_superposition_p"]
    out["extragalactic_ok"] = eg["extragalactic_ok"]
    gc = ovet.globular_cluster_veto(work, c)
    out["globular_ok"] = gc["globular_ok"]
    order = [("wise_quality_ok", "wise_quality"), ("ledger_ok", "ledger"),
             ("companion_ok", "unresolved_companion"),
             ("astrometry_ok", "astrometric_registration"),
             ("extragalactic_ok", "background_source"),
             ("globular_ok", "globular_cluster_sightline")]
    reason = pd.Series("", index=work.index, dtype=object)
    for col, name in order:
        failed = ~out[col].fillna(False).astype(bool) & (reason == "")
        reason[failed] = name
    out["gate_reason"] = reason
    return out


# --------------------------------------------------------------------------
# Pulsars
# --------------------------------------------------------------------------

def _split_id(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = s.astype(str).str.rsplit("|", n=1, expand=True)
    return parts[0], parts[1] if parts.shape[1] > 1 else pd.Series("t", index=s.index)


def screen_pulsars(psr: pd.DataFrame, matches: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Per pulsar: the counterpart (if any), its colour temperature, its chance
    probability from the controls, its provenance vetoes, and the sensitivity."""
    p = cfg["pulsar"]
    if psr is None or not len(psr):
        return pd.DataFrame(), {"status": "NO_DATA_REACHED", "n_hosts": 0}
    out = psr.copy().reset_index(drop=True)
    r_floor, r_max = float(p["match_radius_arcsec"]), float(p["match_radius_max_arcsec"])
    fac = float(p["position_error_factor"])
    out["match_radius_arcsec"] = np.clip(
        np.maximum(r_floor, fac * pd.to_numeric(out["pos_err_arcsec"], errors="coerce")
                   .fillna(r_max).to_numpy(float)), r_floor, r_max)
    n_ctrl = 8 * len(p["control_offsets_arcsec"])

    t_lo, t_hi = float(cfg["ring"]["t_min_k"]), float(cfg["ring"]["t_max_k"])
    for key in ("allwise", "catwise"):
        m = matches.get(key)
        out[f"{key}_match"] = False
        out[f"{key}_dist_arcsec"] = np.nan
        out[f"{key}_n_control_hits"] = 0
        out[f"{key}_n_control_ring_hits"] = 0
        out[f"{key}_p_chance"] = np.nan
        out[f"{key}_p_chance_ring"] = np.nan
        if m is None or not len(m) or "source_id" not in m.columns:
            continue
        jn, kind = _split_id(m["source_id"])
        m = m.assign(_jname=jn, _kind=kind)
        d = pd.to_numeric(m.get("match_dist_arcsec"), errors="coerce")
        m = m.assign(_d=d)
        rad = out.set_index("jname")["match_radius_arcsec"]
        m = m[m["_jname"].isin(rad.index)]
        m = m[m["_d"] <= m["_jname"].map(rad).to_numpy(float)]
        w1c = "W1mag" if key == "allwise" else "W1mag_cat"
        w2c = "W2mag" if key == "allwise" else "W2mag_cat"
        mcol = pd.to_numeric(m.get(w1c), errors="coerce") - pd.to_numeric(m.get(w2c),
                                                                          errors="coerce")
        mt = ph.colour_to_temperature(mcol.to_numpy(float))
        m = m.assign(_ring=(mt >= t_lo) & (mt <= t_hi))
        tgt = m[m["_kind"] == "t"].sort_values("_d").drop_duplicates("_jname")
        ctrl = m[m["_kind"] != "t"].drop_duplicates(["_jname", "_kind"])
        hits = ctrl.groupby("_jname").size()
        ring_hits = ctrl[ctrl["_ring"]].groupby("_jname").size()
        h = out["jname"].map(hits).fillna(0).astype(int)
        hr = out["jname"].map(ring_hits).fillna(0).astype(int)
        out[f"{key}_n_control_hits"] = h
        out[f"{key}_n_control_ring_hits"] = hr
        # Empirical-Bayes chance probability: the local control hits pooled with
        # the global control rate, prior weight = the number of local controls.
        # Sixteen local positions alone cannot measure a rate of 1 %; the whole
        # control set (16 x N_pulsars positions on the same sky) can.
        n_pos = float(n_ctrl * len(out))
        p_g_any = float(h.sum()) / n_pos if n_pos else 0.0
        p_g_ring = float(hr.sum()) / n_pos if n_pos else 0.0
        out[f"{key}_p_chance"] = (h + n_ctrl * p_g_any) / (2.0 * n_ctrl)
        out[f"{key}_p_chance_ring"] = (hr + n_ctrl * p_g_ring) / (2.0 * n_ctrl)
        out.attrs[f"{key}_global_control_rate"] = p_g_any
        out.attrs[f"{key}_global_control_ring_rate"] = p_g_ring
        t = tgt.set_index("_jname")
        has = out["jname"].isin(t.index)
        out[f"{key}_match"] = has
        out.loc[has, f"{key}_dist_arcsec"] = out.loc[has, "jname"].map(t["_d"]).to_numpy()
        cols = {"allwise": ["designation", "W1mag", "e_W1mag", "W2mag", "e_W2mag", "W3mag",
                            "e_W3mag", "W4mag", "e_W4mag", "ph_qual", "cc_flags", "ext_flag",
                            "number_of_neighbours", "number_of_mates", "ra_wise", "dec_wise"],
                "catwise": ["catwise_name", "W1mag_cat", "e_W1mag_cat", "W2mag_cat",
                            "e_W2mag_cat", "pmra_catwise", "pmdec_catwise", "e_pmra_catwise",
                            "e_pmdec_catwise", "ph_qual_cat", "cc_flags_cat"]}[key]
        for col in cols:
            if col in t.columns:
                out[col] = out["jname"].map(t[col])

    # Colour temperature of the counterpart (AllWISE first, CatWISE as fallback).
    w1 = pd.to_numeric(out.get("W1mag"), errors="coerce")
    w2 = pd.to_numeric(out.get("W2mag"), errors="coerce")
    w1c = pd.to_numeric(out.get("W1mag_cat"), errors="coerce")
    w2c = pd.to_numeric(out.get("W2mag_cat"), errors="coerce")
    colour = (w1 - w2).where((w1 - w2).notna(), w1c - w2c)
    out["w1_w2"] = colour
    out["t_colour_k"] = ph.colour_to_temperature(colour.to_numpy(float))
    out["shape_class"] = [ph.classify_excess_shape(t, np.nan, cfg) if np.isfinite(t)
                          else ("no_counterpart" if not (a or b) else "unfit")
                          for t, a, b in zip(out["t_colour_k"], out["allwise_match"],
                                             out["catwise_match"], strict=False)]

    # Provenance vetoes: nebulae, clusters, companions, catalogued counterparts.
    assoc = out.get("assoc", pd.Series("", index=out.index)).astype(str).str.upper()
    binc = out.get("bincomp", pd.Series("", index=out.index)).astype(str).str.upper()
    out["assoc_veto"] = assoc.apply(lambda s: any(tok.upper() in s
                                                   for tok in p["assoc_veto_tokens"]))
    out["companion_veto"] = binc.apply(lambda s: any(tok.upper() == s.strip() or
                                                      s.strip().startswith(tok.upper())
                                                      for tok in p["bincomp_veto_tokens"]))
    out["known_counterpart_veto"] = out["jname"].isin(set(p["known_counterpart_veto"]))
    out["globular_cluster"] = assoc.str.contains("GC:")
    # The chance probability that matters is the one for the hypothesis being
    # tested: for a ring-band counterpart, a random source with a ring-band
    # colour; for anything else, any random source.
    p_any = out[["allwise_p_chance", "catwise_p_chance"]].min(axis=1)
    p_ring = out[["allwise_p_chance_ring", "catwise_p_chance_ring"]].min(axis=1)
    out["p_chance_any"] = p_any
    out["p_chance_ring"] = p_ring
    out["p_chance"] = np.where(out["shape_class"] == "ring_band", p_ring, p_any)
    out["chance_ok"] = pd.Series(out["p_chance"], index=out.index) <= float(p["chance_p_max"])

    # Spin-down power, distance, and the per-host ring sensitivity.
    edot = ph.spin_down_luminosity_w(out["p0_s"].to_numpy(float), out["p1"].to_numpy(float))
    out["edot_w"] = edot
    d_pc = pd.to_numeric(out["dist_kpc"], errors="coerce").to_numpy(float) * 1000.0
    for t in cfg["ring"]["t_nominal_k"]:
        out[f"f_min_{int(t)}K_W2"] = ph.min_intercept_fraction(
            edot, t, d_pc, "W2", float(cfg["survey_limits_jy"]["W2"]))
        out[f"ring_radius_{int(t)}K_au"] = ph.ring_equilibrium_radius_au(edot, t)

    has = out["allwise_match"] | out["catwise_match"]
    reason = pd.Series("", index=out.index, dtype=object)
    for col, name in (("known_counterpart_veto", "catalogued_counterpart"),
                      ("assoc_veto", "nebula_cluster_or_optical_association"),
                      ("companion_veto", "non_degenerate_companion"),
                      ("globular_cluster", "globular_cluster_sightline")):
        f = out[col].astype(bool) & (reason == "")
        reason[f] = name
    f = ~out["chance_ok"].fillna(False) & (reason == "")
    reason[f] = "chance_coincidence"
    f = (out["shape_class"] == "companion") & (reason == "")
    reason[f] = "companion_colour"
    f = (out["shape_class"].isin(["unfit"])) & (reason == "")
    reason[f] = "single_band_no_colour"
    out["veto_reason"] = np.where(has, reason, "")
    out["verdict"] = np.where(~has, "no_counterpart",
                              np.where(out["veto_reason"] == "", "surviving", "rejected"))
    out["ring_candidate"] = (out["verdict"] == "surviving") & \
        (out["shape_class"] == "ring_band")
    f500 = out["f_min_500K_W2"].to_numpy(float)
    summary = {
        "status": "OK",
        "n_hosts": int(len(out)),
        "n_with_allwise_counterpart": int(out["allwise_match"].sum()),
        "n_with_catwise_counterpart": int(out["catwise_match"].sum()),
        "n_with_any_counterpart": int(has.sum()),
        "control_positions_per_host": n_ctrl,
        "median_allwise_control_hits": float(out["allwise_n_control_hits"].median()),
        "global_control_rate": {k: float(v) for k, v in out.attrs.items()
                                if k.endswith("control_rate") or k.endswith("ring_rate")},
        # The census: counterparts observed against the number the controls predict.
        "n_counterparts_expected_by_chance": float(np.nansum(p_any.to_numpy(float))),
        "n_ring_band_expected_by_chance": float(np.nansum(p_ring.to_numpy(float))),
        "shape_counts": {k: int(v) for k, v in out.loc[has, "shape_class"]
                         .value_counts().items()},
        "veto_reasons": {k: int(v) for k, v in out.loc[has, "veto_reason"]
                         .value_counts().items() if k},
        "n_surviving": int((out["verdict"] == "surviving").sum()),
        "n_ring_candidates": int(out["ring_candidate"].sum()),
        "sensitivity": {
            "n_hosts_with_edot_and_distance": int(np.isfinite(f500).sum()),
            "n_hosts_500K_ring_detectable_at_f_lt_1": int((f500 < 1.0).sum()),
            "n_hosts_500K_ring_detectable_at_f_lt_0p1": int((f500 < 0.1).sum()),
            "median_f_min_500K": float(np.nanmedian(f500)) if np.isfinite(f500).any()
            else None,
        },
    }
    return out, summary


# --------------------------------------------------------------------------
# Brown dwarfs: duty cycle in the W2 epoch series
# --------------------------------------------------------------------------

def series_stats(t, mag, err) -> dict:
    """Excess-variance statistics of one epoch series."""
    t = np.asarray(t, float)
    m = np.asarray(mag, float)
    e = np.asarray(err, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    n = int(m.size)
    if n < 2:
        return {"n_epochs": n, "chi2_red": np.nan, "amp_mag": np.nan, "mean_mag": np.nan,
                "duty_cycle_high": np.nan, "two_state_sep_mag": np.nan}
    w = 1.0 / e ** 2
    mean = float((w * m).sum() / w.sum())
    chi2 = float((((m - mean) / e) ** 2).sum() / (n - 1))
    amp = float(m.max() - m.min())
    # Two-state split: the threshold that minimises within-state variance.
    order = np.sort(m)
    best, sep, frac = np.inf, np.nan, np.nan
    for k in range(1, n):
        lo, hi = order[:k], order[k:]
        v = lo.var() * k + hi.var() * (n - k)
        if v < best:
            best, sep, frac = v, float(hi.mean() - lo.mean()), float(k / n)
    return {"n_epochs": n, "chi2_red": chi2, "amp_mag": amp, "mean_mag": mean,
            "duty_cycle_high": frac, "two_state_sep_mag": sep}


def screen_bd(epochs: pd.DataFrame, targets: pd.DataFrame, cfg: dict
              ) -> tuple[pd.DataFrame, dict]:
    """Per object: W1 and W2 series statistics; a duty-cycle flag against both
    the quoted errors and the population's own scatter floor."""
    b = cfg["bd"]
    if targets is None or not len(targets):
        return pd.DataFrame(), {"status": "NO_DATA_REACHED", "n_targets": 0}
    rows = []
    ep = epochs if epochs is not None else pd.DataFrame()
    if len(ep):
        ep = ep.copy()
        ep["source_id"] = ep["source_id"].astype(str)
    for _, tgt in targets.iterrows():
        sid = str(tgt["source_id"])
        rec = {"source_id": sid, "spt": tgt.get("spt", ""), "spt_num": tgt.get("spt_num", np.nan)}
        g = ep[ep["source_id"] == sid] if len(ep) else ep
        for band in ("W1", "W2"):
            gb = g[g["band"] == band] if len(g) else g
            st = series_stats(gb["t_yr"] if len(gb) else [], gb["mag"] if len(gb) else [],
                              gb["err"] if len(gb) else [])
            rec.update({f"{band.lower()}_{k}": v for k, v in st.items()})
        rows.append(rec)
    out = pd.DataFrame(rows)
    n_min = int(b["min_epochs"])
    have = out["w2_n_epochs"] >= n_min
    pop_floor = float(np.nanmedian(out.loc[have, "w2_chi2_red"])) if have.any() else np.nan
    thr = max(float(b["chi2_red_min"]),
              float(b["chi2_red_pop_factor"]) * pop_floor if np.isfinite(pop_floor) else 0.0)
    out["w2_chi2_threshold"] = thr
    out["duty_cycle_flag"] = have & (out["w2_chi2_red"] >= thr) & \
        (out["w2_amp_mag"] >= float(b["amp_min_mag"]))
    out["status"] = np.where(out["w2_n_epochs"] == 0, "NO_EPOCHS",
                             np.where(~have, "TOO_FEW_EPOCHS", "TESTED"))
    summary = {
        "status": "OK" if have.any() else ("NO_DATA_REACHED" if not len(ep) else
                                           "INSUFFICIENT_EPOCHS"),
        "n_targets": int(len(out)),
        "n_with_epochs": int((out["w2_n_epochs"] > 0).sum()),
        "n_tested": int(have.sum()),
        "population_w2_chi2_median": pop_floor if np.isfinite(pop_floor) else None,
        "w2_chi2_threshold": thr,
        "n_duty_cycle_flags": int(out["duty_cycle_flag"].sum()),
        "flagged": out.loc[out["duty_cycle_flag"], ["source_id", "spt", "w2_n_epochs",
                                                     "w2_chi2_red", "w2_amp_mag",
                                                     "w2_duty_cycle_high",
                                                     "w1_chi2_red"]].to_dict("records"),
    }
    return out, summary


# --------------------------------------------------------------------------
# Free-floating planets: hotter than cooling allows
# --------------------------------------------------------------------------

def _group_age_gyr(group, cfg: dict) -> float:
    s = str(group or "").strip().lower()
    if not s or s in ("nan", "none", "field", "--"):
        return np.nan
    for k, v in cfg["ffp"]["age_fallback_gyr"].items():
        kl = k.lower()
        if kl in s or s in kl:
            return float(v)
    return np.nan


def screen_ffp(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    f = cfg["ffp"]
    if df is None or not len(df):
        return pd.DataFrame(), {"status": "NO_DATA_REACHED", "n_objects": 0}
    out = df.copy().reset_index(drop=True)
    lbol = pd.to_numeric(out.get("lbol"), errors="coerce").to_numpy(float)
    # Accept either log10(L/Lsun) or L/Lsun; a value above zero is not a log.
    with np.errstate(invalid="ignore", divide="ignore"):
        lbol = np.where(lbol > 0, np.log10(np.where(lbol > 0, lbol, 1.0)), lbol)
    out["log_lbol"] = lbol
    age = pd.to_numeric(out.get("age"), errors="coerce").to_numpy(float)
    # Ages catalogued in Myr are the norm for young groups.
    age_gyr = np.where(age > 5.0, age / 1000.0, age)
    fallback = np.array([_group_age_gyr(g, cfg) for g in
                         out.get("group", pd.Series("", index=out.index))])
    out["age_gyr"] = np.where(np.isfinite(age_gyr), age_gyr, fallback)
    out["age_source"] = np.where(np.isfinite(age_gyr), "catalogue",
                                 np.where(np.isfinite(fallback), "group_fallback", "none"))
    mass = pd.to_numeric(out.get("mass"), errors="coerce").to_numpy(float)
    out["mass_mj"] = np.where(mass < 1.0, mass / ph.M_JUP_MSUN, mass)   # M_sun -> M_J
    ceiling = ph.planetary_cooling_ceiling_log_lsun(out["age_gyr"].to_numpy(float),
                                                    margin_dex=float(f["margin_dex"]),
                                                    mass_limit_mj=float(f["mass_limit_mj"]))
    out["log_l_ceiling_13mj"] = ceiling
    out["excess_over_ceiling_dex"] = out["log_lbol"] - ceiling
    planetary = out["mass_mj"] <= float(f["mass_limit_mj"])
    testable = np.isfinite(out["log_lbol"]) & np.isfinite(ceiling)
    out["testable"] = testable
    out["catalogued_planetary_mass"] = planetary
    out["hotter_than_cooling"] = testable & (out["excess_over_ceiling_dex"] > 0)
    out["flag"] = out["hotter_than_cooling"] & planetary
    out["systematics_not_excluded"] = np.where(
        out["flag"], "age_misassignment|mass_underestimate|unresolved_binary", "")
    summary = {
        "status": "OK" if testable.any() else "NO_TESTABLE_OBJECTS",
        "n_objects": int(len(out)),
        "n_testable": int(testable.sum()),
        "n_catalogued_planetary_mass": int((planetary & testable).sum()),
        "n_hotter_than_ceiling": int(out["hotter_than_cooling"].sum()),
        "n_flags_planetary_and_hot": int(out["flag"].sum()),
        "flagged": out.loc[out["flag"], ["name", "spt", "group", "age_gyr", "age_source",
                                         "log_lbol", "log_l_ceiling_13mj",
                                         "excess_over_ceiling_dex", "mass_mj"]]
        .to_dict("records") if "name" in out.columns else [],
    }
    return out, summary


__all__ = ["screen_wd", "screen_pulsars", "screen_bd", "screen_ffp", "series_stats",
           "BANDS"]

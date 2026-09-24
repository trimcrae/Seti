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


def text_column(df: pd.DataFrame, col: str) -> pd.Series:
    """A plain ``object``-dtype text column with every missing value as ``""``.

    Catalogue text arrives in three different shapes depending on the pandas
    build and on whether the column survived a round trip: an ``object``
    column of Python strings, an all-float column of ``NaN`` when the field was
    absent, or -- once ``future.infer_string`` is on, which is the default in
    pandas 3 and therefore what a fresh runner installs -- an Arrow-backed
    ``str`` column whose missing entries stay ``NA`` through ``astype(str)``.

    Reading such a column with ``.astype(str)`` gives ``"nan"`` in the second
    case (a token test against it is silently meaningless) and a bare float in
    the third (``in`` then raises ``TypeError``).  Both are avoided by mapping
    element by element and sending every missing value to the empty string, so
    a pulsar with no ``assoc`` entry is simply a pulsar with no association
    rather than a crash or a phantom token.
    """
    s = df[col] if col in df.columns else pd.Series("", index=df.index)
    s = pd.Series(s, index=df.index)
    return s.map(lambda v: "" if v is None or v is pd.NA or
                 (isinstance(v, float) and not np.isfinite(v)) else str(v)).astype(object)


def _numcol(df: pd.DataFrame, col: str) -> pd.Series:
    """A float Series for ``col``, all-NaN (same index) when the column is absent.

    ``pd.to_numeric(df.get(col))`` returns a bare scalar for an absent column,
    which then breaks ``.notna()`` / ``.to_numpy()`` -- the free-floating leg
    would have crashed on Faherty+2016, which has no age column.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _ph_qual_ok(df: pd.DataFrame, band: str, allowed=("A", "B", "C")) -> np.ndarray:
    """A band is a detection only when its ph_qual letter is a detection grade.

    AllWISE gives an upper limit (``ph_qual = U``) with a NULL error; treating
    that magnitude as a measurement would manufacture excesses in W3/W4 for
    every faint white dwarf.
    """
    i = _WISE_ORDER.index(band)
    ph_ = text_column(df, "ph_qual")
    letter = ph_.str.slice(i, i + 1).str.upper()
    e = pd.to_numeric(df.get(f"e_{band}mag"), errors="coerce")
    ok = letter.isin(allowed) | (ph_.str.len() < i + 1)
    return (ok & e.notna() & (e > 0)).to_numpy(bool)


def _achromatic_route(ex: pd.DataFrame, th: dict, w: dict) -> pd.Series:
    """Admit a bright *achromatic* excess that the colour test would drop.

    The colour requirement in ``select_excess`` guards against an error in the
    SED anchor: mis-measure Ks and the whole predicted photosphere scales, so
    W1 and W2 rise together with no colour signal.  A cool companion in the
    beam does the same thing -- at 2500-3000 K the excess colour is small even
    when the excess flux is several times the photosphere -- so the companion
    population would never be flagged, never fitted, and never *named* in the
    census.  The discriminant is amplitude: an anchor error of 0.02-0.1 mag
    cannot manufacture an excess fraction of 0.5 (0.44 mag) in both bands.

    This route can only add contaminants, never ring candidates: a 250-800 K
    ring has W1-W2 > 1.3 mag by construction and therefore always enters by
    the colour route, and ``screen_wd`` requires ``excess_route == 'colour'``
    before a row may become a ring candidate.
    """
    frac_min = float(w.get("achromatic_excess_frac_min", np.nan))
    idx = ex.index
    if not np.isfinite(frac_min):
        return pd.Series(False, index=idx)
    e = th["excess"]
    ok = pd.Series(True, index=idx)
    for b in ("W1", "W2"):
        chi = pd.to_numeric(ex.get(f"chi_{b}"), errors="coerce")
        exc = pd.to_numeric(ex.get(f"{b}_excess_jy"), errors="coerce")
        pred = pd.to_numeric(ex.get(f"{b}_pred_jy"), errors="coerce")
        frac = (exc / pred).where(pred > 0)
        ok = ok & (chi >= float(e[f"chi_{b.lower()}_min"])) & (frac >= frac_min)
    return ok.fillna(False).astype(bool)


def wd_chance_census(work: pd.DataFrame, controls: pd.DataFrame | None, cfg: dict
                     ) -> tuple[pd.DataFrame, dict]:
    """Observed white-dwarf excesses against the offset-position controls.

    A control source matters only if, dropped into the host's aperture, it
    could produce the flagged signal: detected in W1 and W2 (ph_qual A/B/C),
    with a ring-band W1-W2 colour for the ring hypothesis, and bright enough
    in W2 to rival the photosphere (W2 <= predicted photospheric W2 +
    ``control_w2_margin_mag``).  Per host the local rate is hits / n_controls;
    the census sums it.
    """
    w = cfg["wd"]
    n_ctrl = int(w.get("control_angles", 8)) * len(w.get("control_offsets_arcsec", [45.0]))
    out = pd.DataFrame(index=work.index)
    out["ctrl_hits_any"] = 0
    out["ctrl_hits_ring"] = 0
    if controls is None or not len(controls) or "source_id" not in controls.columns:
        return out, {"status": "NO_CONTROLS"}
    t_lo, t_hi = float(cfg["ring"]["t_min_k"]), float(cfg["ring"]["t_max_k"])
    c = controls.copy()
    parts = c["source_id"].astype(str).str.rsplit("|", n=1, expand=True)
    c["_host"], c["_pos"] = parts[0], parts[1]
    c = c.sort_values("match_dist_arcsec").drop_duplicates(["_host", "_pos"]) \
        if "match_dist_arcsec" in c.columns else c.drop_duplicates(["_host", "_pos"])
    ok = _allwise_colour_ok(c)
    col = (_numcol(c, "W1mag") - _numcol(c, "W2mag")).where(ok)
    tt = ph.colour_to_temperature(col.to_numpy(float))
    c["_ring"] = (tt >= t_lo) & (tt <= t_hi)
    host = work["source_id"].astype(str)
    pred_w2 = pd.Series(np.nan, index=work.index)
    if "W2_pred_jy" in work.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            pred_w2 = -2.5 * np.log10(_numcol(work, "W2_pred_jy")
                                      / float(BANDS["W2"]["zp_jy"]))
    lim = pd.Series((pred_w2 + float(w.get("control_w2_margin_mag", 1.0))).to_numpy(),
                    index=host.to_numpy())
    lim = lim[~lim.index.duplicated()]
    c["_bright"] = (_numcol(c, "W2mag") <= c["_host"].map(lim)).fillna(False)
    c["_q"] = ok & c["_bright"]
    any_h = c[c["_q"]].groupby("_host").size()
    ring_h = c[c["_q"] & c["_ring"]].groupby("_host").size()
    out["ctrl_hits_any"] = host.map(any_h).fillna(0).astype(int).to_numpy()
    out["ctrl_hits_ring"] = host.map(ring_h).fillna(0).astype(int).to_numpy()
    tot = float(len(work)) * n_ctrl
    p_ring_g = float(out["ctrl_hits_ring"].sum()) / tot if tot else 0.0
    out["p_chance_ring_ctrl"] = (out["ctrl_hits_ring"] + n_ctrl * p_ring_g) / (2.0 * n_ctrl)
    flagged = work["excess_flag"].fillna(False).astype(bool)
    ring = flagged & (work["shape_class"] == "ring_band")
    census = {
        "status": "OK",
        "control_positions_per_host": n_ctrl,
        "n_hosts": int(len(work)),
        "n_control_rows": int(len(c)),
        "excess_capable_blends_expected": round(float((out["ctrl_hits_any"] / n_ctrl).sum()), 1),
        "excess_flagged_observed": int(flagged.sum()),
        "ring_colour_blends_expected": round(float((out["ctrl_hits_ring"] / n_ctrl).sum()), 2),
        "ring_band_observed": int(ring.sum()),
        "global_ring_rate_per_position": p_ring_g,
    }
    return out, census


def screen_wd(df: pd.DataFrame, cfg: dict, *, rng=None, controls=None
              ) -> tuple[pd.DataFrame, dict]:
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
        colour_route = select_excess(ex, th)
        ex["excess_flag"] = colour_route
        ex["excess_route"] = np.where(colour_route.to_numpy(bool), "colour", "")
        achro = _achromatic_route(ex, th, w)
        ex["excess_flag"] = ex["excess_flag"] | achro
        ex["excess_route"] = np.where(achro.to_numpy(bool) & ~colour_route.to_numpy(bool),
                                      "achromatic", ex["excess_route"])
        parts.append(ex)
    none = work[work["sed_anchor"] == "none"].copy()
    if len(none):
        for col in ("chi_W1", "chi_W2", "chi_color", "W1_excess_jy", "W2_excess_jy"):
            none[col] = np.nan
        none["excess_flag"] = False
        none["excess_route"] = ""
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
    # A ring in the Osmanov band is red by construction, so it always enters by
    # the colour route; requiring that here keeps the achromatic admission from
    # ever producing a candidate.
    work["ring_candidate"] = flagged_mask & (work["verdict"] == "surviving") & \
        (work["shape_class"] == "ring_band") & \
        (work.get("excess_route", pd.Series("colour", index=work.index)) == "colour")

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
    cc, census = wd_chance_census(work, controls, cfg)
    for col in cc.columns:
        work[col] = cc[col]
    # The natural twin of a ring is a WD dust disk (sublimation-limited,
    # ~800-1800 K, the polluted-WD phenomenon).  A ring claim needs the fit's
    # UPPER temperature bound below the disk locus, not just its best value,
    # and (with controls) a chance probability for a ring-coloured blend
    # below the pulsar leg's per-host threshold.
    disk_t = float(cfg["debris_locus"]["t_min_k"])
    t_hi_ci = pd.to_numeric(work["t_ring_hi_k"], errors="coerce")
    work["ring_distinct_from_disk"] = (t_hi_ci < disk_t).fillna(False)
    p_ctrl = pd.to_numeric(work["p_chance_ring_ctrl"], errors="coerce") \
        if "p_chance_ring_ctrl" in work.columns else pd.Series(np.nan, index=work.index)
    p_max = float(cfg["pulsar"]["chance_p_max"])
    chance_bad = (p_ctrl > p_max).fillna(False) if census.get("status") == "OK" \
        else pd.Series(False, index=work.index)
    base = work["ring_candidate"].astype(bool)
    work["ring_veto"] = np.where(
        ~base, "", np.where(~work["ring_distinct_from_disk"], "ring_or_disk_ambiguous",
                            np.where(chance_bad, "chance_ring_coloured_blend", "")))
    work["ring_candidate"] = base & (work["ring_veto"] == "")
    # A mechanism name for every flagged excess (the census, host by host).
    sh = work["shape_class"].astype(str)
    route = work["excess_route"].astype(str) if "excess_route" in work.columns \
        else pd.Series("", index=work.index)
    gate = work["gate_reason"].astype(str)
    mech = np.select(
        [~flagged_mask,
         (sh == "companion") | (route == "achromatic") | (gate == "unresolved_companion"),
         gate == "background_source",
         gate.isin(["astrometric_registration", "wise_quality", "globular_cluster_sightline"]),
         gate == "ledger",
         sh == "debris_disk",
         (sh == "ring_band") & ~work["ring_distinct_from_disk"],
         sh == "ring_band"],
        ["", "cool_companion_photosphere", "background_source_chance_superposition",
         "blend_or_misregistered_wise_source", "photometry_ledger_rule",
         "wd_dust_disk_locus", "ring_or_dust_disk_ambiguous", "ring_band"],
        default="warm_ambiguous_or_unfit")
    work["mechanism"] = mech
    surv = flagged_mask & (work["verdict"] == "surviving")
    if census.get("status") == "OK":
        census["ring_band_surviving_gates"] = int((surv & (work["shape_class"] == "ring_band"))
                                                  .sum())
        census["ring_colour_blends_expected_among_surviving_hosts"] = round(float(
            (work.loc[surv, "ctrl_hits_ring"] / census["control_positions_per_host"]).sum()), 2)
    shape_counts = {k: int(v) for k, v in work.loc[flagged_mask, "shape_class"]
                    .value_counts().items()}
    summary = {
        "status": "OK",
        "n_input": int(len(df)),
        "sed_anchor_counts": n_anchor,
        "n_w1_detected": int(work.get("W1_detected", pd.Series(False)).sum()),
        "n_w2_detected": int(work.get("W2_detected", pd.Series(False)).sum()),
        "n_excess_flagged": int(flagged_mask.sum()),
        "excess_route_counts": {k: int(v) for k, v in
                                work.loc[flagged_mask, "excess_route"].value_counts().items()},
        "shape_counts": shape_counts,
        "n_surviving_gates": int((flagged_mask & (work["verdict"] == "surviving")).sum()),
        "n_ring_candidates": int(work["ring_candidate"].sum()),
        "chance_census": census,
        "mechanism_counts": {k: int(v) for k, v in work.loc[flagged_mask, "mechanism"]
                             .value_counts().items()},
        "ring_vetoes": {k: int(v) for k, v in work["ring_veto"].value_counts().items() if k},
        # A gate that could not be evaluated must be visible as such.
        "n_registration_tested": int(pd.Series(work.get("registration_tested", False))
                                     .fillna(False).astype(bool).sum()),
        "ring_band_gate_reasons": {k: int(v) for k, v in work.loc[
            flagged_mask & (work["shape_class"] == "ring_band"), "gate_reason"]
            .value_counts().items() if k},
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


def _allwise_colour_ok(df: pd.DataFrame, w1="W1mag", w2="W2mag") -> pd.Series:
    """AllWISE W1 and W2 are both MEASUREMENTS (ph_qual A/B/C with an error).

    ``ph_qual = U`` is an upper limit whose magnitude column holds the limit,
    not a flux.  Run 35752692549 turned the UUBU source near J1633-2009 into a
    "W1-W2 = 1.38, 720 K" counterpart from two upper limits.
    """
    ph_ = text_column(df, "ph_qual").str.upper()
    ok = ph_.str.slice(0, 1).isin(["A", "B", "C"]) & ph_.str.slice(1, 2).isin(["A", "B", "C"])
    e1 = _numcol(df, "e_W1mag")
    e2 = _numcol(df, "e_W2mag")
    m = _numcol(df, w1).notna() & \
        _numcol(df, w2).notna()
    return (ok & e1.notna() & e2.notna() & m).reindex(df.index).fillna(False).astype(bool)


def screen_pulsars(psr: pd.DataFrame, matches: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Per pulsar: the counterpart (if any), its colour temperature, its chance
    probability from the controls, its provenance vetoes, and the sensitivity.

    Chance probability (per catalogue).  The 16 offset positions give a local
    hit rate ``h/16`` -- an unbiased but coarse estimate.  It is pooled with a
    global prior that is scaled by the host's OWN aperture area: the control
    density per arcsec^2 over every control position, times ``pi r^2``.  The
    earlier unscaled prior averaged 8-arcsec and 1.5-arcsec apertures together,
    so it over-predicted chance matches for well-timed pulsars by ~3x and
    under-predicted them for the poorly localised ones.

    For a ring-band counterpart the relevant rate is that of control sources
    WITH a ring-band colour in the catalogue that supplied the colour -- not the
    smaller of the two catalogues' rates (run 35752692549 used the AllWISE ring
    rate for CatWISE-only counterparts).
    """
    p = cfg["pulsar"]
    if psr is None or not len(psr):
        return pd.DataFrame(), {"status": "NO_DATA_REACHED", "n_hosts": 0}
    out = psr.copy().reset_index(drop=True)
    r_floor, r_max = float(p["match_radius_arcsec"]), float(p["match_radius_max_arcsec"])
    fac = float(p["position_error_factor"])
    pos_err = pd.to_numeric(out["pos_err_arcsec"], errors="coerce").fillna(np.inf)
    out["match_radius_arcsec"] = np.clip(
        np.maximum(r_floor, fac * pos_err.replace(np.inf, r_max).to_numpy(float)),
        r_floor, r_max)
    # A pulsar whose error region is larger than the widest aperture searched
    # has no position to associate a WISE source with: the radius was capped,
    # the pulsar may be anywhere in the error region, and any source inside the
    # cap is a draw from the field.
    out["localised"] = (fac * pos_err <= r_max).to_numpy(bool)
    n_ctrl = 8 * len(p["control_offsets_arcsec"])
    beam = float(cfg.get("contamination", {}).get("beam", {}).get("wise_beam_arcsec", 6.5))
    area = np.pi * out["match_radius_arcsec"].to_numpy(float) ** 2

    t_lo, t_hi = float(cfg["ring"]["t_min_k"]), float(cfg["ring"]["t_max_k"])
    census: dict = {}
    for key in ("allwise", "catwise"):
        m = matches.get(key)
        out[f"{key}_match"] = False
        out[f"{key}_dist_arcsec"] = np.nan
        out[f"{key}_n_sources_in_radius"] = 0
        out[f"{key}_n_sources_in_beam"] = 0
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
        # Aperture-scale neighbours at the TARGET, before the radius cut: how
        # many catalogue sources share the WISE beam with the candidate.
        tall = m[m["_kind"] == "t"]
        nb = tall[tall["_d"] <= beam].groupby("_jname").size()
        out[f"{key}_n_sources_in_beam"] = out["jname"].map(nb).fillna(0).astype(int)
        m = m[m["_d"] <= m["_jname"].map(rad).to_numpy(float)]
        w1c = "W1mag" if key == "allwise" else "W1mag_cat"
        w2c = "W2mag" if key == "allwise" else "W2mag_cat"
        mcol = pd.to_numeric(m.get(w1c), errors="coerce") - pd.to_numeric(m.get(w2c),
                                                                          errors="coerce")
        if key == "allwise":
            mcol = mcol.where(_allwise_colour_ok(m))
        mt = ph.colour_to_temperature(mcol.to_numpy(float))
        m = m.assign(_ring=(mt >= t_lo) & (mt <= t_hi))
        nin = m[m["_kind"] == "t"].groupby("_jname").size()
        out[f"{key}_n_sources_in_radius"] = out["jname"].map(nin).fillna(0).astype(int)
        tgt = m[m["_kind"] == "t"].sort_values("_d").drop_duplicates("_jname")
        ctrl = m[m["_kind"] != "t"].drop_duplicates(["_jname", "_kind"])
        hits = ctrl.groupby("_jname").size()
        ring_hits = ctrl[ctrl["_ring"]].groupby("_jname").size()
        h = out["jname"].map(hits).fillna(0).astype(int)
        hr = out["jname"].map(ring_hits).fillna(0).astype(int)
        out[f"{key}_n_control_hits"] = h
        out[f"{key}_n_control_ring_hits"] = hr
        # Area-scaled empirical-Bayes prior (see the docstring).
        tot_area = float(n_ctrl * area.sum())
        dens_any = float(h.sum()) / tot_area if tot_area else 0.0
        dens_ring = float(hr.sum()) / tot_area if tot_area else 0.0
        prior_any = np.clip(dens_any * area, 0.0, 1.0)
        prior_ring = np.clip(dens_ring * area, 0.0, 1.0)
        out[f"{key}_p_chance"] = (h.to_numpy(float) + n_ctrl * prior_any) / (2.0 * n_ctrl)
        out[f"{key}_p_chance_ring"] = (hr.to_numpy(float) + n_ctrl * prior_ring) / (2.0 * n_ctrl)
        out.attrs[f"{key}_control_density_per_arcsec2"] = dens_any
        out.attrs[f"{key}_control_ring_density_per_arcsec2"] = dens_ring
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
        # The census: observed counterparts against the unbiased local
        # expectation (sum of h/16), overall and split by localisation.
        loc = out["localised"].to_numpy(bool)
        hr_obs = has & out["jname"].map(t["_ring"]).fillna(False).astype(bool)
        census[key] = {
            "observed": int(has.sum()),
            "expected_by_chance": round(float((h / n_ctrl).sum()), 1),
            "observed_localised": int(has[loc].sum()),
            "expected_localised": round(float((h[loc] / n_ctrl).sum()), 1),
            "observed_unlocalised": int(has[~loc].sum()),
            "expected_unlocalised": round(float((h[~loc] / n_ctrl).sum()), 1),
            "ring_colour_observed": int(hr_obs.sum()),
            "ring_colour_expected_by_chance": round(float((hr / n_ctrl).sum()), 2),
            "ring_colour_observed_localised": int(hr_obs[loc].sum()),
            "ring_colour_expected_localised": round(float((hr[loc] / n_ctrl).sum()), 2),
        }

    # Colour: AllWISE only where both bands are measurements, else CatWISE.
    aw_ok = _allwise_colour_ok(out) & out["allwise_match"].astype(bool)
    w1 = _numcol(out, "W1mag")
    w2 = _numcol(out, "W2mag")
    w1c = _numcol(out, "W1mag_cat")
    w2c = _numcol(out, "W2mag_cat")
    e_aw = np.hypot(_numcol(out, "e_W1mag"),
                    _numcol(out, "e_W2mag"))
    e_cw = np.hypot(_numcol(out, "e_W1mag_cat"),
                    _numcol(out, "e_W2mag_cat"))
    cw_ok = out["catwise_match"].astype(bool) & (w1c - w2c).notna()
    colour = (w1 - w2).where(aw_ok, (w1c - w2c).where(cw_ok))
    out["w1_w2"] = colour
    out["w1_w2_err"] = pd.Series(np.where(aw_ok, e_aw, np.where(cw_ok, e_cw, np.nan)),
                                 index=out.index)
    out["colour_source"] = np.where(aw_ok, "allwise", np.where(cw_ok, "catwise", ""))
    out["t_colour_k"] = ph.colour_to_temperature(colour.to_numpy(float))
    out["shape_class"] = [ph.classify_excess_shape(t, np.nan, cfg) if np.isfinite(t)
                          else ("no_counterpart" if not (a or b) else "unfit")
                          for t, a, b in zip(out["t_colour_k"], out["allwise_match"],
                                             out["catwise_match"], strict=False)]
    # A ring-band colour must stay in the band at 2 sigma toward the blue: a
    # colour whose 2-sigma interval reaches stellar/companion colours is not a
    # temperature measurement.  An unknown error counts as too large.
    blue = (colour - 2.0 * out["w1_w2_err"].fillna(np.inf)).to_numpy(float)
    t_blue = ph.colour_to_temperature(blue)
    out["colour_secure"] = np.isfinite(t_blue) & (t_blue <= t_hi)

    # Provenance vetoes: nebulae, clusters, companions, catalogued counterparts.
    assoc = text_column(out, "assoc").str.upper()
    binc = text_column(out, "bincomp").str.upper()
    out["assoc_veto"] = assoc.apply(lambda s: any(tok.upper() in s
                                                   for tok in p["assoc_veto_tokens"]))
    out["companion_veto"] = binc.apply(lambda s: any(tok.upper() == s.strip() or
                                                      s.strip().startswith(tok.upper())
                                                      for tok in p["bincomp_veto_tokens"]))
    out["known_counterpart_veto"] = text_column(out, "jname").isin(
        set(p["known_counterpart_veto"]))
    out["globular_cluster"] = assoc.str.contains("GC:", regex=False)
    # The chance probability that matters is the one for the hypothesis being
    # tested: for a ring-band counterpart, a random source with a ring-band
    # colour IN THE CATALOGUE THAT GAVE THE COLOUR; for anything else, any
    # random source in a catalogue that matched.
    p_any_aw = out["allwise_p_chance"].where(out["allwise_match"].astype(bool))
    p_any_cw = out["catwise_p_chance"].where(out["catwise_match"].astype(bool))
    p_any = pd.concat([p_any_aw, p_any_cw], axis=1).min(axis=1)
    p_ring = pd.Series(np.where(out["colour_source"] == "allwise", out["allwise_p_chance_ring"],
                                np.where(out["colour_source"] == "catwise",
                                         out["catwise_p_chance_ring"], np.nan)),
                       index=out.index)
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
    is_ring = out["shape_class"] == "ring_band"
    reason = pd.Series("", index=out.index, dtype=object)
    for col, name in (("_unloc", "position_not_localised"),
                      ("known_counterpart_veto", "catalogued_counterpart"),
                      ("assoc_veto", "nebula_cluster_or_optical_association"),
                      ("companion_veto", "non_degenerate_companion"),
                      ("globular_cluster", "globular_cluster_sightline")):
        v = ~out["localised"] if col == "_unloc" else out[col].astype(bool)
        f = v & (reason == "")
        reason[f] = name
    f = ~out["chance_ok"].fillna(False) & (reason == "")
    reason[f] = "chance_coincidence"
    f = is_ring & ~out["colour_secure"] & (reason == "")
    reason[f] = "colour_not_secure"
    f = is_ring & (out[["allwise_n_sources_in_beam", "catwise_n_sources_in_beam"]].max(axis=1)
                   >= 2) & (reason == "")
    reason[f] = "aperture_confusion"
    f = (out["shape_class"] == "companion") & (reason == "")
    reason[f] = "companion_colour"
    f = (out["shape_class"].isin(["unfit"])) & (reason == "")
    reason[f] = "single_band_no_colour"
    # Look-elsewhere: chance_p_max is a PER-HOST screen, and ~2,800 localised
    # hosts were searched.  A survivor must also be improbable across all of
    # them: p_trials = 1 - (1 - p)^N_localised.  Run 35860901093's single
    # survivor (J0418-4154, p = 0.006, W1-W2 = 0.48 +/- 0.52) has p_trials ~ 1.
    n_trials = max(int(out["localised"].sum()), 1)
    p_host = pd.to_numeric(pd.Series(out["p_chance"], index=out.index), errors="coerce")
    out["p_chance_trials"] = 1.0 - np.power(1.0 - p_host.clip(0.0, 1.0), n_trials)
    f = (out["p_chance_trials"] > float(p.get("trials_p_max", 1.0))) & (reason == "")
    reason[f] = "not_significant_after_trials"
    out["veto_reason"] = np.where(has, reason, "")
    out["verdict"] = np.where(~has, "no_counterpart",
                              np.where(out["veto_reason"] == "", "surviving", "rejected"))
    out["ring_candidate"] = (out["verdict"] == "surviving") & is_ring
    f500 = out["f_min_500K_W2"].to_numpy(float)
    fate_cols = ["jname", "bname", "localised", "pos_err_arcsec", "match_radius_arcsec",
                 "allwise_dist_arcsec", "catwise_dist_arcsec", "colour_source", "w1_w2",
                 "w1_w2_err", "t_colour_k", "ph_qual", "W1mag", "W2mag", "W1mag_cat",
                 "W2mag_cat", "catwise_n_control_hits", "catwise_n_control_ring_hits",
                 "allwise_n_control_ring_hits", "catwise_n_sources_in_beam",
                 "allwise_n_sources_in_beam", "p_chance_ring", "p_chance_any", "assoc",
                 "bincomp", "dist_kpc", "edot_w", "p_chance_trials", "veto_reason", "verdict"]
    fates = out.loc[has & is_ring, [c for c in fate_cols if c in out.columns]]
    summary = {
        "status": "OK",
        "n_hosts": int(len(out)),
        "n_localised": int(out["localised"].sum()),
        "n_with_allwise_counterpart": int(out["allwise_match"].sum()),
        "n_with_catwise_counterpart": int(out["catwise_match"].sum()),
        "n_with_any_counterpart": int(has.sum()),
        "n_with_any_counterpart_localised": int((has & out["localised"]).sum()),
        "control_positions_per_host": n_ctrl,
        "median_allwise_control_hits": float(out["allwise_n_control_hits"].median()),
        "median_catwise_control_hits": float(out["catwise_n_control_hits"].median()),
        "control_density_per_arcsec2": {k: float(v) for k, v in out.attrs.items()
                                        if k.endswith("per_arcsec2")},
        # Observed vs the unbiased local expectation, per catalogue.
        "chance_census": census,
        "shape_counts": {k: int(v) for k, v in out.loc[has, "shape_class"]
                         .value_counts().items()},
        "colour_source_counts": {k: int(v) for k, v in out.loc[has, "colour_source"]
                                 .value_counts().items()},
        "veto_reasons": {k: int(v) for k, v in out.loc[has, "veto_reason"]
                         .value_counts().items() if k},
        "n_surviving": int((out["verdict"] == "surviving").sum()),
        "n_ring_candidates": int(out["ring_candidate"].sum()),
        "ring_band_fates": _records(fates),
        "sensitivity": {
            "n_hosts_with_edot_and_distance": int(np.isfinite(f500).sum()),
            "n_hosts_500K_ring_detectable_at_f_lt_1": int((f500 < 1.0).sum()),
            "n_hosts_500K_ring_detectable_at_f_lt_0p1": int((f500 < 0.1).sum()),
            "median_f_min_500K": float(np.nanmedian(f500)) if np.isfinite(f500).any()
            else None,
        },
    }
    return out, summary


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records: NaN/inf -> None, numpy scalars -> Python."""
    recs = []
    for r in df.to_dict("records"):
        rec = {}
        for k, v in r.items():
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and not np.isfinite(v):
                v = None
            if v is pd.NA:
                v = None
            rec[k] = v
        recs.append(rec)
    return recs


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


def scan_parity_stats(t, mag, err) -> dict:
    """How much of a series' variance is an even/odd-epoch (scan-direction) split.

    NEOWISE visits a field every ~6 months, alternately on the ascending and
    descending scan, so consecutive epochs see the source at opposite PSF
    orientations.  A blend (an unresolved binary, a neighbour at a few arcsec)
    is measured differently on the two scans and alternates epoch by epoch; real
    atmospheric variability has no reason to lock to that parity.
    """
    t = np.asarray(t, float)
    m = np.asarray(mag, float)
    e = np.asarray(err, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if m.size < 4:
        return {"parity_frac": np.nan, "parity_z": np.nan}
    order = np.argsort(t)
    m, e = m[order], e[order]
    par = np.arange(m.size) % 2 == 0
    me, mo = m[par].mean(), m[~par].mean()
    se = np.sqrt((e[par] ** 2).sum()) / par.sum()
    so = np.sqrt((e[~par] ** 2).sum()) / (~par).sum()
    z = abs(me - mo) / np.hypot(se, so) if np.hypot(se, so) > 0 else np.nan
    var = m.var()
    frac = ((me - mo) ** 2 * par.mean() * (1 - par.mean()) / var) if var > 0 else np.nan
    return {"parity_frac": float(frac), "parity_z": float(z)}


# Empirical W1-W2 of late-T and Y dwarfs (numeric type, T0 = 20), from the
# Kirkpatrick+2011/2019/2021 colour-type relations; CH4 in W1 drives it.
_BD_W1W2 = ((26.0, 1.9), (27.0, 2.2), (28.0, 2.6), (29.0, 3.0), (30.0, 3.5), (31.0, 4.0),
            (32.0, 4.3))


def expected_w1_w2(spt_num):
    x, y = zip(*_BD_W1W2, strict=False)
    v = np.interp(np.asarray(spt_num, float), x, y, left=np.nan, right=y[-1])
    return v


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
            st.update(scan_parity_stats(gb["t_yr"] if len(gb) else [],
                                        gb["mag"] if len(gb) else [],
                                        gb["err"] if len(gb) else []))
            nexp = pd.to_numeric(gb["n_exp"], errors="coerce") if len(gb) and "n_exp" in gb \
                else pd.Series(dtype=float)
            st["mean_n_exp"] = float(nexp.mean()) if len(nexp) else np.nan
            rec.update({f"{band.lower()}_{k}": v for k, v in st.items()})
        rows.append(rec)
    out = pd.DataFrame(rows)
    n_min = int(b["min_epochs"])
    have = out["w2_n_epochs"] >= n_min
    pop_floor = float(np.nanmedian(out.loc[have, "w2_chi2_red"])) if have.any() else np.nan
    # The median estimates the NEOWISE systematic error floor only when the
    # tested sample is large enough that a genuine variable cannot be the
    # median itself; below that the floor is not estimable and applying it
    # would let one variable object raise the bar above its own signal.
    pop_n_min = int(b.get("pop_floor_min_n", 0))
    pop_applied = bool(np.isfinite(pop_floor) and int(have.sum()) >= pop_n_min)
    thr = max(float(b["chi2_red_min"]),
              float(b["chi2_red_pop_factor"]) * pop_floor if pop_applied else 0.0)
    out["w2_chi2_threshold"] = thr
    raw_flag = have & (out["w2_chi2_red"] >= thr) & (out["w2_amp_mag"] >= float(b["amp_min_mag"]))
    # Vet 1 -- is the series the target at all?  CH4 absorption in W1 makes a
    # >= T6 dwarf intrinsically very red (W1-W2 ~ 2-3.5 mag); a measured mean
    # colour far bluer than that is a background source or a blend in the cone,
    # not the brown dwarf (run 35860904385's one flag, the T8 WISE
    # J1813+2835, had W1-W2 = 0.93).  Untestable without W1 epochs.
    tmap = targets.assign(source_id=targets["source_id"].astype(str)).set_index("source_id")
    col_min = float(b.get("w1_w2_min_late_t", 1.5))
    tol = float(b.get("w1_w2_tolerance_mag", 1.0))
    out["w1_w2_mean"] = out["w1_mean_mag"] - out["w2_mean_mag"]
    spn = pd.to_numeric(out["spt_num"], errors="coerce")
    late = spn >= float(b["spt_min_numeric"])
    # The type-dependent expectation, with a tolerance; never looser than the floor.
    out["w1_w2_expected"] = expected_w1_w2(spn)
    need = np.maximum(col_min, out["w1_w2_expected"] - tol)
    out["colour_identity_ok"] = ~(late & (out["w1_w2_mean"] < need)).fillna(False)
    # Vet 2 -- was the cone following the target?  Without a proper motion the
    # cone sat at one epoch's position for a decade while a nearby brown dwarf
    # moved arcseconds; the series then mixes the target with whatever else
    # falls in the cone.
    pmra = pd.to_numeric(out["source_id"].map(tmap["pmra"]) if "pmra" in tmap.columns
                         else pd.Series(np.nan, index=out.index), errors="coerce")
    pmde = pd.to_numeric(out["source_id"].map(tmap["pmdec"]) if "pmdec" in tmap.columns
                         else pd.Series(np.nan, index=out.index), errors="coerce")
    out["pm_known"] = (pmra.notna() & pmde.notna()).to_numpy(bool)
    reason = pd.Series("", index=out.index, dtype=object)
    reason[raw_flag & ~out["colour_identity_ok"]] = "colour_not_the_target"
    reason[raw_flag & (reason == "") & ~out["pm_known"]] = "proper_motion_not_propagated"
    # Vet 3 -- a scan-direction (even/odd epoch) alternation is a blend.
    par = (out["w2_parity_frac"] >= float(b.get("parity_frac_max", 0.5))) & \
        (out["w2_parity_z"] >= float(b.get("parity_z_min", 3.0)))
    reason[raw_flag & (reason == "") & par.fillna(False)] = "scan_parity_blend"
    # Vet 4 -- at the single-exposure detection limit only the upward noise
    # excursions are detected, so epoch means scatter beyond their errors: a
    # series built from far fewer detected exposures per epoch than the
    # population's is detection-limited, not variable.
    pop_nexp = float(np.nanmedian(out.loc[have, "w2_mean_n_exp"])) if have.any() else np.nan
    lim = (out["w2_mean_n_exp"] < float(b.get("n_exp_frac_min", 0.6)) * pop_nexp)
    reason[raw_flag & (reason == "") & lim.fillna(False)] = "detection_limited_series"
    out["duty_cycle_veto"] = reason
    out["duty_cycle_candidate"] = raw_flag
    out["duty_cycle_flag"] = raw_flag & (reason == "")
    out["status"] = np.where(out["w2_n_epochs"] == 0, "NO_EPOCHS",
                             np.where(~have, "TOO_FEW_EPOCHS", "TESTED"))
    summary = {
        "status": "OK" if have.any() else ("NO_DATA_REACHED" if not len(ep) else
                                           "INSUFFICIENT_EPOCHS"),
        "n_targets": int(len(out)),
        "n_with_epochs": int((out["w2_n_epochs"] > 0).sum()),
        "n_tested": int(have.sum()),
        "population_w2_chi2_median": pop_floor if np.isfinite(pop_floor) else None,
        "population_floor_applied": pop_applied,
        "population_floor_min_n": pop_n_min,
        "w2_chi2_threshold": thr,
        "n_duty_cycle_flags": int(out["duty_cycle_flag"].sum()),
        "n_with_pm": int(out["pm_known"].sum()),
        "duty_cycle_vetoed": _records(out.loc[out["duty_cycle_candidate"]
                                              & ~out["duty_cycle_flag"],
                                              ["source_id", "spt", "w2_n_epochs", "w2_chi2_red",
                                               "w2_amp_mag", "w1_w2_mean", "w1_w2_expected",
                                               "w2_parity_frac", "w2_parity_z",
                                               "w2_mean_n_exp", "pm_known",
                                               "duty_cycle_veto"]]),
        "flagged": out.loc[out["duty_cycle_flag"], ["source_id", "spt", "w2_n_epochs",
                                                     "w2_chi2_red", "w2_amp_mag",
                                                     "w2_duty_cycle_high",
                                                     "w1_chi2_red"]].to_dict("records"),
    }
    return out, summary


# --------------------------------------------------------------------------
# Free-floating planets: hotter than cooling allows
# --------------------------------------------------------------------------

def _norm_group(s) -> str:
    t = str(s or "").lower().replace("β", "b").replace("beta", "b")
    return "".join(ch for ch in t if ch.isascii() and ch.isalnum())


def _group_age_gyr(group, cfg: dict) -> float:
    """Age of a named young group, through the config's names and aliases.

    Membership catalogues abbreviate (BANYAN: ``BPMG``, ``THA``, ``ABDMG``;
    Faherty+2016 also ``bPMG``, ``THMG``); the old substring test matched
    none of those against "beta Pic" / "Tuc-Hor", so an object with a group
    would still have had no age.  Matching is on the alphanumeric lower-case
    form, exact first, alias second; a suffix like "?" or "(amb)" is dropped
    by the normalisation.  An unmatched non-empty name stays NaN and is
    counted in the screen summary rather than guessed.
    """
    s = _norm_group(group)
    if not s or s in ("nan", "none", "field", "na", "fld", "old", "young"):
        return np.nan
    ages = {_norm_group(k): float(v) for k, v in cfg["ffp"]["age_fallback_gyr"].items()}
    if s in ages:
        return ages[s]
    aliases = {_norm_group(k): _norm_group(v)
               for k, v in (cfg["ffp"].get("group_aliases") or {}).items()}
    if s in aliases and aliases[s] in ages:
        return ages[aliases[s]]
    # Tolerate a trailing qualifier ("ABDMGamb", "TWAcand"): longest alias prefix.
    best = ""
    for k in list(ages) + list(aliases):
        if s.startswith(k) and len(k) >= 3 and len(k) > len(best):
            best = k
    if best:
        return ages.get(best, ages.get(aliases.get(best, ""), np.nan))
    return np.nan


def screen_ffp(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    f = cfg["ffp"]
    if df is None or not len(df):
        return pd.DataFrame(), {"status": "NO_DATA_REACHED", "n_objects": 0}
    out = df.copy().reset_index(drop=True)
    lbol = _numcol(out, "lbol").to_numpy(float)
    # Accept either log10(L/Lsun) or L/Lsun; a value above zero is not a log.
    with np.errstate(invalid="ignore", divide="ignore"):
        lbol = np.where(lbol > 0, np.log10(np.where(lbol > 0, lbol, 1.0)), lbol)
    out["log_lbol"] = lbol
    age = _numcol(out, "age").to_numpy(float)
    # Ages catalogued in Myr are the norm for young groups.
    age_gyr = np.where(age > 5.0, age / 1000.0, age)
    fallback = np.array([_group_age_gyr(g, cfg) for g in
                         out.get("group", pd.Series("", index=out.index))])
    out["age_gyr"] = np.where(np.isfinite(age_gyr), age_gyr, fallback)
    out["age_source"] = np.where(np.isfinite(age_gyr), "catalogue",
                                 np.where(np.isfinite(fallback), "group_fallback", "none"))
    mass = _numcol(out, "mass").to_numpy(float)
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
    grp = text_column(out, "group").str.strip()
    unmatched = grp[(grp != "") & (out["age_source"] == "none")]
    summary = {
        "status": "OK" if testable.any() else "NO_TESTABLE_OBJECTS",
        "n_objects": int(len(out)),
        "n_with_lbol": int(np.isfinite(out["log_lbol"]).sum()),
        "n_with_group": int((grp != "").sum()),
        "age_source_counts": {k: int(v) for k, v in out["age_source"].value_counts().items()},
        "unmatched_group_names": {k: int(v) for k, v in unmatched.value_counts().head(20).items()},
        "n_testable": int(testable.sum()),
        "n_catalogued_planetary_mass": int((planetary & testable).sum()),
        "n_hotter_than_ceiling": int(out["hotter_than_cooling"].sum()),
        "n_flags_planetary_and_hot": int(out["flag"].sum()),
        "flagged": out.loc[out["flag"], [c for c in (
            "name", "spt", "group", "age_gyr", "age_source", "log_lbol", "log_l_ceiling_13mj",
            "excess_over_ceiling_dex", "mass_mj") if c in out.columns]].to_dict("records"),
    }
    return out, summary


__all__ = ["screen_wd", "screen_pulsars", "screen_bd", "screen_ffp", "series_stats",
           "BANDS"]

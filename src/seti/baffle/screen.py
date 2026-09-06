"""BAFFLE selection, named vetoes, flags, verdict tokens.  Pure; offline.

Selection (deficit track)
-------------------------
``sig_w1 < −sig_min AND resid_w1 < −resid_min`` AND the same in W2.  Both
bands must show the deficit: a W1-only deficit with W2 normal is the T-dwarf
CH₄ absorption signature (veto ``w1_only_methane_like``); a W2-only deficit is
a single-band artefact (``w2_only_single_band``).  W3 / W4 are reported where
measured with a consistency status, never required.

Named vetoes, in order (each candidate records the FIRST that fires and ALL
that fire; each has a counter in the summary):
``saturated``, ``poor_wise_phot_qual``, ``poor_tmass_phot_qual``,
``wise_artifact``, ``extended``, ``bad_profile_fit``, ``wise_variable``,
``gaia_variable``, ``lpv_colour`` (deferred, not discarded: 2MASS 1998–2001 vs
WISE 2010 epoch mismatch — the fix is a same-epoch comparison),
``crowded_match``, ``blend_flux_theft`` (needs a neighbour table; otherwise
counted as ``neighbours_not_checked``), ``multi_peak``.
Report-only flags: ``bad_astrometry`` (RUWE; a 1-AU baffle does not perturb
astrometry) and ``high_pm_epoch_risk`` (the official cross-match propagates
PM, but flag it for the vet).  Survivors carry ``etz``, ``nearby``,
``distance_pc``.

Missing track
-------------
A bright star with a 2MASS match and no AllWISE counterpart is a candidate
only if |b| > b_min, ks in [ks_min, ks_max], 2MASS AAA, not Gaia-variable, not
non_single_star.  The per-band denominators give the missing fraction vs |b|
and vs G so plane confusion is visible rather than mistaken for a screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .locus import (
    BANDS,
    Locus,
    fit_locus,
    locus_quality_mask,
    residuals,
    w3_usable,
    wise_qual_letters,
)

VETO_ORDER = (
    "w1_only_methane_like", "w2_only_single_band",
    "saturated", "poor_wise_phot_qual", "poor_tmass_phot_qual", "wise_artifact",
    "extended", "bad_profile_fit", "wise_variable", "gaia_variable", "lpv_colour",
    "crowded_match", "blend_flux_theft", "multi_peak", "ks_too_bright_for_g",
)
DEFERRED_VETOES = ("lpv_colour",)
REPORT_FLAGS = ("bad_astrometry", "high_pm_epoch_risk", "gks_photospheric", "gks_unmeasured")

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_DEGRADED = "DEGRADED_SOURCE"
VERDICT_DEFICIT_NONE = "NO_MIDIR_DEFICIT_SURVIVOR"
VERDICT_DEFICIT_PENDING = "MIDIR_DEFICIT_CANDIDATES_PENDING_VET"
VERDICT_MISSING_NONE = "NO_MISSING_COUNTERPART_SURVIVOR"
VERDICT_MISSING_PENDING = "MISSING_COUNTERPART_CANDIDATES_PENDING_VET"

DEFAULT_SCREEN_CFG = {
    "sig_min": 5.0, "resid_min": 0.30, "w3_snr_min": 5.0, "w4_snr_min": 5.0,
    "saturated": {"w1mpro_min": 8.0, "w2mpro_min": 7.0, "ks_min": 4.5},
    "wise_ph_qual_ok": ["A", "B"], "tmass_ph_qual_ok": "AAA", "cc_flags_ok": "0000",
    "ext_flag_max": 0, "rchi2_max": 3.0, "var_flag_min": 6,
    "lpv": {"bp_rp_max": 3.0, "jk_max": 1.1},
    "crowded": {"number_of_mates_max": 0, "number_of_neighbours_max": 1,
                "wise_angdist_max_arcsec": 1.5},
    "blend_radius_arcsec": 8.0, "ipd_frac_multi_peak_max": 10, "non_single_star_max": 0,
    "ruwe_report": 1.4, "pm_report_mas_yr": 300.0, "etz_ecl_lat_deg": 0.264,
    "nearby_parallax_mas": 20.0,
    # (G - Ks) consistency: a contaminated Ks is too bright for the star's G by
    # the same amount it is too bright relative to W1; a screen is not.
    "gks": {"veto_min_mag": 0.2, "veto_nsig": 3.0, "consistency_tol_mag": 0.25,
            "photospheric_max_mag": 0.1, "photospheric_nsig": 2.0},
    "missing": {"b_min_deg": 10.0, "ks_min": 5.0, "ks_max": 11.0, "tmass_ph_qual_ok": "AAA",
                "babs_bin_deg": 10.0, "g_bin_mag": 1.0},
}


def _cfg(cfg: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_SCREEN_CFG.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df[col].astype(object).where(df[col].notna(), "").astype(str).str.strip()


def _var_digit(var_flag: pd.Series, pos: int) -> np.ndarray:
    s = var_flag.astype(str).str.ljust(4, "n").str[pos]
    return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# selection + vetoes
# ---------------------------------------------------------------------------
def band_pass(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Per-band boolean deficit passes and the W3 / W4 consistency status."""
    c = _cfg(cfg)
    out = pd.DataFrame(index=df.index)
    for band in ("w1", "w2", "w3"):
        sig, res = _num(df, f"sig_{band}"), _num(df, f"resid_{band}")
        out[f"pass_{band}"] = (sig < -float(c["sig_min"])) & (res < -float(c["resid_min"]))
    r3 = _num(df, "resid_w3")
    w3ok = w3_usable(df, {"w3_snr_min": c["w3_snr_min"], "w3_err_max": c.get("w3_err_max", 0.2)})
    w3_status = np.where(~np.isfinite(r3) | ~w3ok, "unmeasured",
                         np.where(r3 < -float(c["resid_min"]) / 2.0, "deficit",
                                  np.where(r3 > float(c["resid_min"]) / 2.0, "excess", "normal")))
    out["w3_status"] = w3_status
    w4snr = _num(df, "w4snr")
    out["w4_status"] = np.where(w4snr > float(c["w4_snr_min"]), "measured", "unmeasured")
    return out


def blend_flux_theft(df: pd.DataFrame, neighbours: pd.DataFrame | None,
                     radius_arcsec: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """(fires, checked): a BRIGHTER Gaia source within ``radius_arcsec``.

    ``neighbours`` carries ``target_source_id`` (the screened star), ``source_id``,
    ``phot_g_mean_mag`` and either ``sep_arcsec`` or ``ra``/``dec``.  Stars with
    no neighbour rows at all are ``checked = False``.
    """
    n = len(df)
    fires = np.zeros(n, dtype=bool)
    checked = np.zeros(n, dtype=bool)
    if neighbours is None or len(neighbours) == 0 or "target_source_id" not in neighbours.columns:
        return fires, checked
    nb = neighbours.copy()
    tid = pd.to_numeric(nb["target_source_id"], errors="coerce")
    sid = pd.to_numeric(nb.get("source_id", pd.Series(np.nan, index=nb.index)), errors="coerce")
    g_nb = pd.to_numeric(nb.get("phot_g_mean_mag", pd.Series(np.nan, index=nb.index)),
                         errors="coerce")
    own = pd.to_numeric(df.get("source_id", pd.Series(np.nan, index=df.index)), errors="coerce")
    own_g = _num(df, "phot_g_mean_mag")
    own_ra, own_dec = _num(df, "ra"), _num(df, "dec")
    pos = {int(s): i for i, s in enumerate(own) if pd.notna(s)}
    if "sep_arcsec" in nb.columns:
        sep = pd.to_numeric(nb["sep_arcsec"], errors="coerce").to_numpy(dtype=float)
    else:
        sep = np.full(len(nb), np.nan)
        ra_n, dec_n = _num(nb, "ra"), _num(nb, "dec")
        for k, t in enumerate(tid):
            i = pos.get(int(t)) if pd.notna(t) else None
            if i is None:
                continue
            dra = (ra_n[k] - own_ra[i]) * np.cos(np.radians(own_dec[i]))
            sep[k] = 3600.0 * np.hypot(dra, dec_n[k] - own_dec[i])
    for k, t in enumerate(tid):
        i = pos.get(int(t)) if pd.notna(t) else None
        if i is None:
            continue
        checked[i] = True
        if pd.notna(sid.iloc[k]) and pd.notna(own.iloc[i]) and int(sid.iloc[k]) == int(own.iloc[i]):
            continue
        if np.isfinite(sep[k]) and sep[k] <= radius_arcsec and pd.notna(g_nb.iloc[k]) \
                and np.isfinite(own_g[i]) and g_nb.iloc[k] < own_g[i]:
            fires[i] = True
    return fires, checked


def veto_table(df: pd.DataFrame, cfg: dict | None = None,
               neighbours: pd.DataFrame | None = None) -> pd.DataFrame:
    """One boolean column per named veto / report flag, plus ``neighbours_checked``."""
    c = _cfg(cfg)
    bp = band_pass(df, c)
    v = pd.DataFrame(index=df.index)
    p1, p2 = bp["pass_w1"].to_numpy(), bp["pass_w2"].to_numpy()
    v["w1_only_methane_like"] = p1 & ~p2
    v["w2_only_single_band"] = p2 & ~p1
    sat = c["saturated"]
    v["saturated"] = ((_num(df, "w1mpro") < float(sat["w1mpro_min"]))
                      | (_num(df, "w2mpro") < float(sat["w2mpro_min"]))
                      | (_num(df, "ks_m") < float(sat["ks_min"])))
    q1, q2 = wise_qual_letters(_str(df, "ph_qual"))
    good = [str(x).upper() for x in c["wise_ph_qual_ok"]]
    v["poor_wise_phot_qual"] = ~(q1.isin(good) & q2.isin(good)).to_numpy()
    v["poor_tmass_phot_qual"] = (_str(df, "tmass_ph_qual").str.upper()
                                 != str(c["tmass_ph_qual_ok"]).upper()).to_numpy()
    v["wise_artifact"] = (_str(df, "cc_flags") != str(c["cc_flags_ok"])).to_numpy()
    v["extended"] = np.nan_to_num(_num(df, "ext_flag"), nan=0.0) > float(c["ext_flag_max"])
    rmax = float(c["rchi2_max"])
    v["bad_profile_fit"] = (_num(df, "w1rchi2") > rmax) | (_num(df, "w2rchi2") > rmax)
    vf = _str(df, "var_flag")
    vmin = float(c["var_flag_min"])
    v["wise_variable"] = (_var_digit(vf, 0) >= vmin) | (_var_digit(vf, 1) >= vmin)
    v["gaia_variable"] = (_str(df, "phot_variable_flag").str.upper() == "VARIABLE").to_numpy()
    lpv = c["lpv"]
    jk = _num(df, "j_m") - _num(df, "ks_m")
    v["lpv_colour"] = (_num(df, "bp_rp") > float(lpv["bp_rp_max"])) | (jk > float(lpv["jk_max"]))
    cr = c["crowded"]
    mates = np.fmax(np.nan_to_num(_num(df, "wise_number_of_mates"), nan=0.0),
                    np.nan_to_num(_num(df, "tmass_number_of_mates"), nan=0.0))
    neigh = np.fmax(np.nan_to_num(_num(df, "wise_number_of_neighbours"), nan=1.0),
                    np.nan_to_num(_num(df, "tmass_number_of_neighbours"), nan=1.0))
    angd = _num(df, "wise_angular_distance")
    v["crowded_match"] = ((mates > float(cr["number_of_mates_max"]))
                          | (neigh > float(cr["number_of_neighbours_max"]))
                          | (angd > float(cr["wise_angdist_max_arcsec"])))
    fires, checked = blend_flux_theft(df, neighbours, float(c["blend_radius_arcsec"]))
    v["blend_flux_theft"] = fires
    v["neighbours_checked"] = checked
    v["multi_peak"] = ((_num(df, "ipd_frac_multi_peak") > float(c["ipd_frac_multi_peak_max"]))
                       | (np.nan_to_num(_num(df, "non_single_star"), nan=0.0)
                          > float(c["non_single_star_max"])))
    gk = c["gks"]
    rg, sg = _num(df, "resid_gks"), _num(df, "sig_gks")
    r1 = _num(df, "resid_w1")
    v["ks_too_bright_for_g"] = ((rg > float(gk["veto_min_mag"])) & (sg > float(gk["veto_nsig"]))
                                & (np.abs(rg + r1) < float(gk["consistency_tol_mag"])))
    v["gks_photospheric"] = ((np.abs(rg) < float(gk["photospheric_max_mag"]))
                             & (np.abs(sg) < float(gk["photospheric_nsig"])))
    v["gks_unmeasured"] = ~np.isfinite(rg)
    v["bad_astrometry"] = _num(df, "ruwe") > float(c["ruwe_report"])
    pm = np.hypot(np.nan_to_num(_num(df, "pmra"), nan=0.0), np.nan_to_num(_num(df, "pmdec"), nan=0.0))
    v["high_pm_epoch_risk"] = pm > float(c["pm_report_mas_yr"])
    v["pm_total_mas_yr"] = pm
    for col in ("pass_w1", "pass_w2", "pass_w3", "w3_status", "w4_status"):
        v[col] = bp[col].to_numpy()
    return v


def add_flags(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """``etz``, ``nearby``, ``distance_pc`` on a copy."""
    c = _cfg(cfg)
    out = df.copy()
    ecl = _num(out, "ecl_lat")
    plx = _num(out, "parallax")
    out["etz"] = np.abs(ecl) < float(c["etz_ecl_lat_deg"])
    out["nearby"] = plx > float(c["nearby_parallax_mas"])
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(plx > 0, 1000.0 / plx, np.nan)
    out["distance_pc"] = d
    return out


def _first_and_all(v: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cols = [k for k in VETO_ORDER if k in v.columns]
    arr = v[cols].to_numpy(dtype=bool)
    first = [next((cols[j] for j in range(len(cols)) if row[j]), "") for row in arr]
    alls = [";".join(cols[j] for j in range(len(cols)) if row[j]) for row in arr]
    return pd.Series(first, index=v.index), pd.Series(alls, index=v.index)


def screen_deficit(df: pd.DataFrame, cfg: dict | None = None,
                   neighbours: pd.DataFrame | None = None) -> dict:
    """The deficit funnel.  Returns candidates / vetoed / deferred_lpv frames + counters.

    ``df`` must already carry the residual columns from :func:`locus.residuals`.
    """
    c = _cfg(cfg)
    df = add_flags(df, c)
    n_in = int(len(df))
    v = veto_table(df, c, neighbours)
    above = (v["pass_w1"] | v["pass_w2"]).to_numpy()
    two = (v["pass_w1"] & v["pass_w2"]).to_numpy()
    first, alls = _first_and_all(v)
    vetoed_any = v[list(VETO_ORDER)].any(axis=1).to_numpy()
    deferred_only = (first.isin(DEFERRED_VETOES).to_numpy()
                     & ~v[[k for k in VETO_ORDER if k not in DEFERRED_VETOES]].any(axis=1).to_numpy())

    enriched = df.copy()
    for col in v.columns:
        enriched[col] = v[col].to_numpy()
    enriched["first_veto"] = first.to_numpy()
    enriched["vetoes"] = alls.to_numpy()

    counters = {k: int((v[k].to_numpy() & above).sum()) for k in VETO_ORDER}
    counters_first = {k: int(((first == k).to_numpy() & above).sum()) for k in VETO_ORDER}
    n_not_checked = int((above & two & ~v["neighbours_checked"].to_numpy()).sum())
    report = {k: int((v[k].to_numpy() & above & ~vetoed_any).sum()) for k in REPORT_FLAGS}

    survivors = above & ~vetoed_any
    cand = enriched[survivors].copy()
    vet = enriched[above & vetoed_any].copy()
    deferred = enriched[above & deferred_only].copy()
    funnel = {
        "n_screened": n_in,
        "n_with_residuals": int(np.isfinite(_num(df, "sig_w1")).sum()),
        "n_above_threshold_any_band": int(above.sum()),
        "n_above_threshold_two_band": int(two.sum()),
        "n_vetoed": int((above & vetoed_any).sum()),
        "n_deferred_lpv": int(deferred_only[above].sum()),
        "n_candidates": int(survivors.sum()),
        "n_candidates_etz": int(cand["etz"].sum()) if len(cand) else 0,
        "n_candidates_nearby": int(cand["nearby"].sum()) if len(cand) else 0,
    }
    denominators = {
        "n_screened": n_in,
        "n_screened_etz": int(df["etz"].sum()),
        "n_screened_nearby": int(df["nearby"].sum()),
        "n_screened_locus_sample": int(df["is_locus_sample"].fillna(False).astype(bool).sum())
        if "is_locus_sample" in df.columns else None,
        "note": ("rows on disk = the archive-side pre-selection (Ks-W1 or Ks-W2 below the "
                 "pre-cut) plus the uniform locus subsample; any star with a two-band "
                 "deficit deeper than the pre-cut is in it by construction"),
    }
    sort_cols = [k for k in ("sig_w2", "sig_w1") if k in cand.columns]
    if len(cand) and sort_cols:
        cand = cand.sort_values(sort_cols)
    return {"candidates": cand, "vetoed": vet, "deferred_lpv": deferred,
            "counters": counters, "counters_first_veto": counters_first,
            "report_flags": report, "neighbours_not_checked": n_not_checked,
            "funnel": funnel, "denominators": denominators}


# ---------------------------------------------------------------------------
# missing track
# ---------------------------------------------------------------------------
def missing_fractions(missing: pd.DataFrame, denominators: pd.DataFrame | None,
                      cfg: dict | None = None) -> dict:
    """Missing fraction vs |b| and vs G from the per-band denominator tables."""
    c = _cfg(cfg)["missing"]
    bw, gw = float(c["babs_bin_deg"]), float(c["g_bin_mag"])
    out: dict = {"by_babs": [], "by_g": [], "denominator_available": False}
    if denominators is None or len(denominators) == 0:
        return out
    den = denominators.copy()
    den["n"] = pd.to_numeric(den["n"], errors="coerce").fillna(0)
    grouped = den["babs_bin"].notna() if "babs_bin" in den else pd.Series(False, index=den.index)
    out["denominator_available"] = True
    out["n_denominator_total"] = int(den["n"].sum())
    out["n_missing_total"] = int(len(missing))
    out["missing_fraction_total"] = (float(len(missing) / den["n"].sum())
                                     if den["n"].sum() else None)
    if not grouped.any():
        out["note"] = "denominator is a plain COUNT(*) per band; |b| and G structure unavailable"
        return out
    d = den[grouped]
    babs_m = np.floor(np.abs(_num(missing, "b")) / bw)
    g_m = np.floor(_num(missing, "phot_g_mean_mag") / gw)
    by_b = d.groupby(pd.to_numeric(d["babs_bin"]).astype(int))["n"].sum()
    for k, ntot in by_b.items():
        nm = int((babs_m == k).sum())
        out["by_babs"].append({"babs_lo_deg": float(k * bw), "babs_hi_deg": float((k + 1) * bw),
                               "n_missing": nm, "n_total": int(ntot),
                               "fraction": float(nm / ntot) if ntot else None})
    by_g = d.groupby(pd.to_numeric(d["g_bin"]).astype(int))["n"].sum()
    for k, ntot in by_g.items():
        nm = int((g_m == k).sum())
        out["by_g"].append({"g_lo": float(k * gw), "g_hi": float((k + 1) * gw),
                            "n_missing": nm, "n_total": int(ntot),
                            "fraction": float(nm / ntot) if ntot else None})
    return out


def screen_missing(df: pd.DataFrame, cfg: dict | None = None,
                   denominators: pd.DataFrame | None = None) -> dict:
    """The missing-counterpart funnel with its counter table and fraction tables."""
    c = _cfg(cfg)
    m = c["missing"]
    df = add_flags(df, c)
    n_in = int(len(df))
    if n_in == 0:
        empty = df.copy()
        return {"candidates": empty, "counters": {}, "funnel": {"n_missing_rows": 0,
                "n_candidates": 0, "n_candidates_etz": 0, "n_candidates_nearby": 0},
                "fractions": missing_fractions(df, denominators, c),
                "denominators": {"n_missing_rows": 0, "n_missing_etz": 0, "n_missing_nearby": 0}}
    rules = {
        "low_latitude": ~(np.abs(_num(df, "b")) > float(m["b_min_deg"])),
        "ks_out_of_range": ~((_num(df, "ks_m") >= float(m["ks_min"]))
                             & (_num(df, "ks_m") <= float(m["ks_max"]))),
        "poor_tmass_phot_qual": (_str(df, "tmass_ph_qual").str.upper()
                                 != str(m["tmass_ph_qual_ok"]).upper()).to_numpy(),
        "gaia_variable": (_str(df, "phot_variable_flag").str.upper() == "VARIABLE").to_numpy(),
        "non_single_star": np.nan_to_num(_num(df, "non_single_star"), nan=0.0)
        > float(c["non_single_star_max"]),
    }
    rej = np.zeros(n_in, dtype=bool)
    first = np.array([""] * n_in, dtype=object)
    counters = {}
    for name, mask in rules.items():
        mask = np.asarray(mask, dtype=bool)
        counters[name] = int(mask.sum())
        newly = mask & ~rej
        first[newly] = name
        rej |= mask
    out = df.copy()
    out["first_rejection"] = first
    out["rejections"] = [";".join(k for k, mk in rules.items() if bool(np.asarray(mk)[i]))
                         for i in range(n_in)]
    cand = out[~rej].copy()
    funnel = {"n_missing_rows": n_in, "n_rejected": int(rej.sum()),
              "n_candidates": int(len(cand)),
              "n_candidates_etz": int(cand["etz"].sum()) if len(cand) else 0,
              "n_candidates_nearby": int(cand["nearby"].sum()) if len(cand) else 0}
    dens = {"n_missing_rows": n_in, "n_missing_etz": int(df["etz"].sum()),
            "n_missing_nearby": int(df["nearby"].sum())}
    return {"candidates": cand, "counters": counters, "funnel": funnel,
            "fractions": missing_fractions(df, denominators, c), "denominators": dens}


# ---------------------------------------------------------------------------
# sensitivity (honesty check, never the deliverable)
# ---------------------------------------------------------------------------
def sensitivity(sample: pd.DataFrame, locus: Locus, locus_cfg: dict | None,
                screen_cfg: dict | None, mags=(0.2, 0.3, 0.5, 1.0), *,
                max_stars: int = 20000, seed: int = 20260906) -> dict:
    """Inject two-band deficits into the locus sample; report recovery after vetoes."""
    ok, _ = locus_quality_mask(sample, locus_cfg)
    base = sample[ok]
    if len(base) > max_stars:
        base = base.sample(n=int(max_stars), random_state=seed)
    n = int(len(base))
    out = {"n_injected_per_mag": n, "recovered": {}}
    if n == 0:
        out["note"] = "no locus-grade stars to inject into"
        return out
    for dm in mags:
        inj = base.copy()
        for band in ("w1", "w2"):
            inj[f"{band}mpro"] = pd.to_numeric(inj[f"{band}mpro"], errors="coerce") + float(dm)
        if "w3mpro" in inj.columns:
            inj["w3mpro"] = pd.to_numeric(inj["w3mpro"], errors="coerce") + float(dm)
        r = residuals(inj, locus, locus_cfg)
        res = screen_deficit(r, screen_cfg)
        n_c = int(res["funnel"]["n_candidates"])
        n_two = int(res["funnel"]["n_above_threshold_two_band"])
        out["recovered"][f"{float(dm):g}"] = {
            "n_two_band_above_threshold": n_two, "n_survive_vetoes": n_c,
            "fraction_two_band": float(n_two / n), "fraction_survive": float(n_c / n),
            "first_veto_counts": {k: v for k, v in res["counters_first_veto"].items() if v}}
    return out


def fit_and_screen(sample: pd.DataFrame, locus_cfg: dict | None, screen_cfg: dict | None,
                   neighbours: pd.DataFrame | None = None) -> tuple[Locus, pd.DataFrame, dict]:
    """Convenience: fit the locus on the locus subsample, residuals on everything, screen."""
    locus = fit_locus(sample, locus_cfg)
    r = residuals(sample, locus, locus_cfg)
    return locus, r, screen_deficit(r, screen_cfg, neighbours)


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
def deficit_verdict(acq_verdict: str | None, n_screened: int, n_candidates: int) -> str:
    if acq_verdict in (None, "NO_DATA_REACHED", "NO_QUERY_ATTEMPTED") or n_screened == 0:
        return VERDICT_NO_DATA
    tok = (f"{VERDICT_DEFICIT_PENDING} (n={n_candidates})" if n_candidates > 0
           else VERDICT_DEFICIT_NONE)
    if acq_verdict == "PARTIAL_SAMPLE":
        return f"{VERDICT_DEGRADED} (partial deficit sample) {tok}"
    return tok


def missing_verdict(acq_verdict: str | None, n_rows: int, n_candidates: int) -> str:
    if acq_verdict in (None, "NO_DATA_REACHED", "NO_QUERY_ATTEMPTED"):
        return VERDICT_NO_DATA
    tok = (f"{VERDICT_MISSING_PENDING} (n={n_candidates})" if n_candidates > 0
           else VERDICT_MISSING_NONE)
    if acq_verdict == "PARTIAL_SAMPLE":
        return f"{VERDICT_DEGRADED} (partial missing sample) {tok}"
    return tok


def combine_verdicts(deficit: str | None, missing: str | None) -> str:
    parts = [p for p in (deficit, missing) if p]
    if not parts:
        return VERDICT_NO_DATA
    if all(p == VERDICT_NO_DATA for p in parts):
        return VERDICT_NO_DATA
    return " | ".join(parts)


__all__ = ["BANDS", "DEFAULT_SCREEN_CFG", "DEFERRED_VETOES", "REPORT_FLAGS", "VETO_ORDER",
           "VERDICT_DEFICIT_NONE", "VERDICT_DEFICIT_PENDING", "VERDICT_DEGRADED",
           "VERDICT_MISSING_NONE", "VERDICT_MISSING_PENDING", "VERDICT_NO_DATA",
           "add_flags", "band_pass", "blend_flux_theft", "combine_verdicts",
           "deficit_verdict", "fit_and_screen", "missing_fractions", "missing_verdict",
           "screen_deficit", "screen_missing", "sensitivity", "veto_table"]

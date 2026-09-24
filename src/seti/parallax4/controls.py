"""Positive and negative controls.  Pure functions over data the caller fetched.

The controls run FIRST on the runner and gate the sweep: if the photometric
controls fail, no grey event from the sweep is believed (the sweep refuses to
run), and if the astrometric controls fail, the DR4 photocentre verdicts are
not issued.

Photometric (gate for the DR3 sweep)
    P1  grey multi-transit dips injected into REAL Gaia DR3 light curves are
        recovered as tier A at >= ``min_recovery`` where the injected depth
        clears the star's own noise by ``snr_floor``;
    P2  dust-like dips (BP/RP depth ratio 1.7) injected the same way are
        called GREY at <= ``max_dust_as_grey``;
    P4  the bulk reader parsed every row with aligned arrays.
    P3  (reported) Gaia's own eclipsing binaries: how many that produced a
        GREY dip episode are caught by the periodicity test.

Astrometric (gate for the DR4 verdicts)
    A1  simulated DR4-shaped scenes each return their known verdict
        (blended EB -> BLEND_NEIGHBOUR with D recovered; EB on target, grey
        dip on target -> ON_TARGET; window neighbour -> SCAN_ANGLE_FLUX;
        CTI-like -> DETECTOR_FRAME);
    A2  the real DR4 prerelease: the reader recovers the published parallaxes
        and the DR3 photometric transits join on transit_id;
    A3  DR3 varstrometry (Hwang et al. 2020): among blended stars
        (ipd_frac_multi_peak high) eclipsing binaries show more astrometric
        jitter than photometrically quiet stars at the same G, the excess
        grows with photometric amplitude, and among isolated stars the
        excess is smaller.  This is the population-level form of the per-
        transit test DR4 enables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import epochs as E
from . import greydip as G
from . import photocentre as P
from . import simulate as S


# ---------------------------------------------------------------------------
# P1/P2: injection into real light curves
# ---------------------------------------------------------------------------
def injection_trials(phot: pd.DataFrame, cfg: G.GreyConfig, *, every: int = 50,
                     seed: int = 20260924, max_trials: int = 400) -> list[dict]:
    """Inject into every ``every``-th searchable source: one grey 2-transit dip
    and one dust-ratio 2-transit dip (separately), depth log-uniform 3-40%."""
    rng = np.random.default_rng(seed)
    trials: list[dict] = []
    for k, (sid, lc) in enumerate(phot.groupby("source_id", sort=False)):
        if k % every:
            continue
        if len(trials) >= max_trials:
            break
        base_ev, summ = G.detect_source(lc, cfg)
        if summ["status"] != "ok":
            continue
        depth = float(np.exp(rng.uniform(np.log(0.03), np.log(0.40))))
        for kind, ratio in (("grey", 1.0), ("dust", cfg.dust_ratio)):
            inj = G.inject_episode(lc, depth, ratio_bp_rp=ratio, n_transits=2, rng=rng)
            if inj is None:
                continue
            lc2, meta = inj
            ev, _ = G.detect_source(lc2, cfg)
            pre = G.recovered(base_ev, meta)
            hit = G.recovered(ev, meta)
            noise = float(np.nanmax([summ.get("mad_g", np.nan), 1e-4]))
            s_col = summ.get("s_col")
            trials.append({
                "source_id": int(sid), "kind": kind, "depth": depth, "ratio": ratio,
                "g_noise": noise, "snr": depth / noise, "med_flux_g": summ.get("med_flux_g"),
                "s_col": s_col,
                # can this star's colour noise tell grey from 1.7 at this depth?
                "colour_testable": bool(s_col is not None and np.isfinite(s_col)
                                        and s_col / depth <= 0.5 * 0.20),
                "pre_existing_event": pre is not None,
                "recovered": hit is not None,
                "grey_class": hit.get("grey_class") if hit else None,
                "coherence": hit.get("coherence") if hit else None,
                "tier": G.tier(hit) if hit else None,
            })
    return trials


def photometric_gate(trials: list[dict], *, reader_mismatched_rows: int = 0,
                     min_recovery: float = 0.80, max_dust_as_grey: float = 0.05,
                     snr_floor: float = 10.0, min_trials: int = 20) -> dict:
    t = pd.DataFrame(trials)
    rep: dict = {"n_trials": int(len(t))}
    if not len(t):
        rep.update(gate="FAIL", reason="no injection trials")
        return rep
    t = t[~t["pre_existing_event"]]
    if "colour_testable" not in t:
        t = t.assign(colour_testable=True)
    ct = t["colour_testable"].fillna(False).astype(bool)
    rep["frac_colour_testable"] = float(ct[t["snr"] >= snr_floor].mean()) if (t["snr"] >= snr_floor).any() else None
    # the gate is judged where the grey test can succeed: S/N above the floor
    # AND the star's own colour noise at most half the ratio tolerance
    g = t[(t["kind"] == "grey") & (t["snr"] >= snr_floor) & ct]
    d = t[(t["kind"] == "dust") & (t["snr"] >= snr_floor) & ct & t["recovered"]]
    rec = float((g["tier"] == "A").mean()) if len(g) else np.nan
    rec_any = float(g["recovered"].mean()) if len(g) else np.nan
    dust_grey = float((d["grey_class"] == "GREY").mean()) if len(d) else np.nan
    # recovery vs S/N, the completeness curve the sweep's funnel quotes
    bins = [0, 3, 5, 10, 20, 50, 1e9]
    curve = []
    gg = t[t["kind"] == "grey"]
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        s = gg[(gg["snr"] >= lo) & (gg["snr"] < hi)]
        curve.append({"snr_lo": lo, "snr_hi": hi, "n": int(len(s)),
                      "frac_tier_A": float((s["tier"] == "A").mean()) if len(s) else None})
    rep.update(n_grey_above_floor=int(len(g)), recovery_tier_A=rec, recovery_any=rec_any,
               n_dust_recovered=int(len(d)), dust_called_grey=dust_grey, completeness_vs_snr=curve,
               reader_mismatched_rows=int(reader_mismatched_rows))
    fails = []
    if len(g) < min_trials:
        fails.append(f"too few grey trials above S/N {snr_floor}: {len(g)}")
    elif not rec >= min_recovery:
        fails.append(f"P1 recovery {rec:.2f} < {min_recovery}")
    if len(d) >= min_trials and not dust_grey <= max_dust_as_grey:
        fails.append(f"P2 dust called grey {dust_grey:.2f} > {max_dust_as_grey}")
    if len(d) < min_trials:
        fails.append(f"too few recovered dust trials: {len(d)}")
    if reader_mismatched_rows:
        fails.append(f"P4 reader misaligned {reader_mismatched_rows} rows")
    rep["gate"] = "FAIL" if fails else "PASS"
    rep["failures"] = fails
    return rep


def eclipsing_binary_control(events: pd.DataFrame, vari: pd.DataFrame) -> dict:
    """P3: of Gaia's catalogued eclipsing binaries with a GREY dip episode, how
    many does our periodicity test catch, and how many reach tier A anyway."""
    if not len(events) or vari is None or not len(vari) or "in_vari_eclipsing_binary" not in vari:
        return {"n_ecl_with_grey_dip": 0, "note": "no catalogued EB among pilot events"}
    flag = vari["in_vari_eclipsing_binary"].map(lambda x: x is True or x == 1 or str(x).lower() == "true")
    ecl_ids = set(vari.loc[flag, "source_id"])
    e = events[(events["kind"] == "DIP") & (events["grey_class"] == "GREY")
               & events["source_id"].isin(ecl_ids)]
    per_src = e.groupby("source_id").agg(period_class=("period_class", "first"),
                                          n_ep=("n_dip_episodes", "first"),
                                          any_A=("tier", lambda s: bool((s == "A").any())))
    n = int(len(per_src))
    testable = per_src[per_src["period_class"].isin(["PERIODIC", "APERIODIC"])]
    return {"n_ecl_with_grey_dip": n,
            "n_period_testable": int(len(testable)),
            "frac_testable_caught_periodic": float((testable["period_class"] == "PERIODIC").mean())
            if len(testable) else None,
            "n_reaching_tier_A": int(per_src["any_A"].sum()),
            "note": "tier-A EBs are removed downstream by Gaia's own classification; this "
                    "number says how much that veto is carrying"}


# ---------------------------------------------------------------------------
# A1: simulated scenes
# ---------------------------------------------------------------------------
SCENES = (
    ("blended_eb", {}, "BLEND_NEIGHBOUR"),
    ("hidden_blend", {}, "BLEND_UNRESOLVED"),
    ("eb_on_target", {}, "ON_TARGET"),
    ("grey_dip", {}, "ON_TARGET"),
    ("window_neighbour", {"rho_mas": 700.0, "flux_ratio": 0.5}, "SCAN_ANGLE_FLUX"),
    ("cti", {}, "DETECTOR_FRAME"),
    ("single", {}, "NO_FLUX_VARIATION"),
)


def simulated_scene_control(n_seeds: int = 3, cfg: P.PhotocentreConfig | None = None) -> dict:
    rows = []
    for scene, kw, expect in SCENES:
        for seed in range(n_seeds):
            sim = S.simulate_source(scene, seed=seed, **kw)
            tr = E.collapse_transits(E.astrometry_ccd_frame(sim["astro"]))
            ph = E.photometry_from_long(sim["phot"], source_id=int(sim["astro"]["source_id"].iloc[0]))
            j = P.prepare_joint(E.join_photometry_astrometry(ph, tr))
            r = P.fit_photocentre(j, target_flux_g=sim["target_flux_g"],
                                  neighbours=sim["neighbours"], cfg=cfg)
            rows.append({"scene": scene, "seed": seed, "expected": expect, "verdict": r["verdict"],
                         "D_mas": r.get("D_mas"), "D_true_mas": sim["truth"]["D_true_mas"],
                         "ok": r["verdict"] == expect})
    df = pd.DataFrame(rows)
    blend = df[df["scene"] == "blended_eb"]
    d_err = float(np.max(np.abs(blend["D_mas"] - blend["D_true_mas"]) / blend["D_true_mas"]))
    ok = bool(df["ok"].all()) and d_err < 0.05
    return {"gate": "PASS" if ok else "FAIL", "n": int(len(df)), "n_ok": int(df["ok"].sum()),
            "blend_D_max_frac_error": d_err, "rows": rows}


# ---------------------------------------------------------------------------
# A2: the real DR4 prerelease
# ---------------------------------------------------------------------------
#: published parallaxes (mas) of the prerelease sources, from the ESA
#: prerelease page as transcribed by gaia-dr4-explorer's reference table
PRERELEASE_PARALLAX = {
    3937211745905473024: 25.6, 4181040337841125632: 1.0, 4318465066420528000: 2.0,
    1457486023639239296: 13.6, 2309425390592896: 0.9, 435469040545191680: 1.0,
    20694084440761600: 4.9, 1663617687609809280: 10.1, 3926186255616949504: 1.0,
    2237987199365376: 0.0, 10973744521070720: 0.0, 60730287810150016: 0.0,
}


def five_parameter_fit(tr: pd.DataFrame) -> dict:
    """Weighted 5-parameter fit of transit-level AL centroids (jitter solved)."""
    th = tr["theta"].to_numpy(float)
    s, c = np.sin(th), np.cos(th)
    t = tr["t_yr"].to_numpy(float)
    X = np.column_stack([s, c, tr["pf_al"].to_numpy(float), t * s, t * c])
    y = tr["x_al"].to_numpy(float)
    s0 = tr["sx_al"].to_numpy(float)
    jit = P._jitter(X, y, s0)
    beta, cov, r, chi2 = P._wls(X, y, np.sqrt(s0 ** 2 + jit ** 2))
    return {"parallax": float(beta[2]), "parallax_err": float(np.sqrt(cov[2, 2])),
            "pmra": float(beta[3]), "pmdec": float(beta[4]), "jitter_mas": float(jit),
            "n": int(len(y))}


def prerelease_control(ccd: pd.DataFrame, phot_by_source: dict[int, pd.DataFrame] | None = None,
                       neighbours_by_source: dict[int, pd.DataFrame] | None = None,
                       flux_by_source: dict[int, float] | None = None) -> dict:
    tr = E.collapse_transits(ccd)
    rows, joint = [], []
    for sid, g in tr.groupby("source_id"):
        f = five_parameter_fit(g)
        pub = PRERELEASE_PARALLAX.get(int(sid))
        tol = max(0.1, 5 * f["parallax_err"]) + (1.0 if f["jitter_mas"] > 1.0 else 0.0)
        rows.append({"source_id": int(sid), **f, "parallax_published": pub,
                     "ok": pub is not None and abs(f["parallax"] - pub) <= max(tol, 0.06 + 0.05 * pub)})
    for sid, ph in (phot_by_source or {}).items():
        g = tr[tr["source_id"] == int(sid)]
        j = E.join_photometry_astrometry(ph, g)
        n_tid = int(len(set(ph["transit_id"]) & set(ccd.loc[ccd["source_id"] == int(sid), "transit_id"])))
        rec = {"source_id": int(sid), "n_phot_transits": int(len(ph)),
               "n_transit_id_in_dr4": n_tid, "n_joint_usable": int(len(j))}
        if len(j):
            rec["max_abs_time_diff_s"] = float(np.max(np.abs(j["t"] - j["t_astro"])) * 86400)
            jj = P.prepare_joint(j)
            r = P.fit_photocentre(jj, target_flux_g=(flux_by_source or {}).get(int(sid)),
                                  neighbours=(neighbours_by_source or {}).get(int(sid)),
                                  cfg=P.PhotocentreConfig(n_min=10))
            rec.update({k: r.get(k) for k in ("verdict", "D_mas", "D_pa_deg", "sigma_D_max", "p_D",
                                              "ul95_mas", "z_detector", "p_scan_flux", "p_sector",
                                              "phi_rms", "jitter_mas", "parallax_mas",
                                              "p_phot_variable")})
            rec["reasons"] = r.get("reasons")
            rec["neighbours"] = r.get("neighbours")
        joint.append(rec)
    df = pd.DataFrame(rows)
    ok = bool(len(df) and df["ok"].all())
    join_ok = all(j.get("n_joint_usable", 0) >= 10 and j.get("max_abs_time_diff_s", 99) < 30
                  for j in joint) if joint else None
    return {"gate": "PASS" if ok and join_ok is not False else "FAIL",
            "parallax_recovery": rows, "n_sources": int(len(df)), "n_parallax_ok": int(df["ok"].sum()) if len(df) else 0,
            "joint": joint, "join_ok": join_ok}


# ---------------------------------------------------------------------------
# A3: DR3 varstrometry, population level
# ---------------------------------------------------------------------------
def varstrometry_control(ecl: pd.DataFrame, quiet: pd.DataFrame, *, blend_min: float = 8.0,
                         g_bin: float = 0.5, metric: str = "ruwe", p_max: float = 1e-3) -> dict:
    from scipy.stats import mannwhitneyu, spearmanr

    def prep(df, label):
        d = df.copy()
        d = d[np.isfinite(d[metric]) & np.isfinite(d["phot_g_mean_mag"])
              & np.isfinite(d["ipd_frac_multi_peak"])]
        d["amp"] = np.sqrt(d["phot_g_n_obs"].astype(float)) / d["phot_g_mean_flux_over_error"].astype(float)
        d["cls"] = np.where(d["ipd_frac_multi_peak"] >= blend_min, "blend",
                            np.where(d["ipd_frac_multi_peak"] == 0, "iso", "mid"))
        d["gbin"] = np.floor(d["phot_g_mean_mag"] / g_bin).astype(int)
        d["pop"] = label
        return d

    e = prep(ecl, "ecl")
    q = prep(quiet, "quiet")
    rep: dict = {"metric": metric, "n_ecl": int(len(e)), "n_quiet": int(len(q)),
                 "blend_min_ipd_frac_multi_peak": blend_min}
    if len(e) < 200 or len(q) < 200:
        rep.update(gate="DEGRADED", reason="too few stars returned")
        return rep
    # normalise each star by the quiet stars of its own blend class and G bin
    ref = q.groupby(["cls", "gbin"])[metric].median().rename("ref")
    both = pd.concat([e, q], ignore_index=True).join(ref, on=["cls", "gbin"])
    both = both[np.isfinite(both["ref"]) & (both["ref"] > 0)]
    both["z"] = both[metric] / both["ref"]
    out = {}
    for cls in ("blend", "iso"):
        ze = both[(both["pop"] == "ecl") & (both["cls"] == cls)]
        zq = both[(both["pop"] == "quiet") & (both["cls"] == cls)]
        if len(ze) < 30 or len(zq) < 30:
            out[cls] = {"n_ecl": int(len(ze)), "n_quiet": int(len(zq)), "note": "too few"}
            continue
        mw = mannwhitneyu(ze["z"], zq["z"], alternative="greater")
        sp = spearmanr(ze["amp"], ze["z"])
        out[cls] = {"n_ecl": int(len(ze)), "n_quiet": int(len(zq)),
                    "median_z_ecl": float(ze["z"].median()), "median_z_quiet": float(zq["z"].median()),
                    "excess_ratio": float(ze["z"].median() / max(zq["z"].median(), 1e-9)),
                    "p_mwu_ecl_gt_quiet": float(mw.pvalue),
                    "spearman_amp_vs_z": float(sp.statistic), "p_spearman": float(sp.pvalue)}
    rep["classes"] = out
    b, i = out.get("blend", {}), out.get("iso", {})
    ok = (b.get("p_mwu_ecl_gt_quiet", 1) < p_max and b.get("spearman_amp_vs_z", 0) > 0
          and b.get("p_spearman", 1) < p_max
          and b.get("excess_ratio", 0) > i.get("excess_ratio", np.inf))
    rep["gate"] = "PASS" if ok else "FAIL"
    rep["criteria"] = ("blended EBs out-jitter blended quiet stars (MWU p<1e-3), the excess rises "
                       "with photometric amplitude (Spearman>0, p<1e-3), and the excess among "
                       "isolated stars is smaller than among blended ones")
    return rep


# ---------------------------------------------------------------------------
# release day: the test of the test
# ---------------------------------------------------------------------------
def dr4_control_gate(results: pd.DataFrame, controls: pd.DataFrame, *, min_vim_blend: float = 0.7,
                     max_eb_blend: float = 0.1, min_n: int = 10) -> dict:
    """Gaia's own VIMs must mostly come back as blends; bright isolated EBs
    must almost never.  Until both hold, no DR4 photocentre verdict is a
    result.  Sources without a testable verdict (INSUFFICIENT, NOT_IN_DR4...)
    are not counted either way; their number is reported."""
    if controls is None or not len(controls) or results is None or not len(results):
        return {"gate": "NO_CONTROLS", "n_vim": 0, "n_eb": 0}
    m = results.merge(controls[["source_id", "role"]], on="source_id", how="inner")
    testable = ~m["verdict"].isin(["INSUFFICIENT", "NOT_IN_DR4_EPOCH_PRODUCTS", "DR4_SCHEMA_MISMATCH",
                                   "FIT_ERROR", "NO_FLUX_VARIATION"])
    blend = m["verdict"].isin(["BLEND_NEIGHBOUR", "BLEND_UNRESOLVED", "SCAN_ANGLE_FLUX"])
    vim = m[(m["role"] == "VIM_POSITIVE") & testable]
    eb = m[(m["role"] == "SINGLE_EB_NEGATIVE") & testable]
    f_vim = float(blend[vim.index].mean()) if len(vim) else np.nan
    f_eb = float(blend[eb.index].mean()) if len(eb) else np.nan
    rep = {"n_vim": int(len(vim)), "n_eb": int(len(eb)), "frac_vim_blend": f_vim, "frac_eb_blend": f_eb,
           "n_controls_untestable": int((~testable).sum())}
    if len(vim) < min_n or len(eb) < min_n:
        rep["gate"] = "INSUFFICIENT_CONTROLS"
    elif f_vim >= min_vim_blend and f_eb <= max_eb_blend:
        rep["gate"] = "PASS"
    else:
        rep["gate"] = "FAIL"
    return rep

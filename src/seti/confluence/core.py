"""Assemble channel score tables into one star index, and run the confluence test.

Pure functions over in-memory frames; ``run.py`` feeds them files.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .registry import ChannelSpec, independent
from .stats import (
    DEFAULT_EDGES,
    assign_cells,
    benjamini_hochberg,
    conditional_overlap,
    inject_subthreshold,
    percentile_rank,
    stratified_overlap,
    top_q_tail,
)

MATCH_RADIUS_ARCSEC = 2.0


def _uv(ra, dec) -> np.ndarray:
    ra = np.radians(np.asarray(ra, float))
    dec = np.radians(np.asarray(dec, float))
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def _chord(arcsec: float) -> float:
    return 2 * np.sin(np.radians(arcsec / 3600.0) / 2)


# ---------------------------------------------------------------------------
# Star identity
# ---------------------------------------------------------------------------


def unify(frames: dict[str, pd.DataFrame], anchor: pd.DataFrame | None = None,
          radius_arcsec: float = MATCH_RADIUS_ARCSEC) -> tuple[pd.DataFrame, dict]:
    """One long table (channel, star, score, flag, ra, dec) with a shared star key.

    * a row with a Gaia DR3 id is ``gaia:<id>``;
    * a row with only a position is matched to the NEAREST Gaia-keyed position
      (from every frame's Gaia rows, and ``anchor`` -- a covariate table with
      source_id/ra/dec -- when given) within ``radius_arcsec``;
    * rows still unmatched are matched to each other across channels by
      position, so two position-only channels (ZTF x Kepler) can still meet;
    * anything left keeps its own key and meets nobody.

    Epoch note: Gaia positions are J2016.0, ZTF/Kepler/survey positions are
    their own epochs; at 2" a star needs > ~0.4"/yr of proper motion over the
    gap to be lost, which is recorded (``n_unmatched``) rather than corrected.
    """
    rows = []
    for ch, f in frames.items():
        if f is None or not len(f):
            continue
        g = f.copy()
        g["channel"] = ch
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["channel", "star", "score", "flag", "ra", "dec"]), {}
    L = pd.concat(rows, ignore_index=True)
    L["star"] = L["key"].where(L["source_id"].notna(), None)
    L.loc[L["source_id"].notna(), "star"] = "gaia:" + L.loc[
        L["source_id"].notna(), "source_id"].astype("int64").astype(str)

    # Gaia position catalogue
    gpos = L.loc[L["source_id"].notna() & L["ra"].notna(), ["source_id", "ra", "dec"]]
    if anchor is not None and len(anchor):
        gpos = pd.concat([gpos, anchor[["source_id", "ra", "dec"]].dropna()], ignore_index=True)
    gpos = gpos.drop_duplicates("source_id")
    rep = {"n_rows": int(len(L)), "n_gaia_rows": int(L["source_id"].notna().sum()),
           "gaia_position_catalogue": int(len(gpos))}
    need = L["star"].isna() & L["ra"].notna() & L["dec"].notna()
    n_to_match = int(need.sum())
    if n_to_match and len(gpos):
        tree = cKDTree(_uv(gpos["ra"], gpos["dec"]))
        d, i = tree.query(_uv(L.loc[need, "ra"], L.loc[need, "dec"]),
                          distance_upper_bound=_chord(radius_arcsec))
        ok = np.isfinite(d)
        idx = L.index[need][ok]
        L.loc[idx, "star"] = "gaia:" + gpos["source_id"].to_numpy()[i[ok]].astype("int64") \
            .astype(str)
        L.loc[idx, "matched_by"] = "position->gaia"
        rep["n_position_matched_to_gaia"] = int(ok.sum())
    # position-only vs position-only, channel by channel
    left = L["star"].isna() & L["ra"].notna()
    cat = pd.DataFrame(columns=["star", "ra", "dec"])
    for ch in L.loc[left, "channel"].unique():
        sel = left & (L["channel"] == ch)
        sub = L.loc[sel]
        if len(cat):
            tree = cKDTree(_uv(cat["ra"], cat["dec"]))
            d, i = tree.query(_uv(sub["ra"], sub["dec"]),
                              distance_upper_bound=_chord(radius_arcsec))
            ok = np.isfinite(d)
            L.loc[sub.index[ok], "star"] = cat["star"].to_numpy()[i[ok]]
            L.loc[sub.index[ok], "matched_by"] = "position->position"
            sub = sub[~ok]
        new = pd.DataFrame({"star": "pos:" + sub["key"].astype(str), "ra": sub["ra"],
                            "dec": sub["dec"]})
        L.loc[sub.index, "star"] = new["star"]
        cat = pd.concat([cat, new], ignore_index=True)
    L["star"] = L["star"].fillna(L["key"])
    rep["n_unmatched_non_gaia"] = int(L["star"].str.startswith("pos:").sum())
    # one row per (channel, star): keep the most anomalous
    L = L.sort_values("score", ascending=False, na_position="last")
    L = L.drop_duplicates(["channel", "star"])
    return L.reset_index(drop=True), rep


def add_tails(L: pd.DataFrame, qs: Sequence[float] = (0.01, 0.05)) -> pd.DataFrame:
    """Per-channel percentile rank and top-q tails, computed over each channel's
    WHOLE parent (before any joint restriction)."""
    parts = []
    for _ch, g in L.groupby("channel", sort=False):
        g = g.copy()
        g["pct"] = percentile_rank(g["score"]).to_numpy()
        for q in qs:
            g[f"tail_q{q:g}"] = top_q_tail(g["score"], q).to_numpy()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def channel_pairs(specs: dict[str, ChannelSpec], present: Sequence[str],
                  rule: str = "strict") -> list[tuple[str, str]]:
    out = []
    for a, b in itertools.combinations(sorted(present), 2):
        if a in specs and b in specs and independent(specs[a], specs[b], rule):
            out.append((a, b))
    return out


def _tail_col(mode: str) -> str:
    return "flag" if mode == "flag" else f"tail_q{float(mode):g}"


def run_tests(L: pd.DataFrame, cov: pd.DataFrame, specs: dict[str, ChannelSpec],
              modes: Sequence[str] = ("0.01", "0.05", "flag"), min_joint: int = 30,
              covariates: Sequence[str] | None = None, rule: str = "strict",
              exclude_stars: set | None = None, triples: bool = True,
              k_nn: int = 100) -> pd.DataFrame:
    """Every independent pair (and triple) x tail mode -> one result row.

    Headline null: conditional independence given the covariates, with
    kNN-estimated tail propensities and an exact Poisson-binomial count
    (``stats.conditional_overlap``).  Cross-check: coarse covariate cells
    (``stats.assign_cells``), reported as ``p_cells``.  A group counts only if
    both agree (the caller decides; both are in the row).
    """
    covariates = list(covariates if covariates is not None else DEFAULT_EDGES)
    exclude_stars = exclude_stars or set()
    present = sorted(L["channel"].unique())
    parents = {ch: set(g["star"]) - exclude_stars for ch, g in L.groupby("channel")}
    wide = {m: L.pivot_table(index="star", columns="channel", values=_tail_col(m),
                             aggfunc="max", fill_value=False).astype(bool)
            for m in modes}
    cov = cov.set_index("star") if "star" in cov.columns else cov
    groups = [tuple(p) for p in channel_pairs(specs, present, rule)]
    if triples:
        for a, b, c in itertools.combinations(present, 3):
            if all((x, y) in groups or (y, x) in groups
                   for x, y in ((a, b), (a, c), (b, c))):
                groups.append((a, b, c))
    recs = []
    for grp in groups:
        J = set.intersection(*(parents[g] for g in grp))
        if len(J) < min_joint:
            recs.append({"channels": "+".join(grp), "k": len(grp), "n_joint": len(J),
                         "mode": None, "status": "JOINT_TOO_SMALL"})
            continue
        Jl = sorted(J)
        cells, crep = assign_cells(cov.reindex(Jl), covariates)
        covJ = cov.reindex(Jl)
        for m in modes:
            t = wide[m].reindex(Jl, fill_value=False)[list(grp)]
            if (t.sum() == 0).any():
                recs.append({"channels": "+".join(grp), "k": len(grp), "n_joint": len(J),
                             "mode": m, "status": "EMPTY_TAIL_IN_JOINT",
                             "tail_counts": t.sum().to_dict()})
                continue
            r = conditional_overlap(t, covJ, grp, covariates, k=k_nn)
            prep = {"covariates_used": [c for c in covariates if c in covJ.columns
                                        and covJ[c].notna().any()],
                    "covariates_missing": [c for c in covariates if c not in covJ.columns
                                           or not covJ[c].notna().any()]}
            rc = stratified_overlap(t, cells, grp)
            d = r.to_dict()
            d.update(channels="+".join(grp), k=len(grp), mode=m, status="TESTED",
                     members=";".join(map(str, r.members)),
                     p_cells=rc.p_value, expected_cells=round(rc.expected, 4),
                     n_cells_coarse=rc.n_cells,
                     frac_joint_in_singletons=rc.frac_joint_in_singletons,
                     covariates_used=",".join(prep["covariates_used"]),
                     covariates_missing=",".join(prep["covariates_missing"]))
            del d["n_members"]
            recs.append(d)
    out = pd.DataFrame(recs)
    if "p_value" in out:
        tested = out["status"] == "TESTED"
        out.loc[tested, "q_bh"] = benjamini_hochberg(out.loc[tested, "p_value"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def injection_trials(L: pd.DataFrame, cov: pd.DataFrame, pair: tuple[str, str],
                     k_values: Sequence[int] = (3, 5, 10, 20), q: float = 0.05,
                     alert_q: float = 0.005, n_trials: int = 40, seed: int = 11,
                     covariates: Sequence[str] | None = None, alpha: float = 0.01) -> dict:
    """Plant k sub-alert anomalies in BOTH channels of ``pair``; measure recovery.

    The planted stars are drawn from the joint parent among stars in NEITHER
    tail, and get scores between each channel's upper-q and upper-alert_q
    quantiles: every one of them is inside the confluence tail and below the
    level either channel alone would alert on.
    """
    rng = np.random.default_rng(seed)
    a, b = pair
    La = L[L["channel"] == a].set_index("star")
    Lb = L[L["channel"] == b].set_index("star")
    J = sorted(set(La.index) & set(Lb.index))
    covi = cov.set_index("star") if "star" in cov.columns else cov
    covariates = list(covariates if covariates is not None else DEFAULT_EDGES)
    covJ = covi.reindex(J)
    base_a, base_b = top_q_tail(La["score"], q), top_q_tail(Lb["score"], q)
    clean = [s for s in J if not base_a.get(s, False) and not base_b.get(s, False)]
    out = {"pair": list(pair), "n_joint": len(J), "q": q, "alert_q": alert_q,
           "n_trials": n_trials, "alpha": alpha, "by_k": []}
    tb0 = pd.DataFrame({a: base_a.reindex(J).fillna(False).to_numpy(),
                        b: base_b.reindex(J).fillna(False).to_numpy()}, index=J)
    base = conditional_overlap(tb0, covJ, (a, b), covariates)
    out["baseline"] = base.to_dict()
    for k in k_values:
        if k > len(clean):
            continue
        rec, frac, leak = [], [], 0
        for _ in range(n_trials):
            plant = list(rng.choice(clean, size=k, replace=False))
            mod = inject_subthreshold({a: La["score"], b: Lb["score"]}, plant, q, alert_q, rng)
            ta, tb = top_q_tail(mod[a], q), top_q_tail(mod[b], q)
            # sanity: planted stars are NOT in either channel's alert tail
            for ch in (a, b):
                leak += int(top_q_tail(mod[ch], alert_q).reindex(plant).fillna(False).sum())
            t = pd.DataFrame({a: ta.reindex(J).fillna(False).to_numpy(),
                              b: tb.reindex(J).fillna(False).to_numpy()}, index=J)
            r = conditional_overlap(t, covJ, (a, b), covariates)
            rec.append(r.p_value < alpha)
            frac.append(np.mean([p in set(r.members) for p in plant]))
        out["by_k"].append({"k": k, "recovery_rate": float(np.mean(rec)),
                            "mean_frac_planted_in_members": float(np.mean(frac)),
                            "planted_in_alert_tail": leak})
    return out


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

SYSTEMATIC_RULES = (
    ("ruwe", lambda v: v > 1.4, "RUWE>1.4 (unresolved companion / bad astrometry)"),
    ("ipd_frac_multi_peak", lambda v: v > 2, "ipd_frac_multi_peak>2 (resolved double in the IPD)"),
    ("phot_g_mean_mag", lambda v: v < 6.0, "G<6 (saturation regime in most surveys)"),
    ("log_density", lambda v: v > 2.3, "crowded: >200 Gaia neighbours within 30\""),
    ("classprob_dsc_combmod_galaxy", lambda v: v > 0.5, "DSC galaxy probability >0.5"),
    ("classprob_dsc_combmod_quasar", lambda v: v > 0.5, "DSC quasar probability >0.5"),
)


def trace_members(L: pd.DataFrame, cov: pd.DataFrame, tags: pd.DataFrame | None,
                  members: Sequence[str]) -> pd.DataFrame:
    """One row per member star: every channel's score/percentile/flag, the covariates,
    the known-class families, the systematics that fire, and a trace verdict."""
    members = list(dict.fromkeys(members))
    sub = L[L["star"].isin(members)]
    recs = []
    covi = cov.set_index("star") if "star" in cov.columns else cov
    tg = tags.set_index("star") if tags is not None and "star" in tags.columns else tags
    for s in members:
        r = {"star": s}
        g = sub[sub["star"] == s]
        chans = []
        for _, x in g.iterrows():
            chans.append(f"{x['channel']}(score={x['score']:.3g},pct={x['pct']:.4f}"
                         f"{',FLAG' if x['flag'] else ''})")
        r["channels"] = "; ".join(chans)
        r["n_channels"] = len(g)
        if covi is not None and s in covi.index:
            c = covi.loc[s]
            for k in ("ra", "dec", "phot_g_mean_mag", "bp_rp", "ruwe", "abs_b",
                      "ipd_frac_multi_peak", "log_density", "vari_class", "non_single_star"):
                if k in covi.columns:
                    r[k] = c[k]
            sys_hits = [lab for col, fn, lab in SYSTEMATIC_RULES
                        if col in covi.columns and pd.notna(c.get(col)) and fn(float(c[col]))]
        else:
            sys_hits = ["NO_COVARIATES"]
        r["systematics"] = "; ".join(sys_hits)
        fam = ""
        if tg is not None and s in tg.index:
            t = tg.loc[s]
            fam = str(t.get("families", "") or "")
            for k in ("main_id", "otype", "vsx_name", "vsx_type"):
                if k in tg.columns:
                    r[k] = t.get(k)
        r["families"] = fam
        if fam:
            r["trace_verdict"] = "KNOWN_CLASS:" + fam
        elif sys_hits:
            r["trace_verdict"] = "SYSTEMATIC:" + sys_hits[0]
        else:
            r["trace_verdict"] = "UNEXPLAINED"
        recs.append(r)
    return pd.DataFrame(recs)

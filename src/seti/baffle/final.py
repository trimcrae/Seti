"""BAFFLE ``final_verdict``: fold the vet and the patch stage together per candidate.

Why (run 34057027633)
---------------------
Nine ``SURVIVES_VET_NO_HIRES_KS`` stars reached the patch stage and every one
fails on physics the pipeline had *already measured* but never combined:

* five have ``own_flat_chi2`` of 731–27675 on 21 NEOWISE visits — the star's
  own mid-IR light curve is strongly variable.  A passive screen is constant;
  these are mid-IR variables the Gaia / WISE flags missed.
* two sit in the Magellanic Clouds (hundreds of Gaia sources within 10′, 14
  within 10″): NEOWISE and 2MASS photometry there — and in the inner Galactic
  plane — are confusion-dominated, and the ``MODULATED`` one is degenerate
  with the geometry-agnostic odd/even control (``alternation_control_sig``
  5.25 against ``modulation_sig`` 5.67).
* one has ``resid_gks`` = +0.91 at 14.7σ against ``resid_w1`` = −1.20 — the
  contaminated-Ks signature — missed because |resid_gks + resid_w1| = 0.29
  fell just outside the screen's ±0.25 window (now 0.35, and 0.15 mag).
* one sits at the threshold edge (resid −0.300 / −0.315) with
  ``resid_gks`` +0.188 at 3.2σ: contamination-consistent.

Rules (every one evaluated; ``final_flags`` lists all that fire; the verdict
is the first in this precedence)
--------------------------------------------------------------------------
``NEOWISE_VARIABLE``            own_neowise_n_visits ≥ min_own_visits and
                                own_flat_chi2 > own_chi2_max — unless the patch
                                verdict is a NON-degenerate MODULATED: at R ≈ 1 AU
                                the star itself switches annually with its edge
                                neighbours, and that coherent schedule is the
                                geometric signature, not variability
``MODULATION_DEGENERATE``       patch MODULATED but ``modulation_degenerate``
                                or alternation_control_sig ≥ ratio × modulation_sig
``CONFUSION_LIMITED``           n_neighbours within the 10′ search radius >
                                max_neighbours_10arcmin, or |b| < plane_b_deg
                                with gaia_n_10as ≥ plane_gaia_n_10as_min —
                                the Magellanic Clouds and the inner plane,
                                where NEOWISE and 2MASS photometry are
                                confusion-dominated
``KS_CONTAMINATION_CONSISTENT`` resid_gks > ks_resid_min at > ks_nsig and
                                |resid_gks + resid_w1| < ks_consistency_tol
``THRESHOLD_EDGE``              both |resid_w1|, |resid_w2| < resid_min + margin
                                and no other rule fired: reported, and it does
                                NOT survive on its own unless the high-
                                resolution Ks confirms 2MASS
``PATCH_NOT_COHERENT`` / ``PATCH_UNAVAILABLE``   the patch stage did not
                                return COHERENT_PATCH, non-degenerate
                                MODULATED, or ISOLATED_DEFICIT with own_constant
``SURVIVES_ALL``                none of the above fired and the patch verdict
                                is one of those three; ``SURVIVES_ALL_NO_HIRES_KS``
                                when only the high-resolution Ks is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FINAL_RULES = ("NEOWISE_VARIABLE", "MODULATION_DEGENERATE", "CONFUSION_LIMITED",
               "KS_CONTAMINATION_CONSISTENT", "THRESHOLD_EDGE")
FINAL_VERDICTS = FINAL_RULES + ("PATCH_NOT_COHERENT", "PATCH_UNAVAILABLE", "VET_NOT_SURVIVING",
                                "SURVIVES_ALL", "SURVIVES_ALL_NO_HIRES_KS")
SURVIVING = ("SURVIVES_ALL", "SURVIVES_ALL_NO_HIRES_KS")
PATCH_OK = ("COHERENT_PATCH", "MODULATED", "ISOLATED_DEFICIT")

DEFAULTS: dict = {
    "own_chi2_max": 5.0,
    "min_own_visits": 6,
    "modulation_degenerate_ratio": 0.8,
    "max_neighbours_10arcmin": 300,
    "plane_b_deg": 2.0,
    "plane_gaia_n_10as_min": 8,
    "ks_resid_min": 0.15,
    "ks_nsig": 3.0,
    "ks_consistency_tol": 0.35,
    "resid_min": 0.30,              # the screen's resid_min; overridden from screen config
    "threshold_edge_margin": 0.05,
}

_PATCH_COLS = ("status", "n_neighbours", "n_deficit_total", "n_deficit_inside", "coherence_p",
               "profile_shape", "own_status", "own_neowise_n_visits", "own_flat_chi2",
               "own_flat_chi2_w2", "own_offset_from_allwise", "own_constant",
               "modulation_sig", "modulation_null_p", "alternation_control_sig",
               "modulation_degenerate", "best_d_au", "best_R_au", "patch_verdict")


def _cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULTS)
    src = cfg or {}
    fin = src.get("final", src) if isinstance(src, dict) else {}
    out.update({k: v for k, v in (fin or {}).items() if k in DEFAULTS})
    scr = src.get("screen") if isinstance(src, dict) else None
    if isinstance(scr, dict) and "resid_min" in scr and "resid_min" not in (fin or {}):
        out["resid_min"] = float(scr["resid_min"])
    return out


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _bool(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[col].map(lambda x: str(x).strip().lower() in ("true", "1", "1.0")).to_numpy(dtype=bool)


def merge_patches(vetted: pd.DataFrame, patches: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join the patch columns (prefixed ``patch_`` where they would clash)."""
    out = vetted.reset_index(drop=True).copy()
    out["source_id"] = pd.to_numeric(out["source_id"], errors="coerce").astype("int64")
    if patches is None or len(patches) == 0 or "source_id" not in patches.columns:
        out["patch_available"] = False
        for c in _PATCH_COLS:
            if c not in out.columns:
                out[c if c.startswith(("own_", "patch_", "modulation", "alternation", "n_deficit",
                                       "n_neighbours", "coherence", "profile", "best_"))
                    else f"patch_{c}"] = np.nan
        return out
    p = patches.copy()
    p["source_id"] = pd.to_numeric(p["source_id"], errors="coerce").astype("int64")
    keep = [c for c in _PATCH_COLS if c in p.columns]
    p = p[["source_id"] + keep].drop_duplicates("source_id")
    ren = {c: f"patch_{c}" for c in keep if c in out.columns and c != "source_id"}
    p = p.rename(columns=ren)
    out = out.merge(p, on="source_id", how="left")
    out["patch_available"] = out["source_id"].isin(p["source_id"]).to_numpy()
    return out


def final_verdicts(vetted: pd.DataFrame, patches: pd.DataFrame | None, cfg: dict | None = None
                   ) -> tuple[pd.DataFrame, dict]:
    """Per-candidate ``final_verdict`` / ``final_flags`` and the rule counters."""
    c = _cfg(cfg)
    df = merge_patches(vetted, patches)
    n = len(df)
    if n == 0:
        rep = {"n_candidates": 0, "rule_counts_any": {k: 0 for k in FINAL_RULES},
               "verdict_counts": {k: 0 for k in FINAL_VERDICTS}, "n_survive_all": 0,
               "n_survive_all_no_hires_ks": 0, "survivors": []}
        df["final_verdict"], df["final_flags"] = pd.Series(dtype=object), pd.Series(dtype=object)
        return df, rep
    pv = df["patch_verdict"].astype(str).where(df["patch_verdict"].notna(), "") \
        if "patch_verdict" in df.columns else pd.Series([""] * n)
    pv = pv.to_numpy().astype(object)
    visits = _num(df, "own_neowise_n_visits")
    chi2 = _num(df, "own_flat_chi2")
    msig, csig = _num(df, "modulation_sig"), _num(df, "alternation_control_sig")
    degen = _bool(df, "modulation_degenerate")
    n_nb = _num(df, "n_neighbours") if "n_neighbours" in df.columns else _num(df, "patch_n_neighbours")
    b = _num(df, "b")
    n10 = _num(df, "gaia_n_10as")
    rg, sg, r1, r2 = (_num(df, k) for k in ("resid_gks", "sig_gks", "resid_w1", "resid_w2"))
    own_const = _bool(df, "own_constant")
    hires_ok = _bool(df, "ks_confirmed_hires")
    vet_verdict = df["vet_verdict"].astype(str).to_numpy() if "vet_verdict" in df.columns \
        else np.full(n, "SURVIVES_VET_NO_HIRES_KS", dtype=object)

    mod_degenerate = (pv == "MODULATED") & (degen | (csig >= float(c["modulation_degenerate_ratio"]) * msig))
    mod_coherent = (pv == "MODULATED") & ~mod_degenerate
    rules = {
        "NEOWISE_VARIABLE": ((visits >= int(c["min_own_visits"])) & (chi2 > float(c["own_chi2_max"]))
                             & ~mod_coherent),
        "MODULATION_DEGENERATE": mod_degenerate,
        "CONFUSION_LIMITED": ((n_nb > float(c["max_neighbours_10arcmin"]))
                              | ((np.abs(b) < float(c["plane_b_deg"])) & (n10 >= float(c["plane_gaia_n_10as_min"])))),
        "KS_CONTAMINATION_CONSISTENT": ((rg > float(c["ks_resid_min"])) & (sg > float(c["ks_nsig"]))
                                        & (np.abs(rg + r1) < float(c["ks_consistency_tol"]))),
    }
    edge_raw = ((np.abs(r1) < float(c["resid_min"]) + float(c["threshold_edge_margin"]))
                & (np.abs(r2) < float(c["resid_min"]) + float(c["threshold_edge_margin"])))
    any_physics = np.zeros(n, dtype=bool)
    for m in rules.values():
        any_physics |= np.asarray(m, dtype=bool)
    rules["THRESHOLD_EDGE"] = edge_raw & ~any_physics
    patch_ok = ((pv == "COHERENT_PATCH") | ((pv == "MODULATED") & ~rules["MODULATION_DEGENERATE"])
                | ((pv == "ISOLATED_DEFICIT") & own_const))
    patch_avail = df["patch_available"].to_numpy(dtype=bool) & (pv != "")

    verdict = np.full(n, "", dtype=object)
    flags = []
    for i in range(n):
        fired = [k for k in FINAL_RULES if bool(rules[k][i])]
        flags.append(";".join(fired))
        if fired and not (fired == ["THRESHOLD_EDGE"] and hires_ok[i]):
            verdict[i] = fired[0]
        elif vet_verdict[i] not in ("SURVIVES_VET", "SURVIVES_VET_NO_HIRES_KS"):
            verdict[i] = "VET_NOT_SURVIVING"
        elif not patch_avail[i]:
            verdict[i] = "PATCH_UNAVAILABLE"
        elif not patch_ok[i]:
            verdict[i] = "PATCH_NOT_COHERENT"
        elif vet_verdict[i] == "SURVIVES_VET" or hires_ok[i]:
            verdict[i] = "SURVIVES_ALL"
        else:
            verdict[i] = "SURVIVES_ALL_NO_HIRES_KS"
    df["final_flags"] = flags
    df["final_verdict"] = verdict
    for k, m in rules.items():
        df[f"final_{k.lower()}"] = np.asarray(m, dtype=bool)

    first = {k: int((verdict == k).sum()) for k in FINAL_RULES}
    rep = {
        "n_candidates": int(n),
        "n_with_patch": int(patch_avail.sum()),
        "rule_counts_any": {k: int(np.asarray(m).sum()) for k, m in rules.items()},
        "rule_counts_first": first,
        "verdict_counts": {k: int((verdict == k).sum()) for k in FINAL_VERDICTS},
        "n_survive_all": int((verdict == "SURVIVES_ALL").sum()),
        "n_survive_all_no_hires_ks": int((verdict == "SURVIVES_ALL_NO_HIRES_KS").sum()),
        "n_threshold_edge_rescued_by_hires_ks": int((rules["THRESHOLD_EDGE"] & hires_ok).sum()),
        "config": c,
        "survivors": df.loc[np.isin(verdict, SURVIVING),
                            [k for k in ("source_id", "ra", "dec", "l", "b", "phot_g_mean_mag", "ks_m",
                                         "resid_w1", "resid_w2", "resid_gks", "sig_gks", "vet_verdict",
                                         "patch_verdict", "own_flat_chi2", "own_neowise_n_visits",
                                         "n_neighbours", "final_verdict", "final_flags")
                             if k in df.columns]].to_dict(orient="records"),
        "non_survivors": [{"source_id": int(df.loc[i, "source_id"]), "final_verdict": verdict[i],
                           "final_flags": flags[i], "patch_verdict": pv[i],
                           "own_flat_chi2": (None if not np.isfinite(chi2[i]) else float(chi2[i])),
                           "n_neighbours": (None if not np.isfinite(n_nb[i]) else int(n_nb[i])),
                           "resid_gks": (None if not np.isfinite(rg[i]) else float(rg[i]))}
                          for i in range(n) if verdict[i] not in SURVIVING],
    }
    return df, rep


def final_deficit_verdict(rep: dict) -> str:
    a, b = int(rep.get("n_survive_all", 0)), int(rep.get("n_survive_all_no_hires_ks", 0))
    if a + b > 0:
        return f"MIDIR_DEFICIT_CANDIDATES_SURVIVE_ALL (n={a}, no_hires_ks={b})"
    return "NO_MIDIR_DEFICIT_SURVIVOR"


__all__ = ["DEFAULTS", "FINAL_RULES", "FINAL_VERDICTS", "PATCH_OK", "SURVIVING",
           "final_deficit_verdict", "final_verdicts", "merge_patches"]

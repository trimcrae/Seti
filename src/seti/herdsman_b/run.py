"""Runner: HERDSMAN-B census audit end-to-end (stages checkpoint to disk).

``herdsman-b --stage all`` on the runner does: membership pull -> chemistry
join (chunk-checkpointed) -> field pull -> score + report.  Any stage can be
re-run alone; completed checkpoints are skipped, so a killed job resumes at
the chunk where it died.  ``--stage spectro`` is the v2 spectroscopic
crossmatch (GALAH DR3 / APOGEE DR17); it is dispatched separately because it
needs only the membership checkpoint, not the GSP-Phot stages.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .acquire import fetch_chemistry, fetch_field, fetch_membership
from .score import radec_to_rgal, score_census


def _std_members(members: pd.DataFrame, chem: pd.DataFrame) -> pd.DataFrame:
    m = members.merge(chem, on="source_id", how="inner")
    lo = pd.to_numeric(m.get("mh_gspphot_lower"), errors="coerce")
    hi = pd.to_numeric(m.get("mh_gspphot_upper"), errors="coerce")
    sig = (hi - lo) / 2.0
    sig = sig.where(np.isfinite(sig) & (sig > 0), 0.15)  # conservative default
    out = pd.DataFrame({
        "cluster": m["cluster"], "source_id": m["source_id"],
        "prob": pd.to_numeric(m["prob"], errors="coerce"),
        "mh": pd.to_numeric(m["mh_gspphot"], errors="coerce"),
        "mh_sigma": sig,
        "teff": pd.to_numeric(m["teff_gspphot"], errors="coerce"),
        "gmag": pd.to_numeric(m["phot_g_mean_mag"], errors="coerce"),
        "ra": pd.to_numeric(m["ra"], errors="coerce"),
        "dec": pd.to_numeric(m["dec"], errors="coerce"),
        "parallax": pd.to_numeric(m["parallax"], errors="coerce"),
    })
    return out


def _std_field(field: pd.DataFrame) -> pd.DataFrame:
    plx = pd.to_numeric(field["parallax"], errors="coerce")
    d_kpc = 1.0 / plx
    r_gal = radec_to_rgal(field["ra"].to_numpy(float),
                          field["dec"].to_numpy(float), d_kpc.to_numpy(float))
    return pd.DataFrame({
        "mh": pd.to_numeric(field["mh_gspphot"], errors="coerce"),
        "r_gal": r_gal,
        "dist_kpc": d_kpc.to_numpy(float)})


def score_and_write(cfg: Config, members_std: pd.DataFrame,
                    field_std: pd.DataFrame, dump_top: int = 25) -> dict:
    out_dir = cfg.root / "results" / "herdsman_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab = score_census(members_std, field_std)
    tab.to_csv(out_dir / "cluster_scores.csv", index=False)

    cands = tab[tab["assembly_candidate"]] if len(tab) else tab
    dumps = []
    for name in cands.head(dump_top)["cluster"] if len(cands) else []:
        g = members_std[members_std["cluster"] == name]
        dumps.append({"cluster": str(name), "members": [
            {"source_id": int(r.source_id), "mh": float(r.mh),
             "mh_sigma": float(r.mh_sigma), "prob": float(r.prob),
             "teff": float(r.teff), "gmag": float(r.gmag)}
            for r in g.itertuples()]})
    (out_dir / "candidates.json").write_text(json.dumps({
        "n_scored": int(len(tab)),
        "n_candidates": int(cands.shape[0]) if len(tab) else 0,
        "candidates": (cands.to_dict("records") if len(tab) else []),
        "member_dumps": dumps}, indent=2, default=str))

    summary = {
        "n_clusters_scored": int(len(tab)),
        "n_assembly_candidates": int(cands.shape[0]) if len(tab) else 0,
        "x_trim_percentiles": ({p: float(np.nanpercentile(tab["x_trim"], p))
                                for p in (50, 90, 99)} if len(tab) else {}),
        "n_mag_systematic": int(tab["mag_systematic"].sum()) if len(tab) else 0,
        "n_beyond_phot_trust": (int(tab["beyond_phot_trust"].sum())
                                if len(tab) else 0),
        "n_gc_like": int(tab["gc_like"].sum()) if len(tab) else 0,
        "n_field_bin_matched": (int(tab["field_bin_matched"].sum())
                                if len(tab) else 0),
        "top": (tab.head(10)[["cluster", "n_used", "x_trim", "z_census",
                              "field_likeness", "two_pop", "corr_mh_gmag",
                              "mag_systematic", "beyond_phot_trust", "gc_like",
                              "assembly_candidate"]].to_dict("records")
                if len(tab) else []),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    lines = [
        "# HERDSMAN-B run report", "",
        f"Clusters scored: {summary['n_clusters_scored']}; assembly "
        f"candidates: {summary['n_assembly_candidates']}.", "",
        "A candidate is a bound group whose Teff-detrended, interloper-trimmed",
        "[M/H] spread is a >=4-sigma outlier against comparable-N clusters,",
        "exceeds twice its error floor, is unimodal, and mirrors the local",
        "field spread (docs/herdsman.md section 5). v2 vetoes (each a failure",
        "mode v1 demonstrated): |corr(mh_resid, G)| > 0.4 (GSP-Phot",
        "extinction/magnitude systematic), dist >= 2.5 kpc (beyond the",
        "GSP-Phot trust region), and gc_like (globular-cluster heuristic);",
        "the field baseline is distance-matched. Two-population outliers",
        "are reported separately (natural heterogeneous channel).", "",
        "Top of census (by z):", "",
    ]
    for r in summary["top"]:
        corr = r["corr_mh_gmag"]
        corr_s = f"{corr:.2f}" if np.isfinite(corr) else "nan"
        lines.append(f"- {r['cluster']}: n={r['n_used']}, "
                     f"x_trim={r['x_trim']:.2f}, z={r['z_census']:.1f}, "
                     f"field_likeness={r['field_likeness']:.2f}, "
                     f"two_pop={r['two_pop']}, corr_mh_gmag={corr_s}, "
                     f"mag_sys={r['mag_systematic']}, "
                     f"far={r['beyond_phot_trust']}, gc={r['gc_like']}, "
                     f"candidate={r['assembly_candidate']}")
    two_pop_n = int(tab["two_pop"].sum()) if len(tab) else 0
    lines += ["", f"Two-population flags (stripped-nucleus/merger channel): "
                  f"{two_pop_n}. Vetoes: mag_systematic="
                  f"{summary['n_mag_systematic']}, beyond_phot_trust="
                  f"{summary['n_beyond_phot_trust']}, gc_like="
                  f"{summary['n_gc_like']}; distance-matched field bins used "
                  f"for {summary['n_field_bin_matched']} clusters.", "",
              "No-null rule: an empty candidate list at these thresholds is a "
              "domain statement (this census, these quality cuts), not a "
              "result — next moves are deeper chemistry (GALAH/APOGEE "
              "crossmatch), co-moving groups beyond the cluster census, and "
              "per-candidate spectroscopic follow-up."]
    (out_dir / "REPORT.md").write_text("\n".join(lines))
    print("[herdsman-b]", json.dumps(summary["x_trim_percentiles"]),
          f"-> {summary['n_assembly_candidates']} candidates")
    return summary


def herdsman_b_run(cfg: Config | None = None, stage: str = "all") -> dict:
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "herdsman_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    members = chem = field = None
    if stage in ("catalog", "all"):
        members = fetch_membership(out_dir / "members.parquet")
    if stage in ("chem", "all"):
        members = members if members is not None else \
            pd.read_parquet(out_dir / "members.parquet")
        chem = fetch_chemistry(members, out_dir / "chem")
        merged = _std_members(members, chem)
        merged.to_parquet(out_dir / "members_chem.parquet", index=False)
        print(f"[herdsman-b] standardized member-chemistry table: "
              f"{len(merged)} rows")
    if stage in ("field", "all"):
        field = fetch_field(out_dir / "field.parquet")
    if stage in ("score", "all"):
        members_std = pd.read_parquet(out_dir / "members_chem.parquet")
        field_raw = field if field is not None else \
            pd.read_parquet(out_dir / "field.parquet")
        return score_and_write(cfg, members_std, _std_field(field_raw))
    if stage == "spectro":
        # Spectroscopic crossmatch (GALAH DR3 [+ APOGEE DR17 when it
        # resolves]); needs the membership checkpoint (members.parquet from
        # the catalog stage) — GSP-Phot chem/field stages are not required.
        from .spectro import spectro_run
        return spectro_run(cfg)
    return {"stage": stage, "status": "done"}


__all__ = ["herdsman_b_run", "score_and_write"]

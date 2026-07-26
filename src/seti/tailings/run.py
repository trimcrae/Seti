"""TAILINGS stage orchestration; writes ``results/tailings/``.

Stages
------
``acquire``   pull GALAH / APOGEE cool dwarfs and the wide-binary catalogue
``manifold``  fit the natural abundance manifold and the empirical scatter
``sparse``    compute the sparse statistic and the sparse/dense contrast
``vet``       run the contamination funnel over the sparse candidates
``twins``     stage 4 -- co-natal pairs against the engulfment mass budget
``all``       the chain, resuming from whatever checkpoints exist

Every stage checkpoints, so a killed run restarts where it died and a
re-analysis never needs to re-fetch. The reduction is deliberately separable
from acquisition: the manifold, the statistic and the thresholds are the parts
most likely to need revision, and none of them should cost an archive pull.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from . import manifold as M
from . import sparse as S
from . import twins as T
from . import vet as V

CHANNEL = "tailings"


def _cfg_block(cfg: Config) -> dict:
    return (cfg.thresholds or {}).get(CHANNEL, {}) or {}


def _out_dir(cfg: Config) -> Path:
    d = cfg.root / "results" / CHANNEL
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sparse_config(block: dict) -> S.SparseConfig:
    b = (block.get("sparse") or {}).copy()
    known = {f for f in S.SparseConfig.__dataclass_fields__}
    return S.SparseConfig(**{k: v for k, v in b.items() if k in known})


def _vet_config(block: dict) -> V.VetConfig:
    b = (block.get("vet") or {}).copy()
    known = {f for f in V.VetConfig.__dataclass_fields__}
    if "flag_columns" in b:
        b["flag_columns"] = tuple(b["flag_columns"])
    return V.VetConfig(**{k: v for k, v in b.items() if k in known})


def _twin_config(block: dict) -> T.TwinConfig:
    b = (block.get("twins") or {}).copy()
    known = {f for f in T.TwinConfig.__dataclass_fields__}
    return T.TwinConfig(**{k: v for k, v in b.items() if k in known})


# ---------------------------------------------------------------------------
# Stage 1 -- acquisition
# ---------------------------------------------------------------------------
def stage_acquire(cfg: Config, *, surveys: list[str], max_rows: int, out_dir: Path) -> dict:
    from .acquire import apply_element_flags, fetch_survey, fetch_wide_binaries, write_checkpoint

    block = _cfg_block(cfg)
    sample = block.get("sample", {})
    prov = {"surveys": [], "wide_binaries": None}

    for sv in surveys:
        acq = fetch_survey(
            sv,
            teff_max=float(sample.get("teff_max", 6000.0)),
            teff_min=float(sample.get("teff_min", 3000.0)),
            logg_min=float(sample.get("logg_min", 4.0)),
            snr_min=float(sample.get("snr_min", 40.0)),
            max_rows=max_rows,
        )
        prov["surveys"].append(acq.provenance())
        if acq.n_rows:
            tab = apply_element_flags(acq.table, acq.elements)
            write_checkpoint(tab, out_dir / f"stars_{sv.lower()}.parquet")
            print(f"[tailings] {sv}: {acq.n_rows} rows from {acq.source_used}")
        else:
            print(f"[tailings] {sv}: {acq.degradation}")

    wb = fetch_wide_binaries(max_r_chance_align=float(
        (block.get("twins") or {}).get("max_r_chance_align", 0.1)))
    prov["wide_binaries"] = wb.provenance()
    if wb.n_rows:
        write_checkpoint(wb.table, out_dir / "wide_binaries.parquet")

    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return prov


# ---------------------------------------------------------------------------
# Stage 2/3 -- manifold and the sparse statistic
# ---------------------------------------------------------------------------
def reduce_survey(
    stars: pd.DataFrame,
    *,
    survey: str,
    block: dict,
    out_dir: Path,
) -> dict:
    """Fit the manifold, score sparsity and vet, for one survey's table."""
    scfg = _sparse_config(block)
    vcfg = _vet_config(block)
    mblock = block.get("manifold", {})

    stars = V.dedupe(stars, id_col="star_id", snr_col="snr")
    elements = [e for e in stars.columns
                if e in M.NUCLEO_FAMILIES["alpha"] + M.NUCLEO_FAMILIES["odd_z"]
                + M.NUCLEO_FAMILIES["fe_peak"] + M.NUCLEO_FAMILIES["s_light"]
                + M.NUCLEO_FAMILIES["s_heavy"] + M.NUCLEO_FAMILIES["r_mixed"]
                + M.NUCLEO_FAMILIES["r_process"] + M.NUCLEO_FAMILIES["cno"]
                + M.NUCLEO_FAMILIES["light"]]
    elements = [e for e in elements if stars[e].notna().sum() >= int(mblock.get("min_rows", 200))]

    if len(elements) < scfg.min_elements or len(stars) < int(mblock.get("min_rows", 200)):
        return {
            "survey": survey,
            "n_stars": int(len(stars)),
            "n_elements": len(elements),
            "verdict": ("INSUFFICIENT_SAMPLE: too few stars or elements survived the quality "
                        "cuts to define a manifold, so no sparsity claim is possible"),
            "n_sparse": 0,
            "n_vetted": 0,
        }

    mani = M.fit_manifold(
        stars,
        elements,
        teff_col="teff",
        logg_col="logg",
        feh_col="fe_h",
        snr_col="snr",
        degree=int(mblock.get("degree", 2)),
        clip=float(mblock.get("clip_sigma", 4.0)),
        n_iter=int(mblock.get("clip_iterations", 4)),
        min_rows=int(mblock.get("min_rows", 200)),
        min_count=int(mblock.get("scatter_min_count", 40)),
        floor=float(mblock.get("scatter_floor_dex", 0.005)),
    )
    Z, sig = M.zscores(stars, mani, err_prefix="e_")
    stats = S.sparse_statistics(Z, cfg=scfg)
    rates = S.element_flag_rates(stats, Z, cfg=scfg)
    contrast = S.contrast_table(stats)

    pd.DataFrame(mani.to_summary()).to_csv(out_dir / f"manifold_{survey.lower()}.csv", index=False)
    rates.to_csv(out_dir / f"element_flag_rates_{survey.lower()}.csv", index=False)
    contrast.to_csv(out_dir / f"contrast_{survey.lower()}.csv", index=False)

    keep = ["star_id", "ra", "dec", "teff", "logg", "fe_h", "snr", "chi2", "ruwe",
            "vbroad", "rv_scatter", "field_id", "survey"]
    keep = [c for c in keep if c in stars.columns]
    joined = pd.concat([stars[keep].reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
    for el in Z.columns:
        joined[f"z_{el}"] = Z[el].to_numpy()

    cand = joined[joined["classification"] == S.SPARSE].copy()
    n_sparse = int(len(cand))

    if n_sparse:
        cand = V.vet_candidates(cand, cfg=vcfg, survey=survey)
        cand = V.element_rate_veto(cand, rates, cfg=vcfg)
        flagged_mask = (stats["n_discrepant"].to_numpy() > 0)
        cand = V.field_rate_veto(cand, joined, flagged_mask, cfg=vcfg)
        cand = cand.sort_values("z_max", ascending=False, ignore_index=True)
        cand.to_csv(out_dir / f"candidates_{survey.lower()}.csv", index=False)
    n_vetted = int(cand["vet_pass"].sum()) if n_sparse else 0

    counts = stats["classification"].value_counts().to_dict()
    return {
        "survey": survey,
        "n_stars": int(len(stars)),
        "n_elements": len(elements),
        "elements": elements,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "n_sparse": n_sparse,
        "n_vetted": n_vetted,
        "median_sigma_dex": {
            el: round(float(np.nanmedian(sig[el])), 4) for el in list(sig.columns)[:40]
        },
        "contrast_table": contrast.to_dict(orient="records"),
        "top_flag_rate_elements": rates.head(5).to_dict(orient="records"),
        "candidates": (
            cand[cand["vet_pass"]].head(50).to_dict(orient="records") if n_sparse else []
        ),
        "verdict": None,
    }


# ---------------------------------------------------------------------------
# Stage 4 -- co-natal pairs
# ---------------------------------------------------------------------------
def stage_twins(cfg: Config, *, out_dir: Path, block: dict) -> dict:
    from .acquire import join_pairs

    wb_path = out_dir / "wide_binaries.parquet"
    star_files = sorted(out_dir.glob("stars_*.parquet"))
    if not wb_path.exists() or not star_files:
        return {"n_pairs": 0,
                "verdict": ("NO_DATA_REACHED: the wide-binary catalogue or the survey tables "
                            "were not available, so the co-natal stage did not run")}

    pairs = pd.read_parquet(wb_path)
    stars = pd.concat([pd.read_parquet(p) for p in star_files], ignore_index=True)
    elements = [e for e in T.T_COND if e in stars.columns]
    joined = join_pairs(pairs, stars, elements)
    if joined.empty:
        return {"n_pairs": 0,
                "verdict": ("NO_PAIRS_WITH_TWO_SPECTRA: no wide binary had both components "
                            "in the spectroscopic sample")}

    tcfg = _twin_config(block)
    table = T.pair_table(joined, elements, cfg=tcfg)
    table.to_csv(out_dir / "twin_pairs.csv", index=False)
    counts = table["verdict"].value_counts().to_dict()
    unexplainable = table[table["verdict"].isin([T.SPARSE_UNEXPLAINABLE, T.ENGULFMENT_EXCESSIVE])]
    return {
        "n_pairs": int(len(table)),
        "verdict_counts": {k: int(v) for k, v in counts.items()},
        "n_unexplainable": int(len(unexplainable)),
        "unexplainable": unexplainable.head(50).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _overall_verdict(per_survey: list[dict], twins: dict, prov: dict | None) -> str:
    reached = [s for s in per_survey if s.get("n_stars", 0) > 0]
    if not reached:
        return ("NO_DATA_REACHED: no survey catalogue answered, so nothing was searched. "
                "This is an archive-access statement, not a limit on the signature.")
    n_vet = sum(int(s.get("n_vetted", 0)) for s in per_survey)
    n_un = int(twins.get("n_unexplainable", 0))
    degraded = [p for p in (prov or {}).get("surveys", []) if p.get("degraded")]
    prefix = ""
    if degraded:
        names = ", ".join(f"{p['survey']}->{p['source_used']}" for p in degraded)
        prefix = f"DEGRADED_SOURCE ({names}); "
    if n_vet == 0 and n_un == 0:
        return (prefix + "NO_SPARSE_SURVIVOR: every star crossing the amplitude threshold "
                "was rejected as a dense (family-wide) anomaly, a pipeline systematic, or a "
                "binary. Per CLAUDE.md this is a reason to change the question -- more "
                "elements, a second survey, or the differential co-natal channel -- not a "
                "result to write up.")
    return (prefix + f"SPARSE_CANDIDATES_PENDING_REMEASUREMENT: {n_vet} catalogue-level "
            f"sparse survivors and {n_un} co-natal pairs beyond the engulfment budget. "
            "None is a detection until the specific line is re-measured from the raw "
            "spectrum against Teff-matched peers.")


def write_report(cfg: Config, out_dir: Path, summary: dict) -> Path:
    lines: list[str] = []
    lines.append("# TAILINGS — sparse chemical anomaly search\n")
    lines.append(f"**Verdict.** {summary['verdict']}\n")
    lines.append("## The discriminant\n")
    lines.append(
        "Natural abundance space is low-dimensional: ~8-10 independent chemical "
        "dimensions in the solar neighbourhood, and every natural process moves an "
        "element *family*. Industrial refining moves one element. So the statistic is "
        "sparsity, not amplitude: one or two elements extreme with the rest inside "
        "2 sigma. A star with many elements discrepant is a **rejection** here.\n"
    )
    for s in summary.get("per_survey", []):
        lines.append(f"## {s['survey']}\n")
        lines.append(f"- stars after quality cuts: **{s.get('n_stars', 0):,}**")
        lines.append(f"- elements on the manifold: **{s.get('n_elements', 0)}**")
        cc = s.get("class_counts", {})
        if cc:
            lines.append("- classification: " + ", ".join(f"{k} {v:,}" for k, v in cc.items()))
        lines.append(f"- sparse candidates: **{s.get('n_sparse', 0)}**, "
                     f"surviving vetting: **{s.get('n_vetted', 0)}**")
        ct = s.get("contrast_table") or []
        if ct:
            lines.append("\n### Sparse/dense contrast (the headline diagnostic)\n")
            lines.append("| z_max | n | sparse | dense | sparse frac | median z_rest_rms |")
            lines.append("|---|---|---|---|---|---|")
            for r in ct:
                hi = "inf" if not np.isfinite(r["z_max_hi"]) else f"{r['z_max_hi']:g}"
                lines.append(
                    f"| {r['z_max_lo']:g}-{hi} | {r['n']:,} | {r['n_sparse']:,} | "
                    f"{r['n_dense']:,} | {r['sparse_frac']:.3f} | "
                    f"{r['median_z_rest_rms']:.2f} |"
                    if np.isfinite(r["sparse_frac"]) else
                    f"| {r['z_max_lo']:g}-{hi} | {r['n']:,} | {r['n_sparse']:,} | "
                    f"{r['n_dense']:,} | - | - |"
                )
        tf = s.get("top_flag_rate_elements") or []
        if tf:
            lines.append("\n### Highest per-element flag rates (systematics check)\n")
            for r in tf:
                lines.append(f"- `{r['element']}` ({r['family']}): "
                             f"{r['flag_rate']:.2%} of {r['n_measured']:,} measurements")
        lines.append("")

    tw = summary.get("twins", {})
    lines.append("## Stage 4 — co-natal wide binaries\n")
    if tw.get("verdict"):
        lines.append(f"{tw['verdict']}\n")
    else:
        lines.append(f"- pairs with two spectra: **{tw.get('n_pairs', 0):,}**")
        for k, v in (tw.get("verdict_counts") or {}).items():
            lines.append(f"- {k}: {v:,}")
        lines.append(f"- beyond any plausible engulfed-planet budget: "
                     f"**{tw.get('n_unexplainable', 0)}**")
    lines.append("")
    lines.append("## What a survivor still has to pass\n")
    lines.append(
        "Nothing here is a detection. A catalogue-level sparse survivor is a *target*, "
        "and the decisive test is to re-measure the specific line from the raw spectrum "
        "against Teff-matched peers observed with the same instrument, so that blends, "
        "telluric residuals and continuum structure common to the temperature slice "
        "cancel. Until that is done the correct description is 'an unexplained "
        "single-element catalogue outlier'.\n"
    )
    lines.append("## No-null rule (CLAUDE.md)\n")
    lines.append(
        "An empty candidate list at these thresholds is a statement about this corpus, "
        "these elements and these thresholds — not a publishable null. The escalation "
        "path is more elements (optical n-capture lines that the H band cannot reach), "
        "a second survey for cross-confirmation, and the differential co-natal channel, "
        "which reaches ~0.01-0.02 dex where the field channel reaches ~0.03-0.05.\n"
    )
    p = out_dir / "REPORT.md"
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def tailings_run(
    cfg: Config | None = None,
    stage: str = "all",
    surveys: str = "GALAH,APOGEE",
    max_rows: int = 400_000,
) -> dict:
    """Run the TAILINGS funnel. Returns the summary dict written to disk."""
    cfg = cfg or load_config()
    out_dir = _out_dir(cfg)
    block = _cfg_block(cfg)
    survey_list = [s.strip().upper() for s in surveys.split(",") if s.strip()]

    prov = None
    if stage in ("acquire", "all"):
        prov = stage_acquire(cfg, surveys=survey_list, max_rows=max_rows, out_dir=out_dir)
        if stage == "acquire":
            return {"stage": stage, "status": "done", "provenance": prov}

    if prov is None and (out_dir / "provenance.json").exists():
        prov = json.loads((out_dir / "provenance.json").read_text())

    per_survey: list[dict] = []
    if stage in ("manifold", "sparse", "vet", "all"):
        for sv in survey_list:
            p = out_dir / f"stars_{sv.lower()}.parquet"
            if not p.exists():
                per_survey.append({
                    "survey": sv, "n_stars": 0, "n_sparse": 0, "n_vetted": 0,
                    "verdict": "NO_DATA_REACHED: no checkpoint for this survey",
                })
                continue
            per_survey.append(reduce_survey(pd.read_parquet(p), survey=sv,
                                            block=block, out_dir=out_dir))

    twins_out: dict = {}
    if stage in ("twins", "all"):
        twins_out = stage_twins(cfg, out_dir=out_dir, block=block)

    summary = {
        "channel": CHANNEL,
        "stage": stage,
        "verdict": _overall_verdict(per_survey, twins_out, prov),
        "provenance": prov,
        "per_survey": per_survey,
        "twins": twins_out,
        "funnel": {
            "n_stars_total": sum(int(s.get("n_stars", 0)) for s in per_survey),
            "n_sparse_total": sum(int(s.get("n_sparse", 0)) for s in per_survey),
            "n_vetted_total": sum(int(s.get("n_vetted", 0)) for s in per_survey),
            "n_pairs": int(twins_out.get("n_pairs", 0)),
            "n_pairs_unexplainable": int(twins_out.get("n_unexplainable", 0)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(cfg, out_dir, summary)
    print("[tailings] " + json.dumps({"verdict": summary["verdict"].split(":")[0],
                                      **summary["funnel"]}))
    return summary


__all__ = ["reduce_survey", "stage_acquire", "stage_twins", "tailings_run", "write_report"]

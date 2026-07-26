"""Runner: MIDDEN radionuclide search end-to-end (stages checkpoint to disk).

Stages (each resumable; ``all`` runs the chain):

verify-lines     NIST ASD verification of the encoded line list (hard gate;
                 always re-runs — the line list is never silently trusted).
targets          Anchors + Renson CP stars + Gaia A5-F2 dwarfs.
acquire          ObsCore discovery + corpus selection + the batchwise
                 download/measure/discard loop ("acquire-analyze").
score            Census z, candidate logic, REPORT.md.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .measure import Z_LINE, line_flag_rates, score_corpus


def _fmt(v, spec=".2f") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    return format(f, spec) if np.isfinite(f) else "-"


def score_and_write(cfg: Config, meas: pd.DataFrame,
                    corpus: pd.DataFrame | None = None) -> dict:
    out_dir = cfg.root / "results" / "midden"
    out_dir.mkdir(parents=True, exist_ok=True)

    meas = meas[meas["role"].isin(["radionuclide", "rv_ref", "control"])].copy()
    stars, meas_z = score_corpus(meas)
    rates = line_flag_rates(meas_z)
    if len(stars):
        stars.to_csv(out_dir / "scores.csv", index=False)
    rates.to_csv(out_dir / "line_flag_rates.csv", index=False)
    meas_z.to_parquet(out_dir / "measurements_scored.parquet", index=False)

    cands = stars[stars["candidate"]] if len(stars) else pd.DataFrame()
    n_spectra = int(meas_z["dp_id"].nunique())
    n_stars = int(meas_z["star"].nunique())

    anchors = []
    if len(stars) and "priority" in meas_z.columns:
        anchor_names = meas_z.loc[meas_z["priority"] == 0, "star"].unique()
        anchors = stars[stars["star"].isin(anchor_names)].to_dict("records")

    payload = {
        "n_spectra": n_spectra, "n_stars": n_stars,
        "n_candidates": int(len(cands)),
        "candidates": cands.to_dict("records") if len(cands) else [],
        "anchors": anchors,
        "line_flag_rates": rates.to_dict("records"),
    }
    (out_dir / "candidates.json").write_text(
        json.dumps(payload, indent=2, default=str))

    zcols = [c for c in (stars.columns if len(stars) else [])
             if c.startswith("z_") and c != "z_control_max"]
    summary = {
        "n_spectra": n_spectra, "n_stars": n_stars,
        "n_candidates": int(len(cands)),
        "n_control_vetoed": int(stars["any_control_veto"].sum()) if len(stars) else 0,
        "n_multi_epoch_stars": int((stars["n_epochs"] >= 2).sum()) if len(stars) else 0,
        "top": (stars.head(10)[["star", "n_epochs", "candidate",
                                "any_control_veto", *zcols]].to_dict("records")
                if len(stars) else []),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))

    lines = [
        "# MIDDEN run report", "",
        "Survey-scale search for short-lived radionuclides (Tc I resonance",
        "triplet; U II / Th II actinides) in ESO Phase-3 HARPS+FEROS spectra —",
        "the Whitmire & Wright (1980) nuclear-waste-disposal technosignature",
        "(docs/midden.md).", "",
        f"Corpus: {n_spectra} spectra of {n_stars} stars "
        f"({summary['n_multi_epoch_stars']} stars with >= 2 epochs).", "",
        f"## Per-line flag rates (census z >= {Z_LINE:g})", "",
        "| line | role | measured | flagged | rate |",
        "|---|---|---|---|---|",
    ]
    for r in rates.itertuples():
        lines.append(f"| {r.species} {r.wavelength:.2f} | {r.role} | "
                     f"{r.n_measured} | {r.n_flagged} | "
                     f"{_fmt(r.flag_rate, '.4f')} |")
    lines += ["", f"## Candidates: {len(cands)}", ""]
    if len(cands):
        for r in cands.to_dict("records"):
            zs = ", ".join(f"{k[2:]}={_fmt(v)}" for k, v in r.items()
                           if k.startswith("z_") and k != "z_control_max")
            lines.append(f"- **{r['star']}** (epochs {r['n_epochs_candidate']}/"
                         f"{r['n_epochs']}, tc_coherent={r['tc_coherent_any']}): {zs}")
    else:
        lines.append("(none at the current thresholds)")
    lines += ["", "## Prior-claim anchors", ""]
    if anchors:
        for r in anchors:
            zs = ", ".join(f"{k[2:]}={_fmt(v)}" for k, v in r.items()
                           if k.startswith("z_") and k != "z_control_max")
            lines.append(f"- {r['star']}: candidate={r['candidate']}, "
                         f"control_veto={r['any_control_veto']}, {zs}")
    else:
        lines.append("(no anchor spectra entered the corpus this run — check "
                     "the ObsCore match and anchor resolution)")
    lines += [
        "",
        "No-null rule (CLAUDE.md): an empty candidate list here is a domain",
        "statement about THIS corpus (HARPS+FEROS Phase-3, these line windows,",
        "these thresholds), never a publishable result. Next moves are the",
        "UVES/ESPRESSO collections, the Pm II line set, and epoch-resolved",
        "decay-curve tests on any near-threshold star.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[midden] scored {n_stars} stars / {n_spectra} spectra -> "
          f"{summary['n_candidates']} candidates")
    return summary


def midden_run(cfg: Config | None = None, stage: str = "all",
               max_spectra: int = 3000, batch_size: int = 50,
               scratch_dir: str | Path | None = None) -> dict:
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "midden"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = {"acquire-analyze": "acquire"}.get(stage, stage)

    if stage in ("verify-lines", "all"):
        from .lines import verify_against_nist

        verify_against_nist(out_dir / "line_verification.json")
        if stage == "verify-lines":
            return {"stage": stage, "status": "verified"}

    from .acquire import build_targets, process_corpus, query_obscore, select_corpus

    targets = corpus = meas = None
    if stage in ("targets", "all"):
        targets = build_targets(out_dir)
    if stage in ("acquire", "all"):
        targets = targets if targets is not None else \
            pd.read_parquet(out_dir / "targets.parquet")
        obs = query_obscore(targets, out_dir)
        corpus_path = out_dir / "corpus.parquet"
        if corpus_path.exists():
            corpus = pd.read_parquet(corpus_path)
            print(f"[midden] corpus checkpoint exists: {corpus_path}")
        else:
            corpus = select_corpus(obs, targets, max_spectra=max_spectra)
            corpus.to_parquet(corpus_path, index=False)
        scratch = Path(scratch_dir) if scratch_dir else \
            Path(tempfile.gettempdir()) / "midden_scratch"
        meas = process_corpus(corpus, out_dir / "meas", scratch,
                              batch_size=batch_size)
        meas.to_parquet(out_dir / "measurements.parquet", index=False)
    if stage in ("score", "all"):
        meas = meas if meas is not None else \
            pd.read_parquet(out_dir / "measurements.parquet")
        corpus_path = out_dir / "corpus.parquet"
        corpus = corpus if corpus is not None else (
            pd.read_parquet(corpus_path) if corpus_path.exists() else None)
        return score_and_write(cfg, meas, corpus)
    return {"stage": stage, "status": "done"}


__all__ = ["midden_run", "score_and_write"]

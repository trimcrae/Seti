"""Command-line entry point: ``seti <command>``.

Commands
--------
make-sample   Generate the committed synthetic offline sample.
analyze       Run the analysis funnel on a table (default: the sample).
completeness  Build the injection-recovery completeness map.
figures       Render the manuscript figures from analysis outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .pipeline import run_pipeline
from .sample import make_sample


def _cmd_make_sample(args, cfg):
    df = make_sample(seed=args.seed)
    out = cfg.path("sample_dir") / "wd_sample.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")


def _load_input(args, cfg) -> pd.DataFrame:
    path = Path(args.input) if args.input else cfg.path("sample_dir") / "wd_sample.parquet"
    if not path.exists():
        raise SystemExit(f"input not found: {path}; run `seti make-sample` first")
    return pd.read_parquet(path)


def _cmd_analyze(args, cfg):
    df = _load_input(args, cfg)
    out_dir = Path(args.out) if args.out else cfg.path("tables_dir")
    result = run_pipeline(df, cfg=cfg, out_dir=out_dir)
    print(json.dumps({"counts": result.counts,
                      "funnel_counts": result.funnel_counts,
                      "occurrence_limit": result.occurrence_limit}, indent=2))


def _cmd_completeness(args, cfg):
    from .stats.completeness import completeness_map

    df = _load_input(args, cfg)
    clean = df[df.get("label", "clean") == "clean"] if "label" in df else df
    cmap = completeness_map(clean, cfg.thresholds)
    out = cfg.path("tables_dir") / "completeness.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmap.to_parquet(out, index=False)
    print(f"wrote completeness map ({len(cmap)} cells) -> {out}")


def _cmd_forecast(args, cfg):
    from .stats.sensitivity import forecast_sensitivity, headline_limit, minimum_detectable_tau

    fc = forecast_sensitivity(cfg, seed=args.seed)
    out_dir = cfg.path("tables_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out_dir / "forecast.parquet", index=False)
    minimum_detectable_tau(fc).to_parquet(out_dir / "min_detectable_tau.parquet", index=False)
    h = headline_limit(fc)
    (out_dir / "forecast_summary.json").write_text(json.dumps(h, indent=2))
    print(json.dumps({"headline_limit": h,
                      "n_cells": len(fc),
                      "n_detected_real": float(fc["n_detected_real"].iloc[0])}, indent=2))


def _cmd_acquire_run(args, cfg):
    from .acquire_run import acquire_run

    table = acquire_run(cfg, max_dist_pc=args.max_dist_pc, limit=args.limit,
                        dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run OK: wiring valid, schema has {len(table.columns)} columns")
    else:
        out = cfg.path("processed_dir") / "analysis_ready.parquet"
        print(f"wrote analysis-ready table: {len(table)} white dwarfs -> {out}")


def _cmd_science_run(args, cfg):
    from .acquire_run import science_run

    science_run(cfg, max_dist_pc=args.max_dist_pc, limit=args.limit)


def _cmd_contamination_budget(args, cfg):
    from .population import generate_population
    from .stats.contamination_budget import contamination_budget, efficacy_vs_pm

    pop = generate_population(cfg, seed=args.seed)
    budget = contamination_budget(cfg, pop)
    eff = efficacy_vs_pm(cfg, pop)
    out_dir = cfg.path("tables_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    eff.to_parquet(out_dir / "comovement_efficacy.parquet", index=False)
    (out_dir / "contamination_budget.json").write_text(json.dumps(budget, indent=2))
    print(json.dumps(budget, indent=2))


def _cmd_spectra_run(args, cfg):
    from .spectra.run import spectra_run

    spectra_run(cfg, n=args.n, dataset=args.dataset,
                spectype=args.spectype, snr_min=args.snr_min)


def _cmd_paper_numbers(args, cfg):
    from .report import write_numbers_tex

    out = write_numbers_tex(cfg)
    print(f"wrote {out}")


def _cmd_laser_numbers(args, cfg):
    from .report import write_laser_numbers_tex

    out = write_laser_numbers_tex(cfg)
    print(f"wrote {out}")


def _cmd_figures(args, cfg):
    from .figures import render_all

    paths = render_all(cfg)
    for p in paths:
        print(f"wrote {p}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="seti", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-sample")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_cmd_make_sample)

    p = sub.add_parser("analyze")
    p.add_argument("--input")
    p.add_argument("--out")
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("completeness")
    p.add_argument("--input")
    p.set_defaults(func=_cmd_completeness)

    p = sub.add_parser("forecast")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=_cmd_forecast)

    p = sub.add_parser("acquire-run")
    p.add_argument("--max-dist-pc", type=float, default=100.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_acquire_run)

    p = sub.add_parser("science-run")
    p.add_argument("--max-dist-pc", type=float, default=100.0)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_cmd_science_run)

    p = sub.add_parser("spectra-run")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--dataset", default="DESI-EDR")
    p.add_argument("--spectype", default=None)
    p.add_argument("--snr-min", type=float, default=8.0)
    p.set_defaults(func=_cmd_spectra_run)

    p = sub.add_parser("contamination-budget")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=_cmd_contamination_budget)

    p = sub.add_parser("paper-numbers")
    p.set_defaults(func=_cmd_paper_numbers)

    p = sub.add_parser("laser-numbers")
    p.set_defaults(func=_cmd_laser_numbers)

    p = sub.add_parser("figures")
    p.set_defaults(func=_cmd_figures)

    args = parser.parse_args(argv)
    cfg = load_config()
    return args.func(args, cfg)


if __name__ == "__main__":
    main()

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

    p = sub.add_parser("figures")
    p.set_defaults(func=_cmd_figures)

    args = parser.parse_args(argv)
    cfg = load_config()
    return args.func(args, cfg)


if __name__ == "__main__":
    main()

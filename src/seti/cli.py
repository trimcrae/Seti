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

import numpy as np
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


def _cmd_dimming_run(args, cfg):
    from .dimming.run import dimming_run

    dimming_run(cfg, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
                g_min=args.g_min, g_max=args.g_max,
                variable_only=not args.all_stars, band=args.band,
                limit=args.limit, time_budget_s=args.time_budget_s,
                mode=args.mode, box_deg=args.box_deg)


def _cmd_dimming_vet(args, cfg):
    import glob

    from .dimming.vet import vet_candidates

    # Aggregate two candidate classes across all searched fields: the
    # resists-mundane dippers and the secular faders (the artifact-robust class).
    frames = []
    for fp in sorted(glob.glob(str(cfg.root / "results" / "dimming" / "*" /
                                   "dimming_candidates.csv"))):
        df = pd.read_csv(fp)
        if "resists_mundane" in df.columns:
            df = df[df["resists_mundane"].astype(str).str.lower().isin(("true", "1"))]
        if len(df):
            df["field_dir"] = Path(fp).parent.name
            df["cand_type"] = "dipper"
            frames.append(df)
    for fp in sorted(glob.glob(str(cfg.root / "results" / "dimming" / "*" /
                                   "secular_faders.csv"))):
        df = pd.read_csv(fp)
        # Only main-sequence faders: the faint hr=unknown population is a ZTF
        # magnitude-dependent systematic (older field CSVs may still contain it).
        if "hr_class" in df.columns:
            df = df[df["hr_class"] == "main_sequence"]
        if len(df):
            df["field_dir"] = Path(fp).parent.name
            df["cand_type"] = "secular_fader"
            frames.append(df)
    if not frames:
        print("[dimming-vet] no candidates found")
        return
    cand = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    print(f"[dimming-vet] vetting {len(cand)} resists-mundane candidates")
    vetted = vet_candidates(cand)

    # Multi-band achromaticity: a real occultation dims g, r and i together; a
    # single-band excursion is a photometric artefact or a blend.  Run it on the
    # candidates that survived the IR/SIMBAD cut (no point characterising dusty/
    # known ones).  frac_confirmed = fraction of reference-band dips coincident in
    # another band.
    from .dimming.vet import multiband_coincidence, secular_achromatic
    fracs, nbands, dpb, secconf = [], [], [], []
    for _, r in vetted.iterrows():
        ctype = r.get("cand_type", "dipper")
        mb, sc = {}, {}
        if r.get("ir_verdict") in ("clean", "no_ir_data"):
            try:
                if ctype == "dipper":
                    mb = multiband_coincidence(float(r["ra"]), float(r["dec"]))
                else:   # secular fader: confirm the fade is achromatic (g and r)
                    sc = secular_achromatic(float(r["ra"]), float(r["dec"]))
            except Exception as exc:
                print(f"[dimming-vet] band check failed for {r['source_id']}: {exc!r}")
        fracs.append(mb.get("frac_confirmed", float("nan")))
        nbands.append(mb.get("n_bands", 0))
        dpb.append(str(mb.get("dips_per_band", {})))
        secconf.append(sc.get("secular_confirmed", False))
    vetted["frac_confirmed"] = fracs
    vetted["n_bands"] = nbands
    vetted["dips_per_band"] = dpb
    vetted["secular_confirmed"] = secconf
    # Final verdict: a clean candidate whose dips are confirmed achromatic in
    # >=2 bands is the genuinely interesting regime; clean but single-band is an
    # artefact.
    def _final(r):
        if r["ir_verdict"] != "clean":
            return r["ir_verdict"]
        # Secular faders: a monotonic multi-year fade with no IR excess is the
        # remarkable enshrouding case ONLY if the fade is achromatic (present in
        # both g and r); a single-band slow drift is an instrumental/blend artifact.
        if r.get("cand_type", "dipper") == "secular_fader":
            try:
                amp = abs(float(r.get("secular_total_mag", 0) or 0))
            except (TypeError, ValueError):
                amp = 0.0
            if amp < 0.08:        # a few-percent fade is marginal, not noteworthy
                return "marginal_fade"
            return ("clean_secular_fade" if r.get("secular_confirmed")
                    else "single_band_fade")
        f = r["frac_confirmed"]
        if r["n_bands"] < 2 or not np.isfinite(f):
            return "single_band_unconfirmed"
        return "clean_achromatic" if f >= 0.5 else "single_band_artifact"
    vetted["verdict"] = [_final(r) for _, r in vetted.iterrows()]

    out_dir = cfg.root / "results" / "dimming"
    cols = [c for c in ("source_id", "field_dir", "cand_type", "ra", "dec", "score",
                        "max_event_depth", "n_dip_events", "asymmetry",
                        "period_power", "secular_sigma", "secular_total_mag",
                        "hr_class", "W1_W2", "K_W2",
                        "simbad_otype", "ir_verdict", "frac_confirmed", "n_bands",
                        "dips_per_band", "secular_confirmed", "verdict")
            if c in vetted.columns]
    vetted[cols].to_csv(out_dir / "vetting.csv", index=False)
    print(vetted[cols].to_string(index=False))
    gold = vetted[vetted["verdict"].isin(("clean_achromatic", "clean_secular_fade"))]
    print(f"[dimming-vet] {len(gold)} GOLD (clean_achromatic dippers + "
          f"clean_secular_fade) of {len(vetted)} vetted")


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

    p = sub.add_parser("dimming-run")
    p.add_argument("--ra", type=float, default=270.0)
    p.add_argument("--dec", type=float, default=30.0)
    p.add_argument("--radius-deg", type=float, default=1.5)
    p.add_argument("--g-min", type=float, default=13.0)
    p.add_argument("--g-max", type=float, default=18.5)
    p.add_argument("--all-stars", action="store_true",
                   help="search all stars, not only Gaia-flagged variables")
    p.add_argument("--band", default="r")
    p.add_argument("--limit", type=int, default=4000)
    p.add_argument("--time-budget-s", type=float, default=1800.0)
    p.add_argument("--mode", choices=["targets", "region"], default="targets",
                   help="'targets': Gaia stars + per-object ZTF; "
                        "'region': bulk box-sweep of every ZTF source (10-100x more)")
    p.add_argument("--box-deg", type=float, default=0.12,
                   help="box size for region-mode bulk fetch (deg)")
    p.set_defaults(func=_cmd_dimming_run)

    p = sub.add_parser("dimming-vet")
    p.set_defaults(func=_cmd_dimming_vet)

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

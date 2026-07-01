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
                spectype=args.spectype, snr_min=args.snr_min,
                mode=args.mode)


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
    for fp in sorted(glob.glob(str(cfg.root / "results" / "dimming" / "*" /
                                   "glint_candidates.csv"))):
        df = pd.read_csv(fp)
        if "hr_class" in df.columns:
            df = df[df["hr_class"] == "main_sequence"]
        if len(df):
            df["field_dir"] = Path(fp).parent.name
            df["cand_type"] = "glint"
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
    from .dimming.vet import glint_achromatic, multiband_coincidence, secular_achromatic
    fracs, nbands, dpb, secconf, glconf = [], [], [], [], []
    for _, r in vetted.iterrows():
        ctype = r.get("cand_type", "dipper")
        mb, sc, gl = {}, {}, {}
        if r.get("ir_verdict") in ("clean", "no_ir_data"):
            try:
                if ctype == "dipper":
                    mb = multiband_coincidence(float(r["ra"]), float(r["dec"]))
                elif ctype == "secular_fader":
                    sc = secular_achromatic(float(r["ra"]), float(r["dec"]))
                elif ctype == "glint":   # confirm the flash is achromatic (g==r)
                    gl = glint_achromatic(float(r["ra"]), float(r["dec"]))
            except Exception as exc:
                print(f"[dimming-vet] band check failed for {r['source_id']}: {exc!r}")
        fracs.append(mb.get("frac_confirmed", float("nan")))
        nbands.append(mb.get("n_bands", 0))
        dpb.append(str(mb.get("dips_per_band", {})))
        secconf.append(sc.get("secular_confirmed", False))
        glconf.append(gl.get("glint_confirmed", False))
    vetted["frac_confirmed"] = fracs
    vetted["n_bands"] = nbands
    vetted["dips_per_band"] = dpb
    vetted["secular_confirmed"] = secconf
    vetted["glint_confirmed"] = glconf
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
            # Cool active dwarfs (BP-RP > 0.9) fade via starspot/activity cycles --
            # mundane.  A noteworthy enshrouding fade is on a hot, inactive F/G star.
            try:
                bp_rp = float(r.get("bp_rp", "nan"))
            except (TypeError, ValueError):
                bp_rp = float("nan")
            if np.isfinite(bp_rp) and bp_rp > 0.9:
                return "active_dwarf_fade"
            return ("clean_secular_fade" if r.get("secular_confirmed")
                    else "single_band_fade")
        # Glints: a specular flash is achromatic (g and r brighten equally); a
        # blue/chromatic brightening is a stellar flare.
        if r.get("cand_type", "dipper") == "glint":
            return "clean_glint" if r.get("glint_confirmed") else "chromatic_flare"
        f = r["frac_confirmed"]
        if r["n_bands"] < 2 or not np.isfinite(f):
            return "single_band_unconfirmed"
        return "clean_achromatic" if f >= 0.5 else "single_band_artifact"
    vetted["verdict"] = [_final(r) for _, r in vetted.iterrows()]

    out_dir = cfg.root / "results" / "dimming"
    cols = [c for c in ("source_id", "field_dir", "cand_type", "ra", "dec", "score",
                        "max_event_depth", "n_dip_events", "asymmetry",
                        "period_power", "secular_sigma", "secular_total_mag", "bp_rp",
                        "hr_class", "W1_W2", "K_W2",
                        "simbad_otype", "ir_verdict", "frac_confirmed", "n_bands",
                        "dips_per_band", "secular_confirmed",
                        "glint_max_brighten", "glint_confirmed", "verdict")
            if c in vetted.columns]
    vetted[cols].to_csv(out_dir / "vetting.csv", index=False)
    print(vetted[cols].to_string(index=False))
    gold_verdicts = ("clean_achromatic", "clean_secular_fade", "clean_glint")
    gold = vetted[vetted["verdict"].isin(gold_verdicts)]
    print(f"[dimming-vet] {len(gold)} GOLD (clean_achromatic + clean_secular_fade "
          f"+ clean_glint) of {len(vetted)} vetted")


def _cmd_xp_run(args, cfg):
    from .xp.run import xp_run

    xp_run(cfg, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
           g_max=args.g_max, limit=args.limit,
           global_sigma_min=args.global_sigma_min,
           feature_resid_min=args.feature_resid_min)


def _cmd_dimming_characterize(args, cfg):
    from .dimming.characterize import characterize

    res = characterize(args.ra, args.dec,
                       optical_slope_mag_yr=args.optical_slope)
    out_dir = cfg.root / "results" / "dimming"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "characterization.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def _cmd_spectra_confirm(args, cfg):
    from .spectra.confirm import cross_confirm

    # Prefer the triaged shortlist (observed-frame known-line + recurrence cuts
    # already applied); fall back to the raw candidate list.
    triaged = cfg.root / "results" / "spectra_triage" / "priority_targets.csv"
    path = cfg.root / "results" / "spectra" / "laser_candidates.csv"
    if triaged.exists():
        df = pd.read_csv(triaged)
        df = df.sort_values("significance", ascending=False)
    elif path.exists():
        df = pd.read_csv(path)
        if "hunt_rank" in df.columns:
            df = df.sort_values("hunt_rank", ascending=False)
    else:
        print(f"[confirm] no candidates at {path}")
        return
    # Prefer the cleanest beacons: single line in the spectrum.
    if "n_lines_in_spectrum" in df.columns:
        df = df[df["n_lines_in_spectrum"] == 1]
    cands = df.head(args.top).to_dict("records")
    confirmed = cross_confirm(cands, max_candidates=args.top)
    out = pd.DataFrame(confirmed)
    keep = [c for c in ("spec_id", "wavelength", "significance", "width_ratio",
                        "ra", "dec", "data_release", "n_overlap", "confirm_sigma",
                        "cross_confirmed") if c in out.columns]
    dst = cfg.root / "results" / "spectra" / "cross_confirm.csv"
    out[keep].to_csv(dst, index=False)
    n_overlap = int((out["n_overlap"] > 0).sum()) if "n_overlap" in out else 0
    n_conf = int(out["cross_confirmed"].sum()) if "cross_confirmed" in out else 0
    print(out[keep].to_string(index=False))
    print(f"[confirm] {n_overlap}/{len(out)} had an independent spectrum; "
          f"{n_conf} CROSS-CONFIRMED (line present in a second instrument)")


def _cmd_spectra_triage(args, cfg):
    from .spectra.triage import triage_run

    triage_run(cfg.root, v_window_kms=args.v_window,
               recur_tol=args.recur_tol, recur_min=args.recur_min)


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
    p.add_argument("--mode", choices=["emission", "absorption"],
                   default="emission",
                   help="emission=laser lines; absorption=anomalous narrow absorbers")
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

    p = sub.add_parser("xp-run")
    p.add_argument("--ra", type=float, default=180.0)
    p.add_argument("--dec", type=float, default=30.0)
    p.add_argument("--radius-deg", type=float, default=1.0)
    p.add_argument("--g-max", type=float, default=17.5)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--global-sigma-min", type=float, default=8.0)
    p.add_argument("--feature-resid-min", type=float, default=6.0)
    p.set_defaults(func=_cmd_xp_run)

    p = sub.add_parser("dimming-characterize")
    p.add_argument("--ra", type=float, required=True)
    p.add_argument("--dec", type=float, required=True)
    p.add_argument("--optical-slope", type=float, default=None,
                   help="known optical fade rate (mag/yr) if ASAS-SN is down")
    p.set_defaults(func=_cmd_dimming_characterize)

    p = sub.add_parser("spectra-confirm")
    p.add_argument("--top", type=int, default=40)
    p.set_defaults(func=_cmd_spectra_confirm)

    p = sub.add_parser("spectra-triage")
    p.add_argument("--v-window", type=float, default=300.0)
    p.add_argument("--recur-tol", type=float, default=3.0)
    p.add_argument("--recur-min", type=int, default=3)
    p.set_defaults(func=_cmd_spectra_triage)

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

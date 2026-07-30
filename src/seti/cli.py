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


def _cmd_accel_run(args, cfg):
    from .accel.run import accel_run

    accel_run(cfg, limit=args.limit, plx_min=args.plx_min, sig_min=args.sig_min)


def _cmd_accel_xmatch(args, cfg):
    from .accel.crossmatch import run_crossmatch

    cand = args.candidates or str(cfg.root / "results" / "accel"
                                  / "class3_shortlist.csv")
    out_dir = cfg.root / "results" / "accel"
    run_crossmatch(cand, out_dir)


def _cmd_cluster_run(args, cfg):
    from .cluster.run import cluster_run

    cluster_run(cfg, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
                plx_min=args.plx_min, g_max=args.g_max, limit=args.limit,
                excess_z_min=args.excess_z_min, link_pc=args.link_pc)


def _cmd_tidemark_run(args, cfg):
    from .tidemark.run import tidemark_run

    tidemark_run(cfg, channels=args.channels, quick=args.quick, seed=args.seed,
                 do_calibrate=not args.no_calibrate)


def _cmd_tidemark_search(args, cfg):
    from .tidemark.run import tidemark_selfsearch

    tidemark_selfsearch(cfg, grid=args.grid, radius_deg=args.radius_deg,
                        plx_min=args.plx_min, g_max=args.g_max,
                        excess_z_min=args.excess_z_min, limit=args.limit,
                        parent_glob=args.from_parent,
                        quick=args.quick, seed=args.seed)


def _cmd_tidemark_acquire(args, cfg):
    """Fetch the wide-area parent sample only, and checkpoint it to disk."""
    from pathlib import Path

    from .tidemark.acquire import excess_axis, parent_sample

    out = Path(cfg.root) / "results" / "tidemark" / "parent"
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.n_shards <= 1 else f"_shard{args.shard:02d}"
    tbl = parent_sample(grid=args.grid, radius_deg=args.radius_deg,
                        plx_min=args.plx_min, g_max=args.g_max,
                        stride=args.stride, limit=args.limit,
                        shard=args.shard, n_shards=args.n_shards)
    if tbl is None or not len(tbl):
        (out / f"acquire_status{tag}.json").write_text(
            '{"verdict": "NO_DATA_REACHED", "n_rows": 0}')
        print("[tidemark] NO_DATA_REACHED: archive returned nothing")
        return
    # NOTE: the excess axis is deliberately NOT computed here when sharding ---
    # the W1-W2 locus must be fitted once across the whole sky (see
    # tidemark.acquire), so a shard writes raw photometry and the reduce stage
    # fits the locus globally.
    if args.n_shards <= 1:
        tbl = excess_axis(tbl)
    tbl.to_parquet(out / f"parent_sample{tag}.parquet", index=False)
    import json
    (out / f"acquire_status{tag}.json").write_text(json.dumps({
        "verdict": "OK", "n_rows": int(len(tbl)),
        "n_cones": int(tbl["cone"].nunique()) if "cone" in tbl.columns else None,
        "grid": args.grid, "radius_deg": args.radius_deg, "stride": args.stride,
        "g_max": args.g_max, "plx_min": args.plx_min,
        "shard": args.shard, "n_shards": args.n_shards,
        "excess_axis_fitted": bool(args.n_shards <= 1),
    }, indent=2))
    print(f"[tidemark] parent sample shard {args.shard}: {len(tbl)} stars written")


def _cmd_panspermia_run(args, cfg):
    from .panspermia.run import panspermia_run

    panspermia_run(cfg, source_id=args.source_id, search_pc=args.search_pc,
                   g_max=args.g_max, limit=args.limit, t_max_myr=args.t_max_myr,
                   d_min_max_pc=args.d_min_max_pc)


def _cmd_cluster_aggregate(args, cfg):
    from .cluster.aggregate import aggregate_sweep

    agg = aggregate_sweep(cfg.root)
    print(json.dumps({"n_cones": agg["n_cones"], "total_stars": agg["total_stars"],
                      "detection": agg["detection"],
                      "p_phase": agg.get("p_phase")}, indent=2, default=str))


def _cmd_panspermia_mc(args, cfg):
    from .panspermia.uncertainty import run_mc_followup

    run_mc_followup(cfg, n=args.n)


def _cmd_panspermia_targets(args, cfg):
    from .panspermia.run import targets_run

    targets_run(cfg, target=args.target, crossmatch=args.crossmatch,
                max_pc=args.max_pc)


def _cmd_panspermia_dossier(args, cfg):
    from .panspermia.run import dossier_run

    dossier_run(cfg)


def _cmd_lhs1140(args, cfg):
    from .lhs1140.run import lhs1140_run

    lhs1140_run(cfg, sphere_pc=args.sphere_pc)


def _cmd_galactic(args, cfg):
    from .galactic.run import galactic_run

    galactic_run(cfg, t_max_myr=args.t_max_myr, search_pc=args.search_pc,
                 d_cut_pc=args.d_cut_pc, limit=args.limit)


def _cmd_jwst_bio(args, cfg):
    from .jwst_bio.run import jwst_bio_run

    jwst_bio_run(cfg)


def _herdsman_params(args):
    from .herdsman.convergence import ConvergenceParams

    return ConvergenceParams(
        t_max_myr=args.t_max_myr, dt_myr=args.dt_myr, rec_every=args.rec_every,
        r0_pc=args.r0_pc, kappa=args.kappa, lambda_cap=args.lambda_cap,
        n_min=args.n_min, r_now_min_pc=args.r_now_min_pc,
        focus_min=args.focus_min, surprise_min=args.surprise_min,
        sigv_int_max_kms=args.sigv_int_max, min_epochs=args.min_epochs)


def _cmd_herdsman_fetch(args, cfg):
    from .herdsman.stages import fetch_stage

    fetch_stage(cfg, d_max_pc=args.d_max_pc, g_max=args.g_max,
                rv_err_max_kms=args.rv_err_max, sigv_max_kms=args.sigv_max,
                astro_floor_kms=args.astro_floor)


def _cmd_herdsman_scan(args, cfg):
    from .herdsman.stages import scan_stage

    scan_stage(cfg, mode=args.mode, shard=args.shard,
               mocks_per_shard=args.mocks_per_shard,
               mock_cell_pc=args.mock_cell_pc, params=_herdsman_params(args))


def _cmd_herdsman_reduce(args, cfg):
    from .herdsman.stages import reduce_stage

    reduce_stage(cfg, n_mocks_expected=args.n_mocks_expected,
                 astro_floor_kms=args.astro_floor)


def _cmd_herdsman_b(args, cfg):
    from .herdsman_b.run import herdsman_b_run

    herdsman_b_run(cfg, stage=args.stage)


def _cmd_ember(args, cfg):
    from .ember.run import ember_run

    summary = ember_run(cfg, stage=args.stage, n_ra_chunks=args.n_ra_chunks,
                        shard=args.shard, n_shards=args.n_shards,
                        require_all_checks=args.require_all_checks == "true")
    keys = ("verdict", "counts", "pair_audit", "acquisition_failure")
    print(json.dumps({k: v for k, v in summary.items() if k in keys}, indent=2,
                     default=str))
    acq = summary.get("acquisition") or {}
    if acq.get("per_archive"):
        print("=== per-archive acquisition status ===")
        print(json.dumps(acq["per_archive"], indent=2, default=str))


def _cmd_midden(args, cfg):
    from .midden.run import midden_run

    midden_run(cfg, stage=args.stage, max_spectra=args.max_spectra,
               batch_size=args.batch_size)


def _cmd_compass(args, cfg):
    from .compass.run import compass_run

    compass_run(cfg, stage=args.stage,
                radii_pc=tuple(float(r) for r in args.radii.split(",")),
                n_min=args.n_min, n_shuffles=args.n_shuffles,
                band_deg=args.band_deg, sig_min=args.sig_min,
                poe_min=args.poe_min, d_max_pc=args.d_max_pc)


def _cmd_midden_deep(args, cfg):
    from .midden.deepdive import deepdive_run

    deepdive_run(cfg, stage=args.stage, epochs_per_star=args.epochs_per_star,
                 target_epochs=args.target_epochs,
                 radius_arcsec=args.radius_arcsec,
                 batch_size=args.batch_size)


def _cmd_tailings(args, cfg):
    from .tailings.run import tailings_run

    tailings_run(cfg, stage=args.stage, surveys=args.surveys,
                 max_rows=args.max_rows)


def _cmd_tailings_validate(args, cfg):
    """Offline injection-recovery against the channel's published validation target."""
    from .tailings.validate import validate_griffith

    report = validate_griffith(seed=args.seed, n_field=args.n_field,
                               run_vet=not args.no_vet)
    out_dir = cfg.root / "results" / "tailings"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "validation.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[tailings-validate] wrote {path}")
    print("[tailings-validate] " + report["verdict"])
    rec = report["recovery"]
    print(f"[tailings-validate] recovered {rec['n_recovered']}/{rec['n_injected']}; "
          f"misses: {json.dumps(rec['miss_breakdown'])}")
    for alt in report.get("alternative_thresholds", []):
        print(f"[tailings-validate] z_flag={alt['z_flag']} "
              f"max_quiet_excess={alt['max_quiet_excess']}: "
              f"{alt['n_recovered']}/{alt['n_injected']} recovered, "
              f"{alt['n_false_positive']} false positives")


def _cmd_isotherm(args, cfg):
    from .isotherm.run import isotherm_run

    isotherm_run(cfg, stage=args.stage, max_spectra=args.max_spectra,
                 resolution=args.resolution, shard=args.shard,
                 n_shards=args.n_shards)


def _cmd_herdsman(args, cfg):
    from .herdsman.run import herdsman_run

    herdsman_run(cfg, d_max_pc=args.d_max_pc, g_max=args.g_max,
                 rv_err_max_kms=args.rv_err_max, sigv_max_kms=args.sigv_max,
                 astro_floor_kms=args.astro_floor, t_max_myr=args.t_max_myr,
                 dt_myr=args.dt_myr, rec_every=args.rec_every,
                 r0_pc=args.r0_pc, kappa=args.kappa,
                 lambda_cap=args.lambda_cap, n_min=args.n_min,
                 r_now_min_pc=args.r_now_min_pc, focus_min=args.focus_min,
                 surprise_min=args.surprise_min, n_mocks=args.n_mocks,
                 mock_cell_pc=args.mock_cell_pc)


def _cmd_lhs1140_origin(args, cfg):
    from .lhs1140_origin.run import lhs1140_origin_run

    lhs1140_origin_run(cfg, search_pc=args.search_pc, t_max_myr=args.t_max_myr,
                       d_min_max_pc=args.d_min_max_pc, crossmatch=args.crossmatch,
                       max_pc=args.max_pc)


def _cmd_shroud(args, cfg):
    from .shroud.run import shroud_run

    shroud_run(cfg, stage=args.stage, allow_network=not args.offline,
               max_sources=args.max_sources, input_parquet=args.input)


def _cmd_crosscorr(args, cfg):
    from .crosscorr.run import crosscorr_run

    crosscorr_run(cfg)


def _cmd_seti_archive(args, cfg):
    from .seti_archive.run import seti_archive_run

    seti_archive_run(cfg, snr=args.snr)


def _cmd_iso_backtrack(args, cfg):
    from .iso.run import iso_run

    iso_run(cfg, t_max_myr=args.t_max_myr, n_mc=args.n_mc,
            nearby_pc=args.nearby_pc, d_close_pc=args.d_close_pc,
            scan_nearby=not args.no_scan)


def _cmd_derelict(args, cfg):
    from .derelict.run import derelict_run

    summary = derelict_run(cfg, stage=args.stage, limit=args.limit,
                           offline_input=args.offline_input, max_vet=args.max_vet,
                           max_enrich=args.max_enrich,
                           max_control_enrich=args.max_control_enrich,
                           skip_control=args.skip_control,
                           skip_completeness=args.skip_completeness,
                           skip_dark_comets=args.skip_dark_comets,
                           skip_high_albedo=args.skip_high_albedo,
                           completeness_limit=args.completeness_limit)
    # The query-status counts are printed with the verdict on purpose: a reader
    # must be able to tell "the archive answered and the answer was empty" from
    # "the archive was never reached" without opening a file.
    print(json.dumps({"verdict": summary.get("verdict"),
                      "funnel": summary.get("funnel"),
                      "fetch": summary.get("fetch"),
                      "completeness": (summary.get("completeness") or {}).get("verdict"),
                      "dark_comets": summary.get("dark_comets"),
                      "high_albedo": summary.get("high_albedo"),
                      "negative_a1_census": summary.get("negative_a1_census"),
                      "query_status_counts": summary.get("query_status_counts"),
                      "degradation": summary.get("degradation")}, indent=2,
                     default=str))


def _cmd_panspermia_regime(args, cfg):
    from .panspermia.encounters import regime_summary, transfer_regime

    src = cfg.root / "results" / "panspermia" / "encounters_all.csv"
    if not src.exists():
        print(f"[panspermia-regime] no encounter table at {src}; run panspermia-run first")
        return
    df = pd.read_csv(src)
    # A fast encounter can only transfer material by geometric INTERCEPTION, not
    # gravitational capture, so sweep plausible donor-reservoir radii and report
    # which mode (if any) any past encounter satisfies.
    reservoirs = [0.5, 0.2, 0.1, 0.03, 4.8e-3, 2.4e-4]   # pc: outer Oort ... Kuiper
    rows = [regime_summary(df, donor_mass_msun=args.donor_mass, reservoir_pc=r)
            for r in reservoirs]
    out_dir = cfg.root / "results" / "panspermia"
    pd.DataFrame(rows).to_csv(out_dir / "transfer_regime.csv", index=False)
    # Full per-encounter classification at the headline reservoir.
    reg = transfer_regime(df, donor_mass_msun=args.donor_mass,
                          reservoir_pc=args.reservoir_pc)
    reg = reg[reg["t_enc_myr"] < 0].sort_values("d_min_pc")
    cols = [c for c in ("source_id", "dist_pc", "phot_g_mean_mag", "bp_rp",
                        "v_rel_kms", "t_enc_myr", "d_min_pc", "d_min_au",
                        "v_esc_at_dmin_kms", "within_reservoir", "capturable",
                        "focusing_factor", "transfers") if c in reg.columns]
    reg[cols].to_csv(out_dir / "regime_by_encounter.csv", index=False)
    (out_dir / "regime_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    n_ok = int(reg["transfers"].sum())
    print(f"[panspermia-regime] {n_ok} of {len(reg)} past encounters permit ANY "
          f"passive transfer at reservoir={args.reservoir_pc} pc, "
          f"donor={args.donor_mass} Msun")


def _cmd_science_blend(args, cfg):
    from .discriminate.blend import blend_followup

    src = args.candidates or str(cfg.root / "results" / "science"
                                 / "multimodal_candidates.csv")
    cand = pd.read_csv(src)
    if args.top:
        cand = cand.sort_values("multimodal_score", ascending=False).head(args.top)
    blend_followup(cand, cfg.root / "results" / "science")


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


def _cmd_rust_sweep(args, cfg):
    from .rust.run import rust_sweep

    rust_sweep(cfg, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
               box_deg=args.box_deg, min_epochs=args.min_epochs,
               min_epochs_season=args.min_epochs_season,
               min_seasons=args.min_seasons, season_days=args.season_days,
               time_budget_s=args.time_budget_s, max_boxes=args.max_boxes)


def _cmd_rust_vet(args, cfg):
    from .rust.run import rust_vet

    rust_vet(cfg, max_candidates=args.max_candidates, offline=args.offline)


def _cmd_knell_sweep(args, cfg):
    from .knell.run import knell_sweep

    knell_sweep(cfg, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
                box_deg=args.box_deg, min_epochs=args.min_epochs,
                time_budget_s=args.time_budget_s, max_boxes=args.max_boxes,
                max_sources=args.max_sources, seed=args.seed)


def _cmd_knell_vet(args, cfg):
    from .knell.run import knell_vet

    knell_vet(cfg, max_candidates=args.max_candidates, offline=args.offline)


def _cmd_knell_cross(args, cfg):
    from .knell.run import knell_cross

    knell_cross(cfg, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
                max_targets=args.max_targets, seed=args.seed)


def _cmd_tocsin_probe(args, cfg):
    from .tocsin.run import probe

    probe(cfg, out_dir=args.out_dir)


def _cmd_tocsin_targets(args, cfg):
    from .tocsin.run import build_targets

    rec = build_targets(cfg, out_path=args.out)
    print(f"[tocsin] targets verdict={rec['verdict']} n={rec['n_targets']}")


def _cmd_tocsin_screen(args, cfg):
    from .tocsin.run import screen

    screen(cfg, chunks=args.chunks, max_run_seconds=args.max_run_seconds,
           lookback_nights=args.lookback_nights,
           mjd_lo=args.mjd_lo, mjd_hi=args.mjd_hi,
           targets_path=args.targets, out_dir=args.out_dir,
           use_gaia_join=not args.no_gaia_join)


def _cmd_tocsin_assess(args, cfg):
    from .tocsin.run import assess_only

    assess_only(cfg, out_dir=args.out_dir)


def _cmd_vigil_probe(args, cfg):
    from .vigil.run import vigil_probe

    vigil_probe(cfg, ra=args.ra, dec=args.dec)


def _cmd_vigil_sweep(args, cfg):
    from .vigil.run import vigil_sweep

    vigil_sweep(cfg, ra=args.ra, dec=args.dec, radius_deg=args.radius_deg,
                g_max=args.g_max, max_stars=args.max_stars,
                time_budget_s=args.time_budget_s,
                untimely_table=args.untimely_table,
                untimely_service=args.untimely_service,
                use_field_query=not args.no_field_query, w1_max=args.w1_max)


def _cmd_vigil_vet(args, cfg):
    from .vigil.run import vigil_vet

    vigil_vet(cfg, max_candidates=args.max_candidates, offline=args.offline)


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


def _cmd_ossuary(args, cfg):
    from .ossuary.run import run

    run(cfg, stage=args.stage, input_path=args.input, g_max=args.g_max,
        limit_per_band=args.limit_per_band,
        do_followup=not args.no_followup, max_followup=args.max_followup)


def _cmd_cenotaph(args, cfg):
    from .cenotaph.run import cenotaph_run

    res = cenotaph_run(cfg, stage=args.stage, out_dir=args.out,
                       synthetic=args.synthetic, n_synth=args.n_synth,
                       seed=args.seed, z_min=args.z_min,
                       t_assumed_k=args.t_assumed_k, max_fit=args.max_fit,
                       poe_min=args.poe_min, ruwe_max=args.ruwe_max,
                       logg_min=args.logg_min, teff_lo=args.teff_lo,
                       teff_hi=args.teff_hi, plx_min_mas=args.plx_min_mas,
                       probe_plx_lo=args.probe_plx_lo,
                       probe_plx_hi=args.probe_plx_hi)
    if args.stage == "probe":
        return
    print(json.dumps({"verdict": res.get("verdict"),
                      "funnel": res.get("funnel", {})}, indent=2))


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

    p = sub.add_parser("accel-run")
    p.add_argument("--limit", type=int, default=6000)
    p.add_argument("--plx-min", type=float, default=2.0)     # within ~500 pc
    p.add_argument("--sig-min", type=float, default=20.0)
    p.set_defaults(func=_cmd_accel_run)

    p = sub.add_parser("accel-xmatch",
                       help="cross-match the class-3 dark-companion shortlist "
                            "against published Gaia compact-companion catalogues")
    p.add_argument("--candidates", default=None,
                   help="CSV of candidates (default results/accel/class3_shortlist.csv)")
    p.set_defaults(func=_cmd_accel_xmatch)

    p = sub.add_parser("science-blend",
                       help="WISE-blend + co-movement test on the WD IR-excess "
                            "multimodal shortlist")
    p.add_argument("--candidates", default=None,
                   help="CSV (default results/science/multimodal_candidates.csv)")
    p.add_argument("--top", type=int, default=0,
                   help="only test the top-N by multimodal_score (0 = all)")
    p.set_defaults(func=_cmd_science_blend)

    p = sub.add_parser("cluster-run",
                       help="population clustering test: is the Gaia x AllWISE "
                            "IR-excess tail over-clustered in phase space?")
    p.add_argument("--ra", type=float, default=200.0)
    p.add_argument("--dec", type=float, default=0.0)
    p.add_argument("--radius-deg", type=float, default=12.0)
    p.add_argument("--plx-min", type=float, default=2.0)     # within ~500 pc
    p.add_argument("--g-max", type=float, default=16.0)
    p.add_argument("--limit", type=int, default=200000)
    p.add_argument("--excess-z-min", type=float, default=4.0)
    p.add_argument("--link-pc", type=float, default=8.0)
    p.set_defaults(func=_cmd_cluster_run)

    p = sub.add_parser("panspermia-run",
                       help="close-encounter search: which stars passed close and "
                            "slow to K2-18 (hycean biosignature host) in the past?")
    p.add_argument("--source-id", type=int, default=3892950081412683520,
                   help="Gaia DR3 source_id of the anchor (default K2-18)")
    p.add_argument("--search-pc", type=float, default=40.0,
                   help="3D search radius around the anchor (pc)")
    p.add_argument("--g-max", type=float, default=16.0)
    p.add_argument("--limit", type=int, default=400000)
    p.add_argument("--t-max-myr", type=float, default=10.0,
                   help="past-encounter viability window (Myr)")
    p.add_argument("--d-min-max-pc", type=float, default=2.0,
                   help="closest-approach cut defining the shortlist (pc)")
    p.set_defaults(func=_cmd_panspermia_run)

    p = sub.add_parser("cluster-aggregate",
                       help="combine the multi-cone clustering sweep into one "
                            "global/trials-corrected result")
    p.set_defaults(func=_cmd_cluster_aggregate)

    p = sub.add_parser("tidemark-acquire",
                       help="fetch the wide-area Gaia x AllWISE parent sample "
                            "(every star searched) for the spatial rate test")
    p.add_argument("--grid", default="sparse",
                   choices=["sparse", "dense", "plane", "pilot"])
    p.add_argument("--radius-deg", type=float, default=6.0)
    p.add_argument("--plx-min", type=float, default=1.0)
    p.add_argument("--g-max", type=float, default=17.0)
    p.add_argument("--stride", type=int, default=20,
                   help="uniform random_index subsampling stride (position-independent)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1,
                   help="interleaved cone sharding for the workflow matrix")
    p.set_defaults(func=_cmd_tidemark_acquire)

    p = sub.add_parser("tidemark-run",
                       help="TIDEMARK: is any anomaly population's rate per star "
                            "spatially structured? gradient + edge + age vs a "
                            "parent-matched null")
    p.add_argument("--channels", nargs="*", default=None,
                   help="channel names from config/tidemark.yaml (default: all)")
    p.add_argument("--seed", type=int, default=20260726)
    p.add_argument("--quick", action="store_true",
                   help="fewer Monte Carlo draws; for smoke tests, not science")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the injection-sensitivity and gradient-transfer stages")
    p.set_defaults(func=_cmd_tidemark_run)

    p = sub.add_parser("tidemark-search",
                       help="self-sufficient TIDEMARK: acquire a wide-area parent "
                            "sample with an IR-excess axis and test it")
    p.add_argument("--grid", default="sparse",
                   choices=["sparse", "dense", "plane", "pilot"])
    p.add_argument("--radius-deg", type=float, default=6.0)
    p.add_argument("--plx-min", type=float, default=1.0)
    p.add_argument("--g-max", type=float, default=17.0)
    p.add_argument("--excess-z-min", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=400000)
    p.add_argument("--from-parent", default=None,
                   help="glob of already-fetched parent shards to reduce instead "
                        "of refetching, e.g. 'results/tidemark/parent/*.parquet'")
    p.add_argument("--seed", type=int, default=20260726)
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=_cmd_tidemark_search)

    p = sub.add_parser("panspermia-mc",
                       help="Monte-Carlo uncertainty on the K2-18 encounter "
                            "shortlist (robust d_min/t_enc confidence intervals)")
    p.add_argument("--n", type=int, default=3000,
                   help="Monte-Carlo samples per candidate")
    p.set_defaults(func=_cmd_panspermia_mc)

    p = sub.add_parser("panspermia-targets",
                       help="directed-travel destination ranking: which reachable "
                            "neighbours would a K2-18 civilisation choose?")
    p.add_argument("--target", choices=["hycean", "classical"], default="hycean",
                   help="habitability prior: hycean (K2-18-like worlds) or classical")
    p.add_argument("--crossmatch", action="store_true",
                   help="runner-only: cross-match NASA Exoplanet Archive hosts")
    p.add_argument("--max-pc", type=float, default=80.0,
                   help="host-distance limit for the Exoplanet-Archive pull (pc)")
    p.set_defaults(func=_cmd_panspermia_targets)

    p = sub.add_parser("panspermia-dossier",
                       help="runner: exhaustive per-target signature sweep of the "
                            "two candidates (Gaia/WISE/ZTF/XP)")
    p.set_defaults(func=_cmd_panspermia_dossier)

    p = sub.add_parser("lhs1140",
                       help="runner: exhaustive bio/techno-signature sweep of the "
                            "LHS 1140 system, its planets, and its neighbours")
    p.add_argument("--sphere-pc", type=float, default=10.0,
                   help="neighbour-sweep radius around LHS 1140 (pc)")
    p.set_defaults(func=_cmd_lhs1140)

    p = sub.add_parser("galactic-encounters",
                       help="runner: long-baseline (few hundred Myr) Galactic-orbit "
                            "encounter search for LHS 1140 + K2-18 + signature battery")
    p.add_argument("--t-max-myr", type=float, default=300.0,
                   help="lookback baseline for orbit integration (Myr)")
    p.add_argument("--search-pc", type=float, default=250.0,
                   help="present-day sphere radius to draw the RV-complete sample")
    p.add_argument("--d-cut-pc", type=float, default=3.0,
                   help="closest-approach cut defining the encounter shortlist (pc)")
    p.add_argument("--limit", type=int, default=150000,
                   help="max Gaia 6D stars to integrate")
    p.set_defaults(func=_cmd_galactic)

    p = sub.add_parser("jwst-bio",
                       help="runner: real JWST/HST transmission-spectrum biosignature "
                            "analysis of LHS 1140 b (disequilibrium pair + M-dwarf "
                            "abiotic gate + MIRI eclipse discriminant + laser scan)")
    p.set_defaults(func=_cmd_jwst_bio)

    def _add_herdsman_sample_args(p):
        p.add_argument("--d-max-pc", type=float, default=300.0,
                       help="sample sphere radius (pc)")
        p.add_argument("--g-max", type=float, default=14.5,
                       help="G magnitude cut")
        p.add_argument("--rv-err-max", type=float, default=1.5,
                       help="max Gaia radial_velocity_error (km/s)")
        p.add_argument("--sigv-max", type=float, default=0.8,
                       help="max scalar space-velocity error incl. floors (km/s)")
        p.add_argument("--astro-floor", type=float, default=0.3,
                       help="astrophysical RV floor: grav. redshift + convective "
                            "blueshift star-to-star scatter (km/s)")

    def _add_herdsman_detector_args(p):
        p.add_argument("--t-max-myr", type=float, default=20.0,
                       help="scan horizon per time direction (Myr)")
        p.add_argument("--dt-myr", type=float, default=0.25,
                       help="leapfrog step (Myr)")
        p.add_argument("--rec-every", type=int, default=2,
                       help="detect every N steps")
        p.add_argument("--r0-pc", type=float, default=1.0,
                       help="meeting-ball radius floor (pc)")
        p.add_argument("--kappa", type=float, default=1.0,
                       help="error-growth factor in R(t) = r0 + kappa*sigma_v*|t|")
        p.add_argument("--lambda-cap", type=float, default=0.5,
                       help="stop scanning when a typical ball holds this many "
                            "field stars by chance")
        p.add_argument("--n-min", type=int, default=4,
                       help="minimum stars meeting simultaneously")
        p.add_argument("--r-now-min-pc", type=float, default=20.0,
                       help="median present-day pairwise separation floor (pc)")
        p.add_argument("--focus-min", type=float, default=3.0,
                       help="required rms contraction factor")
        p.add_argument("--surprise-min", type=float, default=3.0,
                       help="-log10 Poisson tail to record a candidate")
        p.add_argument("--sigv-int-max", type=float, default=None,
                       help="v2 herd cut: max internal velocity dispersion at "
                            "the meeting (km/s); omit to disable")
        p.add_argument("--min-epochs", type=int, default=1,
                       help="v2 herd cut: minimum epochs a meeting must persist")

    p = sub.add_parser("herdsman",
                       help="runner: kinematic technosignature search — N-star "
                            "orbital convergences (herded stars / heterogeneous "
                            "rendezvous) in the Gaia DR3 6D precision sample "
                            "(monolithic; prefer the staged herdsman-* commands "
                            "on CI)")
    _add_herdsman_sample_args(p)
    _add_herdsman_detector_args(p)
    p.add_argument("--n-mocks", type=int, default=24,
                   help="velocity-shuffled mock catalogues per direction")
    p.add_argument("--mock-cell-pc", type=float, default=40.0,
                   help="shuffle cell size (pc)")
    p.set_defaults(func=_cmd_herdsman)

    p = sub.add_parser("tailings",
                       help="runner: sparse chemical-anomaly search — one or "
                            "two elements extreme with the rest normal, in "
                            "GALAH/APOGEE cool dwarfs where the convective "
                            "envelope forbids natural single-element "
                            "peculiarity (docs/tailings.md)")
    p.add_argument("--stage",
                   choices=("acquire", "manifold", "sparse", "vet", "twins", "all"),
                   default="all",
                   help="stage to run (each checkpoints; 'all' resumes from "
                        "whatever checkpoints exist, so a re-reduction never "
                        "costs an archive pull)")
    p.add_argument("--surveys", default="GALAH,APOGEE",
                   help="comma-separated surveys to search")
    p.add_argument("--max-rows", type=int, default=400_000,
                   help="per-survey row cap for the catalogue pull")
    p.set_defaults(func=_cmd_tailings)

    p = sub.add_parser("tailings-validate",
                       help="offline: inject Griffith et al. 2021's 15 Na-enhanced "
                            "stars (0.3-0.6 dex Na, normal O-through-Ni; "
                            "arXiv:2110.06240) into a GALAH-DR3-like population and "
                            "measure what the sparse statistic recovers. No network. "
                            "Writes results/tailings/validation.json")
    p.add_argument("--seed", type=int, default=20260726,
                   help="RNG seed for the synthetic population and the injection")
    p.add_argument("--n-field", type=int, default=6000,
                   help="number of un-injected field stars to synthesise")
    p.add_argument("--no-vet", action="store_true",
                   help="skip the catalogue-level contamination funnel")
    p.set_defaults(func=_cmd_tailings_validate)

    p = sub.add_parser("rust-sweep",
                       help="runner: RUST stage 1 — one sky field's worth of "
                            "paired ZTF g+r light curves scored for a SECULAR "
                            "RISE in aperiodic variability amplitude (an "
                            "un-station-kept swarm entering a collisional "
                            "cascade; Lacki 2025). Distinct from `dimming`, "
                            "which trends the first moment (docs/rust.md)")
    p.add_argument("--ra", type=float, default=270.0)
    p.add_argument("--dec", type=float, default=30.0)
    p.add_argument("--radius-deg", type=float, default=0.5,
                   help="half-width of the square field to tile")
    p.add_argument("--box-deg", type=float, default=0.12,
                   help="IRSA bulk-fetch tile size")
    p.add_argument("--min-epochs", type=int, default=60,
                   help="minimum epochs per band; a second moment needs far "
                        "more than a median does")
    p.add_argument("--min-epochs-season", type=int, default=8)
    p.add_argument("--min-seasons", type=int, default=4)
    p.add_argument("--season-days", type=float, default=365.25)
    p.add_argument("--time-budget-s", type=float, default=2400.0)
    p.add_argument("--max-boxes", type=int, default=None)
    p.set_defaults(func=_cmd_rust_sweep)

    p = sub.add_parser("rust-vet",
                       help="RUST stage 2: aggregate every field's candidates "
                            "and run the contamination gauntlet — mandatory "
                            "g/r coincidence, achromaticity against the "
                            "extinction law, crowding, saturation, known "
                            "variable classes, Gaia RUWE/NSS, NEOWISE dust "
                            "production")
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--offline", action="store_true",
                   help="skip every network follow-up; verdicts degrade "
                        "explicitly rather than silently")
    p.set_defaults(func=_cmd_rust_vet)

    p = sub.add_parser("knell-sweep",
                       help="runner: KNELL stage 1 — one sky field's worth of "
                            "paired ZTF g+r light curves searched for a PERIOD "
                            "THAT CEASED (signature S32). Cessation is "
                            "established INTRA-SURVEY at FIXED SENSITIVITY: "
                            "every claimed non-detection is scored against the "
                            "injection-measured detection efficiency for that "
                            "star's own period and amplitude in that block's own "
                            "sampling and noise (docs/knell.md §3)")
    p.add_argument("--ra", type=float, default=270.0)
    p.add_argument("--dec", type=float, default=30.0)
    p.add_argument("--radius-deg", type=float, default=0.5,
                   help="half-width of the square field to tile")
    p.add_argument("--box-deg", type=float, default=None,
                   help="IRSA bulk-fetch tile size (default from config/knell.yaml)")
    p.add_argument("--min-epochs", type=int, default=None,
                   help="minimum ZTF epochs per band; a PER-BLOCK periodogram "
                        "needs ~15+ epochs in each of >=4 blocks in each band")
    p.add_argument("--time-budget-s", type=float, default=None)
    p.add_argument("--max-boxes", type=int, default=None)
    p.add_argument("--max-sources", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260726,
                   help="RNG seed for the permutation nulls and the injections")
    p.set_defaults(func=_cmd_knell_sweep)

    p = sub.add_parser("tocsin-probe",
                       help="TOCSIN stage 0: confirm the Rubin alert brokers are "
                            "reachable from the runner and record the LIVE TAP "
                            "schema verbatim. Every ADQL column name in the "
                            "channel is inferred from the brokers' published "
                            "source rather than from a live query, so this must "
                            "succeed before any science claim")
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=_cmd_tocsin_probe)

    p = sub.add_parser("tocsin-targets",
                       help="TOCSIN stage 1: build the Gaia DR3 nearby-star "
                            "target list in equal-volume parallax shells, with "
                            "GSPC synthetic photometry as the fallback baseline "
                            "flux")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_tocsin_targets)

    p = sub.add_parser("tocsin-screen",
                       help="TOCSIN stage 2 (the nightly screen): pull the "
                            "night's LSST difference-image detections on "
                            "catalogued nearby stars, run the achromaticity / "
                            "dipole / mover / glint funnel in BOTH flux "
                            "polarities, and fold the survivors into the "
                            "persistent cross-night ledger")
    p.add_argument("--lookback-nights", type=float, default=None,
                   help="window to pull, in nights (default from config)")
    p.add_argument("--max-run-seconds", type=float, default=None,
                   help="wall-clock budget for the whole dispatch; the chunk "
                        "loop yields before the CI job timeout so the ledger is "
                        "always committed (default from config)")
    p.add_argument("--chunks", type=int, default=1,
                   help="walk this many consecutive windows in one run; stops "
                        "early once the watermark reaches the broker's frontier "
                        "(use to backfill the ~262 nights already available)")
    p.add_argument("--mjd-lo", type=float, default=None)
    p.add_argument("--mjd-hi", type=float, default=None)
    p.add_argument("--targets", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-gaia-join", action="store_true",
                   help="audit mode: disable the broker-side Gaia parallax "
                        "pre-cut to MEASURE what it loses (it cross-matches at "
                        "the catalogue epoch, so it can orphan exactly the "
                        "high-proper-motion dwarfs this channel wants)")
    p.set_defaults(func=_cmd_tocsin_screen)

    p = sub.add_parser("tocsin-assess",
                       help="TOCSIN stage 3 (offline): recompute the ensemble "
                            "rate, the trial-corrected per-target p-values and "
                            "the promotion tiers over the whole accumulated "
                            "ledger, without touching the network")
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=_cmd_tocsin_assess)

    p = sub.add_parser("knell-vet",
                       help="KNELL stage 2: aggregate every field's candidates "
                            "and run the contamination gauntlet — mandatory "
                            "two-band coincidence at a common epoch and period, "
                            "crowding, ZTF's photometric walls, SIMBAD classes, "
                            "and the named astrophysical cessation mechanisms "
                            "(third-body precession, Blazhko, mode switching, "
                            "spot cycles, CV disc states)")
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--offline", action="store_true",
                   help="skip every network follow-up; verdicts degrade "
                        "explicitly rather than silently")
    p.set_defaults(func=_cmd_knell_vet)

    p = sub.add_parser("knell-cross",
                       help="KNELL stage 3 (SECONDARY): catalogued VSX variables "
                            "that ZTF no longer detects. Every candidate carries "
                            "an explicit injection demonstration that ZTF WOULD "
                            "HAVE detected the catalogued period and amplitude; "
                            "without it a catalogue-vs-ZTF difference is a "
                            "statement about two telescopes, not about a star")
    p.add_argument("--ra", type=float, default=270.0)
    p.add_argument("--dec", type=float, default=30.0)
    p.add_argument("--radius-deg", type=float, default=0.5)
    p.add_argument("--max-targets", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260726)
    p.set_defaults(func=_cmd_knell_cross)

    p = sub.add_parser("vigil-probe",
                       help="runner: VIGIL stage 0 — one minimal live call per "
                            "route. Establishes whether the unTimely mid-IR "
                            "variable catalogue (arXiv:2511.22071) is reachable "
                            "and what it is called, and whether NEOWISE "
                            "per-epoch W1/W2 photometry comes back, BEFORE a "
                            "multi-hour sweep is spent finding out")
    p.add_argument("--ra", type=float, default=266.0,
                   help="probe position; the default is near the north ecliptic "
                        "pole, where NEOWISE visit counts are highest")
    p.add_argument("--dec", type=float, default=65.0)
    p.set_defaults(func=_cmd_vigil_probe)

    p = sub.add_parser("vigil-sweep",
                       help="runner: VIGIL stage 1 — one sky field searched for "
                            "stars VARIABLE IN THE MID-INFRARED WHILE CONSTANT "
                            "IN THE OPTICAL at LOW fractional excess (waste heat "
                            "tracking a computational load; docs/vigil.md). The "
                            "confounder is the extreme debris disk, which has "
                            "the same phenomenology at HIGH excess")
    p.add_argument("--ra", type=float, default=266.0)
    p.add_argument("--dec", type=float, default=65.0)
    p.add_argument("--radius-deg", type=float, default=0.4)
    p.add_argument("--g-max", type=float, default=15.0,
                   help="magnitude ceiling on the Gaia sample")
    p.add_argument("--max-stars", type=int, default=400,
                   help="cap on stars sent to per-object NEOWISE fetching")
    p.add_argument("--time-budget-s", type=float, default=3000.0)
    p.add_argument("--untimely-table", default="",
                   help="table name discovered by `vigil-probe` for the unTimely "
                        "mid-IR variable catalogue (arXiv:2511.22071). When given, "
                        "the Gaia field sample is pre-selected to catalogued mid-IR "
                        "variables; when the fetch fails the FULL sample is searched "
                        "and the degradation is recorded, never hidden")
    p.add_argument("--untimely-service", default="",
                   help="TAP endpoint hosting --untimely-table")
    p.add_argument("--no-field-query", action="store_true",
                   help="disable the single field-wide NEOWISE query and fetch "
                        "one cone per star. The field query is ~100x faster (the "
                        "first run measured ~90 s per single-star cone) and falls "
                        "back to per-star cones for anything it misses")
    p.add_argument("--w1-max", type=float, default=14.5,
                   help="W1 ceiling on the field-wide NEOWISE query; bounds the "
                        "row count without touching the G < 15 sample")
    p.set_defaults(func=_cmd_vigil_sweep)

    p = sub.add_parser("vigil-vet",
                       help="VIGIL stage 2: aggregate every field's candidates "
                            "and run the contamination gauntlet — optical "
                            "constancy, SIMBAD YSO/AGB types, the astrometric "
                            "(not colour) AGN test, W4-only cirrus, blends, "
                            "NEOWISE saturation")
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--offline", action="store_true",
                   help="skip every network follow-up; verdicts degrade "
                        "explicitly rather than silently")
    p.set_defaults(func=_cmd_vigil_vet)

    p = sub.add_parser("herdsman-fetch",
                       help="staged runner 1/3: acquire + preprocess the Gaia "
                            "6D detection table, save results/herdsman/"
                            "sample.parquet for the scan fleet")
    _add_herdsman_sample_args(p)
    p.set_defaults(func=_cmd_herdsman_fetch)

    p = sub.add_parser("herdsman-scan",
                       help="staged runner 2/3: one scan shard over the saved "
                            "sample (mode=real: both time directions; "
                            "mode=mock: a disjoint slice of mock indices); "
                            "each completed scan checkpoints to its own JSON")
    _add_herdsman_detector_args(p)
    p.add_argument("--mode", choices=("real", "mock"), default="real")
    p.add_argument("--shard", type=int, default=0,
                   help="mock shard index (global mock k = "
                        "shard*mocks_per_shard + j)")
    p.add_argument("--mocks-per-shard", type=int, default=4)
    p.add_argument("--mock-cell-pc", type=float, default=40.0,
                   help="shuffle cell size (pc)")
    p.set_defaults(func=_cmd_herdsman_scan)

    p = sub.add_parser("herdsman-b",
                       help="runner: completed-assembly audit — census-wide "
                            "chemical-coherence test of cluster membership vs "
                            "Gaia GSP-Phot metallicities (impossible clusters; "
                            "docs/herdsman.md section 5)")
    p.add_argument("--stage", choices=("catalog", "chem", "field", "score",
                                       "spectro", "all"), default="all",
                   help="stage to run (each checkpoints; 'all' resumes; "
                        "'spectro' is the GALAH/APOGEE crossmatch and needs "
                        "the members.parquet checkpoint from 'catalog', not "
                        "the GSP-Phot chem/field stages)")
    p.set_defaults(func=_cmd_herdsman_b)

    p = sub.add_parser("shroud",
                       help="runner: enshrouded-not-destroyed search (S33) — "
                            "the never-analysed Solano+2022 optical-absent / "
                            "IR-present VO catalogue, SED energy budget and "
                            "population decomposition (docs/shroud.md)")
    p.add_argument("--stage",
                   choices=("acquire", "photometry", "analyze", "all"),
                   default="all",
                   help="'acquire' pulls the VO catalogues and stops; "
                        "'photometry' adds the AllWISE/2MASS/Gaia join and the "
                        "offset-position null; 'analyze' reuses the "
                        "checkpointed sample (each stage checkpoints)")
    p.add_argument("--offline", action="store_true",
                   help="never touch the network; use the cached sample or "
                        "emit NO_DATA_REACHED")
    p.add_argument("--max-sources", type=int, default=0,
                   help="truncate the sample (0 = no limit); the truncation is "
                        "recorded in the run provenance")
    p.add_argument("--input", default=None,
                   help="analyse this parquet instead of acquiring")
    p.set_defaults(func=_cmd_shroud)

    p = sub.add_parser(
        "ember",
        help="runner: cross-epoch mid-infrared excess CESSATION search -- an "
             "excess present in IRAS (1983) or AKARI (2006) and absent in WISE "
             "(2010). Signature S1, waste heat that switched off (docs/ember.md)")
    p.add_argument("--stage",
                   choices=("audit", "probe", "acquire", "analyse", "excess",
                            "cessation", "vet", "report", "all"),
                   default="all",
                   help="'audit' is offline and decides which epoch pairs are "
                        "usable; 'probe' issues one query per archive primitive "
                        "and prints row counts and first rows (runner-only); "
                        "'acquire' fetches one RA shard; 'analyse' "
                        "reduces whatever shards are on disk without network")
    p.add_argument("--n-ra-chunks", type=int, default=12,
                   help="RA slices per catalogue pull (smaller queries are safer)")
    p.add_argument("--shard", type=int, default=0,
                   help="which RA shard this process handles")
    p.add_argument("--n-shards", type=int, default=1,
                   help="total number of RA shards")
    p.add_argument("--require-all-checks", choices=("true", "false"),
                   default="true",
                   help="reject candidates that could not be tested on every "
                        "contamination rule")
    p.set_defaults(func=_cmd_ember)

    p = sub.add_parser("isotherm",
                       help="runner: search on the SHAPE of infrared excess in "
                            "temperature space — emissivity index beta, "
                            "temperature-distribution width, silicate-feature "
                            "equivalent width, and >=3 discrete components in "
                            "geometric progression, over CASSIS Spitzer/IRS "
                            "spectra (docs/isotherm.md)")
    p.add_argument("--stage",
                   choices=("probe", "corpus", "screen", "shape", "calibrate",
                            "score", "all"),
                   default="all",
                   help="stage to run (each checkpoints; 'probe' only reports "
                        "archive reachability; 'calibrate' is offline)")
    p.add_argument("--max-spectra", type=int, default=2000,
                   help="cap on spectra analysed per shard")
    p.add_argument("--resolution", type=float, default=100.0,
                   help="rebin to this R before fitting, so BIC counts "
                        "independent resolution elements and not raw pixels")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.set_defaults(func=_cmd_isotherm)

    p = sub.add_parser("midden",
                       help="runner: survey-scale short-lived-radionuclide "
                            "search (Whitmire & Wright 1980) in ESO Phase-3 "
                            "HARPS+FEROS spectra (docs/midden.md)")
    p.add_argument("--stage",
                   choices=("verify-lines", "targets", "acquire",
                            "acquire-analyze", "score", "all"),
                   default="all",
                   help="stage to run (each checkpoints; 'all' resumes; "
                        "verify-lines always re-runs the NIST gate)")
    p.add_argument("--max-spectra", type=int, default=3000,
                   help="v1 corpus cap (spectra, after per-star de-dup)")
    p.add_argument("--batch-size", type=int, default=50,
                   help="FITS per download/measure/discard batch")
    p.set_defaults(func=_cmd_midden)

    p = sub.add_parser("compass",
                       help="runner: orbital-pole coherence patches in Gaia "
                            "DR3 NSS astrometric orbits (docs/compass.md)")
    p.add_argument("--stage", choices=("fetch", "all"), default="all",
                   help="fetch = Gaia pull only; all = pull + scan + null")
    p.add_argument("--radii", default="25,50,100",
                   help="comma-separated neighbourhood radii (pc)")
    p.add_argument("--n-min", type=int, default=8,
                   help="minimum systems per neighbourhood")
    p.add_argument("--n-shuffles", type=int, default=200,
                   help="scanning-law-banded shuffles per radius")
    p.add_argument("--band-deg", type=float, default=5.0,
                   help="ecliptic-latitude shuffle band width (deg)")
    p.add_argument("--sig-min", type=float, default=10.0,
                   help="minimum NSS orbit significance")
    p.add_argument("--poe-min", type=float, default=5.0,
                   help="minimum parallax_over_error")
    p.add_argument("--d-max-pc", type=float, default=2000.0,
                   help="field depth (pc)")
    p.set_defaults(func=_cmd_compass)

    p = sub.add_parser("midden-deep",
                       help="runner: HD 217522 deep-dive — every triplet-"
                            "covering high-resolution ESO spectrum of the "
                            "target and a roAp comparison panel; epoch "
                            "stability + panel-standing scoring")
    p.add_argument("--stage",
                   choices=("discover", "measure", "score", "all"),
                   default="all",
                   help="stage to run (each checkpoints; 'all' resumes)")
    p.add_argument("--epochs-per-star", type=int, default=30,
                   help="epoch cap per comparison star (best SNR first)")
    p.add_argument("--target-epochs", type=int, default=80,
                   help="epoch cap for the target itself")
    p.add_argument("--radius-arcsec", type=float, default=20.0,
                   help="ObsCore position-match box half-width")
    p.add_argument("--batch-size", type=int, default=25,
                   help="FITS per download/measure/discard batch")
    p.set_defaults(func=_cmd_midden_deep)

    p = sub.add_parser("herdsman-reduce",
                       help="staged runner 3/3: aggregate all scan shards "
                            "(tolerant of lost ones), vet candidates, compute "
                            "global p-values, write final results + REPORT")
    p.add_argument("--n-mocks-expected", type=int, default=None,
                   help="warn if fewer mock scans are present than this")
    p.add_argument("--astro-floor", type=float, default=0.3,
                   help="astrophysical RV floor for the rendezvous MC (km/s)")
    p.set_defaults(func=_cmd_herdsman_reduce)

    p = sub.add_parser("lhs1140-origin",
                       help="runner: donor/directed-travel mirror of the K2-18 channel "
                            "for LHS 1140 (classical rocky-HZ prior)")
    p.add_argument("--search-pc", type=float, default=40.0,
                   help="3D search radius around LHS 1140 (pc)")
    p.add_argument("--t-max-myr", type=float, default=10.0,
                   help="past-encounter viability window (Myr)")
    p.add_argument("--d-min-max-pc", type=float, default=2.0,
                   help="closest-approach cut defining the recipient shortlist (pc)")
    p.add_argument("--crossmatch", action="store_true",
                   help="runner-only: cross-match NASA Exoplanet Archive hosts")
    p.add_argument("--max-pc", type=float, default=80.0,
                   help="host-distance limit for the Exoplanet-Archive pull (pc)")
    p.set_defaults(func=_cmd_lhs1140_origin)

    p = sub.add_parser("crosscorr",
                       help="runner: high-resolution transmission cross-correlation "
                            "biosignature search for LHS 1140 b (O2 A-band + H2O)")
    p.set_defaults(func=_cmd_crosscorr)

    p = sub.add_parser("seti-archive",
                       help="runner: targeted radio+optical SETI archive coverage and "
                            "EIRP-limit dossier for LHS 1140")
    p.add_argument("--snr", type=float, default=5.0,
                   help="detection threshold (sigma) for the EIRP limits")
    p.set_defaults(func=_cmd_seti_archive)

    p = sub.add_parser("iso-backtrack",
                       help="runner: back-track known ISOs (Oumuamua/Borisov/3I) "
                            "through the Galactic potential toward LHS 1140 "
                            "(necessary-not-sufficient; MC over radiant+v_inf)")
    p.add_argument("--t-max-myr", type=float, default=200.0,
                   help="backward orbit-integration baseline (Myr)")
    p.add_argument("--n-mc", type=int, default=2000,
                   help="Monte-Carlo draws per ISO over velocity+radiant uncertainty")
    p.add_argument("--nearby-pc", type=float, default=25.0,
                   help="radius of the nearest-Gaia-star context scan (pc)")
    p.add_argument("--d-close-pc", type=float, default=1.0,
                   help="closest-approach distance defining the caveated flag (pc)")
    p.add_argument("--no-scan", action="store_true",
                   help="skip the nearby-star context scan (offline / dynamics only)")
    p.set_defaults(func=_cmd_iso_backtrack)

    p = sub.add_parser("derelict",
                       help="runner: thin-film / high area-to-mass debris via the "
                            "radial non-gravitational acceleration JPL already "
                            "fits (A1 -> beta -> area-to-mass -> R statistic)")
    p.add_argument("--stage", default="all",
                   choices=["all", "probe", "search", "completeness",
                            "dark_comets", "high_albedo"],
                   help="'probe' discovers the SBDB schema only; 'completeness' "
                        "proves the A1|DF constraint returned the whole A1 "
                        "population by pulling the catalogue unconstrained; "
                        "'dark_comets' runs the Seligman named-target census; "
                        "'high_albedo' runs the catalogue-wide p_V > 0.7 screen "
                        "independently of A1; 'search' is the funnel plus the "
                        "light census stages but WITHOUT the ~1.4M-row "
                        "completeness pull")
    p.add_argument("--limit", type=int, default=None,
                   help="cap rows per SBDB query (debugging)")
    p.add_argument("--offline-input", default=None,
                   help="CSV of pre-fetched SBDB rows; skips all network access")
    p.add_argument("--max-vet", type=int, default=60,
                   help="how many survivors get a per-object sbdb.api detail fetch")
    p.add_argument("--max-control-enrich", type=int, default=400,
                   help="cap on per-object enrichment of the comet control sample")
    p.add_argument("--max-enrich", type=int, default=1500,
                   help="cap on per-object sbdb.api enrichment (the bulk query "
                        "rejects sigma_A1, so sigmas come from orbit.model_pars)")
    p.add_argument("--skip-control", action="store_true",
                   help="skip the comet control sample")
    p.add_argument("--skip-completeness", action="store_true",
                   help="skip the unconstrained completeness pull (~1.4M rows). "
                        "Completeness is then UNTESTED, not proven")
    p.add_argument("--skip-dark-comets", action="store_true",
                   help="skip the Seligman et al. dark-comet named-target census")
    p.add_argument("--skip-high-albedo", action="store_true",
                   help="skip the catalogue-wide p_V > 0.7 screen; screen 4 then "
                        "covers only the A1 sample, a much narrower question")
    p.add_argument("--completeness-limit", type=int, default=None,
                   help="cap rows in the unconstrained completeness pull "
                        "(debugging only: a capped pull cannot prove completeness)")
    p.set_defaults(func=_cmd_derelict)

    p = sub.add_parser("panspermia-regime",
                       help="offline: classify K2-18 encounters by transfer mode "
                            "(capture vs interception) across reservoir radii")
    p.add_argument("--donor-mass", type=float, default=0.36,
                   help="donor stellar mass (Msun); K2-18 ~ 0.36")
    p.add_argument("--reservoir-pc", type=float, default=0.2,
                   help="headline donor reservoir radius (pc)")
    p.set_defaults(func=_cmd_panspermia_regime)

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

    p = sub.add_parser("ossuary",
                       help="runner: OSSUARY -- warm dust around stars that "
                            "cannot make it (metal-poor / halo-kinematic hosts; "
                            "docs/ossuary.md)")
    p.add_argument("--stage", choices=("acquire", "analyze", "followup", "all"),
                   default="all",
                   help="'acquire' pulls the Gaia x AllWISE x 2MASS sample "
                        "(runner only); 'analyze' is offline given the sample")
    p.add_argument("--input", default=None,
                   help="sample parquet (default results/ossuary/sample.parquet)")
    p.add_argument("--g-max", type=float, default=None,
                   help="override the G magnitude ceiling on the sample")
    p.add_argument("--limit-per-band", type=int, default=400_000,
                   help="row cap per declination band in the archive pull")
    p.add_argument("--no-followup", action="store_true",
                   help="skip the per-candidate reddening/neighbour/SIMBAD stage")
    p.add_argument("--max-followup", type=int, default=200,
                   help="cap on the number of candidates sent to follow-up")
    p.set_defaults(func=_cmd_ossuary)

    p = sub.add_parser("cenotaph",
                       help="cold-Dyson search: grey attenuation, no mid-IR "
                            "excess, far-IR recovery of the intercepted L")
    p.add_argument("--stage", default="all",
                   choices=["all", "probe", "sample", "twins", "grey", "midir",
                            "farir", "reduce"],
                   help="'probe' runs ONE minimal live query and prints the "
                        "transport status, COUNT(*) and the first rows — use "
                        "it before spending a multi-hour pull")
    p.add_argument("--out", default=None)
    p.add_argument("--synthetic", action="store_true",
                   help="offline smoke run against a synthetic population")
    p.add_argument("--n-synth", type=int, default=4000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--z-min", type=float, default=3.0,
                   help="leg-1 grey significance threshold")
    p.add_argument("--t-assumed-k", type=float, default=50.0,
                   help="assumed re-radiation temperature for the closure test")
    p.add_argument("--max-fit", type=int, default=None,
                   help="cap the number of grey fits (tail + random control)")
    p.add_argument("--poe-min", type=float, default=20.0)
    p.add_argument("--ruwe-max", type=float, default=1.4)
    p.add_argument("--logg-min", type=float, default=3.8)
    p.add_argument("--teff-lo", type=float, default=4000.0)
    p.add_argument("--teff-hi", type=float, default=7000.0)
    p.add_argument("--plx-min-mas", type=float, default=1.0)
    p.add_argument("--probe-plx-lo", type=float, default=2.0,
                   help="probe stage: lower parallax edge of the test shell (mas)")
    p.add_argument("--probe-plx-hi", type=float, default=2.5,
                   help="probe stage: upper parallax edge of the test shell (mas)")
    p.set_defaults(func=_cmd_cenotaph)

    p = sub.add_parser("figures")
    p.set_defaults(func=_cmd_figures)

    args = parser.parse_args(argv)
    cfg = load_config()
    return args.func(args, cfg)


if __name__ == "__main__":
    main()

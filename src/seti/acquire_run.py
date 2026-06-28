"""End-to-end empirical acquisition: build the analysis-ready table from the real
catalogues, then hand it to the pipeline.

This is the single command behind ``make data``.  It pulls the Gaia EDR3
white-dwarf parent sample, its astrometric-quality columns, CatWISE2020 and
2MASS photometry, the AllWISE neighbourhood (crowding), and the control
debris-disk catalogues, then assembles them into the schema the pipeline
consumes.  All network access flows through ``seti.acquire`` (memoised to the
parquet cache), so once the cache is warm the run is offline-reproducible.

The assembly logic (:func:`assemble_analysis_table`) is a pure function with no
network dependency, so it is unit-tested offline against mock catalogue frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config

# Columns the pipeline requires, with safe defaults for optional ones.  Required
# columns (no default) must be supplied by the catalogues or assembly raises.
_OPTIONAL_DEFAULTS = {
    "W1mag_unwise": np.nan,        # unWISE confirmation: shortlist only
    "gaia_nn_arcsec": np.inf,      # nearest Gaia neighbour: no crowding penalty if absent
    "ext_flg": 0,
    "cc_flags": "0000",
    "ph_qual": "AA",
    "n_wise_neighbours": 1,
    "known_disk": False,
    "pmra_error": 1.0,
    "pmdec_error": 1.0,
    "e_pmra_wise": 50.0,
    "e_pmdec_wise": 50.0,
}

_REQUIRED = [
    "source_id", "ra", "dec", "pmra", "pmdec", "teff",
    "W1mag", "e_W1mag", "W2mag", "e_W2mag", "ra_wise", "dec_wise",
    "Jmag", "Hmag", "Ksmag",
]


def assemble_analysis_table(
    gaia_wd: pd.DataFrame,
    astrometry: pd.DataFrame,
    catwise: pd.DataFrame,
    twomass: pd.DataFrame,
    neighbourhood: pd.DataFrame | None = None,
    known: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge catalogue frames into the pipeline's analysis-ready schema.

    Joins are all on Gaia ``source_id``.  Pure / no network: every input is a
    DataFrame.  Fills optional columns with defaults and validates that the
    required columns are present and non-empty.
    """
    df = gaia_wd.copy()
    for other in (astrometry, twomass, catwise):
        if other is not None and len(other):
            cols = [c for c in other.columns if c == "source_id" or c not in df.columns]
            df = df.merge(other[cols], on="source_id", how="left")

    if neighbourhood is not None and len(neighbourhood):
        df = df.merge(neighbourhood[["source_id", "n_wise_neighbours"]],
                      on="source_id", how="left")

    # Known-disk flag from the control sample.
    df["known_disk"] = False
    if known is not None and len(known) and "source_id" in known.columns:
        known_ids = set(known["source_id"].dropna().astype("int64"))
        df.loc[df["source_id"].isin(known_ids), "known_disk"] = True

    # Derived: W2 signal-to-noise from its magnitude error.
    if "w2snr" not in df.columns and "e_W2mag" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["w2snr"] = 1.0 / (0.4 * np.log(10.0) * df["e_W2mag"])

    for col, default in _OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"assembled table missing required columns: {missing}")
    return df


def acquire_run(
    cfg: Config | None = None,
    max_dist_pc: float | None = 100.0,
    pwd_min: float | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Acquire the real catalogues and write the analysis-ready table.

    ``max_dist_pc`` scopes the parent sample (default: the 100 pc sample, the
    forecast anchor).  ``dry_run`` validates wiring without touching the network
    by asserting the acquire functions are importable and returning an empty
    schema-correct frame.
    """
    cfg = cfg or load_config()
    cache = cfg.path("cache_dir")
    pwd_min = pwd_min if pwd_min is not None else cfg.thresholds["sample"]["pwd_min"]

    # Import here so a dry run never requires astroquery/network.
    from .acquire.controls import acquire_control, merge_controls
    from .acquire.gaia_wd import acquire_gaia_astrometry, acquire_gaia_wd
    from .acquire.twomass import acquire_twomass
    from .acquire.wise import acquire_allwise_neighbourhood, acquire_catwise

    if dry_run:
        # Verify the orchestration wiring without any network call.
        for fn in (acquire_gaia_wd, acquire_gaia_astrometry, acquire_catwise,
                   acquire_twomass, acquire_allwise_neighbourhood, acquire_control,
                   merge_controls):
            assert callable(fn)
        return pd.DataFrame(columns=_REQUIRED)

    gaia_wd = acquire_gaia_wd(cache, pwd_min=pwd_min)
    if max_dist_pc is not None and "parallax" in gaia_wd.columns:
        gaia_wd = gaia_wd[gaia_wd["parallax"] >= 1000.0 / max_dist_pc]
    if limit is not None:
        gaia_wd = gaia_wd.head(limit)
    sids = gaia_wd["source_id"].tolist()

    astrometry = acquire_gaia_astrometry(cache, sids)
    catwise = acquire_catwise(cache, gaia_wd[["source_id", "ra", "dec"]])
    twomass = acquire_twomass(cache, sids)
    neighbourhood = acquire_allwise_neighbourhood(cache, sids)

    controls = merge_controls([acquire_control(cache, "MadurgaFavieres2024"),
                               acquire_control(cache, "DebesWIRED2011")])

    table = assemble_analysis_table(gaia_wd, astrometry, catwise, twomass,
                                    neighbourhood, controls)
    out = cfg.path("processed_dir") / "analysis_ready.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    return table


def science_run(
    cfg: Config | None = None,
    max_dist_pc: float = 100.0,
    limit: int | None = None,
) -> dict:
    """Real-data run (CDS-only, runner-friendly): acquire, analyse, write results.

    Writes the analysis-ready table plus small, committable result files under
    ``results/science/`` (candidate table, summary counts, occurrence limit) so a
    CI runner can commit them back to the repository.  Returns the summary dict.
    """
    import json

    from .acquire.science import (
        fetch_catwise,
        fetch_galex,
        fetch_twomass,
        fetch_wd_parent,
    )
    from .pipeline import run_pipeline

    cfg = cfg or load_config()
    pwd_min = cfg.thresholds["sample"]["pwd_min"]

    parent = fetch_wd_parent(max_dist_pc, pwd_min)
    if limit is not None:
        parent = parent.head(limit)
    positions = parent[["source_id", "ra", "dec"]]

    catwise = fetch_catwise(positions)
    twomass = fetch_twomass(positions)
    galex = fetch_galex(positions)   # GALEX NUV/FUV for the UV-deficit / energy-balance axes
    # Published debris-disk catalogues (e.g. Madurga Favieres 2024) lack Gaia ids
    # and a cleanly decodable excess flag, so a reliable subtraction is deferred;
    # the warm debris-disk population is instead separated by the dust locus (warm
    # disks fall inside it, cool candidates outside). known-disk subtraction off.
    known_ids: set[int] = set()
    known = None

    table = assemble_analysis_table(parent, pd.DataFrame(), catwise, twomass,
                                    neighbourhood=None, known=known)
    if galex is not None and len(galex):
        gcols = [c for c in galex.columns if c == "source_id" or c not in table.columns]
        table = table.merge(galex[gcols], on="source_id", how="left")
        print(f"[science] GALEX matched: {int(table.get('NUVmag', pd.Series()).notna().sum())} "
              f"with NUV")
    print(f"[science] analysis-ready table: {len(table)} white dwarfs")

    # --- Co-movement diagnostic on the real data (find what the cut rejects) ---
    import numpy as np

    from .contamination.comovement import pm_consistency_sigma, propagated_offset_arcsec
    epochs = cfg.catalogs.get("models", {}).get("epochs", {})
    ge, we = epochs.get("gaia_ref_epoch", 2016.0), epochs.get("wise_mean_epoch", 2015.4)
    diag = table[np.isfinite(table.get("ra_wise", np.nan))].copy()
    if len(diag):
        off = propagated_offset_arcsec(diag, ge, we)
        pmsig = pm_consistency_sigma(diag)
        cm = cfg.thresholds["contamination"]["comovement"]
        print(f"[diag] n_with_wisepos={len(diag)}; "
              f"offset_arcsec med={np.nanmedian(off):.3f} p90={np.nanpercentile(off,90):.3f}; "
              f"pm_sig med={np.nanmedian(pmsig):.2f} p90={np.nanpercentile(pmsig,90):.2f}")
        print(f"[diag] fail_position(off>{cm['max_position_offset_arcsec']})="
              f"{int((off>cm['max_position_offset_arcsec']).sum())}; "
              f"fail_pm(sig>{cm['pm_consistency_sigma_max']})="
              f"{int((pmsig>cm['pm_consistency_sigma_max']).sum())} of {len(diag)}")
        ex = diag.iloc[0]
        print(f"[diag] example: pmra={ex.get('pmra'):.1f} pmra_wise={ex.get('pmra_wise')} "
              f"e_pmra_wise={ex.get('e_pmra_wise')} pmdec={ex.get('pmdec'):.1f} "
              f"pmdec_wise={ex.get('pmdec_wise')}")

    out_dir = cfg.root / "results" / "science"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = cfg.path("processed_dir")
    proc.mkdir(parents=True, exist_ok=True)
    table.to_parquet(proc / "analysis_ready.parquet", index=False)

    # Write the (large) per-stage analysis tables to a gitignored dir; only the
    # small summaries and the figures land under results/science/.
    sci_tables = proc / "sci_tables"
    result = run_pipeline(table, cfg=cfg, out_dir=sci_tables)

    cand = result.candidates
    cand_cols = [c for c in ["source_id", "ra", "dec", "teff", "t_dust_k", "tau",
                             "anomaly_score", "chi_W1", "chi_W2"] if c in cand.columns]
    cand[cand_cols].to_csv(out_dir / "candidates.csv", index=False)

    # Empirical figures + candidate LaTeX table from the real data (best-effort).
    try:
        from .empirical_figures import render_empirical
        render_empirical(cfg, sci_tables, out_dir / "figures")
    except Exception as exc:
        print(f"[science] empirical figures skipped: {exc!r}")

    # --- Multi-modal anomaly analysis (energy balance, UV deficit, variability,
    #     kinematics): the additional evidence beyond infrared excess ---------
    mm_summary = {}
    try:
        from .indicators.run import indicator_summary, run_multimodal

        vetted = pd.read_parquet(sci_tables / "vetted.parquet")
        scored_p = sci_tables / "excess_scored.parquet"
        if scored_p.exists():
            sc = pd.read_parquet(scored_p)
            keep = [c for c in ["source_id", "tau", "t_dust_k"] if c in sc.columns]
            vetted = vetted.merge(sc[keep], on="source_id", how="left")
        only_clean = vetted[vetted.get("clean", True).astype(bool)] if "clean" in vetted else vetted

        min_axes = cfg.thresholds["indicators"]["multimodal_min_axes"]

        # Shortlist-only light-curve variability (ZTF optical, NEOWISE infrared):
        # expensive per-object queries.  We use a TWO-PASS shortlist so that any
        # object already anomalous on a non-variability axis (UV deficit, energy
        # balance, kinematics) -- not just infrared excess -- also gets its
        # variability measured, giving it a chance to light up additional axes.
        try:
            from .acquire.variability import (
                fetch_neowise_variability,
                fetch_ztf_variability,
            )
            # Pass 1: combine the axes that need no light-curve data, to find every
            # object already flagged on something.
            prelim = run_multimodal(only_clean, cfg.thresholds, min_axes=min_axes)
            flagged_any = prelim.get("n_axes", pd.Series(0, index=prelim.index)) >= 1
            flagged_ids = set(prelim.loc[flagged_any.to_numpy(), "source_id"])

            # Build the shortlist: every already-flagged object first (these can
            # become multi-axis), then the strongest infrared excesses to fill out.
            sl = only_clean.copy()
            sl["_flagged"] = sl["source_id"].isin(flagged_ids)
            if "has_excess" in sl:
                sl = sl[sl["_flagged"] | sl["has_excess"].fillna(False).astype(bool)]
            sort_key = next((c for c in ("chi_W1", "chi_W2") if c in sl.columns), None)
            by = (["_flagged"] + ([sort_key] if sort_key else []))
            sl = sl.sort_values(by, ascending=False)
            pos = sl[["source_id", "ra", "dec"]].drop_duplicates("source_id")
            print(f"[science] variability shortlist: {len(pos)} objects "
                  f"({len(flagged_ids)} already flagged on a non-variability axis)")
            ztf = fetch_ztf_variability(pos)
            neo = fetch_neowise_variability(pos)
            for extra in (ztf, neo):
                if extra is not None and len(extra):
                    ecols = [c for c in extra.columns
                             if c == "source_id" or c not in only_clean.columns]
                    only_clean = only_clean.merge(extra[ecols], on="source_id", how="left")
        except Exception as exc:
            print(f"[science] shortlist variability skipped: {exc!r}")

        comb = run_multimodal(only_clean, cfg.thresholds, min_axes=min_axes)
        mm_summary = indicator_summary(comb)
        mm_cand = comb[comb["multimodal_candidate"]].sort_values(
            "multimodal_score", ascending=False)
        # Deep-dive vetting: annotate every multi-axis candidate with its SIMBAD
        # identity/object type, so an unexamined anomaly is distinguished from an
        # already-classified disk, binary or known variable.
        try:
            from .acquire.science import classify_candidate, fetch_simbad_context
            if len(mm_cand) and {"ra", "dec"} <= set(mm_cand.columns):
                ctx = fetch_simbad_context(mm_cand[["source_id", "ra", "dec"]].head(50))
                if ctx is not None and len(ctx):
                    ccols = [c for c in ctx.columns
                             if c == "source_id" or c not in mm_cand.columns]
                    mm_cand = mm_cand.merge(ctx[ccols], on="source_id", how="left")
                    mm_cand["candidate_class"] = [
                        classify_candidate(o, s) for o, s in
                        zip(mm_cand.get("simbad_otype", ""),
                            mm_cand.get("simbad_sptype", ""), strict=False)]
        except Exception as exc:
            print(f"[science] SIMBAD vetting skipped: {exc!r}")

        mcols = [c for c in ["source_id", "ra", "dec", "teff", "n_axes", "axes_flagged",
                             "multimodal_score", "tau", "t_dust_k", "chi_W2",
                             "score_uv_deficit", "score_energy_balance",
                             "score_periodicity", "score_optical_variability",
                             "score_ir_variability", "ztf_frac_rms", "ztf_ls_period_d",
                             "ztf_ls_fap", "ztf_ls_alias", "neowise_w1_frac_rms",
                             "nuv_deficit_frac", "simbad_id", "simbad_otype",
                             "simbad_sptype", "candidate_class"]
                 if c in mm_cand.columns]
        mm_cand[mcols].to_csv(out_dir / "multimodal_candidates.csv", index=False)
        comb.to_parquet(sci_tables / "multimodal.parquet", index=False)
        try:
            from .empirical_figures import render_multimodal
            render_multimodal(cfg, comb, out_dir / "figures")
        except Exception as exc:
            print(f"[science] multimodal figures skipped: {exc!r}")
        print("[science] multimodal:", json.dumps(mm_summary))
    except Exception as exc:
        print(f"[science] multimodal analysis skipped: {exc!r}")

    summary = {
        "max_dist_pc": max_dist_pc,
        "n_parent": int(len(table)),
        "counts": result.counts,
        "funnel_counts": result.funnel_counts,
        "occurrence_limit": result.occurrence_limit,
        "n_known_disks_matched": len(known_ids),
        "multimodal": mm_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[science] summary:", json.dumps(summary["counts"]))
    print("[science] occurrence limit:", json.dumps(summary["occurrence_limit"]))
    return summary


__all__ = ["assemble_analysis_table", "acquire_run", "science_run"]

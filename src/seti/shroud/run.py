"""Runner: SHROUD end to end.  Every stage checkpoints; ``all`` runs the chain.

acquire    Pull the Solano+2022 VO catalogues (``vanish-neowise`` = the
           optically-absent / IR-present sample, ``vanish-possi`` = the
           no-counterpart control), plus the offset-position null realisations.
photometry Attach POSS-I, modern-optical and infrared photometry by uploaded
           crossmatch, and the wide-radius Gaia pull used for epoch propagation.
classify   Subtract the mundane population, class by class.
budget     Fit both SED models and run the obscuration-vs-destruction test on
           the residual class.
report     Write ``results/shroud/``.

The verdict is a first-class field.  If no archive answered, the run emits
``NO_DATA_REACHED`` and analyses nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import Config, load_config
from . import acquire as acq
from . import classify as cls
from . import sed as sedmod
from . import vet as vetmod


def load_shroud_config(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    p = cfg.root / "config" / "shroud.yaml"
    with p.open() as fh:
        return yaml.safe_load(fh)


# --- stages -----------------------------------------------------------------
def stage_acquire(cfg: Config, sc: dict, out_dir: Path,
                  allow_network: bool = True) -> tuple[pd.DataFrame, dict]:
    df, prov = acq.acquire_sample(sc, out_dir, allow_network=allow_network)
    (out_dir / "acquire_verdict.json").write_text(
        json.dumps(prov, indent=2, default=str))
    if len(df):
        df.to_parquet(out_dir / "sample_positions.parquet", index=False)
    return df, prov


def stage_photometry(sc: dict, df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Join real photometry onto the positions, and measure the chance rate.

    Two things happen here that the published catalogue cannot supply:

    * **W3/W4 and 2MASS.**  ``vanish-neowise`` was matched against NeoWISE,
      which carries only W1/W2, so without this join the energy budget is
      undersampled by construction (see ``sed.energy_budget``).
    * **The offset-position null.**  The same sightlines displaced by a fixed
      angle are pushed through the identical crossmatch, giving a *measured*
      chance-match rate instead of an assumed uniform background.
    """
    xm = sc.get("crossmatch", {})
    merged, provs = acq.build_photometry_table(df, sc, out_dir)
    (out_dir / "photometry_provenance.json").write_text(
        json.dumps(provs, indent=2, default=str))
    merged.to_parquet(out_dir / "photometry.parquet", index=False)

    # Offset-position null: the chance-match rate, measured on the real sky.
    n_real = int(len(df))
    n_real_matched = int((merged.get("n_ir_neighbours", pd.Series(dtype=int))
                          > 0).sum()) if "n_ir_neighbours" in merged else 0
    n_null = n_null_matched = 0
    cat = sc.get("acquire", {}).get("catalogs", {}).get(
        "allwise", "vizier:II/328/allwise")
    for k in range(int(xm.get("offset_null_realisations", 4))):
        off = vetmod.offset_positions(df, float(xm.get("offset_null_arcsec", 45.0)),
                                      seed=k)
        off["source_id"] = off["source_id"].astype(str) + f"_off{k}"
        res, _ = acq.xmatch_upload(off, cat, float(xm.get("radius_arcsec", 5.0)), sc)
        n_null += len(off)
        if len(res) and "source_id" in res.columns:
            n_null_matched += int(res["source_id"].nunique())
    stats = vetmod.chance_match_rate_from_null(n_real_matched, n_real,
                                               n_null_matched, n_null)
    stats.update({"n_real": n_real, "n_real_matched": n_real_matched,
                  "n_null": n_null, "n_null_matched": n_null_matched,
                  "offset_arcsec": float(xm.get("offset_null_arcsec", 45.0))})
    (out_dir / "null_stats.json").write_text(json.dumps(stats, indent=2,
                                                        default=str))

    # Epoch propagation, using the wide-radius Gaia pull.
    gaia_p = out_dir / "xmatch_gaia.parquet"
    if gaia_p.exists():
        merged = apply_epoch_propagation(merged, pd.read_parquet(gaia_p), sc)
        merged.to_parquet(out_dir / "photometry.parquet", index=False)
    return merged


def apply_epoch_propagation(df: pd.DataFrame, gaia: pd.DataFrame,
                            sc: dict) -> pd.DataFrame:
    """Back-propagate every wide-radius Gaia neighbour to the plate epoch."""
    if not len(gaia) or "source_id" not in gaia.columns:
        return df
    g = gaia.rename(columns={c: c.lower() for c in gaia.columns})
    ra_c = next((c for c in ("ra_icrs", "raj2000", "ra", "_ra") if c in g), None)
    dec_c = next((c for c in ("de_icrs", "dej2000", "dec", "de", "_de")
                  if c in g), None)
    pmra_c = next((c for c in ("pmra",) if c in g), None)
    pmde_c = next((c for c in ("pmde", "pmdec") if c in g), None)
    if not all((ra_c, dec_c, pmra_c, pmde_c)):
        return df
    g = g.rename(columns={ra_c: "ra_deg", dec_c: "dec_deg",
                          pmra_c: "pmra", pmde_c: "pmdec"})
    by_id = {k: v for k, v in g.groupby("source_id")}
    recs = []
    for _, row in df.iterrows():
        sid = str(row["source_id"])
        recs.append(vetmod.epoch_propagation_check(
            float(row["ra_deg"]), float(row["dec_deg"]),
            by_id.get(sid, pd.DataFrame()), sc))
    out = df.copy().reset_index(drop=True)
    for k in recs[0] if recs else []:
        out[k] = [r[k] for r in recs]
    return out


def stage_classify(df: pd.DataFrame, sc: dict) -> pd.DataFrame:
    """Chance-match probability, then the mundane-population cascade."""
    if not len(df):
        return cls.classify_table(df, sc)
    xm = sc.get("crossmatch", {})
    r = float(xm.get("radius_arcsec", 5.0))
    out = df.copy()
    if "ir_local_density_per_deg2" in out.columns:
        out["p_chance_match"] = [
            vetmod.chance_match_probability(r, d)
            for d in out["ir_local_density_per_deg2"].to_numpy(float)]
    return cls.classify_table(out, sc)


def stage_budget(df: pd.DataFrame, sc: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Fit both SED models and run the energy budget on every object with IR.

    Fits are run on **every** row that has enough bands, not only the residual
    class, so the mundane classes provide the control distribution of eta that
    the residual class must be shown to differ from.
    """
    budgets: dict = {}
    fits: dict = {}
    rows = []
    for _, row in df.iterrows():
        sid = str(row.get("source_id", ""))
        s = vetmod.build_sed(row)
        f_phot, f_dust = sedmod.fit_both(s, sc)
        fits[sid] = f_dust
        b = sedmod.energy_budget(s, sc, f_dust if f_dust.ok else None)
        budgets[sid] = b
        rows.append({
            "source_id": sid,
            "n_bands_modern": len(s.detected_modern()),
            "n_ir_bands": len(s.detected(sedmod.IR_BANDS)),
            "chi2_red_photosphere": f_phot.chi2_red if f_phot.ok else np.nan,
            "chi2_red_obscured": f_dust.chi2_red if f_dust.ok else np.nan,
            "prefers_dust": bool(f_dust.ok and f_phot.ok
                                 and f_dust.chi2_red < f_phot.chi2_red),
            "teff_fit_k": f_dust.teff_k if f_dust.ok else np.nan,
            "av_fit_mag": f_dust.a_v_mag if f_dust.ok else np.nan,
            "tdust_fit_k": f_dust.t_dust_k if f_dust.ok else np.nan,
            "budget_verdict": b.verdict,
            "eta_max": b.eta_max, "eta_lo": b.eta_lo, "eta_hi": b.eta_hi,
            "eta_trapz_max": b.eta_trapz_max,
            "budget_note": b.note,
        })
    merged = df.merge(pd.DataFrame(rows), on="source_id", how="left") \
        if rows else df.copy()
    return merged, budgets, fits


def stage_report(cfg: Config, sc: dict, df: pd.DataFrame, prov: dict,
                 out_dir: Path, null_stats: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = prov.get("verdict", "NO_DATA_REACHED")

    if not len(df):
        summary = {
            "channel": "shroud", "verdict": verdict,
            "n_sample": 0, "note": prov.get("note", ""),
            "acquire_routes": prov.get("routes", []),
            "population": [], "survivors": [],
            "degraded": True,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                         default=str))
        (out_dir / "REPORT.md").write_text(_report_md(summary, sc))
        print(f"[shroud] {verdict}: no rows analysed")
        return summary

    pop = cls.population_breakdown(df)
    pop.to_csv(out_dir / "population.csv", index=False)

    ir_mask = df["sample"].astype(str).str.contains("ir_present") \
        if "sample" in df.columns else pd.Series(True, index=df.index)
    ctl_mask = df["sample"].astype(str).str.contains("no_counterpart") \
        if "sample" in df.columns else pd.Series(False, index=df.index)
    ratio = {
        "n_ir_present": int(ir_mask.sum()),
        "n_no_counterpart": int(ctl_mask.sum()),
        "obscuration_to_destruction_ratio":
            (float(ir_mask.sum()) / float(ctl_mask.sum())
             if int(ctl_mask.sum()) else None),
    }
    ratio.update(cls.obscuration_vs_destruction_ratio(df))

    surv = df[df.get("survives", pd.Series(False, index=df.index)).fillna(False)] \
        if "survives" in df.columns else df.iloc[0:0]
    # Both counts are taken over SURVIVORS: an object already killed as
    # contamination must not be reported as a physics result either way.
    cons = surv[surv["budget_verdict"] == "ENERGY_CONSERVING_OBSCURATION"] \
        if len(surv) and "budget_verdict" in surv.columns else surv.iloc[0:0]
    faint = surv[surv["budget_verdict"].isin(
        ["IR_TOO_FAINT", "IR_TOO_FAINT_MARGINAL"])] \
        if len(surv) and "budget_verdict" in surv.columns else surv.iloc[0:0]
    undersampled = int((df["budget_verdict"] == "IR_UNDERSAMPLED").sum()) \
        if "budget_verdict" in df.columns else 0

    keep = [c for c in ("source_id", "ra_deg", "dec_deg", "sample", "class",
                        "glat_deg", "ecl_lat_deg", "poss1_e", "w1", "w2", "w3",
                        "w4", "2mass_ks", "n_ir_bands", "tdust_fit_k",
                        "eta_max", "eta_lo", "eta_hi", "budget_verdict",
                        "ftk_class", "vet_flags", "p_chance_match")
            if c in df.columns]
    # survivors.csv is committed back, so it is capped; classified.csv is the
    # full table and travels as a workflow artifact only.
    max_csv = 5000
    if len(surv):
        surv.sort_values("eta_max", ascending=False)[keep].head(max_csv).to_csv(
            out_dir / "survivors.csv", index=False)
    if len(df):
        df[keep].to_csv(out_dir / "classified.csv", index=False)

    summary = {
        "channel": "shroud",
        "verdict": verdict,
        "degraded": verdict not in ("VO_ARCHIVE",),
        "acquire_note": prov.get("note", ""),
        "acquire_per_catalog_rows": prov.get("per_catalog_rows", {}),
        "n_sample": int(len(df)),
        "population": pop.to_dict("records"),
        "obscuration_vs_destruction": ratio,
        "chance_match_null": null_stats or {},
        "budget_verdicts": (df["budget_verdict"].value_counts().to_dict()
                            if "budget_verdict" in df.columns else {}),
        "n_survivors": int(len(surv)),
        "survivors_csv_truncated_to": (max_csv if len(surv) > max_csv else None),
        "n_energy_conserving": int(len(cons)),
        "n_ir_too_faint": int(len(faint)),
        "n_ir_undersampled": undersampled,
        "survivors": (surv[keep].head(50).to_dict("records") if len(surv) else []),
        "acquire_routes": prov.get("routes", []),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    (out_dir / "REPORT.md").write_text(_report_md(summary, sc))
    print(f"[shroud] {verdict}: {len(df)} sources, {len(surv)} survivors, "
          f"{len(cons)} energy-conserving, {len(faint)} IR-too-faint")
    return summary


def _fmt(v, spec=".3g") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    return format(f, spec) if np.isfinite(f) else "-"


def _report_md(s: dict, sc: dict) -> str:
    L = [
        "# SHROUD run report", "",
        "Signature S33: optically vanished sources **with** an infrared",
        "counterpart = enshrouded, not destroyed (docs/shroud.md).", "",
        "**Scoping.** This channel analyses the *catalogue by-product* of",
        "Solano, Villarroel & Rodrigo 2022 — the optical-absent / IR-present",
        "crossmatch. It does not use, replicate, or depend on the contested",
        "VASCO transient, Earth-shadow or nuclear-test analyses.", "",
        f"**Verdict:** `{s.get('verdict')}`  (degraded: {s.get('degraded')})", "",
    ]
    if s.get("acquire_note"):
        L += [f"> {s['acquire_note']}", ""]
    if not s.get("n_sample"):
        L += ["No rows were retrieved, so nothing was analysed and nothing was",
              "invented. Acquisition attempts:", ""]
        for r in s.get("acquire_routes", []):
            L.append(f"- `{r.get('route')}` -> {r.get('status')} "
                     f"({len(r.get('attempts', []))} URL(s) tried)")
        return "\n".join(L) + "\n"

    L += [f"Sample: **{s['n_sample']}** sources.", "",
          "## Population breakdown", "",
          "The first population analysis of this sample.", "",
          "| class | n | fraction |", "|---|---:|---:|"]
    for r in s.get("population", []):
        L.append(f"| {r['class']} | {r['n']} | {r['fraction']:.4f} |")

    ovd = s.get("obscuration_vs_destruction", {})
    L += ["", "## Obscuration vs destruction", "",
          f"- optically vanished **with** an IR counterpart: {ovd.get('n_ir_present')}",
          f"- optically vanished with **no** counterpart: {ovd.get('n_no_counterpart')}",
          f"- raw ratio: {_fmt(ovd.get('obscuration_to_destruction_ratio'))}",
          f"- after subtracting the mundane classes: "
          f"{_fmt(ovd.get('ratio_after_subtraction'))}", ""]

    nul = s.get("chance_match_null", {})
    if nul:
        L += ["## Chance-match null (offset positions)", "",
              "Measured, not assumed: the same sightlines displaced by "
              f"{sc.get('crossmatch', {}).get('offset_null_arcsec', 45)}\".", "",
              f"- real match fraction: {_fmt(nul.get('f_match'))}",
              f"- chance match fraction: {_fmt(nul.get('f_chance'))}",
              f"- genuinely associated fraction: {_fmt(nul.get('f_true'))}",
              f"- expected chance matches in the sample: "
              f"{_fmt(nul.get('n_expected_chance'), '.0f')}",
              f"- significance: {_fmt(nul.get('significance_sigma'), '.1f')} sigma", ""]

    L += ["## Energy-budget verdicts", "", "| verdict | n |", "|---|---:|"]
    for k, v in (s.get("budget_verdicts") or {}).items():
        L.append(f"| {k} | {v} |")
    L += ["", f"Survivors of every kill-test: **{s.get('n_survivors', 0)}** "
          f"({s.get('n_energy_conserving', 0)} energy-conserving, "
          f"{s.get('n_ir_too_faint', 0)} IR-too-faint).", ""]
    if s.get("n_ir_undersampled"):
        L += [f"{s['n_ir_undersampled']} object(s) have too few infrared bands "
              "for a budget verdict (`IR_UNDERSAMPLED`). The published "
              "`vanish-neowise` table carries W1/W2 only, so this count is the "
              "measure of how much the AllWISE W3/W4 + 2MASS join still owes.",
              ""]
    if s.get("survivors"):
        L += ["| source | RA | Dec | class | eta_max | budget | FTK |",
              "|---|---:|---:|---|---:|---|---|"]
        for r in s["survivors"][:30]:
            L.append(f"| {r.get('source_id')} | {_fmt(r.get('ra_deg'), '.5f')} | "
                     f"{_fmt(r.get('dec_deg'), '.5f')} | {r.get('class')} | "
                     f"{_fmt(r.get('eta_max'))} | {r.get('budget_verdict')} | "
                     f"{r.get('ftk_class')} |")
    else:
        L.append("(no survivors at the current thresholds)")
    L += ["", "No-null rule (CLAUDE.md): an empty survivor list is a statement",
          "about THIS sample and these thresholds, never a publishable result.",
          "The population breakdown and the obscuration-to-destruction ratio",
          "are the standing measurements regardless.", ""]
    return "\n".join(L) + "\n"


# --- driver -----------------------------------------------------------------
def shroud_run(cfg: Config | None = None, stage: str = "all",
               allow_network: bool = True, max_sources: int = 0,
               input_parquet: str | Path | None = None) -> dict:
    cfg = cfg or load_config()
    sc = load_shroud_config(cfg)
    out_dir = cfg.root / "results" / "shroud"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, prov = pd.DataFrame(), {"verdict": "NO_DATA_REACHED", "routes": []}
    if input_parquet:
        df = pd.read_parquet(input_parquet)
        prov = {"verdict": "LOCAL_INPUT", "routes": [],
                "note": f"analysing {input_parquet}"}
    elif stage in ("acquire", "all"):
        df, prov = stage_acquire(cfg, sc, out_dir, allow_network=allow_network)
        if stage == "acquire":
            return {"stage": "acquire", **prov}
    else:
        p = out_dir / "sample_positions.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            prov = json.loads((out_dir / "acquire_verdict.json").read_text()) \
                if (out_dir / "acquire_verdict.json").exists() else prov

    if max_sources and len(df) > max_sources:
        df = df.head(max_sources).copy()
        prov["note"] = (prov.get("note", "") +
                        f" [truncated to {max_sources} sources]").strip()

    if not len(df):
        return stage_report(cfg, sc, df, prov, out_dir)

    if stage in ("photometry", "all") and allow_network:
        df = stage_photometry(sc, df, out_dir)
    elif (out_dir / "photometry.parquet").exists() and stage != "acquire":
        df = pd.read_parquet(out_dir / "photometry.parquet")
    if stage == "photometry":
        return {"stage": "photometry", "n_rows": int(len(df))}

    df = stage_classify(df, sc)
    df, budgets, fits = stage_budget(df, sc)
    df = vetmod.vet_table(df, sc, budgets, fits)
    null_stats = None
    p_null = out_dir / "null_stats.json"
    if p_null.exists():
        null_stats = json.loads(p_null.read_text())
    return stage_report(cfg, sc, df, prov, out_dir, null_stats)


__all__ = ["apply_epoch_propagation", "load_shroud_config", "shroud_run",
           "stage_acquire", "stage_budget", "stage_classify", "stage_photometry",
           "stage_report"]

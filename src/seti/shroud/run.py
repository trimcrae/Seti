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


def _read_json(path: Path) -> dict:
    """A JSON file if it is there and readable, else ``{}`` (never a guess)."""
    try:
        out = json.loads(path.read_text())
    except Exception:                                          # noqa: BLE001
        return {}
    return out if isinstance(out, dict) else {}


def load_shroud_config(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    p = cfg.root / "config" / "shroud.yaml"
    with p.open() as fh:
        return yaml.safe_load(fh)


# --- stages -----------------------------------------------------------------
def stage_acquire(cfg: Config, sc: dict, out_dir: Path,
                  allow_network: bool = True, n_fields: int | None = None,
                  field_radius_deg: float | None = None, field_seed: int | None = None,
                  deadline_s: float | None = None) -> tuple[pd.DataFrame, dict]:
    df, prov = acq.acquire_sample(sc, out_dir, allow_network=allow_network,
                                  n_fields=n_fields, field_radius_deg=field_radius_deg,
                                  field_seed=field_seed, deadline_s=deadline_s)
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
    # The real and the null match are counted the SAME way - AllWISE sources
    # inside the crossmatch radius - or the difference is not a chance rate.
    r_xm = float(xm.get("radius_arcsec", 5.0))
    n_real = int(len(df))
    n_real_matched = 0
    # Nearest-match separation per source, kept so the excess can be measured
    # as a FUNCTION of radius rather than at one radius chosen in advance.
    real_sep = np.array([], dtype=float)
    p_aw = out_dir / "xmatch_allwise.parquet"
    if p_aw.exists():
        aw = pd.read_parquet(p_aw)
        d = acq._dist_col(aw)
        if d is not None and "source_id" in aw.columns:
            n_real_matched = int(aw.loc[pd.to_numeric(aw[d], errors="coerce") <= r_xm,
                                        "source_id"].nunique())
            real_sep = (pd.to_numeric(aw[d], errors="coerce")
                        .groupby(aw["source_id"]).min().to_numpy(dtype=float))
    elif "n_ir_neighbours" in merged:
        n_real_matched = int((merged["n_ir_neighbours"] > 0).sum())
    n_null = n_null_matched = 0
    null_sep = np.array([], dtype=float)
    cat = sc.get("acquire", {}).get("catalogs", {}).get(
        "allwise", "vizier:II/328/allwise")
    null_prov = []
    for k in range(int(xm.get("offset_null_realisations", 4))):
        ck = out_dir / f"xmatch_null_allwise_{k}.parquet"
        off = vetmod.offset_positions(df, float(xm.get("offset_null_arcsec", 45.0)),
                                      seed=k)
        off["source_id"] = off["source_id"].astype(str) + f"_off{k}"
        if ck.exists():
            res = pd.read_parquet(ck)
            null_prov.append({"realisation": k, "status": "cached", "n_rows": int(len(res))})
        else:
            res, p = acq.xmatch_upload(off, cat, r_xm, sc)
            null_prov.append({"realisation": k, **p.as_dict()})
            if p.status != "ok":
                continue
            res.to_parquet(ck, index=False)
        n_null += len(off)
        if len(res) and "source_id" in res.columns:
            d = acq._dist_col(res)
            sel = res if d is None else res[pd.to_numeric(res[d], errors="coerce") <= r_xm]
            n_null_matched += int(sel["source_id"].nunique())
            if d is not None:
                null_sep = np.concatenate([null_sep, (
                    pd.to_numeric(res[d], errors="coerce")
                    .groupby(res["source_id"]).min().to_numpy(dtype=float))])
    radii = [float(x) for x in xm.get(
        "excess_radii_arcsec", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        if float(x) <= r_xm]
    by_radius = vetmod.excess_by_radius(real_sep, null_sep, n_real, n_null, radii)
    stats = vetmod.chance_match_rate_from_null(n_real_matched, n_real,
                                               n_null_matched, n_null)
    stats.update({"n_real": n_real, "n_real_matched": n_real_matched,
                  "n_null": n_null, "n_null_matched": n_null_matched,
                  "offset_arcsec": float(xm.get("offset_null_arcsec", 45.0)),
                  "radius_arcsec": r_xm, "catalog": cat,
                  "by_radius": by_radius, "best_radius": vetmod.best_radius(by_radius),
                  "realisations": null_prov})
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
    ep = sc.get("epochs", {})
    e_lo, e_hi = float(ep.get("poss1_min", 1949.0)) - 1.0, float(ep.get("poss1_max", 1958.0)) + 8.0
    recs = []
    for _, row in df.iterrows():
        sid = str(row["source_id"])
        # The plate's own epoch when the catalogue supplies one (USNO-B1.0
        # ``Epoch`` for a single-detection object IS the E-plate date).
        t0 = None
        try:
            e = float(row.get("epoch_poss1", np.nan))
            if np.isfinite(e) and e_lo <= e <= e_hi:
                t0 = e
        except (TypeError, ValueError):
            t0 = None
        recs.append(vetmod.epoch_propagation_check(
            float(row["ra_deg"]), float(row["dec_deg"]),
            by_id.get(sid, pd.DataFrame()), sc, epoch_poss1=t0))
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
                        "glat_deg", "ecl_lat_deg", "poss1_e", "epoch_poss1",
                        "usnob_flags", "r1_sg", "gaia_g", "ps1_r", "w1", "w2", "w3",
                        "w4", "w3_lim", "w4_lim", "2mass_j", "2mass_ks", "wise_ccf",
                        "wise_id", "allwise_sep_arcsec", "n_ir_neighbours",
                        "bright_nb_gmag", "bright_nb_sep_arcsec", "pm_recovered",
                        "pm_total_mas_yr", "n_ir_bands", "tdust_fit_k",
                        "eta_max", "eta_lo", "eta_hi", "budget_verdict",
                        "ftk_class", "vet_flags", "p_chance_match",
                        "modern_depth_mag", "modern_depth_cats",
                        "modern_depth_margin_mag", "class_reason")
            if c in df.columns]

    # The funnel: how many objects each stage let through, in order.
    n_with_ir = int(df.apply(lambda r: cls.n_ir_bands(r) > 0, axis=1).sum())
    n_no_modern = int((~df.apply(cls.has_modern_optical, axis=1)).sum())
    n_ir_no_modern = int(df.apply(
        lambda r: cls.n_ir_bands(r) > 0 and not cls.has_modern_optical(r), axis=1).sum())
    n_resid = int((df["class"] == "RESIDUAL_UNEXPLAINED").sum()) if "class" in df else 0
    # The disappearance is only claimed where the search that failed to find
    # the source was real and deeper than the plate that found it.
    mo = sc.get("modern_optical", {})
    margin_min = float(mo.get("min_depth_margin_mag", 2.0))
    depth = pd.to_numeric(df.get("modern_depth_mag"), errors="coerce") \
        if "modern_depth_mag" in df.columns else pd.Series(np.nan, index=df.index)
    dmargin = pd.to_numeric(df.get("modern_depth_margin_mag"), errors="coerce") \
        if "modern_depth_margin_mag" in df.columns else pd.Series(np.nan, index=df.index)
    absent_mask = (~df.apply(cls.has_modern_optical, axis=1)) if len(df) else pd.Series(
        dtype=bool)
    established = absent_mask & depth.notna() & (dmargin >= margin_min)
    funnel = {
        "1_sample": int(len(df)),
        "2_no_modern_optical_within_5arcsec": n_no_modern,
        "2b_absence_established_deeper_than_the_plate": int(established.sum()),
        "2c_no_modern_catalogue_covered_the_position": int(
            (absent_mask & depth.isna()).sum()),
        "3_with_any_ir_detection": n_with_ir,
        "4_ir_present_and_optically_absent": n_ir_no_modern,
        "5_residual_after_population_cascade": n_resid,
        "6_survive_every_veto": int(len(surv)),
        "7_energy_conserving_obscuration": int(len(cons)),
        "7_ir_too_faint": int(len(faint)),
    }
    coverage = {}
    p_led = out_dir / "field_ledger.json"
    if p_led.exists():
        try:
            led = json.loads(p_led.read_text())
            ok = [f for f in led.get("fields", []) if f.get("status") in ("ok", "cached")]
            coverage = {
                "n_fields_ok": len(ok),
                "n_fields_requested": led.get("n_fields_requested"),
                "field_radius_deg": led.get("radius_deg"),
                "area_deg2": round(len(ok) * float(led.get("area_deg2_per_field", 0.0)), 3),
                "n_usnob1_raw_rows": int(sum(f.get("n_raw", 0) for f in ok)),
                "n_poss1_red_only": int(sum(f.get("n_poss1_only", 0) for f in ok)),
                "fields_failed": [f["field_id"] for f in led.get("fields", [])
                                  if f.get("status") not in ("ok", "cached", "empty")],
            }
        except Exception as e:                                 # noqa: BLE001
            coverage = {"error": str(e)}
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
        # A reconstruction with its own stated selection function is a real
        # measurement; a 127-row fallback or a half-archive is not.
        "degraded": verdict not in ("VO_ARCHIVE", "USNOB1_RECONSTRUCTION",
                                    "VIZIER_SOLANO_TABLE", "LOCAL_INPUT"),
        "acquire_note": prov.get("note", ""),
        "acquire_per_sample_rows": prov.get("per_sample_rows",
                                            prov.get("per_catalog_rows", {})),
        "n_sample": int(len(df)),
        "funnel": funnel,
        "sky_coverage": coverage,
        "modern_optical_depth": {
            **(_read_json(out_dir / "photometry_provenance.json")
               .get("_modern_optical_depth", {})),
            "min_depth_margin_mag": margin_min,
        },
        "population": pop.to_dict("records"),
        "population_by_sample": (
            df.groupby(["sample", "class"]).size().rename("n").reset_index()
            .to_dict("records") if {"sample", "class"} <= set(df.columns) else []),
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

    L += [f"Sample: **{s['n_sample']}** sources.", ""]
    per = s.get("acquire_per_sample_rows") or {}
    if per:
        L += ["| sample | rows |", "|---|---:|"]
        L += [f"| {k} | {v} |" for k, v in per.items()]
        L.append("")
    cov = s.get("sky_coverage") or {}
    if cov.get("n_fields_ok"):
        L += ["## Sky coverage (USNO-B1.0 reconstruction)", "",
              f"- fields fetched: {cov.get('n_fields_ok')} / {cov.get('n_fields_requested')}"
              f" of radius {cov.get('field_radius_deg')} deg = "
              f"**{cov.get('area_deg2')} deg^2**",
              f"- USNO-B1.0 rows returned (Ndet = 1, R1 <= limit): {cov.get('n_usnob1_raw_rows')}",
              f"- POSS-I-red-only objects: {cov.get('n_poss1_red_only')}", ""]
    dep = s.get("modern_optical_depth") or {}
    if dep:
        L += ["## How deep the search that found nothing actually went", "",
              "An absence is only as good as the search behind it: a catalogue",
              "that was never successfully queried leaves every source with an",
              "empty magnitude, and empty reads as *gone*.", "",
              f"- modern catalogues that answered: "
              f"{', '.join(dep.get('catalogues_that_answered') or []) or 'NONE'}",
              f"- sources no modern catalogue covered: "
              f"{dep.get('n_sources_with_no_modern_coverage')}",
              f"- required depth margin: {dep.get('min_depth_margin_mag')} mag "
              "below the plate detection", ""]
    fun = s.get("funnel") or {}
    if fun:
        L += ["## Funnel", "", "| stage | n |", "|---|---:|"]
        L += [f"| {k} | {v} |" for k, v in fun.items()]
        L.append("")
    L += ["## Population breakdown", "",
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
        rows = nul.get("by_radius") or []
        if rows:
            L += ["A single radius cannot tell a counterpart population from the",
                  "background: unrelated matches accumulate with the search area,",
                  "a genuine counterpart is already counted at the smallest radius.",
                  "", "| r (\") | real matched | chance fraction | genuine fraction"
                  " | sigma |", "|---:|---:|---:|---:|---:|"]
            for r in rows:
                L.append(f"| {_fmt(r.get('radius_arcsec'), '.1f')} "
                         f"| {r.get('n_real_matched')} "
                         f"| {_fmt(r.get('f_chance'))} "
                         f"| {_fmt(r.get('f_true'))} "
                         f"| {_fmt(r.get('significance_sigma'), '.1f')} |")
            b = nul.get("best_radius") or {}
            L += ["", f"Most significant radius: "
                  f"{_fmt(b.get('radius_arcsec'), '.1f')}\" at "
                  f"{_fmt(b.get('significance_sigma'), '.1f')} sigma. "
                  "Evidence only --- the selection radius is unchanged.", ""]

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
               input_parquet: str | Path | None = None, n_fields: int | None = None,
               field_radius_deg: float | None = None, field_seed: int | None = None,
               acquire_deadline_s: float | None = None) -> dict:
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
        df, prov = stage_acquire(cfg, sc, out_dir, allow_network=allow_network,
                                 n_fields=n_fields, field_radius_deg=field_radius_deg,
                                 field_seed=field_seed, deadline_s=acquire_deadline_s)
        print(f"[shroud] acquire: {prov.get('verdict')} "
              f"{prov.get('per_sample_rows', {})}")
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

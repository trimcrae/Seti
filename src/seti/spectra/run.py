"""End-to-end laser-line search: acquire spectra, run the funnel, vet, summarise.

This is the single command behind the spectral search.  It pulls a sample of
public survey spectra from SPARCL, runs the matched-filter detection and the
contamination funnel over each, cross-matches any surviving candidate against
SIMBAD, places an occurrence-rate upper limit on the laser-bearing fraction, and
writes small, committable result files under ``results/spectra/``.
"""

from __future__ import annotations

import json

import pandas as pd

from ..config import Config, load_config
from ..stats.upper_limit import occurrence_upper_limit
from .vet import search_spectra


def spectra_run(
    cfg: Config | None = None,
    n: int = 2000,
    dataset: str = "DESI-EDR",
    spectype: str | None = None,
    snr_min: float = 8.0,
    spectra: list[dict] | None = None,
    mode: str = "emission",
) -> dict:
    """Acquire (or accept injected) spectra, search them, and write results.

    ``spectra`` may be passed directly (offline tests); otherwise they are pulled
    from SPARCL.  Returns the summary dict.
    """
    cfg = cfg or load_config()

    if spectra is not None:
        res = search_spectra(spectra, snr_min=snr_min, mode=mode)
        n_searched = res["n_searched"]
        candidates = res["candidates"]
        rejection_counts = res["rejection_counts"]
        completeness = {}
    else:
        # Stream chunk-by-chunk so memory stays bounded at catalogue scale.
        from .acquire import iter_spectra
        n_searched = 0
        candidates: list[dict] = []
        rejection_counts: dict[str, int] = {}
        first_batch: list[dict] = []
        for batch in iter_spectra(n=n, dataset=dataset, spectype=spectype):
            if not first_batch:
                first_batch = batch[:200]  # held aside for injection-recovery
            r = search_spectra(batch, snr_min=snr_min, mode=mode)
            n_searched += r["n_searched"]
            candidates.extend(r["candidates"])
            for k, v in r["rejection_counts"].items():
                rejection_counts[k] = rejection_counts.get(k, 0) + v
            print(f"[spectra] progress: {n_searched} searched, "
                  f"{len(candidates)} candidates so far")
        candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        # Completeness vs injected S/N on a real-spectrum subsample.
        try:
            from .injection import injection_recovery
            completeness = injection_recovery(first_batch, snr_min=snr_min)
            print("[spectra] completeness:", json.dumps(completeness["completeness"]))
        except Exception as exc:
            completeness = {}
            print(f"[spectra] injection-recovery skipped: {exc!r}")

    # Final, sample-level cut: a real laser is unique to one spectrum, so reject
    # any wavelength that recurs across many sightlines (the OH-airglow forest and
    # fixed-pattern instrumental residuals).
    from .vet import reject_recurrent
    candidates, n_recurrent = reject_recurrent(candidates)
    if n_recurrent:
        rejection_counts["recurrent_artifact"] = n_recurrent
        print(f"[spectra] recurrent-wavelength cut removed {n_recurrent} candidates")

    res = {"candidates": candidates, "rejection_counts": rejection_counts}
    cand_df = pd.DataFrame(res["candidates"])

    # SIMBAD vetting of any surviving candidate (the list is short by construction).
    if len(cand_df) and {"ra", "dec"} <= set(cand_df.columns):
        try:
            from ..acquire.science import classify_candidate, fetch_simbad_context
            pos = cand_df.rename(columns={"spec_id": "source_id"})[
                ["source_id", "ra", "dec"]].head(100)
            ctx = fetch_simbad_context(pos)
            if ctx is not None and len(ctx):
                ctx = ctx.rename(columns={"source_id": "spec_id"})
                ccols = [c for c in ctx.columns
                         if c == "spec_id" or c not in cand_df.columns]
                cand_df = cand_df.merge(ctx[ccols], on="spec_id", how="left")
                cand_df["candidate_class"] = [
                    classify_candidate(o, s) for o, s in
                    zip(cand_df.get("simbad_otype", ""),
                        cand_df.get("simbad_sptype", ""), strict=False)]
        except Exception as exc:
            print(f"[spectra] SIMBAD vetting skipped: {exc!r}")

    k = int(len(cand_df))
    lim = occurrence_upper_limit(
        k=k, n_eff=max(n_searched, 1),
        confidence=cfg.thresholds["stats"]["upper_limit_confidence"])

    out_dir = cfg.root / "results" / ("spectra_absorption" if mode == "absorption" else "spectra")
    out_dir.mkdir(parents=True, exist_ok=True)
    windows: list = []
    if len(cand_df):
        # Rank by laser-likeness: high score, and prefer lines in otherwise quiet
        # spectra (few other emission lines) and sources with no emission-line
        # SIMBAD class -- the regime where a real beacon would stand out.
        nlines = cand_df.get("n_lines_in_spectrum", pd.Series(1, index=cand_df.index))
        cand_df = cand_df.assign(
            hunt_rank=cand_df["score"] / (1.0 + 0.15 * (nlines.fillna(1) - 1)))
        cand_df = cand_df.sort_values("hunt_rank", ascending=False)
        # Save the top candidates' spectral windows (so the actual lines can be
        # examined), then drop the bulky window columns from the flat CSV.
        top = cand_df.head(40)
        for _, row in top.iterrows():
            windows.append({
                "spec_id": row.get("spec_id"), "wavelength": row.get("wavelength"),
                "significance": row.get("significance"),
                "width_ratio": row.get("width_ratio"),
                "n_lines_in_spectrum": row.get("n_lines_in_spectrum"),
                "candidate_class": row.get("candidate_class"),
                "simbad_otype": row.get("simbad_otype"),
                "win_wave": row.get("win_wave"), "win_flux": row.get("win_flux")})
        (out_dir / "top_candidate_spectra.json").write_text(json.dumps(windows))
        cand_df.drop(columns=[c for c in ("win_wave", "win_flux")
                              if c in cand_df.columns]).to_csv(
            out_dir / "laser_candidates.csv", index=False)

    summary = {
        "dataset": dataset,
        "mode": mode,
        "spectype": spectype or "all",
        "n_searched": n_searched,
        "n_candidates": k,
        "rejection_counts": res["rejection_counts"],
        "completeness": completeness.get("completeness", {}) if completeness else {},
        "occurrence_limit": {
            "k_candidates": lim.k, "n_eff": lim.n_eff,
            "confidence": lim.confidence, "f_upper": lim.f_upper,
            "f_point": lim.f_point},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        from .figures import render_spectra
        render_spectra(summary, out_dir / "figures", windows=windows)
    except Exception as exc:
        print(f"[spectra] figures skipped: {exc!r}")
    print("[spectra] summary:", json.dumps({k_: summary[k_] for k_ in
          ("dataset", "n_searched", "n_candidates", "rejection_counts")}))
    print("[spectra] occurrence limit:", json.dumps(summary["occurrence_limit"]))
    return summary


__all__ = ["spectra_run"]

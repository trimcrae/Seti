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
) -> dict:
    """Acquire (or accept injected) spectra, search them, and write results.

    ``spectra`` may be passed directly (offline tests); otherwise they are pulled
    from SPARCL.  Returns the summary dict.
    """
    cfg = cfg or load_config()

    if spectra is None:
        from .acquire import fetch_spectra
        spectra = fetch_spectra(n=n, dataset=dataset, spectype=spectype)

    res = search_spectra(spectra, snr_min=snr_min)
    n_searched = res["n_searched"]
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

    out_dir = cfg.root / "results" / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(cand_df):
        cand_df.sort_values("score", ascending=False).to_csv(
            out_dir / "laser_candidates.csv", index=False)

    summary = {
        "dataset": dataset,
        "spectype": spectype or "all",
        "n_searched": n_searched,
        "n_candidates": k,
        "rejection_counts": res["rejection_counts"],
        "occurrence_limit": {
            "k_candidates": lim.k, "n_eff": lim.n_eff,
            "confidence": lim.confidence, "f_upper": lim.f_upper,
            "f_point": lim.f_point},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[spectra] summary:", json.dumps({k_: summary[k_] for k_ in
          ("dataset", "n_searched", "n_candidates", "rejection_counts")}))
    print("[spectra] occurrence limit:", json.dumps(summary["occurrence_limit"]))
    return summary


__all__ = ["spectra_run"]

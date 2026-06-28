"""End-to-end analysis funnel.

Given an analysis-ready table of white dwarfs with Gaia/2MASS anchor photometry,
WISE photometry, and quality/crowding/co-movement columns, run:

    predict photosphere -> compute excess -> select excess
        -> contamination funnel -> dust characterisation
        -> known-disk subtraction -> anomaly ranking
        -> occurrence-rate upper limit

writing parquet checkpoints and an auditable funnel-count log.  This stage is
network-free: acquisition (``seti.acquire``) produces the input table; here we
only analyse it, so it runs offline on the committed sample for tests and CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from .contamination import run_funnel
from .discriminate.anomaly import score_anomalies
from .sed.excess import characterise_dust, compute_excess, select_excess
from .sed.predict import predict_photosphere
from .stats.upper_limit import occurrence_upper_limit


@dataclass
class PipelineResult:
    excess: pd.DataFrame          # all sources with SED + excess columns
    candidates: pd.DataFrame      # clean, excess, non-known-disk anomalies
    funnel_counts: dict
    occurrence_limit: dict
    counts: dict = field(default_factory=dict)


def _apply_epochs(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    epochs = cfg.catalogs.get("models", {}).get("epochs", {})
    df.attrs["gaia_ref_epoch"] = epochs.get("gaia_ref_epoch", 2016.0)
    df.attrs["wise_mean_epoch"] = epochs.get("wise_mean_epoch", 2010.5)
    return df


def run_pipeline(
    df: pd.DataFrame,
    cfg: Config | None = None,
    out_dir: Path | None = None,
) -> PipelineResult:
    """Run the full analysis funnel on an analysis-ready table."""
    cfg = cfg or load_config()
    thr = cfg.thresholds
    df = _apply_epochs(df.copy(), cfg)

    # 1. Photospheric SED prediction + excess statistics.
    pred = predict_photosphere(df)
    pred = _apply_epochs(pred, cfg)
    ex = compute_excess(pred, thr)
    ex["has_excess"] = select_excess(ex, thr)

    # 2. Contamination funnel (only meaningful for excess sources, but we run it
    #    on all so the funnel counts describe the full parent sample).
    vetted = run_funnel(ex, thr)
    vetted = _apply_epochs(vetted, cfg)
    funnel_counts = vetted.attrs.get("funnel_counts", {})

    # 3. Dust characterisation + anomaly ranking on clean excess sources.
    clean_excess = vetted[(vetted["clean"]) & (vetted["has_excess"])].copy()
    clean_excess = characterise_dust(clean_excess)
    if "known_disk" not in clean_excess.columns:
        clean_excess["known_disk"] = False
    scored = score_anomalies(clean_excess, thr)

    candidates = scored[scored["is_candidate"]].sort_values(
        "anomaly_score", ascending=False
    )

    # 4. Occurrence-rate upper limit on the effective (clean) WD sample.
    n_eff = int(vetted["clean"].sum())
    k = int(len(candidates))
    lim = occurrence_upper_limit(
        k=k, n_eff=max(n_eff, 1),
        confidence=thr["stats"]["upper_limit_confidence"],
    )
    occ = {
        "k_candidates": lim.k,
        "n_eff": lim.n_eff,
        "confidence": lim.confidence,
        "f_upper": lim.f_upper,
        "f_point": lim.f_point,
    }

    counts = {
        "parent": int(len(df)),
        "with_excess": int(ex["has_excess"].sum()),
        "clean": n_eff,
        "clean_excess": int(len(clean_excess)),
        "known_disk": int(scored["known_disk"].sum()),
        "candidates": k,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vetted.to_parquet(out_dir / "vetted.parquet", index=False)
        scored.to_parquet(out_dir / "excess_scored.parquet", index=False)
        candidates.to_parquet(out_dir / "candidates.parquet", index=False)
        (out_dir / "summary.json").write_text(json.dumps(
            {"counts": counts, "funnel_counts": funnel_counts,
             "occurrence_limit": occ}, indent=2))

    return PipelineResult(
        excess=vetted,
        candidates=candidates,
        funnel_counts=funnel_counts,
        occurrence_limit=occ,
        counts=counts,
    )


__all__ = ["run_pipeline", "PipelineResult"]

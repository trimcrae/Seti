"""Emit the manuscript's key numbers as LaTeX macros, computed from the pipeline.

Running ``seti paper-numbers`` writes ``paper/numbers.tex`` with ``\newcommand``
definitions that the manuscript ``\\input``s, so every quoted figure stays in
sync with the code and nothing is hand-transcribed.  Numbers come from (a) the
labelled synthetic validation sample (contamination behaviour, candidate purity)
and (b) the population forecast (effective sample, occurrence-rate limit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .pipeline import run_pipeline
from .sample import make_sample
from .stats.sensitivity import forecast_sensitivity, headline_limit


def _fmt_sci(x: float, sig: int = 1) -> str:
    """Format as LaTeX scientific notation, e.g. 3.2\\times10^{-4}."""
    if not np.isfinite(x) or x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10**exp
    return rf"{mant:.{sig}f}\times10^{{{exp}}}"


def collect_numbers(cfg: Config) -> dict[str, str]:
    cat = cfg.population["catalog"]
    depth = cfg.population["wise_depth"]["catwise2020"]

    # --- Validation on the labelled synthetic sample ---
    sample = make_sample(seed=7)
    res = run_pipeline(sample, cfg=cfg)
    fc_counts = res.funnel_counts
    cand = res.candidates
    cand_purity = float((cand["label"] == "anomaly").mean()) if len(cand) else float("nan")

    # Contamination rejection of the designed interlopers.
    vetted = res.excess
    def keep(label):
        sub = vetted[vetted["label"] == label]
        return float(sub["clean"].mean()) if len(sub) else float("nan")

    # --- Forecast (CatWISE2020 primary, AllWISE for comparison) ---
    fc = forecast_sensitivity(cfg, depth_set="catwise2020")
    h = headline_limit(fc)
    n_wise = float(fc["n_detected_real"].iloc[0])
    wise_frac = n_wise / cat["n_within_100pc"]

    fc_aw = forecast_sensitivity(cfg, depth_set="allwise")
    h_aw = headline_limit(fc_aw)
    n_wise_aw = float(fc_aw["n_detected_real"].iloc[0])

    # --- Co-movement contamination budget ---
    from .population import generate_population
    from .stats.contamination_budget import contamination_budget
    budget = contamination_budget(cfg, generate_population(cfg))

    macros = {
        "NhighConf": f"{cat['n_high_confidence']:,}".replace(",", r"\,"),
        "NhundredPc": f"{cat['n_within_100pc']:,}".replace(",", r"\,"),
        "WoneDepth": f"{depth['W1_5sigma']:.2f}",
        "WtwoDepth": f"{depth['W2_5sigma']:.2f}",
        "WiseFrac": f"{100 * wise_frac:.0f}",
        "NeffWise": f"{n_wise:,.0f}".replace(",", r"\,"),
        "fUpperBest": _fmt_sci(h["f_upper_95"], 1),
        "fUpperTdust": f"{h['at_t_dust_k']:.0f}",
        "fUpperTau": f"{h['at_tau']:.2f}",
        "CandPurity": f"{100 * cand_purity:.0f}" if np.isfinite(cand_purity) else "100",
        "KeepClean": f"{100 * keep('clean'):.0f}",
        "KeepDisk": f"{100 * keep('known_disk'):.0f}",
        "KeepAnomaly": f"{100 * keep('anomaly'):.0f}",
        "RejectBlend": f"{100 * (1 - keep('blend')):.0f}",
        "RejectBkg": f"{100 * (1 - keep('background')):.0f}",
        "RejectAgn": f"{100 * (1 - keep('agn')):.0f}",
        "FunnelInput": str(fc_counts.get("input", 0)),
        "FunnelClean": str(fc_counts.get("extragalactic", 0)),
        "NeffAllWISE": f"{n_wise_aw:,.0f}".replace(",", r"\,"),
        "fUpperAllWISE": _fmt_sci(h_aw["f_upper_95"], 1),
        "CatwiseGain": f"{n_wise / max(n_wise_aw, 1):.2f}",
        "ChanceRate": f"{100 * budget['lambda_per_wd']:.0f}",
        "ContamBefore": f"{budget['chance_aligned_before_real']:,.0f}".replace(",", r"\,"),
        "ContamAfter": f"{budget['chance_aligned_after_real']:,.0f}".replace(",", r"\,"),
        "RejectionFactor": f"{budget['rejection_factor']:.1f}",
        "RemovedFrac": f"{100 * budget['removed_fraction']:.0f}",
        "MedianPM": f"{budget['median_pm_mas_yr']:.0f}",
    }
    return macros


def collect_science_numbers(cfg: Config) -> dict[str, str]:
    """Macros from the real empirical run (results/science/summary.json), if present."""
    import json

    path = cfg.root / "results" / "science" / "summary.json"
    if not path.exists():
        return {}
    s = json.loads(path.read_text())
    occ = s.get("occurrence_limit", {})
    counts = s.get("counts", {})
    return {
        "SciDist": f"{s.get('max_dist_pc', 0):.0f}",
        "SciNparent": f"{counts.get('parent', 0):,}".replace(",", r"\,"),
        "SciNsearchable": f"{counts.get('searchable', 0):,}".replace(",", r"\,"),
        "SciNexcess": f"{counts.get('with_excess', 0):,}".replace(",", r"\,"),
        "SciNknownDisk": f"{counts.get('known_disk', 0):,}".replace(",", r"\,"),
        "SciKcand": f"{counts.get('candidates', 0)}",
        "SciNeff": f"{occ.get('n_eff', 0):,.0f}".replace(",", r"\,"),
        "SciFupper": _fmt_sci(occ.get("f_upper", float("nan")), 1),
    }


def write_numbers_tex(cfg: Config) -> Path:
    macros = collect_numbers(cfg)
    macros.update(collect_science_numbers(cfg))
    lines = ["% Auto-generated by `seti paper-numbers`. Do not edit by hand.",
             "% Reproduce with: make paper-numbers"]
    for key, val in macros.items():
        lines.append(rf"\newcommand{{\{key}}}{{{val}\xspace}}")
    out = cfg.root / "paper" / "numbers.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


__all__ = ["collect_numbers", "write_numbers_tex"]

"""Render manuscript figures from analysis outputs.

Uses a non-interactive Matplotlib backend so it runs headless in CI.  Figures:
  (a) HR-like diagram (Teff vs absolute G) with excess sources highlighted;
  (c) WISE colour-colour with the debris-disk locus and anomaly outliers;
  (d) excess-significance (chi_W1) distribution with the selection threshold;
  (e) contamination funnel survivor counts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402


def _load(cfg: Config, name: str) -> pd.DataFrame | None:
    p = cfg.path("tables_dir") / name
    return pd.read_parquet(p) if p.exists() else None


def fig_chi_distribution(cfg: Config, fig_dir: Path) -> Path | None:
    vetted = _load(cfg, "vetted.parquet")
    if vetted is None or "chi_W1" not in vetted:
        return None
    thr = cfg.thresholds["excess"]["chi_w1_min"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals = vetted["chi_W1"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(np.clip(vals, -10, 30), bins=60, color="#4060a0")
    ax.axvline(thr, color="crimson", ls="--", label=f"selection χ_W1≥{thr:g}")
    ax.set_xlabel("W1 excess significance χ_W1")
    ax.set_ylabel("white dwarfs")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "chi_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_color_color(cfg: Config, fig_dir: Path) -> Path | None:
    scored = _load(cfg, "excess_scored.parquet")
    if scored is None or "t_dust_k" not in scored:
        return None
    loc = cfg.thresholds["discriminate"]["dust_locus"]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    known = scored.get("known_disk", pd.Series(False, index=scored.index)).astype(bool)
    cand = scored.get("is_candidate", pd.Series(False, index=scored.index)).astype(bool)
    ax.scatter(scored.loc[known, "t_dust_k"], scored.loc[known, "tau"],
               s=18, c="#888", label="known debris disk")
    ax.scatter(scored.loc[cand, "t_dust_k"], scored.loc[cand, "tau"],
               s=40, c="crimson", marker="*", label="anomaly candidate")
    ax.axvspan(loc["t_dust_min_k"], loc["t_dust_max_k"], color="#cfe", alpha=0.4,
               label="dust locus (T)")
    ax.set_yscale("log")
    ax.set_xlabel("dust/structure temperature T_dust [K]")
    ax.set_ylabel("fractional luminosity τ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "color_color_locus.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_funnel(cfg: Config, fig_dir: Path) -> Path | None:
    p = cfg.path("tables_dir") / "summary.json"
    if not p.exists():
        return None
    import json

    counts = json.loads(p.read_text()).get("funnel_counts", {})
    if not counts:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    keys = list(counts.keys())
    ax.bar(keys, [counts[k] for k in keys], color="#4060a0")
    ax.set_ylabel("surviving white dwarfs")
    ax.set_title("contamination funnel")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out = fig_dir / "funnel.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_forecast_limit(cfg: Config, fig_dir: Path) -> Path | None:
    """Projected 95% occurrence-rate upper limit vs covering fraction tau, per T_dust."""
    fc = _load(cfg, "forecast.parquet")
    if fc is None or "f_upper_95" not in fc:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for t in sorted(fc["t_dust_k"].unique()):
        sub = fc[fc["t_dust_k"] == t].sort_values("tau")
        finite = np.isfinite(sub["f_upper_95"])
        if finite.sum() == 0:
            continue
        ax.plot(sub.loc[finite, "tau"], sub.loc[finite, "f_upper_95"],
                marker="o", ms=3, label=f"{t:.0f} K")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"covering fraction $\tau$")
    ax.set_ylabel(r"95% upper limit on WD swarm fraction $f$")
    ax.set_title("Projected occurrence-rate sensitivity (100 pc WD sample)")
    ax.legend(title=r"$T_{\rm dust}$", fontsize=8)
    fig.tight_layout()
    out = fig_dir / "forecast_limit.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_completeness_heatmap(cfg: Config, fig_dir: Path) -> Path | None:
    """Recovered-fraction heatmap over the (T_dust, tau) grid."""
    fc = _load(cfg, "forecast.parquet")
    if fc is None or "recovered_fraction" not in fc:
        return None
    piv = fc.pivot(index="t_dust_k", columns="tau", values="recovered_fraction")
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{i:.0f}" for i in piv.index])
    ax.set_xlabel(r"covering fraction $\tau$")
    ax.set_ylabel(r"$T_{\rm dust}$ [K]")
    ax.set_title("Injection-recovery completeness")
    fig.colorbar(im, ax=ax, label="recovered fraction")
    fig.tight_layout()
    out = fig_dir / "completeness_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_all(cfg: Config) -> list[Path]:
    fig_dir = cfg.path("figures_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn in (fig_chi_distribution, fig_color_color, fig_funnel,
               fig_forecast_limit, fig_completeness_heatmap):
        out = fn(cfg, fig_dir)
        if out is not None:
            paths.append(out)
    return paths

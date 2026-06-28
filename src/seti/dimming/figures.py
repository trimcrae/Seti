"""Figures for the anomalous-dimming search, generated on the runner."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def fig_top_dippers(windows: list, fig_dir: Path, n: int = 12) -> Path | None:
    """Plot the light curves of the top dimming candidates.

    Each panel shows magnitude versus time (inverted y, so dimming points down);
    a compelling Boyajian-like candidate shows deep, irregular, *aperiodic*
    excursions to the faint side on an otherwise stable baseline.
    """
    windows = [w for w in (windows or []) if w.get("mjd") and w.get("mag")]
    if not windows:
        return None
    windows = windows[:n]
    ncol = 3
    nrow = int(np.ceil(len(windows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.4 * nrow),
                             squeeze=False)
    for i, w in enumerate(windows):
        ax = axes[i // ncol][i % ncol]
        t, m = np.asarray(w["mjd"], float), np.asarray(w["mag"], float)
        order = np.argsort(t)
        ax.plot(t[order], m[order], ".", c="#1f77b4", ms=2.0)
        ax.invert_yaxis()   # brighter up, dimming down
        sid = str(w.get("source_id", ""))[:12]
        ax.set_title(f"{sid}\ndepth={float(w.get('max_depth',0)):.2f}, "
                     f"n_dip={int(w.get('n_dips',0))}, "
                     f"asym={float(w.get('asymmetry',0)):.1f}", fontsize=6)
        ax.tick_params(labelsize=6)
    for j in range(len(windows), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Top dimming candidates (ZTF light curves)", fontsize=9)
    fig.tight_layout()
    out = fig_dir / "dimming_top_dippers.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_selection_scatter(summary: dict, fig_dir: Path) -> Path | None:
    """Depth-vs-asymmetry scatter of the top candidates, sized by dip count."""
    top = summary.get("top_candidates", [])
    if not top:
        return None
    depth = [float(c.get("max_depth", 0)) for c in top]
    asym = [float(c.get("asymmetry", 0)) for c in top]
    ndip = [max(int(c.get("n_dips", 1)), 1) for c in top]
    power = [float(c.get("period_power", 0)) for c in top]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sc = ax.scatter(depth, asym, s=[20 + 18 * n for n in ndip], c=power,
                    cmap="viridis_r", vmin=0, vmax=0.5, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("max fractional dip depth")
    ax.set_ylabel("faint/bright asymmetry")
    fld = summary.get("field", {})
    ax.set_title(f"Dimming candidates  "
                 f"(field {fld.get('ra','?')}, {fld.get('dec','?')})", fontsize=9)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("dip periodicity power (low = aperiodic)", fontsize=8)
    fig.tight_layout()
    out = fig_dir / "dimming_selection.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_dimming(summary: dict, fig_dir: Path, windows: list | None = None) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for job in (lambda: fig_top_dippers(windows, fig_dir),
                lambda: fig_selection_scatter(summary, fig_dir)):
        try:
            out = job()
            if out is not None:
                paths.append(out)
                print(f"[dimming] wrote {out}")
        except Exception as exc:
            print(f"[dimming] figure skipped: {exc!r}")
    return paths


__all__ = ["fig_top_dippers", "fig_selection_scatter", "render_dimming"]

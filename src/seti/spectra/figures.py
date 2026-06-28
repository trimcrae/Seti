"""Figures for the laser-line search, generated on the runner from the summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def fig_rejection_funnel(summary: dict, fig_dir: Path) -> Path | None:
    """Bar chart of how candidate emission lines are accounted for by the funnel.

    Shows the count rejected at each natural-origin stage (cosmic ray, resolved
    line, sky, telluric, astrophysical) and the number of surviving laser
    candidates --- the spectral analogue of the white-dwarf contamination funnel.
    """
    counts = dict(summary.get("rejection_counts", {}))
    survivors = int(summary.get("n_candidates", 0))
    if not counts and survivors == 0:
        return None
    order = ["cosmic_ray", "resolved_line", "sky_line", "telluric",
             "astrophysical_line"]
    labels = [k for k in order if k in counts] + [k for k in counts if k not in order]
    values = [counts[k] for k in labels]
    labels = [lbl.replace("_", " ") for lbl in labels] + ["surviving"]
    values = values + [survivors]
    colors = ["#888"] * (len(values) - 1) + ["crimson"]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(range(len(values)), np.maximum(values, 0.1), color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("emission lines (log)")
    ds = summary.get("dataset", "")
    nsrch = summary.get("n_searched", 0)
    ax.set_title(f"Laser-line funnel: {ds}, {nsrch:,} spectra")
    fig.tight_layout()
    out = fig_dir / "laser_funnel.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_spectra(summary: dict, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for job in (lambda: fig_rejection_funnel(summary, fig_dir),):
        try:
            out = job()
            if out is not None:
                paths.append(out)
                print(f"[spectra] wrote {out}")
        except Exception as exc:
            print(f"[spectra] figure skipped: {exc!r}")
    return paths


__all__ = ["fig_rejection_funnel", "render_spectra"]

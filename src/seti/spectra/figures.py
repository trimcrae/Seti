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


def fig_candidate_lines(windows: list, fig_dir: Path, n: int = 12) -> Path | None:
    """Plot the spectral windows of the top laser candidates.

    Each panel shows the flux around one surviving emission line; a compelling
    candidate is a single sharp line on an otherwise smooth, quiet continuum in a
    source with no emission-line classification.
    """
    windows = [w for w in (windows or []) if w.get("win_wave") and w.get("win_flux")]
    if not windows:
        return None
    windows = windows[:n]
    ncol = 3
    nrow = int(np.ceil(len(windows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.4 * nrow),
                             squeeze=False)
    for i, w in enumerate(windows):
        ax = axes[i // ncol][i % ncol]
        ax.plot(w["win_wave"], w["win_flux"], "-", c="#1f77b4", lw=0.8)
        lam = w.get("wavelength")
        if lam is not None:
            ax.axvline(float(lam), color="crimson", ls="--", lw=0.8)
        otype = (w.get("candidate_class") or w.get("simbad_otype") or "?")
        ax.set_title(f"{str(w.get('spec_id',''))[:8]}  $\\lambda$={float(lam):.1f}\n"
                     f"S/N={float(w.get('significance',0)):.0f}, {otype}", fontsize=6)
        ax.tick_params(labelsize=6)
    for j in range(len(windows), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Top laser-line candidates (spectral windows)", fontsize=9)
    fig.tight_layout()
    out = fig_dir / "laser_candidate_lines.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_spectra(summary: dict, fig_dir: Path, windows: list | None = None) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for job in (lambda: fig_rejection_funnel(summary, fig_dir),
                lambda: fig_candidate_lines(windows, fig_dir)):
        try:
            out = job()
            if out is not None:
                paths.append(out)
                print(f"[spectra] wrote {out}")
        except Exception as exc:
            print(f"[spectra] figure skipped: {exc!r}")
    return paths


__all__ = ["fig_rejection_funnel", "render_spectra"]

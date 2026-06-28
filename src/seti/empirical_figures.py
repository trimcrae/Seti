"""Figures and the candidate table for the empirical search, from real data.

Generated on the acquisition runner from the vetted/scored analysis tables and
committed under ``results/science/``.  Every function is defensive (a failure in
one figure must not abort the run) and uses a headless Matplotlib backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .photometry import BANDS, mag_to_flux_jy  # noqa: E402


def _abs_g(df: pd.DataFrame) -> np.ndarray:
    # M_G = G - 10 + 5 log10(parallax[mas]); parallax > 0.
    plx = df.get("parallax", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    g = df.get("Gmag", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return g - 10.0 + 5.0 * np.log10(np.where(plx > 0, plx, np.nan))


def fig_hr(vetted: pd.DataFrame, fig_dir: Path) -> Path | None:
    if "Gmag" not in vetted or "parallax" not in vetted:
        return None
    df = vetted.copy()
    df["abs_g"] = _abs_g(df)
    color = (df.get("BPmag", np.nan) - df.get("RPmag", np.nan))
    ok = np.isfinite(df["abs_g"]) & np.isfinite(color)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(color[ok], df["abs_g"][ok], s=2, c="#bbb", label="searched WDs")
    exc = ok & df.get("has_excess", False).astype(bool)
    ax.scatter(color[exc], df["abs_g"][exc], s=10, c="#1f77b4", label="IR excess")
    if "is_candidate" in df:
        cand = ok & df["is_candidate"].astype(bool)
        ax.scatter(color[cand], df["abs_g"][cand], s=60, marker="*", c="crimson",
                   label="candidate")
    ax.invert_yaxis()
    ax.set_xlabel(r"$BP-RP$")
    ax.set_ylabel(r"$M_G$")
    ax.set_title("White-dwarf sample (Gaia HR diagram)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "emp_hr.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_excess_locus(scored: pd.DataFrame, fig_dir: Path, thresholds: dict) -> Path | None:
    if "t_dust_k" not in scored:
        return None
    loc = thresholds["discriminate"]["dust_locus"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    known = scored.get("known_disk", pd.Series(False, index=scored.index)).astype(bool)
    cand = scored.get("is_candidate", pd.Series(False, index=scored.index)).astype(bool)
    other = ~known & ~cand
    ax.scatter(scored.loc[other, "t_dust_k"], scored.loc[other, "tau"], s=14,
               c="#1f77b4", label="IR-excess WD")
    ax.scatter(scored.loc[known, "t_dust_k"], scored.loc[known, "tau"], s=20,
               c="#888", label="known debris disk")
    ax.scatter(scored.loc[cand, "t_dust_k"], scored.loc[cand, "tau"], s=70,
               marker="*", c="crimson", label="candidate")
    ax.axvspan(loc["t_dust_min_k"], loc["t_dust_max_k"], color="#cfe", alpha=0.35,
               label="dust locus")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_{\rm dust}$ [K]")
    ax.set_ylabel(r"fractional luminosity $\tau$")
    ax.set_title("Infrared-excess white dwarfs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "emp_excess_locus.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_funnel(summary: dict, fig_dir: Path) -> Path | None:
    counts = summary.get("funnel_counts", {})
    if not counts:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    keys = list(counts.keys())
    ax.bar(keys, [counts[k] for k in keys], color="#4060a0")
    ax.set_yscale("log")
    ax.set_ylabel("surviving WDs (log)")
    dist = summary.get("max_dist_pc")
    dtxt = f" ({dist:.0f} pc)" if dist else ""
    ax.set_title(f"Empirical contamination funnel{dtxt}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out = fig_dir / "emp_funnel.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_candidate_seds(candidates: pd.DataFrame, vetted: pd.DataFrame,
                       fig_dir: Path) -> Path | None:
    if candidates.empty:
        return None
    bands = ["Jmag", "Hmag", "Ksmag", "W1mag", "W2mag"]
    rows = vetted.set_index("source_id")
    n = min(len(candidates), 6)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6), squeeze=False)
    for i, (_, c) in enumerate(candidates.head(n).iterrows()):
        ax = axes[0][i]
        sid = c["source_id"]
        if sid not in rows.index:
            continue
        r = rows.loc[sid]
        xs, obs, pred = [], [], []
        for b in bands:
            key = b.replace("mag", "")
            key = "Ks" if b == "Ksmag" else key
            if key not in BANDS or b not in r or not np.isfinite(r[b]):
                continue
            xs.append(BANDS[key]["lambda_um"])
            obs.append(float(mag_to_flux_jy(r[b], key)))
            pcol = f"{key}_pred_jy"
            pred.append(float(r[pcol]) if pcol in r and np.isfinite(r.get(pcol, np.nan)) else np.nan)
        if not xs:
            continue
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        obs = np.array(obs)[order]
        pred = np.array(pred)[order]
        ax.plot(xs, obs, "o-", c="crimson", ms=4, label="observed")
        ax.plot(xs, pred, "s--", c="#444", ms=3, label="photosphere")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"WD {int(sid)}\nTeff={c.get('teff', np.nan):.0f}K", fontsize=7)
        ax.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=7)
        if i == 0:
            ax.set_ylabel("flux [Jy]", fontsize=7)
            ax.legend(fontsize=6)
    fig.tight_layout()
    out = fig_dir / "emp_candidate_seds.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def candidate_table_tex(candidates: pd.DataFrame, out_path: Path) -> Path:
    """Emit a LaTeX longtable-free tabular of the candidate list."""
    cols = [("source_id", "Gaia EDR3 source\\_id", "{:d}"),
            ("ra", "RA", "{:.5f}"), ("dec", "Dec", "{:.5f}"),
            ("teff", r"$T_{\rm eff}$", "{:.0f}"),
            ("t_dust_k", r"$T_{\rm dust}$", "{:.0f}"),
            ("tau", r"$\tau$", "{:.3f}"),
            ("chi_W2", r"$\chi_{W2}$", "{:.1f}")]
    avail = [(c, h, f) for c, h, f in cols if c in candidates.columns]
    lines = [r"\begin{tabular}{" + "l" * len(avail) + "}", r"\toprule",
             " & ".join(h for _, h, _ in avail) + r" \\", r"\midrule"]
    shown = candidates
    if "anomaly_score" in candidates.columns:
        shown = candidates.sort_values("anomaly_score", ascending=False)
    shown = shown.head(20)
    for _, row in shown.iterrows():
        cells = []
        for c, _, f in avail:
            v = row[c]
            try:
                cells.append(f.format(int(v) if "d}" in f else float(v)))
            except (ValueError, TypeError):
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def fig_energy_balance(comb: pd.DataFrame, fig_dir: Path) -> Path | None:
    """UV-absorbed fraction vs IR-reemitted fraction (the energy-balance plane).

    The diagonal is the thermodynamic Dyson signature: light removed from the UV
    is re-radiated in the infrared, so a structure that intercepts starlight lands
    near unity slope.  Natural confounders (warm dust, brown-dwarf companions) add
    infrared flux without removing ultraviolet flux and so fall to the lower right.
    """
    if "nuv_deficit_frac" not in comb and "score_energy_balance" not in comb:
        return None
    tau = comb.get("tau", pd.Series(np.nan, index=comb.index)).to_numpy(dtype=float)
    nuv = comb.get("nuv_deficit_frac", pd.Series(np.nan, index=comb.index)).to_numpy(dtype=float)
    ok = np.isfinite(tau) & np.isfinite(nuv) & (tau > 0) & (nuv > 0)
    if ok.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    eb = comb.get("flag_energy_balance", pd.Series(False, index=comb.index)).to_numpy(dtype=bool)
    mm = comb.get("multimodal_candidate", pd.Series(False, index=comb.index)).to_numpy(dtype=bool)
    base = ok & ~eb & ~mm
    ax.scatter(tau[base], nuv[base], s=14, c="#1f77b4", alpha=0.7, label="IR-excess WD")
    ax.scatter(tau[ok & eb & ~mm], nuv[ok & eb & ~mm], s=40, c="#ff7f0e",
               label="energy-balanced")
    ax.scatter(tau[ok & mm], nuv[ok & mm], s=110, marker="*", c="crimson",
               label="multi-axis candidate")
    lim = [min(np.nanmin(tau[ok]), np.nanmin(nuv[ok])) * 0.5,
           max(np.nanmax(tau[ok]), np.nanmax(nuv[ok])) * 2.0]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6, label="1:1 (perfect balance)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"IR re-emitted fraction $\tau_{\rm IR}$")
    ax.set_ylabel(r"UV-absorbed fraction $f_{\rm NUV}$")
    ax.set_title("Energy-balance plane")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = fig_dir / "emp_energy_balance.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_multimodal_sed(comb: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Full ultraviolet-to-infrared SED of the top multi-axis candidates.

    Overplots the observed flux (GALEX FUV/NUV, Gaia, 2MASS, WISE) against the
    blackbody photosphere fitted from the Gaia solid angle and ``teff``.  For an
    energy-balanced candidate this single panel shows the diagnostic directly: the
    ultraviolet points fall *below* the photosphere (absorption) while the W1/W2
    points rise *above* it (re-emission).
    """
    from .photometry import band_freq_hz, planck_bnu

    cand = comb[comb.get("multimodal_candidate", False).astype(bool)].copy()
    if "multimodal_score" in cand.columns:
        cand = cand.sort_values("multimodal_score", ascending=False)
    cand = cand.head(6)
    if cand.empty or "sed_scale" not in cand.columns or "teff" not in cand.columns:
        return None
    # (band, magnitude column) in wavelength order.
    band_cols = [("FUV", "FUVmag"), ("NUV", "NUVmag"), ("BP", "BPmag"), ("G", "Gmag"),
                 ("RP", "RPmag"), ("J", "Jmag"), ("H", "Hmag"), ("Ks", "Ksmag"),
                 ("W1", "W1mag"), ("W2", "W2mag")]
    n = len(cand)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow), squeeze=False)
    drew = False
    for i, (_, c) in enumerate(cand.iterrows()):
        ax = axes[i // ncol][i % ncol]
        scale, teff = float(c.get("sed_scale", np.nan)), float(c.get("teff", np.nan))
        if not (np.isfinite(scale) and np.isfinite(teff)):
            ax.axis("off")
            continue
        xs, obs, pred = [], [], []
        for band, mcol in band_cols:
            if mcol not in c or not np.isfinite(c.get(mcol, np.nan)) or band not in BANDS:
                continue
            lam = BANDS[band]["lambda_um"]
            xs.append(lam)
            obs.append(float(mag_to_flux_jy(c[mcol], band)))
            with np.errstate(over="ignore"):
                pred.append(scale * np.pi * float(planck_bnu(teff, band_freq_hz(band))) * 1e26)
        if len(xs) < 3:
            ax.axis("off")
            continue
        drew = True
        order = np.argsort(xs)
        xs, obs, pred = np.array(xs)[order], np.array(obs)[order], np.array(pred)[order]
        ax.plot(xs, pred, "s--", c="#444", ms=3, label="photosphere")
        ax.plot(xs, obs, "o-", c="crimson", ms=4, label="observed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        sid = int(c["source_id"])
        nax = int(c.get("n_axes", 0))
        ax.set_title(f"WD {sid}\n$T={teff:.0f}$K, {nax} axes", fontsize=7)
        ax.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=7)
        if i % ncol == 0:
            ax.set_ylabel("flux [Jy]", fontsize=7)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if not drew:
        plt.close(fig)
        return None
    fig.tight_layout()
    out = fig_dir / "emp_multimodal_sed.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_axes_histogram(comb: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Distribution of the number of independent anomaly axes per object."""
    if "n_axes" not in comb:
        return None
    n_axes = comb["n_axes"].to_numpy(dtype=float)
    n_axes = n_axes[np.isfinite(n_axes)]
    if n_axes.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    kmax = int(np.nanmax(n_axes))
    bins = np.arange(-0.5, kmax + 1.5, 1.0)
    ax.hist(n_axes, bins=bins, color="#4060a0", edgecolor="white")
    ax.axvline(1.5, color="crimson", ls="--", lw=1.0, label=r"$\geq 2$ axes (candidate)")
    ax.set_yscale("log")
    ax.set_xlabel("number of independent anomaly axes")
    ax.set_ylabel("white dwarfs (log)")
    ax.set_title("Multi-axis anomaly coincidence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "emp_axes_hist.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def multimodal_table_tex(comb: pd.DataFrame, out_path: Path) -> Path:
    """LaTeX tabular of the multi-axis (>=2) candidates, ranked by score."""
    cand = comb[comb.get("multimodal_candidate", False).astype(bool)].copy()
    if "multimodal_score" in cand.columns:
        cand = cand.sort_values("multimodal_score", ascending=False)
    cand = cand.head(20)
    cols = [("source_id", "Gaia EDR3 source\\_id", "{:d}"),
            ("teff", r"$T_{\rm eff}$", "{:.0f}"),
            ("n_axes", r"$n_{\rm axes}$", "{:d}"),
            ("axes_flagged", "axes", "{}"),
            ("multimodal_score", "score", "{:.2f}")]
    avail = [(c, h, f) for c, h, f in cols if c in cand.columns]
    lines = [r"\begin{tabular}{" + "l" * len(avail) + "}", r"\toprule",
             " & ".join(h for _, h, _ in avail) + r" \\", r"\midrule"]
    for _, row in cand.iterrows():
        cells = []
        for c, _, f in avail:
            v = row[c]
            try:
                if f == "{}":
                    cells.append(str(v).replace("_", r"\_"))
                elif "d}" in f:
                    cells.append(f.format(int(v)))
                else:
                    cells.append(f.format(float(v)))
            except (ValueError, TypeError):
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    if len(cand) == 0:
        lines.append(r"\multicolumn{" + str(len(avail)) +
                     r"}{c}{\emph{no multi-axis candidates}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def render_multimodal(cfg, comb: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Render the multi-modal figures + candidate table from the combined frame."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    jobs = [
        lambda: fig_energy_balance(comb, fig_dir),
        lambda: fig_axes_histogram(comb, fig_dir),
        lambda: fig_multimodal_sed(comb, fig_dir),
        lambda: multimodal_table_tex(comb, fig_dir / "multimodal_table.tex"),
    ]
    for job in jobs:
        try:
            out = job()
            if out is not None:
                paths.append(out)
                print(f"[science] wrote {out}")
        except Exception as exc:  # never abort the run on a figure error
            print(f"[science] multimodal figure skipped: {exc!r}")
    return paths


def render_empirical(cfg, tables_dir: Path, fig_dir: Path) -> list[Path]:
    """Render all empirical figures + candidate table from committed tables."""
    import json

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def _load(name):
        p = tables_dir / name
        return pd.read_parquet(p) if p.exists() else None

    vetted = _load("vetted.parquet")
    scored = _load("excess_scored.parquet")
    candidates = _load("candidates.parquet")
    summary = {}
    sp = tables_dir / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())

    jobs = []
    if vetted is not None:
        jobs.append(lambda: fig_hr(vetted, fig_dir))
        jobs.append(lambda: fig_funnel(summary, fig_dir))
    if scored is not None:
        jobs.append(lambda: fig_excess_locus(scored, fig_dir, cfg.thresholds))
    if candidates is not None and vetted is not None:
        jobs.append(lambda: fig_candidate_seds(candidates, vetted, fig_dir))
        jobs.append(lambda: candidate_table_tex(candidates, fig_dir / "candidate_table.tex"))
    for job in jobs:
        try:
            out = job()
            if out is not None:
                paths.append(out)
                print(f"[science] wrote {out}")
        except Exception as exc:  # never abort the run on a figure error
            print(f"[science] figure skipped: {exc!r}")
    return paths


__all__ = ["render_empirical", "candidate_table_tex"]

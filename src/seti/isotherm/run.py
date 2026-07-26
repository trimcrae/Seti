"""Runner: ISOTHERM end-to-end (stages checkpoint to disk).

Stages
------
probe       Archive reachability.  Runs FIRST and always; writes
            ``archive_probe.json`` and reports whether CASSIS was reached.
corpus      Build the spectral corpus index (CASSIS / IRSA IRS / VizieR).
screen      Stage-1 cheap screen over every spectrum (~1 s each).
shape       Stage-2 full shape analysis on screen survivors (~20-50 s each).
calibrate   Injection-recovery sensitivity map: at what (SNR, temperature
            ratio) is a cascade separable from a continuous gradient?  Offline.
score       Funnel counts, candidate list, REPORT.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config

CHANNEL = "isotherm"


def _jsonable(obj):
    """Recursively coerce numpy scalars/arrays so json.dumps never chokes."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2, default=str))


def out_dir_for(cfg: Config) -> Path:
    d = cfg.root / "results" / CHANNEL
    d.mkdir(parents=True, exist_ok=True)
    return d


def channel_thresholds(cfg: Config) -> dict:
    """Thresholds from ``config/thresholds.yaml``; module defaults otherwise."""
    from .shape_stats import DEFAULT_THRESHOLDS

    return {**DEFAULT_THRESHOLDS, **(cfg.thresholds.get(CHANNEL) or {})}


# ---------------------------------------------------------------------------
# Analysis of one spectrum
# ---------------------------------------------------------------------------

def analyse_one(lam_um, flux, err, thresholds: dict | None = None,
                resolution: float = 100.0, sys_frac: float = 0.05,
                full: bool = True) -> dict:
    """Prepare and analyse a single spectrum.

    Rebinning to resolution elements happens HERE, before any fit: Spitzer/IRS
    low-res is ~2x oversampled, so a BIC computed on raw pixels over-rewards
    extra components by ~ln(2) per parameter — it would manufacture cascades.
    """
    from .sed_model import apply_systematic_floor, bin_to_resolution
    from .shape_stats import classify, compute_shape_stats, screen_spectrum

    lam_b, flux_b, err_b, _ = bin_to_resolution(lam_um, flux, err,
                                                resolution=resolution)
    if lam_b.size == 0:
        return {"verdict": "INSUFFICIENT_DATA", "flags": [],
                "vetoes": ["empty_spectrum"], "n_binned": 0}
    err_b = apply_systematic_floor(flux_b, err_b, sys_frac=sys_frac)

    screen = screen_spectrum(lam_b, flux_b, err_b, thresholds)
    if not full or not screen.get("pass"):
        return {"verdict": ("screened_out" if not screen.get("pass")
                            else "screen_only"),
                "screen": screen, "n_binned": int(lam_b.size),
                "flags": [], "vetoes": [screen.get("reason", "")]}

    stats = compute_shape_stats(lam_b, flux_b, err_b, thresholds)
    v = classify(stats, thresholds)
    return {**v.to_dict(), "screen": screen, "n_binned": int(lam_b.size)}


# ---------------------------------------------------------------------------
# Sensitivity calibration (offline; no archive needed)
# ---------------------------------------------------------------------------

def cascade_sensitivity(snrs=(60, 150, 400, 1000), ratios=(1.5, 2.0, 2.5, 3.0),
                        t_hot_k: float = 600.0, n_shells: int = 3,
                        lam_lo: float = 5.2, lam_hi: float = 38.0,
                        n_lam: int = 180, seed: int = 17,
                        temp_tol: float = 0.15) -> pd.DataFrame:
    """At what (SNR, temperature ratio) is a cascade separable from a gradient?

    ``temp_tol`` is the fractional tolerance within which a fitted component
    counts as recovering a true one (default 15%).  It is reported in every row
    so the map cannot be read without its tolerance.

    This is a *completeness map*, not a null result: it states the region of
    parameter space in which a detection would have been possible, which is the
    only honest way to report what a non-detection in the rest of it means.
    """
    from .sed_model import fit_gradient, mbb, select_n_components

    rng = np.random.default_rng(seed)
    lam = np.geomspace(lam_lo, lam_hi, int(n_lam))
    rows = []
    for ratio in ratios:
        for snr in snrs:
            m = np.zeros_like(lam)
            temps = [t_hot_k / float(ratio) ** k for k in range(int(n_shells))]
            for t in temps:
                c = mbb(lam, t, 0.0)
                m += c / c.max()
            m /= m.max()
            e = m / float(snr)
            f = m + rng.normal(0, e)
            best, _ = select_n_components(lam, f, e, n_max=4)
            grad = fit_gradient(lam, f, e)
            dbic = float(best.bic - grad.bic)
            matched, frac_err = _temps_match(temps, best.temps_k, temp_tol)
            rows.append({
                "temperature_ratio": float(ratio), "snr": float(snr),
                "n_components_recovered": int(best.n_components),
                "delta_bic_discrete_minus_gradient": dbic,
                "discrete_wins": bool(dbic < -10.0),
                # RECOVERY REQUIRES THE TEMPERATURES TO MATCH, not merely the
                # component count. Keyed on count alone, run 30211326404
                # reported ratio 1.5 / SNR 1000 as "recovered" on a fit of
                # [283, 523, 3000] K against a truth of [600, 400, 267] K --
                # three components, none of them real, including an invented
                # 3000 K one. A sensitivity curve built from that is worthless.
                "n_temps_matched": int(matched),
                "max_frac_temp_error": float(frac_err),
                "temp_tolerance_frac": float(temp_tol),
                "recovered": bool(matched >= int(n_shells) and dbic < -10.0),
                "temps_true_k": [float(t) for t in temps],
                "temps_fit_k": [float(t) for t in best.temps_k],
            })
    return pd.DataFrame(rows)


def _temps_match(true_k, fit_k, tol: float) -> tuple[int, float]:
    """Greedily pair fitted temperatures to true ones within ``tol`` fractional.

    Returns ``(n_matched, worst_fractional_error_over_matched)``.  Each fitted
    component may claim at most one true temperature, so an invented extra
    component cannot manufacture a match and duplicated fits cannot double-count.
    """
    true_list = [float(t) for t in true_k]
    remaining = [float(t) for t in fit_k]
    matched, worst = 0, 0.0
    for t in sorted(true_list, reverse=True):
        if not remaining:
            break
        errs = [abs(f - t) / t for f in remaining]
        j = int(np.argmin(errs))
        if errs[j] <= tol:
            matched += 1
            worst = max(worst, errs[j])
            remaining.pop(j)
    return matched, (worst if matched else float("inf"))


# ---------------------------------------------------------------------------
# Scoring / reporting
# ---------------------------------------------------------------------------

def score_and_write(cfg: Config, results: pd.DataFrame, probe: dict | None = None,
                    sensitivity: pd.DataFrame | None = None,
                    corpus_n: int | None = None) -> dict:
    out_dir = out_dir_for(cfg)
    probe = probe or {}

    n_input = int(corpus_n if corpus_n is not None else len(results))
    n_analysed = int(len(results))
    if n_analysed:
        verdicts = results["verdict"].fillna("unknown")
        counts = verdicts.value_counts().to_dict()
        n_screened_out = int((verdicts == "screened_out").sum())
        n_full = int(n_analysed - n_screened_out)
        cands = results[verdicts.isin(["S5_ISOTHERMAL_REVIEW",
                                       "S6_MATRIOSHKA_CASCADE_REVIEW"])]
    else:
        counts, n_screened_out, n_full = {}, 0, 0
        cands = results.iloc[:0] if len(results.columns) else pd.DataFrame()

    if n_analysed:
        results.to_parquet(out_dir / "shape_results.parquet", index=False)
    if len(cands):
        cands.to_csv(out_dir / "candidates.csv", index=False)
    if sensitivity is not None and len(sensitivity):
        sensitivity.to_csv(out_dir / "sensitivity.csv", index=False)

    archive_verdict = probe.get("verdict", "NOT_PROBED")
    # "No shape anomaly in the corpus" is a claim about SPECTRA that were
    # actually fitted. Rows whose analysis returned INSUFFICIENT_DATA carried no
    # usable spectrum at all -- run 30211326404 indexed 5,480 IRAS LRS catalogue
    # entries and got INSUFFICIENT_DATA on every one, because the catalogue
    # carries broadband fluxes and an LRS class letter, not the spectra. Letting
    # that report as a clean null would be precisely the silent degradation the
    # channel brief forbids: a statement about the sky inferred from zero
    # measurements.
    n_insufficient = int(counts.get("INSUFFICIENT_DATA", 0)) if n_analysed else 0
    n_with_spectra = n_analysed - n_insufficient
    if archive_verdict == "NO_DATA_REACHED" or n_analysed == 0:
        verdict = "NO_DATA_REACHED"
    elif n_with_spectra == 0:
        verdict = "NO_SPECTRA_REACHED"
    elif len(cands):
        verdict = "SHAPE_CANDIDATES_FOR_REVIEW"
    else:
        verdict = "no_shape_anomaly_in_corpus"

    summary = {
        "channel": CHANNEL,
        "verdict": verdict,
        "archive_verdict": archive_verdict,
        "cassis_reachable": bool(probe.get("cassis_reachable", False)),
        "any_spectral_archive_reachable":
            bool(probe.get("any_spectral_archive_reachable", False)),
        "funnel": {
            "n_corpus": n_input,
            "n_analysed": n_analysed,
            "n_screened_out_stage1": n_screened_out,
            "n_full_shape_analysis": n_full,
            # The number that decides whether any statement about the sky is
            # possible at all: rows carrying a usable spectrum, not rows indexed.
            "n_with_usable_spectrum": n_with_spectra,
            "n_insufficient_data": n_insufficient,
            "n_s5_isothermal": int((results["verdict"] == "S5_ISOTHERMAL_REVIEW").sum())
            if n_analysed else 0,
            "n_s6_cascade": int(
                (results["verdict"] == "S6_MATRIOSHKA_CASCADE_REVIEW").sum())
            if n_analysed else 0,
            "n_rejected_natural": int(
                (results["verdict"] == "REJECTED_NATURAL").sum()) if n_analysed else 0,
        },
        "verdict_counts": counts,
        "candidates": (cands.head(50).to_dict("records") if len(cands) else []),
        "sensitivity": (sensitivity.to_dict("records")
                        if sensitivity is not None and len(sensitivity) else []),
        "limitations": [
            "5-38 micron constrains beta only for components whose Wien peak is "
            "in band, i.e. T ~ 130-1000 K; outside that window beta and T are "
            "degenerate and beta is reported as unconstrained.",
            "Three Wien peaks inside 5-38 micron force a cascade temperature "
            "ratio <= 2.8; at ratio ~2 the per-component width cannot be bounded "
            "below the natural floor, so the narrowness tier is unreachable "
            "without far-IR photometry.",
            "The redshift scan loses PAH 6.2/7.7 above z ~ 1.5-3.9, so high-z "
            "interlopers are not excluded by the spectral test alone.",
            "A single-component beta is biased towards 0 for multi-temperature "
            "sources; beta is therefore read off the model the data select.",
        ],
    }
    _write_json(out_dir / "summary.json", summary)

    lines = [
        "# ISOTHERM run report", "",
        "Search on the SHAPE of infrared excess in temperature space —",
        "emissivity index beta, temperature-distribution width, silicate-feature",
        "equivalent width, and component multiplicity in geometric progression",
        "(docs/isotherm.md).", "",
        f"**Archive verdict:** {archive_verdict}  ",
        f"**CASSIS reachable:** {bool(probe.get('cassis_reachable', False))}  ",
        f"**Channel verdict:** {verdict}", "",
        "## Funnel", "",
        "| stage | n |", "|---|---|",
    ]
    for k, v in summary["funnel"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", f"## Candidates: {len(cands)}", ""]
    if len(cands):
        for r in cands.head(50).to_dict("records"):
            lines.append(f"- `{r.get('source_id', '?')}` {r.get('verdict')} "
                         f"tier={r.get('tier', '?')} flags={r.get('flags')}")
    else:
        lines.append("(none at the current thresholds)")

    if sensitivity is not None and len(sensitivity):
        lines += ["", "## Cascade sensitivity (injection-recovery)", "",
                  "| T ratio | SNR | n recovered | dBIC(discrete-gradient) | "
                  "separable |", "|---|---|---|---|---|"]
        for r in sensitivity.to_dict("records"):
            lines.append(f"| {r['temperature_ratio']:g} | {r['snr']:g} | "
                         f"{r['n_components_recovered']} | "
                         f"{r['delta_bic_discrete_minus_gradient']:+.1f} | "
                         f"{r['recovered']} |")

    lines += ["", "## Limitations", ""]
    lines += [f"- {x}" for x in summary["limitations"]]
    lines += [
        "",
        "No-null rule (CLAUDE.md): an empty candidate list is a statement about",
        "THIS corpus at THESE thresholds, never a publishable result. The next",
        "moves are extending the baseline with AKARI FIS / IRAS far-IR",
        "photometry (which re-opens the per-component narrowness tier), and the",
        "IRAS LRS Calgary atlas that Carrigan 2009 used — now anchorable to Gaia",
        "parallaxes, which is exactly what his distance/luminosity degeneracy",
        "lacked.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[isotherm] {verdict}: {n_analysed} analysed, {len(cands)} candidates")
    return summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def isotherm_run(cfg: Config | None = None, stage: str = "all",
                 max_spectra: int = 2000, resolution: float = 100.0,
                 shard: int = 0, n_shards: int = 1) -> dict:
    cfg = cfg or load_config()
    out_dir = out_dir_for(cfg)
    th = channel_thresholds(cfg)

    probe: dict = {}
    if stage in ("probe", "corpus", "screen", "shape", "all"):
        from .acquire import probe_archives

        probe_path = out_dir / "archive_probe.json"
        if stage != "probe" and probe_path.exists():
            probe = json.loads(probe_path.read_text())
        else:
            probe = probe_archives(out_dir)
        if stage == "probe":
            return probe

    if stage in ("calibrate", "all"):
        sens_path = out_dir / "sensitivity.csv"
        if sens_path.exists():
            sens = pd.read_csv(sens_path)
        else:
            sens = cascade_sensitivity()
            sens.to_csv(sens_path, index=False)
        print(f"[isotherm] sensitivity map: {int(sens['recovered'].sum())}"
              f"/{len(sens)} cells separable")
        if stage == "calibrate":
            return {"stage": "calibrate", "n_cells": int(len(sens)),
                    "n_separable": int(sens["recovered"].sum())}
    else:
        sens = None

    results = pd.DataFrame()
    corpus_n = 0
    if stage in ("corpus", "screen", "shape", "all"):
        from .acquire import fetch_cassis_spectrum, fetch_irs_catalog

        corpus_path = out_dir / "corpus.parquet"
        corpus = None
        if corpus_path.exists():
            corpus = pd.read_parquet(corpus_path)
            print(f"[isotherm] corpus checkpoint: {len(corpus)} rows")
        elif probe.get("any_spectral_archive_reachable"):
            try:
                corpus = fetch_irs_catalog(out_dir / "irs_catalog.parquet",
                                           max_rows=int(max_spectra) * 4)
                corpus.to_parquet(corpus_path, index=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[isotherm] corpus build failed: {exc!r}")
                corpus = None
        if corpus is None:
            print("[isotherm] no spectral corpus reachable — emitting "
                  "NO_DATA_REACHED rather than a fabricated result")
            return score_and_write(cfg, pd.DataFrame(), probe, sens, 0)

        corpus_n = int(len(corpus))
        if n_shards > 1:
            corpus = corpus.iloc[int(shard)::int(n_shards)].copy()
        corpus = corpus.head(int(max_spectra))

        rows = []
        batch_path = out_dir / f"shape_shard_{int(shard):03d}.parquet"
        done: set = set()
        if batch_path.exists():
            prev = pd.read_parquet(batch_path)
            rows = prev.to_dict("records")
            done = set(prev.get("source_id", pd.Series(dtype=str)).astype(str))
            print(f"[isotherm] resuming: {len(done)} already analysed")

        for i, r in enumerate(corpus.to_dict("records")):
            sid = str(r.get("aorkey") or r.get("source_id") or r.get("cntr") or i)
            if sid in done:
                continue
            try:
                spec = fetch_cassis_spectrum(int(r["aorkey"])) \
                    if "aorkey" in r and pd.notna(r.get("aorkey")) else None
                if spec is None or spec.empty:
                    rows.append({"source_id": sid, "verdict": "INSUFFICIENT_DATA",
                                 "vetoes": ["no_spectrum"], "flags": []})
                    continue
                res = analyse_one(spec["wavelength_um"].to_numpy(),
                                  spec["flux_jy"].to_numpy(),
                                  spec["err_jy"].to_numpy(),
                                  thresholds=th, resolution=resolution)
                res["source_id"] = sid
                res["ra"] = r.get("ra", np.nan)
                res["dec"] = r.get("dec", np.nan)
                res["tier"] = res.get("stats", {}).get("tier", "none")
                res.pop("stats", None)
                rows.append(res)
            except Exception as exc:  # noqa: BLE001 - one bad spectrum != dead run
                rows.append({"source_id": sid, "verdict": "ERROR",
                             "vetoes": [repr(exc)[:200]], "flags": []})
            if (i + 1) % 25 == 0:
                pd.DataFrame(rows).to_parquet(batch_path, index=False)
                print(f"[isotherm] {i + 1}/{len(corpus)} analysed")
        results = pd.DataFrame(rows)
        if len(results):
            results.to_parquet(batch_path, index=False)

    if stage in ("score", "all"):
        if results.empty:
            shards = sorted(out_dir.glob("shape_shard_*.parquet"))
            if shards:
                results = pd.concat([pd.read_parquet(p) for p in shards],
                                    ignore_index=True)
                if "source_id" in results.columns:
                    results = results.drop_duplicates("source_id")
        if sens is None and (out_dir / "sensitivity.csv").exists():
            sens = pd.read_csv(out_dir / "sensitivity.csv")
        return score_and_write(cfg, results, probe, sens, corpus_n or len(results))

    return {"stage": stage, "status": "done"}


__all__ = ["analyse_one", "cascade_sensitivity", "isotherm_run", "score_and_write"]

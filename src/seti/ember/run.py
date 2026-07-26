"""Stage orchestration for EMBER. Writes ``results/ember/``.

Stages, each checkpointed so a killed job resumes rather than restarts:

``audit``
    Offline. Computes the per-epoch-pair systematics verdict from the band
    model alone and writes ``pair_audit.json``. Runs with no network, and is
    the stage that decides which pairs the science stages are allowed to use.
``acquire``
    Runner-only. Pulls AKARI/IRC, IRAS PSC+FSC, Gaia DR3 and AllWISE, with
    proper motion propagated to each survey's epoch.
``excess``
    Fits the empirical photospheric colour locus per band and measures the
    excess at every epoch.
``cessation``
    Transports each early excess to the late band, forms the difference
    statistic, adjudicates the three-epoch ladder and calibrates the detection
    threshold on the sample's own rising tail.
``vet``
    Runs the contamination funnel.
``report``
    Writes ``summary.json`` and ``REPORT.md``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .bands import BANDS, CANDIDATE_PAIRS, EPOCH_LADDER, audit_pair, rsr_source
from .crossepoch import (
    adjudicate_ladder,
    beam_sum_consistency,
    calibrate_null,
    cessation,
    cessation_mc,
    fit_dust_temperature,
    fit_photosphere_locus,
    measure_count_slope,
    measure_excess,
)
from .vet import vet_all

STAGES = ("audit", "acquire", "analyse", "excess", "cessation", "vet",
          "report", "all")


def _jsonable(x):
    """Recursively convert numpy / dataclass objects into JSON-safe values."""
    if is_dataclass(x) and not isinstance(x, type):
        return _jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not np.isfinite(v) else v
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2))


# --------------------------------------------------------------------------
# Stage 1: systematics audit (offline)
# --------------------------------------------------------------------------
def stage_audit(out_dir: Path) -> dict:
    """Compute the per-pair systematics verdict from the band model alone.

    This stage runs with no network and gates everything downstream: a pair that
    fails here is not searched, and the reason is recorded rather than silently
    dropped.
    """
    audits = [audit_pair(e, late) for e, late in CANDIDATE_PAIRS]
    payload = {
        "rsr_source": {k: rsr_source(b) for k, b in BANDS.items()},
        "epoch_ladder": [{"survey": s, "year": y, "bands": list(bs)}
                         for s, y, bs in EPOCH_LADDER],
        "pairs": [asdict(a) for a in audits],
        "usable_pairs": [f"{a.early}->{a.late}" for a in audits
                         if not a.verdict.startswith("REJECT")],
        "rejected_pairs": [f"{a.early}->{a.late}: {a.verdict}" for a in audits
                           if a.verdict.startswith("REJECT")],
        "note": ("NEOWISE is deliberately absent from the ladder: it flies W1/W2 "
                 "only, so it cannot measure 100-300 K waste heat at any epoch. "
                 "It is used solely as the post-drop flatness requirement."),
    }
    _write(out_dir / "pair_audit.json", payload)
    return payload


# --------------------------------------------------------------------------
# Stage 2: acquisition
# --------------------------------------------------------------------------
def stage_acquire(out_dir: Path, table: pd.DataFrame | None = None,
                  n_ra_chunks: int = 12, shard: int = 0, n_shards: int = 1,
                  fetchers: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Build the working table for this shard's RA slice.

    Pass ``table=`` to run entirely offline. Each shard writes its own parquet,
    so a killed shard costs one slice and can be re-run alone.
    """
    if table is not None:
        return table, {"source": "injected", "n_rows": int(len(table)),
                       "archive_reachable": None}

    from . import acquire as acq

    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    status: dict = {"source": "archives", "shard": shard, "n_shards": n_shards,
                    "errors": []}

    # Real response curves collapse the bandpass systematic; the trapezoid
    # fallback stays correct if this fails, and says so in the audit.
    try:
        rsr_dir = Path(__file__).resolve().parents[1] / "data_assets" / "rsr"
        status["rsr_fetch"] = acq.fetch_rsr_curves(rsr_dir)
    except Exception as exc:  # noqa: BLE001
        status["errors"].append(f"rsr: {exc!r}")

    width = 360.0 / max(1, n_shards)
    ra_lo, ra_hi = shard * width, (shard + 1) * width
    df, slice_status = acq.build_working_table(ra_lo, ra_hi, cache,
                                               n_ra_chunks=n_ra_chunks,
                                               fetchers=fetchers)
    status.update(slice_status)
    if not df.empty:
        df.to_parquet(cache / f"working_{shard:03d}.parquet", index=False)
    return df, status


def load_working_table(out_dir: Path) -> pd.DataFrame:
    """Concatenate every acquired shard. Empty frame when nothing was acquired."""
    cache = out_dir / "cache"
    parts = sorted(cache.glob("working_*.parquet")) if cache.exists() else []
    if not parts:
        return pd.DataFrame()
    frames = []
    for p in parts:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as exc:  # noqa: BLE001
            print(f"[ember] unreadable shard {p.name}: {exc!r}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("match_id")


# --------------------------------------------------------------------------
# Stage 3: excess measurement
# --------------------------------------------------------------------------
def stage_excess(table: pd.DataFrame, out_dir: Path,
                 bands: tuple[str, ...] = ("I12", "I25", "S9W", "L18W", "W3", "W4"),
                 colour_col: str = "bp_rp") -> tuple[pd.DataFrame, dict]:
    """Fit the photospheric locus per band and measure every epoch's excess.

    Expects, per row: ``ks_jy``, ``ks_err_jy``, the colour column, and
    ``f_<band>_jy`` / ``e_<band>_jy`` for each band present.
    """
    df = table.copy()
    loci, meta = {}, {}
    for band in bands:
        fcol, ecol = f"f_{band}_jy", f"e_{band}_jy"
        if fcol not in df.columns:
            continue
        ok = (df[fcol] > 0) & (df["ks_jy"] > 0) & np.isfinite(df[colour_col])
        if ok.sum() < 30:
            meta[band] = {"n_calib": int(ok.sum()), "degraded": True,
                          "note": "too few sources to fit a locus"}
            continue
        locus = fit_photosphere_locus(
            df.loc[ok, colour_col].to_numpy(),
            -2.5 * np.log10(df.loc[ok, "ks_jy"].to_numpy()),
            -2.5 * np.log10(df.loc[ok, fcol].to_numpy()),
            band=band, colour_name=colour_col)
        loci[band] = locus
        meta[band] = {"n_calib": locus.n_calib, "degraded": locus.degraded,
                      "median_scatter_mag": float(np.median(locus.scatter))
                      if locus.scatter.size else None}

        # Eddington deboosting only for the shallow, flux-limited early bands.
        slope = None
        if band in ("I12", "I25", "S9W", "L18W"):
            lo = BANDS[band].faint_5sig_jy or 0.05
            slope = measure_count_slope(df.loc[ok, fcol].to_numpy(), lo, lo * 40)
            meta[band]["count_slope_gamma"] = slope

        rows = []
        for _, r in df.iterrows():
            f, e = r.get(fcol, np.nan), r.get(ecol, np.nan)
            if not np.isfinite(f) or f <= 0:
                rows.append(None)
                continue
            rows.append(measure_excess(band, float(f), float(e if np.isfinite(e) else 0.1 * f),
                                       float(r["ks_jy"]), float(r.get("ks_err_jy", 0.0)),
                                       float(r[colour_col]), locus, count_slope=slope))
        df[f"exc_{band}"] = rows
        df[f"exc_{band}_jy"] = [m.f_exc_jy if m else np.nan for m in rows]
        df[f"chi_{band}"] = [m.chi if m else np.nan for m in rows]

    _write(out_dir / "locus.json", meta)
    return df, {"loci": meta, "bands_fitted": sorted(loci)}


# --------------------------------------------------------------------------
# Stage 4: the cessation statistic
# --------------------------------------------------------------------------
def stage_cessation(df: pd.DataFrame, out_dir: Path,
                    pairs: tuple[tuple[str, str], ...] = CANDIDATE_PAIRS) -> tuple:
    """Form the difference statistic for every usable pair and adjudicate."""
    audits = {f"{e}->{le}": audit_pair(e, le) for e, le in pairs}
    usable = [(e, le) for e, le in pairs
              if not audits[f"{e}->{le}"].verdict.startswith("REJECT")]

    records: list[dict] = []
    for _, r in df.iterrows():
        rec: dict = {"source_id": r.get("source_id", "?")}
        # Fit the excess colour temperature from same-epoch band pairs.
        temps = {}
        for survey, _yr, bs in EPOCH_LADDER:
            ms = [r.get(f"exc_{b}") for b in bs]
            ms = [m for m in ms if m is not None]
            if len(ms) >= 2:
                temps[survey] = fit_dust_temperature(ms)
        t_default = next(iter(temps.values()), (400.0, 150.0, 1500.0, "prior"))

        results: dict[str, object] = {}
        for early, late in usable:
            m_e, m_l = r.get(f"exc_{early}"), r.get(f"exc_{late}")
            if m_e is None or m_l is None:
                continue
            survey = BANDS[early].survey
            t, t_lo, t_hi, src = temps.get(survey, t_default)
            res = cessation(m_e, m_l, t_dust_k=t, t_dust_lo=t_lo, t_dust_hi=t_hi,
                            t_dust_source=src)
            results[f"{early}->{late}"] = res
            rec[f"z_{early}_{late}"] = res.z
            rec[f"fcess_{early}_{late}"] = res.f_cess
            rec[f"verdict_{early}_{late}"] = res.verdict

        ladder, notes = adjudicate_ladder(
            results.get("I12->W3"), results.get("I12->S9W"), results.get("S9W->W3"))
        rec["ladder_verdict"] = ladder
        rec["ladder_notes"] = notes
        primary = results.get("I12->W3") or results.get("S9W->W3") or \
            results.get("I25->W4")
        if primary is not None:
            rec["z"] = primary.z
            rec["f_cess"] = primary.f_cess
            rec["primary_pair"] = f"{primary.early_band}->{primary.late_band}"
            rec["baseline_yr"] = primary.baseline_yr
            rec["early_snr"] = (primary.exc_early_jy / primary.exc_early_err_jy
                                if primary.exc_early_err_jy > 0 else np.nan)
        records.append(rec)

    z_all = np.array([r.get("z", np.nan) for r in records], dtype=float)
    null = calibrate_null(z_all)

    shortlist = [r for r in records
                 if np.isfinite(r.get("z", np.nan)) and r["z"] >= null["threshold"]
                 and r.get("ladder_verdict") in
                 ("monotone_decline", "fade_2007_2010", "fade_1983_2006")]

    # Exact Monte-Carlo confirmation for the shortlist only; the analytic path
    # linearises a transfer that moves by up to a factor of five.
    for rec in shortlist:
        row = df[df["source_id"] == rec["source_id"]]
        if row.empty or "primary_pair" not in rec:
            continue
        e_b, l_b = rec["primary_pair"].split("->")
        m_e, m_l = row.iloc[0].get(f"exc_{e_b}"), row.iloc[0].get(f"exc_{l_b}")
        if m_e is not None and m_l is not None:
            rec["mc"] = cessation_mc(m_e, m_l)

    _write(out_dir / "null_calibration.json", null)
    _write(out_dir / "shortlist.json", shortlist)
    return records, shortlist, {"null": null, "pair_audit": {k: asdict(v)
                                                             for k, v in audits.items()},
                                "n_scored": len(records),
                                "n_shortlist": len(shortlist)}


# --------------------------------------------------------------------------
# Stage 5: vetting
# --------------------------------------------------------------------------
def stage_vet(shortlist: list[dict], df: pd.DataFrame, out_dir: Path,
              require_all: bool = True) -> dict:
    """Run the contamination funnel over the shortlist."""
    by_id = {str(r.get("source_id")): r for _, r in df.iterrows()} if not df.empty else {}
    rows = []
    for rec in shortlist:
        merged = dict(by_id.get(str(rec["source_id"]), {}))
        merged.update({k: v for k, v in rec.items() if k != "mc"})
        # Beam-sum verdict, when the neighbourhood was fetched.
        if "beam_neighbour_fluxes_jy" in merged and "primary_pair" in rec:
            e_b, _l = rec["primary_pair"].split("->")
            audit = audit_pair(e_b, rec["primary_pair"].split("->")[1])
            merged.update(beam_sum_consistency(
                float(merged.get(f"f_{e_b}_jy", np.nan)),
                float(merged.get(f"e_{e_b}_jy", np.nan)),
                list(merged["beam_neighbour_fluxes_jy"]),
                audit.transfer_300k))
        rows.append(merged)
    out = vet_all(rows, require_all=require_all)
    _write(out_dir / "vetting.json",
           {"n_in": out["n_in"], "n_survivors": out["n_survivors"],
            "killed_by_stage": out["killed_by_stage"],
            "results": [asdict(r) for r in out["results"]]})
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def ember_run(cfg, stage: str = "all", table: pd.DataFrame | None = None,
              n_ra_chunks: int = 12, shard: int = 0, n_shards: int = 1,
              require_all_checks: bool = True, fetchers: dict | None = None) -> dict:
    """Run EMBER. ``table=`` bypasses acquisition so tests run with no network.

    ``stage="acquire"`` fetches one RA shard and stops; ``stage="analyse"``
    reads whatever shards are on disk and runs the science without touching an
    archive, which is what the workflow's reduce path uses.
    """
    out_dir = Path(cfg.root) / "results" / "ember"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"channel": "ember", "stage": stage,
                     "signature": "S1 extinguished waste heat"}

    audit = stage_audit(out_dir)
    summary["pair_audit"] = {"usable": audit["usable_pairs"],
                             "rejected": audit["rejected_pairs"],
                             "rsr_source": audit["rsr_source"]}
    if stage == "audit":
        summary["verdict"] = "AUDIT_ONLY"
        _write(out_dir / "summary.json", summary)
        return summary

    if stage == "analyse":
        work = load_working_table(out_dir)
        acq_status = {"source": "cache", "n_rows": int(len(work)),
                      "archive_reachable": bool(len(work))}
    else:
        work, acq_status = stage_acquire(out_dir, table=table,
                                         n_ra_chunks=n_ra_chunks, shard=shard,
                                         n_shards=n_shards, fetchers=fetchers)
    summary["acquisition"] = acq_status

    if stage == "acquire":
        summary["verdict"] = ("ACQUIRED" if (work is not None and not work.empty)
                              else "NO_DATA_REACHED")
        summary["counts"] = {"acquired": int(len(work)) if work is not None else 0}
        _write(out_dir / f"summary_acquire_{shard:03d}.json", summary)
        return summary

    if work is None or work.empty:
        summary["verdict"] = "NO_DATA_REACHED"
        summary["counts"] = {"acquired": 0, "with_excess": 0, "scored": 0,
                             "shortlist": 0, "survivors": 0}
        summary["note"] = (
            "No usable rows were obtained. No candidate is emitted and no "
            "occurrence limit is computed: an unreached archive is not a null "
            "result, it is an absence of data.")
        _write(out_dir / "summary.json", summary)
        return summary

    work, excess_meta = stage_excess(work, out_dir)
    summary["photosphere"] = excess_meta
    if stage == "excess":
        summary["verdict"] = "EXCESS_ONLY"
        _write(out_dir / "summary.json", summary)
        return summary

    records, shortlist, cess_meta = stage_cessation(work, out_dir)
    summary["cessation"] = {k: v for k, v in cess_meta.items() if k != "pair_audit"}
    if stage == "cessation":
        summary["verdict"] = "SCORED"
        _write(out_dir / "summary.json", summary)
        return summary

    vet = stage_vet(shortlist, work, out_dir, require_all=require_all_checks)
    n_excess = int(np.isfinite(work.get("chi_I12", pd.Series(dtype=float))).sum()
                   + np.isfinite(work.get("chi_S9W", pd.Series(dtype=float))).sum())
    summary["counts"] = {
        "acquired": int(len(work)),
        "with_early_photometry": n_excess,
        "scored": cess_meta["n_scored"],
        "shortlist": cess_meta["n_shortlist"],
        "survivors": vet["n_survivors"],
    }
    summary["killed_by_stage"] = vet["killed_by_stage"]
    summary["survivors"] = vet["survivors"]
    summary["null_threshold"] = cess_meta["null"]["threshold"]
    summary["verdict"] = ("CANDIDATES" if vet["n_survivors"] else
                          "NO_SURVIVOR")
    summary["limitations"] = (
        "NEOWISE carries W1/W2 only, so no epoch after 2010 measures 12-25 "
        "micron flux; the decades-long baseline exists exclusively between "
        "IRAS (1983), AKARI (2006-07) and WISE (2010). Sensitivity is set by "
        "the early epoch: IRAS reaches ~0.4 Jy at 12 micron, so the 27-year "
        "pair probes only very large excesses. The published cross-epoch "
        "stability floor for IRAS-to-WISE is 4 percent (HD 172555), and no "
        "fade smaller than a few times that is believable regardless of its "
        "formal significance.")
    _write(out_dir / "summary.json", summary)
    _write(out_dir / "records.json", records[:5000])
    _write_report(out_dir, summary, shortlist, vet)
    return summary


def _write_report(out_dir: Path, summary: dict, shortlist: list[dict],
                  vet: dict) -> None:
    """Human-readable REPORT.md with a mandatory honest-sensitivity section."""
    lines = [
        "# EMBER — mid-infrared waste heat that switched off",
        "",
        f"**Verdict:** `{summary.get('verdict')}`",
        "",
        "## Epoch pairs",
        "",
        "| pair | verdict |",
        "|---|---|",
    ]
    for p in summary.get("pair_audit", {}).get("usable", []):
        lines.append(f"| `{p}` | usable |")
    for p in summary.get("pair_audit", {}).get("rejected", []):
        lines.append(f"| `{p}` | rejected |")

    counts = summary.get("counts", {})
    lines += ["", "## Funnel", "", "| stage | n |", "|---|---|"]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")

    if vet.get("killed_by_stage"):
        lines += ["", "## Rejections by contamination stage", "",
                  "| rule | killed |", "|---|---|"]
        for k, v in vet["killed_by_stage"].items():
            if v:
                lines.append(f"| {k} | {v} |")

    lines += ["", "## Shortlist", ""]
    if not shortlist:
        lines.append("No source exceeded the empirically calibrated threshold.")
    else:
        lines += ["| source | pair | baseline (yr) | z | f_cess | ladder |",
                  "|---|---|---|---|---|---|"]
        for s in shortlist[:50]:
            lines.append(
                f"| {s.get('source_id')} | {s.get('primary_pair', '-')} | "
                f"{s.get('baseline_yr', '-')} | {s.get('z', float('nan')):.1f} | "
                f"{s.get('f_cess', float('nan')):.2f} | {s.get('ladder_verdict')} |")

    lines += [
        "",
        "## Honest sensitivity statement",
        "",
        summary.get("limitations", ""),
        "",
        "The detection threshold is set by the sample's own *rising* tail, not "
        "by a Gaussian assumption: every symmetric systematic populates fades "
        "and rises equally, so only the excess of faders over risers above the "
        "threshold can contain signal. The one asymmetric systematic — the "
        "flux-limited Eddington bias of the early epoch, which fades only — is "
        "corrected explicitly using the source-count slope measured from the "
        "catalogue itself.",
        "",
        "A null here is not a result and is not to be written up as one. The "
        "informative quantity it would produce — the first measurement of the "
        "rate of mid-infrared excess appearance and disappearance at 12–25 "
        "micron — is recorded internally as an honesty check on the funnel.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")

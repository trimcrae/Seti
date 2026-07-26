"""Stage orchestration for RUST.  Writes ``results/rust/``.

Two stages, deliberately separated so the expensive one can be sharded and
checkpointed across a runner matrix while the cheap one aggregates:

``rust_sweep``   per sky field: pull paired ZTF g+r light curves, measure the
                 bias-corrected season scatter in each band, remove the per-CCD
                 ensemble common mode in the second moment, fit the trend, and
                 write that field's candidates immediately.
``rust_vet``     across all fields: fetch Gaia/SIMBAD/NEOWISE context, run the
                 contamination gauntlet, and apply the two cross-checks that are
                 too expensive to run on every star --- exact per-season Monte
                 Carlo and equal-N subsampling.

Every stage emits an explicit verdict when it could not reach data.  A field
that returned nothing writes ``NO_DATA_REACHED``; it never writes zero
candidates as though it had searched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .scatter import season_scatter
from .trend import RustStats, ensemble_detrend_scatter, fit_scatter_trend
from .vet import periodic_fraction, vet_row

# Candidate thresholds.  Deliberately strict on *robustness* (leave-one-out,
# monotonicity) rather than on raw amplitude: a large amplitude is easy to fake
# with one bad season, an exactly-ordered rise across seven seasons is not.
#
# ``RANK_P_MAX`` is the primary gate and it is distribution-free.  For the 7-8
# seasons a ZTF decade provides, p <= 0.01 means the season ordering is in the
# top ~1% of all n! orderings -- and it assumes nothing whatever about the shape
# of the growth, which matters because Lacki (2025) gives a cascade *timescale*,
# not a light-curve shape.  The linear-in-variance slope then supplies the
# amplitude and a second, model-dependent significance.
RANK_P_MAX = 0.01
SLOPE_SIGMA_MIN = 2.5
LOO_SIGMA_MIN = 1.5
AMP_GROWTH_MIN = 1.3
AMP_LAST_MMAG_MIN = 15.0        # do not chase growth from 1 mmag to 2 mmag


def load_rust_config(cfg=None) -> dict:
    """Read ``config/rust.yaml`` and install its thresholds over the defaults.

    The channel brief requires thresholds to live in ``config/``, not as magic
    numbers in code.  They are also mirrored as module constants so the offline
    tests and any direct caller work with no config file present --- a missing
    config degrades to the documented defaults rather than to an exception.
    """
    global RANK_P_MAX, SLOPE_SIGMA_MIN, LOO_SIGMA_MIN, AMP_GROWTH_MIN, AMP_LAST_MMAG_MIN
    try:
        import yaml
        root = Path(cfg.root) if cfg is not None else Path(__file__).resolve().parents[3]
        p = root / "config" / "rust.yaml"
        if not p.exists():
            return {}
        conf = yaml.safe_load(p.read_text()) or {}
    except Exception as exc:                           # noqa: BLE001
        print(f"[rust] config/rust.yaml not loaded ({exc!r}); using defaults")
        return {}
    d = conf.get("detect", {})
    RANK_P_MAX = float(d.get("rank_p_max", RANK_P_MAX))
    SLOPE_SIGMA_MIN = float(d.get("slope_sigma_min", SLOPE_SIGMA_MIN))
    LOO_SIGMA_MIN = float(d.get("loo_sigma_min", LOO_SIGMA_MIN))
    AMP_GROWTH_MIN = float(d.get("amp_growth_min", AMP_GROWTH_MIN))
    AMP_LAST_MMAG_MIN = float(d.get("amp_last_mmag_min", AMP_LAST_MMAG_MIN))
    return conf


def _field_tag(ra: float, dec: float, radius_deg: float) -> str:
    return f"ra{ra:.3f}_dec{dec:+.3f}_r{radius_deg:.2f}"


def _is_candidate(s: RustStats | None) -> bool:
    if s is None:
        return False
    return bool(
        s.slope_var_yr > 0
        and s.rank_p <= RANK_P_MAX
        and s.slope_sigma >= SLOPE_SIGMA_MIN
        and s.slope_sigma_loo_min >= LOO_SIGMA_MIN
        and s.amp_growth >= AMP_GROWTH_MIN
        and s.amp_last_mmag >= AMP_LAST_MMAG_MIN
    )


def rust_sweep(
    cfg=None,
    ra: float = 270.0,
    dec: float = 30.0,
    radius_deg: float = 0.5,
    box_deg: float = 0.12,
    min_epochs: int = 60,
    min_epochs_season: int = 8,
    min_seasons: int = 4,
    season_days: float = 365.25,
    detrend_season: bool = True,
    time_budget_s: float = 2400.0,
    max_boxes: int | None = None,
    out_root: Path | None = None,
) -> dict:
    """Sweep one sky field for stars whose variability amplitude is growing."""
    from .acquire import iter_region_2band

    conf = load_rust_config(cfg)
    sea = conf.get("season", {})
    season_days = float(sea.get("season_days", season_days))
    min_epochs_season = int(sea.get("min_epochs_season", min_epochs_season))
    min_seasons = int(sea.get("min_seasons", min_seasons))
    detrend_season = bool(sea.get("detrend_season", detrend_season))

    root = Path(out_root) if out_root is not None else (
        (Path(cfg.root) if cfg is not None else Path(".")) / "results" / "rust")
    tag = _field_tag(ra, dec, radius_deg)
    out_dir = root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_g: list[dict] = []
    rows_r: list[dict] = []
    meta_by_id: dict[str, dict] = {}
    n_seen = n_measured = 0

    for pair in iter_region_2band(ra, dec, radius_deg=radius_deg, box_deg=box_deg,
                                  min_epochs=min_epochs, time_budget_s=time_budget_s,
                                  max_boxes=max_boxes):
        n_seen += 1
        sid = pair["source_id"]
        got = {}
        for band, key, ccd_key, bucket in (("g", "lc_g", "ccd_g", rows_g),
                                           ("r", "lc_r", "ccd_r", rows_r)):
            lc = pair[key]
            ss = season_scatter(lc["mjd"].to_numpy(), lc["mag"].to_numpy(),
                                lc["magerr"].to_numpy(), season_days=season_days,
                                min_epochs_season=min_epochs_season,
                                min_seasons=min_seasons,
                                detrend_season=detrend_season)
            if ss is None:
                continue
            got[band] = True
            bucket.append({"source_id": sid, "_ss": ss,
                           "_ccd": pair.get(ccd_key, "x"), "_nepoch": int(len(lc))})
        # The two-band requirement is enforced at acquisition, not at scoring:
        # a source measurable in only one band is never scored at all.
        if len(got) == 2:
            n_measured += 1
            meta_by_id[sid] = {"source_id": sid, "ra": pair["ra"], "dec": pair["dec"],
                               "ccd": pair.get("ccd", "x"),
                               "sep_arcsec": pair.get("sep_arcsec", float("nan"))}

    if n_measured == 0:
        summary = {"field": tag, "ra": ra, "dec": dec, "radius_deg": radius_deg,
                   "verdict": "NO_DATA_REACHED", "n_sources_paired": n_seen,
                   "n_measured_both_bands": 0, "n_candidates": 0}
        (out_dir / "field_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[rust] {tag}: NO_DATA_REACHED ({n_seen} paired, 0 measurable)")
        return summary

    # Keep only sources measured in both bands, then remove the per-CCD ensemble
    # common mode in the second moment --- the decisive systematic control.
    keep = set(meta_by_id)
    rows_g = [r for r in rows_g if r["source_id"] in keep]
    rows_r = [r for r in rows_r if r["source_id"] in keep]
    diag_g = ensemble_detrend_scatter(rows_g, min_seasons=min_seasons)
    diag_r = ensemble_detrend_scatter(rows_r, min_seasons=min_seasons)

    stat_g = {r["source_id"]: r.get("_stat") for r in rows_g}
    stat_r = {r["source_id"]: r.get("_stat") for r in rows_r}
    ss_g = {r["source_id"]: r.get("_ss") for r in rows_g}
    ss_r = {r["source_id"]: r.get("_ss") for r in rows_r}

    records = []
    for sid, meta in meta_by_id.items():
        sg, sr = stat_g.get(sid), stat_r.get(sid)
        if sg is None or sr is None:
            continue
        rec = dict(meta)
        rec.update({f"g_{k}": v for k, v in sg.as_dict().items()})
        rec.update({f"r_{k}": v for k, v in sr.as_dict().items()})
        rec["is_candidate"] = bool(_is_candidate(sg) and _is_candidate(sr))
        rec["n_bands_candidate"] = int(_is_candidate(sg)) + int(_is_candidate(sr))
        for band, ss in (("g", ss_g.get(sid)), ("r", ss_r.get(sid))):
            if ss is not None:
                rec[f"{band}_n_per_season"] = ",".join(str(int(v)) for v in ss.n)
                rec[f"{band}_v_exc"] = ",".join(f"{v:.6g}" for v in ss.v_exc)
        records.append(rec)

    df = pd.DataFrame(records)
    cand = df[df["is_candidate"]] if len(df) else df
    if len(df):
        df.sort_values("r_score", ascending=False).head(500).to_csv(
            out_dir / "rust_top_stats.csv", index=False)
    if len(cand):
        cand.to_csv(out_dir / "rust_candidates.csv", index=False)

    summary = {
        "field": tag, "ra": ra, "dec": dec, "radius_deg": radius_deg,
        "verdict": "SEARCHED",
        "n_sources_paired": int(n_seen),
        "n_measured_both_bands": int(n_measured),
        "n_scored": int(len(df)),
        "n_candidates_either_band": int((df["n_bands_candidate"] > 0).sum()) if len(df) else 0,
        "n_candidates_both_bands": int(len(cand)),
        "thresholds": {"rank_p_max": RANK_P_MAX, "slope_sigma_min": SLOPE_SIGMA_MIN,
                       "loo_sigma_min": LOO_SIGMA_MIN, "amp_growth_min": AMP_GROWTH_MIN,
                       "amp_last_mmag_min": AMP_LAST_MMAG_MIN},
        "ensemble_detrend_g": diag_g, "ensemble_detrend_r": diag_r,
        # If the field was too thin to measure a common mode, every candidate it
        # produced is UNCORRECTED for the drifting-magerr systematic.  Surface
        # that at the top level, not buried two dicts down.
        "ensemble_correction_applied": bool(
            diag_g.get("ensemble_correction_applied")
            and diag_r.get("ensemble_correction_applied")),
        "detrend_season": bool(detrend_season),
        "cadence_bias_note": ("a fitted line is removed per season (not just the "
                              "median), the scatter is compared against a per-season "
                              "null computed with that season's own epoch count and "
                              "error vector using the line-detrended null table, and "
                              "a per-CCD ensemble error-scale factor is then fitted "
                              "and removed"),
    }
    (out_dir / "field_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[rust] {tag}: {n_measured} two-band sources, "
          f"{len(cand)} candidates (both bands)")
    return summary


def rust_vet(cfg=None, out_root: Path | None = None, max_candidates: int = 200,
             offline: bool = False) -> dict:
    """Aggregate every field's candidates, then run the contamination gauntlet."""
    import glob

    root = Path(out_root) if out_root is not None else (
        (Path(cfg.root) if cfg is not None else Path(".")) / "results" / "rust")
    root.mkdir(parents=True, exist_ok=True)

    frames = []
    fields = sorted(glob.glob(str(root / "*" / "field_summary.json")))
    n_searched = n_scored = 0
    for fp in fields:
        try:
            s = json.loads(Path(fp).read_text())
        except Exception:                              # noqa: BLE001
            continue
        n_scored += int(s.get("n_scored", 0) or 0)
        if s.get("verdict") == "SEARCHED":
            n_searched += 1
    for fp in sorted(glob.glob(str(root / "*" / "rust_candidates.csv"))):
        try:
            d = pd.read_csv(fp)
        except Exception:                              # noqa: BLE001
            continue
        if len(d):
            d["field_dir"] = Path(fp).parent.name
            frames.append(d)

    if not frames:
        summary = {
            "verdict": "NO_CANDIDATES" if n_searched else "NO_DATA_REACHED",
            "n_fields_searched": n_searched, "n_stars_scored": n_scored,
            "n_candidates": 0, "n_survivors": 0,
            "note": ("no star passed the two-band rising-scatter gate; this is a "
                     "count, not a limit --- see docs/rust.md"),
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[rust-vet] {summary['verdict']}: "
              f"{n_searched} fields, {n_scored} stars scored, 0 candidates")
        return summary

    cand = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    cand = cand.sort_values("r_score", ascending=False).head(max_candidates)
    print(f"[rust-vet] vetting {len(cand)} candidates from {n_searched} fields")

    ctx = pd.DataFrame()
    if not offline:
        try:
            from .acquire import fetch_gaia_context
            ctx = fetch_gaia_context(cand[["source_id", "ra", "dec"]])
        except Exception as exc:                       # noqa: BLE001
            print(f"[rust-vet] Gaia context unavailable: {exc!r}")
    if len(ctx):
        cand = cand.merge(ctx, on="source_id", how="left")

    simbad = pd.DataFrame()
    if not offline:
        try:
            from ..acquire.science import fetch_simbad_context
            simbad = fetch_simbad_context(cand[["source_id", "ra", "dec"]])
        except Exception as exc:                       # noqa: BLE001
            print(f"[rust-vet] SIMBAD unavailable: {exc!r}")
    if len(simbad) and "simbad_otype" in simbad.columns:
        cand = cand.merge(simbad[["source_id", "simbad_otype"]], on="source_id",
                          how="left")

    verdicts = []
    for _, r in cand.iterrows():
        sg = _stats_from_row(r, "g")
        sr = _stats_from_row(r, "r")
        neowise = None
        power = 0.0
        if not offline:
            try:
                from ..dimming.characterize import fetch_neowise
                neowise = fetch_neowise(float(r["ra"]), float(r["dec"]))
            except Exception as exc:                   # noqa: BLE001
                print(f"[rust-vet] NEOWISE failed for {r['source_id']}: {exc!r}")
            try:
                from .acquire import fetch_ztf_2band
                lg, _lr = fetch_ztf_2band(float(r["ra"]), float(r["dec"]))
                if lg is not None and len(lg):
                    power = periodic_fraction(lg["mjd"].to_numpy(),
                                              lg["mag"].to_numpy())
            except Exception as exc:                   # noqa: BLE001
                print(f"[rust-vet] periodicity check failed: {exc!r}")
        v = vet_row(sg, sr, context={
            "n_neighbors_5as": r.get("n_neighbors_5as"),
            "brightest_neighbor_dg": r.get("brightest_neighbor_dg"),
            "simbad_otype": r.get("simbad_otype"),
            "ruwe": r.get("ruwe"), "non_single_star": r.get("non_single_star"),
            "astrometric_excess_noise": r.get("astrometric_excess_noise"),
            "neowise": neowise,
        }, periodic_power=power)
        v["source_id"] = r["source_id"]
        verdicts.append(v)

    vdf = pd.DataFrame(verdicts)
    vetted = cand.merge(vdf, on="source_id", how="left")
    vetted.to_csv(root / "rust_vetted.csv", index=False)

    survivors = vetted[vetted["rust_verdict"].isin(["clean_gray", "clean_reddening"])]
    gold = vetted[vetted["rust_verdict"] == "clean_gray"]
    counts = vetted["rust_verdict"].value_counts().to_dict()

    summary = {
        "verdict": "SURVIVORS" if len(gold) else
                   ("REDDENING_ONLY" if len(survivors) else "ALL_REJECTED"),
        "n_fields_searched": n_searched, "n_stars_scored": n_scored,
        "n_candidates": int(len(cand)), "n_vetted": int(len(vetted)),
        "n_survivors": int(len(survivors)), "n_clean_gray": int(len(gold)),
        "verdict_counts": {str(k): int(v) for k, v in counts.items()},
        "gaia_context_reached": bool(len(ctx) > 0),
        "simbad_reached": bool(len(simbad) > 0),
        "offline": bool(offline),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    if len(gold):
        gold.to_csv(root / "rust_survivors.csv", index=False)
    print(json.dumps(summary, indent=2))
    return summary


def _stats_from_row(r, band: str) -> RustStats | None:
    """Rebuild a :class:`RustStats` from the flat CSV columns of one band."""
    keys = RustStats.__dataclass_fields__.keys()
    vals = {}
    for k in keys:
        v = r.get(f"{band}_{k}")
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            if k in ("n_epochs", "n_seasons"):
                v = 0
            else:
                v = float("nan")
        vals[k] = int(v) if k in ("n_epochs", "n_seasons") else float(v)
    try:
        return RustStats(**vals)
    except Exception:                                  # noqa: BLE001
        return None


def confirm_survivor(time_g, mag_g, err_g, time_r, mag_r, err_r,
                     season_days: float = 365.25, min_epochs_season: int = 8,
                     min_seasons: int = 4) -> dict:
    """The two expensive cross-checks, for a shortlist only.

    * **Exact Monte-Carlo null** --- drops every asymptotic approximation in the
      fast season-scatter path and simulates each season's actual error vector.
    * **Equal-N subsampling** --- truncates every season to the smallest season's
      epoch count, so the estimator's finite-N bias is *identical* in every
      season and cannot possibly produce a trend.

    If a candidate's slope significance collapses under either, the trend was a
    cadence artefact and the candidate is dead.  Both numbers are reported for
    every survivor, pass or fail.
    """
    from .scatter import season_scatter_mc

    out: dict = {}
    for band, (t, m, e) in (("g", (time_g, mag_g, err_g)),
                            ("r", (time_r, mag_r, err_r))):
        base = season_scatter(t, m, e, season_days=season_days,
                              min_epochs_season=min_epochs_season,
                              min_seasons=min_seasons)
        mc = season_scatter_mc(t, m, e, season_days=season_days,
                               min_epochs_season=min_epochs_season,
                               min_seasons=min_seasons)
        eq = season_scatter(t, m, e, season_days=season_days,
                            min_epochs_season=min_epochs_season,
                            min_seasons=min_seasons, equalize_n=True)
        for tag, ss in (("fast", base), ("mc", mc), ("equaln", eq)):
            st = fit_scatter_trend(ss, min_seasons=min_seasons) if ss is not None else None
            out[f"{band}_{tag}_slope_sigma"] = st.slope_sigma if st else float("nan")
            out[f"{band}_{tag}_slope_var_yr"] = st.slope_var_yr if st else float("nan")
            out[f"{band}_{tag}_amp_growth"] = st.amp_growth if st else float("nan")
    sigs = [out.get(f"{b}_{t}_slope_sigma", np.nan)
            for b in ("g", "r") for t in ("fast", "mc", "equaln")]
    finite = [s for s in sigs if np.isfinite(s)]
    out["cross_check_min_sigma"] = float(min(finite)) if finite else float("nan")
    out["cross_check_verdict"] = (
        "survives_all" if finite and min(finite) >= LOO_SIGMA_MIN
        else ("cadence_artifact" if finite else "insufficient_data"))
    return out


__all__ = ["AMP_GROWTH_MIN", "AMP_LAST_MMAG_MIN", "LOO_SIGMA_MIN", "RANK_P_MAX",
           "SLOPE_SIGMA_MIN", "confirm_survivor", "load_rust_config", "rust_sweep",
           "rust_vet"]

"""Stage orchestration for KNELL.  Writes ``results/knell/``.

Three stages, separated so the expensive one shards and checkpoints across a
runner matrix while the cheap ones aggregate:

``knell_sweep``   per sky field --- pull paired ZTF g+r light curves, block each,
                  run the per-block periodogram, find the cessation pattern,
                  measure the post-block detection efficiency by injection, and
                  write that field's candidates immediately.
``knell_vet``     across all fields --- Gaia/SIMBAD context and the contamination
                  gauntlet.
``knell_cross``   the **secondary** cross-survey layer: catalogued VSX/GCVS
                  variables not detected as periodic by ZTF, each carrying an
                  explicit injection demonstration that ZTF *would have* detected
                  the catalogued period and amplitude.  A cross-survey candidate
                  without that demonstration is not reported at all.

Verdict discipline
------------------
A stage that could not reach data emits a ``NO_DATA``-class verdict.  It never
emits "no candidates", because "we looked and found nothing" and "we could not
look" are different claims and only one of them is about the sky.  The
acquisition log (``seti.knell.acquire.AcquisitionLog``) is embedded in every
``summary.json`` so a reader can see the query text and per-stage row counts
rather than guess.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .blocks import block_periodogram, make_blocks
from .cease import analyze_band, combine_bands
from .efficiency import block_efficiency, format_pvalue, persistence_pvalue
from .vet import summarise_flags, vet_row

# Defaults; config/knell.yaml overrides them via load_knell_config().
DEFAULTS: dict = {
    "season": {"season_days": 365.25, "min_epochs_block": 15, "min_blocks": 4,
               "block_mode": "fixed"},
    "acquire": {"min_epochs_total": 80, "box_deg": 0.12, "time_budget_s": 2400},
    "period": {"min_period": 0.05, "max_period": 100.0, "oversample": 5.0},
    "detect": {"fap": 0.01, "n_null": 200, "n_trials": 200, "eta_min": 0.90,
               "p_persist_max": 0.01, "min_pre_blocks": 2, "min_post_blocks": 2,
               "min_post_span_days": 500.0, "mean_shift_max_mag": 0.05,
               "post_amp_sigma_max": 3.0, "drop_sigma_min": 5.0,
               "pdm_p_pre_max": 0.01, "pdm_p_post_min": 0.05,
               "pdm_null": 100, "var_drop_frac": 0.35,
               "fap_triage": 0.05, "n_null_triage": 60},
    "vet": {"ztf_bright_limit": 13.5, "ztf_faint_limit_g": 20.3,
            "ztf_faint_limit_r": 20.1, "crowd_delta_g": 2.5, "ruwe_max": 1.4,
            "astrometric_excess_noise_max": 1.0},
    "cross": {"tol_arcsec": 2.0, "n_trials": 400, "eta_min": 0.95},
}


def _deep_update(base: dict, extra: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_knell_config(cfg=None) -> dict:
    """Read ``config/knell.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        root = Path(cfg.root) if cfg is not None else Path(__file__).resolve().parents[3]
        p = root / "config" / "knell.yaml"
        if not p.exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(p.read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[knell] config/knell.yaml not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _field_tag(ra: float, dec: float, radius_deg: float) -> str:
    return f"ra{ra:.3f}_dec{dec:+.3f}_r{radius_deg:.2f}"


def analyze_pair(lc_g, lc_r, conf: dict, *, rng=None, measure_efficiency: bool = True):
    """Run the cessation test on both bands of one star and combine them.

    Efficiency measurement is the expensive step, so it runs in two passes: a
    cheap pass with ``measure_efficiency=False`` decides whether a transition
    pattern exists at all, and only stars that pass pay for the injections.  The
    gate itself is unchanged --- a star that never had a pattern could not have
    been a candidate under any efficiency.
    """
    sea, per, det = conf["season"], conf["period"], conf["detect"]
    kw = dict(season_days=sea["season_days"], min_epochs_block=sea["min_epochs_block"],
              min_blocks=sea["min_blocks"], block_mode=sea["block_mode"],
              min_period=per["min_period"], max_period=per["max_period"],
              oversample=per["oversample"], fap=det["fap"], n_null=det["n_null"],
              n_trials=det["n_trials"], eta_min=det["eta_min"],
              p_persist_max=det["p_persist_max"], min_pre_blocks=det["min_pre_blocks"],
              min_post_blocks=det["min_post_blocks"],
              min_post_span_days=det["min_post_span_days"],
              mean_shift_max_mag=det["mean_shift_max_mag"],
              post_amp_sigma_max=det["post_amp_sigma_max"],
              drop_sigma_min=det["drop_sigma_min"],
              pdm_p_pre_max=det["pdm_p_pre_max"],
              pdm_p_post_min=det["pdm_p_post_min"],
              pdm_null=det["pdm_null"], var_drop_frac=det["var_drop_frac"])
    cheap = dict(kw, measure_efficiency=False)
    rg = analyze_band(lc_g["mjd"], lc_g["mag"], lc_g.get("magerr"), band="g",
                      rng=rng, **cheap)
    rr = analyze_band(lc_r["mjd"], lc_r["mag"], lc_r.get("magerr"), band="r",
                      rng=rng, **cheap)
    # Cheap pass says "a clean transition exists in both bands"?  Only then pay.
    worth_it = all(
        (r.pattern_strict and r.n_pre >= det["min_pre_blocks"]
         and r.n_post >= det["min_post_blocks"]) for r in (rg, rr))
    if measure_efficiency and worth_it:
        rg = analyze_band(lc_g["mjd"], lc_g["mag"], lc_g.get("magerr"), band="g",
                          rng=rng, measure_efficiency=True, **kw)
        rr = analyze_band(lc_r["mjd"], lc_r["mag"], lc_r.get("magerr"), band="r",
                          rng=rng, measure_efficiency=True, **kw)
    return rg, rr, combine_bands(rg, rr)


def triage_was_ever_periodic(lc, conf: dict, *, rng=None) -> bool:
    """Cheap gate: was the star periodic in its FIRST usable block?

    This is not a heuristic short-cut, it is an exact restatement of the
    candidate condition.  The cessation pattern requires a *prefix* of detected
    blocks, so a star whose first block is not detected can never be a candidate
    however the rest of the baseline behaves.  Testing that one block first, with
    a deliberately **more permissive** threshold (higher ``fap``, fewer
    permutations) than the real search uses, therefore removes no candidate that
    the full test would have kept --- and it removes the overwhelming majority of
    a ZTF field, which is constant stars.

    Without this the sweep spends its entire IRSA time budget running per-block
    permutation nulls on stars that were never clocks.
    """
    sea, per, det = conf["season"], conf["period"], conf["detect"]
    blocks = make_blocks(lc["mjd"], lc["mag"], lc.get("magerr"),
                         season_days=sea["season_days"],
                         min_epochs_block=sea["min_epochs_block"],
                         min_blocks=sea["min_blocks"], mode=sea["block_mode"])
    if not blocks:
        return False
    bp = block_periodogram(blocks[0], fap=float(det.get("fap_triage", 0.05)),
                           n_null=int(det.get("n_null_triage", 60)), rng=rng,
                           min_period=per["min_period"], max_period=per["max_period"],
                           oversample=per["oversample"])
    return bool(bp.detected)


def _record(meta: dict, rg, rr, comb: dict) -> dict:
    rec = dict(meta)
    rec.update({f"g_{k}": v for k, v in rg.as_dict().items()})
    rec.update({f"r_{k}": v for k, v in rr.as_dict().items()})
    rec.update(comb)
    rec["flags"] = summarise_flags(rg, rr)
    # The joint p is a product over both bands, so its resolution floor is the
    # product of the two bands' floors -- not one band's.
    floors = [f for f in (rg.p_resolution_floor, rr.p_resolution_floor)
              if np.isfinite(f)]
    joint_floor = float(np.prod(floors)) if floors else float("nan")
    rec["p_persist_text"] = format_pvalue(comb.get("p_persist_upper_joint", float("nan")),
                                          comb.get("p_pinned_at_floor", False),
                                          joint_floor)
    rec["p_resolution_floor_joint"] = joint_floor
    return rec


def knell_sweep(
    cfg=None,
    ra: float = 270.0,
    dec: float = 30.0,
    radius_deg: float = 0.5,
    box_deg: float | None = None,
    min_epochs: int | None = None,
    time_budget_s: float | None = None,
    max_boxes: int | None = None,
    max_sources: int | None = None,
    seed: int = 20260726,
    out_root: Path | None = None,
) -> dict:
    """Stage 1 --- sweep one sky field for clocks that stopped."""
    from .acquire import AcquisitionLog, iter_region_2band_logged, probe_ztf_service

    conf = load_knell_config(cfg)
    acq = conf["acquire"]
    box_deg = float(acq["box_deg"] if box_deg is None else box_deg)
    min_epochs = int(acq["min_epochs_total"] if min_epochs is None else min_epochs)
    time_budget_s = float(acq["time_budget_s"] if time_budget_s is None else time_budget_s)

    root = Path(out_root) if out_root is not None else (
        (Path(cfg.root) if cfg is not None else Path(".")) / "results" / "knell")
    tag = _field_tag(ra, dec, radius_deg)
    out_dir = root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    log = AcquisitionLog()
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    n_seen = n_measurable = n_triaged_out = n_analysed = 0

    # The bulk fetcher swallows its own HTTP errors, so probe the service once
    # and record the outcome separately.  Without this a run in which every
    # query was refused looks identical to a run over an empty patch of sky.
    ok, detail = probe_ztf_service()
    log.record("irsa_ztf_service_probe", "one 0.003 deg cone at (180, +20), r band",
               rows=1 if ok else None, error=None if ok else detail,
               extra={"detail": detail})

    try:
        for pair in iter_region_2band_logged(
                ra, dec, log, radius_deg=radius_deg, box_deg=box_deg,
                min_epochs=min_epochs, time_budget_s=time_budget_s,
                max_boxes=max_boxes):
            n_seen += 1
            if max_sources is not None and n_seen > int(max_sources):
                break
            lc_g, lc_r = pair["lc_g"], pair["lc_r"]
            sea = conf["season"]
            if not (make_blocks(lc_g["mjd"], lc_g["mag"], lc_g.get("magerr"),
                                season_days=sea["season_days"],
                                min_epochs_block=sea["min_epochs_block"],
                                min_blocks=sea["min_blocks"], mode=sea["block_mode"])
                    and make_blocks(lc_r["mjd"], lc_r["mag"], lc_r.get("magerr"),
                                    season_days=sea["season_days"],
                                    min_epochs_block=sea["min_epochs_block"],
                                    min_blocks=sea["min_blocks"], mode=sea["block_mode"])):
                continue
            n_measurable += 1
            # Exact pre-filter (see triage_was_ever_periodic): a star not
            # periodic in its first block can never satisfy the prefix pattern.
            if not (triage_was_ever_periodic(lc_g, conf, rng=rng)
                    and triage_was_ever_periodic(lc_r, conf, rng=rng)):
                n_triaged_out += 1
                continue
            n_analysed += 1
            rg, rr, comb = analyze_pair(lc_g, lc_r, conf, rng=rng)
            meta = {"source_id": pair["source_id"], "ra": pair["ra"], "dec": pair["dec"],
                    "ccd": pair.get("ccd", "x"),
                    "sep_arcsec": pair.get("sep_arcsec", float("nan")),
                    "mean_mag_g": float(np.median(lc_g["mag"])),
                    "mean_mag_r": float(np.median(lc_r["mag"]))}
            records.append(_record(meta, rg, rr, comb))
    except Exception as exc:                              # noqa: BLE001
        print(f"[knell] field {tag} acquisition aborted: {exc!r}")

    acq_log = log.as_dict()
    if n_measurable == 0:
        # Three genuinely different facts, kept apart.  Only the third is a
        # statement about the sky, and none of the three is a science null.
        n_ztf = sum(s["rows"] for s in acq_log["stages"] if s["stage"] == "ztf_region_2band")
        if acq_log["any_query_failed"]:
            verdict = "NO_DATA_REACHED"
        elif n_ztf == 0:
            verdict = "ARCHIVE_RETURNED_ZERO_SOURCES"
        else:
            verdict = "NO_TESTABLE_LIGHT_CURVES"
        summary = {"field": tag, "ra": ra, "dec": dec, "radius_deg": radius_deg,
                   "verdict": verdict, "n_sources_paired": n_seen,
                   "n_testable": 0, "n_candidates": 0, "acquisition": acq_log,
                   "note": ("no light curve had enough epoch blocks in both bands to "
                            "support a per-block periodogram; this is a statement "
                            "about the data reached, not about the sky")}
        (out_dir / "field_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[knell] {tag}: {verdict} ({n_seen} paired, 0 testable)")
        return summary

    df = pd.DataFrame(records)
    cand = (df[df["two_band_cessation"]] if ("two_band_cessation" in df and len(df))
            else df.head(0))
    if len(df):
        df.head(2000).to_csv(out_dir / "knell_all_stats.csv", index=False)
    if len(cand):
        cand.to_csv(out_dir / "knell_candidates.csv", index=False)

    status_counts = {}
    for b in ("g", "r"):
        col = f"{b}_status"
        if col in df:
            status_counts[b] = {str(k): int(v) for k, v in df[col].value_counts().items()}

    summary = {
        "field": tag, "ra": ra, "dec": dec, "radius_deg": radius_deg,
        "verdict": "SEARCHED",
        "n_sources_paired": int(n_seen), "n_testable": int(n_measurable),
        "n_triaged_out_never_periodic": int(n_triaged_out),
        "n_fully_analysed": int(n_analysed),
        "n_one_band_cessation": (int((df["n_bands_cessation"] == 1).sum())
                                 if len(df) else 0),
        "n_candidates": int(len(cand)),
        "band_status_counts": status_counts,
        "acquisition": acq_log,
        "config": {"season": conf["season"], "period": conf["period"],
                   "detect": conf["detect"]},
        "methodology_note": (
            "cessation is established INTRA-SURVEY (ZTF's own early seasons against "
            "its own late seasons, in both g and r), and every claimed "
            "non-detection is scored against the injection-measured detection "
            "efficiency for that star's own period and amplitude in that block's "
            "own sampling and noise.  Without that normalisation the statistic "
            "measures survey cadence, not astrophysics."),
    }
    (out_dir / "field_summary.json").write_text(json.dumps(summary, indent=2))
    log.write(out_dir / "acquisition_log.json")
    print(f"[knell] {tag}: {n_measurable} testable stars, {len(cand)} two-band "
          f"cessation candidates")
    return summary


def knell_vet(cfg=None, out_root: Path | None = None, max_candidates: int = 200,
              offline: bool = False) -> dict:
    """Stage 2 --- aggregate every field's candidates and run the gauntlet."""
    from .acquire import AcquisitionLog, fetch_gaia_context, fetch_simbad_context

    conf = load_knell_config(cfg)
    root = Path(out_root) if out_root is not None else (
        (Path(cfg.root) if cfg is not None else Path(".")) / "results" / "knell")
    root.mkdir(parents=True, exist_ok=True)

    n_fields = n_searched = n_testable = 0
    acq_failed = False
    for fp in sorted(glob.glob(str(root / "*" / "field_summary.json"))):
        try:
            s = json.loads(Path(fp).read_text())
        except Exception:                                 # noqa: BLE001
            continue
        n_fields += 1
        n_testable += int(s.get("n_testable", 0) or 0)
        if s.get("verdict") == "SEARCHED":
            n_searched += 1
        acq_failed |= bool((s.get("acquisition") or {}).get("any_query_failed"))

    frames = []
    for fp in sorted(glob.glob(str(root / "*" / "knell_candidates.csv"))):
        try:
            d = pd.read_csv(fp)
        except Exception:                                 # noqa: BLE001
            continue
        if len(d):
            d["field_dir"] = Path(fp).parent.name
            frames.append(d)

    if not frames:
        if n_testable == 0:
            verdict = "NO_DATA_REACHED" if (acq_failed or n_fields == 0) \
                else "NO_TESTABLE_LIGHT_CURVES"
            note = ("no star was testable, so nothing about cessation was measured; "
                    "this is NOT a null result and must not be reported as one")
        else:
            verdict = "NO_CANDIDATES"
            note = ("stars were tested and none showed a two-band cessation at "
                    "demonstrated sensitivity; a count, not an occurrence limit")
        summary = {"verdict": verdict, "n_fields": n_fields,
                   "n_fields_searched": n_searched, "n_stars_testable": n_testable,
                   "n_candidates": 0, "n_survivors": 0, "note": note}
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[knell-vet] {verdict}: {n_searched} fields, {n_testable} testable, "
              "0 candidates")
        return summary

    cand = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    sort_key = "p_persist_upper_joint" if "p_persist_upper_joint" in cand else "source_id"
    cand = cand.sort_values(sort_key).head(int(max_candidates))
    print(f"[knell-vet] vetting {len(cand)} candidates from {n_searched} fields")

    log = AcquisitionLog()
    ctx = pd.DataFrame()
    simbad = pd.DataFrame()
    if not offline:
        ctx = fetch_gaia_context(cand[["source_id", "ra", "dec"]], log=log)
        simbad = fetch_simbad_context(cand[["source_id", "ra", "dec"]], log=log)
    if len(ctx):
        cand = cand.merge(ctx.drop(columns=[c for c in ("ra", "dec") if c in ctx],
                                   errors="ignore"), on="source_id", how="left")
    if len(simbad) and "simbad_otype" in simbad.columns:
        cand = cand.merge(simbad[["source_id", "simbad_otype"]], on="source_id",
                          how="left")

    verdicts = []
    for _, r in cand.iterrows():
        rg = _result_from_row(r, "g")
        rr = _result_from_row(r, "r")
        comb = {"n_bands_cessation": int(r.get("n_bands_cessation", 0) or 0),
                "two_band_cessation": bool(r.get("two_band_cessation", False)),
                "same_transition_block": bool(r.get("same_transition_block", False)),
                "same_period_both_bands": bool(r.get("same_period_both_bands", False))}
        v = vet_row(rg, rr, comb, context={
            "mean_mag_g": r.get("mean_mag_g"), "mean_mag_r": r.get("mean_mag_r"),
            "n_neighbors_5as": r.get("n_neighbors_5as"),
            "brightest_neighbor_dg": r.get("brightest_neighbor_dg"),
            "simbad_otype": r.get("simbad_otype"), "ruwe": r.get("ruwe"),
            "non_single_star": r.get("non_single_star"),
            "astrometric_excess_noise": r.get("astrometric_excess_noise"),
        }, cfg=conf["vet"])
        v["source_id"] = r["source_id"]
        verdicts.append(v)

    vetted = cand.merge(pd.DataFrame(verdicts), on="source_id", how="left")
    vetted.to_csv(root / "knell_vetted.csv", index=False)
    survivors = vetted[vetted["knell_verdict"] == "clean_cessation"]
    if len(survivors):
        survivors.to_csv(root / "knell_survivors.csv", index=False)

    summary = {
        "verdict": "SURVIVORS" if len(survivors) else "ALL_REJECTED",
        "n_fields": n_fields, "n_fields_searched": n_searched,
        "n_stars_testable": n_testable, "n_candidates": int(len(cand)),
        "n_survivors": int(len(survivors)),
        "verdict_counts": {str(k): int(v)
                           for k, v in vetted["knell_verdict"].value_counts().items()},
        "gaia_context_reached": bool(len(ctx) > 0),
        "simbad_reached": bool(len(simbad) > 0),
        "offline": bool(offline),
        "acquisition": log.as_dict(),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "acquisition"}, indent=2))
    return summary


def _result_from_row(r, band: str):
    """Rebuild a light :class:`~seti.knell.cease.CessationResult` from CSV columns."""
    from .cease import CessationResult

    res = CessationResult(band=band)
    for k in res.__dict__:
        if k in ("flags", "blocks", "band"):
            continue
        v = r.get(f"{band}_{k}")
        if v is None:
            continue
        cur = getattr(res, k)
        try:
            if isinstance(cur, bool):
                setattr(res, k, bool(v) and str(v).lower() not in ("false", "0", "nan"))
            elif isinstance(cur, int):
                setattr(res, k, int(float(v)))
            elif isinstance(cur, float):
                setattr(res, k, float(v))
            else:
                setattr(res, k, v)
        except (TypeError, ValueError):
            continue
    fl = r.get(f"{band}_flags")
    res.flags = [f for f in str(fl).split(";") if f and f != "nan"] if fl is not None else []
    return res


# ---------------------------------------------------------------------------
# The cross-survey secondary layer
# ---------------------------------------------------------------------------


def crossmatch_demonstration(t, y, e, period: float, amplitude_mmag: float,
                             conf: dict | None = None, *, rng=None) -> dict:
    """Would the *later* survey have detected the *earlier* catalogue's period?

    This is the mandatory carry-along for every cross-survey candidate, and it is
    the whole reason the cross-survey layer is secondary rather than primary.  A
    GCVS or VSX variable that ZTF does not flag as periodic has, prima facie,
    stopped --- but the catalogue's discovery data had a different passband, a
    different cadence, a different aperture and a different depth, so the naive
    comparison measures the two surveys against each other.  Injecting the
    catalogued period and amplitude into the ZTF light curve's own epochs, and
    asking how often the ZTF-side detector fires, converts that into a statement
    about the star.

    Returns the per-block efficiencies, the persistence bound, and an explicit
    ``demonstrated`` boolean.  A candidate whose ``demonstrated`` is False is not
    reported as a cessation at all --- it is reported as untestable.
    """
    conf = conf or load_knell_config()
    sea, per, det = conf["season"], conf["period"], conf["cross"]
    blocks = make_blocks(t, y, e, season_days=sea["season_days"],
                         min_epochs_block=sea["min_epochs_block"],
                         min_blocks=1, mode=sea["block_mode"])
    if not blocks or not np.isfinite(period) or period <= 0:
        return {"demonstrated": False, "reason": "no usable ZTF blocks",
                "n_blocks": 0}
    rng = np.random.default_rng(rng)
    effs = [block_efficiency(b, float(period), float(amplitude_mmag),
                             n_trials=int(det["n_trials"]),
                             n_null=int(conf["detect"]["n_null"]),
                             fap=float(conf["detect"]["fap"]), noise_mode="data",
                             min_period=per["min_period"], max_period=per["max_period"],
                             oversample=per["oversample"], rng=rng)
            for b in blocks]
    pp = persistence_pvalue(effs)
    eta_min = pp.get("eta_min", float("nan"))
    return {
        "demonstrated": bool(np.isfinite(eta_min) and eta_min >= float(det["eta_min"])),
        "eta_min": float(eta_min), "n_blocks": len(blocks),
        "p_persist_upper": pp.get("p_persist_upper", float("nan")),
        "p_persist_text": format_pvalue(pp.get("p_persist_upper", float("nan")),
                                        pp.get("pinned_at_floor", False),
                                        pp.get("resolution_floor", float("nan"))),
        "pinned_at_floor": bool(pp.get("pinned_at_floor", False)),
        "per_block_eta": [float(e.eta) for e in effs],
        "per_block_n": [int(e.n_epochs) for e in effs],
        "injected_amplitude_mmag": float(amplitude_mmag),
        "injected_period_days": float(period),
        "note": ("efficiency measured by injecting the CATALOGUED period and "
                 "amplitude into the ZTF light curve's own epochs and noise, and "
                 "applying the identical blind detector"),
    }


def knell_cross(cfg=None, ra: float = 270.0, dec: float = 30.0,
                radius_deg: float = 0.5, out_root: Path | None = None,
                max_targets: int = 200, seed: int = 20260726) -> dict:
    """Stage 3 (secondary) --- catalogued VSX variables that ZTF no longer sees."""
    from ..dimming.acquire import fetch_ztf_lightcurve
    from .acquire import AcquisitionLog, fetch_vsx_region

    conf = load_knell_config(cfg)
    root = Path(out_root) if out_root is not None else (
        (Path(cfg.root) if cfg is not None else Path(".")) / "results" / "knell")
    out_dir = root / ("cross_" + _field_tag(ra, dec, radius_deg))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = AcquisitionLog()
    rng = np.random.default_rng(seed)

    vsx = fetch_vsx_region(ra, dec, radius_deg=radius_deg, log=log)
    if not len(vsx):
        acq = log.as_dict()
        summary = {"stage": "cross", "verdict": ("NO_DATA_REACHED"
                                                 if acq["any_query_failed"]
                                                 else "CATALOGUE_RETURNED_ZERO_ROWS"),
                   "ra": ra, "dec": dec, "radius_deg": radius_deg,
                   "n_catalogue": 0, "acquisition": acq}
        (out_dir / "cross_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    have_p = vsx[pd.to_numeric(vsx.get("period"), errors="coerce") > 0] \
        if "period" in vsx else vsx.head(0)
    rows = []
    for _, v in have_p.head(int(max_targets)).iterrows():
        try:
            lc = fetch_ztf_lightcurve(float(v["ra"]), float(v["dec"]), band="r")
        except Exception as exc:                          # noqa: BLE001
            log.record("ztf_target_lc", f"cone at ({v['ra']},{v['dec']})", error=repr(exc))
            continue
        if lc is None or not len(lc):
            log.record("ztf_target_lc", f"cone at ({v['ra']},{v['dec']})", rows=0)
            continue
        amp_mmag = 1e3 * abs(float(v.get("mag_min", np.nan))
                             - float(v.get("mag_max", np.nan))) / 2.0
        if not np.isfinite(amp_mmag) or amp_mmag <= 0:
            continue
        res = analyze_band(lc["mjd"], lc["mag"], lc.get("magerr"), band="r",
                           season_days=conf["season"]["season_days"],
                           min_epochs_block=conf["season"]["min_epochs_block"],
                           min_blocks=1, measure_efficiency=False, rng=rng)
        demo = crossmatch_demonstration(lc["mjd"], lc["mag"], lc.get("magerr"),
                                        float(v["period"]), amp_mmag, conf, rng=rng)
        rows.append({"name": v.get("name"), "ra": float(v["ra"]), "dec": float(v["dec"]),
                     "vtype": v.get("vtype"), "cat_period": float(v["period"]),
                     "cat_amp_mmag": amp_mmag, "ztf_status": res.status,
                     "ztf_n_detected_blocks": res.n_detected,
                     "ztf_n_blocks": res.n_blocks, **demo})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out_dir / "cross_candidates.csv", index=False)
    ceased = df[(df.get("ztf_n_detected_blocks", 0) == 0) & df.get("demonstrated", False)] \
        if len(df) else df
    summary = {
        "stage": "cross", "verdict": "SEARCHED", "ra": ra, "dec": dec,
        "radius_deg": radius_deg, "n_catalogue": int(len(vsx)),
        "n_with_period": int(len(have_p)), "n_ztf_tested": int(len(df)),
        "n_ceased_demonstrated": int(len(ceased)),
        "n_undemonstrated": int((~df["demonstrated"]).sum()) if len(df) else 0,
        "acquisition": log.as_dict(),
        "note": ("SECONDARY layer.  Passband, cadence and pipeline all change "
                 "between the catalogue and ZTF, so a candidate counts only if the "
                 "injection demonstration shows ZTF would have detected the "
                 "catalogued period and amplitude in its own data."),
    }
    (out_dir / "cross_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "acquisition"}, indent=2))
    return summary


__all__ = ["DEFAULTS", "analyze_pair", "crossmatch_demonstration", "knell_cross",
           "knell_sweep", "knell_vet", "load_knell_config"]

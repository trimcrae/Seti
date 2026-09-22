"""OSSUARY stage orchestration.

Stages
------
``acquire``   pull the sample tracks from the Gaia archive (runner only).
``analyze``   kinematics -> empirical photosphere locus -> excess -> dust fit ->
              contamination gauntlet.  Fully offline given a sample table.
``followup``  per-candidate reddening, beam neighbours and SIMBAD identity for the
              shortlist only (runner; the shortlist is small by construction).
``report``    ``results/ossuary/summary.json`` + ``REPORT.md``.

Every stage checkpoints.  Every stage that cannot reach data says so in a
first-class ``verdict`` field rather than emitting an empty candidate list that
would read like a null result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from . import acquire as acq
from . import excess as exc
from . import kinematics as kin
from . import vet as vetting

_TRACKS = ("spec", "phot", "halo")


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, pd.Series):
        return _jsonable(x.to_dict())
    return x


def out_dir(cfg: Config) -> Path:
    d = cfg.root / "results" / "ossuary"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Stage 1: acquire
# --------------------------------------------------------------------------

def stage_acquire(cfg: Config, tracks=_TRACKS, g_max: float | None = None,
                  limit_per_band: int = 400_000) -> dict:
    """Pull each track; write ``sample.parquet``.  Runner only."""
    s = cfg.thresholds["ossuary"]["sample"]
    d = out_dir(cfg)
    chk = d / "chunks"
    frames, meta = [], {}
    wise_cols = acq.probe_columns("gaiadr1.allwise_original_valid", acq._ALLWISE_WANT)
    tmass_cols = acq.probe_columns("gaiadr1.tmass_original_valid", acq._TMASS_WANT)
    xt = acq.probe_columns("gaiadr3.tmass_psc_xsc_best_neighbour",
                           {"id": acq._TMASS_XMATCH_ID})

    for t in tracks:
        try:
            df = acq.fetch_track(
                t, chk, feh_max=s["feh_max"],
                poe_min=s["parallax_over_error_min"], ruwe_max=s["ruwe_max"],
                g_max=g_max if g_max is not None else s["g_max"],
                limit_per_band=limit_per_band, wise_cols=wise_cols,
                tmass_cols=tmass_cols, tmass_id=xt.get("id"))
        except Exception as e:  # noqa: BLE001
            print(f"[ossuary] track {t} failed: {e!r}")
            meta[t] = {"n": 0, "error": repr(e)}
            continue
        meta[t] = {"n": int(len(df)),
                   "tmass_degraded": bool(df.attrs.get("tmass_degraded", False))}
        frames.append(df)

    if not frames:
        verdict = {"verdict": "NO_DATA_REACHED", "tracks": meta,
                   "n_sample": 0,
                   "note": "no archive track returned rows; nothing was analysed"}
        (d / "sample_meta.json").write_text(json.dumps(_jsonable(verdict), indent=2))
        return verdict

    sample = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    sample = acq.harmonise(sample)
    sample.to_parquet(d / "sample.parquet", index=False)
    out = {"verdict": "OK", "tracks": meta, "n_sample": int(len(sample)),
           "n_with_feh": int(sample["feh"].notna().sum()),
           "n_spectroscopic_feh": int(sample["feh_is_spectroscopic"].sum())}
    (d / "sample_meta.json").write_text(json.dumps(_jsonable(out), indent=2))
    print("[ossuary]", json.dumps(_jsonable(out)))
    return out


# --------------------------------------------------------------------------
# Stage 2: analyze  (offline given a table)
# --------------------------------------------------------------------------

def _pick_anchor(df: pd.DataFrame) -> str:
    """Ks if 2MASS is usable, else G with the degradation recorded."""
    if "Ksmag" in df.columns and pd.to_numeric(
            df["Ksmag"], errors="coerce").notna().sum() >= max(20, 0.2 * len(df)):
        return "Ks"
    return "G"


def _log(msg: str) -> None:
    """Timestamped progress line, flushed: the runner's stdout is a pipe, and the
    first catalogue-scale run printed nothing in four hours before it was killed."""
    print(f"[ossuary {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Above this many rows the analysis runs its memory-lean path: kinematics on a
# slim frame, excess in chunks, the gauntlet on flagged rows only.  Below it the
# whole frame is carried (which the offline tests inspect row by row).  Both
# paths compute identical flags and verdicts; a test asserts that.
LEAN_ABOVE_ROWS = 300_000
_EXCESS_CHUNK = 500_000

_KIN_COLS = ("ra", "dec", "parallax", "parallax_error", "parallax_over_error",
             "pmra", "pmdec", "pmra_error", "pmdec_error",
             "radial_velocity", "radial_velocity_error", "rv",
             "phot_g_mean_mag", "g_mag", "logg", "logg_gspspec", "logg_gspphot")
_KIN_OUT = ("dist_pc", "has_rv", "U_lsr_kms", "V_lsr_kms", "W_lsr_kms",
            "v_tot_lsr_kms", "v_tan_lsr_kms", "v_tot_err_kms", "kinematic_method",
            "v_tot_or_bound_kms", "population", "halo_flag")


def _kinematics(work: pd.DataFrame, k: dict, s: dict) -> pd.DataFrame:
    """Kinematics computed on a slim frame and written back as columns.

    ``kinematics.classify`` copies its input twice; on a six-million-row frame
    with fifty columns that is several gigabytes per copy, so only the columns
    the calculation reads travel through it.
    """
    slim = work[[c for c in _KIN_COLS if c in work.columns]]
    res = kin.classify(slim, k)
    for col in _KIN_OUT:
        if col in res.columns:
            work[col] = res[col].to_numpy()
    work["luminosity_class"] = kin.luminosity_class(slim, s).to_numpy()
    work["M_G"] = kin.absolute_g(slim).to_numpy()
    work["reduced_pm"] = kin.reduced_proper_motion(slim).to_numpy()
    return work


def _excess_lean(work: pd.DataFrame, usable: dict, e: dict, anchor: str,
                 chunk: int = _EXCESS_CHUNK) -> tuple[pd.Series, pd.DataFrame]:
    """Excess statistics in chunks; returns the flag for every row and the
    full excess columns for the flagged rows only."""
    flag = pd.Series(False, index=work.index)
    parts = []
    n = len(work)
    for start in range(0, n, chunk):
        sub = work.iloc[start:start + chunk]
        ex = exc.compute_excess(sub, usable, e, anchor=anchor)
        f = exc.select_excess(ex, e)
        flag.iloc[start:start + chunk] = f.to_numpy()
        if f.any():
            parts.append(ex[f])
        _log(f"excess: rows {min(start + chunk, n):,}/{n:,}, flagged so far "
             f"{int(flag.sum()):,}")
    flagged = pd.concat(parts) if parts else work.iloc[0:0].copy()
    return flag, flagged


def _characterise(flagged: pd.DataFrame, c: dict, s: dict, e: dict, anchor: str,
                  rng) -> pd.DataFrame:
    """Dust fit for the flagged rows.

    The point estimate is computed for every flagged row.  The Monte-Carlo
    percentiles are computed only for rows that survive the cheap catalogue
    gates (WISE quality, the inherited ledger), because those are the only rows
    whose (T_dust, tau) uncertainty can still influence a verdict -- and a
    catalogue-scale flag list is dominated by the rows those gates remove.
    """
    if not len(flagged):
        return flagged
    wq = vetting.wise_quality_gate(flagged, c)["wise_quality_ok"]
    lg = vetting.ledger_gate(flagged, e, s)["ledger_ok"]
    mc_mask = (wq & lg).fillna(False).to_numpy(bool)
    _log(f"dust fit: {len(flagged):,} flagged rows, Monte-Carlo on "
         f"{int(mc_mask.sum()):,} that pass the catalogue gates")
    return exc.characterise(flagged, e, anchor=anchor, rng=rng, mc_mask=mc_mask)


_DUST_COLS = ("t_dust_k", "t_dust_lo_k", "t_dust_hi_k", "tau", "tau_lo",
              "tau_hi", "dust_fit_chi2", "n_excess_bands")


def _survivor_provenance(surv: pd.DataFrame) -> dict:
    """What the surviving rows actually rest on, as counts rather than prose.

    Three things were true of the 2026-09-22 catalogue run's 584 survivors and
    none of them was visible in the summary:

    * every [Fe/H] came from Gaia GSP-Phot, with no spectroscopic
      confirmation anywhere, while 28.6% of the rows were redder than
      ``bp_rp`` 1.4 -- the regime where that estimator is least reliable;
    * 582 of 584 were classified from a tangential-velocity LOWER BOUND, not
      a space velocity, so the kinematic leg was carrying 4% of the sample;
    * the two independent arguments (metal-poor AND halo-kinematic) agreed
      for 15 rows.

    A survivor selected by one unconfirmed estimator is a statement about that
    estimator, so the count that belongs next to ``n_candidates`` is the count
    where two arguments agree.
    """
    out: dict = {"n": int(len(surv))}
    if not len(surv):
        return out

    def _counts(col):
        if col not in surv.columns:
            return {}
        v = surv[col].map(lambda x: "" if x is None or (isinstance(x, float)
                                                        and not np.isfinite(x)) else str(x))
        return {k: int(n) for k, n in v.value_counts().items()}

    out["feh_provenance"] = _counts("feh_provenance")
    out["kinematic_method"] = _counts("kinematic_method")
    out["population"] = _counts("population")
    for name, col in (("metal_poor", "metal_poor"), ("halo", "halo_flag"),
                      ("two_independent_arguments", "two_independent_arguments"),
                      ("feh_spectroscopic", "feh_spectroscopic"),
                      ("full_space_velocity", "kinematics_is_full_space_velocity"),
                      ("tau_implausible", "tau_implausible"),
                      ("long_band_only", "long_band_only"),
                      ("warm_band_excess", "warm_band_excess")):
        if col in surv.columns:
            out[f"n_{name}"] = int(surv[col].fillna(False).astype(bool).sum())
    def _num(col):
        # DataFrame.get of a missing column returns None, and pd.to_numeric
        # collapses that to a SCALAR float -- the trap that has now taken this
        # channel's funnel down twice.  Name the absence instead.
        if col not in surv.columns:
            return pd.Series(np.nan, index=surv.index, dtype=float)
        return pd.to_numeric(surv[col], errors="coerce")

    for name, col in (("tau", "tau"), ("t_dust_k", "t_dust_k"), ("feh", "feh"),
                      ("bp_rp", "bp_rp")):
        v = _num(col)
        if v.notna().any():
            out[f"{name}_median"] = float(v.median())
    tau = _num("tau")
    if tau.notna().any():
        out["tau_fraction_above_0.1"] = float((tau > 0.1).mean())
        out["n_tau_above_1"] = int((tau > 1.0).sum())
    return out


def analyze(df: pd.DataFrame, cfg: Config, *, anchor: str | None = None,
            rng: np.random.Generator | None = None,
            lean: bool | None = None) -> tuple[pd.DataFrame, dict]:
    """Kinematics -> locus -> excess -> dust -> gauntlet.  Pure and offline.

    ``lean`` selects the catalogue-scale path (default: above
    ``LEAN_ABOVE_ROWS``).  In lean mode the returned frame holds only the
    excess-flagged rows, fully vetted; population counts in the summary still
    cover the whole input.
    """
    th = cfg.thresholds["ossuary"]
    s, k, e, c = th["sample"], th["kinematics"], th["excess"], th["contamination"]

    if df is None or not len(df):
        return pd.DataFrame(), {"verdict": "NO_DATA_REACHED", "n_input": 0,
                                "note": "empty sample table"}
    lean = (len(df) > LEAN_ABOVE_ROWS) if lean is None else bool(lean)
    t0 = time.monotonic()
    _log(f"analyze: {len(df):,} rows, {'lean' if lean else 'full'} path")

    work = df.copy()
    if "G" not in work.columns and "phot_g_mean_mag" in work.columns:
        work["Gmag"] = pd.to_numeric(work["phot_g_mean_mag"], errors="coerce")
        work["e_Gmag"] = 0.01
    anchor = anchor or _pick_anchor(work)

    # --- kinematics -------------------------------------------------------
    work = _kinematics(work, k, s)
    _log(f"kinematics done ({time.monotonic() - t0:.0f} s)")

    # --- empirical photosphere locus -------------------------------------
    bands = [b for b in ("W1", "W2", "W3", "W4") if f"{b}mag" in work.columns]
    nir = [b for b in ("J", "H") if f"{b}mag" in work.columns and anchor == "Ks"]
    # The locus is fitted on dwarfs only: giants have genuinely different
    # infrared colours (and dusty envelopes), so mixing them would both widen the
    # locus and import the very contaminant the channel is trying to exclude.
    ref = work[work["luminosity_class"] == "dwarf"]
    if len(ref) < max(50, int(e["locus_min_per_bin"])):
        ref = work
    loci = exc.fit_loci(ref, e, bands=tuple(bands + nir), anchor=anchor)
    usable = {b: loc for b, loc in loci.items() if loc.n_bins > 0}
    _log(f"locus fitted on {len(ref):,} reference stars: "
         f"{ {b: loc.n_bins for b, loc in usable.items()} } bins "
         f"({time.monotonic() - t0:.0f} s)")

    if not usable:
        return work, {"verdict": "NO_LOCUS", "n_input": int(len(df)),
                      "anchor": anchor,
                      "note": "no colour bin reached the minimum occupancy; "
                              "the empirical photosphere could not be built"}

    # Population counts over the WHOLE input, before any subsetting.
    feh_all = pd.to_numeric(work.get("feh"), errors="coerce")
    metal_poor_all = (feh_all <= s["feh_max"]).fillna(False)
    halo_all = work["halo_flag"].fillna(False).astype(bool)
    population = {
        "n_dwarfs": int((work["luminosity_class"] == "dwarf").sum()),
        "n_giants": int((work["luminosity_class"] == "giant").sum()),
        "n_metal_poor": int(metal_poor_all.sum()),
        "n_halo": int(halo_all.sum()),
        "n_null_reservoir_hosts": int((metal_poor_all | halo_all).sum()),
        "population_counts": {k2: int(v) for k2, v in
                              work["population"].value_counts().items()},
        "kinematic_method_counts": {k2: int(v) for k2, v in
                                    work["kinematic_method"].value_counts().items()},
    }

    if lean:
        flag, flagged = _excess_lean(work, usable, e, anchor)
        work["excess_flag"] = flag
        flagged["excess_flag"] = True
        flagged = _characterise(flagged, c, s, e, anchor, rng)
        vetted = vetting.vet(flagged, c, s, e, k) if len(flagged) else flagged
        n_all = len(work)
        del work
    else:
        work = exc.compute_excess(work, usable, e, anchor=anchor)
        work["excess_flag"] = exc.select_excess(work, e)
        # Dust characterisation only where an excess was flagged (the fit is a
        # 400-draw Monte Carlo per star and is wasted on the other 99.99%).
        flagged = work[work["excess_flag"]].copy()
        if len(flagged):
            flagged = _characterise(flagged, c, s, e, anchor, rng)
            for col in _DUST_COLS:
                work[col] = np.nan
                work.loc[flagged.index, col] = flagged[col]
        vetted = vetting.vet(work, c, s, e, k)
        n_all = len(vetted)
    _log(f"gauntlet done on {int(vetted['excess_flag'].sum()) if len(vetted) else 0:,} "
         f"flagged rows ({time.monotonic() - t0:.0f} s)")

    if len(vetted):
        vetted["candidate"] = vetted["excess_flag"] & (vetted["verdict"] == "surviving")
        n_flagged = int(vetted["excess_flag"].sum())
        counts = vetting.funnel_counts(vetted[vetted["excess_flag"]])
        rejects = {k2: int(v) for k2, v in
                   vetted.loc[vetted["excess_flag"], "reject_reason"]
                   .value_counts().items() if k2}
        n_cand = int(vetted["candidate"].sum())
        provenance = _survivor_provenance(vetted[vetted["candidate"]])
        cirrus = vetting.cirrus_correlation_test(vetted) if not lean else \
            {"tested": False, "reason": "lean path carries flagged rows only; "
                                        "tested at follow-up", "n": 0}
    else:
        vetted = pd.DataFrame(columns=["excess_flag", "candidate", "verdict",
                                       "reject_reason", "luminosity_class"])
        n_flagged, n_cand = 0, 0
        counts = vetting.funnel_counts(vetted)
        rejects = {}
        provenance = _survivor_provenance(vetted)
        cirrus = {"tested": False, "reason": "no flagged rows", "n": 0}

    summary = {
        "verdict": "OK",
        "anchor": anchor,
        "analysis_path": "lean" if lean else "full",
        "n_input": int(len(df)),
        # Sky-coverage honesty: rows from declination bands that hit the ADQL row
        # cap are an arbitrary subset (no ORDER BY), so their sky is biased.
        "n_from_row_limited_bands": int(
            pd.to_numeric(df.get("row_limit_hit"), errors="coerce").fillna(0).sum())
        if "row_limit_hit" in df.columns else 0,
        **population,
        "n_excess_flagged": n_flagged,
        "funnel": counts,
        "reject_reasons": rejects,
        "n_candidates": n_cand,
        # What each surviving row actually rests on.  The selection is a
        # disjunction (metal-poor OR halo-kinematic) and the metallicity may be
        # photometric, so "584 survivors" can mean 584 statements about one
        # unconfirmed estimator.  These counts make that readable without
        # opening the CSV.
        "survivor_provenance": provenance,
        "cirrus_correlation": cirrus,
        "chance_alignment_budget": vetting.expected_chance_alignments(int(n_all), c),
        "wien_peak_k": {b: exc.wien_peak_k(b) for b in ("W1", "W2", "W3", "W4")},
        "locus": {b: loc.to_dict() for b, loc in usable.items()},
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    _log(f"analyze complete: {n_flagged:,} flagged, {n_cand:,} candidates")
    return vetted, summary


# --------------------------------------------------------------------------
# Stage 3: follow-up on the shortlist (runner)
# --------------------------------------------------------------------------

def stage_followup(cfg: Config, cands: pd.DataFrame,
                   fetch_ebv=acq.fetch_ebv,
                   fetch_neighbours=acq.fetch_beam_neighbours,
                   fetch_simbad=acq.fetch_simbad) -> pd.DataFrame:
    """Per-candidate reddening, beam neighbours, and SIMBAD identity.

    Injectable fetchers so the whole stage is exercised offline in the tests.
    """
    th = cfg.thresholds["ossuary"]
    c = th["contamination"]
    if cands is None or not len(cands):
        return pd.DataFrame()

    out = cands.copy()
    try:
        ebv = fetch_ebv(out[["source_id", "ra", "dec"]])
        if ebv is not None and len(ebv):
            out = out.drop(columns=[x for x in ("ebv_sfd",) if x in out.columns])
            out = out.merge(ebv, on="source_id", how="left")
    except Exception as e:  # noqa: BLE001
        print(f"[ossuary] E(B-V) follow-up failed: {e!r}")

    rows = []
    for _, r in out.iterrows():
        cand = r.to_dict()
        try:
            nb = fetch_neighbours(float(cand["ra"]), float(cand["dec"]))
        except Exception as e:  # noqa: BLE001
            print(f"[ossuary] neighbour fetch failed for {cand.get('source_id')}: {e!r}")
            nb = None
        rows.append(vetting.beam_blend_verdict(cand, nb, c))
    for k2 in rows[0]:
        out[k2] = [r[k2] for r in rows]

    try:
        sb = fetch_simbad(out[["source_id", "ra", "dec"]])
        if sb is not None and len(sb):
            out = out.merge(sb, on="source_id", how="left")
    except Exception as e:  # noqa: BLE001
        print(f"[ossuary] SIMBAD follow-up failed: {e!r}")

    # Re-apply the two gates that only become testable after the follow-up.
    cg = vetting.cirrus_gate(out, c)
    for col in cg.columns:
        out[col] = cg[col]
    out["followup_verdict"] = np.where(
        (out["blend_verdict"].isin(["clean", "isolated"]))
        & out["cirrus_ok"].fillna(False),
        "surviving", "rejected")
    return out


# --------------------------------------------------------------------------
# Stage 4: report
# --------------------------------------------------------------------------

_HEADLINE = [
    "source_id", "ra", "dec", "l", "b", "track", "feh", "feh_provenance",
    "population", "kinematic_method", "v_tot_or_bound_kms", "v_tot_err_kms",
    "luminosity_class", "M_G", "bp_rp", "phot_g_mean_mag",
    "W1mag", "W2mag", "W3mag", "W4mag", "Ksmag",
    "chi_W1", "chi_W2", "chi_W3", "chi_W4", "chi_w1_w2", "chi_w1_w3",
    "w1_w2_obs", "w1_w2_locus", "t_dust_k", "t_dust_lo_k", "t_dust_hi_k",
    "tau", "tau_lo", "tau_hi", "dust_fit_chi2", "n_excess_bands",
    "registration_arcsec", "registration_arcsec_unpropagated",
    "chance_superposition_p", "ebv_sfd", "blend_verdict",
    "simbad_id", "simbad_otype", "verdict", "reject_reason",
]


# Committed CSVs stay small: a catalogue-scale flag list is written in full to a
# parquet that the workflow keeps as an artifact, and the CSV carries the most
# significant rows.  The summary records both counts.
MAX_CSV_ROWS = 20_000


def _rank_by_significance(df: pd.DataFrame) -> pd.DataFrame:
    chi = [c for c in ("chi_W1", "chi_W2", "chi_W3") if c in df.columns]
    if not chi:
        return df
    score = df[chi].apply(pd.to_numeric, errors="coerce").max(axis=1)
    return df.assign(_score=score).sort_values("_score", ascending=False).drop(columns="_score")


def write_results(cfg: Config, vetted: pd.DataFrame, summary: dict,
                  followup: pd.DataFrame | None = None) -> dict:
    d = out_dir(cfg)
    cols = [c for c in _HEADLINE if c in vetted.columns]

    if len(vetted):
        flagged = vetted[vetted.get("excess_flag", False)]
        if len(flagged):
            ranked = _rank_by_significance(flagged)
            ranked[cols].head(MAX_CSV_ROWS).to_csv(d / "excess_flagged.csv", index=False)
            if len(flagged) > MAX_CSV_ROWS:
                ranked[cols].to_parquet(d / "excess_flagged_full.parquet", index=False)
            summary["excess_flagged_csv_rows"] = int(min(len(flagged), MAX_CSV_ROWS))
        cands = vetted[vetted.get("candidate", False)]
        if len(cands):
            _rank_by_significance(cands)[cols].to_csv(d / "candidates.csv", index=False)
        giants = vetted[(vetted.get("luminosity_class") == "giant")
                        & vetted.get("excess_flag", False)]
        if len(giants):
            _rank_by_significance(giants)[cols].head(MAX_CSV_ROWS).to_csv(
                d / "giants_excess.csv", index=False)

    if followup is not None and len(followup):
        fcols = [c for c in _HEADLINE + ["followup_verdict", "neighbour_over_excess",
                                         "n_beam_neighbours"]
                 if c in followup.columns]
        followup[fcols].to_csv(d / "followup.csv", index=False)
        summary["followup"] = {
            "n": int(len(followup)),
            "verdicts": {k: int(v) for k, v in
                         followup["followup_verdict"].value_counts().items()},
            "blend_verdicts": {k: int(v) for k, v in
                               followup["blend_verdict"].value_counts().items()},
        }
        summary["n_candidates_after_followup"] = int(
            (followup["followup_verdict"] == "surviving").sum())

    (d / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (d / "REPORT.md").write_text(_report_md(summary))
    print("[ossuary] summary:", json.dumps(_jsonable(
        {k: v for k, v in summary.items() if k != "locus"}), indent=2)[:2000])
    return summary


def _report_md(s: dict) -> str:
    L = ["# OSSUARY — warm dust around stars that cannot make it", ""]
    L.append(f"**Verdict:** `{s.get('verdict')}`")
    if s.get("note"):
        L.append(f"**Note:** {s['note']}")
    L += ["", "## Sample", "",
          f"* input rows: {s.get('n_input', 0):,}",
          f"* dwarfs: {s.get('n_dwarfs', 0):,} | giants (analysed separately): "
          f"{s.get('n_giants', 0):,}",
          f"* metal-poor ([Fe/H] < -1): {s.get('n_metal_poor', 0):,}",
          f"* halo kinematics: {s.get('n_halo', 0):,}",
          f"* hosts with no natural reservoir (metal-poor OR halo): "
          f"{s.get('n_null_reservoir_hosts', 0):,}",
          f"* photosphere anchor: `{s.get('anchor')}`", ""]
    if s.get("population_counts"):
        L += ["### Galactic population", ""]
        L += [f"* `{k}`: {v:,}" for k, v in s["population_counts"].items()]
        L.append("")
    if s.get("funnel"):
        f = s["funnel"]
        removed = f.get("removed_by_stage", {})
        L += ["## Contamination funnel (excess-flagged sources only)", "",
              "Silverberg et al. 2018 measured a ~92% false-positive rate for "
              "AllWISE-selected infrared excesses, so the funnel reports what "
              "each stage removed, not only the running total.", "",
              "| stage | surviving | removed here |", "|---|---|---|"]
        for k, v in f.items():
            if not isinstance(v, int):
                continue
            name = k[len("after_"):] if k.startswith("after_") else k
            r = removed.get(name)
            L.append(f"| {k} | {v:,} | {r if r is not None else ''} |")
        L.append("")
    if s.get("reject_reasons"):
        L += ["### Rejections by first failing gate", ""]
        L += [f"* `{k}`: {v:,}" for k, v in s["reject_reasons"].items()]
        L.append("")
    cc = s.get("cirrus_correlation", {})
    if cc.get("tested"):
        L += ["### Population-level cirrus test", "",
              f"Spearman rho = {cc['spearman_rho']:+.3f} (p = {cc['p_value']:.3g}) "
              f"between flag rate and E(B-V) over {cc['n']:,} stars.",
              f"Flag rate by E(B-V) quartile: {cc.get('flag_rate_by_ebv_quartile')}",
              ""]
    cab = s.get("chance_alignment_budget", {})
    if cab:
        L += ["### Expected chance extragalactic alignments", "",
              f"Over {cab['n_stars']:,} stars, within the "
              f"{cab['registration_radius_arcsec']:.1f}\" registration radius:",
              f"* Hot DOGs (9e-6 arcsec^-2): **{cab['expected_hot_dogs']:.3g}** expected",
              f"* any AllWISE source bright enough to supply the excess: "
              f"**{cab['expected_allwise_interlopers']:.3g}** expected",
              f"* the same, in the full 6.5\" beam without the registration cut: "
              f"{cab['expected_allwise_interlopers_in_full_beam']:.3g} "
              f"(the cut buys a factor "
              f"{cab.get('leverage_of_registration_cut', float('nan')):.0f})", ""]
    L += ["## Candidates", "",
          f"* excess-flagged: {s.get('n_excess_flagged', 0):,}",
          f"* surviving the full gauntlet: **{s.get('n_candidates', 0):,}**"]
    if "n_candidates_after_followup" in s:
        L.append(f"* surviving per-object follow-up: "
                 f"**{s['n_candidates_after_followup']:,}**")
    L += ["", "See `docs/ossuary.md` for the claim, the novelty verdict and the "
          "contamination model.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Auditing a committed candidate table
# --------------------------------------------------------------------------

def audit_candidates(cfg: Config | None = None, *, path: str | Path | None = None,
                     out: str | Path | None = None) -> dict:
    """Re-read a committed ``candidates.csv`` and say what it actually contains.

    This runs offline on the committed table, so a result that is already on
    the branch can be re-examined without re-acquiring six million stars.  It
    reports the conjunction the claim needs, step by step, rather than the
    disjunction the funnel applied: a survivor is only a candidate for *this*
    channel if it is metal-poor **and** kinematically confirmed **and** its
    fitted excess is self-consistent as optically thin dust.

    On the 2026-09-22 table that chain is 584 -> 574 -> 15 -> 0.
    """
    cfg = cfg or load_config()
    d = out_dir(cfg)
    p = Path(path) if path else d / "candidates.csv"
    rec: dict = {"source": str(p), "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                               time.gmtime())}
    if not p.exists():
        rec["status"] = "NO_TABLE"
        return rec
    c = pd.read_csv(p)
    th = cfg.thresholds["ossuary"]
    e = th["excess"]
    feh = pd.to_numeric(c.get("feh"), errors="coerce")
    tau = pd.to_numeric(c.get("tau"), errors="coerce")
    pop = c.get("population", pd.Series("", index=c.index)).astype(str)
    metal_poor = (feh <= float(th["sample"]["feh_max"])).fillna(False)
    halo = pop.eq("halo")
    thin = (tau <= float(e.get("tau_max_debris", 0.1))).fillna(False)
    warm = pd.Series(False, index=c.index)
    for b in ("W1", "W2"):
        if f"chi_{b}" in c.columns:
            warm = warm | (pd.to_numeric(c[f"chi_{b}"], errors="coerce") >= 3.0)
    warm = warm.fillna(False)

    rec["status"] = "OK"
    rec["n_rows"] = int(len(c))
    rec["provenance"] = _survivor_provenance(
        c.assign(metal_poor=metal_poor, halo_flag=halo,
                 two_independent_arguments=metal_poor & halo,
                 **{k: v for k, v in vetting.provenance_flags(c).items()}))
    rec["conjunction"] = {
        "gauntlet_survivors": int(len(c)),
        "and_metal_poor": int(metal_poor.sum()),
        "and_halo_kinematic": int((metal_poor & halo).sum()),
        "and_optically_thin_fit": int((metal_poor & halo & thin).sum()),
        "and_a_W1_W2_excess_3sigma": int((metal_poor & halo & thin & warm).sum()),
    }
    rec["conjunction_without_kinematics"] = {
        "metal_poor": int(metal_poor.sum()),
        "and_optically_thin_fit": int((metal_poor & thin).sum()),
        "and_a_W1_W2_excess_3sigma": int((metal_poor & thin & warm).sum()),
    }
    g = vetting.optical_depth_gate(c, e)
    rec["optical_depth"] = {
        "n_fitted": int(tau.notna().sum()),
        "n_optically_thick_fit": int((~g["optical_depth_ok"]).sum()),
        "n_tau_implausible": int(g["tau_implausible"].sum()),
        "tau_max_physical": float(e.get("tau_max_physical", 1.0)),
        "tau_max_debris": float(e.get("tau_max_debris", 0.1)),
    }
    rec["band_significance"] = {
        b: {"n_ge_3": int((pd.to_numeric(c.get(f"chi_{b}"), errors="coerce") >= 3).sum()),
            "n_ge_5": int((pd.to_numeric(c.get(f"chi_{b}"), errors="coerce") >= 5).sum())}
        for b in ("W1", "W2", "W3", "W4") if f"chi_{b}" in c.columns}
    o = Path(out) if out else d / "survivor_audit.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(rec, indent=2, default=str))
    _log(f"audit: {rec['conjunction']}")
    return rec


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(cfg: Config | None = None, *, stage: str = "all",
        input_path: str | Path | None = None, g_max: float | None = None,
        limit_per_band: int = 400_000, do_followup: bool = True,
        max_followup: int = 200) -> dict:
    cfg = cfg or load_config()
    d = out_dir(cfg)
    sample_path = Path(input_path) if input_path else d / "sample.parquet"

    if stage in ("acquire", "all"):
        meta = stage_acquire(cfg, g_max=g_max, limit_per_band=limit_per_band)
        if meta.get("verdict") != "OK":
            return write_results(cfg, pd.DataFrame(), meta)
        if stage == "acquire":
            return meta

    if not sample_path.exists():
        s = {"verdict": "NO_DATA_REACHED", "n_input": 0,
             "note": f"no sample table at {sample_path}"}
        return write_results(cfg, pd.DataFrame(), s)

    df = pd.read_parquet(sample_path)
    vetted, summary = analyze(df, cfg)
    del df
    if summary.get("verdict") != "OK":
        return write_results(cfg, vetted, summary)

    # Checkpoint the measurement BEFORE the per-object follow-up: the follow-up
    # talks to three services per candidate and can fail or run long, and the
    # first catalogue-scale run lost everything to a step that never finished.
    write_results(cfg, vetted, summary)
    _log("summary.json checkpointed before follow-up")

    followup = None
    if do_followup and stage in ("followup", "all", "analyze"):
        cands = vetted[vetted["candidate"]].head(max_followup)
        if len(cands):
            _log(f"follow-up on {len(cands):,} candidates (cap {max_followup})")
            followup = stage_followup(cfg, cands)
    return write_results(cfg, vetted, summary, followup)


__all__ = ["run", "analyze", "stage_acquire", "stage_followup", "write_results",
           "audit_candidates", "out_dir"]

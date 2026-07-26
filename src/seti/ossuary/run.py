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


def analyze(df: pd.DataFrame, cfg: Config, *, anchor: str | None = None,
            rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, dict]:
    """Kinematics -> locus -> excess -> dust -> gauntlet.  Pure and offline."""
    th = cfg.thresholds["ossuary"]
    s, k, e, c = th["sample"], th["kinematics"], th["excess"], th["contamination"]

    if df is None or not len(df):
        return pd.DataFrame(), {"verdict": "NO_DATA_REACHED", "n_input": 0,
                                "note": "empty sample table"}

    work = df.copy()
    if "G" not in work.columns and "phot_g_mean_mag" in work.columns:
        work["Gmag"] = pd.to_numeric(work["phot_g_mean_mag"], errors="coerce")
        work["e_Gmag"] = 0.01
    anchor = anchor or _pick_anchor(work)

    # --- kinematics -------------------------------------------------------
    work = kin.classify(work, k)
    work["luminosity_class"] = kin.luminosity_class(work, s)
    work["M_G"] = kin.absolute_g(work)
    work["reduced_pm"] = kin.reduced_proper_motion(work)

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

    if not usable:
        return work, {"verdict": "NO_LOCUS", "n_input": int(len(df)),
                      "anchor": anchor,
                      "note": "no colour bin reached the minimum occupancy; "
                              "the empirical photosphere could not be built"}

    work = exc.compute_excess(work, usable, e, anchor=anchor)
    work["excess_flag"] = exc.select_excess(work, e)

    # Dust characterisation only where an excess was flagged (the fit is a
    # 400-draw Monte Carlo per star and is wasted on the other 99.99%).
    flagged = work[work["excess_flag"]].copy()
    if len(flagged):
        flagged = exc.characterise(flagged, e, anchor=anchor, rng=rng)
        for col in ("t_dust_k", "t_dust_lo_k", "t_dust_hi_k", "tau", "tau_lo",
                    "tau_hi", "dust_fit_chi2", "n_excess_bands"):
            work[col] = np.nan
            work.loc[flagged.index, col] = flagged[col]

    # --- contamination gauntlet ------------------------------------------
    vetted = vetting.vet(work, c, s, e, k)
    vetted["candidate"] = vetted["excess_flag"] & (vetted["verdict"] == "surviving")

    counts = vetting.funnel_counts(vetted[vetted["excess_flag"]])
    cirrus = vetting.cirrus_correlation_test(vetted)

    summary = {
        "verdict": "OK",
        "anchor": anchor,
        "n_input": int(len(df)),
        "n_dwarfs": int((vetted["luminosity_class"] == "dwarf").sum()),
        "n_giants": int((vetted["luminosity_class"] == "giant").sum()),
        "n_metal_poor": int(vetted["metal_poor"].sum()),
        "n_halo": int(vetted["halo_flag"].sum()),
        "n_null_reservoir_hosts": int(vetted["null_reservoir_host"].sum()),
        "population_counts": {k2: int(v) for k2, v in
                              vetted["population"].value_counts().items()},
        "kinematic_method_counts": {k2: int(v) for k2, v in
                                    vetted["kinematic_method"].value_counts().items()},
        "n_excess_flagged": int(vetted["excess_flag"].sum()),
        "funnel": counts,
        "reject_reasons": {k2: int(v) for k2, v in
                           vetted.loc[vetted["excess_flag"], "reject_reason"]
                           .value_counts().items() if k2},
        "n_candidates": int(vetted["candidate"].sum()),
        "cirrus_correlation": cirrus,
        "locus": {b: loc.to_dict() for b, loc in usable.items()},
    }
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


def write_results(cfg: Config, vetted: pd.DataFrame, summary: dict,
                  followup: pd.DataFrame | None = None) -> dict:
    d = out_dir(cfg)
    cols = [c for c in _HEADLINE if c in vetted.columns]

    if len(vetted):
        flagged = vetted[vetted.get("excess_flag", False)]
        if len(flagged):
            flagged[cols].to_csv(d / "excess_flagged.csv", index=False)
        cands = vetted[vetted.get("candidate", False)]
        if len(cands):
            cands[cols].to_csv(d / "candidates.csv", index=False)
        giants = vetted[(vetted.get("luminosity_class") == "giant")
                        & vetted.get("excess_flag", False)]
        if len(giants):
            giants[cols].to_csv(d / "giants_excess.csv", index=False)

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
        L += ["## Contamination funnel (excess-flagged sources only)", "",
              "| stage | surviving |", "|---|---|"]
        L += [f"| {k} | {v:,} |" for k, v in s["funnel"].items()]
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
    if summary.get("verdict") != "OK":
        return write_results(cfg, vetted, summary)

    followup = None
    if do_followup and stage in ("followup", "all", "analyze"):
        cands = vetted[vetted["candidate"]].head(max_followup)
        if len(cands):
            followup = stage_followup(cfg, cands)
    return write_results(cfg, vetted, summary, followup)


__all__ = ["run", "analyze", "stage_acquire", "stage_followup", "write_results",
           "out_dir"]

"""RING stage orchestration.

    ring --stage {probe,acquire,screen,assess,all} [--leg {wd,pulsar,bd,ffp,all}]
         [--shard i/n] [--dec-band k] [--out DIR]

Every stage writes under ``results/ring/<leg>/`` and checkpoints; ``assess``
reads whatever the legs produced and composes one verdict that names any leg
that reached no data.  Large intermediates (the white-dwarf sample, the
pulsar match tables) stay in parquet files the workflow keeps as artifacts;
the committed files are the JSON summaries, the report and capped CSVs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import load_config
from . import acquire as acq
from . import assess as ass
from . import screen as scr
from .screen import _records as rscr_records

LEGS = ("wd", "pulsar", "bd", "ffp")
MAX_CSV_ROWS = 20_000


def load_ring_config(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else load_config().root
    with (root / "config" / "ring.yaml").open() as fh:
        return yaml.safe_load(fh)


def out_root(out: str | Path | None = None) -> Path:
    d = Path(out) if out else load_config().root / "results" / "ring"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log(msg: str) -> None:
    print(f"[ring {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def provenance() -> dict:
    """Which run wrote a file, and when.

    Every committed JSON carries this, because the files of one leg can outlive
    the run that wrote them: a later dispatch over other legs re-composes
    ``summary.json`` from whatever is in the checkout, and without a stamp a
    summary can quote numbers from a run it does not describe.
    """
    import os

    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": os.environ.get("GITHUB_RUN_ID") or None,
            "git_sha": os.environ.get("GITHUB_SHA") or None}


def _write(path: Path, obj) -> None:
    if isinstance(obj, dict) and "provenance" not in obj:
        obj = {**obj, "provenance": provenance()}
    acq.write_json(path, ass.json_safe(obj))


def _read_json(path: Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                   # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def stage_probe(cfg: dict, out: Path, *, probe_fn=None, list_fn=None, importer=None) -> dict:
    """Is each endpoint reachable, and does it expose what the leg needs?"""
    from ..ossuary.acquire import probe_columns

    probe_fn = probe_fn or probe_columns
    rec: dict = {"probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "legs": {}}
    # wd: the Gaia-archive table
    t0 = time.monotonic()
    table, cols = acq.probe_gaia_wd_table(cfg["wd"]["gaia_table_candidates"], probe_fn)
    rec["legs"]["wd"] = {"gaia_wd_table": table, "columns": cols,
                         "status": "REACHED_WITH_PRODUCT" if table else "NOT_REACHED",
                         "elapsed_s": round(time.monotonic() - t0, 1)}
    # pulsar: which route can answer
    t0 = time.monotonic()
    importer = importer or __import__
    routes = {}
    try:
        importer("psrqpy")
        routes["psrqpy"] = "importable"
    except Exception as exc:                            # noqa: BLE001
        routes["psrqpy"] = f"not importable: {exc!r}"[:120]
    rec["legs"]["pulsar"] = {"routes": routes, "elapsed_s": round(time.monotonic() - t0, 1)}
    # bd / ffp: VizieR tables by role
    if list_fn is None:
        def list_fn(cat, patterns, required):
            return acq.discover_vizier_table(cat, patterns, required)
    for leg, cat, pats, req in (("bd", cfg["bd"]["vizier_catalogue"], acq.BD_ROLES,
                                 ("name", "ra", "dec", "spt")),
                                ("ffp", cfg["ffp"]["vizier_catalogue"], acq.FFP_ROLES,
                                 ("name", "lbol"))):
        t0 = time.monotonic()
        try:
            d = list_fn(cat, pats, req)
        except Exception as exc:                        # noqa: BLE001
            d = {"status": "QUERY_FAILED", "error": repr(exc)[:300]}
        d["elapsed_s"] = round(time.monotonic() - t0, 1)
        rec["legs"][leg] = d
    _write(out / "probe.json", rec)
    _log("probe: " + json.dumps({k: v.get("status", v.get("routes")) for k, v in
                                 rec["legs"].items()}))
    return rec


# --------------------------------------------------------------------------
# acquire
# --------------------------------------------------------------------------

def stage_acquire(cfg: dict, out: Path, leg: str, *, shard: int = 0, n_shards: int = 1,
                  dec_band: int | None = None, fetchers: dict | None = None) -> dict:
    fetchers = fetchers or {}
    d = out / leg
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if leg == "wd":
        df, meta = (fetchers["wd"](d / "chunks", cfg) if "wd" in fetchers
                    else acq.fetch_wd_leg(d / "chunks", cfg, dec_band=dec_band))
        if len(df):
            df = acq.harmonise_wd(df)
            tag = "" if dec_band is None else f"_band{int(dec_band)}"
            df.to_parquet(d / f"sample{tag}.parquet", index=False)
        meta.update({"n": int(len(df)), "status": "OK" if len(df) else "NO_DATA_REACHED",
                     "dec_band": dec_band, "elapsed_s": round(time.monotonic() - t0, 1)})
        _write(d / ("acquire.json" if dec_band is None else f"acquire_band{int(dec_band)}.json"),
               meta)
    elif leg == "pulsar":
        psr, meta = (fetchers["pulsar"](cfg) if "pulsar" in fetchers
                     else acq.fetch_pulsars(cfg))
        meta["n"] = int(len(psr))
        if len(psr):
            psr.to_csv(d / "pulsars.csv", index=False)
            matches, mm = (fetchers["pulsar_matches"](psr, cfg) if "pulsar_matches" in fetchers
                           else acq.fetch_pulsar_counterparts(psr, cfg))
            meta["matches"] = mm
            for k, v in matches.items():
                if v is not None and len(v):
                    v.to_parquet(d / f"matches_{k}.parquet", index=False)
        meta["status"] = "OK" if len(psr) else "NO_DATA_REACHED"
        meta["elapsed_s"] = round(time.monotonic() - t0, 1)
        _write(d / "acquire.json", meta)
    elif leg == "bd":
        tp = d / "targets.csv"
        # Every shard fetches the (small) target list itself: a targets.csv in
        # the checkout is the COMMITTED list of an earlier run, and reusing it
        # would put shards of one run on different target lists.
        targets, meta = (fetchers["bd_targets"](cfg) if "bd_targets" in fetchers
                         else acq.fetch_bd_targets(cfg))
        if len(targets):
            targets.to_csv(tp, index=False)
        meta["n_targets"] = int(len(targets))
        _write(d / "acquire_targets.json", meta)
        roll = {}
        # A marker written BEFORE the NEOWISE loop: if the job's wall clock
        # kills the process (run 35752692549: `timeout 5400` fired mid-loop and
        # no acquire record was ever written, so the summary said NOT_RUN for
        # a leg whose screen then reported OK), assess finds IN_PROGRESS and
        # names the leg as incomplete instead of silently trusting it.
        _write(d / f"acquire_shard{shard}.json",
               {**meta, "status": "IN_PROGRESS", "shard": shard, "n_shards": n_shards})
        if len(targets):
            roll = (fetchers["bd_neowise"](targets, d, cfg, shard, n_shards)
                    if "bd_neowise" in fetchers
                    else acq.fetch_bd_neowise(targets, d, cfg, shard=shard, n_shards=n_shards))
        status = "OK" if len(targets) else "NO_DATA_REACHED"
        if roll.get("stopped_on_budget"):
            status = "PARTIAL_TIME_BUDGET"
        meta = {**meta, "neowise": roll, "shard": shard, "n_shards": n_shards,
                "status": status,
                "elapsed_s": round(time.monotonic() - t0, 1)}
        _write(d / f"acquire_shard{shard}.json", meta)
    elif leg == "ffp":
        df, meta = (fetchers["ffp"](cfg) if "ffp" in fetchers else acq.fetch_ffp_targets(cfg))
        if len(df):
            df.to_csv(d / "targets.csv", index=False)
        meta.update({"n": int(len(df)), "status": "OK" if len(df) else "NO_DATA_REACHED",
                     "elapsed_s": round(time.monotonic() - t0, 1)})
        _write(d / "acquire.json", meta)
    else:
        raise ValueError(f"unknown leg {leg}")
    _log(f"acquire {leg}: {meta.get('status')} n={meta.get('n', meta.get('n_targets'))} "
         f"({meta.get('elapsed_s')} s)")
    return meta


# --------------------------------------------------------------------------
# screen
# --------------------------------------------------------------------------

def _wd_sample(d: Path) -> pd.DataFrame:
    parts = sorted(d.glob("sample*.parquet"))
    if not parts:
        chunks = sorted((d / "chunks").glob("wd_dec_*.parquet"))
        if not chunks:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(p) for p in chunks], ignore_index=True)
        return acq.harmonise_wd(df.drop_duplicates("source_id"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df.drop_duplicates("source_id").reset_index(drop=True)


def _rank(df: pd.DataFrame, cols) -> pd.DataFrame:
    have = [c for c in cols if c in df.columns]
    if not have:
        return df
    score = df[have].apply(pd.to_numeric, errors="coerce").max(axis=1)
    return df.assign(_s=score).sort_values("_s", ascending=False).drop(columns="_s")


def stage_screen(cfg: dict, out: Path, leg: str, *, rng=None) -> dict:
    d = out / leg
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if leg == "wd":
        df = _wd_sample(d)
        screened, s = scr.screen_wd(df, cfg, rng=rng)
        if len(screened):
            screened.to_parquet(d / "screened.parquet", index=False)
            cols = [c for c in ass._WD_HEADLINE if c in screened.columns]
            flagged = screened[screened["excess_flag"].fillna(False).astype(bool)]
            _rank(flagged, ("chi_W2", "chi_W1"))[cols].head(MAX_CSV_ROWS).to_csv(
                d / "excess.csv", index=False)
    elif leg == "pulsar":
        pp = d / "pulsars.csv"
        psr = pd.read_csv(pp) if pp.exists() else pd.DataFrame()
        matches = {}
        for k in ("allwise", "catwise"):
            mp = d / f"matches_{k}.parquet"
            matches[k] = pd.read_parquet(mp) if mp.exists() else pd.DataFrame()
        screened, s = scr.screen_pulsars(psr, matches, cfg)
        if len(screened):
            cols = [c for c in ass._PSR_HEADLINE if c in screened.columns]
            screened[cols].to_csv(d / "pulsars_screened.csv", index=False)
            screened[screened["verdict"] != "no_counterpart"][cols].to_csv(
                d / "counterparts.csv", index=False)
    elif leg == "bd":
        tp = d / "targets.csv"
        targets = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        eps = sorted(d.glob("epochs_bd_shard*.csv"))
        epochs = pd.concat([pd.read_csv(p) for p in eps], ignore_index=True) if eps \
            else pd.DataFrame()
        screened, s = scr.screen_bd(epochs, targets, cfg)
        if len(screened):
            screened.to_csv(d / "bd_screened.csv", index=False)
        s["n_epoch_rows"] = int(len(epochs))
    elif leg == "ffp":
        tp = d / "targets.csv"
        df = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        screened, s = scr.screen_ffp(df, cfg)
        if len(screened):
            screened.to_csv(d / "ffp_screened.csv", index=False)
    else:
        raise ValueError(f"unknown leg {leg}")
    s["elapsed_s"] = round(time.monotonic() - t0, 1)
    _write(d / "screen.json", s)
    _log(f"screen {leg}: {s.get('status')} " + json.dumps(
        {k: v for k, v in s.items() if isinstance(v, (int, float, str)) and k != 'status'})[:400])
    return s


# --------------------------------------------------------------------------
# assess
# --------------------------------------------------------------------------

def stage_assess(cfg: dict, out: Path, *, followup: bool = True,
                 followup_fetchers: dict | None = None) -> dict:
    t0 = time.monotonic()
    legs, acq_meta = {}, {}
    for leg in LEGS:
        d = out / leg
        legs[leg] = _read_json(d / "screen.json") or {"status": "NOT_RUN"}
        a = _read_json(d / "acquire.json")
        if a is None:
            shards = sorted(d.glob("acquire_shard*.json"))
            a = {"shards": [_read_json(p) for p in shards]} if shards else None
            if a and a["shards"]:
                st = [(x or {}).get("status") for x in a["shards"]]
                a["route"] = (a["shards"][0] or {}).get("route")
                a["shard_status"] = st
                if all(x == "OK" for x in st):
                    a["status"] = "OK"
                elif any(x == "IN_PROGRESS" for x in st):
                    a["status"] = "KILLED_IN_PROGRESS"
                elif any(x in ("OK", "PARTIAL_TIME_BUDGET") for x in st):
                    a["status"] = "PARTIAL"
                else:
                    a["status"] = "NO_DATA_REACHED"
        acq_meta[leg] = a or {"status": "NOT_RUN"}

    # A leg whose screen ran but whose acquisition did not finish is not a
    # complete leg, and its coverage is part of its result: the NEOWISE leg of
    # run 35752692549 screened 11 of 232 targets and was reported as plain OK.
    bd = legs.get("bd") or {}
    if bd.get("status") == "OK":
        n_t = int(bd.get("n_targets") or 0)
        n_e = int(bd.get("n_with_epochs") or 0)
        frac = (n_e / n_t) if n_t else 0.0
        bd["epoch_coverage"] = round(frac, 3)
        a_st = acq_meta["bd"].get("status")
        why = []
        if a_st != "OK":
            why.append(f"acquisition {a_st}")
        if frac < float(cfg["bd"].get("min_epoch_coverage", 0.5)):
            why.append(f"{n_e}/{n_t} targets with epochs")
        if why:
            bd["coverage_degraded"] = "; ".join(why)

    # White-dwarf shortlist follow-up: ring-band candidates plus every survivor
    # of the catalogue gates, capped.
    wd_dir = out / "wd"
    sp = wd_dir / "screened.parquet"
    if legs["wd"].get("status") == "OK" and sp.exists():
        screened = pd.read_parquet(sp)
        flagged = screened[screened["excess_flag"].fillna(False).astype(bool)]
        short = flagged[flagged["verdict"] == "surviving"]
        short = pd.concat([short[short["ring_candidate"].astype(bool)],
                           short[~short["ring_candidate"].astype(bool)]])
        short = short.head(int(cfg["wd"]["shortlist_max"]))
        legs["wd"]["n_shortlist"] = int(len(short))
        if followup and len(short):
            ff = followup_fetchers
            if ff is None:
                from ..acquire.science import fetch_known_disks, fetch_simbad_context
                from ..discriminate.blend import fetch_neighbours
                ff = {"fetch_known_disks": fetch_known_disks,
                      "fetch_neighbours": fetch_neighbours,
                      "fetch_simbad": fetch_simbad_context}
            _log(f"wd follow-up on {len(short)} shortlisted white dwarfs")
            fu = ass.wd_followup(short, cfg, **ff)
            cols = [c for c in ass._WD_HEADLINE if c in fu.columns]
            fu[cols].to_csv(wd_dir / "followup.csv", index=False)
            ring_after = fu[fu["ring_candidate"].astype(bool)
                            & (fu["followup_verdict"] == "surviving")]
            legs["wd"]["n_ring_candidates_after_followup"] = int(len(ring_after))
            legs["wd"]["followup_neighbour_timeouts"] = int(fu.attrs.get("n_neighbour_timeouts", 0))
            legs["wd"]["n_followup"] = int(len(fu))
            legs["wd"]["n_surviving_after_followup"] = int(
                (fu["followup_verdict"] == "surviving").sum())
            legs["wd"]["followup_reasons"] = {k: int(v) for k, v in
                                              fu["followup_reason"].value_counts().items()
                                              if k}
            legs["wd"]["ring_candidates"] = ring_after[
                [c for c in ("source_id", "wd_name", "ra", "dec", "dist_pc", "teff",
                             "t_ring_k", "t_ring_lo_k", "t_ring_hi_k", "tau_ring",
                             "ring_radius_au", "chi_W1", "chi_W2", "simbad_id")
                 if c in ring_after.columns]].head(50).to_dict("records")
        elif len(short):
            legs["wd"]["n_ring_candidates_after_followup"] = None
            legs["wd"]["followup_reasons"] = {"skipped": int(len(short))}

    psr_screened = None
    pp = out / "pulsar" / "pulsars_screened.csv"
    if pp.exists():
        psr_screened = pd.read_csv(pp)
        surv = psr_screened[psr_screened["verdict"] == "surviving"]
        legs["pulsar"]["survivors"] = rscr_records(surv[[c for c in (
            "jname", "bname", "localised", "pos_err_arcsec", "allwise_dist_arcsec",
            "catwise_dist_arcsec", "colour_source", "W1mag", "W2mag", "W1mag_cat",
            "W2mag_cat", "w1_w2", "w1_w2_err", "t_colour_k", "shape_class", "p_chance",
            "edot_w", "dist_kpc", "f_min_500K_W2") if c in surv.columns]].head(50))
    verdict, degraded = ass.compose_verdict(legs)
    prov = provenance()
    leg_prov = {k: (v.get("provenance") or {"generated_at": None, "run_id": None,
                                            "note": "file predates provenance stamps"})
                for k, v in legs.items() if v.get("status") != "NOT_RUN"}
    summary = {
        "verdict": verdict,
        "degraded_legs": degraded,
        "generated_at": prov["generated_at"],
        "assessed_at": prov["generated_at"],
        "run_id": prov["run_id"],
        "git_sha": prov["git_sha"],
        # Which run produced each leg's numbers: a summary composed after a
        # partial dispatch mixes legs from different runs, and says so here.
        "leg_provenance": leg_prov,
        "legs": legs,
        "acquisition": {k: {kk: vv for kk, vv in (v or {}).items()
                            if kk not in ("bands", "shards", "scoreboard")}
                        for k, v in acq_meta.items()},
        "two_target_vet": ass.two_target_vet(psr_screened, cfg),
        "ring_band_k": [cfg["ring"]["t_min_k"], cfg["ring"]["t_max_k"]],
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    summary["consistency"] = ass.consistency_checks(summary)
    ass.write_summary(out, summary)
    _log(f"assess: {verdict}")
    return summary


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(stage: str = "all", leg: str = "all", *, out: str | Path | None = None,
        shard: str = "0/1", dec_band: int | None = None, followup: bool = True,
        cfg: dict | None = None, seed: int = 20260921) -> dict:
    cfg = cfg or load_ring_config()
    o = out_root(out)
    i, n = (int(x) for x in str(shard).split("/"))
    legs = LEGS if leg == "all" else (leg,)
    rng = np.random.default_rng(seed)
    res: dict = {}
    if stage in ("probe", "all"):
        res["probe"] = stage_probe(cfg, o)
    if stage in ("acquire", "all"):
        for lg in legs:
            try:
                res[f"acquire_{lg}"] = stage_acquire(cfg, o, lg, shard=i, n_shards=n,
                                                     dec_band=dec_band)
            except Exception as exc:                    # noqa: BLE001
                _log(f"acquire {lg} FAILED: {exc!r}")
                _write(o / lg / "acquire.json", {"status": "NO_DATA_REACHED",
                                                 "error": repr(exc)[:500]})
    if stage in ("screen", "all"):
        for lg in legs:
            try:
                res[f"screen_{lg}"] = stage_screen(cfg, o, lg, rng=rng)
            except Exception as exc:                    # noqa: BLE001
                _log(f"screen {lg} FAILED: {exc!r}")
                _write(o / lg / "screen.json", {"status": "NO_DATA_REACHED",
                                                "error": repr(exc)[:500]})
    if stage in ("assess", "all"):
        res["assess"] = stage_assess(cfg, o, followup=followup)
    return res


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", choices=("probe", "acquire", "screen", "assess", "all"),
                   default="all")
    p.add_argument("--leg", choices=LEGS + ("all",), default="all")
    p.add_argument("--shard", default="0/1", help="i/n for the NEOWISE (bd) leg")
    p.add_argument("--dec-band", type=int, default=None,
                   help="pull only this 15-degree declination band of the WD leg")
    p.add_argument("--out", default=None, help="output directory (default results/ring)")
    p.add_argument("--no-followup", action="store_true",
                   help="skip the runner-side white-dwarf shortlist follow-up")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ring", description=__doc__)
    add_arguments(p)
    a = p.parse_args(argv)
    run(a.stage, a.leg, out=a.out, shard=a.shard, dec_band=a.dec_band,
        followup=not a.no_followup)


if __name__ == "__main__":
    main()

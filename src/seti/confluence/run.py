"""CONFLUENCE: do the same stars sit in the tails of independent channels?

Stages
------
``inventory``  offline.  Every channel's committed per-star output: parent size,
               how many carry a Gaia id / a position / a score / a flag, which
               are survivor-only, and the parent-overlap matrix of every pair.
               Writes ``inventory.json``.
``harvest``    runner.  Reads other channels' run artifacts laid out under
               ``--harvest-dir`` and extracts per-star scores (``harvest.py``);
               acquires the ACCEL parent (Gaia ``nss_acceleration_astro``).
               Full score tables go to ``--scores-dir`` (an artifact, not git).
``joint``      runner (or offline on committed scores).  Unifies star identity
               across channels, computes each channel's tails over its WHOLE
               parent, and keeps only stars that sit in >= 2 parents of an
               independent pair.  Writes ``joint_long.parquet`` and
               ``channels.json`` -- small enough to commit, and everything
               ``assess`` needs.
``acquire``    runner.  Gaia covariates (+ neighbour density) for the joint stars,
               SIMBAD / VSX / Gaia-vari tags for the tail members and a random
               baseline sample.  Writes ``covariates.parquet``, ``tags.parquet``,
               ``acquire.json``.
``assess``     offline.  The exact stratified-permutation test for every
               independent pair and triple, the positive-control check, the
               residual after removing known classes, injection-recovery, the
               traced member list.  Writes ``summary.json``, ``tests.csv``,
               ``members.csv``, ``residual.csv``, ``injection.json``.

Verdicts (``summary.json["verdict"]``)
--------------------------------------
``NO_DATA_REACHED``              no channel supplied a parent with scores
``NO_TESTABLE_PAIR``             no independent pair has a joint parent >= min_joint
``CONTROLS_FAILED``              positive controls did not lead the overlaps / the
                                 injection was not recovered -- nothing below is believed
``NO_EXCESS_CONFLUENCE``         tested, controls pass, no pair exceeds its matched null
``CONFLUENCE_KNOWN_CLASSES_ONLY`` excess overlap exists and known classes account for it
``RESIDUAL_CONFLUENCE``          excess overlap survives the removal of known classes;
                                 the residual members are listed and traced
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import registry as R
from .core import add_tails, channel_pairs, injection_trials, run_tests, trace_members, unify
from .tagger import tag_table

OUT = R.RESULTS / "confluence"
QS = (0.01, 0.05)
MODES = ("0.01", "0.05", "flag")
MIN_JOINT = 30


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("GITHUB_SHA", "unknown")


def _prov() -> dict:
    return {"generated_utc": _now(), "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "git_sha": _sha()}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, default=str, allow_nan=False))


def load_frames(prefer_harvest: bool = True) -> dict[str, pd.DataFrame]:
    frames = {}
    for spec in R.REGISTRY:
        if spec.status == "tail_only":
            continue
        try:
            f = spec.loader() if spec.loader else None
        except Exception as e:  # noqa: BLE001
            print(f"[confluence] loader {spec.name} failed: {e!r}")
            f = None
        if f is not None and len(f):
            frames[spec.name] = f
    if prefer_harvest and "cenotaph" in frames:
        frames.pop("cenotaph_committed", None)
    return fill_positions(frames)


def fill_positions(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Give Gaia-keyed rows without a position (CENOTAPH's greyfit, RING) the
    position some other file records for the same source_id, so position-only
    channels (ZTF, TAILINGS, Kepler) can meet them.  Sources: every frame's own
    Gaia rows, the committed CENOTAPH shells, the runner's covariate table."""
    parts = [f.loc[f["source_id"].notna() & f["ra"].notna(), ["source_id", "ra", "dec"]]
             for f in frames.values()]
    for p in sorted((R.RESULTS / "cenotaph").glob("shell_*.parquet")):
        parts.append(pd.read_parquet(p, columns=["source_id", "ra", "dec"]))
    cp = OUT / "covariates.parquet"
    if cp.exists():
        parts.append(pd.read_parquet(cp, columns=["source_id", "ra", "dec"]))
    if not parts:
        return frames
    cat = pd.concat(parts, ignore_index=True).dropna()
    cat["source_id"] = cat["source_id"].astype("int64")
    cat = cat.drop_duplicates("source_id").set_index("source_id")
    for f in frames.values():
        need = f["source_id"].notna() & f["ra"].isna()
        if need.any():
            sid = f.loc[need, "source_id"].astype("int64")
            f.loc[need, "ra"] = cat["ra"].reindex(sid).to_numpy()
            f.loc[need, "dec"] = cat["dec"].reindex(sid).to_numpy()
    return frames


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def stage_inventory(out: Path = OUT) -> dict:
    frames = load_frames()
    specs = R.by_name()
    inv = {"channels": {}, "tail_only": R.TAIL_ONLY, **_prov()}
    for s in R.REGISTRY:
        f = frames.get(s.name)
        inv["channels"][s.name] = {
            "status": s.status if f is None or s.status != "pending_harvest" else "scored",
            "family": s.family, "primary": sorted(s.primary), "support": sorted(s.support),
            "observable": s.observable, "parent": s.parent, "notes": s.notes,
            "present": f is not None,
            "n_parent": int(len(f)) if f is not None else 0,
            "n_gaia_id": int(f["source_id"].notna().sum()) if f is not None else 0,
            "n_position": int(f["ra"].notna().sum()) if f is not None else 0,
            "n_scored": int(np.isfinite(f["score"].to_numpy(float)).sum()) if f is not None else 0,
            "n_flag": int(f["flag"].sum()) if f is not None else 0,
        }
    L, urep = unify(frames)
    inv["unify"] = urep
    parents = {ch: set(g["star"]) for ch, g in L.groupby("channel")}
    names = sorted(parents)
    mat = {}
    for a in names:
        for b in names:
            if a < b:
                n = len(parents[a] & parents[b])
                if n:
                    mat[f"{a}|{b}"] = {"n_joint": n,
                                       "independent_strict": R.independent(specs[a], specs[b]),
                                       "independent_relaxed": R.independent(specs[a], specs[b],
                                                                            "relaxed")}
    inv["joint_parent_sizes"] = dict(sorted(mat.items(), key=lambda kv: -kv[1]["n_joint"]))
    _write(out / "inventory.json", inv)
    print(f"[confluence] inventory: {len(frames)} channels with parents; "
          f"{sum(1 for v in mat.values() if v['independent_strict'])} strict-independent "
          f"pairs share stars")
    return inv


# ---------------------------------------------------------------------------
# harvest (runner)
# ---------------------------------------------------------------------------


def stage_resolve(out: Path = OUT) -> dict:
    """ARC keys its stars on KIC / TIC; resolve them to positions (runner)."""
    xi = R.RESULTS / "arc" / "xi_table.csv"
    if not xi.exists():
        return {"status": "NO_ARC_TABLE"}
    from .acquire import fetch_id_positions
    x = pd.read_csv(xi, usecols=["star_key", "star_id", "mission"])
    led2: list = []
    parts, arep = [], {}
    for mission in ("kepler", "tess"):
        ids = pd.to_numeric(x.loc[x["mission"] == mission, "star_id"], errors="coerce")
        ids = ids.dropna().astype("int64")
        if not len(ids):
            continue
        pos, arep[mission] = fetch_id_positions(mission, ids, led2)
        if len(pos):
            pos["star_key"] = mission + ":" + pos["id"].astype("int64").astype(str)
            parts.append(pos[["star_key", "ra", "dec"]])
    if parts:
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).drop_duplicates("star_key").to_csv(out / "arc_positions.csv",
                                                             index=False)
    rep = {"status": {k: v["status"] for k, v in arep.items()},
           "n": int(sum(len(p) for p in parts)), "ledger": led2[-10:]}
    print(f"[confluence] resolve: arc positions {rep['n']} {rep['status']}")
    return rep


def stage_accel(scores_dir: Path = R.SCORES_DIR, out: Path = OUT) -> dict:
    """ACCEL's parent is a whole Gaia table; acquired here, recorded in accel.json."""
    from .acquire import fetch_accel_nss
    led: list = []
    try:
        a = fetch_accel_nss(led)
        a["_s"] = pd.to_numeric(a["significance"], errors="coerce")
        fr = R.standardise(a, sid="source_id", score="_s", flag=None)
        Path(scores_dir).mkdir(parents=True, exist_ok=True)
        fr.to_parquet(Path(scores_dir) / "accel_nss.parquet", index=False)
        rep = {"status": "OK", "n_emitted": int(len(fr)), "ledger": led}
    except Exception as e:  # noqa: BLE001
        rep = {"status": "NO_DATA_REACHED", "error": repr(e)[:400], "ledger": led}
    rep.update(_prov())
    _write(out / "accel.json", rep)
    print(f"[confluence] accel: {rep['status']} {rep.get('n_emitted')}")
    return rep


def stage_harvest(harvest_dir: Path, scores_dir: Path, out: Path = OUT,
                  with_accel: bool = True) -> dict:
    from .harvest import harvest
    rep = harvest(harvest_dir, scores_dir)
    if with_accel:
        rep["channels"]["accel_nss"] = stage_accel(scores_dir, out)
    rep.update(_prov())
    # the manifest can be long; keep it, it is how the next dispatch learns layouts
    _write(out / "harvest.json", rep)
    for ch, r in rep["channels"].items():
        print(f"[confluence] harvest {ch}: {r.get('status')} emitted={r.get('n_emitted')}")
    return rep


# ---------------------------------------------------------------------------
# joint
# ---------------------------------------------------------------------------


def stage_joint(out: Path = OUT, rule: str = "relaxed") -> dict:
    frames = load_frames()
    specs = R.by_name()
    if not frames:
        _write(out / "channels.json", {"verdict": "NO_DATA_REACHED", **_prov()})
        return {"verdict": "NO_DATA_REACHED"}
    anchor = None
    cp = out / "covariates.parquet"
    if cp.exists():
        anchor = pd.read_parquet(cp, columns=["source_id", "ra", "dec"])
    L, urep = unify(frames, anchor=anchor)
    L = add_tails(L, QS)
    pairs = channel_pairs(specs, sorted(L["channel"].unique()), rule)
    parents = {ch: set(g["star"]) for ch, g in L.groupby("channel")}
    keep: set = set()
    pair_sizes = {}
    for a, b in pairs:
        j = parents[a] & parents[b]
        pair_sizes[f"{a}|{b}"] = len(j)
        keep |= j
    J = L[L["star"].isin(keep)].copy()
    cols = ["channel", "star", "source_id", "ra", "dec", "score", "pct", "flag"] + \
        [f"tail_q{q:g}" for q in QS]
    out.mkdir(parents=True, exist_ok=True)
    J[cols].to_parquet(out / "joint_long.parquet", index=False)
    meta = {"channels": {}, "unify": urep, "rule_for_keep": rule,
            "pair_joint_sizes": dict(sorted(pair_sizes.items(), key=lambda kv: -kv[1])),
            "n_joint_stars": len(keep), **_prov()}
    for ch, g in L.groupby("channel"):
        s = pd.to_numeric(g["score"], errors="coerce")
        meta["channels"][ch] = {
            "n_parent": int(len(g)), "n_scored": int(np.isfinite(s.to_numpy(float)).sum()),
            "n_flag": int(g["flag"].sum()),
            **{f"threshold_q{q:g}": (float(s[g[f"tail_q{q:g}"]].min())
                                     if g[f"tail_q{q:g}"].any() else None) for q in QS},
            **{f"n_tail_q{q:g}": int(g[f"tail_q{q:g}"].sum()) for q in QS},
            "n_in_joint": int(g["star"].isin(keep).sum()),
        }
    _write(out / "channels.json", meta)
    print(f"[confluence] joint: {len(keep)} stars in >=2 parents of {len(pairs)} "
          f"independent pairs")
    return meta


# ---------------------------------------------------------------------------
# acquire (runner)
# ---------------------------------------------------------------------------


def stage_acquire(out: Path = OUT, baseline_n: int = 10_000, budget_s: float = 9000) -> dict:
    from .acquire import fetch_covariates, fetch_density, fetch_simbad, fetch_vsx
    J = pd.read_parquet(out / "joint_long.parquet")
    led: list = []
    rep = {**_prov()}
    sids = J["source_id"].dropna().astype("int64").unique()
    cov, rep["covariates"] = fetch_covariates(sids, led, budget_s=budget_s * 0.4)
    if len(cov):
        cov = cov.drop_duplicates("source_id")
        cov["star"] = "gaia:" + cov["source_id"].astype("int64").astype(str)
        den, rep["density"] = fetch_density(cov, led, budget_s=budget_s * 0.2)
        if len(den):
            cov = cov.merge(den, on="source_id", how="left")
        cov.to_parquet(out / "covariates.parquet", index=False)
    rep["n_covariates"] = int(len(cov))
    # positions for tagging: covariate positions first, the channel's own otherwise
    pos = J.groupby("star")[["ra", "dec"]].first()
    if len(cov):
        pos.update(cov.set_index("star")[["ra", "dec"]])
    tails = J[(J["tail_q0.05"]) | (J["flag"])]["star"].unique()
    rng = np.random.default_rng(5)
    allj = J["star"].unique()
    base = rng.choice(allj, size=min(baseline_n, len(allj)), replace=False)
    want = pd.Index(sorted(set(tails) | set(base)))
    tgt = pos.reindex(want).dropna().reset_index().rename(columns={"index": "star"})
    tgt["key"] = tgt["star"]
    sim, rep["simbad"] = fetch_simbad(tgt, led, budget_s=budget_s * 0.2)
    vsx, rep["vsx"] = fetch_vsx(tgt, led, budget_s=budget_s * 0.2)
    tags = pd.DataFrame({"star": want})
    tags["in_baseline_sample"] = tags["star"].isin(set(base))
    tags["is_tail_member"] = tags["star"].isin(set(tails))
    tags["tag_attempted"] = tags["star"].isin(set(tgt["star"]))
    if len(sim):
        tags = tags.merge(sim.rename(columns={"key": "star"})[["star", "main_id", "otype"]],
                          on="star", how="left")
    if len(vsx):
        tags = tags.merge(vsx.rename(columns={"key": "star"})[
            ["star", "vsx_name", "vsx_type", "vsx_period"]], on="star", how="left")
    if len(cov):
        tags = tags.merge(cov[["star", "vari_class", "non_single_star", "phot_variable_flag"]],
                          on="star", how="left")
    tags["simbad_reached"] = rep["simbad"]["status"] != "NO_DATA_REACHED"
    tags["vsx_reached"] = rep["vsx"]["status"] != "NO_DATA_REACHED"
    tags.to_parquet(out / "tags.parquet", index=False)
    rep["n_tag_targets"] = int(len(tgt))
    rep["ledger_tail"] = led[-60:]
    rep["n_ledger_failures"] = int(sum(1 for x in led if not x.get("ok", False)))
    _write(out / "acquire.json", rep)
    print(f"[confluence] acquire: {len(cov)} covariates, simbad {rep['simbad']['status']}, "
          f"vsx {rep['vsx']['status']}")
    return rep


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------


def _offline_covariates(J: pd.DataFrame) -> pd.DataFrame:
    """Covariates from committed files when the runner's table is absent."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    pos = J.groupby("star")[["ra", "dec", "source_id"]].first().reset_index()
    ok = pos["ra"].notna()
    pos["abs_b"] = np.nan
    pos["wise_depth"] = np.nan
    if ok.any():
        c = SkyCoord(pos.loc[ok, "ra"].to_numpy() * u.deg, pos.loc[ok, "dec"].to_numpy() * u.deg)
        pos.loc[ok, "abs_b"] = np.abs(c.galactic.b.deg)
        # WISE (and Gaia) coverage depth rises steeply toward the ecliptic poles
        pos.loc[ok, "wise_depth"] = np.abs(c.barycentrictrueecliptic.lat.deg)
    extra = []
    sh = sorted((R.RESULTS / "cenotaph").glob("shell_*.parquet"))
    if sh:
        extra.append(pd.concat([pd.read_parquet(f, columns=[
            "source_id", "phot_g_mean_mag", "bp_rp", "ruwe", "ipd_frac_multi_peak"])
            for f in sh]))
    bb = R.RESULTS / "baffle_bright" / "bright_residuals.csv"
    if bb.exists():
        extra.append(pd.read_csv(bb, usecols=["source_id", "phot_g_mean_mag", "bp_rp", "ruwe"]))
    # Gaia columns the harvested channels' own tables carry (ignition, cradle, ring)
    for p in sorted(R.SCORES_DIR.glob("_cov_*.parquet")):
        c = pd.read_parquet(p)
        extra.append(c[[x for x in c.columns if x in (
            "source_id", "phot_g_mean_mag", "bp_rp", "ruwe", "ipd_frac_multi_peak",
            "non_single_star", "phot_variable_flag")]])
    if extra:
        e = pd.concat(extra, ignore_index=True)
        e["source_id"] = pd.to_numeric(e["source_id"], errors="coerce")
        e = e.dropna(subset=["source_id"])
        e["source_id"] = e["source_id"].astype("int64")
        # first non-null value per column across sources
        e = e.groupby("source_id", sort=False).first().reset_index()
        pos["source_id"] = pd.to_numeric(pos["source_id"], errors="coerce").astype("Int64")
        e["source_id"] = e["source_id"].astype("Int64")
        pos = pos.merge(e, on="source_id", how="left")
    # (no scan-coverage column offline: Gaia's scan law also runs with ecliptic
    # latitude, so wise_depth stands in for both rather than being counted twice)
    return pos


def stage_assess(out: Path = OUT, min_joint: int = MIN_JOINT, n_inject: int = 40) -> dict:
    t0 = time.time()
    specs = R.by_name()
    jp = out / "joint_long.parquet"
    if not jp.exists():
        stage_joint(out)
    J = pd.read_parquet(jp)
    summary = {"channel": "confluence", **_prov()}
    if J.empty:
        summary.update(verdict="NO_TESTABLE_PAIR", n_joint_stars=0)
        _write(out / "summary.json", summary)
        return summary
    cp = out / "covariates.parquet"
    if cp.exists():
        cov = pd.read_parquet(cp)
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        cov["abs_b"] = np.abs(pd.to_numeric(cov["b"], errors="coerce"))
        cov["scan_coverage"] = pd.to_numeric(cov.get("visibility_periods_used"), errors="coerce")
        c = SkyCoord(cov["ra"].to_numpy() * u.deg, cov["dec"].to_numpy() * u.deg)
        cov["wise_depth"] = np.abs(c.barycentrictrueecliptic.lat.deg)
        if "n_nb" in cov:
            cov["log_density"] = np.log10(pd.to_numeric(cov["n_nb"], errors="coerce").clip(lower=1))
        # stars the runner could not cover fall back to committed covariates
        off = _offline_covariates(J)
        miss = off[~off["star"].isin(set(cov["star"]))]
        cov = pd.concat([cov, miss], ignore_index=True)
        summary["covariate_source"] = "runner (Gaia DR3) + committed fallback"
    else:
        cov = _offline_covariates(J)
        summary["covariate_source"] = "committed files only (runner covariates absent)"
    tp = out / "tags.parquet"
    tags = tag_table(pd.read_parquet(tp)) if tp.exists() else None
    summary["tags_present"] = tags is not None

    res = run_tests(J, cov, specs, MODES, min_joint=min_joint, rule="strict")
    res["rule"] = "strict"
    res_rel = run_tests(J, cov, specs, MODES, min_joint=min_joint, rule="relaxed")
    res_rel = res_rel[~res_rel["channels"].isin(set(res["channels"]))]
    res_rel["rule"] = "relaxed_only"
    tests = pd.concat([res, res_rel], ignore_index=True)
    for c in ("members", "p_value", "q_bh", "observed", "expected", "ratio", "k", "n_joint"):
        if c not in tests:
            tests[c] = np.nan
    tested = tests[tests["status"] == "TESTED"]
    summary["n_groups"] = int(tests["channels"].nunique())
    summary["n_tests"] = int(len(tested))
    summary["n_joint_stars"] = int(J["star"].nunique())

    # --- positive control 1: known classes lead the overlaps ---------------
    ctrl = {"status": "UNTESTED", "reason": "no tags"}
    known = set()
    if tags is not None:
        known = set(tags.loc[tags["known_class"], "star"])
        attempted = set(tags.loc[tags["tag_attempted"], "star"]) \
            if "tag_attempted" in tags else set(tags["star"])
        base = tags[tags.get("in_baseline_sample", True) & tags["star"].isin(attempted)]
        base_rate = float(base["known_class"].mean()) if len(base) else float("nan")
        mem = set()
        for m in tested["members"].dropna():
            mem |= {x for x in str(m).split(";") if x}
        mem_att = mem & attempted
        mem_rate = (len(mem_att & known) / len(mem_att)) if mem_att else float("nan")
        ctrl = {"baseline_known_rate": base_rate, "n_baseline": int(len(base)),
                "overlap_members": len(mem), "overlap_members_tagged": len(mem_att),
                "overlap_known_rate": mem_rate,
                "enrichment": (mem_rate / base_rate) if base_rate and base_rate > 0 else None}
        # "leading": the most extreme overlap members (sum over the member's
        # channels of -log10 percentile) -- are they the known classes?
        lead = J[J["star"].isin(mem_att)].assign(
            lp=lambda d: -np.log10(d["pct"].clip(lower=1e-9)))
        lead = lead.groupby("star")["lp"].sum().sort_values(ascending=False).head(20)
        ctrl["leading20_known_rate"] = float(np.mean([s in known for s in lead.index])) \
            if len(lead) else None
        ctrl["leading20"] = [{"star": s, "sum_neglog_pct": round(float(v), 2),
                              "known": s in known} for s, v in lead.items()]
        ctrl["status"] = ("PASS" if (mem_att and base_rate == base_rate
                                     and mem_rate > base_rate) else
                          ("UNTESTED" if not mem_att else "FAIL"))
    summary["control_known_classes"] = ctrl

    # --- positive control 2: injection-recovery on the largest pairs ---------
    inj = []
    pairs_sorted = res[(res["k"] == 2) & (res["status"] == "TESTED") & (res["mode"] == "0.05")] \
        .sort_values("n_joint", ascending=False)
    for _, r in pairs_sorted.head(3).iterrows():
        a, b = r["channels"].split("+")
        inj.append(injection_trials(J, cov, (a, b), n_trials=n_inject))
    summary["injection"] = [{"pair": x["pair"], "n_joint": x["n_joint"], "by_k": x["by_k"]}
                            for x in inj]
    inj_ok = any(any(k["recovery_rate"] >= 0.8 for k in x["by_k"] if k["k"] <= 20) for x in inj)
    summary["control_injection"] = "PASS" if inj_ok else ("UNTESTED" if not inj else "FAIL")
    _write(out / "injection.json", inj)

    # --- residual after removing known classes -------------------------------
    resid = run_tests(J, cov, specs, MODES, min_joint=min_joint, rule="strict",
                      exclude_stars=known) if known else res.copy()
    resid["rule"] = "strict_minus_known"
    for c in ("members", "p_value", "status"):
        if c not in resid:
            resid[c] = np.nan
    tests = pd.concat([tests, resid], ignore_index=True)

    # --- members + trace ----------------------------------------------------
    sig = tested[(tested["q_bh"] < 0.05) | (tested["p_value"] < 0.01)] \
        if "q_bh" in tested else tested.iloc[0:0]
    all_members = []
    for m in tested["members"].dropna():
        all_members += [x for x in str(m).split(";") if x]
    tr = trace_members(J, cov, tags, all_members) if all_members else pd.DataFrame()
    if len(tr):
        grp_of = {}
        for _, r in tested.iterrows():
            for x in str(r.get("members") or "").split(";"):
                if x:
                    grp_of.setdefault(x, set()).add(f"{r['channels']}@{r['mode']}")
        tr["in_groups"] = tr["star"].map(lambda s: ";".join(sorted(grp_of.get(s, []))))
        tr.to_csv(out / "members.csv", index=False)
        resid_tr = tr[~tr["trace_verdict"].str.startswith("KNOWN_CLASS")]
        resid_tr.to_csv(out / "residual.csv", index=False)
    else:
        resid_tr = pd.DataFrame()
    tests.to_csv(out / "tests.csv", index=False)

    rs = resid[resid["status"] == "TESTED"]
    summary["n_significant_groups"] = int(len(sig))
    summary["significant_groups"] = sig[["channels", "mode", "n_joint", "observed", "expected",
                                         "ratio", "p_value", "q_bh"]].to_dict("records") \
        if len(sig) else []
    summary["n_residual_significant"] = int(((rs["p_value"] < 0.01)).sum()) if len(rs) else 0
    summary["n_members"] = int(len(tr))
    summary["n_residual_members"] = int(len(resid_tr))
    summary["residual_trace_verdicts"] = resid_tr["trace_verdict"].str.split(":").str[0] \
        .value_counts().to_dict() if len(resid_tr) else {}
    summary["funnel"] = {
        "channels_with_parent": int(J["channel"].nunique()),
        "joint_stars": int(J["star"].nunique()),
        "groups_tested": int(tested["channels"].nunique()),
        "tests": int(len(tested)),
        "significant": int(len(sig)),
        "overlap_members": int(len(tr)),
        "after_known_class_removal": int(len(resid_tr)),
        "unexplained": int((resid_tr["trace_verdict"] == "UNEXPLAINED").sum())
        if len(resid_tr) else 0,
    }
    controls_ok = summary["control_injection"] == "PASS" and ctrl.get("status") in ("PASS",)
    if summary["n_tests"] == 0:
        v = "NO_TESTABLE_PAIR"
    elif not controls_ok:
        v = "CONTROLS_FAILED" if (summary["control_injection"] == "FAIL"
                                  or ctrl.get("status") == "FAIL") else "CONTROLS_UNTESTED"
    elif len(sig) == 0:
        v = "NO_EXCESS_CONFLUENCE"
    elif summary["n_residual_significant"] == 0:
        v = "CONFLUENCE_KNOWN_CLASSES_ONLY"
    else:
        v = "RESIDUAL_CONFLUENCE"
    summary["verdict"] = v
    # self-consistency: every member listed is in the joint table
    summary["self_consistency"] = {
        "members_in_joint": bool(set(tr["star"]) <= set(J["star"])) if len(tr) else True,
        "n_tests_rows": int(len(tests)),
    }
    summary["elapsed_s"] = round(time.time() - t0, 1)
    _write(out / "summary.json", summary)
    print(f"[confluence] assess: verdict {v}; {summary['n_tests']} tests; "
          f"{len(sig)} significant; members {len(tr)}; residual {len(resid_tr)}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="confluence")
    ap.add_argument("--stage", default="assess",
                    choices=["inventory", "harvest", "accel", "resolve", "joint", "acquire", "assess",
                             "offline"])
    ap.add_argument("--harvest-dir", default=os.environ.get("CONFLUENCE_HARVEST", "harvest"))
    ap.add_argument("--scores-dir", default=str(R.SCORES_DIR))
    ap.add_argument("--min-joint", type=int, default=MIN_JOINT)
    ap.add_argument("--no-accel", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "inventory":
        stage_inventory()
    elif a.stage == "harvest":
        stage_harvest(Path(a.harvest_dir), Path(a.scores_dir), with_accel=not a.no_accel)
    elif a.stage == "accel":
        stage_accel(Path(a.scores_dir))
    elif a.stage == "resolve":
        stage_resolve()
    elif a.stage == "joint":
        stage_joint()
    elif a.stage == "acquire":
        stage_acquire()
    elif a.stage == "assess":
        stage_assess(min_joint=a.min_joint)
    elif a.stage == "offline":
        stage_inventory()
        stage_joint()
        stage_assess(min_joint=a.min_joint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

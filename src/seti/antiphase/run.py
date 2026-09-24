"""ANTIPHASE stage orchestration.  Writes ``results/antiphase/``.

Stages
------
``controls``  every configured positive control: Sesame check, NEOWISE cone,
              optical (ZTF / Gaia alerts / ASAS-SN), the full ladder.  Writes
              ``controls.json`` with a **gate**: ``FAIL`` if a known natural
              object comes out ``ANTIPHASE_CANDIDATE`` (the ladder leaks) or a
              coupled-event control whose data *do* overlap is not recovered
              as coupled (the detector is blind); ``PASS`` if at least one
              coupled-event control is recovered and nothing failed;
              ``UNTESTED`` if no control's data overlap (the gate then rests
              on the injections alone, and says so).  Runs FIRST.
``shard``     shard ``i/n`` of IGNITION's parent (its shard artifact:
              ``parent_s{i}of{n}.parquet`` + ``epochs_s{i}of{n}.csv``):
              stratified NEOWISE drift correction (magnitude bin x |beta|
              band), ZTF g/r for every star in the footprint (IR-risers first),
              the ladder per star as its light curve lands (checkpointed), then
              the pair/shift null and the injections on this shard's stars.
``reduce``    merge shards; FAP of each coupled star against the pooled null;
              online vet of the survivors (Gaia neighbours from VizieR -> the
              blend model); ``candidates.csv``, ``coupled.csv``, ``summary.json``.

Verdict vocabulary (``summary.json["verdict"]``): ``NO_DATA_REACHED`` (no ZTF
light curve came back), ``NO_ANTIPHASE_CANDIDATE`` (a count over what was
searched, never a limit), ``ANTIPHASE_CANDIDATES``; a ``DEGRADED (...)``
prefix names missing shards, failed ZTF queries and unreachable vet rungs; a
``CONTROLS_FAILED`` prefix means nothing downstream is to be believed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import acquire as acq
from .classify import DEFAULT_CLASSIFY, blend_explains, blend_prediction, final_verdict
from .coupling import DEFAULT_COUPLING, ir_rise_prescore
from .null import injection_efficiency, run_null, score_fap
from .pipeline import evaluate_for_injection, evaluate_pack, ir_arrays, make_pack

OUT = Path("results") / "antiphase"
STAGES = ("controls", "shard", "reduce")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return ""


def load_config(path: str | None = None) -> dict:
    import yaml

    p = Path(path) if path else Path("config") / "antiphase.yaml"
    got = yaml.safe_load(p.read_text()) if p.exists() else {}
    got = got or {}
    got.setdefault("coupling", {})
    got.setdefault("classify", {})
    got.setdefault("energy", {})
    got.setdefault("survey", {})
    got.setdefault("controls", [])
    return got


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(obj), indent=1, default=str))


def _tag(i: int, n: int) -> str:
    return f"s{i}of{n}"


# ===========================================================================
# controls
# ===========================================================================
def control_outcome(ctrl: dict, rec: dict | None, conf: dict) -> str:
    """One control's result against its kind (pure; tested offline)."""
    if rec is None:
        return "UNTESTABLE_NO_DATA"
    v = str(rec.get("verdict"))
    if v == "ANTIPHASE_CANDIDATE":
        return "FAIL_NATURAL_PASSED_AS_CANDIDATE"
    lab = str(rec.get("coupling_label"))
    coupled = lab == "COUPLED"
    kind = str(ctrl.get("kind", ""))
    if coupled:
        if v == "NATURAL":
            return "RECOVERED_NATURAL"
        return "RECOVERED_COUPLED_INCOMPLETE"       # coupled; colour or budget unmeasurable
    minm = int((conf.get("coupling") or {}).get("min_matched", DEFAULT_COUPLING["min_matched"]))
    if int(rec.get("n_matched") or 0) < minm or lab in ("NO_OPTICAL", "NO_IR",
                                                         "INSUFFICIENT_MATCHED"):
        return "UNTESTABLE_NO_OVERLAP"
    if int(rec.get("n_faded") or 0) == 0:
        # no optical fade inside the NEOWISE window: the event is not in the data
        return "UNTESTABLE_NO_FADE_IN_WINDOW"
    if kind == "coupled_event" and lab not in ("COUPLED",):
        return f"MISSED_{lab}"
    return f"NOT_COUPLED_{lab}"


def controls_gate(outcomes: dict, kinds: dict) -> str:
    vals = list(outcomes.values())
    if any(v.startswith("FAIL") for v in vals):
        return "FAIL"
    if any(v.startswith("MISSED") for n, v in outcomes.items() if kinds.get(n) == "coupled_event"):
        return "FAIL"
    if any(v.startswith("RECOVERED") for n, v in outcomes.items()
           if kinds.get(n) == "coupled_event"):
        return "PASS"
    return "UNTESTED"


def _control_meta(ra: float, dec: float, fetchers: dict) -> dict:
    meta: dict = {"ra": ra, "dec": dec}
    nb = fetchers["gaia_cone"](ra, dec, 5.0)
    if nb.get("status") == "OK" and nb.get("rows"):
        rows = sorted(nb["rows"], key=lambda r: (r.get("sep_arcsec") if r.get("sep_arcsec")
                                                 is not None else 99))
        g = rows[0]
        for k in ("source_id", "pmra", "pmdec", "parallax", "phot_g_mean_mag", "bp_rp"):
            meta[k] = g.get(k)
        meta["gaia_sep_arcsec"] = g.get("sep_arcsec")
    meta["gaia_status"] = nb.get("status")
    aw = fetchers["allwise"](ra, dec, 3.0)
    if aw.get("status") == "OK":
        for k in ("w1mpro", "w2mpro", "w3mpro", "cc_flags", "ph_qual"):
            meta[k] = aw.get(k)
    meta["allwise_status"] = aw.get("status")
    return meta


def _default_control_fetchers() -> dict:
    return {"sesame": acq.sesame_resolve, "neowise": acq.neowise_epochs_cone,
            "ztf": acq.fetch_ztf, "gaia_alert": acq.fetch_gaia_alert, "asassn": acq.fetch_asassn,
            "gaia_cone": acq.gaia_neighbours_vizier, "allwise": acq.allwise_vizier}


def run_controls(conf: dict, fetchers: dict | None = None) -> dict:
    f = {**_default_control_fetchers(), **(fetchers or {})}
    out = {"generated_utc": _now(), "git_sha": _git_sha(), "controls": []}
    outcomes, kinds = {}, {}
    for ctrl in conf.get("controls", []):
        name = str(ctrl["name"])
        kinds[name] = str(ctrl.get("kind", ""))
        c = {"name": name, "kind": ctrl.get("kind"), "ref": ctrl.get("ref"),
             "note": ctrl.get("note")}
        # --- position -------------------------------------------------------
        ses = f["sesame"](ctrl.get("alias") or name)
        if ses.get("status") != "OK" and ctrl.get("alias"):
            ses = f["sesame"](name)
        c["sesame"] = ses
        ra, dec = ctrl.get("ra"), ctrl.get("dec")
        if ses.get("status") == "OK":
            if ra is None:
                ra, dec = ses["ra"], ses["dec"]
                c["position_source"] = "sesame"
            else:
                sep = 3600.0 * np.hypot((float(ses["ra"]) - float(ra))
                                        * np.cos(np.radians(float(dec))),
                                        float(ses["dec"]) - float(dec))
                c["sesame_sep_arcsec"] = round(float(sep), 2)
                if sep > 5.0:
                    ra, dec = ses["ra"], ses["dec"]
                    c["position_source"] = "sesame (config disagreed)"
                else:
                    c["position_source"] = "config (Sesame agrees)"
        elif ra is not None:
            c["position_source"] = "config (Sesame unavailable)"
        if ra is None:
            c["outcome"] = "UNTESTABLE_UNRESOLVED"
            outcomes[name] = c["outcome"]
            out["controls"].append(c)
            continue
        ra, dec = float(ra), float(dec)
        c["ra"], c["dec"] = ra, dec
        meta = _control_meta(ra, dec, f)
        c["meta"] = meta
        pmra = float(meta.get("pmra") or 0.0) if np.isfinite(float(meta.get("pmra") or 0.0)) else 0.0
        pmde = float(meta.get("pmdec") or 0.0) if np.isfinite(float(meta.get("pmdec") or 0.0)) else 0.0
        # --- IR --------------------------------------------------------------
        try:
            ep, nrec = f["neowise"](ra, dec, pmra, pmde)
        except Exception as exc:                        # noqa: BLE001
            ep, nrec = pd.DataFrame(), {"status": "FAILED", "error": repr(exc)[:300]}
        c["neowise"] = nrec
        c["neowise_epochs"] = (ep[["band", "t_yr", "mag", "err"]].round(5).to_dict("records")
                               if len(ep) else [])
        # --- optical ---------------------------------------------------------
        bands: dict = {}
        srcs = {}
        for s in ctrl.get("optical", ["ztf"]):
            try:
                if s == "ztf":
                    rec = f["ztf"](ra, dec, pmra, pmde)
                elif s == "gaia_alert" and ctrl.get("gaia_alert"):
                    rec = f["gaia_alert"](ctrl["gaia_alert"])
                elif s == "asassn":
                    rec = f["asassn"](ra, dec)
                else:
                    continue
            except Exception as exc:                    # noqa: BLE001
                rec = {"status": "FAILED", "error": repr(exc)[:300]}
            srcs[s] = {k: v for k, v in rec.items() if k != "bands"}
            for b, v in (rec.get("bands") or {}).items():
                srcs[s][f"n_{b}"] = int(v.get("n", len(v.get("mjd", []))))
                if b not in bands:
                    bands[b] = v
        c["optical_sources"] = srcs
        c["optical_bands"] = sorted(bands)
        if not len(ep) or not bands:
            c["outcome"] = "UNTESTABLE_NO_DATA"
            outcomes[name] = c["outcome"]
            out["controls"].append(c)
            continue
        # ZTF g/r are the survey's bands; a control with fewer runs on what it has
        use = {b: bands[b] for b in ("g", "r") if b in bands}
        if len(use) < 2:
            use = bands
        cc = dict(conf.get("coupling") or {})
        cc["min_optical_bands"] = min(2, len(use))
        cc["min_points_per_bin"] = min(int(cc.get("min_points_per_bin", 5)), 3) \
            if len(use) < 2 else int(cc.get("min_points_per_bin", 5))
        cconf = {**conf, "coupling": cc}
        pack = make_pack(ep, use, meta, cc)
        rec = evaluate_pack(pack, cconf)
        c["result"] = rec
        c["series"] = {"t_yr": pack["t"], "opt": {b: v[0] for b, v in pack["opt"].items()},
                       "ir": {b: v[0] for b, v in pack["ir"].items()}}
        c["outcome"] = control_outcome(ctrl, rec, conf)
        outcomes[name] = c["outcome"]
        out["controls"].append(c)
        print(f"[antiphase] control {name}: {c['outcome']} verdict={rec.get('verdict')} "
              f"label={rec.get('coupling_label')} matched={rec.get('n_matched')} "
              f"faded={rec.get('n_faded')} chroma={rec.get('chroma')} "
              f"energy={rec.get('energy_verdict')} lag={rec.get('best_lag')}", flush=True)
    out["outcomes"] = outcomes
    out["gate"] = controls_gate(outcomes, kinds)
    return out


# ===========================================================================
# shard
# ===========================================================================
def load_ignition_shard(idir: Path, i: int, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag = _tag(i, n)
    pp = idir / f"parent_{tag}.parquet"
    ep = idir / f"epochs_{tag}.csv"
    parent = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
    if len(parent):
        parent = parent.copy()
        parent["source_id"] = parent["source_id"].astype(str)
        parent = parent.drop_duplicates("source_id")
    if ep.exists():
        eps = pd.read_csv(ep, dtype={"source_id": str},
                          usecols=lambda c: c in ("source_id", "band", "t_yr", "mag", "err"))
        for col in ("t_yr", "mag", "err"):
            eps[col] = pd.to_numeric(eps[col], errors="coerce")
        eps = eps.dropna(subset=["t_yr", "mag"]).drop_duplicates(["source_id", "band", "t_yr"])
    else:
        eps = pd.DataFrame(columns=["source_id", "band", "t_yr", "mag", "err"])
    return parent, eps


def correct_drift(eps: pd.DataFrame, parent: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """IGNITION's stratified ensemble (magnitude bin x |ecliptic latitude| band)."""
    from ..ignition.acquire import ecliptic_latitude_deg
    from ..ignition.ensemble import stratified_offsets

    if not len(eps):
        return eps, {"mode": "stratified", "status": "NO_EPOCHS"}
    beta = pd.Series(np.abs(ecliptic_latitude_deg(parent["ra"].to_numpy(float),
                                                  parent["dec"].to_numpy(float))),
                     index=parent["source_id"].astype(str).to_numpy())
    corr, rep = stratified_offsets(eps, beta)
    return corr, rep


def select_targets(parent: pd.DataFrame, eps: pd.DataFrame, conf: dict) -> tuple[pd.DataFrame, dict]:
    sv = conf.get("survey") or {}
    fun = {"n_parent": int(len(parent))}
    p = parent[pd.to_numeric(parent["dec"], errors="coerce") >= acq.ZTF_DEC_MIN].copy()
    fun["n_ztf_footprint"] = int(len(p))
    post = eps[eps["t_yr"] >= 2018.2]
    cnt = post.groupby(["source_id", "band"]).size().unstack("band")
    need = int(sv.get("min_ir_epochs_ztf_era", 6))
    ok_ids = set(cnt.index[(cnt.get("W1", 0) >= need) & (cnt.get("W2", 0) >= need)].astype(str)) \
        if len(cnt) else set()
    p = p[p["source_id"].isin(ok_ids)].copy()
    fun["n_enough_ir_epochs"] = int(len(p))
    # IR-risers first (the order matters only if the budget runs out)
    pre = {}
    for sid, g in eps[eps["source_id"].isin(set(p["source_id"]))].groupby("source_id"):
        t, ir = ir_arrays(g)
        pre[sid] = ir_rise_prescore(t, ir) if len(t) else float("nan")
    p["ir_prescore"] = p["source_id"].map(pre).astype(float)
    p = p.sort_values("ir_prescore", ascending=False, na_position="last")
    return p, fun


def _meta_row(r: dict) -> dict:
    keep = ("source_id", "ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp",
            "teff_gspphot", "w1mpro", "w2mpro", "w3mpro", "ruwe", "phot_variable_flag",
            "non_single_star", "allwise_n_neighbours")
    m = {k: r.get(k) for k in keep if k in r}
    try:
        mu = float(np.hypot(float(r.get("pmra")), float(r.get("pmdec"))))
        m["v_tan_kms"] = 4.74047 * mu / float(r.get("parallax"))
    except (TypeError, ValueError, ZeroDivisionError):
        m["v_tan_kms"] = float("nan")
    return m


def run_shard(conf: dict, idir: Path, out: Path, i: int, n: int, *, ztf_fetch=None,
              budget_s: float | None = None, max_stars: int = 0) -> dict:
    sv = conf.get("survey") or {}
    tag = _tag(i, n)
    out.mkdir(parents=True, exist_ok=True)
    parent, eps = load_ignition_shard(idir, i, n)
    rep: dict = {"shard": tag, "generated_utc": _now(), "git_sha": _git_sha(),
                 "ignition_dir": str(idir)}
    if not len(parent) or not len(eps):
        rep.update({"status": "NO_IGNITION_INPUT", "n_parent": int(len(parent)),
                    "n_epoch_rows": int(len(eps))})
        _write_json(out / f"shard_{tag}.json", rep)
        return rep
    eps_c, ens = correct_drift(eps, parent)
    rep["ensemble"] = ens
    targets, fun = select_targets(parent, eps_c, conf)
    if max_stars:
        targets = targets.head(int(max_stars))
    rep["funnel"] = fun
    stars_p = out / f"stars_{tag}.csv"
    bins_p = out / f"optbins_{tag}.csv"
    stat_p = out / f"ztfstatus_{tag}.csv"
    done = set()
    if stat_p.exists():
        try:
            done = set(pd.read_csv(stat_p, dtype={"source_id": str})["source_id"].astype(str))
        except Exception:                               # noqa: BLE001
            done = set()
    todo = targets[~targets["source_id"].isin(done)]
    rep["n_resumed_done"] = len(done)
    by_sid = {sid: g for sid, g in eps_c[eps_c["source_id"].isin(set(targets["source_id"]))]
              .groupby("source_id")}
    meta_by = {str(r["source_id"]): _meta_row(r) for r in targets.to_dict("records")}
    bright = float(sv.get("ztf_bright_limit", 12.5))
    buf_s, buf_b, buf_z = [], [], []

    def flush():
        for rows, p in ((buf_s, stars_p), (buf_b, bins_p), (buf_z, stat_p)):
            if rows:
                pd.DataFrame(rows).to_csv(p, mode="a", index=False, header=not p.exists())
                rows.clear()

    def on_result(sid, zrec):
        st = zrec.get("status")
        zb = zrec.get("bands") or {}
        zs = {"source_id": sid, "status": st, "n_g": int(zb.get("g", {}).get("n", 0)),
              "n_r": int(zb.get("r", {}).get("n", 0)),
              "med_g": zb.get("g", {}).get("median_mag"), "med_r": zb.get("r", {}).get("median_mag"),
              "error": str(zrec.get("error") or "")[:160]}
        if st == "OK":
            sat = any(v.get("saturation_risk") for v in zb.values()) or \
                any((v.get("median_mag") or 99) < bright for v in zb.values())
            if sat:
                zs["status"] = "ZTF_SATURATED"
            else:
                meta = meta_by.get(sid, {})
                pack = make_pack(by_sid.get(sid), {b: zb[b] for b in ("g", "r") if b in zb},
                                 meta, conf.get("coupling"))
                rec = evaluate_pack(pack, conf)
                rec = {"source_id": sid, **{k: meta.get(k) for k in
                                            ("ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax",
                                             "v_tan_kms", "w1mpro", "w2mpro")}, **rec}
                buf_s.append(rec)
                for b, (m, e) in pack["opt"].items():
                    for t, mm, ee in zip(pack["t"], m, e, strict=False):
                        if np.isfinite(mm):
                            buf_b.append({"source_id": sid, "band": b, "t_yr": round(float(t), 5),
                                          "mag": round(float(mm), 5), "err": round(float(ee), 5)})
        buf_z.append(zs)
        if len(buf_z) >= 200:
            flush()

    t0 = _time.monotonic()
    budget = float(budget_s if budget_s is not None else sv.get("budget_s", 16200))
    fetch_kw = {"radius_arcsec": float(sv.get("ztf_radius_arcsec", 1.5)), "bright_limit": bright}
    zlog = acq.fetch_ztf_many(todo, workers=int(sv.get("ztf_workers", 4)), budget_s=budget,
                              fetch=ztf_fetch, on_result=on_result, **fetch_kw)
    flush()
    rep["ztf"] = zlog
    rep["ztf_elapsed_s"] = round(_time.monotonic() - t0, 1)
    # --- per-shard null and injections over every evaluated star ------------
    packs = build_packs(out, tag, eps_c, meta_by)
    rep["n_packs"] = len(packs)
    nl = run_null(packs, conf.get("coupling"), n_rounds=int(sv.get("null_rounds", 5)),
                  shifts=tuple(sv.get("null_shifts", (-4, -2, 2, 4))),
                  max_stars=int(sv.get("null_max_stars", 3000)), seed=20260924 + i)
    _write_json(out / f"null_{tag}.json", nl)
    inj = injection_efficiency(packs, lambda p: evaluate_for_injection(p, conf),
                               depths=tuple(sv.get("injection_depths", (0.03, 0.05, 0.1, 0.2))),
                               temps=tuple(sv.get("injection_temps_k", (300.0, 600.0, 1000.0))),
                               dur=int(sv.get("injection_dur_epochs", 2)),
                               max_stars=int(sv.get("injection_max_stars", 150)), seed=7 + i)
    rep["injection"] = inj
    rep["null_summary"] = {k: nl.get(k) for k in ("status", "n_stars", "n_rounds_pair",
                                                  "pair_coupled_per_round", "pair_coupled_mean",
                                                  "shift_coupled", "shift_coupled_mean")}
    rep["status"] = "OK"
    _write_json(out / f"shard_{tag}.json", rep)
    # IR epochs of the stars that faded, for the reduce stage's vet
    st = pd.read_csv(stars_p, dtype={"source_id": str}) if stars_p.exists() else pd.DataFrame()
    if len(st):
        fad = set(st.loc[pd.to_numeric(st["n_faded"], errors="coerce") > 0, "source_id"])
        sub = eps_c[eps_c["source_id"].isin(fad)][["source_id", "band", "t_yr", "mag", "err"]]
        sub.round(5).to_csv(out / f"irfaded_{tag}.csv", index=False)
    return rep


def build_packs(out: Path, tag: str, eps_c: pd.DataFrame, meta_by: dict) -> dict:
    bins_p = out / f"optbins_{tag}.csv"
    if not bins_p.exists():
        return {}
    ob = pd.read_csv(bins_p, dtype={"source_id": str}).drop_duplicates(["source_id", "band",
                                                                         "t_yr"])
    packs = {}
    ir_by = {sid: g for sid, g in eps_c[eps_c["source_id"].isin(set(ob["source_id"]))]
             .groupby("source_id")}
    for sid, g in ob.groupby("source_id"):
        if sid not in ir_by:
            continue
        t, ir = ir_arrays(ir_by[sid])
        from .null import align_to

        opt = {}
        for b, gb in g.groupby("band"):
            opt.update(align_to(t, gb["t_yr"].to_numpy(float),
                                {b: (gb["mag"].to_numpy(float), gb["err"].to_numpy(float))},
                                tol_yr=0.01))
        packs[sid] = {"t": t, "opt": opt, "ir": ir, "meta": meta_by.get(sid, {})}
    return packs


# ===========================================================================
# reduce
# ===========================================================================
def _default_vet_fetchers() -> dict:
    return {"gaia_cone": acq.gaia_neighbours_vizier}


def vet_candidate(row: dict, ir_ep: pd.DataFrame, conf: dict, fetchers: dict) -> dict:
    """Blend model with Gaia neighbours (VizieR) -> the verdict re-run with it."""
    kc = {**DEFAULT_CLASSIFY, **(conf.get("classify") or {})}
    out: dict = {}
    nb = fetchers["gaia_cone"](float(row["ra"]), float(row["dec"]), 20.0)
    out["neighbours_status"] = nb.get("status")
    if nb.get("status") != "OK":
        out["blend"] = {"status": "UNTESTED", "error": nb.get("error")}
        return out
    rows = nb.get("rows") or []
    tgt = [r for r in rows if str(r.get("source_id")) == str(row["source_id"])]
    target = dict(tgt[0]) if tgt else {"ra": row["ra"], "dec": row["dec"], "pmra": 0.0,
                                       "pmdec": 0.0}
    target["w1mpro"], target["w2mpro"] = row.get("w1mpro"), row.get("w2mpro")
    others = [r for r in rows if str(r.get("source_id")) != str(row["source_id"])]
    t, _ir = ir_arrays(ir_ep)
    pred = blend_prediction(target, others, t, kc)
    faded = [float(x) for x in str(row.get("faded_t_yr") or "").split(";") if x]
    obs = {"W1": row.get("w1_dmag"), "W2": row.get("w2_dmag")}
    be = blend_explains(pred, faded, t, obs, kc)
    out["blend"] = {**be, "n_neighbours": len(others),
                    "max_predicted_brightening": pred.get("max_brightening"),
                    "nearest": sorted(others, key=lambda r: r.get("sep_arcsec") or 99)[:5]}
    return out


def run_reduce(conf: dict, out: Path, n: int, *, fetchers: dict | None = None,
               run_id: str = "") -> dict:
    f = {**_default_vet_fetchers(), **(fetchers or {})}
    sv = conf.get("survey") or {}
    shard_files = sorted(glob.glob(str(out / f"shard_s*of{n}.json")))
    shards = [json.loads(Path(p).read_text()) for p in shard_files]
    found = {s.get("shard") for s in shards}
    expected = {_tag(i, n) for i in range(n)}
    stars = []
    for i in range(n):
        p = out / f"stars_{_tag(i, n)}.csv"
        if p.exists():
            d = pd.read_csv(p, dtype={"source_id": str})
            d["shard"] = _tag(i, n)
            stars.append(d)
    st = pd.concat(stars, ignore_index=True) if stars else pd.DataFrame()
    n_dup = 0
    if len(st):
        n_dup = int(st.duplicated("source_id").sum())
        st = st.drop_duplicates("source_id", keep="last")
    zst = []
    for i in range(n):
        p = out / f"ztfstatus_{_tag(i, n)}.csv"
        if p.exists():
            zst.append(pd.read_csv(p, dtype={"source_id": str}))
    zs = pd.concat(zst, ignore_index=True).drop_duplicates("source_id", keep="last") \
        if zst else pd.DataFrame(columns=["source_id", "status"])
    # --- null pooled ---------------------------------------------------------
    pair_scores, shift_scores = [], []
    pair_rounds, pair_counts, shift_counts, null_stars = 0, [], [], 0
    for i in range(n):
        p = out / f"null_{_tag(i, n)}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        pair_scores += d.get("pair_scores", [])
        shift_scores += d.get("shift_scores", [])
        pair_rounds = max(pair_rounds, int(d.get("n_rounds_pair", 0)))
        pair_counts.append(d.get("pair_coupled_per_round", []))
        shift_counts.append(list((d.get("shift_coupled") or {}).values()))
        null_stars += int(d.get("n_stars", 0))
    null_trials = null_stars * max(pair_rounds, 1)
    pc = [sum(x[k] for x in pair_counts if len(x) > k) for k in range(pair_rounds)]
    # --- counts ---------------------------------------------------------------
    n_eval = int(len(st))
    labels = st["coupling_label"].value_counts().to_dict() if n_eval else {}
    verdicts = st["verdict"].value_counts().to_dict() if n_eval else {}
    coupled = st[st["coupling_label"] == "COUPLED"].copy() if n_eval else pd.DataFrame()
    real_coupled_null_stars = int(len(coupled))
    if len(coupled):
        coupled["null_fap"] = [score_fap(float(s), pair_scores, null_trials)
                               for s in coupled["score"]]
        coupled["expected_false"] = coupled["null_fap"] * max(n_eval, 1)
    # --- vet the survivors ----------------------------------------------------
    cands = coupled[coupled["verdict"].isin(["ANTIPHASE_CANDIDATE", "COUPLED_INCOMPLETE"])] \
        if len(coupled) else pd.DataFrame()
    irf = []
    for i in range(n):
        p = out / f"irfaded_{_tag(i, n)}.csv"
        if p.exists():
            irf.append(pd.read_csv(p, dtype={"source_id": str}))
    irf = pd.concat(irf, ignore_index=True) if irf else pd.DataFrame(
        columns=["source_id", "band", "t_yr", "mag", "err"])
    vet_rows, unreachable = [], 0
    for _, r in cands.iterrows():
        row = r.to_dict()
        v = vet_candidate(row, irf[irf["source_id"] == row["source_id"]], conf, f)
        if (v.get("blend") or {}).get("status") != "OK":
            unreachable += 1
        # the verdict re-run with the blend result (natural flags carried over)
        nat = {"flags": [x for x in str(row.get("natural_flags") or "").split(";") if x],
               "untested": [x for x in str(row.get("untested") or "").split(";")
                            if x and x != "neighbour_blend"]}
        cd = {"label": "COUPLED", "chroma": row.get("chroma"), "best_lag": row.get("best_lag"),
              "corr_best": row.get("corr_best"), "corr_lag0": row.get("corr_lag0")}
        en = {"verdict": row.get("energy_verdict")}
        fv = final_verdict(cd, en, nat, v.get("blend"), conf.get("classify"))
        sig = bool(np.isfinite(row.get("expected_false", np.nan))
                   and row["expected_false"] < float(sv.get("expected_false_max", 0.1)))
        final = fv["verdict"]
        if final == "ANTIPHASE_CANDIDATE" and not sig:
            final = "NOT_SIGNIFICANT_VS_NULL"
        vet_rows.append({**{k: row.get(k) for k in (
            "source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "v_tan_kms",
            "depth_mag", "depth_err", "chroma", "k_colour", "k_colour_err", "w1_sigma",
            "w2_sigma", "w2_contrast", "w2_slope_sigma", "ir_two_band", "energy_verdict", "ratio_best",
            "ratio_lo", "ratio_hi", "t_bb_k", "best_lag", "score", "null_fap",
            "expected_false", "faded_t_yr")},
            "verdict_shard": row.get("verdict"), "verdict_final": final,
            "natural_class": fv["natural_class"], "reasons": ";".join(fv["reasons"]),
            "untested": ";".join(fv["untested"]),
            "blend_explains": (v.get("blend") or {}).get("explains"),
            "blend_fraction": json.dumps(_jsonable((v.get("blend") or {})
                                                   .get("fraction_explained"))),
            "n_gaia_neighbours_20as": (v.get("blend") or {}).get("n_neighbours")})
    vet = pd.DataFrame(vet_rows)
    out.mkdir(parents=True, exist_ok=True)
    if len(coupled):
        coupled.to_csv(out / "coupled.csv", index=False)
    vet.to_csv(out / "candidates.csv", index=False)
    n_cand = int((vet["verdict_final"] == "ANTIPHASE_CANDIDATE").sum()) if len(vet) else 0
    # --- injections pooled -----------------------------------------------------
    inj: dict = {}
    for s in shards:
        for gr in ((s.get("injection") or {}).get("grid") or []):
            k = (gr["depth"], gr["t_k"])
            a = inj.setdefault(k, {"n": 0, "c": 0.0, "f": 0.0})
            if gr.get("n"):
                a["n"] += gr["n"]
                a["c"] += (gr.get("frac_coupled") or 0) * gr["n"]
                a["f"] += (gr.get("frac_full_ladder") or 0) * gr["n"]
    inj_grid = [{"depth": k[0], "t_k": k[1], "n": v["n"],
                 "frac_coupled": round(v["c"] / v["n"], 4) if v["n"] else None,
                 "frac_full_ladder": round(v["f"] / v["n"], 4) if v["n"] else None}
                for k, v in sorted(inj.items())]
    # --- controls --------------------------------------------------------------
    cpath = out / "controls.json"
    ctrl = json.loads(cpath.read_text()) if cpath.exists() else {}
    gate = ctrl.get("gate", "NOT_RUN")
    # --- funnel ---------------------------------------------------------------
    zc = zs["status"].value_counts().to_dict() if len(zs) else {}
    fun = {"n_parent": sum(int((s.get("funnel") or {}).get("n_parent", 0)) for s in shards),
           "n_ztf_footprint": sum(int((s.get("funnel") or {}).get("n_ztf_footprint", 0))
                                  for s in shards),
           "n_enough_ir_epochs": sum(int((s.get("funnel") or {}).get("n_enough_ir_epochs", 0))
                                     for s in shards),
           "n_ztf_attempted": int(len(zs)), "ztf_status": zc,
           "n_not_attempted_budget": sum(int((s.get("ztf") or {}).get("n_not_attempted", 0))
                                         for s in shards),
           "n_evaluated": n_eval, "coupling_labels": labels, "shard_verdicts": verdicts,
           "n_coupled": real_coupled_null_stars,
           "n_coupled_natural": int((coupled["verdict"] == "NATURAL").sum()) if len(coupled) else 0,
           "natural_classes": (coupled.loc[coupled["verdict"] == "NATURAL", "natural_class"]
                               .value_counts().to_dict() if len(coupled) else {}),
           "n_vetted": int(len(vet)),
           "final_verdicts": vet["verdict_final"].value_counts().to_dict() if len(vet) else {},
           "n_candidates": n_cand}
    degr = []
    if found != expected:
        degr.append(f"shards_missing:{len(expected - found)}")
    if zc.get("FAILED"):
        degr.append(f"ztf_failed:{zc['FAILED']}")
    if fun["n_not_attempted_budget"]:
        degr.append(f"ztf_budget_not_attempted:{fun['n_not_attempted_budget']}")
    if unreachable:
        degr.append(f"vet_unreachable:neighbours:{unreachable}")
    # the smallest expected-false count the null can certify: 1/(trials+1) x N
    floor_ef = n_eval / (null_trials + 1) if null_trials else float("inf")
    if len(vet) and floor_ef >= float(sv.get("expected_false_max", 0.1)):
        degr.append(f"null_resolution:floor_expected_false={floor_ef:.3g}")
    if not zc.get("OK"):
        verdict = "NO_DATA_REACHED"
    else:
        verdict = "ANTIPHASE_CANDIDATES" if n_cand else "NO_ANTIPHASE_CANDIDATE"
    if degr:
        verdict = f"DEGRADED ({'; '.join(degr)}); {verdict}"
    if gate == "FAIL":
        verdict = f"CONTROLS_FAILED; {verdict}"
    elif gate in ("UNTESTED", "NOT_RUN"):
        verdict = f"CONTROLS_{gate}; {verdict}"
    consistency = {
        "labels_sum_to_evaluated": bool(sum(labels.values()) == n_eval),
        "no_duplicate_stars": n_dup == 0, "n_duplicate_rows_dropped": n_dup,
        "shards_found_equals_expected": found == expected,
        "evaluated_le_ztf_ok": bool(n_eval <= int(zc.get("OK", 0))),
    }
    summ = {"channel": "antiphase", "verdict": verdict, "generated_utc": _now(),
            "run_id": run_id or os.environ.get("GITHUB_RUN_ID", ""), "git_sha": _git_sha(),
            "ignition_run_id": sv.get("ignition_run_id"),
            "controls_gate": gate, "controls_outcomes": ctrl.get("outcomes"),
            "funnel": fun,
            "coverage": {"shards_expected": n, "shards_found": len(found & expected),
                         "parent": "IGNITION tiles parent (Gaia DR3 G<14.5 dwarfs, plx>3 mas, "
                                   "|b|>15, AllWISE-photospheric), ZTF dec >= -31",
                         "ensemble": [s.get("ensemble", {}).get("frac_epochs_fine") for s in shards]},
            "null": {"n_stars": null_stars, "n_rounds_pair": pair_rounds,
                     "pair_coupled_per_round": pc,
                     "pair_coupled_mean": float(np.mean(pc)) if pc else None,
                     "shift_coupled_sum_per_shift": [sum(x[k] for x in shift_counts if len(x) > k)
                                                     for k in range(max((len(x) for x in
                                                                         shift_counts),
                                                                        default=0))],
                     "real_coupled": real_coupled_null_stars,
                     "real_coupled_in_null_subsample_note":
                         "null rounds use up to null_max_stars per shard; compare rates, "
                         "not raw counts, when that cap binds",
                     "n_null_trials": null_trials,
                     "floor_expected_false": floor_ef if np.isfinite(floor_ef) else None},
            "injection": inj_grid,
            "self_consistency": consistency,
            "candidates": vet[vet["verdict_final"] == "ANTIPHASE_CANDIDATE"].to_dict("records")
            if len(vet) else []}
    _write_json(out / "summary.json", summ)
    return summ


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="seti.antiphase.run")
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--ignition-dir", default="results/antiphase/ignition")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--config", default="")
    ap.add_argument("--budget-s", type=float, default=None)
    ap.add_argument("--max-stars", type=int, default=0)
    ap.add_argument("--run-id", default="")
    a = ap.parse_args(argv)
    conf = load_config(a.config or None)
    out = Path(a.out)
    if a.stage == "controls":
        res = run_controls(conf)
        _write_json(out / "controls.json", res)
        print(f"[antiphase] controls gate: {res['gate']}  outcomes: {res['outcomes']}")
        return 0
    if a.stage == "shard":
        i, n = (int(x) for x in a.shard.split("/"))
        rep = run_shard(conf, Path(a.ignition_dir), out, i, n, budget_s=a.budget_s,
                        max_stars=a.max_stars)
        print(f"[antiphase] shard {_tag(i, n)}: {rep.get('status')} funnel={rep.get('funnel')} "
              f"ztf={(rep.get('ztf') or {}).get('counts')} null={rep.get('null_summary')}")
        return 0
    n = a.shards or int((conf.get("survey") or {}).get("ignition_shards", 12))
    s = run_reduce(conf, out, n, run_id=a.run_id)
    print(f"[antiphase] {s['verdict']}")
    print(json.dumps(_jsonable(s["funnel"]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

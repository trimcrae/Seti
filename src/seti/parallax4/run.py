"""PARALLAX4 stage orchestration -> results/parallax4/.

Stages
------
probe      DR4 status (DR4_NOT_RELEASED until the tables exist), the CDN file
           list and its real header, the DR4 prerelease + draft data model,
           the SSO per-transit schema through the tolerant reader.
controls   FIRST, and gating: photometric injection/recovery on real DR3
           light curves (pilot CDN files), the simulated DR4 scenes, the real
           prerelease (parallaxes; DR3 photometry joined on transit_id; the
           joint photocentre fit), and the DR3 varstrometry population test.
sweep      one shard of the CDN files: every source through the grey
           detector, checkpointed per file.
reduce     pool the shards; cross-source epoch clustering; tiers.
vet        the surviving tier-A events: Gaia's own variability class,
           duplicated_source, neighbours (misassigned-transit flux match,
           crowding), RUWE/IPD blend indicators.
dr4        inert until DR4 exists; then the photocentre test on the watchlist
           and the survivors.
watchlist  every channel's shortlisted Gaia ids (release-day targets).
summary    summary.json over whatever stages have run.

CLI: ``seti parallax4 --stage {probe,controls,sweep,reduce,vet,dr4,watchlist,summary,all}``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile
import time as _time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import acquire as acq
from . import controls as C
from . import epochs as E
from . import greydip as G
from . import photocentre as P
from . import watchlist as W

OUT = Path("results") / "parallax4"
STAGES = ("probe", "controls", "sweep", "reduce", "vet", "watchlist", "dr4", "summary")
EXTRA_STAGES = ("deepvet",)
EVENT_COLS = ["source_id", "kind", "tier", "t_peak", "t_start", "t_end", "transit_id_peak",
              "n_transits_episode", "delta_g_peak", "z_g_peak", "delta_g", "sigma_g", "delta_bp",
              "sigma_bp", "delta_rp", "sigma_rp", "z_bp_episode", "z_rp_episode", "grey_class",
              "p_grey", "ratio_bp_rp", "ratio_err", "ratio_vs_dust_sigma", "coherence",
              "n_partner_contradicting", "any_variability_reject", "min_n_obs_g", "period_class",
              "period_d", "period_fap", "n_dip_episodes", "rms_out_of_episode_g", "s_int_g",
              "n_ok_g", "med_flux_g", "file"]


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    return o


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, default=_json_default) + "\n")


def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return {}


def _git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _provenance() -> dict:
    return {"generated_utc": _now(), "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "git_sha": _git_sha()}


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    p = Path(path) if path else Path("config") / "parallax4.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    a, b = str(s).split("/")
    i, n = int(a), int(b)
    if not (0 <= i < n):
        raise SystemExit(f"bad shard {s!r}")
    return i, n


def _tag(i: int, n: int) -> str:
    return f"s{i}of{n}"


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, http=acq.http_get, tap=acq.gaia_tap) -> dict:
    rep: dict = {"stage": "probe", **_provenance()}
    if tap is acq.gaia_tap:
        import functools

        tap = functools.partial(acq.gaia_tap, retries=2, deadline_s=300.0, sync_deadline_s=120.0)
    print("[parallax4] probe: DR4 status", flush=True)
    try:
        rep["dr4"] = acq.probe_dr4(tap=tap, http=http)
    except Exception as exc:  # noqa: BLE001
        rep["dr4"] = {"verdict": "ARCHIVE_UNREACHABLE", "error": repr(exc)[:400]}
    _write(out / "dr4_status.json", {**_provenance(), **rep["dr4"]})
    # CDN
    print(f"[parallax4] probe: DR4 -> {rep['dr4'].get('verdict')}; listing the CDN", flush=True)
    try:
        lst = acq.list_cdn_files(http=http)
    except Exception as exc:  # noqa: BLE001
        lst = {"ok": False, "error": repr(exc)[:400], "files": []}
    rep["cdn"] = {k: v for k, v in lst.items() if k not in ("files", "md5", "sizes")}
    rep["cdn"]["n_md5"] = len(lst.get("md5") or {})
    if lst.get("files"):
        _write(out / "cdn_files.json", {**_provenance(), "base": acq.CDN_EPOCH_PHOT,
                                        "files": lst["files"], "md5": lst.get("md5") or {},
                                        "sizes": lst.get("sizes") or {}})
        # the real header of the first file, resolved against the reader roles
        with tempfile.TemporaryDirectory() as td:
            try:
                f0 = acq.download_cdn_file(lst["files"][0], Path(td), http=http,
                                           md5=(lst.get("md5") or {}).get(lst["files"][0]))
                header = E.read_bulk_csv_header(f0)
                from .schema import PHOT_REQUIRED, PHOT_ROLES, resolve

                res = resolve(header, PHOT_ROLES, PHOT_REQUIRED + ("source_id",))
                t0 = _time.monotonic()
                ph = E.read_bulk_csv(f0)
                rep["cdn"]["first_file"] = {
                    "name": lst["files"][0], "bytes": f0.stat().st_size, "header": header,
                    "resolution": res.as_dict(), "n_sources": int(ph["source_id"].nunique()),
                    "n_transits": int(len(ph)), "read_s": round(_time.monotonic() - t0, 2),
                    "mismatched_array_rows": int(ph.attrs.get("mismatched_array_rows", 0)),
                    "t_range": [float(np.nanmin(ph["t"])), float(np.nanmax(ph["t"]))]}
            except Exception as exc:  # noqa: BLE001
                rep["cdn"]["first_file"] = {"error": repr(exc)[:600]}
    # DR4 prerelease + draft data model
    print(f"[parallax4] probe: CDN {rep['cdn'].get('n_files')} files; fetching the DR4 prerelease", flush=True)
    pre_dir = out / "_prerelease"
    try:
        pr = acq.fetch_prerelease(pre_dir, http=http)
    except Exception as exc:  # noqa: BLE001
        pr = {"error": repr(exc)[:400]}
    rep["prerelease"] = pr
    dm = (pr.get("datamodel") or {})
    if dm.get("ok"):
        try:
            parsed = acq.parse_datamodel(Path(dm["path"]))
            rep["datamodel"] = {"members": parsed["members"],
                                "tables_mentioned": {k: v["mentions"] for k, v in parsed["tables"].items()}}
            # every reader role checked against the announced identifiers
            from .schema import ASTRO_REQUIRED, ASTRO_ROLES, PHOT_REQUIRED, PHOT_ROLES, resolve

            idents = sorted({i for v in parsed["tables"].values() for i in v["identifiers"]})
            win = parsed.get("table_windows") or {}
            ph_ids = win.get("epoch_photometry") or idents
            as_ids = win.get("epoch_astrometry") or idents
            rep["datamodel"]["pdf_text_chars"] = parsed.get("pdf_text_chars")
            rep["datamodel"]["phot_roles_resolved"] = resolve(ph_ids, PHOT_ROLES).mapping
            rep["datamodel"]["phot_required_missing"] = resolve(ph_ids, PHOT_ROLES, PHOT_REQUIRED).missing
            rep["datamodel"]["astro_roles_resolved"] = resolve(as_ids, ASTRO_ROLES).mapping
            rep["datamodel"]["astro_required_missing"] = resolve(as_ids, ASTRO_ROLES, ASTRO_REQUIRED).missing
            rep["datamodel"]["epoch_photometry_identifiers"] = ph_ids[:400]
            _write(out / "dr4_datamodel.json", {**_provenance(), **parsed})
        except Exception as exc:  # noqa: BLE001
            rep["datamodel"] = {"error": repr(exc)[:400]}
    print("[parallax4] probe: SSO + vari schemas", flush=True)
    # SSO per-transit astrometry through the tolerant reader (format validation)
    for tab in ("gaiadr3.sso_observation", "gaiafpr.sso_observation"):
        try:
            df = tap(f"SELECT TOP 20 * FROM {tab}")
            from .schema import ASTRO_ROLES, resolve

            r = resolve(df.columns, ASTRO_ROLES)
            rep.setdefault("sso", {})[tab] = {"ok": True, "n_columns": int(len(df.columns)),
                                             "columns": list(df.columns), "roles": r.mapping}
        except Exception as exc:  # noqa: BLE001
            rep.setdefault("sso", {})[tab] = {"ok": False, "error": repr(exc)[:300]}
    for tab in ("gaiadr3.vari_classifier_result", "gaiadr3.vari_eclipsing_binary",
                "gaiadr3.vari_summary"):
        try:
            df = tap(f"SELECT TOP 1 * FROM {tab}")
            rep.setdefault("vari_tables", {})[tab] = {"ok": True, "columns": list(df.columns)}
        except Exception as exc:  # noqa: BLE001
            rep.setdefault("vari_tables", {})[tab] = {"ok": False, "error": repr(exc)[:300]}
    rep["verdict"] = ("PROBE_OK" if lst.get("files") and "error" not in rep["cdn"].get("first_file", {})
                      else "NO_DATA_REACHED")
    _write(out / "probe.json", rep)
    print(f"[parallax4] probe: {rep['verdict']}; DR4: {rep['dr4'].get('verdict')}; "
          f"CDN files: {rep['cdn'].get('n_files')}")
    return rep


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
def _cdn_list(out: Path, http) -> dict:
    d = _read(out / "cdn_files.json")
    if d.get("files"):
        return d
    try:
        lst = acq.list_cdn_files(http=http)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)[:300], "files": []}
    if lst.get("files"):
        _write(out / "cdn_files.json", {**_provenance(), "base": acq.CDN_EPOCH_PHOT,
                                        "files": lst["files"], "md5": lst.get("md5") or {}})
    return lst


def stage_controls(conf: dict, out: Path, *, http=acq.http_get, tap=acq.gaia_tap,
                   prerelease_zip: Path | None = None) -> dict:
    cc = conf.get("controls") or {}
    gcfg = G.GreyConfig.from_dict(conf.get("grey"))
    pcfg = P.PhotocentreConfig.from_dict(conf.get("photocentre"))
    rep: dict = {"stage": "controls", **_provenance()}
    # --- A1 simulated DR4 scenes (no network) -------------------------------
    rep["A1_simulated_scenes"] = C.simulated_scene_control(cfg=pcfg)
    # --- P1/P2/P4 photometric pilot on real CDN files ------------------------
    lst = _cdn_list(out, http)
    files = list(lst.get("files") or [])
    trials: list[dict] = []
    pilot_events = []
    mism = 0
    n_src = 0
    pilot_files = []
    if files:
        idx = np.linspace(0, len(files) - 1, int(cc.get("pilot_files", 3))).round().astype(int)
        with tempfile.TemporaryDirectory() as td:
            for k in sorted(set(idx.tolist())):
                name = files[k]
                try:
                    p = acq.download_cdn_file(name, Path(td), http=http, md5=(lst.get("md5") or {}).get(name))
                    ph = E.read_bulk_csv(p)
                    p.unlink(missing_ok=True)
                except Exception as exc:  # noqa: BLE001
                    pilot_files.append({"file": name, "error": repr(exc)[:300]})
                    continue
                mism += int(ph.attrs.get("mismatched_array_rows", 0))
                ev, counts = G.detect_frame(ph, gcfg)
                if len(ev):
                    ev["file"] = name
                    pilot_events.append(ev)
                n_src += counts["n_searched"]
                trials += C.injection_trials(ph, gcfg, every=int(cc.get("inject_every", 25)),
                                             max_trials=int(cc.get("max_trials", 600)) - len(trials),
                                             seed=20260924 + k)
                pilot_files.append({"file": name, **counts, "n_events": int(len(ev))})
    rep["pilot_files"] = pilot_files
    rep["P_photometric"] = C.photometric_gate(
        trials, reader_mismatched_rows=mism, min_recovery=float(cc.get("min_recovery", 0.8)),
        max_dust_as_grey=float(cc.get("max_dust_as_grey", 0.05)),
        snr_floor=float(cc.get("snr_floor", 10.0)))
    if not files:
        rep["P_photometric"].update(gate="FAIL", reason="CDN file list unreachable")
    pd.DataFrame(trials).to_csv(out / "controls_injections.csv", index=False)
    pev = pd.concat(pilot_events, ignore_index=True) if pilot_events else pd.DataFrame()
    # P3: Gaia's own EBs among the pilot events
    try:
        grey_ids = pev.loc[(pev["grey_class"] == "GREY"), "source_id"].unique() if len(pev) else []
        vari = acq.vari_membership(grey_ids, tap=tap) if len(grey_ids) else pd.DataFrame()
        rep["P3_eclipsing_binaries"] = C.eclipsing_binary_control(pev, vari)
    except Exception as exc:  # noqa: BLE001
        rep["P3_eclipsing_binaries"] = {"error": repr(exc)[:300]}
    rep["pilot_funnel"] = {
        "n_sources_searched": n_src, "n_events": int(len(pev)),
        "by_grey_class": {str(k): int(v) for k, v in pev["grey_class"].value_counts().items()} if len(pev) else {},
        "by_tier": {str(k): int(v) for k, v in pev["tier"].value_counts().items()} if len(pev) else {}}
    # --- A2 the real DR4 prerelease, joined to DR3 photometry ----------------
    try:
        zp = prerelease_zip
        if zp is None:
            cand = [out / "_prerelease" / acq.PRERELEASE_ZIP,
                    Path("tests") / "fixtures" / "parallax4" / acq.PRERELEASE_ZIP]
            zp = next((c for c in cand if c.exists()), None)
        if zp is None:
            pr = acq.fetch_prerelease(out / "_prerelease", http=http)
            zp = Path(pr["epoch_astrometry"]["path"]) if pr.get("epoch_astrometry", {}).get("ok") else None
        if zp is None:
            raise FileNotFoundError("DR4 prerelease zip unavailable")
        ccd = E.read_prerelease_zip(zp)
        ids = [int(s) for s in cc.get("joint_sources", [])]
        phot_by, nb_by, flux_by = {}, {}, {}
        try:
            tabs = acq.datalink_products(ids, http=http)
            for sid, tab in tabs.items():
                phot_by[sid] = E.photometry_from_long(tab, source_id=sid)
        except Exception as exc:  # noqa: BLE001
            rep["A2_datalink_error"] = repr(exc)[:300]
        fx = Path("tests") / "fixtures" / "parallax4" / "dr3_epoch_photometry_1457486023639239296.vot"
        if 1457486023639239296 not in phot_by and fx.exists():
            from astropy.table import Table

            phot_by[1457486023639239296] = E.photometry_from_long(
                Table.read(fx, format="votable"), source_id=1457486023639239296)
            rep["A2_gaia4_photometry_from"] = "fixture"
        try:
            gs = acq.gaia_source_rows(list(phot_by), tap=tap)
            for _, r in gs.iterrows():
                sid = int(r["source_id"])
                flux_by[sid] = float(r["phot_g_mean_flux"])
                cone = acq.gaia_cone(float(r["ra"]), float(r["dec"]), 5.0, tap=tap)
                cone = cone[cone["source_id"] != sid]
                nb_by[sid] = P.neighbour_offsets(float(r["ra"]), float(r["dec"]), cone)
        except Exception as exc:  # noqa: BLE001
            rep["A2_neighbour_error"] = repr(exc)[:300]
        rep["A2_prerelease"] = C.prerelease_control(ccd, phot_by, nb_by, flux_by)
    except Exception as exc:  # noqa: BLE001
        rep["A2_prerelease"] = {"gate": "FAIL", "error": repr(exc)[:400]}
    # --- A3 DR3 varstrometry --------------------------------------------------
    try:
        samples = acq.varstrometry_samples(tap=tap)
        for k, v in samples.items():
            v.to_csv(out / f"varstrometry_{k}.csv.gz", index=False, compression="gzip")
        rep["A3_varstrometry"] = C.varstrometry_control(samples["eclipsing"], samples["quiet"])
        rep["A3_varstrometry_excess_noise"] = C.varstrometry_control(
            samples["eclipsing"], samples["quiet"], metric="astrometric_excess_noise")
    except Exception as exc:  # noqa: BLE001
        rep["A3_varstrometry"] = {"gate": "DEGRADED", "error": repr(exc)[:400]}
    # --- A4 the release-day control list, fixed before DR4 exists -------------
    rep["A4_dr4_controls"] = build_dr4_controls(out, tap=tap)
    ph_gate = rep["P_photometric"].get("gate")
    a_gates = [rep["A1_simulated_scenes"].get("gate"), rep.get("A2_prerelease", {}).get("gate"),
               rep.get("A3_varstrometry", {}).get("gate")]
    rep["gate_photometric"] = "PASS" if ph_gate == "PASS" else "FAIL"
    rep["gate_astrometric"] = ("PASS" if all(g == "PASS" for g in a_gates)
                               else "FAIL" if "FAIL" in a_gates else "DEGRADED")
    rep["verdict"] = f"CONTROLS photometric={rep['gate_photometric']} astrometric={rep['gate_astrometric']}"
    _write(out / "controls.json", rep)
    print(f"[parallax4] {rep['verdict']}")
    return rep


def build_dr4_controls(out: Path, *, tap=acq.gaia_tap) -> dict:
    """Write ``dr4_controls.csv``: the sources whose DR4 verdict is known in
    advance, so release day starts with a test of the test.

    VIM_POSITIVE        every Gaia DR3 Variability-Induced Mover
                        (``nss_vim_fl``): expected BLEND_*
    SINGLE_EB_NEGATIVE  bright isolated clean eclipsing binaries: expected
                        not BLEND (ON_TARGET where the leverage allows)
    """
    rep: dict = {}
    frames = []
    try:
        vim = acq.vim_sources(tap=tap)
        vim = vim.rename(columns={c: c.lower() for c in vim.columns})
        keep = ["source_id"] + [c for c in vim.columns if c.startswith("vim_") or c in ("ra", "dec")]
        v = vim[keep].copy()
        v["role"], v["expected"] = "VIM_POSITIVE", "BLEND"
        frames.append(v)
        rep["n_vim"] = int(len(v))
        rep["vim_columns"] = list(vim.columns)
    except Exception as exc:  # noqa: BLE001
        rep["vim_error"] = repr(exc)[:300]
    try:
        eb = acq.single_eb_controls(tap=tap)
        e = eb[["source_id"] + [c for c in ("ra", "dec", "phot_g_mean_mag", "ruwe") if c in eb]].copy()
        e["role"], e["expected"] = "SINGLE_EB_NEGATIVE", "NOT_BLEND"
        frames.append(e)
        rep["n_single_eb"] = int(len(e))
    except Exception as exc:  # noqa: BLE001
        rep["single_eb_error"] = repr(exc)[:300]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "dr4_controls.csv", index=False)
    rep["status"] = "WRITTEN" if frames else "UNREACHABLE"
    return rep


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def stage_sweep(conf: dict, out: Path, *, shard: int = 0, n_shards: int = 1, max_files: int = 0,
                http=acq.http_get, force: bool = False, files_override: list[str] | None = None,
                local_dir: Path | None = None) -> dict:
    sc = conf.get("sweep") or {}
    gcfg = G.GreyConfig.from_dict(conf.get("grey"))
    tag = _tag(shard, n_shards)
    prog_path = out / f"sweep_{tag}.json"
    ev_path = out / f"events_{tag}.csv"
    ab_path = out / f"eventsAB_{tag}.csv"
    inj_path = out / f"injections_{tag}.csv"
    ctrl = _read(out / "controls.json")
    if not force and ctrl.get("gate_photometric") != "PASS":
        rep = {"stage": "sweep", "shard": shard, "n_shards": n_shards, **_provenance(),
               "verdict": "REFUSED_CONTROLS_NOT_PASSED",
               "reason": f"controls.json gate_photometric={ctrl.get('gate_photometric')!r}"}
        _write(prog_path, rep)
        print(f"[parallax4] sweep {tag}: {rep['verdict']}")
        return rep
    if files_override is not None:
        files, md5 = list(files_override), {}
    else:
        lst = _cdn_list(out, http)
        files, md5 = list(lst.get("files") or []), dict(lst.get("md5") or {})
    mine = files[shard::n_shards]
    if max_files:
        mine = mine[:max_files]
    prog = _read(prog_path) if prog_path.exists() else {}
    done = set(prog.get("done", []))
    t_lo, t_hi, dt = float(sc.get("t_lo", 1600)), float(sc.get("t_hi", 2800)), float(sc.get("time_bin_d", 0.25))
    nb = int(round((t_hi - t_lo) / dt))
    edges = np.linspace(t_lo, t_hi, nb + 1)
    h_tr = np.asarray(prog.get("hist_transits") or np.zeros(nb), dtype=float)
    h_grey = np.asarray(prog.get("hist_grey_episodes") or np.zeros(nb), dtype=float)
    h_all = np.asarray(prog.get("hist_all_episodes") or np.zeros(nb), dtype=float)
    counts = dict(prog.get("counts") or {"n_sources": 0, "n_searched": 0, "n_too_few": 0,
                                         "n_transits_ok_g": 0, "n_events": 0, "n_AB": 0})
    per_file = list(prog.get("files") or [])
    errors = list(prog.get("errors") or [])
    t0 = _time.monotonic()
    budget = float(sc.get("time_budget_s", 19800))
    keep = set(sc.get("keep_tiers", ["A", "B"]))
    inj_every = int(sc.get("inject_every", 400))
    stopped_early = False
    tmp_root = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()))
    for name in mine:
        if name in done:
            continue
        if _time.monotonic() - t0 > budget:
            stopped_early = True
            break
        tf = _time.monotonic()
        try:
            if local_dir is not None:
                p = Path(local_dir) / name
            else:
                p = acq.download_cdn_file(name, tmp_root, http=http, md5=md5.get(name))
            ph = E.read_bulk_csv(p)
            if local_dir is None:
                p.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": name, "error": repr(exc)[:300]})
            _write(prog_path, _sweep_record(shard, n_shards, mine, done, counts, per_file, errors,
                                            h_tr, h_grey, h_all, edges, False))
            continue
        okt = np.isfinite(ph["t"].to_numpy(float)) & ~ph["bad_g"].to_numpy() & np.isfinite(ph["f_g"].to_numpy(float))
        h_tr += np.histogram(ph["t"].to_numpy(float)[okt], bins=edges)[0]
        ev, cnt = G.detect_frame(ph, gcfg)
        for k in ("n_sources", "n_searched", "n_too_few", "n_transits_ok_g"):
            counts[k] = counts.get(k, 0) + int(cnt[k])
        if len(ev):
            ev["file"] = name
            ev = ev.reindex(columns=EVENT_COLS)
            ev.to_csv(ev_path, mode="a", header=not ev_path.exists(), index=False)
            ab = ev[ev["tier"].isin(keep)]
            if len(ab):
                ab.to_csv(ab_path, mode="a", header=not ab_path.exists(), index=False)
            g = ev[ev["grey_class"] == "GREY"]
            h_grey += np.histogram(g["t_peak"].to_numpy(float), bins=edges)[0]
            h_all += np.histogram(ev["t_peak"].to_numpy(float), bins=edges)[0]
            counts["n_events"] += int(len(ev))
            counts["n_AB"] += int(len(ab))
        # completeness bookkeeping at scale
        if inj_every:
            tr = C.injection_trials(ph, gcfg, every=inj_every, max_trials=10 ** 6,
                                    seed=zlib.crc32(name.encode()))
            if tr:
                pd.DataFrame(tr).assign(file=name).to_csv(inj_path, mode="a",
                                                          header=not inj_path.exists(), index=False)
        done.add(name)
        per_file.append({"file": name, "n_sources": int(cnt["n_sources"]), "n_searched": int(cnt["n_searched"]),
                         "n_events": int(len(ev)), "seconds": round(_time.monotonic() - tf, 1),
                         "mismatched_array_rows": int(ph.attrs.get("mismatched_array_rows", 0))})
        _write(prog_path, _sweep_record(shard, n_shards, mine, done, counts, per_file, errors,
                                        h_tr, h_grey, h_all, edges, False))
        print(f"[parallax4] sweep {tag}: {len(done)}/{len(mine)} {name} "
              f"src={cnt['n_searched']} ev={len(ev)} {per_file[-1]['seconds']}s", flush=True)
    rep = _sweep_record(shard, n_shards, mine, done, counts, per_file, errors, h_tr, h_grey, h_all,
                        edges, stopped_early)
    _write(prog_path, rep)
    print(f"[parallax4] sweep {tag}: {rep['verdict']}")
    return rep


def _sweep_record(shard, n_shards, mine, done, counts, per_file, errors, h_tr, h_grey, h_all, edges,
                  stopped_early) -> dict:
    complete = len(done) >= len(mine)
    return {"stage": "sweep", "shard": shard, "n_shards": n_shards, **_provenance(),
            "n_files_planned": len(mine), "n_files_done": len(done), "done": sorted(done),
            "counts": counts, "files": per_file, "errors": errors, "stopped_early": stopped_early,
            "time_edges": [float(edges[0]), float(edges[-1]), int(len(edges) - 1)],
            "hist_transits": h_tr.tolist(), "hist_grey_episodes": h_grey.tolist(),
            "hist_all_episodes": h_all.tolist(),
            "verdict": "SHARD_COMPLETE" if complete else ("SHARD_PARTIAL_TIME_BUDGET" if stopped_early
                                                          else "SHARD_PARTIAL")}


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------
def epoch_clusters(h_tr: np.ndarray, h_ev: np.ndarray, *, factor: float = 10.0, min_events: int = 5,
                   p_max: float = 1e-6) -> np.ndarray:
    """Time bins whose event rate per usable transit is anomalously high
    (Poisson, against the global rate) -- instrument, not sky."""
    from scipy.stats import poisson

    tot_tr, tot_ev = float(h_tr.sum()), float(h_ev.sum())
    if tot_tr <= 0 or tot_ev <= 0:
        return np.zeros(len(h_tr), bool)
    rate = tot_ev / tot_tr
    mu = h_tr * rate
    p = poisson.sf(h_ev - 1, np.maximum(mu, 1e-12))
    return (h_ev >= min_events) & (h_ev > factor * mu) & (p < p_max)


def hp6_of(source_id) -> np.ndarray:
    """HEALPix level-6 (nested) pixel from Gaia source_id (level-12 index in
    the bits above 2^35; level 6 is 12 bits coarser): ~0.92 deg pixels."""
    return (np.asarray(source_id, dtype=np.int64) >> 47).astype(np.int64)


def _all_event_times(out: Path, n_shards: int | None) -> pd.DataFrame:
    files = sorted(glob.glob(str(out / "events_s*of*.csv")))
    if n_shards:
        files = [f for f in files if f.endswith(f"of{int(n_shards)}.csv")]
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, usecols=["source_id", "t_peak"]))
        except Exception:  # noqa: BLE001
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["source_id", "t_peak"])


def local_coincidence(ev: pd.DataFrame, allev: pd.DataFrame, *, dt: float = 0.1) -> dict:
    """For each row of ``ev``: how many OTHER sources in the same level-6
    pixel have a qualifying episode within +-dt, and the Poisson probability
    of that many given the pixel's own event rate over the DR3 baseline."""
    from scipy.stats import poisson

    a = allev.copy()
    a["hp"] = hp6_of(a["source_id"])
    a = a.sort_values(["hp", "t_peak"])
    groups = {k: (g["t_peak"].to_numpy(float), g["source_id"].to_numpy(np.int64))
              for k, g in a.groupby("hp", sort=False)}
    span = float(np.nanmax(a["t_peak"]) - np.nanmin(a["t_peak"])) if len(a) else 1.0
    n_out = np.zeros(len(ev), int)
    p_out = np.ones(len(ev))
    for i, (sid, t) in enumerate(zip(ev["source_id"].to_numpy(np.int64), ev["t_peak"].to_numpy(float),
                                     strict=False)):
        g = groups.get(int(hp6_of([sid])[0]))
        if g is None:
            continue
        tt, ss = g
        lo, hi = np.searchsorted(tt, t - dt), np.searchsorted(tt, t + dt, side="right")
        others = ss[lo:hi]
        k = int(len(np.unique(others[others != sid])))
        n_src_events = int((ss != sid).sum())
        mu = n_src_events * (2 * dt) / max(span, 1.0)
        n_out[i] = k
        p_out[i] = float(poisson.sf(k - 1, max(mu, 1e-12))) if k > 0 else 1.0
    return {"n": n_out, "p": p_out}


def stage_reduce(conf: dict, out: Path, *, n_shards_expected: int | None = None) -> dict:
    rc = conf.get("reduce") or {}
    rep: dict = {"stage": "reduce", **_provenance()}
    recs = [_read(Path(f)) for f in sorted(glob.glob(str(out / "sweep_s*of*.json")))]
    recs = [r for r in recs if r.get("stage") == "sweep"]
    if n_shards_expected:
        recs = [r for r in recs if int(r.get("n_shards", 0)) == int(n_shards_expected)]
    refused = [r for r in recs if r.get("verdict") == "REFUSED_CONTROLS_NOT_PASSED"]
    recs = [r for r in recs if r.get("verdict") != "REFUSED_CONTROLS_NOT_PASSED"]
    if not recs:
        rep.update(verdict="NO_SHARD_OUTPUTS" if not refused else "REFUSED_CONTROLS_NOT_PASSED",
                   n_shards_found=0)
        _write(out / "reduce.json", rep)
        print(f"[parallax4] reduce: {rep['verdict']}")
        return rep
    counts: dict = {}
    for r in recs:
        for k, v in (r.get("counts") or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
    nb = int(recs[0]["time_edges"][2])
    edges = np.linspace(recs[0]["time_edges"][0], recs[0]["time_edges"][1], nb + 1)
    h_tr = np.sum([np.asarray(r["hist_transits"], float) for r in recs], axis=0)
    h_grey = np.sum([np.asarray(r["hist_grey_episodes"], float) for r in recs], axis=0)
    h_all = np.sum([np.asarray(r["hist_all_episodes"], float) for r in recs], axis=0)
    bad = epoch_clusters(h_tr, h_grey, factor=float(rc.get("epoch_rate_factor", 10.0)),
                         min_events=int(rc.get("epoch_min_events", 5)), p_max=float(rc.get("epoch_p_max", 1e-6)))
    bad_all = epoch_clusters(h_tr, h_all, factor=float(rc.get("epoch_rate_factor", 10.0)),
                             min_events=int(rc.get("epoch_min_events", 5)), p_max=float(rc.get("epoch_p_max", 1e-6)))
    flagged = bad | bad_all
    rep["epoch_clusters"] = [{"t_lo": float(edges[i]), "t_hi": float(edges[i + 1]),
                              "n_transits": int(h_tr[i]), "n_grey_episodes": int(h_grey[i]),
                              "n_all_episodes": int(h_all[i])} for i in np.nonzero(flagged)[0]]
    files = sorted(glob.glob(str(out / "eventsAB_s*of*.csv")))
    if n_shards_expected:
        files = [f for f in files if f.endswith(f"of{int(n_shards_expected)}.csv")]
    frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    ab = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_COLS)
    n_before = len(ab)
    ab = ab.drop_duplicates(["source_id", "kind", "t_peak"]).reset_index(drop=True)
    if len(ab):
        bi = np.clip(np.searchsorted(edges, ab["t_peak"].to_numpy(float), side="right") - 1, 0, nb - 1)
        ab["epoch_cluster"] = flagged[bi]
        # the LOCAL form of the same veto: other sources' qualifying episodes
        # in the same ~0.9-deg sky pixel within +-0.1 d (one scan passes a
        # pixel in seconds; an instrument event there hits its neighbours)
        allev = _all_event_times(out, n_shards_expected)
        rep["n_all_events_for_local_veto"] = int(len(allev))
        if len(allev):
            loc = local_coincidence(ab, allev, dt=float(rc.get("local_dt_d", 0.1)))
            ab["n_local_coincident"] = loc["n"]
            ab["p_local_coincident"] = loc["p"]
            ab["epoch_cluster"] = ab["epoch_cluster"] | (loc["p"] < float(rc.get("local_p_max", 1e-3)))
            rep["n_local_coincidence_vetoed"] = int((loc["p"] < float(rc.get("local_p_max", 1e-3))).sum())
        rng = ab["rms_out_of_episode_g"].astype(float).clip(lower=1e-3)
        ab["score"] = (ab["delta_g"].abs() / rng) * np.sqrt(ab["n_transits_episode"].astype(float))
    else:
        ab["epoch_cluster"] = []
        ab["score"] = []
    ab = ab.sort_values(["tier", "score"], ascending=[True, False])
    # The committed table holds tier A only (every tier-A episode, epoch-
    # flagged ones included, 6 significant figures): tier B at catalogue scale
    # is tens of MB.  Every tier-A/B row stays in the per-shard artifacts
    # (eventsAB_s<i>of<n>.csv), and the counts are in reduce.json.
    ab[ab["tier"] == "A"].to_csv(out / "events_AB.csv", index=False, float_format="%.6g")
    a = ab[(ab["tier"] == "A") & ~ab["epoch_cluster"].astype(bool)] if len(ab) else ab
    b = ab[(ab["tier"] == "B") & ~ab["epoch_cluster"].astype(bool)] if len(ab) else ab
    planned = sum(int(r.get("n_files_planned", 0)) for r in recs)
    done = sum(int(r.get("n_files_done", 0)) for r in recs)
    n_err = sum(len(r.get("errors") or []) for r in recs)
    degraded = []
    if n_shards_expected and len(recs) < int(n_shards_expected):
        degraded.append(f"shards_missing:{len(recs)}/{n_shards_expected}")
    if done < planned:
        degraded.append(f"files_incomplete:{done}/{planned}")
    if n_err:
        degraded.append(f"file_errors:{n_err}")
    rep.update(n_shards_found=len(recs), n_files_planned=planned, n_files_done=done, counts=counts,
               n_AB_rows_raw=int(n_before), n_AB=int(len(ab)),
               n_tier_A=int((ab["tier"] == "A").sum()) if len(ab) else 0,
               n_tier_B=int((ab["tier"] == "B").sum()) if len(ab) else 0,
               n_epoch_cluster_vetoed=int(ab["epoch_cluster"].astype(bool).sum()) if len(ab) else 0,
               n_tier_A_after_epoch=int(len(a)), n_tier_B_after_epoch=int(len(b)),
               n_sources_tier_A_after_epoch=int(a["source_id"].nunique()) if len(a) else 0,
               degraded=degraded)
    # completeness at scale from the in-sweep injections
    inj_files = sorted(glob.glob(str(out / "injections_s*of*.csv")))
    if inj_files:
        inj = pd.concat([pd.read_csv(f) for f in inj_files if os.path.getsize(f) > 0], ignore_index=True)
        rep["completeness"] = C.photometric_gate(inj.to_dict("records"))
        rep["completeness"].pop("gate", None)
        rep["completeness"].pop("failures", None)
    rep["verdict"] = (f"DEGRADED ({'; '.join(degraded)}); " if degraded else "") + \
        f"REDUCED: {len(a)} tier-A events on {rep['n_sources_tier_A_after_epoch']} sources"
    _write(out / "reduce.json", rep)
    print(f"[parallax4] reduce: {rep['verdict']}")
    return rep


# ---------------------------------------------------------------------------
# vet
# ---------------------------------------------------------------------------
def _true(x) -> bool:
    """True only for a real true value: NaN from a left join is truthy in
    Python and must not read as 'catalogued eclipsing binary'."""
    if x is None:
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "t", "yes")
    try:
        return bool(np.isfinite(float(x)) and float(x) != 0)
    except (TypeError, ValueError):
        return False


def vet_rules(ev: pd.DataFrame, gs: pd.DataFrame, vari: pd.DataFrame, cones: dict[int, pd.DataFrame],
              vc: dict, vim: set[int] | None = None) -> pd.DataFrame:
    """Pure: every tier-A event against Gaia's own knowledge of the source."""
    vim = vim or set()
    d = ev.merge(gs, on="source_id", how="left", suffixes=("", "_gs"))
    if vari is not None and len(vari):
        d = d.merge(vari.drop_duplicates("source_id"), on="source_id", how="left")
    kills, flags = [], []
    for _, r in d.iterrows():
        k, f = [], []
        if _true(r.get("in_vari_eclipsing_binary")):
            k.append("GAIA_ECLIPSING_BINARY")
        bc = r.get("best_class_name")
        bc = "" if bc is None or (isinstance(bc, float) and not np.isfinite(bc)) else str(bc)
        if bc.upper().startswith("ECL"):
            k.append("GAIA_CLASS_ECL")
        elif bc and bc.lower() not in ("nan", "none", ""):
            f.append(f"gaia_class:{bc}")
        if _true(r.get("duplicated_source")):
            k.append("DUPLICATED_SOURCE")
        if int(r["source_id"]) in vim:
            k.append("GAIA_VIM_BLEND")          # DR3 already saw the photocentre move with flux
        if np.nan_to_num(float(r.get("ruwe") or 0)) > float(vc.get("ruwe_flag", 1.4)):
            f.append("ruwe_high")
        if np.nan_to_num(float(r.get("ipd_frac_multi_peak") or 0)) > float(vc.get("ipd_multi_flag", 10.0)):
            f.append("ipd_multi_peak")
        cone = cones.get(int(r["source_id"]))
        if cone is not None and len(cone):
            nbs = cone[cone["source_id"] != r["source_id"]]
            g_t = float(r.get("phot_g_mean_mag") or np.nan)
            crowd = nbs[(nbs["sep_arcsec"] <= float(vc.get("crowd_arcsec", 2.0)))
                        & (nbs["phot_g_mean_mag"] - g_t <= float(vc.get("crowd_dmag", 3.0)))]
            if len(crowd):
                f.append(f"crowded:{len(crowd)}")
            # a transit assigned the WRONG source: the episode flux equals a
            # neighbour's (dip) or target+neighbour (brightening)
            fmed = float(r.get("med_flux_g") or np.nan)
            dg, sg = float(r.get("delta_g") or np.nan), float(r.get("sigma_g") or np.nan)
            if np.isfinite(fmed) and np.isfinite(dg) and len(nbs) and "phot_g_mean_flux" in nbs:
                obs = fmed * (1 + dg)
                sig = max(fmed * sg, 0.02 * fmed) * float(vc.get("match_sigma", 3.0))
                fn = nbs["phot_g_mean_flux"].to_numpy(float)
                target = fn if dg < 0 else fmed + fn
                hit = np.isfinite(target) & (np.abs(obs - target) <= sig)
                if hit.any():
                    k.append("FLUX_MATCHES_NEIGHBOUR")
        kills.append(";".join(k))
        flags.append(";".join(f))
    d["kill_reasons"] = kills
    d["flags"] = flags
    d["vet_class"] = np.where(d["kill_reasons"] != "", "KILLED",
                              np.where(d["flags"] != "", "SURVIVES_FLAGGED", "SURVIVES"))
    return d


def stage_vet(conf: dict, out: Path, *, tap=acq.gaia_tap) -> dict:
    """Two levels.  Catalogue level for EVERY tier-A source (IN-list TAP, cheap):
    Gaia EB table / ECL class, DR3 VIM, duplicated_source, RUWE / IPD flags.
    Neighbour level (a 10" cone each, ~8 s per query) for the ``max_vet``
    highest-scoring catalogue survivors: the misassigned-transit flux match
    and crowding.  A catalogue survivor that was not cone-checked is
    ``SURVIVES_CATALOG_ONLY`` and says so."""
    vc = conf.get("vet") or {}
    rep: dict = {"stage": "vet", **_provenance()}
    p = out / "events_AB.csv"
    if not p.exists():
        rep.update(verdict="NO_EVENTS_TO_VET")
        _write(out / "vet.json", rep)
        return rep
    ab = pd.read_csv(p)
    a = ab[(ab["tier"] == "A") & ~ab["epoch_cluster"].astype(bool)].copy() if len(ab) else ab
    a = a.sort_values("score", ascending=False)
    ids = list(dict.fromkeys(a["source_id"].astype(np.int64).tolist()))[: int(vc.get("max_catalog", 60000))]
    a = a[a["source_id"].isin(ids)]
    rep["n_sources_vetted"] = len(ids)
    if not ids:
        rep.update(verdict="NO_TIER_A_EVENTS")
        pd.DataFrame().to_csv(out / "vetted.csv", index=False)
        _write(out / "vet.json", rep)
        return rep
    degraded = []
    try:
        gs = acq.gaia_source_rows(ids, tap=tap)
    except Exception as exc:  # noqa: BLE001
        gs = pd.DataFrame({"source_id": ids})
        degraded.append(f"gaia_source:{exc!r}"[:200])
    try:
        vari = acq.vari_membership(ids, tap=tap)
    except Exception as exc:  # noqa: BLE001
        vari = pd.DataFrame()
        degraded.append(f"vari:{exc!r}"[:200])
    try:
        vim = acq.vim_membership(ids, tap=tap)
    except Exception as exc:  # noqa: BLE001
        vim = set()
        degraded.append(f"vim:{exc!r}"[:200])
    level1 = vet_rules(a, gs, vari, {}, vc, vim=vim)
    alive = level1[level1["vet_class"] != "KILLED"].sort_values("score", ascending=False)
    cone_ids = list(dict.fromkeys(alive["source_id"].astype(np.int64).tolist()))[: int(vc.get("max_vet", 400))]
    cones: dict[int, pd.DataFrame] = {}
    n_cone_fail = 0
    gsi = gs.set_index("source_id") if "source_id" in gs else pd.DataFrame()
    for sid in cone_ids:
        if sid not in gsi.index or not np.isfinite(gsi.loc[sid].get("ra", np.nan)):
            continue
        r = gsi.loc[sid]
        try:
            cones[int(sid)] = acq.gaia_cone(float(r["ra"]), float(r["dec"]),
                                            float(vc.get("cone_arcsec", 10.0)), tap=tap)
        except Exception:  # noqa: BLE001
            n_cone_fail += 1
    if n_cone_fail:
        degraded.append(f"cones_failed:{n_cone_fail}/{len(cone_ids)}")
    v = vet_rules(a, gs, vari, cones, vc, vim=vim)
    checked = v["source_id"].isin(list(cones))
    v["cone_checked"] = checked
    v.loc[~checked & (v["vet_class"] != "KILLED"), "vet_class"] = "SURVIVES_CATALOG_ONLY"
    v.to_csv(out / "vetted.csv", index=False, float_format="%.6g")

    def best(s):
        for c in ("SURVIVES", "SURVIVES_FLAGGED", "SURVIVES_CATALOG_ONLY"):
            if (s == c).any():
                return c
        return "KILLED"

    per_src = v.groupby("source_id")["vet_class"].agg(best)
    kr: dict[str, int] = {}
    for x in v.drop_duplicates("source_id")["kill_reasons"].fillna(""):
        for n in str(x).split(";"):
            if n:
                kr[n] = kr.get(n, 0) + 1
    rep.update(classes={str(k): int(c) for k, c in per_src.value_counts().items()},
               kill_reasons_by_source=kr, degraded=degraded, n_cone_checked=int(len(cones)),
               survivors=v[v["vet_class"].isin(["SURVIVES", "SURVIVES_FLAGGED"])]
               .sort_values("score", ascending=False)
               .head(60)[[c for c in ("source_id", "vet_class", "flags", "kind", "t_peak", "delta_g", "delta_bp",
                                      "delta_rp", "ratio_bp_rp", "ratio_err", "n_transits_episode",
                                      "score", "phot_g_mean_mag", "bp_rp", "ruwe", "best_class_name",
                                      "rms_out_of_episode_g", "n_dip_episodes", "period_class")
                          if c in v.columns]].to_dict("records"))
    n_surv = int(per_src.isin(["SURVIVES", "SURVIVES_FLAGGED"]).sum())
    rep["verdict"] = (f"DEGRADED ({'; '.join(degraded)}); " if degraded else "") + \
        (f"{int((per_src != 'KILLED').sum())} of {len(ids)} tier-A sources survive the catalogue vet; "
         f"{n_surv} of {len(cones)} cone-checked survive the neighbour vet")
    _write(out / "vet.json", rep)
    print(f"[parallax4] vet: {rep['verdict']}")
    return rep


def stage_deepvet(conf: dict, out: Path, *, http=acq.http_get, tap=acq.gaia_tap,
                  simbad=None, vsx=None, max_n: int = 60) -> dict:
    """Trace each Gaia-vet survivor to a mechanism (``deepvet.classify_fate``)."""
    from . import deepvet as DV

    simbad = simbad or DV.simbad_type
    vsx = vsx or DV.vsx_type
    gcfg = G.GreyConfig.from_dict(conf.get("grey"))
    rep: dict = {"stage": "deepvet", **_provenance()}
    vt = out / "vetted.csv"
    if not vt.exists() or os.path.getsize(vt) < 2:
        rep.update(verdict="NO_SURVIVORS_TO_DEEPVET")
        _write(out / "deepvet.json", rep)
        return rep
    v = pd.read_csv(vt)
    v = v[v["vet_class"].isin(["SURVIVES", "SURVIVES_FLAGGED"])].sort_values("score", ascending=False)
    top = v.drop_duplicates("source_id").head(int(max_n))
    ids = top["source_id"].astype(np.int64).tolist()
    rep["n_targets"] = len(ids)
    try:
        lcs = acq.datalink_products(ids, http=http)
    except Exception as exc:  # noqa: BLE001
        lcs = {}
        rep["datalink_error"] = repr(exc)[:300]
    rows, lightcurves = [], {}
    for _, r in top.iterrows():
        sid = int(r["source_id"])
        ra, dec = float(r.get("ra", np.nan)), float(r.get("dec", np.nan))
        rec = {k: r.get(k) for k in ("source_id", "vet_class", "flags", "kind", "t_peak", "delta_g",
                                     "delta_bp", "delta_rp", "ratio_bp_rp", "ratio_err",
                                     "n_transits_episode", "score", "phot_g_mean_mag", "bp_rp",
                                     "ruwe", "min_n_obs_g", "rms_out_of_episode_g", "ra", "dec")
               if k in r}
        try:
            from astropy import units as u
            from astropy.coordinates import SkyCoord

            gal = SkyCoord(ra * u.deg, dec * u.deg).galactic
            rec.update(l_deg=float(gal.l.deg), b_deg=float(gal.b.deg))
        except Exception:  # noqa: BLE001
            pass
        if sid in lcs:
            try:
                lc = E.photometry_from_long(lcs[sid], source_id=sid)
                ev, _ = G.detect_source(lc, gcfg)
                hit = [e for e in ev if abs(e["t_peak"] - float(r["t_peak"])) < 0.5]
                rec["reproduced"] = bool(hit) and G.tier(hit[0]) == "A"
                w = lc[(lc["t"] > float(r["t_peak"]) - 3) & (lc["t"] < float(r["t_peak"]) + 3)]
                lightcurves[str(sid)] = w[["t", "f_g", "e_g", "f_bp", "e_bp", "f_rp", "e_rp", "n_obs_g",
                                           "bad_g", "bad_bp", "bad_rp"]].round(6).to_dict("records")
            except Exception as exc:  # noqa: BLE001
                rec["refetch_error"] = repr(exc)[:200]
        else:
            rec["reproduced"] = None
        if np.isfinite(ra):
            rec.update(simbad(ra, dec))
            rec.update(vsx(ra, dec))
            try:
                cone = acq.gaia_cone(ra, dec, 30.0, tap=tap)
                rec["n_gaia_30as"] = int(len(cone)) - 1
            except Exception:  # noqa: BLE001
                rec["n_gaia_30as"] = None
        fate, fl = DV.classify_fate(rec)
        rec["fate"], rec["fate_flags"] = fate, ";".join(fl)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "deepvet.csv", index=False, float_format="%.6g")
    _write(out / "deepvet_lightcurves.json", lightcurves)
    fates = df["fate"].map(lambda x: str(x).split("(")[0]).value_counts() if len(df) else pd.Series(dtype=int)
    rep.update(fates={str(k): int(c) for k, c in fates.items()},
               unexplained=df[df["fate"] == "UNEXPLAINED"].to_dict("records") if len(df) else [])
    rep["verdict"] = f"{int((df['fate'] == 'UNEXPLAINED').sum()) if len(df) else 0} of {len(df)} deep-vetted UNEXPLAINED"
    _write(out / "deepvet.json", rep)
    print(f"[parallax4] deepvet: {rep['verdict']}; fates={rep['fates']}")
    return rep


# ---------------------------------------------------------------------------
# watchlist + dr4
# ---------------------------------------------------------------------------
def stage_watchlist(conf: dict, out: Path, *, root: Path | str = ".") -> dict:
    df, st = W.build_watchlist(root)
    u = W.unique_watchlist(df)
    df.to_csv(out / "dr4_watchlist_rows.csv.gz", index=False, compression="gzip")
    u.to_csv(out / "dr4_watchlist.csv", index=False)
    rep = {"stage": "watchlist", **_provenance(), **st,
           "note": "every Gaia DR3 id a channel shortlisted, flagged, vetted or named; the DR4 "
                   "photocentre test runs on all of them on release day, priority 1 first"}
    _write(out / "watchlist.json", rep)
    print(f"[parallax4] watchlist: {st['n_unique']} unique source_ids")
    return rep


def stage_dr4(conf: dict, out: Path, *, http=acq.http_get, tap=acq.gaia_tap) -> dict:
    """Inert until DR4: returns DR4_NOT_RELEASED cleanly.  When DR4 exists it
    fetches epoch astrometry + photometry by DataLink for the watchlist (and
    PARALLAX4's own survivors) and runs the photocentre test on each."""
    dc = conf.get("dr4") or {}
    rep: dict = {"stage": "dr4", **_provenance()}
    status = acq.probe_dr4(tap=tap, http=http)
    rep["dr4_probe"] = status
    _write(out / "dr4_status.json", {**_provenance(), **status})
    if status.get("verdict") != "DR4_AVAILABLE":
        rep["verdict"] = status.get("verdict", "ARCHIVE_UNREACHABLE")
        _write(out / "dr4.json", rep)
        print(f"[parallax4] dr4: {rep['verdict']}")
        return rep
    # controls FIRST: the sources whose answer is known before release
    ctrl = pd.read_csv(out / "dr4_controls.csv") if (out / "dr4_controls.csv").exists() else pd.DataFrame()
    targets = ctrl["source_id"].astype(np.int64).tolist() if len(ctrl) else []
    vt = out / "vetted.csv"
    if vt.exists() and os.path.getsize(vt) > 1:
        v = pd.read_csv(vt)
        if "source_id" in v:
            targets += v["source_id"].astype(np.int64).tolist()
    wl = out / "dr4_watchlist.csv"
    if wl.exists():
        targets += pd.read_csv(wl)["source_id"].astype(np.int64).tolist()
    targets = list(dict.fromkeys(targets))[: int(dc.get("max_targets", 200000))]
    pcfg = P.PhotocentreConfig.from_dict(conf.get("photocentre"))
    rows, schema_errors = [], []
    types = list(dc.get("datalink_types", ["EPOCH_ASTROMETRY", "EPOCH_PHOTOMETRY"]))
    for i in range(0, len(targets), 200):
        chunk = targets[i:i + 200]
        try:
            ast = acq.datalink_products(chunk, retrieval_type=types[0], release=dc.get("release", "Gaia DR4"), http=http)
            pho = acq.datalink_products(chunk, retrieval_type=types[1], release=dc.get("release", "Gaia DR4"), http=http)
        except Exception as exc:  # noqa: BLE001
            schema_errors.append({"chunk": i, "error": repr(exc)[:300]})
            continue
        try:
            pos = acq.gaia_source_rows(chunk, tap=tap).set_index("source_id")
        except Exception:  # noqa: BLE001
            pos = pd.DataFrame()
        for sid in chunk:
            if sid not in ast or sid not in pho:
                rows.append({"source_id": sid, "verdict": "NOT_IN_DR4_EPOCH_PRODUCTS"})
                continue
            try:
                tr = E.collapse_transits(E.astrometry_ccd_frame(ast[sid]))
                ph = E.photometry_from_long(pho[sid], source_id=sid)
                j = P.prepare_joint(E.join_photometry_astrometry(ph, tr))
                r = P.fit_photocentre(j, cfg=pcfg)
                if r["verdict"] == "BLEND_UNRESOLVED" and sid in pos.index:
                    # only now is a neighbour table worth a query: is the
                    # vector pointing at a catalogued star?
                    pr = pos.loc[sid]
                    cone = acq.gaia_cone(float(pr["ra"]), float(pr["dec"]), 5.0, tap=tap)
                    nb = P.neighbour_offsets(float(pr["ra"]), float(pr["dec"]),
                                             cone[cone["source_id"] != sid])
                    r = P.fit_photocentre(j, cfg=pcfg, neighbours=nb,
                                          target_flux_g=float(pr.get("phot_g_mean_flux", np.nan)))
                rows.append({"source_id": sid, **{k: v for k, v in r.items()
                                                  if k not in ("sectors", "neighbours")}})
            except E.SchemaError as exc:
                schema_errors.append({"source_id": sid, "missing": exc.resolution.missing})
                rows.append({"source_id": sid, "verdict": "DR4_SCHEMA_MISMATCH"})
            except Exception as exc:  # noqa: BLE001
                rows.append({"source_id": sid, "verdict": "FIT_ERROR", "reasons": [repr(exc)[:200]]})
    df = pd.DataFrame(rows)
    gate = C.dr4_control_gate(df, ctrl)
    if len(df):
        df["controls_gate"] = gate["gate"]
    df.to_csv(out / "dr4_photocentre.csv", index=False)
    rep.update(n_targets=len(targets), verdicts={str(k): int(v) for k, v in df["verdict"].value_counts().items()}
               if len(df) else {}, schema_errors=schema_errors[:50], controls=gate)
    if not len(df):
        rep["verdict"] = "DR4_AVAILABLE_NO_PRODUCTS"
    elif gate["gate"] != "PASS":
        rep["verdict"] = f"DR4_CONTROLS_{gate['gate']}: photocentre verdicts NOT issued as results"
    else:
        rep["verdict"] = "DR4_PHOTOCENTRE_RUN"
    _write(out / "dr4.json", rep)
    print(f"[parallax4] dr4: {rep['verdict']}")
    return rep


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def stage_summary(conf: dict, out: Path) -> dict:
    probe, ctrl, red, vet = (_read(out / f) for f in ("probe.json", "controls.json", "reduce.json", "vet.json"))
    dr4, wl = _read(out / "dr4_status.json"), _read(out / "watchlist.json")
    rep: dict = {"channel": "parallax4", **_provenance()}
    counts = red.get("counts") or {}
    funnel = {
        "cdn_files_listed": (probe.get("cdn") or {}).get("n_files"),
        "files_done": red.get("n_files_done"), "files_planned": red.get("n_files_planned"),
        "sources_read": counts.get("n_sources"), "sources_searched": counts.get("n_searched"),
        "transits_searched": counts.get("n_transits_ok_g"),
        "qualifying_episodes": counts.get("n_events"),
        "tier_AB_episodes": red.get("n_AB"), "tier_A": red.get("n_tier_A"),
        "tier_A_after_epoch_veto": red.get("n_tier_A_after_epoch"),
        "tier_A_sources_after_epoch_veto": red.get("n_sources_tier_A_after_epoch"),
        "vetted_sources": vet.get("n_sources_vetted"), "vet_classes": vet.get("classes"),
    }
    rep["funnel"] = funnel
    rep["controls"] = {"photometric": ctrl.get("gate_photometric"), "astrometric": ctrl.get("gate_astrometric"),
                       "P_photometric": {k: (ctrl.get("P_photometric") or {}).get(k) for k in
                                         ("recovery_tier_A", "dust_called_grey", "n_grey_above_floor",
                                          "n_dust_recovered", "failures")},
                       "A2_prerelease": {k: (ctrl.get("A2_prerelease") or {}).get(k) for k in
                                         ("gate", "n_parallax_ok", "n_sources", "join_ok")},
                       "A3_varstrometry": {k: (ctrl.get("A3_varstrometry") or {}).get(k)
                                           for k in ("gate", "classes")}}
    rep["dr4"] = dr4.get("verdict")
    rep["watchlist_unique"] = wl.get("n_unique")
    # self-consistency
    checks = []
    seq = [funnel.get(k) for k in ("sources_read", "sources_searched")]
    if all(isinstance(x, (int, float)) for x in seq):
        checks.append({"check": "searched<=read", "ok": seq[1] <= seq[0]})
    ab, a = funnel.get("tier_AB_episodes"), funnel.get("tier_A")
    if isinstance(ab, int) and isinstance(a, int):
        checks.append({"check": "tierA<=tierAB", "ok": a <= ab})
    qa = funnel.get("qualifying_episodes")
    if isinstance(qa, int) and isinstance(ab, int):
        checks.append({"check": "tierAB<=qualifying", "ok": ab <= qa})
    rep["self_consistency"] = {"ok": all(c["ok"] for c in checks), "checks": checks}
    if ctrl.get("gate_photometric") != "PASS":
        verdict = "CONTROLS_NOT_PASSED" if ctrl else "NO_CONTROLS_RUN"
    elif not red or red.get("verdict") in ("NO_SHARD_OUTPUTS", "REFUSED_CONTROLS_NOT_PASSED"):
        verdict = "NO_DATA_REACHED"
    else:
        n_surv = sum(v for k, v in (vet.get("classes") or {}).items()
                     if k in ("SURVIVES", "SURVIVES_FLAGGED"))
        verdict = (f"GREY_EVENTS_SURVIVING_DR3_VET: {n_surv} sources await the DR4 photocentre test"
                   if n_surv else "NO_TIER_A_SURVIVOR")
        if red.get("degraded"):
            verdict = f"DEGRADED ({'; '.join(red['degraded'])}); {verdict}"
    rep["verdict"] = verdict
    rep["note"] = ("A tier-A survivor is a grey, colour-constrained, multi-transit, non-periodic "
                   "episode on a source Gaia does not classify as an eclipsing binary, not flagged "
                   "duplicated, with no neighbour whose flux reproduces the episode.  It is NOT a "
                   "detection until DR4's per-transit photocentre shows the variation is on the "
                   "target (ON_TARGET) -- the test this channel exists to run.")
    _write(out / "summary.json", rep)
    print(f"[parallax4] summary: {verdict}")
    return rep


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def run(stage: str = "all", *, out_dir=None, shard: int = 0, n_shards: int = 1, max_files: int = 0,
        conf: dict | None = None, force: bool = False) -> dict:
    conf = conf if conf is not None else load_config()
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out)
        elif s == "controls":
            rep = stage_controls(conf, out)
        elif s == "sweep":
            rep = stage_sweep(conf, out, shard=shard, n_shards=n_shards, max_files=max_files, force=force)
        elif s == "reduce":
            rep = stage_reduce(conf, out, n_shards_expected=n_shards if n_shards > 1 else None)
        elif s == "vet":
            rep = stage_vet(conf, out)
        elif s == "watchlist":
            rep = stage_watchlist(conf, out)
        elif s == "dr4":
            rep = stage_dr4(conf, out)
        elif s == "summary":
            rep = stage_summary(conf, out)
        elif s == "deepvet":
            rep = stage_deepvet(conf, out)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES + ('all',)}")
    return rep


def add_arguments(p) -> None:
    p.add_argument("--stage", default="all", help="|".join(STAGES) + "|all, or a comma list")
    p.add_argument("--shard", default="0/1", help="i/n (sweep)")
    p.add_argument("--shards", type=int, default=0, help="planned shard count (reduce)")
    p.add_argument("--max-files", type=int, default=0, help="cap on CDN files per sweep shard")
    p.add_argument("--out-dir", default="", help="results directory (default results/parallax4)")
    p.add_argument("--config", default="", help="alternative config yaml")
    p.add_argument("--force", action="store_true", help="sweep even if controls did not pass")


def run_from_args(a, _cfg=None) -> int:
    shard, n = parse_shard(a.shard)
    n_shards = a.shards or n
    conf = load_config(a.config or None)
    rep = run(a.stage, out_dir=a.out_dir or None, shard=shard, n_shards=n_shards,
              max_files=a.max_files, conf=conf, force=a.force)
    if isinstance(rep, dict) and rep.get("verdict"):
        print(f"[parallax4] verdict: {rep['verdict']}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti parallax4",
                                description="PARALLAX4: Gaia DR4 intake -- grey per-transit anomalies "
                                            "with no photocentre shift")
    add_arguments(p)
    return run_from_args(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

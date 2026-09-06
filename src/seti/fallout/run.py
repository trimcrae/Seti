"""FALLOUT stage orchestration; writes ``results/fallout/``.

Stages
------
``probe``     HEAD/GET every registered bulk-download route; no download.
``acquire``   pull GALAH DR4 (optionally APOGEE DR17) through
              ``seti.tailings.acquire`` in one broad box; discover the extra
              columns; checkpoint ``stars_<survey>.parquet`` and write
              ``acquisition.json``.
``screen``    per survey and per sample (dwarf primary, giant secondary):
              peer residuals, the per-element error floors, the
              five-hypothesis fit for every star, the raw-space fit, the no-Ba
              fit, leave-one-out above the prefilter, and the La/CN
              diagnostic. Checkpoints ``scores_*.parquet`` and
              ``vectors_*.parquet``.
``assess``    population-level calibration: the whole-sample and
              shuffled-element nulls, the working threshold, the vetoes at
              that threshold, the testable-conditioned sensitivity curve, then
              ``summary.json`` and ``REPORT.md``.
``vet``       re-apply every veto to every catalogue-level survivor from the
              committed ``candidates_*.csv`` (offline-capable: sigmas are
              rebuilt from the recorded error floors when the vectors are not
              on disk), write ``vet.json`` with per-star reasons and refresh
              the summary verdict from the *vetted* survivor count.
``all``       the chain, ending in ``vet``.

Entry points: :func:`fallout_run` (library), :func:`main` (``python -m
seti.fallout.run``), :func:`register` (adds the ``fallout`` sub-parser to a
CLI; ``src/seti/cli.py`` is not edited here).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from . import acquire as ACQ
from . import pattern as P
from . import yields as Y

CHANNEL = "fallout"
STAGES: tuple[str, ...] = ("probe", "acquire", "screen", "assess", "vet", "all")

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_DEGRADED = "DEGRADED_SOURCE"
VERDICT_NO_PATTERN = "NO_FISSION_PATTERN"
VERDICT_CANDIDATES = "FISSION_PATTERN_CANDIDATES_PENDING_VET"

_KEEP_COLS = ("star_id", "ra", "dec", "teff", "logg", "fe_h", "snr", "flag_sp", "age",
              "binary_flag", "ruwe", "vsini", "chi2_sp", "mass", "C", "N", "survey")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_block(cfg: Config) -> dict:
    """``config/fallout.yaml`` -> the ``fallout`` block (empty if absent)."""
    import yaml

    p = Path(cfg.root) / "config" / f"{CHANNEL}.yaml"
    if not p.exists():
        return {}
    doc = yaml.safe_load(p.read_text()) or {}
    return dict(doc.get(CHANNEL) or {})


def _out_dir(cfg: Config, out_dir: str | Path | None = None) -> Path:
    d = Path(out_dir) if out_dir is not None else Path(cfg.root) / "results" / CHANNEL
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pattern_config(block: dict) -> P.PatternConfig:
    b = dict(block.get("pattern") or {})
    known = set(P.PatternConfig.__dataclass_fields__)
    kw = {k: v for k, v in b.items() if k in known}
    for key in ("elements", "core_elements", "heavy_peak_elements"):
        if key in kw:
            kw[key] = tuple(kw[key])
    # PyYAML reads "1.0e6" as a string (it wants "1.0e+6"); coerce every
    # numeric field so a config typo cannot turn into a TypeError mid-run.
    types = {f.name: f.type for f in P.PatternConfig.__dataclass_fields__.values()}
    for k, v in list(kw.items()):
        t = str(types.get(k, ""))
        if t == "float":
            kw[k] = float(v)
        elif t == "int":
            kw[k] = int(v)
    return P.PatternConfig(**kw)


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(block: dict, out_dir: Path, surveys: list[str], *, route_probe_fn=None) -> dict:
    rep = {"generated_utc": _now_utc(), "surveys": ACQ.probe_routes(
        block, surveys, route_probe_fn=route_probe_fn)}
    _json(out_dir / "probe.json", rep)
    for sv, r in rep["surveys"].items():
        print(f"[fallout] probe {sv}: {r['n_live']} live / {r['n_eligible']} eligible of "
              f"{r['n_routes']} routes")
        for rr in r["routes"]:
            print(f"    HTTP {rr.get('status')!s:>4} len={rr.get('content_length')} "
                  f"{rr.get('url')} [{rr.get('why')}]")
    return rep


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def stage_acquire(block: dict, out_dir: Path, surveys: list[str], *, max_rows: int | None,
                  cache_dir=None, inject: dict | None = None) -> dict:
    from ..tailings.acquire import write_checkpoint

    inject = inject or {}
    rep = {"generated_utc": _now_utc(), "surveys": []}
    for sv in surveys:
        pull = ACQ.fetch_survey(sv, block=block, max_rows=max_rows, cache_dir=cache_dir,
                                **{k: v for k, v in inject.items()
                                   if k in ("route_probe_fn", "read_fn", "probe_fn", "query_fn")})
        prov = pull.provenance()
        rep["surveys"].append(prov)
        for line in pull.log:
            print(f"[fallout] {line}")
        if pull.n_rows:
            write_checkpoint(pull.table, out_dir / f"stars_{sv.lower()}.parquet")
    _json(out_dir / "acquisition.json", rep)
    return rep


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def screen_sample(stars: pd.DataFrame, *, survey: str, sample: str, block: dict,
                  out_dir: Path) -> dict:
    """Score one (survey, sample): peer residuals, error floors, fits, LOO. Checkpoints."""
    from ..tailings.acquire import write_checkpoint

    cfg = _pattern_config(block)
    peer = dict(block.get("peer") or {})
    loo_prefilter = float((block.get("report") or {}).get("loo_prefilter_lr", 4.0))

    elements = [e for e in cfg.elements if e in stars.columns
                and pd.to_numeric(stars[e], errors="coerce").notna().sum() >= int(peer.get("min_rows", 200))]
    n = int(len(stars))
    if len(elements) < cfg.min_elements or n < int(peer.get("min_rows", 200)):
        return {"survey": survey, "sample": sample, "n_stars": n, "n_elements": len(elements),
                "elements": elements, "verdict": (
                    f"INSUFFICIENT: {len(elements)} usable pattern elements "
                    f"(need {cfg.min_elements}) or {n} stars (need {peer.get('min_rows', 200)})")}

    T = P.build_templates(elements, horizon_yr=cfg.horizon_yr)
    elements = list(T.elements)

    resid, scatter, notes = P.peer_residuals(
        stars, elements, degree=int(peer.get("degree", 2)), clip=float(peer.get("clip_sigma", 4.0)),
        n_iter=int(peer.get("clip_iterations", 4)), min_rows=int(peer.get("min_rows", 200)))
    # The error model: quoted errors floored at this sample's measured peer
    # scatter, per element (the first real run showed quoted errors 2-4x low).
    floors = P.error_floors(stars, elements, scatter, err_prefix="e_", cfg=cfg)
    floor_map = {el: d["floor_dex"] for el, d in floors.items()}

    frame = stars.copy()
    for el in elements:
        frame[f"peer_{el}"] = resid[el].to_numpy()
    obs, sig, flagged, info = P.assemble_vectors(frame, elements, value_prefix="peer_",
                                                 err_prefix="e_", flag_prefix="f_", cfg=cfg,
                                                 fallback_sigma=scatter, sigma_floor=floor_map)
    raw = P.raw_vectors(stars, elements)
    for el in elements:
        frame[f"raw_{el}"] = raw[el].to_numpy()
    obs_raw, sig_raw, _, _ = P.assemble_vectors(frame, elements, value_prefix="raw_",
                                                err_prefix="e_", flag_prefix="f_", cfg=cfg,
                                                fallback_sigma=scatter, sigma_floor=floor_map)

    fit = P.fit_patterns(obs, sig, T, cfg)
    fit["fission_lr_raw"] = P.fission_lr_only(obs_raw, sig_raw, T, cfg)
    fit["lr_noba"] = P.lr_without(obs, sig, T, cfg, drop=(cfg.ba_element,))
    core_idx = [T.index(e) for e in cfg.core_elements if e in T.elements]
    fit["flagged_core"] = flagged[:, core_idx].any(axis=1) if core_idx else False
    fit["n_flagged"] = flagged.sum(axis=1)
    testable = P.testable_mask(obs, T, cfg)
    fit["testable"] = testable

    pre = fit["fission_lr"].to_numpy() >= loo_prefilter
    loo_cols = [f"lr_without_{el}" for el in elements] + ["lr_loo_min", "lr_loo_driver"]
    for c in loo_cols:
        fit[c] = "" if c == "lr_loo_driver" else np.nan
    if pre.any():
        loo = P.leave_one_out(obs[pre], sig[pre], T, cfg)
        for c in loo_cols:
            fit.loc[pre, c] = loo[c].to_numpy()
    fit["lr_loo_driver"] = fit["lr_loo_driver"].astype(str)

    keep = [c for c in _KEEP_COLS if c in stars.columns]
    scores = pd.concat([stars[keep].reset_index(drop=True),
                        stars[[e for e in elements + ["Li"] if e in stars.columns]].reset_index(drop=True),
                        fit.reset_index(drop=True)], axis=1)
    scores["sample"] = sample
    for k, el in enumerate(elements):
        scores[f"peer_{el}"] = resid[el].to_numpy()
        scores[f"sig_{el}"] = sig[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            scores[f"z_{el}"] = np.where(np.isfinite(obs[:, k]), obs[:, k] / sig[:, k], np.nan)

    la_diag = P.la_diagnostics(scores, cfg=cfg, min_rows=int(peer.get("min_rows", 200)))

    tag = f"{survey.lower()}_{sample}"
    write_checkpoint(scores, out_dir / f"scores_{tag}.parquet")
    vec = pd.DataFrame({**{f"v_{el}": obs[:, k] for k, el in enumerate(elements)},
                        **{f"s_{el}": sig[:, k] for k, el in enumerate(elements)},
                        "testable": testable})
    write_checkpoint(vec, out_dir / f"vectors_{tag}.parquet")
    _json(out_dir / "templates.json", {
        "generated_utc": _now_utc(), "templates": T.to_dict(1.0),
        "discriminant_ratios_at_a1": {
            "fission": P.discriminant_ratios(dict(zip(T.elements, np.log10(1 + T.F), strict=True))),
            "s": P.discriminant_ratios(dict(zip(T.elements, np.log10(1 + T.S), strict=True))),
            "r": P.discriminant_ratios(dict(zip(T.elements, np.log10(1 + T.R), strict=True)))},
        "table": Y.template_table(elements, amplitude=1.0, horizon_yr=cfg.horizon_yr)})

    counts = fit["classification"].value_counts().to_dict()
    return {
        "survey": survey, "sample": sample, "n_stars": n, "n_elements": len(elements),
        "elements": elements, "class_counts": {k: int(v) for k, v in counts.items()},
        "n_lr_prefilter": int(pre.sum()),
        "n_above_lr_min": int((fit["fission_lr"] >= cfg.lr_min).sum()),
        "n_above_lr_min_raw": int((fit["fission_lr_raw"] >= cfg.lr_min).sum()),
        "n_testable": int(testable.sum()),
        "testable_fraction": float(testable.mean()) if n else float("nan"),
        "n_unexplained": int((fit["classification"] == P.UNEXPLAINED).sum()),
        "peer_scatter_dex": {k: round(float(v), 4) for k, v in scatter.items()},
        "error_model": {"mode": cfg.error_floor_mode, "systematic_floor_dex": cfg.systematic_floor_dex,
                        "per_element": floors},
        "la_diagnostics": la_diag,
        "peer_notes": notes, "vector_columns": info, "verdict": None,
    }


def stage_screen(block: dict, out_dir: Path, surveys: list[str]) -> list[dict]:
    out: list[dict] = []
    for sv in surveys:
        p = out_dir / f"stars_{sv.lower()}.parquet"
        if not p.exists():
            out.append({"survey": sv, "sample": None, "n_stars": 0,
                        "verdict": f"{VERDICT_NO_DATA}: no checkpoint for {sv}"})
            continue
        stars = pd.read_parquet(p)
        samples = ACQ.split_samples(stars, block.get("samples") or {})
        for name, tab in samples.items():
            print(f"[fallout] {sv}/{name}: {len(tab)} stars in the box")
            out.append(screen_sample(tab, survey=sv, sample=name, block=block, out_dir=out_dir))
    return out


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
_SHOW_COLS = ["star_id", "teff", "logg", "fe_h", "snr", "n_measured", "fission_lr", "enrich_lr",
              "reduced_chi2_best", "lr_noba", "lr_loo_min", "lr_loo_driver", "n_heavy_coherent",
              "fission_lr_raw", "a_f", "classification", "natural_class", "first_veto", "veto_reasons"]


def _records(df: pd.DataFrame, elements: list[str], n: int) -> list[dict]:
    if df is None or not len(df):
        return []
    cols = [c for c in _SHOW_COLS + list(elements) + [f"peer_{e}" for e in elements] if c in df.columns]
    return df[cols].head(n).to_dict(orient="records")


def assess_sample(rec: dict, *, block: dict, out_dir: Path) -> dict:
    cfg = _pattern_config(block)
    nb = dict(block.get("null") or {})
    sb = dict(block.get("sensitivity") or {})
    tag = f"{rec['survey'].lower()}_{rec['sample']}"
    sp = out_dir / f"scores_{tag}.parquet"
    vp = out_dir / f"vectors_{tag}.parquet"
    if not sp.exists() or not vp.exists():
        return {**rec, "verdict": rec.get("verdict") or "INSUFFICIENT: no screen checkpoint"}
    scores = pd.read_parquet(sp)
    vec = pd.read_parquet(vp)
    elements = list(rec["elements"])
    T = P.build_templates(elements, horizon_yr=cfg.horizon_yr)
    obs = np.column_stack([vec[f"v_{el}"].to_numpy(dtype=float) for el in T.elements])
    sig = np.column_stack([vec[f"s_{el}"].to_numpy(dtype=float) for el in T.elements])
    testable = vec["testable"].to_numpy(dtype=bool) if "testable" in vec.columns \
        else P.testable_mask(obs, T, cfg)

    rng = np.random.default_rng(int(nb.get("seed", 20260906)))
    shuffled = P.shuffled_null(obs, sig, T, cfg, n_perm=int(nb.get("n_perm", 3)),
                               max_rows=int(nb.get("max_rows", 20000)), rng=rng)
    sample_null = P.sample_null(scores["fission_lr"].to_numpy())
    thr, thr_why = P.derive_threshold(cfg, shuffled)

    lr = scores["fission_lr"].to_numpy(dtype=float)
    en = scores["enrich_lr"].to_numpy(dtype=float)
    pass_lr = np.isfinite(lr) & (lr >= thr) & np.isfinite(en) & (en >= cfg.enrich_min)
    raw_lr = scores["fission_lr_raw"].to_numpy(dtype=float)
    raw_pass = np.isfinite(raw_lr) & (raw_lr >= thr)
    n_removed_by_peer = int((raw_pass & ~pass_lr).sum())
    la_suspect = bool((rec.get("la_diagnostics") or {}).get("la_cn_suspect", True))

    cand = scores.loc[pass_lr].copy()
    if len(cand):
        cand = P.apply_vetoes(cand, cfg=cfg, lr_threshold=thr,
                              flagged_core=cand["flagged_core"].to_numpy(dtype=bool),
                              la_cn_suspect=la_suspect)
        cand = cand.sort_values("fission_lr", ascending=False, ignore_index=True)
        cand.to_csv(out_dir / f"candidates_{tag}.csv", index=False)
        counters = P.veto_counters(cand)
    else:
        counters = {name: 0 for name in P.VETOES}
        counters["first_veto"] = {name: 0 for name in P.VETOES}
        counters["n_pass"] = 0
    counters["teff_peer_residual_raw_only"] = n_removed_by_peer

    sens = P.sensitivity_curve(obs, sig, T, cfg, lr_threshold=thr,
                               amplitudes=tuple(sb.get("amplitudes", (0.5, 1, 2, 3, 5, 10, 20))),
                               n_inject=int(sb.get("n_inject", 1500)), rng=rng, testable=testable)
    n_show = int((block.get("report") or {}).get("max_candidates", 50))
    survivors = cand[cand["vet_pass"]] if len(cand) else cand
    unexplained = cand[cand["veto_unexplained_by_all_templates"]] if len(cand) else cand
    return {
        **rec,
        "threshold": {"lr_used": thr, "why": thr_why, "lr_min_config": cfg.lr_min,
                      "enrich_min": cfg.enrich_min, "max_reduced_chi2": cfg.max_reduced_chi2},
        "null_sample": sample_null,
        "null_shuffled": {k: v for k, v in shuffled.items() if k != "lr"},
        "n_pass_lr": int(pass_lr.sum()),
        "n_raw_pass_lr": int(raw_pass.sum()),
        "vetoes": counters,
        "n_survivors": int(len(survivors)),
        "n_unexplained_above_threshold": int(len(unexplained)),
        "sensitivity": sens,
        "survivors": _records(survivors, elements, n_show),
        "unexplained_top": _records(unexplained, elements, 10),
        "vetoed_top": _records(cand[~cand["vet_pass"]] if len(cand) else cand, elements, 10),
        "verdict": None,
    }


def _overall_verdict(acq: dict | None, per_sample: list[dict], vet: dict | None = None
                     ) -> tuple[str, str]:
    surveys = (acq or {}).get("surveys") or []
    reached = [s for s in surveys if int(s.get("n_rows", 0)) > 0]
    if not reached:
        vs = sorted({str(s.get("verdict")) for s in surveys}) or ["no acquisition record"]
        return VERDICT_NO_DATA, (
            f"{VERDICT_NO_DATA}: no survey catalogue delivered rows ({', '.join(vs)}). "
            "This is an archive-access statement, not a limit on the signature.")
    scored = [s for s in per_sample if s.get("threshold")]
    degraded = [s for s in reached if s.get("degraded")]
    prefix = ""
    if degraded:
        names = "; ".join(f"{s['survey']}->{s.get('source_used')}: {s.get('degradation')}"
                          for s in degraded)
        prefix = f"{VERDICT_DEGRADED} ({names}); "
    if not scored:
        why = "; ".join(str(s.get("verdict")) for s in per_sample) or "no sample was screened"
        return VERDICT_DEGRADED, (
            f"{VERDICT_DEGRADED}: rows were retrieved but no sample could be scored ({why}). "
            "This is a coverage statement, not a limit on the signature.")

    def n_surv(s: dict) -> int:
        key = f"{s.get('survey')}/{s.get('sample')}"
        if vet and key in (vet.get("samples") or {}):
            return int(vet["samples"][key].get("n_survivors_vetted", 0))
        return int(s.get("n_survivors", 0))

    prim = [s for s in scored if str(s.get("sample")) == "dwarf"]
    n_prim = sum(n_surv(s) for s in prim)
    n_sec = sum(n_surv(s) for s in scored if s not in prim)
    n_unex = sum(int(s.get("n_unexplained_above_threshold", 0)) for s in scored)
    code = VERDICT_CANDIDATES if (n_prim or n_sec) else VERDICT_NO_PATTERN
    vetted = " (after the vet stage)" if vet else ""
    if code == VERDICT_NO_PATTERN:
        text = (prefix + f"{VERDICT_NO_PATTERN}: no star in any sample kept the fission-only "
                f"preference through the vetoes at the null-calibrated threshold{vetted}; "
                f"{n_unex} above-threshold stars are UNEXPLAINED_BY_ALL_TEMPLATES and are listed "
                "separately. Per CLAUDE.md this is a reason to change the question (a second survey, "
                "the APOGEE Ce/Nd panel, differential co-natal pairs), not a result to write up.")
    else:
        text = (prefix + f"{VERDICT_CANDIDATES}: {n_prim} cool-dwarf and {n_sec} giant "
                f"(lower-weight) catalogue-level survivors{vetted}; {n_unex} above-threshold stars "
                "are UNEXPLAINED_BY_ALL_TEMPLATES and listed separately. None is a detection until "
                "the n-capture lines are re-measured from the raw spectra against Teff-matched peers "
                "and the pattern is confirmed element by element.")
    if degraded and prefix:
        return VERDICT_DEGRADED, text
    return code, text


def _load_screen_records(block: dict, out_dir: Path, surveys: list[str]) -> list[dict]:
    screen = []
    if (out_dir / "screen.json").exists():
        return json.loads((out_dir / "screen.json").read_text()).get("samples") or []
    for sv in surveys:
        for name in (block.get("samples") or {}):
            sp = out_dir / f"scores_{sv.lower()}_{name}.parquet"
            if sp.exists():
                sc = pd.read_parquet(sp)
                els = [c[5:] for c in sc.columns if c.startswith("peer_")]
                screen.append({"survey": sv, "sample": name, "n_stars": int(len(sc)),
                               "n_elements": len(els), "elements": els})
    return screen


def _build_summary(block: dict, out_dir: Path, per_sample: list[dict], acq: dict | None,
                   stage: str, vet: dict | None = None) -> dict:
    cfg = _pattern_config(block)
    code, text = _overall_verdict(acq, per_sample, vet)
    templates = json.loads((out_dir / "templates.json").read_text()) \
        if (out_dir / "templates.json").exists() else {
            "templates": P.build_templates(cfg.elements, horizon_yr=cfg.horizon_yr).to_dict(1.0)}

    def key(s):
        return f"{s.get('survey')}/{s.get('sample')}"

    def n_vetted(s):
        if vet and key(s) in (vet.get("samples") or {}):
            return int(vet["samples"][key(s)].get("n_survivors_vetted", 0))
        return None

    funnel = {
        "n_rows_acquired": {s["survey"]: int(s.get("n_rows", 0)) for s in ((acq or {}).get("surveys") or [])},
        "per_sample": [{"survey": s.get("survey"), "sample": s.get("sample"),
                        "n_stars": int(s.get("n_stars", 0)), "n_elements": int(s.get("n_elements", 0)),
                        "n_testable": int(s.get("n_testable", 0)),
                        "testable_fraction": s.get("testable_fraction"),
                        "n_above_lr_min": int(s.get("n_above_lr_min", 0)),
                        "n_pass_lr": int(s.get("n_pass_lr", 0)),
                        "n_unexplained_above_threshold": int(s.get("n_unexplained_above_threshold", 0)),
                        "n_survivors": int(s.get("n_survivors", 0)),
                        "n_survivors_vetted": n_vetted(s)} for s in per_sample],
        "n_survivors_dwarf": sum(int(s.get("n_survivors", 0)) for s in per_sample if s.get("sample") == "dwarf"),
        "n_survivors_giant": sum(int(s.get("n_survivors", 0)) for s in per_sample if s.get("sample") == "giant"),
    }
    if vet:
        funnel["n_survivors_vetted_dwarf"] = sum(n_vetted(s) or 0 for s in per_sample if s.get("sample") == "dwarf")
        funnel["n_survivors_vetted_giant"] = sum(n_vetted(s) or 0 for s in per_sample if s.get("sample") == "giant")
    summary = {
        "channel": CHANNEL,
        "generated_utc": _now_utc(),
        "stage": stage,
        "verdict_code": code,
        "verdict": text,
        "funnel": funnel,
        "vetoes": {key(s): s.get("vetoes") for s in per_sample if s.get("vetoes")},
        "error_model": {key(s): s.get("error_model") for s in per_sample if s.get("error_model")},
        "la_diagnostics": {key(s): s.get("la_diagnostics") for s in per_sample if s.get("la_diagnostics")},
        "acquisition": acq,
        "columns_found": {s["survey"]: s.get("columns_found") for s in ((acq or {}).get("surveys") or [])},
        "templates": templates,
        "thresholds": {key(s): s.get("threshold") for s in per_sample if s.get("threshold")},
        "vet": vet,
        "per_sample": per_sample,
    }
    return summary


def stage_assess(block: dict, out_dir: Path, surveys: list[str], screen: list[dict] | None,
                 acq: dict | None, stage: str) -> dict:
    if screen is None:
        screen = _load_screen_records(block, out_dir, surveys)
    per_sample = [assess_sample(r, block=block, out_dir=out_dir) if r.get("sample") else r
                  for r in screen]
    summary = _build_summary(block, out_dir, per_sample, acq, stage)
    _json(out_dir / "summary.json", summary)
    write_report(out_dir, summary)
    print("[fallout] " + json.dumps({"verdict": summary["verdict_code"],
                                     **{k: v for k, v in summary["funnel"].items() if k != "per_sample"}}))
    return summary


# ---------------------------------------------------------------------------
# vet
# ---------------------------------------------------------------------------
def vet_candidates(cand: pd.DataFrame, *, elements: list[str], cfg: P.PatternConfig,
                   lr_threshold: float, floors: dict[str, float] | None, la_cn_suspect: bool,
                   refit: bool = True) -> tuple[pd.DataFrame, dict]:
    """Re-vet a candidate table, rebuilding sigmas from the recorded floors if needed.

    When the table has no ``sig_<El>`` columns (the first real run's CSVs), the
    sigma of every element is ``sqrt(floor^2 + systematic^2)`` -- the quoted
    error is not in the CSV, and the floor is >= it by construction, so this is
    the inflated error model to within the systematic term. With ``refit`` the
    fission ratio, the no-Ba ratio and the leave-one-out are recomputed from
    the ``peer_<El>`` residuals under that error model; the original values are
    kept as ``*_screen``.
    """
    out = cand.copy()
    notes: dict = {"sigma_source": "sig_ columns", "refit": False}
    T = P.build_templates(elements, horizon_yr=cfg.horizon_yr)
    els = list(T.elements)
    have_sig = all(f"sig_{el}" in out.columns for el in els)
    have_peer = all(f"peer_{el}" in out.columns for el in els)
    if not have_peer:
        notes["sigma_source"] = "none: no peer_ columns; vetoes applied to the recorded fit only"
        vet = P.apply_vetoes(out, cfg=cfg, lr_threshold=lr_threshold, la_cn_suspect=la_cn_suspect)
        return vet, notes
    obs = np.column_stack([pd.to_numeric(out[f"peer_{el}"], errors="coerce").to_numpy(dtype=float)
                           for el in els])
    if have_sig:
        sig = np.column_stack([pd.to_numeric(out[f"sig_{el}"], errors="coerce").to_numpy(dtype=float)
                               for el in els])
    else:
        fl = np.array([float((floors or {}).get(el, cfg.error_default_dex) or cfg.error_default_dex)
                       for el in els])
        sig = np.tile(np.sqrt(fl ** 2 + cfg.systematic_floor_dex ** 2), (len(out), 1))
        notes["sigma_source"] = "rebuilt from recorded per-element floors (quoted errors not in CSV)"
        for k, el in enumerate(els):
            out[f"sig_{el}"] = sig[:, k]
    if refit and len(out):
        notes["refit"] = True
        for c in ("fission_lr", "enrich_lr", "lr_noba", "lr_loo_min", "lr_loo_driver",
                  "reduced_chi2_best", "chi2_f", "chi2_natural", "a_f"):
            if c in out.columns:
                out[f"{c}_screen"] = out[c]
        fit = P.fit_patterns(obs, sig, T, cfg)
        for c in ("n_measured", "chi2_null", "chi2_f", "a_f", "chi2_natural", "chi2_best",
                  "reduced_chi2_best", "reduced_chi2_f", "fission_lr", "enrich_lr", "classification"):
            out[c] = fit[c].to_numpy()
        out["lr_noba"] = P.lr_without(obs, sig, T, cfg, drop=(cfg.ba_element,))
        loo = P.leave_one_out(obs, sig, T, cfg)
        for c in loo.columns:
            out[c] = loo[c].to_numpy()
        for k, el in enumerate(els):
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"z_{el}"] = np.where(np.isfinite(obs[:, k]), obs[:, k] / sig[:, k], np.nan)
    vet = P.apply_vetoes(out, cfg=cfg, lr_threshold=lr_threshold, la_cn_suspect=la_cn_suspect)
    # the threshold itself is a veto when the refit moved a star below it
    below = ~(np.isfinite(vet["fission_lr"].to_numpy(dtype=float))
              & (vet["fission_lr"].to_numpy(dtype=float) >= lr_threshold))
    vet["below_threshold_after_refit"] = below
    vet["vet_pass"] = vet["vet_pass"].to_numpy(dtype=bool) & ~below
    vet.loc[below & (vet["first_veto"] == ""), "first_veto"] = "below_threshold_after_refit"
    return vet, notes


def stage_vet(block: dict, out_dir: Path, surveys: list[str], acq: dict | None, stage: str) -> dict:
    cfg = _pattern_config(block)
    summary_path = out_dir / "summary.json"
    prior = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    per_sample = list(prior.get("per_sample") or [])
    acq = acq or prior.get("acquisition")
    vet = {"generated_utc": _now_utc(), "samples": {}, "stars": []}
    for sv in surveys:
        for name in (block.get("samples") or {}):
            tag = f"{sv.lower()}_{name}"
            key = f"{sv}/{name}"
            cp = out_dir / f"candidates_{tag}.csv"
            rec = next((s for s in per_sample if s.get("survey") == sv and s.get("sample") == name), {})
            if not cp.exists():
                vet["samples"][key] = {"n_candidates": 0, "n_survivors_vetted": 0,
                                       "note": "no candidates file"}
                continue
            cand = pd.read_csv(cp)
            if "sample" not in cand.columns:
                cand["sample"] = name
            elements = list(rec.get("elements") or [c[5:] for c in cand.columns if c.startswith("peer_")])
            thr = float((rec.get("threshold") or {}).get("lr_used", cfg.lr_min))
            em = ((rec.get("error_model") or {}).get("per_element") or {})
            floors = {el: d.get("floor_dex") for el, d in em.items()} if em else \
                dict(rec.get("peer_scatter_dex") or {})
            la_diag = rec.get("la_diagnostics") or {}
            la_suspect = bool(la_diag.get("la_cn_suspect", True))
            vetted, notes = vet_candidates(cand, elements=elements, cfg=cfg, lr_threshold=thr,
                                           floors=floors, la_cn_suspect=la_suspect)
            vetted.to_csv(out_dir / f"vetted_{tag}.csv", index=False)
            counters = P.veto_counters(vetted)
            counters["below_threshold_after_refit"] = int(vetted["below_threshold_after_refit"].sum())
            surv = vetted[vetted["vet_pass"]]
            vet["samples"][key] = {
                "n_candidates": int(len(vetted)),
                "n_survivors_screen": int(rec.get("n_survivors", 0)),
                "n_survivors_vetted": int(len(surv)),
                "lr_threshold": thr,
                "sigma": notes,
                "la_cn_suspect": la_suspect,
                "la_reason": la_diag.get("reason"),
                "vetoes": counters,
                "survivors": _records(surv, elements, 50),
            }
            for _, row in vetted.iterrows():
                prev_pass = bool(row.get("vet_pass_screen", False)) if "vet_pass_screen" in vetted.columns else None
                vet["stars"].append({
                    "star_id": row.get("star_id"), "sample": name, "survey": sv,
                    "fission_lr_screen": row.get("fission_lr_screen"),
                    "fission_lr": row.get("fission_lr"),
                    "reduced_chi2_best": row.get("reduced_chi2_best"),
                    "n_heavy_coherent": row.get("n_heavy_coherent"),
                    "lr_loo_min": row.get("lr_loo_min"), "lr_loo_driver": row.get("lr_loo_driver"),
                    "first_veto": row.get("first_veto"), "veto_reasons": row.get("veto_reasons"),
                    "vet_pass": bool(row.get("vet_pass")), "passed_at_screen": prev_pass,
                })
            if rec:
                rec["n_survivors_vetted"] = int(len(surv))
                rec["vetoes_vet"] = counters
    _json(out_dir / "vet.json", vet)
    if per_sample:
        summary = _build_summary(block, out_dir, per_sample, acq, stage, vet=vet)
        for k in ("templates", "columns_found"):
            if prior.get(k) and not summary.get(k):
                summary[k] = prior[k]
        _json(summary_path, summary)
        write_report(out_dir, summary)
        print("[fallout] vet: " + json.dumps({k: v.get("n_survivors_vetted") for k, v in vet["samples"].items()}))
        return summary
    print("[fallout] vet: no summary.json to refresh; vet.json written")
    return vet


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(out_dir: Path, summary: dict) -> Path:
    L: list[str] = []
    L.append("# FALLOUT — fission-product abundance pattern search\n")
    L.append(f"Generated {summary.get('generated_utc')} (UTC).\n")
    L.append(f"**Verdict.** {summary['verdict']}\n")
    L.append("## The discriminant\n")
    L.append("Fission product, once its short-lived members have decayed, is a fixed *shape* in "
             "n-capture space: light peak Zr-Mo-Ru, heavy peak Ba-La-Ce-Pr-Nd, a ~1000x valley "
             "(Ag-Sb), almost nothing past Sm, and no Pb. Against solar that is "
             "`[Nd/Ba] >> 0, [Ce/Ba] > 0, [La/Ba] > 0, [Mo/Zr] > 0, [Ru/Zr] > 0, [Eu/Nd] < 0`. "
             "The s-process gives `[Nd/Ba] < 0` and `[Mo/Zr] < 0`; the r-process gives "
             "`[Eu/Nd] > 0`. Each star is fitted as solar + {s, r, s+r, fission}; the statistic is "
             "the fission-only log-likelihood ratio against the best natural mixture, and a star "
             "is only a candidate if its best model actually fits (reduced chi2 below the cap), "
             "if removing any one element leaves the preference standing, and if at least two "
             "heavy-peak elements are individually up.\n")
    t = (summary.get("templates") or {}).get("templates") or {}
    if t.get("elements"):
        L.append("### Template vectors at amplitude 1 (Nd doubled), dex\n")
        L.append("| element | fission | s | r |")
        L.append("|---|---|---|---|")
        for el, f, s, r in zip(t["elements"], t["fission_dex"], t["s_dex"], t["r_dex"], strict=True):
            L.append(f"| {el} | {f:+.3f} | {s:+.3f} | {r:+.3f} |")
        L.append("")
    acq = summary.get("acquisition") or {}
    L.append("## Acquisition\n")
    for s in acq.get("surveys") or []:
        L.append(f"- **{s.get('survey')}**: {s.get('verdict')} via {s.get('route')} from "
                 f"`{s.get('source_used')}` — {s.get('n_rows', 0):,} rows, "
                 f"{s.get('n_elements', 0)} elements")
        if s.get("degradation"):
            L.append(f"  - degradation: {s['degradation']}")
        cf = s.get("columns_found") or {}
        if cf:
            L.append(f"  - elements found: {', '.join(cf.get('elements') or [])}")
            L.append(f"  - extras found: {cf.get('extras')}; absent: {cf.get('extras_absent')}")
        for line in s.get("log") or []:
            L.append(f"  - {line}")
    if not (acq.get("surveys") or []):
        L.append("- no acquisition record")
    L.append("")
    vet = summary.get("vet") or {}
    for s in summary.get("per_sample") or []:
        key = f"{s.get('survey')}/{s.get('sample')}"
        L.append(f"## {key}\n")
        if s.get("verdict") and not s.get("threshold"):
            L.append(f"{s['verdict']}\n")
            continue
        L.append(f"- stars in the box: **{s.get('n_stars', 0):,}**; pattern elements: "
                 f"**{s.get('n_elements', 0)}** ({', '.join(s.get('elements') or [])})")
        tf = s.get("testable_fraction")
        if tf is not None:
            L.append(f"- testable (≥{5} elements and ≥2 of La/Ce/Nd measured): "
                     f"**{s.get('n_testable', 0):,}** = {float(tf):.1%} of the sample")
        cc = s.get("class_counts") or {}
        if cc:
            L.append("- classification: " + ", ".join(f"{k} {v:,}" for k, v in cc.items()))
        em = (s.get("error_model") or {}).get("per_element") or {}
        if em:
            L.append("- error model: quoted error floored at the measured peer scatter — "
                     + ", ".join(f"{el} {d['floor_dex']:.2f}" + (f" (×{d['inflation']})" if d.get("inflation") else "")
                                 for el, d in em.items()))
        ld = s.get("la_diagnostics") or {}
        if ld:
            L.append(f"- La diagnostic: suspect={ld.get('la_cn_suspect')} — {ld.get('reason')}; "
                     f"correlations {ld.get('correlations')}")
        th = s.get("threshold") or {}
        L.append(f"- threshold: ln LR ≥ **{th.get('lr_used', float('nan')):.2f}** ({th.get('why')}); "
                 f"enrichment ln LR ≥ {th.get('enrich_min')}; reduced chi2 ≤ {th.get('max_reduced_chi2')}")
        ns = (s.get("null_shuffled") or {}).get("quantiles") or {}
        nn = (s.get("null_sample") or {}).get("quantiles") or {}
        if ns or nn:
            L.append("- null quantiles (ln LR): shuffled " + ", ".join(f"{k} {v:.2f}" for k, v in ns.items())
                     + " | sample " + ", ".join(f"{k} {v:.2f}" for k, v in nn.items()))
        L.append(f"- above threshold: **{s.get('n_pass_lr', 0)}** (raw-space: {s.get('n_raw_pass_lr', 0)}); "
                 f"unexplained by all templates: **{s.get('n_unexplained_above_threshold', 0)}**; "
                 f"survivors after vetoes: **{s.get('n_survivors', 0)}**"
                 + (f"; after the vet stage: **{s.get('n_survivors_vetted')}**"
                    if s.get("n_survivors_vetted") is not None else ""))
        vt = s.get("vetoes") or {}
        if vt:
            L.append("- vetoes (independent counts): " + ", ".join(
                f"{k} {v}" for k, v in vt.items() if k not in ("first_veto", "n_pass")))
        sens = s.get("sensitivity") or []
        if sens:
            L.append("\n### Sensitivity (injected fission pattern into real vectors)\n")
            L.append("| a_f | Δ[Nd/H] dex | LR pass (testable) | LR + LOO pass (testable) | LR pass (all) | LR + LOO pass (all) |")
            L.append("|---|---|---|---|---|---|")
            for r in sens:
                L.append(f"| {r['a_f']:g} | {r['nd_dex']:+.2f} | "
                         f"{r.get('frac_lr_pass_testable', float('nan')):.2f} | "
                         f"{r.get('frac_lr_and_loo_pass_testable', float('nan')):.2f} | "
                         f"{r.get('frac_lr_pass_all', r.get('frac_lr_pass', float('nan'))):.2f} | "
                         f"{r.get('frac_lr_and_loo_pass_all', r.get('frac_lr_and_loo_pass', float('nan'))):.2f} |")
            if sens[0].get("testable_fraction") is not None:
                L.append(f"\nTestable fraction {sens[0]['testable_fraction']:.1%} "
                         f"({sens[0].get('n_testable', 0):,} injected).")
        for title, key2 in (("Survivors (catalogue-level; pending re-measurement)", "survivors"),
                            ("Unexplained by all templates (listed, never candidates)", "unexplained_top")):
            rows = s.get(key2) or []
            if rows:
                L.append(f"\n### {title}\n")
                L.append("| star | Teff | log g | [Fe/H] | n_el | ln LR | red. chi2 | no-Ba | LOO min (driver) | heavy≥2σ | raw | a_f |")
                L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
                for c in rows[:25]:
                    L.append(f"| {c.get('star_id')} | {float(c.get('teff', np.nan)):.0f} | "
                             f"{float(c.get('logg', np.nan)):.2f} | {float(c.get('fe_h', np.nan)):+.2f} | "
                             f"{int(c.get('n_measured', 0))} | {float(c.get('fission_lr', np.nan)):.1f} | "
                             f"{float(c.get('reduced_chi2_best', np.nan)):.1f} | "
                             f"{float(c.get('lr_noba', np.nan)):.1f} | {float(c.get('lr_loo_min', np.nan)):.1f} "
                             f"({c.get('lr_loo_driver')}) | {c.get('n_heavy_coherent', '-')} | "
                             f"{float(c.get('fission_lr_raw', np.nan)):.1f} | "
                             f"{float(c.get('a_f', np.nan)):.2f} |")
        vs = (vet.get("samples") or {}).get(key)
        if vs:
            L.append("\n### Vet stage\n")
            L.append(f"- candidates re-vetted: {vs.get('n_candidates')}; survivors at screen: "
                     f"{vs.get('n_survivors_screen')}; **after vet: {vs.get('n_survivors_vetted')}**")
            L.append(f"- sigma: {(vs.get('sigma') or {}).get('sigma_source')}; refit: "
                     f"{(vs.get('sigma') or {}).get('refit')}; La suspect: {vs.get('la_cn_suspect')} "
                     f"({vs.get('la_reason')})")
            vv = vs.get("vetoes") or {}
            L.append("- vetoes: " + ", ".join(f"{k} {v}" for k, v in vv.items()
                                              if k not in ("first_veto", "n_pass")))
        L.append("")
    L.append("## What a survivor still has to pass\n")
    L.append("Nothing here is a detection. A survivor is a *target*: the Ba II, La II, Ce II, Nd II "
             "and Eu II lines must be re-measured from the raw HERMES spectrum against Teff-matched "
             "peers, the pattern must hold element by element, and the star must be checked for an "
             "unresolved companion. Until then the correct description is 'an n-capture vector the "
             "s+r mixture does not fit'.\n")
    L.append("## No-null rule (CLAUDE.md)\n")
    L.append("An empty survivor list at this threshold is a statement about GALAH DR4's element panel "
             "and precision, not a publishable null. The escalation path is the APOGEE Ce/Nd panel as a "
             "second survey, the co-natal differential channel, and high-resolution re-measurement of "
             "the strongest ambiguous and unexplained stars.\n")
    p = out_dir / "REPORT.md"
    p.write_text("\n".join(L) + "\n")
    return p


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def fallout_run(cfg: Config | None = None, *, stage: str = "all", surveys: str | None = None,
                max_rows: int | None = None, cache_dir=None, out_dir=None, block: dict | None = None,
                inject: dict | None = None) -> dict:
    """Run the FALLOUT funnel. Returns the dict written to disk for that stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    cfg = cfg or load_config()
    block = block if block is not None else load_block(cfg)
    out = _out_dir(cfg, out_dir)
    sv_list = [s.strip().upper() for s in (surveys or ",".join(block.get("surveys") or ["GALAH"])).split(",")
               if s.strip()]
    inject = inject or {}

    if stage == "probe":
        return stage_probe(block, out, sv_list, route_probe_fn=inject.get("route_probe_fn"))

    acq = None
    if stage in ("acquire", "all"):
        acq = stage_acquire(block, out, sv_list, max_rows=max_rows, cache_dir=cache_dir, inject=inject)
        if stage == "acquire":
            return acq
    if acq is None and (out / "acquisition.json").exists():
        acq = json.loads((out / "acquisition.json").read_text())

    if stage == "vet":
        return stage_vet(block, out, sv_list, acq, stage)

    screen = None
    if stage in ("screen", "all"):
        screen = stage_screen(block, out, sv_list)
        _json(out / "screen.json", {"generated_utc": _now_utc(), "samples": screen})
        if stage == "screen":
            return {"stage": stage, "samples": screen}
    if screen is None and (out / "screen.json").exists():
        screen = json.loads((out / "screen.json").read_text()).get("samples")

    summary = stage_assess(block, out, sv_list, screen, acq, stage)
    if stage == "all":
        summary = stage_vet(block, out, sv_list, acq, stage)
    return summary


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--surveys", default=None,
                   help="comma-separated surveys (default: config/fallout.yaml surveys)")
    p.add_argument("--max-rows", type=int, default=None, help="row cap for the catalogue pull")
    p.add_argument("--cache-dir", default=None, help="where the bulk FITS is streamed to")
    p.add_argument("--out-dir", default=None, help="override results/fallout/")


def _cmd_fallout(args, cfg) -> dict:
    return fallout_run(cfg, stage=args.stage, surveys=args.surveys, max_rows=args.max_rows,
                       cache_dir=args.cache_dir, out_dir=args.out_dir)


def register(sub) -> argparse.ArgumentParser:
    """Add the ``fallout`` sub-command to an argparse sub-parser collection.

    The parser's default ``func`` follows the repository convention
    ``func(args, cfg)``.
    """
    p = sub.add_parser("fallout",
                       help="runner: fission-product abundance PATTERN in GALAH cool dwarfs — "
                            "[Nd/Ba], [Ce/Ba], [Mo/Zr] up with [Eu/Nd] down, fitted against "
                            "s/r/s+r mixtures with a leave-one-out pattern test (docs/fallout.md)")
    _add_arguments(p)
    p.set_defaults(func=_cmd_fallout)
    return p


def main(argv: list[str] | None = None) -> int:
    """``python -m seti.fallout.run --stage all``."""
    p = argparse.ArgumentParser(prog="seti fallout", description=__doc__.split("\n\n")[0])
    _add_arguments(p)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    _cmd_fallout(args, load_config())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["STAGES", "assess_sample", "fallout_run", "load_block", "main", "register",
           "screen_sample", "stage_acquire", "stage_assess", "stage_probe", "stage_screen",
           "stage_vet", "vet_candidates", "write_report"]

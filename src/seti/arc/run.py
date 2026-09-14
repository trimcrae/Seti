"""Stage orchestration for ARC.  Writes ``results/arc/``.

Stages
------
``probe``    schema discovery on every configured flare and star catalogue:
             the real table, its real columns, the resolved roles, the row
             count.  Writes ``probe.json``.  Nothing is fetched.
``acquire``  pull the flare tables and the per-star context tables (chunked,
             retried, logged).  Writes ``data/<cat>_flares.parquet``,
             ``data/<mission>_<name>_stars.parquet``, ``acquire.json`` and
             ``acquisition_log.json``.
``screen``   per flare catalogue: join the per-star amplitude / Teff / radius /
             Prot, compute the spot ceiling and xi per star.  Writes
             ``xi_<cat>.csv`` and ``screen_<cat>.json``.
``assess``   merge every ``xi_*.csv`` into ``xi_table.csv``; for the
             shortlist (xi > 0) fetch Berger+2020 / TIC parameters and the
             Gaia DR3 cone context; recompute the ceiling with the better
             parameters; run the gauntlet.  Writes ``summary.json``,
             ``candidates.csv`` and ``candidates.json``.

Verdict vocabulary (``summary.json["verdict"]``)
------------------------------------------------
``NO_DATA_REACHED``             no flare catalogue produced usable rows
                                (``data_status`` keeps QUERY_FAILED apart
                                from QUERY_RETURNED_ZERO_ROWS)
``NO_CEILING_EXCESS``           stars were assessed; nothing reached interest
``CEILING_EXCESS_CANDIDATES``   >= 1 star at interest or candidate tier
``degraded`` lists every source that failed or veto that could not be
applied.  None of these is ever written up as a result; a null is a reason to
change the question (CLAUDE.md).

Entry points
------------
``arc_run(stage="all", out_dir=..., ...)`` and ``main(argv)`` so the module
runs as ``python -m seti.arc.run --stage probe``.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..metronome.run import normalise_time_system
from ..metronome.windows import guess_time_system
from .ceiling import (
    B_DEFAULT_G,
    DEFAULT_GEOMETRIC_FACTOR,
    energy_to_bolometric,
    normalise_amplitude,
    percentiles,
    star_ceiling,
)
from .vet import DEFAULT_VET, assign_tiers, funnel, rejection_counters

DEFAULTS: dict = {
    "physics": {"b_gauss": B_DEFAULT_G, "f_max": 1.0,
                "geometric_factor": DEFAULT_GEOMETRIC_FACTOR, "independent_gap_days": 0.5,
                "fallback_teff_k": 5777.0, "fallback_radius_rsun": 1.0,
                "band_fraction_kepler": 0.35, "amplitude_min_frac": 1e-4},
    "catalogues": {},
    "star_catalogues": {},
    "param_tables": {"kepler": ["J/AJ/159/280/table1", "J/AJ/159/280", "V/133/kic"],
                     "tess": ["IV/39/tic82", "IV/38/tic"]},
    "gaia": {"table": "I/355/gaiadr3", "vari_table": "I/358/vclassre",
             "cone_radius_arcsec": 12.0, "match_arcsec": 2.0},
    "vet": dict(DEFAULT_VET),
    "shortlist": {"max_stars": 3000, "include_watch": True},
    "acquire": {"chunk_rows": 50000, "max_rows": 0},
}

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NONE = "NO_CEILING_EXCESS"
VERDICT_CANDIDATES = "CEILING_EXCESS_CANDIDATES"
STAGES = ("probe", "acquire", "screen", "assess")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_update(base: dict, extra: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_arc_config(path: Path | None = None) -> dict:
    """``config/arc.yaml`` over :data:`DEFAULTS`; a missing file degrades."""
    try:
        import yaml
        if path is None:
            path = Path(__file__).resolve().parents[3] / "config" / "arc.yaml"
        if not Path(path).exists():
            return _deep_update(DEFAULTS, {})
        return _deep_update(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {})
    except Exception as exc:                              # noqa: BLE001
        print(f"[arc] config not loaded ({exc!r}); using defaults")
        return _deep_update(DEFAULTS, {})


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _enabled_catalogues(conf: dict, catalogues=None) -> dict:
    cats = {k: v for k, v in (conf.get("catalogues") or {}).items() if v.get("enabled", True)}
    if catalogues:
        want = {c.strip() for c in catalogues if c.strip()}
        cats = {k: v for k, v in cats.items() if k in want}
    return cats


def _star_specs(conf: dict, missions) -> list[tuple[str, dict]]:
    out = []
    for m in sorted(missions):
        for spec in (conf.get("star_catalogues") or {}).get(m, []) or []:
            out.append((m, spec))
    return out


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def stage_probe(conf: dict, out: Path, *, catalogues=None, query_fn=None, log=None,
                fetch_fn=None) -> dict:
    from .acquire import AcquisitionLog, discover_table, tap_query

    log = log or AcquisitionLog(prefix="arc/acquire")
    query_fn = query_fn or tap_query
    cats = _enabled_catalogues(conf, catalogues)
    found, stars = {}, {}
    for name, spec in cats.items():
        disc = discover_table(name, spec["preferred"], "flares", tuple(spec.get("keywords") or ()),
                              query_fn=query_fn, log=log, overrides=spec.get("columns"),
                              fetch_fn=fetch_fn)
        d = disc.as_dict()
        d.update({"mission": spec.get("mission"), "preferred": spec["preferred"],
                  "energy": spec.get("energy")})
        found[name] = d
    for mission, spec in _star_specs(conf, {s.get("mission") for s in cats.values()}):
        disc = discover_table(spec["name"], spec.get("preferred") or "", "stars",
                              tuple(spec.get("keywords") or ()), query_fn=query_fn, log=log,
                              overrides=spec.get("columns"), fetch_fn=fetch_fn)
        d = disc.as_dict()
        d["mission"] = mission
        stars[f"{mission}_{spec['name']}"] = d
    rep = {"stage": "probe", "generated_utc": _now(), "catalogues": found,
           "star_catalogues": stars,
           "n_usable": sum(1 for d in found.values() if d["status"] == "OK"),
           "n_star_tables_usable": sum(1 for d in stars.values() if d["status"] == "OK"),
           "acquisition": log.as_dict()}
    _write(out / "probe.json", rep)
    print(f"[arc] probe: {rep['n_usable']}/{len(found)} flare catalogues, "
          f"{rep['n_star_tables_usable']}/{len(stars)} star tables usable")
    return rep


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------
def stage_acquire(conf: dict, out: Path, *, catalogues=None, query_fn=None,
                  max_rows: int | None = None, log=None, fetch_fn=None) -> dict:
    from .acquire import AcquisitionLog, discover_table, fetch_table, tap_query

    log = log or AcquisitionLog(prefix="arc/acquire")
    query_fn = query_fn or tap_query
    acq = conf.get("acquire") or {}
    max_rows = int(acq.get("max_rows") or 0) if max_rows is None else int(max_rows)
    chunk = int(acq.get("chunk_rows", 50000))
    cats = _enabled_catalogues(conf, catalogues)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    per_cat = {}
    for name, spec in cats.items():
        disc = discover_table(name, spec["preferred"], "flares", tuple(spec.get("keywords") or ()),
                              query_fn=query_fn, log=log, overrides=spec.get("columns"),
                              fetch_fn=fetch_fn)
        # ``discovery_route`` is which SEED found the table (the preferred id or
        # the keyword search); ``route`` is which ACCESS route served its rows
        # (``tap`` / ``asu_tsv`` / ``astroquery``), which is what a dispatch
        # made during a TAPVizieR outage has to be able to show.
        rec = {"table": disc.table, "roles": disc.roles, "discovery_status": disc.status,
               "discovery_route": disc.route, "route": "none",
               "mission": spec.get("mission"), "n_rows_catalogue": disc.n_rows}
        if disc.table is None:
            rec.update({"status": disc.status, "n_flares": 0, "n_stars": 0})
            per_cat[name] = rec
            continue
        df = fetch_table(disc, query_fn=query_fn, log=log, chunk_rows=chunk,
                         max_rows=max_rows or None)
        rec["route"] = str(df.attrs.get("route", "none")) if len(df) else "none"
        if not len(df) or "star_id" not in df.columns:
            failed = any(s["stage"] == f"fetch_{name}" and s["status"] == "QUERY_FAILED"
                         for s in log.stages)
            rec.update({"status": "QUERY_FAILED" if failed else "QUERY_RETURNED_ZERO_ROWS",
                        "n_flares": 0, "n_stars": 0})
            per_cat[name] = rec
            continue
        df["mission"] = spec.get("mission")
        df["catalogue"] = name
        if "t_peak" not in df.columns and "t_start" in df.columns:
            df["t_peak"] = df["t_start"]
        if "t_peak" in df.columns:
            guess = guess_time_system(df["t_peak"].to_numpy(dtype=float), spec.get("mission", ""))
            df, native = normalise_time_system(df, guess, str(spec.get("mission", "")))
            rec.update({"time_system_guess": guess, "time_system_native": native})
        path = data / f"{name}_flares.parquet"
        df.to_parquet(path, index=False)
        rec.update({"status": "OK", "n_flares": int(len(df)),
                    "n_stars": int(df["star_id"].nunique()), "path": str(path)})
        per_cat[name] = rec

    star_recs = {}
    for mission, spec in _star_specs(conf, {s.get("mission") for s in cats.values()}):
        key = f"{mission}_{spec['name']}"
        disc = discover_table(spec["name"], spec.get("preferred") or "", "stars",
                              tuple(spec.get("keywords") or ()), query_fn=query_fn, log=log,
                              overrides=spec.get("columns"), fetch_fn=fetch_fn)
        rec = {"table": disc.table, "roles": disc.roles, "mission": mission,
               "discovery_status": disc.status, "discovery_route": disc.route, "route": "none",
               "amplitude_unit": spec.get("amplitude_unit", "auto")}
        if disc.table is None:
            rec.update({"status": disc.status, "n_rows": 0})
            star_recs[key] = rec
            continue
        df = fetch_table(disc, query_fn=query_fn, log=log, chunk_rows=chunk)
        rec["route"] = str(df.attrs.get("route", "none")) if len(df) else "none"
        if not len(df):
            failed = any(s["stage"] == f"fetch_{spec['name']}" and s["status"] == "QUERY_FAILED"
                         for s in log.stages)
            rec.update({"status": "QUERY_FAILED" if failed else "QUERY_RETURNED_ZERO_ROWS",
                        "n_rows": 0})
            star_recs[key] = rec
            continue
        df["source"] = spec["name"]
        path = data / f"{key}_stars.parquet"
        df.to_parquet(path, index=False)
        rec.update({"status": "OK", "n_rows": int(len(df)), "path": str(path)})
        star_recs[key] = rec

    rep = {"stage": "acquire", "generated_utc": _now(), "catalogues": per_cat,
           "star_catalogues": star_recs, "acquisition": log.as_dict()}
    _write(out / "acquire.json", rep)
    log.write(out / "acquisition_log.json")
    n_ok = sum(1 for r in per_cat.values() if r.get("status") == "OK")
    print(f"[arc] acquire: {n_ok}/{len(per_cat)} flare catalogues, "
          f"{sum(1 for r in star_recs.values() if r.get('status') == 'OK')}/{len(star_recs)} "
          f"star tables fetched")
    return rep


# ---------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------
def _first_finite(*vals) -> float:
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f) and f > 0:
            return f
    return float("nan")


def build_star_context(flares: pd.DataFrame, star_tables: list[tuple[str, pd.DataFrame, str]],
                       physics: dict) -> pd.DataFrame:
    """One row per star: the amplitude (MAX over sources, as a fraction), Prot,
    Teff, radius, logg with their sources, and the catalogue flag.

    ``star_tables`` is ``[(name, df, amplitude_unit), ...]`` in priority order
    for Teff / radius / logg / Prot.  The flare catalogue's own per-star
    columns (Okamoto, Shibayama and Tu carry them per flare row) come first.
    """
    sids = flares["star_id"].astype(str).unique()
    ctx = pd.DataFrame({"star_id": sids})
    # --- the flare table's own per-star columns (median over rows) -----------
    g = flares.groupby(flares["star_id"].astype(str))
    own: dict[str, pd.Series] = {}
    for col in ("rot_amplitude", "prot", "teff", "radius", "logg"):
        if col in flares.columns:
            own[col] = g[col].median()
    if "flag" in flares.columns:
        own["flag"] = g["flag"].agg(lambda s: ";".join(sorted({str(x) for x in s.dropna()
                                                                 if str(x).strip()})))
    amp_cols, amp_units = [], {}
    if "rot_amplitude" in own:
        vals, unit = normalise_amplitude(own["rot_amplitude"].to_numpy(dtype=float),
                                         physics.get("amplitude_unit_own", "auto"))
        ctx["amp_own"] = ctx["star_id"].map(pd.Series(vals, index=own["rot_amplitude"].index))
        amp_cols.append("amp_own")
        amp_units["own"] = unit
    for col in ("prot", "teff", "radius", "logg", "flag"):
        if col in own:
            ctx[f"{col}_own"] = ctx["star_id"].map(own[col])
    # --- the star tables, in priority order ------------------------------------
    for name, df, unit in star_tables:
        if df is None or not len(df) or "star_id" not in df.columns:
            continue
        d = df.copy()
        d["star_id"] = d["star_id"].astype(str)
        d = d.drop_duplicates("star_id", keep="first").set_index("star_id")
        if "rot_amplitude" in d.columns:
            vals, u = normalise_amplitude(d["rot_amplitude"].to_numpy(dtype=float), unit)
            ctx[f"amp_{name}"] = ctx["star_id"].map(pd.Series(vals, index=d.index))
            amp_cols.append(f"amp_{name}")
            amp_units[name] = u
        for col in ("prot", "teff", "radius", "logg", "flag"):
            if col in d.columns:
                ctx[f"{col}_{name}"] = ctx["star_id"].map(d[col])
    # --- resolve ---------------------------------------------------------------
    amin = float(physics.get("amplitude_min_frac", 1e-4))
    if amp_cols:
        vals = ctx[amp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        vals = np.where(np.isfinite(vals) & (vals >= amin), vals, -np.inf)
        has = np.isfinite(vals).any(axis=1) & (vals > -np.inf).any(axis=1)
        best = np.argmax(vals, axis=1)
        ctx["amplitude_frac"] = np.where(has, vals[np.arange(len(vals)), best], np.nan)
        ctx["amplitude_source"] = [amp_cols[i][4:] if h else None
                                   for i, h in zip(best, has, strict=True)]
        ctx["amplitude_unit"] = ctx["amplitude_source"].map(lambda s: amp_units.get(s))
        ctx["amplitude_unit_guessed"] = ctx["amplitude_source"].map(
            lambda s: bool(s) and _unit_was_guessed(s, star_tables, physics))
    else:
        ctx["amplitude_frac"] = np.nan
        ctx["amplitude_source"] = None
        ctx["amplitude_unit"] = None
        ctx["amplitude_unit_guessed"] = False
    order = ["own"] + [n for n, _, _ in star_tables]
    for col in ("prot", "teff", "radius", "logg"):
        val = pd.Series(np.nan, index=ctx.index, dtype=float)
        src = pd.Series([None] * len(ctx), index=ctx.index, dtype=object)
        for n in order:
            c = f"{col}_{n}"
            if c not in ctx.columns:
                continue
            v = pd.to_numeric(ctx[c], errors="coerce")
            take = val.isna() & v.notna() & (v > 0)
            val[take] = v[take]
            src[take] = n
        ctx[col] = val
        ctx[f"{col}_source"] = src
    flag_cols = [f"flag_{n}" for n in order if f"flag_{n}" in ctx.columns]
    # ``astype(str)`` leaves a float NaN as the float when the column is
    # already object-typed with real NaNs mixed in, and the join then raises
    # ``TypeError: sequence item 0: expected str instance, float found`` --
    # which is what killed run 34798271601 AFTER it had successfully assessed
    # 266 Okamoto stars.  Coerce every cell, and drop the null spellings.
    _nulls = {"nan", "none", "<na>", "nat", ""}
    ctx["catalogue_flag"] = (
        ctx[flag_cols]
        .apply(lambda col: col.map(lambda v: "" if v is None or (isinstance(v, float)
                                                                 and pd.isna(v))
                                   else str(v).strip()))
        .agg(lambda r: ";".join(x for x in r if str(x).strip().lower() not in _nulls), axis=1)
        if flag_cols else "")
    return ctx


def _unit_was_guessed(source: str, star_tables, physics) -> bool:
    if source == "own":
        return str(physics.get("amplitude_unit_own", "auto")).lower() == "auto"
    for name, _, unit in star_tables:
        if name == source:
            return str(unit or "auto").lower() == "auto"
    return True


def screen_catalogue(flares: pd.DataFrame, star_tables: list, name: str, mission: str,
                     conf: dict, *, energy_spec: dict | None = None) -> tuple[list[dict], dict]:
    """Per-star ceiling records for one flare catalogue."""
    phys = conf.get("physics") or {}
    espec = dict(energy_spec or {})
    fl = flares.copy()
    fl["star_id"] = fl["star_id"].astype(str)
    ctx = build_star_context(fl, star_tables, phys).set_index("star_id")
    # energies -> bolometric erg
    fallback_t, fallback_r = float(phys.get("fallback_teff_k", 5777.0)), \
        float(phys.get("fallback_radius_rsun", 1.0))
    ecol = next((c for c in ("energy", "energy_max", "equiv_duration") if c in fl.columns), None)
    if ecol is None:
        return [], {"stage": "screen", "catalogue": name, "mission": mission,
                    "status": "NO_ENERGY_COLUMN", "generated_utc": _now()}
    kind = str(espec.get("kind", "bolometric"))
    if ecol == "equiv_duration":
        kind = "equivalent_duration"
    teff_row = fl["star_id"].map(ctx["teff"]).fillna(fallback_t).to_numpy(dtype=float)
    rad_row = fl["star_id"].map(ctx["radius"]).fillna(fallback_r).to_numpy(dtype=float)
    e_bol, einfo = energy_to_bolometric(
        fl[ecol].to_numpy(dtype=float), kind=kind, factor=float(espec.get("factor", 1.0)),
        log10=espec.get("log10", "auto"), radius_rsun=rad_row, t_star_k=teff_row,
        band_fraction=float(phys.get("band_fraction_kepler", 0.35)))
    fl["e_bol"] = e_bol
    has_t = "t_peak" in fl.columns and fl["t_peak"].notna().any()
    records = []
    n_assessable = 0
    for sid, g in fl.groupby("star_id"):
        c = ctx.loc[sid]
        teff, rad = float(c["teff"]), float(c["radius"])
        assumed = not (np.isfinite(teff) and teff > 0 and np.isfinite(rad) and rad > 0)
        if not (np.isfinite(teff) and teff > 0):
            teff = fallback_t
        if not (np.isfinite(rad) and rad > 0):
            rad = fallback_r
        amp = float(c["amplitude_frac"]) if pd.notna(c["amplitude_frac"]) else float("nan")
        t = g["t_peak"].to_numpy(dtype=float) if has_t else None
        rec = star_ceiling(g["e_bol"].to_numpy(dtype=float), t, amp, rad, teff,
                           b_gauss=float(phys.get("b_gauss", B_DEFAULT_G)),
                           geometric_factor=float(phys.get("geometric_factor",
                                                           DEFAULT_GEOMETRIC_FACTOR)),
                           independent_gap_days=float(phys.get("independent_gap_days", 0.5)))
        if "sector" in g.columns and rec["flares_above"]:
            secs = g.dropna(subset=["t_peak"]).sort_values("t_peak") if has_t else g
            for fa in rec["flares_above"]:
                if fa["t_peak"] is not None and has_t:
                    m = (secs["t_peak"] - fa["t_peak"]).abs().idxmin()
                    fa["sector"] = _json_default(secs.loc[m, "sector"])
        rec.update({
            "record_key": f"{name}:{mission}:{sid}", "star_key": f"{mission}:{sid}",
            "star_id": sid, "catalogue": name, "mission": mission,
            "amplitude_source": c["amplitude_source"], "amplitude_unit": c["amplitude_unit"],
            "amplitude_unit_guessed": bool(c["amplitude_unit_guessed"]),
            "prot": float(c["prot"]) if pd.notna(c["prot"]) else float("nan"),
            "prot_source": c["prot_source"], "teff_source": c["teff_source"],
            "radius_source": c["radius_source"],
            "logg": float(c["logg"]) if pd.notna(c["logg"]) else float("nan"),
            "params_assumed": bool(assumed), "has_peak_times": bool(has_t),
            "catalogue_flag": str(c["catalogue_flag"] or ""),
            "energy_kind": einfo["kind"], "energy_log10_input": einfo["log10_input"],
            # the raw energies travel with stars near the bound so assess can
            # recompute with better parameters
            "energies_json": (json.dumps([float(x) for x in g["e_bol"]])
                              if rec["assessable"] and rec["xi_nominal_max"] > -1.0 else ""),
            "t_peaks_json": (json.dumps([float(x) if np.isfinite(x) else None for x in t])
                             if (t is not None and rec["assessable"]
                                 and rec["xi_nominal_max"] > -1.0) else ""),
        })
        rec["flares_above"] = json.dumps(rec["flares_above"])
        n_assessable += int(rec["assessable"])
        records.append(rec)
    xc = [r["xi_conservative_max"] for r in records if r["assessable"]]
    xn = [r["xi_nominal_max"] for r in records if r["assessable"]]
    rep = {"stage": "screen", "catalogue": name, "mission": mission, "generated_utc": _now(),
           "n_flares": int(len(fl)), "n_stars": int(len(records)),
           "n_stars_assessable": int(n_assessable),
           "n_stars_no_amplitude": int(sum(1 for r in records
                                           if not np.isfinite(r["amplitude_frac"]))),
           "n_stars_params_assumed": int(sum(1 for r in records if r["params_assumed"])),
           "n_xi_conservative_positive": int(sum(1 for x in xc if x > 0)),
           "n_xi_nominal_positive": int(sum(1 for x in xn if x > 0)),
           "xi_conservative": percentiles(xc), "xi_nominal": percentiles(xn),
           "energy": einfo, "has_peak_times": bool(has_t),
           "amplitude_sources": pd.Series([r["amplitude_source"] for r in records]).value_counts(
               dropna=False).rename(lambda k: str(k)).to_dict(),
           "status": "OK"}
    return records, rep


def _load_star_tables(out: Path, mission: str, conf: dict) -> list:
    tabs = []
    for spec in (conf.get("star_catalogues") or {}).get(mission, []) or []:
        p = out / "data" / f"{mission}_{spec['name']}_stars.parquet"
        if p.exists():
            try:
                tabs.append((spec["name"], pd.read_parquet(p), spec.get("amplitude_unit", "auto")))
            except Exception:                             # noqa: BLE001
                continue
    return tabs


def stage_screen(conf: dict, out: Path, *, catalogues=None,
                 flares_by_catalogue: dict | None = None,
                 star_tables_by_mission: dict | None = None) -> dict:
    cats = _enabled_catalogues(conf, catalogues)
    reports = {}
    for name, spec in cats.items():
        mission = str(spec.get("mission", "kepler"))
        if flares_by_catalogue is not None and name in flares_by_catalogue:
            fl = flares_by_catalogue[name]
        else:
            p = out / "data" / f"{name}_flares.parquet"
            if not p.exists():
                reports[name] = {"stage": "screen", "catalogue": name, "status": "NO_FLARES_FILE",
                                 "generated_utc": _now()}
                continue
            fl = pd.read_parquet(p)
        if not len(fl):
            reports[name] = {"stage": "screen", "catalogue": name, "status": "NO_FLARES",
                             "generated_utc": _now()}
            continue
        tabs = (star_tables_by_mission or {}).get(mission)
        if tabs is None:
            tabs = _load_star_tables(out, mission, conf)
        recs, rep = screen_catalogue(fl, tabs, name, mission, conf, energy_spec=spec.get("energy"))
        pd.DataFrame(recs).to_csv(out / f"xi_{name}.csv", index=False)
        _write(out / f"screen_{name}.json", rep)
        reports[name] = rep
        print(f"[arc] screen {name}: {rep.get('n_stars', 0)} stars, "
              f"{rep.get('n_stars_assessable', 0)} assessable, "
              f"{rep.get('n_xi_conservative_positive', 0)} above the conservative ceiling")
    return reports


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------
def _recompute(rec: dict, teff: float, rad: float, phys: dict) -> dict:
    """Re-run the ceiling for one record with better stellar parameters."""
    if not rec.get("energies_json"):
        return rec
    try:
        e = np.asarray(json.loads(rec["energies_json"]), dtype=float)
        t = (np.asarray([np.nan if x is None else x for x in json.loads(rec["t_peaks_json"])],
                        dtype=float) if rec.get("t_peaks_json") else None)
    except (ValueError, TypeError):
        return rec
    new = star_ceiling(e, t, float(rec.get("amplitude_frac", np.nan)), rad, teff,
                       b_gauss=float(phys.get("b_gauss", B_DEFAULT_G)),
                       geometric_factor=float(phys.get("geometric_factor",
                                                       DEFAULT_GEOMETRIC_FACTOR)),
                       independent_gap_days=float(phys.get("independent_gap_days", 0.5)))
    out = dict(rec)
    for k, v in new.items():
        out[k] = json.dumps(v) if k == "flares_above" else v
    out["params_assumed"] = False
    out["teff_source"] = out["radius_source"] = "assess_params"
    return out


def stage_assess(conf: dict, out: Path, *, offline: bool = False, query_fn=None, cone_fn=None,
                 records: list[dict] | None = None, log=None,
                 acquire_report: dict | None = None) -> dict:
    from .acquire import AcquisitionLog

    log = log or AcquisitionLog(prefix="arc/assess")
    phys = conf.get("physics") or {}
    vconf = conf.get("vet") or {}
    if records is None:
        frames = []
        for fp in sorted(glob.glob(str(out / "xi_*.csv"))):
            if Path(fp).name == "xi_table.csv":
                continue
            try:
                d = pd.read_csv(fp, dtype={"star_id": str, "star_key": str, "record_key": str,
                                           "flares_above": str, "energies_json": str,
                                           "t_peaks_json": str, "catalogue_flag": str})
            except Exception:                             # noqa: BLE001
                continue
            if len(d):
                frames.append(d)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df) and "record_key" in df:
            df = df.drop_duplicates("record_key")
        records = df.to_dict(orient="records") if len(df) else []
    for r in records:
        for k in ("energies_json", "t_peaks_json", "flares_above", "catalogue_flag"):
            if not isinstance(r.get(k), str):
                r[k] = "" if k != "flares_above" else "[]"
    screens = []
    for fp in sorted(glob.glob(str(out / "screen_*.json"))):
        try:
            screens.append(json.loads(Path(fp).read_text()))
        except Exception:                                 # noqa: BLE001
            continue
    if acquire_report is None and (out / "acquire.json").exists():
        try:
            acquire_report = json.loads((out / "acquire.json").read_text())
        except Exception:                                 # noqa: BLE001
            acquire_report = None
    acq_cats = (acquire_report or {}).get("catalogues") or {}
    statuses = {str(v.get("status")) for v in acq_cats.values()}

    # --- nothing arrived -------------------------------------------------------
    if not records:
        if acq_cats and statuses <= {"QUERY_RETURNED_ZERO_ROWS"}:
            data_status = "QUERY_RETURNED_ZERO_ROWS"
        elif acq_cats and "OK" in statuses:
            data_status = "FLARES_ARRIVED_BUT_NO_STAR_RECORD"
        else:
            data_status = "QUERY_FAILED"
        summary = {"verdict": VERDICT_NO_DATA, "data_status": data_status,
                   "generated_utc": _now(), "n_stars": 0, "n_stars_assessable": 0,
                   "funnel": {"xi_conservative_positive": 0}, "tiers": rejection_counters([])["tiers"],
                   "rejection_counters": rejection_counters([]),
                   "xi_distribution": {"conservative": {"n": 0}, "nominal": {"n": 0}},
                   "catalogues": {k: {"status": v.get("status"), "n_flares": v.get("n_flares", 0),
                                      "n_stars": v.get("n_stars", 0), "table": v.get("table"),
                                      "route": v.get("route", "none")}
                                  for k, v in acq_cats.items()},
                   "acquisition": (acquire_report or {}).get("acquisition", log.as_dict()),
                   "candidates": [], "degraded": [f"{k}:{v.get('status')}" for k, v in
                                                  acq_cats.items() if v.get("status") != "OK"],
                   "note": ("no star was assessed, so nothing about the ceiling was measured; "
                            "this is NOT a null result and must not be reported as one")}
        _write(out / "summary.json", summary)
        _write(out / "candidates.json", {"generated_utc": summary["generated_utc"],
                                         "verdict": VERDICT_NO_DATA, "candidates": []})
        pd.DataFrame(columns=["record_key", "star_key", "tier"]).to_csv(
            out / "candidates.csv", index=False)
        pd.DataFrame(records).to_csv(out / "xi_table.csv", index=False)
        print(f"[arc] assess: {VERDICT_NO_DATA} ({data_status})")
        return summary

    # --- the shortlist: every star above either bound ---------------------------
    def _x(r, k):
        try:
            return float(r.get(k, np.nan))
        except (TypeError, ValueError):
            return float("nan")

    sconf = conf.get("shortlist") or {}
    above = [r for r in records if np.isfinite(_x(r, "xi_conservative_max"))
             and _x(r, "xi_conservative_max") > 0]
    nominal = [r for r in records if r not in above and np.isfinite(_x(r, "xi_nominal_max"))
               and _x(r, "xi_nominal_max") > 0]
    above.sort(key=lambda r: -_x(r, "xi_conservative_max"))
    nominal.sort(key=lambda r: -_x(r, "xi_nominal_max"))
    short = above + (nominal if sconf.get("include_watch", True) else [])
    short = short[:int(sconf.get("max_stars", 3000) or 3000)]
    short_keys = {r["star_key"] for r in short}

    contexts: dict[str, dict] = {}
    params_by_mission: dict[str, pd.DataFrame] = {}
    param_records: dict = {}
    gaia_reached_frac = float("nan")
    if short and not offline:
        from .acquire import fetch_star_params_by_id, gaia_context, tap_query
        query_fn = query_fn or tap_query
        gconf = conf.get("gaia") or {}
        for mission in sorted({str(r["mission"]) for r in short}):
            ids = sorted({str(r["star_id"]) for r in short if str(r["mission"]) == mission})
            pos, prec = fetch_star_params_by_id(ids, mission, query_fn=query_fn, log=log,
                                                tables=conf.get("param_tables"))
            param_records[mission] = prec
            params_by_mission[mission] = pos
            if not len(pos):
                continue
            g = gaia_context(pos, cone_fn=cone_fn, log=log,
                             gaia_table=str(gconf.get("table", "I/355/gaiadr3")),
                             vari_table=str(gconf.get("vari_table", "I/358/vclassre")),
                             radius_arcsec=float(gconf.get("cone_radius_arcsec", 12.0)),
                             match_arcsec=float(gconf.get("match_arcsec", 2.0)))
            for sid, c in g.items():
                contexts[f"{mission}:{sid}"] = c
        n_reached = sum(1 for k in short_keys if contexts.get(k, {}).get("gaia_reached"))
        gaia_reached_frac = n_reached / len(short_keys) if short_keys else float("nan")

    # --- better stellar parameters for the shortlist; recompute --------------------
    vetted_in = []
    for r in records:
        rec = dict(r)
        key = rec["star_key"]
        ctx = dict(contexts.get(key, {}))
        ctx["mission"] = rec.get("mission")
        pos = params_by_mission.get(str(rec.get("mission")))
        if pos is not None and len(pos) and key in short_keys:
            hit = pos[pos["star_id"] == str(rec["star_id"])]
            if len(hit):
                h = hit.iloc[0]
                teff = _first_finite(h.get("teff"))
                rad = _first_finite(h.get("radius"))
                lg = _first_finite(h.get("logg"))
                if np.isfinite(lg):
                    ctx["logg"] = lg
                if np.isfinite(rad):
                    ctx["radius_rsun"] = rad
                if bool(rec.get("params_assumed")) and np.isfinite(teff) and np.isfinite(rad):
                    rec = _recompute(rec, teff, rad, phys)
                flag = str(h.get("flag", "") or "")
                if flag and flag.lower() not in ("nan", "none"):
                    rec["catalogue_flag"] = ";".join(x for x in (rec.get("catalogue_flag", ""),
                                                                 flag) if x)
                ruwe = _first_finite(h.get("ruwe"))
                if np.isfinite(ruwe) and not np.isfinite(_x(ctx, "ruwe")):
                    ctx["ruwe"] = ruwe
        if "logg" not in ctx:
            ctx["logg"] = _x(rec, "logg")
        if "radius_rsun" not in ctx:
            ctx["radius_rsun"] = _x(rec, "radius_rsun")
        contexts[key] = ctx
        vetted_in.append(rec)
    vetted = assign_tiers(vetted_in, contexts, vconf)
    counters = rejection_counters(vetted)
    fun = funnel(vetted)

    vdf = pd.DataFrame(vetted)
    drop = [c for c in ("veto_detail", "stage2_pulls", "energies_json", "t_peaks_json")
            if c in vdf.columns]
    vdf.drop(columns=drop).to_csv(out / "xi_table.csv", index=False)
    cands = [r for r in vetted if r.get("tier") in ("candidate", "interest")]
    cands.sort(key=lambda r: -_x(r, "xi_conservative_max"))
    watch = [r for r in vetted if r.get("tier") == "watch"]
    watch.sort(key=lambda r: -_x(r, "xi_nominal_max"))

    degraded = [f"{k}:{v.get('status')}" for k, v in acq_cats.items() if v.get("status") != "OK"]
    star_acq = (acquire_report or {}).get("star_catalogues") or {}
    degraded += [f"stars_{k}:{v.get('status')}" for k, v in star_acq.items()
                 if v.get("status") != "OK"]
    if short and not offline and not (np.isfinite(gaia_reached_frac) and gaia_reached_frac >= 1.0):
        degraded.append(f"gaia_context:{gaia_reached_frac:.2f}_of_shortlist_reached")
    if offline and short:
        degraded.append("gaia_context:offline")

    # A star without a rotational amplitude has no spot area, so it has no
    # ceiling and cannot be tested.  Run 34792280736 reached 100,000 flares on
    # 576 stars and assessed NONE of them, because every rotation catalogue
    # failed -- and still reported NO_CEILING_EXCESS, which reads as a
    # measurement of a sky that was never looked at.  Nothing assessable is
    # NO_DATA_REACHED, whatever else arrived.
    n_assessable_now = int(sum(1 for r in vetted if r.get("assessable")))
    if cands:
        verdict = VERDICT_CANDIDATES
    elif n_assessable_now > 0:
        verdict = VERDICT_NONE
    else:
        verdict = VERDICT_NO_DATA
        degraded.append(
            f"no star assessable: {len(vetted)} stars carried flares but none carried a "
            "rotational amplitude, so no ceiling could be computed")
    xc = [_x(r, "xi_conservative_max") for r in vetted if r.get("assessable")]
    xn = [_x(r, "xi_nominal_max") for r in vetted if r.get("assessable")]
    per_cat = {}
    for s in screens:
        per_cat[s.get("catalogue")] = {k: s.get(k) for k in
                                       ("status", "n_flares", "n_stars", "n_stars_assessable",
                                        "n_stars_no_amplitude", "n_stars_params_assumed",
                                        "n_xi_conservative_positive", "n_xi_nominal_positive",
                                        "amplitude_sources")}
    for k, v in acq_cats.items():
        per_cat.setdefault(k, {}).update({"acquire_status": v.get("status"),
                                          "table": v.get("table"),
                                          "n_flares_acquired": v.get("n_flares", 0)})
    slim = ("record_key", "star_key", "star_id", "catalogue", "mission", "tier", "first_veto",
            "flags", "xi_conservative_max", "xi_nominal_max", "n_above_conservative",
            "n_above_nominal", "n_independent", "n_flares", "e_flare_max_erg",
            "e_mag_conservative_erg", "e_mag_nominal_erg", "amplitude_frac",
            "amplitude_source", "amplitude_unit", "prot", "prot_source", "teff_k", "radius_rsun",
            "logg", "t_spot_k", "params_assumed", "catalogue_flag", "centroid")
    cand_rows = [{k: r.get(k) for k in slim} for r in cands]
    for row, r in zip(cand_rows, cands, strict=True):
        c = contexts.get(r["star_key"], {})
        row.update({"ruwe": c.get("ruwe"), "gaia_nss": c.get("nss"), "gmag": c.get("gmag"),
                    "plx": c.get("plx"), "vari_class": c.get("vari_class"),
                    "n_neighbours_12arcsec": len(c.get("neighbours") or []),
                    "gaia_reached": bool(c.get("gaia_reached", False)),
                    "stage2_pulls": json.dumps(r.get("stage2_pulls") or [])})
    pd.DataFrame(cand_rows, columns=list(slim) + ["ruwe", "gaia_nss", "gmag", "plx", "vari_class",
                                                  "n_neighbours_12arcsec", "gaia_reached",
                                                  "stage2_pulls"]).to_csv(
        out / "candidates.csv", index=False)
    summary = {
        "verdict": verdict, "generated_utc": _now(),
        "n_candidates": int(sum(1 for r in cands if r["tier"] == "candidate")),
        "n_interest": int(sum(1 for r in cands if r["tier"] == "interest")),
        "n_watch": int(len(watch)),
        "n_stars": int(len(vetted)),
        "n_stars_assessable": int(sum(1 for r in vetted if r.get("assessable"))),
        "n_shortlisted": int(len(short)),
        "funnel": fun,
        "xi_distribution": {"conservative": percentiles(xc), "nominal": percentiles(xn)},
        "rejection_counters": counters,
        "tiers": counters["tiers"],
        "catalogues": per_cat,
        "degraded": degraded,
        "offline": bool(offline),
        "gaia_reached_fraction_of_shortlist": gaia_reached_frac,
        "param_tables": param_records,
        "physics": phys,
        "vet": vconf,
        "candidates": [{k: row.get(k) for k in ("record_key", "star_key", "catalogue", "tier",
                                                 "first_veto", "flags", "xi_conservative_max",
                                                 "n_above_conservative", "amplitude_frac",
                                                 "prot", "ruwe", "centroid")}
                       for row in cand_rows],
        "acquisition": {"acquire": (acquire_report or {}).get("acquisition"),
                        "assess": log.as_dict()},
        "note": ("a ceiling-excess star here is a CATALOGUE-LEVEL statement (E_flare above the "
                 "f = 1, B = 3 kG spot bound at the conservative area) pending the stage-2 "
                 "pixel-centroid test; NO_CEILING_EXCESS is a count, not an occurrence limit, "
                 "and is not written up (CLAUDE.md)"),
    }
    _write(out / "summary.json", summary)
    _write(out / "candidates.json", {
        "generated_utc": summary["generated_utc"], "verdict": verdict,
        "candidates": [dict(row, veto_detail=r.get("veto_detail"),
                            stage2_pulls=r.get("stage2_pulls") or [],
                            context={k: v for k, v in contexts.get(r["star_key"], {}).items()
                                     if k != "neighbours"} |
                            {"neighbours": (contexts.get(r["star_key"], {}).get("neighbours")
                                            or [])[:20]})
                       for row, r in zip(cand_rows, cands, strict=True)],
        "watch": [{k: r.get(k) for k in slim} for r in watch[:200]],
    })
    print(f"[arc] assess: {verdict} — {summary['n_candidates']} candidate, "
          f"{summary['n_interest']} interest, {summary['n_watch']} watch of "
          f"{summary['n_stars_assessable']} assessable stars")
    return summary


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def arc_run(stage: str = "all", *, out_dir=None, catalogues=None, max_rows: int | None = None,
            offline: bool = False, query_fn=None, cone_fn=None, conf: dict | None = None,
            fetch_fn=None) -> dict:
    conf = conf or load_arc_config()
    out = Path(out_dir) if out_dir else Path("results") / "arc"
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage in ("all", "", None) else tuple(s.strip() for s in stage.split(","))
    rep: dict = {}
    for s in stages:
        if s == "probe":
            rep = stage_probe(conf, out, catalogues=catalogues, query_fn=query_fn,
                              fetch_fn=fetch_fn)
        elif s == "acquire":
            rep = stage_acquire(conf, out, catalogues=catalogues, query_fn=query_fn,
                                max_rows=max_rows, fetch_fn=fetch_fn)
        elif s == "screen":
            rep = stage_screen(conf, out, catalogues=catalogues)
        elif s == "assess":
            rep = stage_assess(conf, out, offline=offline, query_fn=query_fn, cone_fn=cone_fn)
        else:
            raise SystemExit(f"unknown stage {s!r}; choose from {STAGES}")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(prog="seti arc",
                                description="ARC (S59): superflares above the starspot energy ceiling")
    p.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    p.add_argument("--out-dir", default="results/arc", help="results directory")
    p.add_argument("--catalogues", default="",
                   help="comma-separated catalogue keys from config/arc.yaml (default all)")
    p.add_argument("--max-rows", type=int, default=-1, help="acquire row cap (-1 = config)")
    p.add_argument("--offline", action="store_true", help="assess without Gaia / param queries")
    a = p.parse_args(argv)
    cats = [c for c in a.catalogues.split(",") if c.strip()] or None
    rep = arc_run(a.stage, out_dir=a.out_dir, catalogues=cats,
                  max_rows=None if a.max_rows < 0 else a.max_rows, offline=a.offline)
    v = rep.get("verdict") if isinstance(rep, dict) else None
    if v:
        print(f"[arc] verdict: {v}")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULTS", "STAGES", "VERDICT_CANDIDATES", "VERDICT_NONE", "VERDICT_NO_DATA",
           "arc_run", "build_star_context", "load_arc_config", "main", "screen_catalogue",
           "stage_acquire", "stage_assess", "stage_probe", "stage_screen"]

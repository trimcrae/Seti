"""DERELICT orchestration: acquire -> screen -> vet -> report.

Writes ``results/derelict/``:

``schema.json``      what the SBDB API actually accepted (field discovery + which
                     constraint strategy worked).  The API contract was unverified
                     at build time, so this file is the record of what is real.
``nongrav.csv``      every object with a fitted A1 that survived the screens,
                     with beta / area-to-mass / R attached.
``screened.csv``     EVERY object pulled, with every screen column and the
                     implied beta / area-to-mass / R attached.  This is the
                     parent sample: without it a zero at any gate is
                     unauditable and there is no denominator for a rate.
``negative_a1.csv``  screen 3: the sunward-acceleration census -- the empirical
                     false-positive floor for screen 1.
``control_comets.csv`` the comet sample, run through the identical machinery.
                     Comets *should* light up; if they do not, the pipeline is
                     broken, not the sky.
``candidates.json``  survivors with their full vetting record.
``summary.json``     verdict + funnel counts + degradation.
``REPORT.md``        human-readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ..config import Config, load_config
from .acquire import (
    DEFAULT_FIELDS,
    FetchResult,
    discover_fields,
    enrich_from_details,
    fetch_nongrav_table,
    fetch_object_detail,
)
from .radiation import OUMUAMUA_A1_AU_DAY2, REFERENCE_OBJECTS, amr_from_a1, beta_from_a1
from .screen import ScreenParams, run_screens
from .vet import UNEXPLAINED, VetParams, dedupe, vet_table

CHANNEL = "derelict"


def load_derelict_config(cfg: Config | None = None) -> dict:
    """Read ``config/derelict.yaml``; fall back to module defaults if absent."""
    cfg = cfg or load_config()
    p = cfg.root / "config" / "derelict.yaml"
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def _out_dir(cfg: Config) -> Path:
    d = cfg.root / "results" / CHANNEL
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reference_table() -> list[dict]:
    """The physics benchmark table, emitted with every run so the conversions
    used are auditable from the results alone."""
    rows = [{"object": r.name, "amr_m2_kg": r.amr_m2_kg, "beta": r.beta,
             "q_pr": r.q_pr, "note": r.note} for r in REFERENCE_OBJECTS]
    rows.append({"object": "oumuamua_A1_au_day2", "amr_m2_kg": float(
        amr_from_a1(OUMUAMUA_A1_AU_DAY2)), "beta": float(
        beta_from_a1(OUMUAMUA_A1_AU_DAY2)), "q_pr": 1.0,
        "note": f"A1 = {OUMUAMUA_A1_AU_DAY2:.4e} au/day^2 from Micheli et al. 2018"})
    return rows


def derelict_run(cfg: Config | None = None,
                 *,
                 stage: str = "all",
                 limit: int | None = None,
                 offline_input: str | None = None,
                 max_vet: int = 60,
                 max_enrich: int = 1500,
                 max_control_enrich: int = 400,
                 skip_control: bool = False,
                 transport=None) -> dict:
    """Run the channel.  Returns the summary dict (also written to disk)."""
    cfg = cfg or load_config()
    dcfg = load_derelict_config(cfg)
    sp = ScreenParams.from_config(dcfg)
    vp = VetParams.from_config(dcfg)
    out = _out_dir(cfg)

    summary: dict = {
        "channel": CHANNEL,
        "verdict": "NOT_RUN",
        "stage": stage,
        "degradation": [],
        "funnel": {},
        "reference_physics": _reference_table(),
        "conversions": {
            "beta_per_A1_au_day2": float(beta_from_a1(1.0)),
            "amr_per_A1_au_day2_m2_kg_qpr1": float(amr_from_a1(1.0)),
            "q_pr": sp.q_pr,
            "rho_natural_kg_m3": sp.rho_natural_kg_m3,
            "albedo_assumed": sp.albedo_assumed,
        },
    }

    # --- schema discovery -----------------------------------------------------
    schema: dict = {"attempted": False}
    available = None
    if offline_input is None and stage in {"all", "probe", "acquire"}:
        schema["attempted"] = True
        available, notes = discover_fields(transport=transport)
        schema["notes"] = notes
        schema["fields_discovered"] = sorted(available) if available else None
        schema["fields_requested"] = list(DEFAULT_FIELDS)
        (out / "schema.json").write_text(json.dumps(schema, indent=2))
        if stage == "probe":
            summary["verdict"] = "PROBE_ONLY"
            summary["schema"] = schema
            _write(out, summary, None, None, None, None)
            return summary

    # --- acquisition ----------------------------------------------------------
    if offline_input:
        table = pd.read_csv(offline_input)
        fetch = FetchResult(table=table, status="OK", strategy="offline_input",
                            n_rows=len(table), fields_returned=tuple(table.columns))
        control = pd.DataFrame()
        summary["degradation"].append(f"offline input {offline_input}: no archive touched")
    else:
        fetch = fetch_nongrav_table(kind="a", transport=transport, limit=limit,
                                    available_fields=available)
        table = fetch.table
        control = pd.DataFrame()
        if not skip_control:
            cf = fetch_nongrav_table(kind="c", transport=transport, limit=limit,
                                     available_fields=available)
            control = cf.table
            summary["control_fetch"] = cf.to_dict()
        summary["fetch"] = fetch.to_dict()
        schema["strategy_used"] = fetch.strategy
        schema["fields_returned"] = list(fetch.fields_returned)
        (out / "schema.json").write_text(json.dumps(schema, indent=2))

    if table is None or len(table) == 0:
        # These three are NOT the same statement and must never collapse:
        #   - the server has no such column      -> our query is wrong
        #   - the column exists but is all null  -> our constraint is wrong
        #   - the endpoint was unreachable       -> no data was seen at all
        # There are unambiguously objects with a fitted A1 in SBDB (every
        # non-gravitational comet solution has one), so a zero here is a query
        # defect until proven otherwise -- never an occurrence limit.
        status = getattr(fetch, "status", "NO_DATA_REACHED")
        # An empty table can never be an "OK" verdict, whatever the fetch said
        # (an empty offline CSV would otherwise report success on zero rows).
        summary["verdict"] = "NO_DATA_REACHED" if status == "OK" else status
        summary["funnel"] = {
            "input": 0,
            "rows_returned_by_server": int(getattr(fetch, "n_rows_raw", 0)),
            "a1_column_present": bool(getattr(fetch, "a1_column_present", False)),
        }
        summary["degradation"].append(
            "NO rows with a fitted A1 were obtained. This is a QUERY DEFECT "
            "until proven otherwise -- SBDB certainly contains objects with a "
            "fitted A1 (every non-gravitational comet solution has one). "
            "No screening performed and no candidate emitted. "
            f"Server returned {getattr(fetch, 'n_rows_raw', 0)} raw rows; "
            f"A1 column present: {getattr(fetch, 'a1_column_present', False)}; "
            f"fields the server rejected: "
            f"{list(getattr(fetch, 'fields_rejected', ()))}.")
        _write(out, summary, None, None, None, None)
        return summary

    table = dedupe(table)

    # --- enrich the minimal-field fallback ------------------------------------
    # The unconstrained fallback deliberately pulls only the identity + A1
    # columns (a full-column pull over ~1.4M rows would OOM the runner).  Screen
    # 1 fails CLOSED without orbit-quality metadata, so on that path every row
    # would be rejected for a reason that is about our query, not the sky.  Fill
    # the gap per object BEFORE screening -- the A1 population is small enough
    # that this is cheap, and vetting would fetch these records anyway.
    prefetched: dict[str, dict] = {}
    # The bulk query REJECTS `sigma_A1` outright (measured, run 30203392288), and
    # every SNR screen is meaningless without sigmas.  So enrich whenever the
    # sigmas are missing or all-null -- not only on the minimal-field fallback.
    sig_missing = ("sigma_A1" not in table.columns
                   or not table["sigma_A1"].notna().any())
    needs_enrich = offline_input is None and (
        sig_missing or fetch.strategy == "unconstrained_full_pull")
    summary["sigma_A1_in_bulk"] = not sig_missing
    if needs_enrich:
        subset = table.head(max_enrich)
        summary["degradation"].append(
            f"bulk pull lacks usable sigma_A1 (or minimal-field fallback used); "
            f"enriching {len(subset)} of {len(table)} objects per-object from "
            "sbdb.api (orbit.model_pars carries A1/A2/A3 AND their sigmas) "
            "before screening")
        if len(table) > max_enrich:
            summary["degradation"].append(
                f"ENRICHMENT CAPPED at {max_enrich}; {len(table) - max_enrich} "
                "objects were NOT enriched and will fail the quality gate for "
                "lack of metadata, not for lack of signal")
        for _, row in subset.iterrows():
            key = str(row.get("full_name") or row.get("pdes") or "")
            sstr = str(row.get("spkid") or row.get("pdes") or key)
            prefetched[key] = fetch_object_detail(sstr, transport=transport)
        table = enrich_from_details(table, prefetched)
        summary["enriched_from_detail"] = int(
            sum(1 for d in prefetched.values() if d.get("ok")))
        summary["enrich_failures"] = int(
            sum(1 for d in prefetched.values() if not d.get("ok")))

    # --- screening ------------------------------------------------------------
    sr = run_screens(table, sp)
    summary["funnel"] = sr.funnel
    summary["degradation"].extend(sr.notes)
    if len(sr.table) == 0:
        summary["verdict"] = "SCHEMA_UNUSABLE"
        _write(out, summary, None, None, None, None)
        return summary

    scr = sr.table
    # The FULL annotated table, not just survivors.  Without it a zero at any
    # gate is unauditable, and a channel that publishes only its survivors has
    # no denominator (see the TIDEMARK note in STATUS.md).
    _safe_csv(scr, out / "screened.csv")
    survivors = scr[scr["screen_a1_only"]].copy()
    survivors = survivors.sort_values("R", ascending=False, na_position="last")
    negative = scr[scr["screen_negative_a1"]].copy()
    high_albedo = scr[scr["screen_albedo"]].copy()

    # --- control sample -------------------------------------------------------
    # Comets run through the IDENTICAL machinery.  They should light up: their
    # radial acceleration is real and is outgassing.  `a1_fitted` here is also
    # the standing proof that the field names and the response parsing work --
    # it needs no sigmas, so it is diagnostic even when enrichment is skipped.
    control_funnel = {}
    if control is not None and len(control):
        control = dedupe(control)
        if offline_input is None and max_control_enrich > 0:
            csub = control.head(max_control_enrich)
            cdetails = {}
            for _, row in csub.iterrows():
                key = str(row.get("full_name") or row.get("pdes") or "")
                sstr = str(row.get("spkid") or row.get("pdes") or key)
                cdetails[key] = fetch_object_detail(sstr, transport=transport)
            control = enrich_from_details(control, cdetails)
            summary["control_enriched"] = int(
                sum(1 for d in cdetails.values() if d.get("ok")))
            if len(control) > max_control_enrich:
                summary["degradation"].append(
                    f"control enrichment capped at {max_control_enrich} of "
                    f"{len(control)} comets; unenriched rows lack sigmas and so "
                    "cannot register on any SNR screen")
        cr = run_screens(control, sp)
        control_funnel = cr.funnel
        _safe_csv(cr.table, out / "control_comets.csv")
        summary["control_funnel"] = control_funnel

    # --- vetting --------------------------------------------------------------
    details: dict[str, dict] = dict(prefetched)   # reuse, never refetch
    to_vet = survivors.head(max_vet).copy()
    if offline_input is None and len(to_vet):
        for _, row in to_vet.iterrows():
            key = str(row.get("full_name") or row.get("pdes") or "")
            if key in details and details[key].get("ok"):
                continue
            sstr = str(row.get("spkid") or row.get("pdes") or key)
            details[key] = fetch_object_detail(sstr, transport=transport)
    vet_df = vet_table(to_vet, details, vp)
    if len(vet_df):
        for c in ("verdict", "flags", "notes", "detail_ok"):
            to_vet[c] = vet_df[c].values
        survivors = pd.concat(
            [to_vet, survivors.iloc[len(to_vet):]], axis=0)

    n_unexplained = int((vet_df["verdict"] == UNEXPLAINED).sum()) if len(vet_df) else 0
    summary["funnel"]["vetted"] = int(len(vet_df))
    summary["funnel"]["unexplained_after_vetting"] = n_unexplained
    summary["vetting_breakdown"] = (
        vet_df["verdict"].value_counts().to_dict() if len(vet_df) else {})

    # --- verdict --------------------------------------------------------------
    if n_unexplained > 0:
        summary["verdict"] = "CANDIDATES_UNEXPLAINED"
    elif int(summary["funnel"].get("screen1_a1_only", 0)) > 0:
        summary["verdict"] = "ALL_SURVIVORS_EXPLAINED"
    else:
        summary["verdict"] = "NO_SURVIVORS"
    summary["n_objects_with_fitted_A1"] = int(summary["funnel"].get("a1_fitted", 0))

    _write(out, summary, survivors, negative, high_albedo, vet_df)
    return summary


def _safe_csv(df: pd.DataFrame | None, path: Path) -> None:
    if df is None or len(df) == 0:
        path.write_text("")
        return
    d = df.copy()
    for c in d.columns:
        if d[c].apply(lambda v: isinstance(v, (list, dict, set, tuple))).any():
            d[c] = d[c].apply(json.dumps if d[c].dtype == object else str)
    d.to_csv(path, index=False)


def _write(out: Path, summary: dict, survivors, negative, high_albedo, vet_df) -> None:
    _safe_csv(survivors, out / "nongrav.csv")
    _safe_csv(negative, out / "negative_a1.csv")
    _safe_csv(high_albedo, out / "high_albedo.csv")
    cands = []
    if survivors is not None and len(survivors):
        keep = [c for c in ("full_name", "pdes", "spkid", "class", "H", "diameter",
                            "albedo", "a", "e", "i", "data_arc", "condition_code",
                            "n_obs_used", "A1", "sigma_A1", "A2", "sigma_A2",
                            "A3", "sigma_A3", "a1_snr", "a2_state", "a3_state",
                            "nonradial_constrained", "beta_implied",
                            "amr_implied_m2_kg", "areal_density_kg_m2",
                            "R", "R_lo", "R_hi", "diameter_source",
                            "verdict", "flags", "notes")
                if c in survivors.columns]
        cands = json.loads(survivors[keep].to_json(orient="records"))
    (out / "candidates.json").write_text(json.dumps(cands, indent=2, default=str))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "REPORT.md").write_text(_report_md(summary, cands))


def _report_md(summary: dict, cands: list[dict]) -> str:
    f = summary.get("funnel", {})
    lines = [
        "# DERELICT — thin-film debris via radiation-pressure acceleration",
        "",
        f"**Verdict:** `{summary.get('verdict')}`",
        "",
        "## Conversions used",
        "",
        f"- `beta = {summary['conversions']['beta_per_A1_au_day2']:.6g} * A1[au/day^2]`",
        f"- `AMR  = {summary['conversions']['amr_per_A1_au_day2_m2_kg_qpr1']:.6g}"
        f" * A1[au/day^2]` m^2/kg at Q_pr = {summary['conversions']['q_pr']}",
        f"- `AMR_natural = 3 / (2 D rho)`, rho = "
        f"{summary['conversions']['rho_natural_kg_m3']} kg/m^3",
        "",
        "## Funnel",
        "",
        "| stage | n |", "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in f.items()]
    if summary.get("control_funnel"):
        lines += ["", "## Comet control sample (should light up)", "",
                  "| stage | n |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in summary["control_funnel"].items()]
    if summary.get("vetting_breakdown"):
        lines += ["", "## Vetting", "", "| verdict | n |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in summary["vetting_breakdown"].items()]
    if summary.get("degradation"):
        lines += ["", "## Degradation", ""]
        lines += [f"- {d}" for d in summary["degradation"]]
    if cands:
        lines += ["", "## Survivors (top by R)", "",
                  "| object | A1/sigma | R | AMR (m^2/kg) | verdict |", "|---|---|---|---|---|"]
        for c in cands[:40]:
            lines.append(
                f"| {c.get('full_name', '?')} | {_fmt(c.get('a1_snr'))} | "
                f"{_fmt(c.get('R'))} | {_fmt(c.get('amr_implied_m2_kg'))} | "
                f"{c.get('verdict', '-')} |")
    lines += ["", "_No candidate here is a detection claim. Every survivor is a "
              "systematic until traced; see `docs/derelict.md`._", ""]
    return "\n".join(lines)


def _fmt(v) -> str:
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return "-"

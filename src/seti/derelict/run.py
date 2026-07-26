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
    QueryLog,
    discover_fields,
    enrich_from_details,
    fetch_nongrav_table,
    fetch_object_detail,
)
from .census import (
    CensusParams,
    completeness_probe,
    dark_comet_census,
    high_albedo_census,
    negative_a1_census,
)
from .radiation import OUMUAMUA_A1_AU_DAY2, REFERENCE_OBJECTS, amr_from_a1, beta_from_a1
from .screen import ScreenParams, run_screens
from .vet import UNEXPLAINED, VetParams, dedupe, vet_table

CHANNEL = "derelict"

#: Stages the CLI and the workflow can dispatch individually.
STAGES: tuple[str, ...] = ("all", "probe", "search", "completeness",
                           "dark_comets", "high_albedo")


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
                 skip_completeness: bool = False,
                 skip_dark_comets: bool = False,
                 skip_high_albedo: bool = False,
                 completeness_limit: int | None = None,
                 searcher=None,
                 tries: int = 3,
                 transport=None) -> dict:
    """Run the channel.  Returns the summary dict (also written to disk)."""
    cfg = cfg or load_config()
    dcfg = load_derelict_config(cfg)
    sp = ScreenParams.from_config(dcfg)
    vp = VetParams.from_config(dcfg)
    cp = CensusParams.from_config(dcfg)
    out = _out_dir(cfg)

    # Every query the run issues, in order, verbatim.  This is the standing
    # guard against the failure mode of run 30203392288: without it, "the
    # archive was never reached" and "the archive answered with nothing" are
    # indistinguishable in the outputs.
    qlog = QueryLog()

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
            _write(out, summary, qlog=qlog, cp=cp)
            return summary

    # --- standalone census stages ---------------------------------------------
    # Each of these answers a question the 22-row A1 parent sample cannot, and
    # each is dispatchable on its own so a heavy pull can be re-run without
    # re-running the whole channel.  A standalone stage writes its OWN outputs
    # and a `summary_<stage>.json`; it never overwrites the main summary, so a
    # partial run cannot silently replace a full one.
    if stage == "dark_comets":
        summary["dark_comets"] = _run_dark_comets(out, sp, cp, transport, qlog,
                                                  tries=tries)
        summary["verdict"] = "DARK_COMET_CENSUS_ONLY"
        _write_stage(out, "dark_comets", summary, qlog, cp)
        return summary
    if stage == "high_albedo":
        summary["high_albedo"] = _run_high_albedo(
            out, sp, cp, transport, qlog, searcher, limit,
            skip_control=skip_control, tries=tries)
        summary["verdict"] = "HIGH_ALBEDO_CENSUS_ONLY"
        _write_stage(out, "high_albedo", summary, qlog, cp)
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
                                    available_fields=available, log=qlog,
                                    tries=tries)
        table = fetch.table
        control = pd.DataFrame()
        if not skip_control:
            cf = fetch_nongrav_table(kind="c", transport=transport, limit=limit,
                                     available_fields=available, log=qlog,
                                     tries=tries)
            control = cf.table
            summary["control_fetch"] = cf.to_dict()
        summary["fetch"] = fetch.to_dict()
        schema["strategy_used"] = fetch.strategy
        schema["fields_returned"] = list(fetch.fields_returned)
        (out / "schema.json").write_text(json.dumps(schema, indent=2))

    # --- completeness cross-check (decisive; runs BEFORE anything is believed) -
    # 22 rows is small enough that "the A1|DF constraint is subtly wrong" and
    # "the A1 population really is that small" are indistinguishable.  They are
    # completely different statements, so prove which one it is before screening
    # anything: pull the catalogue UNCONSTRAINED with a minimal column set,
    # count non-null A1 client-side, and compare designation SETS.
    if offline_input is None and not skip_completeness and stage in {"all", "completeness"}:
        comp = {}
        for kind_label, kind_code, constrained, chunks in (
                ("asteroid", "a", table, cp.completeness_chunk_classes),
                ("comet", "c", control, cp.completeness_comet_chunk_classes)):
            if kind_code == "c" and skip_control:
                continue
            cr = completeness_probe(
                kind_code, constrained, transport=transport, log=qlog,
                fields=tuple(cp.completeness_fields),
                limit=completeness_limit, class_chunks=chunks,
                max_listed=cp.completeness_max_listed, tries=tries)
            comp[kind_label] = cr.to_dict()
            summary["degradation"].extend(
                f"completeness[{kind_label}]: {n}" for n in cr.notes)
        comp["verdict"] = _combined_completeness_verdict(comp)
        (out / "completeness.json").write_text(json.dumps(comp, indent=2, default=str))
        summary["completeness"] = comp
        if stage == "completeness":
            summary["verdict"] = f"COMPLETENESS_ONLY:{comp['verdict']}"
            _write_stage(out, "completeness", summary, qlog, cp)
            return summary
    elif stage == "completeness":
        summary["verdict"] = "COMPLETENESS_NOT_RUN"
        summary["degradation"].append(
            "the completeness stage was requested but is unavailable offline or "
            "was explicitly skipped; completeness is UNTESTED, not proven")
        _write_stage(out, "completeness", summary, qlog, cp)
        return summary

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
        _write(out, summary, qlog=qlog, cp=cp)
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
            prefetched[key] = fetch_object_detail(
                sstr, transport=transport, log=qlog, tries=tries,
                label=f"enrich:asteroid:{sstr}")
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
        _write(out, summary, qlog=qlog, cp=cp)
        return summary

    scr = sr.table
    # The FULL annotated table, not just survivors.  Without it a zero at any
    # gate is unauditable, and a channel that publishes only its survivors has
    # no denominator (see the TIDEMARK note in STATUS.md).
    _safe_csv(scr, out / "screened.csv")
    survivors = scr[scr["screen_a1_only"]].copy()
    survivors = survivors.sort_values("R", ascending=False, na_position="last")

    # --- control sample -------------------------------------------------------
    # Comets run through the IDENTICAL machinery.  They should light up: their
    # radial acceleration is real and is outgassing.  `a1_fitted` here is also
    # the standing proof that the field names and the response parsing work --
    # it needs no sigmas, so it is diagnostic even when enrichment is skipped.
    control_funnel = {}
    control_scr = pd.DataFrame()
    if control is not None and len(control):
        control = dedupe(control)
        if offline_input is None and max_control_enrich > 0:
            csub = control.head(max_control_enrich)
            cdetails = {}
            for _, row in csub.iterrows():
                key = str(row.get("full_name") or row.get("pdes") or "")
                sstr = str(row.get("spkid") or row.get("pdes") or key)
                cdetails[key] = fetch_object_detail(
                    sstr, transport=transport, log=qlog, tries=tries,
                    label=f"enrich:comet:{sstr}")
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
        control_scr = cr.table
        _safe_csv(cr.table, out / "control_comets.csv")
        summary["control_funnel"] = control_funnel

    # --- screen 3: the negative-A1 census, as a RATE with its denominator ------
    # Radiation pressure cannot push sunward, so every object here is a
    # systematic, and |R| is the size-normalised magnitude of that systematic --
    # the empirical false-positive floor the R threshold has to clear.  Measured
    # on the comet control as well, because a floor estimated from 272 objects
    # is worth far more than one estimated from 22.
    negative, neg_summary = negative_a1_census(
        {"asteroid": scr, "comet": control_scr}, sp)
    summary["negative_a1_census"] = neg_summary
    for label, st in (neg_summary.get("populations") or {}).items():
        summary["funnel"][f"screen3_negative_a1_{label}"] = int(st.get("n_negative", 0))
        summary["funnel"][f"screen3_denominator_{label}"] = int(st.get("n_a1_fitted", 0))

    # --- screen 4: the INDEPENDENT, catalogue-wide albedo census ---------------
    # Deliberately not restricted to the A1 sample: whether an orbit solution
    # happened to include a non-gravitational term has nothing to do with
    # whether the body reflects 80% of the light that hits it.
    high_albedo = scr[scr["screen_albedo"]].copy()   # the A1-sample view
    if offline_input is None and not skip_high_albedo and stage in {"all", "search"}:
        ha = _run_high_albedo(out, sp, cp, transport, qlog, searcher, limit,
                              skip_control=skip_control, tries=tries)
        summary["high_albedo"] = ha
        summary["funnel"]["screen4_albedo_catalogue_wide"] = int(
            ha.get("asteroid", {}).get("n_above_cut", 0))
        summary["funnel"]["screen4_albedo_confirmed_two_sources"] = int(
            (ha.get("asteroid", {}).get("crosscheck") or {}).get(
                "n_confirmed_above_cut", 0))
    else:
        _safe_csv(high_albedo, out / "high_albedo_a1_sample.csv")
        summary["degradation"].append(
            "the catalogue-wide albedo census did not run; screen 4 covers only "
            "the A1 sample, which is a far narrower question")

    # --- the dark-comet named-target census -----------------------------------
    if offline_input is None and not skip_dark_comets and stage in {"all", "search"}:
        summary["dark_comets"] = _run_dark_comets(out, sp, cp, transport, qlog,
                                                  tries=tries)

    # --- vetting --------------------------------------------------------------
    details: dict[str, dict] = dict(prefetched)   # reuse, never refetch
    to_vet = survivors.head(max_vet).copy()
    if offline_input is None and len(to_vet):
        for _, row in to_vet.iterrows():
            key = str(row.get("full_name") or row.get("pdes") or "")
            if key in details and details[key].get("ok"):
                continue
            sstr = str(row.get("spkid") or row.get("pdes") or key)
            details[key] = fetch_object_detail(
                sstr, transport=transport, log=qlog, tries=tries,
                label=f"vet:{sstr}")
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

    _write(out, summary, survivors=survivors, negative=negative,
           high_albedo=high_albedo, vet_df=vet_df, qlog=qlog, cp=cp)
    return summary


def _combined_completeness_verdict(comp: dict) -> str:
    """One verdict over the per-kind completeness probes.

    Deliberately pessimistic: the census is only ``CONSTRAINT_COMPLETE`` when
    **every** probed kind agreed.  Anything else keeps the worst outcome, so a
    comet-side disagreement can never be hidden behind an asteroid-side pass.
    """
    from .census import CONSTRAINT_COMPLETE, CONSTRAINT_INCOMPLETE, PROBE_FAILED

    verdicts = [v.get("verdict") for k, v in comp.items()
                if isinstance(v, dict) and v.get("verdict")]
    if not verdicts:
        return PROBE_FAILED
    for bad in (CONSTRAINT_INCOMPLETE, PROBE_FAILED):
        if bad in verdicts:
            return bad
    if all(v == CONSTRAINT_COMPLETE for v in verdicts):
        return CONSTRAINT_COMPLETE
    return next(v for v in verdicts if v != CONSTRAINT_COMPLETE)


def _run_dark_comets(out: Path, sp: ScreenParams, cp: CensusParams,
                     transport, qlog: QueryLog, tries: int = 3) -> dict:
    """The Seligman et al. dark-comet named-target census.

    Those papers selected on large **non-radial** acceleration -- the complement
    of what this channel wants -- so the question worth asking of their sample
    is which members have a JPL-fitted ``A1`` at all, which are A1-*only*, and
    what ``R`` they sit at.  Designations come from ``config/derelict.yaml``
    with their source paper attached and are resolved through ``sbdb.api``; an
    unresolvable designation is reported, never silently dropped.
    """
    if not cp.dark_comets:
        return {"status": "NO_TARGETS_CONFIGURED",
                "note": "config/derelict.yaml carries no dark_comets.targets; "
                        "the census cannot run on a list recalled from memory"}
    table, dsummary = dark_comet_census(cp.dark_comets, params=sp,
                                        transport=transport, log=qlog,
                                        tries=tries)
    _safe_csv(table, out / "dark_comets.csv")
    dsummary["status"] = "OK"
    dsummary["provenance"] = (
        "designations read verbatim from the fetched full text in "
        "results/derelictlit/ (Seligman et al. 2023 arXiv:2212.08115 Table 1; "
        "Seligman et al. 2024 PNAS 121 e2406424121 Tables 1-2), never recalled")
    return dsummary


def _run_high_albedo(out: Path, sp: ScreenParams, cp: CensusParams,
                     transport, qlog: QueryLog, searcher, limit: int | None,
                     *, skip_control: bool = False, tries: int = 3) -> dict:
    """Screen 4, run catalogue-wide and independently of the A1 sample."""
    result: dict = {}
    frames = []
    for label, kind in (("asteroid", "a"), ("comet", "c")):
        if kind == "c" and skip_control:
            continue
        hr = high_albedo_census(
            params=sp, kind=kind, transport=transport, log=qlog, limit=limit,
            searcher=searcher, crosscheck=cp.high_albedo_crosscheck,
            max_crosscheck=cp.high_albedo_max_crosscheck, tries=tries)
        result[label] = hr.to_dict()
        if len(hr.table):
            t = hr.table.copy()
            t["population"] = label
            frames.append(t)
    if frames:
        _safe_csv(pd.concat(frames, ignore_index=True, sort=False),
                  out / "high_albedo.csv")
    else:
        _safe_csv(None, out / "high_albedo.csv")
    return result


def _write_stage(out: Path, stage: str, summary: dict, qlog: QueryLog,
                 cp: CensusParams) -> None:
    """Write a standalone stage's own summary WITHOUT clobbering the main one.

    A stage run on its own has not screened anything, so letting it overwrite
    ``summary.json`` would replace a full funnel with a partial one and make the
    channel look like it had regressed.
    """
    summary["queries"], note = _query_payload(qlog, cp)
    if note:
        summary["queries_truncated"] = note
    (out / "queries.json").write_text(json.dumps(qlog.to_list(), indent=2, default=str))
    (out / f"summary_{stage}.json").write_text(
        json.dumps(summary, indent=2, default=str))


def _query_payload(qlog: QueryLog | None, cp: CensusParams | None
                   ) -> tuple[list[dict], dict | None]:
    """The query log for ``summary.json``, with any truncation made explicit."""
    if qlog is None:
        return [], None
    records = qlog.to_list()
    cap = cp.max_queries_in_summary if cp else 4000
    if len(records) <= cap:
        return records, None
    return records[:cap], {
        "n_total": len(records), "n_inlined": cap,
        "complete_log": "results/derelict/queries.json",
        "note": "summary.json inlines the first N queries only; the COMPLETE, "
                "unredacted log is written to queries.json. Nothing is dropped.",
    }


def _safe_csv(df: pd.DataFrame | None, path: Path) -> None:
    if df is None or len(df) == 0:
        path.write_text("")
        return
    d = df.copy()
    for c in d.columns:
        if d[c].apply(lambda v: isinstance(v, (list, dict, set, tuple))).any():
            d[c] = d[c].apply(json.dumps if d[c].dtype == object else str)
    d.to_csv(path, index=False)


def _write(out: Path, summary: dict, survivors=None, negative=None,
           high_albedo=None, vet_df=None, *, qlog: QueryLog | None = None,
           cp: CensusParams | None = None) -> None:
    _safe_csv(survivors, out / "nongrav.csv")
    _safe_csv(negative, out / "negative_a1.csv")
    # The A1-sample view of screen 4 keeps its own file; `high_albedo.csv` is
    # written by the catalogue-wide census, which is a different (much larger)
    # question and must not be conflated with it.
    _safe_csv(high_albedo, out / "high_albedo_a1_sample.csv")
    # Every query, verbatim and unredacted: label, URL, HTTP status, per-query
    # status and row count.  A reader must be able to reconstruct exactly what
    # was asked and exactly what came back.
    if qlog is not None:
        summary["queries"], note = _query_payload(qlog, cp)
        if note:
            summary["queries_truncated"] = note
        summary["query_status_counts"] = qlog.counts()
        (out / "queries.json").write_text(
            json.dumps(qlog.to_list(), indent=2, default=str))
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
    comp = summary.get("completeness") or {}
    if comp:
        lines += ["", "## Is the A1 census complete?", "",
                  f"**{comp.get('verdict')}**", "",
                  "| kind | constrained | unconstrained rows | non-null A1 | "
                  "missing | extra | verdict |", "|---|---|---|---|---|---|---|"]
        for k, v in comp.items():
            if not isinstance(v, dict):
                continue
            lines.append(
                f"| {k} | {v.get('n_constrained')} | {v.get('n_unconstrained_rows')} "
                f"| {v.get('n_unconstrained_nonnull_A1')} | "
                f"{v.get('n_missing_from_constrained')} | "
                f"{v.get('n_extra_in_constrained')} | {v.get('verdict')} |")
    neg = summary.get("negative_a1_census") or {}
    if neg.get("populations"):
        lines += ["", "## Screen 3 — the sunward-acceleration floor", "",
                  "| population | negative | denominator (A1 fitted) | rate | "
                  "max \\|R\\| | median \\|R\\| |", "|---|---|---|---|---|---|"]
        for k, v in neg["populations"].items():
            lines.append(
                f"| {k} | {v.get('n_negative')} | {v.get('n_a1_fitted')} | "
                f"{_fmt(v.get('rate'))} | {_fmt(v.get('abs_R_max'))} | "
                f"{_fmt(v.get('abs_R_median'))} |")
        lines += ["", f"Flag threshold R = {_fmt(neg.get('r_flag_threshold'))}; "
                  f"measured floor max |R| = {_fmt(neg.get('floor_abs_R_max'))}. "
                  "Radiation pressure cannot push sunward, so every row above is "
                  "a systematic."]
    dc = summary.get("dark_comets") or {}
    if dc.get("n_targets"):
        lines += ["", "## Dark-comet named-target census", "",
                  f"- targets: {dc.get('n_targets')}, resolved: "
                  f"{dc.get('n_resolved')}, unresolved: {dc.get('n_unresolved')}",
                  f"- with a JPL-fitted A1 (value + sigma): "
                  f"{dc.get('n_with_A1_and_sigma')}",
                  f"- A1-only (screen 1 pass): {dc.get('n_a1_only')}",
                  f"- no A1 fitted at all: {', '.join(dc.get('no_A1_fitted') or []) or 'none'}"]
        if dc.get("unresolved"):
            lines.append(f"- **UNRESOLVED designations:** {', '.join(dc['unresolved'])}")
    ha = summary.get("high_albedo") or {}
    if ha:
        lines += ["", "## Screen 4 — catalogue-wide albedo (independent of A1)", "",
                  "| population | status | strategy | rows returned | above cut | "
                  "confirmed by IRSA |", "|---|---|---|---|---|---|"]
        for k, v in ha.items():
            if not isinstance(v, dict):
                continue
            lines.append(
                f"| {k} | {v.get('status')} | {v.get('strategy')} | "
                f"{v.get('n_rows_returned')} | {v.get('n_above_cut')} | "
                f"{(v.get('crosscheck') or {}).get('n_confirmed_above_cut', '-')} |")
    if summary.get("query_status_counts"):
        lines += ["", "## Queries issued", "",
                  "| status | n |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in summary["query_status_counts"].items()]
        lines += ["", "_Every query is recorded verbatim (URL, HTTP status, row "
                  "count) in `queries.json`. `QUERY_FAILED` and "
                  "`QUERY_RETURNED_ZERO_ROWS` are different statements and are "
                  "never merged._"]
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

"""Catalogue-scale census stages for DERELICT.

Four things that the main funnel (``run.py``) cannot do from its own 22-row
parent sample, and that are the difference between "we screened what the
constraint gave us" and "we screened the whole population":

:func:`completeness_probe`
    Proves the ``sb-cdata={"AND":["A1|DF"]}`` constraint really did return the
    **whole** A1 population, by pulling the catalogue *unconstrained* with a
    minimal column set and comparing designation sets.  22 rows is small enough
    that "the constraint is subtly wrong" and "the population really is that
    small" are indistinguishable without this check -- and they are completely
    different statements.

:func:`dark_comet_census`
    The named-target census of the Seligman et al. 2023/2024 dark comets.  Those
    papers selected on **non-radial** acceleration, i.e. the complement of what
    this channel wants, so the interesting question is which of them have a JPL
    ``A1`` at all and what ``R`` they sit at.  Designations are resolved through
    ``sbdb.api`` one at a time and an unresolvable designation is *reported*,
    never silently dropped.

:func:`negative_a1_census`
    Screen 3 as a **rate with its denominator**, measured on the comet control
    (hundreds of objects) as well as the asteroid sample (tens).  The negative
    tail is the empirical false-positive floor for screen 1: radiation pressure
    cannot push sunward, so every significant ``A1 < 0`` is a systematic, and
    its ``|R|`` is the size-normalised magnitude of that systematic.

:func:`high_albedo_census`
    Screen 4 run **independently of A1**, catalogue-wide: geometric albedo above
    the natural-regolith ceiling is a technosignature in its own right and has
    nothing to do with whether an orbit fit happened to include a non-grav term.
    Cross-checked against IRSA (NEOWISE) so a single-source albedo is never
    believed on its own.

Everything here is offline-testable: the network is reached only through the
injected ``transport`` / ``searcher`` callables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .acquire import (
    SBDB_QUERY_URL,
    STATUS_OK,
    STATUS_QUERY_FAILED,
    STATUS_ZERO_ROWS,
    QueryLog,
    Transport,
    _build_url,
    fetch_object_detail,
    model_par_values,
    query_with_field_pruning,
)
from .radiation import amr_from_a1, amr_natural, diameter_from_h
from .screen import ScreenParams, run_screens

# =============================================================================
# Verdict vocabularies.  Every one of these is a distinct statement; none may
# collapse into another.
# =============================================================================
#: The constrained query returned exactly the objects the unconstrained pull says
#: have a non-null A1.  The census is provably complete.
CONSTRAINT_COMPLETE = "CONSTRAINT_COMPLETE"
#: The unconstrained pull found objects with an A1 that the constraint missed.
#: The constraint is wrong and the primary path must switch to the full pull.
CONSTRAINT_INCOMPLETE = "CONSTRAINT_INCOMPLETE"
#: Nothing missing, but the constrained set contains objects the probe did not
#: see.  That is a defect in the PROBE (truncation, paging, parse), not in the
#: constraint -- and it means completeness is still unproven.
PROBE_INCONSISTENT = "PROBE_INCONSISTENT"
#: The unconstrained pull never returned.  Completeness is UNTESTED; this is
#: never reported as agreement.
PROBE_FAILED = "PROBE_FAILED"

#: IRSA cross-check outcomes.
XCHECK_OK = "OK"
XCHECK_UNREACHED = "IRSA_NOT_REACHED"
XCHECK_NO_TABLE = "NO_ALBEDO_TABLE_FOUND"
XCHECK_NO_COLUMN = "NO_ALBEDO_COLUMN_FOUND"
XCHECK_SKIPPED = "SKIPPED"

#: Minimal column set for the unconstrained completeness pull.  The full
#: catalogue is ~1.4M asteroid rows; every extra column is another ~15 MB of
#: JSON, and the probe only has to answer "which objects have an A1 at all?".
COMPLETENESS_FIELDS: tuple[str, ...] = (
    "spkid", "full_name", "pdes", "kind", "class", "A1", "A2", "A3",
)

#: Columns the independent albedo screen needs.  ``albedo_sigma`` is requested
#: speculatively -- if SBDB has no such field the self-healing pruner drops it
#: and the run records that the uncertainty was UNAVAILABLE rather than assuming
#: the point estimate is exact.
HIGH_ALBEDO_FIELDS: tuple[str, ...] = (
    "spkid", "full_name", "pdes", "name", "kind", "class", "neo",
    "H", "diameter", "albedo", "albedo_sigma", "diameter_sigma",
    "e", "a", "q", "i",
    "n_obs_used", "data_arc", "condition_code", "rms", "first_obs", "last_obs",
    "A1", "A2", "A3",
)

#: SBDB orbit-class codes, used ONLY as a chunking axis when a single
#: unconstrained pull fails (memory or timeout).  Not trusted as an exhaustive
#: list: the union's row count is compared against the server's own ``count``,
#: and a shortfall is reported rather than glossed.
ASTEROID_CLASS_CHUNKS: tuple[str, ...] = (
    "IEO", "ATE", "APO", "AMO", "MCA", "IMB", "MBA", "OMB", "TJN",
    "CEN", "TNO", "AST", "PAA", "HYA",
)
COMET_CLASS_CHUNKS: tuple[str, ...] = (
    "COM", "CTc", "ETc", "HTC", "JFc", "JFC", "PAR", "HYP",
)

#: Give up on the chunked fallback after this many consecutive chunk failures.
#: Chunking exists to survive a pull that is too big, not one that is
#: unreachable; without this the retry backoff alone costs minutes per class.
_MAX_CONSECUTIVE_CHUNK_FAILURES = 3


# =============================================================================
# Designation handling
# =============================================================================
def normalise_designation(value) -> str:
    """Canonical key for comparing designations across two SBDB responses.

    SBDB prints the same object as ``"(2005 VL1)"``, ``"2005 VL1"`` and
    ``"523599 (2003 RM)"`` depending on the field, and the dark-comet papers
    print a third form again.  Every comparison in this module goes through
    here so that a formatting difference can never masquerade as a missing
    object -- which would turn a formatting bug into "the constraint is
    incomplete".
    """
    s = str(value if value is not None else "").strip()
    if not s:
        return ""
    s = s.replace("(", " ").replace(")", " ")
    return " ".join(s.split()).upper()


def designation_keys(row: pd.Series | dict) -> set[str]:
    """Every identifier form a row offers, normalised.

    Two rows are the same object if *any* key matches, because the constrained
    and unconstrained pulls do not necessarily populate the same identifier
    columns.
    """
    out: set[str] = set()
    for col in ("spkid", "pdes", "full_name", "name"):
        try:
            val = row[col] if col in row else None
        except (KeyError, TypeError):
            val = None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        key = normalise_designation(val)
        if key:
            out.add(key)
    return out


def _primary_key(row: pd.Series | dict) -> str:
    """The single best identifier for a row: spkid > pdes > full_name."""
    for col in ("spkid", "pdes", "full_name"):
        try:
            val = row[col] if col in row else None
        except (KeyError, TypeError):
            val = None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        key = normalise_designation(val)
        if key:
            return key
    return ""


def designation_index(df: pd.DataFrame) -> dict[str, str]:
    """Map every identifier form present -> the row's primary key."""
    index: dict[str, str] = {}
    if df is None or len(df) == 0:
        return index
    for _, row in df.iterrows():
        prim = _primary_key(row)
        if not prim:
            continue
        for key in designation_keys(row):
            index[key] = prim
    return index


def designation_variants(value) -> set[str]:
    """Normalised designation plus the space-free form.

    Used only for **cross-archive** matching. IRSA/NEOWISE tables render minor
    planet designations inconsistently (``2005 VL1`` vs ``2005VL1`` vs a packed
    form), and an unmatched row is indistinguishable from a body with no
    independent albedo -- which would silently weaken the cross-check into
    "nothing agreed" when the truth is "nothing was compared". Within SBDB the
    plain normalisation is enough, so this is not used there.
    """
    base = normalise_designation(value)
    if not base:
        return set()
    return {base, base.replace(" ", "")}


def designation_groups(df: pd.DataFrame) -> dict[str, set[str]]:
    """``{primary key: every identifier form that row carries}``.

    Set membership has to be decided per OBJECT, not per identifier string. The
    constrained pull may identify a body by ``pdes`` while the unconstrained one
    leads with ``spkid``; comparing raw key sets would then report a perfectly
    matched object as both "missing" and "extra". Grouping by primary key and
    asking whether the two groups *intersect* is the only comparison that cannot
    manufacture a false ``CONSTRAINT_INCOMPLETE`` out of a schema difference.
    """
    groups: dict[str, set[str]] = {}
    if df is None or len(df) == 0:
        return groups
    for _, row in df.iterrows():
        prim = _primary_key(row)
        if not prim:
            continue
        groups.setdefault(prim, set()).update(designation_keys(row))
    return groups


# =============================================================================
# 1. Completeness cross-check of the A1|DF constraint
# =============================================================================
@dataclass
class CompletenessResult:
    """Outcome of comparing a constrained pull against an unconstrained one."""
    kind: str = "a"
    verdict: str = PROBE_FAILED
    strategy: str = ""
    n_constrained: int = 0
    n_unconstrained_rows: int = 0
    n_unconstrained_nonnull_A1: int = 0
    server_count: int | None = None
    missing_from_constrained: list[str] = field(default_factory=list)
    extra_in_constrained: list[str] = field(default_factory=list)
    fields_rejected: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "verdict": self.verdict,
            "strategy": self.strategy,
            "n_constrained": int(self.n_constrained),
            "n_unconstrained_rows": int(self.n_unconstrained_rows),
            "n_unconstrained_nonnull_A1": int(self.n_unconstrained_nonnull_A1),
            "server_count": self.server_count,
            "n_missing_from_constrained": len(self.missing_from_constrained),
            "missing_from_constrained": self.missing_from_constrained,
            "n_extra_in_constrained": len(self.extra_in_constrained),
            "extra_in_constrained": self.extra_in_constrained,
            "fields_rejected": self.fields_rejected,
            "notes": self.notes,
            "queries": self.queries,
        }


def _unconstrained_pull(kind: str, fields: tuple[str, ...], *,
                        transport: Transport | None,
                        log: QueryLog | None,
                        limit: int | None,
                        tries: int,
                        class_chunks: Sequence[str] | None,
                        ) -> tuple[pd.DataFrame | None, str, list[str], list[str], int | None]:
    """Pull the whole small-body table with no ``sb-cdata`` constraint.

    Tries one monolithic request first (it is the ground truth and it is one
    query), and only chunks by ``sb-class`` if that fails.  Chunking is the
    fallback rather than the default because ``sb-class`` might be silently
    ignored by the server, in which case each chunk would return the entire
    catalogue -- correct, but 14x the bandwidth.

    Returns ``(table, strategy, notes, fields_rejected, server_count)``.
    """
    notes: list[str] = []
    base = {"sb-kind": kind, "full-prec": "1"}
    if limit:
        base["limit"] = limit

    df, meta, err, rejected = query_with_field_pruning(
        fields, base, transport=transport, log=log, tries=tries,
        label=f"completeness:{kind}:unconstrained_single")
    if df is not None:
        return (df, "unconstrained_single", notes, list(rejected),
                meta.get("server_count"))
    notes.append(f"single unconstrained pull failed: {err}")

    if not class_chunks:
        return None, "", notes, list(rejected), None

    frames: list[pd.DataFrame] = []
    counts: list[int] = []
    all_rejected = set(rejected)
    consecutive_failures = 0
    for cls in class_chunks:
        params = dict(base)
        params["sb-class"] = cls
        d, m, e, rej = query_with_field_pruning(
            fields, params, transport=transport, log=log, tries=tries,
            label=f"completeness:{kind}:unconstrained_class:{cls}")
        all_rejected |= set(rej)
        if d is None:
            notes.append(f"class chunk {cls} failed: {e}")
            consecutive_failures += 1
            # The chunked path exists to survive a pull that is too big or too
            # slow, not to survive an unreachable archive.  If the first few
            # chunks all fail the endpoint is simply down, and grinding through
            # the remaining classes only burns retry backoff.
            if consecutive_failures >= _MAX_CONSECUTIVE_CHUNK_FAILURES:
                notes.append(
                    f"abandoned sb-class chunking after "
                    f"{consecutive_failures} consecutive failures: the endpoint "
                    "is unreachable, not merely slow")
                break
            continue
        consecutive_failures = 0
        frames.append(d)
        if m.get("server_count") is not None:
            counts.append(int(m["server_count"]))
    if not frames:
        notes.append("every sb-class chunk failed; unconstrained pull unavailable")
        return None, "", notes, sorted(all_rejected), None
    out = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("spkid", "pdes", "full_name") if c in out.columns]
    if keys:
        before = len(out)
        out = out.drop_duplicates(subset=keys).reset_index(drop=True)
        if before != len(out):
            notes.append(
                f"sb-class chunks overlapped: {before} -> {len(out)} rows after "
                "dedupe (the server may be ignoring sb-class, which is harmless "
                "for completeness but wasteful)")
    notes.append(
        f"chunked by sb-class over {len(frames)} of {len(class_chunks)} classes; "
        "the class list is a FALLBACK axis and is not asserted to be exhaustive")
    return out, "unconstrained_by_class", notes, sorted(all_rejected), (
        sum(counts) if counts else None)


def completeness_probe(kind: str,
                       constrained: pd.DataFrame,
                       *,
                       transport: Transport | None = None,
                       log: QueryLog | None = None,
                       fields: tuple[str, ...] = COMPLETENESS_FIELDS,
                       limit: int | None = None,
                       tries: int = 3,
                       class_chunks: Sequence[str] | None = None,
                       max_listed: int = 500) -> CompletenessResult:
    """Is ``sb-cdata={"AND":["A1|DF"]}`` returning the whole A1 population?

    Pulls every small body of this ``kind`` with no constraint, counts non-null
    ``A1`` **client-side**, and compares the resulting designation set against
    what the constrained query returned.

    The comparison is the point.  A count match alone would not be enough: two
    different sets can have the same size.  So the verdict is driven by the set
    difference in *both* directions, and the differing designations are listed
    (up to ``max_listed``) so a disagreement is actionable rather than merely
    reported.
    """
    res = CompletenessResult(kind=kind)
    res.n_constrained = int(len(constrained)) if constrained is not None else 0
    # Slice this probe's own queries out of the shared log, so completeness.json
    # is self-contained: the reader sees the literal URL that was asked and what
    # came back, without having to correlate against queries.json by hand.
    log_start = len(log.records) if log is not None else 0

    df, strategy, notes, rejected, server_count = _unconstrained_pull(
        kind, fields, transport=transport, log=log, limit=limit, tries=tries,
        class_chunks=class_chunks)
    res.strategy = strategy
    res.notes.extend(notes)
    res.fields_rejected = list(rejected)
    res.server_count = server_count
    if log is not None:
        res.queries = [r.to_dict() for r in log.records[log_start:]]

    if df is None:
        res.verdict = PROBE_FAILED
        res.notes.append(
            "the unconstrained pull never returned, so constraint completeness "
            "is UNTESTED. This is NOT agreement and must not be read as one.")
        return res

    res.n_unconstrained_rows = int(len(df))
    if "A1" not in df.columns:
        res.verdict = PROBE_FAILED
        res.notes.append(
            f"the unconstrained response has no A1 column (got {list(df.columns)}); "
            "the probe cannot count what it cannot see")
        return res

    if server_count is not None and int(server_count) != len(df):
        res.notes.append(
            f"server reported count={server_count} but {len(df)} rows were parsed; "
            "the pull may be truncated, so a 'complete' verdict would be unsafe")

    have_a1 = df[df["A1"].notna()].copy()
    res.n_unconstrained_nonnull_A1 = int(len(have_a1))

    con_groups = designation_groups(constrained if constrained is not None
                                    else pd.DataFrame())
    unc_groups = designation_groups(have_a1)
    con_keys = {k for keys in con_groups.values() for k in keys}
    unc_keys = {k for keys in unc_groups.values() for k in keys}

    # An object is "missing" only when NONE of its identifier forms appears
    # anywhere in the constrained set -- otherwise a spkid-vs-pdes schema
    # difference would be read as a missing object and manufacture a false
    # CONSTRAINT_INCOMPLETE.
    missing = sorted(p for p, keys in unc_groups.items() if not (keys & con_keys))
    extra = sorted(p for p, keys in con_groups.items() if not (keys & unc_keys))

    res.missing_from_constrained = missing[:max_listed]
    res.extra_in_constrained = extra[:max_listed]
    if len(missing) > max_listed:
        res.notes.append(f"{len(missing)} missing designations, listing first {max_listed}")
    if len(extra) > max_listed:
        res.notes.append(f"{len(extra)} extra designations, listing first {max_listed}")

    if missing:
        res.verdict = CONSTRAINT_INCOMPLETE
        res.notes.append(
            f"{len(missing)} objects have a non-null A1 in the unconstrained pull "
            "but were NOT returned by the constrained query. The constrained path "
            "is not a complete census and the primary path must switch to the "
            "unconstrained pull.")
    elif extra:
        res.verdict = PROBE_INCONSISTENT
        res.notes.append(
            f"{len(extra)} objects came back from the constrained query but were "
            "not seen in the unconstrained pull. Nothing is missing from the "
            "science sample, but the probe itself is unreliable (truncation or "
            "paging), so completeness remains UNPROVEN.")
    else:
        res.verdict = CONSTRAINT_COMPLETE
        res.notes.append(
            "the constrained and unconstrained designation sets are identical: "
            "the A1 census is complete for this kind.")
    res.notes.append(
        f"counts: constrained={res.n_constrained}, unconstrained rows="
        f"{res.n_unconstrained_rows}, of which non-null A1="
        f"{res.n_unconstrained_nonnull_A1}")
    return res


# =============================================================================
# 2. Dark-comet named-target census
# =============================================================================
def row_from_detail(designation: str, detail: dict | None) -> dict:
    """Flatten one ``sbdb.api`` record into the columns the screens expect.

    This is the per-object path in its purest form: identity, orbit quality,
    elements, physical parameters, and -- the reason the channel needs it at all
    -- ``orbit.model_pars``, which carries ``A1``/``A2``/``A3`` **and their
    sigmas**.  The bulk query rejects ``sigma_A1`` outright, so this record is
    the only source of the uncertainties every SNR screen depends on.

    A record that did not resolve produces a row with ``resolved=False`` and
    nothing invented.  An unresolvable designation is a *finding*, not a
    silently dropped entry.
    """
    row: dict = {"query_designation": designation, "resolved": False,
                 "full_name": designation, "detail_error": ""}
    if not detail or not detail.get("ok"):
        row["detail_error"] = str((detail or {}).get("error") or "no record")
        return row

    row["resolved"] = True
    obj = detail.get("object") or {}
    row["full_name"] = str(obj.get("fullname") or designation)
    for src, dest in (("spkid", "spkid"), ("des", "pdes"), ("kind", "kind"),
                      ("neo", "neo"), ("pha", "pha")):
        if obj.get(src) is not None:
            row[dest] = obj[src]
    oc = obj.get("orbit_class") or {}
    if oc.get("code"):
        row["class"] = str(oc["code"])

    orbit = detail.get("orbit") or {}
    for key in ("data_arc", "condition_code", "n_obs_used", "rms",
                "first_obs", "last_obs", "epoch"):
        if orbit.get(key) is not None:
            row[key] = (orbit[key] if key in ("first_obs", "last_obs")
                        else pd.to_numeric(orbit[key], errors="coerce"))
    for el in orbit.get("elements") or []:
        if isinstance(el, dict) and el.get("name"):
            row[str(el["name"])] = pd.to_numeric(el.get("value"), errors="coerce")

    for p in detail.get("phys_par") or []:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        name = str(p["name"])
        if name in ("H", "diameter", "albedo", "rot_per", "GM", "density"):
            row[name] = pd.to_numeric(p.get("value"), errors="coerce")
            sig = pd.to_numeric(p.get("sigma"), errors="coerce")
            if pd.notna(sig):
                row[f"{name}_sigma"] = sig

    pars = model_par_values(detail)
    row["model_pars"] = sorted(pars)
    for name, pv in pars.items():
        upper = str(name).strip().upper()
        if upper in {"A1", "A2", "A3", "DT"}:
            row[upper] = pv.get("value")
            row[f"sigma_{upper}"] = pv.get("sigma")
    return row


def dark_comet_census(targets: Sequence[tuple[str, str]],
                      *,
                      params: ScreenParams,
                      transport: Transport | None = None,
                      log: QueryLog | None = None,
                      detail_fetcher: Callable[..., dict] | None = None,
                      tries: int = 3) -> tuple[pd.DataFrame, dict]:
    """Resolve every named dark comet and put it through the channel's physics.

    ``targets`` is ``[(designation, provenance), …]`` where provenance names the
    paper and table the designation was read from -- the repository has been
    bitten twice by designations and arXiv ids recalled from memory, so the
    origin travels with the object.

    Returns ``(table, summary)``.  The table has one row per target, resolved or
    not; the summary carries the counts that answer the actual question: how
    many of the dark comets have a JPL-fitted ``A1`` at all, how many are
    A1-only, and what ``R`` they sit at.
    """
    fetch = detail_fetcher or fetch_object_detail
    rows: list[dict] = []
    for desig, source in targets:
        detail = fetch(desig, transport=transport, tries=tries, log=log,
                       label=f"dark_comet:{desig}")
        row = row_from_detail(desig, detail)
        row["source"] = source
        rows.append(row)

    table = pd.DataFrame(rows)
    if "A1" not in table.columns:
        table["A1"] = np.nan
    if "sigma_A1" not in table.columns:
        table["sigma_A1"] = np.nan

    resolved = table[table["resolved"].astype(bool)].copy()
    if len(resolved):
        sr = run_screens(resolved, params)
        if len(sr.table):
            scored = sr.table
            # Re-attach the unresolved rows so the census reports every target.
            unresolved = table[~table["resolved"].astype(bool)].copy()
            table = pd.concat([scored, unresolved], ignore_index=True, sort=False)

    a1_fitted = table["A1"].notna() & table.get(
        "sigma_A1", pd.Series(np.nan, index=table.index)).notna()
    summary = {
        "n_targets": int(len(table)),
        "n_resolved": int(table["resolved"].astype(bool).sum()),
        "n_unresolved": int((~table["resolved"].astype(bool)).sum()),
        "unresolved": sorted(
            table.loc[~table["resolved"].astype(bool), "query_designation"]
            .astype(str).tolist()),
        "n_with_A1_value": int(table["A1"].notna().sum()),
        "n_with_A1_and_sigma": int(a1_fitted.sum()),
        "no_A1_fitted": sorted(
            table.loc[table["resolved"].astype(bool) & table["A1"].isna(),
                      "query_designation"].astype(str).tolist()),
    }
    for col, key in (("s1_a1_significant", "n_a1_significant"),
                     ("screen_a1_only", "n_a1_only"),
                     ("screen_a1_only_strict", "n_a1_only_strict"),
                     ("screen_negative_a1", "n_negative_a1"),
                     ("screen_r_flag", "n_r_flag")):
        summary[key] = int(table[col].fillna(False).astype(bool).sum()) \
            if col in table.columns else 0
    if "R" in table.columns:
        summary["R_max"] = (float(np.nanmax(table["R"].to_numpy(dtype=float)))
                            if table["R"].notna().any() else None)
    return table, summary


# =============================================================================
# 3. The negative-A1 census, as a rate with its denominator
# =============================================================================
def _abs_r(row: pd.Series, p: ScreenParams) -> tuple[float, float, str]:
    """``|R|`` for a sunward acceleration: the size-normalised systematic.

    :func:`~seti.derelict.radiation.r_statistic` deliberately refuses a negative
    ``A1`` (radiation pressure cannot push sunward, so ``R`` would be
    meaningless as a *physical* quantity).  But the **magnitude** is exactly what
    screen 3 is for: it says how large a spurious area-to-mass the orbit fit
    manufactured, in the units the flag threshold is expressed in.
    """
    a1 = pd.to_numeric(row.get("A1"), errors="coerce")
    if pd.isna(a1) or a1 == 0:
        return float("nan"), float("nan"), "no A1"
    g = float(row.get("g_1au", 1.0) or 1.0)
    amr = abs(float(amr_from_a1(float(a1) * g, q_pr=p.q_pr)))

    diam = pd.to_numeric(row.get("diameter"), errors="coerce")
    if pd.notna(diam) and float(diam) > 0:
        d_m, source = float(diam) * 1000.0, "measured"
    else:
        h = pd.to_numeric(row.get("H"), errors="coerce")
        if pd.isna(h):
            return float("nan"), float("nan"), "no diameter and no H"
        alb = pd.to_numeric(row.get("albedo"), errors="coerce")
        p_use = float(alb) if pd.notna(alb) and 0 < float(alb) <= 1 else p.albedo_assumed
        d_m = float(diameter_from_h(float(h), albedo=p_use))
        source = "H_albedo_measured" if pd.notna(alb) else "H_albedo_assumed"
    return amr / float(amr_natural(d_m, rho_kg_m3=p.rho_natural_kg_m3)), d_m, source


def negative_a1_census(populations: dict[str, pd.DataFrame],
                       p: ScreenParams) -> tuple[pd.DataFrame, dict]:
    """Screen 3 across every population, as a rate with an explicit denominator.

    A count of negative-A1 objects is uninterpretable on its own -- one object
    out of 22 and one out of 272 are different measurements.  So this returns
    ``n_negative / n_a1_fitted`` per population, and records every negative
    object's ``|R|``, which is the empirical false-positive floor that the
    ``R`` flag threshold must sit above.

    The comet control matters here specifically because it is an order of
    magnitude larger than the asteroid sample: the floor is much better measured
    on 272 objects than on 22.
    """
    frames: list[pd.DataFrame] = []
    stats: dict[str, dict] = {}
    for label, df in (populations or {}).items():
        if df is None or len(df) == 0:
            stats[label] = {"n_rows": 0, "n_a1_fitted": 0, "n_negative": 0,
                            "rate": None, "note": "empty population"}
            continue
        a1 = pd.to_numeric(df.get("A1"), errors="coerce") if "A1" in df.columns \
            else pd.Series(dtype=float)
        sig = pd.to_numeric(df.get("sigma_A1"), errors="coerce") \
            if "sigma_A1" in df.columns else pd.Series(np.nan, index=df.index)
        n_fitted = int((a1.notna() & sig.notna() & (sig > 0)).sum())

        if "screen_negative_a1" in df.columns:
            mask = df["screen_negative_a1"].fillna(False).astype(bool)
        else:
            snr = a1 / sig.where(sig > 0)
            mask = ((a1 < 0) & (snr.abs() >= p.neg_a1_snr_min)).fillna(False)
        sub = df[mask].copy()
        sub["population"] = label

        abs_r, d_m, src = [], [], []
        for _, row in sub.iterrows():
            r, d, s = _abs_r(row, p)
            abs_r.append(r)
            d_m.append(d)
            src.append(s)
        sub["abs_R"] = abs_r
        sub["abs_R_diameter_m"] = d_m
        sub["abs_R_diameter_source"] = src
        frames.append(sub)

        vals = np.asarray([v for v in abs_r if np.isfinite(v)], dtype=float)
        stats[label] = {
            "n_rows": int(len(df)),
            "n_a1_fitted": n_fitted,
            "n_negative": int(len(sub)),
            "rate": (float(len(sub)) / n_fitted) if n_fitted else None,
            "denominator": "objects with a fitted A1 AND a positive sigma_A1",
            "abs_R_n": int(vals.size),
            "abs_R_max": float(vals.max()) if vals.size else None,
            "abs_R_median": float(np.median(vals)) if vals.size else None,
            "abs_R_values": [float(v) for v in vals],
        }

    table = pd.concat(frames, ignore_index=True, sort=False) if frames \
        else pd.DataFrame()
    all_vals = np.asarray(
        [v for s in stats.values() for v in (s.get("abs_R_values") or [])],
        dtype=float)
    summary = {
        "populations": stats,
        "n_negative_total": int(len(table)),
        "floor_abs_R_max": float(all_vals.max()) if all_vals.size else None,
        "floor_abs_R_median": float(np.median(all_vals)) if all_vals.size else None,
        "r_flag_threshold": float(p.r_flag),
        "flag_threshold_above_floor": (
            bool(p.r_flag > all_vals.max()) if all_vals.size else None),
        "interpretation": (
            "Radiation pressure cannot push sunward, so every row here is a "
            "systematic. |R| is the size-normalised size of that systematic and "
            "therefore the empirical false-positive floor for screen 1; the "
            "R flag threshold must sit above it."),
    }
    return table, summary


# =============================================================================
# 4. The independent, catalogue-wide high-albedo screen
# =============================================================================
def albedo_strategies(albedo_min: float) -> tuple[tuple[str, str | None], ...]:
    """Constraint syntaxes for ``albedo > albedo_min``, most specific first.

    The SBDB constraint grammar for numeric comparison was never verified from
    the runner (``?info=field`` 400s), so this is a ladder, not an assertion.
    The last two rungs need no comparison operator at all, so the screen cannot
    fail merely because the operator token is spelled differently -- it degrades
    to pulling more rows and filtering client-side.  Every rung's result is
    filtered client-side regardless, so a server-side operator that means
    something *other* than "greater than" cannot leak rows into the survivors.
    """
    return (
        ("cdata_albedo_gt", '{"AND":["albedo|GT|' + str(albedo_min) + '"]}'),
        ("cdata_albedo_range",
         '{"AND":["albedo|RG|' + str(albedo_min) + '|1.5"]}'),
        ("cdata_albedo_defined", '{"AND":["albedo|DF"]}'),
        ("unconstrained_full_pull", None),
    )


@dataclass
class HighAlbedoResult:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = STATUS_QUERY_FAILED
    strategy: str = ""
    n_rows_returned: int = 0
    n_above_cut: int = 0
    n_significant: int = 0
    albedo_sigma_available: bool = False
    fields_rejected: list[str] = field(default_factory=list)
    crosscheck: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "strategy": self.strategy,
                "n_rows_returned": int(self.n_rows_returned),
                "n_above_cut": int(self.n_above_cut),
                "n_significant": int(self.n_significant),
                "albedo_sigma_available": bool(self.albedo_sigma_available),
                "fields_rejected": self.fields_rejected,
                "crosscheck": self.crosscheck,
                "notes": self.notes}


def high_albedo_census(*,
                       params: ScreenParams,
                       kind: str = "a",
                       transport: Transport | None = None,
                       log: QueryLog | None = None,
                       fields: tuple[str, ...] = HIGH_ALBEDO_FIELDS,
                       limit: int | None = None,
                       tries: int = 3,
                       searcher: Callable[[str], pd.DataFrame] | None = None,
                       crosscheck: bool = True,
                       max_crosscheck: int = 500) -> HighAlbedoResult:
    """Every catalogued small body with a geometric albedo above the cut.

    **Independent of A1 by construction.**  Whether an orbit solution happened to
    include a non-gravitational term has nothing to do with whether the body
    reflects 80% of the light that hits it, and screen 4 restricted to the
    22-object A1 sample was answering a much narrower question than the one
    worth asking.

    Fresh ice and E-type enstatite reach ~0.6; > 0.7 is outside the natural
    regolith range and trivial for aluminised film.  The expected dominant
    population is nonetheless **albedo fit artefacts on short arcs**, so the
    orbit-quality columns are carried through so that can be seen rather than
    assumed.
    """
    res = HighAlbedoResult()
    base = {"sb-kind": kind, "full-prec": "1"}
    if limit:
        base["limit"] = limit

    for name, cdata in albedo_strategies(params.albedo_min):
        params_q = dict(base)
        if cdata:
            params_q["sb-cdata"] = cdata
        use_fields = fields
        if cdata is None:
            # A whole-catalogue pull needs a narrow column set or it OOMs.
            use_fields = tuple(f for f in fields if f in {
                "spkid", "full_name", "pdes", "kind", "class", "H", "diameter",
                "albedo", "condition_code", "data_arc", "n_obs_used"})
        df, _meta, err, rejected = query_with_field_pruning(
            use_fields, params_q, transport=transport, log=log, tries=tries,
            label=f"high_albedo:{kind}:{name}")
        res.fields_rejected = sorted(set(res.fields_rejected) | set(rejected))
        if df is None:
            res.notes.append(f"[{name}] query failed: {err}")
            continue
        res.strategy = name
        res.n_rows_returned = int(len(df))
        if "albedo" not in df.columns:
            res.notes.append(
                f"[{name}] response has no albedo column (got {list(df.columns)}); "
                "this is a QUERY defect, not an absence of bright objects")
            continue
        if len(df) == 0:
            res.status = STATUS_ZERO_ROWS
            res.notes.append(
                f"[{name}] the server answered with zero rows. That is a real "
                "answer, not a failure -- but it is only believable if a less "
                "specific rung also returns zero.")
            continue
        break
    else:
        # Careful: "every strategy FAILED" and "every strategy answered with
        # zero rows" are different statements, and the second one is a
        # measurement.  Only overwrite the status if nothing ever answered.
        if res.status != STATUS_ZERO_ROWS:
            res.status = STATUS_QUERY_FAILED
            res.notes.append("every albedo strategy failed; the screen is UNTESTED")
        else:
            res.notes.append(
                "every strategy answered and every answer was empty: no "
                "catalogued object exceeds the albedo cut. That is a "
                "measurement, not a failure.")
        return res

    if "albedo" not in df.columns:
        res.status = STATUS_QUERY_FAILED
        return res

    # ALWAYS filter client-side, whatever the server was asked.  A constraint
    # operator that does not mean what we assumed cannot then leak rows in.
    alb = pd.to_numeric(df["albedo"], errors="coerce")
    out = df[alb >= params.albedo_min].copy()
    res.n_above_cut = int(len(out))
    # ``status`` describes the QUERY, not the screen.  The query succeeded, so
    # it stays OK even when nothing clears the cut -- "no object is that bright"
    # is a result and ``n_above_cut`` is where it is reported.
    res.status = STATUS_OK
    if not len(out):
        res.notes.append(
            f"the query returned {len(df)} rows and NONE has albedo >= "
            f"{params.albedo_min}. The query worked; the sky is what is empty.")

    sigma_col = next((c for c in ("albedo_sigma", "sigma_albedo", "albedo_err")
                      if c in out.columns and pd.to_numeric(
                          out[c], errors="coerce").notna().any()), None)
    res.albedo_sigma_available = sigma_col is not None
    if sigma_col:
        s = pd.to_numeric(out[sigma_col], errors="coerce")
        excess_snr = (pd.to_numeric(out["albedo"], errors="coerce")
                      - params.albedo_min) / s.where(s > 0)
        out["albedo_excess_snr"] = excess_snr
        out["albedo_excess_significant"] = (
            excess_snr >= params.albedo_snr_min).fillna(False)
        res.n_significant = int(out["albedo_excess_significant"].sum())
    else:
        out["albedo_excess_snr"] = np.nan
        out["albedo_excess_significant"] = False
        res.notes.append(
            "SBDB exposes no albedo uncertainty, so the EXCESS over the cut "
            "could not be tested for significance from this source alone. "
            "Recorded as UNTESTED, never as a pass -- the IRSA cross-check "
            "below is what supplies an independent value and its error.")

    if crosscheck and len(out):
        desigs = [str(v) for v in
                  (out["pdes"] if "pdes" in out.columns else out["full_name"])
                  .head(max_crosscheck).tolist()]
        xc = irsa_albedo_crosscheck(desigs, searcher=searcher)
        res.crosscheck = {k: v for k, v in xc.items() if k != "matches"}
        matches = xc.get("matches") or {}
        keys = (out["pdes"] if "pdes" in out.columns else out["full_name"]).astype(str)

        def _lookup(raw, field):
            for k in designation_variants(raw):
                hit = matches.get(k)
                if hit:
                    return hit.get(field)
            return None

        out["neowise_albedo"] = keys.map(lambda v: _lookup(v, "albedo"))
        out["neowise_albedo_sigma"] = keys.map(lambda v: _lookup(v, "albedo_sigma"))
        out["neowise_table"] = xc.get("table") or ""
        out["albedo_crosscheck"] = np.where(
            xc.get("status") != XCHECK_OK, xc.get("status", XCHECK_UNREACHED),
            np.where(out["neowise_albedo"].notna(), "MATCHED", "NO_MATCH"))
        confirmed = (pd.to_numeric(out["neowise_albedo"], errors="coerce")
                     >= params.albedo_min)
        out["albedo_confirmed_two_sources"] = confirmed.fillna(False)
        res.crosscheck["n_matched"] = int(out["neowise_albedo"].notna().sum())
        res.crosscheck["n_confirmed_above_cut"] = int(
            out["albedo_confirmed_two_sources"].sum())
    else:
        out["albedo_crosscheck"] = XCHECK_SKIPPED
        out["albedo_confirmed_two_sources"] = False
        res.crosscheck = {"status": XCHECK_SKIPPED}
        res.notes.append(
            "no IRSA cross-check was run, so every albedo here rests on a "
            "SINGLE source and none of it is confirmed")

    order = [c for c in ("albedo", "albedo_excess_snr") if c in out.columns]
    if order:
        out = out.sort_values(order[0], ascending=False)
    res.table = out.reset_index(drop=True)
    res.notes.append(
        "expected dominant population: albedo fit artefacts on short arcs -- the "
        "orbit-quality columns (condition_code, data_arc, n_obs_used, rms) are "
        "carried so that can be checked rather than assumed")
    return res


# --- IRSA / NEOWISE cross-check ----------------------------------------------
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

#: Substrings that identify a candidate NEOWISE/WISE diameter+albedo table.
_ALBEDO_TABLE_HINTS = ("neowise", "neowiser", "wise_diam", "diam_alb", "nea_diam")
#: Column names, in preference order, that could hold a geometric albedo.
_ALBEDO_COLUMN_HINTS = ("albedo", "pv", "p_v", "geom_albedo")
#: Column names that could hold the object designation.
_DESIG_COLUMN_HINTS = ("pdes", "des", "designation", "name", "object_name",
                       "targetname", "objname", "mpc_desig")


def _default_searcher(query: str) -> pd.DataFrame:  # pragma: no cover - runner-only
    import pyvo

    return pyvo.dal.TAPService(IRSA_TAP_URL).search(query).to_table().to_pandas()


def irsa_albedo_crosscheck(designations: Iterable[str],
                           *,
                           searcher: Callable[[str], pd.DataFrame] | None = None,
                           tap_url: str = IRSA_TAP_URL,
                           max_in_clause: int = 200) -> dict:
    """Independent albedo for each designation from IRSA (NEOWISE).

    A geometric albedo above 0.7 from a single catalogue is a fit artefact until
    a second, independent measurement agrees.  IRSA table names drift between
    releases, so the table and its columns are **discovered** rather than
    hard-coded: a renamed table looks exactly like an unreachable archive, and
    those must never be confused.

    Returns a status dict.  Every failure mode has its own status and none of
    them is ever reported as "checked and agreed".
    """
    search = searcher or _default_searcher
    out: dict = {"tap_url": tap_url, "status": XCHECK_UNREACHED,
                 "table": "", "albedo_column": "", "designation_column": "",
                 "matches": {}, "notes": [], "queries": []}

    def _run(query: str, what: str) -> pd.DataFrame | None:
        out["queries"].append({"what": what, "query": query})
        try:
            return search(query)
        except Exception as exc:  # noqa: BLE001
            out["notes"].append(f"{what} failed: {type(exc).__name__}: {exc}")
            return None

    tables = _run(
        "SELECT table_name FROM TAP_SCHEMA.tables WHERE "
        + " OR ".join(f"table_name LIKE '%{h}%'" for h in _ALBEDO_TABLE_HINTS),
        "table discovery")
    if tables is None or len(tables) == 0:
        out["status"] = XCHECK_UNREACHED if tables is None else XCHECK_NO_TABLE
        out["notes"].append(
            "no NEOWISE-like table was found at IRSA, so every albedo in the "
            "screen rests on a single source and is UNCONFIRMED")
        return out

    names = [str(v) for v in tables[tables.columns[0]].tolist()]
    out["tables_discovered"] = names[:50]

    for table in names:
        cols = _run(
            "SELECT column_name FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '{table}'", f"columns of {table}")
        if cols is None or len(cols) == 0:
            continue
        colnames = [str(v) for v in cols[cols.columns[0]].tolist()]
        lower = {c.lower(): c for c in colnames}
        alb = next((lower[h] for h in _ALBEDO_COLUMN_HINTS if h in lower), None)
        desig = next((lower[h] for h in _DESIG_COLUMN_HINTS if h in lower), None)
        if not alb or not desig:
            continue
        err = next((lower[c] for c in (f"{alb.lower()}_err", f"{alb.lower()}_sigma",
                                       f"sigma_{alb.lower()}", "albedo_err")
                    if c in lower), None)
        out.update({"table": table, "albedo_column": alb,
                    "designation_column": desig, "albedo_sigma_column": err or ""})

        wanted = [d for d in designations if str(d).strip()]
        matches: dict[str, dict] = {}
        for i in range(0, len(wanted), max_in_clause):
            chunk = wanted[i:i + max_in_clause]
            in_clause = ", ".join("'" + str(d).replace("'", "''") + "'" for d in chunk)
            sel = f"{desig}, {alb}" + (f", {err}" if err else "")
            rows = _run(f"SELECT {sel} FROM {table} WHERE {desig} IN ({in_clause})",
                        f"albedo lookup chunk {i // max_in_clause}")
            if rows is None:
                continue
            for _, r in rows.iterrows():
                rec = {
                    "albedo": pd.to_numeric(r.get(alb), errors="coerce"),
                    "albedo_sigma": (pd.to_numeric(r.get(err), errors="coerce")
                                     if err else None),
                    "table": table,
                }
                # Index under every rendering of the designation, so a
                # whitespace convention cannot turn a real match into a silent
                # "no independent albedo".
                for key in designation_variants(r.get(desig)):
                    matches[key] = rec
        out["matches"] = matches
        out["status"] = XCHECK_OK
        out["n_queried"] = len(wanted)
        out["n_matched"] = len(matches)
        return out

    out["status"] = XCHECK_NO_COLUMN
    out["notes"].append(
        "IRSA answered but no discovered table carries both a designation and an "
        "albedo column; the cross-check is UNTESTED, not passed")
    return out


# =============================================================================
# Config plumbing
# =============================================================================
@dataclass
class CensusParams:
    """Knobs for the census stages.  Mirrors ``config/derelict.yaml``."""
    completeness_fields: tuple[str, ...] = COMPLETENESS_FIELDS
    completeness_max_listed: int = 500
    completeness_chunk_classes: tuple[str, ...] = ASTEROID_CLASS_CHUNKS
    completeness_comet_chunk_classes: tuple[str, ...] = COMET_CLASS_CHUNKS
    high_albedo_max_crosscheck: int = 500
    high_albedo_crosscheck: bool = True
    max_queries_in_summary: int = 4000
    dark_comets: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_config(cls, d: dict) -> CensusParams:
        c = (d or {}).get("census", {}) or {}
        dc = (d or {}).get("dark_comets", {}) or {}
        targets: list[tuple[str, str]] = []
        for group, entries in (dc.get("targets") or {}).items():
            for entry in entries or []:
                targets.append((str(entry), str(group)))
        return cls(
            completeness_fields=tuple(c.get("completeness_fields")
                                      or COMPLETENESS_FIELDS),
            completeness_max_listed=int(c.get("completeness_max_listed", 500)),
            completeness_chunk_classes=tuple(c.get("asteroid_class_chunks")
                                             or ASTEROID_CLASS_CHUNKS),
            completeness_comet_chunk_classes=tuple(c.get("comet_class_chunks")
                                                   or COMET_CLASS_CHUNKS),
            high_albedo_max_crosscheck=int(c.get("high_albedo_max_crosscheck", 500)),
            high_albedo_crosscheck=bool(c.get("high_albedo_crosscheck", True)),
            max_queries_in_summary=int(c.get("max_queries_in_summary", 4000)),
            dark_comets=tuple(targets),
        )


def sbdb_query_url(params: dict) -> str:
    """The literal URL a set of SBDB query parameters produces (for the log)."""
    return _build_url(SBDB_QUERY_URL, params)

"""IGNITION's THIRD route to the parent sample: ESA's Gaia x **IRSA's** AllWISE.

Why this module exists
----------------------
``results/ignition/probe.json`` on ``main`` (commit 97c3f99, generated
2026-09-14T02:27Z) records ``gaia_shape_working: null`` and

    inner_cone             TIMED_OUT  QueryTimeout('no answer within 480 s')
    inner_cone_postfilter  TIMED_OUT  QueryTimeout('no answer within 480 s')
    flat                   TIMED_OUT  QueryTimeout('no answer within 420 s')

with the async queue answering ``HTTP 500`` and the run before it
``Error 503 ... maximum number of synchronous queued jobs (150) reached``.

The three shapes differ in *where* the cuts are written, and all three died.
What they have in common is the table: every one of them reaches
``gaiadr1.allwise_original_valid`` --- ESA's ~750-million-row AllWISE mirror ---
through ``gaiadr3.allwise_best_neighbour``.  No rearrangement of the predicates
avoids that table, so no *fourth arrangement* of the same join can help either.
The only shape that can help is one that does not touch it at all.

That shape is :data:`seti.ignition.sample.SHAPE_GAIA_ONLY` (``gaia_only``): an
indexed cone on ``gaiadr3.gaia_source`` alone, carrying every Gaia cut and no
WISE column whatever.  It is not a parent sample by itself --- it has no
W1 and no W2 --- and this module supplies the missing half from the archive
AllWISE actually lives in: **IRSA's own TAP service**,
``https://irsa.ipac.caltech.edu/TAP`` (the same service the channel's NEOWISE
acquisition already uses, :mod:`seti.ignition.acquire`).

What is identical to the ESA route, and what is not
---------------------------------------------------
Identical (this is the whole point --- only the AllWISE *transport* differs):

* **the Gaia cuts, in SQL, at ESA.**  Unlike the VizieR route, this one does
  not hand the Gaia side to a mirror: ``gaia_only`` is built by
  :func:`seti.ignition.sample.build_query` from the same
  :func:`~seti.ignition.sample.gaia_predicates` as every other shape, so the
  cone, the parallax shell, ``|b| > 15``, the dwarf cut and the variability
  flag are the archive's own, expressed exactly as ``inner_cone`` expresses
  them.  ``select_parent`` still re-applies all of them locally;
* **the AllWISE cuts.**  :func:`seti.ignition.vizier_route.apply_allwise_predicates`
  --- the pandas spelling of :func:`seti.ignition.sample.allwise_predicates`,
  ``-0.1 < W1-W2 < 0.15``, ``ext_flag = 0``, ``cc_flags = '0000'`` --- is
  applied to every chunk this route returns, after the match and never before
  it (see below);
* **the column set.**  :func:`seti.ignition.vizier_route.to_parent_frame` emits
  exactly :func:`seti.ignition.sample.parent_columns`, in that order, so no
  stage downstream of ``sample`` can tell the three routes apart;
* **the match**.  :func:`seti.ignition.vizier_route.match_allwise` is reused
  unchanged: the PM-propagated Gaia(2016.0) -> AllWISE(2010.5) positional
  match, ``allwise_sep_arcsec``, ``allwise_n_neighbours``, and an unmatched
  Gaia row dropped exactly as the archive's inner join drops it.

Not identical, and recorded as such:

* **the cross-match**.  As for the VizieR route, there is no
  ``allwise_best_neighbour`` outside the ESA archive, so the separations and
  neighbour counts are *measured here* rather than read from the archive's
  table;
* **the AllWISE cone is fetched UNFILTERED**.  The colour and flag predicates
  are deliberately NOT sent to IRSA in the ``WHERE``.  Cutting the AllWISE side
  before the positional match would let a Gaia star whose true counterpart was
  removed by colour match a *different* source inside the 3 arcsec radius, and
  would make ``allwise_n_neighbours`` --- the blend indicator --- count only
  the sources that survived a cut.  The archive applies its predicates after
  the join; so does this;
* **truncation is measured, not assumed**.  IRSA does have ``COUNT(*)``, so
  every cone is counted as well as fetched and a short read is flagged
  ``truncated`` against the count.  (The VizieR route cannot do this: ASU has
  a row cap and no count.)
* **allsky mode**.  A parallax shell is not a sky chunk, so there is no cone to
  fetch the AllWISE side over.  Rather than fabricate a match, a shell unit is
  recorded ``NOT_SUPPORTED_FOR_MODE`` and contributes nothing.

Asserted and unverified, and how that is settled
------------------------------------------------
The table name ``allwise_p3as_psd`` and every column spelling in
:data:`DEFAULT_IRSA` are **asserted and unverified** (``config/ignition.yaml``
-> ``sample.irsa`` carries the same note, exactly as ``sample.vizier`` does).
There is no network in the sandbox, so nothing here may *assume* they are
right.  :func:`verify_service` asks IRSA itself, at runtime, before a single
science row is fetched:

* ``TAP_SCHEMA.tables`` for every table whose name contains ``allwise``;
* ``TAP_SCHEMA.columns`` for that table's column names, falling back to
  ``SELECT TOP 1 *`` if TAP_SCHEMA is not served.

A table that is not in the service's own list is ``CATALOGUE_NOT_FOUND`` with
the names it *did* return; a column list that cannot be resolved is
``COLUMNS_UNRESOLVED`` with the service's own text.  Neither is a crash and
neither invents a row.

And the distinction the record must never lose: **a service that did not
answer has told you nothing about the sky.**  A refused query is
``QUERY_FAILED``; an answered query with no rows is
``QUERY_RETURNED_ZERO_ROWS``.  ``TAP_SCHEMA`` refusing to answer is
``QUERY_FAILED``, *not* ``CATALOGUE_NOT_FOUND``.
"""

from __future__ import annotations

import pandas as pd

from .sample import (
    DEFAULT_SAMPLE,
    GAIA_TAP,
    QUERY_FAILED,
    QUERY_TIMED_OUT,
    ROUTE_IRSA,
    SHAPE_GAIA_ONLY,
    GaiaQueryFailed,
    build_query,
    run_gaia_query,
    unwrap_result,
)
from .vizier_route import (  # the SECOND route's shared, already-tested machinery
    CUT_ALLWISE,
    REQUIRED_ALLWISE,
    STATUS_COLUMNS,
    STATUS_DISABLED,
    STATUS_FAILED,
    STATUS_NOT_FOUND,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    STATUS_ZERO,
    apply_allwise_predicates,
    match_allwise,
    resolve_columns,
    to_parent_frame,
)

#: IRSA's TAP endpoint --- the spelling already used by
#: ``seti.vigil.acquire.IRSA_TAP`` and ``seti.isotherm.acquire._IRSA_TAP``.
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"

DEFAULT_IRSA: dict = {
    "enabled": True,
    # --- asserted, UNVERIFIED (no network in the sandbox; verify_service asks) ---
    # The AllWISE Source Catalogue at IRSA.  `table_candidates` are the other
    # spellings that have been seen for the same catalogue at TAP services; the
    # probe records which one (if any) the service's own TAP_SCHEMA lists.
    "allwise_table": "allwise_p3as_psd",
    "table_candidates": ["allwise_p3as_psd", "allwise.allwise_p3as_psd",
                         "wise_allwise_p3as_psd", "allwise_p3as_psd_v1"],
    # --- request shape ---
    "cone_pad_arcmin": 1.0,        # the AllWISE cone is the Gaia cone plus this
    "match_radius_arcsec": 3.0,    # Gaia(2016.0 -> 2010.5) against AllWISE
    "max_rows": 400000,            # TOP cap per cone; a cone at the cap is `capped`
    "count_cone": True,            # ask COUNT(*) as well, so truncation is measurable
    "timeout_s": 900.0,
    "allow_sync": True,            # IRSA has no 150-job sync ceiling of ESA's kind
    "w1_max": None,                # optional server-side W1 bound; None = unfiltered
    # --- column names at IRSA, first match wins, case-insensitive (UNVERIFIED) ---
    # Left: the logical name the rest of the channel uses (and that
    # sample.parent_columns() spells).  Right: candidate spellings at IRSA.
    "columns": {
        "allwise_designation": ["designation", "allwise", "source_name"],
        "ra": ["ra", "ra_pm", "raj2000"],
        "dec": ["dec", "dec_pm", "dej2000"],
        "w1mpro": ["w1mpro"],
        "w1mpro_error": ["w1sigmpro"],
        "w2mpro": ["w2mpro"],
        "w2mpro_error": ["w2sigmpro"],
        "w3mpro": ["w3mpro"],
        "w3mpro_error": ["w3sigmpro"],
        "cc_flags": ["cc_flags", "ccf"],
        "ph_qual": ["ph_qual", "qph"],
        "ext_flag": ["ext_flg", "ext_flag", "ex"],
    },
}

#: Everything in :data:`DEFAULT_IRSA` that is an assertion about a service this
#: sandbox cannot reach.  Written into every probe record so the next dispatch
#: reads what was assumed next to what the service actually said.
ASSERTED_UNVERIFIED = ("allwise_table", "table_candidates", "columns")


def irsa_conf(conf: dict | None = None) -> dict:
    """``sample.irsa`` merged over :data:`DEFAULT_IRSA` (one level deep)."""
    got = dict((conf or {}).get("irsa") or {})
    out = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
           for k, v in DEFAULT_IRSA.items()}
    for k, v in got.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def enabled(conf: dict | None = None) -> bool:
    return bool(irsa_conf(conf).get("enabled", True))


def endpoints(conf: dict | None = None) -> dict:
    """The two services this route needs, named even when it is never used."""
    return {"gaia": GAIA_TAP, "allwise": IRSA_TAP}


# ---------------------------------------------------------------------------
# Transport.  The ladder and the time-box are sample.run_gaia_query's, not a
# second copy: only the endpoint and the client change.
# ---------------------------------------------------------------------------
def _lower(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).lower() for c in df.columns})


def _t_pyvo_async(adql: str) -> pd.DataFrame:
    import pyvo  # noqa: PLC0415  runner-only

    return _lower(pyvo.dal.TAPService(IRSA_TAP).run_async(adql).to_table().to_pandas())


def _t_pyvo_sync(adql: str) -> pd.DataFrame:
    import pyvo  # noqa: PLC0415  runner-only

    return _lower(pyvo.dal.TAPService(IRSA_TAP).run_sync(adql).to_table().to_pandas())


#: ``(name, queue, callable)`` in :func:`seti.ignition.sample.run_gaia_query`'s
#: own format, so this route inherits its retries, its per-attempt time-box, its
#: deadline handling and its QUERY_FAILED / QUERY_RETURNED_ZERO_ROWS split.
IRSA_TRANSPORTS: tuple[tuple[str, str, object], ...] = (
    ("pyvo_async", "async", _t_pyvo_async),
    ("pyvo_sync", "sync", _t_pyvo_sync),
)


def query_fn_with_record(*, label: str = "ignition_irsa", timeout_s: float | None = 900.0,
                         deadline: float | None = None, allow_sync: bool = True):
    """An ``adql -> (df, record)`` callable against IRSA; raises on failure."""
    def _fn(adql: str):
        df, rec = run_gaia_query(adql, label=label, timeout_s=timeout_s, deadline=deadline,
                                 allow_sync=allow_sync, transports=IRSA_TRANSPORTS,
                                 tag="ignition")
        if rec["status"] in (QUERY_FAILED, QUERY_TIMED_OUT):
            raise GaiaQueryFailed(rec)
        return df, rec
    return _fn


def _ask(fn, adql: str, *, label: str, service: str) -> tuple[pd.DataFrame, dict]:
    """One time-boxed call through an injected callable.  Never raises.

    ``status`` is :data:`STATUS_OK`, :data:`STATUS_ZERO` or
    :data:`STATUS_FAILED`, and the difference between the last two is the whole
    point: a service that did not answer has said nothing about the sky.
    """
    rec: dict = {"label": label, "service": service, "query": adql.strip()[:4000]}
    try:
        df, qrec = unwrap_result(fn(adql))
    except GaiaQueryFailed as exc:
        rec.update(status=STATUS_FAILED, n_rows=0,
                   error=str(exc.record.get("error") or exc),
                   archive_status=exc.record.get("status"),
                   attempts=exc.record.get("attempts"))
        return pd.DataFrame(), rec
    except Exception as exc:                               # noqa: BLE001
        rec.update(status=STATUS_FAILED, n_rows=0, error=repr(exc))
        return pd.DataFrame(), rec
    df = _lower(df if df is not None else pd.DataFrame())
    for k in ("transport", "queue", "seconds"):
        if qrec.get(k) is not None:
            rec[k] = qrec[k]
    rec.update(status=(STATUS_OK if len(df) else STATUS_ZERO), n_rows=int(len(df)))
    return df, rec


def _irsa_fn(conf: dict, fetch_fn=None, deadline: float | None = None):
    ic = irsa_conf(conf)
    return fetch_fn or query_fn_with_record(timeout_s=float(ic.get("timeout_s") or 900.0),
                                            allow_sync=bool(ic.get("allow_sync", True)),
                                            deadline=deadline)


def _gaia_fn(conf: dict, query_fn=None, deadline: float | None = None):
    if query_fn is not None:
        return query_fn
    from .sample import query_fn_with_record as gaia_query_fn_with_record

    c = {**DEFAULT_SAMPLE, **(conf or {})}
    return gaia_query_fn_with_record(label="ignition_irsa_gaia", allow_sync=False,
                                     timeout_s=float(c.get("query_timeout_s") or 1200.0),
                                     deadline=deadline)


# ---------------------------------------------------------------------------
# Verification: ask the service its own names before trusting any of ours
# ---------------------------------------------------------------------------
def discover_tables(conf: dict | None = None, *, fetch_fn=None) -> tuple[list[str], dict]:
    """Every table at IRSA whose name contains ``allwise``, as IRSA spells it."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    adql = ("SELECT table_name FROM TAP_SCHEMA.tables "
            "WHERE table_name LIKE '%allwise%' OR table_name LIKE '%ALLWISE%'")
    df, rec = _ask(_irsa_fn(c, fetch_fn), adql, label="irsa_tables", service=IRSA_TAP)
    if not len(df):
        return [], rec
    col = "table_name" if "table_name" in df.columns else df.columns[0]
    names = sorted({str(v).strip() for v in df[col] if str(v).strip()})
    return names, rec


def discover_columns(table: str, conf: dict | None = None, *,
                     fetch_fn=None) -> tuple[list[str], dict]:
    """``table``'s column names as the service reports them.

    ``TAP_SCHEMA.columns`` first; a service that does not serve TAP_SCHEMA is
    asked the table itself with ``SELECT TOP 1 *``.  Both are recorded, and the
    returned status distinguishes "did not answer" from "answered, named
    nothing".
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    fn = _irsa_fn(c, fetch_fn)
    safe = str(table).replace("'", "''")
    df, rec = _ask(fn, f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name = '{safe}'",
                   label="irsa_columns", service=IRSA_TAP)
    if len(df):
        col = "column_name" if "column_name" in df.columns else df.columns[0]
        names = [str(v).strip() for v in df[col] if str(v).strip()]
        if names:
            return names, rec
    peek, prec = _ask(fn, f"SELECT TOP 1 * FROM {table}", label="irsa_columns_peek",
                      service=IRSA_TAP)
    rec = {**rec, "fallback": prec}
    names = [str(x) for x in peek.columns]
    if names:
        rec["status"] = STATUS_OK
    elif prec["status"] == STATUS_FAILED and rec.get("status") == STATUS_FAILED:
        rec["status"] = STATUS_FAILED
    else:
        # Both answered and neither named a column: that IS an answer, and it is
        # not a transport failure.
        rec["status"] = STATUS_ZERO
    return names, rec


def verify_service(conf: dict | None = None, *, fetch_fn=None) -> dict:
    """Settle the asserted table and column names against IRSA's own answer.

    Returns a record with ``status`` one of :data:`STATUS_OK`,
    :data:`STATUS_FAILED` (the service did not answer --- nothing has been
    learned about the table), :data:`STATUS_NOT_FOUND` (it answered and the
    table is not among its own tables) or :data:`STATUS_COLUMNS` (the table is
    there and the columns this route cannot do without are not).
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    ic = irsa_conf(c)
    wanted = [str(ic["allwise_table"])] + [str(x) for x in (ic.get("table_candidates") or [])]
    rep: dict = {"service": IRSA_TAP, "table_requested": str(ic["allwise_table"]),
                 "table_candidates": wanted,
                 "asserted_unverified": list(ASSERTED_UNVERIFIED),
                 "verified": False, "table": None}
    names, trec = discover_tables(c, fetch_fn=fetch_fn)
    rep["tables_query"] = trec
    rep["tables_seen"] = names[:80]
    if trec["status"] == STATUS_FAILED:
        rep.update(status=STATUS_FAILED, error=trec.get("error"),
                   reason=("IRSA's TAP_SCHEMA did not answer, so the service has said "
                           "NOTHING about whether the AllWISE table is there: this is "
                           "QUERY_FAILED and must never be recorded as "
                           "CATALOGUE_NOT_FOUND"))
        return rep
    have = {n.lower(): n for n in names}
    table = next((have[w.lower()] for w in wanted if w.lower() in have), None)
    if table is None:
        rep.update(status=STATUS_NOT_FOUND,
                   error=(f"none of {wanted} is in IRSA's own TAP_SCHEMA.tables; the service "
                          f"answered with {names[:40]!r}"))
        return rep
    rep["table"] = table
    cols, crec = discover_columns(table, c, fetch_fn=fetch_fn)
    rep["columns_query"] = crec
    rep["columns_seen"] = cols[:120]
    got, missing = resolve_columns(cols, ic["columns"])
    rep["columns_resolved"] = got
    rep["columns_missing"] = missing
    hard = [k for k in REQUIRED_ALLWISE if k in missing]
    if not cols and crec.get("status") == STATUS_FAILED:
        rep.update(status=STATUS_FAILED,
                   error=(f"IRSA did not answer when asked for {table}'s column names: "
                          f"{crec.get('error')}"))
        return rep
    if hard:
        rep.update(status=STATUS_COLUMNS,
                   error=(f"required AllWISE columns unresolved on {table} at {IRSA_TAP}: "
                          f"{hard}; the service's own column list is {cols[:60]!r}"
                          + (f"; {crec.get('error')}" if crec.get("error") else "")))
        return rep
    rep.update(status=STATUS_OK, verified=True,
               cuts_not_applied=[CUT_ALLWISE[k] for k in CUT_ALLWISE if k in missing])
    return rep


# ---------------------------------------------------------------------------
# The two halves of one chunk
# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    v = float(x)
    return str(int(v)) if v == int(v) else f"{v:g}"


def allwise_adql(conf: dict | None = None, *, field: dict, table: str, columns: dict,
                 max_rows: int | None = None, count_only: bool = False) -> str:
    """The AllWISE cone at IRSA, in the service's own column spellings.

    **No colour and no flag predicate goes in this WHERE.**  They are applied
    by :func:`seti.ignition.vizier_route.apply_allwise_predicates` *after* the
    positional match, because cutting the AllWISE side first would change which
    source a Gaia star matches inside the radius and would make
    ``allwise_n_neighbours`` count only the survivors of a cut.  The archive
    joins first and filters second; so does this.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    ic = irsa_conf(c)
    pad = float(ic.get("cone_pad_arcmin") or 0.0) / 60.0
    radius = float(field["radius_deg"]) + pad
    ra_col, dec_col = columns["ra"], columns["dec"]
    where = [f"1 = CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), "
             f"CIRCLE('ICRS', {_fmt(field['ra'])}, {_fmt(field['dec'])}, {_fmt(radius)}))"]
    w1_max = ic.get("w1_max")
    if w1_max is not None and columns.get("w1mpro"):
        where.append(f"{columns['w1mpro']} < {_fmt(w1_max)}")
    clause = "\n  AND ".join(where)
    if count_only:
        return f"SELECT COUNT(*) AS n\nFROM {table}\nWHERE {clause}"
    top = f"TOP {int(max_rows)} " if max_rows else ""
    sel = ", ".join(dict.fromkeys(columns[k] for k in columns))
    return f"SELECT {top}{sel}\nFROM {table}\nWHERE {clause}"


def fetch_gaia_chunk(conf: dict | None = None, *, field: dict | None = None,
                     plx_lo: float | None = None, plx_hi: float | None = None,
                     cap: int | None = None, query_fn=None) -> tuple[pd.DataFrame, dict]:
    """The ``gaia_only`` shape for one chunk, at the ESA archive."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    adql = build_query(c, field=field, plx_lo=plx_lo, plx_hi=plx_hi, cap=cap,
                       shape=SHAPE_GAIA_ONLY)
    out, rec = _ask(_gaia_fn(c, query_fn), adql, label="irsa_route_gaia_only", service=GAIA_TAP)
    rec["shape"] = SHAPE_GAIA_ONLY
    rec["cap"] = int(cap) if cap else None
    rec["capped"] = bool(cap and len(out) >= int(cap))
    if "source_id" in out:
        out = out.copy()
        out["source_id"] = out["source_id"].astype(str).str.strip()
    return out, rec


def fetch_allwise_cone(conf: dict | None = None, *, field: dict, verified: dict,
                       fetch_fn=None) -> tuple[pd.DataFrame, dict]:
    """The AllWISE side of one cone at IRSA, mapped onto the ESA column names."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    ic = irsa_conf(c)
    table = str(verified["table"])
    got = dict(verified.get("columns_resolved") or {})
    max_rows = int(ic.get("max_rows") or 0) or None
    fn = _irsa_fn(c, fetch_fn)
    rec: dict = {"service": IRSA_TAP, "table": table, "max_rows": max_rows,
                 "columns_resolved": got, "count_star": None}
    if ic.get("count_cone", True):
        cdf, crec = _ask(fn, allwise_adql(c, field=field, table=table, columns=got,
                                          count_only=True),
                         label="irsa_allwise_count", service=IRSA_TAP)
        rec["count_query"] = crec
        if crec["status"] == STATUS_OK and len(cdf):
            try:
                rec["count_star"] = int(pd.to_numeric(cdf.iloc[0, 0]))
            except Exception:                              # noqa: BLE001
                rec["count_star"] = None
    raw, qrec = _ask(fn, allwise_adql(c, field=field, table=table, columns=got,
                                      max_rows=max_rows),
                     label="irsa_allwise_cone", service=IRSA_TAP)
    rec["cone_query"] = qrec
    if qrec["status"] == STATUS_FAILED:
        rec.update(status=STATUS_FAILED, n_rows=0, error=qrec.get("error"))
        return pd.DataFrame(), rec
    have = set(raw.columns)
    missing = [k for k, actual in got.items() if str(actual).lower() not in have]
    if [k for k in REQUIRED_ALLWISE if k in missing]:
        rec.update(status=STATUS_COLUMNS, n_rows=int(len(raw)),
                   columns_missing=missing,
                   error=(f"{table} answered without {missing!r}; the rows it returned carry "
                          f"{sorted(have)[:40]!r}"))
        return pd.DataFrame(), rec
    out = pd.DataFrame({logical: raw[str(actual).lower()]
                        for logical, actual in got.items() if str(actual).lower() in have})
    rec["columns_missing"] = missing
    rec["n_rows"] = int(len(out))
    rec["capped"] = bool(max_rows and len(out) >= max_rows)
    rec["truncated"] = bool(rec["count_star"] is not None and len(out) < rec["count_star"])
    rec["cuts_not_applied"] = [CUT_ALLWISE[k] for k in CUT_ALLWISE if k in missing]
    rec["status"] = STATUS_OK if len(out) else STATUS_ZERO
    return out, rec


def _match_conf(conf: dict) -> dict:
    """``conf`` with the IRSA match radius where :func:`match_allwise` reads it.

    ``match_allwise`` is the VizieR route's, reused unchanged --- it takes its
    radius from ``sample.vizier.match_radius_arcsec``.  Rather than fork the
    matcher, this route hands it a conf whose radius is
    ``sample.irsa.match_radius_arcsec``.
    """
    ic = irsa_conf(conf)
    vz = dict(conf.get("vizier") or {})
    vz["match_radius_arcsec"] = float(ic["match_radius_arcsec"])
    return {**conf, "vizier": vz}


# ---------------------------------------------------------------------------
# One chunk, end to end
# ---------------------------------------------------------------------------
def fetch_unit(conf: dict | None = None, unit: dict | None = None, *, cap: int | None = None,
               query_fn=None, fetch_fn=None, label: str = "",
               verified: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One chunk of the parent sample over the IRSA route.

    Returns ``(rows, record)``.  ``rows`` carries exactly
    :func:`seti.ignition.sample.parent_columns` and has already passed the
    AllWISE predicates the ESA route writes in SQL; the Gaia-only cuts are both
    in the ``gaia_only`` SQL and re-applied by ``select_parent``, which the
    caller runs over every route's frame alike.

    ``verified`` short-circuits :func:`verify_service` with an earlier unit's
    answer; ``query_fn`` and ``fetch_fn`` are the ESA and IRSA transports and
    are always injectable, so nothing here reaches the network in a test.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    unit = dict(unit or {})
    field = unit.get("field")
    rec: dict = {"route": ROUTE_IRSA, "label": label, "unit": unit,
                 "endpoints": endpoints(c)}
    if not enabled(c):
        rec.update(status=STATUS_DISABLED, n_rows=0,
                   reason="config sample.irsa.enabled is false")
        return pd.DataFrame(), rec
    if field is None:
        rec.update(status=STATUS_UNSUPPORTED, n_rows=0,
                   reason=("a parallax shell is not a sky chunk, so there is no cone to pull "
                           "the AllWISE side over; the IRSA route serves mode=fields only and "
                           "will not fabricate a cross-match for mode=allsky"))
        return pd.DataFrame(), rec

    v = dict(verified) if (verified and verified.get("verified")) else \
        verify_service(c, fetch_fn=fetch_fn)
    rec["verify"] = v
    if v.get("status") != STATUS_OK:
        rec.update(status=v.get("status", STATUS_FAILED), n_rows=0,
                   error=v.get("error") or v.get("reason") or v.get("status"))
        return pd.DataFrame(), rec

    gaia, grec = fetch_gaia_chunk(c, field=field, cap=cap, query_fn=query_fn)
    rec["gaia"] = grec
    if grec["status"] != STATUS_OK:
        rec.update(status=grec["status"], n_rows=0,
                   error=grec.get("error") or grec["status"])
        return pd.DataFrame(), rec

    wise, wrec = fetch_allwise_cone(c, field=field, verified=v, fetch_fn=fetch_fn)
    rec["allwise"] = wrec
    if wrec["status"] != STATUS_OK:
        rec.update(status=wrec["status"], n_rows=0,
                   error=wrec.get("error") or wrec["status"])
        return pd.DataFrame(), rec

    joined, mrec = match_allwise(gaia, wise, _match_conf(c))
    rec["match"] = mrec
    kept, counters = apply_allwise_predicates(joined, c,
                                              applied=dict(v.get("columns_resolved") or {}))
    rec["allwise_cut_counters"] = counters
    out = to_parent_frame(kept, c)
    rec["n_rows"] = int(len(out))
    rec["capped"] = bool(grec.get("capped") or wrec.get("capped"))
    rec["truncated"] = bool(wrec.get("truncated"))
    rec["cuts_not_applied"] = sorted(set(list(v.get("cuts_not_applied") or [])
                                         + list(wrec.get("cuts_not_applied") or [])
                                         + list(counters.get("cuts_not_applied") or [])))
    rec["status"] = STATUS_OK if len(out) else STATUS_ZERO
    rec["row_count_note"] = (
        "the Gaia half is the gaia_only shape at ESA, capped by its own TOP; the AllWISE "
        "half is an IRSA cone with COUNT(*) alongside it, so `truncated` is measured rather "
        "than assumed. n_rows is what this chunk actually returned and is not a COUNT(*) of "
        "the parent selection.")
    return out, rec


def probe_route(conf: dict | None = None, *, query_fn=None, fetch_fn=None, cap: int = 5
                ) -> tuple[dict, pd.DataFrame]:
    """One minimal call per service, so ``probe.json`` names this route too.

    Settles the asserted table and column names against IRSA's own TAP_SCHEMA,
    sends the ``gaia_only`` shape to ESA (the fourth shape --- the one that
    touches no AllWISE table), and pulls a handful of rows of the first
    configured field end to end.  Returns ``(report, rows)``; the rows let the
    probe test the NEOWISE routes on a real star even when neither of the two
    earlier parent routes answered.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    ic = irsa_conf(c)
    fields = list(c.get("fields") or [])
    field = fields[0] if fields else {"ra": 266.0, "dec": 65.0, "radius_deg": 0.1}
    rep: dict = {"route": ROUTE_IRSA, "enabled": bool(ic.get("enabled", True)),
                 "endpoints": endpoints(c),
                 "table_requested": str(ic["allwise_table"]),
                 "asserted_unverified": list(ASSERTED_UNVERIFIED),
                 "gaia_shape": SHAPE_GAIA_ONLY, "field": field}
    if not rep["enabled"]:
        rep.update(status=STATUS_DISABLED, usable=False, n_rows=0)
        return rep, pd.DataFrame()
    rep["verify"] = verify_service(c, fetch_fn=fetch_fn)
    rows, urec = fetch_unit(c, {"field": field}, cap=int(cap), query_fn=query_fn,
                            fetch_fn=fetch_fn, label="probe",
                            verified=rep["verify"])
    rep["parent_chunk"] = urec
    # The gaia_only shape is probed WHATEVER IRSA said: whether ESA will serve a
    # cone on gaia_source alone is the fact this whole route turns on, and it is
    # worth recording even on a night IRSA is down.  `fetch_unit` verifies the
    # AllWISE side first and stops there when it cannot, so on that night the
    # query has not been sent yet --- and exactly once either way.
    grec = urec.get("gaia")
    if grec is None:
        _g, grec = fetch_gaia_chunk(c, field=field, cap=int(cap), query_fn=query_fn)
    rep["gaia"] = grec
    rep["table"] = (rep["verify"] or {}).get("table")
    rep["columns_resolved"] = (rep["verify"] or {}).get("columns_resolved")
    rep["n_rows"] = int(len(rows))
    rep["status"] = urec.get("status", STATUS_FAILED)
    rep["gaia_only_status"] = grec.get("status")
    rep["usable"] = rep["status"] in (STATUS_OK, STATUS_ZERO)
    return rep, rows


__all__ = ["ASSERTED_UNVERIFIED", "DEFAULT_IRSA", "IRSA_TAP", "IRSA_TRANSPORTS", "ROUTE_IRSA",
           "STATUS_COLUMNS", "STATUS_DISABLED", "STATUS_FAILED", "STATUS_NOT_FOUND",
           "STATUS_OK", "STATUS_UNSUPPORTED", "STATUS_ZERO",
           "allwise_adql", "discover_columns", "discover_tables", "enabled", "endpoints",
           "fetch_allwise_cone", "fetch_gaia_chunk", "fetch_unit", "irsa_conf", "probe_route",
           "query_fn_with_record", "verify_service"]

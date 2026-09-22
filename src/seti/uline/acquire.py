"""Runner-only acquisition for ULINE: JPL, CDMS and VizieR U-line tables.

Nothing here runs in a test: every network function takes an injectable
``fetch_fn(url) -> str`` or ``query_fn(adql) -> DataFrame`` so the offline
suite can script a healthy, an empty and a dead archive.  The pure pieces
(species-directory parsing, link discovery, column-role resolution, the
unidentified flag) are what the tests exercise.

Discipline inherited from ``seti.metronome.acquire``: every fetch is logged
with a status (``OK`` / ``QUERY_FAILED`` / ``QUERY_RETURNED_ZERO_ROWS``); a
source that fails is *recorded*, never faked; and no column is queried by a
name that was not seen in ``TAP_SCHEMA.columns``.

Archive specifics
-----------------
* **JPL**: ``catdir.cat`` is the species directory (tag, name, line count,
  log Q on the 7-temperature grid); each species is ``c<tag:06d>.cat``.
* **CDMS**: the classic *entries* page is script-rendered (the necrofrontier
  probe saw no species in its HTML), so the roots in ``config/uline.yaml``
  are fetched and *whatever* comes back is mined by regex for
  ``c<tag>.cat`` links and for a partition-function table (tag, name,
  log Q).  Nothing beyond the roots is hard-coded.
* **VizieR TAP**: table names in ``TAP_SCHEMA`` carry literal double quotes,
  so discovery uses ``LIKE '%J/ApJ/787/112/%'`` (leading ``%``) and the
  selected table is quoted in ``FROM``.  Roles (frequency, identification,
  intensity, error) are resolved from ``TAP_SCHEMA.columns`` by regex, with
  the unit column deciding MHz vs GHz.
"""

from __future__ import annotations

import json
import re
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..metronome.acquire import (
    ROUTE_ASTROQUERY,
    ROUTE_ASU,
    ROUTE_NONE,
    ROUTE_TAP,
    VIZIER_ASU,
    VIZIER_ASU_MIRRORS,
    VIZIER_TAP_MIRRORS,
    AdqlNotTranslatable,
    VizierResult,
    VizierRouteError,
    asu_catalogue_tables,
    reset_route_state,
    route_log_summary,
    vizier_table,
)
from .lines import (
    Entry,
    SpeciesMatch,
    match_species,
    parse_cat,
    parse_catdir_report,
    parse_partition_table,
    unparsed_catdir_lines,
)

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

STATUS_OK = "OK"
STATUS_FAILED = "QUERY_FAILED"
STATUS_ZERO = "QUERY_RETURNED_ZERO_ROWS"


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------
@dataclass
class AcquisitionLog:
    stages: list[dict] = field(default_factory=list)
    prefix: str = "uline/acquire"

    def record(self, stage: str, what: str, *, rows: int | None = None,
               error: str | None = None, extra: dict | None = None) -> str:
        if error is not None or rows is None:
            status = STATUS_FAILED
        elif rows == 0:
            status = STATUS_ZERO
        else:
            status = STATUS_OK
        rec = {"stage": stage, "status": status, "rows": int(rows or 0), "what": str(what)[:2000]}
        if error:
            rec["error"] = str(error)[:2000]     # the FULL text: a bare status is undiagnosable
        if extra:
            rec.update(extra)
        self.stages.append(rec)
        print(f"[{self.prefix}] {stage}: {status} rows={rec['rows']}"
              + (f" error={rec.get('error')}" if error else ""))
        return status

    def as_dict(self) -> dict:
        n_fail = sum(1 for s in self.stages if s["status"] == STATUS_FAILED)
        n_zero = sum(1 for s in self.stages if s["status"] == STATUS_ZERO)
        n_ok = sum(1 for s in self.stages if s["status"] == STATUS_OK)
        return {"stages": self.stages, "n_stages": len(self.stages), "n_ok": n_ok,
                "n_query_failed": n_fail, "n_query_returned_zero_rows": n_zero,
                "any_query_failed": bool(n_fail > 0)}

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
def http_text(url: str, *, timeout: float = 180.0) -> str:
    """GET a URL and return its text (runner only)."""
    import requests  # noqa: PLC0415  runner-only; keeps the module importable offline

    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "seti-uline/1.0 (+github actions; astronomy)"})
    r.raise_for_status()
    return r.text


def fetch_text(url: str, *, fetch_fn=None, retries: int = 3, timeout: float = 180.0,
               log: AcquisitionLog | None = None, stage: str = "fetch",
               errors: list | None = None) -> str | None:
    """Fetch with retries; ``None`` (and a logged failure) when every attempt fails.

    ``errors``, when given, receives the ``repr`` of the **last** exception.
    Without it the caller learns only that the fetch returned ``None``, and
    this channel's whole absence discipline is that a door that did not open
    is recorded with the reason it gave — "403" and "404" and "connection
    reset" are three different statements about an archive, and collapsing
    them into "no body" throws the diagnosis away.
    """
    fn = fetch_fn or (lambda u: http_text(u, timeout=timeout))
    last = None
    for attempt in range(max(int(retries), 1)):
        try:
            text = fn(url)
            if text is None:
                raise RuntimeError("fetch returned None")
            text = str(text)
            if log is not None:
                log.record(stage, url, rows=len(text.splitlines()),
                           extra={"bytes": len(text.encode("utf-8", "replace"))})
            return text
        except Exception as exc:                              # noqa: BLE001
            last = exc
            print(f"[uline/acquire] {stage} attempt {attempt + 1}/{retries} failed: {exc!r}")
            if attempt + 1 < retries:
                _time.sleep(2.0 * (attempt + 1))
    if log is not None:
        log.record(stage, url, error=repr(last))
    if errors is not None:
        errors.append(repr(last))
    return None


def tap_endpoints(url: str = VIZIER_TAP) -> list[str]:
    """The configured TAP host first, then every other host in the ladder."""
    return [url] + [u for u in VIZIER_TAP_MIRRORS if u != url]


def tap_query(adql: str, *, url: str = VIZIER_TAP, retries: int = 3, fetch_fn=None,
              allow_non_tap: bool = True) -> pd.DataFrame:
    """ADQL against VizieR TAP, then every other route (runner only).

    Route ladder (each attempt recorded with its endpoint and error text):

    1. the configured TAP host (``url``), async then sync, ``retries`` times;
    2. every other TAP host in :data:`VIZIER_TAP_MIRRORS` --- a second CDS
       hostname and the CfA mirror;
    3. VizieR's NON-TAP ASU interface, by translating the ADQL
       (:func:`seti.metronome.acquire.asu_query`);
    4. ``astroquery.vizier`` where the query is a plain table pull.

    On 2026-09-13 TAPVizieR answered 503 to five attempts on each of two
    spellings and both this channel and ARC reported ``NO_DATA_REACHED`` for an
    infrastructure reason.  Routes 2-4 exist so that outcome needs the whole of
    VizieR to be down, not one service.  Every route failing still raises, with
    every endpoint and every error in the message --- never a fabricated row.
    """
    from ..metronome import acquire as _m  # noqa: PLC0415  (shared route ladder)

    return _m.tap_query(adql, url=tap_endpoints(url), retries=retries, fetch_fn=fetch_fn,
                        allow_non_tap=allow_non_tap)


# ---------------------------------------------------------------------------
# JPL
# ---------------------------------------------------------------------------
def match_all_species(entries: list[Entry], species_conf: dict) -> dict[str, SpeciesMatch]:
    """{species: :class:`SpeciesMatch`} for every configured species.

    Matching is by NORMALISED FORMULA first (the configured regexes are an
    additional route, never the only one) and every near miss is kept, so a
    species that still finds nothing reports the catalogue's own spellings.
    """
    out: dict[str, SpeciesMatch] = {}
    for group in ("targets", "baseline", "contaminants"):
        for sp, spec in (species_conf.get(group) or {}).items():
            formula = str((spec or {}).get("formula") or sp)
            out[sp] = match_species(entries, formula, (spec or {}).get("patterns") or [])
    return out


def jpl_inventory(catdir_text: str, species_conf: dict) -> tuple[dict[str, list[Entry]], dict]:
    """({species: [catdir entries]}, report) for every configured species."""
    entries, unparsed = parse_catdir_report(catdir_text)
    matches = match_all_species(entries, species_conf)
    inv = {sp: m.entries for sp, m in matches.items()}
    rep = {"n_entries": len(entries), "n_unparsed_lines": len(unparsed),
           "unparsed_sample": unparsed_catdir_lines(catdir_text),
           "matched_names": {sp: m.matched_names for sp, m in matches.items()},
           "near_miss_names": {sp: m.near_miss_names for sp, m in matches.items()}}
    return inv, rep


# ---------------------------------------------------------------------------
# CDMS
# ---------------------------------------------------------------------------
_CAT_LINK_RE = re.compile(r"c(\d{6})\.cat", re.IGNORECASE)
_LOOSE_TAG_NAME_RE = re.compile(r"(?<![\d.])(\d{6})[ \t]+([A-Za-z][A-Za-z0-9\-+,=()]{0,25})")


def cdms_cat_tags(html: str) -> list[int]:
    """Every ``c<tag>.cat`` referenced in a page, in order of first appearance."""
    seen: dict[int, None] = {}
    for m in _CAT_LINK_RE.finditer(html or ""):
        seen.setdefault(int(m.group(1)), None)
    return list(seen)


def cdms_inventory(pages: dict[str, str], species_conf: dict) -> tuple[list[Entry], dict]:
    """Mine the fetched CDMS root pages for (tag, name, log Q) entries.

    A partition-function table (any page with ``lg(Q(T))`` headers or rows of
    ``tag name nlines q…``) gives entries with a Q grid; a page with only
    ``c<tag>.cat`` links gives tags without names, which are recorded but
    cannot be attributed to a species.  Returns the entries and a report of
    what each page yielded.
    """
    entries: dict[int, Entry] = {}
    report: dict = {}
    for url, text in (pages or {}).items():
        if text is None:
            report[url] = {"status": STATUS_FAILED}
            continue
        parsed = parse_partition_table(text, database="cdms")
        tags = cdms_cat_tags(text)
        for e in parsed:
            entries.setdefault(e.tag, e)
        # loose tag/name pairs (entries page, if it ever renders server-side)
        n_loose = 0
        if not parsed:
            from .lines import _strip_html
            plain = _strip_html(text)
            for m in _LOOSE_TAG_NAME_RE.finditer(plain):
                tag, name = int(m.group(1)), m.group(2).strip()
                if tag not in entries and name:
                    entries[tag] = Entry(tag=tag, name=name, nlines=0, temps=[], qlog=[],
                                         database="cdms")
                    n_loose += 1
        report[url] = {"status": STATUS_OK if (parsed or tags or n_loose) else STATUS_ZERO,
                       "n_partition_entries": len(parsed), "n_cat_links": len(tags),
                       "n_loose_tag_names": n_loose, "bytes": len(text)}
    allentries = list(entries.values())
    matches = match_all_species(allentries, species_conf)
    inv = {sp: m.entries for sp, m in matches.items()}
    return allentries, {"pages": report, "n_entries": len(allentries), "species": inv,
                        "matched_names": {sp: m.matched_names for sp, m in matches.items()},
                        "near_miss_names": {sp: m.near_miss_names for sp, m in matches.items()}}


# ---------------------------------------------------------------------------
# .cat fetching
# ---------------------------------------------------------------------------
def fetch_cats(entries: list[Entry], url_template: str, *, fetch_fn=None,
               log: AcquisitionLog | None = None, retries: int = 3,
               database: str = "jpl") -> tuple[pd.DataFrame, list[dict]]:
    """Fetch and parse every entry's ``.cat``; concatenated table + per-entry report."""
    frames, reps = [], []
    for e in entries:
        url = url_template.format(tag=int(e.tag))
        text = fetch_text(url, fetch_fn=fetch_fn, retries=retries, log=log,
                          stage=f"cat_{database}_{e.tag}")
        rep = {"tag": int(e.tag), "name": e.name, "database": database, "url": url}
        if text is None:
            rep.update({"status": STATUS_FAILED, "n_lines": 0})
            reps.append(rep)
            continue
        df = parse_cat(text, tag=int(e.tag))
        df["entry_id"] = f"{database}:{e.tag}"
        df["entry_name"] = e.name
        df["database"] = database
        rep.update({"status": STATUS_OK if len(df) else STATUS_ZERO, "n_lines": int(len(df)),
                    "has_partition_function": bool(e.qlog)
                    and bool(np.isfinite(np.asarray(e.qlog, dtype=float)).any())})
        reps.append(rep)
        if len(df):
            frames.append(df)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return table, reps


# ---------------------------------------------------------------------------
# VizieR: table discovery and column-role resolution (pure parts)
# ---------------------------------------------------------------------------
DEFAULT_COLUMN_PATTERNS: dict[str, list[str]] = {
    "freq": [r"^freq(uency)?$", r"^nu$", r"^f$", r"^freq_?rest$", r"^frest$", r"^fobs$",
             r"^freqobs$", r"^freq_?obs$", r"^nu_?rest$", r"^frequ$", r"^freq[0-9]$",
             r"^rest_?freq$", r"^f_?rest$", r"^fcen$", r"^f_?obs$", r"^freq_?mhz$",
             r"^freq_?ghz$", r"^nu_?ghz$", r"^nu_?mhz$"],
    "freq_err": [r"^e_?freq(uency)?$", r"^e_?nu$", r"^e_?f$", r"^freq_?err$", r"^efreq$",
                 r"^err_?freq$", r"^dfreq$", r"^e_?frest$"],
    "ident": [r"^species$", r"^mol(ecule)?$", r"^ident(ification)?$", r"^id$", r"^name$",
              r"^assign(ment)?$", r"^line$", r"^formula$", r"^molec$", r"^spec$",
              r"^carrier$", r"^mole$", r"^spname$", r"^molname$"],
    "intensity": [r"^t_?a\*?$", r"^ta$", r"^tmb$", r"^t_?mb$", r"^t_?peak$", r"^tpk$",
                  r"^peak$", r"^tpeak$", r"^t_?max$", r"^i_?peak$", r"^ipeak$", r"^sp$",
                  r"^int(ensity)?$", r"^w$", r"^area$", r"^flux$", r"^s$", r"^snu$",
                  r"^tb$", r"^t$", r"^amp$", r"^tant$", r"^tr$", r"^t_?r$", r"^w_?int$"],
    "transition": [r"^trans(ition)?$", r"^qn$", r"^qns$", r"^quantum$", r"^levels$",
                   r"^qnum$"],
}

#: A value in the identification column that means "unidentified".
DEFAULT_ULINE_PATTERNS: list[str] = [r"^\s*u\s*$", r"^\s*u[-_ ]?line", r"^\s*unid", r"^\s*\?+\s*$",
                                     r"^\s*unknown", r"^\s*u\s*\d*\s*$", r"^\s*u\s*[-:]"]


def _canon(name: str) -> str:
    return re.sub(r"_+", "_", str(name).strip().lower()).strip("_")


def resolve_line_columns(columns: pd.DataFrame | list, patterns: dict | None = None
                         ) -> dict[str, str | None]:
    """{role: original column name or None} by exact canonicalised regex."""
    patterns = patterns or DEFAULT_COLUMN_PATTERNS
    if isinstance(columns, pd.DataFrame):
        names = [str(c) for c in columns["column_name"].tolist()]
    else:
        names = [str(c) for c in columns]
    canon = {_canon(c): c for c in names}
    out: dict[str, str | None] = {}
    for role, pats in patterns.items():
        hit = None
        for p in pats:
            rx = re.compile(p)
            hit = next((orig for cc, orig in canon.items() if rx.match(cc)), None)
            if hit is not None:
                break
        out[role] = hit
    return out


def frequency_scale(unit: str | None, sample=None) -> tuple[float, str]:
    """Multiplier to MHz from the column unit, else from the magnitude of a sample."""
    u = (unit or "").strip().lower()
    if "ghz" in u:
        return 1000.0, "unit:GHz"
    if "mhz" in u:
        return 1.0, "unit:MHz"
    if "khz" in u:
        return 1e-3, "unit:kHz"
    if "thz" in u:
        return 1e6, "unit:THz"
    if sample is not None:
        v = pd.to_numeric(pd.Series(sample), errors="coerce").dropna()
        if len(v):
            med = float(v.median())
            if med < 5000.0:
                return 1000.0, "guessed:GHz_from_magnitude"
            return 1.0, "guessed:MHz_from_magnitude"
    return 1.0, "assumed:MHz"


def unidentified_mask(values, patterns=None) -> np.ndarray:
    pats = [re.compile(p, re.IGNORECASE) for p in (patterns or DEFAULT_ULINE_PATTERNS)]
    out = []
    for v in values:
        s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        out.append(any(p.search(s) for p in pats))
    return np.asarray(out, dtype=bool)


def unquote_table(name: str) -> str:
    return str(name).strip().strip('"').strip()


def _tidy_tables(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or not len(df):
        return pd.DataFrame(columns=["table_name", "description"])
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    if "table_name" not in df:
        return pd.DataFrame(columns=["table_name", "description"])
    df = df.copy()
    df["table_name"] = df["table_name"].map(unquote_table)
    if "description" not in df:
        df["description"] = ""
    return df[["table_name", "description"]]


def tables_like_adql(pattern: str, limit: int = 60) -> str:
    """ADQL for a table-name search.

    **The leading ``%`` is load-bearing.**  TAPVizieR stores ``table_name``
    *with* its literal double quotes (``"J/ApJ/787/112/table2"``), so
    ``LIKE 'J/ApJ/787/112/%'`` matches nothing at all; this repository has
    been bitten by it twice (``scripts/necrofrontier_probe.py``,
    ``docs/baffle.md``).  Every LIKE here opens with ``%``.
    """
    return (f"SELECT TOP {int(limit)} table_name, description FROM TAP_SCHEMA.tables "
            f"WHERE table_name LIKE '%{pattern}%'")


def _like_any(column: str, word: str) -> str:
    variants = dict.fromkeys([word, word.lower(), word.capitalize(), word.upper()])
    return "(" + " OR ".join(f"{column} LIKE '%{v}%'" for v in variants) + ")"


def tables_described_adql(terms_all=(), terms_any=(), limit: int = 60) -> str:
    """ADQL for a DESCRIPTION search: every ``terms_all`` AND any ``terms_any``.

    Used when the asserted catalogue id is absent from ``TAP_SCHEMA`` — e.g.
    ``description LIKE '%unidentified%'`` combined with ``'%Orion%'`` or
    ``'%line survey%'``.  Case variants are OR-ed because TAPVizieR's LIKE is
    case-sensitive.
    """
    clauses = [_like_any("description", w) for w in terms_all if str(w).strip()]
    anyw = [_like_any("description", w) for w in terms_any if str(w).strip()]
    if anyw:
        clauses.append("(" + " OR ".join(anyw) + ")")
    where = " AND ".join(clauses) if clauses else "1 = 1"
    return (f"SELECT TOP {int(limit)} table_name, description FROM TAP_SCHEMA.tables "
            f"WHERE {where}")


def columns_described_adql(terms_any=(), limit: int = 400) -> str:
    """ADQL for a COLUMN-description search --- where the word actually lives.

    Run 35038662696 asked ``TAP_SCHEMA.tables`` for a description containing
    "unidentified" AND (Orion OR "line survey") and got **zero** tables, which
    reads as "there are no U-line tables in VizieR".  That is wrong: a VizieR
    TABLE description is a one-line title ("Spectral survey of Orion KL"), and
    the word "unidentified" lives in the description of a COLUMN --- the flag
    or note column that marks a line as unassigned.

    Searching ``TAP_SCHEMA.columns`` instead returns the table of every
    U-line-bearing catalogue in VizieR, whatever its source, which is the
    population this channel wants rather than one asserted id.  ``table_name``
    repeats once per matching column and is de-duplicated by the caller.
    """
    anyw = [_like_any("description", w) for w in terms_any if str(w).strip()]
    where = "(" + " OR ".join(anyw) + ")" if anyw else "1 = 1"
    return (f"SELECT TOP {int(limit)} table_name, column_name, description "
            f"FROM TAP_SCHEMA.columns WHERE {where}")


def list_tables_with_column_described(terms_any=(), *, query_fn=None, limit: int = 400
                                      ) -> pd.DataFrame:
    """``table_name, column_name, description`` for every matching COLUMN.

    One row per matching column, so a table with two such columns appears
    twice; ``n_tables`` in the caller counts distinct names.
    """
    query_fn = query_fn or tap_query
    df = query_fn(columns_described_adql(terms_any, limit))
    if df is None or not len(df):
        return pd.DataFrame(columns=["table_name", "column_name", "description"])
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    for c in ("column_name", "description"):
        if c not in df:
            df[c] = ""
    df = df.copy()
    df["table_name"] = df["table_name"].map(unquote_table)
    return df[["table_name", "column_name", "description"]]


def list_tables_like(pattern: str, *, query_fn=None, limit: int = 60) -> pd.DataFrame:
    query_fn = query_fn or tap_query
    return _tidy_tables(query_fn(tables_like_adql(pattern, limit)))


def list_tables_described(word=None, *, terms_all=(), terms_any=(), query_fn=None,
                          limit: int = 60) -> pd.DataFrame:
    """Tables whose description matches (case variants OR-ed)."""
    query_fn = query_fn or tap_query
    all_terms = list(terms_all) or ([word] if word else [])
    return _tidy_tables(query_fn(tables_described_adql(all_terms, terms_any, limit)))


def table_columns(table: str, *, query_fn=None) -> pd.DataFrame:
    """``column_name, unit, description`` of one table from TAP_SCHEMA.columns."""
    query_fn = query_fn or tap_query
    t = unquote_table(table)
    df = query_fn("SELECT TOP 2000 column_name, unit, description FROM TAP_SCHEMA.columns "
                  f"WHERE table_name = '{t}' OR table_name = '\"{t}\"'")
    if df is None or not len(df):
        return pd.DataFrame(columns=["column_name", "unit", "description"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    for c in ("unit", "description"):
        if c not in df:
            df[c] = ""
    return df[["column_name", "unit", "description"]]


def count_rows(table: str, *, query_fn=None) -> int | None:
    """``COUNT(*)``, or ``None`` when no route can answer it.

    A row count has no non-TAP equivalent, so with TAP down this is *unknown*
    (``None``), not a failure: the table is still selected on its columns.
    """
    query_fn = query_fn or tap_query
    try:
        df = query_fn(f'SELECT COUNT(*) AS n FROM "{unquote_table(table)}"')
    except VizierRouteError as exc:
        if exc.asu_supported:
            raise
        print(f"[uline/acquire] row count unavailable without TAP for {table!r}: {exc}")
        return None
    if df is None or not len(df):
        return None
    return int(df.iloc[0, 0])


@dataclass
class LineTableDiscovery:
    source: str
    pattern: str
    table: str | None = None
    roles: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    descriptions: dict = field(default_factory=dict)
    n_rows: int | None = None
    status: str = STATUS_ZERO
    scoreboard: list[dict] = field(default_factory=list)
    frame_hint: str = ""
    #: every ADQL sent, with its row count or its error text
    queries: list[dict] = field(default_factory=list)
    #: the DESCRIPTION search run when the asserted catalogue id was absent
    fallback: dict = field(default_factory=dict)
    #: which route answered discovery: ``tap`` | ``asu_tsv`` | ``none``
    route: str = ROUTE_TAP
    #: every route attempt (TAP host, ASU host, ReadMe), with its error text
    routes: list[dict] = field(default_factory=list)

    @property
    def errors(self) -> list[dict]:
        return [q for q in self.queries if q.get("status") == STATUS_FAILED]

    @property
    def route_errors(self) -> list[dict]:
        return [r for r in self.routes if r.get("status") == STATUS_FAILED]

    def as_dict(self) -> dict:
        return {"source": self.source, "pattern": self.pattern, "table": self.table,
                "roles": self.roles, "units": self.units, "descriptions": self.descriptions,
                "n_rows": self.n_rows, "status": self.status, "scoreboard": self.scoreboard,
                "frame_hint": self.frame_hint, "queries": self.queries,
                "errors": self.errors, "fallback": self.fallback, "route": self.route,
                "routes": self.routes, "route_errors": self.route_errors}


def traced_query(query_fn, trace: list[dict], stage: str):
    """Wrap ``query_fn`` so every ADQL — and the error text of every failure —
    is recorded verbatim.  A bare ``QUERY_FAILED`` is what made the first
    dispatch undiagnosable."""
    def run(adql: str):
        try:
            df = query_fn(adql)
        except Exception as exc:                              # noqa: BLE001
            trace.append({"stage": stage, "adql": str(adql), "status": STATUS_FAILED,
                          "rows": 0, "error": repr(exc)[:2000]})
            raise
        n = 0 if df is None else int(len(df))
        trace.append({"stage": stage, "adql": str(adql),
                      "status": STATUS_OK if n else STATUS_ZERO, "rows": n})
        return df
    return run


def non_tap_line_table(source: str, pattern: str, column_patterns=None, *, fetch_fn=None,
                       log: AcquisitionLog | None = None
                       ) -> tuple[tuple | None, list[dict], list[dict]]:
    """Discovery WITHOUT TAP: the catalogue's own ASU metadata (ReadMe as backstop).

    Answers the only question discovery asks --- does this catalogue exist, and
    which of its tables has a frequency column and an identification column ---
    from ``viz-bin/asu-tsv?-source=<cat>&-meta.all``, which is served by the
    VizieR web application and not by TAPVizieR.  Returns
    ``(best, attempts, scoreboard)``; ``best`` is ``None`` when the metadata
    could not be reached OR when it exposes no usable table, and the two are
    told apart by ``attempts``.
    """
    try:
        tabs, attempts = asu_catalogue_tables(pattern, fetch_fn=fetch_fn)
    except VizierRouteError as exc:
        if log:
            log.record(f"tables_{source}_non_tap", f"ASU meta ~ {pattern!r}", error=str(exc),
                       extra={"route": ROUTE_ASU, "route_attempts": exc.attempts})
        return None, list(exc.attempts), []
    except Exception as exc:                                  # noqa: BLE001
        if log:
            log.record(f"tables_{source}_non_tap", f"ASU meta ~ {pattern!r}", error=repr(exc),
                       extra={"route": ROUTE_ASU})
        return None, [{"route": ROUTE_ASU, "endpoint": VIZIER_ASU, "status": STATUS_FAILED,
                       "error": repr(exc)[:2000], "what": f"meta {pattern}"}], []
    board, best = [], None
    for _, row in tabs.iterrows():
        cols = [str(c) for c in (row.get("columns") or [])]
        roles = resolve_line_columns(cols, column_patterns) if cols else {}
        usable = bool(roles.get("freq") and roles.get("ident"))
        board.append({"table": str(row["table_name"]), "route": ROUTE_ASU,
                      "description": str(row.get("description", ""))[:200], "roles": roles,
                      "n_rows": row.get("n_rows"), "usable": usable, "columns": cols[:60]})
        if usable and best is None:
            units = {r: str((row.get("units") or {}).get(c, "")) for r, c in roles.items() if c}
            descs = {r: str((row.get("descriptions") or {}).get(c, "")) for r, c in roles.items()
                     if c}
            best = (str(row["table_name"]), row.get("n_rows"), roles, units, descs)
    if log:
        log.record(f"tables_{source}_non_tap", f"ASU meta ~ {pattern!r}", rows=int(len(tabs)),
                   extra={"route": ROUTE_ASU, "usable": bool(best),
                          "endpoint": next((a["endpoint"] for a in attempts
                                            if a["status"] == STATUS_OK), VIZIER_ASU)})
    return best, list(attempts), board


def discover_line_table(source: str, pattern: str, *, query_fn=None,
                        log: AcquisitionLog | None = None, column_patterns=None,
                        fallback_terms_all=(), fallback_terms_any=(), limit: int = 60,
                        fetch_fn=None, allow_non_tap: bool = True) -> LineTableDiscovery:
    """Pick the table under ``pattern`` that has a frequency and an identification column.

    Every candidate's roles and row count go on the scoreboard; the winner is
    the usable table with the most rows.  ``frame_hint`` records whether the
    frequency column's description mentions rest / LSR / observed.

    When the asserted catalogue id yields no table at all — or no *usable*
    one — a DESCRIPTION search is run and **every** table it returns is
    recorded in ``fallback`` so the next dispatch can be pointed at the real
    catalogue.  The fallback is diagnostic only: no table is ever selected
    from it automatically, and a new catalogue id has to be asserted in
    ``config/uline.yaml`` with a ``verify`` note first.
    """
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog()
    d = LineTableDiscovery(source=source, pattern=pattern)
    tabs_q = traced_query(query_fn, d.queries, f"tables_{source}")
    tabs = pd.DataFrame(columns=["table_name", "description"])
    try:
        tabs = list_tables_like(pattern, query_fn=tabs_q, limit=limit)
        log.record(f"tables_{source}", pattern, rows=int(len(tabs)),
                   extra={"adql": tables_like_adql(pattern, limit)})
    except Exception as exc:                                  # noqa: BLE001
        log.record(f"tables_{source}", pattern, error=repr(exc),
                   extra={"adql": tables_like_adql(pattern, limit)})
        d.status = STATUS_FAILED
    best = None
    for _, row in tabs.iterrows():
        t = str(row["table_name"])
        cq = traced_query(query_fn, d.queries, f"columns_{source}")
        try:
            cols = table_columns(t, query_fn=cq)
            roles = resolve_line_columns(cols, column_patterns)
            n = count_rows(t, query_fn=cq) if roles.get("freq") else None
            log.record(f"columns_{source}", t, rows=int(len(cols)))
        except Exception as exc:                              # noqa: BLE001
            log.record(f"columns_{source}", t, error=repr(exc))
            d.scoreboard.append({"table": t, "status": STATUS_FAILED, "error": repr(exc)[:2000]})
            continue
        usable = bool(roles.get("freq") and roles.get("ident"))
        entry = {"table": t, "description": str(row.get("description", ""))[:200],
                 "roles": roles, "n_rows": n, "usable": usable,
                 "columns": [str(c) for c in cols["column_name"].tolist()][:60]}
        d.scoreboard.append(entry)
        if usable and (best is None or (n or 0) > (best[1] or 0)):
            units = {r: str(cols.loc[cols["column_name"] == c, "unit"].iloc[0])
                     for r, c in roles.items() if c is not None and (cols["column_name"] == c).any()}
            descs = {r: str(cols.loc[cols["column_name"] == c, "description"].iloc[0])
                     for r, c in roles.items() if c is not None and (cols["column_name"] == c).any()}
            best = (t, n, roles, units, descs)
    if best is None and allow_non_tap:
        # TAP could not name a usable table.  Before falling back to the
        # DESCRIPTION search (which is also TAP), ask VizieR itself, without
        # TAP, whether this catalogue exists and what its tables look like.
        nt_best, nt_attempts, nt_board = non_tap_line_table(
            source, pattern, column_patterns, fetch_fn=fetch_fn, log=log)
        d.routes.extend(nt_attempts)
        d.scoreboard.extend(nt_board)
        if nt_best is not None:
            best = nt_best
            d.route = ROUTE_ASU
            d.queries.append({"stage": f"tables_{source}", "adql": f"ASU meta -source={pattern}",
                              "status": STATUS_OK, "rows": len(nt_board), "route": ROUTE_ASU,
                              "endpoint": next((a["endpoint"] for a in nt_attempts
                                                if a["status"] == STATUS_OK), VIZIER_ASU)})
    if best is None:
        d.route = ROUTE_NONE
        if d.status != STATUS_FAILED:
            d.status = STATUS_ZERO
        d.fallback = description_fallback(source, query_fn=query_fn, log=log, trace=d.queries,
                                          terms_all=fallback_terms_all,
                                          terms_any=fallback_terms_any, limit=limit)
        return d
    d.table, d.n_rows, d.roles, d.units, d.descriptions = best
    d.status = STATUS_OK
    fd = (d.descriptions.get("freq") or "").lower()
    hints = [w for w in ("rest", "lsr", "observed", "sky", "laboratory", "measured") if w in fd]
    d.frame_hint = ",".join(hints)
    return d


def description_fallback(source: str, *, query_fn=None, log: AcquisitionLog | None = None,
                         trace: list[dict] | None = None, terms_all=(), terms_any=(),
                         limit: int = 60) -> dict:
    """DESCRIPTION search for the real catalogue behind an absent table id.

    Returns the ADQL, the row count and **every** table the search returned
    (name + description), or the error text if the query itself failed.
    """
    terms_all = [t for t in (terms_all or ()) if str(t).strip()]
    terms_any = [t for t in (terms_any or ()) if str(t).strip()]
    if not terms_all and not terms_any:
        return {"status": "NOT_ATTEMPTED", "reason": "no fallback description terms configured"}
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog()
    adql = tables_described_adql(terms_all, terms_any, limit)
    out = {"terms_all": terms_all, "terms_any": terms_any, "adql": adql}
    fq = traced_query(query_fn, trace if trace is not None else [], f"fallback_{source}")
    try:
        df = _tidy_tables(fq(adql))
    except Exception as exc:                                  # noqa: BLE001
        log.record(f"fallback_{source}", adql, error=repr(exc), extra={"adql": adql})
        out.update({"status": STATUS_FAILED, "error": repr(exc)[:2000], "n_tables": 0,
                    "tables": []})
        return out
    log.record(f"fallback_{source}", adql, rows=int(len(df)), extra={"adql": adql})
    out.update({"status": STATUS_OK if len(df) else STATUS_ZERO, "n_tables": int(len(df)),
                "tables": [{"table_name": str(r["table_name"]),
                            "description": str(r["description"])[:300]}
                           for _, r in df.iterrows()]})
    out["by_column"] = column_census(terms_all + terms_any, query_fn=fq, log=log,
                                     source=source)
    return out


def column_census(terms_any=(), *, query_fn=None, log: AcquisitionLog | None = None,
                  source: str = "", limit: int = 400) -> dict:
    """Every VizieR table carrying a COLUMN whose description matches.

    The table-description search is the wrong instrument and run 35038662696
    proved it: zero tables for "unidentified", which would read as "VizieR has
    no U-line tables". The word is a COLUMN description. This census is
    diagnostic only --- nothing is ever selected from it automatically, exactly
    as with the table search --- but it is what tells the next dispatch which
    catalogue ids actually exist to be asserted.
    """
    terms = [t for t in (terms_any or ()) if str(t).strip()]
    if not terms:
        return {"status": "NOT_ATTEMPTED", "reason": "no column-description terms"}
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog()
    adql = columns_described_adql(terms, limit)
    out: dict = {"terms_any": terms, "adql": adql}
    try:
        df = list_tables_with_column_described(terms, query_fn=query_fn, limit=limit)
    except Exception as exc:                                  # noqa: BLE001
        log.record(f"column_census_{source}", adql, error=repr(exc), extra={"adql": adql})
        out.update({"status": STATUS_FAILED, "error": repr(exc)[:2000], "n_tables": 0,
                    "tables": []})
        return out
    log.record(f"column_census_{source}", adql, rows=int(len(df)), extra={"adql": adql})
    by_table: dict[str, list[dict]] = {}
    for _, r in df.iterrows():
        by_table.setdefault(str(r["table_name"]), []).append(
            {"column": str(r["column_name"]), "description": str(r["description"])[:200]})
    out.update({"status": STATUS_OK if by_table else STATUS_ZERO,
                "n_columns": int(len(df)), "n_tables": len(by_table),
                "tables": [{"table_name": t, "columns": cols[:6]}
                           for t, cols in sorted(by_table.items())]})
    return out


def fetch_line_table(disc: LineTableDiscovery, *, query_fn=None, log: AcquisitionLog | None = None,
                     max_rows: int = 200000, uline_patterns=None, fetch_fn=None,
                     allow_non_tap: bool = True) -> pd.DataFrame:
    """Pull the whole line table through verified columns; canonical columns out.

    TAP first; when it fails, the same non-TAP ladder discovery uses (ASU, then
    ``astroquery.vizier``).  The route that served the rows is recorded in the
    acquisition log and in ``df.attrs["route"]`` --- a run that says ``OK``
    must also say which archive door it came through.

    Output columns: ``freq_mhz, freq_err_mhz, ident, intensity, transition,
    unidentified`` plus ``row`` (the original row order).
    """
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog()
    if not disc.table or not disc.roles.get("freq"):
        return pd.DataFrame()
    cols = [c for c in (disc.roles.get(r) for r in ("freq", "freq_err", "ident", "intensity",
                                                    "transition")) if c]
    sel = ", ".join(f'"{c}"' for c in dict.fromkeys(cols))
    adql = f'SELECT TOP {int(max_rows)} {sel} FROM "{disc.table}"'
    route, endpoint = ROUTE_TAP, ""
    df = None
    try:
        df = query_fn(adql)
        log.record(f"fetch_{disc.source}", adql, rows=int(len(df)) if df is not None else None,
                   extra={"route": ROUTE_TAP})
    except Exception as exc:                                  # noqa: BLE001
        if not allow_non_tap:
            log.record(f"fetch_{disc.source}", adql, error=repr(exc), extra={"route": ROUTE_TAP})
            return pd.DataFrame()
        disc.routes.append({"route": ROUTE_TAP, "endpoint": "query_fn", "status": STATUS_FAILED,
                            "what": adql, "error": repr(exc)[:2000]})
        # TAP is down: the same rows, through VizieR's non-TAP doors.
        res = vizier_table(disc.table, columns=list(dict.fromkeys(cols)), max_rows=max_rows,
                           tap_urls=(), fetch_fn=fetch_fn, log=log,
                           stage=f"fetch_{disc.source}_non_tap")
        disc.routes.extend(res.attempts)
        if res.route == ROUTE_NONE:
            log.record(f"fetch_{disc.source}", adql, error=repr(exc), extra={
                "route": ROUTE_NONE,
                "route_errors": [f"{a['endpoint']}: {a.get('error', '')}" for a in res.errors]})
            return pd.DataFrame()
        df, route, endpoint = res.rows, res.route, res.endpoint
        disc.queries.append({"stage": f"fetch_{disc.source}", "adql": adql, "status": STATUS_OK,
                             "rows": int(len(df)), "route": route, "endpoint": endpoint})
    if df is None or not len(df):
        return pd.DataFrame()
    missing = [c for c in dict.fromkeys(cols) if c not in df.columns]
    if missing and route != ROUTE_TAP:
        # A non-TAP route returns the catalogue's own labels; keep only what
        # really came back rather than renaming something else into the role.
        for role, col in list(disc.roles.items()):
            if col in missing:
                disc.roles[role] = None
        if not disc.roles.get("freq") or not disc.roles.get("ident"):
            log.record(f"fetch_{disc.source}", adql, error=(
                f"route {route} returned columns {list(df.columns)[:20]} without the resolved "
                f"frequency/identification columns {missing}"), extra={"route": route})
            return pd.DataFrame()
    out = pd.DataFrame({"row": np.arange(len(df))})
    scale, how = frequency_scale(disc.units.get("freq"), df[disc.roles["freq"]])
    out["freq_mhz"] = pd.to_numeric(df[disc.roles["freq"]], errors="coerce") * scale
    out.attrs["freq_scale"] = how
    if disc.roles.get("freq_err"):
        escale, _ = frequency_scale(disc.units.get("freq_err") or disc.units.get("freq"))
        out["freq_err_mhz"] = pd.to_numeric(df[disc.roles["freq_err"]], errors="coerce") * escale
    else:
        out["freq_err_mhz"] = np.nan
    out["ident"] = df[disc.roles["ident"]].astype(str) if disc.roles.get("ident") else ""
    out["intensity"] = (pd.to_numeric(df[disc.roles["intensity"]], errors="coerce")
                        if disc.roles.get("intensity") else np.nan)
    out["transition"] = (df[disc.roles["transition"]].astype(str)
                         if disc.roles.get("transition") else "")
    out["unidentified"] = unidentified_mask(out["ident"].tolist(), uline_patterns)
    scale_how = out.attrs.get("freq_scale")
    out = out[np.isfinite(out["freq_mhz"])].reset_index(drop=True)
    out.attrs["freq_scale"] = scale_how
    out.attrs["route"] = route
    out.attrs["endpoint"] = endpoint
    return out


__all__ = ["DEFAULT_COLUMN_PATTERNS", "DEFAULT_ULINE_PATTERNS", "ROUTE_ASTROQUERY", "ROUTE_ASU",
           "ROUTE_NONE", "ROUTE_TAP", "VIZIER_ASU", "VIZIER_ASU_MIRRORS", "VIZIER_TAP",
           "VIZIER_TAP_MIRRORS", "AcquisitionLog", "AdqlNotTranslatable", "LineTableDiscovery",
           "STATUS_FAILED", "STATUS_OK", "STATUS_ZERO", "VizierResult", "VizierRouteError",
           "cdms_cat_tags", "cdms_inventory", "column_census", "columns_described_adql",
           "count_rows", "description_fallback",
           "discover_line_table", "fetch_cats", "fetch_line_table", "fetch_text",
           "frequency_scale", "jpl_inventory", "list_tables_described", "list_tables_like",
           "list_tables_with_column_described",
           "match_all_species", "non_tap_line_table", "reset_route_state", "resolve_line_columns",
           "route_log_summary", "table_columns", "tables_described_adql", "tables_like_adql",
           "tap_endpoints", "tap_query", "traced_query", "unidentified_mask", "unquote_table",
           "vizier_table"]

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

from .lines import Entry, find_species, parse_cat, parse_catdir, parse_partition_table

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
            rec["error"] = str(error)[:500]
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
               log: AcquisitionLog | None = None, stage: str = "fetch") -> str | None:
    """Fetch with retries; ``None`` (and a logged failure) when every attempt fails."""
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
    return None


def tap_query(adql: str, *, url: str = VIZIER_TAP, retries: int = 3) -> pd.DataFrame:
    """ADQL against VizieR TAP: async first, sync on failure (runner only)."""
    import pyvo  # noqa: PLC0415

    last = None
    for attempt in range(max(int(retries), 1)):
        try:
            svc = pyvo.dal.TAPService(url)
            try:
                return svc.run_async(adql).to_table().to_pandas()
            except Exception as exc:                          # noqa: BLE001
                print(f"[uline/acquire] async TAP failed ({exc!r}); trying sync")
                return svc.search(adql).to_table().to_pandas()
        except Exception as exc:                              # noqa: BLE001
            last = exc
            print(f"[uline/acquire] TAP attempt {attempt + 1}/{retries} failed: {exc!r}")
            _time.sleep(4.0 * (attempt + 1))
    raise RuntimeError(f"TAP query failed after {retries} attempts: {last!r}")


# ---------------------------------------------------------------------------
# JPL
# ---------------------------------------------------------------------------
def jpl_inventory(catdir_text: str, species_conf: dict) -> dict[str, list[Entry]]:
    """{species: [matching catdir entries]} for every configured species."""
    entries = parse_catdir(catdir_text)
    out: dict[str, list[Entry]] = {}
    for group in ("targets", "baseline", "contaminants"):
        for sp, spec in (species_conf.get(group) or {}).items():
            out[sp] = find_species(entries, spec.get("patterns") or [rf"^{re.escape(sp)}\b"])
    return out


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
    inv: dict[str, list[Entry]] = {}
    allentries = list(entries.values())
    for group in ("targets", "baseline", "contaminants"):
        for sp, spec in (species_conf.get(group) or {}).items():
            inv[sp] = find_species(allentries, spec.get("patterns") or [rf"^{re.escape(sp)}\b"])
    return allentries, {"pages": report, "n_entries": len(allentries), "species": inv}


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


def list_tables_like(pattern: str, *, query_fn=None, limit: int = 60) -> pd.DataFrame:
    query_fn = query_fn or tap_query
    adql = (f"SELECT TOP {int(limit)} table_name, description FROM TAP_SCHEMA.tables "
            f"WHERE table_name LIKE '%{pattern}%'")
    df = query_fn(adql)
    if df is None or not len(df):
        return pd.DataFrame(columns=["table_name", "description"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["table_name"] = df["table_name"].map(unquote_table)
    return df[["table_name"] + [c for c in ("description",) if c in df]]


def list_tables_described(word: str, *, query_fn=None, limit: int = 60) -> pd.DataFrame:
    """Tables whose description mentions ``word`` (case variants OR-ed)."""
    query_fn = query_fn or tap_query
    variants = dict.fromkeys([word, word.lower(), word.capitalize(), word.upper()])
    where = " OR ".join(f"description LIKE '%{v}%'" for v in variants)
    df = query_fn(f"SELECT TOP {int(limit)} table_name, description FROM TAP_SCHEMA.tables "
                  f"WHERE {where}")
    if df is None or not len(df):
        return pd.DataFrame(columns=["table_name", "description"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["table_name"] = df["table_name"].map(unquote_table)
    return df


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
    query_fn = query_fn or tap_query
    df = query_fn(f'SELECT COUNT(*) AS n FROM "{unquote_table(table)}"')
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

    def as_dict(self) -> dict:
        return {"source": self.source, "pattern": self.pattern, "table": self.table,
                "roles": self.roles, "units": self.units, "descriptions": self.descriptions,
                "n_rows": self.n_rows, "status": self.status, "scoreboard": self.scoreboard,
                "frame_hint": self.frame_hint}


def discover_line_table(source: str, pattern: str, *, query_fn=None,
                        log: AcquisitionLog | None = None, column_patterns=None
                        ) -> LineTableDiscovery:
    """Pick the table under ``pattern`` that has a frequency and an identification column.

    Every candidate's roles and row count go on the scoreboard; the winner is
    the usable table with the most rows.  ``frame_hint`` records whether the
    frequency column's description mentions rest / LSR / observed.
    """
    query_fn = query_fn or tap_query
    log = log or AcquisitionLog()
    d = LineTableDiscovery(source=source, pattern=pattern)
    try:
        tabs = list_tables_like(pattern, query_fn=query_fn)
        log.record(f"tables_{source}", pattern, rows=int(len(tabs)))
    except Exception as exc:                                  # noqa: BLE001
        log.record(f"tables_{source}", pattern, error=repr(exc))
        d.status = STATUS_FAILED
        return d
    if not len(tabs):
        d.status = STATUS_ZERO
        return d
    best = None
    for _, row in tabs.iterrows():
        t = str(row["table_name"])
        try:
            cols = table_columns(t, query_fn=query_fn)
            roles = resolve_line_columns(cols, column_patterns)
            n = count_rows(t, query_fn=query_fn) if roles.get("freq") else None
            log.record(f"columns_{source}", t, rows=int(len(cols)))
        except Exception as exc:                              # noqa: BLE001
            log.record(f"columns_{source}", t, error=repr(exc))
            d.scoreboard.append({"table": t, "status": STATUS_FAILED, "error": repr(exc)[:200]})
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
    if best is None:
        d.status = STATUS_ZERO
        return d
    d.table, d.n_rows, d.roles, d.units, d.descriptions = best
    d.status = STATUS_OK
    fd = (d.descriptions.get("freq") or "").lower()
    hints = [w for w in ("rest", "lsr", "observed", "sky", "laboratory", "measured") if w in fd]
    d.frame_hint = ",".join(hints)
    return d


def fetch_line_table(disc: LineTableDiscovery, *, query_fn=None, log: AcquisitionLog | None = None,
                     max_rows: int = 200000, uline_patterns=None) -> pd.DataFrame:
    """Pull the whole line table through verified columns; canonical columns out.

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
    try:
        df = query_fn(adql)
        log.record(f"fetch_{disc.source}", adql, rows=int(len(df)) if df is not None else None)
    except Exception as exc:                                  # noqa: BLE001
        log.record(f"fetch_{disc.source}", adql, error=repr(exc))
        return pd.DataFrame()
    if df is None or not len(df):
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
    out = out[np.isfinite(out["freq_mhz"])].reset_index(drop=True)
    return out


__all__ = ["DEFAULT_COLUMN_PATTERNS", "DEFAULT_ULINE_PATTERNS", "AcquisitionLog",
           "LineTableDiscovery", "STATUS_FAILED", "STATUS_OK", "STATUS_ZERO", "VIZIER_TAP",
           "cdms_cat_tags", "cdms_inventory", "count_rows", "discover_line_table", "fetch_cats",
           "fetch_line_table", "fetch_text", "frequency_scale", "jpl_inventory",
           "list_tables_described", "list_tables_like", "resolve_line_columns", "table_columns",
           "tap_query", "unidentified_mask", "unquote_table"]

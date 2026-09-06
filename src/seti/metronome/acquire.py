"""Runner-only VizieR access for METRONOME, with runtime schema discovery.

The sandbox has no archive egress, so nothing here runs in a test; the pure
pieces --- column-role resolution, table scoring, the acquisition log --- are
what the tests exercise, and every network function takes an injectable
``query_fn`` so a failed or empty archive can be simulated offline.

Why discovery rather than hard-coded ids.  VizieR catalogue numbers and
column names are not stable facts: ``seti.tailings`` lost three dispatches to a
renumbered GALAH table and a mangled ``[Mg/Fe]`` column before it started
asking ``TAP_SCHEMA`` what the service actually holds.  This module therefore
treats the table ids in ``config/metronome.yaml`` as *preferred* seeds, lists
every table under that id, reads their real column names, resolves the roles
this channel needs (star id, peak/start/end time, energy, sector, rotation
period, position) with :func:`resolve_event_columns`, and records the whole
scoreboard in the probe artefact.  A column name that was not verified against
``TAP_SCHEMA.columns`` is never interpolated into a query.

The acquisition log separates ``QUERY_FAILED`` (the service did not answer or
errored) from ``QUERY_RETURNED_ZERO_ROWS`` (it answered, with nothing): these
are different facts about the world and the summary must not collapse them.
"""

from __future__ import annotations

import json
import re
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VIZIER_TAP = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap"

STATUS_OK = "OK"
STATUS_FAILED = "QUERY_FAILED"
STATUS_ZERO = "QUERY_RETURNED_ZERO_ROWS"


# ---------------------------------------------------------------------------
# Acquisition log (same discipline as seti.knell.acquire.AcquisitionLog)
# ---------------------------------------------------------------------------
@dataclass
class AcquisitionLog:
    stages: list[dict] = field(default_factory=list)
    prefix: str = "metronome/acquire"

    def record(self, stage: str, query: str, *, rows: int | None = None,
               error: str | None = None, extra: dict | None = None) -> None:
        if error is not None or rows is None:
            status = STATUS_FAILED
        elif rows == 0:
            status = STATUS_ZERO
        else:
            status = STATUS_OK
        rec = {"stage": stage, "status": status, "rows": int(rows or 0),
               "query": str(query)[:2000]}
        if error:
            rec["error"] = str(error)[:500]
        if extra:
            rec.update(extra)
        self.stages.append(rec)
        print(f"[{self.prefix}] {stage}: {status} rows={rec['rows']}"
              + (f" error={rec.get('error')}" if error else ""))

    def as_dict(self) -> dict:
        n_fail = sum(1 for s in self.stages if s["status"] == STATUS_FAILED)
        n_zero = sum(1 for s in self.stages if s["status"] == STATUS_ZERO)
        n_ok = sum(1 for s in self.stages if s["status"] == STATUS_OK)
        return {"stages": self.stages, "n_stages": len(self.stages), "n_ok": n_ok,
                "n_query_failed": n_fail, "n_query_returned_zero_rows": n_zero,
                "any_query_failed": bool(n_fail > 0),
                "total_rows": int(sum(s["rows"] for s in self.stages))}

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))


# ---------------------------------------------------------------------------
# TAP primitives
# ---------------------------------------------------------------------------
def _retry(fn, retries: int = 3, label: str = "query", base_sleep: float = 4.0):
    last = None
    for attempt in range(int(retries)):
        try:
            return fn()
        except Exception as exc:                          # noqa: BLE001
            last = exc
            print(f"[metronome/acquire] {label} attempt {attempt + 1}/{retries} failed: {exc!r}")
            _time.sleep(base_sleep * (attempt + 1))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


def tap_query(adql: str, *, url: str = VIZIER_TAP, retries: int = 3) -> pd.DataFrame:
    """ADQL against VizieR TAP: async first, sync on the last attempt."""
    import pyvo  # noqa: PLC0415  runner-only; keeps the module importable offline

    def _go():
        svc = pyvo.dal.TAPService(url)
        try:
            return svc.run_async(adql).to_table().to_pandas()
        except Exception as exc:                          # noqa: BLE001
            print(f"[metronome/acquire] async TAP failed ({exc!r}); trying sync")
            return svc.search(adql).to_table().to_pandas()

    return _retry(_go, retries=retries, label="TAP query")


def unquote_table(name: str) -> str:
    return str(name).strip().strip('"').strip()


def list_tables(pattern: str, *, query_fn=None, limit: int = 60) -> pd.DataFrame:
    """Tables in ``TAP_SCHEMA.tables`` whose name contains ``pattern``."""
    query_fn = query_fn or tap_query
    adql = (f"SELECT TOP {int(limit)} table_name, description FROM TAP_SCHEMA.tables "
            f"WHERE table_name LIKE '%{pattern}%'")
    df = query_fn(adql)
    if df is None or not len(df):
        return pd.DataFrame(columns=["table_name", "description"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["table_name"] = df["table_name"].map(unquote_table)
    return df[["table_name"] + [c for c in ("description",) if c in df]]


def search_tables(keywords, *, query_fn=None, limit: int = 60) -> pd.DataFrame:
    """Keyword discovery over table descriptions (the fallback route)."""
    query_fn = query_fn or tap_query
    pats = []
    for k in keywords:
        for v in {k, k.lower(), k.upper(), k.capitalize()}:
            pats.append(f"description LIKE '%{v}%'")
    adql = (f"SELECT TOP {int(limit)} table_name, description FROM TAP_SCHEMA.tables "
            "WHERE " + " OR ".join(dict.fromkeys(pats)))
    df = query_fn(adql)
    if df is None or not len(df):
        return pd.DataFrame(columns=["table_name", "description"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["table_name"] = df["table_name"].map(unquote_table)
    return df


def table_columns(table: str, *, query_fn=None) -> list[str]:
    """Real column names of one table from ``TAP_SCHEMA.columns``."""
    query_fn = query_fn or tap_query
    t = unquote_table(table)
    adql = ("SELECT TOP 2000 column_name FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '{t}' OR table_name = '\"{t}\"'")
    df = query_fn(adql)
    if df is None or not len(df):
        return []
    col = "column_name" if "column_name" in df.columns else df.columns[0]
    return [str(c) for c in df[col].tolist()]


def count_rows(table: str, *, query_fn=None) -> int | None:
    query_fn = query_fn or tap_query
    df = query_fn(f'SELECT COUNT(*) AS n FROM "{unquote_table(table)}"')
    if df is None or not len(df):
        return None
    return int(df.iloc[0, 0])


# ---------------------------------------------------------------------------
# Column-role resolution (pure)
# ---------------------------------------------------------------------------
_ROLE_PATTERNS: dict[str, list[str]] = {
    # order within a role = preference
    "star_id": [r"^kic$", r"^kic_?id$", r"^kepid$", r"^tic$", r"^tic_?id$", r"^ticid$",
                r"^epic$", r"^id$", r"^star$", r"^name$", r"^source$"],
    "t_peak": [r"^t_?peak$", r"^tpk$", r"^peak_?time$", r"^bjd_?peak$", r"^peak$",
               r"^tmax$", r"^t_?max$", r"^time$", r"^bjd$", r"^tflare$"],
    "t_start": [r"^t_?start$", r"^t_?beg(in)?$", r"^start$", r"^bjd_?start$", r"^tstart$",
                r"^t_?ini$", r"^t1$", r"^t0$"],
    "t_end": [r"^t_?end$", r"^t_?stop$", r"^end$", r"^stop$", r"^bjd_?end$", r"^t2$",
              r"^tend$"],
    "energy": [r"^e$", r"^energy$", r"^e_?flare$", r"^ebol$", r"^log_?e$", r"^loge$",
               r"^ed$", r"^e_?bol$"],
    "amplitude": [r"^amp(l|litude)?$", r"^a$", r"^fpeak$", r"^f_?peak$", r"^dflux$",
                  r"^rel_?amp$"],
    "sector": [r"^sector$", r"^sec$", r"^sectors$", r"^quarter$", r"^q$", r"^camp(aign)?$"],
    "prot": [r"^prot$", r"^p_?rot$", r"^rot_?per$", r"^rotper$", r"^per$", r"^period$",
             r"^p$"],
    "ra": [r"^ra_?icrs$", r"^raj2000$", r"^_?ra$", r"^ra_?deg$", r"^radeg$"],
    "dec": [r"^de_?icrs$", r"^dej2000$", r"^_?dec?$", r"^dec_?deg$", r"^dedeg$"],
    "duration": [r"^dur(ation)?$", r"^tdur$", r"^t_?dur$", r"^length$"],
}


def _canon(name: str) -> str:
    return re.sub(r"_+", "_", str(name).strip().lower()).strip("_")


def resolve_columns(columns, roles: dict[str, list[str]] | None = None) -> dict[str, str]:
    """Map real column names onto the roles this channel needs.

    Exact (canonicalised) regex matches only --- ``LIKE``-style substring
    matching is what made ``Per`` match ``Perr`` in an earlier channel.  The
    *first* pattern in the preference list that matches any column wins, and
    the result is the *real* column name, which is the only thing that ever
    goes into a query.
    """
    roles = roles or _ROLE_PATTERNS
    canon = {c: _canon(c) for c in columns}
    out: dict[str, str] = {}
    for role, pats in roles.items():
        for pat in pats:
            rx = re.compile(pat)
            hit = next((c for c, k in canon.items() if rx.match(k)), None)
            if hit is not None:
                out[role] = hit
                break
    return out


def resolve_event_columns(columns) -> dict[str, str]:
    return resolve_columns(columns)


def score_event_table(columns) -> tuple[int, dict[str, str], str]:
    """Rank a table as a per-flare event list: ``(score, roles, reason)``.

    Usable = a star id and at least one of peak / start time.  Peak time is
    worth more than start (the statistic is on peaks), energy and sector are
    bonuses, and a rotation-period column is a bonus because it means the
    veto's P_rot can come from the same catalogue that defined the flares.
    """
    roles = resolve_event_columns(columns)
    if "star_id" not in roles or not ({"t_peak", "t_start"} & set(roles)):
        missing = [r for r in ("star_id", "t_peak/t_start")
                   if (r == "star_id" and "star_id" not in roles)
                   or (r != "star_id" and not ({"t_peak", "t_start"} & set(roles)))]
        return 0, roles, "rejected: no " + ", ".join(missing)
    score = 10
    score += 5 if "t_peak" in roles else 0
    score += 3 if "t_start" in roles else 0
    score += 2 if "t_end" in roles else 0
    score += 2 if "energy" in roles else 0
    score += 1 if "amplitude" in roles else 0
    score += 2 if "sector" in roles else 0
    score += 2 if "prot" in roles else 0
    score += 1 if ("ra" in roles and "dec" in roles) else 0
    return score, roles, "usable"


@dataclass
class DiscoveredTable:
    catalogue: str
    table: str | None
    columns: list[str]
    roles: dict[str, str]
    n_rows: int | None
    status: str
    route: str = "preferred"
    scoreboard: list[dict] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"catalogue": self.catalogue, "table": self.table, "n_columns": len(self.columns),
                "columns": self.columns[:80], "roles": self.roles, "n_rows": self.n_rows,
                "status": self.status, "route": self.route, "scoreboard": self.scoreboard,
                "note": self.note}


def discover_event_table(catalogue: str, preferred: str, keywords=(), *, query_fn=None,
                         log: AcquisitionLog | None = None) -> DiscoveredTable:
    """List the tables under ``preferred``, score them, fall back to keywords."""
    query_fn = query_fn or tap_query
    board: list[dict] = []
    best: DiscoveredTable | None = None
    best_key: tuple = (-1, -1)
    routes = [("preferred", lambda: list_tables(preferred, query_fn=query_fn))]
    if keywords:
        routes.append(("keyword", lambda: search_tables(keywords, query_fn=query_fn)))
    any_failed = False
    for route, lister in routes:
        try:
            tabs = lister()
        except Exception as exc:                          # noqa: BLE001
            any_failed = True
            if log:
                log.record(f"discover_{catalogue}_{route}", f"TAP_SCHEMA.tables ~ {preferred!r}",
                           error=repr(exc))
            continue
        if log:
            log.record(f"discover_{catalogue}_{route}", f"TAP_SCHEMA.tables ~ {preferred!r}",
                       rows=int(len(tabs)))
        for _, row in tabs.iterrows():
            t = unquote_table(row["table_name"])
            try:
                cols = table_columns(t, query_fn=query_fn)
            except Exception as exc:                      # noqa: BLE001
                board.append({"table": t, "route": route, "score": 0,
                              "reason": f"columns query failed: {exc!r}"[:200]})
                continue
            score, roles, reason = score_event_table(cols)
            entry = {"table": t, "route": route, "score": int(score), "reason": reason,
                     "roles": roles, "n_columns": len(cols)}
            if score > 0:
                try:
                    entry["n_rows"] = count_rows(t, query_fn=query_fn)
                except Exception as exc:                  # noqa: BLE001
                    entry["n_rows"] = None
                    entry["count_error"] = repr(exc)[:200]
            board.append(entry)
            if score > 0:
                n = entry.get("n_rows") or 0
                # A per-flare table has many rows; a per-star table with a
                # "first flare time" column does not.  Rows break score ties.
                key = (score + (2 if n >= 1000 else 0), n)
                if key > best_key:
                    best_key = key
                    best = DiscoveredTable(catalogue, t, cols, roles, entry.get("n_rows"),
                                           STATUS_OK, route=route)
        if best is not None:
            break
    if best is None:
        status = STATUS_FAILED if any_failed and not board else STATUS_ZERO
        return DiscoveredTable(catalogue, None, [], {}, None, status, route="none",
                               scoreboard=board,
                               note=("no table under the preferred id or the keyword "
                                     "search exposes a star id plus a peak/start time"))
    best.scoreboard = board
    if best.n_rows == 0:
        best.status = STATUS_ZERO
    return best


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_events(disc: DiscoveredTable, *, query_fn=None, log: AcquisitionLog | None = None,
                 chunk_rows: int = 50000, max_rows: int | None = None) -> pd.DataFrame:
    """Pull the event table in ``recno`` chunks, renamed to the channel's roles.

    Chunking by VizieR's ``recno`` (present when discovery saw it) bounds every
    request and makes a lost chunk re-fetchable on its own; without it the
    table is pulled in one async query.  Output columns: ``star_id, t_peak,
    t_start, t_end, energy, amplitude, sector, prot, ra, dec`` (missing roles
    absent).
    """
    query_fn = query_fn or tap_query
    if disc.table is None or not disc.roles:
        return pd.DataFrame()
    roles = dict(disc.roles)
    if "t_peak" not in roles and "t_start" in roles:
        roles["t_peak_from_start"] = roles["t_start"]
    sel = ", ".join(f'"{c}"' for c in dict.fromkeys(roles.values()))
    has_recno = any(_canon(c) == "recno" for c in disc.columns)
    n_total = disc.n_rows if disc.n_rows is not None else None
    limit = int(max_rows) if max_rows else None
    frames = []
    label = f"fetch_{disc.catalogue}"
    if has_recno and n_total and n_total > chunk_rows:
        top = n_total if limit is None else min(n_total, limit)
        lo = 1
        while lo <= top:
            hi = min(lo + chunk_rows - 1, top)
            adql = (f'SELECT {sel} FROM "{disc.table}" WHERE recno BETWEEN {lo} AND {hi}')
            try:
                df = query_fn(adql)
            except Exception as exc:                      # noqa: BLE001
                if log:
                    log.record(label, adql, error=repr(exc), extra={"chunk": [lo, hi]})
                lo = hi + 1
                continue
            n = int(len(df)) if df is not None else 0
            if log:
                log.record(label, adql, rows=n, extra={"chunk": [lo, hi]})
            if n:
                frames.append(df)
            lo = hi + 1
    else:
        top = f"TOP {limit} " if limit else ""
        adql = f'SELECT {top}{sel} FROM "{disc.table}"'
        try:
            df = query_fn(adql)
        except Exception as exc:                          # noqa: BLE001
            if log:
                log.record(label, adql, error=repr(exc))
            return pd.DataFrame()
        n = int(len(df)) if df is not None else 0
        if log:
            log.record(label, adql, rows=n)
        if n:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    inv = {}
    for role, col in roles.items():
        for c in out.columns:
            if str(c) == col or _canon(c) == _canon(col):
                inv.setdefault(c, role)
    out = out.rename(columns=inv)
    if "t_peak" in out.columns:
        out["t_peak_source"] = "t_peak"
    elif "t_start" in out.columns:
        # The catalogue has no peak time: the start time is the event time,
        # and the record says so.  (Peak - start is a flare-duration offset,
        # constant to within the rise time, so a clock survives the substitution.)
        out["t_peak"] = out["t_start"]
        out["t_peak_source"] = "t_start"
    out = out.drop(columns=[c for c in ("t_peak_from_start",) if c in out.columns])
    for c in ("t_peak", "t_start", "t_end", "energy", "amplitude", "prot", "ra", "dec"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "star_id" in out.columns:
        out["star_id"] = out["star_id"].astype(str).str.strip()
    return out


def discover_and_fetch_rotation(catalogue: str, preferred: str, keywords=(), *,
                                query_fn=None, log: AcquisitionLog | None = None,
                                max_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    """A rotation-period table: ``star_id, prot`` plus the discovery record."""
    query_fn = query_fn or tap_query
    rec = {"catalogue": catalogue, "preferred": preferred, "table": None, "roles": {},
           "status": STATUS_FAILED, "scoreboard": []}
    try:
        tabs = list_tables(preferred, query_fn=query_fn)
        if not len(tabs) and keywords:
            tabs = search_tables(keywords, query_fn=query_fn)
    except Exception as exc:                              # noqa: BLE001
        if log:
            log.record(f"discover_rot_{catalogue}", f"TAP_SCHEMA.tables ~ {preferred!r}",
                       error=repr(exc))
        rec["error"] = repr(exc)[:300]
        return pd.DataFrame(), rec
    if log:
        log.record(f"discover_rot_{catalogue}", f"TAP_SCHEMA.tables ~ {preferred!r}",
                   rows=int(len(tabs)))
    best = None
    for _, row in tabs.iterrows():
        t = unquote_table(row["table_name"])
        try:
            cols = table_columns(t, query_fn=query_fn)
        except Exception as exc:                          # noqa: BLE001
            rec["scoreboard"].append({"table": t, "reason": f"columns failed: {exc!r}"[:200]})
            continue
        roles = resolve_columns(cols)
        ok = "star_id" in roles and "prot" in roles
        rec["scoreboard"].append({"table": t, "roles": {k: roles[k] for k in roles
                                                          if k in ("star_id", "prot")},
                                  "usable": ok})
        if ok and best is None:
            best = (t, roles)
    if best is None:
        rec["status"] = STATUS_ZERO
        return pd.DataFrame(), rec
    t, roles = best
    top = f"TOP {int(max_rows)} " if max_rows else ""
    adql = f'SELECT {top}"{roles["star_id"]}", "{roles["prot"]}" FROM "{t}"'
    try:
        df = query_fn(adql)
    except Exception as exc:                              # noqa: BLE001
        if log:
            log.record(f"fetch_rot_{catalogue}", adql, error=repr(exc))
        rec.update({"table": t, "roles": roles, "error": repr(exc)[:300]})
        return pd.DataFrame(), rec
    n = int(len(df)) if df is not None else 0
    if log:
        log.record(f"fetch_rot_{catalogue}", adql, rows=n)
    rec.update({"table": t, "roles": {"star_id": roles["star_id"], "prot": roles["prot"]},
                "status": STATUS_OK if n else STATUS_ZERO, "n_rows": n})
    if not n:
        return pd.DataFrame(), rec
    out = pd.DataFrame({"star_id": df.iloc[:, 0].astype(str).str.strip(),
                        "prot": pd.to_numeric(df.iloc[:, 1], errors="coerce")})
    out["prot_source"] = catalogue
    return out, rec


def fetch_positions_by_id(ids, mission: str, *, query_fn=None,
                          log: AcquisitionLog | None = None, chunk: int = 200,
                          tables: dict | None = None) -> pd.DataFrame:
    """Positions for a shortlist of KIC / TIC ids (for the cone crossmatches).

    Kepler: the Kepler Input Catalog on VizieR (``V/133/kic``); TESS: the TIC
    (``IV/39/tic82`` preferred, ``IV/38/tic`` fallback).  Column names are read
    from ``TAP_SCHEMA`` first, never assumed.
    """
    query_fn = query_fn or tap_query
    tables = tables or {"kepler": ["V/133/kic"], "tess": ["IV/39/tic82", "IV/38/tic"]}
    ids = [str(i).strip() for i in ids if str(i).strip()]
    if not ids:
        return pd.DataFrame(columns=["star_id", "ra", "dec"])
    for t in tables.get(mission, []):
        try:
            cols = table_columns(t, query_fn=query_fn)
        except Exception as exc:                          # noqa: BLE001
            if log:
                log.record(f"positions_{mission}", f"columns of {t}", error=repr(exc))
            continue
        roles = resolve_columns(cols)
        if not {"star_id", "ra", "dec"} <= set(roles):
            if log:
                log.record(f"positions_{mission}", f"columns of {t}", rows=len(cols),
                           extra={"note": f"roles unresolved: {roles}"})
            continue
        frames = []
        for i in range(0, len(ids), int(chunk)):
            block = ids[i:i + int(chunk)]
            numeric = all(re.fullmatch(r"\d+", b) for b in block)
            vals = ", ".join(b if numeric else f"'{b}'" for b in block)
            adql = (f'SELECT "{roles["star_id"]}", "{roles["ra"]}", "{roles["dec"]}" '
                    f'FROM "{t}" WHERE "{roles["star_id"]}" IN ({vals})')
            try:
                df = query_fn(adql)
            except Exception as exc:                      # noqa: BLE001
                if log:
                    log.record(f"positions_{mission}", adql, error=repr(exc))
                continue
            n = int(len(df)) if df is not None else 0
            if log:
                log.record(f"positions_{mission}", adql[:300], rows=n)
            if n:
                frames.append(pd.DataFrame({"star_id": df.iloc[:, 0].astype(str).str.strip(),
                                            "ra": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
                                            "dec": pd.to_numeric(df.iloc[:, 2], errors="coerce")}))
        if frames:
            return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["star_id", "ra", "dec"])


def _vizier_cone(table: str, ra: float, dec: float, radius_arcsec: float):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    v = Vizier(columns=["**"], row_limit=50)
    res = v.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius_arcsec * u.arcsec,
                         catalog=table)
    if res is None or len(res) == 0:
        return pd.DataFrame()
    return res[0].to_pandas()


def fetch_variable_context(positions: pd.DataFrame, catalogues: dict, *, cone_fn=None,
                           log: AcquisitionLog | None = None, radius_arcsec: float = 3.0
                           ) -> tuple[dict, dict]:
    """Catalogued periods per star from VSX / Gaia DR3 vari / ZTF periodic variables.

    ``catalogues`` maps a name to ``{"table": ..., "period_patterns": [...],
    "type_patterns": [...]}``.  Returns ``({star_id: [(source, period, vtype)]},
    {source: reached_bool})``; the per-source ``reached`` flag is what turns a
    missing veto into ``variability_catalogue_unreached`` rather than a pass.
    """
    cone_fn = cone_fn or _vizier_cone
    out: dict[str, list] = {}
    reached: dict[str, bool] = {}
    for name, spec in (catalogues or {}).items():
        table = spec.get("table")
        n_ok = n_fail = n_hits = 0
        for _, r in positions.iterrows():
            ra, dec = float(r.get("ra", np.nan)), float(r.get("dec", np.nan))
            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue
            try:
                df = cone_fn(table, ra, dec, radius_arcsec)
                n_ok += 1
            except Exception as exc:                      # noqa: BLE001
                n_fail += 1
                if log and n_fail <= 3:
                    log.record(f"vari_{name}", f"cone {table} ({ra:.5f},{dec:.5f})",
                               error=repr(exc))
                continue
            if df is None or not len(df):
                continue
            roles = resolve_columns(df.columns, {
                "period": spec.get("period_patterns") or [r"^period$", r"^per$", r"^p$",
                                                          r"^pf$", r"^p_?f$"],
                "vtype": spec.get("type_patterns") or [r"^type$", r"^vtype$", r"^class$",
                                                       r"^best_?class_?name$", r"^vartype$"]})
            for _, row in df.iterrows():
                p = pd.to_numeric(row.get(roles.get("period")), errors="coerce") \
                    if roles.get("period") else np.nan
                vt = str(row.get(roles.get("vtype"), "")) if roles.get("vtype") else ""
                out.setdefault(str(r["star_id"]), []).append((name, float(p) if pd.notna(p)
                                                              else float("nan"), vt))
                n_hits += 1
        reached[name] = n_ok > 0 and n_fail == 0
        if log:
            log.record(f"vari_{name}", f"{n_ok + n_fail} cones on {table} r={radius_arcsec}\"",
                       rows=n_hits if n_ok else None,
                       error=None if n_ok else "every cone failed",
                       extra={"n_cones_ok": n_ok, "n_cones_failed": n_fail})
    return out, reached


__all__ = ["STATUS_FAILED", "STATUS_OK", "STATUS_ZERO", "VIZIER_TAP", "AcquisitionLog",
           "DiscoveredTable", "count_rows", "discover_and_fetch_rotation",
           "discover_event_table", "fetch_events", "fetch_positions_by_id",
           "fetch_variable_context", "list_tables", "resolve_columns",
           "resolve_event_columns", "score_event_table", "search_tables", "table_columns",
           "tap_query", "unquote_table"]

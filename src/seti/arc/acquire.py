"""Runner-only VizieR / Gaia access for ARC, with runtime schema discovery.

Everything network-facing takes an injectable ``query_fn`` (ADQL -> DataFrame)
or ``cone_fn`` (table, ra, dec, radius -> DataFrame) so a failed or empty
archive can be simulated offline; the tests never open a socket.

The TAP primitives, the acquisition log and the column-canonicalisation are
:mod:`seti.metronome.acquire`'s --- the same VizieR service, the same lesson
(no column name goes into a query that was not seen in ``TAP_SCHEMA.columns``).
What is new here is the role set: a flare table must expose a star id and an
energy (or an equivalent duration); a star table must expose a star id and a
rotational amplitude, a rotation period or a radius; and the Gaia context
for the shortlist comes from a 12" cone on ``I/355/gaiadr3`` (RUWE, NSS,
parallax, G, and every neighbour bright enough to matter) plus a 2" cone on
``I/358/vclassre`` for the variability class.

Rotational vs flare amplitude.  Per-flare tables (Yang & Liu ``Amp``, Günther
``Ampl``) carry the FLARE amplitude under the same names star tables use for
the ROTATIONAL amplitude.  So a bare ``amp`` resolves to ``flare_amplitude``
in a flare table and to ``rot_amplitude`` in a star table; only explicitly
rotational names (``BVAmp``, ``Rper``, ``Rvar``, ``Sph``, ...) resolve to
``rot_amplitude`` in both.  A config ``columns:`` override wins over either,
provided the column exists.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..metronome.acquire import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_ZERO,
    VIZIER_TAP,
    AcquisitionLog,
    _canon,
    _vizier_cone,
    clean_star_id,
    count_rows,
    list_tables,
    reset_route_state,
    resolve_columns,
    route_log_summary,
    search_tables,
    table_columns,
    tap_query,
    unquote_table,
)

ROT_AMPLITUDE_EXPLICIT = [r"^bv_?amp$", r"^rper$", r"^r_?per$", r"^rvar$", r"^r_?var$",
                          r"^sph$", r"^s_?ph$", r"^avar$", r"^a_?var$", r"^rot_?amp$",
                          r"^amp_?rot$", r"^amp_?var$", r"^var_?amp$", r"^dfrot$"]
AMPLITUDE_BARE = [r"^amp$", r"^ampl$", r"^amplitude$", r"^a$"]

ROLE_PATTERNS: dict[str, list[str]] = {
    "star_id": [r"^kic$", r"^kic_?id$", r"^kepid$", r"^tic$", r"^tic_?id$", r"^ticid$",
                r"^epic$", r"^id$", r"^star$", r"^name$", r"^kepler$"],
    "t_peak": [r"^t_?peak$", r"^tpk$", r"^peak_?time$", r"^bjd_?peak$", r"^peak$",
               r"^tmax$", r"^t_?max$", r"^time$", r"^bjd$", r"^tflare$", r"^t_?fl$"],
    # ``Begin`` / ``End`` are Yang & Liu 2019's spellings (J/ApJS/241/29/table2:
    # recno, KIC, Q, Begin, End, logE).  The first pass matched ``End`` and
    # not ``Begin``, so every Yang & Liu star carried ``no_peak_times`` and a
    # null quarter in its stage-2 pull list while the start time sat in the
    # table unread (run 35055720417).
    "t_start": [r"^t_?start$", r"^t_?beg(in)?$", r"^start$", r"^begin$", r"^beg$",
                r"^bjd_?start$", r"^tstart$", r"^t_?ini$", r"^t1$", r"^t0$"],
    "t_end": [r"^t_?end$", r"^t_?stop$", r"^end$", r"^stop$", r"^bjd_?end$", r"^t2$"],
    "energy": [r"^e$", r"^ebol$", r"^e_?bol$", r"^energy$", r"^e_?flare$", r"^log_?e$",
               r"^loge$", r"^log_?ebol$", r"^ekp$", r"^e_?kp$", r"^eflare$", r"^e_?tot$",
               r"^ef$", r"^e_?f$"],
    "energy_max": [r"^e_?max$", r"^emax$", r"^log_?emax$", r"^ed_?max$", r"^edmax$",
                   r"^max_?e$"],
    "equiv_duration": [r"^ed$", r"^eqd$", r"^equiv_?dur(ation)?$", r"^ed_?s$"],
    "rot_amplitude": list(ROT_AMPLITUDE_EXPLICIT),
    "flare_amplitude": [r"^f_?amp$", r"^famp$", r"^flare_?amp$", r"^fpeak$", r"^f_?peak$",
                        r"^dflux$"],
    "prot": [r"^prot$", r"^p_?rot$", r"^rot_?per$", r"^rotper$", r"^per$", r"^period$",
             r"^p$"],
    "teff": [r"^teff$", r"^t_?eff$", r"^temp$", r"^teff_?k$"],
    "radius": [r"^rad$", r"^radius$", r"^r_?star$", r"^rstar$", r"^rs$", r"^r_?\*$",
               r"^r$", r"^rsun$"],
    "logg": [r"^logg$", r"^log_?g$"],
    "mass": [r"^mass$", r"^m_?star$", r"^m$"],
    "flag": [r"^flag$", r"^f_?flag$", r"^flags$", r"^n_?flag$", r"^qual(ity)?$",
             r"^bin(ary)?$", r"^mult$", r"^note$", r"^q_?flag$", r"^cflag$", r"^class$"],
    "sector": [r"^sector$", r"^sec$", r"^sectors$", r"^quarter$", r"^q$", r"^camp(aign)?$"],
    "ra": [r"^ra_?icrs$", r"^raj2000$", r"^_?ra$", r"^ra_?deg$", r"^radeg$"],
    "dec": [r"^de_?icrs$", r"^dej2000$", r"^_?dec?$", r"^dec_?deg$", r"^dedeg$"],
    "gaia_id": [r"^gaia$", r"^gaiadr[23]$", r"^dr[23]_?id$", r"^source$", r"^source_?id$",
                r"^gaia_?id$", r"^gaiaedr3$"],
    "ruwe": [r"^ruwe$"],
    "nss": [r"^nss$", r"^non_?single_?star$"],
    "plx": [r"^plx$", r"^parallax$"],
    "gmag": [r"^gmag$", r"^phot_?g_?mean_?mag$", r"^g$"],
    "vari_class": [r"^class$", r"^best_?class_?name$", r"^vartype$", r"^type$", r"^varflag$"],
    "evol": [r"^evol$", r"^evstate$", r"^evol_?state$"],
}


def resolve_arc_columns(columns, kind: str = "flares", overrides: dict | None = None
                        ) -> dict[str, str]:
    """Roles for a table of ``kind`` ``flares`` | ``stars``; overrides win when
    the named column really exists."""
    roles = resolve_columns(columns, ROLE_PATTERNS)
    bare = resolve_columns(columns, {"bare": AMPLITUDE_BARE}).get("bare")
    if bare is not None:
        if kind == "stars" and "rot_amplitude" not in roles:
            roles["rot_amplitude"] = bare
        elif kind == "flares" and "flare_amplitude" not in roles:
            roles["flare_amplitude"] = bare
    have = {str(c): str(c) for c in columns}
    canon = {_canon(c): str(c) for c in columns}
    for role, col in (overrides or {}).items():
        real = have.get(str(col)) or canon.get(_canon(col))
        if real is not None:
            roles[role] = real
    return roles


def score_table(columns, kind: str = "flares", overrides: dict | None = None
                ) -> tuple[int, dict[str, str], str]:
    """Rank a table for its role: per-flare energies, or per-star context."""
    roles = resolve_arc_columns(columns, kind, overrides)
    if "star_id" not in roles:
        return 0, roles, "rejected: no star_id"
    if kind == "flares":
        if not ({"energy", "equiv_duration", "energy_max"} & set(roles)):
            return 0, roles, "rejected: no energy / equivalent duration"
        score = 10
        score += 4 if "energy" in roles else 0
        score += 3 if "t_peak" in roles else 0
        score += 1 if "t_start" in roles else 0
        score += 2 if "rot_amplitude" in roles else 0
        score += 1 if "prot" in roles else 0
        score += 1 if "teff" in roles else 0
        score += 1 if "radius" in roles else 0
        score += 1 if "sector" in roles else 0
        return score, roles, "usable"
    if not ({"rot_amplitude", "prot", "radius"} & set(roles)):
        return 0, roles, "rejected: no rotational amplitude / period / radius"
    score = 10
    score += 4 if "rot_amplitude" in roles else 0
    score += 2 if "prot" in roles else 0
    score += 1 if "teff" in roles else 0
    score += 1 if "radius" in roles else 0
    score += 1 if "logg" in roles else 0
    score += 1 if "flag" in roles else 0
    return score, roles, "usable"


@dataclass
class DiscoveredTable:
    catalogue: str
    kind: str
    table: str | None
    columns: list[str]
    roles: dict[str, str]
    n_rows: int | None
    status: str
    route: str = "preferred"
    scoreboard: list[dict] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"catalogue": self.catalogue, "kind": self.kind, "table": self.table,
                "n_columns": len(self.columns), "columns": self.columns[:80],
                "roles": self.roles, "n_rows": self.n_rows, "status": self.status,
                "route": self.route, "scoreboard": self.scoreboard, "note": self.note}


def discover_table(catalogue: str, preferred: str, kind: str = "flares", keywords=(), *,
                   query_fn=None, log: AcquisitionLog | None = None,
                   overrides: dict | None = None, exact: bool = False,
                   fetch_fn=None) -> DiscoveredTable:
    """List the tables under ``preferred`` (a catalogue id or a full table name),
    score each for ``kind``, fall back to a keyword search.  Row counts break
    score ties toward the per-flare table.

    The listing is ``seti.metronome.acquire.list_tables``, which asks
    ``TAP_SCHEMA.tables`` first and, when no TAP host answers, asks VizieR's
    NON-TAP ASU interface for one row of ``-source=<preferred>`` instead ---
    ASU has no ``TAP_SCHEMA``, so a well-formed TSV body is the existence
    proof and its header is the column list.  Those columns are then handed to
    :func:`table_columns` as ``known``, so a TAP-down run resolves the roles
    without a second round trip and without ever inventing a column name.
    ``fetch_fn`` (URL -> text) is injectable for the offline tests.
    """
    query_fn = query_fn or tap_query
    board: list[dict] = []
    best: DiscoveredTable | None = None
    best_key: tuple = (-1, -1)
    routes = []
    if preferred:
        routes.append(("preferred",
                       lambda: list_tables(preferred, query_fn=query_fn, fetch_fn=fetch_fn)))
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
                           error=repr(exc),
                           extra={"route": "none",
                                  "route_attempts": list(getattr(exc, "attempts", []))})
            continue
        if log:
            log.record(f"discover_{catalogue}_{route}", f"TAP_SCHEMA.tables ~ {preferred!r}",
                       rows=int(len(tabs)),
                       extra={"route": str(tabs.attrs.get("route", "tap")),
                              "route_attempts": list(tabs.attrs.get("attempts", []))})
        listing_route = str(tabs.attrs.get("route", "tap"))
        for _, row in tabs.iterrows():
            t = unquote_table(row["table_name"])
            if exact and route == "preferred" and t != unquote_table(preferred):
                continue
            raw = row["columns"] if "columns" in tabs.columns else None
            known = [str(c) for c in raw] if isinstance(raw, (list, tuple)) else []
            if known and listing_route == "asu_tsv":
                # The non-TAP listing carries the service's own header, which
                # IS the column list (``-out.all``): asking TAP for
                # TAP_SCHEMA.columns of the same table would only repeat the
                # outage that sent us here.
                cols = known
            else:
                try:
                    cols = table_columns(t, query_fn=query_fn, fetch_fn=fetch_fn, known=known)
                except Exception as exc:                  # noqa: BLE001
                    board.append({"table": t, "route": route, "score": 0,
                                  "reason": f"columns query failed: {exc!r}"[:200]})
                    continue
            score, roles, reason = score_table(cols, kind, overrides)
            entry = {"table": t, "route": route, "score": int(score), "reason": reason,
                     "roles": roles, "n_columns": len(cols),
                     # The NAMES, not just the count: a rejected table that
                     # records only "no star_id" and "19 columns" cannot be
                     # fixed without another round trip.  Okamoto+2021's
                     # table2 scored 0 as a flares table and resolved cleanly
                     # as a stars table in the same run, which is only
                     # diagnosable with the header in hand.
                     "columns": [str(c) for c in cols][:60]}
            if score > 0:
                try:
                    entry["n_rows"] = count_rows(t, query_fn=query_fn)
                except Exception as exc:                  # noqa: BLE001
                    entry["n_rows"] = None
                    entry["count_error"] = repr(exc)[:200]
            board.append(entry)
            if score > 0:
                n = entry.get("n_rows") or 0
                key = (score + (2 if (kind == "flares" and n >= 500) else 0), n)
                if key > best_key:
                    best_key = key
                    best = DiscoveredTable(catalogue, kind, t, cols, roles, entry.get("n_rows"),
                                           STATUS_OK, route=route)
        if best is not None:
            break
    if best is None:
        status = STATUS_FAILED if any_failed and not board else STATUS_ZERO
        return DiscoveredTable(catalogue, kind, None, [], {}, None, status, route="none",
                               scoreboard=board,
                               note=f"no table under {preferred!r} or the keyword search "
                                    f"exposes the roles a {kind} table needs")
    best.scoreboard = board
    if best.n_rows == 0:
        best.status = STATUS_ZERO
    return best


def fetch_table(disc: DiscoveredTable, *, query_fn=None, log: AcquisitionLog | None = None,
                chunk_rows: int = 50000, max_rows: int | None = None) -> pd.DataFrame:
    """Pull every resolved role column, in ``recno`` chunks when the table has
    one, renamed to the role names.  Numeric roles are coerced; ``star_id`` is
    a stripped string.

    Which ROUTE served each chunk (``tap`` / ``asu_tsv`` / ``astroquery``) is
    read off the frame the query returned and recorded on every log stage and,
    for the caller, in ``out.attrs["route"]`` / ``out.attrs["routes"]``: a run
    that got its rows while TAPVizieR was down must be able to say so.
    """
    query_fn = query_fn or tap_query
    if disc.table is None or not disc.roles:
        return pd.DataFrame()
    roles = dict(disc.roles)
    sel = ", ".join(f'"{c}"' for c in dict.fromkeys(roles.values()))
    has_recno = any(_canon(c) == "recno" for c in disc.columns)
    n_total = disc.n_rows
    limit = int(max_rows) if max_rows else None
    frames = []
    routes: list[str] = []
    label = f"fetch_{disc.catalogue}"

    def _route_extra(df) -> dict:
        attrs = getattr(df, "attrs", {}) or {}
        route = str(attrs.get("route") or "unknown")
        routes.append(route)
        return {"route": route, "endpoint": str(attrs.get("endpoint", ""))}

    if has_recno and n_total and n_total > chunk_rows:
        top = n_total if limit is None else min(n_total, limit)
        lo = 1
        while lo <= top:
            hi = min(lo + chunk_rows - 1, top)
            adql = f'SELECT {sel} FROM "{disc.table}" WHERE recno BETWEEN {lo} AND {hi}'
            try:
                df = query_fn(adql)
            except Exception as exc:                      # noqa: BLE001
                if log:
                    log.record(label, adql, error=repr(exc),
                               extra={"chunk": [lo, hi], "route": "none",
                                      "route_attempts": list(getattr(exc, "attempts", []))})
                lo = hi + 1
                continue
            n = int(len(df)) if df is not None else 0
            if log:
                log.record(label, adql, rows=n, extra={"chunk": [lo, hi], **_route_extra(df)})
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
                log.record(label, adql, error=repr(exc),
                           extra={"route": "none",
                                  "route_attempts": list(getattr(exc, "attempts", []))})
            return pd.DataFrame()
        n = int(len(df)) if df is not None else 0
        if log:
            log.record(label, adql, rows=n, extra=_route_extra(df))
        if n:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    inv: dict = {}
    for role, col in roles.items():
        for c in out.columns:
            if str(c) == col or _canon(c) == _canon(col):
                inv.setdefault(c, role)
    out = out.rename(columns=inv)
    for c in ("t_peak", "t_start", "t_end", "energy", "energy_max", "equiv_duration",
              "rot_amplitude", "flare_amplitude", "prot", "teff", "radius", "logg", "mass",
              "ra", "dec", "ruwe", "nss", "plx", "gmag"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "star_id" in out.columns:
        out["star_id"] = out["star_id"].map(_clean_id)
    seen = list(dict.fromkeys(r for r in routes if r))
    out.attrs["routes"] = list(routes)
    out.attrs["route"] = seen[0] if len(seen) == 1 else ("mixed" if seen else "unknown")
    return out


#: One id spelling for every table (shared with METRONOME, which joins the same
#: rotation tables to the same flare tables).
_clean_id = clean_star_id


# ---------------------------------------------------------------------------
# stellar parameters and positions for the shortlist, by id
# ---------------------------------------------------------------------------
#: Berger+2020 (J/AJ/159/280) is TWO tables: ``table1`` carries the Gaia
#: cross-match (KIC, Gaia id, RUWE, positions) and ``table2`` the derived
#: Teff / logg / [Fe/H] / R* / M* / evolutionary state.  Run 35055720417 asked
#: only ``table1`` and the bare catalogue id, got positions and RUWE, and
#: left every Yang & Liu star at ``stellar_params_assumed`` with a solar
#: radius.  ``table2`` has no position columns, which is why the merge below
#: no longer requires ``ra`` / ``dec`` of every table: positions come from the
#: first table that has them, each parameter from the first table that has it.
PARAM_TABLES = {"kepler": ["J/AJ/159/280/table2", "J/AJ/159/280/table1", "V/133/kic"],
                "tess": ["IV/39/tic82", "IV/38/tic"]}
PARAM_VALUE_COLS = ("ra", "dec", "teff", "radius", "logg", "mass", "flag", "gaia_id", "ruwe",
                    "evol")
_NEED = ("ra", "dec", "teff", "radius")


def fetch_star_params_by_id(ids, mission: str, *, query_fn=None,
                            log: AcquisitionLog | None = None, chunk: int = 200,
                            tables: dict | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Positions and Teff / radius / logg / mass / flags / Gaia id for a
    shortlist of KIC or TIC ids, merged across the tables in priority order.

    A table must expose ``star_id`` and either a position or a stellar
    parameter.  Per star, the position comes from the first table that has
    one and each parameter from the first table that has it, with
    ``<col>_source`` naming the table; a later table is asked only about the
    stars still missing a position, a Teff or a radius.  Kepler: Berger+2020
    ``table2`` (parameters), ``table1`` (Gaia id, RUWE, positions), then the
    KIC; TESS: the TIC.  Every table's columns are read from ``TAP_SCHEMA``
    first."""
    query_fn = query_fn or tap_query
    tables = tables or PARAM_TABLES
    ids = [str(i).strip() for i in ids if str(i).strip()]
    record: list[dict] = []
    if not ids:
        return pd.DataFrame(columns=["star_id", "ra", "dec"]), record
    merged: dict[str, dict] = {}

    def _done(sid: str) -> bool:
        m = merged.get(sid)
        return bool(m) and all(_is_finite(m.get(c)) for c in _NEED)

    for t in tables.get(mission, []):
        want = [i for i in ids if not _done(i)]
        if not want:
            break
        try:
            cols = table_columns(t, query_fn=query_fn)
        except Exception as exc:                          # noqa: BLE001
            record.append({"table": t, "status": STATUS_FAILED, "error": repr(exc)[:200]})
            if log:
                log.record(f"params_{mission}", f"columns of {t}", error=repr(exc))
            continue
        if not cols:
            record.append({"table": t, "status": STATUS_ZERO, "note": "no columns"})
            continue
        roles = resolve_arc_columns(cols, "stars")
        has_pos = {"ra", "dec"} <= set(roles)
        has_par = bool({"teff", "radius", "logg", "mass"} & set(roles))
        if "star_id" not in roles or not (has_pos or has_par):
            record.append({"table": t, "status": "ROLES_UNRESOLVED", "roles": roles})
            continue
        keep = [r for r in ("star_id",) + PARAM_VALUE_COLS if r in roles]
        sel = ", ".join(f'"{roles[r]}"' for r in keep)
        n_rows = 0
        for i in range(0, len(want), int(chunk)):
            block = want[i:i + int(chunk)]
            numeric = all(re.fullmatch(r"\d+", b) for b in block)
            vals = ", ".join(b if numeric else f"'{b}'" for b in block)
            adql = f'SELECT {sel} FROM "{t}" WHERE "{roles["star_id"]}" IN ({vals})'
            try:
                df = query_fn(adql)
            except Exception as exc:                      # noqa: BLE001
                if log:
                    log.record(f"params_{mission}", adql[:300], error=repr(exc))
                continue
            n = int(len(df)) if df is not None else 0
            if log:
                log.record(f"params_{mission}", adql[:300], rows=n)
            if not n:
                continue
            out = pd.DataFrame({r: df.iloc[:, j] for j, r in enumerate(keep)})
            out["star_id"] = out["star_id"].map(_clean_id)
            for c in ("ra", "dec", "teff", "radius", "logg", "mass", "ruwe"):
                if c in out:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
            for _, row in out.iterrows():
                sid = str(row["star_id"])
                m = merged.setdefault(sid, {"star_id": sid})
                for c in PARAM_VALUE_COLS:
                    if c not in out.columns:
                        continue
                    v = row[c]
                    if _is_null(v) or _is_finite(m.get(c)) or (m.get(c) not in (None, "")
                                                               and c in ("flag", "gaia_id",
                                                                         "evol")):
                        continue
                    if c in ("ra", "dec"):
                        # a position is taken as a PAIR from one table
                        if not (_is_finite(row.get("ra")) and _is_finite(row.get("dec"))):
                            continue
                    m[c] = v
                    m[f"{c}_source"] = t
                m.setdefault("params_source", t)
            n_rows += n
        record.append({"table": t, "status": STATUS_OK if n_rows else STATUS_ZERO,
                       "n_rows": n_rows, "roles": {r: roles[r] for r in keep},
                       "has_positions": has_pos, "has_parameters": has_par})
    if not merged:
        return pd.DataFrame(columns=["star_id", "ra", "dec"]), record
    out = pd.DataFrame(list(merged.values()))
    for c in ("ra", "dec", "teff", "radius", "logg", "mass", "ruwe"):
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in out.columns else np.nan
    return out, record


def _is_null(v) -> bool:
    if v is None:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except TypeError:
        return False
    return isinstance(v, str) and v.strip().lower() in ("", "nan", "none", "<na>")


def _is_finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Gaia DR3 context by cone (RUWE, NSS, neighbours, variability class)
# ---------------------------------------------------------------------------
def _sep_arcsec(ra0, dec0, ra, dec):
    d2r = math.pi / 180.0
    dra = (np.asarray(ra, dtype=float) - float(ra0)) * math.cos(float(dec0) * d2r)
    ddec = np.asarray(dec, dtype=float) - float(dec0)
    return np.hypot(dra, ddec) * 3600.0


def parse_gaia_cone(df: pd.DataFrame, ra: float, dec: float, *, match_arcsec: float = 2.0
                    ) -> dict:
    """The target (nearest source within ``match_arcsec``) and every other
    source in the cone, from a ``I/355/gaiadr3`` cone result."""
    out = {"target_found": False, "ruwe": float("nan"), "nss": float("nan"),
           "plx": float("nan"), "gmag": float("nan"), "gaia_source": None,
           "target_sep_arcsec": float("nan"), "neighbours": [], "n_in_cone": 0}
    if df is None or not len(df):
        return out
    roles = resolve_columns(df.columns, {k: ROLE_PATTERNS[k] for k in
                                         ("ra", "dec", "ruwe", "nss", "plx", "gmag", "gaia_id")})
    if "ra" not in roles or "dec" not in roles:
        return out
    sep = _sep_arcsec(ra, dec, pd.to_numeric(df[roles["ra"]], errors="coerce"),
                      pd.to_numeric(df[roles["dec"]], errors="coerce"))
    out["n_in_cone"] = int(len(df))
    order = np.argsort(sep)
    i0 = int(order[0])
    g = pd.to_numeric(df[roles["gmag"]], errors="coerce") if "gmag" in roles else None
    if np.isfinite(sep[i0]) and sep[i0] <= float(match_arcsec):
        out["target_found"] = True
        out["target_sep_arcsec"] = float(sep[i0])
        for role in ("ruwe", "nss", "plx"):
            if role in roles:
                out[role] = float(pd.to_numeric(df[roles[role]].iloc[i0], errors="coerce"))
        if g is not None:
            out["gmag"] = float(g.iloc[i0])
        if "gaia_id" in roles:
            out["gaia_source"] = str(df[roles["gaia_id"]].iloc[i0])
        rest = [int(i) for i in order[1:]]
    else:
        rest = [int(i) for i in order]
    out["neighbours"] = [(float(sep[i]), float(g.iloc[i]) if g is not None else float("nan"))
                         for i in rest]
    return out


def gaia_context(positions: pd.DataFrame, *, cone_fn=None, log: AcquisitionLog | None = None,
                 gaia_table: str = "I/355/gaiadr3", vari_table: str = "I/358/vclassre",
                 radius_arcsec: float = 12.0, match_arcsec: float = 2.0) -> dict:
    """Per star: ``{gaia_reached, ruwe, nss, plx, gmag, neighbours, vari_class, ...}``.
    A star whose Gaia cone failed carries ``gaia_reached = False`` (the veto
    is then *unapplied*, not passed); a failed variability cone only leaves
    ``vari_class`` as ``None`` with ``vari_reached = False``."""
    cone_fn = cone_fn or _vizier_cone
    out: dict[str, dict] = {}
    n_ok = n_fail = n_vok = n_vfail = 0
    for _, r in positions.iterrows():
        sid = str(r["star_id"])
        ra, dec = float(r.get("ra", np.nan)), float(r.get("dec", np.nan))
        ctx = {"gaia_reached": False, "vari_reached": False, "vari_class": None,
               "ruwe": float("nan"), "nss": float("nan"), "plx": float("nan"),
               "gmag": float("nan"), "neighbours": [], "target_found": False}
        if not (np.isfinite(ra) and np.isfinite(dec)):
            ctx["note"] = "no position"
            out[sid] = ctx
            continue
        try:
            df = cone_fn(gaia_table, ra, dec, radius_arcsec)
            ctx.update(parse_gaia_cone(df, ra, dec, match_arcsec=match_arcsec))
            ctx["gaia_reached"] = True
            n_ok += 1
        except Exception as exc:                          # noqa: BLE001
            n_fail += 1
            ctx["gaia_error"] = repr(exc)[:200]
            if log and n_fail <= 3:
                log.record("gaia_cone", f"cone {gaia_table} ({ra:.5f},{dec:.5f})",
                           error=repr(exc))
        if vari_table:
            try:
                dv = cone_fn(vari_table, ra, dec, match_arcsec)
                ctx["vari_reached"] = True
                n_vok += 1
                if dv is not None and len(dv):
                    vr = resolve_columns(dv.columns, {"vari_class": ROLE_PATTERNS["vari_class"]})
                    if "vari_class" in vr:
                        ctx["vari_class"] = str(dv[vr["vari_class"]].iloc[0])
            except Exception as exc:                      # noqa: BLE001
                n_vfail += 1
                ctx["vari_error"] = repr(exc)[:200]
        out[sid] = ctx
    if log:
        log.record("gaia_cone", f"{n_ok + n_fail} cones on {gaia_table} r={radius_arcsec}\"",
                   rows=n_ok if n_ok else None, error=None if n_ok or not n_fail else
                   "every cone failed", extra={"n_cones_ok": n_ok, "n_cones_failed": n_fail})
        if vari_table:
            log.record("vari_cone", f"{n_vok + n_vfail} cones on {vari_table} r={match_arcsec}\"",
                       rows=n_vok if n_vok else None,
                       error=None if n_vok or not n_vfail else "every cone failed",
                       extra={"n_cones_ok": n_vok, "n_cones_failed": n_vfail})
    return out


__all__ = ["PARAM_TABLES", "ROLE_PATTERNS", "STATUS_FAILED", "STATUS_OK", "STATUS_ZERO",
           "VIZIER_TAP", "AcquisitionLog", "DiscoveredTable", "discover_table",
           "fetch_star_params_by_id", "fetch_table", "gaia_context", "parse_gaia_cone",
           "reset_route_state", "resolve_arc_columns", "route_log_summary", "score_table",
           "tap_query"]

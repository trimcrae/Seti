"""Runner-only archive access for FORGE, with runtime schema discovery.

Everything network-facing takes an injectable callable (``query_fn`` for ADQL
against VizieR, ``cone_fn`` for a positional query, ``tap_fn`` for Simbad,
``run_query`` for the ESA Gaia archive) so a failed or empty archive can be
simulated offline; the tests never open a socket.

Routes (all shared with ARC / METRONOME, :mod:`seti.metronome.acquire`):
TAPVizieR and its mirrors, then the non-TAP ASU interface with the ReadMe
inventory as the table-existence check, then ``astroquery.vizier``.  No column
name goes into a query that was not seen in the archive's own listing.

What is FORGE's own here:

* the role set of an interferometric excess table -- a star identifier (HD /
  HIP / name), the excess (``fCSE``, ``f_disk``, ``Excess``, ``Null``...) and
  its error (VizieR's ``e_<col>`` convention first), an optional significance,
  a band, and the detection flag;
* verification of the embedded rows against what the archive returns;
* star positions and spectral types from Simbad TAP by identifier;
* the companion context (WDS within the interferometric field, Gaia DR3
  where it has the star);
* the broadband leg: 2MASS / AllWISE cones for the sample and the Gaia DR3
  x 2MASS x AllWISE join for the < 30 pc population.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..arc.acquire import gaia_context
from ..metronome.acquire import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_ZERO,
    AcquisitionLog,
    _canon,
    _vizier_cone,
    count_rows,
    list_tables,
    resolve_columns,
    search_tables,
    table_columns,
    tap_query,
    unquote_table,
)
from .physics import BAND_WAVELENGTH_UM, Measurement
from .tables import TargetTable, normalise_name

STATUS_NOT_ATTEMPTED = "DISCOVERY_NOT_ATTEMPTED"
#: discovery started but ran out of its own clock: not a statement about the
#: archive, and never to be read as "the catalogue holds nothing usable"
STATUS_TRUNCATED = "DISCOVERY_TRUNCATED_BY_BUDGET"

ROLE_PATTERNS: dict[str, list[str]] = {
    "hd": [r"^hd$", r"^hd_?(id|num(ber)?)$", r"^hdn$"],
    "hip": [r"^hip$", r"^hip_?(id|num(ber)?)$"],
    "name": [r"^name$", r"^star$", r"^target$", r"^object$", r"^id$", r"^sname$",
             r"^simbad(name)?$", r"^source$", r"^ident$", r"^bayer$", r"^oname$"],
    "excess": [r"^f_?cse$", r"^fcse$", r"^f_?disk$", r"^fdisk$", r"^f_?dust$", r"^fdust$",
               r"^f_?exc(ess)?$", r"^excess$", r"^fex$", r"^null_?exc(ess)?$", r"^nexc$",
               r"^n_?excess$", r"^exc$", r"^e_?n$", r"^en$", r"^null$", r"^ex$",
               r"^f_?h$", r"^fh$", r"^f_?k$", r"^fk$", r"^f_?cse_?h$", r"^f_?cse_?k$",
               r"^f$", r"^ratio$", r"^flux_?ratio$"],
    "significance": [r"^chi$", r"^chi_?f(cse)?$", r"^snr$", r"^sig(nif(icance)?)?$",
                     r"^n_?sig(ma)?$", r"^sigma$", r"^signif$", r"^det_?sig$"],
    "band": [r"^band$", r"^filter$", r"^lambda$", r"^wave(length)?$", r"^lam$"],
    "teff": [r"^teff$", r"^t_?eff$"],
    "sptype": [r"^sp_?type$", r"^spt$", r"^sp$", r"^type$", r"^sptyp$"],
    "flag": [r"^n_?f_?cse$", r"^flag$", r"^det(ection)?$", r"^f_?det$", r"^note$", r"^n_?f$",
             r"^l_?f_?cse$", r"^l_?excess$", r"^l_?null$", r"^status$", r"^class$", r"^exc_?flag$"],
    "limit_flag": [r"^l_?f_?cse$", r"^l_?f$", r"^l_?excess$", r"^l_?null$", r"^l_?fdisk$",
                   r"^l_?exc$", r"^l_?p$", r"^l_?pol$", r"^l_?p_?ppm$"],
    "polarisation": [r"^p$", r"^pol$", r"^p_?pct$", r"^ppm$", r"^p_?ppm$", r"^q$", r"^u$",
                     r"^pol_?deg$", r"^p_?bar$"],
    "zodi": [r"^z$", r"^zodi$", r"^n_?zodi$", r"^zodis$", r"^z_?level$"],
    "ra": [r"^ra_?icrs$", r"^raj2000$", r"^_?ra$", r"^ra_?deg$", r"^radeg$"],
    "dec": [r"^de_?icrs$", r"^dej2000$", r"^_?dec?$", r"^dec_?deg$", r"^dedeg$"],
}

_NUMERIC_ROLES = ("excess", "excess_err", "significance", "teff", "polarisation",
                  "polarisation_err", "zodi", "zodi_err", "ra", "dec")


def resolve_excess_columns(columns, overrides: dict | None = None) -> dict[str, str]:
    """Roles for an excess table.  ``excess_err`` is ``e_<excess>`` when that
    column exists (VizieR's convention), else a generic error column.
    Overrides win when the named column really exists."""
    cols = [str(c) for c in columns]
    roles = resolve_columns(cols, ROLE_PATTERNS)
    canon = {_canon(c): c for c in cols}
    for base in ("excess", "polarisation", "zodi"):
        col = roles.get(base)
        if col is None:
            continue
        k = _canon(col)
        for cand in (f"e_{k}", f"e{k}", f"{k}_err", f"err_{k}", f"{k}_error", f"sig_{k}",
                     f"d{k}", f"{k}_e", f"s_{k}", f"sigma_{k}"):
            if cand in canon:
                roles[f"{base}_err"] = canon[cand]
                break
    if "excess" in roles and "excess_err" not in roles:
        for cand in ("e_f", "err", "error", "e_excess", "sigma", "unc", "e_fcse", "e_null", "e_ex"):
            if cand in canon and canon[cand] != roles["excess"]:
                roles["excess_err"] = canon[cand]
                break
    have = {str(c): str(c) for c in cols}
    for role, col in (overrides or {}).items():
        real = have.get(str(col)) or canon.get(_canon(col))
        if real is not None:
            roles[role] = real
    if "hd" not in roles and "hip" not in roles and "name" not in roles:
        # a lone identifier column sometimes carries the catalogue's own name
        for c in cols:
            if re.search(r"^(hd|hip|star|name|target)", _canon(c)):
                roles["name"] = c
                break
    return roles


def score_excess_table(columns, role: str = "excess", overrides: dict | None = None
                       ) -> tuple[int, dict[str, str], str]:
    """Rank a table: it must name the star and carry the quantity ``role``
    needs (an excess, a polarisation, or -- for a models table -- anything)."""
    roles = resolve_excess_columns(columns, overrides)
    if not ({"hd", "hip", "name"} & set(roles)):
        return 0, roles, "rejected: no star identifier"
    score = 10
    if role == "excess":
        if "excess" not in roles:
            if "zodi" in roles:
                return 0, roles, "rejected: zodi level only (needs the paper's model to convert)"
            return 0, roles, "rejected: no excess column"
        score += 4 if "excess_err" in roles else 0
        score += 2 if "significance" in roles else 0
        score += 1 if "flag" in roles else 0
        score += 1 if "sptype" in roles else 0
        score += 1 if "hd" in roles else 0
    elif role == "polarimetry":
        if "polarisation" not in roles:
            return 0, roles, "rejected: no polarisation column"
        score += 3 if "polarisation_err" in roles else 0
    else:
        score += 1
    return score, roles, "usable"


@dataclass
class DiscoveredTable:
    name: str
    role: str
    table: str | None
    columns: list[str]
    roles: dict[str, str]
    n_rows: int | None
    status: str
    route: str = "preferred"
    scoreboard: list[dict] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "role": self.role, "table": self.table,
                "n_columns": len(self.columns), "columns": self.columns[:80],
                "roles": self.roles, "n_rows": self.n_rows, "status": self.status,
                "route": self.route, "scoreboard": self.scoreboard, "note": self.note}


def discover_table(name: str, preferred: str, role: str = "excess", keywords=(), *,
                   query_fn=None, log: AcquisitionLog | None = None,
                   overrides: dict | None = None, fetch_fn=None,
                   budget_s: float | None = None) -> DiscoveredTable:
    """List every table under ``preferred``, score each for ``role``, fall back
    to a keyword search of the table descriptions.  The listing degrades to
    the ReadMe inventory when no TAP host answers (``list_tables``).

    ``budget_s`` is a cooperative clock on THIS table's discovery, checked
    between calls.  Without it one catalogue can eat a whole job: the route
    ladder (TAP mirrors, then ASU, then astroquery, each retrying) runs once
    per route, and then a column query plus a row count runs once per table in
    the catalogue, so a slow VizieR turns a "cheap" probe into hours.  The
    caller's overall budget only decides whether to START a table; this
    decides when to stop one.  Whatever was scored before the clock ran out is
    kept -- a partial scoreboard is a real result -- and the status says the
    discovery was truncated rather than that the catalogue is empty.
    """
    query_fn = query_fn or tap_query
    t_start = time.monotonic()

    def _out_of_time() -> bool:
        return budget_s is not None and (time.monotonic() - t_start) > float(budget_s)

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
    truncated = False
    for route, lister in routes:
        if _out_of_time():
            truncated = True
            break
        try:
            tabs = lister()
        except Exception as exc:                          # noqa: BLE001
            any_failed = True
            if log:
                log.record(f"discover_{name}_{route}", f"TAP_SCHEMA.tables ~ {preferred!r}",
                           error=repr(exc),
                           extra={"route": "none",
                                  "route_attempts": list(getattr(exc, "attempts", []))})
            continue
        if log:
            log.record(f"discover_{name}_{route}", f"TAP_SCHEMA.tables ~ {preferred!r}",
                       rows=int(len(tabs)),
                       extra={"route": str(tabs.attrs.get("route", "tap")),
                              "route_attempts": list(tabs.attrs.get("attempts", []))})
        listing_route = str(tabs.attrs.get("route", "tap"))
        for _, row in tabs.iterrows():
            if _out_of_time():
                truncated = True
                break
            t = unquote_table(row["table_name"])
            raw = row["columns"] if "columns" in tabs.columns else None
            known = [str(c) for c in raw] if isinstance(raw, (list, tuple)) else []
            if known and listing_route == "asu_tsv":
                cols = known
            else:
                try:
                    cols = table_columns(t, query_fn=query_fn, fetch_fn=fetch_fn, known=known)
                except Exception as exc:                  # noqa: BLE001
                    board.append({"table": t, "route": route, "score": 0,
                                  "reason": f"columns query failed: {exc!r}"[:200]})
                    continue
            score, roles, reason = score_excess_table(cols, role, overrides)
            entry = {"table": t, "route": route, "score": int(score), "reason": reason,
                     "roles": roles, "n_columns": len(cols),
                     "columns": [str(c) for c in cols][:60],
                     "description": str(row.get("description", ""))[:200]}
            if score > 0:
                try:
                    entry["n_rows"] = count_rows(t, query_fn=query_fn)
                except Exception as exc:                  # noqa: BLE001
                    entry["n_rows"] = None
                    entry["count_error"] = repr(exc)[:200]
            board.append(entry)
            if score > 0:
                n = entry.get("n_rows") or 0
                key = (score, n)
                if key > best_key:
                    best_key = key
                    best = DiscoveredTable(name, role, t, cols, roles, entry.get("n_rows"),
                                           STATUS_OK, route=route)
        if best is not None or truncated:
            break
    if best is None:
        # A truncated discovery is NOT "the catalogue holds nothing usable":
        # it is "this run did not finish looking", and it must not read as a
        # statement about the archive.
        if truncated:
            return DiscoveredTable(
                name, role, None, [], {}, None, STATUS_TRUNCATED, route="none",
                scoreboard=board,
                note=f"discovery budget {budget_s:.0f}s exhausted after "
                     f"{len(board)} table(s); not a statement about {preferred!r}")
        status = STATUS_FAILED if any_failed and not board else STATUS_ZERO
        return DiscoveredTable(name, role, None, [], {}, None, status, route="none",
                               scoreboard=board,
                               note=f"no table under {preferred!r} or the keyword search "
                                    f"exposes the roles a {role} table needs")
    best.scoreboard = board
    if truncated:
        best.note = (f"discovery budget {budget_s:.0f}s exhausted after {len(board)} "
                     "table(s); a better-scoring table may not have been reached")
    if best.n_rows == 0:
        best.status = STATUS_ZERO
    return best


# ---------------------------------------------------------------------------
# fetching an excess table into Measurements
# ---------------------------------------------------------------------------
def _to_float(v) -> float:
    try:
        s = str(v).strip()
        if s in ("", "--", "nan", "NaN", "None"):
            return float("nan")
        return float(s.replace("<", "").replace(">", ""))
    except (TypeError, ValueError):
        return float("nan")


def _is_limit(v, flag) -> bool:
    s = str(v).strip()
    f = str(flag).strip() if flag is not None else ""
    return s.startswith("<") or f == "<" or f.lower() in ("ul", "upper", "limit")


def fetch_excess_table(disc: DiscoveredTable, spec: dict, targets: TargetTable, *,
                       query_fn=None, log: AcquisitionLog | None = None,
                       max_rows: int = 5000) -> tuple[dict[str, list[Measurement]], dict]:
    """Rows of one discovered table as :class:`Measurement` sets keyed by the
    HD key.  Stars the targets asset does not know are ADDED to it (keyed by
    HD when the table carries HD numbers, else by their own name) so the
    archive, not the transcription, defines the sample.

    ``spec`` (from config) says what the excess column means: ``band``,
    ``wl_um``, ``unit`` (``percent`` / ``fraction``), ``instrument``, ``epoch``.
    """
    query_fn = query_fn or tap_query
    record = {"name": disc.name, "table": disc.table, "status": STATUS_FAILED, "n_rows": 0,
              "n_matched": 0, "n_added": 0, "n_measurements": 0, "unmatched": []}
    if disc.table is None or not disc.roles:
        record["status"] = disc.status
        return {}, record
    roles = dict(disc.roles)
    want = [r for r in ("hd", "hip", "name", "excess", "excess_err", "significance", "band",
                        "flag", "limit_flag", "sptype", "teff", "ra", "dec") if r in roles]
    sel = ", ".join(f'"{roles[r]}"' for r in dict.fromkeys(want))
    adql = f'SELECT TOP {int(max_rows)} {sel} FROM "{disc.table}"'
    try:
        df = query_fn(adql)
    except Exception as exc:                              # noqa: BLE001
        if log:
            log.record(f"fetch_{disc.name}", adql, error=repr(exc))
        record["error"] = repr(exc)[:300]
        return {}, record
    n = int(len(df)) if df is not None else 0
    if log:
        log.record(f"fetch_{disc.name}", adql, rows=n,
                   extra={"route": str(getattr(df, "attrs", {}).get("route", "unknown"))})
    record["n_rows"] = n
    record["status"] = STATUS_OK if n else STATUS_ZERO
    if not n:
        return {}, record
    # the frame comes back with the real names (any route may re-case them)
    colmap = {}
    for r in want:
        for c in df.columns:
            if str(c) == roles[r] or _canon(c) == _canon(roles[r]):
                colmap[r] = c
                break
    unit = str(spec.get("unit", "percent")).lower()
    scale = 100.0 if unit == "fraction" else 1.0
    band = str(spec.get("band", "K"))
    wl = float(spec.get("wl_um") or BAND_WAVELENGTH_UM.get(band, 2.2))
    out: dict[str, list[Measurement]] = {}
    for _, row in df.iterrows():
        key = None
        for r, kind in (("hd", "hd"), ("hip", "hip"), ("name", "name")):
            if r in colmap:
                v = row[colmap[r]]
                if kind == "hd":
                    key = targets.key_for(f"HD {v}") if re.fullmatch(r"\s*\d+(\.0)?\s*", str(v)) \
                        else targets.key_for(v)
                elif kind == "hip":
                    key = targets.key_for(f"HIP {v}") if re.fullmatch(r"\s*\d+(\.0)?\s*", str(v)) \
                        else targets.key_for(v)
                else:
                    key = targets.key_for(v)
                if key:
                    break
        if key is None:
            # an archive star the asset does not list: add it under its HD
            # number when the table has one, else under its own name
            hd = _to_float(row[colmap["hd"]]) if "hd" in colmap else float("nan")
            if math.isfinite(hd) and hd > 0:
                key = f"HD {int(hd)}"
            else:
                nm = str(row[colmap["name"]]).strip() if "name" in colmap else ""
                if not nm and "hip" in colmap:
                    nm = f"HIP {row[colmap['hip']]}"
                key = normalise_name(nm).upper() if nm else None
            if not key:
                record["unmatched"].append(str(row.to_dict())[:120])
                continue
            targets.add(key, name=str(row[colmap["name"]]) if "name" in colmap else "",
                        hip=str(row[colmap["hip"]]) if "hip" in colmap else "",
                        sptype=str(row[colmap["sptype"]]) if "sptype" in colmap else "",
                        samples=disc.name, membership_status="archive")
            record["n_added"] += 1
        else:
            record["n_matched"] += 1
        if "excess" not in colmap:
            continue
        raw = row[colmap["excess"]]
        val = _to_float(raw) * scale
        if not math.isfinite(val):
            continue
        err = _to_float(row[colmap["excess_err"]]) * scale if "excess_err" in colmap else float("nan")
        sig = _to_float(row[colmap["significance"]]) if "significance" in colmap else float("nan")
        if not math.isfinite(err) and math.isfinite(sig) and sig > 0:
            err = abs(val) / sig
        limit = _is_limit(raw, row[colmap["limit_flag"]] if "limit_flag" in colmap else None)
        if not math.isfinite(err) and not limit:
            continue
        b = band
        if "band" in colmap:
            bb = str(row[colmap["band"]]).strip()
            if bb.upper() in BAND_WAVELENGTH_UM:
                b = bb.upper() if bb.upper() != "KS" else "K"
        out.setdefault(key, []).append(Measurement(
            band=b, wl_um=float(BAND_WAVELENGTH_UM.get(b, wl)) if "band" in colmap else wl,
            value_pct=float(val), err_pct=float(err) if math.isfinite(err) else 0.0,
            kind="upper" if limit else "meas", instrument=str(spec.get("instrument", "")),
            epoch=str(spec.get("epoch", "")), source=disc.name, verified=True,
            origin="archive", survey=disc.name))
        record["n_measurements"] += 1
    record["unmatched"] = record["unmatched"][:20]
    return out, record


def fetch_polarimetry_table(disc: DiscoveredTable, spec: dict, targets: TargetTable, *,
                            query_fn=None, log: AcquisitionLog | None = None,
                            max_rows: int = 5000) -> tuple[dict[str, dict], dict]:
    """Rows of a discovered polarimetry table as per-star records.

    The polarimetric null is context, NOT a term in the likelihood ratio, and
    the distinction is physical rather than a convenience.  Both families the
    statistic compares -- a grey body at ~1500 K and sub-micron grains at their
    sublimation temperature -- emit THERMALLY at H and K (1500 K peaks at
    ~1.9 um), and thermal emission from an optically thin, randomly oriented
    swarm is essentially unpolarised in both cases.  A polarisation limit
    therefore constrains the SCATTERED-light fraction, which is a different
    axis from the emissivity law, and folding it into chi^2 would let a
    constraint that does not discriminate between the families masquerade as
    evidence that does.  It is carried per star so it can be attached to any
    survivor, where a scattering constraint is genuinely informative.

    Unlike the sample, stars this table lists that the targets asset does not
    are NOT added: a polarimetric null on a star with no measured infrared
    excess says nothing this channel can use.
    """
    query_fn = query_fn or tap_query
    record = {"name": disc.name, "table": disc.table, "role": "polarimetry",
              "status": STATUS_FAILED, "n_rows": 0, "n_matched": 0, "n_values": 0}
    if disc.table is None or not disc.roles:
        record["status"] = disc.status
        return {}, record
    roles = dict(disc.roles)
    want = [r for r in ("hd", "hip", "name", "polarisation", "polarisation_err", "band",
                        "limit_flag", "flag") if r in roles]
    if "polarisation" not in want:
        record["status"] = STATUS_ZERO
        record["note"] = "no polarisation column"
        return {}, record
    sel = ", ".join(f'"{roles[r]}"' for r in dict.fromkeys(want))
    adql = f'SELECT TOP {int(max_rows)} {sel} FROM "{disc.table}"'
    try:
        df = query_fn(adql)
    except Exception as exc:                              # noqa: BLE001
        if log:
            log.record(f"fetch_{disc.name}", adql, error=repr(exc))
        record["error"] = repr(exc)[:300]
        return {}, record
    n = int(len(df)) if df is not None else 0
    if log:
        log.record(f"fetch_{disc.name}", adql, rows=n)
    record["n_rows"] = n
    record["status"] = STATUS_OK if n else STATUS_ZERO
    if not n:
        return {}, record
    colmap = {}
    for r in want:
        for c in df.columns:
            if str(c) == roles[r] or _canon(c) == _canon(roles[r]):
                colmap[r] = c
                break
    # ppm unless the table says otherwise; a fraction or percent is scaled up
    unit = str(spec.get("unit", "ppm")).lower()
    scale = {"fraction": 1e6, "percent": 1e4, "ppm": 1.0}.get(unit, 1.0)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = None
        for r, kind in (("hd", "hd"), ("hip", "hip"), ("name", "name")):
            if r in colmap:
                v = row[colmap[r]]
                if kind in ("hd", "hip"):
                    pref = kind.upper()
                    key = targets.key_for(f"{pref} {v}") \
                        if re.fullmatch(r"\s*\d+(\.0)?\s*", str(v)) else targets.key_for(v)
                else:
                    key = targets.key_for(v)
                if key:
                    break
        if key is None:
            continue
        record["n_matched"] += 1
        raw = row[colmap["polarisation"]]
        val = _to_float(raw) * scale
        if not math.isfinite(val):
            continue
        err = (_to_float(row[colmap["polarisation_err"]]) * scale
               if "polarisation_err" in colmap else float("nan"))
        limit = _is_limit(raw, row[colmap["limit_flag"]] if "limit_flag" in colmap else None)
        prev = out.get(key)
        rec = {"key": key, "p_ppm": float(val),
               "e_p_ppm": float(err) if math.isfinite(err) else None,
               "kind": "upper" if limit else "meas",
               "instrument": str(spec.get("instrument", "")),
               "epoch": str(spec.get("epoch", "")), "source": disc.name, "verified": True}
        # keep the TIGHTEST constraint when a star is listed more than once
        if prev is None or rec["p_ppm"] < prev["p_ppm"]:
            out[key] = rec
        record["n_values"] = len(out)
    return out, record


def verify_embedded(excess_asset: pd.DataFrame, archive: dict[str, list[Measurement]],
                    reached: dict[str, str], targets: TargetTable, *,
                    tol_abs_pct: float = 0.03, tol_rel: float = 0.05) -> tuple[list[dict], dict]:
    """Mark every embedded row ``verified`` / ``discrepant`` / ``not_in_archive``
    / ``table_not_reached`` / ``membership`` / ``no_archive_table``.

    ``reached`` maps a survey name to its fetch status.  A row is verified
    when the archive holds, for the same star, survey and band, a value
    within ``tol_abs_pct`` + ``tol_rel`` and an error within the same."""
    rows = []
    counts: dict[str, int] = {}
    for _, r in excess_asset.iterrows():
        rec = dict(r)
        survey = str(r.get("survey", ""))
        key = targets.key_for(r.get("key")) or str(r.get("key")).strip()
        val = _to_float(r.get("value_pct"))
        if not math.isfinite(val):
            status = "membership"
        elif not str(r.get("vizier", "")).strip():
            status = "no_archive_table"
        elif reached.get(survey) not in (STATUS_OK,):
            status = "table_not_reached"
        else:
            cands = [m for m in archive.get(key, []) if m.source == survey
                     and m.band.upper() == str(r.get("band", "")).upper()]
            if not cands:
                status = "not_in_archive"
            else:
                m = cands[0]
                err = _to_float(r.get("err_pct"))
                tol = tol_abs_pct + tol_rel * abs(m.value_pct)
                ok_v = abs(m.value_pct - val) <= tol
                ok_e = (not math.isfinite(err)) or abs(m.err_pct - err) <= tol
                status = "verified" if (ok_v and ok_e) else "discrepant"
                rec["archive_value_pct"] = m.value_pct
                rec["archive_err_pct"] = m.err_pct
        rec["verify"] = status
        counts[status] = counts.get(status, 0) + 1
        rows.append(rec)
    return rows, counts


# ---------------------------------------------------------------------------
# positions and spectral types (Simbad TAP)
# ---------------------------------------------------------------------------
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap"


def _simbad_tap(adql: str, url: str = SIMBAD_TAP) -> pd.DataFrame:
    import pyvo  # noqa: PLC0415  runner-only

    svc = pyvo.dal.TAPService(url)
    return svc.search(adql).to_table().to_pandas()


def resolve_positions(keys, *, tap_fn=None, url: str = SIMBAD_TAP, chunk: int = 60,
                      log: AcquisitionLog | None = None) -> pd.DataFrame:
    """``key, ra, dec, sptype, main_id, plx_mas`` from Simbad by identifier,
    in chunks.  Unresolved keys are absent from the frame (never invented)."""
    tap_fn = tap_fn or _simbad_tap
    keys = [str(k) for k in dict.fromkeys(keys) if str(k).strip()]
    frames = []
    for i in range(0, len(keys), int(chunk)):
        block = keys[i:i + int(chunk)]
        ids = ", ".join("'" + k.replace("'", "''") + "'" for k in block)
        adql = ("SELECT i.id AS asked, b.main_id, b.ra, b.dec, b.sp_type, b.plx_value "
                "FROM ident AS i JOIN basic AS b ON b.oid = i.oidref "
                f"WHERE i.id IN ({ids})")
        try:
            df = tap_fn(adql, url)
        except Exception as exc:                          # noqa: BLE001
            if log:
                log.record("simbad_positions", adql[:300], error=repr(exc))
            continue
        n = int(len(df)) if df is not None else 0
        if log:
            log.record("simbad_positions", adql[:300], rows=n)
        if n:
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            frames.append(pd.DataFrame({
                "key": df["asked"].astype(str), "main_id": df.get("main_id", ""),
                "ra": pd.to_numeric(df["ra"], errors="coerce"),
                "dec": pd.to_numeric(df["dec"], errors="coerce"),
                "sptype": df.get("sp_type", "").astype(str),
                "plx_mas": pd.to_numeric(df.get("plx_value"), errors="coerce")}))
    if not frames:
        return pd.DataFrame(columns=["key", "main_id", "ra", "dec", "sptype", "plx_mas"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("key")


# ---------------------------------------------------------------------------
# companions: WDS inside the field, Gaia DR3 where it has the star
# ---------------------------------------------------------------------------
def parse_wds(df: pd.DataFrame) -> list[dict]:
    """Every WDS pair in a cone as ``{sep_arcsec, dmag, comp}``."""
    if df is None or not len(df):
        return []
    cols = {_canon(c): c for c in df.columns}
    sep_col = next((cols[c] for c in ("sep2", "sep_2", "sep", "sep1", "sep_1") if c in cols), None)
    m1 = next((cols[c] for c in ("mag1", "mag_1") if c in cols), None)
    m2 = next((cols[c] for c in ("mag2", "mag_2") if c in cols), None)
    comp = next((cols[c] for c in ("comp", "components", "disc") if c in cols), None)
    out = []
    for _, r in df.iterrows():
        sep = _to_float(r[sep_col]) if sep_col else float("nan")
        a = _to_float(r[m1]) if m1 else float("nan")
        b = _to_float(r[m2]) if m2 else float("nan")
        out.append({"sep_arcsec": sep, "dmag": (b - a) if (math.isfinite(a) and math.isfinite(b))
                    else float("nan"), "comp": str(r[comp]) if comp else ""})
    return out


def companion_context(positions: pd.DataFrame, conf: dict, *, cone_fn=None,
                      log: AcquisitionLog | None = None,
                      embedded: pd.DataFrame | None = None) -> dict[str, dict]:
    """Per star: WDS pairs inside the field, Gaia RUWE / NSS / neighbours,
    and the verdict ``known_companion`` with its reason.  A star whose cones
    all failed is ``companion_assessed = False`` (the veto is unapplied)."""
    cone_fn = cone_fn or _vizier_cone
    out: dict[str, dict] = {}
    emb: dict[str, list[dict]] = {}
    if embedded is not None and len(embedded):
        for _, r in embedded.iterrows():
            emb.setdefault(str(r.get("key")), []).append(dict(r))
    wds_t = conf.get("wds_table", "B/wds/wds")
    r_wds = float(conf.get("wds_radius_arcsec", 3.0))
    sep_max = float(conf.get("wds_sep_max_arcsec", 1.5))
    dmag_max = float(conf.get("wds_dmag_max", 6.0))
    pos = positions.set_index("key") if len(positions) else pd.DataFrame()
    gaia = {}
    if len(positions):
        gpos = positions.rename(columns={"key": "star_id"})[["star_id", "ra", "dec"]]
        gaia = gaia_context(gpos, cone_fn=cone_fn, log=log,
                            gaia_table=conf.get("gaia_table", "I/355/gaiadr3"),
                            vari_table="", radius_arcsec=float(conf.get("gaia_radius_arcsec", 12.0)),
                            match_arcsec=float(conf.get("gaia_match_arcsec", 3.0)))
    keys = list(dict.fromkeys(list(pos.index) + list(emb)))
    for key in keys:
        ctx = {"known_companion": False, "companion_note": "", "companion_assessed": False,
               "wds": [], "wds_reached": False, "gaia": gaia.get(key, {}),
               "embedded": emb.get(key, [])}
        if key in pos.index:
            ra, dec = float(pos.loc[key, "ra"]), float(pos.loc[key, "dec"])
            if math.isfinite(ra) and math.isfinite(dec):
                try:
                    dw = cone_fn(wds_t, ra, dec, r_wds)
                    ctx["wds"] = parse_wds(dw)
                    ctx["wds_reached"] = True
                except Exception as exc:                  # noqa: BLE001
                    ctx["wds_error"] = repr(exc)[:200]
        close = [w for w in ctx["wds"] if math.isfinite(w["sep_arcsec"]) and w["sep_arcsec"] <= sep_max
                 and (not math.isfinite(w["dmag"]) or w["dmag"] <= dmag_max)]
        if close:
            ctx["known_companion"] = True
            ctx["companion_note"] = "WDS pair at {:.2f} arcsec, dmag {:.1f}".format(
                close[0]["sep_arcsec"], close[0]["dmag"])
        for e in ctx["embedded"]:
            sep = _to_float(e.get("sep_arcsec"))
            if not math.isfinite(sep) or sep <= sep_max:
                ctx["known_companion"] = True
                ctx["companion_note"] = ctx["companion_note"] or str(e.get("note") or e.get("source"))
        g = ctx["gaia"]
        if g.get("gaia_reached") and g.get("target_found"):
            ruwe = g.get("ruwe", float("nan"))
            nss = g.get("nss", float("nan"))
            if isinstance(nss, (int, float)) and math.isfinite(nss) and nss > 0:
                ctx["gaia_companion_suspect"] = f"NSS flag {int(nss)}"
            elif isinstance(ruwe, (int, float)) and math.isfinite(ruwe) and \
                    ruwe > float(conf.get("ruwe_max", 1.4)):
                ctx["gaia_companion_suspect"] = f"RUWE {ruwe:.2f}"
        ctx["companion_assessed"] = bool(ctx["wds_reached"] or ctx["embedded"])
        out[key] = ctx
    if log:
        n_ok = sum(1 for c in out.values() if c["wds_reached"])
        log.record("wds_cone", f"{len(out)} cones on {wds_t} r={r_wds}\"", rows=n_ok or None,
                   error=None if n_ok or not out else "every WDS cone failed")
    return out


# ---------------------------------------------------------------------------
# broadband leg: 2MASS + AllWISE for the sample (cones), and the population
# ---------------------------------------------------------------------------
_TMASS_ROLES = {"ks": [r"^kmag$", r"^ks_?m(ag)?$", r"^k_?mag$", r"^ksmag$"],
                "e_ks": [r"^e_kmag$", r"^ks_?msigcom$", r"^e_ks_?mag$", r"^e_ksmag$"],
                "h": [r"^hmag$", r"^h_?m(ag)?$"], "j": [r"^jmag$", r"^j_?m(ag)?$"],
                "qual": [r"^qflg$", r"^ph_?qual$", r"^q_?flg$"],
                "rflg": [r"^rflg$", r"^rd_?flg$"],
                "ra": ROLE_PATTERNS["ra"], "dec": ROLE_PATTERNS["dec"]}
_WISE_ROLES = {"w1": [r"^w1mag$", r"^w1mpro$", r"^w1$"], "e_w1": [r"^e_w1mag$", r"^w1sigmpro$", r"^w1mpro_error$"],
               "w2": [r"^w2mag$", r"^w2mpro$", r"^w2$"], "e_w2": [r"^e_w2mag$", r"^w2sigmpro$", r"^w2mpro_error$"],
               "w3": [r"^w3mag$", r"^w3mpro$", r"^w3$"], "e_w3": [r"^e_w3mag$", r"^w3sigmpro$", r"^w3mpro_error$"],
               "w4": [r"^w4mag$", r"^w4mpro$", r"^w4$"], "e_w4": [r"^e_w4mag$", r"^w4sigmpro$", r"^w4mpro_error$"],
               "qual": [r"^qph$", r"^ph_?qual$"], "cc": [r"^ccf$", r"^cc_?flags$"],
               "ext": [r"^ex$", r"^ext_?flag$"], "ra": ROLE_PATTERNS["ra"], "dec": ROLE_PATTERNS["dec"]}


def _nearest(df: pd.DataFrame, ra: float, dec: float, roles: dict) -> pd.Series | None:
    if df is None or not len(df) or "ra" not in roles or "dec" not in roles:
        return None
    r = pd.to_numeric(df[roles["ra"]], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(df[roles["dec"]], errors="coerce").to_numpy(dtype=float)
    sep = np.hypot((r - ra) * math.cos(math.radians(dec)), d - dec) * 3600.0
    i = int(np.nanargmin(sep)) if np.isfinite(sep).any() else None
    if i is None:
        return None
    row = df.iloc[i].copy()
    row["_sep_arcsec"] = float(sep[i])
    return row


def sample_photometry(positions: pd.DataFrame, conf: dict, *, cone_fn=None,
                      log: AcquisitionLog | None = None) -> pd.DataFrame:
    """2MASS and AllWISE photometry of every positioned sample star (nearest
    source in a small cone), with the quality flags the broadband leg needs."""
    cone_fn = cone_fn or _vizier_cone
    r = float(conf.get("cone_arcsec", 5.0))
    tm_t, wi_t = conf.get("sample_tmass_table", "II/246/out"), conf.get("sample_allwise_table", "II/328/allwise")
    rows = []
    n_tm = n_wi = 0
    for _, p in positions.iterrows():
        key, ra, dec = str(p["key"]), float(p["ra"]), float(p["dec"])
        rec = {"key": key, "ra": ra, "dec": dec, "tmass_reached": False, "wise_reached": False}
        if not (math.isfinite(ra) and math.isfinite(dec)):
            rows.append(rec)
            continue
        try:
            dt = cone_fn(tm_t, ra, dec, r)
            rec["tmass_reached"] = True
            n_tm += 1
            roles = resolve_columns(dt.columns if dt is not None else [], _TMASS_ROLES)
            near = _nearest(dt, ra, dec, roles)
            if near is not None:
                for k in ("ks", "e_ks", "h", "j"):
                    if k in roles:
                        rec[k] = _to_float(near[roles[k]])
                for k in ("qual", "rflg"):
                    if k in roles:
                        rec[f"tmass_{k}"] = str(near[roles[k]])
                rec["tmass_sep_arcsec"] = float(near["_sep_arcsec"])
        except Exception as exc:                          # noqa: BLE001
            rec["tmass_error"] = repr(exc)[:160]
        try:
            dw = cone_fn(wi_t, ra, dec, r)
            rec["wise_reached"] = True
            n_wi += 1
            roles = resolve_columns(dw.columns if dw is not None else [], _WISE_ROLES)
            near = _nearest(dw, ra, dec, roles)
            if near is not None:
                for k in ("w1", "e_w1", "w2", "e_w2", "w3", "e_w3", "w4", "e_w4"):
                    if k in roles:
                        rec[k] = _to_float(near[roles[k]])
                for k in ("qual", "cc", "ext"):
                    if k in roles:
                        rec[f"wise_{k}"] = str(near[roles[k]])
                rec["wise_sep_arcsec"] = float(near["_sep_arcsec"])
        except Exception as exc:                          # noqa: BLE001
            rec["wise_error"] = repr(exc)[:160]
        rows.append(rec)
    if log:
        log.record("sample_2mass", f"{len(positions)} cones on {tm_t}", rows=n_tm or None,
                   error=None if n_tm or not len(positions) else "every 2MASS cone failed")
        log.record("sample_allwise", f"{len(positions)} cones on {wi_t}", rows=n_wi or None,
                   error=None if n_wi or not len(positions) else "every AllWISE cone failed")
    return pd.DataFrame(rows)


# --- the < 30 pc population through the ESA Gaia archive -------------------
_WISE_WANT = {"designation": ["designation"], "w1": ["w1mpro"], "e_w1": ["w1mpro_error", "w1sigmpro"],
              "w2": ["w2mpro"], "e_w2": ["w2mpro_error", "w2sigmpro"], "w3": ["w3mpro"],
              "e_w3": ["w3mpro_error", "w3sigmpro"], "w4": ["w4mpro"], "e_w4": ["w4mpro_error", "w4sigmpro"],
              "wise_qual": ["ph_qual"], "wise_cc": ["cc_flags"], "wise_ext": ["ext_flag"]}
_TMASS_WANT = {"designation": ["designation"], "j": ["j_m"], "h": ["h_m"], "ks": ["ks_m"],
               "e_ks": ["ks_msigcom"], "tmass_qual": ["ph_qual"]}


def _gaia_run(query: str, retries: int = 3) -> pd.DataFrame:
    from astroquery.gaia import Gaia  # noqa: PLC0415  runner-only

    last = None
    for attempt in range(retries):
        try:
            job = Gaia.launch_job(query) if attempt == retries - 1 else Gaia.launch_job_async(query)
            df = job.get_results().to_pandas()
            return df.rename(columns={c: str(c).lower() for c in df.columns})
        except Exception as exc:                          # noqa: BLE001
            last = exc
            print(f"[forge/acquire] gaia attempt {attempt + 1}/{retries} failed: {exc!r}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gaia query failed after {retries} attempts: {last!r}")


def probe_columns(table: str, want: dict, run_query) -> dict:
    """``{logical: real}`` for every wanted column a TOP 1 probe shows."""
    try:
        probe = run_query(f"SELECT TOP 1 * FROM {table}")
    except Exception as exc:                              # noqa: BLE001
        print(f"[forge/acquire] column probe failed for {table}: {exc!r}")
        return {}
    have = {str(c).lower(): str(c) for c in probe.columns}
    out = {}
    for logical, cands in want.items():
        for c in cands:
            if c.lower() in have:
                out[logical] = have[c.lower()]
                break
    return out


def population_query(dec_lo: float, dec_hi: float, wise_cols: dict, tmass_cols: dict, *,
                     plx_min: float, poe_min: float, ruwe_max: float, g_max: float,
                     top: int, tmass_bridge: bool = True) -> str:
    wsel = ", ".join(f"w.{a} AS {k}" for k, a in wise_cols.items() if k != "designation")
    tsel = ", ".join(f"t.{a} AS {k}" for k, a in tmass_cols.items() if k != "designation")
    if tmass_bridge:
        tjoin = ("JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS bt ON g.source_id = bt.source_id\n"
                 "JOIN gaiadr3.tmass_psc_xsc_join AS tj ON bt.clean_tmass_psc_xsc_oid = tj.clean_tmass_psc_xsc_oid\n"
                 "JOIN gaiadr1.tmass_original_valid AS t ON tj.original_psc_source_id = t.designation\n")
    else:
        tjoin = ("JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS bt ON g.source_id = bt.source_id\n"
                 "JOIN gaiadr1.tmass_original_valid AS t ON bt.original_ext_source_id = t.designation\n")
    return (f"SELECT TOP {int(top)} g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error, "
            "g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.non_single_star, g.teff_gspphot, "
            f"{tsel}, {wsel}\n"
            "FROM gaiadr3.gaia_source AS g\n" + tjoin +
            "JOIN gaiadr3.allwise_best_neighbour AS bw ON g.source_id = bw.source_id\n"
            "JOIN gaiadr1.allwise_original_valid AS w ON bw.original_ext_source_id = w.designation\n"
            f"WHERE g.parallax > {plx_min} AND g.parallax_over_error > {poe_min} "
            f"AND g.ruwe < {ruwe_max} AND g.phot_g_mean_mag < {g_max} "
            f"AND g.dec >= {dec_lo} AND g.dec < {dec_hi}")


def population_photometry(conf: dict, out_dir: Path, *, run_query=None,
                          log: AcquisitionLog | None = None) -> tuple[pd.DataFrame, dict]:
    """The < 30 pc population with 2MASS and AllWISE, in declination bands,
    checkpointed to ``out_dir``.  Both 2MASS join chains are tried (the
    bridge first; a chain that matches nothing returns zero rows, not an
    error, so zero rows on the first chain means try the second)."""
    run_query = run_query or _gaia_run
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pc = conf.get("population", {})
    band = float(pc.get("dec_band_deg", 30))
    edges = np.arange(-90.0, 90.0 + 1e-9, band)
    wise_cols = probe_columns("gaiadr1.allwise_original_valid", _WISE_WANT, run_query)
    tmass_cols = probe_columns("gaiadr1.tmass_original_valid", _TMASS_WANT, run_query)
    record = {"wise_columns": wise_cols, "tmass_columns": tmass_cols, "bands": [], "chain": None}
    if not wise_cols.get("designation") or not wise_cols.get("w1") or not tmass_cols.get("ks"):
        record["status"] = STATUS_FAILED
        record["error"] = "AllWISE / 2MASS mirror columns not resolved"
        if log:
            log.record("population_probe", "TOP 1 probes of the Gaia mirrors", error=record["error"])
        return pd.DataFrame(), record
    frames = []
    chain = record["chain"]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        part = out_dir / f"population_dec_{int(lo):+04d}_{int(hi):+04d}.parquet"
        if part.exists():
            df = pd.read_parquet(part)
            frames.append(df)
            record["bands"].append({"dec": [lo, hi], "n_rows": int(len(df)), "from": "checkpoint"})
            continue
        got = None
        for bridge in ([True, False] if chain is None else [chain == "bridge"]):
            q = population_query(lo, hi, wise_cols, tmass_cols,
                                 plx_min=float(pc.get("parallax_min_mas", 33.333)),
                                 poe_min=float(pc.get("parallax_over_error_min", 10.0)),
                                 ruwe_max=float(pc.get("ruwe_max", 1.4)),
                                 g_max=float(pc.get("g_max", 15.0)),
                                 top=int(pc.get("top_per_band", 200000)), tmass_bridge=bridge)
            try:
                df = run_query(q)
            except Exception as exc:                      # noqa: BLE001
                if log:
                    log.record("population_band", q[:400], error=repr(exc),
                               extra={"dec": [lo, hi], "chain": "bridge" if bridge else "direct"})
                continue
            n = int(len(df)) if df is not None else 0
            if log:
                log.record("population_band", q[:400], rows=n,
                           extra={"dec": [lo, hi], "chain": "bridge" if bridge else "direct"})
            if n:
                got = df
                chain = record["chain"] = "bridge" if bridge else "direct"
                break
        if got is not None:
            got.to_parquet(part, index=False)
            frames.append(got)
            record["bands"].append({"dec": [lo, hi], "n_rows": int(len(got)), "from": chain})
        else:
            record["bands"].append({"dec": [lo, hi], "n_rows": 0, "from": "none"})
    if not frames:
        record["status"] = STATUS_ZERO
        return pd.DataFrame(), record
    out = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
    record["status"] = STATUS_OK
    record["n_rows"] = int(len(out))
    return out, record


__all__ = ["DiscoveredTable", "ROLE_PATTERNS", "SIMBAD_TAP", "STATUS_FAILED",
           "STATUS_NOT_ATTEMPTED", "STATUS_OK", "STATUS_ZERO", "AcquisitionLog",
           "companion_context", "discover_table", "fetch_excess_table",
           "fetch_polarimetry_table", "parse_wds",
           "population_photometry", "population_query", "probe_columns", "resolve_excess_columns",
           "resolve_positions", "sample_photometry", "score_excess_table", "tap_query",
           "verify_embedded"]

"""JPL Small-Body Database acquisition for DERELICT.  Runner-only.

The sandbox has no egress (``CONNECT tunnel failed, response 403`` for
``ssd-api.jpl.nasa.gov``), so nothing here is exercised locally -- the offline
tests drive it through injected fake transports.  On the runner it must be
*defensive*, because the exact constraint syntax of the SBDB Query API is the
least-certain part of the design.  Every function therefore:

* records what it actually did in a provenance dict (which strategy worked,
  which fields the server accepted, the row count);
* falls back to a coarser strategy rather than failing;
* returns an explicit ``NO_DATA_REACHED``-style status instead of an empty
  frame pretending to be a clean null.

Endpoints
---------
``sbdb_query.api``
    Bulk table query.  ``fields=`` selects columns, ``sb-kind=a|c`` selects
    asteroids/comets, ``sb-cdata=`` carries a JSON constraint tree, and
    ``full-prec=1`` returns full precision.  Response is
    ``{"signature":…, "count":N, "fields":[…], "data":[[…],…]}``.
``sbdb.api``
    Per-object detail.  ``sstr=<id>&cov=mat&full-prec=1&phys-par=1`` returns the
    fitted ``orbit.model_pars`` (which reveals whether the non-grav law was the
    inverse-square one) and the covariance matrix (needed to ask whether A2 and
    A3 are *jointly* consistent with zero, not just marginally).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

SBDB_QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
SBDB_OBJECT_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
SBDB_QUERY_DOC_URL = "https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html"

USER_AGENT = "Seti-derelict/1.0 (mailto:trimcrae@gmail.com)"

#: Columns we want.  Anything the server rejects is dropped and recorded rather
#: than aborting the run -- the field list is the second-least-certain part.
DEFAULT_FIELDS: tuple[str, ...] = (
    "spkid", "full_name", "pdes", "name", "kind", "class", "neo", "pha",
    "H", "diameter", "albedo", "rot_per",
    "e", "a", "q", "i", "om", "w", "ma", "epoch", "moid",
    "n_obs_used", "data_arc", "condition_code", "rms", "first_obs", "last_obs",
    "A1", "A2", "A3", "DT",
    "sigma_A1", "sigma_A2", "sigma_A3", "sigma_DT",
)

#: Fields without which the channel cannot function at all.
REQUIRED_FIELDS: frozenset[str] = frozenset({"full_name", "A1"})


# --- transport ----------------------------------------------------------------
def _default_transport(url: str, timeout: float = 600.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


Transport = Callable[[str], bytes]


@dataclass
class FetchResult:
    """A bulk-query outcome, degradation included as a first-class field."""
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = "NO_DATA_REACHED"
    strategy: str = ""
    fields_requested: tuple[str, ...] = ()
    fields_returned: tuple[str, ...] = ()
    fields_dropped: tuple[str, ...] = ()
    n_rows: int = 0
    signature: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict:
        return {"status": self.status, "strategy": self.strategy,
                "n_rows": self.n_rows,
                "fields_requested": list(self.fields_requested),
                "fields_returned": list(self.fields_returned),
                "fields_dropped": list(self.fields_dropped),
                "signature": self.signature, "errors": self.errors}


#: Characters the SBDB constraint syntax needs to survive URL encoding.
_URL_SAFE = '{}[]|"'


def _build_url(base: str, params: dict[str, Any]) -> str:
    clean = {k: v for k, v in params.items() if v is not None}
    query = urllib.parse.urlencode(clean, safe=_URL_SAFE)
    return f"{base}?{query}"


def _parse_query_payload(raw: bytes) -> tuple[pd.DataFrame, dict]:
    """Turn an ``sbdb_query.api`` JSON body into a DataFrame + signature."""
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    if "message" in payload and "data" not in payload:
        raise ValueError(f"SBDB error: {payload.get('message')}")
    cols = payload.get("fields") or []
    rows = payload.get("data") or []
    df = pd.DataFrame(rows, columns=cols)
    # SBDB returns everything as strings; coerce the numeric columns.
    for c in df.columns:
        if c in {"full_name", "pdes", "name", "kind", "class", "first_obs",
                 "last_obs", "spkid", "prefix"}:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, payload.get("signature", {})


def _attempt(transport: Transport, url: str, tries: int = 3,
             pause: float = 5.0) -> tuple[pd.DataFrame | None, dict, str | None]:
    last = None
    for i in range(tries):
        try:
            df, sig = _parse_query_payload(transport(url))
            return df, sig, None
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            last = f"HTTP {exc.code} for {url}: {body}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc} for {url}"
        if i + 1 < tries:
            time.sleep(pause * (i + 1))
    return None, {}, last


# --- schema discovery ---------------------------------------------------------
def discover_fields(transport: Transport | None = None) -> tuple[set[str] | None, list[str]]:
    """Best-effort discovery of the field names the server actually accepts.

    Tries the documented ``info=field`` probe, then a one-row probe query.
    Returns ``(field_names_or_None, notes)``.  ``None`` means discovery failed
    and the caller should just try its field list and prune on error -- which is
    why field pruning exists.
    """
    tr = transport or _default_transport
    notes: list[str] = []
    for params in ({"fields": "full_name", "limit": 1, "info": "field"},
                   {"fields": "full_name", "limit": 1}):
        url = _build_url(SBDB_QUERY_URL, params)
        try:
            payload = json.loads(tr(url).decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"probe {params} failed: {type(exc).__name__}: {exc}")
            continue
        for key in ("field", "fields", "available_fields"):
            val = payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                names = {str(d.get("name")) for d in val if d.get("name")}
                if names:
                    notes.append(f"discovered {len(names)} fields via {key}")
                    return names, notes
        notes.append(f"probe {params} returned keys {sorted(payload)[:12]}")
    return None, notes


def _prune_fields(fields: tuple[str, ...], available: set[str] | None
                  ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not available:
        return fields, ()
    keep = tuple(f for f in fields if f in available)
    drop = tuple(f for f in fields if f not in available)
    missing_required = REQUIRED_FIELDS - set(keep)
    if missing_required:
        # Discovery is more likely wrong than the API; keep everything and let
        # the server arbitrate.
        return fields, ()
    return keep, drop


# --- the bulk pull ------------------------------------------------------------
#: Constraint syntaxes to try, in decreasing confidence.  ``None`` = no
#: constraint at all (pull everything and filter client-side).
CDATA_STRATEGIES: tuple[tuple[str, str | None], ...] = (
    ("cdata_A1_defined", '{"AND":["A1|DF"]}'),
    ("cdata_A1_notnull", '{"AND":["A1|NE|0"]}'),
    ("cdata_A1_defined_bare", '{"AND":["A1|DF"]}'.replace('"', "")),
    ("unconstrained_full_pull", None),
)


def fetch_nongrav_table(kind: str = "a",
                        fields: tuple[str, ...] = DEFAULT_FIELDS,
                        transport: Transport | None = None,
                        limit: int | None = None,
                        strategies: tuple[tuple[str, str | None], ...] | None = None,
                        available_fields: set[str] | None = None,
                        tries: int = 3) -> FetchResult:
    """Pull the small-body table, preferring a server-side ``A1`` constraint.

    ``kind='a'`` is the science sample (asteroids); ``kind='c'`` is the comet
    **control** sample -- objects whose radial acceleration is known to be
    outgassing, which is exactly the contamination the channel must not
    mistake for radiation pressure.

    Walks :data:`CDATA_STRATEGIES` until one returns rows.  The final strategy
    pulls the whole catalogue unconstrained (~1.4M asteroid rows) and lets the
    caller filter client-side, so an unexpected constraint syntax costs
    bandwidth, not the run.
    """
    tr = transport or _default_transport
    strategies = strategies or CDATA_STRATEGIES
    kept, dropped = _prune_fields(fields, available_fields)
    res = FetchResult(fields_requested=fields, fields_dropped=dropped)

    for name, cdata in strategies:
        params = {"fields": ",".join(kept), "sb-kind": kind, "full-prec": "1"}
        if cdata:
            params["sb-cdata"] = cdata
        if limit:
            params["limit"] = limit
        df, sig, err = _attempt(tr, _build_url(SBDB_QUERY_URL, params), tries=tries)
        if df is None:
            res.errors.append(f"[{name}] {err}")
            continue
        if df.empty:
            res.errors.append(f"[{name}] returned 0 rows")
            continue
        # An unconstrained pull still has to be reduced to the A1 population.
        if cdata is None and "A1" in df.columns:
            before = len(df)
            df = df[df["A1"].notna()].copy()
            res.errors.append(
                f"[{name}] client-side A1 filter: {before} -> {len(df)} rows")
        res.table = df.reset_index(drop=True)
        res.status = "OK" if len(res.table) else "NO_ROWS_WITH_A1"
        res.strategy = name
        res.fields_returned = tuple(df.columns)
        res.n_rows = len(res.table)
        res.signature = sig
        return res

    res.status = "NO_DATA_REACHED"
    return res


# --- per-object detail --------------------------------------------------------
def fetch_object_detail(designation: str, transport: Transport | None = None,
                        tries: int = 3, pause: float = 2.0) -> dict:
    """``sbdb.api`` record for one object: model_pars, covariance, phys pars.

    ``orbit.model_pars`` is what tells us whether the fitted non-grav law was
    the inverse-square one (so ``A1`` really is a radiation-pressure
    coefficient) or a cometary ``g(r)`` (so it is not).  The covariance lets us
    ask whether A2 and A3 are *jointly* zero rather than marginally.
    Returns ``{"ok": False, "error": …}`` rather than raising.
    """
    tr = transport or _default_transport
    url = _build_url(SBDB_OBJECT_URL, {
        "sstr": designation, "cov": "mat", "full-prec": "1",
        "phys-par": "1", "discovery": "1",
    })
    last = None
    for i in range(tries):
        try:
            payload = json.loads(tr(url).decode("utf-8", errors="replace"))
            payload["ok"] = "object" in payload or "orbit" in payload
            if not payload["ok"]:
                payload["error"] = payload.get("message", "no object/orbit key")
            return payload
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            if i + 1 < tries:
                time.sleep(pause * (i + 1))
    return {"ok": False, "error": last, "url": url}


def model_par_names(detail: dict) -> list[str]:
    """Extract the fitted non-grav parameter names from an ``sbdb.api`` record."""
    orbit = (detail or {}).get("orbit") or {}
    pars = orbit.get("model_pars") or []
    out = []
    for p in pars:
        if isinstance(p, dict) and p.get("name"):
            out.append(str(p["name"]))
        elif isinstance(p, (list, tuple)) and p:
            out.append(str(p[0]))
    return out


def model_par_values(detail: dict) -> dict[str, dict]:
    """``{name: {"value": float, "sigma": float, "units": str}}`` from a record."""
    out: dict[str, dict] = {}
    for p in (detail or {}).get("orbit", {}).get("model_pars") or []:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        try:
            value = float(p.get("value"))
        except (TypeError, ValueError):
            value = float("nan")
        try:
            sigma = float(p.get("sigma"))
        except (TypeError, ValueError):
            sigma = float("nan")
        out[str(p["name"])] = {"value": value, "sigma": sigma,
                               "units": p.get("units"), "title": p.get("title")}
    return out


def object_is_comet(detail: dict) -> bool:
    """True when ``sbdb.api`` classifies the object as a comet."""
    obj = (detail or {}).get("object") or {}
    if obj.get("kind") and str(obj["kind"]).lower().startswith("c"):
        return True
    return bool(obj.get("orbit_class", {}).get("code", "") in {"COM", "CTc", "HTC", "JFc",
                                                              "JFC", "ETc", "PAR", "HYP"})

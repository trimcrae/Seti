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

# HTTPS, as every other channel in this repository uses.  The plain-http
# endpoint answered 503 to all six of ARC's discovery queries on its first
# dispatch (run 34787802564, 2026-09-13) while the necrofrontier probe reached
# the https endpoint repeatedly the same day.
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
# Tried in order when the primary answers 503 / refuses.  On 2026-09-13 BOTH
# the https and the http spelling of tapvizier.cds.unistra.fr answered 503 to
# five attempts each (results/uline/summary.json, results/arc/summary.json), so
# a second HOST and a route that is not TAP at all are now part of the ladder.
#
# verify: ``tapvizier.u-strasbg.fr`` is the historical CDS hostname for the same
# service (u-strasbg.fr is the pre-2021 name of the domain that became
# cds.unistra.fr, and CDS kept the old names resolving); ``vizier.cfa.harvard.edu``
# is the CfA VizieR mirror.  Neither has been reached from this sandbox --- there
# is no egress here --- so both are ASSERTED, not confirmed, and the first runner
# dispatch that reaches (or fails to reach) them is what settles it.  A host that
# fails is recorded with its error text, never dropped silently.
VIZIER_TAP_MIRRORS = (
    "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
    "https://tapvizier.u-strasbg.fr/TAPVizieR/tap",      # verify: unconfirmed mirror
    "https://vizier.cfa.harvard.edu/TAPVizieR/tap",      # verify: unconfirmed mirror
    "http://tapvizier.cds.unistra.fr/TAPVizieR/tap",
)

#: VizieR's ASU interface --- the NON-TAP route.  ``asu-tsv`` is served by the
#: main VizieR web application, not by the TAP service, so it survives a TAP
#: outage; it is the route most likely to answer while TAPVizieR is down.
VIZIER_ASU = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
VIZIER_ASU_MIRRORS = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "https://vizier.u-strasbg.fr/viz-bin/asu-tsv",       # verify: unconfirmed mirror
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv",    # verify: unconfirmed mirror
)
#: The catalogue's ReadMe --- the last table-existence check when even the ASU
#: metadata form does not answer.
VIZIER_README = "https://cdsarc.cds.unistra.fr/ftp/{catalogue}/ReadMe"

ROUTE_TAP = "tap"
ROUTE_ASU = "asu_tsv"
ROUTE_README = "readme"
ROUTE_ASTROQUERY = "astroquery"
ROUTE_NONE = "none"

STATUS_OK = "OK"
STATUS_FAILED = "QUERY_FAILED"
STATUS_ZERO = "QUERY_RETURNED_ZERO_ROWS"

#: Every route attempt made in this process, newest last: ``{route, endpoint,
#: status, rows, error, what}``.  A run that was served by a fallback can say
#: so even when the caller only kept the DataFrame.
ROUTE_LOG: list[dict] = []
_ROUTE_LOG_MAX = 500
#: Circuit breaker, per endpoint, for the DEFAULT transports only (an injected
#: ``fetch_fn`` never touches it):
#: ``{endpoint: {n, at, state, ever_ok, grace_used}}``.  After
#: ``_ROUTE_FAIL_LIMIT`` consecutive failures an endpoint is skipped for
#: ``_ROUTE_FAIL_COOLDOWN_S`` with a recorded reason, then tried again.  When
#: TAPVizieR is down for an hour, forty more four-minute retry ladders say
#: nothing the first two said --- and they cost the run the time the working
#: route needed.
#:
#: ARC's dispatch of 2026-09-14 (run 34792280736) showed the other failure
#: mode: TAPVizieR was INTERMITTENT, the first discovery query succeeded, two
#: later ones 503'd, and a five-minute cooldown then turned a flaky service
#: into a total outage --- every remaining catalogue read ``skipped: ... failed
#: 2 times in this process, 82s ago``.  A breaker that does that is worse than
#: no breaker, so:
#:
#: * the cooldown is short (:data:`_ROUTE_FAIL_COOLDOWN_S`, seconds, not
#:   minutes) and expiring it HALF-OPENS the endpoint --- one live attempt,
#:   cheap (one try, no retry ladder), and any success closes the breaker;
#: * an endpoint that has ALREADY SERVED a query in this process is never
#:   skipped at all: it is intermittent, not down, and the only way to learn
#:   that it is back is to ask.  It is degraded instead --- one cheap attempt
#:   per query instead of the retry ladder --- so every later catalogue still
#:   gets a genuine attempt at the cost of one fast 503;
#: * every transition is recorded in :data:`BREAKER_LOG`, which rides along in
#:   ``AcquisitionLog.as_dict()["breaker"]`` so the behaviour is visible in the
#:   artefact instead of having to be inferred from skip messages.
_ROUTE_FAILS: dict[str, dict] = {}
_ROUTE_FAIL_LIMIT = 2
_ROUTE_FAIL_COOLDOWN_S = 45.0

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"

#: Every breaker transition in this process, newest last.
BREAKER_LOG: list[dict] = []
_BREAKER_LOG_MAX = 200


def reset_route_state() -> None:
    """Forget the route log, the breaker log and every endpoint's state."""
    ROUTE_LOG.clear()
    _ROUTE_FAILS.clear()
    BREAKER_LOG.clear()


def _breaker_note(endpoint: str, transition: str, reason: str = "") -> dict:
    rec = {"endpoint": str(endpoint), "transition": str(transition),
           "reason": str(reason)[:300], "at_unix": float(_time.time())}
    BREAKER_LOG.append(rec)
    del BREAKER_LOG[:-_BREAKER_LOG_MAX]
    print(f"[metronome/acquire] breaker {endpoint}: {transition}"
          + (f" ({reason})" if reason else ""))
    return rec


def _breaker_block(endpoint: str) -> dict:
    return _ROUTE_FAILS.setdefault(str(endpoint), {"n": 0, "at": 0.0, "state": STATE_CLOSED,
                                                   "ever_ok": False, "pass_noted": False})


def breaker_state(endpoint: str) -> str:
    """``closed`` | ``half_open`` | ``open`` for one endpoint, right now."""
    blk = _ROUTE_FAILS.get(str(endpoint))
    if not blk:
        return STATE_CLOSED
    return str(blk.get("state", STATE_CLOSED))


def breaker_summary() -> dict:
    """Breaker state per endpoint plus every transition, for the artefact."""
    return {"fail_limit": int(_ROUTE_FAIL_LIMIT),
            "cooldown_s": float(_ROUTE_FAIL_COOLDOWN_S),
            "endpoints": {ep: {"state": blk.get("state", STATE_CLOSED),
                               "consecutive_failures": int(blk.get("n", 0)),
                               "ever_ok": bool(blk.get("ever_ok", False))}
                          for ep, blk in _ROUTE_FAILS.items()},
            "transitions": list(BREAKER_LOG),
            "n_transitions": len(BREAKER_LOG)}


def _circuit_open(endpoint: str) -> str:
    """``""`` when the endpoint may be tried, else why it is being skipped.

    Not a pure predicate: letting an endpoint through after its cooldown, or
    because it has already served this process, IS the half-open transition,
    and it is recorded here so the decision is auditable.
    """
    ep = str(endpoint)
    blk = _ROUTE_FAILS.get(ep)
    if not blk or blk["n"] < _ROUTE_FAIL_LIMIT:
        return ""
    waited = _time.time() - blk["at"]
    if waited >= _ROUTE_FAIL_COOLDOWN_S:
        # Half-open: one live attempt.  Dropping the count below the limit is
        # what makes a SUCCESS close the breaker and a fresh failure reopen it.
        blk["n"] = _ROUTE_FAIL_LIMIT - 1
        blk["state"] = STATE_HALF_OPEN
        _breaker_note(ep, f"{STATE_OPEN}->{STATE_HALF_OPEN}",
                      f"cooldown of {_ROUTE_FAIL_COOLDOWN_S:.0f}s elapsed ({waited:.0f}s)")
        return ""
    if blk.get("ever_ok"):
        # It answered earlier in this process, so it is INTERMITTENT, not down:
        # never skipped, just degraded to one cheap attempt per query.  This is
        # the rule ARC run 34792280736 broke -- there the primary served the
        # first discovery query and every later catalogue was skipped outright.
        if not blk.get("pass_noted"):
            blk["pass_noted"] = True
            _breaker_note(ep, f"{STATE_OPEN}->{STATE_HALF_OPEN}",
                          "endpoint already served a query in this process: never skipped, "
                          "one cheap attempt per query until it answers again")
        blk["state"] = STATE_HALF_OPEN
        return ""
    return (f"skipped: {endpoint} failed {blk['n']} times in this process, "
            f"{waited:.0f}s ago (retried after {_ROUTE_FAIL_COOLDOWN_S:.0f}s)")


def _circuit_fail(endpoint: str, error: str = "") -> None:
    blk = _breaker_block(endpoint)
    blk["n"] += 1
    blk["at"] = _time.time()
    if blk["n"] >= _ROUTE_FAIL_LIMIT:
        was = blk["state"]
        blk["state"] = STATE_OPEN
        if was == STATE_CLOSED:      # the open event, once; not every failed probe
            _breaker_note(endpoint, f"{STATE_CLOSED}->{STATE_OPEN}",
                          f"{blk['n']} consecutive failures: {error}"[:300])


def _circuit_ok(endpoint: str) -> None:
    blk = _breaker_block(endpoint)
    if blk["state"] != STATE_CLOSED or blk["n"]:
        _breaker_note(endpoint, f"{blk['state']}->{STATE_CLOSED}", "the endpoint answered")
    blk.update({"n": 0, "at": _time.time(), "state": STATE_CLOSED, "ever_ok": True,
                "pass_noted": False})


def route_note(route: str, endpoint: str, *, status: str, what: str = "",
               rows: int | None = None, error: str | None = None,
               body_head: str | None = None) -> dict:
    """Record one route attempt (and return it).

    ``body_head`` is the head of the server's own response, recorded whenever a
    request comes back with no rows.  A zero-row ASU response with no error line
    is otherwise indistinguishable from an empty sky, and the two demand
    opposite responses: IGNITION's probe reported ``QUERY_RETURNED_ZERO_ROWS``
    for a one-degree AllWISE cone, which cannot be an empty sky in a catalogue
    of 750 million sources, and there was nothing recorded to say why.
    """
    rec = {"route": str(route), "endpoint": str(endpoint), "status": str(status),
           "what": str(what)[:300]}
    if rows is not None:
        rec["rows"] = int(rows)
    if error is not None:
        rec["error"] = str(error)[:2000]
    if body_head is not None:
        rec["body_head"] = str(body_head)[:1200]
    ROUTE_LOG.append(rec)
    del ROUTE_LOG[:-_ROUTE_LOG_MAX]
    return rec


def route_log_summary() -> dict:
    """What served this process, and what failed, per route."""
    out: dict[str, dict] = {}
    for rec in ROUTE_LOG:
        blk = out.setdefault(rec["route"], {"n_ok": 0, "n_failed": 0, "endpoints": []})
        blk["n_ok" if rec["status"] == STATUS_OK else "n_failed"] += 1
        if rec["endpoint"] not in blk["endpoints"]:
            blk["endpoints"].append(rec["endpoint"])
    return {"routes": out, "n_attempts": len(ROUTE_LOG),
            "served_by": next((r["route"] for r in reversed(ROUTE_LOG)
                               if r["status"] == STATUS_OK), ROUTE_NONE),
            # endpoint states only: the full transition list rides in
            # AcquisitionLog.as_dict()["breaker"], and once is enough
            "breaker": {k: v for k, v in breaker_summary().items() if k != "transitions"}}


class VizierRouteError(RuntimeError):
    """Every route failed.  Carries every endpoint and every error verbatim.

    ``asu_supported`` is False when the query itself has no non-TAP equivalent
    (a ``COUNT(*)``, a description search): the caller can then degrade to
    "unknown" instead of treating it as an archive failure.
    """

    def __init__(self, message: str, *, attempts: list[dict] | None = None,
                 asu_supported: bool = True, adql: str = ""):
        super().__init__(message)
        self.attempts = list(attempts or [])
        self.asu_supported = bool(asu_supported)
        self.adql = str(adql)


class AdqlNotTranslatable(ValueError):
    """This ADQL has no ASU equivalent (recorded, never guessed at)."""


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
            rec["error"] = str(error)[:2000]
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
                "total_rows": int(sum(s["rows"] for s in self.stages)),
                # the route ladder and the breaker that governed it, so a run
                # that was served by (or starved by) a fallback says so
                "routes": route_log_summary(), "breaker": breaker_summary()}

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
            if attempt + 1 < int(retries):       # no sleep after the last attempt
                _time.sleep(base_sleep * (attempt + 1))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


def tap_query(adql: str, *, url=None, retries: int = 3, fetch_fn=None,
              allow_non_tap: bool = True, tap_fn=None) -> pd.DataFrame:
    """ADQL against VizieR TAP: async first, sync on the last attempt.

    Every endpoint in ``VIZIER_TAP_MIRRORS`` is tried in turn before the query
    is called failed, because a 503 from one endpoint is routine and says
    nothing about the sky.  The failure message names every endpoint tried.

    When every TAP endpoint is down (2026-09-13: all of them, for hours) the
    query is translated to VizieR's NON-TAP ASU interface by
    :func:`asu_query` and served from there --- the ADQL this repository emits
    is a narrow dialect (``SELECT [TOP n] cols FROM "table" [WHERE ...]``,
    ``TAP_SCHEMA.tables LIKE``, ``TAP_SCHEMA.columns WHERE table_name =``) and
    every one of those forms has an exact ASU equivalent.  A query with no
    equivalent (``COUNT(*)``, a description search) raises a
    :class:`VizierRouteError` with ``asu_supported=False`` rather than a guess.

    ``url`` may be a single endpoint or a sequence of them; the signature and
    the raise-on-total-failure behaviour are unchanged.  ``tap_fn(adql,
    endpoint) -> DataFrame`` replaces the pyvo transport (the offline tests
    drive the ladder and the breaker through it); the circuit breaker applies
    to it exactly as to the default transport, so breaker behaviour is
    testable without a socket.
    """
    if tap_fn is None:
        import pyvo  # noqa: PLC0415  runner-only; keeps the module importable offline
    else:
        pyvo = None

    if url is None:
        endpoints = list(VIZIER_TAP_MIRRORS)
    elif isinstance(url, str):
        endpoints = [url]
    else:
        endpoints = [str(u) for u in url]
    errors: list[str] = []
    attempts: list[dict] = []
    for endpoint in endpoints:
        skip = _circuit_open(endpoint)
        if skip:
            errors.append(f"{endpoint}: {skip}")
            attempts.append(route_note(ROUTE_TAP, endpoint, status=STATUS_FAILED, what=adql,
                                       error=skip))
            continue

        def _go(endpoint=endpoint):
            if tap_fn is not None:
                return tap_fn(adql, endpoint)
            svc = pyvo.dal.TAPService(endpoint)
            try:
                return svc.run_async(adql).to_table().to_pandas()
            except Exception as exc:                      # noqa: BLE001
                print(f"[metronome/acquire] async TAP failed ({exc!r}); trying sync")
                return svc.search(adql).to_table().to_pandas()

        # A half-open endpoint gets ONE cheap probe: the point is to find out
        # whether it is back, not to spend the retry ladder on it again.
        n_try = 1 if breaker_state(endpoint) == STATE_HALF_OPEN else retries
        try:
            df = _retry(_go, retries=n_try, label=f"TAP query ({endpoint})")
        except Exception as exc:                          # noqa: BLE001
            _circuit_fail(endpoint, repr(exc))
            errors.append(f"{endpoint}: {exc!r}")
            attempts.append(route_note(ROUTE_TAP, endpoint, status=STATUS_FAILED,
                                       what=adql, error=repr(exc)))
            continue
        _circuit_ok(endpoint)
        attempts.append(route_note(ROUTE_TAP, endpoint, status=STATUS_OK, what=adql,
                                   rows=0 if df is None else int(len(df))))
        if df is not None:
            df.attrs["route"] = ROUTE_TAP
            df.attrs["endpoint"] = endpoint
        return df
    asu_supported = True
    if allow_non_tap:
        try:
            df, asu_attempts = asu_query(adql, fetch_fn=fetch_fn)
            attempts.extend(asu_attempts)
            return df
        except AdqlNotTranslatable as exc:
            asu_supported = False
            errors.append(f"{VIZIER_ASU} (non-TAP): {exc}")
            attempts.append(route_note(ROUTE_ASU, VIZIER_ASU, status=STATUS_FAILED,
                                       what=adql, error=str(exc)))
        except VizierRouteError as exc:
            errors.extend(f"{a['endpoint']}: {a.get('error', '')}" for a in exc.attempts)
            attempts.extend(exc.attempts)
        except Exception as exc:                          # noqa: BLE001
            errors.append(f"{VIZIER_ASU} (non-TAP): {exc!r}")
            attempts.append(route_note(ROUTE_ASU, VIZIER_ASU, status=STATUS_FAILED,
                                       what=adql, error=repr(exc)))
    raise VizierRouteError("TAP query failed at every endpoint -- " + " | ".join(errors),
                           attempts=attempts, asu_supported=asu_supported, adql=adql)


# ---------------------------------------------------------------------------
# The NON-TAP route: VizieR's ASU interface (viz-bin/asu-tsv)
# ---------------------------------------------------------------------------
def split_catalogue(name: str) -> tuple[str, str | None]:
    """``"J/ApJ/787/112/table2"`` -> ``("J/ApJ/787/112", "table2")``.

    A journal catalogue id is ``J/<journal>/<volume>/<page>``; every other
    VizieR catalogue is ``<class>/<number>``.  Whatever follows is the table.
    """
    parts = [p for p in unquote_table(name).strip("/").split("/") if p]
    if not parts:
        return "", None
    head = 4 if parts[0].upper() == "J" else 2
    cat = "/".join(parts[:head])
    tail = "/".join(parts[head:])
    return cat, (tail or None)


def _http_status(exc) -> int | None:
    """The HTTP status behind a requests exception, when there is one."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _asu_http_text(url: str, *, timeout: float = 180.0) -> str:
    """GET a URL as text (runner only), short-circuiting a dead endpoint.

    A 4xx is a fact about the REQUEST (a form this host does not serve, a
    catalogue id it does not know), not about the host, so it is recorded as a
    failed attempt but never counted against the endpoint: letting a 404 on
    the ``-meta.all`` form open the breaker would take the ``-source=`` row
    form --- the one that works --- down with it.
    """
    import requests  # noqa: PLC0415  runner-only; keeps the module importable offline

    base = url.split("?", 1)[0]
    skip = _circuit_open(base)
    if skip:
        raise RuntimeError(skip)
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "seti-vizier/1.0 (+github actions; astronomy)"})
        r.raise_for_status()
    except Exception as exc:                              # noqa: BLE001
        status = _http_status(exc)
        if status is None or status >= 500:
            _circuit_fail(base, repr(exc))
        raise
    _circuit_ok(base)
    return r.text


def asu_url(catalogue: str, *, base: str = VIZIER_ASU, columns=None, max_rows: int = 100000,
            constraints: dict | None = None, meta: bool = False) -> str:
    """The ASU request for one catalogue / table.

    ``-source`` is the catalogue or table id, ``-out.max`` the row cap,
    ``-out.all`` every column (or ``-out=<name>`` per requested column) and
    ``-out.form=TSV`` the tab-separated text this module parses.  ``meta=True``
    asks for the METADATA of the catalogue (its tables and their columns)
    instead of rows --- the table-existence check that replaces TAP_SCHEMA.
    """
    from urllib.parse import quote  # noqa: PLC0415

    src = unquote_table(catalogue)
    parts = [f"-source={quote(src, safe='/+')}"]
    if meta:
        parts.append("-meta.all")
        parts.append("-out.form=TSV")
        return f"{base}?" + "&".join(parts)
    parts.append(f"-out.max={int(max_rows)}")
    cols = [unquote_table(c) for c in (columns or []) if str(c).strip()]
    if cols:
        parts.extend(f"-out={quote(c, safe='+_.-')}" for c in dict.fromkeys(cols))
    else:
        parts.append("-out.all")
    parts.append("-out.form=TSV")
    for col, expr in (constraints or {}).items():
        # ``+`` is NOT safe in a constraint VALUE.  A literal ``+`` in a query
        # string decodes to a SPACE, so ``-c=266+65`` reached VizieR as the
        # unsigned, dotless pair "266 65" and it could not read that as a
        # position at all --- it answered with an empty #RESOURCE (IGNITION
        # probe run 34799195807).  Percent-encoding makes a plus a plus and a
        # space a space, so a signed declination survives the wire.
        parts.append(f"{quote(str(col), safe='+_.-')}={quote(str(expr), safe='_.-,<>=')}")
    return f"{base}?" + "&".join(parts)


def _asu_error_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines()
            if ln.startswith("#***") or ln.startswith("****")]


def asu_body_head(text: str, limit: int = 900) -> str:
    """The head of an ASU response, with whitespace made visible.

    Tabs and newlines are escaped so a one-line JSON field still shows the
    record structure, and ``#Column``/``#***`` lines survive intact.  This is
    what turns "zero rows" from a guess into a reading of what VizieR said.
    """
    s = str(text or "")
    head = s[: int(limit)]
    out = head.replace("\t", "\\t").replace("\r", "").replace("\n", "\\n")
    return out + ("..." if len(s) > int(limit) else "")


def _parse_column_line(line: str) -> tuple[str, str, str]:
    """``#Column\tTpeak\t(d)\tFlare peak time\t[ucd=...]`` -> name, unit, description."""
    cells = [c.strip() for c in str(line).split("\t") if c.strip() != ""]
    name = unit = desc = ""
    for c in cells[1:]:
        if not name and not c.startswith(("(", "[")):
            name = unquote_table(c)
        elif c.startswith("(") and not unit:
            unit = c.strip("()")
        elif not c.startswith("[") and name and not desc:
            desc = c
    return name, unit, desc


def _asu_column_meta(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """``({column: unit}, {column: description})`` from a body's ``#Column`` lines."""
    units: dict[str, str] = {}
    descs: dict[str, str] = {}
    for ln in (text or "").splitlines():
        if ln.startswith("#Column"):
            name, unit, desc = _parse_column_line(ln)
            if name:
                units[name] = unit
                descs[name] = desc
    return units, descs


def parse_asu_tsv(text: str) -> pd.DataFrame:
    """Parse an ``asu-tsv`` body into a DataFrame.

    The body is ``#``-prefixed metadata (including one ``#Column`` line per
    column), then a header line of tab-separated column names, then a unit
    line, then a line of dashes, then the data.  Column names can carry the
    literal double quotes VizieR uses for awkward labels, so every header cell
    is unquoted; a blank header cell falls back to the ``#Column`` name at that
    position.  VizieR's own error lines (``#***``) are kept in
    ``df.attrs["asu_errors"]`` --- an error page must never read as zero rows.
    """
    lines = (text or "").splitlines()
    meta_cols: list[str] = []
    for ln in lines:
        if ln.startswith("#Column"):
            cells = [c.strip() for c in ln.split("\t")]
            name = ""
            for c in cells[1:]:
                if c and not (c.startswith("(") or c.startswith("[")):
                    name = c
                    break
            meta_cols.append(unquote_table(name))
    body = [ln for ln in lines if not ln.startswith("#")]
    while body and not body[0].strip():
        body.pop(0)
    errors = _asu_error_lines(text)
    if not body:
        out = pd.DataFrame()
        out.attrs["asu_errors"] = errors
        return out
    dash = None
    for i, ln in enumerate(body[:6]):
        cells = [c.strip() for c in ln.split("\t")]
        # ``-+``, not ``-{2,}``: VizieR writes a one-character rule under a
        # one-character column, and requiring two dashes meant the rule line
        # was not recognised at all, so body[0] (the rule) became the header.
        if cells and all(re.fullmatch(r"-+", c) for c in cells if c != ""):
            dash = i
            break
    if dash is None:
        header_i, data_from = 0, 1
    elif dash >= 2:
        header_i, data_from = dash - 2, dash + 1
    else:
        header_i, data_from = max(dash - 1, 0), dash + 1

    def _is_rule(line: str) -> bool:
        cells = [c.strip() for c in line.split("\t")]
        return bool(cells) and all(re.fullmatch(r"-+", c) for c in cells if c != "")

    # A rule is never a header.  Four of ARC's catalogues -- McQuillan+2014
    # (the rotational amplitudes the whole channel needs), Okamoto+2021,
    # Tu+2022 and Guenther+2020 -- came back with columns literally named
    # "--------", "--------__1", ... because the offset landed on a second
    # dashed rule, and every role then failed to resolve.  Walk up to the
    # nearest line that is neither blank nor a rule.
    if _is_rule(body[header_i]) or not body[header_i].strip():
        for j in range(header_i - 1, -1, -1):
            if body[j].strip() and not _is_rule(body[j]):
                header_i = j
                break
        else:
            for j in range(header_i + 1, min(dash if dash is not None else len(body),
                                             len(body))):
                if body[j].strip() and not _is_rule(body[j]):
                    header_i = j
                    break
    # ... and a rule is never data either: ASU emits a second rule under the
    # units line in some tables, and it used to arrive as the first row.
    while data_from < len(body) and (not body[data_from].strip()
                                     or _is_rule(body[data_from])):
        data_from += 1
    header = [unquote_table(c) for c in body[header_i].split("\t")]
    header = [h if h else (meta_cols[i] if i < len(meta_cols) else f"col{i}")
              for i, h in enumerate(header)]
    # The ``#Column`` block is the service's OWN column list and is
    # authoritative; the positional guess above (relative to the dashed rule)
    # picks the units or the description row whenever a table's preamble has a
    # different number of lines.  That is why ARC's run 34796... resolved ZERO
    # roles on tables that plainly carry a KIC column -- McQuillan+2014 (14
    # columns), Tu+2022 (11), Guenther+2020 (34, 30).  When the metadata names
    # match the table's width, they win.
    widest = max((len(ln.split("\t")) for ln in body[data_from:] if ln.strip()), default=0)
    if meta_cols and widest and len(meta_cols) == widest:
        if len(header) != widest or not all(
                h == m for h, m in zip(header, meta_cols, strict=False)):
            header = list(meta_cols)
    # VizieR repeats a column name in some tables (McQuillan+2014 and
    # Guenther+2020 both did, and both failed ARC's run 34792280736 with
    # TypeError('arg must be a list, tuple, 1-d array, or Series') -- a
    # duplicate label makes ``out[c]`` a DataFrame, which to_numeric refuses).
    # Suffix the repeats so every column is addressable, and record what was
    # renamed rather than hiding it.
    seen: dict[str, int] = {}
    deduped: list[str] = []
    renamed: list[str] = []
    for h in header:
        if h in seen:
            seen[h] += 1
            new = f"{h}__{seen[h]}"
            renamed.append(f"{h} -> {new}")
            deduped.append(new)
        else:
            seen[h] = 0
            deduped.append(h)
    header = deduped
    rows = []
    for ln in body[data_from:]:
        if not ln.strip():
            continue
        cells = [c.strip() for c in ln.split("\t")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        rows.append(cells[:len(header)])
    out = pd.DataFrame(rows, columns=header)
    for c in out.columns:
        s = out[c]
        if isinstance(s, pd.DataFrame):          # belt and braces: never to_numeric a frame
            continue
        s = s.replace("", None)
        num = pd.to_numeric(s, errors="coerce")
        out[c] = num if len(s) and num.notna().sum() == s.notna().sum() and s.notna().any() else s
    out.attrs["asu_errors"] = errors
    out.attrs["asu_renamed_columns"] = renamed
    return out


def parse_asu_meta(text: str) -> dict:
    """Parse an ``-meta.all`` body into ``{catalogue, title, tables: {...}}``.

    Each table block opens with ``#Table``/``#Name:`` and carries one
    ``#Column`` line per column.  This is the table-EXISTENCE check that
    replaces ``TAP_SCHEMA.tables`` / ``TAP_SCHEMA.columns`` when TAP is down.
    """
    out: dict = {"catalogue": "", "title": "", "tables": {}}
    cur: dict | None = None
    pending = False
    for ln in (text or "").splitlines():
        if not ln.startswith("#"):
            continue
        if ln.startswith("#Table"):
            pending = True
            cur = None
            continue
        m = re.match(r"^#Name:\s*(.+?)\s*$", ln)
        if m:
            name = unquote_table(m.group(1))
            if pending or ("/" in name and out["catalogue"] and name != out["catalogue"]):
                cur = out["tables"].setdefault(name, {"table_name": name, "description": "",
                                                      "columns": [], "units": {},
                                                      "descriptions": {}})
                pending = False
            elif not out["catalogue"]:
                out["catalogue"] = name
            continue
        m = re.match(r"^#Title:\s*(.*?)\s*$", ln)
        if m:
            if cur is not None:
                cur["description"] = m.group(1)
            elif not out["title"]:
                out["title"] = m.group(1)
            continue
        if ln.startswith("#Column") and cur is not None:
            name, unit, desc = _parse_column_line(ln)
            if name:
                cur["columns"].append(name)
                cur["units"][name] = unit
                cur["descriptions"][name] = desc
    return out


def parse_readme(text: str, catalogue: str = "") -> dict:
    """Tables and column labels from a catalogue ReadMe (the last existence check).

    The *File Summary* lists ``table2.dat`` with its record count; each
    *Byte-by-byte Description* block names the columns in its ``Label`` field.
    """
    out: dict = {"catalogue": catalogue, "title": "", "tables": {}}
    cur = None
    for ln in (text or "").splitlines():
        m = re.match(r"^\s*(\S+?)\.dat\s+\d+\s+(\d+)\s+(.*?)\s*$", ln)
        if m and not ln.lstrip().startswith("-"):
            name = f"{catalogue}/{m.group(1)}" if catalogue else m.group(1)
            out["tables"].setdefault(name, {"table_name": name, "n_rows": int(m.group(2)),
                                            "description": m.group(3), "columns": [],
                                            "units": {}, "descriptions": {}})
            continue
        m = re.match(r"^Byte-by-byte Description of file:\s*(\S+?)\.dat\s*$", ln)
        if m:
            name = f"{catalogue}/{m.group(1)}" if catalogue else m.group(1)
            cur = out["tables"].setdefault(name, {"table_name": name, "description": "",
                                                 "columns": [], "units": {}, "descriptions": {}})
            continue
        if cur is not None:
            m = re.match(r"^\s*\d+\s*-?\s*\d*\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$", ln)
            if m and re.match(r"^[A-Za-z_]", m.group(3)):
                label = m.group(3)
                cur["columns"].append(label)
                cur["units"][label] = m.group(2)
                cur["descriptions"][label] = m.group(4).strip()
    return out


def asu_meta(catalogue: str, *, fetch_fn=None, bases=None, readme_url: str = VIZIER_README
             ) -> tuple[dict, list[dict]]:
    """Catalogue metadata over the ASU ``-meta.all`` form, ReadMe as the backstop."""
    cat, _ = split_catalogue(catalogue)
    fetch = fetch_fn or _asu_http_text
    attempts: list[dict] = []
    for base in (bases or VIZIER_ASU_MIRRORS):
        url = asu_url(cat, base=base, meta=True)
        try:
            text = fetch(url)
            meta = parse_asu_meta(text)
        except Exception as exc:                          # noqa: BLE001
            attempts.append(route_note(ROUTE_ASU, base, status=STATUS_FAILED,
                                       what=f"meta {cat}", error=repr(exc)))
            continue
        if meta["tables"]:
            attempts.append(route_note(ROUTE_ASU, base, status=STATUS_OK, what=f"meta {cat}",
                                       rows=len(meta["tables"])))
            return meta, attempts
        attempts.append(route_note(ROUTE_ASU, base, status=STATUS_ZERO, what=f"meta {cat}",
                                   rows=0, error="; ".join(_asu_error_lines(text)) or None))
    url = readme_url.format(catalogue=cat)
    try:
        meta = parse_readme(fetch(url), cat)
    except Exception as exc:                              # noqa: BLE001
        attempts.append(route_note(ROUTE_README, url, status=STATUS_FAILED,
                                   what=f"ReadMe {cat}", error=repr(exc)))
        return {"catalogue": cat, "title": "", "tables": {}}, attempts
    attempts.append(route_note(ROUTE_README, url,
                               status=STATUS_OK if meta["tables"] else STATUS_ZERO,
                               what=f"ReadMe {cat}", rows=len(meta["tables"])))
    return meta, attempts


def _asu_body_names(text: str) -> list[str]:
    """Every ``#Name:`` the ASU body claims for itself."""
    return [unquote_table(ln.split(":", 1)[1]).strip("/") for ln in (text or "").splitlines()
            if ln.startswith("#Name:") and ":" in ln]


def _asu_identifies(requested: str, text: str) -> str | None:
    """The body's own name for what it served, or ``None`` if it is not ours.

    A TSV body that names a DIFFERENT catalogue than the ``-source=`` asked
    for is not evidence that ours exists (a mirror serving a cached or
    unrelated page), so it is rejected rather than believed.  A body with no
    ``#Name:`` line contradicts nothing and is taken at face value.
    """
    want = unquote_table(requested).strip("/").lower()
    names = _asu_body_names(text)
    if not names:
        return unquote_table(requested).strip("/")
    for n in names:
        low = n.lower()
        if low == want or low.startswith(want + "/"):
            return n                      # the exact table, or the table under our catalogue
    if any(want.startswith(n.lower() + "/") for n in names):
        return unquote_table(requested).strip("/")
    return None


def asu_table_exists(table: str, *, fetch_fn=None, bases=None, max_rows: int = 1
                     ) -> tuple[dict | None, list[dict]]:
    """Does this exact table exist, over ASU?  Ask it for ONE ROW.

    The ASU interface addresses a catalogue or table as
    ``-source=J/ApJ/906/72/table2`` and has no ``TAP_SCHEMA`` of its own, so
    the non-TAP spelling of "does ``TAP_SCHEMA.tables`` contain X" is "ask ASU
    for one row of ``-source=X``".  A well-formed TSV body --- a header of
    column names, no ``#***`` error line --- IS the existence proof, and the
    column names in that header are the table's real columns (which is all
    discovery needs; it never interpolates a name it has not seen).

    Returns ``({table_name, description, columns, units, descriptions,
    n_rows}, attempts)`` or ``(None, attempts)``.  Every endpoint tried is in
    ``attempts`` with its verbatim error.
    """
    fetch = fetch_fn or _asu_http_text
    name = unquote_table(table)
    attempts: list[dict] = []
    for base in (bases or VIZIER_ASU_MIRRORS):
        url = asu_url(name, base=base, max_rows=int(max_rows))
        try:
            text = fetch(url)
        except Exception as exc:                          # noqa: BLE001
            attempts.append(route_note(ROUTE_ASU, base, status=STATUS_FAILED,
                                       what=f"exists? {name}", error=repr(exc)))
            continue
        errs = _asu_error_lines(text)
        df = parse_asu_tsv(text)
        cols = [str(c) for c in df.columns]
        if not cols:
            attempts.append(route_note(
                ROUTE_ASU, base, status=STATUS_FAILED, what=f"exists? {name}",
                error="; ".join(errs) or "no column header in the ASU body"))
            continue
        served = _asu_identifies(name, text)
        if served is None:
            attempts.append(route_note(
                ROUTE_ASU, base, status=STATUS_FAILED, what=f"exists? {name}",
                error=f"the ASU body names {_asu_body_names(text)}, not {name!r}"))
            continue
        units, descs = _asu_column_meta(text)
        title = next((ln.split(":", 1)[1].strip() for ln in text.splitlines()
                      if ln.startswith("#Title:")), "")
        rec = {"table_name": served, "description": title, "columns": cols,
               "units": {c: units.get(c, "") for c in cols},
               "descriptions": {c: descs.get(c, "") for c in cols}, "n_rows": None}
        attempts.append(route_note(ROUTE_ASU, base, status=STATUS_OK, what=f"exists? {name}",
                                   rows=int(len(df))))
        return rec, attempts
    return None, attempts


def asu_readme_tables(catalogue: str, *, fetch_fn=None, readme_url: str = VIZIER_README
                      ) -> tuple[dict, list[dict]]:
    """A catalogue's own table inventory, from its ReadMe's File Summary.

    ``-meta.all`` is not a reliable enumeration: asked for a bare catalogue id
    it named a single table for both ``J/ApJS/209/5`` and ``J/ApJS/255/17`` on
    2026-09-14.  The ReadMe lists every ``<name>.dat`` with its record count,
    and its Byte-by-byte blocks give the real column labels, so a table found
    this way can be SCORED without a further round trip.
    """
    cat, _ = split_catalogue(unquote_table(catalogue).strip("%"))
    fetch = fetch_fn or _asu_http_text
    url = readme_url.format(catalogue=cat)
    try:
        meta = parse_readme(fetch(url), cat)
    except Exception as exc:                              # noqa: BLE001
        return {}, [route_note(ROUTE_README, url, status=STATUS_FAILED,
                               what=f"ReadMe {cat}", error=repr(exc))]
    tables = dict(meta.get("tables") or {})
    return tables, [route_note(ROUTE_README, url,
                               status=STATUS_OK if tables else STATUS_ZERO,
                               what=f"ReadMe {cat}", rows=len(tables))]


def asu_catalogue_tables(pattern: str, *, fetch_fn=None, bases=None, limit: int = 60
                         ) -> tuple[pd.DataFrame, list[dict]]:
    """``table_name, description, columns`` for every table under ``pattern``.

    The non-TAP replacement for ``SELECT ... FROM TAP_SCHEMA.tables WHERE
    table_name LIKE '%pattern%'``: it answers the only question discovery
    really asks --- does this catalogue exist, and what tables does it have?

    Order of routes.  When ``pattern`` names an EXACT table
    (``J/ApJ/906/72/table2``, which is how every ARC catalogue is asserted),
    the one-row existence check (:func:`asu_table_exists`) comes first: it is
    the ASU form most likely to answer, and its header gives the real columns.
    Otherwise --- and if that fails --- the catalogue's ``-meta.all`` metadata
    and then its ReadMe are tried, and a bare catalogue id finally falls back
    to a one-row request on the catalogue itself.  ARC's 2026-09-14 dispatch
    failed exactly here: with TAP 503ing, ``-meta.all`` was the ONLY non-TAP
    route and three catalogues resolved to nothing at all.
    """
    pat = unquote_table(pattern).strip("%")
    _cat, tail = split_catalogue(pat)
    attempts: list[dict] = []
    rows: list[dict] = []
    if tail:
        rec, att = asu_table_exists(pat, fetch_fn=fetch_fn, bases=bases)
        attempts.extend(att)
        if rec is not None:
            rows.append(rec)
    def _merge(blocks: dict) -> None:
        have = {str(r["table_name"]).lower() for r in rows}
        for name, blk in (blocks or {}).items():
            if pat and pat.strip("/").lower() not in name.lower():
                continue
            if str(name).lower() in have:
                continue
            rows.append({"table_name": name, "description": blk.get("description", ""),
                         "columns": list(blk.get("columns") or []),
                         "units": dict(blk.get("units") or {}),
                         "descriptions": dict(blk.get("descriptions") or {}),
                         "n_rows": blk.get("n_rows")})
            have.add(str(name).lower())

    if not rows:
        meta, meta_attempts = asu_meta(pat, fetch_fn=fetch_fn, bases=bases)
        attempts.extend(meta_attempts)
        _merge(meta.get("tables", {}))
    if not tail and len(rows) < 2:
        # A CATALOGUE has tables; ``-meta.all`` on a bare id came back naming
        # exactly ONE of them on 2026-09-14, which is how ARC lost the flares
        # half of Shibayama+2013 (it saw only J/ApJS/209/5/stars) and the
        # rotation half of Santos+2021 (only .../table1, the per-quarter Teff
        # table).  Both were then reported as catalogues that expose no usable
        # table at all.  The ReadMe's File Summary is the catalogue's own
        # inventory, so it is consulted whenever the metadata route named
        # fewer than two tables, and the two listings are unioned.
        rm, rm_attempts = asu_readme_tables(pat, fetch_fn=fetch_fn)
        attempts.extend(rm_attempts)
        _merge(rm)
    if not rows and not tail:
        rec, att = asu_table_exists(pat, fetch_fn=fetch_fn, bases=bases)
        attempts.extend(att)
        if rec is not None:
            rows.append(rec)
    df = pd.DataFrame(rows, columns=["table_name", "description", "columns", "units",
                                     "descriptions", "n_rows"])
    if len(df) > int(limit):
        df = df.head(int(limit))
    df.attrs["route"] = ROUTE_ASU
    df.attrs["attempts"] = attempts
    if not len(df):
        raise VizierRouteError(
            f"no non-TAP metadata for {pat!r} -- " + " | ".join(
                f"{a['endpoint']}: {a.get('error', a['status'])}" for a in attempts),
            attempts=attempts)
    return df, attempts


def asu_table_columns(table: str, *, fetch_fn=None, bases=None) -> tuple[pd.DataFrame, list[dict]]:
    """``column_name, unit, description`` of one table, without TAP."""
    name = unquote_table(table)
    df, attempts = asu_catalogue_tables(name, fetch_fn=fetch_fn, bases=bases, limit=200)
    hit = df[df["table_name"].str.lower() == name.lower()]
    if not len(hit):
        hit = df[df["table_name"].str.lower().str.endswith(name.lower().split("/")[-1])]
    if not len(hit):
        out = pd.DataFrame(columns=["column_name", "unit", "description"])
        out.attrs["route"] = ROUTE_ASU
        out.attrs["attempts"] = attempts
        return out, attempts
    row = hit.iloc[0]
    cols = list(row["columns"])
    out = pd.DataFrame({"column_name": cols,
                        "unit": [row["units"].get(c, "") for c in cols],
                        "description": [row["descriptions"].get(c, "") for c in cols]})
    out.attrs["route"] = ROUTE_ASU
    out.attrs["attempts"] = attempts
    return out, attempts


def asu_rows(catalogue: str, *, columns=None, max_rows: int = 100000, fetch_fn=None,
             bases=None, constraints: dict | None = None, row_slice=None
             ) -> tuple[pd.DataFrame, list[dict]]:
    """Rows of one table over the non-TAP ASU interface."""
    fetch = fetch_fn or _asu_http_text
    attempts: list[dict] = []
    table = unquote_table(catalogue)
    for base in (bases or VIZIER_ASU_MIRRORS):
        url = asu_url(table, base=base, columns=columns, max_rows=max_rows,
                      constraints=constraints)
        try:
            text = fetch(url)
            df = parse_asu_tsv(text)
        except Exception as exc:                          # noqa: BLE001
            attempts.append(route_note(ROUTE_ASU, base, status=STATUS_FAILED, what=url,
                                       error=repr(exc)))
            continue
        errs = df.attrs.get("asu_errors") or []
        if errs and not len(df):
            attempts.append(route_note(ROUTE_ASU, base, status=STATUS_FAILED, what=url,
                                       error="; ".join(errs)))
            continue
        if row_slice is not None:
            lo, hi = row_slice
            df = df.iloc[int(lo):int(hi)].reset_index(drop=True)
        attempts.append(route_note(ROUTE_ASU, base,
                                   status=STATUS_OK if len(df) else STATUS_ZERO,
                                   what=url, rows=int(len(df)),
                                   body_head=None if len(df) else asu_body_head(text)))
        df.attrs["route"] = ROUTE_ASU
        df.attrs["endpoint"] = base
        return df, attempts
    raise VizierRouteError(
        f"the ASU route failed for {table!r} -- " + " | ".join(
            f"{a['endpoint']}: {a.get('error', a['status'])}" for a in attempts),
        attempts=attempts)


def asu_constraint_ladder(catalogue: str, *, columns=None, constraints: dict | None = None,
                          max_rows: int = 5, fetch_fn=None, bases=None,
                          max_steps: int = 24) -> list[dict]:
    """Add one ASU parameter at a time and record where the rows stop coming.

    A zero-row ASU response names neither the parameter that killed it nor the
    reason.  IGNITION's probe run 34796722335 came back
    ``QUERY_RETURNED_ZERO_ROWS`` for *both* ``I/355/gaiadr3`` under the parent
    cuts and ``II/328/allwise`` under a bare one-degree cone; the second cannot
    be an empty sky, so at least one parameter spelling is wrong and nothing in
    the record said which.  This walks the request up from the barest form that
    can possibly work:

    ``bare`` (``-source`` and a row cap only) -> ``columns`` (the requested
    ``-out=`` list) -> one cumulative step per entry of ``constraints``, in the
    order given, so the cone is separated from the column cuts.

    The first step whose row count drops to zero is the culprit, and every
    step records the URL and, when it returns nothing, the head of VizieR's
    own response.  Returns the list of steps; never raises --- a ladder is a
    diagnostic and must not be able to fail a run.
    """
    fetch = fetch_fn or _asu_http_text
    table = unquote_table(catalogue)
    cols = [str(c) for c in (columns or []) if str(c).strip()]
    cons = dict(constraints or {})
    cap = max(1, int(max_rows))

    plan: list[tuple[str, str, list[str], dict]] = [("bare", "-source + -out.max", [], {})]
    if cols:
        plan.append(("columns", f"-out= x{len(cols)}", cols, {}))
    acc: dict = {}
    for key, val in cons.items():
        acc = {**acc, str(key): val}
        plan.append((f"+{key}", f"{key}={val}", cols, dict(acc)))
    plan = plan[: max(1, int(max_steps))]

    endpoints = list(bases or VIZIER_ASU_MIRRORS)
    steps: list[dict] = []
    base = endpoints[0] if endpoints else VIZIER_ASU
    for i, (label, added, step_cols, step_cons) in enumerate(plan):
        url = asu_url(table, base=base, columns=step_cols or None, max_rows=cap,
                      constraints=step_cons or None)
        step: dict = {"step": i, "label": label, "added": added, "endpoint": base, "url": url}
        try:
            text = fetch(url)
        except Exception as exc:                          # noqa: BLE001
            step.update(status=STATUS_FAILED, error=repr(exc))
            steps.append(step)
            if i == 0 and len(endpoints) > 1:             # the host, not the request
                base = endpoints[1]
            continue
        try:
            df = parse_asu_tsv(text)
        except Exception as exc:                          # noqa: BLE001
            step.update(status=STATUS_FAILED, error=repr(exc),
                        body_head=asu_body_head(text))
            steps.append(step)
            continue
        errs = list(df.attrs.get("asu_errors") or [])
        step["rows"] = int(len(df))
        step["columns"] = [str(c) for c in df.columns][:40]
        if errs:
            step["asu_errors"] = errs[:10]
        step["status"] = STATUS_OK if len(df) else STATUS_ZERO
        if not len(df):
            step["body_head"] = asu_body_head(text)
        steps.append(step)
    return steps


def asu_position_spellings(ra: float, dec: float, radius_deg: float) -> list[tuple[str, dict]]:
    """Candidate ASU spellings of one cone, most standard first.

    IGNITION's ladder (probe run 34799195807) put the blame on the cone
    parameter itself: ``-source`` alone served five rows of both
    ``I/355/gaiadr3`` and ``II/328/allwise``, and adding ``-c=266+65`` --- with
    no radius, no equinox, no column cut --- took both to zero.  VizieR's reply
    carried an EMPTY ``#RESOURCE=``/``#Name:``/``#Title:``, which is what it
    returns when it could not resolve the target at all: the position was not
    read as a position.  An unsigned, dotless pair is ambiguous with VizieR's
    sexagesimal form ("266 65" as hours and minutes is not a sky position), so
    these spellings differ in the decimal point, the explicit declination sign
    and whether the two coordinates are separate parameters.

    Which one VizieR actually accepts is a MEASUREMENT, not a guess: the ladder
    sends them all and the record says which returned rows.
    """
    ra_f, dec_f, rad = float(ra), float(dec), float(radius_deg)
    sign = "+" if dec_f >= 0 else "-"
    adec = abs(dec_f)
    return [
        ("decimal_signed", {"-c": f"{ra_f:.6f} {sign}{adec:.6f}", "-c.rd": f"{rad:.6f}",
                            "-c.eq": "J2000"}),
        ("decimal_unsigned", {"-c": f"{ra_f:.6f} {dec_f:.6f}", "-c.rd": f"{rad:.6f}",
                              "-c.eq": "J2000"}),
        ("split_ra_dec", {"-c.ra": f"{ra_f:.6f}", "-c.dec": f"{dec_f:.6f}",
                          "-c.rd": f"{rad:.6f}", "-c.eq": "J2000"}),
        ("radius_arcmin", {"-c": f"{ra_f:.6f} {sign}{adec:.6f}", "-c.rm": f"{rad * 60.0:.4f}",
                           "-c.eq": "J2000"}),
        ("bounding_box", {"-c": f"{ra_f:.6f} {sign}{adec:.6f}", "-c.bd": f"{rad * 2:.6f}",
                          "-c.eq": "J2000"}),
        ("integer_unsigned", {"-c": f"{ra_f:g} {dec_f:g}", "-c.rd": f"{rad:g}",
                              "-c.eq": "J2000"}),
    ]


def asu_position_ladder(catalogue: str, *, ra: float, dec: float, radius_deg: float,
                        columns=None, max_rows: int = 5, fetch_fn=None, bases=None
                        ) -> dict:
    """Send every cone spelling and report which one VizieR actually honours.

    Returns ``{steps, working, verdict}``.  ``working`` is the label of the
    first spelling that came back with rows, or ``None`` --- in which case the
    catalogue has no reachable cone over ASU at all and the channel must say so
    rather than report an empty sky.
    """
    fetch = fetch_fn or _asu_http_text
    table = unquote_table(catalogue)
    cols = [str(c) for c in (columns or []) if str(c).strip()]
    endpoints = list(bases or VIZIER_ASU_MIRRORS)
    base = endpoints[0] if endpoints else VIZIER_ASU
    steps: list[dict] = []
    working: str | None = None
    for i, (label, cons) in enumerate(asu_position_spellings(ra, dec, radius_deg)):
        url = asu_url(table, base=base, columns=cols or None, max_rows=max(1, int(max_rows)),
                      constraints=cons)
        step: dict = {"step": i, "label": label, "constraints": cons,
                      "endpoint": base, "url": url}
        try:
            text = fetch(url)
            df = parse_asu_tsv(text)
        except Exception as exc:                          # noqa: BLE001
            step.update(status=STATUS_FAILED, error=repr(exc))
            steps.append(step)
            continue
        errs = list(df.attrs.get("asu_errors") or [])
        step["rows"] = int(len(df))
        step["status"] = STATUS_OK if len(df) else STATUS_ZERO
        if errs:
            step["asu_errors"] = errs[:10]
        if not len(df):
            step["body_head"] = asu_body_head(text, 400)
        steps.append(step)
        if len(df) and working is None:
            working = label
            break                                        # the first that works is enough
    if working:
        verdict = {"status": "SPELLING_FOUND", "working": working,
                   "note": f"VizieR honours the {working!r} cone spelling; "
                           "the others above it returned nothing"}
    else:
        verdict = {"status": "NO_CONE_SPELLING_WORKS",
                   "tried": [s["label"] for s in steps],
                   "note": ("no ASU cone spelling returned a row for this catalogue; the "
                            "cone route is unavailable here and no empty sky may be inferred")}
    return {"catalogue": table, "steps": steps, "working": working, "verdict": verdict}


def ladder_verdict(steps: list[dict]) -> dict:
    """Which ladder step first lost the rows, stated plainly."""
    ok = [s for s in (steps or []) if s.get("status") == STATUS_OK]
    if not steps:
        return {"status": "NOT_RUN", "note": "no ladder step ran"}
    if not ok:
        first = steps[0]
        return {"status": "BARE_REQUEST_EMPTY",
                "culprit": first.get("label"),
                "note": ("even -source with a row cap and no constraint returned nothing, so "
                         "the catalogue id or the ASU form itself is wrong, not any cut"),
                "body_head": first.get("body_head"), "asu_errors": first.get("asu_errors")}
    last_ok = ok[-1]
    after = [s for s in steps if int(s.get("step", -1)) > int(last_ok.get("step", -1))]
    if not after:
        return {"status": "ALL_STEPS_RETURNED_ROWS",
                "note": "every step of the ladder returned rows; the empty result is elsewhere"}
    culprit = after[0]
    return {"status": "CONSTRAINT_ZEROED_THE_QUERY",
            "last_ok": last_ok.get("label"), "last_ok_rows": last_ok.get("rows"),
            "culprit": culprit.get("label"), "culprit_added": culprit.get("added"),
            "culprit_status": culprit.get("status"),
            "body_head": culprit.get("body_head"), "asu_errors": culprit.get("asu_errors"),
            "note": (f"rows survived {last_ok.get('label')} and died on "
                     f"{culprit.get('added')}")}


def astroquery_rows(catalogue: str, *, columns=None, max_rows: int = 100000, vizier_fn=None
                    ) -> tuple[pd.DataFrame, list[dict]]:
    """The last route: ``astroquery.vizier.Vizier`` (a declared dependency)."""
    table = unquote_table(catalogue)
    endpoint = "astroquery.vizier.Vizier"
    skip = _circuit_open(endpoint) if vizier_fn is None else ""
    if skip:
        raise VizierRouteError(skip, attempts=[
            route_note(ROUTE_ASTROQUERY, endpoint, status=STATUS_FAILED, what=table, error=skip)])

    def _default(cat, cols, limit):
        from astroquery.vizier import Vizier  # noqa: PLC0415  runner-only

        v = Vizier(columns=list(cols) if cols else ["**"], row_limit=int(limit))
        res = v.get_catalogs(cat)
        if res is None or len(res) == 0:
            return pd.DataFrame()
        return res[0].to_pandas()

    fn = vizier_fn or _default
    try:
        df = fn(table, list(columns or []), int(max_rows))
    except Exception as exc:                              # noqa: BLE001
        if vizier_fn is None:
            _circuit_fail(endpoint)
        att = [route_note(ROUTE_ASTROQUERY, endpoint, status=STATUS_FAILED, what=table,
                          error=repr(exc))]
        raise VizierRouteError(f"astroquery route failed for {table!r}: {exc!r}",
                               attempts=att) from exc
    if vizier_fn is None:
        _circuit_ok(endpoint)
    df = pd.DataFrame() if df is None else df
    att = [route_note(ROUTE_ASTROQUERY, endpoint,
                      status=STATUS_OK if len(df) else STATUS_ZERO, what=table, rows=int(len(df)))]
    df.attrs["route"] = ROUTE_ASTROQUERY
    return df, att


# ---------------------------------------------------------------------------
# ADQL -> ASU translation (the narrow dialect this repository emits)
# ---------------------------------------------------------------------------
_SELECT_RE = re.compile(r"^\s*SELECT\s+(?:TOP\s+(\d+)\s+)?(.*?)\s+FROM\s+(.*?)\s*$",
                        re.IGNORECASE | re.DOTALL)


def _split_select(cols: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in cols:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [unquote_table(c) for c in out if c.strip()]


def translate_adql(adql: str) -> dict:
    """What this ADQL asks for, in ASU terms.

    ``{"kind": "rows"|"tables"|"columns", ...}`` or a raise.  Deliberately
    narrow: a form that is not recognised is reported, never approximated.
    """
    text = " ".join(str(adql).split())
    m = _SELECT_RE.match(text)
    if not m:
        raise AdqlNotTranslatable(f"not a SELECT this translator recognises: {text[:200]!r}")
    top, cols, tail = m.group(1), m.group(2), m.group(3)
    parts = re.split(r"\s+WHERE\s+", tail, maxsplit=1, flags=re.IGNORECASE)
    table = unquote_table(parts[0])
    where = parts[1] if len(parts) > 1 else ""
    limit = int(top) if top else None
    low = table.lower()
    if low == "tap_schema.tables":
        if re.search(r"description\s+LIKE", where, re.IGNORECASE):
            raise AdqlNotTranslatable(
                "a TAP_SCHEMA description search has no ASU equivalent (VizieR's free-text "
                "search is not a table service); the catalogue id route is the non-TAP one")
        pats = re.findall(r"table_name\s+LIKE\s+'([^']*)'", where, re.IGNORECASE)
        if not pats:
            raise AdqlNotTranslatable(f"no table_name LIKE pattern in {where[:200]!r}")
        return {"kind": "tables", "pattern": pats[0].strip("%"), "limit": limit or 60}
    if low == "tap_schema.columns":
        names = re.findall(r"table_name\s*=\s*'([^']*)'", where, re.IGNORECASE)
        if not names:
            raise AdqlNotTranslatable(f"no table_name = '...' in {where[:200]!r}")
        return {"kind": "columns", "table": unquote_table(names[0])}
    if re.search(r"\bCOUNT\s*\(", cols, re.IGNORECASE):
        raise AdqlNotTranslatable(
            "COUNT(*) has no ASU equivalent: the non-TAP route reports an unknown row count "
            "rather than inventing one")
    select = _split_select(cols)
    if select == ["*"]:
        select = []
    out = {"kind": "rows", "table": table, "columns": select, "limit": limit,
           "constraints": {}, "row_slice": None}
    if where:
        m = re.match(r"^\s*recno\s+BETWEEN\s+(\d+)\s+AND\s+(\d+)\s*$", where, re.IGNORECASE)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            # ASU has no recno window, but it returns rows in recno order, so
            # "first hi rows, keep the last hi-lo+1" is the SAME rows -- an
            # exact translation, not an approximation.
            out["limit"] = hi
            out["row_slice"] = (lo - 1, hi)
            return out
        m = re.match(r"^\s*\"?([A-Za-z_][A-Za-z0-9_]*)\"?\s+IN\s*\((.*)\)\s*$", where,
                     re.IGNORECASE | re.DOTALL)
        if m:
            vals = [v.strip().strip("'") for v in m.group(2).split(",") if v.strip()]
            # verify: VizieR's ASU takes a comma-separated value list as an OR
            # constraint on a column.  Asserted from the ASU documentation and
            # NOT confirmed from here; a run that gets zero rows this way is
            # recorded as zero rows over route asu_tsv, not as a detection.
            out["constraints"] = {m.group(1): ",".join(vals)}
            out["limit"] = out["limit"] or max(len(vals) * 10, 100)
            return out
        raise AdqlNotTranslatable(f"WHERE clause has no ASU equivalent: {where[:200]!r}")
    return out


def asu_query(adql: str, *, fetch_fn=None, bases=None) -> tuple[pd.DataFrame, list[dict]]:
    """Serve one ADQL query over the non-TAP ASU route, or say why it cannot be."""
    plan = translate_adql(adql)
    if plan["kind"] == "tables":
        df, attempts = asu_catalogue_tables(plan["pattern"], fetch_fn=fetch_fn, bases=bases,
                                            limit=plan["limit"])
        return df[["table_name", "description", "columns"]].copy(), attempts
    if plan["kind"] == "columns":
        return asu_table_columns(plan["table"], fetch_fn=fetch_fn, bases=bases)
    return asu_rows(plan["table"], columns=plan["columns"], max_rows=plan["limit"] or 100000,
                    fetch_fn=fetch_fn, bases=bases, constraints=plan["constraints"],
                    row_slice=plan["row_slice"])


# ---------------------------------------------------------------------------
# The route ladder
# ---------------------------------------------------------------------------
@dataclass
class VizierResult:
    """Rows plus the route that served them, and every route that did not."""

    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    route: str = ROUTE_NONE
    endpoint: str = ""
    status: str = STATUS_FAILED
    attempts: list[dict] = field(default_factory=list)

    @property
    def errors(self) -> list[dict]:
        return [a for a in self.attempts if a["status"] == STATUS_FAILED]

    def as_dict(self) -> dict:
        return {"route": self.route, "endpoint": self.endpoint, "status": self.status,
                "n_rows": int(len(self.rows)), "attempts": self.attempts,
                "errors": self.errors}


def vizier_table(catalogue: str, *, columns=None, max_rows: int = 100000, adql: str | None = None,
                 tap_fn=None, tap_urls=None, fetch_fn=None, asu_bases=None, vizier_fn=None,
                 allow_astroquery: bool = True, log: AcquisitionLog | None = None,
                 stage: str = "vizier_table") -> VizierResult:
    """Every route to one VizieR table, in order, each recorded with its error.

    1. TAPVizieR (the current primary), 2. every other TAP host in
    ``VIZIER_TAP_MIRRORS``, 3. the NON-TAP ASU interface, 4.
    ``astroquery.vizier``.  The result says which route served the rows
    (``route == "asu_tsv"`` when TAP is down and ASU answered); when every
    route fails the rows are empty, the status is ``QUERY_FAILED`` and every
    endpoint and error is in ``attempts``.  No route ever invents a row.
    """
    table = unquote_table(catalogue)
    cols = [unquote_table(c) for c in (columns or []) if str(c).strip()]
    sel = ", ".join(f'"{c}"' for c in dict.fromkeys(cols)) if cols else "*"
    query = adql or f'SELECT TOP {int(max_rows)} {sel} FROM "{table}"'
    tap_fn = tap_fn or (lambda a, u: tap_query(a, url=u, retries=1, allow_non_tap=False))
    res = VizierResult()
    for endpoint in (tap_urls or VIZIER_TAP_MIRRORS):
        try:
            df = tap_fn(query, endpoint)
        except Exception as exc:                          # noqa: BLE001
            res.attempts.append(route_note(ROUTE_TAP, endpoint, status=STATUS_FAILED,
                                           what=query, error=repr(exc)))
            continue
        df = pd.DataFrame() if df is None else df
        res.attempts.append(route_note(ROUTE_TAP, endpoint,
                                       status=STATUS_OK if len(df) else STATUS_ZERO,
                                       what=query, rows=int(len(df))))
        res.rows, res.route, res.endpoint = df, ROUTE_TAP, endpoint
        res.status = STATUS_OK if len(df) else STATUS_ZERO
        break
    if res.route == ROUTE_NONE:
        try:
            df, attempts = asu_rows(table, columns=cols, max_rows=max_rows, fetch_fn=fetch_fn,
                                    bases=asu_bases)
            res.attempts.extend(attempts)
            res.rows, res.route = df, ROUTE_ASU
            res.endpoint = str(df.attrs.get("endpoint", VIZIER_ASU))
            res.status = STATUS_OK if len(df) else STATUS_ZERO
        except VizierRouteError as exc:
            res.attempts.extend(exc.attempts)
    if res.route == ROUTE_NONE and allow_astroquery:
        try:
            df, attempts = astroquery_rows(table, columns=cols, max_rows=max_rows,
                                           vizier_fn=vizier_fn)
            res.attempts.extend(attempts)
            res.rows, res.route, res.endpoint = df, ROUTE_ASTROQUERY, "astroquery.vizier.Vizier"
            res.status = STATUS_OK if len(df) else STATUS_ZERO
        except VizierRouteError as exc:
            res.attempts.extend(exc.attempts)
        except Exception as exc:                          # noqa: BLE001
            res.attempts.append(route_note(ROUTE_ASTROQUERY, "astroquery.vizier.Vizier",
                                           status=STATUS_FAILED, what=table, error=repr(exc)))
    if log is not None:
        log.record(stage, query, rows=int(len(res.rows)) if res.route != ROUTE_NONE else None,
                   error=None if res.route != ROUTE_NONE else
                   " | ".join(f"{a['endpoint']}: {a.get('error', '')}" for a in res.errors)[:2000],
                   extra={"route": res.route, "endpoint": res.endpoint,
                          "route_attempts": res.attempts})
    return res


def unquote_table(name: str) -> str:
    return str(name).strip().strip('"').strip()


def list_tables(pattern: str, *, query_fn=None, limit: int = 60, fetch_fn=None,
                allow_non_tap: bool = True) -> pd.DataFrame:
    """Tables in ``TAP_SCHEMA.tables`` whose name contains ``pattern``.

    When TAP cannot answer, this degrades to the NON-TAP existence check
    (:func:`asu_catalogue_tables`: the catalogue's own ASU metadata, ReadMe as
    the backstop) rather than raising --- "TAPVizieR is down" is not the same
    fact as "this catalogue does not exist", and only the second one is about
    the sky.  ``df.attrs["route"]`` says which route answered.  If the non-TAP
    route fails too, the raise names every endpoint tried on both.
    """
    query_fn = query_fn or tap_query
    adql = (f"SELECT TOP {int(limit)} table_name, description FROM TAP_SCHEMA.tables "
            f"WHERE table_name LIKE '%{pattern}%'")
    route, attempts = ROUTE_TAP, []
    try:
        df = query_fn(adql)
    except Exception as tap_exc:                          # noqa: BLE001
        if not allow_non_tap:
            raise
        attempts = list(getattr(tap_exc, "attempts", []))
        try:
            df, asu_attempts = asu_catalogue_tables(pattern, fetch_fn=fetch_fn, limit=limit)
        except VizierRouteError as asu_exc:
            raise VizierRouteError(
                f"no route to TAP_SCHEMA.tables ~ {pattern!r}: TAP: {tap_exc!r} | "
                f"non-TAP: {asu_exc}", attempts=attempts + list(asu_exc.attempts)) from tap_exc
        route, attempts = ROUTE_ASU, attempts + asu_attempts
    if df is None or not len(df):
        out = pd.DataFrame(columns=["table_name", "description"])
        out.attrs["route"] = route
        return out
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    df = df.copy()
    df["table_name"] = df["table_name"].map(unquote_table)
    keep = ["table_name"] + [c for c in ("description", "columns") if c in df]
    out = df[keep]
    out.attrs["route"] = str(df.attrs.get("route", route))
    out.attrs["attempts"] = attempts
    return out


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


def table_columns(table: str, *, query_fn=None, fetch_fn=None, known=None,
                  allow_non_tap: bool = True) -> list[str]:
    """Real column names of one table from ``TAP_SCHEMA.columns``, UNQUOTED.

    Degrades, in order, to column names already carried by a non-TAP table
    listing (``known``) and then to the catalogue's ASU metadata.

    TAPVizieR quotes awkward labels, and it quotes them in ``TAP_SCHEMA`` for
    COLUMNS exactly as it does for tables: the rows come back as ``"KIC"``,
    ``"Teff"``, ``"BP-RP"`` --- the double quotes are part of the string.  That
    is the whole of ARC run 35040024670, which reached **zero rows on all five
    flare catalogues** where the run before it had 5,785 stars.  Nothing about
    the sky changed; TAPVizieR came back up, discovery switched from the ASU
    route (which returns bare header cells) to TAP, and every role pattern ---
    ``^kic$`` against ``"KIC"`` --- stopped matching.  The scoreboard recorded
    ``rejected: no star_id`` for a table whose second column is plainly ``KIC``.

    Unquoting here fixes it for every caller at once, which is the only place
    it can be fixed once: a role resolver that strips quotes itself would still
    hand the quoted name on to whoever selects the column.
    """
    query_fn = query_fn or tap_query
    t = unquote_table(table)
    adql = ("SELECT TOP 2000 column_name FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '{t}' OR table_name = '\"{t}\"'")
    try:
        df = query_fn(adql)
    except Exception:                                     # noqa: BLE001
        if not allow_non_tap:
            raise
        if known:
            return [unquote_table(c) for c in known]
        df, _ = asu_table_columns(t, fetch_fn=fetch_fn)
    if df is None or not len(df):
        return [unquote_table(c) for c in (known or [])]
    col = "column_name" if "column_name" in df.columns else df.columns[0]
    return [unquote_table(c) for c in df[col].tolist()]


def count_rows(table: str, *, query_fn=None) -> int | None:
    """``COUNT(*)``, or ``None`` when no route can answer it.

    A row count has no ASU equivalent, so when TAP is down this returns
    ``None`` (*unknown*) instead of raising: discovery then ranks tables on
    their columns alone, which is what it does for any table whose count fails.
    """
    query_fn = query_fn or tap_query
    try:
        df = query_fn(f'SELECT COUNT(*) AS n FROM "{unquote_table(table)}"')
    except VizierRouteError as exc:
        if exc.asu_supported:
            raise
        print(f"[metronome/acquire] row count unavailable without TAP for {table!r}: {exc}")
        return None
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
    # MEASURED column names (ARC run 35040...: results/arc/probe.json), which
    # the first METRONOME dispatch never saw because TAP_SCHEMA's quoting hid
    # every column: Okamoto+2021 and Shibayama+2013 carry the flare PEAK time
    # as ``Date`` (BJD-2454833), Tu+2022 as ``PDate`` (peak date, BTJD),
    # Guenther+2020 as ``tpeak``; Yang & Liu 2019 carry no peak at all, only
    # ``Begin`` / ``End`` --- and ``^t_?beg(in)?$`` never matched ``begin``.
    "t_peak": [r"^t_?peak$", r"^tpk$", r"^peak_?time$", r"^bjd_?peak$", r"^peak$",
               r"^p_?date$", r"^peak_?date$", r"^date$", r"^tmax$", r"^t_?max$", r"^time$",
               r"^bjd$", r"^tflare$", r"^t_?fl$"],
    "t_start": [r"^t_?start$", r"^t_?beg(in)?$", r"^beg(in)?$", r"^start$", r"^bjd_?start$",
                r"^bjd_?beg(in)?$", r"^tstart$", r"^start_?time$", r"^t_?ini$", r"^t1$",
                r"^t0$"],
    "t_end": [r"^t_?end$", r"^t_?stop$", r"^end$", r"^stop$", r"^bjd_?end$", r"^t2$",
              r"^tend$", r"^end_?time$", r"^fin(ish)?$"],
    "energy": [r"^e$", r"^energy$", r"^e_?flare$", r"^ebol$", r"^log_?e$", r"^loge$",
               r"^log_?ebol$", r"^ekp$", r"^e_?kp$", r"^eflare$", r"^ed$", r"^e_?bol$"],
    "amplitude": [r"^amp(l|litude)?$", r"^a$", r"^fpeak$", r"^f_?peak$", r"^dflux$",
                  r"^rel_?amp$"],
    "sector": [r"^sector$", r"^sec$", r"^sectors$", r"^quarter$", r"^q$", r"^camp(aign)?$"],
    "prot": [r"^prot$", r"^p_?rot$", r"^rot_?per$", r"^rotper$", r"^per$", r"^period$",
             r"^p$"],
    "ra": [r"^ra_?icrs$", r"^raj2000$", r"^_?ra$", r"^ra_?deg$", r"^radeg$"],
    "dec": [r"^de_?icrs$", r"^dej2000$", r"^_?dec?$", r"^dec_?deg$", r"^dedeg$"],
    "duration": [r"^dur(ation)?$", r"^tdur$", r"^t_?dur$", r"^length$"],
    # A per-event classification the catalogue itself carries (Tu+2022's
    # ``Label``; a ``Flag``): kept so a shortlisted star's events can be read
    # back by class instead of guessed at.
    "label": [r"^label$", r"^flag$", r"^flags$", r"^class$", r"^type$", r"^note$", r"^qual(ity)?$"],
}

_TIME_OFFSET_RE = re.compile(r"(?:B?JD|TJD|BKJD|BTJD)\s*[-−–]\s*(2\s?4\d{5}(?:\.\d+)?)", re.I)


def time_offset_from_description(desc: str) -> float | None:
    """The offset a catalogue SAYS its time column carries (``BJD-2454833``,
    ``BJD-2400000``, ``BJD - 2457000``), or ``None`` when it says nothing.

    Okamoto+2021's and Shibayama+2013's ``Date`` are described as
    ``BJD-2400000`` --- not MJD (``JD-2400000.5``), which the value-range
    guess called them; the difference is a constant half day, harmless to
    a clock inside one catalogue but wrong against the published quarter
    windows.  The catalogue's own words win over the guess whenever present.
    """
    m = _TIME_OFFSET_RE.search(str(desc or ""))
    if not m:
        return None
    try:
        return float(m.group(1).replace(" ", ""))
    except ValueError:
        return None


def _canon(name: str) -> str:
    """Canonical form of a column name for role matching.

    The leading ``strip('"')`` is defence in depth against TAPVizieR's quoting:
    ``TAP_SCHEMA.columns`` hands back ``"KIC"`` with the double quotes inside
    the string, and ``^kic$`` does not match ``"kic"``.  ``table_columns``
    unquotes at the source, which is where it matters for the query; this makes
    a quoted name from any other path resolve too, instead of being silently
    reported as a missing role.
    """
    return re.sub(r"_+", "_", str(name).strip().strip('"').strip().lower()).strip("_")


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


def clean_star_id(v) -> str:
    """One spelling of a star id across every table: ``"KIC 757099"``,
    ``757099.0`` and ``757099`` are the same star, and a rotation table that
    spells it one way must join a flare table that spells it another."""
    s = str(v).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return re.sub(r"^(KIC|TIC|EPIC)\s*", "", s, flags=re.I).strip()


def column_descriptions(table: str, *, query_fn=None) -> dict[str, dict]:
    """``{column: {unit, description}}`` from ``TAP_SCHEMA.columns``.

    Used by the probe to record what the catalogue SAYS its time column is
    (``BJD-2454833``, ``BTJD``, ...) beside the value-range guess, so a wrong
    time system is visible in the artefact rather than discovered in a
    candidate.  Empty when the service does not answer; never required.
    """
    query_fn = query_fn or tap_query
    t = unquote_table(table)
    adql = ("SELECT TOP 2000 column_name, unit, description FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '{t}' OR table_name = '\"{t}\"'")
    df = query_fn(adql)
    out: dict[str, dict] = {}
    if df is None or not len(df):
        return out
    cols = {str(c).lower(): c for c in df.columns}
    name_c = cols.get("column_name", df.columns[0])
    for _, r in df.iterrows():
        name = unquote_table(r[name_c])
        out[name] = {"unit": str(r[cols["unit"]]) if "unit" in cols else "",
                     "description": str(r[cols["description"]]) if "description" in cols else ""}
    return out


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
    def _preferred_non_tap():
        """The catalogue's OWN metadata, asked directly, when TAP_SCHEMA is
        simply silent about it.

        ``list_tables`` degrades to this route when TAP *fails*, but not when
        TAP *answers with zero rows* --- and those are different facts.  A
        catalogue that TAPVizieR does not index (large tables are not always
        in its schema) is still served by ASU and still has a ReadMe, and
        Pietras+2022 has returned zero TAP rows under its bibcode id and under
        an author keyword on two separate dispatches.  Asking ASU costs one
        HTTP request and turns "TAP does not list it" into either the table or
        a recorded absence.
        """
        df, _ = asu_catalogue_tables(preferred, fetch_fn=None)
        return df

    routes = [("preferred", lambda: list_tables(preferred, query_fn=query_fn))]
    if keywords:
        routes.append(("keyword", lambda: search_tables(keywords, query_fn=query_fn)))
    routes.append(("preferred_non_tap", _preferred_non_tap))
    any_failed = False
    for route, lister in routes:
        if route == "preferred_non_tap" and any_failed:
            # TAP itself failed, and ``list_tables`` already fell through to
            # the non-TAP route inside that failure.  Asking again would only
            # re-walk endpoints the circuit breaker has just opened.
            continue
        try:
            tabs = lister()
        except Exception as exc:                          # noqa: BLE001
            if route == "preferred_non_tap":
                # A supplementary route that could not be reached does not
                # change what TAP said.  QUERY_RETURNED_ZERO_ROWS and
                # QUERY_FAILED are different facts, and only the TAP routes
                # decide between them, so this is a note and not an error.
                if log:
                    log.record(f"discover_{catalogue}_{route}", f"ASU metadata ~ {preferred!r}",
                               rows=0, extra={"note": repr(exc)[:300]})
                continue
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
                # the non-TAP listing already carries the real column names;
                # handing them over stops a second round trip per table and
                # lets discovery work at all when TAP_SCHEMA has no row for it
                cols = table_columns(t, query_fn=query_fn,
                                     known=list(row.get("columns") or [])
                                     if "columns" in tabs.columns else None)
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
    if "label" in out.columns:
        out["label"] = out["label"].astype(str).str.strip()
    if "star_id" in out.columns:
        out["star_id"] = out["star_id"].map(clean_star_id)
    if "sector" in out.columns:
        out["sector"] = pd.to_numeric(out["sector"], errors="coerce")
    return out


def discover_and_fetch_rotation(catalogue: str, preferred: str, keywords=(), *,
                                query_fn=None, log: AcquisitionLog | None = None,
                                max_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    """A rotation-period table: ``star_id, prot`` plus the discovery record."""
    query_fn = query_fn or tap_query
    rec = {"catalogue": catalogue, "preferred": preferred, "table": None, "roles": {},
           "status": STATUS_FAILED, "scoreboard": []}
    try:
        # An empty preferred id means "discover by keyword"; LIKE '%%' would
        # otherwise match the first sixty tables in the service.
        tabs = list_tables(preferred, query_fn=query_fn) if preferred else pd.DataFrame()
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
    # positions ride along when the table has them: the star tables of
    # Tu+2022 and Guenther+2020 carry _RA/_DE, and a shortlist that already
    # has a position does not need the TIC round trip that reached only 45%
    # of the first run's shortlist
    keep = ["star_id", "prot"] + [r for r in ("ra", "dec") if r in roles]
    sel = ", ".join(f'"{roles[r]}"' for r in keep)
    adql = f'SELECT {top}{sel} FROM "{t}"'
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
    rec.update({"table": t, "roles": {r: roles[r] for r in keep},
                "status": STATUS_OK if n else STATUS_ZERO, "n_rows": n})
    if not n:
        return pd.DataFrame(), rec
    out = pd.DataFrame({"star_id": df.iloc[:, 0].map(clean_star_id),
                        "prot": pd.to_numeric(df.iloc[:, 1], errors="coerce")})
    for j, r in enumerate(keep[2:], start=2):
        out[r] = pd.to_numeric(df.iloc[:, j], errors="coerce")
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
                frames.append(pd.DataFrame({"star_id": df.iloc[:, 0].map(clean_star_id),
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
    {star_id: {source, ...}})`` --- the second map lists, per star, the sources
    whose cone *answered* (with or without a match).  A star whose cone on some
    source failed carries ``variability_catalogue_unreached`` for that veto
    rather than a pass; a star whose cones all answered is fully vetted even
    if another star's cone timed out.
    """
    cone_fn = cone_fn or _vizier_cone
    out: dict[str, list] = {}
    reached: dict[str, set] = {}
    for name, spec in (catalogues or {}).items():
        table = spec.get("table")
        n_ok = n_fail = n_hits = 0
        for _, r in positions.iterrows():
            sid = str(r["star_id"])
            ra, dec = float(r.get("ra", np.nan)), float(r.get("dec", np.nan))
            if not (np.isfinite(ra) and np.isfinite(dec)):
                continue
            try:
                df = cone_fn(table, ra, dec, radius_arcsec)
                n_ok += 1
                reached.setdefault(sid, set()).add(name)
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
                out.setdefault(sid, []).append((name, float(p) if pd.notna(p)
                                                else float("nan"), vt))
                n_hits += 1
        if log:
            log.record(f"vari_{name}", f"{n_ok + n_fail} cones on {table} r={radius_arcsec}\"",
                       rows=n_hits if n_ok else None,
                       error=None if n_ok else "every cone failed",
                       extra={"n_cones_ok": n_ok, "n_cones_failed": n_fail})
    return out, reached


__all__ = ["BREAKER_LOG", "ROUTE_ASTROQUERY", "ROUTE_ASU", "ROUTE_NONE", "ROUTE_README",
           "ROUTE_TAP", "ROUTE_LOG", "STATE_CLOSED", "STATE_HALF_OPEN", "STATE_OPEN",
           "STATUS_FAILED", "STATUS_OK", "STATUS_ZERO", "VIZIER_ASU",
           "VIZIER_ASU_MIRRORS", "VIZIER_README", "VIZIER_TAP", "VIZIER_TAP_MIRRORS",
           "AdqlNotTranslatable", "AcquisitionLog", "DiscoveredTable", "VizierResult",
           "VizierRouteError", "astroquery_rows", "asu_body_head", "asu_catalogue_tables",
           "asu_constraint_ladder", "asu_meta",
           "asu_query", "asu_readme_tables", "asu_rows", "asu_table_columns",
           "asu_table_exists", "asu_url",
           "breaker_state", "breaker_summary", "clean_star_id", "column_descriptions",
           "count_rows", "ladder_verdict",
           "discover_and_fetch_rotation", "discover_event_table", "fetch_events",
           "fetch_positions_by_id", "fetch_variable_context", "list_tables",
           "parse_asu_meta", "parse_asu_tsv", "parse_readme", "reset_route_state",
           "resolve_columns", "resolve_event_columns", "route_log_summary", "route_note",
           "score_event_table", "search_tables", "split_catalogue", "table_columns",
           "tap_query", "time_offset_from_description", "translate_adql", "unquote_table",
           "vizier_table"]

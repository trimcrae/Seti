"""The IGNITION parent sample: Gaia DR3 x AllWISE, selected in the archive.

Runner-only where it queries; :func:`build_query` and :func:`select_parent`
are pure and tested offline.

Why the query has three shapes
------------------------------
Run 34787803862 (``--stage probe``) asked for ``SELECT TOP 5`` inside a
one-degree cone and was killed by ``canceling statement due to statement
timeout`` on every attempt, then by ``Error 503 ... maximum number of
synchronous queued jobs (150) reached``.  Five rows cannot be a volume problem,
so it was a *plan* problem and a *queue* problem:

* the query was one flat ``WHERE`` over a three-table join, so the predicates on
  ``gaiadr1.allwise_original_valid`` (``w1mpro - w2mpro``, ``ext_flag``,
  ``cc_flags``) and the absolute-magnitude expression sat at the same level as
  ``CONTAINS(...)``.  The planner is free to start from the ~750-million-row
  AllWISE mirror and scan it before the Gaia spatial index ever cuts the cone,
  and ``TOP 5`` does not help because the hash join must build its side first;
* the ``503`` says the request went to the **synchronous** endpoint, which is
  where the shared 150-job ceiling lives.

Both are fixed here.  The cut that an index can serve --- the cone, the parallax
shell, the ``random_index`` slice --- now goes in an inner sub-select on
``gaiadr3.gaia_source`` **alone**, together with every other Gaia-only cut, and
the AllWISE tables are joined to that small intermediate result.  The science
cuts are unchanged: :func:`gaia_predicates` and :func:`allwise_predicates` are
the single source of both, and each shape only decides *where* they are written.

:data:`SHAPES`, tried in order, and recorded in ``probe.json`` so the next run
uses the shape that answered instead of re-deriving it:

``inner_cone``
    inner sub-select on ``gaia_source``; AllWISE joined outside with the
    ``w.`` predicates in the outer ``WHERE``.
``inner_cone_postfilter``
    the same inner sub-select, but **no** ``w.`` predicates in SQL at all --- the
    AllWISE colour/quality cuts become a pandas post-filter, which
    :func:`select_parent` already applies to every frame regardless.
``flat``
    the original single-``WHERE`` join, kept last so that what failed is still
    executable and still on the record.

Transport: :func:`run_gaia_query` walks ``astroquery`` async, ``pyvo`` async,
then the sync endpoints, time-boxes every attempt, and records which queue
served the query.  Bulk queries run with ``allow_sync=False``.

Selection (``config/ignition.yaml`` -> ``sample``):

* ``G < 14.5``, ``parallax > 3 mas``, ``parallax_over_error > 10``,
  ``|b| > 15 deg``, ``ruwe < 1.4``, ``phot_variable_flag != 'VARIABLE'``;
* ``0.6 < bp_rp < 2.5`` with an **absolute-magnitude dwarf cut**
  ``M_G > a + s (bp_rp - c0)`` (``M_G = G + 5 log10(plx) - 10``), which removes
  giants and subgiants while keeping equal-mass binaries;
* the AllWISE cross-match through the archive's own neighbour table, requiring a
  **photospheric 2010 colour** ``-0.1 < W1 - W2 < 0.15`` (a negative colour is a
  blend, docs/channel-brief.md §4), ``ext_flag = 0``, ``cc_flags = '0000'`` and
  A/B photometry in W1 and W2;
* a **kinematic-age proxy**, ``v_tan > 25 km/s``, carried as a flag, not a cut.

Two sampling modes.  ``fields``: one cone per configured field --- the mode
the proven field-wide NEOWISE route needs.  ``allsky``: parallax shells (a
monolithic query times out; ``seti/herdsman/acquire.py``) with
``MOD(random_index, n_shards * stride) = shard`` subsampling, where the stride
is set from a ``COUNT(*)`` of the shell so that the subsample is *uniform* on
the sky and its fraction of the parent is known exactly.  A ``TOP`` cap alone
would return a sky-contiguous block (Gaia rows come back in HEALPix order),
which is neither random nor honest about its denominator.
"""

from __future__ import annotations

import re
import textwrap
import threading
import time as _time

import numpy as np
import pandas as pd

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

#: v_tan [km/s] = _K * mu [mas/yr] / parallax [mas]
_K = 4.740470446

QUERY_OK = "OK"
QUERY_ZERO = "QUERY_RETURNED_ZERO_ROWS"
QUERY_FAILED = "QUERY_FAILED"
QUERY_TIMED_OUT = "TIMED_OUT"

#: Candidate query shapes, tried in this order.  See the module docstring.
SHAPES: tuple[str, ...] = ("inner_cone", "inner_cone_postfilter", "flat")

_ALIAS_RE = re.compile(r"\bg\.")

DEFAULT_SAMPLE: dict = {
    "mode": "fields",
    "g_max": 14.5,
    "parallax_min_mas": 3.0,
    "parallax_over_error_min": 10.0,
    "abs_b_min_deg": 15.0,
    "ruwe_max": 1.4,
    "bp_rp_min": 0.6,
    "bp_rp_max": 2.5,
    "dwarf_cut": {"intercept": 2.0, "slope": 3.3, "colour0": 0.6},
    "w1w2_max": 0.15,
    "w1w2_min": -0.10,
    "v_tan_old_kms": 25.0,
    "cap_per_shard": 20000,
    "parallax_shell_edges_mas": [3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 1000.0],
    "count_parent": True,
    "fields": [],
    "allwise_columns": {"designation": "designation", "ext_flag": "ext_flag",
                        "ph_qual": "ph_qual", "cc_flags": "cc_flags", "var_flag": "var_flag"},
    # Query plan.  "auto" = try SHAPES in order and keep the one that answered
    # (probe.json's gaia_shape_working seeds this on later runs).
    "query_shape": "auto",
    # The inner sub-select's own TOP, as a multiple of the outer cap: it bounds
    # the intermediate result without biasing it, because the AllWISE join and
    # the colour cut only ever remove rows.  Never applied to COUNT(*).
    "inner_top_multiplier": 200,
    "inner_top_max": 2000000,
    "query_timeout_s": 1200.0,
}

GAIA_COLS = ("g.source_id, g.ra, g.dec, g.b, g.parallax, g.parallax_over_error, g.pmra, g.pmdec, "
             "g.ruwe, g.phot_g_mean_mag, g.bp_rp, g.phot_variable_flag, g.non_single_star, "
             "g.teff_gspphot, g.random_index")


def _wise_cols(conf: dict) -> tuple[str, dict]:
    ac = {**DEFAULT_SAMPLE["allwise_columns"], **(conf.get("allwise_columns") or {})}
    sel = (f"w.{ac['designation']} AS allwise_designation, w.w1mpro, w.w1mpro_error, "
           f"w.w2mpro, w.w2mpro_error, w.w3mpro, w.w3mpro_error, "
           f"w.{ac['cc_flags']} AS cc_flags, w.{ac['ph_qual']} AS ph_qual, "
           f"w.{ac['ext_flag']} AS ext_flag, xw.angular_distance AS allwise_sep_arcsec, "
           f"xw.number_of_neighbours AS allwise_n_neighbours")
    return sel, ac


def gaia_predicates(conf: dict | None = None, *, plx_lo: float | None = None,
                    plx_hi: float | None = None, field: dict | None = None,
                    mod: tuple[int, int] | None = None) -> list[str]:
    """Every cut that needs only ``gaiadr3.gaia_source``, index-usable ones first.

    One definition, shared by all three shapes in :data:`SHAPES`: that is what
    makes "only the shape changes" true rather than asserted.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    dc = {**DEFAULT_SAMPLE["dwarf_cut"], **(c.get("dwarf_cut") or {})}
    out: list[str] = []
    if field is not None:
        out.append(f"1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
                   f"CIRCLE('ICRS', {float(field['ra'])}, {float(field['dec'])}, "
                   f"{float(field['radius_deg'])}))")
    if plx_lo is not None:
        out.append(f"g.parallax >= {float(plx_lo)}")
    if plx_hi is not None:
        out.append(f"g.parallax < {float(plx_hi)}")
    if mod is not None:
        out.append(f"MOD(g.random_index, {int(mod[0])}) = {int(mod[1])}")
    out += [
        f"g.phot_g_mean_mag < {float(c['g_max'])}",
        f"g.parallax > {float(c['parallax_min_mas'])}",
        f"g.parallax_over_error > {float(c['parallax_over_error_min'])}",
        f"ABS(g.b) > {float(c['abs_b_min_deg'])}",
        f"g.ruwe < {float(c['ruwe_max'])}",
        "g.phot_variable_flag != 'VARIABLE'",
        f"g.bp_rp > {float(c['bp_rp_min'])} AND g.bp_rp < {float(c['bp_rp_max'])}",
        (f"g.phot_g_mean_mag + 5 * LOG10(g.parallax) - 10 > {float(dc['intercept'])} + "
         f"{float(dc['slope'])} * (g.bp_rp - {float(dc['colour0'])})"),
    ]
    return out


def allwise_predicates(conf: dict | None = None) -> list[str]:
    """Every cut that needs the AllWISE mirror ``w`` --- the ones that broke the plan."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    _sel, ac = _wise_cols(c)
    return [
        f"w.w1mpro - w.w2mpro < {float(c['w1w2_max'])}",
        f"w.w1mpro - w.w2mpro > {float(c['w1w2_min'])}",
        f"w.{ac['ext_flag']} = 0",
        f"w.{ac['cc_flags']} = '0000'",
    ]


def _join(conf: dict, left: str) -> str:
    """The two AllWISE joins hung off ``left`` (a table name or a sub-select)."""
    _sel, ac = _wise_cols(conf)
    return (f"FROM {left}\n"
            "  JOIN gaiadr3.allwise_best_neighbour AS xw ON xw.source_id = g.source_id\n"
            "  JOIN gaiadr1.allwise_original_valid AS w\n"
            f"    ON w.{ac['designation']} = xw.original_ext_source_id")


def _from(conf: dict) -> str:
    return _join(conf, "gaiadr3.gaia_source AS g")


#: Alias of ``gaia_source`` *inside* the sub-select.  Deliberately not ``g``:
#: the derived table is aliased ``g`` for the outer query, and an ADQL parser
#: should never be asked to resolve a shadowed alias when a rename is free.
INNER_ALIAS = "gs"


def _realias(text: str, alias: str = INNER_ALIAS) -> str:
    return _ALIAS_RE.sub(f"{alias}.", text)


def inner_top(conf: dict | None = None, cap: int | None = None) -> int | None:
    """The inner sub-select's ``TOP``: ``None`` (unbounded) unless ``cap`` is set."""
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    if not cap:
        return None
    mult = int(c.get("inner_top_multiplier") or 0)
    if mult <= 0:
        return None
    return int(min(int(cap) * mult, int(c.get("inner_top_max") or 0) or int(cap) * mult))


def build_query(conf: dict | None = None, *, plx_lo: float | None = None,
                plx_hi: float | None = None, field: dict | None = None,
                shard: int = 0, n_shards: int = 1, stride: int = 1,
                cap: int | None = None, count_only: bool = False,
                shape: str = "inner_cone") -> str:
    """ADQL for one parallax shell or one field cone, optionally subsampled.

    ``MOD(random_index, n_shards * stride) = shard`` selects a uniform
    ``1 / (n_shards * stride)`` of the parent.  ``count_only`` returns the
    ``COUNT(*)`` of the *unsubsampled* selection --- the denominator.  ``shape``
    is one of :data:`SHAPES` and changes only *where* the cuts are written.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    sel, _ac = _wise_cols(c)
    shape = str(shape or "inner_cone")
    if shape not in SHAPES:
        raise ValueError(f"unknown query shape {shape!r}; choose from {SHAPES}")
    m = int(max(n_shards, 1)) * int(max(stride, 1))
    # The subsample is part of the selection, but never of its denominator.
    mod = (m, int(shard)) if (m > 1 and not count_only) else None
    gp = gaia_predicates(c, plx_lo=plx_lo, plx_hi=plx_hi, field=field, mod=mod)
    wp = allwise_predicates(c)
    top = f"TOP {int(cap)} " if (cap and not count_only) else ""

    if shape == "flat":
        where = "\n  AND ".join(gp + wp)
        if count_only:
            return f"SELECT COUNT(*) AS n\n{_from(c)}\nWHERE {where}"
        return f"SELECT {top}{GAIA_COLS},\n  {sel}\n{_from(c)}\nWHERE {where}"

    itop = None if count_only else inner_top(c, cap)
    inner = (f"SELECT {f'TOP {itop} ' if itop else ''}{_realias(GAIA_COLS)}\n"
             f"FROM gaiadr3.gaia_source AS {INNER_ALIAS}\n"
             "WHERE " + "\n  AND ".join(_realias(p) for p in gp))
    src = "(\n" + textwrap.indent(inner, "  ") + "\n) AS g"
    tail = ("\nWHERE " + "\n  AND ".join(wp)) if shape == "inner_cone" else ""
    if count_only:
        return f"SELECT COUNT(*) AS n\n{_join(c, src)}{tail}"
    return f"SELECT {top}{GAIA_COLS},\n  {sel}\n{_join(c, src)}{tail}"


def parallax_shells(conf: dict | None = None) -> list[tuple[float, float]]:
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    edges = sorted(float(x) for x in c["parallax_shell_edges_mas"])
    edges = [e for e in edges if e >= float(c["parallax_min_mas"])]
    if not edges or edges[0] > float(c["parallax_min_mas"]):
        edges = [float(c["parallax_min_mas"])] + edges
    return list(zip(edges[:-1], edges[1:], strict=False))


# --------------------------------------------------------------------------
# Pure: derived quantities and the same cuts re-applied locally
# --------------------------------------------------------------------------
def select_parent(df: pd.DataFrame, conf: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Re-apply the selection to a frame (belt and braces) and add the flags.

    Adds ``abs_g``, ``v_tan_kms``, ``kinematically_old``, ``w1w2_2010``,
    ``agn_like_2010`` (never true after the cut, kept explicit).  Returns the
    kept frame and a counter dict naming what each cut removed.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    dc = {**DEFAULT_SAMPLE["dwarf_cut"], **(c.get("dwarf_cut") or {})}
    d = df.copy()
    d.columns = [str(x).lower() for x in d.columns]
    counters: dict = {"n_in": int(len(d))}
    if not len(d):
        counters["n_out"] = 0
        return d, counters
    d = d.reset_index(drop=True)
    num = {k: pd.to_numeric(d.get(k, pd.Series(np.nan, index=d.index)), errors="coerce")
           for k in ("phot_g_mean_mag", "parallax", "parallax_over_error", "ruwe", "bp_rp", "b",
                     "pmra", "pmdec", "w1mpro", "w2mpro", "ext_flag")}
    plx = num["parallax"]
    d["abs_g"] = num["phot_g_mean_mag"] + 5.0 * np.log10(plx.clip(lower=1e-6)) - 10.0
    mu = np.hypot(num["pmra"].fillna(0.0), num["pmdec"].fillna(0.0))
    d["v_tan_kms"] = _K * mu / plx.clip(lower=1e-6)
    d["kinematically_old"] = d["v_tan_kms"] > float(c["v_tan_old_kms"])
    d["w1w2_2010"] = num["w1mpro"] - num["w2mpro"]
    d["agn_like_2010"] = d["w1w2_2010"] > 0.8

    # Every mask is computed on the full frame; cuts are applied in sequence and
    # each counter is what that cut removed from the survivors of the previous one.
    masks: list[tuple[str, pd.Series]] = [
        ("g_max", num["phot_g_mean_mag"] < float(c["g_max"])),
        ("parallax", plx > float(c["parallax_min_mas"])),
        ("parallax_over_error", num["parallax_over_error"] > float(c["parallax_over_error_min"])),
    ]
    if num["b"].notna().any():
        masks.append(("galactic_latitude", num["b"].abs() > float(c["abs_b_min_deg"])))
    masks.append(("ruwe", num["ruwe"] < float(c["ruwe_max"])))
    if "phot_variable_flag" in d:
        masks.append(("gaia_variable",
                      d["phot_variable_flag"].astype(str).str.upper() != "VARIABLE"))
    masks.append(("colour", (num["bp_rp"] > float(c["bp_rp_min"]))
                  & (num["bp_rp"] < float(c["bp_rp_max"]))))
    masks.append(("dwarf", d["abs_g"] > float(dc["intercept"]) + float(dc["slope"])
                  * (num["bp_rp"] - float(dc["colour0"]))))
    masks.append(("w1w2_photospheric", (d["w1w2_2010"] < float(c["w1w2_max"]))
                  & (d["w1w2_2010"] > float(c["w1w2_min"]))))
    if num["ext_flag"].notna().any():
        masks.append(("ext_flag", num["ext_flag"].fillna(0) == 0))
    if "ph_qual" in d:
        pq = d["ph_qual"].astype(str).str.upper()
        masks.append(("ph_qual", pq.str[:1].isin(["A", "B"]) & pq.str[1:2].isin(["A", "B"])))
    alive = pd.Series(True, index=d.index)
    for name, keep in masks:
        keep = keep.fillna(False).astype(bool)
        counters[f"cut_{name}"] = int((alive & ~keep).sum())
        alive &= keep
    d = d[alive]
    counters["n_out"] = int(len(d))
    counters["n_kinematically_old"] = int(d["kinematically_old"].sum())
    return d.reset_index(drop=True), counters


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
class QueryTimeout(RuntimeError):
    """A transport did not answer within its time-box."""


class GaiaQueryFailed(RuntimeError):
    """No transport answered.  Carries the full :func:`run_gaia_query` record."""

    def __init__(self, record: dict):
        self.record = dict(record or {})
        super().__init__(f"{self.record.get('status')}: {self.record.get('error')}")


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).lower() for c in df.columns})


def _t_astroquery_async(adql: str) -> pd.DataFrame:
    from astroquery.gaia import Gaia  # noqa: PLC0415  runner-only

    return _lower(Gaia.launch_job_async(adql).get_results().to_pandas())


def _t_pyvo_async(adql: str) -> pd.DataFrame:
    import pyvo  # noqa: PLC0415  runner-only

    return _lower(pyvo.dal.TAPService(GAIA_TAP).run_async(adql).to_table().to_pandas())


def _t_astroquery_sync(adql: str) -> pd.DataFrame:
    from astroquery.gaia import Gaia  # noqa: PLC0415  runner-only

    return _lower(Gaia.launch_job(adql).get_results().to_pandas())


def _t_pyvo_sync(adql: str) -> pd.DataFrame:
    import pyvo  # noqa: PLC0415  runner-only

    return _lower(pyvo.dal.TAPService(GAIA_TAP).run_sync(adql).to_table().to_pandas())


#: ``(name, queue, callable)``.  Async first --- the ``503`` of run 34787803862
#: ("maximum number of synchronous queued jobs (150) reached") is the sync
#: endpoint's shared ceiling, so nothing but a trivial probe belongs there.
GAIA_TRANSPORTS: tuple[tuple[str, str, object], ...] = (
    ("astroquery_async", "async", _t_astroquery_async),
    ("pyvo_async", "async", _t_pyvo_async),
    ("astroquery_sync", "sync", _t_astroquery_sync),
    ("pyvo_sync", "sync", _t_pyvo_sync),
)


def call_with_timeout(fn, arg, timeout_s: float | None):
    """Run ``fn(arg)`` on a daemon thread; raise :class:`QueryTimeout` if late.

    A daemon thread rather than an executor: a hung archive call must not keep
    the interpreter (and the runner job) alive at exit.  This is what turns a
    broken plan from 1,490 s of retries into a bounded, recorded ``TIMED_OUT``.
    """
    if not timeout_s or timeout_s <= 0:
        return fn(arg)
    box: dict = {}

    def _go():
        try:
            box["value"] = fn(arg)
        except BaseException as exc:                   # noqa: BLE001
            box["error"] = exc

    th = threading.Thread(target=_go, daemon=True)
    th.start()
    th.join(float(timeout_s))
    if th.is_alive():
        raise QueryTimeout(f"no answer within {float(timeout_s):.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else float(deadline) - _time.monotonic()


def run_gaia_query(adql: str, *, label: str = "ignition", timeout_s: float | None = 1200.0,
                   retries_per_transport: int = 2, base_sleep: float = 4.0,
                   allow_sync: bool = True, deadline: float | None = None,
                   transports=None, tag: str = "ignition") -> tuple[pd.DataFrame, dict]:
    """Execute ADQL over the transport ladder.  Never raises.

    Returns ``(df, record)``; ``record["status"]`` is :data:`QUERY_OK`,
    :data:`QUERY_ZERO`, :data:`QUERY_TIMED_OUT` or :data:`QUERY_FAILED`, and
    ``record["queue"]`` names the queue (``async``/``sync``) that served it.
    ``deadline`` is a :func:`time.monotonic` instant past which no new attempt
    is started --- the probe's wall-clock budget.
    """
    transports = GAIA_TRANSPORTS if transports is None else transports
    rec: dict = {"label": label, "status": QUERY_FAILED, "n_rows": 0, "transport": None,
                 "queue": None, "attempts": [], "query": adql.strip()[:4000], "error": None,
                 "seconds": None, "allow_sync": bool(allow_sync)}
    t0 = _time.monotonic()
    timed_out_only = True
    for name, queue, fn in transports:
        if queue == "sync" and not allow_sync:
            rec["attempts"].append({"transport": name, "queue": queue, "ok": False,
                                    "error": "sync endpoint not used for this query (the "
                                             "150-job ceiling lives there)"})
            continue
        for attempt in range(max(1, int(retries_per_transport))):
            left = _remaining(deadline)
            if left is not None and left <= 0:
                rec.update(status=QUERY_TIMED_OUT, seconds=round(_time.monotonic() - t0, 1),
                           error=rec["error"] or "budget exhausted before the attempt")
                return pd.DataFrame(), rec
            per = timeout_s if left is None else (min(timeout_s, left) if timeout_s else left)
            ta = _time.monotonic()
            try:
                df = call_with_timeout(fn, adql, per)
            except Exception as exc:                   # noqa: BLE001
                err = repr(exc)
                rec["attempts"].append({"transport": name, "queue": queue, "ok": False,
                                        "error": err, "seconds": round(_time.monotonic() - ta, 1)})
                rec["error"] = err
                print(f"[{tag}] {label}: {name} attempt {attempt + 1}/{retries_per_transport} "
                      f"failed: {err[:300]}", flush=True)
                if not isinstance(exc, QueryTimeout):
                    timed_out_only = False
                else:
                    break            # a retry of a query that ran out of time is the same query
                left = _remaining(deadline)
                nap = base_sleep * (2 ** attempt)
                if left is not None:
                    nap = min(nap, max(left - 1.0, 0.0))
                if nap > 0:
                    _time.sleep(nap)
                continue
            n = int(len(df))
            rec["attempts"].append({"transport": name, "queue": queue, "ok": True, "n_rows": n,
                                    "seconds": round(_time.monotonic() - ta, 1)})
            rec.update(status=QUERY_OK if n else QUERY_ZERO, n_rows=n, transport=name,
                       queue=queue, seconds=round(_time.monotonic() - t0, 1))
            return df, rec
    rec["seconds"] = round(_time.monotonic() - t0, 1)
    if timed_out_only and rec["error"]:
        rec["status"] = QUERY_TIMED_OUT
    print(f"[{tag}] {label}: {rec['status']} on every transport", flush=True)
    return pd.DataFrame(), rec


def gaia_query(adql: str, retries: int = 2, tag: str = "ignition", *, label: str = "gaia",
               allow_sync: bool = True, timeout_s: float | None = 1200.0,
               deadline: float | None = None) -> pd.DataFrame:
    """:func:`run_gaia_query` as a plain ``adql -> DataFrame``; raises on failure."""
    df, rec = run_gaia_query(adql, label=label, timeout_s=timeout_s, allow_sync=allow_sync,
                             retries_per_transport=retries, deadline=deadline, tag=tag)
    if rec["status"] in (QUERY_FAILED, QUERY_TIMED_OUT):
        raise GaiaQueryFailed(rec)
    return df


def query_fn_with_record(*, label: str = "gaia", allow_sync: bool = False,
                         timeout_s: float | None = 1200.0, deadline: float | None = None):
    """An ``adql -> (df, record)`` callable; raises :class:`GaiaQueryFailed` on failure.

    ``allow_sync=False`` by default: bulk queries stay on the async queue.
    """
    def _fn(adql: str):
        df, rec = run_gaia_query(adql, label=label, timeout_s=timeout_s, allow_sync=allow_sync,
                                 deadline=deadline)
        if rec["status"] in (QUERY_FAILED, QUERY_TIMED_OUT):
            raise GaiaQueryFailed(rec)
        return df, rec
    return _fn


def unwrap_result(res) -> tuple[pd.DataFrame, dict]:
    """Accept either ``df`` or ``(df, record)`` from an injected ``query_fn``."""
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        return res[0], dict(res[1])
    return res, {}


def _transport_note(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("transport", "queue", "seconds") if rec.get(k) is not None}


def _count(adql: str, query_fn, ledger: list, label: str, shape: str = "") -> int | None:
    try:
        df, qrec = unwrap_result(query_fn(adql))
        n = int(pd.to_numeric(df.iloc[0, 0])) if len(df) else 0
        ledger.append({"label": label, "status": "OK", "n": n, "shape": shape,
                       "query": adql[:1500], **_transport_note(qrec)})
        return n
    except Exception as exc:                           # noqa: BLE001
        ledger.append({"label": label, "status": "QUERY_FAILED", "error": repr(exc),
                       "shape": shape, "query": adql[:1500]})
        return None


def _shape_order(conf: dict, shape: str | None, working: str | None) -> list[str]:
    """Preferred shape first, then the rest of :data:`SHAPES` as fallbacks."""
    pref: list[str] = []
    for s in (working, shape, conf.get("query_shape")):
        if s and s != "auto" and s in SHAPES and s not in pref:
            pref.append(s)
    return pref + [s for s in SHAPES if s not in pref]


def fetch_parent(conf: dict | None = None, *, mode: str | None = None, n_shards: int = 1,
                 cap_per_shard: int | None = None, query_fn=None,
                 fields: list[dict] | None = None,
                 shape: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Pull the parent sample and report its denominator honestly.

    Returns ``(stars, report)``.  ``report["status"]`` is ``OK``,
    ``QUERY_RETURNED_ZERO_ROWS`` or ``QUERY_FAILED``; ``report["parent_count"]``
    is the archive's ``COUNT(*)`` of the full selection when it could be
    measured, and ``report["subsample_fraction"]`` the fraction actually pulled.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    mode = mode or str(c.get("mode", "fields"))
    cap = int(cap_per_shard or c["cap_per_shard"])
    query_fn = query_fn or query_fn_with_record(label="ignition_sample", allow_sync=False,
                                                timeout_s=float(c.get("query_timeout_s")
                                                                 or 1200.0))
    fields = fields if fields is not None else list(c.get("fields") or [])
    ledger: list[dict] = []
    frames: list[pd.DataFrame] = []
    n_failed = 0
    parent_count: int | None = 0
    working: str | None = None
    t0 = _time.monotonic()

    units: list[dict]
    if mode == "fields":
        units = [{"field": f} for f in fields]
    else:
        units = [{"plx_lo": lo, "plx_hi": hi} for lo, hi in parallax_shells(c)]
    total_cap = cap * max(int(n_shards), 1)
    per_unit = []
    for u in units:
        label = (f"field_ra{u['field']['ra']}_dec{u['field']['dec']}" if "field" in u
                 else f"shell_{u['plx_lo']}_{u['plx_hi']}")
        n_unit = None
        stride = 1
        if c.get("count_parent", True):
            for sh in _shape_order(c, shape, working):
                n_unit = _count(build_query(c, count_only=True, shape=sh, **u), query_fn,
                                ledger, f"count_{label}", shape=sh)
                if n_unit is not None:
                    working = sh
                    break
            if n_unit is None:
                parent_count = None
            elif parent_count is not None:
                parent_count += n_unit
        if mode != "fields" and n_unit:
            # Spread the run's total cap across shells in proportion to their size.
            share = max(int(total_cap / max(len(units), 1)), 1)
            stride = max(int(np.ceil(n_unit / share)), 1)
        answered = False
        for sh in _shape_order(c, shape, working):
            q = build_query(c, stride=stride, shape=sh,
                            cap=(cap * max(int(n_shards), 1) if mode == "fields" else None), **u)
            try:
                df, qrec = unwrap_result(query_fn(q))
            except Exception as exc:                   # noqa: BLE001
                ledger.append({"label": label, "status": "QUERY_FAILED", "error": repr(exc),
                               "shape": sh, "query": q[:2000]})
                print(f"[ignition] {label}: QUERY_FAILED on shape={sh} {exc!r}")
                continue
            df = df.rename(columns={x: str(x).lower() for x in df.columns})
            st = "OK" if len(df) else "QUERY_RETURNED_ZERO_ROWS"
            ledger.append({"label": label, "status": st, "n_rows": int(len(df)),
                           "stride": stride, "shape": sh, "query": q[:2000],
                           **_transport_note(qrec)})
            per_unit.append({"unit": label, "n_parent": n_unit, "n_rows": int(len(df)),
                             "stride": stride, "fraction": (1.0 / stride), "shape": sh,
                             **_transport_note(qrec)})
            working, answered = sh, True
            if len(df):
                df["sample_unit"] = label
                df["subsample_stride"] = stride
                df["query_shape"] = sh
                frames.append(df)
            break
        if not answered:
            n_failed += 1
            per_unit.append({"unit": label, "n_parent": n_unit, "n_rows": 0,
                             "status": "QUERY_FAILED",
                             "shapes_tried": _shape_order(c, shape, working)})

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(raw):
        raw = raw.drop_duplicates("source_id")
    stars, counters = select_parent(raw, c) if len(raw) else (raw, {"n_in": 0, "n_out": 0})
    if len(stars):
        # Uniform, seed-fixed order so shards partition the same way on every run.
        stars = stars.sort_values("random_index" if "random_index" in stars else "source_id")
        stars = stars.reset_index(drop=True)
    status = ("OK" if len(stars) else
              ("QUERY_FAILED" if n_failed == len(units) and units else
               "QUERY_RETURNED_ZERO_ROWS"))
    n_pulled = int(len(raw))
    report = {
        "status": status, "mode": mode, "n_units": len(units), "n_units_failed": n_failed,
        "query_shape_requested": shape or c.get("query_shape") or "auto",
        "query_shape_used": working,
        "n_rows_pulled": n_pulled, "n_after_local_cuts": int(len(stars)),
        "parent_count": parent_count,
        "subsample_fraction": (n_pulled / parent_count if parent_count else None),
        "per_unit": per_unit, "local_cut_counters": counters, "ledger": ledger,
        "elapsed_s": round(_time.monotonic() - t0, 1),
        "denominator_note": (
            "parent_count is the archive COUNT(*) of the full selection (all units); "
            "n_rows_pulled is what this run actually searched. Any verdict is a count "
            "over n_rows_pulled, never a statement about the parent."),
    }
    return stars, report


__all__ = ["DEFAULT_SAMPLE", "GAIA_COLS", "GAIA_TAP", "GAIA_TRANSPORTS", "QUERY_FAILED",
           "QUERY_OK", "QUERY_TIMED_OUT", "QUERY_ZERO", "SHAPES", "GaiaQueryFailed",
           "QueryTimeout", "allwise_predicates", "call_with_timeout", "build_query", "fetch_parent", "gaia_predicates",
           "gaia_query", "inner_top", "parallax_shells", "query_fn_with_record", "run_gaia_query",
           "select_parent", "unwrap_result"]

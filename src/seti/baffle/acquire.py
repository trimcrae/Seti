"""BAFFLE acquisition (runner only — the sandbox has no archive egress).

Two tracks, both sharded on 5° declination bands and checkpointed per leaf
chunk to ``results/baffle/chunks/*.parquet``, with a per-chunk acquisition
ledger that keeps QUERY_OK / QUERY_RETURNED_ZERO_ROWS / QUERY_FAILED /
QUERY_TRUNCATED apart:

``deficit``
    Gaia DR3 ``gaia_source`` × ``allwise_best_neighbour`` ×
    ``allwise_original_valid`` × ``tmass_psc_xsc_best_neighbour`` ×
    ``tmass_original_valid`` — the in-archive pre-selection
    ``(Ks − W1) < pre_cut_w1 OR (Ks − W2) < pre_cut_w2`` unioned with a uniform
    ``random_index < locus_random_index_max`` calibration subsample (≈0.5 % of
    DR3) that becomes the photospheric locus.  Each returned row is flagged
    ``is_locus_sample``.
``missing``
    bright (G < 13) stars with a 2MASS match and NO AllWISE counterpart at all
    (``LEFT OUTER JOIN gaiadr3.allwise_neighbourhood … WHERE xn.source_id IS
    NULL``), plus one per-band denominator query (G < 13 with a 2MASS match,
    binned in |b| and G) so the missing fraction is a *number* and Galactic
    plane confusion is visible as a function of |b|.

Every query builder below is a pure string function; the transport is only
touched by :func:`run_gaia_query` and is injectable everywhere as ``runner``.

Hard-won archive lessons, inherited from CENOTAPH (``seti.cenotaph.acquire``)
----------------------------------------------------------------------------
* **The sync endpoint is a 60 s execution cut, not a row cap.**  A query that
  outgrows it is cut mid-stream at a power of two plus one rows.  The defence
  is *smaller queries*, not a bigger TOP: every band is measured against
  ``SELECT COUNT(*)`` over the *same* FROM/WHERE text, and a shortfall is
  ``QUERY_TRUNCATED`` — never a result.
* **The anonymous async result store fills up** ("Filesystem quota exceeded for
  user anonymous").  It is not permanent, so a transport that reports it goes on
  a *cooldown* of N queries rather than being disabled.
* **``random_index`` is a uniform, indexed permutation** over 1,811,709,771 rows,
  so halving on it is both cheap and statistically neutral.  A band that fails
  or truncates is split recursively on ``random_index`` halves down to a floor,
  and every leaf is a ledger entry.
* **One failed chunk must never destroy the others.**  Each leaf is contained,
  checkpointed to its own parquet, and reused on re-run.
* **Column names are probed at runtime** (``TOP 1 *`` of each mirror table); a
  missing column degrades to NaN rather than costing a runner job.

Column names that could NOT be verified offline (resolved by the probe):
``w1mjd_mean`` vs ``w1mjdmean`` on ``allwise_original_valid``; any 2MASS
contamination flag on ``tmass_original_valid`` (``cc_flg``/``ccflg``/
``contamination_flag`` — the Gaia mirror may carry none); and the 2MASS join key
on ``tmass_psc_xsc_best_neighbour`` (``original_psc_source_id`` vs
``original_ext_source_id``).
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

# Ledger vocabulary — copied verbatim from CENOTAPH so the two channels' ledgers
# read the same.  These are four different facts and are never merged.
QUERY_OK = "QUERY_OK"
QUERY_ZERO = "QUERY_RETURNED_ZERO_ROWS"
QUERY_FAILED = "QUERY_FAILED"
QUERY_TRUNCATED = "QUERY_TRUNCATED"

GAIA_RANDOM_INDEX_MAX = 1_811_709_771
LOCUS_RANDOM_INDEX_MAX = 9_058_548          # ≈ 0.5 % of DR3

TABLES = {
    "gaia": "gaiadr3.gaia_source",
    "xw": "gaiadr3.allwise_best_neighbour",
    "wise": "gaiadr1.allwise_original_valid",
    "xt": "gaiadr3.tmass_psc_xsc_best_neighbour",
    "tmass": "gaiadr1.tmass_original_valid",
    "xn": "gaiadr3.allwise_neighbourhood",
}

GAIA_COLUMNS = (
    "source_id", "ra", "dec", "l", "b", "ecl_lat", "parallax", "parallax_error",
    "parallax_over_error", "pmra", "pmdec", "ruwe", "phot_g_mean_mag",
    "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "phot_variable_flag",
    "non_single_star", "ipd_frac_multi_peak", "phot_bp_rp_excess_factor",
    "random_index",
)

# Cross-match columns: output name -> archive column (fixed names in DR3).
XW_COLUMNS = {"wise_angular_distance": "angular_distance",
              "wise_number_of_neighbours": "number_of_neighbours",
              "wise_number_of_mates": "number_of_mates"}
XT_COLUMNS = {"tmass_angular_distance": "angular_distance",
              "tmass_number_of_neighbours": "number_of_neighbours",
              "tmass_number_of_mates": "number_of_mates"}

# AllWISE mirror: output name -> candidate archive names, first hit wins.
WISE_WANT = {
    "wise_designation": ["designation", "allwise"],
    "wise_ra": ["ra", "raj2000"], "wise_dec": ["dec", "dej2000", "de"],
    "w1mpro": ["w1mpro"], "w1mpro_error": ["w1mpro_error", "w1sigmpro"],
    "w1snr": ["w1snr"], "w1rchi2": ["w1rchi2"],
    "w2mpro": ["w2mpro"], "w2mpro_error": ["w2mpro_error", "w2sigmpro"],
    "w2snr": ["w2snr"], "w2rchi2": ["w2rchi2"],
    "w3mpro": ["w3mpro"], "w3mpro_error": ["w3mpro_error", "w3sigmpro"],
    "w3snr": ["w3snr"],
    "w4mpro": ["w4mpro"], "w4mpro_error": ["w4mpro_error", "w4sigmpro"],
    "w4snr": ["w4snr"],
    "cc_flags": ["cc_flags", "ccf"], "ext_flag": ["ext_flag", "ext_flg"],
    "var_flag": ["var_flag", "var_flg"], "ph_qual": ["ph_qual", "qph"],
    "w1mjd_mean": ["w1mjd_mean", "w1mjdmean"],
    "w2mjd_mean": ["w2mjd_mean", "w2mjdmean"],
}
# 2MASS mirror.
TMASS_WANT = {
    "tmass_designation": ["designation"],
    "j_m": ["j_m", "jmag"], "j_msigcom": ["j_msigcom", "j_cmsig", "e_jmag"],
    "h_m": ["h_m", "hmag"], "h_msigcom": ["h_msigcom", "h_cmsig", "e_hmag"],
    "ks_m": ["ks_m", "kmag", "ksmag"],
    "ks_msigcom": ["ks_msigcom", "ks_cmsig", "e_kmag"],
    "tmass_ph_qual": ["ph_qual"],
    "tmass_cc_flg": ["cc_flg", "ccflg", "contamination_flag", "cc_flag"],
}
TMASS_XMATCH_ID = ["original_psc_source_id", "original_ext_source_id"]

# Required for the queries to mean anything; anything else degrades to NaN.
_WISE_REQUIRED = ("wise_designation", "w1mpro", "w2mpro")
_TMASS_REQUIRED = ("tmass_designation", "ks_m")


# ===========================================================================
# Column resolution
# ===========================================================================
@dataclass
class ColumnMap:
    """Resolved archive column names (output name -> real column)."""

    wise: dict = field(default_factory=dict)
    tmass: dict = field(default_factory=dict)
    tmass_id: str = "original_ext_source_id"
    missing_wise: list = field(default_factory=list)
    missing_tmass: list = field(default_factory=list)
    probed: bool = False

    def as_dict(self) -> dict:
        return {"wise": dict(self.wise), "tmass": dict(self.tmass),
                "tmass_id": self.tmass_id, "missing_wise": list(self.missing_wise),
                "missing_tmass": list(self.missing_tmass), "probed": self.probed}


def default_columns() -> ColumnMap:
    """The assumed (first-candidate) names — used offline and as the fallback."""
    # The 2MASS contamination flag is speculative on the Gaia mirror: it is only
    # selected when a live probe has seen it, never assumed.
    return ColumnMap(wise={k: v[0] for k, v in WISE_WANT.items()},
                     tmass={k: v[0] for k, v in TMASS_WANT.items() if k != "tmass_cc_flg"},
                     tmass_id=TMASS_XMATCH_ID[1], missing_tmass=["tmass_cc_flg"], probed=False)


def resolve_from_columns(want: dict, available) -> tuple[dict, list]:
    """Pure: pick the first candidate present in ``available`` for each want."""
    have = {str(c).lower() for c in available}
    out, missing = {}, []
    for logical, cands in want.items():
        hit = next((c for c in cands if c.lower() in have), None)
        if hit is None:
            missing.append(logical)
        else:
            out[logical] = hit
    return out, missing


def probe_columns(table: str, runner=None, timeout_s: float = 300.0) -> list[str]:
    """Live ``TOP 1 *`` probe of ``table``; returns its lowercased column names."""
    runner = runner or run_gaia_query
    df, rec = runner(f"SELECT TOP 1 * FROM {table}", label=f"probe {table}",
                     expect_rows=None, timeout_s=timeout_s)
    if rec["status"] == QUERY_FAILED:
        raise RuntimeError(f"column probe failed for {table}: {rec['error']}")
    return [str(c).lower() for c in df.columns]


def resolve_columns(runner=None, timeout_s: float = 300.0) -> ColumnMap:
    """Resolve every AllWISE / 2MASS column against the live archive.

    A table that cannot be probed keeps the assumed names (recorded as
    ``probed=False`` for that table in the ledger); a column that does not
    exist is dropped and its output column becomes NaN downstream.
    """
    cm = default_columns()
    try:
        cols = probe_columns(TABLES["wise"], runner, timeout_s)
        cm.wise, cm.missing_wise = resolve_from_columns(WISE_WANT, cols)
        cm.probed = True
    except Exception as exc:                                       # noqa: BLE001
        print(f"[baffle] AllWISE column probe failed ({exc!r}); assuming names", flush=True)
    try:
        cols = probe_columns(TABLES["tmass"], runner, timeout_s)
        cm.tmass, cm.missing_tmass = resolve_from_columns(TMASS_WANT, cols)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[baffle] 2MASS column probe failed ({exc!r}); assuming names", flush=True)
    try:
        cols = probe_columns(TABLES["xt"], runner, timeout_s)
        hit, _ = resolve_from_columns({"id": TMASS_XMATCH_ID}, cols)
        if hit.get("id"):
            cm.tmass_id = hit["id"]
    except Exception as exc:                                       # noqa: BLE001
        print(f"[baffle] 2MASS xmatch probe failed ({exc!r}); assuming "
              f"{cm.tmass_id}", flush=True)
    for req in _WISE_REQUIRED:
        if req not in cm.wise:
            raise RuntimeError(f"AllWISE mirror unusable: no {req} column")
    for req in _TMASS_REQUIRED:
        if req not in cm.tmass:
            raise RuntimeError(f"2MASS mirror unusable: no {req} column")
    print(f"[baffle] columns: wise missing {cm.missing_wise or 'none'}; "
          f"2mass missing {cm.missing_tmass or 'none'}; xt id {cm.tmass_id}", flush=True)
    return cm


# ===========================================================================
# Query builders — pure string functions
# ===========================================================================
def dec_bands(width_deg: float = 5.0) -> list[tuple[float, float]]:
    """Declination bands ``[lo, hi)`` covering the sky; the last is closed."""
    n = int(round(180.0 / float(width_deg)))
    edges = [-90.0 + i * 180.0 / n for i in range(n + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n)]


def bands_for_shard(bands: list, shard: int, n_shards: int) -> list[int]:
    """Round-robin assignment of band indices to a shard (balances the plane)."""
    n_shards = max(1, int(n_shards))
    return [i for i in range(len(bands)) if i % n_shards == int(shard) % n_shards]


def _dec_predicate(dec_lo: float, dec_hi: float) -> str:
    op_hi = "<=" if dec_hi >= 90.0 else "<"
    return f"g.dec >= {float(dec_lo):g} AND g.dec {op_hi} {float(dec_hi):g}"


def _ridx_predicate(ridx_lo, ridx_hi) -> str:
    s = ""
    if ridx_lo is not None:
        s += f"\n  AND g.random_index >= {int(ridx_lo)}"
    if ridx_hi is not None:
        s += f"\n  AND g.random_index < {int(ridx_hi)}"
    return s


def gaia_select(alias: str = "g") -> str:
    return ", ".join(f"{alias}.{c}" for c in GAIA_COLUMNS)


def _aliased(alias: str, mapping: dict, skip=()) -> str:
    return ", ".join(f"{alias}.{real} AS {out}" for out, real in mapping.items()
                     if out not in skip)


def wise_select(cols: ColumnMap) -> str:
    return _aliased("w", cols.wise)


def tmass_select(cols: ColumnMap) -> str:
    return _aliased("t", cols.tmass)


def xmatch_select() -> str:
    return _aliased("xw", XW_COLUMNS) + ", " + _aliased("xt", XT_COLUMNS)


def deficit_from(cols: ColumnMap) -> str:
    return f"""FROM {TABLES['gaia']} AS g
  JOIN {TABLES['xw']} AS xw ON xw.source_id = g.source_id
  JOIN {TABLES['wise']} AS w ON w.{cols.wise['wise_designation']} = xw.original_ext_source_id
  JOIN {TABLES['xt']} AS xt ON xt.source_id = g.source_id
  JOIN {TABLES['tmass']} AS t ON t.{cols.tmass['tmass_designation']} = xt.{cols.tmass_id}"""


def deficit_where(dec_lo: float, dec_hi: float, acq: dict, cols: ColumnMap,
                  ridx_lo=None, ridx_hi=None) -> str:
    """Shared verbatim by COUNT(*) and SELECT — that is what makes the count a ruler."""
    ks = f"t.{cols.tmass['ks_m']}"
    w1 = f"w.{cols.wise['w1mpro']}"
    w2 = f"w.{cols.wise['w2mpro']}"
    return (f"""WHERE {_dec_predicate(dec_lo, dec_hi)}
  AND g.phot_g_mean_mag < {float(acq['g_max']):g}
  AND ( ({ks} - {w1}) < {float(acq['pre_cut_w1']):g}
     OR ({ks} - {w2}) < {float(acq['pre_cut_w2']):g}
     OR g.random_index < {int(acq['locus_random_index_max'])} )"""
            + _ridx_predicate(ridx_lo, ridx_hi))


def build_deficit_query(dec_lo: float, dec_hi: float, acq: dict, cols: ColumnMap | None = None,
                        *, top: int = 2_000_000, ridx_lo=None, ridx_hi=None) -> str:
    cols = cols or default_columns()
    return (f"SELECT TOP {int(top)}\n  {gaia_select()},\n  {xmatch_select()},\n"
            f"  {wise_select(cols)},\n  {tmass_select(cols)}\n"
            f"{deficit_from(cols)}\n{deficit_where(dec_lo, dec_hi, acq, cols, ridx_lo, ridx_hi)}")


def build_deficit_count(dec_lo: float, dec_hi: float, acq: dict, cols: ColumnMap | None = None,
                        *, ridx_lo=None, ridx_hi=None) -> str:
    cols = cols or default_columns()
    return (f"SELECT COUNT(*) AS n\n{deficit_from(cols)}\n"
            f"{deficit_where(dec_lo, dec_hi, acq, cols, ridx_lo, ridx_hi)}")


def missing_from(cols: ColumnMap) -> str:
    return f"""FROM {TABLES['gaia']} AS g
  JOIN {TABLES['xt']} AS xt ON xt.source_id = g.source_id
  JOIN {TABLES['tmass']} AS t ON t.{cols.tmass['tmass_designation']} = xt.{cols.tmass_id}
  LEFT OUTER JOIN {TABLES['xn']} AS xn ON xn.source_id = g.source_id"""


def missing_where(dec_lo: float, dec_hi: float, acq: dict, ridx_lo=None, ridx_hi=None) -> str:
    return (f"""WHERE {_dec_predicate(dec_lo, dec_hi)}
  AND g.phot_g_mean_mag < {float(acq['g_max_missing']):g}
  AND xn.source_id IS NULL""" + _ridx_predicate(ridx_lo, ridx_hi))


def build_missing_query(dec_lo: float, dec_hi: float, acq: dict, cols: ColumnMap | None = None,
                        *, top: int = 2_000_000, ridx_lo=None, ridx_hi=None) -> str:
    cols = cols or default_columns()
    return (f"SELECT TOP {int(top)}\n  {gaia_select()},\n  {_aliased('xt', XT_COLUMNS)},\n"
            f"  {tmass_select(cols)}\n{missing_from(cols)}\n"
            f"{missing_where(dec_lo, dec_hi, acq, ridx_lo, ridx_hi)}")


def build_missing_count(dec_lo: float, dec_hi: float, acq: dict, cols: ColumnMap | None = None,
                        *, ridx_lo=None, ridx_hi=None) -> str:
    cols = cols or default_columns()
    return (f"SELECT COUNT(*) AS n\n{missing_from(cols)}\n"
            f"{missing_where(dec_lo, dec_hi, acq, ridx_lo, ridx_hi)}")


def denominator_from(cols: ColumnMap) -> str:
    return f"""FROM {TABLES['gaia']} AS g
  JOIN {TABLES['xt']} AS xt ON xt.source_id = g.source_id
  JOIN {TABLES['tmass']} AS t ON t.{cols.tmass['tmass_designation']} = xt.{cols.tmass_id}"""


def build_missing_denominator_query(dec_lo: float, dec_hi: float, acq: dict,
                                    cols: ColumnMap | None = None, *,
                                    babs_bin_deg: float = 10.0, g_bin_mag: float = 1.0,
                                    grouped: bool = True) -> str:
    """G < g_max_missing stars WITH a 2MASS match, binned in |b| and G.

    ``grouped=False`` is the plain ``COUNT(*)`` fallback if the archive rejects
    the GROUP BY (one number per band; the |b| structure is then lost).
    """
    cols = cols or default_columns()
    where = (f"WHERE {_dec_predicate(dec_lo, dec_hi)}\n"
             f"  AND g.phot_g_mean_mag < {float(acq['g_max_missing']):g}")
    if not grouped:
        return f"SELECT COUNT(*) AS n\n{denominator_from(cols)}\n{where}"
    return (f"SELECT FLOOR(ABS(g.b) / {float(babs_bin_deg):g}) AS babs_bin, "
            f"FLOOR(g.phot_g_mean_mag / {float(g_bin_mag):g}) AS g_bin, COUNT(*) AS n\n"
            f"{denominator_from(cols)}\n{where}\nGROUP BY babs_bin, g_bin")


def build_neighbours_query(ra: float, dec: float, radius_arcmin: float,
                           top: int = 5000) -> str:
    """Gaia cone (columns as the tracks, minus WISE) for the patch stage."""
    r = float(radius_arcmin) / 60.0
    return (f"SELECT TOP {int(top)} {gaia_select()}\nFROM {TABLES['gaia']} AS g\n"
            f"WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
            f"CIRCLE('ICRS', {float(ra):.7f}, {float(dec):.7f}, {r:.7f}))")


# ===========================================================================
# Transport layer (runner only), with the cooldown and a time-box
# ===========================================================================
_ASYNC_DEAD_SIGNATURES = ("cannot find result", "path does not exists",
                          "path does not exist", "quota exceeded",
                          "exceeds allowed quota")
_TRANSPORT_COOLDOWN_QUERIES = 25
_TRANSPORT_COOLDOWN: dict[str, int] = {}


def reset_transport_state() -> None:
    _TRANSPORT_COOLDOWN.clear()


class QueryTimeout(RuntimeError):
    """A transport did not answer within the time-box."""


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).lower() for c in df.columns})


def _t_astroquery_async(query: str) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    return _lower(Gaia.launch_job_async(query).get_results().to_pandas())


def _t_pyvo_async(query: str) -> pd.DataFrame:
    import pyvo

    return _lower(pyvo.dal.TAPService(GAIA_TAP).run_async(query).to_table().to_pandas())


def _t_astroquery_sync(query: str) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    return _lower(Gaia.launch_job(query).get_results().to_pandas())


def _t_pyvo_sync(query: str) -> pd.DataFrame:
    import pyvo

    return _lower(pyvo.dal.TAPService(GAIA_TAP).run_sync(query).to_table().to_pandas())


# Async first (no server row cap; results here are small).  Sync is the fallback
# and is assumed to truncate until the COUNT(*) proves otherwise.
GAIA_TRANSPORTS = (
    ("astroquery_async", _t_astroquery_async),
    ("pyvo_async", _t_pyvo_async),
    ("astroquery_sync", _t_astroquery_sync),
    ("pyvo_sync", _t_pyvo_sync),
)


def _is_dead_async(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _ASYNC_DEAD_SIGNATURES)


def _call_with_timeout(fn, arg, timeout_s: float | None):
    """Run ``fn(arg)`` on a daemon thread; raise :class:`QueryTimeout` if late.

    A daemon thread rather than an executor: a hung archive call must not keep
    the interpreter (and the runner job) alive at exit.
    """
    if not timeout_s or timeout_s <= 0:
        return fn(arg)
    box: dict = {}

    def _go():
        try:
            box["value"] = fn(arg)
        except BaseException as exc:                                # noqa: BLE001
            box["error"] = exc

    th = threading.Thread(target=_go, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise QueryTimeout(f"no answer within {timeout_s:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def run_gaia_query(query: str, *, label: str = "gaia", expect_rows: int | None = None,
                   timeout_s: float | None = 1500.0, retries_per_transport: int = 2,
                   base_sleep: float = 4.0, cooldown_queries: int | None = None
                   ) -> tuple[pd.DataFrame, dict]:
    """Execute ADQL; return ``(df, record)``.  Never raises.

    ``record["status"]`` is one of :data:`QUERY_OK`, :data:`QUERY_ZERO`,
    :data:`QUERY_TRUNCATED` (fewer rows than ``expect_rows``) or
    :data:`QUERY_FAILED` (no transport answered).
    """
    cooldown = _TRANSPORT_COOLDOWN_QUERIES if cooldown_queries is None else int(cooldown_queries)
    record: dict = {"label": label, "status": QUERY_FAILED, "n_rows": 0,
                    "expected_rows": expect_rows, "transport": None, "attempts": [],
                    "query": query.strip(), "error": None, "seconds": None}
    best: pd.DataFrame | None = None
    best_transport = None
    t0 = time.time()
    for name, fn in GAIA_TRANSPORTS:
        if _TRANSPORT_COOLDOWN.get(name, 0) > 0:
            _TRANSPORT_COOLDOWN[name] -= 1
            record["attempts"].append({"transport": name, "ok": False,
                                       "error": f"on cooldown ({_TRANSPORT_COOLDOWN[name]} left)"})
            continue
        for attempt in range(max(1, int(retries_per_transport))):
            ta = time.time()
            try:
                df = _call_with_timeout(fn, query, timeout_s)
            except Exception as exc:                                # noqa: BLE001
                err = repr(exc)
                record["attempts"].append({"transport": name, "ok": False, "error": err,
                                           "seconds": round(time.time() - ta, 1)})
                record["error"] = err
                print(f"[baffle] {label}: {name} attempt {attempt + 1} failed: {err[:300]}",
                      flush=True)
                if _is_dead_async(exc):
                    _TRANSPORT_COOLDOWN[name] = cooldown
                    print(f"[baffle] {label}: {name} on cooldown for {cooldown} queries "
                          "(ESA anonymous result-store quota exceeded)", flush=True)
                    break
                if isinstance(exc, QueryTimeout):
                    break       # a retry of a timed-out query is the same query
                time.sleep(base_sleep * (2 ** attempt))
                continue
            n = int(len(df))
            record["attempts"].append({"transport": name, "ok": True, "n_rows": n,
                                       "seconds": round(time.time() - ta, 1)})
            if best is None or n > len(best):
                best, best_transport = df, name
            if expect_rows is not None and n < int(expect_rows):
                print(f"[baffle] {label}: {name} returned {n} rows but COUNT(*) says "
                      f"{expect_rows} — truncated, trying the next transport", flush=True)
                break
            record.update(status=QUERY_OK if n else QUERY_ZERO, n_rows=n, transport=name,
                          seconds=round(time.time() - t0, 1))
            return df, record
    record["seconds"] = round(time.time() - t0, 1)
    if best is not None:
        record.update(status=QUERY_TRUNCATED, n_rows=int(len(best)), transport=best_transport)
        return best, record
    print(f"[baffle] {label}: QUERY_FAILED on every transport", flush=True)
    return pd.DataFrame(), record


def gaia_count(query: str, runner=None, *, label: str = "count",
               timeout_s: float | None = 900.0) -> tuple[int | None, dict]:
    """Run a ``SELECT COUNT(*)`` — the truncation ruler.  ``None`` if unreachable."""
    runner = runner or run_gaia_query
    df, rec = runner(query, label=label, expect_rows=None, timeout_s=timeout_s)
    if rec["status"] in (QUERY_FAILED, QUERY_TRUNCATED) or df is None or df.empty:
        return None, rec
    try:
        return int(df.iloc[0, 0]), rec
    except Exception:                                               # noqa: BLE001
        return None, rec


# ===========================================================================
# Ledger
# ===========================================================================
class AcquisitionLedger:
    """Per-chunk record of what the archive said, never collapsed prematurely."""

    def __init__(self, entries=None):
        self.entries: list[dict] = list(entries or [])

    def add(self, **entry) -> dict:
        entry.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.entries.append(entry)
        return entry

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": self.entries,
                                    "summary": summarise_ledger(self.entries)},
                                   indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> AcquisitionLedger:
        doc = json.loads(Path(path).read_text())
        return cls(doc.get("entries") if isinstance(doc, dict) else doc)


def summarise_ledger(entries: list, track: str | None = None) -> dict:
    """Collapse the ledger into the honest top-level acquisition verdict.

    Row totals come only from ``band_total`` entries (the leaves would double
    count).  ``QUERY_FAILED`` (archive not reached) and
    ``QUERY_RETURNED_ZERO_ROWS`` (reached; nothing matched) stay apart.
    """
    ents = [e for e in entries if track is None or e.get("track") == track]
    leaves = [e for e in ents if e.get("kind") == "leaf"]
    totals = [e for e in ents if e.get("kind") == "band_total"]
    by = {QUERY_OK: 0, QUERY_ZERO: 0, QUERY_FAILED: 0, QUERY_TRUNCATED: 0}
    for e in leaves:
        by[e.get("status", QUERY_FAILED)] = by.get(e.get("status", QUERY_FAILED), 0) + 1
    n_rows = sum(int(e.get("n_rows") or 0) for e in totals)
    n_expected = sum(int(e["expected_rows"]) for e in totals
                     if e.get("expected_rows") is not None)
    n_bands_failed = sum(1 for e in totals if e.get("status") == QUERY_FAILED)
    n_bands_partial = sum(1 for e in totals if e.get("status") == QUERY_TRUNCATED)
    failures = [{k: e.get(k) for k in ("chunk", "track", "status", "n_rows",
                                       "expected_rows", "error")}
                for e in leaves if e.get("status") in (QUERY_FAILED, QUERY_TRUNCATED)]
    if not leaves and not totals:
        verdict = "NO_QUERY_ATTEMPTED"
    elif totals and n_bands_failed == len(totals):
        verdict = "NO_DATA_REACHED"
    elif leaves and by[QUERY_FAILED] == len(leaves):
        verdict = "NO_DATA_REACHED"
    elif n_rows == 0 and by[QUERY_FAILED] == 0 and by[QUERY_TRUNCATED] == 0:
        verdict = "QUERY_RETURNED_ZERO_ROWS"
    elif by[QUERY_FAILED] or by[QUERY_TRUNCATED] or n_bands_failed or n_bands_partial:
        verdict = "PARTIAL_SAMPLE"
    else:
        verdict = "COMPLETE"
    return {
        "acquisition_verdict": verdict,
        "n_bands": len(totals), "n_bands_failed": n_bands_failed,
        "n_bands_partial": n_bands_partial,
        "n_leaves": len(leaves), "n_leaves_ok": by[QUERY_OK],
        "n_leaves_zero_rows": by[QUERY_ZERO], "n_leaves_failed": by[QUERY_FAILED],
        "n_leaves_truncated": by[QUERY_TRUNCATED],
        "n_rows_returned": n_rows,
        "n_rows_expected_by_count_star": n_expected or None,
        "completeness": round(n_rows / n_expected, 4) if n_expected else None,
        "failures": failures[:60],
        "note": ("QUERY_FAILED means the archive was not reached; "
                 "QUERY_RETURNED_ZERO_ROWS means it was reached and the cuts matched "
                 "nothing; QUERY_TRUNCATED means the server capped the result below "
                 "its own COUNT(*). These are never merged."),
    }


# ===========================================================================
# Chunked, checkpointed fetch
# ===========================================================================
@dataclass
class FetchContext:
    track: str
    acq: dict
    cols: ColumnMap
    chunk_dir: Path
    ledger: AcquisitionLedger
    runner: object = None
    top: int = 2_000_000
    max_depth: int = 6
    timeout_s: float = 1500.0
    count_timeout_s: float = 900.0

    def __post_init__(self):
        self.runner = self.runner or run_gaia_query
        self.chunk_dir = Path(self.chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)


def _leaf_path(ctx: FetchContext, band_idx: int, ridx_lo, ridx_hi) -> Path:
    lo = "all" if ridx_lo is None else int(ridx_lo)
    hi = "all" if ridx_hi is None else int(ridx_hi)
    return ctx.chunk_dir / f"{ctx.track}_b{band_idx:02d}_r{lo}_{hi}.parquet"


def _build(ctx: FetchContext, dec_lo, dec_hi, ridx_lo, ridx_hi, count: bool) -> str:
    if ctx.track == "deficit":
        f = build_deficit_count if count else build_deficit_query
    elif ctx.track == "missing":
        f = build_missing_count if count else build_missing_query
    else:
        raise ValueError(f"unknown track {ctx.track!r}")
    kw = dict(ridx_lo=ridx_lo, ridx_hi=ridx_hi)
    if not count:
        kw["top"] = ctx.top
    return f(dec_lo, dec_hi, ctx.acq, ctx.cols, **kw)


def annotate_chunk(df: pd.DataFrame, track: str, band_idx: int, dec_lo: float,
                   dec_hi: float, acq: dict) -> pd.DataFrame:
    """Add provenance and the ``is_locus_sample`` flag; fill absent columns with NaN."""
    out = df.copy()
    out["track"] = track
    out["dec_band_index"] = int(band_idx)
    out["dec_lo"] = float(dec_lo)
    out["dec_hi"] = float(dec_hi)
    if "random_index" in out.columns:
        ri = pd.to_numeric(out["random_index"], errors="coerce")
        out["is_locus_sample"] = (ri < int(acq["locus_random_index_max"])).fillna(False)
    else:
        out["is_locus_sample"] = False
    want = list(GAIA_COLUMNS) + list(XT_COLUMNS) + list(TMASS_WANT)
    if track == "deficit":
        want += list(XW_COLUMNS) + list(WISE_WANT)
    for c in want:
        if c not in out.columns:
            out[c] = np.nan
    return out


def _fetch_leaf(ctx: FetchContext, band_idx: int, dec_lo: float, dec_hi: float,
                ridx_lo, ridx_hi, n_expected, depth: int) -> pd.DataFrame:
    """Fetch one slice; split it on ``random_index`` if it failed or truncated."""
    tag = f"{ctx.track} band{band_idx:02d}[{dec_lo:g},{dec_hi:g})"
    if ridx_lo is not None or ridx_hi is not None:
        tag += f" ridx[{ridx_lo},{ridx_hi})"
    path = _leaf_path(ctx, band_idx, ridx_lo, ridx_hi)
    if path.exists():
        df = pd.read_parquet(path)
        ctx.ledger.add(chunk=tag, track=ctx.track, kind="leaf", status=QUERY_OK if len(df)
                       else QUERY_ZERO, n_rows=int(len(df)), expected_rows=n_expected,
                       depth=depth, reused=True, file=path.name)
        print(f"[baffle] {tag}: reused checkpoint ({len(df)} rows)", flush=True)
        return df

    q = _build(ctx, dec_lo, dec_hi, ridx_lo, ridx_hi, count=False)
    # Only the band-level COUNT(*) is exact; a half-estimate below it must not
    # make the transport ladder call a Poisson fluctuation a truncation.
    df, rec = ctx.runner(q, label=tag, expect_rows=n_expected if depth == 0 else None,
                         timeout_s=ctx.timeout_s)
    status = rec["status"]
    n = int(rec.get("n_rows") or 0)
    entry = dict(chunk=tag, track=ctx.track, kind="leaf", status=status, n_rows=n,
                 expected_rows=n_expected, depth=depth, transport=rec.get("transport"),
                 seconds=rec.get("seconds"), error=rec.get("error"),
                 ridx=[ridx_lo, ridx_hi], query=rec.get("query"))

    # A slice materially short of its expectation pays for an exact recount;
    # a slice that hit TOP is truncated by construction.
    if status in (QUERY_OK, QUERY_ZERO):
        if n >= int(ctx.top):
            status = entry["status"] = QUERY_TRUNCATED
            entry["note"] = f"hit TOP {ctx.top}"
        elif n_expected and n < 0.9 * n_expected and depth > 0:
            exact, _ = gaia_count(_build(ctx, dec_lo, dec_hi, ridx_lo, ridx_hi, count=True),
                                  ctx.runner, label=f"{tag} recount",
                                  timeout_s=ctx.count_timeout_s)
            if exact is not None:
                entry["expected_rows"] = n_expected = exact
                if n < exact:
                    status = entry["status"] = QUERY_TRUNCATED
        elif n_expected and n < n_expected and depth == 0:
            status = entry["status"] = QUERY_TRUNCATED

    if status in (QUERY_TRUNCATED, QUERY_FAILED) and depth < ctx.max_depth:
        lo_i = 0 if ridx_lo is None else int(ridx_lo)
        hi_i = int(ctx.acq.get("gaia_random_index_max", GAIA_RANDOM_INDEX_MAX)) \
            if ridx_hi is None else int(ridx_hi)
        mid = (lo_i + hi_i) // 2
        if mid > lo_i:
            entry["note"] = (f"{status} ({n} rows, expected {n_expected}); "
                             f"split on random_index at {mid}")
            entry["kind"] = "split"
            ctx.ledger.add(**entry)
            print(f"[baffle] {tag}: {status} — splitting on random_index at {mid}",
                  flush=True)
            half = (int(n_expected) // 2) if n_expected else None
            left = _fetch_leaf(ctx, band_idx, dec_lo, dec_hi, lo_i, mid, half, depth + 1)
            right = _fetch_leaf(ctx, band_idx, dec_lo, dec_hi, mid,
                                None if ridx_hi is None else hi_i, half, depth + 1)
            parts = [p for p in (left, right) if len(p)]
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if status in (QUERY_OK, QUERY_ZERO):
        df = annotate_chunk(df, ctx.track, band_idx, dec_lo, dec_hi, ctx.acq)
        df.to_parquet(path, index=False)
        entry["file"] = path.name
    ctx.ledger.add(**entry)
    print(f"[baffle] {tag}: {n} rows (expected {n_expected}, {status}, "
          f"via {rec.get('transport')}, {rec.get('seconds')} s)", flush=True)
    return df if status in (QUERY_OK, QUERY_ZERO) else pd.DataFrame()


def fetch_band(ctx: FetchContext, band_idx: int, dec_lo: float, dec_hi: float) -> pd.DataFrame:
    """One declination band: one COUNT(*) ruler, then the leaf tree."""
    tag = f"{ctx.track} band{band_idx:02d}[{dec_lo:g},{dec_hi:g})"
    n_expected, crec = gaia_count(_build(ctx, dec_lo, dec_hi, None, None, count=True),
                                  ctx.runner, label=f"{tag} count",
                                  timeout_s=ctx.count_timeout_s)
    ctx.ledger.add(chunk=f"{tag} count", track=ctx.track, kind="count",
                   status=crec["status"], n_rows=n_expected, error=crec.get("error"),
                   seconds=crec.get("seconds"))
    if n_expected is None:
        print(f"[baffle] {tag}: COUNT(*) unavailable — truncation detectable only "
              f"through TOP", flush=True)
    elif n_expected == 0:
        print(f"[baffle] {tag}: COUNT(*) = 0 — valid query, nothing matches", flush=True)
    else:
        print(f"[baffle] {tag}: COUNT(*) = {n_expected}", flush=True)
    df = _fetch_leaf(ctx, band_idx, dec_lo, dec_hi, None, None, n_expected, 0)
    got = int(len(df))
    leaves = [e for e in ctx.ledger.entries
              if e.get("kind") == "leaf" and e.get("track") == ctx.track
              and str(e.get("chunk", "")).startswith(tag)]
    n_failed = sum(1 for e in leaves if e["status"] == QUERY_FAILED)
    if leaves and n_failed == len(leaves):
        status = QUERY_FAILED
    elif n_failed or any(e["status"] == QUERY_TRUNCATED for e in leaves):
        status = QUERY_TRUNCATED
    elif n_expected is not None and got < n_expected:
        status = QUERY_TRUNCATED
    else:
        status = QUERY_OK if got else QUERY_ZERO
    ctx.ledger.add(chunk=tag, track=ctx.track, kind="band_total", status=status,
                   n_rows=got, expected_rows=n_expected, band_index=int(band_idx),
                   dec_lo=float(dec_lo), dec_hi=float(dec_hi), n_leaves=len(leaves),
                   n_leaves_failed=n_failed)
    return df


def fetch_missing_denominator(ctx: FetchContext, band_idx: int, dec_lo: float, dec_hi: float,
                              *, babs_bin_deg: float = 10.0, g_bin_mag: float = 1.0
                              ) -> pd.DataFrame:
    """Per-band denominator for the missing fraction, checkpointed like a leaf."""
    tag = f"missing-denominator band{band_idx:02d}[{dec_lo:g},{dec_hi:g})"
    path = ctx.chunk_dir / f"missing_denominator_b{band_idx:02d}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        ctx.ledger.add(chunk=tag, track="missing", kind="denominator", status=QUERY_OK,
                       n_rows=int(df["n"].sum()) if "n" in df else 0, reused=True)
        return df
    q = build_missing_denominator_query(dec_lo, dec_hi, ctx.acq, ctx.cols,
                                        babs_bin_deg=babs_bin_deg, g_bin_mag=g_bin_mag)
    df, rec = ctx.runner(q, label=tag, expect_rows=None, timeout_s=ctx.count_timeout_s)
    grouped = True
    if rec["status"] == QUERY_FAILED or not {"babs_bin", "g_bin", "n"} <= set(df.columns):
        grouped = False
        q = build_missing_denominator_query(dec_lo, dec_hi, ctx.acq, ctx.cols, grouped=False)
        df, rec = ctx.runner(q, label=f"{tag} (plain COUNT)", expect_rows=None,
                             timeout_s=ctx.count_timeout_s)
        if rec["status"] != QUERY_FAILED and len(df):
            df = pd.DataFrame({"babs_bin": [np.nan], "g_bin": [np.nan],
                               "n": [int(df.iloc[0, 0])]})
    if rec["status"] == QUERY_FAILED:
        ctx.ledger.add(chunk=tag, track="missing", kind="denominator", status=QUERY_FAILED,
                       n_rows=0, error=rec.get("error"), query=rec.get("query"))
        return pd.DataFrame(columns=["babs_bin", "g_bin", "n"])
    df = df[["babs_bin", "g_bin", "n"]].copy() if len(df) else \
        pd.DataFrame(columns=["babs_bin", "g_bin", "n"])
    df["dec_band_index"] = int(band_idx)
    df["dec_lo"], df["dec_hi"] = float(dec_lo), float(dec_hi)
    df["grouped"] = grouped
    df["babs_bin_deg"], df["g_bin_mag"] = float(babs_bin_deg), float(g_bin_mag)
    df.to_parquet(path, index=False)
    total = int(pd.to_numeric(df["n"], errors="coerce").fillna(0).sum()) if len(df) else 0
    ctx.ledger.add(chunk=tag, track="missing", kind="denominator",
                   status=QUERY_OK if total else QUERY_ZERO, n_rows=total,
                   grouped=grouped, transport=rec.get("transport"), seconds=rec.get("seconds"))
    print(f"[baffle] {tag}: {total} G<{ctx.acq['g_max_missing']:g} stars with 2MASS "
          f"({'binned' if grouped else 'plain count'})", flush=True)
    return df


def fetch_track(track: str, acq: dict, chunk_dir: Path, ledger: AcquisitionLedger, *,
                band_indices=None, cols: ColumnMap | None = None, runner=None,
                screen_missing: dict | None = None) -> pd.DataFrame:
    """Pull one track over the given bands (default all).  Never raises per band."""
    cols = cols or default_columns()
    bands = dec_bands(float(acq.get("dec_band_deg", 5.0)))
    idx = list(range(len(bands))) if band_indices is None else [int(i) for i in band_indices]
    ctx = FetchContext(track=track, acq=acq, cols=cols, chunk_dir=chunk_dir, ledger=ledger,
                       runner=runner, top=int(acq.get("top", 2_000_000)),
                       max_depth=int(acq.get("max_split_depth", 6)),
                       timeout_s=float(acq.get("query_timeout_s", 1500.0)),
                       count_timeout_s=float(acq.get("count_timeout_s", 900.0)))
    sm = screen_missing or {}
    frames = []
    for i in idx:
        lo, hi = bands[i]
        try:
            df = fetch_band(ctx, i, lo, hi)
            if len(df):
                frames.append(df)
        except Exception as exc:                                    # noqa: BLE001
            ledger.add(chunk=f"{track} band{i:02d}[{lo:g},{hi:g})", track=track,
                       kind="band_total", status=QUERY_FAILED, n_rows=0,
                       expected_rows=None, band_index=i, dec_lo=lo, dec_hi=hi,
                       error=f"uncaught: {exc!r}")
            print(f"[baffle] {track} band {i}: QUERY_FAILED (uncaught {exc!r}); continuing",
                  flush=True)
        if track == "missing":
            try:
                fetch_missing_denominator(ctx, i, lo, hi,
                                          babs_bin_deg=float(sm.get("babs_bin_deg", 10.0)),
                                          g_bin_mag=float(sm.get("g_bin_mag", 1.0)))
            except Exception as exc:                                # noqa: BLE001
                ledger.add(chunk=f"missing-denominator band{i:02d}", track="missing",
                           kind="denominator", status=QUERY_FAILED, n_rows=0,
                           error=f"uncaught: {exc!r}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("source_id").reset_index(drop=True)


def fetch_gaia_neighbours(ra: float, dec: float, radius_arcmin: float = 3.0, *,
                          runner=None, timeout_s: float = 300.0) -> pd.DataFrame:
    """Thin helper for the patch stage: Gaia cone, the track columns minus WISE."""
    runner = runner or run_gaia_query
    df, rec = runner(build_neighbours_query(ra, dec, radius_arcmin),
                     label=f"neighbours ({ra:.4f},{dec:.4f})", expect_rows=None,
                     timeout_s=timeout_s)
    if rec["status"] == QUERY_FAILED:
        raise RuntimeError(f"neighbour cone failed: {rec['error']}")
    return df


def load_chunks(chunk_dir: Path, track: str | None = None) -> pd.DataFrame:
    """Assemble the checkpointed leaves (dedup on source_id; denominators excluded)."""
    chunk_dir = Path(chunk_dir)
    files = sorted(p for p in chunk_dir.glob("*.parquet")
                   if not p.name.startswith("missing_denominator"))
    frames = []
    for p in files:
        if track is not None and not p.name.startswith(f"{track}_"):
            continue
        try:
            frames.append(pd.read_parquet(p))
        except Exception as exc:                                    # noqa: BLE001
            print(f"[baffle] unreadable chunk {p.name}: {exc!r}", flush=True)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "source_id" in df.columns:
        df = df.drop_duplicates("source_id")
    return df.reset_index(drop=True)


def load_denominators(chunk_dir: Path) -> pd.DataFrame:
    files = sorted(Path(chunk_dir).glob("missing_denominator_b*.parquet"))
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["babs_bin", "g_bin", "n", "dec_band_index"])


def slice_bounds(n_expected: int, target: int, lo: int = 0,
                 hi: int = GAIA_RANDOM_INDEX_MAX) -> list[tuple[int, int | None]]:
    """CENOTAPH's pre-split planner (kept for callers that know a count up front)."""
    k = max(1, math.ceil(int(n_expected) / max(1, int(target))))
    if k == 1:
        return [(None, None)]
    step = max(1, (hi - lo) // k)
    return [(lo + i * step, None if i == k - 1 else lo + (i + 1) * step) for i in range(k)]


__all__ = [
    "GAIA_COLUMNS", "GAIA_RANDOM_INDEX_MAX", "LOCUS_RANDOM_INDEX_MAX", "QUERY_FAILED",
    "QUERY_OK", "QUERY_TRUNCATED", "QUERY_ZERO", "TABLES", "AcquisitionLedger",
    "ColumnMap", "FetchContext", "annotate_chunk", "bands_for_shard",
    "build_deficit_count", "build_deficit_query", "build_missing_count",
    "build_missing_denominator_query", "build_missing_query", "build_neighbours_query",
    "dec_bands", "default_columns", "fetch_band", "fetch_gaia_neighbours",
    "fetch_missing_denominator", "fetch_track", "gaia_count", "load_chunks",
    "load_denominators", "probe_columns", "resolve_columns", "resolve_from_columns",
    "reset_transport_state", "run_gaia_query", "slice_bounds", "summarise_ledger",
]

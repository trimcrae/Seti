"""The IGNITION parent sample: Gaia DR3 x AllWISE, selected in the archive.

Runner-only where it queries; :func:`build_query` and :func:`select_parent`
are pure and tested offline.

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

import time as _time

import numpy as np
import pandas as pd

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

#: v_tan [km/s] = _K * mu [mas/yr] / parallax [mas]
_K = 4.740470446

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


def _where(conf: dict) -> str:
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    dc = {**DEFAULT_SAMPLE["dwarf_cut"], **(c.get("dwarf_cut") or {})}
    _sel, ac = _wise_cols(c)
    return (
        f"g.phot_g_mean_mag < {float(c['g_max'])}\n"
        f"  AND g.parallax > {float(c['parallax_min_mas'])}\n"
        f"  AND g.parallax_over_error > {float(c['parallax_over_error_min'])}\n"
        f"  AND ABS(g.b) > {float(c['abs_b_min_deg'])}\n"
        f"  AND g.ruwe < {float(c['ruwe_max'])}\n"
        f"  AND g.phot_variable_flag != 'VARIABLE'\n"
        f"  AND g.bp_rp > {float(c['bp_rp_min'])} AND g.bp_rp < {float(c['bp_rp_max'])}\n"
        f"  AND g.phot_g_mean_mag + 5 * LOG10(g.parallax) - 10 > "
        f"{float(dc['intercept'])} + {float(dc['slope'])} * (g.bp_rp - {float(dc['colour0'])})\n"
        f"  AND w.w1mpro - w.w2mpro < {float(c['w1w2_max'])}\n"
        f"  AND w.w1mpro - w.w2mpro > {float(c['w1w2_min'])}\n"
        f"  AND w.{ac['ext_flag']} = 0\n"
        f"  AND w.{ac['cc_flags']} = '0000'"
    )


def _from(conf: dict) -> str:
    _sel, ac = _wise_cols(conf)
    return ("FROM gaiadr3.gaia_source AS g\n"
            "  JOIN gaiadr3.allwise_best_neighbour AS xw ON xw.source_id = g.source_id\n"
            "  JOIN gaiadr1.allwise_original_valid AS w\n"
            f"    ON w.{ac['designation']} = xw.original_ext_source_id")


def build_query(conf: dict | None = None, *, plx_lo: float | None = None,
                plx_hi: float | None = None, field: dict | None = None,
                shard: int = 0, n_shards: int = 1, stride: int = 1,
                cap: int | None = None, count_only: bool = False) -> str:
    """ADQL for one parallax shell or one field cone, optionally subsampled.

    ``MOD(random_index, n_shards * stride) = shard`` selects a uniform
    ``1 / (n_shards * stride)`` of the parent.  ``count_only`` returns the
    ``COUNT(*)`` of the *unsubsampled* selection --- the denominator.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    sel, _ac = _wise_cols(c)
    where = _where(c)
    if plx_lo is not None:
        where += f"\n  AND g.parallax >= {float(plx_lo)}"
    if plx_hi is not None:
        where += f"\n  AND g.parallax < {float(plx_hi)}"
    if field is not None:
        where += (f"\n  AND 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
                  f"CIRCLE('ICRS', {float(field['ra'])}, {float(field['dec'])}, "
                  f"{float(field['radius_deg'])}))")
    if count_only:
        return f"SELECT COUNT(*) AS n\n{_from(c)}\nWHERE {where}"
    m = int(max(n_shards, 1)) * int(max(stride, 1))
    if m > 1:
        where += f"\n  AND MOD(g.random_index, {m}) = {int(shard)}"
    top = f"TOP {int(cap)} " if cap else ""
    return f"SELECT {top}{GAIA_COLS},\n  {sel}\n{_from(c)}\nWHERE {where}"


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
def gaia_query(adql: str, retries: int = 4, tag: str = "ignition") -> pd.DataFrame:
    """Robust Gaia ADQL: async with backoff, sync on the last try (herdsman's)."""
    from astroquery.gaia import Gaia

    last = None
    for attempt in range(retries):
        try:
            job = Gaia.launch_job(adql) if attempt == retries - 1 else Gaia.launch_job_async(adql)
            df = job.get_results().to_pandas()
            return df.rename(columns={c: c.lower() for c in df.columns})
        except Exception as exc:                       # noqa: BLE001
            last = exc
            print(f"[{tag}] Gaia attempt {attempt + 1}/{retries} failed: {exc!r}")
            _time.sleep(2 ** attempt)
    raise RuntimeError(f"Gaia query failed after {retries} attempts: {last!r}")


def _count(adql: str, query_fn, ledger: list, label: str) -> int | None:
    try:
        df = query_fn(adql)
        n = int(pd.to_numeric(df.iloc[0, 0])) if len(df) else 0
        ledger.append({"label": label, "status": "OK", "n": n, "query": adql[:1500]})
        return n
    except Exception as exc:                           # noqa: BLE001
        ledger.append({"label": label, "status": "QUERY_FAILED", "error": repr(exc),
                       "query": adql[:1500]})
        return None


def fetch_parent(conf: dict | None = None, *, mode: str | None = None, n_shards: int = 1,
                 cap_per_shard: int | None = None, query_fn=None,
                 fields: list[dict] | None = None) -> tuple[pd.DataFrame, dict]:
    """Pull the parent sample and report its denominator honestly.

    Returns ``(stars, report)``.  ``report["status"]`` is ``OK``,
    ``QUERY_RETURNED_ZERO_ROWS`` or ``QUERY_FAILED``; ``report["parent_count"]``
    is the archive's ``COUNT(*)`` of the full selection when it could be
    measured, and ``report["subsample_fraction"]`` the fraction actually pulled.
    """
    c = {**DEFAULT_SAMPLE, **(conf or {})}
    mode = mode or str(c.get("mode", "fields"))
    cap = int(cap_per_shard or c["cap_per_shard"])
    query_fn = query_fn or gaia_query
    fields = fields if fields is not None else list(c.get("fields") or [])
    ledger: list[dict] = []
    frames: list[pd.DataFrame] = []
    n_failed = 0
    parent_count: int | None = 0
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
            n_unit = _count(build_query(c, count_only=True, **u), query_fn, ledger,
                            f"count_{label}")
            if n_unit is None:
                parent_count = None
            elif parent_count is not None:
                parent_count += n_unit
        if mode != "fields" and n_unit:
            # Spread the run's total cap across shells in proportion to their size.
            share = max(int(total_cap / max(len(units), 1)), 1)
            stride = max(int(np.ceil(n_unit / share)), 1)
        q = build_query(c, stride=stride, cap=(cap * max(int(n_shards), 1)
                                               if mode == "fields" else None), **u)
        try:
            df = query_fn(q)
            df = df.rename(columns={x: str(x).lower() for x in df.columns})
            st = "OK" if len(df) else "QUERY_RETURNED_ZERO_ROWS"
            ledger.append({"label": label, "status": st, "n_rows": int(len(df)),
                           "stride": stride, "query": q[:2000]})
            per_unit.append({"unit": label, "n_parent": n_unit, "n_rows": int(len(df)),
                             "stride": stride, "fraction": (1.0 / stride)})
            if len(df):
                df["sample_unit"] = label
                df["subsample_stride"] = stride
                frames.append(df)
        except Exception as exc:                       # noqa: BLE001
            n_failed += 1
            ledger.append({"label": label, "status": "QUERY_FAILED", "error": repr(exc),
                           "query": q[:2000]})
            per_unit.append({"unit": label, "n_parent": n_unit, "n_rows": 0,
                             "status": "QUERY_FAILED"})
            print(f"[ignition] {label}: QUERY_FAILED {exc!r}")

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


__all__ = ["DEFAULT_SAMPLE", "GAIA_COLS", "GAIA_TAP", "build_query", "fetch_parent",
           "gaia_query", "parallax_shells", "select_parent"]

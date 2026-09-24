"""Runner-only archive calls for CONFLUENCE.

* ``fetch_accel_nss``   the whole Gaia DR3 ``nss_acceleration_astro`` table: the
                        ACCEL channel's parent and score, by definition.
* ``fetch_covariates``  Gaia DR3 covariates for every star in a joint parent:
                        G, BP-RP, RUWE, l, b, ecliptic latitude, scan coverage
                        (visibility periods, matched transits), crowding
                        indicators, variability / multiplicity flags and the
                        DR3 variability classifier's best class.
* ``fetch_density``     Gaia neighbour counts in 30" (budgeted; the anonymous
                        ESA queue refuses large positional joins, so this may
                        come back partial -- the report says how partial).
* ``fetch_simbad`` / ``fetch_vsx``   the known-class tags for tail members.

Nothing here invents a row: an unreachable service returns an empty frame
and a ledger entry that says ``NO_DATA_REACHED``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).lower() for c in df.columns})


def gaia_query(adql: str, upload: pd.DataFrame | None = None, name: str = "t",
               retries: int = 3, ledger: list | None = None, label: str = "") -> pd.DataFrame:
    """astroquery async with backoff, sync on the last attempt."""
    from astropy.table import Table
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    up = Table.from_pandas(upload.reset_index(drop=True)) if upload is not None else None
    last = None
    for a in range(retries):
        sync = a + 1 == retries
        t0 = time.time()
        try:
            fn = Gaia.launch_job if sync else Gaia.launch_job_async
            kw = {"upload_resource": up, "upload_table_name": name} if up is not None else {}
            df = _lower(fn(adql, **kw).get_results().to_pandas())
            if sync and len(df) in (2000, 3000, 50000):
                # the anonymous synchronous endpoint silently caps its rows;
                # a result exactly at a cap is a truncation, never a parent
                raise RuntimeError(f"sync result truncated at {len(df)} rows")
            if ledger is not None:
                ledger.append({"label": label, "attempt": a + 1, "ok": True, "rows": len(df),
                               "sync": sync, "s": round(time.time() - t0, 1)})
            return df
        except Exception as e:  # noqa: BLE001
            last = e
            if ledger is not None:
                ledger.append({"label": label, "attempt": a + 1, "ok": False, "sync": sync,
                               "error": repr(e)[:300], "s": round(time.time() - t0, 1)})
            time.sleep(5 * (a + 1))
    raise RuntimeError(f"gaia query failed ({label}): {last!r}")


def tap_query(url: str, adql: str, upload: pd.DataFrame | None = None, name: str = "t",
              retries: int = 3, ledger: list | None = None, label: str = "") -> pd.DataFrame:
    import pyvo
    import requests
    from astropy.table import Table

    class _S(requests.Session):   # pyvo's run_sync takes no timeout; bound every call
        def request(self, *a, **kw):
            kw.setdefault("timeout", 300)
            return super().request(*a, **kw)

    svc = pyvo.dal.TAPService(url, session=_S())
    last = None
    for a in range(retries):
        t0 = time.time()
        try:
            kw = {}
            if upload is not None:
                kw["uploads"] = {name: Table.from_pandas(upload.reset_index(drop=True))}
            df = _lower(svc.run_sync(adql, maxrec=2_000_000, **kw).to_table().to_pandas())
            if ledger is not None:
                ledger.append({"label": label, "attempt": a + 1, "ok": True, "rows": len(df),
                               "s": round(time.time() - t0, 1)})
            return df
        except Exception as e:  # noqa: BLE001
            last = e
            if ledger is not None:
                ledger.append({"label": label, "attempt": a + 1, "ok": False,
                               "error": repr(e)[:300], "s": round(time.time() - t0, 1)})
            time.sleep(5 * (a + 1))
    raise RuntimeError(f"tap query failed ({label}): {last!r}")


def _chunked(frame: pd.DataFrame, size: int, fn: Callable[[pd.DataFrame], pd.DataFrame],
             budget_s: float, ledger: list, label: str) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    parts, n_ok, n_fail, n_skip = [], 0, 0, 0
    for i in range(0, len(frame), size):
        if time.time() - t0 > budget_s:
            n_skip += 1
            continue
        try:
            parts.append(fn(frame.iloc[i:i + size]))
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            ledger.append({"label": label, "chunk": i // size, "error": repr(e)[:300]})
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    n = n_ok + n_fail + n_skip
    rep = {"chunks": n, "ok": n_ok, "failed": n_fail, "skipped_budget": n_skip,
           "status": "OK" if n_ok == n else ("NO_DATA_REACHED" if n_ok == 0 else "PARTIAL")}
    return out, rep


# ---------------------------------------------------------------------------


def fetch_accel_nss(ledger: list, n_chunks: int = 12) -> pd.DataFrame:
    """The whole table, in source_id ranges, with no join.

    Run 36006344044 asked for it joined to gaia_source in one async job: the
    ESA archive worked on it for 7,225 s twice and returned HTTP 500 ("results
    is null"), and the sync fallback came back with exactly 2,000 rows -- the
    anonymous sync cap, i.e. a truncation.  Positions are not needed (ACCEL
    rows are keyed on source_id), so there is no join, and the table goes in
    ranges so one slow slice costs one slice.
    """
    edges = np.linspace(0, 6.917528443525529e18, n_chunks + 1).astype(np.int64)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        q = ("SELECT source_id, significance, nss_solution_type "
             "FROM gaiadr3.nss_acceleration_astro "
             f"WHERE source_id >= {int(lo)} AND source_id < {int(hi)}")
        parts.append(gaia_query(q, ledger=ledger, label=f"nss_acceleration_astro[{lo}]"))
    return pd.concat(parts, ignore_index=True)


COV_QUERY = """
SELECT t.source_id, g.ra, g.dec, g.l, g.b, g.ecl_lat, g.phot_g_mean_mag, g.bp_rp,
       g.ruwe, g.parallax, g.parallax_over_error, g.visibility_periods_used,
       g.astrometric_matched_transits, g.ipd_frac_multi_peak, g.ipd_gof_harmonic_amplitude,
       g.phot_bp_rp_excess_factor, g.phot_variable_flag, g.non_single_star,
       g.classprob_dsc_combmod_quasar, g.classprob_dsc_combmod_galaxy,
       v.best_class_name AS vari_class, v.best_class_score AS vari_score
FROM tap_upload.t AS t
JOIN gaiadr3.gaia_source AS g ON g.source_id = t.source_id
LEFT OUTER JOIN gaiadr3.vari_classifier_result AS v ON v.source_id = t.source_id
"""


def fetch_covariates(source_ids, ledger: list, chunk: int = 5_000,
                     budget_s: float = 5400) -> tuple[pd.DataFrame, dict]:
    ids = pd.DataFrame({"source_id": np.unique(np.asarray(source_ids, dtype=np.int64))})
    return _chunked(ids, chunk, lambda f: gaia_query(COV_QUERY, f, ledger=ledger,
                                                     label="covariates"),
                    budget_s, ledger, "covariates")


def fetch_density(stars: pd.DataFrame, ledger: list, radius_arcsec: float = 30.0,
                  chunk: int = 2_000, budget_s: float = 2400) -> tuple[pd.DataFrame, dict]:
    r = radius_arcsec / 3600.0
    q = ("SELECT t.source_id, COUNT(*) AS n_nb FROM tap_upload.t AS t "
         "JOIN gaiadr3.gaia_source AS g ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
         f"CIRCLE('ICRS', t.ra, t.dec, {r:.7f})) GROUP BY t.source_id")
    up = stars[["source_id", "ra", "dec"]].dropna().copy()
    return _chunked(up, chunk, lambda f: gaia_query(q, f, ledger=ledger, retries=2,
                                                    label="density"),
                    budget_s, ledger, "density")


def fetch_simbad(stars: pd.DataFrame, ledger: list, radius_arcsec: float = 3.0,
                 chunk: int = 5_000, budget_s: float = 1800) -> tuple[pd.DataFrame, dict]:
    r = radius_arcsec / 3600.0
    q = ("SELECT t.skey, b.main_id, b.otype, "
         "DISTANCE(POINT('ICRS', t.ra, t.dec), POINT('ICRS', b.ra, b.dec)) * 3600.0 AS sep "
         "FROM TAP_UPLOAD.t AS t JOIN basic AS b ON 1 = CONTAINS(POINT('ICRS', b.ra, b.dec), "
         f"CIRCLE('ICRS', t.ra, t.dec, {r:.7f}))")
    # "key" is too close to reserved words in some ADQL parsers: upload as skey
    up = stars[["key", "ra", "dec"]].dropna().rename(columns={"key": "skey"})
    out, rep = _chunked(up, chunk, lambda f: tap_query(SIMBAD_TAP, q, f, ledger=ledger,
                                                       label="simbad"),
                        budget_s, ledger, "simbad")
    if len(out):
        out = out.rename(columns={"skey": "key"})
        out = out.sort_values("sep").drop_duplicates("key")
    return out, rep


def fetch_vsx(stars: pd.DataFrame, ledger: list, radius_arcsec: float = 3.0,
              chunk: int = 1_000, budget_s: float = 1800) -> tuple[pd.DataFrame, dict]:
    r = radius_arcsec / 3600.0
    q = ('SELECT t.skey, v."Name" AS vsx_name, v."Type" AS vsx_type, v."Period" AS vsx_period, '
         'DISTANCE(POINT(\'ICRS\', t.ra, t.dec), POINT(\'ICRS\', v."RAJ2000", v."DEJ2000")) '
         '* 3600.0 AS sep FROM TAP_UPLOAD.t AS t JOIN "B/vsx/vsx" AS v '
         'ON 1 = CONTAINS(POINT(\'ICRS\', v."RAJ2000", v."DEJ2000"), '
         f"CIRCLE('ICRS', t.ra, t.dec, {r:.7f}))")
    # "key" is too close to reserved words in some ADQL parsers: upload as skey
    up = stars[["key", "ra", "dec"]].dropna().rename(columns={"key": "skey"})
    out, rep = _chunked(up, chunk, lambda f: tap_query(VIZIER_TAP, q, f, ledger=ledger,
                                                       label="vsx"),
                        budget_s, ledger, "vsx")
    if len(out):
        out = out.rename(columns={"skey": "key"})
        out = out.sort_values("sep").drop_duplicates("key")
    return out, rep


#: VizieR tables that turn a mission id into a position.
ID_TABLES = {
    "kepler": ('"V/133/kic"', '"KIC"', '"RAJ2000"', '"DEJ2000"'),
    "tess": ('"IV/39/tic82"', '"TIC"', '"RAJ2000"', '"DEJ2000"'),
}


def fetch_id_positions(mission: str, ids, ledger: list, chunk: int = 1_000,
                       budget_s: float = 1200) -> tuple[pd.DataFrame, dict]:
    """KIC / TIC ids -> (id, ra, dec) through VizieR, by upload join on the id."""
    table, idc, rac, dec = ID_TABLES[mission]
    q = (f"SELECT t.id, v.{rac} AS ra, v.{dec} AS dec FROM TAP_UPLOAD.t AS t "
         f"JOIN {table} AS v ON v.{idc} = t.id")
    up = pd.DataFrame({"id": np.unique(np.asarray(ids, dtype=np.int64))})
    return _chunked(up, chunk, lambda f: tap_query(VIZIER_TAP, q, f, ledger=ledger,
                                                   label=f"ids_{mission}"),
                    budget_s, ledger, f"ids_{mission}")

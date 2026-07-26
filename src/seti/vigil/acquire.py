"""Runner-only archive access for VIGIL.

The sandbox has no archive egress (every call returns ``CONNECT tunnel failed,
response 403``); everything in this module runs on a GitHub Actions runner.

Two things this module is built around, both of them lessons paid for by earlier
runs in this repository:

**1. A failed query and an empty query are different events.**  Every call here
returns a :class:`QueryResult` carrying an explicit status --- ``OK``,
``QUERY_RETURNED_ZERO_ROWS``, ``QUERY_FAILED``, ``ROUTE_UNAVAILABLE`` --- the
query text, the row count, and, where the service supports it, an independent
``COUNT(*)``.  A previous run reported ``NO_DATA_REACHED`` while masking 703,555
fetched rows; another accepted 8,193 rows against a ``COUNT(*)`` of 199,572
because the server hit its 60 s limit and returned a partial result *as a
success*.  Both failure modes are checked for here, and the ledger is written
into ``summary.json``.

**2. Proper motion must be propagated to the survey epoch.**  NEOWISE spans
2014-2024; Gaia positions are at epoch 2016.0.  A 200 mas/yr star drifts ~1.6"
across the mission, which is comparable to the cone radii these queries use --- so
an unpropagated search silently loses exactly the nearby, well-characterised
stars a technosignature search most wants.  There is currently **no**
PM-propagated ``neowiser_p1bs_psd`` fetcher anywhere in this repository;
:func:`fetch_neowise_epochs` is it.  It propagates to the mission mid-epoch and
widens the cone by the residual sweep, then reports the drift it corrected.

The unTimely variable catalogue
-------------------------------
arXiv:2511.22071 is the channel's intended pre-selector (>8M W1 and >7M W2
mid-IR variables from WISE/NEOWISE unTimely coadds).  Its access route is *not*
assumed: :func:`probe_untimely` tries every plausible route --- VizieR TAP,
IRSA TAP, NOIRLab Astro Data Lab TAP (which already serves the parent unTimely
Catalog), and the paper's own data-availability statement --- and records what
each one returned.  Note the architecture does not depend on it: the catalogue
supplies *scale* (which sources to characterise), while the per-epoch NEOWISE
photometry supplies the modulation index, morphology and colour statistics that
no variability catalogue contains.  If the catalogue is unreachable the channel
degrades to a field-by-field NEOWISE sweep and says so.
"""

from __future__ import annotations

import time as _time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

# Epochs
GAIA_EPOCH = 2016.0
NEOWISE_MID_EPOCH = 2019.0      # NEOWISE-R ran 2013.9 - 2024.6
NEOWISE_START = 2013.9
NEOWISE_END = 2024.6
ALLWISE_EPOCH = 2010.5

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
DATALAB_TAP = "https://datalab.noirlab.edu/tap"

UNTIMELY_HINTS = ("untimely", "unwise", "neowise", "mid-infrared variab",
                  "mid-IR variab", "wise variab")

# VizieR's TAP parser rejects ``LOWER(col)`` inside a WHERE clause --- the first
# probe run got back a bare ADQL syntax error from it while IRSA and NOIRLab
# accepted the same query.  So the case-insensitive match is done by spelling the
# variants out rather than by a function call, which every ADQL 2.0 parser takes.
def _case_variants(h: str) -> list[str]:
    out = {h.lower(), h.upper(), h.title(), h.capitalize()}
    if h.lower() == "untimely":
        out.add("unTimely")            # the catalogue's own capitalisation
    return sorted(out)


# --------------------------------------------------------------------------
@dataclass
class QueryResult:
    """One archive call, with enough context to audit it after the fact."""

    label: str
    service: str
    status: str                 # OK | QUERY_RETURNED_ZERO_ROWS | QUERY_FAILED | ROUTE_UNAVAILABLE
    n_rows: int = 0
    count_star: int | None = None
    truncated: bool | None = None
    query: str = ""
    error: str = ""
    elapsed_s: float = float("nan")
    n_rows_raw: int | None = None        # before any post-query quality cleaning
    n_rows_cleaned_out: int | None = None
    data: pd.DataFrame | None = field(default=None, repr=False)

    def to_ledger(self) -> dict:
        d = asdict(self)
        d.pop("data", None)
        d["query"] = self.query[:2000]
        return d


def _tap(url: str):
    import pyvo
    return pyvo.dal.TAPService(url)


def run_tap(url: str, adql: str, label: str, count_query: str | None = None,
            retries: int = 3, async_first: bool = True) -> QueryResult:
    """Execute one ADQL query and classify the outcome honestly.

    When ``count_query`` is supplied the returned row count is compared against
    it and ``truncated`` is set.  A partial result returned as a success is the
    single most dangerous archive failure mode in this repository's history, and
    the only defence is to ask the server how many rows there should have been.
    """
    t0 = _time.monotonic()
    count_star = None
    if count_query:
        try:
            svc = _tap(url)
            cdf = svc.search(count_query).to_table().to_pandas()
            if len(cdf):
                count_star = int(np.asarray(cdf.iloc[0])[0])
        except Exception as exc:                       # noqa: BLE001
            print(f"[vigil] {label}: COUNT(*) unavailable: {exc!r}")

    last = ""
    for attempt in range(retries):
        try:
            svc = _tap(url)
            if async_first and attempt < retries - 1:
                job = svc.submit_job(adql)
                job.run()
                job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=1800)
                if job.phase != "COMPLETED":
                    raise RuntimeError(f"async job phase={job.phase}")
                df = job.fetch_result().to_table().to_pandas()
            else:
                df = svc.search(adql).to_table().to_pandas()
            n = int(len(df))
            trunc = (count_star is not None and n < count_star)
            status = "OK" if n else "QUERY_RETURNED_ZERO_ROWS"
            return QueryResult(label=label, service=url, status=status, n_rows=n,
                               count_star=count_star, truncated=trunc, query=adql,
                               elapsed_s=_time.monotonic() - t0, data=df)
        except Exception as exc:                       # noqa: BLE001
            last = repr(exc)
            print(f"[vigil] {label} attempt {attempt + 1}/{retries} failed: {last}")
            _time.sleep(3.0 * (attempt + 1))
    return QueryResult(label=label, service=url, status="QUERY_FAILED", query=adql,
                       error=last, count_star=count_star,
                       elapsed_s=_time.monotonic() - t0)


# --------------------------------------------------------------------------
# unTimely variable catalogue: find it, do not assume it
# --------------------------------------------------------------------------
def probe_untimely(hints=UNTIMELY_HINTS) -> dict:
    """Try every plausible route to the unTimely mid-IR variable catalogue.

    Returns a dict with one entry per route, each carrying the query, the status
    and any matching table names.  The verdict field says whether *any* route
    reached it --- and, if not, says so as a transport failure rather than as a
    scientific statement.
    """
    out: dict = {"routes": [], "tables_found": [], "reachable": False}
    clauses: list[str] = []
    for h in hints:
        for v in _case_variants(h):
            clauses.append(f"description LIKE '%{v}%'")
            if " " not in h:
                clauses.append(f"table_name LIKE '%{v}%'")
    like = " OR ".join(clauses)
    for url in (VIZIER_TAP, IRSA_TAP, DATALAB_TAP):
        q = f"SELECT table_name, description FROM TAP_SCHEMA.tables WHERE {like}"
        r = run_tap(url, q, label=f"untimely_schema@{url}", retries=2,
                    async_first=False)
        out["routes"].append(r.to_ledger())
        if r.status == "OK" and r.data is not None:
            for _, row in r.data.iterrows():
                name = str(row.get("table_name", ""))
                out["tables_found"].append({"service": url, "table": name,
                                            "description": str(row.get("description", ""))[:300]})
    out["reachable"] = bool(out["tables_found"])
    n_failed = sum(1 for r in out["routes"] if r["status"] == "QUERY_FAILED")
    out["n_routes"] = len(out["routes"])
    out["n_routes_failed"] = n_failed
    if out["reachable"]:
        out["verdict"] = "CATALOGUE_TABLE_DISCOVERED"
    elif n_failed == len(out["routes"]):
        out["verdict"] = "ALL_TAP_ROUTES_FAILED"
    elif n_failed:
        # Some routes answered and some errored: a partial search cannot support
        # "the catalogue is not there".  Say which it is.
        out["verdict"] = "NOT_FOUND_BUT_SEARCH_INCOMPLETE"
    else:
        out["verdict"] = "CATALOGUE_NOT_FOUND_ON_ANY_TAP_ROUTE"
    return out


def fetch_untimely_variables(table: str, service: str, ra: float, dec: float,
                             radius_deg: float, max_rows: int = 200_000,
                             extra_where: str = "") -> QueryResult:
    """Pull mid-IR variables from a discovered unTimely table over one sky cone."""
    where = (f"1 = CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))")
    if extra_where:
        where += f" AND ({extra_where})"
    q = f"SELECT TOP {max_rows} * FROM {table} WHERE {where}"
    cq = f"SELECT COUNT(*) FROM {table} WHERE {where}"
    return run_tap(service, q, label=f"untimely_cone@{table}", count_query=cq)


# --------------------------------------------------------------------------
# Proper motion
# --------------------------------------------------------------------------
def propagate_pm(ra, dec, pmra_mas_yr, pmdec_mas_yr, from_epoch: float,
                 to_epoch: float):
    """Move positions from ``from_epoch`` to ``to_epoch``.

    ``pmra`` is the ``mu_alpha*`` convention (already includes ``cos(dec)``), as
    Gaia reports it.  Non-finite proper motions are treated as zero *and the
    caller is expected to widen the cone* --- silently dropping them would remove
    faint stars for which Gaia has no five-parameter solution.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    pmra = np.nan_to_num(np.asarray(pmra_mas_yr, dtype=float))
    pmdec = np.nan_to_num(np.asarray(pmdec_mas_yr, dtype=float))
    dt = float(to_epoch - from_epoch)
    cosd = np.cos(np.radians(dec))
    cosd = np.where(np.abs(cosd) < 1e-6, 1e-6, cosd)
    ra_new = ra + (pmra * dt / 3.6e6) / cosd
    dec_new = dec + pmdec * dt / 3.6e6
    return ra_new, dec_new


def pm_sweep_arcsec(pmra_mas_yr, pmdec_mas_yr,
                    span_yr: float = NEOWISE_END - NEOWISE_START) -> float:
    """Total angular distance a star traverses over the NEOWISE mission."""
    mu = float(np.hypot(np.nan_to_num(pmra_mas_yr), np.nan_to_num(pmdec_mas_yr)))
    return mu * span_yr / 1000.0


# --------------------------------------------------------------------------
# NEOWISE per-epoch photometry --- the characteriser
# --------------------------------------------------------------------------
_NEOWISE_COLS = ("mjd", "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro",
                 "qual_frame", "cc_flags", "ph_qual", "saa_sep", "moon_masked",
                 "w1rchi2", "w2rchi2")


def fetch_neowise_epochs(ra: float, dec: float, pmra: float = 0.0, pmdec: float = 0.0,
                         radius_arcsec: float = 2.5, from_epoch: float = GAIA_EPOCH,
                         retries: int = 3) -> QueryResult:
    """PM-propagated NEOWISE single-exposure W1/W2 photometry for one star.

    The cone is centred on the *mission mid-epoch* position and widened by half
    the mission-long proper-motion sweep, so the whole track is inside the search
    radius rather than only its Gaia-epoch end.  The correction applied is
    reported in the result label so a null on a high-PM star can be distinguished
    from a null caused by the bug this guards against.
    """
    ra_m, dec_m = propagate_pm(ra, dec, pmra, pmdec, from_epoch, NEOWISE_MID_EPOCH)
    sweep = pm_sweep_arcsec(pmra, pmdec)
    rad = float(radius_arcsec + 0.5 * sweep)
    cols = ", ".join(_NEOWISE_COLS)
    q = (f"SELECT {cols} FROM neowiser_p1bs_psd WHERE "
         f"1 = CONTAINS(POINT('ICRS', ra, dec), "
         f"CIRCLE('ICRS', {float(ra_m):.7f}, {float(dec_m):.7f}, {rad / 3600.0:.9f}))")
    cq = (f"SELECT COUNT(*) FROM neowiser_p1bs_psd WHERE "
          f"1 = CONTAINS(POINT('ICRS', ra, dec), "
          f"CIRCLE('ICRS', {float(ra_m):.7f}, {float(dec_m):.7f}, {rad / 3600.0:.9f}))")
    res = run_tap(IRSA_TAP, q, label=f"neowise_epochs_pm{sweep:.2f}as",
                  count_query=cq, retries=retries, async_first=False)
    if res.data is not None and len(res.data):
        # Keep the RAW count next to the cleaned one.  Without this the ledger
        # reads like a truncation ("27 rows against COUNT(*) = 32") when what
        # actually happened is that frame-quality cleaning removed five rows.
        res.n_rows_raw = int(len(res.data))
        res.data = clean_neowise(res.data)
        res.n_rows = int(len(res.data))
        res.n_rows_cleaned_out = res.n_rows_raw - res.n_rows
        if res.n_rows == 0:
            res.status = "QUERY_RETURNED_ZERO_ROWS_AFTER_QUALITY_CUTS"
    return res


def fetch_neowise_field(ra: float, dec: float, radius_deg: float = 0.4,
                        w1_max: float = 14.5, max_rows: int = 3_000_000,
                        retries: int = 3) -> QueryResult:
    """**One** NEOWISE query for a whole field, instead of one per star.

    The first run measured ~90 s for a single-star cone (COUNT(*) plus the query),
    which caps a 400-star field at a few dozen stars inside any sane wall-clock
    budget --- and scale is this programme's second priority after novelty.  A
    single cone over the field returns every exposure of every source in it, and
    :func:`seti.vigil.run.group_neowise_by_star` then assigns rows to stars with a
    KD-tree.  One query replaces four hundred.

    ``w1_max`` keeps the row count bounded: the channel's stars are Gaia G < 15,
    which are comfortably brighter than this in W1, so the cut costs nothing real
    while removing the faint-source bulk that dominates the row count.  The
    ``COUNT(*)`` comparison is retained because a silently truncated field query
    would now cost the whole field rather than one star.
    """
    cols = "ra, dec, " + ", ".join(_NEOWISE_COLS)
    where = (f"1 = CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {float(ra):.7f}, {float(dec):.7f}, {float(radius_deg):.6f})) "
             f"AND w1mpro < {w1_max}")
    q = f"SELECT TOP {max_rows} {cols} FROM neowiser_p1bs_psd WHERE {where}"
    cq = f"SELECT COUNT(*) FROM neowiser_p1bs_psd WHERE {where}"
    res = run_tap(IRSA_TAP, q, label="neowise_field", count_query=cq,
                  retries=retries, async_first=True)
    if res.data is not None and len(res.data):
        res.n_rows_raw = int(len(res.data))
        res.data = clean_neowise(res.data)
        res.n_rows = int(len(res.data))
        res.n_rows_cleaned_out = res.n_rows_raw - res.n_rows
        if res.n_rows == 0:
            res.status = "QUERY_RETURNED_ZERO_ROWS_AFTER_QUALITY_CUTS"
    return res


def clean_neowise(df: pd.DataFrame) -> pd.DataFrame:
    """Standard NEOWISE frame-quality cleaning, applied identically in both bands."""
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    if "qual_frame" in d:
        d = d[pd.to_numeric(d["qual_frame"], errors="coerce").fillna(0) > 0]
    if "cc_flags" in d:
        d = d[d["cc_flags"].astype(str).str.startswith("00")]
    if "saa_sep" in d:
        sep = pd.to_numeric(d["saa_sep"], errors="coerce")
        d = d[(sep.isna()) | (sep > 0)]
    if "moon_masked" in d:
        mm = d["moon_masked"].astype(str)
        d = d[~mm.str.startswith("1")]
    for b in ("w1", "w2"):
        m, e = f"{b}mpro", f"{b}sigmpro"
        if m in d:
            d[m] = pd.to_numeric(d[m], errors="coerce")
        if e in d:
            d[e] = pd.to_numeric(d[e], errors="coerce")
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------
# The stellar sample and its ancillary photometry
# --------------------------------------------------------------------------
def fetch_gaia_field(ra: float, dec: float, radius_deg: float = 0.5,
                     g_max: float = 16.0, plx_over_err_min: float = 10.0,
                     max_rows: int = 60_000) -> QueryResult:
    """Gaia DR3 stars in one field, with the astrometry the channel needs.

    Selected on parallax significance because the channel's AGN veto rests on
    astrometry, not colour: a ~350 K circumstellar shroud has W1-W2 = 3.2 and
    sits *inside* the Stern/Assef AGN colour box, so colour cannot separate the
    two.  A significant parallax and proper motion can.
    """
    where = f"""
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                           CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
          AND phot_g_mean_mag < {g_max}
          AND parallax_over_error > {plx_over_err_min}
    """
    cols = ("source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, ruwe, "
            "phot_g_mean_mag, bp_rp, phot_variable_flag, non_single_star, "
            "phot_g_mean_flux_over_error, astrometric_excess_noise, teff_gspphot")
    # Gaia's ADQL takes TOP, not LIMIT.  The row cap is therefore visible in the
    # query text, and the COUNT(*) below is what makes a silent truncation
    # detectable rather than a "successful" partial result.
    q = f"SELECT TOP {max_rows} {cols} FROM gaiadr3.gaia_source {where}"
    cq = f"SELECT COUNT(*) AS n FROM gaiadr3.gaia_source {where}"
    try:
        from astroquery.gaia import Gaia
        t0 = _time.monotonic()
        count_star = None
        try:
            count_star = int(Gaia.launch_job(cq).get_results().to_pandas().iloc[0, 0])
        except Exception as exc:                       # noqa: BLE001
            print(f"[vigil] Gaia COUNT(*) unavailable: {exc!r}")
        job = Gaia.launch_job_async(q)
        df = job.get_results().to_pandas()
        df.columns = [c.lower() for c in df.columns]
        n = int(len(df))
        return QueryResult(label="gaia_field", service="gaia", n_rows=n,
                           status="OK" if n else "QUERY_RETURNED_ZERO_ROWS",
                           count_star=count_star,
                           truncated=(count_star is not None and n < count_star),
                           query=q, elapsed_s=_time.monotonic() - t0, data=df)
    except Exception as exc:                           # noqa: BLE001
        return QueryResult(label="gaia_field", service="gaia", status="QUERY_FAILED",
                           query=q, error=repr(exc))


def fetch_allwise_for(positions: pd.DataFrame, radius_arcsec: float = 3.0,
                      from_epoch: float = GAIA_EPOCH) -> pd.DataFrame:
    """AllWISE W1-W4 + 2MASS JHKs for a shortlist, PM-propagated to 2010.5.

    W3/W4 are pulled *only* to run the cirrus veto and to bound the excess: the
    ledger is explicit that AllWISE W4 is unreliable for faint stars and that a
    W4-only signal is cirrus, so W4 is used to reject, never to detect.
    """
    rows = []
    for _, p in positions.iterrows():
        ra_w, dec_w = propagate_pm(float(p["ra"]), float(p["dec"]),
                                   float(p.get("pmra", 0.0) or 0.0),
                                   float(p.get("pmdec", 0.0) or 0.0),
                                   from_epoch, ALLWISE_EPOCH)
        q = f"""
            SELECT designation, ra, dec, w1mpro, w1sigmpro, w2mpro, w2sigmpro,
                   w3mpro, w3sigmpro, w4mpro, w4sigmpro, cc_flags, ext_flg, ph_qual,
                   w1sat, w2sat, j_m_2mass, j_msig_2mass, h_m_2mass, h_msig_2mass,
                   k_m_2mass, k_msig_2mass
            FROM allwise_p3as_psd
            WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                               CIRCLE('ICRS', {float(ra_w):.7f}, {float(dec_w):.7f},
                                      {radius_arcsec / 3600.0:.9f}))
        """
        r = run_tap(IRSA_TAP, q, label="allwise_cone", retries=2, async_first=False)
        if r.status != "OK" or r.data is None or not len(r.data):
            rows.append({"source_id": p.get("source_id"), "allwise_ok": False,
                         "allwise_status": r.status})
            continue
        d = r.data.copy()
        d.columns = [c.lower() for c in d.columns]
        rec = {"source_id": p.get("source_id"), "allwise_ok": True,
               "allwise_status": "OK", "n_allwise_in_cone": int(len(d))}
        rec.update({k: (float(v) if pd.notna(v) and not isinstance(v, str) else v)
                    for k, v in d.iloc[0].to_dict().items()})
        rows.append(rec)
    return pd.DataFrame(rows)


def fetch_optical_constancy(ra: float, dec: float, radius_arcsec: float = 2.0) -> dict:
    """Optical variability from ZTF g+r --- the YSO/dipper/AGN veto.

    Returns a dict that always states whether the optical was *measured*.  A star
    with no ZTF coverage is not an optically constant star, and the discriminator
    treats the two differently.
    """
    out = {"optical_measured": False, "optical_fvar": float("nan"),
           "optical_n_epochs": 0, "optical_source": "ztf", "optical_status": ""}
    try:
        from ..dimming.acquire import fetch_ztf_lightcurve
    except Exception as exc:                           # noqa: BLE001
        out["optical_status"] = f"ZTF_CLIENT_UNAVAILABLE:{exc!r}"
        return out
    fvars, n_tot = [], 0
    for band in ("g", "r"):
        try:
            lc = fetch_ztf_lightcurve(ra, dec, band=band, radius_arcsec=radius_arcsec)
        except Exception as exc:                       # noqa: BLE001
            out["optical_status"] = f"QUERY_FAILED:{exc!r}"
            continue
        if lc is None or not len(lc):
            continue
        m = pd.to_numeric(lc["mag"], errors="coerce").to_numpy()
        e = pd.to_numeric(lc["magerr"], errors="coerce").to_numpy()
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        if ok.sum() < 20:
            continue
        m, e = m[ok], e[ok]
        n_tot += int(m.size)
        f = 10.0 ** (-0.4 * (m - np.median(m)))
        fe = 0.4 * np.log(10.0) * f * e
        nxs = (np.var(f, ddof=1) - np.mean(fe**2)) / np.mean(f) ** 2
        fvars.append(float(np.sqrt(nxs)) if nxs > 0 else 0.0)
    if fvars:
        out.update({"optical_measured": True, "optical_fvar": float(max(fvars)),
                    "optical_n_epochs": n_tot, "optical_status": "OK"})
    elif not out["optical_status"]:
        out["optical_status"] = "QUERY_RETURNED_ZERO_ROWS"
    return out


def fetch_simbad_type(ra: float, dec: float, radius_arcsec: float = 5.0) -> str:
    """SIMBAD object type, for the YSO/AGB/AGN literature veto."""
    try:
        from astroquery.simbad import Simbad
        s = Simbad()
        s.add_votable_fields("otype")
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        t = s.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius_arcsec * u.arcsec)
        if t is None or not len(t):
            return ""
        for key in ("OTYPE", "otype", "main_type"):
            if key in t.colnames:
                return str(t[key][0])
        return ""
    except Exception as exc:                           # noqa: BLE001
        print(f"[vigil] SIMBAD lookup failed: {exc!r}")
        return ""


__all__ = ["ALLWISE_EPOCH", "DATALAB_TAP", "GAIA_EPOCH", "IRSA_TAP",
           "NEOWISE_END", "NEOWISE_MID_EPOCH", "NEOWISE_START", "UNTIMELY_HINTS",
           "VIZIER_TAP", "QueryResult", "clean_neowise", "fetch_allwise_for",
           "fetch_gaia_field", "fetch_neowise_epochs", "fetch_neowise_field",
           "fetch_optical_constancy",
           "fetch_simbad_type", "fetch_untimely_variables", "pm_sweep_arcsec",
           "probe_untimely", "propagate_pm", "run_tap"]

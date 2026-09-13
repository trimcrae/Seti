"""Runner-only archive access for GROWTH, plus the (pure) Kepler <-> TESS join.

The sandbox has no archive egress, so nothing network-bound here runs in a
test; every fetch takes an injectable ``query_fn`` / ``gaia_fn`` so a failed or
empty archive can be simulated offline, and the join is a pure function over
three DataFrames.

Endpoints
---------
NASA Exoplanet Archive TAP **sync** endpoint (reached from the runner by the
necrofrontier probe, entries ``exoarchive_koi_depth`` / ``exoarchive_toi_depth``
/ ``exoarchive_tess_kepler_xmatch``):

``https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=<ADQL>&format=csv``

* ``cumulative`` --- the KOI cumulative table: ``koi_depth`` [ppm], ``koi_ror``,
  ``koi_impact``, ``koi_duration`` [h], ``koi_dor`` (a/R*), the four
  ``koi_fpflag_*`` and the dispositions.  The 2009--2013 epoch.
* ``toi`` --- the TESS Objects of Interest table: ``pl_trandep`` [ppm],
  ``pl_trandurh`` [h], ``tid`` (TIC id), ``tfopwg_disp``.  The 2018--2026 epoch.
* ``ps`` --- Planetary Systems: ``pl_name`` <-> ``hostname`` <-> ``tic_id`` for
  rows with ``disc_facility LIKE '%Kepler%'``; ``pl_trandep`` there is in
  **percent**, converted to ppm on read.  **MEASURED (run 34787801172):**
  ``ps.tic_id`` is a *string* spelled ``"TIC 122298563"``, not an integer --- a
  bare ``pd.to_numeric`` on it is all-NaN, which silently emptied the whole
  name -> TIC map and drove ``joined_by_route["tic_id"]`` to 0.
  :func:`parse_tic_id` is the only place that column is read.

The KOI -> TOI join therefore runs four routes, each counted on its own
(:data:`JOIN_ROUTES`): ``name_planet`` (KOI ``kepler_name`` -> ``ps.pl_name``),
``name_host`` (the KOI's host, taken from its own ``kepler_name`` or inherited
from a sibling KOI on the same ``kepid``, -> ``ps.hostname``), ``position_tic``
(the KOI position -> the nearest ``ps`` row with a TIC id, no names needed) and
``position_period`` (KOI position -> TOI position directly).  The first three
end at a TIC id and are finished by the period; the fourth needs no catalogue
crossmatch at all.

Gaia DR3 neighbours (dilution term) come from the Gaia TAP.  The upload-table
crossmatch is tried first (chunked, retried, every attempt's exception text
recorded); when the ESA archive refuses or times out the anonymous upload ---
which is what it did on run 34787801172 for every target --- the chunk falls
back to **per-target sync cones through pyvo** against :data:`GAIA_TAP`, the
pattern ``src/seti/baffle/acquire.py`` uses.  A target neither route reached
keeps ``neighbours_status = QUERY_FAILED`` and is reported as ``not_checked``
rather than pretending it is isolated.

Two facts are never collapsed: ``QUERY_FAILED`` (the service did not answer or
errored) and ``QUERY_RETURNED_ZERO_ROWS`` (it answered, with nothing).
"""

from __future__ import annotations

import io
import json
import re
import time as _time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

EXOARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
#: ESA Gaia TAP (`verify` in ``config/growth.yaml``: ``gaia.tap_url``).  The
#: per-target cone fallback goes here through pyvo, exactly as
#: ``src/seti/baffle/acquire.py`` reaches Gaia.
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
USER_AGENT = "Seti-growth/1.0 (mailto:trimcrae@gmail.com)"

STATUS_OK = "OK"
STATUS_FAILED = "QUERY_FAILED"
STATUS_ZERO = "QUERY_RETURNED_ZERO_ROWS"

#: Columns pulled from each table.  A name the server rejects 400s the whole
#: query, so the lists are the documented column names as of the 2026-09 probe.
KOI_COLUMNS: tuple[str, ...] = (
    "kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition",
    "koi_period", "koi_period_err1", "koi_depth", "koi_depth_err1", "koi_depth_err2",
    "koi_ror", "koi_ror_err1", "koi_ror_err2", "koi_impact", "koi_impact_err1",
    "koi_impact_err2", "koi_duration", "koi_duration_err1", "koi_duration_err2",
    "koi_ingress", "koi_dor", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
    "ra", "dec", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_tce_delivname", "koi_count",
)
TOI_COLUMNS: tuple[str, ...] = (
    "toi", "tid", "tfopwg_disp", "pl_orbper", "pl_trandep", "pl_trandeperr1",
    "pl_trandeperr2", "pl_trandurh", "pl_trandurherr1", "pl_trandurherr2", "pl_rade",
    "st_teff", "st_logg", "st_rad", "st_tmag", "ra", "dec",
)
PS_COLUMNS: tuple[str, ...] = (
    "pl_name", "hostname", "tic_id", "default_flag", "pl_ratror", "pl_trandep",
    "pl_orbper", "disc_facility", "pl_refname", "ra", "dec",
)
#: The ``ps`` predicate.  Kept here (and in ``config/growth.yaml``,
#: ``archive.ps_where``) so the recorded ADQL and the code-side re-filter in
#: :func:`fetch_ps_kepler` can never disagree about what was asked for.
PS_WHERE_KEPLER = "disc_facility like '%Kepler%'"

#: Join routes, in the order they are tried.  Each is counted separately in
#: ``acquire.json["join"]["joined_by_route"]``.
JOIN_ROUTES: tuple[str, ...] = ("name_planet", "name_host", "position_tic", "position_period")
#: The routes that end at a TIC id (the period then picks the planet).
TIC_ROUTES: tuple[str, ...] = ("name_planet", "name_host", "position_tic")


# ---------------------------------------------------------------------------
# Acquisition log (same discipline as seti.metronome.acquire.AcquisitionLog)
# ---------------------------------------------------------------------------
@dataclass
class AcquisitionLog:
    stages: list[dict] = field(default_factory=list)
    prefix: str = "growth/acquire"

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
            print(f"[growth/acquire] {label} attempt {attempt + 1}/{retries} failed: {exc!r}")
            if attempt + 1 < retries:
                _time.sleep(base_sleep * (attempt + 1))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


def _default_transport(url: str, timeout: float = 600.0) -> str:
    import requests  # noqa: PLC0415  runner-only

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def build_url(adql: str, *, url: str = EXOARCHIVE_TAP, fmt: str = "csv") -> str:
    q = urllib.parse.urlencode({"query": adql, "format": fmt}, quote_via=urllib.parse.quote)
    return f"{url}?{q}"


def parse_tap_csv(text: str) -> pd.DataFrame:
    """CSV body -> DataFrame; an archive error page raises instead of parsing."""
    head = (text or "").lstrip()[:400]
    if not head:
        return pd.DataFrame()
    if head.startswith("ERROR") or head.lower().startswith("<!doctype") or head.startswith("<html"):
        raise RuntimeError(f"archive returned an error body: {head[:200]!r}")
    return pd.read_csv(io.StringIO(text))


def tap_sync(adql: str, *, url: str = EXOARCHIVE_TAP, retries: int = 3,
             transport=None, timeout: float = 600.0) -> pd.DataFrame:
    """One ADQL query against the Exoplanet Archive sync endpoint, retried."""
    transport = transport or _default_transport
    full = build_url(adql, url=url)
    return _retry(lambda: parse_tap_csv(transport(full, timeout)), retries=retries,
                  label="exoarchive TAP")


def table_query(table: str, columns, where: str | None = None) -> str:
    cols = ",".join(columns)
    q = f"select {cols} from {table}"
    if where:
        q += f" where {where}"
    return q


# ---------------------------------------------------------------------------
# The three tables
# ---------------------------------------------------------------------------
def _fetch(name: str, adql: str, *, query_fn, log: AcquisitionLog) -> tuple[pd.DataFrame, str]:
    try:
        df = query_fn(adql)
    except Exception as exc:                              # noqa: BLE001
        log.record(f"fetch_{name}", adql, error=repr(exc))
        return pd.DataFrame(), STATUS_FAILED
    if df is None:
        log.record(f"fetch_{name}", adql, error="query_fn returned None")
        return pd.DataFrame(), STATUS_FAILED
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    log.record(f"fetch_{name}", adql, rows=int(len(df)))
    return df, (STATUS_OK if len(df) else STATUS_ZERO)


def fetch_koi(*, query_fn=None, log: AcquisitionLog | None = None,
              columns=KOI_COLUMNS) -> tuple[pd.DataFrame, str]:
    log = log or AcquisitionLog()
    query_fn = query_fn or tap_sync
    return _fetch("koi_cumulative", table_query("cumulative", columns), query_fn=query_fn, log=log)


def fetch_toi(*, query_fn=None, log: AcquisitionLog | None = None,
              columns=TOI_COLUMNS) -> tuple[pd.DataFrame, str]:
    log = log or AcquisitionLog()
    query_fn = query_fn or tap_sync
    return _fetch("toi", table_query("toi", columns), query_fn=query_fn, log=log)


def fetch_ps_kepler(*, query_fn=None, log: AcquisitionLog | None = None,
                    columns=PS_COLUMNS, where: str = PS_WHERE_KEPLER,
                    require_kepler: bool = True) -> tuple[pd.DataFrame, str]:
    """``ps`` rows for Kepler-discovered planets; ``pl_trandep`` % -> ppm.

    The ``disc_facility`` predicate is sent to the server *and* re-applied here:
    run 34787801172 returned 28,592 rows for it, and a WHERE the server had
    quietly dropped would look exactly the same from outside.  The re-filter
    makes that observable --- ``n_ps_rows_not_kepler_discovered`` in the join
    report is the number of rows the server sent that the predicate excludes.
    """
    log = log or AcquisitionLog()
    query_fn = query_fn or tap_sync
    df, status = _fetch("ps_kepler", table_query("ps", columns, where),
                        query_fn=query_fn, log=log)
    if require_kepler and len(df) and "disc_facility" in df:
        keep = df["disc_facility"].astype(str).str.contains("kepler", case=False, na=False)
        if not bool(keep.all()):
            print(f"[growth/acquire] ps: dropping {int((~keep).sum())} row(s) whose "
                  "disc_facility does not contain 'Kepler' (the server did not apply the WHERE)")
        df = df[keep].reset_index(drop=True)
    if len(df) and "pl_trandep" in df:
        # The Planetary Systems table quotes transit depth in PERCENT; the KOI
        # and TOI tables quote ppm.  Convert here, once, and say so in the column.
        df["pl_trandep_ppm"] = pd.to_numeric(df["pl_trandep"], errors="coerce") * 1.0e4
    return df, status


# ---------------------------------------------------------------------------
# The join (pure)
# ---------------------------------------------------------------------------
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
#: A trailing planet letter after the host designation: "Kepler-22 b", "Kepler-22b".
_PLANET_LETTER = re.compile(r"(?<=\d)\s*[b-z]\s*$", re.IGNORECASE)
#: ``ps.tic_id`` is a string: an optional "TIC" (any case, optional separator)
#: in front of the integer.  ``"TIC 122298563"``, ``"tic122298563"``, ``122298563``.
_TIC_VALUE = re.compile(r"^\s*(?:tic\s*[-_]?\s*)?(\d+)\s*$", re.IGNORECASE)


def _norm_name(s) -> str:
    """Strip, collapse internal whitespace, case-fold, drop every non-alphanumeric.

    ``"Kepler-22 b"``, ``" kepler-22  B "`` and ``"Kepler 22b"`` all become
    ``"kepler22b"``; a missing / NaN / blank name becomes ``""`` and never
    matches anything.  The 2026-09-13 run joined nothing partly because names
    were compared with the punctuation and spacing the two tables happen to
    use, which is not a property either archive promises.
    """
    if s is None or s != s:
        return ""
    return _NON_ALNUM.sub("", " ".join(str(s).split()).strip().casefold())


def host_of_planet_name(planet_name) -> str:
    """Normalised HOST of a planet name: ``"Kepler-22 b"`` -> ``"kepler22"``.

    A name with no trailing planet letter (already a host) normalises unchanged,
    so this is safe to apply to ``ps.hostname`` as well.
    """
    if planet_name is None or planet_name != planet_name:
        return ""
    t = " ".join(str(planet_name).split()).strip()
    return _norm_name(_PLANET_LETTER.sub("", t)) if t else ""


def parse_tic_id(value) -> float:
    """``ps.tic_id`` -> a float TIC id, ``nan`` when it is absent or unparsable.

    **This is the 2026-09-13 defect.**  The Exoplanet Archive spells this column
    as text --- ``"TIC 122298563"`` --- so ``pd.to_numeric(ps["tic_id"])`` is
    all-NaN, the ``dropna`` that followed emptied the name -> TIC map, and every
    name route reported zero (``koi_with_tic_id: 0`` with 28,592 ``ps`` rows in
    hand).  Numeric input is passed through so a numeric column still works.
    """
    if value is None or value != value:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if np.isfinite(v) and v > 0 else float("nan")
    m = _TIC_VALUE.match(str(value))
    return float(m.group(1)) if m else float("nan")


def period_match(p_kepler: float, p_tess: float, *, tol: float = 1e-3,
                 alias_max: int = 4) -> tuple[bool, float, str]:
    """Same period within ``tol`` or an integer alias ``n`` / ``1/n`` (n <= alias_max).

    Returns ``(matched, ratio, alias)`` with ``alias`` in ``{"1", "n", "1/n"}``
    formatted as the integer, e.g. ``"2"`` when TESS found twice the Kepler
    period (alternate transits) or ``"1/2"`` when it found half of it.
    """
    if not (np.isfinite(p_kepler) and np.isfinite(p_tess)) or p_kepler <= 0 or p_tess <= 0:
        return False, float("nan"), ""
    r = float(p_tess / p_kepler)
    if abs(r - 1.0) < tol:
        return True, r, "1"
    for n in range(2, int(alias_max) + 1):
        if abs(r - n) < n * tol:
            return True, r, str(n)
        if abs(r - 1.0 / n) < tol / n:
            return True, r, f"1/{n}"
    return False, r, ""


def _unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    v1 = _unit_vectors([ra1], [dec1])[0]
    v2 = _unit_vectors([ra2], [dec2])[0]
    d = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)) * 3600.0)


def _empty_ps_maps() -> dict:
    return {"name_to_tic": {}, "host_to_tic": {}, "ref_by_name": {},
            "pos": pd.DataFrame(columns=["_tic", "ra", "dec"])}


def prepare_ps(ps: pd.DataFrame | None, rep: dict) -> dict:
    """The three ``ps`` lookups, plus every diagnostic the 2026-09-13 run lacked.

    ``name_to_tic`` and ``host_to_tic`` prefer the ``default_flag = 1`` row of a
    planet (that is the archive's own adopted solution); the remaining rows are
    kept only as *depth references* --- ``ps_n_refs`` and the min/max published
    depth for the planet, which say how heterogeneous the literature already is
    for it.  ``pos`` is one row per star for the name-free ``position_tic``
    route.  Every count lands in ``rep`` so a zero is traceable next time.
    """
    maps = _empty_ps_maps()
    rep["n_ps_rows"] = int(len(ps)) if ps is not None else 0
    for k in ("n_ps_rows_kepler", "n_ps_rows_not_kepler_discovered", "n_ps_default_rows",
              "n_ps_tic_id_parsed", "n_ps_tic_id_unparsed", "n_ps_distinct_planet_names",
              "n_ps_distinct_hosts", "n_ps_stars_with_tic_id"):
        rep[k] = 0
    rep["ps_tic_id_samples"] = []
    if ps is None or not len(ps):
        return maps

    p = ps.copy()
    p.columns = [str(c).strip().lower() for c in p.columns]
    if "disc_facility" in p:
        keep = p["disc_facility"].astype(str).str.contains("kepler", case=False, na=False)
        rep["n_ps_rows_not_kepler_discovered"] = int((~keep).sum())
        p = p[keep]
    else:
        # The column was not selected, so "is this really the Kepler subset?"
        # cannot be answered here; say so rather than implying it was checked.
        rep["n_ps_rows_not_kepler_discovered"] = None
    rep["n_ps_rows_kepler"] = int(len(p))
    if not len(p):
        return maps

    raw_tic = p["tic_id"] if "tic_id" in p else pd.Series([None] * len(p), index=p.index)
    p = p.assign(_tic=[parse_tic_id(v) for v in raw_tic])
    rep["ps_tic_id_samples"] = [str(v) for v in raw_tic.dropna().astype(str).head(3)]
    rep["n_ps_tic_id_parsed"] = int(np.isfinite(p["_tic"]).sum())
    rep["n_ps_tic_id_unparsed"] = int(len(p) - rep["n_ps_tic_id_parsed"])
    p["_name"] = [_norm_name(v) for v in (p["pl_name"] if "pl_name" in p
                                          else [None] * len(p))]
    p["_host"] = [host_of_planet_name(v) for v in (p["hostname"] if "hostname" in p
                                                   else p.get("pl_name", [None] * len(p)))]
    rep["n_ps_distinct_planet_names"] = int(p.loc[p["_name"] != "", "_name"].nunique())
    rep["n_ps_distinct_hosts"] = int(p.loc[p["_host"] != "", "_host"].nunique())
    dflag = (pd.to_numeric(p["default_flag"], errors="coerce").fillna(0.0)
             if "default_flag" in p else pd.Series(0.0, index=p.index))
    p["_default"] = (dflag == 1).astype(int)
    rep["n_ps_default_rows"] = int(p["_default"].sum())

    # Depth references: every ps row for a planet, default or not (the brief's
    # "keep the rest for the depth references").
    dep = pd.to_numeric(p.get("pl_trandep_ppm", pd.Series(np.nan, index=p.index)),
                        errors="coerce")
    for nm, grp in p.assign(_dep=dep).groupby("_name", sort=False):
        if not nm:
            continue
        d = grp["_dep"].to_numpy(dtype=float)
        finite = d[np.isfinite(d)]
        head = grp.sort_values("_default", ascending=False, kind="stable").iloc[0]
        maps["ref_by_name"][nm] = {
            "ps_pl_ratror": float(pd.to_numeric(head.get("pl_ratror"), errors="coerce")),
            "ps_pl_trandep_ppm": float(pd.to_numeric(head.get("pl_trandep_ppm"),
                                                     errors="coerce")),
            "ps_refname": head.get("pl_refname"),
            "ps_n_refs": int(len(grp)),
            "ps_trandep_ppm_min": float(finite.min()) if len(finite) else float("nan"),
            "ps_trandep_ppm_max": float(finite.max()) if len(finite) else float("nan")}

    withtic = p[np.isfinite(p["_tic"])].sort_values("_default", ascending=False, kind="stable")
    for _, r in withtic.iterrows():
        if r["_name"]:
            maps["name_to_tic"].setdefault(r["_name"], int(r["_tic"]))
        if r["_host"]:
            maps["host_to_tic"].setdefault(r["_host"], int(r["_tic"]))
    rep["n_ps_stars_with_tic_id"] = int(len(maps["host_to_tic"]))

    if {"ra", "dec"} <= set(withtic.columns):
        pos = withtic.assign(ra=pd.to_numeric(withtic["ra"], errors="coerce"),
                             dec=pd.to_numeric(withtic["dec"], errors="coerce"))
        pos = pos.dropna(subset=["ra", "dec"]).drop_duplicates("_tic")
        maps["pos"] = pos[["_tic", "ra", "dec"]].reset_index(drop=True)
    return maps


def _koi_name_keys(koi: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Normalised planet name and HOST name for every KOI row.

    Only *confirmed* KOIs carry a ``kepler_name``, so a KOI with an empty one
    inherits its host from a sibling KOI on the same ``kepid`` that does have a
    name --- that is what lets an unnamed candidate reach a TIC id through its
    star, which is the whole point of the ``name_host`` route.
    """
    src = koi["kepler_name"] if "kepler_name" in koi else pd.Series([None] * len(koi),
                                                                   index=koi.index)
    names = [_norm_name(v) for v in src]
    hosts = [host_of_planet_name(v) for v in src]
    if "kepid" in koi:
        by_kepid: dict[str, str] = {}
        for kid, h in zip(koi["kepid"], hosts, strict=True):
            key = str(kid)
            if h and key not in by_kepid:
                by_kepid[key] = h
        hosts = [h or by_kepid.get(str(kid), "")
                 for h, kid in zip(hosts, koi["kepid"], strict=True)]
    return names, hosts


def _match_positions(ra_a, dec_a, ra_b, dec_b, radius_arcsec: float) -> list[list[int]]:
    """Indices of every ``b`` within ``radius_arcsec`` of each ``a`` (a KD-tree)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    tree = cKDTree(_unit_vectors(ra_b, dec_b))
    chord = 2.0 * np.sin(np.deg2rad(float(radius_arcsec) / 3600.0) / 2.0)
    return tree.query_ball_point(_unit_vectors(ra_a, dec_a), r=chord)


def join_kepler_tess(koi: pd.DataFrame, toi: pd.DataFrame, ps: pd.DataFrame | None = None, *,
                     pos_radius_arcsec: float = 2.0, period_tol: float = 1e-3,
                     alias_max: int = 4, routes=JOIN_ROUTES) -> tuple[pd.DataFrame, dict]:
    """Every KOI that a TOI re-detects, with the route that joined it.

    Four routes, tried in order and counted separately
    (``rep["joined_by_route"]``):

    ``name_planet``     ``cumulative.kepler_name`` == ``ps.pl_name`` (both
                        normalised: whitespace collapsed, case-folded, every
                        non-alphanumeric removed) gives the TIC id.
    ``name_host``       the KOI's host --- from its own ``kepler_name``, or
                        inherited from a sibling KOI on the same ``kepid`` ---
                        matched to ``ps.hostname``.  This is the route for the
                        ~6,800 KOIs with **no** ``kepler_name`` at all.
    ``position_tic``    the KOI position matched to the nearest ``ps`` row with
                        a TIC id, within ``pos_radius_arcsec``.  No names.
    ``position_period`` the nearest TOI within ``pos_radius_arcsec`` whose
                        period matches --- no catalogue crossmatch at all.

    The first three end at a TIC id; ``toi.tid == tic_id`` then gives the TOIs
    on that star and the period picks the planet (an integer alias ``n`` /
    ``1/n``, n <= ``alias_max``, counts as a match and is recorded as a veto).
    A TOI is used at most once.  ``routes`` restricts which are run
    (``config/growth.yaml``, ``join.routes``).
    """
    routes = tuple(r for r in JOIN_ROUTES if r in set(routes or JOIN_ROUTES))
    rep: dict = {"n_koi": int(len(koi)), "n_toi": int(len(toi)),
                 "n_ps_kepler_rows": int(len(ps)) if ps is not None else 0,
                 "routes_enabled": list(routes),
                 "joined_by_route": dict.fromkeys(JOIN_ROUTES, 0),
                 "koi_with_tic_by_route": dict.fromkeys(TIC_ROUTES, 0),
                 "tic_matched_no_period_match": 0, "koi_with_tic_id": 0,
                 "position_matched_no_period_match": 0}
    maps = prepare_ps(ps, rep)
    if not len(koi) or not len(toi):
        rep.update(n_joined=0, n_toi_used=0, n_koi_with_tess_counterpart=0,
                   fraction_koi_with_tess_counterpart=0.0,
                   join_statement=_join_statement(rep))
        return pd.DataFrame(), rep

    # A fresh RangeIndex on both: every lookup below is ``.loc`` by label, and a
    # duplicated index in an input table would make those return frames.
    koi = koi.reset_index(drop=True)
    toi = toi.reset_index(drop=True)
    for c in ("koi_period", "ra", "dec"):
        koi[c] = pd.to_numeric(koi.get(c), errors="coerce")
    for c in ("pl_orbper", "ra", "dec", "tid"):
        toi[c] = pd.to_numeric(toi.get(c), errors="coerce")

    names, hosts = _koi_name_keys(koi)
    koi["_name"], koi["_host"] = names, hosts
    rep["n_koi_with_kepler_name"] = int(sum(1 for n in names if n))
    rep["n_koi_with_host_name"] = int(sum(1 for h in hosts if h))

    # --- resolve a TIC id per KOI, by the first route that answers ----------
    tic = np.full(len(koi), np.nan)
    tic_route = [""] * len(koi)
    if "name_planet" in routes and maps["name_to_tic"]:
        for i, n in enumerate(names):
            t = maps["name_to_tic"].get(n)
            if t is not None:
                tic[i], tic_route[i] = float(t), "name_planet"
    if "name_host" in routes and maps["host_to_tic"]:
        for i, h in enumerate(hosts):
            if not tic_route[i]:
                t = maps["host_to_tic"].get(h)
                if t is not None:
                    tic[i], tic_route[i] = float(t), "name_host"
    pos = maps["pos"]
    if "position_tic" in routes and len(pos):
        need = [i for i in range(len(koi)) if not tic_route[i]]
        sub = koi.iloc[need]
        ok = np.isfinite(sub["ra"].to_numpy(float)) & np.isfinite(sub["dec"].to_numpy(float))
        need = [i for i, keep in zip(need, ok, strict=True) if keep]
        if need:
            sub = koi.iloc[need]
            hits = _match_positions(sub["ra"], sub["dec"], pos["ra"], pos["dec"],
                                    pos_radius_arcsec)
            for i, idxs in zip(need, hits, strict=True):
                if not idxs:
                    continue
                ra_i, dec_i = float(koi["ra"].iloc[i]), float(koi["dec"].iloc[i])
                j = min(idxs, key=lambda j: _sep_arcsec(ra_i, dec_i, pos["ra"].iloc[j],
                                                        pos["dec"].iloc[j]))
                tic[i], tic_route[i] = float(pos["_tic"].iloc[j]), "position_tic"
    koi["tic_id_ps"] = tic
    koi["tic_route"] = tic_route
    rep["koi_with_tic_id"] = int(np.isfinite(tic).sum())
    for r in TIC_ROUTES:
        rep["koi_with_tic_by_route"][r] = int(sum(1 for x in tic_route if x == r))

    toi_by_tid: dict[int, list[int]] = {}
    for i, t in toi["tid"].items():
        if np.isfinite(t):
            toi_by_tid.setdefault(int(t), []).append(i)
    # The ceiling, stated separately from the yield: of the TIC ids the KOIs
    # resolved, how many TESS ever made a TOI for.  A low join with
    # ``n_koi_tics_with_a_toi`` also low means TESS has no candidate on that
    # star (nothing to fix); a low join with it HIGH means the period test is
    # what is rejecting, which is a different problem.
    koi_tics = {int(t) for t in tic if np.isfinite(t)}
    rep["n_distinct_koi_tics"] = int(len(koi_tics))
    rep["n_koi_tics_with_a_toi"] = int(len(koi_tics & set(toi_by_tid)))
    rep["n_toi_on_koi_tics"] = int(sum(len(toi_by_tid.get(t, [])) for t in koi_tics))

    used_toi: set[int] = set()
    rows: list[dict] = []
    hidden = ("_name", "_host")

    def _emit(ki, ti, route, ratio, alias, sep):
        # Only ``ra``/``dec`` collide between the two tables; every other KOI
        # column is ``koi_*`` / ``kepid`` / ``kepoi_name`` / ``kepler_name`` and
        # every TOI column is ``toi`` / ``tid`` / ``tfopwg_disp`` / ``pl_*`` / ``st_*``.
        k, t = koi.loc[ki], toi.loc[ti]
        rec = {("ra_kepler" if c == "ra" else "dec_kepler" if c == "dec" else c): k[c]
               for c in koi.columns if c not in hidden}
        rec.update({("ra_tess" if c == "ra" else "dec_tess" if c == "dec" else c): t[c]
                    for c in toi.columns})
        rec.update({"join_route": route, "period_ratio": ratio, "period_alias": alias,
                    "join_sep_arcsec": sep,
                    "tic_id": k.get("tic_id_ps") if route in TIC_ROUTES else t.get("tid")})
        rec.update(maps["ref_by_name"].get(k["_name"], {}))
        rows.append(rec)
        used_toi.add(ti)

    def _best_toi(period_k: float, candidates) -> tuple | None:
        best = None
        for ti in candidates:
            if ti in used_toi:
                continue
            ok, ratio, alias = period_match(float(period_k), float(toi.loc[ti, "pl_orbper"]),
                                            tol=period_tol, alias_max=alias_max)
            if ok and (best is None or alias == "1"):
                best = (ti, ratio, alias)
                if alias == "1":
                    break
        return best

    # --- routes 1-3: whatever gave a TIC id, finished by the period ---------
    # In ROUTE ORDER, not table order: a TOI is used once, so the more reliable
    # route must claim it first (a `position_tic` KOI must not take the TOI that
    # a named KOI would have matched).
    joined_koi: set = set()
    index = list(koi.index)
    for route in TIC_ROUTES:
        if route not in routes:
            continue
        for pos_i in (i for i, r in enumerate(tic_route) if r == route):
            ki = index[pos_i]
            k = koi.loc[ki]
            if not np.isfinite(k["tic_id_ps"]):
                continue
            best = _best_toi(k["koi_period"], toi_by_tid.get(int(k["tic_id_ps"]), []))
            if best is None:
                rep["tic_matched_no_period_match"] += 1
                continue
            ti, ratio, alias = best
            _emit(ki, ti, route, ratio, alias,
                  _sep_arcsec(k["ra"], k["dec"], toi.loc[ti, "ra"], toi.loc[ti, "dec"]))
            joined_koi.add(ki)
            rep["joined_by_route"][route] += 1

    # --- route 4: position + period, no catalogue crossmatch ----------------
    if "position_period" in routes:
        rest = koi.loc[[i for i in koi.index if i not in joined_koi]].dropna(subset=["ra", "dec"])
        tpos = toi.dropna(subset=["ra", "dec"])
        if len(rest) and len(tpos):
            hits = _match_positions(rest["ra"], rest["dec"], tpos["ra"], tpos["dec"],
                                    pos_radius_arcsec)
            for (ki, k), idxs in zip(rest.iterrows(), hits, strict=True):
                if not idxs:
                    continue
                best = _best_toi(k["koi_period"], [tpos.index[j] for j in idxs])
                if best is None:
                    rep["position_matched_no_period_match"] += 1
                    continue
                ti, ratio, alias = best
                _emit(ki, ti, "position_period", ratio, alias,
                      _sep_arcsec(k["ra"], k["dec"], toi.loc[ti, "ra"], toi.loc[ti, "dec"]))
                rep["joined_by_route"]["position_period"] += 1

    out = pd.DataFrame(rows)
    if len(out):
        out["planet_key"] = out["kepoi_name"].astype(str)
        # The Kepler position is the target position for the Gaia cones.
        out["ra"] = out["ra_kepler"]
        out["dec"] = out["dec_kepler"]
    rep["n_joined"] = int(len(out))
    rep["n_toi_used"] = int(len(used_toi))
    rep["n_koi_with_tess_counterpart"] = int(len(out))
    rep["fraction_koi_with_tess_counterpart"] = (
        float(len(out)) / float(len(koi)) if len(koi) else 0.0)
    rep["join_statement"] = _join_statement(rep)
    return out, rep


def _join_statement(rep: dict) -> str:
    """One plain sentence for ``summary.json``: how many KOIs reached TESS, how."""
    by = rep.get("joined_by_route") or {}
    n_koi = int(rep.get("n_koi", 0))
    n_join = int(rep.get("n_joined", 0))
    frac = (100.0 * n_join / n_koi) if n_koi else 0.0
    parts = ", ".join(f"{r}={int(by.get(r, 0))}" for r in JOIN_ROUTES)
    return (f"{n_join} of {n_koi} KOIs reached a TESS counterpart ({frac:.1f} %) — by route: "
            f"{parts}; {int(rep.get('koi_with_tic_id', 0))} KOIs resolved a TIC id "
            f"(by route: " + ", ".join(
                f"{r}={int((rep.get('koi_with_tic_by_route') or {}).get(r, 0))}"
                for r in TIC_ROUTES)
            + f"), of which {int(rep.get('n_koi_tics_with_a_toi', 0))} of "
              f"{int(rep.get('n_distinct_koi_tics', 0))} distinct TICs have any TOI at all and "
              f"{int(rep.get('tic_matched_no_period_match', 0))} KOIs had none whose period "
              f"matched; {int(rep.get('position_matched_no_period_match', 0))} positional "
              "matches failed the period test")


# ---------------------------------------------------------------------------
# Gaia DR3 neighbours (runner-only default; injectable)
# ---------------------------------------------------------------------------
class GaiaRouteFailed(RuntimeError):
    """A Gaia route gave up.  ``attempts`` carries the exception text of each try.

    The 2026-09-13 run reported ``n_targets_failed: 108`` and nothing else,
    which made the failure undiagnosable from the committed report.  Every
    attempt now carries its own ``{"attempt", "transport", "error"}`` record and
    those land verbatim in ``acquire.json`` under ``gaia.errors``.
    """

    def __init__(self, message: str, attempts=None):
        super().__init__(message)
        self.attempts: list[dict] = list(attempts or [])


def gaia_neighbours_upload(targets: pd.DataFrame, radius_arcsec: float, *,
                           retries: int = 3, attempts: list | None = None) -> pd.DataFrame:
    """Gaia TAP upload crossmatch: every DR3 source within ``radius_arcsec``.

    ``targets`` carries integer ``key``, ``ra``, ``dec``.  Returns
    ``key, source_id, ra, dec, phot_g_mean_mag, sep_arcsec``.  Async with
    backoff, sync on the last attempt (the herdsman pattern).

    **The anonymous ESA archive often refuses this.**  Run 34787801172 got
    ``Error 500 ... canceling statement due to statement timeout`` on all three
    attempts for a single 108-target chunk: a ``tap_upload`` join against the
    whole ``gaiadr3.gaia_source`` is not a query the anonymous queue will
    finish.  Each attempt's text is appended to ``attempts`` (when given) and
    carried on the raised :class:`GaiaRouteFailed`, and the caller then falls
    back to :func:`gaia_neighbours_cones`.
    """
    from astropy.table import Table  # noqa: PLC0415
    from astroquery.gaia import Gaia  # noqa: PLC0415

    rec: list[dict] = attempts if attempts is not None else []
    up = Table.from_pandas(targets[["key", "ra", "dec"]].reset_index(drop=True))
    r_deg = float(radius_arcsec) / 3600.0
    adql = (
        "SELECT t.key AS key, g.source_id, g.ra, g.dec, g.phot_g_mean_mag, "
        "DISTANCE(POINT('ICRS', t.ra, t.dec), POINT('ICRS', g.ra, g.dec)) * 3600.0 AS sep_arcsec "
        "FROM tap_upload.targets AS t JOIN gaiadr3.gaia_source AS g "
        f"ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', t.ra, t.dec, {r_deg:.8f}))")
    for attempt in range(int(retries)):
        transport = "astroquery_sync" if attempt + 1 == retries else "astroquery_async"
        try:
            launch = Gaia.launch_job if transport == "astroquery_sync" else Gaia.launch_job_async
            job = launch(adql, upload_resource=up, upload_table_name="targets")
            out = job.get_results().to_pandas()
            rec.append({"attempt": attempt + 1, "transport": transport, "ok": True,
                        "n_rows": int(len(out))})
            return out
        except Exception as exc:                          # noqa: BLE001
            rec.append({"attempt": attempt + 1, "transport": transport, "ok": False,
                        "error": repr(exc)[:500]})
            print(f"[growth/acquire] Gaia upload xmatch attempt {attempt + 1}/{retries} "
                  f"({transport}) failed: {exc!r}")
            if attempt + 1 < int(retries):
                _time.sleep(5.0 * (attempt + 1))
    raise GaiaRouteFailed(
        f"Gaia upload crossmatch failed after {retries} attempts "
        f"(n={len(targets)}, r={radius_arcsec}\")", rec)


def gaia_cone_adql(ra: float, dec: float, radius_arcsec: float) -> str:
    """The small per-target cone the fallback sends (baffle's shape, one target)."""
    r_deg = float(radius_arcsec) / 3600.0
    return ("SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
            f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {float(ra):.7f}, {float(dec):.7f}, {r_deg:.8f}))")


def pyvo_sync_transport(tap_url: str = GAIA_TAP):
    """``adql -> DataFrame`` through ``pyvo``'s sync endpoint (runner only)."""

    def _run(adql: str) -> pd.DataFrame:
        import pyvo  # noqa: PLC0415  runner-only

        df = pyvo.dal.TAPService(tap_url).run_sync(adql).to_table().to_pandas()
        return df.rename(columns={c: str(c).lower() for c in df.columns})

    return _run


def gaia_neighbours_cones(targets: pd.DataFrame, radius_arcsec: float, *,
                          retries: int = 2, tap_url: str = GAIA_TAP, transport=None,
                          base_sleep: float = 2.0, attempts: list | None = None
                          ) -> tuple[pd.DataFrame, list[int]]:
    """Per-target sync cones --- the fallback when the upload route is refused.

    One small ADQL per target through ``pyvo`` against ``tap_url``
    (:data:`GAIA_TAP`), retried.  Returns ``(neighbours, failed_keys)``: the
    neighbour rows carry the target's ``key`` so the caller can join them back,
    and a target whose every attempt failed is listed in ``failed_keys`` and
    stays ``QUERY_FAILED`` --- it is never silently reported as isolated.
    """
    run = transport or pyvo_sync_transport(tap_url)
    rec: list[dict] = attempts if attempts is not None else []
    frames: list[pd.DataFrame] = []
    failed: list[int] = []
    for _, t in targets.reset_index(drop=True).iterrows():
        key = int(t["key"])
        last = None
        for attempt in range(max(1, int(retries))):
            try:
                df = run(gaia_cone_adql(float(t["ra"]), float(t["dec"]), radius_arcsec))
                df = (df if df is not None else pd.DataFrame()).copy()
                df["key"] = key
                frames.append(df)
                last = None
                break
            except Exception as exc:                      # noqa: BLE001
                last = exc
                if attempt + 1 < max(1, int(retries)):
                    _time.sleep(base_sleep * (attempt + 1))
        if last is not None:
            failed.append(key)
            rec.append({"key": key, "transport": "pyvo_sync", "ok": False,
                        "error": repr(last)[:500]})
    neigh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["source_id", "ra", "dec", "phot_g_mean_mag", "key"])
    return neigh, failed


def _attach_targets(df: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """Give neighbour rows their ``planet_key`` and a separation, from ``key``."""
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    df["key"] = pd.to_numeric(df["key"], errors="coerce").astype(int)
    df = df.merge(part[["key", "planet_key", "ra", "dec"]].rename(
        columns={"ra": "ra_target", "dec": "dec_target"}), on="key", how="left")
    if "sep_arcsec" not in df:
        df["sep_arcsec"] = [_sep_arcsec(a, b, c, d) for a, b, c, d in
                            zip(df["ra_target"], df["dec_target"], df["ra"], df["dec"],
                                strict=True)]
    return df


def fetch_gaia_neighbours(stars: pd.DataFrame, *, radius_arcsec: float = 21.0,
                          chunk: int = 300, gaia_fn=None, cone_fn=None,
                          log: AcquisitionLog | None = None,
                          checkpoint_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Neighbours for every star in ``stars`` (``planet_key, ra, dec``), chunked.

    Two routes per chunk, in order:

    1. ``gaia_fn`` --- the upload crossmatch (:func:`gaia_neighbours_upload`);
    2. ``cone_fn`` --- per-target sync cones (:func:`gaia_neighbours_cones`),
       used when the upload route raises.  ``cone_fn=None`` means no fallback:
       the caller decides, so an injected ``gaia_fn`` never reaches the network
       by accident.

    Returns ``(neighbours, status)``: one neighbour row per (target, Gaia
    source) and one status row per target, with ``neighbours_status``
    (``OK`` / ``QUERY_FAILED``) and ``neighbours_route`` (``upload`` /
    ``cones`` / ``checkpoint`` / ``""``).  A target neither route reached is
    ``QUERY_FAILED`` and is reported as ``not_checked`` rather than assumed
    isolated.  Every attempt's exception text is recorded on the log stage
    (``attempts``), and a finished chunk is written to ``checkpoint_dir`` so a
    re-run does not re-query it.
    """
    log = log or AcquisitionLog()
    gaia_fn = gaia_fn or gaia_neighbours_upload
    ck = Path(checkpoint_dir) if checkpoint_dir else None
    if ck is not None:
        ck.mkdir(parents=True, exist_ok=True)
    st = stars.dropna(subset=["ra", "dec"]).drop_duplicates("planet_key").reset_index(drop=True)
    st["key"] = np.arange(len(st), dtype=int)
    neigh_frames, status = [], []
    n_chunks = int(np.ceil(len(st) / max(int(chunk), 1))) if len(st) else 0
    for i in range(n_chunks):
        part = st.iloc[i * chunk:(i + 1) * chunk]
        label = f"gaia_neighbours_chunk_{i + 1}of{n_chunks}"
        query = f"gaia neighbours r={radius_arcsec}\" n={len(part)}"
        keys = list(part["key"].astype(int))
        by_key = dict(zip(part["key"].astype(int), part["planet_key"], strict=True))

        ckfile = (ck / f"chunk_{i + 1:04d}.csv") if ck is not None else None
        if ckfile is not None and ckfile.exists():
            try:
                df = pd.read_csv(ckfile)
            except Exception:                             # noqa: BLE001
                df = None
            if df is not None:
                log.record(label, query, rows=int(len(df)),
                           extra={"route": "checkpoint", "n_targets": len(part),
                                  "n_targets_ok": len(part), "n_targets_failed": 0})
                if len(df):
                    neigh_frames.append(df)
                status.extend({"planet_key": by_key[k], "neighbours_status": STATUS_OK,
                               "neighbours_route": "checkpoint"} for k in keys)
                continue

        attempts: list[dict] = []
        route, failed_keys, df = "upload", [], None
        try:
            df = gaia_fn(part[["key", "ra", "dec"]], radius_arcsec)
        except Exception as exc:                          # noqa: BLE001
            attempts.extend(getattr(exc, "attempts", None) or
                            [{"transport": "upload", "ok": False, "error": repr(exc)[:500]}])
            if cone_fn is None:
                log.record(label, query, error=repr(exc), extra={
                    "route": "upload", "attempts": attempts, "n_targets": len(part),
                    "n_targets_ok": 0, "n_targets_failed": len(part)})
                status.extend({"planet_key": by_key[k], "neighbours_status": STATUS_FAILED,
                               "neighbours_route": ""} for k in keys)
                continue
            print(f"[growth/acquire] {label}: upload route failed ({exc!r}); "
                  f"falling back to {len(part)} per-target cones")
            route = "cones"
            try:
                df, failed_keys = cone_fn(part[["key", "ra", "dec"]], radius_arcsec,
                                          attempts=attempts)
            except Exception as exc2:                     # noqa: BLE001
                attempts.append({"transport": "cones", "ok": False, "error": repr(exc2)[:500]})
                log.record(label, query, error=repr(exc2), extra={
                    "route": "cones", "attempts": attempts, "n_targets": len(part),
                    "n_targets_ok": 0, "n_targets_failed": len(part)})
                status.extend({"planet_key": by_key[k], "neighbours_status": STATUS_FAILED,
                               "neighbours_route": ""} for k in keys)
                continue

        df = df if df is not None else pd.DataFrame()
        ok_keys = [k for k in keys if k not in set(failed_keys)]
        if len(df):
            df = _attach_targets(df, part)
            neigh_frames.append(df)
            if ckfile is not None and not failed_keys:
                try:
                    df.to_csv(ckfile, index=False)
                except Exception:                         # noqa: BLE001
                    pass
        log.record(label, query, rows=int(len(df)) if not failed_keys or len(df) else None,
                   error=(f"{len(failed_keys)} of {len(part)} targets unreachable on every route"
                          if failed_keys else None),
                   extra={"route": route, "attempts": attempts, "n_targets": len(part),
                          "n_targets_ok": len(ok_keys), "n_targets_failed": len(failed_keys)})
        status.extend({"planet_key": by_key[k],
                       "neighbours_status": STATUS_FAILED if k in set(failed_keys) else STATUS_OK,
                       "neighbours_route": "" if k in set(failed_keys) else route}
                      for k in keys)
    neigh = pd.concat(neigh_frames, ignore_index=True) if neigh_frames else pd.DataFrame(
        columns=["key", "planet_key", "source_id", "ra", "dec", "phot_g_mean_mag", "sep_arcsec"])
    return neigh, pd.DataFrame(status, columns=["planet_key", "neighbours_status",
                                                "neighbours_route"])


__all__ = [
    "EXOARCHIVE_TAP", "GAIA_TAP", "JOIN_ROUTES", "KOI_COLUMNS", "PS_COLUMNS",
    "PS_WHERE_KEPLER", "STATUS_FAILED", "STATUS_OK", "STATUS_ZERO", "TIC_ROUTES",
    "TOI_COLUMNS", "AcquisitionLog", "GaiaRouteFailed", "build_url", "fetch_gaia_neighbours",
    "fetch_koi", "fetch_ps_kepler", "fetch_toi", "gaia_cone_adql", "gaia_neighbours_cones",
    "gaia_neighbours_upload", "host_of_planet_name", "join_kepler_tess", "parse_tap_csv",
    "parse_tic_id", "period_match", "prepare_ps", "pyvo_sync_transport", "table_query",
    "tap_sync",
]

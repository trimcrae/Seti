"""ISOTHERM acquisition (runner only; sandbox egress is blocked with HTTP 403).

Primary corpus: **CASSIS**, the Cornell Atlas of Spitzer/IRS Sources — ~13,000
public low-resolution 5-38 micron spectra.  To our knowledge CASSIS has never
been used for a technosignature search of any kind.

Four bands cannot measure an emissivity index or detect a silicate feature, and
that is not an opinion — Wright et al. 2014 (Gh II) state it outright: "a
broadband search cannot distinguish the spectral features of dust from that of
other sources of MIR radiation."  The shape statistics need spectra, so the
corpus choice follows from the physics.

Access routes, tried in order and each probed independently so the run reports
exactly which one worked:

1. **CASSIS direct HTTP** — positional search and ASCII spectrum download.
2. **IRSA TAP, Spitzer/IRS Enhanced Products** — the merged low-res 5.2-38
   micron spectra.  Table names drift between IRSA releases, so the table is
   *discovered* from ``TAP_SCHEMA`` rather than hard-coded (the same dynamic
   resolution ``herdsman_b.acquire`` uses for VizieR columns).
3. **VizieR** — the CASSIS catalogue (Lebouteiller et al. 2011) and the IRAS
   LRS Calgary atlas (Volk & Cohen), the latter being Carrigan 2009's corpus.
4. **Photometric backstop** — Gaia + 2MASS + WISE + AKARI + IRAS SED assembly
   at catalogue scale, if no spectral archive is reachable.

Every candidate is anchored to Gaia.  Carrigan 2009 — the only prior search that
thresholded on SED shape at all — died on the distance/luminosity degeneracy: a
nearby 1 L_sun sphere and a distant 10^3 L_sun red giant produce the same
spectrum.  Gaia parallaxes break exactly that degeneracy, which is the single
biggest reason a shape-based search is worth re-running now.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

_CASSIS_BASE = "https://cassis.sirtf.com/atlas"
_CASSIS_ALT = "https://cassis.astro.cornell.edu/atlas"
_IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
_VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
_GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

# VizieR catalogues.  CASSIS = Lebouteiller+ 2011 ApJS 196, 8.  The IRAS LRS
# Calgary atlas is Carrigan 2009's input corpus.
_VIZIER_CASSIS = ("J/ApJS/196/8/catalog", "J/ApJS/196/8/table1")
_VIZIER_IRAS_LRS = ("III/197/lrs", "III/197/catalog", "III/103A/catalog")

_HTTP_TIMEOUT = 60.0


def _retry(fn, retries: int = 3, label: str = "fetch"):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[isotherm] {label} attempt {attempt + 1}/{retries} "
                  f"failed: {exc!r}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


# ---------------------------------------------------------------------------
# Reachability probe — runs FIRST and is reported as a first-class result
# ---------------------------------------------------------------------------

def _probe_http(url: str, expect: str = "") -> dict:
    import requests

    try:
        r = requests.get(url, timeout=_HTTP_TIMEOUT)
        body = r.text[:4000] if r.content else ""
        return {"reachable": bool(r.status_code == 200 and r.content),
                "status": int(r.status_code), "bytes": len(r.content or b""),
                "content_type": r.headers.get("Content-Type", ""),
                "expect_found": bool(expect and expect.lower() in body.lower()),
                "snippet": body[:300]}
    except Exception as exc:  # noqa: BLE001
        return {"reachable": False, "status": 0, "bytes": 0,
                "error": repr(exc)[:300]}


def _probe_tap(url: str, query: str) -> dict:
    try:
        import pyvo

        tap = pyvo.dal.TAPService(url)
        rows = tap.search(query).to_table()
        return {"reachable": True, "n_rows": int(len(rows)),
                "columns": [str(c) for c in rows.colnames][:40]}
    except Exception as exc:  # noqa: BLE001
        return {"reachable": False, "error": repr(exc)[:300]}


def discover_irs_tables() -> dict:
    """Find IRSA TAP tables that could hold Spitzer/IRS spectra.

    Table names drift between IRSA releases, so they are discovered rather than
    hard-coded.  A hard-coded name that has been renamed looks exactly like an
    unreachable archive, and this channel must never confuse the two.
    """
    q = ("SELECT table_name FROM TAP_SCHEMA.tables "
         "WHERE table_name LIKE '%irs%' OR table_name LIKE '%spitzer%' "
         "OR table_name LIKE '%cassis%'")
    try:
        import pyvo

        rows = pyvo.dal.TAPService(_IRSA_TAP).search(q).to_table()
        names = [str(r["table_name"]) for r in rows]
        return {"ok": True, "n": len(names), "tables": names[:200]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)[:300], "tables": []}


def probe_archives(out_dir: Path | None = None) -> dict:
    """Probe every access route; write and return a reachability report.

    Emits an explicit ``NO_DATA_REACHED`` style verdict rather than silently
    degrading, per the channel brief.
    """
    report: dict = {"probed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ")}

    report["cassis_home"] = _probe_http(f"{_CASSIS_BASE}/", expect="cassis")
    report["cassis_alt_home"] = _probe_http(f"{_CASSIS_ALT}/", expect="cassis")
    # Positional cone search around a bright, certainly-observed IRS target.
    report["cassis_radec"] = _probe_http(
        f"{_CASSIS_BASE}/cgi/radec.py?ra=83.8221&dec=-5.3911&radius=600")
    report["cassis_ascii"] = _probe_http(
        f"{_CASSIS_BASE}/cgi/ascii.py?aorkey=3608064&ext=0")

    report["irsa_tap"] = _probe_tap(
        _IRSA_TAP, "SELECT TOP 1 table_name FROM TAP_SCHEMA.tables")
    report["irsa_irs_tables"] = discover_irs_tables()
    report["vizier_tap"] = _probe_tap(
        _VIZIER_TAP, "SELECT TOP 1 table_name FROM TAP_SCHEMA.tables")
    report["gaia_tap"] = _probe_tap(
        _GAIA_TAP, "SELECT TOP 1 source_id FROM gaiadr3.gaia_source")

    for name, tabs in (("vizier_cassis", _VIZIER_CASSIS),
                       ("vizier_iras_lrs", _VIZIER_IRAS_LRS)):
        got = {"reachable": False, "table": None}
        for t in tabs:
            r = _probe_tap(_VIZIER_TAP, f'SELECT TOP 1 * FROM "{t}"')
            if r.get("reachable"):
                got = {"reachable": True, "table": t,
                       "columns": r.get("columns", [])}
                break
        report[name] = got

    spectral_ok = bool(
        report["cassis_ascii"].get("reachable")
        or report["cassis_radec"].get("reachable")
        or report["vizier_cassis"].get("reachable")
        or (report["irsa_irs_tables"].get("ok")
            and report["irsa_irs_tables"].get("n", 0) > 0))
    report["cassis_reachable"] = bool(
        report["cassis_home"].get("reachable")
        or report["cassis_ascii"].get("reachable")
        or report["cassis_radec"].get("reachable"))
    report["any_spectral_archive_reachable"] = spectral_ok
    report["photometric_backstop_available"] = bool(
        report["irsa_tap"].get("reachable") or report["vizier_tap"].get("reachable"))
    report["verdict"] = (
        "SPECTRAL_ARCHIVE_REACHED" if spectral_ok
        else ("PHOTOMETRIC_BACKSTOP_ONLY"
              if report["photometric_backstop_available"] else "NO_DATA_REACHED"))

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "archive_probe.json").write_text(
            json.dumps(report, indent=2, default=str))
    print(f"[isotherm] archive probe verdict: {report['verdict']} "
          f"(cassis_reachable={report['cassis_reachable']})")
    return report


# ---------------------------------------------------------------------------
# Spectral corpus
# ---------------------------------------------------------------------------

def parse_cassis_ascii(text: str) -> pd.DataFrame:
    """Parse a CASSIS ASCII spectrum: wavelength, flux (Jy), error (Jy).

    Tolerant of the several column layouts CASSIS has shipped: takes the first
    three numeric columns and drops non-positive wavelengths.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#;/|":
            continue
        parts = line.replace(",", " ").split()
        try:
            vals = [float(p) for p in parts[:4]]
        except ValueError:
            continue
        if len(vals) >= 3:
            rows.append(vals[:3])
    if not rows:
        return pd.DataFrame(columns=["wavelength_um", "flux_jy", "err_jy"])
    arr = np.asarray(rows, float)
    df = pd.DataFrame(arr, columns=["wavelength_um", "flux_jy", "err_jy"])
    df = df[(df["wavelength_um"] > 0) & np.isfinite(df).all(axis=1)]
    return df.sort_values("wavelength_um").reset_index(drop=True)


def fetch_cassis_spectrum(aorkey: int, ext: int = 0,
                          base: str = _CASSIS_BASE) -> pd.DataFrame:
    """Download one CASSIS low-resolution spectrum by AOR key."""
    import requests

    def _go():
        r = requests.get(f"{base}/cgi/ascii.py?aorkey={int(aorkey)}&ext={int(ext)}",
                         timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        return parse_cassis_ascii(r.text)

    return _retry(_go, retries=3, label=f"cassis aorkey {aorkey}")


def fetch_irs_catalog(out_path: Path, table: str | None = None,
                      max_rows: int = 40000) -> pd.DataFrame:
    """Positional catalogue of Spitzer/IRS spectra from IRSA TAP (checkpointed)."""
    out_path = Path(out_path)
    if out_path.exists():
        print(f"[isotherm] IRS catalog checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)

    import pyvo

    tap = pyvo.dal.TAPService(_IRSA_TAP)
    tables = [table] if table else discover_irs_tables().get("tables", [])
    last = None
    for t in tables:
        try:
            rows = _retry(
                lambda t=t: tap.search(
                    f"SELECT TOP {int(max_rows)} * FROM {t}").to_table(),
                retries=2, label=f"IRSA {t}")
            df = rows.to_pandas()
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            df["_source_table"] = t
            df.to_parquet(out_path, index=False)
            print(f"[isotherm] IRS catalog: {len(df)} rows from {t}")
            return df
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[isotherm] IRSA table {t} unusable: {exc!r}")
    raise RuntimeError(f"no usable IRS table on IRSA TAP: {last!r}")


def fetch_vizier_table(table: str, out_path: Path,
                       max_rows: int = 40000) -> pd.DataFrame:
    """Pull a VizieR table via TAPVizieR (checkpointed)."""
    out_path = Path(out_path)
    if out_path.exists():
        print(f"[isotherm] {table} checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    import pyvo

    tap = pyvo.dal.TAPService(_VIZIER_TAP)
    rows = _retry(
        lambda: tap.search(f'SELECT TOP {int(max_rows)} * FROM "{table}"').to_table(),
        retries=3, label=f"vizier {table}")
    df = rows.to_pandas().rename(columns=lambda c: str(c).lower())
    df.to_parquet(out_path, index=False)
    print(f"[isotherm] {table}: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Gaia anchoring — this is what Carrigan 2009 lacked
# ---------------------------------------------------------------------------

_GAIA_CONE = """
SELECT TOP 5 source_id, ra, dec, parallax, parallax_over_error, ruwe,
       phot_g_mean_mag, bp_rp, teff_gspphot
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {rad}))
ORDER BY phot_g_mean_mag
"""


def gaia_anchor(ra_deg: float, dec_deg: float, radius_arcsec: float = 3.0) -> dict:
    """Nearest Gaia DR3 source, with the distance the shape fit needs.

    Without a parallax an IRS spectrum cannot distinguish a nearby low-luminosity
    emitter from a distant luminous one — the degeneracy that made Carrigan
    2009's three best candidates inconclusive.
    """
    out = {"gaia_source_id": None, "parallax": np.nan, "distance_pc": np.nan,
           "ruwe": np.nan, "phot_g_mean_mag": np.nan, "matched": False}
    try:
        import pyvo

        q = _GAIA_CONE.format(ra=float(ra_deg), dec=float(dec_deg),
                              rad=float(radius_arcsec) / 3600.0)
        rows = pyvo.dal.TAPService(_GAIA_TAP).search(q).to_table()
        if len(rows) == 0:
            return out
        r = rows[0]
        plx = float(r["parallax"]) if r["parallax"] is not None else np.nan
        out.update(gaia_source_id=int(r["source_id"]), parallax=plx,
                   distance_pc=(1000.0 / plx if np.isfinite(plx) and plx > 0
                                else np.nan),
                   ruwe=float(r["ruwe"]) if r["ruwe"] is not None else np.nan,
                   phot_g_mean_mag=float(r["phot_g_mean_mag"])
                   if r["phot_g_mean_mag"] is not None else np.nan,
                   matched=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[isotherm] Gaia anchor failed at {ra_deg},{dec_deg}: {exc!r}")
    return out


# ---------------------------------------------------------------------------
# Photometric backstop — used only if no spectral archive is reachable
# ---------------------------------------------------------------------------

# Effective wavelengths (micron) for SED assembly.  The far-IR bands are what
# give the photometric path any beta leverage at all: WISE alone cannot do it.
PHOTOMETRIC_BANDS = {
    "j_2mass": 1.235, "h_2mass": 1.662, "k_2mass": 2.159,
    "w1": 3.353, "w2": 4.603, "w3": 11.561, "w4": 22.088,
    "akari_s9w": 8.61, "akari_l18w": 18.39,
    "iras_12": 11.59, "iras_25": 23.88, "iras_60": 61.85, "iras_100": 101.94,
    "akari_n60": 65.0, "akari_wides": 90.0, "akari_widel": 140.0,
    "akari_n160": 160.0,
}

# Wien peaks of the WISE bands, for the coverage table in docs/isotherm.md:
# W1 3.4 um -> 852 K, W2 4.6 -> 630 K, W3 12 -> 241 K, W4 22 -> 132 K.  W3 and
# W4 depth is frozen at the 2010 cryogenic mission — NEOWISE-R, CatWISE2020 and
# the unWISE coadds are W1/W2 only — so no future broadband data will improve
# amplitude-based selection in the cold regime.  Shape is the way forward.
WISE_BAND_WIEN_K = {"w1": 852.0, "w2": 630.0, "w3": 241.0, "w4": 132.0}


def photometry_to_spectrum(row: dict, bands: dict | None = None,
                           frac_err_floor: float = 0.05) -> tuple:
    """Turn a photometric row into (wavelength, F_nu, sigma) for the SED fitter.

    Expects ``<band>_jy`` and optionally ``<band>_jy_err``.  Bands without a
    finite flux are dropped, and the count of surviving bands is returned so the
    caller can degrade honestly instead of fitting three points with five
    parameters.
    """
    bands = bands or PHOTOMETRIC_BANDS
    lam, flx, err = [], [], []
    for name, wl in bands.items():
        f = row.get(f"{name}_jy", np.nan)
        if f is None or not np.isfinite(f) or f <= 0:
            continue
        e = row.get(f"{name}_jy_err", np.nan)
        if e is None or not np.isfinite(e) or e <= 0:
            e = frac_err_floor * f
        lam.append(wl)
        flx.append(float(f))
        err.append(float(max(e, frac_err_floor * f)))
    order = np.argsort(lam)
    return (np.asarray(lam)[order], np.asarray(flx)[order],
            np.asarray(err)[order], len(lam))


__all__ = [
    "PHOTOMETRIC_BANDS", "WISE_BAND_WIEN_K", "discover_irs_tables",
    "fetch_cassis_spectrum", "fetch_irs_catalog", "fetch_vizier_table",
    "gaia_anchor", "parse_cassis_ascii", "photometry_to_spectrum",
    "probe_archives",
]

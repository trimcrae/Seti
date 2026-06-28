"""Robust, CDS-only real-data acquisition for execution on a CI runner.

GitHub Actions runners have unrestricted outbound network, so the empirical
catalogue pull runs there (where Gaia/VizieR/CDS are reachable) even though the
interactive sandbox is egress-restricted.  This module uses only VizieR and the
CDS X-Match service -- no Gaia TAP IN-lists, no uploads -- so it scales to the
full 100 pc sample without authentication, and it is deliberately defensive
about catalogue column names (which vary) and verbose so failures are diagnosable
from CI logs alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def fetch_wd_parent(max_dist_pc: float, pwd_min: float, row_limit: int = -1) -> pd.DataFrame:
    """Gentile Fusillo et al. 2021 WD catalogue, filtered to the nearby sample."""
    from astroquery.vizier import Vizier

    plx_min = 1000.0 / max_dist_pc
    v = Vizier(columns=["**"], row_limit=row_limit,
               column_filters={"Plx": f">={plx_min:.4f}", "Pwd": f">={pwd_min}"})
    cats = v.get_catalogs("J/MNRAS/508/3877")
    raw = cats[0].to_pandas()
    print(f"[science] WD parent raw rows: {len(raw)}; columns: {list(raw.columns)[:25]}")

    # Defensive column mapping (VizieR names vary across mirrors/versions).
    def req(cands):
        col = _find_col(raw, cands)
        if col is None:
            raise KeyError(f"none of {cands} found in WD catalogue columns: "
                           f"{list(raw.columns)}")
        return raw[col]

    out = pd.DataFrame()
    out["source_id"] = req(["GaiaEDR3", "Gaia", "Source", "DR3Name", "WD"])
    out["ra"] = req(["RA_ICRS", "RAJ2000", "_RA", "RAdeg", "RAdeg"])
    out["dec"] = req(["DE_ICRS", "DEJ2000", "_DE", "DEdeg"])
    plx = _find_col(raw, ["Plx", "Plx1", "parallax"])
    eplx = _find_col(raw, ["e_Plx", "e_Plx1", "parallax_error"])
    out["parallax"] = raw[plx]
    if eplx:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["parallax_over_error"] = raw[plx] / raw[eplx]
    pmra = _find_col(raw, ["pmRA", "pmra"])
    pmdec = _find_col(raw, ["pmDE", "pmdec"])
    if pmra:
        out["pmra"] = raw[pmra]
    if pmdec:
        out["pmdec"] = raw[pmdec]
    out["teff"] = req(["TeffH", "Teff", "teffH", "TeffHe", "Teff_H"])
    logg = _find_col(raw, ["loggH", "logg", "loggHe"])
    if logg:
        out["logg"] = raw[logg]
    pwd = _find_col(raw, ["Pwd", "PWD"])
    out["pwd"] = raw[pwd]
    for cand, name in [("RUWE", "ruwe"), ("Gmag", "Gmag")]:
        col = _find_col(raw, [cand])
        if col:
            out[name] = raw[col]
    out = out.dropna(subset=["source_id", "ra", "dec", "teff"]).reset_index(drop=True)
    print(f"[science] WD parent usable rows (<= {max_dist_pc:.0f} pc, Pwd>={pwd_min}): {len(out)}")
    return out


def _xmatch(positions: pd.DataFrame, vizier_table: str, radius_arcsec: float) -> pd.DataFrame:
    from astropy import units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch

    frames = []
    step = 50_000
    for start in range(0, len(positions), step):
        sub = positions.iloc[start:start + step][["source_id", "ra", "dec"]].copy()
        # XMatch requires an astropy Table (not a pandas DataFrame) for the upload.
        cat1 = Table.from_pandas(sub)
        res = XMatch.query(cat1=cat1, cat2=vizier_table,
                           max_distance=radius_arcsec * u.arcsec,
                           colRA1="ra", colDec1="dec")
        frames.append(res.to_pandas())
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"[science] X-Match {vizier_table}: {len(out)} matches for {len(positions)} inputs")
    return out


def _map(raw: pd.DataFrame, targets: dict[str, list[str]]) -> pd.DataFrame:
    """Build a frame mapping target column -> first matching candidate in raw."""
    out = pd.DataFrame()
    for target, cands in targets.items():
        col = _find_col(raw, cands)
        if col is not None:
            out[target] = raw[col].to_numpy()
    return out


def fetch_catwise(positions: pd.DataFrame, radius_arcsec: float = 3.0) -> pd.DataFrame:
    raw = _xmatch(positions, "vizier:II/365/catwise", radius_arcsec)
    if raw.empty:
        return raw
    print(f"[science] CatWISE X-Match columns: {list(raw.columns)}")
    out = _map(raw, {
        "source_id": ["source_id"],
        "ra_wise": ["RA_ICRS", "RAPMdeg", "RAdeg", "RAJ2000", "_RAJ2000", "ra_2", "RAICRS"],
        "dec_wise": ["DE_ICRS", "DEPMdeg", "DEdeg", "DEJ2000", "_DEJ2000", "dec_2", "DEICRS"],
        "W1mag": ["W1mproPM", "W1mag", "W1mpro"],
        "e_W1mag": ["e_W1mproPM", "e_W1mag", "e_W1mpro"],
        "W2mag": ["W2mproPM", "W2mag", "W2mpro"],
        "e_W2mag": ["e_W2mproPM", "e_W2mag", "e_W2mpro"],
        "pmra_wise": ["pmRA", "pmRAPM"],
        "pmdec_wise": ["pmDE", "pmDEPM"],
        "e_pmra_wise": ["e_pmRA"],
        "e_pmdec_wise": ["e_pmDE"],
        "cc_flags": ["ccf", "cc_flags", "abf"],
        "ph_qual": ["qph", "ph_qual"],
        "angDist": ["angDist"],
    })
    if "angDist" in out.columns:
        out = out.sort_values("angDist").drop_duplicates("source_id").drop(columns="angDist")
    return out


_TWOMASS_TARGETS = {
    "source_id": ["source_id"],
    "Jmag": ["Jmag"], "e_Jmag": ["e_Jmag"],
    "Hmag": ["Hmag"], "e_Hmag": ["e_Hmag"],
    "Ksmag": ["Kmag", "Ksmag"], "e_Ksmag": ["e_Kmag", "e_Ksmag"],
    "angDist": ["angDist"],
}


def fetch_twomass(positions: pd.DataFrame, radius_arcsec: float = 8.0) -> pd.DataFrame:
    # Larger radius than CatWISE: 2MASS epoch (~1999) is ~17 yr before Gaia
    # (2016), so high-proper-motion nearby white dwarfs shift by several arcsec.
    raw = _xmatch(positions, "vizier:II/246/out", radius_arcsec)
    if raw.empty:
        return raw
    print(f"[science] 2MASS X-Match columns: {list(raw.columns)}")
    out = _map(raw, _TWOMASS_TARGETS)
    if "angDist" in out.columns:
        out = out.sort_values("angDist").drop_duplicates("source_id").drop(columns="angDist")
    return out


def fetch_known_disks(positions: pd.DataFrame, radius_arcsec: float = 2.0) -> set:
    """Source_ids matching the Madurga Favieres 2024 WD IR-excess control sample."""
    try:
        raw = _xmatch(positions, "vizier:J/A+A/688/A168", radius_arcsec)
        if "source_id" in raw.columns and len(raw):
            return set(raw["source_id"].dropna().astype("int64"))
    except Exception as exc:  # control catalogue is optional; never fatal
        print(f"[science] known-disk X-Match skipped: {exc!r}")
    return set()


__all__ = ["fetch_wd_parent", "fetch_catwise", "fetch_twomass", "fetch_known_disks"]

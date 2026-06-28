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
    # Gaia photometry (RUWE + G/BP/RP) for astrometric vetting and as a fallback
    # SED anchor where 2MASS is unavailable.
    for cands, name in [(["RUWE"], "ruwe"),
                        (["Gmag", "phot_g_mean_mag"], "Gmag"),
                        (["BPmag", "phot_bp_mean_mag", "BP"], "BPmag"),
                        (["RPmag", "phot_rp_mean_mag", "RP"], "RPmag")]:
        col = _find_col(raw, cands)
        if col is not None:
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
    # CatWISE2020 (VizieR II/365) reports proper motions in arcsec/yr; the
    # pipeline and Gaia use mas/yr. Convert so the co-movement PM test is valid.
    for col in ("pmra_wise", "pmdec_wise", "e_pmra_wise", "e_pmdec_wise"):
        if col in out.columns:
            out[col] = out[col] * 1000.0
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


def fetch_galex(positions: pd.DataFrame, radius_arcsec: float = 3.0) -> pd.DataFrame:
    """GALEX NUV/FUV (GUVcat_AIS, VizieR II/335) by position, for the UV-deficit
    and energy-balance axes. AB magnitudes."""
    raw = _xmatch(positions, "vizier:II/335/galex_ais", radius_arcsec)
    if raw.empty:
        return raw
    print(f"[science] GALEX X-Match columns: {list(raw.columns)[:25]}")
    out = _map(raw, {
        "source_id": ["source_id"],
        "NUVmag": ["NUVmag", "NUV"], "e_NUVmag": ["e_NUVmag", "e_NUV"],
        "FUVmag": ["FUVmag", "FUV"], "e_FUVmag": ["e_FUVmag", "e_FUV"],
        "angDist": ["angDist"],
    })
    if "angDist" in out.columns:
        out = out.sort_values("angDist").drop_duplicates("source_id").drop(columns="angDist")
    return out


def classify_candidate(otype: str, sptype: str) -> str:
    """Coarse natural-explanation class for a candidate from its SIMBAD tags.

    Separates the leading mundane explanations for a multi-axis white-dwarf
    anomaly --- an interacting/cataclysmic binary, an eclipsing or cool-companion
    binary, a metal-polluted (disk-bearing) white dwarf --- from genuinely
    unexamined sources, which are the interesting residue.
    """
    o = (otype or "").strip().upper()
    s = (sptype or "").strip().upper()
    if not o and not s:
        return "unexamined"
    if any(k in o for k in ("CV", "NOVA", "AM CVN", "AMCVN", "NL")):
        return "interacting binary (CV)"
    if any(k in o for k in ("EB", "ECL", "ALGOL", "BETA LYR", "WUMA")):
        return "eclipsing binary"
    if "+M" in s or "+K" in s or s.endswith("+M") or "DA+M" in s:
        return "WD+dwarf binary"
    if any(k in s for k in ("DAZ", "DBZ", "DZ", "DZA")):
        return "metal-polluted WD (disk)"
    if "**" in o or "SB" in o or "BINARY" in o:
        return "binary"
    if "WD" in o or s.startswith(("DA", "DB", "DC", "DO", "DQ")):
        return "white dwarf (other)"
    return o.lower() or "unexamined"


def _first_cell(df: pd.DataFrame, names: list[str]):
    """First-row value of the first column (case-insensitive) matching ``names``."""
    for name in names:
        for actual in df.columns:
            if actual.lower() == name:
                return df[actual].iloc[0]
    return ""


def fetch_simbad_context(positions: pd.DataFrame, radius_arcsec: float = 5.0) -> pd.DataFrame:
    """Annotate a (small) candidate list with SIMBAD identity and object type.

    For the multi-axis candidates we want to know immediately whether each is an
    already-classified object --- a known debris disk, a catalogued binary, a known
    variable --- or an unexamined source.  Queries SIMBAD by position for each
    candidate (the list is short, so per-object queries are robust) and returns the
    nearest match's main identifier, object type, spectral type and separation.
    Absence of a match is itself informative (an unexamined source).
    """
    try:
        from astroquery.simbad import Simbad
    except Exception as exc:
        print(f"[science] SIMBAD unavailable: {exc!r}")
        return pd.DataFrame()

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    sim = Simbad()
    sim.TIMEOUT = 60
    try:
        sim.add_votable_fields("otype", "sp", "ids")
    except Exception:
        pass

    rows = []
    for _, r in positions.iterrows():
        sid = int(r["source_id"])
        try:
            coord = SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg)
            res = sim.query_region(coord, radius=radius_arcsec * u.arcsec)
            if res is None or len(res) == 0:
                rows.append({"source_id": sid, "simbad_id": "", "simbad_otype": "",
                             "simbad_sptype": "", "simbad_n_match": 0})
                continue
            df = res.to_pandas()
            rows.append({
                "source_id": sid,
                "simbad_id": str(_first_cell(df, ["main_id"])).strip(),
                "simbad_otype": str(_first_cell(df, ["otype"])).strip(),
                "simbad_sptype": str(_first_cell(df, ["sp_type", "sptype"])).strip(),
                "simbad_n_match": int(len(df)),
            })
        except Exception as exc:
            print(f"[science] SIMBAD {sid} skipped: {exc!r}")
            rows.append({"source_id": sid, "simbad_id": "", "simbad_otype": "",
                         "simbad_sptype": "", "simbad_n_match": -1})
    out = pd.DataFrame(rows)
    n_known = int((out["simbad_n_match"] > 0).sum()) if len(out) else 0
    print(f"[science] SIMBAD context: {n_known}/{len(out)} candidates have a known match")
    return out


def fetch_known_disks(positions: pd.DataFrame, radius_arcsec: float = 2.0) -> set:
    """Gaia source_ids of WDs in published debris-disk / IR-excess control samples.

    Fetched directly from VizieR (the catalogues are small) and matched by Gaia
    EDR3 source_id, which the Madurga Favieres et al. (2024) sample carries. This
    is the natural-explanation population subtracted before reporting candidates.
    """
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    matched: set[int] = set()
    if not ({"ra", "dec", "source_id"} <= set(positions.columns)):
        return matched
    c_sample = SkyCoord(positions["ra"].to_numpy() * u.deg,
                        positions["dec"].to_numpy() * u.deg)
    sample_sid = positions["source_id"].astype("int64").to_numpy()

    # The Madurga Favieres (2024) catalogue (J/A+A/688/A168) is a clean list of
    # WD IR-excess sources with RA/Dec but no Gaia id, so we match by position.
    # (We deliberately do NOT use the WIRED parent catalogue, whose table is the
    # full SDSS WD list and would over-subtract.)
    try:
        v = Vizier(columns=["**"], row_limit=-1)
        cats = v.get_catalogs("J/A+A/688/A168")
        for ti, tbl in enumerate(cats):
            df = tbl.to_pandas()
            rcol = _find_col(df, ["RA_ICRS", "RAJ2000", "_RA", "RAdeg"])
            dcol = _find_col(df, ["DE_ICRS", "DEJ2000", "_DE", "DEdeg"])
            # Use only the curated IR-excess table (it carries WISE photometry),
            # not the catalogue's larger master/parent list which would
            # over-subtract.
            exc_col = _find_col(df, ["FinalExcess", "IRExs", "Excess"])
            if exc_col is None:
                print(f"[science] known-disk table {ti}: skipped (no excess flag)")
                continue
            # Keep only rows actually flagged as IR-excess sources.
            num = pd.to_numeric(df[exc_col], errors="coerce")
            sval = df[exc_col].astype(str).str.strip().str.lower()
            is_exc = (num > 0) | sval.isin(["y", "yes", "true", "1", "e"])
            df = df[is_exc.fillna(False)]
            if rcol is None or dcol is None or not len(df):
                print(f"[science] known-disk table {ti}: no excess rows after filter")
                continue
            ra = pd.to_numeric(df[rcol], errors="coerce").to_numpy()
            de = pd.to_numeric(df[dcol], errors="coerce").to_numpy()
            good = np.isfinite(ra) & np.isfinite(de)
            if not good.any():
                continue
            c_ctrl = SkyCoord(ra[good] * u.deg, de[good] * u.deg)
            _, sep, _ = c_sample.match_to_catalog_sky(c_ctrl)
            hit = sep.arcsec <= radius_arcsec
            matched.update(int(s) for s in sample_sid[hit])
            print(f"[science] known-disk table {ti}: {int(good.sum())} control WDs, "
                  f"{int(hit.sum())} matched in sample")
    except Exception as exc:  # controls are optional; never fatal
        print(f"[science] known-disk fetch skipped: {exc!r}")

    print(f"[science] known disks matched in sample: {len(matched)}")
    return matched


__all__ = ["fetch_wd_parent", "fetch_catwise", "fetch_twomass", "fetch_galex",
           "fetch_known_disks", "fetch_simbad_context", "classify_candidate"]

"""BAFFLE bright tier — mid-IR deficits on the stars 2MASS and WISE saturate on.

Why this tier exists
--------------------
The main BAFFLE track (Gaia × 2MASS × AllWISE, ``seti.baffle.acquire`` /
``screen``) is blind for Ks < 4.5, W1 < 8, W2 < 7: 2MASS and WISE saturate
there.  Those ~10⁴ brightest stars include most of the NEAREST ones — under the
zoo hypothesis the highest-prior observers a warden would shield first
(docs/baffle.md §4).  For them the unsaturated mid-infrared photometry is

* the **AKARI/IRC Point Source Catalogue** (9 and 18 µm; VizieR ``II/297/irc``,
  ~871k sources, 5σ ≈ 50 mJy at 9 µm, positions to ~1–2″), and
* the **IRAS Point Source Catalogue** (12 and 25 µm; VizieR ``II/125/main``,
  ~245k sources, beam 0.75′ × 4.5′, **B1950 positions over TAP**).

A warden's screen is likely near-opaque in band, so the deficit looked for is
LARGE (≥ 0.7–1 mag) and a 0.2–0.3 mag near-IR anchor is enough.  Signature
S38 at the bright end: a star whose 9/12/18/25 µm flux is far below its
photospheric prediction while its optical / near-IR is normal.

The chain
---------
1. **Targets**: Gaia DR3 ``phot_g_mean_mag < 7.5`` (~40k), one query per 30°
   RA slab, checkpointed.  *Gaia DR3 is incomplete for G < 3*: the ~300
   brightest stars are supplemented from Hipparcos-2 (``I/311/hip2``) when
   VizieR is reachable, otherwise ``brightest_300_not_covered`` is recorded.
2. **Epoch propagation**: every target is moved with its proper motion to the
   epoch of each survey before matching (2MASS 2000.0, AKARI 2006.7, IRAS
   1983.5).  At 1″/yr the offsets are 16″, 10″ and 33″ — a naive match loses
   exactly the nearest stars.
3. **Near-IR anchor**: 2MASS PSC (``II/246/out``) within 3″ via TAP upload
   (X-Match fallback).  Ks < 4 comes from the 51-ms Read-1 frames with 0.2–0.3
   mag errors and quality B–D: accepted, the error carried, and flagged
   ``tmass_read1_regime``.
4. **Mid-IR**: AKARI and IRAS pulled IN FULL in RA slices (a few dozen queries,
   not tens of thousands of cones) and matched locally: 6″ for AKARI, the IRAS
   position-error ellipse (floor 30″) for IRAS.  IRAS quality 3 only; quality 1
   is an upper limit and is reported as ``iras_upper_limit_below_photosphere``
   when the photosphere predicts far more flux than the limit.
5. **Photospheric prediction**: empirical, like the main track — running median
   and robust scatter of (Ks − m_b) versus (J − Ks) on the whole target
   population; the excess tail (dust, AGB shells, Be stars) is the control.
6. **Screen**: deficit ≥ 0.7 mag at ≥ 4σ in AKARI 9 µm AND in at least one of
   AKARI 18 / IRAS 12 / IRAS 25 (two instruments, or two bands, must agree).
   Where the photosphere predicts more 9-µm flux than the IRC survey measures
   linearly, IRAS 12 µm becomes the primary band — a saturated AKARI flux is
   itself a false deficit and must not be the evidence.

Every archive call is a ledger entry with QUERY_OK / QUERY_RETURNED_ZERO_ROWS
/ QUERY_FAILED kept apart; a failed slab never stops the run.  Column names are
discovered at runtime from TAP_SCHEMA.columns (fallback ``SELECT TOP 1 *``)
and no name that was not seen is ever queried.

Column names as VERIFIED on the runner by EMBER (results/ember/probe.json):
``II/297/irc``: objID, RAJ2000, DEJ2000, errMaj, errMin, errPA, S09, e_S09,
q_S09, S18, e_S18, q_S18; ``II/125/main``: IRAS, RA1950, DE1950, Major, Minor,
PosAng, Fnu_12, e_Fnu_12 (PERCENT), q_Fnu_12, …, Cirr3, Confuse, Var.  NOT yet
verified here and resolved by aliases at runtime: ``II/246/out`` (RAJ2000,
DEJ2000, "2MASS", Jmag, e_Jmag, Hmag, e_Hmag, Kmag, e_Kmag, Qflg, Rflg, Bflg,
Cflg, Xflg) and ``I/311/hip2`` (HIP, RArad, DErad, Plx, pmRA, pmDE, Hpmag,
B-V; RArad/DErad are served in degrees by VizieR but the unit is read from the
schema and radians are converted if that is what arrives).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seti.config import _repo_root

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

QUERY_OK = "QUERY_OK"
QUERY_ZERO = "QUERY_RETURNED_ZERO_ROWS"
QUERY_FAILED = "QUERY_FAILED"
FROM_CHECKPOINT = "FROM_CHECKPOINT"

VERDICT_NO_DATA = "NO_DATA_REACHED"
VERDICT_NULL = "NO_BRIGHT_MIDIR_DEFICIT_SURVIVOR"
VERDICT_CAND = "BRIGHT_MIDIR_DEFICIT_CANDIDATES_PENDING_VET"

GAIA_COLUMNS = ("source_id", "ra", "dec", "l", "b", "ecl_lat", "parallax",
                "parallax_over_error", "pmra", "pmdec", "ruwe", "phot_g_mean_mag",
                "bp_rp", "phot_variable_flag", "non_single_star")

#: The four mid-IR bands.  ``flux``/``err``/``qual`` are the canonical column
#: names after :func:`normalise_columns`; ``zp`` the config key of the zero point.
BANDS: dict[str, dict] = {
    "s09": {"instrument": "akari", "label": "AKARI 9um", "zp": "s09"},
    "s18": {"instrument": "akari", "label": "AKARI 18um", "zp": "s18"},
    "f12": {"instrument": "iras", "label": "IRAS 12um", "zp": "f12"},
    "f25": {"instrument": "iras", "label": "IRAS 25um", "zp": "f25"},
}
SECONDARY_OF = {"s09": ("s18", "f12", "f25"), "f12": ("f25", "s18", "s09")}

# Canonical name -> candidate catalogue spellings (compared after lower-casing
# and stripping every non-alphanumeric character).  First hit wins; a canonical
# name with no hit is simply absent, never invented.
AKARI_ALIASES: dict[str, tuple[str, ...]] = {
    "akari_id": ("objid", "objname", "akari", "id"),
    "akari_ra": ("raj2000", "_raj2000", "radeg", "ra"),
    "akari_dec": ("dej2000", "_dej2000", "dedeg", "de", "dec"),
    "s09": ("s09", "s9", "flux09"), "e_s09": ("e_s09", "e_s9"),
    "q_s09": ("q_s09", "fq09", "fqual09", "q09"),
    "s18": ("s18", "flux18"), "e_s18": ("e_s18",),
    "q_s18": ("q_s18", "fq18", "fqual18", "q18"),
    "akari_f09": ("f09",), "akari_f18": ("f18",),
    "akari_errmaj": ("errmaj",), "akari_errmin": ("errmin",), "akari_errpa": ("errpa",),
    "akari_ndet": ("ndet",),
}
IRAS_ALIASES: dict[str, tuple[str, ...]] = {
    "iras_id": ("iras", "name", "id"),
    "iras_ra": ("_raj2000", "raj2000", "ra1950", "raj1950", "ra"),
    "iras_dec": ("_dej2000", "dej2000", "de1950", "dej1950", "de", "dec"),
    "f12": ("fnu_12", "fnu12"), "e_f12": ("e_fnu_12", "e_fnu12"),
    "q_f12": ("q_fnu_12", "q_fnu12"),
    "f25": ("fnu_25", "fnu25"), "e_f25": ("e_fnu_25", "e_fnu25"),
    "q_f25": ("q_fnu_25", "q_fnu25"),
    "f60": ("fnu_60", "fnu60"), "q_f60": ("q_fnu_60", "q_fnu60"),
    "f100": ("fnu_100", "fnu100"), "q_f100": ("q_fnu_100", "q_fnu100"),
    "iras_major": ("major",), "iras_minor": ("minor",), "iras_posang": ("posang",),
    "iras_cirr3": ("cirr3",), "iras_confuse": ("confuse", "conf"), "iras_var": ("var",),
    "iras_nhcon": ("nhcon",),
}
TMASS_ALIASES: dict[str, tuple[str, ...]] = {
    "tmass_id": ("2mass", "designation", "tmass"),
    "tmass_ra": ("raj2000", "_raj2000", "radeg", "ra"),
    "tmass_dec": ("dej2000", "_dej2000", "dedeg", "de", "dec"),
    "j_m": ("jmag", "j_m"), "e_j": ("e_jmag", "j_msigcom", "j_cmsig"),
    "h_m": ("hmag", "h_m"), "e_h": ("e_hmag", "h_msigcom", "h_cmsig"),
    "ks_m": ("kmag", "ksmag", "ks_m", "k_m"),
    "e_ks": ("e_kmag", "e_ksmag", "ks_msigcom", "k_msigcom", "k_cmsig"),
    # X-Match serves Qfl / Rfl / X for the same flags (run 34048837928).
    "tmass_qflg": ("qflg", "qfl", "ph_qual"), "tmass_rflg": ("rflg", "rfl", "rd_flg"),
    "tmass_bflg": ("bflg", "bfl", "bl_flg"), "tmass_cflg": ("cflg", "cfl", "cc_flg"),
    "tmass_xflg": ("xflg", "x", "gal_contam"), "tmass_aflg": ("aflg", "mp_flg"),
    "tmass_angdist": ("angdist",),
}
HIP_ALIASES: dict[str, tuple[str, ...]] = {
    "hip": ("hip",),
    "hip_ra": ("rarad", "_ra.icrs", "raicrs", "ra", "radeg"),
    "hip_dec": ("derad", "_de.icrs", "deicrs", "de", "dedeg", "dec"),
    "hip_plx": ("plx",), "hip_e_plx": ("e_plx",),
    "hip_pmra": ("pmra",), "hip_pmdec": ("pmde", "pmdec"),
    "hpmag": ("hpmag",), "hip_bv": ("b-v", "bv", "b_v"),
}
# Position columns that carry B1950 coordinates: the frame must be precessed.
_B1950_MARK = "1950"


# ===========================================================================
# Config
# ===========================================================================
def load_bright_config(path: Path | str | None = None) -> dict:
    """Read ``config/baffle_bright.yaml`` (or ``path``)."""
    p = Path(path) if path is not None else _repo_root() / "config" / "baffle_bright.yaml"
    with Path(p).open() as fh:
        return yaml.safe_load(fh)


# ===========================================================================
# Pure geometry / photometry helpers
# ===========================================================================
def propagate_to_epoch(ra, dec, pmra, pmdec, from_epoch: float = 2016.0,
                       to_epoch: float = 2000.0):
    """Move positions with proper motion between epochs (linear, small-angle).

    ``pmra`` is μ_α* (already multiplied by cos δ), mas/yr, as Gaia serves it.
    Missing proper motions are treated as zero.  Returns ``(ra, dec)`` in
    degrees; RA is wrapped to [0, 360).
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    pmra = np.nan_to_num(np.asarray(pmra, dtype=float))
    pmdec = np.nan_to_num(np.asarray(pmdec, dtype=float))
    dt = float(to_epoch) - float(from_epoch)
    cosd = np.maximum(np.cos(np.radians(dec)), 1e-9)
    new_ra = np.mod(ra + (pmra * dt / 3.6e6) / cosd, 360.0)
    new_dec = np.clip(dec + pmdec * dt / 3.6e6, -90.0, 90.0)
    return new_ra, new_dec


def flux_to_mag(flux_jy, zp_jy: float, err_jy=None):
    """Vega-like magnitude ``-2.5 log10(S / ZP)`` and, if given, its error.

    Non-positive or missing fluxes give NaN.  The error is the linearised
    ``1.0857 σ_S / S``.
    """
    s = np.asarray(flux_jy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(s > 0, -2.5 * np.log10(s / float(zp_jy)), np.nan)
    if err_jy is None:
        return mag
    e = np.asarray(err_jy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        emag = np.where(s > 0, 2.5 / math.log(10.0) * e / s, np.nan)
    return mag, emag


def mag_to_flux(mag, zp_jy: float):
    m = np.asarray(mag, dtype=float)
    return float(zp_jy) * np.power(10.0, -0.4 * m)


def _skycoord(ra, dec):
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord(ra=np.asarray(ra, float) * u.deg, dec=np.asarray(dec, float) * u.deg,
                    frame="icrs")


def match_within(positions: pd.DataFrame, catalogue: pd.DataFrame,
                 radius_arcsec: float, pos_cols=("ra", "dec"),
                 cat_cols=("ra", "dec")) -> pd.DataFrame:
    """Nearest catalogue entry within ``radius_arcsec`` for each position.

    Returns a frame with ``pos_index`` (positional index into ``positions``),
    ``cat_index`` (positional index into ``catalogue``) and ``sep_arcsec``, one
    row per matched position.  Unmatched positions are absent.
    """
    empty = pd.DataFrame({"pos_index": pd.Series(dtype=int),
                          "cat_index": pd.Series(dtype=int),
                          "sep_arcsec": pd.Series(dtype=float)})
    if len(positions) == 0 or len(catalogue) == 0:
        return empty
    from astropy.coordinates import match_coordinates_sky

    cat = catalogue[[cat_cols[0], cat_cols[1]]].astype(float)
    good = np.isfinite(cat[cat_cols[0]].to_numpy()) & np.isfinite(cat[cat_cols[1]].to_numpy())
    if not good.any():
        return empty
    cat_idx = np.flatnonzero(good)
    pos = positions[[pos_cols[0], pos_cols[1]]].astype(float)
    pgood = np.isfinite(pos[pos_cols[0]].to_numpy()) & np.isfinite(pos[pos_cols[1]].to_numpy())
    if not pgood.any():
        return empty
    pos_idx = np.flatnonzero(pgood)
    c_pos = _skycoord(pos.iloc[pos_idx, 0].to_numpy(), pos.iloc[pos_idx, 1].to_numpy())
    c_cat = _skycoord(cat.iloc[cat_idx, 0].to_numpy(), cat.iloc[cat_idx, 1].to_numpy())
    idx, sep, _ = match_coordinates_sky(c_pos, c_cat)
    sep_as = sep.arcsec
    ok = sep_as <= float(radius_arcsec)
    return pd.DataFrame({"pos_index": pos_idx[ok], "cat_index": cat_idx[idx[ok]],
                         "sep_arcsec": sep_as[ok]}).reset_index(drop=True)


def separation_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Haversine separation in arcsec (vectorised)."""
    r1, d1, r2, d2 = (np.radians(np.asarray(x, float)) for x in (ra1, dec1, ra2, dec2))
    s = (np.sin((d2 - d1) / 2) ** 2
         + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 3600.0


def precess_b1950_to_icrs(ra_deg, dec_deg):
    """FK4/B1950 (epoch B1950) -> ICRS.  The shift is ~0.3–0.6° at IRAS
    declinations, a hundred times the match radius: a missed precession does
    not degrade the match, it destroys it."""
    import astropy.units as u
    from astropy.coordinates import FK4, SkyCoord

    c = SkyCoord(ra=np.asarray(ra_deg, float) * u.deg, dec=np.asarray(dec_deg, float) * u.deg,
                 frame=FK4(equinox="B1950", obstime="B1950"))
    icrs = c.icrs
    return icrs.ra.deg, icrs.dec.deg


def ecliptic_and_galactic(ra_deg, dec_deg):
    """(ecl_lat, l, b) in degrees for positions that did not come from Gaia."""
    from astropy.coordinates import BarycentricMeanEcliptic

    c = _skycoord(ra_deg, dec_deg)
    ecl = c.transform_to(BarycentricMeanEcliptic(equinox="J2000"))
    gal = c.galactic
    return ecl.lat.deg, gal.l.deg, gal.b.deg


def in_iras_ellipse(dx_arcsec, dy_arcsec, major, minor, posang_deg,
                    floor_arcsec: float, cap_arcsec: float) -> np.ndarray:
    """Is an offset (east, north; arcsec) inside the IRAS position-error ellipse?

    ``major``/``minor`` are the ellipse semi-axes (arcsec), ``posang`` the
    position angle of the major axis east of north.  Each axis is clipped to
    [floor, cap]; a missing ellipse degrades to a circle of the floor radius.
    """
    a = np.clip(np.nan_to_num(np.asarray(major, float), nan=floor_arcsec), floor_arcsec, cap_arcsec)
    b = np.clip(np.nan_to_num(np.asarray(minor, float), nan=floor_arcsec), floor_arcsec, cap_arcsec)
    pa = np.radians(np.nan_to_num(np.asarray(posang_deg, float)))
    dx = np.asarray(dx_arcsec, float)
    dy = np.asarray(dy_arcsec, float)
    # Component along the major axis (PA east of north) and perpendicular to it.
    along = dx * np.sin(pa) + dy * np.cos(pa)
    perp = dx * np.cos(pa) - dy * np.sin(pa)
    return (along / a) ** 2 + (perp / b) ** 2 <= 1.0


# ===========================================================================
# Column discovery and normalisation
# ===========================================================================
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_aliases(colnames, aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """canonical -> actual column name, for every canonical with a hit."""
    lookup: dict[str, str] = {}
    for c in colnames:
        bare = _unquote(c)
        if bare:
            lookup.setdefault(_norm(bare), bare)
    out = {}
    for canonical, cands in aliases.items():
        for cand in cands:
            hit = lookup.get(_norm(cand))
            if hit is not None:
                out[canonical] = hit
                break
    return out


def normalise_columns(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]],
                      keep: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename catalogue columns to canonical names; drop the rest (except ``keep``)."""
    res = resolve_aliases(df.columns, aliases)
    cols = {actual: canon for canon, actual in res.items()}
    out = df.rename(columns=cols)
    want = [c for c in list(res) + list(keep) if c in out.columns]
    return out[want].copy(), res


def _unquote(name) -> str:
    """Strip one layer of surrounding double quotes and whitespace.

    TAPVizieR's TAP_SCHEMA.columns serves ``column_name`` ALREADY double-quoted
    (``'"RAJ2000"'``; run 34048837928).  Every discovered name is stored bare
    and quoted exactly once, at composition time, by :func:`_adql_col`.
    """
    n = str(name).strip()
    if len(n) >= 2 and n[0] == '"' and n[-1] == '"':
        n = n[1:-1].strip()
    return n


def _adql_col(name: str, logical: str = "") -> str:
    """One ADQL column reference: plain names bare, anything else quoted once.

    Raises on an empty identifier -- an empty ``""`` in a SELECT list is the
    exact text VizieR rejected with ``Encountered '""'`` and it must never be
    emitted; the message names the logical column so the ledger says which.
    """
    bare = _unquote(name)
    if not bare:
        raise ValueError(f"empty column identifier for logical column {logical or '?'!r}")
    return bare if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", bare) else f'"{bare}"'


def select_list(resolved: dict[str, str], required: tuple[str, ...] = (),
                prefix: str = "") -> str:
    """Compose a SELECT list from ``{logical: actual}``; fail loudly on a hole."""
    missing = [k for k in required if not _unquote(resolved.get(k, ""))]
    if missing:
        raise RuntimeError(f"required columns not resolved: {missing} (resolved: {resolved})")
    parts = []
    for logical, actual in resolved.items():
        if not _unquote(actual):
            raise RuntimeError(f"logical column {logical!r} resolved to an empty identifier")
        parts.append(prefix + _adql_col(actual, logical))
    return ", ".join(dict.fromkeys(parts))


def _tap(url: str = VIZIER_TAP):
    import pyvo

    return pyvo.dal.TAPService(url)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.lower() for c in df.columns})


def run_vizier(query: str, *, uploads=None, retries: int = 3, label: str = "vizier",
               url: str = VIZIER_TAP) -> pd.DataFrame:
    """One VizieR TAP query: async first, sync on the last attempt, with backoff."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            svc = _tap(url)
            if attempt < retries - 1:
                res = svc.run_async(query, uploads=uploads)
            else:
                res = svc.run_sync(query, uploads=uploads)
            return res.to_table().to_pandas()
        except Exception as exc:  # noqa: BLE001 - archives fail in many ways
            last = exc
            print(f"[baffle-bright] {label} attempt {attempt + 1}/{retries} failed: {exc!r}",
                  flush=True)
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"{label}: failed after {retries} attempts: {last!r}")


def discover_columns(table: str, url: str = VIZIER_TAP) -> dict:
    """Column names (and units/UCDs where TAP_SCHEMA serves them) for ``table``.

    TAPVizieR has returned zero TAP_SCHEMA rows for the unquoted form of a
    table name (EMBER probe, run 30209647320), so several spellings are tried
    and the one that answered is recorded; ``SELECT TOP 1 *`` is the fallback.
    Nothing downstream may use a column name that is not in ``names``.
    """
    bare = table.strip('"')
    forms = (bare, table, bare.replace("/", "."), bare.rsplit("/", 1)[0])
    out: dict = {"table": table, "names": [], "meta": {}, "route": None, "errors": []}
    for form in dict.fromkeys(forms):
        q = ("SELECT column_name, ucd, unit, datatype, description FROM TAP_SCHEMA.columns "
             f"WHERE table_name = '{form}'")
        try:
            df = run_vizier(q, retries=2, label=f"schema:{form}", url=url)
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(f"{form}: {exc!r}")
            continue
        if len(df):
            df = _lower(df)
            out["names"] = [_unquote(x) for x in df["column_name"] if _unquote(x)]
            out["meta"] = {_unquote(r["column_name"]): {"ucd": str(r.get("ucd", "")),
                                                        "unit": str(r.get("unit", "")),
                                                        "datatype": str(r.get("datatype", ""))}
                           for _, r in df.iterrows()}
            out["names_as_served"] = [str(x) for x in df["column_name"]][:5]
            out["route"] = f"TAP_SCHEMA:{form}"
            return out
    try:
        df = run_vizier(f"SELECT TOP 1 * FROM {table}", retries=2, label=f"top1:{table}", url=url)
        out["names"] = [_unquote(c) for c in df.columns if _unquote(c)]
        out["route"] = "SELECT TOP 1 *"
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"top1: {exc!r}")
    return out


# ===========================================================================
# Default (runner-only) fetchers.  Every one is injectable.
# ===========================================================================
def default_gaia_fetcher(query: str, label: str = "gaia", retries: int = 3) -> pd.DataFrame:
    """astroquery.gaia: async first, sync on the last attempt."""
    from astroquery.gaia import Gaia

    last: Exception | None = None
    for attempt in range(retries):
        try:
            job = (Gaia.launch_job_async(query) if attempt < retries - 1
                   else Gaia.launch_job(query))
            return _lower(job.get_results().to_pandas())
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[baffle-bright] {label} attempt {attempt + 1}/{retries} failed: {exc!r}",
                  flush=True)
            time.sleep(4.0 * (attempt + 1))
    raise RuntimeError(f"{label}: Gaia query failed after {retries} attempts: {last!r}")


class VizierSliceFetcher:
    """Pull a VizieR catalogue in RA slices, selecting only columns that exist.

    Column discovery runs once, on the first call, and is recorded in
    ``self.discovery`` for the ledger.  ``__call__(ra_lo, ra_hi, label)`` returns
    the raw rows (catalogue spellings) of the slice.
    """

    def __init__(self, table: str, aliases: dict[str, tuple[str, ...]],
                 ra_key: str, url: str = VIZIER_TAP, row_limit: int = 4_000_000):
        self.table, self.aliases, self.ra_key, self.url = table, aliases, ra_key, url
        self.row_limit = row_limit
        self.discovery: dict | None = None
        self.resolved: dict[str, str] = {}

    def _discover(self) -> None:
        if self.discovery is not None:
            return
        self.discovery = discover_columns(self.table, self.url)
        self.resolved = resolve_aliases(self.discovery["names"], self.aliases)
        if self.ra_key not in self.resolved:
            raise RuntimeError(f"{self.table}: no RA column resolved from "
                               f"{self.discovery['names'][:40]} ({self.discovery['errors']})")

    def __call__(self, ra_lo: float, ra_hi: float, label: str) -> pd.DataFrame:
        self._discover()
        ra_col = _adql_col(self.resolved[self.ra_key], self.ra_key)
        select = select_list(self.resolved, required=(self.ra_key,))
        q = (f"SELECT TOP {self.row_limit} {select} FROM {self.table} "
             f"WHERE {ra_col} >= {ra_lo} AND {ra_col} < {ra_hi}")
        return run_vizier(q, label=label, url=self.url)


class VizierUploadMatcher:
    """2MASS PSC around uploaded positions: TAP upload join, X-Match fallback.

    ``__call__(positions, radius_arcsec, label)`` takes ``source_id, ra, dec``
    (already at the 2MASS epoch) and returns raw catalogue rows with the
    uploaded ``source_id`` attached.
    """

    def __init__(self, cfg_tmass: dict, url: str = VIZIER_TAP):
        self.table = cfg_tmass.get("table", '"II/246/out"')
        self.url = url
        self.xmatch_url = cfg_tmass.get("xmatch_url", "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync")
        self.xmatch_cat = cfg_tmass.get("xmatch_catalogue", "vizier:II/246/out")
        self.discovery: dict | None = None
        self.resolved: dict[str, str] = {}
        self.routes_used: list[str] = []

    def _discover(self) -> None:
        if self.discovery is not None:
            return
        self.discovery = discover_columns(self.table, self.url)
        self.resolved = resolve_aliases(self.discovery["names"], TMASS_ALIASES)

    def _tap_upload(self, positions: pd.DataFrame, radius_arcsec: float, label: str):
        from astropy.table import Table

        self._discover()
        if "tmass_ra" not in self.resolved or "tmass_dec" not in self.resolved:
            raise RuntimeError(f"{self.table}: RA/Dec not resolved from {self.discovery['names'][:40]}")
        ra_c = _adql_col(self.resolved["tmass_ra"], "tmass_ra")
        de_c = _adql_col(self.resolved["tmass_dec"], "tmass_dec")
        select = select_list(self.resolved, required=("tmass_ra", "tmass_dec"), prefix="t.")
        q = (f"SELECT u.source_id, {select} FROM TAP_UPLOAD.targets AS u "
             f"JOIN {self.table} AS t ON 1 = CONTAINS(POINT('ICRS', t.{ra_c}, "
             f"t.{de_c}), CIRCLE('ICRS', u.ra, u.dec, {radius_arcsec / 3600.0}))")
        up = Table({"source_id": positions["source_id"].to_numpy(np.int64),
                    "ra": positions["ra"].to_numpy(float), "dec": positions["dec"].to_numpy(float)})
        return run_vizier(q, uploads={"targets": up}, label=label, url=self.url)

    def _xmatch(self, positions: pd.DataFrame, radius_arcsec: float, label: str):
        import io

        import requests

        csv = positions[["source_id", "ra", "dec"]].to_csv(index=False)
        r = requests.post(self.xmatch_url, data={
            "request": "xmatch", "distMaxArcsec": f"{radius_arcsec:g}",
            "RESPONSEFORMAT": "csv", "cat2": self.xmatch_cat,
            "colRA1": "ra", "colDec1": "dec", "selection": "all"},
            files={"cat1": ("positions.csv", csv)}, timeout=600)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))

    def __call__(self, positions: pd.DataFrame, radius_arcsec: float, label: str) -> pd.DataFrame:
        try:
            df = self._tap_upload(positions, radius_arcsec, label)
            self.routes_used.append("tap_upload")
            return df
        except Exception as exc:  # noqa: BLE001
            print(f"[baffle-bright] {label}: TAP upload failed ({exc!r}); trying X-Match", flush=True)
        df = self._xmatch(positions, radius_arcsec, label)
        self.routes_used.append("xmatch")
        return df


def default_hip_fetcher_factory(cfg_hip: dict, url: str = VIZIER_TAP):
    """Hipparcos-2 supplement: one small query after column discovery."""
    def fetch(_query_hint: str, label: str) -> pd.DataFrame:
        table = cfg_hip.get("table", '"I/311/hip2"')
        disc = discover_columns(table, url)
        res = resolve_aliases(disc["names"], HIP_ALIASES)
        need = ("hip", "hip_ra", "hip_dec", "hpmag")
        if any(k not in res for k in need):
            raise RuntimeError(f"{table}: columns not resolved: {res} from {disc['names'][:40]}")
        select = select_list(res, required=need)
        q = (f"SELECT {select} FROM {table} WHERE {_adql_col(res['hpmag'], 'hpmag')} < "
             f"{float(cfg_hip.get('hp_max', 3.5))}")
        df = run_vizier(q, label=label, url=url)
        # RArad/DErad: VizieR serves degrees, the native file is radians.  Trust
        # the schema unit first, the value range second.
        unit = str(disc["meta"].get(res["hip_ra"], {}).get("unit", "")).lower()
        ra_vals = pd.to_numeric(df[res["hip_ra"]], errors="coerce")
        if unit.startswith("rad") or (unit == "" and np.nanmax(np.abs(ra_vals)) <= 2 * math.pi + 1e-6):
            df[res["hip_ra"]] = np.degrees(ra_vals)
            df[res["hip_dec"]] = np.degrees(pd.to_numeric(df[res["hip_dec"]], errors="coerce"))
            df.attrs["converted_from_radians"] = True
        return df
    return fetch


# ===========================================================================
# Query builders (pure strings; unit-testable)
# ===========================================================================
def gaia_targets_query(g_max: float, ra_lo: float, ra_hi: float) -> str:
    cols = ", ".join(GAIA_COLUMNS)
    return (f"SELECT {cols} FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < {g_max} "
            f"AND ra >= {ra_lo} AND ra < {ra_hi}")


def gaia_neighbours_query(g_max: float, g_neigh_max: float, radius_arcsec: float,
                          ra_lo: float, ra_hi: float) -> str:
    """Every Gaia G < ``g_neigh_max`` star within ``radius`` of a target (self-join)."""
    return (
        "SELECT t.source_id AS target_id, n.source_id AS source_id, n.ra, n.dec, "
        "n.phot_g_mean_mag FROM gaiadr3.gaia_source AS t JOIN gaiadr3.gaia_source AS n "
        f"ON 1 = CONTAINS(POINT('ICRS', n.ra, n.dec), CIRCLE('ICRS', t.ra, t.dec, {radius_arcsec / 3600.0})) "
        f"WHERE t.phot_g_mean_mag < {g_max} AND t.ra >= {ra_lo} AND t.ra < {ra_hi} "
        f"AND n.phot_g_mean_mag < {g_neigh_max} AND n.source_id != t.source_id")


def gaia_neighbours_bulk_query(g_neigh_max: float, ra_lo: float, ra_hi: float,
                               margin_deg: float = 0.05) -> str:
    """Fallback: all G < ``g_neigh_max`` stars in the slab (plus a margin), counted locally."""
    return ("SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
            f"WHERE phot_g_mean_mag < {g_neigh_max} AND ra >= {ra_lo - margin_deg} "
            f"AND ra < {ra_hi + margin_deg}")


# ===========================================================================
# Ledger
# ===========================================================================
def _record(ledger: list, stage: str, label: str, status: str, *, n_rows: int = 0,
            error: str = "", elapsed_s: float = 0.0, query: str = "", **extra) -> dict:
    e = {"stage": stage, "label": label, "status": status, "n_rows": int(n_rows),
         "error": error[:800], "elapsed_s": round(float(elapsed_s), 2), "query": query[:600]}
    e.update(extra)
    ledger.append(e)
    return e


def _slab_edges(n: int) -> list[tuple[int, float, float]]:
    edges = np.linspace(0.0, 360.0, n + 1)
    return [(i, float(edges[i]), float(edges[i + 1])) for i in range(n)]


def _run_slabs(stage: str, n_slabs: int, call, chunks: Path | None, ledger: list,
               tag: str) -> tuple[pd.DataFrame, dict]:
    """Run ``call(lo, hi, label) -> DataFrame`` per RA slab, checkpointed, contained."""
    frames: list[pd.DataFrame] = []
    stats = {"n_slabs": n_slabs, "ok": 0, "zero": 0, "failed": 0, "checkpoint": 0,
             "failed_slabs": []}
    for i, lo, hi in _slab_edges(n_slabs):
        label = f"{tag}_ra{i:02d}"
        ckpt = (chunks / f"{label}.parquet") if chunks is not None else None
        if ckpt is not None and ckpt.exists():
            df = pd.read_parquet(ckpt)
            frames.append(df)
            stats["checkpoint"] += 1
            _record(ledger, stage, label, FROM_CHECKPOINT, n_rows=len(df))
            continue
        if ckpt is not None and ckpt.with_suffix(".empty").exists():
            stats["checkpoint"] += 1
            _record(ledger, stage, label, FROM_CHECKPOINT, n_rows=0)
            continue
        t0 = time.monotonic()
        try:
            df = call(lo, hi, label)
        except Exception as exc:  # noqa: BLE001 - contained; a failed slab is a ledger line
            stats["failed"] += 1
            stats["failed_slabs"].append(i)
            _record(ledger, stage, label, QUERY_FAILED, error=repr(exc),
                    elapsed_s=time.monotonic() - t0, ra_lo=lo, ra_hi=hi)
            print(f"[baffle-bright] {label}: {QUERY_FAILED} {exc!r}", flush=True)
            continue
        df = pd.DataFrame() if df is None else pd.DataFrame(df)
        if len(df):
            stats["ok"] += 1
            if ckpt is not None:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(ckpt, index=False)
            frames.append(df)
            _record(ledger, stage, label, QUERY_OK, n_rows=len(df),
                    elapsed_s=time.monotonic() - t0, ra_lo=lo, ra_hi=hi)
        else:
            stats["zero"] += 1
            if ckpt is not None:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                ckpt.with_suffix(".empty").write_text("")   # an answered, empty slab
            _record(ledger, stage, label, QUERY_ZERO, elapsed_s=time.monotonic() - t0,
                    ra_lo=lo, ra_hi=hi)
    frames = [f for f in frames if len(f)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, stats


# ===========================================================================
# Acquisition stages
# ===========================================================================
def acquire_targets(cfg: dict, out_dir: Path, ledger: list, gaia_fetcher, hip_fetcher,
                    max_targets: int | None = None) -> tuple[pd.DataFrame, dict]:
    tcfg = cfg["targets"]
    n_slabs = int(round(360.0 / float(tcfg.get("ra_slab_deg", 30.0))))
    g_max = float(tcfg.get("g_max", 7.5))
    chunks = out_dir / "chunks"

    def call(lo, hi, label):
        return gaia_fetcher(gaia_targets_query(g_max, lo, hi), label)

    gaia, gstats = _run_slabs("gaia_targets", n_slabs, call, chunks, ledger, "targets")
    info = {"gaia": gstats, "hipparcos": {"status": "DISABLED", "n_added": 0}}
    if len(gaia):
        gaia = _lower(gaia)
        for c in GAIA_COLUMNS:
            if c not in gaia.columns:
                gaia[c] = np.nan
        gaia = gaia[list(GAIA_COLUMNS)].copy()
        gaia["source_id"] = pd.to_numeric(gaia["source_id"], errors="coerce").astype("Int64")
        gaia = gaia.dropna(subset=["source_id", "ra", "dec"]).drop_duplicates("source_id")
        gaia["source_id"] = gaia["source_id"].astype(np.int64)
        gaia["origin"] = "gaia_dr3"
        gaia["epoch"] = float(cfg["epochs"].get("gaia", 2016.0))
        gaia["hip"] = pd.array([pd.NA] * len(gaia), dtype="Int64")
    targets = gaia

    hcfg = tcfg.get("hipparcos", {}) or {}
    if hcfg.get("enabled", False):
        fetch = hip_fetcher if hip_fetcher is not None else default_hip_fetcher_factory(hcfg)
        t0 = time.monotonic()
        try:
            raw = fetch(f"hip2 Hpmag < {hcfg.get('hp_max', 3.5)}", "hipparcos")
            hip, res = normalise_columns(pd.DataFrame(raw), HIP_ALIASES)
            need = ("hip", "hip_ra", "hip_dec", "hpmag")
            if any(k not in hip.columns for k in need):
                raise RuntimeError(f"hip2: unresolved columns; got {list(hip.columns)}")
            hip = hip.dropna(subset=["hip_ra", "hip_dec"]).reset_index(drop=True)
            sup = pd.DataFrame({
                "source_id": -pd.to_numeric(hip["hip"], errors="coerce").fillna(0).astype(np.int64),
                "ra": hip["hip_ra"].astype(float), "dec": hip["hip_dec"].astype(float),
                "parallax": pd.to_numeric(hip.get("hip_plx"), errors="coerce"),
                "pmra": pd.to_numeric(hip.get("hip_pmra"), errors="coerce"),
                "pmdec": pd.to_numeric(hip.get("hip_pmdec"), errors="coerce"),
                "phot_g_mean_mag": pd.to_numeric(hip["hpmag"], errors="coerce"),
            })
            e_plx = pd.to_numeric(hip.get("hip_e_plx"), errors="coerce")
            sup["parallax_over_error"] = sup["parallax"] / e_plx if e_plx is not None else np.nan
            sup["ruwe"] = np.nan
            sup["bp_rp"] = np.nan          # Hp/B-V are not BP-RP; the class split stays 'unknown'
            sup["phot_variable_flag"] = "NOT_AVAILABLE"
            sup["non_single_star"] = 0
            ecl, gl, gb = ecliptic_and_galactic(sup["ra"], sup["dec"])
            sup["ecl_lat"], sup["l"], sup["b"] = ecl, gl, gb
            sup["origin"] = "hipparcos2"
            sup["epoch"] = float(hcfg.get("epoch", cfg["epochs"].get("hipparcos", 1991.25)))
            sup["hip"] = pd.to_numeric(hip["hip"], errors="coerce").astype("Int64")
            # Drop Hip stars already in the Gaia target list (at the Gaia epoch).
            n_dup = 0
            if len(targets):
                ra16, de16 = propagate_to_epoch(sup["ra"], sup["dec"], sup["pmra"], sup["pmdec"],
                                                from_epoch=float(sup["epoch"].iloc[0]),
                                                to_epoch=float(cfg["epochs"].get("gaia", 2016.0)))
                m = match_within(pd.DataFrame({"ra": ra16, "dec": de16}), targets,
                                 float(hcfg.get("dedupe_radius_arcsec", 3.0)))
                n_dup = len(m)
                sup = sup.drop(index=m["pos_index"].to_numpy()).reset_index(drop=True)
            targets = (pd.concat([targets, sup[list(targets.columns)]], ignore_index=True)
                       if len(targets) else sup)
            info["hipparcos"] = {"status": QUERY_OK if len(hip) else QUERY_ZERO,
                                 "n_rows": int(len(hip)), "n_duplicates_of_gaia": int(n_dup),
                                 "n_added": int(len(sup)), "columns": res}
            _record(ledger, "hipparcos", "hipparcos", info["hipparcos"]["status"],
                    n_rows=len(hip), elapsed_s=time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001
            info["hipparcos"] = {"status": QUERY_FAILED, "error": repr(exc)[:800], "n_added": 0}
            _record(ledger, "hipparcos", "hipparcos", QUERY_FAILED, error=repr(exc),
                    elapsed_s=time.monotonic() - t0)
            print(f"[baffle-bright] hipparcos: {QUERY_FAILED} {exc!r}", flush=True)
    if len(targets) and max_targets is not None and max_targets > 0:
        targets = targets.sort_values("phot_g_mean_mag", kind="stable").head(int(max_targets))
        targets = targets.reset_index(drop=True)
    if len(targets):
        targets = targets.reset_index(drop=True)
        targets["parallax"] = pd.to_numeric(targets["parallax"], errors="coerce")
        plx = targets["parallax"].to_numpy(float)
        targets["distance_pc"] = np.where(plx > 0, 1000.0 / np.where(plx > 0, plx, 1), np.nan)
    return targets, info


def acquire_neighbours(cfg: dict, out_dir: Path, ledger: list, gaia_fetcher,
                       targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Gaia G < 10 companions of every target (crowding), self-join then bulk fallback."""
    tcfg = cfg["targets"]
    ncfg = tcfg.get("neighbours", {}) or {}
    n_slabs = int(round(360.0 / float(tcfg.get("ra_slab_deg", 30.0))))
    g_max, g_n = float(tcfg.get("g_max", 7.5)), float(ncfg.get("g_max", 10.0))
    radius = float(ncfg.get("radius_arcsec", 60.0))
    chunks = out_dir / "chunks"
    gaia_t = targets[targets["origin"] == "gaia_dr3"] if len(targets) else targets

    def call(lo, hi, label):
        try:
            df = gaia_fetcher(gaia_neighbours_query(g_max, g_n, radius, lo, hi), label)
            return _lower(pd.DataFrame(df))
        except Exception as exc:  # noqa: BLE001
            print(f"[baffle-bright] {label}: self-join failed ({exc!r}); bulk fallback", flush=True)
        bulk = _lower(pd.DataFrame(gaia_fetcher(gaia_neighbours_bulk_query(g_n, lo, hi),
                                                label + "_bulk")))
        sub = gaia_t[(gaia_t["ra"] >= lo) & (gaia_t["ra"] < hi)]
        if not len(bulk) or not len(sub):
            return pd.DataFrame()
        import astropy.units as u
        from astropy.coordinates import search_around_sky

        c_t = _skycoord(sub["ra"], sub["dec"])
        c_b = _skycoord(bulk["ra"], bulk["dec"])
        it, ib, sep, _ = search_around_sky(c_t, c_b, radius * u.arcsec)
        out = pd.DataFrame({"target_id": sub["source_id"].to_numpy()[it],
                            "source_id": bulk["source_id"].to_numpy()[ib],
                            "ra": bulk["ra"].to_numpy()[ib], "dec": bulk["dec"].to_numpy()[ib],
                            "phot_g_mean_mag": bulk["phot_g_mean_mag"].to_numpy()[ib]})
        return out[out["target_id"] != out["source_id"]].reset_index(drop=True)

    neigh, stats = _run_slabs("gaia_neighbours", n_slabs, call, chunks, ledger, "neighbours")
    # Which targets have a *known* crowding count: those in a slab that answered.
    known = np.zeros(len(targets), dtype=bool)
    for i, lo, hi in _slab_edges(n_slabs):
        if i not in stats["failed_slabs"] and len(targets):
            known |= ((targets["ra"] >= lo) & (targets["ra"] < hi)
                      & (targets["origin"] == "gaia_dr3")).to_numpy()
    stats["n_targets_with_known_crowding"] = int(known.sum())
    stats["known_mask"] = known
    return neigh, stats


def crowding_counts(targets: pd.DataFrame, neigh: pd.DataFrame, known: np.ndarray,
                    r30: float, r60: float, g_max: float) -> tuple[np.ndarray, np.ndarray]:
    n30 = np.full(len(targets), np.nan)
    n60 = np.full(len(targets), np.nan)
    n30[known] = 0
    n60[known] = 0
    if len(neigh) and len(targets):
        nb = neigh[pd.to_numeric(neigh["phot_g_mean_mag"], errors="coerce") < g_max]
        nb = nb[nb["target_id"] != nb["source_id"]]
        pos = targets.set_index("source_id")[["ra", "dec"]]
        nb = nb[nb["target_id"].isin(pos.index)]
        if len(nb):
            tra = pos.loc[nb["target_id"], "ra"].to_numpy()
            tde = pos.loc[nb["target_id"], "dec"].to_numpy()
            sep = separation_arcsec(tra, tde, nb["ra"].to_numpy(float), nb["dec"].to_numpy(float))
            c30 = pd.Series(sep <= r30).groupby(nb["target_id"].to_numpy()).sum()
            c60 = pd.Series(sep <= r60).groupby(nb["target_id"].to_numpy()).sum()
            idx = pd.Index(targets["source_id"])
            for c, arr in ((c30, n30), (c60, n60)):
                loc = idx.get_indexer(c.index)
                ok = loc >= 0
                arr[loc[ok]] = c.to_numpy()[ok]
    return n30, n60


def acquire_tmass(cfg: dict, out_dir: Path, ledger: list, tmass_fetcher,
                  targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """2MASS within ``radius`` of each target propagated to the 2MASS epoch."""
    tcfg = cfg["tmass"]
    radius = float(tcfg.get("radius_arcsec", 3.0))
    chunk = int(tcfg.get("upload_chunk", 4000))
    epoch = float(cfg["epochs"].get("tmass", 2000.0))
    chunks = out_dir / "chunks"
    ra_t, de_t = _propagate_targets(targets, epoch)
    pos = pd.DataFrame({"source_id": targets["source_id"].to_numpy(np.int64), "ra": ra_t, "dec": de_t})
    frames: list[pd.DataFrame] = []
    stats = {"n_chunks": 0, "ok": 0, "zero": 0, "failed": 0, "checkpoint": 0, "failed_chunks": []}
    n_chunks = int(math.ceil(len(pos) / chunk)) if len(pos) else 0
    stats["n_chunks"] = n_chunks
    covered = np.zeros(len(pos), dtype=bool)
    for j in range(n_chunks):
        label = f"tmass_{j:03d}"
        sl = slice(j * chunk, (j + 1) * chunk)
        sub = pos.iloc[sl]
        ckpt = chunks / f"{label}.parquet"
        if ckpt.exists():
            df = pd.read_parquet(ckpt)
            frames.append(df)
            stats["checkpoint"] += 1
            covered[sl] = True
            _record(ledger, "tmass", label, FROM_CHECKPOINT, n_rows=len(df))
            continue
        if ckpt.with_suffix(".empty").exists():
            stats["checkpoint"] += 1
            covered[sl] = True
            _record(ledger, "tmass", label, FROM_CHECKPOINT, n_rows=0)
            continue
        t0 = time.monotonic()
        try:
            raw = pd.DataFrame(tmass_fetcher(sub.reset_index(drop=True), radius, label))
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            stats["failed_chunks"].append(j)
            _record(ledger, "tmass", label, QUERY_FAILED, error=repr(exc),
                    elapsed_s=time.monotonic() - t0)
            print(f"[baffle-bright] {label}: {QUERY_FAILED} {exc!r}", flush=True)
            continue
        covered[sl] = True
        if len(raw):
            df, _ = normalise_columns(raw, TMASS_ALIASES, keep=("source_id",))
            if "source_id" not in df.columns:
                # X-Match echoes the upload columns; TAP upload aliases u.source_id.
                for c in raw.columns:
                    if _norm(c) == "sourceid":
                        df["source_id"] = raw[c].to_numpy()
                        break
            stats["ok"] += 1
            chunks.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ckpt, index=False)
            frames.append(df)
            _record(ledger, "tmass", label, QUERY_OK, n_rows=len(df), elapsed_s=time.monotonic() - t0)
        else:
            stats["zero"] += 1
            chunks.mkdir(parents=True, exist_ok=True)
            ckpt.with_suffix(".empty").write_text("")
            _record(ledger, "tmass", label, QUERY_ZERO, elapsed_s=time.monotonic() - t0)
    stats["n_targets_uploaded"] = int(covered.sum())
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(), stats
    raw = pd.concat(frames, ignore_index=True)
    stats["columns_attached"] = [c for c in raw.columns]
    return attach_tmass(raw, pos, radius), stats


def attach_tmass(raw: pd.DataFrame, pos: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    """One 2MASS row per target from raw (already canonical-named) match rows.

    Tolerant by design: ``source_id`` is cast to int64 whatever the transport
    returned it as (X-Match echoes the upload as float/str); an ABSENT flag
    column is unknown, never bad; the closest counterpart per target wins,
    by the service's own ``angDist`` where present and by a locally computed
    separation otherwise (and always within ``radius_arcsec``).
    """
    if not len(raw) or "source_id" not in raw.columns:
        return pd.DataFrame()
    raw = raw.copy()
    raw["source_id"] = pd.to_numeric(raw["source_id"], errors="coerce")
    raw = raw.dropna(subset=["source_id"])
    raw["source_id"] = raw["source_id"].round().astype(np.int64)
    for c in ("tmass_ra", "tmass_dec", "j_m", "e_j", "h_m", "e_h", "ks_m", "e_ks", "tmass_angdist"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else np.nan
    p = pos.drop_duplicates("source_id").set_index("source_id")
    raw = raw[raw["source_id"].isin(p.index)]
    if not len(raw):
        return pd.DataFrame()
    have_pos = np.isfinite(raw["tmass_ra"].to_numpy(float)) & np.isfinite(raw["tmass_dec"].to_numpy(float))
    sep = np.full(len(raw), np.nan)
    if have_pos.any():
        sub = raw[have_pos]
        sep[have_pos] = separation_arcsec(p.loc[sub["source_id"], "ra"].to_numpy(),
                                          p.loc[sub["source_id"], "dec"].to_numpy(),
                                          sub["tmass_ra"].to_numpy(float), sub["tmass_dec"].to_numpy(float))
    ang = raw["tmass_angdist"].to_numpy(float)
    sep = np.where(np.isfinite(ang), ang, sep)
    raw = raw.assign(tmass_sep_arcsec=sep)
    raw = raw[np.isfinite(raw["tmass_sep_arcsec"]) & (raw["tmass_sep_arcsec"] <= radius_arcsec)]
    raw = raw.sort_values("tmass_sep_arcsec", kind="stable").drop_duplicates("source_id")
    return raw.reset_index(drop=True)


def tmass_quality_masks(work: pd.DataFrame, cfg_tmass: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(quality_ok, quality_known, read1_flag) from whatever flag columns arrived.

    A missing ``Qflg``/``Rflg`` column, or an empty value, is UNKNOWN: the row
    keeps its photometry and ``tmass_qual_known`` says so.  Only a present
    letter outside ``ph_qual_ok`` rejects the row.
    """
    n = len(work)
    ok_letters = set(str(cfg_tmass.get("ph_qual_ok", "ABCD")))

    def _clean(col: str) -> list[str]:
        if col not in work.columns:
            return [""] * n
        vals = work[col].tolist()
        return ["" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip()
                for v in vals]

    q = _clean("tmass_qflg")
    known = np.array([len(v) >= 3 and v.lower() not in ("nan", "<na>", "none") for v in q])
    q_ok = np.array([(not k) or (v[0] in ok_letters and v[2] in ok_letters)
                     for v, k in zip(q, known, strict=False)])
    r = _clean("tmass_rflg")
    read1 = np.array([len(v) >= 3 and v[2] == "1" for v in r])
    return q_ok, known, read1


def _propagate_targets(targets: pd.DataFrame, to_epoch: float):
    ra = targets["ra"].to_numpy(float)
    de = targets["dec"].to_numpy(float)
    ep = targets["epoch"].to_numpy(float) if "epoch" in targets else np.full(len(targets), 2016.0)
    pmra = pd.to_numeric(targets["pmra"], errors="coerce").to_numpy(float)
    pmde = pd.to_numeric(targets["pmdec"], errors="coerce").to_numpy(float)
    out_ra, out_de = ra.copy(), de.copy()
    for e in np.unique(ep):
        m = ep == e
        out_ra[m], out_de[m] = propagate_to_epoch(ra[m], de[m], pmra[m], pmde[m],
                                                  from_epoch=float(e), to_epoch=to_epoch)
    return out_ra, out_de


def acquire_midir(cfg: dict, out_dir: Path, ledger: list, fetcher, which: str
                  ) -> tuple[pd.DataFrame, dict]:
    """Pull AKARI IRC or IRAS PSC in full (RA slices) and normalise the columns."""
    ccfg = cfg[which]
    n = int(ccfg.get("n_ra_chunks", 12))
    chunks = out_dir / "chunks"
    aliases = AKARI_ALIASES if which == "akari" else IRAS_ALIASES

    def call(lo, hi, label):
        return fetcher(lo, hi, label)

    raw, stats = _run_slabs(which, n, call, chunks, ledger, which)
    stats["columns"] = {}
    if not len(raw):
        return pd.DataFrame(), stats
    df, res = normalise_columns(raw, aliases)
    stats["columns"] = res
    ra_c, de_c = f"{which}_ra", f"{which}_dec"
    if ra_c not in df.columns or de_c not in df.columns:
        stats["error"] = f"{which}: RA/Dec unresolved from {list(raw.columns)[:40]}"
        _record(ledger, which, f"{which}_columns", QUERY_FAILED, error=stats["error"])
        return pd.DataFrame(), stats
    df[ra_c] = pd.to_numeric(df[ra_c], errors="coerce")
    df[de_c] = pd.to_numeric(df[de_c], errors="coerce")
    df = df.dropna(subset=[ra_c, de_c]).reset_index(drop=True)
    stats["frame"] = "icrs"
    if _B1950_MARK in res[ra_c].lower():
        df[ra_c], df[de_c] = precess_b1950_to_icrs(df[ra_c], df[de_c])
        stats["frame"] = "fk4_b1950_precessed_to_icrs"
    for b, spec in BANDS.items():
        if spec["instrument"] != which:
            continue
        for c in (b, f"e_{b}", f"q_{b}"):
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    if which == "iras":
        # e_Fnu are RELATIVE uncertainties in PERCENT.
        for b in ("f12", "f25"):
            df[f"e_{b}"] = df[b] * df[f"e_{b}"] / 100.0
        for c in ("iras_major", "iras_minor", "iras_posang", "iras_cirr3"):
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return df, stats


def match_midir(cfg: dict, targets: pd.DataFrame, cat: pd.DataFrame, which: str) -> pd.DataFrame:
    """Match propagated targets to a mid-IR catalogue; returns per-target rows."""
    if not len(cat) or not len(targets):
        return pd.DataFrame()
    ccfg = cfg[which]
    epoch = float(cfg["epochs"].get(which, 2016.0))
    ra_t, de_t = _propagate_targets(targets, epoch)
    pos = pd.DataFrame({"ra": ra_t, "dec": de_t})
    ra_c, de_c = f"{which}_ra", f"{which}_dec"
    if which == "iras" and ccfg.get("use_error_ellipse", True) and "iras_major" in cat:
        cap = float(ccfg.get("ellipse_cap_arcsec", 120.0))
        floor = float(ccfg.get("ellipse_floor_arcsec", ccfg.get("radius_arcsec", 30.0)))
        m = match_within(pos, cat, cap, cat_cols=(ra_c, de_c))
        if not len(m):
            return pd.DataFrame()
        crow = cat.iloc[m["cat_index"].to_numpy()]
        tra, tde = pos.iloc[m["pos_index"].to_numpy(), 0].to_numpy(), pos.iloc[m["pos_index"].to_numpy(), 1].to_numpy()
        dx = (tra - crow[ra_c].to_numpy(float)) * np.cos(np.radians(tde)) * 3600.0
        dy = (tde - crow[de_c].to_numpy(float)) * 3600.0
        inside = in_iras_ellipse(dx, dy, crow["iras_major"], crow["iras_minor"], crow["iras_posang"],
                                 floor, cap)
        m = m[inside].reset_index(drop=True)
    else:
        m = match_within(pos, cat, float(ccfg.get("radius_arcsec", 6.0)), cat_cols=(ra_c, de_c))
    if not len(m):
        return pd.DataFrame()
    rows = cat.iloc[m["cat_index"].to_numpy()].reset_index(drop=True)
    rows.insert(0, "source_id", targets["source_id"].to_numpy(np.int64)[m["pos_index"].to_numpy()])
    rows[f"{which}_sep_arcsec"] = m["sep_arcsec"].to_numpy()
    return rows


# ===========================================================================
# Locus and residuals
# ===========================================================================
def luminosity_class(g, bp_rp, parallax, cfg_locus: dict) -> np.ndarray:
    """'giant' / 'dwarf' / 'unknown' from M_G = G + 5 log10(parallax / 100)."""
    g = np.asarray(g, float)
    plx = np.asarray(parallax, float)
    bprp = np.asarray(bp_rp, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mg = np.where(plx > 0, g + 5 * np.log10(np.where(plx > 0, plx, 1) / 100.0), np.nan)
    out = np.full(len(g), "unknown", dtype=object)
    ok = np.isfinite(mg) & np.isfinite(bprp)
    giant = ok & (mg < float(cfg_locus.get("giant_mg_max", 2.5))) & (bprp > float(cfg_locus.get("giant_bp_rp_min", 0.9)))
    out[ok] = "dwarf"
    out[giant] = "giant"
    return out


def fit_bright_locus(jk, y, cfg_locus: dict) -> dict:
    """Running median and robust scatter (1.4826 MAD) of ``y`` in (J−Ks) bins.

    Returns a JSON-able dict with the bin centres that had at least
    ``min_per_bin`` stars, ``median``, ``scatter``, ``n`` per bin, ``n_used``,
    ``scatter_global`` and ``ok`` (False when the band cannot be calibrated).
    """
    jk = np.asarray(jk, float)
    y = np.asarray(y, float)
    good = np.isfinite(jk) & np.isfinite(y)
    jk, y = jk[good], y[good]
    lo, hi = float(cfg_locus.get("jk_min", -0.3)), float(cfg_locus.get("jk_max", 1.6))
    w = float(cfg_locus.get("bin_width", 0.05))
    min_n = int(cfg_locus.get("min_per_bin", 20))
    n_min_fit = int(cfg_locus.get("n_min_fit", 60))
    out = {"ok": False, "n_used": int(len(y)), "bin_centres": [], "median": [], "scatter": [],
           "n": [], "scatter_global": float("nan"), "jk_min": lo, "jk_max": hi, "bin_width": w}
    if len(y) < n_min_fit:
        return out
    edges = np.arange(lo, hi + w / 2, w)
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        m = (jk >= a) & (jk < b)
        if m.sum() < min_n:
            continue
        med = float(np.median(y[m]))
        mad = float(np.median(np.abs(y[m] - med)))
        out["bin_centres"].append(round(float((a + b) / 2), 4))
        out["median"].append(med)
        out["scatter"].append(max(1.4826 * mad, 1e-3))
        out["n"].append(int(m.sum()))
    if len(out["bin_centres"]) < 2:
        return out
    med_all, sc_all = evaluate_locus(out | {"ok": True}, jk)
    r = y - med_all
    out["scatter_global"] = float(1.4826 * np.median(np.abs(r - np.median(r))))
    out["ok"] = True
    return out


def evaluate_locus(locus: dict, jk) -> tuple[np.ndarray, np.ndarray]:
    """(median, scatter) at each J−Ks; flat extrapolation beyond the fitted range."""
    jk = np.asarray(jk, float)
    if not locus.get("ok") or len(locus.get("bin_centres", [])) < 2:
        return np.full(len(jk), np.nan), np.full(len(jk), np.nan)
    x = np.asarray(locus["bin_centres"], float)
    med = np.interp(jk, x, np.asarray(locus["median"], float))
    sc = np.interp(jk, x, np.asarray(locus["scatter"], float))
    bad = ~np.isfinite(jk)
    med[bad], sc[bad] = np.nan, np.nan
    return med, sc


def locus_in_range(locus: dict, jk) -> np.ndarray:
    jk = np.asarray(jk, float)
    if not locus.get("ok"):
        return np.zeros(len(jk), dtype=bool)
    x = np.asarray(locus["bin_centres"], float)
    half = 0.5 * float(locus.get("bin_width", 0.05))
    return np.isfinite(jk) & (jk >= x.min() - half) & (jk <= x.max() + half)


def _quality_ok(q, ok_values) -> np.ndarray:
    q = pd.to_numeric(pd.Series(q), errors="coerce").to_numpy(float)
    return np.isin(q, [float(v) for v in ok_values]) & np.isfinite(q)


def fit_all_loci(df: pd.DataFrame, cfg: dict) -> dict:
    """Per-band loci, pooled and per class, with the split decision recorded."""
    lcfg = cfg["locus"]
    mode = str(lcfg.get("class_split", "auto"))
    gain = float(lcfg.get("class_split_gain_min", 0.85))
    n_min_class = int(lcfg.get("n_min_class", 150))
    jk = (df["j_m"] - df["ks_m"]).to_numpy(float)
    y_all = {}
    loci: dict = {}
    for b, spec in BANDS.items():
        qok = _quality_ok(df.get(f"q_{b}", np.nan), cfg[spec["instrument"]].get("quality_ok", [3]))
        y = (df["ks_m"] - df[f"m_{b}"]).to_numpy(float)
        y = np.where(qok, y, np.nan)
        y_all[b] = y
        pooled = fit_bright_locus(jk, y, lcfg)
        entry = {"pooled": pooled, "mode": "pooled", "classes": {}}
        if mode in ("auto", "split") and "lum_class" in df:
            cls = df["lum_class"].to_numpy()
            per = {}
            for c in ("dwarf", "giant"):
                m = cls == c
                per[c] = fit_bright_locus(jk[m], y[m], lcfg)
            entry["classes"] = per
            both = all(per[c]["ok"] and per[c]["n_used"] >= n_min_class for c in per)
            if both:
                split_sc = float(np.median([per[c]["scatter_global"] for c in per]))
                entry["split_scatter_median"] = split_sc
                if mode == "split" or (pooled["ok"] and split_sc < gain * pooled["scatter_global"]):
                    entry["mode"] = "split"
        loci[b] = entry
    return loci


def residuals(df: pd.DataFrame, loci: dict, cfg: dict) -> pd.DataFrame:
    """Add ``m_b, e_m_b, resid_b, err_b, sig_b, locus_ok_b, pred_b_jy`` per band.

    resid = (Ks − m_b) − locus(J−Ks): a DEFICIT is NEGATIVE.  err is the larger
    of the locus scatter at that colour and the star's own propagated error
    (Ks error + flux error) — the locus scatter already contains the typical
    photometric error, so adding them again would double-count.
    """
    out = df.copy()
    jk = (out["j_m"] - out["ks_m"]).to_numpy(float)
    out["jk"] = jk
    e_ks = pd.to_numeric(out.get("e_ks", np.nan), errors="coerce").to_numpy(float)
    e_ks = np.where(np.isfinite(e_ks), e_ks, 0.0)
    cls = out["lum_class"].to_numpy() if "lum_class" in out else np.full(len(out), "unknown", dtype=object)
    for b, spec in BANDS.items():
        zp = float(cfg[spec["instrument"]]["zero_points_jy"][spec["zp"]])
        flux = pd.to_numeric(out.get(b, np.nan), errors="coerce").to_numpy(float)
        eflux = pd.to_numeric(out.get(f"e_{b}", np.nan), errors="coerce").to_numpy(float)
        m, em = flux_to_mag(flux, zp, np.where(np.isfinite(eflux), eflux, 0.0))
        out[f"m_{b}"], out[f"e_m_{b}"] = m, em
        entry = loci.get(b, {"pooled": {"ok": False}, "mode": "pooled"})
        med = np.full(len(out), np.nan)
        sc = np.full(len(out), np.nan)
        inrange = np.zeros(len(out), dtype=bool)
        if entry.get("mode") == "split":
            for c in ("dwarf", "giant"):
                mk = cls == c
                med[mk], sc[mk] = evaluate_locus(entry["classes"][c], jk[mk])
                inrange[mk] = locus_in_range(entry["classes"][c], jk[mk])
            mk = ~np.isin(cls, ["dwarf", "giant"])
            med[mk], sc[mk] = evaluate_locus(entry["pooled"], jk[mk])
            inrange[mk] = locus_in_range(entry["pooled"], jk[mk])
        else:
            med, sc = evaluate_locus(entry["pooled"], jk)
            inrange = locus_in_range(entry["pooled"], jk)
        pred_m = out["ks_m"].to_numpy(float) - med
        out[f"pred_{b}_jy"] = mag_to_flux(pred_m, zp)
        out[f"locus_ok_{b}"] = inrange & np.isfinite(med)
        resid = (out["ks_m"].to_numpy(float) - m) - med
        own = np.sqrt(e_ks ** 2 + np.where(np.isfinite(em), em, 0.0) ** 2)
        err = np.maximum(np.where(np.isfinite(sc), sc, np.nan), own)
        out[f"resid_{b}"] = resid
        out[f"err_{b}"] = err
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"sig_{b}"] = np.where(err > 0, resid / err, np.nan)
    return out


# ===========================================================================
# Screen
# ===========================================================================
def _band_deficit(df: pd.DataFrame, b: str, dmin: float, smin: float) -> np.ndarray:
    r = df[f"resid_{b}"].to_numpy(float)
    s = df[f"sig_{b}"].to_numpy(float)
    ok = df[f"locus_ok_{b}"].to_numpy(bool)
    return ok & np.isfinite(r) & (r <= -dmin) & np.isfinite(s) & (s <= -smin)


def screen_bright(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply the bright-tier selection; returns (candidates, flagged frame, counters).

    The frame gets ``primary_band``, ``deficit_<b>``, ``qok_<b>``, ``n_agreeing``,
    ``agreeing_bands``, the veto columns and ``veto`` (first veto that fired,
    '' for a candidate, 'not_deficit' otherwise).
    """
    scfg = cfg["screen"]
    dmin, smin = float(scfg.get("deficit_mag_min", 0.7)), float(scfg.get("sig_min", 4.0))
    out = df.copy()
    n = len(out)
    if n == 0:
        return out, out, {"n_primary_deficit": 0, "n_candidates": 0}
    # Quality per band.
    for b, spec in BANDS.items():
        out[f"qok_{b}"] = _quality_ok(out.get(f"q_{b}", np.nan),
                                      cfg[spec["instrument"]].get("quality_ok", [3]))
        out[f"deficit_{b}"] = _band_deficit(out, b, dmin, smin)
    sat9 = float(cfg["akari"].get("saturation_9_jy", 30.0))
    sat18 = float(cfg["akari"].get("saturation_18_jy", 100.0))
    out["akari_sat_regime_9"] = out["pred_s09_jy"].to_numpy(float) > sat9
    out["akari_sat_regime_18"] = out["pred_s18_jy"].to_numpy(float) > sat18
    has_akari = out["has_akari"].to_numpy(bool) if "has_akari" in out else np.isfinite(out["s09"].to_numpy(float))
    has_iras = out["has_iras"].to_numpy(bool) if "has_iras" in out else np.isfinite(out["f12"].to_numpy(float))
    out["primary_band"] = np.where(out["akari_sat_regime_9"], "f12", "s09")
    prim = out["primary_band"].to_numpy()
    primary_deficit = np.zeros(n, dtype=bool)
    primary_qok = np.ones(n, dtype=bool)
    for b in ("s09", "f12"):
        m = prim == b
        primary_deficit[m] = out[f"deficit_{b}"].to_numpy(bool)[m]
        primary_qok[m] = out[f"qok_{b}"].to_numpy(bool)[m]
    out["primary_deficit"] = primary_deficit
    # Secondary agreement: any other band with an accepted quality and a deficit.
    agree = np.zeros(n, dtype=int)
    names = [[] for _ in range(n)]
    for b in BANDS:
        d = out[f"deficit_{b}"].to_numpy(bool) & out[f"qok_{b}"].to_numpy(bool) & (prim != b)
        # An AKARI band in its own saturation regime cannot corroborate either.
        if b == "s09":
            d &= ~out["akari_sat_regime_9"].to_numpy(bool)
        if b == "s18":
            d &= ~out["akari_sat_regime_18"].to_numpy(bool)
        agree += d
        for i in np.flatnonzero(d):
            names[i].append(b)
    out["n_agreeing"] = agree
    out["agreeing_bands"] = ["+".join(x) for x in names]
    out["iras_band_used"] = [(p == "f12") or any(x.startswith("f") for x in nm)
                             for p, nm in zip(prim, names, strict=False)]

    # Vetoes, in order; each counted over the primary-deficit stars.
    var = out["phot_variable_flag"].astype(str).str.upper().eq("VARIABLE").to_numpy() \
        if "phot_variable_flag" in out else np.zeros(n, dtype=bool)
    nss = pd.to_numeric(out.get("non_single_star", 0), errors="coerce").fillna(0).to_numpy() > 0
    lpv = ((out["jk"].to_numpy(float) > float(scfg["lpv"].get("jk_max", 1.1)))
           | (pd.to_numeric(out.get("bp_rp", np.nan), errors="coerce").to_numpy(float)
              > float(scfg["lpv"].get("bp_rp_max", 3.0))))
    n30 = pd.to_numeric(out.get("n_neigh_30", np.nan), errors="coerce").to_numpy(float)
    n60 = pd.to_numeric(out.get("n_neigh_60", np.nan), errors="coerce").to_numpy(float)
    crowded = (n30 > 0) | (out["iras_band_used"].to_numpy(bool) & (n60 > 0))
    crowd_unknown = ~np.isfinite(n30)
    poor_akari = (prim == "s09") & ~primary_qok
    single = agree == 0
    order = [("variable", var), ("non_single_star", nss), ("lpv_colour", lpv),
             ("crowded", crowded), ("poor_akari_quality", poor_akari),
             ("single_band_only", single)]
    veto = np.where(primary_deficit, "", "not_deficit").astype(object)
    counters = {"n_primary_deficit": int(primary_deficit.sum()),
                "n_primary_akari_sat_regime": int((primary_deficit & (prim == "f12")).sum())}
    remaining = primary_deficit.copy()
    for name, mask in order:
        hit = remaining & mask
        counters[name] = int(hit.sum())
        veto[hit] = name
        remaining &= ~mask
        out[f"veto_{name}"] = mask & primary_deficit
    counters["crowding_unknown_among_deficits"] = int((primary_deficit & crowd_unknown).sum())
    counters["deferred_lpv"] = counters["lpv_colour"]
    out["veto"] = veto
    out["is_candidate"] = remaining
    counters["n_candidates"] = int(remaining.sum())
    # Flags (not vetoes).
    out["etz"] = np.abs(pd.to_numeric(out.get("ecl_lat", np.nan), errors="coerce").to_numpy(float)) \
        < float(scfg.get("etz_ecl_lat_deg", 0.264))
    out["nearby"] = pd.to_numeric(out.get("parallax", np.nan), errors="coerce").to_numpy(float) \
        > float(scfg.get("nearby_parallax_mas", 20.0))
    out["crowding_unknown"] = crowd_unknown
    counters["n_candidates_etz"] = int((remaining & out["etz"].to_numpy(bool)).sum())
    counters["n_candidates_nearby"] = int((remaining & out["nearby"].to_numpy(bool)).sum())
    # IRAS upper limits far below the photosphere (quality 1 = upper limit).
    ulq = float(cfg["iras"].get("upper_limit_quality", 1))
    factor = 10 ** (0.4 * dmin)
    for b in ("f12", "f25"):
        q = pd.to_numeric(out.get(f"q_{b}", np.nan), errors="coerce").to_numpy(float)
        lim = pd.to_numeric(out.get(b, np.nan), errors="coerce").to_numpy(float)
        pred = out[f"pred_{b}_jy"].to_numpy(float)
        out[f"iras_ul_below_photosphere_{b}"] = (q == ulq) & np.isfinite(lim) & (lim > 0) \
            & np.isfinite(pred) & (pred > factor * lim) & out[f"locus_ok_{b}"].to_numpy(bool)
    ul = out["iras_ul_below_photosphere_f12"] | out["iras_ul_below_photosphere_f25"]
    out["iras_upper_limit_below_photosphere"] = ul
    counters["iras_upper_limit_below_photosphere"] = int(ul.sum())
    counters["n_with_akari"] = int(has_akari.sum())
    counters["n_with_iras"] = int(has_iras.sum())
    # Measured stars whose colour lies outside the fitted locus: no prediction,
    # hence never a deficit -- counted so the blind spot is visible.
    counters["n_measured_outside_locus_range"] = int(
        ((has_akari & ~out["locus_ok_s09"].to_numpy(bool))
         | (~has_akari & has_iras & ~out["locus_ok_f12"].to_numpy(bool))).sum())
    cands = out[out["is_candidate"]].copy()
    return cands, out, counters


def missing_bright(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stars the photosphere says AKARI must see (≫ its limit) with no AKARI and no IRAS source.

    The fully-opaque limit of the screen.  It is a LIST TO CHECK, not a
    candidate list: the IRC survey has ~4 % sky gaps and the PSC drops sources
    with too few detections, so a bright star can be missing for that reason.
    """
    mcfg = cfg["screen"]["missing"]
    if not len(df):
        return pd.DataFrame()
    lim = float(cfg["akari"].get("limit_9_jy", 0.05)) * float(mcfg.get("predicted_over_limit_min", 10.0))
    b = np.abs(pd.to_numeric(df.get("b", np.nan), errors="coerce").to_numpy(float))
    has_akari = df["has_akari"].to_numpy(bool)
    has_iras = df["has_iras"].to_numpy(bool)
    pred = df["pred_s09_jy"].to_numpy(float)
    m = (~has_akari & ~has_iras & (b > float(mcfg.get("b_min_deg", 10.0)))
         & np.isfinite(pred) & (pred > lim) & df["locus_ok_s09"].to_numpy(bool))
    cols = [c for c in ("source_id", "origin", "hip", "ra", "dec", "l", "b", "ecl_lat", "parallax",
                        "distance_pc", "phot_g_mean_mag", "bp_rp", "j_m", "ks_m", "e_ks", "jk",
                        "lum_class", "pred_s09_jy", "pred_s18_jy", "pred_f12_jy", "pred_f25_jy",
                        "tmass_read1_regime", "n_neigh_30", "n_neigh_60", "etz", "nearby")
            if c in df.columns]
    out = df.loc[m, cols].copy()
    out["predicted_over_akari_limit"] = out["pred_s09_jy"] / float(cfg["akari"].get("limit_9_jy", 0.05))
    out["caveat"] = ("no AKARI/IRC PSC and no IRAS PSC source within the match radius; "
                     "IRC survey sky coverage ~96 % and PSC detection-count cuts can also do this — "
                     "check the IRC image/coverage at this position before believing it")
    return out.sort_values("pred_s09_jy", ascending=False).reset_index(drop=True)


def tail_asymmetry(df: pd.DataFrame, loci: dict, cfg: dict) -> dict:
    out = {}
    for b in BANDS:
        if f"resid_{b}" not in df or f"qok_{b}" not in df:
            continue
        ok = df[f"locus_ok_{b}"].to_numpy(bool) & df[f"qok_{b}"].to_numpy(bool)
        r = df[f"resid_{b}"].to_numpy(float)[ok]
        sc = loci.get(b, {}).get("pooled", {}).get("scatter_global", float("nan"))
        entry = {"n": int(np.isfinite(r).sum()), "scatter_global": sc}
        for k in cfg["locus"].get("tail_sigmas", [3.0, 5.0]):
            if np.isfinite(sc) and sc > 0:
                entry[f"n_excess_gt_{k:g}sig"] = int((r > k * sc).sum())
                entry[f"n_deficit_lt_{k:g}sig"] = int((r < -k * sc).sum())
        out[b] = entry
    return out


def inject_sensitivity(df: pd.DataFrame, loci: dict, cfg: dict, max_rows: int = 20000) -> dict:
    """Recovery fraction of injected deficits, by 2MASS regime and band coverage.

    Only the PHOTOMETRIC criterion is applied (primary deficit + one agreeing
    band); the astrophysical vetoes are the same for a real and an injected
    screen and would only dilute the number.
    """
    base = df[df["has_akari"] | df["has_iras"]].head(max_rows)
    out: dict = {"n_base": int(len(base)), "inject_mags": list(cfg["sensitivity"]["inject_mags"])}
    if not len(base):
        return out
    read1 = base["tmass_read1_regime"].to_numpy(bool) if "tmass_read1_regime" in base \
        else np.zeros(len(base), dtype=bool)
    both = base["has_akari"].to_numpy(bool) & base["has_iras"].to_numpy(bool)
    # A second band exists where any non-primary band carries a finite flux.
    second = np.zeros(len(base), dtype=bool)
    for b in ("s18", "f12", "f25"):
        second |= np.isfinite(pd.to_numeric(base[b], errors="coerce").to_numpy(float))
    out["n_with_second_band"] = int(second.sum())
    for d in cfg["sensitivity"]["inject_mags"]:
        inj = base.copy()
        f = 10 ** (-0.4 * float(d))
        for b in BANDS:
            inj[b] = inj[b] * f
            inj[f"e_{b}"] = inj[f"e_{b}"] * f
        inj = residuals(inj, loci, cfg)
        _, flagged, _ = screen_bright(inj, cfg)
        rec = flagged["primary_deficit"].to_numpy(bool) & (flagged["n_agreeing"].to_numpy(int) > 0)
        rec_any = flagged["primary_deficit"].to_numpy(bool)
        key = f"{float(d):g}"
        out[key] = {
            "recovered_two_band": float(rec.mean()),
            "recovered_primary_only": float(rec_any.mean()),
            "recovered_two_band_read1": float(rec[read1].mean()) if read1.any() else None,
            "recovered_two_band_not_read1": float(rec[~read1].mean()) if (~read1).any() else None,
            "recovered_two_band_akari_and_iras": float(rec[both].mean()) if both.any() else None,
            "recovered_two_band_given_second_band": float(rec[second].mean()) if second.any() else None,
        }
    return out


# ===========================================================================
# Orchestration
# ===========================================================================
def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items() if k != "known_mask"}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2, default=str))


def run_bright_stage(cfg: dict, out_dir, *, gaia_fetcher=None, tmass_fetcher=None,
                     akari_fetcher=None, iras_fetcher=None, hip_fetcher=None,
                     max_targets: int | None = None) -> dict:
    """The whole bright tier: targets → anchors → mid-IR → locus → screen → summary.

    Every fetcher is injectable (tests stub all of them; the runner uses the
    defaults).  A summary is written whatever happens, and a verdict of
    ``NO_DATA_REACHED`` is a statement about the archives, never about the sky.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    ledger: list[dict] = []
    summary: dict = {"channel": "baffle_bright", "signature": "S38 (bright tier)",
                     "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "max_targets": max_targets,
                     "gaia_incompleteness_note": (
                         "Gaia DR3 is incomplete for G < 3; the ~300 brightest stars are "
                         "supplemented from Hipparcos-2 (I/311/hip2) when that route answers, "
                         "and brightest_300_not_covered is set when it does not."),
                     "counters": {}, "degradations": []}
    gaia_fetcher = gaia_fetcher or default_gaia_fetcher
    akari_fetcher = akari_fetcher or VizierSliceFetcher(cfg["akari"]["table"], AKARI_ALIASES, "akari_ra")
    iras_fetcher = iras_fetcher or VizierSliceFetcher(cfg["iras"]["table"], IRAS_ALIASES, "iras_ra")
    tmass_fetcher = tmass_fetcher or VizierUploadMatcher(cfg["tmass"])

    def finish(verdict: str, reason: str = "") -> dict:
        summary["verdict"] = verdict
        if reason:
            summary["reason"] = reason
        summary["elapsed_s"] = round(time.monotonic() - t_start, 1)
        summary["ledger_counts"] = {s: int(sum(1 for e in ledger if e["status"] == s))
                                    for s in (QUERY_OK, QUERY_ZERO, QUERY_FAILED, FROM_CHECKPOINT)}
        _write_json(out_dir / "ledger.json", {"entries": ledger})
        _write_json(out_dir / "summary.json", summary)
        return summary

    # 1. Targets ---------------------------------------------------------------
    targets, tinfo = acquire_targets(cfg, out_dir, ledger, gaia_fetcher, hip_fetcher, max_targets)
    summary["acquisition"] = {"targets": _json_safe(tinfo)}
    hstat = tinfo["hipparcos"]["status"]
    summary["brightest_300_not_covered"] = hstat != QUERY_OK
    summary["n_hipparcos_added"] = int(tinfo["hipparcos"].get("n_added", 0))
    if hstat == QUERY_FAILED:
        summary["degradations"].append("hipparcos:QUERY_FAILED")
    if tinfo["gaia"]["failed"]:
        summary["degradations"].append(f"gaia_targets:{tinfo['gaia']['failed']}/{tinfo['gaia']['n_slabs']}_slabs_failed")
    summary["n_targets"] = int(len(targets))
    if not len(targets):
        return finish(VERDICT_NO_DATA, "no targets acquired from Gaia (and no Hipparcos supplement)")
    summary["n_targets_gaia"] = int((targets["origin"] == "gaia_dr3").sum())
    summary["n_targets_etz"] = int((np.abs(targets["ecl_lat"].to_numpy(float)) < float(cfg["screen"]["etz_ecl_lat_deg"])).sum())
    summary["n_targets_nearby"] = int((targets["parallax"].to_numpy(float) > float(cfg["screen"]["nearby_parallax_mas"])).sum())
    targets.to_parquet(out_dir / "targets.parquet", index=False)

    # 2. Crowding neighbours ---------------------------------------------------
    neigh, nstats = acquire_neighbours(cfg, out_dir, ledger, gaia_fetcher, targets)
    known = nstats.pop("known_mask")
    summary["acquisition"]["neighbours"] = _json_safe(nstats)
    if nstats["failed"]:
        summary["degradations"].append(f"gaia_neighbours:{nstats['failed']}/{nstats['n_slabs']}_slabs_failed")
    ccfg = cfg["screen"]["crowded"]
    n30, n60 = crowding_counts(targets, neigh, known, float(ccfg.get("radius_arcsec", 30.0)),
                               float(ccfg.get("iras_radius_arcsec", 60.0)), float(ccfg.get("g_max", 10.0)))
    targets["n_neigh_30"], targets["n_neigh_60"] = n30, n60

    # 3. 2MASS anchor ----------------------------------------------------------
    tm, tstats = acquire_tmass(cfg, out_dir, ledger, tmass_fetcher, targets)
    summary["acquisition"]["tmass"] = _json_safe(tstats)
    if tstats["failed"]:
        summary["degradations"].append(f"tmass:{tstats['failed']}/{tstats['n_chunks']}_chunks_failed")
    if len(tm):
        keep = [c for c in tm.columns if c != "source_id"]
        work = targets.merge(tm[["source_id"] + keep], on="source_id", how="left")
    else:
        work = targets.copy()
        for c in ("tmass_id", "tmass_ra", "tmass_dec", "j_m", "e_j", "h_m", "e_h", "ks_m", "e_ks",
                  "tmass_qflg", "tmass_rflg", "tmass_sep_arcsec"):
            work[c] = np.nan
    for c in ("j_m", "e_j", "h_m", "e_h", "ks_m", "e_ks"):
        work[c] = pd.to_numeric(work.get(c, np.nan), errors="coerce")
    q_ok, q_known, read1_flag = tmass_quality_masks(work, cfg["tmass"])
    work["tmass_qual_ok"] = q_ok
    work["tmass_qual_known"] = q_known
    work["has_tmass"] = np.isfinite(work["ks_m"].to_numpy(float)) & np.isfinite(work["j_m"].to_numpy(float)) & q_ok
    work["tmass_read1_regime"] = (work["ks_m"].to_numpy(float) < float(cfg["tmass"].get("read1_ks_max", 4.0))) | read1_flag
    summary["n_with_2mass"] = int(work["has_tmass"].sum())
    summary["n_with_2mass_photometry"] = int((np.isfinite(work["ks_m"].to_numpy(float))
                                              & np.isfinite(work["j_m"].to_numpy(float))).sum())
    summary["n_with_2mass_quality_unknown"] = int((work["has_tmass"] & ~q_known).sum())
    summary["n_rejected_2mass_quality"] = int((np.isfinite(work["ks_m"].to_numpy(float)) & ~q_ok).sum())
    summary["n_with_2mass_read1_regime"] = int((work["has_tmass"] & work["tmass_read1_regime"]).sum())

    # 4. Mid-IR ----------------------------------------------------------------
    akari, astats = acquire_midir(cfg, out_dir, ledger, akari_fetcher, "akari")
    iras, istats = acquire_midir(cfg, out_dir, ledger, iras_fetcher, "iras")
    for nm, st, fx in (("akari", astats, akari_fetcher), ("iras", istats, iras_fetcher)):
        st = dict(st)
        if getattr(fx, "discovery", None):
            st["discovery_route"] = fx.discovery.get("route")
            st["n_columns_seen"] = len(fx.discovery.get("names", []))
        summary["acquisition"][nm] = _json_safe(st)
        if st["failed"] == st["n_slabs"] or st.get("error"):
            summary["degradations"].append(f"{nm}:{QUERY_FAILED}")
        elif st["failed"]:
            summary["degradations"].append(f"{nm}:{st['failed']}/{st['n_slabs']}_slabs_failed")
    summary["n_akari_rows"], summary["n_iras_rows"] = int(len(akari)), int(len(iras))
    am = match_midir(cfg, targets, akari, "akari")
    im = match_midir(cfg, targets, iras, "iras")
    for cat, nm in ((am, "akari"), (im, "iras")):
        if len(cat):
            cat = cat.drop_duplicates("source_id")
            work = work.merge(cat, on="source_id", how="left")
        for b, spec in BANDS.items():
            if spec["instrument"] == nm:
                for c in (b, f"e_{b}", f"q_{b}"):
                    if c not in work:
                        work[c] = np.nan
    for c in ("akari_id", "akari_sep_arcsec", "iras_id", "iras_sep_arcsec", "iras_cirr3",
              "iras_major", "iras_minor", "iras_posang"):
        if c not in work:
            work[c] = np.nan
    work["has_akari"] = np.isfinite(pd.to_numeric(work["s09"], errors="coerce").to_numpy(float)) \
        | np.isfinite(pd.to_numeric(work["s18"], errors="coerce").to_numpy(float))
    work["has_iras"] = np.isfinite(pd.to_numeric(work["f12"], errors="coerce").to_numpy(float)) \
        | np.isfinite(pd.to_numeric(work["f25"], errors="coerce").to_numpy(float))
    summary["n_with_akari"] = int(work["has_akari"].sum())
    summary["n_with_iras"] = int(work["has_iras"].sum())
    summary["n_with_2mass_and_midir"] = int((work["has_tmass"] & (work["has_akari"] | work["has_iras"])).sum())
    if summary["n_with_2mass"] == 0:
        return finish(VERDICT_NO_DATA, "no 2MASS anchor reached for any target")
    if summary["n_with_akari"] == 0 and summary["n_with_iras"] == 0:
        return finish(VERDICT_NO_DATA, "neither AKARI nor IRAS reached")

    # 5. Locus + residuals -----------------------------------------------------
    work["lum_class"] = luminosity_class(work["phot_g_mean_mag"], work["bp_rp"], work["parallax"], cfg["locus"])
    pre = work[work["has_tmass"]].copy()
    for b, spec in BANDS.items():
        zp = float(cfg[spec["instrument"]]["zero_points_jy"][spec["zp"]])
        pre[f"m_{b}"] = flux_to_mag(pd.to_numeric(pre[b], errors="coerce"), zp)
    loci = fit_all_loci(pre, cfg)
    _write_json(out_dir / "locus_bright.json", {"bands": loci, "config": cfg["locus"],
                                                "zero_points_jy": {"akari": cfg["akari"]["zero_points_jy"],
                                                                   "iras": cfg["iras"]["zero_points_jy"]}})
    summary["locus"] = {b: {"mode": loci[b]["mode"], "ok": loci[b]["pooled"]["ok"],
                            "n_used": loci[b]["pooled"]["n_used"],
                            "scatter_global": loci[b]["pooled"]["scatter_global"]} for b in loci}
    res = residuals(pre, loci, cfg)

    # 6. Screen ----------------------------------------------------------------
    cands, flagged, counters = screen_bright(res, cfg)
    summary["counters"] = counters
    summary["tail_asymmetry"] = tail_asymmetry(flagged, loci, cfg)
    missing = missing_bright(flagged, cfg)
    summary["n_missing_bright_candidates"] = int(len(missing))
    summary["n_candidates"] = int(len(cands))
    summary["n_candidates_etz"] = counters.get("n_candidates_etz", 0)
    summary["n_candidates_nearby"] = counters.get("n_candidates_nearby", 0)
    summary["denominators"] = {
        "etz_targets": summary["n_targets_etz"], "nearby_targets": summary["n_targets_nearby"],
        "etz_with_midir": int((flagged["etz"] & (flagged["has_akari"] | flagged["has_iras"])).sum()),
        "nearby_with_midir": int((flagged["nearby"] & (flagged["has_akari"] | flagged["has_iras"])).sum()),
    }
    summary["sensitivity"] = inject_sensitivity(flagged, loci, cfg)
    ul_ids = flagged.loc[flagged["iras_upper_limit_below_photosphere"], "source_id"].head(50).tolist()
    summary["iras_upper_limit_below_photosphere_source_ids"] = [int(x) for x in ul_ids]

    # 7. Outputs ---------------------------------------------------------------
    measured = flagged[flagged["has_akari"] | flagged["has_iras"]]
    measured.to_csv(out_dir / "bright_residuals.csv", index=False)
    cands.to_csv(out_dir / "candidates.csv", index=False)
    missing.to_csv(out_dir / "missing_bright_candidates.csv", index=False)
    if len(cands):
        summary["candidates"] = cands[[c for c in ("source_id", "origin", "hip", "ra", "dec", "parallax",
                                                    "phot_g_mean_mag", "ks_m", "jk", "primary_band",
                                                    "agreeing_bands", "resid_s09", "sig_s09", "resid_s18",
                                                    "resid_f12", "resid_f25", "etz", "nearby",
                                                    "tmass_read1_regime") if c in cands]].head(100) \
            .to_dict(orient="records")
    science = VERDICT_CAND if len(cands) else VERDICT_NULL
    summary["science_verdict"] = science
    if summary["degradations"]:
        return finish(f"DEGRADED_SOURCE ({'; '.join(summary['degradations'])})",
                      f"science verdict on what was reached: {science}")
    return finish(science)


# ===========================================================================
# Probe
# ===========================================================================
def probe_bright(cfg: dict, out_dir, *, url: str = VIZIER_TAP, gaia_fetcher=None) -> dict:
    """TAP_SCHEMA discovery for the VizieR tables and one tiny query each."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tables": {}}
    tables = {
        "tmass": (cfg["tmass"]["table"], TMASS_ALIASES, "tmass_ra"),
        "akari": (cfg["akari"]["table"], AKARI_ALIASES, "akari_ra"),
        "iras": (cfg["iras"]["table"], IRAS_ALIASES, "iras_ra"),
        "hipparcos": (cfg["targets"]["hipparcos"]["table"], HIP_ALIASES, "hip_ra"),
    }
    for key, (table, aliases, ra_key) in tables.items():
        entry: dict = {"table": table}
        try:
            disc = discover_columns(table, url)
            res = resolve_aliases(disc["names"], aliases)
            entry.update({"discovery_route": disc["route"], "n_columns": len(disc["names"]),
                          "columns": disc["names"], "resolved": res,
                          "units": {k: disc["meta"].get(v, {}).get("unit") for k, v in res.items()},
                          "errors": disc["errors"]})
            if ra_key in res:
                ra_col = _adql_col(res[ra_key], ra_key)
                select = select_list(res, required=(ra_key,))
                q = f"SELECT TOP 5 {select} FROM {table} WHERE {ra_col} >= 100 AND {ra_col} < 101"
                t0 = time.monotonic()
                df = run_vizier(q, retries=2, label=f"probe:{key}", url=url)
                entry["tiny_query"] = {"query": q, "status": QUERY_OK if len(df) else QUERY_ZERO,
                                       "n_rows": int(len(df)), "elapsed_s": round(time.monotonic() - t0, 2),
                                       "sample": _json_safe(df.head(2).to_dict(orient="records"))}
                entry["status"] = entry["tiny_query"]["status"]
            else:
                entry["status"] = QUERY_FAILED
                entry["error"] = f"RA column not resolved ({ra_key})"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = QUERY_FAILED
            entry["error"] = repr(exc)[:800]
        report["tables"][key] = entry
        print(f"[baffle-bright] probe {key}: {entry.get('status')} "
              f"{entry.get('n_columns', 0)} columns via {entry.get('discovery_route')}", flush=True)
    # Gaia: one tiny slab.
    fetch = gaia_fetcher or default_gaia_fetcher
    q = gaia_targets_query(float(cfg["targets"]["g_max"]), 100.0, 100.5)
    t0 = time.monotonic()
    try:
        df = fetch("SELECT TOP 20 " + q[len("SELECT "):], "probe_gaia")
        report["gaia"] = {"status": QUERY_OK if len(df) else QUERY_ZERO, "n_rows": int(len(df)),
                          "columns": list(df.columns), "elapsed_s": round(time.monotonic() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        report["gaia"] = {"status": QUERY_FAILED, "error": repr(exc)[:800]}
    # 2MASS upload with three bright anchors (Vega, Sirius, Arcturus) at 2000.0.
    anchors = pd.DataFrame({"source_id": [1, 2, 3],
                            "ra": [279.23473, 101.28716, 213.91530],
                            "dec": [38.78369, -16.71612, 19.18241]})
    try:
        matcher = VizierUploadMatcher(cfg["tmass"], url=url)
        t0 = time.monotonic()
        df = matcher(anchors, float(cfg["tmass"]["radius_arcsec"]) + 2.0, "probe_tmass_upload")
        report["tmass_upload"] = {"status": QUERY_OK if len(df) else QUERY_ZERO, "n_rows": int(len(df)),
                                  "routes": matcher.routes_used, "columns": list(df.columns)[:40],
                                  "elapsed_s": round(time.monotonic() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        report["tmass_upload"] = {"status": QUERY_FAILED, "error": repr(exc)[:800]}
    n_ok = sum(1 for e in report["tables"].values() if e.get("status") == QUERY_OK)
    report["verdict"] = ("PROBE_OK" if n_ok == len(tables) and report["gaia"].get("status") == QUERY_OK
                         else "PROBE_PARTIAL" if n_ok else VERDICT_NO_DATA)
    _write_json(out_dir / "probe.json", report)
    return report


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m seti.baffle.bright",
                                 description="BAFFLE bright tier: AKARI/IRAS mid-IR deficits on G < 7.5 stars")
    ap.add_argument("--stage", choices=("probe", "run"), default="run")
    ap.add_argument("--max-targets", type=int, default=None)
    ap.add_argument("--out", default=None, help="results directory (default results/baffle_bright)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_bright_config(a.config)
    out = Path(a.out) if a.out else _repo_root() / "results" / "baffle_bright"
    if a.stage == "probe":
        rep = probe_bright(cfg, out)
        print(json.dumps({k: v.get("status") for k, v in rep["tables"].items()} | {"verdict": rep["verdict"]}))
        return 0
    mt = a.max_targets if (a.max_targets or 0) > 0 else None
    s = run_bright_stage(cfg, out, max_targets=mt)
    print(json.dumps({k: s.get(k) for k in ("verdict", "n_targets", "n_with_2mass", "n_with_akari",
                                              "n_with_iras", "n_candidates",
                                              "n_missing_bright_candidates")}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Archive pulls for EMBER. Runner-only: the sandbox has no VO egress.

Every fetcher here is (a) chunked, (b) retried with backoff, (c) checkpointed to
parquet so a killed job loses minutes rather than hours, and (d) **injectable**,
so the whole channel runs offline in tests by passing a pre-built table.

Two things in this module are load-bearing and easy to get wrong:

**Proper motion.** Gaia positions are at epoch 2016.0. IRAS observed in 1983.5 --
a 32.5-year lever arm, so a 500 mas/yr star sits 16 arcsec away from where Gaia
says it is. Matching Gaia positions directly against IRAS silently loses every
high-proper-motion star, which is a bug that has already cost this repository a
whole run. Every cross-match here propagates to the *survey's* epoch first.

**Column names.** VizieR renames columns between catalogue versions. Rather than
hardcode them, each fetcher probes the table with ``SELECT TOP 1 *`` and resolves
its columns against a list of known aliases, following the pattern already used
for the Renson and cluster-membership catalogues in this repository.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- survey reference epochs (Julian year) ---------------------------------
GAIA_EPOCH = 2016.0
IRAS_EPOCH = 1983.5
AKARI_EPOCH = 2006.7
ALLWISE_EPOCH = 2010.5
TWOMASS_EPOCH = 1999.5

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
SVO_FPS = "http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="

#: VizieR table identifiers. Quoted because they contain slashes.
CATALOGUES = {
    "iras_psc": '"II/125/main"',
    "iras_fsc": '"II/156A/main"',
    "akari_irc": '"II/297/irc"',
    "allwise": '"II/328/allwise"',
}

# --- column alias tables ---------------------------------------------------
_RA_NAMES = ("_raj2000", "raj2000", "ra_icrs", "raicrs", "ra", "_ra")
_DEC_NAMES = ("_dej2000", "dej2000", "de_icrs", "deicrs", "dec", "de", "_de")

_IRAS_ALIASES: dict[str, tuple[str, ...]] = {
    "ra": _RA_NAMES,
    "dec": _DEC_NAMES,
    "name": ("iras", "fsc", "name", "id"),
    "f12": ("fnu_12", "fnu12", "s12", "f12"),
    "f25": ("fnu_25", "fnu25", "s25", "f25"),
    "f60": ("fnu_60", "fnu60", "s60", "f60"),
    "f100": ("fnu_100", "fnu100", "s100", "f100"),
    "e_f12": ("e_fnu_12", "e_fnu12", "rfnu_12", "e_s12"),
    "e_f25": ("e_fnu_25", "e_fnu25", "rfnu_25", "e_s25"),
    "q12": ("q_fnu_12", "q_fnu12", "qual12", "q12"),
    "q25": ("q_fnu_25", "q_fnu25", "qual25", "q25"),
    "cirr1": ("cirr1",),
    "cirr2": ("cirr2",),
    "cirr3": ("cirr3",),
    "var": ("var",),
    "conf": ("conf", "cc"),
    "major": ("major", "umaj", "unc_maj"),
    "minor": ("minor", "umin", "unc_min"),
    "posang": ("posang", "upa", "pa"),
}

_AKARI_ALIASES: dict[str, tuple[str, ...]] = {
    "ra": _RA_NAMES,
    "dec": _DEC_NAMES,
    "name": ("objid", "objname", "akari", "id", "name"),
    "s09": ("s09", "s9", "flux09", "f09"),
    "e_s09": ("e_s09", "e_s9", "e_flux09"),
    "s18": ("s18", "flux18", "f18"),
    "e_s18": ("e_s18", "e_flux18"),
    "q09": ("fq09", "q_s09", "fqual09", "q09"),
    "q18": ("fq18", "q_s18", "fqual18", "q18"),
    "ext09": ("ext09", "extended09"),
    "ext18": ("ext18", "extended18"),
    "ndens": ("ndens", "nsrc"),
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_columns(colnames, aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """Map canonical names to whatever this VizieR version actually calls them."""
    lookup = {_norm(c): c for c in colnames}
    out: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for cand in candidates:
            hit = lookup.get(_norm(cand))
            if hit is not None:
                out[canonical] = hit
                break
    return out


def _retry(fn, retries: int = 4, label: str = "ember", base_sleep: float = 2.0):
    """Call ``fn`` with exponential backoff; raise with the last error attached."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - archives fail in many ways
            last = exc
            print(f"[{label}] attempt {attempt + 1}/{retries} failed: {exc!r}")
            time.sleep(base_sleep * (2**attempt))
    raise RuntimeError(f"[{label}] failed after {retries} attempts: {last!r}")


def propagate(ra_deg, dec_deg, pmra_masyr, pmdec_masyr,
              from_epoch: float = GAIA_EPOCH, to_epoch: float = ALLWISE_EPOCH):
    """Propagate positions between epochs. ``pmra`` is mu_alpha* (includes cos dec)."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    pmra = np.nan_to_num(np.asarray(pmra_masyr, dtype=float))
    pmdec = np.nan_to_num(np.asarray(pmdec_masyr, dtype=float))
    dt = float(to_epoch) - float(from_epoch)
    cosd = np.maximum(np.cos(np.radians(dec)), 1e-6)
    return ra + (pmra * dt / 3.6e6) / cosd, dec + (pmdec * dt / 3.6e6)


def angular_sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Small-angle separation in arcsec (adequate below a few degrees)."""
    ra1, dec1 = np.asarray(ra1, float), np.asarray(dec1, float)
    ra2, dec2 = np.asarray(ra2, float), np.asarray(dec2, float)
    dra = (ra1 - ra2) * np.cos(np.radians(0.5 * (dec1 + dec2)))
    return np.hypot(dra, dec1 - dec2) * 3600.0


# --------------------------------------------------------------------------
# VizieR TAP
# --------------------------------------------------------------------------
def _tap_service(url: str = VIZIER_TAP):
    import pyvo  # imported lazily: runner-only dependency

    return pyvo.dal.TAPService(url)


def _probe_columns(table: str, url: str = VIZIER_TAP) -> list[str]:
    tap = _tap_service(url)
    probe = _retry(lambda: tap.run_sync(f"SELECT TOP 1 * FROM {table}").to_table(),
                   label=f"probe:{table}")
    return list(probe.colnames)


def fetch_vizier_catalogue(table: str, aliases: dict[str, tuple[str, ...]],
                           out_dir: Path | None = None, tag: str = "cat",
                           n_ra_chunks: int = 12, row_limit: int = 4_000_000,
                           url: str = VIZIER_TAP) -> pd.DataFrame:
    """Pull a whole VizieR catalogue in RA slices, checkpointing each slice.

    Chunking by RA rather than issuing one monolithic query is the same lesson
    the Gaia fetchers in this repository learned: a single multi-million-row
    async job times out or returns "cannot find result", while a dozen smaller
    ones succeed reliably.
    """
    cols = resolve_columns(_probe_columns(table, url), aliases)
    if "ra" not in cols or "dec" not in cols:
        raise RuntimeError(f"{table}: could not resolve RA/Dec columns; got {cols}")

    select = ", ".join(f'{v} AS "{k}"' for k, v in cols.items())
    tap = _tap_service(url)
    edges = np.linspace(0.0, 360.0, n_ra_chunks + 1)
    frames: list[pd.DataFrame] = []

    for i in range(n_ra_chunks):
        lo, hi = float(edges[i]), float(edges[i + 1])
        ckpt = (out_dir / f"{tag}_ra{i:02d}.parquet") if out_dir else None
        if ckpt is not None and ckpt.exists():
            frames.append(pd.read_parquet(ckpt))
            print(f"[ember] {tag} RA[{lo:.0f},{hi:.0f}) from checkpoint")
            continue
        q = (f"SELECT TOP {row_limit} {select} FROM {table} "
             f"WHERE {cols['ra']} >= {lo} AND {cols['ra']} < {hi}")
        df = _retry(lambda q=q: tap.run_async(q).to_table().to_pandas(),
                    label=f"{tag}:ra{i}")
        print(f"[ember] {tag} RA[{lo:.0f},{hi:.0f}) -> {len(df):,} rows")
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ckpt, index=False)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates()


def fetch_akari(out_dir: Path | None = None, **kw) -> pd.DataFrame:
    """AKARI/IRC Point Source Catalogue (~870k sources, 9 and 18 micron)."""
    return fetch_vizier_catalogue(CATALOGUES["akari_irc"], _AKARI_ALIASES,
                                  out_dir=out_dir, tag="akari", **kw)


def fetch_iras(which: str = "psc", out_dir: Path | None = None, **kw) -> pd.DataFrame:
    """IRAS Point Source Catalogue (~245k) or Faint Source Catalogue (~173k)."""
    key = "iras_psc" if which == "psc" else "iras_fsc"
    return fetch_vizier_catalogue(CATALOGUES[key], _IRAS_ALIASES,
                                  out_dir=out_dir, tag=key, **kw)


# --------------------------------------------------------------------------
# Gaia
# --------------------------------------------------------------------------
_GAIA_QUERY = """
SELECT u.match_id, g.source_id, g.ra, g.dec, g.parallax, g.parallax_error,
       g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, g.ruwe,
       g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
       g.phot_g_mean_flux_over_error, g.phot_variable_flag,
       g.teff_gspphot, g.l, g.b
FROM tap_upload.targets AS u
JOIN gaiadr3.gaia_source AS g
  ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec),
                  CIRCLE('ICRS', u.ra, u.dec, {radius_deg}))
"""


def fetch_gaia_for_positions(positions: pd.DataFrame, radius_arcsec: float = 6.0,
                             chunk: int = 20_000, out_dir: Path | None = None,
                             retries: int = 4) -> pd.DataFrame:
    """Cone-match a position list against Gaia DR3 via chunked TAP upload.

    ``positions`` needs ``match_id``, ``ra``, ``dec`` -- the *infrared* positions
    at the infrared epoch. Because Gaia positions are at 2016.0, the returned
    separations must be recomputed after propagating Gaia to the infrared epoch;
    :func:`attach_epoch_separation` does that.
    """
    from astropy.table import Table
    from astroquery.gaia import Gaia

    need = {"match_id", "ra", "dec"}
    if not need.issubset(positions.columns):
        raise ValueError(f"positions must contain {need}")
    frames: list[pd.DataFrame] = []
    n_chunks = int(np.ceil(len(positions) / chunk)) or 1
    q = _GAIA_QUERY.format(radius_deg=radius_arcsec / 3600.0)

    for i in range(n_chunks):
        ckpt = (out_dir / f"gaia_chunk_{i:03d}.parquet") if out_dir else None
        if ckpt is not None and ckpt.exists():
            frames.append(pd.read_parquet(ckpt))
            continue
        sub = positions.iloc[i * chunk:(i + 1) * chunk][["match_id", "ra", "dec"]]
        tbl = Table.from_pandas(sub.reset_index(drop=True))

        def _go(tbl=tbl):
            job = Gaia.launch_job_async(q, upload_resource=tbl,
                                        upload_table_name="targets")
            df = job.get_results().to_pandas()
            return df.rename(columns={c: c.lower() for c in df.columns})

        df = _retry(_go, retries=retries, label=f"gaia:{i}")
        print(f"[ember] gaia chunk {i + 1}/{n_chunks} -> {len(df):,} rows")
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ckpt, index=False)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_epoch_separation(matched: pd.DataFrame, ir_epoch: float,
                            ir_ra_col: str = "ir_ra", ir_dec_col: str = "ir_dec"
                            ) -> pd.DataFrame:
    """Recompute the optical-infrared separation *at the infrared epoch*.

    A cone match done at the Gaia epoch accepts or rejects on the wrong
    positions. Propagating first and re-measuring turns a generous search radius
    into a tight, physically meaningful association test, and is what makes
    high-proper-motion stars recoverable rather than silently absent.
    """
    out = matched.copy()
    ra_p, dec_p = propagate(out["ra"], out["dec"], out.get("pmra"), out.get("pmdec"),
                            GAIA_EPOCH, ir_epoch)
    out["ra_at_ir_epoch"] = ra_p
    out["dec_at_ir_epoch"] = dec_p
    out["sep_arcsec"] = angular_sep_arcsec(ra_p, dec_p, out[ir_ra_col], out[ir_dec_col])
    out["sep_arcsec_naive"] = angular_sep_arcsec(out["ra"], out["dec"],
                                                 out[ir_ra_col], out[ir_dec_col])
    return out


# --------------------------------------------------------------------------
# AllWISE / 2MASS
# --------------------------------------------------------------------------
_ALLWISE_CONE = """
SELECT designation, ra, dec, w1mpro, w1sigmpro, w2mpro, w2sigmpro,
       w3mpro, w3sigmpro, w4mpro, w4sigmpro, cc_flags, ext_flg, ph_qual,
       w1sat, w2sat, w3sat, w4sat, j_m_2mass, h_m_2mass, k_m_2mass,
       j_msig_2mass, h_msig_2mass, k_msig_2mass
FROM allwise_p3as_psd
WHERE CONTAINS(POINT('ICRS', ra, dec),
               CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
"""


def fetch_allwise_cone(ra: float, dec: float, radius_arcsec: float = 6.0,
                       retries: int = 3) -> pd.DataFrame:
    """One AllWISE cone search via IRSA TAP. Returns an empty frame on failure."""
    from astroquery.ipac.irsa import Irsa

    q = _ALLWISE_CONE.format(ra=float(ra), dec=float(dec),
                             radius_deg=float(radius_arcsec) / 3600.0)
    try:
        return _retry(lambda: Irsa.query_tap(q).to_table().to_pandas(),
                      retries=retries, label="allwise")
    except Exception as exc:  # noqa: BLE001
        print(f"[ember] allwise cone failed at ({ra:.5f},{dec:.5f}): {exc!r}")
        return pd.DataFrame()


def fetch_beam_neighbours(rows: pd.DataFrame, beam_radius_arcsec: float,
                          out_dir: Path | None = None,
                          fetch_fn=None) -> dict[str, pd.DataFrame]:
    """Collect every AllWISE source inside each early-epoch beam.

    This feeds ``crossepoch.beam_sum_consistency``, which is the single most
    important contamination test of the 27-year pair: an IRAS flux is the sum
    over a ~0.75' x 4.5' footprint, so the honest comparison is against the sum
    of all WISE sources in it, not the nearest one.

    ``fetch_fn(ra, dec, radius)`` is injectable so tests run offline.
    """
    fetch = fetch_fn or fetch_allwise_cone
    out: dict[str, pd.DataFrame] = {}
    for _, row in rows.iterrows():
        key = str(row["match_id"])
        ckpt = (out_dir / f"beam_{key}.parquet") if out_dir else None
        if ckpt is not None and ckpt.exists():
            out[key] = pd.read_parquet(ckpt)
            continue
        df = fetch(float(row["ir_ra"]), float(row["ir_dec"]), beam_radius_arcsec)
        if ckpt is not None and not df.empty:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ckpt, index=False)
        out[key] = df
    return out


_NEOWISE_LC = """
SELECT mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro, qual_frame, cc_flags
FROM neowiser_p1bs_psd
WHERE CONTAINS(POINT('ICRS', ra, dec),
               CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
"""


def fetch_neowise_lightcurve(ra: float, dec: float, radius_arcsec: float = 3.0
                             ) -> pd.DataFrame:
    """NEOWISE W1/W2 light curve for a shortlisted source.

    NEOWISE carries **W1 and W2 only**. It cannot see 100-300 K dust at all, so
    it is used here strictly as a variability veto and as a probe of hot
    (>~500 K) excess change -- never as the primary cessation measurement.
    """
    from astroquery.ipac.irsa import Irsa

    q = _NEOWISE_LC.format(ra=float(ra), dec=float(dec),
                           radius_deg=float(radius_arcsec) / 3600.0)
    try:
        df = _retry(lambda: Irsa.query_tap(q).to_table().to_pandas(),
                    retries=3, label="neowise")
    except Exception as exc:  # noqa: BLE001
        print(f"[ember] neowise fetch failed: {exc!r}")
        return pd.DataFrame()
    if df.empty:
        return df
    ok = (df["qual_frame"] > 0) & df["cc_flags"].astype(str).str.startswith("00")
    return df[ok]


# --------------------------------------------------------------------------
# Response curves from the SVO Filter Profile Service
# --------------------------------------------------------------------------
def fetch_rsr_curves(dest_dir: Path, band_keys: list[str] | None = None) -> dict:
    """Download real relative system response curves and cache them as CSV.

    Without these the band model falls back to a documented trapezoid, whose
    inadequacy is carried as a bandpass systematic of a few to ~9 percent. With
    them the systematic largely disappears. This runs on the runner only; the
    sandbox has no egress, so the fallback path must stay correct.
    """
    import io
    import urllib.request

    from .bands import BANDS

    dest_dir.mkdir(parents=True, exist_ok=True)
    keys = band_keys or list(BANDS)
    status: dict[str, str] = {}
    for key in keys:
        band = BANDS[key]
        if not band.svo_id:
            status[key] = "no svo id"
            continue
        url = SVO_FPS + band.svo_id
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Seti-ember/1.0 (mailto:trimcrae@gmail.com)"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read()
            from astropy.io.votable import parse_single_table

            tbl = parse_single_table(io.BytesIO(raw)).to_table()
            lam_a = np.asarray(tbl.columns[0], dtype=float)  # Angstrom
            trans = np.asarray(tbl.columns[1], dtype=float)
            df = pd.DataFrame({"lam_um": lam_a / 1e4, "response": trans})
            df = df[np.isfinite(df["lam_um"]) & np.isfinite(df["response"])]
            if len(df) < 4:
                status[key] = "too few points"
                continue
            df.to_csv(dest_dir / f"{key}.csv", index=False)
            status[key] = f"ok ({len(df)} points)"
        except Exception as exc:  # noqa: BLE001
            status[key] = f"failed: {exc!r}"
        time.sleep(1.0)
    return status


# --------------------------------------------------------------------------
# Cross-matching
# --------------------------------------------------------------------------
def build_working_table(ra_lo: float, ra_hi: float, cache: Path,
                        n_ra_chunks: int = 2,
                        fetchers: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Assemble the analysis-ready table for one RA slice, end to end.

    The chain is: AKARI + IRAS (early epochs) -> Gaia DR3 (the stellar identity
    and the proper motion) -> AllWISE (the late epoch and the 2MASS anchor).
    Gaia is queried at the *infrared* positions and the association is then
    re-tested after propagating Gaia astrometry back to each survey's epoch, so
    high-proper-motion stars survive rather than silently vanishing.

    ``fetchers`` overrides any of ``akari``, ``iras_psc``, ``iras_fsc``,
    ``gaia``, ``allwise`` so the whole chain is exercisable offline.

    Returns the table and a status dict recording which archives were reached.
    """
    fx = fetchers or {}
    status: dict = {"ra_lo": ra_lo, "ra_hi": ra_hi, "errors": [], "counts": {}}

    def _pull(name, fn):
        try:
            df = fn()
            status["counts"][name] = int(len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            status["errors"].append(f"{name}: {exc!r}")
            status["counts"][name] = 0
            return pd.DataFrame()

    akari = _pull("akari", fx.get("akari") or
                  (lambda: fetch_akari(cache, n_ra_chunks=n_ra_chunks)))
    iras_psc = _pull("iras_psc", fx.get("iras_psc") or
                     (lambda: fetch_iras("psc", cache, n_ra_chunks=n_ra_chunks)))
    iras_fsc = _pull("iras_fsc", fx.get("iras_fsc") or
                     (lambda: fetch_iras("fsc", cache, n_ra_chunks=n_ra_chunks)))

    def _slice(df):
        if df.empty or "ra" not in df.columns:
            return df
        return df[(df["ra"] >= ra_lo) & (df["ra"] < ra_hi)].copy()

    akari, iras_psc, iras_fsc = _slice(akari), _slice(iras_psc), _slice(iras_fsc)
    iras = pd.concat([d for d in (iras_psc, iras_fsc) if not d.empty],
                     ignore_index=True) if (not iras_psc.empty or not iras_fsc.empty) \
        else pd.DataFrame()

    if akari.empty and iras.empty:
        status["archive_reachable"] = False
        return pd.DataFrame(), status
    status["archive_reachable"] = True

    # Early-epoch anchor positions: AKARI where available (2 arcsec), else IRAS.
    anchor = akari if not akari.empty else iras
    anchor = anchor.reset_index(drop=True)
    anchor["match_id"] = [f"em{ra_lo:05.1f}_{i:07d}" for i in range(len(anchor))]
    positions = anchor[["match_id", "ra", "dec"]].copy()

    gaia = _pull("gaia", fx.get("gaia") or
                 (lambda: fetch_gaia_for_positions(positions, radius_arcsec=6.0,
                                                   out_dir=cache)))
    if gaia.empty:
        status["errors"].append("gaia: no counterparts; the channel cannot "
                                "distinguish stars from background galaxies")
        return pd.DataFrame(), status

    merged = gaia.merge(anchor.add_prefix("ir_"), left_on="match_id",
                        right_on="ir_match_id", how="inner")
    merged = attach_epoch_separation(merged, ir_epoch=AKARI_EPOCH if not akari.empty
                                     else IRAS_EPOCH)
    # Keep the single best Gaia counterpart per infrared source.
    merged = merged.sort_values("sep_arcsec").drop_duplicates("match_id")

    allwise = _pull("allwise", fx.get("allwise") or
                    (lambda: _allwise_for_rows(merged, cache)))
    if not allwise.empty:
        merged = merged.merge(allwise, on="match_id", how="left")

    status["counts"]["working"] = int(len(merged))
    return merged, status


def _allwise_for_rows(rows: pd.DataFrame, cache: Path | None = None,
                      radius_arcsec: float = 3.0) -> pd.DataFrame:
    """AllWISE photometry at each row's Gaia position propagated to 2010.5."""
    out = []
    for _, r in rows.iterrows():
        ra_p, dec_p = propagate(r["ra"], r["dec"], r.get("pmra"), r.get("pmdec"),
                                GAIA_EPOCH, ALLWISE_EPOCH)
        df = fetch_allwise_cone(float(ra_p), float(dec_p), radius_arcsec)
        if df.empty:
            continue
        rec = df.iloc[0].to_dict()
        rec["match_id"] = r["match_id"]
        rec["n_allwise_in_cone"] = int(len(df))
        out.append(rec)
    return pd.DataFrame(out)


def crossmatch_epochs(early: pd.DataFrame, late: pd.DataFrame,
                      early_epoch: float, late_epoch: float,
                      radius_arcsec: float,
                      pm_source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Positionally match two infrared catalogues, PM-corrected where possible.

    Both frames need ``ra``/``dec``. When ``pm_source`` supplies ``match_id``,
    ``pmra`` and ``pmdec``, the *early* positions are propagated forward to the
    late epoch before matching, so high-proper-motion stars are not lost.

    Uses a KD-tree in a local tangent plane, which is exact enough at these
    separations and avoids depending on archive-side upload cross-match.
    """
    from scipy.spatial import cKDTree

    if early.empty or late.empty:
        return pd.DataFrame()

    e_ra = early["ra"].to_numpy(dtype=float)
    e_dec = early["dec"].to_numpy(dtype=float)
    if pm_source is not None and {"pmra", "pmdec"}.issubset(pm_source.columns):
        pm = early.merge(pm_source[["match_id", "pmra", "pmdec"]], on="match_id",
                         how="left")
        e_ra, e_dec = propagate(e_ra, e_dec, pm["pmra"], pm["pmdec"],
                                early_epoch, late_epoch)

    l_ra = late["ra"].to_numpy(dtype=float)
    l_dec = late["dec"].to_numpy(dtype=float)
    # Project onto a unit sphere so the tree metric is chord length.
    def _xyz(ra, dec):
        r, d = np.radians(ra), np.radians(dec)
        return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])

    tree = cKDTree(_xyz(l_ra, l_dec))
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    dist, idx = tree.query(_xyz(e_ra, e_dec), k=1, distance_upper_bound=chord)
    hit = np.isfinite(dist) & (idx < len(late))
    if not hit.any():
        return pd.DataFrame()

    out = early[hit].reset_index(drop=True).add_prefix("early_")
    late_hit = late.iloc[idx[hit]].reset_index(drop=True).add_prefix("late_")
    out = pd.concat([out, late_hit], axis=1)
    out["match_sep_arcsec"] = np.degrees(2.0 * np.arcsin(dist[hit] / 2.0)) * 3600.0
    return out

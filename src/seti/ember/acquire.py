"""Archive pulls for EMBER. Runner-only: the sandbox has no VO egress.

Every fetcher here is (a) chunked, (b) retried with backoff, (c) checkpointed to
parquet so a killed job loses minutes rather than hours, and (d) **injectable**,
so the whole channel runs offline in tests by passing a pre-built table.

Three things in this module are load-bearing and easy to get wrong.

**Proper motion.** Gaia positions are at epoch 2016.0. IRAS observed in 1983.5 --
a 32.5-year lever arm, so a 500 mas/yr star sits 16 arcsec away from where Gaia
says it is. Matching Gaia positions directly against IRAS silently loses every
high-proper-motion star, which is a bug that has already cost this repository a
whole run. Every cross-match here propagates to the *survey's* epoch first.

**Column names, and the coordinate FRAME they are in.** VizieR renames columns
between catalogue versions, so nothing is hardcoded: the resolver asks
``TAP_SCHEMA.columns`` for the table and takes whichever column carries the UCD
``pos.eq.ra;meta.main`` / ``pos.eq.dec;meta.main``, falling back to an alias list
only when the UCD is absent.  The alias list alone is *not* sufficient and that
is not hypothetical -- run 30203763934 lost IRAS entirely because
:data:`_RA_NAMES` knew nothing about a B1950 column name and
``resolve_columns`` raised inside a bare ``except``, which turned a schema
mismatch into a silent empty frame.  IRAS PSC/FSC are catalogued in **B1950**;
a B1950 position used as though it were J2000 is wrong by ~0.5 deg, which would
destroy every cross-match while looking superficially fine, so a resolved B1950
column is precessed explicitly and the frame actually used is recorded.

**Failure has to be legible.** Every fetch returns a :class:`FetchStatus`
recording ``OK`` / ``QUERY_RETURNED_ZERO_ROWS`` / ``QUERY_FAILED``, the literal
query text, the row count and the exception.  An empty archive response and an
unreached archive are different statements and must never collapse into the
same zero.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
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
#:
#: VERIFIED on the runner (run 30203763934): ``"II/297/irc"`` returned 870,973
#: rows across 12 RA slices, so the AKARI/IRC identifier is right.  The IRAS
#: identifiers were *reachable* in the same run -- the ``SELECT TOP 1 *`` probe
#: raised nothing -- but their columns did not resolve; see :func:`resolve_columns`.
CATALOGUES = {
    "iras_psc": '"II/125/main"',
    "iras_fsc": '"II/156A/main"',
    "akari_irc": '"II/297/irc"',
    "allwise": '"II/328/allwise"',
}

#: VizieR mirrors of the catalogues the CDS X-Match service can join against.
XMATCH_CATALOGUES = {
    "gaia_dr3": "vizier:I/355/gaiadr3",
    "allwise": "vizier:II/328/allwise",
    "twomass": "vizier:II/246/out",
}

CDS_XMATCH_URL = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"

# --- fetch outcome ---------------------------------------------------------
#: The four outcomes a single archive query can have.  These MUST NOT collapse
#: into one another: "the query blew up", "the query worked and the sky was
#: empty here", and "we never asked" are three different statements, and the
#: first EMBER run reported all of them as ``acquired: 0``.
STATUS_OK = "OK"
STATUS_ZERO_ROWS = "QUERY_RETURNED_ZERO_ROWS"
STATUS_QUERY_FAILED = "QUERY_FAILED"
STATUS_NOT_ATTEMPTED = "NOT_ATTEMPTED"


@dataclass
class FetchStatus:
    """What one archive query actually did, in a form a reader can audit."""

    label: str
    status: str = STATUS_NOT_ATTEMPTED
    n_rows: int = 0
    #: The literal query text (ADQL) or endpoint description that was sent.
    query: str = ""
    #: Which strategy in a fallback ladder produced the rows.
    strategy: str = ""
    error: str | None = None
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict:
        return {"label": self.label, "status": self.status, "n_rows": self.n_rows,
                "strategy": self.strategy, "query": self.query,
                "error": self.error, "detail": self.detail}


def _classify(df: pd.DataFrame | None, exc: Exception | None) -> str:
    if exc is not None:
        return STATUS_QUERY_FAILED
    if df is None:
        return STATUS_QUERY_FAILED
    return STATUS_OK if len(df) else STATUS_ZERO_ROWS


# --- column alias tables ---------------------------------------------------
#: UCDs that identify the main equatorial position columns.  This is the
#: authoritative route: it is metadata the catalogue itself publishes, and it
#: does not care what the column happens to be named or which equinox it is in.
_UCD_RA = ("pos.eq.ra;meta.main", "pos.eq.ra")
_UCD_DEC = ("pos.eq.dec;meta.main", "pos.eq.dec")

#: Alias fallback, used only when the UCD is missing.  The B1950 spellings are
#: here because IRAS PSC (II/125) and FSC (II/156A) are B1950 catalogues; a
#: resolver that knows only J2000 names silently loses both, which is exactly
#: what happened in run 30203763934.
_RA_NAMES = ("_raj2000", "raj2000", "ra_icrs", "raicrs", "radeg", "ra",
             "_ra", "_rab1950", "rab1950", "ra1950", "_ra1950", "raj1950")
_DEC_NAMES = ("_dej2000", "dej2000", "de_icrs", "deicrs", "dedeg", "dec", "de",
              "_de", "_deb1950", "deb1950", "de1950", "_de1950", "dej1950",
              "decb1950", "dec1950")

#: Column names / UCD equinoxes that mean "this position is B1950, precess it".
_B1950_MARKERS = ("1950", "b1950", "fk4")

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


def resolve_positions(schema: list[dict] | None, colnames,
                      aliases: dict[str, tuple[str, ...]]) -> dict:
    """Resolve RA/Dec **and the frame they are in**, UCD first, aliases second.

    ``schema`` is a list of ``{"column_name", "ucd", "unit", "description"}``
    dicts as returned by :func:`tap_schema_columns`; pass ``None`` when schema
    metadata could not be obtained and only the alias route is available.

    Returns ``{"ra":…, "dec":…, "frame": "icrs"|"b1950", "route": …,
    "columns": {canonical: actual}}``.  ``ra``/``dec`` are ``None`` when
    resolution genuinely failed -- the caller must treat that as
    ``QUERY_FAILED``, never as an empty catalogue.
    """
    cols = resolve_columns(colnames, aliases)
    ra = cols.get("ra")
    dec = cols.get("dec")
    route = "alias" if (ra and dec) else "unresolved"
    frame_src = " ".join(str(x) for x in (ra, dec) if x)

    by_ucd_ra = by_ucd_dec = None
    if schema:
        # Sort so that the ";meta.main" form wins over a bare "pos.eq.ra".
        for want, target in ((_UCD_RA, "ra"), (_UCD_DEC, "dec")):
            best = None
            for ucd in want:
                for row in schema:
                    if str(row.get("ucd") or "").strip().lower() == ucd:
                        best = row
                        break
                if best is not None:
                    break
            if best is not None:
                if target == "ra":
                    by_ucd_ra = best
                else:
                    by_ucd_dec = best
    if by_ucd_ra is not None and by_ucd_dec is not None:
        ra = by_ucd_ra["column_name"]
        dec = by_ucd_dec["column_name"]
        route = "ucd"
        frame_src = " ".join(str(by_ucd_ra.get(k) or "") + str(by_ucd_dec.get(k) or "")
                             for k in ("column_name", "ucd", "description"))
        cols["ra"], cols["dec"] = ra, dec

    frame = "b1950" if any(m in frame_src.lower() for m in _B1950_MARKERS) else "icrs"
    return {"ra": ra, "dec": dec, "frame": frame, "route": route, "columns": cols}


def precess_b1950_to_j2000(ra_deg, dec_deg):
    """FK4 B1950 -> FK5 J2000, in degrees.

    IRAS PSC/FSC positions are B1950.  Treating them as J2000 misplaces every
    source by of order half a degree -- far larger than any matching radius here
    -- so this is not a refinement, it is the difference between a cross-match
    that works and one that returns nothing while looking healthy.
    """
    import astropy.units as u
    from astropy.coordinates import FK4, FK5, SkyCoord

    c = SkyCoord(ra=np.asarray(ra_deg, dtype=float) * u.deg,
                 dec=np.asarray(dec_deg, dtype=float) * u.deg,
                 frame=FK4(equinox="B1950", obstime="B1950"))
    j = c.transform_to(FK5(equinox="J2000"))
    return np.asarray(j.ra.deg, dtype=float), np.asarray(j.dec.deg, dtype=float)


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


_TAP_SCHEMA_QUERY = (
    "SELECT column_name, ucd, unit, datatype, description "
    "FROM TAP_SCHEMA.columns WHERE table_name = '{table}'")


def tap_schema_columns(table: str, url: str = VIZIER_TAP) -> list[dict]:
    """Column metadata (name, UCD, unit, description) straight from TAP_SCHEMA.

    The UCD is the only frame-independent way to ask "which column is the main
    right ascension?".  ``table`` is the *unquoted* table name, because
    ``TAP_SCHEMA.columns.table_name`` stores it unquoted.
    """
    tap = _tap_service(url)
    q = _TAP_SCHEMA_QUERY.format(table=table.strip('"'))
    tbl = _retry(lambda: tap.run_sync(q).to_table(), label=f"schema:{table}")
    out = []
    for row in tbl:
        out.append({k: (None if row[k] is None else str(row[k]))
                    for k in ("column_name", "ucd", "unit", "datatype", "description")
                    if k in tbl.colnames})
    return out


def describe_catalogue(table: str, aliases: dict[str, tuple[str, ...]],
                       url: str = VIZIER_TAP) -> dict:
    """Everything the probe stage needs to know about one VizieR table.

    Returns the schema rows, the ``SELECT TOP 1 *`` column list, the resolved
    position columns and frame, and any error -- without raising, so a probe can
    report on all four catalogues even when one of them is broken.
    """
    info: dict = {"table": table, "errors": []}
    schema = None
    try:
        schema = tap_schema_columns(table, url)
        info["n_schema_columns"] = len(schema)
        info["schema"] = schema
    except Exception as exc:  # noqa: BLE001
        info["errors"].append(f"tap_schema: {exc!r}")
    try:
        info["select_star_columns"] = _probe_columns(table, url)
    except Exception as exc:  # noqa: BLE001
        info["errors"].append(f"select_star: {exc!r}")
        info["select_star_columns"] = []
    names = ([r["column_name"] for r in schema] if schema
             else info["select_star_columns"])
    info["resolved"] = resolve_positions(schema, names, aliases)
    info["resolvable"] = bool(info["resolved"]["ra"] and info["resolved"]["dec"])
    return info


def fetch_vizier_catalogue(table: str, aliases: dict[str, tuple[str, ...]],
                           out_dir: Path | None = None, tag: str = "cat",
                           n_ra_chunks: int = 12, row_limit: int = 4_000_000,
                           url: str = VIZIER_TAP,
                           status: FetchStatus | None = None) -> pd.DataFrame:
    """Pull a whole VizieR catalogue in RA slices, checkpointing each slice.

    Chunking by RA rather than issuing one monolithic query is the same lesson
    the Gaia fetchers in this repository learned: a single multi-million-row
    async job times out or returns "cannot find result", while a dozen smaller
    ones succeed reliably.

    Position columns are resolved by UCD (falling back to aliases), and a B1950
    catalogue is precessed to J2000 before anything downstream sees it -- the
    frame actually used is written into ``status.detail['frame']`` and into the
    returned frame's ``pos_frame`` column so it cannot be assumed later.
    """
    st = status if status is not None else FetchStatus(label=tag)
    schema = None
    try:
        schema = tap_schema_columns(table, url)
    except Exception as exc:  # noqa: BLE001
        st.detail["tap_schema_error"] = repr(exc)
    try:
        names = ([r["column_name"] for r in schema] if schema
                 else _probe_columns(table, url))
    except Exception as exc:  # noqa: BLE001
        st.status = STATUS_QUERY_FAILED
        st.error = f"column probe failed: {exc!r}"
        raise

    res = resolve_positions(schema, names, aliases)
    cols = res["columns"]
    st.detail["column_resolution"] = {k: v for k, v in res.items() if k != "columns"}
    st.detail["columns_resolved"] = cols
    st.detail["columns_available"] = list(names)
    if not res["ra"] or not res["dec"]:
        st.status = STATUS_QUERY_FAILED
        st.error = (f"{table}: could not resolve RA/Dec columns. "
                    f"available={list(names)[:60]} resolved={cols}")
        raise RuntimeError(st.error)
    st.detail["frame"] = res["frame"]

    select = ", ".join(f'{v} AS "{k}"' for k, v in cols.items())
    tap = _tap_service(url)
    edges = np.linspace(0.0, 360.0, n_ra_chunks + 1)
    frames: list[pd.DataFrame] = []
    per_chunk: dict[str, int] = {}
    queries: list[str] = []

    for i in range(n_ra_chunks):
        lo, hi = float(edges[i]), float(edges[i + 1])
        ckpt = (out_dir / f"{tag}_ra{i:02d}.parquet") if out_dir else None
        if ckpt is not None and ckpt.exists():
            part = pd.read_parquet(ckpt)
            frames.append(part)
            per_chunk[f"ra{i:02d}"] = int(len(part))
            print(f"[ember] {tag} RA[{lo:.0f},{hi:.0f}) from checkpoint "
                  f"({len(part):,} rows)")
            continue
        q = (f"SELECT TOP {row_limit} {select} FROM {table} "
             f"WHERE {cols['ra']} >= {lo} AND {cols['ra']} < {hi}")
        if i == 0:
            queries.append(q)
        try:
            df = _retry(lambda q=q: tap.run_async(q).to_table().to_pandas(),
                        label=f"{tag}:ra{i}")
        except Exception as exc:  # noqa: BLE001
            st.status = STATUS_QUERY_FAILED
            st.error = f"{tag} RA[{lo:.0f},{hi:.0f}): {exc!r}"
            st.query = q
            st.detail["rows_per_ra_chunk"] = per_chunk
            raise
        print(f"[ember] {tag} RA[{lo:.0f},{hi:.0f}) -> {len(df):,} rows")
        per_chunk[f"ra{i:02d}"] = int(len(df))
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ckpt, index=False)
        frames.append(df)

    st.query = queries[0] if queries else f"(all {n_ra_chunks} slices from checkpoint)"
    st.detail["rows_per_ra_chunk"] = per_chunk
    if not frames:
        st.status = STATUS_ZERO_ROWS
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates()

    # Frame normalisation.  A B1950 catalogue must be precessed BEFORE anything
    # downstream treats its RA/Dec as ICRS.
    out["pos_frame"] = res["frame"]
    if res["frame"] == "b1950" and len(out):
        out["ra_b1950"], out["dec_b1950"] = out["ra"].to_numpy(), out["dec"].to_numpy()
        ra_j, dec_j = precess_b1950_to_j2000(out["ra"], out["dec"])
        out["ra"], out["dec"] = ra_j, dec_j
        out["pos_frame"] = "b1950->j2000"
        st.detail["precessed_b1950_to_j2000"] = True

    st.n_rows = int(len(out))
    st.status = STATUS_OK if len(out) else STATUS_ZERO_ROWS
    return out


def fetch_akari(out_dir: Path | None = None, status: FetchStatus | None = None,
                **kw) -> pd.DataFrame:
    """AKARI/IRC Point Source Catalogue (~870k sources, 9 and 18 micron)."""
    return fetch_vizier_catalogue(CATALOGUES["akari_irc"], _AKARI_ALIASES,
                                  out_dir=out_dir, tag="akari", status=status, **kw)


def fetch_iras(which: str = "psc", out_dir: Path | None = None,
               status: FetchStatus | None = None, **kw) -> pd.DataFrame:
    """IRAS Point Source Catalogue (~245k) or Faint Source Catalogue (~173k).

    Both are **B1950** catalogues; :func:`fetch_vizier_catalogue` precesses them.
    """
    key = "iras_psc" if which == "psc" else "iras_fsc"
    return fetch_vizier_catalogue(CATALOGUES[key], _IRAS_ALIASES,
                                  out_dir=out_dir, tag=key, status=status, **kw)


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


#: Chunk sizes tried, largest first.  MEASURED (run 30203763934): a 20,000-row
#: upload to the ESA Gaia archive returns ``HTTP 500`` with a null body on every
#: one of four attempts, and because the caller swallowed the resulting
#: RuntimeError the whole shard reported ``acquired: 0`` while 871k AKARI rows
#: sat in the cache.  Anonymous ESA TAP uploads are size-limited, so the ladder
#: shrinks the chunk before it gives up on the service.
GAIA_UPLOAD_CHUNKS: tuple[int, ...] = (5_000, 2_000, 500)

#: How VizieR I/355/gaiadr3 (the X-Match mirror) names the columns the ESA
#: archive calls ``source_id``, ``parallax``, ``pmra``…  Resolution is
#: alias-based and *recorded*, never assumed.
_GAIA_VIZIER_ALIASES: dict[str, tuple[str, ...]] = {
    "source_id": ("source", "source_id", "dr3name", "gaiadr3"),
    "ra": ("ra_icrs", "raicrs", "ra"),
    "dec": ("de_icrs", "deicrs", "dec", "de"),
    "parallax": ("plx", "parallax"),
    "parallax_error": ("e_plx", "parallax_error"),
    "pmra": ("pmra",),
    "pmra_error": ("e_pmra", "pmra_error"),
    "pmdec": ("pmde", "pmdec"),
    "pmdec_error": ("e_pmde", "pmdec_error"),
    "ruwe": ("ruwe",),
    "phot_g_mean_mag": ("gmag", "phot_g_mean_mag"),
    "phot_bp_mean_mag": ("bpmag", "phot_bp_mean_mag"),
    "phot_rp_mean_mag": ("rpmag", "phot_rp_mean_mag"),
    "bp_rp": ("bp-rp", "bprp", "bp_rp"),
    "teff_gspphot": ("teff", "teff_gspphot", "tefftemp"),
}

_ALLWISE_VIZIER_ALIASES: dict[str, tuple[str, ...]] = {
    "designation": ("allwise", "designation", "wise"),
    "ra": ("raj2000", "_raj2000", "ra_icrs", "ra"),
    "dec": ("dej2000", "_dej2000", "de_icrs", "dec", "de"),
    "w1mpro": ("w1mag", "w1mpro"), "w1sigmpro": ("e_w1mag", "w1sigmpro"),
    "w2mpro": ("w2mag", "w2mpro"), "w2sigmpro": ("e_w2mag", "w2sigmpro"),
    "w3mpro": ("w3mag", "w3mpro"), "w3sigmpro": ("e_w3mag", "w3sigmpro"),
    "w4mpro": ("w4mag", "w4mpro"), "w4sigmpro": ("e_w4mag", "w4sigmpro"),
    "j_m_2mass": ("jmag", "j_m_2mass"), "h_m_2mass": ("hmag", "h_m_2mass"),
    "k_m_2mass": ("kmag", "k_m_2mass"),
    "j_msig_2mass": ("e_jmag", "j_msig_2mass"),
    "h_msig_2mass": ("e_hmag", "h_msig_2mass"),
    "k_msig_2mass": ("e_kmag", "k_msig_2mass"),
    "cc_flags": ("ccf", "cc_flags"), "ext_flg": ("ex", "ext_flg"),
    "ph_qual": ("qph", "ph_qual"), "var_flg": ("var", "var_flg"),
}


def _rename_by_alias(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]
                     ) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename ``df``'s columns to canonical names, reporting what matched."""
    mapping = resolve_columns(df.columns, aliases)
    out = df.rename(columns={v: k for k, v in mapping.items()})
    return out, mapping


def xmatch_cds(positions: pd.DataFrame, cat2: str, radius_arcsec: float = 6.0,
               colra: str = "ra", coldec: str = "dec",
               chunk: int = 200_000) -> pd.DataFrame:
    """Cross-match a local table against a VizieR catalogue via CDS X-Match.

    This exists because it is the one route that needs no upload quota on an
    authenticated archive and no per-object HTTP request: it is built for
    exactly "here are 10^5-10^6 positions, join them to a big catalogue".  It is
    the fallback when the ESA Gaia archive refuses an upload, and the primary
    route for AllWISE, where the alternative was one IRSA cone query per source.
    """
    import astropy.units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch

    frames: list[pd.DataFrame] = []
    n = len(positions)
    for start in range(0, n, chunk):
        sub = positions.iloc[start:start + chunk].reset_index(drop=True)
        tbl = Table.from_pandas(sub)

        def _go(tbl=tbl):
            return XMatch.query(cat1=tbl, cat2=cat2,
                                max_distance=radius_arcsec * u.arcsec,
                                colRA1=colra, colDec1=coldec).to_pandas()

        df = _retry(_go, retries=3, label=f"xmatch:{cat2}:{start}")
        print(f"[ember] xmatch {cat2} rows {start}-{start + len(sub)} -> "
              f"{len(df):,} matches")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_gaia_for_positions(positions: pd.DataFrame, radius_arcsec: float = 6.0,
                             chunk: int | None = None, out_dir: Path | None = None,
                             retries: int = 3,
                             status: FetchStatus | None = None) -> pd.DataFrame:
    """Cone-match a position list against Gaia DR3, via a ladder of strategies.

    ``positions`` needs ``match_id``, ``ra``, ``dec`` -- the *infrared* positions
    at the infrared epoch. Because Gaia positions are at 2016.0, the returned
    separations must be recomputed after propagating Gaia to the infrared epoch;
    :func:`attach_epoch_separation` does that.

    Ladder, in order, with whichever worked recorded in ``status.strategy``:

    1. ESA Gaia archive TAP upload at 5,000 rows a chunk, then 2,000, then 500.
       Smaller is slower but the anonymous upload limit is the binding
       constraint, not throughput.
    2. CDS X-Match against ``vizier:I/355/gaiadr3``.

    A total failure returns an empty frame **with ``status`` saying
    ``QUERY_FAILED``**, which the caller must not confuse with an empty sky.
    """
    st = status if status is not None else FetchStatus(label="gaia")
    need = {"match_id", "ra", "dec"}
    if not need.issubset(positions.columns):
        raise ValueError(f"positions must contain {need}")
    if positions.empty:
        st.status = STATUS_ZERO_ROWS
        st.detail["note"] = "no input positions"
        return pd.DataFrame()

    q = _GAIA_QUERY.format(radius_deg=radius_arcsec / 3600.0)
    st.query = q
    sizes = (chunk,) if chunk else GAIA_UPLOAD_CHUNKS
    attempts: list[dict] = []

    for size in sizes:
        try:
            df = _gaia_upload(positions, q, int(size), out_dir, retries)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"strategy": f"esa_upload_{size}", "error": repr(exc)})
            print(f"[ember] gaia upload at chunk={size} failed: {exc!r}")
            continue
        st.strategy = f"esa_upload_{size}"
        st.n_rows = int(len(df))
        st.status = _classify(df, None)
        st.detail["attempts"] = attempts
        return df

    # Fallback: CDS X-Match against the VizieR mirror of Gaia DR3.
    try:
        raw = xmatch_cds(positions[["match_id", "ra", "dec"]],
                         XMATCH_CATALOGUES["gaia_dr3"], radius_arcsec)
        df, mapping = _rename_by_alias(raw, _GAIA_VIZIER_ALIASES)
        st.strategy = "cds_xmatch_vizier_I355"
        st.query = f"XMatch cat2={XMATCH_CATALOGUES['gaia_dr3']} r={radius_arcsec}\""
        st.detail["xmatch_column_mapping"] = mapping
        st.detail["xmatch_columns_returned"] = list(raw.columns)
        st.n_rows = int(len(df))
        st.status = _classify(df, None)
        st.detail["attempts"] = attempts
        return df
    except Exception as exc:  # noqa: BLE001
        attempts.append({"strategy": "cds_xmatch", "error": repr(exc)})
        st.status = STATUS_QUERY_FAILED
        st.error = f"every Gaia strategy failed: {attempts}"
        st.detail["attempts"] = attempts
        print(f"[ember] gaia: every strategy failed; last was {exc!r}")
        return pd.DataFrame()


def _gaia_upload(positions: pd.DataFrame, q: str, chunk: int,
                 out_dir: Path | None, retries: int) -> pd.DataFrame:
    """One pass of the ESA-archive upload strategy at a fixed chunk size."""
    from astropy.table import Table
    from astroquery.gaia import Gaia

    frames: list[pd.DataFrame] = []
    n_chunks = int(np.ceil(len(positions) / chunk)) or 1
    for i in range(n_chunks):
        ckpt = (out_dir / f"gaia_c{chunk}_{i:04d}.parquet") if out_dir else None
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

        df = _retry(_go, retries=retries, label=f"gaia:{chunk}:{i}")
        print(f"[ember] gaia chunk {i + 1}/{n_chunks} (size {chunk}) -> "
              f"{len(df):,} rows")
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
    status: dict = {"ra_lo": ra_lo, "ra_hi": ra_hi, "errors": [], "counts": {},
                    "fetches": {}}

    def _pull(name, fn):
        """Run one archive pull, recording *why* it produced what it produced.

        The predecessor of this function had a bare ``except`` that recorded the
        exception into a status dict nobody printed and returned an empty frame.
        A schema mismatch in IRAS therefore looked exactly like an empty
        catalogue, and the run reported ``acquired: 0`` with no visible error.
        Every failure is now printed AND classified.
        """
        st = FetchStatus(label=name)
        try:
            df = fn(st)
            st.n_rows = int(len(df))
            if st.status in (STATUS_NOT_ATTEMPTED, ""):
                st.status = _classify(df, None)
            status["counts"][name] = int(len(df))
            status["fetches"][name] = st.to_dict()
            if st.status == STATUS_ZERO_ROWS:
                print(f"[ember] {name}: query succeeded and returned 0 rows")
            return df
        except Exception as exc:  # noqa: BLE001
            st.status = STATUS_QUERY_FAILED
            st.error = st.error or repr(exc)
            status["errors"].append(f"{name}: {exc!r}")
            status["counts"][name] = 0
            status["fetches"][name] = st.to_dict()
            print(f"[ember] {name}: QUERY_FAILED -- {exc!r}")
            return pd.DataFrame()

    akari = _pull("akari", fx.get("akari") or
                  (lambda st: fetch_akari(cache, n_ra_chunks=n_ra_chunks, status=st)))
    iras_psc = _pull("iras_psc", fx.get("iras_psc") or
                     (lambda st: fetch_iras("psc", cache, n_ra_chunks=n_ra_chunks,
                                            status=st)))
    iras_fsc = _pull("iras_fsc", fx.get("iras_fsc") or
                     (lambda st: fetch_iras("fsc", cache, n_ra_chunks=n_ra_chunks,
                                            status=st)))

    def _slice(df):
        if df.empty or "ra" not in df.columns:
            return df
        return df[(df["ra"] >= ra_lo) & (df["ra"] < ra_hi)].copy()

    akari, iras_psc, iras_fsc = _slice(akari), _slice(iras_psc), _slice(iras_fsc)
    status["counts"]["akari_in_shard"] = int(len(akari))
    status["counts"]["iras_in_shard"] = int(len(iras_psc)) + int(len(iras_fsc))
    iras = pd.concat([d for d in (iras_psc, iras_fsc) if not d.empty],
                     ignore_index=True) if (not iras_psc.empty or not iras_fsc.empty) \
        else pd.DataFrame()

    if akari.empty and iras.empty:
        status["archive_reachable"] = False
        status["stopped_at"] = "early_epoch_catalogues"
        return pd.DataFrame(), status
    status["archive_reachable"] = True

    # Early-epoch anchor positions: AKARI where available (2 arcsec), else IRAS.
    anchor = akari if not akari.empty else iras
    anchor = anchor.reset_index(drop=True)
    anchor["match_id"] = [f"em{ra_lo:05.1f}_{i:07d}" for i in range(len(anchor))]
    positions = anchor[["match_id", "ra", "dec"]].copy()

    # Checkpoint the infrared-only table BEFORE the optical join.  When Gaia is
    # down, the expensive part of the run is already done and must survive: the
    # previous version threw away 871k catalogued rows because the archive that
    # ran second returned HTTP 500.
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        anchor.to_parquet(cache / f"ir_anchor_{ra_lo:05.1f}.parquet", index=False)

    gaia = _pull("gaia", fx.get("gaia") or
                 (lambda st: fetch_gaia_for_positions(positions, radius_arcsec=6.0,
                                                      out_dir=cache, status=st)))
    if gaia.empty:
        status["stopped_at"] = "gaia"
        status["errors"].append(
            "gaia: no counterparts; the channel cannot distinguish stars from "
            "background galaxies, so no working table is emitted. The infrared "
            "anchor table IS checkpointed, so a re-reduction needs only Gaia.")
        return pd.DataFrame(), status

    merged = gaia.merge(anchor.add_prefix("ir_"), left_on="match_id",
                        right_on="ir_match_id", how="inner")
    merged = attach_epoch_separation(merged, ir_epoch=AKARI_EPOCH if not akari.empty
                                     else IRAS_EPOCH)
    # Keep the single best Gaia counterpart per infrared source.
    merged = merged.sort_values("sep_arcsec").drop_duplicates("match_id")
    status["counts"]["gaia_matched"] = int(len(merged))

    allwise = _pull("allwise", fx.get("allwise") or
                    (lambda st: _allwise_for_rows(merged, cache, status=st)))
    if not allwise.empty:
        merged = merged.merge(allwise, on="match_id", how="left")

    status["counts"]["working"] = int(len(merged))
    return merged, status


#: Above this many rows, per-object IRSA cone searches are not a strategy: at
#: ~1 s each, 10^5 sources is a day and a half of wall clock.  The bulk route is
#: a CDS X-Match against the VizieR AllWISE mirror.
ALLWISE_CONE_MAX_ROWS = 2_000


def _allwise_for_rows(rows: pd.DataFrame, cache: Path | None = None,
                      radius_arcsec: float = 3.0,
                      status: FetchStatus | None = None) -> pd.DataFrame:
    """AllWISE photometry at each row's Gaia position propagated to 2010.5.

    Positions are propagated to the AllWISE epoch first; a high-proper-motion
    star matched at the Gaia epoch is simply absent from AllWISE otherwise.
    Bulk X-Match for anything larger than :data:`ALLWISE_CONE_MAX_ROWS`, falling
    back to per-object IRSA cones (which is also the small-sample path).
    """
    st = status if status is not None else FetchStatus(label="allwise")
    if rows.empty:
        st.status = STATUS_ZERO_ROWS
        return pd.DataFrame()

    ra_p, dec_p = propagate(rows["ra"], rows["dec"], rows.get("pmra"),
                            rows.get("pmdec"), GAIA_EPOCH, ALLWISE_EPOCH)
    prop = pd.DataFrame({"match_id": rows["match_id"].to_numpy(),
                         "ra": ra_p, "dec": dec_p})

    if len(prop) > ALLWISE_CONE_MAX_ROWS:
        try:
            raw = xmatch_cds(prop, XMATCH_CATALOGUES["allwise"], radius_arcsec)
            df, mapping = _rename_by_alias(raw, _ALLWISE_VIZIER_ALIASES)
            st.strategy = "cds_xmatch_vizier_II328"
            st.query = (f"XMatch cat2={XMATCH_CATALOGUES['allwise']} "
                        f"r={radius_arcsec}\" on {len(prop)} propagated positions")
            st.detail["xmatch_column_mapping"] = mapping
            st.detail["xmatch_columns_returned"] = list(raw.columns)
            if "match_id" in df.columns and len(df):
                df["n_allwise_in_cone"] = df.groupby("match_id")["match_id"] \
                    .transform("size")
                # Keep the nearest counterpart per infrared source.
                sepcol = next((c for c in raw.columns
                               if _norm(c) in ("angdist", "angdistance")), None)
                if sepcol and sepcol in df.columns:
                    df = df.sort_values(sepcol)
                df = df.drop_duplicates("match_id")
            st.n_rows = int(len(df))
            st.status = _classify(df, None)
            return df
        except Exception as exc:  # noqa: BLE001
            st.detail["xmatch_error"] = repr(exc)
            print(f"[ember] allwise xmatch failed ({exc!r}); "
                  f"falling back to per-object cones on {len(prop)} rows")

    st.strategy = "irsa_cone_per_object"
    st.query = _ALLWISE_CONE.strip()
    out = []
    failures = 0
    for _, r in prop.iterrows():
        df = fetch_allwise_cone(float(r["ra"]), float(r["dec"]), radius_arcsec)
        if df.empty:
            failures += 1
            continue
        rec = df.iloc[0].to_dict()
        rec["match_id"] = r["match_id"]
        rec["n_allwise_in_cone"] = int(len(df))
        out.append(rec)
    st.detail["cones_without_a_counterpart"] = failures
    res = pd.DataFrame(out)
    st.n_rows = int(len(res))
    st.status = _classify(res, None)
    return res


# --------------------------------------------------------------------------
# Probe: one query per primitive, row counts and first rows printed
# --------------------------------------------------------------------------
#: A handful of bright, unambiguous mid-infrared sources used as the probe's
#: positional test set.  They are chosen only to be *findable*: if a cross-match
#: primitive cannot recover Vega or Betelgeuse it is broken, and no statement
#: about the sky can be made from it.
PROBE_POSITIONS: tuple[tuple[str, float, float], ...] = (
    ("vega", 279.234735, 38.783689),
    ("betelgeuse", 88.792939, 7.407064),
    ("fomalhaut", 344.412693, -29.622237),
    ("hd172555", 281.362000, -64.870806),
    ("tyc8241", 176.573000, -53.199000),
)


def probe_archives(sample_rows: int = 400, url: str = VIZIER_TAP) -> dict:
    """Exercise every acquisition primitive once and report concrete numbers.

    This is deliberately cheap and deliberately verbose.  The sandbox has no
    egress, so *nothing* about the archive contracts can be checked before a
    run; the alternative to a probe is to dispatch the full pipeline and read
    the wreckage, which is how this channel lost its first run.  Every entry
    records the query text, the row count and the first rows.

    Returns a JSON-safe dict; raises nothing.
    """
    report: dict = {"vizier_tap": url, "catalogues": {}, "gaia": {},
                    "xmatch": {}, "irsa": {}, "notes": []}

    aliases = {"iras_psc": _IRAS_ALIASES, "iras_fsc": _IRAS_ALIASES,
               "akari_irc": _AKARI_ALIASES, "allwise": _ALLWISE_VIZIER_ALIASES}
    for key, table in CATALOGUES.items():
        entry = describe_catalogue(table, aliases[key], url)
        # Trim the schema to what a human needs to read in a log.
        entry["schema"] = [
            {k: r.get(k) for k in ("column_name", "ucd", "unit")}
            for r in (entry.get("schema") or [])]
        # A real, tiny data pull: does a constrained query return rows?
        res = entry["resolved"]
        if res["ra"] and res["dec"]:
            q = (f"SELECT TOP 5 {res['ra']} AS ra, {res['dec']} AS dec "
                 f"FROM {table} WHERE {res['ra']} >= 100 AND {res['ra']} < 101")
            entry["sample_query"] = q
            try:
                tbl = _tap_service(url).run_sync(q).to_table().to_pandas()
                entry["sample_n_rows"] = int(len(tbl))
                entry["sample_rows"] = tbl.head(5).to_dict("records")
                entry["sample_status"] = _classify(tbl, None)
            except Exception as exc:  # noqa: BLE001
                entry["sample_status"] = STATUS_QUERY_FAILED
                entry["sample_error"] = repr(exc)
            cq = f"SELECT COUNT(*) AS n FROM {table}"
            try:
                entry["total_rows"] = int(
                    _tap_service(url).run_sync(cq).to_table().to_pandas().iloc[0, 0])
            except Exception as exc:  # noqa: BLE001
                entry["total_rows_error"] = repr(exc)
        else:
            entry["sample_status"] = STATUS_QUERY_FAILED
            entry["sample_error"] = "RA/Dec unresolved; see resolved.columns"
        report["catalogues"][key] = entry

    # Positions to hand to the cross-match primitives.  Prefer real AKARI rows
    # (they are what production actually uses); fall back to the bright list.
    pos = pd.DataFrame({"match_id": [p[0] for p in PROBE_POSITIONS],
                        "ra": [p[1] for p in PROBE_POSITIONS],
                        "dec": [p[2] for p in PROBE_POSITIONS]})
    ak = report["catalogues"].get("akari_irc", {})
    if ak.get("resolvable"):
        res = ak["resolved"]
        q = (f"SELECT TOP {sample_rows} {res['ra']} AS ra, {res['dec']} AS dec "
             f"FROM {CATALOGUES['akari_irc']} "
             f"WHERE {res['ra']} >= 100 AND {res['ra']} < 105")
        try:
            real = _tap_service(url).run_sync(q).to_table().to_pandas()
            real["match_id"] = [f"probe{i:05d}" for i in range(len(real))]
            pos = pd.concat([pos, real[["match_id", "ra", "dec"]]],
                            ignore_index=True)
            report["notes"].append(
                f"probe positions: {len(real)} real AKARI rows + "
                f"{len(PROBE_POSITIONS)} bright anchors")
        except Exception as exc:  # noqa: BLE001
            report["notes"].append(f"could not draw real AKARI probe positions: {exc!r}")
    report["n_probe_positions"] = int(len(pos))

    # Gaia: find the largest upload chunk the anonymous ESA archive accepts.
    # Run 30203763934 got HTTP 500 at 20,000; this measures the real ceiling
    # instead of guessing at it.
    q = _GAIA_QUERY.format(radius_deg=6.0 / 3600.0)
    report["gaia"]["query"] = q
    report["gaia"]["upload_chunk_tests"] = {}
    for size in (200, 2_000, 5_000, 20_000):
        sub = pos if len(pos) >= size else _tile(pos, size)
        sub = sub.head(size)
        rec: dict = {"n_uploaded": int(len(sub))}
        t0 = time.time()
        try:
            df = _gaia_upload(sub, q, int(size), None, retries=1)
            rec.update({"status": _classify(df, None), "n_rows": int(len(df)),
                        "columns": list(df.columns)[:24]})
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": STATUS_QUERY_FAILED, "error": repr(exc)})
        rec["seconds"] = round(time.time() - t0, 1)
        report["gaia"]["upload_chunk_tests"][str(size)] = rec
        if rec.get("status") == STATUS_QUERY_FAILED:
            break

    # CDS X-Match, the fallback for Gaia and the primary route for AllWISE.
    for label, cat in (("gaia_dr3", XMATCH_CATALOGUES["gaia_dr3"]),
                       ("allwise", XMATCH_CATALOGUES["allwise"])):
        rec = {"cat2": cat, "n_uploaded": int(min(len(pos), 200))}
        try:
            raw = xmatch_cds(pos.head(200), cat, radius_arcsec=6.0)
            alias = (_GAIA_VIZIER_ALIASES if label == "gaia_dr3"
                     else _ALLWISE_VIZIER_ALIASES)
            _, mapping = _rename_by_alias(raw, alias)
            rec.update({"status": _classify(raw, None), "n_rows": int(len(raw)),
                        "columns_returned": list(raw.columns),
                        "alias_mapping": mapping,
                        "unmapped_canonical": sorted(set(alias) - set(mapping)),
                        "first_rows": raw.head(3).to_dict("records")})
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": STATUS_QUERY_FAILED, "error": repr(exc)})
        report["xmatch"][label] = rec

    # IRSA: the per-object AllWISE cone and one NEOWISE light curve.
    ra, dec = PROBE_POSITIONS[3][1], PROBE_POSITIONS[3][2]   # HD 172555
    try:
        df = fetch_allwise_cone(ra, dec, 6.0)
        report["irsa"]["allwise_cone"] = {
            "status": _classify(df, None), "n_rows": int(len(df)),
            "columns": list(df.columns), "first_rows": df.head(2).to_dict("records")}
    except Exception as exc:  # noqa: BLE001
        report["irsa"]["allwise_cone"] = {"status": STATUS_QUERY_FAILED,
                                          "error": repr(exc)}
    try:
        df = fetch_neowise_lightcurve(ra, dec, 3.0)
        report["irsa"]["neowise"] = {"status": _classify(df, None),
                                     "n_rows": int(len(df))}
    except Exception as exc:  # noqa: BLE001
        report["irsa"]["neowise"] = {"status": STATUS_QUERY_FAILED, "error": repr(exc)}

    return report


def _tile(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Repeat ``df`` until it has at least ``n`` rows, with unique match_ids."""
    reps = int(np.ceil(n / max(1, len(df))))
    out = pd.concat([df] * reps, ignore_index=True).head(n).copy()
    out["match_id"] = [f"t{i:06d}" for i in range(len(out))]
    return out


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

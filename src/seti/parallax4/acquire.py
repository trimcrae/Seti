"""Runner-only archive access for PARALLAX4.  Nothing here runs in the offline
suite: every function takes its transport as an argument (``http`` / ``tap``)
so tests pass stubs, and the sandbox (no egress) never reaches a live service.

Routes
------
* Gaia DR3 epoch photometry, bulk: the ESA CDN directory
  ``http://cdn.gea.esac.esa.int/Gaia/gdr3/Photometry/epoch_photometry/``
  (csv.gz, one row per source, array cells; the HEALPix-ordered partition of
  the ~11.75 M sources with published light curves).
* Gaia DR3 epoch photometry, per source: DataLink
  ``https://gea.esac.esa.int/data-server/data`` (RETRIEVAL_TYPE=EPOCH_PHOTOMETRY).
* Gaia TAP ``https://gea.esac.esa.int/tap-server/tap`` via astroquery
  (async with backoff, sync on the last attempt -- the herdsman pattern), and
  IN-list queries rather than ``tap_upload`` joins: GROWTH run 34787801172
  showed the anonymous queue timing out on upload joins against gaia_source.
* The DR4 prerelease: ``https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/
  Gaia_DR4/dr4-prerelease/`` (epoch astrometry for 12 sources and the draft
  DR4 data model, both 2026-06-26).
"""

from __future__ import annotations

import hashlib
import io
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CDN_EPOCH_PHOT = "http://cdn.gea.esac.esa.int/Gaia/gdr3/Photometry/epoch_photometry/"
CDN_GDR4 = "http://cdn.gea.esac.esa.int/Gaia/gdr4/"
DATALINK = "https://gea.esac.esa.int/data-server/data"
PRERELEASE_DIR = ("https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/"
                  "dr4-prerelease/")
PRERELEASE_ZIP = "gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip"
PRERELEASE_SHA256 = "07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b"
DATAMODEL_ZIP = "gaia-dr4-prerelease-draft-data-model_2026-06-26.zip"

GAIA_SOURCE_VET_COLUMNS = (
    "source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, phot_g_mean_mag, "
    "phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, phot_g_mean_flux, phot_g_mean_flux_over_error, "
    "phot_g_n_obs, phot_bp_rp_excess_factor, ruwe, astrometric_excess_noise, "
    "astrometric_excess_noise_sig, ipd_frac_multi_peak, ipd_gof_harmonic_amplitude, "
    "ipd_frac_odd_win, duplicated_source, phot_variable_flag, non_single_star, "
    "radial_velocity_error, visibility_periods_used")


# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------
def http_get(url: str, *, timeout: float = 120.0, retries: int = 4, stream_to: Path | None = None,
             params: dict | None = None, method: str = "GET") -> tuple[int, bytes | None]:
    """(status, body) -- body None when streamed to disk.  Retries on transport
    errors and 5xx; a 404 is returned, not raised (DR4 absence IS a 404)."""
    import requests

    last: Exception | None = None
    for attempt in range(retries):
        try:
            if method == "HEAD":
                r = requests.head(url, timeout=timeout, allow_redirects=True)
                return r.status_code, None
            with requests.get(url, timeout=timeout, params=params, stream=stream_to is not None) as r:
                if r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                if stream_to is not None and r.status_code == 200:
                    tmp = Path(str(stream_to) + ".part")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                    tmp.replace(stream_to)
                    return r.status_code, None
                return r.status_code, r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last!r}")


class DeadlineExceeded(RuntimeError):
    """A remote call did not return within its deadline."""


def with_deadline(fn, deadline_s: float, label: str = "call"):
    """Run ``fn`` in a daemon thread; raise DeadlineExceeded after
    ``deadline_s``.  astroquery's TAP calls have no overall timeout, and one
    hung socket must not eat a job's whole budget (the thread is abandoned,
    and being a daemon it does not block interpreter exit)."""
    import threading

    box: dict = {}

    def target():
        try:
            box["v"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["e"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(deadline_s)
    if th.is_alive():
        raise DeadlineExceeded(f"{label}: no answer within {deadline_s:.0f}s")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def gaia_tap(query: str, *, retries: int = 4, tag: str = "parallax4",
             deadline_s: float = 900.0) -> pd.DataFrame:
    """Gaia ADQL: async with exponential backoff, sync on the last try."""
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    last = None
    t0 = time.monotonic()
    for attempt in range(retries):
        try:
            sync = attempt == retries - 1
            job = with_deadline(lambda sync=sync: (Gaia.launch_job(query) if sync else Gaia.launch_job_async(query))
                                .get_results(), deadline_s, f"TAP {query[:60]}")
            df = job.to_pandas()
            print(f"[{tag}] TAP ok {len(df)} rows in {time.monotonic() - t0:.0f}s: {query[:80]}", flush=True)
            return df.rename(columns={c: c.lower() for c in df.columns})
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[{tag}] TAP attempt {attempt + 1}/{retries} failed: {exc!r}"[:400], flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gaia TAP failed after {retries} attempts: {last!r}")


# ---------------------------------------------------------------------------
# DR4 probe: inert until the tables exist
# ---------------------------------------------------------------------------
def probe_dr4(*, tap=gaia_tap, http=http_get) -> dict:
    """DR4_NOT_RELEASED when the archive answers and has no DR4 schema;
    ARCHIVE_UNREACHABLE when it does not answer; DR4_AVAILABLE otherwise.

    Three independent looks, each recorded: the TAP schema list, the CDN
    ``gdr4`` directory, and the DataLink service with RELEASE='Gaia DR4'."""
    rep: dict = {"checks": {}}
    schemas: list[str] = []
    try:
        df = tap("SELECT schema_name FROM tap_schema.schemas")
        schemas = sorted(str(s) for s in df.get("schema_name", []))
        dr4 = [s for s in schemas if "dr4" in s.lower()]
        rep["checks"]["tap_schemas"] = {"ok": True, "n": len(schemas), "dr4_schemas": dr4,
                                        "has_gaiadr3": any(s.lower() == "gaiadr3" for s in schemas)}
        if dr4:
            t = tap("SELECT schema_name, table_name FROM tap_schema.tables WHERE "
                    + " OR ".join(f"schema_name = '{s}'" for s in dr4))
            rep["checks"]["tap_schemas"]["dr4_tables"] = sorted(str(x) for x in t.get("table_name", []))
    except Exception as exc:  # noqa: BLE001
        rep["checks"]["tap_schemas"] = {"ok": False, "error": repr(exc)[:400]}
    try:
        st, _ = http(CDN_GDR4, timeout=60.0, retries=2, method="HEAD")
        rep["checks"]["cdn_gdr4"] = {"ok": True, "status": int(st)}
    except Exception as exc:  # noqa: BLE001
        rep["checks"]["cdn_gdr4"] = {"ok": False, "error": repr(exc)[:300]}
    tap_ok = rep["checks"]["tap_schemas"].get("ok")
    dr4_tap = bool(rep["checks"]["tap_schemas"].get("dr4_schemas"))
    cdn = rep["checks"].get("cdn_gdr4", {})
    dr4_cdn = cdn.get("ok") and cdn.get("status") == 200
    if dr4_tap or dr4_cdn:
        rep["verdict"] = "DR4_AVAILABLE"
    elif tap_ok and rep["checks"]["tap_schemas"].get("has_gaiadr3"):
        rep["verdict"] = "DR4_NOT_RELEASED"
    elif cdn.get("ok") and cdn.get("status") in (403, 404):
        rep["verdict"] = "DR4_NOT_RELEASED"
    else:
        rep["verdict"] = "ARCHIVE_UNREACHABLE"
    rep["scheduled_release"] = "2026-12-02 (ESA Gaia DR4 content page)"
    return rep


# ---------------------------------------------------------------------------
# CDN bulk epoch photometry
# ---------------------------------------------------------------------------
_HREF = re.compile(r'href="(EpochPhotometry_[0-9]+-[0-9]+\.csv\.gz)"')


def list_cdn_files(*, http=http_get, base: str = CDN_EPOCH_PHOT) -> dict:
    st, body = http(base, timeout=120.0)
    if st != 200 or body is None:
        return {"ok": False, "status": st, "files": []}
    txt = body.decode("utf-8", "replace")
    files = sorted(set(_HREF.findall(txt)))
    md5 = {}
    if "_MD5SUM.txt" in txt:
        try:
            st2, b2 = http(base + "_MD5SUM.txt", timeout=120.0)
            if st2 == 200 and b2:
                for line in b2.decode("utf-8", "replace").splitlines():
                    parts = line.split()
                    if len(parts) == 2:
                        md5[parts[1].lstrip("*")] = parts[0]
        except Exception:  # noqa: BLE001
            pass
    sizes = {}
    for m in re.finditer(r'href="(EpochPhotometry_[0-9]+-[0-9]+\.csv\.gz)".*?(\d+(?:\.\d+)?[KMG]?)\s*$',
                         txt, flags=re.M):
        sizes[m.group(1)] = m.group(2)
    return {"ok": True, "status": st, "files": files, "n_files": len(files), "md5": md5,
            "sizes": sizes}


def healpix_range(fname: str) -> tuple[int, int] | None:
    m = re.search(r"_(\d+)-(\d+)\.csv", fname)
    return (int(m.group(1)), int(m.group(2))) if m else None


def download_cdn_file(fname: str, dest_dir: Path, *, http=http_get, md5: str | None = None,
                      base: str = CDN_EPOCH_PHOT) -> Path:
    dest = Path(dest_dir) / fname
    st, _ = http(base + fname, timeout=600.0, stream_to=dest)
    if st != 200 or not dest.exists():
        raise RuntimeError(f"{fname}: HTTP {st}")
    if md5:
        h = hashlib.md5()
        with open(dest, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != md5:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"{fname}: md5 mismatch")
    return dest


# ---------------------------------------------------------------------------
# DataLink (per source)
# ---------------------------------------------------------------------------
def datalink_products(source_ids, *, retrieval_type: str = "EPOCH_PHOTOMETRY",
                      release: str = "Gaia DR3", http=http_get, batch: int = 500) -> dict[int, object]:
    """source_id -> astropy Table for one DataLink retrieval type.

    The service answers a multi-ID request with a zip of per-source VOTables
    whose member names carry the source_id."""
    from astropy.table import Table

    out: dict[int, object] = {}
    ids = [int(s) for s in source_ids]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        params = {"RETRIEVAL_TYPE": retrieval_type, "ID": ",".join(str(s) for s in chunk),
                  "FORMAT": "votable", "RELEASE": release, "DATA_STRUCTURE": "INDIVIDUAL",
                  "VALID_DATA": "false"}
        st, body = http(DATALINK, params=params, timeout=600.0)
        if st != 200 or not body:
            continue
        if body[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                for name in z.namelist():
                    m = re.search(r"(\d{6,20})", name)
                    if not m:
                        continue
                    try:
                        out[int(m.group(1))] = Table.read(io.BytesIO(z.read(name)), format="votable")
                    except Exception:  # noqa: BLE001
                        continue
        elif len(chunk) == 1:
            try:
                out[chunk[0]] = Table.read(io.BytesIO(body), format="votable")
            except Exception:  # noqa: BLE001
                pass
    return out


# ---------------------------------------------------------------------------
# TAP helpers (IN-lists; no uploads)
# ---------------------------------------------------------------------------
def _in_chunks(ids, n=1500):
    ids = [int(x) for x in ids]
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def gaia_source_rows(source_ids, *, tap=gaia_tap, columns: str = GAIA_SOURCE_VET_COLUMNS) -> pd.DataFrame:
    frames = []
    for ch in _in_chunks(source_ids):
        q = (f"SELECT {columns} FROM gaiadr3.gaia_source WHERE source_id IN "
             f"({','.join(str(s) for s in ch)})")
        frames.append(tap(q))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def vari_membership(source_ids, *, tap=gaia_tap) -> pd.DataFrame:
    """Gaia's own variability verdicts for the ids: classifier best class and
    eclipsing-binary table membership (with its frequency)."""
    ids = list(source_ids)
    rows = pd.DataFrame({"source_id": np.asarray(ids, dtype=np.int64)})
    cls, ecl = [], []
    for ch in _in_chunks(ids):
        lst = ",".join(str(s) for s in ch)
        try:
            cls.append(tap("SELECT source_id, best_class_name, best_class_score FROM "
                           f"gaiadr3.vari_classifier_result WHERE source_id IN ({lst})"))
        except Exception as exc:  # noqa: BLE001
            print(f"[parallax4] vari_classifier_result chunk failed: {exc!r}"[:300])
        try:
            ecl.append(tap("SELECT source_id, frequency, global_ranking FROM "
                           f"gaiadr3.vari_eclipsing_binary WHERE source_id IN ({lst})"))
        except Exception as exc:  # noqa: BLE001
            print(f"[parallax4] vari_eclipsing_binary chunk failed: {exc!r}"[:300])
    if cls:
        c = pd.concat(cls, ignore_index=True)
        rows = rows.merge(c, on="source_id", how="left")
    if ecl:
        e = pd.concat(ecl, ignore_index=True).rename(columns={"frequency": "ecl_frequency"})
        e["in_vari_eclipsing_binary"] = True
        rows = rows.merge(e, on="source_id", how="left")
    rows.attrs["n_chunks_ok"] = {"classifier": len(cls), "eclipsing": len(ecl)}
    return rows


def gaia_cone(ra: float, dec: float, radius_arcsec: float, *, tap=gaia_tap) -> pd.DataFrame:
    r = float(radius_arcsec) / 3600.0
    return tap("SELECT source_id, ra, dec, phot_g_mean_mag, phot_g_mean_flux, phot_bp_mean_flux, "
               "phot_rp_mean_flux, bp_rp, parallax, "
               f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra:.8f}, {dec:.8f})) * 3600.0 AS sep_arcsec "
               "FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
               f"CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {r:.8f}))")


def varstrometry_samples(*, tap=gaia_tap, n_random_index: int = 20_000_000,
                         g_lo: float = 13.0, g_hi: float = 17.5) -> dict[str, pd.DataFrame]:
    """The two DR3 populations for the varstrometry control: eclipsing
    binaries (strong variables) and photometrically quiet stars, each with the
    fields that carry blending (ipd_frac_multi_peak, ipd_gof_harmonic_amplitude)
    and astrometric jitter (astrometric_excess_noise, ruwe)."""
    cols = ("g.source_id, g.phot_g_mean_mag, g.bp_rp, g.phot_g_mean_flux_over_error, g.phot_g_n_obs, "
            "g.astrometric_excess_noise, g.ruwe, g.ipd_frac_multi_peak, g.ipd_gof_harmonic_amplitude, "
            "g.visibility_periods_used, g.parallax, g.phot_bp_rp_excess_factor")
    ecl = tap(f"SELECT {cols} FROM gaiadr3.vari_eclipsing_binary AS v JOIN gaiadr3.gaia_source AS g "
              f"ON g.source_id = v.source_id WHERE g.phot_g_mean_mag BETWEEN {g_lo} AND {g_hi} "
              f"AND g.random_index < {int(n_random_index) * 5} AND g.astrometric_params_solved > 3")
    quiet = tap(f"SELECT {cols} FROM gaiadr3.gaia_source AS g WHERE g.phot_g_mean_mag BETWEEN "
                f"{g_lo} AND {g_hi} AND g.random_index < {int(n_random_index) // 40} "
                "AND g.phot_variable_flag = 'NOT_AVAILABLE' AND g.astrometric_params_solved > 3")
    return {"eclipsing": ecl, "quiet": quiet}


# ---------------------------------------------------------------------------
# DR4 release-day controls, fixed today
# ---------------------------------------------------------------------------
def vim_sources(*, tap=gaia_tap) -> pd.DataFrame:
    """Gaia DR3's own Variability-Induced Movers (``gaiadr3.nss_vim_fl``):
    sources whose photocentre DR3 found to move with their brightness -- the
    Hipparcos VIM idea (Wielen 1996) in Gaia's non-single-star pipeline
    (Gaia Collaboration, Arenou et al. 2023).  They are the natural POSITIVE
    control for the DR4 per-transit test."""
    return tap("SELECT * FROM gaiadr3.nss_vim_fl")


def single_eb_controls(*, tap=gaia_tap, n: int = 300) -> pd.DataFrame:
    """Bright, isolated, astrometrically clean Gaia eclipsing binaries: the
    NEGATIVE control (their variation is on the target; no photocentre shift)."""
    df = tap(f"SELECT TOP {int(n) * 2} g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.ruwe, "
             "g.ipd_frac_multi_peak, g.ipd_gof_harmonic_amplitude, g.duplicated_source, "
             "v.global_ranking, v.frequency "
             "FROM gaiadr3.vari_eclipsing_binary AS v JOIN gaiadr3.gaia_source AS g "
             "ON g.source_id = v.source_id WHERE g.phot_g_mean_mag < 13.5 AND g.ruwe < 1.1 "
             "AND g.ipd_frac_multi_peak = 0 AND g.ipd_gof_harmonic_amplitude < 0.02 "
             "ORDER BY v.global_ranking DESC")
    if "duplicated_source" in df:
        dup = df["duplicated_source"].map(lambda x: str(x).strip().lower() in ("true", "1"))
        df = df[~dup]
    return df.head(int(n)).reset_index(drop=True)


def vim_membership(source_ids, *, tap=gaia_tap) -> set[int]:
    out: set[int] = set()
    for ch in _in_chunks(source_ids):
        df = tap("SELECT source_id FROM gaiadr3.nss_vim_fl WHERE source_id IN "
                 f"({','.join(str(s) for s in ch)})")
        out |= {int(x) for x in df.get("source_id", [])}
    return out


# ---------------------------------------------------------------------------
# DR4 prerelease + draft data model
# ---------------------------------------------------------------------------
def fetch_prerelease(dest_dir: Path, *, http=http_get) -> dict:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    try:
        st, body = http(PRERELEASE_DIR, timeout=60.0, retries=2)
        if st == 200 and body:
            rep["directory"] = sorted(set(re.findall(r'href="([^"?/][^"]*\.(?:zip|txt|pdf|html|csv))"',
                                                     body.decode("utf-8", "replace"))))
    except Exception as exc:  # noqa: BLE001
        rep["directory_error"] = repr(exc)[:300]
    for name, key in ((PRERELEASE_ZIP, "epoch_astrometry"), (DATAMODEL_ZIP, "datamodel")):
        p = dest_dir / name
        try:
            st, _ = http(PRERELEASE_DIR + name, timeout=300.0, stream_to=p)
            ok = st == 200 and p.exists()
            sha = hashlib.sha256(p.read_bytes()).hexdigest() if ok else None
            rep[key] = {"ok": ok, "status": st, "path": str(p) if ok else None, "sha256": sha,
                        "bytes": p.stat().st_size if ok else 0}
        except Exception as exc:  # noqa: BLE001
            rep[key] = {"ok": False, "error": repr(exc)[:300]}
    ea = rep.get("epoch_astrometry", {})
    if ea.get("ok"):
        ea["sha256_matches_fixture"] = ea.get("sha256") == PRERELEASE_SHA256
    return rep


_COLNAME = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")


def parse_datamodel(zip_path: Path) -> dict:
    """What the draft DR4 data model says, without assuming its file format.

    Records the member list, and for each member whose name or text mentions
    an epoch table, the snake_case identifiers that appear in it -- the raw
    material for checking every reader role against the announced names."""
    out: dict = {"members": [], "tables": {}}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            out["members"].append({"name": info.filename, "bytes": info.file_size})
            low = info.filename.lower()
            if not low.endswith((".html", ".htm", ".txt", ".csv", ".xml", ".json", ".tex", ".md")):
                continue
            try:
                txt = z.read(info.filename).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                continue
            txt_plain = re.sub(r"<[^>]+>", " ", txt)
            for key in ("epoch_photometry", "epoch_astrometry", "epoch_radial_velocity",
                        "epoch_rv", "sso_observation", "vari_", "gaia_source"):
                if key in low or key in txt_plain[:20000]:
                    ids = sorted(set(_COLNAME.findall(txt_plain)))
                    out["tables"].setdefault(info.filename, {"mentions": [], "identifiers": ids[:4000]})
                    out["tables"][info.filename]["mentions"].append(key)
    return out

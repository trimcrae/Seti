"""MIDDEN acquisition (runner only; every stage checkpoints to disk).

v1 corpus = ESO Phase-3 HARPS + FEROS 1D spectra for three target classes:

1. **Anchors** — the contested single-star prior claims (Przybylski's Star
   HD 101065, HD 965, HR 465, HD 25354).  Sanity anchors: the pipeline must
   naturally re-observe them.  Coordinates are resolved by name via Sesame on
   the runner, with encoded fallbacks (fallbacks get a wider match radius).
2. **Renson & Manfroid Ap/Bp/CP stars** — VizieR III/260 via TAPVizieR with
   dynamic column resolution (same pattern as herdsman_b.acquire): a TOP-1
   probe maps whatever the current column names are onto ra/dec/name.
3. **Bright A5-F2 main-sequence stars** — the Whitmire & Wright predicted
   repository class, selected from Gaia DR3 (teff_gspphot 6800-8300 K,
   logg_gspphot > 3.8, G < 9, parallax > 2 mas).

Spectra are found by uploading the target list to ESO's TAP service
(https://archive.eso.org/tap_obs, table ivoa.ObsCore) in checkpointed chunks
and position-matching, then downloaded and analyzed in a process-and-discard
loop: ~50 FITS to scratch, measure, append to a checkpointed parquet, DELETE
the FITS, next batch.  The runner disk (~14 GB) never accumulates spectra.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

_ESO_TAP = "https://archive.eso.org/tap_obs"
_VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
_SESAME = "https://cds.unistra.fr/cgi-bin/nph-sesame/-op/SN"

# Renson & Manfroid 2009 "General catalogue of Ap and Am stars" — try in order.
_RENSON_TABLES = ['"III/260/catalog"', '"III/260/stars"', '"III/260/table1"']
_RA_NAMES = ("raj2000", "_ra", "ra_icrs", "ra")
_DEC_NAMES = ("dej2000", "_de", "de_icrs", "dec", "de")
_NAME_NAMES = ("hd", "name", "renson", "recno")
_SPT_NAMES = ("sptype", "sptypes", "spt", "sp")

_MATCH_RADIUS_DEG = 2.0 / 3600.0        # resolved positions
_COARSE_RADIUS_DEG = 30.0 / 3600.0      # fallback anchor positions

# (name, fallback ra/dec deg J2000).  Sesame resolution runner-side overrides;
# fallbacks are approximate and therefore matched at the coarse radius.
ANCHORS = (
    ("HD 101065", 174.4043, -46.7097),   # Przybylski's Star
    ("HD 965", 3.5427, -0.7568),
    ("HD 9996", 24.6329, 45.4000),       # = HR 465
    ("HD 25354", 60.9, 39.3),
)

_GAIA_AF_QUERY = """
SELECT TOP {top} source_id, ra, dec, phot_g_mean_mag, parallax,
       teff_gspphot, logg_gspphot
FROM gaiadr3.gaia_source
WHERE teff_gspphot BETWEEN {teff_lo} AND {teff_hi}
  AND logg_gspphot > {logg_min}
  AND phot_g_mean_mag < {g_max}
  AND parallax > {plx_min}
ORDER BY phot_g_mean_mag
"""

# ESO's tap_obs rejects TAP_UPLOAD entirely (verified live: "Unknown table
# TAP_UPLOAD.targets"), so discovery is a bulk metadata pull in declination
# bands with the target crossmatch done locally.
_OBSCORE_BAND_QUERY = """
SELECT dp_id, access_url, instrument_name, obs_collection,
       s_ra, s_dec, snr, t_min, em_min, em_max, target_name
FROM ivoa.ObsCore
WHERE dataproduct_type = 'spectrum'
  AND (instrument_name = 'HARPS' OR instrument_name = 'FEROS')
  AND s_dec >= {dec_lo} AND s_dec < {dec_hi}
"""


def _retry(fn, retries=3, label="fetch"):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[midden] {label} attempt {attempt + 1}/{retries} "
                  f"failed: {exc!r}")
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"{label} failed after {retries} attempts: {last!r}")


# ---------------------------------------------------------------------------
# Target list
# ---------------------------------------------------------------------------

def resolve_anchor(name: str, fallback_ra: float, fallback_dec: float,
                   timeout: float = 30.0) -> dict:
    """Sesame name resolution with encoded fallback (coarse-matched)."""
    import re

    import requests

    try:
        r = requests.get(f"{_SESAME}?{name.replace(' ', '+')}", timeout=timeout)
        r.raise_for_status()
        m = re.search(r"%J\s+(\d+\.\d+)\s+([+-]?\d+\.\d+)", r.text)
        if m:
            return {"name": name, "ra": float(m.group(1)),
                    "dec": float(m.group(2)), "rad": _MATCH_RADIUS_DEG,
                    "resolved": True}
    except Exception as exc:  # noqa: BLE001
        print(f"[midden] Sesame failed for {name}: {exc!r}")
    return {"name": name, "ra": fallback_ra, "dec": fallback_dec,
            "rad": _COARSE_RADIUS_DEG, "resolved": False}


def _resolve_renson_columns(colnames) -> dict:
    out = {}
    low = {c.lower(): c for c in colnames}
    for key, cands in (("ra", _RA_NAMES), ("dec", _DEC_NAMES),
                       ("name", _NAME_NAMES), ("spt", _SPT_NAMES)):
        for c in cands:
            hit = next((low[x] for x in low if x == c), None) or \
                next((low[x] for x in low if x.startswith(c)), None)
            if hit:
                out[key] = hit
                break
    return out


def fetch_renson(out_path: Path) -> pd.DataFrame:
    """Renson & Manfroid CP-star catalog via TAPVizieR (checkpointed)."""
    if out_path.exists():
        print(f"[midden] renson checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    import pyvo

    tap = pyvo.dal.TAPService(_VIZIER_TAP)
    last_err = None
    for table in _RENSON_TABLES:
        try:
            probe = tap.run_sync(f"SELECT TOP 1 * FROM {table}").to_table()
            cols = _resolve_renson_columns(probe.colnames)
            if "ra" not in cols or "dec" not in cols:
                raise RuntimeError(f"{table}: unresolved columns {probe.colnames}")
            want = [cols["ra"], cols["dec"]] + \
                [cols[k] for k in ("name", "spt") if k in cols]
            q = "SELECT " + ", ".join(f'"{c}"' for c in want) + f" FROM {table}"
            print(f"[midden] pulling Renson CP stars from {table} (cols {cols}) ...")
            res = tap.run_async(q).to_table().to_pandas()
            res.columns = (["ra", "dec"]
                           + [k for k in ("name", "spt") if k in cols])
            res["ra"] = pd.to_numeric(res["ra"], errors="coerce")
            res["dec"] = pd.to_numeric(res["dec"], errors="coerce")
            res = res.dropna(subset=["ra", "dec"]).reset_index(drop=True)
            if "name" not in res:
                res["name"] = [f"renson_{i}" for i in range(len(res))]
            res["name"] = res["name"].astype(str)
            res.to_parquet(out_path, index=False)
            print(f"[midden] {len(res)} Renson CP stars -> {out_path}")
            return res
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[midden] {table} failed: {exc!r}")
    raise RuntimeError(f"Renson fetch failed everywhere: {last_err!r}")


def fetch_gaia_af(out_path: Path, top: int = 15000, teff_lo: float = 6800.0,
                  teff_hi: float = 8300.0, logg_min: float = 3.8,
                  g_max: float = 9.0, plx_min: float = 2.0) -> pd.DataFrame:
    """Bright A5-F2 dwarfs (the predicted repository class) from Gaia DR3."""
    if out_path.exists():
        print(f"[midden] gaia-af checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    from astroquery.gaia import Gaia

    q = _GAIA_AF_QUERY.format(top=top, teff_lo=teff_lo, teff_hi=teff_hi,
                              logg_min=logg_min, g_max=g_max, plx_min=plx_min)

    def _go():
        job = Gaia.launch_job_async(q)
        df = job.get_results().to_pandas()
        return df.rename(columns={c: c.lower() for c in df.columns})

    df = _retry(_go, retries=4, label="gaia A5-F2 pull")
    df.to_parquet(out_path, index=False)
    print(f"[midden] {len(df)} Gaia A5-F2 dwarfs -> {out_path}")
    return df


def attach_gaia_teff(targets: pd.DataFrame, out_dir: Path,
                     chunk: int = 2000) -> pd.DataFrame:
    """Best-effort Teff for non-Gaia targets by positional upload join.

    Failure is tolerated (Teff stays NaN; such stars score against the whole
    corpus instead of a Teff-matched slice).  Checkpointed per chunk.
    """
    need = targets[~np.isfinite(pd.to_numeric(targets["teff"],
                                              errors="coerce"))].copy()
    if not len(need):
        return targets
    from astropy.table import Table
    from astroquery.gaia import Gaia

    out_dir.mkdir(parents=True, exist_ok=True)
    q = """
    SELECT t.tid AS tid, g.teff_gspphot AS teff_gspphot,
           g.phot_g_mean_mag AS gmag
    FROM tap_upload.targets AS t
    JOIN gaiadr3.gaia_source AS g
      ON CONTAINS(POINT('ICRS', g.ra, g.dec),
                  CIRCLE('ICRS', t.ra, t.dec, 0.000833)) = 1
    WHERE g.phot_g_mean_mag < 12
    """
    frames = []
    idx = need.index.to_numpy()
    n_chunks = int(np.ceil(len(idx) / chunk))
    for i in range(n_chunks):
        part = out_dir / f"teff_chunk_{i:03d}.parquet"
        if part.exists():
            frames.append(pd.read_parquet(part))
            continue
        sub = need.loc[idx[i * chunk:(i + 1) * chunk]]
        tbl = Table({"tid": sub["tid"].to_numpy(np.int64),
                     "ra": sub["ra"].to_numpy(float),
                     "dec": sub["dec"].to_numpy(float)})
        try:
            job = Gaia.launch_job_async(q, upload_resource=tbl,
                                        upload_table_name="targets")
            df = job.get_results().to_pandas()
            df = df.rename(columns={c: c.lower() for c in df.columns})
            df = df.sort_values("gmag").drop_duplicates("tid")
            df.to_parquet(part, index=False)
            frames.append(df)
            print(f"[midden] teff chunk {i + 1}/{n_chunks}: {len(df)} matches")
        except Exception as exc:  # noqa: BLE001
            print(f"[midden] teff chunk {i} failed (tolerated): {exc!r}")
    if frames:
        teff = pd.concat(frames, ignore_index=True)
        m = targets.merge(teff[["tid", "teff_gspphot"]], on="tid", how="left")
        targets = targets.copy()
        fill = pd.to_numeric(m["teff_gspphot"], errors="coerce").to_numpy()
        cur = pd.to_numeric(targets["teff"], errors="coerce").to_numpy()
        targets["teff"] = np.where(np.isfinite(cur), cur, fill)
    return targets


def build_targets(out_dir: Path, max_gaia: int = 15000) -> pd.DataFrame:
    """Anchors + Renson CP stars + Gaia A5-F2 dwarfs -> targets.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "targets.parquet"
    if out_path.exists():
        print(f"[midden] targets checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)

    rows = []
    for name, ra, dec in ANCHORS:
        a = resolve_anchor(name, ra, dec)
        rows.append({"name": a["name"], "ra": a["ra"], "dec": a["dec"],
                     "rad": a["rad"], "teff": np.nan, "priority": 0,
                     "source": "anchor" if a["resolved"] else "anchor_coarse"})

    renson = fetch_renson(out_dir / "renson.parquet")
    for r in renson.itertuples():
        rows.append({"name": f"renson_{r.name}", "ra": float(r.ra),
                     "dec": float(r.dec), "rad": _MATCH_RADIUS_DEG,
                     "teff": np.nan, "priority": 1, "source": "renson"})

    gaia = fetch_gaia_af(out_dir / "gaia_af.parquet", top=max_gaia)
    for r in gaia.itertuples():
        rows.append({"name": f"gaia_{int(r.source_id)}", "ra": float(r.ra),
                     "dec": float(r.dec), "rad": _MATCH_RADIUS_DEG,
                     "teff": float(r.teff_gspphot), "priority": 2,
                     "source": "gaia_af"})

    targets = pd.DataFrame(rows)
    targets["tid"] = np.arange(len(targets), dtype=np.int64)
    targets = attach_gaia_teff(targets, out_dir / "teff_chunks")
    targets.to_parquet(out_path, index=False)
    print(f"[midden] {len(targets)} targets "
          f"({(targets['priority'] == 0).sum()} anchors, "
          f"{(targets['priority'] == 1).sum()} Renson, "
          f"{(targets['priority'] == 2).sum()} Gaia A5-F2) -> {out_path}")
    return targets


# ---------------------------------------------------------------------------
# ESO ObsCore discovery
# ---------------------------------------------------------------------------

def query_obscore(targets: pd.DataFrame, out_dir: Path,
                  dec_band_deg: float = 10.0) -> pd.DataFrame:
    """Discover HARPS/FEROS spectra for the targets (checkpointed).

    Bulk-pulls ObsCore spectrum metadata in declination bands (ESO's TAP has
    no upload support), then position-matches against the target list locally
    with a unit-vector KD-tree at each target's own match radius.
    """
    out_path = out_dir / "obscore.parquet"
    if out_path.exists():
        print(f"[midden] obscore checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    import pyvo
    from scipy.spatial import cKDTree

    chunk_dir = out_dir / "obscore_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    tap = pyvo.dal.TAPService(_ESO_TAP)

    def _unit(ra_deg, dec_deg):
        ra = np.radians(np.asarray(ra_deg, float))
        dec = np.radians(np.asarray(dec_deg, float))
        return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                         np.sin(dec)], axis=1)

    t_xyz = _unit(targets["ra"], targets["dec"])
    t_tree = cKDTree(t_xyz)
    rad_max = float(np.nanmax(targets["rad"].to_numpy(float)))
    chord_max = 2.0 * np.sin(np.radians(rad_max) / 2.0)

    frames = []
    edges = np.arange(-90.0, 90.0 + dec_band_deg, dec_band_deg)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        part = chunk_dir / f"obscore_band_{i:03d}.parquet"
        if part.exists():
            band = pd.read_parquet(part)
        else:
            q = _OBSCORE_BAND_QUERY.format(dec_lo=lo, dec_hi=hi)

            def _go(q=q):
                res = tap.run_async(q)
                df = res.to_table().to_pandas()
                return df.rename(columns={c: c.lower() for c in df.columns})

            band = _retry(_go, retries=3, label=f"obscore band {i}")
            band.to_parquet(part, index=False)          # checkpoint
            print(f"[midden] obscore band {i + 1}/{len(edges) - 1} "
                  f"[{lo:+.0f},{hi:+.0f}): {len(band)} spectra")
        if not len(band):
            continue
        # Local crossmatch: nearest target within the generous radius, then
        # exact per-target radius check.
        b_xyz = _unit(band["s_ra"], band["s_dec"])
        dist, idx = t_tree.query(b_xyz, k=1,
                                 distance_upper_bound=chord_max)
        ok = np.isfinite(dist)
        if not ok.any():
            continue
        sub = band[ok].reset_index(drop=True)
        tid_idx = idx[ok]
        sep_deg = np.degrees(2.0 * np.arcsin(np.clip(dist[ok] / 2.0, 0, 1)))
        per_rad = targets["rad"].to_numpy(float)[tid_idx]
        keep = sep_deg <= per_rad
        if not keep.any():
            continue
        sub = sub[keep].reset_index(drop=True)
        sub["tid"] = targets["tid"].to_numpy(np.int64)[tid_idx[keep]]
        frames.append(sub)
    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    obs.to_parquet(out_path, index=False)
    print(f"[midden] ObsCore discovery: {len(obs)} matched spectrum rows "
          f"-> {out_path}")
    return obs


def select_corpus(obs: pd.DataFrame, targets: pd.DataFrame,
                  max_spectra: int = 3000,
                  epochs_per_star: int = 3) -> pd.DataFrame:
    """De-duplicate to <= epochs_per_star epochs/star, cap the corpus.

    Anchors first, then Renson, then Gaia A5-F2; within a class every star
    gets its first (highest-SNR) epoch before any star gets a second, so the
    cap trades depth for breadth, never the reverse.
    """
    if not len(obs):
        return obs
    meta = targets[["tid", "name", "teff", "priority", "source"]]
    df = obs.merge(meta, on="tid", how="left")
    df = df.sort_values("dp_id").drop_duplicates("dp_id")
    df["snr"] = pd.to_numeric(df.get("snr"), errors="coerce").fillna(0.0)
    df = df.sort_values(["tid", "snr"], ascending=[True, False])
    df = df.drop_duplicates(["tid", "t_min"])           # one file per epoch
    df["epoch_rank"] = df.groupby("tid").cumcount()
    df = df[df["epoch_rank"] < epochs_per_star]
    df = df.sort_values(["priority", "epoch_rank", "snr"],
                        ascending=[True, True, False])
    df = df.head(max_spectra).reset_index(drop=True)
    df = df.rename(columns={"name": "star"})
    print(f"[midden] corpus: {len(df)} spectra, {df['tid'].nunique()} stars "
          f"(anchors present: {sorted(df.loc[df['priority'] == 0, 'star'].unique())})")
    return df


# ---------------------------------------------------------------------------
# Process-and-discard analysis loop
# ---------------------------------------------------------------------------

def _default_fetch(row, dest: Path, timeout: float = 300.0) -> None:
    """Download one Phase-3 FITS via its ObsCore access_url."""
    import requests

    def _go():
        with requests.get(row["access_url"], stream=True, timeout=timeout,
                          headers={"User-Agent": "seti-midden/0.1"}) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)

    _retry(_go, retries=3, label=f"download {row['dp_id']}")


def process_corpus(corpus: pd.DataFrame, out_dir: Path, scratch_dir: Path,
                   batch_size: int = 50, fetch_fn=None,
                   line_set=None) -> pd.DataFrame:
    """Batchwise download -> measure -> checkpoint -> DELETE loop.

    Each batch lands in its own ``meas_batch_NNNN.parquet``; a re-run (or an
    artifact-seeded resume) skips completed batches, so a killed job loses at
    most one batch of work.  FITS files are removed in a ``finally`` so the
    scratch dir never accumulates even on analysis errors.
    """
    from .measure import analyze_spectrum

    fetch_fn = fetch_fn or _default_fetch
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    n_batches = int(np.ceil(len(corpus) / batch_size))
    frames = []
    for b in range(n_batches):
        part = out_dir / f"meas_batch_{b:04d}.parquet"
        if part.exists():
            frames.append(pd.read_parquet(part))
            continue
        batch = corpus.iloc[b * batch_size:(b + 1) * batch_size]
        rows, files = [], []
        try:
            for _, r in batch.iterrows():
                dest = scratch_dir / f"{str(r['dp_id']).replace('/', '_')}.fits"
                files.append(dest)
                meta = {"star": r["star"], "tid": int(r["tid"]),
                        "dp_id": r["dp_id"],
                        "teff": float(pd.to_numeric(r.get("teff"),
                                                    errors="coerce")),
                        "priority": int(r["priority"]), "source": r["source"],
                        "instrument": r.get("instrument_name", ""),
                        "t_min": float(pd.to_numeric(r.get("t_min"),
                                                     errors="coerce")),
                        "snr_obscore": float(r.get("snr", np.nan))}
                try:
                    fetch_fn(r, dest)
                    rows.extend(analyze_spectrum(dest, meta, line_set=line_set))
                except Exception as exc:  # noqa: BLE001 — one bad file != a dead run
                    print(f"[midden] spectrum {r['dp_id']} failed: {exc!r}")
                    rows.append({**meta, "species": "ERROR", "wavelength": np.nan,
                                 "role": "error", "error": repr(exc)})
        finally:
            for f in files:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
        df = pd.DataFrame(rows)
        df.to_parquet(part, index=False)          # checkpoint
        frames.append(df)
        print(f"[midden] batch {b + 1}/{n_batches}: {len(batch)} spectra, "
              f"{len(df)} measurement rows")
    if not frames:
        return pd.DataFrame()
    meas = pd.concat(frames, ignore_index=True)
    return meas


__all__ = ["ANCHORS", "attach_gaia_teff", "build_targets", "fetch_gaia_af",
           "fetch_renson", "process_corpus", "query_obscore", "resolve_anchor",
           "select_corpus"]

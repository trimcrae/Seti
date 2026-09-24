"""ZTF DR light curves straight from IRSA's HATS parquet on S3 (runner only).

Why: the IRSA light-curve API is the survey's bottleneck.  Run 36006324981
measured ~56 s a star per worker on the positional route; run 36018334549 got
object ids from a 400-star ``TAP_UPLOAD`` join in 12-19 s but then spent
1,754-2,234 s on the multi-ID light-curve requests and lost 417/577 and
553/553 ids to timeouts.  The ids are cheap; the light-curve service is not.

IRSA publishes the same light curves as a HATS catalogue on AWS S3
(``s3://ipac-irsa-ztf/ztf/enhanced/dr24/lc/hats``, anonymous, one row per ZTF
object with a nested ``lightcurve`` column).  A HATS partition file is sorted
by ``_healpix_29``; its parquet row groups carry min/max statistics on that
column, so a star's rows can be read by fetching only the row group(s) whose
``_healpix_29`` range covers the star's cone --- not the partition.

Nothing about the layout is assumed: :func:`probe` resolves the collection's
primary table from its own properties file, reads ``partition_info.csv``,
opens one partition's footer and reports the column names, the row-group count
and sizes, and the time to read one star's row group.  The survey only uses
this route after a probe says it works.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

BUCKET = "ipac-irsa-ztf"
COLLECTION = "ztf/enhanced/dr24/lc/hats"
REGION = "us-west-2"


def _fs():
    import pyarrow.fs as pafs

    return pafs.S3FileSystem(anonymous=True, region=REGION)


def _read_text(fs, path: str) -> str:
    with fs.open_input_stream(path) as f:
        return f.read().decode("utf-8", "replace")


def resolve_catalog(fs=None, collection: str = COLLECTION) -> dict:
    """The primary catalogue directory of the HATS collection."""
    fs = fs or _fs()
    root = f"{BUCKET}/{collection}"
    out = {"root": root}
    for name in ("collection.properties", "properties", "hats.properties"):
        try:
            txt = _read_text(fs, f"{root}/{name}")
        except Exception as exc:                        # noqa: BLE001
            out[f"{name}_error"] = repr(exc)[:200]
            continue
        props = {}
        for ln in txt.splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                props[k.strip()] = v.strip()
        out["properties_file"] = name
        out["properties"] = props
        prim = props.get("hats_primary_table_url") or props.get("obs_collection")
        if name == "collection.properties" and prim:
            out["catalog"] = f"{root}/{prim.strip('/')}"
        else:
            out["catalog"] = root
        break
    if "catalog" not in out:
        import pyarrow.fs as pafs

        try:
            infos = fs.get_file_info(pafs.FileSelector(root))
            out["listing"] = [i.path for i in infos][:50]
            dirs = [i.path for i in infos if i.type == pafs.FileType.Directory]
            if len(dirs) == 1:
                out["catalog"] = dirs[0]
        except Exception as exc:                        # noqa: BLE001
            out["listing_error"] = repr(exc)[:200]
    return out


def partition_info(fs, catalog: str) -> pd.DataFrame:
    import io

    for name in ("partition_info.csv", "dataset/partition_info.csv"):
        try:
            txt = _read_text(fs, f"{catalog}/{name}")
            df = pd.read_csv(io.StringIO(txt))
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:                               # noqa: BLE001
            continue
    raise RuntimeError("partition_info.csv not found")


def healpix_nested(ra, dec, order: int) -> np.ndarray:
    import astropy.units as u
    from astropy_healpix import HEALPix

    hp = HEALPix(nside=2 ** int(order), order="nested")
    return np.asarray(hp.lonlat_to_healpix(np.asarray(ra, float) * u.deg,
                                           np.asarray(dec, float) * u.deg), dtype=np.int64)


def locate_partitions(pinfo: pd.DataFrame, ra, dec) -> np.ndarray:
    """(Norder, Npix) row index of pinfo holding each position (-1 if none)."""
    ra, dec = np.atleast_1d(ra), np.atleast_1d(dec)
    ncol = "Norder" if "Norder" in pinfo else pinfo.columns[0]
    pcol = "Npix" if "Npix" in pinfo else pinfo.columns[-1]
    key = {(int(o), int(p)): k for k, (o, p) in enumerate(zip(pinfo[ncol], pinfo[pcol],
                                                              strict=False))}
    out = np.full(ra.size, -1, dtype=np.int64)
    for o in sorted(set(int(x) for x in pinfo[ncol])):
        pix = healpix_nested(ra, dec, o)
        for i, p in enumerate(pix):
            if out[i] < 0 and (o, int(p)) in key:
                out[i] = key[(o, int(p))]
    return out


def partition_path(catalog: str, norder: int, npix: int) -> str:
    d = (int(npix) // 10000) * 10000
    return f"{catalog}/dataset/Norder={int(norder)}/Dir={d}/Npix={int(npix)}.parquet"


def h29_range(ra: float, dec: float, radius_arcsec: float) -> tuple[int, int]:
    """A conservative _healpix_29 interval covering a small cone: the order-k
    pixel containing the centre, at the finest k whose pixel is >= 4x the cone,
    expanded to order 29 (plus its 8 neighbours, as a union of intervals)."""
    import astropy.units as u
    from astropy_healpix import HEALPix

    # order-k pixel size ~ 58.6 deg / 2^k; want >= 4 * radius
    k = int(np.clip(np.floor(np.log2(58.6 * 3600.0 / (4.0 * max(radius_arcsec, 0.5)))), 0, 29))
    hp = HEALPix(nside=2 ** k, order="nested")
    c = int(hp.lonlat_to_healpix(ra * u.deg, dec * u.deg))
    nb = [int(x) for x in np.atleast_1d(hp.neighbours(c)) if int(x) >= 0] + [c]
    shift = 2 * (29 - k)
    lo = min(nb) << shift
    hi = ((max(nb) + 1) << shift) - 1
    return lo, hi


def probe(test_positions=((106.329042, 6.205389),), radius_arcsec: float = 2.0) -> dict:
    """Measure the route on real positions; never assumed."""
    out: dict = {"started": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())}
    t0 = _time.monotonic()
    try:
        fs = _fs()
        cat = resolve_catalog(fs)
        out["resolve"] = cat
        pinfo = partition_info(fs, cat["catalog"])
        out["n_partitions"] = int(len(pinfo))
        out["partition_columns"] = list(pinfo.columns)
        import pyarrow.parquet as pq

        tests = []
        for ra, dec in test_positions:
            rec: dict = {"ra": ra, "dec": dec}
            k = int(locate_partitions(pinfo, [ra], [dec])[0])
            if k < 0:
                rec["status"] = "NO_PARTITION"
                tests.append(rec)
                continue
            row = pinfo.iloc[k]
            path = partition_path(cat["catalog"], row.iloc[0], row.iloc[-1])
            rec["path"] = path
            ta = _time.monotonic()
            pf = pq.ParquetFile(fs.open_input_file(path))
            md = pf.metadata
            rec["footer_s"] = round(_time.monotonic() - ta, 2)
            rec["n_row_groups"] = md.num_row_groups
            rec["n_rows"] = md.num_rows
            rec["columns"] = [md.schema.column(i).path for i in range(min(md.num_columns, 60))]
            sizes = [md.row_group(i).total_byte_size for i in range(md.num_row_groups)]
            rec["row_group_mb_median"] = round(float(np.median(sizes)) / 1e6, 2) if sizes else None
            # which row groups cover the cone
            names = [md.schema.column(i).path for i in range(md.num_columns)]
            hcol = names.index("_healpix_29") if "_healpix_29" in names else None
            lo, hi = h29_range(ra, dec, radius_arcsec)
            want = []
            if hcol is not None:
                for i in range(md.num_row_groups):
                    st = md.row_group(i).column(hcol).statistics
                    if st is None or not st.has_min_max or not (st.max < lo or st.min > hi):
                        want.append(i)
            else:
                want = list(range(md.num_row_groups))
            rec["row_groups_needed"] = len(want)
            ta = _time.monotonic()
            cols = [c for c in ("objectid", "filterid", "filtercode", "ra", "dec", "nepochs",
                                "_healpix_29", "lightcurve") if any(n == c or n.startswith(c + ".")
                                                                   for n in names)]
            tab = pf.read_row_groups(want[:4], columns=cols) if want else None
            rec["read_s"] = round(_time.monotonic() - ta, 2)
            if tab is not None:
                df = tab.to_pandas()
                rec["n_rows_read"] = int(len(df))
                if "ra" in df and "dec" in df:
                    sep = 3600.0 * np.hypot((df["ra"] - ra) * np.cos(np.radians(dec)),
                                            df["dec"] - dec)
                    near = df[sep < radius_arcsec]
                    rec["n_objects_in_cone"] = int(len(near))
                    if len(near) and "lightcurve" in near:
                        lc = near["lightcurve"].iloc[0]
                        rec["lightcurve_type"] = str(type(lc))[:80]
                        try:
                            rec["lightcurve_len"] = int(len(lc))
                            rec["lightcurve_sample"] = str(lc)[:400]
                        except Exception:               # noqa: BLE001
                            pass
            rec["status"] = "OK"
            tests.append(rec)
        out["tests"] = tests
        out["status"] = "OK"
    except Exception as exc:                            # noqa: BLE001
        out["status"] = "FAILED"
        out["error"] = repr(exc)[:500]
    out["elapsed_s"] = round(_time.monotonic() - t0, 1)
    return out


__all__ = ["h29_range", "locate_partitions", "partition_info", "partition_path", "probe",
           "resolve_catalog"]

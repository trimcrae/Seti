#!/usr/bin/env python3
"""Measure -- not just cite -- the systematics that govern an AKARI-FIS / IRAS
x optical-NIR crossmatch.

Downloads the actual catalogues and computes:
  1. Exact column inventory of AKARI FIS BSC (II/298) and IRAS PSC (II/125),
     so flag column names are established by inspection, not memory.
  2. Distribution of the per-source position-error ellipse (major/minor/PA).
  3. Flux distributions per band -> the EMPIRICAL practical detection limit
     (where the source counts turn over), band by band.
  4. Flag-value histograms: AKARI NScanC/NScanP/MConf*/FQual*, IRAS CIRR1/2/3.
  5. Sky-density of FIR sources vs |b| -> where cirrus takes over.
  6. Gaia DR3 stellar surface density all-sky (HEALPix) -> the exact
     chance-coincidence rate for any crossmatch radius, per |b| bin.
  7. A real crossmatch + offset (null) control against Gaia/2MASS/AllWISE at
     high latitude, to confirm 6 empirically.

Runs on the GitHub Actions runner (sandbox has no egress).
Outputs under results/farir_stats/ -- summaries only, no bulk tables in git.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import pathlib
import re
import shutil
import sys
import time
import traceback
import urllib.parse
import urllib.request

import numpy as np

OUT = pathlib.Path("results/farir_stats")
OUT.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path("work_farir")
WORK.mkdir(parents=True, exist_ok=True)

MAIL = "trimcrae@gmail.com"
UA = {"User-Agent": f"Seti-farir-stats/1.0 (mailto:{MAIL})"}
RESULT: dict = {}


def log(*a):
    print(*a, flush=True)


def stage(name):
    """Decorator: never let one stage kill the run."""
    def deco(fn):
        def wrapped(*a, **k):
            log("\n" + "=" * 78 + f"\n{name}\n" + "=" * 78)
            t0 = time.time()
            try:
                out = fn(*a, **k)
                log(f"-- {name}: OK ({time.time()-t0:.0f}s)")
                return out
            except Exception as e:  # noqa: BLE001
                log(f"!! {name}: FAILED -> {e}")
                traceback.print_exc()
                RESULT.setdefault("failed_stages", []).append({"stage": name,
                                                               "error": str(e)})
                return None
        return wrapped
    return deco


def fetch(url: str, dest: pathlib.Path, tries: int = 3, timeout: int = 300):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, \
                    dest.open("wb") as f:
                shutil.copyfileobj(r, f)
            log(f"OK   {dest.name:44s} {dest.stat().st_size:11d}B  <- {url}")
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"RETRY({i+1}/{tries}) {url} -> {e}")
            time.sleep(3 * (i + 1))
    log(f"FAIL {url}")
    return None


VIZ_HOSTS = ["https://cdsarc.cds.unistra.fr/ftp",
             "https://cdsarc.u-strasbg.fr/ftp",
             "https://vizier.cfa.harvard.edu/ftp"]


def viz_get(cat: str, fname: str, dest: pathlib.Path):
    for host in VIZ_HOSTS:
        if fetch(f"{host}/{cat}/{fname}", dest, tries=2, timeout=600):
            return dest
    return None


def parse_readme_files(readme_text: str) -> list[tuple[str, int, int]]:
    """Pull (filename, lrecl, nrecords) out of a CDS ReadMe 'File Summary'."""
    out = []
    for m in re.finditer(r"^(\S+\.dat(?:\.gz)?)\s+(\d+)\s+(\d+)", readme_text,
                         re.M):
        out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def read_cds(cat: str, tag: str):
    """Download a VizieR catalogue + ReadMe and parse it with astropy's CDS
    reader, which honours the byte-by-byte description exactly."""
    from astropy.io import ascii as apascii

    rm = WORK / f"ReadMe_{tag}"
    if not any(fetch(f"{h}/{cat}/ReadMe", rm, tries=1, timeout=120)
               for h in VIZ_HOSTS):
        raise RuntimeError(f"no ReadMe for {cat}")
    rmtext = rm.read_text("utf-8", "replace")
    (OUT / f"readme_{tag}.txt").write_text(rmtext, "utf-8")

    files = parse_readme_files(rmtext)
    log(f"ReadMe lists data files: {files}")
    # Prefer the biggest .dat -- that is the catalogue proper.
    files = [f for f in files if not f[0].lower().startswith(("readme", "notes"))]
    files.sort(key=lambda t: -t[2])
    last_err = None
    for fname, lrecl, nrec in files[:3]:
        for cand in (fname, fname + ".gz"):
            dat = WORK / f"{tag}_{cand.replace('/', '_')}"
            if not viz_get(cat, cand, dat):
                continue
            try:
                if cand.endswith(".gz"):
                    plain = dat.with_suffix("")
                    with gzip.open(dat, "rb") as fi, plain.open("wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    src = plain
                else:
                    src = dat
                t = apascii.read(str(src), format="cds", readme=str(rm))
                log(f"parsed {cand}: {len(t)} rows x {len(t.colnames)} cols")
                return t, rmtext
            except Exception as e:  # noqa: BLE001
                last_err = e
                log(f"   parse of {cand} failed: {e}")
    raise RuntimeError(f"could not parse any data file for {cat}: {last_err}")


def col_inventory(t, tag: str):
    inv = []
    for c in t.colnames:
        col = t[c]
        d = {"name": c, "dtype": str(col.dtype),
             "unit": str(getattr(col, "unit", "") or ""),
             "description": str(getattr(col, "description", "") or "")}
        try:
            v = np.asarray(col)
            if v.dtype.kind in "fiu":
                fv = np.asarray(col, dtype=float)
                good = np.isfinite(fv)
                if hasattr(col, "mask"):
                    good &= ~np.asarray(col.mask)
                d["n_valid"] = int(good.sum())
                if good.sum():
                    g = fv[good]
                    d["min"] = float(np.min(g))
                    d["max"] = float(np.max(g))
                    d["median"] = float(np.median(g))
                    d["mean"] = float(np.mean(g))
                    q = np.percentile(g, [1, 5, 16, 84, 95, 99])
                    d["pct"] = {k: float(x) for k, x in
                                zip(["p1", "p5", "p16", "p84", "p95", "p99"], q)}
                    u = np.unique(g)
                    if len(u) <= 25:
                        d["values"] = {str(x): int((g == x).sum()) for x in u}
        except Exception as e:  # noqa: BLE001
            d["stat_error"] = str(e)
        inv.append(d)
    json.dump(inv, (OUT / f"columns_{tag}.json").open("w"), indent=1)
    with (OUT / f"columns_{tag}.txt").open("w") as f:
        for d in inv:
            f.write(f"{d['name']:16s} {d.get('unit',''):10s} {d['dtype']:8s} "
                    f"n={d.get('n_valid','-')}  med={d.get('median','-')}  "
                    f"{d['description'][:80]}\n")
            if "values" in d:
                f.write(f"{'':16s} values: {d['values']}\n")
    log(f"wrote columns_{tag}.{{json,txt}}  ({len(inv)} columns)")
    return inv


def pick(t, *cands):
    low = {c.lower(): c for c in t.colnames}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def arr(t, c):
    """Column -> float array with masked entries as NaN (never fill values)."""
    col = t[c]
    v = np.array(col, dtype=float)
    m = getattr(col, "mask", None)
    if m is not None:
        m = np.asarray(m)
        if m.shape == v.shape:
            v = v.copy()
            v[m] = np.nan
    return v


FULL_SKY_DEG2 = 41252.96


def band_area_deg2(b1: float, b2: float) -> float:
    """Sky area in deg^2 of the |b| in [b1, b2) double band (both hemispheres).

    Omega(|b| in [b1,b2)) = 4*pi*(sin b2 - sin b1) sr, i.e. a fraction
    (sin b2 - sin b1) of the whole sphere. Check: b1=0, b2=90 -> 41253 deg^2.
    """
    return FULL_SKY_DEG2 * (math.sin(math.radians(b2))
                            - math.sin(math.radians(b1)))


def galactic(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg, frame="icrs")
    g = c.galactic
    return np.asarray(g.l.deg), np.asarray(g.b.deg)


# ============================================================ 1-5 AKARI + IRAS
@stage("AKARI FIS BSC (VizieR II/298): columns, errors, fluxes, flags")
def akari_fis():
    t, rmtext = read_cds("II/298", "akari_fis")
    RESULT["akari_fis_nrows"] = len(t)
    RESULT["akari_fis_columns"] = list(t.colnames)
    inv = col_inventory(t, "akari_fis")

    res: dict = {"n_sources": len(t)}

    # --- position error ellipse
    for key in ("posErrMj", "posErrMi", "posErrPA", "e_RAJ2000", "e_DEJ2000",
                "errMaj", "errMin", "errPA"):
        c = pick(t, key)
        if c:
            v = arr(t, c)
            v = v[np.isfinite(v)]
            if v.size:
                res[f"poserr_{c}"] = {
                    "n": int(v.size), "median": float(np.median(v)),
                    "mean": float(np.mean(v)),
                    "pct": dict(zip(["p5", "p16", "p50", "p84", "p95", "p99"],
                                    [float(x) for x in np.percentile(
                                        v, [5, 16, 50, 84, 95, 99])]))}

    # --- fluxes: empirical turnover = practical detection limit
    flux_cols = {}
    for band, cands in {"65": ("S65", "F65", "Fnu65"),
                        "90": ("S90", "F90", "Fnu90"),
                        "140": ("S140", "F140", "Fnu140"),
                        "160": ("S160", "F160", "Fnu160")}.items():
        c = pick(t, *cands)
        if c:
            flux_cols[band] = c
    res["flux_columns"] = flux_cols

    ra_c = pick(t, "RAJ2000", "_RAJ2000", "RA")
    de_c = pick(t, "DEJ2000", "_DEJ2000", "DE")
    lg, bg = galactic(arr(t, ra_c), arr(t, de_c))
    res["galactic_available"] = True

    hi = np.abs(bg) > 30.0   # cirrus-quiet
    lo = np.abs(bg) < 10.0   # cirrus-dominated
    res["n_highlat_b30"] = int(hi.sum())
    res["n_lowlat_b10"] = int(lo.sum())

    for band, c in flux_cols.items():
        v = arr(t, c)
        ok = np.isfinite(v) & (v > 0)
        entry = {"column": c, "n_positive": int(ok.sum())}
        if ok.sum():
            g = v[ok]
            entry["pct"] = dict(zip(
                ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p99"],
                [float(x) for x in np.percentile(
                    g, [1, 5, 10, 25, 50, 75, 90, 99])]))
            entry["min"] = float(g.min())
            entry["max"] = float(g.max())
            # log-flux histogram: the turnover is the practical limit
            lgf = np.log10(g)
            h, edges = np.histogram(lgf, bins=np.arange(-2.0, 4.01, 0.1))
            entry["loghist_edges"] = [round(float(x), 3) for x in edges]
            entry["loghist_all"] = [int(x) for x in h]
            pk = int(np.argmax(h))
            entry["turnover_Jy_all"] = round(float(10 ** ((edges[pk] +
                                                           edges[pk + 1]) / 2)), 4)
            for lbl, msk in (("highlat_b30", hi), ("lowlat_b10", lo)):
                m = ok & msk
                if m.sum() > 50:
                    hh, _ = np.histogram(np.log10(v[m]),
                                         bins=np.arange(-2.0, 4.01, 0.1))
                    entry[f"loghist_{lbl}"] = [int(x) for x in hh]
                    p = int(np.argmax(hh))
                    entry[f"turnover_Jy_{lbl}"] = round(
                        float(10 ** ((edges[p] + edges[p + 1]) / 2)), 4)
                    entry[f"n_{lbl}"] = int(m.sum())
        res[f"flux_{band}um"] = entry

    # --- flag columns: anything that looks like quality/confusion/scan
    flagpat = re.compile(r"(qual|flag|conf|scan|ndens|nscan|mconf|fqual|q_)",
                         re.I)
    flags = {}
    for c in t.colnames:
        if not flagpat.search(c):
            continue
        try:
            v = np.asarray(t[c])
            if v.dtype.kind in "fiu":
                fv = arr(t, c)
                fv = fv[np.isfinite(fv)]
            else:
                fv = v
            u, cnt = np.unique(fv, return_counts=True)
            if len(u) <= 40:
                flags[c] = {str(k): int(n) for k, n in zip(u, cnt)}
            else:
                flags[c] = {"n_unique": int(len(u)),
                            "median": float(np.median(fv))
                            if fv.dtype.kind in "fiu" else None}
            flags[c]["_description"] = str(
                getattr(t[c], "description", "") or "")
        except Exception as e:  # noqa: BLE001
            flags[c] = {"error": str(e)}
    res["flag_columns"] = flags

    # --- sky density vs |b|
    bins = np.array([0, 5, 10, 15, 20, 30, 40, 50, 60, 90], dtype=float)
    dens = []
    for i in range(len(bins) - 1):
        m = (np.abs(bg) >= bins[i]) & (np.abs(bg) < bins[i + 1])
        area = band_area_deg2(bins[i], bins[i + 1])
        dens.append({"b_lo": float(bins[i]), "b_hi": float(bins[i + 1]),
                     "n": int(m.sum()), "area_deg2": round(area, 1),
                     "per_deg2": round(float(m.sum()) / area, 4)})
    res["density_vs_b"] = dens
    res["all_sky_per_deg2"] = round(len(t) / FULL_SKY_DEG2, 4)

    json.dump(res, (OUT / "akari_fis_stats.json").open("w"), indent=1)
    # keep positions for the crossmatch stage
    np.save(WORK / "akari_ra.npy", arr(t, ra_c))
    np.save(WORK / "akari_de.npy", arr(t, de_c))
    np.save(WORK / "akari_b.npy", bg)
    log(json.dumps({k: v for k, v in res.items()
                    if k in ("n_sources", "flux_columns", "all_sky_per_deg2")},
                   indent=1))
    return res


@stage("IRAS PSC (VizieR II/125): columns, error ellipse, CIRR flags")
def iras_psc():
    t, rmtext = read_cds("II/125", "iras_psc")
    RESULT["iras_psc_nrows"] = len(t)
    inv = col_inventory(t, "iras_psc")
    res: dict = {"n_sources": len(t), "columns": list(t.colnames)}

    for key in ("Major", "Minor", "PosAng", "MAJOR", "MINOR", "POSANG",
                "e_RA", "e_DE", "unc_maj", "unc_min"):
        c = pick(t, key)
        if c:
            v = arr(t, c)
            v = v[np.isfinite(v)]
            if v.size:
                res[f"poserr_{c}"] = {
                    "n": int(v.size), "unit": str(t[c].unit or ""),
                    "median": float(np.median(v)),
                    "pct": dict(zip(["p5", "p16", "p50", "p84", "p95", "p99"],
                                    [float(x) for x in np.percentile(
                                        v, [5, 16, 50, 84, 95, 99])]))}

    for c in t.colnames:
        if re.search(r"cirr", c, re.I):
            v = arr(t, c)
            v = v[np.isfinite(v)]
            u, cnt = np.unique(v, return_counts=True)
            res[f"flag_{c}"] = {
                "description": str(getattr(t[c], "description", "") or ""),
                "unit": str(t[c].unit or ""),
                "n_valid": int(v.size),
                "hist": {str(int(k)): int(n) for k, n in zip(u, cnt)}
                if len(u) <= 60 else
                {"median": float(np.median(v)),
                 "pct": [float(x) for x in np.percentile(v, [5, 50, 95])]}}

    flagpat = re.compile(r"(qual|flag|conf|cirr|nh|rel|q_)", re.I)
    fl = {}
    for c in t.colnames:
        if flagpat.search(c):
            try:
                v = np.asarray(t[c])
                fv = arr(t, c) if v.dtype.kind in "fiu" else v
                if v.dtype.kind in "fiu":
                    fv = fv[np.isfinite(fv)]
                u, cnt = np.unique(fv, return_counts=True)
                fl[c] = ({str(k): int(n) for k, n in zip(u, cnt)}
                         if len(u) <= 40 else {"n_unique": int(len(u))})
                fl[c]["_description"] = str(getattr(t[c], "description", "") or "")
            except Exception as e:  # noqa: BLE001
                fl[c] = {"error": str(e)}
    res["flag_columns"] = fl

    fx = {}
    for band in ("12", "25", "60", "100"):
        c = pick(t, f"F{band}", f"S{band}", f"Fnu_{band}", f"Flux{band}")
        if c:
            v = arr(t, c)
            v = v[np.isfinite(v) & (v > 0)]
            if v.size:
                fx[band] = {"column": c, "unit": str(t[c].unit or ""),
                            "n": int(v.size),
                            "pct": dict(zip(["p1", "p10", "p50", "p90", "p99"],
                                            [float(x) for x in np.percentile(
                                                v, [1, 10, 50, 90, 99])]))}
    res["fluxes"] = fx

    ra_c = pick(t, "RAJ2000", "_RAJ2000", "RA1950", "RA")
    de_c = pick(t, "DEJ2000", "_DEJ2000", "DE1950", "DE")
    if ra_c and de_c:
        try:
            ra = arr(t, ra_c)
            de = arr(t, de_c)
            _, bg = galactic(ra, de)
            res["n_highlat_b30"] = int((np.abs(bg) > 30).sum())
            res["all_sky_per_deg2"] = round(len(t) / 41252.96, 4)
            np.save(WORK / "iras_ra.npy", ra)
            np.save(WORK / "iras_de.npy", de)
            np.save(WORK / "iras_b.npy", bg)
        except Exception as e:  # noqa: BLE001
            res["galactic_error"] = str(e)

    json.dump(res, (OUT / "iras_psc_stats.json").open("w"), indent=1)
    return res


# ==================================================== 6. Gaia density all-sky
@stage("Gaia DR3 all-sky stellar surface density (HEALPix) for chance coincidence")
def gaia_density():
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    res = {}
    try:
        q = ("SELECT gaia_healpix_index(4, source_id) AS hpx, COUNT(*) AS n "
             "FROM gaiadr3.gaia_source GROUP BY hpx")
        log("submitting Gaia HEALPix(4) GROUP BY ...")
        job = Gaia.launch_job_async(q, dump_to_file=False)
        r = job.get_results()
        hpx = np.asarray(r["hpx"], dtype=np.int64)
        n = np.asarray(r["n"], dtype=np.float64)
        nside = 16
        npix = 12 * nside * nside
        counts = np.zeros(npix)
        counts[hpx] = n
        area = 41252.96 / npix          # deg^2 per pixel
        dens = counts / area            # stars deg^-2
        np.save(WORK / "gaia_dens_nside16.npy", dens)
        res["method"] = "healpix_nside16_groupby"
        res["npix"] = npix
        res["pix_area_deg2"] = round(area, 4)
        res["total_sources"] = float(counts.sum())
        res["density_deg2"] = {
            "min": float(dens.min()), "max": float(dens.max()),
            "median": float(np.median(dens)), "mean": float(dens.mean()),
            "pct": dict(zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"],
                            [float(x) for x in np.percentile(
                                dens, [1, 5, 25, 50, 75, 95, 99])]))}
        # density by |b|
        try:
            import healpy as hp
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            th, ph = hp.pix2ang(nside, np.arange(npix), nest=True)
            c = SkyCoord(ra=np.degrees(ph) * u.deg,
                         dec=(90 - np.degrees(th)) * u.deg, frame="icrs")
            bb = np.abs(c.galactic.b.deg)
            byb = []
            for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45),
                           (45, 60), (60, 90)]:
                m = (bb >= lo) & (bb < hi)
                if m.sum():
                    byb.append({"b_lo": lo, "b_hi": hi, "npix": int(m.sum()),
                                "median_per_deg2": round(float(np.median(dens[m])), 1),
                                "mean_per_deg2": round(float(dens[m].mean()), 1)})
            res["density_by_absb"] = byb
            np.save(WORK / "gaia_dens_absb.npy", bb)
        except Exception as e:  # noqa: BLE001
            res["healpy_error"] = str(e)
    except Exception as e:  # noqa: BLE001
        log(f"HEALPix aggregate failed ({e}); falling back to cone counts")
        res["method"] = "cone_counts_fallback"
        res["healpix_error"] = str(e)
        rows = []
        rng = np.random.default_rng(7)
        try:
            ra = np.load(WORK / "akari_ra.npy")
            de = np.load(WORK / "akari_de.npy")
            idx = rng.choice(len(ra), size=min(200, len(ra)), replace=False)
            pts = list(zip(ra[idx], de[idx]))
        except Exception:  # noqa: BLE001
            pts = [(float(rng.uniform(0, 360)),
                    float(np.degrees(np.arcsin(rng.uniform(-1, 1)))))
                   for _ in range(200)]
        for i, (a, d) in enumerate(pts):
            try:
                j = Gaia.launch_job(
                    "SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE "
                    "1=CONTAINS(POINT('ICRS',ra,dec),"
                    f"CIRCLE('ICRS',{a:.6f},{d:.6f},0.1))")
                n = int(j.get_results()["n"][0])
                rows.append({"ra": a, "dec": d, "n_r0.1deg": n,
                             "per_deg2": round(n / (math.pi * 0.1 ** 2), 1)})
            except Exception as e:  # noqa: BLE001
                log(f"  cone {i} failed: {e}")
            if i % 25 == 0:
                log(f"  cone {i}/{len(pts)}")
        res["cones"] = rows
        if rows:
            v = np.array([r["per_deg2"] for r in rows])
            res["density_deg2"] = {"median": float(np.median(v)),
                                   "mean": float(v.mean()),
                                   "pct": [float(x) for x in
                                           np.percentile(v, [5, 50, 95])]}
    json.dump(res, (OUT / "gaia_density.json").open("w"), indent=1)
    return res


@stage("Chance-coincidence: expected N(Gaia) inside r, per AKARI/IRAS source")
def chance_coincidence(gdens):
    """Fold the measured Gaia density map through the FIR source positions."""
    res = {}
    try:
        dens = np.load(WORK / "gaia_dens_nside16.npy")
        import healpy as hp
        nside = 16
        have_map = True
    except Exception as e:  # noqa: BLE001
        log(f"no density map ({e}); using scalar densities")
        have_map = False

    RADII = [3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120]
    for tag in ("akari", "iras"):
        try:
            ra = np.load(WORK / f"{tag}_ra.npy")
            de = np.load(WORK / f"{tag}_de.npy")
            bg = np.load(WORK / f"{tag}_b.npy")
        except Exception:  # noqa: BLE001
            continue
        if have_map:
            ipix = hp.ang2pix(nside, ra, de, lonlat=True, nest=True)
            d_at = dens[ipix]                      # Gaia stars deg^-2 per source
        else:
            d_at = np.full(len(ra), 43800.0)       # 1.81e9 / 41253
        out = {"n_fir_sources": int(len(ra)),
               "gaia_density_at_fir_positions_per_deg2": {
                   "median": float(np.median(d_at)),
                   "mean": float(d_at.mean()),
                   "pct": dict(zip(["p5", "p25", "p50", "p75", "p95", "p99"],
                                   [float(x) for x in np.percentile(
                                       d_at, [5, 25, 50, 75, 95, 99])]))},
               "by_radius": []}
        for r_as in RADII:
            area = math.pi * (r_as / 3600.0) ** 2      # deg^2
            mu = d_at * area                            # expected chance stars
            row = {"radius_arcsec": r_as,
                   "area_deg2": area,
                   "mu_median": float(np.median(mu)),
                   "mu_mean": float(mu.mean()),
                   "P_ge1_median": float(1 - math.exp(-float(np.median(mu)))),
                   "P_ge1_mean": float(np.mean(1 - np.exp(-mu))),
                   "frac_sources_P_ge1_gt_0p5":
                       float(np.mean((1 - np.exp(-mu)) > 0.5))}
            for lbl, m in (("b_gt_30", np.abs(bg) > 30),
                           ("b_gt_45", np.abs(bg) > 45),
                           ("b_lt_10", np.abs(bg) < 10)):
                if m.sum() > 10:
                    row[f"mu_median_{lbl}"] = float(np.median(mu[m]))
                    row[f"P_ge1_mean_{lbl}"] = float(np.mean(1 - np.exp(-mu[m])))
            out["by_radius"].append(row)
        res[tag] = out
        log(f"{tag}: " + json.dumps(
            [{k: round(v, 5) for k, v in r.items()
              if k in ("radius_arcsec", "mu_median", "P_ge1_mean")}
             for r in out["by_radius"]]))
    json.dump(res, (OUT / "chance_coincidence.json").open("w"), indent=1)
    return res


@stage("Empirical xmatch + offset null control (high latitude)")
def xmatch_null():
    """Real CDS X-Match of AKARI FIS positions vs Gaia/2MASS/AllWISE, and the
    same positions displaced by +5 arcmin in Dec (pure chance alignments)."""
    from astroquery.xmatch import XMatch
    from astropy.table import Table
    import astropy.units as u

    ra = np.load(WORK / "akari_ra.npy")
    de = np.load(WORK / "akari_de.npy")
    bg = np.load(WORK / "akari_b.npy")
    hi = np.abs(bg) > 30
    rng = np.random.default_rng(11)
    idx = np.where(hi)[0]
    idx = rng.choice(idx, size=min(2500, len(idx)), replace=False)
    res = {"n_test_sources": int(len(idx)), "selection": "|b|>30 deg",
           "offset_arcmin": 5.0}

    CATS = {"gaia_dr3": "vizier:I/355/gaiadr3",
            "2mass": "vizier:II/246/out",
            "allwise": "vizier:II/328/allwise"}
    for tag, cat in CATS.items():
        for mode in ("real", "offset"):
            dd = de[idx] + (5.0 / 60.0 if mode == "offset" else 0.0)
            dd = np.clip(dd, -89.9, 89.9)
            tb = Table({"id": np.arange(len(idx)), "ra": ra[idx], "dec": dd})
            try:
                m = XMatch.query(cat1=tb, cat2=cat,
                                 max_distance=60 * u.arcsec,
                                 colRA1="ra", colDec1="dec")
                sep = np.asarray(m["angDist"], dtype=float)
                ids = np.asarray(m["id"], dtype=int)
                ent = {"n_matches_within_60as": int(len(m))}
                for r_as in (3, 5, 10, 15, 20, 30, 40, 60):
                    k = sep <= r_as
                    ent[f"n_within_{r_as}as"] = int(k.sum())
                    ent[f"frac_sources_with_match_{r_as}as"] = round(
                        float(len(np.unique(ids[k])) / len(idx)), 5)
                    ent[f"mean_matches_per_source_{r_as}as"] = round(
                        float(k.sum() / len(idx)), 5)
                res[f"{tag}_{mode}"] = ent
                log(f"{tag} {mode}: " + json.dumps(
                    {k: v for k, v in ent.items() if "frac_sources" in k}))
            except Exception as e:  # noqa: BLE001
                log(f"xmatch {tag} {mode} failed: {e}")
                res[f"{tag}_{mode}"] = {"error": str(e)}
            time.sleep(2)
    json.dump(res, (OUT / "xmatch_null.json").open("w"), indent=1)
    return res


@stage("SFD 100um cirrus brightness at the FIR source positions")
def cirrus_levels():
    """Fraction of sky above a given 100um cirrus surface brightness."""
    res = {}
    try:
        from dustmaps.config import config as dmconfig
        dmconfig["data_dir"] = str(WORK / "dustmaps")
        import dustmaps.sfd
        dustmaps.sfd.fetch()
        from dustmaps.sfd import SFDQuery
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        sfd = SFDQuery()
        rng = np.random.default_rng(3)
        n = 400000
        lon = rng.uniform(0, 360, n)
        lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
        c = SkyCoord(lon * u.deg, lat * u.deg, frame="galactic")
        ebv = sfd(c)
        # SFD: E(B-V) = p * I_100 with p = 0.0184 mag / (MJy/sr) after
        # temperature correction -> invert for an indicative I_100.
        i100 = ebv / 0.0184
        bb = np.abs(lat)
        res["sfd_ebv_allsky"] = {
            "median": float(np.median(ebv)),
            "pct": dict(zip(["p5", "p25", "p50", "p75", "p90", "p95", "p99"],
                            [float(x) for x in np.percentile(
                                ebv, [5, 25, 50, 75, 90, 95, 99])]))}
        res["i100_MJy_sr_allsky"] = {
            "median": float(np.median(i100)),
            "pct": dict(zip(["p5", "p25", "p50", "p75", "p90", "p95", "p99"],
                            [float(x) for x in np.percentile(
                                i100, [5, 25, 50, 75, 90, 95, 99])]))}
        res["frac_sky_above_I100"] = {
            f"{thr}": round(float((i100 > thr).mean()), 4)
            for thr in (1, 2, 3, 5, 10, 20, 50, 100)}
        byb = []
        for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45),
                       (45, 60), (60, 90)]:
            m = (bb >= lo) & (bb < hi)
            if m.sum():
                byb.append({"b_lo": lo, "b_hi": hi,
                            "median_ebv": round(float(np.median(ebv[m])), 4),
                            "median_I100_MJy_sr": round(float(np.median(i100[m])), 3),
                            "p90_I100_MJy_sr": round(float(np.percentile(i100[m], 90)), 3)})
        res["by_absb"] = byb
        try:
            ra = np.load(WORK / "akari_ra.npy")
            de = np.load(WORK / "akari_de.npy")
            cc = SkyCoord(ra * u.deg, de * u.deg, frame="icrs")
            e2 = sfd(cc)
            res["at_akari_positions"] = {
                "median_ebv": float(np.median(e2)),
                "median_I100_MJy_sr": float(np.median(e2 / 0.0184)),
                "pct_I100": dict(zip(["p5", "p50", "p95"],
                                     [float(x / 0.0184) for x in
                                      np.percentile(e2, [5, 50, 95])]))}
        except Exception as e:  # noqa: BLE001
            res["akari_positions_error"] = str(e)
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e)
        log(f"dustmaps unavailable: {e}")
    json.dump(res, (OUT / "cirrus_levels.json").open("w"), indent=1)
    return res


def main():
    akari_fis()
    iras_psc()
    g = gaia_density()
    chance_coincidence(g)
    xmatch_null()
    cirrus_levels()
    json.dump(RESULT, (OUT / "run_summary.json").open("w"), indent=1)
    log("\nALL STAGES ATTEMPTED")


if __name__ == "__main__":
    main()
    sys.exit(0)

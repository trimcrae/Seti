"""Readers: Gaia epoch photometry and epoch astrometry -> normalised per-transit frames.

Every reader resolves its columns at run time (``schema.resolve``) and hands
back the same normalised shape whatever it was given:

``photometry`` (one row per source x transit)
    source_id, transit_id, t (BJD_TCB - 2455197.5, d), f_g, e_g, f_bp, e_bp,
    f_rp, e_rp, n_obs_g, bad_g, bad_bp, bad_rp, vrej_g, vrej_bp, vrej_rp

``astrometry`` (one row per source x transit, the CCDs of a transit combined)
    source_id, transit_id, t (d, barycentric where the correction exists),
    t_yr (years from J2017.5, the DR4 reference epoch), theta (rad),
    pf_al, x_al (mas), sx_al (mas), n_ccd, excess_noise (mas), g_mag,
    dist_ci (pix), cf_al

Time.  DR3 epoch photometry times are barycentric TCB days from
JD 2455197.5.  DR4 ``obs_time_tcb`` is int64 ns from the same origin **at
Gaia**; the draft DR4 data model defines ``obs_time_bary_corr`` as
TCB(barycentric) - TCB(at Gaia) for the AF4 CCD, so t_bary = obs_time_tcb +
obs_time_bary_corr (the prerelease VOTable's ``refposition=BARYCENTER`` is
contradicted by the data model).  Verified here on real data: for Gaia-4,
63 of 65 DR3 photometric transit_ids are present in the DR4 prerelease and
the barycentric times agree to ~5 s.

Direction convention.  The along-scan unit vector in (alpha*, delta) is
(sin theta, cos theta) with theta = ``scan_pos_angle``.  Verified: a
weighted 5-parameter fit of ``centroid_pos_al`` with this convention recovers
the published parallaxes of all 12 prerelease sources (HD 114762 25.63 vs
25.6 mas, Gaia-4 13.62 vs 13.6, LSPM J1324+6124 10.11 vs 10.1, ...).
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    ASTRO_REQUIRED,
    ASTRO_ROLES,
    PHOT_REQUIRED,
    PHOT_ROLES,
    Resolution,
    as_uint16_flags,
    parse_array_column,
    resolve,
)

NS_PER_DAY = 86400.0e9
#: J2017.5 (TCB) in days from JD 2455197.5 -- the DR4 reference epoch
#: (gaiasupdate.constants.DR4_REFERENCE_EPOCH = Time('2017.5', format='jyear')).
J2017P5_DAYS = (2451545.0 + 17.5 * 365.25) - 2455197.5
PHOT_COLUMNS = ["source_id", "transit_id", "t", "f_g", "e_g", "f_bp", "e_bp", "f_rp", "e_rp",
                "n_obs_g", "bad_g", "bad_bp", "bad_rp", "vrej_g", "vrej_bp", "vrej_rp"]
ASTRO_COLUMNS = ["source_id", "transit_id", "t", "t_yr", "theta", "pf_al", "x_al", "sx_al",
                 "n_ccd", "excess_noise", "g_mag", "dist_ci", "cf_al"]


class SchemaError(ValueError):
    """A required role resolved to no column.  Carries the Resolution."""

    def __init__(self, msg: str, resolution: Resolution):
        super().__init__(msg)
        self.resolution = resolution


def _to_frame(obj) -> pd.DataFrame:
    """astropy Table / VOTable / DataFrame -> DataFrame, keeping array cells
    as numpy object cells and masked scalars as NaN."""
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        from astropy.table import Table
    except Exception:  # pragma: no cover
        Table = None  # noqa: N806
    if Table is not None and isinstance(obj, Table):
        out = {}
        for name in obj.colnames:
            col = obj[name]
            if getattr(col, "ndim", 1) > 1 or col.dtype == object:
                vals = []
                for v in col:
                    if np.ma.isMaskedArray(v):
                        v = np.ma.filled(v.astype(float) if v.dtype.kind in "iub" else v,
                                         np.nan)
                    vals.append(np.asarray(v))
                out[name] = vals
            else:
                a = col
                if np.ma.isMaskedArray(a) or hasattr(a, "mask"):
                    m = np.ma.getmaskarray(a)
                    if a.dtype.kind in "iu" and m.any():
                        arr = np.asarray(np.ma.filled(a.astype(float), np.nan))
                    elif a.dtype.kind == "b":
                        arr = np.asarray(np.ma.filled(a, False))
                    elif a.dtype.kind == "f":
                        arr = np.asarray(np.ma.filled(a, np.nan))
                    else:
                        arr = np.asarray(np.ma.filled(a))
                else:
                    arr = np.asarray(a)
                out[name] = arr
        return pd.DataFrame(out)
    raise TypeError(f"cannot read {type(obj)!r} as a table")


def _is_array_cell(v) -> bool:
    return isinstance(v, (list, tuple, np.ndarray)) and not isinstance(v, (str, bytes))


def _column_arrays(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """A column of array cells (real arrays or text) -> flat values + lengths."""
    vals = series.to_numpy(dtype=object)
    if len(vals) and any(_is_array_cell(v) for v in vals[: min(len(vals), 5)]):
        lens = np.array([len(v) if _is_array_cell(v) else 0 for v in vals], dtype=np.int64)
        flat = [np.asarray(v, dtype=float).ravel() if _is_array_cell(v) else np.zeros(0)
                for v in vals]
        return (np.concatenate(flat) if flat else np.zeros(0)), lens
    return parse_array_column(vals)


# ---------------------------------------------------------------------------
# Photometry
# ---------------------------------------------------------------------------
def _flag(df: pd.DataFrame, res: Resolution, role: str, n: int) -> np.ndarray:
    c = res.get(role)
    if c is None:
        return np.zeros(n, dtype=bool)
    v = pd.to_numeric(df[c].map(lambda x: {"true": 1, "false": 0}.get(str(x).lower(), x)),
                      errors="coerce").to_numpy(dtype=float, copy=True)
    return np.nan_to_num(v, nan=0.0) != 0


def photometry_from_long(obj, source_id: int | None = None) -> pd.DataFrame:
    """Long form (one row per transit): the DR3 DataLink EPOCH_PHOTOMETRY shape.

    ``source_id`` is required when the table has no source_id column (a
    per-source DataLink file carries it only in its file name)."""
    df = _to_frame(obj)
    res = resolve(df.columns, PHOT_ROLES, PHOT_REQUIRED)
    if res.missing:
        raise SchemaError(f"epoch photometry lacks roles {res.missing}", res)
    n = len(df)
    num = lambda role: pd.to_numeric(df[res.get(role)], errors="coerce").to_numpy(  # noqa: E731
        dtype=float, copy=True) if res.get(role) else np.full(n, np.nan)
    if res.get("source_id"):
        sid = pd.to_numeric(df[res.get("source_id")], errors="coerce").to_numpy()
    elif source_id is not None:
        sid = np.full(n, int(source_id))
    else:
        raise SchemaError("no source_id column and none supplied", res)
    tid = (pd.to_numeric(df[res.get("transit_id")], errors="coerce").to_numpy()
           if res.get("transit_id") else np.arange(n))
    bad_common = _flag(df, res, "rejected", n) | _flag(df, res, "noisy", n)
    other = {b: (as_uint16_flags(pd.to_numeric(df[res.get(f"{b}_other_flags")],
                                               errors="coerce").to_numpy(dtype=float, copy=True))
                 if res.get(f"{b}_other_flags") else np.zeros(n, np.uint16)) for b in ("g", "bp", "rp")}
    out = pd.DataFrame({
        "source_id": np.asarray(sid).astype(np.int64),
        "transit_id": np.asarray(tid).astype(np.int64),
        "t": num("g_time"),
        "f_g": num("g_flux"), "e_g": num("g_flux_error"),
        "f_bp": num("bp_flux"), "e_bp": num("bp_flux_error"),
        "f_rp": num("rp_flux"), "e_rp": num("rp_flux_error"),
        "n_obs_g": num("g_n_obs"),
        "bad_g": bad_common,
        "bad_bp": bad_common | _flag(df, res, "bp_reject", n) | _flag(df, res, "bp_unavailable", n),
        "bad_rp": bad_common | _flag(df, res, "rp_reject", n) | _flag(df, res, "rp_unavailable", n),
        "vrej_g": _flag(df, res, "g_var_reject", n),
        "vrej_bp": _flag(df, res, "bp_var_reject", n),
        "vrej_rp": _flag(df, res, "rp_var_reject", n),
    })
    # g_time missing on a transit where BP/RP exist: fall back to their time.
    for b in ("bp_time", "rp_time"):
        if res.get(b):
            tb = pd.to_numeric(df[res.get(b)], errors="coerce").to_numpy(dtype=float, copy=True)
            miss = ~np.isfinite(out["t"].to_numpy())
            if miss.any():
                tt = out["t"].to_numpy(copy=True)
                tt[miss] = tb[miss]
                out["t"] = tt
    out.attrs["other_flags_nonzero"] = {b: int((v != 0).sum()) for b, v in other.items()}
    out.attrs["resolution"] = res.as_dict()
    return out


def photometry_from_wide(obj) -> pd.DataFrame:
    """Wide form (one row per source, array cells): the DR3 CDN bulk shape and
    the shape a DR4 TAP/DataLink 'COMBINED' product is expected to take."""
    df = _to_frame(obj)
    res = resolve(df.columns, PHOT_ROLES, PHOT_REQUIRED + ("source_id",))
    if res.missing:
        raise SchemaError(f"wide epoch photometry lacks roles {res.missing}", res)
    sid = pd.to_numeric(df[res.get("source_id")], errors="coerce").to_numpy()
    # The length of every array column must agree row by row; the G time
    # column defines the transit count.
    base_vals, base_len = _column_arrays(df[res.get("g_time")])
    cols: dict[str, np.ndarray] = {}
    roles = {"transit_id": "transit_id", "f_g": "g_flux", "e_g": "g_flux_error",
             "f_bp": "bp_flux", "e_bp": "bp_flux_error", "f_rp": "rp_flux",
             "e_rp": "rp_flux_error", "n_obs_g": "g_n_obs", "noisy": "noisy",
             "rejected": "rejected", "vrej_g": "g_var_reject", "vrej_bp": "bp_var_reject",
             "vrej_rp": "rp_var_reject", "bp_reject": "bp_reject", "rp_reject": "rp_reject",
             "bp_unav": "bp_unavailable", "rp_unav": "rp_unavailable",
             "bp_time": "bp_time", "rp_time": "rp_time"}
    n_tot = int(base_len.sum())
    mismatched_rows = 0
    for key, role in roles.items():
        c = res.get(role)
        if c is None:
            continue
        v, ln = _column_arrays(df[c])
        if not np.array_equal(ln, base_len):
            # Zero-length arrays stand for "column absent on this row"; any
            # other mismatch would misalign transits silently -- refuse it.
            ok = (ln == 0) | (ln == base_len)
            if not ok.all():
                mismatched_rows += int((~ok).sum())
            full = np.full(n_tot, np.nan)
            edges_b = np.concatenate([[0], np.cumsum(base_len)])
            edges_v = np.concatenate([[0], np.cumsum(ln)])
            for i in np.nonzero(ln == base_len)[0]:
                full[edges_b[i]:edges_b[i + 1]] = v[edges_v[i]:edges_v[i + 1]]
            v = full
        cols[key] = v
    n = n_tot
    nanv = np.full(n, np.nan)
    flag = lambda k: np.nan_to_num(cols.get(k, np.zeros(n)), nan=0.0) != 0  # noqa: E731
    t = base_vals.copy()
    for k in ("bp_time", "rp_time"):
        if k in cols:
            miss = ~np.isfinite(t)
            t[miss] = cols[k][miss]
    bad_common = flag("noisy") | flag("rejected")
    out = pd.DataFrame({
        "source_id": np.repeat(np.asarray(sid).astype(np.int64), base_len),
        "transit_id": (np.nan_to_num(cols["transit_id"], nan=-1).astype(np.int64)
                       if "transit_id" in cols else np.arange(n, dtype=np.int64)),
        "t": t,
        "f_g": cols.get("f_g", nanv), "e_g": cols.get("e_g", nanv),
        "f_bp": cols.get("f_bp", nanv), "e_bp": cols.get("e_bp", nanv),
        "f_rp": cols.get("f_rp", nanv), "e_rp": cols.get("e_rp", nanv),
        "n_obs_g": cols.get("n_obs_g", nanv),
        "bad_g": bad_common,
        "bad_bp": bad_common | flag("bp_reject") | flag("bp_unav"),
        "bad_rp": bad_common | flag("rp_reject") | flag("rp_unav"),
        "vrej_g": flag("vrej_g"), "vrej_bp": flag("vrej_bp"), "vrej_rp": flag("vrej_rp"),
    })
    out.attrs["resolution"] = res.as_dict()
    out.attrs["mismatched_array_rows"] = mismatched_rows
    return out


def read_bulk_csv_header(fh_or_path) -> list[str]:
    """The column names of a Gaia CDN csv(.gz), skipping ECSV '#' lines."""
    opener = _open_text(fh_or_path)
    with opener as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            return [c.strip().strip('"') for c in line.rstrip("\n").split(",")]
    return []


def _open_text(p):
    if hasattr(p, "read"):
        return io.TextIOWrapper(p, encoding="utf-8") if isinstance(p.read(0), bytes) else p
    p = Path(p)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, encoding="utf-8")


def read_bulk_csv(path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Read one Gaia CDN epoch-photometry csv(.gz) into the normalised long frame.

    Only the columns the detector needs are parsed (``usecols`` from the
    discovered header), so an extra column in a later release costs nothing."""
    header = read_bulk_csv_header(path)
    res = resolve(header, PHOT_ROLES, PHOT_REQUIRED + ("source_id",))
    if res.missing:
        raise SchemaError(f"bulk file lacks roles {res.missing}; header={header[:40]}", res)
    use = sorted(set(res.mapping.values()))
    df = pd.read_csv(path, comment="#", usecols=use, dtype=str, nrows=max_rows,
                     keep_default_na=False)
    return photometry_from_wide(df)


# ---------------------------------------------------------------------------
# Astrometry
# ---------------------------------------------------------------------------
def astrometry_ccd_frame(obj) -> pd.DataFrame:
    """Per-CCD long frame from an epoch-astrometry table.

    Handles the prerelease shape (per-transit rows, per-CCD array cells of
    length 10 -- or 0) and a per-transit scalar shape (SSO observations,
    or a DR4 table that ships one centroid per transit)."""
    df = _to_frame(obj)
    res = resolve(df.columns, ASTRO_ROLES, ASTRO_REQUIRED)
    if res.missing:
        raise SchemaError(f"epoch astrometry lacks roles {res.missing}", res)
    c_al = res.get("centroid_al")
    sample = df[c_al].iloc[0] if len(df) else None
    per_ccd = _is_array_cell(sample) or (isinstance(sample, str) and sample.strip()[:1] in "[({")
    n_tr = len(df)

    def scalar(role, default=np.nan):
        c = res.get(role)
        if c is None:
            return np.full(n_tr, default, dtype=float)
        return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float, copy=True)

    if per_ccd:
        x, ln = _column_arrays(df[c_al])
        n_ccd = ln
        rep = lambda a: np.repeat(a, n_ccd)  # noqa: E731

        def arr(role, default=np.nan):
            c = res.get(role)
            if c is None:
                return np.full(int(n_ccd.sum()), default, dtype=float)
            v, lv = _column_arrays(df[c])
            if np.array_equal(lv, n_ccd):
                return v
            if (lv == 0).all():
                return np.full(int(n_ccd.sum()), default, dtype=float)
            # per-transit scalar in an array-shaped table: broadcast it
            if (lv <= 1).all():
                vals = np.array([v[i] if lv[i] else default for i in range(len(lv))])
                return rep(vals)
            raise ValueError(f"column {c} has per-row lengths that disagree with {c_al}")

        t_ccd = arr("obs_time")
        theta = arr("scan_angle")
        ex = arr("centroid_al_error")
        used = arr("used_al", default=1.0)
        flags = arr("ccd_proc_flags", default=0.0)
        dist_ci = arr("dist_to_ci")
        cf = arr("colour_factor_al")
        tr_index = np.repeat(np.arange(n_tr), n_ccd)
    else:
        x = scalar("centroid_al")
        n_ccd = np.ones(n_tr, dtype=np.int64)
        t_ccd = scalar("obs_time")
        theta = scalar("scan_angle")
        ex = scalar("centroid_al_error")
        used = scalar("used_al", default=1.0)
        flags = scalar("ccd_proc_flags", default=0.0)
        dist_ci = scalar("dist_to_ci")
        cf = scalar("colour_factor_al")
        tr_index = np.arange(n_tr)

    # Time units: int64 ns (DR4) vs days (SSO epoch is JD-ish; handled by caller).
    tc = res.get("obs_time")
    ns = tc is not None and str(tc).lower() in ("obs_time_tcb", "obstimetcb")
    bary = scalar("bary_corr", default=0.0)
    t_days = t_ccd / NS_PER_DAY if ns else t_ccd
    bary_days = (bary / NS_PER_DAY) if ns else np.zeros(n_tr)
    sid = (pd.to_numeric(df[res.get("source_id")], errors="coerce").to_numpy()
           if res.get("source_id") else np.zeros(n_tr))
    tid = pd.to_numeric(df[res.get("transit_id")], errors="coerce").to_numpy()
    out = pd.DataFrame({
        "source_id": sid[tr_index].astype(np.int64),
        "transit_id": tid[tr_index].astype(np.int64),
        "t": t_days + bary_days[tr_index],
        "bary_ok": np.isfinite(bary_days[tr_index]),
        "theta": np.deg2rad(theta),
        "pf_al": scalar("parallax_factor_al")[tr_index],
        "x_al": x,
        "sx_al": ex,
        "used": np.nan_to_num(used, nan=0.0) != 0,
        "ccd_flags": as_uint16_flags(flags),
        "excess_noise": scalar("excess_noise", default=0.0)[tr_index],
        "g_mag": scalar("g_mag")[tr_index],
        "dist_ci": dist_ci,
        "cf_al": cf,
        "multipeak": (scalar("multipeak", default=0.0) != 0)[tr_index],
        "blended": (scalar("blended", default=0.0) != 0)[tr_index],
    })
    out.attrs["resolution"] = res.as_dict()
    out.attrs["per_ccd"] = bool(per_ccd)
    return out


def collapse_transits(ccd: pd.DataFrame, *, use_flag: bool = True) -> pd.DataFrame:
    """Combine the CCDs of each FoV transit into one AL measurement.

    Only CCDs AGIS used (``used_by_agis_al``; NaN centroids are never used in
    the prerelease) with finite centroid, error and parallax factor enter.
    The transit value is the inverse-variance mean; its error is the formal
    combined error.  Within-transit CCD scatter is recorded (``chi2_ccd``) so
    a transit whose CCDs disagree can be refused downstream.
    """
    m = (np.isfinite(ccd["x_al"]) & np.isfinite(ccd["sx_al"]) & (ccd["sx_al"] > 0)
         & np.isfinite(ccd["pf_al"]) & np.isfinite(ccd["theta"]) & ccd["bary_ok"])
    if use_flag:
        m &= ccd["used"]
    c = ccd[m].copy()
    if not len(c):
        return pd.DataFrame(columns=ASTRO_COLUMNS + ["chi2_ccd"])
    c["w"] = 1.0 / c["sx_al"] ** 2
    c["wx"] = c["w"] * c["x_al"]
    g = c.groupby(["source_id", "transit_id"], sort=False)
    agg = g.agg(t=("t", "median"), theta_s=("theta", lambda a: np.mean(np.sin(a))),
                theta_c=("theta", lambda a: np.mean(np.cos(a))), pf_al=("pf_al", "first"),
                w=("w", "sum"), wx=("wx", "sum"), n_ccd=("x_al", "size"),
                excess_noise=("excess_noise", "first"), g_mag=("g_mag", "first"),
                dist_ci=("dist_ci", "median"), cf_al=("cf_al", "mean"),
                multipeak=("multipeak", "any"), blended=("blended", "any")).reset_index()
    agg["x_al"] = agg["wx"] / agg["w"]
    agg["sx_al"] = 1.0 / np.sqrt(agg["w"])
    agg["theta"] = np.arctan2(agg["theta_s"], agg["theta_c"])
    c = c.merge(agg[["source_id", "transit_id", "x_al"]].rename(columns={"x_al": "xm"}),
                on=["source_id", "transit_id"])
    c["r2"] = c["w"] * (c["x_al"] - c["xm"]) ** 2
    chi = c.groupby(["source_id", "transit_id"], sort=False)["r2"].sum().rename("chi2_ccd")
    agg = agg.merge(chi.reset_index(), on=["source_id", "transit_id"])
    agg["t_yr"] = (agg["t"] - J2017P5_DAYS) / 365.25
    return agg[ASTRO_COLUMNS + ["chi2_ccd", "multipeak", "blended"]].sort_values(
        ["source_id", "t"]).reset_index(drop=True)


def read_prerelease_zip(path) -> pd.DataFrame:
    """The ESA DR4 epoch-astrometry prerelease archive -> per-CCD frame."""
    import zipfile

    from astropy.io.votable import parse
    from astropy.table import Table

    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xml", ".vot", ".votable"))]
        if not names:
            raise FileNotFoundError(f"no VOTable member in {path}: {z.namelist()}")
        raw = z.read(names[0])
    tab = Table.read(io.BytesIO(raw), format="votable")
    release = ""
    try:
        vo = parse(io.BytesIO(raw))
        params = list(getattr(vo, "params", []))
        for res in vo.iter_resources() if hasattr(vo, "iter_resources") else vo.resources:
            params += list(res.params)
        for t in vo.iter_tables():
            params += list(t.params)
        for prm in params:
            if str(prm.name).lower() == "release":
                release = str(prm.value)
                break
    except Exception:  # noqa: BLE001
        pass
    ccd = astrometry_ccd_frame(tab)
    ccd.attrs["release"] = release
    return ccd


def join_photometry_astrometry(phot: pd.DataFrame, astro: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (source_id, transit_id).  The DR3/DR4 transit_id is the
    same identifier (verified on Gaia-4); a join key mismatch surfaces as an
    empty or thin result, which callers report as a coverage number."""
    a = astro.rename(columns={"t": "t_astro", "g_mag": "g_mag_astro"})
    j = phot.merge(a, on=["source_id", "transit_id"], how="inner")
    return j.sort_values(["source_id", "t"]).reset_index(drop=True)

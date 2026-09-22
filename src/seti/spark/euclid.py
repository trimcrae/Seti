"""S49 — the spark, resolved: Euclid Q1 NISP line features on Gaia stars.

The Q1 release (March 2025) ships ``euclid_q1_spe_lines_line_features`` at
IRSA: per source and per redshift-solution rank, every fitted line with its
observed central wavelength, flux, EW, FWHM, S/N, and — the column that makes
the slitless killers testable — ``spe_line_n_dith``, the number of dithers
the feature was measured in.  The line *name* in that table is the
pipeline's interpretation under the rank's redshift; this channel treats the
name as a descriptor and re-derives every identification from the observed
wavelength alone (``lines.py``), because a star with an unidentified feature
is exactly the object the pipeline would have labelled a galaxy at some z.

Layout
------
* schema discovery and role resolution (``*_ROLES``): no column is queried
  that ``TAP_SCHEMA.columns`` did not list;
* the chunked pull (``strip_adql`` / ``fetch_strip``): one declination strip
  of one Q1 field per query, the line table joined in-archive to the MER
  catalogue for positions and morphology, split adaptively when a strip
  truncates at ``maxrec``;
* the pure screen (``screen_features``): Gaia join, the veto ladder, the
  recurrence test, the trials correction.  Offline-tested end to end.
"""

from __future__ import annotations

import json
import math
import time as _time

import numpy as np
import pandas as pd

from . import lines as L
from .gaia import crossmatch, neighbours_within, propagate, sep_arcsec

# --------------------------------------------------------------------------
# Roles: which real column plays which part (resolved against TAP_SCHEMA)
# --------------------------------------------------------------------------
LINE_ROLES: dict[str, tuple[str, ...]] = {
    "object_id": ("object_id",),
    "rank": ("spe_rank",),
    "line_id": ("spe_line_id",),
    "line_name": ("spe_line_name",),
    "line_flag": ("spe_line_flag",),
    "n_dith": ("spe_line_n_dith",),
    "wl": ("spe_line_central_wl_gf", "spe_line_central_wl", "spe_line_central_wl_di"),
    "wl_err": ("spe_line_central_wl_err_gf", "spe_line_central_wl_err", "spe_line_central_wl_err_di"),
    "flux": ("spe_line_flux_gf", "spe_line_flux", "spe_line_flux_di"),
    "flux_err": ("spe_line_flux_err_gf", "spe_line_flux_err", "spe_line_flux_err_di"),
    "ew": ("spe_line_ew_gf", "spe_line_ew", "spe_line_ew_di"),
    "fwhm": ("spe_line_fwhm_gf", "spe_line_fwhm"),
    "fwhm_err": ("spe_line_fwhm_err_gf",),
    "snr": ("spe_line_snr_gf", "spe_line_snr", "spe_line_snr_di"),
    "snr_di": ("spe_line_snr_di",),
    "cont": ("spe_line_cont_gf", "spe_line_cont"),
    "qual": ("spe_line_qual_gf", "spe_line_qual"),
    "aon": ("spe_line_aon",),
}
LINE_REQUIRED = ("object_id", "wl", "snr")

MER_ROLES: dict[str, tuple[str, ...]] = {
    "object_id": ("object_id",),
    "ra": ("right_ascension", "ra", "ra_deg"),
    "dec": ("declination", "dec", "dec_deg"),
    "point_like_prob": ("point_like_prob",),
    "point_like_flag": ("point_like_flag",),
    "spurious_flag": ("spurious_flag",),
    "det_quality_flag": ("det_quality_flag",),
    "mumax_minus_mag": ("mumax_minus_mag",),
    "flux_h": ("flux_h_2fwhm_aper", "flux_h_templfit", "flux_h_1fwhm_aper", "flux_h_sersic"),
    "flux_vis": ("flux_vis_psf", "flux_vis_2fwhm_aper", "flux_vis_1fwhm_aper"),
    "segmentation_area": ("segmentation_area",),
}
MER_REQUIRED = ("object_id", "ra", "dec")

SPECTRA_ROLES: dict[str, tuple[str, ...]] = {
    "object_id": ("object_id",),
    "spe_class": ("spe_class", "spe_classification", "classification"),
    "spe_z": ("spe_z", "z"),
    "spe_z_prob": ("spe_z_prob", "z_prob"),
    "spe_quality": ("spe_quality", "quality"),
    "prob_star": ("spe_class_prob_star", "prob_star"),
    "prob_galaxy": ("spe_class_prob_galaxy", "prob_galaxy"),
}

STANDARD_COLS = ("object_id", "rank", "line_id", "line_name", "line_flag", "n_dith", "wl_um",
                 "wl_err_um", "flux", "flux_err", "ew", "fwhm_um", "snr", "snr_di", "cont",
                 "qual", "aon", "ra", "dec", "point_like_prob", "point_like_flag",
                 "spurious_flag", "det_quality_flag", "mumax_minus_mag", "flux_h", "flux_vis",
                 "segmentation_area", "field")

VETO_NAMES = ("low_snr", "few_dithers", "band_edge", "broad", "non_positive_flux",
              "stellar_line", "galaxy_pattern", "blend", "not_point_like", "spurious",
              "recurrent_wavelength")


def resolve_roles(columns, roles: dict[str, tuple[str, ...]]) -> dict[str, str | None]:
    """First candidate present in ``columns`` for every role (case-insensitive)."""
    have = {str(c).lower(): str(c) for c in columns}
    out: dict[str, str | None] = {}
    for role, cands in roles.items():
        out[role] = next((have[c.lower()] for c in cands if c.lower() in have), None)
    return out


def missing_required(roles: dict, required) -> list[str]:
    return [r for r in required if not roles.get(r)]


def schema_adql(table: str) -> str:
    return ("SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
            f"WHERE table_name = '{table}'")


def discover_columns(table: str, query_fn) -> tuple[list[str], dict]:
    """``TAP_SCHEMA.columns`` for one table; ``(names, record)``.  Never raises."""
    rec = {"table": table, "status": "QUERY_FAILED", "n_columns": 0}
    t0 = _time.monotonic()
    try:
        df = query_fn(schema_adql(table))
        if isinstance(df, tuple):
            df = df[0]
        df = pd.DataFrame(df)
        df.columns = [str(c).lower() for c in df.columns]
        names = [str(x) for x in df["column_name"].tolist()] if "column_name" in df else []
        rec.update(status="OK" if names else "QUERY_RETURNED_ZERO_ROWS", n_columns=len(names),
                   columns=names, seconds=round(_time.monotonic() - t0, 1))
        return names, rec
    except Exception as exc:                            # noqa: BLE001
        rec.update(error=repr(exc)[:400], seconds=round(_time.monotonic() - t0, 1))
        return [], rec


# --------------------------------------------------------------------------
# Wavelength scale
# --------------------------------------------------------------------------
def infer_wavelength_scale(values, declared: str = "auto") -> tuple[float, str]:
    """Factor that turns the table's wavelength column into microns, and its name.

    ``auto`` reads the value range: a median above 3000 is Ångström, between
    300 and 3000 nanometre, below 30 micron.  Recorded in the ledger, never
    assumed: Euclid documents Ångström, and a wrong factor would put every
    feature outside the band and yield a silent, disguised null.
    """
    d = str(declared or "auto").lower()
    table = {"angstrom": (1e-4, "angstrom"), "a": (1e-4, "angstrom"), "aa": (1e-4, "angstrom"),
             "nm": (1e-3, "nm"), "um": (1.0, "um"), "micron": (1.0, "um")}
    if d in table:
        return table[d]
    v = pd.to_numeric(pd.Series(np.atleast_1d(values)), errors="coerce").dropna()
    if not len(v):
        return 1e-4, "angstrom(assumed:no values)"
    med = float(v.median())
    if med > 3000:
        return 1e-4, "angstrom(inferred)"
    if med > 300:
        return 1e-3, "nm(inferred)"
    return 1.0, "um(inferred)"


# --------------------------------------------------------------------------
# Fields, strips and the ADQL
# --------------------------------------------------------------------------
def field_strips(field: dict, strip_deg: float) -> list[dict]:
    """Declination strips covering the field's cone, with the RA range at each strip."""
    ra0, dec0, r = float(field["ra"]), float(field["dec"]), float(field["radius_deg"])
    edges = np.arange(dec0 - r, dec0 + r + 1e-9, float(strip_deg))
    if edges[-1] < dec0 + r - 1e-9:
        edges = np.append(edges, dec0 + r)
    strips = []
    for k in range(len(edges) - 1):
        lo, hi = float(edges[k]), float(edges[k + 1])
        dmid = 0.5 * (lo + hi)
        # half-chord of the cone at this declination, widened by the strip's own height
        dd = min(abs(dmid - dec0) + 0.5 * (hi - lo), r)
        half = math.sqrt(max(r * r - dd * dd, 0.0)) if dd < r else 0.0
        half = min(max(half + 0.05, 0.05), r) / max(math.cos(math.radians(dmid)), 1e-3)
        strips.append({"k": k, "dec_lo": round(lo, 5), "dec_hi": round(hi, 5),
                       "ra_lo": round(ra0 - half, 5), "ra_hi": round(ra0 + half, 5)})
    return strips


def units_for_shard(fields: dict, strip_deg: float, shard: int = 0, n_shards: int = 1) -> list[dict]:
    """Every (field, strip) unit, round-robin assigned to shards."""
    units = []
    for name, f in fields.items():
        for s in field_strips(f, strip_deg):
            units.append({"field": name, **s, "ra0": float(f["ra"]), "dec0": float(f["dec"]),
                          "radius_deg": float(f["radius_deg"])})
    return [u for i, u in enumerate(units) if i % max(1, n_shards) == shard]


def _ra_clause(col: str, ra_lo: float, ra_hi: float) -> str:
    if ra_lo < 0 or ra_hi > 360:
        # the range crosses RA = 0: express it as a disjunction
        lo, hi = ra_lo % 360, ra_hi % 360
        return f"({col} >= {lo:.5f} OR {col} <= {hi:.5f})"
    return f"{col} BETWEEN {ra_lo:.5f} AND {ra_hi:.5f}"


def strip_adql(tables: dict, roles_line: dict, roles_mer: dict, unit: dict, *,
               snr_min: float, top: int | None = None) -> str:
    """Line features joined to MER for one declination strip."""
    lcols = [f"l.{c} AS l_{r}" for r, c in roles_line.items() if c]
    mcols = [f"m.{c} AS m_{r}" for r, c in roles_mer.items() if c and r != "object_id"]
    ra, dec = roles_mer["ra"], roles_mer["dec"]
    sel = ", ".join(lcols + mcols)
    top_s = f"TOP {int(top)} " if top else ""
    return (f"SELECT {top_s}{sel} FROM {tables['lines']} l JOIN {tables['mer']} m "
            f"ON l.{roles_line['object_id']} = m.{roles_mer['object_id']} "
            f"WHERE m.{dec} BETWEEN {unit['dec_lo']:.5f} AND {unit['dec_hi']:.5f} "
            f"AND {_ra_clause('m.' + ra, unit['ra_lo'], unit['ra_hi'])} "
            f"AND l.{roles_line['snr']} >= {float(snr_min):.2f}")


def denominator_adql(spectra_table: str, roles_spec: dict, tables: dict, roles_mer: dict,
                     unit: dict, *, top: int | None = None) -> str:
    """Every object WITH a spectrum in the strip (for the stars-with-spectra count)."""
    scols = [f"s.{c} AS s_{r}" for r, c in roles_spec.items() if c and r != "object_id"]
    mcols = [f"m.{c} AS m_{r}" for r, c in roles_mer.items()
             if c and r in ("ra", "dec", "point_like_prob", "spurious_flag", "flux_h")]
    ra, dec = roles_mer["ra"], roles_mer["dec"]
    top_s = f"TOP {int(top)} " if top else ""
    sel = ", ".join([f"m.{roles_mer['object_id']} AS m_object_id"] + mcols + scols)
    return (f"SELECT {top_s}{sel} FROM {spectra_table} s JOIN {tables['mer']} m "
            f"ON s.{roles_spec['object_id']} = m.{roles_mer['object_id']} "
            f"WHERE m.{dec} BETWEEN {unit['dec_lo']:.5f} AND {unit['dec_hi']:.5f} "
            f"AND {_ra_clause('m.' + ra, unit['ra_lo'], unit['ra_hi'])}")


def fetch_strip(adql_fn, unit: dict, query_fn, *, maxrec: int, label: str,
                depth: int = 0, max_depth: int = 4, log: list | None = None) -> tuple[pd.DataFrame, dict]:
    """Run ``adql_fn(unit)``; a strip that fills ``maxrec`` is split in two and retried.

    ``query_fn(adql, maxrec) -> DataFrame``.  Never raises; a failed sub-strip
    is recorded and the rest of the strip still returns.
    """
    adql = adql_fn(unit)
    rec = {"label": label, "unit": {k: unit[k] for k in ("field", "dec_lo", "dec_hi", "ra_lo", "ra_hi")},
           "status": "QUERY_FAILED", "n_rows": 0, "depth": depth}
    t0 = _time.monotonic()
    try:
        df = query_fn(adql, maxrec)
        if isinstance(df, tuple):
            df = df[0]
        df = pd.DataFrame(df)
        df.columns = [str(c).lower() for c in df.columns]
    except Exception as exc:                            # noqa: BLE001
        rec.update(error=repr(exc)[:500], seconds=round(_time.monotonic() - t0, 1), query=adql[:1500])
        if log is not None:
            log.append(rec)
        print(f"[spark] {label} {unit['field']} dec {unit['dec_lo']}..{unit['dec_hi']}: FAILED {exc!r}"[:300],
              flush=True)
        return pd.DataFrame(), rec
    n = int(len(df))
    rec.update(n_rows=n, seconds=round(_time.monotonic() - t0, 1),
               status="OK" if n else "QUERY_RETURNED_ZERO_ROWS", truncated=bool(maxrec and n >= maxrec))
    if maxrec and n >= maxrec and depth < max_depth:
        rec["status"] = "SPLIT"
        if log is not None:
            log.append(rec)
        mid = 0.5 * (unit["dec_lo"] + unit["dec_hi"])
        parts = []
        for lo, hi in ((unit["dec_lo"], mid), (mid, unit["dec_hi"])):
            sub = {**unit, "dec_lo": lo, "dec_hi": hi}
            d, _ = fetch_strip(adql_fn, sub, query_fn, maxrec=maxrec, label=label, depth=depth + 1,
                               max_depth=max_depth, log=log)
            parts.append(d)
        return pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame(), rec
    if log is not None:
        log.append(rec)
    return df, rec


# --------------------------------------------------------------------------
# Normalisation of the joined rows
# --------------------------------------------------------------------------
def normalise_features(raw: pd.DataFrame, field: str, wavelength_unit: str = "auto") -> tuple[pd.DataFrame, dict]:
    """Joined ``l_*`` / ``m_*`` rows → :data:`STANDARD_COLS` (wavelengths in µm)."""
    d = raw.copy()
    d.columns = [str(c).lower() for c in d.columns]
    out = pd.DataFrame(index=d.index)
    for role in ("object_id", "rank", "line_id", "line_name", "line_flag", "n_dith", "flux", "flux_err",
                 "ew", "snr", "snr_di", "cont", "qual", "aon"):
        out[role] = d[f"l_{role}"] if f"l_{role}" in d else np.nan
    scale, unit_name = infer_wavelength_scale(d["l_wl"] if "l_wl" in d else [], wavelength_unit)
    out["wl_um"] = pd.to_numeric(d["l_wl"], errors="coerce") * scale if "l_wl" in d else np.nan
    out["wl_err_um"] = pd.to_numeric(d["l_wl_err"], errors="coerce") * scale if "l_wl_err" in d else np.nan
    out["fwhm_um"] = pd.to_numeric(d["l_fwhm"], errors="coerce") * scale if "l_fwhm" in d else np.nan
    for role in ("ra", "dec", "point_like_prob", "point_like_flag", "spurious_flag", "det_quality_flag",
                 "mumax_minus_mag", "flux_h", "flux_vis", "segmentation_area"):
        out[role] = d[f"m_{role}"] if f"m_{role}" in d else np.nan
    out["field"] = field
    for c in ("snr", "snr_di", "flux", "flux_err", "ew", "cont", "n_dith", "rank", "point_like_prob",
              "ra", "dec", "spurious_flag", "det_quality_flag", "aon", "qual"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[list(STANDARD_COLS)]
    return out, {"wavelength_scale": scale, "wavelength_unit": unit_name, "n_rows": int(len(out))}


def resel_um(lam_um, R: float) -> np.ndarray:
    return np.asarray(lam_um, float) / float(R)


def dedupe_features(feat: pd.DataFrame, R: float) -> pd.DataFrame:
    """One row per (object, wavelength) across ranks: the highest-S/N fit wins.

    The pipeline lists the same feature under every redshift rank with a
    different name; the alternative names are kept in ``names`` and the
    number of ranks that carried it in ``n_ranks``.
    """
    if not len(feat):
        out = feat.copy()
        out["names"], out["n_ranks"] = [], []
        return out
    d = feat.sort_values(["object_id", "snr"], ascending=[True, False]).reset_index(drop=True)
    keep_rows, names, nranks = [], [], []
    for _oid, g in d.groupby("object_id", sort=False):
        kept: list[tuple[int, float, list, set]] = []
        for i, row in g.iterrows():
            wl = float(row["wl_um"])
            if not np.isfinite(wl):
                continue
            tol = wl / float(R)
            for k in kept:
                if abs(k[1] - wl) <= tol:
                    nm = str(row["line_name"]) if pd.notna(row["line_name"]) else ""
                    if nm and nm not in k[2]:
                        k[2].append(nm)
                    if pd.notna(row.get("rank")):
                        k[3].add(int(row["rank"]))
                    break
            else:
                nm = str(row["line_name"]) if pd.notna(row["line_name"]) else ""
                kept.append((i, wl, [nm] if nm else [],
                             {int(row["rank"])} if pd.notna(row.get("rank")) else set()))
        for i, _wl, nms, rk in kept:
            keep_rows.append(i)
            names.append("|".join(nms))
            nranks.append(len(rk))
    out = d.loc[keep_rows].copy()
    out["names"] = names
    out["n_ranks"] = nranks
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------
def _p_single(snr: float) -> float:
    from scipy.stats import norm
    return float(norm.sf(float(snr))) if np.isfinite(snr) else 1.0


def n_resels(band_um, R: float) -> int:
    lo, hi = float(band_um[0]), float(band_um[1])
    return int(math.ceil(float(R) * math.log(hi / lo)))


def screen_features(feat: pd.DataFrame, gaia: pd.DataFrame, conf: dict, *,
                    n_stars_with_spectra: int | None = None,
                    gaia_field: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """The S49 veto ladder over deduplicated features.  Pure.

    ``gaia`` is the parallax-selected star list (``ra, dec, pmra, pmdec,
    parallax_over_error, phot_g_mean_mag, ruwe, ...``); ``gaia_field`` (all
    Gaia sources, to the neighbour depth) drives the blend test and defaults
    to ``gaia``.  Returns ``(screened, funnel)``; ``screened`` carries one
    boolean column per veto in :data:`VETO_NAMES`, ``vetoes`` (the names that
    fired), ``survivor``, the descriptors and the trials correction.
    """
    c = conf
    R = float(c.get("resolving_power", 450.0))
    band = c.get("band_um", [1.21, 1.89])
    edge = float(c.get("edge_margin_um", 0.02))
    funnel: dict = {"n_features_in": int(len(feat)), "n_objects_in": int(feat["object_id"].nunique()) if len(feat) else 0}
    if gaia_field is None:
        gaia_field = gaia

    d = dedupe_features(feat, R)
    funnel["n_features_deduped"] = int(len(d))
    if not len(d):
        d = d.reindex(columns=list(d.columns) + list(VETO_NAMES) + ["gaia_source_id", "survivor"])
        funnel.update(n_features_on_stars=0, n_stars_with_features=0, n_survivors=0,
                      veto_counts={v: 0 for v in VETO_NAMES})
        return d, funnel

    # ---- Gaia join (positions propagated to the Euclid epoch) ----
    g = propagate(gaia, float(c.get("gaia_epoch", 2016.0)), float(c.get("euclid_epoch", 2024.3))) if len(gaia) else gaia
    objs = d.drop_duplicates("object_id")[["object_id", "ra", "dec"]].reset_index(drop=True)
    if len(g):
        idx, sep = crossmatch(objs["ra"], objs["dec"], g["ra_ep"], g["dec_ep"],
                              float(c.get("gaia_match_arcsec", 1.0)))
    else:
        idx, sep = np.full(len(objs), -1), np.full(len(objs), np.inf)
    objs["gaia_idx"] = idx
    objs["gaia_sep_arcsec"] = sep
    d = d.merge(objs[["object_id", "gaia_idx", "gaia_sep_arcsec"]], on="object_id", how="left")
    on_star = d["gaia_idx"] >= 0
    funnel["n_features_on_stars"] = int(on_star.sum())
    funnel["n_features_not_on_gaia_star"] = int((~on_star).sum())
    d = d[on_star].copy().reset_index(drop=True)
    gi = d["gaia_idx"].to_numpy(int)
    for col, name in (("source_id", "gaia_source_id"), ("phot_g_mean_mag", "gaia_g"), ("bp_rp", "gaia_bp_rp"),
                      ("parallax", "gaia_parallax"), ("parallax_over_error", "gaia_plx_over_err"),
                      ("ruwe", "gaia_ruwe"), ("pmra", "gaia_pmra"), ("pmdec", "gaia_pmdec"),
                      ("phot_variable_flag", "gaia_var_flag")):
        d[name] = g[col].to_numpy()[gi] if (len(g) and col in g) else np.nan
    funnel["n_stars_with_features"] = int(d["gaia_source_id"].nunique()) if len(d) else 0

    # ---- per-feature vetoes ----
    res = resel_um(d["wl_um"], R)
    d["resel_um"] = res
    d["low_snr"] = ~(d["snr"] >= float(c.get("snr_min", 5.0)))
    if d["n_dith"].notna().any():
        d["few_dithers"] = ~(d["n_dith"] >= int(c.get("n_dith_min", 2)))
        funnel["n_dith_testable"] = True
    else:
        d["few_dithers"] = False
        funnel["n_dith_testable"] = False
    d["band_edge"] = (d["wl_um"] < float(band[0]) + edge) | (d["wl_um"] > float(band[1]) - edge)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = d["fwhm_um"] / res
    d["fwhm_resel"] = ratio
    d["broad"] = ratio.notna() & (ratio > float(c.get("max_fwhm_resel", 2.5)))
    d["non_positive_flux"] = d["flux"].notna() & ~(d["flux"] > 0)
    tol = float(c.get("stellar_tol_resel", 1.5)) * res
    # discrete stellar lines veto; the broad molecular bands are a descriptor here
    # (the FWHM test is what separates an unresolved feature from a band at R ~ 450)
    sl = [L.stellar_line_match(w, t, include_bands=False) for w, t in zip(d["wl_um"], tol, strict=True)]
    d["stellar_line"] = [m is not None for m in sl]
    d["stellar_line_name"] = [m.name if m else "" for m in sl]
    sb = [L.stellar_line_match(w, t) for w, t in zip(d["wl_um"], tol, strict=True)]
    d["stellar_band_name"] = [m.name if (m and m.kind == "band") else "" for m in sb]
    d["industrial_flag"] = [L.industrial_flag(w, t) or "" for w, t in zip(d["wl_um"], tol, strict=True)]
    d["single_line_z"] = [json.dumps(L.single_line_interpretations(w)) for w in d["wl_um"]]

    # galaxy pattern: every deduped feature of the object above redshift_min_snr
    zmin_snr = float(c.get("redshift_min_snr", 3.0))
    pat: dict = {}
    for oid, grp in d.groupby("object_id"):
        lam = grp.loc[grp["snr"] >= zmin_snr, "wl_um"].tolist()
        pat[oid] = L.redshift_pattern(lam, z_tol=float(c.get("redshift_z_tol", 0.003)))
    d["galaxy_pattern"] = [bool(pat[o]["vetoed"]) for o in d["object_id"]]
    d["galaxy_z"] = [pat[o]["z"] if pat[o]["vetoed"] else np.nan for o in d["object_id"]]
    d["galaxy_ids"] = [json.dumps(pat[o]["ids"]) if pat[o]["vetoed"] else "" for o in d["object_id"]]

    # blend: a Gaia neighbour within blend_radius brighter than G + dmag
    br = float(c.get("blend_radius_arcsec", 6.0))
    bdm = float(c.get("blend_dmag", 2.0))
    d["blend"] = False
    d["n_neighbours_blend_radius"] = 0
    d["n_dispersion_neighbours"] = 0
    if len(gaia_field):
        gf = propagate(gaia_field, float(c.get("gaia_epoch", 2016.0)), float(c.get("euclid_epoch", 2024.3)))
        stars = d.drop_duplicates("gaia_source_id")[["gaia_source_id", "ra", "dec", "gaia_g"]].reset_index(drop=True)
        nb = neighbours_within(stars["ra"], stars["dec"], gf["ra_ep"], gf["dec_ep"], br)
        dr = float(c.get("dispersion_neighbour_radius_arcsec", 140.0))
        dh = float(c.get("dispersion_neighbour_halfwidth_arcsec", 3.0))
        nd = neighbours_within(stars["ra"], stars["dec"], gf["ra_ep"], gf["dec_ep"], dr)
        gg = pd.to_numeric(gf["phot_g_mean_mag"], errors="coerce").to_numpy(float)
        gid = gf["source_id"].to_numpy() if "source_id" in gf else None
        blend_map, nbl_map, ndisp_map = {}, {}, {}
        for k in range(len(stars)):
            sid = stars["gaia_source_id"].iloc[k]
            g0 = float(stars["gaia_g"].iloc[k]) if pd.notna(stars["gaia_g"].iloc[k]) else np.nan
            js = [j for j in nb[k] if gid is None or gid[j] != sid]
            if gid is None:
                seps = sep_arcsec(stars["ra"].iloc[k], stars["dec"].iloc[k], gf["ra_ep"].to_numpy()[js], gf["dec_ep"].to_numpy()[js]) if js else np.array([])
                js = [j for j, s in zip(js, seps, strict=True) if s > 1e-3]
            nbl_map[sid] = len(js)
            blend_map[sid] = bool(js) and bool(np.nanmin(gg[js] - g0) < bdm) if np.isfinite(g0) else bool(js)
            # along-dispersion (assumed ~E-W for RGS000/RGS180): |Δdec| <= halfwidth, within dr
            jd = [j for j in nd[k] if (gid is None or gid[j] != sid)]
            if jd:
                ddec = np.abs(gf["dec_ep"].to_numpy()[jd] - float(stars["dec"].iloc[k])) * 3600.0
                jd = [j for j, dd in zip(jd, ddec, strict=True) if dd <= dh and j not in js]
            ndisp_map[sid] = len(jd)
        d["blend"] = d["gaia_source_id"].map(blend_map).fillna(False).astype(bool)
        d["n_neighbours_blend_radius"] = d["gaia_source_id"].map(nbl_map).fillna(0).astype(int)
        d["n_dispersion_neighbours"] = d["gaia_source_id"].map(ndisp_map).fillna(0).astype(int)

    plmin = float(c.get("point_like_prob_min", 0.5))
    npl = d["point_like_prob"].notna() & (d["point_like_prob"] < plmin)
    d["not_point_like"] = npl if bool(c.get("require_point_like", False)) else False
    d["point_like_below_min"] = npl
    d["spurious"] = d["spurious_flag"].notna() & (d["spurious_flag"] != 0)

    # recurrence over every star feature that passes the S/N floor
    bin_um = float(c.get("recurrence_bin_resel", 1.0))
    minst = int(c.get("recurrence_min_stars", 3))
    d["recurrent_wavelength"] = False
    d["recurrence_n_stars"] = 0
    strong = d[~d["low_snr"]]
    if len(strong):
        b = np.floor(np.log(strong["wl_um"].to_numpy(float)) * R / bin_um).astype(int)
        tmp = pd.DataFrame({"bin": b, "sid": strong["gaia_source_id"].to_numpy()})
        counts = tmp.groupby("bin")["sid"].nunique()
        allb = np.floor(np.log(d["wl_um"].to_numpy(float)) * R / bin_um).astype(int)
        nrec = np.array([int(counts.get(x, 0)) for x in allb])
        # neighbouring bins too (a feature straddling a bin edge)
        nrec_adj = np.array([int(max(counts.get(x - 1, 0), counts.get(x + 1, 0))) for x in allb])
        d["recurrence_n_stars"] = np.maximum(nrec, nrec_adj)
        d["recurrent_wavelength"] = d["recurrence_n_stars"] >= minst
        funnel["recurrent_bins"] = [{"wl_um": round(float(math.exp((k + 0.5) * bin_um / R)), 4), "n_stars": int(v)}
                                    for k, v in counts.items() if v >= minst]

    vet_cols = list(VETO_NAMES)
    d["vetoes"] = ["|".join(v for v in vet_cols if bool(row[v])) for _, row in d[vet_cols].iterrows()]
    d["survivor"] = d["vetoes"] == ""

    # ---- trials ----
    nres = n_resels(band, R)
    nstars = int(n_stars_with_spectra) if n_stars_with_spectra else funnel["n_stars_with_features"]
    ntrials = max(1, nstars * nres)
    d["p_single"] = [_p_single(s) for s in d["snr"]]
    d["p_global"] = np.minimum(1.0, d["p_single"] * ntrials)
    d["significant_after_trials"] = d["p_global"] < float(c.get("trials_alpha", 0.01))
    funnel.update(n_resels=nres, n_stars_for_trials=nstars, n_trials=ntrials,
                  n_survivors=int(d["survivor"].sum()),
                  n_survivors_significant=int((d["survivor"] & d["significant_after_trials"]).sum()),
                  veto_counts={v: int(d[v].sum()) for v in vet_cols},
                  veto_counts_among_snr_pass={v: int((d[v] & ~d["low_snr"]).sum()) for v in vet_cols},
                  n_pass_snr=int((~d["low_snr"]).sum()),
                  industrial_flag_counts={k: int(v) for k, v in d.loc[d["industrial_flag"] != "", "industrial_flag"].value_counts().items()})
    d = d.sort_values(["survivor", "snr"], ascending=[False, False]).reset_index(drop=True)
    return d, funnel


def stars_with_spectra(spec_objs: pd.DataFrame, gaia: pd.DataFrame, conf: dict) -> tuple[int, pd.DataFrame]:
    """How many parallax stars have a NISP spectrum (the trials denominator)."""
    if not len(spec_objs) or not len(gaia):
        return 0, pd.DataFrame()
    g = propagate(gaia, float(conf.get("gaia_epoch", 2016.0)), float(conf.get("euclid_epoch", 2024.3)))
    s = spec_objs.copy()
    s.columns = [str(c).lower() for c in s.columns]
    ra = s["m_ra"] if "m_ra" in s else s.get("ra")
    dec = s["m_dec"] if "m_dec" in s else s.get("dec")
    idx, sep = crossmatch(pd.to_numeric(ra, errors="coerce"), pd.to_numeric(dec, errors="coerce"),
                          g["ra_ep"], g["dec_ep"], float(conf.get("gaia_match_arcsec", 1.0)))
    ok = idx >= 0
    matched = s[ok].copy()
    matched["gaia_source_id"] = g["source_id"].to_numpy()[idx[ok]] if "source_id" in g else idx[ok]
    matched["gaia_sep_arcsec"] = sep[ok]
    return int(matched["gaia_source_id"].nunique()), matched

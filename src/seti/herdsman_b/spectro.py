"""HERDSMAN-B v2 spectroscopic crossmatch: GALAH DR3 (+ APOGEE DR17).

The v1 GSP-Phot census produced exactly one formal candidate (Hogg_4) and
killed it in vetting on a magnitude-correlated systematic — photometric
metallicities cannot separate a real field-sampled [Fe/H] spread from
extinction systematics beyond ~2.5 kpc.  Spectroscopic [Fe/H] can: GALAH
DR3 (Buder et al. 2021) and APOGEE DR17 measure iron from resolved lines at
0.05-0.1 dex per star, so a gathered population's field-like spread survives
while extinction-driven fake spreads do not.

Acquisition follows the ``fetch_membership`` pattern (acquire.py): TAPVizieR,
candidate table names probed with TOP 1, columns resolved dynamically and
case-insensitively, per-chunk parquet checkpoints, retries with backoff.
Only rows with clean quality flags (flag_sp == 0 and flag_fe_h == 0 for
GALAH; ASPCAPflag == 0 for APOGEE) are used.

Scoring reuses the self-calibrating machinery of score.py: robust MAD-std
spread, error floor from e_fe_h + a co-natal floor, census-relative z among
comparable-N clusters (``census_z``), two-population gap (``two_pop_split``),
and a spectro-field baseline built from the survey's own non-member stars
(binned by R_gal when the survey pull carries geometry, global otherwise).
Surveys are scored separately — GALAH and APOGEE [Fe/H] zero-points differ
by up to ~0.1 dex, and mixing them inside one cluster would manufacture the
very spread this audit hunts for.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .score import (
    _field_spread_at,
    _mad_std,
    build_field_spread,
    census_z,
    radec_to_rgal,
    two_pop_split,
)

_VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

# (survey, VizieR catalog, preferred-table-name hints in priority order).
# Table names under each catalog are discovered at runtime from tap_schema
# and probed with TOP 1 — the hints only order the probe, so VizieR naming
# drift cannot break the pull.  GALAH DR3 = Buder et al. 2021.  APOGEE DR17
# mirrors are probed opportunistically; a clean resolution failure logs and
# the run proceeds GALAH-only.
GALAH_CATALOGS = ("J/MNRAS/506/150",)
APOGEE_CATALOGS = ("III/286", "III/284")
_TABLE_HINTS = ("galah", "allstar", "catalog", "main", "stars", "table")

# Column candidates, compared in *normalized* space: lowercase with all
# non-alphanumerics stripped, so "[Fe/H]", "__Fe_H_" and "Fe_H" all read
# "feh", and "GaiaEDR3"/"gaia_edr3" read "gaiaedr3".
_COLSPEC_REQUIRED = {
    "source_id": ("gaiaedr3", "gaiadr3", "dr3name", "gaiaid", "sourceid",
                  "gaia", "edr3id", "dr3sourceid"),
    "fe_h": ("feh",),
    "e_fe_h": ("efeh",),
}
_COLSPEC_OPTIONAL = {
    "flag_sp": ("flagsp", "aspcapflag", "starflag", "qflag"),
    "flag_fe_h": ("flagfeh", "ffeh"),
    "ra": ("raj2000", "radeg", "raicrs", "ra"),
    "dec": ("dej2000", "dedeg", "deicrs", "dec"),
    "plx": ("plx", "parallax"),
}

SPECTRO_QUALITY = {"prob_min": 0.7, "n_min": 6, "e_fe_h_max": 0.3}
SPECTRO_CANDIDATE = {"z_min": 4.0, "x_min": 2.0, "field_likeness_min": 0.5}
SPECTRO_FLOOR_DEX = 0.03
_REPORT_MARKER = "## Spectroscopic crossmatch (v2)"


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_spectro_columns(colnames) -> dict | None:
    """Map heterogeneous VizieR column names onto the standard schema.

    Exact normalized match wins; a prefix match is the fallback.  Returns
    None when any *required* column (source_id, fe_h, e_fe_h) is missing.
    """
    normed = {}
    for c in colnames:
        normed.setdefault(_norm(c), c)
    out, used = {}, set()

    def pick(cands):
        for cand in cands:                       # pass 1: exact
            hit = normed.get(cand)
            if hit is not None and hit not in used:
                return hit
        for cand in cands:                       # pass 2: prefix
            for key, orig in normed.items():
                if key.startswith(cand) and orig not in used:
                    return orig
        return None

    for std, cands in _COLSPEC_REQUIRED.items():
        hit = pick(cands)
        if hit is None:
            return None
        out[std] = hit
        used.add(hit)
    for std, cands in _COLSPEC_OPTIONAL.items():
        hit = pick(cands)
        if hit is not None:
            out[std] = hit
            used.add(hit)
    return out


def _list_tables(tap, catalog: str) -> list[str]:
    """Tables under a VizieR catalog, hint-ordered; falls back to guesses."""
    names = []
    try:
        res = tap.run_sync(
            "SELECT table_name FROM tap_schema.tables "
            f"WHERE table_name LIKE '{catalog}/%'").to_table()
        names = [str(x).strip('"') for x in res["table_name"]]
    except Exception as exc:  # noqa: BLE001
        print(f"[herdsman-b spectro] tap_schema listing failed for "
              f"{catalog}: {exc!r}")
    if not names:
        names = [f"{catalog}/{suffix}" for suffix in
                 ("galah3", "catalog", "allstar", "table1", "stars")]

    def rank(n: str) -> int:
        low = n.lower()
        for i, h in enumerate(_TABLE_HINTS):
            if h in low:
                return i
        return len(_TABLE_HINTS)

    return sorted(names, key=rank)


def _resolve_table(tap, catalogs) -> tuple[str, dict] | None:
    """First (quoted table name, column map) that probes + resolves."""
    for catalog in catalogs:
        for name in _list_tables(tap, catalog):
            table = f'"{name}"'
            try:
                probe = tap.run_sync(f"SELECT TOP 1 * FROM {table}").to_table()
            except Exception as exc:  # noqa: BLE001
                print(f"[herdsman-b spectro] probe failed for {table}: {exc!r}")
                continue
            cols = resolve_spectro_columns(probe.colnames)
            if cols is None:
                print(f"[herdsman-b spectro] {table}: required columns "
                      f"unresolved in {list(probe.colnames)}")
                continue
            print(f"[herdsman-b spectro] resolved {table} -> {cols}")
            return table, cols
    return None


def fetch_spectro_catalog(survey: str, catalogs, out_path: Path,
                          dec_step: float = 15.0,
                          retries: int = 4) -> pd.DataFrame | None:
    """Pull a spectroscopic catalog via TAPVizieR (checkpointed; resumable).

    Chunks by declination band (each band its own parquet checkpoint) so a
    killed job resumes at the band where it died; a missing dec column falls
    back to a single whole-table pull.  Returns the standardized-name raw
    table (quality flags NOT yet applied — that is ``standardize_spectro``,
    kept pure so it is testable offline), or None when no candidate table
    resolves (caller decides whether that is fatal).
    """
    if out_path.exists():
        print(f"[herdsman-b spectro] {survey} checkpoint exists: {out_path}")
        return pd.read_parquet(out_path)
    import pyvo

    tap = pyvo.dal.TAPService(_VIZIER_TAP)
    resolved = _resolve_table(tap, catalogs)
    if resolved is None:
        print(f"[herdsman-b spectro] {survey}: no table resolved "
              f"in {catalogs}")
        return None
    table, cols = resolved
    select = ", ".join(f'"{v}"' for v in cols.values())

    def run_query(q: str) -> pd.DataFrame:
        last = None
        for attempt in range(retries):
            try:
                df = tap.run_async(q).to_table().to_pandas()
                df.columns = list(cols.keys())
                return df
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"[herdsman-b spectro] {survey} query attempt "
                      f"{attempt + 1}/{retries} failed: {exc!r}")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{survey} query failed: {last!r}")

    chunk_dir = out_path.parent / f"{survey}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    if "dec" in cols:
        edges = np.arange(-90.0, 90.0 + dec_step, dec_step)
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
            part = chunk_dir / f"{survey}_chunk_{k:02d}.parquet"
            if part.exists():
                frames.append(pd.read_parquet(part))
                continue
            df = run_query(f'SELECT {select} FROM {table} '
                           f'WHERE "{cols["dec"]}" >= {lo} '
                           f'AND "{cols["dec"]}" < {hi}')
            df.to_parquet(part, index=False)               # checkpoint
            frames.append(df)
            print(f"[herdsman-b spectro] {survey} band {k + 1}/"
                  f"{len(edges) - 1} [{lo:+.0f},{hi:+.0f}): {len(df)} rows")
    else:
        frames.append(run_query(f"SELECT {select} FROM {table}"))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(out_path, index=False)
    print(f"[herdsman-b spectro] {survey}: {len(raw)} raw rows -> {out_path}")
    return raw


def standardize_spectro(raw: pd.DataFrame) -> pd.DataFrame:
    """Quality-filter + type a standardized-name raw pull (pure, offline).

    Keeps only rows with a valid Gaia source_id, finite fe_h, finite
    e_fe_h below ``e_fe_h_max``, and every present quality flag exactly 0
    (GALAH flag_sp / flag_fe_h; APOGEE ASPCAPflag lands in flag_sp).
    Attaches r_gal + dist_kpc when the pull carries ra/dec/plx geometry.
    """
    sid = pd.to_numeric(raw["source_id"], errors="coerce")
    # VizieR sometimes serves the id as a designation string; salvage digits.
    if sid.isna().mean() > 0.5:
        sid = pd.to_numeric(
            raw["source_id"].astype(str).str.extract(r"(\d{15,})")[0],
            errors="coerce")
    fe_h = pd.to_numeric(raw["fe_h"], errors="coerce")
    e_fe_h = pd.to_numeric(raw["e_fe_h"], errors="coerce")
    keep = (sid.notna() & np.isfinite(fe_h) & np.isfinite(e_fe_h)
            & (e_fe_h > 0) & (e_fe_h <= SPECTRO_QUALITY["e_fe_h_max"]))
    for flag_col in ("flag_sp", "flag_fe_h"):
        if flag_col in raw.columns:
            flag = pd.to_numeric(raw[flag_col], errors="coerce")
            keep &= flag.notna() & (flag == 0)
    out = pd.DataFrame({"source_id": sid[keep].astype(np.int64),
                        "fe_h": fe_h[keep], "e_fe_h": e_fe_h[keep]})
    r_gal = np.full(len(out), np.nan)
    dist = np.full(len(out), np.nan)
    if {"ra", "dec", "plx"} <= set(raw.columns):
        ra = pd.to_numeric(raw["ra"], errors="coerce")[keep].to_numpy(float)
        dec = pd.to_numeric(raw["dec"], errors="coerce")[keep].to_numpy(float)
        plx = pd.to_numeric(raw["plx"], errors="coerce")[keep].to_numpy(float)
        good = np.isfinite(plx) & (plx > 0.05)
        dist[good] = 1.0 / plx[good]
        geo = good & np.isfinite(ra) & np.isfinite(dec)
        r_gal[geo] = radec_to_rgal(ra[geo], dec[geo], dist[geo])
    out["r_gal"] = r_gal
    out["dist_kpc"] = dist
    return out.drop_duplicates("source_id", keep="first") \
        .reset_index(drop=True)


def _spectro_field_spread(field_fe_h: np.ndarray, field_r_gal: np.ndarray):
    """(FieldSpread | None, global MAD-std) baseline from non-member stars."""
    s_global = _mad_std(field_fe_h)
    geo = np.isfinite(field_r_gal)
    if geo.sum() >= 1000:
        fs = build_field_spread(pd.DataFrame(
            {"mh": field_fe_h[geo], "r_gal": field_r_gal[geo]}))
        return fs, s_global
    return None, s_global


def score_spectro_census(members: pd.DataFrame, spectro: pd.DataFrame,
                         survey: str = "galah") -> pd.DataFrame:
    """Score every cluster with enough spectroscopic members (pure, offline).

    ``members``: source_id, cluster, prob (the members.parquet checkpoint).
    ``spectro``: standardized quality rows — source_id, fe_h, e_fe_h,
    r_gal, dist_kpc (geometry may be all-NaN).
    """
    q = SPECTRO_QUALITY
    mem = members[pd.to_numeric(members["prob"], errors="coerce")
                  >= q["prob_min"]]
    joined = mem.merge(spectro, on="source_id", how="inner")

    # Spectro-field baseline: every quality row not in ANY cluster (at any
    # membership probability, not just prob >= 0.7).
    in_cluster = spectro["source_id"].isin(members["source_id"])
    fld = spectro[~in_cluster]
    fs, s_global = _spectro_field_spread(fld["fe_h"].to_numpy(float),
                                         fld["r_gal"].to_numpy(float))

    rows = []
    for name, g in joined.groupby("cluster"):
        if len(g) < q["n_min"]:
            continue
        fe_h = g["fe_h"].to_numpy(float)
        e_fe = g["e_fe_h"].to_numpy(float)
        resid = fe_h - np.median(fe_h)
        s = _mad_std(resid)
        e_c = float(np.sqrt(np.median(e_fe) ** 2 + SPECTRO_FLOOR_DEX ** 2))
        gap, two_pop = two_pop_split(resid)
        rg = g["r_gal"].to_numpy(float)
        dk = g["dist_kpc"].to_numpy(float)
        r_gal = float(np.nanmedian(rg)) if np.isfinite(rg).any() else np.nan
        dist = float(np.nanmedian(dk)) if np.isfinite(dk).any() else np.nan
        if fs is not None and np.isfinite(r_gal):
            s_field, _ = _field_spread_at(fs, r_gal)
        else:
            s_field = float(s_global)
        fl = float(min(s, s_field) / max(s, s_field)) \
            if s > 0 and s_field > 0 else np.nan
        rows.append({"cluster": name, "survey": survey,
                     "n_spectro": int(len(g)),
                     "s_spectro": s, "err_floor": e_c,
                     "x_spectro": float(s / e_c) if e_c > 0 else np.nan,
                     "two_pop": two_pop, "gap": gap,
                     "fe_h_median": float(np.median(fe_h)),
                     "r_gal_kpc": r_gal, "dist_kpc": dist,
                     "s_field": float(s_field), "field_likeness": fl})
    tab = pd.DataFrame(rows)
    if not len(tab):
        return tab
    tab["z_census"] = census_z(tab["x_spectro"].to_numpy(float),
                               tab["n_spectro"].to_numpy(float))
    c = SPECTRO_CANDIDATE
    tab["spectro_candidate"] = ((tab["z_census"] >= c["z_min"])
                                & (tab["x_spectro"] >= c["x_min"])
                                & (~tab["two_pop"])
                                & (tab["field_likeness"]
                                   >= c["field_likeness_min"]))
    return tab.sort_values("z_census", ascending=False).reset_index(drop=True)


def _append_report(report_path: Path, section: str) -> None:
    """Replace-or-append the spectro section of REPORT.md (idempotent)."""
    text = report_path.read_text() if report_path.exists() else \
        "# HERDSMAN-B run report\n"
    if _REPORT_MARKER in text:
        text = text[:text.index(_REPORT_MARKER)].rstrip() + "\n"
    report_path.write_text(text.rstrip() + "\n\n" + section.strip() + "\n")


def spectro_write(cfg: Config, tab: pd.DataFrame, joined_members: pd.DataFrame,
                  surveys_used: list[str], dump_top: int = 25) -> dict:
    """Write spectro_scores.csv, spectro_candidates.json, REPORT section."""
    out_dir = cfg.root / "results" / "herdsman_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_dir / "spectro_scores.csv", index=False)

    cands = tab[tab["spectro_candidate"]] if len(tab) else tab
    dumps = []
    for _, row in (cands.head(dump_top).iterrows() if len(cands) else []):
        g = joined_members[(joined_members["cluster"] == row["cluster"])
                           & (joined_members["survey"] == row["survey"])]
        dumps.append({"cluster": str(row["cluster"]),
                      "survey": str(row["survey"]),
                      "members": [{"source_id": int(r.source_id),
                                   "fe_h": float(r.fe_h),
                                   "e_fe_h": float(r.e_fe_h),
                                   "prob": float(r.prob)}
                                  for r in g.itertuples()]})
    (out_dir / "spectro_candidates.json").write_text(json.dumps({
        "n_clusters_spectro": int(len(tab)),
        "n_spectro_candidates": int(cands.shape[0]) if len(tab) else 0,
        "surveys": surveys_used,
        "candidates": (cands.to_dict("records") if len(tab) else []),
        "member_dumps": dumps}, indent=2, default=str))

    n_cand = int(cands.shape[0]) if len(tab) else 0
    lines = [
        _REPORT_MARKER, "",
        f"Surveys: {', '.join(surveys_used) or 'none'}. Clusters with >= "
        f"{SPECTRO_QUALITY['n_min']} flag-clean spectroscopic members at "
        f"prob >= {SPECTRO_QUALITY['prob_min']}: {len(tab)}; spectro "
        f"candidates: {n_cand}.", "",
        "Spectroscopic [Fe/H] (GALAH flag_sp = flag_fe_h = 0; APOGEE "
        "ASPCAPflag = 0)", "is immune to the GSP-Phot extinction/magnitude "
        "systematic that killed the", "v1 photometric candidate; a spectro "
        "candidate has census-z >= 4, spread", ">= 2x its error floor, is "
        "unimodal, and mirrors the survey's own", "non-member field spread. "
        "Surveys are scored separately (zero-point", "offsets between GALAH "
        "and APOGEE would otherwise fake a spread).", "",
        "Top by census z:", "",
    ]
    for r in (tab.head(10).to_dict("records") if len(tab) else []):
        lines.append(f"- {r['cluster']} [{r['survey']}]: n={r['n_spectro']}, "
                     f"s={r['s_spectro']:.3f}, x={r['x_spectro']:.2f}, "
                     f"z={r['z_census']:.1f}, "
                     f"field_likeness={r['field_likeness']:.2f}, "
                     f"two_pop={r['two_pop']}, "
                     f"candidate={r['spectro_candidate']}")
    _append_report(out_dir / "REPORT.md", "\n".join(lines))

    summary = {"n_clusters_spectro": int(len(tab)),
               "n_spectro_candidates": n_cand,
               "spectro_surveys": surveys_used}
    summary_path = out_dir / "summary.json"
    merged = {}
    if summary_path.exists():
        try:
            merged = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            merged = {}
    merged.update(summary)
    summary_path.write_text(json.dumps(merged, indent=2, default=str))
    print(f"[herdsman-b spectro] {len(tab)} clusters scored "
          f"({', '.join(surveys_used) or 'no surveys'}) -> {n_cand} candidates")
    return summary


def spectro_run(cfg: Config | None = None) -> dict:
    """Runner stage: fetch GALAH (+ APOGEE), join membership, score, write.

    Needs results/herdsman_b/members.parquet (the catalog stage checkpoint).
    GALAH failing to resolve is fatal only when APOGEE also fails; APOGEE is
    always best-effort (log and proceed GALAH-only).
    """
    cfg = cfg or load_config()
    out_dir = cfg.root / "results" / "herdsman_b"
    members_path = out_dir / "members.parquet"
    if not members_path.exists():
        raise RuntimeError(
            f"spectro stage needs the membership checkpoint {members_path}; "
            "run --stage catalog (or all) first")
    members = pd.read_parquet(members_path)

    pulls = []
    for survey, catalogs in (("galah", GALAH_CATALOGS),
                             ("apogee", APOGEE_CATALOGS)):
        try:
            raw = fetch_spectro_catalog(survey, catalogs,
                                        out_dir / f"{survey}.parquet")
        except Exception as exc:  # noqa: BLE001
            print(f"[herdsman-b spectro] {survey} fetch failed: {exc!r}")
            raw = None
        if raw is not None and len(raw):
            pulls.append((survey, raw))
    if not pulls:
        raise RuntimeError("no spectroscopic catalog could be fetched "
                           "(GALAH and APOGEE both failed)")

    tabs, joined_frames, surveys_used = [], [], []
    for survey, raw in pulls:
        std = standardize_spectro(raw)
        print(f"[herdsman-b spectro] {survey}: {len(std)} quality rows "
              f"(of {len(raw)})")
        if not len(std):
            continue
        surveys_used.append(survey)
        tabs.append(score_spectro_census(members, std, survey=survey))
        j = members[pd.to_numeric(members["prob"], errors="coerce")
                    >= SPECTRO_QUALITY["prob_min"]].merge(
            std, on="source_id", how="inner")
        j["survey"] = survey
        joined_frames.append(j)
    tab = pd.concat([t for t in tabs if len(t)], ignore_index=True) \
        if any(len(t) for t in tabs) else pd.DataFrame()
    if len(tab):
        tab = tab.sort_values("z_census",
                              ascending=False).reset_index(drop=True)
    joined = pd.concat(joined_frames, ignore_index=True) \
        if joined_frames else pd.DataFrame(
            columns=["cluster", "survey", "source_id", "fe_h", "e_fe_h",
                     "prob"])
    return spectro_write(cfg, tab, joined, surveys_used)


__all__ = ["spectro_run", "spectro_write", "score_spectro_census",
           "standardize_spectro", "fetch_spectro_catalog",
           "resolve_spectro_columns", "SPECTRO_QUALITY", "SPECTRO_CANDIDATE",
           "SPECTRO_FLOOR_DEX"]

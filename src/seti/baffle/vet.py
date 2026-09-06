"""BAFFLE ``vet`` stage: per-candidate archive checks that the screen cannot do.

Why this stage exists (run 34053752510, the first real screen)
----------------------------------------------------------------
135 two-band-deficit survivors, 107 of them at |b| < 10°, no two within 10′ of
each other, and ``neighbours_not_checked = 2055`` because no neighbour table
was ever supplied.  Many have a *flat* WISE SED (W1 ≈ W2 ≈ W3) sitting ~1 mag
below K_s — the AllWISE counterpart is a fainter object than the 2MASS/Gaia
star: a deblended fragment or a wrong counterpart in a crowded field, not a
shadow.  This stage asks the archives the questions that separate those cases,
in **~8 upload queries**, never per-star cones.

Deficit-candidate vet (``candidates.csv`` + ``deferred_lpv.csv``)
-----------------------------------------------------------------
1. **Gaia neighbours** — one Gaia-archive upload join within 10″: brightest
   neighbour's G and separation, counts within 3″/6″/10″.  Veto
   ``blend_flux_theft`` (a neighbour within 6″ with G < G_cand + 1.5); report
   ``crowded_field`` (≥ 3 Gaia sources within 6″).
2. **AllWISE proper** (VizieR ``II/328/allwise``, 3″ after propagation to
   2010.5): ``nb``/``na`` (blend components / active deblend), ``w1sat``/
   ``w2sat``, fluxes, ``w1rchi2``, ``cc_flags``, ``ph_qual``, W1–W4.  Veto
   ``deblended_component`` (nb > 1 or na > 0) and ``saturated_pixels``.
3. **CatWISE2020** (``II/365/catwise``, 3″ at 2015.5) and **unWISE**
   (``II/363/unwise``, 3″ at ~2014): independent W1/W2 with better deblending.
   Their K_s − W residuals are evaluated against the SAME locus
   (``locus.json``): ``catwise_photospheric`` (within 3σ of 0 → the AllWISE
   photometry was wrong, veto), ``catwise_confirms_deficit`` (both bands
   < −0.3 mag at > 3σ → survives), ``catwise_missing``.
4. **W3 consistency**: ``w3_excess`` (resid_w3 > +0.5, a blend / dusty
   neighbour signature → veto) and ``w3_deficit_consistent`` (< −0.3, note).
   The screen's ``resid_w3`` was NaN for every candidate because the Gaia
   mirror serves no ``w3snr``; the locus now falls back to ``w3mpro_error``
   and this stage re-evaluates W3 from the re-pulled AllWISE row.

``vet_verdict`` ∈ {``SURVIVES_VET``, ``BLEND``, ``DEBLENDED_COMPONENT``,
``ALLWISE_PHOTOMETRY_WRONG``, ``W3_INCONSISTENT``, ``INCONCLUSIVE``}.

Missing-track vet
-----------------
The Gaia × AllWISE cross-match table lacking an entry is **not** the same as
WISE having no source.  The screen's missing fraction by G already shows
39 % at G 4–5 and 14 % at |b| < 10°: that is the cross-match's behaviour on
saturated and crowded sources (the best-neighbour algorithm drops them), not
an absence of 3–5 µm light.  This stage measures the *real* absence rate by
a direct positional match of the ``nearby`` / ``etz`` candidates plus a
uniform random control of the rest: ``wise_source_present_within_6as``,
``wise_source_present_6_to_15as`` (astrometric offset, saturated bright
star), ``no_wise_source_within_15as`` — and for the last group CatWISE /
unWISE presence and the nearest AllWISE source's ``cc_flags``.

Transport
---------
Every fetcher is injectable and the offline tests inject all of them.  The
VizieR path reuses ``seti.baffle.bright``'s discover-then-quote-once machinery
(``discover_columns``, ``resolve_aliases``, ``select_list``, ``run_vizier``):
no column name reaches the wire that TAP_SCHEMA did not serve, and an empty
identifier raises before composition (the ``Encountered '""'`` failure).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bright import (
    VIZIER_TAP,
    _adql_col,
    _lower,
    discover_columns,
    resolve_aliases,
    run_vizier,
    select_list,
)
from .locus import Locus, luminosity_class

VET_VERDICTS = ("SURVIVES_VET", "BLEND", "DEBLENDED_COMPONENT", "ALLWISE_PHOTOMETRY_WRONG",
                "W3_INCONSISTENT", "INCONCLUSIVE")
VET_VETOES = ("blend_flux_theft", "deblended_component", "saturated_pixels",
              "catwise_photospheric", "w3_excess")
VET_NOTES = ("crowded_field", "catwise_confirms_deficit", "catwise_missing",
             "w3_deficit_consistent", "gaia_neighbours_unavailable", "allwise_unavailable",
             "allwise_missing", "catwise_unavailable")

QUERY_OK = "QUERY_OK"
QUERY_ZERO = "QUERY_RETURNED_ZERO_ROWS"
QUERY_FAILED = "QUERY_FAILED"

DEFAULTS: dict = {
    "gaia_neighbour_radius_arcsec": 10.0,
    "blend_radius_arcsec": 6.0,
    "blend_dg_max": 1.5,
    "crowded_radius_arcsec": 6.0,
    "crowded_n_min": 3,
    "epochs": {"gaia": 2016.0, "allwise": 2010.5, "catwise": 2015.5, "unwise": 2014.0},
    "allwise": {"table": '"II/328/allwise"', "radius_arcsec": 3.0,
                "xmatch_catalogue": "vizier:II/328/allwise"},
    "catwise": {"table": '"II/365/catwise"', "radius_arcsec": 3.0,
                "xmatch_catalogue": "vizier:II/365/catwise"},
    "unwise": {"table": '"II/363/unwise"', "radius_arcsec": 3.0,
               "xmatch_catalogue": "vizier:II/363/unwise"},
    "upload_chunk": 4000,
    "xmatch_url": "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync",
    "photospheric_nsig": 3.0,
    "deficit_mag": -0.30,
    "deficit_nsig": 3.0,
    "w3_excess_mag": 0.5,
    "w3_deficit_mag": -0.30,
    "w3_err_max": 0.2,
    "include_deferred_lpv": True,
    "missing": {"radius_close_arcsec": 6.0, "radius_far_arcsec": 15.0,
                "nearest_radius_arcsec": 60.0, "n_control": 1000, "seed": 20260906},
}

# --- VizieR column aliases (canonical -> candidates; resolved against TAP_SCHEMA) ---
ALLWISE_ALIASES = {
    "designation": ("AllWISE", "designation"),
    "ra": ("RAJ2000", "RA_pm", "ra"), "dec": ("DEJ2000", "DE_pm", "dec"),
    "w1": ("W1mag", "w1mpro"), "e_w1": ("e_W1mag", "w1sigmpro", "w1mpro_error"),
    "w2": ("W2mag", "w2mpro"), "e_w2": ("e_W2mag", "w2sigmpro", "w2mpro_error"),
    "w3": ("W3mag", "w3mpro"), "e_w3": ("e_W3mag", "w3sigmpro", "w3mpro_error"),
    "w4": ("W4mag", "w4mpro"), "e_w4": ("e_W4mag", "w4sigmpro", "w4mpro_error"),
    "cc_flags": ("ccf", "cc_flags"), "ph_qual": ("qph", "ph_qual"),
    "ext_flag": ("ex", "ext_flg", "ext_flag"), "var_flag": ("var", "var_flg"),
    "nb": ("nb",), "na": ("na",),
    "w1sat": ("W1sat", "w1sat"), "w2sat": ("W2sat", "w2sat"),
    "w1flux": ("W1flux", "w1flux"), "w2flux": ("W2flux", "w2flux"),
    "w1rchi2": ("W1rchi2", "chi2W1", "w1rchi2"), "w2rchi2": ("W2rchi2", "chi2W2", "w2rchi2"),
    "w1snr": ("W1snr", "snrW1", "w1snr"), "w3snr": ("W3snr", "snrW3", "w3snr"),
    "d2m": ("d2M", "d2m"),
}
CATWISE_ALIASES = {
    "designation": ("Name", "CatWISE", "designation", "source_name"),
    "ra": ("RA_ICRS", "RAPMdeg", "RAJ2000", "ra"), "dec": ("DE_ICRS", "DEPMdeg", "DEJ2000", "dec"),
    "w1": ("W1mproPM", "w1mpropm", "W1mpro", "w1mpro"),
    "e_w1": ("e_W1mproPM", "w1sigmpropm", "e_W1mpro", "w1sigmpro"),
    "w2": ("W2mproPM", "w2mpropm", "W2mpro", "w2mpro"),
    "e_w2": ("e_W2mproPM", "w2sigmpropm", "e_W2mpro", "w2sigmpro"),
    "cc_flags": ("ccf", "cc_flags"), "ab_flags": ("abf", "ab_flags"),
    "pmra": ("pmRA", "pmra"), "pmdec": ("pmDE", "pmdec"),
}
UNWISE_ALIASES = {
    "designation": ("objID", "unwise_objid", "designation"),
    "ra": ("RAJ2000", "ra", "RA_ICRS"), "dec": ("DEJ2000", "dec", "DE_ICRS"),
    "w1": ("W1mag", "w1mag", "mag_w1"), "e_w1": ("e_W1mag", "e_w1mag"),
    "w2": ("W2mag", "w2mag", "mag_w2"), "e_w2": ("e_W2mag", "e_w2mag"),
    "w1flux": ("FW1", "flux_w1", "fw1"), "e_w1flux": ("e_FW1", "dflux_w1"),
    "w2flux": ("FW2", "flux_w2", "fw2"), "e_w2flux": ("e_FW2", "dflux_w2"),
    "flags_w1": ("fW1", "flags_unwise_w1", "flags_w1"), "flags_w2": ("fW2", "flags_unwise_w2"),
}
# unWISE fluxes are AB nanomaggies: Vega = AB - offset (Jarrett+2011 / Schlafly+2019).
UNWISE_VEGA_OFFSET = {"w1": 2.699, "w2": 3.339}


def _cfg(cfg: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    src = (cfg or {}).get("vet", cfg) if isinstance(cfg, dict) else {}
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ===========================================================================
# Geometry
# ===========================================================================
def propagate(ra, dec, pmra, pmdec, from_epoch: float, to_epoch: float):
    """Move ICRS positions between epochs (pmra includes cos dec, as Gaia reports)."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    pmra = np.nan_to_num(np.asarray(pmra, dtype=float))
    pmdec = np.nan_to_num(np.asarray(pmdec, dtype=float))
    dt = float(to_epoch) - float(from_epoch)
    cosd = np.cos(np.radians(dec))
    cosd = np.where(np.abs(cosd) < 1e-6, 1e-6, cosd)
    return ra + (pmra * dt / 3.6e6) / cosd, dec + pmdec * dt / 3.6e6


def separation_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    ra1, dec1, ra2, dec2 = (np.radians(np.asarray(x, dtype=float)) for x in (ra1, dec1, ra2, dec2))
    s = (np.sin((dec2 - dec1) / 2) ** 2
         + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 3600.0


# ===========================================================================
# Query builders (pure)
# ===========================================================================
def gaia_neighbours_upload_query(radius_arcsec: float) -> str:
    """Gaia archive upload join: every gaia_source within ``radius`` of each target."""
    return (
        "SELECT u.source_id AS target_source_id, g.source_id, g.ra, g.dec, "
        "g.phot_g_mean_mag, g.bp_rp, g.parallax, g.pmra, g.pmdec, g.ruwe\n"
        "FROM tap_upload.targets AS u\n"
        "JOIN gaiadr3.gaia_source AS g ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
        f"CIRCLE('ICRS', u.ra, u.dec, {float(radius_arcsec) / 3600.0:.8f}))")


def vizier_upload_query(table: str, resolved: dict, radius_arcsec: float,
                        required=("ra", "dec")) -> str:
    """VizieR TAP upload join composed from discovered (bare) names, quoted once."""
    holes = [k for k in ("ra", "dec") if not resolved.get(k)]
    if holes:
        raise RuntimeError(f"{table}: position columns not resolved: {holes} (resolved: {resolved})")
    ra_c = _adql_col(resolved["ra"], "ra")
    de_c = _adql_col(resolved["dec"], "dec")
    select = select_list(resolved, required=tuple(required), prefix="t.")
    return (f"SELECT u.source_id, {select} FROM TAP_UPLOAD.targets AS u "
            f"JOIN {table} AS t ON 1 = CONTAINS(POINT('ICRS', t.{ra_c}, t.{de_c}), "
            f"CIRCLE('ICRS', u.ra, u.dec, {float(radius_arcsec) / 3600.0:.8f}))")


# ===========================================================================
# Fetchers (runner only; every one injectable)
# ===========================================================================
def default_gaia_upload_fetcher(positions: pd.DataFrame, radius_arcsec: float,
                                label: str = "gaia-neighbours", retries: int = 3) -> pd.DataFrame:
    """astroquery upload join (``upload_resource``, as ossuary's ``_run_query`` uses)."""
    from astropy.table import Table
    from astroquery.gaia import Gaia

    up = Table({"source_id": positions["source_id"].to_numpy(np.int64),
                "ra": positions["ra"].to_numpy(float), "dec": positions["dec"].to_numpy(float)})
    q = gaia_neighbours_upload_query(radius_arcsec)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            job = Gaia.launch_job_async(q, upload_resource=up, upload_table_name="targets")
            return _lower(job.get_results().to_pandas())
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            print(f"[baffle-vet] {label} attempt {attempt + 1}/{retries} failed: {exc!r}",
                  flush=True)
            time.sleep(4.0 * (attempt + 1))
    raise RuntimeError(f"{label}: Gaia upload join failed after {retries} attempts: {last!r}")


class UploadMatcher:
    """One VizieR catalogue around uploaded positions (TAP upload; X-Match fallback).

    ``__call__(positions, radius_arcsec, label)`` takes ``source_id, ra, dec``
    already at the catalogue epoch and returns canonical-named rows with the
    uploaded ``source_id`` attached.  Columns are discovered once from
    TAP_SCHEMA and quoted exactly once by :func:`select_list`.
    """

    def __init__(self, name: str, table: str, aliases: dict, *, url: str = VIZIER_TAP,
                 xmatch_url: str | None = None, xmatch_catalogue: str | None = None,
                 chunk: int = 4000):
        self.name, self.table, self.aliases, self.url = name, table, aliases, url
        self.xmatch_url = xmatch_url or DEFAULTS["xmatch_url"]
        self.xmatch_catalogue = xmatch_catalogue
        self.chunk = int(chunk)
        self.discovery: dict | None = None
        self.resolved: dict[str, str] = {}
        self.routes_used: list[str] = []

    def _discover(self) -> None:
        if self.discovery is None:
            self.discovery = discover_columns(self.table, self.url)
            self.resolved = resolve_aliases(self.discovery["names"], self.aliases)
            if "ra" not in self.resolved or "dec" not in self.resolved:
                raise RuntimeError(f"{self.table}: RA/Dec not resolved from "
                                   f"{self.discovery['names'][:40]}")

    def _tap_upload(self, pos: pd.DataFrame, radius_arcsec: float, label: str) -> pd.DataFrame:
        from astropy.table import Table

        self._discover()
        q = vizier_upload_query(self.table, self.resolved, radius_arcsec)
        up = Table({"source_id": pos["source_id"].to_numpy(np.int64),
                    "ra": pos["ra"].to_numpy(float), "dec": pos["dec"].to_numpy(float)})
        raw = run_vizier(q, uploads={"targets": up}, label=label, url=self.url)
        return self._canonical(raw)

    def _canonical(self, raw: pd.DataFrame) -> pd.DataFrame:
        low = {str(c).lower(): c for c in raw.columns}
        out = pd.DataFrame(index=raw.index)
        out["source_id"] = pd.to_numeric(raw[low["source_id"]], errors="coerce") \
            if "source_id" in low else np.nan
        for canon, actual in self.resolved.items():
            key = actual.lower()
            if key in low:
                out[canon] = raw[low[key]].to_numpy()
        return out

    def _xmatch(self, pos: pd.DataFrame, radius_arcsec: float, label: str) -> pd.DataFrame:
        import io

        import requests

        if not self.xmatch_catalogue:
            raise RuntimeError(f"{label}: no X-Match catalogue configured")
        csv = pos[["source_id", "ra", "dec"]].to_csv(index=False)
        r = requests.post(self.xmatch_url, data={
            "request": "xmatch", "distMaxArcsec": f"{radius_arcsec:g}",
            "RESPONSEFORMAT": "csv", "cat2": self.xmatch_catalogue,
            "colRA1": "ra", "colDec1": "dec", "selection": "all"},
            files={"cat1": ("positions.csv", csv)}, timeout=600)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text))
        self.resolved = self.resolved or resolve_aliases(raw.columns, self.aliases)
        return self._canonical(raw)

    def __call__(self, positions: pd.DataFrame, radius_arcsec: float, label: str) -> pd.DataFrame:
        frames = []
        for i in range(0, len(positions), self.chunk):
            pos = positions.iloc[i:i + self.chunk]
            lab = f"{label}[{i}:{i + len(pos)}]"
            try:
                frames.append(self._tap_upload(pos, radius_arcsec, lab))
                self.routes_used.append("tap_upload")
            except Exception as exc:                                # noqa: BLE001
                print(f"[baffle-vet] {lab}: TAP upload failed ({exc!r}); trying X-Match",
                      flush=True)
                frames.append(self._xmatch(pos, radius_arcsec, lab))
                self.routes_used.append("xmatch")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def default_matchers(cfg: dict | None = None) -> dict:
    c = _cfg(cfg)
    out = {}
    for name, aliases in (("allwise", ALLWISE_ALIASES), ("catwise", CATWISE_ALIASES),
                          ("unwise", UNWISE_ALIASES)):
        sub = c[name]
        out[name] = UploadMatcher(name, sub["table"], aliases, xmatch_url=c["xmatch_url"],
                                  xmatch_catalogue=sub.get("xmatch_catalogue"),
                                  chunk=int(c["upload_chunk"]))
    return out


# ===========================================================================
# Ledger + fetch wrapper
# ===========================================================================
@dataclass
class VetLedger:
    entries: list = field(default_factory=list)

    def record(self, label: str, status: str, *, n_rows: int = 0, seconds: float = 0.0,
               error: str | None = None, **extra) -> dict:
        e = {"label": label, "status": status, "n_rows": int(n_rows),
             "seconds": round(float(seconds), 1), "error": error, "utc": _now()}
        e.update(extra)
        self.entries.append(e)
        return e

    def n_failed(self) -> int:
        return sum(1 for e in self.entries if e["status"] == QUERY_FAILED)

    def all_failed(self) -> bool:
        return bool(self.entries) and self.n_failed() == len(self.entries)


def _fetch(ledger: VetLedger, label: str, fn, *args) -> pd.DataFrame | None:
    """Run one fetch; ledger the outcome; ``None`` means the archive was not reached."""
    t0 = time.monotonic()
    try:
        df = fn(*args)
    except Exception as exc:                                        # noqa: BLE001
        ledger.record(label, QUERY_FAILED, seconds=time.monotonic() - t0, error=repr(exc)[:500])
        print(f"[baffle-vet] {label}: QUERY_FAILED {exc!r}", flush=True)
        return None
    df = pd.DataFrame() if df is None else _lower(pd.DataFrame(df))
    ledger.record(label, QUERY_OK if len(df) else QUERY_ZERO, n_rows=len(df),
                  seconds=time.monotonic() - t0)
    print(f"[baffle-vet] {label}: {len(df)} rows in {time.monotonic() - t0:.1f} s", flush=True)
    return df


# ===========================================================================
# Per-candidate assembly
# ===========================================================================
def nearest_per_target(matches: pd.DataFrame, targets: pd.DataFrame, radius_arcsec: float,
                       ra_col: str = "ra", dec_col: str = "dec") -> pd.DataFrame:
    """Nearest catalogue row per uploaded target (sep computed locally) plus a count."""
    cols = ["source_id", "sep_arcsec", "n_within"]
    if matches is None or len(matches) == 0 or "source_id" not in matches.columns:
        return pd.DataFrame(columns=cols)
    m = matches.copy()
    m["source_id"] = pd.to_numeric(m["source_id"], errors="coerce")
    t = targets.set_index("source_id")
    sid = m["source_id"].to_numpy()
    ok = np.isin(sid, t.index.to_numpy())
    m = m[ok].copy()
    if not len(m):
        return pd.DataFrame(columns=cols)
    tra = t.loc[m["source_id"].to_numpy(), "ra"].to_numpy(float)
    tde = t.loc[m["source_id"].to_numpy(), "dec"].to_numpy(float)
    m["sep_arcsec"] = separation_arcsec(tra, tde, _num(m, ra_col), _num(m, dec_col))
    m = m[m["sep_arcsec"] <= float(radius_arcsec)]
    if not len(m):
        return pd.DataFrame(columns=cols)
    m = m.sort_values(["source_id", "sep_arcsec"])
    counts = m.groupby("source_id").size().rename("n_within")
    near = m.drop_duplicates("source_id", keep="first").set_index("source_id")
    near = near.join(counts)
    return near.reset_index()


def gaia_neighbour_stats(neigh: pd.DataFrame | None, cands: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Brightest-neighbour G / sep and counts within 3″/6″/10″ per candidate."""
    c = _cfg(cfg)
    r_blend = float(c["blend_radius_arcsec"])
    out = pd.DataFrame({"source_id": cands["source_id"].to_numpy()})
    for col in ("gaia_n_3as", "gaia_n_6as", "gaia_n_10as"):
        out[col] = np.nan
    out["gaia_brightest_neighbour_g"] = np.nan
    out["gaia_brightest_neighbour_sep_arcsec"] = np.nan
    out["gaia_neighbours_checked"] = False
    out["blend_flux_theft"] = False
    out["crowded_field"] = False
    if neigh is None or len(neigh) == 0 or "target_source_id" not in neigh.columns:
        return out
    n = neigh.copy()
    n["target_source_id"] = pd.to_numeric(n["target_source_id"], errors="coerce")
    n["source_id"] = pd.to_numeric(n["source_id"], errors="coerce")
    cpos = cands.set_index("source_id")
    for i, sid in enumerate(out["source_id"]):
        rows = n[n["target_source_id"] == sid]
        if not len(rows):
            continue
        out.loc[i, "gaia_neighbours_checked"] = True
        sep = separation_arcsec(float(cpos.loc[sid, "ra"]), float(cpos.loc[sid, "dec"]),
                                _num(rows, "ra"), _num(rows, "dec"))
        g = _num(rows, "phot_g_mean_mag")
        others = rows["source_id"].to_numpy() != sid
        out.loc[i, "gaia_n_3as"] = int((sep <= 3.0).sum())
        out.loc[i, "gaia_n_6as"] = int((sep <= 6.0).sum())
        out.loc[i, "gaia_n_10as"] = int((sep <= 10.0).sum())
        if others.any():
            g_o, s_o = g[others], sep[others]
            fin = np.isfinite(g_o)
            if fin.any():
                k = int(np.nanargmin(np.where(fin, g_o, np.inf)))
                out.loc[i, "gaia_brightest_neighbour_g"] = g_o[k]
                out.loc[i, "gaia_brightest_neighbour_sep_arcsec"] = s_o[k]
            g_c = float(cpos.loc[sid, "phot_g_mean_mag"]) if "phot_g_mean_mag" in cpos else np.nan
            if np.isfinite(g_c):
                out.loc[i, "blend_flux_theft"] = bool(
                    np.any((s_o <= r_blend) & fin & (g_o < g_c + float(c["blend_dg_max"]))))
        out.loc[i, "crowded_field"] = bool(
            (sep <= float(c["crowded_radius_arcsec"])).sum() >= int(c["crowded_n_min"]))
    return out


def unwise_vega_mags(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ``w1``/``w2`` (Vega) from AB nanomaggy fluxes when the mirror serves only fluxes."""
    out = df.copy()
    for b in ("w1", "w2"):
        flux = _num(out, f"{b}flux")
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = 22.5 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan)) - UNWISE_VEGA_OFFSET[b]
            eflux = _num(out, f"e_{b}flux")
            emag = 1.0857 * np.where(flux > 0, eflux / np.where(flux > 0, flux, np.nan), np.nan)
        have = _num(out, b) if b in out.columns else np.full(len(out), np.nan)
        out[b] = np.where(np.isfinite(have), have, mag)
        ehave = _num(out, f"e_{b}") if f"e_{b}" in out.columns else np.full(len(out), np.nan)
        out[f"e_{b}"] = np.where(np.isfinite(ehave), ehave, emag)
    return out


def locus_residuals(cands: pd.DataFrame, w1, e_w1, w2, e_w2, locus: Locus | None,
                    lcfg: dict | None = None) -> dict:
    """K_s − W residuals of independent photometry against the SAME locus."""
    n = len(cands)
    if locus is None:
        return {k: np.full(n, np.nan) for k in ("resid_w1", "sig_w1", "resid_w2", "sig_w2")}
    jk = _num(cands, "j_m") - _num(cands, "ks_m")
    ks, e_ks = _num(cands, "ks_m"), np.nan_to_num(_num(cands, "ks_msigcom"))
    cls = (cands["lum_class"].astype(str).to_numpy() if "lum_class" in cands.columns
           else luminosity_class(cands, lcfg).to_numpy().astype(str))
    out = {}
    for band, w, e in (("w1", w1, e_w1), ("w2", w2, e_w2)):
        w = np.asarray(w, dtype=float)
        e = np.nan_to_num(np.asarray(e, dtype=float))
        med, sc = locus.predict(jk, cls, band)
        resid = (ks - w) - med
        out[f"resid_{band}"] = resid
        out[f"sig_{band}"] = resid / np.sqrt(sc ** 2 + e_ks ** 2 + e ** 2)
    return out


def classify_independent(resid: dict, cfg: dict) -> np.ndarray:
    """'photospheric' / 'confirms_deficit' / 'ambiguous' / 'missing' per star."""
    c = _cfg(cfg)
    r1, s1, r2, s2 = (resid[k] for k in ("resid_w1", "sig_w1", "resid_w2", "sig_w2"))
    have = np.isfinite(r1) & np.isfinite(r2)
    photo = have & (np.abs(s1) < float(c["photospheric_nsig"])) & (np.abs(s2) < float(c["photospheric_nsig"]))
    conf = have & (r1 < float(c["deficit_mag"])) & (r2 < float(c["deficit_mag"])) \
        & (s1 < -float(c["deficit_nsig"])) & (s2 < -float(c["deficit_nsig"]))
    out = np.where(~have, "missing", np.where(photo, "photospheric",
                                             np.where(conf, "confirms_deficit", "ambiguous")))
    return out.astype(object)


def w3_residual(cands: pd.DataFrame, w3, e_w3, locus: Locus | None, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """resid_w3 against the locus's W3 bins (NaN where no W3 locus / no usable W3)."""
    c = _cfg(cfg)
    n = len(cands)
    if locus is None or not (locus.has("all", "w3") or any(locus.has(k, "w3") for k in locus.classes())):
        return np.full(n, np.nan), np.full(n, np.nan)
    w3 = np.asarray(w3, dtype=float)
    e_w3 = np.asarray(e_w3, dtype=float)
    usable = np.isfinite(w3) & np.isfinite(e_w3) & (e_w3 < float(c["w3_err_max"]))
    jk = _num(cands, "j_m") - _num(cands, "ks_m")
    ks, e_ks = _num(cands, "ks_m"), np.nan_to_num(_num(cands, "ks_msigcom"))
    cls = (cands["lum_class"].astype(str).to_numpy() if "lum_class" in cands.columns
           else np.full(n, "dwarf"))
    med, sc = locus.predict(jk, cls, "w3")
    resid = np.where(usable, (ks - w3) - med, np.nan)
    sig = resid / np.sqrt(sc ** 2 + e_ks ** 2 + np.nan_to_num(e_w3) ** 2)
    return resid, sig


def decide(row: pd.Series) -> tuple[str, str, str]:
    """(vet_verdict, vetoes, notes) from one assembled row.  Precedence documented."""
    vetoes, notes = [], []
    if bool(row.get("blend_flux_theft", False)):
        vetoes.append("blend_flux_theft")
    if bool(row.get("crowded_field", False)):
        notes.append("crowded_field")
    if not bool(row.get("gaia_neighbours_checked", False)):
        notes.append("gaia_neighbours_unavailable")
    if bool(row.get("deblended_component", False)):
        vetoes.append("deblended_component")
    if bool(row.get("saturated_pixels", False)):
        vetoes.append("saturated_pixels")
    aw = str(row.get("allwise_status", "unavailable"))
    if aw == "unavailable":
        notes.append("allwise_unavailable")
    elif aw == "missing":
        notes.append("allwise_missing")
    ind = str(row.get("independent_class", "missing"))
    if ind == "photospheric":
        vetoes.append("catwise_photospheric")
    elif ind == "confirms_deficit":
        notes.append("catwise_confirms_deficit")
    elif ind == "missing":
        notes.append("catwise_unavailable" if str(row.get("catwise_status", "")) == "unavailable"
                     and str(row.get("unwise_status", "")) == "unavailable" else "catwise_missing")
    w3s = str(row.get("w3_status", "unmeasured"))
    if w3s == "excess":
        vetoes.append("w3_excess")
    elif w3s == "deficit":
        notes.append("w3_deficit_consistent")
    if "blend_flux_theft" in vetoes:
        verdict = "BLEND"
    elif "deblended_component" in vetoes:
        verdict = "DEBLENDED_COMPONENT"
    elif "saturated_pixels" in vetoes or "catwise_photospheric" in vetoes:
        verdict = "ALLWISE_PHOTOMETRY_WRONG"
    elif "w3_excess" in vetoes:
        verdict = "W3_INCONSISTENT"
    elif (bool(row.get("gaia_neighbours_checked", False)) and aw == "matched"
          and ind == "confirms_deficit"):
        verdict = "SURVIVES_VET"
    else:
        verdict = "INCONCLUSIVE"
    return verdict, ";".join(vetoes), ";".join(notes)


# ===========================================================================
# Deficit-candidate vet
# ===========================================================================
def vet_deficit_candidates(cands: pd.DataFrame, cfg: dict, locus: Locus | None, ledger: VetLedger,
                           *, gaia_fetcher=None, matchers: dict | None = None,
                           locus_cfg: dict | None = None) -> pd.DataFrame:
    """Assemble every measurement and the verdict for each candidate row."""
    c = _cfg(cfg)
    ep = c["epochs"]
    cands = cands.reset_index(drop=True).copy()
    cands["source_id"] = pd.to_numeric(cands["source_id"], errors="coerce").astype("int64")
    n = len(cands)
    if n == 0:
        return cands
    matchers = matchers if matchers is not None else default_matchers(c)
    gaia_fetcher = gaia_fetcher or default_gaia_upload_fetcher
    pos_gaia = cands[["source_id", "ra", "dec"]].copy()

    # 1. Gaia neighbours (one upload join)
    neigh = _fetch(ledger, "gaia-neighbours (upload)", gaia_fetcher, pos_gaia,
                   float(c["gaia_neighbour_radius_arcsec"]), "gaia-neighbours")
    gstats = gaia_neighbour_stats(neigh, cands, c)
    out = cands.merge(gstats, on="source_id", how="left")

    # 2-3. AllWISE proper, CatWISE, unWISE (one upload each, PM-propagated)
    tables = {}
    for name in ("allwise", "catwise", "unwise"):
        sub = c[name]
        ra_e, de_e = propagate(cands["ra"], cands["dec"], _num(cands, "pmra"), _num(cands, "pmdec"),
                               float(ep["gaia"]), float(ep[name]))
        pos = pd.DataFrame({"source_id": cands["source_id"], "ra": ra_e, "dec": de_e})
        fn = matchers.get(name)
        raw = (_fetch(ledger, f"{name} (upload, {sub['radius_arcsec']}\")", fn, pos,
                      float(sub["radius_arcsec"]), name) if fn is not None else None)
        if raw is not None and name == "unwise" and len(raw):
            raw = unwise_vega_mags(raw)
        near = nearest_per_target(raw, pos, float(sub["radius_arcsec"])) if raw is not None else None
        tables[name] = (raw is not None, near)

    def _attach(name: str, cols: dict):
        avail, near = tables[name]
        out[f"{name}_status"] = "unavailable" if not avail else "missing"
        for outcol in cols.values():
            # object dtype: designations and flag strings share the table with floats
            out[outcol] = pd.Series([np.nan] * len(out), index=out.index, dtype=object)
        out[f"{name}_sep_arcsec"] = np.nan
        out[f"{name}_n_within"] = np.nan
        if avail and near is not None and len(near):
            near = near.set_index("source_id")
            hit = out["source_id"].isin(near.index).to_numpy()
            idx = out.loc[hit, "source_id"].to_numpy()
            out.loc[hit, f"{name}_status"] = "matched"
            out.loc[hit, f"{name}_sep_arcsec"] = near.loc[idx, "sep_arcsec"].to_numpy()
            out.loc[hit, f"{name}_n_within"] = near.loc[idx, "n_within"].to_numpy()
            for canon, outcol in cols.items():
                if canon in near.columns:
                    out.loc[hit, outcol] = near.loc[idx, canon].to_numpy()

    _attach("allwise", {"designation": "allwise_designation", "w1": "allwise_w1", "e_w1": "allwise_e_w1",
                        "w2": "allwise_w2", "e_w2": "allwise_e_w2", "w3": "allwise_w3",
                        "e_w3": "allwise_e_w3", "w4": "allwise_w4", "e_w4": "allwise_e_w4",
                        "cc_flags": "allwise_cc_flags", "ph_qual": "allwise_ph_qual",
                        "ext_flag": "allwise_ext_flag", "nb": "allwise_nb", "na": "allwise_na",
                        "w1sat": "allwise_w1sat", "w2sat": "allwise_w2sat",
                        "w1flux": "allwise_w1flux", "w2flux": "allwise_w2flux",
                        "w1rchi2": "allwise_w1rchi2", "w2rchi2": "allwise_w2rchi2",
                        "w3snr": "allwise_w3snr"})
    _attach("catwise", {"designation": "catwise_designation", "w1": "catwise_w1", "e_w1": "catwise_e_w1",
                        "w2": "catwise_w2", "e_w2": "catwise_e_w2", "cc_flags": "catwise_cc_flags",
                        "ab_flags": "catwise_ab_flags"})
    _attach("unwise", {"designation": "unwise_designation", "w1": "unwise_w1", "e_w1": "unwise_e_w1",
                       "w2": "unwise_w2", "e_w2": "unwise_e_w2", "flags_w1": "unwise_flags_w1",
                       "flags_w2": "unwise_flags_w2"})

    # AllWISE-proper flags
    nb, na = _num(out, "allwise_nb"), _num(out, "allwise_na")
    out["deblended_component"] = (nb > 1) | (na > 0)
    out["saturated_pixels"] = (_num(out, "allwise_w1sat") > 0) | (_num(out, "allwise_w2sat") > 0)
    out["allwise_flags_available"] = np.isfinite(nb) | np.isfinite(na)
    out["allwise_sat_available"] = np.isfinite(_num(out, "allwise_w1sat"))

    # Independent photometry against the same locus
    for name in ("catwise", "unwise"):
        r = locus_residuals(out, _num(out, f"{name}_w1"), _num(out, f"{name}_e_w1"),
                            _num(out, f"{name}_w2"), _num(out, f"{name}_e_w2"), locus, locus_cfg)
        for k, v in r.items():
            out[f"{name}_{k}"] = v
        out[f"{name}_class"] = classify_independent(r, c)
    cat_cls = out["catwise_class"].astype(str).to_numpy()
    un_cls = out["unwise_class"].astype(str).to_numpy()
    # CatWISE leads (deeper, PM-aware); unWISE fills in and can contradict it.
    ind = np.where(cat_cls != "missing", cat_cls, un_cls).astype(object)
    both = (cat_cls != "missing") & (un_cls != "missing")
    ind[both & (cat_cls != un_cls) & ((cat_cls == "photospheric") | (un_cls == "photospheric"))] = "photospheric"
    ind[both & (cat_cls != un_cls) & ~((cat_cls == "photospheric") | (un_cls == "photospheric"))] = "ambiguous"
    out["independent_class"] = ind

    # 4. W3 consistency from the re-pulled AllWISE row (falls back to the screen's W3)
    w3 = np.where(np.isfinite(_num(out, "allwise_w3")), _num(out, "allwise_w3"), _num(out, "w3mpro"))
    e_w3 = np.where(np.isfinite(_num(out, "allwise_e_w3")), _num(out, "allwise_e_w3"),
                    _num(out, "w3mpro_error"))
    r3, s3 = w3_residual(out, w3, e_w3, locus, c)
    out["vet_resid_w3"], out["vet_sig_w3"] = r3, s3
    out["w3_status"] = np.where(~np.isfinite(r3), "unmeasured",
                                np.where(r3 > float(c["w3_excess_mag"]), "excess",
                                         np.where(r3 < float(c["w3_deficit_mag"]), "deficit", "normal")))
    if locus is None or not any(locus.has(k, "w3") for k in list(locus.classes())):
        out["w3_status_note"] = "no W3 locus in locus.json (re-screen with the w3mpro_error fallback)"

    verdicts = [decide(row) for _, row in out.iterrows()]
    out["vet_verdict"] = [v[0] for v in verdicts]
    out["vet_vetoes"] = [v[1] for v in verdicts]
    out["vet_notes"] = [v[2] for v in verdicts]
    return out


# ===========================================================================
# Missing-track vet
# ===========================================================================
def select_missing_targets(missing: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """``nearby`` / ``etz`` rows plus a uniform random control of the rest."""
    c = _cfg(cfg)["missing"]
    m = missing.reset_index(drop=True).copy()
    if not len(m):
        m["vet_group"] = pd.Series(dtype=object)
        return m
    flag = np.zeros(len(m), dtype=bool)
    for col in ("nearby", "etz"):
        if col in m.columns:
            flag |= m[col].map(lambda x: str(x).lower() in ("true", "1")).to_numpy()
    prio = m[flag].copy()
    prio["vet_group"] = np.where(prio["etz"].map(lambda x: str(x).lower() in ("true", "1"))
                                 if "etz" in prio.columns else False, "etz", "nearby")
    rest = m[~flag]
    n_ctrl = min(int(c["n_control"]), len(rest))
    ctrl = rest.sample(n=n_ctrl, random_state=int(c["seed"])) if n_ctrl else rest.iloc[:0]
    ctrl = ctrl.copy()
    ctrl["vet_group"] = "control"
    return pd.concat([prio, ctrl], ignore_index=True)


def vet_missing(missing: pd.DataFrame, cfg: dict, ledger: VetLedger, *,
                matchers: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Direct positional AllWISE / CatWISE / unWISE match of the missing-track targets."""
    c = _cfg(cfg)
    mc, ep = c["missing"], c["epochs"]
    targets = select_missing_targets(missing, c)
    rep: dict = {"n_targets": int(len(targets)),
                 "n_by_group": targets["vet_group"].value_counts().to_dict() if len(targets) else {}}
    if not len(targets):
        rep["missing_vet_verdict"] = "NO_DATA_REACHED"
        return targets, rep
    targets["source_id"] = pd.to_numeric(targets["source_id"], errors="coerce").astype("int64")
    matchers = matchers if matchers is not None else default_matchers(c)
    r_close, r_far, r_near = (float(mc["radius_close_arcsec"]), float(mc["radius_far_arcsec"]),
                              float(mc["nearest_radius_arcsec"]))

    ra_w, de_w = propagate(targets["ra"], targets["dec"], _num(targets, "pmra"), _num(targets, "pmdec"),
                           float(ep["gaia"]), float(ep["allwise"]))
    pos_w = pd.DataFrame({"source_id": targets["source_id"], "ra": ra_w, "dec": de_w})
    raw = _fetch(ledger, f"missing: allwise (upload, {r_far:g}\")", matchers["allwise"], pos_w, r_far,
                 "missing-allwise")
    out = targets.copy()
    out["wise_status"] = "unavailable" if raw is None else "no_wise_source_within_15as"
    out["allwise_sep_arcsec"] = np.nan
    out["allwise_w1"], out["allwise_w2"] = np.nan, np.nan
    out["allwise_cc_flags"] = pd.Series([""] * len(out), index=out.index, dtype=object)
    if raw is not None:
        near = nearest_per_target(raw, pos_w, r_far)
        if len(near):
            near = near.set_index("source_id")
            hit = out["source_id"].isin(near.index).to_numpy()
            idx = out.loc[hit, "source_id"].to_numpy()
            sep = near.loc[idx, "sep_arcsec"].to_numpy(float)
            out.loc[hit, "allwise_sep_arcsec"] = sep
            for canon, col in (("w1", "allwise_w1"), ("w2", "allwise_w2"), ("cc_flags", "allwise_cc_flags")):
                if canon in near.columns:
                    out.loc[hit, col] = near.loc[idx, canon].to_numpy()
            out.loc[hit, "wise_status"] = np.where(sep <= r_close, "wise_source_present_within_6as",
                                                   "wise_source_present_6_to_15as")
    # the truly-absent group: CatWISE / unWISE presence and the nearest AllWISE source
    absent = out["wise_status"].eq("no_wise_source_within_15as").to_numpy()
    out["catwise_present"], out["unwise_present"] = np.nan, np.nan
    out["nearest_allwise_sep_arcsec"] = np.nan
    out["nearest_allwise_cc_flags"] = pd.Series([""] * len(out), index=out.index, dtype=object)
    if absent.any():
        sub = out[absent]
        for name, col in (("catwise", "catwise_present"), ("unwise", "unwise_present")):
            ra_e, de_e = propagate(sub["ra"], sub["dec"], _num(sub, "pmra"), _num(sub, "pmdec"),
                                   float(ep["gaia"]), float(ep[name]))
            pos = pd.DataFrame({"source_id": sub["source_id"], "ra": ra_e, "dec": de_e})
            r = _fetch(ledger, f"missing: {name} (upload, {r_far:g}\")", matchers.get(name), pos, r_far,
                       f"missing-{name}") if matchers.get(name) is not None else None
            if r is None:
                continue
            near = nearest_per_target(r, pos, r_far)
            present = out.loc[absent, "source_id"].isin(near["source_id"]).to_numpy() if len(near) \
                else np.zeros(int(absent.sum()), dtype=bool)
            out.loc[absent, col] = present.astype(float)
        pos_n = pos_w[pos_w["source_id"].isin(sub["source_id"])]
        r = _fetch(ledger, f"missing: allwise nearest (upload, {r_near:g}\")", matchers["allwise"],
                   pos_n, r_near, "missing-allwise-nearest")
        if r is not None:
            near = nearest_per_target(r, pos_n, r_near)
            if len(near):
                near = near.set_index("source_id")
                hit = out["source_id"].isin(near.index).to_numpy() & absent
                idx = out.loc[hit, "source_id"].to_numpy()
                out.loc[hit, "nearest_allwise_sep_arcsec"] = near.loc[idx, "sep_arcsec"].to_numpy()
                if "cc_flags" in near.columns:
                    out.loc[hit, "nearest_allwise_cc_flags"] = near.loc[idx, "cc_flags"].to_numpy()
    cat_p, un_p = _num(out, "catwise_present"), _num(out, "unwise_present")
    truly = absent & ~(cat_p > 0) & ~(un_p > 0) & (np.isfinite(cat_p) | np.isfinite(un_p))
    out["truly_missing"] = truly
    out["missing_vet_status"] = np.where(out["wise_status"].eq("unavailable"), "unavailable",
                                         np.where(truly, "truly_missing",
                                                  np.where(absent, "absent_in_allwise_only",
                                                           out["wise_status"])))

    counters = out["wise_status"].value_counts().to_dict()
    by_group = {g: d["wise_status"].value_counts().to_dict() for g, d in out.groupby("vet_group")}
    rep.update({
        "counters": {k: int(v) for k, v in counters.items()},
        "counters_by_group": {g: {k: int(v) for k, v in d.items()} for g, d in by_group.items()},
        "n_absent_in_allwise": int(absent.sum()),
        "n_truly_missing": int(truly.sum()),
        "control_no_wise_fraction": (float((out["vet_group"].eq("control")
                                            & absent).sum() / max(1, int(out["vet_group"].eq("control").sum())))
                                     if int(out["vet_group"].eq("control").sum()) else None),
        "truly_missing": out.loc[truly, [c_ for c_ in ("source_id", "ra", "dec", "b", "phot_g_mean_mag",
                                                       "ks_m", "parallax", "etz", "nearby", "vet_group",
                                                       "nearest_allwise_sep_arcsec",
                                                       "nearest_allwise_cc_flags")
                                         if c_ in out.columns]].to_dict(orient="records"),
        "note": ("the Gaia x AllWISE best-neighbour table lacking an entry is the cross-match's "
                 "behaviour on saturated / crowded sources; this is the directly measured "
                 "absence rate"),
    })
    if raw is None:
        rep["missing_vet_verdict"] = "NO_DATA_REACHED"
    elif truly.any():
        rep["missing_vet_verdict"] = f"TRULY_MISSING_COUNTERPARTS_PENDING (n={int(truly.sum())})"
    else:
        rep["missing_vet_verdict"] = "NO_TRULY_MISSING_COUNTERPART"
    return out, rep


# ===========================================================================
# Stage
# ===========================================================================
_VET_COMPACT = (
    "source_id", "ra", "dec", "l", "b", "ecl_lat", "parallax", "distance_pc", "phot_g_mean_mag",
    "bp_rp", "lum_class", "jk", "j_m", "ks_m", "w1mpro", "w2mpro", "w3mpro", "w3mpro_error",
    "resid_w1", "sig_w1", "resid_w2", "sig_w2", "etz", "nearby", "vet_source",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    return o


def run_vet_stage(cfg: dict, out_dir, *, gaia_fetcher=None, matchers=None,
                  locus_cfg: dict | None = None) -> dict:
    """Vet the screen's outputs on disk; write ``vet.json``, ``vet_table.csv``,
    ``vetted_candidates.csv``, ``missing_vet.json``, ``missing_vet.csv``."""
    c = _cfg(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ledger = VetLedger()
    locus = None
    lp = out / "locus.json"
    if lp.exists():
        try:
            locus = Locus.load(lp)
        except Exception as exc:                                    # noqa: BLE001
            print(f"[baffle-vet] locus.json unreadable: {exc!r}", flush=True)
    cands = _read_csv(out / "candidates.csv")
    cands["vet_source"] = "candidates" if len(cands) else pd.Series(dtype=object)
    if c["include_deferred_lpv"]:
        dl = _read_csv(out / "deferred_lpv.csv")
        if len(dl):
            dl["vet_source"] = "deferred_lpv"
            cands = pd.concat([cands, dl], ignore_index=True)
    missing = _read_csv(out / "missing_candidates.csv")

    rep: dict = {"stage": "vet", "generated_utc": _now(), "n_candidates_in": int(len(cands)),
                 "n_from_deferred_lpv": int((cands.get("vet_source") == "deferred_lpv").sum())
                 if len(cands) else 0,
                 "locus_loaded": locus is not None,
                 "locus_has_w3": bool(locus is not None and any(locus.has(k, "w3") for k in locus.classes())),
                 "n_missing_in": int(len(missing))}
    if len(cands):
        table = vet_deficit_candidates(cands, c, locus, ledger, gaia_fetcher=gaia_fetcher,
                                       matchers=matchers, locus_cfg=locus_cfg)
    else:
        table = cands
    if len(table):
        table.to_csv(out / "vet_table.csv", index=False)
        surv = table[table["vet_verdict"] == "SURVIVES_VET"]
        surv.to_csv(out / "vetted_candidates.csv", index=False)
        rep["verdict_counts"] = {k: int((table["vet_verdict"] == k).sum()) for k in VET_VERDICTS}
        rep["veto_counters"] = {k: int(table["vet_vetoes"].astype(str).str.split(";").map(lambda s, k=k: k in s).sum())
                                for k in VET_VETOES}
        rep["note_counters"] = {k: int(table["vet_notes"].astype(str).str.split(";").map(lambda s, k=k: k in s).sum())
                                for k in VET_NOTES}
        rep["n_survivors"] = int(len(surv))
        rep["n_survivors_etz"] = int(surv["etz"].map(lambda x: str(x).lower() == "true").sum()) \
            if "etz" in surv.columns else 0
        rep["n_survivors_nearby"] = int(surv["nearby"].map(lambda x: str(x).lower() == "true").sum()) \
            if "nearby" in surv.columns else 0
        rep["survivors"] = surv[[k for k in _VET_COMPACT + ("vet_notes", "independent_class",
                                                            "catwise_resid_w1", "catwise_resid_w2",
                                                            "unwise_resid_w1", "unwise_resid_w2",
                                                            "w3_status", "gaia_n_6as")
                                 if k in surv.columns]].to_dict(orient="records")
        rep["allwise_columns_missing"] = [k for k in ("nb", "na", "w1sat", "w2sat", "w1rchi2")
                                          if not table[f"allwise_{k}"].notna().any()]
    else:
        pd.DataFrame().to_csv(out / "vet_table.csv", index=False)
        pd.DataFrame().to_csv(out / "vetted_candidates.csv", index=False)
        rep["verdict_counts"] = {k: 0 for k in VET_VERDICTS}
        rep["n_survivors"] = 0

    if len(missing):
        mtable, mrep = vet_missing(missing, c, ledger, matchers=matchers)
        mtable.to_csv(out / "missing_vet.csv", index=False)
    else:
        mrep = {"n_targets": 0, "missing_vet_verdict": "NO_DATA_REACHED",
                "note": "no missing_candidates.csv rows on disk"}
        pd.DataFrame().to_csv(out / "missing_vet.csv", index=False)
    mrep.update(stage="missing_vet", generated_utc=_now())

    deficit_reached = any(e["status"] != QUERY_FAILED for e in ledger.entries
                          if not e["label"].startswith("missing"))
    if not len(cands):
        rep["verdict_deficit_after_vet"] = "NO_DATA_REACHED" if not (out / "candidates.csv").exists() \
            else "NO_MIDIR_DEFICIT_SURVIVOR"
        rep["note"] = "no candidate rows to vet"
    elif not deficit_reached:
        rep["verdict_deficit_after_vet"] = "NO_DATA_REACHED"
        rep["note"] = "every vet archive query failed; nothing was vetted"
    elif rep["n_survivors"] > 0:
        rep["verdict_deficit_after_vet"] = f"MIDIR_DEFICIT_CANDIDATES_SURVIVE_VET (n={rep['n_survivors']})"
    else:
        rep["verdict_deficit_after_vet"] = "NO_MIDIR_DEFICIT_SURVIVOR"
    if ledger.all_failed():
        rep["verdict_deficit_after_vet"] = "NO_DATA_REACHED"
        mrep["missing_vet_verdict"] = "NO_DATA_REACHED"
    rep["ledger"] = ledger.entries
    rep["n_queries"] = len(ledger.entries)
    rep["n_queries_failed"] = ledger.n_failed()
    rep["seconds"] = round(time.monotonic() - t0, 1)
    rep["missing_vet_verdict"] = mrep["missing_vet_verdict"]
    (out / "vet.json").write_text(json.dumps(_json_safe(rep), indent=2, default=str))
    (out / "missing_vet.json").write_text(json.dumps(_json_safe(mrep), indent=2, default=str))
    print(f"[baffle-vet] {rep['verdict_deficit_after_vet']} | {mrep['missing_vet_verdict']} "
          f"({rep['n_queries']} queries, {rep['n_queries_failed']} failed)", flush=True)
    return rep


__all__ = ["ALLWISE_ALIASES", "CATWISE_ALIASES", "DEFAULTS", "UNWISE_ALIASES", "VET_NOTES",
           "VET_VERDICTS", "VET_VETOES", "UploadMatcher", "VetLedger", "classify_independent",
           "decide", "default_gaia_upload_fetcher", "default_matchers", "gaia_neighbour_stats",
           "gaia_neighbours_upload_query", "locus_residuals", "nearest_per_target", "propagate",
           "run_vet_stage", "select_missing_targets", "separation_arcsec", "unwise_vega_mags",
           "vet_deficit_candidates", "vet_missing", "vizier_upload_query", "w3_residual"]

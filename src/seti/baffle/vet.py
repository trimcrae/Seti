"""BAFFLE ``vet`` stage: per-candidate archive checks that the screen cannot do.

Why this stage exists (runs 34053752510 and 34054735689)
----------------------------------------------------------
The first real screen left 135 two-band-deficit survivors, 107 at |b| < 10°,
with ``neighbours_not_checked = 2055``.  The first vet (8 upload queries, 0
failures) found 19 deblended AllWISE components, 15 stars whose AllWISE
photometry was contradicted by CatWISE/unWISE, 1 W3 excess — and 102
INCONCLUSIVE, every one ``independent_class = ambiguous``.  Two reasons:

* the unWISE zero point was wrong by exactly the AB→Vega offset (unWISE
  ``FW1``/``FW2`` are **Vega** nanomaggies, Schlafly+2019; ``22.5 − 2.5
  log10(F)`` is already Vega).  Fixed, and guarded by a self-check that no
  naming can fool: the median of (unWISE − CatWISE) over the matched
  candidates must sit within ``zeropoint_check_max_mag`` of zero, otherwise
  ``unwise_zeropoint_suspect`` is recorded with the measured offset and unWISE
  does not vote.  The same check runs on CatWISE vs AllWISE-proper.
* CatWISE agrees with AllWISE (median residuals −0.34 / −0.39) — but CatWISE
  is built from the same WISE frames, and the survivors' SED is *grey*
  (W1 ≈ W2 ≈ W3 photospheric, Ks too bright relative to all three).  The
  natural mechanism with that signature is a **2MASS blend that WISE
  deblends**: the 2MASS PSF is 2.5–3″ and the PSC sums close pairs, while
  AllWISE's PSF fit separates ≳ 2–3″ pairs — Ks is the sum, W1/W2/W3 are the
  star's own.  Hence the four tests below.

Deficit-candidate vet (``candidates.csv`` + ``deferred_lpv.csv``)
-----------------------------------------------------------------
1. **Gaia neighbours** (one Gaia-archive upload join, 10″): brightest
   neighbour, counts within 3″/4″/6″/10″, the nearest neighbour's ΔG.
   Veto ``blend_flux_theft`` (within 6″, G < G_cand + 1.5); veto
   ``unresolved_in_2mass`` (within 4″, G < G_cand + 4: a 4-mag-fainter red
   neighbour still adds tenths in Ks); note ``crowded_field`` (≥ 3 within 6″);
   ``gaia_isolated`` counts stars with no neighbour within 10″ at all.
2. **AllWISE proper** (VizieR ``II/328/allwise``, 3″ at 2010.5): ``nb``/``na``,
   fluxes, rchi2, flags, W1–W4.  Veto ``deblended_component`` (nb > 1 or
   na > 0) and ``saturated_pixels`` (``w1sat``/``w2sat`` > 0 — VizieR does not
   serve them, so they come from IRSA ``allwise_p3as_psd`` in one extra upload).
3. **CatWISE2020** / **unWISE** (3″ at 2015.5 / 2014): independent W1/W2
   against the SAME locus (``locus.json``): ``catwise_photospheric`` (veto),
   ``catwise_confirms_deficit``, ``catwise_missing``.
4. **W3 consistency**: ``w3_excess`` (veto) / ``w3_deficit_consistent`` (note).
5. **2MASS PSC flags** (VizieR ``II/246/out``, 3″ at 2000): veto
   ``tmass_blend`` if the K-band ``Bflg`` > 1 or ``Cflg``(K) ≠ '0'; note
   ``tmass_close_pair`` if ``prox`` < 6″; ``Xflg`` and ``pxCntr`` reported.
6. **Independent higher-resolution Ks** (1.5″ at ~2010): UKIDSS GPS
   (``II/316/gps6``), UKIDSS LAS (``II/319/las9``), VHS (``II/367/vhs_dr5``)
   and VVV (table discovered under ``LIKE '%VVV%'``).
   ``resid_ks_hires = Ks_hires − Ks_2MASS``: ≥ 0.2 mag fainter and within
   0.2 of −resid_w1 → veto ``tmass_ks_contaminated``; within 0.1 of 2MASS →
   ``ks_confirmed_hires``, the deficit is real *in the star* (the deciding test).
7. **G − Ks consistency** (no network; computed by the screen for every row
   from the (G−Ks)(BP−RP) locus per class and Galactic zone): a contaminated
   Ks gives ``resid_gks ≈ −resid_w1``; a screen leaves ``resid_gks ≈ 0``.
   Veto ``ks_too_bright_for_g``; note ``gks_photospheric``.

``vet_verdict`` ∈ {``SURVIVES_VET``, ``SURVIVES_VET_NO_HIRES_KS``, ``BLEND``,
``DEBLENDED_COMPONENT``, ``ALLWISE_PHOTOMETRY_WRONG``, ``W3_INCONSISTENT``,
``TMASS_BLEND``, ``KS_CONTAMINATED``, ``INCONCLUSIVE``}.  SURVIVES_VET requires
Gaia 4″ clean AND 2MASS flags clean AND CatWISE confirms AND NOT
ks_too_bright_for_g AND the high-resolution Ks confirming 2MASS; with no
high-resolution coverage the verdict is ``SURVIVES_VET_NO_HIRES_KS`` (listed
separately).  INCONCLUSIVE rows say why in ``vet_notes``.

Missing-track vet
-----------------
The Gaia × AllWISE cross-match table lacking an entry is **not** the same as
WISE having no source.  The screen's missing fraction by G already shows
39 % at G 4–5 and 14 % at |b| < 10°: that is the cross-match's behaviour on
saturated and crowded sources (the best-neighbour algorithm drops them), not
an absence of 3–5 µm light.  This stage measures the *real* absence rate by a
direct positional match (``nearby`` / ``etz`` candidates plus a uniform random
control): ``wise_source_present_within_6as``, ``wise_source_present_6_to_15as``
(astrometric offset, saturated bright star), ``no_wise_source_within_15as`` —
and for the last group CatWISE / unWISE presence, the nearest AllWISE source's
``cc_flags``, and the **brightest AllWISE source within 3′**.  Run 34054735689
showed every absent star sitting 20–60″ from a D/H/d/h-flagged fragment: a
star inside a bright star's artefact region, or the bright star itself whose
primary entry was lost at saturation, where AllWISE, CatWISE and unWISE all
suppress detections.  Such stars (brightest neighbour W1 < 6, or the star's
own predicted W1 ≈ Ks − 0.05 below the 8.2 saturation regime) are
``ARTEFACT_REGION_OR_SATURATED``, not truly missing.

Transport
---------
Every fetcher is injectable and the offline tests inject all of them.  The
VizieR / IRSA path reuses ``seti.baffle.bright``'s discover-then-quote-once
machinery (``discover_columns``, ``resolve_aliases``, ``select_list``,
``run_vizier``): no column name reaches the wire that TAP_SCHEMA did not
serve, an empty identifier raises before composition, and the unit /
description TAP_SCHEMA serves for every resolved column is recorded in
``vet.json`` so a zero-point question can be answered from the ledger.
Query budget: 9–10 uploads for the deficit vet + 2–6 for the missing vet.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bright import (
    IRSA_TAP,
    VIZIER_TAP,
    _adql_col,
    _lower,
    discover_columns,
    resolve_aliases,
    run_vizier,
    select_list,
)
from .locus import Locus, luminosity_class

VET_VERDICTS = ("SURVIVES_VET", "SURVIVES_VET_NO_HIRES_KS", "BLEND", "DEBLENDED_COMPONENT",
                "ALLWISE_PHOTOMETRY_WRONG", "W3_INCONSISTENT", "TMASS_BLEND", "KS_CONTAMINATED",
                "INCONCLUSIVE")
SURVIVOR_VERDICTS = ("SURVIVES_VET", "SURVIVES_VET_NO_HIRES_KS")
VET_VETOES = ("blend_flux_theft", "unresolved_in_2mass", "deblended_component", "saturated_pixels",
              "catwise_photospheric", "w3_excess", "tmass_blend", "tmass_ks_contaminated",
              "ks_too_bright_for_g")
VET_NOTES = ("crowded_field", "gaia_isolated", "catwise_confirms_deficit", "catwise_missing",
             "w3_deficit_consistent", "gaia_neighbours_unavailable", "allwise_unavailable",
             "allwise_missing", "catwise_unavailable", "tmass_close_pair", "tmass_flags_unavailable",
             "tmass_flags_missing", "ks_confirmed_hires", "ks_hires_ambiguous", "no_hires_ks_coverage",
             "hires_ks_unavailable", "gks_photospheric", "gks_unmeasured", "unwise_zeropoint_suspect",
             "catwise_zeropoint_suspect", "catwise_not_confirming")

QUERY_OK = "QUERY_OK"
QUERY_ZERO = "QUERY_RETURNED_ZERO_ROWS"
QUERY_FAILED = "QUERY_FAILED"

DEFAULTS: dict = {
    "gaia_neighbour_radius_arcsec": 10.0,
    "blend_radius_arcsec": 6.0,
    "blend_dg_max": 1.5,
    "crowded_radius_arcsec": 6.0,
    "crowded_n_min": 3,
    "gaia_2mass_radius_arcsec": 4.0,
    "gaia_2mass_dg_max": 4.0,
    "epochs": {"gaia": 2016.0, "allwise": 2010.5, "catwise": 2015.5, "unwise": 2014.0},
    "allwise": {"table": '"II/328/allwise"', "radius_arcsec": 3.0,
                "xmatch_catalogue": "vizier:II/328/allwise"},
    "catwise": {"table": '"II/365/catwise"', "radius_arcsec": 3.0,
                "xmatch_catalogue": "vizier:II/365/catwise"},
    "unwise": {"table": '"II/363/unwise"', "radius_arcsec": 3.0,
               "xmatch_catalogue": "vizier:II/363/unwise"},
    "irsa_allwise": {"table": "allwise_p3as_psd", "url": IRSA_TAP, "radius_arcsec": 3.0},
    "tmass": {"table": '"II/246/out"', "radius_arcsec": 3.0, "epoch": 2000.0,
              "close_pair_arcsec": 6.0, "xmatch_catalogue": "vizier:II/246/out"},
    "hires_ks": {"radius_arcsec": 1.5, "epoch": 2010.0, "contaminated_min_mag": 0.2,
                 "consistency_tol_mag": 0.2, "confirm_tol_mag": 0.1,
                 "surveys": {"vvv": {"table": "", "discover_like": "%VVV%",
                                     "prefer": ['"II/376/vvv2"', '"II/348/vvv2"']},
                             "gps": {"table": '"II/316/gps6"'},
                             "vhs": {"table": '"II/367/vhs_dr5"'},
                             "las": {"table": '"II/319/las9"'}}},
    "upload_chunk": 4000,
    "xmatch_url": "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync",
    "unwise_flux_system": "vega",
    "zeropoint_check_max_mag": 0.3,
    "indep_err_floor_mag": 0.02,
    "photospheric_nsig": 3.0,
    "deficit_mag": -0.30,
    "deficit_nsig": 3.0,
    "w3_excess_mag": 0.5,
    "w3_deficit_mag": -0.30,
    "w3_err_max": 0.2,
    "gks": {"veto_min_mag": 0.2, "veto_nsig": 3.0, "consistency_tol_mag": 0.25,
            "photospheric_max_mag": 0.1, "photospheric_nsig": 2.0},
    "include_deferred_lpv": True,
    "missing": {"radius_close_arcsec": 6.0, "radius_far_arcsec": 15.0,
                "nearest_radius_arcsec": 60.0, "artefact_radius_arcsec": 180.0,
                "artefact_w1_max": 6.0, "predicted_w1_offset": 0.05, "saturation_w1": 8.2,
                "n_control": 1000, "seed": 20260906},
}

# --- VizieR / IRSA column aliases (canonical -> candidates; resolved against TAP_SCHEMA) ---
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
IRSA_ALLWISE_ALIASES = {
    "designation": ("designation",), "ra": ("ra",), "dec": ("dec",),
    "w1sat": ("w1sat",), "w2sat": ("w2sat",), "nb": ("nb",), "na": ("na",),
    "w1": ("w1mpro",), "w2": ("w2mpro",), "cc_flags": ("cc_flags",),
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
TMASS_PSC_ALIASES = {
    "designation": ("2MASS", "designation"),
    "ra": ("RAJ2000", "ra"), "dec": ("DEJ2000", "dec"),
    "j": ("Jmag", "j_m"), "h": ("Hmag", "h_m"), "ks": ("Kmag", "ks_m", "k_m"),
    "e_ks": ("e_Kmag", "ks_msigcom", "k_msigcom"),
    "qflg": ("Qflg", "ph_qual"), "rflg": ("Rflg", "rd_flg"), "bflg": ("Bflg", "bl_flg"),
    "cflg": ("Cflg", "cc_flg"), "xflg": ("Xflg", "gal_contam"), "aflg": ("Aflg", "mp_flg"),
    "prox": ("prox",), "pxpa": ("pxPA", "pxpa"), "pxcntr": ("pxCntr", "pxcntr"),
}
HIRES_KS_ALIASES = {
    "designation": ("Name", "ID", "objID", "sourceID", "designation", "iauname"),
    "ra": ("RAJ2000", "RA_ICRS", "ra"), "dec": ("DEJ2000", "DE_ICRS", "dec"),
    "ks": ("Ksmag", "Kmag", "Ksmag3", "Kmag3", "Ksap3", "Kap3", "Kspmag", "Kpmag", "kAperMag3",
           "ksAperMag3", "ksmag", "kmag"),
    "e_ks": ("e_Ksmag", "e_Kmag", "e_Ksmag3", "e_Kmag3", "e_Ksap3", "e_Kap3", "e_Kspmag",
             "e_Kpmag", "kAperMag3Err", "ksAperMag3Err"),
    "class": ("mCl", "Kcl", "Kscl", "mergedClass", "cl", "pStar"),
}
# unWISE fluxes are VEGA nanomaggies (Schlafly+2019); an AB-system mirror would need these.
UNWISE_VEGA_OFFSET = {"w1": 2.699, "w2": 3.339}
HIRES_PRIORITY = ("vvv", "gps", "vhs", "las")


def _cfg(cfg: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    out["hires_ks"]["surveys"] = {k: dict(v) for k, v in DEFAULTS["hires_ks"]["surveys"].items()}
    src = (cfg or {}).get("vet", cfg) if isinstance(cfg, dict) else {}
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict) and isinstance(out[k].get(k2), dict):
                    out[k][k2] = {**out[k][k2], **v2}
                else:
                    out[k][k2] = v2
        else:
            out[k] = v
    return out


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    return df[col].astype(object).where(df[col].notna(), "").astype(str).str.strip()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _obj(n: int, index, fill=np.nan) -> pd.Series:
    return pd.Series([fill] * n, index=index, dtype=object)


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
    """TAP upload join composed from discovered (bare) names, quoted once."""
    holes = [k for k in ("ra", "dec") if not resolved.get(k)]
    if holes:
        raise RuntimeError(f"{table}: position columns not resolved: {holes} (resolved: {resolved})")
    ra_c = _adql_col(resolved["ra"], "ra")
    de_c = _adql_col(resolved["dec"], "dec")
    select = select_list(resolved, required=tuple(required), prefix="t.")
    return (f"SELECT u.source_id, {select} FROM TAP_UPLOAD.targets AS u "
            f"JOIN {table} AS t ON 1 = CONTAINS(POINT('ICRS', t.{ra_c}, t.{de_c}), "
            f"CIRCLE('ICRS', u.ra, u.dec, {float(radius_arcsec) / 3600.0:.8f}))")


def table_discovery_query(like: str) -> str:
    return f"SELECT table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE '{like}'"


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


def discover_table(like: str, prefer=(), url: str = VIZIER_TAP, run=None) -> str | None:
    """First TAP_SCHEMA table matching ``like``; ``prefer`` wins when present."""
    run = run or run_vizier
    df = _lower(run(table_discovery_query(like), label=f"tables:{like}", url=url))
    names = [str(x) for x in df["table_name"]] if len(df) and "table_name" in df.columns else []
    bare = {n.strip('"'): n for n in names}
    for p in prefer:
        if p.strip('"') in bare:
            return f'"{p.strip(chr(34))}"'
    return f'"{names[0].strip(chr(34))}"' if names else None


class UploadMatcher:
    """One catalogue around uploaded positions (TAP upload; X-Match fallback).

    ``__call__(positions, radius_arcsec, label)`` takes ``source_id, ra, dec``
    already at the catalogue epoch and returns canonical-named rows with the
    uploaded ``source_id`` attached.  Columns are discovered once from
    TAP_SCHEMA (unit / description kept in ``meta``) and quoted exactly once.
    """

    def __init__(self, name: str, table: str | None, aliases: dict, *, url: str = VIZIER_TAP,
                 xmatch_url: str | None = None, xmatch_catalogue: str | None = None,
                 chunk: int = 4000, discover_like: str | None = None, prefer=()):
        self.name, self.table, self.aliases, self.url = name, table, aliases, url
        self.xmatch_url = xmatch_url or DEFAULTS["xmatch_url"]
        self.xmatch_catalogue = xmatch_catalogue
        self.chunk = int(chunk)
        self.discover_like, self.prefer = discover_like, tuple(prefer)
        self.discovery: dict | None = None
        self.resolved: dict[str, str] = {}
        self.meta: dict[str, dict] = {}
        self.routes_used: list[str] = []

    def _discover(self) -> None:
        if not self.table and self.discover_like:
            self.table = discover_table(self.discover_like, self.prefer, self.url)
            if not self.table:
                raise RuntimeError(f"{self.name}: no table matches {self.discover_like!r}")
        if self.discovery is None:
            self.discovery = discover_columns(self.table, self.url)
            self.resolved = resolve_aliases(self.discovery["names"], self.aliases)
            allmeta = self.discovery.get("meta") or {}
            self.meta = {canon: allmeta.get(actual, {}) for canon, actual in self.resolved.items()}
            if "ra" not in self.resolved or "dec" not in self.resolved:
                raise RuntimeError(f"{self.table}: RA/Dec not resolved from "
                                   f"{self.discovery['names'][:40]}")

    def describe(self) -> dict:
        return {"table": self.table, "resolved": dict(self.resolved), "meta": dict(self.meta),
                "routes_used": list(self.routes_used)}

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
    """Every catalogue matcher the stage uses, keyed by name."""
    c = _cfg(cfg)
    out = {}
    for name, aliases in (("allwise", ALLWISE_ALIASES), ("catwise", CATWISE_ALIASES),
                          ("unwise", UNWISE_ALIASES), ("tmass", TMASS_PSC_ALIASES)):
        sub = c[name]
        out[name] = UploadMatcher(name, sub["table"], aliases, xmatch_url=c["xmatch_url"],
                                  xmatch_catalogue=sub.get("xmatch_catalogue"),
                                  chunk=int(c["upload_chunk"]))
    ir = c["irsa_allwise"]
    out["irsa_allwise"] = UploadMatcher("irsa_allwise", ir["table"], IRSA_ALLWISE_ALIASES,
                                        url=ir.get("url", IRSA_TAP), chunk=int(c["upload_chunk"]))
    for name, sub in c["hires_ks"]["surveys"].items():
        out[name] = UploadMatcher(name, sub.get("table") or None, HIRES_KS_ALIASES,
                                  chunk=int(c["upload_chunk"]),
                                  discover_like=sub.get("discover_like"),
                                  prefer=tuple(sub.get("prefer") or ()))
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


def brightest_per_target(matches: pd.DataFrame, targets: pd.DataFrame, radius_arcsec: float,
                         mag_col: str = "w1") -> pd.DataFrame:
    """Brightest (smallest ``mag_col``) catalogue row per target within ``radius``."""
    cols = ["source_id", "sep_arcsec", "n_within"]
    if matches is None or len(matches) == 0 or "source_id" not in matches.columns:
        return pd.DataFrame(columns=cols)
    m = matches.copy()
    m["source_id"] = pd.to_numeric(m["source_id"], errors="coerce")
    t = targets.set_index("source_id")
    m = m[np.isin(m["source_id"].to_numpy(), t.index.to_numpy())].copy()
    if not len(m):
        return pd.DataFrame(columns=cols)
    tra = t.loc[m["source_id"].to_numpy(), "ra"].to_numpy(float)
    tde = t.loc[m["source_id"].to_numpy(), "dec"].to_numpy(float)
    m["sep_arcsec"] = separation_arcsec(tra, tde, _num(m, "ra"), _num(m, "dec"))
    m = m[m["sep_arcsec"] <= float(radius_arcsec)]
    if not len(m):
        return pd.DataFrame(columns=cols)
    m["_mag"] = _num(m, mag_col)
    m = m.sort_values(["source_id", "_mag"], na_position="last")
    counts = m.groupby("source_id").size().rename("n_within")
    best = m.drop_duplicates("source_id", keep="first").set_index("source_id").join(counts)
    return best.drop(columns=["_mag"]).reset_index()


def gaia_neighbour_stats(neigh: pd.DataFrame | None, cands: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Neighbour counts within 3″/4″/6″/10″, brightest and nearest neighbours, the vetoes."""
    c = _cfg(cfg)
    r_blend, r_2m = float(c["blend_radius_arcsec"]), float(c["gaia_2mass_radius_arcsec"])
    out = pd.DataFrame({"source_id": cands["source_id"].to_numpy()})
    for col in ("gaia_n_3as", "gaia_n_4as", "gaia_n_6as", "gaia_n_10as",
                "gaia_brightest_neighbour_g", "gaia_brightest_neighbour_sep_arcsec",
                "gaia_nearest_neighbour_sep_arcsec", "gaia_nearest_neighbour_dg"):
        out[col] = np.nan
    out["gaia_neighbours_checked"] = False
    out["blend_flux_theft"] = False
    out["unresolved_in_2mass"] = False
    out["crowded_field"] = False
    out["gaia_isolated"] = False
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
        out.loc[i, "gaia_n_4as"] = int((sep <= r_2m).sum())
        out.loc[i, "gaia_n_6as"] = int((sep <= 6.0).sum())
        out.loc[i, "gaia_n_10as"] = int((sep <= 10.0).sum())
        g_c = float(cpos.loc[sid, "phot_g_mean_mag"]) if "phot_g_mean_mag" in cpos else np.nan
        out.loc[i, "gaia_isolated"] = not others.any()
        if others.any():
            g_o, s_o = g[others], sep[others]
            fin = np.isfinite(g_o)
            if fin.any():
                k = int(np.nanargmin(np.where(fin, g_o, np.inf)))
                out.loc[i, "gaia_brightest_neighbour_g"] = g_o[k]
                out.loc[i, "gaia_brightest_neighbour_sep_arcsec"] = s_o[k]
            kn = int(np.argmin(s_o))
            out.loc[i, "gaia_nearest_neighbour_sep_arcsec"] = s_o[kn]
            if np.isfinite(g_c) and np.isfinite(g_o[kn]):
                out.loc[i, "gaia_nearest_neighbour_dg"] = g_o[kn] - g_c
            if np.isfinite(g_c):
                out.loc[i, "blend_flux_theft"] = bool(
                    np.any((s_o <= r_blend) & fin & (g_o < g_c + float(c["blend_dg_max"]))))
                out.loc[i, "unresolved_in_2mass"] = bool(
                    np.any((s_o <= r_2m) & fin & (g_o < g_c + float(c["gaia_2mass_dg_max"]))))
        out.loc[i, "crowded_field"] = bool(
            (sep <= float(c["crowded_radius_arcsec"])).sum() >= int(c["crowded_n_min"]))
    return out


def unwise_vega_mags(df: pd.DataFrame, flux_system: str = "vega") -> pd.DataFrame:
    """Fill ``w1``/``w2`` (Vega) from nanomaggy fluxes when the mirror serves only fluxes.

    unWISE fluxes are **Vega** nanomaggies (Schlafly+2019): ``m = 22.5 − 2.5
    log10(F)``.  ``flux_system="ab"`` additionally subtracts the AB→Vega
    offsets — the mistake of run 34054735689, kept only as an explicit option.
    """
    out = df.copy()
    for b in ("w1", "w2"):
        flux = _num(out, f"{b}flux")
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = 22.5 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan))
            if str(flux_system).lower() == "ab":
                mag = mag - UNWISE_VEGA_OFFSET[b]
            eflux = _num(out, f"e_{b}flux")
            emag = 1.0857 * np.where(flux > 0, eflux / np.where(flux > 0, flux, np.nan), np.nan)
        have = _num(out, b) if b in out.columns else np.full(len(out), np.nan)
        out[b] = np.where(np.isfinite(have), have, mag)
        ehave = _num(out, f"e_{b}") if f"e_{b}" in out.columns else np.full(len(out), np.nan)
        out[f"e_{b}"] = np.where(np.isfinite(ehave), ehave, emag)
    return out


def zeropoint_check(a: np.ndarray, b: np.ndarray, max_offset: float, label: str) -> dict:
    """Median(a − b) over stars measured in both; suspect if beyond ``max_offset``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    rep = {"label": label, "n": int(ok.sum()), "median_offset_mag": None, "suspect": False}
    if ok.sum() >= 3:
        off = float(np.median(a[ok] - b[ok]))
        rep["median_offset_mag"] = off
        rep["robust_sigma_mag"] = float(1.4826 * np.median(np.abs(a[ok] - b[ok] - off)))
        rep["suspect"] = bool(abs(off) > float(max_offset))
    return rep


def locus_residuals(cands: pd.DataFrame, w1, e_w1, w2, e_w2, locus: Locus | None,
                    lcfg: dict | None = None, err_floor: float = 0.0) -> dict:
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
        e = np.nan_to_num(np.asarray(e, dtype=float)) if e is not None else np.zeros(n)
        e = np.maximum(e, float(err_floor))
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
    if locus is None or not any(locus.has(k, "w3") for k in locus.classes()):
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


def tmass_flag_char(flags: pd.Series, pos: int = 2) -> pd.Series:
    """Band character (J=0, H=1, K=2) of a 3-character 2MASS flag string."""
    s = flags.astype(str).str.strip().str.ljust(3, "?")
    return s.str[pos]


def decide(row: pd.Series) -> tuple[str, str, str]:
    """(vet_verdict, vetoes, notes) from one assembled row.  Precedence documented."""
    vetoes, notes = [], []

    def flag(k):
        return bool(row.get(k, False)) and str(row.get(k)).lower() not in ("nan", "none", "")

    if flag("blend_flux_theft"):
        vetoes.append("blend_flux_theft")
    if flag("unresolved_in_2mass"):
        vetoes.append("unresolved_in_2mass")
    if flag("crowded_field"):
        notes.append("crowded_field")
    if flag("gaia_isolated"):
        notes.append("gaia_isolated")
    gaia_ok = flag("gaia_neighbours_checked")
    if not gaia_ok:
        notes.append("gaia_neighbours_unavailable")
    if flag("deblended_component"):
        vetoes.append("deblended_component")
    if flag("saturated_pixels"):
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
    else:
        notes.append("catwise_not_confirming")
    if flag("unwise_zeropoint_suspect"):
        notes.append("unwise_zeropoint_suspect")
    if flag("catwise_zeropoint_suspect"):
        notes.append("catwise_zeropoint_suspect")
    w3s = str(row.get("w3_status", "unmeasured"))
    if w3s == "excess":
        vetoes.append("w3_excess")
    elif w3s == "deficit":
        notes.append("w3_deficit_consistent")
    # 2MASS PSC flags
    ts = str(row.get("tmass_status", "unavailable"))
    if flag("tmass_blend"):
        vetoes.append("tmass_blend")
    if flag("tmass_close_pair"):
        notes.append("tmass_close_pair")
    if ts == "unavailable":
        notes.append("tmass_flags_unavailable")
    elif ts == "missing":
        notes.append("tmass_flags_missing")
    tmass_ok = ts == "matched" and not flag("tmass_blend")
    # high-resolution Ks
    hs = str(row.get("hires_status", "unavailable"))
    if flag("tmass_ks_contaminated"):
        vetoes.append("tmass_ks_contaminated")
    hires_confirms = flag("ks_confirmed_hires")
    if hires_confirms:
        notes.append("ks_confirmed_hires")
    elif hs == "matched":
        notes.append("ks_hires_ambiguous")
    elif hs == "missing":
        notes.append("no_hires_ks_coverage")
    else:
        notes.append("hires_ks_unavailable")
    # G - Ks consistency
    if flag("ks_too_bright_for_g"):
        vetoes.append("ks_too_bright_for_g")
    if flag("gks_photospheric"):
        notes.append("gks_photospheric")
    if flag("gks_unmeasured"):
        notes.append("gks_unmeasured")

    if "blend_flux_theft" in vetoes or "unresolved_in_2mass" in vetoes:
        verdict = "BLEND"
    elif "deblended_component" in vetoes:
        verdict = "DEBLENDED_COMPONENT"
    elif "saturated_pixels" in vetoes or "catwise_photospheric" in vetoes:
        verdict = "ALLWISE_PHOTOMETRY_WRONG"
    elif "w3_excess" in vetoes:
        verdict = "W3_INCONSISTENT"
    elif "tmass_blend" in vetoes:
        verdict = "TMASS_BLEND"
    elif "tmass_ks_contaminated" in vetoes or "ks_too_bright_for_g" in vetoes:
        verdict = "KS_CONTAMINATED"
    elif gaia_ok and aw == "matched" and ind == "confirms_deficit" and tmass_ok:
        if hires_confirms:
            verdict = "SURVIVES_VET"
        elif hs == "missing":
            verdict = "SURVIVES_VET_NO_HIRES_KS"
        else:
            verdict = "INCONCLUSIVE"        # hires Ks answered but neither confirms nor contaminates
    else:
        verdict = "INCONCLUSIVE"
    return verdict, ";".join(vetoes), ";".join(notes)


# ===========================================================================
# Deficit-candidate vet
# ===========================================================================
def vet_deficit_candidates(cands: pd.DataFrame, cfg: dict, locus: Locus | None, ledger: VetLedger,
                           *, gaia_fetcher=None, matchers: dict | None = None,
                           locus_cfg: dict | None = None, report: dict | None = None) -> pd.DataFrame:
    """Assemble every measurement and the verdict for each candidate row."""
    c = _cfg(cfg)
    ep = c["epochs"]
    report = report if report is not None else {}
    cands = cands.reset_index(drop=True).copy()
    cands["source_id"] = pd.to_numeric(cands["source_id"], errors="coerce").astype("int64")
    n = len(cands)
    if n == 0:
        return cands
    matchers = matchers if matchers is not None else default_matchers(c)
    gaia_fetcher = gaia_fetcher or default_gaia_upload_fetcher
    pos_gaia = cands[["source_id", "ra", "dec"]].copy()

    def _pos(epoch: float) -> pd.DataFrame:
        ra_e, de_e = propagate(cands["ra"], cands["dec"], _num(cands, "pmra"), _num(cands, "pmdec"),
                               float(ep["gaia"]), float(epoch))
        return pd.DataFrame({"source_id": cands["source_id"], "ra": ra_e, "dec": de_e})

    # 1. Gaia neighbours (one upload join)
    neigh = _fetch(ledger, "gaia-neighbours (upload)", gaia_fetcher, pos_gaia,
                   float(c["gaia_neighbour_radius_arcsec"]), "gaia-neighbours")
    out = cands.merge(gaia_neighbour_stats(neigh, cands, c), on="source_id", how="left")

    # 2-3, 5-6. One upload per catalogue, PM-propagated to its epoch
    tables: dict = {}

    def _match(name: str, epoch: float, radius: float, label: str, post=None):
        fn = matchers.get(name)
        pos = _pos(epoch)
        if fn is None:
            tables[name] = (False, None)
            return
        raw = _fetch(ledger, label, fn, pos, radius, name)
        if raw is not None and post is not None and len(raw):
            raw = post(raw)
        tables[name] = (raw is not None, nearest_per_target(raw, pos, radius) if raw is not None else None)

    _match("allwise", ep["allwise"], c["allwise"]["radius_arcsec"], f"allwise (upload, {c['allwise']['radius_arcsec']}\")")
    _match("catwise", ep["catwise"], c["catwise"]["radius_arcsec"], f"catwise (upload, {c['catwise']['radius_arcsec']}\")")
    _match("unwise", ep["unwise"], c["unwise"]["radius_arcsec"], f"unwise (upload, {c['unwise']['radius_arcsec']}\")",
           post=lambda raw: unwise_vega_mags(raw, str(c["unwise_flux_system"])))
    _match("tmass", c["tmass"]["epoch"], c["tmass"]["radius_arcsec"], f"2mass psc (upload, {c['tmass']['radius_arcsec']}\")")
    hcfg = c["hires_ks"]
    for name in HIRES_PRIORITY:
        if name in hcfg["surveys"] and name in matchers:
            _match(name, hcfg["epoch"], hcfg["radius_arcsec"], f"hires ks {name} (upload, {hcfg['radius_arcsec']}\")")

    def _attach(name: str, cols: dict):
        avail, near = tables.get(name, (False, None))
        out[f"{name}_status"] = "unavailable" if not avail else "missing"
        for outcol in cols.values():
            out[outcol] = _obj(len(out), out.index)
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
    _attach("tmass", {"designation": "tmass_psc_designation", "ks": "tmass_psc_ks", "e_ks": "tmass_psc_e_ks",
                      "qflg": "tmass_qflg", "rflg": "tmass_rflg", "bflg": "tmass_bflg", "cflg": "tmass_cflg",
                      "xflg": "tmass_xflg", "aflg": "tmass_aflg", "prox": "tmass_prox",
                      "pxcntr": "tmass_pxcntr"})
    for name in HIRES_PRIORITY:
        _attach(name, {"designation": f"hires_{name}_designation", "ks": f"hires_{name}_ks",
                       "e_ks": f"hires_{name}_e_ks", "class": f"hires_{name}_class"})

    out = out.copy()          # defragment after the column-by-column attach

    # 2b. IRSA route for w1sat / w2sat when VizieR did not serve them
    sat_missing = not (np.isfinite(_num(out, "allwise_w1sat")).any() or np.isfinite(_num(out, "allwise_w2sat")).any())
    report["allwise_sat_route"] = "vizier" if not sat_missing else "none"
    if sat_missing and matchers.get("irsa_allwise") is not None and tables.get("allwise", (False,))[0]:
        pos = _pos(ep["allwise"])
        raw = _fetch(ledger, f"irsa allwise_p3as_psd (upload, {c['irsa_allwise']['radius_arcsec']}\")",
                     matchers["irsa_allwise"], pos, float(c["irsa_allwise"]["radius_arcsec"]), "irsa_allwise")
        if raw is not None:
            near = nearest_per_target(raw, pos, float(c["irsa_allwise"]["radius_arcsec"]))
            if len(near):
                near = near.set_index("source_id")
                hit = out["source_id"].isin(near.index).to_numpy()
                idx = out.loc[hit, "source_id"].to_numpy()
                for canon, outcol in (("w1sat", "allwise_w1sat"), ("w2sat", "allwise_w2sat"),
                                      ("nb", "allwise_nb"), ("na", "allwise_na")):
                    if canon in near.columns:
                        cur = out.loc[hit, outcol]
                        new = near.loc[idx, canon].to_numpy()
                        out.loc[hit, outcol] = np.where(pd.to_numeric(cur, errors="coerce").notna(), cur, new)
                report["allwise_sat_route"] = "irsa"

    # AllWISE-proper flags
    nb, na = _num(out, "allwise_nb"), _num(out, "allwise_na")
    out["deblended_component"] = (nb > 1) | (na > 0)
    out["saturated_pixels"] = (_num(out, "allwise_w1sat") > 0) | (_num(out, "allwise_w2sat") > 0)
    out["allwise_flags_available"] = np.isfinite(nb) | np.isfinite(na)
    out["allwise_sat_available"] = np.isfinite(_num(out, "allwise_w1sat"))

    # Zero-point self-checks that no column naming can fool
    zmax = float(c["zeropoint_check_max_mag"])
    zp = {"catwise_vs_allwise_w1": zeropoint_check(_num(out, "catwise_w1"), _num(out, "allwise_w1"), zmax, "catwise_w1 - allwise_w1"),
          "catwise_vs_allwise_w2": zeropoint_check(_num(out, "catwise_w2"), _num(out, "allwise_w2"), zmax, "catwise_w2 - allwise_w2")}
    for b in ("w1", "w2"):
        # CatWISE is the reference; with too few CatWISE matches the AllWISE-proper
        # magnitudes stand in, so a zero-point error can never go unchecked.
        chk = zeropoint_check(_num(out, f"unwise_{b}"), _num(out, f"catwise_{b}"), zmax, f"unwise_{b} - catwise_{b}")
        if chk["n"] < 3:
            chk = zeropoint_check(_num(out, f"unwise_{b}"), _num(out, f"allwise_{b}"), zmax,
                                  f"unwise_{b} - allwise_{b} (catwise absent)")
        zp[f"unwise_vs_catwise_{b}"] = chk
    unwise_suspect = bool(zp["unwise_vs_catwise_w1"]["suspect"] or zp["unwise_vs_catwise_w2"]["suspect"])
    catwise_suspect = bool(zp["catwise_vs_allwise_w1"]["suspect"] or zp["catwise_vs_allwise_w2"]["suspect"])
    report["zeropoint_checks"] = zp
    report["unwise_zeropoint_suspect"] = unwise_suspect
    report["catwise_zeropoint_suspect"] = catwise_suspect
    out["unwise_zeropoint_suspect"] = unwise_suspect
    out["catwise_zeropoint_suspect"] = catwise_suspect
    if unwise_suspect:
        print(f"[baffle-vet] unWISE zero point suspect: median offsets "
              f"{zp['unwise_vs_catwise_w1']['median_offset_mag']} / "
              f"{zp['unwise_vs_catwise_w2']['median_offset_mag']} mag — unWISE does not vote", flush=True)

    # Independent photometry against the same locus
    floor = float(c["indep_err_floor_mag"])
    for name in ("catwise", "unwise"):
        r = locus_residuals(out, _num(out, f"{name}_w1"), _num(out, f"{name}_e_w1"),
                            _num(out, f"{name}_w2"), _num(out, f"{name}_e_w2"), locus, locus_cfg, floor)
        for k, v in r.items():
            out[f"{name}_{k}"] = v
        out[f"{name}_class"] = classify_independent(r, c)
    cat_cls = out["catwise_class"].astype(str).to_numpy()
    un_cls = out["unwise_class"].astype(str).to_numpy()
    if unwise_suspect:
        un_cls = np.full(len(out), "missing", dtype=object)
    if catwise_suspect:
        cat_cls = np.full(len(out), "missing", dtype=object)
    ind = np.where(cat_cls != "missing", cat_cls, un_cls).astype(object)
    both = (cat_cls != "missing") & (un_cls != "missing")
    ind[both & (cat_cls != un_cls) & ((cat_cls == "photospheric") | (un_cls == "photospheric"))] = "photospheric"
    ind[both & (cat_cls != un_cls) & ~((cat_cls == "photospheric") | (un_cls == "photospheric"))] = "ambiguous"
    out["independent_class"] = ind

    # 4. W3 consistency from the re-pulled AllWISE row (falls back to the screen's W3)
    w3 = np.where(np.isfinite(_num(out, "allwise_w3")), _num(out, "allwise_w3"), _num(out, "w3mpro"))
    e_w3 = np.where(np.isfinite(_num(out, "allwise_e_w3")), _num(out, "allwise_e_w3"), _num(out, "w3mpro_error"))
    r3, s3 = w3_residual(out, w3, e_w3, locus, c)
    out["vet_resid_w3"], out["vet_sig_w3"] = r3, s3
    out["w3_status"] = np.where(~np.isfinite(r3), "unmeasured",
                                np.where(r3 > float(c["w3_excess_mag"]), "excess",
                                         np.where(r3 < float(c["w3_deficit_mag"]), "deficit", "normal")))
    if locus is None or not any(locus.has(k, "w3") for k in list(locus.classes())):
        out["w3_status_note"] = "no W3 locus in locus.json (re-screen with the w3mpro_error fallback)"

    # 5. 2MASS PSC flags: K-band character of Bflg / Cflg, prox
    bk = tmass_flag_char(_str(out, "tmass_bflg"), 2)
    ck = tmass_flag_char(_str(out, "tmass_cflg"), 2)
    matched_t = out["tmass_status"].eq("matched").to_numpy()
    out["tmass_bflg_k"] = pd.to_numeric(bk, errors="coerce").to_numpy()
    out["tmass_cflg_k"] = ck.where(matched_t, "").to_numpy()
    out["tmass_blend"] = matched_t & ((pd.to_numeric(bk, errors="coerce").to_numpy() > 1)
                                      | (~ck.isin(["0", "?"]).to_numpy() & (ck != "").to_numpy()))
    out["tmass_close_pair"] = matched_t & (_num(out, "tmass_prox") < float(c["tmass"]["close_pair_arcsec"]))

    # 6. Independent higher-resolution Ks (first survey with a match, in priority order)
    hk = np.full(len(out), np.nan)
    he = np.full(len(out), np.nan)
    hsurvey = _obj(len(out), out.index, "")
    any_avail = any(tables.get(nm, (False,))[0] for nm in HIRES_PRIORITY)
    for nm in HIRES_PRIORITY:
        k = _num(out, f"hires_{nm}_ks")
        take = np.isfinite(k) & ~np.isfinite(hk)
        hk[take] = k[take]
        he[take] = _num(out, f"hires_{nm}_e_ks")[take]
        hsurvey[take] = nm
    out["hires_survey"] = hsurvey
    out["hires_ks"], out["hires_e_ks"] = hk, he
    out["hires_status"] = np.where(np.isfinite(hk), "matched", "missing" if any_avail else "unavailable")
    resid_ks = hk - _num(out, "ks_m")
    out["resid_ks_hires"] = resid_ks
    r1 = _num(out, "resid_w1")
    out["tmass_ks_contaminated"] = (np.isfinite(resid_ks) & (resid_ks >= float(hcfg["contaminated_min_mag"]))
                                    & (np.abs(resid_ks + r1) < float(hcfg["consistency_tol_mag"])))
    out["ks_confirmed_hires"] = np.isfinite(resid_ks) & (np.abs(resid_ks) < float(hcfg["confirm_tol_mag"]))

    # 7. G - Ks consistency (columns come from the screen; NaN when it predates the G-Ks locus)
    gk = c["gks"]
    rg, sg = _num(out, "resid_gks"), _num(out, "sig_gks")
    out["ks_too_bright_for_g"] = ((rg > float(gk["veto_min_mag"])) & (sg > float(gk["veto_nsig"]))
                                  & (np.abs(rg + r1) < float(gk["consistency_tol_mag"])))
    out["gks_photospheric"] = (np.abs(rg) < float(gk["photospheric_max_mag"])) & (np.abs(sg) < float(gk["photospheric_nsig"]))
    out["gks_unmeasured"] = ~np.isfinite(rg)

    verdicts = [decide(row) for _, row in out.iterrows()]
    out["vet_verdict"] = [v[0] for v in verdicts]
    out["vet_vetoes"] = [v[1] for v in verdicts]
    out["vet_notes"] = [v[2] for v in verdicts]
    report["matchers"] = {k: v.describe() for k, v in matchers.items() if isinstance(v, UploadMatcher)}
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
    r_close, r_far, r_near, r_art = (float(mc["radius_close_arcsec"]), float(mc["radius_far_arcsec"]),
                                     float(mc["nearest_radius_arcsec"]), float(mc["artefact_radius_arcsec"]))

    ra_w, de_w = propagate(targets["ra"], targets["dec"], _num(targets, "pmra"), _num(targets, "pmdec"),
                           float(ep["gaia"]), float(ep["allwise"]))
    pos_w = pd.DataFrame({"source_id": targets["source_id"], "ra": ra_w, "dec": de_w})
    raw = _fetch(ledger, f"missing: allwise (upload, {r_far:g}\")", matchers["allwise"], pos_w, r_far,
                 "missing-allwise")
    out = targets.copy()
    out["wise_status"] = "unavailable" if raw is None else "no_wise_source_within_15as"
    out["allwise_sep_arcsec"] = np.nan
    out["allwise_w1"], out["allwise_w2"] = np.nan, np.nan
    out["allwise_cc_flags"] = _obj(len(out), out.index, "")
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
    # the absent group: CatWISE / unWISE presence, the nearest AllWISE source, the
    # brightest AllWISE source within 3' (artefact region of a bright star)
    absent = out["wise_status"].eq("no_wise_source_within_15as").to_numpy()
    out["catwise_present"], out["unwise_present"] = np.nan, np.nan
    out["nearest_allwise_sep_arcsec"] = np.nan
    out["nearest_allwise_cc_flags"] = _obj(len(out), out.index, "")
    out["brightest_allwise_3am_w1"], out["brightest_allwise_3am_sep_arcsec"] = np.nan, np.nan
    out["brightest_allwise_3am_cc_flags"] = _obj(len(out), out.index, "")
    out["predicted_w1"] = _num(out, "ks_m") - float(mc["predicted_w1_offset"])
    out["bright_star_artefact_region"] = False
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
        r = _fetch(ledger, f"missing: allwise brightest (upload, {r_art:g}\")", matchers["allwise"],
                   pos_n, r_art, "missing-allwise-brightest")
        if r is not None:
            best = brightest_per_target(r, pos_n, r_art, "w1")
            if len(best):
                best = best.set_index("source_id")
                hit = out["source_id"].isin(best.index).to_numpy() & absent
                idx = out.loc[hit, "source_id"].to_numpy()
                if "w1" in best.columns:
                    out.loc[hit, "brightest_allwise_3am_w1"] = _num(best.loc[idx], "w1")
                out.loc[hit, "brightest_allwise_3am_sep_arcsec"] = best.loc[idx, "sep_arcsec"].to_numpy()
                if "cc_flags" in best.columns:
                    out.loc[hit, "brightest_allwise_3am_cc_flags"] = best.loc[idx, "cc_flags"].to_numpy()
        bw1 = _num(out, "brightest_allwise_3am_w1")
        out["bright_star_artefact_region"] = absent & ((bw1 < float(mc["artefact_w1_max"]))
                                                       | (_num(out, "predicted_w1") < float(mc["saturation_w1"])))
    art = out["bright_star_artefact_region"].to_numpy(dtype=bool)
    cat_p, un_p = _num(out, "catwise_present"), _num(out, "unwise_present")
    truly = absent & ~art & ~(cat_p > 0) & ~(un_p > 0) & (np.isfinite(cat_p) | np.isfinite(un_p))
    out["truly_missing"] = truly
    out["missing_vet_status"] = np.where(out["wise_status"].eq("unavailable"), "unavailable",
                                         np.where(art, "ARTEFACT_REGION_OR_SATURATED",
                                                  np.where(truly, "truly_missing",
                                                           np.where(absent, "absent_in_allwise_only",
                                                                    out["wise_status"]))))

    counters = out["wise_status"].value_counts().to_dict()
    by_group = {g: d["wise_status"].value_counts().to_dict() for g, d in out.groupby("vet_group")}
    keep = [c_ for c_ in ("source_id", "ra", "dec", "b", "phot_g_mean_mag", "ks_m", "predicted_w1",
                          "parallax", "etz", "nearby", "vet_group", "nearest_allwise_sep_arcsec",
                          "nearest_allwise_cc_flags", "brightest_allwise_3am_w1",
                          "brightest_allwise_3am_sep_arcsec", "brightest_allwise_3am_cc_flags")
            if c_ in out.columns]
    rep.update({
        "counters": {k: int(v) for k, v in counters.items()},
        "counters_by_group": {g: {k: int(v) for k, v in d.items()} for g, d in by_group.items()},
        "status_counters": {k: int(v) for k, v in out["missing_vet_status"].value_counts().to_dict().items()},
        "n_absent_in_allwise": int(absent.sum()),
        "n_artefact_region_or_saturated": int(art.sum()),
        "n_artefact_by_bright_neighbour": int((absent & (_num(out, "brightest_allwise_3am_w1") < float(mc["artefact_w1_max"]))).sum()),
        "n_artefact_by_own_saturation": int((absent & (_num(out, "predicted_w1") < float(mc["saturation_w1"]))).sum()),
        "n_truly_missing": int(truly.sum()),
        "control_no_wise_fraction": (float((out["vet_group"].eq("control")
                                            & absent).sum() / max(1, int(out["vet_group"].eq("control").sum())))
                                     if int(out["vet_group"].eq("control").sum()) else None),
        "artefact_region_or_saturated": out.loc[art, keep].to_dict(orient="records"),
        "truly_missing": out.loc[truly, keep].to_dict(orient="records"),
        "note": ("the Gaia x AllWISE best-neighbour table lacking an entry is the cross-match's "
                 "behaviour on saturated / crowded sources; this is the directly measured "
                 "absence rate.  ARTEFACT_REGION_OR_SATURATED: the brightest AllWISE source within "
                 "3' has W1 < artefact_w1_max, or the star's own predicted W1 (Ks - 0.05) is in the "
                 "saturation regime -- AllWISE, CatWISE and unWISE all suppress detections there"),
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
    "resid_w1", "sig_w1", "resid_w2", "sig_w2", "resid_gks", "sig_gks", "etz", "nearby", "vet_source",
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


def _count_tokens(col: pd.Series, tokens) -> dict:
    parts = col.astype(str).str.split(";")
    return {k: int(parts.map(lambda s, k=k: k in s).sum()) for k in tokens}


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
                 "locus_has_gks": bool(locus is not None and any(locus.has(k, "gks") for k in locus.classes())),
                 "screen_has_gks_columns": bool(len(cands) and "resid_gks" in cands.columns),
                 "n_missing_in": int(len(missing))}
    if len(cands):
        table = vet_deficit_candidates(cands, c, locus, ledger, gaia_fetcher=gaia_fetcher,
                                       matchers=matchers, locus_cfg=locus_cfg, report=rep)
    else:
        table = cands
    if len(table):
        table.to_csv(out / "vet_table.csv", index=False)
        surv = table[table["vet_verdict"].isin(SURVIVOR_VERDICTS)]
        surv.to_csv(out / "vetted_candidates.csv", index=False)
        rep["verdict_counts"] = {k: int((table["vet_verdict"] == k).sum()) for k in VET_VERDICTS}
        rep["veto_counters"] = _count_tokens(table["vet_vetoes"], VET_VETOES)
        rep["note_counters"] = _count_tokens(table["vet_notes"], VET_NOTES)
        rep["n_survivors"] = int((table["vet_verdict"] == "SURVIVES_VET").sum())
        rep["n_survivors_no_hires_ks"] = int((table["vet_verdict"] == "SURVIVES_VET_NO_HIRES_KS").sum())
        rep["n_vetted_candidates_written"] = int(len(surv))
        rep["n_survivors_etz"] = int(surv["etz"].map(lambda x: str(x).lower() == "true").sum()) \
            if "etz" in surv.columns else 0
        rep["n_survivors_nearby"] = int(surv["nearby"].map(lambda x: str(x).lower() == "true").sum()) \
            if "nearby" in surv.columns else 0
        rep["hires_ks_coverage"] = {k: int(v) for k, v in table["hires_status"].value_counts().to_dict().items()}
        rep["hires_ks_by_survey"] = {k: int(v) for k, v in table["hires_survey"].astype(str).value_counts().to_dict().items() if k}
        rep["survivors"] = surv[[k for k in _VET_COMPACT + ("vet_verdict", "vet_notes", "independent_class",
                                                            "catwise_resid_w1", "catwise_resid_w2",
                                                            "unwise_resid_w1", "unwise_resid_w2",
                                                            "w3_status", "gaia_n_4as", "gaia_n_6as",
                                                            "tmass_bflg", "tmass_cflg", "tmass_prox",
                                                            "hires_survey", "hires_ks", "resid_ks_hires")
                                 if k in surv.columns]].to_dict(orient="records")
        rep["allwise_columns_missing"] = [k for k in ("nb", "na", "w1sat", "w2sat", "w1rchi2")
                                          if not pd.to_numeric(table[f"allwise_{k}"], errors="coerce").notna().any()]
        rep["tmass_columns_missing"] = [k for k in ("bflg", "cflg", "xflg", "prox", "pxcntr")
                                        if table[f"tmass_{k}"].astype(str).replace("nan", "").eq("").all()]
    else:
        pd.DataFrame().to_csv(out / "vet_table.csv", index=False)
        pd.DataFrame().to_csv(out / "vetted_candidates.csv", index=False)
        rep["verdict_counts"] = {k: 0 for k in VET_VERDICTS}
        rep["n_survivors"] = rep["n_survivors_no_hires_ks"] = 0

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
    n_surv_all = rep["n_survivors"] + rep["n_survivors_no_hires_ks"]
    if not len(cands):
        rep["verdict_deficit_after_vet"] = "NO_DATA_REACHED" if not (out / "candidates.csv").exists() \
            else "NO_MIDIR_DEFICIT_SURVIVOR"
        rep["note"] = "no candidate rows to vet"
    elif not deficit_reached:
        rep["verdict_deficit_after_vet"] = "NO_DATA_REACHED"
        rep["note"] = "every vet archive query failed; nothing was vetted"
    elif n_surv_all > 0:
        rep["verdict_deficit_after_vet"] = (f"MIDIR_DEFICIT_CANDIDATES_SURVIVE_VET (n={rep['n_survivors']}, "
                                            f"no_hires_ks={rep['n_survivors_no_hires_ks']})")
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


__all__ = ["ALLWISE_ALIASES", "CATWISE_ALIASES", "DEFAULTS", "HIRES_KS_ALIASES", "HIRES_PRIORITY",
           "IRSA_ALLWISE_ALIASES", "SURVIVOR_VERDICTS", "TMASS_PSC_ALIASES", "UNWISE_ALIASES",
           "VET_NOTES", "VET_VERDICTS", "VET_VETOES", "UploadMatcher", "VetLedger",
           "brightest_per_target", "classify_independent", "decide", "default_gaia_upload_fetcher",
           "default_matchers", "discover_table", "gaia_neighbour_stats",
           "gaia_neighbours_upload_query", "locus_residuals", "nearest_per_target", "propagate",
           "run_vet_stage", "select_missing_targets", "separation_arcsec", "table_discovery_query",
           "tmass_flag_char", "unwise_vega_mags", "vet_deficit_candidates", "vet_missing",
           "vizier_upload_query", "w3_residual", "zeropoint_check"]

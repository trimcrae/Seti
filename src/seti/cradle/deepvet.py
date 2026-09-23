"""CRADLE deep vet: trace every CANDIDATE to its mundane readings, at aperture scale.

The ``assess`` stage kills on catalogue columns.  Run 35741356662 left twelve
``CANDIDATE`` rows, three of them with the SIMBAD veto recorded ``untested``
because :func:`seti.vigil.acquire.fetch_simbad_type` returns ``""`` for BOTH
"no SIMBAD object within 5 arcsec" and "the query raised" --- the pipeline could
not tell a clean star from a failed lookup, and said so honestly.  This stage
closes that and the rest of the handoff list, one archive route per check, each
check recorded ``TESTED`` / ``UNTESTED`` with its route status, never silently
passed:

1. **SIMBAD by TAP** (``basic`` + ``otypes`` + ``has_ref``/``ref``): an empty
   cone is ``QUERY_RETURNED_ZERO_ROWS``, a failure is ``QUERY_FAILED``.  The
   star's own type and every other type the object carries, every object within
   the W4 beam, and the titles of every paper that cites the star (the
   literature check: is it already a known extreme debris disk?).
2. **AllWISE, full row** (IRSA ``allwise_p3as_psd``, ``SELECT *``): W3/W4 SNR,
   ``w3nm/w3m``, ``w4nm/w4m`` (fraction of single frames that detect it),
   ``nb``/``na``, ``ext_flg``, ``cc_flags``, ``moon_lev``, the aperture-vs-profile
   magnitude (extended flux the profile fit did not take), and every AllWISE
   neighbour within 30 arcsec.
3. **Chance-blend rate, measured locally**: every AllWISE W3/W4 source in a
   10-arcmin field around the star gives the surface density of sources at
   least as bright as the star's own excess, hence the Poisson probability of
   one sitting inside half a W3 FWHM, where neither ``ext_flg`` nor the profile
   chi^2 can see it.
4. **Legacy Surveys DR10/DR9 Tractor** (NOIRLab Data Lab): every optical source
   within 12 arcsec (galaxies to r ~ 23.5), and the Tractor's *deblended*
   unWISE W3/W4 forced photometry of the star itself.  If the excess belongs to
   a neighbouring galaxy, the forced photometry gives it to the galaxy.
5. **Gaia DR3 neighbours of every magnitude** within 15 arcsec (the ``ages``
   stage only counted neighbours within 4 mag at 6.5 arcsec and 2 mag at 12).
6. **VizieR**: the full footprint of the star (every catalogue within 6 arcsec,
   with its description), plus explicit 12-arcsec cones on VSX, WDS, Gaia DR3
   variability, 2MASX and the published WISE-excess catalogues.
7. **unWISE images** (``unwise.me`` AllWISE coadds; IRSA Atlas fallback): the
   flux-weighted W3 and W4 centroids against W1's, calibrated on the other
   point sources in the same cutout.  An excess from an offset blend drags the
   W3 centroid off the star by (excess fraction) x (offset).

Output: ``results/cradle/deepvet.json`` (per-target verdicts, route ledger,
population blend estimate) and ``results/cradle/deepvet_targets.csv``.
"""

from __future__ import annotations

import io
import json
import math
import re
import tarfile
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
DATALAB_TAP = "https://datalab.noirlab.edu/tap"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

GAIA_EPOCH = 2016.0
ALLWISE_EPOCH = 2010.3
SIMBAD_EPOCH = 2000.0

#: Vega zero points (Jy), Wright et al. 2010 / Jarrett et al. 2011
ZP_JY = {"W1": 309.540, "W2": 171.787, "W3": 31.674, "W4": 8.363}

W3_FWHM = 6.5
W4_FWHM = 12.0

DEFAULT_DEEPVET: dict = {
    "simbad_radius_arcsec": 15.0,
    "self_match_arcsec": 3.0,
    "allwise_radius_arcsec": 30.0,
    "density_radius_arcmin": 10.0,
    "blend_radius_arcsec": W3_FWHM / 2.0,
    "ls_radius_arcsec": 12.0,
    "gaia_radius_arcsec": 15.0,
    "vizier_footprint_radius_arcsec": 6.0,
    "vizier_cone_radius_arcsec": 12.0,
    "cutout_size_px": 161,            # unWISE 2.75"/px -> 7.4 arcmin
    "centroid_radius_px": 2.5,
    "offset_sigma_kill": 3.0,
    "offset_floor_arcsec": 0.35,
    "ls_galaxy_w3_share_kill": 0.3,
    "ls_chi_min": 3.0,
    "w4_snr_threshold_flag": 7.0,
    "frame_fraction_flag": 0.5,
    "wall_budget_s": 5400.0,          # candidates are first in the queue
    "simbad_kill_types": ["G", "AGN", "QSO", "Sy", "LIN", "BLL", "Bla", "YSO", "TTau", "TT*",
                          "Or*", "AGB", "Mi*", "LP*", "C*", "S*", "PN", "pA*", "RG*", "HII",
                          "Ae*", "Be*", "EmO", "Y*O", "Y*?", "IR", "sg*", "s*b", "s*r", "s*y",
                          "HH", "out", "MoC", "cor", "glb", "RNe", "DNe", "Cld"],
    "simbad_blend_types": ["G", "AGN", "QSO", "Sy", "LIN", "BLL", "Bla", "IR", "Rad", "rG",
                           "ClG", "GrG", "LSB", "EmG", "SBG", "bCG", "H2G", "PaG", "YSO",
                           "Y*O", "TT*", "HII", "RNe", "Cld", "smm", "mm", "FIR", "MIR", "NIR"],
    "vizier_cones": {
        "B/vsx/vsx": "variable (VSX)",
        "B/wds/wds": "double star (WDS)",
        "I/358/vclassre": "Gaia DR3 variability class",
        "VII/233/xsc": "2MASS extended source",
        "J/ApJS/225/15": "Cotten & Song 2016 WISE excess",
        "J/ApJS/212/10": "Patel+2014 WISE 12/22 excess",
        "J/MNRAS/471/770": "McDonald+2017 Tycho-Gaia IR excess",
        "J/MNRAS/427/343": "McDonald+2012 Hipparcos IR excess",
        "J/ApJ/910/27": "Moor+2021 extreme debris disks",
    },
}

#: keyword classes for SIMBAD reference titles and VizieR catalogue descriptions
KEYWORDS = {
    "debris_or_excess": r"debris|infrared excess|ir excess|mid-infrared excess|warm dust|"
                        r"exozodi|circumstellar (dis[ck]|dust)|dusty|excess emission|"
                        r"extreme debris",
    "youth": r"young|pre-main|t tauri|moving group|\bassociation\b|yso|star[- ]forming|"
             r"lithium|protoplanetary",
    "multiplicity": r"binar|companion|double|multiple|visual pair|wide pair",
    "variability": r"variab|flare|eclips|periodic|light curve",
    "extragalactic": r"galax|quasar|\bqso\b|\bagn\b|active galactic|redshift",
    "evolved": r"giant|agb|post-agb|evolved|red clump",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def propagate(ra: float, dec: float, pmra: float, pmdec: float, dt_yr: float) -> tuple[float, float]:
    """Linear proper-motion propagation (mas/yr; pmra includes cos dec)."""
    pmra = 0.0 if not np.isfinite(pmra) else pmra
    pmdec = 0.0 if not np.isfinite(pmdec) else pmdec
    d = dec + pmdec * dt_yr / 3.6e6
    c = math.cos(math.radians(0.5 * (dec + d))) or 1e-9
    return ra + pmra * dt_yr / 3.6e6 / c, d


def sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    ra1, dec1, ra2, dec2 = (np.radians(np.asarray(x, dtype=float)) for x in (ra1, dec1, ra2, dec2))
    s = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 3600.0


def mag_to_jy(m, band: str):
    return ZP_JY[band] * 10 ** (-0.4 * np.asarray(m, dtype=float))


def jy_to_mag(f, band: str):
    return -2.5 * np.log10(np.asarray(f, dtype=float) / ZP_JY[band])


def classify_text(text: str) -> list[str]:
    t = (text or "").lower()
    return [k for k, rx in KEYWORDS.items() if re.search(rx, t)]


def type_hits(types, kill_types) -> list[str]:
    """SIMBAD otypes that match a kill list (exact, or prefix for the long labels)."""
    out = []
    for t in types:
        t = str(t or "").strip()
        if not t:
            continue
        for k in kill_types:
            if t == k or (len(k) >= 3 and t.startswith(k)):
                out.append(t)
                break
    return sorted(set(out))


# ---------------------------------------------------------------------------
# archive routes (all injectable)
# ---------------------------------------------------------------------------
@dataclass
class Route:
    label: str
    status: str
    n_rows: int = 0
    error: str = ""
    elapsed_s: float = float("nan")
    data: object = field(default=None, repr=False)

    def ledger(self) -> dict:
        return {"label": self.label, "status": self.status, "n_rows": self.n_rows,
                "error": self.error[:300], "elapsed_s": round(self.elapsed_s, 2)
                if np.isfinite(self.elapsed_s) else None}


#: wall-clock cap per TAP attempt.  pyvo's sync search has no timeout of its
#: own, and the first deepvet dispatch (run 35860601552) sat > 40 min in one
#: stage with nothing written --- an unindexed cone on a billion-row table
#: never returns, it just holds the job.
TAP_TIMEOUT_S = 150.0


def call_with_timeout(fn, timeout_s: float, *args, **kw):
    """Run fn in a DAEMON thread; raise TimeoutError past the cap.

    A daemon thread, not a ThreadPoolExecutor: the executor's workers are joined
    at interpreter exit, so an abandoned hung call would hold the job open anyway.
    """
    import threading  # noqa: PLC0415
    box: dict = {}

    def run():
        try:
            box["v"] = fn(*args, **kw)
        except BaseException as exc:                      # noqa: BLE001
            box["e"] = exc
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(f"no answer in {timeout_s:.0f} s")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _search_df(url: str, adql: str) -> pd.DataFrame:
    import pyvo  # noqa: PLC0415
    return pyvo.dal.TAPService(url).search(adql).to_table().to_pandas()


def tap_query(url: str, adql: str, label: str, retries: int = 3,
              timeout_s: float | None = None) -> Route:
    """Synchronous TAP call: OK / QUERY_RETURNED_ZERO_ROWS / QUERY_FAILED (incl. timeout)."""
    t0 = _time.monotonic()
    last = ""
    tmo = TAP_TIMEOUT_S if timeout_s is None else timeout_s
    for attempt in range(retries):
        try:
            df = call_with_timeout(_search_df, tmo, url, adql)
            n = int(len(df))
            return Route(label, "OK" if n else "QUERY_RETURNED_ZERO_ROWS", n,
                         elapsed_s=_time.monotonic() - t0, data=df)
        except Exception as exc:                          # noqa: BLE001
            last = repr(exc)
            print(f"[deepvet] {label} attempt {attempt + 1}/{retries}: {last[:300]}")
            _time.sleep(2.0 * (attempt + 1))
    return Route(label, "QUERY_FAILED", error=last, elapsed_s=_time.monotonic() - t0)


def http_get(url: str, timeout: float = 120.0, retries: int = 3) -> tuple[int, bytes, str]:
    import requests  # noqa: PLC0415
    last = ""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            return r.status_code, r.content, ""
        except Exception as exc:                          # noqa: BLE001
            last = repr(exc)
            _time.sleep(2.0 * (attempt + 1))
    return -1, b"", last


def _cone(ra_col: str, dec_col: str, ra: float, dec: float, r_deg: float) -> str:
    return (f"CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), "
            f"CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {r_deg:.8f})) = 1")


# --- 1. SIMBAD --------------------------------------------------------------
def simbad_check(t: dict, c: dict, tap=tap_query) -> dict:
    ra, dec = t["ra_2000"], t["dec_2000"]
    r = c["simbad_radius_arcsec"] / 3600.0
    q = ("SELECT b.oid, b.main_id, b.otype, b.sp_type, b.nbref, b.ra, b.dec "
         f"FROM basic AS b WHERE {_cone('b.ra', 'b.dec', ra, dec, r)}")
    rt = tap(SIMBAD_TAP, q, f"simbad_cone:{t['source_id']}")
    out = {"route": rt.ledger(), "status": "UNTESTED"}
    if rt.status == "QUERY_FAILED":
        return out
    out["status"] = "TESTED"
    df = rt.data if rt.data is not None else pd.DataFrame()
    objs = []
    if len(df):
        df = df.copy()
        df["sep_arcsec"] = sep_arcsec(ra, dec, df["ra"], df["dec"])
        df = df.sort_values("sep_arcsec")
        for _, o in df.iterrows():
            objs.append({"main_id": str(o.get("main_id")), "otype": str(o.get("otype")),
                         "sp_type": str(o.get("sp_type") or ""),
                         "nbref": int(o["nbref"]) if pd.notna(o.get("nbref")) else None,
                         "sep_arcsec": round(float(o["sep_arcsec"]), 2),
                         "oid": int(o["oid"]) if pd.notna(o.get("oid")) else None})
    out["objects"] = objs
    self_obj = next((o for o in objs if o["sep_arcsec"] <= c["self_match_arcsec"]), None)
    out["self"] = self_obj
    out["in_simbad"] = self_obj is not None
    types, refs, idents = [], [], []
    if self_obj and self_obj.get("oid") is not None:
        oid = self_obj["oid"]
        r2 = tap(SIMBAD_TAP, f"SELECT otype FROM otypes WHERE oidref = {oid}",
                 f"simbad_otypes:{t['source_id']}")
        if r2.data is not None and len(r2.data):
            types = sorted(set(str(x) for x in r2.data["otype"]))
        r3 = tap(SIMBAD_TAP, "SELECT r.bibcode, r.year, r.title FROM has_ref AS h "
                 f"JOIN ref AS r ON h.oidbibref = r.oidbib WHERE h.oidref = {oid}",
                 f"simbad_refs:{t['source_id']}")
        if r3.data is not None and len(r3.data):
            for _, x in r3.data.iterrows():
                title = str(x.get("title") or "")
                refs.append({"bibcode": str(x.get("bibcode")), "year": x.get("year"),
                             "title": title[:240], "classes": classify_text(title)})
        r4 = tap(SIMBAD_TAP, f"SELECT id FROM ident WHERE oidref = {oid}",
                 f"simbad_ident:{t['source_id']}")
        if r4.data is not None and len(r4.data):
            idents = sorted(str(x) for x in r4.data["id"])
        out["sub_routes"] = [r2.ledger(), r3.ledger(), r4.ledger()]
    out["self_types"] = sorted(set(types + ([self_obj["otype"]] if self_obj else [])))
    out["identifiers"] = idents[:60]
    out["refs"] = refs
    out["n_refs"] = len(refs)
    out["ref_class_counts"] = {k: sum(1 for x in refs if k in x["classes"]) for k in KEYWORDS}
    out["kill_types_hit"] = type_hits(out["self_types"], c["simbad_kill_types"])
    others = [o for o in objs if o is not self_obj]
    out["blend_objects_w3"] = [o for o in others if o["sep_arcsec"] <= W3_FWHM
                               and type_hits([o["otype"]], c["simbad_blend_types"])]
    out["blend_objects_w4"] = [o for o in others if o["sep_arcsec"] <= W4_FWHM
                               and type_hits([o["otype"]], c["simbad_blend_types"])]
    out["double_star_type"] = any(x in ("**", "SB*", "EB*", "El*") for x in out["self_types"])
    return out


# --- 2/3. AllWISE ------------------------------------------------------------
_AW_KEEP = re.compile(r"^(designation|ra|dec|sigra|sigdec|nb|na|ext_flg|cc_flags|ph_qual|"
                      r"var_flg|moon_lev|det_bit|w[1-4](mpro|sigmpro|snr|rchi2|sat|nm|m|mag|"
                      r"sigm|mcor|flg|mag_\d|sigm_\d|sky|sigsk|conf|flux|sigflux|mjdmean)|"
                      r"tmass_key|r_2mass|n_2mass|xscprox)$")


def allwise_check(t: dict, c: dict, tap=tap_query) -> dict:
    ra, dec = t["ra_wise"], t["dec_wise"]
    r = c["allwise_radius_arcsec"] / 3600.0
    rt = tap(IRSA_TAP, f"SELECT * FROM allwise_p3as_psd WHERE {_cone('ra', 'dec', ra, dec, r)}",
             f"allwise_row:{t['source_id']}")
    out = {"route": rt.ledger(), "status": "UNTESTED"}
    if rt.status != "OK":
        return out
    df = rt.data.copy()
    df.columns = [str(x).lower() for x in df.columns]
    df["sep_arcsec"] = sep_arcsec(ra, dec, df["ra"], df["dec"])
    df = df.sort_values("sep_arcsec")
    me = df.iloc[0]
    if float(me["sep_arcsec"]) > c["self_match_arcsec"]:
        out["error"] = f"nearest AllWISE source at {float(me['sep_arcsec']):.2f} arcsec"
        return out
    out["status"] = "TESTED"
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in me.items() if _AW_KEEP.match(k)
           or k == "sep_arcsec"}
    row = {k: (None if isinstance(v, float) and not np.isfinite(v) else
               (v.decode() if isinstance(v, bytes) else v)) for k, v in row.items()}
    out["row"] = row

    def g(k):
        v = pd.to_numeric(pd.Series([row.get(k)]), errors="coerce").iloc[0]
        return float(v) if pd.notna(v) else float("nan")

    ff = {}
    for b in ("w3", "w4"):
        nm, m = g(f"{b}nm"), g(f"{b}m")
        ff[b] = nm / m if m and np.isfinite(m) and m > 0 else float("nan")
    out["frame_fraction"] = ff
    # aperture (8.25" W3, 16.5" W4, curve-of-growth corrected) minus profile-fit
    for b in ("w3", "w4"):
        ap, pr = g(f"{b}mag"), g(f"{b}mpro")
        e = math.hypot(g(f"{b}sigm") if np.isfinite(g(f"{b}sigm")) else 0.0,
                       g(f"{b}sigmpro") if np.isfinite(g(f"{b}sigmpro")) else 0.0)
        out[f"{b}_ap_minus_pro"] = ap - pr if np.isfinite(ap) and np.isfinite(pr) else float("nan")
        out[f"{b}_ap_minus_pro_sig"] = (ap - pr) / e if e > 0 and np.isfinite(ap - pr) else float("nan")
    nbrs = []
    for _, o in df.iloc[1:].iterrows():
        nbrs.append({"designation": str(o.get("designation")), "sep_arcsec": round(float(o["sep_arcsec"]), 2),
                     **{f"w{i}mpro": (None if pd.isna(o.get(f"w{i}mpro")) else float(o.get(f"w{i}mpro")))
                        for i in (1, 2, 3, 4)},
                     **{f"w{i}snr": (None if pd.isna(o.get(f"w{i}snr")) else float(o.get(f"w{i}snr")))
                        for i in (3, 4)}})
    out["neighbours"] = nbrs
    out["n_neighbours_w4beam"] = sum(1 for x in nbrs if x["sep_arcsec"] <= W4_FWHM)
    out["n_neighbours_w3_detected_w4beam"] = sum(
        1 for x in nbrs if x["sep_arcsec"] <= W4_FWHM and (x.get("w3snr") or 0) >= 3)
    return out


def density_check(t: dict, c: dict, tap=tap_query) -> dict:
    """Local surface density of AllWISE sources at least as bright as the excess."""
    ra, dec = t["ra_wise"], t["dec_wise"]
    rad_deg = c["density_radius_arcmin"] / 60.0
    q = ("SELECT ra, dec, w3mpro, w3snr, w4mpro, w4snr, ext_flg FROM allwise_p3as_psd WHERE "
         f"{_cone('ra', 'dec', ra, dec, rad_deg)} AND (w3snr >= 3 OR w4snr >= 3)")
    rt = tap(IRSA_TAP, q, f"allwise_density:{t['source_id']}")
    out = {"route": rt.ledger(), "status": "UNTESTED"}
    if rt.status == "QUERY_FAILED":
        return out
    df = rt.data if rt.data is not None else pd.DataFrame(columns=["ra", "dec", "w3mpro", "w3snr",
                                                                  "w4mpro", "w4snr"])
    df = df.copy()
    df.columns = [str(x).lower() for x in df.columns]
    if len(df):
        df["sep"] = sep_arcsec(ra, dec, df["ra"], df["dec"])
        df = df[df["sep"] > 30.0]                  # the star and its own halo out
    area_as2 = math.pi * ((rad_deg * 3600.0) ** 2 - 30.0 ** 2)
    f3 = float(t.get("W3_excess_jy") or np.nan)
    f4 = float(t.get("W4_excess_jy") or np.nan)
    w3 = pd.to_numeric(df.get("w3mpro"), errors="coerce") if len(df) else pd.Series(dtype=float)
    s3 = pd.to_numeric(df.get("w3snr"), errors="coerce") if len(df) else pd.Series(dtype=float)
    w4 = pd.to_numeric(df.get("w4mpro"), errors="coerce") if len(df) else pd.Series(dtype=float)
    s4 = pd.to_numeric(df.get("w4snr"), errors="coerce") if len(df) else pd.Series(dtype=float)
    out["status"] = "TESTED"
    rb = c["blend_radius_arcsec"]
    res = {}
    for frac in (1.0, 0.5):
        m3 = (s3 >= 3) & (mag_to_jy(w3, "W3") >= frac * f3) if np.isfinite(f3) else s3 < -1
        n3 = int(m3.sum())
        m34 = m3 & (s4 >= 2) & (mag_to_jy(w4, "W4") >= frac * 0.5 * f4) if np.isfinite(f4) else m3 & False
        n34 = int(m34.sum())
        dens3, dens34 = n3 / area_as2, n34 / area_as2
        res[f"frac{frac:g}"] = {
            "n_w3_ge": n3, "n_w3w4_ge": n34,
            "density_w3_per_arcmin2": dens3 * 3600.0,
            "p_blend_w3": 1.0 - math.exp(-dens3 * math.pi * rb ** 2),
            "p_blend_w3w4": 1.0 - math.exp(-dens34 * math.pi * rb ** 2),
        }
    out["field_n_rows"] = int(len(df))
    out["area_arcmin2"] = area_as2 / 3600.0
    out["blend_radius_arcsec"] = rb
    out["by_fraction_of_excess"] = res
    return out


# --- 4. Legacy Surveys ---------------------------------------------------------
_LS_COLS = ("ra, dec, type, ref_cat, ref_id, brick_primary, maskbits, shape_r, "
            "flux_g, flux_r, flux_z, flux_w1, flux_w2, flux_w3, flux_w4, "
            "flux_ivar_r, flux_ivar_w3, flux_ivar_w4")


def ls_check(t: dict, c: dict, tap=tap_query) -> dict:
    ra, dec = t["ra_gaia"], t["dec_gaia"]
    r = c["ls_radius_arcsec"] / 3600.0
    tried = []
    rt = None
    for table in ("ls_dr10.tractor", "ls_dr9.tractor"):
        # q3c first: it is the index Data Lab's tables carry; a bare ADQL
        # CONTAINS may not be translated onto it and then scans the table.
        for where in (f"q3c_radial_query(ra, dec, {ra:.8f}, {dec:.8f}, {r:.8f}) = 1",
                      _cone("ra", "dec", ra, dec, r)):
            rt = tap(DATALAB_TAP, f"SELECT {_LS_COLS} FROM {table} WHERE {where}",
                     f"ls:{table}:{t['source_id']}", retries=2)
            tried.append(rt.ledger())
            if rt.status == "OK":
                break
            if rt.status == "QUERY_RETURNED_ZERO_ROWS":
                break                                  # the spelling worked; no coverage here
        if rt is not None and rt.status == "OK":
            out_table = table
            break
    out = {"routes": tried, "status": "UNTESTED"}
    if rt is None or rt.status != "OK":
        if all(x["status"] == "QUERY_RETURNED_ZERO_ROWS" for x in tried if x["status"] != "QUERY_FAILED") \
                and any(x["status"] == "QUERY_RETURNED_ZERO_ROWS" for x in tried):
            out["status"] = "NOT_IN_FOOTPRINT"
        return out
    df = rt.data.copy()
    df.columns = [str(x).lower() for x in df.columns]
    if "brick_primary" in df:
        bp = df["brick_primary"].astype(str).str.lower().isin(["true", "1", "t"])
        if bp.any():
            df = df[bp]
    df["sep_arcsec"] = sep_arcsec(ra, dec, df["ra"], df["dec"])
    df = df.sort_values("sep_arcsec").reset_index(drop=True)
    out["table"] = out_table
    me_idx = None
    for i, o in df.iterrows():
        if o["sep_arcsec"] <= 1.5:
            me_idx = i
            break
    out["status"] = "TESTED"
    nm2jy = 3.631e-6

    def fx(o, b):
        v = pd.to_numeric(pd.Series([o.get(f"flux_{b}")]), errors="coerce").iloc[0]
        return float(v) * nm2jy if pd.notna(v) else float("nan")

    def fe(o, b):
        v = pd.to_numeric(pd.Series([o.get(f"flux_ivar_{b}")]), errors="coerce").iloc[0]
        return (1.0 / math.sqrt(float(v))) * nm2jy if pd.notna(v) and float(v) > 0 else float("nan")

    srcs = []
    for i, o in df.iterrows():
        srcs.append({"sep_arcsec": round(float(o["sep_arcsec"]), 2), "type": str(o.get("type")),
                     "ref_cat": str(o.get("ref_cat") or "").strip(), "is_target": i == me_idx,
                     "r_mag_ab": (22.5 - 2.5 * math.log10(float(o["flux_r"])) if pd.notna(o.get("flux_r"))
                                  and float(o["flux_r"]) > 0 else None),
                     "shape_r": None if pd.isna(o.get("shape_r")) else float(o.get("shape_r")),
                     "w3_jy": fx(o, "w3"), "w3_err_jy": fe(o, "w3"),
                     "w4_jy": fx(o, "w4"), "w4_err_jy": fe(o, "w4")})
    out["sources"] = srcs
    gal = [s for s in srcs if not s["is_target"] and s["type"].strip().upper() not in ("PSF", "DUP")]
    out["galaxies_w3"] = [s for s in gal if s["sep_arcsec"] <= W3_FWHM]
    out["galaxies_w4"] = [s for s in gal if s["sep_arcsec"] <= W4_FWHM]
    out["any_source_w3"] = [s for s in srcs if not s["is_target"] and s["sep_arcsec"] <= W3_FWHM]
    f3x = float(t.get("W3_excess_jy") or np.nan)
    share = [s["w3_jy"] / f3x for s in out["galaxies_w3"] if np.isfinite(s["w3_jy"]) and f3x > 0]
    out["max_galaxy_w3_share_of_excess"] = max(share) if share else 0.0
    if me_idx is None:
        out["target_found"] = False
        return out
    out["target_found"] = True
    me = df.iloc[me_idx]
    f3, e3 = fx(me, "w3"), fe(me, "w3")
    f4, e4 = fx(me, "w4"), fe(me, "w4")
    p3 = float(mag_to_jy(t["w3mpro"], "W3")) - f3x
    p4 = float(mag_to_jy(t["w4mpro"], "W4")) - float(t.get("W4_excess_jy") or np.nan)
    e3t = math.hypot(e3, 0.05 * f3) if np.isfinite(e3) else float("nan")
    e4t = math.hypot(e4, 0.05 * f4) if np.isfinite(e4) else float("nan")
    out.update({"target_type": str(me.get("type")), "target_ref_cat": str(me.get("ref_cat")),
                "w3_forced_jy": f3, "w3_forced_err_jy": e3, "w3_photosphere_jy": p3,
                "w3_forced_excess_jy": f3 - p3,
                "chi_w3_forced": (f3 - p3) / e3t if e3t and np.isfinite(e3t) else float("nan"),
                "w4_forced_jy": f4, "w4_forced_err_jy": e4, "w4_photosphere_jy": p4,
                "chi_w4_forced": (f4 - p4) / e4t if e4t and np.isfinite(e4t) else float("nan"),
                "w3_forced_over_allwise": f3 / float(mag_to_jy(t["w3mpro"], "W3"))})
    return out


# --- 5. Gaia neighbours ---------------------------------------------------------
def gaia_check(t: dict, c: dict, tap=tap_query) -> dict:
    ra, dec = t["ra_gaia"], t["dec_gaia"]
    r = c["gaia_radius_arcsec"] / 3600.0
    q = ("SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec, ruwe "
         f"FROM gaiadr3.gaia_source WHERE {_cone('ra', 'dec', ra, dec, r)}")
    rt = tap(GAIA_TAP, q, f"gaia_nbrs:{t['source_id']}")
    out = {"route": rt.ledger(), "status": "UNTESTED"}
    if rt.status != "OK":
        return out
    df = rt.data.copy()
    df.columns = [str(x).lower() for x in df.columns]
    df["sep_arcsec"] = sep_arcsec(ra, dec, df["ra"], df["dec"])
    df = df[df["source_id"].astype(str) != str(t["source_id"])].sort_values("sep_arcsec")
    out["status"] = "TESTED"
    out["neighbours"] = [{"source_id": str(o["source_id"]), "sep_arcsec": round(float(o["sep_arcsec"]), 2),
                          "g": None if pd.isna(o["phot_g_mean_mag"]) else round(float(o["phot_g_mean_mag"]), 2),
                          "bp_rp": None if pd.isna(o.get("bp_rp")) else round(float(o["bp_rp"]), 2),
                          "parallax": None if pd.isna(o.get("parallax")) else round(float(o["parallax"]), 3)}
                         for _, o in df.iterrows()]
    out["n_w3beam"] = sum(1 for x in out["neighbours"] if x["sep_arcsec"] <= W3_FWHM)
    out["n_w4beam"] = sum(1 for x in out["neighbours"] if x["sep_arcsec"] <= W4_FWHM)
    return out


# --- 6. VizieR -----------------------------------------------------------------
def vizier_check(t: dict, c: dict, footprint: bool = True) -> dict:
    out = {"status": "UNTESTED", "cones": {}, "footprint": None}
    try:
        from astropy import units as u  # noqa: PLC0415
        from astropy.coordinates import SkyCoord  # noqa: PLC0415
        from astroquery.vizier import Vizier  # noqa: PLC0415
    except Exception as exc:                              # noqa: BLE001
        out["error"] = repr(exc)
        return out
    pos = SkyCoord(t["ra_2000"], t["dec_2000"], unit="deg")
    ok_any = False
    for cat, what in c["vizier_cones"].items():
        try:
            v = Vizier(columns=["**", "+_r"], row_limit=50, timeout=120)
            tl = v.query_region(pos, radius=c["vizier_cone_radius_arcsec"] * u.arcsec, catalog=cat)
            rows = []
            for tab in tl:
                for rr in tab:
                    d = {}
                    for k in tab.colnames[:14]:
                        val = rr[k]
                        try:
                            val = val.item()
                        except Exception:                 # noqa: BLE001
                            pass
                        if isinstance(val, bytes):
                            val = val.decode(errors="replace")
                        if isinstance(val, float) and not np.isfinite(val):
                            val = None
                        d[k] = val if isinstance(val, (int, float, str, type(None))) else str(val)
                    rows.append(d)
            out["cones"][cat] = {"what": what, "status": "OK" if rows else "ZERO_ROWS", "rows": rows[:10]}
            ok_any = True
        except Exception as exc:                          # noqa: BLE001
            out["cones"][cat] = {"what": what, "status": "FAILED", "error": repr(exc)[:200]}
    if footprint:
        try:
            v = Vizier(columns=["+_r"], row_limit=3, timeout=240)
            tl = v.query_region(pos, radius=c["vizier_footprint_radius_arcsec"] * u.arcsec)
            fp = []
            for key in tl.keys():
                tab = tl[key]
                desc = str(tab.meta.get("description", ""))
                rmin = None
                if "_r" in tab.colnames and len(tab):
                    try:
                        rmin = float(np.nanmin(np.asarray(tab["_r"], dtype=float)))
                    except Exception:                     # noqa: BLE001
                        rmin = None
                fp.append({"catalog": str(key), "description": desc[:200], "n": int(len(tab)),
                           "r_min": rmin, "classes": classify_text(desc + " " + str(key))})
            out["footprint"] = fp
            out["footprint_class_counts"] = {k: sum(1 for x in fp if k in x["classes"]) for k in KEYWORDS}
            ok_any = True
        except Exception as exc:                          # noqa: BLE001
            out["footprint_error"] = repr(exc)[:300]
    out["status"] = "TESTED" if ok_any else "UNTESTED"
    return out


# --- 7. unWISE / Atlas images -------------------------------------------------
def _read_fits_bytes(b: bytes):
    from astropy.io import fits  # noqa: PLC0415
    return fits.open(io.BytesIO(b))


def fetch_unwise(ra: float, dec: float, size_px: int, get=http_get) -> tuple[dict, dict]:
    """{band: (image, header)} from unwise.me's AllWISE coadds (tar.gz of FITS)."""
    url = (f"https://unwise.me/cutout_fits?version=allwise&ra={ra:.6f}&dec={dec:.6f}"
           f"&size={size_px}&bands=1234&file_img_m=on")
    code, content, err = get(url, timeout=180)
    info = {"url": url, "http": code, "error": err[:200], "n_bytes": len(content)}
    imgs = {}
    if code != 200 or not content:
        return imgs, info
    try:
        tf = tarfile.open(fileobj=io.BytesIO(content), mode="r:*")
        names = tf.getnames()
        info["members"] = names[:20]
        for n in names:
            low = n.lower()
            if "img-m" not in low and "img_m" not in low:
                continue
            m = re.search(r"-w([1-4])-", low)
            if not m:
                continue
            band = int(m.group(1))
            h = _read_fits_bytes(tf.extractfile(n).read())
            imgs.setdefault(band, (np.array(h[0].data, dtype=float), h[0].header))
    except Exception as exc:                              # noqa: BLE001
        info["parse_error"] = repr(exc)[:200]
    return imgs, info


def fetch_atlas(ra: float, dec: float, size_arcsec: float, get=http_get) -> tuple[dict, dict]:
    """Fallback: IRSA AllWISE Atlas cutouts via IBE (1.375"/px, PSF-convolved)."""
    info = {}
    url = f"https://irsa.ipac.caltech.edu/ibe/search/wise/allwise/p3am_cdd?POS={ra:.6f},{dec:.6f}"
    code, content, err = get(url, timeout=120)
    info["search"] = {"url": url, "http": code, "error": err[:200]}
    imgs = {}
    if code != 200:
        return imgs, info
    ids = re.findall(rb"\b(\d{4}[pm]\d{3}_ac51)\b", content)
    if not ids:
        info["error"] = "no coadd_id in IBE search response"
        return imgs, info
    cid = ids[0].decode()
    info["coadd_id"] = cid
    for band in (1, 2, 3, 4):
        u = (f"https://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/{cid[:2]}/{cid[:4]}/{cid}/"
             f"{cid}-w{band}-int-3.fits?center={ra:.6f},{dec:.6f}&size={size_arcsec:.0f}arcsec")
        code, content, err = get(u, timeout=180)
        info[f"w{band}"] = {"http": code, "error": err[:200]}
        if code == 200 and content:
            try:
                h = _read_fits_bytes(content)
                imgs[band] = (np.array(h[0].data, dtype=float), h[0].header)
            except Exception as exc:                      # noqa: BLE001
                info[f"w{band}"]["parse_error"] = repr(exc)[:200]
    return imgs, info


def centroid(img: np.ndarray, x0: float, y0: float, r: float, r_in: float = 12.0,
             r_out: float = 20.0, iters: int = 5) -> dict:
    """Iterative flux-weighted centroid with a local annulus background."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x, y = float(x0), float(y0)
    ann = ((xx - x0) ** 2 + (yy - y0) ** 2 >= r_in ** 2) & ((xx - x0) ** 2 + (yy - y0) ** 2 < r_out ** 2)
    finite = np.isfinite(img)
    if not (ann & finite).any():
        return {"ok": False}
    sky = float(np.nanmedian(img[ann & finite]))
    rms = float(1.4826 * np.nanmedian(np.abs(img[ann & finite] - sky)))
    flux = float("nan")
    for _ in range(iters):
        ap = ((xx - x) ** 2 + (yy - y) ** 2 <= r ** 2) & finite
        w = np.clip(img[ap] - sky, 0, None)
        s = float(w.sum())
        if s <= 0:
            return {"ok": False, "sky": sky, "rms": rms}
        x, y = float((w * xx[ap]).sum() / s), float((w * yy[ap]).sum() / s)
        flux = float((img[ap] - sky).sum())
    ap = ((xx - x) ** 2 + (yy - y) ** 2 <= (2 * r) ** 2) & finite
    w = np.clip(img[ap] - sky, 0, None)
    s = float(w.sum()) or 1.0
    m2 = float((w * ((xx[ap] - x) ** 2 + (yy[ap] - y) ** 2)).sum() / s)
    npix = int((((xx - x) ** 2 + (yy - y) ** 2) <= r ** 2).sum())
    snr = flux / (rms * math.sqrt(npix)) if rms > 0 else float("nan")
    return {"ok": True, "x": x, "y": y, "sky": sky, "rms": rms, "flux": flux, "snr": snr,
            "m2_px2": m2}


def find_peaks(img: np.ndarray, nsig: float = 30.0, min_sep: int = 8, border: int = 22) -> list:
    """Bright local maxima (for the in-cutout calibration of the W3-W1 offset)."""
    finite = img[np.isfinite(img)]
    if not finite.size:
        return []
    sky = float(np.median(finite))
    rms = float(1.4826 * np.median(np.abs(finite - sky))) or 1.0
    ny, nx = img.shape
    pk = []
    for yy in range(border, ny - border):
        row = img[yy]
        for xx in range(border, nx - border):
            v = row[xx]
            if not np.isfinite(v) or v < sky + nsig * rms:
                continue
            win = img[yy - 2:yy + 3, xx - 2:xx + 3]
            if v >= np.nanmax(win):
                pk.append((float(v), xx, yy))
    pk.sort(reverse=True)
    keep = []
    for v, xx, yy in pk:
        if all((xx - a) ** 2 + (yy - b) ** 2 >= min_sep ** 2 for _, a, b in keep):
            keep.append((v, xx, yy))
    return keep[:60]


def image_check(t: dict, c: dict, fetch_primary=fetch_unwise, fetch_fallback=fetch_atlas) -> dict:
    """W3/W4 centroid offsets from W1, calibrated on the cutout's other point sources."""
    from astropy.wcs import WCS  # noqa: PLC0415
    ra, dec = t["ra_wise"], t["dec_wise"]
    size = int(c["cutout_size_px"])
    imgs, info = fetch_primary(ra, dec, size)
    route = "unwise_allwise"
    if not (1 in imgs and 3 in imgs):
        imgs2, info2 = fetch_fallback(ra, dec, size * 2.75)
        info = {"unwise": info, "atlas": info2}
        imgs, route = imgs2, "irsa_atlas"
    out = {"status": "UNTESTED", "route": route, "info": info}
    if not (1 in imgs and 3 in imgs):
        return out
    res = {}
    img1, h1 = imgs[1]
    w1 = WCS(h1)
    px_as = float(np.sqrt(abs(np.linalg.det(w1.pixel_scale_matrix)))) * 3600.0
    x0, y0 = (float(v) for v in w1.all_world2pix(ra, dec, 0))
    r = c["centroid_radius_px"] * (2.75 / px_as)
    c1 = centroid(img1, x0, y0, r)
    if not c1.get("ok"):
        out["error"] = "W1 centroid failed"
        return out
    ra1, de1 = (float(v) for v in w1.all_pix2world(c1["x"], c1["y"], 0))
    res["w1"] = {"snr": c1["snr"], "m2_arcsec2": c1["m2_px2"] * px_as ** 2,
                 "offset_from_catalogue_arcsec": float(sep_arcsec(ra, dec, ra1, de1))}
    peaks = find_peaks(img1)
    for band in (3, 4):
        if band not in imgs:
            continue
        imgb, hb = imgs[band]
        wb = WCS(hb)
        xb, yb = (float(v) for v in wb.all_world2pix(ra1, de1, 0))
        rb = r * (2.0 if band == 4 else 1.0)
        cb = centroid(imgb, xb, yb, rb, r_in=12.0 * (2.0 if band == 4 else 1.0),
                      r_out=20.0 * (2.0 if band == 4 else 1.0))
        if not cb.get("ok"):
            res[f"w{band}"] = {"ok": False}
            continue
        rab, deb = (float(v) for v in wb.all_pix2world(cb["x"], cb["y"], 0))
        off = float(sep_arcsec(ra1, de1, rab, deb))
        # calibration: the same W_band - W1 offset for the cutout's other bright sources
        cal = []
        for _, px, py in peaks:
            if (px - x0) ** 2 + (py - y0) ** 2 < (30.0 / px_as) ** 2:
                continue
            cc1 = centroid(img1, px, py, r)
            if not cc1.get("ok"):
                continue
            rr1, dd1 = (float(v) for v in w1.all_pix2world(cc1["x"], cc1["y"], 0))
            qx, qy = (float(v) for v in wb.all_world2pix(rr1, dd1, 0))
            ccb = centroid(imgb, qx, qy, rb, r_in=12.0 * (2.0 if band == 4 else 1.0),
                           r_out=20.0 * (2.0 if band == 4 else 1.0))
            if not ccb.get("ok") or not np.isfinite(ccb["snr"]) or ccb["snr"] < 10:
                continue
            rrb, ddb = (float(v) for v in wb.all_pix2world(ccb["x"], ccb["y"], 0))
            cal.append({"offset": float(sep_arcsec(rr1, dd1, rrb, ddb)), "snr": ccb["snr"],
                        "m2": ccb["m2_px2"] * px_as ** 2})
        # the expected centroid noise at this SNR: sigma ~ FWHM / (2.35 SNR) per axis
        fwhm = W4_FWHM if band == 4 else W3_FWHM
        sig_stat = fwhm / (2.355 * max(cb["snr"], 1e-3)) if np.isfinite(cb["snr"]) else float("nan")
        cal_off = np.array([x["offset"] for x in cal if x["snr"] >= 10]) if cal else np.array([])
        sig_sys = float(np.median(cal_off)) if cal_off.size >= 3 else float("nan")
        # Rayleigh: the median 2-D offset of pure noise is 1.177 sigma per axis
        sig_tot = math.hypot(sig_stat, (sig_sys / 1.177) if np.isfinite(sig_sys) else 0.0)
        m2c = np.array([x["m2"] for x in cal if x["snr"] >= 10]) if cal else np.array([])
        res[f"w{band}"] = {"ok": True, "snr": cb["snr"], "offset_from_w1_arcsec": off,
                           "sigma_stat_arcsec": sig_stat, "field_median_offset_arcsec": sig_sys,
                           "n_field_sources": int(cal_off.size),
                           # offset in units of the per-axis sigma: > 3 happens by
                           # chance with P = exp(-4.5) ~ 1 % for a 2-D Gaussian
                           "offset_significance": off / sig_tot if sig_tot > 0 else float("nan"),
                           "m2_arcsec2": cb["m2_px2"] * px_as ** 2,
                           "field_median_m2_arcsec2": float(np.median(m2c)) if m2c.size >= 3 else None}
    out["status"] = "TESTED" if res.get("w3", {}).get("ok") else "UNTESTED"
    out["pixel_arcsec"] = px_as
    out["result"] = res
    return out


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def judge(t: dict, chk: dict, c: dict) -> dict:
    kills, flags, untested, tested = [], [], [], []

    def mark(name, st):
        (tested if st == "TESTED" else untested).append(name)

    sb = chk.get("simbad", {})
    mark("simbad", sb.get("status"))
    if sb.get("status") == "TESTED":
        if sb.get("kill_types_hit"):
            kills.append(f"simbad_type:{'/'.join(sb['kill_types_hit'])}")
        if sb.get("blend_objects_w3"):
            kills.append("simbad_blend_object_in_w3_beam:" +
                         ",".join(f"{o['main_id']}({o['otype']},{o['sep_arcsec']}\")"
                                  for o in sb["blend_objects_w3"]))
        elif sb.get("blend_objects_w4"):
            flags.append("simbad_blend_object_in_w4_beam")
        if sb.get("double_star_type"):
            flags.append("simbad_double_or_binary_type")
        if sb.get("ref_class_counts", {}).get("debris_or_excess"):
            flags.append(f"literature_debris_or_excess_refs:{sb['ref_class_counts']['debris_or_excess']}")
        if sb.get("ref_class_counts", {}).get("youth"):
            flags.append(f"literature_youth_refs:{sb['ref_class_counts']['youth']}")
    aw = chk.get("allwise", {})
    mark("allwise_row", aw.get("status"))
    if aw.get("status") == "TESTED":
        row = aw.get("row", {})
        nb = row.get("nb")
        if nb is not None and float(nb) > 1:
            kills.append(f"allwise_nb:{int(nb)}")
        ext = row.get("ext_flg")
        if ext not in (None, 0, "0"):
            kills.append(f"allwise_ext_flg:{ext}")
        cc = str(row.get("cc_flags") or "0000")
        if len(cc) >= 4 and (cc[2] != "0" or cc[3] != "0"):
            kills.append(f"allwise_cc_flags_w3w4:{cc}")
        for b in ("w3", "w4"):
            ff = aw.get("frame_fraction", {}).get(b)
            if ff is not None and np.isfinite(ff) and ff < c["frame_fraction_flag"]:
                flags.append(f"{b}_frame_detection_fraction:{ff:.2f}")
        s3 = aw.get("w3_ap_minus_pro_sig")
        if s3 is not None and np.isfinite(s3) and s3 < -3:
            flags.append(f"w3_aperture_brighter_than_profile:{aw['w3_ap_minus_pro']:.2f}mag")
        w4snr = row.get("w4snr")
        if w4snr is not None and float(w4snr) < c["w4_snr_threshold_flag"]:
            flags.append(f"w4_at_detection_threshold:snr={float(w4snr):.1f}"
                         " (T_bb set by a W4 flux selected at >=5 sigma; Eddington-biased high, T_bb low)")
        if aw.get("n_neighbours_w3_detected_w4beam"):
            flags.append(f"allwise_w3_neighbour_in_w4_beam:{aw['n_neighbours_w3_detected_w4beam']}")
    ls = chk.get("ls", {})
    mark("legacy_surveys", "TESTED" if ls.get("status") == "TESTED" else "UNTESTED")
    if ls.get("status") == "TESTED":
        if ls.get("max_galaxy_w3_share_of_excess", 0) >= c["ls_galaxy_w3_share_kill"]:
            kills.append(f"ls_galaxy_in_w3_beam_carries_{ls['max_galaxy_w3_share_of_excess']:.0%}_of_excess")
        elif ls.get("galaxies_w3"):
            flags.append(f"ls_galaxy_in_w3_beam:{len(ls['galaxies_w3'])}")
        elif ls.get("galaxies_w4"):
            flags.append(f"ls_galaxy_in_w4_beam:{len(ls['galaxies_w4'])}")
        ch = ls.get("chi_w3_forced")
        if ls.get("target_found") and ch is not None and np.isfinite(ch) and ch < c["ls_chi_min"]:
            kills.append(f"no_w3_excess_in_deblended_forced_photometry:chi={ch:.1f}")
    gn = chk.get("gaia", {})
    mark("gaia_neighbours", gn.get("status"))
    if gn.get("status") == "TESTED" and gn.get("n_w3beam"):
        flags.append(f"gaia_source_in_w3_beam:{gn['n_w3beam']}")
    im = chk.get("image", {})
    mark("image_centroid", im.get("status"))
    if im.get("status") == "TESTED":
        w3 = im["result"].get("w3", {})
        if w3.get("ok"):
            off, sig = w3["offset_from_w1_arcsec"], w3.get("offset_significance", float("nan"))
            if np.isfinite(sig) and sig > c["offset_sigma_kill"] and off > c["offset_floor_arcsec"]:
                kills.append(f"w3_centroid_offset_from_star:{off:.2f}\" ({sig:.1f} sigma)")
    vz = chk.get("vizier", {})
    mark("vizier", vz.get("status"))
    if vz.get("status") == "TESTED":
        for cat, rr in vz.get("cones", {}).items():
            if rr.get("status") == "OK":
                flags.append(f"vizier:{cat}:{rr['what']}")
    dn = chk.get("density", {})
    mark("blend_density", dn.get("status"))
    verdict = "KILLED" if kills else ("SURVIVES_DEEP_VET" if not untested else "SURVIVES_WITH_UNTESTED")
    return {"verdict": verdict, "kills": kills, "flags": flags, "tested": tested, "untested": untested}


# ---------------------------------------------------------------------------
def build_targets(summary: dict, shortlist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cnd in summary.get("candidates", []):
        rows.append({"source_id": str(cnd["source_id"]), "role": "CANDIDATE"})
    for cnd in summary.get("in_cell_age_undetermined", []):
        rows.append({"source_id": str(cnd["source_id"]), "role": "IN_CELL_AGE_UNDETERMINED"})
    for cnd in summary.get("controls", []):
        if cnd.get("found") and cnd.get("source_id") is not None:
            rows.append({"source_id": str(cnd["source_id"]), "role": f"CONTROL:{cnd['name']}"})
    tg = pd.DataFrame(rows).drop_duplicates("source_id")
    sl = shortlist.copy()
    sl["source_id"] = sl["source_id"].astype(str)
    keep = [x for x in ("source_id", "ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag",
                        "w1mpro", "w2mpro", "w3mpro", "w4mpro", "w3_snr", "w4_snr",
                        "W3_excess_jy", "W4_excess_jy", "chi_W3", "chi_W4", "t_bb_k",
                        "log_f_fmax_1gyr", "f_ir") if x in sl.columns]
    tg = tg.merge(sl[keep].drop_duplicates("source_id"), on="source_id", how="left")
    ra_w, de_w, ra_s, de_s = [], [], [], []
    for _, r in tg.iterrows():
        a = propagate(float(r["ra"]), float(r["dec"]), float(r.get("pmra", 0.0)),
                      float(r.get("pmdec", 0.0)), ALLWISE_EPOCH - GAIA_EPOCH)
        b = propagate(float(r["ra"]), float(r["dec"]), float(r.get("pmra", 0.0)),
                      float(r.get("pmdec", 0.0)), SIMBAD_EPOCH - GAIA_EPOCH)
        ra_w.append(a[0]); de_w.append(a[1]); ra_s.append(b[0]); de_s.append(b[1])  # noqa: E702
    tg["ra_gaia"], tg["dec_gaia"] = tg["ra"], tg["dec"]
    tg["ra_wise"], tg["dec_wise"] = ra_w, de_w
    tg["ra_2000"], tg["dec_2000"] = ra_s, de_s
    return tg


def population_blend_estimate(per: list[dict], n_parent: int) -> dict:
    """Expected number of parent stars carrying a blend at least as bright as a typical excess."""
    p = [x["checks"].get("density", {}).get("by_fraction_of_excess", {}).get("frac1", {}).get("p_blend_w3w4")
         for x in per if x["role"] == "CANDIDATE"]
    p = [v for v in p if v is not None and np.isfinite(v)]
    if not p:
        return {"status": "UNTESTED"}
    med = float(np.median(p))
    return {"status": "TESTED", "median_p_blend_w3w4_candidates": med,
            "n_parent_ks_w1_photospheric": n_parent,
            "expected_blends_in_parent_at_candidate_excess": med * n_parent,
            "note": "Poisson rate of an AllWISE W3(+W4) source at least as bright as the "
                    "candidates' own excess within half a W3 FWHM, times the parent size. "
                    "Compare with the number of candidates."}


def run_deepvet(out_dir: str | Path = "results/cradle", conf: dict | None = None,
                routes: dict | None = None, roles: tuple = ("CANDIDATE", "IN_CELL_AGE_UNDETERMINED",
                                                            "CONTROL")) -> dict:
    c = {**DEFAULT_DEEPVET, **(conf or {})}
    rt = {"tap": tap_query, "vizier": vizier_check, "image": image_check, **(routes or {})}
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    shortlist = pd.read_csv(out_dir / "shortlist.csv", low_memory=False)
    tg = build_targets(summary, shortlist)
    tg = tg[tg["role"].map(lambda r: any(r.startswith(x) for x in roles))]
    per = []
    t0 = _time.monotonic()
    skipped = []
    for _, r in tg.iterrows():
        t = r.to_dict()
        if _time.monotonic() - t0 > float(c.get("wall_budget_s", 5400.0)):
            skipped.append({"source_id": t["source_id"], "role": t["role"]})
            continue
        full = t["role"] == "CANDIDATE" or t["role"].startswith("CONTROL")
        print(f"[deepvet] {t['role']} {t['source_id']}")
        chk = {"simbad": simbad_check(t, c, rt["tap"]),
               "allwise": allwise_check(t, c, rt["tap"]),
               "ls": ls_check(t, c, rt["tap"]),
               "gaia": gaia_check(t, c, rt["tap"])}
        if full:
            chk["density"] = density_check(t, c, rt["tap"])
            try:
                chk["image"] = call_with_timeout(rt["image"], 600.0, t, c)
            except Exception as exc:                      # noqa: BLE001
                chk["image"] = {"status": "UNTESTED", "error": repr(exc)[:300]}
            try:
                chk["vizier"] = call_with_timeout(rt["vizier"], 420.0, t, c, footprint=True)
            except Exception as exc:                      # noqa: BLE001
                chk["vizier"] = {"status": "UNTESTED", "error": repr(exc)[:300]}
        v = judge(t, chk, c)
        per.append({"source_id": t["source_id"], "role": t["role"],
                    "t_bb_k": t.get("t_bb_k"), "log_f_fmax_1gyr": t.get("log_f_fmax_1gyr"),
                    **v, "checks": chk})
        # checkpoint after every target: a killed job keeps what it finished
        (out_dir / "deepvet_partial.json").write_text(json.dumps(
            {"stage": "deepvet_partial", "generated_utc": _now(), "n_done": len(per),
             "n_planned": int(len(tg)), "targets": per}, indent=1, default=_json_default))
        print(f"[deepvet]   -> {v['verdict']} kills={v['kills']} untested={v['untested']} "
              f"({_time.monotonic() - t0:.0f}s elapsed)", flush=True)
    n_parent = int(summary.get("funnel", {}).get("n_ks_w1_photospheric") or 0)
    cand = [x for x in per if x["role"] == "CANDIDATE"]
    rep = {
        "stage": "deepvet", "generated_utc": _now(),
        "input_summary_generated_utc": summary.get("generated_utc"),
        "input_verdict": summary.get("verdict"),
        "n_targets": len(per),
        "n_candidates_in": len(cand),
        "candidate_verdicts": {k: sum(1 for x in cand if x["verdict"] == k)
                               for k in ("KILLED", "SURVIVES_DEEP_VET", "SURVIVES_WITH_UNTESTED")},
        "simbad_closed": {"n_candidates_tested": sum(1 for x in cand if "simbad" in x["tested"]),
                          "n_candidates_not_in_simbad": sum(
                              1 for x in cand if x["checks"]["simbad"].get("status") == "TESTED"
                              and not x["checks"]["simbad"].get("in_simbad"))},
        "population_blend": population_blend_estimate(per, n_parent),
        "elapsed_s": round(_time.monotonic() - t0, 1),
        "skipped_on_wall_budget": skipped,
        "targets": per,
    }
    (out_dir / "deepvet.json").write_text(json.dumps(rep, indent=1, default=_json_default))
    flat = []
    for x in per:
        ch = x["checks"]
        flat.append({"source_id": x["source_id"], "role": x["role"], "verdict": x["verdict"],
                     "kills": ";".join(x["kills"]), "flags": ";".join(x["flags"]),
                     "untested": ";".join(x["untested"]),
                     "simbad_main_id": (ch["simbad"].get("self") or {}).get("main_id"),
                     "simbad_types": "|".join(ch["simbad"].get("self_types", [])),
                     "simbad_n_refs": ch["simbad"].get("n_refs"),
                     "w3snr": ch["allwise"].get("row", {}).get("w3snr"),
                     "w4snr": ch["allwise"].get("row", {}).get("w4snr"),
                     "w3_frame_frac": ch["allwise"].get("frame_fraction", {}).get("w3"),
                     "w4_frame_frac": ch["allwise"].get("frame_fraction", {}).get("w4"),
                     "ls_status": ch["ls"].get("status"),
                     "ls_chi_w3_forced": ch["ls"].get("chi_w3_forced"),
                     "ls_n_galaxies_w4beam": len(ch["ls"].get("galaxies_w4", []) or []),
                     "p_blend_w3w4": ch.get("density", {}).get("by_fraction_of_excess", {})
                     .get("frac1", {}).get("p_blend_w3w4"),
                     "w3_offset_arcsec": ch.get("image", {}).get("result", {}).get("w3", {})
                     .get("offset_from_w1_arcsec"),
                     "w3_offset_sig": ch.get("image", {}).get("result", {}).get("w3", {})
                     .get("offset_significance")})
    pd.DataFrame(flat).to_csv(out_dir / "deepvet_targets.csv", index=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "targets"}, indent=1, default=_json_default))
    return rep


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, bytes):
        return o.decode(errors="replace")
    return str(o)


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415
    p = argparse.ArgumentParser(prog="seti.cradle.deepvet")
    p.add_argument("--out-dir", default="results/cradle")
    p.add_argument("--roles", default="CANDIDATE,IN_CELL_AGE_UNDETERMINED,CONTROL")
    a = p.parse_args(argv)
    run_deepvet(a.out_dir, roles=tuple(a.roles.split(",")))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())

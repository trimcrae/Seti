"""Vetting a promoted ZTF target against the systematics the funnel cannot see.

The ZTF funnel (``ztf_live``) judges each alert on what ALeRCE serves: a
difference flux, a real/bogus score, a position.  It never sees the reference
image, the star's neighbours, or the star's own multi-year history.  Those are
exactly where the two false candidates so far came from (docs/tocsin-ztf.md
8c, 8d): a saturated star alerting on its own subtraction residual, and a
faint catalogued star sitting inside a saturated neighbour's PSF and inheriting
that residual as a "+103 % flash" on every visit.

So when the ledger promotes a star, this module pulls, for that one star, what
the funnel could not afford for a hundred thousand:

* **Gaia DR3 within 60"**: every neighbour with its magnitude and proper
  motion, and the target's own row, so a saturated neighbour inside the
  exclusion rule (``targets.bright_neighbour_radius_arcsec``) is named.
* **ALeRCE within 5"**: every ZTF object at the position and its FULL
  detection history since 2018, with the fields the funnel discards ---
  ``distnr`` (distance to the nearest reference-catalogue source),
  ``corrected`` (whether one lies within 1.4") and ``magpsf_corr`` (from
  which the reference magnitude follows).  Amplitudes are re-measured against
  BOTH the target's Gaia baseline and the ZTF reference flux, per band.
* **SIMBAD** and **VSX** within 30": known variables and known types.
* **IRSA's ZTF light-curve service** within 3": the data-release PSF
  photometry of whatever is there, 2018 to now --- the direct record of the
  brightness the difference images were made against.

The verdict is a list of named flags and one classification, each traceable to
a number in the record.  Nothing here changes the ledger: the rule that acts
on a flag lives in the target list (``targets.flag_bright_neighbours``) and
the ledger (``Ledger.remove_targets``), so the action is deterministic and
reproducible without this module's network access.

Runner-only for the acquisition; the analysis (:func:`analyse`) is pure and
tested offline.
"""

from __future__ import annotations

import csv
import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ledger import Ledger
from .photometry import ab_to_njy, njy_to_ab
from .run import _repo_root, _utc, _write_json, load_targets
from .targets import GSPC_MAG_COLUMN, bright_neighbour_radius_arcsec, propagate_pm
from .ztf_live import ZTF_FID_BAND, AlerceZtfAPI, _isdiffpos_sign, _num, ztf_config

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
IRSA_LC = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

DEFAULTS = {
    "gaia_radius_arcsec": 60.0,
    "alerce_radius_arcsec": 5.0,
    "catalogue_radius_arcsec": 30.0,
    "irsa_radius_arcsec": 3.0,
    #: |a - 1| below this, in a band with at least `self_flux_min_n` detections,
    #: means the difference flux IS the star's own flux: the reference lacks it.
    "self_flux_tolerance": 0.15,
    "self_flux_min_n": 3,
    #: A ZTF reference this much brighter than the target's Gaia magnitude is
    #: another star's flux.
    "reference_mismatch_mag": 2.0,
    "max_detections_recorded": 600,
    "results_subdir": "vet",
    "timeout_s": 120.0,
}


def _clean_row(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if isinstance(v, (np.floating, float)):
            out[str(k)] = float(v) if math.isfinite(float(v)) else None
        elif isinstance(v, (np.integer,)):
            out[str(k)] = int(v)
        elif isinstance(v, (np.bool_,)):
            out[str(k)] = bool(v)
        elif isinstance(v, bytes):
            out[str(k)] = v.decode("utf-8", "replace")
        else:
            out[str(k)] = v
    return out


def _sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = (np.radians(float(x)) for x in (ra1, dec1, ra2, dec2))
    s = (math.sin((d2 - d1) / 2) ** 2
         + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2)
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(max(0.0, s))))) * 3600.0


def mjd_to_jyear(mjd: float) -> float:
    return 2000.0 + (float(mjd) - 51544.5) / 365.25


# ---------------------------------------------------------------------------
# Queries (each a small ADQL/HTTP request; all injectable for tests)
# ---------------------------------------------------------------------------
def gaia_cone_adql(ra: float, dec: float, radius_arcsec: float) -> str:
    r_deg = float(radius_arcsec) / 3600.0
    return ("SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, "
            "phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, ruwe, "
            "phot_variable_flag, non_single_star, "
            "DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', "
            f"{float(ra):.7f}, {float(dec):.7f})) * 3600.0 AS sep_arcsec_2016 "
            "FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {float(ra):.7f}, {float(dec):.7f}, {r_deg:.8f}))")


def gaia_target_adql(source_id: str) -> str:
    syn = ", ".join(f"s.{c} AS {c}" for c in GSPC_MAG_COLUMN.values())
    return ("SELECT g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, g.pmra, g.pmdec, "
            "g.pmra_error, g.pmdec_error, g.phot_g_mean_mag, g.bp_rp, g.ruwe, "
            f"g.phot_variable_flag, g.non_single_star, {syn} "
            "FROM gaiadr3.gaia_source AS g LEFT JOIN gaiadr3.synthetic_photometry_gspc AS s "
            f"ON s.source_id = g.source_id WHERE g.source_id = {int(source_id)}")


def simbad_cone_adql(ra: float, dec: float, radius_arcsec: float) -> str:
    r_deg = float(radius_arcsec) / 3600.0
    return ("SELECT main_id, otype, sp_type, ra, dec, "
            f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {float(ra):.7f}, {float(dec):.7f}))"
            " * 3600.0 AS sep_arcsec FROM basic WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {float(ra):.7f}, {float(dec):.7f}, {r_deg:.8f}))")


def vsx_cone_adql(ra: float, dec: float, radius_arcsec: float) -> str:
    r_deg = float(radius_arcsec) / 3600.0
    return ('SELECT * FROM "B/vsx/vsx" WHERE 1 = CONTAINS(POINT(\'ICRS\', RAJ2000, DEJ2000), '
            f"CIRCLE('ICRS', {float(ra):.7f}, {float(dec):.7f}, {r_deg:.8f}))")


def irsa_lightcurve_url(ra: float, dec: float, radius_arcsec: float) -> str:
    r_deg = float(radius_arcsec) / 3600.0
    return f"{IRSA_LC}?POS=CIRCLE {float(ra):.6f} {float(dec):.6f} {r_deg:.6f}&FORMAT=csv"


def _pyvo_transport(tap_url: str, timeout_s: float):
    """``adql -> list[dict]`` through pyvo's sync endpoint with a hard timeout."""

    def _run(adql: str) -> list[dict]:
        import pyvo  # noqa: PLC0415  runner-only
        import requests  # noqa: PLC0415

        class _S(requests.Session):
            def request(self, *a, **kw):                    # noqa: D102
                kw.setdefault("timeout", float(timeout_s))
                return super().request(*a, **kw)

        svc = pyvo.dal.TAPService(tap_url, session=_S())
        tab = svc.run_sync(adql).to_table()
        rows = []
        for r in tab:
            rows.append({str(c).lower(): (r[c].item() if hasattr(r[c], "item") else r[c])
                         for c in tab.colnames})
        return [_clean_row(r) for r in rows]

    return _run


def _http_text(timeout_s: float):
    def _get(url: str) -> str:
        import requests  # noqa: PLC0415

        r = requests.get(url, timeout=float(timeout_s))
        r.raise_for_status()
        return r.text

    return _get


@dataclass
class Services:
    """Every network dependency of a vet, replaceable one by one."""

    gaia: object = None      # adql -> list[dict]
    simbad: object = None
    vizier: object = None
    alerce: object = None    # AlerceZtfAPI-like: _get(path, params), detections(oid), non_detections(oid)
    irsa: object = None      # url -> csv text
    log: list[dict] = field(default_factory=list)

    @classmethod
    def live(cls, z: dict, timeout_s: float) -> Services:
        return cls(gaia=_pyvo_transport(GAIA_TAP, timeout_s),
                   simbad=_pyvo_transport(SIMBAD_TAP, timeout_s),
                   vizier=_pyvo_transport(VIZIER_TAP, timeout_s),
                   alerce=AlerceZtfAPI(z.get("alerce_ztf_api", "https://api.alerce.online/ztf/v1"),
                                       timeout=timeout_s),
                   irsa=_http_text(timeout_s))

    def call(self, name: str, fn, *args):
        """Run one service call, recording success or the verbatim error."""
        t0 = time.monotonic()
        try:
            out = fn(*args)
            self.log.append({"service": name, "ok": True,
                             "elapsed_s": round(time.monotonic() - t0, 1)})
            return out, None
        except Exception as exc:                            # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"[:400]
            self.log.append({"service": name, "ok": False, "error": err,
                             "elapsed_s": round(time.monotonic() - t0, 1)})
            return None, err


def alerce_objects_near(api, ra: float, dec: float, radius_arcsec: float) -> list[dict]:
    payload = api._get("objects", [("ra", f"{float(ra):.6f}"), ("dec", f"{float(dec):.6f}"),
                                   ("radius", f"{float(radius_arcsec):.2f}"),
                                   ("page_size", "50"), ("page", "1"), ("count", "false")])
    items = (payload or {}).get("items") if isinstance(payload, dict) else payload
    return [r for r in (items or []) if isinstance(r, dict)]


def parse_irsa_csv(text: str) -> list[dict]:
    if not text or not text.strip():
        return []
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("\\")]
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = []
    for r in reader:
        rows.append({str(k).strip().lower(): (v.strip() if isinstance(v, str) else v)
                     for k, v in r.items() if k is not None})
    return rows


# ---------------------------------------------------------------------------
# The analysis (pure)
# ---------------------------------------------------------------------------
def _median_mad(vals: list[float]) -> tuple[float | None, float | None]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None, None
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med)))


def summarise_object(obj: dict, dets: list[dict], nondets: list[dict],
                     target: dict, p_ra: float, p_dec: float,
                     max_recorded: int = 600) -> dict:
    """One ALeRCE object: its history re-measured against both baselines."""
    rec = {"oid": str(obj.get("oid")),
           "meanra": _num(obj.get("meanra")), "meandec": _num(obj.get("meandec")),
           "ndet": obj.get("ndet"), "firstmjd": _num(obj.get("firstmjd")),
           "lastmjd": _num(obj.get("lastmjd")), "corrected": obj.get("corrected"),
           "stellar": obj.get("stellar"), "class": obj.get("class"),
           "probability": _num(obj.get("probability")),
           "n_detections_pulled": len(dets), "n_non_detections_pulled": len(nondets)}
    if rec["meanra"] is not None and rec["meandec"] is not None:
        rec["sep_from_propagated_target_arcsec"] = _sep_arcsec(
            p_ra, p_dec, rec["meanra"], rec["meandec"])
    distnr, corrected_flags = [], []
    a_gaia: dict[str, list[float]] = {}
    a_ref: dict[str, list[float]] = {}
    ref_mag: dict[str, list[float]] = {}
    diff_mag: dict[str, list[float]] = {}
    drb = []
    years: dict[str, int] = {}
    kept_rows = []
    for d in sorted(dets, key=lambda r: _num(r.get("mjd")) or 0.0):
        mjd = _num(d.get("mjd"))
        mag = _num(d.get("magpsf"))
        sign = _isdiffpos_sign(d.get("isdiffpos"))
        try:
            band = ZTF_FID_BAND.get(int(d.get("fid")), "")
        except (TypeError, ValueError):
            band = ""
        if mjd is None or mag is None or sign is None or not band:
            continue
        dn = _num(d.get("distnr"))
        if dn is not None:
            distnr.append(dn)
        corr = d.get("corrected")
        if isinstance(corr, str):
            corr = corr.strip().lower() in ("t", "true", "1")
        if corr is not None:
            corrected_flags.append(bool(corr))
        rel = _num(d.get("drb"))
        if rel is None:
            rel = _num(d.get("rb"))
        if rel is not None:
            drb.append(rel)
        yr = str(int(mjd_to_jyear(mjd)))
        years[yr] = years.get(yr, 0) + 1
        dflux = sign * float(ab_to_njy(mag))
        diff_mag.setdefault(band, []).append(mag)
        row = {"mjd": mjd, "band": band, "magpsf": mag, "isdiffpos": sign,
               "distnr": dn, "corrected": corr, "magpsf_corr": _num(d.get("magpsf_corr")),
               "drb": rel, "dubious": d.get("dubious"), "candid": d.get("candid")}
        col = GSPC_MAG_COLUMN.get(band)
        base = _num(target.get(col)) if col else None
        if base is not None:
            ag = dflux / float(ab_to_njy(base))
            a_gaia.setdefault(band, []).append(ag)
            row["a_vs_gaia"] = ag
        mc = _num(d.get("magpsf_corr"))
        if corr and mc is not None:
            total = float(ab_to_njy(mc))
            ref = total - dflux
            if ref > 0:
                rm = float(njy_to_ab(ref))
                ref_mag.setdefault(band, []).append(rm)
                a_ref.setdefault(band, []).append(dflux / ref)
                row["reference_mag"] = rm
                row["a_vs_reference"] = dflux / ref
        kept_rows.append(row)
    rec["detections_by_year"] = dict(sorted(years.items()))
    rec["distnr_arcsec"] = {"min": min(distnr) if distnr else None,
                            "median": float(np.median(distnr)) if distnr else None,
                            "max": max(distnr) if distnr else None, "n": len(distnr)}
    rec["corrected_fraction"] = (sum(corrected_flags) / len(corrected_flags)
                                 if corrected_flags else None)
    rec["drb"] = {"median": float(np.median(drb)) if drb else None,
                  "min": min(drb) if drb else None}
    rec["difference_mag"] = {b: dict(zip(("median", "mad"), _median_mad(v), strict=True))
                             for b, v in diff_mag.items()}
    rec["reference_mag"] = {b: dict(zip(("median", "mad"), _median_mad(v), strict=True),
                                    n=len(v)) for b, v in ref_mag.items()}
    rec["amplitude_vs_gaia"] = {b: dict(zip(("median", "mad"), _median_mad(v), strict=True),
                                        n=len(v)) for b, v in a_gaia.items()}
    rec["amplitude_vs_reference"] = {b: dict(zip(("median", "mad"), _median_mad(v),
                                                 strict=True), n=len(v))
                                     for b, v in a_ref.items()}
    rec["n_detections_recorded"] = min(len(kept_rows), int(max_recorded))
    rec["detections"] = kept_rows[-int(max_recorded):]
    return rec


def analyse(source_id: str, target: dict, epoch_jyear: float,
            gaia_neighbours: list[dict], alerce_objects: list[dict],
            simbad_rows: list[dict], vsx_rows: list[dict], irsa_rows: list[dict],
            saturation_mag: float | None, ledger_rec: dict | None = None,
            opts: dict | None = None) -> dict:
    """Flags and a classification, from the records alone."""
    o = {**DEFAULTS, **(opts or {})}
    flags: list[str] = []
    why: list[str] = []
    sid = str(source_id)
    out: dict = {"source_id": sid, "epoch_jyear": float(epoch_jyear), "target": target}

    # -- the target at the epoch ------------------------------------------
    ra0, dec0 = _num(target.get("ra")), _num(target.get("dec"))
    p_ra, p_dec = ra0, dec0
    if ra0 is not None and dec0 is not None:
        pr, pd_ = propagate_pm(np.array([ra0]), np.array([dec0]),
                               np.array([_num(target.get("pmra")) or 0.0]),
                               np.array([_num(target.get("pmdec")) or 0.0]),
                               to_epoch=float(epoch_jyear))
        p_ra, p_dec = float(pr[0]), float(pd_[0])
    out["target_propagated"] = {"ra": p_ra, "dec": p_dec}
    pm = math.hypot(_num(target.get("pmra")) or 0.0, _num(target.get("pmdec")) or 0.0)
    out["pm_mas_yr"] = pm
    out["drift_since_2018_arcsec"] = pm * (float(epoch_jyear) - 2018.5) / 1000.0

    # -- Gaia neighbours ---------------------------------------------------
    neigh = []
    for r in gaia_neighbours:
        if str(r.get("source_id")) == sid:
            continue
        nra, ndec = _num(r.get("ra")), _num(r.get("dec"))
        if nra is None or ndec is None:
            continue
        npr, npd = propagate_pm(np.array([nra]), np.array([ndec]),
                                np.array([_num(r.get("pmra")) or 0.0]),
                                np.array([_num(r.get("pmdec")) or 0.0]),
                                to_epoch=float(epoch_jyear))
        g = _num(r.get("phot_g_mean_mag"))
        row = {**_clean_row(r), "sep_arcsec_epoch": _sep_arcsec(p_ra, p_dec, npr[0], npd[0])}
        if saturation_mag is not None and g is not None:
            rad = float(bright_neighbour_radius_arcsec(np.array([g]), saturation_mag)[0])
            row["exclusion_radius_arcsec"] = rad
            row["inside_exclusion_radius"] = bool(rad > 0 and row["sep_arcsec_epoch"] <= rad)
        neigh.append(row)
    neigh.sort(key=lambda r: r["sep_arcsec_epoch"])
    out["gaia_neighbours"] = {"n": len(neigh), "rows": neigh}
    offenders = [r for r in neigh if r.get("inside_exclusion_radius")]
    if offenders:
        b = min(offenders, key=lambda r: r["sep_arcsec_epoch"] / r["exclusion_radius_arcsec"])
        flags.append("saturated_neighbour")
        why.append(f"Gaia DR3 {b['source_id']} (G {b['phot_g_mean_mag']:.2f}) lies "
                   f"{b['sep_arcsec_epoch']:.2f}\" from the target at the event epoch, inside "
                   f"its {b['exclusion_radius_arcsec']:.1f}\" exclusion radius for a "
                   f"{saturation_mag:.1f} mag saturation limit")
        out["saturated_neighbour"] = b

    # -- ALeRCE objects ----------------------------------------------------
    objs = []
    for rec in alerce_objects:
        objs.append(rec)
    out["alerce"] = {"n_objects": len(objs), "objects": objs}
    target_mag = {b: _num(target.get(c)) for b, c in GSPC_MAG_COLUMN.items()}
    n_self = 0
    for rec in objs:
        for b, s in (rec.get("amplitude_vs_gaia") or {}).items():
            if (s.get("n", 0) >= int(o["self_flux_min_n"]) and s.get("median") is not None
                    and abs(s["median"] - 1.0) <= float(o["self_flux_tolerance"])):
                n_self += 1
                why.append(f"{rec['oid']} {b}: median difference flux is "
                           f"{s['median']:.2f} x the target's own Gaia flux over {s['n']} "
                           f"detections (MAD {s['mad']:.2f}) --- the difference image "
                           f"holds the whole star, i.e. the reference does not")
        for b, s in (rec.get("reference_mag") or {}).items():
            tm = target_mag.get(b)
            if s.get("median") is None:
                continue
            if tm is not None and tm - s["median"] >= float(o["reference_mismatch_mag"]):
                if "reference_brighter_than_target" not in flags:
                    flags.append("reference_brighter_than_target")
                why.append(f"{rec['oid']} {b}: ZTF's reference source at the position is "
                           f"{s['median']:.2f} mag against the target's Gaia synthetic "
                           f"{tm:.2f} --- the reference flux is another star's")
            if saturation_mag is not None and s["median"] < float(saturation_mag):
                if "reference_saturated" not in flags:
                    flags.append("reference_saturated")
                why.append(f"{rec['oid']} {b}: the reference source is {s['median']:.2f} mag, "
                           f"brighter than the {saturation_mag:.1f} mag saturation limit")
        yrs = rec.get("detections_by_year") or {}
        if len(yrs) >= 3 and sum(yrs.values()) >= 30:
            amps = [s for s in (rec.get("amplitude_vs_gaia") or {}).values()
                    if s.get("median") and s.get("mad") is not None and s["n"] >= 10]
            if amps and all(s["mad"] / abs(s["median"]) < 0.15 for s in amps):
                if "persistent_constant_residual" not in flags:
                    flags.append("persistent_constant_residual")
                why.append(f"{rec['oid']}: {sum(yrs.values())} detections across "
                           f"{len(yrs)} calendar years at a near-constant amplitude "
                           f"(MAD/median < 0.15) --- a fixed subtraction residual, not "
                           f"a transient")
    if n_self:
        flags.append("self_flux")

    # -- catalogues --------------------------------------------------------
    sb = []
    for r in simbad_rows:
        sb.append(_clean_row(r))
    out["simbad"] = {"n": len(sb), "rows": sb}
    var_types = ("V*", "EB", "RR", "Ce", "Mi", "Ro", "BY", "Er", "Fl", "Pu", "dS", "LP", "SN",
                 "No", "CV", "Sy", "Ir")
    for r in sb:
        ot = str(r.get("otype") or "")
        sep = _num(r.get("sep_arcsec"))
        if sep is not None and sep <= 3.0 and any(t in ot for t in var_types):
            flags.append("known_variable_simbad")
            why.append(f"SIMBAD {r.get('main_id')} ({ot}) at {sep:.2f}\"")
            break
    vs = [_clean_row(r) for r in vsx_rows]
    out["vsx"] = {"n": len(vs), "rows": vs}
    for r in vs:
        ra_v = _num(r.get("raj2000"))
        de_v = _num(r.get("dej2000"))
        sep = (_sep_arcsec(p_ra, p_dec, ra_v, de_v)
               if ra_v is not None and de_v is not None else _num(r.get("sep_arcsec")))
        r["sep_arcsec"] = sep
        if sep is not None and sep <= 3.0:
            flags.append("known_variable_vsx")
            why.append(f"VSX {r.get('name')} ({r.get('type')}) at {sep:.2f}\"")
            break

    # -- IRSA light curve --------------------------------------------------
    by_oid: dict[str, dict] = {}
    for r in irsa_rows:
        oid = str(r.get("oid"))
        m = _num(r.get("mag"))
        mjd = _num(r.get("mjd"))
        if m is None or mjd is None:
            continue
        d = by_oid.setdefault(oid, {"oid": oid, "filter": r.get("filtercode"), "n": 0,
                                    "mags": [], "mjd_min": mjd, "mjd_max": mjd,
                                    "ra": _num(r.get("ra")), "dec": _num(r.get("dec"))})
        d["n"] += 1
        d["mags"].append(m)
        d["mjd_min"] = min(d["mjd_min"], mjd)
        d["mjd_max"] = max(d["mjd_max"], mjd)
    lc = []
    for d in by_oid.values():
        med, mad = _median_mad(d.pop("mags"))
        d["mag_median"] = med
        d["mag_mad"] = mad
        if d.get("ra") is not None and d.get("dec") is not None:
            d["sep_arcsec_epoch"] = _sep_arcsec(p_ra, p_dec, d["ra"], d["dec"])
        lc.append(d)
    lc.sort(key=lambda d: -d["n"])
    out["irsa_lightcurve"] = {"n_rows": len(irsa_rows), "objects": lc}
    for d in lc:
        band = {"zg": "g", "zr": "r", "zi": "i"}.get(str(d.get("filter")), None)
        tm = target_mag.get(band) if band else None
        if (d.get("mag_median") is not None and tm is not None and d["n"] >= 10
                and tm - d["mag_median"] >= float(o["reference_mismatch_mag"])):
            if "dr_photometry_brighter_than_target" not in flags:
                flags.append("dr_photometry_brighter_than_target")
            why.append(f"IRSA DR light curve {d['oid']} ({d.get('filter')}): median "
                       f"{d['mag_median']:.2f} mag over {d['n']} epochs against the target's "
                       f"Gaia {tm:.2f} --- the source ZTF measures here is another star")

    # -- ledger context ----------------------------------------------------
    if ledger_rec:
        evs = ledger_rec.get("events") or []
        out["ledger"] = {"tier": ledger_rec.get("tier"), "n_events": ledger_rec.get("n_events"),
                         "n_visits": ledger_rec.get("n_visits"),
                         "duty_cycle": ledger_rec.get("duty_cycle"),
                         "p_binomial": ledger_rec.get("p_binomial"),
                         "nights": [e.get("night") for e in evs],
                         "a_by_band": {b: [(e.get("per_band") or {}).get(b, {}).get("a")
                                           for e in evs if b in (e.get("per_band") or {})]
                                       for b in ("g", "r", "i")}}

    # -- verdict -----------------------------------------------------------
    if "saturated_neighbour" in flags or "reference_saturated" in flags:
        cls = "systematic:saturated_neighbour_residual"
    elif "reference_brighter_than_target" in flags or "dr_photometry_brighter_than_target" in flags:
        cls = "systematic:blended_with_brighter_star"
    elif "self_flux" in flags:
        cls = "systematic:reference_missing_target"
    elif "known_variable_simbad" in flags or "known_variable_vsx" in flags:
        cls = "astrophysical:known_variable"
    elif "persistent_constant_residual" in flags:
        cls = "systematic:persistent_residual"
    else:
        cls = "unexplained"
    out["flags"] = flags
    out["explanation"] = why
    out["classification"] = cls
    return out


# ---------------------------------------------------------------------------
# Orchestration (runner)
# ---------------------------------------------------------------------------
def _target_row(source_id: str, targets, services: Services) -> tuple[dict | None, str]:
    if targets is not None and "source_id" in targets:
        hit = targets[targets["source_id"].astype(str) == str(source_id)]
        if len(hit):
            return _clean_row(hit.iloc[0].to_dict()), "target_list"
    rows, err = services.call("gaia_target", services.gaia, gaia_target_adql(source_id))
    if rows:
        return _clean_row(rows[0]), "gaia_tap"
    return None, err or "not_found"


def vet_target(source_id: str, *, z: dict, conf: dict, targets, ledger: Ledger,
               services: Services, out_dir: Path, opts: dict | None = None) -> dict:
    o = {**DEFAULTS, **(opts or {})}
    sid = str(source_id)
    rec_l = ledger.targets.get(sid)
    target, source = _target_row(sid, targets, services)
    if target is None:
        rec = {"source_id": sid, "vetted_at_utc": _utc(), "verdict": "NO_TARGET_ROW",
               "error": source, "services": services.log}
        _write_json(out_dir / f"{sid}.json", rec)
        return rec
    if rec_l and rec_l.get("events"):
        epoch = mjd_to_jyear(float(np.median([e["mjd"] for e in rec_l["events"]])))
    else:
        epoch = mjd_to_jyear(float(np.median(rec_l["visit_mjds"]))) if (
            rec_l and rec_l.get("visit_mjds")) else 2026.5
    ra0, dec0 = float(target["ra"]), float(target["dec"])
    pr, pd_ = propagate_pm(np.array([ra0]), np.array([dec0]),
                           np.array([_num(target.get("pmra")) or 0.0]),
                           np.array([_num(target.get("pmdec")) or 0.0]), to_epoch=epoch)
    p_ra, p_dec = float(pr[0]), float(pd_[0])

    neigh, _e = services.call("gaia_cone", services.gaia,
                              gaia_cone_adql(p_ra, p_dec, float(o["gaia_radius_arcsec"])))
    objs, _e = services.call("alerce_cone", alerce_objects_near, services.alerce, p_ra, p_dec,
                             float(o["alerce_radius_arcsec"]))
    summaries = []
    for ob in objs or []:
        oid = str(ob.get("oid"))
        dets, _e = services.call(f"alerce_detections:{oid}", services.alerce.detections, oid)
        nds, _e = services.call(f"alerce_non_detections:{oid}",
                                services.alerce.non_detections, oid)
        summaries.append(summarise_object(ob, dets or [], nds or [], target, p_ra, p_dec,
                                          int(o["max_detections_recorded"])))
    sb, _e = services.call("simbad", services.simbad,
                           simbad_cone_adql(p_ra, p_dec, float(o["catalogue_radius_arcsec"])))
    vs, _e = services.call("vsx", services.vizier,
                           vsx_cone_adql(p_ra, p_dec, float(o["catalogue_radius_arcsec"])))
    lc_text, _e = services.call("irsa_lightcurve", services.irsa,
                                irsa_lightcurve_url(p_ra, p_dec, float(o["irsa_radius_arcsec"])))
    lc_rows = parse_irsa_csv(lc_text or "")
    if lc_text:
        (out_dir / f"{sid}_irsa_lc.csv").write_text(lc_text)

    sat = z.get("saturation_mag")
    rec = analyse(sid, target, epoch, neigh or [], summaries, sb or [], vs or [], lc_rows,
                  None if sat is None else float(sat), rec_l, o)
    rec["vetted_at_utc"] = _utc()
    rec["target_source"] = source
    rec["verdict"] = "OK"
    rec["services"] = list(services.log)
    rec["queries"] = {"gaia_cone": gaia_cone_adql(p_ra, p_dec, float(o["gaia_radius_arcsec"])),
                      "irsa_lightcurve": irsa_lightcurve_url(p_ra, p_dec,
                                                             float(o["irsa_radius_arcsec"]))}
    _write_json(out_dir / f"{sid}.json", rec)
    services.log.clear()
    return rec


def vet(cfg=None, source_ids: list[str] | None = None, tiers: tuple[str, ...] = ("candidate",),
        out_dir: str | Path | None = None, targets_path: str | Path | None = None,
        services: Services | None = None, opts: dict | None = None) -> dict:
    """Vet the named targets and every ledger target at the named tiers."""
    conf, z = ztf_config(cfg)
    root = Path(cfg.root) if cfg is not None else _repo_root()
    o = {**DEFAULTS, **(z.get("vet") or {}), **(opts or {})}
    res = Path(out_dir) if out_dir else root / z["results_dir"]
    out = res / str(o["results_subdir"])
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.load(root / z["ledger_path"] if out_dir is None else res / "ledger.json")
    ids = [str(s) for s in (source_ids or [])]
    for tid, rec in ledger.targets.items():
        if rec.get("tier") in tiers and tid not in ids:
            ids.append(tid)
    tp = Path(targets_path) if targets_path else root / ".cache" / "tocsin_ztf" / "targets.parquet"
    targets = load_targets(tp)
    services = services or Services.live(z, float(o["timeout_s"]))
    index = {"vetted_at_utc": _utc(), "tiers": list(tiers), "targets": {}}
    for sid in ids:
        print(f"[tocsin-ztf-vet] {sid}")
        rec = vet_target(sid, z=z, conf=conf, targets=targets, ledger=ledger,
                         services=services, out_dir=out, opts=o)
        index["targets"][sid] = {"verdict": rec.get("verdict"),
                                 "classification": rec.get("classification"),
                                 "flags": rec.get("flags", []),
                                 "tier": (ledger.targets.get(sid) or {}).get("tier")}
        print(f"[tocsin-ztf-vet]   {rec.get('verdict')} {rec.get('classification')} "
              f"{rec.get('flags')}")
    index["n"] = len(ids)
    _write_json(out / "index.json", index)
    return index


__all__ = ["Services", "analyse", "summarise_object", "vet", "vet_target",
           "gaia_cone_adql", "gaia_target_adql", "simbad_cone_adql", "vsx_cone_adql",
           "irsa_lightcurve_url", "parse_irsa_csv", "alerce_objects_near"]

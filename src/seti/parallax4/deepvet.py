"""Deep vet of the sources that survive the Gaia-internal vet: each traced
to a mechanism, or left UNEXPLAINED with its evidence.

For every survivor (``vetted.csv``, classes SURVIVES / SURVIVES_FLAGGED):

* its full DR3 light curve re-fetched by DataLink and the detector re-run
  (the episode must reproduce from an independent retrieval);
* the transits around the episode, kept verbatim for inspection;
* SIMBAD object type (5") and VSX variability type (10");
* Gaia sky density (sources within 30") and Galactic latitude;
* regime flags: bright (G < 11: gating/saturation), partial transits
  (fewer than 5 AF CCDs in an episode transit), crowding.

``classify_fate`` is pure: known eclipsing binary (VSX), young stellar
object / dipper (natural dust, the leading grey-occulter explanation),
other catalogued variable, or UNEXPLAINED.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap"
ZTF_LC = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
#: Gaia DR3 epoch-photometry time origin (BJD 2455197.5) in MJD
GAIA_T0_MJD = 2455197.5 - 2400000.5
VIZIER_TAP = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap"

EB_VSX = re.compile(r"^(E|EA|EB|EW|EC|ED|ESD|ELL|E/|EA/|EB/|EW/)", re.I)
YSO_VSX = re.compile(r"(YSO|INS|IN\b|INA|INB|INT|ISA|ISB|UXOR|UX|FU|EXOR|DIP|TTS|CTTS|WTTS|ORION|IS\b)", re.I)
YSO_SIMBAD = re.compile(r"(YSO|TTau|TT\*|Or\*|Orion|Ae\*|Y\*O|Y\*\?|HerbigHaro|HH|pr\*|PreMain|YSO_Candidate)",
                        re.I)
EB_SIMBAD = re.compile(r"(EB\*|EclBin|SB\*)", re.I)


def classify_fate(row: dict) -> tuple[str, list[str]]:
    """(fate, flags) for one survivor from its gathered evidence."""
    flags: list[str] = []
    vsx = str(row.get("vsx_type") or "")
    otype = str(row.get("simbad_otype") or "")
    g = row.get("phot_g_mean_mag")
    if g is not None and np.isfinite(g) and g < 11:
        flags.append("bright_regime_G<11")
    if (row.get("n_gaia_30as") or 0) >= 30:
        flags.append(f"crowded:{int(row.get('n_gaia_30as'))}_within_30as")
    if row.get("min_n_obs_g") is not None and np.isfinite(row.get("min_n_obs_g", np.nan)) \
            and row["min_n_obs_g"] < 5:
        flags.append("partial_transit")
    if row.get("reproduced") is False:
        flags.append("episode_not_reproduced_on_refetch")
    zc = row.get("ztf_class")
    if zc == "ZTF_ECLIPSING_PHASED":
        return f"ECLIPSING_BINARY(ZTF P={row.get('ztf_period_d'):.5f} d, Gaia dips in phase)", flags
    if zc in ("ZTF_ECLIPSING_PHASE_AMBIGUOUS", "ZTF_PERIODIC_NOT_PHASED"):
        return (f"ECLIPSING_IN_ZTF(P={row.get('ztf_period_d'):.5f} d, Gaia phase "
                f"{'ambiguous' if zc.endswith('AMBIGUOUS') else 'NOT matched'})"), flags
    if zc == "ZTF_DIPS_APERIODIC":
        flags.append("ztf_dips_aperiodic")
    if vsx and EB_VSX.match(vsx):
        return "KNOWN_ECLIPSING_BINARY(VSX:" + vsx + ")", flags
    if (vsx and YSO_VSX.search(vsx)) or (otype and YSO_SIMBAD.search(otype)):
        return "YOUNG_STELLAR_OBJECT(" + (vsx or otype) + ")", flags
    if otype and EB_SIMBAD.search(otype):
        return "KNOWN_BINARY(SIMBAD:" + otype + ")", flags
    if vsx:
        return "KNOWN_VARIABLE(VSX:" + vsx + ")", flags
    if "episode_not_reproduced_on_refetch" in flags:
        return "NOT_REPRODUCED", flags
    return "UNEXPLAINED", flags


def _tap_rows(url: str, adql: str, deadline_s: float = 60.0) -> pd.DataFrame:
    import pyvo

    from .acquire import with_deadline

    res = with_deadline(lambda: pyvo.dal.TAPService(url).run_sync(adql), deadline_s, adql[:40])
    return res.to_table().to_pandas()


def simbad_type(ra: float, dec: float, r_arcsec: float = 5.0) -> dict:
    r = r_arcsec / 3600.0
    try:
        df = _tap_rows(SIMBAD_TAP, "SELECT TOP 5 main_id, otype, "
                       f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra:.7f}, {dec:.7f})) * 3600 AS sep "
                       f"FROM basic WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, "
                       f"{dec:.7f}, {r:.8f})) ORDER BY sep")
        if not len(df):
            return {"simbad_status": "NO_MATCH"}
        return {"simbad_status": "OK", "simbad_main_id": str(df.iloc[0]["main_id"]),
                "simbad_otype": str(df.iloc[0]["otype"]), "simbad_sep_arcsec": float(df.iloc[0]["sep"])}
    except Exception as exc:  # noqa: BLE001
        return {"simbad_status": f"ERROR:{exc!r}"[:120]}


def vsx_type(ra: float, dec: float, r_arcsec: float = 10.0) -> dict:
    r = r_arcsec / 3600.0
    try:
        df = _tap_rows(VIZIER_TAP, 'SELECT TOP 5 * FROM "B/vsx/vsx" WHERE 1 = CONTAINS(POINT(\'ICRS\', '
                       f"RAJ2000, DEJ2000), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r:.8f}))")
        if not len(df):
            return {"vsx_status": "NO_MATCH"}
        cols = {c.lower(): c for c in df.columns}
        r0 = df.iloc[0]
        return {"vsx_status": "OK", "vsx_name": str(r0.get(cols.get("name", "Name"), "")),
                "vsx_type": str(r0.get(cols.get("type", "Type"), "")),
                "vsx_period": r0.get(cols.get("period", "Period"))}
    except Exception as exc:  # noqa: BLE001
        return {"vsx_status": f"ERROR:{exc!r}"[:120]}


def fetch_ztf(ra: float, dec: float, *, radius_arcsec: float = 1.5, http=None) -> pd.DataFrame:
    """ZTF DR light curve (IRSA), good-quality points only (catflags == 0)."""
    import io as _io

    from .acquire import http_get

    http = http or http_get
    r = radius_arcsec / 3600.0
    st, body = http(ZTF_LC, params={"POS": f"CIRCLE {ra:.6f} {dec:.6f} {r:.6f}", "BANDNAME": "g,r",
                                    "FORMAT": "csv"}, timeout=120.0, retries=2)
    if st != 200 or not body:
        return pd.DataFrame()
    df = pd.read_csv(_io.BytesIO(body))
    if not len(df) or "mjd" not in df:
        return pd.DataFrame()
    if "catflags" in df:
        df = df[df["catflags"] == 0]
    return df


def ztf_eclipse_test(ztf: pd.DataFrame, gaia_dip_t: list[float], *, min_points: int = 60,
                     sde_min: float = 9.0, phase_tol: float = 0.03) -> dict:
    """Box-least-squares on the ZTF light curve (the band with more points),
    then: do the Gaia dip episodes fall in the ZTF eclipse at that period (or
    its double, for primary+secondary)?  ``gaia_dip_t`` in Gaia days
    (BJD - 2455197.5); ZTF MJD are converted (barycentric vs geocentric
    differs by <= 8.3 min, well inside ``phase_tol`` for P > 0.2 d)."""
    from astropy.timeseries import BoxLeastSquares

    out: dict = {"ztf_n": int(len(ztf)), "ztf_class": "ZTF_NO_DATA"}
    if len(ztf) < min_points:
        return out
    band = ztf["filtercode"].value_counts().idxmax() if "filtercode" in ztf else None
    z = ztf[ztf["filtercode"] == band] if band is not None else ztf
    t = z["mjd"].to_numpy(float) - GAIA_T0_MJD
    f = 10 ** (-0.4 * (z["mag"].to_numpy(float) - np.median(z["mag"])))
    e = np.clip(z["magerr"].to_numpy(float) * 0.921 * f, 1e-4, None)
    if len(t) < min_points:
        return out
    bls = BoxLeastSquares(t, f, e)
    periods = np.exp(np.linspace(np.log(0.2), np.log(min(300.0, np.ptp(t) / 2)), 30000))
    durations = [0.02, 0.05, 0.1, 0.2]
    res = bls.power(periods, [d for d in durations if d < periods.min()] or [0.02], objective="snr")
    pw = np.asarray(res.power)
    sde = float((pw.max() - np.mean(pw)) / (np.std(pw) + 1e-12))
    k = int(np.argmax(pw))
    P, t0, dur, depth = float(res.period[k]), float(res.transit_time[k]), float(res.duration[k]), float(res.depth[k])
    rms = float(1.4826 * np.median(np.abs(f - np.median(f))))
    out.update(ztf_band=str(band), ztf_period_d=P, ztf_t0=t0, ztf_duration_d=dur, ztf_depth=depth,
               ztf_sde=sde, ztf_rms=rms)
    lo = f < np.median(f) - 5 * max(rms, 1e-4)
    out["ztf_n_low_points"] = int(lo.sum())
    if sde >= sde_min and depth > 5 * rms:
        phased = []
        for P_try in (P, 2 * P, P / 2):
            ph = [((tg - t0) / P_try + 0.5) % 1.0 - 0.5 for tg in gaia_dip_t]
            half = dur / P_try / 2 + phase_tol
            ok = [abs(x) <= half or abs(abs(x) - 0.5) <= half for x in ph]
            phased.append((P_try, all(ok) and len(ok) > 0, ph))
        good = [x for x in phased if x[1]]
        # chance that n unrelated dip times all land in the (primary or
        # secondary) window at one of the 3 trial periods
        n_d = max(len(gaia_dip_t), 1)
        p_chance = float(min(1.0, 3 * (2 * (dur / P + 2 * phase_tol)) ** n_d))
        out["phase_p_chance"] = p_chance
        if good and p_chance < 0.01:
            out.update(ztf_class="ZTF_ECLIPSING_PHASED", ztf_period_d=float(good[0][0]),
                       gaia_dip_phases=[float(v) for v in good[0][2]])
        elif good:
            out.update(ztf_class="ZTF_ECLIPSING_PHASE_AMBIGUOUS", gaia_dip_phases=[float(v) for v in good[0][2]])
        else:
            out["ztf_class"] = "ZTF_PERIODIC_NOT_PHASED"
    elif out["ztf_n_low_points"] >= 3:
        out["ztf_class"] = "ZTF_DIPS_APERIODIC"
    else:
        out["ztf_class"] = "ZTF_QUIET"
    return out

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

from . import vsx_association as VA

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
    vsx_value = VA.classification_type(row)
    vsx = vsx_value or ""
    vsx_gap = any(str(k).startswith("vsx_") for k in row) and vsx_value is None
    vsx_type_gap = vsx_value == ""
    if vsx_gap:
        flags.append("vsx_target_association_unresolved")
    if vsx_type_gap:
        flags.append("vsx_type_unavailable")
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
    identity_gap = zc in ("ZTF_AMBIGUOUS_OBJECTS", "ZTF_IDENTITY_UNRESOLVED")
    if identity_gap:
        flags.append("ztf_object_identity_unresolved")
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
    if identity_gap:
        return "EVIDENCE_INCOMPLETE(ZTF_OBJECT_IDENTITY)", flags
    if vsx_gap:
        return "EVIDENCE_INCOMPLETE(VSX_TARGET_ASSOCIATION)", flags
    if vsx_type_gap:
        return "EVIDENCE_INCOMPLETE(VSX_CLASSIFICATION)", flags
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


def vsx_type(ra: float, dec: float, r_arcsec: float = 10.0, *,
             target_astrometry: dict | None = None,
             association_receipt: dict | None = None) -> dict:
    """Bounded VSX cone candidates, never an arbitrary first-row target type.

    The default live caller supplies no reviewed position-epoch/cross-ID receipt.
    A zero-row cone at an unverified position epoch is not catalogue absence.
    Optional receipts are a caller trust boundary, not authenticated here.
    """
    base = {"vsx_association_status": "CATALOGUE_QUERY_UNAVAILABLE",
            "vsx_receipt_authenticated_by_program": False}
    try:
        query_ra, query_dec = float(ra), float(dec)
        if not np.isfinite(query_ra) or not np.isfinite(query_dec) \
                or not 0 <= query_ra < 360 or not -90 <= query_dec <= 90 \
                or not np.isfinite(r_arcsec) or r_arcsec <= 0:
            return {**base, "vsx_status": "INVALID_QUERY_POSITION_OR_RADIUS"}
        context, _ = VA.receipt_context(target_astrometry, association_receipt)
        query_epoch = None
        if context is not None:
            direction = context["direction"]
            query_ra = float(np.rad2deg(np.arctan2(direction[1], direction[0])) % 360)
            query_dec = float(np.rad2deg(np.arcsin(np.clip(direction[2], -1, 1))))
            query_epoch = context["epoch"]
        r = r_arcsec / 3600.0
        query = (f'SELECT TOP {VA.MAX_CANDIDATES + 1} * FROM "B/vsx/vsx" '
                 "WHERE 1 = CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), "
                 f"CIRCLE('ICRS', {query_ra:.10f}, {query_dec:.10f}, {r:.10f}))")
        base.update(vsx_query=query, vsx_query_radius_arcsec=float(r_arcsec),
                    vsx_query_ra=query_ra, vsx_query_dec=query_dec,
                    vsx_query_position_epoch_jyear=query_epoch,
                    vsx_query_position_epoch_status=(
                        "CALLER_REVIEWED_RECEIPT" if query_epoch is not None else "UNVERIFIED"))
        df = _tap_rows(VIZIER_TAP, query)
        result = VA.audit_candidates(df, target=target_astrometry, receipt=association_receipt)
        if context is not None and context["tolerance"] > r_arcsec:
            result.pop("vsx_type", None)
            result["vsx_association_status"] = "MATCH_TOLERANCE_EXCEEDS_QUERY_RADIUS"
        return {**base, **result}
    except Exception as exc:  # noqa: BLE001
        return {**base, "vsx_status": f"ERROR:{exc!r}"[:120]}


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
    # Catalogue identifiers are opaque keys. Mixed/null CSV columns otherwise
    # become float64 and can silently merge distinct identifiers above 2**53.
    df = pd.read_csv(_io.BytesIO(body), dtype={"oid": "string"})
    if not len(df) or "mjd" not in df:
        return pd.DataFrame()
    if "catflags" in df:
        df = df[df["catflags"] == 0]
    return df



def _ztf_object_identity(z: pd.DataFrame) -> dict:
    """One coherent catalogue light curve; not proof of Gaia association.

    ZTF may represent one physical source with several field-specific IDs.
    Resolving those requires separate astrometric evidence, so neither the
    nearest row nor a shared magnitude is enough to combine them here.
    """
    base = {"ztf_n_band": int(len(z)),
            "ztf_target_association": "NOT_ESTABLISHED_BY_OBJECT_ID"}
    if "oid" not in z:
        return {**base, "ztf_identity_status": "MISSING_OBJECT_ID",
                "ztf_n_invalid_ids": int(len(z)), "ztf_n_objects": None,
                "ztf_object_ids": []}
    ids, invalid = [], 0
    for value in z["oid"]:
        # Accept exact integer keys and their CSV spellings, never floats:
        # even an integral float may already have been rounded.
        if isinstance(value, (bool, np.bool_)):
            invalid += 1
            continue
        if isinstance(value, (int, np.integer)):
            value = str(value)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value) \
                or not value.strip("0"):
            invalid += 1
            continue
        ids.append(value)
    unique = sorted(set(ids))
    base.update(ztf_n_invalid_ids=invalid, ztf_n_objects=len(unique),
                ztf_object_ids=unique)
    if invalid:
        return {**base, "ztf_identity_status": "MALFORMED_OBJECT_ID"}
    if len(unique) != 1:
        return {**base, "ztf_identity_status": "MULTIPLE_OBJECT_IDS"}
    return {**base, "ztf_identity_status": "SINGLE_OBJECT_ID",
            "ztf_object_id": unique[0]}


def _ztf_phase_chance_bound(period: float, duration: float, phase_tol: float,
                            n_dips: int) -> tuple[float, list[dict]]:
    """Conditional union bound for the three fixed ZTF alias ephemerides.

    Each trial accepts two circular phase intervals centered at 0 and 0.5.
    Their total measure is 4 * half_width until they touch, then 1. Under
    independent uniform Gaia episode phases, all n fall in one trial with
    probability measure**n. Sum those probabilities: no independence BETWEEN
    aliases is assumed. Gaia cadence/correlated episodes or uncertain fitted
    ephemerides can violate this null; this is not an empirical false-alarm
    probability or a validation of the target association.
    """
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be finite and positive")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not np.isfinite(phase_tol) or phase_tol < 0:
        raise ValueError("phase_tol must be finite and nonnegative")
    if isinstance(n_dips, (bool, np.bool_)) or not isinstance(n_dips, (int, np.integer)) \
            or n_dips < 0:
        raise ValueError("n_dips must be a nonnegative integer")
    trials = []
    for period_try in (period, 2 * period, period / 2):
        half = duration / period_try / 2 + phase_tol
        fraction = min(1.0, 4 * half)
        trials.append({"period_d": float(period_try), "half_width_phase": float(half),
                       "accepted_phase_fraction": float(fraction), "n_dips": int(n_dips),
                       "all_dips_probability": float(fraction ** n_dips)})
    return float(min(1.0, sum(x["all_dips_probability"] for x in trials))), trials


def ztf_eclipse_test(ztf: pd.DataFrame, gaia_dip_t: list[float], *, min_points: int = 60,
                     sde_min: float = 9.0, phase_tol: float = 0.03) -> dict:
    """Box-least-squares on the ZTF light curve (the band with more points),
    after requiring one exact catalogue object ID in that band; then: do the
    Gaia dip episodes fall in the ZTF eclipse at that period (or
    its double, for primary+secondary)?  ``gaia_dip_t`` in Gaia days
    (BJD - 2455197.5); ZTF MJD are converted (barycentric vs geocentric
    differs by <= 8.3 min, well inside ``phase_tol`` for P > 0.2 d)."""
    from astropy.timeseries import BoxLeastSquares

    out: dict = {"ztf_n": int(len(ztf)), "ztf_class": "ZTF_NO_DATA"}
    if len(ztf) < min_points:
        return out
    band = ztf["filtercode"].value_counts().idxmax() if "filtercode" in ztf else None
    z = ztf[ztf["filtercode"] == band] if band is not None else ztf
    out.update(ztf_band=str(band), **_ztf_object_identity(z))
    if out["ztf_identity_status"] != "SINGLE_OBJECT_ID":
        out["ztf_class"] = ("ZTF_AMBIGUOUS_OBJECTS"
                            if out["ztf_identity_status"] == "MULTIPLE_OBJECT_IDS"
                            else "ZTF_IDENTITY_UNRESOLVED")
        return out
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
        p_chance, trials = _ztf_phase_chance_bound(P, dur, phase_tol, len(gaia_dip_t))
        phased = []
        for trial in trials:
            P_try, half = trial["period_d"], trial["half_width_phase"]
            ph = [((tg - t0) / P_try + 0.5) % 1.0 - 0.5 for tg in gaia_dip_t]
            ok = [abs(x) <= half or abs(abs(x) - 0.5) <= half for x in ph]
            phased.append((P_try, all(ok) and len(ok) > 0, ph))
        good = [x for x in phased if x[1]]
        out.update(phase_p_chance=p_chance, phase_window_trials=trials,
                   phase_p_chance_method="UNION_BOUND_FIXED_ZTF_EPHEMERIS",
                   phase_p_chance_assumptions="INDEPENDENT_UNIFORM_GAIA_EPISODE_PHASES",
                   phase_p_chance_is_empirical=False)
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

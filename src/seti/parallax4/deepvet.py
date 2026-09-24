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

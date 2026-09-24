"""The archive rungs of the IGNITION contaminant ladder (runner only).

Folded in from the re-vet of tiles run 35740159635 (``revet.py``,
docs/ignition.md 7.4), so every future tile is vetted by the assess stage
itself.  Each rung fetches, then hands the numbers to a PURE decision in
``vet.py``; the fetchers are injectable so the whole rung is tested offline.

| rung | fetch | decision (``vet.py``) | kills |
|---|---|---|---|
| approaching neighbour | Gaia DR3 cone, 30" | ``neighbour_contamination`` | ``approaching_neighbour`` |
| optical trend | ZTF (IRSA), ASAS-SN Sky Patrol | ``optical_rise_verdict`` | ``optical_not_flat`` / ``rcrb_like`` |
| optical scatter | Gaia DR3 per-obs scatter vs G/BP-RP peers | ``gaia_scatter_verdict`` | ``optical_variable_gaia`` |

A rung that could not reach its archive is recorded as **unreachable**: the
star's check is listed in ``untested_checks`` and the summary carries a
``vet_unreachable:<rung>:<n>`` degradation.  An unreachable check is never a
pass.
"""

from __future__ import annotations

import pandas as pd

from .vet import (
    DEFAULT_VET,
    gaia_scatter_verdict,
    neighbour_contamination,
    optical_rise_verdict,
)

ESA_TAP = "https://gea.esac.esa.int/tap-server/tap"


def fetch_gaia_cone(ra: float, dec: float, radius_arcsec: float = 30.0) -> pd.DataFrame:
    """Gaia DR3 sources in a cone (raises on failure)."""
    from .revet import _tap, _timed

    q = ("SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp "
         "FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
         f"CIRCLE('ICRS', {float(ra)}, {float(dec)}, {float(radius_arcsec) / 3600.0}))")
    df, err = _timed(lambda: _tap(ESA_TAP, q), 180)
    if df is None:
        raise RuntimeError(f"gaia cone: {err}")
    df.columns = [c.lower() for c in df.columns]
    df["source_id"] = df["source_id"].astype(str)
    return df


def _default_fetchers() -> dict:
    from .revet import asassn_lightcurve, gaia_variability_proxy, ztf_lightcurve

    return {"gaia_cone": fetch_gaia_cone, "ztf": ztf_lightcurve,
            "asassn": asassn_lightcurve, "gaia_scatter": gaia_variability_proxy}


def _bands_from(survey: str, rec: dict) -> dict:
    """Per-filter band dicts out of a ZTF / ASAS-SN fetch result."""
    out = {}
    if isinstance(rec, dict) and rec.get("status") == "OK":
        for k, v in rec.items():
            if isinstance(v, dict):
                out[f"{survey}:{k}"] = v
    return out


def online_vet(cands: pd.DataFrame, stars: pd.DataFrame | None = None,
               conf: dict | None = None, fetchers: dict | None = None) -> tuple[dict, dict]:
    """Run the archive rungs on ``cands``.  Returns (per-star results, unreachable counts).

    Per star: ``{"neighbour": {...}, "optical": {...}, "gaia_scatter": {...},
    "flags": [...], "untested": [...]}``.
    """
    c = {**DEFAULT_VET, **(conf or {})}
    f = {**_default_fetchers(), **(fetchers or {})}
    unreachable = {"gaia_neighbours": 0, "optical": 0, "gaia_scatter": 0}
    res: dict = {}
    asassn_dead = None
    scatter: dict = {}
    if len(cands):
        try:
            peers = stars if stars is not None and len(stars) else cands
            scatter = f["gaia_scatter"](cands, peers) or {}
        except Exception as exc:                      # noqa: BLE001
            scatter = {"status": "FAILED", "error": repr(exc)[:200]}
    for _, row in cands.iterrows():
        r = row.to_dict()
        sid = str(r["source_id"])
        flags: list[str] = []
        untested: list[str] = []
        out: dict = {}
        # --- approaching neighbour ------------------------------------------
        try:
            cone = f["gaia_cone"](float(r["ra"]), float(r["dec"]),
                                  float(c["neighbour_radius_arcsec"]))
            cone = cone.copy()
            cone["source_id"] = cone["source_id"].astype(str)
            tgt = cone[cone["source_id"] == sid]
            target = tgt.iloc[0].to_dict() if len(tgt) else r
            nb = cone[cone["source_id"] != sid]
            rise = r.get("w1_rise_mag")
            nc = neighbour_contamination(target, nb, float(rise) if rise is not None
                                         else float("nan"), c)
            out["neighbour"] = nc
            if nc["status"] == "approaching_neighbour":
                flags.append("approaching_neighbour")
        except Exception as exc:                      # noqa: BLE001
            out["neighbour"] = {"status": "unreachable", "error": repr(exc)[:200]}
            untested.append("neighbour_motion")
            unreachable["gaia_neighbours"] += 1
        # --- optical trend ---------------------------------------------------
        fetched, reached = {}, False
        for survey in ("ztf", "asassn"):
            if survey == "asassn" and asassn_dead:
                fetched[survey] = {"status": "SKIPPED", "error": asassn_dead}
                continue
            try:
                rec = f[survey](row)
            except Exception as exc:                  # noqa: BLE001
                rec = {"status": "FAILED", "error": repr(exc)[:200]}
            fetched[survey] = rec
            st = str((rec or {}).get("status"))
            if st in ("OK", "NO_ROWS", "NOT_IN_FOOTPRINT"):
                reached = True
            elif survey == "asassn" and (st == "CLIENT_MISSING" or any(
                    k in str(rec.get("error")) for k in ("ConnectTimeout", "Max retries",
                                                         "Connection refused"))):
                # the service itself is down: do not spend 5 min per star on it
                asassn_dead = str(rec.get("error"))[:160]
        bands = {**_bands_from("ztf", fetched.get("ztf")),
                 **_bands_from("asassn", fetched.get("asassn"))}
        ov = optical_rise_verdict(bands, r.get("w1_slope_mag_yr"), c)
        ov["sources"] = {k: (v or {}).get("status") for k, v in fetched.items()}
        out["optical"] = ov
        if ov["status"] == "brightening":
            flags.append("optical_not_flat")
        elif ov["status"] == "fading":
            flags.append("rcrb_like")
        elif ov["status"] in ("untested", "inconsistent"):
            untested.append("optical_flatness")
            if not reached:
                unreachable["optical"] += 1
        # --- Gaia scatter ----------------------------------------------------
        if isinstance(scatter, dict) and scatter.get("status") == "OK":
            gv = gaia_scatter_verdict(scatter.get(sid), c)
        else:
            gv = {"status": "unreachable", "error": (scatter or {}).get("error")}
            unreachable["gaia_scatter"] += 1
        out["gaia_scatter"] = gv
        if gv["status"] == "variable":
            flags.append("optical_variable_gaia")
        out["flags"], out["untested"] = flags, untested
        res[sid] = out
    return res, unreachable


__all__ = ["fetch_gaia_cone", "online_vet"]

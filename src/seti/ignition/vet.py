"""The IGNITION contaminant ladder.  Pure, with an optical-series hook.

Every rule below is a named veto with its own counter (``summarise``), so the
funnel partitions the sample and the summary can say *which* contaminant took
what.  Order matters: the first rule that fires names the verdict.

| Rule | Kills | Why |
|---|---|---|
| ``extragalactic`` | obscured-AGN turn-on (NGC 6447) | no significant parallax / PM |
| ``star_forming_region`` | YSO accretion outbursts | position in a configured box |
| ``galactic_plane`` | YSOs, crowding | ``|b|`` below the floor |
| ``gaia_variable`` | Miras, novae, R CrB, YSOs | Gaia DR3 says VARIABLE |
| ``saturated`` | NEOWISE bright-source bias | W1 < 8 |
| ``deblended`` | latent / deblending artefacts | ``nb > 1`` or ``na > 0`` on most frames |
| ``poor_photometry`` | marginal detections | too many frames cut on ``ph_qual`` |
| ``already_excess_2010`` | anything with an old excess | AllWISE W1-W2 not photospheric |
| ``rcrb_like`` | R CrB dust puffs | optical *fades* while the IR rises |
| ``optical_not_flat`` | novae, AGB dust formation, YSOs | optical slope or scatter |

A star whose optical light curve could not be checked is **not** an optically
flat star; it is carried with ``optical: not_checked`` and its verdict says
``optical_untested`` rather than ``clean``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..vigil.vet import galactic_latitude

DEFAULT_VET: dict = {
    "parallax_over_error_min": 10.0,
    "pm_sig_min": 5.0,
    "galactic_latitude_min_deg": 15.0,
    "w1_saturation_mag": 8.0,
    "w2_saturation_mag": 7.0,
    "frac_nb_gt1_max": 0.2,
    "frac_na_gt0_max": 0.2,
    "ph_qual_cut_frac_max": 0.5,
    "agn_w1w2_min": 0.8,
    "w1w2_2010_max": 0.15,
    "optical_slope_sigma_max": 3.0,
    "optical_rms_max": 0.05,
    "optical_min_points": 20,
    # Known star-forming regions and young associations, generous boxes
    # (ra_min, ra_max, dec_min, dec_max) in degrees.
    "star_forming_boxes": [
        {"name": "Taurus", "ra": [60.0, 75.0], "dec": [15.0, 32.0]},
        {"name": "Orion", "ra": [78.0, 92.0], "dec": [-12.0, 12.0]},
        {"name": "Perseus", "ra": [50.0, 58.0], "dec": [29.0, 34.0]},
        {"name": "Ophiuchus", "ra": [243.0, 250.0], "dec": [-27.0, -21.0]},
        {"name": "UpperSco", "ra": [235.0, 250.0], "dec": [-30.0, -15.0]},
        {"name": "Lupus", "ra": [232.0, 246.0], "dec": [-43.0, -33.0]},
        {"name": "Chamaeleon", "ra": [160.0, 172.0], "dec": [-80.0, -74.0]},
        {"name": "Serpens", "ra": [275.0, 280.0], "dec": [-3.0, 3.0]},
        {"name": "CoronaAustralis", "ra": [283.0, 288.0], "dec": [-39.0, -35.0]},
        {"name": "Cepheus", "ra": [310.0, 340.0], "dec": [60.0, 72.0]},
    ],
}


@dataclass
class VetResult:
    verdict: str
    flags: list[str] = field(default_factory=list)
    optical: str = "not_checked"
    optical_slope_mag_yr: float = float("nan")
    optical_slope_sigma: float = float("nan")
    optical_rms_mag: float = float("nan")
    optical_n: int = 0
    region: str = ""
    untested_checks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["flags"] = ";".join(self.flags)
        d["untested_checks"] = ";".join(self.untested_checks)
        return d


def _f(row: dict, key: str, default=float("nan")) -> float:
    try:
        v = row.get(key, default)
        v = float(v)
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def in_star_forming_region(ra: float, dec: float, boxes) -> str:
    """Name of the first configured box containing (ra, dec), else ''."""
    for b in boxes or []:
        r0, r1 = (float(x) for x in b["ra"])
        d0, d1 = (float(x) for x in b["dec"])
        if d0 <= dec <= d1 and r0 <= ra <= r1:
            return str(b.get("name", "box"))
    return ""


def optical_flatness(t_yr, mag, err=None, conf: dict | None = None) -> dict:
    """Weighted slope, its significance, and the rms of an optical series.

    Returns ``{"status": flat | brightening | fading | insufficient, ...}``.
    A significantly *positive* slope (fading in magnitude) while the IR rises
    is the R CrB sign; a significantly negative one is a nova / AGB / YSO
    brightening the IR is merely following.
    """
    c = {**DEFAULT_VET, **(conf or {})}
    t = np.asarray(t_yr, float)
    m = np.asarray(mag, float)
    e = (np.asarray(err, float) if err is not None else np.full(t.size, np.nan))
    ok = np.isfinite(t) & np.isfinite(m)
    t, m, e = t[ok], m[ok], e[ok]
    if t.size < int(c["optical_min_points"]):
        return {"status": "insufficient", "n": int(t.size), "slope": float("nan"),
                "slope_sigma": float("nan"), "rms": float("nan")}
    e = np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(e[np.isfinite(e) & (e > 0)])
                 if np.any(np.isfinite(e) & (e > 0)) else 0.02)
    w = 1.0 / e**2
    tc = t - np.mean(t)
    X = np.column_stack([np.ones(t.size), tc])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], m * sw, rcond=None)
    resid = m - X @ beta
    chi2_red = float(np.sum(w * resid**2) / max(t.size - 2, 1))
    cov = np.linalg.inv((X * sw[:, None]).T @ (X * sw[:, None]))
    slope_err = float(np.sqrt(cov[1, 1]) * np.sqrt(max(1.0, chi2_red)))
    slope = float(beta[1])
    sig = slope / slope_err if slope_err > 0 else float("nan")
    rms = float(np.std(resid, ddof=2)) if t.size > 2 else float("nan")
    if np.isfinite(sig) and abs(sig) >= float(c["optical_slope_sigma_max"]):
        status = "fading" if slope > 0 else "brightening"
    elif rms > float(c["optical_rms_max"]):
        status = "variable"
    else:
        status = "flat"
    return {"status": status, "n": int(t.size), "slope": slope, "slope_sigma": float(sig),
            "rms": rms}


def load_optical_series(optical_dir: Path | str | None, source_id: str) -> pd.DataFrame | None:
    """``<optical_dir>/<source_id>.csv`` with columns ``mjd|t_yr, mag[, magerr]``."""
    if not optical_dir:
        return None
    p = Path(optical_dir) / f"{source_id}.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except Exception:                                  # noqa: BLE001
        return None
    d.columns = [str(x).lower() for x in d.columns]
    if "t_yr" not in d and "mjd" in d:
        d["t_yr"] = 2000.0 + (pd.to_numeric(d["mjd"], errors="coerce") - 51544.5) / 365.25
    if "t_yr" not in d or "mag" not in d:
        return None
    return d


def vet_star(row: dict, conf: dict | None = None, optical: pd.DataFrame | None = None
             ) -> VetResult:
    """Run the ladder on one screened star.  ``row`` carries the merged columns."""
    c = {**DEFAULT_VET, **(conf or {})}
    flags: list[str] = []
    untested: list[str] = []
    res = VetResult(verdict="clean")

    ra, dec = _f(row, "ra"), _f(row, "dec")
    poe = _f(row, "parallax_over_error")
    pmra, pmdec = _f(row, "pmra", 0.0), _f(row, "pmdec", 0.0)
    pm_err = _f(row, "pm_error", float("nan"))
    pm = float(np.hypot(pmra, pmdec))
    pm_sig = pm / pm_err if np.isfinite(pm_err) and pm_err > 0 else float("nan")
    astrometric = (np.isfinite(poe) and poe >= float(c["parallax_over_error_min"]))
    if np.isfinite(pm_sig):
        astrometric = astrometric or pm_sig >= float(c["pm_sig_min"])
    if not astrometric:
        flags.append("extragalactic")

    if np.isfinite(ra) and np.isfinite(dec):
        reg = in_star_forming_region(ra, dec, c["star_forming_boxes"])
        res.region = reg
        if reg:
            flags.append("star_forming_region")
        b = _f(row, "b", galactic_latitude(ra, dec))
        if abs(b) < float(c["galactic_latitude_min_deg"]):
            flags.append("galactic_plane")
    else:
        untested.append("position")

    if str(row.get("phot_variable_flag", "")).upper() == "VARIABLE":
        flags.append("gaia_variable")

    w1 = _f(row, "w1_median", _f(row, "w1mpro"))
    w2 = _f(row, "w2_median", _f(row, "w2mpro"))
    if np.isfinite(w1) and w1 < float(c["w1_saturation_mag"]):
        flags.append("saturated")
    elif np.isfinite(w2) and w2 < float(c["w2_saturation_mag"]):
        flags.append("saturated")

    nb, na = _f(row, "frac_nb_gt1"), _f(row, "frac_na_gt0")
    if (np.isfinite(nb) and nb > float(c["frac_nb_gt1_max"])) or \
       (np.isfinite(na) and na > float(c["frac_na_gt0_max"])):
        flags.append("deblended")
    elif not np.isfinite(nb) and not np.isfinite(na):
        untested.append("deblending")

    n_raw = _f(row, "n_exp_raw", float("nan"))
    cut_pq = _f(row, "cut_ph_qual", float("nan"))
    if np.isfinite(n_raw) and n_raw > 0 and np.isfinite(cut_pq):
        if cut_pq / n_raw > float(c["ph_qual_cut_frac_max"]):
            flags.append("poor_photometry")
    else:
        untested.append("ph_qual")

    w1w2 = _f(row, "w1w2_2010", _f(row, "w1mpro") - _f(row, "w2mpro"))
    if np.isfinite(w1w2):
        if w1w2 > float(c["agn_w1w2_min"]):
            flags.append("nearest_agn_like")
        if w1w2 > float(c["w1w2_2010_max"]):
            flags.append("already_excess_2010")
    else:
        untested.append("allwise_2010_colour")

    if bool(row.get("grey_rise_suspect", False)):
        flags.append("grey_rise_suspect")

    # --- optical hook ------------------------------------------------------
    if optical is not None and len(optical):
        opt = optical_flatness(optical["t_yr"], optical["mag"],
                               optical["magerr"] if "magerr" in optical else None, c)
        res.optical = opt["status"]
        res.optical_slope_mag_yr = opt["slope"]
        res.optical_slope_sigma = opt["slope_sigma"]
        res.optical_rms_mag = opt["rms"]
        res.optical_n = opt["n"]
        if opt["status"] == "fading":
            flags.append("rcrb_like")
        elif opt["status"] in ("brightening", "variable"):
            flags.append("optical_not_flat")
        elif opt["status"] == "insufficient":
            untested.append("optical_flatness")
    else:
        res.optical = "not_checked"
        untested.append("optical_flatness")

    res.flags = flags
    res.untested_checks = untested
    kill_order = ("extragalactic", "already_excess_2010", "star_forming_region",
                  "galactic_plane", "gaia_variable", "saturated", "deblended",
                  "poor_photometry", "rcrb_like", "optical_not_flat")
    for k in kill_order:
        if k in flags:
            res.verdict = f"rejected_{k}"
            return res
    res.verdict = "clean" if "optical_flatness" not in untested else "clean_optical_untested"
    return res


def summarise(results: list[VetResult] | list[dict]) -> dict:
    """Counts per verdict and per flag."""
    verd: dict = {}
    flg: dict = {}
    for r in results:
        d = r.as_dict() if isinstance(r, VetResult) else dict(r)
        v = str(d.get("verdict", ""))
        verd[v] = verd.get(v, 0) + 1
        for f in str(d.get("flags", "")).split(";"):
            if f:
                flg[f] = flg.get(f, 0) + 1
    return {"verdicts": verd, "flags": flg}


__all__ = ["DEFAULT_VET", "VetResult", "in_star_forming_region", "load_optical_series",
           "optical_flatness", "summarise", "vet_star"]

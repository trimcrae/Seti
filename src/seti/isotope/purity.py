"""ISOTOPE detector: isotopic purity without a nucleosynthetic package.

Pure functions over the canonical grain table (``acquire.canonicalise``).

The physics (docs/necrofrontier.md §2.XII, S46).  Natural solids carry
isotope anomalies only as *correlated* packages — a supernova X grain is
²⁸Si-rich AND carries ⁴⁴Ti (seen as a ⁴⁴Ca excess), ²⁶Al/²⁷Al ~ 0.1–0.6,
¹⁵N and ¹²C excesses — or as per-mil mass-dependent fractionation
(δ²⁹Si ≈ ½ δ³⁰Si at small amplitude).  Technology makes mono-isotopic
material: quantum-grade ²⁸Si sits at δ²⁹Si ≈ δ³⁰Si ≈ −999 ‰ with the C, N,
Al–Mg and Ca–Ti of whatever ordinary (solar / terrestrial) feedstock it was
made from.  So the discriminant is not purity alone but purity *with solar
partners*; and the purity floor is set empirically by the database's own most
extreme classified X grain, not by a model.

Classes (``classify_table``)
----------------------------
``NO_SILICON``               δ²⁹Si / δ³⁰Si value or error absent: the purity test cannot run
``ORDINARY``                 inside the natural envelope, no other tag
``FRACTIONATION``            δ²⁹ ≈ ½ δ³⁰ within errors at small, resolved amplitude
``NATURAL_PACKAGE``          purity beyond the envelope (or ¹²C beyond) WITH a
                             partner resolved away from solar — a supernova grain
``INSUFFICIENT_PANEL``       beyond the envelope but too few partners measured to
                             test; never a candidate
``PURITY_CANDIDATE``         beyond the envelope in both δ²⁹Si and δ³⁰Si by > 3σ,
                             every measured partner consistent with solar at 3σ,
                             at least ``min_partners_for_candidate`` of them
``CARBON_PURITY_CANDIDATE``  the carbon analogue: ¹²C/¹³C beyond the most extreme
                             classified natural grain, with solar N and solar Si
plus the flag ``CONTAMINATION_SUSPECT`` (δ²⁹ ≈ δ³⁰ ≈ −999 with no C and no N
measured: what a silicon-wafer fragment looks like).

Nothing here talks to a network or a file.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .acquire import PARTNER_ROLES

CLASS_NO_SILICON = "NO_SILICON"
CLASS_ORDINARY = "ORDINARY"
CLASS_FRACTIONATION = "FRACTIONATION"
CLASS_NATURAL = "NATURAL_PACKAGE"
CLASS_INSUFFICIENT = "INSUFFICIENT_PANEL"
CLASS_CANDIDATE = "PURITY_CANDIDATE"
CLASS_CARBON = "CARBON_PURITY_CANDIDATE"
FLAG_CONTAMINATION = "CONTAMINATION_SUSPECT"
CLASSES = (CLASS_NO_SILICON, CLASS_ORDINARY, CLASS_FRACTIONATION, CLASS_NATURAL,
           CLASS_INSUFFICIENT, CLASS_CANDIDATE, CLASS_CARBON)

DEFAULT_SOLAR = {"c12c13": 89.0, "n14n15": 272.0, "al26al27": 0.0, "ti44ti48": 0.0221,
                 "d44ca": 0.0, "d29si": 0.0, "d30si": 0.0}
DEFAULT_PACKAGE = {"c12c13": 300.0, "n14n15": 50.0, "al26al27": 0.1, "ti44ti48": 0.05,
                   "d44ca": 300.0}
DEFAULT_THRESHOLDS = {
    "envelope_sigma": 3.0, "partner_sigma": 3.0, "package_sigma": 3.0, "al26_limit_max": 0.01,
    "min_partners_for_candidate": 2, "fractionation_slope": 0.5, "fractionation_max_permil": 60.0,
    "fractionation_min_sigma": 2.0, "wafer_permil": -950.0, "wafer_equal_permil": 60.0,
    "carbon_envelope_sigma": 3.0, "carbon_si_solar_sigma": 3.0, "frontier_n": 25,
}
DEFAULT_ENVELOPE_FALLBACK = {"d29si_min": -650.0, "d30si_min": -1000.0}


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _thr(conf: dict) -> dict:
    t = dict(DEFAULT_THRESHOLDS)
    t.update({k: v for k, v in (conf.get("thresholds") or {}).items() if v is not None})
    return t


# ---------------------------------------------------------------------------
# the empirical envelope
# ---------------------------------------------------------------------------
def _type_mask(canon: pd.DataFrame, regex: str | None) -> np.ndarray:
    if "grain_type" not in canon.columns or not regex:
        return np.zeros(len(canon), dtype=bool)
    try:
        rx = re.compile(str(regex))
    except re.error:
        return np.zeros(len(canon), dtype=bool)
    return np.array([bool(rx.search(str(t))) for t in canon["grain_type"].tolist()], dtype=bool)


def convex_hull_vertices(points: np.ndarray) -> list[list[float]]:
    """Hull vertices (counter-clockwise) of an (N, 2) array; [] if degenerate."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return []
    try:
        from scipy.spatial import ConvexHull  # noqa: PLC0415
        hull = ConvexHull(pts)
        return [[float(x), float(y)] for x, y in pts[hull.vertices]]
    except Exception:                                     # noqa: BLE001  collinear etc.
        return []


def inside_hull(points: np.ndarray, hull: list[list[float]]) -> np.ndarray:
    """Point-in-convex-polygon test (ray casting on the hull polygon)."""
    pts = np.asarray(points, dtype=float)
    out = np.zeros(len(pts), dtype=bool)
    if len(hull) < 3 or not len(pts):
        return out
    poly = np.asarray(hull, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    n = len(poly)
    inside = np.zeros(len(pts), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) if yj != yi else 1e-300) + xi)
        inside ^= cond
        j = i
    out[np.isfinite(x) & np.isfinite(y)] = inside[np.isfinite(x) & np.isfinite(y)]
    return out


def mixing_line_distance(d29, d30, end: tuple[float, float]) -> np.ndarray:
    """Distance (‰) from each grain to the segment solar (0,0) → the X end-member."""
    p = np.column_stack([np.asarray(d29, dtype=float), np.asarray(d30, dtype=float)])
    e = np.asarray(end, dtype=float)
    ee = float(e @ e)
    if not np.isfinite(ee) or ee <= 0:
        return np.full(len(p), np.nan)
    t = np.clip((p @ e) / ee, 0.0, 1.0)
    proj = np.outer(t, e)
    return np.sqrt(((p - proj) ** 2).sum(axis=1))


def x_envelope(canon: pd.DataFrame, conf: dict) -> dict:
    """The database's own most extreme classified X grain, the X hull, the X package.

    Falls back to the configured Lin et al. 2010 end-member when the table
    carries no classified X grain (and says so in ``source``).
    """
    thr = _thr(conf)
    xmask = _type_mask(canon, conf.get("x_type_regex") or r"(?i)^\s*x")
    d29 = canon["d29si"].to_numpy(dtype=float)
    d30 = canon["d30si"].to_numpy(dtype=float)
    ok = xmask & np.isfinite(d29) & np.isfinite(d30)
    env: dict = {"n_x_grains": int(xmask.sum()), "n_x_with_si": int(ok.sum()),
                 "x_type_regex": conf.get("x_type_regex")}
    fb = dict(DEFAULT_ENVELOPE_FALLBACK)
    fb.update(conf.get("envelope_fallback") or {})
    if ok.any():
        i29 = int(np.flatnonzero(ok)[np.argmin(d29[ok])])
        i30 = int(np.flatnonzero(ok)[np.argmin(d30[ok])])
        r = np.hypot(d29[ok], d30[ok])
        iend = int(np.flatnonzero(ok)[np.argmax(r)])
        env.update({
            "source": "empirical_x_grains",
            "d29si_min": float(d29[i29]), "d29si_min_grain": str(canon["grain_id"].iloc[i29]),
            "d29si_min_err": _f(canon["d29si_err"].iloc[i29]),
            "d30si_min": float(d30[i30]), "d30si_min_grain": str(canon["grain_id"].iloc[i30]),
            "d30si_min_err": _f(canon["d30si_err"].iloc[i30]),
            "end_member": [float(d29[iend]), float(d30[iend])],
            "end_member_grain": str(canon["grain_id"].iloc[iend]),
            "hull": convex_hull_vertices(np.column_stack([d29[ok], d30[ok]])),
        })
    else:
        env.update({"source": "config_fallback_no_classified_x_grain",
                    "d29si_min": float(fb["d29si_min"]), "d30si_min": float(fb["d30si_min"]),
                    "d29si_min_grain": None, "d30si_min_grain": None,
                    "end_member": [float(fb["d29si_min"]), float(fb["d30si_min"])],
                    "end_member_grain": None, "hull": []})
    # the X package: empirical medians of the X grains' partners, else config
    pkg = dict(DEFAULT_PACKAGE)
    pkg.update(conf.get("x_package") or {})
    min_n = int(conf.get("x_package_min_n", 10))
    package, package_source = {}, {}
    for p in PARTNER_ROLES:
        v = canon[p].to_numpy(dtype=float) if p in canon.columns else np.full(len(canon), np.nan)
        m = xmask & np.isfinite(v)
        if m.sum() >= min_n:
            package[p] = float(np.nanmedian(v[m]))
            package_source[p] = f"median_of_{int(m.sum())}_x_grains"
        else:
            package[p] = float(pkg.get(p, np.nan))
            package_source[p] = "config_fallback"
    env["x_package"] = package
    env["x_package_source"] = package_source
    # the carbon envelope: the most extreme 12C/13C of any CLASSIFIED natural grain
    nat = _type_mask(canon, conf.get("natural_type_regex"))
    c = canon["c12c13"].to_numpy(dtype=float)
    mc = nat & np.isfinite(c)
    if "grain_type" not in canon.columns or not (canon["grain_type"].astype(str).str.strip() != "").any():
        env["carbon"] = {"source": "not_run_no_type_column", "c12c13_max_natural": None}
    elif mc.any():
        imax = int(np.flatnonzero(mc)[np.argmax(c[mc])])
        env["carbon"] = {"source": "empirical_classified_grains", "n_classified_with_c": int(mc.sum()),
                         "c12c13_max_natural": float(c[imax]),
                         "c12c13_max_grain": str(canon["grain_id"].iloc[imax]),
                         "c12c13_max_type": str(canon["grain_type"].iloc[imax])}
    else:
        env["carbon"] = {"source": "not_run_no_classified_grain_with_c", "c12c13_max_natural": None}
    env["thresholds"] = thr
    return env


# ---------------------------------------------------------------------------
# partner tests
# ---------------------------------------------------------------------------
def partner_tests(row, env: dict, conf: dict) -> dict:
    """Per partner: measured?, z to solar, solar-consistent?, z to the X package, excluded?

    ²⁶Al/²⁷Al is one-sided (solar = 0, extinct): a value is solar-consistent
    when it is not resolved above zero; an upper limit is solar-consistent by
    construction and excludes the package when below ``al26_limit_max`` (the X
    grains sit at 0.1–0.6).  The Ti slot uses ⁴⁴Ti/⁴⁸Ti when measured, else
    δ⁴⁴Ca — one slot, so a grain is not double-counted.
    """
    thr = _thr(conf)
    solar = dict(DEFAULT_SOLAR)
    solar.update(conf.get("solar") or {})
    package = env.get("x_package") or DEFAULT_PACKAGE
    ks, kp = float(thr["partner_sigma"]), float(thr["package_sigma"])
    out: dict = {}
    for p in PARTNER_ROLES:
        v, e, lim = _f(row.get(p)), _f(row.get(f"{p}_err")), _f(row.get(f"{p}_limit"))
        rec = {"value": v, "err": e, "limit": lim, "measured": False, "z_solar": np.nan,
               "solar_consistent": None, "z_package": np.nan, "package_excluded": None}
        s, pk = float(solar.get(p, np.nan)), float(package.get(p, np.nan))
        if p == "al26al27":
            if np.isfinite(v) and np.isfinite(e) and e > 0:
                rec.update(measured=True, z_solar=(v - s) / e, z_package=(pk - v) / e if np.isfinite(pk) else np.nan)
                rec["solar_consistent"] = bool(rec["z_solar"] <= ks)
                rec["package_excluded"] = bool(np.isfinite(rec["z_package"]) and rec["z_package"] >= kp)
            elif np.isfinite(lim):
                rec.update(measured=True, z_solar=0.0, solar_consistent=True,
                           package_excluded=bool(lim < float(thr["al26_limit_max"])))
        else:
            if np.isfinite(v) and np.isfinite(e) and e > 0:
                rec.update(measured=True, z_solar=(v - s) / e,
                           z_package=(v - pk) / e if np.isfinite(pk) else np.nan)
                rec["solar_consistent"] = bool(abs(rec["z_solar"]) <= ks)
                rec["package_excluded"] = bool(np.isfinite(rec["z_package"]) and abs(rec["z_package"]) >= kp)
            elif np.isfinite(v) and not (np.isfinite(e) and e > 0):
                rec["note"] = "value without error: not testable"
        out[p] = rec
    # one Ti/Ca slot
    if out["ti44ti48"]["measured"]:
        out["d44ca"] = dict(out["d44ca"], measured=False, note="superseded by 44Ti/48Ti slot")
    return out


def _panel_summary(tests: dict) -> dict:
    measured = [p for p, t in tests.items() if t["measured"]]
    return {
        "n_partners_measured": len(measured),
        "partners_measured": measured,
        "partners_anomalous": [p for p in measured if tests[p]["solar_consistent"] is False],
        "partners_solar": [p for p in measured if tests[p]["solar_consistent"] is True],
        "partners_package_excluded": [p for p in measured if tests[p]["package_excluded"] is True],
        "c_or_n_measured": bool("c12c13" in measured or "n14n15" in measured),
    }


# ---------------------------------------------------------------------------
# per-grain classification
# ---------------------------------------------------------------------------
def _fractionation(d29, e29, d30, e30, thr: dict) -> bool:
    slope = float(thr["fractionation_slope"])
    resid = d29 - slope * d30
    sig = np.sqrt(e29 ** 2 + (slope * e30) ** 2)
    if not np.isfinite(sig) or sig <= 0:
        return False
    small = abs(d30) <= float(thr["fractionation_max_permil"]) and abs(d29) <= float(thr["fractionation_max_permil"])
    resolved = (abs(d30) >= float(thr["fractionation_min_sigma"]) * e30) or \
               (abs(d29) >= float(thr["fractionation_min_sigma"]) * e29)
    return bool(small and resolved and abs(resid) <= 3.0 * sig)


def classify_grain(row, env: dict, conf: dict) -> dict:
    """One grain → class, flags, the numbers behind the decision."""
    thr = _thr(conf)
    k = float(thr["envelope_sigma"])
    d29, e29 = _f(row.get("d29si")), _f(row.get("d29si_err"))
    d30, e30 = _f(row.get("d30si")), _f(row.get("d30si_err"))
    rec: dict = {"grain_id": str(row.get("grain_id", "")), "meteorite": str(row.get("meteorite", "")),
                 "grain_type": str(row.get("grain_type", "")), "d29si": d29, "d29si_err": e29,
                 "d30si": d30, "d30si_err": e30, "beyond_envelope": False, "flags": [],
                 "reason": "", "grade": ""}
    tests = partner_tests(row, env, conf)
    panel = _panel_summary(tests)
    rec.update(panel)
    rec["partner_tests"] = tests
    if not (np.isfinite(d29) and np.isfinite(d30)):
        rec.update({"class": CLASS_NO_SILICON, "reason": "si_value_missing"})
        return rec
    if not (np.isfinite(e29) and np.isfinite(e30) and e29 > 0 and e30 > 0):
        rec.update({"class": CLASS_NO_SILICON, "reason": "si_error_missing"})
        rec["flags"].append("si_err_missing")
        return rec
    min29, min30 = float(env["d29si_min"]), float(env["d30si_min"])
    rec["margin29_sigma"] = (min29 - d29) / e29
    rec["margin30_sigma"] = (min30 - d30) / e30
    beyond = bool(d29 + k * e29 < min29 and d30 + k * e30 < min30)
    rec["beyond_envelope"] = beyond
    wafer = (d29 <= float(thr["wafer_permil"]) and d30 <= float(thr["wafer_permil"])
             and abs(d29 - d30) <= float(thr["wafer_equal_permil"]))
    rec["wafer_like"] = bool(wafer)
    if wafer and not panel["c_or_n_measured"]:
        rec["flags"].append(FLAG_CONTAMINATION)
    if beyond:
        if panel["n_partners_measured"] == 0:
            rec.update({"class": CLASS_INSUFFICIENT, "reason": "no_partner_measured"})
        elif panel["partners_anomalous"]:
            rec.update({"class": CLASS_NATURAL,
                        "reason": "partner_resolved_from_solar:" + ",".join(panel["partners_anomalous"])})
        elif panel["n_partners_measured"] < int(thr["min_partners_for_candidate"]):
            rec.update({"class": CLASS_INSUFFICIENT,
                        "reason": f"only_{panel['n_partners_measured']}_partner_measured"})
        else:
            n_ex = len(panel["partners_package_excluded"])
            grade = "A" if n_ex == panel["n_partners_measured"] else ("B" if n_ex else "C")
            rec.update({"class": CLASS_CANDIDATE, "grade": grade,
                        "reason": f"beyond_envelope_solar_partners_{panel['n_partners_measured']}"
                                  f"_package_excluded_{n_ex}"})
        return rec
    # the carbon analogue
    carbon = env.get("carbon") or {}
    cmax = carbon.get("c12c13_max_natural")
    c, ce = _f(row.get("c12c13")), _f(row.get("c12c13_err"))
    if cmax is not None and np.isfinite(c) and np.isfinite(ce) and ce > 0 \
            and c - float(thr["carbon_envelope_sigma"]) * ce > float(cmax):
        rec["beyond_carbon_envelope"] = True
        ks = float(thr["carbon_si_solar_sigma"])
        si_solar = abs(d29) <= ks * e29 and abs(d30) <= ks * e30
        n = tests["n14n15"]
        if not si_solar:
            rec.update({"class": CLASS_NATURAL, "reason": "carbon_beyond_but_si_anomalous"})
        elif not n["measured"]:
            rec.update({"class": CLASS_INSUFFICIENT, "reason": "carbon_beyond_no_nitrogen"})
        elif n["solar_consistent"] is False:
            rec.update({"class": CLASS_NATURAL, "reason": "carbon_beyond_nitrogen_anomalous"})
        else:
            others = [p for p in ("al26al27", "ti44ti48", "d44ca") if tests[p]["measured"]
                      and tests[p]["solar_consistent"] is False]
            if others:
                rec.update({"class": CLASS_NATURAL,
                            "reason": "carbon_beyond_partner_anomalous:" + ",".join(others)})
            else:
                rec.update({"class": CLASS_CARBON, "grade": "A" if n["package_excluded"] else "B",
                            "reason": "carbon_beyond_envelope_solar_n_solar_si"})
        return rec
    if _fractionation(d29, e29, d30, e30, thr):
        rec.update({"class": CLASS_FRACTIONATION, "reason": "mass_dependent_line"})
        return rec
    rec.update({"class": CLASS_ORDINARY, "reason": "inside_envelope"})
    return rec


def classify_table(canon: pd.DataFrame, conf: dict, *, roles: dict | None = None
                   ) -> tuple[pd.DataFrame, dict, dict]:
    """Every grain → (results frame, envelope, counts).

    ``roles`` (from ``RoleResolution.roles``) lets the counts say which
    partner tests could not run at all because the column was never found.
    """
    env = x_envelope(canon, conf) if len(canon) else {
        "source": "no_table", "d29si_min": np.nan, "d30si_min": np.nan, "hull": [],
        "end_member": [np.nan, np.nan], "x_package": {}, "carbon": {"source": "no_table"}}
    recs = [classify_grain(row, env, conf) for row in canon.to_dict(orient="records")]
    if len(canon):
        d29 = canon["d29si"].to_numpy(dtype=float)
        d30 = canon["d30si"].to_numpy(dtype=float)
        dist = mixing_line_distance(d29, d30, tuple(env["end_member"]))
        inh = inside_hull(np.column_stack([d29, d30]), env.get("hull") or [])
        for r, dd, hh in zip(recs, dist, inh, strict=True):
            r["mixing_line_distance"] = float(dd) if np.isfinite(dd) else np.nan
            r["inside_x_hull"] = bool(hh)
    df = pd.DataFrame(recs) if recs else pd.DataFrame(columns=["grain_id", "class"])
    counts = {c: int((df["class"] == c).sum()) if len(df) else 0 for c in CLASSES}
    counts["n_grains"] = int(len(df))
    counts["n_beyond_envelope"] = int(df["beyond_envelope"].sum()) if "beyond_envelope" in df and len(df) else 0
    counts[FLAG_CONTAMINATION] = int(df["flags"].map(lambda f: FLAG_CONTAMINATION in f).sum()) \
        if "flags" in df and len(df) else 0
    counts["candidates_by_grade"] = {g: int(((df["class"].isin([CLASS_CANDIDATE, CLASS_CARBON]))
                                             & (df["grade"] == g)).sum()) if len(df) else 0
                                     for g in ("A", "B", "C")}
    tests_not_run = []
    if roles:
        for p in PARTNER_ROLES:
            if roles.get(p) is None:
                tests_not_run.append(f"partner:{p}:column_not_found")
            elif roles.get(f"{p}_err") is None and p != "al26al27":
                tests_not_run.append(f"partner:{p}:error_column_not_found")
        for s in ("d29si", "d30si"):
            if roles.get(s) is None:
                tests_not_run.append(f"envelope:{s}:column_not_found")
            elif roles.get(f"{s}_err") is None:
                tests_not_run.append(f"envelope:{s}:error_column_not_found")
        if roles.get("grain_type") is None:
            tests_not_run.append("envelope:grain_type:column_not_found(config_fallback_envelope)")
            tests_not_run.append("carbon:grain_type:column_not_found")
    counts["tests_not_run"] = tests_not_run
    return df, env, counts


def frontier(results: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """The most ²⁸Si-pure grains regardless of class — so nothing hides behind a label."""
    if not len(results) or "d29si" not in results:
        return pd.DataFrame()
    df = results.copy()
    df["purity_rank_metric"] = df["d29si"].astype(float) + df["d30si"].astype(float)
    df = df[np.isfinite(df["purity_rank_metric"])].sort_values("purity_rank_metric")
    cols = [c for c in ("grain_id", "meteorite", "grain_type", "class", "grade", "d29si", "d29si_err",
                        "d30si", "d30si_err", "n_partners_measured", "partners_solar",
                        "partners_anomalous", "flags", "reason", "mixing_line_distance") if c in df]
    return df[cols].head(int(n)).reset_index(drop=True)


def flatten_for_csv(results: pd.DataFrame) -> pd.DataFrame:
    """Per-partner columns instead of the nested dict, for ``grains.csv`` / ``candidates.csv``."""
    if not len(results):
        return pd.DataFrame()
    out = results.drop(columns=[c for c in ("partner_tests",) if c in results]).copy()
    for p in PARTNER_ROLES:
        out[f"{p}"] = results["partner_tests"].map(lambda t, p=p: t[p]["value"])
        out[f"{p}_err"] = results["partner_tests"].map(lambda t, p=p: t[p]["err"])
        out[f"{p}_z_solar"] = results["partner_tests"].map(lambda t, p=p: t[p]["z_solar"])
        out[f"{p}_solar_ok"] = results["partner_tests"].map(lambda t, p=p: t[p]["solar_consistent"])
        out[f"{p}_pkg_excluded"] = results["partner_tests"].map(lambda t, p=p: t[p]["package_excluded"])
    for c in ("flags", "partners_measured", "partners_anomalous", "partners_solar",
              "partners_package_excluded"):
        if c in out:
            out[c] = out[c].map(lambda v: "|".join(v) if isinstance(v, (list, tuple)) else v)
    return out


__all__ = [
    "CLASSES", "CLASS_CANDIDATE", "CLASS_CARBON", "CLASS_FRACTIONATION", "CLASS_INSUFFICIENT",
    "CLASS_NATURAL", "CLASS_NO_SILICON", "CLASS_ORDINARY", "DEFAULT_PACKAGE", "DEFAULT_SOLAR",
    "DEFAULT_THRESHOLDS", "FLAG_CONTAMINATION", "classify_grain", "classify_table",
    "convex_hull_vertices", "flatten_for_csv", "frontier", "inside_hull", "mixing_line_distance",
    "partner_tests", "x_envelope",
]

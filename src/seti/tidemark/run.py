"""TIDEMARK orchestration --- run the three spatial statistics over any anomaly
catalogue that declares a parent sample.

Stages
------
``acquire``    build (or load) a wide-area parent sample with an anomaly axis;
``analyse``    per channel: gradient, edge, age, each against a parent-matched null;
``calibrate``  inject a synthetic front **into the real parent sample** and
               measure what this dataset could actually have detected;
``reduce``     combine channels, write ``results/tidemark/``.

The calibration stage is not decoration.  A null result is only interpretable
next to the sensitivity that produced it, and a sensitivity computed on a
synthetic toy is not the sensitivity of the real footprint.  Injecting into the
actual parent --- with its actual magnitude limit, extinction, crowding and
footprint --- converts "we found nothing" into "a bubble of radius R and contrast
C would have been found at this confidence, and was not".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, load_config
from . import ingest
from .agerate import age_proxies, age_rate_test
from .edge import edge_scan_1d, edge_scan_cap, edge_scan_shell3d
from .gradient import gradient_test
from .inject import inject_bubble
from .nulls import MIN_ANOMALIES_PER_TEST, MatchedNull

CHANNEL = "tidemark"

_DEFAULTS = {
    "matched_null": {"n_feature_bins": 5, "min_pool": 25, "n_null": 500},
    "gradient": {"n_bins": 10,
                 "coordinates": ["R_gal_kpc", "abs_z_gal_kpc", "l_deg"]},
    "edge": {"n_bins_1d": 24, "widths": [1, 2, 3, 4, 6], "n_null_scan": 300,
             "shell_centres_per_axis": 4, "shell_n_bins": 20,
             "cap_directions": 96, "smooth_order": 3, "min_expected": 3.0},
    "age": {"n_bins": 8, "n_null": 400},
    "calibration": {"contrasts": [2.0, 4.0], "radii_pc": [400.0, 900.0],
                    "n_null": 200, "n_trials": 2},
    # Covariates that control detectability.  ``strict`` includes heliocentric
    # distance: the radial gradient must then be a difference *between
    # directions at matched distance*, which no distance-dependent selection
    # effect can fake.  ``permissive`` drops it and is a cross-check only.
    "covariates": {
        "strict": ["phot_g_mean_mag", "dist_pc", "bp_rp", "ebv",
                   "log_local_density", "n_obs"],
        "permissive": ["phot_g_mean_mag", "bp_rp", "ebv", "log_local_density"],
    },
}


def _cfg_block(cfg: Config | None) -> dict:
    block = dict(_DEFAULTS)
    try:
        user = (cfg.thresholds or {}).get("tidemark") or {}
    except Exception:                                        # noqa: BLE001
        user = {}
    for k, v in user.items():
        block[k] = {**block.get(k, {}), **v} if isinstance(v, dict) else v
    return block


def _out_dir(cfg: Config) -> Path:
    try:
        return Path(cfg.path("results_dir")) / CHANNEL
    except Exception:                                        # noqa: BLE001
        return Path(cfg.root) / "results" / CHANNEL


def _strip_private(obj):
    """Remove the ``_inside_mask`` side channel (numpy arrays used for the
    independence check) before anything is serialised."""
    if isinstance(obj, dict):
        # Keys are not always strings (stratum-collapse levels are ints).
        return {k: _strip_private(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(x) for x in o.tolist()]
    return str(o)


# --- covariates, escalation, independence ------------------------------------
#: Column names different channels use for the same physical covariate.  A
#: channel that calls its magnitude ``g_mag`` must not silently go unmatched
#: because the global default list says ``phot_g_mean_mag``.
_COVARIATE_ALIASES = {
    "magnitude": ("phot_g_mean_mag", "g_mag", "gmag", "phot_g_mean_mag_corr",
                  "mag", "vmag", "kmag"),
    "distance": ("dist_pc", "distance_pc", "parallax"),
    "colour": ("bp_rp", "g_rp", "bp_g", "j_k"),
    "extinction": ("ebv", "a_g", "ag_gspphot", "ebpminrp_gspphot"),
    "crowding": ("log_local_density",),
    "epochs": ("n_obs", "n_epochs", "nepochs", "astrometric_n_good_obs_al"),
}
#: Matching without a depth proxy is barely matching at all -- apparent
#: magnitude is the dominant detectability variable in every photometric survey.
_ESSENTIAL_COVARIATE_FAMILIES = ("magnitude",)


def _resolve_covariates(cat, parent, blocks: dict, mode: str):
    """Union the channel's declared covariates with the global defaults.

    The channel knows its own column names; the global list knows which
    *physical* covariates matter.  Using only the global list silently drops any
    covariate the channel spells differently --- which is how a 255k-star
    catalogue got matched on three columns and no magnitude at all.
    """
    declared = [c for c in (cat.covariates or ())]
    defaults = list(blocks["covariates"][mode])
    requested = declared + [c for c in defaults if c not in declared]
    if mode == "permissive":
        requested = [c for c in requested if c not in _COVARIATE_ALIASES["distance"]]
    used = [c for c in requested if c in parent.columns]
    missing = [c for c in requested if c not in parent.columns]

    families, warnings = {}, []
    for fam, names in _COVARIATE_ALIASES.items():
        hit = next((c for c in used if c in names), None)
        families[fam] = hit
        if hit is None and fam in _ESSENTIAL_COVARIATE_FAMILIES:
            warnings.append(
                f"no {fam} covariate is available in this parent sample "
                f"(looked for {list(names)}); the matched null cannot correct for "
                "the dominant detectability variable and any structure it finds "
                "may simply be a depth map")
    return used, {"declared_by_channel": declared, "requested": requested,
                  "used": used, "missing_from_parent": missing,
                  "families_matched": families, "warnings": warnings,
                  "essential_covariates_present": not warnings}


def _escalate(fn, n_start: int, n_max: int, max_rounds: int = 2) -> dict:
    """Re-run a statistic with more null draws while its p-value sits on the
    Monte Carlo floor.

    ``p == 1/(n_null+1)`` says only "no null realisation was this extreme".  It
    is a bound.  Escalating turns it into a measurement, or --- if it survives
    the cap --- into an honest inequality that the verdict logic must not treat
    as a resolved detection.
    """
    n = int(n_start)
    res = fn(n)
    rounds = 0
    while (isinstance(res, dict) and res.get("floor_limited")
           and n < n_max and rounds < max_rounds):
        n = min(n * 8, int(n_max))
        res = fn(n)
        rounds += 1
    if isinstance(res, dict):
        res["escalation"] = {"n_null_final": n, "rounds": rounds,
                             "capped": bool(res.get("floor_limited"))}
    return res


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.sum(a & b))
    union = float(np.sum(a | b))
    return inter / union if union > 0 else 0.0


def _independence(tests: dict, threshold: float = 0.5) -> dict:
    """Group tests that fired on substantially the same anomalies.

    Three "independent geometries" returning the same p-value are not three
    pieces of evidence.  Each edge geometry reports which anomalies produced its
    step; tests whose firing sets overlap by more than ``threshold`` (Jaccard)
    are one feature seen three ways, and the trials correction must count them
    once.
    """
    names = [k for k, v in tests.items() if isinstance(v.get("_inside_mask"), np.ndarray)]
    overlaps, groups = {}, []
    assigned: dict[str, int] = {}
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            j = _jaccard(tests[a]["_inside_mask"], tests[bname]["_inside_mask"])
            overlaps[f"{a}|{bname}"] = round(float(j), 4)
    for a in names:
        placed = False
        for gi, grp in enumerate(groups):
            if any(overlaps.get(f"{m}|{a}", overlaps.get(f"{a}|{m}", 0.0)) >= threshold
                   for m in grp):
                grp.append(a)
                assigned[a] = gi
                placed = True
                break
        if not placed:
            assigned[a] = len(groups)
            groups.append([a])
    n_without_mask = len([k for k in tests if k not in names])
    return {"pairwise_jaccard": overlaps,
            "groups": [sorted(g) for g in groups],
            "n_independent_groups": len(groups) + n_without_mask,
            "threshold": threshold,
            "note": ("edge geometries whose firing anomaly sets overlap above the "
                     "threshold are counted once; identical p-values across "
                     "'independent' geometries usually mean one feature")}


# --- the per-channel analysis ----------------------------------------------
def analyse_catalogue(cat: ingest.AnomalyCatalogue, *, blocks: dict | None = None,
                      seed: int = 20260726, quick: bool = False) -> dict:
    """Run every spatial statistic the catalogue can support.

    Returns a dict whose ``verdict`` is the honest state of the measurement: a
    channel with no parent sample, no positions or too few anomalies is reported
    as such rather than given a p-value it cannot support.
    """
    b = {**_DEFAULTS, **(blocks or {})}
    out = {"channel": cat.name, **cat.summary()}
    if not cat.usable:
        out["tested"] = False
        return out

    parent = age_proxies(cat.parent)
    mask = cat.anomaly_mask
    n_null = int(b["matched_null"]["n_null"]) if not quick else 60
    n_scan = int(b["edge"]["n_null_scan"]) if not quick else 40
    max_null = int(b["matched_null"].get("max_n_null", 8000)) if not quick else 240
    max_scan = int(b["edge"].get("max_n_null_scan", 4000)) if not quick else 160

    out["tested"] = True
    out["modes"] = {}
    out["covariate_resolution"] = {}
    for mode in ("strict", "permissive"):
        covs, cov_report = _resolve_covariates(cat, parent, b, mode)
        out["covariate_resolution"][mode] = cov_report
        if not covs:
            out["modes"][mode] = {"error": "no detectability covariate available",
                                  "covariate_resolution": cov_report}
            continue
        try:
            null = MatchedNull(parent, mask, covs,
                               n_bins=int(b["matched_null"]["n_feature_bins"]),
                               min_pool=int(b["matched_null"]["min_pool"]), seed=seed)
        except (ValueError, KeyError) as exc:
            out["modes"][mode] = {"error": str(exc)}
            continue

        res = {"covariates": covs, "covariate_resolution": cov_report,
               "null_diagnostics": null.diagnostics(
                   extra_balance_cols=("R_gal_kpc", "abs_z_gal_kpc")).as_dict(),
               "gradient": {}, "edge": {}}

        # --- gradient over each coordinate ---------------------------------
        for coord in b["gradient"]["coordinates"]:
            if coord not in parent.columns:
                continue
            x = parent[coord].to_numpy(float)
            if not np.isfinite(x).any():
                continue
            res["gradient"][coord] = _escalate(
                lambda nn, _x=x, _c=coord, null=null: gradient_test(
                    _x, null, name=_c, n_bins=int(b["gradient"]["n_bins"]),
                    n_null=nn, periodic=_c.startswith("l_"), seed=seed + 1),
                n_null, max_null)

        # --- edge, three geometries ----------------------------------------
        smooth = {c: parent[c].to_numpy(float)
                  for c in ("R_gal_kpc", "abs_z_gal_kpc") if c in parent.columns}
        for coord in ("R_gal_kpc", "abs_z_gal_kpc"):
            if coord in parent.columns and np.isfinite(parent[coord]).any():
                res["edge"][coord] = _escalate(
                    lambda nn, _c=coord, null=null, parent=parent: edge_scan_1d(
                        parent[_c].to_numpy(float), null, name=_c,
                        n_bins=int(b["edge"]["n_bins_1d"]),
                        widths=tuple(b["edge"]["widths"]), n_null=nn,
                        smooth_order=int(b["edge"]["smooth_order"]),
                        min_expected=float(b["edge"]["min_expected"]), seed=seed + 2),
                    n_scan, max_scan)
        if {"X_pc", "Y_pc", "Z_pc"} <= set(parent.columns):
            xyz = parent[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
            if np.isfinite(xyz).all(axis=1).sum() > 100:
                res["edge"]["shell_3d"] = _escalate(
                    lambda nn, null=null, smooth=smooth, xyz=xyz: edge_scan_shell3d(
                        xyz, null, n_per_axis=int(b["edge"]["shell_centres_per_axis"]),
                        n_bins=int(b["edge"]["shell_n_bins"]),
                        widths=tuple(b["edge"]["widths"][:4]), n_null=nn,
                        min_expected=float(b["edge"]["min_expected"]),
                        smooth_coords=smooth,
                        smooth_order=int(b["edge"]["smooth_order"]), seed=seed + 3),
                    n_scan, max_scan)
        if {"l_deg", "b_deg"} <= set(parent.columns):
            res["edge"]["sky_cap"] = _escalate(
                lambda nn, null=null, parent=parent: edge_scan_cap(
                    parent["l_deg"].to_numpy(float), parent["b_deg"].to_numpy(float),
                    null, n_directions=int(b["edge"]["cap_directions"]),
                    widths=tuple(b["edge"]["widths"][:4]), n_null=nn,
                    min_expected=float(b["edge"]["min_expected"]), seed=seed + 4),
                n_scan, max_scan)
        out["modes"][mode] = res

    # --- age (the filter clock) --------------------------------------------
    out["age"] = age_rate_test(
        parent, mask, covariates=[c for c in b["covariates"]["strict"]
                                  if c in parent.columns and c != "dist_pc"],
        n_bins=int(b["age"]["n_bins"]),
        n_null=int(b["age"]["n_null"]) if not quick else 60, seed=seed + 5)

    # --- headline ------------------------------------------------------------
    # Every p-value is accompanied by (a) whether it is resolved or sitting on
    # the Monte Carlo floor, (b) how many anomalies actually entered it, and
    # (c) the residual imbalance of the coordinate it tested.  A number without
    # those three is not interpretable, and a verdict built on such a number is
    # how a floor artefact becomes a "detection".
    strict = out["modes"].get("strict", {})
    entries: dict = {}
    for fam, block in (("gradient", strict.get("gradient") or {}),
                       ("edge", strict.get("edge") or {})):
        for k, v in block.items():
            if not isinstance(v, dict):
                continue
            key = f"{fam}:{k}"
            pv = v.get("headline_p") if fam == "gradient" else v.get("p_value")
            entries[key] = {
                "p_value": pv,
                "p_repr": v.get("p_repr"),
                "floor_limited": bool(v.get("floor_limited")),
                "insufficient": bool(v.get("insufficient")),
                "verdict": v.get("verdict"),
                "reason": v.get("reason"),
                "n_anom": v.get("n_anom"),
                "n_anom_total": v.get("n_anom_total", cat.n_anomaly),
                "coordinate_balance": (v.get("coordinate_balance") or {}).get("std_diff"),
                "balance_quality": (v.get("coordinate_balance") or {}).get("quality"),
                "n_null": (v.get("escalation") or {}).get("n_null_final"),
                "_inside_mask": v.get("_inside_mask"),
            }
    age = out.get("age") or {}
    if age.get("shape_p_value") is not None:
        entries["age:shape"] = {"p_value": age.get("shape_p_value"),
                                "p_repr": f"{age.get('shape_p_value'):.4g}",
                                "floor_limited": False, "insufficient": False,
                                "verdict": "OK", "n_anom": cat.n_anomaly,
                                "n_anom_total": cat.n_anomaly,
                                "coordinate_balance": None, "_inside_mask": None}

    indep = _independence({k: v for k, v in entries.items()
                           if v.get("_inside_mask") is not None})
    for v in entries.values():
        v.pop("_inside_mask", None)

    # Only *resolved, sufficiently-powered* tests can support a claim.
    usable = {k: v for k, v in entries.items()
              if v["p_value"] is not None and np.isfinite(v["p_value"])
              and not v["insufficient"]}
    resolved = {k: v for k, v in usable.items() if not v["floor_limited"]}
    n_eff = max(indep["n_independent_groups"], 1)

    best = min(resolved, key=lambda k: resolved[k]["p_value"]) if resolved else None
    best_p = resolved[best]["p_value"] if best else None
    fam_p = (1.0 - (1.0 - best_p) ** n_eff) if best_p is not None else None

    out["tests"] = entries
    out["independence"] = indep
    out["n_tests_run"] = len(entries)
    out["n_tests_usable"] = len(usable)
    out["n_tests_resolved"] = len(resolved)
    out["n_effective_independent_tests"] = n_eff
    out["insufficient_tests"] = {k: v.get("reason") for k, v in entries.items()
                                 if v["insufficient"]}
    out["floor_limited_tests"] = {k: v["p_repr"] for k, v in usable.items()
                                  if v["floor_limited"]}
    out["p_values"] = {k: v["p_value"] for k, v in usable.items()}
    out["p_values_repr"] = {k: v["p_repr"] for k, v in entries.items()}
    out["best_test"] = best
    out["best_p"] = best_p
    out["best_p_family_corrected"] = fam_p
    # Retained under the old name so downstream readers do not silently get None.
    out["best_p_trials_corrected"] = fam_p

    # --- verdict gates -------------------------------------------------------
    cov = (out.get("covariate_resolution") or {}).get("strict") or {}
    bal = resolved[best]["coordinate_balance"] if best else None
    gates = {
        "any_resolved_test": bool(resolved),
        "family_p_below_alpha": bool(fam_p is not None and fam_p < 0.05),
        "not_floor_limited": bool(best is not None),
        "tested_coordinate_balanced": bool(
            bal is None or abs(bal) < 0.10),
        "essential_covariates_present": bool(cov.get("essential_covariates_present", False)),
        "anomaly_population_vetted": bool(getattr(cat, "vetted", False)),
        "sufficient_anomalies_in_winning_test": bool(
            best is not None
            and (resolved[best]["n_anom"] or 0) >= MIN_ANOMALIES_PER_TEST),
    }
    failed = [k for k, v in gates.items() if not v]
    out["verdict_gates"] = gates
    out["failed_gates"] = failed

    if not usable:
        out["result"] = "NOT_TESTABLE"
    elif not gates["any_resolved_test"] and out["floor_limited_tests"]:
        out["result"] = "STRUCTURE_UNRESOLVED"
    elif not gates["family_p_below_alpha"]:
        out["result"] = "CLEAN_NULL"
    elif not gates["essential_covariates_present"]:
        out["result"] = "STRUCTURE_UNCORRECTED"
    elif not gates["anomaly_population_vetted"]:
        out["result"] = "STRUCTURE_UNVETTED_POPULATION"
    elif not gates["tested_coordinate_balanced"]:
        out["result"] = "STRUCTURE_CONFOUNDED"
    else:
        out["result"] = "DETECTION"
    out["detection"] = bool(out["result"] == "DETECTION")
    out["result_explanation"] = _RESULT_MEANING.get(out["result"], "")
    if getattr(cat, "caveat", None):
        out["population_caveat"] = cat.caveat
        out["result"] = out["result"] + " [" + cat.caveat_tag + "]" \
            if out["result"] != "CLEAN_NULL" else out["result"]
    return out


#: What each result string commits to.  Kept next to the logic so a reader of
#: the JSON never has to guess how strong a claim is being made.
_RESULT_MEANING = {
    "DETECTION": "resolved, family-corrected, covariate-balanced structure in a "
                 "vetted anomaly population",
    "STRUCTURE_UNRESOLVED": "the most extreme statistic never exceeded any null "
                            "realisation even after escalating the draw count; "
                            "the p-value is a bound, not a measurement",
    "STRUCTURE_UNCORRECTED": "significant, but the parent sample lacks a covariate "
                             "the matched null needs (typically apparent "
                             "magnitude); the structure may be a depth map",
    "STRUCTURE_UNVETTED_POPULATION": "significant, but the anomaly set is a bare "
                                     "score percentile rather than a vetted "
                                     "candidate list; most likely traces the "
                                     "survey rather than the sky",
    "STRUCTURE_CONFOUNDED": "significant, but the tested coordinate is itself "
                            "poorly balanced between anomalies and the matched "
                            "null (|SMD| >= 0.10), which is the leading route to "
                            "a spurious gradient or edge",
    "CLEAN_NULL": "no structure beyond the parent-matched null",
    "NOT_TESTABLE": "no statistic could be computed on this catalogue",
}


# --- injection calibration on the real parent -------------------------------
def gradient_transfer(cat: ingest.AnomalyCatalogue, *, coord: str = "R_gal_kpc",
                      blocks: dict | None = None, slopes=(0.0, 0.3, 0.6),
                      seed: int = 909, quick: bool = False) -> dict:
    """Measure how much of a *known* injected gradient this test recovers.

    This is the correction's honest price tag.  The matched null is built from
    detectability covariates, and those covariates (apparent magnitude, distance,
    extinction, crowding) are themselves correlated with Galactic position --- so
    matching on them absorbs part of any real positional trend along with the
    selection effect.  The test is therefore **conservative**: it under-reports a
    true gradient by a factor that this function measures.

    Returns the fitted transfer coefficient ``recovered / injected``.  A measured
    slope should be read as an intrinsic slope of at least ``measured /
    transfer``; quoting the raw measured slope as the astrophysical amplitude
    would understate it, and quoting it without this calibration at all would be
    the sort of unstated systematic this project exists to avoid.
    """
    b = {**_DEFAULTS, **(blocks or {})}
    if coord not in cat.parent.columns or cat.n_parent < 500:
        return {"performed": False, "reason": f"{coord} unavailable"}
    from .inject import inject_gradient
    parent = cat.parent
    base = max(cat.n_anomaly / max(cat.n_parent, 1), 1e-4)
    covs = [c for c in b["covariates"]["strict"] if c in parent.columns]
    x = parent[coord].to_numpy(float)
    n_null = 120 if quick else 250
    rows = []
    for s in slopes:
        m = inject_gradient(parent, coord=coord, slope_ln_per_unit=float(s),
                            base_rate=base, seed=seed + int(1000 * s),
                            detect_strength=0.0)
        if m.sum() < 30:
            continue
        try:
            null = MatchedNull(parent, m, covs,
                               n_bins=int(b["matched_null"]["n_feature_bins"]),
                               min_pool=int(b["matched_null"]["min_pool"]), seed=seed)
        except (ValueError, KeyError):
            continue
        g = gradient_test(x, null, name=coord, n_bins=int(b["gradient"]["n_bins"]),
                          n_null=n_null, seed=seed + 1)
        rows.append({"injected_ln_per_unit": float(s),
                     "recovered_ln_per_unit": g.get("slope_ln_per_unit"),
                     "p_value": (g.get("terms", {}).get("slope") or {}).get("p_monte_carlo"),
                     "n_injected": int(m.sum())})
    inj = np.array([r["injected_ln_per_unit"] for r in rows], float)
    rec = np.array([r["recovered_ln_per_unit"] or np.nan for r in rows], float)
    ok = np.isfinite(inj) & np.isfinite(rec)
    transfer = (float(np.sum(inj[ok] * rec[ok]) / np.sum(inj[ok] ** 2))
                if ok.sum() >= 2 and np.sum(inj[ok] ** 2) > 0 else None)
    return {"performed": True, "coordinate": coord, "trials": rows,
            "transfer_coefficient": transfer,
            "interpretation": (
                "measured slope / transfer_coefficient estimates the intrinsic "
                "slope; the deficit is signal absorbed by matching on covariates "
                "that themselves correlate with Galactic position")}


def calibrate(cat: ingest.AnomalyCatalogue, *, blocks: dict | None = None,
              seed: int = 4242, quick: bool = False) -> dict:
    """Inject a synthetic sharp-edged bubble into the **real** parent sample and
    report what contrast/radius this dataset could actually have detected."""
    b = {**_DEFAULTS, **(blocks or {})}
    if cat.n_parent < 500 or not {"X_pc", "Y_pc", "Z_pc"} <= set(cat.parent.columns):
        return {"performed": False, "reason": "parent sample too small or 3D-less"}
    parent = cat.parent
    base = max(cat.n_anomaly / max(cat.n_parent, 1), 1e-4)
    xyz = parent[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
    ok = np.isfinite(xyz).all(axis=1)
    centre = np.median(xyz[ok], axis=0) if ok.any() else np.zeros(3)
    covs = [c for c in b["covariates"]["strict"] if c in parent.columns]
    n_null = int(b["calibration"]["n_null"]) if not quick else 40
    rows = []
    for radius in b["calibration"]["radii_pc"]:
        for contrast in b["calibration"]["contrasts"]:
            for trial in range(int(b["calibration"]["n_trials"]) if not quick else 1):
                m = inject_bubble(parent, centre_pc=tuple(centre), radius_pc=float(radius),
                                  contrast=float(contrast), base_rate=base,
                                  seed=seed + trial + int(radius) + int(100 * contrast),
                                  detect_strength=0.0)
                if m.sum() < 30:
                    rows.append({"radius_pc": radius, "contrast": contrast,
                                 "trial": trial, "n_injected": int(m.sum()),
                                 "recovered": None,
                                 "note": "too few injected anomalies to test"})
                    continue
                try:
                    null = MatchedNull(parent, m, covs,
                                       n_bins=int(b["matched_null"]["n_feature_bins"]),
                                       min_pool=int(b["matched_null"]["min_pool"]), seed=seed)
                except (ValueError, KeyError) as exc:
                    rows.append({"radius_pc": radius, "contrast": contrast,
                                 "trial": trial, "error": str(exc)})
                    continue
                sc = edge_scan_shell3d(xyz, null, n_per_axis=3,
                                       n_bins=int(b["edge"]["shell_n_bins"]),
                                       widths=(1, 2, 3), n_null=n_null,
                                       min_expected=float(b["edge"]["min_expected"]),
                                       seed=seed + 9)
                rows.append({"radius_pc": float(radius), "contrast": float(contrast),
                             "trial": int(trial), "n_injected": int(m.sum()),
                             "p_value": sc.get("p_value"),
                             "max_abs_score": sc.get("max_abs_score"),
                             "recovered": bool(sc.get("significant"))})
    got = [r for r in rows if r.get("recovered") is not None]
    return {"performed": True, "base_rate": base,
            "centre_pc": [float(v) for v in centre],
            "n_configurations": len(rows), "trials": rows,
            "recovery_fraction": (float(np.mean([r["recovered"] for r in got]))
                                  if got else None),
            "note": "injected into the real parent sample, so the sensitivity "
                    "quoted is this footprint's, not a toy's"}


# --- the run ----------------------------------------------------------------
def tidemark_run(cfg: Config | None = None, *, channels=None, catalogues=None,
                 stage: str = "all", quick: bool = False, seed: int = 20260726,
                 out_dir: Path | None = None, do_calibrate: bool = True) -> dict:
    """Run TIDEMARK over the configured anomaly channels (or supplied catalogues)."""
    cfg = cfg or load_config()
    blocks = _cfg_block(cfg)
    base = Path(out_dir) if out_dir else _out_dir(cfg)
    base.mkdir(parents=True, exist_ok=True)

    if catalogues is None:
        specs = _load_specs(cfg)
        if channels:
            specs = {k: v for k, v in specs.items() if k in set(channels)}
        cats = list(ingest.load_all(cfg.root, specs).values())
    else:
        cats = list(catalogues)

    usable = [c for c in cats if c.usable]
    if len(usable) > 1:
        try:
            cats.append(ingest.union_catalogue(usable))
        except Exception as exc:                             # noqa: BLE001
            print(f"[tidemark] union failed: {exc!r}")

    results, per_channel = [], {}
    for cat in cats:
        print(f"[tidemark] {cat.name}: verdict={cat.verdict} "
              f"n_parent={cat.n_parent} n_anom={cat.n_anomaly}")
        res = analyse_catalogue(cat, blocks=blocks, seed=seed, quick=quick)
        if do_calibrate and cat.usable and stage in ("all", "calibrate"):
            res["calibration"] = calibrate(cat, blocks=blocks, quick=quick)
            res["gradient_transfer"] = gradient_transfer(cat, blocks=blocks,
                                                         quick=quick)
            # Restate the measured amplitude with the absorption divided back out.
            tc = res["gradient_transfer"].get("transfer_coefficient")
            meas = ((res.get("modes", {}).get("strict", {}).get("gradient", {})
                     .get("R_gal_kpc") or {}).get("slope_ln_per_unit"))
            if tc and meas is not None and np.isfinite(meas) and tc > 0.05:
                res["R_gal_slope_deabsorbed_ln_per_kpc"] = float(meas / tc)
        results.append(res)
        per_channel[cat.name] = res
        cdir = base / cat.name
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "summary.json").write_text(
            json.dumps(_strip_private(res), indent=2, default=_jsonable))
        # The selection function itself, per star: the auditable deliverable.
        if cat.usable:
            try:
                covs = [c for c in blocks["covariates"]["strict"] if c in cat.parent.columns]
                null = MatchedNull(cat.parent, cat.anomaly_mask, covs,
                                   n_bins=int(blocks["matched_null"]["n_feature_bins"]),
                                   min_pool=int(blocks["matched_null"]["min_pool"]), seed=seed)
                cols = [c for c in (cat.id_col, "ra", "dec", "l_deg", "b_deg",
                                    "dist_pc", "R_gal_kpc", "z_gal_kpc")
                        if c in cat.parent.columns]
                w = cat.parent[cols].copy()
                w["selection_weight"] = null.weights
                w["is_anomaly"] = cat.anomaly_mask
                w[w["is_anomaly"] | (np.arange(len(w)) % 20 == 0)].to_csv(
                    cdir / "selection_weights.csv", index=False)
            except Exception as exc:                          # noqa: BLE001
                print(f"[tidemark] weight export failed for {cat.name}: {exc!r}")

    verdicts = {c.name: c.verdict for c in cats}
    tested = [r for r in results if r.get("tested")]
    detections = [r["channel"] for r in tested if r.get("detection")]
    # Rank results by how strong a claim they represent; the run's verdict is
    # the strongest claim any channel actually earned, never an aggregate of
    # unresolved or ungated ones.
    order = ["DETECTION", "STRUCTURE_CONFOUNDED", "STRUCTURE_UNVETTED_POPULATION",
             "STRUCTURE_UNCORRECTED", "STRUCTURE_UNRESOLVED", "CLEAN_NULL",
             "NOT_TESTABLE"]
    seen = [str(r.get("result", "NOT_TESTABLE")).split(" [")[0] for r in tested]
    verdict = next((s for s in order if s in seen), "NO_TESTABLE_CATALOGUE")
    if not tested:
        verdict = "NO_TESTABLE_CATALOGUE"
    summary = {
        "channel": CHANNEL,
        "verdict": verdict,
        "verdict_meaning": _RESULT_MEANING.get(verdict,
                                               "no catalogue supplied a parent sample"),
        "n_catalogues": len(cats), "n_tested": len(tested),
        "n_untested": len(cats) - len(tested),
        "catalogue_verdicts": verdicts,
        "results_by_channel": {r["channel"]: r.get("result") for r in tested},
        "detections": detections,
        "failed_gates_by_channel": {r["channel"]: r.get("failed_gates")
                                    for r in tested},
        "best_p_by_channel": {r["channel"]: r.get("best_p_family_corrected")
                              for r in tested},
        "p_repr_by_channel": {r["channel"]: r.get("p_values_repr") for r in tested},
        "insufficient_tests_by_channel": {r["channel"]: r.get("insufficient_tests")
                                          for r in tested},
        "channels": per_channel,
        "predictions_discriminated": _PREDICTIONS,
        "reporting_rules": {
            "floor_limited_p": "reported as an inequality (p_repr), never as a "
                               "point estimate; escalated before being believed",
            "trials_correction": "Sidak over n_effective_independent_tests, where "
                                 "edge geometries firing on overlapping anomaly "
                                 "sets are counted once",
            "insufficient": "a statistic with fewer than "
                            f"{MIN_ANOMALIES_PER_TEST} anomalies carrying its own "
                            "coordinate returns INSUFFICIENT_ANOMALIES, not a p-value",
        },
    }
    (base / "summary.json").write_text(
        json.dumps(_strip_private(summary), indent=2, default=_jsonable))
    print("[tidemark]", json.dumps({"verdict": verdict, "n_tested": len(tested),
                                    "results": summary["results_by_channel"],
                                    "verdicts": verdicts}, default=_jsonable))
    return summary


#: The three published, mutually exclusive spatial predictions this channel can
#: tell apart.  Recorded in the output so a result is read against them.
_PREDICTIONS = {
    "outward_rim": {
        "source": "Cirkovic & Bradbury 2006, New Astronomy 11, 628 "
                  "(doi 10.1016/j.newast.2006.04.003; astro-ph/0506110)",
        "prediction": "postbiological ETI migrates outward; maximum probability in a "
                      "ring on the periphery of the Milky Way",
        "signature": "positive slope of anomaly rate with R_gal",
    },
    "inward_centre": {
        "source": "Wright, Carroll-Nellenback, Frank & Scharf 2021, RNAAS 5, 141 "
                  "(doi 10.3847/2515-5172/ac0910)",
        "prediction": "settlement wavefronts strongly biased toward the Galactic centre",
        "signature": "negative slope of anomaly rate with R_gal",
    },
    "no_structure": {
        "source": "Wright et al. 2014 (G-hat I), ApJ 792, 26 (arXiv:1408.1133)",
        "prediction": "rotational shear mixes any Fermi bubble on a rotation "
                      "timescale, so no coherent front should survive in a disk",
        "signature": "no gradient and no edge beyond the matched null",
    },
    "bounded_region": {
        "source": "Carrigan 2010, JBIS 63, 90 (arXiv:1001.5455); "
                  "Landis 1998, JBIS 51, 163; Hanson et al. 2021 (arXiv:2102.01522)",
        "prediction": "a colonised region whose boundary is the observable",
        "signature": "a sharp step in rate at a shell radius / cap edge",
    },
}


def _load_specs(cfg: Config) -> dict:
    p = Path(cfg.root) / "config" / "tidemark.yaml"
    if not p.exists():
        return {}
    import yaml
    doc = yaml.safe_load(p.read_text()) or {}
    return doc.get("channels") or {}


def tidemark_selfsearch(cfg: Config | None = None, *, table: pd.DataFrame | None = None,
                        grid: str = "sparse", excess_z_min: float = 4.0,
                        plx_min: float = 1.0, g_max: float = 17.0,
                        radius_deg: float = 6.0, limit: int = 400000,
                        parent_glob: str | None = None,
                        quick: bool = False, seed: int = 20260726) -> dict:
    """Self-sufficient path: build a wide-area Gaia x AllWISE parent sample with
    an IR-excess anomaly axis, then run the full TIDEMARK analysis on it.

    This exists so the channel is not blocked on sibling channels publishing
    their parent samples.  The critical detail is in ``acquire.parent_sample``:
    the excess locus is fitted **globally**, over all cones at once.  Fitting it
    per cone would normalise every field to its own median and delete exactly the
    field-to-field rate differences this channel exists to measure.
    """
    cfg = cfg or load_config()
    if table is None and parent_glob:
        # Reduce path: concatenate the sharded acquisition, then fit the excess
        # locus ONCE over the whole sky (never per shard).
        frames = []
        for f in sorted(Path(cfg.root).glob(parent_glob)):
            try:
                frames.append(pd.read_parquet(f))
                print(f"[tidemark] read {f.name}: {len(frames[-1])} rows")
            except Exception as exc:                          # noqa: BLE001
                print(f"[tidemark] could not read {f}: {exc!r}")
        if frames:
            table = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
            print(f"[tidemark] combined parent sample: {len(table)} stars")
    if table is None:
        from .acquire import parent_sample
        table = parent_sample(grid=grid, radius_deg=radius_deg, plx_min=plx_min,
                              g_max=g_max, limit=limit)
    if table is None or not len(table):
        base = _out_dir(cfg)
        base.mkdir(parents=True, exist_ok=True)
        summary = {"channel": CHANNEL, "verdict": ingest.NO_DATA_REACHED,
                   "note": "no rows returned by the archive"}
        (base / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    from .acquire import excess_axis
    parent = excess_axis(table)
    cat = ingest.from_frames("gaia_allwise_ir_excess", parent,
                             mask=(parent["ir_excess_z"].to_numpy(float) >= excess_z_min),
                             score_col="ir_excess_z",
                             provenance={"grid": grid, "excess_z_min": excess_z_min,
                                         "n_cones": int(parent.get("cone", pd.Series(
                                             dtype=float)).nunique() or 0)})
    return tidemark_run(cfg, catalogues=[cat], quick=quick, seed=seed)


__all__ = ["tidemark_run", "tidemark_selfsearch", "analyse_catalogue", "calibrate"]

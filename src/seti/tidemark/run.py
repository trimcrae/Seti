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
from .nulls import MatchedNull

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

    out["tested"] = True
    out["modes"] = {}
    for mode in ("strict", "permissive"):
        covs = [c for c in b["covariates"][mode] if c in parent.columns]
        if not covs:
            out["modes"][mode] = {"error": "no detectability covariate available"}
            continue
        try:
            null = MatchedNull(parent, mask, covs,
                               n_bins=int(b["matched_null"]["n_feature_bins"]),
                               min_pool=int(b["matched_null"]["min_pool"]), seed=seed)
        except (ValueError, KeyError) as exc:
            out["modes"][mode] = {"error": str(exc)}
            continue

        res = {"covariates": covs,
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
            res["gradient"][coord] = gradient_test(
                x, null, name=coord, n_bins=int(b["gradient"]["n_bins"]),
                n_null=n_null, periodic=coord.startswith("l_"),
                seed=seed + 1)

        # --- edge, three geometries ----------------------------------------
        smooth = {c: parent[c].to_numpy(float)
                  for c in ("R_gal_kpc", "abs_z_gal_kpc") if c in parent.columns}
        for coord in ("R_gal_kpc", "abs_z_gal_kpc"):
            if coord in parent.columns and np.isfinite(parent[coord]).any():
                res["edge"][coord] = edge_scan_1d(
                    parent[coord].to_numpy(float), null, name=coord,
                    n_bins=int(b["edge"]["n_bins_1d"]),
                    widths=tuple(b["edge"]["widths"]), n_null=n_scan,
                    smooth_order=int(b["edge"]["smooth_order"]),
                    min_expected=float(b["edge"]["min_expected"]), seed=seed + 2)
        if {"X_pc", "Y_pc", "Z_pc"} <= set(parent.columns):
            xyz = parent[["X_pc", "Y_pc", "Z_pc"]].to_numpy(float)
            if np.isfinite(xyz).all(axis=1).sum() > 100:
                res["edge"]["shell_3d"] = edge_scan_shell3d(
                    xyz, null, n_per_axis=int(b["edge"]["shell_centres_per_axis"]),
                    n_bins=int(b["edge"]["shell_n_bins"]),
                    widths=tuple(b["edge"]["widths"][:4]), n_null=n_scan,
                    min_expected=float(b["edge"]["min_expected"]),
                    smooth_coords=smooth,
                    smooth_order=int(b["edge"]["smooth_order"]), seed=seed + 3)
        if {"l_deg", "b_deg"} <= set(parent.columns):
            res["edge"]["sky_cap"] = edge_scan_cap(
                parent["l_deg"].to_numpy(float), parent["b_deg"].to_numpy(float),
                null, n_directions=int(b["edge"]["cap_directions"]),
                widths=tuple(b["edge"]["widths"][:4]), n_null=n_scan,
                min_expected=float(b["edge"]["min_expected"]), seed=seed + 4)
        out["modes"][mode] = res

    # --- age (the filter clock) --------------------------------------------
    out["age"] = age_rate_test(
        parent, mask, covariates=[c for c in b["covariates"]["strict"]
                                  if c in parent.columns and c != "dist_pc"],
        n_bins=int(b["age"]["n_bins"]),
        n_null=int(b["age"]["n_null"]) if not quick else 60, seed=seed + 5)

    # --- headline ----------------------------------------------------------
    strict = out["modes"].get("strict", {})
    grad_p = {k: v.get("headline_p") for k, v in (strict.get("gradient") or {}).items()
              if isinstance(v, dict)}
    edge_p = {k: v.get("p_value") for k, v in (strict.get("edge") or {}).items()
              if isinstance(v, dict)}
    all_p = {f"gradient:{k}": v for k, v in grad_p.items() if v is not None}
    all_p.update({f"edge:{k}": v for k, v in edge_p.items() if v is not None})
    age_p = (out["age"] or {}).get("shape_p_value")
    if age_p is not None:
        all_p["age:shape"] = age_p
    finite = {k: v for k, v in all_p.items() if v is not None and np.isfinite(v)}
    n_tests = max(len(finite), 1)
    best = min(finite, key=finite.get) if finite else None
    out["p_values"] = finite
    out["best_test"] = best
    out["best_p"] = finite.get(best) if best else None
    # Sidak correction over the family of tests actually run.  Deliberately
    # conservative: these tests are *correlated* (a bubble offset from the Sun
    # produces a radial gradient, a longitude dipole and a shell edge at once),
    # so treating them as independent over-corrects rather than under-corrects.
    out["best_p_trials_corrected"] = (1.0 - (1.0 - finite[best]) ** n_tests) if best else None
    out["n_tests_in_family"] = n_tests
    # Monte Carlo resolution floor: a p-value at the floor means "as extreme as
    # any draw produced", not "exactly this small".
    out["p_resolution_floor"] = {"gradient": 1.0 / (n_null + 1),
                                 "edge_scan": 1.0 / (n_scan + 1)}
    out["trials_correction_note"] = (
        "Sidak over correlated tests; conservative. Compare best_p against the "
        "Monte Carlo floor before reading it as a bound.")
    out["detection"] = bool(out["best_p_trials_corrected"] is not None
                            and out["best_p_trials_corrected"] < 0.05)
    return out


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
        (cdir / "summary.json").write_text(json.dumps(res, indent=2, default=_jsonable))
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
    summary = {
        "channel": CHANNEL,
        "verdict": ("DETECTION" if detections else
                    ("CLEAN_NULL" if tested else "NO_TESTABLE_CATALOGUE")),
        "n_catalogues": len(cats), "n_tested": len(tested),
        "catalogue_verdicts": verdicts,
        "detections": detections,
        "best_p_by_channel": {r["channel"]: r.get("best_p_trials_corrected")
                              for r in tested},
        "channels": per_channel,
        "predictions_discriminated": _PREDICTIONS,
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    print("[tidemark]", json.dumps({"verdict": summary["verdict"],
                                    "n_tested": len(tested),
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
